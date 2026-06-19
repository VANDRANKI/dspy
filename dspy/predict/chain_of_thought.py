"""Chain-of-thought reasoning module for DSPy.

This module provides :class:`ChainOfThought`, a thin wrapper around
:class:`~dspy.predict.predict.Predict` that prepends a ``reasoning``
output field to any signature.  The model is forced to produce its
step-by-step rationale before generating the final outputs, which
improves accuracy on complex reasoning tasks.
"""

from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    """A reasoning module that forces the model to think step by step.

    :class:`ChainOfThought` modifies the provided *signature* by inserting
    a ``reasoning`` output field *before* all other output fields.  This
    causes the underlying :class:`~dspy.predict.predict.Predict` module to
    generate an intermediate chain-of-thought trace before committing to the
    final answer fields, which typically improves accuracy on multi-step or
    knowledge-intensive tasks.

    The module delegates to a single :attr:`predict` sub-module so that
    optimizers (teleprompters) can tune its demonstrations and instructions
    transparently.

    Args:
        signature (str | type[Signature]): The DSPy signature defining the
            input and output fields.  A plain string such as
            ``"question -> answer"`` is accepted and converted automatically.
        rationale_field (FieldInfo | None): A custom Pydantic
            :class:`~pydantic.fields.FieldInfo` to use as the reasoning
            field.  When ``None`` (the default) a new
            :func:`~dspy.OutputField` with the description
            ``"${reasoning}"`` is created.  Pass a custom field to change
            the description or constraints on the rationale.
        rationale_field_type (type): Python type to annotate the reasoning
            field with.  Ignored when *rationale_field* carries its own
            ``annotation``.  Defaults to ``str``.
        **config: Additional keyword arguments forwarded verbatim to the
            underlying :class:`~dspy.predict.predict.Predict` module
            (e.g. ``temperature``, ``n``).

    Attributes:
        predict (dspy.Predict): The wrapped Predict module operating on the
            extended signature (with the prepended reasoning field).

    Example::

        import dspy

        # Simple string signature
        cot = dspy.ChainOfThought("question -> answer")
        result = cot(question="What is 7 * 8?")
        print(result.reasoning)  # intermediate chain-of-thought
        print(result.answer)     # "56"

        # Custom rationale field with a descriptive prompt
        from pydantic import Field
        rationale = Field(description="Think through the problem step by step.")
        cot = dspy.ChainOfThought("question -> answer", rationale_field=rationale)
    """

    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ) -> None:
        super().__init__()
        signature = ensure_signature(signature)
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Run the chain-of-thought prediction synchronously.

        Args:
            **kwargs: Input field values matching the signature's input fields.

        Returns:
            dspy.Prediction: A prediction object containing the ``reasoning``
            field plus all output fields defined in the original signature.
        """
        return self.predict(**kwargs)

    async def aforward(self, **kwargs: Any) -> dspy.Prediction:
        """Run the chain-of-thought prediction asynchronously.

        Args:
            **kwargs: Input field values matching the signature's input fields.

        Returns:
            dspy.Prediction: A prediction object containing the ``reasoning``
            field plus all output fields defined in the original signature.
        """
        return await self.predict.acall(**kwargs)
