from typing import Any

from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature


class MultiChainComparison(Module):
    """A self-refinement module that compares M reasoning attempts and selects the best answer.

    ``MultiChainComparison`` appends *M* ``InputField`` slots to the given
    signature — one per student reasoning attempt — and prepends a
    ``rationale`` ``OutputField``. A single :class:`~dspy.predict.Predict`
    call then receives all attempts and produces a corrected rationale together
    with the final answer field from the original signature.

    This mirrors the *self-consistency + comparison* strategy: rather than
    majority-voting over raw answers, the LM is given explicit access to each
    attempt's reasoning chain and is instructed to reconcile them holistically.

    Args:
        signature: The base DSPy signature (class or string shorthand) that
            defines the task's input and output fields.
        M: Number of student reasoning attempts to compare. Defaults to ``3``.
        temperature: Sampling temperature forwarded to the underlying
            :class:`~dspy.predict.Predict` call. Defaults to ``0.7``.
        **config: Additional keyword arguments passed through to
            :class:`~dspy.predict.Predict`.

    Examples:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        cot = dspy.ChainOfThought("question -> answer")
        completions = [cot(question="What is 2+2?") for _ in range(3)]

        mcc = dspy.MultiChainComparison("question -> answer", M=3)
        result = mcc(completions=completions, question="What is 2+2?")
        print(result.answer)  # "4"
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        M: int = 3,  # noqa: N803
        temperature: float = 0.7,
        **config: Any,
    ) -> None:
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx+1}",
                InputField(
                    prefix=f"Student Attempt #{idx+1}:",
                    desc="${reasoning attempt}",
                ),
            )

        signature = signature.prepend(
            "rationale",
            OutputField(
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    def forward(self, completions: list[Prediction], **kwargs: Any) -> Prediction:
        """Run the comparison over *M* completions and return the best prediction.

        Each entry in ``completions`` must contain a ``rationale`` (or
        ``reasoning``) field and the signature's last output field. These are
        formatted as student attempt strings and injected into the extended
        signature before the underlying :class:`~dspy.predict.Predict` call.

        Args:
            completions: A list of exactly ``M`` :class:`~dspy.primitives.Prediction`
                objects produced by earlier forward passes of the wrapped module.
            **kwargs: Additional keyword arguments forwarded to the underlying
                :class:`~dspy.predict.Predict` call (e.g., the original input
                fields such as ``question``).

        Returns:
            A :class:`~dspy.primitives.Prediction` containing the corrected
            ``rationale`` field and the final answer field from the original
            signature.

        Raises:
            AssertionError: If ``len(completions) != self.M``.
        """
        attempts: list[str] = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert (
            len(attempts) == self.M
        ), f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)
