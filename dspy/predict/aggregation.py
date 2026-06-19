"""Aggregation utilities for combining multiple DSPy predictions.

Currently provides :func:`majority`, which selects the most common value
for a target output field across a collection of completions.  This is
useful after sampling the same signature multiple times (e.g. with
``n > 1``) and wanting a single consensus prediction.
"""

from typing import Callable, Optional, Union

from dspy.evaluate import normalize_text
from dspy.primitives.prediction import Completions, Prediction


def default_normalize(s: str) -> Optional[str]:
    """Normalize a string value for majority-vote comparison.

    Applies :func:`~dspy.evaluate.normalize_text` (lowercasing, punctuation
    removal, whitespace collapsing) and returns ``None`` for empty strings so
    they are excluded from the vote.

    Args:
        s (str): Raw field value to normalize.

    Returns:
        str | None: Normalized string, or ``None`` if the result is empty.
    """
    return normalize_text(s) or None


def majority(
    prediction_or_completions: Union[Prediction, Completions, list],
    normalize: Optional[Callable[[str], Optional[str]]] = default_normalize,
    field: Optional[str] = None,
) -> Prediction:
    """Return the majority-vote prediction for a target output field.

    Scans all completions in *prediction_or_completions*, optionally
    normalizes each value with *normalize*, and returns the first completion
    whose field value matches the most-frequent normalized value.

    Tie-breaking policy: when two values appear equally often, the one that
    appears earlier in the completion list wins.

    Values for which *normalize* returns ``None`` are excluded from the vote.
    If *all* values normalize to ``None``, the raw (un-normalized) values are
    used as a fallback so that a result is always returned.

    Args:
        prediction_or_completions (Prediction | Completions | list): A
            :class:`~dspy.primitives.prediction.Prediction` with multiple
            completions, a bare :class:`~dspy.primitives.prediction.Completions`
            object, or a plain list of dicts mapping field names to values.
        normalize (Callable[[str], str | None] | None): Function applied to
            each field value before counting votes.  Pass ``None`` to skip
            normalization.  Defaults to :func:`default_normalize`.
        field (str | None): Name of the output field to vote on.  When
            ``None``, the last output field in the signature is used (or the
            last key in the first completion dict if no signature is
            available).

    Returns:
        Prediction: A single-completion :class:`~dspy.primitives.prediction.Prediction`
        wrapping the winning completion.

    Raises:
        AssertionError: If *prediction_or_completions* is not one of the
            supported types.

    Example::

        import dspy

        completions = [
            {"answer": "Paris"},
            {"answer": "paris"},
            {"answer": "London"},
        ]
        winner = majority(completions, field="answer")
        # winner.answer == "Paris"  (first occurrence of the majority value)
    """
    assert any(isinstance(prediction_or_completions, t) for t in [Prediction, Completions, list])

    # Get the completions
    if isinstance(prediction_or_completions, Prediction):
        completions = prediction_or_completions.completions
    else:
        completions = prediction_or_completions

    try:
        signature = completions.signature
    except Exception:
        signature = None

    if not field:
        if signature:
            field = list(signature.output_fields.keys())[-1]
        else:
            field = list(completions[0].keys())[-1]

    # Normalize
    normalize = normalize if normalize else lambda x: x
    normalized_values = [normalize(completion[field]) for completion in completions]
    normalized_values_ = [x for x in normalized_values if x is not None]

    # Count votes; fall back to raw values when all normalize to None
    value_counts: dict = {}
    for value in normalized_values_ or normalized_values:
        value_counts[value] = value_counts.get(value, 0) + 1

    majority_value = max(value_counts, key=value_counts.get)

    # Return the first completion whose normalized field value matches
    for completion in completions:
        if normalize(completion[field]) == majority_value:
            break

    return Prediction.from_completions([completion], signature=signature)
