# Metrics and Evaluation in DSPy

This guide explains how to write effective metric functions and evaluate your DSPy programs.

## What is a Metric?

In DSPy, a metric is a callable that accepts a `(gold, prediction, trace=None)` signature and returns a `float` (or `bool`). Optimizers like `BootstrapFewShot` and `MIPROv2` use the metric to guide compilation.

```python
from typing import Any


def exact_match(gold: Any, prediction: Any, trace: Any = None) -> bool:
    """Return True if prediction matches gold exactly (case-insensitive).

    Args:
        gold: The gold example from your dataset.
        prediction: The prediction produced by your DSPy program.
        trace: Optional trace object passed by DSPy optimizers.

    Returns:
        True if the normalized answer strings match.
    """
    gold_answer = gold.answer.strip().lower()
    pred_answer = prediction.answer.strip().lower()
    return gold_answer == pred_answer
```

## Returning Floats for Partial Credit

Binary metrics are easy to satisfy but give optimizers little signal. Float metrics enable gradient-like feedback:

```python
from difflib import SequenceMatcher


def fuzzy_match_score(
    gold: Any,
    prediction: Any,
    trace: Any = None,
    threshold: float = 0.8,
) -> float:
    """Compute fuzzy string similarity between gold and predicted answers.

    Args:
        gold: Gold example with `.answer` attribute.
        prediction: Prediction with `.answer` attribute.
        trace: Optional optimizer trace.
        threshold: Minimum similarity to count as correct (for logging).

    Returns:
        Similarity ratio in [0.0, 1.0].
    """
    ratio = SequenceMatcher(
        None,
        gold.answer.lower(),
        prediction.answer.lower(),
    ).ratio()
    if trace and ratio < threshold:
        trace.setdefault("low_similarity_examples", []).append(
            {"gold": gold.answer, "pred": prediction.answer, "score": ratio}
        )
    return ratio
```

## Composite Metrics

Combine multiple signals with weighted averaging:

```python

def composite_qa_metric(
    gold: Any,
    prediction: Any,
    trace: Any = None,
) -> float:
    """Score QA predictions on both answer correctness and citation presence.

    Args:
        gold: Gold example with `.answer` and `.context` attributes.
        prediction: Prediction with `.answer` and `.citations` attributes.
        trace: Optional optimizer trace.

    Returns:
        Weighted score in [0.0, 1.0].
    """
    answer_score = float(gold.answer.lower() in prediction.answer.lower())
    citation_score = float(
        len(getattr(prediction, "citations", [])) > 0
    )
    return 0.7 * answer_score + 0.3 * citation_score
```

## Evaluating with `dspy.Evaluate`

```python
import dspy
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=dev_examples,
    metric=composite_qa_metric,
    num_threads=8,
    display_progress=True,
    display_table=5,  # show first 5 results
)

score = evaluator(my_program)
print(f"Dev score: {score:.2%}")
```

## Best Practices

1. **Separate train/dev/test sets** — optimize on train, select checkpoints on dev, report only on test.
2. **Avoid metric leakage** — don't look at test labels when writing the metric.
3. **Use `trace` for debugging** — store mismatches in `trace` to inspect after evaluation.
4. **Profile slow metrics** — a metric called millions of times in optimization must be fast.
5. **Log score distributions** — check for bimodal distributions that indicate metric issues.
