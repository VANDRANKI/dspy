# Evaluation and Optimization in DSPy

This guide explains how to measure program quality and use DSPy optimisers
to improve it automatically.

## 1. Defining a metric

A metric is a callable that scores a (example, prediction, trace) triple.
Return a float in [0, 1] or a bool.

```python
import dspy

def exact_match(example, prediction, trace=None):
    """Score 1.0 if predicted answer matches gold, else 0.0."""
    return float(example.answer.strip().lower() == prediction.answer.strip().lower())

def f1_score(example, prediction, trace=None):
    gold_tokens = set(example.answer.lower().split())
    pred_tokens = set(prediction.answer.lower().split())
    if not gold_tokens or not pred_tokens:
        return 0.0
    precision = len(gold_tokens & pred_tokens) / len(pred_tokens)
    recall = len(gold_tokens & pred_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

## 2. Evaluating a program

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(
    devset=dev_examples,
    metric=exact_match,
    num_threads=4,
    display_progress=True,
)

score = evaluate(my_program)
print(f"Exact match: {score:.1%}")
```

## 3. Optimising with BootstrapFewShot

`BootstrapFewShot` automatically selects and adds few-shot demonstrations
that improve the metric score.

```python
from dspy.teleprompt import BootstrapFewShot

optimiser = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)
optimised_program = optimiser.compile(my_program, trainset=train_examples)
```

## 4. Optimising with MIPRO

MIPRO jointly optimises instructions and few-shot examples.

```python
from dspy.teleprompt import MIPROv2

optimiser = MIPROv2(
    metric=exact_match,
    auto="medium",  # light | medium | heavy
)
optimised_program = optimiser.compile(
    my_program,
    trainset=train_examples,
    num_trials=20,
)
```

## 5. Saving and loading optimised programs

```python
# Save
optimised_program.save("optimised_qa.json")

# Load
loaded = MyQAProgram()
loaded.load("optimised_qa.json")
```

## 6. Assertions as a training signal

Use `dspy.Assert` to hard-fail on constraint violations and
`dspy.Suggest` for soft hints that also guide optimisation.

```python
class ConstrainedQA(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        pred = self.predict(question=question)
        dspy.Suggest(
            len(pred.answer.split()) <= 50,
            "Answer should be concise (at most 50 words).",
        )
        return pred
```

## Optimiser selection guide

| Optimiser | Best when | Cost |
|---|---|---|
| `LabeledFewShot` | You have high-quality hand-written demos | Low |
| `BootstrapFewShot` | You have labelled training data | Medium |
| `MIPROv2` | You need the best accuracy possible | High |
| `BootstrapFewShotWithRandomSearch` | Quick exploration before MIPROv2 | Medium |
