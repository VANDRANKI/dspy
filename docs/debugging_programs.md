# Debugging DSPy Programs

Practical guide for diagnosing failures in DSPy programs and optimizers.

---

## Inspecting predictions

### Print the full prompt sent to the LM

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Enable prompt inspection
with dspy.context(lm=lm):
    pred = my_program(question="What is 2 + 2?")

# Inspect the last call
print(dspy.inspect_history(n=1))
```

### Access intermediate field values

```python
pred = my_program(question="What is the capital of France?")
print(pred.answer)          # final answer
print(pred.reasoning)       # chain-of-thought trace (if using CoT)
```

---

## Optimizer selection guide

| Scenario | Recommended optimizer |
|----------|-----------------------|
| < 50 training examples | `BootstrapFewShot` |
| 50–200 examples, need best few-shot demos | `BootstrapFewShotWithRandomSearch` |
| Large labelled dataset, want instruction tuning | `MIPROv2` |
| Very large dataset, fine-tuning budget | `BootstrapFinetune` |

### `MIPROv2` tips

- Set `num_candidates` to at least 10 for meaningful exploration.
- Pass `requires_permission_to_run=False` in automated pipelines to skip the
  interactive confirmation prompt.
- Use `minibatch=True` with large training sets to reduce API cost during
  optimisation.

---

## Common errors

### `dspy.AssertionError: assertion failed`

A `dspy.Assert` statement in your program evaluated to `False`. Inspect the
asserted condition and the corresponding prediction fields to understand why
the constraint was violated.

### `Predict` returns an empty string for a required field

This usually means the LM output did not match the expected field format.
Check:

1. Whether the `Signature` field names are unambiguous.
2. Whether you need to add `dspy.ChainOfThought` instead of `dspy.Predict` to
   give the LM room to reason before outputting the field.

### Rate limit errors during optimisation

Reduce `max_bootstrapped_demos` or add `asyncify=True` to spread requests:

```python
optimizer = dspy.BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,   # reduce from default 8
    max_labeled_demos=4,
)
```

---

## Saving and loading optimised programs

```python
# Save
optimised_program.save("my_program.json")

# Load
loaded = MyDSPyProgram()
loaded.load("my_program.json")
```

Always test a loaded program against a held-out evaluation set before
deploying to production.
