# Optimizer Selection Guide

DSPy provides several optimizers (also called teleprompters) that compile
programs by searching for better prompts or few-shot examples.

## Choosing an optimizer

| Optimizer | When to use |
|-----------|-------------|
| `BootstrapFewShot` | Fast iteration; no labelled data needed |
| `BootstrapFewShotWithRandomSearch` | Better quality; runs multiple trials |
| `MIPRO` | Best quality; requires a validation set |
| `SignatureOptimizer` | Optimize the instruction text only |

## Example

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
optimized = teleprompter.compile(my_program, trainset=trainset)
```

## Tips

- Start with `BootstrapFewShot` to validate your metric before running heavier optimizers
- Use a validation set of at least 50 examples for `MIPRO`
- Log optimizer traces to understand which demos were selected
