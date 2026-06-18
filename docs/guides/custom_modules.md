# Writing Custom DSPy Modules

This guide explains how to create a custom `Module` that works
seamlessly with DSPy optimizers.

## Basic Structure

```python
import dspy
from typing import Any

class MyModule(dspy.Module):
    """A custom module that combines retrieval with chain-of-thought."""

    def __init__(self, num_passages: int = 3) -> None:
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        """Run the module on a single question.

        Args:
            question: The question to answer.

        Returns:
            A Prediction with at minimum an `answer` field.
        """
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

## Optimizer Compatibility

For a module to be optimizable with `BootstrapFewShot` or similar:

1. All sub-modules must be assigned as instance attributes in `__init__`.
2. `forward()` must accept keyword arguments and return a `Prediction`.
3. Do **not** mutate `self` inside `forward()` — it breaks multi-threading.

## Saving and Loading

```python
# Save
module = MyModule()
optimized = teleprompter.compile(module, trainset=trainset)
optimized.save("my_module.json")

# Load
loaded = MyModule()
loaded.load("my_module.json")
```

## Testing Your Module

```python
import pytest
import dspy
from unittest.mock import patch

def test_my_module_forward():
    module = MyModule(num_passages=2)
    with patch.object(module.retrieve, 'forward', return_value=dspy.Prediction(
        passages=["Paris is the capital of France.", "France is in Europe."]
    )):
        pred = module(question="What is the capital of France?")
        assert hasattr(pred, 'answer')
```
