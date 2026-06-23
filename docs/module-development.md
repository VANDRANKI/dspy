# DSPy Module Development Guide

This guide explains how to build and contribute custom DSPy modules.

## What is a DSPy Module?

A DSPy `Module` is a reusable, composable unit of LLM computation. Modules can be:
- **Signatures**: Define the input/output schema for an LLM call
- **Predictors**: Wrap a signature with a prompting strategy (Chain of Thought, ReAct, etc.)
- **Programs**: Compose multiple modules into a pipeline

## Creating a Custom Module

```python
import dspy
from typing import Optional

class SentimentClassifier(dspy.Module):
    """Classify sentiment of text with confidence scoring."""

    def __init__(self, labels: Optional[list[str]] = None) -> None:
        """Initialize the sentiment classifier.

        Args:
            labels: Custom sentiment labels. Defaults to ['positive', 'negative', 'neutral'].
        """
        super().__init__()
        self.labels = labels or ["positive", "negative", "neutral"]
        self.classify = dspy.ChainOfThought(
            "text -> sentiment: str, confidence: float"
        )

    def forward(self, text: str) -> dspy.Prediction:
        """Run classification on the given text.

        Args:
            text: Input text to classify.

        Returns:
            Prediction with `sentiment` and `confidence` fields.
        """
        result = self.classify(text=text)
        if result.sentiment not in self.labels:
            result.sentiment = "neutral"  # fallback for out-of-schema responses
        return result
```

## Running the Test Suite

```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
pytest tests/ -v

# Run a specific module's tests
pytest tests/predict/ -v

# With coverage
pytest tests/ --cov=dspy --cov-report=term-missing
```

## Optimization Best Practices

1. **Start small** — optimize with 20-50 examples before scaling
2. **Use labeled examples** for teleprompters that require training signal
3. **Metric function** must return a numeric score in `[0, 1]`
4. **Save compiled programs** with `program.save("optimized.json")` for reuse

## Code Quality

```bash
# Run pre-commit checks
pre-commit run --all-files

# Type checking
pyright dspy/
```

## Submitting a PR

- Reference any open issue in the PR description
- Add tests for new functionality under `tests/`
- Ensure all pre-commit checks pass
- Keep changes focused — avoid bundling unrelated improvements
