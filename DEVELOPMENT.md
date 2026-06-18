# DSPy Development Guide

## Setup

```bash
git clone https://github.com/VANDRANKI/dspy.git
cd dspy
pip install -e ".[dev]"
# or with uv:
uv sync
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_primitives.py -v

# Filter by name
pytest tests/ -k "test_chain_of_thought" -v
```

## Code Quality

```bash
uv run ruff format .        # format
uv run ruff check .         # lint
uv run mypy dspy/           # type check
```

## Type Annotations

All public APIs must have complete type hints:

```python
from typing import Any, Optional, Union

class Predict(Module):
    def forward(self, **kwargs: Any) -> Prediction:
        """Run the predictor on the given inputs."""
        ...
```

## Adding an Optimizer

1. Subclass `Teleprompter` in `dspy/teleprompt/`
2. Implement `compile(student, *, trainset, ...)` with full type hints
3. Add a test in `tests/teleprompt/`
4. Document it in `docs/`

## Commit Convention

```
feat: add BootstrapFewShotWithRandomSearch optimizer
fix: prevent duplicate examples in few-shot pool
docs: clarify field description role in Signature
test: add coverage for multi-hop ChainOfThought
```
