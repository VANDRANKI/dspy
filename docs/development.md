# DSPy Development Guide

This guide helps contributors set up a local environment and understand
the DSPy codebase structure.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended)

## Setup

```bash
git clone https://github.com/stanfordnlp/dspy
cd dspy

# Install all dev dependencies
uv sync --all-groups
# or: pip install -e ".[dev]"
```

## Running Tests

```bash
# All unit tests
pytest tests/ -v

# A single file
pytest tests/test_primitives.py -v

# Skip tests that require API keys
pytest tests/ -v -m "not llm_call"
```

## Architecture Overview

```
dspy/
├── primitives/    # Signature, Module, Example, Prediction
├── predict/       # Predict, ChainOfThought, ReAct, ...
├── retrieve/      # Retrieval modules
├── teleprompt/    # Optimizers (BootstrapFewShot, MIPRO, ...)
├── evaluate/      # Evaluation utilities and metrics
├── utils/         # Shared helpers
└── clients/       # LM and RM client adapters
```

### Key Concepts

- **Signature**: Declarative input/output spec for a module (like a type signature).
- **Module**: A composable unit of computation, similar to `torch.nn.Module`.
- **Optimizer** (Teleprompter): Compiles a module by selecting few-shot examples,
  tuning instructions, or fine-tuning weights.
- **Example**: A single training or validation data point.

## Adding a New Optimizer

1. Create `dspy/teleprompt/<optimizer_name>.py`.
2. Subclass `Teleprompter` from `dspy.teleprompt.teleprompt`.
3. Implement `compile(self, student, *, trainset, **kwargs) -> Module`.
4. Register in `dspy/teleprompt/__init__.py` and `dspy/__init__.py`.
5. Add tests in `tests/teleprompt/test_<optimizer_name>.py`.

## Code Style

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy dspy/
```

- Use type hints on all public functions.
- Google-style docstrings.
- Avoid `import *` — explicit imports only.
- Tests should be deterministic; mock LM calls in unit tests.

## Pre-Commit Checklist

- [ ] `uv run ruff format .` applied
- [ ] `uv run ruff check .` passes
- [ ] `pytest tests/` passes (with LLM tests mocked)
- [ ] New optimizer or module has tests and a docstring
- [ ] `dspy/__init__.py` updated to export new public symbols
