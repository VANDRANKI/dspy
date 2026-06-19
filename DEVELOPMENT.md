# DSPy Development Guide

This document covers everything you need to get started contributing to DSPy.

---

## Prerequisites

- Python 3.9 or later
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

---

## Installing DSPy in Development Mode

### Using uv (recommended)

```bash
git clone https://github.com/stanfordnlp/dspy.git
cd dspy
uv pip install -e ".[dev]"
```

### Using pip

```bash
git clone https://github.com/stanfordnlp/dspy.git
cd dspy
pip install -e ".[dev]"
```

The `[dev]` extra installs testing and linting dependencies (pytest, ruff, mypy, pre-commit, etc.).

---

## Running Tests

```bash
# Run the full test suite
pytest tests/

# Run a specific test file
pytest tests/predict/test_chain_of_thought.py -v

# Run tests matching a keyword
pytest tests/ -k "test_signature" -v

# Stop on first failure and show short tracebacks
pytest tests/ -x --tb=short

# Run with the helper script (passes extra args through)
./scripts/run_tests.sh -k "test_signature"
```

See `scripts/run_tests.sh` for the convenience wrapper.

---

## Running the Linter and Formatter

DSPy uses [pre-commit](https://pre-commit.com/) to run Ruff (lint + format) and other checks.

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run all hooks against all files manually
pre-commit run --all-files

# Run only Ruff lint
ruff check dspy/ tests/

# Auto-fix Ruff issues
ruff check --fix dspy/ tests/

# Run only Ruff formatter
ruff format dspy/ tests/
```

---

## Project Structure

```
dspy/
├── dspy/                   # Core library
│   ├── signatures/         # Signature definitions (field declarations, type annotations)
│   ├── predict/            # Prediction modules (Predict, ChainOfThought, ReAct, etc.)
│   ├── retrieve/           # Retrieval modules (ColBERT, BM25 wrappers, etc.)
│   ├── teleprompt/         # Optimizers / Teleprompters (BootstrapFewShot, MIPRO, etc.)
│   ├── evaluate/           # Evaluation utilities and metrics
│   ├── utils/              # Shared utilities (logging, caching, threading)
│   └── primitives/         # Core primitives (Example, Module, Program)
├── tests/                  # Test suite mirroring dspy/ structure
│   ├── predict/
│   ├── signatures/
│   ├── teleprompt/
│   └── ...
├── examples/               # End-to-end example programs
├── docs/                   # Documentation source (MkDocs)
├── scripts/                # Developer helper scripts
└── pyproject.toml          # Project metadata and dependencies
```

---

## Key Concepts

### Signatures

A `Signature` declares the input and output fields of a DSPy module, similar to a typed function signature. Field descriptions guide the language model on what each field should contain.

```python
import dspy

class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

### Modules

Modules are composable building blocks. Each module wraps one or more LM calls and can be optimized. The most commonly used modules are:

| Module | Description |
|--------|-------------|
| `dspy.Predict` | Single LM call using a Signature |
| `dspy.ChainOfThought` | Adds a `reasoning` field before the answer |
| `dspy.ReAct` | Tool-using agent loop |
| `dspy.ProgramOfThought` | Generates and executes code |
| `dspy.MultiChainComparison` | Samples multiple reasoning chains and picks the best |

Custom modules subclass `dspy.Module` and implement `forward()`:

```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

### Optimizers (Teleprompters)

Optimizers tune module prompts or weights using a training set and a metric. They live in `dspy.teleprompt`:

| Optimizer | Description |
|-----------|-------------|
| `BootstrapFewShot` | Generates few-shot demonstrations via bootstrapping |
| `BootstrapFewShotWithRandomSearch` | Adds random search over candidate demo sets |
| `MIPRO` | Multi-stage prompt optimization with Bayesian search |
| `COPRO` | Coordinate-ascent prompt optimization |
| `BootstrapFinetune` | Fine-tunes LM weights using bootstrapped traces |

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
compiled_rag = optimizer.compile(RAG(), trainset=trainset)
```

---

## Writing Tests

- Place tests in `tests/` mirroring the source path (e.g., `dspy/predict/cot.py` → `tests/predict/test_cot.py`).
- Use `pytest` fixtures for LM mocking — avoid real API calls in unit tests.
- Mark slow or integration tests with `@pytest.mark.slow` so they can be skipped in fast CI runs.

---

## Opening a Pull Request

1. Fork the repository and create a feature branch.
2. Make your changes with clear, conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`).
3. Ensure `pre-commit run --all-files` passes.
4. Ensure `pytest tests/` passes (or at minimum the tests relevant to your change).
5. Open a PR against `main` with a clear description of what changed and why.
