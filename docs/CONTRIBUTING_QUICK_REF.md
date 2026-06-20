# DSPy Contributor Quick Reference

## Setup

```bash
git clone https://github.com/stanfordnlp/dspy.git
cd dspy
pip install -e ".[dev]"
```

## Running tests

```bash
# Fast unit tests (no LM calls)
pytest tests/ -m "not integration" -v

# Integration tests (requires API key)
OPENAI_API_KEY=... pytest tests/ -m integration -v
```

## Writing a good Signature

```python
class SummariseDocument(dspy.Signature):
    """Summarise a document in one paragraph."""

    document: str = dspy.InputField(desc="The full document text to summarise.")
    summary: str = dspy.OutputField(
        desc="A concise one-paragraph summary of the document."
    )
```

Guidelines:
- The docstring is shown to the LM as the task instruction — make it clear.
- `desc` on each field adds context that helps the LM format its output.
- Output field names should be unambiguous; avoid generic names like `output`.

## Optimizer selection

| Dataset size | Recommended |
|-------------|-------------|
| < 50 examples | `BootstrapFewShot` |
| 50–200 | `BootstrapFewShotWithRandomSearch` |
| 200+ | `MIPROv2` |

## Before submitting a PR

```bash
pre-commit run --all-files
pytest tests/ -m "not integration"
```
