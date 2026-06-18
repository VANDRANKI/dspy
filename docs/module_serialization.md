# DSPy Module Serialization Guide

This guide covers how to save and load compiled DSPy modules, what state
is persisted, common pitfalls, and how to validate loaded modules before
using them in production.

## What Gets Saved

When you call `module.save(path)`, DSPy serializes:

- **Named parameters** — `dspy.Predict` instances assigned as attributes
- **Compiled demos** — few-shot examples learned by the optimizer
- **Sub-module structure** — all nested modules reachable through attributes

What is **NOT saved**:

- The LM configuration (`dspy.settings.lm`) — set this in your app startup code
- Module docstrings and Python class definitions — the class must still exist at load time
- Any Python object stored outside a `Predict` parameter (e.g., a `list` attribute)

## Basic Save and Load

```python
import dspy

class RAGModule(dspy.Module):
    def __init__(self, num_passages: int = 3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retrieve(question).passages
        context = "\n".join(passages)
        return self.generate(context=context, question=question)

# Compile
module = RAGModule()
optimizer = dspy.BootstrapFewShot(metric=my_metric)
compiled = optimizer.compile(module, trainset=trainset)

# Save
compiled.save("compiled_rag.json")

# Load (must use the same class definition)
loaded = RAGModule()
loaded.load("compiled_rag.json")
```

## Validating Loaded Modules

After loading, verify the module has the expected demos and parameters:

```python
def validate_loaded_module(module: dspy.Module, min_demos: int = 3) -> None:
    """Verify a loaded module is properly compiled and ready for use.

    Args:
        module: The loaded DSPy module.
        min_demos: Minimum number of few-shot demos expected per Predict.

    Raises:
        ValueError: If the module appears uncompiled or has too few demos.
    """
    for name, param in module.named_predictors():
        demos = getattr(param, "demos", [])
        if len(demos) < min_demos:
            raise ValueError(
                f"Predictor '{name}' has only {len(demos)} demos "
                f"(expected >= {min_demos}). Module may not be compiled."
            )
    print(f"Module validated: {len(list(module.named_predictors()))} predictors OK.")


# Usage
loaded = RAGModule()
loaded.load("compiled_rag.json")
validate_loaded_module(loaded, min_demos=3)
```

## Signature Mismatch on Load

If you change a module's `Signature` fields after compiling, the loaded
demos may reference fields that no longer exist. DSPy will silently ignore
missing fields in old demos, which degrades performance.

Detect this by comparing field names:

```python
def check_signature_compatibility(
    module: dspy.Module,
    path: str,
) -> None:
    """Warn if saved demos reference fields not in the current signature.

    Args:
        module: Module with current class definition (not yet loaded).
        path: Path to the saved module file.
    """
    import json
    with open(path) as f:
        saved = json.load(f)

    for name, param in module.named_predictors():
        saved_param = saved.get(name, {})
        current_fields = set(param.signature.fields.keys())

        for demo in saved_param.get("demos", []):
            demo_fields = set(demo.keys())
            stale = demo_fields - current_fields - {"augmented"}
            if stale:
                print(
                    f"WARNING: Predictor '{name}' has demos with stale fields: "
                    f"{stale}. These will be ignored."
                )
```

## Sub-Modules in Lists (Known Limitation)

Sub-modules stored in Python `list` or `dict` attributes are **not tracked**
by the optimizer or serializer:

```python
class BrokenModule(dspy.Module):
    def __init__(self):
        # WRONG: list-stored modules are invisible to save/load
        self.predictors = [dspy.Predict("question -> answer") for _ in range(3)]

class CorrectModule(dspy.Module):
    def __init__(self):
        # CORRECT: attribute-assigned modules are tracked
        self.predict_0 = dspy.Predict("question -> answer")
        self.predict_1 = dspy.Predict("question -> answer")
        self.predict_2 = dspy.Predict("question -> answer")
```

## Integration Test Pattern

Always add a round-trip test to catch serialization regressions:

```python
import tempfile
import os
import dspy

def test_rag_module_save_load_roundtrip():
    module = RAGModule()
    # Manually add demos to simulate a compiled module
    module.generate.demos = [
        dspy.Example(context="AI is powerful", question="What is AI?", answer="Artificial Intelligence"),
        dspy.Example(context="Python is used in AI", question="What language?", answer="Python"),
        dspy.Example(context="DSPy automates prompting", question="What is DSPy?", answer="A framework"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        module.save(path)
        loaded = RAGModule()
        loaded.load(path)
        assert len(loaded.generate.demos) == 3
        assert loaded.generate.demos[0].answer == "Artificial Intelligence"
    finally:
        os.unlink(path)
