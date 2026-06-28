# DSPy Module Composition Guide

DSPy programs are built by composing `Module` subclasses. This guide covers
how to define typed `Signature`s, build multi-hop pipelines, and prepare
programs for optimisation with `Teleprompter`s.

## Signatures: The Contract

A `Signature` declares the input/output contract for a single LM call:

```python
import dspy


class AnswerWithCitations(dspy.Signature):
    """Answer a question grounded in retrieved passages, citing sources."""

    question: str = dspy.InputField(desc="The user question to answer.")
    passages: list[str] = dspy.InputField(
        desc="Retrieved text passages relevant to the question."
    )
    answer: str = dspy.OutputField(
        desc="A concise, factual answer in 1–2 sentences."
    )
    citations: list[str] = dspy.OutputField(
        desc="List of passage indices used (e.g. ['0', '2'])."
    )
```

Tips for good signatures:
- Keep the docstring short — it becomes part of the instruction sent to the LM.
- Use descriptive `desc` strings; they guide the LM without hardcoded prompts.
- Prefer `list[str]` over `str` for structured multi-value outputs.

---

## Modules: Composable Programs

```python
class RAGPipeline(dspy.Module):
    """Two-stage RAG: retrieve relevant passages then generate a cited answer."""

    def __init__(self, num_passages: int = 5) -> None:
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.answer = dspy.ChainOfThought(AnswerWithCitations)

    def forward(self, question: str) -> dspy.Prediction:
        """Run retrieval then generation.

        Args:
            question: The user's natural-language question.

        Returns:
            A Prediction containing `answer` and `citations` fields.
        """
        passages = self.retrieve(question).passages
        return self.answer(question=question, passages=passages)
```

---

## Multi-Hop Pipelines

Chain modules to decompose complex reasoning:

```python
class MultiHopQA(dspy.Module):
    """Multi-hop QA: generate sub-queries, retrieve for each, then synthesise."""

    def __init__(self, hops: int = 2) -> None:
        super().__init__()
        self.generate_subquery = dspy.ChainOfThought(
            "question -> sub_question"
        )
        self.retrieve = dspy.Retrieve(k=3)
        self.synthesise = dspy.ChainOfThought(
            "question, contexts -> answer"
        )
        self.hops = hops

    def forward(self, question: str) -> dspy.Prediction:
        contexts: list[str] = []
        for _ in range(self.hops):
            sub_question = self.generate_subquery(
                question=question
            ).sub_question
            passages = self.retrieve(sub_question).passages
            contexts.extend(passages)
        return self.synthesise(question=question, contexts=contexts)
```

---

## Predictor Primitives

| Primitive | When to Use |
|-----------|------------|
| `dspy.Predict` | Structured extraction; no reasoning trace needed |
| `dspy.ChainOfThought` | Tasks that benefit from step-by-step reasoning |
| `dspy.ProgramOfThought` | Math/code tasks; generates & executes code |
| `dspy.ReAct` | Tool-use; interleaves thinking and action calls |
| `dspy.Retrieve` | Fetch passages from a configured `dspy.Retrieve` RM |

---

## Assertion-Driven Refinement

Use `dspy.Assert` to express constraints the LM must satisfy:

```python
class ConstrainedSummary(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.summarise = dspy.ChainOfThought("document -> summary")

    def forward(self, document: str) -> dspy.Prediction:
        prediction = self.summarise(document=document)
        dspy.Assert(
            len(prediction.summary.split()) <= 100,
            "Summary must be 100 words or fewer.",
        )
        dspy.Suggest(
            prediction.summary.endswith("."),
            "Summary should end with a period.",
        )
        return prediction
```

- `dspy.Assert`: hard constraint; retries until satisfied or `max_retries` hit.
- `dspy.Suggest`: soft constraint; provides feedback but does not retry.

---

## Optimisation with MIPRO

```python
from dspy.teleprompt import MIPROv2

optimiser = MIPROv2(
    metric=your_metric_fn,
    num_candidates=10,
    init_temperature=1.0,
)

train_set = [dspy.Example(question=q, answer=a) for q, a in training_data]

optimised_program = optimiser.compile(
    student=RAGPipeline(),
    trainset=train_set,
    num_batches=5,
)

# Save for later use
optimised_program.save("optimised_rag.json")

# Load and use
loaded = RAGPipeline()
loaded.load("optimised_rag.json")
```

---

## Testing Modules

```python
import dspy

# Use a cheap, fast model for unit tests
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", max_tokens=256))

rag = RAGPipeline(num_passages=3)
prediction = rag(question="What year was Python created?")

assert "1991" in prediction.answer or "Guido" in prediction.answer
assert isinstance(prediction.citations, list)
```

---

## Common Pitfalls

- **Not calling `super().__init__()`**: DSPy cannot trace sub-modules for
  optimisation unless `__init__` is properly chained.
- **Mutable default arguments**: Use `None` defaults with `num_passages: int = 5`
  pattern, not `passages: list = []`.
- **Skipping assertions in optimisation**: `dspy.Assert` is ignored during
  compilation unless `assert_transform_module` is applied. Check the docs.
- **Large `k` without re-ranking**: Passing 20 passages to an LM often hurts
  more than it helps; start with 3–5 and increase only after evaluating recall.
