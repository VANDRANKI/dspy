<p align="center">
  <img align="center" src="docs/docs/static/img/dspy_logo.png" width="460px" />
</p>
<p align="left">


## DSPy: _Programming_—not prompting—Foundation Models

**Documentation:** [DSPy Docs](https://dspy.ai/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)


----

DSPy is the framework for _programming—rather than prompting—language models_. It allows you to iterate fast on **building modular AI systems** and offers algorithms for **optimizing their prompts and weights**, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional _Python code_ and use DSPy to **teach your LM to deliver high-quality outputs**. Learn more via our [official documentation site](https://dspy.ai/) or meet the community, seek help, or start contributing via this GitHub repo and our [Discord server](https://discord.gg/XCGy2WDCQB).


## Documentation: [dspy.ai](https://dspy.ai)

**Please go to the [DSPy Docs at dspy.ai](https://dspy.ai)**


## Installation

```bash
pip install dspy
```

To install the very latest from `main`:

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
```


## Quick Start

### Basic Usage

```python
import dspy

# Configure your language model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define a simple signature
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

# Use a predictor
predict = dspy.Predict(BasicQA)
result = predict(question="What is the capital of France?")
print(result.answer)  # Paris
```

### Chain of Thought

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class MathSolver(dspy.Signature):
    """Solve math problems step by step."""
    problem: str = dspy.InputField()
    solution: str = dspy.OutputField()

# ChainOfThought automatically adds intermediate reasoning
cot = dspy.ChainOfThought(MathSolver)
result = cot(problem="If a train travels 120 miles in 2 hours, what is its speed?")
print(result.solution)
```

### Optimizer Usage (MIPROv2)

```python
import dspy
from dspy.teleprompt import MIPROv2

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define your module
class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Prepare training data
trainset = [
    dspy.Example(question="What is photosynthesis?", answer="...").with_inputs("question"),
    # ... more examples
]

# Define a metric
def exact_match(example, prediction, trace=None):
    return example.answer.lower() == prediction.answer.lower()

# Compile with MIPROv2
optimizer = MIPROv2(metric=exact_match, auto="medium")
compiled_rag = optimizer.compile(RAGPipeline(), trainset=trainset)
```

### Optimizer Usage (BootstrapFewShot)

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class Classifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> sentiment")

    def forward(self, text):
        return self.classify(text=text)

trainset = [
    dspy.Example(text="I love this!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="This is terrible.", sentiment="negative").with_inputs("text"),
    # ... more examples
]

def sentiment_metric(example, prediction, trace=None):
    return example.sentiment == prediction.sentiment

# Bootstrap few-shot examples automatically
optimizer = BootstrapFewShot(metric=sentiment_metric, max_bootstrapped_demos=4)
compiled = optimizer.compile(Classifier(), trainset=trainset)
```

### Tips and Best Practices

- **Start with `dspy.Predict`**: Use it for straightforward tasks before reaching for more complex modules.
- **Use `dspy.ChainOfThought`** when the task benefits from step-by-step reasoning (math, logic, multi-hop QA).
- **Write clear signatures**: The docstring and field descriptions are part of the prompt — be precise.
- **Collect labeled examples** before running optimizers; even 20–50 examples can significantly improve compiled programs.
- **Choose the right optimizer**:
  - `BootstrapFewShot` — fast, low data requirements, good for few-shot demos.
  - `MIPROv2` — state-of-the-art for instruction + demo optimization; use `auto="light"` to start.
  - `GRPO` / `SIMBA` — for weight optimization when you have sufficient data.
- **Cache LM calls** during development by setting `dspy.configure(lm=lm, cache=True)` to avoid redundant API costs.
- **Inspect traces** with `dspy.inspect_history(n=5)` to debug prompts and intermediate reasoning steps.
- **Save and load** compiled programs with `program.save("my_program")` and `dspy.load("my_program")`.


## 📜 Citation & Reading More

If you're looking to understand the framework, please go to the [DSPy Docs at dspy.ai](https://dspy.ai).

If you're looking to understand the underlying research, this is a set of our papers:

**[Jul'25] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)**       
**[Jun'24] [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)**       
**[Oct'23] [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)**     
[Jul'24] [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930)     
[Jun'24] [Prompts as Auto-Optimized Training Hyperparameters](https://arxiv.org/abs/2406.11706)    
[Feb'24] [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)         
[Jan'24] [In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178)       
[Dec'23] [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382)   
[Dec'22] [Demonstrate-Search-Predict: Composing Retrieval & Language Models for Knowledge-Intensive NLP](https://arxiv.org/abs/2212.14024.pdf)

To stay up to date or learn more, follow [@DSPyOSS](https://twitter.com/DSPyOSS) on Twitter or the DSPy page on LinkedIn.

The **DSPy** logo is designed by **Chuyi Zhang**.

If you use DSPy or DSP in a research paper, please cite our work as follows:

```
@inproceedings{khattab2024dspy,
  title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
  author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
@article{khattab2022demonstrate,
  title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
  author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
  journal={arXiv preprint arXiv:2212.14024},
  year={2022}
}
```
