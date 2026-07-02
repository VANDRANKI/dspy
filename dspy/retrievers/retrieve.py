import random
from typing import Any

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages: list[dict[str, Any]]) -> Prediction:
    """Merge a list of per-passage dicts into a single `Prediction` of parallel lists.

    Args:
        passages: A list of dicts, one per retrieved passage, each sharing the same keys
            (e.g. `long_text`, `score`). Must contain at least one element.

    Returns:
        A `Prediction` whose fields are lists collected across all `passages`. The
        `long_text` key, if present, is renamed to `passages`.
    """
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k: int = 3, callbacks: list | None = None) -> None:
        """Initialize a retrieval parameter backed by the configured retrieval model.

        Args:
            k: Default number of passages to retrieve when not overridden per-call.
            callbacks: Optional list of callbacks to invoke around retrieval calls.
        """
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self) -> None:
        pass

    def dump_state(self) -> dict[str, Any]:
        """Serialize the retriever's configuration for saving.

        Returns:
            A dict containing the `k` setting.
        """
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore the retriever's configuration from a previously dumped state.

        Args:
            state: A dict of attribute name to value, as produced by `dump_state`.
        """
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs) -> list[str] | Prediction | list[Prediction]:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs,
    ) -> list[str] | Prediction | list[Prediction]:
        k = k if k is not None else self.k

        import dspy

        if not dspy.settings.rm:
            raise AssertionError("No RM is loaded.")

        passages = dspy.settings.rm(query, k=k, **kwargs)

        from collections.abc import Iterable
        if not isinstance(passages, Iterable):
            # it's not an iterable yet; make it one.
            # TODO: we should unify the type signatures of dspy.Retriever
            passages = [passages]
        passages = [psg.long_text for psg in passages]

        return Prediction(passages=passages)

# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.
