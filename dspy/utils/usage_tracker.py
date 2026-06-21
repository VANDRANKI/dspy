"""Usage tracking utilities for DSPy."""

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel

from dspy.dsp.utils.settings import settings


class UsageTracker:
    """Tracks LM usage data within a context.

    Typically instantiated via the :func:`track_usage` context manager rather
    than directly.  Usage entries are keyed by LM name and stored as a list of
    raw dicts so that per-call granularity is preserved.
    """

    def __init__(self) -> None:
        # Map of LM name to list of usage entries. For example:
        # {
        #     "openai/gpt-4o-mini": [
        #         {"prompt_tokens": 100, "completion_tokens": 200},
        #         {"prompt_tokens": 300, "completion_tokens": 400},
        #     ],
        # }
        self.usage_data: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def _flatten_usage_entry(self, usage_entry: dict[str, Any]) -> dict[str, Any]:
        """Convert any Pydantic model values in *usage_entry* to plain dicts.

        Some LM backends (e.g., litellm) attach Pydantic model instances such
        as ``PromptTokensDetailsWrapper`` to their usage payloads.  This method
        recursively replaces those instances with their ``model_dump()``
        representation so that callers always receive a JSON-serialisable dict.

        Args:
            usage_entry: A raw usage dict as returned by an LM call.

        Returns:
            A new dict with identical keys but with all Pydantic model values
            replaced by their ``model_dump()`` equivalents.
        """
        result: dict[str, Any] = {}
        for key, value in usage_entry.items():
            if isinstance(value, BaseModel):
                # Convert Pydantic models to dicts, like `PromptTokensDetailsWrapper` from litellm.
                result[key] = value.model_dump()
            else:
                result[key] = value
        return result

    def _merge_usage_entries(
        self, usage_entry1: dict[str, Any] | None, usage_entry2: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Recursively merge two usage entry dicts by summing numeric fields.

        Merge rules:
        - If either entry is ``None`` or empty, return a copy of the other.
        - For keys whose values are both dicts, recurse.
        - For all other keys, add the two values treating ``None`` as ``0``.

        Args:
            usage_entry1: The first usage dict (may be ``None`` or empty).
            usage_entry2: The second usage dict (may be ``None`` or empty).

        Returns:
            A new dict containing the element-wise sum of both entries.
        """
        if usage_entry1 is None or len(usage_entry1) == 0:
            return dict(usage_entry2)  # type: ignore[arg-type]
        if usage_entry2 is None or len(usage_entry2) == 0:
            return dict(usage_entry1)

        result = dict(usage_entry2)
        for k, v in usage_entry1.items():
            current_v = result.get(k)
            if isinstance(v, dict) or isinstance(current_v, dict):
                result[k] = self._merge_usage_entries(current_v, v)
            elif current_v is not None or v is not None:
                result[k] = (current_v or 0) + (v or 0)
        return result

    def add_usage(self, lm: str, usage_entry: dict[str, Any]) -> None:
        """Add a usage entry to the tracker."""
        if len(usage_entry) > 0:
            self.usage_data[lm].append(self._flatten_usage_entry(usage_entry))

    def get_total_tokens(self) -> dict[str, dict[str, Any]]:
        """Calculate total tokens from all tracked usage."""
        total_usage_by_lm: dict[str, dict[str, Any]] = {}
        for lm, usage_entries in self.usage_data.items():
            total_usage: dict[str, Any] = {}
            for usage_entry in usage_entries:
                total_usage = self._merge_usage_entries(total_usage, usage_entry)
            total_usage_by_lm[lm] = total_usage
        return total_usage_by_lm


@contextmanager
def track_usage() -> Generator[UsageTracker, None, None]:
    """Context manager for tracking LM token usage across DSPy calls.

    Creates a fresh :class:`UsageTracker`, installs it into the DSPy settings
    context so that all LM calls made within the ``with`` block record their
    usage, then yields the tracker for inspection.

    Yields:
        A :class:`UsageTracker` instance.  After the ``with`` block exits,
        call :meth:`~UsageTracker.get_total_tokens` to retrieve the aggregated
        token counts keyed by LM name.

    Example::

        with dspy.track_usage() as tracker:
            pred = my_program(question="What is 2+2?")

        print(tracker.get_total_tokens())
        # {"openai/gpt-4o-mini": {"prompt_tokens": 42, "completion_tokens": 7}}
    """
    tracker = UsageTracker()

    with settings.context(usage_tracker=tracker):
        yield tracker
