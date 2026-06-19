"""Assertion utilities for DSPy program validation.

Provides lightweight helpers to enforce invariants on DSPy module
inputs and outputs without disrupting the prediction pipeline.
"""

from __future__ import annotations

from typing import Any


def assert_non_empty(value: Any, field_name: str) -> None:
    """Assert that a DSPy field value is non-empty.

    Args:
        value: The field value to check (string, list, or dict).
        field_name: Human-readable name used in the error message.

    Raises:
        AssertionError: If ``value`` is falsy (empty string, empty list,
            ``None``, etc.).
    """
    assert value, f"DSPy field '{field_name}' must not be empty, got: {value!r}"


def assert_max_tokens(text: str, field_name: str, max_tokens: int = 4096) -> None:
    """Assert that a text field does not exceed a token budget.

    Uses a simple whitespace-split approximation (1 token ≈ 1 word) to
    avoid adding a hard tokenizer dependency.

    Args:
        text: The text to check.
        field_name: Human-readable name used in the error message.
        max_tokens: Maximum number of approximate tokens allowed.

    Raises:
        AssertionError: If the approximate token count exceeds ``max_tokens``.
    """
    approx_tokens = len(text.split())
    assert approx_tokens <= max_tokens, (
        f"DSPy field '{field_name}' exceeds token budget: "
        f"{approx_tokens} > {max_tokens}"
    )
