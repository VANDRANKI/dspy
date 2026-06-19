"""Unit tests for dspy.utils.assertions."""

import pytest
from dspy.utils.assertions import assert_non_empty, assert_max_tokens


class TestAssertNonEmpty:
    def test_non_empty_string_passes(self):
        assert_non_empty("hello", "answer")  # no exception

    def test_empty_string_raises(self):
        with pytest.raises(AssertionError, match="must not be empty"):
            assert_non_empty("", "answer")

    def test_none_raises(self):
        with pytest.raises(AssertionError):
            assert_non_empty(None, "answer")

    def test_non_empty_list_passes(self):
        assert_non_empty([1, 2, 3], "items")

    def test_empty_list_raises(self):
        with pytest.raises(AssertionError):
            assert_non_empty([], "items")


class TestAssertMaxTokens:
    def test_short_text_passes(self):
        assert_max_tokens("hello world", "answer", max_tokens=10)

    def test_exact_limit_passes(self):
        text = " ".join(["word"] * 10)
        assert_max_tokens(text, "answer", max_tokens=10)

    def test_over_limit_raises(self):
        text = " ".join(["word"] * 11)
        with pytest.raises(AssertionError, match="exceeds token budget"):
            assert_max_tokens(text, "answer", max_tokens=10)
