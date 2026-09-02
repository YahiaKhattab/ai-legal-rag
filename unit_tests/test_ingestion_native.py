"""Unit tests for legal_rag.ingestion.native.uses_right_to_left_digit_storage.

The real function reads a PyMuPDF `page.get_text("rawdict", ...)` payload.
We build tiny fake objects that only implement `.get_text()` returning the
exact nested dict shape the function expects, so no real PDF is needed.
"""

from __future__ import annotations

from legal_rag.ingestion.native import uses_right_to_left_digit_storage


class _FakePage:
    def __init__(self, rawdict):
        self._rawdict = rawdict

    def get_text(self, mode, sort=False):
        assert mode == "rawdict"
        return self._rawdict


def _char(c, x):
    return {"c": c, "bbox": [x, 0, x + 5, 10]}


def _rawdict(chars):
    return {"blocks": [{"lines": [{"spans": [{"chars": chars}]}]}]}


def test_returns_true_when_arabic_digits_decrease_in_x():
    # Digits "1" then "2" stored with decreasing x (right-to-left order).
    chars = [_char("\u0661", x=100), _char("\u0662", x=80)]
    page = _FakePage(_rawdict(chars))
    assert uses_right_to_left_digit_storage(page) is True


def test_returns_false_when_arabic_digits_increase_in_x():
    chars = [_char("\u0661", x=80), _char("\u0662", x=100)]
    page = _FakePage(_rawdict(chars))
    assert uses_right_to_left_digit_storage(page) is False


def test_ignores_non_digit_characters():
    chars = [_char("a", x=100), _char("b", x=80)]
    page = _FakePage(_rawdict(chars))
    assert uses_right_to_left_digit_storage(page) is False


def test_returns_false_for_empty_page():
    page = _FakePage({"blocks": []})
    assert uses_right_to_left_digit_storage(page) is False


def test_majority_vote_across_multiple_pairs():
    # Two decreasing pairs vs one increasing pair -> True overall.
    chars = [
        _char("\u0661", x=100),
        _char("\u0662", x=80),
        _char("\u0663", x=60),
        _char("\u0664", x=90),
    ]
    page = _FakePage(_rawdict(chars))
    assert uses_right_to_left_digit_storage(page) is True


def test_handles_missing_optional_keys_gracefully():
    page = _FakePage({"blocks": [{"lines": [{"spans": [{}]}]}]})
    assert uses_right_to_left_digit_storage(page) is False
