"""Helpers for interpreting native PDF text-layer ordering."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

_ARABIC_DIGITS = frozenset(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
)


def uses_right_to_left_digit_storage(page: Any) -> bool:
    """Return whether Arabic digit runs are stored from right to left.

    Some Arabic PDFs store every glyph in decreasing horizontal order. That is
    correct for Arabic letters, but it reverses the logical value of numbers in
    plain-text extraction. We inspect character coordinates and only enable the
    correction when decreasing-x digit pairs outnumber increasing-x pairs.
    """

    raw_page = page.get_text("rawdict", sort=False)
    decreasing_pairs = 0
    increasing_pairs = 0

    for block in raw_page.get("blocks", []):
        for line in block.get("lines", []):
            characters = [
                character
                for span in line.get("spans", [])
                for character in span.get("chars", [])
            ]
            for left, right in pairwise(characters):
                if left.get("c") not in _ARABIC_DIGITS:
                    continue
                if right.get("c") not in _ARABIC_DIGITS:
                    continue

                left_x = float(left["bbox"][0])
                right_x = float(right["bbox"][0])
                if right_x < left_x:
                    decreasing_pairs += 1
                elif right_x > left_x:
                    increasing_pairs += 1

    return decreasing_pairs > increasing_pairs
