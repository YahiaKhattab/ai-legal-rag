from typing import Any

from legal_rag.ingestion.native import uses_right_to_left_digit_storage


class FakePage:
    def __init__(self, characters: list[tuple[str, float]]) -> None:
        self._characters = characters

    def get_text(self, output: str, *, sort: bool) -> dict[str, Any]:
        assert output == "rawdict"
        assert sort is False
        chars = [
            {"c": character, "bbox": (x, 0.0, x + 1.0, 1.0)}
            for character, x in self._characters
        ]
        return {"blocks": [{"lines": [{"spans": [{"chars": chars}]}]}]}


def test_detects_right_to_left_digit_storage() -> None:
    page = FakePage([("٤", 40.0), ("٢", 30.0), ("\u0660", 20.0), ("٢", 10.0)])

    assert uses_right_to_left_digit_storage(page)


def test_keeps_left_to_right_digit_storage_unchanged() -> None:
    page = FakePage([("٢", 10.0), ("\u0660", 20.0), ("٢", 30.0), ("٤", 40.0)])

    assert not uses_right_to_left_digit_storage(page)


def test_requires_coordinate_evidence_before_correcting() -> None:
    page = FakePage([("ق", 40.0), ("٢", 30.0), ("م", 20.0)])

    assert not uses_right_to_left_digit_storage(page)
