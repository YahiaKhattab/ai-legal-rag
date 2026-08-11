from legal_rag.ingestion.normalization import normalize_text


def test_normalizes_arabic_for_retrieval() -> None:
    source = (
        "  إِسْتِخْدَامُ الـتِّكْنُولُوجِيَا\u200f   "
        "المالية  "
    )

    assert normalize_text(source) == "استخدام التكنولوجيا المالية"


def test_preserves_line_boundaries() -> None:
    source = "المادة الأولى\n\nنص المادة"

    assert normalize_text(source) == "المادة الاولي\nنص المادة"


def test_reverses_arabic_digit_runs_only_when_requested() -> None:
    source = "قانون رقم ٠٢ لسنة ٤٢٠٢"

    assert normalize_text(source) == source
    assert normalize_text(source, reverse_arabic_digit_runs=True) == (
        "قانون رقم ٢٠ لسنة ٢٠٢٤"
    )


def test_removes_corrupted_private_use_glyphs() -> None:
    source = "رئيس مجلس الادارة\n\uf020\uf021\uf022\nرقم الايداع"

    assert normalize_text(source) == "رئيس مجلس الادارة\nرقم الايداع"
