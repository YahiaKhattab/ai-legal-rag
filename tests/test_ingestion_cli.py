from pathlib import Path

import pytest

import legal_rag.ingestion.cli as cli_module
from legal_rag.ingestion.cli import _expand_inputs, _parser
from legal_rag.ingestion.models import IngestionStatus
from legal_rag.ingestion.pipeline import IngestionSummary


def test_parser_exposes_provenance_and_token_contract() -> None:
    arguments = _parser().parse_args(
        [
            "law.pdf",
            "--document-type",
            "regulation",
            "--source",
            "Test Authority",
            "--target-tokens",
            "350",
            "--overlap-tokens",
            "50",
            "--maximum-tokens",
            "470",
        ]
    )

    assert arguments.document_type == "regulation"
    assert arguments.source == "Test Authority"
    assert arguments.language == "auto"
    assert arguments.target_tokens == 350
    assert arguments.overlap_tokens == 50
    assert arguments.maximum_tokens == 470


def test_expand_inputs_sorts_supported_document_files(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "rules.docx").touch()
    (tmp_path / "legacy.doc").touch()

    assert _expand_inputs([tmp_path]) == [
        tmp_path / "a.pdf",
        tmp_path / "b.pdf",
        tmp_path / "notes.txt",
        tmp_path / "rules.docx",
    ]


def test_main_reports_processed_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf_path = tmp_path / "law.pdf"
    pdf_path.touch()

    class FakePipeline:
        def __init__(self, **options: object) -> None:
            assert options["expected_language"] == "auto"

        def ingest(
            self,
            path: Path,
            output: Path,
            **options: object,
        ) -> IngestionSummary:
            assert path == pdf_path
            assert options["source"] == "Test Authority"
            return IngestionSummary(
                status=IngestionStatus.PROCESSED,
                source_file=path.name,
                document_id="a" * 64,
                processed_records=2,
                direct_records=1,
                ocr_records=1,
                failed_records=0,
                chunks=3,
                sources_output=output / "sources.jsonl",
                chunks_output=output / "chunks.jsonl",
                report_output=output / "report.json",
            )

    monkeypatch.setattr(cli_module, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(
        "sys.argv",
        ["legal-rag-ingest", str(pdf_path), "--source", "Test Authority"],
    )

    assert cli_module.main() == 0
    assert "PROCESSED law.pdf" in capsys.readouterr().out


def test_main_rejects_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(tmp_path)])

    with pytest.raises(SystemExit, match="No supported PDF, DOCX, or TXT files found"):
        cli_module.main()


def test_pages_rejects_non_pdf_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    txt_path = tmp_path / "law.txt"
    txt_path.write_text("legal text", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["legal-rag-ingest", str(txt_path), "--pages", "1"])

    with pytest.raises(SystemExit, match="every input is PDF"):
        cli_module.main()
