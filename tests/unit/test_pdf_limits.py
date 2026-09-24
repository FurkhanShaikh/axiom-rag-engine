"""
PDF extraction is bounded by page count and time, in a killable process.

Uploaded PDFs were parsed by pypdf in a worker thread with no page cap and no
time limit; a thread cannot be cancelled, so one crafted file could pin a worker
indefinitely, and parser errors outside the few caught types surfaced as 500s.
"""

from __future__ import annotations

import multiprocessing

import pytest

from axiom_rag_engine.corpus.ingest import IngestionError, extract_text
from axiom_rag_engine.utils.pdf_extract import PdfExtractionError, extract_pdf_text
from tests.conftest import make_pdf

_PDF = make_pdf("Alpha cells use lithium iron phosphate chemistry.")


def test_extracts_text_in_a_child_process() -> None:
    text = extract_pdf_text(_PDF, max_pages=10, timeout_seconds=60)
    assert "lithium iron phosphate" in text


def test_refuses_pdfs_over_the_page_limit() -> None:
    with pytest.raises(PdfExtractionError, match="1 pages; the limit is 0"):
        extract_pdf_text(_PDF, max_pages=0, timeout_seconds=60)


def test_overrunning_extraction_is_killed() -> None:
    # A timeout far below the child's start-up time forces the overrun path.
    with pytest.raises(PdfExtractionError, match="exceeded"):
        extract_pdf_text(_PDF, max_pages=10, timeout_seconds=0.001)
    assert multiprocessing.active_children() == []  # nothing left running


def test_any_parser_failure_is_an_ingestion_error() -> None:
    with pytest.raises(IngestionError, match="could not read PDF"):
        extract_text(b"%PDF-1.4 this is not really a pdf", filename="bad.pdf")


def test_ingest_uses_the_configured_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from axiom_rag_engine.config.settings import get_settings

    monkeypatch.setenv("AXIOM_CORPUS_PDF_TIMEOUT_SECONDS", "0.001")
    get_settings.cache_clear()
    with pytest.raises(IngestionError, match="AXIOM_CORPUS_PDF_TIMEOUT_SECONDS"):
        extract_text(_PDF, filename="slow.pdf")
