"""PDF text extraction in a child process, bounded by page count and time.

Uploaded PDFs are untrusted input, and pypdf work on a crafted file (huge page
trees, deeply nested content streams) can run for a very long time. Parsing in
a worker thread could not be stopped — a thread cannot be cancelled — so one bad
upload could pin a worker indefinitely. Extraction now runs in a spawned child
process that is killed when it overruns, and documents over a page limit are
refused before any page is parsed.

This module is deliberately light (stdlib + a lazy pypdf import): the spawned
child imports it, and importing the corpus package instead would pull in the
whole engine.
"""

from __future__ import annotations

import io
import multiprocessing
from multiprocessing.connection import Connection


class PdfExtractionError(Exception):
    """The PDF was refused, unreadable, or took too long to parse."""


def _worker(data: bytes, max_pages: int, conn: Connection) -> None:
    """Child-process entry point: send ("ok", text) or ("error", reason)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
        if page_count > max_pages:
            conn.send(
                (
                    "error",
                    f"PDF has {page_count} pages; the limit is {max_pages} "
                    "(AXIOM_CORPUS_MAX_PDF_PAGES).",
                )
            )
            return
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        conn.send(("ok", "\n\n".join(p for p in pages if p)))
    except Exception as exc:  # any parser failure is a bad document, not a crash
        conn.send(("error", f"could not read PDF: {exc}"))
    finally:
        conn.close()


def extract_pdf_text(data: bytes, *, max_pages: int, timeout_seconds: float) -> str:
    """Extract a PDF's text, joined by blank lines, in a killable child process.

    Raises:
        PdfExtractionError: over ``max_pages``, unreadable, the parser crashed,
            or extraction exceeded ``timeout_seconds`` (the child is killed).
    """
    # spawn, not fork: the server process has threads (event loop, executors),
    # and forking a threaded process can deadlock the child.
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(data, max_pages, sender), daemon=True)
    process.start()
    sender.close()  # the child holds the only writer, so its exit reads as EOF
    try:
        if not receiver.poll(timeout_seconds):
            raise PdfExtractionError(
                f"PDF extraction exceeded {timeout_seconds:g}s (AXIOM_CORPUS_PDF_TIMEOUT_SECONDS)."
            )
        status, payload = receiver.recv()
    except EOFError as exc:
        raise PdfExtractionError("PDF extraction process exited without a result.") from exc
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        receiver.close()
    if status != "ok":
        raise PdfExtractionError(str(payload))
    return str(payload)
