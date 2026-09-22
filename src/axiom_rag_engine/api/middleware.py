"""
Axiom Engine — request-body size limit.

A hard cap on raw request-body size. The default (128 KiB) is far larger than
any legitimate AxiomRequest (user_query is capped at 10k chars + small configs)
and small enough to stop trivial OOM / slow-parse DoS with oversized bodies.
Document ingestion carries whole documents, so ``/v1/documents*`` gets its own,
larger cap.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

CallNext = Callable[[Request], Awaitable[Response]]


def make_body_size_middleware(
    max_body_bytes: int, max_document_bytes: int
) -> Callable[[Request, CallNext], Awaitable[Response]]:
    """Build an ``http`` middleware enforcing the two body-size caps."""

    def _limit_for(path: str) -> int:
        return max_document_bytes if path.startswith("/v1/documents") else max_body_bytes

    async def enforce_body_size(request: Request, call_next: CallNext) -> Response:
        if request.method in ("POST", "PUT", "PATCH"):
            cap = _limit_for(request.url.path)
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > cap:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": f"Request body exceeds {cap} bytes."},
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Malformed Content-Length header."},
                    )
            else:
                # No Content-Length (chunked transfer-encoding): buffer the stream
                # with a hard byte cap so oversized bodies cannot bypass the limit.
                chunks: list[bytes] = []
                total = 0
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > cap:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": f"Request body exceeds {cap} bytes."},
                        )
                    chunks.append(chunk)
                # Cache the buffered body so downstream handlers can still read it.
                request._body = b"".join(chunks)
        return await call_next(request)

    return enforce_body_size
