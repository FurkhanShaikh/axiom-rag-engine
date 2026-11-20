"""
Axiom Engine — request-body size limit.

A hard cap on raw request-body size enforced as pure ASGI middleware. The
default (128 KiB) is far larger than any legitimate AxiomRequest (user_query is
capped at 10k chars + small configs) and small enough to stop trivial OOM /
slow-parse DoS with oversized bodies. Document ingestion carries whole
documents, so ``/v1/documents*`` gets its own, larger cap.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _header_value(scope: Scope, target: bytes) -> str | None:
    """Return the decoded value for ``target`` (lowercase header name), if present."""
    for name, value in scope.get("headers", ()):
        if bytes(name).lower() == target:
            return bytes(value).decode("latin-1")
    return None


async def _send_json_error(
    status_code: int,
    detail: str,
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    response = JSONResponse(status_code=status_code, content={"detail": detail})
    await response(scope, receive, send)


class BodySizeLimitMiddleware:
    """Pure ASGI middleware enforcing request-body size caps without BaseHTTPMiddleware."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int, max_document_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_document_bytes = max_document_bytes

    def _limit_for(self, path: str) -> int:
        return self.max_document_bytes if path.startswith("/v1/documents") else self.max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "GET") not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        cap = self._limit_for(scope.get("path", ""))
        declared = _header_value(scope, b"content-length")
        if declared is not None:
            try:
                content_length = int(declared)
            except ValueError:
                await _send_json_error(
                    400, "Malformed Content-Length header.", scope, receive, send
                )
                return
            if content_length < 0:
                await _send_json_error(
                    400, "Malformed Content-Length header.", scope, receive, send
                )
                return
            if content_length > cap:
                await _send_json_error(
                    413, f"Request body exceeds {cap} bytes.", scope, receive, send
                )
                return

        chunks: list[bytes] = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            msg_type = message.get("type")
            if msg_type == "http.disconnect":
                return
            if msg_type != "http.request":
                continue
            chunk = message.get("body", b"")
            if chunk:
                total += len(chunk)
                if total > cap:
                    await _send_json_error(
                        413, f"Request body exceeds {cap} bytes.", scope, receive, send
                    )
                    return
                chunks.append(chunk)
            more_body = bool(message.get("more_body", False))

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
