"""
The JSON endpoint cancels its pipeline when the client disconnects.

Starlette does not cancel a handler when its client goes away, so
``POST /v1/synthesize`` kept running — and spending LLM budget — for a response
nobody would read. The SSE endpoint already stopped on disconnect.
"""

from __future__ import annotations

import asyncio

import pytest

from axiom_rag_engine.api.routes import synthesize as synthesize_mod
from axiom_rag_engine.api.routes.synthesize import (
    ClientDisconnectedError,
    run_unless_disconnected,
)


class _Request:
    """Stands in for a Starlette Request: disconnects after ``after`` polls."""

    def __init__(self, after: int | None) -> None:
        self.after = after
        self.polls = 0

    async def is_disconnected(self) -> bool:
        self.polls += 1
        return self.after is not None and self.polls > self.after


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synthesize_mod, "_DISCONNECT_POLL_SECONDS", 0.001)


async def test_work_is_cancelled_when_the_client_disconnects() -> None:
    progressed: list[int] = []
    cancelled = asyncio.Event()

    async def _pipeline() -> str:
        try:
            for step in range(1000):
                progressed.append(step)
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "finished"

    with pytest.raises(ClientDisconnectedError):
        await run_unless_disconnected(_Request(after=3), _pipeline())

    assert cancelled.is_set()
    steps = len(progressed)
    await asyncio.sleep(0.02)
    assert len(progressed) == steps  # nothing runs after cancellation
    assert steps < 1000


async def test_result_is_returned_when_the_client_stays() -> None:
    async def _pipeline() -> str:
        await asyncio.sleep(0.005)
        return "finished"

    assert await run_unless_disconnected(_Request(after=None), _pipeline()) == "finished"


async def test_pipeline_errors_propagate() -> None:
    async def _pipeline() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_unless_disconnected(_Request(after=None), _pipeline())
