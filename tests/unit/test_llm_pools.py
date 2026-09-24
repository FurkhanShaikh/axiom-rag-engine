"""
Verification calls are not queued behind synthesis calls.

One process-wide semaphore covered every LLM call, so a few slow synthesis
calls held every slot while cheap verifier calls — many per request, often on
another provider — waited. Synthesis, verification and auxiliary calls now
have separate pools (LLM-4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.utils import llm


@pytest.fixture(autouse=True)
def _fresh_pools(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("AXIOM_MAX_CONCURRENT_LLM", "1")
    get_settings.cache_clear()
    llm._llm_semaphores.clear()
    yield
    llm._llm_semaphores.clear()
    get_settings.cache_clear()


def test_nodes_map_to_pools() -> None:
    assert llm.llm_pool("synthesizer") == "synthesis"
    assert {llm.llm_pool(n) for n in ("semantic", "corroboration", "contradiction")} == {
        "verification"
    }
    assert llm.llm_pool("reranker") == "auxiliary"
    assert llm.llm_pool("embedding") == "auxiliary"


async def test_verifier_call_runs_while_synthesis_pool_is_full() -> None:
    release = asyncio.Event()

    async def slow_synthesis() -> str:
        await release.wait()
        return "answer"

    async def verdict() -> str:
        return "passed"

    synthesis = asyncio.create_task(llm._with_retry("synthesizer", "m", slow_synthesis))
    await asyncio.sleep(0)  # the synthesis call now holds the only synthesis slot
    assert llm.get_llm_semaphore("synthesis").locked()

    assert await asyncio.wait_for(llm._with_retry("semantic", "m", verdict), 1.0) == "passed"

    release.set()
    assert await synthesis == "answer"


async def test_synthesis_calls_still_share_their_limit() -> None:
    running = 0
    peak = 0

    async def call() -> Any:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    await asyncio.gather(*(llm._with_retry("synthesizer", "m", call) for _ in range(4)))
    assert peak == 1


def test_verifier_limit_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXIOM_MAX_CONCURRENT_VERIFIER_LLM", "7")
    get_settings.cache_clear()
    assert llm.get_llm_semaphore("verification")._value == 7
    assert llm.get_llm_semaphore("synthesis")._value == 1


def test_verifier_limit_defaults_to_the_general_limit() -> None:
    assert llm.get_llm_semaphore("verification")._value == 1
