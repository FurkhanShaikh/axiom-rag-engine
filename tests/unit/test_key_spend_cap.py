"""
Per-API-key daily spend cap (API-7).

Rate limits cap requests, not money. With ``AXIOM_KEY_DAILY_BUDGET_USD`` each
key's LLM cost is summed per UTC day — failed and cancelled runs included — and
a key at its cap is refused with 429 before any model is called.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from axiom_rag_engine.api.routes.synthesize import _key_id
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend
from axiom_rag_engine.spend import (
    MemorySpendLedger,
    RedisSpendLedger,
    seconds_until_utc_midnight,
    utc_day,
)

_KEY = "spend-test-key-0001"  # test fixture, not a secret
_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
# No "models": with auth on, callers may not choose them (the server defaults are
# used, and litellm is patched in every test that runs the pipeline).
_BODY = {"request_id": "s", "user_query": "alpha batteries chemistry"}


def _app(**overrides: Any) -> Any:
    return create_app(
        Settings(env="test", api_keys=[_KEY], **overrides),
        search_backend=MockSearchBackend(
            [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
        ),
    )


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    return response


async def _not_json(**kwargs: Any) -> MagicMock:
    return _reply("not json")


async def test_memory_ledger_sums_per_day_and_keeps_two_days() -> None:
    ledger = MemorySpendLedger()
    await ledger.add("k", "2026-09-22", 1.0)
    await ledger.add("k", "2026-09-23", 0.25)
    await ledger.add("k", "2026-09-23", 0.5)
    await ledger.add("k", "2026-09-24", 2.0)
    assert await ledger.spent("k", "2026-09-23") == 0.75
    assert await ledger.spent("k", "2026-09-22") == 0.0  # pruned
    assert await ledger.spent("other", "2026-09-24") == 0.0


def test_day_boundaries_are_utc() -> None:
    late = datetime(2026, 9, 24, 23, 59, 30, tzinfo=UTC).timestamp()
    assert utc_day(late) == "2026-09-24"
    assert seconds_until_utc_midnight(late) == 30


class _FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.data: dict[str, float] = {}
        self.expiry: dict[str, int] = {}
        self.fail = fail

    async def get(self, key: str) -> str | None:
        if self.fail:
            raise ConnectionError("down")
        return str(self.data[key]) if key in self.data else None

    async def incrbyfloat(self, key: str, amount: float) -> float:
        if self.fail:
            raise ConnectionError("down")
        self.data[key] = self.data.get(key, 0.0) + amount
        return self.data[key]

    async def expire(self, key: str, seconds: int) -> None:
        self.expiry[key] = seconds


async def test_redis_ledger_shares_totals_with_expiry() -> None:
    redis = _FakeRedis()
    await RedisSpendLedger(redis).add("k", "2026-09-24", 0.4)
    await RedisSpendLedger(redis).add("k", "2026-09-24", 0.1)  # another replica
    assert await RedisSpendLedger(redis).spent("k", "2026-09-24") == 0.5
    assert redis.expiry["axiom:spend:2026-09-24:k"] >= 24 * 3600


async def test_redis_outage_falls_back_to_the_process_total() -> None:
    ledger = RedisSpendLedger(_FakeRedis(fail=True))
    await ledger.add("k", "2026-09-24", 0.3)
    assert await ledger.spent("k", "2026-09-24") == 0.3


def test_key_at_its_cap_is_refused_before_any_model_call() -> None:
    app = _app(key_daily_budget_usd=1.0)
    called = MagicMock(side_effect=AssertionError("no model call expected"))
    with TestClient(app) as client, patch("litellm.acompletion", called):
        ledger = app.state.services.spend_ledger
        client.portal.call(ledger.add, _key_id(_KEY), utc_day(), 1.0)
        resp = client.post("/v1/synthesize", json=_BODY, headers={"X-API-Key": _KEY})
        stream = client.post("/v1/synthesize/stream", json=_BODY, headers={"X-API-Key": _KEY})
    assert resp.status_code == 429
    assert "AXIOM_KEY_DAILY_BUDGET_USD" in resp.json()["detail"]
    assert 0 < int(resp.headers["Retry-After"]) <= 24 * 3600
    assert stream.status_code == 429
    called.assert_not_called()


def test_failed_runs_are_charged_to_the_key() -> None:
    app = _app(key_daily_budget_usd=5.0, max_llm_calls_per_request=2)
    metric = {"key_id": _key_id(_KEY)}
    before = REGISTRY.get_sample_value("axiom_key_spend_usd_total", metric) or 0.0
    with (
        TestClient(app) as client,
        patch("litellm.acompletion", side_effect=_not_json),
        patch("litellm.completion_cost", return_value=0.01),
    ):
        resp = client.post("/v1/synthesize", json=_BODY, headers={"X-API-Key": _KEY})
        spent = client.portal.call(app.state.services.spend_ledger.spent, _key_id(_KEY), utc_day())
    assert resp.status_code == 500  # two unparseable replies fail the synthesizer
    assert spent == 0.02  # two calls, each priced at $0.01
    after = REGISTRY.get_sample_value("axiom_key_spend_usd_total", metric) or 0.0
    assert round(after - before, 6) == 0.02


def test_stream_runs_are_charged_to_the_key() -> None:
    app = _app(key_daily_budget_usd=5.0, max_llm_calls_per_request=2)
    with (
        TestClient(app) as client,
        patch("litellm.acompletion", side_effect=_not_json),
        patch("litellm.completion_cost", return_value=0.01),
    ):
        client.post("/v1/synthesize/stream", json=_BODY, headers={"X-API-Key": _KEY})
        spent = client.portal.call(app.state.services.spend_ledger.spent, _key_id(_KEY), utc_day())
    assert spent == 0.02


def test_no_cap_means_no_refusal() -> None:
    app = _app()
    with TestClient(app) as client:
        client.portal.call(app.state.services.spend_ledger.add, _key_id(_KEY), utc_day(), 1e6)
        with patch("litellm.acompletion", side_effect=_not_json):
            resp = client.post("/v1/synthesize", json=_BODY, headers={"X-API-Key": _KEY})
    assert resp.status_code != 429
