"""
Observability hygiene (OBS-5).

- The text formatter prefixed the request ID by mutating the shared log record,
  so a second handler (or a second format call) stacked the prefix.
- ``/metrics`` exposes model usage and spend; it can now require a token.
- The dashboard shows the token and cost metrics the README advertises.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

from fastapi.testclient import TestClient

from axiom_rag_engine.config.logging import _TextFormatter, request_id_ctx
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

_SCRAPE_CREDENTIAL = "scrape-me"  # test fixture, not a secret
_DASHBOARD = Path(__file__).resolve().parents[2] / "deploy" / "grafana" / "axiom-engine.json"


def test_request_id_prefix_is_not_stacked_across_handlers() -> None:
    logger = logging.getLogger("axiom_test.two_handlers")
    logger.propagate = False
    streams = [io.StringIO(), io.StringIO()]
    handlers = [logging.StreamHandler(s) for s in streams]
    for handler in handlers:
        handler.setFormatter(_TextFormatter())
        logger.addHandler(handler)
    token = request_id_ctx.set("req-42")
    try:
        logger.warning("synthesis done")
    finally:
        request_id_ctx.reset(token)
        for handler in handlers:
            logger.removeHandler(handler)

    for stream in streams:
        line = stream.getvalue()
        assert line.count("[req-42]") == 1
        assert line.rstrip().endswith("[req-42] synthesis done")


def test_formatting_leaves_the_record_untouched() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("you",), None)
    token = request_id_ctx.set("req-1")
    try:
        formatter = _TextFormatter()
        assert formatter.format(record).endswith("[req-1] hello you")
        assert formatter.format(record).endswith("[req-1] hello you")
    finally:
        request_id_ctx.reset(token)
    assert record.msg == "hello %s"


def _client(**overrides: object) -> TestClient:
    app = create_app(Settings(env="test", **overrides), search_backend=MockSearchBackend([]))
    return TestClient(app)


def test_metrics_are_open_by_default() -> None:
    with _client() as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "axiom_llm_cost_usd_total" in resp.text


def test_metrics_token_is_enforced() -> None:
    with _client(metrics_token=_SCRAPE_CREDENTIAL) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401
        ok = client.get("/metrics", headers={"Authorization": f"Bearer {_SCRAPE_CREDENTIAL}"})
    assert ok.status_code == 200
    assert "axiom_llm_tokens_total" in ok.text


def test_scrapes_are_not_rate_limited() -> None:
    with _client(rate_limit="2/minute") as client:
        codes = {client.get("/metrics").status_code for _ in range(5)}
    assert codes == {200}


def test_dashboard_shows_tokens_and_cost() -> None:
    exprs = [t["expr"] for p in json.loads(_DASHBOARD.read_text())["panels"] for t in p["targets"]]
    assert any("axiom_llm_tokens_total" in e for e in exprs)
    assert any("axiom_llm_cost_usd_total" in e for e in exprs)


def test_dashboard_queries_name_real_metrics() -> None:
    # Every axiom_* series a panel queries must be one the app exports.
    with _client() as client:
        exported = client.get("/metrics").text
    exprs = [t["expr"] for p in json.loads(_DASHBOARD.read_text())["panels"] for t in p["targets"]]
    for name in {m for e in exprs for m in re.findall(r"axiom_[a-z0-9_]+", e)}:
        base = re.sub(r"_(bucket|count|sum)$", "", name)
        assert base in exported, name
