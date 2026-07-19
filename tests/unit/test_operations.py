"""Tests for the operator-facing surface added to ease release ops.

Covers:
  - `AuditStore` ring-buffer semantics.
  - `GET /v1/audits/{request_id}` end-to-end (enabled + disabled).
  - `GET /v1/status` config snapshot.
  - CLI `check-config` section rendering + source annotation.
"""

from __future__ import annotations

from axiom_rag_engine.__main__ import _render_config_text
from axiom_rag_engine.audit_store import AuditStore
from axiom_rag_engine.config.settings import get_settings

# ---------------------------------------------------------------------------
# AuditStore
# ---------------------------------------------------------------------------


def test_audit_store_disabled_when_maxsize_zero() -> None:
    store = AuditStore(0)
    assert store.enabled is False
    store.put("req_1", {"x": 1})
    assert store.get("req_1") is None
    assert store.list_ids() == []


def test_audit_store_ring_eviction() -> None:
    store = AuditStore(maxsize=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.put("c", {"v": 3})
    assert store.list_ids() == ["b", "c"]
    assert store.get("a") is None
    assert store.get("c") == {"v": 3}


def test_audit_store_reinsert_moves_to_end() -> None:
    store = AuditStore(maxsize=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.put("a", {"v": 99})
    assert store.list_ids() == ["b", "a"]
    assert store.get("a") == {"v": 99}


def test_audit_store_returns_copy() -> None:
    store = AuditStore(maxsize=1)
    payload = {"v": 1}
    store.put("a", payload)
    retrieved = store.get("a")
    assert retrieved is not None
    retrieved["v"] = 999
    assert store.get("a") == {"v": 1}


# ---------------------------------------------------------------------------
# /v1/audits/{request_id}
# ---------------------------------------------------------------------------


def test_audit_endpoint_404_when_retention_disabled(client) -> None:
    resp = client.get("/v1/audits/req_missing")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "disabled" in detail.lower()


def test_audit_endpoint_returns_retained_entry(monkeypatch, client) -> None:
    from axiom_rag_engine.main import app

    # Swap in a populated store — lifespan created one with retention=0 under
    # the default test settings.
    store = AuditStore(maxsize=5)
    store.put(
        "req_001",
        {
            "request_id": "req_001",
            "status": "success",
            "recorded_at": 1.0,
            "audit_trail": [{"node": "retriever", "event_type": "start"}],
        },
    )
    app.state.audit_store = store

    resp = client.get("/v1/audits/req_001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "req_001"
    assert body["audit_trail"][0]["node"] == "retriever"


def test_audit_endpoint_returns_404_for_unknown_id(monkeypatch, client) -> None:
    from axiom_rag_engine.main import app

    app.state.audit_store = AuditStore(maxsize=5)
    resp = client.get("/v1/audits/never_heard_of")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /v1/status
# ---------------------------------------------------------------------------


def test_status_endpoint_reports_core_fields(client) -> None:
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "axiom-rag-engine"
    assert "uptime_seconds" in body and body["uptime_seconds"] >= 0
    assert body["engine_ready"] is True
    assert body["cache"]["ttl_seconds"] > 0
    assert "limits" in body and body["limits"]["rate_limit"]
    # Never expose raw secrets
    assert "api_keys" not in body
    assert isinstance(body["api_keys_configured"], bool)
    assert body["audit_retention"]["enabled"] is False


def test_status_endpoint_reflects_audit_retention(monkeypatch, client) -> None:
    from axiom_rag_engine.main import app

    app.state.audit_store = AuditStore(maxsize=42)
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["audit_retention"] == {
        "enabled": True,
        "capacity": 42,
        "retained": 0,
    }


# ---------------------------------------------------------------------------
# check-config rendering
# ---------------------------------------------------------------------------


def test_check_config_render_groups_by_section(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_RATE_LIMIT", "13/second")
    get_settings.cache_clear()
    settings = get_settings()
    rendered = _render_config_text(settings, settings.redacted_dict())
    assert "[Runtime]" in rendered
    assert "[LLM defaults]" in rendered
    assert "[Audit & logging]" in rendered
    # Source annotation present and correct
    assert "[env]" in rendered  # rate_limit came from env in this test
    assert "[default]" in rendered  # at least one default
    assert "AXIOM_RATE_LIMIT" in rendered


def test_check_config_hides_secrets(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_API_KEYS", "super-secret-key-1,super-secret-key-2")
    get_settings.cache_clear()
    settings = get_settings()
    rendered = _render_config_text(settings, settings.redacted_dict())
    assert "super-secret" not in rendered
    assert "***" in rendered
