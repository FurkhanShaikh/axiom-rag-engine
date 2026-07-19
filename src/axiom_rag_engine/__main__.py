"""Axiom Engine command-line interface.

Installed as the `axiom-rag-engine` console script via `[project.scripts]` in
pyproject.toml. Subcommands:

    axiom-rag-engine serve                      Run the FastAPI HTTP server.
    axiom-rag-engine probe "..."                Send a test query to a running server.
    axiom-rag-engine check-config               Print the resolved Settings (secrets redacted).
    axiom-rag-engine audit <request_id>         Fetch a retained audit trail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Config-section grouping for `check-config` — cosmetic only; unknown keys
# are placed under "Other" so new Settings fields stay visible without edits.
# ---------------------------------------------------------------------------

_CONFIG_SECTIONS: list[tuple[str, list[str]]] = [
    ("Runtime", ["env", "docs_enabled", "log_format"]),
    ("Auth", ["api_keys"]),
    (
        "LLM defaults",
        ["default_synthesizer_model", "default_verifier_model", "ollama_api_base"],
    ),
    (
        "LLM budget & concurrency",
        [
            "max_llm_calls_per_request",
            "max_tokens_per_request",
            "max_concurrent_llm",
            "allowed_metric_models",
        ],
    ),
    (
        "Rate limiting & cache",
        ["rate_limit", "cache_ttl_seconds", "cache_max_size", "redis_url"],
    ),
    (
        "Search & scoring",
        [
            "allow_mock_search",
            "authoritative_domains",
            "low_quality_domains",
            "exclude_default_domains",
        ],
    ),
    (
        "Verification",
        ["semantic_verification_enabled", "min_usable_ranking_score"],
    ),
    ("Audit & logging", ["audit_retention", "log_audit_events"]),
    (
        "Security limits",
        ["cors_origins", "trusted_proxy_ips", "max_body_bytes"],
    ),
]


def _env_name_for(field_name: str, alias: str | None) -> str:
    """Map a Settings field to its canonical env var name."""
    if alias:
        return alias
    return f"AXIOM_{field_name.upper()}"


def _source_for(env_name: str) -> str:
    """Return where the effective value came from."""
    if env_name in os.environ:
        return "env"
    # Settings loads .env at process start, so if the var is in the file we
    # treat it as .env-sourced — best-effort; a present key with no value is
    # still reported as env.
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(dotenv_path):
        try:
            with open(dotenv_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.lstrip().startswith(f"{env_name}="):
                        return ".env"
        except OSError:
            pass
    return "default"


def _render_config_text(settings: Any, redacted: dict[str, Any]) -> str:
    """Render redacted settings as sectioned text with value + source."""
    fields = type(settings).model_fields
    env_names: dict[str, str] = {
        name: _env_name_for(name, field.alias) for name, field in fields.items()
    }

    used: set[str] = set()
    lines: list[str] = []
    lines.append("Axiom Engine — resolved configuration")
    lines.append("=" * 68)

    def _emit_section(title: str, keys: list[str]) -> None:
        present = [k for k in keys if k in redacted]
        if not present:
            return
        lines.append("")
        lines.append(f"[{title}]")
        width = max(len(k) for k in present)
        for key in present:
            used.add(key)
            env_name = env_names.get(key, f"AXIOM_{key.upper()}")
            source = _source_for(env_name)
            value = redacted[key]
            lines.append(f"  {key:<{width}} = {value!r:<40}  [{source}]  ({env_name})")

    for title, keys in _CONFIG_SECTIONS:
        _emit_section(title, keys)

    leftovers = sorted(k for k in redacted if k not in used)
    if leftovers:
        _emit_section("Other", leftovers)

    lines.append("")
    lines.append(
        "Source legend: [env] = exported; [.env] = dotenv file; [default] = Settings default."
    )
    lines.append(
        "TAVILY_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY are read by vendor SDKs and not shown."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "axiom_rag_engine.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from axiom_rag_engine.cli.probe import run_probe

    return run_probe(
        query=args.query,
        server_url=args.url,
        model=args.model,
        debug=args.debug,
    )


def _cmd_check_config(args: argparse.Namespace) -> int:
    from axiom_rag_engine.config.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    data = settings.redacted_dict()

    if args.format == "json":
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        return 0

    sys.stdout.write(_render_config_text(settings, data))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from axiom_rag_engine.cli.audit import run_audit

    return run_audit(
        request_id=args.request_id,
        server_url=args.url,
        api_key=args.api_key,
        pretty=not args.json,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom-rag-engine",
        description="Citation-verified RAG service.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the Axiom Engine HTTP server.")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")
    serve.set_defaults(func=_cmd_serve)

    probe = sub.add_parser("probe", help="Send a test query to a running server.")
    probe.add_argument("query", help="The question to ask.")
    probe.add_argument("--url", default="http://localhost:8000", help="Server URL.")
    probe.add_argument(
        "--model",
        default="ollama/gemma4:e4b",
        help="LiteLLM model ID for synthesizer + verifier.",
    )
    probe.add_argument("--debug", action="store_true", help="Include audit trail in output.")
    probe.set_defaults(func=_cmd_probe)

    check = sub.add_parser(
        "check-config",
        help="Print the resolved runtime configuration (secrets redacted).",
    )
    check.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    check.set_defaults(func=_cmd_check_config)

    audit = sub.add_parser(
        "audit",
        help="Fetch a retained audit trail by request_id.",
    )
    audit.add_argument("request_id", help="The request_id to look up.")
    audit.add_argument("--url", default="http://localhost:8000", help="Server URL.")
    audit.add_argument(
        "--api-key",
        default=None,
        help="API key (falls back to AXIOM_API_KEY env var).",
    )
    audit.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of the human-readable event log.",
    )
    audit.set_defaults(func=_cmd_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
