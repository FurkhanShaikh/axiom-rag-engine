"""Generate meaning-preserving, reworded paraphrases of SciFact claims.

Purpose: a controlled test of *when* dense retrieval helps. Paraphrasing a claim
with synonyms lowers its lexical overlap with the gold document without changing
which document answers it — so BM25 (lexical) should degrade while dense
(semantic) holds up. Running the retrieval eval on original vs. paraphrased
queries isolates vocabulary mismatch as the single variable, which is a proxy
for Axiom's real regime: colloquial web-search queries against formal pages.

Paraphrases are cached to a JSONL (gitignored, alongside SciFact) so the A/B is
rerunnable without regenerating. Regeneration needs a local Ollama model.

Usage:
    uv run python evals/make_paraphrases.py --limit 80 --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
SCIFACT_DIR = EVALS_DIR / "data" / "scifact"
OUT_PATH = SCIFACT_DIR / "paraphrases_dev.jsonl"

_PROMPT = (
    "Rewrite this scientific claim to mean exactly the same thing but using "
    "different words and synonyms wherever possible. Keep it to one sentence. "
    "Reply with ONLY the rewritten sentence, no preamble or quotes.\n\nClaim: {claim}"
)


def _echo(msg: str = "") -> None:
    sys.stdout.write(f"{msg}\n")


def _generate(prompt: str, model: str, base_url: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7},
            "think": False,
        }
    ).encode()
    req = urllib.request.Request(  # noqa: S310 — operator-controlled Ollama host
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        return json.loads(resp.read()).get("response", "").strip()


def _labeled_claims() -> list[dict]:
    claims = []
    for line in (SCIFACT_DIR / "claims_dev.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("evidence"):
            claims.append(obj)
    # Deterministic order (by id) so a given --limit always covers the same claims.
    return sorted(claims, key=lambda c: str(c["id"]))


def main() -> None:
    from _env import load_dotenv

    load_dotenv()
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80, help="Claims to paraphrase (0 = all)")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama generation model")
    args = parser.parse_args()
    base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

    claims = _labeled_claims()
    sample = claims[: args.limit] if args.limit else claims

    # Resume: keep any already-cached paraphrases so a re-run only fills gaps.
    existing: dict[str, dict] = {}
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["claim_id"])] = row

    _echo(f"Paraphrasing {len(sample)} claims with {args.model} @ {base}")
    out_rows: list[dict] = []
    for i, claim in enumerate(sample, 1):
        cid = str(claim["id"])
        if cid in existing and existing[cid].get("paraphrase"):
            out_rows.append(existing[cid])
            continue
        para = _generate(_PROMPT.format(claim=claim["claim"]), args.model, base)
        out_rows.append({"claim_id": cid, "original": claim["claim"], "paraphrase": para})
        if i % 10 == 0:
            _echo(f"  {i}/{len(sample)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
        encoding="utf-8",
    )
    _echo(f"Wrote {len(out_rows)} paraphrases to {OUT_PATH}")


if __name__ == "__main__":
    main()
