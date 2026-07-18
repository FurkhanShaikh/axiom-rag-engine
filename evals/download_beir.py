"""Download a BEIR-format IR dataset for the retrieval eval.

BEIR (https://github.com/beir-cellar/beir) hosts standard IR benchmarks as zip
archives: corpus.jsonl, queries.jsonl, and qrels/*.tsv. The retrieval eval reads
that format directly (see retrieval_eval.load_beir), so this just fetches and
unpacks one into evals/data/beir/<name>/ (gitignored, like SciFact).

Datasets vary in what they stress. `arguana` (argument -> best counter-argument)
has low query/answer lexical overlap, so it is a clean test of semantic
retrieval — the confound-free complement to the SciFact paraphrase A/B.

Usage:
    uv run python evals/download_beir.py arguana
"""

from __future__ import annotations

import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

BEIR_DIR = Path(__file__).resolve().parent / "data" / "beir"
_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _echo(msg: str = "") -> None:
    sys.stdout.write(f"{msg}\n")


def _ssl_context() -> ssl.SSLContext:
    # Windows Python often lacks the CA for this host in its default store;
    # certifi's bundle (shipped with the toolchain) resolves it.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(name: str) -> Path:
    dest = BEIR_DIR / name
    if (dest / "corpus.jsonl").exists() and (dest / "queries.jsonl").exists():
        _echo(f"{name} already present at {dest}")
        return dest

    url = f"{_BASE_URL}/{name}.zip"
    _echo(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=180, context=_ssl_context()) as resp:  # noqa: S310
        blob = resp.read()
    _echo(f"  got {len(blob) / 1e6:.1f} MB, extracting ...")

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            # Archives nest under "<name>/"; flatten into dest.
            rel = member.split(f"{name}/", 1)[-1]
            if not rel or member.endswith("/"):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(member))

    corpus_n = sum(1 for _ in (dest / "corpus.jsonl").open(encoding="utf-8"))
    queries_n = sum(1 for _ in (dest / "queries.jsonl").open(encoding="utf-8"))
    _echo(f"Extracted to {dest}: {corpus_n} docs, {queries_n} queries")
    return dest


def main() -> None:
    if len(sys.argv) < 2:
        _echo("Usage: python evals/download_beir.py <dataset-name>  (e.g. arguana)")
        sys.exit(1)
    download(sys.argv[1])


if __name__ == "__main__":
    main()
