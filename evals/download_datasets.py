"""Download eval datasets into evals/data/ (gitignored).

Fetches SciFact (https://github.com/allenai/scifact) by default, used by the
semantic-verifier, retrieval, and corpus evals. The archive is ~3 MB; only the
claim and corpus files are kept.

``asqa`` fetches the ASQA dev split (948 ambiguous questions with gold short
answers, Apache-2.0) for calibration_eval.py, via the Hugging Face rows API
(mirror ``din0s/asqa``; the original Google Storage URL is gone).

Usage:
    python tasks.py evals download
    # or directly:
    uv run python evals/download_datasets.py            # SciFact
    uv run python evals/download_datasets.py asqa       # ASQA dev
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
ASQA_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=din0s/asqa&config=default"
    "&split=dev&offset={offset}&length={length}"
)

EVALS_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALS_DIR / "data"
SCIFACT_DIR = DATA_DIR / "scifact"
_WANTED_FILES = ("claims_dev.jsonl", "claims_train.jsonl", "corpus.jsonl")
ASQA_PATH = DATA_DIR / "asqa" / "dev.jsonl"


def _echo(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def download_scifact() -> None:
    if all((SCIFACT_DIR / name).exists() for name in _WANTED_FILES):
        _echo(f"SciFact already present in {SCIFACT_DIR} - nothing to do.")
        return

    SCIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _echo(f"Downloading SciFact from {SCIFACT_URL} ...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "scifact.tar.gz"
        urllib.request.urlretrieve(SCIFACT_URL, archive)  # noqa: S310 - fixed https URL
        _echo("Extracting ...")
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if name in _WANTED_FILES and member.isfile():
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    (SCIFACT_DIR / name).write_bytes(src.read())
                    _echo(f"  wrote {SCIFACT_DIR / name}")
    _echo("Done.")


def _fetch_json(url: str, attempts: int = 5) -> dict:
    """GET JSON, retrying transient server errors (the rows API 502s under load)."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
                return dict(json.load(resp))
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        _echo(f"  transient error, retrying ({attempt}/{attempts}) ...")
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def download_asqa() -> None:
    if ASQA_PATH.exists():
        _echo(f"ASQA already present at {ASQA_PATH} - nothing to do.")
        return
    ASQA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset, page = 0, 100
    _echo("Downloading ASQA dev from the Hugging Face rows API ...")
    while True:
        data = _fetch_json(ASQA_ROWS_URL.format(offset=offset, length=page))
        batch = [item["row"] for item in data.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(data.get("num_rows_total", 0)):
            break
    with ASQA_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _echo(f"  wrote {len(rows)} questions to {ASQA_PATH}")


if __name__ == "__main__":
    if sys.argv[1:] == ["asqa"]:
        download_asqa()
    else:
        download_scifact()
