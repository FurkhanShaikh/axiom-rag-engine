"""Download eval datasets into evals/data/ (gitignored).

Currently fetches SciFact (https://github.com/allenai/scifact), used by
semantic_verifier_eval.py. The archive is ~3 MB; only the claim and corpus
files are kept.

Usage:
    python tasks.py evals download
    # or directly:
    uv run python evals/download_datasets.py
"""

from __future__ import annotations

import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

EVALS_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALS_DIR / "data"
SCIFACT_DIR = DATA_DIR / "scifact"
_WANTED_FILES = ("claims_dev.jsonl", "claims_train.jsonl", "corpus.jsonl")


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


if __name__ == "__main__":
    download_scifact()
