"""Package eval results and caches so published numbers can be re-graded.

Raw results (``evals/results``) and the caches that make reruns reproducible
(live Tavily responses, rerank grades, paraphrases, optionally embeddings) are
gitignored. ``pack`` puts them in one ``.tar.gz`` with a manifest (SHA-256 per
file, git commit, date) to attach to a GitHub release; ``unpack`` verifies the
manifest and restores the files, after which the evals rerun against the same
search results instead of today's live web. Datasets are not bundled: they are
public and fetched with ``python tasks.py evals download``.

Usage:
    python tasks.py evals bundle -- pack                     # -> evals-bundle-<date>.tar.gz
    python tasks.py evals bundle -- pack --with-embeddings   # also the embedding cache
    python tasks.py evals bundle -- unpack evals-bundle-2026-09-24.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

EVALS_DIR = Path(__file__).resolve().parent
MANIFEST = "MANIFEST.json"

# Paths relative to evals/: results, and the caches that pin live inputs.
_RESULTS = "results"
_CACHES = (
    "data/query_expansion_cache.json",  # live Tavily responses (query-expansion eval)
    "data/calibration_search_cache.json",  # live Tavily responses (tier calibration)
    "data/rerank_cache",  # LLM rerank grades
    "data/scifact/paraphrases_dev.jsonl",  # LLM paraphrases (vocabulary-mismatch A/B)
)
_EMBEDDINGS = "data/embeddings"


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(root: Path, with_embeddings: bool = False) -> list[Path]:
    """The files under ``root`` (an evals/ directory) that belong in a bundle."""
    entries = [_RESULTS, *_CACHES, *([_EMBEDDINGS] if with_embeddings else [])]
    files: list[Path] = []
    for entry in entries:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
    return files


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def pack(root: Path, out: Path, with_embeddings: bool = False) -> dict:
    """Write ``out`` (tar.gz) with every bundled file and a manifest."""
    files = collect(root, with_embeddings)
    manifest = {
        "created": time.strftime("%Y-%m-%d"),
        "git_commit": _git_commit(root),
        "files": {},
    }
    with tarfile.open(out, "w:gz") as tar:
        for path in files:
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            manifest["files"][rel] = {"sha256": _sha256(data), "bytes": len(data)}
            tar.add(path, arcname=rel)
        blob = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    return manifest


def _safe(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(parts) and not PurePosixPath(name).is_absolute() and ".." not in parts


def _read(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"unreadable member: {member.name}")
    return handle.read()


def unpack(archive: Path, root: Path) -> dict:
    """Verify ``archive`` against its manifest and restore its files under ``root``.

    Raises ValueError, writing nothing, if a member is missing, unlisted,
    altered, or would land outside ``root``.
    """
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        if MANIFEST not in members:
            raise ValueError("bundle has no MANIFEST.json")
        manifest = json.loads(_read(tar, members.pop(MANIFEST)))
        listed = manifest.get("files", {})
        if set(members) != set(listed):
            raise ValueError("bundle contents do not match its manifest")
        payload: dict[str, bytes] = {}
        for name, member in members.items():
            if not _safe(name):
                raise ValueError(f"unsafe path in bundle: {name}")
            data = _read(tar, member)
            if _sha256(data) != listed[name]["sha256"]:
                raise ValueError(f"checksum mismatch: {name}")
            payload[name] = data
    for name, data in payload.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack", help="bundle results and caches")
    p_pack.add_argument("--out", type=Path, default=None)
    p_pack.add_argument("--with-embeddings", action="store_true")
    p_unpack = sub.add_parser("unpack", help="verify and restore a bundle")
    p_unpack.add_argument("archive", type=Path)
    args = parser.parse_args()

    if args.command == "pack":
        out = args.out or Path(f"evals-bundle-{time.strftime('%Y-%m-%d')}.tar.gz")
        manifest = pack(EVALS_DIR, out, args.with_embeddings)
        _echo(f"Wrote {out} ({len(manifest['files'])} files, commit {manifest['git_commit']}).")
        _echo("Attach it to a GitHub release and link it from BENCHMARKS.md.")
    else:
        manifest = unpack(args.archive, EVALS_DIR)
        _echo(
            f"Restored {len(manifest['files'])} files from {manifest['created']} "
            f"(commit {manifest['git_commit']}) into {EVALS_DIR}."
        )


if __name__ == "__main__":
    main()
