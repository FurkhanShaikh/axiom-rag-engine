"""
Eval results and caches can be bundled for a release and restored (EVAL-7).

Raw results and the caches that pin live inputs (Tavily responses, rerank
grades) were gitignored, so published numbers could not be re-graded. A bundle
carries them with a checksummed manifest; restoring verifies it first.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_EVALS = Path(__file__).resolve().parents[2] / "evals"
_spec = importlib.util.spec_from_file_location("axiom_eval_bundle", _EVALS / "bundle.py")
assert _spec and _spec.loader
bundle = importlib.util.module_from_spec(_spec)
sys.modules["axiom_eval_bundle"] = bundle
_spec.loader.exec_module(bundle)


def _evals_tree(root: Path) -> None:
    (root / "results").mkdir(parents=True)
    (root / "results" / "retrieval-bm25.json").write_text('{"summary": {}}', encoding="utf-8")
    (root / "data" / "rerank_cache").mkdir(parents=True)
    (root / "data" / "rerank_cache" / "grades.json").write_text("{}", encoding="utf-8")
    (root / "data" / "query_expansion_cache.json").write_text('{"q": []}', encoding="utf-8")
    (root / "data" / "scifact").mkdir()
    (root / "data" / "scifact" / "corpus.jsonl").write_text("dataset", encoding="utf-8")
    (root / "data" / "embeddings").mkdir()
    (root / "data" / "embeddings" / "m.npy").write_bytes(b"\x00" * 8)


def test_round_trip_restores_results_and_caches(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _evals_tree(src)
    archive = tmp_path / "b.tar.gz"
    manifest = bundle.pack(src, archive)

    assert set(manifest["files"]) == {
        "results/retrieval-bm25.json",
        "data/rerank_cache/grades.json",
        "data/query_expansion_cache.json",
    }  # datasets and (by default) embeddings are not bundled
    restored = bundle.unpack(archive, dst)
    assert restored["files"] == manifest["files"]
    assert (dst / "data" / "query_expansion_cache.json").read_text(encoding="utf-8") == '{"q": []}'


def test_embeddings_are_opt_in(tmp_path: Path) -> None:
    _evals_tree(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in bundle.collect(tmp_path, True)}
    assert "data/embeddings/m.npy" in names


def _archive(tmp_path: Path, files: dict[str, bytes], manifest: dict) -> Path:
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {**files, "MANIFEST.json": json.dumps(manifest).encode()}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_altered_files_are_refused(tmp_path: Path) -> None:
    manifest = {"files": {"results/r.json": {"sha256": "0" * 64, "bytes": 2}}}
    archive = _archive(tmp_path, {"results/r.json": b"{}"}, manifest)
    with pytest.raises(ValueError, match="checksum"):
        bundle.unpack(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing written


def test_paths_outside_the_evals_dir_are_refused(tmp_path: Path) -> None:
    data = b"x"
    name = "../escape.txt"
    manifest = {"files": {name: {"sha256": bundle._sha256(data), "bytes": 1}}}
    with pytest.raises(ValueError, match="unsafe path"):
        bundle.unpack(_archive(tmp_path, {name: data}, manifest), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_unlisted_members_are_refused(tmp_path: Path) -> None:
    archive = _archive(tmp_path, {"results/extra.json": b"{}"}, {"files": {}})
    with pytest.raises(ValueError, match="manifest"):
        bundle.unpack(archive, tmp_path / "out")
