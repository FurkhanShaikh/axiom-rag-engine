"""
Supply-chain and compose hardening guards (DEP-2 / #58).

Enforces:
- Every third-party GitHub Action in `.github/` is pinned by full 40-char commit SHA.
- `docker-compose.yml` uses no `:latest` tags on pulled images and does not
  publish Ollama's unauthenticated port 11434.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
_SHA_PIN_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_SERVICE_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
_IMAGE_RE = re.compile(r"^\s+image:\s*([^\s#]+)")
_PORTS_RE = re.compile(r"^\s+ports:\s*$")
_BUILD_RE = re.compile(r"^\s+build:\s*$")


def _github_yaml_files() -> list[Path]:
    github_dir = _REPO_ROOT / ".github"
    files = sorted((github_dir / "workflows").glob("*.yml"))
    files.extend(sorted((github_dir / "actions").glob("*/action.yml")))
    return files


def test_github_actions_are_pinned_by_commit_sha() -> None:
    files = _github_yaml_files()
    assert files, "Expected workflow/action YAML files under .github/"

    checked = 0
    unpinned: list[str] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _USES_RE.match(line)
            if not match:
                continue
            ref = match.group(1)
            if ref.startswith("./"):
                continue
            checked += 1
            if not _SHA_PIN_RE.match(ref):
                unpinned.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {ref}")

    assert checked > 0, "Expected at least one third-party GitHub Action reference"
    assert not unpinned, "Unpinned GitHub Actions found:\n" + "\n".join(unpinned)


def test_compose_images_pinned_and_ollama_not_published() -> None:
    compose = _REPO_ROOT / "docker-compose.yml"
    services: dict[str, dict[str, object]] = {}
    current: str | None = None
    in_services = False

    for line in compose.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line == "services:":
            in_services = True
            continue
        if in_services and not line.startswith(" "):
            in_services = False
            current = None
            continue
        if not in_services:
            continue

        svc_match = _SERVICE_RE.match(line)
        if svc_match:
            current = svc_match.group(1)
            services[current] = {"image": None, "has_build": False, "has_ports": False}
            continue
        if current is None:
            continue

        if _BUILD_RE.match(line):
            services[current]["has_build"] = True
        elif _PORTS_RE.match(line):
            services[current]["has_ports"] = True
        elif img_match := _IMAGE_RE.match(line):
            services[current]["image"] = img_match.group(1)

    assert "ollama" in services
    assert not services["ollama"]["has_ports"], "Ollama port must not be published"

    for name, info in services.items():
        image = info["image"]
        assert isinstance(image, str), f"Service {name} is missing an image"
        if info["has_build"]:
            continue
        assert not image.endswith(":latest"), f"Service {name} uses :latest ({image})"
        assert ":" in image or "@sha256:" in image, f"Service {name} image lacks tag ({image})"
