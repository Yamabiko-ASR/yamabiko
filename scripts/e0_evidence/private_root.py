"""Private-root boundary for the scripts-only E0 evidence harness."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.e0_evidence._common import require_uuid4
from scripts.e0_evidence.schema import SCHEMA_VERSION


PRIVATE_ROOT_ENV = "YAMABIKO_E0_PRIVATE_ROOT"
ROOT_MANIFEST_NAME = "root.private.json"


def _resolved(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("private paths must be absolute")
    resolved: Path | None = None
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        pass
    if resolved is None:
        raise ValueError("private path cannot be resolved") from None
    return resolved


def _contains(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _overlaps(first: Path, second: Path) -> bool:
    return _contains(first, second) or _contains(second, first)


@dataclass(frozen=True, slots=True)
class PrivateBoundary:
    root: Path
    worktrees: tuple[Path, ...]
    excluded_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        root = _resolved(self.root)
        worktrees = tuple(_resolved(path) for path in self.worktrees)
        excluded = tuple(_resolved(path) for path in self.excluded_roots)
        if not root.is_dir():
            raise ValueError("private root must be an existing directory")
        if any(_overlaps(root, worktree) for worktree in worktrees):
            raise ValueError("private root overlaps a Git worktree")
        if any(_overlaps(root, denied) for denied in excluded):
            raise ValueError("private root overlaps an excluded root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "worktrees", worktrees)
        object.__setattr__(self, "excluded_roots", excluded)


def discover_git_worktrees(repo: Path) -> tuple[Path, ...]:
    repository = _resolved(Path(repo))
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "worktree", "list", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        pass
    if completed is None:
        raise RuntimeError("cannot enumerate Git worktrees") from None

    worktrees = tuple(
        _resolved(Path(line.removeprefix("worktree ")))
        for line in completed.stdout.splitlines()
        if line.startswith("worktree ")
    )
    if not worktrees:
        raise RuntimeError("Git returned no worktrees")
    return worktrees


def resolve_private_root(
    repo: Path,
    environ: Mapping[str, str],
    *,
    worktrees: Iterable[Path] | None = None,
    excluded_roots: Iterable[Path] = (),
) -> Path:
    repository = _resolved(Path(repo))
    raw = environ.get(PRIVATE_ROOT_ENV)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{PRIVATE_ROOT_ENV} is required")
    root = _resolved(Path(raw))
    if not root.is_dir():
        raise ValueError("private root must be an existing directory")

    discovered = (
        discover_git_worktrees(repository)
        if worktrees is None
        else tuple(worktrees)
    )
    known_worktrees = tuple(
        dict.fromkeys((repository, *(_resolved(path) for path in discovered)))
    )
    denied_roots = tuple(_resolved(path) for path in excluded_roots)
    if any(_overlaps(root, worktree) for worktree in known_worktrees):
        raise ValueError("private root overlaps a Git worktree")
    if any(_overlaps(root, denied) for denied in denied_roots):
        raise ValueError("private root overlaps an excluded root")
    return root


def parse_root_manifest(payload: object) -> tuple[str, tuple[Path, ...]]:
    """Return a root manifest's ID and the excluded roots ``root-init`` persisted.

    A manifest written before exclusions were persisted has no
    ``excluded_roots`` key and persists none.
    """

    required = {"schema_version", "private_root_id"}
    if (
        not isinstance(payload, dict)
        or not required <= set(payload) <= required | {"excluded_roots"}
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("private root manifest has an invalid wire shape")
    root_id = require_uuid4(payload["private_root_id"], "private_root_id")
    raw_roots = payload.get("excluded_roots", [])
    if not isinstance(raw_roots, list) or any(
        type(item) is not str or not Path(item).is_absolute() for item in raw_roots
    ):
        raise ValueError("private root manifest excluded_roots must be absolute paths")
    return root_id, tuple(_resolved(Path(item)) for item in raw_roots)


def persisted_excluded_roots(root: Path) -> tuple[Path, ...]:
    """Read the excluded roots persisted in ``root``'s manifest, if it has one."""

    try:
        raw = (root / ROOT_MANIFEST_NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeError):
        raise ValueError("private root manifest is unreadable") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("private root manifest is unreadable") from None
    return parse_root_manifest(payload)[1]


def require_private_path(path: Path, boundary: PrivateBoundary) -> Path:
    if not isinstance(boundary, PrivateBoundary):
        raise TypeError("boundary must be PrivateBoundary")
    candidate = _resolved(Path(path))
    if not _contains(boundary.root, candidate):
        raise ValueError("path is outside the private root")
    if any(_contains(worktree, candidate) for worktree in boundary.worktrees):
        raise ValueError("private path enters a Git worktree")
    if any(_contains(denied, candidate) for denied in boundary.excluded_roots):
        raise ValueError("private path enters an excluded root")
    return candidate


def redact_private_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: private path redacted"
