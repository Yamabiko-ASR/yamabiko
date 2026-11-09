"""Detect the unused PyAV/FFmpeg media stack in a frozen candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence


# Direct execution puts ``scripts/`` rather than the repository root on
# sys.path. Bootstrap the root before importing the sibling inventory module;
# ``-I`` callers therefore do not depend on an editable-install .pth file.
if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bundle_inventory import InventoryEntry, load_inventory


FORBIDDEN_MEDIA_EXPRESSIONS = (
    r"(?<![a-z0-9])faster_whisper[./\\]audio(?:[./\\]|$)",
    r"(?<![a-z0-9])av(?:[./\\]|$)",
    r"(?<![a-z0-9])(?:lib)?avcodec(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?avdevice(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?avfilter(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?avformat(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?avutil(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?swresample(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?swscale(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?openh264(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?x264(?:[-_.]|$)",
    r"(?<![a-z0-9])(?:lib)?x265(?:[-_.]|$)",
    r"(?<![a-z0-9])ffmpeg(?:[-_.]|$)",
    r"(?<![a-z0-9])ffprobe(?:[-_.]|$)",
    r"(?<![a-z0-9])ffplay(?:[-_.]|$)",
)

_FORBIDDEN_MEDIA = tuple(
    re.compile(expression, flags=re.IGNORECASE)
    for expression in FORBIDDEN_MEDIA_EXPRESSIONS
)
_INVENTORY_PATH_FIELDS = ("logical_name", "source_path", "destination")


def _normalized_path_text(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _is_forbidden(value: str) -> bool:
    normalized = _normalized_path_text(value)
    return any(expression.search(normalized) for expression in _FORBIDDEN_MEDIA)


def find_forbidden_media(
    entries: Iterable[InventoryEntry],
    dist_root: Path | None = None,
) -> tuple[str, ...]:
    """Return sorted, unique diagnostics for forbidden media components."""

    diagnostics: set[str] = set()
    for entry in entries:
        for field in _INVENTORY_PATH_FIELDS:
            value = getattr(entry, field)
            if not isinstance(value, str):
                raise ValueError(f"inventory {field} must be a string")
            if _is_forbidden(value):
                diagnostics.add(f"inventory.{field}: {value}")

    if dist_root is not None:
        root = Path(dist_root)
        if not root.exists():
            raise FileNotFoundError(f"dist root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"dist root is not a directory: {root}")

        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root).as_posix()
            if _is_forbidden(relative):
                diagnostics.add(f"dist: {relative}")

    return tuple(sorted(diagnostics))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail if a PyInstaller inventory or bundle contains PyAV media files."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="path to build/yamabiko/bundle-inventory.json",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        help="optional final dist/yamabiko tree to inspect",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        entries = load_inventory(args.inventory)
        diagnostics = find_forbidden_media(entries, args.dist)
    except (OSError, TypeError, ValueError) as exc:
        print(f"bundle media audit error: {exc}", file=sys.stderr)
        return 2

    for diagnostic in diagnostics:
        print(diagnostic, file=sys.stderr)
    return 1 if diagnostics else 0


if __name__ == "__main__":
    raise SystemExit(main())
