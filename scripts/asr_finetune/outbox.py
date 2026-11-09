"""The only files a pilot pod task may receive: ``check_outbox_file``.

Pod tasks (``python -m scripts.asr_bench.remote run --tasks``) upload files
from ``<pilot>\\outbox`` only (``<pilot>`` is ``<bench root>\\ft-pilot-v1``),
and each file passes ``check_outbox_file`` when the tasks file is read and
again right before its upload. The check fails closed: a missing, empty or
unreadable hygiene list refuses every file.

The lists it reads stay on this machine:

- ``<pilot>\\hygiene\\bench-lines.json``: ``{"lines": [...]}``, the normalized
  reference line of every bench set (required);
- ``<pilot>\\hygiene\\titles.json``: ``{"titles": [...]}``, every bench title
  and every locked title and alias, normalized (required);
- when present, ``<bench root>\\config\\locked-titles.private.json``
  (``titles[].key`` and ``aliases``) and the owner's franchise exclusions
  ``<bench root>\\config\\ft-franchise-exclusions.private.json`` (``names``).

Refusals name the file and the rule, never the title or line that matched.
"""

from __future__ import annotations

import functools
import gzip
import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.asr_bench.sets import bench_root
from scripts.asr_finetune.pilot import pilot_root
from scripts.e0_evidence.normalize_ja import normalize_minimal

MAX_BYTES = 64_000_000
# Text checks: a title or alias shorter than this matches whole tokens only;
# a value matches a bench line when they share a shingle of this length.
SHORT_TITLE_CHARS = 4
SHINGLE_CHARS = 8
CV_AUDIO = "cv-audio"
_TEXT_SUFFIXES = (".jsonl.gz", ".jsonl", ".json")


class OutboxRefusal(ValueError):
    """A file that must not reach a pod."""


def outbox_dir() -> Path:
    return pilot_root() / "outbox"


def hygiene_dir() -> Path:
    return pilot_root() / "hygiene"


def _norm(text: str) -> str:
    """``normalize_minimal`` (NFKC, no whitespace, punctuation or symbols),
    case-folded."""
    return normalize_minimal(text).casefold()


def _tokens(text: str) -> set[str]:
    """The case-folded runs of ``text`` (after NFKC) between whitespace,
    punctuation and symbols."""
    tokens: set[str] = set()
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", text):
        if character.isspace() or unicodedata.category(character)[0] in ("P", "S"):
            if current:
                tokens.add("".join(current).casefold())
                current = []
        else:
            current.append(character)
    if current:
        tokens.add("".join(current).casefold())
    return tokens


@dataclass(frozen=True)
class _Hygiene:
    shingles: frozenset[str]
    short_titles: frozenset[str]
    long_titles: re.Pattern[str] | None

    def finding(self, value: str) -> str | None:
        """Which rule ``value`` breaks, or None."""
        normalized = _norm(value)
        if self.long_titles is not None and self.long_titles.search(normalized):
            return "it holds a bench, locked or franchise title or alias"
        if self.short_titles and (normalized in self.short_titles or _tokens(value) & self.short_titles):
            return "it holds a bench, locked or franchise title or alias (whole token)"
        if any(normalized[i:i + SHINGLE_CHARS] in self.shingles
               for i in range(len(normalized) - SHINGLE_CHARS + 1)):
            return f"it holds a shingle ({SHINGLE_CHARS} characters) of a bench reference line"
        return None


def _refuse(path: Path, why: str) -> OutboxRefusal:
    return OutboxRefusal(f"refusing to upload {path}: {why}")


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OutboxRefusal(f"the {what} {path} cannot be read ({type(error).__name__}); "
                            "the outbox guard fails closed") from None


def _strings(raw: Any, key: str, path: Path, what: str) -> list[str]:
    values = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise OutboxRefusal(f"the {what} {path} holds no {key!r} list of strings; the outbox guard fails closed")
    return values


def _hygiene_files() -> tuple[Path, Path, Path, Path]:
    config = bench_root() / "config"
    return (hygiene_dir() / "bench-lines.json", hygiene_dir() / "titles.json",
            config / "locked-titles.private.json", config / "ft-franchise-exclusions.private.json")


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


def _hygiene() -> _Hygiene:
    files = _hygiene_files()
    return _load_hygiene(files, tuple(_stamp(path) for path in files))


@functools.lru_cache(maxsize=2)
def _load_hygiene(files: tuple[Path, Path, Path, Path], stamps: tuple[Any, ...]) -> _Hygiene:
    """The lists, read again whenever a file changes (``stamps``)."""
    lines_path, titles_path, locked_path, franchise_path = files
    for path in (lines_path, titles_path):
        if not path.is_file():
            raise OutboxRefusal(f"the hygiene list {path} is missing; the outbox guard fails closed "
                                "(build it with scripts.asr_finetune.hygiene)")
    lines = _strings(_read_json(lines_path, "hygiene list"), "lines", lines_path, "hygiene list")
    titles = _strings(_read_json(titles_path, "hygiene list"), "titles", titles_path, "hygiene list")
    if locked_path.is_file():
        raw = _read_json(locked_path, "locked-title list")
        entries = raw.get("titles") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            raise OutboxRefusal(f"the locked-title list {locked_path} has no 'titles' list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise OutboxRefusal(f"the locked-title list {locked_path} holds an entry that is not an object")
            titles += [v for v in [entry.get("key"), *(entry.get("aliases") or [])] if isinstance(v, str)]
    if franchise_path.is_file():
        raw = _read_json(franchise_path, "franchise exclusion list")
        names = raw.get("names") if isinstance(raw, dict) else None
        if not isinstance(names, list) or not all(isinstance(v, str) for v in names):
            raise OutboxRefusal(f"the franchise exclusion list {franchise_path} has no 'names' list of strings")
        titles += names
    shingles = frozenset(
        normalized[i:i + SHINGLE_CHARS]
        for normalized in map(_norm, lines)
        for i in range(len(normalized) - SHINGLE_CHARS + 1)
    )
    normalized_titles = {title for title in map(_norm, titles) if title}
    short = frozenset(title for title in normalized_titles if len(title) < SHORT_TITLE_CHARS)
    long = sorted((title for title in normalized_titles if len(title) >= SHORT_TITLE_CHARS), key=len, reverse=True)
    pattern = re.compile("|".join(map(re.escape, long))) if long else None
    return _Hygiene(shingles, short, pattern)


@functools.lru_cache(maxsize=2)
def _guard(root: str) -> Any:
    from scripts.asr_bench.remote import Guard

    return Guard()


def _guard_check(path: Path) -> None:
    from scripts.asr_bench.remote import PolicyRefusal

    try:
        _guard(str(bench_root())).check(path)
    except PolicyRefusal as refusal:
        raise OutboxRefusal(str(refusal)) from None


def _walk(raw: Any) -> Iterator[str]:
    """Every string of a decoded JSON value, object keys included."""
    if isinstance(raw, str):
        yield raw
    elif isinstance(raw, dict):
        for key, value in raw.items():
            yield key
            yield from _walk(value)
    elif isinstance(raw, list):
        for value in raw:
            yield from _walk(value)


def _documents(path: Path, kind: str) -> Iterator[tuple[str, Any]]:
    """(where, decoded value) of a JSON file or of each JSON Lines line."""
    if kind == ".json":
        yield "", json.loads(path.read_text(encoding="utf-8"))
        return
    opener = gzip.open if kind == ".jsonl.gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield f" (line {number})", json.loads(line)


def _allowed_clips() -> set[str]:
    path = outbox_dir() / "allow.json"
    raw = _read_json(path, "allow-list")
    clips = ((raw.get("cv") or {}).get("clips") if isinstance(raw, dict) else None) or []
    return {clip["path"] for clip in clips if isinstance(clip, dict) and isinstance(clip.get("path"), str)}


def text_finding(value: str) -> str | None:
    """The text rule of ``check_outbox_file`` that ``value`` breaks (it holds a
    bench, locked or franchise title or alias, or a shingle of a bench
    reference line), or None; never what matched. Fails closed like the check:
    a missing or unreadable hygiene list raises ``OutboxRefusal``."""
    return _hygiene().finding(value)


def check_outbox_file(path: Path) -> None:
    """Refuse unless: the resolved path lies under outbox_dir(); the suffix is
    .json, .jsonl or .jsonl.gz (.mp3 only under outbox\\cv-audio\\ and named in
    outbox\\allow.json); the size is at most 64 MB; remote.Guard admits it; and
    no decoded string value contains a normalized bench, locked or franchise
    title or alias (whole-token match below 4 characters) or any 8-character
    shingle of a normalized bench reference line (lists from <pilot>\\hygiene\\,
    built by scripts.asr_finetune.hygiene). A missing hygiene list refuses too:
    the guard fails closed. The text checks also apply to the file's path under
    the outbox."""
    resolved = Path(path).resolve()
    root = outbox_dir().resolve()
    if root not in resolved.parents:
        raise _refuse(path, f"it is not inside the outbox {root}")
    if not resolved.is_file():
        raise _refuse(path, "it is not a file")
    relative = resolved.relative_to(root)
    hygiene = _hygiene()
    finding = hygiene.finding(relative.as_posix())
    if finding:
        # Not named: its name holds what matched.
        raise OutboxRefusal(f"refusing to upload a file of the outbox {root}: "
                            + finding.replace("it holds", "its path below the outbox holds", 1))
    name = resolved.name.lower()
    kind = next((suffix for suffix in _TEXT_SUFFIXES if name.endswith(suffix)), None)
    if kind is None:
        if not name.endswith(".mp3"):
            raise _refuse(path, "only .json, .jsonl and .jsonl.gz files (and allow-listed Common Voice .mp3 "
                                f"clips under {CV_AUDIO}\\) may be uploaded")
        if len(relative.parts) != 2 or relative.parts[0] != CV_AUDIO:
            raise _refuse(path, f"an .mp3 file must lie directly in the outbox's {CV_AUDIO} folder")
        if resolved.name not in _allowed_clips():
            raise _refuse(path, "the clip is not named in the outbox's allow.json")
    size = resolved.stat().st_size
    if size > MAX_BYTES:
        raise _refuse(path, f"it has {size:,} bytes, over the {MAX_BYTES:,}-byte limit")
    _guard_check(resolved)
    if kind is None:
        return
    finding = None
    try:
        for where, document in _documents(resolved, kind):
            finding = next((f + where for f in map(hygiene.finding, _walk(document)) if f), None)
            if finding:
                break
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as error:
        raise _refuse(path, f"it cannot be decoded as {kind} ({type(error).__name__})") from None
    if finding:
        raise _refuse(path, finding)
