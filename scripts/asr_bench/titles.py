"""Series titles from release file and folder names, and the locked titles.

``title_key`` reduces a release file or folder name to its series title: it
drops the media suffix, bracketed groups (release group, quality tags, CRC,
alternative titles), and everything from the first season, episode or
release marker on (``S01``, ``S01E02``, ``S01-05``, ``Season 2``,
``Seasons 1-2``, ``2nd Season``, `` - 03``, ``E05``, a trailing episode
number or version tag such as ``02``, ``02v2``, ``02 v2`` or ``v2``,
``1080p``, ``WEB-DL`` and the like), then casefolds and collapses
punctuation to single spaces. Every season of a series therefore shares one
key.

The E0 locked confirmation titles (L1-L4) are excluded from every bench set
by title, in any season and under any of their names. Their identity stays
private: the names live only in ``locked_titles_path()`` under the bench
root, never in the repository, as
``{"schema_version": "asr-bench-locked-titles-1.0",
"titles": [{"key": ..., "aliases": [...]}]}``. An alias matches as whole
words anywhere in a file or folder name (media suffix and bracketed groups
removed), as a substring for non-ASCII aliases, so the locked check does
not depend on where ``title_key`` cuts a name; ``key`` only labels the
title. ``load_locked_titles`` fails closed: a missing,
empty or malformed list, or one that misses a locked fixture's title, stops
the builder.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from .sets import bench_root

MEDIA_SUFFIXES = (".mkv", ".mp4", ".m4v", ".webm", ".avi", ".mov", ".m2ts")
LOCKED_TITLES_SCHEMA = "asr-bench-locked-titles-1.0"
_BRACKETS_RE = re.compile(r"\[[^\[\]]*\]|\([^()]*\)|\{[^{}]*\}|【[^【】]*】|「[^「」]*」")
_TITLE_END_RE = re.compile(
    r"""(?ix)
    \bS\d{1,2}(?=$|[^a-z0-9]|[ep]\d)      # S01, S01E02, S01-05, S01+SP, S01P01
    | \bSeasons?\s*\d+
    | \b\d+(?:st|nd|rd|th)\s+Season\b
    | \s-\s*\d                             # " - 03"
    | \bEP?\d{1,4}\b                       # E03, EP03
    | \b\d{1,4}(?:\s*v\d+)?\s*$            # trailing episode number, version tag
    | \bv\d+\s*$                           # trailing version tag
    | \b(?:\d{3,4}p|WEB(?:-?DL|-?Rip)?|BD(?:Rip)?|Blu-?Ray|HEVC|AVC|AV1|x26[45]|H\s?26[45]
         |AAC\d*|Opus|FLAC|E-?AC-?3|DDP?|Dual[\s-]Audio|Multi[\s-]?Subs?|10[\s-]?bits?|REPACK)\b
    """
)
_SEASON_EPISODE_RE = re.compile(
    r"(?i)\bS(\d{1,2})\s?E(\d{1,4})"
    r"|\bS(\d{1,2})\s+-\s*(\d{1,4})(?:v\d+)?\b"
    r"|\bSeason\s*(\d{1,2})\s+-\s*(\d{1,4})(?:v\d+)?\b"
    r"|\b(\d{1,2})(?:st|nd|rd|th)\s+Season\s+-\s*(\d{1,4})(?:v\d+)?\b"
)
_DASH_EPISODE_RE = re.compile(r"\s-\s*(\d{1,4})(?:v\d+)?\b")
_TRAILING_EPISODE_RE = re.compile(r"\b(\d{1,4})(?:\s*v\d+)?\s*$")
_FRACTIONAL_EPISODE_RE = re.compile(r"\s-\s*\d{1,4}\.\d+(?!\d)")
_FOLDER_SEASON_RE = re.compile(r"(?i)\bS(\d{1,2})\b|\bSeason\s*(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\s+Season\b")
_SEASONS_RE = re.compile(r"(?i)\bseasons\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _bare(name: str) -> str:
    """The name without media suffix and bracketed groups, dots and
    underscores turned into spaces (scene-style names)."""
    if Path(name).suffix.lower() in MEDIA_SUFFIXES:
        name = Path(name).stem
    while (stripped := _BRACKETS_RE.sub(" ", name)) != name:
        name = stripped
    return unicodedata.normalize("NFKC", re.sub(r"[._]+", " ", name))


def title_key(name: str) -> str | None:
    """The series title of a release file or folder name, or None when the
    name holds no title (``S01E01-Episode name.mkv``, ``Season 1``)."""
    name = _bare(name)
    end = _TITLE_END_RE.search(name)
    key = _normalize(name[: end.start()] if end else name)
    return key or None


def season_episode(name: str) -> tuple[int | None, int | None]:
    """Season and episode numbers of an episode file name, when it holds
    them (``S02E05``, ``S2 - 05``, ``Season 02 - 05``, ``Title - 05``,
    ``Title 05``). A fractional episode number (``Title - 16.5``, a recap or
    special) gives neither."""
    stem = Path(name).stem if Path(name).suffix.lower() in MEDIA_SUFFIXES else name
    while (stripped := _BRACKETS_RE.sub(" ", stem)) != stem:
        stem = stripped
    if _FRACTIONAL_EPISODE_RE.search(stem):
        return None, None
    name = _bare(name)
    if match := _SEASON_EPISODE_RE.search(name):
        season, episode = (int(group) for group in match.groups() if group is not None)
        return season, episode
    if match := _DASH_EPISODE_RE.search(name) or _TRAILING_EPISODE_RE.search(name):
        return None, int(match[1])
    return None, None


def folder_season(name: str) -> int | None:
    """The season a folder name holds (``Title S2``, ``Title Season 01``,
    ``2nd Season``), or None when it names none, several, or ``Seasons``.
    Bracketed groups are ignored, as in ``title_key``."""
    name = _bare(name)
    if _SEASONS_RE.search(name):
        return None
    found = {int(next(group for group in match.groups() if group)) for match in _FOLDER_SEASON_RE.finditer(name)}
    return found.pop() if len(found) == 1 else None


def _contains(key: str, alias: str) -> bool:
    if not alias.isascii():
        return alias.replace(" ", "") in key.replace(" ", "")
    words, needle = key.split(), alias.split()
    return any(words[i : i + len(needle)] == needle for i in range(len(words) - len(needle) + 1))


def locked_titles_path() -> Path:
    return bench_root() / "config" / "locked-titles.private.json"


@dataclass(frozen=True)
class LockedTitles:
    """The private locked-title list: each key with its normalized aliases,
    and the digest of the file it was read from."""

    titles: tuple[tuple[str, tuple[str, ...]], ...]
    sha256: str

    def title_of(self, key: str | None) -> str | None:
        """The locked title one of whose aliases a title key or normalized
        name contains, or None."""
        if key is None:
            return None
        for title, aliases in self.titles:
            if any(_contains(key, alias) for alias in aliases):
                return title
        return None

    def path_title(self, path: Path) -> str | None:
        """The locked title named anywhere in a file's own name or the name of
        any folder above it. The whole name is searched, not only its title
        key, so a change to ``title_key`` cannot narrow this check."""
        path = Path(path)
        for name in (path.name, *(parent.name for parent in path.parents)):
            if name and (title := self.title_of(_normalize(_bare(name)))):
                return title
        return None


def load_locked_titles(
    locked_sources: Iterable[tuple[str, Path]], path: Path | None = None
) -> LockedTitles:
    """Read the private locked-title list and check it against the E0
    locked fixtures (label, source path): each source file's title must be
    listed. Any problem stops the caller."""
    path = Path(path or locked_titles_path())

    def refuse(problem: str) -> NoReturn:
        raise SystemExit(
            f"locked-title list {path}: {problem}. The bench builders refuse to run without "
            f"a complete list (schema {LOCKED_TITLES_SCHEMA}, see docs/ASR-BENCH.md)."
        )

    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError:
        refuse("missing")
    if not raw_bytes.strip():
        refuse("empty")
    try:
        raw = json.loads(raw_bytes)
    except ValueError:
        refuse("not valid JSON")
    if not isinstance(raw, dict) or raw.get("schema_version") != LOCKED_TITLES_SCHEMA:
        refuse(f"schema_version is not {LOCKED_TITLES_SCHEMA}")
    entries = raw.get("titles")
    if not isinstance(entries, list) or not entries:
        refuse("lists no titles")
    titles: list[tuple[str, tuple[str, ...]]] = []
    for entry in entries:
        key = entry.get("key") if isinstance(entry, dict) else None
        aliases = entry.get("aliases") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not key.strip() or not isinstance(aliases, list):
            refuse("every title needs a key and a list of aliases")
        normalized = tuple(_normalize(alias) for alias in aliases if isinstance(alias, str))
        if not normalized or not all(normalized) or len(normalized) != len(aliases):
            refuse(f"title {key!r} needs non-empty text aliases")
        titles.append((key, normalized))
    locked = LockedTitles(tuple(titles), hashlib.sha256(raw_bytes).hexdigest())
    checked = 0
    for label, source in locked_sources:
        checked += 1
        if locked.title_of(title_key(Path(source).name)) is None:
            refuse(f"no listed title matches the source file of locked fixture {label}")
    if not checked:
        refuse("no E0 locked fixtures to check it against")
    return locked
