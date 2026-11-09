"""Per-series intelligence layer: profile ingest, disk cache, and the
translation context hint composed from AniList tags + synopsis.

No Qt and no network of its own -- the only wire I/O goes through the
injected AniListClient (``fetch_media_intel``), which owns the failure
taxonomy (429/403/transport -> AniListUnavailable). Spoilers are dropped
at ingest so no downstream consumer can leak one: spoiler-flagged tags
are discarded, and ``~!...!~`` spoiler blocks in the description are
removed WITH their text. Hint-side filtering (rank/category/cap) lives
in ``context_hint``.

The profile cache shares the gazetteer cache directory (one cache-root
convention): ``<app_data_dir>/gazetteers/profile-<media_id>.json``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from yamabiko.fsutil import atomic_write_text

from .gazetteer import gazetteer_path

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tag:
    name: str
    rank: int
    category: str


@dataclass(frozen=True)
class Relation:
    relation_type: str | None
    media_id: int | None
    title: str | None


@dataclass(frozen=True)
class SeriesProfile:
    media_id: int
    genres: list[str]
    tags: list[Tag]
    synopsis_first_sentence: str
    # ANIME relation edges only: Relation carries no medium field (its
    # binding shape is relation_type/media_id/title) and its only
    # consumer, the gazetteer relations carry, is ANIME-only, so
    # MANGA/other nodes are dropped at ingest.
    relations: list[Relation]


def profile_path(media_id: int, cache_dir: Path | None = None) -> Path:
    """profile-<media_id>.json in the gazetteer cache directory.

    The default resolves via gazetteer_path (which routes through
    paths.app_data_dir(), never cwd -- the frozen-bundle trap, hard rule
    #8), so the profile always sits beside the series' gazetteer cache.
    """
    base = Path(cache_dir) if cache_dir is not None else gazetteer_path(media_id).parent
    return base / f"profile-{media_id}.json"


_HTML_TAG_RE = re.compile(r"<[^>]*>")
# AniList spoiler blocks (~!hidden!~): removed WITH their content -- the
# description-level analog of the spoiler-flagged tag drop above. An
# opener with no closer (the live API serves descriptions truncated
# mid-text) drops through to the end: the fail-safe direction is
# over-removal, never exposing half a spoiler.
_SPOILER_RE = re.compile(r"~!.*?(?:!~|$)", re.DOTALL)
# Markdown emphasis markers only -- the emphasized TEXT is kept.
_MD_MARK_RE = re.compile(r"\*{1,3}|_{1,3}|~~")
# Trailing "(Source: ...)" attribution. The closing paren is OPTIONAL:
# the live API serves the attribution truncated mid-word with the paren
# never closed ("(Source: Seven Seas Entertainm", observed 2026-07-10),
# and that tail must not leak into a prompt hint.
_SOURCE_RE = re.compile(r"\(\s*source\s*:[^)]*\)?\s*$", re.IGNORECASE)
# Fullwidth 。！？ included: a Japanese-language AniList description has
# no ASCII terminators, so without them the WHOLE description became the
# "first sentence" and context_hint's 120-char cap then cut it mid-word
# (the word-boundary trim is a no-op on spaceless text).
_SENTENCE_END_RE = re.compile(r"[.!?。！？]")


def _synopsis_first_sentence(description: str) -> str:
    """Clean an AniList description down to its first sentence.

    Order matters: HTML tags, spoiler blocks (delimiters AND enclosed
    text), markdown markers, and the trailing Source attribution are
    stripped BEFORE sentence-splitting, so none of them can masquerade
    as (or hide) the first terminator. The terminator itself is kept; a
    description with no terminator yields the whole cleaned text.
    """
    text = _HTML_TAG_RE.sub("", description)
    text = _SPOILER_RE.sub(" ", text)
    text = _MD_MARK_RE.sub("", text)
    text = _SOURCE_RE.sub("", text).strip()
    m = _SENTENCE_END_RE.search(text)
    sentence = text[: m.end()] if m else text
    return " ".join(sentence.split())


def build_profile(client, media_id: int,
                  cache_dir: Path | None = None) -> SeriesProfile:
    """One fetch_media_intel request -> ingested SeriesProfile, persisted
    atomically (fsutil) to profile-<media_id>.json. Always rewrites the
    cache file, so a stale or corrupt profile heals on the next pick."""
    intel = client.fetch_media_intel(media_id)
    tags = [
        Tag(str(t.get("name") or ""), int(t.get("rank") or 0),
            str(t.get("category") or ""))
        for t in intel.get("tags") or []
        if not (t.get("isMediaSpoiler") or t.get("isGeneralSpoiler"))
    ]
    relations = [
        Relation(r.get("relationType"), r.get("id"), r.get("title"))
        for r in intel.get("relations") or []
        if r.get("type") == "ANIME" and r.get("id") is not None
    ]
    profile = SeriesProfile(
        media_id=media_id,
        genres=[str(g) for g in intel.get("genres") or []],
        tags=tags,
        synopsis_first_sentence=_synopsis_first_sentence(
            str(intel.get("description") or "")),
        relations=relations,
    )
    atomic_write_text(
        profile_path(media_id, cache_dir),
        json.dumps(asdict(profile), ensure_ascii=False, indent=2),
    )
    return profile


def _quarantine(path: Path) -> None:
    """Move a corrupt profile cache aside so the next pick backfills it.

    Mirrors gazetteer._quarantine: both profile writers are
    existence-gated (the tray's pick thread builds only on a gazetteer
    cache miss, and backfills only while profile-<id>.json is MISSING),
    so a corrupt file left in place would satisfy the is_file() gate on
    every re-pick and keep the series hint dead forever -- there is no
    other retry path."""
    backup = path.with_suffix(".json.corrupt.bak")
    _log.warning("corrupt series profile %s; quarantining to %s",
                 path, backup.name, exc_info=True)
    try:
        path.replace(backup)
    except OSError:
        _log.warning("could not quarantine %s", path, exc_info=True)


def load_profile(media_id: int,
                 cache_dir: Path | None = None) -> SeriesProfile | None:
    """Cache read, no network: the profile previously written by
    build_profile, or None when absent or corrupt. Corrupt logs a WARNING
    and means "no hint for this series", never a crash; the corrupt file
    is quarantined aside (.json.corrupt.bak, mirroring
    SeriesGazetteer.load) so the next pick's existence-gated
    build/backfill regenerates it instead of short-circuiting on the
    still-present file forever."""
    path = profile_path(media_id, cache_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SeriesProfile(
            media_id=int(data["media_id"]),
            genres=[str(g) for g in data["genres"]],
            tags=[Tag(str(t["name"]), int(t["rank"]), str(t["category"]))
                  for t in data["tags"]],
            synopsis_first_sentence=str(data["synopsis_first_sentence"]),
            relations=[Relation(r["relation_type"], r["media_id"], r["title"])
                       for r in data["relations"]],
        )
    except (OSError, ValueError, TypeError, KeyError):
        _quarantine(path)
        return None


_HINT_MAX_CHARS = 120
_HINT_TAG_MIN_RANK = 60
_HINT_TAG_LIMIT = 5
# Domain-signal categories only: setting, themes, cast traits. Rank
# alone is NOT enough (Cast-Main Cast / Demographic / Technical tags
# rank high but say nothing a translator can use).
_HINT_CATEGORY_PREFIXES = ("Setting-", "Theme-", "Cast-Traits")
# ALWAYS excluded, independent of the prefix list above -- a Sexual
# Content tag must never reach a prompt hint even if the prefix list
# ever grows to cover it.
_HINT_CATEGORY_BANNED = "Sexual Content"


def context_hint(profile: SeriesProfile) -> str:
    """One prompt-ready line: "<tag1>, <tag2>; <synopsis first sentence>".

    Tags: rank >= 60, category in the domain-signal prefixes, Sexual
    Content always excluded, max 5, rank-descending. Hard cap 120 chars,
    truncated at a word boundary (no ellipsis). "" when no qualifying
    tags and no synopsis -- callers then send no series line at all.
    """
    picked = sorted(
        (t for t in profile.tags
         if t.rank >= _HINT_TAG_MIN_RANK
         and t.category != _HINT_CATEGORY_BANNED
         and t.category.startswith(_HINT_CATEGORY_PREFIXES)),
        key=lambda t: t.rank, reverse=True,
    )[:_HINT_TAG_LIMIT]
    parts = []
    if picked:
        parts.append(", ".join(t.name for t in picked))
    if profile.synopsis_first_sentence:
        parts.append(profile.synopsis_first_sentence)
    hint = "; ".join(parts)
    if len(hint) <= _HINT_MAX_CHARS:
        return hint
    clipped = hint[:_HINT_MAX_CHARS]
    if not hint[_HINT_MAX_CHARS].isspace() and " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip()
