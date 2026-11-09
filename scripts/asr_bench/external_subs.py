"""Japanese subtitles from a local archive or public sources, aligned to episodes without a Japanese track.

    python -m scripts.asr_bench.external_subs build --inventory <inventory json> --out <folder>
        [--archive <folder of zips>] [--no-network] [--title KEY ...]
        [--episodes-per-title N [--windows-per-episode K]] [--controls]
    python -m scripts.asr_bench.external_subs spot --video <episode file name> [--root <folder>] [--count 3]

``build`` takes the inventory written by ``scripts.asr_bench.inventory``. It
handles every inventory title (or the ``--title`` ones) with episodes that
have Japanese audio, no embedded Japanese dialogue track, an episode number,
room for the windows, an English dialogue track and a readable file not
found incomplete; every other episode is listed as skipped with the reason.
Per episode it tries its subtitle files in order and keeps the first that
passes the Japanese text check and the acceptance rule:

- Archive (``--archive``, first): a local folder of zip files, one per
  series (a jimaku.cc dump). ``index_archive`` lists each zip's subtitle
  files (SRT, ASS, SSA) with the season and episode numbers of their names;
  nested archives are skipped and listed, as are names tagged with another
  language, names that span several episodes, and, in a zip whose name
  says ``Part 2`` or ``2nd Cour``, files numbered without a season (numbered
  within the part). Zips and members whose names hold a locked title are
  skipped and only counted. A zip serves an inventory title when one of its
  names (``title_key`` of the zip name without its jimaku id, or a title
  shared by at least 3 or a fifth of its subtitle files) equals one of the
  title's names (the title, the file-name title of at least half its
  files, and parenthesized alternatives in the folder names that hold the
  title, such as ``(Attack on Titan)``); when those zips hold no file for an
  episode, the zips one of whose names starts with a title name of two or
  more words are asked (the full romaji title). ``archive_candidates`` then
  picks the files for one episode (season s, missing taken as 1, episode e;
  k the sum of the highest episode numbers of the title's earlier local
  seasons): files whose own name says season s, numbered e, e + k (absolute
  numbering) or, in a zip not split in parts, e + m - 1 where m is the
  first episode number of season s there; else files without a season in a
  zip whose name, or whose other files, say season s, numbered e or e + k;
  else files without a season in a zip without one, numbered e for season 1
  or e + k. Only the strongest of these tiers is used; when it is one of
  the two weaker tiers and its files come from more than one zip, the
  episode is ambiguous: listed, and the archive is not used for it. Netflix
  captions come first, then other SRT files, then ASS; within each, files
  numbered e, then e + k, then e + m - 1; at most ``ARCHIVE_MAX_FILES``
  distinct files per episode.
- Network (only when the archive has no file for the episode, or it is
  ambiguous, and without ``--no-network``): the titles and seasons listed in
  ``TITLES``, from the kitsunekko.net Japanese directory and, for a title
  kitsunekko.net lacks, the public kitsunekko-mirror Git repository (which
  also mirrors jimaku.cc). Every request waits at least
  ``REQUEST_INTERVAL_S`` after the previous one, and only files for episodes
  on disk are fetched. Downloads are cached under ``SOURCE_CACHE``, shared
  by every output folder.
- Japanese text check (``JAPANESE_TEXT``, before alignment): the dialogue
  cues must be Japanese, not bilingual or another language: most hold kana,
  few hold Latin words or characters used only in Chinese.
- Timing reference: the episode's own English dialogue subtitle track,
  without signs, songs, karaoke, titles and positioned events (styles,
  override tags, and events repeated in an English signs or songs track).
  Alignment is subtitle-to-subtitle only: no audio is read and no VAD or
  speech model is run.
- Alignment (``ALIGNMENT``): a global offset from the cross-correlation of
  the two cue-activity signals, then a dynamic program that gives each
  Japanese dialogue cue an offset (keeping cue order) or drops it, paying a
  penalty per offset change or drop, maximizing overlap with English
  dialogue time; each segment's offset is refined to 10 ms. Dropped cues
  between two segments with the same offset are kept at that offset;
  dropped cues at the head, at the tail, or between different offsets are
  left out. Cues without dialogue (sound descriptions, lyrics) follow their
  nearest dialogue cue. An ASS file's dialogue events are read the way the
  English track's are.
- Acceptance (``ACCEPTANCE``): at most two offset groups of at least 50
  cues each, kept Japanese dialogue time at least 75% on English dialogue
  time, English dialogue time at least 55% covered, and a sharp timing peak
  (the best overlap of the kept cues shifted by 2 to 60 s is at most 0.91
  of the aligned overlap). ``--controls`` aligns each episode against a
  different episode's Japanese file to check that the rule rejects it.
- Completeness: an accepted episode the inventory did not decode must pass
  ``inventory.audio_completeness`` (no unwritten blocks, a full decode of
  the Japanese audio without an error line); one that fails is rejected.

With ``--episodes-per-title N`` only the episodes the builder can take are
tried: per title, ``selection.walk`` over the episodes with at least one
subtitle file, sorted by season and episode, until N are accepted or
``3 * N`` have been tried (an episode repeating the season and episode
number of one accepted is passed over). Without it every episode is tried.

Accepted episodes are written as UTF-8 SRT to ``<out>/<title>/<episode file
stem>.ja.srt`` (cue text as published, identical continuation cues merged,
times shifted and clipped to the video), and every aligned or skipped
episode is listed in ``<out>/manifest.json``, with ``<out>/archive-index.json``
for the archive. ``--out`` must name a new folder or one this command wrote:
an output a built set records must not change. Locked titles are refused by
the private list (``titles.locked_titles_path()``), so the E0 private root
must be set, and the inventory must have been built under the same list.

``spot`` prints evenly spaced Japanese cues of one accepted episode next to
the English cues they overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.e0_evidence.normalize_ja import normalize_minimal
from scripts.e0_evidence.subtitle_proposals import _parse_srt

from .build_silver import STATE_FILE, clean_cue_text, e0_boundary, load_fixtures, locked_titles
from .inventory import audio_completeness, tool_version
from .selection import DEFAULT_WINDOWS, sort_key, walk, windows
from .sets import bench_root
from .titles import LockedTitles, season_episode, title_key

MANIFEST_SCHEMA = "asr-bench-external-subs-1.1"
READABLE_MANIFEST_SCHEMAS = ("asr-bench-external-subs-1.0", MANIFEST_SCHEMA)
INVENTORY_SCHEMAS = ("asr-bench-inventory-1.0", "asr-bench-inventory-1.1", "asr-bench-inventory-1.2")
REQUEST_INTERVAL_S = 1.2
USER_AGENT = "yamabiko-asr-bench/1.0 (private research; one request per second or slower)"
KITSUNEKKO = "https://kitsunekko.net"
MIRROR_REPO = "Ajatt-Tools/kitsunekko-mirror"
TEXT_CODECS = ("ass", "ssa", "subrip", "webvtt", "mov_text", "text")
FRAME_S = 0.01
LYRIC_MARKS = "♪♫"

ALIGNMENT = {
    "method": "subtitle-to-subtitle piecewise offset fit against the English dialogue track",
    "frame_s": FRAME_S,
    "global_search_s": 1500.0,
    "segment_search_s": 300.0,
    "segment_step_s": 0.1,
    "refine_step_s": FRAME_S,
    "switch_penalty_s": 10.0,
    "outside_weight": 0.5,
    "reattach_tolerance_s": 0.25,
    "peak_span_s": 60.0,
    "peak_exclusion_s": 2.0,
}
ALIGNMENT_TEXT = (
    "Japanese dialogue cues (text left after markup, parenthesized annotations and "
    "normalize_minimal, no lyric mark) are aligned to the union of English dialogue "
    "cue spans on a frame_s grid. Global offset: argmax of the cross-correlation of "
    "both activity signals within +-global_search_s. Then a dynamic program over the "
    "cues assigns each one an offset within +-segment_search_s of the global offset "
    "(step segment_step_s) or drops it; a cue scores overlap - outside_weight * "
    "(duration - overlap) seconds, a dropped cue 0, and every offset change or drop "
    "costs switch_penalty_s; an offset may decrease only by the silence before the "
    "cue, so cues keep their order. Each segment's offset is refined on the frame "
    "grid within +-segment_step_s. Dropped runs between segments whose offsets agree "
    "within reattach_tolerance_s take that offset; other dropped runs are left out."
)
ACCEPTANCE = {
    "max_offset_groups": 2,
    "min_group_cues": 50,
    "min_anchor_on_en": 0.75,
    "min_en_covered": 0.55,
    "max_peak_ratio": 0.91,
}
ACCEPTANCE_TEXT = (
    "offset_groups (segment offsets within reattach_tolerance_s form one group) at "
    "most max_offset_groups, every group at least min_group_cues dialogue cues; "
    "anchor_on_en (share of the kept cues' time on English dialogue time) at least "
    "min_anchor_on_en; en_covered (share of English dialogue time covered by the "
    "written Japanese dialogue cues) at least min_en_covered; peak_ratio (largest "
    "overlap of the kept cues shifted rigidly by peak_exclusion_s to peak_span_s, "
    "over their overlap at the aligned position) at most max_peak_ratio"
)
_STYLE_DROP_RE = re.compile(
    r"sign|song|opening|ending|\bop\b|^op|\bed\b|karaoke|kfx|title|credit|note|lyric"
    r"|romaji|kanji|insert|next ?ep|logo|trad|\btl\b",
    re.IGNORECASE,
)
_TAG_DROP_RE = re.compile(r"\\(?:pos|move|org|fn|p[1-9]|[kK][fo]?\d)")
_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
_EN_SIGNS_RE = re.compile(r"sign|song", re.IGNORECASE)
_EN_NOT_DIALOGUE_RE = re.compile(r"sign|song|forced|\bcc\b|sdh|commentary", re.IGNORECASE)
_KITSUNEKKO_LINK_RE = re.compile(r'<a href="(subtitles/[^"]+)"[^>]*>\s*<strong>([^<]*)</strong>')


@dataclass(frozen=True)
class Source:
    """Where the subtitle file of a source episode is found: a directory of one
    site, a file-name pattern whose ``ep`` group is the source episode number,
    and how a local episode number maps to it."""

    site: str  # "kitsunekko" or "kitsunekko-mirror"
    directory: str
    pattern: str
    kind: str
    episode_add: int = 0
    halves: bool = False  # local files are halves: file m holds source episode ceil(m / 2)

    def source_episode(self, episode: int) -> int:
        return (episode + 1) // 2 + self.episode_add if self.halves else episode + self.episode_add

    def describe(self) -> dict:
        mapping = "ceil(local / 2)" if self.halves else "local"
        if self.episode_add:
            mapping += f" + {self.episode_add}"
        return {"site": self.site, "directory": self.directory, "pattern": self.pattern,
                "kind": self.kind, "source_episode": mapping}


@dataclass(frozen=True)
class TitleSources:
    season: int
    sources: tuple[Source, ...]


# kitsunekko.net comes first; the mirror is used only for titles kitsunekko.net
# lacks. Within a title, sources are tried in order per episode.
_JJK = "Jujutsu Kaisen 2nd Season"
_MT = "Mushoku Tensei: Isekai Ittara Honki Dasu"
_FRIEREN = "subtitles/anime_tv/Sousou no Frieren 2nd Season"
TITLES: dict[str, TitleSources] = {
    "jujutsu kaisen": TitleSources(2, (
        Source("kitsunekko", _JJK, r"^Jujutsu\.Kaisen\.S02E(?P<ep>\d+)\.WEBRip\.Netflix\.ja\[cc\]\.srt$",
               "netflix-ja-cc", 24),
        Source("kitsunekko", _JJK, r"^Jujutsu Kaisen \(2023\) - (?P<ep>\d+) .*\(TBS [^)]*\).*\.srt$",
               "tv-captions", 24),
    )),
    "mushoku tensei": TitleSources(1, (
        Source("kitsunekko", _MT,
               r"^Mushoku\.Tensei \.Jobless\.Reincarnation\.S01E(?P<ep>\d+)\..*WEBRip\.Netflix"
               r"(?:\.ja\[cc\])?\.srt(?:\.srt)?$", "netflix-ja-cc"),
        Source("kitsunekko", _MT, r"^Mushoku Tensei - Isekai Ittara Honki Dasu - (?P<ep>\d+) "
               r"Funimation版同期調整\.srt$", "netflix-ja-cc-retimed"),
    )),
    # kitsunekko.net has the first season only.
    "sousou no frieren": TitleSources(2, (
        Source("kitsunekko-mirror", _FRIEREN, r"\.S02E(?P<ep>\d+)\..*\.WEBRip\.Netflix\.ja\[cc\]\.srt$",
               "netflix-ja-cc", 28),
        Source("kitsunekko-mirror", _FRIEREN, r"\.S02E(?P<ep>\d+)\..*\.WEBRip\.Amazon\.ja-jp\[sdh\]\.srt$",
               "amazon-ja-sdh"),
    )),
    # The twelve mini-episode files are the halves of TV episodes 1 to 6.
    "smoking behind the supermarket with you": TitleSources(1, (
        Source("kitsunekko", "Super no Ura de Yani S Futari", r"^\.S01E(?P<ep>\d+)\..*\.ja\.srt$",
               "netflix-ja-cc", halves=True),
    )),
    "turn a gundam": TitleSources(1, (
        Source("kitsunekko", "Turn A Gundam",
               r"^\[EG\]Turn-A Gundam BD (?P<ep>\d+)(?: V\d)?\(1080p 10bit\)\[[0-9A-F]{8}\](?:\.jp)?\.srt$",
               "bandai-ja-retimed-bd"),
    )),
}


def root() -> Path:
    """The external-subtitle folder silver-v3 and silver-v4 were built from;
    later outputs go to the folder ``build --out`` names."""
    return bench_root() / "external-subs"


def source_cache() -> Path:
    """Network downloads, shared by every output folder. Files are only
    added, never rewritten."""
    return root() / "sources"


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


JAPANESE_TEXT = {"min_kana_share": 0.8, "max_latin_share": 0.2, "max_chinese_share": 0.05}
JAPANESE_TEXT_TEXT = (
    "over the dialogue cues: the share holding hiragana or katakana at least "
    "min_kana_share, the share holding a run of four Latin letters at most "
    "max_latin_share, the share holding a character used only in Chinese "
    "(们这说么吗呢吧你个) at most max_chinese_share"
)
_KANA_RE = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u30fc]")
_LATIN_RE = re.compile(r"[A-Za-z]{4,}")
_CHINESE_ONLY_RE = re.compile(r"[们这说么吗呢吧你个]")

ARCHIVE_INDEX_SCHEMA = "asr-bench-archive-index-1.0"
ARCHIVE_MAX_FILES = 3
SUBTITLE_SUFFIXES = (".srt", ".ass", ".ssa")
NESTED_SUFFIXES = (".zip", ".rar", ".7z")
MIN_SHARED_FILES = 3
_ZIP_ID_RE = re.compile(r"\s*\[\d+\]\s*$|^\d+_")
_LANGUAGE_TAIL_RE = re.compile(
    r"(?i)(?:[.\s_-]+(?:ja|jp|jpn|jap|japanese|ja-jp|jp-jp)(?:\[[^\]]*\])?|[.\s_]+track\d+)+$"
)
_OTHER_LANGUAGE_RE = re.compile(
    r"(?i)(?:^|[.\s_(\[,-])(?:chs|cht|chi|zho?|zh|sc|tc|gb|big5|en|eng|english|chinese|简|繁|中文|英)(?:$|[.\s_)\],-])"
)
_MULTI_EPISODE_RE = re.compile(r"(?<!\w)\d{1,4}(?:-|~|\s~\s)\d{1,4}(?!\w)")
_PART_RE = re.compile(r"(?i)\b(?:part|cour)\s*[2-9]\b|\b[2-9](?:nd|rd|th)\s+cour\b")
_PART_TAIL_RE = re.compile(r"(?i)\s+(?:part|cour)\s*\d+\s*$|\s+\d+(?:st|nd|rd|th)\s+cour\s*$")
_ZIP_SEASON_RE = re.compile(
    r"(?i)\b(\d{1,2})(?:st|nd|rd|th)\s+season\b|\bseason\s*(\d{1,2})\b|\bS(\d{1,2})\b|\s(\d{1,2})\s*$"
)
_EPISODE_PATTERNS = (
    re.compile(r"第\s*(\d{1,4})\s*[話回]"),
    re.compile(r"\s-\s*(\d{1,4})(?:v\d+)?(?![\d.])"),
    re.compile(r"(?i)(?<![a-z])EP?\s?(\d{1,4})(?:v\d+)?(?![\d.])"),
    re.compile(r"#(\d{1,4})(?![\d.])"),
    re.compile(r"(?<![\d.])(\d{1,4})(?:\s*v\d+)?\s*$"),
)
_NUMBER_BRACKET_RE = re.compile(r"\[(\d{1,4})\]")
_BRACKETS_RE = re.compile(r"\[[^\[\]]*\]|\([^()]*\)|\{[^{}]*\}|【[^【】]*】|「[^「」]*」")
_PARENTHESIZED_RE = re.compile(r"\(([^()]*)\)")
_NETFLIX_RE = re.compile(r"(?i)\bWEBRip\.Netflix\b|Netflix\.ja\[cc\]")


def japanese_text(cues: Sequence[tuple[float, float, str]]) -> dict:
    """Shares of dialogue cues holding kana, Latin words and Chinese-only
    characters (see JAPANESE_TEXT_TEXT)."""
    texts = [clean_cue_text(text) for _, _, text in cues if is_dialogue(text)]
    count = max(len(texts), 1)
    return {
        "dialogue_cues": len(texts),
        "kana_share": round(sum(bool(_KANA_RE.search(t)) for t in texts) / count, 4),
        "latin_share": round(sum(bool(_LATIN_RE.search(t)) for t in texts) / count, 4),
        "chinese_share": round(sum(bool(_CHINESE_ONLY_RE.search(t)) for t in texts) / count, 4),
    }


def japanese_text_problem(stats: Mapping) -> str | None:
    rule = JAPANESE_TEXT
    if not stats["dialogue_cues"]:
        return "no dialogue cues left (every event is non-dialogue, positioned or markup)"
    if stats["kana_share"] < rule["min_kana_share"]:
        return f"not Japanese text: kana in {stats['kana_share']} of the cues < {rule['min_kana_share']}"
    if stats["latin_share"] > rule["max_latin_share"]:
        return f"not Japanese text: Latin words in {stats['latin_share']} of the cues > {rule['max_latin_share']}"
    if stats["chinese_share"] > rule["max_chinese_share"]:
        return (f"not Japanese text: Chinese-only characters in {stats['chinese_share']} of the cues > "
                f"{rule['max_chinese_share']}")
    return None


def _stem(member: str) -> str:
    name = member.rsplit("/", 1)[-1]
    for suffix in SUBTITLE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _cleaned(stem: str) -> str:
    """Brackets removed, underscores and dots (but decimal points) turned
    into spaces, NFKC."""
    while (stripped := _BRACKETS_RE.sub(" ", stem)) != stem:
        stem = stripped
    stem = re.sub(r"_+|(?<!\d)\.|\.(?!\d)", " ", stem)
    return " ".join(unicodedata.normalize("NFKC", stem).split())


def subtitle_numbers(member: str) -> tuple[int | None, int | None, str | None]:
    """Season (when the name states it), episode, and why the file cannot
    serve one episode (None when it can)."""
    stem = _LANGUAGE_TAIL_RE.sub("", _stem(member))
    if _OTHER_LANGUAGE_RE.search(stem):
        return None, None, "name tagged with another language"
    cleaned = _cleaned(stem)
    if _MULTI_EPISODE_RE.search(cleaned):
        return None, None, "name spans several episodes"
    season, episode = season_episode(stem)
    if season is not None:
        return season, episode, None
    if _PART_RE.search(cleaned):
        return None, None, "numbered within a part"
    for pattern in _EPISODE_PATTERNS:
        if match := pattern.search(cleaned):
            return None, int(match[1]), None
    if match := _NUMBER_BRACKET_RE.search(stem):
        return None, int(match[1]), None
    return None, None, "no episode number"


def _kind(member: str) -> int:
    """0 Netflix captions, 1 other SRT, 2 ASS or SSA."""
    if member.lower().endswith(".srt"):
        return 0 if _NETFLIX_RE.search(member) else 1
    return 2


def zip_series(zip_name: str) -> str:
    """The series name of an archive zip: file name without ``.zip`` and its
    jimaku id (``Title [12345]``, ``12345_Title``)."""
    return _ZIP_ID_RE.sub("", zip_name[: -len(".zip")] if zip_name.lower().endswith(".zip") else zip_name).strip()


def zip_name_season(series: str) -> int | None:
    """The season a series name states (``2nd Season``, ``Season 2``, ``S2``,
    a trailing ``Title 2``), ignoring a trailing ``Part N`` or ``Cour N``."""
    name = _PART_TAIL_RE.sub("", " ".join(unicodedata.normalize("NFKC", series).replace("_", " ").split()))
    if match := _ZIP_SEASON_RE.search(name):
        return int(next(group for group in match.groups() if group))
    return None


def _zip_of(zip_name: str, members: Sequence[zipfile.ZipInfo], locked: LockedTitles) -> tuple[dict, int]:
    """One zip's index entry and the number of members skipped as locked."""
    series = zip_series(zip_name)
    subtitles, skipped, keys, locked_members = [], [], Counter(), 0
    for info in members:
        if info.is_dir():
            continue
        if locked.path_title(Path(info.filename)) is not None:
            locked_members += 1
            continue
        lower = info.filename.lower()
        if lower.endswith(NESTED_SUFFIXES):
            skipped.append({"member": info.filename, "reason": "nested archive"})
            continue
        if not lower.endswith(SUBTITLE_SUFFIXES):
            continue
        season, episode, problem = subtitle_numbers(info.filename)
        if (key := title_key(_LANGUAGE_TAIL_RE.sub("", _stem(info.filename)))) is not None:
            keys[key] += 1
        if problem:
            skipped.append({"member": info.filename, "reason": problem})
            continue
        subtitles.append({"member": info.filename, "size": info.file_size, "crc32": info.CRC,
                          "season": season, "episode": episode, "kind": _kind(info.filename)})
    part = bool(_PART_RE.search(series))
    if part:
        for entry in [s for s in subtitles if s["season"] is None]:
            subtitles.remove(entry)
            skipped.append({"member": entry["member"], "reason": "numbered within a part"})
    season, basis = zip_name_season(series), "name"
    if season is None:
        stated = {s["season"] for s in subtitles if s["season"] is not None}
        season, basis = (stated.pop(), "files") if len(stated) == 1 else (None, None)
    total = max(len(subtitles) + len(skipped), 1)
    names = {title_key(series)} | {k for k, n in keys.items() if n >= MIN_SHARED_FILES or n >= 0.2 * total}
    return {
        "zip": zip_name,
        "series": series,
        "names": sorted(n for n in names if n),
        "season": season,
        "season_basis": basis,
        "part": part,
        "subtitles": subtitles,
        "skipped": skipped,
    }, locked_members


def index_archive(folder: Path, locked: LockedTitles) -> dict:
    """Every zip directly in ``folder`` whose name holds no locked title,
    with its subtitle files and skipped members; locked zips and members
    are counted, never named."""
    zips, locked_zips, locked_members, unreadable = [], 0, 0, []
    for path in sorted(Path(folder).glob("*.zip")):
        series = zip_series(path.name)
        if locked.path_title(path) is not None or locked.title_of(title_key(series)) is not None:
            locked_zips += 1
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                entry, locked_count = _zip_of(path.name, archive.infolist(), locked)
        except (OSError, zipfile.BadZipFile) as error:
            unreadable.append({"zip": path.name, "error": str(error)})
            continue
        stat = path.stat()
        entry.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        zips.append(entry)
        locked_members += locked_count
    reasons = Counter(skip["reason"] for z in zips for skip in z["skipped"])
    return {
        "schema_version": ARCHIVE_INDEX_SCHEMA,
        "folder": str(folder),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "zips_indexed": len(zips),
            "locked_zips_skipped": locked_zips,
            "locked_members_skipped": locked_members,
            "unreadable_zips": len(unreadable),
            "subtitle_files": sum(len(z["subtitles"]) for z in zips),
            "skipped_members": dict(sorted(reasons.items())),
        },
        "unreadable": unreadable,
        "zips": zips,
    }


def title_names(title: str, records: Sequence[Mapping]) -> set[str]:
    """An inventory title's names for archive matching: the title, the
    file-name title of at least half its files, and the parenthesized
    alternatives in its folder names whose title holds the inventory title
    (``(Attack on Titan)`` in ``Shingeki no Kyojin (Attack on Titan) (Season
    1)``)."""
    names = {title}
    file_keys = Counter(r["title_keys"]["file"] for r in records if r["title_keys"]["file"])
    names |= {key for key, count in file_keys.items() if 2 * count >= len(records)}
    words = title.split()
    for record in records:
        for folder in record["folders"]:
            key = (title_key(folder) or "").split()
            if any(key[i : i + len(words)] == words for i in range(len(key) - len(words) + 1)):
                names |= {title_key(group) for group in _PARENTHESIZED_RE.findall(folder)}
    return {name for name in names if name and len(name) >= 3}


def match_series(names: set[str], zips: Sequence[Mapping]) -> tuple[list[Mapping], list[Mapping]]:
    """The zips one of whose names equals a title name, and, apart from
    those, the zips one of whose names starts with a title name of two or
    more words followed by more words (the full romaji title, ``Mushoku
    Tensei: Isekai Ittara Honki Dasu`` for ``mushoku tensei``)."""
    exact = [z for z in zips if names & set(z["names"])]
    leads = tuple(f"{name} " for name in names if " " in name)
    taken = {id(z) for z in exact}
    prefix = [z for z in zips if id(z) not in taken and leads and any(n.startswith(leads) for n in z["names"])]
    return exact, prefix


def season_offsets(records: Sequence[Mapping]) -> dict[int, int]:
    """Per local season (missing taken as 1), the sum of the highest episode
    numbers of the earlier seasons: the shift to absolute numbering."""
    highest: dict[int, int] = {}
    for record in records:
        if record["episode"] is not None:
            season = record["season"] or 1
            highest[season] = max(highest.get(season, 0), record["episode"])
    return {season: sum(n for s, n in highest.items() if s < season) for season in highest}


NUMBERINGS = ("episode", "absolute", "season start")


def archive_candidates(record: Mapping, offset: int, series: Sequence[Mapping]) -> tuple[list[dict], dict]:
    """The archive files for one episode, best first, and how they were
    found (see the module docstring); none when the match is ambiguous."""
    season, episode = record["season"] or 1, record["episode"]
    tiers: dict[str, list[dict]] = {"file season": [], "zip season": [], "no season": []}
    for entry in series:
        numbers = {episode + offset: "absolute"} if offset else {}
        stated = [f["episode"] for f in entry["subtitles"] if f["season"] == season]
        if stated and min(stated) > 1 and not entry["part"]:
            numbers.setdefault(episode + min(stated) - 1, "season start")
        numbers[episode] = "episode"
        for file in entry["subtitles"]:
            numbering = numbers.get(file["episode"])
            if numbering is None:
                continue
            if file["season"] is not None:
                tier = "file season" if file["season"] == season else None
            elif numbering == "season start":
                tier = None
            elif entry["season"] is not None:
                tier = "zip season" if entry["season"] == season else None
            else:
                tier = "no season" if numbering == "absolute" or season == 1 else None
            if tier:
                tiers[tier].append({**file, "zip": entry["zip"], "tier": tier, "numbering": numbering})
    for tier, files in tiers.items():
        if not files:
            continue
        zips = sorted({file["zip"] for file in files})
        if tier != "file season" and len(zips) > 1:
            return [], {"tier": tier, "zips": zips, "ambiguous": True}
        files.sort(key=lambda f: (f["kind"], NUMBERINGS.index(f["numbering"]), len(f["member"]), f["member"]))
        distinct, seen = [], set()
        for file in files:
            if (file["size"], file["crc32"]) not in seen:
                seen.add((file["size"], file["crc32"]))
                distinct.append(file)
        return distinct[:ARCHIVE_MAX_FILES], {"tier": tier, "zips": zips, "ambiguous": False}
    return [], {"tier": None, "zips": [], "ambiguous": False}


def read_member(folder: Path, zip_name: str, member: str) -> bytes:
    """A member's bytes; ``zipfile`` checks its CRC."""
    with zipfile.ZipFile(Path(folder) / zip_name) as archive:
        return archive.read(member)


class Fetcher:
    """HTTP GET with one global minimum interval between requests."""

    def __init__(self) -> None:
        self._last = 0.0
        self.requests = 0
        self._mirror_commit: str | None = None
        self._listings: dict[tuple[str, str], dict[str, dict]] = {}

    def get(self, url: str, accept: str | None = None) -> bytes:
        wait = REQUEST_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        headers = {"User-Agent": USER_AGENT}
        if accept:
            headers["Accept"] = accept
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                return response.read()
        finally:
            self._last = time.monotonic()
            self.requests += 1

    def mirror_commit(self) -> str:
        if self._mirror_commit is None:
            raw = self.get(f"https://api.github.com/repos/{MIRROR_REPO}/commits/main",
                           accept="application/vnd.github.sha")
            self._mirror_commit = raw.decode("ascii").strip()
        return self._mirror_commit

    def listing(self, site: str, directory: str) -> dict[str, dict]:
        """File name -> {"url", "git_blob_sha"} for one source directory. When
        the site cannot be reached, the files cached from earlier runs."""
        key = (site, directory)
        if key not in self._listings:
            try:
                self._listings[key] = self._fetch_listing(site, directory)
            except OSError as exc:
                cache = source_cache() / site / _safe_name(directory)
                if not cache.is_dir():
                    raise
                print(f"listing {site} {directory!r} failed ({exc}); using the cached files", flush=True)
                self._listings[key] = {
                    meta.name[: -len(".json")]: {**json.loads(meta.read_text(encoding="utf-8")), "git_blob_sha": None}
                    for meta in cache.glob("*.json")
                }
        return self._listings[key]

    def _fetch_listing(self, site: str, directory: str) -> dict[str, dict]:
        files: dict[str, dict] = {}
        if site == "kitsunekko":
            query = urllib.parse.quote_plus(f"subtitles/japanese/{directory}/")
            page = self.get(f"{KITSUNEKKO}/dirlist.php?dir={query}").decode("utf-8", "replace")
            for href, name in _KITSUNEKKO_LINK_RE.findall(page):
                path = urllib.parse.quote(html.unescape(href), safe="/%")
                files[html.unescape(name).strip()] = {"url": f"{KITSUNEKKO}/{path}", "git_blob_sha": None}
        elif site == "kitsunekko-mirror":
            commit = self.mirror_commit()
            api = (f"https://api.github.com/repos/{MIRROR_REPO}/contents/"
                   f"{urllib.parse.quote(directory)}?ref={commit}")
            for entry in json.loads(self.get(api, accept="application/vnd.github+json")):
                if entry.get("type") == "file":
                    raw = (f"https://raw.githubusercontent.com/{MIRROR_REPO}/{commit}/"
                           f"{urllib.parse.quote(entry['path'])}")
                    files[entry["name"]] = {"url": raw, "git_blob_sha": entry["sha"]}
        else:
            raise ValueError(f"unknown source site {site!r}")
        return files

    def fetch(self, source: Source, name: str) -> tuple[bytes, dict]:
        """The file's bytes and provenance, from the cache when present."""
        cached = source_cache() / source.site / _safe_name(source.directory) / _safe_name(name)
        meta_path = cached.with_name(cached.name + ".json")
        if cached.exists() and meta_path.exists():
            return cached.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))
        entry = self.listing(source.site, source.directory)[name]
        data = self.get(entry["url"])
        if entry["git_blob_sha"] is not None:
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            if blob != entry["git_blob_sha"]:
                raise RuntimeError(f"{entry['url']}: git blob {blob} does not match {entry['git_blob_sha']}")
        meta = {
            "url": entry["url"],
            "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return data, meta


# Cues are (start_s, end_s, text) tuples.

def _ass_time(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _ass_events(text: str) -> list[dict]:
    section, fields, events = "", None, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.lower()
        elif section == "[events]" and stripped.lower().startswith("format:"):
            fields = [field.strip().lower() for field in stripped[7:].split(",")]
        elif section == "[events]" and fields and stripped.startswith(("Dialogue:", "Comment:")):
            kind, rest = stripped.split(":", 1)
            values = rest.split(",", len(fields) - 1)
            if len(values) == len(fields):
                event = {name: value.strip() for name, value in zip(fields[:-1], values[:-1])}
                event["text"], event["kind"] = values[-1], kind
                events.append(event)
    return events


def english_cues(path: Path, signs_path: Path | None) -> list[tuple[float, float, str]]:
    """English dialogue cues of an extracted track: no comment events, no
    events whose style names signs, songs, titles or credits, no positioned,
    karaoke or drawing events, none repeated in the signs or songs track."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix == ".srt":
        return sorted(
            (cue.original_start_ms / 1000, cue.original_end_ms / 1000, _OVERRIDE_RE.sub("", cue.raw_text).strip())
            for cue in _parse_srt(raw.encode("utf-8"))
            if not _TAG_DROP_RE.search(cue.raw_text) and _OVERRIDE_RE.sub("", cue.raw_text).strip()
        )
    repeated = set()
    if signs_path is not None:
        repeated = {(e["start"], e["end"], e["text"])
                    for e in _ass_events(signs_path.read_text(encoding="utf-8-sig", errors="replace"))}
    cues = []
    for event in _ass_events(raw):
        if event["kind"] != "Dialogue" or _STYLE_DROP_RE.search(event.get("style", "")):
            continue
        if _TAG_DROP_RE.search(event["text"]) or (event["start"], event["end"], event["text"]) in repeated:
            continue
        text = _OVERRIDE_RE.sub("", event["text"]).replace("\\N", " ").replace("\\n", " ").strip()
        start, end = _ass_time(event["start"]), _ass_time(event["end"])
        if text and end > start:
            cues.append((start, end, text))
    return sorted(cues)


def decode_subtitle(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp932")


_JA_TAG_DROP_RE = re.compile(r"\\(?:pos|move|org|p[1-9]|[kK][fo]?\d)")


def japanese_cues(data: bytes, name: str = ".srt") -> list[tuple[float, float, str]]:
    """Cues of a Japanese subtitle file: SRT as published (trailing spaces
    on a line ignored); ASS or SSA dialogue events without comments, events
    in a style that names signs, songs, titles or credits, and positioned,
    karaoke or drawing events, line breaks kept."""
    text = decode_subtitle(data).replace("\r\n", "\n").replace("\r", "\n")
    if not name.lower().endswith((".ass", ".ssa")):
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return [(cue.original_start_ms / 1000, cue.original_end_ms / 1000, cue.raw_text)
                for cue in _parse_srt(text.encode("utf-8"))]
    cues = []
    for event in _ass_events(text):
        if event["kind"] != "Dialogue" or _STYLE_DROP_RE.search(event.get("style", "")):
            continue
        if _JA_TAG_DROP_RE.search(event["text"]):
            continue
        line = _OVERRIDE_RE.sub("", event["text"]).replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
        start, end = _ass_time(event["start"]), _ass_time(event["end"])
        if line.strip() and end > start:
            cues.append((start, end, line.strip()))
    return sorted(cues)


def is_dialogue(text: str) -> bool:
    return not any(mark in text for mark in LYRIC_MARKS) and bool(normalize_minimal(clean_cue_text(text)))


def _frames(seconds: float) -> int:
    return int(round(seconds / FRAME_S))


def _activity(cues: Sequence[tuple[float, float]], length: int, shift: int = 0) -> np.ndarray:
    active = np.zeros(length, np.int8)
    for start, end in cues:
        first, last = max(0, _frames(start) + shift), min(length, _frames(end) + shift)
        if last > first:
            active[first:last] = 1
    return active


def _prefix(active: np.ndarray) -> np.ndarray:
    prefix = np.zeros(active.size + 1)
    np.cumsum(active, out=prefix[1:])
    return prefix


def global_offset(jp: Sequence[tuple[float, float]], en: Sequence[tuple[float, float]]) -> float:
    reach = _frames(ALIGNMENT["global_search_s"])
    length = _frames(max(end for _, end in (*jp, *en))) + reach + 1
    size = 1 << (2 * length - 1).bit_length()
    spectrum = np.fft.rfft(_activity(en, length), size) * np.conj(np.fft.rfft(_activity(jp, length), size))
    correlation = np.fft.irfft(spectrum, size)
    lags = np.concatenate([np.arange(0, reach + 1), np.arange(-reach, 0)])
    values = np.concatenate([correlation[: reach + 1], correlation[size - reach:]])
    return float(lags[int(np.argmax(values))] * FRAME_S)


def _piecewise(starts: np.ndarray, ends: np.ndarray, prefix: np.ndarray, offsets: np.ndarray):
    """Per cue: offset index and whether it is dropped (see ALIGNMENT_TEXT)."""
    count, states = len(starts), len(offsets)
    step = offsets[1] - offsets[0]
    shifts = np.round(offsets / FRAME_S).astype(np.int64)
    limit = prefix.size - 1
    penalty, weight = ALIGNMENT["switch_penalty_s"], ALIGNMENT["outside_weight"]
    index = np.arange(states)

    def score(i: int) -> np.ndarray:
        first = np.clip(_frames(starts[i]) + shifts, 0, limit)
        last = np.clip(_frames(ends[i]) + shifts, 0, limit)
        overlap = (prefix[last] - prefix[first]) * FRAME_S
        return overlap - weight * ((ends[i] - starts[i]) - overlap)

    placed_kind = np.zeros((count, states), np.int8)  # 0 stay, 1 from placed, 2 from dropped, 3 from dropped elsewhere
    placed_from = np.zeros((count, states), np.int32)
    dropped_entered = np.zeros((count, states), bool)
    placed, dropped = score(0), np.zeros(states)
    last_end = ends[0]
    for i in range(1, count):
        gap = max(0.0, starts[i] - last_end)
        last_end = max(last_end, ends[i])
        bound = np.clip(index + int(math.floor(gap / step + 1e-9)), 0, states - 1)
        best_placed = np.maximum.accumulate(placed)
        arg_placed = np.maximum.accumulate(np.where(placed == best_placed, index, 0))
        best_dropped = np.maximum.accumulate(dropped)
        arg_dropped = np.maximum.accumulate(np.where(dropped == best_dropped, index, 0))
        options = np.stack([placed, best_placed[bound] - penalty, dropped, best_dropped[bound] - penalty])
        kind = np.argmax(options, axis=0)
        placed_kind[i] = kind
        placed_from[i] = np.where(kind == 1, arg_placed[bound], np.where(kind == 3, arg_dropped[bound], index))
        enter = placed - penalty
        dropped_entered[i] = enter > dropped
        placed, dropped = options[kind, index] + score(i), np.maximum(enter, dropped)
    in_drop = bool(dropped.max() > placed.max())
    state = int(np.argmax(dropped if in_drop else placed))
    path, is_dropped = np.zeros(count, np.int64), np.zeros(count, bool)
    for i in range(count - 1, -1, -1):
        path[i], is_dropped[i] = state, in_drop
        if i == 0:
            break
        if in_drop:
            in_drop = not dropped_entered[i, state]
        else:
            kind = placed_kind[i, state]
            if kind in (1, 3):
                state = int(placed_from[i, state])
            in_drop = kind in (2, 3)
    return path, is_dropped


def align(jp_cues: Sequence[tuple[float, float, str]], en_cues: Sequence[tuple[float, float, str]],
          duration_s: float) -> dict:
    """Offsets (None for dropped) and status for every Japanese cue."""
    dialogue = [i for i, (_, _, text) in enumerate(jp_cues) if is_dialogue(text)]
    if not dialogue:
        raise ValueError("no Japanese dialogue cues")
    starts = np.array([jp_cues[i][0] for i in dialogue])
    ends = np.array([jp_cues[i][1] for i in dialogue])
    en_spans = [(start, end) for start, end, _ in en_cues]
    centre = global_offset(list(zip(starts, ends)), en_spans)
    reach, step = ALIGNMENT["segment_search_s"], ALIGNMENT["segment_step_s"]
    length = _frames(max(duration_s, float(ends.max()) + abs(centre)) + reach) + 10
    prefix = _prefix(_activity(en_spans, length))
    offsets = centre + np.arange(-reach, reach + step / 2, step)
    path, is_dropped = _piecewise(starts, ends, prefix, offsets)

    runs: list[list] = []
    first = 0
    for i in range(1, len(dialogue) + 1):
        if i == len(dialogue) or (path[i], is_dropped[i]) != (path[first], is_dropped[first]):
            runs.append([first, i, None if is_dropped[first] else float(offsets[path[first]]), "kept"])
            first = i
    limit = prefix.size - 1
    for run in runs:
        start_index, end_index, offset, _ = run
        if offset is None:
            continue
        best = -1.0
        for fine in np.arange(-step, step + FRAME_S / 2, FRAME_S):
            shift = _frames(offset + fine)
            value = float(np.sum(
                prefix[np.clip(np.round(ends[start_index:end_index] / FRAME_S).astype(np.int64) + shift, 0, limit)]
                - prefix[np.clip(np.round(starts[start_index:end_index] / FRAME_S).astype(np.int64) + shift, 0, limit)]
            ))
            if value > best:
                best, run[2] = value, round(float(offset + fine), 3)
    for position, run in enumerate(runs):
        if run[2] is not None:
            continue
        before = next((r[2] for r in reversed(runs[:position]) if r[3] == "kept"), None)
        after = next((r[2] for r in runs[position + 1:] if r[3] == "kept"), None)
        if before is None:
            run[3] = "dropped_head"
        elif after is None:
            run[3] = "dropped_tail"
        elif abs(before - after) <= ALIGNMENT["reattach_tolerance_s"]:
            run[2], run[3] = before, "reattached"
        else:
            run[3] = "dropped_between"

    offset_of: list[float | None] = [None] * len(jp_cues)
    status_of: list[str] = [""] * len(jp_cues)
    for start_index, end_index, offset, status in runs:
        for k in range(start_index, end_index):
            offset_of[dialogue[k]], status_of[dialogue[k]] = offset, status
    dialogue_starts = starts.tolist()
    for i, (start, _, _) in enumerate(jp_cues):
        if status_of[i]:
            continue
        k = int(np.searchsorted(dialogue_starts, start, side="right")) - 1
        if k < 0 or (k + 1 < len(dialogue) and dialogue_starts[k + 1] - start < start - dialogue_starts[k]):
            k += 1
        offset_of[i], status_of[i] = offset_of[dialogue[k]], "follows_dialogue"
    return {
        "global_offset_s": round(centre, 3),
        "dialogue": dialogue,
        "offsets": offset_of,
        "status": status_of,
        "runs": [{"first_cue": dialogue[a], "last_cue": dialogue[b - 1], "dialogue_cues": b - a,
                  "offset_s": o, "status": s} for a, b, o, s in runs],
    }


def output_cues(jp_cues: Sequence[tuple[float, float, str]], aligned: Mapping, duration_s: float) -> list[tuple]:
    """Shifted cues inside the video, identical continuation cues merged."""
    shifted = []
    for (start, end, text), offset in zip(jp_cues, aligned["offsets"]):
        text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()
        if offset is None or not text:
            continue
        start, end = max(0.0, start + offset), min(duration_s, end + offset)
        if end - start >= 0.05:
            shifted.append([round(start, 3), round(end, 3), text])
    shifted.sort(key=lambda cue: (cue[0], cue[1]))
    merged: list[list] = []
    last_by_text: dict[str, list] = {}
    for cue in shifted:
        earlier = last_by_text.get(cue[2])
        if earlier is not None and abs(cue[0] - earlier[1]) <= 0.05:
            earlier[1] = max(earlier[1], cue[1])
            continue
        merged.append(cue)
        last_by_text[cue[2]] = cue
    return [tuple(cue) for cue in merged]


def quality(jp_cues: Sequence[tuple[float, float, str]], en_cues: Sequence[tuple[float, float, str]],
            aligned: Mapping, duration_s: float) -> dict:
    length = _frames(duration_s) + 1
    en_spans = [(start, end) for start, end, _ in en_cues]
    english = _activity(en_spans, length)

    def placed(statuses: tuple[str, ...]) -> list[tuple[float, float]]:
        return [(start + aligned["offsets"][i], end + aligned["offsets"][i])
                for i in aligned["dialogue"] for start, end, _ in [jp_cues[i]]
                if aligned["status"][i] in statuses]

    kept = placed(("kept",))
    written = placed(("kept", "reattached"))
    kept_active, written_active = _activity(kept, length), _activity(written, length)
    groups: list[list] = []
    for run in sorted((r for r in aligned["runs"] if r["status"] == "kept"), key=lambda r: r["offset_s"]):
        if groups and run["offset_s"] - groups[-1][-1]["offset_s"] <= ALIGNMENT["reattach_tolerance_s"]:
            groups[-1].append(run)
        else:
            groups.append([run])
    source_s = sum(jp_cues[i][1] - jp_cues[i][0] for i in aligned["dialogue"])
    statuses = Counter(aligned["status"][i] for i in aligned["dialogue"])
    onsets = []
    for start, end in kept:
        best = max(en_cues, key=lambda cue: min(end, cue[1]) - max(start, cue[0]))
        if min(end, best[1]) - max(start, best[0]) >= 0.5 * (end - start):
            onsets.append(abs(start - best[0]))
    return {
        "dialogue_cues": len(aligned["dialogue"]),
        "dialogue_cue_status": dict(sorted(statuses.items())),
        "offset_groups": [{"offset_s": round(statistics.median(r["offset_s"] for r in group), 3),
                           "dialogue_cues": sum(r["dialogue_cues"] for r in group)} for group in groups],
        "kept_share": round(sum(e - s for s, e in written) / source_s, 4) if source_s else 0.0,
        "en_dialogue_s": round(float(english.sum()) * FRAME_S, 2),
        "jp_dialogue_s": round(float(written_active.sum()) * FRAME_S, 2),
        "anchor_on_en": round(float((kept_active & english).sum()) / max(int(kept_active.sum()), 1), 4),
        "jp_on_en": round(float((written_active & english).sum()) / max(int(written_active.sum()), 1), 4),
        "en_covered": round(float((written_active & english).sum()) / max(int(english.sum()), 1), 4),
        "peak_ratio": round(peak_ratio(kept, en_spans, duration_s), 4),
        "onset_median_abs_s": round(statistics.median(onsets), 3) if onsets else None,
    }


def peak_ratio(kept: Sequence[tuple[float, float]], en_spans: Sequence[tuple[float, float]], duration_s: float) -> float:
    """Best overlap of the kept cues shifted rigidly by peak_exclusion_s to
    peak_span_s, over their overlap where they were aligned."""
    if not kept:
        return 1.0
    span = ALIGNMENT["peak_span_s"]
    pad = _frames(span)
    length = _frames(duration_s) + 2 * pad + 10
    prefix = _prefix(_activity(en_spans, length, shift=pad))
    shifts_s = np.arange(-span, span + ALIGNMENT["segment_step_s"] / 2, ALIGNMENT["segment_step_s"])
    shifts = np.round(shifts_s / FRAME_S).astype(np.int64)
    total = np.zeros(shifts.size)
    for start, end in kept:
        total += (prefix[np.clip(_frames(end) + pad + shifts, 0, length)]
                  - prefix[np.clip(_frames(start) + pad + shifts, 0, length)])
    at_zero = total[int(np.argmin(np.abs(shifts_s)))]
    far = total[np.abs(shifts_s) >= ALIGNMENT["peak_exclusion_s"]]
    return float(far.max() / at_zero) if at_zero > 0 else 1.0


def acceptance(q: Mapping) -> list[str]:
    """Reasons the rule rejects an alignment; empty when it is accepted."""
    rule, reasons = ACCEPTANCE, []
    groups = q["offset_groups"]
    if not groups:
        return ["no cue kept"]
    if len(groups) > rule["max_offset_groups"]:
        reasons.append(f"{len(groups)} offset groups > {rule['max_offset_groups']}")
    if min(group["dialogue_cues"] for group in groups) < rule["min_group_cues"]:
        reasons.append(f"an offset group has fewer than {rule['min_group_cues']} cues")
    if q["anchor_on_en"] < rule["min_anchor_on_en"]:
        reasons.append(f"anchor_on_en {q['anchor_on_en']} < {rule['min_anchor_on_en']}")
    if q["en_covered"] < rule["min_en_covered"]:
        reasons.append(f"en_covered {q['en_covered']} < {rule['min_en_covered']}")
    if q["peak_ratio"] > rule["max_peak_ratio"]:
        reasons.append(f"peak_ratio {q['peak_ratio']} > {rule['max_peak_ratio']}")
    return reasons


def _timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def srt_bytes(cues: Sequence[tuple[float, float, str]]) -> bytes:
    blocks = [f"{n}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n"
              for n, (start, end, text) in enumerate(cues, start=1)]
    return "\n".join(blocks).encode("utf-8")


def english_tracks(subtitles: Sequence[Mapping]) -> tuple[Mapping | None, Mapping | None]:
    """The English dialogue track (Full first, then Dialogue, then the first)
    and the English signs or songs track, if any. CC and SDH tracks are
    skipped: on dual-audio releases they caption the dub."""
    english = [s for s in subtitles if s["language"] == "eng" and s["codec"] in TEXT_CODECS]
    dialogue = [s for s in english if not s.get("forced") and not _EN_NOT_DIALOGUE_RE.search(s["title"])]
    signs = [s for s in english if _EN_SIGNS_RE.search(s["title"])]

    def rank(stream: Mapping) -> tuple[int, int]:
        title = stream["title"].lower()
        return (0 if "full" in title else 1 if "dialogue" in title else 2), stream["ordinal"]

    return (min(dialogue, key=rank) if dialogue else None), (signs[0] if signs else None)


def extract_track(ffmpeg: str, record: Mapping, stream: Mapping, work: Path) -> Path:
    """The subtitle stream as .ass (ASS/SSA) or .srt, cached by file size and
    modification time. Any FFmpeg error line raises: the file is incomplete or
    damaged."""
    video = Path(record["path"])
    suffix = ".ass" if stream["codec"] in ("ass", "ssa") else ".srt"
    out = work / f"{video.stem}.s{stream['ordinal']}{suffix}"
    stamp = {"size": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns}
    stamp_path = out.with_name(out.name + ".json")
    if out.exists() and stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp:
        return out
    work.mkdir(parents=True, exist_ok=True)
    codec = ["-c:s", "copy"] if suffix == ".ass" else ["-c:s", "srt"]
    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-map", f"0:s:{stream['ordinal']}",
         *codec, str(out)],
        capture_output=True,
    )
    errors = completed.stderr.decode("utf-8", "replace").strip().splitlines()
    if completed.returncode != 0 or errors:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"extracting subtitle stream #{stream['index']} reports: "
                           f"{errors[-1] if errors else completed.returncode}")
    stamp_path.write_text(json.dumps(stamp))
    return out


def _reference(ffmpeg: str, record: Mapping, out: Path) -> tuple[list, dict]:
    dialogue_track, signs_track = english_tracks(record["subtitles"])
    if dialogue_track is None:
        raise LookupError("no English dialogue subtitle track to align against")
    work = out / "work" / record["title"]
    track = extract_track(ffmpeg, record, dialogue_track, work)
    signs = extract_track(ffmpeg, record, signs_track, work) if signs_track and track.suffix == ".ass" else None
    cues = english_cues(track, signs)
    reference = {
        "dialogue_stream": {k: dialogue_track[k] for k in ("index", "language", "title", "codec")},
        "dialogue_track_file": _relative(track, out),
        "signs_stream": {k: signs_track[k] for k in ("index", "language", "title", "codec")} if signs else None,
        "signs_track_file": _relative(signs, out) if signs else None,
        "dialogue_cues": len(cues),
    }
    return cues, reference


def reference_cues(entry: Mapping, folder: Path) -> list[tuple[float, float, str]]:
    """The English dialogue cues an entry was aligned against, from the
    extracted tracks kept under ``folder/work``."""
    reference = entry["reference"]
    signs = reference.get("signs_track_file")
    return english_cues(folder / reference["dialogue_track_file"], folder / signs if signs else None)


def _source_files(fetcher: Fetcher, source: Source, episode: int) -> list[str]:
    target = source.source_episode(episode)
    names = [name for name in fetcher.listing(source.site, source.directory)
             if (match := re.search(source.pattern, name)) and int(match["ep"]) == target]
    return sorted(names, key=lambda name: (len(name), name))[:1]


def _relative(path: Path, folder: Path) -> str:
    return path.relative_to(folder).as_posix()


KIND_NAMES = ("netflix-captions", "srt", "ass")


@dataclass(frozen=True)
class Candidate:
    """One subtitle file to try for an episode: its name, where it comes
    from, and how to read its bytes with their provenance."""

    name: str
    describe: dict
    load: Callable[[], tuple[bytes, dict]]


def _archive_candidate(folder: Path, file: Mapping, match: str) -> Candidate:
    def load() -> tuple[bytes, dict]:
        data = read_member(folder, file["zip"], file["member"])
        return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    describe = {
        "site": "archive", "zip": file["zip"], "file": file["member"], "kind": KIND_NAMES[file["kind"]],
        "series_match": match, "tier": file["tier"], "numbering": file["numbering"],
        "file_season": file["season"], "file_episode": file["episode"], "crc32": f"{file['crc32']:08x}",
    }
    return Candidate(file["member"], describe, load)


def episode_candidates(
    record: Mapping, offset: int, archive: tuple[Path, Sequence[Mapping], Sequence[Mapping]] | None,
    fetcher: Fetcher | None,
) -> tuple[list[Candidate], dict | None]:
    """The subtitle files to try for one episode, and what the archive
    found: first among the zips whose name equals a title name, then, when
    they hold no file for the episode, among those whose name starts with
    one. The network sources (``fetcher`` set) are asked only when the
    archive has no usable file for the episode."""
    found = None
    if archive is not None:
        folder, exact, prefix = archive
        for match, series in (("name", exact), ("prefix", prefix)):
            files, found = archive_candidates(record, offset, series)
            found["series_match"] = match
            if files:
                return [_archive_candidate(folder, file, match) for file in files], found
            if found["ambiguous"]:
                break
    sources = TITLES.get(record["title"])
    if fetcher is None or sources is None or record["season"] != sources.season:
        return [], found
    candidates = []
    for source in sources.sources:
        for name in _source_files(fetcher, source, record["episode"]):
            candidates.append(Candidate(
                name, {**source.describe(), "file": name},
                lambda source=source, name=name: fetcher.fetch(source, name),
            ))
    return candidates, found


def _completeness(record: Mapping, checked: Mapping | None) -> dict | None:
    """Who checked the episode's Japanese audio for completeness, and the
    outcome; None when nobody did."""
    if record.get("audio_decode_errors") is not None:
        return {"checked_by": "inventory", "audio_decode_errors": record["audio_decode_errors"],
                "unwritten_blocks": record.get("unwritten_blocks") or []}
    if checked is None:
        return None
    return {"checked_by": "external_subs", "audio_decode_errors": checked["audio_decode_errors"],
            "unwritten_blocks": checked["unwritten_blocks"], "problem": checked["problem"]}


def process_episode(
    ffmpeg: str, record: Mapping, candidates: Sequence[Candidate], out: Path
) -> tuple[dict, list, list]:
    """The manifest entry of one episode, the source files tried as (file name,
    cues), and the English reference cues. Candidates are tried in order; the
    first that passes the Japanese text check and the acceptance rule is
    written, once the episode's audio is known complete."""
    en_cues, reference = _reference(ffmpeg, record, out)
    if len(en_cues) < 50:
        raise LookupError(f"English dialogue track has {len(en_cues)} usable cues (< 50)")
    duration = float(record["duration_s"])
    tried, attempts, seen = [], [], set()
    checked = None
    for candidate in candidates:
        try:
            data, meta = candidate.load()
        except (OSError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
            attempts.append({"source": candidate.describe, "accepted": False, "reason": f"unreadable: {exc}"})
            continue
        if meta["sha256"] in seen:
            continue
        seen.add(meta["sha256"])
        attempt = {"source": {**candidate.describe, **meta}}
        try:
            jp = japanese_cues(data, candidate.name)
            attempt["japanese_text"] = japanese_text(jp)
            if problem := japanese_text_problem(attempt["japanese_text"]):
                attempts.append({**attempt, "accepted": False, "reason": problem})
                continue
            aligned = align(jp, en_cues, duration)
        except ValueError as exc:
            attempts.append({**attempt, "accepted": False, "reason": f"unusable: {exc}"})
            continue
        tried.append((candidate.name, jp))
        q = quality(jp, en_cues, aligned, duration)
        reasons = acceptance(q)
        if not reasons and record.get("audio_decode_errors") is None:
            checked = checked or audio_completeness(ffmpeg, Path(record["path"]), record["japanese_audio_ordinal"])
            if checked["problem"]:
                reasons = [checked["problem"]]
        attempt.update({
            "alignment": {"global_offset_s": aligned["global_offset_s"], "runs": aligned["runs"]},
            "quality": q, "accepted": not reasons, "reason": "; ".join(reasons) or None,
        })
        attempts.append(attempt)
        if checked is not None and checked["problem"]:
            return _entry(record, reference, attempt, attempts, checked), tried, en_cues
        if not reasons:
            cues = output_cues(jp, aligned, duration)
            target = out / record["title"] / f"{Path(record['file']).stem}.ja.srt"
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = srt_bytes(cues)
            target.write_bytes(payload)
            attempt.update({"subtitle_file": _relative(target, out),
                            "subtitle_sha256": hashlib.sha256(payload).hexdigest(),
                            "subtitle_cues": len(cues)})
            return _entry(record, reference, attempt, attempts, checked), tried, en_cues
    if not attempts:
        raise LookupError("no readable subtitle file for this episode")
    scored = [a for a in attempts if "quality" in a]
    best = max(scored, key=lambda a: a["quality"]["anchor_on_en"]) if scored else attempts[0]
    return _entry(record, reference, best, attempts, checked), tried, en_cues


def _entry(record: Mapping, reference: Mapping, chosen: Mapping, attempts: list, checked: Mapping | None) -> dict:
    video = Path(record["path"])
    return {
        "title": record["title"],
        "video_file": record["file"],
        "video_path": record["path"],
        "video_size": video.stat().st_size,
        "video_duration_s": record["duration_s"],
        "season": record["season"],
        "episode": record["episode"],
        "subtitle_file": chosen.get("subtitle_file"),
        "subtitle_sha256": chosen.get("subtitle_sha256"),
        "subtitle_cues": chosen.get("subtitle_cues"),
        "source": chosen["source"],
        "japanese_text": chosen.get("japanese_text"),
        "reference": reference,
        "alignment": chosen.get("alignment"),
        "quality": chosen.get("quality"),
        "audio_completeness": _completeness(record, checked),
        "accepted": bool(chosen["accepted"]),
        "reason": chosen.get("reason"),
        "attempts": [{"source_file": a["source"]["file"], "site": a["source"]["site"],
                      "accepted": a["accepted"], "reason": a["reason"]} for a in attempts],
    }


def _skip_reason(record: Mapping, locked: LockedTitles, windows_per_episode: int) -> str | None:
    if locked.path_title(Path(record["path"])) is not None:
        return "locked title"
    if record.get("japanese_dialogue_subtitle_ordinal") is not None:
        return "has an embedded Japanese dialogue track"
    reason = record.get("reason") or ""
    if reason.startswith(("unreadable", "incomplete")) or record.get("audio_decode_errors") or record.get(
        "unwritten_blocks"
    ):
        return reason or "incomplete"
    if record.get("japanese_audio_ordinal") is None:
        return reason or "no Japanese audio stream"
    if record["episode"] is None:
        return "no episode number"
    if not record["duration_s"] or windows(record["duration_s"], windows_per_episode) is None:
        return f"too short for {windows_per_episode} windows"
    if english_tracks(record["subtitles"])[0] is None:
        return "no English dialogue subtitle track to align against"
    if not Path(record["path"]).exists():
        return "file no longer on disk"
    return None


def controls(results: Sequence[tuple[dict, list, list]]) -> list[dict]:
    """Each episode aligned against the first file tried for another episode
    of the same title (the next one whose file it did not try itself); the
    acceptance rule should reject every one."""
    out = []
    for position, (entry, _, en_cues) in enumerate(results):
        own = {attempt["source_file"] for attempt in entry["attempts"]}
        for _, other_tried, _ in [*results[position + 1:], *results[:position]]:
            if not other_tried or other_tried[0][0] in own:
                continue
            name, jp = other_tried[0]
            aligned = align(jp, en_cues, entry["video_duration_s"])
            q = quality(jp, en_cues, aligned, entry["video_duration_s"])
            reasons = acceptance(q)
            out.append({"title": entry["title"], "video_file": entry["video_file"], "against": name,
                        "quality": q, "accepted": not reasons, "reason": "; ".join(reasons) or None})
            break
    return out


def _spread(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    return f"min {min(values):.3f} median {statistics.median(values):.3f} max {max(values):.3f}"


def recorded_manifests() -> set[Path]:
    """External-subtitle manifests a set under ``bench_root()/sets`` was
    built from."""
    recorded = {(root() / "manifest.json").resolve()}
    for set_json in (bench_root() / "sets").glob("*/set.json"):
        try:
            external = json.loads(set_json.read_text(encoding="utf-8")).get("build", {}).get("external_subs")
        except (OSError, ValueError):
            continue
        if external and external.get("manifest"):
            recorded.add((bench_root() / external["manifest"]).resolve())
    return recorded


def build(
    inventory_path: Path, out: Path, archive_folder: Path | None, network: bool, only: Sequence[str],
    per_title: int | None, windows_per_episode: int, run_controls: bool, locked: LockedTitles,
) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg must be on PATH")
    manifest_file = out / "manifest.json"
    if manifest_file.resolve() in recorded_manifests():
        raise SystemExit(f"{out} holds external subtitles a built set records; choose another --out")
    inventory_bytes = Path(inventory_path).read_bytes()
    inventory = json.loads(inventory_bytes)
    if inventory.get("schema_version") not in INVENTORY_SCHEMAS:
        raise SystemExit(f"{inventory_path}: unsupported inventory schema {inventory.get('schema_version')!r}")
    if inventory.get("locked_titles_sha256") != locked.sha256:
        raise SystemExit(f"{inventory_path} was built under another locked-title list; rebuild the inventory")
    by_title: dict[str, list[dict]] = {}
    locked_files = 0
    for record in inventory["files"]:
        if record["title"] and not record["locked_title"] and locked.title_of(record["title"]) is None:
            by_title.setdefault(record["title"], []).append(record)
        elif record["title"]:
            locked_files += 1
    unknown = sorted(set(only) - set(by_title))
    if unknown:
        raise SystemExit(f"titles not in the inventory (or locked): {unknown}")
    selected = [
        title for title, records in sorted(by_title.items())
        if (not only or title in only) and any(
            r["japanese_dialogue_subtitle_ordinal"] is None and r["japanese_audio_ordinal"] is not None
            for r in records
        )
    ]

    out.mkdir(parents=True, exist_ok=True)
    index, index_sha256, series = None, None, None
    if archive_folder is not None:
        index = index_archive(archive_folder, locked)
        payload = (json.dumps(index, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        (out / "archive-index.json").write_bytes(payload)
        index_sha256 = hashlib.sha256(payload).hexdigest()
        series = index["zips"]
        counts = index["counts"]
        print(f"archive: {counts['zips_indexed']} zips indexed, {counts['subtitle_files']} subtitle files; "
              f"skipped {counts['locked_zips_skipped']} locked zips and {counts['locked_members_skipped']} "
              f"locked members (not named), members {counts['skipped_members']}", flush=True)

    fetcher = Fetcher() if network else None
    entries, skipped, control_rows, ambiguous = [], [], [], []
    matches: dict[str, dict] = {}
    for title in selected:
        records = sorted(by_title[title], key=sort_key)
        names = title_names(title, records)
        exact, prefix = match_series(names, series) if series is not None else ([], [])
        offsets = season_offsets(records)
        matches[title] = {"names": sorted(names), "zips": [z["zip"] for z in exact],
                          "prefix_zips": [z["zip"] for z in prefix],
                          "season_offsets": {str(s): k for s, k in sorted(offsets.items())}}
        print(f"== {title}: archive zips {matches[title]['zips']}, by prefix {matches[title]['prefix_zips']}",
              flush=True)
        pending: list[tuple[dict, list[Candidate]]] = []
        for record in records:
            reason = _skip_reason(record, locked, windows_per_episode)
            if reason is None:
                archive = (archive_folder, exact, prefix) if series is not None else None
                try:
                    candidates, found = episode_candidates(
                        record, offsets.get(record["season"] or 1, 0), archive, fetcher
                    )
                except OSError as exc:
                    candidates, found = [], None
                    reason = f"network sources unavailable: {exc}"
                if found and found["ambiguous"]:
                    ambiguous.append({"title": title, "video_file": record["file"], "season": record["season"],
                                      "episode": record["episode"], "series_match": found["series_match"],
                                      "tier": found["tier"], "zips": found["zips"]})
                if not candidates and reason is None:
                    reason = (f"ambiguous archive match ({found['tier']}: {', '.join(found['zips'])})"
                              if found and found["ambiguous"] else
                              "no subtitle file for this episode in the archive or the configured sources")
            if reason is not None:
                skipped.append({"title": title, "video_file": record["file"], "reason": reason})
                continue
            pending.append((record, candidates))

        results, taken = [], set()

        def try_episode(item: tuple[dict, list[Candidate]], title: str = title) -> bool:
            record, candidates = item
            number = (record["season"], record["episode"])
            if number in taken:
                skipped.append({"title": title, "video_file": record["file"],
                                "reason": "repeats the season and episode number of an accepted episode"})
                return False
            try:
                entry, tried, en_cues = process_episode(ffmpeg, record, candidates, out)
            except (LookupError, RuntimeError, OSError) as exc:
                skipped.append({"title": title, "video_file": record["file"], "reason": str(exc)})
                print(f"skip {record['file']}: {exc}", flush=True)
                return False
            results.append((entry, tried, en_cues))
            entries.append(entry)
            q = entry["quality"] or {}
            print(
                f"{'accept' if entry['accepted'] else 'reject'} {record['file']}: "
                f"{entry['source']['site']} {entry['source']['file']} | groups "
                f"{[g['offset_s'] for g in q.get('offset_groups', [])]} anchor {q.get('anchor_on_en')} "
                f"covered {q.get('en_covered')} peak {q.get('peak_ratio')}"
                + (f" | {entry['reason']}" if entry["reason"] else ""),
                flush=True,
            )
            if entry["accepted"]:
                taken.add(number)
            return entry["accepted"]

        if per_title is None:
            for item in pending:
                try_episode(item)
        else:
            walk(pending, per_title, try_episode, max_tries=3 * per_title)
        matches[title]["episodes_with_files"] = len(pending)
        matches[title]["episodes_tried"] = len(results)
        if run_controls:
            control_rows += controls(results)
        keep = {out / e["subtitle_file"] for e in entries if e["title"] == title and e["accepted"]}
        for stale in (out / title).glob("*.ja.srt") if (out / title).is_dir() else ():
            if stale not in keep:
                stale.unlink()

    previous = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    done = set(selected)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": "scripts.asr_bench.external_subs",
        "inventory": {"path": str(inventory_path), "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                      "schema_version": inventory["schema_version"]},
        "locked_titles_sha256": locked.sha256,
        "ffmpeg_version": tool_version(ffmpeg),
        "archive": None if index is None else {
            "folder": str(archive_folder),
            "index": "archive-index.json",
            "index_sha256": index_sha256,
            "counts": index["counts"],
            "max_files_per_episode": ARCHIVE_MAX_FILES,
            "min_shared_files": MIN_SHARED_FILES,
        },
        "network": network,
        "selection": {"episodes_per_title": per_title, "windows_per_episode": windows_per_episode,
                      "max_tries_per_title": None if per_title is None else 3 * per_title},
        "sources": {
            "kitsunekko": f"{KITSUNEKKO}/dirlist.php?dir=subtitles%2Fjapanese%2F",
            "kitsunekko-mirror": f"https://github.com/{MIRROR_REPO}",
            "kitsunekko_mirror_commit": (fetcher._mirror_commit if fetcher else None)
            or previous.get("sources", {}).get("kitsunekko_mirror_commit"),
            "request_interval_s": REQUEST_INTERVAL_S,
            "user_agent": USER_AGENT,
            "titles": {title: {"season": s.season, "sources": [src.describe() for src in s.sources]}
                       for title, s in TITLES.items()},
        },
        "japanese_text": {**JAPANESE_TEXT, "description": JAPANESE_TEXT_TEXT},
        "alignment_method": {**ALIGNMENT, "description": ALIGNMENT_TEXT},
        "acceptance_rule": {**ACCEPTANCE, "description": ACCEPTANCE_TEXT},
        "title_matches": {**{t: m for t, m in previous.get("title_matches", {}).items() if t not in done}, **matches},
        "ambiguous": [a for a in previous.get("ambiguous", []) if a["title"] not in done] + ambiguous,
        "episodes": [e for e in previous.get("episodes", []) if e["title"] not in done] + entries,
        "skipped": [s for s in previous.get("skipped", []) if s["title"] not in done] + skipped,
        "controls": [c for c in previous.get("controls", []) if c["title"] not in done] + control_rows,
    }
    temporary = manifest_file.with_name("manifest.json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(manifest_file)

    for title in selected:
        mine = [e for e in entries if e["title"] == title]
        accepted = [e for e in mine if e["accepted"]]
        title_skips = Counter(s["reason"].split(":")[0] for s in skipped if s["title"] == title)
        print(f"{title}: {len(by_title[title])} on disk, {matches[title]['episodes_with_files']} with a subtitle "
              f"file, {len(mine)} aligned, {len(accepted)} accepted; skipped {dict(title_skips)}")
        for key in ("anchor_on_en", "en_covered", "peak_ratio"):
            print(f"  accepted {key}: {_spread([e['quality'][key] for e in accepted])}")
        sites = Counter(f"{e['source']['site']} {e['source']['kind']}" for e in accepted)
        print(f"  sources: {dict(sites)}")
        mine_controls = [c for c in control_rows if c["title"] == title]
        if mine_controls:
            print(f"  controls: {len(mine_controls)} aligned, {sum(c['accepted'] for c in mine_controls)} "
                  f"accepted; anchor_on_en {_spread([c['quality']['anchor_on_en'] for c in mine_controls])}; "
                  f"peak_ratio {_spread([c['quality']['peak_ratio'] for c in mine_controls])}")
    print(f"{len(ambiguous)} episodes with an ambiguous archive match; {locked_files} locked "
          f"inventory files left out; {fetcher.requests if fetcher else 0} requests; wrote {manifest_file}")
    return manifest


def spot(video_file: str, count: int, folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    matches = [e for e in manifest["episodes"] if e["video_file"] == video_file and e["accepted"]]
    if len(matches) != 1:
        raise SystemExit(f"{video_file!r}: {len(matches)} accepted manifest entries")
    entry = matches[0]
    payload = (folder / entry["subtitle_file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["subtitle_sha256"]:
        raise SystemExit(f"{entry['subtitle_file']} does not match its manifest sha256")
    cues = [(c.original_start_ms / 1000, c.original_end_ms / 1000, c.raw_text) for c in _parse_srt(payload)]
    dialogue = [cue for cue in cues if is_dialogue(cue[2])]
    en_cues = reference_cues(entry, folder)
    print(f"{entry['video_file']} <- {entry['source']['file']} ({entry['source']['kind']})")
    for index in range(count):
        start, end, text = dialogue[(2 * index + 1) * len(dialogue) // (2 * count)]
        print(f"JP {_timestamp(start)} --> {_timestamp(end)}  {' '.join(text.split())}")
        overlapping = [cue for cue in en_cues if cue[0] < end and start < cue[1]]
        for en_start, en_end, en_text in overlapping or [(None, None, "(no English cue)")]:
            times = f"{_timestamp(en_start)} --> {_timestamp(en_end)}" if en_start is not None else " " * 29
            print(f"EN {times}  {en_text}")


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="find, align, score and write the contract files")
    build_parser.add_argument("--inventory", type=Path, required=True, help="inventory JSON")
    build_parser.add_argument("--out", type=Path, required=True,
                              help="output folder (never one a built set records)")
    build_parser.add_argument("--archive", type=Path, help="folder of subtitle zips, one per series")
    build_parser.add_argument("--no-network", action="store_true", help="never use the network sources")
    build_parser.add_argument("--title", action="append", default=[], help="title key; repeatable (default: all)")
    build_parser.add_argument("--episodes-per-title", type=int,
                              help="stop a title once this many episodes are accepted (default: try all)")
    build_parser.add_argument("--windows-per-episode", type=int, default=DEFAULT_WINDOWS,
                              help="windows an episode must have room for")
    build_parser.add_argument("--controls", action="store_true",
                              help="also align each episode against another episode's file")
    build_parser.add_argument("--state", type=Path, help=f"E0 rebuild state (default: <E0 root>/{STATE_FILE})")
    spot_parser = commands.add_parser("spot", help="print Japanese cues of an accepted episode next to English")
    spot_parser.add_argument("--video", required=True, help="episode file name")
    spot_parser.add_argument("--root", type=Path, default=root(), help="external-subtitle folder")
    spot_parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.archive is not None and not args.archive.is_dir():
            parser.error(f"--archive {args.archive} is not a folder")
        if args.episodes_per_title is not None and args.episodes_per_title < 1:
            parser.error("--episodes-per-title must be at least 1")
        locked = locked_titles(load_fixtures(args.state or e0_boundary().root / STATE_FILE))
        build(args.inventory, args.out, args.archive, not args.no_network, args.title, args.episodes_per_title,
              args.windows_per_episode, args.controls, locked)
    else:
        spot(args.video, args.count, args.root)


if __name__ == "__main__":
    main()
