"""Inventory of anime episodes for the episode-based silver set.

    python -m scripts.asr_bench.inventory --media <folder or file> [--media ...] [--skip GLOB ...] --out <json>
        [--decode-per-title N [--windows-per-episode K]]

Every video file below each ``--media`` folder (recursively; a file argument
is taken as is) is probed with ffprobe; a folder whose name matches a
``--skip`` glob is skipped with everything below it. Per file the inventory
records the series title, the locked title it belongs to, if any, its season
and episode numbers, its audio streams (index, language, title, channels)
and subtitle streams (index, language, title, codec), and the streams a
silver set would use:

- Title: ``titles.title_key`` of the innermost folder below the media
  argument whose name holds a title, a leading number dropped (``01 Title``
  names ``title``, so each numbered season folder is its own title); the
  file name's key when no folder holds one. File and folder names that name
  different titles do not exclude an episode; the inventory lists them.
- Season and episode: ``titles.season_episode`` of the file name; a file
  name without a season takes the season of the innermost folder below the
  media argument that names exactly one (``titles.folder_season``).
- Japanese audio: the first audio stream tagged ``jpn`` whose title does not
  say commentary; when no stream is tagged, the only audio stream of a
  release whose names announce no dub or dual audio.
- Japanese dialogue subtitles: a ``jpn`` text track whose title does not say
  signs, songs or forced; a track titled Full wins, then CC or SDH, then the
  first. Japanese signs, songs or forced text tracks are listed separately.

An episode is eligible when its title is known and not locked, it is
readable, both streams exist and its Japanese audio stream is complete: no
64 KiB block at 25, 50, 75 or 99% of the file is all zero bytes (space a
download has not written yet) and the stream decodes in full without a
single FFmpeg error line (``ffmpeg -v error -i <file> -map 0:a:<n> -f null
-``); otherwise the record says why. A file that cannot be read (an
incomplete download, or one removed during the scan) is listed as
unreadable; one with unwritten blocks or whose audio decode reports errors
(a download still running) is listed as incomplete.

Without ``--decode-per-title`` every readable episode with Japanese audio is
decoded. With it, only the episodes ``build_silver_episodes`` can take are:
per title, among the episodes with an embedded Japanese dialogue track, an
episode number and room for ``--windows-per-episode`` windows (default 2),
``selection.walk`` checks the evenly spaced picks, each failing pick falling
to its nearest untried neighbour, until N episodes per title are complete
(episodes repeating a season and episode number already taken are passed
over). The other episodes with an embedded track are listed as not decode
checked, and episodes without one are not decoded here:
``scripts.asr_bench.external_subs`` checks the ones it accepts.

Locked titles come from the private list (``titles.locked_titles_path()``),
checked against the E0 locked fixtures, so the E0 private root must be set;
without a complete list the inventory refuses to run. An episode is locked
when its file name or any folder name above it names a locked title,
whatever title it resolves to. No VAD or ASR model is run. Only names and
stream metadata are written, never subtitle text, to ``--out``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .build_silver import STATE_FILE, e0_boundary, load_fixtures, locked_titles
from .selection import DEFAULT_WINDOWS, sort_key, walk, windows
from .sets import bench_root
from .titles import MEDIA_SUFFIXES, LockedTitles, folder_season, season_episode, title_key

INVENTORY_SCHEMA = "asr-bench-inventory-1.2"
READABLE_SCHEMAS = ("asr-bench-inventory-1.1", INVENTORY_SCHEMA)
NOT_DECODE_CHECKED = "audio decode not checked"
UNWRITTEN_OFFSETS = (0.25, 0.5, 0.75, 0.99)
UNWRITTEN_BLOCK = 1 << 16
NO_DIALOGUE_SUBTITLE = "no Japanese dialogue subtitle track"
JAPANESE = ("jpn", "ja")
UNTAGGED = ("", "und")
TEXT_SUBTITLE_CODECS = ("subrip", "ass", "ssa", "webvtt", "mov_text", "text")
NOT_DIALOGUE_RE = re.compile(r"sign|song|forced", re.IGNORECASE)
_FULL_RE = re.compile(r"full", re.IGNORECASE)
_CAPTIONS_RE = re.compile(r"\b(?:cc|sdh)\b", re.IGNORECASE)
_COMMENTARY_RE = re.compile(r"commentary", re.IGNORECASE)
_NOT_JAPANESE_ONLY_RE = re.compile(r"dual[\s._-]?audio|multi[\s._-]?audio|\bdub|english", re.IGNORECASE)
_LEADING_NUMBER_RE = re.compile(r"^\d{1,3} (?=\S)")


def default_path() -> Path:
    return bench_root() / "inventory" / "episodes-v3.json"


def find_media(media: Sequence[Path], skip: Sequence[str]) -> tuple[list[tuple[Path, Path]], list[str]]:
    """(media argument, video file) pairs in path order, and the skipped
    folders."""
    found: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for root in map(Path, media):
        if root.is_file():
            if root.suffix.lower() in MEDIA_SUFFIXES:
                found.append((root.parent, root))
            continue
        if not root.is_dir():
            raise SystemExit(f"--media {root} does not exist")
        for directory, folders, files in os.walk(root):
            kept = []
            for name in sorted(folders):
                if any(fnmatch.fnmatch(name, pattern) for pattern in skip):
                    skipped.append(str(Path(directory, name)))
                else:
                    kept.append(name)
            folders[:] = kept
            found.extend(
                (root, Path(directory, name)) for name in files
                if Path(name).suffix.lower() in MEDIA_SUFFIXES
            )
    return sorted(found, key=lambda pair: str(pair[1])), skipped


def probe(ffprobe: str, path: Path) -> dict:
    """ffprobe's streams and duration; an unreadable file (an incomplete
    download, or one removed during the scan) yields no streams and
    ffprobe's last error line."""
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-of", "json", "-show_entries",
         "format=duration:stream=index,codec_type,codec_name,channels"
         ":stream_tags=language,title:stream_disposition=forced",
         str(path)],
        capture_output=True,
    )
    if completed.returncode != 0:
        errors = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        return {"streams": [], "format": {}, "error": errors[-1] if errors else "ffprobe failed"}
    return json.loads(completed.stdout)


def audio_decode_errors(ffmpeg: str, path: Path, ordinal: int) -> list[str]:
    """FFmpeg's error lines from decoding one audio stream in full. Any line
    means the file is incomplete (a download still running) or damaged: a
    window cut from it could pair subtitle text with missing audio."""
    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", f"0:a:{ordinal}", "-f", "null", "-"],
        capture_output=True,
    )
    errors = completed.stderr.decode("utf-8", "replace").strip().splitlines()
    if completed.returncode and not errors:
        errors = [f"ffmpeg exited with {completed.returncode}"]
    return [error.replace(f"{path}: ", "") for error in errors]


def unwritten_blocks(path: Path) -> list[float]:
    """The offsets (fractions of the file size) among UNWRITTEN_OFFSETS whose
    64 KiB block is all zero bytes: space a download has reserved but not
    written yet. Compressed video never holds such a block."""
    size = path.stat().st_size
    found = []
    with open(path, "rb") as handle:
        for offset in UNWRITTEN_OFFSETS:
            handle.seek(max(0, int(size * offset) - UNWRITTEN_BLOCK // 2))
            block = handle.read(UNWRITTEN_BLOCK)
            if block and block.count(0) == len(block):
                found.append(offset)
    return found


def audio_completeness(ffmpeg: str, path: Path, ordinal: int) -> dict:
    """Whether one audio stream of a file is complete: no unwritten blocks,
    then a full decode without an error line. ``problem`` is the reason it
    is not (``incomplete: ...``), or None."""
    zeros = unwritten_blocks(path)
    if zeros:
        return {
            "unwritten_blocks": zeros, "audio_decode_errors": None,
            "problem": f"incomplete: all-zero 64 KiB blocks at {', '.join(f'{z:.0%}' for z in zeros)} "
                       f"of the file (a download still running)",
        }
    errors = audio_decode_errors(ffmpeg, path, ordinal)
    return {
        "unwritten_blocks": [], "audio_decode_errors": len(errors),
        "problem": None if not errors else (
            f"incomplete: decoding the Japanese audio reports {len(errors)} error "
            f"line(s), first: {errors[0]}"
        ),
    }


def _streams(probed: Mapping, codec_type: str) -> list[dict]:
    chosen = []
    for ordinal, stream in enumerate(s for s in probed["streams"] if s["codec_type"] == codec_type):
        tags = stream.get("tags", {})
        record = {
            "ordinal": ordinal,
            "index": stream["index"],
            "language": tags.get("language", ""),
            "title": tags.get("title", ""),
            "codec": stream.get("codec_name", ""),
        }
        if codec_type == "audio":
            record["channels"] = stream.get("channels")
        else:
            record["forced"] = bool(stream.get("disposition", {}).get("forced"))
        chosen.append(record)
    return chosen


def japanese_audio(audio: Sequence[Mapping], names: Sequence[str]) -> tuple[int | None, str | None]:
    """Ordinal of the Japanese audio stream and the basis for choosing it."""
    tagged = [s for s in audio if s["language"] in JAPANESE and not _COMMENTARY_RE.search(s["title"])]
    if tagged:
        return tagged[0]["ordinal"], "language jpn"
    if (
        len(audio) == 1 and audio[0]["language"] in UNTAGGED
        and not any(_NOT_JAPANESE_ONLY_RE.search(name) for name in names)
    ):
        return 0, "only audio stream, untagged, release names announce no dub"
    return None, None


def _japanese_text(subtitles: Sequence[Mapping]) -> list[Mapping]:
    return [s for s in subtitles if s["language"] in JAPANESE and s["codec"] in TEXT_SUBTITLE_CODECS]


def _not_dialogue(stream: Mapping) -> bool:
    return stream["forced"] or bool(NOT_DIALOGUE_RE.search(stream["title"]))


def japanese_dialogue_subtitle(subtitles: Sequence[Mapping]) -> int | None:
    """Ordinal of the Japanese dialogue track: Full first, then CC/SDH."""
    candidates = [s for s in _japanese_text(subtitles) if not _not_dialogue(s)]
    if not candidates:
        return None

    def rank(stream: Mapping) -> tuple[int, int]:
        title = stream["title"]
        return (0 if _FULL_RE.search(title) else 1 if _CAPTIONS_RE.search(title) else 2), stream["ordinal"]

    return min(candidates, key=rank)["ordinal"]


def japanese_non_dialogue_subtitles(subtitles: Sequence[Mapping]) -> list[int]:
    """Ordinals of the Japanese signs, songs or forced text tracks."""
    return [s["ordinal"] for s in _japanese_text(subtitles) if _not_dialogue(s)]


def names_agree(left: str, right: str) -> bool:
    """One key's words appear, in order and contiguous, in the other's."""
    short, long = sorted((left.split(), right.split()), key=len)
    return any(long[i : i + len(short)] == short for i in range(len(long) - len(short) + 1))


def folder_title_key(name: str) -> str | None:
    """``title_key`` of a folder name without a leading number (``01 Title``)."""
    key = title_key(name)
    return _LEADING_NUMBER_RE.sub("", key) if key else None


def resolve_title(root: Path, path: Path) -> tuple[str | None, dict, str | None]:
    """The title of an episode, the keys it was read from, and the name that
    gave it (``folder`` or ``file``): the innermost folder below the media
    argument whose name holds a title, else the file name."""
    folders = path.relative_to(root).parts[:-1]
    keys = {"file": title_key(path.name), "folders": [folder_title_key(name) for name in folders]}
    for key in reversed(keys["folders"]):
        if key:
            return key, keys, "folder"
    if keys["file"]:
        return keys["file"], keys, "file"
    return None, keys, None


def record_season(root: Path, path: Path, season: int | None) -> int | None:
    """The file name's season, else that of the innermost folder below the
    media argument that names exactly one."""
    if season is not None:
        return season
    for folder in reversed(path.relative_to(root).parts[:-1]):
        if (found := folder_season(folder)) is not None:
            return found
    return None


def inventory_record(root: Path, path: Path, probed: Mapping, locked_list: LockedTitles) -> dict:
    """An episode's names and stream metadata; ``finish_record`` adds the
    completeness check and the verdict."""
    title, keys, basis = resolve_title(root, path)
    season, episode = season_episode(path.name)
    audio = _streams(probed, "audio")
    subtitles = _streams(probed, "subtitle")
    names = (path.name, *path.relative_to(root).parts[:-1])
    audio_ordinal, audio_basis = japanese_audio(audio, names)
    duration = probed.get("format", {}).get("duration")
    return {
        "path": str(path),
        "file": path.name,
        "media_root": str(root),
        "folders": list(path.relative_to(root).parts[:-1]),
        "title": title,
        "title_keys": keys,
        "title_basis": basis,
        "locked_title": locked_list.path_title(path),
        "season": record_season(root, path, season),
        "episode": episode,
        "duration_s": round(float(duration), 3) if duration else None,
        "audio": audio,
        "subtitles": subtitles,
        "japanese_audio_ordinal": audio_ordinal,
        "japanese_audio_basis": audio_basis,
        "japanese_dialogue_subtitle_ordinal": japanese_dialogue_subtitle(subtitles),
        "japanese_non_dialogue_subtitle_ordinals": japanese_non_dialogue_subtitles(subtitles),
        "readable": "error" not in probed,
        "unreadable_error": probed["error"].replace(f"{path}: ", "") if "error" in probed else None,
        "unwritten_blocks": None,
        "audio_decode_errors": None,
        "eligible": False,
        "reason": None,
    }


def can_decode(record: Mapping) -> bool:
    """Title known and not locked, readable, a Japanese audio stream."""
    return bool(
        not record["locked_title"] and record["title"] and record["readable"]
        and record["japanese_audio_ordinal"] is not None
    )


def check_completeness(ffmpeg: str, record: dict) -> str | None:
    """Run ``audio_completeness`` on the record's Japanese audio stream and
    store its outcome; the problem, or None when complete."""
    outcome = audio_completeness(ffmpeg, Path(record["path"]), record["japanese_audio_ordinal"])
    record["unwritten_blocks"] = outcome["unwritten_blocks"]
    record["audio_decode_errors"] = outcome["audio_decode_errors"]
    record["completeness_problem"] = outcome["problem"]
    return outcome["problem"]


def finish_record(record: dict, unchecked: str | None) -> None:
    """Set ``reason`` and ``eligible``. ``unchecked`` is why an otherwise
    eligible episode was not checked for completeness, if it was not."""
    problem = record.pop("completeness_problem", None)
    if record["locked_title"]:
        reason = f"locked title {record['locked_title']}"
    elif record["title"] is None:
        reason = "no title in the file or folder names"
    elif not record["readable"]:
        reason = f"unreadable: {record['unreadable_error']}"
    elif record["japanese_audio_ordinal"] is None:
        reason = "no Japanese audio stream"
    elif problem:
        reason = problem
    elif record["japanese_dialogue_subtitle_ordinal"] is None:
        reason = NO_DIALOGUE_SUBTITLE
    elif record["audio_decode_errors"] is None:
        reason = f"{NOT_DECODE_CHECKED}: {unchecked}"
    else:
        reason = None
    record["reason"], record["eligible"] = reason, reason is None


def bounded_decode(
    ffmpeg: str, records: Sequence[dict], per_title: int, windows_per_episode: int
) -> dict[str, str]:
    """Check completeness only for the episodes the builder can take (see
    the module docstring); why each other episode with an embedded track was
    not checked, by path."""
    unchecked: dict[str, str] = {}
    by_title: dict[str, list[dict]] = {}
    for record in records:
        if not can_decode(record) or record["japanese_dialogue_subtitle_ordinal"] is None:
            continue
        if record["episode"] is None:
            unchecked[record["path"]] = "no episode number"
        elif not record["duration_s"] or windows(record["duration_s"], windows_per_episode) is None:
            unchecked[record["path"]] = f"too short for {windows_per_episode} windows"
        else:
            by_title.setdefault(record["title"], []).append(record)
            unchecked[record["path"]] = f"not among the {per_title} episodes per title checked"
    for title, candidates in sorted(by_title.items()):
        candidates.sort(key=sort_key)
        taken: set[tuple] = set()

        def accept(record: dict) -> bool:
            number = (record["season"], record["episode"])
            if number in taken:
                unchecked[record["path"]] = "repeats the season and episode number of an episode taken"
                return False
            problem = check_completeness(ffmpeg, record)
            print(f"decode check {title} {record['file']}: {problem or 'complete'}", file=sys.stderr, flush=True)
            if problem:
                return False
            taken.add(number)
            return True

        walk(candidates, per_title, accept)
    return unchecked


def summarize(records: Sequence[Mapping]) -> dict[str, dict]:
    """Per title: episodes, how many are readable, have Japanese audio, an
    incomplete Japanese audio stream, a Japanese dialogue track or are
    eligible, and the locked title."""
    titles: dict[str, dict] = {}
    for record in records:
        name = record["title"] or f"(no title) {record['file']}"
        entry = titles.setdefault(name, {
            "episodes": 0, "readable": 0, "japanese_audio": 0, "audio_incomplete": 0,
            "japanese_dialogue_subtitle": 0, "eligible": 0, "locked_title": None,
        })
        entry["episodes"] += 1
        entry["readable"] += record["readable"]
        entry["japanese_audio"] += record["japanese_audio_ordinal"] is not None
        entry["audio_incomplete"] += bool(record["audio_decode_errors"] or record["unwritten_blocks"])
        entry["japanese_dialogue_subtitle"] += record["japanese_dialogue_subtitle_ordinal"] is not None
        entry["eligible"] += record["eligible"]
        entry["locked_title"] = entry["locked_title"] or record["locked_title"]
    return dict(sorted(titles.items()))


def tool_version(tool: str) -> str:
    completed = subprocess.run([tool, "-version"], capture_output=True, check=True, text=True)
    return completed.stdout.splitlines()[0]


def build_inventory(
    media: Sequence[Path], skip: Sequence[str], out: Path, locked: LockedTitles,
    decode_per_title: int | None = None, windows_per_episode: int = DEFAULT_WINDOWS,
) -> dict:
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise SystemExit("ffprobe and ffmpeg must be on PATH")
    found, skipped = find_media(media, skip)
    records = []
    for root, path in found:
        records.append(inventory_record(root, path, probe(ffprobe, path), locked))
        print(f"\rprobed {len(records)} of {len(found)} files", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if decode_per_title is None:
        unchecked: dict[str, str] = {}
        for position, record in enumerate(records, start=1):
            if can_decode(record):
                check_completeness(ffmpeg, record)
            print(f"\rdecode checked {position} of {len(records)} files", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    else:
        unchecked = bounded_decode(ffmpeg, records, decode_per_title, windows_per_episode)
    for record in records:
        finish_record(record, unchecked.get(record["path"]))
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ffprobe_version": tool_version(ffprobe),
        "ffmpeg_version": tool_version(ffmpeg),
        "locked_titles_sha256": locked.sha256,
        "media": [str(Path(path)) for path in media],
        "skip_globs": list(skip),
        "skipped_folders": skipped,
        "decode_check": {
            "episodes_per_title": decode_per_title,
            "windows_per_episode": windows_per_episode if decode_per_title is not None else None,
            "unwritten_block_offsets": list(UNWRITTEN_OFFSETS),
            "unwritten_block_bytes": UNWRITTEN_BLOCK,
            "rule": (
                "every readable episode with Japanese audio" if decode_per_title is None else
                "per title, selection.walk over the episodes with an embedded Japanese dialogue track, "
                "an episode number and room for windows_per_episode windows, sorted by "
                "selection.sort_key, until episodes_per_title are complete"
            ),
        },
        "titles": summarize(records),
        "files": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(inventory, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(out)
    return inventory


def load_inventory(path: Path) -> dict:
    inventory = json.loads(Path(path).read_text(encoding="utf-8"))
    if inventory.get("schema_version") not in READABLE_SCHEMAS:
        raise ValueError(f"unsupported inventory schema {inventory.get('schema_version')!r}")
    return inventory


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--media", type=Path, action="append", required=True,
                        help="folder (searched recursively) or video file; repeatable")
    parser.add_argument("--skip", action="append", default=[],
                        help="glob of folder names to skip with everything below them; repeatable")
    parser.add_argument("--out", type=Path, required=True,
                        help="inventory JSON to write (never an inventory a built set records)")
    parser.add_argument("--decode-per-title", type=int,
                        help="check completeness only until this many episodes per title are complete")
    parser.add_argument("--windows-per-episode", type=int, default=DEFAULT_WINDOWS,
                        help="windows an episode must have room for in the bounded check")
    parser.add_argument("--state", type=Path, help=f"E0 rebuild state (default: <E0 root>/{STATE_FILE})")
    args = parser.parse_args(argv)
    if args.decode_per_title is not None and args.decode_per_title < 1:
        parser.error("--decode-per-title must be at least 1")
    locked = locked_titles(load_fixtures(args.state or e0_boundary().root / STATE_FILE))
    inventory = build_inventory(args.media, args.skip, args.out, locked, args.decode_per_title,
                                args.windows_per_episode)
    for folder in inventory["skipped_folders"]:
        print(f"skipped folder {folder}")
    print(f"{'title':40} {'episodes':>8} {'readable':>8} {'jpn audio':>9} {'incomplete':>10} "
          f"{'jpn dialogue sub':>16} {'eligible':>8}  locked")
    for title, entry in inventory["titles"].items():
        print(
            f"{title[:40]:40} {entry['episodes']:8d} {entry['readable']:8d} "
            f"{entry['japanese_audio']:9d} {entry['audio_incomplete']:10d} "
            f"{entry['japanese_dialogue_subtitle']:16d} {entry['eligible']:8d}  {entry['locked_title'] or ''}"
        )
    for record in inventory["files"]:
        if record["locked_title"]:
            continue
        if not record["readable"] or record["audio_decode_errors"] or record["unwritten_blocks"]:
            print(f"listed, not used: {record['file']}: {record['reason']}")
        elif record["title_basis"] == "folder" and record["title_keys"]["file"] and not names_agree(
            record["title_keys"]["file"], record["title"]
        ):
            print(f"title {record['title']!r} from its folder, file name reads "
                  f"{record['title_keys']['file']!r}: {record['file']}")
    eligible = sum(record["eligible"] for record in inventory["files"])
    print(f"wrote {len(inventory['files'])} files ({eligible} eligible) to {args.out}")


if __name__ == "__main__":
    main()
