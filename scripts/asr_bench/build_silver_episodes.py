"""Build the episode silver set: real anime audio referenced by each episode's
own Japanese subtitle track, many episodes across the development titles.

    python -m scripts.asr_bench.build_silver_episodes --out <bench_root>/sets/<set id>
        [--set-id ID] [--episodes-per-title N] [--windows-per-episode K] [--replace]
        [--inventory <json>] [--external-subs <folder>] [--reference-check <json>]

The defaults (8 episodes per title, 2 windows per episode, inventory
``bench_root()/inventory/episodes-v3.json``, external subtitles
``bench_root()/external-subs``, reference check
``bench_root()/inventory/reference-check.json``) rebuild silver-v4. ``--out``
is required; a folder that already holds a set is refused unless
``--replace`` is given. The set id is the folder name unless ``--set-id``
names one.

Input is the inventory written by ``scripts.asr_bench.inventory``. Eligible
episodes there (title known and not locked, readable, a Japanese audio
stream found complete, a Japanese dialogue subtitle track) also need room
for the windows. An episode without an embedded Japanese dialogue track is
eligible when an external Japanese subtitle for that exact video file is
accepted in ``<external-subs>/manifest.json`` (schema
``asr-bench-external-subs-1.0`` or ``-1.1``: per episode its ``title``,
``video_file``, ``video_size``, ``subtitle_file`` relative to the folder,
``subtitle_sha256``, source, alignment and quality, ``accepted``, and in
1.1 ``audio_completeness``); the subtitle's digest and the video's size must
match the manifest, and when the inventory did not check the episode's
audio for completeness the manifest entry must record a clean check. An
embedded track always takes precedence. An episode with an embedded track
whose release carries more than one audio stream (a dub) is used only when
``scripts.asr_bench.check_reference`` passed it, and every other such
episode of its title, for the same file path and size: the Japanese cues
must follow the Japanese audio, not the dub, and be timed to it. Per title
at most N eligible episodes are taken, evenly spaced through the title's
episodes sorted by season and episode number (``selection.spaced``).

Each episode gives K 240 s windows inside [start + 120 s, end - 120 s], so
opening and ending songs stay out: two windows are centred at 30% and 70% of
that span (``selection.windows``). FFmpeg decodes the Japanese audio stream
to 16 kHz mono float32 (``-ac 1 -ar 16000``) and the subtitle tracks to SRT
in one pass; each window is cut from the decoded samples. An ASS dialogue
track first loses its on-screen text: events in a style named signs, songs
or forced, and events positioned with ``\\pos`` or ``\\move``.

Reference lines are the dialogue cues overlapping a window, clipped to it,
with times relative to the window start. Cue text is cleaned as in
``build_silver`` (markup, format characters, parenthesized SDH
annotations). Dropped: song lyrics (a cue holding ``♪``, ``♫`` or ``♬``, or a cue
whose text also appears, at an overlapping time, in a Japanese signs, songs
or forced track; such a track whose SRT conversion cannot be parsed is left
out of this check and recorded), cues left without text by
``normalize_minimal``, and cues clipped to under 0.3 s. Items are labelled
``title``, ``episode_file``,
``window_index``, ``cluster`` (the episode, so statistics resample the
windows of an episode together) and ``subtitle_source`` (``embedded`` or
``external``). An episode FFmpeg cannot decode (an incomplete download) is
listed and skipped. No VAD or ASR model selects or filters anything. Each
chosen episode's path, and an external subtitle's path and source file name,
is checked again against the private locked-title list, itself checked
against the E0 locked fixtures (the E0 private root must be set), and the
inventory must have been built under the same list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.e0_evidence.normalize_ja import normalize_minimal

from .build_silver import STATE_FILE, TEXT_CLEANING, clean_cue_text, e0_boundary, load_fixtures
from .build_silver import locked_titles, mark_overlaps
from .external_subs import READABLE_MANIFEST_SCHEMAS as EXTERNAL_SUBS_SCHEMAS
from .external_subs import root as external_subs_dir
from .inventory import NO_DIALOGUE_SUBTITLE, NOT_DIALOGUE_RE, load_inventory, tool_version
from .inventory import default_path as default_inventory_path
from .selection import DEFAULT_WINDOWS, EDGE_SKIP_S, WINDOW_S, sort_key, spaced, window_centres, windows
from .sets import SAMPLE_RATE_HZ, bench_root, write_audio, write_set
from .titles import LockedTitles

DEFAULT_EPISODES_PER_TITLE = 8
RULES = {
    "max_episodes_per_title": DEFAULT_EPISODES_PER_TITLE,
    "window_s": WINDOW_S,
    "edge_skip_s": EDGE_SKIP_S,
    "window_centres": window_centres(DEFAULT_WINDOWS),
    "min_line_s": 0.3,
    "lyric_marks": "♪♫♬",
}
RULE_TEXT = {
    "eligibility": (
        "inventory-eligible (title known and not a locked title, readable, the Japanese "
        "audio found complete, a Japanese dialogue subtitle track; or, without an "
        "embedded dialogue track, an accepted external Japanese subtitle aligned to that "
        "video file, its audio found complete by the inventory or the external-subtitle "
        "builder), for an embedded track of a release with more than one audio stream "
        "a passing reference check of this and every other such episode of the title, "
        "and long enough for the windows"
    ),
    "selection": (
        "per title at most max_episodes_per_title episodes, evenly spaced through the "
        "title's eligible episodes sorted by season, episode number and file name "
        "(index (2i + 1) * n // (2 * max))"
    ),
    "windows": (
        "window_s windows centred at window_centres of [edge_skip_s, duration - "
        "edge_skip_s], cut from the decoded 16 kHz mono float32 audio"
    ),
    "lines": (
        "dialogue cues overlapping the window, clipped to it, times relative to its "
        "start; dropped: cues holding a lyric mark, cues whose normalize_minimal text "
        "also appears at an overlapping time in a Japanese signs, songs or forced "
        "track, cues empty after normalize_minimal, cues clipped to under min_line_s"
    ),
    "ass_dialogue": (
        "an ASS dialogue track loses events in a style whose name says signs, songs or "
        "forced and events positioned with \\pos or \\move before conversion to SRT"
    ),
    "text_cleaning": TEXT_CLEANING,
}
CHECK_SCHEMA = "asr-bench-reference-check-1.0"
ASS_CODECS = ("ass", "ssa")
_POSITIONED_RE = re.compile(r"\\(?:pos|move)\s*\(")


def default_check_path() -> Path:
    return bench_root() / "inventory" / "reference-check.json"


def load_reference_checks(path: Path) -> tuple[dict[str, dict], str | None]:
    """``check_reference`` results by video path, and the file's digest;
    nothing when there is no file."""
    if not path.exists():
        return {}, None
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema_version") != CHECK_SCHEMA:
        raise SystemExit(f"{path}: schema_version is not {CHECK_SCHEMA}")
    return {episode["path"]: episode for episode in report["episodes"]}, hashlib.sha256(raw).hexdigest()


def reference_problem(record: Mapping, result: Mapping | None) -> str | None:
    """Why the reference check does not clear this episode, or None."""
    if result is None:
        return "no reference check result (run scripts.asr_bench.check_reference)"
    try:
        size = Path(record["path"]).stat().st_size
    except OSError as error:
        return f"unreadable: {error.strerror}"
    if size != result["size"]:
        return "the reference check ran on another version of the file"
    if not result["passed"]:
        return f"reference check failed: {'; '.join(result['problems'])}"
    return None


def load_external_subs(directory: Path) -> tuple[dict[tuple[str, str], dict], str | None]:
    """The accepted external subtitles of ``directory/manifest.json`` keyed by
    (title, video file name), and the manifest's digest; nothing when there
    is no manifest."""
    path = directory / "manifest.json"
    if not path.exists():
        return {}, None
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") not in EXTERNAL_SUBS_SCHEMAS:
        raise SystemExit(f"{path}: schema_version is not one of {EXTERNAL_SUBS_SCHEMAS}")
    accepted: dict[tuple[str, str], dict] = {}
    for entry in manifest["episodes"]:
        if entry.get("accepted") is not True:
            continue
        key = (entry["title"], entry["video_file"])
        if key in accepted:
            raise SystemExit(f"{path}: {key} is accepted twice")
        accepted[key] = entry
    return accepted, hashlib.sha256(raw).hexdigest()


def read_external(
    record: Mapping, entry: Mapping, directory: Path, locked: LockedTitles
) -> tuple[bytes | None, str | None]:
    """The bytes of an accepted external subtitle, read once and checked
    against the manifest digest, or why it cannot serve this episode. When
    the inventory did not check the episode's audio for completeness, the
    manifest entry must record a clean check."""
    subtitle = directory / entry["subtitle_file"]
    if not subtitle.resolve().is_relative_to(directory.resolve()):
        return None, f"external subtitle {entry['subtitle_file']} lies outside {directory}"
    for name in (subtitle, Path(entry["source"]["file"]), Path(entry["source"].get("zip") or "")):
        if (title_locked := locked.path_title(name)) is not None:
            raise AssertionError(f"external subtitle {name} names the locked title {title_locked}")
    if record["audio_decode_errors"] is None:
        checked = entry.get("audio_completeness") or {}
        if checked.get("audio_decode_errors") != 0 or checked.get("unwritten_blocks"):
            return None, "audio not found complete by the inventory or the external-subtitle builder"
    try:
        data = subtitle.read_bytes()
    except OSError:
        return None, f"external subtitle {entry['subtitle_file']} missing"
    if hashlib.sha256(data).hexdigest() != entry["subtitle_sha256"]:
        return None, f"external subtitle {entry['subtitle_file']} differs from its manifest digest"
    try:
        size = Path(record["path"]).stat().st_size
    except OSError as error:
        return None, f"unreadable: {error.strerror}"
    if size != entry["video_size"]:
        return None, "external subtitle was aligned to a video file of another size"
    return data, None


def without_sign_events(script: str) -> tuple[str, Counter]:
    """An ASS script without its on-screen text events (a style named signs,
    songs or forced, or positioned with \\pos or \\move), and the counts of
    dropped events."""
    kept: list[str] = []
    dropped: Counter = Counter()
    fields: list[str] = []
    in_events = False
    for line in script.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_events = stripped.lower() == "[events]"
        elif in_events and stripped.lower().startswith("format:"):
            fields = [field.strip().lower() for field in stripped.split(":", 1)[1].split(",")]
        elif in_events and fields and stripped.startswith("Dialogue:"):
            event = dict(zip(fields, stripped.split(":", 1)[1].split(",", len(fields) - 1)))
            if NOT_DIALOGUE_RE.search(event.get("style", "")):
                dropped["sign_style"] += 1
                continue
            if _POSITIONED_RE.search(event.get("text", "")):
                dropped["positioned"] += 1
                continue
        kept.append(line)
    return "".join(kept), dropped


def episode_tags(records: Sequence[Mapping]) -> list[str]:
    """``eNN`` per episode (``sSSeNN`` when the title spans seasons); an
    episode without a number uses its position in the sorted list."""
    seasons = {record["season"] or 0 for record in records}
    tags = []
    for position, record in enumerate(records, start=1):
        number = record["episode"] if record["episode"] is not None else position
        tag = f"e{number:02d}"
        tags.append(f"s{record['season'] or 0:02d}{tag}" if len(seasons) > 1 else tag)
    if len(set(tags)) != len(tags):
        raise RuntimeError(f"{records[0]['title']}: episode numbers collide: {tags}")
    return tags


def _decode(
    ffmpeg: str, record: Mapping, scratch: Path, external_srt: bytes | None
) -> tuple[np.ndarray, tuple, list[tuple], Counter, dict[int, str]]:
    """Decoded audio, dialogue cues (from the embedded track, or the
    external SRT when given), the non-dialogue Japanese tracks' cues as
    (start_ms, end_ms, normalized text), the ASS events dropped as
    on-screen text, and the non-dialogue tracks whose SRT conversion cannot
    be parsed (left out of the song filter) with the parser's message."""
    from scripts.e0_evidence.subtitle_proposals import _parse_srt

    audio_path, dialogue_path, ass_path = scratch / "audio.f32", scratch / "dialogue.srt", None
    extra = {ordinal: scratch / f"track{ordinal}.srt" for ordinal in record["japanese_non_dialogue_subtitle_ordinals"]}
    command = [
        ffmpeg, "-v", "error", "-nostdin", "-y", "-i", record["path"],
        "-map", f"0:a:{record['japanese_audio_ordinal']}", "-ac", "1", "-ar", str(SAMPLE_RATE_HZ),
        "-c:a", "pcm_f32le", "-f", "f32le", str(audio_path),
    ]
    if external_srt is None:
        ordinal = record["japanese_dialogue_subtitle_ordinal"]
        if record["subtitles"][ordinal]["codec"] in ASS_CODECS:
            ass_path = scratch / "dialogue.ass"
            command += ["-map", f"0:s:{ordinal}", "-c:s", "copy", "-f", "ass", str(ass_path)]
        else:
            command += ["-map", f"0:s:{ordinal}", "-f", "srt", str(dialogue_path)]
    for ordinal, path in extra.items():
        command += ["-map", f"0:s:{ordinal}", "-f", "srt", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    dropped: Counter = Counter()
    if ass_path is not None:
        script, dropped = without_sign_events(ass_path.read_text(encoding="utf-8-sig"))
        ass_path.write_text(script, encoding="utf-8")
        subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(ass_path), "-f", "srt", str(dialogue_path)],
            check=True, capture_output=True,
        )
    pcm = np.fromfile(audio_path, dtype="<f4")
    dialogue = _parse_srt(dialogue_path.read_bytes() if external_srt is None else external_srt)
    not_dialogue, unreadable = [], {}
    for ordinal, path in extra.items():
        if not path.stat().st_size:
            continue
        try:
            cues = _parse_srt(path.read_bytes())
        except ValueError as error:
            unreadable[ordinal] = str(error)
            continue
        not_dialogue += [
            (cue.original_start_ms, cue.original_end_ms, normalize_minimal(clean_cue_text(cue.raw_text)))
            for cue in cues
        ]
    return pcm, dialogue, not_dialogue, dropped, unreadable


def window_lines(
    cues: Sequence, not_dialogue: Sequence[tuple[int, int, str]], start_s: float, end_s: float
) -> tuple[list[dict], Counter]:
    lines: list[dict] = []
    stats: Counter = Counter()
    for cue in cues:
        cue_start, cue_end = cue.original_start_ms / 1000, cue.original_end_ms / 1000
        if cue_end <= start_s or cue_start >= end_s:
            continue
        stats["cues"] += 1
        if any(mark in cue.raw_text for mark in RULES["lyric_marks"]):
            stats["dropped_lyric"] += 1
            continue
        text = clean_cue_text(cue.raw_text)
        normalized = normalize_minimal(text)
        if not normalized:
            stats["dropped_empty"] += 1
            continue
        if any(
            other_text == normalized
            and other_start < cue.original_end_ms and cue.original_start_ms < other_end
            for other_start, other_end, other_text in not_dialogue
        ):
            stats["dropped_signs_songs_track"] += 1
            continue
        line_start = round(max(cue_start, start_s) - start_s, 3)
        line_end = round(min(cue_end, end_s) - start_s, 3)
        if line_end - line_start < RULES["min_line_s"]:
            stats["dropped_short"] += 1
            continue
        lines.append({
            "line_id": f"c{cue.source_ordinal:04d}",
            "start_s": line_start,
            "end_s": line_end,
            "text": text,
            "source": "subtitle",
            "overlap": False,
        })
    mark_overlaps(lines)
    stats["lines"] = len(lines)
    return lines, stats


def _relative_to_root(path: Path) -> str:
    """``path`` relative to the bench root (POSIX), or absolute when outside it."""
    try:
        return Path(path).resolve().relative_to(bench_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def build(
    out_dir: Path, inventory_path: Path, locked: LockedTitles, external_dir: Path, check_path: Path,
    per_title: int = DEFAULT_EPISODES_PER_TITLE, windows_per_episode: int = DEFAULT_WINDOWS,
    set_id: str | None = None, replace: bool = False,
) -> None:
    if (out_dir / "set.json").exists() and not replace:
        raise SystemExit(f"{out_dir} already holds a set; pass --replace to rebuild it in place")
    rules = {**RULES, "max_episodes_per_title": per_title, "window_centres": window_centres(windows_per_episode)}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg must be on PATH")
    inventory_bytes = Path(inventory_path).read_bytes()
    inventory = load_inventory(inventory_path)
    if inventory.get("locked_titles_sha256") != locked.sha256:
        raise SystemExit(
            f"{inventory_path} was built under another locked-title list; rebuild the inventory"
        )

    accepted, external_sha256 = load_external_subs(external_dir)
    checks, check_sha256 = load_reference_checks(check_path)
    checked: dict[str, Counter] = {}
    for record in inventory["files"]:
        if record["reason"] is None and len(record["audio"]) > 1:
            problem = reference_problem(record, checks.get(record["path"]))
            checked.setdefault(record["title"], Counter())["failed" if problem else "passed"] += 1
    external_for: dict[str, tuple[dict, bytes]] = {}
    skipped: list[dict] = []
    by_title: dict[str, list[dict]] = {}
    reasons: dict[str, Counter] = {}
    for record in inventory["files"]:
        title = record["title"] or f"(no title) {record['file']}"
        reason = record["reason"]
        if reason == NO_DIALOGUE_SUBTITLE and (entry := accepted.get((record["title"], record["file"]))):
            data, reason = read_external(record, entry, external_dir, locked)
            if reason is None:
                external_for[record["path"]] = entry, data
        elif reason is None and len(record["audio"]) > 1:
            reason = reference_problem(record, checks.get(record["path"]))
            if reason is None and (failed := checked[record["title"]]["failed"]):
                reason = (
                    f"reference check failed for {failed} of {sum(checked[record['title']].values())} "
                    f"checked episodes of this title"
                )
        if reason is None and (
            record["duration_s"] is None or windows(record["duration_s"], windows_per_episode) is None
        ):
            reason = f"too short for {windows_per_episode} windows"
        reasons.setdefault(title, Counter())[reason.split(":")[0] if reason else "eligible"] += 1
        if reason is not None:
            skipped.append({"file": record["file"], "title": title, "reason": reason})
            continue
        by_title.setdefault(title, []).append(record)
    excluded_titles = {
        title: dict(counts) for title, counts in sorted(reasons.items()) if title not in by_title
    }

    (out_dir / "set.json").unlink(missing_ok=True)
    if (out_dir / "audio").exists():
        shutil.rmtree(out_dir / "audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    episodes: dict[str, dict] = {}
    title_counts: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(dir=out_dir) as scratch_name:
        scratch = Path(scratch_name)
        for title, records in sorted(by_title.items()):
            records.sort(key=sort_key)
            chosen = [records[index] for index in spaced(len(records), per_title)]
            slug = title.replace(" ", "-")
            counts = title_counts[title] = {
                "inventory_episodes": sum(reasons[title].values()),
                "eligible_episodes": len(records),
                "external_subtitle_episodes": sum(record["path"] in external_for for record in records),
                "selected_episodes": 0,
                "items": 0,
                "lines": 0,
            }
            for record, tag in zip(chosen, episode_tags(chosen)):
                if (title_locked := locked.path_title(Path(record["path"]))) is not None:
                    raise AssertionError(f"{record['file']} belongs to the locked title {title_locked}")
                entry, external_srt = external_for.get(record["path"], (None, None))
                try:
                    pcm, cues, not_dialogue, ass_dropped, unreadable = _decode(ffmpeg, record, scratch, external_srt)
                except subprocess.CalledProcessError as error:
                    problem = error.stderr.decode("utf-8", "replace").strip().splitlines()
                    reason = f"decode failed: {problem[-1] if problem else f'exit {error.returncode}'}"
                    skipped.append({"file": record["file"], "title": title, "reason": reason})
                    print(f"{slug}-{tag}: skipped, {reason}", flush=True)
                    continue
                duration_s = pcm.size / SAMPLE_RATE_HZ
                spans = windows(duration_s, windows_per_episode)
                if spans is None:
                    reason = f"decoded audio ({duration_s:.1f} s) too short for {windows_per_episode} windows"
                    skipped.append({"file": record["file"], "title": title, "reason": reason})
                    print(f"{slug}-{tag}: skipped, {reason}", flush=True)
                    continue
                counts["selected_episodes"] += 1
                audio = record["audio"][record["japanese_audio_ordinal"]]
                if entry:
                    source = "external"
                    subtitle_info = {"external_subtitle": {**entry, "manifest_sha256": external_sha256}}
                    subtitle_label = f"external {entry['subtitle_file']!r}"
                else:
                    source = "embedded"
                    subtitle = record["subtitles"][record["japanese_dialogue_subtitle_ordinal"]]
                    subtitle_info = {
                        "subtitle_stream": {key: subtitle[key] for key in ("index", "language", "title", "codec")},
                        "ass_events_dropped": dict(ass_dropped),
                    }
                    subtitle_label = f"subtitle #{subtitle['index']} {subtitle['codec']} {subtitle['title']!r}"
                episode = episodes[record["file"]] = {
                    "title": title,
                    "audio_stream": {key: audio[key] for key in ("index", "language", "title", "channels")},
                    "audio_basis": record["japanese_audio_basis"],
                    "subtitle_source": source,
                    **subtitle_info,
                    "non_dialogue_subtitle_ordinals": record["japanese_non_dialogue_subtitle_ordinals"],
                    **({"non_dialogue_subtitles_unreadable": unreadable} if unreadable else {}),
                    "reference_check": None if record["path"] not in checks or entry else {
                        "passed": checks[record["path"]]["passed"],
                        "japanese_track": checks[record["path"]]["tracks"][
                            str(checks[record["path"]]["japanese_track_ordinal"])
                        ].get("versus"),
                    },
                    "decoded_duration_s": round(duration_s, 3),
                    "cue_count": len(cues),
                    "windows": [],
                }
                for window_index, (start_s, _) in enumerate(spans):
                    first = round(start_s * SAMPLE_RATE_HZ)
                    last = first + round(rules["window_s"] * SAMPLE_RATE_HZ)
                    window_start, window_end = first / SAMPLE_RATE_HZ, last / SAMPLE_RATE_HZ
                    lines, stats = window_lines(cues, not_dialogue, window_start, window_end)
                    item_id = f"{slug}-{tag}-w{window_index}"
                    audio_file = f"audio/{item_id}.f32"
                    items.append({
                        "item_id": item_id,
                        "audio": audio_file,
                        "sample_count": write_audio(out_dir / audio_file, pcm[first:last]),
                        "condition": {
                            "title": title,
                            "episode_file": record["file"],
                            "window_index": window_index,
                            "cluster": f"{slug}-{tag}",
                            "subtitle_source": source,
                        },
                        "lines": lines,
                        "speech_free": [],
                    })
                    episode["windows"].append({
                        "item_id": item_id, "start_s": window_start, "end_s": window_end, **stats,
                    })
                    counts["items"] += 1
                    counts["lines"] += len(lines)
                print(
                    f"{slug}-{tag}: {duration_s:.0f} s, audio #{audio['index']} "
                    f"{audio['language'] or 'untagged'}, {subtitle_label}"
                    + (f", ASS events dropped {dict(ass_dropped)}" if ass_dropped else "")
                    + (f", unreadable non-dialogue tracks {sorted(unreadable)}" if unreadable else "")
                    + ", lines " + " + ".join(str(window["lines"]) for window in episode["windows"]),
                    flush=True,
                )
    written = write_set(
        out_dir,
        set_id=set_id or out_dir.name,
        kind="silver",
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        build={
            "builder": "scripts.asr_bench.build_silver_episodes",
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "inventory_schema": inventory["schema_version"],
            "ffmpeg_version": tool_version(ffmpeg),
            "rules": {**rules, **RULE_TEXT},
            "locked_titles_sha256": locked.sha256,
            "external_subs": None if external_sha256 is None else {
                "manifest": _relative_to_root(external_dir / "manifest.json"),
                "manifest_sha256": external_sha256,
                "accepted_episodes": len(accepted),
                "eligible_episodes": len(external_for),
            },
            "reference_check": None if check_sha256 is None else {
                "path": str(check_path),
                "sha256": check_sha256,
                "episodes": len(checks),
            },
            "excluded_titles": excluded_titles,
            "titles": title_counts,
            "skipped_episodes": skipped,
            "episodes": episodes,
        },
        items=items,
    )
    for title, counts in excluded_titles.items():
        print(f"excluded title {title}: {', '.join(f'{r} ({n})' for r, n in counts.items())}")
    unreadable = sum(skip["reason"].startswith(("unreadable", "decode failed")) for skip in skipped)
    print(f"{unreadable} episodes unreadable or failing to decode, listed in build.skipped_episodes")
    hours = sum(item.duration_s for item in written.items) / 3600
    line_count = sum(len(item.lines) for item in written.items)
    print(
        f"wrote {len(written.items)} items ({hours:.2f} h, {line_count} lines) from "
        f"{len(episodes)} episodes of {len(title_counts)} titles to {out_dir}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="set folder, e.g. <bench root>/sets/<set id>")
    parser.add_argument("--set-id", help="set id (default: the --out folder name)")
    parser.add_argument("--episodes-per-title", type=int, default=DEFAULT_EPISODES_PER_TITLE,
                        help=f"episodes taken per title (default {DEFAULT_EPISODES_PER_TITLE})")
    parser.add_argument("--windows-per-episode", type=int, default=DEFAULT_WINDOWS,
                        help=f"windows cut per episode (default {DEFAULT_WINDOWS})")
    parser.add_argument("--replace", action="store_true", help="rebuild a set folder that already holds one")
    parser.add_argument("--inventory", type=Path, default=default_inventory_path())
    parser.add_argument("--state", type=Path, help=f"E0 rebuild state (default: <E0 root>/{STATE_FILE})")
    parser.add_argument("--external-subs", type=Path, default=external_subs_dir(),
                        help="folder of external Japanese subtitles and their manifest.json")
    parser.add_argument("--reference-check", type=Path, default=default_check_path(),
                        help="check_reference result for releases with more than one audio stream")
    args = parser.parse_args(argv)
    if args.episodes_per_title < 1 or args.windows_per_episode < 1:
        parser.error("--episodes-per-title and --windows-per-episode must be at least 1")
    locked = locked_titles(load_fixtures(args.state or e0_boundary().root / STATE_FILE))
    build(args.out, args.inventory, locked, args.external_subs, args.reference_check,
          args.episodes_per_title, args.windows_per_episode, args.set_id, args.replace)


if __name__ == "__main__":
    main()
