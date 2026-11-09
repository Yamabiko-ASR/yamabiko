"""Music and effects beds for the synthetic ASR bench set.

Beds come from development titles only: the episodes of the D1-D8 fixtures
plus the other episodes of the same series found next to them in the same
folder. A fixture whose title is a locked confirmation title (L1-L4, listed
privately, see ``titles``) is excluded, in any season and under any of the
title's names, so the locked set is never tuned on, not even through
background music.

Each source episode's Japanese subtitle track and audio stream are decoded in
one FFmpeg pass (audio downmixed to 16 kHz mono). Stretches of at least 4 s
that lie at least 1 s from every subtitle cue and outside the first and last
90 s (opening and ending songs) are cut into 4-12 s beds; beds quieter than
-50 dBFS RMS (silence) are dropped. No VAD or ASR model screens the beds.

Beds are cached under ``bench_root()/beds``: ``beds.json`` plus one 16 kHz
mono float32 file per bed. ``python -m scripts.asr_bench.beds`` builds the
cache (``--rebuild`` replaces it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .build_silver import STATE_FILE, Fixture, e0_boundary, load_fixtures, locked_titles
from .sets import SAMPLE_RATE_HZ, bench_root, write_audio

BEDS_SCHEMA = "asr-bench-beds-1.0"
RULES = {
    "edge_skip_s": 90.0,
    "cue_margin_s": 1.0,
    "min_stretch_s": 4.0,
    "min_bed_s": 4.0,
    "max_bed_s": 12.0,
    "min_rms_dbfs": -50.0,
}
VIDEO_SUFFIXES = (".mkv", ".mp4")
# The episode number follows " - ", a space, or "SnnE"; the text before it
# names the series (season included). Beds group episodes by series; the
# locked exclusion works on whole titles (``titles.title_key``).
_EPISODE_RE = re.compile(
    r"^(?P<series>.+?)(?:\s+-\s+|\s+|(?<=S\d\d)E)(?P<episode>\d{2,3})(?=[\s\[(.]|v\d|$)"
)
_NOT_DIALOGUE_RE = re.compile(r"sign|song|forced", re.IGNORECASE)


@dataclass(frozen=True)
class Bed:
    bed_id: str
    source_file: str
    start_s: float
    end_s: float
    rms_dbfs: float
    audio: str
    sample_count: int


@dataclass(frozen=True)
class BedSource:
    path: Path
    series: str
    audio_stream_ordinal: int
    subtitle_stream_ordinal: int | None
    fixture_label: str | None


def beds_root() -> Path:
    return bench_root() / "beds"


def series_key(file_name: str) -> str | None:
    match = _EPISODE_RE.match(Path(file_name).stem)
    return match["series"].strip() if match else None


def bed_sources(fixtures: Mapping[str, Fixture]) -> tuple[list[BedSource], list[str]]:
    """Episodes eligible for beds and the development labels excluded
    because their title is a locked confirmation title."""
    titles = locked_titles(fixtures)
    locked = [fixture for fixture in fixtures.values() if fixture.locked]
    locked_dirs = {fixture.source.parent for fixture in locked}
    locked_files = {fixture.source.name for fixture in locked}
    series: dict[tuple[Path, str], list[Fixture]] = {}
    excluded: list[str] = []
    for label, fixture in sorted(fixtures.items()):
        if fixture.locked or not label.startswith("D"):
            continue
        key = series_key(fixture.source.name)
        if key is None:
            raise RuntimeError(f"{label}: cannot read a series name from its source file")
        if titles.path_title(fixture.source) or fixture.source.parent in locked_dirs:
            excluded.append(label)
            continue
        series.setdefault((fixture.source.parent, key), []).append(fixture)
    sources: list[BedSource] = []
    for (folder, key), members in sorted(series.items()):
        by_name = {member.source.name: member for member in members}
        audio_ordinals = {member.audio_stream_ordinal for member in members}
        if len(audio_ordinals) != 1:
            raise RuntimeError(f"{key}: fixtures disagree on the audio stream")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in VIDEO_SUFFIXES or series_key(path.name) != key:
                continue
            if path.name in locked_files:
                raise AssertionError(f"{path.name} is a locked fixture source")
            member = by_name.get(path.name)
            sources.append(BedSource(
                path=path,
                series=key,
                audio_stream_ordinal=next(iter(audio_ordinals)),
                subtitle_stream_ordinal=member.subtitle_stream_ordinal if member else None,
                fixture_label=member.label if member else None,
            ))
    return sources, excluded


def _streams(ffprobe: str, path: Path) -> list[dict]:
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-of", "json", "-show_entries",
         "stream=index,codec_type:stream_tags=language,title", str(path)],
        capture_output=True, check=True,
    )
    return json.loads(completed.stdout)["streams"]


def _tags(stream: Mapping) -> tuple[str, str]:
    tags = stream.get("tags", {})
    return tags.get("language", ""), tags.get("title", "")


def pick_subtitle(subtitles: Sequence[Mapping], preferred_title: str | None) -> int | None:
    """Ordinal of the Japanese dialogue track: language ``jpn``, title not
    signs/songs/forced; a track titled like the series' fixture track wins."""
    candidates = [
        ordinal for ordinal, stream in enumerate(subtitles)
        if _tags(stream)[0] == "jpn" and not _NOT_DIALOGUE_RE.search(_tags(stream)[1])
    ]
    for ordinal in candidates:
        if _tags(subtitles[ordinal])[1] == preferred_title:
            return ordinal
    return candidates[0] if candidates else None


def _decode(
    ffmpeg: str, path: Path, audio_ordinal: int, subtitle_ordinal: int, scratch: Path
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    from scripts.e0_evidence.subtitle_proposals import _parse_srt

    audio_path, subtitle_path = scratch / "audio.f32", scratch / "subtitle.srt"
    subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(path),
         "-map", f"0:a:{audio_ordinal}", "-ac", "1", "-ar", str(SAMPLE_RATE_HZ),
         "-c:a", "pcm_f32le", "-f", "f32le", str(audio_path),
         "-map", f"0:s:{subtitle_ordinal}", "-f", "srt", str(subtitle_path)],
        check=True,
    )
    pcm = np.fromfile(audio_path, dtype="<f4")
    cues = [
        (cue.original_start_ms, cue.original_end_ms)
        for cue in _parse_srt(subtitle_path.read_bytes())
    ]
    return pcm, cues


def free_stretches(cues_ms: Sequence[tuple[int, int]], duration_s: float) -> list[tuple[float, float]]:
    """Spans of at least ``min_stretch_s`` that keep ``cue_margin_s`` from
    every cue and ``edge_skip_s`` from both ends."""
    low, high = RULES["edge_skip_s"], duration_s - RULES["edge_skip_s"]
    margin = RULES["cue_margin_s"]
    blocked = sorted((start / 1000 - margin, end / 1000 + margin) for start, end in cues_ms)
    stretches: list[tuple[float, float]] = []
    cursor = low
    for start, end in blocked:
        if start >= high:
            break
        if start - cursor >= RULES["min_stretch_s"]:
            stretches.append((cursor, start))
        cursor = max(cursor, end)
    if high - cursor >= RULES["min_stretch_s"]:
        stretches.append((cursor, high))
    return stretches


def cut_stretch(start: float, end: float) -> list[tuple[float, float]]:
    """Equal pieces of at most ``max_bed_s``; a stretch of at least 4 s
    never yields a piece shorter than ``min_bed_s``."""
    count = math.ceil((end - start) / RULES["max_bed_s"])
    piece = (end - start) / count
    return [(start + index * piece, start + (index + 1) * piece) for index in range(count)]


def rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 0 else -math.inf


def build_beds(out_dir: Path, state_path: Path | None = None) -> dict:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must be on PATH")
    state_path = state_path or e0_boundary().root / STATE_FILE
    fixtures = load_fixtures(state_path)
    sources, excluded = bed_sources(fixtures)
    if excluded:
        print(f"excluded (locked title): {', '.join(excluded)}")
    # Replace only what a previous build wrote.
    (out_dir / "beds.json").unlink(missing_ok=True)
    if (out_dir / "audio").exists():
        shutil.rmtree(out_dir / "audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    # The fixture track's title for each series guides extra episodes.
    fixture_titles: dict[str, str] = {}
    for source in sources:
        if source.subtitle_stream_ordinal is not None:
            subtitles = [s for s in _streams(ffprobe, source.path) if s["codec_type"] == "subtitle"]
            fixture_titles.setdefault(
                source.series, _tags(subtitles[source.subtitle_stream_ordinal])[1]
            )
    beds: list[Bed] = []
    source_records: list[dict] = []
    with tempfile.TemporaryDirectory(dir=out_dir) as scratch_name:
        scratch = Path(scratch_name)
        for source in sources:
            streams = _streams(ffprobe, source.path)
            subtitles = [s for s in streams if s["codec_type"] == "subtitle"]
            audios = [s for s in streams if s["codec_type"] == "audio"]
            record = {
                "file": source.path.name,
                "series": source.series,
                "fixture_label": source.fixture_label,
                "audio_stream_ordinal": source.audio_stream_ordinal,
                "subtitle_stream_ordinal": source.subtitle_stream_ordinal,
                "bed_count": 0,
                "skipped": None,
            }
            source_records.append(record)
            subtitle = source.subtitle_stream_ordinal
            if subtitle is None:
                subtitle = pick_subtitle(subtitles, fixture_titles.get(source.series))
                if subtitle is None:
                    record["skipped"] = "no Japanese dialogue subtitle track"
                    print(f"skip {source.path.name}: {record['skipped']}")
                    continue
                record["subtitle_stream_ordinal"] = subtitle
            audio_language = _tags(audios[source.audio_stream_ordinal])[0]
            if audio_language not in ("jpn", ""):
                record["skipped"] = f"audio stream language is {audio_language}"
                print(f"skip {source.path.name}: {record['skipped']}")
                continue
            record["subtitle_title"] = _tags(subtitles[subtitle])[1]
            pcm, cues = _decode(ffmpeg, source.path, source.audio_stream_ordinal, subtitle, scratch)
            duration_s = pcm.size / SAMPLE_RATE_HZ
            record.update(duration_s=round(duration_s, 3), cue_count=len(cues))
            name_hash = hashlib.sha256(source.path.name.encode("utf-8")).hexdigest()[:8]
            for stretch in free_stretches(cues, duration_s):
                for start, end in cut_stretch(*stretch):
                    first = round(start * SAMPLE_RATE_HZ)
                    last = round(end * SAMPLE_RATE_HZ)
                    samples = pcm[first:last]
                    level = rms_dbfs(samples)
                    if level < RULES["min_rms_dbfs"]:
                        continue
                    bed_id = f"bed-{name_hash}-{round(start * 1000):07d}"
                    audio = f"audio/{bed_id}.f32"
                    beds.append(Bed(
                        bed_id=bed_id,
                        source_file=source.path.name,
                        start_s=round(first / SAMPLE_RATE_HZ, 4),
                        end_s=round(last / SAMPLE_RATE_HZ, 4),
                        rms_dbfs=round(level, 2),
                        audio=audio,
                        sample_count=write_audio(out_dir / audio, samples),
                    ))
                    record["bed_count"] += 1
            print(f"{source.path.name}: {len(cues)} cues, {record['bed_count']} beds", flush=True)
    manifest = {
        "schema_version": BEDS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": RULES,
        "excluded_fixture_labels": excluded,
        "sources": source_records,
        "beds": [asdict(bed) for bed in beds],
    }
    (out_dir / "beds.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(beds)} beds from {sum(r['bed_count'] > 0 for r in source_records)} episodes to {out_dir}")
    return manifest


def load_beds(out_dir: Path | None = None) -> tuple[dict, list[Bed]]:
    """Read the bed cache, building it when absent or built under other rules."""
    out_dir = out_dir or beds_root()
    manifest_path = out_dir / "beds.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != BEDS_SCHEMA or manifest.get("rules") != RULES:
            manifest = None
    if manifest is None:
        manifest = build_beds(out_dir)
    return manifest, [Bed(**raw) for raw in manifest["beds"]]


def load_bed_audio(bed: Bed, out_dir: Path | None = None) -> np.ndarray:
    samples = np.fromfile((out_dir or beds_root()) / bed.audio, dtype="<f4")
    if samples.size != bed.sample_count:
        raise ValueError(f"{bed.bed_id}: audio has {samples.size} samples, expected {bed.sample_count}")
    return samples


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=beds_root())
    parser.add_argument("--rebuild", action="store_true", help="replace an existing cache")
    parser.add_argument("--state", type=Path, help=f"E0 rebuild state (default: <E0 root>/{STATE_FILE})")
    args = parser.parse_args(argv)
    if args.rebuild or not (args.out / "beds.json").is_file():
        build_beds(args.out, args.state)
    else:
        _, beds = load_beds(args.out)
        print(f"{len(beds)} beds cached in {args.out}")


if __name__ == "__main__":
    main()
