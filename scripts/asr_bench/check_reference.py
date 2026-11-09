"""Model-free check that an episode's embedded Japanese subtitle track follows
its Japanese audio stream, in time and in language, when the release also
carries a dub.

    python -m scripts.asr_bench.check_reference [--title <title> ...] [--inventory <json>] --out <json>

Every eligible episode of the inventory (default
``bench_root()/inventory/episodes-v3.json``) with more than one audio stream
is checked, or only those of the given titles. FFmpeg decodes every audio
stream to 16 kHz mono and every text subtitle track to SRT (an ASS track
loses its on-screen text events, as in ``build_silver_episodes``). A cue
counts when its text survives the bench's cleaning (``clean_cue_text`` and
``normalize_minimal``) and holds no lyric mark, so sound-effect captions and
songs are left out. No VAD or ASR model is involved: the measures use frame
log energy (25 ms frames, 10 ms hop, dB) only.

Music and effects are shared by the Japanese stream and the dub, so where
one stream is at least 6 dB louder than the other, its voice speaks and the
other's does not. For each other audio stream:

- ``japanese_only_coverage``: the share of Japanese-only frames (Japanese
  stream at least 6 dB louder) inside the track's cues;
  ``other_only_coverage``: the same for frames where the other stream is
  at least 6 dB louder. ``coverage_difference`` is the first minus the
  second: positive when the cues mark the Japanese voice rather than the
  dub's.
- ``best_lag_s``: the shift of the cues (-2 s to +2 s in 0.1 s steps) that
  maximises the coverage difference; near 0 when the cues are timed to the
  Japanese audio.

The Japanese track of an episode passes when, against every other audio
stream, the coverage difference over the whole episode is positive at lag 0
and the best lag lies within 0.3 s. The same measures are reported inside
each of the two windows ``build_silver_episodes`` cuts, for reading only:
240 s hold too few exclusive-voice frames for a stable lag (on releases
whose tracks pass, window lags scatter up to 2 s where the coverage
difference barely changes with the shift). Every other text track is
measured over the whole episode as a control (a track timed to the dub
shows a negative difference), with the distance from each Japanese cue
start to the nearest cue start of that track. ``build_silver_episodes``
uses an episode with more than one audio stream only when every checked
episode of its title passed, each for the same file path and size: a
release whose Japanese track is off its audio in one episode is not
trusted in the others. The result is written to ``--out`` (silver-v4 used
``bench_root()/inventory/reference-check.json``), never subtitle text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.e0_evidence.normalize_ja import normalize_minimal

from .build_silver import clean_cue_text
from .build_silver_episodes import ASS_CODECS, CHECK_SCHEMA, RULES, without_sign_events
from .inventory import TEXT_SUBTITLE_CODECS, load_inventory, tool_version
from .inventory import default_path as default_inventory_path
from .selection import windows
from .sets import SAMPLE_RATE_HZ

FRAME = round(0.025 * SAMPLE_RATE_HZ)
HOP = round(0.010 * SAMPLE_RATE_HZ)
EXCLUSIVE_DB = 6.0
LAGS_S = [round(step * 0.1, 1) for step in range(-20, 21)]
MAX_LAG_S = 0.3


def frame_energy_db(pcm: np.ndarray) -> np.ndarray:
    squares = np.concatenate(([0.0], np.cumsum(pcm.astype(np.float64) ** 2)))
    starts = np.arange(0, max(pcm.size - FRAME, 0) + 1, HOP)
    return 10 * np.log10((squares[starts + FRAME] - squares[starts]) / FRAME + 1e-10)


def frame_mask(spans: Sequence[tuple[float, float]], frames: int, lag_s: float = 0.0) -> np.ndarray:
    """Frames whose centre lies inside any span shifted by ``lag_s``."""
    edges = np.zeros(frames + 1, dtype=np.int32)
    centre_s = FRAME / 2 / SAMPLE_RATE_HZ
    for start, end in spans:
        first = max(int(np.ceil((start + lag_s - centre_s) * SAMPLE_RATE_HZ / HOP)), 0)
        last = min(int(np.floor((end + lag_s - centre_s) * SAMPLE_RATE_HZ / HOP)) + 1, frames)
        if first < last:
            edges[first] += 1
            edges[last] -= 1
    return np.cumsum(edges[:-1]) > 0


def exclusive_voice(spans: Sequence[tuple[float, float]], japanese: np.ndarray, other: np.ndarray) -> dict:
    """Coverage of Japanese-only and other-only frames by the cues, at lag 0
    and at the lag that maximises their difference."""
    difference = japanese - other
    japanese_only, other_only = difference >= EXCLUSIVE_DB, difference <= -EXCLUSIVE_DB
    counts = {"japanese_only_frames": int(japanese_only.sum()), "other_only_frames": int(other_only.sum())}
    if not spans or not japanese_only.any() or not other_only.any():
        return counts

    def coverage(lag_s: float) -> tuple[float, float]:
        inside = frame_mask(spans, difference.size, lag_s)
        return float(inside[japanese_only].mean()), float(inside[other_only].mean())

    by_lag = {lag: coverage(lag) for lag in LAGS_S}
    best = max(LAGS_S, key=lambda lag: (by_lag[lag][0] - by_lag[lag][1], -abs(lag)))
    ours, theirs = by_lag[0.0]
    return {
        **counts,
        "japanese_only_coverage": round(ours, 3),
        "other_only_coverage": round(theirs, 3),
        "coverage_difference": round(ours - theirs, 3),
        "best_lag_s": best,
        "coverage_difference_at_best_lag": round(by_lag[best][0] - by_lag[best][1], 3),
    }


def start_distances(reference: Sequence[tuple[float, float]], other: Sequence[tuple[float, float]]) -> dict:
    """Distance from each Japanese cue start to the nearest start of another track."""
    starts = np.array(sorted(start for start, _ in other))
    if not reference or not starts.size:
        return {}
    distances = np.array([np.min(np.abs(starts - start)) for start, _ in reference])
    return {
        "median_start_distance_s": round(float(np.median(distances)), 3),
        "starts_within_0_2_s": round(float(np.mean(distances <= 0.2)), 3),
    }


def _spans(srt: bytes) -> list[tuple[float, float]]:
    from scripts.e0_evidence.subtitle_proposals import _parse_srt

    if not srt.strip():
        return []
    return [
        (cue.original_start_ms / 1000, cue.original_end_ms / 1000)
        for cue in _parse_srt(srt)
        if not any(mark in cue.raw_text for mark in RULES["lyric_marks"])
        and normalize_minimal(clean_cue_text(cue.raw_text))
    ]


def decode(ffmpeg: str, record: Mapping, scratch: Path) -> tuple[list[np.ndarray], dict[int, list | str], dict]:
    """Every audio stream's samples; every text track's cue spans, or why
    they could not be read; the on-screen text events dropped per ASS track."""
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", record["path"]]
    for stream in record["audio"]:
        command += ["-map", f"0:a:{stream['ordinal']}", "-ac", "1", "-ar", str(SAMPLE_RATE_HZ),
                    "-c:a", "pcm_f32le", "-f", "f32le", str(scratch / f"audio{stream['ordinal']}.f32")]
    texts = [stream for stream in record["subtitles"] if stream["codec"] in TEXT_SUBTITLE_CODECS]
    for stream in texts:
        target = scratch / f"track{stream['ordinal']}"
        if stream["codec"] in ASS_CODECS:
            command += ["-map", f"0:s:{stream['ordinal']}", "-c:s", "copy", "-f", "ass", f"{target}.ass"]
        else:
            command += ["-map", f"0:s:{stream['ordinal']}", "-f", "srt", f"{target}.srt"]
    subprocess.run(command, check=True, capture_output=True)
    tracks: dict[int, list | str] = {}
    dropped: dict[int, dict] = {}
    for stream in texts:
        target = scratch / f"track{stream['ordinal']}"
        if stream["codec"] in ASS_CODECS:
            script, counts = without_sign_events(Path(f"{target}.ass").read_text(encoding="utf-8-sig"))
            dropped[stream["ordinal"]] = dict(counts)
            Path(f"{target}.ass").write_text(script, encoding="utf-8")
            subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", f"{target}.ass", "-f", "srt",
                            f"{target}.srt"], check=True, capture_output=True)
        try:
            tracks[stream["ordinal"]] = _spans(Path(f"{target}.srt").read_bytes())
        except ValueError as error:
            tracks[stream["ordinal"]] = f"unreadable: {error}"
    audio = [np.fromfile(scratch / f"audio{stream['ordinal']}.f32", dtype="<f4") for stream in record["audio"]]
    return audio, tracks, dropped


def _label(stream: Mapping) -> str:
    return f"audio{stream['ordinal']}_{stream['language'] or 'und'}"


def _problems(versus: Mapping[str, Mapping]) -> list[str]:
    problems = []
    for label, result in versus.items():
        if "coverage_difference" not in result:
            problems.append(f"no cues or no exclusive voice frames against {label}")
            continue
        if result["coverage_difference"] <= 0:
            problems.append(f"cues cover {label} voice as much as Japanese ({result['coverage_difference']})")
        if abs(result["best_lag_s"]) > MAX_LAG_S:
            problems.append(f"cues fit the audio best shifted by {result['best_lag_s']} s")
    return problems


def check_episode(ffmpeg: str, record: Mapping, scratch: Path) -> dict:
    audio, tracks, dropped = decode(ffmpeg, record, scratch)
    frames = min(frame_energy_db(pcm).size for pcm in audio)
    energy = [frame_energy_db(pcm)[:frames] for pcm in audio]
    japanese = record["japanese_audio_ordinal"]
    reference = record["japanese_dialogue_subtitle_ordinal"]
    others = [stream for stream in record["audio"] if stream["ordinal"] != japanese]
    measured: dict[str, dict] = {}
    for ordinal, spans in tracks.items():
        stream = record["subtitles"][ordinal]
        entry: dict = {key: stream[key] for key in ("index", "language", "title", "codec")}
        entry["ass_events_dropped"] = dropped.get(ordinal)
        if isinstance(spans, str):
            entry["problem"] = spans
        else:
            entry["cues"] = len(spans)
            entry["versus"] = {
                _label(other): exclusive_voice(spans, energy[japanese], energy[other["ordinal"]])
                for other in others
            }
            if ordinal != reference and isinstance(tracks[reference], list):
                entry["start_distance_from_japanese_track"] = start_distances(tracks[reference], spans)
        measured[str(ordinal)] = entry
    ours = measured[str(reference)]
    problems = [ours["problem"]] if "problem" in ours else _problems(ours["versus"])
    window_results = []
    if isinstance(spans := tracks[reference], list):
        centre_s = FRAME / 2 / SAMPLE_RATE_HZ
        for start_s, end_s in windows(audio[japanese].size / SAMPLE_RATE_HZ) or []:
            first = max(int(np.ceil((start_s - centre_s) * SAMPLE_RATE_HZ / HOP)), 0)
            last = min(int(np.floor((end_s - centre_s) * SAMPLE_RATE_HZ / HOP)) + 1, frames)
            offset = first * HOP / SAMPLE_RATE_HZ
            relative = [(start - offset, end - offset) for start, end in spans]
            versus = {
                _label(other): exclusive_voice(relative, energy[japanese][first:last], energy[other["ordinal"]][first:last])
                for other in others
            }
            window_results.append({"start_s": round(start_s, 3), "end_s": round(end_s, 3), "versus": versus})
    return {
        "path": record["path"],
        "size": Path(record["path"]).stat().st_size,
        "file": record["file"],
        "title": record["title"],
        "audio": [{key: stream[key] for key in ("index", "language", "title")} for stream in record["audio"]],
        "japanese_audio_ordinal": japanese,
        "japanese_track_ordinal": reference,
        "tracks": measured,
        "windows": window_results,
        "passed": not problems,
        "problems": problems,
    }


def _versus_text(versus: Mapping[str, Mapping]) -> str:
    return "; ".join(
        f"{label}: J-only {value.get('japanese_only_coverage')} other-only "
        f"{value.get('other_only_coverage')} diff {value.get('coverage_difference')} "
        f"best lag {value.get('best_lag_s')}"
        for label, value in versus.items()
    )


def _print(result: Mapping) -> None:
    """The Japanese track, its windows, and the control tracks in English or
    in a dub's language."""
    print(f"{'PASS' if result['passed'] else 'FAIL'} {result['title']} | {result['file']}", flush=True)
    dubs = {stream["language"] for stream in result["audio"]} | {"eng"}
    for ordinal, entry in result["tracks"].items():
        ours = int(ordinal) == result["japanese_track_ordinal"]
        if not ours and entry["language"] not in dubs:
            continue
        detail = entry.get("problem") or _versus_text(entry["versus"])
        starts = entry.get("start_distance_from_japanese_track")
        print(f"   {'japanese' if ours else 'control '} #{entry['index']} {entry['language']} "
              f"{entry['title']!r}: {detail}" + (f"; start distance {starts}" if starts else ""), flush=True)
        if ours:
            for index, window in enumerate(result["windows"]):
                print(f"     window {index} {window['start_s']:.0f}-{window['end_s']:.0f} s: "
                      f"{_versus_text(window['versus'])}", flush=True)
    for problem in result["problems"]:
        print(f"   problem: {problem}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title", action="append", help="check only this inventory title; repeatable")
    parser.add_argument("--inventory", type=Path, default=default_inventory_path())
    parser.add_argument("--out", type=Path, required=True,
                        help="result JSON to write (never a result a built set records)")
    args = parser.parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg must be on PATH")
    inventory_bytes = args.inventory.read_bytes()
    inventory = load_inventory(args.inventory)
    chosen = [
        record for record in inventory["files"]
        if record["eligible"] and len(record["audio"]) > 1 and (not args.title or record["title"] in args.title)
    ]
    episodes, failed_decode = [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as scratch_name:
        for record in chosen:
            try:
                result = check_episode(ffmpeg, record, Path(scratch_name))
            except subprocess.CalledProcessError as error:
                failed_decode.append({"file": record["file"], "path": record["path"],
                                      "reason": f"decode failed (exit {error.returncode})"})
                continue
            episodes.append(result)
            _print(result)
    report = {
        "schema_version": CHECK_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ffmpeg_version": tool_version(ffmpeg),
        "inventory": str(args.inventory),
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "titles": args.title,
        "frame_s": FRAME / SAMPLE_RATE_HZ,
        "hop_s": HOP / SAMPLE_RATE_HZ,
        "exclusive_db": EXCLUSIVE_DB,
        "lag_range_s": [LAGS_S[0], LAGS_S[-1]],
        "lag_step_s": 0.1,
        "max_lag_s": MAX_LAG_S,
        "episodes": episodes,
        "failed_decode": failed_decode,
    }
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    for title in sorted({episode["title"] for episode in episodes}):
        results = [episode for episode in episodes if episode["title"] == title]
        print(f"{title}: {sum(episode['passed'] for episode in results)} of {len(results)} episodes pass")
    print(f"{len(failed_decode)} episodes failed to decode; wrote {args.out}")


if __name__ == "__main__":
    main()
