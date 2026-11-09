"""Build the synthetic ASR bench set: known-transcript scenes under music and
effects beds.

Speech sources:

* Common Voice 17.0 ja test clips (CC0, ``fsicoli/common_voice_17_0`` at a
  pinned revision), filtered by votes and sentence length only. No ASR or VAD
  screens them: screening with one candidate biases the comparison toward it.
* VOICEVOX ENGINE renderings (``voicevox.py``) of the hand-written dialogue
  lines in ``dialogue_lines.ja.txt`` plus a sample of the Common Voice
  sentences, spoken by several characters and styles.

Every line is trimmed of its leading and trailing quiet audio (20 ms frames
more than 40 dB below the line's loudest frame, keeping 50 ms; an energy
threshold, not a VAD) and normalized to -23 dBFS RMS.

Scenes last 60 s: lines in sequence separated by 0.2-2.0 s gaps, some
0.1-0.3 s rapid turns, and about 10% overlapping pairs (the second line starts
before the first ends; both carry ``overlap=true``). Half the scenes are
Common Voice speech, half VOICEVOX. Each scene is rendered clean and over a
bed track at SNR +10, +5, 0 and -5 dB, where SNR is speech RMS over the
union of line spans against bed RMS over the scene. The bed track tiles beds
(``beds.py``) with crossfades under the whole scene, each bed at its level in
the source mix. Each item is peak-limited to -1 dBFS by one gain, which
leaves its SNR unchanged. ``speech_free`` is the complement of the line
spans, 0.15 s away from every line, in spans of at least 0.5 s.

    python -m scripts.asr_bench.build_synthetic --out <bench_root>/sets/synthetic-v1 [--scenes 40] [--seed 105]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.error
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import voicevox
from .beds import Bed, load_bed_audio, load_beds
from .sets import SAMPLE_RATE_HZ, bench_root, write_audio, write_set

CV_REPO = "fsicoli/common_voice_17_0"
CV_REVISION = "8262c16bf297c87a9cd88c51997c4758ed7a8ba2"
CV_TSV = "transcript/ja/test.tsv"
CV_TAR = "audio/ja/test/ja_test_0.tar"
CV_LICENSE = "CC0-1.0"
CV_ATTRIBUTION = "Mozilla Common Voice 17.0, Japanese test split (via fsicoli/common_voice_17_0)"
CV_MIN_UP_VOTES = 2
CV_SENTENCE_CHARS = (6, 40)

DIALOGUE_FILE = Path(__file__).with_name("dialogue_lines.ja.txt")
SCENE_S = 60.0
SNR_DB = (10, 5, 0, -5)
LEAD_S = (0.3, 1.5)
TAIL_MIN_S = 0.5
GAP_S = (0.2, 2.0)
RAPID_TURN_S = (0.1, 0.3)
RAPID_TURN_PROBABILITY = 0.2
OVERLAP_PROBABILITY = 0.1
OVERLAP_S = (0.2, 1.0)
VOICEVOX_CV_SHARE = 0.35
VOICES_PER_SCENE = 3
FIT_LOOKAHEAD = 6
LINE_RMS_DBFS = -23.0
PEAK_DBFS = -1.0
TRIM_FRAME_S = 0.02
TRIM_FLOOR_DB = 40.0
TRIM_PAD_S = 0.05
CROSSFADE_S = 0.5
FREE_MARGIN_S = 0.15
FREE_MIN_S = 0.5


@dataclass(frozen=True)
class LineAudio:
    text: str
    source: str
    voice: str
    samples: np.ndarray


def _db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def prepare_line(samples: np.ndarray) -> np.ndarray | None:
    """Trim quiet edges and normalize to ``LINE_RMS_DBFS``; ``None`` for
    digital silence."""
    frame = round(TRIM_FRAME_S * SAMPLE_RATE_HZ)
    count = samples.size // frame
    if count == 0:
        return None
    energy = np.sqrt(np.mean(np.square(samples[: count * frame].reshape(count, frame), dtype=np.float64), axis=1))
    peak = float(energy.max())
    if peak <= 0.0:
        return None
    active = np.flatnonzero(energy >= peak * _db_to_gain(-TRIM_FLOOR_DB))
    pad = round(TRIM_PAD_S * SAMPLE_RATE_HZ)
    start = max(0, int(active[0]) * frame - pad)
    end = min(samples.size, (int(active[-1]) + 1) * frame + pad)
    trimmed = samples[start:end].astype(np.float64)
    return trimmed * (_db_to_gain(LINE_RMS_DBFS) / _rms(trimmed))


# Common Voice ---------------------------------------------------------------


def _hf_download(filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=CV_REPO, filename=filename, repo_type="dataset",
        revision=CV_REVISION, cache_dir=str(cache_dir),
    ))


def cv_candidates(tsv_path: Path, rng: np.random.Generator) -> list[dict[str, str]]:
    """Eligible rows ordered so that consecutive clips come from distinct
    speakers: one clip per speaker per round, speakers in seeded order."""
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    low, high = CV_SENTENCE_CHARS
    by_client: dict[str, list[dict[str, str]]] = {}
    for row in sorted(rows, key=lambda row: row["path"]):
        if (
            int(row["up_votes"]) >= CV_MIN_UP_VOTES
            and int(row["down_votes"]) == 0
            and low <= len(row["sentence"]) <= high
        ):
            by_client.setdefault(row["client_id"], []).append(row)
    clients = sorted(by_client)
    rng.shuffle(clients)
    for client in clients:
        rng.shuffle(by_client[client])
    ordered: list[dict[str, str]] = []
    for round_index in range(max(len(clips) for clips in by_client.values())):
        ordered.extend(
            by_client[client][round_index] for client in clients
            if round_index < len(by_client[client])
        )
    return ordered


class CommonVoiceClips:
    """Decodes clips from the test tar on demand (FFmpeg, 16 kHz mono)."""

    def __init__(self, tar_path: Path, ffmpeg: str) -> None:
        self._archive = tarfile.open(tar_path)
        self._members = {
            Path(member.name).name: member
            for member in self._archive.getmembers() if member.isfile()
        }
        self._ffmpeg = ffmpeg

    def close(self) -> None:
        self._archive.close()

    def decode(self, clip: str) -> np.ndarray | None:
        member = self._members.get(clip)
        if member is None:
            return None
        handle = self._archive.extractfile(member)
        if handle is None:
            return None
        completed = subprocess.run(
            [self._ffmpeg, "-v", "error", "-i", "-", "-ac", "1",
             "-ar", str(SAMPLE_RATE_HZ), "-f", "f32le", "-"],
            input=handle.read(), capture_output=True, check=False,
        )
        if completed.returncode != 0:
            return None
        return np.frombuffer(completed.stdout, dtype="<f4")


# Scene assembly ---------------------------------------------------------------


class Pool:
    """Pending lines; a scene takes the first of the next few that fits."""

    def __init__(self, render: Callable[[object, str | None], LineAudio | None],
                 entries: Iterator[object]) -> None:
        self._render = render
        self._entries = entries
        self._pending: list[object] = []
        self.failed: list[object] = []

    def take(self, fits: Callable[[float], bool], previous_voice: str | None) -> LineAudio | None:
        index = 0
        while index < FIT_LOOKAHEAD:
            if index == len(self._pending):
                entry = next(self._entries, None)
                if entry is None:
                    return None
                self._pending.append(entry)
            line = self._render(self._pending[index], previous_voice)
            if line is None:
                self.failed.append(self._pending.pop(index))
                continue
            if fits(line.samples.size / SAMPLE_RATE_HZ):
                self._pending.pop(index)
                return line
            index += 1
        return None


def assemble_scene(rng: np.random.Generator, pool: Pool) -> tuple[np.ndarray, list[dict], list[str]]:
    """Place lines from ``pool`` into one scene; return the speech track,
    line records and the voices used in order of first appearance."""
    total = round(SCENE_S * SAMPLE_RATE_HZ)
    limit_s = SCENE_S - TAIL_MIN_S
    track = np.zeros(total, dtype=np.float64)
    lines: list[dict] = []
    voices: list[str] = []
    previous_end = 0.0
    previous_duration = 0.0
    previous_voice: str | None = None
    while True:
        if not lines:
            start_offset = rng.uniform(*LEAD_S)
            overlap = False
        else:
            overlap = not lines[-1]["overlap"] and rng.random() < OVERLAP_PROBABILITY
            if rng.random() < RAPID_TURN_PROBABILITY:
                gap = rng.uniform(*RAPID_TURN_S)
            else:
                gap = rng.uniform(*GAP_S)
            overlap_draw = rng.uniform(*OVERLAP_S)
        def start_for(duration: float) -> float:
            if not lines:
                return start_offset
            if overlap:
                # Overlap at most half of either line so each stays audible.
                amount = min(overlap_draw, 0.5 * previous_duration, 0.5 * duration)
                return previous_end - amount
            return previous_end + gap
        line = pool.take(lambda duration: start_for(duration) + duration <= limit_s, previous_voice)
        if line is None:
            break
        duration = line.samples.size / SAMPLE_RATE_HZ
        start_s = start_for(duration)
        first = round(start_s * SAMPLE_RATE_HZ)
        track[first: first + line.samples.size] += line.samples
        if overlap:
            lines[-1]["overlap"] = True
        lines.append({
            "line_id": f"L{len(lines) + 1:02d}",
            "start_s": round(first / SAMPLE_RATE_HZ, 5),
            "end_s": round((first + line.samples.size) / SAMPLE_RATE_HZ, 5),
            "text": line.text,
            "source": line.source,
            "overlap": overlap,
        })
        if line.voice not in voices:
            voices.append(line.voice)
        previous_end = (first + line.samples.size) / SAMPLE_RATE_HZ
        previous_duration = duration
        previous_voice = line.voice
    return track, lines, voices


def speech_mask(lines: Sequence[dict], total: int) -> np.ndarray:
    mask = np.zeros(total, dtype=bool)
    for line in lines:
        mask[round(line["start_s"] * SAMPLE_RATE_HZ): round(line["end_s"] * SAMPLE_RATE_HZ)] = True
    return mask


def speech_free_spans(lines: Sequence[dict], duration_s: float) -> list[list[float]]:
    spans = sorted((line["start_s"], line["end_s"]) for line in lines)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    free: list[list[float]] = []
    cursor, cursor_margin = 0.0, 0.0
    for start, end in [*merged, [duration_s, duration_s]]:
        low = cursor + cursor_margin
        high = start - (FREE_MARGIN_S if start < duration_s else 0.0)
        if high - low >= FREE_MIN_S:
            free.append([round(low, 4), round(high, 4)])
        cursor, cursor_margin = end, FREE_MARGIN_S
    return free


def bed_track(beds_cycle: Iterator[Bed], total: int) -> tuple[np.ndarray, list[str]]:
    """Beds laid end to end with equal-power crossfades, each at its level in
    the source mix (scaling a near-silent bed up to a common level would turn
    its clicks into loud peaks)."""
    fade = round(CROSSFADE_S * SAMPLE_RATE_HZ)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False) * (np.pi / 2)
    fade_in, fade_out = np.sin(ramp), np.cos(ramp)
    track = np.zeros(total, dtype=np.float64)
    bed_ids: list[str] = []
    offset = 0
    while offset < total:
        bed = next(beds_cycle)
        segment = load_bed_audio(bed).astype(np.float64)
        if bed_ids:
            segment[:fade] *= fade_in
        end = offset + segment.size
        if end < total:
            segment[-fade:] *= fade_out
        take = min(segment.size, total - offset)
        track[offset: offset + take] += segment[:take]
        bed_ids.append(bed.bed_id)
        if end >= total:
            break
        offset = end - fade
    return track, bed_ids


def peak_limit(mix: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(mix)))
    ceiling = _db_to_gain(PEAK_DBFS)
    return mix * (ceiling / peak) if peak > ceiling else mix


# Build --------------------------------------------------------------------------


def _cycle(values: Sequence, rng: np.random.Generator) -> Iterator:
    while True:
        order = list(values)
        rng.shuffle(order)
        yield from order


def build(out_dir: Path, scenes: int, seed: int, cache_dir: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg must be on PATH")
    cv_rng, voice_rng, scene_rng, bed_rng, script_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(5)
    )
    bed_manifest, beds = load_beds()
    dialogue = [
        line.strip() for line in DIALOGUE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tsv_path = _hf_download(CV_TSV, cache_dir)
    tar_path = _hf_download(CV_TAR, cache_dir)
    candidates = cv_candidates(tsv_path, cv_rng)
    clips = CommonVoiceClips(tar_path, ffmpeg)
    beds_cycle = _cycle(beds, bed_rng)

    def render_cv(row: object, _previous: str | None) -> LineAudio | None:
        assert isinstance(row, dict)
        decoded = clips.decode(row["path"])
        prepared = None if decoded is None else prepare_line(decoded)
        if prepared is None:
            return None
        return LineAudio(row["sentence"], "cv17", f"cv17:{row['client_id'][:12]}", prepared)

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "audio").exists():
        shutil.rmtree(out_dir / "audio")
    scene_sources = ["cv17" if index % 2 == 0 else "voicevox" for index in range(scenes)]
    rendered: dict[int, tuple[np.ndarray, list[dict], list[str]]] = {}
    used_sentences: set[str] = set()
    cv_iter = iter(candidates)
    cv_pool = Pool(render_cv, cv_iter)
    for index, source in enumerate(scene_sources):
        if source == "cv17":
            track, lines, voices = assemble_scene(scene_rng, cv_pool)
            used_sentences.update(line["text"] for line in lines)
            rendered[index] = (track, lines, voices)

    def unused_sentences() -> Iterator[str]:
        # Common Voice sentences that no Common Voice scene speaks.
        for row in cv_iter:
            if row["sentence"] not in used_sentences:
                used_sentences.add(row["sentence"])
                yield row["sentence"]

    cv_sample = unused_sentences()
    dialogue_cycle = _cycle(dialogue, script_rng)

    def script() -> Iterator[object]:
        while True:
            if script_rng.random() < VOICEVOX_CV_SHARE:
                sentence = next(cv_sample, None)
                if sentence is not None:
                    yield ("cv_sentence", sentence)
                    continue
            yield ("dialogue", next(dialogue_cycle))

    style_use: dict[str, int] = {}
    with ExitStack() as stack:
        stack.callback(clips.close)
        client = stack.enter_context(voicevox.running_engine())
        voice_list = voicevox.resolve_voices(client.speakers())
        render_cache = bench_root() / "cache" / f"voicevox-{voicevox.ENGINE_VERSION}"
        render_cache.mkdir(parents=True, exist_ok=True)
        scene_voices: list[voicevox.Voice] = []

        def synthesize(voice: voicevox.Voice, text: str) -> np.ndarray:
            key = hashlib.sha256(f"{voice.style_id}\n{text}".encode()).hexdigest()[:24]
            cached = render_cache / f"{key}.f32"
            if cached.is_file():
                return np.fromfile(cached, dtype="<f4")
            samples = client.synthesize(text, voice.style_id)
            write_audio(cached, samples)
            return samples

        def render_voicevox(entry: object, previous: str | None) -> LineAudio | None:
            kind, text = entry  # type: ignore[misc]
            options = [voice for voice in scene_voices if voice.label != previous]
            voice = options[int(voice_rng.integers(len(options)))]
            try:
                samples = synthesize(voice, text)
            except urllib.error.HTTPError as exc:  # the engine rejected the text
                print(f"VOICEVOX failed on {text!r} ({voice.label}): {exc}", flush=True)
                return None
            prepared = prepare_line(samples)
            if prepared is None:
                return None
            return LineAudio(text, "voicevox", voice.label, prepared)

        voicevox_pool = Pool(render_voicevox, script())
        characters = sorted({voice.character for voice in voice_list})
        for index, source in enumerate(scene_sources):
            if source != "voicevox":
                continue
            chosen = voice_rng.choice(len(characters), size=VOICES_PER_SCENE, replace=False)
            scene_voices[:] = []
            for character_index in chosen:
                styles = [v for v in voice_list if v.character == characters[int(character_index)]]
                scene_voices.append(styles[int(voice_rng.integers(len(styles)))])
            rendered[index] = assemble_scene(scene_rng, voicevox_pool)
            for label in rendered[index][2]:
                style_use[label] = style_use.get(label, 0) + 1

    total = round(SCENE_S * SAMPLE_RATE_HZ)
    items: list[dict] = []
    line_counts: dict[str, int] = {}
    overlap_pairs = 0
    pair_count = 0
    for index, source in enumerate(scene_sources):
        speech, lines, voices = rendered[index]
        scene_id = f"s{index + 1:02d}"
        for line in lines:
            line_counts[line["source"]] = line_counts.get(line["source"], 0) + 1
        pair_count += max(0, len(lines) - 1)
        overlap_pairs += sum(
            1 for a, b in zip(lines, lines[1:]) if a["overlap"] and b["overlap"] and b["start_s"] < a["end_s"]
        )
        mask = speech_mask(lines, total)
        speech_rms = _rms(speech[mask])
        free = speech_free_spans(lines, SCENE_S)
        bed, bed_ids = bed_track(beds_cycle, total)
        bed_rms = _rms(bed)
        conditions: list[tuple[str, int | None, np.ndarray, list[str]]] = [("clean", None, speech, [])]
        for snr in SNR_DB:
            gain = speech_rms / (bed_rms * _db_to_gain(snr))
            conditions.append((f"snr{snr:+d}", snr, speech + gain * bed, bed_ids))
        for suffix, snr, mix, ids in conditions:
            item_id = f"{scene_id}-{suffix}"
            audio = f"audio/{item_id}.f32"
            items.append({
                "item_id": item_id,
                "audio": audio,
                "sample_count": write_audio(out_dir / audio, peak_limit(mix)),
                "condition": {
                    "scene_id": scene_id,
                    "snr_db": snr,
                    "speech_source": source,
                    "voices": voices,
                    "bed_ids": ids,
                },
                "lines": lines,
                "speech_free": free,
            })
        print(f"{scene_id} {source}: {len(lines)} lines, {len(voices)} voices, {len(bed_ids)} beds", flush=True)

    credits = {}
    for voice in voice_list:
        entry = credits.setdefault(voice.character, {"name": voice.character, "credit": voice.credit, "styles": []})
        entry["styles"].append({
            "style": voice.style, "style_id": voice.style_id,
            "register": voice.register, "scenes": style_use.get(voice.label, 0),
        })
    written = write_set(
        out_dir,
        set_id=out_dir.name,
        kind="synthetic",
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        build={
            "builder": "scripts.asr_bench.build_synthetic",
            "seed": seed,
            "scenes": scenes,
            "scene_s": SCENE_S,
            "snr_db": list(SNR_DB),
            "snr_definition": "speech RMS over the union of line spans vs bed RMS over the scene",
            "gap_s": list(GAP_S),
            "rapid_turn_s": list(RAPID_TURN_S),
            "rapid_turn_probability": RAPID_TURN_PROBABILITY,
            "overlap_probability": OVERLAP_PROBABILITY,
            "overlap_pairs": overlap_pairs,
            "adjacent_pairs": pair_count,
            "line_rms_dbfs": LINE_RMS_DBFS,
            "line_trim": f"edges below -{TRIM_FLOOR_DB:g} dB of the loudest {TRIM_FRAME_S * 1000:g} ms frame, {TRIM_PAD_S * 1000:g} ms kept",
            "peak_limit_dbfs": PEAK_DBFS,
            "speech_free": f"line-span complement, {FREE_MARGIN_S} s margins, spans >= {FREE_MIN_S} s",
            "line_counts": line_counts,
            "common_voice": {
                "dataset": CV_REPO,
                "revision": CV_REVISION,
                "license": CV_LICENSE,
                "attribution": CV_ATTRIBUTION,
                "files": [CV_TSV, CV_TAR],
                "filter": f"up_votes >= {CV_MIN_UP_VOTES}, down_votes == 0, {CV_SENTENCE_CHARS[0]}-{CV_SENTENCE_CHARS[1]} sentence characters",
                "eligible_clips": len(candidates),
                "eligible_speakers": len({row["client_id"] for row in candidates}),
                "undecodable_clips": len(cv_pool.failed),
            },
            "voicevox": {
                "engine": f"VOICEVOX ENGINE {voicevox.ENGINE_VERSION} (windows-cpu)",
                "characters": list(credits.values()),
                "credits": [entry["credit"] for entry in credits.values()],
                "failed_renderings": len(voicevox_pool.failed),
            },
            "dialogue_lines": {
                "file": DIALOGUE_FILE.name,
                "sha256": hashlib.sha256(DIALOGUE_FILE.read_bytes()).hexdigest(),
                "count": len(dialogue),
            },
            "beds": {
                "count": len(beds),
                "rules": bed_manifest["rules"],
                "excluded_fixture_labels": bed_manifest["excluded_fixture_labels"],
                "source_files": sorted({bed.source_file for bed in beds}),
                "crossfade_s": CROSSFADE_S,
                "levels": "each bed at its level in the source mix",
            },
        },
        items=items,
    )
    minutes = sum(item.duration_s for item in written.items) / 60
    print(f"wrote {len(written.items)} items ({minutes:.1f} min) to {out_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=bench_root() / "sets" / "synthetic-v1")
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--cache-dir", type=Path, default=bench_root() / "cache" / "hf",
                        help="Hugging Face download cache for Common Voice")
    args = parser.parse_args(argv)
    if args.scenes < 2 or args.scenes % 2:
        parser.error("--scenes must be an even number of at least 2")
    build(args.out, args.scenes, args.seed, args.cache_dir)


if __name__ == "__main__":
    main()
