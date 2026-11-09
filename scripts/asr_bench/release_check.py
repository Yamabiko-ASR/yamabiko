"""Check the bench's sound-tag delay line against production.

Usage::

    python -m scripts.asr_bench.release_check --set <set dir> --candidate <json>
        --name <candidate> [--items ID[,ID...]] --out <json>

Each item is chunked twice:

- the bench path: ``run._chunk_item`` with the candidate's assisted Silero
  (its precomputed tags) and emission positions from ``run._decision_sample``
  under the tags directory's layout, exactly as ``run.py`` does;
- production: ``SileroChunker`` with the same Silero graph and the bundled
  ``SoundTagger`` (``yamabiko/vad/sound_tagger.py``, its own delay line and
  inference), fed the item, the runner's zero tail and one more second of
  zeros (so every frame the bench feeds leaves the delay line through a real
  hop) in 480-sample capture blocks.

Production is observed on the chunker instance: every frame its gate
receives is recorded with its tag term, its fused probability and the audio
position at that moment (the end of the capture block being processed), and
a coordinate gate fed the same probabilities names each emitted window's
span. The report gives, per item and pooled: the frames the gate receives at
another position than ``_decision_sample`` says, the largest tag term and
fused probability differences inside the item and in the zero tail, and the
windows that differ in span, emit reason or emission position (production
windows emitted by a frame past the bench's audio are listed apart).

The candidate must have a ``vad_assist`` with tags alone (no Silero assist,
no frontend), production's fusion (``voice`` + ``shout``, weight
``ASSIST_WEIGHT``) and a tags directory in production's layout; the tags'
graph should be the one production ships for the terms to match.
"""

from __future__ import annotations

from yamabiko import hf_env  # noqa: F401  # pin the model cache first

import argparse
import asyncio
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench import run as bench
from scripts.asr_bench.candidates import Candidate, load_candidate, resolve_vad, shipped_config
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, Item, load_set
from scripts.asr_bench.tag_sounds import TagLayout, load_layout, load_tags
from yamabiko.vad import sound_tagger
from yamabiko.vad.chunker import chunker_config_from_config
from yamabiko.vad.silero_chunker import FrameGate, SileroChunker, SileroVAD

_FRAME = 512
_BLOCK = 480
_EXTRA_SAMPLES = SAMPLE_RATE_HZ  # zeros production hears after the runner's tail


def _checked(candidate: Candidate) -> TagLayout:
    assist = candidate.vad_assist
    if candidate.frontend is not None or assist is None or assist.silero is not None:
        raise ValueError(f"{candidate.name}: needs a vad_assist with tags alone and no frontend")
    tags = assist.tags
    if tags is None:
        raise ValueError(f"{candidate.name}: needs vad_assist tags")
    if (tags.threshold is not None or tags.weight != sound_tagger.ASSIST_WEIGHT
            or sorted(tags.groups) != ["shout", "voice"]):
        raise ValueError(f"{candidate.name}: production fuses voice + shout with weight "
                         f"{sound_tagger.ASSIST_WEIGHT}")
    layout = load_layout(Path(tags.tags_dir))
    if (layout.hop_frames, layout.lookahead_frames) != (sound_tagger.HOP_FRAMES,
                                                         sound_tagger.LOOKAHEAD_FRAMES):
        raise ValueError(f"{tags.tags_dir}: layout {layout}, production runs "
                         f"({sound_tagger.HOP_FRAMES}, {sound_tagger.LOOKAHEAD_FRAMES})")
    return layout


class _Observed:
    """What production's gate received and emitted, frame by frame."""

    def __init__(self) -> None:
        self.position = 0
        self.tags: list[float] = []
        self.fused: list[float] = []
        self.positions: list[int] = []
        # (start, end, emission position, emit reason, emitting frame)
        self.windows: list[tuple[int, int, int, str, int]] = []


class _TrackedQueue(asyncio.Queue):
    """Capture queue that keeps the end of the block the chunker took last
    and signals once every block has been processed."""

    def __init__(self, observed: _Observed) -> None:
        super().__init__(maxsize=64)
        self._observed = observed
        self.fed = False
        self.drained = asyncio.Event()

    async def get(self):
        if self.fed and self.empty():
            self.drained.set()
        block = await super().get()
        self._observed.position = block.start_sample + block.pcm.size
        return block


async def _production(audio: np.ndarray, *, config, threshold: float, genre: str,
                      silero: SileroVAD, tagger: sound_tagger.SoundTagger) -> _Observed:
    from yamabiko.capture.envelope import CaptureBlock
    from yamabiko.state import AppState

    silero.reset_states()
    observed = _Observed()
    chunker = SileroChunker(config, threshold=threshold, state=AppState(genre=genre),
                            model=silero, tagger=tagger)
    coordinates = FrameGate(config, threshold=threshold, frame_samples=_FRAME)
    gate = chunker._gate
    gate_push = gate.push

    def push(frame: np.ndarray, probability: float):
        ordinal = len(observed.fused)
        observed.fused.append(float(probability))
        observed.positions.append(observed.position)
        span = coordinates.push(np.full(_FRAME, ordinal, dtype=np.float32), probability)
        emitted = gate_push(frame, probability)
        if (span is None) != (emitted is None):
            raise RuntimeError("the coordinate gate and production's gate disagree")
        if emitted is not None:
            ordinals = span.reshape(-1, _FRAME)[:, 0].astype(np.int64)
            start, end = int(ordinals[0]) * _FRAME, (int(ordinals[-1]) + 1) * _FRAME
            if not np.array_equal(emitted, audio[start:end]):
                raise RuntimeError("production's window audio does not match its span")
            observed.windows.append((start, end, observed.position,
                                     gate.last_emission.reason.value, ordinal))
        return emitted

    gate.push = push
    gate_frames = chunker._gate_frames

    async def tagged(frames, out_queue) -> None:
        frames = list(frames)
        observed.tags.extend(float(term) for _, term, _ in frames)
        await gate_frames(frames, out_queue)

    chunker._gate_frames = tagged
    audio_q = _TrackedQueue(observed)
    chunk_q: asyncio.Queue = asyncio.Queue()

    async def feed() -> None:
        for start in range(0, audio.size, _BLOCK):
            await audio_q.put(CaptureBlock(pcm=audio[start:start + _BLOCK].copy(),
                                           stream_generation=0, start_sample=start))
        audio_q.fed = True

    feeder = asyncio.create_task(feed())
    vad = asyncio.create_task(chunker.run(audio_q, chunk_q))
    drained = asyncio.create_task(audio_q.drained.wait())
    try:
        await asyncio.wait({vad, drained}, return_when=asyncio.FIRST_COMPLETED)
        if vad.done():
            raise RuntimeError("production's chunker stopped early") from vad.exception()
    finally:
        for task in (feeder, vad, drained):
            task.cancel()
        await asyncio.gather(feeder, vad, drained, return_exceptions=True)
    if chunker.discontinuity_resets:
        raise RuntimeError(f"unexpected resets: {dict(chunker.discontinuity_resets)}")
    return observed


def _max_abs(a: np.ndarray, b: np.ndarray) -> float | None:
    return float(np.max(np.abs(a - b))) if a.size else None


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "mean": sum(values) / len(values),
            "max": max(values)}


def check_item(item: Item, pcm: np.ndarray, candidate: Candidate, layout: TagLayout, *,
               config, threshold: float, flush: int, genre: str, primary: SileroVAD,
               silero: SileroVAD, tagger: sound_tagger.SoundTagger) -> dict[str, Any]:
    tags = candidate.vad_assist.tags
    total = item.sample_count + flush
    track = load_tags(tags.tags_path(item.item_id), item.sample_count)
    term = bench._tag_term(track, candidate.vad_assist, -(-total // _FRAME))
    assisted = bench._AssistedVAD(primary, None, threshold)
    assisted.start_item(None, term)
    decision = functools.partial(bench._decision_sample, layout=layout)
    windows, pending, probabilities = asyncio.run(bench._chunk_item(
        pcm, config=config, threshold=threshold, vad_model=assisted, flush_samples=flush,
        genre=genre, decision_sample=decision))
    frames = probabilities.size

    audio = np.concatenate([pcm, np.zeros(flush + _EXTRA_SAMPLES, np.float32)])
    observed = asyncio.run(_production(audio, config=config, threshold=threshold, genre=genre,
                                       silero=silero, tagger=tagger))
    if len(observed.fused) < frames:
        raise RuntimeError(f"{item.item_id}: production released {len(observed.fused)} of "
                           f"{frames} frames")
    expected = np.array([decision(k, total) for k in range(frames)], dtype=np.int64)
    positions = np.asarray(observed.positions[:frames], dtype=np.int64)
    late = positions != expected
    inside = (np.arange(frames) * _FRAME + _FRAME // 2) < item.sample_count
    prod_tags = np.asarray(observed.tags[:frames], dtype=np.float32)
    prod_fused = np.asarray(observed.fused[:frames], dtype=np.float32)
    bench_windows = [(w.start_sample, w.end_sample, w.emitted_at_sample, w.emit_reason)
                     for w in windows]
    prod_windows = [window[:4] for window in observed.windows if window[4] < frames]
    beyond = [window[:4] for window in observed.windows if window[4] >= frames]
    return {
        "frames": frames,
        "frames_released_elsewhere": int(late.sum()),
        "first_released_elsewhere": (
            None if not late.any() else {
                "frame": int(np.argmax(late)), "bench": int(expected[late][0]),
                "production": int(positions[late][0])}),
        "tag_term_max_abs_diff": {"item": _max_abs(prod_tags[inside], term[:frames][inside]),
                                  "zero_tail": _max_abs(prod_tags[~inside], term[:frames][~inside])},
        "fused_max_abs_diff": {"item": _max_abs(prod_fused[inside], probabilities[inside]),
                               "zero_tail": _max_abs(prod_fused[~inside], probabilities[~inside])},
        "bench_windows": len(bench_windows),
        "production_windows": len(prod_windows),
        "windows_equal": bench_windows == prod_windows,
        "differing_windows": {
            "bench_only": [list(w) for w in bench_windows if w not in prod_windows],
            "production_only": [list(w) for w in prod_windows if w not in bench_windows],
        },
        "bench_open_at_end": pending,
        "production_windows_past_bench_audio": [list(w) for w in beyond],
        # What the delay line adds to each emission: production's position
        # minus the emitting frame's own capture block end.
        "delay_line_wait_s": _summary([
            (window[2] - bench._decision_sample(window[4], total)) / SAMPLE_RATE_HZ
            for window in observed.windows if window[4] < frames
        ]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", required=True, type=Path, help="bench set directory")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate JSON file")
    parser.add_argument("--name", help="candidate to select from a candidates list")
    parser.add_argument("--items", help="comma-separated item ids (default: all)")
    parser.add_argument("--out", required=True, type=Path, help="report JSON")
    args = parser.parse_args(argv)

    bench_set = load_set(args.set)
    candidate = load_candidate(args.candidate, args.name)
    layout = _checked(candidate)
    wanted = None if args.items is None else [i for i in args.items.split(",") if i]
    by_id = {item.item_id: item for item in bench_set.items}
    unknown = sorted(set(wanted or ()) - set(by_id))
    if unknown:
        raise SystemExit(f"items not in {bench_set.set_id}: {unknown}")
    items = bench_set.items if wanted is None else [by_id[i] for i in wanted]

    cfg = shipped_config()
    cfg["vad"].update(candidate.vad_overrides)
    config, threshold = chunker_config_from_config(cfg)
    flush = SAMPLE_RATE_HZ * (max(config.min_silence_ms, config.glue_max_silence_ms) + 1000) // 1000
    genre = cfg.get("genre", "variety")
    artifact = resolve_vad(candidate.vad)
    primary = SileroVAD(artifact.path, SAMPLE_RATE_HZ, expected_sha256=artifact.sha256)
    silero = SileroVAD(artifact.path, SAMPLE_RATE_HZ, expected_sha256=artifact.sha256)
    tagger = sound_tagger.SoundTagger(sound_tagger.default_model_path())
    manifest = json.loads((Path(candidate.vad_assist.tags.tags_dir) / "manifest.json")
                          .read_text(encoding="utf-8"))

    per_item: dict[str, Any] = {}
    for item in items:
        result = check_item(item, bench_set.load_audio(item), candidate, layout, config=config,
                            threshold=threshold, flush=flush, genre=genre, primary=primary,
                            silero=silero, tagger=tagger)
        per_item[item.item_id] = result
        print(f"{item.item_id}: {result['bench_windows']} bench / "
              f"{result['production_windows']} production windows, equal "
              f"{result['windows_equal']}; frames released elsewhere "
              f"{result['frames_released_elsewhere']} of {result['frames']}; tag term max diff "
              f"{result['tag_term_max_abs_diff']}", flush=True)
    report = {
        "set_id": bench_set.set_id,
        "candidate": candidate.to_json(),
        "layout": layout.to_json(),
        "tags_identity": manifest["identity"],
        "production_tagger": {"model": sound_tagger.MODEL_FILE, "sha256": sound_tagger.MODEL_SHA256,
                              "hop_frames": sound_tagger.HOP_FRAMES,
                              "lookahead_frames": sound_tagger.LOOKAHEAD_FRAMES,
                              "weight": sound_tagger.ASSIST_WEIGHT},
        "extra_zeros_samples": _EXTRA_SAMPLES,
        "pooled": {
            "items": len(per_item),
            "frames": sum(r["frames"] for r in per_item.values()),
            "frames_released_elsewhere": sum(r["frames_released_elsewhere"]
                                             for r in per_item.values()),
            "windows": sum(r["bench_windows"] for r in per_item.values()),
            "items_with_equal_windows": sum(r["windows_equal"] for r in per_item.values()),
        },
        "items": per_item,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"]))
    return 0 if report["pooled"]["items_with_equal_windows"] == len(per_item) else 1


if __name__ == "__main__":
    raise SystemExit(main())
