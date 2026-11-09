"""Run one bench candidate over one bench set.

Usage::

    python -m scripts.asr_bench.run --set <set dir> --candidate <json>
        [--name <candidate>] --out <run dir> [--workers N]

Each item's audio is fed in 480-sample capture blocks (the 30 ms block the
shipped capture produces) through the production ``SileroChunker`` built from
the shipped ``default_config.yaml`` (plus the candidate's ``chunker``
overrides), followed by a short stretch of digital silence so a window still
open at the end of the item can close. Every emitted window is decoded with
the candidate's ASR backend, configured by the shipped ``asr`` section plus
the candidate's ``asr_config`` overrides, and written to ``windows.jsonl``
with the audio position at which the chunker emitted it and the wall time of
its decode; ``run.json`` records the candidate, the effective chunker and ASR
settings, the set, the code and package versions, the GPU the timings come
from (``nvidia-smi``) and the timing.

A candidate ``frontend`` replaces the audio the chunker hears (feed ``vad``
or ``both``) and/or the audio each window's decode hears (feed ``asr`` or
``both``, cut from the same sample span) with precomputed per-item files.
A ``vad_assist`` block raises the speech probability the chunker receives,
frame by frame, to the maximum of the candidate's Silero, a second Silero
over a precomputed frontend and a precomputed sound-tag term; the chunker's
``FrameGate`` consumes the fused value unchanged and the decode hears the
same audio as without the assist. With a sound-tag term, the windows'
emission positions follow production's delay line (``SoundTagAssist``):
a frame reaches the gate only once the tagger has tagged its hop, under
the hop and look-ahead the tags directory's manifest records
(``tag_sounds.TagLayout``).

A candidate ``strategy`` replaces the single decode of each window: its
mode's factory in ``_STRATEGIES`` builds a ``WindowStrategy`` that gets each
item's ASR audio and per-frame speech probabilities, turns each window into
its record fields and reports a ``strategy`` section for ``run.json``
(``island_recover``: ``scripts/asr_bench/island.py``; ``context_decode``:
``scripts/asr_bench/context_decode.py``; ``incremental_emit``:
``scripts/asr_bench/incremental.py``; ``fw_kwargs``:
``scripts/asr_bench/fw_kwargs.py``). With a ``vad_assist`` those
probabilities are the fused ones the chunker's gate received.

Window spans come from a second ``FrameGate`` that receives the same speech
probabilities as the chunker's own gate but frame ordinals instead of audio
(the E0 replay's coordinate technique), so every span is the exact sample
range of the emitted audio. The parakeet backend needs the bench venv
(``onnx-asr``), the sherpa backend a venv with ``sherpa-onnx``;
faster-whisper runs from either venv.

``--workers N`` splits the items over N spawned processes that each load
the models once and take the next item when they finish one; the windows
are written in set order, so ``windows.jsonl`` is the one-process run's
apart from faster-whisper's temperature fallback, which samples. The
processes share the GPU and the CPU, so each window's ``asr_s`` and the
``timing`` (summed over the processes) include their contention:
``run.json`` then says ``timing_valid`` false and the scorer compares no
timing of the run. ``throughput`` records every run's wall clock, the basis
of RunPod cost estimates. Processes, not CTranslate2's ``inter_threads``
(faster-whisper's ``num_workers``): those share one model copy but run in
parallel only inside CTranslate2 and need production's ``build_asr`` to
create the model differently, while the chunker, Silero and each
strategy's Python stay under one GIL.
"""

from __future__ import annotations

from yamabiko import hf_env  # noqa: F401  # pin the model cache first

import argparse
import asyncio
import copy
import dataclasses
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from scripts.asr_bench import context_decode, fw_kwargs, incremental, island
from scripts.asr_bench.candidates import (
    AsrSpec,
    AssistSileroSpec,
    Candidate,
    FrontendSpec,
    VadAssistSpec,
    load_candidate,
    model_artifact_sha256,
    resolve_vad,
    shipped_config,
)
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, BenchSet, Item, load_set
from scripts.asr_bench.tag_sounds import TagLayout, TagTrack, load_layout, load_tags, silero_frames
from scripts.e0_evidence.wire import canonical_sha256
from yamabiko.vad.chunker import ChunkerConfig, chunker_config_from_config
from yamabiko.vad.silero_chunker import FrameGate, SileroChunker, SileroVAD

RUN_SCHEMA_VERSION = "asr-bench-run-1.3"
# Run schemas the scorer reads; 1.0 runs carry no emission timing, 1.1 runs
# with sound tags time emission without the tag delay line, and runs before
# 1.3 ran in one process (no ``workers``, ``timing_valid`` or ``throughput``).
RUN_SCHEMA_VERSIONS = (
    "asr-bench-run-1.0", "asr-bench-run-1.1", "asr-bench-run-1.2", RUN_SCHEMA_VERSION,
)

_BLOCK_SAMPLES = 480
_FRAME_SAMPLES = 512
# Frame ordinals travel through the gate as float32; integers stay exact
# below 2**24, so items may run for up to 2**24 frames (about 149 hours).
_MAX_FRAMES = 1 << 24
_PACKAGES = (
    "ctranslate2", "faster-whisper", "onnxruntime", "onnx-asr", "sherpa-onnx",
    "torch", "transformers", "numpy",
)
# The reazonspeech k2 library's recognizer settings: greedy search over
# 80-dim fbank. The thread count only changes speed.
_SHERPA_FEATURE_DIM = 80
_SHERPA_DECODING = "greedy_search"
_SHERPA_THREADS = 4
# transformers-native ASR model types the transformers_asr backend decodes,
# each with the language it is forced to (the model's own spelling).
TRANSFORMERS_MODEL_TYPES = {"qwen3_asr": "Japanese", "cohere_asr": "ja"}
# A window is at most max_chunk_ms plus two pads; Japanese runs well under
# 20 tokens per second, so this bound only stops a runaway loop.
_TRANSFORMERS_MAX_NEW_TOKENS = 256
_TRANSFORMERS_DEVICE = "cuda"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_log = logging.getLogger("asr_bench.run")

Decoder = Callable[[np.ndarray], tuple[str, str]]
# A built decoder: the decode function, its identity for run.json, and the
# production ASR backend object (``FasterWhisperASR``; None for the others).
BuiltDecoder = tuple[Decoder, dict[str, Any], Any]


class _FeedQueue(asyncio.Queue):
    """Capture queue that signals once the chunker asks for a block after the
    feeder has queued the last one, i.e. every block has been processed."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self.fed = False
        self.drained = asyncio.Event()

    async def get(self):
        if self.fed and self.empty():
            self.drained.set()
        return await super().get()


_ASSIST_COUNTS = ("frames", "frames_raised", "frames_lifted", "lifted_by_silero", "lifted_by_tags")


class _AssistedVAD:
    """The candidate's Silero on the chunker's audio, fused frame by frame
    with the ``vad_assist`` tracks: the maximum of its probability, the
    second Silero's on the assist audio of the same frame, and the tag
    term of that frame. The chunker (and the coordinate gate) receive the
    fused value unchanged; ``reset_states`` resets both Silero models, as
    the chunker resets its model at every true window close.

    ``counts`` tallies frames: raised (fused above the primary), lifted
    (primary below the gate threshold, fused at or above it), and lifted by
    each track on its own.
    """

    def __init__(self, primary: SileroVAD, assist: SileroVAD | None, threshold: float) -> None:
        self._primary = primary
        self._assist = assist
        self._threshold = threshold
        self._audio: np.ndarray | None = None
        self._tags: np.ndarray | None = None
        self._next_frame = 0
        self.counts = dict.fromkeys(_ASSIST_COUNTS, 0)

    def start_item(self, audio: np.ndarray | None, tags: np.ndarray | None) -> None:
        """``audio``: the assist audio with the chunker's trailing silence;
        ``tags``: the tag term per chunker frame."""
        if (audio is None) != (self._assist is None):
            raise ValueError("assist audio must be given exactly when a second Silero runs")
        self._audio, self._tags, self._next_frame = audio, tags, 0

    def __call__(self, frame: np.ndarray) -> float:
        probability = float(self._primary(frame))
        ordinal = self._next_frame
        self._next_frame += 1
        below = probability < self._threshold
        fused = probability
        if self._audio is not None:
            start = ordinal * _FRAME_SAMPLES
            second = float(self._assist(self._audio[start:start + _FRAME_SAMPLES]))
            self.counts["lifted_by_silero"] += below and second >= self._threshold
            fused = max(fused, second)
        if self._tags is not None:
            tag = float(self._tags[ordinal])
            self.counts["lifted_by_tags"] += below and tag >= self._threshold
            fused = max(fused, tag)
        self.counts["frames"] += 1
        self.counts["frames_raised"] += fused > probability
        self.counts["frames_lifted"] += below and fused >= self._threshold
        return fused

    def reset_states(self) -> None:
        self._primary.reset_states()
        if self._assist is not None:
            self._assist.reset_states()


def _tag_term(track: TagTrack, spec: VadAssistSpec, frame_count: int) -> np.ndarray:
    """The ``vad_assist.tags`` term per chunker frame: over the item and the
    track's tagged tail of zeros, then 0."""
    tags = spec.tags
    assert tags is not None
    missing = [group for group in tags.groups if group not in track.names]
    if missing:
        raise ValueError(f"vad_assist tags {tags.name}: groups {missing} not in {track.names}")
    total = np.sum([track.column(group, tail=True) for group in tags.groups], axis=0)
    if tags.threshold is not None:
        term = (total >= tags.threshold).astype(np.float32)
    else:
        term = np.minimum(np.float32(1.0), np.float32(tags.weight) * total)
    return silero_frames(term, frame_count)


def _decision_sample(frame_ordinal: int, total_samples: int,
                     layout: TagLayout | None = None) -> int:
    """The audio position at which the gate has decided frame
    ``frame_ordinal``: one past the last sample fed to the chunker when the
    gate received the frame. Without sound tags that is the end of the
    capture block holding the frame, capped at the ``total_samples`` fed.
    With tags in ``layout`` the frame waits in production's delay line
    until the chunker has the audio its tag window needs: the end of the
    capture block holding the last sample ``layout.release_sample`` names.
    That is not capped: a live stream keeps delivering audio, so a frame in
    the last second of the zero tail may be decided after it."""
    if layout is None:
        frame_end = (frame_ordinal + 1) * _FRAME_SAMPLES
        return min(-(-frame_end // _BLOCK_SAMPLES) * _BLOCK_SAMPLES, total_samples)
    release = layout.release_sample(frame_ordinal)
    return -(-release // _BLOCK_SAMPLES) * _BLOCK_SAMPLES


def _emission_record(layout: TagLayout | None) -> dict[str, Any]:
    if layout is None:
        return {
            "decision": "chunker_frame", "tag_layout": None,
            "definition": "emitted_at_s: end of the 480-sample capture block holding the frame "
                          "that emitted the window, capped at the samples fed",
        }
    return {
        "decision": "tag_release", "tag_layout": layout.to_json(),
        "definition": "emitted_at_s: end of the 480-sample capture block holding the last "
                      "sample of the sound-tag window that tags the emitting frame's hop "
                      "(production's SoundTagAssist delay line under tag_layout; "
                      "tag_sounds.TagLayout.release_sample), not capped at the samples fed; "
                      "the tagger's own inference time is not included",
    }


class _SpanProbe:
    """Silero wrapper that mirrors every probability into a coordinate gate.

    The chunker calls this once per 512-sample frame, then pushes the same
    probability into its own gate, so both gates emit at the same frames and
    each coordinate emission names the frames of the matching audio window.
    Every probability is also kept, indexed by frame ordinal, for the window
    strategy, and each span records the frame at which its window went live:
    the frame that opened the gate, or for a later piece of a dip-split
    utterance the frame that emitted the piece before it.
    """

    def __init__(self, model: SileroVAD | _AssistedVAD, config: ChunkerConfig,
                 threshold: float) -> None:
        self._model = model
        self.gate = FrameGate(config, threshold=threshold, frame_samples=_FRAME_SAMPLES)
        self._next_frame = 0
        # (start sample, end sample, emitting frame, live-from frame).
        self.spans: deque[tuple[int, int, int, int]] = deque()
        self.probabilities: list[float] = []
        self._live_from: int | None = None

    def __call__(self, frame: np.ndarray) -> float:
        probability = float(self._model(frame))
        ordinal = self._next_frame
        if ordinal >= _MAX_FRAMES:
            raise RuntimeError("item too long for exact frame coordinates")
        self.probabilities.append(probability)
        self._next_frame += 1
        coordinates = np.full(_FRAME_SAMPLES, ordinal, dtype=np.float32)
        emitted = self.gate.push(coordinates, probability)
        if emitted is not None:
            if emitted.size == 0 or emitted.size % _FRAME_SAMPLES:
                raise RuntimeError("coordinate gate emitted a partial or empty window")
            ordinals = emitted.reshape(-1, _FRAME_SAMPLES)[:, 0].astype(np.int64)
            if np.any(np.diff(ordinals) != 1):
                raise RuntimeError("coordinate gate emitted non-contiguous frames")
            # The window comes out while the chunker processes this frame.
            if self._live_from is None:
                raise RuntimeError("coordinate gate emitted a window it never opened")
            self.spans.append((
                int(ordinals[0]) * _FRAME_SAMPLES,
                (int(ordinals[-1]) + 1) * _FRAME_SAMPLES,
                ordinal,
                self._live_from,
            ))
            # A dip cut leaves the gate open: the next piece is live from here.
            self._live_from = ordinal if self.gate.state_snapshot().in_speech else None
        elif self._live_from is None and self.gate.state_snapshot().in_speech:
            self._live_from = ordinal
        return probability

    def reset_states(self) -> None:
        self._model.reset_states()


@dataclasses.dataclass(frozen=True)
class Window:
    start_sample: int
    end_sample: int
    # One past the last sample fed to the chunker when the window came out:
    # the decision position (``_decision_sample``) of the frame that emitted it.
    emitted_at_sample: int
    audio: np.ndarray
    emit_reason: str
    # The same clock when the gate first held this window open: the
    # decision position of the frame that opened it (for a later dip piece,
    # of the frame that emitted the piece before it). The window's audio
    # from start_sample up to the decided frames is known from then on.
    opened_at_sample: int


async def _chunk_item(
    pcm: np.ndarray,
    *,
    config: ChunkerConfig,
    threshold: float,
    vad_model: SileroVAD | _AssistedVAD,
    flush_samples: int,
    genre: str,
    decision_sample: Callable[[int, int], int],
) -> tuple[list[Window], bool, np.ndarray]:
    """Feed one item through a fresh production chunker.

    Returns the emitted windows in order, whether the gate still held an
    open window after the trailing silence, and the speech probability of
    every frame fed (frame ``k`` covers samples ``512 k`` to ``512 (k + 1)``).
    ``decision_sample(frame, fed samples)`` places each window's opening and
    emission on the audio clock.
    """
    from yamabiko.capture.envelope import CaptureBlock
    from yamabiko.state import AppState

    vad_model.reset_states()
    probe = _SpanProbe(vad_model, config, threshold)
    chunker = SileroChunker(config, threshold=threshold, state=AppState(genre=genre), model=probe)
    audio = np.concatenate([pcm.astype(np.float32, copy=False), np.zeros(flush_samples, np.float32)])
    audio_q = _FeedQueue(maxsize=64)
    chunk_q: asyncio.Queue = asyncio.Queue()

    async def _feed() -> None:
        for start in range(0, audio.size, _BLOCK_SAMPLES):
            await audio_q.put(CaptureBlock(
                pcm=audio[start:start + _BLOCK_SAMPLES].copy(),
                stream_generation=0,
                start_sample=start,
            ))
        audio_q.fed = True

    feeder = asyncio.create_task(_feed(), name="feed")
    vad = asyncio.create_task(chunker.run(audio_q, chunk_q), name="vad-chunker")
    drained = asyncio.create_task(audio_q.drained.wait(), name="drained")
    try:
        await asyncio.wait({vad, drained}, return_when=asyncio.FIRST_COMPLETED)
        if vad.done():
            error = vad.exception()
            raise RuntimeError("the chunker stopped before the item ended") from error
    finally:
        for task in (feeder, vad, drained):
            task.cancel()
        await asyncio.gather(feeder, vad, drained, return_exceptions=True)
    if chunker.discontinuity_resets:
        raise RuntimeError(f"unexpected chunker resets: {dict(chunker.discontinuity_resets)}")

    windows: list[Window] = []
    while not chunk_q.empty():
        segment = chunk_q.get_nowait()
        if not probe.spans:
            raise RuntimeError("chunker emitted a window the coordinate gate did not")
        start, end, ordinal, live_from = probe.spans.popleft()
        if segment.audio.size != end - start or not np.array_equal(
            segment.audio, audio[start:end]
        ):
            raise RuntimeError("window audio does not match its coordinate span")
        windows.append(Window(
            start, end, decision_sample(ordinal, audio.size), segment.audio,
            segment.vad_emit_reason, opened_at_sample=decision_sample(live_from, audio.size),
        ))
    if probe.spans:
        raise RuntimeError("coordinate gate emitted a window the chunker did not")
    return windows, probe.gate.state_snapshot().in_speech, np.asarray(probe.probabilities)


def _check_frontend(spec: FrontendSpec | AssistSileroSpec, bench_set: BenchSet) -> None:
    """Fail before any model loads if a frontend file is missing or sized wrong."""
    for item in bench_set.items:
        path = spec.audio_path(item.item_id)
        if not path.is_file():
            raise FileNotFoundError(f"frontend {spec.name}: {path} is missing")
        if path.stat().st_size != 4 * item.sample_count:
            raise ValueError(
                f"frontend {spec.name}: {path} holds {path.stat().st_size} bytes, "
                f"expected {4 * item.sample_count} ({item.sample_count} float32 samples)"
            )


def _load_frontend(spec: FrontendSpec | AssistSileroSpec, item: Item) -> tuple[np.ndarray, str]:
    """The item's frontend audio and the SHA-256 of its file."""
    path = spec.audio_path(item.item_id)
    data = path.read_bytes()
    if len(data) != 4 * item.sample_count:
        raise ValueError(
            f"frontend {spec.name}: {path} has {len(data)} bytes, "
            f"expected {4 * item.sample_count}"
        )
    pcm = np.frombuffer(data, dtype="<f4")
    if not np.all(np.isfinite(pcm)):
        raise ValueError(f"frontend {spec.name}: {path} holds non-finite samples")
    return pcm, hashlib.sha256(data).hexdigest()


def _check_vad_assist(spec: VadAssistSpec, bench_set: BenchSet,
                      layout: TagLayout | None) -> None:
    """Fail before any model loads if an assist file is missing or does not
    fit its item, or a tag file lacks a selected group or the tail its
    manifest's ``layout`` names."""
    if spec.silero is not None:
        _check_frontend(spec.silero, bench_set)
    if spec.tags is not None:
        assert layout is not None
        for item in bench_set.items:
            track = load_tags(spec.tags.tags_path(item.item_id), item.sample_count)
            missing = [group for group in spec.tags.groups if group not in track.names]
            if missing:
                raise ValueError(f"vad_assist tags {spec.tags.name}: {item.item_id} lacks "
                                 f"groups {missing}")
            if track.tail_frames != layout.tail_frames:
                raise ValueError(f"vad_assist tags {spec.tags.name}: {item.item_id} has a tail "
                                 f"of {track.tail_frames} frames, its manifest "
                                 f"{layout.tail_frames}")


def _assist_record(spec: VadAssistSpec, counts: Mapping[str, int],
                   audio_files: Mapping[str, str], tag_files: Mapping[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "fusion": "max(silero(chunker audio), silero(assist audio), tag term) per 512-sample "
                  "frame, fed unchanged to the production chunker",
        "counts": dict(counts),
        "silero": None, "tags": None,
    }
    if spec.silero is not None:
        record["silero"] = {
            "name": spec.silero.name, "audio_dir": spec.silero.audio_dir,
            "model": "the candidate's vad graph, own state",
            "files_sha256": dict(audio_files), "manifest_sha256": canonical_sha256(audio_files),
        }
    if spec.tags is not None:
        tags = spec.tags
        record["tags"] = {
            "name": tags.name, "tags_dir": tags.tags_dir, "groups": list(tags.groups),
            "rule": tags.rule, "weight": tags.weight, "threshold": tags.threshold,
            "grid": "chunker frame k takes the 40 ms tag frame holding sample 512 k + 256; "
                    "past the item, the track's tagged tail of zeros (emission.tag_layout "
                    "tail_frames), then 0",
            "files_sha256": dict(tag_files), "manifest_sha256": canonical_sha256(tag_files),
        }
    return record


def _faster_whisper_decoder(
    cfg: dict[str, Any], candidate: Candidate
) -> BuiltDecoder:
    return _whisper_decoder(cfg["asr"], candidate.asr)


def _whisper_decoder(asr_section: dict[str, Any], spec: AsrSpec) -> BuiltDecoder:
    """Production ``build_asr`` and ``transcribe`` for ``spec``'s model, set
    up by an ``asr`` config section (already overridden)."""
    from yamabiko.asr.asr_client import build_asr
    from yamabiko.models import whisper_revision_for

    asr_cfg = copy.deepcopy(asr_section)
    settings = asr_cfg["faster_whisper"]
    settings["model"] = spec.model
    settings["compute_type"] = spec.compute_type
    model_dir = spec.local_model_dir
    if model_dir is not None:
        if not (model_dir / "model.bin").is_file():
            raise ValueError(f"{model_dir} is not a CTranslate2 model directory")
        model = {"kind": "local_dir", "artifact_sha256": model_artifact_sha256(model_dir)}
    else:
        model = {"kind": "hf_repo", "repo": spec.model, "revision": whisper_revision_for(spec.model)}
    asr = build_asr(asr_cfg)
    asr.warmup()

    def decode(audio: np.ndarray) -> tuple[str, str]:
        result = asr.transcribe(audio, SAMPLE_RATE_HZ)
        return result.text, result.outcome.value

    return decode, {"backend": spec.backend, "model": model, "effective_config": asr_cfg}, asr


def _text_only_result(raw_text: str) -> tuple[str, str]:
    """Production's text-derived wrapper gates for a decoder without Whisper
    confidence scores: the empty and compression-loop drops."""
    from scripts.e0_evidence.ledger import text_only_wrapper_gate
    from yamabiko.asr.asr_client import ASROutcome

    gate = text_only_wrapper_gate(raw_text)
    if not gate.assembled_text:
        return "", ASROutcome.DEPENDENCY_NO_SEGMENTS.value
    if gate.compression_rejected:
        return "", ASROutcome.COMPRESSION_LOOP_REJECTED.value
    return gate.assembled_text, ASROutcome.TEXT_AVAILABLE.value


def _boundary_pad(cfg: dict[str, Any]) -> tuple[int, np.ndarray]:
    """``asr.boundary_pad_ms`` after overrides, and that much silence."""
    boundary_pad_ms = int(cfg["asr"]["boundary_pad_ms"])
    return boundary_pad_ms, np.zeros(SAMPLE_RATE_HZ * boundary_pad_ms // 1000, dtype=np.float32)


def _parakeet_decoder(
    cfg: dict[str, Any], candidate: Candidate
) -> BuiltDecoder:
    from scripts.e0_evidence.nemo_tdt import (
        EXECUTION_PROVIDER,
        installed_runtime_versions,
        load_nemo_tdt,
    )

    installed_runtime_versions()
    model_dir = candidate.asr.local_model_dir
    assert model_dir is not None
    digest = model_artifact_sha256(model_dir)
    boundary_pad_ms = int(cfg["asr"]["boundary_pad_ms"])
    decoder = load_nemo_tdt(model_dir, digest, boundary_pad_ms=boundary_pad_ms)
    pad = np.zeros(decoder.boundary_pad_samples, dtype=np.float32)

    def decode(audio: np.ndarray) -> tuple[str, str]:
        # Padded exactly as production pads a Whisper call; only the
        # text-derived wrapper gates apply (no confidence scores).
        padded = np.concatenate([pad, audio, pad]) if pad.size else audio
        result = decoder.model.recognize(padded, sample_rate=SAMPLE_RATE_HZ)
        return _text_only_result(result.text)

    return decode, {
        "backend": candidate.asr.backend,
        "model": {"kind": "local_dir", "artifact_sha256": digest},
        "execution_provider": EXECUTION_PROVIDER,
        "boundary_pad_ms": boundary_pad_ms,
    }, None


def sherpa_transducer_files(model_dir: Path) -> dict[str, Path]:
    """The fp32 ``encoder``/``decoder``/``joiner`` ONNX files and ``tokens.txt``
    of a sherpa-onnx transducer export; ``*.int8.*`` files are ignored."""
    files: dict[str, Path] = {}
    for part in ("encoder", "decoder", "joiner"):
        matches = sorted(
            path for path in model_dir.glob(f"{part}*.onnx") if ".int8." not in path.name
        )
        if len(matches) != 1:
            found = [path.name for path in matches]
            raise ValueError(f"{model_dir} needs exactly one fp32 {part}*.onnx, found {found}")
        files[part] = matches[0]
    files["tokens"] = model_dir / "tokens.txt"
    if not files["tokens"].is_file():
        raise ValueError(f"{model_dir} has no tokens.txt")
    return files


def _sherpa_transducer_decoder(
    cfg: dict[str, Any], candidate: Candidate
) -> BuiltDecoder:
    import sherpa_onnx

    model_dir = candidate.asr.local_model_dir
    assert model_dir is not None
    files = sherpa_transducer_files(model_dir)
    digest = model_artifact_sha256(model_dir)
    boundary_pad_ms, pad = _boundary_pad(cfg)
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=str(files["tokens"]),
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        joiner=str(files["joiner"]),
        num_threads=_SHERPA_THREADS,
        sample_rate=SAMPLE_RATE_HZ,
        feature_dim=_SHERPA_FEATURE_DIM,
        decoding_method=_SHERPA_DECODING,
        provider="cpu",
    )

    def decode(audio: np.ndarray) -> tuple[str, str]:
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE_HZ, np.concatenate([pad, audio, pad]))
        recognizer.decode_stream(stream)
        return _text_only_result(stream.result.text)

    return decode, {
        "backend": candidate.asr.backend,
        "model": {"kind": "local_dir", "artifact_sha256": digest},
        "files": {part: path.name for part, path in files.items()},
        "execution_provider": "CPUExecutionProvider",
        "num_threads": _SHERPA_THREADS,
        "feature_dim": _SHERPA_FEATURE_DIM,
        "decoding_method": _SHERPA_DECODING,
        "boundary_pad_ms": boundary_pad_ms,
    }, None


def _transformers_decoder(
    cfg: dict[str, Any], candidate: Candidate
) -> BuiltDecoder:
    import torch
    import transformers

    spec = candidate.asr
    model_dir = spec.local_model_dir
    assert model_dir is not None
    model_type = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type not in TRANSFORMERS_MODEL_TYPES:
        raise ValueError(
            f"{model_dir} has model_type {model_type!r}; transformers_asr decodes "
            f"{sorted(TRANSFORMERS_MODEL_TYPES)}"
        )
    language = TRANSFORMERS_MODEL_TYPES[model_type]
    digest = model_artifact_sha256(model_dir)
    boundary_pad_ms, pad = _boundary_pad(cfg)
    dtype = getattr(torch, spec.compute_type)
    num_beams = spec.beam_size or 1
    processor = transformers.AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model_class = (
        transformers.AutoModelForMultimodalLM if model_type == "qwen3_asr"
        else transformers.AutoModelForSpeechSeq2Seq
    )
    model = model_class.from_pretrained(model_dir, dtype=dtype, local_files_only=True)
    model = model.to(_TRANSFORMERS_DEVICE).eval()
    generation = model.generation_config
    if model_type == "cohere_asr" and generation.decoder_start_token_id is None:
        # A checkpoint without generation_config.json (efwkjn's) leaves the
        # start id unset. The processor always passes the whole decoder
        # prompt; its first token is the start id Cohere-derived checkpoints
        # declare (Kitsune T-0.6B's generation_config.json names it too).
        generation.decoder_start_token_id = processor.get_decoder_prompt_ids(language)[0]
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(generation, key) is None:
            setattr(generation, key, getattr(model.config, key, None))
    generate_kwargs = {
        "max_new_tokens": _TRANSFORMERS_MAX_NEW_TOKENS, "num_beams": num_beams, "do_sample": False,
    }

    def transcribe(audio: np.ndarray) -> str:
        padded = np.concatenate([pad, audio, pad])
        with torch.inference_mode():
            if model_type == "qwen3_asr":
                inputs = processor.apply_transcription_request(audio=[padded], language=language)
                inputs = inputs.to(model.device, model.dtype)
                output = model.generate(**inputs, **generate_kwargs)
                new_tokens = output[:, inputs["input_ids"].shape[1]:]
                return processor.decode(new_tokens, return_format="transcription_only")[0]
            inputs = processor(
                audio=padded, language=language, punctuation=True, sampling_rate=SAMPLE_RATE_HZ,
            ).to(model.device, model.dtype)
            output = model.generate(**inputs, **generate_kwargs)
            return processor.decode(output[0], skip_special_tokens=True)

    transcribe(np.zeros(SAMPLE_RATE_HZ, dtype=np.float32))  # warm up, as build_asr does

    def decode(audio: np.ndarray) -> tuple[str, str]:
        return _text_only_result(transcribe(audio))

    return decode, {
        "backend": spec.backend,
        "model": {"kind": "local_dir", "artifact_sha256": digest},
        "model_type": model_type,
        "language": language,
        "dtype": spec.compute_type,
        "device": str(model.device),
        "attn_implementation": model.config._attn_implementation,
        "generate": generate_kwargs,
        "special_token_ids": {
            key: getattr(generation, key)
            for key in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id")
        },
        "boundary_pad_ms": boundary_pad_ms,
    }, None


_DECODERS: dict[str, Callable[[dict[str, Any], Candidate], BuiltDecoder]] = {
    "faster_whisper": _faster_whisper_decoder,
    "parakeet_onnx": _parakeet_decoder,
    "sherpa_transducer": _sherpa_transducer_decoder,
    "transformers_asr": _transformers_decoder,
}


@dataclasses.dataclass(frozen=True)
class StrategyContext:
    """What a window strategy is built from."""

    # The shipped config with the candidate's chunker and ASR overrides.
    cfg: dict[str, Any]
    candidate: Candidate
    # The candidate's decoder, and its production ASR object (None unless
    # faster_whisper) for a strategy that drives the loaded model itself.
    decode: Decoder
    backend: Any
    chunker: ChunkerConfig
    threshold: float
    # Samples per speech probability.
    frame_samples: int
    # Builds a further faster-whisper decoder from overrides of the shipped
    # asr section and a faster_whisper AsrSpec.
    whisper_decoder: Callable[[Mapping[str, Any], AsrSpec], BuiltDecoder]
    # decision_sample(frame, fed samples): the audio position at which the
    # chunker's gate has decided that frame (Window.emitted_at_sample and
    # opened_at_sample use it). Monotonic in the frame; frames may share one.
    decision_sample: Callable[[int, int], int]


class WindowStrategy(Protocol):
    """How each emitted window becomes its record; built once per run."""

    def start_item(self, item: Item, asr_audio: np.ndarray, probabilities: np.ndarray) -> None:
        """Before an item's windows: the audio windows are decoded from (the
        item's ASR audio plus the chunker's zero tail, indexed like window
        samples) and the speech probability of every frame the chunker
        heard (frame ``k`` covers samples ``frame_samples * k`` onward)."""

    def decode_window(self, window: Window) -> dict[str, Any]:
        """The window's record fields: ``text`` and ``asr_outcome`` at least.
        They replace the runner's own (``emit_reason``, ``emitted_at_s``,
        ``asr_s``, the wall time of this call) and may add keys; ``item_id``,
        ``window_index``, ``start_s`` and ``end_s`` stay the runner's."""

    def record(self) -> dict[str, Any] | None:
        """The run.json ``strategy`` section, after the last item."""


class _PlainStrategy:
    """One decode of each window's audio."""

    def __init__(self, context: StrategyContext) -> None:
        self._decode = context.decode
        self._audio = np.zeros(0, np.float32)

    def start_item(self, item: Item, asr_audio: np.ndarray, probabilities: np.ndarray) -> None:
        self._audio = asr_audio

    def decode_window(self, window: Window) -> dict[str, Any]:
        text, outcome = self._decode(self._audio[window.start_sample:window.end_sample])
        return {"text": text, "asr_outcome": outcome}

    def record(self) -> None:
        return None


# Strategy mode (candidates.STRATEGY_PARSERS) -> factory.
_STRATEGIES: dict[str, Callable[[StrategyContext], WindowStrategy]] = {
    context_decode.MODE: context_decode.make_strategy,
    fw_kwargs.MODE: fw_kwargs.make_strategy,
    "island_recover": island.IslandRecover,
    incremental.MODE: incremental.IncrementalEmit,
}
# Strategy mode -> the keys of its run.json section that add up over items.
# With --workers each process reports its own section; the run's section
# sums these keys (numbers, or counts per label) and takes each other key,
# which the processes must agree on, once. The modes not listed summarize
# all of a run's windows in percentiles and run in one process only.
_ADDITIVE_STRATEGY_KEYS: dict[str, frozenset[str]] = {
    fw_kwargs.MODE: fw_kwargs.ADDITIVE_RECORD_KEYS,
    "island_recover": island.ADDITIVE_RECORD_KEYS,
}
_FIXED_WINDOW_KEYS = frozenset({"item_id", "window_index", "start_s", "end_s"})


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _git_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        return {
            "head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _gpu_identity() -> dict[str, Any]:
    """The GPU this run's timings come from, as ``nvidia-smi`` names it: CUDA
    device 0, i.e. the first entry of ``CUDA_VISIBLE_DEVICES`` when it is set
    (an index or a UUID), else index 0. ``model`` is None without a visible
    GPU or without ``nvidia-smi``."""
    identity: dict[str, Any] = {"model": None, "uuid": None, "driver": None,
                                "source": "nvidia-smi"}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    wanted = "0" if visible is None else visible.split(",")[0].strip()
    if not wanted:
        return identity
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return identity
    for line in listing.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and wanted in fields[:2]:
            identity.update(model=fields[2], uuid=fields[1], driver=fields[3])
            break
    return identity


def _apply_overrides(section: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Write validated overrides into a config section, nested sections key by key."""
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            _apply_overrides(section[key], value)
        else:
            section[key] = value


def _shipped_whisper_decoder(overrides: Mapping[str, Any], spec: AsrSpec) -> BuiltDecoder:
    """A further faster-whisper decoder: ``spec``'s model under the shipped
    ``asr`` section with ``overrides`` (validated like a candidate's
    ``asr_config``)."""
    section = shipped_config()["asr"]
    _apply_overrides(section, overrides)
    return _whisper_decoder(section, spec)


@dataclasses.dataclass(frozen=True)
class _Setup:
    """What every process of a run derives from the shipped config and the candidate."""

    # The shipped config with the candidate's chunker and ASR overrides.
    cfg: dict[str, Any]
    config: ChunkerConfig
    threshold: float
    # Digital silence fed after each item, so a window still open can close.
    flush_samples: int
    # The sound tags' layout; None without a vad_assist tags track.
    layout: TagLayout | None
    genre: str


def _setup(candidate: Candidate) -> _Setup:
    cfg = shipped_config()
    if cfg["audio"]["sample_rate"] != SAMPLE_RATE_HZ:
        raise ValueError("the shipped config is not 16 kHz")
    cfg["vad"].update(candidate.vad_overrides)
    _apply_overrides(cfg["asr"], candidate.asr_overrides)
    config, threshold = chunker_config_from_config(cfg)
    flush_ms = max(config.min_silence_ms, config.glue_max_silence_ms) + 1000
    assist = candidate.vad_assist
    layout = None
    if assist is not None and assist.tags is not None:
        layout = load_layout(Path(assist.tags.tags_dir))
    return _Setup(cfg, config, threshold, SAMPLE_RATE_HZ * flush_ms // 1000, layout,
                  cfg.get("genre", "variety"))


@dataclasses.dataclass(frozen=True)
class _ItemResult:
    """One item's ``windows.jsonl`` lines and what run.json sums over items."""

    lines: list[str]
    vad_seconds: float
    asr_seconds: float
    # The gate still held a window open after the trailing silence.
    open_at_end: bool
    # SHA-256 of the frontend, assist audio and tag files read; None when not read.
    frontend_sha256: str | None
    assist_audio_sha256: str | None
    assist_tags_sha256: str | None


class _Pipeline:
    """One process's models, built once, and the work on each item: the
    production chunker, then every window through the strategy."""

    def __init__(self, bench_set: BenchSet, candidate: Candidate, setup: _Setup) -> None:
        begin = time.perf_counter()
        self._bench_set = bench_set
        self._candidate = candidate
        self._setup = setup
        artifact = self.vad_artifact = resolve_vad(candidate.vad)
        vad_model: SileroVAD | _AssistedVAD = SileroVAD(
            artifact.path, SAMPLE_RATE_HZ, expected_sha256=artifact.sha256)
        assist = candidate.vad_assist
        if assist is not None:
            second = None
            if assist.silero is not None:
                second = SileroVAD(artifact.path, SAMPLE_RATE_HZ, expected_sha256=artifact.sha256)
            vad_model = _AssistedVAD(vad_model, second, setup.threshold)
        self._vad_model = vad_model
        decode, self.asr_identity, backend = _DECODERS[candidate.asr.backend](setup.cfg, candidate)
        self._context = StrategyContext(
            cfg=setup.cfg, candidate=candidate, decode=decode, backend=backend, chunker=setup.config,
            threshold=setup.threshold, frame_samples=_FRAME_SAMPLES,
            whisper_decoder=_shipped_whisper_decoder,
            decision_sample=functools.partial(_decision_sample, layout=setup.layout),
        )
        self.strategy: WindowStrategy = (
            _PlainStrategy(self._context) if candidate.strategy is None
            else _STRATEGIES[candidate.strategy.mode](self._context)
        )
        self.load_seconds = time.perf_counter() - begin

    def assist_counts(self) -> dict[str, int] | None:
        model = self._vad_model
        return dict(model.counts) if isinstance(model, _AssistedVAD) else None

    def process(self, item: Item) -> _ItemResult:
        setup = self._setup
        flush_samples = setup.flush_samples
        frontend = self._candidate.frontend
        assist = self._candidate.vad_assist
        pcm = self._bench_set.load_audio(item)
        vad_pcm = asr_pcm = pcm
        frontend_sha256 = assist_audio_sha256 = assist_tags_sha256 = None
        if frontend is not None:
            frontend_pcm, frontend_sha256 = _load_frontend(frontend, item)
            if frontend.feeds_vad:
                vad_pcm = frontend_pcm
            if frontend.feeds_asr:
                asr_pcm = frontend_pcm
        vad_model = self._vad_model
        if isinstance(vad_model, _AssistedVAD):
            assert assist is not None
            frame_count = -(-(item.sample_count + flush_samples) // _FRAME_SAMPLES)
            assist_pcm = assist_tags = None
            if assist.silero is not None:
                stem, assist_audio_sha256 = _load_frontend(assist.silero, item)
                assist_pcm = np.concatenate([stem, np.zeros(flush_samples, np.float32)])
            if assist.tags is not None:
                track = load_tags(assist.tags.tags_path(item.item_id), item.sample_count)
                assist_tags_sha256 = track.sha256
                assist_tags = _tag_term(track, assist, frame_count)
            vad_model.start_item(assist_pcm, assist_tags)
        begin = time.perf_counter()
        windows, pending, probabilities = asyncio.run(_chunk_item(
            vad_pcm, config=setup.config, threshold=setup.threshold, vad_model=vad_model,
            flush_samples=flush_samples, genre=setup.genre,
            decision_sample=self._context.decision_sample,
        ))
        vad_seconds = time.perf_counter() - begin
        # Windows may reach into the trailing silence, so the ASR source
        # gets the same zero tail the chunker heard.
        asr_source = np.concatenate([asr_pcm, np.zeros(flush_samples, np.float32)])
        self.strategy.start_item(item, asr_source, probabilities)
        lines: list[str] = []
        begin = time.perf_counter()
        for window_index, window in enumerate(windows):
            decode_begin = time.perf_counter()
            fields = self.strategy.decode_window(window)
            decode_seconds = time.perf_counter() - decode_begin
            start = min(window.start_sample, item.sample_count)
            end = min(window.end_sample, item.sample_count)
            record = {
                "item_id": item.item_id,
                "window_index": window_index,
                "start_s": start / SAMPLE_RATE_HZ,
                "end_s": end / SAMPLE_RATE_HZ,
                "text": fields["text"],
                "emit_reason": window.emit_reason,
                "asr_outcome": fields["asr_outcome"],
                "emitted_at_s": window.emitted_at_sample / SAMPLE_RATE_HZ,
                "asr_s": decode_seconds,
            }
            record.update(
                (key, value) for key, value in fields.items() if key not in _FIXED_WINDOW_KEYS
            )
            lines.append(json.dumps(record, ensure_ascii=False))
        asr_seconds = time.perf_counter() - begin
        return _ItemResult(lines, vad_seconds, asr_seconds, pending, frontend_sha256,
                           assist_audio_sha256, assist_tags_sha256)


class _Totals:
    """The run's ``windows.jsonl`` and what run.json sums over items, fed
    the items' results in set order. The file opens with the first result,
    so a run whose models fail to load leaves none behind."""

    def __init__(self, bench_set: BenchSet, path: Path) -> None:
        self._items = bench_set.items
        self._path = path
        self._sink: Any = None
        self.vad_seconds = 0.0
        self.asr_seconds = 0.0
        self.window_count = 0
        self.pending_items: list[str] = []
        self.frontend_files: dict[str, str] = {}
        self.assist_audio_files: dict[str, str] = {}
        self.assist_tag_files: dict[str, str] = {}

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()

    def add(self, position: int, result: _ItemResult) -> None:
        item = self._items[position]
        if self._sink is None:
            self._sink = self._path.open("w", encoding="utf-8", newline="\n")
        for line in result.lines:
            self._sink.write(line + "\n")
        self.vad_seconds += result.vad_seconds
        self.asr_seconds += result.asr_seconds
        self.window_count += len(result.lines)
        if result.open_at_end:
            self.pending_items.append(item.item_id)
        for files, digest in ((self.frontend_files, result.frontend_sha256),
                              (self.assist_audio_files, result.assist_audio_sha256),
                              (self.assist_tag_files, result.assist_tags_sha256)):
            if digest is not None:
                files[item.item_id] = digest
        _log.info(
            "%d/%d %s: %d windows, %.1f s audio",
            position + 1, len(self._items), item.item_id, len(result.lines), item.duration_s,
        )


@dataclasses.dataclass(frozen=True)
class _ProcessReport:
    """What one process reports besides its items' results."""

    load_seconds: float
    asr_identity: dict[str, Any]
    strategy: dict[str, Any] | None
    assist_counts: dict[str, int] | None


# How often the parent of worker processes checks that they are alive.
_POLL_SECONDS = 1.0


def _worker_main(number: int, bench_set: BenchSet, candidate: Candidate,
                 tasks: Any, results: Any) -> None:
    """One process of a ``--workers`` run: builds its models, then processes
    the item positions it takes from ``tasks`` until it takes None. Any
    failure goes to ``results`` with its traceback."""
    logging.basicConfig(level=logging.WARNING,
                        format=f"%(levelname)s worker {number} %(name)s: %(message)s")
    try:
        pipeline = _Pipeline(bench_set, candidate, _setup(candidate))
        results.put(("ready", number, pipeline.load_seconds, pipeline.asr_identity))
        while (position := tasks.get()) is not None:
            results.put(("item", position, pipeline.process(bench_set.items[position])))
        results.put(("done", number, pipeline.strategy.record(), pipeline.assist_counts()))
    except BaseException:  # reported to the parent, which stops the run
        results.put(("error", number, traceback.format_exc()))


def _run_workers(bench_set: BenchSet, candidate: Candidate, workers: int,
                 totals: _Totals) -> tuple[float, list[_ProcessReport]]:
    """Process the items in ``workers`` spawned processes, each taking the
    next item when it finishes one, and add their results to ``totals`` in
    set order. Returns when the last process had its models loaded
    (``time.perf_counter``) and each process's report."""
    spawn = multiprocessing.get_context("spawn")
    tasks = spawn.Queue()
    results = spawn.Queue()
    for position in range(len(bench_set.items)):
        tasks.put(position)
    for _ in range(workers):
        tasks.put(None)
    processes = [
        spawn.Process(target=_worker_main, args=(number, bench_set, candidate, tasks, results),
                      name=f"asr-bench-worker-{number}", daemon=True)
        for number in range(workers)
    ]
    loaded: dict[int, tuple[float, dict[str, Any]]] = {}
    finished: dict[int, tuple[dict[str, Any] | None, dict[str, int] | None]] = {}
    waiting: dict[int, _ItemResult] = {}
    next_position = 0
    loaded_at = time.perf_counter()
    all_exited = False
    try:
        for process in processes:
            process.start()
        while len(finished) < workers:
            try:
                kind, key, *payload = results.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                for process in processes:
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(f"{process.name} exited with code {process.exitcode}") from None
                # A process flushes its messages before it exits: a second
                # empty poll after every process exited means one never sent its end.
                if all_exited:
                    raise RuntimeError("a worker process exited without reporting its end") from None
                all_exited = all(process.exitcode is not None for process in processes)
                continue
            if kind == "error":
                raise RuntimeError(f"asr-bench-worker-{key} failed:\n{payload[0]}")
            if kind == "ready":
                loaded[key] = (payload[0], payload[1])
                loaded_at = time.perf_counter()
            elif kind == "item":
                waiting[key] = payload[0]
                while next_position in waiting:
                    totals.add(next_position, waiting.pop(next_position))
                    next_position += 1
            else:
                finished[key] = (payload[0], payload[1])
        if next_position != len(bench_set.items):
            raise RuntimeError(f"the workers returned {next_position} of {len(bench_set.items)} items")
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=30)
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
    return loaded_at, [
        _ProcessReport(loaded[number][0], loaded[number][1], *finished[number])
        for number in range(workers)
    ]


def _merged_strategy(mode: str | None,
                     sections: Sequence[dict[str, Any] | None]) -> dict[str, Any] | None:
    """The run's strategy section from each process's: one process's as it
    is; else the mode's additive keys summed and every other key taken once
    (a process that decoded no window may leave one None)."""
    if mode is None or len(sections) == 1:
        return sections[0]
    additive = _ADDITIVE_STRATEGY_KEYS[mode]
    keys = list(sections[0] or {})
    if any(set(section or {}) != set(keys) for section in sections):
        raise RuntimeError("the worker processes' strategy sections have different keys")
    merged: dict[str, Any] = {}
    for key in keys:
        values = [section[key] for section in sections if section is not None]
        if key in additive:
            if isinstance(values[0], Mapping):
                counts: Counter = Counter()
                for value in values:
                    counts.update(value)
                merged[key] = dict(counts)
            else:
                merged[key] = sum(values)
            continue
        present = [value for value in values if value is not None]
        if any(value != present[0] for value in present[1:]):
            raise RuntimeError(f"the worker processes' strategy sections disagree on {key!r}")
        merged[key] = present[0] if present else None
    return merged


def run(bench_set: BenchSet, candidate: Candidate, out_dir: Path, workers: int = 1) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    mode = None if candidate.strategy is None else candidate.strategy.mode
    if workers > 1 and mode is not None and mode not in _ADDITIVE_STRATEGY_KEYS:
        raise ValueError(f"strategy {mode} summarizes all of a run's windows in run.json; "
                         "it runs with one worker only")
    wall_begin = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.json"
    windows_path = out_dir / "windows.jsonl"
    if run_path.exists() or windows_path.exists():
        raise FileExistsError(f"{out_dir} already holds a run; choose a new --out")

    setup = _setup(candidate)
    frontend = candidate.frontend
    if frontend is not None:
        _check_frontend(frontend, bench_set)
    assist = candidate.vad_assist
    if assist is not None:
        _check_vad_assist(assist, bench_set, setup.layout)

    load_begin = time.perf_counter()
    totals = _Totals(bench_set, windows_path)
    try:
        if workers == 1:
            pipeline = _Pipeline(bench_set, candidate, setup)
            loaded_at = time.perf_counter()
            artifact = pipeline.vad_artifact
            for position, item in enumerate(bench_set.items):
                totals.add(position, pipeline.process(item))
            reports = [_ProcessReport(pipeline.load_seconds, pipeline.asr_identity,
                                      pipeline.strategy.record(), pipeline.assist_counts())]
        else:
            # Resolved here first, so the processes find a VAD given by URL cached.
            artifact = resolve_vad(candidate.vad)
            loaded_at, reports = _run_workers(bench_set, candidate, workers, totals)
    finally:
        totals.close()
    items_end = time.perf_counter()

    asr_identity = reports[0].asr_identity
    if any(report.asr_identity != asr_identity for report in reports[1:]):
        raise RuntimeError("the worker processes built different ASR decoders")
    assist_counts = None
    if assist is not None:
        assist_counts = {
            key: sum(report.assist_counts[key] for report in reports if report.assist_counts)
            for key in _ASSIST_COUNTS
        }
    audio_seconds = sum(item.duration_s for item in bench_set.items)
    load_wall_seconds = loaded_at - load_begin
    rtf_definition = "(vad_seconds + asr_seconds) / audio_seconds; load and warmup excluded"
    if workers > 1:
        rtf_definition += f"; summed over {workers} processes running at once"
    run_record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate": candidate.to_json(),
        "set": {
            "set_id": bench_set.set_id,
            "kind": bench_set.kind,
            "path": str(bench_set.root.resolve()),
            "item_count": len(bench_set.items),
            "line_count": sum(len(item.lines) for item in bench_set.items),
            "audio_seconds": audio_seconds,
        },
        "vad": {
            **artifact.identity(),
            "path": str(artifact.path),
            "threshold": setup.threshold,
            "chunker_config": dataclasses.asdict(setup.config),
        },
        "asr": asr_identity,
        "strategy": _merged_strategy(mode, [report.strategy for report in reports]),
        "frontend": None if frontend is None else {
            "name": frontend.name,
            "feed": frontend.feed,
            "audio_dir": frontend.audio_dir,
            "files_sha256": totals.frontend_files,
            "manifest_sha256": canonical_sha256(totals.frontend_files),
        },
        "vad_assist": None if assist is None else _assist_record(
            assist, assist_counts, totals.assist_audio_files, totals.assist_tag_files),
        "emission": _emission_record(setup.layout),
        "feed": {
            "block_samples": _BLOCK_SAMPLES,
            "trailing_silence_samples": setup.flush_samples,
            "items_with_open_window_at_end": totals.pending_items,
        },
        "window_count": totals.window_count,
        "git": _git_identity(),
        "gpu": _gpu_identity(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "workers": workers,
        # Decodes of several processes share the GPU and CPU: their asr_s,
        # the line delays built on it and this timing include contention.
        "timing_valid": workers == 1,
        "timing": {
            "load_seconds": max(report.load_seconds for report in reports),
            "vad_seconds": totals.vad_seconds,
            "asr_seconds": totals.asr_seconds,
            "wall_seconds": time.perf_counter() - wall_begin,
            "rtf": (totals.vad_seconds + totals.asr_seconds) / audio_seconds,
            "rtf_definition": rtf_definition,
        },
        "throughput": {
            "load_wall_seconds": load_wall_seconds,
            "wall_seconds": items_end - load_begin,
            "rtf": (items_end - load_begin - load_wall_seconds) / audio_seconds,
            "definition": "wall clock from the start of model loading: load_wall_seconds until "
                          "every process had loaded its models, wall_seconds until the last "
                          "window was written; rtf = (wall_seconds - load_wall_seconds) / "
                          "audio_seconds, so load_wall_seconds + audio_seconds x rtf is the "
                          "run's wall clock (the basis of RunPod cost estimates)",
        },
    }
    temporary = out_dir / ".run.json.tmp"
    temporary.write_text(json.dumps(run_record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(run_path)
    return run_record


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", required=True, type=Path, help="bench set directory")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate JSON file")
    parser.add_argument("--name", help="candidate to select from a candidates list")
    parser.add_argument("--out", required=True, type=Path, help="new run directory")
    parser.add_argument("--workers", type=_positive_int, default=1,
                        help="processes, each with its own models (default 1); with more than "
                             "one the run's timing is not comparable (timing_valid false)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    _log.setLevel(logging.INFO)

    bench_set = load_set(args.set)
    candidate = load_candidate(args.candidate, args.name)
    record = run(bench_set, candidate, args.out, workers=args.workers)
    timing, throughput = record["timing"], record["throughput"]
    print(
        f"{candidate.name} on {bench_set.set_id}: {record['window_count']} windows, "
        f"{record['set']['audio_seconds']:.1f} s audio, rtf {timing['rtf']:.3f} "
        f"(vad {timing['vad_seconds']:.1f} s, asr {timing['asr_seconds']:.1f} s, "
        f"load {timing['load_seconds']:.1f} s) -> {args.out}"
    )
    if args.workers > 1:
        print(f"{args.workers} workers: timing summed over the processes, not comparable; "
              f"wall {throughput['wall_seconds']:.1f} s (load {throughput['load_wall_seconds']:.1f} s), "
              f"throughput rtf {throughput['rtf']:.4f}")
    if record["feed"]["items_with_open_window_at_end"]:
        print(f"warning: gate still open at item end: {record['feed']['items_with_open_window_at_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
