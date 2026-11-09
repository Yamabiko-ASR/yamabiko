"""Synchronous, coordinate-exact replay of Yamabiko's production VAD gate.

The replay drives production's classes directly: every 512-sample frame waits
in the sound-tag assist's delay line (``yamabiko/vad/sound_tagger.py``), Silero
scores it when it leaves, and ``FrameGate`` receives ``max(silero, tag term)``,
as ``SileroChunker`` does.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.e0_evidence import contracts
from scripts.e0_evidence._common import file_sha256_and_size, require_sha256
from scripts.e0_evidence.ledger import EvidenceSink
from scripts.e0_evidence.assets import require_local_evaluation
from scripts.e0_evidence.pcm import (
    CaptureReplayStream,
    capture_replay_provenance_sha256,
)
from scripts.e0_evidence.schema import (
    DiagnosticCode,
    EndpointEmitReason,
    EvidenceRecord,
    EvidenceRecordDraft,
    ObservationStatus,
    PrivateContentClass,
    RecordKind,
    ReplayStopKind,
    SampleSpan,
    Stage,
    new_opaque_id,
)
from scripts.e0_evidence.wire import canonical_sha256
from yamabiko import __version__ as yamabiko_version
from yamabiko.models import SILERO_SHA256
from yamabiko.vad import sound_tagger
from yamabiko.vad.chunker import ChunkerConfig, chunker_config_from_config
from yamabiko.vad.silero_chunker import (
    FrameGate,
    FrameGateState,
    SileroChunker,
    SileroVAD,
    _default_model_path,
)
from yamabiko.vad.sound_tagger import SoundTagAssist, SoundTagger


_FRAME_SAMPLES = 512
_MAX_EXACT_FLOAT32_INTEGER = 1 << 24
_PROBE_FIELDS = frozenset(
    {"_buf", "_pending", "_piece_start_k", "_dip_pad_len"}
)


class ReplayModel(Protocol):
    artifact_sha256: str

    def __call__(self, audio_frame: np.ndarray) -> float: ...

    def reset_states(self) -> None: ...


class ReplayTagger(Protocol):
    artifact_sha256: str

    def __call__(self, window: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    window_id: str
    source_spans: tuple[SampleSpan, ...]
    audio: np.ndarray
    pcm_sha256: str
    emit_reason: EndpointEmitReason
    forced: bool
    dip_cut: bool
    state_before_emit: FrameGateState | None = None
    state_after_emit: FrameGateState | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    windows: tuple[ReplayWindow, ...]
    pending_spans: tuple[SampleSpan, ...]
    partial_frame_span: SampleSpan | None
    stop_kind: ReplayStopKind
    integrity_records: tuple[EvidenceRecord, ...]
    terminal_gate_state: FrameGateState | None = None


def _mapping_block(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key in value:
        if type(key) is not str:
            raise TypeError(f"{field_name} keys must be strings")
    return value  # type: ignore[return-value]


def _probability(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise TypeError(f"{field_name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability")
    return result


def production_chunker_config(
    full_config: Mapping[str, object],
) -> tuple[ChunkerConfig, float]:
    """Build the VAD gate through production's seam, refusing non-16 kHz audio."""

    full = _mapping_block(full_config, "full_config")
    _mapping_block(full["audio"], "full_config.audio")
    _mapping_block(full["vad"], "full_config.vad")
    config, threshold = chunker_config_from_config(full)
    if type(config.sample_rate) is not int or config.sample_rate != 16_000:
        raise ValueError("E0 replay requires audio.sample_rate 16000")
    return config, _probability(threshold, "vad.threshold")


def _config_mapping(config: ChunkerConfig) -> dict[str, object]:
    if not isinstance(config, ChunkerConfig):
        raise TypeError("config must be ChunkerConfig")
    return {
        "sample_rate": config.sample_rate,
        "min_chunk_ms": config.min_chunk_ms,
        "max_chunk_ms": config.max_chunk_ms,
        "min_silence_ms": config.min_silence_ms,
        "speech_pad_ms": config.speech_pad_ms,
        "glue_max_silence_ms": config.glue_max_silence_ms,
        "attach_max_ms": config.attach_max_ms,
        "drop_floor_prob": config.drop_floor_prob,
        "drop_floor_ms": config.drop_floor_ms,
        "dip_split_enabled": config.dip_split_enabled,
        "dip_split_threshold": config.dip_split_threshold,
        "dip_split_min_piece_ms": config.dip_split_min_piece_ms,
    }


def _effective_config_sha256(config: Mapping[str, object], threshold: float) -> str:
    return canonical_sha256(
        {
            "config": dict(config),
            "frame_samples": _FRAME_SAMPLES,
            "threshold": threshold,
        }
    )


@dataclass(frozen=True, slots=True)
class VadArtifact:
    """The exact Silero ONNX bytes one run replays, hashed before it starts."""

    path: Path
    sha256: str
    byte_count: int

    @property
    def production_pin(self) -> bool:
        return self.sha256 == SILERO_SHA256

    def identity(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "production_pin": self.production_pin,
        }


def resolve_vad_artifact(
    path: Path | None = None,
    sha256: str | None = None,
) -> VadArtifact:
    """Hash a declared candidate, or production's pinned file when none is declared."""

    if (path is None) != (sha256 is None):
        raise ValueError("a declared vad_artifact needs both path and sha256")
    if path is None:
        path = _default_model_path()
        expected = SILERO_SHA256
        mismatch = "the pinned production Silero artifact does not match its SHA-256"
    else:
        expected = require_sha256(sha256, "vad_artifact.sha256")
        mismatch = "vad_artifact does not match its declared sha256"
    try:
        actual, byte_count = file_sha256_and_size(path)
    except OSError:
        raise ValueError("the Silero VAD artifact is unreadable") from None
    if actual != expected:
        raise ValueError(mismatch)
    return VadArtifact(path=path, sha256=actual, byte_count=byte_count)


def load_production_model(
    config: ChunkerConfig,
    artifact: VadArtifact | None = None,
) -> SileroVAD:
    """Instantiate the replay VAD through production's ``SileroVAD`` path.

    ``artifact`` defaults to the exact file production loads.
    """

    if not isinstance(config, ChunkerConfig):
        raise TypeError("config must be ChunkerConfig")
    if config.sample_rate != 16_000:
        raise ValueError("production Silero replay requires 16000 Hz")
    if artifact is None:
        artifact = resolve_vad_artifact()
    elif not isinstance(artifact, VadArtifact):
        raise TypeError("artifact must be VadArtifact")
    model = SileroVAD(
        artifact.path,
        sample_rate=config.sample_rate,
        expected_sha256=artifact.sha256,
    )
    model.artifact_sha256 = artifact.sha256
    return model


@dataclass(frozen=True, slots=True)
class TaggerArtifact:
    """The bundled sound-tagger ONNX bytes one run replays, hashed before it starts."""

    path: Path
    sha256: str
    byte_count: int


def resolve_tagger_artifact() -> TaggerArtifact:
    """Hash the bundled sound tagger production loads; it must be the pinned graph."""

    path = sound_tagger.default_model_path()
    try:
        actual, byte_count = file_sha256_and_size(path)
    except OSError:
        raise ValueError("the bundled sound tagger is unreadable") from None
    if actual != contracts.SOUND_TAGGER_MODEL_SHA256:
        raise ValueError("the bundled sound tagger does not match its pinned SHA-256")
    return TaggerArtifact(path=path, sha256=actual, byte_count=byte_count)


def load_production_tagger(artifact: TaggerArtifact | None = None) -> SoundTagger:
    """Instantiate the replay's sound tagger through production's ``SoundTagger``.

    ``artifact`` defaults to the bundled file production loads.
    """

    if artifact is None:
        artifact = resolve_tagger_artifact()
    elif not isinstance(artifact, TaggerArtifact):
        raise TypeError("artifact must be TaggerArtifact")
    tagger = SoundTagger(artifact.path)
    tagger.artifact_sha256 = artifact.sha256
    return tagger


def vad_assist_layout() -> dict[str, object]:
    """The assist's fusion weight, groups and streaming layout, read from production."""

    return {
        "assist_weight": sound_tagger.ASSIST_WEIGHT,
        "frame_samples": sound_tagger.FRAME_SAMPLES,
        "hop_frames": sound_tagger.HOP_FRAMES,
        "lookahead_frames": sound_tagger.LOOKAHEAD_FRAMES,
        "shout_classes": sound_tagger.SHOUT_CLASSES,
        "tag_frame_samples": sound_tagger.TAG_FRAME_SAMPLES,
        "voice_classes": sound_tagger.VOICE_CLASSES,
        "window_frames": sound_tagger.WINDOW_FRAMES,
    }


def _verify_vad_assist_contract(tagger_sha256: str) -> dict[str, object]:
    """Return the assist identity a replay contract records, once it is approved."""

    tagger_source_sha256 = contracts.source_sha256(sound_tagger)
    chunker_source_sha256 = contracts.source_sha256(SileroChunker)
    layout = vad_assist_layout()
    if (
        yamabiko_version != contracts.YAMABIKO_VERSION_PIN
        or tagger_source_sha256 != contracts.SOUND_TAGGER_SOURCE_SHA256
        or chunker_source_sha256 != contracts.SILERO_CHUNKER_SOURCE_SHA256
        or tagger_sha256 != contracts.SOUND_TAGGER_MODEL_SHA256
        or sound_tagger.MODEL_SHA256 != contracts.SOUND_TAGGER_MODEL_SHA256
        or canonical_sha256(layout)
        != canonical_sha256(contracts.SOUND_TAG_ASSIST_LAYOUT)
    ):
        raise RuntimeError("the VAD sound-tag assist is not approved for E0 replay")
    return {
        "model_sha256": tagger_sha256,
        "sound_tagger_source_sha256": tagger_source_sha256,
        "silero_chunker_source_sha256": chunker_source_sha256,
        "layout": layout,
    }


def _verify_silero_vad_contract() -> str:
    source_sha256 = contracts.source_sha256(SileroVAD)
    if (
        yamabiko_version != contracts.YAMABIKO_VERSION_PIN
        or source_sha256 != contracts.SILERO_VAD_SOURCE_SHA256
    ):
        raise RuntimeError("SileroVAD source is not approved for E0 replay")
    return source_sha256


def _verify_probe_contract(*gates: FrameGate) -> str:
    source_sha256 = contracts.source_sha256(FrameGate)
    if (
        yamabiko_version != contracts.YAMABIKO_VERSION_PIN
        or source_sha256 != contracts.FRAME_GATE_SOURCE_SHA256
    ):
        raise RuntimeError(
            "FrameGate private-state probe is not approved for this source version"
        )
    for gate in gates:
        for field_name in _PROBE_FIELDS:
            if not hasattr(gate, field_name):
                raise RuntimeError(
                    f"FrameGate probe field {field_name!r} is unavailable"
                )
    return source_sha256


def _probe(gate: FrameGate, field_name: str) -> object:
    if field_name not in _PROBE_FIELDS:
        raise RuntimeError(f"FrameGate probe field {field_name!r} is not approved")
    return getattr(gate, field_name)


def _coordinate_indices(
    coordinate_audio: np.ndarray,
    *,
    master_sample_count: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if (
        not isinstance(coordinate_audio, np.ndarray)
        or coordinate_audio.ndim != 1
        or coordinate_audio.size == 0
        or coordinate_audio.size % _FRAME_SAMPLES
    ):
        raise ValueError("coordinate gate emitted an invalid sample count")
    rows = coordinate_audio.reshape(-1, _FRAME_SAMPLES)
    ordinals_f32 = rows[:, 0]
    if not np.all(rows == ordinals_f32[:, np.newaxis]):
        raise ValueError("coordinate gate emitted a nonconstant ordinal row")
    if not np.all(np.isfinite(ordinals_f32)):
        raise ValueError("coordinate gate emitted a nonfinite ordinal")
    ordinals = ordinals_f32.astype(np.int64)
    if not np.array_equal(ordinals_f32, ordinals.astype(np.float32)):
        raise ValueError("coordinate gate emitted a nonintegral ordinal")
    if np.any(ordinals < 0) or np.any(ordinals > _MAX_EXACT_FLOAT32_INTEGER):
        raise ValueError("coordinate ordinal exceeds float32's exact integer range")
    if ordinals.size > 1 and np.any(np.diff(ordinals) <= 0):
        raise ValueError("coordinate ordinals must be ordered and unique")
    coordinates = (
        ordinals[:, np.newaxis] * np.int64(_FRAME_SAMPLES)
        + np.arange(_FRAME_SAMPLES, dtype=np.int64)
    ).reshape(-1)
    if coordinates[-1] >= master_sample_count:
        raise ValueError("coordinate gate emitted beyond the master timeline")
    return coordinates, tuple(int(item) for item in ordinals)


def _compact_spans(
    timeline_id: str,
    coordinates: np.ndarray,
) -> tuple[SampleSpan, ...]:
    if coordinates.size == 0:
        return ()
    boundaries = np.flatnonzero(np.diff(coordinates) != 1) + 1
    groups = np.split(coordinates, boundaries)
    return tuple(
        SampleSpan(timeline_id, int(group[0]), int(group[-1]) + 1)
        for group in groups
    )


def _retained_frames(gate: FrameGate) -> Sequence[np.ndarray]:
    if gate.state_snapshot().in_speech:
        frames = _probe(gate, "_buf")
        piece_start = _probe(gate, "_piece_start_k")
        dip_pad_len = _probe(gate, "_dip_pad_len")
        if not isinstance(frames, list):
            raise RuntimeError("FrameGate _buf probe has an invalid type")
        if type(piece_start) is not int or type(dip_pad_len) is not int:
            raise RuntimeError("FrameGate dip coordinate probe has an invalid type")
        start = 0 if piece_start == 0 else dip_pad_len + piece_start
        if not 0 <= start <= len(frames):
            raise RuntimeError("FrameGate retained-piece probe is out of bounds")
        return frames[start:]
    frames = _probe(gate, "_pending")
    if not hasattr(frames, "__iter__"):
        raise RuntimeError("FrameGate _pending probe has an invalid type")
    return tuple(frames)  # type: ignore[arg-type]


def _retained_payload(gate: FrameGate, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    frames = tuple(_retained_frames(gate))
    if not frames:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(frames).astype(dtype, copy=False)


def _state_mapping(state: FrameGateState) -> Mapping[str, object]:
    return {
        "in_speech": state.in_speech,
        "retained_frames": state.retained_frames,
        "speech_frames": state.speech_frames,
        "consecutive_silence_frames": state.consecutive_silence_frames,
        "silence_since_speech_frames": state.silence_since_speech_frames,
        "frames_since_open": state.frames_since_open,
    }


def _validate_payload_pair(
    audio: np.ndarray,
    coordinate_audio: np.ndarray,
    capture: CaptureReplayStream,
) -> tuple[np.ndarray, tuple[SampleSpan, ...], tuple[int, ...]]:
    if audio.ndim != 1 or audio.size != coordinate_audio.size:
        raise ValueError("audio and coordinate gates emitted different sample counts")
    manifest = capture.master.manifest
    coordinates, ordinals = _coordinate_indices(
        coordinate_audio,
        master_sample_count=manifest.sample_count,
    )
    expected = capture.samples[coordinates]
    if not np.array_equal(audio, expected):
        raise ValueError("audio gate output does not match replay coordinates")
    spans = _compact_spans(manifest.timeline_id, coordinates)
    return coordinates, spans, ordinals


def _emit_record(
    sink: EvidenceSink,
    records: list[EvidenceRecord],
    draft: EvidenceRecordDraft,
) -> EvidenceRecord:
    record = sink.emit(draft)
    if not isinstance(record, EvidenceRecord):
        raise TypeError("evidence sink must return EvidenceRecord")
    records.append(record)
    return record


def _emit_diagnostic(
    sink: EvidenceSink,
    records: list[EvidenceRecord],
    replay_contract_id: str,
    code: DiagnosticCode,
    source_spans: tuple[SampleSpan, ...],
) -> EvidenceRecord:
    return _emit_record(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=None,
            stage=Stage.ENDPOINT,
            kind=RecordKind.DIAGNOSTIC,
            artifact_id=new_opaque_id(),
            parent_ids=(replay_contract_id,),
            source_spans=source_spans,
            decision=code.value,
            diagnostics=(code,),
            classification=PrivateContentClass.PRIVATE_AUDIO,
        ),
    )


def _validate_capture(capture: CaptureReplayStream) -> tuple[str, str, str]:
    if not isinstance(capture, CaptureReplayStream):
        raise TypeError("capture must be CaptureReplayStream")
    master = capture.master
    manifest = master.manifest
    require_local_evaluation(manifest.rights)
    if (
        manifest.sample_rate_hz != 16_000
        or manifest.channels != 1
        or manifest.layout != "mono"
        or manifest.dtype != "<f4"
        or master.samples.dtype != np.dtype("<f4")
        or master.samples.ndim != 1
        or master.samples.size != manifest.sample_count
        or master.samples.flags.writeable
        or capture.samples.dtype != np.dtype("<f4")
        or capture.samples.ndim != 1
        or capture.samples.size != manifest.sample_count
        or capture.samples.flags.writeable
    ):
        raise ValueError("replay requires immutable mono 16000 Hz little-endian float32")
    master_sha256 = hashlib.sha256(master.samples.tobytes(order="C")).hexdigest()
    if master_sha256 not in {
        manifest.byte_sha256,
        manifest.canonical_pcm_sha256,
    } or manifest.byte_sha256 != manifest.canonical_pcm_sha256:
        raise ValueError("canonical master PCM hash does not match its manifest")
    provenance_sha256 = capture_replay_provenance_sha256(capture.provenance)
    replay_sha256 = hashlib.sha256(capture.samples.tobytes(order="C")).hexdigest()
    if (
        provenance_sha256 != capture.provenance_sha256
        or replay_sha256 != capture.provenance["output_pcm_sha256"]
        or capture.provenance["canonical_pcm_sha256"] != master_sha256
    ):
        raise ValueError("capture replay PCM does not match its provenance")
    return master_sha256, replay_sha256, provenance_sha256


def _retained_coordinate_spans(
    gate: FrameGate,
    *,
    timeline_id: str,
    master_sample_count: int,
    held: range = range(0),
) -> tuple[SampleSpan, ...]:
    """Spans of the gate's retained frames and of the ``held`` frame ordinals
    still waiting in the assist's delay line."""

    retained = _retained_payload(gate, dtype=np.dtype(np.float32))
    indices = np.zeros(0, dtype=np.int64)
    if retained.size:
        indices, _ = _coordinate_indices(
            retained,
            master_sample_count=master_sample_count,
        )
    held_indices = np.arange(
        held.start * _FRAME_SAMPLES,
        held.stop * _FRAME_SAMPLES,
        dtype=np.int64,
    )
    return _compact_spans(timeline_id, np.union1d(indices, held_indices))


class _AssistFeed:
    """A capture stream's complete frames as production's sound-tag assist
    releases them: ``(frame, tag term, ordinal)``, in order.

    Each frame enters ``SoundTagAssist`` with its ordinal, as
    ``SileroChunker.run`` admits it, and comes back once the window that tags
    it has its look-ahead. With ``drain``, the frames still held after the
    last one are released as production's capture-stall drain releases them:
    tagged with zeros for the look-ahead that never arrives, nothing reset.
    ``admitted`` counts the frames handed to the assist so far.
    """

    def __init__(
        self,
        tagger: ReplayTagger,
        samples: np.ndarray,
        frame_count: int,
        *,
        drain: bool,
    ) -> None:
        self._assist = SoundTagAssist(tagger)
        self._samples = samples
        self._frame_count = frame_count
        self._drain = drain
        self.admitted = 0

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, object]]:
        # The assist awaits its tagger in a worker thread. Each call runs to
        # completion before the next frame is admitted, so at most one
        # inference is in flight, as in production.
        with asyncio.Runner() as runner:
            for ordinal in range(self._frame_count):
                if ordinal > _MAX_EXACT_FLOAT32_INTEGER:
                    raise ValueError(
                        "frame ordinal exceeds float32's exact integer range"
                    )
                start = ordinal * _FRAME_SAMPLES
                self.admitted = ordinal + 1
                yield from runner.run(
                    self._assist.push(
                        self._samples[start:start + _FRAME_SAMPLES],
                        ordinal,
                    )
                )
            if self._drain:
                yield from runner.run(self._assist.drain())


def _emit_replay_terminal(
    sink: EvidenceSink,
    records: list[EvidenceRecord],
    contract_record: EvidenceRecord,
    stop_kind: ReplayStopKind,
    *,
    timeline_id: str,
    master_sample_count: int,
    pending_spans: tuple[SampleSpan, ...],
    window_records: Sequence[EvidenceRecord],
    gate_records: Sequence[EvidenceRecord],
) -> SampleSpan | None:
    """Emit the replay's end diagnostics and terminal record.

    Returns the trailing partial frame the gate never scored, if any.
    """

    complete_samples = master_sample_count - master_sample_count % _FRAME_SAMPLES
    partial_frame_span = (
        SampleSpan(timeline_id, complete_samples, master_sample_count)
        if complete_samples < master_sample_count
        else None
    )
    partial_spans = (
        (partial_frame_span,) if partial_frame_span is not None else ()
    )
    contract_id = contract_record.record_id
    diagnostic_records: list[EvidenceRecord] = []
    if stop_kind is ReplayStopKind.CANCELLED:
        affected = pending_spans + partial_spans
        if affected:
            diagnostic_records.append(
                _emit_diagnostic(
                    sink,
                    records,
                    contract_id,
                    DiagnosticCode.ENDPOINT_PENDING_SHUTDOWN,
                    affected,
                )
            )
        diagnostic_records.append(
            _emit_diagnostic(
                sink,
                records,
                contract_id,
                DiagnosticCode.SHUTDOWN_CANCELLED,
                affected,
            )
        )
    else:
        if pending_spans:
            diagnostic_records.append(
                _emit_diagnostic(
                    sink,
                    records,
                    contract_id,
                    DiagnosticCode.ENDPOINT_PENDING_EOF,
                    pending_spans,
                )
            )
        if partial_spans:
            diagnostic_records.append(
                _emit_diagnostic(
                    sink,
                    records,
                    contract_id,
                    DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF,
                    partial_spans,
                )
            )
        if not window_records and not pending_spans and not partial_spans:
            diagnostic_records.append(
                _emit_diagnostic(
                    sink,
                    records,
                    contract_id,
                    DiagnosticCode.VAD_NOT_DETECTED,
                    (),
                )
            )

    _emit_record(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=None,
            stage=Stage.INTEGRITY,
            kind=RecordKind.INTEGRITY,
            artifact_id=new_opaque_id(),
            parent_ids=(
                contract_id,
                *(record.record_id for record in window_records),
                *(record.record_id for record in gate_records),
                *(record.record_id for record in diagnostic_records),
            ),
            source_spans=pending_spans + partial_spans,
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={
                "replay_contract_id": contract_id,
                "stop_kind": stop_kind.value,
                "window_count": len(window_records),
                "gate_transition_count": len(gate_records),
                "pending_span_count": len(pending_spans),
                "has_partial_frame": partial_frame_span is not None,
                "diagnostic_codes": tuple(
                    record.diagnostics[0].value for record in diagnostic_records
                ),
            },
        ),
    )
    return partial_frame_span


def replay_master(
    capture: CaptureReplayStream,
    config: ChunkerConfig,
    threshold: float,
    model: ReplayModel,
    tagger: ReplayTagger,
    sink: EvidenceSink,
    stop_kind: ReplayStopKind = ReplayStopKind.ORDINARY_EOF,
) -> ReplayResult:
    """Replay a master's production capture stream through complete VAD frames.

    Frames and windows carry production-capture samples on the canonical
    master's timeline. The gate is never flushed: at ordinary EOF only the
    assist's delay line drains, as production's does when capture stalls.
    """

    master_sha256, replay_sha256, capture_replay_sha256 = _validate_capture(capture)
    master = capture.master
    if not isinstance(config, ChunkerConfig):
        raise TypeError("config must be ChunkerConfig")
    if config.sample_rate != 16_000:
        raise ValueError("production replay requires a 16000 Hz ChunkerConfig")
    threshold = _probability(threshold, "threshold")
    if not isinstance(stop_kind, ReplayStopKind):
        raise TypeError("stop_kind must be ReplayStopKind")
    model_sha256 = require_sha256(
        getattr(model, "artifact_sha256", None),
        "model.artifact_sha256",
    )
    if not callable(model) or not callable(getattr(model, "reset_states", None)):
        raise TypeError("model must be callable and expose reset_states()")
    tagger_sha256 = require_sha256(
        getattr(tagger, "artifact_sha256", None),
        "tagger.artifact_sha256",
    )
    if not callable(tagger):
        raise TypeError("tagger must be callable")

    audio_gate = FrameGate(config, threshold=threshold, frame_samples=_FRAME_SAMPLES)
    coordinate_gate = FrameGate(
        config,
        threshold=threshold,
        frame_samples=_FRAME_SAMPLES,
    )
    silero_vad_source_sha256 = _verify_silero_vad_contract()
    probe_source_sha256 = _verify_probe_contract(audio_gate, coordinate_gate)
    vad_assist = _verify_vad_assist_contract(tagger_sha256)
    config_mapping = _config_mapping(config)
    effective_config_sha256 = _effective_config_sha256(config_mapping, threshold)
    records: list[EvidenceRecord] = []
    windows: list[ReplayWindow] = []
    window_records: list[EvidenceRecord] = []
    gate_records: list[EvidenceRecord] = []
    frame_record_ids: dict[int, str] = {}
    timeline_id = master.manifest.timeline_id
    master_sample_count = master.manifest.sample_count
    complete_frame_count = master_sample_count // _FRAME_SAMPLES
    partial_frame_samples = master_sample_count % _FRAME_SAMPLES

    contract_record = _emit_record(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=None,
            stage=Stage.INTEGRITY,
            kind=RecordKind.INTEGRITY,
            artifact_id=master.manifest.master_id,
            observation_status=ObservationStatus.OBSERVED,
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={
                "master_pcm_sha256": master_sha256,
                "replay_pcm_sha256": replay_sha256,
                "capture_replay_sha256": capture_replay_sha256,
                "model_sha256": model_sha256,
                "probe_source_sha256": probe_source_sha256,
                "silero_vad_source_sha256": silero_vad_source_sha256,
                "probe_contract_version": contracts.PROBE_CONTRACT_VERSION,
                "effective_config_sha256": effective_config_sha256,
                "frame_samples": _FRAME_SAMPLES,
                "threshold": threshold,
                "negative_threshold": max(threshold - 0.15, 0.01),
                "timeline_id": timeline_id,
                "master_sample_count": master_sample_count,
                "complete_frame_count": complete_frame_count,
                "partial_frame_samples": partial_frame_samples,
                "stop_kind": stop_kind.value,
                "config": config_mapping,
                "vad_assist": vad_assist,
            },
        ),
    )

    feed = _AssistFeed(
        tagger,
        capture.samples,
        complete_frame_count,
        drain=stop_kind is ReplayStopKind.ORDINARY_EOF,
    )
    tagged_frames = iter(feed)
    # Frames the coordinate gate has seen; the ones admitted after them still
    # wait in the assist's delay line.
    gated_frames = 0
    try:
        for audio_frame, tag_term, ordinal in tagged_frames:
            if type(ordinal) is not int or ordinal != gated_frames:
                raise ValueError("the sound-tag assist released frames out of order")
            start = ordinal * _FRAME_SAMPLES
            end = start + _FRAME_SAMPLES
            if not np.array_equal(audio_frame, capture.samples[start:end]):
                raise ValueError("the sound-tag assist released another frame's audio")
            # SileroChunker._gate_frames: Silero scores the frame as it leaves
            # the delay line, and the gate hears the larger of the two.
            silero_probability = _probability(model(audio_frame), "model probability")
            sound_tag_term = _probability(tag_term, "sound-tag term")
            probability = max(silero_probability, sound_tag_term)

            before_audio = audio_gate.state_snapshot()
            before_coordinate = coordinate_gate.state_snapshot()
            if before_audio != before_coordinate:
                raise ValueError(
                    "audio and coordinate gates entered a frame out of lockstep"
                )
            coordinate_frame = np.full(
                _FRAME_SAMPLES,
                ordinal,
                dtype=np.float32,
            )
            audio_emit = audio_gate.push(audio_frame, probability)
            coordinate_emit = coordinate_gate.push(coordinate_frame, probability)
            gated_frames += 1
            after_audio = audio_gate.state_snapshot()
            after_coordinate = coordinate_gate.state_snapshot()
            if after_audio != after_coordinate:
                raise ValueError(
                    "audio and coordinate gates left a frame out of lockstep"
                )
            if (audio_emit is None) != (coordinate_emit is None):
                raise ValueError("audio and coordinate gates disagreed on emission")
            # The frame record carries the gate state the frame left behind, and
            # precedes any window or transition record the frame caused.
            frame_record = _emit_record(
                sink,
                records,
                EvidenceRecordDraft(
                    arm_id=None,
                    stage=Stage.VAD,
                    kind=RecordKind.FRAME,
                    artifact_id=new_opaque_id(),
                    parent_ids=(contract_record.record_id,),
                    source_spans=(SampleSpan(timeline_id, start, end),),
                    classification=PrivateContentClass.PRIVATE_AUDIO,
                    details={
                        "frame_ordinal": ordinal,
                        "vad_probability": probability,
                        "silero_probability": silero_probability,
                        "sound_tag_term": sound_tag_term,
                        "gate_in_speech_after": after_audio.in_speech,
                    },
                ),
            )
            frame_record_ids[ordinal] = frame_record.record_id

            emit_reason: EndpointEmitReason | None = None
            window_record: EvidenceRecord | None = None
            if audio_emit is not None and coordinate_emit is not None:
                audio_emission = audio_gate.last_emission
                coordinate_emission = coordinate_gate.last_emission
                if (
                    audio_emission is None
                    or audio_emission != coordinate_emission
                    or audio_emission.after != after_audio
                ):
                    raise ValueError(
                        "audio and coordinate gate emission metadata disagree"
                    )
                forced = audio_gate.last_emit_forced
                dip_cut = audio_gate.last_emit_was_dip_cut
                if (
                    type(forced) is not bool
                    or type(dip_cut) is not bool
                    or forced != coordinate_gate.last_emit_forced
                    or dip_cut != coordinate_gate.last_emit_was_dip_cut
                    or (forced and dip_cut)
                ):
                    raise ValueError(
                        "audio and coordinate gate emission flags disagree"
                    )
                coordinates, spans, emitted_ordinals = _validate_payload_pair(
                    audio_emit,
                    coordinate_emit,
                    capture,
                )
                emit_reason = EndpointEmitReason(audio_emission.reason.value)
                immutable_audio = np.ascontiguousarray(audio_emit, dtype="<f4").copy()
                immutable_audio.setflags(write=False)
                pcm_sha256 = hashlib.sha256(
                    immutable_audio.tobytes(order="C")
                ).hexdigest()
                window_id = new_opaque_id()
                window = ReplayWindow(
                    window_id=window_id,
                    source_spans=spans,
                    audio=immutable_audio,
                    pcm_sha256=pcm_sha256,
                    emit_reason=emit_reason,
                    forced=forced,
                    dip_cut=dip_cut,
                    state_before_emit=audio_emission.before,
                    state_after_emit=audio_emission.after,
                )
                window_record = _emit_record(
                    sink,
                    records,
                    EvidenceRecordDraft(
                        arm_id=None,
                        stage=Stage.ENDPOINT,
                        kind=RecordKind.WINDOW,
                        artifact_id=window_id,
                        parent_ids=(
                            contract_record.record_id,
                            *(frame_record_ids[item] for item in emitted_ordinals),
                        ),
                        source_spans=spans,
                        decision=emit_reason.value,
                        classification=PrivateContentClass.PRIVATE_AUDIO,
                        details={
                            "window_seq": len(windows),
                            "pcm_sha256": pcm_sha256,
                            "forced": forced,
                            "dip_cut": dip_cut,
                            "state_before_emit": _state_mapping(
                                audio_emission.before
                            ),
                            "state_after_emit": _state_mapping(
                                audio_emission.after
                            ),
                        },
                    ),
                )
                window_records.append(window_record)
                windows.append(window)
                if not dip_cut:
                    model.reset_states()
                del coordinates

            if (
                before_audio.in_speech != after_audio.in_speech
                or emit_reason is EndpointEmitReason.DIP_CUT
            ):
                retained_spans = _retained_coordinate_spans(
                    coordinate_gate,
                    timeline_id=timeline_id,
                    master_sample_count=master_sample_count,
                )
                parent_ids = [contract_record.record_id, frame_record.record_id]
                if window_record is not None:
                    parent_ids.append(window_record.record_id)
                gate_records.append(
                    _emit_record(
                        sink,
                        records,
                        EvidenceRecordDraft(
                            arm_id=None,
                            stage=Stage.ENDPOINT,
                            kind=RecordKind.GATE_TRANSITION,
                            artifact_id=new_opaque_id(),
                            parent_ids=tuple(parent_ids),
                            source_spans=retained_spans,
                            decision=(
                                emit_reason.value if emit_reason is not None else None
                            ),
                            classification=PrivateContentClass.PRIVATE_AUDIO,
                            details={
                                "in_speech_before": before_audio.in_speech,
                                "in_speech_after": after_audio.in_speech,
                                "state_before": _state_mapping(before_audio),
                                "state_after": _state_mapping(after_audio),
                            },
                        ),
                    )
                )
        if (
            stop_kind is ReplayStopKind.ORDINARY_EOF
            and gated_frames != complete_frame_count
        ):
            raise RuntimeError("the sound-tag assist kept frames after its drain")
    except KeyboardInterrupt:
        # Cancellation is never EOF: record the audio the gate still held and
        # the frames still in the assist's delay line as pending shutdown,
        # close the replay as cancelled, and keep unwinding. The spans come
        # from the coordinate gate alone, because an interrupt between the two
        # pushes leaves the audio gate one frame ahead.
        _emit_replay_terminal(
            sink,
            records,
            contract_record,
            ReplayStopKind.CANCELLED,
            timeline_id=timeline_id,
            master_sample_count=master_sample_count,
            pending_spans=_retained_coordinate_spans(
                coordinate_gate,
                timeline_id=timeline_id,
                master_sample_count=master_sample_count,
                held=range(gated_frames, feed.admitted),
            ),
            window_records=window_records,
            gate_records=gate_records,
        )
        raise
    finally:
        tagged_frames.close()

    pending_audio = _retained_payload(audio_gate, dtype=np.dtype("<f4"))
    pending_coordinates = _retained_payload(
        coordinate_gate,
        dtype=np.dtype(np.float32),
    )
    pending_spans: tuple[SampleSpan, ...] = ()
    if pending_audio.size or pending_coordinates.size:
        _, pending_spans, _ = _validate_payload_pair(
            pending_audio,
            pending_coordinates,
            capture,
        )
    held = range(gated_frames, complete_frame_count)
    if held:
        # A requested shutdown abandons the frames still in the delay line,
        # unscored, as production's cancelled chunker does.
        pending_spans = _retained_coordinate_spans(
            coordinate_gate,
            timeline_id=timeline_id,
            master_sample_count=master_sample_count,
            held=held,
        )
    partial_frame_span = _emit_replay_terminal(
        sink,
        records,
        contract_record,
        stop_kind,
        timeline_id=timeline_id,
        master_sample_count=master_sample_count,
        pending_spans=pending_spans,
        window_records=window_records,
        gate_records=gate_records,
    )

    return ReplayResult(
        windows=tuple(windows),
        pending_spans=pending_spans,
        partial_frame_span=partial_frame_span,
        stop_kind=stop_kind,
        integrity_records=tuple(records),
        terminal_gate_state=audio_gate.state_snapshot(),
    )
