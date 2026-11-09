"""onnx-asr NeMo TDT candidate decoder for the E0 P0 and P1 arms.

The candidate is a separate system under test, selected per run request and
frozen per cohort (runbook section 5, "Benchmarking a non-Whisper decoder").
It decodes the windows P0 replays, or the oracle slices P1 cuts, padded exactly
as production pads a Whisper call, and records each call with the same record
kinds, parent links, source projection, and wrapper decisions as the
faster-whisper path in ``decoder``.

The decoder reports no ``avg_logprob`` or ``no_speech_prob``, so only the
text-derived production gates apply (``ledger.text_only_wrapper_gate``).
``onnx_asr`` is imported only when a model loads, so the harness and its tests
import cleanly without it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scripts.e0_evidence._common import require_sha256
from scripts.e0_evidence.decoder import (
    DecoderInstrumentationError,
    OracleComponent,
    _ArmEvidenceSink,
    _arm_sink,
    _emit,
    _immutable_audio,
    _mapping_for_source,
    _source_map_sha256,
    _source_map_wire,
    _union_spans,
    _validate_window,
    oracle_slice,
    project_decoder_span,
)
from scripts.e0_evidence.ledger import (
    EvidenceSink,
    raw_hypothesis_diagnostics,
    text_only_wrapper_gate,
)
from scripts.e0_evidence.pcm import CaptureReplayStream
from scripts.e0_evidence.replay import ReplayWindow
from scripts.e0_evidence.schema import (
    ArmKind,
    DecoderAdapter,
    DiagnosticCode,
    EvidenceRecord,
    EvidenceRecordDraft,
    JsonValue,
    ObservationStatus,
    PrivateContentClass,
    RecordKind,
    SampleSpan,
    Stage,
    TerminalOutcome,
    TextRole,
    new_opaque_id,
)
from yamabiko.asr.asr_client import ASROutcome, ASRResult


NEMO_TDT_ADAPTER = DecoderAdapter.ONNX_ASR_NEMO_TDT.value
NEMO_TDT_MODEL_TYPE = "nemo-conformer-tdt"
ONNX_ASR_VERSION = "0.12.0"
# The bench venv carries the release onnxruntime, which has no GPU provider.
EXECUTION_PROVIDER = "CPUExecutionProvider"
# The full-precision export. ``load_model(..., quantization=None)`` globs these
# exact names, so a directory holding only ``*.int8.onnx`` files cannot load.
FP32_MODEL_FILES = (
    "config.json",
    "decoder_joint-model.onnx",
    "decoder_joint-model.onnx_data",
    "encoder-model.onnx",
    "encoder-model.onnx_data",
    "vocab.txt",
)
_SAMPLE_RATE = 16_000
_CONTRACT_VERSION = 1
_MAPPING_VERSION = 1
# onnx-asr 0.12.0 stamps each token with the start of the encoder frame that
# emitted it: ``_AsrWithDecoding.window_step`` (10 ms) times the model config's
# ``subsampling_factor``.
_FEATURE_STEP_SECONDS = 0.01


def require_fp32_artifact(path: Path) -> Path:
    """Refuse a model directory that lacks any file of the fp32 export."""

    if not isinstance(path, Path) or not path.is_dir():
        raise ValueError("model_artifact_path must be a NeMo TDT model directory")
    missing = tuple(name for name in FP32_MODEL_FILES if not (path / name).is_file())
    if missing:
        raise ValueError(
            "model_artifact_path lacks the fp32 nemo-conformer-tdt files: "
            + ", ".join(missing)
        )
    _subsampling_factor(path)
    return path


def _subsampling_factor(path: Path) -> int:
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("NeMo TDT config.json is unreadable") from None
    if not isinstance(config, dict) or config.get("model_type") != NEMO_TDT_MODEL_TYPE:
        raise ValueError("model_artifact_path config.json is not nemo-conformer-tdt")
    factor = config.get("subsampling_factor", 8)
    if type(factor) is not int or factor < 1:
        raise ValueError("NeMo TDT subsampling_factor must be a positive integer")
    return factor


def installed_runtime_versions() -> tuple[str, str]:
    """Return the installed (onnx-asr, onnxruntime) versions, pinning onnx-asr."""

    try:
        onnx_asr_version = version("onnx-asr")
    except PackageNotFoundError:
        raise ValueError(
            "onnx-asr is not installed; run this adapter from the bench venv"
        ) from None
    if onnx_asr_version != ONNX_ASR_VERSION:
        raise ValueError(f"the NeMo TDT adapter requires onnx-asr {ONNX_ASR_VERSION}")
    try:
        onnxruntime_version = version("onnxruntime")
    except PackageNotFoundError:
        raise ValueError("onnxruntime is not installed") from None
    return onnx_asr_version, onnxruntime_version


def decoder_identity(
    model_artifact_manifest: Mapping[str, object],
    *,
    onnx_asr_version: str,
    onnxruntime_version: str,
) -> dict[str, object]:
    """The run-start, cohort, and model-runtime identity of this adapter."""

    return {
        "name": NEMO_TDT_ADAPTER,
        "model_type": NEMO_TDT_MODEL_TYPE,
        "quantization": None,
        "onnx_asr_version": onnx_asr_version,
        "onnxruntime_version": onnxruntime_version,
        "execution_provider": EXECUTION_PROVIDER,
        "model_artifact_manifest": dict(model_artifact_manifest),
    }


@dataclass(frozen=True, slots=True)
class NemoTdtDecoder:
    """A loaded fp32 model plus the production boundary pad it decodes with."""

    model: object = field(repr=False)
    model_artifact_sha256: str
    boundary_pad_samples: int
    timestamp_frame_seconds: float


def load_nemo_tdt(
    model_dir: Path,
    model_artifact_sha256: str,
    *,
    boundary_pad_ms: int,
) -> NemoTdtDecoder:
    """Load the fp32 export on onnxruntime's CPU provider, offline.

    ``boundary_pad_ms`` is production's ``asr.boundary_pad_ms``; the pad is
    computed exactly as ``FasterWhisperASR.transcribe`` computes it.
    """

    path = require_fp32_artifact(model_dir)
    factor = _subsampling_factor(path)
    digest = require_sha256(model_artifact_sha256, "model_artifact_sha256")
    if type(boundary_pad_ms) is not int or boundary_pad_ms < 0:
        raise ValueError("boundary_pad_ms must be a nonnegative integer")
    onnx_asr = importlib.import_module("onnx_asr")
    if getattr(onnx_asr, "__version__", None) != ONNX_ASR_VERSION:
        raise DecoderInstrumentationError(
            f"the NeMo TDT adapter requires onnx-asr {ONNX_ASR_VERSION}"
        )
    # An existing local directory makes onnx-asr resolve offline, and
    # quantization=None selects the unquantized file names above.
    model = onnx_asr.load_model(
        NEMO_TDT_MODEL_TYPE,
        path,
        quantization=None,
        providers=[EXECUTION_PROVIDER],
    ).with_timestamps()
    return NemoTdtDecoder(
        model=model,
        model_artifact_sha256=digest,
        boundary_pad_samples=int(round(_SAMPLE_RATE * boundary_pad_ms / 1000)),
        timestamp_frame_seconds=_FEATURE_STEP_SECONDS * factor,
    )


@dataclass(frozen=True, slots=True)
class _Recognition:
    text: str
    token_count: int
    # Start times of the tokens that carry text, in decoder-input seconds.
    text_token_seconds: tuple[float, ...]


def _recognition(result: object) -> _Recognition:
    text = getattr(result, "text", None)
    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    if type(text) is not str:
        raise DecoderInstrumentationError("onnx-asr returned non-string text")
    if not isinstance(tokens, Sequence) or not isinstance(timestamps, Sequence):
        raise DecoderInstrumentationError("onnx-asr returned no token timestamps")
    tokens = tuple(tokens)
    timestamps = tuple(timestamps)
    if len(tokens) != len(timestamps) or any(type(item) is not str for item in tokens):
        raise DecoderInstrumentationError("onnx-asr tokens and timestamps disagree")
    previous = 0.0
    for value in timestamps:
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < previous
        ):
            raise DecoderInstrumentationError(
                "onnx-asr token timestamps must be finite, nonnegative, and ordered"
            )
        previous = float(value)
    # The parakeet-ja export emits a bare word-boundary token at the first
    # encoder frame of every window, whatever the audio there, and onnx-asr
    # drops it from the text. Only tokens that contribute text locate the
    # hypothesis.
    text_token_seconds = tuple(
        float(value) for token, value in zip(tokens, timestamps) if token.strip()
    )
    if bool(text_token_seconds) != bool(text.strip()):
        raise DecoderInstrumentationError("onnx-asr text and tokens disagree")
    return _Recognition(
        text=text,
        token_count=len(tokens),
        text_token_seconds=text_token_seconds,
    )


@dataclass(frozen=True, slots=True)
class NemoTdtCallEvidence:
    call_mode: ArmKind
    call_record: EvidenceRecord
    decoder_audio: np.ndarray
    raw_texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NemoTdtArmResult:
    """One window's candidate records, shaped like ``DecoderArmResult``."""

    arm: ArmKind
    call: NemoTdtCallEvidence
    asr_result: ASRResult
    raw_records: tuple[EvidenceRecord, ...]
    wrapper_record: EvidenceRecord
    evidence_records: tuple[EvidenceRecord, ...]


def _decode(
    *,
    decoder: NemoTdtDecoder,
    sink: _ArmEvidenceSink,
    call_mode: ArmKind,
    source_audio: np.ndarray,
    source_spans: tuple[SampleSpan, ...],
    source_artifact_id: str,
    source_pcm_sha256: str,
    oracle: Mapping[str, JsonValue],
) -> NemoTdtArmResult:
    if not isinstance(decoder, NemoTdtDecoder):
        raise TypeError("decoder must be NemoTdtDecoder")
    pad = decoder.boundary_pad_samples
    silence = np.zeros(pad, dtype=source_audio.dtype)
    decoder_audio = _immutable_audio(
        np.concatenate([silence, source_audio, silence]) if pad else source_audio
    )
    decoder_hash = hashlib.sha256(decoder_audio.tobytes(order="C")).hexdigest()
    mapping = _mapping_for_source(source_spans, source_audio.size, decoder_audio.size)
    mapping_sha256 = _source_map_sha256(mapping, decoder_audio.size)
    recognized = _recognition(
        decoder.model.recognize(decoder_audio, sample_rate=_SAMPLE_RATE)
    )
    # One raw hypothesis per window: the window's whole decoded text. A
    # window whose tokens decode to no text has nothing to project or score.
    yielded = bool(recognized.text_token_seconds)

    records: list[EvidenceRecord] = []
    call_record = _emit(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=sink.arm_id,
            stage=Stage.DECODER,
            kind=RecordKind.DECODER_CALL,
            artifact_id=new_opaque_id(),
            source_spans=tuple(
                interval.source_span
                for interval in mapping
                if interval.source_span is not None
            ),
            observation_status=ObservationStatus.OBSERVED,
            decision=None if yielded else DiagnosticCode.DECODER_NO_YIELD.value,
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={
                "call_mode": call_mode.value,
                "decoder_adapter": NEMO_TDT_ADAPTER,
                "decoder_contract_version": _CONTRACT_VERSION,
                "decoder_input_pcm_sha256": decoder_hash,
                "decoder_input_samples": int(decoder_audio.size),
                "mapping_version": _MAPPING_VERSION,
                "model_artifact_sha256": decoder.model_artifact_sha256,
                "nemo_tdt": {
                    "model_type": NEMO_TDT_MODEL_TYPE,
                    "timestamp_frame_seconds": decoder.timestamp_frame_seconds,
                    "token_count": recognized.token_count,
                },
                **oracle,
                "parent_call_record_id": None,
                "raw_segment_count": int(yielded),
                "sample_rate": _SAMPLE_RATE,
                "source_artifact_id": source_artifact_id,
                "source_pcm_sha256": source_pcm_sha256,
                "source_to_decoder_map": _source_map_wire(mapping),
                "source_to_decoder_map_id": new_opaque_id(),
                "source_to_decoder_map_sha256": mapping_sha256,
                "synthetic_left_pad_samples": pad,
                "synthetic_right_pad_samples": pad,
            },
        ),
    )

    raw_records: list[EvidenceRecord] = []
    raw_texts: tuple[str, ...] = ()
    if yielded:
        first_token = recognized.text_token_seconds[0]
        last_token = recognized.text_token_seconds[-1]
        # The span covers the encoder frames of the first and last text token.
        projected = project_decoder_span(
            first_token,
            last_token + decoder.timestamp_frame_seconds,
            _SAMPLE_RATE,
            decoder_audio.size,
            mapping,
        )
        gate = text_only_wrapper_gate(recognized.text)
        text_id = new_opaque_id()
        raw_record = _emit(
            sink,
            records,
            EvidenceRecordDraft(
                arm_id=sink.arm_id,
                stage=Stage.DECODER,
                kind=RecordKind.RAW_HYPOTHESIS,
                artifact_id=new_opaque_id(),
                parent_ids=(call_record.record_id,),
                source_spans=projected.source_spans,
                classification=PrivateContentClass.PRIVATE_AUDIO,
                diagnostics=raw_hypothesis_diagnostics(
                    timestamp_clamped=projected.timestamp_clamped,
                    stochastic=False,
                ),
                details={
                    "end_decoder_sample": projected.decoder_end,
                    "nemo_tdt": {
                        "compression_ratio": gate.compression_ratio,
                        "last_text_token_seconds": last_token,
                    },
                    "raw_end_seconds": last_token + decoder.timestamp_frame_seconds,
                    "raw_start_seconds": first_token,
                    "segment_ordinal": 0,
                    "start_decoder_sample": projected.decoder_start,
                    "synthetic_anchor_sample": projected.synthetic_anchor_sample,
                    "synthetic_anchor_timeline_id": (
                        projected.synthetic_anchor_timeline_id
                    ),
                    "synthetic_pad_hypothesis": projected.synthetic_pad_hypothesis,
                    "text_id": text_id,
                    "timestamp_clamped": projected.timestamp_clamped,
                },
            ),
        )
        raw_records.append(raw_record)
        raw_texts = (recognized.text,)
        sink.emit_private_text(
            canonical_record_id=raw_record.record_id,
            arm_id=sink.arm_id,
            text_id=text_id,
            role=TextRole.RAW_HYPOTHESIS,
            text=recognized.text,
        )
        if gate.compression_rejected:
            decision = DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value
            outcome = ASROutcome.COMPRESSION_LOOP_REJECTED
            wrapper_text = ""
        else:
            decision = TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value
            outcome = ASROutcome.TEXT_AVAILABLE
            wrapper_text = gate.assembled_text
        terminal_parents = (raw_record.record_id,)
    else:
        decision = DiagnosticCode.WRAPPER_EMPTY.value
        outcome = ASROutcome.DEPENDENCY_NO_SEGMENTS
        wrapper_text = ""
        terminal_parents = (call_record.record_id,)
    wrapper_text_id = new_opaque_id() if wrapper_text else None
    wrapper_record = _emit(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=sink.arm_id,
            stage=Stage.WRAPPER,
            kind=RecordKind.WRAPPER_DECISION,
            artifact_id=new_opaque_id(),
            parent_ids=terminal_parents,
            source_spans=_union_spans(
                span for record in raw_records for span in record.source_spans
            ),
            decision=decision,
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={} if wrapper_text_id is None else {"text_id": wrapper_text_id},
        ),
    )
    if wrapper_text_id is not None:
        sink.emit_private_text(
            canonical_record_id=wrapper_record.record_id,
            arm_id=sink.arm_id,
            text_id=wrapper_text_id,
            role=TextRole.WRAPPER_HYPOTHESIS,
            text=wrapper_text,
        )
    return NemoTdtArmResult(
        arm=call_mode,
        call=NemoTdtCallEvidence(
            call_mode=call_mode,
            call_record=call_record,
            decoder_audio=decoder_audio,
            raw_texts=raw_texts,
        ),
        asr_result=ASRResult(
            text=wrapper_text,
            duration_s=source_audio.size / _SAMPLE_RATE,
            outcome=outcome,
        ),
        raw_records=tuple(raw_records),
        wrapper_record=wrapper_record,
        evidence_records=tuple(records),
    )


_NO_ORACLE: Mapping[str, JsonValue] = {
    "oracle_component_id": None,
    "oracle_component_map_sha256": None,
    "oracle_event_ids": (),
    "oracle_reference_set_id": None,
    "oracle_reference_set_sha256": None,
    "oracle_slice_pcm_sha256": None,
}


def run_nemo_p0(
    window: ReplayWindow,
    decoder: NemoTdtDecoder,
    sink: EvidenceSink,
) -> NemoTdtArmResult:
    """Decode one production replay window with the candidate."""

    bound_sink = _arm_sink(sink)
    _validate_window(window)
    return _decode(
        decoder=decoder,
        sink=bound_sink,
        call_mode=ArmKind.P0,
        source_audio=window.audio,
        source_spans=window.source_spans,
        source_artifact_id=window.window_id,
        source_pcm_sha256=window.pcm_sha256,
        oracle=_NO_ORACLE,
    )


def run_nemo_p1(
    component: OracleComponent,
    capture: CaptureReplayStream,
    decoder: NemoTdtDecoder,
    sink: EvidenceSink,
) -> NemoTdtArmResult:
    """Decode one predeclared oracle slice with the candidate."""

    bound_sink = _arm_sink(sink)
    cut = oracle_slice(component, capture)
    return _decode(
        decoder=decoder,
        sink=bound_sink,
        call_mode=ArmKind.P1,
        source_audio=cut.audio,
        source_spans=(cut.span,),
        source_artifact_id=component.oracle_component_id,
        source_pcm_sha256=cut.pcm_sha256,
        oracle={
            "oracle_component_id": component.oracle_component_id,
            "oracle_component_map_sha256": component.component_map_sha256,
            "oracle_event_ids": component.event_ids,
            "oracle_reference_set_id": (
                component.reference_set_manifest.reference_set_id
            ),
            "oracle_reference_set_sha256": (
                component.reference_set_manifest.reference_set_sha256
            ),
            "oracle_slice_pcm_sha256": cut.pcm_sha256,
        },
    )


__all__ = (
    "EXECUTION_PROVIDER",
    "FP32_MODEL_FILES",
    "NEMO_TDT_ADAPTER",
    "NEMO_TDT_MODEL_TYPE",
    "ONNX_ASR_VERSION",
    "NemoTdtArmResult",
    "NemoTdtCallEvidence",
    "NemoTdtDecoder",
    "decoder_identity",
    "installed_runtime_versions",
    "load_nemo_tdt",
    "require_fp32_artifact",
    "run_nemo_p0",
    "run_nemo_p1",
)
