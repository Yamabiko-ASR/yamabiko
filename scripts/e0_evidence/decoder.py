"""Version-pinned decoder instrumentation for E0 diagnostic arms.

The production Yamabiko wrapper remains the P0/P1 execution path.  This
module records its exact padded ndarray and lazy dependency yields, then uses
the same immutable bytes for the P2 direct-decoder controls.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np
import ctranslate2

from scripts.e0_evidence import contracts
from scripts.e0_evidence._common import require_sha256, require_uuid4
from scripts.e0_evidence.assets import require_local_evaluation
from scripts.e0_evidence.ledger import (
    EvidenceSink,
    decoder_call_diagnostics,
    decoder_sample_bounds,
    raw_hypothesis_diagnostics,
    stochastic_tail_start,
)
from scripts.e0_evidence.pcm import (
    CanonicalMaster,
    CaptureReplayStream,
    capture_replay_provenance_sha256,
)
from scripts.e0_evidence.replay import ReplayWindow
from scripts.e0_evidence.references import (
    ReferenceSetManifest,
    load_reference_set,
)
from scripts.e0_evidence.schema import (
    ArmKind,
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
from scripts.e0_evidence.wire import canonical_sha256
from yamabiko import __version__ as yamabiko_version
from yamabiko.asr.asr_client import (
    ASROutcome,
    ASRResult,
    FasterWhisperASR,
    _COMPRESSION_RATIO_THRESHOLD,
    _LOGPROB_THRESHOLD,
    _MIN_COMPRESSION_GATE_LEN,
    _NO_SPEECH_THRESHOLD,
)
# Bound by name: tests replace the module attribute with a fake factory, and the
# pinned contract must always come from the real class.
from yamabiko.asr.faster_whisper_audio_compat import import_whisper_model


_SAMPLE_RATE = 16_000
_ORACLE_CONTEXT_SAMPLES = 8_000
_SOURCE_MAP_VERSION = 1
_MODEL_ARTIFACT_ATTRIBUTE = "_yamabiko_e0_model_artifact_sha256"


class DecoderInstrumentationError(RuntimeError):
    """Raised when the recorder cannot prove its production parity contract."""


class DecoderParityError(DecoderInstrumentationError):
    """Raised when a P2a direct rerun's greedy segments differ from P0."""


class _ArmEvidenceSink(EvidenceSink, Protocol):
    arm_id: str

    def emit_private_text(
        self,
        *,
        canonical_record_id: str,
        arm_id: str,
        text_id: str,
        role: TextRole,
        text: str,
    ) -> str: ...


def _plain_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer of at least {minimum}")
    return value


def _finite_float(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise TypeError(f"{field_name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _probability(value: object, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class SourceDecoderInterval:
    decoder_start: int
    decoder_end: int
    source_span: SampleSpan | None

    def __post_init__(self) -> None:
        _plain_int(self.decoder_start, "decoder_start")
        _plain_int(self.decoder_end, "decoder_end", minimum=1)
        if self.decoder_end <= self.decoder_start:
            raise ValueError("decoder intervals must be non-empty half-open ranges")
        if self.source_span is not None:
            if not isinstance(self.source_span, SampleSpan):
                raise TypeError("source_span must be SampleSpan or None")
            if self.source_span.sample_count != self.decoder_end - self.decoder_start:
                raise ValueError("real source and decoder interval lengths must agree")


@dataclass(frozen=True, slots=True)
class ProjectedSpan:
    decoder_start: int
    decoder_end: int
    source_spans: tuple[SampleSpan, ...]
    synthetic_pad_hypothesis: bool
    synthetic_anchor_timeline_id: str | None = None
    synthetic_anchor_sample: int | None = None
    timestamp_clamped: bool = False

    def __post_init__(self) -> None:
        _plain_int(self.decoder_start, "decoder_start")
        _plain_int(self.decoder_end, "decoder_end", minimum=1)
        if self.decoder_end <= self.decoder_start:
            raise ValueError("projected decoder spans must be non-empty")
        spans = tuple(self.source_spans)
        for span in spans:
            if not isinstance(span, SampleSpan):
                raise TypeError("source_spans must contain SampleSpan values")
        object.__setattr__(self, "source_spans", spans)
        if type(self.synthetic_pad_hypothesis) is not bool:
            raise TypeError("synthetic_pad_hypothesis must be bool")
        if type(self.timestamp_clamped) is not bool:
            raise TypeError("timestamp_clamped must be bool")
        if self.synthetic_pad_hypothesis:
            if spans:
                raise ValueError("synthetic-only hypotheses cannot claim source spans")
            require_uuid4(
                self.synthetic_anchor_timeline_id,
                "synthetic_anchor_timeline_id",
            )
            _plain_int(self.synthetic_anchor_sample, "synthetic_anchor_sample")
        elif (
            self.synthetic_anchor_timeline_id is not None
            or self.synthetic_anchor_sample is not None
        ):
            raise ValueError("real projections must not carry a synthetic anchor")


@dataclass(frozen=True, slots=True)
class RawSegmentEvidence:
    segment_id: str
    start_decoder_sample: int
    end_decoder_sample: int
    projected_source_spans: tuple[SampleSpan, ...]
    text_id: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    temperature: float
    raw_start_seconds: float
    raw_end_seconds: float
    timestamp_clamped: bool
    synthetic_pad_hypothesis: bool
    synthetic_anchor_timeline_id: str | None = None
    synthetic_anchor_sample: int | None = None

    def __post_init__(self) -> None:
        require_uuid4(self.segment_id, "segment_id")
        _plain_int(self.start_decoder_sample, "start_decoder_sample")
        _plain_int(self.end_decoder_sample, "end_decoder_sample", minimum=1)
        if self.end_decoder_sample <= self.start_decoder_sample:
            raise ValueError("raw decoder spans must be non-empty")
        spans = tuple(self.projected_source_spans)
        for span in spans:
            if not isinstance(span, SampleSpan):
                raise TypeError("projected_source_spans must contain SampleSpan values")
        object.__setattr__(self, "projected_source_spans", spans)
        require_uuid4(self.text_id, "text_id")
        object.__setattr__(
            self,
            "no_speech_prob",
            _probability(self.no_speech_prob, "no_speech_prob"),
        )
        object.__setattr__(
            self,
            "avg_logprob",
            _finite_float(self.avg_logprob, "avg_logprob"),
        )
        compression = _finite_float(self.compression_ratio, "compression_ratio")
        if compression < 0.0:
            raise ValueError("compression_ratio must be nonnegative")
        object.__setattr__(self, "compression_ratio", compression)
        object.__setattr__(
            self,
            "temperature",
            _finite_float(self.temperature, "temperature"),
        )
        raw_start = _finite_float(self.raw_start_seconds, "raw_start_seconds")
        raw_end = _finite_float(self.raw_end_seconds, "raw_end_seconds")
        if raw_start < 0.0 or raw_end <= raw_start:
            raise ValueError("raw dependency seconds must form a nonnegative span")
        object.__setattr__(self, "raw_start_seconds", raw_start)
        object.__setattr__(self, "raw_end_seconds", raw_end)
        if type(self.timestamp_clamped) is not bool:
            raise TypeError("timestamp_clamped must be bool")
        if type(self.synthetic_pad_hypothesis) is not bool:
            raise TypeError("synthetic_pad_hypothesis must be bool")
        if self.synthetic_pad_hypothesis:
            if spans:
                raise ValueError(
                    "synthetic-only raw segments cannot claim source spans"
                )
            require_uuid4(
                self.synthetic_anchor_timeline_id,
                "synthetic_anchor_timeline_id",
            )
            _plain_int(self.synthetic_anchor_sample, "synthetic_anchor_sample")
        elif (
            self.synthetic_anchor_timeline_id is not None
            or self.synthetic_anchor_sample is not None
        ):
            raise ValueError("real raw segments must not carry a synthetic anchor")


@dataclass(frozen=True, slots=True)
class OracleComponent:
    oracle_component_id: str
    reference_set_manifest: ReferenceSetManifest
    master: CanonicalMaster
    event_ids: tuple[str, ...]
    component_map_sha256: str
    event_spans: tuple[SampleSpan, ...] = field(init=False)

    def __post_init__(self) -> None:
        require_uuid4(self.oracle_component_id, "oracle_component_id")
        if not isinstance(self.reference_set_manifest, ReferenceSetManifest):
            raise TypeError("reference_set_manifest must be ReferenceSetManifest")
        if not isinstance(self.master, CanonicalMaster):
            raise TypeError("master must be CanonicalMaster")
        reference_set = load_reference_set(self.reference_set_manifest, self.master)
        event_ids = tuple(self.event_ids)
        if not event_ids:
            raise ValueError("event_ids must be non-empty")
        for event_id in event_ids:
            require_uuid4(event_id, "event_ids item")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_ids must not contain duplicates")
        event_by_id = {event.event_id: event for event in reference_set.events}
        unknown = set(event_ids) - set(event_by_id)
        if unknown:
            raise ValueError("oracle component names an unknown reference event")
        pairs = [(event_id, event_by_id[event_id].span) for event_id in event_ids]
        pairs.sort(key=lambda item: (item[1].start, item[1].end, item[0]))
        object.__setattr__(self, "event_ids", tuple(item[0] for item in pairs))
        object.__setattr__(self, "event_spans", tuple(item[1] for item in pairs))
        require_sha256(self.component_map_sha256, "component_map_sha256")
        expected = oracle_component_map_sha256(
            self.oracle_component_id,
            self.reference_set_manifest,
            self.master,
            self.event_ids,
        )
        if self.component_map_sha256 != expected:
            raise ValueError("oracle component map hash does not match its membership")


@dataclass(frozen=True, slots=True)
class DecoderCallEvidence:
    call_mode: ArmKind
    call_record: EvidenceRecord
    sample_rate: int
    decoder_audio: np.ndarray
    decoder_input_pcm_sha256: str
    source_to_decoder_map_id: str
    source_to_decoder_map_sha256: str
    source_to_decoder_map: tuple[SourceDecoderInterval, ...]
    call_kwargs: Mapping[str, object]
    effective_arguments: Mapping[str, object]
    effective_argument_sequence_kinds: Mapping[str, str]
    effective_arguments_sha256: str
    delegate_signature_sha256: str
    delegate_instance_id: str
    delegate_class_sha256: str
    model_artifact_sha256: str
    delegate: object = field(compare=False, repr=False)
    raw_segments: tuple[RawSegmentEvidence, ...]
    raw_texts: tuple[str, ...]
    raw_record_ids: tuple[str, ...]
    changed_argument_keys: tuple[str, ...]
    source_artifact_id: str
    source_pcm_sha256: str
    # The wrapper's no_speech_only_threshold for this call (None or 1.0: off).
    # Kept in memory only, so a direct arm applies its parent's wrapper;
    # verify-run reads the same value from run-start's effective_config.
    no_speech_only_threshold: float | None
    oracle_component_id: str | None = None
    oracle_reference_set_id: str | None = None
    oracle_reference_set_sha256: str | None = None
    oracle_event_ids: tuple[str, ...] = ()
    oracle_component_map_sha256: str | None = None
    oracle_slice_pcm_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_mode, ArmKind):
            raise TypeError("call_mode must be ArmKind")
        if not isinstance(self.call_record, EvidenceRecord):
            raise TypeError("call_record must be EvidenceRecord")
        if self.call_record.kind is not RecordKind.DECODER_CALL:
            raise ValueError("call_record must be a decoder-call record")
        if self.sample_rate != _SAMPLE_RATE:
            raise ValueError("decoder evidence requires 16000 Hz")
        _require_decoder_audio(self.decoder_audio, "decoder_audio")
        require_sha256(self.decoder_input_pcm_sha256, "decoder_input_pcm_sha256")
        actual = hashlib.sha256(self.decoder_audio.tobytes(order="C")).hexdigest()
        if actual != self.decoder_input_pcm_sha256:
            raise ValueError("decoder audio hash mismatch")
        require_uuid4(self.source_to_decoder_map_id, "source_to_decoder_map_id")
        require_sha256(self.source_to_decoder_map_sha256, "source_to_decoder_map_sha256")
        mapping = tuple(self.source_to_decoder_map)
        _validate_source_map(mapping, self.decoder_audio.size)
        object.__setattr__(self, "source_to_decoder_map", mapping)
        object.__setattr__(
            self, "call_kwargs", MappingProxyType(dict(self.call_kwargs))
        )
        object.__setattr__(
            self,
            "effective_arguments",
            MappingProxyType(dict(self.effective_arguments)),
        )
        object.__setattr__(
            self,
            "effective_argument_sequence_kinds",
            MappingProxyType(dict(self.effective_argument_sequence_kinds)),
        )
        require_sha256(self.effective_arguments_sha256, "effective_arguments_sha256")
        require_sha256(self.delegate_signature_sha256, "delegate_signature_sha256")
        require_uuid4(self.delegate_instance_id, "delegate_instance_id")
        require_sha256(self.delegate_class_sha256, "delegate_class_sha256")
        require_sha256(self.model_artifact_sha256, "model_artifact_sha256")
        if not callable(getattr(self.delegate, "transcribe", None)):
            raise TypeError("delegate must expose transcribe")
        segments = tuple(self.raw_segments)
        texts = tuple(self.raw_texts)
        record_ids = tuple(self.raw_record_ids)
        if not (len(segments) == len(texts) == len(record_ids)):
            raise ValueError("raw segment, text, and record counts must agree")
        for text in texts:
            if type(text) is not str or not text:
                raise ValueError("raw dependency text must be non-empty")
        for record_id in record_ids:
            require_uuid4(record_id, "raw_record_ids item")
        object.__setattr__(self, "raw_segments", segments)
        object.__setattr__(self, "raw_texts", texts)
        object.__setattr__(self, "raw_record_ids", record_ids)
        changes = tuple(self.changed_argument_keys)
        if len(changes) != len(set(changes)):
            raise ValueError("changed_argument_keys must not contain duplicates")
        object.__setattr__(self, "changed_argument_keys", changes)
        require_uuid4(self.source_artifact_id, "source_artifact_id")
        require_sha256(self.source_pcm_sha256, "source_pcm_sha256")
        threshold = self.no_speech_only_threshold
        if threshold is not None and (
            type(threshold) not in (int, float) or not 0.0 < threshold <= 1.0
        ):
            raise ValueError("no_speech_only_threshold must be None or in (0, 1]")
        oracle_values = (
            self.oracle_component_id,
            self.oracle_reference_set_id,
            self.oracle_reference_set_sha256,
            self.oracle_component_map_sha256,
            self.oracle_slice_pcm_sha256,
        )
        if any(value is not None for value in oracle_values):
            if not all(value is not None for value in oracle_values):
                raise ValueError("oracle provenance fields must be all null or all set")
            require_uuid4(self.oracle_component_id, "oracle_component_id")
            require_uuid4(self.oracle_reference_set_id, "oracle_reference_set_id")
            require_sha256(self.oracle_reference_set_sha256, "oracle_reference_set_sha256")
            require_sha256(self.oracle_component_map_sha256, "oracle_component_map_sha256")
            require_sha256(self.oracle_slice_pcm_sha256, "oracle_slice_pcm_sha256")
            event_ids = tuple(self.oracle_event_ids)
            if not event_ids:
                raise ValueError("oracle_event_ids must be non-empty")
            for event_id in event_ids:
                require_uuid4(event_id, "oracle_event_ids item")
            if len(event_ids) != len(set(event_ids)):
                raise ValueError("oracle_event_ids must not contain duplicates")
            object.__setattr__(self, "oracle_event_ids", event_ids)
        elif self.oracle_event_ids:
            raise ValueError("non-oracle calls cannot carry oracle_event_ids")


@dataclass(frozen=True, slots=True)
class DecoderArmResult:
    arm: ArmKind
    call: DecoderCallEvidence
    asr_result: ASRResult
    raw_records: tuple[EvidenceRecord, ...]
    wrapper_records: tuple[EvidenceRecord, ...]
    wrapper_record: EvidenceRecord
    evidence_records: tuple[EvidenceRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ArmKind) or self.call.call_mode is not self.arm:
            raise TypeError("arm and call mode must agree")
        if not isinstance(self.asr_result, ASRResult):
            raise TypeError("asr_result must be ASRResult")
        raw_records = tuple(self.raw_records)
        wrapper_records = tuple(self.wrapper_records)
        evidence_records = tuple(self.evidence_records)
        if self.wrapper_record not in wrapper_records:
            raise ValueError("wrapper_record must be the terminal wrapper record")
        object.__setattr__(self, "raw_records", raw_records)
        object.__setattr__(self, "wrapper_records", wrapper_records)
        object.__setattr__(self, "evidence_records", evidence_records)


@dataclass(frozen=True, slots=True)
class _RawSnapshot:
    start_seconds: float
    end_seconds: float
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    temperature: float


@dataclass(frozen=True, slots=True)
class _ArgumentContract:
    call_kwargs: Mapping[str, object]
    effective_arguments: Mapping[str, object]
    sequence_kinds: Mapping[str, str]
    safe_arguments: Mapping[str, JsonValue]
    safe_arguments_sha256: str
    signature_sha256: str


@dataclass(frozen=True, slots=True)
class _WrapperClassification:
    text: str
    outcome: ASROutcome
    kept_indices: tuple[int, ...]
    confidence_rejected_indices: tuple[int, ...]
    compression_rejected: bool


def _arm_sink(sink: EvidenceSink) -> _ArmEvidenceSink:
    arm_id = getattr(sink, "arm_id", None)
    emitter = getattr(sink, "emit_private_text", None)
    require_uuid4(arm_id, "sink.arm_id")
    if not callable(emitter):
        raise TypeError("decoder sink must expose emit_private_text")
    return cast(_ArmEvidenceSink, sink)


def _require_decoder_audio(audio: object, field_name: str) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"{field_name} must be an ndarray")
    if audio.ndim != 1 or audio.dtype != np.dtype("<f4"):
        raise ValueError(f"{field_name} must be one-dimensional little-endian float32")
    if audio.size < 1 or not audio.flags.c_contiguous or audio.flags.writeable:
        raise ValueError(f"{field_name} must be non-empty, contiguous, and immutable")
    if not np.isfinite(audio).all():
        raise ValueError(f"{field_name} must contain only finite samples")
    return audio


def _immutable_audio(audio: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(audio, dtype="<f4").copy()
    result.setflags(write=False)
    return result


def _package_version() -> str:
    try:
        result = version("faster-whisper")
    except PackageNotFoundError as exc:
        raise DecoderInstrumentationError("faster-whisper is not installed") from exc
    if result != contracts.FASTER_WHISPER_VERSION:
        raise DecoderInstrumentationError(
            "decoder instrumentation requires faster-whisper "
            f"{contracts.FASTER_WHISPER_VERSION}"
        )
    return result


def _verify_wrapper_contract() -> str:
    if yamabiko_version != contracts.YAMABIKO_VERSION_PIN:
        raise DecoderInstrumentationError("Yamabiko wrapper version is not pinned")
    source_sha256 = contracts.source_sha256(FasterWhisperASR.transcribe)
    if source_sha256 != contracts.WRAPPER_SOURCE_SHA256:
        raise DecoderInstrumentationError("Yamabiko wrapper source changed")
    return source_sha256


@lru_cache(maxsize=1)
def _pinned_transcribe_contract() -> tuple[inspect.Signature, tuple[str, ...], str]:
    """Inspect the real faster-whisper class, imported through the audio shim."""

    unbound = inspect.signature(import_whisper_model().transcribe)
    signature = unbound.replace(
        parameters=tuple(unbound.parameters.values())[1:]
    )
    argument_names = tuple(signature.parameters)[1:]
    signature_sha256 = hashlib.sha256(str(signature).encode("utf-8")).hexdigest()
    return signature, argument_names, signature_sha256


def _signature_sha256(delegate: object) -> str:
    method = getattr(delegate, "transcribe", None)
    if not callable(method):
        raise TypeError("decoder delegate must expose transcribe")
    signature = inspect.signature(method)
    pinned_signature, _argument_names, pinned_sha256 = (
        _pinned_transcribe_contract()
    )
    if signature != pinned_signature:
        raise DecoderInstrumentationError(
            "delegate transcribe signature differs from faster-whisper 1.2.1"
        )
    signature_sha256 = hashlib.sha256(str(signature).encode("utf-8")).hexdigest()
    if signature_sha256 != pinned_sha256:
        raise DecoderInstrumentationError("delegate signature hash mismatch")
    return signature_sha256


def _delegate_class_sha256(delegate: object) -> str:
    delegate_type = type(delegate)
    identity = f"{delegate_type.__module__}.{delegate_type.__qualname__}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def bind_model_artifact(
    asr: FasterWhisperASR,
    model_artifact_sha256: str,
) -> FasterWhisperASR:
    """Bind a precomputed model/cache artifact digest before any E0 call."""

    if not isinstance(asr, FasterWhisperASR):
        raise TypeError("asr must be FasterWhisperASR")
    digest = require_sha256(model_artifact_sha256, "model_artifact_sha256")
    try:
        setattr(asr._model, _MODEL_ARTIFACT_ATTRIBUTE, digest)
    except (AttributeError, TypeError) as exc:
        raise TypeError("decoder delegate cannot retain E0 model provenance") from exc
    return asr


def _model_artifact_sha256(delegate: object) -> str:
    try:
        value = getattr(delegate, _MODEL_ARTIFACT_ATTRIBUTE)
    except AttributeError as exc:
        raise DecoderInstrumentationError(
            "decoder model artifact hash was not bound before inference"
        ) from exc
    return require_sha256(value, "delegate model artifact SHA-256")


def _freeze_argument(
    value: object, sequence_kinds: dict[str, str], path: str
) -> object:
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"effective argument {path} must be finite")
        return value
    if isinstance(value, np.generic):
        return _freeze_argument(value.item(), sequence_kinds, path)
    if isinstance(value, Enum):
        return _freeze_argument(value.value, sequence_kinds, path)
    if isinstance(value, list):
        sequence_kinds[path] = "list"
        return tuple(
            _freeze_argument(item, sequence_kinds, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, tuple):
        sequence_kinds[path] = "tuple"
        return tuple(
            _freeze_argument(item, sequence_kinds, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"effective argument {path} has a non-string key")
            result[key] = _freeze_argument(item, sequence_kinds, f"{path}.{key}")
        return MappingProxyType(result)
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_argument(asdict(value), sequence_kinds, path)
    raise TypeError(f"effective argument {path} has an unsupported type")


def _safe_argument(value: object) -> JsonValue:
    if value is None or type(value) in (bool, int, float):
        return cast(JsonValue, value)
    if type(value) is str:
        encoded = value.encode("utf-8")
        return {
            "kind": "private_utf8",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "utf8_bytes": len(encoded),
        }
    if isinstance(value, tuple):
        return tuple(_safe_argument(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _safe_argument(item) for key, item in value.items()}
    raise TypeError("frozen effective argument cannot be serialized")


def _resolve_arguments(
    delegate: object,
    audio: np.ndarray,
    kwargs: Mapping[str, object],
) -> _ArgumentContract:
    _package_version()
    signature_sha256 = _signature_sha256(delegate)
    signature = inspect.signature(delegate.transcribe)
    try:
        bound = signature.bind_partial(audio, **dict(kwargs))
    except TypeError as exc:
        raise DecoderInstrumentationError("decoder arguments do not bind") from exc
    bound.apply_defaults()
    if "audio" not in bound.arguments:
        raise DecoderInstrumentationError("bound decoder call is missing audio")
    bound.arguments.pop("audio")
    _signature, argument_names, _signature_sha256_value = (
        _pinned_transcribe_contract()
    )
    if tuple(bound.arguments) != argument_names:
        raise DecoderInstrumentationError(
            "not every pinned decoder argument was resolved"
        )
    sequence_kinds: dict[str, str] = {}
    frozen: dict[str, object] = {}
    for name, value in bound.arguments.items():
        frozen[name] = _freeze_argument(
            copy.deepcopy(value),
            sequence_kinds,
            name,
        )
    safe = {name: _safe_argument(value) for name, value in frozen.items()}
    return _ArgumentContract(
        call_kwargs=MappingProxyType(
            {
                name: _freeze_argument(copy.deepcopy(value), {}, f"call_kwargs.{name}")
                for name, value in kwargs.items()
            }
        ),
        effective_arguments=MappingProxyType(frozen),
        sequence_kinds=MappingProxyType(sequence_kinds),
        safe_arguments=MappingProxyType(safe),
        safe_arguments_sha256=canonical_sha256(safe),
        signature_sha256=signature_sha256,
    )


def _thaw_argument(
    value: object, sequence_kinds: Mapping[str, str], path: str
) -> object:
    if isinstance(value, tuple):
        items = [
            _thaw_argument(item, sequence_kinds, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(items) if sequence_kinds.get(path) == "tuple" else items
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_argument(item, sequence_kinds, f"{path}.{key}")
            for key, item in value.items()
        }
    return copy.deepcopy(value)


def _thaw_arguments(
    arguments: Mapping[str, object],
    sequence_kinds: Mapping[str, str],
) -> dict[str, object]:
    return {
        name: _thaw_argument(value, sequence_kinds, name)
        for name, value in arguments.items()
    }


def _snapshot_segment(segment: object) -> _RawSnapshot:
    text = getattr(segment, "text", None)
    if type(text) is not str or not text:
        raise DecoderInstrumentationError("dependency yielded an empty segment text")
    temperature = getattr(segment, "temperature", None)
    if temperature is None:
        raise DecoderInstrumentationError("dependency segment temperature is missing")
    start = _finite_float(getattr(segment, "start", None), "segment.start")
    end = _finite_float(getattr(segment, "end", None), "segment.end")
    if start < 0.0 or end <= start:
        raise DecoderInstrumentationError("dependency segment timestamps are invalid")
    compression = _finite_float(
        getattr(segment, "compression_ratio", None),
        "segment.compression_ratio",
    )
    if compression < 0.0:
        raise DecoderInstrumentationError("segment compression ratio is negative")
    return _RawSnapshot(
        start_seconds=start,
        end_seconds=end,
        text=text,
        no_speech_prob=_probability(
            getattr(segment, "no_speech_prob", None),
            "segment.no_speech_prob",
        ),
        avg_logprob=_finite_float(
            getattr(segment, "avg_logprob", None),
            "segment.avg_logprob",
        ),
        compression_ratio=compression,
        temperature=_finite_float(temperature, "segment.temperature"),
    )


class RecordingModelProxy:
    """One-call delegating proxy that snapshots raw segments without prefetch."""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.decoder_audio: np.ndarray | None = None
        self.decoder_input_pcm_sha256: str | None = None
        self.argument_contract: _ArgumentContract | None = None
        self.raw_snapshots: list[_RawSnapshot] = []
        self.completed = False
        self._called = False

    def transcribe(
        self,
        audio: np.ndarray,
        **kwargs: object,
    ) -> tuple[Iterable[object], object]:
        if self._called:
            raise DecoderInstrumentationError("recording proxy is one-call only")
        self._called = True
        if not isinstance(audio, np.ndarray) or audio.ndim != 1:
            raise DecoderInstrumentationError("wrapper passed a non-mono ndarray")
        captured = _immutable_audio(audio)
        self.decoder_audio = captured
        self.decoder_input_pcm_sha256 = hashlib.sha256(
            captured.tobytes(order="C")
        ).hexdigest()
        self.argument_contract = _resolve_arguments(self.delegate, captured, kwargs)
        segments, info = self.delegate.transcribe(audio, **kwargs)

        def recording_iterator() -> Iterable[object]:
            for segment in segments:
                self.raw_snapshots.append(_snapshot_segment(segment))
                yield segment
            self.completed = True

        return recording_iterator(), info


def _validate_source_map(
    mapping: Sequence[SourceDecoderInterval],
    decoder_samples: int,
) -> tuple[SourceDecoderInterval, ...]:
    _plain_int(decoder_samples, "decoder_samples", minimum=1)
    intervals = tuple(mapping)
    if not intervals:
        raise ValueError("source-to-decoder mapping must not be empty")
    cursor = 0
    real_count = 0
    for index, interval in enumerate(intervals):
        if not isinstance(interval, SourceDecoderInterval):
            raise TypeError("mapping must contain SourceDecoderInterval values")
        if interval.decoder_start != cursor:
            raise ValueError("source-to-decoder mapping must be contiguous")
        cursor = interval.decoder_end
        if interval.source_span is not None:
            real_count += 1
        elif index not in (0, len(intervals) - 1):
            raise ValueError("synthetic intervals are allowed only at boundaries")
    if cursor != decoder_samples:
        raise ValueError("source-to-decoder mapping does not cover the input")
    if real_count < 1:
        raise ValueError("source-to-decoder mapping needs real source audio")
    return intervals


def _source_map_wire(
    mapping: Sequence[SourceDecoderInterval],
) -> tuple[Mapping[str, JsonValue], ...]:
    result: list[Mapping[str, JsonValue]] = []
    for interval in mapping:
        source = interval.source_span
        result.append(
            {
                "decoder_end": interval.decoder_end,
                "decoder_start": interval.decoder_start,
                "source_span": (
                    None
                    if source is None
                    else {
                        "end": source.end,
                        "start": source.start,
                        "timeline_id": source.timeline_id,
                    }
                ),
            }
        )
    return tuple(result)


def _source_map_sha256(
    mapping: Sequence[SourceDecoderInterval],
    decoder_samples: int,
) -> str:
    return canonical_sha256(
        {
            "decoder_samples": decoder_samples,
            "intervals": _source_map_wire(mapping),
            "mapping_version": _SOURCE_MAP_VERSION,
            "sample_rate": _SAMPLE_RATE,
        }
    )


def _mapping_for_source(
    source_spans: Sequence[SampleSpan],
    real_samples: int,
    decoder_samples: int,
) -> tuple[SourceDecoderInterval, ...]:
    spans = tuple(source_spans)
    if not spans or sum(span.sample_count for span in spans) != real_samples:
        raise ValueError("source spans do not account for the real decoder audio")
    padding = decoder_samples - real_samples
    if padding < 0 or padding % 2:
        raise DecoderInstrumentationError("wrapper padding is not symmetric")
    pad = padding // 2
    intervals: list[SourceDecoderInterval] = []
    cursor = 0
    if pad:
        intervals.append(SourceDecoderInterval(0, pad, None))
        cursor = pad
    for span in spans:
        intervals.append(
            SourceDecoderInterval(cursor, cursor + span.sample_count, span)
        )
        cursor += span.sample_count
    if pad:
        intervals.append(SourceDecoderInterval(cursor, cursor + pad, None))
        cursor += pad
    return _validate_source_map(intervals, decoder_samples)


def _nearest_synthetic_anchor(
    decoder_start: int,
    decoder_end: int,
    mapping: Sequence[SourceDecoderInterval],
) -> tuple[str, int]:
    midpoint = (decoder_start + decoder_end) / 2.0
    candidates: list[tuple[float, int, str, int]] = []
    for interval in mapping:
        source = interval.source_span
        if source is None:
            continue
        candidates.append(
            (
                abs(midpoint - interval.decoder_start),
                interval.decoder_start,
                source.timeline_id,
                source.start,
            )
        )
        candidates.append(
            (
                abs(midpoint - interval.decoder_end),
                interval.decoder_end,
                source.timeline_id,
                source.end,
            )
        )
    if not candidates:
        raise ValueError("synthetic hypothesis has no real source boundary")
    _, _, timeline_id, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return timeline_id, sample


def project_decoder_span(
    start_s: float,
    end_s: float,
    sample_rate: int,
    decoder_samples: int,
    mapping: Sequence[SourceDecoderInterval],
) -> ProjectedSpan:
    """Project one dependency segment without assigning synthetic pad to source.

    Timestamps are clamped to the decoder input by ``decoder_sample_bounds``;
    the result reports whether clamping moved either boundary.
    """

    start_seconds = _finite_float(start_s, "start_s")
    end_seconds = _finite_float(end_s, "end_s")
    if start_seconds < 0.0 or end_seconds <= start_seconds:
        raise ValueError("decoder timestamps must form a positive nonnegative span")
    _plain_int(sample_rate, "sample_rate", minimum=1)
    intervals = _validate_source_map(mapping, decoder_samples)
    decoder_start, decoder_end, clamped = decoder_sample_bounds(
        start_seconds,
        end_seconds,
        sample_rate,
        decoder_samples,
    )
    projected: list[SampleSpan] = []
    for interval in intervals:
        source = interval.source_span
        if source is None:
            continue
        overlap_start = max(decoder_start, interval.decoder_start)
        overlap_end = min(decoder_end, interval.decoder_end)
        if overlap_end <= overlap_start:
            continue
        candidate = SampleSpan(
            source.timeline_id,
            source.start + overlap_start - interval.decoder_start,
            source.start + overlap_end - interval.decoder_start,
        )
        if (
            projected
            and projected[-1].timeline_id == candidate.timeline_id
            and projected[-1].end == candidate.start
        ):
            previous = projected[-1]
            projected[-1] = SampleSpan(
                previous.timeline_id, previous.start, candidate.end
            )
        else:
            projected.append(candidate)
    if projected:
        return ProjectedSpan(
            decoder_start=decoder_start,
            decoder_end=decoder_end,
            source_spans=tuple(projected),
            synthetic_pad_hypothesis=False,
            timestamp_clamped=clamped,
        )
    timeline_id, sample = _nearest_synthetic_anchor(
        decoder_start,
        decoder_end,
        intervals,
    )
    return ProjectedSpan(
        decoder_start=decoder_start,
        decoder_end=decoder_end,
        source_spans=(),
        synthetic_pad_hypothesis=True,
        synthetic_anchor_timeline_id=timeline_id,
        synthetic_anchor_sample=sample,
        timestamp_clamped=clamped,
    )


def _classify_wrapper(
    snapshots: Sequence[_RawSnapshot],
    no_speech_only_threshold: float | None,
) -> _WrapperClassification:
    """Reproduce ``FasterWhisperASR.transcribe``'s decisions over raw yields.

    ``no_speech_only_threshold`` is the wrapper's
    ``asr.faster_whisper.no_speech_only_threshold`` (None or 1.0: off). As in
    production, a segment whose ``no_speech_prob`` exceeds it is dropped
    before the paired gate runs; both drops are confidence-gate decisions.
    """
    kept_indices: list[int] = []
    confidence_rejected: list[int] = []
    kept_text: list[str] = []
    ratios: list[float] = []
    for index, snapshot in enumerate(snapshots):
        if (
            no_speech_only_threshold is not None
            and snapshot.no_speech_prob > no_speech_only_threshold
        ):
            confidence_rejected.append(index)
            continue
        if (
            snapshot.no_speech_prob > _NO_SPEECH_THRESHOLD
            and snapshot.avg_logprob < _LOGPROB_THRESHOLD
        ):
            confidence_rejected.append(index)
            continue
        kept_indices.append(index)
        kept_text.append(snapshot.text)
        ratios.append(snapshot.compression_ratio)
    text = "".join(kept_text).strip()
    compression_rejected = bool(
        len(text) >= _MIN_COMPRESSION_GATE_LEN
        and ratios
        and max(ratios) > _COMPRESSION_RATIO_THRESHOLD
    )
    if not snapshots:
        outcome = ASROutcome.DEPENDENCY_NO_SEGMENTS
    elif not kept_indices:
        outcome = ASROutcome.ALL_SEGMENTS_CONFIDENCE_REJECTED
    elif not text:
        outcome = ASROutcome.ASSEMBLED_TEXT_EMPTY
    elif compression_rejected:
        outcome = ASROutcome.COMPRESSION_LOOP_REJECTED
    else:
        outcome = ASROutcome.TEXT_AVAILABLE
    if compression_rejected:
        text = ""
    return _WrapperClassification(
        text=text,
        outcome=outcome,
        kept_indices=tuple(kept_indices),
        confidence_rejected_indices=tuple(confidence_rejected),
        compression_rejected=compression_rejected,
    )


def _materialize_raw(
    snapshots: Sequence[_RawSnapshot],
    mapping: Sequence[SourceDecoderInterval],
    decoder_samples: int,
) -> tuple[tuple[RawSegmentEvidence, ...], tuple[str, ...]]:
    evidence: list[RawSegmentEvidence] = []
    texts: list[str] = []
    for snapshot in snapshots:
        projected = project_decoder_span(
            snapshot.start_seconds,
            snapshot.end_seconds,
            _SAMPLE_RATE,
            decoder_samples,
            mapping,
        )
        evidence.append(
            RawSegmentEvidence(
                segment_id=new_opaque_id(),
                start_decoder_sample=projected.decoder_start,
                end_decoder_sample=projected.decoder_end,
                projected_source_spans=projected.source_spans,
                text_id=new_opaque_id(),
                no_speech_prob=snapshot.no_speech_prob,
                avg_logprob=snapshot.avg_logprob,
                compression_ratio=snapshot.compression_ratio,
                temperature=snapshot.temperature,
                raw_start_seconds=snapshot.start_seconds,
                raw_end_seconds=snapshot.end_seconds,
                timestamp_clamped=projected.timestamp_clamped,
                synthetic_pad_hypothesis=projected.synthetic_pad_hypothesis,
                synthetic_anchor_timeline_id=projected.synthetic_anchor_timeline_id,
                synthetic_anchor_sample=projected.synthetic_anchor_sample,
            )
        )
        texts.append(snapshot.text)
    return tuple(evidence), tuple(texts)


def _emit(
    sink: _ArmEvidenceSink,
    records: list[EvidenceRecord],
    draft: EvidenceRecordDraft,
) -> EvidenceRecord:
    record = sink.emit(draft)
    if not isinstance(record, EvidenceRecord):
        raise TypeError("evidence sink must return EvidenceRecord")
    records.append(record)
    return record


def _raw_details(
    ordinal: int,
    segment: RawSegmentEvidence,
) -> Mapping[str, JsonValue]:
    return {
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "end_decoder_sample": segment.end_decoder_sample,
        "no_speech_prob": segment.no_speech_prob,
        "raw_end_seconds": segment.raw_end_seconds,
        "raw_start_seconds": segment.raw_start_seconds,
        "segment_ordinal": ordinal,
        "start_decoder_sample": segment.start_decoder_sample,
        "synthetic_anchor_sample": segment.synthetic_anchor_sample,
        "synthetic_anchor_timeline_id": segment.synthetic_anchor_timeline_id,
        "synthetic_pad_hypothesis": segment.synthetic_pad_hypothesis,
        "temperature": segment.temperature,
        "text_id": segment.text_id,
        "timestamp_clamped": segment.timestamp_clamped,
    }


def _union_spans(spans: Iterable[SampleSpan]) -> tuple[SampleSpan, ...]:
    ordered = sorted(spans, key=lambda span: (span.timeline_id, span.start, span.end))
    result: list[SampleSpan] = []
    for span in ordered:
        if (
            result
            and result[-1].timeline_id == span.timeline_id
            and span.start <= result[-1].end
        ):
            previous = result[-1]
            result[-1] = SampleSpan(
                previous.timeline_id,
                previous.start,
                max(previous.end, span.end),
            )
        else:
            result.append(span)
    return tuple(result)


def _emit_call_result(
    *,
    sink: _ArmEvidenceSink,
    call_mode: ArmKind,
    source_artifact_id: str,
    source_pcm_sha256: str,
    decoder_audio: np.ndarray,
    mapping: tuple[SourceDecoderInterval, ...],
    mapping_id: str,
    mapping_sha256: str,
    argument_contract: _ArgumentContract,
    delegate: object,
    delegate_instance_id: str,
    snapshots: tuple[_RawSnapshot, ...],
    raw_segments: tuple[RawSegmentEvidence, ...],
    raw_texts: tuple[str, ...],
    classification: _WrapperClassification,
    no_speech_only_threshold: float | None,
    duration_s: float,
    parent_call_record_id: str | None,
    changed_argument_keys: tuple[str, ...],
    oracle_component_id: str | None,
    oracle_reference_set_id: str | None,
    oracle_reference_set_sha256: str | None,
    oracle_event_ids: tuple[str, ...],
    oracle_component_map_sha256: str | None,
    oracle_slice_pcm_sha256: str | None,
) -> DecoderArmResult:
    records: list[EvidenceRecord] = []
    decoder_hash = hashlib.sha256(decoder_audio.tobytes(order="C")).hexdigest()
    left_pad = mapping[0].decoder_end if mapping[0].source_span is None else 0
    right_pad = (
        mapping[-1].decoder_end - mapping[-1].decoder_start
        if mapping[-1].source_span is None
        else 0
    )
    temperatures = tuple(segment.temperature for segment in raw_segments)
    stochastic_from = stochastic_tail_start(temperatures)
    call_record = _emit(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=sink.arm_id,
            stage=Stage.DECODER,
            kind=RecordKind.DECODER_CALL,
            artifact_id=new_opaque_id(),
            parent_ids=(parent_call_record_id,) if parent_call_record_id else (),
            source_spans=tuple(
                interval.source_span
                for interval in mapping
                if interval.source_span is not None
            ),
            observation_status=ObservationStatus.OBSERVED,
            decision=(
                DiagnosticCode.DECODER_NO_YIELD.value if not raw_segments else None
            ),
            diagnostics=decoder_call_diagnostics(temperatures),
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={
                "call_mode": call_mode.value,
                "changed_argument_keys": changed_argument_keys,
                "decoder_contract_version": contracts.DECODER_CONTRACT_VERSION,
                "decoder_input_pcm_sha256": decoder_hash,
                "decoder_input_samples": int(decoder_audio.size),
                "decoder_random_seed": contracts.DECODER_RANDOM_SEED,
                "delegate_class_sha256": _delegate_class_sha256(delegate),
                "delegate_instance_id": delegate_instance_id,
                "delegate_signature_sha256": argument_contract.signature_sha256,
                "dependency_version": contracts.FASTER_WHISPER_VERSION,
                "effective_arguments": argument_contract.safe_arguments,
                "effective_arguments_sha256": argument_contract.safe_arguments_sha256,
                "mapping_version": _SOURCE_MAP_VERSION,
                "model_artifact_sha256": _model_artifact_sha256(delegate),
                "oracle_component_id": oracle_component_id,
                "oracle_component_map_sha256": oracle_component_map_sha256,
                "oracle_event_ids": oracle_event_ids,
                "oracle_reference_set_id": oracle_reference_set_id,
                "oracle_reference_set_sha256": oracle_reference_set_sha256,
                "oracle_slice_pcm_sha256": oracle_slice_pcm_sha256,
                "parent_call_record_id": parent_call_record_id,
                "raw_segment_count": len(raw_segments),
                "sample_rate": _SAMPLE_RATE,
                "source_artifact_id": source_artifact_id,
                "source_pcm_sha256": source_pcm_sha256,
                "source_to_decoder_map": _source_map_wire(mapping),
                "source_to_decoder_map_id": mapping_id,
                "source_to_decoder_map_sha256": mapping_sha256,
                "synthetic_left_pad_samples": left_pad,
                "synthetic_right_pad_samples": right_pad,
                "wrapper_source_sha256": contracts.WRAPPER_SOURCE_SHA256,
            },
        ),
    )
    raw_records: list[EvidenceRecord] = []
    for ordinal, (segment, text) in enumerate(zip(raw_segments, raw_texts)):
        raw_record = _emit(
            sink,
            records,
            EvidenceRecordDraft(
                arm_id=sink.arm_id,
                stage=Stage.DECODER,
                kind=RecordKind.RAW_HYPOTHESIS,
                artifact_id=segment.segment_id,
                parent_ids=(call_record.record_id,),
                source_spans=segment.projected_source_spans,
                classification=PrivateContentClass.PRIVATE_AUDIO,
                diagnostics=raw_hypothesis_diagnostics(
                    timestamp_clamped=segment.timestamp_clamped,
                    stochastic=ordinal >= stochastic_from,
                ),
                details=_raw_details(ordinal, segment),
            ),
        )
        raw_records.append(raw_record)
        sink.emit_private_text(
            canonical_record_id=raw_record.record_id,
            arm_id=sink.arm_id,
            text_id=segment.text_id,
            role=TextRole.RAW_HYPOTHESIS,
            text=text,
        )

    wrapper_records: list[EvidenceRecord] = []
    for index in classification.confidence_rejected_indices:
        raw_record = raw_records[index]
        wrapper_records.append(
            _emit(
                sink,
                records,
                EvidenceRecordDraft(
                    arm_id=sink.arm_id,
                    stage=Stage.WRAPPER,
                    kind=RecordKind.WRAPPER_DECISION,
                    artifact_id=new_opaque_id(),
                    parent_ids=(raw_record.record_id,),
                    source_spans=raw_record.source_spans,
                    decision=DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value,
                    classification=PrivateContentClass.PRIVATE_AUDIO,
                ),
            )
        )

    kept_raw_records = [raw_records[index] for index in classification.kept_indices]
    if classification.compression_rejected:
        terminal_decision = DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value
        terminal_parents = tuple(record.record_id for record in kept_raw_records)
        terminal_text_id = None
    elif classification.text:
        terminal_decision = TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value
        terminal_parents = tuple(record.record_id for record in kept_raw_records)
        terminal_text_id = new_opaque_id()
    else:
        terminal_decision = DiagnosticCode.WRAPPER_EMPTY.value
        if kept_raw_records:
            terminal_parents = tuple(record.record_id for record in kept_raw_records)
        elif wrapper_records:
            terminal_parents = tuple(record.record_id for record in wrapper_records)
        else:
            terminal_parents = (call_record.record_id,)
        terminal_text_id = None
    terminal_sources = _union_spans(
        span for record in kept_raw_records for span in record.source_spans
    )
    wrapper_record = _emit(
        sink,
        records,
        EvidenceRecordDraft(
            arm_id=sink.arm_id,
            stage=Stage.WRAPPER,
            kind=RecordKind.WRAPPER_DECISION,
            artifact_id=new_opaque_id(),
            parent_ids=terminal_parents,
            source_spans=terminal_sources,
            decision=terminal_decision,
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details=(
                {"text_id": terminal_text_id} if terminal_text_id is not None else {}
            ),
        ),
    )
    wrapper_records.append(wrapper_record)
    if terminal_text_id is not None:
        sink.emit_private_text(
            canonical_record_id=wrapper_record.record_id,
            arm_id=sink.arm_id,
            text_id=terminal_text_id,
            role=TextRole.WRAPPER_HYPOTHESIS,
            text=classification.text,
        )

    call = DecoderCallEvidence(
        call_mode=call_mode,
        call_record=call_record,
        sample_rate=_SAMPLE_RATE,
        decoder_audio=decoder_audio,
        decoder_input_pcm_sha256=decoder_hash,
        source_to_decoder_map_id=mapping_id,
        source_to_decoder_map_sha256=mapping_sha256,
        source_to_decoder_map=mapping,
        call_kwargs=argument_contract.call_kwargs,
        effective_arguments=argument_contract.effective_arguments,
        effective_argument_sequence_kinds=argument_contract.sequence_kinds,
        effective_arguments_sha256=argument_contract.safe_arguments_sha256,
        delegate_signature_sha256=argument_contract.signature_sha256,
        delegate_instance_id=delegate_instance_id,
        delegate_class_sha256=_delegate_class_sha256(delegate),
        model_artifact_sha256=_model_artifact_sha256(delegate),
        delegate=delegate,
        raw_segments=raw_segments,
        raw_texts=raw_texts,
        raw_record_ids=tuple(record.record_id for record in raw_records),
        changed_argument_keys=changed_argument_keys,
        source_artifact_id=source_artifact_id,
        source_pcm_sha256=source_pcm_sha256,
        no_speech_only_threshold=no_speech_only_threshold,
        oracle_component_id=oracle_component_id,
        oracle_reference_set_id=oracle_reference_set_id,
        oracle_reference_set_sha256=oracle_reference_set_sha256,
        oracle_event_ids=oracle_event_ids,
        oracle_component_map_sha256=oracle_component_map_sha256,
        oracle_slice_pcm_sha256=oracle_slice_pcm_sha256,
    )
    return DecoderArmResult(
        arm=call_mode,
        call=call,
        asr_result=ASRResult(
            text=classification.text,
            duration_s=duration_s,
            outcome=classification.outcome,
        ),
        raw_records=tuple(raw_records),
        wrapper_records=tuple(wrapper_records),
        wrapper_record=wrapper_record,
        evidence_records=tuple(records),
    )


def _validate_window(window: ReplayWindow) -> None:
    if not isinstance(window, ReplayWindow):
        raise TypeError("window must be ReplayWindow")
    require_uuid4(window.window_id, "window.window_id")
    _require_decoder_audio(window.audio, "window.audio")
    require_sha256(window.pcm_sha256, "window.pcm_sha256")
    actual = hashlib.sha256(window.audio.tobytes(order="C")).hexdigest()
    if actual != window.pcm_sha256:
        raise ValueError("replay window PCM hash mismatch")
    if sum(span.sample_count for span in window.source_spans) != window.audio.size:
        raise ValueError("replay window source spans do not cover its audio")


def _run_wrapper(
    *,
    source_audio: np.ndarray,
    source_spans: tuple[SampleSpan, ...],
    source_artifact_id: str,
    source_pcm_sha256: str,
    asr: FasterWhisperASR,
    sink: EvidenceSink,
    call_mode: ArmKind,
    oracle_component_id: str | None = None,
    oracle_reference_set_id: str | None = None,
    oracle_reference_set_sha256: str | None = None,
    oracle_event_ids: tuple[str, ...] = (),
    oracle_component_map_sha256: str | None = None,
    oracle_slice_pcm_sha256: str | None = None,
) -> DecoderArmResult:
    bound_sink = _arm_sink(sink)
    if not isinstance(asr, FasterWhisperASR):
        raise TypeError("asr must be FasterWhisperASR built by build_asr")
    _verify_wrapper_contract()
    delegate = asr._model
    _model_artifact_sha256(delegate)
    delegate_instance_id = new_opaque_id()
    proxy = RecordingModelProxy(delegate)
    asr._model = proxy
    try:
        ctranslate2.set_random_seed(contracts.DECODER_RANDOM_SEED)
        production_result = asr.transcribe(source_audio, _SAMPLE_RATE)
    finally:
        asr._model = delegate
    if not proxy.completed:
        raise DecoderInstrumentationError("production wrapper did not exhaust segments")
    if (
        proxy.decoder_audio is None
        or proxy.decoder_input_pcm_sha256 is None
        or proxy.argument_contract is None
    ):
        raise DecoderInstrumentationError("recording proxy did not capture the call")
    pad = int(round(_SAMPLE_RATE * asr._boundary_pad_ms / 1000))
    expected_parts: list[np.ndarray] = []
    if pad > 0:
        expected_parts.append(np.zeros(pad, dtype=source_audio.dtype))
    expected_parts.append(source_audio)
    if pad > 0:
        expected_parts.append(np.zeros(pad, dtype=source_audio.dtype))
    expected_audio = (
        np.concatenate(expected_parts) if len(expected_parts) > 1 else source_audio
    )
    if not np.array_equal(proxy.decoder_audio, expected_audio):
        raise DecoderInstrumentationError(
            "captured decoder input differs from wrapper padding"
        )
    mapping = _mapping_for_source(
        source_spans,
        source_audio.size,
        proxy.decoder_audio.size,
    )
    mapping_id = new_opaque_id()
    mapping_sha256 = _source_map_sha256(mapping, proxy.decoder_audio.size)
    snapshots = tuple(proxy.raw_snapshots)
    raw_segments, raw_texts = _materialize_raw(
        snapshots,
        mapping,
        proxy.decoder_audio.size,
    )
    # The wrapper's own setting, which build_asr read from the run's effective
    # config (as with _boundary_pad_ms above).
    no_speech_only_threshold = asr._no_speech_only_threshold
    classification = _classify_wrapper(snapshots, no_speech_only_threshold)
    if production_result.text != classification.text:
        raise DecoderInstrumentationError(
            "recorded wrapper decisions do not reproduce ASRResult.text"
        )
    if production_result.outcome is not classification.outcome:
        raise DecoderInstrumentationError(
            "recorded wrapper outcome does not reproduce ASRResult.outcome"
        )
    emitted = _emit_call_result(
        sink=bound_sink,
        call_mode=call_mode,
        source_artifact_id=source_artifact_id,
        source_pcm_sha256=source_pcm_sha256,
        decoder_audio=proxy.decoder_audio,
        mapping=mapping,
        mapping_id=mapping_id,
        mapping_sha256=mapping_sha256,
        argument_contract=proxy.argument_contract,
        delegate=delegate,
        delegate_instance_id=delegate_instance_id,
        snapshots=snapshots,
        raw_segments=raw_segments,
        raw_texts=raw_texts,
        classification=classification,
        no_speech_only_threshold=no_speech_only_threshold,
        duration_s=source_audio.size / _SAMPLE_RATE,
        parent_call_record_id=None,
        changed_argument_keys=(),
        oracle_component_id=oracle_component_id,
        oracle_reference_set_id=oracle_reference_set_id,
        oracle_reference_set_sha256=oracle_reference_set_sha256,
        oracle_event_ids=oracle_event_ids,
        oracle_component_map_sha256=oracle_component_map_sha256,
        oracle_slice_pcm_sha256=oracle_slice_pcm_sha256,
    )
    if emitted.asr_result != production_result:
        raise DecoderInstrumentationError(
            "emitted wrapper result differs from production"
        )
    return emitted


def run_p0(
    window: ReplayWindow,
    asr: FasterWhisperASR,
    sink: EvidenceSink,
) -> DecoderArmResult:
    """Capture the actual production wrapper call for one replay window."""

    _validate_window(window)
    return _run_wrapper(
        source_audio=window.audio,
        source_spans=window.source_spans,
        source_artifact_id=window.window_id,
        source_pcm_sha256=window.pcm_sha256,
        asr=asr,
        sink=sink,
        call_mode=ArmKind.P0,
    )


def _oracle_slice_bounds(
    master_samples: int,
    event_spans: Sequence[SampleSpan],
) -> tuple[int, int]:
    start = max(0, min(span.start for span in event_spans) - _ORACLE_CONTEXT_SAMPLES)
    end = min(
        master_samples,
        max(span.end for span in event_spans) + _ORACLE_CONTEXT_SAMPLES,
    )
    if end <= start:
        raise ValueError("oracle slice must be non-empty")
    return start, end


def oracle_component_map_sha256(
    oracle_component_id: str,
    reference_set_manifest: ReferenceSetManifest,
    master: CanonicalMaster,
    event_ids: Sequence[str],
) -> str:
    """Hash predeclared P1 membership, context policy, and resulting slice."""

    require_uuid4(oracle_component_id, "oracle_component_id")
    if not isinstance(reference_set_manifest, ReferenceSetManifest):
        raise TypeError("reference_set_manifest must be ReferenceSetManifest")
    if not isinstance(master, CanonicalMaster):
        raise TypeError("master must be CanonicalMaster")
    reference_set = load_reference_set(reference_set_manifest, master)
    event_by_id = {event.event_id: event for event in reference_set.events}
    requested_event_ids = tuple(event_ids)
    if not requested_event_ids or len(requested_event_ids) != len(
        set(requested_event_ids)
    ):
        raise ValueError("oracle component event IDs must be non-empty and unique")
    unknown = set(requested_event_ids) - set(event_by_id)
    if unknown:
        raise ValueError("oracle component names an unknown reference event")
    pairs = sorted(
        ((event_id, event_by_id[event_id].span) for event_id in requested_event_ids),
        key=lambda item: (item[1].start, item[1].end, item[0]),
    )
    start, end = _oracle_slice_bounds(
        master.manifest.sample_count,
        tuple(item[1] for item in pairs),
    )
    return canonical_sha256(
        {
            "context_samples_each_side": _ORACLE_CONTEXT_SAMPLES,
            "events": tuple(
                {
                    "event_id": event_id,
                    "span": {
                        "end": span.end,
                        "start": span.start,
                        "timeline_id": span.timeline_id,
                    },
                }
                for event_id, span in pairs
            ),
            "master_id": master.manifest.master_id,
            "oracle_component_id": oracle_component_id,
            "reference_set_id": reference_set_manifest.reference_set_id,
            "reference_set_sha256": reference_set_manifest.reference_set_sha256,
            "slice": {
                "end": end,
                "start": start,
                "timeline_id": master.manifest.timeline_id,
            },
        }
    )


def _validate_oracle_master(master: CanonicalMaster) -> str:
    require_local_evaluation(master.manifest.rights)
    _require_decoder_audio(master.samples, "component.master.samples")
    manifest = master.manifest
    if (
        manifest.sample_rate_hz != _SAMPLE_RATE
        or manifest.channels != 1
        or manifest.layout != "mono"
        or manifest.dtype != "<f4"
        or manifest.sample_count != master.samples.size
    ):
        raise ValueError("oracle master must be immutable canonical 16000 Hz mono PCM")
    actual = hashlib.sha256(master.samples.tobytes(order="C")).hexdigest()
    if manifest.byte_sha256 != actual or manifest.canonical_pcm_sha256 != actual:
        raise ValueError("oracle master PCM hash mismatch")
    return actual


def _validate_oracle_capture(
    capture: CaptureReplayStream,
    master: CanonicalMaster,
) -> None:
    if not isinstance(capture, CaptureReplayStream):
        raise TypeError("capture must be CaptureReplayStream")
    if capture.master != master:
        raise ValueError("capture replay stream belongs to a different master")
    _require_decoder_audio(capture.samples, "capture.samples")
    actual = hashlib.sha256(capture.samples.tobytes(order="C")).hexdigest()
    if (
        actual != capture.provenance["output_pcm_sha256"]
        or capture_replay_provenance_sha256(capture.provenance)
        != capture.provenance_sha256
    ):
        raise ValueError("capture replay PCM hash mismatch")


@dataclass(frozen=True, slots=True)
class OracleSlice:
    """One P1 decoder source: a component envelope plus real context."""

    audio: np.ndarray
    pcm_sha256: str
    span: SampleSpan


def oracle_slice(
    component: OracleComponent,
    capture: CaptureReplayStream,
) -> OracleSlice:
    """Revalidate a predeclared component and cut its slice from the capture stream.

    The slice comes from the same production capture stream that P0 replays, so
    P1 changes only the window boundaries.
    """

    if not isinstance(component, OracleComponent):
        raise TypeError("component must be OracleComponent")
    _validate_oracle_master(component.master)
    _validate_oracle_capture(capture, component.master)
    load_reference_set(component.reference_set_manifest, component.master)
    expected_map = oracle_component_map_sha256(
        component.oracle_component_id,
        component.reference_set_manifest,
        component.master,
        component.event_ids,
    )
    if expected_map != component.component_map_sha256:
        raise ValueError("oracle component map changed before inference")
    start, end = _oracle_slice_bounds(
        component.master.manifest.sample_count,
        component.event_spans,
    )
    slice_audio = _immutable_audio(capture.samples[start:end])
    return OracleSlice(
        audio=slice_audio,
        pcm_sha256=hashlib.sha256(slice_audio.tobytes(order="C")).hexdigest(),
        span=SampleSpan(component.master.manifest.timeline_id, start, end),
    )


def run_p1(
    component: OracleComponent,
    capture: CaptureReplayStream,
    asr: FasterWhisperASR,
    sink: EvidenceSink,
) -> DecoderArmResult:
    """Run the current wrapper on a predeclared event component plus real context."""

    if not isinstance(component, OracleComponent):
        raise TypeError("component must be OracleComponent")
    _arm_sink(sink)
    cut = oracle_slice(component, capture)
    return _run_wrapper(
        source_audio=cut.audio,
        source_spans=(cut.span,),
        source_artifact_id=component.oracle_component_id,
        source_pcm_sha256=cut.pcm_sha256,
        asr=asr,
        sink=sink,
        call_mode=ArmKind.P1,
        oracle_component_id=component.oracle_component_id,
        oracle_reference_set_id=component.reference_set_manifest.reference_set_id,
        oracle_reference_set_sha256=(
            component.reference_set_manifest.reference_set_sha256
        ),
        oracle_event_ids=component.event_ids,
        oracle_component_map_sha256=component.component_map_sha256,
        oracle_slice_pcm_sha256=cut.pcm_sha256,
    )


def _direct_snapshots(
    delegate: object,
    decoder_audio: np.ndarray,
    effective_arguments: Mapping[str, object],
    sequence_kinds: Mapping[str, str],
) -> tuple[_RawSnapshot, ...]:
    _package_version()
    _signature_sha256(delegate)
    kwargs = _thaw_arguments(effective_arguments, sequence_kinds)
    ctranslate2.set_random_seed(contracts.DECODER_RANDOM_SEED)
    segments, _info = delegate.transcribe(decoder_audio, **kwargs)
    return tuple(_snapshot_segment(segment) for segment in segments)


def _parity_mismatch(
    expected_segments: Sequence[RawSegmentEvidence],
    expected_texts: Sequence[str],
    actual_segments: Sequence[RawSegmentEvidence],
    actual_texts: Sequence[str],
) -> str | None:
    """Return the first greedy-segment difference between P0 and P2a, if any.

    Segments from a call's first temperature > 0 segment onward are stochastic
    (see ``stochastic_tail_start``) and exempt; both calls must reach that
    fallback at the same ordinal, or both must stay greedy throughout.
    """

    fields = (
        "start_decoder_sample",
        "end_decoder_sample",
        "raw_start_seconds",
        "raw_end_seconds",
        "timestamp_clamped",
        "no_speech_prob",
        "avg_logprob",
        "compression_ratio",
        "temperature",
    )
    expected_tail = stochastic_tail_start(
        segment.temperature for segment in expected_segments
    )
    actual_tail = stochastic_tail_start(
        segment.temperature for segment in actual_segments
    )
    shared = min(expected_tail, actual_tail)
    for index in range(shared):
        if expected_texts[index] != actual_texts[index]:
            return f"raw[{index}].text"
        for field_name in fields:
            if getattr(expected_segments[index], field_name) != getattr(
                actual_segments[index], field_name
            ):
                return f"raw[{index}].{field_name}"
    expected_stochastic = expected_tail < len(expected_segments)
    actual_stochastic = actual_tail < len(actual_segments)
    if expected_tail != actual_tail or expected_stochastic != actual_stochastic:
        if shared < len(expected_segments) and shared < len(actual_segments):
            return f"raw[{shared}].temperature"
        return "raw_segment_count"
    return None


def _direct_arm(
    *,
    parent_call: DecoderCallEvidence,
    delegate: object,
    sink: EvidenceSink,
    call_mode: ArmKind,
    effective_arguments: Mapping[str, object],
    sequence_kinds: Mapping[str, str],
    changed_argument_keys: tuple[str, ...],
    require_parity: bool,
) -> DecoderArmResult:
    bound_sink = _arm_sink(sink)
    _verify_wrapper_contract()
    if delegate is not parent_call.delegate:
        raise DecoderInstrumentationError(
            "direct arm delegate is not the captured P0 model instance"
        )
    if _model_artifact_sha256(delegate) != parent_call.model_artifact_sha256:
        raise DecoderInstrumentationError("direct arm model artifact hash changed")
    decoder_audio = _immutable_audio(parent_call.decoder_audio)
    before_sha256 = hashlib.sha256(decoder_audio.tobytes(order="C")).hexdigest()
    if before_sha256 != parent_call.decoder_input_pcm_sha256:
        raise DecoderInstrumentationError("direct arm input changed before inference")
    snapshots = _direct_snapshots(
        delegate,
        decoder_audio,
        effective_arguments,
        sequence_kinds,
    )
    after_sha256 = hashlib.sha256(decoder_audio.tobytes(order="C")).hexdigest()
    if after_sha256 != before_sha256:
        raise DecoderInstrumentationError("delegate mutated the direct arm input")
    mapping = parent_call.source_to_decoder_map
    raw_segments, raw_texts = _materialize_raw(
        snapshots,
        mapping,
        decoder_audio.size,
    )
    if require_parity:
        mismatch = _parity_mismatch(
            parent_call.raw_segments,
            parent_call.raw_texts,
            raw_segments,
            raw_texts,
        )
        if mismatch is not None:
            raise DecoderParityError(f"P2a parity mismatch: {mismatch}")
    safe_arguments = {
        name: _safe_argument(value) for name, value in effective_arguments.items()
    }
    contract = _ArgumentContract(
        call_kwargs=MappingProxyType(dict(effective_arguments)),
        effective_arguments=MappingProxyType(dict(effective_arguments)),
        sequence_kinds=MappingProxyType(dict(sequence_kinds)),
        safe_arguments=MappingProxyType(safe_arguments),
        safe_arguments_sha256=canonical_sha256(safe_arguments),
        signature_sha256=_signature_sha256(delegate),
    )
    classification = _classify_wrapper(snapshots, parent_call.no_speech_only_threshold)
    real_samples = sum(
        interval.decoder_end - interval.decoder_start
        for interval in mapping
        if interval.source_span is not None
    )
    return _emit_call_result(
        sink=bound_sink,
        call_mode=call_mode,
        source_artifact_id=parent_call.source_artifact_id,
        source_pcm_sha256=parent_call.source_pcm_sha256,
        decoder_audio=decoder_audio,
        mapping=mapping,
        mapping_id=parent_call.source_to_decoder_map_id,
        mapping_sha256=parent_call.source_to_decoder_map_sha256,
        argument_contract=contract,
        delegate=delegate,
        delegate_instance_id=parent_call.delegate_instance_id,
        snapshots=snapshots,
        raw_segments=raw_segments,
        raw_texts=raw_texts,
        classification=classification,
        no_speech_only_threshold=parent_call.no_speech_only_threshold,
        duration_s=real_samples / _SAMPLE_RATE,
        parent_call_record_id=parent_call.call_record.record_id,
        changed_argument_keys=changed_argument_keys,
        oracle_component_id=parent_call.oracle_component_id,
        oracle_reference_set_id=parent_call.oracle_reference_set_id,
        oracle_reference_set_sha256=parent_call.oracle_reference_set_sha256,
        oracle_event_ids=parent_call.oracle_event_ids,
        oracle_component_map_sha256=parent_call.oracle_component_map_sha256,
        oracle_slice_pcm_sha256=parent_call.oracle_slice_pcm_sha256,
    )


def run_p2a(
    p0_call: DecoderCallEvidence,
    delegate: object,
    sink: EvidenceSink,
) -> DecoderArmResult:
    """Prove a fully explicit direct call matches P0 before any ablation."""

    if (
        not isinstance(p0_call, DecoderCallEvidence)
        or p0_call.call_mode is not ArmKind.P0
    ):
        raise TypeError("p0_call must be captured P0 decoder evidence")
    if delegate is not p0_call.delegate:
        raise DecoderInstrumentationError(
            "P2a must reuse the captured P0 delegate instance"
        )
    resolved = _resolve_arguments(
        delegate,
        p0_call.decoder_audio,
        p0_call.call_kwargs,
    )
    if (
        resolved.effective_arguments != p0_call.effective_arguments
        or resolved.safe_arguments_sha256 != p0_call.effective_arguments_sha256
    ):
        raise DecoderParityError("P2a effective arguments differ from P0")
    return _direct_arm(
        parent_call=p0_call,
        delegate=delegate,
        sink=sink,
        call_mode=ArmKind.P2A,
        effective_arguments=resolved.effective_arguments,
        sequence_kinds=resolved.sequence_kinds,
        changed_argument_keys=(),
        require_parity=True,
    )


def run_p2b(
    p2a_call: DecoderCallEvidence,
    delegate: object,
    sink: EvidenceSink,
) -> DecoderArmResult:
    """Disable only faster-whisper's two no-speech-threshold decisions."""

    if (
        not isinstance(p2a_call, DecoderCallEvidence)
        or p2a_call.call_mode is not ArmKind.P2A
    ):
        raise TypeError("p2a_call must be completed P2a decoder evidence")
    if delegate is not p2a_call.delegate:
        raise DecoderInstrumentationError(
            "P2b must reuse the captured P0 delegate instance"
        )
    effective = copy.deepcopy(dict(p2a_call.effective_arguments))
    if effective.get("no_speech_threshold") != 0.6:
        raise DecoderInstrumentationError(
            "P2b requires the pinned P2a no_speech_threshold of 0.6"
        )
    before = copy.deepcopy(effective)
    effective["no_speech_threshold"] = None
    changed = tuple(key for key in effective if effective[key] != before[key])
    if changed != ("no_speech_threshold",):
        raise DecoderInstrumentationError("P2b changed more than no_speech_threshold")
    return _direct_arm(
        parent_call=p2a_call,
        delegate=delegate,
        sink=sink,
        call_mode=ArmKind.P2B,
        effective_arguments=MappingProxyType(effective),
        sequence_kinds=p2a_call.effective_argument_sequence_kinds,
        changed_argument_keys=changed,
        require_parity=False,
    )
