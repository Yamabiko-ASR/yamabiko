"""Immutable types and closed vocabularies for E0 evidence wire version 1.0."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias, TypeVar, Union

from scripts.e0_evidence._common import require_sha256, require_uuid4


SCHEMA_VERSION = "1.0"
MONOTONIC_CLOCK_ID = "perf_counter_ns"

JsonScalar: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[
    JsonScalar,
    Mapping[str, "JsonValue"],
    tuple["JsonValue", ...],
]


class ArmKind(str, Enum):
    P0 = "p0"
    P1 = "p1"
    P2A = "p2a"
    P2B = "p2b"


class DecoderAdapter(str, Enum):
    """The decoder a run benchmarks: production's or a non-Whisper candidate."""

    FASTER_WHISPER = "faster_whisper"
    ONNX_ASR_NEMO_TDT = "onnx_asr_nemo_tdt"


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    NOT_RUN = "not_run"
    OUT_OF_SCOPE = "out_of_scope"
    INSTRUMENTATION_MISSING = "instrumentation_missing"
    RECORD_LOST = "record_lost"


class Stage(str, Enum):
    RUN = "run"
    MASTER = "master"
    REFERENCE = "reference"
    CAPTURE = "capture"
    VAD = "vad"
    ENDPOINT = "endpoint"
    DECODER = "decoder"
    WRAPPER = "wrapper"
    DEDUP = "dedup"
    LANGUAGE_POLICY = "language_policy"
    SENTENCE = "sentence"
    TAIL = "tail"
    TRANSLATION = "translation"
    BROADCAST = "broadcast"
    RENDER = "render"
    SCORING = "scoring"
    INTEGRITY = "integrity"
    LEDGER = "ledger"


class RecordKind(str, Enum):
    RUN_STATUS = "run_status"
    ARM_STATUS = "arm_status"
    FRAME = "frame"
    GATE_TRANSITION = "gate_transition"
    WINDOW = "window"
    DECODER_CALL = "decoder_call"
    RAW_HYPOTHESIS = "raw_hypothesis"
    WRAPPER_DECISION = "wrapper_decision"
    DIAGNOSTIC = "diagnostic"
    EVENT_FATE = "event_fate"
    MEASUREMENT = "measurement"
    INTEGRITY = "integrity"
    DISCONTINUITY = "discontinuity"


class EndpointEmitReason(str, Enum):
    SILENCE_CLOSE = "endpoint_silence_close"
    GLUE_TIMEOUT = "endpoint_glue_timeout"
    FORCE_SPLIT = "endpoint_force_split"
    DIP_CUT = "endpoint_dip_cut"
    UNKNOWN = "unknown_endpoint_reason"


class DiagnosticCode(str, Enum):
    VAD_NOT_DETECTED = "vad_not_detected"
    ENDPOINT_PENDING_EOF = "endpoint_pending_eof"
    ENDPOINT_PARTIAL_FRAME_EOF = "endpoint_partial_frame_eof"
    ENDPOINT_SILENCE_CLOSE = "endpoint_silence_close"
    ENDPOINT_GLUE_TIMEOUT = "endpoint_glue_timeout"
    ENDPOINT_FORCE_SPLIT = "endpoint_force_split"
    ENDPOINT_DIP_CUT = "endpoint_dip_cut"
    ENDPOINT_BOUNDARY_TRUNCATED = "endpoint_boundary_truncated"
    UNKNOWN_ENDPOINT_REASON = "unknown_endpoint_reason"
    DECODER_NO_YIELD = "decoder_no_yield"
    UNKNOWN_DECODER_GATE = "unknown_decoder_gate"
    DECODER_TIMESTAMP_CLAMPED = "decoder_timestamp_clamped"
    DECODER_STOCHASTIC_FALLBACK = "decoder_stochastic_fallback"
    WRAPPER_CONFIDENCE_GATE = "wrapper_confidence_gate"
    WRAPPER_COMPRESSION_REJECT = "wrapper_compression_reject"
    WRAPPER_EMPTY = "wrapper_empty"
    DEDUP_EMPTY = "dedup_empty"
    LANGUAGE_POLICY = "language_policy"
    SENTENCE_EMPTY = "sentence_empty"
    TAIL_BUFFERED = "tail_buffered"
    TAIL_EXPIRED = "tail_expired"
    DISPLAY_POLICY_INTERJECTION = "display_policy_interjection"
    TRANSLATION_ERROR = "translation_error"
    TRANSLATION_EMPTY = "translation_empty"
    BROADCAST_ERROR = "broadcast_error"
    RENDER_UNOBSERVED = "render_unobserved"
    CAPTURE_DISCONTINUITY = "capture_discontinuity"
    CALLBACK_OVERFLOW = "callback_overflow"
    PAUSED_DISCARD = "paused_discard"
    ENDPOINT_PENDING_SHUTDOWN = "endpoint_pending_shutdown"
    ASR_QUEUE_TIMEOUT = "asr_queue_timeout"
    SHUTDOWN_CANCELLED = "shutdown_cancelled"
    LEDGER_OVERFLOW = "ledger_overflow"
    # Arm terminal reasons for an exception that stopped the arm.
    DECODER_PARITY_MISMATCH = "decoder_parity_mismatch"
    DECODER_INSTRUMENTATION = "decoder_instrumentation"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    HARNESS_VALIDATION = "harness_validation"


class TerminalOutcome(str, Enum):
    WRAPPER_HYPOTHESIS_AVAILABLE = "wrapper_hypothesis_available"
    NO_DECODER_WINDOW = "no_decoder_window"
    DECODER_NO_YIELD = "decoder_no_yield"
    DECODER_EVENT_UNREPRESENTED = "decoder_event_unrepresented"
    WRAPPER_REJECTED = "wrapper_rejected"
    WRAPPER_EMPTY = "wrapper_empty"
    REFERENCE_INVALID = "reference_invalid"
    UNKNOWN = "unknown"
    PRODUCT_JAPANESE_AVAILABLE = "product_japanese_available"
    RENDERED = "rendered"
    CAPTURE_LOSS = "capture_loss"
    POLICY_REJECTED = "policy_rejected"
    DELIVERY_FAILED = "delivery_failed"
    RECORD_LOST = "record_lost"


class ArmStatus(str, Enum):
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_RUN = "not_run"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReplayStopKind(str, Enum):
    ORDINARY_EOF = "ordinary_eof"
    CANCELLED = "cancelled"


class TextRole(str, Enum):
    VERBATIM_REFERENCE = "verbatim_reference"
    EDITORIAL_SUBTITLE = "editorial_subtitle"
    RAW_HYPOTHESIS = "raw_hypothesis"
    WRAPPER_HYPOTHESIS = "wrapper_hypothesis"
    TRANSLATION = "translation"


class LossCode(str, Enum):
    LEDGER_OVERFLOW = "ledger_overflow"
    RECORD_LOST = "record_lost"


class StreamName(str, Enum):
    CANONICAL = "canonical"
    PRIVATE_TEXT = "private_text"


class IntegrityFailureCode(str, Enum):
    SEQUENCE_GAP = "sequence_gap"
    STREAM_HASH_MISMATCH = "stream_hash_mismatch"
    INVALID_OR_TRUNCATED_JSONL = "invalid_or_truncated_jsonl"
    FINAL_MANIFEST_MISSING = "final_manifest_missing"
    LOSS_REPORTED = "loss_reported"
    ARM_TERMINAL_STATUS_MISSING = "arm_terminal_status_missing"
    START_MANIFEST_MISSING = "start_manifest_missing"
    START_MANIFEST_INVALID = "start_manifest_invalid"
    FINAL_MANIFEST_INVALID = "final_manifest_invalid"
    STREAM_MISSING = "stream_missing"
    STREAM_SUMMARY_MISMATCH = "stream_summary_mismatch"
    RECORD_ID_DUPLICATE = "record_id_duplicate"
    RECORD_SCOPE_MISMATCH = "record_scope_mismatch"
    UNKNOWN_ARM = "unknown_arm"
    PARENT_LINK_INVALID = "parent_link_invalid"
    CLASSIFICATION_MISMATCH = "classification_mismatch"
    CANONICAL_PAYLOAD_INVALID = "canonical_payload_invalid"
    PRIVATE_TEXT_LINK_MISMATCH = "private_text_link_mismatch"
    ARM_STATUS_MISMATCH = "arm_status_mismatch"
    FINAL_STATUS_MISMATCH = "final_status_mismatch"
    STREAM_WRITE_MISMATCH = "stream_write_mismatch"


class PrivateContentClass(str, Enum):
    PUBLIC_ALLOWLISTED = "public_allowlisted"
    INTERNAL_AGGREGATE = "internal_aggregate"
    PRIVATE_METADATA = "private_metadata"
    PRIVATE_TEXT = "private_text"
    PRIVATE_AUDIO = "private_audio"


class CapabilityDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class Partition(str, Enum):
    HISTORICAL_INCIDENT = "historical_incident"
    BLOCKER_CLASS = "blocker_class"
    DEVELOPMENT = "development"
    LOCKED_CONFIRMATION = "locked_confirmation"


_EnumT = TypeVar("_EnumT", bound=Enum)


def new_opaque_id() -> str:
    """Return a random, content-independent UUIDv4 identifier."""

    return str(uuid.uuid4())


def _require_nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _require_int(
    value: object,
    field_name: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer, not bool")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _require_enum(
    value: object,
    enum_type: type[_EnumT],
    field_name: str,
) -> _EnumT:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}")
    return value


def _freeze_json(value: object, path: str) -> JsonValue:
    if value is None or type(value) in (bool, int, float, str):
        return value  # type: ignore[return-value]
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(child, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def freeze_json_mapping(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _freeze_json(value, field_name)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise TypeError(f"{field_name} must be a mapping")
    return frozen


def _freeze_strings(
    values: Iterable[object],
    field_name: str,
    *,
    opaque_ids: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings")
    result = tuple(values)
    validator = require_uuid4 if opaque_ids else _require_nonempty_string
    for index, value in enumerate(result):
        validator(value, f"{field_name}[{index}]")
    return result  # type: ignore[return-value]


def _freeze_spans(
    values: Iterable[object],
    field_name: str,
) -> tuple["SampleSpan", ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of SampleSpan")
    spans = tuple(values)
    for index, span in enumerate(spans):
        if not isinstance(span, SampleSpan):
            raise TypeError(f"{field_name}[{index}] must be SampleSpan")
    return spans  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RightsRecord:
    rights_record_id: str
    fixture_id: str
    local_evaluation: CapabilityDecision
    redistribution: CapabilityDecision
    training: CapabilityDecision
    transcript_publication: CapabilityDecision
    reviewer_id: str
    evidence_ids: tuple[str, ...]
    retention_version: str

    def __post_init__(self) -> None:
        require_uuid4(self.rights_record_id, "rights_record_id")
        require_uuid4(self.fixture_id, "fixture_id")
        decisions = (
            self.local_evaluation,
            self.redistribution,
            self.training,
            self.transcript_publication,
        )
        for name, decision in zip(
            (
                "local_evaluation",
                "redistribution",
                "training",
                "transcript_publication",
            ),
            decisions,
        ):
            _require_enum(decision, CapabilityDecision, name)
        if self.reviewer_id:
            require_uuid4(self.reviewer_id, "reviewer_id")
        evidence_ids = _freeze_strings(
            self.evidence_ids,
            "evidence_ids",
            opaque_ids=True,
        )
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_ids must not contain duplicates")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        _require_nonempty_string(self.retention_version, "retention_version")
        if any(
            decision is not CapabilityDecision.PENDING
            for decision in decisions
        ):
            if not self.reviewer_id or not evidence_ids:
                raise ValueError(
                    "reviewed capability decisions require reviewer and evidence"
                )


@dataclass(frozen=True, slots=True)
class FixtureManifest:
    fixture_id: str
    partition: Partition
    title_id: str
    rights_record_id: str

    def __post_init__(self) -> None:
        require_uuid4(self.fixture_id, "fixture_id")
        _require_enum(self.partition, Partition, "partition")
        require_uuid4(self.title_id, "title_id")
        require_uuid4(self.rights_record_id, "rights_record_id")


@dataclass(frozen=True, slots=True)
class SampleSpan:
    timeline_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        require_uuid4(self.timeline_id, "timeline_id")
        _require_int(self.start, "start")
        _require_int(self.end, "end")
        if self.end <= self.start:
            raise ValueError("sample spans must be non-empty half-open ranges")

    @property
    def sample_count(self) -> int:
        return self.end - self.start


def validate_sample_span(
    span: SampleSpan,
    timeline_sample_counts: Mapping[str, int],
) -> SampleSpan:
    """Validate a span against the named immutable master timeline."""

    if not isinstance(span, SampleSpan):
        raise TypeError("span must be SampleSpan")
    if not isinstance(timeline_sample_counts, Mapping):
        raise TypeError("timeline_sample_counts must be a mapping")
    if span.timeline_id not in timeline_sample_counts:
        raise ValueError("span references an unknown timeline_id")
    sample_count = _require_int(
        timeline_sample_counts[span.timeline_id],
        f"timeline_sample_counts[{span.timeline_id!r}]",
    )
    if span.end > sample_count:
        raise ValueError("span end exceeds the master timeline")
    return span


@dataclass(frozen=True, slots=True)
class ArmDefinition:
    arm_id: str
    kind: ArmKind
    parent_arm_id: str | None
    parameters: Mapping[str, JsonValue]
    declared_changes: tuple[str, ...]

    def __post_init__(self) -> None:
        require_uuid4(self.arm_id, "arm_id")
        _require_enum(self.kind, ArmKind, "kind")
        if self.parent_arm_id is not None:
            require_uuid4(self.parent_arm_id, "parent_arm_id")
        object.__setattr__(
            self,
            "parameters",
            freeze_json_mapping(self.parameters, field_name="parameters"),
        )
        object.__setattr__(
            self,
            "declared_changes",
            _freeze_strings(self.declared_changes, "declared_changes"),
        )


@dataclass(frozen=True, slots=True)
class RecordEnvelope:
    schema_version: str
    record_id: str
    run_id: str
    arm_id: str | None
    session_id: str
    seq: int
    observed_monotonic_ns: int
    clock_id: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        require_uuid4(self.record_id, "record_id")
        require_uuid4(self.run_id, "run_id")
        if self.arm_id is not None:
            require_uuid4(self.arm_id, "arm_id")
        require_uuid4(self.session_id, "session_id")
        _require_int(self.seq, "seq")
        _require_int(self.observed_monotonic_ns, "observed_monotonic_ns")
        if self.clock_id != MONOTONIC_CLOCK_ID:
            raise ValueError(f"clock_id must be {MONOTONIC_CLOCK_ID!r}")


def _validate_record_payload(record: object) -> None:
    _require_enum(record.stage, Stage, "stage")
    _require_enum(record.kind, RecordKind, "kind")
    require_uuid4(record.artifact_id, "artifact_id")
    parent_ids = _freeze_strings(
        record.parent_ids,
        "parent_ids",
        opaque_ids=True,
    )
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("parent_ids must not contain duplicates")
    object.__setattr__(
        record,
        "parent_ids",
        parent_ids,
    )
    object.__setattr__(
        record,
        "source_spans",
        _freeze_spans(record.source_spans, "source_spans"),
    )
    _require_enum(
        record.observation_status,
        ObservationStatus,
        "observation_status",
    )
    if record.decision is not None:
        _require_nonempty_string(record.decision, "decision")
    diagnostics = tuple(record.diagnostics)
    for index, diagnostic in enumerate(diagnostics):
        _require_enum(diagnostic, DiagnosticCode, f"diagnostics[{index}]")
    if len(diagnostics) != len(set(diagnostics)):
        raise ValueError("diagnostics must not contain duplicates")
    object.__setattr__(record, "diagnostics", diagnostics)
    _require_enum(record.classification, PrivateContentClass, "classification")
    object.__setattr__(
        record,
        "details",
        freeze_json_mapping(record.details, field_name="details"),
    )


@dataclass(frozen=True, slots=True)
class EvidenceRecordDraft:
    arm_id: str | None
    stage: Stage
    kind: RecordKind
    artifact_id: str
    parent_ids: tuple[str, ...] = ()
    source_spans: tuple[SampleSpan, ...] = ()
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    decision: str | None = None
    diagnostics: tuple[DiagnosticCode, ...] = ()
    classification: PrivateContentClass = PrivateContentClass.PRIVATE_METADATA
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm_id is not None:
            require_uuid4(self.arm_id, "arm_id")
        _validate_record_payload(self)


@dataclass(frozen=True, slots=True)
class EvidenceRecord(RecordEnvelope):
    stage: Stage
    kind: RecordKind
    artifact_id: str
    parent_ids: tuple[str, ...] = ()
    source_spans: tuple[SampleSpan, ...] = ()
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    decision: str | None = None
    diagnostics: tuple[DiagnosticCode, ...] = ()
    classification: PrivateContentClass = PrivateContentClass.PRIVATE_METADATA
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RecordEnvelope.__post_init__(self)
        _validate_record_payload(self)


@dataclass(frozen=True, slots=True)
class WindowEvidence:
    window_id: str
    source_spans: tuple[SampleSpan, ...]
    pcm_sha256: str
    emit_reason: EndpointEmitReason
    decoder_input_samples: int
    decoder_input_pcm_sha256: str
    synthetic_left_pad_samples: int
    synthetic_right_pad_samples: int
    source_to_decoder_map_id: str

    def __post_init__(self) -> None:
        require_uuid4(self.window_id, "window_id")
        spans = _freeze_spans(self.source_spans, "source_spans")
        if not spans:
            raise ValueError("source_spans must not be empty")
        object.__setattr__(self, "source_spans", spans)
        require_sha256(self.pcm_sha256, "pcm_sha256")
        _require_enum(self.emit_reason, EndpointEmitReason, "emit_reason")
        _require_int(self.decoder_input_samples, "decoder_input_samples", minimum=1)
        require_sha256(
            self.decoder_input_pcm_sha256,
            "decoder_input_pcm_sha256",
        )
        _require_int(self.synthetic_left_pad_samples, "synthetic_left_pad_samples")
        _require_int(self.synthetic_right_pad_samples, "synthetic_right_pad_samples")
        real_samples = sum(span.sample_count for span in spans)
        if (
            real_samples
            + self.synthetic_left_pad_samples
            + self.synthetic_right_pad_samples
            != self.decoder_input_samples
        ):
            raise ValueError(
                "decoder_input_samples does not match real and synthetic samples"
            )
        require_uuid4(
            self.source_to_decoder_map_id,
            "source_to_decoder_map_id",
        )


@dataclass(frozen=True, slots=True)
class EventFate:
    event_id: str
    arm_id: str
    outcome: TerminalOutcome
    first_missing_stage: Stage | None
    reason: str
    evidence_record_ids: tuple[str, ...]
    diagnostic_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_uuid4(self.event_id, "event_id")
        require_uuid4(self.arm_id, "arm_id")
        _require_enum(self.outcome, TerminalOutcome, "outcome")
        if self.first_missing_stage is not None:
            _require_enum(
                self.first_missing_stage,
                Stage,
                "first_missing_stage",
            )
        _require_nonempty_string(self.reason, "reason")
        object.__setattr__(
            self,
            "evidence_record_ids",
            _freeze_strings(
                self.evidence_record_ids,
                "evidence_record_ids",
                opaque_ids=True,
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_record_ids",
            _freeze_strings(
                self.diagnostic_record_ids,
                "diagnostic_record_ids",
                opaque_ids=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class RunStartManifest:
    schema_version: str
    run_id: str
    session_id: str
    clock_id: str
    canonical_stream_id: str
    private_text_stream_id: str | None
    arm_definitions: tuple[ArmDefinition, ...]
    classification: PrivateContentClass
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        require_uuid4(self.run_id, "run_id")
        require_uuid4(self.session_id, "session_id")
        if self.clock_id != MONOTONIC_CLOCK_ID:
            raise ValueError(f"clock_id must be {MONOTONIC_CLOCK_ID!r}")
        require_uuid4(self.canonical_stream_id, "canonical_stream_id")
        if self.private_text_stream_id is not None:
            require_uuid4(
                self.private_text_stream_id,
                "private_text_stream_id",
            )
            if self.private_text_stream_id == self.canonical_stream_id:
                raise ValueError("stream IDs must be distinct")
        definitions = tuple(self.arm_definitions)
        if not definitions:
            raise ValueError("arm_definitions must not be empty")
        for index, definition in enumerate(definitions):
            if not isinstance(definition, ArmDefinition):
                raise TypeError(
                    f"arm_definitions[{index}] must be ArmDefinition"
                )
        arm_ids = tuple(definition.arm_id for definition in definitions)
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("arm definition IDs must be unique")
        for definition in definitions:
            if (
                definition.parent_arm_id is not None
                and definition.parent_arm_id not in arm_ids
            ):
                raise ValueError("parent arm is not part of this run")
        reserved_ids = {
            self.run_id,
            self.session_id,
            self.canonical_stream_id,
            *(arm_ids),
        }
        if self.private_text_stream_id is not None:
            reserved_ids.add(self.private_text_stream_id)
        expected_count = 3 + len(arm_ids) + (
            self.private_text_stream_id is not None
        )
        if len(reserved_ids) != expected_count:
            raise ValueError("run, session, stream, and arm IDs must be distinct")
        object.__setattr__(self, "arm_definitions", definitions)
        _require_enum(self.classification, PrivateContentClass, "classification")
        object.__setattr__(
            self,
            "details",
            freeze_json_mapping(self.details, field_name="details"),
        )

    @property
    def requested_arm_ids(self) -> tuple[str, ...]:
        return tuple(definition.arm_id for definition in self.arm_definitions)


@dataclass(frozen=True, slots=True)
class PrivateTextRecord:
    schema_version: str
    private_record_id: str
    run_id: str
    session_id: str
    stream_id: str
    stream_seq: int
    canonical_record_id: str
    arm_id: str
    text_id: str
    role: TextRole
    classification: PrivateContentClass
    text: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        for field_name, value in (
            ("private_record_id", self.private_record_id),
            ("run_id", self.run_id),
            ("session_id", self.session_id),
            ("stream_id", self.stream_id),
            ("canonical_record_id", self.canonical_record_id),
            ("arm_id", self.arm_id),
            ("text_id", self.text_id),
        ):
            require_uuid4(value, field_name)
        _require_int(self.stream_seq, "stream_seq")
        _require_enum(self.role, TextRole, "role")
        _require_enum(self.classification, PrivateContentClass, "classification")
        _require_nonempty_string(self.text, "text")


@dataclass(frozen=True, slots=True)
class StreamSummary:
    name: StreamName
    stream_id: str
    record_count: int
    first_seq: int | None
    last_seq: int | None
    byte_sha256: str
    classification: PrivateContentClass

    def __post_init__(self) -> None:
        _require_enum(self.name, StreamName, "name")
        require_uuid4(self.stream_id, "stream_id")
        count = _require_int(self.record_count, "record_count")
        if count == 0:
            if self.first_seq is not None or self.last_seq is not None:
                raise ValueError("empty stream sequence bounds must be null")
        else:
            if self.first_seq is None or self.last_seq is None:
                raise ValueError("nonempty stream sequence bounds are required")
            _require_int(self.first_seq, "first_seq")
            _require_int(self.last_seq, "last_seq")
            if self.last_seq < self.first_seq:
                raise ValueError("stream sequence bounds are reversed")
        require_sha256(self.byte_sha256, "byte_sha256")
        _require_enum(self.classification, PrivateContentClass, "classification")


@dataclass(frozen=True, slots=True)
class ArmTerminalStatus:
    arm_id: str
    status: ArmStatus
    terminal_record_id: str
    reason: str | None = None

    def __post_init__(self) -> None:
        require_uuid4(self.arm_id, "arm_id")
        _require_enum(self.status, ArmStatus, "status")
        if self.status in {ArmStatus.REQUESTED, ArmStatus.RUNNING}:
            raise ValueError("arm terminal status must be terminal")
        require_uuid4(self.terminal_record_id, "terminal_record_id")
        if self.status is ArmStatus.COMPLETE:
            if self.reason is not None:
                raise ValueError("complete arm status must not have a reason")
        else:
            _require_nonempty_string(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class RunFinalManifest:
    schema_version: str
    run_id: str
    session_id: str
    run_start_sha256: str
    status: RunStatus
    streams: tuple[StreamSummary, ...]
    arm_statuses: tuple[ArmTerminalStatus, ...]
    loss_counts: Mapping[LossCode, int]
    incomplete_reasons: tuple[IntegrityFailureCode, ...]
    classification: PrivateContentClass
    finalized_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        require_uuid4(self.run_id, "run_id")
        require_uuid4(self.session_id, "session_id")
        require_sha256(self.run_start_sha256, "run_start_sha256")
        _require_enum(self.status, RunStatus, "status")
        if self.status is RunStatus.RUNNING:
            raise ValueError("final run status must be terminal")
        streams = tuple(self.streams)
        for index, summary in enumerate(streams):
            if not isinstance(summary, StreamSummary):
                raise TypeError(f"streams[{index}] must be StreamSummary")
        if StreamName.CANONICAL not in {item.name for item in streams}:
            raise ValueError("final manifest must include the canonical stream")
        if len(streams) != len({item.name for item in streams}):
            raise ValueError("stream names must be unique")
        if len(streams) != len({item.stream_id for item in streams}):
            raise ValueError("stream IDs must be unique")
        object.__setattr__(self, "streams", streams)
        arm_statuses = tuple(self.arm_statuses)
        for index, arm_status in enumerate(arm_statuses):
            if not isinstance(arm_status, ArmTerminalStatus):
                raise TypeError(
                    f"arm_statuses[{index}] must be ArmTerminalStatus"
                )
        if len(arm_statuses) != len({item.arm_id for item in arm_statuses}):
            raise ValueError("arm terminal statuses must be unique")
        if len(arm_statuses) != len(
            {item.terminal_record_id for item in arm_statuses}
        ):
            raise ValueError("arm terminal record IDs must be unique")
        object.__setattr__(self, "arm_statuses", arm_statuses)
        if not isinstance(self.loss_counts, Mapping):
            raise TypeError("loss_counts must be a mapping")
        if set(self.loss_counts) != set(LossCode):
            raise ValueError("loss_counts must contain every LossCode")
        losses: dict[LossCode, int] = {}
        for code, count in self.loss_counts.items():
            _require_enum(code, LossCode, "loss_counts key")
            losses[code] = _require_int(count, f"loss_counts[{code.value}]")
        object.__setattr__(self, "loss_counts", MappingProxyType(losses))
        reasons = tuple(self.incomplete_reasons)
        for index, reason in enumerate(reasons):
            _require_enum(
                reason,
                IntegrityFailureCode,
                f"incomplete_reasons[{index}]",
            )
        if len(reasons) != len(set(reasons)):
            raise ValueError("incomplete_reasons must not contain duplicates")
        if self.status is RunStatus.COMPLETE and reasons:
            raise ValueError("complete run must not have incomplete reasons")
        if self.status is not RunStatus.COMPLETE and not reasons:
            raise ValueError("incomplete run must name an incomplete reason")
        object.__setattr__(self, "incomplete_reasons", reasons)
        _require_enum(self.classification, PrivateContentClass, "classification")
        _require_int(self.finalized_monotonic_ns, "finalized_monotonic_ns")


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    run_id: str | None
    complete: bool
    declared_status: RunStatus | None
    computed_streams: tuple[StreamSummary, ...]
    computed_arm_statuses: tuple[ArmTerminalStatus, ...]
    reasons: tuple[IntegrityFailureCode, ...]

    def __post_init__(self) -> None:
        if self.run_id is not None:
            require_uuid4(self.run_id, "run_id")
        if type(self.complete) is not bool:
            raise TypeError("complete must be bool")
        if self.declared_status is not None:
            _require_enum(self.declared_status, RunStatus, "declared_status")
        streams = tuple(self.computed_streams)
        for index, summary in enumerate(streams):
            if not isinstance(summary, StreamSummary):
                raise TypeError(
                    f"computed_streams[{index}] must be StreamSummary"
                )
        object.__setattr__(self, "computed_streams", streams)
        statuses = tuple(self.computed_arm_statuses)
        for index, arm_status in enumerate(statuses):
            if not isinstance(arm_status, ArmTerminalStatus):
                raise TypeError(
                    "computed_arm_statuses"
                    f"[{index}] must be ArmTerminalStatus"
                )
        object.__setattr__(self, "computed_arm_statuses", statuses)
        reasons = tuple(self.reasons)
        for index, reason in enumerate(reasons):
            _require_enum(
                reason,
                IntegrityFailureCode,
                f"reasons[{index}]",
            )
        if len(reasons) != len(set(reasons)):
            raise ValueError("reasons must not contain duplicates")
        if self.complete and (
            reasons or self.declared_status is not RunStatus.COMPLETE
        ):
            raise ValueError("complete report must have a clean complete status")
        object.__setattr__(self, "reasons", reasons)
