"""Deterministic event fates and aggregate scoring for the E0 ASR harness."""

from __future__ import annotations

import hashlib
import math
import random
import unicodedata
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

from ._common import require_sha256, require_uuid4
from .align import (
    AlignmentPolicy,
    AlignmentStatus,
    ComponentAlignment,
    EditKind,
    EventAlignment,
    JA_TIME_DP_1_0,
    ScoredEvent,
    ScoredHypothesis,
)
from .ledger import EvidenceSink
from .normalize_ja import JA_LENIENT_1_1, JA_MINIMAL_1_1, JA_WORDS_1_0
from .references import ReferenceEvent, ReferenceSet
from .schema import (
    ArmKind,
    ArmStatus,
    DiagnosticCode,
    EvidenceRecord,
    EvidenceRecordDraft,
    EventFate,
    FixtureManifest,
    ObservationStatus,
    Partition,
    PrivateContentClass,
    RecordKind,
    SampleSpan,
    Stage,
    TerminalOutcome,
)
from .wire import canonical_sha256


SCORE_SCHEMA_VERSION = "e0-score-3.0"
# Repository-relative sources whose bytes define every scored value: the
# normalizer, the aligner, the metric code, and the ledger-to-score join.
_SCORER_SOURCE_PATHS = (
    "scripts/e0_evidence/align.py",
    "scripts/e0_evidence/normalize_ja.py",
    "scripts/e0_evidence/score.py",
    "scripts/e0_evidence/score_runs.py",
)


def _scorer_source_sha256() -> str:
    # Line endings are folded to LF so the identity follows the committed
    # content rather than a checkout's line-ending conversion.
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in sorted(_SCORER_SOURCE_PATHS):
        payload = (package / relative.rsplit("/", 1)[1]).read_bytes()
        digest.update(relative.encode("ascii") + b"\0")
        digest.update(payload.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


SCORER_SOURCE_SHA256 = _scorer_source_sha256()
RAW_UNICODE_1_0_VERSION = "raw-unicode-1.0"
RAW_UNICODE_1_0_SHA256 = canonical_sha256(
    {
        "semantics": ("identity_unicode_code_points", "symmetric"),
        "transforms": (),
        "version": RAW_UNICODE_1_0_VERSION,
    }
)


def _nonempty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _plain_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer of at least {minimum}")
    return value


def _freeze_ids(
    values: Iterable[str],
    field_name: str,
    *,
    unique: bool = True,
) -> tuple[str, ...]:
    result = tuple(values)
    for value in result:
        require_uuid4(value, f"{field_name} item")
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _freeze_spans(
    values: Iterable[SampleSpan],
    field_name: str,
) -> tuple[SampleSpan, ...]:
    result = tuple(values)
    if any(not isinstance(span, SampleSpan) for span in result):
        raise TypeError(f"{field_name} must contain SampleSpan values")
    return tuple(
        sorted(result, key=lambda span: (span.timeline_id, span.start, span.end))
    )


def _optional_bool(value: object, field_name: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise TypeError(f"{field_name} must be bool or None")
    return value


_USABILITY_SEMANTICS = (
    "assigned_hypothesis_required",
    "resolved_event_alignment_required",
    "empty_normalized_reference_is_unusable",
    "short_reference_requires_exact_recovery",
    "long_reference_cer_is_inclusive",
    "long_reference_requires_nonpunctuation_match",
)


@dataclass(frozen=True, slots=True)
class UsabilityPolicy:
    version: str
    minimal_policy_version: str
    minimal_policy_sha256: str
    short_reference_max_characters: int
    long_cer_numerator: int
    long_cer_denominator: int
    recovered_category_prefixes: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.version, "version")
        _nonempty(self.minimal_policy_version, "minimal_policy_version")
        require_sha256(self.minimal_policy_sha256, "minimal_policy_sha256")
        _plain_int(
            self.short_reference_max_characters,
            "short_reference_max_characters",
            minimum=1,
        )
        numerator = _plain_int(
            self.long_cer_numerator,
            "long_cer_numerator",
            minimum=1,
        )
        denominator = _plain_int(
            self.long_cer_denominator,
            "long_cer_denominator",
            minimum=1,
        )
        if numerator > denominator:
            raise ValueError("long CER threshold must not exceed one")
        prefixes = tuple(self.recovered_category_prefixes)
        if not prefixes or any(type(prefix) is not str or len(prefix) != 1 for prefix in prefixes):
            raise ValueError("recovered_category_prefixes must be one-letter prefixes")
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("recovered_category_prefixes must be unique")
        object.__setattr__(self, "recovered_category_prefixes", prefixes)
        require_sha256(self.sha256, "sha256")
        expected = canonical_sha256(
            {
                "long_cer_denominator": denominator,
                "long_cer_numerator": numerator,
                "minimal_policy_sha256": self.minimal_policy_sha256,
                "minimal_policy_version": self.minimal_policy_version,
                "recovered_category_prefixes": prefixes,
                "semantics": _USABILITY_SEMANTICS,
                "short_reference_max_characters": (
                    self.short_reference_max_characters
                ),
                "version": self.version,
            }
        )
        if self.sha256 != expected:
            raise ValueError("sha256 does not match the usability policy")


def _make_usability_policy() -> UsabilityPolicy:
    payload = {
        "long_cer_denominator": 4,
        "long_cer_numerator": 1,
        "minimal_policy_sha256": JA_MINIMAL_1_1.sha256,
        "minimal_policy_version": JA_MINIMAL_1_1.version,
        "recovered_category_prefixes": ("L", "M", "N"),
        "semantics": _USABILITY_SEMANTICS,
        "short_reference_max_characters": 4,
        "version": "ja-event-usable-1.1",
    }
    return UsabilityPolicy(
        version=payload["version"],
        minimal_policy_version=payload["minimal_policy_version"],
        minimal_policy_sha256=payload["minimal_policy_sha256"],
        short_reference_max_characters=payload[
            "short_reference_max_characters"
        ],
        long_cer_numerator=payload["long_cer_numerator"],
        long_cer_denominator=payload["long_cer_denominator"],
        recovered_category_prefixes=payload["recovered_category_prefixes"],
        sha256=canonical_sha256(payload),
    )


JA_EVENT_USABLE_1_1 = _make_usability_policy()
USABILITY_CURVE_POLICY_VERSION = "empirical-cer-breakpoints-1.0"
USABILITY_CURVE_POLICY_SHA256 = canonical_sha256(
    {
        "primary_usability_policy_sha256": JA_EVENT_USABLE_1_1.sha256,
        "semantics": (
            "include_zero_and_primary_threshold",
            "include_every_observed_long_event_cer_breakpoint",
            "short_event_exactness_does_not_vary",
            "unknowns_remain_in_denominator",
        ),
        "version": USABILITY_CURVE_POLICY_VERSION,
    }
)


class MetricStatus(str, Enum):
    DEFINED = "defined"
    ZERO_DENOMINATOR = "zero_denominator"
    INCOMPLETE = "incomplete"
    INSUFFICIENT_UNITS = "insufficient_units"


class NormalizationLevel(str, Enum):
    RAW = "raw"
    MINIMAL = "minimal"
    LENIENT = "lenient"
    WORDS = "words"


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    """Evidence used immediately before emitting an arm's terminal status."""

    arm_id: str
    kind: ArmKind
    status: ArmStatus
    records: tuple[EvidenceRecord, ...]
    scored_events: tuple[ScoredEvent, ...]
    raw_hypotheses: tuple[ScoredHypothesis, ...]
    reference_invalid_event_ids: tuple[str, ...]
    alignment_policy: AlignmentPolicy = JA_TIME_DP_1_0

    def __post_init__(self) -> None:
        require_uuid4(self.arm_id, "arm_id")
        if not isinstance(self.kind, ArmKind):
            raise TypeError("kind must be ArmKind")
        if not isinstance(self.status, ArmStatus):
            raise TypeError("status must be ArmStatus")
        if self.status in {ArmStatus.REQUESTED, ArmStatus.RUNNING}:
            raise ValueError("status must be the arm's proposed terminal status")
        records = tuple(self.records)
        if any(not isinstance(record, EvidenceRecord) for record in records):
            raise TypeError("records must contain EvidenceRecord values")
        record_ids = tuple(record.record_id for record in records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("records must have globally unique record IDs")
        if any(
            record.arm_id == self.arm_id and record.kind is RecordKind.ARM_STATUS
            for record in records
        ):
            raise ValueError(
                "derive event fates before writing the terminal arm status"
            )
        scored_events = tuple(self.scored_events)
        raw_hypotheses = tuple(self.raw_hypotheses)
        if any(not isinstance(event, ScoredEvent) for event in scored_events):
            raise TypeError("scored_events must contain ScoredEvent values")
        if any(
            not isinstance(hypothesis, ScoredHypothesis)
            for hypothesis in raw_hypotheses
        ):
            raise TypeError("raw_hypotheses must contain ScoredHypothesis values")
        event_ids = tuple(event.event_id for event in scored_events)
        hypothesis_ids = tuple(
            hypothesis.hypothesis_id for hypothesis in raw_hypotheses
        )
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("scored_events must have unique IDs")
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("raw_hypotheses must have unique IDs")
        if set(event_ids) & set(hypothesis_ids):
            raise ValueError("event and raw-hypothesis IDs must be globally distinct")
        raw_record_ids = {
            record.record_id
            for record in records
            if record.arm_id == self.arm_id
            and record.kind is RecordKind.RAW_HYPOTHESIS
        }
        if self.status is ArmStatus.COMPLETE:
            if set(hypothesis_ids) != raw_record_ids:
                raise ValueError(
                    "raw hypotheses must correspond exactly to arm raw records"
                )
        elif not set(hypothesis_ids) <= raw_record_ids:
            raise ValueError(
                "incomplete-arm raw hypotheses must name surviving raw records"
            )
        records_by_id = {record.record_id: record for record in records}
        raw_records_by_id = {
            record.record_id: record
            for record in records
            if record.arm_id == self.arm_id
            and record.kind is RecordKind.RAW_HYPOTHESIS
        }
        for hypothesis in raw_hypotheses:
            raw_record = raw_records_by_id[hypothesis.hypothesis_id]
            if hypothesis.source_spans != raw_record.source_spans:
                raise ValueError(
                    "raw-hypothesis spans must match their canonical records"
                )
            if not hypothesis.source_spans and {
                "synthetic_anchor_timeline_id",
                "synthetic_anchor_sample",
            } <= set(raw_record.details):
                if (
                    hypothesis.synthetic_anchor_timeline_id
                    != raw_record.details["synthetic_anchor_timeline_id"]
                    or hypothesis.synthetic_anchor_sample
                    != raw_record.details["synthetic_anchor_sample"]
                ):
                    raise ValueError(
                        "synthetic hypothesis anchors must match canonical records"
                    )
        for record in records:
            if record.arm_id != self.arm_id or record.kind not in {
                RecordKind.DECODER_CALL,
                RecordKind.RAW_HYPOTHESIS,
                RecordKind.WRAPPER_DECISION,
            }:
                continue
            if (
                self.status is ArmStatus.COMPLETE
                and any(
                    parent_id not in records_by_id
                    for parent_id in record.parent_ids
                )
            ):
                raise ValueError("target-arm evidence has a dangling parent chain")
        invalid_ids = _freeze_ids(
            self.reference_invalid_event_ids,
            "reference_invalid_event_ids",
        )
        if not set(invalid_ids) <= set(event_ids):
            raise ValueError("invalid-reference IDs must name scored events")
        if not isinstance(self.alignment_policy, AlignmentPolicy):
            raise TypeError("alignment_policy must be AlignmentPolicy")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "scored_events", scored_events)
        object.__setattr__(self, "raw_hypotheses", raw_hypotheses)
        object.__setattr__(self, "reference_invalid_event_ids", invalid_ids)


@dataclass(frozen=True, slots=True)
class ScalarObservation:
    status: ObservationStatus
    value: int | float | None
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ObservationStatus):
            raise TypeError("status must be ObservationStatus")
        _nonempty(self.unit, "unit")
        if self.status is ObservationStatus.OBSERVED:
            if type(self.value) not in (int, float):
                raise ValueError("observed values must be numeric")
            assert self.value is not None
            if not math.isfinite(self.value) or self.value < 0:
                raise ValueError("observed values must be finite and nonnegative")
        elif self.value is not None:
            raise ValueError("unobserved values must be None")


@dataclass(frozen=True, slots=True)
class ResourceObservations:
    rtf: ScalarObservation
    cold_start_seconds: ScalarObservation
    peak_rss_bytes: ScalarObservation
    peak_vram_bytes: ScalarObservation

    def __post_init__(self) -> None:
        expected_units = {
            "rtf": "ratio",
            "cold_start_seconds": "seconds",
            "peak_rss_bytes": "bytes",
            "peak_vram_bytes": "bytes",
        }
        for field_name, unit in expected_units.items():
            value = getattr(self, field_name)
            if not isinstance(value, ScalarObservation):
                raise TypeError(f"{field_name} must be ScalarObservation")
            if value.unit != unit:
                raise ValueError(f"{field_name} must use {unit!r}")


@dataclass(frozen=True, slots=True)
class VocalSoundCounts:
    """Minimal-normalized characters and chains the words level removes.

    References count each scored reference event's own lexical text; outputs
    count each wrapper-accepted hypothesis.
    """

    reference_sound_characters: int
    output_sound_characters: int
    reference_chains: int
    output_chains: int

    def __post_init__(self) -> None:
        for field_name in (
            "reference_sound_characters",
            "output_sound_characters",
            "reference_chains",
            "output_chains",
        ):
            _plain_int(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class EventScore:
    reference: ReferenceEvent
    fate: EventFate
    raw_nonblank: bool | None
    lexically_usable: bool | None
    paced_deadline_met: bool | None
    output_latency_seconds: ScalarObservation

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ReferenceEvent):
            raise TypeError("reference must be ReferenceEvent")
        if not isinstance(self.fate, EventFate):
            raise TypeError("fate must be EventFate")
        if self.fate.event_id != self.reference.event_id:
            raise ValueError("fate and reference event IDs must match")
        _optional_bool(self.raw_nonblank, "raw_nonblank")
        _optional_bool(self.lexically_usable, "lexically_usable")
        _optional_bool(self.paced_deadline_met, "paced_deadline_met")
        if not isinstance(self.output_latency_seconds, ScalarObservation):
            raise TypeError("output_latency_seconds must be ScalarObservation")
        if self.output_latency_seconds.unit != "seconds":
            raise ValueError("output latency must use seconds")
        if (
            self.paced_deadline_met is True
            and self.output_latency_seconds.status is not ObservationStatus.OBSERVED
        ):
            raise ValueError("a met deadline requires an observed output latency")
        if (
            self.lexically_usable is True
            and self.fate.outcome
            is not TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
        ):
            raise ValueError("lexical usability requires wrapper availability")


@dataclass(frozen=True, slots=True)
class NormalizedAlignmentSet:
    level: NormalizationLevel
    normalization_policy_version: str
    normalization_policy_sha256: str
    alignment_policy_version: str
    alignment_policy_sha256: str
    components: tuple[ComponentAlignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.level, NormalizationLevel):
            raise TypeError("level must be NormalizationLevel")
        _nonempty(self.normalization_policy_version, "normalization_policy_version")
        require_sha256(self.normalization_policy_sha256, "normalization_policy_sha256")
        _nonempty(self.alignment_policy_version, "alignment_policy_version")
        require_sha256(self.alignment_policy_sha256, "alignment_policy_sha256")
        expected = {
            NormalizationLevel.RAW: (
                RAW_UNICODE_1_0_VERSION,
                RAW_UNICODE_1_0_SHA256,
            ),
            NormalizationLevel.MINIMAL: (
                JA_MINIMAL_1_1.version,
                JA_MINIMAL_1_1.sha256,
            ),
            NormalizationLevel.LENIENT: (
                JA_LENIENT_1_1.version,
                JA_LENIENT_1_1.sha256,
            ),
            NormalizationLevel.WORDS: (
                JA_WORDS_1_0.version,
                JA_WORDS_1_0.sha256,
            ),
        }[self.level]
        if (
            self.normalization_policy_version,
            self.normalization_policy_sha256,
        ) != expected:
            raise ValueError("normalization identity does not match the level")
        components = tuple(self.components)
        if any(
            not isinstance(component, ComponentAlignment) for component in components
        ):
            raise TypeError("components must contain ComponentAlignment values")
        event_ids = [
            event_id
            for component in components
            for event_id in component.component_event_ids
        ]
        hypothesis_ids = [
            hypothesis_id
            for component in components
            for hypothesis_id in component.component_hypothesis_ids
        ]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("alignment components repeat an event")
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("alignment components repeat a hypothesis")
        if set(event_ids) & set(hypothesis_ids):
            raise ValueError("event and hypothesis IDs must be globally distinct")
        object.__setattr__(self, "components", components)

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(
            event_id
            for component in self.components
            for event_id in component.component_event_ids
        )

    @property
    def hypothesis_ids(self) -> frozenset[str]:
        return frozenset(
            hypothesis_id
            for component in self.components
            for hypothesis_id in component.component_hypothesis_ids
        )


@dataclass(frozen=True, slots=True)
class ScoredFixture:
    fixture: FixtureManifest
    arm_id: str
    timeline_id: str
    sample_rate_hz: int
    sample_count: int
    model_runtime_sha256: str
    events: tuple[EventScore, ...]
    normalized_alignments: tuple[NormalizedAlignmentSet, ...]
    expected_wrapper_hypothesis_ids: tuple[str, ...]
    vad_observation_status: ObservationStatus
    vad_positive_spans: tuple[SampleSpan, ...]
    decoder_window_observation_status: ObservationStatus
    decoder_window_spans: tuple[SampleSpan, ...]
    boundary_truncated_event_ids: tuple[str, ...]
    duplicate_event_ids: tuple[str, ...]
    # Duplicate rule: an event is duplicated when the wrapper-accepted
    # hypotheses its minimal-level alignment consumes came from two or more
    # decoder calls (distinct first parents of their raw hypothesis records).
    # One call that splits a line into segments is not a duplicate.
    duplicate_observation_status: ObservationStatus
    hard_event_ids: tuple[str, ...]
    hard_stratum_version: str
    hard_stratum_sha256: str
    non_speech_control_spans: tuple[SampleSpan, ...]
    hallucination_output_ids: tuple[str, ...]
    resources: ResourceObservations
    vocal_sounds: VocalSoundCounts

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, FixtureManifest):
            raise TypeError("fixture must be FixtureManifest")
        require_uuid4(self.arm_id, "arm_id")
        require_uuid4(self.timeline_id, "timeline_id")
        if _plain_int(self.sample_rate_hz, "sample_rate_hz", minimum=1) != 16_000:
            raise ValueError("E0 scoring requires the canonical 16000 Hz timeline")
        sample_count = _plain_int(self.sample_count, "sample_count", minimum=1)
        require_sha256(self.model_runtime_sha256, "model_runtime_sha256")
        events = tuple(self.events)
        if any(not isinstance(event, EventScore) for event in events):
            raise TypeError("events must contain EventScore values")
        event_ids = tuple(event.reference.event_id for event in events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("events must have unique IDs")
        if any(event.fate.arm_id != self.arm_id for event in events):
            raise ValueError("event fates must use the fixture arm")
        alignments = tuple(self.normalized_alignments)
        if any(
            not isinstance(item, NormalizedAlignmentSet) for item in alignments
        ):
            raise TypeError(
                "normalized_alignments must contain NormalizedAlignmentSet values"
            )
        if {item.level for item in alignments} != set(NormalizationLevel) or len(
            alignments
        ) != len(NormalizationLevel):
            raise ValueError(
                "exactly raw, minimal, lenient, and words alignments are required"
            )
        alignments = tuple(sorted(alignments, key=lambda item: item.level.value))
        if any(item.event_ids != frozenset(event_ids) for item in alignments):
            raise ValueError("each normalization must cover every event exactly once")
        expected_hypothesis_ids = _freeze_ids(
            self.expected_wrapper_hypothesis_ids,
            "expected_wrapper_hypothesis_ids",
        )
        if set(event_ids) & set(expected_hypothesis_ids):
            raise ValueError("event and wrapper-hypothesis IDs must be distinct")
        if any(
            item.hypothesis_ids != frozenset(expected_hypothesis_ids)
            for item in alignments
        ):
            raise ValueError(
                "each normalization must cover the expected wrapper hypotheses"
            )
        object.__setattr__(
            self,
            "expected_wrapper_hypothesis_ids",
            expected_hypothesis_ids,
        )
        memberships = {
            tuple(
                sorted(
                    (
                        tuple(sorted(component.component_event_ids)),
                        tuple(sorted(component.component_hypothesis_ids)),
                    )
                    for component in item.components
                )
            )
            for item in alignments
        }
        if len(memberships) != 1:
            raise ValueError("normalization profiles must share temporal membership")
        alignment_identities = {
            (item.alignment_policy_version, item.alignment_policy_sha256)
            for item in alignments
        }
        if len(alignment_identities) != 1:
            raise ValueError("normalization profiles must share alignment policy")
        lexical_reference_ids = {
            event.reference.event_id
            for event in events
            if event.reference.intelligibility == "intelligible"
            and event.reference.verbatim_ref_id is not None
        }
        for item in alignments:
            for component in item.components:
                if set(component.no_score_event_ids) & lexical_reference_ids:
                    raise ValueError(
                        "intelligible events with verbatim text cannot be no-score"
                    )
                if any(
                    alignment.event_id not in lexical_reference_ids
                    for alignment in component.event_alignments
                ):
                    raise ValueError(
                        "only intelligible events with verbatim text are aligned"
                    )
        for event in events:
            span = event.reference.span
            if span.timeline_id != self.timeline_id or span.end > sample_count:
                raise ValueError("reference event lies outside the fixture timeline")
        for field_name in (
            "vad_positive_spans",
            "decoder_window_spans",
            "non_speech_control_spans",
        ):
            spans = _freeze_spans(getattr(self, field_name), field_name)
            if any(
                span.timeline_id != self.timeline_id or span.end > sample_count
                for span in spans
            ):
                raise ValueError(f"{field_name} lies outside the fixture timeline")
            object.__setattr__(self, field_name, spans)
        for status_name, spans_name in (
            ("vad_observation_status", "vad_positive_spans"),
            ("decoder_window_observation_status", "decoder_window_spans"),
            ("duplicate_observation_status", "duplicate_event_ids"),
        ):
            status = getattr(self, status_name)
            if not isinstance(status, ObservationStatus):
                raise TypeError(f"{status_name} must be ObservationStatus")
            if status is not ObservationStatus.OBSERVED and getattr(self, spans_name):
                raise ValueError(f"unobserved {spans_name} must be empty")
        event_id_set = set(event_ids)
        for field_name in (
            "boundary_truncated_event_ids",
            "duplicate_event_ids",
            "hard_event_ids",
        ):
            values = _freeze_ids(getattr(self, field_name), field_name)
            if not set(values) <= event_id_set:
                raise ValueError(f"{field_name} must name fixture events")
            object.__setattr__(self, field_name, values)
        _nonempty(self.hard_stratum_version, "hard_stratum_version")
        require_sha256(self.hard_stratum_sha256, "hard_stratum_sha256")
        hallucination_ids = _freeze_ids(
            self.hallucination_output_ids,
            "hallucination_output_ids",
        )
        if not set(hallucination_ids) <= set(alignments[0].hypothesis_ids):
            raise ValueError("hallucination IDs must name scored hypotheses")
        if hallucination_ids and not self.non_speech_control_spans:
            raise ValueError("hallucinations require an observed non-speech control")
        object.__setattr__(self, "hallucination_output_ids", hallucination_ids)
        reference_spans = tuple(event.reference.span for event in events)
        if any(
            _positive_overlap(control, reference)
            for control in self.non_speech_control_spans
            for reference in reference_spans
        ):
            raise ValueError("non-speech controls must not overlap reference events")
        if not isinstance(self.resources, ResourceObservations):
            raise TypeError("resources must be ResourceObservations")
        if not isinstance(self.vocal_sounds, VocalSoundCounts):
            raise TypeError("vocal_sounds must be VocalSoundCounts")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "normalized_alignments", alignments)


def _positive_overlap(left: SampleSpan, right: SampleSpan) -> bool:
    return (
        left.timeline_id == right.timeline_id
        and max(left.start, right.start) < min(left.end, right.end)
    )


def _span_gap(left: SampleSpan, right: SampleSpan) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _hypothesis_has_direct_edge(
    event: ScoredEvent,
    hypothesis: ScoredHypothesis,
    policy: AlignmentPolicy,
) -> bool:
    return any(
        span.timeline_id == event.span.timeline_id
        and _span_gap(event.span, span) <= policy.collar_samples
        for span in hypothesis.source_spans
    )


def _record_codes(record: EvidenceRecord) -> tuple[DiagnosticCode, ...]:
    result = list(record.diagnostics)
    if record.decision is not None:
        try:
            code = DiagnosticCode(record.decision)
        except ValueError:
            pass
        else:
            if code not in result:
                result.append(code)
    return tuple(result)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _record_sort_key(record: EvidenceRecord) -> tuple[int, str, str]:
    return (record.observed_monotonic_ns, record.session_id, record.record_id)


def _record_is_in_arm_scope(record: EvidenceRecord, arm_id: str) -> bool:
    return record.arm_id is None or record.arm_id == arm_id


def _alignment_membership(
    alignments: Sequence[ComponentAlignment],
) -> tuple[frozenset[str], frozenset[str]]:
    event_ids: list[str] = []
    hypothesis_ids: list[str] = []
    for alignment in alignments:
        if not isinstance(alignment, ComponentAlignment):
            raise TypeError("alignments must contain ComponentAlignment values")
        event_ids.extend(alignment.component_event_ids)
        hypothesis_ids.extend(alignment.component_hypothesis_ids)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("alignments repeat an event")
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise ValueError("alignments repeat a hypothesis")
    return frozenset(event_ids), frozenset(hypothesis_ids)


def _diagnostic_for_unwindowed_event(
    event: ReferenceEvent,
    records: Sequence[EvidenceRecord],
    arm_id: str,
) -> tuple[DiagnosticCode, EvidenceRecord | None]:
    diagnostic_records = tuple(
        record
        for record in records
        if record.kind is RecordKind.DIAGNOSTIC
        and record.observation_status is ObservationStatus.OBSERVED
        and _record_is_in_arm_scope(record, arm_id)
    )
    specific = tuple(
        record
        for record in diagnostic_records
        if record.source_spans
        and any(_positive_overlap(event.span, span) for span in record.source_spans)
    )
    candidates = specific or tuple(
        record for record in diagnostic_records if not record.source_spans
    )
    priority = (
        DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF,
        DiagnosticCode.ENDPOINT_PENDING_EOF,
        DiagnosticCode.VAD_NOT_DETECTED,
        DiagnosticCode.UNKNOWN_ENDPOINT_REASON,
    )
    for code in priority:
        matches = sorted(
            (record for record in candidates if code in _record_codes(record)),
            key=_record_sort_key,
        )
        if matches:
            return code, matches[0]
    return DiagnosticCode.UNKNOWN_ENDPOINT_REASON, None


def _coverage_samples(span: SampleSpan, covering: Iterable[SampleSpan]) -> int:
    intervals = sorted(
        (
            max(span.start, candidate.start),
            min(span.end, candidate.end),
        )
        for candidate in covering
        if candidate.timeline_id == span.timeline_id
        and max(span.start, candidate.start) < min(span.end, candidate.end)
    )
    if not intervals:
        return 0
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _record_has_ancestor(
    record: EvidenceRecord,
    ancestor_id: str,
    records_by_id: dict[str, EvidenceRecord],
) -> bool:
    pending = list(record.parent_ids)
    seen: set[str] = set()
    while pending:
        record_id = pending.pop()
        if record_id == ancestor_id:
            return True
        if record_id in seen:
            continue
        seen.add(record_id)
        parent = records_by_id.get(record_id)
        if parent is not None:
            pending.extend(parent.parent_ids)
    return False


def _declared_raw_count(record: EvidenceRecord) -> int | None:
    value = record.details.get("raw_segment_count")
    return value if type(value) is int and value >= 0 else None


def _source_window_for_call(
    call: EvidenceRecord,
    windows: Sequence[EvidenceRecord],
    arm_kind: ArmKind,
) -> tuple[EvidenceRecord | None, bool]:
    source_artifact_id = call.details.get("source_artifact_id")
    source_pcm_sha256 = call.details.get("source_pcm_sha256")
    try:
        require_uuid4(source_artifact_id, "decoder source_artifact_id")
        require_sha256(source_pcm_sha256, "decoder source_pcm_sha256")
    except (TypeError, ValueError):
        return None, False
    if arm_kind is ArmKind.P1:
        return None, True
    matches = tuple(
        window
        for window in windows
        if window.artifact_id == source_artifact_id
        and window.details.get("pcm_sha256") == source_pcm_sha256
        and window.source_spans == call.source_spans
    )
    return (matches[0], True) if len(matches) == 1 else (None, False)


def _call_parent_chain_is_complete(
    call: EvidenceRecord,
    arm_kind: ArmKind,
    records_by_id: dict[str, EvidenceRecord],
) -> bool:
    if call.details.get("call_mode") != arm_kind.value:
        return False
    parent_call_id = call.details.get("parent_call_record_id")
    if arm_kind in {ArmKind.P0, ArmKind.P1}:
        return parent_call_id is None and not call.parent_ids
    expected_parent_kind = {
        ArmKind.P2A: ArmKind.P0,
        ArmKind.P2B: ArmKind.P2A,
    }[arm_kind]
    if type(parent_call_id) is not str or call.parent_ids != (parent_call_id,):
        return False
    parent = records_by_id.get(parent_call_id)
    if (
        parent is None
        or parent.kind is not RecordKind.DECODER_CALL
        or parent.observation_status is not ObservationStatus.OBSERVED
        or parent.details.get("call_mode") != expected_parent_kind.value
        or parent.source_spans != call.source_spans
    ):
        return False
    inherited_fields = (
        "decoder_input_pcm_sha256",
        "delegate_instance_id",
        "model_artifact_sha256",
        "oracle_component_id",
        "oracle_component_map_sha256",
        "oracle_event_ids",
        "oracle_reference_set_id",
        "oracle_reference_set_sha256",
        "oracle_slice_pcm_sha256",
        "source_artifact_id",
        "source_pcm_sha256",
        "source_to_decoder_map",
        "source_to_decoder_map_id",
        "source_to_decoder_map_sha256",
    )
    if any(call.details.get(field) != parent.details.get(field) for field in inherited_fields):
        return False
    return _call_parent_chain_is_complete(
        parent,
        expected_parent_kind,
        records_by_id,
    )


def _p1_call_owns_event(
    call: EvidenceRecord,
    event: ReferenceEvent,
    events_by_id: dict[str, ReferenceEvent],
    reference_identity: tuple[str, str] | None,
) -> bool:
    details = call.details
    oracle_event_ids = details.get("oracle_event_ids")
    if (
        type(oracle_event_ids) is not tuple
        or not oracle_event_ids
        or len(oracle_event_ids) != len(set(oracle_event_ids))
        or event.event_id not in oracle_event_ids
        or any(type(event_id) is not str for event_id in oracle_event_ids)
        or any(event_id not in events_by_id for event_id in oracle_event_ids)
        or len(call.source_spans) != 1
        or reference_identity is None
    ):
        return False
    component_id = details.get("oracle_component_id")
    reference_set_id = details.get("oracle_reference_set_id")
    reference_set_sha256 = details.get("oracle_reference_set_sha256")
    component_map_sha256 = details.get("oracle_component_map_sha256")
    slice_pcm_sha256 = details.get("oracle_slice_pcm_sha256")
    source_artifact_id = details.get("source_artifact_id")
    source_pcm_sha256 = details.get("source_pcm_sha256")
    try:
        require_uuid4(component_id, "oracle_component_id")
        require_uuid4(reference_set_id, "oracle_reference_set_id")
        require_sha256(reference_set_sha256, "oracle_reference_set_sha256")
        require_sha256(component_map_sha256, "oracle_component_map_sha256")
        require_sha256(slice_pcm_sha256, "oracle_slice_pcm_sha256")
    except (TypeError, ValueError):
        return False
    if (
        (reference_set_id, reference_set_sha256) != reference_identity
        or source_artifact_id != component_id
        or source_pcm_sha256 != slice_pcm_sha256
    ):
        return False
    members = tuple(events_by_id[event_id] for event_id in oracle_event_ids)
    master_ids = {member.master_id for member in members}
    timeline_ids = {member.span.timeline_id for member in members}
    source_span = call.source_spans[0]
    if (
        len(master_ids) != 1
        or timeline_ids != {source_span.timeline_id}
        or any(
            source_span.start > member.span.start
            or source_span.end < member.span.end
            for member in members
        )
    ):
        return False
    ordered_members = tuple(
        sorted(
            members,
            key=lambda member: (
                member.span.start,
                member.span.end,
                member.event_id,
            ),
        )
    )
    component_start = min(member.span.start for member in ordered_members)
    component_end = max(member.span.end for member in ordered_members)
    if source_span.start != max(0, component_start - 8_000):
        return False
    if source_span.end > component_end + 8_000:
        return False
    expected_map_sha256 = canonical_sha256(
        {
            "context_samples_each_side": 8_000,
            "events": tuple(
                {
                    "event_id": member.event_id,
                    "span": {
                        "end": member.span.end,
                        "start": member.span.start,
                        "timeline_id": member.span.timeline_id,
                    },
                }
                for member in ordered_members
            ),
            "master_id": ordered_members[0].master_id,
            "oracle_component_id": component_id,
            "reference_set_id": reference_set_id,
            "reference_set_sha256": reference_set_sha256,
            "slice": {
                "end": source_span.end,
                "start": source_span.start,
                "timeline_id": source_span.timeline_id,
            },
        }
    )
    return component_map_sha256 == expected_map_sha256


def _validate_complete_arm_call_graph(
    arm: ArmEvidence,
    events_by_id: dict[str, ReferenceEvent],
    records: Sequence[EvidenceRecord],
    records_by_id: dict[str, EvidenceRecord],
    windows: Sequence[EvidenceRecord],
    decoder_calls: Sequence[EvidenceRecord],
    raw_records: Sequence[EvidenceRecord],
    p1_reference_identity: tuple[str, str] | None,
) -> None:
    if arm.status is not ArmStatus.COMPLETE:
        return
    all_target_calls = tuple(
        record
        for record in records
        if record.kind is RecordKind.DECODER_CALL
        and record.arm_id == arm.arm_id
    )
    if tuple(decoder_calls) != all_target_calls:
        raise ValueError("a complete arm cannot contain an unobserved decoder call")
    all_target_raw_records = tuple(
        record
        for record in records
        if record.kind is RecordKind.RAW_HYPOTHESIS
        and record.arm_id == arm.arm_id
    )
    if tuple(raw_records) != all_target_raw_records:
        raise ValueError("a complete arm cannot contain an unobserved raw record")
    target_call_ids = {call.record_id for call in decoder_calls}
    if any(
        len(record.parent_ids) != 1
        or record.parent_ids[0] not in target_call_ids
        for record in raw_records
    ):
        raise ValueError(
            "complete-arm raw records must partition under target decoder calls"
        )
    window_artifact_ids = tuple(window.artifact_id for window in windows)
    if len(window_artifact_ids) != len(set(window_artifact_ids)):
        raise ValueError("window artifacts must be unique within an arm run")
    for call in decoder_calls:
        if not _call_parent_chain_is_complete(call, arm.kind, records_by_id):
            raise ValueError("decoder-call arm mode or parent chain is invalid")
        _source_window, source_complete = _source_window_for_call(
            call,
            windows,
            arm.kind,
        )
        if not source_complete:
            raise ValueError("decoder-call source provenance is incomplete")
        actual_raw_count = sum(
            record.parent_ids == (call.record_id,)
            for record in raw_records
        )
        declared_raw_count = _declared_raw_count(call)
        if (
            declared_raw_count is None
            or actual_raw_count != declared_raw_count
        ):
            raise ValueError("decoder-call raw evidence count is incomplete")

    if arm.kind is ArmKind.P1:
        claimed_event_ids: list[str] = []
        for call in decoder_calls:
            oracle_event_ids = call.details.get("oracle_event_ids")
            if type(oracle_event_ids) is not tuple or not oracle_event_ids:
                raise ValueError("P1 calls require oracle event membership")
            first_event = events_by_id.get(oracle_event_ids[0])
            if first_event is None or not _p1_call_owns_event(
                call,
                first_event,
                events_by_id,
                p1_reference_identity,
            ):
                raise ValueError("P1 oracle provenance is incomplete")
            claimed_event_ids.extend(oracle_event_ids)
        if (
            len(claimed_event_ids) != len(set(claimed_event_ids))
            or set(claimed_event_ids) != set(events_by_id)
        ):
            raise ValueError("P1 oracle components must partition reference events")
        return

    call_source_artifact_ids = tuple(
        call.details.get("source_artifact_id") for call in decoder_calls
    )
    if arm.kind is ArmKind.P0:
        if (
            len(call_source_artifact_ids) != len(set(call_source_artifact_ids))
            or set(call_source_artifact_ids) != set(window_artifact_ids)
        ):
            raise ValueError("P0 requires exactly one decoder call per window")
        return

    expected_parent_kind = {
        ArmKind.P2A: ArmKind.P0,
        ArmKind.P2B: ArmKind.P2A,
    }[arm.kind]
    expected_parents = tuple(
        record
        for record in records
        if record.kind is RecordKind.DECODER_CALL
        and record.observation_status is ObservationStatus.OBSERVED
        and record.details.get("call_mode") == expected_parent_kind.value
    )
    expected_parent_source_ids = tuple(
        parent.details.get("source_artifact_id") for parent in expected_parents
    )
    if (
        len(expected_parent_source_ids) != len(set(expected_parent_source_ids))
        or set(expected_parent_source_ids) != set(window_artifact_ids)
        or any(
            not _source_window_for_call(
                parent,
                windows,
                expected_parent_kind,
            )[1]
            for parent in expected_parents
        )
    ):
        raise ValueError("P2 parent calls must cover every source window exactly once")
    expected_parent_ids = tuple(parent.record_id for parent in expected_parents)
    claimed_parent_ids = tuple(
        call.details.get("parent_call_record_id") for call in decoder_calls
    )
    if (
        len(claimed_parent_ids) != len(set(claimed_parent_ids))
        or set(claimed_parent_ids) != set(expected_parent_ids)
    ):
        raise ValueError("P2 requires exactly one child per parent decoder call")


def _known_diagnostic(reason: str) -> tuple[DiagnosticCode, ...]:
    try:
        return (DiagnosticCode(reason),)
    except ValueError:
        return ()


def _emit_derived_diagnostic(
    event: ReferenceEvent,
    arm_id: str,
    code: DiagnosticCode,
    parent_ids: tuple[str, ...],
    sink: EvidenceSink,
) -> EvidenceRecord:
    return sink.emit(
        EvidenceRecordDraft(
            arm_id=arm_id,
            stage=Stage.ENDPOINT,
            kind=RecordKind.DIAGNOSTIC,
            artifact_id=event.event_id,
            parent_ids=parent_ids,
            source_spans=(event.span,),
            decision=code.value,
            diagnostics=(code,),
            classification=PrivateContentClass.PRIVATE_AUDIO,
        )
    )


def _emit_fate_record(
    event: ReferenceEvent,
    fate: EventFate,
    sink: EvidenceSink,
) -> None:
    parents = _ordered_unique(
        (*fate.evidence_record_ids, *fate.diagnostic_record_ids)
    )
    sink.emit(
        EvidenceRecordDraft(
            arm_id=fate.arm_id,
            stage=Stage.SCORING,
            kind=RecordKind.EVENT_FATE,
            artifact_id=event.event_id,
            parent_ids=parents,
            source_spans=(event.span,),
            decision=fate.outcome.value,
            diagnostics=_known_diagnostic(fate.reason),
            classification=PrivateContentClass.PRIVATE_AUDIO,
        )
    )


def _make_fate(
    event: ReferenceEvent,
    arm_id: str,
    outcome: TerminalOutcome,
    first_missing_stage: Stage | None,
    reason: str,
    evidence_records: Iterable[EvidenceRecord],
    diagnostic_records: Iterable[EvidenceRecord],
    sink: EvidenceSink,
) -> EventFate:
    evidence_ids = _ordered_unique(
        record.record_id for record in sorted(evidence_records, key=_record_sort_key)
    )
    diagnostic_ids = _ordered_unique(
        record.record_id
        for record in sorted(diagnostic_records, key=_record_sort_key)
    )
    fate = EventFate(
        event_id=event.event_id,
        arm_id=arm_id,
        outcome=outcome,
        first_missing_stage=first_missing_stage,
        reason=reason,
        evidence_record_ids=evidence_ids,
        diagnostic_record_ids=diagnostic_ids,
    )
    _emit_fate_record(event, fate, sink)
    return fate


def scored_reference_events(
    reference_set: ReferenceSet,
    reference_texts: Mapping[str, str],
) -> tuple[ScoredEvent, ...]:
    """Give each frozen reference event its own lexical text, if it has one."""

    verbatim = {
        item.verbatim_ref_id: item for item in reference_set.verbatim_references
    }
    result: list[ScoredEvent] = []
    for event in reference_set.events:
        if event.intelligibility != "intelligible" or event.verbatim_ref_id is None:
            # Unintelligible, partially intelligible, and coverage-only speech
            # has no lexical reference: its span is a no-score region.
            result.append(ScoredEvent(event.event_id, event.span, None, no_score=True))
            continue
        unit = verbatim[event.verbatim_ref_id]
        # A verbatim unit spanning multiple events has intentionally unresolved
        # per-event lexical ownership; temporal fate attribution still works.
        text = reference_texts[unit.text_id] if len(unit.event_ids) == 1 else None
        result.append(ScoredEvent(event.event_id, event.span, text))
    return tuple(result)


def derive_event_fates(
    events: Sequence[ReferenceEvent],
    arm: ArmEvidence,
    alignments: Sequence[ComponentAlignment],
    sink: EvidenceSink,
) -> tuple[EventFate, ...]:
    """Derive the first missing representation for each event in one arm."""

    references = tuple(events)
    if any(not isinstance(event, ReferenceEvent) for event in references):
        raise TypeError("events must contain ReferenceEvent values")
    event_ids = tuple(event.event_id for event in references)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("events must have unique IDs")
    references_by_id = {event.event_id: event for event in references}
    scored_by_id = {event.event_id: event for event in arm.scored_events}
    if set(event_ids) != set(scored_by_id):
        raise ValueError("events and arm.scored_events must correspond exactly")
    if any(
        scored_by_id[event.event_id].span != event.span for event in references
    ):
        raise ValueError("scored-event spans must match canonical reference spans")
    alignment_event_ids, alignment_hypothesis_ids = _alignment_membership(alignments)
    expected_hypothesis_ids = {
        hypothesis.hypothesis_id for hypothesis in arm.raw_hypotheses
    }
    if arm.status is ArmStatus.COMPLETE:
        if (
            alignment_event_ids != frozenset(event_ids)
            or alignment_hypothesis_ids != expected_hypothesis_ids
        ):
            raise ValueError(
                "alignments must cover the arm events and raw hypotheses"
            )
    elif (
        not alignment_event_ids <= frozenset(event_ids)
        or not alignment_hypothesis_ids <= expected_hypothesis_ids
    ):
        raise ValueError(
            "incomplete-arm alignments must name only surviving inputs"
        )

    # Fate connectivity follows the resolved minimal alignment: an event is
    # represented only by the hypotheses that alignment assigns to it, so a
    # neighbour's segment inside the collar cannot rescue a dropped event. Two
    # cases have no lexical assignment and fall back to direct temporal edges:
    # events in an ``alignment_unresolved`` component and no-score events.
    assigned_hypotheses: dict[str, frozenset[str]] = {}
    temporal_fallback_event_ids: set[str] = set()
    for alignment in alignments:
        if alignment.status is AlignmentStatus.RESOLVED:
            assigned_hypotheses.update(
                (item.event_id, frozenset(item.hypothesis_ids))
                for item in alignment.event_alignments
            )
            temporal_fallback_event_ids.update(alignment.no_score_event_ids)
        else:
            temporal_fallback_event_ids.update(alignment.component_event_ids)

    records = tuple(sorted(arm.records, key=_record_sort_key))
    records_by_id = {record.record_id: record for record in records}
    hypotheses_by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in arm.raw_hypotheses
    }
    windows = tuple(
        record
        for record in records
        if record.kind is RecordKind.WINDOW
        and record.observation_status is ObservationStatus.OBSERVED
    )
    decoder_calls = tuple(
        record
        for record in records
        if record.kind is RecordKind.DECODER_CALL
        and record.arm_id == arm.arm_id
        and record.observation_status is ObservationStatus.OBSERVED
    )
    raw_records = tuple(
        record
        for record in records
        if record.kind is RecordKind.RAW_HYPOTHESIS
        and record.arm_id == arm.arm_id
        and record.observation_status is ObservationStatus.OBSERVED
    )
    wrappers = tuple(
        record
        for record in records
        if record.kind is RecordKind.WRAPPER_DECISION
        and record.arm_id == arm.arm_id
        and record.observation_status is ObservationStatus.OBSERVED
    )
    p1_reference_identities = {
        (
            record.details.get("oracle_reference_set_id"),
            record.details.get("oracle_reference_set_sha256"),
        )
        for record in decoder_calls
        if record.details.get("call_mode") == ArmKind.P1.value
    }
    p1_reference_identity = (
        next(iter(p1_reference_identities))
        if arm.kind is ArmKind.P1
        and len(p1_reference_identities) == 1
        and all(
            type(value) is str
            for value in next(iter(p1_reference_identities))
        )
        else None
    )
    _validate_complete_arm_call_graph(
        arm,
        references_by_id,
        records,
        records_by_id,
        windows,
        decoder_calls,
        raw_records,
        p1_reference_identity,
    )

    results: list[EventFate] = []
    for event in references:
        scored_event = scored_by_id[event.event_id]
        base_evidence: list[EvidenceRecord] = []
        if event.event_id in arm.reference_invalid_event_ids:
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.REFERENCE_INVALID,
                    Stage.REFERENCE,
                    TerminalOutcome.REFERENCE_INVALID.value,
                    base_evidence,
                    (),
                    sink,
                )
            )
            continue
        if arm.status is not ArmStatus.COMPLETE:
            interruption_codes = {
                DiagnosticCode.ENDPOINT_PENDING_SHUTDOWN,
                DiagnosticCode.ASR_QUEUE_TIMEOUT,
                DiagnosticCode.SHUTDOWN_CANCELLED,
            }
            specific_interruptions = tuple(
                record
                for record in records
                if interruption_codes.intersection(_record_codes(record))
                and _record_is_in_arm_scope(record, arm.arm_id)
                and record.source_spans
                and any(
                    _positive_overlap(event.span, span)
                    for span in record.source_spans
                )
            )
            interruption_records = specific_interruptions or tuple(
                record
                for record in records
                if interruption_codes.intersection(_record_codes(record))
                and _record_is_in_arm_scope(record, arm.arm_id)
                and not record.source_spans
            )
            first_interruption = min(
                interruption_records,
                key=_record_sort_key,
                default=None,
            )
            interruption_reason = (
                _record_codes(first_interruption)[0].value
                if first_interruption is not None
                else arm.status.value
            )
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    first_interruption.stage
                    if first_interruption is not None
                    else Stage.RUN,
                    interruption_reason,
                    interruption_records,
                    interruption_records,
                    sink,
                )
            )
            continue

        overlapping_windows = (
            ()
            if arm.kind is ArmKind.P1
            else tuple(
                record
                for record in windows
                if any(
                    _positive_overlap(event.span, span)
                    for span in record.source_spans
                )
            )
        )
        spatial_calls = tuple(
            record
            for record in decoder_calls
            if any(_positive_overlap(event.span, span) for span in record.source_spans)
        )
        if arm.kind is ArmKind.P1:
            owned_calls = tuple(
                record
                for record in decoder_calls
                if type(record.details.get("oracle_event_ids")) is tuple
                and event.event_id in record.details["oracle_event_ids"]
            )
            covering_calls = owned_calls or spatial_calls
        else:
            covering_calls = spatial_calls
        if not covering_calls and arm.kind is ArmKind.P1:
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    Stage.DECODER,
                    "oracle_decoder_call_missing",
                    (),
                    (),
                    sink,
                )
            )
            continue
        if not covering_calls and not overlapping_windows:
            code, diagnostic = _diagnostic_for_unwindowed_event(
                event,
                records,
                arm.arm_id,
            )
            stage = (
                Stage.VAD
                if code
                in {
                    DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF,
                    DiagnosticCode.VAD_NOT_DETECTED,
                }
                else Stage.ENDPOINT
            )
            diagnostic_records = () if diagnostic is None else (diagnostic,)
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.NO_DECODER_WINDOW,
                    stage,
                    code.value,
                    (*base_evidence, *diagnostic_records),
                    diagnostic_records,
                    sink,
                )
            )
            continue
        if not covering_calls:
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    Stage.DECODER,
                    DiagnosticCode.UNKNOWN_DECODER_GATE.value,
                    (*base_evidence, *overlapping_windows),
                    (),
                    sink,
                )
            )
            continue

        source_windows: list[EvidenceRecord] = []
        call_provenance_complete = not (
            arm.kind is ArmKind.P1 and len(covering_calls) != 1
        )
        for call in covering_calls:
            source_window, source_complete = _source_window_for_call(
                call,
                windows,
                arm.kind,
            )
            if source_window is not None and source_window not in source_windows:
                source_windows.append(source_window)
            call_provenance_complete &= source_complete
            call_provenance_complete &= _call_parent_chain_is_complete(
                call,
                arm.kind,
                records_by_id,
            )
            if arm.kind is ArmKind.P1:
                call_provenance_complete &= _p1_call_owns_event(
                    call,
                    event,
                    references_by_id,
                    p1_reference_identity,
                )
        event_windows = tuple(source_windows)
        base_evidence.extend(event_windows)
        base_evidence.extend(covering_calls)
        if not call_provenance_complete:
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    Stage.INTEGRITY,
                    "decoder_provenance_incomplete",
                    base_evidence,
                    (),
                    sink,
                )
            )
            continue
        coverage_spans = tuple(
            span for record in covering_calls for span in record.source_spans
        )
        coverage = _coverage_samples(event.span, coverage_spans)
        nonterminal_diagnostics: list[EvidenceRecord] = []
        if 0 < coverage < event.span.sample_count:
            boundary = _emit_derived_diagnostic(
                event,
                arm.arm_id,
                DiagnosticCode.ENDPOINT_BOUNDARY_TRUNCATED,
                tuple(record.record_id for record in covering_calls),
                sink,
            )
            nonterminal_diagnostics.append(boundary)

        call_ids = {record.record_id for record in covering_calls}
        call_raw_records = tuple(
            record
            for record in raw_records
            if len(record.parent_ids) == 1 and record.parent_ids[0] in call_ids
        )
        incomplete_call_records: list[EvidenceRecord] = []
        for call in covering_calls:
            actual_count = sum(
                record.parent_ids == (call.record_id,)
                for record in call_raw_records
            )
            declared_count = _declared_raw_count(call)
            if declared_count is None or actual_count != declared_count:
                incomplete_call_records.append(call)
        if not call_raw_records:
            if incomplete_call_records:
                diagnostics = (*nonterminal_diagnostics, *incomplete_call_records)
                results.append(
                    _make_fate(
                        event,
                        arm.arm_id,
                        TerminalOutcome.UNKNOWN,
                        Stage.INTEGRITY,
                        ObservationStatus.RECORD_LOST.value,
                        base_evidence,
                        diagnostics,
                        sink,
                    )
                )
                continue
            no_yield_records = tuple(
                record
                for record in covering_calls
                if record.decision == DiagnosticCode.DECODER_NO_YIELD.value
                or DiagnosticCode.DECODER_NO_YIELD in record.diagnostics
            )
            diagnostics = (*nonterminal_diagnostics, *no_yield_records)
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.DECODER_NO_YIELD,
                    Stage.DECODER,
                    TerminalOutcome.DECODER_NO_YIELD.value,
                    base_evidence,
                    diagnostics,
                    sink,
                )
            )
            continue

        if event.event_id in temporal_fallback_event_ids:
            connected_raw_records = tuple(
                record
                for record in call_raw_records
                if _hypothesis_has_direct_edge(
                    scored_event,
                    hypotheses_by_id[record.record_id],
                    arm.alignment_policy,
                )
            )
        else:
            assigned_ids = assigned_hypotheses[event.event_id]
            connected_raw_records = tuple(
                record
                for record in call_raw_records
                if record.record_id in assigned_ids
            )
        if not connected_raw_records:
            if incomplete_call_records:
                diagnostics = (*nonterminal_diagnostics, *incomplete_call_records)
                results.append(
                    _make_fate(
                        event,
                        arm.arm_id,
                        TerminalOutcome.UNKNOWN,
                        Stage.INTEGRITY,
                        ObservationStatus.RECORD_LOST.value,
                        (*base_evidence, *call_raw_records),
                        diagnostics,
                        sink,
                    )
                )
                continue
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.DECODER_EVENT_UNREPRESENTED,
                    Stage.DECODER,
                    TerminalOutcome.DECODER_EVENT_UNREPRESENTED.value,
                    (*base_evidence, *call_raw_records),
                    nonterminal_diagnostics,
                    sink,
                )
            )
            continue

        base_evidence.extend(connected_raw_records)
        connected_ids = {record.record_id for record in connected_raw_records}
        related_wrappers = tuple(
            wrapper
            for wrapper in wrappers
            if any(
                _record_has_ancestor(wrapper, raw_id, records_by_id)
                for raw_id in connected_ids
            )
        )
        available_wrappers = tuple(
            wrapper
            for wrapper in related_wrappers
            if wrapper.decision
            == TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value
        )
        rejected_wrappers = tuple(
            wrapper
            for wrapper in related_wrappers
            if wrapper.decision
            in {
                DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value,
                DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value,
            }
        )
        empty_wrappers = tuple(
            wrapper
            for wrapper in related_wrappers
            if wrapper.decision == DiagnosticCode.WRAPPER_EMPTY.value
        )
        available_raw_ids = {
            raw_id
            for raw_id in connected_ids
            if any(
                _record_has_ancestor(wrapper, raw_id, records_by_id)
                for wrapper in available_wrappers
            )
        }
        rejected_raw_ids = {
            raw_id
            for raw_id in connected_ids
            if any(
                _record_has_ancestor(wrapper, raw_id, records_by_id)
                for wrapper in rejected_wrappers
            )
        }
        empty_raw_ids = {
            raw_id
            for raw_id in connected_ids
            if raw_id not in rejected_raw_ids
            and any(
                _record_has_ancestor(wrapper, raw_id, records_by_id)
                for wrapper in empty_wrappers
            )
        }
        contradictory_raw_ids = available_raw_ids & (
            rejected_raw_ids | empty_raw_ids
        )
        accounted_raw_ids = (
            available_raw_ids | rejected_raw_ids | empty_raw_ids
        )
        accepted_text = "".join(
            hypothesis.text
            for hypothesis in sorted(
                (
                    hypotheses_by_id[raw_id]
                    for raw_id in available_raw_ids
                ),
                key=lambda hypothesis: (
                    hypothesis.timeline_id,
                    hypothesis.extent_start,
                    hypothesis.extent_end,
                    hypothesis.hypothesis_id,
                ),
            )
        ).strip()
        if incomplete_call_records:
            diagnostics = (*nonterminal_diagnostics, *incomplete_call_records)
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    Stage.INTEGRITY,
                    ObservationStatus.RECORD_LOST.value,
                    (*base_evidence, *related_wrappers),
                    diagnostics,
                    sink,
                )
            )
        elif contradictory_raw_ids:
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    Stage.INTEGRITY,
                    "wrapper_evidence_contradictory",
                    (*base_evidence, *related_wrappers),
                    nonterminal_diagnostics,
                    sink,
                )
            )
        elif (
            accounted_raw_ids == connected_ids
            and available_raw_ids
            and accepted_text
        ):
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE,
                    None,
                    TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value,
                    (*base_evidence, *available_wrappers),
                    nonterminal_diagnostics,
                    sink,
                )
            )
        elif rejected_raw_ids == connected_ids:
            diagnostics = (*nonterminal_diagnostics, *rejected_wrappers)
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.WRAPPER_REJECTED,
                    Stage.WRAPPER,
                    TerminalOutcome.WRAPPER_REJECTED.value,
                    (*base_evidence, *rejected_wrappers),
                    diagnostics,
                    sink,
                )
            )
        elif accounted_raw_ids == connected_ids and (
            available_raw_ids or empty_raw_ids
        ):
            diagnostics = (
                *nonterminal_diagnostics,
                *rejected_wrappers,
                *empty_wrappers,
            )
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.WRAPPER_EMPTY,
                    Stage.WRAPPER,
                    TerminalOutcome.WRAPPER_EMPTY.value,
                    (*base_evidence, *related_wrappers),
                    diagnostics,
                    sink,
                )
            )
        else:
            results.append(
                _make_fate(
                    event,
                    arm.arm_id,
                    TerminalOutcome.UNKNOWN,
                    Stage.WRAPPER,
                    "wrapper_evidence_incomplete",
                    (*base_evidence, *related_wrappers),
                    nonterminal_diagnostics,
                    sink,
                )
            )
    return tuple(results)


def event_is_lexically_usable(
    event: ReferenceEvent,
    outcome: TerminalOutcome,
    alignment: EventAlignment | None,
    policy: UsabilityPolicy,
) -> bool | None:
    """Apply the frozen event-level lexical-usability policy.

    Only an event whose fate is ``wrapper_hypothesis_available`` can be usable,
    so an unknown fate stays unknown and every other fate is unusable whatever
    its alignment says.
    """

    if not isinstance(event, ReferenceEvent):
        raise TypeError("event must be ReferenceEvent")
    if not isinstance(outcome, TerminalOutcome):
        raise TypeError("outcome must be TerminalOutcome")
    if not isinstance(policy, UsabilityPolicy):
        raise TypeError("policy must be UsabilityPolicy")
    if event.intelligibility != "intelligible" or event.verbatim_ref_id is None:
        return None
    if outcome is TerminalOutcome.UNKNOWN:
        return None
    if outcome is not TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE:
        return False
    if alignment is None:
        return None
    if not isinstance(alignment, EventAlignment):
        raise TypeError("alignment must be EventAlignment or None")
    if alignment.event_id != event.event_id:
        raise ValueError("event and alignment IDs must match")
    if not alignment.hypothesis_ids:
        return False
    reference_characters = sum(
        operation.reference_character is not None for operation in alignment.operations
    )
    if reference_characters == 0:
        return False
    errors = alignment.substitutions + alignment.deletions + alignment.insertions
    if reference_characters <= policy.short_reference_max_characters:
        return errors == 0
    if errors * policy.long_cer_denominator > (
        reference_characters * policy.long_cer_numerator
    ):
        return False
    return any(
        operation.kind is EditKind.MATCH
        and operation.reference_character is not None
        and unicodedata.category(operation.reference_character)[0]
        in policy.recovered_category_prefixes
        for operation in alignment.operations
    )


@dataclass(frozen=True, slots=True)
class RateMetric:
    numerator: int
    denominator: int
    unknown: int
    status: MetricStatus
    value: float | None
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True, slots=True)
class UsabilityCurvePoint:
    threshold_numerator: int
    threshold_denominator: int
    recall: RateMetric


@dataclass(frozen=True, slots=True)
class CerMetric:
    substitutions: int
    deletions: int
    insertions: int
    reference_characters: int
    unresolved_events: int
    unresolved_components: int
    unassigned_insertions: int
    status: MetricStatus
    value: float | None


@dataclass(frozen=True, slots=True)
class HallucinationMetric:
    count: int
    control_samples: int
    sample_rate_hz: int | None
    status: MetricStatus
    per_minute: float | None


@dataclass(frozen=True, slots=True)
class LatencySummary:
    survivor_count: int
    unknown_count: int
    p50_seconds: float | None
    p95_seconds: float | None


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    cer_raw: CerMetric
    cer_minimal: CerMetric
    cer_lenient: CerMetric
    cer_words: CerMetric
    wrapper_available_recall: RateMetric
    lexically_usable_recall: RateMetric
    raw_nonblank_recall: RateMetric
    usability_curve: tuple[UsabilityCurvePoint, ...]
    deadline_recall: RateMetric
    vad_duration_miss: RateMetric
    vad_false_alarm: RateMetric
    vad_event_recall: RateMetric
    vad_event_precision: RateMetric
    decoder_window_union_coverage: RateMetric
    boundary_truncation_rate: RateMetric
    merge_rate: RateMetric
    fragmentation_rate: RateMetric
    duplication_rate: RateMetric
    hallucinations: HallucinationMetric
    latency: LatencySummary


@dataclass(frozen=True, slots=True)
class MacroMetric:
    value: float | None
    total_titles: int
    eligible_titles: int
    zero_denominator_titles: int
    incomplete_titles: int
    status: MetricStatus


@dataclass(frozen=True, slots=True)
class TitleMacroMetrics:
    cer_raw: MacroMetric
    cer_minimal: MacroMetric
    cer_lenient: MacroMetric
    cer_words: MacroMetric
    wrapper_available_recall: MacroMetric
    lexically_usable_recall: MacroMetric
    raw_nonblank_recall: MacroMetric
    deadline_recall: MacroMetric
    vad_duration_miss: MacroMetric
    vad_false_alarm: MacroMetric
    vad_event_recall: MacroMetric
    vad_event_precision: MacroMetric
    decoder_window_union_coverage: MacroMetric
    boundary_truncation_rate: MacroMetric
    merge_rate: MacroMetric
    fragmentation_rate: MacroMetric
    duplication_rate: MacroMetric
    hallucinations: MacroMetric


@dataclass(frozen=True, slots=True)
class TitleScore:
    title_id: str
    fixture_count: int
    metrics: AggregateMetrics


@dataclass(frozen=True, slots=True)
class ObservationSummary:
    unit: str
    observed_count: int
    not_run_count: int
    out_of_scope_count: int
    instrumentation_missing_count: int
    record_lost_count: int
    minimum: float | None
    median: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    rtf: ObservationSummary
    cold_start_seconds: ObservationSummary
    peak_rss_bytes: ObservationSummary
    peak_vram_bytes: ObservationSummary


@dataclass(frozen=True, slots=True)
class UnknownCounts:
    event_fates: int
    raw_nonblank: int
    lexical_usability: int
    deadlines: int
    unresolved_alignment_components: int
    instrumentation_missing: int
    record_lost: int


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A 95% title-cluster interval; see ``_bootstrap_intervals`` for the rule."""

    metric_name: str
    lower: float | None
    upper: float | None
    confidence: float
    requested_replicates: int
    valid_replicates: int
    title_count: int
    status: MetricStatus


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    level: NormalizationLevel
    version: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ScoreReport:
    schema_version: str
    scorer_source_sha256: str
    arm_id: str
    partition: Partition
    model_runtime_sha256: str
    fixture_count: int
    title_count: int
    usability_policy_version: str
    usability_policy_sha256: str
    usability_curve_policy_version: str
    usability_curve_policy_sha256: str
    normalization_policies: tuple[PolicyIdentity, ...]
    alignment_policy_version: str
    alignment_policy_sha256: str
    hard_stratum_version: str
    hard_stratum_sha256: str
    hard_event_set_sha256: str
    micro: AggregateMetrics
    title_scores: tuple[TitleScore, ...]
    title_macro: TitleMacroMetrics
    hard_stratum: AggregateMetrics
    resources: ResourceSummary
    unknown_counts: UnknownCounts
    vocal_sounds: VocalSoundCounts
    bootstrap_seed: int
    bootstrap_requested_replicates: int
    bootstrap_unit: str
    bootstrap_quantile_method: str
    bootstrap_intervals: tuple[BootstrapInterval, ...]


def _rate(
    numerator: int,
    denominator: int,
    *,
    unknown: int = 0,
    complete: bool = True,
) -> RateMetric:
    _plain_int(numerator, "numerator")
    _plain_int(denominator, "denominator")
    _plain_int(unknown, "unknown")
    if numerator + unknown > denominator:
        raise ValueError("rate successes plus unknowns exceed the denominator")
    if denominator == 0:
        return RateMetric(
            numerator,
            denominator,
            unknown,
            (
                MetricStatus.ZERO_DENOMINATOR
                if complete and unknown == 0
                else MetricStatus.INCOMPLETE
            ),
            None,
            None,
            None,
        )
    if not complete and unknown == 0:
        return RateMetric(
            numerator,
            denominator,
            unknown,
            MetricStatus.INCOMPLETE,
            None,
            None,
            None,
        )
    lower = numerator / denominator
    upper = (numerator + unknown) / denominator
    return RateMetric(
        numerator,
        denominator,
        unknown,
        MetricStatus.DEFINED if complete and unknown == 0 else MetricStatus.INCOMPLETE,
        lower,
        lower,
        upper,
    )


def percentile_type7(values: Sequence[float], probability: float) -> float | None:
    """Linear (Hyndman-Fan type 7) quantile; ``probability`` lies in [0, 1]."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _merged_intervals(
    spans: Iterable[SampleSpan],
    timeline_id: str,
) -> tuple[tuple[int, int], ...]:
    intervals = sorted(
        (span.start, span.end)
        for span in spans
        if span.timeline_id == timeline_id and span.end > span.start
    )
    if not intervals:
        return ()
    result: list[tuple[int, int]] = []
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            result.append((start, end))
            start, end = next_start, next_end
    result.append((start, end))
    return tuple(result)


def _interval_samples(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_samples(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
) -> int:
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def event_is_eligible(reference: ReferenceEvent, outcome: TerminalOutcome) -> bool:
    """Whether an event enters the lexical recall and CER denominators."""

    return (
        reference.intelligibility == "intelligible"
        and reference.verbatim_ref_id is not None
        and outcome is not TerminalOutcome.REFERENCE_INVALID
    )


def _eligible_event(score: EventScore) -> bool:
    return event_is_eligible(score.reference, score.fate.outcome)


def duplicate_event_ids(
    event_alignments: Iterable[EventAlignment],
    decoder_call_ids: Mapping[str, str],
) -> tuple[str, ...]:
    """Return events whose aligned hypotheses came from two or more decoder calls.

    ``decoder_call_ids`` maps each aligned hypothesis ID to the decoder call
    record that produced it. A line split into segments by one call is not a
    duplicate; the same speech decoded again by another call is.
    """

    duplicated: list[str] = []
    for alignment in event_alignments:
        if not isinstance(alignment, EventAlignment):
            raise TypeError("event_alignments must contain EventAlignment values")
        try:
            calls = {decoder_call_ids[item] for item in alignment.hypothesis_ids}
        except KeyError:
            raise ValueError("an aligned hypothesis has no decoder call") from None
        if len(calls) >= 2:
            duplicated.append(alignment.event_id)
    return tuple(duplicated)


def _profile(fixture: ScoredFixture, level: NormalizationLevel) -> NormalizedAlignmentSet:
    return next(
        alignment
        for alignment in fixture.normalized_alignments
        if alignment.level is level
    )


def _cer_metric(
    fixtures: Sequence[ScoredFixture],
    level: NormalizationLevel,
    hard_ids_by_fixture: dict[str, frozenset[str]] | None = None,
) -> CerMetric:
    substitutions = 0
    deletions = 0
    insertions = 0
    reference_characters = 0
    unresolved_event_ids: set[str] = set()
    unresolved_components = 0
    unassigned_insertions = 0
    for fixture in fixtures:
        profile = _profile(fixture, level)
        hard_ids = (
            None
            if hard_ids_by_fixture is None
            else hard_ids_by_fixture[fixture.fixture.fixture_id]
        )
        eligible_ids = {
            event.reference.event_id
            for event in fixture.events
            if _eligible_event(event)
        }
        selected_ids = (
            eligible_ids
            if hard_ids is None
            else eligible_ids & set(hard_ids)
        )
        include_orphan_components = hard_ids is None or (
            bool(selected_ids) and selected_ids == eligible_ids
        )
        for component in profile.components:
            component_event_ids = set(component.component_event_ids)
            component_selected_ids = component_event_ids & selected_ids
            if component.status is AlignmentStatus.ALIGNMENT_UNRESOLVED:
                if component_selected_ids or (
                    not component_event_ids and include_orphan_components
                ):
                    unresolved_components += 1
                    unresolved_event_ids.update(component_selected_ids)
                continue
            assert component.substitutions is not None
            assert component.deletions is not None
            assert component.insertions is not None
            attributed_insertions = sum(
                alignment.insertions
                for alignment in component.event_alignments
            )
            orphan_insertions = max(
                0,
                component.insertions - attributed_insertions,
            )
            # No-score events are outside lexical alignment; hypotheses they
            # absorbed carry no counts, so only scored events decide ownership.
            scored_event_ids = {
                alignment.event_id for alignment in component.event_alignments
            }
            if not scored_event_ids:
                if include_orphan_components:
                    substitutions += component.substitutions
                    deletions += component.deletions
                    insertions += component.insertions
                else:
                    unassigned_insertions += orphan_insertions
                continue
            if not scored_event_ids & selected_ids:
                continue
            selected = tuple(
                alignment
                for alignment in component.event_alignments
                if alignment.event_id in selected_ids
            )
            if scored_event_ids <= selected_ids:
                substitutions += component.substitutions
                deletions += component.deletions
                insertions += component.insertions
            else:
                substitutions += sum(item.substitutions for item in selected)
                deletions += sum(item.deletions for item in selected)
                insertions += sum(item.insertions for item in selected)
                unassigned_insertions += orphan_insertions
            reference_characters += sum(
                operation.reference_character is not None
                for alignment in selected
                for operation in alignment.operations
            )
    incomplete = bool(unresolved_components or unassigned_insertions)
    if incomplete:
        status = MetricStatus.INCOMPLETE
        value = None
    elif reference_characters == 0:
        status = MetricStatus.ZERO_DENOMINATOR
        value = None
    else:
        status = MetricStatus.DEFINED
        value = (substitutions + deletions + insertions) / reference_characters
    return CerMetric(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_characters=reference_characters,
        unresolved_events=len(unresolved_event_ids),
        unresolved_components=unresolved_components,
        unassigned_insertions=unassigned_insertions,
        status=status,
        value=value,
    )


def _usability_curve(
    fixtures: Sequence[ScoredFixture],
    hard_ids_by_fixture: dict[str, frozenset[str]] | None = None,
) -> tuple[UsabilityCurvePoint, ...]:
    denominator = 0
    unknown = 0
    fixed_successes = 0
    thresholded_errors: list[Fraction] = []
    for fixture in fixtures:
        hard_ids = (
            None
            if hard_ids_by_fixture is None
            else hard_ids_by_fixture[fixture.fixture.fixture_id]
        )
        alignment_by_event: dict[str, EventAlignment | None] = {}
        for component in _profile(fixture, NormalizationLevel.MINIMAL).components:
            if component.status is AlignmentStatus.ALIGNMENT_UNRESOLVED:
                alignment_by_event.update(
                    (event_id, None) for event_id in component.component_event_ids
                )
            else:
                alignment_by_event.update(
                    (alignment.event_id, alignment)
                    for alignment in component.event_alignments
                )
        for event_score in fixture.events:
            if not _eligible_event(event_score) or (
                hard_ids is not None
                and event_score.reference.event_id not in hard_ids
            ):
                continue
            denominator += 1
            if event_score.fate.outcome is TerminalOutcome.UNKNOWN:
                unknown += 1
                continue
            if (
                event_score.fate.outcome
                is not TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
            ):
                continue
            alignment = alignment_by_event[event_score.reference.event_id]
            if alignment is None:
                unknown += 1
                continue
            reference_characters = sum(
                operation.reference_character is not None
                for operation in alignment.operations
            )
            if reference_characters == 0 or not alignment.hypothesis_ids:
                continue
            errors = (
                alignment.substitutions
                + alignment.deletions
                + alignment.insertions
            )
            if (
                reference_characters
                <= JA_EVENT_USABLE_1_1.short_reference_max_characters
            ):
                fixed_successes += errors == 0
                continue
            recovered = any(
                operation.kind is EditKind.MATCH
                and operation.reference_character is not None
                and unicodedata.category(operation.reference_character)[0]
                in JA_EVENT_USABLE_1_1.recovered_category_prefixes
                for operation in alignment.operations
            )
            if recovered:
                thresholded_errors.append(Fraction(errors, reference_characters))
    thresholds = {
        Fraction(0, 1),
        Fraction(
            JA_EVENT_USABLE_1_1.long_cer_numerator,
            JA_EVENT_USABLE_1_1.long_cer_denominator,
        ),
        *thresholded_errors,
    }
    return tuple(
        UsabilityCurvePoint(
            threshold_numerator=threshold.numerator,
            threshold_denominator=threshold.denominator,
            recall=_rate(
                fixed_successes
                + sum(error <= threshold for error in thresholded_errors),
                denominator,
                unknown=unknown,
            ),
        )
        for threshold in sorted(thresholds)
    )


def _aggregate_metrics(
    fixtures: Sequence[ScoredFixture],
    hard_ids_by_fixture: dict[str, frozenset[str]] | None = None,
) -> AggregateMetrics:
    selected_events: list[EventScore] = []
    for fixture in fixtures:
        hard_ids = (
            None
            if hard_ids_by_fixture is None
            else hard_ids_by_fixture[fixture.fixture.fixture_id]
        )
        selected_events.extend(
            event
            for event in fixture.events
            if _eligible_event(event)
            and (hard_ids is None or event.reference.event_id in hard_ids)
        )

    denominator = len(selected_events)
    wrapper_available = _rate(
        sum(
            event.fate.outcome
            is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
            for event in selected_events
        ),
        denominator,
        unknown=sum(
            event.fate.outcome is TerminalOutcome.UNKNOWN
            for event in selected_events
        ),
    )
    lexical = _rate(
        sum(event.lexically_usable is True for event in selected_events),
        denominator,
        unknown=sum(event.lexically_usable is None for event in selected_events),
    )
    raw_nonblank = _rate(
        sum(event.raw_nonblank is True for event in selected_events),
        denominator,
        unknown=sum(event.raw_nonblank is None for event in selected_events),
    )
    deadlines = _rate(
        sum(event.paced_deadline_met is True for event in selected_events),
        denominator,
        unknown=sum(
            event.paced_deadline_met is None for event in selected_events
        ),
    )

    observed_latencies = sorted(
        float(event.output_latency_seconds.value)
        for event in selected_events
        if event.output_latency_seconds.status is ObservationStatus.OBSERVED
        and event.output_latency_seconds.value is not None
    )
    latency = LatencySummary(
        survivor_count=len(observed_latencies),
        unknown_count=sum(
            event.output_latency_seconds.status
            in {
                ObservationStatus.INSTRUMENTATION_MISSING,
                ObservationStatus.RECORD_LOST,
            }
            for event in selected_events
        ),
        p50_seconds=percentile_type7(observed_latencies, 0.50),
        p95_seconds=percentile_type7(observed_latencies, 0.95),
    )

    reference_samples = 0
    vad_missed_samples = 0
    non_speech_samples = 0
    vad_false_alarm_samples = 0
    reference_event_count = 0
    detected_event_count = 0
    predicted_event_count = 0
    true_predicted_event_count = 0
    window_covered_samples = 0
    vad_unknown_reference_samples = 0
    vad_unknown_non_speech_samples = 0
    vad_unknown_events = 0
    vad_precision_complete = True
    window_unknown_reference_samples = 0
    truncated_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    duplicate_unknown_ids: set[str] = set()
    merge_ids: set[str] = set()
    fragment_ids: set[str] = set()
    structural_event_ids: set[str] = set()
    structural_unresolved_ids: set[str] = set()
    hallucination_count = 0
    control_samples = 0
    control_sample_rates: set[int] = set()

    for fixture in fixtures:
        hard_ids = (
            None
            if hard_ids_by_fixture is None
            else hard_ids_by_fixture[fixture.fixture.fixture_id]
        )
        fixture_detection_events = tuple(
            event
            for event in fixture.events
            if hard_ids is None or event.reference.event_id in hard_ids
        )
        selected_ids = {
            event.reference.event_id for event in fixture_detection_events
        }
        structural_event_ids.update(selected_ids)
        reference_intervals = _merged_intervals(
            (event.reference.span for event in fixture_detection_events),
            fixture.timeline_id,
        )
        all_speech_intervals = _merged_intervals(
            (event.reference.span for event in fixture.events),
            fixture.timeline_id,
        )
        reference_duration = _interval_samples(reference_intervals)
        reference_samples += reference_duration
        non_speech_duration = fixture.sample_count - _interval_samples(
            all_speech_intervals
        )
        non_speech_samples += non_speech_duration

        if fixture.vad_observation_status is ObservationStatus.OBSERVED:
            vad_intervals = _merged_intervals(
                fixture.vad_positive_spans,
                fixture.timeline_id,
            )
            covered = _intersection_samples(reference_intervals, vad_intervals)
            all_speech_covered = _intersection_samples(
                all_speech_intervals,
                vad_intervals,
            )
            vad_missed_samples += reference_duration - covered
            vad_false_alarm_samples += (
                _interval_samples(vad_intervals) - all_speech_covered
            )
            reference_event_count += len(fixture_detection_events)
            detected_event_count += sum(
                any(
                    max(event.reference.span.start, start)
                    < min(event.reference.span.end, end)
                    for start, end in vad_intervals
                )
                for event in fixture_detection_events
            )
            predicted_event_count += len(vad_intervals)
            true_predicted_event_count += sum(
                any(
                    max(event.reference.span.start, start)
                    < min(event.reference.span.end, end)
                    for event in fixture.events
                )
                for start, end in vad_intervals
            )
        else:
            vad_precision_complete = False
            vad_unknown_reference_samples += reference_duration
            vad_unknown_non_speech_samples += non_speech_duration
            vad_unknown_events += len(fixture_detection_events)
            reference_event_count += len(fixture_detection_events)

        if fixture.decoder_window_observation_status is ObservationStatus.OBSERVED:
            window_intervals = _merged_intervals(
                fixture.decoder_window_spans,
                fixture.timeline_id,
            )
            window_covered_samples += _intersection_samples(
                reference_intervals,
                window_intervals,
            )
        else:
            window_unknown_reference_samples += reference_duration

        truncated_ids.update(set(fixture.boundary_truncated_event_ids) & selected_ids)
        if fixture.duplicate_observation_status is ObservationStatus.OBSERVED:
            duplicate_ids.update(set(fixture.duplicate_event_ids) & selected_ids)
        else:
            duplicate_unknown_ids.update(selected_ids)
        minimal = _profile(fixture, NormalizationLevel.MINIMAL)
        for component in minimal.components:
            component_selected = set(component.component_event_ids) & selected_ids
            if not component_selected:
                continue
            if component.status is AlignmentStatus.ALIGNMENT_UNRESOLVED:
                structural_unresolved_ids.update(component_selected)
                continue
            for group in component.groups:
                selected_group_events = set(group.event_ids) & selected_ids
                if len(group.event_ids) > 1:
                    merge_ids.update(selected_group_events)
                if len(group.hypothesis_ids) > 1:
                    fragment_ids.update(selected_group_events)

        if hard_ids_by_fixture is None:
            control_duration = _interval_samples(
                _merged_intervals(
                    fixture.non_speech_control_spans,
                    fixture.timeline_id,
                )
            )
            control_samples += control_duration
            if control_duration:
                control_sample_rates.add(fixture.sample_rate_hz)
            hallucination_count += len(fixture.hallucination_output_ids)

    vad_duration_miss = _rate(
        vad_missed_samples,
        reference_samples,
        unknown=vad_unknown_reference_samples,
    )
    vad_false_alarm = _rate(
        vad_false_alarm_samples,
        non_speech_samples,
        unknown=vad_unknown_non_speech_samples,
    )
    vad_event_recall = _rate(
        detected_event_count,
        reference_event_count,
        unknown=vad_unknown_events,
    )
    vad_event_precision = _rate(
        true_predicted_event_count,
        predicted_event_count,
        complete=vad_precision_complete,
    )
    window_coverage = _rate(
        window_covered_samples,
        reference_samples,
        unknown=window_unknown_reference_samples,
    )
    structural_denominator = len(structural_event_ids)
    structural_unknown = len(structural_unresolved_ids)
    truncation = _rate(
        len(truncated_ids),
        structural_denominator,
        unknown=0,
    )
    merging = _rate(
        len(merge_ids),
        structural_denominator,
        unknown=structural_unknown,
    )
    fragmentation = _rate(
        len(fragment_ids),
        structural_denominator,
        unknown=structural_unknown,
    )
    # An unresolved component assigns no hypotheses, so its events' duplicate
    # state is unknown just as their merge and fragment state is.
    duplication = _rate(
        len(duplicate_ids),
        structural_denominator,
        unknown=len(duplicate_unknown_ids | structural_unresolved_ids),
    )

    if hard_ids_by_fixture is not None:
        hallucinations = HallucinationMetric(
            count=0,
            control_samples=0,
            sample_rate_hz=None,
            status=MetricStatus.ZERO_DENOMINATOR,
            per_minute=None,
        )
    elif control_samples == 0:
        hallucinations = HallucinationMetric(
            count=hallucination_count,
            control_samples=0,
            sample_rate_hz=None,
            status=MetricStatus.ZERO_DENOMINATOR,
            per_minute=None,
        )
    elif len(control_sample_rates) != 1:
        hallucinations = HallucinationMetric(
            count=hallucination_count,
            control_samples=control_samples,
            sample_rate_hz=None,
            status=MetricStatus.INCOMPLETE,
            per_minute=None,
        )
    else:
        sample_rate = next(iter(control_sample_rates))
        hallucinations = HallucinationMetric(
            count=hallucination_count,
            control_samples=control_samples,
            sample_rate_hz=sample_rate,
            status=MetricStatus.DEFINED,
            per_minute=(hallucination_count * sample_rate * 60) / control_samples,
        )

    usability_curve = _usability_curve(fixtures, hard_ids_by_fixture)
    primary_curve_point = next(
        point
        for point in usability_curve
        if (
            point.threshold_numerator,
            point.threshold_denominator,
        )
        == (
            JA_EVENT_USABLE_1_1.long_cer_numerator,
            JA_EVENT_USABLE_1_1.long_cer_denominator,
        )
    )
    if (
        primary_curve_point.recall.numerator,
        primary_curve_point.recall.denominator,
        primary_curve_point.recall.unknown,
    ) != (lexical.numerator, lexical.denominator, lexical.unknown):
        raise ValueError(
            "event lexical-usability values do not match minimal alignments"
        )

    return AggregateMetrics(
        cer_raw=_cer_metric(fixtures, NormalizationLevel.RAW, hard_ids_by_fixture),
        cer_minimal=_cer_metric(
            fixtures,
            NormalizationLevel.MINIMAL,
            hard_ids_by_fixture,
        ),
        cer_lenient=_cer_metric(
            fixtures,
            NormalizationLevel.LENIENT,
            hard_ids_by_fixture,
        ),
        cer_words=_cer_metric(
            fixtures,
            NormalizationLevel.WORDS,
            hard_ids_by_fixture,
        ),
        wrapper_available_recall=wrapper_available,
        lexically_usable_recall=lexical,
        raw_nonblank_recall=raw_nonblank,
        usability_curve=usability_curve,
        deadline_recall=deadlines,
        vad_duration_miss=vad_duration_miss,
        vad_false_alarm=vad_false_alarm,
        vad_event_recall=vad_event_recall,
        vad_event_precision=vad_event_precision,
        decoder_window_union_coverage=window_coverage,
        boundary_truncation_rate=truncation,
        merge_rate=merging,
        fragmentation_rate=fragmentation,
        duplication_rate=duplication,
        hallucinations=hallucinations,
        latency=latency,
    )


def _macro_metric(values: Sequence[object]) -> MacroMetric:
    numeric: list[float] = []
    zero_denominator = 0
    incomplete = 0
    for value in values:
        status = value.status
        point = value.value if isinstance(value, (RateMetric, CerMetric)) else value.per_minute
        if status is MetricStatus.ZERO_DENOMINATOR:
            zero_denominator += 1
        elif point is not None:
            numeric.append(float(point))
            if status is MetricStatus.INCOMPLETE:
                incomplete += 1
        else:
            incomplete += 1
    if not numeric:
        status = (
            MetricStatus.ZERO_DENOMINATOR
            if zero_denominator == len(values)
            else MetricStatus.INCOMPLETE
        )
        point = None
    else:
        status = MetricStatus.INCOMPLETE if incomplete else MetricStatus.DEFINED
        point = sum(numeric) / len(numeric)
    return MacroMetric(
        value=point,
        total_titles=len(values),
        eligible_titles=len(numeric),
        zero_denominator_titles=zero_denominator,
        incomplete_titles=incomplete,
        status=status,
    )


def _title_macro(title_scores: Sequence[TitleScore]) -> TitleMacroMetrics:
    def values(name: str) -> tuple[object, ...]:
        return tuple(getattr(title.metrics, name) for title in title_scores)

    return TitleMacroMetrics(
        cer_raw=_macro_metric(values("cer_raw")),
        cer_minimal=_macro_metric(values("cer_minimal")),
        cer_lenient=_macro_metric(values("cer_lenient")),
        cer_words=_macro_metric(values("cer_words")),
        wrapper_available_recall=_macro_metric(
            values("wrapper_available_recall")
        ),
        lexically_usable_recall=_macro_metric(
            values("lexically_usable_recall")
        ),
        raw_nonblank_recall=_macro_metric(values("raw_nonblank_recall")),
        deadline_recall=_macro_metric(values("deadline_recall")),
        vad_duration_miss=_macro_metric(values("vad_duration_miss")),
        vad_false_alarm=_macro_metric(values("vad_false_alarm")),
        vad_event_recall=_macro_metric(values("vad_event_recall")),
        vad_event_precision=_macro_metric(values("vad_event_precision")),
        decoder_window_union_coverage=_macro_metric(
            values("decoder_window_union_coverage")
        ),
        boundary_truncation_rate=_macro_metric(
            values("boundary_truncation_rate")
        ),
        merge_rate=_macro_metric(values("merge_rate")),
        fragmentation_rate=_macro_metric(values("fragmentation_rate")),
        duplication_rate=_macro_metric(values("duplication_rate")),
        hallucinations=_macro_metric(values("hallucinations")),
    )


def _observation_summary(
    observations: Sequence[ScalarObservation],
) -> ObservationSummary:
    if not observations:
        raise ValueError("observation summaries require at least one observation")
    units = {observation.unit for observation in observations}
    if len(units) != 1:
        raise ValueError("observation units must match")
    values = sorted(
        float(observation.value)
        for observation in observations
        if observation.status is ObservationStatus.OBSERVED
        and observation.value is not None
    )
    return ObservationSummary(
        unit=next(iter(units)),
        observed_count=sum(
            observation.status is ObservationStatus.OBSERVED
            for observation in observations
        ),
        not_run_count=sum(
            observation.status is ObservationStatus.NOT_RUN
            for observation in observations
        ),
        out_of_scope_count=sum(
            observation.status is ObservationStatus.OUT_OF_SCOPE
            for observation in observations
        ),
        instrumentation_missing_count=sum(
            observation.status is ObservationStatus.INSTRUMENTATION_MISSING
            for observation in observations
        ),
        record_lost_count=sum(
            observation.status is ObservationStatus.RECORD_LOST
            for observation in observations
        ),
        minimum=values[0] if values else None,
        median=float(median(values)) if values else None,
        maximum=values[-1] if values else None,
    )


def _resource_summary(fixtures: Sequence[ScoredFixture]) -> ResourceSummary:
    return ResourceSummary(
        rtf=_observation_summary(tuple(fixture.resources.rtf for fixture in fixtures)),
        cold_start_seconds=_observation_summary(
            tuple(fixture.resources.cold_start_seconds for fixture in fixtures)
        ),
        peak_rss_bytes=_observation_summary(
            tuple(fixture.resources.peak_rss_bytes for fixture in fixtures)
        ),
        peak_vram_bytes=_observation_summary(
            tuple(fixture.resources.peak_vram_bytes for fixture in fixtures)
        ),
    )


def _unknown_counts(fixtures: Sequence[ScoredFixture]) -> UnknownCounts:
    all_events = tuple(event for fixture in fixtures for event in fixture.events)
    eligible_events = tuple(event for event in all_events if _eligible_event(event))
    all_components = tuple(
        component
        for fixture in fixtures
        for component in _profile(fixture, NormalizationLevel.MINIMAL).components
    )
    scalar_observations = tuple(
        observation
        for fixture in fixtures
        for observation in (
            fixture.resources.rtf,
            fixture.resources.cold_start_seconds,
            fixture.resources.peak_rss_bytes,
            fixture.resources.peak_vram_bytes,
            *(event.output_latency_seconds for event in fixture.events),
        )
    )
    stage_statuses = tuple(
        status
        for fixture in fixtures
        for status in (
            fixture.vad_observation_status,
            fixture.decoder_window_observation_status,
        )
    )
    return UnknownCounts(
        event_fates=sum(
            event.fate.outcome is TerminalOutcome.UNKNOWN for event in all_events
        ),
        raw_nonblank=sum(event.raw_nonblank is None for event in eligible_events),
        lexical_usability=sum(
            event.lexically_usable is None for event in eligible_events
        ),
        deadlines=sum(
            event.paced_deadline_met is None for event in eligible_events
        ),
        unresolved_alignment_components=sum(
            component.status is AlignmentStatus.ALIGNMENT_UNRESOLVED
            for component in all_components
        ),
        instrumentation_missing=sum(
            observation.status is ObservationStatus.INSTRUMENTATION_MISSING
            for observation in scalar_observations
        )
        + sum(
            status is ObservationStatus.INSTRUMENTATION_MISSING
            for status in stage_statuses
        ),
        record_lost=sum(
            observation.status is ObservationStatus.RECORD_LOST
            for observation in scalar_observations
        )
        + sum(
            status is ObservationStatus.RECORD_LOST for status in stage_statuses
        ),
    )


def _vocal_sound_totals(fixtures: Sequence[ScoredFixture]) -> VocalSoundCounts:
    return VocalSoundCounts(
        reference_sound_characters=sum(
            fixture.vocal_sounds.reference_sound_characters for fixture in fixtures
        ),
        output_sound_characters=sum(
            fixture.vocal_sounds.output_sound_characters for fixture in fixtures
        ),
        reference_chains=sum(
            fixture.vocal_sounds.reference_chains for fixture in fixtures
        ),
        output_chains=sum(fixture.vocal_sounds.output_chains for fixture in fixtures),
    )


_BOOTSTRAP_LOWER_TAIL = 0.025
_BOOTSTRAP_UPPER_TAIL = 0.975


def _binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    """Return P(X >= successes) for X ~ Binomial(trials, probability)."""

    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_terms = tuple(
        math.lgamma(trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(trials - count + 1)
        + count * log_p
        + (trials - count) * log_q
        for count in range(successes, trials + 1)
    )
    peak = max(log_terms)
    return math.exp(peak) * math.fsum(math.exp(term - peak) for term in log_terms)


def _clopper_pearson_lower(successes: int, trials: int, tail: float) -> float:
    """Return the exact one-sided binomial lower bound at tail probability ``tail``.

    This is the smallest success probability whose upper tail at the observed
    count reaches ``tail``. Bisection stops at double precision and keeps the
    conservative endpoint.
    """

    if successes == 0:
        return 0.0
    if successes == trials:
        return tail ** (1.0 / trials)
    low = 0.0
    high = successes / trials
    for _ in range(128):
        middle = (low + high) / 2
        if middle in (low, high):
            break
        if _binomial_upper_tail(successes, trials, middle) < tail:
            low = middle
        else:
            high = middle
    return low


def _bootstrap_intervals(
    title_scores: Sequence[TitleScore],
    partition: Partition,
    *,
    seed: int,
    replicates: int,
) -> tuple[BootstrapInterval, ...]:
    """Return 95% intervals that resample whole titles with replacement.

    Each replicate pools the drawn titles' counts; the bounds are the type-7
    2.5% and 97.5% replicate quantiles. A percentile bootstrap over three or
    four titles cannot fall below the worst title's own rate, so for the recall
    rates the reported lower bound is the minimum of that bootstrap bound and
    the exact one-sided Clopper-Pearson bound on the pooled counts of all
    titles, at the same 2.5% tail. CER is not a binomial proportion, so its
    interval stays bootstrap-only. Fewer than two titles, a non-locked
    partition, or zero replicates give ``insufficient_units``.
    """

    metric_names = (
        "cer_minimal",
        "wrapper_available_recall",
        "lexically_usable_recall",
        "raw_nonblank_recall",
        "deadline_recall",
    )
    title_count = len(title_scores)
    if (
        partition is not Partition.LOCKED_CONFIRMATION
        or title_count < 2
        or replicates == 0
    ):
        return tuple(
            BootstrapInterval(
                metric_name=name,
                lower=None,
                upper=None,
                confidence=0.95,
                requested_replicates=replicates,
                valid_replicates=0,
                title_count=title_count,
                status=MetricStatus.INSUFFICIENT_UNITS,
            )
            for name in metric_names
        )
    random_source = random.Random(seed)
    draw_matrix = tuple(
        tuple(random_source.randrange(title_count) for _ in range(title_count))
        for _ in range(replicates)
    )
    intervals: list[BootstrapInterval] = []
    for metric_name in metric_names:
        title_values = tuple(
            getattr(title.metrics, metric_name) for title in title_scores
        )
        if any(value.status is MetricStatus.INCOMPLETE for value in title_values):
            intervals.append(
                BootstrapInterval(
                    metric_name=metric_name,
                    lower=None,
                    upper=None,
                    confidence=0.95,
                    requested_replicates=replicates,
                    valid_replicates=0,
                    title_count=title_count,
                    status=MetricStatus.INCOMPLETE,
                )
            )
            continue
        replicate_values: list[float] = []
        for draw in draw_matrix:
            selected = tuple(
                getattr(title_scores[index].metrics, metric_name) for index in draw
            )
            if isinstance(selected[0], CerMetric):
                cer_values = tuple(
                    value for value in selected if isinstance(value, CerMetric)
                )
                denominator = sum(value.reference_characters for value in cer_values)
                if denominator == 0:
                    continue
                errors = sum(
                    value.substitutions + value.deletions + value.insertions
                    for value in cer_values
                )
                replicate_values.append(errors / denominator)
            else:
                rate_values = tuple(value for value in selected if isinstance(value, RateMetric))
                denominator = sum(value.denominator for value in rate_values)
                if denominator == 0:
                    continue
                replicate_values.append(
                    sum(value.numerator for value in rate_values) / denominator
                )
        valid = len(replicate_values)
        complete_replicates = valid == replicates
        lower = (
            percentile_type7(replicate_values, _BOOTSTRAP_LOWER_TAIL)
            if complete_replicates
            else None
        )
        if lower is not None and isinstance(title_values[0], RateMetric):
            pooled = tuple(
                value for value in title_values if isinstance(value, RateMetric)
            )
            lower = min(
                lower,
                _clopper_pearson_lower(
                    sum(value.numerator for value in pooled),
                    sum(value.denominator for value in pooled),
                    _BOOTSTRAP_LOWER_TAIL,
                ),
            )
        intervals.append(
            BootstrapInterval(
                metric_name=metric_name,
                lower=lower,
                upper=(
                    percentile_type7(replicate_values, _BOOTSTRAP_UPPER_TAIL)
                    if complete_replicates
                    else None
                ),
                confidence=0.95,
                requested_replicates=replicates,
                valid_replicates=valid,
                title_count=title_count,
                status=(
                    MetricStatus.DEFINED
                    if complete_replicates
                    else MetricStatus.INCOMPLETE
                ),
            )
        )
    return tuple(intervals)


def compute_metrics(
    fixtures: Sequence[ScoredFixture],
    *,
    seed: int = 105,
    bootstrap_replicates: int = 10_000,
) -> ScoreReport:
    """Aggregate one arm and partition without dropping misses or unknowns."""

    scored_fixtures = tuple(fixtures)
    if not scored_fixtures:
        raise ValueError("fixtures must not be empty")
    if any(
        not isinstance(fixture, ScoredFixture) for fixture in scored_fixtures
    ):
        raise TypeError("fixtures must contain ScoredFixture values")
    _plain_int(seed, "seed")
    _plain_int(bootstrap_replicates, "bootstrap_replicates")
    arm_ids = {fixture.arm_id for fixture in scored_fixtures}
    fixture_ids = tuple(
        fixture.fixture.fixture_id for fixture in scored_fixtures
    )
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("one score report cannot repeat a fixture")
    event_ids = tuple(
        event.reference.event_id
        for fixture in scored_fixtures
        for event in fixture.events
    )
    hypothesis_ids = tuple(
        hypothesis_id
        for fixture in scored_fixtures
        for hypothesis_id in fixture.expected_wrapper_hypothesis_ids
    )
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("one score report cannot repeat an event ID")
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise ValueError("one score report cannot repeat a hypothesis ID")
    if set(event_ids) & set(hypothesis_ids):
        raise ValueError("report event and hypothesis IDs must be globally distinct")
    partitions = {fixture.fixture.partition for fixture in scored_fixtures}
    model_hashes = {fixture.model_runtime_sha256 for fixture in scored_fixtures}
    if len(arm_ids) != 1:
        raise ValueError("one score report cannot mix arms")
    if len(partitions) != 1:
        raise ValueError("one score report cannot mix partitions")
    if len(model_hashes) != 1:
        raise ValueError("one score report cannot mix model/runtime identities")
    hard_stratum_identities = {
        (fixture.hard_stratum_version, fixture.hard_stratum_sha256)
        for fixture in scored_fixtures
    }
    if len(hard_stratum_identities) != 1:
        raise ValueError("one score report cannot mix hard-stratum definitions")
    alignment_identities = {
        (
            alignment.alignment_policy_version,
            alignment.alignment_policy_sha256,
        )
        for fixture in scored_fixtures
        for alignment in fixture.normalized_alignments
    }
    if len(alignment_identities) != 1:
        raise ValueError("one score report cannot mix alignment policies")
    normalization_identities = {
        level: {
            (
                alignment.normalization_policy_version,
                alignment.normalization_policy_sha256,
            )
            for fixture in scored_fixtures
            for alignment in fixture.normalized_alignments
            if alignment.level is level
        }
        for level in NormalizationLevel
    }
    if any(len(identities) != 1 for identities in normalization_identities.values()):
        raise ValueError("one score report cannot mix normalization policies")

    fixtures_by_title: dict[str, list[ScoredFixture]] = {}
    for fixture in scored_fixtures:
        fixtures_by_title.setdefault(fixture.fixture.title_id, []).append(fixture)
    title_scores = tuple(
        TitleScore(
            title_id=title_id,
            fixture_count=len(title_fixtures),
            metrics=_aggregate_metrics(tuple(title_fixtures)),
        )
        for title_id, title_fixtures in sorted(fixtures_by_title.items())
    )
    hard_ids = {
        fixture.fixture.fixture_id: frozenset(fixture.hard_event_ids)
        for fixture in scored_fixtures
    }
    alignment_version, alignment_sha256 = next(iter(alignment_identities))
    hard_stratum_version, hard_stratum_sha256 = next(
        iter(hard_stratum_identities)
    )
    hard_event_set_sha256 = canonical_sha256(
        {
            "fixtures": tuple(
                {
                    "fixture_id": fixture.fixture.fixture_id,
                    "hard_event_ids": tuple(sorted(fixture.hard_event_ids)),
                }
                for fixture in sorted(
                    scored_fixtures,
                    key=lambda item: item.fixture.fixture_id,
                )
            )
        }
    )
    partition = next(iter(partitions))
    policies = tuple(
        PolicyIdentity(
            level=level,
            version=next(iter(normalization_identities[level]))[0],
            sha256=next(iter(normalization_identities[level]))[1],
        )
        for level in NormalizationLevel
    )
    return ScoreReport(
        schema_version=SCORE_SCHEMA_VERSION,
        arm_id=next(iter(arm_ids)),
        partition=partition,
        model_runtime_sha256=next(iter(model_hashes)),
        fixture_count=len(scored_fixtures),
        title_count=len(title_scores),
        scorer_source_sha256=SCORER_SOURCE_SHA256,
        usability_policy_version=JA_EVENT_USABLE_1_1.version,
        usability_policy_sha256=JA_EVENT_USABLE_1_1.sha256,
        usability_curve_policy_version=USABILITY_CURVE_POLICY_VERSION,
        usability_curve_policy_sha256=USABILITY_CURVE_POLICY_SHA256,
        normalization_policies=policies,
        alignment_policy_version=alignment_version,
        alignment_policy_sha256=alignment_sha256,
        hard_stratum_version=hard_stratum_version,
        hard_stratum_sha256=hard_stratum_sha256,
        hard_event_set_sha256=hard_event_set_sha256,
        micro=_aggregate_metrics(scored_fixtures),
        title_scores=title_scores,
        title_macro=_title_macro(title_scores),
        hard_stratum=_aggregate_metrics(scored_fixtures, hard_ids),
        resources=_resource_summary(scored_fixtures),
        unknown_counts=_unknown_counts(scored_fixtures),
        vocal_sounds=_vocal_sound_totals(scored_fixtures),
        bootstrap_seed=seed,
        bootstrap_requested_replicates=bootstrap_replicates,
        bootstrap_unit="title",
        bootstrap_quantile_method="linear_type7",
        bootstrap_intervals=_bootstrap_intervals(
            title_scores,
            partition,
            seed=seed,
            replicates=bootstrap_replicates,
        ),
    )


__all__ = (
    "JA_EVENT_USABLE_1_1",
    "RAW_UNICODE_1_0_SHA256",
    "RAW_UNICODE_1_0_VERSION",
    "USABILITY_CURVE_POLICY_SHA256",
    "USABILITY_CURVE_POLICY_VERSION",
    "AggregateMetrics",
    "ArmEvidence",
    "BootstrapInterval",
    "CerMetric",
    "EventScore",
    "HallucinationMetric",
    "LatencySummary",
    "MacroMetric",
    "MetricStatus",
    "NormalizationLevel",
    "NormalizedAlignmentSet",
    "ObservationSummary",
    "PolicyIdentity",
    "RateMetric",
    "ResourceObservations",
    "ResourceSummary",
    "SCORER_SOURCE_SHA256",
    "SCORE_SCHEMA_VERSION",
    "ScalarObservation",
    "ScoreReport",
    "ScoredFixture",
    "TitleMacroMetrics",
    "TitleScore",
    "UnknownCounts",
    "UsabilityCurvePoint",
    "UsabilityPolicy",
    "VocalSoundCounts",
    "compute_metrics",
    "derive_event_fates",
    "duplicate_event_ids",
    "event_is_eligible",
    "event_is_lexically_usable",
    "percentile_type7",
    "scored_reference_events",
)
