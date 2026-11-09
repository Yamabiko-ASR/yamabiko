from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import cast

from scripts.e0_evidence._common import (
    exclusive_write_bytes,
    require_sha256,
    require_uuid4,
)
from scripts.e0_evidence.ledger import (
    iter_canonical_records,
    require_complete_run,
    verify_run,
)
from scripts.e0_evidence.private_root import PrivateBoundary, require_private_path
from scripts.e0_evidence.references import ReferenceEvent
from scripts.e0_evidence.registry import load_registered_reference
from scripts.e0_evidence.schema import (
    ArmDefinition,
    ArmKind,
    ArmStatus,
    CapabilityDecision,
    IntegrityReport,
    JsonValue,
    Partition,
    PrivateContentClass,
    RecordKind,
    RunStartManifest,
    TerminalOutcome,
    new_opaque_id,
)
from scripts.e0_evidence.score import (
    AggregateMetrics,
    BootstrapInterval,
    CerMetric,
    HallucinationMetric,
    LatencySummary,
    MacroMetric,
    MetricStatus,
    NormalizationLevel,
    ObservationSummary,
    PolicyIdentity,
    RateMetric,
    ResourceSummary,
    SCORE_SCHEMA_VERSION,
    ScoreReport,
    TitleMacroMetrics,
    TitleScore,
    UnknownCounts,
    UsabilityCurvePoint,
    VocalSoundCounts,
    event_is_eligible,
)
from scripts.e0_evidence.wire import (
    canonical_json_bytes,
    canonical_json_mapping_from_bytes,
    canonical_sha256,
    evidence_record_to_wire,
    inherited_classification,
    run_start_from_wire,
)


PRIVATE_REPORT_SCHEMA_VERSION = "e0-private-report-2.0"
PRIVATE_SCORE_BUNDLE_SCHEMA_VERSION = "e0-private-score-bundle-2.0"
PUBLIC_EXPORT_REVIEW_SCHEMA_VERSION = "e0-public-export-review-1.0"
PUBLIC_AGGREGATE_SCHEMA_VERSION = "e0-public-aggregate-1.0"
PUBLIC_EXPORT_RECEIPT_SCHEMA_VERSION = "e0-public-export-receipt-1.0"
PUBLIC_EXPORT_PROFILE_VERSION = "e0-public-topline-1.1"

_ASCII_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PUBLIC_AGGREGATE_NAME = "aggregate-public.json"
_PUBLIC_RECEIPT_NAME = "aggregate-public.receipt.json"
_PRIVATE_REPORT_NAME = "report.json"
_PRIVATE_MARKDOWN_NAME = "report.md"
_PRIVATE_SCORE_BUNDLE_NAME = "score-bundle.private.json"
_DERIVED_VIEW_NAMES = frozenset({"frames", "windows", "event-fates"})

_CER_PUBLIC_LEAVES = (
    "substitutions",
    "deletions",
    "insertions",
    "reference_characters",
    "unresolved_events",
    "unresolved_components",
    "unassigned_insertions",
    "status",
    "value",
)
_RATE_PUBLIC_LEAVES = (
    "numerator",
    "denominator",
    "unknown",
    "status",
    "value",
    "lower_bound",
    "upper_bound",
)
_MACRO_PUBLIC_LEAVES = (
    "value",
    "total_titles",
    "eligible_titles",
    "zero_denominator_titles",
    "incomplete_titles",
    "status",
)
_UNKNOWN_PUBLIC_LEAVES = (
    "event_fates",
    "raw_nonblank",
    "lexical_usability",
    "deadlines",
    "unresolved_alignment_components",
    "instrumentation_missing",
    "record_lost",
)
_BOOTSTRAP_PUBLIC_LEAVES = (
    "lower",
    "upper",
    "confidence",
    "requested_replicates",
    "valid_replicates",
    "title_count",
    "status",
)
_PUBLIC_RATE_NAMES = (
    "wrapper_available_recall",
    "lexically_usable_recall",
    "raw_nonblank_recall",
    "deadline_recall",
)
_PUBLIC_BOOTSTRAP_NAMES = (
    "cer_minimal",
    *_PUBLIC_RATE_NAMES,
)
_PUBLIC_IDENTITY_FIELDS = (
    "score.schema_version",
    "score.partition",
    "score.model_runtime_sha256",
    "score.fixture_count",
    "score.title_count",
    "score.usability_policy_version",
    "score.usability_policy_sha256",
    "score.usability_curve_policy_version",
    "score.usability_curve_policy_sha256",
    "score.alignment_policy_version",
    "score.alignment_policy_sha256",
    "score.hard_stratum_version",
    "score.hard_stratum_sha256",
    "score.bootstrap_seed",
    "score.bootstrap_requested_replicates",
    "score.bootstrap_unit",
    "score.bootstrap_quantile_method",
)


def _public_field_cap() -> frozenset[str]:
    result = set(_PUBLIC_IDENTITY_FIELDS)
    for level in NormalizationLevel:
        for leaf in ("version", "sha256"):
            result.add(f"score.normalization_policies.{level.value}.{leaf}")
    for scope in ("micro", "hard_stratum"):
        for leaf in _CER_PUBLIC_LEAVES:
            result.add(f"score.{scope}.cer_minimal.{leaf}")
        for metric_name in _PUBLIC_RATE_NAMES:
            for leaf in _RATE_PUBLIC_LEAVES:
                result.add(f"score.{scope}.{metric_name}.{leaf}")
    for metric_name in _PUBLIC_BOOTSTRAP_NAMES:
        for leaf in _MACRO_PUBLIC_LEAVES:
            result.add(f"score.title_macro.{metric_name}.{leaf}")
        for leaf in _BOOTSTRAP_PUBLIC_LEAVES:
            result.add(f"score.bootstrap_intervals.{metric_name}.{leaf}")
    for leaf in _UNKNOWN_PUBLIC_LEAVES:
        result.add(f"score.unknown_counts.{leaf}")
    return frozenset(result)


PUBLIC_EXPORT_FIELD_CAP = _public_field_cap()
PUBLIC_EXPORT_PROFILE_SHA256 = canonical_sha256(
    {
        "field_cap": tuple(sorted(PUBLIC_EXPORT_FIELD_CAP)),
        "semantics": (
            "construct_new_public_object",
            "review_may_approve_only_named_leaf_fields",
            "never_copy_private_report_containers",
            "omit_per_title_content_timing_resource_and_media_identity",
            "label_each_report_by_sanitized_arm_kind",
        ),
        "version": PUBLIC_EXPORT_PROFILE_VERSION,
    }
)


class PrivateReportError(ValueError):
    """A private report could not be validated or written."""


class PublicExportPrivacyError(ValueError):
    """A public export failed closed before private data was written."""


@dataclass(frozen=True, slots=True)
class PublicExportReview:
    review_id: str
    reviewer_id: str
    evidence_ids: tuple[str, ...]
    decision: CapabilityDecision
    source_report_sha256: str
    profile_version: str
    profile_sha256: str
    approved_fields: tuple[str, ...]
    schema_version: str = PUBLIC_EXPORT_REVIEW_SCHEMA_VERSION
    classification: PrivateContentClass = PrivateContentClass.PRIVATE_METADATA

    def __post_init__(self) -> None:
        require_uuid4(self.review_id, "review_id")
        require_uuid4(self.reviewer_id, "reviewer_id")
        evidence_ids = tuple(self.evidence_ids)
        if not evidence_ids:
            raise ValueError("evidence_ids must not be empty")
        for index, evidence_id in enumerate(evidence_ids):
            require_uuid4(evidence_id, f"evidence_ids[{index}]")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        if not isinstance(self.decision, CapabilityDecision):
            raise TypeError("decision must be CapabilityDecision")
        require_sha256(self.source_report_sha256, "source_report_sha256")
        _require_ascii_token(self.profile_version, "profile_version")
        require_sha256(self.profile_sha256, "profile_sha256")
        approved_fields = tuple(self.approved_fields)
        if not approved_fields:
            raise ValueError("approved_fields must not be empty")
        if any(type(field) is not str or not field for field in approved_fields):
            raise TypeError("approved_fields must contain non-empty strings")
        if len(approved_fields) != len(set(approved_fields)):
            raise ValueError("approved_fields must be unique")
        object.__setattr__(self, "approved_fields", tuple(sorted(approved_fields)))
        if self.schema_version != PUBLIC_EXPORT_REVIEW_SCHEMA_VERSION:
            raise ValueError("unsupported public export review schema")
        if self.classification is not PrivateContentClass.PRIVATE_METADATA:
            raise ValueError("public export reviews must remain private metadata")


@dataclass(frozen=True, slots=True)
class PrivateReportArtifacts:
    report_json: Path
    report_markdown: Path
    derived_views: tuple[Path, ...]
    report_sha256: str
    classification: PrivateContentClass


@dataclass(frozen=True, slots=True)
class PublicExportArtifacts:
    aggregate_json: Path
    receipt_json: Path
    aggregate_sha256: str
    receipt_sha256: str
    receipt_id: str


@dataclass(frozen=True, slots=True)
class VerifiedSource:
    run_path: Path
    source_wire: Mapping[str, JsonValue]
    integrity: IntegrityReport
    start: RunStartManifest
    definitions: Mapping[str, ArmDefinition]
    arm_profiles: Mapping[str, str]
    cohort_fixture_ids: tuple[str, ...]
    cohort_id: str
    cohort_manifest_sha256: str
    fixture_id: str
    model_runtime_sha256: str
    partition: Partition


@dataclass(frozen=True, slots=True)
class ReportSet:
    report_id: str
    sources: tuple[VerifiedSource, ...]
    score_sources: tuple[Mapping[str, JsonValue], ...]
    score_reports: tuple[ScoreReport, ...]
    cohort_fixture_ids: tuple[str, ...]
    cohort_id: str
    cohort_manifest_sha256: str
    model_runtime_sha256: str
    partition: Partition
    classification: PrivateContentClass


def _require_ascii_token(value: object, field_name: str) -> str:
    if type(value) is not str or _ASCII_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an ASCII version token")
    return value


def _json_value(
    value: object,
    path: str = "$",
) -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return cast(JsonValue, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return cast(JsonValue, value)
    if isinstance(value, Enum):
        return cast(JsonValue, value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return cast(
            JsonValue,
            {
                item.name: _json_value(getattr(value, item.name), f"{path}.{item.name}")
                for item in fields(value)
            },
        )
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            result[key] = _json_value(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return tuple(
            _json_value(child, f"{path}[{index}]") for index, child in enumerate(value)
        )
    raise TypeError(f"{path} is not JSON serializable")


def score_report_to_private_wire(report: ScoreReport) -> dict[str, JsonValue]:
    if not isinstance(report, ScoreReport):
        raise TypeError("report must be ScoreReport")
    if report.schema_version != SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported score report schema")
    wire = cast(dict[str, JsonValue], _json_value(report, "score"))
    wire["normalization_policies"] = {
        policy.level.value: {
            "version": policy.version,
            "sha256": policy.sha256,
        }
        for policy in report.normalization_policies
    }
    wire["bootstrap_intervals"] = {
        interval.metric_name: {
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence": interval.confidence,
            "requested_replicates": interval.requested_replicates,
            "valid_replicates": interval.valid_replicates,
            "title_count": interval.title_count,
            "status": interval.status.value,
        }
        for interval in report.bootstrap_intervals
    }
    return wire


def _wire_object(
    value: object,
    expected_fields: set[str] | frozenset[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(expected_fields):
        raise PrivateReportError(f"{field_name} has an invalid wire shape")
    return cast(Mapping[str, object], value)


def _wire_sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PrivateReportError(f"{field_name} must be an array")
    return cast(Sequence[object], value)


def _wire_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PrivateReportError(f"{field_name} must be an integer")
    return value


def _wire_optional_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PrivateReportError(f"{field_name} must be finite or null")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise PrivateReportError(f"{field_name} is below its minimum")
    if maximum is not None and numeric > maximum:
        raise PrivateReportError(f"{field_name} is above its maximum")
    return numeric


def _wire_status(value: object, field_name: str) -> MetricStatus:
    try:
        return MetricStatus(value)
    except (TypeError, ValueError):
        raise PrivateReportError(f"{field_name} is not a metric status") from None


def _rate_from_private_wire(value: object, field_name: str) -> RateMetric:
    raw = _wire_object(
        value,
        {item.name for item in fields(RateMetric)},
        field_name,
    )
    numerator = _wire_int(raw["numerator"], f"{field_name}.numerator")
    denominator = _wire_int(raw["denominator"], f"{field_name}.denominator")
    unknown = _wire_int(raw["unknown"], f"{field_name}.unknown")
    if numerator + unknown > denominator:
        raise PrivateReportError(f"{field_name} exceeds its denominator")
    status = _wire_status(raw["status"], f"{field_name}.status")
    metric_value = _wire_optional_number(
        raw["value"],
        f"{field_name}.value",
        maximum=1.0,
    )
    lower = _wire_optional_number(
        raw["lower_bound"],
        f"{field_name}.lower_bound",
        maximum=1.0,
    )
    upper = _wire_optional_number(
        raw["upper_bound"],
        f"{field_name}.upper_bound",
        maximum=1.0,
    )
    if denominator == 0 and metric_value is not None:
        raise PrivateReportError(f"{field_name} has a zero denominator value")
    if lower is not None and upper is not None and lower > upper:
        raise PrivateReportError(f"{field_name} has reversed bounds")
    return RateMetric(
        numerator=numerator,
        denominator=denominator,
        unknown=unknown,
        status=status,
        value=metric_value,
        lower_bound=lower,
        upper_bound=upper,
    )


def _cer_from_private_wire(value: object, field_name: str) -> CerMetric:
    raw = _wire_object(
        value,
        {item.name for item in fields(CerMetric)},
        field_name,
    )
    return CerMetric(
        substitutions=_wire_int(raw["substitutions"], f"{field_name}.substitutions"),
        deletions=_wire_int(raw["deletions"], f"{field_name}.deletions"),
        insertions=_wire_int(raw["insertions"], f"{field_name}.insertions"),
        reference_characters=_wire_int(
            raw["reference_characters"],
            f"{field_name}.reference_characters",
        ),
        unresolved_events=_wire_int(
            raw["unresolved_events"], f"{field_name}.unresolved_events"
        ),
        unresolved_components=_wire_int(
            raw["unresolved_components"],
            f"{field_name}.unresolved_components",
        ),
        unassigned_insertions=_wire_int(
            raw["unassigned_insertions"],
            f"{field_name}.unassigned_insertions",
        ),
        status=_wire_status(raw["status"], f"{field_name}.status"),
        value=_wire_optional_number(raw["value"], f"{field_name}.value"),
    )


def _curve_point_from_private_wire(
    value: object,
    field_name: str,
) -> UsabilityCurvePoint:
    raw = _wire_object(
        value,
        {item.name for item in fields(UsabilityCurvePoint)},
        field_name,
    )
    numerator = _wire_int(
        raw["threshold_numerator"],
        f"{field_name}.threshold_numerator",
    )
    denominator = _wire_int(
        raw["threshold_denominator"],
        f"{field_name}.threshold_denominator",
        minimum=1,
    )
    if numerator > denominator:
        raise PrivateReportError(f"{field_name} threshold exceeds one")
    return UsabilityCurvePoint(
        threshold_numerator=numerator,
        threshold_denominator=denominator,
        recall=_rate_from_private_wire(raw["recall"], f"{field_name}.recall"),
    )


def _hallucination_from_private_wire(
    value: object,
    field_name: str,
) -> HallucinationMetric:
    raw = _wire_object(
        value,
        {item.name for item in fields(HallucinationMetric)},
        field_name,
    )
    sample_rate = raw["sample_rate_hz"]
    if sample_rate is not None:
        sample_rate = _wire_int(
            sample_rate,
            f"{field_name}.sample_rate_hz",
            minimum=1,
        )
    return HallucinationMetric(
        count=_wire_int(raw["count"], f"{field_name}.count"),
        control_samples=_wire_int(
            raw["control_samples"], f"{field_name}.control_samples"
        ),
        sample_rate_hz=cast(int | None, sample_rate),
        status=_wire_status(raw["status"], f"{field_name}.status"),
        per_minute=_wire_optional_number(raw["per_minute"], f"{field_name}.per_minute"),
    )


def _latency_from_private_wire(value: object, field_name: str) -> LatencySummary:
    raw = _wire_object(
        value,
        {item.name for item in fields(LatencySummary)},
        field_name,
    )
    return LatencySummary(
        survivor_count=_wire_int(raw["survivor_count"], f"{field_name}.survivor_count"),
        unknown_count=_wire_int(raw["unknown_count"], f"{field_name}.unknown_count"),
        p50_seconds=_wire_optional_number(
            raw["p50_seconds"], f"{field_name}.p50_seconds"
        ),
        p95_seconds=_wire_optional_number(
            raw["p95_seconds"], f"{field_name}.p95_seconds"
        ),
    )


def _aggregate_from_private_wire(
    value: object,
    field_name: str,
) -> AggregateMetrics:
    raw = _wire_object(
        value,
        {item.name for item in fields(AggregateMetrics)},
        field_name,
    )
    curve = _wire_sequence(raw["usability_curve"], f"{field_name}.usability_curve")
    return AggregateMetrics(
        cer_raw=_cer_from_private_wire(raw["cer_raw"], f"{field_name}.cer_raw"),
        cer_minimal=_cer_from_private_wire(
            raw["cer_minimal"], f"{field_name}.cer_minimal"
        ),
        cer_lenient=_cer_from_private_wire(
            raw["cer_lenient"], f"{field_name}.cer_lenient"
        ),
        cer_words=_cer_from_private_wire(
            raw["cer_words"], f"{field_name}.cer_words"
        ),
        wrapper_available_recall=_rate_from_private_wire(
            raw["wrapper_available_recall"],
            f"{field_name}.wrapper_available_recall",
        ),
        lexically_usable_recall=_rate_from_private_wire(
            raw["lexically_usable_recall"],
            f"{field_name}.lexically_usable_recall",
        ),
        raw_nonblank_recall=_rate_from_private_wire(
            raw["raw_nonblank_recall"], f"{field_name}.raw_nonblank_recall"
        ),
        usability_curve=tuple(
            _curve_point_from_private_wire(
                point,
                f"{field_name}.usability_curve[{index}]",
            )
            for index, point in enumerate(curve)
        ),
        deadline_recall=_rate_from_private_wire(
            raw["deadline_recall"], f"{field_name}.deadline_recall"
        ),
        vad_duration_miss=_rate_from_private_wire(
            raw["vad_duration_miss"], f"{field_name}.vad_duration_miss"
        ),
        vad_false_alarm=_rate_from_private_wire(
            raw["vad_false_alarm"], f"{field_name}.vad_false_alarm"
        ),
        vad_event_recall=_rate_from_private_wire(
            raw["vad_event_recall"], f"{field_name}.vad_event_recall"
        ),
        vad_event_precision=_rate_from_private_wire(
            raw["vad_event_precision"], f"{field_name}.vad_event_precision"
        ),
        decoder_window_union_coverage=_rate_from_private_wire(
            raw["decoder_window_union_coverage"],
            f"{field_name}.decoder_window_union_coverage",
        ),
        boundary_truncation_rate=_rate_from_private_wire(
            raw["boundary_truncation_rate"],
            f"{field_name}.boundary_truncation_rate",
        ),
        merge_rate=_rate_from_private_wire(
            raw["merge_rate"], f"{field_name}.merge_rate"
        ),
        fragmentation_rate=_rate_from_private_wire(
            raw["fragmentation_rate"], f"{field_name}.fragmentation_rate"
        ),
        duplication_rate=_rate_from_private_wire(
            raw["duplication_rate"], f"{field_name}.duplication_rate"
        ),
        hallucinations=_hallucination_from_private_wire(
            raw["hallucinations"], f"{field_name}.hallucinations"
        ),
        latency=_latency_from_private_wire(raw["latency"], f"{field_name}.latency"),
    )


def _macro_from_private_wire(value: object, field_name: str) -> MacroMetric:
    raw = _wire_object(
        value,
        {item.name for item in fields(MacroMetric)},
        field_name,
    )
    return MacroMetric(
        value=_wire_optional_number(raw["value"], f"{field_name}.value"),
        total_titles=_wire_int(raw["total_titles"], f"{field_name}.total_titles"),
        eligible_titles=_wire_int(
            raw["eligible_titles"], f"{field_name}.eligible_titles"
        ),
        zero_denominator_titles=_wire_int(
            raw["zero_denominator_titles"],
            f"{field_name}.zero_denominator_titles",
        ),
        incomplete_titles=_wire_int(
            raw["incomplete_titles"], f"{field_name}.incomplete_titles"
        ),
        status=_wire_status(raw["status"], f"{field_name}.status"),
    )


def _title_macro_from_private_wire(
    value: object,
    field_name: str,
) -> TitleMacroMetrics:
    expected = {item.name for item in fields(TitleMacroMetrics)}
    raw = _wire_object(value, expected, field_name)
    return TitleMacroMetrics(
        **{
            name: _macro_from_private_wire(raw[name], f"{field_name}.{name}")
            for name in sorted(expected)
        }
    )


def _observation_from_private_wire(
    value: object,
    field_name: str,
) -> ObservationSummary:
    raw = _wire_object(
        value,
        {item.name for item in fields(ObservationSummary)},
        field_name,
    )
    unit = raw["unit"]
    try:
        unit = _require_ascii_token(unit, f"{field_name}.unit")
    except (TypeError, ValueError):
        raise PrivateReportError(f"{field_name}.unit is invalid") from None
    return ObservationSummary(
        unit=unit,
        observed_count=_wire_int(raw["observed_count"], f"{field_name}.observed_count"),
        not_run_count=_wire_int(raw["not_run_count"], f"{field_name}.not_run_count"),
        out_of_scope_count=_wire_int(
            raw["out_of_scope_count"], f"{field_name}.out_of_scope_count"
        ),
        instrumentation_missing_count=_wire_int(
            raw["instrumentation_missing_count"],
            f"{field_name}.instrumentation_missing_count",
        ),
        record_lost_count=_wire_int(
            raw["record_lost_count"], f"{field_name}.record_lost_count"
        ),
        minimum=_wire_optional_number(raw["minimum"], f"{field_name}.minimum"),
        median=_wire_optional_number(raw["median"], f"{field_name}.median"),
        maximum=_wire_optional_number(raw["maximum"], f"{field_name}.maximum"),
    )


def _resource_from_private_wire(value: object, field_name: str) -> ResourceSummary:
    expected = {item.name for item in fields(ResourceSummary)}
    raw = _wire_object(value, expected, field_name)
    return ResourceSummary(
        **{
            name: _observation_from_private_wire(raw[name], f"{field_name}.{name}")
            for name in sorted(expected)
        }
    )


def _unknown_from_private_wire(value: object, field_name: str) -> UnknownCounts:
    expected = {item.name for item in fields(UnknownCounts)}
    raw = _wire_object(value, expected, field_name)
    return UnknownCounts(
        **{
            name: _wire_int(raw[name], f"{field_name}.{name}")
            for name in sorted(expected)
        }
    )


def _vocal_sounds_from_private_wire(
    value: object,
    field_name: str,
) -> VocalSoundCounts:
    expected = {item.name for item in fields(VocalSoundCounts)}
    raw = _wire_object(value, expected, field_name)
    return VocalSoundCounts(
        **{
            name: _wire_int(raw[name], f"{field_name}.{name}")
            for name in sorted(expected)
        }
    )


def _title_score_from_private_wire(value: object, field_name: str) -> TitleScore:
    raw = _wire_object(
        value,
        {item.name for item in fields(TitleScore)},
        field_name,
    )
    try:
        title_id = require_uuid4(raw["title_id"], f"{field_name}.title_id")
    except (TypeError, ValueError):
        raise PrivateReportError(f"{field_name}.title_id is invalid") from None
    return TitleScore(
        title_id=title_id,
        fixture_count=_wire_int(
            raw["fixture_count"], f"{field_name}.fixture_count", minimum=1
        ),
        metrics=_aggregate_from_private_wire(raw["metrics"], f"{field_name}.metrics"),
    )


def _policy_identities_from_private_wire(
    value: object,
) -> tuple[PolicyIdentity, ...]:
    expected = {level.value for level in NormalizationLevel}
    raw = _wire_object(value, expected, "score.normalization_policies")
    result: list[PolicyIdentity] = []
    for level in NormalizationLevel:
        policy = _wire_object(
            raw[level.value],
            {"version", "sha256"},
            f"score.normalization_policies.{level.value}",
        )
        try:
            version = _require_ascii_token(
                policy["version"],
                f"score.normalization_policies.{level.value}.version",
            )
            sha256 = require_sha256(
                policy["sha256"],
                f"score.normalization_policies.{level.value}.sha256",
            )
        except (TypeError, ValueError):
            raise PrivateReportError(
                "normalization policy identity is invalid"
            ) from None
        result.append(PolicyIdentity(level=level, version=version, sha256=sha256))
    return tuple(result)


def _bootstrap_from_private_wire(
    value: object,
    requested_replicates: int,
) -> tuple[BootstrapInterval, ...]:
    expected = set(_PUBLIC_BOOTSTRAP_NAMES)
    raw = _wire_object(value, expected, "score.bootstrap_intervals")
    intervals: list[BootstrapInterval] = []
    expected_fields = {item.name for item in fields(BootstrapInterval)} - {
        "metric_name"
    }
    for metric_name in _PUBLIC_BOOTSTRAP_NAMES:
        interval = _wire_object(
            raw[metric_name],
            expected_fields,
            f"score.bootstrap_intervals.{metric_name}",
        )
        requested = _wire_int(
            interval["requested_replicates"],
            f"score.bootstrap_intervals.{metric_name}.requested_replicates",
        )
        valid = _wire_int(
            interval["valid_replicates"],
            f"score.bootstrap_intervals.{metric_name}.valid_replicates",
        )
        if requested != requested_replicates or valid > requested:
            raise PrivateReportError("bootstrap replicate accounting is invalid")
        intervals.append(
            BootstrapInterval(
                metric_name=metric_name,
                lower=_wire_optional_number(
                    interval["lower"],
                    f"score.bootstrap_intervals.{metric_name}.lower",
                ),
                upper=_wire_optional_number(
                    interval["upper"],
                    f"score.bootstrap_intervals.{metric_name}.upper",
                ),
                confidence=cast(
                    float,
                    _wire_optional_number(
                        interval["confidence"],
                        f"score.bootstrap_intervals.{metric_name}.confidence",
                        maximum=1.0,
                    ),
                ),
                requested_replicates=requested,
                valid_replicates=valid,
                title_count=_wire_int(
                    interval["title_count"],
                    f"score.bootstrap_intervals.{metric_name}.title_count",
                ),
                status=_wire_status(
                    interval["status"],
                    f"score.bootstrap_intervals.{metric_name}.status",
                ),
            )
        )
    if any(interval.confidence is None for interval in intervals):
        raise PrivateReportError("bootstrap confidence is required")
    return tuple(intervals)


def score_report_from_private_wire(value: object) -> ScoreReport:
    expected = {item.name for item in fields(ScoreReport)}
    raw = _wire_object(value, expected, "score")
    if raw["schema_version"] != SCORE_SCHEMA_VERSION:
        raise PrivateReportError("unsupported score report schema")
    try:
        arm_id = require_uuid4(raw["arm_id"], "score.arm_id")
        scorer_hash = require_sha256(
            raw["scorer_source_sha256"], "score.scorer_source_sha256"
        )
        partition = Partition(raw["partition"])
        model_hash = require_sha256(
            raw["model_runtime_sha256"], "score.model_runtime_sha256"
        )
        usability_version = _require_ascii_token(
            raw["usability_policy_version"], "score.usability_policy_version"
        )
        usability_hash = require_sha256(
            raw["usability_policy_sha256"], "score.usability_policy_sha256"
        )
        curve_version = _require_ascii_token(
            raw["usability_curve_policy_version"],
            "score.usability_curve_policy_version",
        )
        curve_hash = require_sha256(
            raw["usability_curve_policy_sha256"],
            "score.usability_curve_policy_sha256",
        )
        alignment_version = _require_ascii_token(
            raw["alignment_policy_version"], "score.alignment_policy_version"
        )
        alignment_hash = require_sha256(
            raw["alignment_policy_sha256"], "score.alignment_policy_sha256"
        )
        hard_version = _require_ascii_token(
            raw["hard_stratum_version"], "score.hard_stratum_version"
        )
        hard_hash = require_sha256(
            raw["hard_stratum_sha256"], "score.hard_stratum_sha256"
        )
        hard_event_hash = require_sha256(
            raw["hard_event_set_sha256"], "score.hard_event_set_sha256"
        )
    except (TypeError, ValueError):
        raise PrivateReportError("score identity is invalid") from None
    fixture_count = _wire_int(raw["fixture_count"], "score.fixture_count", minimum=1)
    title_count = _wire_int(raw["title_count"], "score.title_count", minimum=1)
    raw_titles = _wire_sequence(raw["title_scores"], "score.title_scores")
    title_scores = tuple(
        _title_score_from_private_wire(title, f"score.title_scores[{index}]")
        for index, title in enumerate(raw_titles)
    )
    if (
        len(title_scores) != title_count
        or len({title.title_id for title in title_scores}) != title_count
        or sum(title.fixture_count for title in title_scores) != fixture_count
    ):
        raise PrivateReportError("score title accounting is invalid")
    bootstrap_seed = _wire_int(raw["bootstrap_seed"], "score.bootstrap_seed")
    bootstrap_requested = _wire_int(
        raw["bootstrap_requested_replicates"],
        "score.bootstrap_requested_replicates",
    )
    if raw["bootstrap_unit"] != "title":
        raise PrivateReportError("score bootstrap unit is invalid")
    if raw["bootstrap_quantile_method"] != "linear_type7":
        raise PrivateReportError("score bootstrap method is invalid")
    return ScoreReport(
        schema_version=SCORE_SCHEMA_VERSION,
        scorer_source_sha256=scorer_hash,
        arm_id=arm_id,
        partition=partition,
        model_runtime_sha256=model_hash,
        fixture_count=fixture_count,
        title_count=title_count,
        usability_policy_version=usability_version,
        usability_policy_sha256=usability_hash,
        usability_curve_policy_version=curve_version,
        usability_curve_policy_sha256=curve_hash,
        normalization_policies=_policy_identities_from_private_wire(
            raw["normalization_policies"]
        ),
        alignment_policy_version=alignment_version,
        alignment_policy_sha256=alignment_hash,
        hard_stratum_version=hard_version,
        hard_stratum_sha256=hard_hash,
        hard_event_set_sha256=hard_event_hash,
        micro=_aggregate_from_private_wire(raw["micro"], "score.micro"),
        title_scores=title_scores,
        title_macro=_title_macro_from_private_wire(
            raw["title_macro"], "score.title_macro"
        ),
        hard_stratum=_aggregate_from_private_wire(
            raw["hard_stratum"], "score.hard_stratum"
        ),
        resources=_resource_from_private_wire(raw["resources"], "score.resources"),
        unknown_counts=_unknown_from_private_wire(
            raw["unknown_counts"], "score.unknown_counts"
        ),
        vocal_sounds=_vocal_sounds_from_private_wire(
            raw["vocal_sounds"], "score.vocal_sounds"
        ),
        bootstrap_seed=bootstrap_seed,
        bootstrap_requested_replicates=bootstrap_requested,
        bootstrap_unit="title",
        bootstrap_quantile_method="linear_type7",
        bootstrap_intervals=_bootstrap_from_private_wire(
            raw["bootstrap_intervals"],
            bootstrap_requested,
        ),
    )


def _integrity_to_wire(report: IntegrityReport) -> dict[str, JsonValue]:
    if not isinstance(report, IntegrityReport):
        raise TypeError("report must be IntegrityReport")
    return cast(dict[str, JsonValue], _json_value(report, "integrity"))


def _private_report_classification(
    integrities: Sequence[IntegrityReport],
) -> PrivateContentClass:
    classification = PrivateContentClass.INTERNAL_AGGREGATE
    for integrity in integrities:
        for stream in integrity.computed_streams:
            classification = inherited_classification(
                classification,
                (stream.classification,),
            )
    return classification


def _report_directory(report_dir: Path, boundary: PrivateBoundary) -> Path:
    report_path = require_private_path(Path(report_dir), boundary)
    try:
        relative = report_path.relative_to(boundary.root)
    except ValueError:
        raise PrivateReportError(
            "report directory is outside the private root"
        ) from None
    if len(relative.parts) != 2 or relative.parts[0] != "reports":
        raise PrivateReportError("report directory must be reports/<report_id>")
    try:
        require_uuid4(relative.parts[1], "report_id")
    except (TypeError, ValueError):
        raise PrivateReportError("report directory has an invalid report_id") from None
    return report_path


def _run_directory(run_dir: Path, boundary: PrivateBoundary) -> Path:
    run_path = require_private_path(Path(run_dir), boundary)
    try:
        relative = run_path.relative_to(boundary.root)
    except ValueError:
        raise PrivateReportError("source run is outside the private root") from None
    if len(relative.parts) != 2 or relative.parts[0] != "runs":
        raise PrivateReportError("source run must be runs/<run_id>")
    try:
        require_uuid4(relative.parts[1], "source run_id")
    except (TypeError, ValueError):
        raise PrivateReportError("source run directory has an invalid run_id") from None
    return run_path


def _manifest_uuid(
    details: Mapping[str, object],
    field_name: str,
) -> str:
    try:
        return require_uuid4(details.get(field_name), field_name)
    except (TypeError, ValueError):
        raise PrivateReportError(f"run start {field_name} is invalid") from None


def _manifest_sha256(
    details: Mapping[str, object],
    field_name: str,
) -> str:
    try:
        return require_sha256(details.get(field_name), field_name)
    except (TypeError, ValueError):
        raise PrivateReportError(f"run start {field_name} is invalid") from None


def _manifest_partition(details: Mapping[str, object]) -> Partition:
    try:
        return Partition(details.get("partition"))
    except (TypeError, ValueError):
        raise PrivateReportError("run start partition is invalid") from None


def _verify_source(run_dir: Path, boundary: PrivateBoundary) -> VerifiedSource:
    run_path = _run_directory(run_dir, boundary)
    start_path = require_private_path(run_path / "run-start.json", boundary)
    final_path = require_private_path(run_path / "run-final.json", boundary)
    try:
        start_before = start_path.read_bytes()
        final_before = final_path.read_bytes()
    except OSError:
        raise PrivateReportError("source run manifests are unreadable") from None
    try:
        require_complete_run(run_path)
        integrity = verify_run(run_path)
    except Exception:
        raise PrivateReportError("source run is not complete and verified") from None
    if not integrity.complete or integrity.run_id is None:
        raise PrivateReportError("source run is not complete and verified")
    try:
        start_raw = start_path.read_bytes()
        final_raw = final_path.read_bytes()
    except OSError:
        raise PrivateReportError("source run manifests are unreadable") from None
    if start_raw != start_before or final_raw != final_before:
        raise PrivateReportError("source run manifests changed during verification")
    try:
        start_payload = canonical_json_mapping_from_bytes(start_raw, "run start")
        start = run_start_from_wire(start_payload)
    except (TypeError, ValueError, UnicodeError):
        raise PrivateReportError("verified run start is invalid") from None
    if start.run_id != integrity.run_id or start.run_id != run_path.name:
        raise PrivateReportError("source run identity does not match its directory")

    details = cast(Mapping[str, object], start.details)
    cohort_id = _manifest_uuid(details, "cohort_id")
    cohort_manifest_sha256 = _manifest_sha256(
        details,
        "cohort_manifest_sha256",
    )
    fixture_id = _manifest_uuid(details, "fixture_id")
    model_runtime_sha256 = _manifest_sha256(details, "model_runtime_sha256")
    partition = _manifest_partition(details)

    raw_fixture_ids = details.get("cohort_fixture_ids")
    if not isinstance(raw_fixture_ids, Sequence) or isinstance(
        raw_fixture_ids,
        (str, bytes),
    ):
        raise PrivateReportError("run start cohort_fixture_ids is invalid")
    try:
        cohort_fixture_ids = tuple(
            require_uuid4(item, f"cohort_fixture_ids[{index}]")
            for index, item in enumerate(raw_fixture_ids)
        )
    except (TypeError, ValueError):
        raise PrivateReportError("run start cohort_fixture_ids is invalid") from None
    if (
        not cohort_fixture_ids
        or len(cohort_fixture_ids) != len(set(cohort_fixture_ids))
        or cohort_fixture_ids != tuple(sorted(cohort_fixture_ids))
        or fixture_id not in cohort_fixture_ids
    ):
        raise PrivateReportError("run start cohort fixture accounting is invalid")

    definitions = {
        definition.arm_id: definition for definition in start.arm_definitions
    }
    raw_profiles = details.get("arm_profile_sha256")
    if not isinstance(raw_profiles, Mapping) or set(raw_profiles) != set(definitions):
        raise PrivateReportError("run start arm profile map is invalid")
    arm_profiles: dict[str, str] = {}
    for arm_id, definition in definitions.items():
        try:
            profile_sha256 = require_sha256(
                raw_profiles[arm_id],
                f"arm_profile_sha256.{arm_id}",
            )
        except (KeyError, TypeError, ValueError):
            raise PrivateReportError("run start arm profile map is invalid") from None
        parameters = definition.parameters
        if (
            parameters.get("arm_profile_sha256") != profile_sha256
            or parameters.get("cohort_id") != cohort_id
            or parameters.get("cohort_manifest_sha256") != cohort_manifest_sha256
            or parameters.get("model_runtime_sha256") != model_runtime_sha256
        ):
            raise PrivateReportError("arm definition provenance is inconsistent")
        arm_profiles[arm_id] = profile_sha256

    source_wire: dict[str, JsonValue] = {
        "run_id": start.run_id,
        "run_start_sha256": hashlib.sha256(start_raw).hexdigest(),
        "run_final_sha256": hashlib.sha256(final_raw).hexdigest(),
        "cohort_id": cohort_id,
        "cohort_manifest_sha256": cohort_manifest_sha256,
        "fixture_id": fixture_id,
    }
    return VerifiedSource(
        run_path=run_path,
        source_wire=source_wire,
        integrity=integrity,
        start=start,
        definitions=definitions,
        arm_profiles=arm_profiles,
        cohort_fixture_ids=cohort_fixture_ids,
        cohort_id=cohort_id,
        cohort_manifest_sha256=cohort_manifest_sha256,
        fixture_id=fixture_id,
        model_runtime_sha256=model_runtime_sha256,
        partition=partition,
    )


def verify_report_sources(
    run_dirs: Sequence[Path],
    boundary: PrivateBoundary,
) -> tuple[VerifiedSource, ...]:
    """Verify each complete source run once and return them in run-ID order."""

    if isinstance(run_dirs, (str, bytes, Path)):
        raise TypeError("run_dirs must be a sequence of Path values")
    sources = tuple(_verify_source(Path(run_dir), boundary) for run_dir in run_dirs)
    if not sources:
        raise ValueError("run_dirs must not be empty")
    run_ids = tuple(cast(str, source.source_wire["run_id"]) for source in sources)
    if len(run_ids) != len(set(run_ids)):
        raise PrivateReportError("report sources must not repeat a run")
    return tuple(
        sorted(sources, key=lambda source: cast(str, source.source_wire["run_id"]))
    )


def _normalize_score_sources(
    score_sources: Mapping[str, Sequence[str]],
    reports: Sequence[ScoreReport],
    sources: Sequence[VerifiedSource],
) -> tuple[Mapping[str, JsonValue], ...]:
    if not isinstance(score_sources, Mapping):
        raise TypeError("score_sources must map arm IDs to source run IDs")
    report_by_arm = {report.arm_id: report for report in reports}
    if len(report_by_arm) != len(reports):
        raise ValueError("score_reports must not repeat an arm")
    if set(score_sources) != set(report_by_arm):
        raise PrivateReportError("score source arms do not match score reports")
    source_by_run = {
        cast(str, source.source_wire["run_id"]): source for source in sources
    }
    used_run_ids: set[str] = set()
    normalized: list[Mapping[str, JsonValue]] = []
    for arm_id in sorted(report_by_arm):
        try:
            require_uuid4(arm_id, "score_sources arm_id")
        except (TypeError, ValueError):
            raise PrivateReportError("score source arm ID is invalid") from None
        raw_run_ids = score_sources[arm_id]
        if not isinstance(raw_run_ids, Sequence) or isinstance(
            raw_run_ids,
            (str, bytes),
        ):
            raise TypeError("score source run IDs must be a sequence")
        try:
            run_ids = tuple(
                require_uuid4(item, f"score_sources.{arm_id}[{index}]")
                for index, item in enumerate(raw_run_ids)
            )
        except (TypeError, ValueError):
            raise PrivateReportError("score source run ID is invalid") from None
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise PrivateReportError("score source run IDs must be nonempty and unique")
        run_ids = tuple(sorted(run_ids))
        if set(run_ids) != set(source_by_run):
            raise PrivateReportError(
                "every score arm must use every frozen cohort source run"
            )
        try:
            mapped_sources = tuple(source_by_run[run_id] for run_id in run_ids)
        except KeyError:
            raise PrivateReportError("score source names an unbound run") from None

        report = report_by_arm[arm_id]
        if report.fixture_count != len(mapped_sources):
            raise PrivateReportError("score fixture count does not match its sources")
        invariants: set[tuple[object, ...]] = set()
        for source in mapped_sources:
            definition = source.definitions.get(arm_id)
            completed = {
                status.arm_id
                for status in source.integrity.computed_arm_statuses
                if status.status is ArmStatus.COMPLETE
            }
            if definition is None or arm_id not in completed:
                raise PrivateReportError(
                    "mapped score arm is not declared and complete in every source"
                )
            invariants.add(
                (
                    source.arm_profiles[arm_id],
                    definition.kind,
                    definition.parent_arm_id,
                    source.model_runtime_sha256,
                    source.partition,
                    source.cohort_id,
                    source.cohort_manifest_sha256,
                )
            )
        if len(invariants) != 1:
            raise PrivateReportError("mapped score arm provenance is not invariant")
        first = mapped_sources[0]
        if (
            report.model_runtime_sha256 != first.model_runtime_sha256
            or report.partition is not first.partition
        ):
            raise PrivateReportError("score report model or partition is inconsistent")
        used_run_ids.update(run_ids)
        normalized.append(
            {
                "arm_id": arm_id,
                "arm_kind": first.definitions[arm_id].kind.value,
                "source_run_ids": run_ids,
            }
        )
    if used_run_ids != set(source_by_run):
        raise PrivateReportError("every report source must contribute to a score")
    return tuple(normalized)


def _validate_report_set(
    report_id: str,
    sources: Sequence[VerifiedSource],
    reports: Sequence[ScoreReport],
    score_sources: Mapping[str, Sequence[str]],
) -> ReportSet:
    require_uuid4(report_id, "report_id")
    ordered_sources = tuple(sources)
    unordered_reports = tuple(reports)
    if not ordered_sources:
        raise ValueError("report sources must not be empty")
    if not unordered_reports:
        raise ValueError("score_reports must not be empty")
    if any(not isinstance(report, ScoreReport) for report in unordered_reports):
        raise TypeError("score_reports must contain ScoreReport values")
    ordered_reports = tuple(sorted(unordered_reports, key=lambda report: report.arm_id))

    first = ordered_sources[0]
    identity = (
        first.cohort_id,
        first.cohort_manifest_sha256,
        first.model_runtime_sha256,
        first.partition,
        first.cohort_fixture_ids,
    )
    if any(
        (
            source.cohort_id,
            source.cohort_manifest_sha256,
            source.model_runtime_sha256,
            source.partition,
            source.cohort_fixture_ids,
        )
        != identity
        for source in ordered_sources
    ):
        raise PrivateReportError(
            "a report bundle must contain one cohort, model, and partition"
        )
    fixture_ids = tuple(source.fixture_id for source in ordered_sources)
    if len(fixture_ids) != len(set(fixture_ids)) or set(fixture_ids) != set(
        first.cohort_fixture_ids
    ):
        raise PrivateReportError(
            "every expected cohort fixture must be represented exactly once"
        )
    normalized_score_sources = _normalize_score_sources(
        score_sources,
        ordered_reports,
        ordered_sources,
    )
    classification = _private_report_classification(
        tuple(source.integrity for source in ordered_sources)
    )
    return ReportSet(
        report_id=report_id,
        sources=ordered_sources,
        score_sources=normalized_score_sources,
        score_reports=ordered_reports,
        cohort_fixture_ids=first.cohort_fixture_ids,
        cohort_id=first.cohort_id,
        cohort_manifest_sha256=first.cohort_manifest_sha256,
        model_runtime_sha256=first.model_runtime_sha256,
        partition=first.partition,
        classification=classification,
    )


def _require_ledger_fate_accounting(
    report_set: ReportSet,
    boundary: PrivateBoundary,
) -> None:
    """Recount each scored arm's fates from its source ledgers and references.

    Wrapper-available recall, the unknown-fate count, and the fixture count must
    equal what the bound ledgers record against each run's registered reference
    manifest, so no report can credit an event its ledger did not.
    """

    events_by_run: dict[str, tuple[ReferenceEvent, ...]] = {}
    outcomes_by_run: dict[str, dict[tuple[str, str], TerminalOutcome]] = {}
    for source in report_set.sources:
        details = cast(Mapping[str, object], source.start.details)
        reference_set_id = _manifest_uuid(details, "reference_set_id")
        reference_set_sha256 = _manifest_sha256(details, "reference_set_sha256")
        try:
            manifest = load_registered_reference(reference_set_id, boundary)
        except (OSError, TypeError, ValueError):
            raise PrivateReportError("source reference set is unavailable") from None
        if (
            manifest.reference_set_sha256 != reference_set_sha256
            or manifest.fixture_id != source.fixture_id
        ):
            raise PrivateReportError("source reference set changed after its run")
        outcomes: dict[tuple[str, str], TerminalOutcome] = {}
        for record in iter_canonical_records(source.run_path):
            if record.kind is not RecordKind.EVENT_FATE or record.arm_id is None:
                continue
            key = (record.arm_id, record.artifact_id)
            try:
                outcome = TerminalOutcome(record.decision)
            except ValueError:
                raise PrivateReportError("source event fate is invalid") from None
            if key in outcomes:
                raise PrivateReportError("source event fate is repeated")
            outcomes[key] = outcome
        events_by_run[source.start.run_id] = manifest.events
        outcomes_by_run[source.start.run_id] = outcomes

    reports = {report.arm_id: report for report in report_set.score_reports}
    for mapping in report_set.score_sources:
        arm_id = cast(str, mapping["arm_id"])
        run_ids = cast(Sequence[str], mapping["source_run_ids"])
        numerator = denominator = unknown = unknown_fates = 0
        for run_id in run_ids:
            for event in events_by_run[run_id]:
                outcome = outcomes_by_run[run_id].get((arm_id, event.event_id))
                if outcome is None:
                    raise PrivateReportError("a scored event has no source fate")
                unknown_fates += outcome is TerminalOutcome.UNKNOWN
                if event_is_eligible(event, outcome):
                    denominator += 1
                    numerator += outcome is TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
                    unknown += outcome is TerminalOutcome.UNKNOWN
        report = reports[arm_id]
        recall = report.micro.wrapper_available_recall
        if (
            (recall.numerator, recall.denominator, recall.unknown)
            != (numerator, denominator, unknown)
            or report.unknown_counts.event_fates != unknown_fates
            or report.fixture_count != len(run_ids)
        ):
            raise PrivateReportError("score report disagrees with its source event fates")


def _report_set_bundle(report_set: ReportSet) -> dict[str, JsonValue]:
    return {
        "schema_version": PRIVATE_SCORE_BUNDLE_SCHEMA_VERSION,
        "classification": report_set.classification.value,
        "report_id": report_set.report_id,
        "cohort_id": report_set.cohort_id,
        "cohort_fixture_ids": report_set.cohort_fixture_ids,
        "cohort_manifest_sha256": report_set.cohort_manifest_sha256,
        "model_runtime_sha256": report_set.model_runtime_sha256,
        "partition": report_set.partition.value,
        "sources": tuple(source.source_wire for source in report_set.sources),
        "score_sources": report_set.score_sources,
        "score_reports": tuple(
            score_report_to_private_wire(report) for report in report_set.score_reports
        ),
    }


def _build_private_report(
    report_set: ReportSet,
    score_bundle_sha256: str,
) -> dict[str, JsonValue]:
    require_sha256(score_bundle_sha256, "score_bundle_sha256")
    return {
        "schema_version": PRIVATE_REPORT_SCHEMA_VERSION,
        "classification": report_set.classification.value,
        "report_id": report_set.report_id,
        "score_bundle_sha256": score_bundle_sha256,
        "cohort_id": report_set.cohort_id,
        "cohort_fixture_ids": report_set.cohort_fixture_ids,
        "cohort_manifest_sha256": report_set.cohort_manifest_sha256,
        "model_runtime_sha256": report_set.model_runtime_sha256,
        "partition": report_set.partition.value,
        "sources": tuple(source.source_wire for source in report_set.sources),
        "score_sources": report_set.score_sources,
        "integrities": tuple(
            _integrity_to_wire(source.integrity) for source in report_set.sources
        ),
        "score_reports": tuple(
            score_report_to_private_wire(report) for report in report_set.score_reports
        ),
    }


def private_report_sha256(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("private report payload must be a mapping")
    normalized = cast(Mapping[str, JsonValue], _json_value(payload))
    return canonical_sha256(normalized)


def render_private_report_markdown(payload: Mapping[str, object]) -> str:
    _require_private_report_envelope(payload)
    reports = cast(Sequence[object], payload["score_reports"])
    lines = [
        "# Yamabiko E0 private report",
        "",
        f"Classification: `{payload['classification']}`",
    ]
    integrities = cast(Sequence[object], payload["integrities"])
    lines.extend(
        (
            "",
            f"Report ID: `{payload['report_id']}`",
            f"Source runs: `{len(integrities)}`",
            f"Cohort ID: `{payload['cohort_id']}`",
        )
    )
    for index, raw_report in enumerate(reports, start=1):
        if not isinstance(raw_report, Mapping):
            raise PrivateReportError("score report is not an object")
        lines.extend(
            (
                "",
                f"## Score report {index}",
                "",
                f"- Arm ID: `{raw_report.get('arm_id', 'unavailable')}`",
                f"- Partition: `{raw_report.get('partition', 'unavailable')}`",
                f"- Fixtures: `{raw_report.get('fixture_count', 'unavailable')}`",
                f"- Titles: `{raw_report.get('title_count', 'unavailable')}`",
                "",
                "| Scope | Metric | Value | Status |",
                "|---|---|---:|---|",
            )
        )
        for scope in ("micro", "title_macro", "hard_stratum"):
            raw_scope = raw_report.get(scope)
            if not isinstance(raw_scope, Mapping):
                continue
            for metric_name in _PUBLIC_BOOTSTRAP_NAMES:
                metric = raw_scope.get(metric_name)
                if not isinstance(metric, Mapping):
                    continue
                lines.append(
                    "| "
                    f"{scope} | {metric_name} | {metric.get('value')} | "
                    f"{metric.get('status')} |"
                )
    return "\n".join(lines) + "\n"


def _normalize_score_reports(
    score_reports: ScoreReport | Sequence[ScoreReport],
) -> tuple[ScoreReport, ...]:
    if isinstance(score_reports, ScoreReport):
        return (score_reports,)
    if isinstance(score_reports, (str, bytes)):
        raise TypeError("score_reports must contain ScoreReport values")
    return tuple(score_reports)


def write_private_report(
    report_dir: Path,
    run_dirs: Sequence[Path],
    score_reports: ScoreReport | Sequence[ScoreReport],
    score_sources: Mapping[str, Sequence[str]],
    boundary: PrivateBoundary,
    *,
    write_views: bool = False,
) -> PrivateReportArtifacts:
    write_score_bundle(
        report_dir,
        run_dirs,
        score_reports,
        score_sources,
        boundary,
    )
    return write_private_report_from_bundle(
        report_dir,
        boundary,
        write_views=write_views,
    )


def _write_private_payload(
    report_path: Path,
    payload: Mapping[str, JsonValue],
    boundary: PrivateBoundary,
    *,
    derived_views: Mapping[
        str,
        Sequence[Mapping[str, JsonValue]],
    ]
    | None = None,
) -> PrivateReportArtifacts:
    report_bytes = canonical_json_bytes(payload, newline=True)
    markdown_bytes = render_private_report_markdown(payload).encode("utf-8")
    report_json_path = require_private_path(
        report_path / _PRIVATE_REPORT_NAME,
        boundary,
    )
    markdown_path = require_private_path(
        report_path / _PRIVATE_MARKDOWN_NAME,
        boundary,
    )
    if report_json_path.exists() or markdown_path.exists():
        raise PrivateReportError("private report already exists")
    if derived_views is not None and any(
        name not in _DERIVED_VIEW_NAMES for name in derived_views
    ):
        raise ValueError("unsupported derived view name")
    view_paths: list[Path] = []
    created_paths: list[Path] = []
    try:
        if derived_views is not None:
            for name, records in sorted(derived_views.items()):
                view_path = require_private_path(
                    report_path / f"{name}.jsonl",
                    boundary,
                )
                view_bytes = b"".join(
                    canonical_json_bytes(record, newline=True) for record in records
                )
                exclusive_write_bytes(view_path, view_bytes)
                view_paths.append(view_path)
                created_paths.append(view_path)
        exclusive_write_bytes(markdown_path, markdown_bytes)
        created_paths.append(markdown_path)
        exclusive_write_bytes(report_json_path, report_bytes)
        created_paths.append(report_json_path)
    except (OSError, UnicodeError):
        for created_path in reversed(created_paths):
            try:
                created_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise PrivateReportError("private report write failed") from None
    return PrivateReportArtifacts(
        report_json=report_json_path,
        report_markdown=markdown_path,
        derived_views=tuple(view_paths),
        report_sha256=hashlib.sha256(report_bytes[:-1]).hexdigest(),
        classification=PrivateContentClass(payload["classification"]),
    )


def write_score_bundle(
    report_dir: Path,
    run_dirs: Sequence[Path],
    scores: ScoreReport | Sequence[ScoreReport],
    score_sources: Mapping[str, Sequence[str]],
    boundary: PrivateBoundary,
) -> Path:
    report_path = _report_directory(report_dir, boundary)
    sources = verify_report_sources(run_dirs, boundary)
    reports = _normalize_score_reports(scores)
    report_set = _validate_report_set(
        report_path.name,
        sources,
        reports,
        score_sources,
    )
    _require_ledger_fate_accounting(report_set, boundary)
    bundle = _report_set_bundle(report_set)
    reports_root = require_private_path(boundary.root / "reports", boundary)
    try:
        reports_root.mkdir(parents=True, exist_ok=True)
        report_path.mkdir(exist_ok=False)
    except OSError:
        raise PrivateReportError(
            "private report directory reservation failed"
        ) from None
    bundle_path = require_private_path(
        report_path / _PRIVATE_SCORE_BUNDLE_NAME,
        boundary,
    )
    try:
        exclusive_write_bytes(
            bundle_path,
            canonical_json_bytes(bundle, newline=True),
        )
    except (OSError, UnicodeError):
        try:
            report_path.rmdir()
        except OSError:
            pass
        raise PrivateReportError("private score bundle write failed") from None
    return bundle_path


def _score_bundle_from_wire(
    value: Mapping[str, object],
    report_path: Path,
    boundary: PrivateBoundary,
) -> ReportSet:
    raw = _wire_object(
        value,
        {
            "schema_version",
            "classification",
            "report_id",
            "cohort_id",
            "cohort_fixture_ids",
            "cohort_manifest_sha256",
            "model_runtime_sha256",
            "partition",
            "sources",
            "score_sources",
            "score_reports",
        },
        "private score bundle",
    )
    if raw["schema_version"] != PRIVATE_SCORE_BUNDLE_SCHEMA_VERSION:
        raise PrivateReportError("unsupported private score bundle schema")
    try:
        classification = PrivateContentClass(raw["classification"])
        report_id = require_uuid4(raw["report_id"], "private score bundle report_id")
    except (TypeError, ValueError):
        raise PrivateReportError("private score bundle identity is invalid") from None
    if classification is PrivateContentClass.PUBLIC_ALLOWLISTED:
        raise PrivateReportError("private score bundle cannot already be public")
    if report_id != report_path.name:
        raise PrivateReportError("private score bundle report mismatch")

    raw_sources = _wire_sequence(raw["sources"], "private score sources")
    if not raw_sources:
        raise PrivateReportError("private score bundle has no sources")
    declared_sources: list[Mapping[str, object]] = []
    source_run_ids: list[str] = []
    for index, item in enumerate(raw_sources):
        source = _wire_object(
            item,
            {
                "run_id",
                "run_start_sha256",
                "run_final_sha256",
                "cohort_id",
                "cohort_manifest_sha256",
                "fixture_id",
            },
            f"private score sources[{index}]",
        )
        try:
            run_id = require_uuid4(source["run_id"], "source run_id")
            require_sha256(source["run_start_sha256"], "source run_start_sha256")
            require_sha256(source["run_final_sha256"], "source run_final_sha256")
            require_uuid4(source["cohort_id"], "source cohort_id")
            require_sha256(
                source["cohort_manifest_sha256"],
                "source cohort_manifest_sha256",
            )
            require_uuid4(source["fixture_id"], "source fixture_id")
        except (TypeError, ValueError):
            raise PrivateReportError(
                "private score source identity is invalid"
            ) from None
        declared_sources.append(source)
        source_run_ids.append(run_id)
    if source_run_ids != sorted(source_run_ids) or len(source_run_ids) != len(
        set(source_run_ids)
    ):
        raise PrivateReportError("private score sources are not sorted and unique")
    sources = tuple(
        _verify_source(boundary.root / "runs" / run_id, boundary)
        for run_id in source_run_ids
    )
    if any(
        dict(declared) != dict(source.source_wire)
        for declared, source in zip(declared_sources, sources, strict=True)
    ):
        raise PrivateReportError("private score source manifest binding changed")

    raw_reports = _wire_sequence(raw["score_reports"], "private score reports")
    if not raw_reports:
        raise PrivateReportError("private score bundle is empty")
    reports = tuple(score_report_from_private_wire(report) for report in raw_reports)
    if tuple(report.arm_id for report in reports) != tuple(
        sorted(report.arm_id for report in reports)
    ):
        raise PrivateReportError("private score reports are not sorted")

    raw_score_sources = _wire_sequence(
        raw["score_sources"],
        "private score source mappings",
    )
    score_source_mapping: dict[str, tuple[str, ...]] = {}
    declared_mappings: list[Mapping[str, JsonValue]] = []
    for index, item in enumerate(raw_score_sources):
        mapping = _wire_object(
            item,
            {"arm_id", "arm_kind", "source_run_ids"},
            f"private score source mappings[{index}]",
        )
        try:
            arm_id = require_uuid4(mapping["arm_id"], "score source arm_id")
            arm_kind = ArmKind(mapping["arm_kind"])
        except (TypeError, ValueError):
            raise PrivateReportError("private score source arm is invalid") from None
        run_id_values = _wire_sequence(
            mapping["source_run_ids"],
            "score source run IDs",
        )
        try:
            run_ids = tuple(
                require_uuid4(item, f"score source run_ids[{run_index}]")
                for run_index, item in enumerate(run_id_values)
            )
        except (TypeError, ValueError):
            raise PrivateReportError("private score source run ID is invalid") from None
        if run_ids != tuple(sorted(run_ids)):
            raise PrivateReportError("private score source run IDs are not sorted")
        if arm_id in score_source_mapping:
            raise PrivateReportError("private score source arms are not unique")
        score_source_mapping[arm_id] = run_ids
        declared_mappings.append(
            {
                "arm_id": arm_id,
                "arm_kind": arm_kind.value,
                "source_run_ids": run_ids,
            }
        )
    if tuple(score_source_mapping) != tuple(sorted(score_source_mapping)):
        raise PrivateReportError("private score source mappings are not sorted")

    report_set = _validate_report_set(
        report_id,
        sources,
        reports,
        score_source_mapping,
    )
    _require_ledger_fate_accounting(report_set, boundary)
    if tuple(declared_mappings) != report_set.score_sources:
        raise PrivateReportError("private score source mappings are not canonical")
    try:
        raw_fixture_ids = tuple(
            require_uuid4(item, "bundle cohort fixture_id")
            for item in _wire_sequence(
                raw["cohort_fixture_ids"],
                "bundle cohort fixture IDs",
            )
        )
        raw_cohort_id = require_uuid4(raw["cohort_id"], "bundle cohort_id")
        raw_cohort_hash = require_sha256(
            raw["cohort_manifest_sha256"],
            "bundle cohort_manifest_sha256",
        )
        raw_model_hash = require_sha256(
            raw["model_runtime_sha256"],
            "bundle model_runtime_sha256",
        )
        raw_partition = Partition(raw["partition"])
    except (TypeError, ValueError):
        raise PrivateReportError("private score bundle provenance is invalid") from None
    if (
        raw_fixture_ids != report_set.cohort_fixture_ids
        or raw_cohort_id != report_set.cohort_id
        or raw_cohort_hash != report_set.cohort_manifest_sha256
        or raw_model_hash != report_set.model_runtime_sha256
        or raw_partition is not report_set.partition
        or classification is not report_set.classification
    ):
        raise PrivateReportError("private score bundle provenance changed")
    return report_set


def _ledger_derived_views(
    run_paths: Sequence[Path],
) -> dict[str, tuple[Mapping[str, JsonValue], ...]]:
    selected = {
        "frames": RecordKind.FRAME,
        "windows": RecordKind.WINDOW,
        "event-fates": RecordKind.EVENT_FATE,
    }
    records = tuple(
        record for run_path in run_paths for record in iter_canonical_records(run_path)
    )
    return {
        name: tuple(
            evidence_record_to_wire(record)
            for record in records
            if record.kind is record_kind
        )
        for name, record_kind in selected.items()
    }


def _load_score_bundle(
    report_dir: Path,
    boundary: PrivateBoundary,
) -> tuple[ReportSet, bytes]:
    report_path = _report_directory(report_dir, boundary)
    if not report_path.is_dir():
        raise PrivateReportError("private report directory does not exist")
    bundle_path = require_private_path(
        report_path / _PRIVATE_SCORE_BUNDLE_NAME,
        boundary,
    )
    try:
        raw = bundle_path.read_bytes()
        bundle = canonical_json_mapping_from_bytes(raw, "private score bundle")
    except (OSError, UnicodeError, TypeError, ValueError):
        raise PrivateReportError("private score bundle is invalid") from None
    return _score_bundle_from_wire(bundle, report_path, boundary), raw


def load_verified_score_bundle(report_dir: Path, boundary: PrivateBoundary) -> ReportSet:
    """Reload a score bundle, re-verifying every source run and fate account."""

    return _load_score_bundle(report_dir, boundary)[0]


def _private_report_payload_from_bundle(
    report_dir: Path,
    boundary: PrivateBoundary,
) -> tuple[ReportSet, dict[str, JsonValue]]:
    report_set, raw = _load_score_bundle(report_dir, boundary)
    return report_set, _build_private_report(report_set, hashlib.sha256(raw).hexdigest())


def write_private_report_from_bundle(
    report_dir: Path,
    boundary: PrivateBoundary,
    *,
    write_views: bool = False,
) -> PrivateReportArtifacts:
    if type(write_views) is not bool:
        raise TypeError("write_views must be bool")
    report_set, payload = _private_report_payload_from_bundle(report_dir, boundary)
    report_path = _report_directory(report_dir, boundary)
    derived_views = (
        _ledger_derived_views(tuple(source.run_path for source in report_set.sources))
        if write_views
        else None
    )
    return _write_private_payload(
        report_path,
        payload,
        boundary,
        derived_views=derived_views,
    )


def _require_private_report_envelope(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise PrivateReportError("private report must be an object")
    if payload.get("schema_version") != PRIVATE_REPORT_SCHEMA_VERSION:
        raise PrivateReportError("unsupported private report schema")
    try:
        classification = PrivateContentClass(payload.get("classification"))
    except (TypeError, ValueError):
        raise PrivateReportError("invalid private report classification") from None
    if classification is PrivateContentClass.PUBLIC_ALLOWLISTED:
        raise PrivateReportError("private report cannot already be public")
    expected_fields = {
        "schema_version",
        "classification",
        "report_id",
        "score_bundle_sha256",
        "cohort_id",
        "cohort_fixture_ids",
        "cohort_manifest_sha256",
        "model_runtime_sha256",
        "partition",
        "sources",
        "score_sources",
        "integrities",
        "score_reports",
    }
    if set(payload) != expected_fields:
        raise PrivateReportError("private report has an invalid shape")
    try:
        require_uuid4(payload["report_id"], "private report report_id")
        require_sha256(
            payload["score_bundle_sha256"],
            "private report score_bundle_sha256",
        )
        require_uuid4(payload["cohort_id"], "private report cohort_id")
        require_sha256(
            payload["cohort_manifest_sha256"],
            "private report cohort_manifest_sha256",
        )
        require_sha256(
            payload["model_runtime_sha256"],
            "private report model_runtime_sha256",
        )
        Partition(payload["partition"])
    except (TypeError, ValueError):
        raise PrivateReportError("private report provenance is invalid") from None
    for field_name in (
        "cohort_fixture_ids",
        "sources",
        "score_sources",
        "integrities",
    ):
        if not _wire_sequence(payload[field_name], field_name):
            raise PrivateReportError(f"private report {field_name} is empty")
    reports = payload.get("score_reports")
    if (
        not isinstance(reports, Sequence)
        or isinstance(reports, (str, bytes))
        or not reports
    ):
        raise PrivateReportError("private report must contain score reports")
    if any(not isinstance(report, Mapping) for report in reports):
        raise PrivateReportError("score reports must be objects")


def _require_public_review(review: PublicExportReview) -> None:
    if not isinstance(review, PublicExportReview):
        raise TypeError("review must be PublicExportReview")
    if review.decision is not CapabilityDecision.APPROVED:
        raise PublicExportPrivacyError("public export review is not approved")
    if review.profile_version != PUBLIC_EXPORT_PROFILE_VERSION:
        raise PublicExportPrivacyError("public export profile version mismatch")
    if review.profile_sha256 != PUBLIC_EXPORT_PROFILE_SHA256:
        raise PublicExportPrivacyError("public export profile hash mismatch")
    unapproved = set(review.approved_fields) - PUBLIC_EXPORT_FIELD_CAP
    if unapproved:
        raise PublicExportPrivacyError("review requests a non-public field")


def _report_leaf(report: Mapping[str, object], field_name: str) -> object:
    if not field_name.startswith("score."):
        raise PublicExportPrivacyError("review requests a non-public field")
    value: object = report
    for component in field_name.removeprefix("score.").split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise PublicExportPrivacyError("reviewed public field is missing")
        value = value[component]
    if isinstance(value, (Mapping, list, tuple)):
        raise PublicExportPrivacyError("reviewed public field is not a leaf")
    return value


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise PublicExportPrivacyError(f"invalid value for {field_name}")
    return value


def _require_public_number(
    value: object,
    field_name: str,
    *,
    bounded: bool = False,
) -> int | float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PublicExportPrivacyError(f"invalid value for {field_name}")
    numeric = float(value)
    if numeric < 0 or (bounded and numeric > 1):
        raise PublicExportPrivacyError(f"invalid value for {field_name}")
    return cast(int | float, value)


def _validate_public_leaf(field_name: str, value: object) -> JsonValue:
    leaf = field_name.rsplit(".", 1)[-1]
    if field_name == "score.schema_version":
        if value != SCORE_SCHEMA_VERSION:
            raise PublicExportPrivacyError("invalid public score schema")
        return cast(JsonValue, value)
    if field_name == "score.partition":
        if value not in {partition.value for partition in Partition}:
            raise PublicExportPrivacyError("invalid public partition")
        return cast(JsonValue, value)
    if leaf == "sha256" or leaf.endswith("_sha256"):
        try:
            return require_sha256(value, field_name)
        except (TypeError, ValueError):
            raise PublicExportPrivacyError("invalid public hash") from None
    if leaf == "version" or leaf.endswith("_version"):
        try:
            return _require_ascii_token(value, field_name)
        except (TypeError, ValueError):
            raise PublicExportPrivacyError("invalid public version") from None
    if leaf == "status":
        if value not in {status.value for status in MetricStatus}:
            raise PublicExportPrivacyError("invalid public metric status")
        return cast(JsonValue, value)
    if field_name == "score.bootstrap_unit":
        if value != "title":
            raise PublicExportPrivacyError("invalid public bootstrap unit")
        return "title"
    if field_name == "score.bootstrap_quantile_method":
        if value != "linear_type7":
            raise PublicExportPrivacyError("invalid public bootstrap method")
        return "linear_type7"
    if leaf in {
        "fixture_count",
        "title_count",
        "bootstrap_seed",
        "bootstrap_requested_replicates",
        "substitutions",
        "deletions",
        "insertions",
        "reference_characters",
        "unresolved_events",
        "unresolved_components",
        "unassigned_insertions",
        "numerator",
        "denominator",
        "unknown",
        "total_titles",
        "eligible_titles",
        "zero_denominator_titles",
        "incomplete_titles",
        "event_fates",
        "raw_nonblank",
        "lexical_usability",
        "deadlines",
        "unresolved_alignment_components",
        "instrumentation_missing",
        "record_lost",
        "requested_replicates",
        "valid_replicates",
    }:
        return _require_nonnegative_int(value, field_name)
    if leaf in {"lower_bound", "upper_bound", "confidence"}:
        return cast(
            JsonValue,
            _require_public_number(value, field_name, bounded=True),
        )
    if leaf in {"value", "lower", "upper"}:
        bounded = ".cer_minimal." not in field_name
        return cast(
            JsonValue,
            _require_public_number(value, field_name, bounded=bounded),
        )
    raise PublicExportPrivacyError("public field has no value validator")


def build_public_aggregate(
    private_payload: Mapping[str, object],
    review: PublicExportReview,
) -> dict[str, JsonValue]:
    _require_private_report_envelope(private_payload)
    _require_public_review(review)
    if private_report_sha256(private_payload) != review.source_report_sha256:
        raise PublicExportPrivacyError("review does not match the private report")
    raw_reports = cast(Sequence[object], private_payload["score_reports"])
    arm_kind_by_id: dict[str, str] = {}
    for item in cast(Sequence[object], private_payload["score_sources"]):
        if not isinstance(item, Mapping):
            raise PublicExportPrivacyError("private score source is invalid")
        try:
            arm_id = require_uuid4(item.get("arm_id"), "score source arm_id")
            arm_kind = ArmKind(item.get("arm_kind")).value
        except (TypeError, ValueError):
            raise PublicExportPrivacyError(
                "private score source arm is invalid"
            ) from None
        if arm_id in arm_kind_by_id:
            raise PublicExportPrivacyError("private score source arm is duplicated")
        arm_kind_by_id[arm_id] = arm_kind
    public_reports: list[JsonValue] = []
    for raw_report in raw_reports:
        report = cast(Mapping[str, object], raw_report)
        selected: dict[str, JsonValue] = {}
        for field_name in review.approved_fields:
            selected[field_name] = _validate_public_leaf(
                field_name,
                _report_leaf(report, field_name),
            )
        public_report: dict[str, JsonValue] = {"fields": selected}
        try:
            report_arm_id = require_uuid4(report.get("arm_id"), "report arm_id")
            public_report["arm_kind"] = arm_kind_by_id[report_arm_id]
        except (KeyError, TypeError, ValueError):
            raise PublicExportPrivacyError(
                "private report arm mapping is incomplete"
            ) from None
        public_reports.append(public_report)
    return {
        "schema_version": PUBLIC_AGGREGATE_SCHEMA_VERSION,
        "classification": PrivateContentClass.PUBLIC_ALLOWLISTED.value,
        "profile_version": PUBLIC_EXPORT_PROFILE_VERSION,
        "profile_sha256": PUBLIC_EXPORT_PROFILE_SHA256,
        "source_report_sha256": review.source_report_sha256,
        "approved_fields": list(review.approved_fields),
        "reports": public_reports,
    }


def public_export_review_to_wire(
    review: PublicExportReview,
) -> dict[str, JsonValue]:
    if not isinstance(review, PublicExportReview):
        raise TypeError("review must be PublicExportReview")
    return {
        "schema_version": review.schema_version,
        "classification": review.classification.value,
        "review_id": review.review_id,
        "reviewer_id": review.reviewer_id,
        "evidence_ids": review.evidence_ids,
        "decision": review.decision.value,
        "source_report_sha256": review.source_report_sha256,
        "profile_version": review.profile_version,
        "profile_sha256": review.profile_sha256,
        "approved_fields": review.approved_fields,
    }


def public_export_review_from_wire(
    value: Mapping[str, object],
) -> PublicExportReview:
    expected = {
        "schema_version",
        "classification",
        "review_id",
        "reviewer_id",
        "evidence_ids",
        "decision",
        "source_report_sha256",
        "profile_version",
        "profile_sha256",
        "approved_fields",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PublicExportPrivacyError("public export review has an invalid shape")
    try:
        evidence_ids = value["evidence_ids"]
        approved_fields = value["approved_fields"]
        if (
            not isinstance(evidence_ids, Sequence)
            or isinstance(evidence_ids, (str, bytes))
            or not isinstance(approved_fields, Sequence)
            or isinstance(approved_fields, (str, bytes))
        ):
            raise TypeError
        return PublicExportReview(
            schema_version=cast(str, value["schema_version"]),
            classification=PrivateContentClass(value["classification"]),
            review_id=cast(str, value["review_id"]),
            reviewer_id=cast(str, value["reviewer_id"]),
            evidence_ids=tuple(cast(Sequence[str], evidence_ids)),
            decision=CapabilityDecision(value["decision"]),
            source_report_sha256=cast(str, value["source_report_sha256"]),
            profile_version=cast(str, value["profile_version"]),
            profile_sha256=cast(str, value["profile_sha256"]),
            approved_fields=tuple(cast(Sequence[str], approved_fields)),
        )
    except (TypeError, ValueError):
        raise PublicExportPrivacyError("public export review is invalid") from None


def _build_public_receipt(
    aggregate: Mapping[str, JsonValue],
    review: PublicExportReview,
    receipt_id: str,
    private_review_sha256: str,
) -> dict[str, JsonValue]:
    require_uuid4(receipt_id, "receipt_id")
    require_sha256(private_review_sha256, "private_review_sha256")
    aggregate_sha256 = canonical_sha256(aggregate)
    return {
        "schema_version": PUBLIC_EXPORT_RECEIPT_SCHEMA_VERSION,
        "classification": PrivateContentClass.PUBLIC_ALLOWLISTED.value,
        "receipt_id": receipt_id,
        "aggregate_name": _PUBLIC_AGGREGATE_NAME,
        "aggregate_sha256": aggregate_sha256,
        "source_report_sha256": review.source_report_sha256,
        "private_review_sha256": private_review_sha256,
        "profile_version": PUBLIC_EXPORT_PROFILE_VERSION,
        "profile_sha256": PUBLIC_EXPORT_PROFILE_SHA256,
        "approved_fields": review.approved_fields,
        "privacy": {
            "contains_absolute_paths": False,
            "contains_private_text": False,
            "contains_media_or_reference_hashes": False,
            "contains_per_title_rows": False,
            "contains_private_report_hash_only": True,
        },
        "rights_effect": {
            "grants_redistribution": False,
            "grants_training": False,
            "grants_transcript_publication": False,
            "changes_private_rights_records": False,
        },
    }


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _public_destination(path: Path, boundary: PrivateBoundary) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PublicExportPrivacyError("public destination must be explicit")
    resolved = candidate.resolve(strict=False)
    if _is_within(boundary.root, resolved):
        raise PublicExportPrivacyError("public destination cannot be private")
    if any(_is_within(root, resolved) for root in boundary.excluded_roots):
        raise PublicExportPrivacyError("public destination is excluded")
    return resolved


def export_public(
    report_dir: Path,
    review_path: Path,
    destination: Path,
    boundary: PrivateBoundary,
    *,
    receipt_id_factory: Callable[[], str] = new_opaque_id,
) -> PublicExportArtifacts:
    report_path = _report_directory(report_dir, boundary)
    if not report_path.is_dir():
        raise PublicExportPrivacyError("private report directory does not exist")
    private_path = require_private_path(
        report_path / _PRIVATE_REPORT_NAME,
        boundary,
    )
    private_review_path = require_private_path(Path(review_path), boundary)
    try:
        review_relative = private_review_path.relative_to(boundary.root)
    except ValueError:
        raise PublicExportPrivacyError("review is outside the private root") from None
    if not review_relative.parts or review_relative.parts[0] != "export-reviews":
        raise PublicExportPrivacyError("review is not in export-reviews")
    output_dir = _public_destination(destination, boundary)
    aggregate_path = output_dir / _PUBLIC_AGGREGATE_NAME
    receipt_path = output_dir / _PUBLIC_RECEIPT_NAME
    try:
        private_raw = private_path.read_bytes()
        review_raw = private_review_path.read_bytes()
        private_payload = canonical_json_mapping_from_bytes(
            private_raw,
            "private report",
        )
        review_payload = canonical_json_mapping_from_bytes(
            review_raw,
            "public export review",
        )
    except (OSError, UnicodeError, TypeError, ValueError):
        raise PublicExportPrivacyError("private export input is invalid") from None
    review = public_export_review_from_wire(review_payload)
    # Rebuild the private report from its score bundle (re-verifying every
    # source run) so a report.json edited after rendering cannot be exported.
    _, rebuilt_payload = _private_report_payload_from_bundle(report_path, boundary)
    if private_report_sha256(rebuilt_payload) != private_report_sha256(
        private_payload
    ):
        raise PublicExportPrivacyError("private report does not match its score bundle")
    aggregate = build_public_aggregate(private_payload, review)
    receipt_id = receipt_id_factory()
    receipt = _build_public_receipt(
        aggregate,
        review,
        receipt_id,
        hashlib.sha256(review_raw[:-1]).hexdigest(),
    )
    aggregate_bytes = canonical_json_bytes(aggregate, newline=True)
    receipt_bytes = canonical_json_bytes(receipt, newline=True)
    created_paths: list[Path] = []
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
    except OSError:
        raise PublicExportPrivacyError(
            "public export destination reservation failed"
        ) from None
    try:
        exclusive_write_bytes(aggregate_path, aggregate_bytes)
        created_paths.append(aggregate_path)
        exclusive_write_bytes(receipt_path, receipt_bytes)
        created_paths.append(receipt_path)
    except (OSError, UnicodeError):
        for partial_path in reversed(created_paths):
            try:
                partial_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise PublicExportPrivacyError("public export write failed") from None
    return PublicExportArtifacts(
        aggregate_json=aggregate_path,
        receipt_json=receipt_path,
        aggregate_sha256=hashlib.sha256(aggregate_bytes[:-1]).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_bytes[:-1]).hexdigest(),
        receipt_id=receipt_id,
    )


__all__ = (
    "PRIVATE_REPORT_SCHEMA_VERSION",
    "PRIVATE_SCORE_BUNDLE_SCHEMA_VERSION",
    "PUBLIC_AGGREGATE_SCHEMA_VERSION",
    "PUBLIC_EXPORT_FIELD_CAP",
    "PUBLIC_EXPORT_PROFILE_SHA256",
    "PUBLIC_EXPORT_PROFILE_VERSION",
    "PUBLIC_EXPORT_RECEIPT_SCHEMA_VERSION",
    "PUBLIC_EXPORT_REVIEW_SCHEMA_VERSION",
    "PrivateReportArtifacts",
    "PrivateReportError",
    "PublicExportArtifacts",
    "PublicExportPrivacyError",
    "PublicExportReview",
    "ReportSet",
    "VerifiedSource",
    "build_public_aggregate",
    "export_public",
    "load_verified_score_bundle",
    "private_report_sha256",
    "public_export_review_from_wire",
    "public_export_review_to_wire",
    "render_private_report_markdown",
    "score_report_to_private_wire",
    "score_report_from_private_wire",
    "verify_report_sources",
    "write_private_report",
    "write_private_report_from_bundle",
    "write_score_bundle",
)
