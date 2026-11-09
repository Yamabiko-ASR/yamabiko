"""Canonical JSON codec for E0 evidence wire version 1.0."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Iterable, Mapping, TypeVar, cast

from .schema import (
    SCHEMA_VERSION,
    ArmKind,
    ArmStatus,
    ArmTerminalStatus,
    ArmDefinition,
    DiagnosticCode,
    EvidenceRecord,
    IntegrityFailureCode,
    JsonValue,
    LossCode,
    ObservationStatus,
    PrivateContentClass,
    PrivateTextRecord,
    RecordKind,
    RunFinalManifest,
    RunStartManifest,
    RunStatus,
    SampleSpan,
    Stage,
    StreamName,
    StreamSummary,
    TextRole,
)


class WireValidationError(ValueError):
    """A wire value cannot be interpreted under schema version 1.0."""


_EnumT = TypeVar("_EnumT", bound=Enum)
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "run_id",
        "arm_id",
        "session_id",
        "seq",
        "observed_monotonic_ns",
        "clock_id",
        "stage",
        "kind",
        "artifact_id",
        "parent_ids",
        "source_spans",
        "observation_status",
        "decision",
        "diagnostics",
        "classification",
        "details",
    }
)
_SPAN_FIELDS = frozenset({"timeline_id", "start", "end"})
_ARM_DEFINITION_FIELDS = frozenset(
    {
        "arm_id",
        "definition_sha256",
        "declared_changes",
        "kind",
        "parameters",
        "parent_arm_id",
    }
)
_RUN_START_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "session_id",
        "clock_id",
        "canonical_stream_id",
        "private_text_stream_id",
        "arm_definitions",
        "classification",
        "details",
    }
)
_PRIVATE_TEXT_FIELDS = frozenset(
    {
        "schema_version",
        "private_record_id",
        "run_id",
        "session_id",
        "stream_id",
        "stream_seq",
        "canonical_record_id",
        "arm_id",
        "text_id",
        "role",
        "classification",
        "text",
    }
)
_STREAM_SUMMARY_FIELDS = frozenset(
    {
        "name",
        "stream_id",
        "record_count",
        "first_seq",
        "last_seq",
        "byte_sha256",
        "classification",
    }
)
_ARM_STATUS_FIELDS = frozenset(
    {"arm_id", "status", "terminal_record_id", "reason"}
)
_RUN_FINAL_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "session_id",
        "run_start_sha256",
        "status",
        "streams",
        "arm_statuses",
        "loss_counts",
        "incomplete_reasons",
        "classification",
        "finalized_monotonic_ns",
    }
)
_PRIVACY_RANK = {
    PrivateContentClass.PUBLIC_ALLOWLISTED: 0,
    PrivateContentClass.INTERNAL_AGGREGATE: 1,
    PrivateContentClass.PRIVATE_METADATA: 2,
    PrivateContentClass.PRIVATE_TEXT: 3,
    PrivateContentClass.PRIVATE_AUDIO: 4,
}


def _json_ready(value: object, path: str = "$") -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            result[key] = _json_ready(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def canonical_json_bytes(
    value: Mapping[str, JsonValue],
    *,
    newline: bool = False,
) -> bytes:
    """Encode JSON deterministically as compact, sorted UTF-8 bytes."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON root must be a mapping")
    encoded = json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if newline:
        return encoded + b"\n"
    return encoded


def canonical_sha256(value: Mapping[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_mapping_from_bytes(
    raw: bytes,
    artifact_name: str,
) -> dict[str, object]:
    """Decode one canonical JSON object terminated by exactly one LF."""

    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if type(artifact_name) is not str or not artifact_name:
        raise TypeError("artifact_name must be a non-empty string")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise WireValidationError(f"{artifact_name} is not one complete JSON line")
    body = raw[:-1]
    if not body or b"\n" in body or b"\r" in body:
        raise WireValidationError(f"{artifact_name} is not canonical JSON")

    def reject_constant(value: str) -> object:
        raise WireValidationError(
            f"{artifact_name} contains non-finite JSON value {value}"
        )

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise WireValidationError(
                    f"{artifact_name} contains duplicate field {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except WireValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireValidationError(f"{artifact_name} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise WireValidationError(f"{artifact_name} must be a JSON object")
    canonical = canonical_json_bytes(
        cast(Mapping[str, JsonValue], decoded),
        newline=True,
    )
    if canonical != raw:
        raise WireValidationError(f"{artifact_name} is not canonically encoded")
    return decoded


def arm_definition_sha256(definition: ArmDefinition) -> str:
    if not isinstance(definition, ArmDefinition):
        raise TypeError("definition must be ArmDefinition")
    wire_value = cast(
        Mapping[str, JsonValue],
        {
            "arm_id": definition.arm_id,
            "declared_changes": list(definition.declared_changes),
            "kind": definition.kind.value,
            "parameters": _json_ready(definition.parameters),
            "parent_arm_id": definition.parent_arm_id,
        },
    )
    return canonical_sha256(wire_value)


def evidence_record_to_wire(record: EvidenceRecord) -> dict[str, JsonValue]:
    if not isinstance(record, EvidenceRecord):
        raise TypeError("record must be EvidenceRecord")
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": record.schema_version,
            "record_id": record.record_id,
            "run_id": record.run_id,
            "arm_id": record.arm_id,
            "session_id": record.session_id,
            "seq": record.seq,
            "observed_monotonic_ns": record.observed_monotonic_ns,
            "clock_id": record.clock_id,
            "stage": record.stage.value,
            "kind": record.kind.value,
            "artifact_id": record.artifact_id,
            "parent_ids": list(record.parent_ids),
            "source_spans": [
                {
                    "timeline_id": span.timeline_id,
                    "start": span.start,
                    "end": span.end,
                }
                for span in record.source_spans
            ],
            "observation_status": record.observation_status.value,
            "decision": record.decision,
            "diagnostics": [diagnostic.value for diagnostic in record.diagnostics],
            "classification": record.classification.value,
            "details": _json_ready(record.details),
        },
    )


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(f"missing fields {missing!r}")
        if unknown:
            parts.append(f"unknown fields {unknown!r}")
        raise WireValidationError(f"{path} has " + " and ".join(parts))


def _enum_from_wire(
    enum_type: type[_EnumT],
    value: object,
    field_name: str,
) -> _EnumT:
    if type(value) is not str:
        raise WireValidationError(f"{field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise WireValidationError(
            f"{field_name} has unknown meaning under schema_version {SCHEMA_VERSION}"
        ) from exc


def _list_from_wire(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise WireValidationError(f"{field_name} must be a JSON array")
    return value


def _span_from_wire(value: object, index: int) -> SampleSpan:
    field_name = f"source_spans[{index}]"
    if not isinstance(value, Mapping):
        raise WireValidationError(f"{field_name} must be an object")
    _require_exact_fields(value, _SPAN_FIELDS, field_name)
    try:
        return SampleSpan(
            timeline_id=value["timeline_id"],  # type: ignore[arg-type]
            start=value["start"],  # type: ignore[arg-type]
            end=value["end"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(f"{field_name}: {exc}") from exc


def evidence_record_from_wire(value: Mapping[str, object]) -> EvidenceRecord:
    if not isinstance(value, Mapping):
        raise WireValidationError("evidence record must be an object")
    _require_exact_fields(value, _RECORD_FIELDS, "evidence record")
    if value["schema_version"] != SCHEMA_VERSION:
        raise WireValidationError(
            f"schema_version has unknown meaning: expected {SCHEMA_VERSION!r}"
        )

    arm_id = value["arm_id"]
    if arm_id is not None and type(arm_id) is not str:
        raise WireValidationError("arm_id must be a UUIDv4 string or null")
    decision = value["decision"]
    if decision is not None and type(decision) is not str:
        raise WireValidationError("decision must be a string or null")

    parent_ids = _list_from_wire(value["parent_ids"], "parent_ids")
    if any(type(parent_id) is not str for parent_id in parent_ids):
        raise WireValidationError("parent_ids must contain only UUIDv4 strings")
    source_spans = tuple(
        _span_from_wire(span, index)
        for index, span in enumerate(
            _list_from_wire(value["source_spans"], "source_spans")
        )
    )
    diagnostics = tuple(
        _enum_from_wire(DiagnosticCode, diagnostic, f"diagnostics[{index}]")
        for index, diagnostic in enumerate(
            _list_from_wire(value["diagnostics"], "diagnostics")
        )
    )
    details = value["details"]
    if not isinstance(details, Mapping):
        raise WireValidationError("details must be an object")

    try:
        return EvidenceRecord(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            record_id=value["record_id"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            arm_id=arm_id,
            session_id=value["session_id"],  # type: ignore[arg-type]
            seq=value["seq"],  # type: ignore[arg-type]
            observed_monotonic_ns=value[  # type: ignore[arg-type]
                "observed_monotonic_ns"
            ],
            clock_id=value["clock_id"],  # type: ignore[arg-type]
            stage=_enum_from_wire(Stage, value["stage"], "stage"),
            kind=_enum_from_wire(RecordKind, value["kind"], "kind"),
            artifact_id=value["artifact_id"],  # type: ignore[arg-type]
            parent_ids=tuple(parent_ids),  # type: ignore[arg-type]
            source_spans=source_spans,
            observation_status=_enum_from_wire(
                ObservationStatus,
                value["observation_status"],
                "observation_status",
            ),
            decision=decision,
            diagnostics=diagnostics,
            classification=_enum_from_wire(
                PrivateContentClass,
                value["classification"],
                "classification",
            ),
            details=details,  # type: ignore[arg-type]
        )
    except WireValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise WireValidationError(str(exc)) from exc


def inherited_classification(
    own: PrivateContentClass,
    parents: Iterable[PrivateContentClass],
) -> PrivateContentClass:
    if not isinstance(own, PrivateContentClass):
        raise TypeError("own must be PrivateContentClass")
    inherited = own
    for index, parent in enumerate(parents):
        if not isinstance(parent, PrivateContentClass):
            raise TypeError(f"parents[{index}] must be PrivateContentClass")
        if _PRIVACY_RANK[parent] > _PRIVACY_RANK[inherited]:
            inherited = parent
    return inherited


def _arm_definition_to_wire(
    definition: ArmDefinition,
) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "arm_id": definition.arm_id,
            "definition_sha256": arm_definition_sha256(definition),
            "declared_changes": list(definition.declared_changes),
            "kind": definition.kind.value,
            "parameters": _json_ready(definition.parameters),
            "parent_arm_id": definition.parent_arm_id,
        },
    )


def _arm_definition_from_wire(value: object, index: int) -> ArmDefinition:
    field_name = f"arm_definitions[{index}]"
    if not isinstance(value, Mapping):
        raise WireValidationError(f"{field_name} must be an object")
    _require_exact_fields(value, _ARM_DEFINITION_FIELDS, field_name)
    declared_changes = _list_from_wire(
        value["declared_changes"],
        f"{field_name}.declared_changes",
    )
    if any(type(item) is not str for item in declared_changes):
        raise WireValidationError(
            f"{field_name}.declared_changes must contain strings"
        )
    parameters = value["parameters"]
    if not isinstance(parameters, Mapping):
        raise WireValidationError(f"{field_name}.parameters must be an object")
    parent_arm_id = value["parent_arm_id"]
    if parent_arm_id is not None and type(parent_arm_id) is not str:
        raise WireValidationError(
            f"{field_name}.parent_arm_id must be a UUIDv4 string or null"
        )
    try:
        definition = ArmDefinition(
            arm_id=value["arm_id"],  # type: ignore[arg-type]
            kind=_enum_from_wire(ArmKind, value["kind"], f"{field_name}.kind"),
            parent_arm_id=parent_arm_id,
            parameters=parameters,  # type: ignore[arg-type]
            declared_changes=tuple(declared_changes),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(f"{field_name}: {exc}") from exc
    if value["definition_sha256"] != arm_definition_sha256(definition):
        raise WireValidationError(
            f"{field_name}.definition_sha256 does not match definition"
        )
    return definition


def run_start_to_wire(manifest: RunStartManifest) -> dict[str, JsonValue]:
    if not isinstance(manifest, RunStartManifest):
        raise TypeError("manifest must be RunStartManifest")
    return cast(
        dict[str, JsonValue],
        {
            "arm_definitions": [
                _arm_definition_to_wire(definition)
                for definition in manifest.arm_definitions
            ],
            "canonical_stream_id": manifest.canonical_stream_id,
            "classification": manifest.classification.value,
            "clock_id": manifest.clock_id,
            "details": _json_ready(manifest.details),
            "private_text_stream_id": manifest.private_text_stream_id,
            "run_id": manifest.run_id,
            "schema_version": manifest.schema_version,
            "session_id": manifest.session_id,
        },
    )


def run_start_from_wire(value: Mapping[str, object]) -> RunStartManifest:
    if not isinstance(value, Mapping):
        raise WireValidationError("run start manifest must be an object")
    _require_exact_fields(value, _RUN_START_FIELDS, "run start manifest")
    if value["schema_version"] != SCHEMA_VERSION:
        raise WireValidationError("run start schema_version is not supported")
    definitions = tuple(
        _arm_definition_from_wire(item, index)
        for index, item in enumerate(
            _list_from_wire(value["arm_definitions"], "arm_definitions")
        )
    )
    details = value["details"]
    if not isinstance(details, Mapping):
        raise WireValidationError("run start details must be an object")
    private_stream = value["private_text_stream_id"]
    if private_stream is not None and type(private_stream) is not str:
        raise WireValidationError(
            "private_text_stream_id must be a UUIDv4 string or null"
        )
    try:
        return RunStartManifest(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            session_id=value["session_id"],  # type: ignore[arg-type]
            clock_id=value["clock_id"],  # type: ignore[arg-type]
            canonical_stream_id=value["canonical_stream_id"],  # type: ignore[arg-type]
            private_text_stream_id=private_stream,
            arm_definitions=definitions,
            classification=_enum_from_wire(
                PrivateContentClass,
                value["classification"],
                "classification",
            ),
            details=details,  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(str(exc)) from exc


def private_text_record_to_wire(
    record: PrivateTextRecord,
) -> dict[str, JsonValue]:
    if not isinstance(record, PrivateTextRecord):
        raise TypeError("record must be PrivateTextRecord")
    return cast(
        dict[str, JsonValue],
        {
            "arm_id": record.arm_id,
            "canonical_record_id": record.canonical_record_id,
            "classification": record.classification.value,
            "private_record_id": record.private_record_id,
            "role": record.role.value,
            "run_id": record.run_id,
            "schema_version": record.schema_version,
            "session_id": record.session_id,
            "stream_id": record.stream_id,
            "stream_seq": record.stream_seq,
            "text": record.text,
            "text_id": record.text_id,
        },
    )


def private_text_record_from_wire(
    value: Mapping[str, object],
) -> PrivateTextRecord:
    if not isinstance(value, Mapping):
        raise WireValidationError("private text record must be an object")
    _require_exact_fields(value, _PRIVATE_TEXT_FIELDS, "private text record")
    if value["schema_version"] != SCHEMA_VERSION:
        raise WireValidationError("private text schema_version is not supported")
    try:
        return PrivateTextRecord(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            private_record_id=value["private_record_id"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            session_id=value["session_id"],  # type: ignore[arg-type]
            stream_id=value["stream_id"],  # type: ignore[arg-type]
            stream_seq=value["stream_seq"],  # type: ignore[arg-type]
            canonical_record_id=value["canonical_record_id"],  # type: ignore[arg-type]
            arm_id=value["arm_id"],  # type: ignore[arg-type]
            text_id=value["text_id"],  # type: ignore[arg-type]
            role=_enum_from_wire(TextRole, value["role"], "role"),
            classification=_enum_from_wire(
                PrivateContentClass,
                value["classification"],
                "classification",
            ),
            text=value["text"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(str(exc)) from exc


def _stream_summary_to_wire(summary: StreamSummary) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "byte_sha256": summary.byte_sha256,
            "classification": summary.classification.value,
            "first_seq": summary.first_seq,
            "last_seq": summary.last_seq,
            "name": summary.name.value,
            "record_count": summary.record_count,
            "stream_id": summary.stream_id,
        },
    )


def _stream_summary_from_wire(value: object, index: int) -> StreamSummary:
    field_name = f"streams[{index}]"
    if not isinstance(value, Mapping):
        raise WireValidationError(f"{field_name} must be an object")
    _require_exact_fields(value, _STREAM_SUMMARY_FIELDS, field_name)
    try:
        return StreamSummary(
            name=_enum_from_wire(StreamName, value["name"], f"{field_name}.name"),
            stream_id=value["stream_id"],  # type: ignore[arg-type]
            record_count=value["record_count"],  # type: ignore[arg-type]
            first_seq=value["first_seq"],  # type: ignore[arg-type]
            last_seq=value["last_seq"],  # type: ignore[arg-type]
            byte_sha256=value["byte_sha256"],  # type: ignore[arg-type]
            classification=_enum_from_wire(
                PrivateContentClass,
                value["classification"],
                f"{field_name}.classification",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(f"{field_name}: {exc}") from exc


def _arm_status_to_wire(status: ArmTerminalStatus) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "arm_id": status.arm_id,
            "reason": status.reason,
            "status": status.status.value,
            "terminal_record_id": status.terminal_record_id,
        },
    )


def _arm_status_from_wire(value: object, index: int) -> ArmTerminalStatus:
    field_name = f"arm_statuses[{index}]"
    if not isinstance(value, Mapping):
        raise WireValidationError(f"{field_name} must be an object")
    _require_exact_fields(value, _ARM_STATUS_FIELDS, field_name)
    reason = value["reason"]
    if reason is not None and type(reason) is not str:
        raise WireValidationError(f"{field_name}.reason must be a string or null")
    try:
        return ArmTerminalStatus(
            arm_id=value["arm_id"],  # type: ignore[arg-type]
            status=_enum_from_wire(
                ArmStatus,
                value["status"],
                f"{field_name}.status",
            ),
            terminal_record_id=value["terminal_record_id"],  # type: ignore[arg-type]
            reason=reason,
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(f"{field_name}: {exc}") from exc


def run_final_to_wire(manifest: RunFinalManifest) -> dict[str, JsonValue]:
    if not isinstance(manifest, RunFinalManifest):
        raise TypeError("manifest must be RunFinalManifest")
    return cast(
        dict[str, JsonValue],
        {
            "arm_statuses": [
                _arm_status_to_wire(status) for status in manifest.arm_statuses
            ],
            "classification": manifest.classification.value,
            "finalized_monotonic_ns": manifest.finalized_monotonic_ns,
            "incomplete_reasons": [
                reason.value for reason in manifest.incomplete_reasons
            ],
            "loss_counts": {
                code.value: manifest.loss_counts[code] for code in LossCode
            },
            "run_id": manifest.run_id,
            "run_start_sha256": manifest.run_start_sha256,
            "schema_version": manifest.schema_version,
            "session_id": manifest.session_id,
            "status": manifest.status.value,
            "streams": [
                _stream_summary_to_wire(summary)
                for summary in manifest.streams
            ],
        },
    )


def run_final_from_wire(value: Mapping[str, object]) -> RunFinalManifest:
    if not isinstance(value, Mapping):
        raise WireValidationError("run final manifest must be an object")
    _require_exact_fields(value, _RUN_FINAL_FIELDS, "run final manifest")
    if value["schema_version"] != SCHEMA_VERSION:
        raise WireValidationError("run final schema_version is not supported")
    loss_value = value["loss_counts"]
    if not isinstance(loss_value, Mapping):
        raise WireValidationError("loss_counts must be an object")
    if set(loss_value) != {code.value for code in LossCode}:
        raise WireValidationError("loss_counts must contain every LossCode")
    losses = {code: loss_value[code.value] for code in LossCode}
    try:
        return RunFinalManifest(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            session_id=value["session_id"],  # type: ignore[arg-type]
            run_start_sha256=value["run_start_sha256"],  # type: ignore[arg-type]
            status=_enum_from_wire(RunStatus, value["status"], "status"),
            streams=tuple(
                _stream_summary_from_wire(item, index)
                for index, item in enumerate(
                    _list_from_wire(value["streams"], "streams")
                )
            ),
            arm_statuses=tuple(
                _arm_status_from_wire(item, index)
                for index, item in enumerate(
                    _list_from_wire(value["arm_statuses"], "arm_statuses")
                )
            ),
            loss_counts=losses,  # type: ignore[arg-type]
            incomplete_reasons=tuple(
                _enum_from_wire(
                    IntegrityFailureCode,
                    item,
                    f"incomplete_reasons[{index}]",
                )
                for index, item in enumerate(
                    _list_from_wire(
                        value["incomplete_reasons"],
                        "incomplete_reasons",
                    )
                )
            ),
            classification=_enum_from_wire(
                PrivateContentClass,
                value["classification"],
                "classification",
            ),
            finalized_monotonic_ns=value["finalized_monotonic_ns"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise WireValidationError(str(exc)) from exc
