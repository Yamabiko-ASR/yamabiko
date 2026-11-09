"""Strict, private import of human E0 annotation worksheets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from ._common import atomic_write_bytes, require_uuid4
from .normalize_ja import normalize_minimal
from .private_root import PrivateBoundary, require_private_path
from .references import (
    ACOUSTIC_TAGS,
    AnnotationInputRole,
    AnnotationPass,
    ReferenceSetManifest,
    requires_double_annotation,
)
from .registry import load_registered_master
from .schema import SCHEMA_VERSION, JsonValue, SampleSpan, new_opaque_id
from .wire import canonical_json_bytes, canonical_json_mapping_from_bytes


WORKSHEET_COLUMNS = (
    "start_seconds",
    "end_seconds",
    "event_type",
    "japanese",
    "intelligibility",
    "speaker",
    "overlap",
    "confidence",
    "acoustic_tags",
    "notes",
)
_WORK_FIELDS = frozenset(
    {
        "annotator_id",
        "annotator_slot",
        "completed_at_utc",
        "events",
        "fixture_id",
        "human_signature_receipt_id",
        "input_roles",
        "master_id",
        "packet_id",
        "pass_id",
        "pilot_id",
        "prohibited_inputs",
        "sample_count",
        "sample_rate_hz",
        "schema_version",
        "selection_label",
        "status",
        "timeline_id",
        "verbatim_references",
    }
)
_WORK_SCHEMA_VERSION = "e0-human-annotation-work-1.0"
_PASS_SCHEMA_VERSION = "e0-annotation-pass-2.0"
_PASS_TEXT_SCHEMA_VERSION = "e0-annotation-pass-text-1.0"
_EVENT_TYPES = frozenset({"speech", "non_speech"})
_INTELLIGIBILITY = frozenset(
    {"intelligible", "partially_intelligible", "unintelligible", "not_applicable"}
)
_OVERLAP = frozenset({"none", "overlap"})
_CONFIDENCE = frozenset({"high", "medium", "low"})
_TIMESTAMP_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{3}$")
_SPEAKER_RE = re.compile(r"^speaker_[1-9][0-9]*$")
_PASS_FIELDS = frozenset(
    {
        "schema_version",
        "work_schema_version",
        "pass_id",
        "annotator_id",
        "annotator_slot",
        "packet_id",
        "pilot_id",
        "selection_label",
        "fixture_id",
        "master_id",
        "timeline_id",
        "sample_rate_hz",
        "sample_count",
        "reference_version",
        "input_roles",
        "prohibited_inputs",
        "status",
        "completed_at_utc",
        "human_signature_receipt_id",
        "content_sha256",
        "worksheet_byte_count",
        "pass_text_sha256",
        "events",
        "verbatim_references",
        "non_speech_regions",
        "subtitle_proposal_sha256",
        "ffmpeg_sha256",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "row_number",
        "span",
        "intelligibility",
        "speaker",
        "overlap",
        "confidence",
        "acoustic_tags",
        "verbatim_ref_id",
        "note_id",
        "annotation_version",
    }
)
_VERBATIM_FIELDS = frozenset(
    {"verbatim_ref_id", "event_ids", "text_id", "annotation_version"}
)
_NON_SPEECH_FIELDS = frozenset(
    {
        "region_id",
        "row_number",
        "span",
        "intelligibility",
        "speaker",
        "overlap",
        "confidence",
        "acoustic_tags",
        "note_id",
    }
)
# Subtitle-seeded passes keep an exact copy of their editorial sidecar here.
_EDITORIAL_SIDECAR_NAME = "editorial.private.json"
# Worksheet timestamps are milliseconds on the 16 kHz master timeline.
_SAMPLES_PER_MILLISECOND = 16


@dataclass(frozen=True, slots=True)
class AnnotationWorksheetRow:
    row_number: int
    span: SampleSpan
    event_type: str
    text: str
    intelligibility: str
    speaker: str
    overlap: str
    confidence: str
    acoustic_tags: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class ParsedAnnotationWorksheet:
    content_sha256: str
    byte_count: int
    rows: tuple[AnnotationWorksheetRow, ...]


@dataclass(frozen=True, slots=True)
class AnnotationPassBundle:
    pass_id: str
    directory: Path
    content_sha256: str
    pass_text_sha256: str
    pass_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class RegisteredAnnotationPass:
    annotation_pass: AnnotationPass
    status: str
    bundle: AnnotationPassBundle
    manifest: Mapping[str, object]
    texts: Mapping[str, str]
    editorial_sidecar: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _AnnotationPolicy:
    input_roles: tuple[str, ...]
    prohibited_inputs: tuple[str, ...]
    speech_text: str

    @property
    def subtitle_seeded(self) -> bool:
        return "existing_subtitle" in self.input_roles


_ANNOTATION_POLICIES = (
    _AnnotationPolicy(
        ("audio_only",),
        ("candidate_output", "editorial_subtitles", "other_annotation_pass"),
        "required",
    ),
    _AnnotationPolicy(
        ("audio_only", "existing_subtitle"),
        ("candidate_output", "other_annotation_pass"),
        "required",
    ),
    _AnnotationPolicy(
        ("audio_only", "speech_event_layer"),
        ("candidate_output", "editorial_subtitles", "other_annotation_pass"),
        "forbidden",
    ),
)


def _plain_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer of at least {minimum}")
    return value


def _nonempty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _timestamp_samples(value: str, field_name: str) -> int:
    if not _TIMESTAMP_RE.fullmatch(value):
        raise ValueError(f"{field_name} timestamp must use exactly three decimals")
    seconds, milliseconds = value.split(".", 1)
    return (int(seconds) * 1_000 + int(milliseconds)) * _SAMPLES_PER_MILLISECOND


def _parse_tags(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    tags = tuple(value.split(";"))
    if any(not tag or tag.strip() != tag for tag in tags):
        raise ValueError("acoustic tag list is invalid")
    if len(tags) != len(set(tags)) or any(tag not in ACOUSTIC_TAGS for tag in tags):
        raise ValueError("acoustic tag list is invalid")
    if "music_effects_noise" in tags and ({"music", "effects_noise"} & set(tags)):
        raise ValueError("acoustic tag list is contradictory")
    if {"code_switch", "non_japanese_speech"} <= set(tags):
        raise ValueError("acoustic tag list is contradictory")
    return tags


def _parse_row(
    values: list[str],
    row_number: int,
    timeline_id: str,
    sample_count: int,
    policy: _AnnotationPolicy,
) -> AnnotationWorksheetRow:
    if len(values) != len(WORKSHEET_COLUMNS):
        raise ValueError("worksheet row must contain exactly ten columns")
    if any("\r" in value or "\n" in value or "\x00" in value for value in values):
        raise ValueError("worksheet fields must remain on one physical row")
    (
        start_text,
        end_text,
        event_type,
        text,
        intelligibility,
        speaker,
        overlap,
        confidence,
        tag_text,
        notes,
    ) = values
    start = _timestamp_samples(start_text, "start_seconds")
    end = _timestamp_samples(end_text, "end_seconds")
    if end <= start:
        raise ValueError("worksheet span must be non-empty")
    if end > sample_count:
        raise ValueError("worksheet span exceeds the master timeline")
    if event_type not in _EVENT_TYPES:
        raise ValueError("event_type is invalid")
    if intelligibility not in _INTELLIGIBILITY:
        raise ValueError("intelligibility is invalid")
    if overlap not in _OVERLAP:
        raise ValueError("overlap is invalid")
    if confidence not in _CONFIDENCE:
        raise ValueError("confidence is invalid")
    tags = _parse_tags(tag_text)
    if (overlap == "overlap") != ("overlap" in tags):
        raise ValueError("overlap column and tag must agree")

    if event_type == "speech":
        if not _SPEAKER_RE.fullmatch(speaker):
            raise ValueError("speech speaker is invalid")
        if intelligibility == "not_applicable":
            raise ValueError("speech intelligibility is invalid")
        if policy.speech_text == "forbidden":
            if text:
                raise ValueError("coverage-only speech text must be empty")
        elif intelligibility in {"intelligible", "partially_intelligible"}:
            if not text.strip():
                raise ValueError("intelligible speech text must not be empty")
            if intelligibility == "intelligible" and not normalize_minimal(text):
                raise ValueError("intelligible speech text has no scorable characters")
        elif text:
            raise ValueError("unintelligible speech text must be empty")
        if "non_speech_control" in tags:
            raise ValueError("non_speech_control tag requires non_speech")
    else:
        if policy.speech_text == "forbidden":
            raise ValueError("coverage-only passes must not contain non_speech rows")
        if text:
            raise ValueError("non_speech text must be empty")
        if intelligibility != "not_applicable":
            raise ValueError("non_speech intelligibility must be not_applicable")
        if speaker != "none" or overlap != "none":
            raise ValueError("non_speech speaker and overlap must be none")

    return AnnotationWorksheetRow(
        row_number=row_number,
        span=SampleSpan(timeline_id, start, end),
        event_type=event_type,
        text=text,
        intelligibility=intelligibility,
        speaker=speaker,
        overlap=overlap,
        confidence=confidence,
        acoustic_tags=tags,
        notes=notes,
    )


def parse_annotation_worksheet(
    raw: bytes,
    *,
    timeline_id: str,
    sample_count: int,
    policy: _AnnotationPolicy | None = None,
) -> ParsedAnnotationWorksheet:
    """Parse one exact beginner worksheet byte snapshot."""

    if type(raw) is not bytes:
        raise TypeError("worksheet must be exact bytes")
    require_uuid4(timeline_id, "timeline_id")
    _plain_int(sample_count, "sample_count", minimum=1)
    if policy is None:
        policy = _ANNOTATION_POLICIES[0]
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeError:
        raise ValueError("worksheet is not valid UTF-8") from None
    if "\x00" in decoded:
        raise ValueError("worksheet contains a NUL character")
    reader = csv.reader(
        io.StringIO(decoded, newline=""),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        strict=True,
    )
    parsed_rows: list[AnnotationWorksheetRow] = []
    try:
        header = next(reader)
        if tuple(header) != WORKSHEET_COLUMNS:
            raise ValueError(f"worksheet header is invalid (row={reader.line_num})")
        for values in reader:
            # A spreadsheet may save trailing rows whose fields are all empty.
            if all(not value.strip() for value in values):
                continue
            row_number = reader.line_num
            try:
                row = _parse_row(values, row_number, timeline_id, sample_count, policy)
            except ValueError as exc:
                raise ValueError(f"{exc} (row={row_number})") from None
            parsed_rows.append(row)
    except StopIteration:
        raise ValueError("worksheet header is missing") from None
    except csv.Error:
        raise ValueError(
            f"worksheet TSV is malformed (row={reader.line_num})"
        ) from None
    rows = tuple(parsed_rows)
    if not rows:
        raise ValueError("worksheet must contain at least one annotation row")
    return ParsedAnnotationWorksheet(
        content_sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        rows=rows,
    )


def _read_work(path: Path, boundary: PrivateBoundary) -> dict[str, object]:
    try:
        value = json.loads(
            require_private_path(path, boundary).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("annotation work template is unreadable") from None
    if not isinstance(value, dict) or set(value) != _WORK_FIELDS:
        raise ValueError("annotation work template has an invalid wire shape")
    return value


def _annotation_policy(
    input_roles: object,
    prohibited_inputs: object,
) -> _AnnotationPolicy:
    if type(input_roles) is not list or type(prohibited_inputs) is not list:
        raise ValueError("annotation input policy is invalid")
    for policy in _ANNOTATION_POLICIES:
        if (
            input_roles == list(policy.input_roles)
            and prohibited_inputs == list(policy.prohibited_inputs)
        ):
            return policy
    raise ValueError("annotation input policy is invalid")


def _validate_work(
    work: dict[str, object],
    reference_version: str,
    master_id: str,
    boundary: PrivateBoundary,
) -> _AnnotationPolicy:
    if work["schema_version"] != _WORK_SCHEMA_VERSION:
        raise ValueError("annotation work schema version is unsupported")
    for field_name in (
        "pass_id",
        "annotator_id",
        "packet_id",
        "pilot_id",
        "fixture_id",
        "master_id",
        "timeline_id",
    ):
        require_uuid4(work[field_name], field_name)
    for field_name in ("annotator_slot", "selection_label"):
        _nonempty(work[field_name], field_name)
    _nonempty(reference_version, "reference_version")
    require_uuid4(master_id, "master_id")
    if work["master_id"] != master_id:
        raise ValueError("annotation work belongs to a different master")
    policy = _annotation_policy(work["input_roles"], work["prohibited_inputs"])
    if (
        work["status"] != "awaiting_human"
        or work["completed_at_utc"] is not None
        or work["human_signature_receipt_id"] is not None
        or work["events"] != []
        or work["verbatim_references"] != []
    ):
        raise ValueError("annotation work template was modified")
    if _plain_int(work["sample_rate_hz"], "sample_rate_hz", minimum=1) != 16_000:
        raise ValueError("annotation master must use 16000 Hz")
    _plain_int(work["sample_count"], "sample_count", minimum=1)

    master = load_registered_master(master_id, boundary)
    for field_name in (
        "master_id",
        "fixture_id",
        "timeline_id",
        "sample_rate_hz",
        "sample_count",
    ):
        if work[field_name] != getattr(master, field_name):
            raise ValueError(f"annotation work {field_name} does not match master")
    return policy


def _span_wire(span: SampleSpan) -> dict[str, object]:
    return {"timeline_id": span.timeline_id, "start": span.start, "end": span.end}


def _read_editorial_sidecar(
    path: Path | None,
    work: dict[str, object],
    policy: _AnnotationPolicy,
    boundary: PrivateBoundary,
) -> bytes | None:
    if not policy.subtitle_seeded:
        if path is not None:
            raise ValueError(
                "only subtitle-seeded passes take an editorial subtitle sidecar"
            )
        return None
    if path is None:
        raise ValueError("subtitle-seeded passes require the editorial subtitle sidecar")
    try:
        raw = require_private_path(Path(path), boundary).read_bytes()
    except OSError:
        raise ValueError("editorial subtitle sidecar is unreadable") from None
    from .subtitle_proposals import load_editorial_sidecar

    if load_editorial_sidecar(raw)["master_id"] != work["master_id"]:
        raise ValueError("editorial subtitle sidecar belongs to a different master")
    return raw


def _subtitle_binding(sidecar_raw: bytes | None) -> dict[str, object]:
    if sidecar_raw is None:
        return {"subtitle_proposal_sha256": None, "ffmpeg_sha256": None}
    from .subtitle_proposals import load_editorial_sidecar

    return {
        "subtitle_proposal_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
        "ffmpeg_sha256": load_editorial_sidecar(sidecar_raw)["ffmpeg_sha256"],
    }


def _compile_payloads(
    work: dict[str, object],
    reference_version: str,
    parsed: ParsedAnnotationWorksheet,
    policy: _AnnotationPolicy,
    sidecar_raw: bytes | None,
) -> tuple[dict[str, object], dict[str, object]]:
    events: list[dict[str, object]] = []
    verbatim: list[dict[str, object]] = []
    non_speech: list[dict[str, object]] = []
    texts: dict[str, object] = {}
    for row in parsed.rows:
        note_id = None
        if row.notes:
            note_id = new_opaque_id()
            texts[note_id] = {"role": "annotation_note", "text": row.notes}
        if row.event_type == "speech":
            event_id = new_opaque_id()
            verbatim_ref_id = None
            if row.text:
                verbatim_ref_id = new_opaque_id()
                text_id = new_opaque_id()
                texts[text_id] = {"role": "verbatim_reference", "text": row.text}
                verbatim.append(
                    {
                        "verbatim_ref_id": verbatim_ref_id,
                        "event_ids": [event_id],
                        "text_id": text_id,
                        "annotation_version": reference_version,
                    }
                )
            events.append(
                {
                    "event_id": event_id,
                    "row_number": row.row_number,
                    "span": _span_wire(row.span),
                    "intelligibility": row.intelligibility,
                    "speaker": row.speaker,
                    "overlap": row.overlap,
                    "confidence": row.confidence,
                    "acoustic_tags": list(row.acoustic_tags),
                    "verbatim_ref_id": verbatim_ref_id,
                    "note_id": note_id,
                    "annotation_version": reference_version,
                }
            )
        else:
            non_speech.append(
                {
                    "region_id": new_opaque_id(),
                    "row_number": row.row_number,
                    "span": _span_wire(row.span),
                    "intelligibility": row.intelligibility,
                    "speaker": row.speaker,
                    "overlap": row.overlap,
                    "confidence": row.confidence,
                    "acoustic_tags": list(row.acoustic_tags),
                    "note_id": note_id,
                }
            )

    text_payload: dict[str, object] = {
        "schema_version": _PASS_TEXT_SCHEMA_VERSION,
        "pass_id": work["pass_id"],
        "texts": texts,
    }
    text_bytes = canonical_json_bytes(
        cast(Mapping[str, JsonValue], text_payload),
        newline=True,
    )
    manifest: dict[str, object] = {
        "schema_version": _PASS_SCHEMA_VERSION,
        "work_schema_version": work["schema_version"],
        "pass_id": work["pass_id"],
        "annotator_id": work["annotator_id"],
        "annotator_slot": work["annotator_slot"],
        "packet_id": work["packet_id"],
        "pilot_id": work["pilot_id"],
        "selection_label": work["selection_label"],
        "fixture_id": work["fixture_id"],
        "master_id": work["master_id"],
        "timeline_id": work["timeline_id"],
        "sample_rate_hz": work["sample_rate_hz"],
        "sample_count": work["sample_count"],
        "reference_version": reference_version,
        "input_roles": list(policy.input_roles),
        "prohibited_inputs": list(policy.prohibited_inputs),
        "status": "compiled_unsigned",
        "completed_at_utc": None,
        "human_signature_receipt_id": None,
        "content_sha256": parsed.content_sha256,
        "worksheet_byte_count": parsed.byte_count,
        "pass_text_sha256": hashlib.sha256(text_bytes).hexdigest(),
        "events": events,
        "verbatim_references": verbatim,
        "non_speech_regions": non_speech,
        **_subtitle_binding(sidecar_raw),
    }
    return manifest, text_payload


def _write_private_file(path: Path, data: bytes) -> None:
    """Durably replace ``path``, reporting a failed write without its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_bytes(path, data)
    except OSError:
        raise ValueError("annotation pass bundle write failed") from None


def _closed_mapping(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has an invalid wire shape")
    return value


def _record_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _registered_span(value: object, manifest: Mapping[str, object]) -> SampleSpan:
    item = _closed_mapping(
        value,
        frozenset({"timeline_id", "start", "end"}),
        "annotation span",
    )
    span = SampleSpan(
        cast(str, item["timeline_id"]),
        cast(int, item["start"]),
        cast(int, item["end"]),
    )
    if (
        span.timeline_id != manifest["timeline_id"]
        or span.start < 0
        or span.end > manifest["sample_count"]
    ):
        raise ValueError("annotation pass span is outside its master timeline")
    return span


def load_registered_annotation_pass(
    pass_id: str,
    boundary: PrivateBoundary,
) -> RegisteredAnnotationPass:
    """Reopen and fully verify one compiled unsigned annotation pass."""

    require_uuid4(pass_id, "pass_id")
    if not isinstance(boundary, PrivateBoundary):
        raise TypeError("boundary must be PrivateBoundary")
    directory = require_private_path(
        boundary.root / "annotation-passes" / pass_id,
        boundary,
    )
    try:
        worksheet_raw = (directory / "worksheet.private.tsv").read_bytes()
        text_raw = (directory / "pass-text.private.json").read_bytes()
        manifest_raw = (directory / "pass.private.json").read_bytes()
        text_payload = canonical_json_mapping_from_bytes(
            text_raw,
            "annotation pass text",
        )
        manifest = canonical_json_mapping_from_bytes(
            manifest_raw,
            "annotation pass manifest",
        )
    except (OSError, TypeError, ValueError):
        raise ValueError("registered annotation pass is unreadable") from None
    try:
        _closed_mapping(manifest, _PASS_FIELDS, "annotation pass manifest")
        policy = _annotation_policy(
            manifest["input_roles"],
            manifest["prohibited_inputs"],
        )
        if (
            manifest["schema_version"] != _PASS_SCHEMA_VERSION
            or manifest["work_schema_version"] != _WORK_SCHEMA_VERSION
            or manifest["pass_id"] != pass_id
            or manifest["status"] != "compiled_unsigned"
            or manifest["completed_at_utc"] is not None
            or manifest["human_signature_receipt_id"] is not None
            or manifest["sample_rate_hz"] != 16_000
        ):
            raise ValueError("annotation pass identity or policy is invalid")
        for field_name in (
            "pass_id",
            "annotator_id",
            "packet_id",
            "pilot_id",
            "fixture_id",
            "master_id",
            "timeline_id",
        ):
            require_uuid4(manifest[field_name], field_name)
        _nonempty(manifest["annotator_slot"], "annotator_slot")
        _nonempty(manifest["selection_label"], "selection_label")
        reference_version = _nonempty(
            manifest["reference_version"],
            "reference_version",
        )
        sample_count = _plain_int(
            manifest["sample_count"],
            "sample_count",
            minimum=1,
        )
        master = load_registered_master(cast(str, manifest["master_id"]), boundary)
        for field_name in (
            "master_id",
            "fixture_id",
            "timeline_id",
            "sample_rate_hz",
            "sample_count",
        ):
            if manifest[field_name] != getattr(master, field_name):
                raise ValueError(f"annotation pass {field_name} does not match master")
        if (
            manifest["worksheet_byte_count"] != len(worksheet_raw)
            or manifest["content_sha256"] != hashlib.sha256(worksheet_raw).hexdigest()
            or manifest["pass_text_sha256"] != hashlib.sha256(text_raw).hexdigest()
        ):
            raise ValueError("annotation pass byte hash does not match")
        editorial_sidecar: dict[str, object] | None = None
        if policy.subtitle_seeded:
            try:
                sidecar_raw = (directory / _EDITORIAL_SIDECAR_NAME).read_bytes()
            except OSError:
                raise ValueError("annotation pass editorial sidecar is unreadable") from None
            from .subtitle_proposals import load_editorial_sidecar

            editorial_sidecar = load_editorial_sidecar(sidecar_raw)
            if (
                editorial_sidecar["master_id"] != manifest["master_id"]
                or {
                    name: manifest[name]
                    for name in ("subtitle_proposal_sha256", "ffmpeg_sha256")
                }
                != _subtitle_binding(sidecar_raw)
            ):
                raise ValueError("annotation pass editorial sidecar does not match")
        elif _subtitle_binding(None) != {
            name: manifest[name] for name in ("subtitle_proposal_sha256", "ffmpeg_sha256")
        }:
            raise ValueError("annotation pass has an unexpected editorial sidecar")

        text_root = _closed_mapping(
            text_payload,
            frozenset({"schema_version", "pass_id", "texts"}),
            "annotation pass text",
        )
        if (
            text_root["schema_version"] != _PASS_TEXT_SCHEMA_VERSION
            or text_root["pass_id"] != pass_id
            or not isinstance(text_root["texts"], dict)
        ):
            raise ValueError("annotation pass text identity is invalid")
        raw_texts = cast(dict[object, object], text_root["texts"])
        texts: dict[str, str] = {}
        roles: dict[str, str] = {}
        for raw_text_id, raw_record in raw_texts.items():
            text_id = require_uuid4(raw_text_id, "text_id")
            record = _closed_mapping(
                raw_record,
                frozenset({"role", "text"}),
                "annotation private text",
            )
            role = record["role"]
            text = record["text"]
            if role not in {"verbatim_reference", "annotation_note"}:
                raise ValueError("annotation private text role is invalid")
            if type(text) is not str or not text:
                raise ValueError("annotation private text must not be empty")
            roles[text_id] = cast(str, role)
            texts[text_id] = text

        events = _record_list(manifest["events"], "annotation events")
        verbatim_records = _record_list(
            manifest["verbatim_references"],
            "annotation verbatim references",
        )
        non_speech = _record_list(
            manifest["non_speech_regions"],
            "annotation non-speech regions",
        )
        verbatim_by_id: dict[str, dict[str, object]] = {}
        expected_text_roles: dict[str, str] = {}

        def expect_text(text_id: str, role: str) -> None:
            if text_id in expected_text_roles:
                raise ValueError("annotation pass reuses a private text ID")
            expected_text_roles[text_id] = role

        for raw_record in verbatim_records:
            record = _closed_mapping(
                raw_record,
                _VERBATIM_FIELDS,
                "annotation verbatim reference",
            )
            verbatim_id = require_uuid4(record["verbatim_ref_id"], "verbatim_ref_id")
            text_id = require_uuid4(record["text_id"], "text_id")
            event_ids = record["event_ids"]
            if (
                verbatim_id in verbatim_by_id
                or text_id in expected_text_roles
                or not isinstance(event_ids, list)
                or len(event_ids) != 1
                or record["annotation_version"] != reference_version
            ):
                raise ValueError("annotation verbatim reference is invalid")
            require_uuid4(event_ids[0], "event_id")
            verbatim_by_id[verbatim_id] = record
            expect_text(text_id, "verbatim_reference")

        event_ids: set[str] = set()
        used_verbatim_ids: set[str] = set()
        row_numbers: set[int] = set()
        events_by_row: dict[int, dict[str, object]] = {}
        for raw_event in events:
            event = _closed_mapping(raw_event, _EVENT_FIELDS, "annotation event")
            event_id = require_uuid4(event["event_id"], "event_id")
            row_number = _plain_int(event["row_number"], "row_number", minimum=2)
            if event_id in event_ids or row_number in row_numbers:
                raise ValueError("annotation event identity is duplicated")
            event_ids.add(event_id)
            row_numbers.add(row_number)
            events_by_row[row_number] = event
            _registered_span(event["span"], manifest)
            if (
                event["intelligibility"] not in _INTELLIGIBILITY - {"not_applicable"}
                or not _SPEAKER_RE.fullmatch(cast(str, event["speaker"]))
                or event["overlap"] not in _OVERLAP
                or event["confidence"] not in _CONFIDENCE
                or event["annotation_version"] != reference_version
            ):
                raise ValueError("annotation event labels are invalid")
            tags = event["acoustic_tags"]
            if not isinstance(tags, list) or _parse_tags(";".join(tags)) != tuple(tags):
                raise ValueError("annotation event tags are invalid")
            if (event["overlap"] == "overlap") != ("overlap" in tags):
                raise ValueError("annotation event overlap is invalid")
            verbatim_id = event["verbatim_ref_id"]
            if verbatim_id is not None:
                linked_id = require_uuid4(verbatim_id, "verbatim_ref_id")
                linked = verbatim_by_id.get(linked_id)
                if linked is None or linked["event_ids"] != [event_id]:
                    raise ValueError("annotation event text linkage is invalid")
                used_verbatim_ids.add(linked_id)
            note_id = event["note_id"]
            if note_id is not None:
                expect_text(require_uuid4(note_id, "note_id"), "annotation_note")

        if {
            cast(str, record["event_ids"][0]) for record in verbatim_by_id.values()
        } - event_ids or used_verbatim_ids != set(verbatim_by_id):
            raise ValueError("annotation verbatim reference names an unknown event")

        region_ids: set[str] = set()
        regions_by_row: dict[int, dict[str, object]] = {}
        for raw_region in non_speech:
            region = _closed_mapping(
                raw_region,
                _NON_SPEECH_FIELDS,
                "annotation non-speech region",
            )
            region_id = require_uuid4(region["region_id"], "region_id")
            row_number = _plain_int(region["row_number"], "row_number", minimum=2)
            if region_id in region_ids or row_number in row_numbers:
                raise ValueError("annotation row identity is duplicated")
            region_ids.add(region_id)
            row_numbers.add(row_number)
            regions_by_row[row_number] = region
            _registered_span(region["span"], manifest)
            tags = region["acoustic_tags"]
            if (
                region["intelligibility"] != "not_applicable"
                or region["speaker"] != "none"
                or region["overlap"] != "none"
                or region["confidence"] not in _CONFIDENCE
                or not isinstance(tags, list)
                or _parse_tags(";".join(tags)) != tuple(tags)
            ):
                raise ValueError("annotation non-speech labels are invalid")
            note_id = region["note_id"]
            if note_id is not None:
                expect_text(require_uuid4(note_id, "note_id"), "annotation_note")
        if not row_numbers or roles != expected_text_roles:
            raise ValueError("annotation pass private text coverage is invalid")

        parsed = parse_annotation_worksheet(
            worksheet_raw,
            timeline_id=cast(str, manifest["timeline_id"]),
            sample_count=sample_count,
            policy=policy,
        )
        if {row.row_number for row in parsed.rows} != row_numbers:
            raise ValueError("annotation pass rows do not match worksheet")
        for row in parsed.rows:
            record = (
                events_by_row.get(row.row_number)
                if row.event_type == "speech"
                else regions_by_row.get(row.row_number)
            )
            if record is None or (
                record["span"] != _span_wire(row.span)
                or record["intelligibility"] != row.intelligibility
                or record["speaker"] != row.speaker
                or record["overlap"] != row.overlap
                or record["confidence"] != row.confidence
                or record["acoustic_tags"] != list(row.acoustic_tags)
            ):
                raise ValueError("annotation pass row does not match worksheet")
            note_id = record["note_id"]
            if row.notes:
                if note_id is None or texts.get(cast(str, note_id)) != row.notes:
                    raise ValueError("annotation pass note does not match worksheet")
            elif note_id is not None:
                raise ValueError("annotation pass has an unexpected note")
            if row.event_type == "speech":
                verbatim_id = record["verbatim_ref_id"]
                if row.text:
                    linked = verbatim_by_id.get(cast(str, verbatim_id))
                    if linked is None or texts.get(cast(str, linked["text_id"])) != row.text:
                        raise ValueError("annotation pass text does not match worksheet")
                elif verbatim_id is not None:
                    raise ValueError("annotation pass has unexpected verbatim text")

        annotation_pass = AnnotationPass(
            pass_id=pass_id,
            annotator_id=cast(str, manifest["annotator_id"]),
            reference_version=reference_version,
            content_sha256=cast(str, manifest["content_sha256"]),
            input_roles=tuple(
                AnnotationInputRole(value) for value in policy.input_roles
            ),
        )
    except (TypeError, ValueError, KeyError):
        raise ValueError("registered annotation pass is invalid") from None
    bundle = AnnotationPassBundle(
        pass_id=pass_id,
        directory=directory,
        content_sha256=hashlib.sha256(worksheet_raw).hexdigest(),
        pass_text_sha256=hashlib.sha256(text_raw).hexdigest(),
        pass_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
    return RegisteredAnnotationPass(
        annotation_pass=annotation_pass,
        status="compiled_unsigned",
        bundle=bundle,
        manifest=manifest,
        texts=texts,
        editorial_sidecar=editorial_sidecar,
    )


def _bundle_from_disk(
    directory: Path,
    expected_raw: bytes,
    work: dict[str, object],
    reference_version: str,
    boundary: PrivateBoundary,
    policy: _AnnotationPolicy,
    sidecar_raw: bytes | None,
) -> AnnotationPassBundle:
    worksheet_path = directory / "worksheet.private.tsv"
    text_path = directory / "pass-text.private.json"
    manifest_path = directory / "pass.private.json"
    try:
        worksheet_raw = worksheet_path.read_bytes()
        text_raw = text_path.read_bytes()
        manifest_raw = manifest_path.read_bytes()
        text_payload = canonical_json_mapping_from_bytes(
            text_raw, "annotation pass text"
        )
        manifest = canonical_json_mapping_from_bytes(
            manifest_raw, "annotation pass manifest"
        )
    except (OSError, TypeError, ValueError):
        raise ValueError("conflicting or incomplete annotation pass bundle") from None
    content_sha256 = hashlib.sha256(worksheet_raw).hexdigest()
    pass_text_sha256 = hashlib.sha256(text_raw).hexdigest()
    expected_identity = {
        "pass_id": work["pass_id"],
        "annotator_id": work["annotator_id"],
        "annotator_slot": work["annotator_slot"],
        "packet_id": work["packet_id"],
        "pilot_id": work["pilot_id"],
        "selection_label": work["selection_label"],
        "fixture_id": work["fixture_id"],
        "master_id": work["master_id"],
        "timeline_id": work["timeline_id"],
        "sample_rate_hz": work["sample_rate_hz"],
        "sample_count": work["sample_count"],
        "reference_version": reference_version,
        "input_roles": list(policy.input_roles),
        "prohibited_inputs": list(policy.prohibited_inputs),
        "status": "compiled_unsigned",
        "completed_at_utc": None,
        "human_signature_receipt_id": None,
        **_subtitle_binding(sidecar_raw),
    }
    if (
        worksheet_raw != expected_raw
        or manifest.get("schema_version") != _PASS_SCHEMA_VERSION
        or manifest.get("work_schema_version") != _WORK_SCHEMA_VERSION
        or manifest.get("content_sha256") != content_sha256
        or manifest.get("worksheet_byte_count") != len(worksheet_raw)
        or manifest.get("pass_text_sha256") != pass_text_sha256
        or text_payload.get("schema_version") != _PASS_TEXT_SCHEMA_VERSION
        or text_payload.get("pass_id") != work["pass_id"]
        or any(manifest.get(name) != value for name, value in expected_identity.items())
    ):
        raise ValueError("conflicting annotation pass reuse")
    return load_registered_annotation_pass(cast(str, work["pass_id"]), boundary).bundle


def import_annotation_pass(
    worksheet_path: Path,
    work_path: Path,
    reference_version: str,
    master_id: str,
    boundary: PrivateBoundary,
    *,
    editorial_sidecar_path: Path | None = None,
) -> AnnotationPassBundle:
    """Register one exact returned worksheet as an unsigned private pass bundle.

    A subtitle-seeded pass also takes the editorial sidecar its proposal
    worksheet came from; the bundle keeps an exact copy of it.
    """

    if not isinstance(boundary, PrivateBoundary):
        raise TypeError("boundary must be PrivateBoundary")
    worksheet = require_private_path(Path(worksheet_path), boundary)
    work = _read_work(Path(work_path), boundary)
    policy = _validate_work(work, reference_version, master_id, boundary)
    sidecar_raw = _read_editorial_sidecar(
        editorial_sidecar_path,
        work,
        policy,
        boundary,
    )
    try:
        raw = worksheet.read_bytes()
    except OSError:
        raise ValueError("annotation worksheet is unreadable") from None
    parsed = parse_annotation_worksheet(
        raw,
        timeline_id=cast(str, work["timeline_id"]),
        sample_count=cast(int, work["sample_count"]),
        policy=policy,
    )
    pass_id = cast(str, work["pass_id"])
    directory = require_private_path(
        boundary.root / "annotation-passes" / pass_id,
        boundary,
    )
    manifest_path = directory / "pass.private.json"
    if manifest_path.exists():
        return _bundle_from_disk(
            directory,
            raw,
            work,
            reference_version,
            boundary,
            policy,
            sidecar_raw,
        )
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("conflicting or incomplete annotation pass bundle")

    manifest, text_payload = _compile_payloads(
        work,
        reference_version,
        parsed,
        policy,
        sidecar_raw,
    )
    worksheet_target = directory / "worksheet.private.tsv"
    text_target = directory / "pass-text.private.json"
    manifest_target = directory / "pass.private.json"
    _write_private_file(worksheet_target, raw)
    _write_private_file(
        text_target,
        canonical_json_bytes(
            cast(Mapping[str, JsonValue], text_payload),
            newline=True,
        ),
    )
    if sidecar_raw is not None:
        _write_private_file(directory / _EDITORIAL_SIDECAR_NAME, sidecar_raw)
    # The structural manifest is the bundle's commit marker and is written last.
    _write_private_file(
        manifest_target,
        canonical_json_bytes(cast(Mapping[str, JsonValue], manifest), newline=True),
    )
    return _bundle_from_disk(
        directory,
        raw,
        work,
        reference_version,
        boundary,
        policy,
        sidecar_raw,
    )


def assemble_reference_draft_from_pass(
    pass_id: str,
    reference_set_id: str,
    draft_path: Path,
    text_path: Path,
    boundary: PrivateBoundary,
) -> ReferenceSetManifest:
    """Build and validate a non-adjudicated draft from one registered pass."""

    require_uuid4(reference_set_id, "reference_set_id")
    registered = load_registered_annotation_pass(pass_id, boundary)
    master_id = cast(str, registered.manifest["master_id"])
    master = load_registered_master(master_id, boundary)
    if requires_double_annotation(master.partition):
        raise ValueError(
            f"{master.partition.value} reference requires two adjudicated passes"
        )

    draft_target = require_private_path(Path(draft_path), boundary)
    text_target = require_private_path(Path(text_path), boundary)
    if draft_target == text_target:
        raise ValueError("reference draft and text sidecar paths must differ")

    events = [
        {
            "event_id": event["event_id"],
            "master_id": master.master_id,
            "span": event["span"],
            "intelligibility": event["intelligibility"],
            "verbatim_ref_id": event["verbatim_ref_id"],
            "tags": event["acoustic_tags"],
            "annotation_version": event["annotation_version"],
        }
        for event in cast(list[dict[str, object]], registered.manifest["events"])
    ]
    verbatim = cast(
        list[dict[str, object]],
        registered.manifest["verbatim_references"],
    )
    non_speech_controls = [
        {
            "region_id": region["region_id"],
            "span": region["span"],
            "tags": region["acoustic_tags"],
        }
        for region in cast(
            list[dict[str, object]],
            registered.manifest["non_speech_regions"],
        )
    ]
    annotation_pass = registered.annotation_pass
    texts: dict[str, object] = {}
    for reference in verbatim:
        text_id = cast(str, reference["text_id"])
        texts[text_id] = {
            "role": "verbatim_reference",
            "text": registered.texts[text_id],
        }
    # Each editorial cue maps to every speech event its span positively
    # overlaps; a cue that overlaps no event is dropped.
    editorial_subtitles: list[dict[str, object]] = []
    sidecar = registered.editorial_sidecar
    cues = [] if sidecar is None else cast(list[dict[str, object]], sidecar["cues"])
    for cue in cues:
        cue_start = cast(int, cue["relative_start_ms"]) * _SAMPLES_PER_MILLISECOND
        cue_end = cast(int, cue["relative_end_ms"]) * _SAMPLES_PER_MILLISECOND
        overlapping: list[object] = []
        for event in events:
            span = cast(dict[str, int], event["span"])
            if span["start"] < cue_end and cue_start < span["end"]:
                overlapping.append(event["event_id"])
        if not overlapping:
            continue
        text_id = new_opaque_id()
        editorial_subtitles.append(
            {
                "subtitle_id": new_opaque_id(),
                "event_ids": overlapping,
                "text_id": text_id,
                "annotation_version": annotation_pass.reference_version,
            }
        )
        texts[text_id] = {"role": "editorial_subtitle", "text": cue["raw_text"]}
    draft: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "reference_set_id": reference_set_id,
        "fixture_id": master.fixture_id,
        "master_id": master.master_id,
        "partition": master.partition.value,
        "reference_version": annotation_pass.reference_version,
        "reference_sha256": "0" * 64,
        "reference_text_sha256": "0" * 64,
        "reference_set_sha256": "0" * 64,
        "events": events,
        "verbatim_references": verbatim,
        "editorial_subtitles": editorial_subtitles,
        "annotation_passes": [
            {
                "pass_id": annotation_pass.pass_id,
                "annotator_id": annotation_pass.annotator_id,
                "reference_version": annotation_pass.reference_version,
                "content_sha256": annotation_pass.content_sha256,
                "input_roles": [role.value for role in annotation_pass.input_roles],
            }
        ],
        "adjudication": None,
        "non_speech_controls": non_speech_controls,
    }
    text_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "reference_set_id": reference_set_id,
        "texts": texts,
    }

    _write_private_file(
        text_target,
        canonical_json_bytes(cast(Mapping[str, JsonValue], text_payload), newline=True),
    )
    _write_private_file(
        draft_target,
        canonical_json_bytes(cast(Mapping[str, JsonValue], draft), newline=True),
    )
    from .registry import validate_reference_draft

    return validate_reference_draft(
        draft_target,
        text_target,
        master.master_id,
        boundary,
    )


def verify_reference_annotation_passes(
    manifest: ReferenceSetManifest,
    boundary: PrivateBoundary,
) -> None:
    """Bind every declared reference pass to its exact registered bundle."""

    if not isinstance(manifest, ReferenceSetManifest):
        raise TypeError("manifest must be ReferenceSetManifest")
    for declared in manifest.annotation_passes:
        registered = load_registered_annotation_pass(declared.pass_id, boundary)
        if registered.annotation_pass != declared:
            raise ValueError("declared annotation pass does not match registered bundle")
        if (
            registered.manifest["master_id"] != manifest.master_id
            or registered.manifest["fixture_id"] != manifest.fixture_id
        ):
            raise ValueError("declared annotation pass belongs to another reference")


__all__ = (
    "AnnotationPassBundle",
    "AnnotationWorksheetRow",
    "ParsedAnnotationWorksheet",
    "RegisteredAnnotationPass",
    "WORKSHEET_COLUMNS",
    "assemble_reference_draft_from_pass",
    "import_annotation_pass",
    "load_registered_annotation_pass",
    "parse_annotation_worksheet",
    "verify_reference_annotation_passes",
)
