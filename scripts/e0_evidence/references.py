"""Closed, layered reference contracts for E0 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence, TypeVar, cast

from scripts.e0_evidence._common import require_sha256, require_uuid4
from scripts.e0_evidence.pcm import CanonicalMaster
from scripts.e0_evidence.schema import (
    SCHEMA_VERSION,
    FixtureManifest,
    JsonValue,
    Partition,
    SampleSpan,
    validate_sample_span,
)
from scripts.e0_evidence.wire import canonical_sha256


_UNBOUND_REFERENCE_TEXT_SHA256 = "0" * 64
_RecordT = TypeVar("_RecordT")

# Closed vocabulary for event and non-speech-control acoustic tags. Worksheet
# import, frozen references, and the declared hard stratum share this set.
ACOUSTIC_TAGS = frozenset(
    {
        "background_speech",
        "breath",
        "code_switch",
        "effects_noise",
        "high_f0_nonmodal",
        "incomplete_word",
        "laugh",
        "long_endpoint",
        "loud_dialogue",
        "music",
        "music_effects_noise",
        "non_japanese_speech",
        "non_speech_control",
        "overlap",
        "quiet_dialogue",
        "rapid_turns",
        "short_acknowledgements",
        "singing",
    }
)


class AnnotationInputRole(str, Enum):
    AUDIO_ONLY = "audio_only"
    EXISTING_SUBTITLE = "existing_subtitle"
    SPEECH_EVENT_LAYER = "speech_event_layer"
    INDEPENDENT_PASSES = "independent_passes"


def _nonempty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _strings(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be strings")
    result = tuple(values)  # type: ignore[arg-type]
    for index, value in enumerate(result):
        _nonempty(value, f"{field_name}[{index}]")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result  # type: ignore[return-value]


def _opaque_ids(values: object, field_name: str) -> tuple[str, ...]:
    result = _strings(values, field_name)
    for index, value in enumerate(result):
        require_uuid4(value, f"{field_name}[{index}]")
    return result


def _acoustic_tags(values: object, field_name: str) -> tuple[str, ...]:
    tags = _strings(values, field_name)
    if any(tag not in ACOUSTIC_TAGS for tag in tags):
        raise ValueError(f"{field_name} must contain only acoustic tags")
    return tags


def _records(
    values: object,
    record_type: type[_RecordT],
    field_name: str,
) -> tuple[_RecordT, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be {record_type.__name__} records")
    result = tuple(values)  # type: ignore[arg-type]
    for index, value in enumerate(result):
        if not isinstance(value, record_type):
            raise TypeError(f"{field_name}[{index}] must be {record_type.__name__}")
    return result


def _input_roles(values: object, field_name: str) -> tuple[AnnotationInputRole, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be AnnotationInputRole values")
    roles = tuple(values)  # type: ignore[arg-type]
    for index, role in enumerate(roles):
        if not isinstance(role, AnnotationInputRole):
            raise TypeError(f"{field_name}[{index}] must be AnnotationInputRole")
    if not roles:
        raise ValueError(f"{field_name} must not be empty")
    if len(roles) != len(set(roles)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return roles


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    event_id: str
    master_id: str
    span: SampleSpan
    intelligibility: str
    verbatim_ref_id: str | None
    tags: tuple[str, ...]
    annotation_version: str

    def __post_init__(self) -> None:
        require_uuid4(self.event_id, "event_id")
        require_uuid4(self.master_id, "master_id")
        if not isinstance(self.span, SampleSpan):
            raise TypeError("span must be SampleSpan")
        _nonempty(self.intelligibility, "intelligibility")
        if self.verbatim_ref_id is not None:
            require_uuid4(self.verbatim_ref_id, "verbatim_ref_id")
        object.__setattr__(self, "tags", _acoustic_tags(self.tags, "tags"))
        _nonempty(self.annotation_version, "annotation_version")


@dataclass(frozen=True, slots=True)
class NonSpeechControl:
    """An annotated region without speech, on the master timeline."""

    region_id: str
    span: SampleSpan
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        require_uuid4(self.region_id, "region_id")
        if not isinstance(self.span, SampleSpan):
            raise TypeError("span must be SampleSpan")
        object.__setattr__(self, "tags", _acoustic_tags(self.tags, "tags"))


@dataclass(frozen=True, slots=True)
class VerbatimReference:
    verbatim_ref_id: str
    event_ids: tuple[str, ...]
    text_id: str
    annotation_version: str

    def __post_init__(self) -> None:
        require_uuid4(self.verbatim_ref_id, "verbatim_ref_id")
        event_ids = _opaque_ids(self.event_ids, "event_ids")
        if not event_ids:
            raise ValueError("event_ids must not be empty")
        object.__setattr__(self, "event_ids", event_ids)
        require_uuid4(self.text_id, "text_id")
        _nonempty(self.annotation_version, "annotation_version")


@dataclass(frozen=True, slots=True)
class EditorialSubtitle:
    subtitle_id: str
    event_ids: tuple[str, ...]
    text_id: str
    annotation_version: str

    def __post_init__(self) -> None:
        require_uuid4(self.subtitle_id, "subtitle_id")
        event_ids = _opaque_ids(self.event_ids, "event_ids")
        if not event_ids:
            raise ValueError("event_ids must not be empty")
        object.__setattr__(self, "event_ids", event_ids)
        require_uuid4(self.text_id, "text_id")
        _nonempty(self.annotation_version, "annotation_version")


@dataclass(frozen=True, slots=True)
class AnnotationPass:
    pass_id: str
    annotator_id: str
    reference_version: str
    content_sha256: str
    input_roles: tuple[AnnotationInputRole, ...]

    def __post_init__(self) -> None:
        require_uuid4(self.pass_id, "pass_id")
        require_uuid4(self.annotator_id, "annotator_id")
        _nonempty(self.reference_version, "reference_version")
        require_sha256(self.content_sha256, "content_sha256")
        roles = _input_roles(self.input_roles, "input_roles")
        if AnnotationInputRole.AUDIO_ONLY not in roles:
            raise ValueError("independent annotation must include audio_only input")
        if AnnotationInputRole.INDEPENDENT_PASSES in roles:
            raise ValueError("independent annotation cannot consume another pass")
        object.__setattr__(self, "input_roles", roles)


@dataclass(frozen=True, slots=True)
class Adjudication:
    adjudication_id: str
    adjudicator_id: str
    input_pass_ids: tuple[str, ...]
    input_roles: tuple[AnnotationInputRole, ...]
    output_reference_version: str
    output_content_sha256: str

    def __post_init__(self) -> None:
        require_uuid4(self.adjudication_id, "adjudication_id")
        require_uuid4(self.adjudicator_id, "adjudicator_id")
        pass_ids = _opaque_ids(self.input_pass_ids, "input_pass_ids")
        if not pass_ids:
            raise ValueError("input_pass_ids must not be empty")
        object.__setattr__(self, "input_pass_ids", pass_ids)
        roles = _input_roles(self.input_roles, "input_roles")
        if AnnotationInputRole.INDEPENDENT_PASSES not in roles:
            raise ValueError("adjudication must consume independent passes")
        object.__setattr__(self, "input_roles", roles)
        _nonempty(self.output_reference_version, "output_reference_version")
        require_sha256(self.output_content_sha256, "output_content_sha256")


@dataclass(frozen=True, slots=True)
class ReferenceSetManifest:
    schema_version: str
    reference_set_id: str
    fixture_id: str
    master_id: str
    partition: Partition
    reference_version: str
    reference_sha256: str
    reference_set_sha256: str
    events: tuple[ReferenceEvent, ...]
    verbatim_references: tuple[VerbatimReference, ...]
    editorial_subtitles: tuple[EditorialSubtitle, ...]
    annotation_passes: tuple[AnnotationPass, ...]
    adjudication: Adjudication | None
    non_speech_controls: tuple[NonSpeechControl, ...]
    reference_text_sha256: str = _UNBOUND_REFERENCE_TEXT_SHA256

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        require_uuid4(self.reference_set_id, "reference_set_id")
        require_uuid4(self.fixture_id, "fixture_id")
        require_uuid4(self.master_id, "master_id")
        if not isinstance(self.partition, Partition):
            raise TypeError("partition must be Partition")
        _nonempty(self.reference_version, "reference_version")
        require_sha256(self.reference_sha256, "reference_sha256")
        require_sha256(self.reference_set_sha256, "reference_set_sha256")
        require_sha256(self.reference_text_sha256, "reference_text_sha256")
        object.__setattr__(
            self,
            "events",
            _records(self.events, ReferenceEvent, "events"),
        )
        object.__setattr__(
            self,
            "verbatim_references",
            _records(
                self.verbatim_references,
                VerbatimReference,
                "verbatim_references",
            ),
        )
        object.__setattr__(
            self,
            "editorial_subtitles",
            _records(
                self.editorial_subtitles,
                EditorialSubtitle,
                "editorial_subtitles",
            ),
        )
        object.__setattr__(
            self,
            "annotation_passes",
            _records(
                self.annotation_passes,
                AnnotationPass,
                "annotation_passes",
            ),
        )
        if self.adjudication is not None and not isinstance(
            self.adjudication,
            Adjudication,
        ):
            raise TypeError("adjudication must be Adjudication or None")
        object.__setattr__(
            self,
            "non_speech_controls",
            _records(
                self.non_speech_controls,
                NonSpeechControl,
                "non_speech_controls",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferenceSet:
    manifest: ReferenceSetManifest
    events: tuple[ReferenceEvent, ...]
    verbatim_references: tuple[VerbatimReference, ...]
    editorial_subtitles: tuple[EditorialSubtitle, ...]
    non_speech_controls: tuple[NonSpeechControl, ...]


def _span_wire(span: SampleSpan) -> Mapping[str, JsonValue]:
    return {
        "end": span.end,
        "start": span.start,
        "timeline_id": span.timeline_id,
    }


def reference_content_sha256(manifest: ReferenceSetManifest) -> str:
    if not isinstance(manifest, ReferenceSetManifest):
        raise TypeError("manifest must be ReferenceSetManifest")
    payload = cast(
        Mapping[str, JsonValue],
        {
            "editorial_subtitles": [
                {
                    "annotation_version": unit.annotation_version,
                    "event_ids": list(unit.event_ids),
                    "subtitle_id": unit.subtitle_id,
                    "text_id": unit.text_id,
                }
                for unit in manifest.editorial_subtitles
            ],
            "events": [
                {
                    "annotation_version": event.annotation_version,
                    "event_id": event.event_id,
                    "intelligibility": event.intelligibility,
                    "master_id": event.master_id,
                    "span": _span_wire(event.span),
                    "tags": list(event.tags),
                    "verbatim_ref_id": event.verbatim_ref_id,
                }
                for event in manifest.events
            ],
            "fixture_id": manifest.fixture_id,
            "master_id": manifest.master_id,
            "partition": manifest.partition.value,
            "non_speech_controls": [
                {
                    "region_id": control.region_id,
                    "span": _span_wire(control.span),
                    "tags": list(control.tags),
                }
                for control in sorted(
                    manifest.non_speech_controls,
                    key=lambda item: (item.span.start, item.region_id),
                )
            ],
            "reference_set_id": manifest.reference_set_id,
            "reference_version": manifest.reference_version,
            "schema_version": manifest.schema_version,
            "verbatim_references": [
                {
                    "annotation_version": reference.annotation_version,
                    "event_ids": list(reference.event_ids),
                    "text_id": reference.text_id,
                    "verbatim_ref_id": reference.verbatim_ref_id,
                }
                for reference in manifest.verbatim_references
            ],
        },
    )
    return canonical_sha256(payload)


def reference_set_manifest_sha256(manifest: ReferenceSetManifest) -> str:
    if not isinstance(manifest, ReferenceSetManifest):
        raise TypeError("manifest must be ReferenceSetManifest")
    adjudication = manifest.adjudication
    payload = cast(
        Mapping[str, JsonValue],
        {
            "adjudication": (
                None
                if adjudication is None
                else {
                    "adjudication_id": adjudication.adjudication_id,
                    "adjudicator_id": adjudication.adjudicator_id,
                    "input_pass_ids": list(adjudication.input_pass_ids),
                    "input_roles": [role.value for role in adjudication.input_roles],
                    "output_content_sha256": adjudication.output_content_sha256,
                    "output_reference_version": (adjudication.output_reference_version),
                }
            ),
            "annotation_passes": [
                {
                    "annotator_id": item.annotator_id,
                    "content_sha256": item.content_sha256,
                    "input_roles": [role.value for role in item.input_roles],
                    "pass_id": item.pass_id,
                    "reference_version": item.reference_version,
                }
                for item in manifest.annotation_passes
            ],
            "hash_kind": "e0-reference-set-manifest-4",
            "reference_content_sha256": manifest.reference_sha256,
            "reference_text_sha256": manifest.reference_text_sha256,
        },
    )
    return canonical_sha256(payload)


def adjudicated_reference_content_sha256(manifest: ReferenceSetManifest) -> str:
    """Bind adjudication to both reference structure and exact private text."""

    if not isinstance(manifest, ReferenceSetManifest):
        raise TypeError("manifest must be ReferenceSetManifest")
    structural_sha256 = reference_content_sha256(manifest)
    if manifest.reference_text_sha256 == _UNBOUND_REFERENCE_TEXT_SHA256:
        # Wire 1.0 manifests written before transcript-sidecar binding used the
        # structural digest directly. Registry-frozen manifests always take
        # the bound branch below.
        return structural_sha256
    return canonical_sha256(
        {
            "reference_content_sha256": structural_sha256,
            "reference_text_sha256": manifest.reference_text_sha256,
        }
    )


_DOUBLE_ANNOTATED_PARTITIONS = frozenset(
    (Partition.BLOCKER_CLASS, Partition.LOCKED_CONFIRMATION)
)


def requires_double_annotation(partition: Partition) -> bool:
    """Blocker and locked references need adjudicated seeded and coverage passes."""

    if not isinstance(partition, Partition):
        raise TypeError("partition must be Partition")
    return partition in _DOUBLE_ANNOTATED_PARTITIONS


def load_reference_set(
    manifest: ReferenceSetManifest,
    master: CanonicalMaster,
) -> ReferenceSet:
    if not isinstance(manifest, ReferenceSetManifest):
        raise TypeError("manifest must be ReferenceSetManifest")
    if not isinstance(master, CanonicalMaster):
        raise TypeError("master must be CanonicalMaster")
    master_manifest = master.manifest
    if manifest.master_id != master_manifest.master_id:
        raise ValueError("reference set belongs to a different master")
    if manifest.fixture_id != master_manifest.fixture_id:
        raise ValueError("reference set belongs to a different fixture")
    if manifest.partition is not master_manifest.partition:
        raise ValueError("reference partition does not match master partition")

    content_sha256 = reference_content_sha256(manifest)
    if content_sha256 != manifest.reference_sha256:
        raise ValueError("reference content SHA-256 does not match manifest")

    event_ids = tuple(event.event_id for event in manifest.events)
    if not event_ids or len(event_ids) != len(set(event_ids)):
        raise ValueError("reference event IDs must be nonempty and unique")
    for event in manifest.events:
        if event.master_id != master_manifest.master_id:
            raise ValueError("reference event belongs to a different master")
        try:
            validate_sample_span(
                event.span,
                {master_manifest.timeline_id: master_manifest.sample_count},
            )
        except (TypeError, ValueError):
            raise ValueError("reference event span is invalid") from None

    control_ids = tuple(control.region_id for control in manifest.non_speech_controls)
    if len(control_ids) != len(set(control_ids)):
        raise ValueError("non-speech control IDs must be unique")
    for control in manifest.non_speech_controls:
        try:
            validate_sample_span(
                control.span,
                {master_manifest.timeline_id: master_manifest.sample_count},
            )
        except (TypeError, ValueError):
            raise ValueError(
                "non-speech control span is outside the master timeline"
            ) from None
        if any(
            control.span.start < event.span.end and event.span.start < control.span.end
            for event in manifest.events
        ):
            raise ValueError("non-speech control overlaps a reference event")

    verbatim_ids = tuple(
        reference.verbatim_ref_id for reference in manifest.verbatim_references
    )
    if len(verbatim_ids) != len(set(verbatim_ids)):
        raise ValueError("verbatim reference IDs must be unique")
    verbatim_by_id = {
        reference.verbatim_ref_id: reference
        for reference in manifest.verbatim_references
    }
    event_by_id = {event.event_id: event for event in manifest.events}
    known_event_ids = set(event_ids)
    for event in manifest.events:
        if event.verbatim_ref_id is None:
            continue
        reference = verbatim_by_id.get(event.verbatim_ref_id)
        if reference is None or event.event_id not in reference.event_ids:
            raise ValueError("speech event and verbatim layers are inconsistent")
    for reference in manifest.verbatim_references:
        if not set(reference.event_ids).issubset(known_event_ids):
            raise ValueError("verbatim reference names an unknown speech event")
        if any(
            event_by_id[event_id].verbatim_ref_id != reference.verbatim_ref_id
            for event_id in reference.event_ids
        ):
            raise ValueError("speech event and verbatim layers are inconsistent")

    subtitle_ids = tuple(
        subtitle.subtitle_id for subtitle in manifest.editorial_subtitles
    )
    if len(subtitle_ids) != len(set(subtitle_ids)):
        raise ValueError("editorial subtitle IDs must be unique")
    for subtitle in manifest.editorial_subtitles:
        if not set(subtitle.event_ids).issubset(known_event_ids):
            raise ValueError("editorial subtitle names an unknown speech event")

    pass_ids = tuple(item.pass_id for item in manifest.annotation_passes)
    annotator_ids = tuple(item.annotator_id for item in manifest.annotation_passes)
    if len(pass_ids) != len(set(pass_ids)):
        raise ValueError("annotation pass IDs must be unique")
    if len(annotator_ids) != len(set(annotator_ids)):
        raise ValueError("independent annotation requires distinct annotators")

    required = requires_double_annotation(manifest.partition)
    if required and len(manifest.annotation_passes) != 2:
        raise ValueError("double annotation requires exactly two independent passes")
    if required and {
        frozenset(annotation_pass.input_roles)
        for annotation_pass in manifest.annotation_passes
    } != {
        frozenset(
            (
                AnnotationInputRole.AUDIO_ONLY,
                AnnotationInputRole.EXISTING_SUBTITLE,
            )
        ),
        frozenset(
            (
                AnnotationInputRole.AUDIO_ONLY,
                AnnotationInputRole.SPEECH_EVENT_LAYER,
            )
        ),
    }:
        raise ValueError(
            "double annotation requires asymmetric subtitle-seeded and coverage-only passes"
        )
    if required and manifest.adjudication is None:
        raise ValueError("required reference adjudication is missing")

    adjudication = manifest.adjudication
    if adjudication is not None:
        if len(adjudication.input_pass_ids) != 2 or set(
            adjudication.input_pass_ids
        ) != set(pass_ids):
            raise ValueError("adjudication passes do not match independent passes")
        if adjudication.output_reference_version != manifest.reference_version:
            raise ValueError("adjudication output version does not match reference")
        if adjudication.output_content_sha256 != adjudicated_reference_content_sha256(
            manifest
        ):
            raise ValueError("adjudication output hash does not match reference")
        if manifest.reference_version in {
            item.reference_version for item in manifest.annotation_passes
        }:
            raise ValueError("adjudication must produce a new reference version")

    reference_set_sha256 = reference_set_manifest_sha256(manifest)
    if reference_set_sha256 != manifest.reference_set_sha256:
        raise ValueError("reference set SHA-256 does not match manifest")
    if reference_set_sha256 != master_manifest.annotation_set_sha256:
        raise ValueError("reference set SHA-256 does not match master annotation set")

    return ReferenceSet(
        manifest=manifest,
        events=manifest.events,
        verbatim_references=manifest.verbatim_references,
        editorial_subtitles=manifest.editorial_subtitles,
        non_speech_controls=manifest.non_speech_controls,
    )


def validate_partition_split(fixtures: Sequence[FixtureManifest]) -> None:
    if isinstance(fixtures, (str, bytes)):
        raise TypeError("fixtures must be FixtureManifest records")
    development_titles: set[str] = set()
    locked_titles: set[str] = set()
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, FixtureManifest):
            raise TypeError(f"fixtures[{index}] must be FixtureManifest")
        if fixture.partition is Partition.DEVELOPMENT:
            development_titles.add(fixture.title_id)
        elif fixture.partition is Partition.LOCKED_CONFIRMATION:
            locked_titles.add(fixture.title_id)
    if development_titles & locked_titles:
        raise ValueError("development and locked confirmation have title overlap")
