from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.e0_evidence.schema import (
    ArmDefinition,
    ArmKind,
    CapabilityDecision,
    FixtureManifest,
    Partition,
    PrivateContentClass,
    RightsRecord,
    SCHEMA_VERSION,
    SampleSpan,
    validate_sample_span,
)
from scripts.e0_evidence.wire import (
    arm_definition_sha256,
    canonical_json_bytes,
    evidence_record_from_wire,
    evidence_record_to_wire,
    inherited_classification,
)
from scripts.e0_evidence.assets import (
    register_fixture,
    resolve_fixture_source,
    resolve_private_locator_asset,
)
from scripts.e0_evidence.private_root import (
    PRIVATE_ROOT_ENV,
    PrivateBoundary,
    resolve_private_root,
)
from scripts.e0_evidence.pcm import (
    MasterManifest,
    MasterTransformDag,
    TransformStep,
    load_canonical_master,
    transform_step_sha256,
)
from scripts.e0_evidence.references import (
    Adjudication,
    AnnotationInputRole,
    AnnotationPass,
    EditorialSubtitle,
    ReferenceEvent,
    ReferenceSetManifest,
    VerbatimReference,
    load_reference_set,
    reference_content_sha256,
    reference_set_manifest_sha256,
    validate_partition_split,
)


FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "e0-evidence-wire-1.0.synthetic.jsonl"
)
MASTER_TIMELINES = {"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa": 1_024}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _task2b_contract(
    tmp_path: Path,
) -> tuple[
    PrivateBoundary,
    Path,
    np.ndarray,
    MasterManifest,
    ReferenceSetManifest,
]:
    repo = tmp_path / "repo"
    private = tmp_path / "private"
    repo.mkdir()
    private.mkdir()
    boundary = PrivateBoundary(
        root=private,
        worktrees=(repo,),
        excluded_roots=(),
    )

    source_artifact_id = "90000000-0000-4000-8000-000000000001"
    source_native_id = "90000000-0000-4000-8000-000000000002"
    master_id = "90000000-0000-4000-8000-000000000003"
    fixture_id = "90000000-0000-4000-8000-000000000004"
    timeline_id = "90000000-0000-4000-8000-000000000005"
    expected = np.asarray([0.0, 0.25, -0.5, 1.0], dtype="<f4")
    pcm_bytes = expected.tobytes(order="C")
    pcm_path = private / "masters" / f"{master_id}.f32le"
    pcm_path.parent.mkdir()
    pcm_path.write_bytes(pcm_bytes)

    source_media_hash = _digest(b"synthetic-source-media")
    source_native_samples = np.linspace(-1.0, 1.0, 12, dtype="<f4")
    source_native_bytes = source_native_samples.tobytes(order="C")
    source_native_path = private / "masters" / f"{source_native_id}.f32le"
    source_native_path.write_bytes(source_native_bytes)
    source_native_hash = _digest(source_native_bytes)
    canonical_hash = _digest(pcm_bytes)
    source_native_step = TransformStep(
        transform_id="90000000-0000-4000-8000-000000000006",
        transform_version="ffmpeg-decode-1.0",
        transform_sha256="0" * 64,
        parent_artifact_id=source_artifact_id,
        parent_byte_sha256=source_media_hash,
        output_artifact_id=source_native_id,
        output_byte_sha256=source_native_hash,
        tool="ffmpeg",
        tool_version="ffmpeg version synthetic-test",
        command_argv=(
            "ffmpeg",
            "-i",
            "<source>",
            "-map",
            "0:a:1",
            "-af",
            "anull",
            "-f",
            "f32le",
            "<source-native>",
        ),
        filter_graph="anull",
        stream_index=1,
        excerpt_start_samples=0,
        excerpt_end_samples=12,
        input_sample_rate_hz=48_000,
        input_channels=1,
        input_layout="mono",
        output_sample_rate_hz=48_000,
        output_channels=1,
        output_layout="mono",
        output_sample_count=12,
        channel_operation="preserve",
        resampler="none",
        resampler_parameters={},
        dtype_conversion="little-endian-float32",
    )
    source_native_step = replace(
        source_native_step,
        transform_sha256=transform_step_sha256(source_native_step),
    )
    canonical_step = TransformStep(
        transform_id="90000000-0000-4000-8000-000000000007",
        transform_version="canonical-pcm-1.0",
        transform_sha256="0" * 64,
        parent_artifact_id=source_native_id,
        parent_byte_sha256=source_native_hash,
        output_artifact_id=master_id,
        output_byte_sha256=canonical_hash,
        tool="ffmpeg",
        tool_version="ffmpeg version synthetic-test",
        command_argv=(
            "ffmpeg",
            "-f",
            "f32le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
            "<source-native>",
            "-af",
            (
                "aresample=isr=48000:osr=16000:resampler=soxr:"
                "precision=28:cutoff=0.91:cheby=0:async=0"
            ),
            "-f",
            "f32le",
            "<canonical>",
        ),
        filter_graph=(
            "aresample=isr=48000:osr=16000:resampler=soxr:"
            "precision=28:cutoff=0.91:cheby=0:async=0"
        ),
        stream_index=0,
        excerpt_start_samples=0,
        excerpt_end_samples=12,
        input_sample_rate_hz=48_000,
        input_channels=1,
        input_layout="mono",
        output_sample_rate_hz=16_000,
        output_channels=1,
        output_layout="mono",
        output_sample_count=4,
        channel_operation="mono-pass-through",
        resampler="soxr",
        resampler_parameters={
            "async": 0,
            "cheby": 0,
            "cutoff": 0.91,
            "isr": 48_000,
            "osr": 16_000,
            "precision": 28,
        },
        dtype_conversion="little-endian-float32",
    )
    canonical_step = replace(
        canonical_step,
        transform_sha256=transform_step_sha256(canonical_step),
    )

    event_id = "90000000-0000-4000-8000-000000000008"
    verbatim_ref_id = "90000000-0000-4000-8000-000000000009"
    event = ReferenceEvent(
        event_id=event_id,
        master_id=master_id,
        span=SampleSpan(timeline_id, 0, 2),
        intelligibility="intelligible",
        verbatim_ref_id=verbatim_ref_id,
        tags=("quiet_dialogue",),
        annotation_version="event-1.0",
    )
    verbatim = VerbatimReference(
        verbatim_ref_id=verbatim_ref_id,
        event_ids=(event_id,),
        text_id="90000000-0000-4000-8000-000000000010",
        annotation_version="verbatim-1.0",
    )
    editorial = EditorialSubtitle(
        subtitle_id="90000000-0000-4000-8000-000000000011",
        event_ids=(event_id,),
        text_id="90000000-0000-4000-8000-000000000012",
        annotation_version="editorial-1.0",
    )
    annotation_passes = (
        AnnotationPass(
            pass_id="90000000-0000-4000-8000-000000000013",
            annotator_id="90000000-0000-4000-8000-000000000014",
            reference_version="pass-a-1.0",
            content_sha256=_digest(b"independent-pass-a"),
            input_roles=(
                AnnotationInputRole.AUDIO_ONLY,
                AnnotationInputRole.EXISTING_SUBTITLE,
            ),
        ),
        AnnotationPass(
            pass_id="90000000-0000-4000-8000-000000000015",
            annotator_id="90000000-0000-4000-8000-000000000016",
            reference_version="pass-b-1.0",
            content_sha256=_digest(b"independent-pass-b"),
            input_roles=(
                AnnotationInputRole.AUDIO_ONLY,
                AnnotationInputRole.SPEECH_EVENT_LAYER,
            ),
        ),
    )
    references = ReferenceSetManifest(
        schema_version=SCHEMA_VERSION,
        reference_set_id="90000000-0000-4000-8000-000000000017",
        fixture_id=fixture_id,
        master_id=master_id,
        partition=Partition.BLOCKER_CLASS,
        reference_version="adjudicated-1.0",
        reference_sha256="0" * 64,
        reference_set_sha256="0" * 64,
        events=(event,),
        verbatim_references=(verbatim,),
        editorial_subtitles=(editorial,),
        annotation_passes=annotation_passes,
        adjudication=None,
        non_speech_controls=(),
    )
    reference_hash = reference_content_sha256(references)
    references = replace(
        references,
        reference_sha256=reference_hash,
        adjudication=Adjudication(
            adjudication_id="90000000-0000-4000-8000-000000000018",
            adjudicator_id="90000000-0000-4000-8000-000000000019",
            input_pass_ids=tuple(item.pass_id for item in annotation_passes),
            input_roles=(
                AnnotationInputRole.AUDIO_ONLY,
                AnnotationInputRole.INDEPENDENT_PASSES,
            ),
            output_reference_version="adjudicated-1.0",
            output_content_sha256=reference_hash,
        ),
    )
    reference_set_hash = reference_set_manifest_sha256(references)
    references = replace(
        references,
        reference_set_sha256=reference_set_hash,
    )
    master = MasterManifest(
        schema_version=SCHEMA_VERSION,
        master_id=master_id,
        fixture_id=fixture_id,
        rights=RightsRecord(
            rights_record_id="90000000-0000-4000-8000-000000000029",
            fixture_id=fixture_id,
            local_evaluation=CapabilityDecision.APPROVED,
            redistribution=CapabilityDecision.PENDING,
            training=CapabilityDecision.PENDING,
            transcript_publication=CapabilityDecision.PENDING,
            reviewer_id="90000000-0000-4000-8000-000000000030",
            evidence_ids=(
                "90000000-0000-4000-8000-000000000031",
            ),
            retention_version="retention-1.0",
        ),
        partition=Partition.BLOCKER_CLASS,
        timeline_id=timeline_id,
        parent_artifact_ids=(source_native_id,),
        byte_sha256=canonical_hash,
        canonical_pcm_sha256=canonical_hash,
        sample_rate_hz=16_000,
        channels=1,
        layout="mono",
        dtype="<f4",
        sample_count=int(expected.size),
        duration_samples=int(expected.size),
        transform_dag=MasterTransformDag(
            source_native=source_native_step,
            canonical=canonical_step,
        ),
        provenance_class="controlled-source-decode",
        rights_class="local-evaluation-approved",
        retention_version="retention-1.0",
        annotation_set_sha256=reference_set_hash,
    )

    source = private / "source.mkv"
    source.write_bytes(b"synthetic-source-media")
    (private / "locator.private.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "fixtures": {
                    fixture_id: {
                        "source": str(source.resolve()),
                        "masters": {
                            source_native_id: str(source_native_path.resolve()),
                            master_id: str(pcm_path.resolve()),
                        },
                    }
                },
            }
        ),
        "utf-8",
    )
    return boundary, pcm_path, expected, master, references


def test_source_native_v1_1_requires_sample_exact_trim(tmp_path: Path) -> None:
    boundary, _, _, master, _ = _task2b_contract(tmp_path)
    source_native = master.transform_dag.source_native
    trim_filter = (
        f"atrim=start_sample={source_native.excerpt_start_samples}:"
        f"end_sample={source_native.excerpt_end_samples},asetpts=N/SR/TB"
    )
    exact = replace(
        source_native,
        transform_version="ffmpeg-decode-1.1",
        transform_sha256="0" * 64,
        command_argv=(
            "ffmpeg",
            "-i",
            "<source>",
            "-map",
            "0:a:1",
            "-af",
            trim_filter,
            "-f",
            "f32le",
            "<source-native>",
        ),
        filter_graph=trim_filter,
    )
    exact = replace(exact, transform_sha256=transform_step_sha256(exact))
    exact_master = replace(
        master,
        transform_dag=replace(master.transform_dag, source_native=exact),
    )

    loaded = load_canonical_master(exact_master, boundary)

    assert loaded.manifest == exact_master

    inexact = replace(exact, filter_graph="anull", transform_sha256="0" * 64)
    inexact = replace(inexact, transform_sha256=transform_step_sha256(inexact))
    inexact_master = replace(
        master,
        transform_dag=replace(master.transform_dag, source_native=inexact),
    )
    with pytest.raises(ValueError, match="sample-exact trim"):
        load_canonical_master(inexact_master, boundary)

    untrimmed_command = replace(
        exact,
        command_argv=tuple(
            "anull" if item == trim_filter else item for item in exact.command_argv
        ),
        transform_sha256="0" * 64,
    )
    untrimmed_command = replace(
        untrimmed_command,
        transform_sha256=transform_step_sha256(untrimmed_command),
    )
    untrimmed_master = replace(
        master,
        transform_dag=replace(
            master.transform_dag,
            source_native=untrimmed_command,
        ),
    )
    with pytest.raises(ValueError, match="command.*sample-exact trim"):
        load_canonical_master(untrimmed_master, boundary)


@pytest.mark.parametrize(
    ("span_args", "timelines", "expected_error", "expected_sample_count"),
    [
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", True, 10),
            MASTER_TIMELINES,
            TypeError,
            None,
            id="boolean-start",
        ),
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 0, False),
            MASTER_TIMELINES,
            TypeError,
            None,
            id="boolean-end",
        ),
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", -1, 10),
            MASTER_TIMELINES,
            ValueError,
            None,
            id="negative-start",
        ),
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 10, 10),
            MASTER_TIMELINES,
            ValueError,
            None,
            id="empty-range",
        ),
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 11, 10),
            MASTER_TIMELINES,
            ValueError,
            None,
            id="reversed-range",
        ),
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 1_020, 1_024),
            MASTER_TIMELINES,
            None,
            4,
            id="exclusive-end-at-master-boundary",
        ),
        pytest.param(
            ("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", 0, 1),
            MASTER_TIMELINES,
            ValueError,
            None,
            id="unknown-timeline",
        ),
        pytest.param(
            ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 1_023, 1_025),
            MASTER_TIMELINES,
            ValueError,
            None,
            id="end-past-master",
        ),
    ],
)
def test_sample_spans_enforce_half_open_integer_master_bounds(
    span_args: tuple[str, object, object],
    timelines: dict[str, int],
    expected_error: type[Exception] | None,
    expected_sample_count: int | None,
) -> None:
    if expected_error is not None:
        with pytest.raises(expected_error):
            span = SampleSpan(*span_args)
            validate_sample_span(span, timelines)
        return

    span = SampleSpan(*span_args)
    validate_sample_span(span, timelines)

    assert span.sample_count == expected_sample_count


def test_golden_record_round_trips_without_losing_wire_meaning() -> None:
    fixture_bytes = FIXTURE.read_bytes()
    wire_value = json.loads(fixture_bytes)

    record = evidence_record_from_wire(wire_value)
    encoded = evidence_record_to_wire(record)

    assert record.arm_id == "11111111-1111-4111-8111-111111111111"
    assert record.clock_id == "perf_counter_ns"
    assert [
        (span.timeline_id, span.start, span.end)
        for span in record.source_spans
    ] == [
        ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 0, 160),
        ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 320, 480),
    ]
    assert encoded == wire_value
    assert canonical_json_bytes(encoded, newline=True) == fixture_bytes

    unknown_stage = copy.deepcopy(wire_value)
    unknown_stage["stage"] = "future_stage_with_changed_meaning"
    with pytest.raises(ValueError, match="stage"):
        evidence_record_from_wire(unknown_stage)

    for field_name in ("parent_ids", "diagnostics"):
        duplicate_wire = copy.deepcopy(wire_value)
        duplicate_wire[field_name].append(duplicate_wire[field_name][0])
        with pytest.raises(ValueError, match=field_name):
            evidence_record_from_wire(duplicate_wire)

        existing_values = getattr(record, field_name)
        with pytest.raises(ValueError, match=field_name):
            replace(
                record,
                **{field_name: existing_values + (existing_values[0],)},
            )


def test_arm_hash_covers_nested_parameters_and_privacy_inherits_restriction() -> None:
    mutable_parameters = {
        "decoder": {
            "beam_size": 5,
            "gates": {"no_speech_threshold": 0.6},
        }
    }
    definition = ArmDefinition(
        arm_id="22222222-2222-4222-8222-222222222222",
        kind=ArmKind.P2B,
        parent_arm_id="11111111-1111-4111-8111-111111111111",
        parameters=mutable_parameters,
        declared_changes=("decoder.gates.no_speech_threshold",),
    )
    changed_definition = ArmDefinition(
        arm_id=definition.arm_id,
        kind=definition.kind,
        parent_arm_id=definition.parent_arm_id,
        parameters={
            "decoder": {
                "beam_size": 5,
                "gates": {"no_speech_threshold": None},
            }
        },
        declared_changes=definition.declared_changes,
    )

    original_hash = arm_definition_sha256(definition)
    mutable_parameters["decoder"]["gates"]["no_speech_threshold"] = 0.9

    assert definition.parameters["decoder"]["gates"]["no_speech_threshold"] == 0.6
    assert arm_definition_sha256(definition) == original_hash
    assert arm_definition_sha256(changed_definition) != original_hash
    with pytest.raises(TypeError):
        definition.parameters["decoder"]["gates"]["no_speech_threshold"] = None

    assert inherited_classification(
        PrivateContentClass.PUBLIC_ALLOWLISTED,
        (
            PrivateContentClass.INTERNAL_AGGREGATE,
            PrivateContentClass.PRIVATE_TEXT,
        ),
    ) is PrivateContentClass.PRIVATE_TEXT
    assert inherited_classification(
        PrivateContentClass.PRIVATE_AUDIO,
        (PrivateContentClass.PUBLIC_ALLOWLISTED,),
    ) is PrivateContentClass.PRIVATE_AUDIO


@pytest.mark.parametrize(
    "case",
    [
        "repo-root",
        "tilde-relative-inputs",
        "worktree-root",
        "excluded-root",
        "external-root",
        "malformed-root-manifest",
        "malformed-locator-entry",
        "redacted-locator-error",
        "redacted-write-error",
        "other-capability",
        "rights-before-poison-locator",
    ],
)
def test_private_boundary_and_rights_fail_closed_before_locator_access(
    tmp_path: Path,
    case: str,
) -> None:
    repo = tmp_path / "repo"
    worktree = repo / "worktree"
    excluded = tmp_path / "excluded"
    external = tmp_path / "private"
    for directory in (repo, worktree, excluded, external):
        directory.mkdir(parents=True, exist_ok=True)

    if case == "tilde-relative-inputs":
        with pytest.raises(ValueError, match="absolute"):
            resolve_private_root(
                repo,
                {"YAMABIKO_E0_PRIVATE_ROOT": "~"},
                worktrees=(repo, worktree),
                excluded_roots=(excluded,),
            )

    if case == "repo-root":
        with pytest.raises(ValueError, match="Git worktree"):
            resolve_private_root(
                repo,
                {"YAMABIKO_E0_PRIVATE_ROOT": str(repo)},
                worktrees=(),
                excluded_roots=(excluded,),
            )
        return

    if case == "worktree-root":
        candidate = worktree / "private"
        candidate.mkdir()
        with pytest.raises(ValueError, match="Git worktree"):
            resolve_private_root(
                repo,
                {"YAMABIKO_E0_PRIVATE_ROOT": str(candidate)},
                worktrees=(repo, worktree),
                excluded_roots=(excluded,),
            )
        return

    if case == "excluded-root":
        candidate = excluded / "private"
        candidate.mkdir()
        with pytest.raises(ValueError, match="excluded root"):
            resolve_private_root(
                repo,
                {"YAMABIKO_E0_PRIVATE_ROOT": str(candidate)},
                worktrees=(repo, worktree),
                excluded_roots=(excluded,),
            )
        return

    root = resolve_private_root(
        repo,
        {"YAMABIKO_E0_PRIVATE_ROOT": str(external)},
        worktrees=(repo, worktree),
        excluded_roots=(excluded,),
    )
    boundary = PrivateBoundary(
        root=root,
        worktrees=(repo, worktree),
        excluded_roots=(excluded,),
    )

    if case == "tilde-relative-inputs":
        with pytest.raises(ValueError, match="explicit absolute"):
            register_fixture(
                Path("~/relative-source.mkv"),
                Partition.BLOCKER_CLASS,
                boundary,
            )

        source = tmp_path / "source.mkv"
        source.write_bytes(b"synthetic-not-media")
        (external / "locator.private.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "fixtures": {
                        "99999999-9999-4999-8999-999999999999": {
                            "source": "~/relative-source.mkv",
                        },
                    },
                }
            ),
            "utf-8",
        )
        with pytest.raises(ValueError, match="absolute"):
            register_fixture(source, Partition.BLOCKER_CLASS, boundary)
        assert not (external / "root.private.json").exists()
        return

    if case == "malformed-root-manifest":
        source = tmp_path / "source.mkv"
        source.write_bytes(b"synthetic-not-media")
        (external / "root.private.json").write_text("not-json", "utf-8")

        with pytest.raises(ValueError, match="root manifest"):
            register_fixture(source, Partition.BLOCKER_CLASS, boundary)

        assert not (external / "locator.private.json").exists()
        assert not (external / "fixtures").exists()
        assert not (external / "rights").exists()
        return

    if case == "malformed-locator-entry":
        source = tmp_path / "source.mkv"
        source.write_bytes(b"synthetic-not-media")
        locator_path = external / "locator.private.json"
        malformed_entries = (
            ("not-an-opaque-id", str(source.resolve())),
            ("99999999-9999-4999-8999-999999999999", "relative.mkv"),
        )

        for fixture_id, located_source in malformed_entries:
            locator_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "fixtures": {
                            fixture_id: {"source": located_source},
                        },
                    }
                ),
                "utf-8",
            )
            with pytest.raises(ValueError, match="locator"):
                register_fixture(source, Partition.BLOCKER_CLASS, boundary)
            assert not (external / "root.private.json").exists()
            assert not (external / "fixtures").exists()
            assert not (external / "rights").exists()
        return

    if case == "redacted-write-error":
        source = tmp_path / "source.mkv"
        source.write_bytes(b"synthetic-not-media")
        (external / "fixtures").write_bytes(b"poison")

        with pytest.raises(ValueError, match="record write failed") as caught:
            register_fixture(source, Partition.BLOCKER_CLASS, boundary)

        assert str(external) not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        return

    if case == "external-root":
        from scripts.e0_evidence.report import (
            PUBLIC_EXPORT_PROFILE_SHA256,
            PUBLIC_EXPORT_PROFILE_VERSION,
            PublicExportPrivacyError,
            PublicExportReview,
            build_public_aggregate,
            private_report_sha256,
        )

        source = tmp_path / "source.mkv"
        source.write_bytes(b"synthetic-not-media")

        fixture = register_fixture(source, Partition.BLOCKER_CLASS, boundary)

        fixture_file = external / "fixtures" / fixture.fixture_id / "fixture.private.json"
        rights_file = external / "rights" / f"{fixture.rights_record_id}.private.json"
        locator = json.loads((external / "locator.private.json").read_text("utf-8"))
        rights = json.loads(rights_file.read_text("utf-8"))
        assert fixture_file.is_file()
        assert locator["fixtures"][fixture.fixture_id]["source"] == str(source.resolve())
        assert rights["local_evaluation"] == "pending"
        assert rights["redistribution"] == "pending"
        assert rights["training"] == "pending"
        assert rights["transcript_publication"] == "pending"

        arm_id = "34444444-4444-4444-8444-444444444444"
        private_payload = {
            "schema_version": "e0-private-report-2.0",
            "classification": "private_text",
            "report_id": "35555555-5555-4555-8555-555555555555",
            "score_bundle_sha256": "1" * 64,
            "cohort_id": "36666666-6666-4666-8666-666666666666",
            "cohort_fixture_ids": [fixture.fixture_id],
            "cohort_manifest_sha256": "2" * 64,
            "model_runtime_sha256": "3" * 64,
            "partition": "blocker_class",
            "sources": [{"run_id": "37777777-7777-4777-8777-777777777777"}],
            "score_sources": [
                {
                    "arm_id": arm_id,
                    "arm_kind": "p0",
                    "source_run_ids": ["37777777-7777-4777-8777-777777777777"],
                }
            ],
            "integrities": [{"complete": True}],
            "score_reports": [
                {
                    "arm_id": arm_id,
                    "fixture_count": 1,
                    "source_path": str(source.resolve()),
                    "private_text": "公開してはいけない秘密の台詞",
                }
            ],
        }
        review = PublicExportReview(
            review_id="31111111-1111-4111-8111-111111111111",
            reviewer_id="32222222-2222-4222-8222-222222222222",
            evidence_ids=("33333333-3333-4333-8333-333333333333",),
            decision=CapabilityDecision.APPROVED,
            source_report_sha256=private_report_sha256(private_payload),
            profile_version=PUBLIC_EXPORT_PROFILE_VERSION,
            profile_sha256=PUBLIC_EXPORT_PROFILE_SHA256,
            approved_fields=("score.fixture_count",),
        )

        aggregate = build_public_aggregate(private_payload, review)
        aggregate_bytes = canonical_json_bytes(aggregate, newline=True)

        assert aggregate["reports"] == [
            {"arm_kind": "p0", "fields": {"score.fixture_count": 1}}
        ]
        assert str(source.resolve()).encode("utf-8") not in aggregate_bytes
        assert "秘密の台詞".encode("utf-8") not in aggregate_bytes
        for private_field in ("score.source_path", "score.private_text"):
            with pytest.raises(PublicExportPrivacyError):
                build_public_aggregate(
                    private_payload,
                    replace(review, approved_fields=(private_field,)),
                )
        return

    fixture_id = "55555555-5555-4555-8555-555555555555"
    evidence_id = "66666666-6666-4666-8666-666666666666"
    reviewer_id = "77777777-7777-4777-8777-777777777777"
    local_decision = (
        CapabilityDecision.DENIED
        if case == "rights-before-poison-locator"
        else CapabilityDecision.PENDING
    )
    rights = RightsRecord(
        rights_record_id="88888888-8888-4888-8888-888888888888",
        fixture_id=fixture_id,
        local_evaluation=local_decision,
        redistribution=CapabilityDecision.APPROVED,
        training=CapabilityDecision.PENDING,
        transcript_publication=CapabilityDecision.PENDING,
        reviewer_id=reviewer_id,
        evidence_ids=(evidence_id,),
        retention_version="retention-1.0",
    )

    if case == "other-capability":
        with pytest.raises(PermissionError, match="local evaluation"):
            resolve_fixture_source(fixture_id, rights, boundary)
        return

    if case == "redacted-locator-error":
        poison_locator = external / "locator.private.json"
        poison_locator.mkdir()
        approved = replace(
            rights,
            local_evaluation=CapabilityDecision.APPROVED,
        )

        with pytest.raises(ValueError, match="locator is unreadable") as caught:
            resolve_fixture_source(fixture_id, approved, boundary)

        assert str(external) not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        return

    poison_locator = external / "locator.private.json"
    poison_locator.mkdir()

    with pytest.raises(PermissionError, match="local evaluation"):
        resolve_fixture_source(fixture_id, rights, boundary)
    with pytest.raises(PermissionError, match="local evaluation"):
        resolve_private_locator_asset(
            fixture_id,
            rights,
            "masters",
            "99999999-9999-4999-8999-999999999998",
            boundary,
        )


def test_root_init_exclusion_persists_for_a_boundary_built_without_the_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.e0_evidence.cli import main
    from scripts.e0_evidence.registry import boundary_from_environment

    repo = Path(__file__).resolve().parents[2]
    excluded = tmp_path / "excluded"
    excluded.mkdir()
    source = excluded / "source.mkv"
    source.write_bytes(b"synthetic-not-media")
    monkeypatch.setenv(PRIVATE_ROOT_ENV, str(tmp_path / "private"))
    assert main(("root-init", "--exclude-root", str(excluded))) == 0

    boundary = boundary_from_environment(repo, os.environ)

    with pytest.raises(ValueError, match="excluded root"):
        register_fixture(source, Partition.BLOCKER_CLASS, boundary)



@pytest.mark.parametrize(
    "case",
    [
        "positive",
        "positive-stereo",
        "master-rights-before-locator",
        "altered-byte",
        "truncated-pcm",
        "count-mismatch",
        "dtype-mismatch",
        "rate-mismatch",
        "non-finite",
        "source-parent-byte-change",
        "transform-parent-mismatch",
        "transform-hash-mismatch",
        "transform-cycle",
        "transform-semantic-mismatch",
        "invalid-event-span",
        "verbatim-link-mismatch",
        "candidate-output-leakage",
        "missing-adjudication",
        "locked-missing-adjudication",
        "dependent-annotation-pass",
        "two-seeded-annotation-passes",
        "two-coverage-only-annotation-passes",
        "duplicate-annotator",
        "wrong-adjudication-passes",
        "title-overlap",
    ],
)
def test_immutable_master_and_reference_integrity(
    tmp_path: Path,
    case: str,
) -> None:
    if case == "candidate-output-leakage":
        leaked_event = {
            "event_id": "90000000-0000-4000-8000-000000000020",
            "master_id": "90000000-0000-4000-8000-000000000021",
            "span": SampleSpan(
                "90000000-0000-4000-8000-000000000022",
                0,
                1,
            ),
            "intelligibility": "intelligible",
            "verbatim_ref_id": None,
            "tags": (),
            "annotation_version": "event-1.0",
            "candidate_output": "must never enter a reference",
        }
        with pytest.raises(TypeError, match="candidate_output"):
            ReferenceEvent(**leaked_event)
        with pytest.raises(ValueError, match="candidate_output"):
            AnnotationInputRole("candidate_output")
        return

    if case == "title-overlap":
        shared_title_id = "90000000-0000-4000-8000-000000000023"
        fixtures = (
            FixtureManifest(
                fixture_id="90000000-0000-4000-8000-000000000024",
                partition=Partition.DEVELOPMENT,
                title_id=shared_title_id,
                rights_record_id="90000000-0000-4000-8000-000000000025",
            ),
            FixtureManifest(
                fixture_id="90000000-0000-4000-8000-000000000026",
                partition=Partition.LOCKED_CONFIRMATION,
                title_id=shared_title_id,
                rights_record_id="90000000-0000-4000-8000-000000000027",
            ),
        )
        with pytest.raises(ValueError, match="title overlap"):
            validate_partition_split(fixtures)
        return

    boundary, pcm_path, expected, master, references = _task2b_contract(tmp_path)

    if case == "master-rights-before-locator":
        locator = boundary.root / "locator.private.json"
        locator.unlink()
        locator.mkdir()
        master = replace(
            master,
            rights=replace(
                master.rights,
                local_evaluation=CapabilityDecision.DENIED,
            ),
        )
        with pytest.raises(PermissionError, match="local evaluation"):
            load_canonical_master(master, boundary)
        return

    def manifest_for_bytes(raw: bytes, sample_count: int) -> MasterManifest:
        raw_hash = _digest(raw)
        canonical_step = replace(
            master.transform_dag.canonical,
            transform_sha256="0" * 64,
            output_byte_sha256=raw_hash,
        )
        canonical_step = replace(
            canonical_step,
            transform_sha256=transform_step_sha256(canonical_step),
        )
        return replace(
            master,
            byte_sha256=raw_hash,
            canonical_pcm_sha256=raw_hash,
            sample_count=sample_count,
            duration_samples=sample_count,
            transform_dag=replace(
                master.transform_dag,
                canonical=canonical_step,
            ),
        )

    def bind_reference_contract(
        reference_manifest: ReferenceSetManifest,
        master_manifest: MasterManifest,
    ) -> tuple[ReferenceSetManifest, MasterManifest]:
        content_hash = reference_content_sha256(reference_manifest)
        adjudication = reference_manifest.adjudication
        if adjudication is not None:
            adjudication = replace(
                adjudication,
                output_content_sha256=content_hash,
            )
        reference_manifest = replace(
            reference_manifest,
            reference_sha256=content_hash,
            adjudication=adjudication,
        )
        reference_set_hash = reference_set_manifest_sha256(reference_manifest)
        reference_manifest = replace(
            reference_manifest,
            reference_set_sha256=reference_set_hash,
        )
        master_manifest = replace(
            master_manifest,
            annotation_set_sha256=reference_set_hash,
        )
        return reference_manifest, master_manifest

    if case == "positive-stereo":
        source_native = master.transform_dag.source_native
        source_native_bytes = np.linspace(
            -1.0,
            1.0,
            source_native.output_sample_count * 2,
            dtype="<f4",
        ).tobytes(order="C")
        source_native_path = (
            boundary.root
            / "masters"
            / f"{source_native.output_artifact_id}.f32le"
        )
        source_native_path.write_bytes(source_native_bytes)
        source_native = replace(
            source_native,
            transform_sha256="0" * 64,
            output_byte_sha256=_digest(source_native_bytes),
            input_channels=2,
            input_layout="stereo",
            output_channels=2,
            output_layout="stereo",
        )
        source_native = replace(
            source_native,
            transform_sha256=transform_step_sha256(source_native),
        )
        stereo_filter = (
            "pan=mono|c0=0.5*c0+0.5*c1,"
            "aresample=isr=48000:osr=16000:resampler=soxr:"
            "precision=28:cutoff=0.91:cheby=0:async=0"
        )
        canonical_step = replace(
            master.transform_dag.canonical,
            transform_sha256="0" * 64,
            parent_byte_sha256=source_native.output_byte_sha256,
            command_argv=(
                "ffmpeg",
                "-f",
                "f32le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                "<source-native>",
                "-af",
                stereo_filter,
                "-f",
                "f32le",
                "<canonical>",
            ),
            filter_graph=stereo_filter,
            input_channels=2,
            input_layout="stereo",
            channel_operation="stereo-0.5l-plus-0.5r",
        )
        canonical_step = replace(
            canonical_step,
            transform_sha256=transform_step_sha256(canonical_step),
        )
        master = replace(
            master,
            transform_dag=MasterTransformDag(
                source_native=source_native,
                canonical=canonical_step,
            ),
        )

        canonical = load_canonical_master(master, boundary)

        assert np.array_equal(canonical.samples, expected)
        return

    if case == "altered-byte":
        damaged = bytearray(pcm_path.read_bytes())
        damaged[-1] ^= 0x01
        pcm_path.write_bytes(damaged)
        with pytest.raises(ValueError, match="byte SHA-256"):
            load_canonical_master(master, boundary)
        return

    if case == "truncated-pcm":
        truncated = b"\x00\x00\x00"
        pcm_path.write_bytes(truncated)
        master = manifest_for_bytes(truncated, sample_count=1)
        with pytest.raises(ValueError, match="divisible by four"):
            load_canonical_master(master, boundary)
        return

    if case == "count-mismatch":
        master = replace(master, sample_count=master.sample_count + 1)
        with pytest.raises(ValueError, match="sample count"):
            load_canonical_master(master, boundary)
        return

    if case == "dtype-mismatch":
        master = replace(master, dtype="float32-native")
        with pytest.raises(ValueError, match="dtype"):
            load_canonical_master(master, boundary)
        return

    if case == "rate-mismatch":
        master = replace(master, sample_rate_hz=48_000)
        with pytest.raises(ValueError, match="sample rate"):
            load_canonical_master(master, boundary)
        return

    if case == "non-finite":
        non_finite = np.asarray([0.0, np.nan, 0.5, 1.0], dtype="<f4").tobytes()
        pcm_path.write_bytes(non_finite)
        master = manifest_for_bytes(non_finite, sample_count=4)
        with pytest.raises(ValueError, match="non-finite"):
            load_canonical_master(master, boundary)
        return

    if case == "source-parent-byte-change":
        (boundary.root / "source.mkv").write_bytes(b"changed-source-media")
        with pytest.raises(ValueError, match="source parent"):
            load_canonical_master(master, boundary)
        return

    if case == "transform-parent-mismatch":
        canonical_step = replace(
            master.transform_dag.canonical,
            parent_byte_sha256=_digest(b"wrong-parent"),
            transform_sha256="0" * 64,
        )
        canonical_step = replace(
            canonical_step,
            transform_sha256=transform_step_sha256(canonical_step),
        )
        master = replace(
            master,
            transform_dag=replace(
                master.transform_dag,
                canonical=canonical_step,
            ),
        )
        with pytest.raises(ValueError, match="parent"):
            load_canonical_master(master, boundary)
        return

    if case == "transform-hash-mismatch":
        canonical_step = replace(
            master.transform_dag.canonical,
            transform_sha256="f" * 64,
        )
        master = replace(
            master,
            transform_dag=replace(
                master.transform_dag,
                canonical=canonical_step,
            ),
        )
        with pytest.raises(ValueError, match="transform SHA-256"):
            load_canonical_master(master, boundary)
        return

    if case == "transform-cycle":
        source_native = replace(
            master.transform_dag.source_native,
            output_artifact_id=(
                master.transform_dag.source_native.parent_artifact_id
            ),
            transform_sha256="0" * 64,
        )
        source_native = replace(
            source_native,
            transform_sha256=transform_step_sha256(source_native),
        )
        canonical_step = replace(
            master.transform_dag.canonical,
            parent_artifact_id=source_native.output_artifact_id,
            transform_sha256="0" * 64,
        )
        canonical_step = replace(
            canonical_step,
            transform_sha256=transform_step_sha256(canonical_step),
        )
        master = replace(
            master,
            parent_artifact_ids=(canonical_step.parent_artifact_id,),
            transform_dag=MasterTransformDag(
                source_native=source_native,
                canonical=canonical_step,
            ),
        )
        with pytest.raises(ValueError, match="acyclic"):
            load_canonical_master(master, boundary)
        return

    if case == "transform-semantic-mismatch":
        canonical_step = replace(
            master.transform_dag.canonical,
            resampler="banana",
            transform_sha256="0" * 64,
        )
        canonical_step = replace(
            canonical_step,
            transform_sha256=transform_step_sha256(canonical_step),
        )
        master = replace(
            master,
            transform_dag=replace(
                master.transform_dag,
                canonical=canonical_step,
            ),
        )
        with pytest.raises(ValueError, match="resampler"):
            load_canonical_master(master, boundary)
        return

    if case == "invalid-event-span":
        invalid_event = replace(
            references.events[0],
            span=SampleSpan(master.timeline_id, 0, master.sample_count + 1),
        )
        references = replace(
            references,
            events=(invalid_event,),
        )
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="span"):
            load_reference_set(references, canonical)
        return

    if case == "verbatim-link-mismatch":
        references = replace(
            references,
            events=(replace(references.events[0], verbatim_ref_id=None),),
        )
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="verbatim"):
            load_reference_set(references, canonical)
        return

    if case == "locked-missing-adjudication":
        references = replace(
            references,
            partition=Partition.LOCKED_CONFIRMATION,
            adjudication=None,
        )
        master = replace(
            master,
            partition=Partition.LOCKED_CONFIRMATION,
        )
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="adjudication"):
            load_reference_set(references, canonical)
        return

    if case == "dependent-annotation-pass":
        with pytest.raises(ValueError, match="cannot consume"):
            replace(
                references.annotation_passes[1],
                input_roles=(
                    AnnotationInputRole.AUDIO_ONLY,
                    AnnotationInputRole.INDEPENDENT_PASSES,
                ),
            )
        return

    if case in {
        "two-seeded-annotation-passes",
        "two-coverage-only-annotation-passes",
    }:
        role = (
            AnnotationInputRole.EXISTING_SUBTITLE
            if case == "two-seeded-annotation-passes"
            else AnnotationInputRole.SPEECH_EVENT_LAYER
        )
        references = replace(
            references,
            annotation_passes=tuple(
                replace(
                    annotation_pass,
                    input_roles=(AnnotationInputRole.AUDIO_ONLY, role),
                )
                for annotation_pass in references.annotation_passes
            ),
        )
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="asymmetric"):
            load_reference_set(references, canonical)
        return

    canonical = load_canonical_master(master, boundary)

    if case == "missing-adjudication":
        references = replace(references, adjudication=None)
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="adjudication"):
            load_reference_set(references, canonical)
        return

    if case == "duplicate-annotator":
        duplicate = replace(
            references.annotation_passes[1],
            annotator_id=references.annotation_passes[0].annotator_id,
        )
        references = replace(
            references,
            annotation_passes=(references.annotation_passes[0], duplicate),
        )
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="annotator"):
            load_reference_set(references, canonical)
        return

    if case == "wrong-adjudication-passes":
        references = replace(
            references,
            adjudication=replace(
                references.adjudication,
                input_pass_ids=(
                    references.annotation_passes[0].pass_id,
                    "90000000-0000-4000-8000-000000000028",
                ),
            ),
        )
        references, master = bind_reference_contract(references, master)
        canonical = load_canonical_master(master, boundary)
        with pytest.raises(ValueError, match="passes"):
            load_reference_set(references, canonical)
        return

    assert np.array_equal(canonical.samples, expected)
    assert canonical.samples.dtype.str == "<f4"
    assert not canonical.samples.flags.writeable
    with pytest.raises(ValueError):
        canonical.samples.setflags(write=True)

    reference_set = load_reference_set(references, canonical)
    assert reference_set.events == references.events
    assert reference_set.verbatim_references[0].text_id != ""
    assert reference_set.editorial_subtitles[0].text_id != ""
    assert not hasattr(reference_set.events[0], "text")


def _task3_run(
    tmp_path: Path,
    *,
    finish_all_arms: bool = True,
    report_loss: bool = False,
) -> dict[str, object]:
    from scripts.e0_evidence import contracts
    from scripts.e0_evidence.ledger import LedgerWriter
    from scripts.e0_evidence.schema import (
        ArmStatus,
        EvidenceRecordDraft,
        LossCode,
        RecordKind,
        RunStartManifest,
        Stage,
        TerminalOutcome,
        TextRole,
    )

    run_id = "a1000000-0000-4000-8000-000000000001"
    session_id = "a1000000-0000-4000-8000-000000000002"
    arm_a = "a1000000-0000-4000-8000-000000000003"
    arm_b = "a1000000-0000-4000-8000-000000000004"
    text_a = "a1000000-0000-4000-8000-000000000005"
    text_b = "a1000000-0000-4000-8000-000000000006"
    event_id = "a1000000-0000-4000-8000-000000000010"
    start = RunStartManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        session_id=session_id,
        clock_id="perf_counter_ns",
        canonical_stream_id="a1000000-0000-4000-8000-000000000007",
        private_text_stream_id="a1000000-0000-4000-8000-000000000008",
        arm_definitions=(
            ArmDefinition(
                arm_id=arm_a,
                kind=ArmKind.P0,
                parent_arm_id=None,
                parameters={"boundary_mode": "production"},
                declared_changes=(),
            ),
            ArmDefinition(
                arm_id=arm_b,
                kind=ArmKind.P1,
                parent_arm_id=arm_a,
                parameters={"boundary_mode": "oracle"},
                declared_changes=("boundary_mode",),
            ),
        ),
        classification=PrivateContentClass.PRIVATE_METADATA,
        details={
            "approved_contracts": contracts.approved_contracts(),
            "config_sha256": _digest(b"synthetic-config"),
            "event_fates_required": True,
            "reference_event_ids": (event_id,),
        },
    )
    issued_number = 0

    def deterministic_id() -> str:
        nonlocal issued_number
        issued_number += 1
        return f"a3000000-0000-4000-8000-{issued_number:012d}"

    clock_values = iter(range(100, 110))
    run_dir = tmp_path / "run"
    boundary = PrivateBoundary(
        root=tmp_path,
        worktrees=(),
        excluded_roots=(),
    )
    writer = LedgerWriter.create(
        run_dir,
        start,
        boundary=boundary,
        id_factory=deterministic_id,
        monotonic_ns=lambda: next(clock_values),
    )
    start_before_emit = (run_dir / "run-start.json").is_file()

    first = writer.emit(
        EvidenceRecordDraft(
            arm_id=arm_a,
            stage=Stage.DECODER,
            kind=RecordKind.RAW_HYPOTHESIS,
            artifact_id="a2000000-0000-4000-8000-000000000001",
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={"text_id": text_a},
        )
    )
    first_bytes = (run_dir / "ledger.jsonl").read_bytes()
    private_a = writer.emit_private_text(
        canonical_record_id=first.record_id,
        arm_id=arm_a,
        text_id=text_a,
        role=TextRole.RAW_HYPOTHESIS,
        text="秘密の台詞",
    )
    second = writer.emit(
        EvidenceRecordDraft(
            arm_id=arm_b,
            stage=Stage.WRAPPER,
            kind=RecordKind.WRAPPER_DECISION,
            artifact_id="a2000000-0000-4000-8000-000000000002",
            parent_ids=(first.record_id,),
            classification=PrivateContentClass.PRIVATE_AUDIO,
            details={"text_id": text_b},
        )
    )
    after_second_bytes = (run_dir / "ledger.jsonl").read_bytes()
    private_b = writer.emit_private_text(
        canonical_record_id=second.record_id,
        arm_id=arm_b,
        text_id=text_b,
        role=TextRole.WRAPPER_HYPOTHESIS,
        text="もう一つの秘密",
    )
    for arm_id in (arm_a, arm_b):
        writer.emit(
            EvidenceRecordDraft(
                arm_id=arm_id,
                stage=Stage.SCORING,
                kind=RecordKind.EVENT_FATE,
                artifact_id=event_id,
                decision=TerminalOutcome.UNKNOWN.value,
                classification=PrivateContentClass.PRIVATE_METADATA,
            )
        )
    arm_a_final = None
    arm_b_final = None
    if finish_all_arms:
        arm_b_final = writer.finish_arm(arm_b, ArmStatus.COMPLETE)
    arm_a_final = writer.finish_arm(arm_a, ArmStatus.COMPLETE)
    if report_loss:
        # A loss the writer counted itself; no production path takes a
        # caller-reported loss.
        writer._loss_counts[LossCode.RECORD_LOST] += 1
    final = writer.finalize()
    return {
        "after_second_bytes": after_second_bytes,
        "arm_a_final": arm_a_final,
        "arm_b_final": arm_b_final,
        "final": final,
        "first": first,
        "first_bytes": first_bytes,
        "private_ids": (private_a, private_b),
        "run_dir": run_dir,
        "second": second,
        "start": start,
        "start_before_emit": start_before_emit,
    }


def test_ledger_finalizes_complete_run_with_contiguous_unique_records_and_isolated_text(
    tmp_path: Path,
) -> None:
    from scripts.e0_evidence.ledger import (
        iter_canonical_records,
        require_complete_run,
        verify_run,
    )
    from scripts.e0_evidence.schema import (
        ArmStatus,
        RecordKind,
        RunStatus,
        StreamName,
    )

    artifacts = _task3_run(tmp_path)
    run_dir = artifacts["run_dir"]
    assert isinstance(run_dir, Path)
    ledger_bytes = (run_dir / "ledger.jsonl").read_bytes()
    private_bytes = (run_dir / "private-text.jsonl").read_bytes()
    records = tuple(iter_canonical_records(run_dir))

    assert artifacts["start_before_emit"] is True
    assert artifacts["after_second_bytes"].startswith(artifacts["first_bytes"])
    assert [record.seq for record in records] == [0, 1, 2, 3, 4, 5]
    assert [record.observed_monotonic_ns for record in records] == [
        100,
        101,
        102,
        103,
        104,
        105,
    ]
    assert len({record.record_id for record in records}) == len(records)
    terminal_records = [
        record for record in records if record.kind is RecordKind.ARM_STATUS
    ]
    assert {record.arm_id for record in terminal_records} == {
        "a1000000-0000-4000-8000-000000000003",
        "a1000000-0000-4000-8000-000000000004",
    }
    assert all(record.decision == ArmStatus.COMPLETE.value for record in terminal_records)
    assert ledger_bytes == b"".join(
        canonical_json_bytes(evidence_record_to_wire(record), newline=True)
        for record in records
    )
    assert ledger_bytes.endswith(b"\n")
    assert b"\n\n" not in ledger_bytes

    private_values = [
        json.loads(line)
        for line in private_bytes.splitlines()
    ]
    assert [item["stream_seq"] for item in private_values] == [0, 1]
    assert len({item["stream_id"] for item in private_values}) == 1
    assert [item["canonical_record_id"] for item in private_values] == [
        records[0].record_id,
        records[1].record_id,
    ]
    assert tuple(
        item["private_record_id"] for item in private_values
    ) == artifacts["private_ids"]
    assert "秘密の台詞".encode("utf-8") in private_bytes
    assert "もう一つの秘密".encode("utf-8") in private_bytes
    assert "秘密".encode("utf-8") not in ledger_bytes

    final = artifacts["final"]
    assert final.status is RunStatus.COMPLETE
    assert len(final.arm_statuses) == 2
    assert all(item.status is ArmStatus.COMPLETE for item in final.arm_statuses)
    assert all(count == 0 for count in final.loss_counts.values())
    summaries = {summary.name: summary for summary in final.streams}
    canonical = summaries[StreamName.CANONICAL]
    private = summaries[StreamName.PRIVATE_TEXT]
    assert (
        canonical.record_count,
        canonical.first_seq,
        canonical.last_seq,
        canonical.byte_sha256,
    ) == (6, 0, 5, _digest(ledger_bytes))
    assert (
        private.record_count,
        private.first_seq,
        private.last_seq,
        private.byte_sha256,
    ) == (2, 0, 1, _digest(private_bytes))

    report = verify_run(run_dir)
    assert report.complete
    assert report.reasons == ()
    require_complete_run(run_dir)


@pytest.mark.parametrize(
    ("damage", "expected_reason"),
    [
        ("middle-sequence-gap", "sequence_gap"),
        ("content-byte-changed", "stream_hash_mismatch"),
        ("partial-final-line", "invalid_or_truncated_jsonl"),
        ("missing-final-manifest", "final_manifest_missing"),
        ("nonzero-loss-counter", "loss_reported"),
        ("missing-requested-arm-status", "arm_terminal_status_missing"),
    ],
)
def test_ledger_verification_rejects_damaged_or_incomplete_run(
    tmp_path: Path,
    damage: str,
    expected_reason: str,
) -> None:
    from scripts.e0_evidence.ledger import (
        IncompleteRunError,
        require_complete_run,
        verify_run,
    )
    from scripts.e0_evidence.schema import IntegrityFailureCode, RunStatus

    artifacts = _task3_run(
        tmp_path,
        finish_all_arms=damage != "missing-requested-arm-status",
        report_loss=damage == "nonzero-loss-counter",
    )
    run_dir = artifacts["run_dir"]
    assert isinstance(run_dir, Path)
    ledger_path = run_dir / "ledger.jsonl"
    final_path = run_dir / "run-final.json"

    if damage == "middle-sequence-gap":
        lines = ledger_path.read_bytes().splitlines(keepends=True)
        ledger_path.write_bytes(lines[0] + b"".join(lines[2:]))
    elif damage == "content-byte-changed":
        raw = ledger_path.read_bytes()
        changed = raw.replace(
            b'"text_id":"a1000000-0000-4000-8000-000000000005"',
            b'"text_id":"a1000000-0000-4000-8000-000000000009"',
            1,
        )
        assert changed != raw and len(changed) == len(raw)
        ledger_path.write_bytes(changed)
    elif damage == "partial-final-line":
        raw = ledger_path.read_bytes()
        ledger_path.write_bytes(raw[:-10])
    elif damage == "missing-final-manifest":
        final_path.unlink()

    if damage in {"nonzero-loss-counter", "missing-requested-arm-status"}:
        assert artifacts["final"].status is not RunStatus.COMPLETE

    paths = (
        run_dir / "run-start.json",
        ledger_path,
        run_dir / "private-text.jsonl",
        final_path,
    )
    before = tuple(path.read_bytes() if path.exists() else None for path in paths)

    report = verify_run(run_dir)

    assert not report.complete
    assert IntegrityFailureCode(expected_reason) in report.reasons
    with pytest.raises(IncompleteRunError):
        require_complete_run(run_dir)
    after = tuple(path.read_bytes() if path.exists() else None for path in paths)
    assert after == before


def test_cli_verify_run_uses_the_real_ledger_completeness_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.e0_evidence.cli import main

    artifacts = _task3_run(tmp_path)
    run_dir = artifacts["run_dir"]
    start = artifacts["start"]
    assert isinstance(run_dir, Path)

    assert main(("verify-run", "--run-dir", str(run_dir))) == 0
    valid = capsys.readouterr()
    assert valid.err == ""
    assert valid.out == (
        f"run_id={start.run_id} status=complete reasons=0\n"
    )

    ledger_path = run_dir / "ledger.jsonl"
    lines = ledger_path.read_bytes().splitlines(keepends=True)
    ledger_path.write_bytes(lines[0] + b"".join(lines[2:]))

    assert main(("verify-run", "--run-dir", str(run_dir))) == 1
    damaged = capsys.readouterr()
    assert damaged.err == ""
    damaged_fields = damaged.out.strip().split()
    assert damaged_fields[:2] == [
        f"run_id={start.run_id}",
        "status=incomplete",
    ]
    assert damaged_fields[2].startswith("reasons=")
    assert int(damaged_fields[2].removeprefix("reasons=")) > 0
