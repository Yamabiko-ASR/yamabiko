from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e0_evidence.annotations import (
    assemble_reference_draft_from_pass,
    import_annotation_pass,
    load_registered_annotation_pass,
    parse_annotation_worksheet,
    verify_reference_annotation_passes,
)
from scripts.e0_evidence.cli import main
from scripts.e0_evidence.pcm import CanonicalMaster
from scripts.e0_evidence.private_root import PrivateBoundary
from scripts.e0_evidence.references import AnnotationInputRole, ReferenceSetManifest
from scripts.e0_evidence.registry import (
    freeze_reference_draft,
    load_registered_master,
    master_manifest_to_wire,
)
from scripts.e0_evidence.schema import Partition, SCHEMA_VERSION
from scripts.e0_evidence.wire import canonical_json_bytes


HEADER = (
    "start_seconds\tend_seconds\tevent_type\tjapanese\tintelligibility\t"
    "speaker\toverlap\tconfidence\tacoustic_tags\tnotes"
)
MASTER_ID = "10000000-0000-4000-8000-000000000001"
FIXTURE_ID = "10000000-0000-4000-8000-000000000002"
TIMELINE_ID = "10000000-0000-4000-8000-000000000003"
PASS_ID = "10000000-0000-4000-8000-000000000004"
ANNOTATOR_ID = "10000000-0000-4000-8000-000000000005"
PACKET_ID = "10000000-0000-4000-8000-000000000006"
PILOT_ID = "10000000-0000-4000-8000-000000000007"
SAMPLE_COUNT = 3_840_000


def _work_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "e0-human-annotation-work-1.0",
        "pass_id": PASS_ID,
        "annotator_id": ANNOTATOR_ID,
        "annotator_slot": "A",
        "packet_id": PACKET_ID,
        "pilot_id": PILOT_ID,
        "selection_label": "D1",
        "fixture_id": FIXTURE_ID,
        "master_id": MASTER_ID,
        "timeline_id": TIMELINE_ID,
        "sample_rate_hz": 16_000,
        "sample_count": SAMPLE_COUNT,
        "input_roles": ["audio_only"],
        "prohibited_inputs": [
            "candidate_output",
            "editorial_subtitles",
            "other_annotation_pass",
        ],
        "status": "awaiting_human",
        "completed_at_utc": None,
        "human_signature_receipt_id": None,
        "events": [],
        "verbatim_references": [],
    }
    payload.update(changes)
    return payload


def _private_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worksheet: bytes,
    *,
    work_changes: dict[str, object] | None = None,
) -> tuple[PrivateBoundary, Path, Path]:
    root = tmp_path / "private"
    root.mkdir()
    boundary = PrivateBoundary(root, (), ())
    worksheet_path = root / "returned.private.tsv"
    worksheet_path.write_bytes(worksheet)
    work_path = root / "pass-a.work.private.json"
    work_path.write_bytes(
        canonical_json_bytes(
            _work_payload(**(work_changes or {})),  # type: ignore[arg-type]
            newline=True,
        )
    )
    master = SimpleNamespace(
        master_id=MASTER_ID,
        fixture_id=FIXTURE_ID,
        timeline_id=TIMELINE_ID,
        sample_rate_hz=16_000,
        sample_count=SAMPLE_COUNT,
        partition=Partition.DEVELOPMENT,
    )
    monkeypatch.setattr(
        "scripts.e0_evidence.annotations.load_registered_master",
        lambda master_id, private_boundary: master,
    )
    return boundary, worksheet_path, work_path


def _valid_worksheet(*, bom: bool = False) -> bytes:
    text = "\n".join(
        (
            HEADER,
            (
                "0.842\t2.115\tspeech\t今日はgame nightだよ\tintelligible\t"
                "speaker_1\tnone\thigh\tcode_switch\t語尾を再確認"
            ),
            (
                '1.900\t2.470\tspeech\t"Thank you"\tintelligible\tspeaker_2\t'
                "overlap\tmedium\toverlap;non_japanese_speech\t"
            ),
            (
                "3.000\t4.000\tnon_speech\t\tnot_applicable\tnone\tnone\t"
                "high\tnon_speech_control\t音楽のみ"
            ),
        )
    ) + "\n"
    return (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")


def _editorial_sidecar(root: Path) -> Path:
    """Write the editorial sidecar a seeded proposal worksheet came from."""

    path = root / "pass-a.editorial.private.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e0-editorial-subtitle-proposal-2.0",
                "master_id": MASTER_ID,
                "subtitle_stream_index": 0,
                "subtitle_sha256": "a" * 64,
                "ffmpeg_sha256": "b" * 64,
                "source_sample_rate_hz": 48_000,
                "clip_start_samples": 0,
                "clip_end_samples": 240 * 48_000,
                "cues": [
                    {
                        "source_ordinal": 1,
                        "raw_text": "既存字幕",
                        "original_start_ms": 0,
                        "original_end_ms": 1_250,
                        "relative_start_ms": 0,
                        "relative_end_ms": 1_250,
                    }
                ],
            },
            newline=True,
        )
    )
    return path


def _assert_uuid4(value: object) -> None:
    assert isinstance(value, str)
    parsed = uuid.UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value


def test_valid_mixed_language_and_non_speech_pass_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _valid_worksheet(bom=True)
    parsed = parse_annotation_worksheet(
        raw,
        timeline_id=TIMELINE_ID,
        sample_count=SAMPLE_COUNT,
    )

    assert parsed.content_sha256 == hashlib.sha256(raw).hexdigest()
    assert parsed.rows[0].span.start == 842 * 16
    assert parsed.rows[0].span.end == 2_115 * 16
    assert parsed.rows[0].text == "今日はgame nightだよ"
    assert parsed.rows[1].acoustic_tags == ("overlap", "non_japanese_speech")
    assert parsed.rows[2].event_type == "non_speech"

    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        raw,
    )
    bundle = import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
    )

    bundle_dir = boundary.root / "annotation-passes" / PASS_ID
    assert bundle.directory == bundle_dir
    assert (bundle_dir / "worksheet.private.tsv").read_bytes() == raw
    manifest = json.loads((bundle_dir / "pass.private.json").read_text("utf-8"))
    private_text = json.loads(
        (bundle_dir / "pass-text.private.json").read_text("utf-8")
    )
    assert manifest["status"] == "compiled_unsigned"
    assert manifest["content_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["human_signature_receipt_id"] is None
    assert len(manifest["events"]) == 2
    assert len(manifest["non_speech_regions"]) == 1
    assert len(manifest["verbatim_references"]) == 2
    for event in manifest["events"]:
        _assert_uuid4(event["event_id"])
    encoded_manifest = canonical_json_bytes(manifest, newline=True)
    assert "今日はgame nightだよ".encode() not in encoded_manifest
    assert b'"Thank you"' not in encoded_manifest
    assert "語尾を再確認".encode() not in encoded_manifest
    assert {
        record["role"] for record in private_text["texts"].values()
    } == {"annotation_note", "verbatim_reference"}
    assert {record["text"] for record in private_text["texts"].values()} >= {
        "今日はgame nightだよ",
        '"Thank you"',
        "語尾を再確認",
        "音楽のみ",
    }
    registered = load_registered_annotation_pass(PASS_ID, boundary)
    assert registered.status == "compiled_unsigned"
    assert registered.annotation_pass.pass_id == PASS_ID
    assert registered.annotation_pass.annotator_id == ANNOTATOR_ID
    assert registered.annotation_pass.reference_version == "pass-a-1.0"
    assert registered.annotation_pass.content_sha256 == hashlib.sha256(raw).hexdigest()
    assert set(registered.texts.values()) >= {
        "今日はgame nightだよ",
        '"Thank you"',
        "語尾を再確認",
        "音楽のみ",
    }


def test_seeded_import_preserves_subtitle_input_policy_on_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded_worksheet = (
        HEADER
        + "\n0.000\t1.250\tspeech\t既存字幕\tintelligible\tspeaker_1\tnone\thigh\tquiet_dialogue\t\n"
    ).encode("utf-8")
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        seeded_worksheet,
        work_changes={
            "input_roles": ["audio_only", "existing_subtitle"],
            "prohibited_inputs": ["candidate_output", "other_annotation_pass"],
        },
    )

    sidecar = _editorial_sidecar(boundary.root)
    import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
        editorial_sidecar_path=sidecar,
    )

    registered = load_registered_annotation_pass(PASS_ID, boundary)
    assert registered.annotation_pass.input_roles == (
        AnnotationInputRole.AUDIO_ONLY,
        AnnotationInputRole.EXISTING_SUBTITLE,
    )
    assert registered.manifest["input_roles"] == ["audio_only", "existing_subtitle"]
    assert registered.manifest["prohibited_inputs"] == [
        "candidate_output",
        "other_annotation_pass",
    ]
    assert len(registered.manifest["verbatim_references"]) == 1
    assert registered.manifest["subtitle_proposal_sha256"] == (
        hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )


def test_coverage_only_import_preserves_policy_without_verbatim_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coverage_worksheet = (
        HEADER
        + "\n0.000\t1.250\tspeech\t\tintelligible\tspeaker_1\tnone\thigh\tquiet_dialogue\t\n"
    ).encode("utf-8")
    coverage_changes = {
        "input_roles": ["audio_only", "speech_event_layer"],
        "prohibited_inputs": [
            "candidate_output",
            "editorial_subtitles",
            "other_annotation_pass",
        ],
    }
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        coverage_worksheet,
        work_changes=coverage_changes,
    )

    import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
    )

    registered = load_registered_annotation_pass(PASS_ID, boundary)
    assert registered.annotation_pass.input_roles == (
        AnnotationInputRole.AUDIO_ONLY,
        AnnotationInputRole.SPEECH_EVENT_LAYER,
    )
    assert registered.manifest["input_roles"] == [
        "audio_only",
        "speech_event_layer",
    ]
    assert registered.manifest["prohibited_inputs"] == [
        "candidate_output",
        "editorial_subtitles",
        "other_annotation_pass",
    ]
    assert registered.manifest["verbatim_references"] == []

    nonempty_path = tmp_path / "nonempty-coverage"
    nonempty_path.mkdir()
    nonempty_boundary, nonempty_worksheet, nonempty_work = _private_contract(
        nonempty_path,
        monkeypatch,
        (
            HEADER
            + "\n0.000\t1.250\tspeech\t字幕\tintelligible\tspeaker_1\tnone\thigh\tquiet_dialogue\t\n"
        ).encode("utf-8"),
        work_changes=coverage_changes,
    )
    with pytest.raises(ValueError, match="coverage-only speech text must be empty"):
        import_annotation_pass(
            nonempty_worksheet,
            nonempty_work,
            "pass-a-1.0",
            MASTER_ID,
            nonempty_boundary,
        )

    for name, changes in (
        (
            "seeded-prohibitions",
            {
                "input_roles": ["audio_only", "existing_subtitle"],
                "prohibited_inputs": [
                    "candidate_output",
                    "editorial_subtitles",
                    "other_annotation_pass",
                ],
            },
        ),
        (
            "coverage-prohibitions",
            {
                "input_roles": ["audio_only", "speech_event_layer"],
                "prohibited_inputs": ["candidate_output", "other_annotation_pass"],
            },
        ),
    ):
        invalid_path = tmp_path / name
        invalid_path.mkdir()
        invalid_boundary, invalid_worksheet, invalid_work = _private_contract(
            invalid_path,
            monkeypatch,
            coverage_worksheet,
            work_changes=changes,
        )
        with pytest.raises(ValueError, match="annotation input policy is invalid"):
            import_annotation_pass(
                invalid_worksheet,
                invalid_work,
                "pass-a-1.0",
                MASTER_ID,
                invalid_boundary,
            )


def test_coverage_only_import_rejects_non_speech_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worksheet = (
        HEADER
        + "\n0.000\t1.250\tspeech\t\tintelligible\tspeaker_1\tnone\thigh\t\t"
        + "\n2.000\t3.000\tnon_speech\t\tnot_applicable\tnone\tnone\thigh\t"
        + "non_speech_control\t\n"
    ).encode("utf-8")
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        worksheet,
        work_changes={
            "selection_label": "D5",
            "input_roles": ["audio_only", "speech_event_layer"],
            "prohibited_inputs": [
                "candidate_output",
                "editorial_subtitles",
                "other_annotation_pass",
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="coverage-only passes must not contain non_speech rows",
    ):
        import_annotation_pass(
            worksheet_path,
            work_path,
            "pass-b-1.0",
            MASTER_ID,
            boundary,
        )
    assert not (boundary.root / "annotation-passes" / PASS_ID).exists()


@pytest.mark.parametrize(
    ("row", "match"),
    (
        ("bad\theader", "header"),
        (
            "0.84\t1.000\tspeech\tはい\tintelligible\tspeaker_1\tnone\thigh\t\t",
            "timestamp",
        ),
        (
            "1.000\t1.000\tspeech\tはい\tintelligible\tspeaker_1\tnone\thigh\t\t",
            "span",
        ),
        (
            "0.000\t240.001\tspeech\tはい\tintelligible\tspeaker_1\tnone\thigh\t\t",
            "timeline",
        ),
        (
            "0.000\t1.000\tvoice\tはい\tintelligible\tspeaker_1\tnone\thigh\t\t",
            "event_type",
        ),
        (
            "0.000\t1.000\tspeech\t\tintelligible\tspeaker_1\tnone\thigh\t\t",
            "text",
        ),
        (
            "0.000\t1.000\tspeech\t……\tintelligible\tspeaker_1\tnone\thigh\t\t",
            "intelligible speech text has no scorable characters",
        ),
        (
            "0.000\t1.000\tspeech\tはい\tintelligible\tspeaker_1\tnone\thigh\tunknown\t",
            "tag",
        ),
        (
            "0.000\t1.000\tspeech\tはい\tintelligible\tspeaker_1\toverlap\thigh\t\t",
            "overlap",
        ),
        (
            "0.000\t1.000\tnon_speech\tnoise\tnot_applicable\tnone\tnone\t"
            "high\tnon_speech_control\t",
            "text",
        ),
    ),
)
def test_invalid_worksheet_table(row: str, match: str) -> None:
    raw = (
        (row + "\n").encode("utf-8")
        if match == "header"
        else (HEADER + "\n" + row + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError, match=match):
        parse_annotation_worksheet(
            raw,
            timeline_id=TIMELINE_ID,
            sample_count=SAMPLE_COUNT,
        )


def test_exact_byte_hash_idempotence_and_conflicting_pass_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _valid_worksheet()
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        raw,
    )
    first = import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
    )
    bundle_files = {
        path.name: path.read_bytes() for path in first.directory.iterdir()
    }

    second = import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
    )
    assert second == first
    assert {
        path.name: path.read_bytes() for path in second.directory.iterdir()
    } == bundle_files

    text_path = second.directory / "pass-text.private.json"
    original_text = text_path.read_bytes()
    text_path.write_bytes(original_text.replace(b"Thank you", b"Thanks!!!"))
    with pytest.raises(ValueError, match="annotation pass"):
        load_registered_annotation_pass(PASS_ID, boundary)
    text_path.write_bytes(original_text)

    manifest_path = second.directory / "pass.private.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["events"][0]["verbatim_ref_id"] = None
    manifest_path.write_bytes(canonical_json_bytes(manifest, newline=True))
    with pytest.raises(ValueError, match="annotation pass"):
        load_registered_annotation_pass(PASS_ID, boundary)
    manifest_path.write_bytes(original_manifest)

    work = _work_payload(fixture_id="10000000-0000-4000-8000-000000000099")
    work_path.write_bytes(canonical_json_bytes(work, newline=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fixture_id"):
        import_annotation_pass(
            worksheet_path,
            work_path,
            "pass-a-1.0",
            MASTER_ID,
            boundary,
        )
    work_path.write_bytes(
        canonical_json_bytes(
            _work_payload(),  # type: ignore[arg-type]
            newline=True,
        )
    )

    worksheet_path.write_bytes(raw.replace(b"Thank you", b"Thanks!!!"))
    with pytest.raises(ValueError, match="conflicting"):
        import_annotation_pass(
            worksheet_path,
            work_path,
            "pass-a-1.0",
            MASTER_ID,
            boundary,
        )

    non_speech = (
        HEADER
        + "\n0.000\t240.000\tnon_speech\t\tnot_applicable\tnone\tnone\thigh\t"
        "non_speech_control\t全編に発話なし\n"
    ).encode("utf-8")
    parsed = parse_annotation_worksheet(
        non_speech,
        timeline_id=TIMELINE_ID,
        sample_count=SAMPLE_COUNT,
    )
    assert not any(row.event_type == "speech" for row in parsed.rows)


def test_cli_imports_annotation_pass_and_prints_only_id_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        _valid_worksheet(),
    )
    monkeypatch.setattr(
        "scripts.e0_evidence.cli._boundary",
        lambda args: boundary,
    )

    assert main(
        (
            "annotation-pass-import",
            "--worksheet",
            str(worksheet_path),
            "--work",
            str(work_path),
            "--reference-version",
            "pass-a-1.0",
            "--master-id",
            MASTER_ID,
        )
    ) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == f"pass_id={PASS_ID} status=compiled_unsigned\n"
    assert sorted(
        path.name
        for path in (boundary.root / "annotation-passes" / PASS_ID).iterdir()
    ) == ["pass-text.private.json", "pass.private.json", "worksheet.private.tsv"]


def test_cli_builds_single_pass_reference_draft_and_refuses_adjudication_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        _valid_worksheet(),
    )
    import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
    )
    registered = load_registered_annotation_pass(PASS_ID, boundary)
    monkeypatch.setattr("scripts.e0_evidence.cli._boundary", lambda args: boundary)
    validated: list[tuple[Path, Path, str, PrivateBoundary]] = []

    def validate(
        draft_path: Path,
        text_path: Path,
        master_id: str,
        private_boundary: PrivateBoundary,
    ) -> SimpleNamespace:
        validated.append((draft_path, text_path, master_id, private_boundary))
        return SimpleNamespace(
            fixture_id=FIXTURE_ID,
            reference_set_id="10000000-0000-4000-8000-000000000008",
        )

    monkeypatch.setattr(
        "scripts.e0_evidence.registry.validate_reference_draft",
        validate,
    )
    draft_path = boundary.root / "drafts" / "reference.private.json"
    text_path = boundary.root / "drafts" / "reference-text.private.json"
    command = (
        "reference-draft-from-pass",
        "--pass-id",
        PASS_ID,
        "--reference-set-id",
        "10000000-0000-4000-8000-000000000008",
        "--draft",
        str(draft_path),
        "--text",
        str(text_path),
    )

    assert main(command) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        "fixture_id=10000000-0000-4000-8000-000000000002 "
        "reference_set_id=10000000-0000-4000-8000-000000000008 "
        "status=draft\n"
    )
    draft = json.loads(draft_path.read_text("utf-8"))
    private_text = json.loads(text_path.read_text("utf-8"))
    assert draft == {
        "schema_version": SCHEMA_VERSION,
        "reference_set_id": "10000000-0000-4000-8000-000000000008",
        "fixture_id": FIXTURE_ID,
        "master_id": MASTER_ID,
        "partition": Partition.DEVELOPMENT.value,
        "reference_version": "pass-a-1.0",
        "reference_sha256": "0" * 64,
        "reference_text_sha256": "0" * 64,
        "reference_set_sha256": "0" * 64,
        "events": [
            {
                "event_id": event["event_id"],
                "master_id": MASTER_ID,
                "span": event["span"],
                "intelligibility": event["intelligibility"],
                "verbatim_ref_id": event["verbatim_ref_id"],
                "tags": event["acoustic_tags"],
                "annotation_version": event["annotation_version"],
            }
            for event in registered.manifest["events"]
        ],
        "verbatim_references": registered.manifest["verbatim_references"],
        "editorial_subtitles": [],
        "annotation_passes": [
            {
                "pass_id": PASS_ID,
                "annotator_id": ANNOTATOR_ID,
                "reference_version": "pass-a-1.0",
                "content_sha256": registered.annotation_pass.content_sha256,
                "input_roles": ["audio_only"],
            }
        ],
        "adjudication": None,
        "non_speech_controls": [
            {
                "region_id": region["region_id"],
                "span": region["span"],
                "tags": ["non_speech_control"],
            }
            for region in registered.manifest["non_speech_regions"]
        ],
    }
    text_ids = {
        item["text_id"] for item in registered.manifest["verbatim_references"]
    }
    assert private_text == {
        "schema_version": SCHEMA_VERSION,
        "reference_set_id": "10000000-0000-4000-8000-000000000008",
        "texts": {
            text_id: {
                "role": "verbatim_reference",
                "text": registered.texts[text_id],
            }
            for text_id in text_ids
        },
    }
    assert draft_path.read_bytes() == canonical_json_bytes(draft, newline=True)
    assert text_path.read_bytes() == canonical_json_bytes(private_text, newline=True)
    assert validated == [(draft_path, text_path, MASTER_ID, boundary)]

    blocker_master = SimpleNamespace(
        master_id=MASTER_ID,
        fixture_id=FIXTURE_ID,
        timeline_id=TIMELINE_ID,
        sample_rate_hz=16_000,
        sample_count=SAMPLE_COUNT,
        partition=Partition.BLOCKER_CLASS,
    )
    monkeypatch.setattr(
        "scripts.e0_evidence.annotations.load_registered_master",
        lambda master_id, private_boundary: blocker_master,
    )
    refused_draft = boundary.root / "drafts" / "blocker.private.json"
    refused_text = boundary.root / "drafts" / "blocker-text.private.json"
    refused = (*command[:-4], "--draft", str(refused_draft), "--text", str(refused_text))
    assert main(refused) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "status=invalid error=ValueError\n"
    assert not refused_draft.exists()
    assert not refused_text.exists()
    assert len(validated) == 1


def test_single_seeded_pass_cannot_build_or_freeze_locked_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.e0_evidence.test_replay import _master as replay_master

    seeded_worksheet = (
        HEADER
        + "\n0.000\t1.250\tspeech\t既存字幕\tintelligible\tspeaker_1\tnone\thigh\tquiet_dialogue\t\n"
    ).encode("utf-8")
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        seeded_worksheet,
        work_changes={
            "selection_label": "L1",
            "input_roles": ["audio_only", "existing_subtitle"],
            "prohibited_inputs": ["candidate_output", "other_annotation_pass"],
        },
    )
    synthetic = replay_master(4)
    locked_master = replace(
        synthetic.manifest,
        master_id=MASTER_ID,
        fixture_id=FIXTURE_ID,
        rights=replace(synthetic.manifest.rights, fixture_id=FIXTURE_ID),
        partition=Partition.LOCKED_CONFIRMATION,
        timeline_id=TIMELINE_ID,
        sample_count=SAMPLE_COUNT,
        duration_samples=SAMPLE_COUNT,
        annotation_set_sha256="0" * 64,
    )
    master_path = boundary.root / "masters" / MASTER_ID / "master.private.json"
    master_path.parent.mkdir(parents=True)
    master_path.write_bytes(
        canonical_json_bytes(master_manifest_to_wire(locked_master), newline=True)
    )
    (boundary.root / "locator.private.json").write_bytes(
        canonical_json_bytes(
            {"schema_version": SCHEMA_VERSION, "fixtures": {FIXTURE_ID: {}}},
            newline=True,
        )
    )
    monkeypatch.setattr(
        "scripts.e0_evidence.annotations.load_registered_master",
        load_registered_master,
    )
    monkeypatch.setattr(
        "scripts.e0_evidence.registry.load_canonical_master",
        lambda manifest, private_boundary: CanonicalMaster(
            manifest,
            synthetic.samples,
        ),
    )
    import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
        editorial_sidecar_path=_editorial_sidecar(boundary.root),
    )
    registered = load_registered_annotation_pass(PASS_ID, boundary)
    reference_set_id = "10000000-0000-4000-8000-000000000008"
    draft_path = boundary.root / "drafts" / "reference.private.json"
    text_path = boundary.root / "drafts" / "reference-text.private.json"

    with pytest.raises(ValueError, match="requires two adjudicated passes"):
        assemble_reference_draft_from_pass(
            PASS_ID,
            reference_set_id,
            draft_path,
            text_path,
            boundary,
        )
    assert not draft_path.exists()
    assert not text_path.exists()

    draft_path.parent.mkdir(exist_ok=True)
    draft_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "reference_set_id": reference_set_id,
                "fixture_id": FIXTURE_ID,
                "master_id": MASTER_ID,
                "partition": Partition.LOCKED_CONFIRMATION.value,
                "reference_version": "pass-a-1.0",
                "reference_sha256": "0" * 64,
                "reference_text_sha256": "0" * 64,
                "reference_set_sha256": "0" * 64,
                "events": [
                    {
                        "event_id": event["event_id"],
                        "master_id": MASTER_ID,
                        "span": event["span"],
                        "intelligibility": event["intelligibility"],
                        "verbatim_ref_id": event["verbatim_ref_id"],
                        "tags": event["acoustic_tags"],
                        "annotation_version": event["annotation_version"],
                    }
                    for event in registered.manifest["events"]
                ],
                "verbatim_references": registered.manifest["verbatim_references"],
                "editorial_subtitles": [],
                "annotation_passes": [
                    {
                        "pass_id": PASS_ID,
                        "annotator_id": ANNOTATOR_ID,
                        "reference_version": "pass-a-1.0",
                        "content_sha256": registered.annotation_pass.content_sha256,
                        "input_roles": ["audio_only", "existing_subtitle"],
                    }
                ],
                "adjudication": None,
                "non_speech_controls": [],
            },
            newline=True,
        )
    )
    text_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "reference_set_id": reference_set_id,
                "texts": {
                    item["text_id"]: {
                        "role": "verbatim_reference",
                        "text": registered.texts[item["text_id"]],
                    }
                    for item in registered.manifest["verbatim_references"]
                },
            },
            newline=True,
        )
    )

    with pytest.raises(ValueError, match="exactly two independent passes"):
        freeze_reference_draft(draft_path, text_path, MASTER_ID, boundary)
    assert not (boundary.root / "references").exists()
    assert load_registered_master(MASTER_ID, boundary).annotation_set_sha256 == (
        "0" * 64
    )


def test_reference_pass_binding_rejects_declared_mismatch_and_worksheet_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary, worksheet_path, work_path = _private_contract(
        tmp_path,
        monkeypatch,
        _valid_worksheet(),
    )
    import_annotation_pass(
        worksheet_path,
        work_path,
        "pass-a-1.0",
        MASTER_ID,
        boundary,
    )
    registered = load_registered_annotation_pass(PASS_ID, boundary)
    reference = ReferenceSetManifest(
        schema_version=SCHEMA_VERSION,
        reference_set_id="10000000-0000-4000-8000-000000000008",
        fixture_id=FIXTURE_ID,
        master_id=MASTER_ID,
        partition=Partition.DEVELOPMENT,
        reference_version="reference-1.0",
        reference_sha256="0" * 64,
        reference_set_sha256="0" * 64,
        events=(),
        verbatim_references=(),
        editorial_subtitles=(),
        annotation_passes=(registered.annotation_pass,),
        adjudication=None,
        non_speech_controls=(),
    )
    verify_reference_annotation_passes(reference, boundary)

    mismatch = replace(
        reference,
        annotation_passes=(
            replace(registered.annotation_pass, content_sha256="f" * 64),
        ),
    )
    with pytest.raises(ValueError, match="declared annotation pass"):
        verify_reference_annotation_passes(mismatch, boundary)

    registered.bundle.directory.joinpath("worksheet.private.tsv").write_bytes(
        _valid_worksheet().replace(b"Thank you", b"Thanks!!!")
    )
    with pytest.raises(ValueError, match="registered annotation pass"):
        verify_reference_annotation_passes(reference, boundary)
