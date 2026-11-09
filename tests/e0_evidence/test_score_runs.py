"""The committed ``score`` path from verified ledgers to a private score bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.e0_evidence import contracts
from scripts.e0_evidence.cli import main
from scripts.e0_evidence.decoder import run_p0
from scripts.e0_evidence.ledger import LedgerWriter
from scripts.e0_evidence.private_root import PRIVATE_ROOT_ENV
from scripts.e0_evidence.references import (
    NonSpeechControl,
    ReferenceEvent,
    VerbatimReference,
    adjudicated_reference_content_sha256,
    reference_content_sha256,
    reference_set_manifest_sha256,
)
from scripts.e0_evidence.registry import (
    master_manifest_to_wire,
    reference_manifest_to_wire,
)
from scripts.e0_evidence.report import (
    PUBLIC_EXPORT_PROFILE_SHA256,
    PUBLIC_EXPORT_PROFILE_VERSION,
    PrivateReportError,
    PublicExportPrivacyError,
    PublicExportReview,
    export_public,
    private_report_sha256,
    public_export_review_to_wire,
    write_score_bundle,
)
from scripts.e0_evidence.runtime import (
    DEFAULT_HARD_STRATUM_TAGS,
    HARD_STRATUM_VERSION,
)
from scripts.e0_evidence.schema import (
    MONOTONIC_CLOCK_ID,
    SCHEMA_VERSION,
    ArmDefinition,
    ArmKind,
    ArmStatus,
    CapabilityDecision,
    EvidenceRecordDraft,
    ObservationStatus,
    PrivateContentClass,
    RecordKind,
    RunStartManifest,
    SampleSpan,
    Stage,
    TerminalOutcome,
    new_opaque_id,
)
from scripts.e0_evidence.score_runs import score_runs
from scripts.e0_evidence.wire import canonical_json_bytes, canonical_sha256
from tests.e0_evidence.test_decoder import (
    _MODEL_ARTIFACT_SHA256,
    _build_asr,
    _Delegate,
    _Segment,
    _window,
)
from tests.e0_evidence.test_integrity import _task2b_contract


_ARM_ID = "95000000-0000-4000-8000-000000000001"
_RUN_ID = "95000000-0000-4000-8000-000000000002"
_COHORT_ID = "95000000-0000-4000-8000-000000000003"
_TITLE_ID = "95000000-0000-4000-8000-000000000004"
_REJECTED_EVENT_ID = "95000000-0000-4000-8000-000000000005"
_REJECTED_VERBATIM_ID = "95000000-0000-4000-8000-000000000006"
_REJECTED_TEXT_ID = "95000000-0000-4000-8000-000000000007"
_CONTROL_REGION_ID = "95000000-0000-4000-8000-000000000008"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _WriterArmSink:
    def __init__(self, writer: LedgerWriter, arm_id: str) -> None:
        self._writer = writer
        self.arm_id = arm_id

    def emit(self, draft: EvidenceRecordDraft):  # noqa: ANN201 - EvidenceSink
        return self._writer.emit(draft)

    def emit_private_text(self, **kwargs: object) -> str:
        return self._writer.emit_private_text(**kwargs)  # type: ignore[arg-type]


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload, newline=True))  # type: ignore[arg-type]


def _verified_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_event_is_control: bool = False,
    accepted_text: str = "猫",
    second_line_gated: bool = True,
) -> SimpleNamespace:
    """Register a two-event fixture and write one complete P0 run for it.

    The references are "猫" and "犬". The decoder yields ``accepted_text`` ("猫"
    by default) over the first event and "犬" over the second; the production
    wrapper's confidence gate rejects "犬", so only the first event has
    wrapper-available text. With ``second_line_gated=False`` the gate keeps "犬"
    as well. With ``first_event_is_control`` the first event's span is a frozen
    non-speech control instead, so the accepted text lies in a control and
    overlaps no reference event.
    """

    boundary, _, _, master, references = _task2b_contract(tmp_path)
    monkeypatch.setenv(PRIVATE_ROOT_ENV, str(boundary.root))
    root = boundary.root
    timeline_id = master.timeline_id
    accepted_event = references.events[0]
    assert accepted_event.span == SampleSpan(timeline_id, 0, 2)
    rejected_event = ReferenceEvent(
        event_id=_REJECTED_EVENT_ID,
        master_id=master.master_id,
        span=SampleSpan(timeline_id, 2, 4),
        intelligibility="intelligible",
        verbatim_ref_id=_REJECTED_VERBATIM_ID,
        tags=(),
        annotation_version=accepted_event.annotation_version,
    )
    accepted_verbatim = references.verbatim_references[0]
    rejected_verbatim = VerbatimReference(
        verbatim_ref_id=_REJECTED_VERBATIM_ID,
        event_ids=(_REJECTED_EVENT_ID,),
        text_id=_REJECTED_TEXT_ID,
        annotation_version=accepted_verbatim.annotation_version,
    )
    text_entries = {
        accepted_verbatim.text_id: {"role": "verbatim_reference", "text": "猫"},
        _REJECTED_TEXT_ID: {"role": "verbatim_reference", "text": "犬"},
    }
    if first_event_is_control:
        del text_entries[accepted_verbatim.text_id]
        events: tuple[ReferenceEvent, ...] = (rejected_event,)
        verbatim = (rejected_verbatim,)
        controls = (
            NonSpeechControl(
                region_id=_CONTROL_REGION_ID,
                span=accepted_event.span,
                tags=("non_speech_control",),
            ),
        )
    else:
        events = (accepted_event, rejected_event)
        verbatim = (accepted_verbatim, rejected_verbatim)
        controls = ()
    texts = {
        "schema_version": SCHEMA_VERSION,
        "reference_set_id": references.reference_set_id,
        "texts": text_entries,
    }
    frozen = replace(
        references,
        events=events,
        verbatim_references=verbatim,
        editorial_subtitles=(),
        non_speech_controls=controls,
        reference_text_sha256=canonical_sha256(texts),  # type: ignore[arg-type]
    )
    frozen = replace(frozen, reference_sha256=reference_content_sha256(frozen))
    assert frozen.adjudication is not None
    frozen = replace(
        frozen,
        adjudication=replace(
            frozen.adjudication,
            output_content_sha256=adjudicated_reference_content_sha256(frozen),
        ),
    )
    frozen = replace(frozen, reference_set_sha256=reference_set_manifest_sha256(frozen))
    master = replace(master, annotation_set_sha256=frozen.reference_set_sha256)
    reference_dir = root / "references" / frozen.reference_set_id
    _write_private_json(
        root / "masters" / master.master_id / "master.private.json",
        master_manifest_to_wire(master),
    )
    _write_private_json(
        reference_dir / "reference.private.json",
        reference_manifest_to_wire(frozen),
    )
    _write_private_json(reference_dir / "text.private.json", texts)
    _write_private_json(
        root / "fixtures" / master.fixture_id / "fixture.private.json",
        {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": master.fixture_id,
            "partition": master.partition.value,
            "title_id": _TITLE_ID,
            "rights_record_id": master.rights.rights_record_id,
        },
    )

    model_runtime_sha256 = _digest("synthetic-model-runtime")
    cohort_manifest_sha256 = _digest("synthetic-cohort")
    arm_profile_sha256 = _digest("synthetic-p0-profile")
    start = RunStartManifest(
        schema_version=SCHEMA_VERSION,
        run_id=_RUN_ID,
        session_id=new_opaque_id(),
        clock_id=MONOTONIC_CLOCK_ID,
        canonical_stream_id=new_opaque_id(),
        private_text_stream_id=new_opaque_id(),
        arm_definitions=(
            ArmDefinition(
                arm_id=_ARM_ID,
                kind=ArmKind.P0,
                parent_arm_id=None,
                parameters={
                    "arm_profile_sha256": arm_profile_sha256,
                    "cohort_id": _COHORT_ID,
                    "cohort_manifest_sha256": cohort_manifest_sha256,
                    "decoder_random_seed": 105,
                    "model_artifact_sha256": _MODEL_ARTIFACT_SHA256,
                    "model_runtime_sha256": model_runtime_sha256,
                },
                declared_changes=(),
            ),
        ),
        classification=PrivateContentClass.PRIVATE_METADATA,
        details={
            "approved_contracts": contracts.approved_contracts(),
            "arm_profile_sha256": {_ARM_ID: arm_profile_sha256},
            "cohort_fixture_ids": [master.fixture_id],
            "cohort_id": _COHORT_ID,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "event_fates_required": True,
            "hard_stratum": {
                "version": HARD_STRATUM_VERSION,
                "tags": list(DEFAULT_HARD_STRATUM_TAGS),
            },
            "fixture_id": master.fixture_id,
            "master_id": master.master_id,
            "master_pcm_sha256": master.canonical_pcm_sha256,
            "model_runtime_sha256": model_runtime_sha256,
            "partition": master.partition.value,
            "reference_event_ids": [event.event_id for event in events],
            "reference_set_id": frozen.reference_set_id,
            "reference_set_sha256": frozen.reference_set_sha256,
        },
    )
    writer = LedgerWriter.create(root / "runs" / _RUN_ID, start, boundary=boundary)
    delegate = _Delegate(
        lambda _index, _kwargs: iter(
            (
                _Segment(start=0.0, end=0.000125, text=accepted_text),
                _Segment(
                    start=0.000125,
                    end=0.00025,
                    text="犬",
                    no_speech_prob=0.9 if second_line_gated else 0.1,
                    avg_logprob=-1.5 if second_line_gated else -0.2,
                ),
            )
        )
    )
    result = run_p0(
        _window(
            np.linspace(-0.5, 0.5, 4, dtype="<f4"),
            (SampleSpan(timeline_id, 0, 4),),
        ),
        _build_asr(monkeypatch, delegate, boundary_pad_ms=0),
        _WriterArmSink(writer, _ARM_ID),
    )
    assert result.asr_result.text == accepted_text + ("" if second_line_gated else "犬")
    for event in events:
        outcome = (
            TerminalOutcome.WRAPPER_REJECTED
            if second_line_gated and event is rejected_event
            else TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE
        )
        writer.emit(
            EvidenceRecordDraft(
                arm_id=_ARM_ID,
                stage=Stage.SCORING,
                kind=RecordKind.EVENT_FATE,
                artifact_id=event.event_id,
                source_spans=(event.span,),
                decision=outcome.value,
                classification=PrivateContentClass.PRIVATE_AUDIO,
            )
        )
    for name, unit, value in (
        ("rtf", "ratio", 0.5),
        ("cold_start_seconds", "seconds", 1.25),
        ("peak_rss_bytes", "bytes", 4_096),
        ("peak_vram_bytes", "bytes", None),
    ):
        details: dict[str, object] = {"measurement_name": name, "unit": unit}
        if name == "peak_vram_bytes":
            details["vram_scope"] = "device_delta"
        if value is not None:
            details["value"] = value
        writer.emit(
            EvidenceRecordDraft(
                arm_id=_ARM_ID,
                stage=Stage.RUN,
                kind=RecordKind.MEASUREMENT,
                artifact_id=new_opaque_id(),
                observation_status=(
                    ObservationStatus.OBSERVED
                    if value is not None
                    else ObservationStatus.INSTRUMENTATION_MISSING
                ),
                details=details,
            )
        )
    writer.finish_arm(_ARM_ID, ArmStatus.COMPLETE)
    writer.finalize()
    return SimpleNamespace(boundary=boundary, run_id=_RUN_ID, arm_id=_ARM_ID)


def test_score_command_bundles_wrapper_recall_and_cer_without_crediting_rejected_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _verified_run(tmp_path, monkeypatch)

    assert main(("score", "--run-id", run.run_id, "--bootstrap-replicates", "0")) == 0

    output = capsys.readouterr()
    assert output.err == ""
    fields = dict(item.split("=", 1) for item in output.out.split())
    assert fields.keys() == {"report_id", "status"}
    assert fields["status"] == "complete"
    report_dir = run.boundary.root / "reports" / fields["report_id"]
    bundle = json.loads((report_dir / "score-bundle.private.json").read_text("utf-8"))
    (score,) = bundle["score_reports"]
    assert score["arm_id"] == run.arm_id
    recall = score["micro"]["wrapper_available_recall"]
    assert (recall["numerator"], recall["denominator"], recall["unknown"]) == (1, 2, 0)
    assert recall["value"] == 0.5
    # Crediting the rejected "犬" would align it to the second event (CER 0).
    cer = score["micro"]["cer_minimal"]
    assert (
        cer["substitutions"],
        cer["deletions"],
        cer["insertions"],
        cer["reference_characters"],
    ) == (0, 1, 0, 2)
    assert cer["value"] == 0.5
    assert score["micro"]["raw_nonblank_recall"]["numerator"] == 2

    assert main(("report", "--report-dir", str(report_dir))) == 0
    assert capsys.readouterr().out == "status=complete views=0\n"


def test_score_bundle_rejects_wrapper_recall_that_disagrees_with_ledger_fates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _verified_run(tmp_path, monkeypatch)
    run_dirs = (run.boundary.root / "runs" / run.run_id,)
    (honest,) = score_runs(run_dirs, run.boundary, bootstrap_replicates=0).reports
    recall = honest.micro.wrapper_available_recall
    # Credit the wrapper-rejected event as available as well.
    inflated = replace(
        honest,
        micro=replace(
            honest.micro,
            wrapper_available_recall=replace(
                recall,
                numerator=2,
                value=1.0,
                lower_bound=1.0,
                upper_bound=1.0,
            ),
        ),
    )
    sources = {run.arm_id: (run.run_id,)}
    rejected_dir = run.boundary.root / "reports" / new_opaque_id()

    with pytest.raises(PrivateReportError):
        write_score_bundle(rejected_dir, run_dirs, (inflated,), sources, run.boundary)

    assert not rejected_dir.exists()
    accepted_dir = run.boundary.root / "reports" / new_opaque_id()
    bundle = write_score_bundle(accepted_dir, run_dirs, (honest,), sources, run.boundary)
    assert bundle.parent == accepted_dir


def test_export_public_refuses_a_report_edited_after_rendering_from_its_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _verified_run(tmp_path, monkeypatch)
    assert main(("score", "--run-id", run.run_id, "--bootstrap-replicates", "0")) == 0
    fields = dict(item.split("=", 1) for item in capsys.readouterr().out.split())
    report_dir = run.boundary.root / "reports" / fields["report_id"]
    assert main(("report", "--report-dir", str(report_dir))) == 0
    report_json = report_dir / "report.json"
    payload = json.loads(report_json.read_text("utf-8"))
    # Inflate the rendered recall; the bundle and its ledgers still say 0.5.
    payload["score_reports"][0]["micro"]["wrapper_available_recall"]["value"] = 1.0
    report_json.write_bytes(canonical_json_bytes(payload, newline=True))
    review = PublicExportReview(
        review_id=new_opaque_id(),
        reviewer_id=new_opaque_id(),
        evidence_ids=(new_opaque_id(),),
        decision=CapabilityDecision.APPROVED,
        source_report_sha256=private_report_sha256(payload),
        profile_version=PUBLIC_EXPORT_PROFILE_VERSION,
        profile_sha256=PUBLIC_EXPORT_PROFILE_SHA256,
        approved_fields=("score.micro.wrapper_available_recall.value",),
    )
    review_path = (
        run.boundary.root / "export-reviews" / f"{review.review_id}.private.json"
    )
    _write_private_json(review_path, public_export_review_to_wire(review))
    destination = tmp_path / "public-export"

    with pytest.raises(PublicExportPrivacyError, match="does not match its score bundle"):
        export_public(report_dir, review_path, destination, run.boundary)

    assert not destination.exists()


def _score_report(run: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> dict:
    assert main(("score", "--run-id", run.run_id, "--bootstrap-replicates", "0")) == 0
    fields = dict(item.split("=", 1) for item in capsys.readouterr().out.split())
    bundle_path = (
        run.boundary.root / "reports" / fields["report_id"] / "score-bundle.private.json"
    )
    (score,) = json.loads(bundle_path.read_text("utf-8"))["score_reports"]
    return score


def test_accepted_output_inside_a_non_speech_control_counts_as_a_hallucination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _verified_run(tmp_path, monkeypatch, first_event_is_control=True)

    score = _score_report(run, capsys)

    hallucinations = score["micro"]["hallucinations"]
    assert hallucinations["count"] == 1
    assert hallucinations["control_samples"] == 2
    assert hallucinations["status"] == "defined"


def test_event_tagged_with_a_declared_hard_tag_is_scored_in_the_hard_stratum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _verified_run(tmp_path, monkeypatch)
    # Of the two events only the first carries a declared tag (quiet_dialogue).

    score = _score_report(run, capsys)

    hard_recall = score["hard_stratum"]["wrapper_available_recall"]
    assert (hard_recall["numerator"], hard_recall["denominator"]) == (1, 1)
    assert score["hard_stratum_version"] == "e0-hard-stratum-1.0"


def test_words_cer_does_not_charge_a_laugh_the_output_appends_to_the_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Both lines are kept and the first one's output ends in a laugh.
    run = _verified_run(
        tmp_path,
        monkeypatch,
        accepted_text="猫ははは",
        second_line_gated=False,
    )

    score = _score_report(run, capsys)

    minimal = score["micro"]["cer_minimal"]
    words = score["micro"]["cer_words"]
    assert (
        minimal["substitutions"],
        minimal["deletions"],
        minimal["insertions"],
        minimal["reference_characters"],
    ) == (0, 0, 3, 2)
    assert (
        words["substitutions"],
        words["deletions"],
        words["insertions"],
        words["reference_characters"],
    ) == (0, 0, 0, 2)
    assert words["value"] == 0.0
    assert score["vocal_sounds"] == {
        "reference_sound_characters": 0,
        "output_sound_characters": len("ははは"),
        "reference_chains": 0,
        "output_chains": 0,
    }
