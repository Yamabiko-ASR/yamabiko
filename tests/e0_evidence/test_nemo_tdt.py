"""The onnx-asr NeMo TDT candidate through the production ledger and score."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.e0_evidence.cli import main
from scripts.e0_evidence.ledger import LedgerWriter, verify_run
from scripts.e0_evidence.nemo_tdt import (
    NEMO_TDT_ADAPTER,
    decoder_identity,
    load_nemo_tdt,
    run_nemo_p0,
)
from scripts.e0_evidence.registry import load_registered_master
from scripts.e0_evidence.replay import ReplayWindow
from scripts.e0_evidence.runtime import _artifact_manifest
from scripts.e0_evidence.schema import (
    MONOTONIC_CLOCK_ID,
    SCHEMA_VERSION,
    ArmDefinition,
    ArmKind,
    ArmStatus,
    DiagnosticCode,
    EndpointEmitReason,
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
from scripts.e0_evidence.wire import canonical_sha256
from tests.e0_evidence.test_score_runs import _WriterArmSink, _verified_run


_ARM_ID = "98000000-0000-4000-8000-000000000001"
_RUN_ID = "98000000-0000-4000-8000-000000000002"
_COHORT_ID = "98000000-0000-4000-8000-000000000003"


class _FakeParakeet:
    """Stands in for onnx-asr's timestamped NeMo TDT adapter."""

    def __init__(self, results: list[SimpleNamespace]) -> None:
        self._results = results
        self.calls: list[np.ndarray] = []

    def with_timestamps(self) -> _FakeParakeet:
        return self

    def recognize(self, waveform: np.ndarray, *, sample_rate: int) -> SimpleNamespace:
        assert sample_rate == 16_000
        self.calls.append(np.array(waveform, copy=True))
        return self._results.pop(0)


def _window(window_id: str, audio: np.ndarray, span: SampleSpan) -> ReplayWindow:
    immutable = np.ascontiguousarray(audio, dtype="<f4").copy()
    immutable.setflags(write=False)
    return ReplayWindow(
        window_id=window_id,
        source_spans=(span,),
        audio=immutable,
        pcm_sha256=hashlib.sha256(immutable.tobytes(order="C")).hexdigest(),
        emit_reason=EndpointEmitReason.SILENCE_CLOSE,
        forced=False,
        dip_cut=False,
    )


def test_parakeet_p0_ledger_verifies_and_score_credits_only_the_window_with_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Registers a two-event fixture: "猫" over samples [0, 2), "犬" over [2, 4).
    registered = _verified_run(tmp_path, monkeypatch)
    boundary = registered.boundary
    fixture_start = json.loads(
        (boundary.root / "runs" / registered.run_id / "run-start.json").read_bytes()
    )["details"]
    timeline_id = load_registered_master(fixture_start["master_id"], boundary).timeline_id

    model_dir = tmp_path / "parakeet-tdt_ctc-0.6b-ja-onnx"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemo-conformer-tdt",
                "features_size": 80,
                "subsampling_factor": 8,
                "max_tokens_per_step": 10,
            }
        ),
        "utf-8",
    )
    for name in (
        "decoder_joint-model.onnx",
        "decoder_joint-model.onnx_data",
        "encoder-model.onnx",
        "encoder-model.onnx_data",
        "vocab.txt",
    ):
        (model_dir / name).write_bytes(name.encode("ascii"))
    manifest = _artifact_manifest(model_dir)
    model_artifact_sha256 = canonical_sha256(manifest)

    fake = _FakeParakeet(
        [
            SimpleNamespace(text="猫", timestamps=[0.0], tokens=["\u2581猫"]),
            SimpleNamespace(text="", timestamps=[], tokens=[]),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx_asr",
        SimpleNamespace(
            __version__="0.12.0",
            load_model=lambda model_type, path, **_kwargs: fake,
        ),
    )

    model_runtime_sha256 = hashlib.sha256(b"parakeet-runtime").hexdigest()
    cohort_manifest_sha256 = hashlib.sha256(b"parakeet-cohort").hexdigest()
    arm_profile_sha256 = hashlib.sha256(b"parakeet-p0-profile").hexdigest()
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
                    "decoder_adapter": NEMO_TDT_ADAPTER,
                    "model_artifact_sha256": model_artifact_sha256,
                    "model_runtime_sha256": model_runtime_sha256,
                },
                declared_changes=(),
            ),
        ),
        classification=PrivateContentClass.PRIVATE_METADATA,
        details={
            **{
                key: fixture_start[key]
                for key in (
                    "approved_contracts",
                    "event_fates_required",
                    "fixture_id",
                    "hard_stratum",
                    "master_id",
                    "master_pcm_sha256",
                    "partition",
                    "reference_event_ids",
                    "reference_set_id",
                    "reference_set_sha256",
                )
            },
            "arm_profile_sha256": {_ARM_ID: arm_profile_sha256},
            "cohort_fixture_ids": [fixture_start["fixture_id"]],
            "cohort_id": _COHORT_ID,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "decoder_adapter": decoder_identity(
                manifest,
                onnx_asr_version="0.12.0",
                onnxruntime_version="1.23.2",
            ),
            "model_runtime_sha256": model_runtime_sha256,
        },
    )
    writer = LedgerWriter.create(boundary.root / "runs" / _RUN_ID, start, boundary=boundary)
    sink = _WriterArmSink(writer, _ARM_ID)
    decoder = load_nemo_tdt(model_dir, model_artifact_sha256, boundary_pad_ms=0)

    spoken = run_nemo_p0(
        _window(
            "98000000-0000-4000-8000-000000000011",
            np.array([0.25, -0.5], dtype="<f4"),
            SampleSpan(timeline_id, 0, 2),
        ),
        decoder,
        sink,
    )
    silent = run_nemo_p0(
        _window(
            "98000000-0000-4000-8000-000000000012",
            np.array([0.0, 0.0], dtype="<f4"),
            SampleSpan(timeline_id, 2, 4),
        ),
        decoder,
        sink,
    )

    assert len(fake.calls) == 2
    assert spoken.wrapper_record.decision == (
        TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE.value
    )
    assert silent.wrapper_record.decision == DiagnosticCode.WRAPPER_EMPTY.value
    assert silent.call.call_record.decision == DiagnosticCode.DECODER_NO_YIELD.value
    event_ids = fixture_start["reference_event_ids"]
    for event_id, span, outcome in (
        (event_ids[0], SampleSpan(timeline_id, 0, 2), TerminalOutcome.WRAPPER_HYPOTHESIS_AVAILABLE),
        (event_ids[1], SampleSpan(timeline_id, 2, 4), TerminalOutcome.DECODER_NO_YIELD),
    ):
        writer.emit(
            EvidenceRecordDraft(
                arm_id=_ARM_ID,
                stage=Stage.SCORING,
                kind=RecordKind.EVENT_FATE,
                artifact_id=event_id,
                source_spans=(span,),
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

    report = verify_run(boundary.root / "runs" / _RUN_ID)
    assert report.complete, report.reasons

    assert main(("score", "--run-id", _RUN_ID, "--bootstrap-replicates", "0")) == 0
    fields = dict(item.split("=", 1) for item in capsys.readouterr().out.split())
    bundle = json.loads(
        (
            boundary.root / "reports" / fields["report_id"] / "score-bundle.private.json"
        ).read_text("utf-8")
    )
    (score,) = bundle["score_reports"]
    recall = score["micro"]["wrapper_available_recall"]
    assert (recall["numerator"], recall["denominator"]) == (1, 2)
    cer = score["micro"]["cer_minimal"]
    assert (
        cer["substitutions"],
        cer["deletions"],
        cer["insertions"],
        cer["reference_characters"],
    ) == (0, 1, 0, 2)
