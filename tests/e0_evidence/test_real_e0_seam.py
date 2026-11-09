"""One real E0 seam: immutable master -> Silero/FrameGate -> Whisper evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.e0_evidence import contracts
from scripts.e0_evidence.assets import require_local_evaluation
from scripts.e0_evidence.decoder import bind_model_artifact, run_p0, run_p2a
from scripts.e0_evidence.ledger import LedgerWriter, stochastic_tail_start, verify_run
from scripts.e0_evidence.pcm import (
    CanonicalMaster,
    load_canonical_master,
    load_capture_replay_stream,
)
from scripts.e0_evidence.private_root import require_private_path
from scripts.e0_evidence.registry import (
    _load_rights,
    boundary_from_environment,
    load_registered_master,
    master_manifest_from_wire,
)
from scripts.e0_evidence.replay import (
    load_production_model,
    load_production_tagger,
    production_chunker_config,
    replay_master,
    resolve_vad_artifact,
)
from scripts.e0_evidence.runtime import (
    _ArmSink,
    _config_for_resolved_model,
    _hash_artifact,
    _invariant_arm_parameters,
    _resolve_configured_model_artifact,
)
from scripts.e0_evidence.schema import (
    MONOTONIC_CLOCK_ID,
    SCHEMA_VERSION,
    ArmDefinition,
    ArmKind,
    ArmStatus,
    EvidenceRecordDraft,
    PrivateContentClass,
    RecordKind,
    RunStartManifest,
    RunStatus,
    SampleSpan,
    Stage,
    TerminalOutcome,
    new_opaque_id,
)
from scripts.e0_evidence.wire import canonical_sha256
from yamabiko import hf_env  # noqa: F401  # pin the production HF cache first
from yamabiko import models
from yamabiko.asr.asr_client import ASRResult, FasterWhisperASR, build_asr
from yamabiko.config import load_full_config


pytestmark = pytest.mark.real


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _private_master() -> tuple[object, CanonicalMaster]:
    fixture_id = os.environ.get("YAMABIKO_E0_REAL_FIXTURE_ID")
    private_root = os.environ.get("YAMABIKO_E0_PRIVATE_ROOT")
    if not fixture_id or not private_root:
        pytest.skip(
            "YAMABIKO_E0_PRIVATE_ROOT and YAMABIKO_E0_REAL_FIXTURE_ID are required"
        )

    root = Path(private_root)
    if not root.is_dir():
        pytest.skip("the declared E0 private root is unavailable")
    boundary = boundary_from_environment(_repo(), os.environ)
    fixture_path = require_private_path(
        boundary.root / "fixtures" / fixture_id / "fixture.private.json",
        boundary,
    )
    if not fixture_path.is_file():
        pytest.skip("the declared rights-approved fixture is unavailable")
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        pytest.fail("the declared private fixture manifest is unreadable")
    if (
        not isinstance(fixture, dict)
        or fixture.get("fixture_id") != fixture_id
        or type(fixture.get("rights_record_id")) is not str
    ):
        pytest.fail("the declared private fixture manifest is invalid")

    # Rights are checked, including their append-only review history, before
    # any master manifest is selected or any private PCM is opened.
    rights = _load_rights(fixture["rights_record_id"], boundary)
    try:
        require_local_evaluation(rights)
    except PermissionError:
        pytest.skip("the declared fixture is not approved for local evaluation")

    master_ids: list[str] = []
    masters_root = require_private_path(boundary.root / "masters", boundary)
    for path in masters_root.glob("*/master.private.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            manifest = master_manifest_from_wire(value)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            pytest.fail("a registered private master manifest is invalid")
        if manifest.fixture_id == fixture_id:
            master_ids.append(manifest.master_id)
    if not master_ids:
        pytest.skip("the declared fixture has no registered immutable master")
    assert len(master_ids) == 1, "the real-seam fixture must name exactly one master"

    manifest = load_registered_master(master_ids[0], boundary)
    assert manifest.rights == rights
    return boundary, load_canonical_master(manifest, boundary)


def _require_cached_silero() -> None:
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    hit = try_to_load_from_cache(
        repo_id=models.SILERO_REPO,
        filename=models.SILERO_FILE,
        revision=models.SILERO_REVISION,
    )
    if hit is None or hit is _CACHED_NO_EXIST or not Path(hit).is_file():
        pytest.skip("the pinned Silero ONNX model is not cached")


def test_real_e0_seam_replays_and_verifies_complete_ledger() -> None:
    """A real replay and P0/P2a decode must write a ledger that verifies complete."""

    if os.environ.get("YAMABIKO_E0_RUN_SEAM") != "1":
        pytest.skip(
            "set YAMABIKO_E0_RUN_SEAM=1 to write a seam run into the private root"
        )
    boundary, master = _private_master()
    capture = load_capture_replay_stream(master, boundary)
    _require_cached_silero()

    from yamabiko.gpu import detect_vram_mib

    if detect_vram_mib() is None:
        pytest.skip("CUDA is unavailable")

    full_config = load_full_config()
    try:
        model_artifact = _resolve_configured_model_artifact(full_config)
    except ValueError:
        pytest.skip("the configured faster-whisper snapshot is not fully cached")
    model_artifact_sha256 = _hash_artifact(model_artifact)
    vad_artifact = resolve_vad_artifact()
    config_sha256 = canonical_sha256(full_config)

    p0_arm_id = new_opaque_id()
    p2a_arm_id = new_opaque_id()
    run_id = new_opaque_id()
    # The seam scores nothing: it declares one event covering the master, and
    # each arm records that event's fate as unknown.
    seam_event_id = new_opaque_id()
    start = RunStartManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        session_id=new_opaque_id(),
        clock_id=MONOTONIC_CLOCK_ID,
        canonical_stream_id=new_opaque_id(),
        private_text_stream_id=new_opaque_id(),
        arm_definitions=(
            ArmDefinition(
                arm_id=p0_arm_id,
                kind=ArmKind.P0,
                parent_arm_id=None,
                parameters=_invariant_arm_parameters(
                    ArmKind.P0,
                    config_sha256=config_sha256,
                    model_artifact_sha256=model_artifact_sha256,
                    vad_artifact_sha256=vad_artifact.sha256,
                ),
                declared_changes=(),
            ),
            ArmDefinition(
                arm_id=p2a_arm_id,
                kind=ArmKind.P2A,
                parent_arm_id=p0_arm_id,
                parameters=_invariant_arm_parameters(
                    ArmKind.P2A,
                    config_sha256=config_sha256,
                    model_artifact_sha256=model_artifact_sha256,
                    vad_artifact_sha256=vad_artifact.sha256,
                ),
                declared_changes=(),
            ),
        ),
        classification=PrivateContentClass.PRIVATE_METADATA,
        details={
            "approved_contracts": contracts.approved_contracts(),
            "event_fates_required": True,
            "reference_event_ids": (seam_event_id,),
            "fixture_id": master.manifest.fixture_id,
            "master_id": master.manifest.master_id,
            "master_pcm_sha256": master.manifest.canonical_pcm_sha256,
            "capture_replay": capture.provenance,
            "config_sha256": config_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "vad_artifact": vad_artifact.identity(),
        },
    )
    run_dir = require_private_path(boundary.root / "runs" / run_id, boundary)
    writer = LedgerWriter.create(run_dir, start, boundary=boundary)

    master_bytes = master.samples.tobytes(order="C")
    assert master.samples.size == master.manifest.sample_count
    assert hashlib.sha256(master_bytes).hexdigest() == master.manifest.canonical_pcm_sha256

    gate_config, threshold = production_chunker_config(full_config)
    vad_model = load_production_model(gate_config, vad_artifact)
    tagger = load_production_tagger()
    replay = replay_master(capture, gate_config, threshold, vad_model, tagger, writer)
    assert replay.windows, "the declared speech slice emitted no decoder window"
    replay_contract = next(
        record
        for record in replay.integrity_records
        if record.kind is RecordKind.INTEGRITY
    )
    assert replay_contract.details["master_pcm_sha256"] == master.manifest.canonical_pcm_sha256
    assert replay_contract.details["replay_pcm_sha256"] == (
        capture.provenance["output_pcm_sha256"]
    )
    assert replay_contract.details["master_sample_count"] == master.manifest.sample_count
    assert replay_contract.details["model_sha256"] == vad_model.artifact_sha256
    assert replay_contract.details["vad_assist"]["model_sha256"] == (
        contracts.SOUND_TAGGER_MODEL_SHA256
    )

    construction_config = _config_for_resolved_model(full_config, model_artifact)
    asr = build_asr(construction_config["asr"])
    assert isinstance(asr, FasterWhisperASR)
    bind_model_artifact(asr, model_artifact_sha256)

    p0_results = []
    p2a_results = []
    for window in replay.windows:
        assert window.audio.size == sum(span.sample_count for span in window.source_spans)
        assert hashlib.sha256(window.audio.tobytes(order="C")).hexdigest() == window.pcm_sha256

        p0 = run_p0(window, asr, _ArmSink(writer, p0_arm_id))
        p2a = run_p2a(p0.call, p0.call.delegate, _ArmSink(writer, p2a_arm_id))
        p0_results.append(p0)
        p2a_results.append(p2a)

        assert p0.call.model_artifact_sha256 == model_artifact_sha256
        assert p2a.call.model_artifact_sha256 == model_artifact_sha256
        assert p0.call.decoder_input_pcm_sha256 == hashlib.sha256(
            p0.call.decoder_audio.tobytes(order="C")
        ).hexdigest()
        assert p0.call.call_record.details["decoder_input_samples"] == p0.call.decoder_audio.size
        assert p0.call.call_record.details["raw_segment_count"] == len(p0.call.raw_segments)
        assert np.array_equal(p0.call.decoder_audio, p2a.call.decoder_audio)
        assert p0.call.decoder_input_pcm_sha256 == p2a.call.decoder_input_pcm_sha256
        assert p0.call.effective_arguments == p2a.call.effective_arguments
        # Temperature-fallback segments are sampled and need not repeat.
        p0_tail = stochastic_tail_start(
            segment.temperature for segment in p0.call.raw_segments
        )
        p2a_tail = stochastic_tail_start(
            segment.temperature for segment in p2a.call.raw_segments
        )
        assert p0_tail == p2a_tail
        assert (p0_tail < len(p0.call.raw_segments)) == (
            p2a_tail < len(p2a.call.raw_segments)
        )
        assert p0.call.raw_texts[:p0_tail] == p2a.call.raw_texts[:p2a_tail]

        parity_fields = (
            "start_decoder_sample",
            "end_decoder_sample",
            "raw_start_seconds",
            "raw_end_seconds",
            "timestamp_clamped",
            "projected_source_spans",
            "no_speech_prob",
            "avg_logprob",
            "compression_ratio",
            "temperature",
            "synthetic_pad_hypothesis",
            "synthetic_anchor_timeline_id",
            "synthetic_anchor_sample",
        )
        for raw_p0, raw_p2a in zip(
            p0.call.raw_segments[:p0_tail],
            p2a.call.raw_segments[:p2a_tail],
        ):
            assert all(
                getattr(raw_p0, field_name) == getattr(raw_p2a, field_name)
                for field_name in parity_fields
            )
            assert raw_p0.end_decoder_sample <= p0.call.decoder_audio.size
            assert all(
                math.isfinite(float(value))
                for value in (
                    raw_p0.start_decoder_sample,
                    raw_p0.end_decoder_sample,
                    raw_p0.no_speech_prob,
                    raw_p0.avg_logprob,
                    raw_p0.compression_ratio,
                    raw_p0.temperature,
                )
            )

        for result in (p0, p2a):
            assert isinstance(result.asr_result, ASRResult)
            assert isinstance(result.asr_result.text, str)
            assert result.asr_result.duration_s == pytest.approx(
                window.audio.size / 16_000,
                abs=1 / 16_000,
            )

    assert sum(len(result.call.raw_segments) for result in p0_results) >= 1
    assert len(p0_results) == len(p2a_results) == len(replay.windows)
    for arm_id in (p0_arm_id, p2a_arm_id):
        writer.emit(
            EvidenceRecordDraft(
                arm_id=arm_id,
                stage=Stage.SCORING,
                kind=RecordKind.EVENT_FATE,
                artifact_id=seam_event_id,
                source_spans=(
                    SampleSpan(
                        master.manifest.timeline_id,
                        0,
                        master.manifest.sample_count,
                    ),
                ),
                decision=TerminalOutcome.UNKNOWN.value,
                classification=PrivateContentClass.PRIVATE_AUDIO,
            )
        )
    writer.finish_arm(p0_arm_id, ArmStatus.COMPLETE)
    writer.finish_arm(p2a_arm_id, ArmStatus.COMPLETE)
    final = writer.finalize()
    assert final.status is RunStatus.COMPLETE
    integrity = verify_run(run_dir)
    assert integrity.complete is True, integrity.reasons
    assert len(integrity.computed_arm_statuses) == 2
    assert all(item.status is ArmStatus.COMPLETE for item in integrity.computed_arm_statuses)
