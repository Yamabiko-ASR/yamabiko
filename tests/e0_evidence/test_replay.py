"""Direct behavior checks for exact production-FrameGate replay."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, replace

import numpy as np
import pytest
import soxr

from scripts.e0_evidence import contracts
from scripts.e0_evidence.ledger import (
    EvidenceSink,
    _capture_replay_semantic_reasons,
    _replay_semantic_reasons,
    _validate_canonical_payload,
    _validate_replay_contract_details,
)
from scripts.e0_evidence.pcm import (
    CanonicalMaster,
    CaptureReplayStream,
    MasterManifest,
    MasterTransformDag,
    TransformStep,
    derive_capture_replay_stream,
)
from scripts.e0_evidence.replay import (
    production_chunker_config,
    replay_master,
    vad_assist_layout,
)
from scripts.e0_evidence.schema import (
    MONOTONIC_CLOCK_ID,
    SCHEMA_VERSION,
    ArmDefinition,
    ArmKind,
    CapabilityDecision,
    DiagnosticCode,
    EndpointEmitReason,
    EvidenceRecord,
    EvidenceRecordDraft,
    IntegrityFailureCode,
    Partition,
    PrivateContentClass,
    RecordKind,
    ReplayStopKind,
    RightsRecord,
    RunStartManifest,
    SampleSpan,
)
from scripts.e0_evidence.wire import canonical_sha256
from yamabiko.capture.resampler import StreamingResampler
from yamabiko.vad import silero_chunker, sound_tagger
from yamabiko.vad.chunker import ChunkerConfig
from yamabiko.vad.silero_chunker import FrameGate, FrameGateState, SileroChunker


_RUN_ID = "81000000-0000-4000-8000-000000000001"
_SESSION_ID = "81000000-0000-4000-8000-000000000002"
_MASTER_ID = "81000000-0000-4000-8000-000000000003"
_FIXTURE_ID = "81000000-0000-4000-8000-000000000004"
_TIMELINE_ID = "81000000-0000-4000-8000-000000000005"
_PARENT_ID = "81000000-0000-4000-8000-000000000006"
_TRANSFORM_ID = "81000000-0000-4000-8000-000000000007"
_OUTPUT_ID = "81000000-0000-4000-8000-000000000008"
_RIGHTS_ID = "81000000-0000-4000-8000-000000000009"
_REVIEWER_ID = "81000000-0000-4000-8000-000000000010"
_RIGHTS_EVIDENCE_ID = "81000000-0000-4000-8000-000000000011"
_MODEL_SHA256 = hashlib.sha256(b"scripted-silero-model").hexdigest()


class _ScriptedModel:
    artifact_sha256 = _MODEL_SHA256

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)
        self.calls: list[np.ndarray] = []
        self.reset_count = 0

    def __call__(self, frame: np.ndarray) -> float:
        self.calls.append(frame.copy())
        try:
            return next(self._probabilities)
        except StopIteration as exc:
            raise AssertionError("replay scored an unexpected frame") from exc

    def reset_states(self) -> None:
        self.reset_count += 1


class _SilentTagger:
    """Stand-in for the bundled tagger that hears no voice: Silero alone decides."""

    artifact_sha256 = sound_tagger.MODEL_SHA256

    def __call__(self, window: np.ndarray) -> np.ndarray:
        return np.zeros(sound_tagger.WINDOW_FRAMES, dtype=np.float32)


class _ContentSilero:
    """Scores a frame from its level alone and logs every call and reset."""

    artifact_sha256 = _MODEL_SHA256

    def __init__(self) -> None:
        self.log: list[object] = []

    def __call__(self, frame: np.ndarray) -> float:
        level = float(np.abs(frame).mean())
        self.log.append(level)
        return 0.8 if 0.2 < level < 0.3 else 0.05

    def reset_states(self) -> None:
        self.log.append("reset")


class _LevelTagger:
    """Hears a voice in each 40 ms tag frame whose centre sample is near 0.5."""

    artifact_sha256 = sound_tagger.MODEL_SHA256

    def __call__(self, window: np.ndarray) -> np.ndarray:
        step = sound_tagger.TAG_FRAME_SAMPLES
        centres = window[step // 2::step]
        return np.where((centres > 0.4) & (centres < 0.6), 0.9, 0.0).astype(np.float32)


class _RecordingSink(EvidenceSink):
    def __init__(self) -> None:
        self.records: list[EvidenceRecord] = []

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        record = EvidenceRecord(
            schema_version=SCHEMA_VERSION,
            record_id=f"81000000-0000-4000-8000-{len(self.records) + 100:012d}",
            run_id=_RUN_ID,
            arm_id=draft.arm_id,
            session_id=_SESSION_ID,
            seq=len(self.records),
            observed_monotonic_ns=len(self.records),
            clock_id=MONOTONIC_CLOCK_ID,
            stage=draft.stage,
            kind=draft.kind,
            artifact_id=draft.artifact_id,
            parent_ids=draft.parent_ids,
            source_spans=draft.source_spans,
            observation_status=draft.observation_status,
            decision=draft.decision,
            diagnostics=draft.diagnostics,
            classification=draft.classification,
            details=draft.details,
        )
        self.records.append(record)
        return record


def _master(sample_count: int, *, samples: np.ndarray | None = None) -> CanonicalMaster:
    if samples is None:
        samples = np.linspace(-0.95, 0.95, sample_count, dtype="<f4")
    else:
        samples = np.ascontiguousarray(samples, dtype="<f4").copy()
    samples.setflags(write=False)
    pcm_sha256 = hashlib.sha256(samples.tobytes(order="C")).hexdigest()
    step = TransformStep(
        transform_id=_TRANSFORM_ID,
        transform_version="synthetic-replay-1.0",
        transform_sha256="0" * 64,
        parent_artifact_id=_PARENT_ID,
        parent_byte_sha256="1" * 64,
        output_artifact_id=_OUTPUT_ID,
        # The synthetic source-native master is the mono 16 kHz PCM itself.
        output_byte_sha256=pcm_sha256,
        tool="synthetic",
        tool_version="synthetic 1",
        command_argv=("synthetic",),
        filter_graph="anull",
        stream_index=0,
        excerpt_start_samples=0,
        excerpt_end_samples=sample_count,
        input_sample_rate_hz=16_000,
        input_channels=1,
        input_layout="mono",
        output_sample_rate_hz=16_000,
        output_channels=1,
        output_layout="mono",
        output_sample_count=sample_count,
        channel_operation="mono-pass-through",
        resampler="none",
        resampler_parameters={},
        dtype_conversion="little-endian-float32",
    )
    manifest = MasterManifest(
        schema_version=SCHEMA_VERSION,
        master_id=_MASTER_ID,
        fixture_id=_FIXTURE_ID,
        rights=RightsRecord(
            rights_record_id=_RIGHTS_ID,
            fixture_id=_FIXTURE_ID,
            local_evaluation=CapabilityDecision.APPROVED,
            redistribution=CapabilityDecision.PENDING,
            training=CapabilityDecision.PENDING,
            transcript_publication=CapabilityDecision.PENDING,
            reviewer_id=_REVIEWER_ID,
            evidence_ids=(_RIGHTS_EVIDENCE_ID,),
            retention_version="retention-1.0",
        ),
        partition=Partition.BLOCKER_CLASS,
        timeline_id=_TIMELINE_ID,
        parent_artifact_ids=(_PARENT_ID,),
        byte_sha256=pcm_sha256,
        canonical_pcm_sha256=pcm_sha256,
        sample_rate_hz=16_000,
        channels=1,
        layout="mono",
        dtype="<f4",
        sample_count=sample_count,
        duration_samples=sample_count,
        transform_dag=MasterTransformDag(source_native=step, canonical=step),
        provenance_class="synthetic-direct-behavior-test",
        rights_class="pending",
        retention_version="retention-1.0",
        annotation_set_sha256="3" * 64,
    )
    return CanonicalMaster(manifest=manifest, samples=samples)


def _full_config(
    *,
    sample_rate: int = 16_000,
    min_chunk_ms: int = 64,
    max_chunk_ms: int = 8_000,
    min_silence_ms: int = 64,
    speech_pad_ms: int = 0,
    attach_max_ms: int = 6_000,
    drop_floor_prob: float = 0.10,
    drop_floor_ms: int = 2_000,
    dip_split_enabled: bool = False,
    dip_split_threshold: float = 0.15,
    dip_split_min_piece_ms: int = 1_000,
    threshold: float | None = 0.3,
) -> Mapping[str, object]:
    vad: dict[str, object] = {
        "min_chunk_ms": min_chunk_ms,
        "max_chunk_ms": max_chunk_ms,
        "min_silence_ms": min_silence_ms,
        "speech_pad_ms": speech_pad_ms,
        "glue_max_silence_ms": 1_500,
        "attach_max_ms": attach_max_ms,
        "drop_floor_prob": drop_floor_prob,
        "drop_floor_ms": drop_floor_ms,
        "dip_split_enabled": dip_split_enabled,
        "dip_split_threshold": dip_split_threshold,
        "dip_split_min_piece_ms": dip_split_min_piece_ms,
    }
    if threshold is not None:
        vad["threshold"] = threshold
    return {
        "audio": {"sample_rate": sample_rate},
        "vad": vad,
    }


def _diagnostics(result_records: tuple[EvidenceRecord, ...]) -> tuple[DiagnosticCode, ...]:
    return tuple(
        diagnostic
        for record in result_records
        for diagnostic in record.diagnostics
    )


def _pcm_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(audio.tobytes(order="C")).hexdigest()


def _capture(master: CanonicalMaster) -> CaptureReplayStream:
    # The synthetic masters above are mono 16 kHz, so production capture passes
    # their source-native samples through unchanged.
    return derive_capture_replay_stream(master, master.samples)


def _production_gate(
    samples: np.ndarray,
    config: ChunkerConfig,
    threshold: float,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: object | None = None,
    tagger: object | None = None,
) -> tuple[list[object], list[float], FrameGateState]:
    """Run production ``SileroChunker`` over ``samples`` in 480-sample blocks.

    Returns its segments, the probability its gate received for every frame,
    and the gate's state once capture has stalled and the assist has drained.
    Without ``model`` and ``tagger`` the chunker loads the shipped models.
    """

    monkeypatch.setattr(silero_chunker, "STALL_DRAIN_S", 0.05)
    gates: list[FrameGate] = []
    probabilities: list[float] = []

    class _RecordingGate(FrameGate):
        def __post_init__(self) -> None:
            super().__post_init__()
            gates.append(self)

        def push(self, audio_frame: np.ndarray, prob: float) -> np.ndarray | None:
            probabilities.append(prob)
            return super().push(audio_frame, prob)

    monkeypatch.setattr(silero_chunker, "FrameGate", _RecordingGate)
    chunker = SileroChunker(config, threshold=threshold, model=model, tagger=tagger)
    frame_count = samples.size // 512

    async def run() -> list[object]:
        in_queue: asyncio.Queue = asyncio.Queue()
        out_queue: asyncio.Queue = asyncio.Queue()
        for start in range(0, samples.size, 480):
            in_queue.put_nowait(samples[start:start + 480].copy())
        task = asyncio.create_task(chunker.run(in_queue, out_queue))
        try:
            async with asyncio.timeout(120):
                while len(probabilities) < frame_count:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return [out_queue.get_nowait() for _ in range(out_queue.qsize())]

    segments = asyncio.run(run())
    (gate,) = gates
    return segments, probabilities, gate.state_snapshot()


def _assert_replay_matches_production(
    result: object,
    replay_model_log: list[object] | None,
    production: tuple[list[object], list[float], FrameGateState],
    production_model_log: list[object] | None,
) -> None:
    segments, probabilities, gate_state = production
    assert [window.emit_reason.value for window in result.windows] == [
        segment.vad_emit_reason for segment in segments
    ]
    for window, segment in zip(result.windows, segments, strict=True):
        np.testing.assert_array_equal(window.audio, segment.audio)
    frames = [
        record for record in result.integrity_records if record.kind is RecordKind.FRAME
    ]
    assert [record.details["vad_probability"] for record in frames] == probabilities
    assert replay_model_log == production_model_log
    assert result.terminal_gate_state == gate_state


def test_replay_reproduces_the_production_chunker_gate_with_the_sound_tag_assist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tag-only line (Silero hears nothing), a Silero-only line (the tagger
    # hears nothing), and a tag line that closes only once the assist drains
    # at the end of the master, then a partial frame.
    levels = ((1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (0.5, 0.25), (2.0, 0.0), (0.5, 0.5), (0.3, 0.0))
    signal = np.concatenate(
        [np.full(int(seconds * 16_000), level, dtype="<f4") for seconds, level in levels]
        + [np.zeros(44, dtype="<f4")]
    )
    capture = _capture(_master(signal.size, samples=signal))
    config, threshold = production_chunker_config(_full_config())
    replay_model = _ContentSilero()
    production_model = _ContentSilero()

    result = replay_master(
        capture, config, threshold, replay_model, _LevelTagger(), _RecordingSink()
    )
    production = _production_gate(
        capture.samples,
        config,
        threshold,
        monkeypatch,
        model=production_model,
        tagger=_LevelTagger(),
    )

    assert len(result.windows) == 3
    _assert_replay_matches_production(
        result, replay_model.log, production, production_model.log
    )


@pytest.mark.parametrize(
    ("pin", "unapproved"),
    (
        ("SOUND_TAGGER_SOURCE_SHA256", "0" * 64),
        ("SILERO_CHUNKER_SOURCE_SHA256", "0" * 64),
        ("SOUND_TAGGER_MODEL_SHA256", "0" * 64),
        ("SOUND_TAG_ASSIST_LAYOUT", {**vad_assist_layout(), "assist_weight": 1.0}),
    ),
    ids=("tagger-source", "chunker-source", "tagger-model", "layout"),
)
def test_replay_refuses_an_unapproved_vad_assist_before_recording(
    pin: str,
    unapproved: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contracts, pin, unapproved)
    config, threshold = production_chunker_config(_full_config())
    sink = _RecordingSink()

    with pytest.raises(RuntimeError, match="sound-tag assist is not approved"):
        replay_master(
            _capture(_master(2 * 512)),
            config,
            threshold,
            _ScriptedModel([]),
            _SilentTagger(),
            sink,
        )

    assert sink.records == []


def _stereo_48k_master(
    canonical: np.ndarray,
    source_native: np.ndarray,
) -> CanonicalMaster:
    base = _master(canonical.size)
    samples = np.ascontiguousarray(canonical, dtype="<f4").copy()
    samples.setflags(write=False)
    pcm_sha256 = _pcm_sha256(samples)
    native_frames = source_native.size // 2
    dag = base.manifest.transform_dag
    source_native_step = replace(
        dag.source_native,
        output_byte_sha256=_pcm_sha256(source_native),
        excerpt_end_samples=native_frames,
        input_sample_rate_hz=48_000,
        input_channels=2,
        input_layout="stereo",
        output_sample_rate_hz=48_000,
        output_channels=2,
        output_layout="stereo",
        output_sample_count=native_frames,
        channel_operation="preserve",
    )
    manifest = replace(
        base.manifest,
        byte_sha256=pcm_sha256,
        canonical_pcm_sha256=pcm_sha256,
        transform_dag=MasterTransformDag(
            source_native=source_native_step,
            canonical=dag.canonical,
        ),
    )
    return CanonicalMaster(manifest=manifest, samples=samples)


def test_replay_lockstep_reconstructs_ordinary_and_force_split() -> None:
    master = _master(9 * 512)
    with pytest.raises(ValueError, match="E0 replay requires audio.sample_rate 16000"):
        production_chunker_config(
            _full_config(sample_rate=48_000, max_chunk_ms=128, threshold=None)
        )
    config, threshold = production_chunker_config(
        _full_config(max_chunk_ms=128, threshold=None)
    )
    model = _ScriptedModel(
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.01, 0.01]
    )
    sink = _RecordingSink()

    result = replay_master(
        _capture(master), config, threshold, model, _SilentTagger(), sink
    )

    assert config.sample_rate == 16_000
    assert threshold == 0.3
    assert result.pending_spans == ()
    assert result.partial_frame_span is None
    assert result.stop_kind is ReplayStopKind.ORDINARY_EOF
    assert len(result.windows) == 2
    first, second = result.windows
    assert first.source_spans == (SampleSpan(_TIMELINE_ID, 0, 2560),)
    assert first.forced is True
    assert first.dip_cut is False
    assert first.emit_reason is EndpointEmitReason.FORCE_SPLIT
    assert second.source_spans == (SampleSpan(_TIMELINE_ID, 2560, 4608),)
    assert second.forced is False
    assert second.dip_cut is False
    assert second.emit_reason is EndpointEmitReason.SILENCE_CLOSE
    assert first.state_before_emit is not None
    assert first.state_after_emit is not None
    assert first.state_before_emit.frames_since_open == 5
    assert first.state_after_emit.retained_frames == 0
    assert second.state_before_emit is not None
    assert second.state_before_emit.consecutive_silence_frames == 2
    np.testing.assert_array_equal(first.audio, master.samples[0:2560])
    np.testing.assert_array_equal(second.audio, master.samples[2560:4608])
    assert first.pcm_sha256 == _pcm_sha256(master.samples[0:2560])
    assert second.pcm_sha256 == _pcm_sha256(master.samples[2560:4608])
    assert len(model.calls) == 9
    assert model.reset_count == 2
    assert result.terminal_gate_state is not None
    assert result.terminal_gate_state.retained_frames == 0
    integrity = next(
        record
        for record in result.integrity_records
        if record.kind is RecordKind.INTEGRITY
    )
    assert integrity.details["model_sha256"] == _MODEL_SHA256
    assert integrity.details["master_pcm_sha256"] == master.manifest.canonical_pcm_sha256
    assert integrity.details["probe_contract_version"] == 4
    approved = contracts.approved_contracts()
    _validate_replay_contract_details(integrity.details, approved)
    legacy_details = dict(integrity.details)
    for key in ("silero_vad_source_sha256", "replay_pcm_sha256", "capture_replay_sha256"):
        legacy_details.pop(key)
    legacy_details["probe_contract_version"] = 2
    with pytest.raises(ValueError, match="invalid shape"):
        _validate_replay_contract_details(legacy_details, approved)
    forged_details = dict(integrity.details)
    forged_details["silero_vad_source_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="silero_vad_source_sha256 is not the run's approved contract",
    ):
        _validate_replay_contract_details(forged_details, approved)
    forged_assist = dict(integrity.details)
    forged_assist["vad_assist"] = {
        **integrity.details["vad_assist"],
        "sound_tagger_source_sha256": "0" * 64,
    }
    with pytest.raises(
        ValueError,
        match="vad_assist.sound_tagger_source_sha256 is not the run's approved contract",
    ):
        _validate_replay_contract_details(forged_assist, approved)
    assert integrity.details["effective_config_sha256"] == canonical_sha256(
        {
            "config": asdict(config),
            "frame_samples": 512,
            "threshold": threshold,
        }
    )
    frame_records = tuple(
        record
        for record in result.integrity_records
        if record.kind is RecordKind.FRAME
    )
    assert [record.details["frame_ordinal"] for record in frame_records] == list(range(9))
    np.testing.assert_array_equal(np.concatenate(model.calls), master.samples)


def test_replay_frames_record_gate_open_exactly_inside_the_emitted_speech_region() -> None:
    master = _master(9 * 512)
    config, threshold = production_chunker_config(_full_config())
    # Two pre-roll frames, three speech frames, then two silent frames close
    # the gate; the last two frames stay pending.
    model = _ScriptedModel([0.01, 0.01, 0.9, 0.9, 0.9, 0.01, 0.01, 0.01, 0.01])

    result = replay_master(
        _capture(master), config, threshold, model, _SilentTagger(), _RecordingSink()
    )

    (window,) = result.windows
    assert window.source_spans == (SampleSpan(_TIMELINE_ID, 0, 7 * 512),)
    frames = tuple(
        record
        for record in result.integrity_records
        if record.kind is RecordKind.FRAME
    )
    assert [record.details["gate_in_speech_after"] for record in frames] == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_replay_preserves_attached_preroll_and_exact_dip_right_piece() -> None:
    master = _master(8 * 512)
    config, threshold = production_chunker_config(
        _full_config(
            min_silence_ms=96,
            attach_max_ms=320,
            drop_floor_prob=0.10,
            drop_floor_ms=320,
            dip_split_enabled=True,
            dip_split_threshold=0.15,
            dip_split_min_piece_ms=64,
        )
    )
    model = _ScriptedModel([0.2, 0.2, 0.9, 0.9, 0.9, 0.05, 0.05, 0.9])
    sink = _RecordingSink()

    result = replay_master(
        _capture(master), config, threshold, model, _SilentTagger(), sink
    )

    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.source_spans == (SampleSpan(_TIMELINE_ID, 0, 2560),)
    assert window.forced is False
    assert window.dip_cut is True
    assert window.emit_reason is EndpointEmitReason.DIP_CUT
    np.testing.assert_array_equal(window.audio, master.samples[0:2560])
    assert window.pcm_sha256 == _pcm_sha256(master.samples[0:2560])
    assert result.pending_spans == (SampleSpan(_TIMELINE_ID, 2560, 4096),)
    assert result.partial_frame_span is None
    assert _diagnostics(result.integrity_records) == (
        DiagnosticCode.ENDPOINT_PENDING_EOF,
    )
    assert len(model.calls) == 8
    assert model.reset_count == 0


@pytest.mark.parametrize(
    (
        "sample_count",
        "probabilities",
        "stop_kind",
        "pending_bounds",
        "partial_bounds",
        "retained_frames",
        "expected_diagnostics",
    ),
    (
        (
            300,
            [],
            ReplayStopKind.ORDINARY_EOF,
            None,
            (0, 300),
            0,
            (DiagnosticCode.ENDPOINT_PARTIAL_FRAME_EOF,),
        ),
        (
            1024,
            [0.9, 0.9],
            ReplayStopKind.ORDINARY_EOF,
            (0, 1024),
            None,
            2,
            (DiagnosticCode.ENDPOINT_PENDING_EOF,),
        ),
        (
            1536,
            [0.2, 0.2, 0.2],
            ReplayStopKind.ORDINARY_EOF,
            (0, 1536),
            None,
            3,
            (DiagnosticCode.ENDPOINT_PENDING_EOF,),
        ),
        # A requested shutdown abandons the frames still waiting in the
        # assist's delay line: Silero never scores them, and they are pending.
        (
            1024,
            [],
            ReplayStopKind.CANCELLED,
            (0, 1024),
            None,
            0,
            (
                DiagnosticCode.ENDPOINT_PENDING_SHUTDOWN,
                DiagnosticCode.SHUTDOWN_CANCELLED,
            ),
        ),
    ),
    ids=("partial-frame", "open-piece", "retained-pending", "cancelled"),
)
def test_replay_reports_termination_state(
    sample_count: int,
    probabilities: list[float],
    stop_kind: ReplayStopKind,
    pending_bounds: tuple[int, int] | None,
    partial_bounds: tuple[int, int] | None,
    retained_frames: int,
    expected_diagnostics: tuple[DiagnosticCode, ...],
) -> None:
    master = _master(sample_count)
    config, threshold = production_chunker_config(_full_config())
    model = _ScriptedModel(probabilities)
    sink = _RecordingSink()

    result = replay_master(
        _capture(master),
        config,
        threshold,
        model,
        _SilentTagger(),
        sink,
        stop_kind=stop_kind,
    )

    expected_pending = (
        ()
        if pending_bounds is None
        else (SampleSpan(_TIMELINE_ID, *pending_bounds),)
    )
    expected_partial = (
        None
        if partial_bounds is None
        else SampleSpan(_TIMELINE_ID, *partial_bounds)
    )
    assert result.windows == ()
    assert result.pending_spans == expected_pending
    assert result.partial_frame_span == expected_partial
    assert result.stop_kind is stop_kind
    assert result.terminal_gate_state is not None
    assert result.terminal_gate_state.retained_frames == retained_frames
    actual_diagnostics = _diagnostics(result.integrity_records)
    assert actual_diagnostics == expected_diagnostics
    assert len(model.calls) == len(probabilities)
    assert model.reset_count == 0
    # verify-run accepts every record the replay wrote.
    approved = contracts.approved_contracts()
    for record in result.integrity_records:
        _validate_canonical_payload(
            record.kind,
            record.decision,
            record.details,
            record.observation_status,
            approved=approved,
        )
    assert _replay_semantic_reasons(result.integrity_records) == ()


def test_replay_feeds_vad_and_windows_the_production_capture_path_of_stereo_48k() -> None:
    native_frames = 3 * 9 * 512
    time = np.arange(native_frames) / 48_000
    left = (0.3 * np.sin(2 * np.pi * 440 * time)).astype("<f4")
    right = (0.2 * np.sin(2 * np.pi * 1_300 * time + 0.3)).astype("<f4")
    source_native = np.stack([left, right], axis=1).reshape(-1)
    # Production capture: per-frame mean downmix, then one continuous MQ stream.
    production = StreamingResampler(48_000, 16_000)
    downmix = np.ascontiguousarray(
        source_native.reshape(-1, 2).mean(axis=1),
        dtype=np.float32,
    )
    expected = np.concatenate([production.process(downmix), production.finish()])
    # Stand-in for the registered FFmpeg master: 0.5L+0.5R through another SoXR
    # recipe, aligned with but not equal to the production samples.
    canonical = soxr.resample(0.5 * left + 0.5 * right, 48_000, 16_000, quality="VHQ")
    master = _stereo_48k_master(canonical, source_native)

    capture = derive_capture_replay_stream(master, source_native)

    np.testing.assert_array_equal(capture.samples, expected)
    assert not np.array_equal(capture.samples, master.samples)
    assert capture.provenance["alignment"]["lag_samples"] == 0
    assert capture.provenance["resampler_quality"] == "MQ"
    assert capture.provenance["soxr_version"] == soxr.__version__
    model = _ScriptedModel([0.9] * 7 + [0.01] * 2)
    config, threshold = production_chunker_config(_full_config())
    result = replay_master(
        capture, config, threshold, model, _SilentTagger(), _RecordingSink()
    )
    np.testing.assert_array_equal(np.concatenate(model.calls), expected)
    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.source_spans == (SampleSpan(_TIMELINE_ID, 0, 9 * 512),)
    np.testing.assert_array_equal(window.audio, expected)
    contract = result.integrity_records[0]
    assert contract.details["replay_pcm_sha256"] == _pcm_sha256(expected)
    assert contract.details["master_pcm_sha256"] == master.manifest.canonical_pcm_sha256
    assert contract.details["capture_replay_sha256"] == canonical_sha256(
        capture.provenance
    )
    start = RunStartManifest(
        schema_version=SCHEMA_VERSION,
        run_id=_RUN_ID,
        session_id=_SESSION_ID,
        clock_id=MONOTONIC_CLOCK_ID,
        canonical_stream_id="81000000-0000-4000-8000-000000000012",
        private_text_stream_id=None,
        arm_definitions=(
            ArmDefinition(
                arm_id="81000000-0000-4000-8000-000000000013",
                kind=ArmKind.P0,
                parent_arm_id=None,
                parameters={},
                declared_changes=(),
            ),
        ),
        classification=PrivateContentClass.PRIVATE_METADATA,
        details={
            "master_pcm_sha256": master.manifest.canonical_pcm_sha256,
            "capture_replay": capture.provenance,
        },
    )
    assert _capture_replay_semantic_reasons(result.integrity_records, start) == ()
    unbound = replace(
        start,
        details={"master_pcm_sha256": master.manifest.canonical_pcm_sha256},
    )
    assert _capture_replay_semantic_reasons(result.integrity_records, unbound) == (
        IntegrityFailureCode.CANONICAL_PAYLOAD_INVALID,
    )
    with pytest.raises(ValueError, match="aligned"):
        derive_capture_replay_stream(
            _stereo_48k_master(np.roll(canonical, 3), source_native),
            source_native,
        )
