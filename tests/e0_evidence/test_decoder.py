"""Direct behavior checks for auditable decoder-arm evidence."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.e0_evidence import contracts
from scripts.e0_evidence import decoder as decoder_module
from scripts.e0_evidence.decoder import (
    DecoderInstrumentationError,
    DecoderParityError,
    OracleComponent,
    SourceDecoderInterval,
    bind_model_artifact,
    oracle_component_map_sha256,
    project_decoder_span,
    run_p0,
    run_p1,
    run_p2a,
    run_p2b,
)
from scripts.e0_evidence.ledger import (
    EvidenceSink,
    _decoder_private_semantic_reasons,
    _decoder_semantic_reasons,
)
from scripts.e0_evidence.pcm import CanonicalMaster, derive_capture_replay_stream
from scripts.e0_evidence.references import (
    ReferenceEvent,
    ReferenceSetManifest,
    reference_content_sha256,
    reference_set_manifest_sha256,
)
from scripts.e0_evidence.replay import ReplayWindow
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
    Partition,
    PrivateContentClass,
    RunStartManifest,
    SampleSpan,
    TextRole,
)
from yamabiko.asr import asr_client
from yamabiko.asr.asr_client import ASROutcome, FasterWhisperASR
from yamabiko.asr.faster_whisper_audio_compat import import_whisper_model


_RUN_ID = "82000000-0000-4000-8000-000000000001"
_SESSION_ID = "82000000-0000-4000-8000-000000000002"
_TIMELINE_ID = "82000000-0000-4000-8000-000000000003"
_MODEL_ARTIFACT_SHA256 = hashlib.sha256(b"synthetic-decoder-model").hexdigest()


class _ArmSink(EvidenceSink):
    def __init__(
        self,
        ordinal: int,
        shared_records: list[EvidenceRecord] | None = None,
    ) -> None:
        self.arm_id = f"82000000-0000-4000-8000-{ordinal:012d}"
        # Arms verified together share one record list so record IDs stay unique.
        self.records: list[EvidenceRecord] = (
            [] if shared_records is None else shared_records
        )
        self.private_texts: list[Mapping[str, object]] = []

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        record = EvidenceRecord(
            schema_version=SCHEMA_VERSION,
            record_id=f"82000000-0000-4000-8001-{len(self.records):012d}",
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

    def emit_private_text(
        self,
        *,
        canonical_record_id: str,
        arm_id: str,
        text_id: str,
        role: TextRole,
        text: str,
    ) -> str:
        assert arm_id == self.arm_id
        self.private_texts.append(
            {
                "canonical_record_id": canonical_record_id,
                "arm_id": arm_id,
                "text_id": text_id,
                "role": role,
                "text": text,
            }
        )
        return f"82000000-0000-4000-8002-{len(self.private_texts):012d}"


class _Segment:
    def __init__(
        self,
        *,
        start: float,
        end: float,
        text: str,
        no_speech_prob: float = 0.1,
        avg_logprob: float = -0.2,
        compression_ratio: float = 1.2,
        temperature: float = 0.0,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.temperature = temperature


_BatchFactory = Callable[[int, Mapping[str, object]], Iterable[object]]


class _Delegate:
    def __init__(self, batches: _BatchFactory) -> None:
        self._batches = batches
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def transcribe(
        self,
        audio: np.ndarray,
        **kwargs: object,
    ) -> tuple[Iterable[object], object]:
        call_index = len(self.calls)
        self.calls.append((audio.copy(), dict(kwargs)))
        return self._batches(call_index, kwargs), SimpleNamespace()


# The fake accepts arbitrary explicit kwargs at runtime but advertises the exact
# installed 1.2.1 call contract to the recorder and direct parity adapter. The
# real class is loaded through the audio shim because collection runs this
# before any test, and a raw import would make every later
# install_audio_compat() in the same session raise.
_Delegate.transcribe.__signature__ = inspect.signature(import_whisper_model().transcribe)  # type: ignore[attr-defined]


def _build_asr(
    monkeypatch: pytest.MonkeyPatch,
    delegate: _Delegate,
    *,
    boundary_pad_ms: int,
    no_speech_only_threshold: float | None = None,
) -> FasterWhisperASR:
    class _Factory:
        def __new__(cls, *args: object, **kwargs: object) -> _Delegate:
            return delegate

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _Factory,
    )
    monkeypatch.setattr(asr_client, "_whisper_cached", lambda _model: False)
    monkeypatch.setattr(asr_client, "_whisper_revision_for", lambda _model: None)
    backend = asr_client.build_asr(
        {
            "boundary_pad_ms": boundary_pad_ms,
            "faster_whisper": {
                "model": "synthetic-model",
                "compute_type": "float16",
                "device": "cuda",
                "beam_size": 5,
                "initial_prompt": "prompt",
                "no_speech_only_threshold": no_speech_only_threshold,
            },
        }
    )
    assert isinstance(backend, FasterWhisperASR)
    return bind_model_artifact(backend, _MODEL_ARTIFACT_SHA256)


def _window(audio: np.ndarray, spans: tuple[SampleSpan, ...]) -> ReplayWindow:
    immutable = np.ascontiguousarray(audio, dtype="<f4").copy()
    immutable.setflags(write=False)
    return ReplayWindow(
        window_id="82000000-0000-4000-8000-000000000020",
        source_spans=spans,
        audio=immutable,
        pcm_sha256=hashlib.sha256(immutable.tobytes(order="C")).hexdigest(),
        emit_reason=EndpointEmitReason.SILENCE_CLOSE,
        forced=False,
        dip_cut=False,
    )


def test_run_p0_records_actual_padded_input_and_raw_segments_lazily_through_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_consumed_first = False

    def record_drop(*args: object, **kwargs: object) -> None:
        nonlocal wrapper_consumed_first
        wrapper_consumed_first = True

    monkeypatch.setattr(asr_client, "_log_junk_drop", record_drop)
    rejected = _Segment(
        start=0.0,
        end=0.0005,
        text="junk",
        no_speech_prob=0.91,
        avg_logprob=-1.4,
        compression_ratio=1.1,
        temperature=0.2,
    )
    kept = _Segment(
        start=0.001,
        end=0.00125,
        text=" kept ",
        no_speech_prob=0.08,
        avg_logprob=-0.15,
        compression_ratio=1.35,
        temperature=0.0,
    )

    def batches(_index: int, _kwargs: Mapping[str, object]) -> Iterable[object]:
        yield rejected
        assert wrapper_consumed_first, "recording proxy consumed the iterator eagerly"
        yield kept

    delegate = _Delegate(batches)
    asr = _build_asr(monkeypatch, delegate, boundary_pad_ms=1)
    spans = (
        SampleSpan(_TIMELINE_ID, 100, 102),
        SampleSpan(_TIMELINE_ID, 200, 202),
    )
    window = _window(np.array([0.25, -0.5, 0.75, -1.0], dtype="<f4"), spans)
    sink = _ArmSink(101)

    result = run_p0(window, asr, sink)

    captured, raw_kwargs = delegate.calls[0]
    assert captured.shape == (36,)
    np.testing.assert_array_equal(captured[:16], np.zeros(16, dtype="<f4"))
    np.testing.assert_array_equal(captured[16:20], window.audio)
    np.testing.assert_array_equal(captured[20:], np.zeros(16, dtype="<f4"))
    assert (
        result.call.decoder_input_pcm_sha256
        == hashlib.sha256(captured.tobytes(order="C")).hexdigest()
    )
    assert result.call.decoder_input_pcm_sha256 != window.pcm_sha256
    assert raw_kwargs == {
        "language": "ja",
        "beam_size": 5,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "vad_filter": False,
        "initial_prompt": "prompt",
        "condition_on_previous_text": False,
    }
    assert result.call.call_kwargs == raw_kwargs
    assert len(result.call.raw_segments) == 2
    assert result.call.raw_segments[0].synthetic_pad_hypothesis is True
    assert result.call.raw_segments[0].synthetic_anchor_sample == 100
    assert result.call.raw_segments[1].projected_source_spans == spans
    assert result.call.raw_segments[1].start_decoder_sample == 16
    assert result.call.raw_segments[1].end_decoder_sample == 20
    assert result.call.raw_segments[1].no_speech_prob == pytest.approx(0.08)
    assert result.call.raw_segments[1].avg_logprob == pytest.approx(-0.15)
    assert result.call.raw_segments[1].compression_ratio == pytest.approx(1.35)
    assert result.call.raw_segments[1].temperature == pytest.approx(0.0)
    assert result.asr_result.text == "kept"
    assert result.asr_result.duration_s == pytest.approx(4 / 16_000)
    assert asr._model is delegate
    assert [item["role"] for item in sink.private_texts] == [
        TextRole.RAW_HYPOTHESIS,
        TextRole.RAW_HYPOTHESIS,
        TextRole.WRAPPER_HYPOTHESIS,
    ]


def test_run_p0_rejects_outcome_that_disagrees_with_independent_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegate = _Delegate(
        lambda _index, _kwargs: iter((_Segment(
            start=0.0, end=0.001, text="kept",
        ),))
    )
    asr = _build_asr(monkeypatch, delegate, boundary_pad_ms=0)
    original = decoder_module._classify_wrapper

    def wrong_outcome(snapshots, no_speech_only_threshold):
        classification = original(snapshots, no_speech_only_threshold)
        return replace(
            classification,
            outcome=ASROutcome.ASSEMBLED_TEXT_EMPTY,
        )

    monkeypatch.setattr(decoder_module, "_classify_wrapper", wrong_outcome)
    with pytest.raises(DecoderInstrumentationError, match="outcome"):
        run_p0(
            _window(
                np.ones(16, dtype="<f4"),
                (SampleSpan(_TIMELINE_ID, 500, 516),),
            ),
            asr,
            _ArmSink(102),
        )


def test_p2a_resolves_defaults_and_proves_raw_parity_before_p2b_changes_only_no_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _Segment(start=0.001, end=0.00125, text=" same", compression_ratio=1.1)
    mismatch = _Segment(start=0.001, end=0.00125, text=" same", compression_ratio=1.7)
    counterfactual = _Segment(
        start=0.00125,
        end=0.0015,
        text=" revealed",
        no_speech_prob=0.92,
        avg_logprob=-1.3,
        temperature=0.4,
    )

    def batches(index: int, kwargs: Mapping[str, object]) -> Iterable[object]:
        if index == 1:
            return iter((mismatch,))
        if "no_speech_threshold" in kwargs and kwargs["no_speech_threshold"] is None:
            return iter((baseline, counterfactual))
        return iter((baseline,))

    delegate = _Delegate(batches)
    asr = _build_asr(monkeypatch, delegate, boundary_pad_ms=1)
    window = _window(
        np.array([0.1, 0.2, 0.3, 0.4], dtype="<f4"),
        (SampleSpan(_TIMELINE_ID, 300, 304),),
    )
    p0 = run_p0(window, asr, _ArmSink(201))

    assert len(p0.call.effective_arguments) == 34
    assert p0.call.effective_arguments["task"] == "transcribe"
    assert p0.call.effective_arguments["temperature"] == (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )
    assert p0.call.effective_arguments["no_speech_threshold"] == 0.6
    with pytest.raises(DecoderParityError, match="compression_ratio"):
        run_p2a(p0.call, delegate, _ArmSink(202))

    p2a = run_p2a(p0.call, delegate, _ArmSink(203))
    p2b = run_p2b(p2a.call, delegate, _ArmSink(204))

    parity_fields = (
        "start_decoder_sample",
        "end_decoder_sample",
        "projected_source_spans",
        "no_speech_prob",
        "avg_logprob",
        "compression_ratio",
        "temperature",
        "synthetic_pad_hypothesis",
        "synthetic_anchor_timeline_id",
        "synthetic_anchor_sample",
    )
    assert p2a.call.raw_texts == p0.call.raw_texts
    assert all(
        getattr(p2a.call.raw_segments[0], field_name)
        == getattr(p0.call.raw_segments[0], field_name)
        for field_name in parity_fields
    )
    assert len(delegate.calls[2][1]) == 34
    assert delegate.calls[2][1]["no_speech_threshold"] == 0.6
    assert delegate.calls[3][1]["no_speech_threshold"] is None
    assert [call[0].shape for call in delegate.calls] == [(36,), (36,), (36,), (36,)]
    assert (
        len({hashlib.sha256(call[0].tobytes()).hexdigest() for call in delegate.calls})
        == 1
    )
    assert p2a.call.decoder_input_pcm_sha256 == p0.call.decoder_input_pcm_sha256
    assert p2b.call.decoder_input_pcm_sha256 == p2a.call.decoder_input_pcm_sha256
    assert p2b.call.changed_argument_keys == ("no_speech_threshold",)
    assert p2b.call.effective_arguments["no_speech_threshold"] is None
    assert set(p2b.call.effective_arguments) == set(p2a.call.effective_arguments)
    assert all(
        p2b.call.effective_arguments[key] == p2a.call.effective_arguments[key]
        for key in p2a.call.effective_arguments
        if key != "no_speech_threshold"
    )
    assert p2b.call.delegate_instance_id == p2a.call.delegate_instance_id
    assert len(p2b.call.raw_segments) == 2
    with pytest.raises(DecoderInstrumentationError, match="delegate"):
        run_p2b(p2a.call, _Delegate(batches), _ArmSink(205))


def test_project_decoder_span_rounds_half_up_and_preserves_real_islands_between_pads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = (
        SourceDecoderInterval(0, 3, None),
        SourceDecoderInterval(3, 5, SampleSpan(_TIMELINE_ID, 100, 102)),
        SourceDecoderInterval(5, 7, SampleSpan(_TIMELINE_ID, 200, 202)),
        SourceDecoderInterval(7, 10, None),
    )

    projected = project_decoder_span(0.25, 0.75, 10, 10, mapping)
    synthetic = project_decoder_span(0.05, 0.15, 10, 10, mapping)

    assert (projected.decoder_start, projected.decoder_end) == (3, 8)
    assert projected.source_spans == (
        SampleSpan(_TIMELINE_ID, 100, 102),
        SampleSpan(_TIMELINE_ID, 200, 202),
    )
    assert projected.synthetic_pad_hypothesis is False
    assert projected.synthetic_anchor_timeline_id is None
    assert projected.synthetic_anchor_sample is None
    assert (synthetic.decoder_start, synthetic.decoder_end) == (1, 2)
    assert synthetic.source_spans == ()
    assert synthetic.synthetic_pad_hypothesis is True
    assert synthetic.synthetic_anchor_timeline_id == _TIMELINE_ID
    assert synthetic.synthetic_anchor_sample == 100

    # P1 uses the same coordinate contract, but its sole real interval must be
    # a frozen reference component's full envelope plus exactly 8,000 samples
    # of real master context on each available side.
    from tests.e0_evidence.test_replay import _master as replay_master

    base_master = replay_master(20_000)
    event_ids = (
        "82000000-0000-4000-8003-000000000001",
        "82000000-0000-4000-8003-000000000002",
    )
    events = (
        ReferenceEvent(
            event_id=event_ids[0],
            master_id=base_master.manifest.master_id,
            span=SampleSpan(base_master.manifest.timeline_id, 9_000, 9_010),
            intelligibility="intelligible",
            verbatim_ref_id=None,
            tags=(),
            annotation_version="event-1.0",
        ),
        ReferenceEvent(
            event_id=event_ids[1],
            master_id=base_master.manifest.master_id,
            span=SampleSpan(base_master.manifest.timeline_id, 10_000, 10_020),
            intelligibility="intelligible",
            verbatim_ref_id=None,
            tags=(),
            annotation_version="event-1.0",
        ),
    )
    references = ReferenceSetManifest(
        schema_version=SCHEMA_VERSION,
        reference_set_id="82000000-0000-4000-8003-000000000003",
        fixture_id=base_master.manifest.fixture_id,
        master_id=base_master.manifest.master_id,
        partition=Partition.DEVELOPMENT,
        reference_version="oracle-1.0",
        reference_sha256="0" * 64,
        reference_set_sha256="0" * 64,
        events=events,
        verbatim_references=(),
        editorial_subtitles=(),
        annotation_passes=(),
        adjudication=None,
        non_speech_controls=(),
    )
    references = replace(
        references,
        reference_sha256=reference_content_sha256(references),
    )
    references = replace(
        references,
        reference_set_sha256=reference_set_manifest_sha256(references),
    )
    master = CanonicalMaster(
        replace(
            base_master.manifest,
            partition=Partition.DEVELOPMENT,
            annotation_set_sha256=references.reference_set_sha256,
        ),
        base_master.samples,
    )
    component_id = "82000000-0000-4000-8003-000000000004"
    component = OracleComponent(
        oracle_component_id=component_id,
        reference_set_manifest=references,
        master=master,
        event_ids=event_ids,
        component_map_sha256=oracle_component_map_sha256(
            component_id,
            references,
            master,
            event_ids,
        ),
    )
    oracle_delegate = _Delegate(lambda _index, _kwargs: iter(()))
    capture = derive_capture_replay_stream(master, master.samples)
    oracle = run_p1(
        component,
        capture,
        _build_asr(monkeypatch, oracle_delegate, boundary_pad_ms=1),
        _ArmSink(250),
    )
    oracle_real = tuple(
        interval
        for interval in oracle.call.source_to_decoder_map
        if interval.source_span is not None
    )
    assert tuple(interval.source_span for interval in oracle_real) == (
        SampleSpan(master.manifest.timeline_id, 1_000, 18_020),
    )
    assert oracle.call.oracle_event_ids == event_ids
    assert oracle.call.oracle_reference_set_sha256 == references.reference_set_sha256
    assert (
        oracle.call.oracle_component_map_sha256 != oracle.call.oracle_slice_pcm_sha256
    )
    denied = CanonicalMaster(
        replace(
            master.manifest,
            rights=replace(
                master.manifest.rights,
                local_evaluation=CapabilityDecision.DENIED,
            ),
        ),
        master.samples,
    )
    denied_component = OracleComponent(
        oracle_component_id=component_id,
        reference_set_manifest=references,
        master=denied,
        event_ids=event_ids,
        component_map_sha256=oracle_component_map_sha256(
            component_id,
            references,
            denied,
            event_ids,
        ),
    )
    with pytest.raises(PermissionError):
        run_p1(
            denied_component,
            capture,
            _build_asr(monkeypatch, oracle_delegate, boundary_pad_ms=1),
            _ArmSink(251),
        )
    assert len(oracle_delegate.calls) == 1


def test_wrapper_decisions_stay_segment_local_and_compression_is_window_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans = (
        SampleSpan(_TIMELINE_ID, 1_000, 1_016),
        SampleSpan(_TIMELINE_ID, 2_000, 2_016),
    )
    audio = np.linspace(-0.8, 0.8, 32, dtype="<f4")
    confidence_delegate = _Delegate(
        lambda _index, _kwargs: iter(
            (
                _Segment(
                    start=0.0,
                    end=0.001,
                    text="target",
                    no_speech_prob=0.9,
                    avg_logprob=-1.5,
                ),
                _Segment(start=0.001, end=0.002, text="elsewhere"),
            )
        )
    )
    confidence = run_p0(
        _window(audio, spans),
        _build_asr(monkeypatch, confidence_delegate, boundary_pad_ms=0),
        _ArmSink(301),
    )
    confidence_drop = next(
        record
        for record in confidence.wrapper_records
        if record.decision == DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value
    )
    assert confidence_drop.source_spans == (spans[0],)
    assert confidence.asr_result.text == "elsewhere"
    assert confidence.wrapper_record.parent_ids == (
        confidence.raw_records[1].record_id,
    )

    compression_delegate = _Delegate(
        lambda _index, _kwargs: iter(
            tuple(
                _Segment(
                    start=index * 0.001,
                    end=(index + 1) * 0.001,
                    text="abcdefghij",
                    compression_ratio=3.0,
                )
                for index in range(3)
            )
        )
    )
    compression = run_p0(
        _window(
            np.linspace(-0.5, 0.5, 48, dtype="<f4"),
            (SampleSpan(_TIMELINE_ID, 3_000, 3_048),),
        ),
        _build_asr(monkeypatch, compression_delegate, boundary_pad_ms=0),
        _ArmSink(302),
    )
    assert compression.asr_result.text == ""
    assert compression.wrapper_record.decision == (
        DiagnosticCode.WRAPPER_COMPRESSION_REJECT.value
    )
    assert compression.wrapper_record.parent_ids == tuple(
        record.record_id for record in compression.raw_records
    )

    zero_delegate = _Delegate(lambda _index, _kwargs: iter(()))
    zero = run_p0(
        _window(np.ones(16, dtype="<f4"), (SampleSpan(_TIMELINE_ID, 4_000, 4_016),)),
        _build_asr(monkeypatch, zero_delegate, boundary_pad_ms=0),
        _ArmSink(303),
    )
    assert zero.call.call_record.decision == DiagnosticCode.DECODER_NO_YIELD.value
    assert zero.wrapper_record.decision == DiagnosticCode.WRAPPER_EMPTY.value
    assert all(
        record.decision != "decoder_no_speech" for record in zero.evidence_records
    )


def _verifier_start(
    *arms: tuple[ArmKind, str, str | None],
    effective_config: Mapping[str, object] | None = None,
) -> RunStartManifest:
    details: dict[str, object] = {"approved_contracts": contracts.approved_contracts()}
    if effective_config is not None:
        details["effective_config"] = effective_config
    return RunStartManifest(
        schema_version=SCHEMA_VERSION,
        run_id=_RUN_ID,
        session_id=_SESSION_ID,
        clock_id=MONOTONIC_CLOCK_ID,
        canonical_stream_id="82000000-0000-4000-8004-000000000001",
        private_text_stream_id="82000000-0000-4000-8004-000000000002",
        arm_definitions=tuple(
            ArmDefinition(
                arm_id=arm_id,
                kind=kind,
                parent_arm_id=parent_arm_id,
                parameters={
                    "decoder_random_seed": 105,
                    "model_artifact_sha256": _MODEL_ARTIFACT_SHA256,
                },
                declared_changes=(),
            )
            for kind, arm_id, parent_arm_id in arms
        ),
        classification=PrivateContentClass.PRIVATE_METADATA,
        details=details,
    )


def test_p0_completes_and_flags_dependency_timestamps_past_the_decoder_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # faster-whisper 1.2.1 never clamps timestamp-token times to the audio, so a
    # hallucination on a short garbled window can end, or even start, after the
    # padded decoder input.
    overshoot = _Segment(start=0.0015, end=0.005, text=" tail")
    beyond = _Segment(start=0.004, end=0.0045, text=" ghost")
    delegate = _Delegate(lambda _index, _kwargs: iter((overshoot, beyond)))
    asr = _build_asr(monkeypatch, delegate, boundary_pad_ms=1)
    sink = _ArmSink(401)
    window = _window(
        np.linspace(-0.4, 0.4, 16, dtype="<f4"),
        (SampleSpan(_TIMELINE_ID, 700, 716),),
    )

    result = run_p0(window, asr, sink)

    assert result.call.decoder_audio.size == 48
    first, second = result.call.raw_segments
    assert (first.start_decoder_sample, first.end_decoder_sample) == (24, 48)
    assert (first.raw_start_seconds, first.raw_end_seconds) == (0.0015, 0.005)
    assert first.timestamp_clamped is True
    assert first.projected_source_spans == (SampleSpan(_TIMELINE_ID, 708, 716),)
    assert (second.start_decoder_sample, second.end_decoder_sample) == (47, 48)
    assert second.timestamp_clamped is True
    assert second.synthetic_pad_hypothesis is True
    assert second.synthetic_anchor_sample == 716
    assert result.asr_result.text == "tail ghost"
    clamped = DiagnosticCode.DECODER_TIMESTAMP_CLAMPED
    assert [record.diagnostics for record in result.raw_records] == [
        (clamped,),
        (clamped,),
    ]
    assert [
        (
            record.details["raw_start_seconds"],
            record.details["raw_end_seconds"],
            record.details["timestamp_clamped"],
        )
        for record in result.raw_records
    ] == [(0.0015, 0.005, True), (0.004, 0.0045, True)]
    start = _verifier_start((ArmKind.P0, sink.arm_id, None))
    assert _decoder_semantic_reasons(tuple(sink.records), start) == ()


def test_p2a_exempts_temperature_fallback_segments_but_rejects_greedy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CTranslate2 seeds its sampling generator once per thread, so reseeding
    # before P2a cannot repeat P0's temperature-fallback draw.
    greedy = _Segment(start=0.001, end=0.0015, text=" same")
    p0_draw = _Segment(
        start=0.0015, end=0.002, text=" first draw", avg_logprob=-0.9, temperature=0.2
    )
    p2a_draw = _Segment(
        start=0.0015, end=0.0025, text=" other draw", avg_logprob=-0.7, temperature=0.4
    )
    greedy_drift = _Segment(start=0.001, end=0.0015, text=" same", avg_logprob=-0.25)

    def batches(index: int, _kwargs: Mapping[str, object]) -> Iterable[object]:
        if index == 0:
            return iter((greedy, p0_draw))
        if index == 1:
            return iter((greedy, p2a_draw))
        return iter((greedy_drift, p2a_draw))

    delegate = _Delegate(batches)
    asr = _build_asr(monkeypatch, delegate, boundary_pad_ms=1)
    records: list[EvidenceRecord] = []
    p0_sink = _ArmSink(501, records)
    p2a_sink = _ArmSink(502, records)
    window = _window(
        np.linspace(-0.3, 0.3, 32, dtype="<f4"),
        (SampleSpan(_TIMELINE_ID, 900, 932),),
    )
    p0 = run_p0(window, asr, p0_sink)

    p2a = run_p2a(p0.call, delegate, p2a_sink)

    stochastic = DiagnosticCode.DECODER_STOCHASTIC_FALLBACK
    assert p2a.call.raw_texts == (" same", " other draw")
    assert p0.call.call_record.diagnostics == (stochastic,)
    assert p2a.call.call_record.diagnostics == (stochastic,)
    assert [record.diagnostics for record in p2a.raw_records] == [(), (stochastic,)]
    start = _verifier_start(
        (ArmKind.P0, p0_sink.arm_id, None),
        (ArmKind.P2A, p2a_sink.arm_id, p0_sink.arm_id),
    )
    assert _decoder_semantic_reasons(tuple(records), start) == ()
    private_texts = tuple(
        SimpleNamespace(
            canonical_record_id=item["canonical_record_id"],
            text=item["text"],
        )
        for item in (*p0_sink.private_texts, *p2a_sink.private_texts)
    )
    assert _decoder_private_semantic_reasons(tuple(records), private_texts, start) == ()
    with pytest.raises(DecoderParityError, match=r"raw\[0\]\.avg_logprob"):
        run_p2a(p0.call, delegate, _ArmSink(503))


@pytest.mark.parametrize("threshold", [0.95, None, 1.0])
def test_no_speech_only_gate_replays_from_the_runs_effective_threshold(
    monkeypatch: pytest.MonkeyPatch,
    threshold: float | None,
) -> None:
    """A confident phantom (no_speech_prob 1.0, avg_logprob -0.3) and a
    whitespace-only segment with the same scores both pass the paired gate.
    With the run's no_speech_only_threshold at 0.95, P0 and P2a record both as
    wrapper_confidence_gate drops; with it null, both stay kept. 1.0, the
    "off" value a config can still express once the shipped default is a
    number, keeps both too. The verifier takes the threshold from
    run-start's effective_config.

    Dropping the whitespace segment changes neither the assembled text nor the
    outcome, so only its decision record shows the gate was modelled.
    Seeded defects: ignore the threshold in _classify_wrapper -> run_p0 raises
    on the phantom; skip it for whitespace-only text -> the blank stays kept
    -> red; verify with a fixed threshold instead of effective_config -> one
    parametrized case reports invalid -> red; keep "< 1.0" in _WrapperCall or
    the ledger -> the 1.0 case raises or reports invalid -> red.
    """
    clean = _Segment(start=0.0, end=0.0006, text="本物のセリフ",
                     no_speech_prob=0.02, avg_logprob=-0.2)
    phantom = _Segment(start=0.0006, end=0.0012, text="おはようございます。",
                       no_speech_prob=1.0, avg_logprob=-0.3)
    blank = _Segment(start=0.0012, end=0.002, text=" ",
                     no_speech_prob=1.0, avg_logprob=-0.3)
    delegate = _Delegate(lambda _index, _kwargs: iter((clean, phantom, blank)))
    asr = _build_asr(
        monkeypatch, delegate, boundary_pad_ms=0, no_speech_only_threshold=threshold,
    )
    records: list[EvidenceRecord] = []
    p0_sink = _ArmSink(601, records)
    p2a_sink = _ArmSink(602, records)
    window = _window(
        np.linspace(-0.3, 0.3, 32, dtype="<f4"),
        (SampleSpan(_TIMELINE_ID, 5_000, 5_032),),
    )

    p0 = run_p0(window, asr, p0_sink)
    p2a = run_p2a(p0.call, delegate, p2a_sink)

    clean_id, phantom_id, blank_id = (record.record_id for record in p0.raw_records)
    gated = tuple(
        record.parent_ids
        for record in p0.wrapper_records
        if record.decision == DiagnosticCode.WRAPPER_CONFIDENCE_GATE.value
    )
    if threshold in (None, 1.0):
        assert p0.asr_result.text == "本物のセリフおはようございます。"
        assert gated == ()
        assert p0.wrapper_record.parent_ids == (clean_id, phantom_id, blank_id)
    else:
        assert p0.asr_result.text == "本物のセリフ"
        assert gated == ((phantom_id,), (blank_id,))
        assert p0.wrapper_record.parent_ids == (clean_id,)
    assert p2a.asr_result == p0.asr_result
    start = _verifier_start(
        (ArmKind.P0, p0_sink.arm_id, None),
        (ArmKind.P2A, p2a_sink.arm_id, p0_sink.arm_id),
        effective_config={
            "asr": {"faster_whisper": {"no_speech_only_threshold": threshold}},
        },
    )
    assert _decoder_semantic_reasons(tuple(records), start) == ()
    private_texts = tuple(
        SimpleNamespace(
            canonical_record_id=item["canonical_record_id"],
            text=item["text"],
        )
        for item in (*p0_sink.private_texts, *p2a_sink.private_texts)
    )
    assert _decoder_private_semantic_reasons(tuple(records), private_texts, start) == ()
