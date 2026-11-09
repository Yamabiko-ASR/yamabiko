"""Focused contracts for pause epochs at the capture/VAD boundary."""
from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yamabiko.capture.envelope import (
    CaptureBlock,
    CaptureDiscontinuity,
    CaptureGapReason,
    CaptureSequenceTracker,
)
from yamabiko.capture.audio import AudioConfig, LoopbackCapture
from yamabiko.main import Segment
from yamabiko.state import AppState, PauseState
from yamabiko.vad.chunker import ChunkerConfig
from yamabiko.vad.silero_chunker import SileroChunker


def _pcm(frames: int, value: float = 0.0) -> np.ndarray:
    return np.full(frames, value, dtype=np.float32)


class _ImmediateLoop:
    def call_soon_threadsafe(self, callback, *args) -> None:
        callback(*args)


class _RecordingResampler:
    instances: list["_RecordingResampler"] = []

    def __init__(self, _input_rate: int, _output_rate: int) -> None:
        self.actions: list[str] = []
        self.instances.append(self)

    def reset(self) -> None:
        self.actions.append("reset")

    def process(self, pcm: np.ndarray) -> np.ndarray:
        self.actions.append("process")
        return np.full(3, pcm[0], dtype=np.float32)


def _open_callback(cap: LoopbackCapture, *, rate: int):
    captured = {}
    cap._pa = MagicMock()
    cap._device_info = {
        "index": 0,
        "defaultSampleRate": rate,
        "maxInputChannels": 1,
    }
    stream = MagicMock()

    def fake_open(**kwargs):
        captured["callback"] = kwargs["stream_callback"]
        return stream

    cap._pa.open.side_effect = fake_open
    with patch("yamabiko.capture.audio.pyaudio.paFloat32", 1):
        cap._open_stream()
    return captured["callback"]


class _ContentModel:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def __call__(self, frame: np.ndarray) -> float:
        self.calls += 1
        return 0.9 if frame[0] > 0.4 else 0.01

    def reset_states(self) -> None:
        self.resets += 1


def _chunker(state: AppState, model: _ContentModel) -> SileroChunker:
    return SileroChunker(
        ChunkerConfig(
            sample_rate=16_000,
            min_chunk_ms=64,
            max_chunk_ms=320,
            min_silence_ms=64,
            speech_pad_ms=0,
        ),
        threshold=0.5,
        state=state,
        model=model,
    )


def test_app_state_publishes_idempotent_immutable_pause_edges() -> None:
    state = AppState()
    initial = state.pause_snapshot()

    assert initial == PauseState(epoch=0, paused=False)
    paused = state.pause()
    assert paused == PauseState(epoch=1, paused=True)
    assert state.pause() is None
    assert state.pause_snapshot() is paused

    resumed = state.resume()
    assert resumed == PauseState(epoch=2, paused=False)
    assert state.resume() is None
    assert state.pause_snapshot() is resumed

    # A previously-read pair cannot tear or change underneath the callback.
    assert initial == PauseState(epoch=0, paused=False)
    with pytest.raises(FrozenInstanceError):
        initial.paused = True  # type: ignore[misc]
    assert "pause_epoch" not in state.state_frame()
    normalized = AppState(paused=1)  # type: ignore[arg-type]
    assert normalized.paused is True
    assert normalized.pause_snapshot() == PauseState(0, True)
    with pytest.raises((TypeError, ValueError)):
        PauseState(-1, False)
    with pytest.raises(TypeError):
        PauseState(True, False)
    with pytest.raises(TypeError):
        PauseState(0, 1)  # type: ignore[arg-type]


def test_pause_epoch_fields_are_additive_and_tracker_preserves_them() -> None:
    legacy_block = CaptureBlock(_pcm(512), 3, 0)
    legacy_marker = CaptureDiscontinuity(CaptureGapReason.INPUT_STATUS_GAP)
    legacy_segment = Segment(_pcm(512), 1.0)

    assert legacy_block.pause_epoch is None
    assert legacy_marker.pause_epoch is None
    assert legacy_segment.pause_epoch is None

    stamped = CaptureBlock(_pcm(512), 3, 512, pause_epoch=4)
    observed = CaptureSequenceTracker().observe(stamped)
    marker = CaptureDiscontinuity(
        CaptureGapReason.USER_RESUME,
        pause_epoch=4,
    )

    assert observed.pause_epoch == 4
    assert marker.pause_epoch == 4
    with pytest.raises(ValueError, match="pause_epoch"):
        CaptureBlock(_pcm(1), 0, 0, pause_epoch=-1)


def test_capture_resets_soxr_before_epoch_input_and_accounts_paused_output() -> None:
    state = AppState()
    queue: asyncio.Queue = asyncio.Queue()
    events = []
    capture = LoopbackCapture(
        AudioConfig(sample_rate=16_000),
        _ImmediateLoop(),
        queue,
        pause_state_probe=state.pause_snapshot,
        on_capture_event=events.append,
    )
    _RecordingResampler.instances.clear()
    source = _pcm(480, 0.25)

    with patch(
        "yamabiko.capture.audio.StreamingResampler",
        _RecordingResampler,
    ):
        callback = _open_callback(capture, rate=48_000)
        assert state.pause() == PauseState(1, True)
        callback(source.tobytes(), source.size, None, 0)
        assert queue.empty()

        assert state.resume() == PauseState(2, False)
        callback(source.tobytes(), source.size, None, 0)

    converter = _RecordingResampler.instances[0]
    assert converter.actions == ["reset", "process", "reset", "process"]
    assert len(events) == 1
    discard = events[0]
    assert (
        discard.event_kind,
        discard.reason,
        discard.start_sample,
        discard.sample_count,
        discard.pause_epoch,
    ) == ("paused_discard", CaptureGapReason.USER_PAUSE, 0, 3, 1)

    resumed = queue.get_nowait()
    assert (resumed.stream_generation, resumed.start_sample) == (0, 3)
    assert resumed.pause_epoch == 2


def test_callback_free_pause_resume_resets_once_and_16khz_keeps_clock() -> None:
    state = AppState()
    queue: asyncio.Queue = asyncio.Queue()
    events = []
    capture = LoopbackCapture(
        AudioConfig(sample_rate=16_000),
        _ImmediateLoop(),
        queue,
        pause_state_probe=state.pause_snapshot,
        on_capture_event=events.append,
    )
    callback = _open_callback(capture, rate=16_000)
    source = _pcm(480, 0.25)

    state.pause()
    callback(source.tobytes(), source.size, None, 0)
    state.resume()
    callback(source.tobytes(), source.size, None, 0)

    resumed = queue.get_nowait()
    assert (resumed.stream_generation, resumed.start_sample) == (0, 480)
    assert resumed.pause_epoch == 2
    assert np.array_equal(resumed.pcm, source)
    assert events[0].sample_count == 480

    # A second stream proves a callback-free 0 -> 1 -> 2 observes one edge
    # at callback time, so its stateful converter resets exactly once.
    state2 = AppState()
    capture2 = LoopbackCapture(
        AudioConfig(sample_rate=16_000),
        _ImmediateLoop(),
        asyncio.Queue(),
        pause_state_probe=state2.pause_snapshot,
    )
    _RecordingResampler.instances.clear()
    with patch(
        "yamabiko.capture.audio.StreamingResampler",
        _RecordingResampler,
    ):
        callback2 = _open_callback(capture2, rate=48_000)
        state2.pause()
        state2.resume()
        callback2(source.tobytes(), source.size, None, 0)
    assert _RecordingResampler.instances[0].actions == ["reset", "process"]


@pytest.mark.asyncio
async def test_apply_pause_edges_survive_a_full_queue_in_order() -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait("already-buffered-audio")
    state = AppState()
    capture = LoopbackCapture(
        AudioConfig(),
        loop,
        queue,
        pause_state_probe=state.pause_snapshot,
    )

    paused = state.pause()
    resumed = state.resume()
    assert paused is not None and resumed is not None
    capture.apply_pause_edge(paused)
    assert capture._discontinuity_pending.is_set()
    capture.apply_pause_edge(resumed)

    assert queue.get_nowait() == "already-buffered-audio"
    first = await asyncio.wait_for(queue.get(), timeout=1.0)
    second = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert [(first.reason, first.pause_epoch), (second.reason, second.pause_epoch)] == [
        (CaptureGapReason.USER_PAUSE, 1),
        (CaptureGapReason.USER_RESUME, 2),
    ]
    await asyncio.sleep(0)
    assert not capture._discontinuity_pending.is_set()


@pytest.mark.asyncio
async def test_chunker_rejects_pre_pause_and_paused_epochs_after_resume() -> None:
    state = AppState()
    pre_pause = CaptureBlock(_pcm(4 * 512, 0.5), 0, 0, pause_epoch=0)
    assert state.pause() == PauseState(1, True)
    while_paused = CaptureBlock(
        _pcm(4 * 512, 0.9),
        0,
        4 * 512,
        pause_epoch=1,
    )
    assert state.resume() == PauseState(2, False)
    resumed = CaptureBlock(
        np.concatenate([_pcm(4 * 512, 0.7), _pcm(2 * 512)]),
        0,
        8 * 512,
        pause_epoch=2,
    )
    model = _ContentModel()
    chunker = _chunker(state, model)
    in_queue: asyncio.Queue = asyncio.Queue()
    out_queue: asyncio.Queue = asyncio.Queue()
    for block in (pre_pause, while_paused, resumed):
        in_queue.put_nowait(block)

    task = asyncio.create_task(chunker.run(in_queue, out_queue))
    try:
        segment = await asyncio.wait_for(out_queue.get(), timeout=1.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert model.calls == 6
    assert len(segment.audio) == 6 * 512
    assert np.count_nonzero(segment.audio == np.float32(0.7)) == 4 * 512
    assert not np.any(segment.audio == np.float32(0.5))
    assert not np.any(segment.audio == np.float32(0.9))
    assert segment.discontinuity_before is True
    assert segment.pause_epoch == 2


@pytest.mark.asyncio
async def test_pause_clears_only_unfinished_text_while_inflight_lines_drain() -> None:
    """One product flow: pause clears a buffered tail without touching
    completed English context; an already-emitted old-epoch segment may drain,
    but neither its tail nor the pre-pause tail can join current-epoch text.
    """
    from types import SimpleNamespace

    from yamabiko.main import (
        PipelineApp,
        Segment,
        translation_stage,
    )
    from yamabiko.sentencing.remainder import PendingRemainderState
    from yamabiko.translation.prompt import ContextBuffer

    class _Translator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def translate(self, text, *_args, **_kwargs) -> str:
            self.calls.append(text)
            return f"English line {len(self.calls)}"

    class _Capture:
        def __init__(self) -> None:
            self.edges: list[PauseState] = []

        def apply_pause_edge(self, edge: PauseState) -> None:
            self.edges.append(edge)

    class _Manager:
        def __init__(self) -> None:
            self.frames: list[dict] = []

        async def broadcast(self, frame: dict):
            self.frames.append(frame)
            return None

    class _Stats:
        def __init__(self) -> None:
            self.segments: list[Segment] = []

        def record(self, segment: Segment) -> None:
            self.segments.append(segment)

    state = AppState()
    remainder = PendingRemainderState()
    capture = _Capture()
    manager = _Manager()
    control = SimpleNamespace(
        app_state=state,
        capture=capture,
        remainder_state=remainder,
        manager=manager,
        post_asr_telemetry=None,
    )
    context = ContextBuffer(max_lines=2)
    context.push("Earlier completed line one")
    context.push("Earlier completed line two")
    translator = _Translator()
    stats = _Stats()
    in_queue: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    task = asyncio.create_task(translation_stage(
        in_queue,
        translator,
        context,
        state,
        stats,
        None,
        3,
        done,
        manager,
        remainder_state=remainder,
    ))

    await in_queue.put(Segment(
        _pcm(1),
        1.0,
        jp="最初です。でも",
        pause_epoch=0,
    ))
    deadline = asyncio.get_running_loop().time() + 1.0
    while len(translator.calls) < 1:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0)
    assert remainder.text == "でも"
    completed_context = context.snapshot()

    # This Segment was already emitted downstream before Pause, but remains
    # queued until after both control edges.
    in_queue.put_nowait(Segment(
        _pcm(1),
        2.0,
        jp="古いです。続き",
        pause_epoch=0,
    ))
    await PipelineApp.set_paused(control, True)
    await PipelineApp.set_paused(control, False)
    assert context.snapshot() == completed_context
    assert remainder.active_epoch == 2
    assert remainder.text == ""

    await in_queue.put(Segment(
        _pcm(1),
        3.0,
        jp="新しいです。",
        pause_epoch=2,
    ))
    await asyncio.wait_for(task, timeout=2.0)

    assert translator.calls == ["最初です。", "古いです。", "新しいです。"]
    assert [segment.pause_epoch for segment in stats.segments] == [0, 0, 2]
    assert remainder.text == ""
    assert capture.edges == [PauseState(1, True), PauseState(2, False)]
    assert [frame["paused"] for frame in manager.frames if frame["type"] == "state"] == [
        True,
        False,
    ]


class _TailTranslator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def translate(self, text, *_args, **_kwargs) -> str:
        self.calls.append(text)
        return f"English line {len(self.calls)}"


def _start_translation(in_queue, translator, state, remainder, telemetry):
    from yamabiko.main import LatencyStats, translation_stage
    from yamabiko.translation.prompt import ContextBuffer

    return asyncio.create_task(translation_stage(
        in_queue,
        translator,
        ContextBuffer(max_lines=2),
        state,
        LatencyStats(),
        None,
        0,
        asyncio.Event(),
        None,
        post_asr_telemetry=telemetry,
        remainder_state=remainder,
    ))


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


@pytest.mark.asyncio
async def test_pause_records_the_buffered_tail_as_cleared_before_any_segment(
    caplog,
) -> None:
    """A pause abandons the unfinished tail at the edge itself; its lineage
    must end there, not wait for a later transcript that may never come.

    Seeded defect: drop the ``TAIL_DISCONTINUITY_CLEARED`` terminal for the
    ref returned by ``advance_epoch`` in ``PipelineApp.set_paused`` -> no
    cleared line is logged at the edge -> red.
    """
    import logging
    from types import SimpleNamespace

    from yamabiko.main import PipelineApp
    from yamabiko.post_asr_telemetry import PostASRTelemetry
    from yamabiko.sentencing.remainder import PendingRemainderState

    caplog.set_level(logging.INFO, logger="yamabiko")
    state = AppState()
    remainder = PendingRemainderState()
    telemetry = PostASRTelemetry(run_id="pause-tail")
    translator = _TailTranslator()
    in_queue: asyncio.Queue = asyncio.Queue()
    task = _start_translation(in_queue, translator, state, remainder, telemetry)
    try:
        await in_queue.put(Segment(_pcm(1), 1.0, jp="最初です。でも", pause_epoch=0))
        deadline = asyncio.get_running_loop().time() + 2.0
        while not translator.calls:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)
        assert remainder.text == "でも"

        control = SimpleNamespace(
            app_state=state,
            capture=SimpleNamespace(apply_pause_edge=lambda edge: None),
            remainder_state=remainder,
            manager=None,
            post_asr_telemetry=telemetry,
        )
        await PipelineApp.set_paused(control, True)
    finally:
        await _stop(task)

    messages = [record.getMessage() for record in caplog.records]
    tail_edges = [
        message for message in messages
        if message.startswith("post_asr edge:") and " kind=tail " in message
    ]
    assert len(tail_edges) == 1
    tail_seq = tail_edges[0].split("artifact_seq=")[1].split()[0]
    cleared = [
        message for message in messages
        if message.startswith("post_asr terminal:")
        and " fate=tail_discontinuity_cleared " in message
    ]
    assert cleared == [
        f"post_asr terminal: run_id=pause-tail artifact_seq={tail_seq} "
        "fate=tail_discontinuity_cleared phase=None attempted=None "
        "accepted=None failed=None"
    ]
    assert translator.calls == ["最初です。"]


@pytest.mark.asyncio
async def test_stale_epoch_segment_never_force_flushes_its_long_tail() -> None:
    """A segment emitted before a pause edge may still drain its complete
    sentences, but its over-cap unfinished tail belongs to the old epoch and
    must not be force-flushed into translation.

    Seeded defect: drop the ``segment_epoch == remainder_state.active_epoch``
    guard on the force-flush branch -> ``[...] ああ...`` is translated -> red.
    """
    from yamabiko.sentencing.remainder import PendingRemainderState

    state = AppState()
    state.pause()
    state.resume()
    remainder = PendingRemainderState(active_epoch=2)
    translator = _TailTranslator()
    in_queue: asyncio.Queue = asyncio.Queue()
    task = _start_translation(in_queue, translator, state, remainder, None)
    try:
        await in_queue.put(Segment(
            _pcm(1), 1.0, jp="古いです。" + "あ" * 301, pause_epoch=0,
        ))
        await in_queue.put(Segment(_pcm(1), 2.0, jp="新しいです。", pause_epoch=2))
        deadline = asyncio.get_running_loop().time() + 2.0
        while "新しいです。" not in translator.calls:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)
    finally:
        await _stop(task)

    assert translator.calls == ["古いです。", "新しいです。"]
    assert remainder.text == ""
