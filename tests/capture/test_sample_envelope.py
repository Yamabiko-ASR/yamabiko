"""Focused capture-coordinate and migration contract tests."""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yamabiko.capture.audio import AudioConfig, LoopbackCapture
from yamabiko.capture.envelope import (
    CaptureBlock,
    CaptureDiscontinuity,
    CaptureGapReason,
    CaptureSequenceTracker,
    GapExtent,
)
from yamabiko.state import PauseState
from yamabiko.vad.chunker import ChunkerConfig
from yamabiko.vad.silero_chunker import SileroChunker


def _pcm(size: int, value: float = 0.0) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


class _ScriptedResampler:
    scripts: list[list[np.ndarray]] = []
    instances: list["_ScriptedResampler"] = []

    def __init__(self, _in_rate, _out_rate) -> None:
        self.outputs = self.scripts.pop(0)
        self.reset_calls = 0
        self.instances.append(self)

    def process(self, _pcm: np.ndarray) -> np.ndarray:
        return self.outputs.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


def _open_callback(cap: LoopbackCapture, *, rate: int = 48_000):
    captured = {}
    stream = MagicMock()
    cap._pa = MagicMock()
    cap._device_info = {
        "index": 0,
        "defaultSampleRate": rate,
        "maxInputChannels": 1,
    }

    def fake_open(**kwargs):
        captured["callback"] = kwargs["stream_callback"]
        return stream

    cap._pa.open.side_effect = fake_open
    with patch("yamabiko.capture.audio.pyaudio.paFloat32", 1):
        cap._open_stream()
    return captured["callback"]


def test_tracker_reports_contiguous_blocks_and_exact_same_generation_jump():
    tracker = CaptureSequenceTracker()

    first = tracker.observe(CaptureBlock(_pcm(4), 7, 0))
    contiguous = tracker.observe(CaptureBlock(_pcm(3), 7, 4))
    jumped = tracker.observe(CaptureBlock(_pcm(2), 7, 11))

    assert (first.stream_generation, first.start_sample) == (7, 0)
    assert first.exact_gap_samples is None
    assert contiguous.exact_gap_samples == 0
    assert jumped.exact_gap_samples == 4
    assert not first.generation_changed
    assert not contiguous.generation_changed
    assert not jumped.generation_changed


def test_tracker_marks_generation_change_unknown_and_rejects_overlap():
    tracker = CaptureSequenceTracker()
    tracker.observe(CaptureBlock(_pcm(4), 7, 0))

    changed = tracker.observe(CaptureBlock(_pcm(3), 8, 0))

    assert changed.generation_changed
    assert changed.exact_gap_samples is None
    with pytest.raises(ValueError, match="overlap|out of order"):
        tracker.observe(CaptureBlock(_pcm(2), 8, 2))


def test_tracker_clears_coordinate_baseline_across_legacy_arrays_both_ways():
    tracker = CaptureSequenceTracker()
    tracker.observe(CaptureBlock(_pcm(4), 2, 0))

    legacy = tracker.observe(_pcm(3))
    after_legacy = tracker.observe(CaptureBlock(_pcm(2), 2, 20))
    contiguous_again = tracker.observe(CaptureBlock(_pcm(1), 2, 22))

    assert legacy.pcm.shape == (3,)
    assert legacy.stream_generation is None
    assert legacy.start_sample is None
    assert legacy.exact_gap_samples is None
    assert not legacy.generation_changed
    assert after_legacy.exact_gap_samples is None
    assert not after_legacy.generation_changed
    assert contiguous_again.exact_gap_samples == 0


def test_capture_block_rejects_mutable_alias_instead_of_copying_it():
    owner = _pcm(8)
    alias = owner[2:6]

    with pytest.raises(ValueError, match="own"):
        CaptureBlock(alias, 0, 0)


def test_callback_stamps_generation_and_exact_variable_output_cursor():
    loop = MagicMock()
    cap = LoopbackCapture(AudioConfig(sample_rate=16_000), loop, MagicMock())
    _ScriptedResampler.instances.clear()
    _ScriptedResampler.scripts = [
        [_pcm(0), _pcm(3, 0.1), _pcm(2, 0.2)],
        [_pcm(4, 0.3)],
    ]
    source = _pcm(480)

    with patch(
        "yamabiko.capture.audio.StreamingResampler",
        _ScriptedResampler,
    ):
        first_callback = _open_callback(cap)
        for _ in range(3):
            first_callback(source.tobytes(), source.size, None, 0)
        second_callback = _open_callback(cap)
        second_callback(source.tobytes(), source.size, None, 0)

    blocks = [
        call.args[1]
        for call in loop.call_soon_threadsafe.call_args_list
        if call.args[0] == cap._enqueue
    ]
    first_generation = blocks[0].stream_generation
    assert [block.stream_generation for block in blocks] == [
        first_generation,
        first_generation,
        first_generation + 1,
    ]
    assert [block.start_sample for block in blocks] == [0, 3, 0]
    assert [block.pcm.size for block in blocks] == [3, 2, 4]
    assert all(block.pcm.flags.owndata for block in blocks)


def test_queue_drop_is_exact_after_conversion_and_never_resets_soxr():
    loop = MagicMock()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(object())
    events = []
    cap = LoopbackCapture(
        AudioConfig(sample_rate=16_000),
        loop,
        queue,
        on_capture_event=events.append,
    )
    _ScriptedResampler.instances.clear()
    _ScriptedResampler.scripts = [[_pcm(3), _pcm(5)]]
    source = _pcm(480)

    with patch(
        "yamabiko.capture.audio.StreamingResampler",
        _ScriptedResampler,
    ):
        callback = _open_callback(cap)
        callback(source.tobytes(), source.size, None, 0)
        first = loop.call_soon_threadsafe.call_args_list[-1].args[1]
        cap._enqueue(first)
        callback(source.tobytes(), source.size, None, 0)

    blocks = [
        call.args[1]
        for call in loop.call_soon_threadsafe.call_args_list
        if call.args[0] == cap._enqueue
    ]
    assert [block.start_sample for block in blocks] == [0, 3]
    assert _ScriptedResampler.instances[0].reset_calls == 0
    assert len(events) == 1
    event = events[0]
    assert event.event_kind == "output_dropped"
    assert event.extent is GapExtent.EXACT_OUTPUT_SAMPLES
    assert event.reason is CaptureGapReason.AUDIO_QUEUE_OVERFLOW
    assert (event.stream_generation, event.start_sample, event.sample_count) == (
        first.stream_generation,
        0,
        3,
    )


@pytest.mark.asyncio
async def test_unknown_callback_notices_remain_ordered_and_content_free():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    events = []
    cap = LoopbackCapture(
        AudioConfig(sample_rate=16_000),
        loop,
        queue,
        on_capture_event=events.append,
    )
    callback = _open_callback(cap, rate=16_000)
    source = _pcm(480)

    callback(
        source.tobytes(),
        source.size,
        None,
        __import__("pyaudiowpatch").paInputOverflow,
    )
    callback(b"\x00\x01\x02", 1, None, 0)

    for _ in range(100):
        if queue.qsize() == 2:
            break
        await asyncio.sleep(0)
    markers = [queue.get_nowait(), queue.get_nowait()]
    source_events = [event for event in events if event.event_kind == "source_gap"]
    barrier_drops = [
        event for event in events if event.event_kind == "output_dropped"
    ]

    assert [marker.reason for marker in markers] == [
        CaptureGapReason.INPUT_STATUS_GAP,
        CaptureGapReason.TRANSFORM_FAILURE,
    ]
    assert all(marker.extent is GapExtent.UNKNOWN_SOURCE_DURATION for marker in markers)
    assert [event.reason for event in source_events] == [
        CaptureGapReason.INPUT_STATUS_GAP,
        CaptureGapReason.TRANSFORM_FAILURE,
    ]
    assert all(event.sample_count is None for event in source_events)
    assert len(barrier_drops) == 1
    assert barrier_drops[0].reason is None
    assert [event.event_sequence for event in events] == list(
        range(1, len(events) + 1),
    )

    cap._request_discontinuity(CaptureGapReason.STREAM_REOPENED)
    marker = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert marker.reason is CaptureGapReason.STREAM_REOPENED
    reopened = [
        event
        for event in events
        if event.reason is CaptureGapReason.STREAM_REOPENED
    ][0]
    assert reopened.extent is GapExtent.UNKNOWN_SOURCE_DURATION
    assert reopened.stream_generation is None
    assert reopened.start_sample is None
    assert reopened.sample_count is None


def test_paused_blocks_log_one_continuity_run_closed_by_resume(caplog):
    """A paused stretch is one telemetry run, not one INFO line per block.

    Seeded defect: drop the key comparison in _emit_capture_event so every
    event starts a run -> 300 "run started" lines -> red.
    """
    loop = MagicMock()
    events = []
    cap = LoopbackCapture(
        AudioConfig(sample_rate=16_000),
        loop,
        MagicMock(),
        on_capture_event=events.append,
        pause_state_probe=lambda: PauseState(epoch=1, paused=True),
    )
    callback = _open_callback(cap, rate=16_000)
    source = _pcm(480)

    with caplog.at_level(logging.INFO, logger="yamabiko.capture.audio"):
        for _ in range(300):
            callback(source.tobytes(), source.size, None, 0)
        for call in loop.call_soon_threadsafe.call_args_list:
            call.args[0](*call.args[1:])
        cap.apply_pause_edge(PauseState(epoch=2, paused=False))

    messages = [record.getMessage() for record in caplog.records]
    started = [
        message
        for message in messages
        if message.startswith("capture continuity run started")
    ]
    ended = [
        message
        for message in messages
        if message.startswith("capture continuity run ended")
    ]
    assert started == [
        "capture continuity run started event_kind=paused_discard "
        "reason=user_pause stream_generation=0 pause_epoch=1 start_sample=0",
    ]
    assert len(ended) == 1
    assert "events=300 sample_count=144000 " in ended[0]
    assert len(events) == 300


class _ContentModel:
    def __init__(self) -> None:
        self.resets = 0

    def __call__(self, frame: np.ndarray) -> float:
        return 0.9 if frame[0] > 0.4 else 0.01

    def reset_states(self) -> None:
        self.resets += 1


async def _emit_one_segment(payloads: list[object]):
    in_queue: asyncio.Queue = asyncio.Queue()
    out_queue: asyncio.Queue = asyncio.Queue()
    chunker = SileroChunker(
        ChunkerConfig(
            min_chunk_ms=64,
            max_chunk_ms=320,
            min_silence_ms=64,
            speech_pad_ms=0,
        ),
        threshold=0.5,
        model=_ContentModel(),
    )
    task = asyncio.create_task(chunker.run(in_queue, out_queue))
    try:
        for payload in payloads:
            await in_queue.put(payload)
        return await asyncio.wait_for(out_queue.get(), timeout=1.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize(
    "boundary_kind",
    ["exact_gap", "generation_change", "unknown_marker_then_gap"],
)
@pytest.mark.asyncio
async def test_silero_normalizes_envelopes_without_changing_emit_metadata(
    boundary_kind: str,
):
    utterance = np.concatenate([_pcm(4 * 512, 0.5), _pcm(2 * 512)])
    prefix = CaptureBlock(_pcm(256, 0.1), 4, 0)
    if boundary_kind == "exact_gap":
        payloads = [prefix, CaptureBlock(utterance, 4, 512)]
    elif boundary_kind == "generation_change":
        payloads = [prefix, CaptureBlock(utterance, 5, 0)]
    else:
        payloads = [
            prefix,
            CaptureDiscontinuity(CaptureGapReason.INPUT_STATUS_GAP),
            CaptureBlock(utterance, 4, 512),
        ]

    control = await _emit_one_segment([utterance.copy()])
    observed = await _emit_one_segment(payloads)

    assert control.discontinuity_before is False
    assert observed.discontinuity_before is True
    assert np.array_equal(observed.audio, control.audio)
    assert (
        observed.vad_emit_reason,
        observed.vad_gate_open_after_emit,
        observed.vad_retained_frames_after_emit,
    ) == (
        control.vad_emit_reason,
        control.vad_gate_open_after_emit,
        control.vad_retained_frames_after_emit,
    )
