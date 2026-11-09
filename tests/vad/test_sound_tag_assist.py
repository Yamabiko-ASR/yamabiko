"""The sound-tag VAD assist: frame alignment, its place in front of Silero,
and what happens to held frames at discontinuities and capture stalls.

The tagger is faked here (a function of the window it is handed), so these
tests isolate the streaming layout and the chunker wiring. The shipped graph
itself is covered by ``tests/test_real_sound_tagger.py``.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from yamabiko.capture.envelope import CaptureDiscontinuity, CaptureGapReason
from yamabiko.vad import silero_chunker
from yamabiko.vad.chunker import ChunkerConfig
from yamabiko.vad.silero_chunker import SileroChunker
from yamabiko.vad.sound_tagger import WINDOW_FRAMES, SoundTagAssist

_FRAME = 512
_LOUD = 0.5


class _ContentSilero:
    """Speech probability from the frame's level; logs every call and reset
    in order, each call by the frame's first sample."""

    def __init__(self) -> None:
        self.log: list[tuple[str, float]] = []

    def __call__(self, frame: np.ndarray) -> float:
        self.log.append(("call", float(frame[0])))
        return 0.9 if float(np.abs(frame).mean()) > 0.25 else 0.01

    def reset_states(self) -> None:
        self.log.append(("reset", 0.0))

    @property
    def calls(self) -> int:
        return sum(1 for kind, _ in self.log if kind == "call")


def _silent_tagger(window: np.ndarray) -> np.ndarray:
    return np.zeros(WINDOW_FRAMES, dtype=np.float32)


def _frames(levels: list[tuple[float, float]]) -> np.ndarray:
    """Concatenated (seconds, level) runs, cut to whole 512-sample frames."""
    audio = np.concatenate([
        np.full(int(seconds * 16000), level, dtype=np.float32)
        for seconds, level in levels
    ])
    return audio[: audio.size // _FRAME * _FRAME]


def _blocks(audio: np.ndarray, size: int = 480) -> list[np.ndarray]:
    return [audio[start:start + size].copy() for start in range(0, audio.size, size)]


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.01)


def _drain_queue(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


async def test_each_frame_takes_the_tag_frame_holding_its_centre_sample():
    """Frame k's term comes from tag frame (512 k + 256) // 640 of the window
    that ends 0.24 s after that tag frame's block.

    The fake tagger reports, for every tag frame of the window it gets, the
    window sample at that tag frame's centre. Sample n of the stream holds
    (n + 1) * 1e-6, so a term names the stream sample it was read from; a
    window ending one hop early or late, a wrong kept row or a wrong
    512-to-640 mapping reads another sample.
    """
    frame_count = 190
    audio = (np.arange(frame_count * _FRAME, dtype=np.float64) + 1) * 1e-6
    audio = audio.astype(np.float32)
    assist = SoundTagAssist(lambda window: window[320::640].copy())

    tagged = []
    for ordinal in range(frame_count):
        frame = audio[ordinal * _FRAME:(ordinal + 1) * _FRAME]
        tagged.extend(await assist.push(frame, ordinal))
    tagged.extend(await assist.drain())

    assert [meta for _, _, meta in tagged] == list(range(frame_count))
    padded = np.concatenate([audio, np.zeros(1280, dtype=np.float32)])
    for frame, term, ordinal in tagged:
        assert frame[0] == audio[ordinal * _FRAME]
        centre = (ordinal * _FRAME + _FRAME // 2) // 640 * 640 + 320
        assert term == padded[centre], f"frame {ordinal}"


async def test_block_frames_leave_on_the_frame_that_completes_their_window():
    """Block b (chunker frames 15 b to 15 b + 14) is tagged by the window that
    ends 0.24 s (3,840 samples) after the block, at sample 7,680 b + 11,520,
    halfway through frame 15 b + 22. All 15 frames leave on that frame's
    push: waiting for the next hop boundary or one frame more would delay
    every gate decision, and every line, by up to 0.24 s.
    """
    frame_count = 15 * 6
    assist = SoundTagAssist(_silent_tagger)
    released_on: dict[int, int] = {}
    for ordinal in range(frame_count):
        for _, _, meta in await assist.push(np.zeros(_FRAME, dtype=np.float32), ordinal):
            released_on[meta] = ordinal

    expected = {
        k: 15 * (k // 15) + 22 for k in range(frame_count)
        if 15 * (k // 15) + 22 < frame_count
    }
    assert released_on == expected


async def _run_chunker(chunker: SileroChunker, items: list, until) -> list:
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(chunker.run(in_q, out_q))
    try:
        for item in items:
            in_q.put_nowait(item)
        await _until(lambda: until() or task.done())
        if task.done():
            task.result()
        return _drain_queue(out_q)
    finally:
        task.cancel()


async def test_silent_tagger_keeps_segments_and_silero_calls_unchanged():
    """With a tagger that hears nothing, the assisted chunker emits exactly
    what the Silero-only chunker emits and drives Silero through the same
    calls and resets in the same order.

    Silero must score a frame only when it leaves the delay line: scored on
    arrival, the resets at window closes would land up to 22 frames late in
    Silero's call sequence. Losing or reordering a held frame changes the
    segments.
    """
    runs = [(1.2, _LOUD), (0.8, 0.0), (0.5, _LOUD), (2.0, 0.0), (9.0, _LOUD), (1.0, 0.0)]
    audio = _frames(runs)
    # A distinct first sample per frame makes the call log order-sensitive.
    ordinals = np.arange(audio.size // _FRAME, dtype=np.float32) * 1e-5
    audio[::_FRAME] += ordinals
    items = [*_blocks(audio), CaptureDiscontinuity(reason=CaptureGapReason.INPUT_STATUS_GAP)]
    frame_count = audio.size // _FRAME
    cfg = ChunkerConfig()

    results = []
    for tagger in (None, _silent_tagger):
        model = _ContentSilero()
        chunker = SileroChunker(cfg, threshold=0.3, model=model, tagger=tagger)
        segments = await _run_chunker(
            chunker, items,
            lambda: model.calls == frame_count and sum(chunker.discontinuity_resets.values()) == 1,
        )
        results.append((model.log, segments))

    (plain_log, plain_segments), (assisted_log, assisted_segments) = results
    assert len(plain_segments) >= 3
    assert assisted_log == plain_log
    assert len(assisted_segments) == len(plain_segments)
    for plain, assisted in zip(plain_segments, assisted_segments):
        assert np.array_equal(assisted.audio, plain.audio)
        assert assisted.vad_emit_reason == plain.vad_emit_reason
        assert assisted.force_split == plain.force_split
        assert assisted.discontinuity_before == plain.discontinuity_before


async def test_discontinuity_emits_held_line_then_starts_tagger_from_silence():
    """At a discontinuity the frames still held in the delay line reach the
    gate first, so a line that closed within the last 0.70 s still emits;
    then the tagger starts over with zeros, so no window after the gap holds
    audio from before it.
    """
    line = _frames([(1.5, _LOUD), (0.7, 0.0)])
    after_gap = np.full(3 * 16000, 0.001, dtype=np.float32)
    windows: list[np.ndarray] = []

    def recording_tagger(window: np.ndarray) -> np.ndarray:
        windows.append(window.copy())
        return np.zeros(WINDOW_FRAMES, dtype=np.float32)

    chunker = SileroChunker(
        ChunkerConfig(), threshold=0.3, model=_ContentSilero(), tagger=recording_tagger,
    )
    items = [
        *_blocks(line),
        CaptureDiscontinuity(reason=CaptureGapReason.INPUT_STATUS_GAP),
        *_blocks(after_gap),
    ]
    segments = await _run_chunker(
        chunker, items, lambda: any(np.any(w == np.float32(0.001)) for w in windows[-1:]),
    )

    assert len(segments) == 1
    assert np.count_nonzero(segments[0].audio == np.float32(_LOUD)) == int(1.5 * 16000)
    mixed = [
        index for index, window in enumerate(windows)
        if np.any(window == np.float32(_LOUD)) and np.any(window == np.float32(0.001))
    ]
    assert mixed == [], f"windows {mixed} hold audio from both sides of the gap"


async def test_capture_stall_releases_held_frames_to_the_gate(monkeypatch):
    """When capture stops delivering blocks (WASAPI loopback sends none while
    the endpoint is silent), held frames reach the gate after STALL_DRAIN_S
    instead of waiting for the next audio."""
    monkeypatch.setattr(silero_chunker, "STALL_DRAIN_S", 0.2)
    line = _frames([(1.5, _LOUD), (0.7, 0.0)])
    chunker = SileroChunker(
        ChunkerConfig(), threshold=0.3, model=_ContentSilero(), tagger=_silent_tagger,
    )
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(chunker.run(in_q, out_q))
    try:
        for block in _blocks(line):
            in_q.put_nowait(block)
        segment = await asyncio.wait_for(out_q.get(), timeout=2.0)
    finally:
        task.cancel()
    assert np.count_nonzero(segment.audio == np.float32(_LOUD)) == int(1.5 * 16000)


async def test_tagger_inference_does_not_block_the_event_loop():
    """Inference runs in a worker thread: a 0.25 s tagger call never stalls
    other coroutines on the loop."""

    def slow_tagger(window: np.ndarray) -> np.ndarray:
        time.sleep(0.25)
        return np.zeros(WINDOW_FRAMES, dtype=np.float32)

    assist = SoundTagAssist(slow_tagger)
    ticks = [time.perf_counter()]
    done = asyncio.Event()

    async def ticker() -> None:
        while not done.is_set():
            await asyncio.sleep(0.01)
            ticks.append(time.perf_counter())

    ticking = asyncio.create_task(ticker())
    audio = _frames([(3.0, _LOUD)])
    tagged = []
    for ordinal in range(audio.size // _FRAME):
        tagged.extend(await assist.push(audio[ordinal * _FRAME:(ordinal + 1) * _FRAME], ordinal))
    ticks.append(time.perf_counter())
    done.set()
    await ticking

    assert len(tagged) > 0
    stall = max(np.diff(ticks))
    assert stall < 0.15, f"event loop stalled for {stall:.3f} s"
