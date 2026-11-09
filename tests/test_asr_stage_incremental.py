"""The ASR stage's incremental emission: scheduling, ordering into
transcript_q and the switch-off path (yamabiko/main.py asr_stage,
yamabiko/asr/incremental.py)."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from yamabiko.asr.incremental import OpenWindow, StepJob, StepLane
from yamabiko.main import Segment, _assemble_sentences, asr_stage
from yamabiko.sentencing.remainder import PendingRemainderState
from yamabiko.sentencing.splitter import SentenceSplitter
from yamabiko.supervisor import ErrorTracker

_SR = 16000


class _Windows:
    """The chunker side, driven by the test: one window, opened at sample 0,
    whose decided audio is the first ``decided_samples`` of ``audio``."""

    def __init__(self, window_id, audio, *, pause_epoch=None):
        self.window_id = window_id
        self.audio = audio
        self.pause_epoch = pause_epoch
        self.decided_samples = 0
        self.open = True

    @property
    def open_window_id(self):
        return self.window_id if self.open else None

    def open_window_snapshot(self):
        if not self.open:
            return None
        return OpenWindow(
            self.window_id, 0, self.audio[: self.decided_samples].copy(),
            self.pause_epoch, False,
        )


class _ASR:
    """Scripted transcript per input length; records every input length."""

    def __init__(self, texts_by_length):
        self.texts = dict(texts_by_length)
        self.lengths: list[int] = []

    def transcribe(self, audio, sample_rate):
        self.lengths.append(len(audio))
        return SimpleNamespace(text=self.texts[len(audio)], duration_s=len(audio) / sample_rate)


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.01)


def _final(window_id, samples, **fields):
    return Segment(
        audio=np.zeros(samples, dtype=np.float32), t_chunk_emit=time.perf_counter(),
        window_id=window_id, **fields,
    )


def _job(window_id, due):
    return StepJob(window_id, due, np.zeros(due, dtype=np.float32), None, False, 0.0)


async def test_a_waiting_final_runs_before_a_due_step():
    """Window 0's final waits in chunk_q while window 1's first step is due:
    the final decodes first.

    Seeded defect: let the dispatcher take the step lane before chunk_q ->
    the 16000-sample step decode runs first (red)."""
    windows = _Windows(1, np.zeros(3 * _SR, dtype=np.float32))
    windows.decided_samples = _SR
    asr = _ASR({48000: "前の文です。", _SR: "今日は"})
    chunk_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    chunk_q.put_nowait(_final(0, 48000))
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        chunk_q, out_q, asr, pool, _SR, tracker=ErrorTracker(),
        open_windows=windows, incremental_enabled=lambda: True,
    ))
    try:
        await _until(lambda: len(asr.lengths) == 2)
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    assert asr.lengths == [48000, _SR]


def test_a_newer_step_replaces_the_pending_step_of_its_window():
    """Seeded defect: keep the older job (setdefault) -> take returns the
    job due at 16000 (red)."""
    lane = StepLane(_SR)
    lane.offer(_job(3, 16000))
    lane.offer(_job(3, 32000))
    job = lane.take(32000, 3)
    assert job.due_sample == 32000
    assert lane.steps_dropped == 1
    assert lane.take(32000, 3) is None


def test_a_step_due_more_than_one_step_ago_is_dropped():
    """Seeded defect: no stale check, or a 2 * STEP_MS bound -> the first
    job runs (red); a >= bound -> the job exactly one STEP_MS late is
    dropped (red)."""
    lane = StepLane(_SR)
    lane.offer(_job(3, 16000))
    assert lane.take(16000 + _SR + 1, 3) is None
    assert lane.steps_dropped == 1
    lane.offer(_job(3, 40000))
    assert lane.take(40000 + _SR, 3).due_sample == 40000


async def test_items_leave_in_window_order_commits_first_then_the_remainder():
    """Window 0: two step decodes agree on 今日は雨です。, which goes out early
    with window 0's pause epoch; window 0's final then sends only the rest;
    window 1's final follows.

    Seeded defect: forward the whole final text (no strip) -> the second item
    repeats 今日は雨です。 (red); stamp no epoch on early items -> pause_epoch
    None (red); put commits after the final -> the order changes (red)."""
    windows = _Windows(0, np.zeros(3 * _SR, dtype=np.float32), pause_epoch=7)
    asr = _ASR({
        _SR: "今日は雨です。明日は",
        _SR + _SR // 4: "今日は雨です。明日は晴れ",
        48000: "今日は雨です。明日は晴れるでしょう。",
        24000: "次の文です。",
    })
    chunk_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        chunk_q, out_q, asr, pool, _SR, tracker=ErrorTracker(),
        open_windows=windows, incremental_enabled=lambda: True,
    ))
    try:
        windows.decided_samples = _SR  # the first step is due
        await _until(lambda: len(asr.lengths) == 1)
        windows.decided_samples = _SR + _SR // 4  # the CONFIRM_MS step is due
        await _until(lambda: out_q.qsize() == 1)
        windows.open = False  # window 0 closes
        chunk_q.put_nowait(_final(0, 48000, pause_epoch=7))
        chunk_q.put_nowait(_final(1, 24000, pause_epoch=7))
        await _until(lambda: out_q.qsize() == 3)
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    items = [out_q.get_nowait() for _ in range(3)]
    assert [
        (s.jp, s.window_id, s.early, s.follows_early, s.pause_epoch) for s in items
    ] == [
        ("今日は雨です。", 0, True, False, 7),
        ("明日は晴れるでしょう。", 0, False, True, 7),
        ("次の文です。", 1, False, False, 7),
    ]


def test_a_final_remainder_after_early_commits_is_held_and_joins_the_next_line():
    """Window 0's sentences went out early and its final remainder 明日は is
    one unfinished sentence: it is held, as the tail of the whole window
    would have been, and the next window's text completes it.

    Seeded defect: the trailing-partial rule ignores follows_early -> 明日は
    is sent alone (red)."""
    splitter = SentenceSplitter()
    remainder_state = PendingRemainderState()
    held = _assemble_sentences(
        Segment(audio=None, t_chunk_emit=0.0, jp="明日は", window_id=0, follows_early=True),
        remainder_state, splitter, 0.0, telemetry=None, incoming_ref=None,
    )
    joined = _assemble_sentences(
        Segment(audio=None, t_chunk_emit=0.0, jp="晴れるでしょう。", window_id=1),
        remainder_state, splitter, 1.0, telemetry=None, incoming_ref=None,
    )
    assert (held.sentences, joined.sentences) == ((), ("明日は晴れるでしょう。",))


async def test_force_split_dedup_strips_only_the_windows_first_emitted_piece():
    """Window 0 was cut at max_chunk_ms and window 1's decodes repeat its tail
    とてもいい天気. Only window 1's first emitted piece, an early sentence, loses
    it; the window's later early sentence and its final remainder, which also
    open with words of that tail, go out whole.

    Seeded defect: dedup every early sentence -> いい天気だ。 becomes だ。 (red);
    dedup the final remainder -> いい天気が続く。 becomes が続く。 (red); dedup
    a step decode before the emitter keeps it -> the committed text no longer
    aligns with the final decode and the remainder repeats ですね。 (red)."""
    windows = _Windows(1, np.zeros(3 * _SR, dtype=np.float32))
    windows.decided_samples = _SR  # window 1's first step is due
    asr = _ASR({
        48000: "今日はとてもいい天気",
        _SR: "とてもいい天気ですね。いい",
        _SR + _SR // 4: "とてもいい天気ですね。いい天気だ。",
        _SR + _SR // 2: "とてもいい天気ですね。いい天気だ。いい天気が",
        40000: "とてもいい天気ですね。いい天気だ。いい天気が続く。",
    })
    chunk_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    chunk_q.put_nowait(_final(0, 48000, force_split=True))
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        chunk_q, out_q, asr, pool, _SR, tracker=ErrorTracker(),
        open_windows=windows, incremental_enabled=lambda: True,
    ))
    try:
        await _until(lambda: len(asr.lengths) == 2)  # window 0's final, then the step
        windows.decided_samples = _SR + _SR // 4  # the CONFIRM_MS step is due
        await _until(lambda: out_q.qsize() == 2)
        windows.decided_samples = _SR + _SR // 2  # the next CONFIRM_MS step
        await _until(lambda: out_q.qsize() == 3)
        windows.open = False  # window 1 closes
        chunk_q.put_nowait(_final(1, 40000))
        await _until(lambda: out_q.qsize() == 4)
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    assert [out_q.get_nowait().jp for _ in range(4)] == [
        "今日はとてもいい天気", "ですね。", "いい天気だ。", "いい天気が続く。",
    ]


def _fields(seg: Segment) -> dict:
    """Every Segment field, audio as bytes. t_asr_done (a clock reading that
    differs between any two runs) counts only as set or not."""
    import dataclasses

    out = {}
    for item in dataclasses.fields(seg):
        if item.name == "t_asr_done":
            out[item.name] = getattr(seg, item.name) is not None
            continue
        value = getattr(seg, item.name)
        out[item.name] = value.tobytes() if isinstance(value, np.ndarray) else value
    return out


async def _run_finals(open_windows, enabled, t):
    """Both runs get the same inputs, including the chunker's emit time ``t``."""
    asr = _ASR({16000: "ちょっと待ってください", 48000: "くださいって言ったのに", 32000: "今日は雨です。"})
    chunk_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    chunk_q.put_nowait(Segment(
        audio=np.zeros(16000, dtype=np.float32), t_chunk_emit=t, force_split=True, window_id=4,
    ))
    chunk_q.put_nowait(Segment(audio=np.zeros(48000, dtype=np.float32), t_chunk_emit=t, window_id=5))
    kwargs = {} if open_windows is None else {
        "open_windows": open_windows, "incremental_enabled": enabled,
    }
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        chunk_q, out_q, asr, pool, _SR, tracker=ErrorTracker(), **kwargs,
    ))
    try:
        await _until(lambda: out_q.qsize() == 2)
        await asyncio.sleep(0.3)  # room for a step decode that must not happen
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    return [_fields(out_q.get_nowait()) for _ in range(out_q.qsize())], asr.lengths


async def test_switch_off_gives_the_same_outputs_as_the_path_without_steps():
    """With asr.incremental.enabled off (AppState, as build_pipeline seeds
    it), an open window whose steps are due gets no step decode, and every
    transcript item equals the one the stage sends without incremental
    emission.

    Seeded defect: ignore incremental_enabled in due_step, or have
    AppState.incremental_enabled return True -> a 32000-sample step decode
    runs (red)."""
    from yamabiko.state import AppState

    state = AppState()
    state.set_incremental(False)
    windows = _Windows(6, np.zeros(3 * _SR, dtype=np.float32))
    windows.decided_samples = 2 * _SR
    t = time.perf_counter()
    off_items, off_lengths = await _run_finals(windows, state.incremental_enabled, t)
    plain_items, plain_lengths = await _run_finals(None, None, t)
    assert off_lengths == plain_lengths == [16000, 48000]
    assert off_items == plain_items
    assert [item["jp"] for item in plain_items] == ["ちょっと待ってください", "って言ったのに"]


async def test_a_dispatcher_failure_escalates_instead_of_ending_the_stage_silently():
    """An exception in the dispatcher loop itself (outside the supervised
    final and step paths) must escalate for a watchdog restart. Unhandled,
    it ends the ASR task and no transcript reaches translation again.

    Seeded defect: remove the try/except around asr_stage's dispatcher loop
    -> the stage task ends with the RuntimeError and nothing escalates (red).
    """
    windows = _Windows(0, np.zeros(_SR, dtype=np.float32))
    escalations: list[str] = []

    def enabled():
        raise RuntimeError("switch read failed")

    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        asyncio.Queue(), asyncio.Queue(), _ASR({}), pool, _SR,
        tracker=ErrorTracker(), on_escalate=escalations.append,
        open_windows=windows, incremental_enabled=enabled,
    ))
    try:
        await asyncio.wait_for(task, 5.0)
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    assert escalations == ["asr"]
