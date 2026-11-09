"""Force-flush when partial_remainder exceeds 300 chars.

The design contract (docs/design/sentence-boundaries.md): if a trailing
partial sentence grows past `_PARTIAL_REMAINDER_CAP`, the translation
stage must force-flush it instead of buffering indefinitely. The forced
flush prefixes the flushed text with "[...]" so downstream viewers know
the sentence was cut.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import pytest

from yamabiko.main import (
    Segment,
    LatencyStats,
    translation_stage,
    _PARTIAL_REMAINDER_CAP,
)
from yamabiko.state import AppState
from yamabiko.supervisor import ErrorTracker
from yamabiko.translation.prompt import ContextBuffer


class _RecordingTranslator:
    def __init__(self):
        self.calls: list[str] = []

    async def translate(self, jp, genre, context):
        self.calls.append(jp)
        return "EN:" + jp[:10]


class _RecordingWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload):
        self.sent.append(payload)


def test_partial_cap_constant_exists():
    """Sanity: the cap constant is defined where the integration expects it."""
    from yamabiko import main as _m
    assert hasattr(_m, "_PARTIAL_REMAINDER_CAP")
    assert _m._PARTIAL_REMAINDER_CAP == 300


@pytest.mark.asyncio
async def test_400_char_partial_force_flushes_with_ellipsis_prefix():
    """A ~400-char open-ended JP block (partial then terminator) must:
      1. flush the accumulated partial once it exceeds the cap, with a "[...]" prefix
      2. follow up with the final sentence when it terminates
      3. leave pending_remainder empty after flush
    """
    from yamabiko.display.broadcast import ConnectionManager

    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _RecordingTranslator()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)

    # 400 chars of JP with one terminator followed by a long partial.
    # Structure: "完結文。" + 400 chars (no terminator) — when translation
    # stage splits this, sentences[-1] is the huge partial and sentences
    # has >1 elements → cap path triggers.
    long_partial = "あ" * (_PARTIAL_REMAINDER_CAP + 50)
    jp_text = "最初の文です。" + long_partial
    audio = np.zeros(1, dtype=np.float32)
    await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(), jp=jp_text))

    task = asyncio.create_task(
        translation_stage(
            q, translator, ctx, state, stats, None,
            max_segments=2, done=done, manager=manager,
            tracker=ErrorTracker(),
        )
    )
    try:
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        task.cancel()

    # Two broadcasts — the complete first sentence, then the force-flushed
    # partial with "[...]" prefix.
    assert len(ws.sent) == 2
    flushed_jp = ws.sent[1]["jp"]
    assert flushed_jp.startswith("[...]"), f"expected force-flushed partial to be prefixed with [...]; got {flushed_jp[:30]!r}"
    # And the translator was called for both sentences.
    assert len(translator.calls) == 2
    assert translator.calls[1].startswith("[...]")


@pytest.mark.asyncio
async def test_partial_below_cap_still_buffers():
    """A partial under the cap must be buffered, not force-flushed."""
    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _RecordingTranslator()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()

    # Complete + short partial: "完結。" + 50 char tail
    jp_text = "完結です。" + "あ" * 50
    audio = np.zeros(1, dtype=np.float32)
    await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(), jp=jp_text))

    task = asyncio.create_task(
        translation_stage(
            q, translator, ctx, state, stats, None,
            max_segments=1, done=done, manager=None,
            tracker=ErrorTracker(),
        )
    )
    try:
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        task.cancel()

    # Only one translation call: the partial tail was buffered, not force-flushed.
    assert len(translator.calls) == 1
    assert translator.calls[0] == "完結です。"
    # And no call contains the [...] prefix.
    assert not any(c.startswith("[...]") for c in translator.calls)


@pytest.mark.asyncio
async def test_buffered_partial_survives_non_ja_next_chunk():
    """A buffered JP partial must NOT be discarded when the NEXT chunk is a
    non-JA ASR hallucination -- it must be recovered by the following JP chunk.

    Seeded defect: clear pending_remainder BEFORE the is_japanese guard
    -> 'それで' is lost -> no translator call contains it.
    """
    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _RecordingTranslator()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    audio = np.zeros(1, dtype=np.float32)

    # 1) multi-sentence chunk -> buffers the trailing partial "それで"
    await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(), jp="最初の文です。それで"))
    # 2) English ASR hallucination -> the combined text trips the non-JA guard
    await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(), jp="I will let him know."))
    # 3) JP chunk -> must recover the buffered "それで"
    await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(), jp="終わりです。"))

    task = asyncio.create_task(
        translation_stage(
            q, translator, ctx, state, stats, None,
            max_segments=2, done=done, manager=None,
            tracker=ErrorTracker(),
        )
    )
    try:
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        task.cancel()

    assert any("それで" in c for c in translator.calls), (
        f"buffered JP partial 'それで' was silently dropped after a non-JA "
        f"chunk; translator calls were {translator.calls}"
    )
    # The English hallucination itself produced no translation.
    assert not any("will let" in c for c in translator.calls)


# --- pending_remainder age bound (P2, 2026-07-12 review) --------------------
# A buffered unterminated JP tail deliberately survives dropped non-JA
# chunks (test above), which also meant it survived FOREVER: a tail like
# 「でも」 could be glued onto the next JP chunk minutes or hours later,
# stitching a subtitle across unrelated dialogue. translation_stage now
# stamps the remainder with the module's _monotonic clock seam and
# discards it at consumption time once older than
# _PENDING_REMAINDER_MAX_AGE_S. The clock is faked by monkeypatching
# yamabiko.main._monotonic -- patching time.monotonic itself would
# corrupt the event loop's own timer.


async def _drive_remainder_age(monkeypatch, *, advance_s: float):
    """Buffer a partial ('それで'), advance the fake clock by advance_s,
    then feed the next JP chunk; return the recorded translate calls."""
    from yamabiko import main as _m

    clock = {"t": 5000.0}
    monkeypatch.setattr(_m, "_monotonic", lambda: clock["t"])

    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _RecordingTranslator()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    audio = np.zeros(1, dtype=np.float32)

    task = asyncio.create_task(
        translation_stage(
            q, translator, ctx, state, stats, None,
            max_segments=2, done=done, manager=None,
            tracker=ErrorTracker(),
        )
    )
    try:
        # 1) multi-sentence chunk -> buffers the trailing partial "それで"
        #    (stamped at t=5000).
        await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(),
                            jp="最初の文です。それで"))
        # The remainder is stamped BEFORE the sentence's translate call, so
        # once the first call is visible the buffer timestamp is in place.
        for _ in range(300):
            await asyncio.sleep(0.01)
            if translator.calls:
                break
        clock["t"] += advance_s
        # 2) the next JP chunk -- glue target / stale-discard subject.
        await q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter(),
                            jp="終わりです。"))
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    return translator.calls


@pytest.mark.asyncio
async def test_fresh_pending_remainder_still_glues_within_age_bound(monkeypatch):
    """A remainder younger than the bound keeps the existing glue behavior
    (this is the pre-fix contract; it must not regress)."""
    from yamabiko import main as _m

    calls = await _drive_remainder_age(
        monkeypatch, advance_s=_m._PENDING_REMAINDER_MAX_AGE_S - 1.0,
    )
    assert calls[0] == "最初の文です。"
    assert calls[1] == "それで終わりです。", (
        f"a FRESH buffered partial must still glue onto the next JP chunk; "
        f"got {calls!r}"
    )


@pytest.mark.asyncio
async def test_stale_pending_remainder_is_discarded_not_glued(monkeypatch, caplog):
    """A remainder OLDER than _PENDING_REMAINDER_MAX_AGE_S is discarded at
    consumption time (with a debug log) instead of being glued onto a chunk
    from a conversation that has long moved on.

    Seeded defect: drop the age check before `raw = pending_remainder +
    seg.jp` in translation_stage -> the 31-s-old 'それで' glues -> red.
    """
    from yamabiko import main as _m

    caplog.set_level(logging.DEBUG, logger="yamabiko")
    calls = await _drive_remainder_age(
        monkeypatch, advance_s=_m._PENDING_REMAINDER_MAX_AGE_S + 1.0,
    )
    assert calls == ["最初の文です。", "終わりです。"], (
        f"a stale buffered partial was glued onto a chunk arriving past the "
        f"age bound; got {calls!r}"
    )
    assert any(
        "stale" in r.getMessage() and "remainder" in r.getMessage()
        for r in caplog.records
    ), "no debug log recorded the stale-remainder discard"


@pytest.mark.asyncio
async def test_build_ruby_runs_off_the_event_loop():
    """Furigana segmentation (build_ruby -> fugashi readings) must run off
    the event-loop thread so the hot path doesn't block on it.

    Seeded defect: drop the asyncio.to_thread wrapper around build_ruby
    in translation_stage -> it runs on the loop thread -> ident == loop ident
    -> this fails.
    """
    import threading

    loop_ident = threading.get_ident()

    class _Glossary:
        def current_genre(self):
            return "variety"

        def set_genre(self, g):
            pass

    class _RubyPipeline:
        def __init__(self):
            self.ident = None
            self._glossary = _Glossary()
            self._series_id = None
            self._applied_epoch = 0

        def build_ruby(self, jp):
            self.ident = threading.get_ident()
            return []

        # translation_stage's series-reload branch runs before build_ruby and
        # reads applied_series_epoch() each sentence; with state.series_epoch
        # and this both 0 (no set_series called) the branch is a no-op,
        # isolating the build_ruby off-loop property under test.
        def current_series_id(self):
            return self._series_id

        def applied_series_epoch(self):
            return self._applied_epoch

        def set_gazetteer(self, gazetteer, media_id, epoch=0):
            self._series_id = media_id
            self._applied_epoch = epoch

    pipe = _RubyPipeline()
    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _RecordingTranslator()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    await q.put(Segment(audio=np.zeros(1, dtype=np.float32),
                        t_chunk_emit=time.perf_counter(), jp="日本語です。"))

    task = asyncio.create_task(
        translation_stage(
            q, translator, ctx, state, stats, None,
            max_segments=1, done=done, manager=None,
            pipeline=pipe, tracker=ErrorTracker(),
        )
    )
    try:
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        task.cancel()

    assert pipe.ident not in (None, loop_ident), (
        "build_ruby ran on the event-loop thread"
    )
