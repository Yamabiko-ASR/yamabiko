"""Pipeline wiring smoke tests for main.py.

Scope: ensure queue wiring + stage coroutines connect end-to-end and the
supervisor escalation hook fires on a crashing translator. Full integration
with Silero + faster-whisper is out of scope here (heavy models); we mock at
the ASR / translator boundary and drive the stages directly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest

from yamabiko.display.broadcast import ConnectionManager
from yamabiko.main import (
    Segment,
    LatencyStats,
    asr_stage,
    translation_stage,
)
from yamabiko.state import AppState
from yamabiko.supervisor import ErrorTracker
from yamabiko.translation.prompt import ContextBuffer
from yamabiko.post_asr_telemetry import Fate as PostASRFate, PostASRTelemetry


class _FakeASR:
    def __init__(self, text: str):
        self._text = text

    def transcribe(self, audio, sample_rate):
        return SimpleNamespace(text=self._text, duration_s=len(audio) / sample_rate)


class _SeqASR:
    """Returns one scripted transcript per transcribe() call."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)

    def transcribe(self, audio, sample_rate):
        return SimpleNamespace(text=self._texts.pop(0), duration_s=1.0)


class _FakeTranslator:
    def __init__(self, en: str = "hello", raise_exc: Exception | None = None):
        self._en = en
        self._raise = raise_exc
        self.calls = 0

    async def translate(self, jp, genre, context):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._en


class _RecordingWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_pipeline_happy_path_broadcasts_subtitle(caplog):
    """Feed one canned Segment through asr+translation; assert ConnectionManager fans out a subtitle."""
    from concurrent.futures import ThreadPoolExecutor

    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()

    audio = np.zeros(16000, dtype=np.float32)
    await chunk_q.put(Segment(audio=audio, t_chunk_emit=time.perf_counter()))

    pool = ThreadPoolExecutor(max_workers=1)
    asr = _FakeASR(text="こんにちは。")
    translator = _FakeTranslator(en="Hello.")
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)
    telemetry = PostASRTelemetry(run_id="happy-path-run")
    caplog.set_level(logging.INFO, logger="yamabiko")

    asr_task = asyncio.create_task(
        asr_stage(
            chunk_q, transcript_q, asr, pool, 16000,
            tracker=ErrorTracker(), post_asr_telemetry=telemetry,
        )
    )
    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=1, done=done, manager=manager,
            tracker=ErrorTracker(),
            post_asr_telemetry=telemetry,
        )
    )
    try:
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        asr_task.cancel()
        tr_task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    assert len(ws.sent) == 1
    msg = ws.sent[0]
    assert msg["type"] == "subtitle"
    assert msg["jp"] == "こんにちは。"
    assert msg["en"] == "Hello."

    asr_windows = [
        record.getMessage() for record in caplog.records
        if record.name == "yamabiko"
        and record.getMessage().startswith("asr window:")
    ]
    assert len(asr_windows) == 1
    assert "run_id=happy-path-run window_seq=0" in asr_windows[0]

    telemetry_messages = [
        record.getMessage() for record in caplog.records
        if record.name == "yamabiko"
        and record.getMessage().startswith("post_asr ")
    ]
    assert (
        "post_asr root: run_id=happy-path-run artifact_seq=0 "
        "kind=asr_window window_seq=0"
    ) in telemetry_messages
    assert (
        "post_asr edge: run_id=happy-path-run artifact_seq=1 "
        "parent_artifact_seq=0 kind=sentence ordinal=0"
    ) in telemetry_messages
    assert any(
        "artifact_seq=1 fate=display_send_completed" in message
        and "attempted=1 accepted=1 failed=0" in message
        for message in telemetry_messages
    )
    assert all(
        "こんにちは。" not in message and "Hello." not in message
        for message in telemetry_messages
    )


class _FakeLatencyLog:
    """Captures latency.log lines so transcript-gating is observable."""

    def __init__(self):
        self.lines: list[str] = []

    def write(self, s: str) -> None:
        self.lines.append(s)

    def flush(self) -> None:
        pass


async def _drive_one_segment(*, transcript_enabled: bool, jp: str, en: str):
    """Run one JP segment through translation_stage with a capturing latency
    log; return the parsed latency.log records."""
    import json as _json

    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    latency_log = _FakeLatencyLog()
    translator = _FakeTranslator(en=en)

    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, latency_log,
            max_segments=1, done=done,
            tracker=ErrorTracker(),
            transcript_enabled=transcript_enabled,
        )
    )
    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp=jp,
    ))
    try:
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        tr_task.cancel()
        try:
            await asyncio.wait_for(tr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    return [_json.loads(line) for line in latency_log.lines]


@pytest.mark.asyncio
async def test_transcript_logging_on_records_raw_jp_and_en():
    """logging.transcript_log_enabled=True -> the latency.log record for a
    segment carries the RAW jp (ASR text) and en (translation), so an accuracy
    report is diagnosable as an ASR mishearing (jp wrong) vs a translation
    error (jp right, en wrong) from the log alone.

    Seeded defect: drop the `if transcript_enabled:` jp/en assignment in
    translation_stage -> the keys are absent -> red.
    """
    records = await _drive_one_segment(
        transcript_enabled=True, jp="悪い。", en="It is bad.",
    )
    assert records, "expected one latency record for the segment"
    rec = records[0]
    assert rec["jp"] == "悪い。", f"raw ASR text must be logged; got {rec!r}"
    assert rec["en"] == "It is bad.", f"raw translation must be logged; got {rec!r}"


@pytest.mark.asyncio
async def test_transcript_logging_off_omits_text_keeps_lengths():
    """Default (transcript_log_enabled=False) -> latency.log keeps jp_len/en_len
    but NOT the raw text, so the privacy-safe default never writes user speech
    transcripts to disk.

    Seeded defect: make translation_stage always write jp/en (ignore the flag)
    -> the absence asserts fail -> red.
    """
    records = await _drive_one_segment(
        transcript_enabled=False, jp="悪い。", en="It is bad.",
    )
    assert records, "expected one latency record for the segment"
    rec = records[0]
    assert "jp" not in rec and "en" not in rec, (
        f"transcript text must be omitted when the flag is off; got {rec!r}"
    )
    assert "jp_len" in rec and "en_len" in rec, (
        f"length metrics must still be present; got {rec!r}"
    )


class _RaisingLatencyLog:
    def write(self, _text: str) -> int:
        raise OSError(28, "disk full")

    def flush(self) -> None:
        raise AssertionError("flush must not precede subtitle delivery")


@pytest.mark.asyncio
async def test_translation_stage_broadcasts_when_latency_sink_raises():
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)
    task = asyncio.create_task(
        translation_stage(
            transcript_q,
            _FakeTranslator(en="hello"),
            ContextBuffer(max_lines=2),
            AppState(genre="variety"),
            LatencyStats(),
            _RaisingLatencyLog(),
            max_segments=1,
            done=done,
            manager=manager,
            tracker=ErrorTracker(),
        )
    )
    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp="こんにちは。",
    ))
    try:
        for _ in range(100):
            if ws.sent:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    assert ws.sent, "subtitle was dropped when latency output raised"
    assert ws.sent[-1]["type"] == "subtitle"
    assert ws.sent[-1]["en"] == "hello"


# asr_stage must run the transcript dedup ONLY across a force-split
# boundary (max_chunk_ms cut mid-speech -- the one case that genuinely
# re-emits the prior tail). Across a silence-terminated boundary the
# chunks are unrelated utterances, and a coincidental suffix/prefix
# match (aizuchi echoes: そうですね…) must not strip the new transcript.


async def _run_asr_stage(segments: list[Segment], texts: list[str]) -> list[Segment]:
    """Drive asr_stage with scripted transcripts; return emitted segments."""
    from concurrent.futures import ThreadPoolExecutor

    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    for seg in segments:
        await chunk_q.put(seg)
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(
        asr_stage(chunk_q, transcript_q, _SeqASR(texts), pool, 16000,
                  tracker=ErrorTracker())
    )
    out: list[Segment] = []
    try:
        await asyncio.wait_for(_drain(transcript_q, chunk_q), timeout=3.0)
        while not transcript_q.empty():
            out.append(transcript_q.get_nowait())
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    return out


async def _drain(transcript_q: asyncio.Queue, chunk_q: asyncio.Queue) -> None:
    while not chunk_q.empty():
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)  # let the in-flight segment finish


@pytest.mark.asyncio
async def test_asr_stage_dedups_reemitted_tail_after_force_split_chunk():
    """Seeded defect: drop the prev-chunk-was-force-split gate (or the
    dedup call itself) in asr_stage -> the second jp keeps/loses the
    re-emitted tail and this assertion goes red."""
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()
    segs = [
        Segment(audio=audio, t_chunk_emit=t, force_split=True),
        Segment(audio=audio, t_chunk_emit=t),
    ]
    out = await _run_asr_stage(segs, ["ちょっと待ってください", "くださいって言ったのに"])
    assert [s.jp for s in out] == ["ちょっと待ってください", "って言ったのに"]


@pytest.mark.asyncio
async def test_asr_stage_keeps_echo_after_silence_terminated_chunk():
    """The aizuchi-echo case: prev chunk ended at a SILENCE boundary, so
    even a long coincidental overlap must not be stripped. Seeded defect:
    run the dedup unconditionally in asr_stage."""
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()
    segs = [
        Segment(audio=audio, t_chunk_emit=t, force_split=False),
        Segment(audio=audio, t_chunk_emit=t),
    ]
    out = await _run_asr_stage(segs, ["そうですね", "そうですね、たしかに"])
    assert [s.jp for s in out] == ["そうですね", "そうですね、たしかに"]


@pytest.mark.asyncio
async def test_asr_stage_skips_force_split_dedup_across_capture_gap():
    """A capture discontinuity breaks force-split transcript adjacency."""
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()
    segs = [
        Segment(audio=audio, t_chunk_emit=t, force_split=True),
        Segment(audio=audio, t_chunk_emit=t, discontinuity_before=True),
    ]

    out = await _run_asr_stage(
        segs,
        ["ちょっと待ってください", "くださいって言ったのに"],
    )

    assert [s.jp for s in out] == [
        "ちょっと待ってください", "くださいって言ったのに",
    ]


@pytest.mark.asyncio
async def test_asr_stage_skips_dedup_after_force_split_empty_transcript():
    """An empty ASR result breaks transcript adjacency even when its audio
    chunk was force-split, so the following chunk must not dedup against the
    last non-empty transcript."""
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()
    segs = [
        Segment(audio=audio, t_chunk_emit=t),
        Segment(audio=audio, t_chunk_emit=t, force_split=True),
        Segment(audio=audio, t_chunk_emit=t),
    ]
    out = await _run_asr_stage(
        segs,
        ["ちょっと待ってください", "", "くださいって言ったのに"],
    )
    assert [s.jp for s in out] == [
        "ちょっと待ってください", "くださいって言ったのに",
    ]


@pytest.mark.asyncio
async def test_asr_stage_logs_one_private_safe_outcome_for_typed_and_legacy_results(
    caplog, monkeypatch,
):
    from contextlib import suppress
    from concurrent.futures import ThreadPoolExecutor

    import yamabiko.main as main_module
    from yamabiko.asr.asr_client import ASROutcome, ASRResult

    class _ResultsASR:
        def __init__(self):
            self.results = [
                ASRResult("秘密の台詞", 1.0, ASROutcome.TEXT_AVAILABLE),
                SimpleNamespace(text="", duration_s=1.0),
            ]

        def transcribe(self, audio, sample_rate):
            return self.results.pop(0)

    audio = np.zeros(16000, dtype=np.float32)
    now = time.perf_counter()
    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    await chunk_q.put(Segment(
        audio=audio,
        t_chunk_emit=now,
        force_split=True,
        discontinuity_before=True,
        vad_emit_reason="endpoint_force_split",
    ))
    await chunk_q.put(Segment(
        audio=audio,
        t_chunk_emit=now,
        vad_emit_reason="SENSITIVE legacy endpoint value",
        vad_retained_frames_after_emit="SENSITIVE retained value",
    ))
    clock = iter((10.0, 10.125, 20.0, 20.250))
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: next(clock))
    pool = ThreadPoolExecutor(max_workers=1)
    caplog.set_level(logging.INFO, logger="yamabiko")
    task = asyncio.create_task(
        asr_stage(chunk_q, transcript_q, _ResultsASR(), pool, 16000,
                  tracker=ErrorTracker())
    )
    try:
        await asyncio.wait_for(_drain(transcript_q, chunk_q), timeout=3.0)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        pool.shutdown(wait=False, cancel_futures=True)

    messages = [record.getMessage() for record in caplog.records
                if record.getMessage().startswith("asr window:")]
    assert len(messages) == 2
    assert "outcome=text_available" in messages[0]
    assert "input_ms=1000" in messages[0]
    assert "stage_ms=125" in messages[0]
    assert "force_split=true" in messages[0]
    assert "vad_emit_reason=endpoint_force_split" in messages[0]
    assert "gate_open_after=false" in messages[0]
    assert "retained_frames_after=0" in messages[0]
    assert "discontinuity_before=true" in messages[0]
    assert "outcome=untyped_backend_result" in messages[1]
    assert "stage_ms=250" in messages[1]
    assert "force_split=false" in messages[1]
    assert "vad_emit_reason=unavailable" in messages[1]
    assert "retained_frames_after=-1" in messages[1]
    assert "discontinuity_before=false" in messages[1]
    assert all("SENSITIVE legacy endpoint value" not in message
               for message in messages)
    assert all("SENSITIVE retained value" not in message for message in messages)
    assert all("stage_ms=" in message for message in messages)
    assert all("秘密の台詞" not in message for message in messages)


# --- P1-8a: a wedged (never-returning) asr.transcribe must escalate ---------
# asr_stage awaits the transcribe executor call; a CUDA/CT2 hang used to
# leave that await pending FOREVER with /livez green (the event loop is
# fine) and the supervisor blind (a hung thread raises nothing, so the
# 10-in-60s tracker can never fire). And because cuda_pool is single-
# worker, every queued transcribe is poisoned behind the hung one, so
# in-process recovery is impossible: the only correct response is the
# SAME typed exit-45 escalation the stage-error tracker uses, promptly.


class _EscalationTarget:
    """Stub carrying exactly the state PipelineApp.on_stage_escalate
    mutates (the typed exit-45 request + the done event), so tests invoke
    the REAL escalation body and assert on its observable side effects
    rather than on a spy's call count."""

    def __init__(self):
        self.stage_escalation_requested = False
        self.done = asyncio.Event()

    def escalate(self, stage: str) -> None:
        from yamabiko.main import PipelineApp
        PipelineApp.on_stage_escalate(self, stage)


@pytest.mark.asyncio
async def test_asr_stage_escalates_wedged_transcribe_to_exit45_path(caplog):
    """A transcribe that never returns must, after the injected timeout,
    invoke the stage-escalation callback directly -- observable as the
    typed exit-45 request (stage_escalation_requested) plus done set on
    the escalation target, and a CRITICAL log naming the asr stage.

    Seeded defect: drop the asyncio.wait_for bound around the transcribe
    executor await in asr_stage -> the stage blocks forever on the wedge,
    done never fires -> the wait_for below times out -> red.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release = threading.Event()

    class _WedgedASR:
        def transcribe(self, audio, sample_rate):
            release.wait(timeout=30)  # the wedge: blocks until released
            return SimpleNamespace(text="", duration_s=0.0)

    target = _EscalationTarget()
    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    await chunk_q.put(Segment(
        audio=np.zeros(4, dtype=np.float32), t_chunk_emit=time.perf_counter(),
    ))
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda-test")
    caplog.set_level(logging.CRITICAL, logger="yamabiko")
    task = asyncio.create_task(asr_stage(
        chunk_q, transcript_q, _WedgedASR(), pool, 16000,
        tracker=ErrorTracker(),
        on_escalate=target.escalate,
        transcribe_timeout_s=0.1,
    ))
    try:
        await asyncio.wait_for(target.done.wait(), timeout=3.0)
    finally:
        task.cancel()
        release.set()  # unblock the abandoned worker so pool shutdown is instant
        pool.shutdown(wait=False, cancel_futures=True)

    assert target.stage_escalation_requested is True, (
        "wedged transcribe timed out but the typed exit-45 request "
        "(stage_escalation_requested) was never armed"
    )
    crit = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert any("transcribe" in r.getMessage() for r in crit), (
        f"no CRITICAL log naming the wedged transcribe; got {crit!r}"
    )


@pytest.mark.asyncio
async def test_asr_stage_quick_transcribe_timeout_error_is_not_a_wedge():
    """asr.transcribe RAISING TimeoutError promptly (a backend-internal
    timeout that returned) is a finished call, not a hung thread: the
    executor future is DONE, nothing poisons the cuda_pool, and escalating
    would restart the whole pipeline over a one-chunk error. It must be
    treated like any other transcribe exception (counted by the tracker,
    stage keeps consuming) -- same wedge-vs-quick-error discrimination as
    _stop_capture_bounded (on 3.11+ asyncio.TimeoutError IS the builtin).

    Seeded defect: treat every TimeoutError from the transcribe await as a
    wedge (skip the fut.cancelled() check) -> the quick error escalates ->
    stage_escalation_requested True -> red.
    """
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()

    class _QuickTimeoutThenOkASR:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("backend device query timed out (returned)")
            return SimpleNamespace(text="こんにちは", duration_s=1.0)

    from concurrent.futures import ThreadPoolExecutor

    target = _EscalationTarget()
    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    await chunk_q.put(Segment(audio=audio, t_chunk_emit=t))
    await chunk_q.put(Segment(audio=audio, t_chunk_emit=t))
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        chunk_q, transcript_q, _QuickTimeoutThenOkASR(), pool, 16000,
        tracker=ErrorTracker(),
        on_escalate=target.escalate,
        transcribe_timeout_s=5.0,
    ))
    try:
        seg = await asyncio.wait_for(transcript_q.get(), timeout=3.0)
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    assert seg.jp == "こんにちは", "the chunk after the quick error must still emit"
    assert target.stage_escalation_requested is False, (
        "a transcribe that RAISED TimeoutError quickly was misread as a "
        "wedge and escalated -- pipeline restart over a one-chunk error"
    )


@pytest.mark.asyncio
async def test_asr_stage_reraises_the_worker_exception_that_landed_on_the_deadline(
    monkeypatch, caplog,
):
    """Deadline race: wait_for raises TimeoutError while the executor
    future has ALREADY completed with a NON-timeout exception (the worker
    finished a moment before the deadline handle fired, so the task's
    cancellation arrived after the future was done and its result was
    discarded). The stage must re-raise the worker's REAL exception --
    retrieving it off the future -- instead of the mislabelled deadline
    TimeoutError, which hides the true cause AND leaves the future's
    exception unretrieved (the loop's never-retrieved handler noise).

    The race is forced deterministically at the wait_for seam: the
    worker's ValueError is allowed to land ON the future first, then
    TimeoutError is delivered exactly as the loop delivers it when its
    timeout handle beats the awaiting task's resume. The handler under
    test (asr_stage's except branch) is the real one.

    Seeded defect: drop the `exc = fut.exception()` re-raise from the
    timeout handler so the bare ``raise`` runs -> the supervisor logs
    TimeoutError and never sees the ValueError -> red.
    """
    from concurrent.futures import ThreadPoolExecutor

    import yamabiko.main as _main_mod

    class _RaisingASR:
        def transcribe(self, audio, sample_rate):
            raise ValueError("CT2 backend rejected the audio buffer")

    async def _racing_wait_for(fut, timeout):
        # Let the worker's exception land on the future (done, NOT
        # cancelled), then raise the deadline the way the loop does when
        # its timeout handle fires before the awaiting task resumes.
        await asyncio.wait([fut])
        raise TimeoutError("deadline")

    monkeypatch.setattr(_main_mod.asyncio, "wait_for", _racing_wait_for)
    caplog.set_level(logging.ERROR, logger="yamabiko")

    target = _EscalationTarget()
    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    await chunk_q.put(Segment(
        audio=np.zeros(4, dtype=np.float32), t_chunk_emit=time.perf_counter(),
    ))
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(asr_stage(
        chunk_q, transcript_q, _RaisingASR(), pool, 16000,
        tracker=ErrorTracker(),
        on_escalate=target.escalate,
        transcribe_timeout_s=30.0,
    ))
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if any(r.exc_info for r in caplog.records):
                break
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    raised = [r.exc_info[0] for r in caplog.records if r.exc_info]
    assert ValueError in raised, (
        f"the worker's real exception never reached the supervisor; the "
        f"stage re-raised the deadline instead. Logged: {raised!r}"
    )
    assert TimeoutError not in raised, (
        f"the stage re-raised the deadline TimeoutError over a worker "
        f"exception that had already landed. Logged: {raised!r}"
    )
    assert target.stage_escalation_requested is False, (
        "a completed (failed) transcribe was escalated as a CUDA wedge"
    )


class _FlakyASR:
    """Scripted ASR: each transcribe() pops one item; an Exception
    instance raises, anything else becomes the transcript text."""

    def __init__(self, script: list):
        self._script = list(script)

    def transcribe(self, audio, sample_rate):
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, duration_s=1.0)


@pytest.mark.asyncio
async def test_asr_stage_skips_dedup_on_chunk_after_transcribe_exception():
    """P2 (2026-07-12 review): if asr.transcribe raises, the supervisor
    swallows the exception BEFORE the prev_forced/prev_transcript update,
    so the NEXT chunk was deduped against the chunk TWO back -- exactly
    the non-adjacent boundary the prev_forced gate exists to prevent --
    and could lose a coincidentally-matching >=3-char prefix. The stage
    must reset the dedup state on a transcribe failure so the following
    boundary skips dedup instead of using stale state.

    Chunk 1 is force-split (arming the dedup for the next boundary),
    chunk 2's transcribe raises, and chunk 3 coincidentally starts with
    chunk 1's tail. Stale-state dedup WOULD truncate chunk 3 (the exact
    pairing the force-split test above proves dedups when adjacent).

    Seeded defect: drop the prev-state reset from the transcribe except
    path -> chunk 3 emits as 'って言ったのに' (prefix lost) -> red.
    """
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor

    chunk_q: asyncio.Queue = asyncio.Queue()
    transcript_q: asyncio.Queue = asyncio.Queue()
    for seg in [
        Segment(audio=audio, t_chunk_emit=t, force_split=True),
        Segment(audio=audio, t_chunk_emit=t),
        Segment(audio=audio, t_chunk_emit=t),
    ]:
        await chunk_q.put(seg)
    asr = _FlakyASR([
        "ちょっと待ってください",
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        "くださいって言ったのに",
    ])
    pool = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(
        asr_stage(chunk_q, transcript_q, asr, pool, 16000,
                  tracker=ErrorTracker())
    )
    out: list[Segment] = []
    try:
        await asyncio.wait_for(_drain(transcript_q, chunk_q), timeout=3.0)
        while not transcript_q.empty():
            out.append(transcript_q.get_nowait())
    finally:
        task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    assert [s.jp for s in out] == [
        "ちょっと待ってください", "くださいって言ったのに",
    ], (
        "the chunk AFTER a transcribe exception was deduped against the "
        "chunk TWO back (stale prev_forced/prev_transcript)"
    )


@pytest.mark.asyncio
async def test_genre_switch_glossary_io_runs_off_event_loop_thread():
    """A pending genre switch makes translation_stage call
    GlossaryManager.set_genre, which holds the glossary lock through a
    synchronous JSON save (mkstemp + write + fsync + os.replace) AND a
    reload of the new genre file -- all disk I/O. Called inline in the
    async stage it blocks the event loop for the full duration (the
    adjacent build_ruby call is already offloaded; this was the
    structural gap). The stage must dispatch set_genre via
    asyncio.to_thread AND the switch must still take effect.

    The WS/tray genre-command tests exercise the routing layer only --
    §3.9 wired-not-exercised for this async-boundary property.

    Seeded defect (one line): revert main.py to
    ``pipeline._glossary.set_genre(state.genre)`` (drop the
    asyncio.to_thread dispatch) -> set_genre records the event-loop
    thread ident -> red.
    """
    import threading

    from yamabiko.main import translation_stage
    from yamabiko.translation.prompt import ContextBuffer as _Ctx

    loop_thread_ident = threading.get_ident()
    seen: dict = {}

    class _ThreadRecordingGlossary:
        def __init__(self):
            self._genre = "variety"

        def current_genre(self):
            return self._genre

        def set_genre(self, genre):
            seen["ident"] = threading.get_ident()
            self._genre = genre

    class _FakeRomanizationPipeline:
        def __init__(self):
            self._glossary = _ThreadRecordingGlossary()
            self._series_id = None
            self._applied_epoch = 0

        def build_ruby(self, jp):
            return []

        # The series-reload branch in translation_stage reads
        # applied_series_epoch() every sentence; with state.series_epoch and
        # this both 0 (no set_series called) the branch is a no-op, so the
        # genre-switch property under test is isolated.
        def current_series_id(self):
            return self._series_id

        def applied_series_epoch(self):
            return self._applied_epoch

        def set_gazetteer(self, gazetteer, media_id, epoch=0):
            self._series_id = media_id
            self._applied_epoch = epoch

    pipeline = _FakeRomanizationPipeline()
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    # state.genre differs from the glossary's current genre -> the
    # switch branch fires on the first translated sentence.
    state = AppState(genre="anime")
    translator = _FakeTranslator(en="Hello.")

    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp="こんにちは。",
    ))
    try:
        await asyncio.wait_for(
            translation_stage(
                transcript_q, translator, _Ctx(max_lines=2), state,
                LatencyStats(), None, max_segments=1, done=done,
                manager=None, pipeline=pipeline,
                tracker=ErrorTracker(),
            ),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        pass

    assert seen, (
        "set_genre was never invoked despite state.genre ('anime') "
        "differing from the glossary's current genre ('variety'); the "
        "genre-switch branch in _one_iter did not fire"
    )
    assert pipeline._glossary.current_genre() == "anime", (
        f"the genre switch lost its side effect: glossary still on "
        f"{pipeline._glossary.current_genre()!r}"
    )
    assert seen["ident"] != loop_thread_ident, (
        "GlossaryManager.set_genre ran ON the event-loop thread; its "
        "lock-held JSON save+reload blocks every other stage for the "
        "full disk round-trip. Dispatch it via asyncio.to_thread."
    )


@pytest.mark.asyncio
async def test_series_switch_reloads_cached_gazetteer_and_swaps_it_in(
    tmp_path, monkeypatch,
):
    """A palette selection changes the active series mid-run (set_series bumps
    AppState.series_epoch). On the next translated sentence, translation_stage
    must detect that state.series_epoch != pipeline.applied_series_epoch(),
    load THAT series' cached gazetteer off the event loop, and swap it into the
    pipeline (recording the new id + epoch). Mirrors the genre-reload harness
    above; this is the §3.9 async-boundary property the WS set_series routing
    tests don't reach.

    Seeded defect (one line): in main.py, delete the branch body (or the
    set_gazetteer call) -> the pipeline never reloads, current_series_id()
    stays None, and the asserts below go red.
    """
    from types import SimpleNamespace

    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path
    from yamabiko.main import translation_stage
    from yamabiko.translation.prompt import ContextBuffer as _Ctx

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    # A cached gazetteer on disk for the newly-selected series.
    SeriesGazetteer({"レックス": "Rex"}).save(gazetteer_path(4224))

    class _SeriesRecordingPipeline:
        def __init__(self):
            # No series loaded yet (boot with no active_series pointer).
            self._series_id = None
            self._gazetteer = None
            self._applied_epoch = 0
            # current_genre matches state.genre so the genre branch is a
            # no-op and only the series branch fires.
            self._glossary = SimpleNamespace(
                current_genre=lambda: "variety",
                set_genre=lambda genre: None,
            )

        def current_series_id(self):
            return self._series_id

        def applied_series_epoch(self):
            return self._applied_epoch

        def set_gazetteer(self, gazetteer, media_id, epoch=0):
            self._gazetteer = gazetteer
            self._series_id = media_id
            self._applied_epoch = epoch

        def build_ruby(self, jp):
            return []

    pipeline = _SeriesRecordingPipeline()
    state = AppState(genre="variety")
    state.set_series(4224, "Test Series")  # the palette selection

    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _FakeTranslator(en="Hello.")
    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp="こんにちは。",
    ))
    try:
        await asyncio.wait_for(
            translation_stage(
                transcript_q, translator, _Ctx(max_lines=2), state,
                LatencyStats(), None, max_segments=1, done=done,
                manager=None, pipeline=pipeline,
                tracker=ErrorTracker(),
            ),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        pass

    assert pipeline.current_series_id() == 4224, (
        "the series-switch branch did not stamp the new media_id on the "
        "pipeline; state.series_media_id (4224) != current_series_id() so the "
        "hot loop would reload every sentence"
    )
    assert pipeline._gazetteer is not None, (
        "the cached gazetteer for series 4224 was not loaded + swapped in"
    )
    assert pipeline._gazetteer.lookup("レックス") == "Rex", (
        "the swapped-in gazetteer is not the one cached on disk for series 4224"
    )


@pytest.mark.asyncio
async def test_repicking_same_series_reloads_a_freshly_built_cache(
    tmp_path, monkeypatch,
):
    """Re-picking the ALREADY-active series must reload a cache that appeared
    between the two picks. The first pick found no cache on disk (empty AniList
    result) so gazetteer=None was applied for media 7. The cache is then built.
    The user re-picks 7; the hot loop must reload it. The OLD media-id
    inequality gate (state.series_media_id != pipeline.current_series_id())
    skipped this reload forever, because the id never changed. The epoch gate
    catches it: every set_series bumps AppState.series_epoch, so the re-pick's
    epoch differs from the one the pipeline recorded, triggering the reload.

    Seeded defect (one line): revert the hot-loop guard to
    ``if state.series_media_id != pipeline.current_series_id():`` -> on the
    re-pick the ids are equal (both 7), the reload is skipped, the freshly
    built gazetteer never loads, and the final lookup assertion goes red.
    """
    from types import SimpleNamespace

    from yamabiko.gazetteer.gazetteer import SeriesGazetteer, gazetteer_path
    from yamabiko.main import translation_stage
    from yamabiko.translation.prompt import ContextBuffer as _Ctx

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    class _EpochRecordingPipeline:
        def __init__(self):
            self._series_id = None
            self._gazetteer = None
            self._applied_epoch = 0
            self._glossary = SimpleNamespace(
                current_genre=lambda: "variety",
                set_genre=lambda genre: None,
            )

        def current_series_id(self):
            return self._series_id

        def applied_series_epoch(self):
            return self._applied_epoch

        def set_gazetteer(self, gazetteer, media_id, epoch=0):
            self._gazetteer = gazetteer
            self._series_id = media_id
            self._applied_epoch = epoch

        def build_ruby(self, jp):
            return []

    pipeline = _EpochRecordingPipeline()
    state = AppState(genre="variety")
    translator = _FakeTranslator(en="Hello.")

    async def _run_one_iteration():
        transcript_q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()
        await transcript_q.put(Segment(
            audio=np.zeros(1, dtype=np.float32),
            t_chunk_emit=time.perf_counter(),
            jp="こんにちは。",
        ))
        try:
            await asyncio.wait_for(
                translation_stage(
                    transcript_q, translator, _Ctx(max_lines=2), state,
                    LatencyStats(), None, max_segments=1, done=done,
                    manager=None, pipeline=pipeline,
                    tracker=ErrorTracker(),
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            pass

    # First pick: no cache on disk yet -> gazetteer stays None for media 7.
    state.set_series(7, "X")
    await _run_one_iteration()
    assert pipeline.current_series_id() == 7, (
        "first pick did not stamp the series id onto the pipeline"
    )
    assert pipeline._gazetteer is None, (
        "no cache exists for series 7 yet, so the pipeline gazetteer must be "
        f"None after the first pick; got {pipeline._gazetteer!r}"
    )

    # The cache is built after the fact, then the user re-picks the SAME series.
    SeriesGazetteer({"レックス": "Rex"}).save(gazetteer_path(7))
    state.set_series(7, "X")  # same media_id, but a new epoch
    await _run_one_iteration()

    assert pipeline._gazetteer is not None, (
        "re-picking series 7 after its cache was built did not reload the "
        "gazetteer; the media-id gate skipped the reload because the id was "
        "unchanged"
    )
    assert pipeline._gazetteer.lookup("レックス") == "Rex", (
        "the reloaded gazetteer is not the freshly built cache for series 7"
    )


# --- series-context prompt hint (translation.series_context_hint) -----------
# The stage loads the cached AniList profile once per series epoch (the same
# trigger as the gazetteer reload above), composes context_hint, and routes it
# into build_user_message. Assertions target the user message the REAL
# Translator actually composed (captured at the fake HTTP wire), never kwarg
# wiring. The pick-time profile writer is LIVE: the tray's build thread
# persists profile-<media_id>.json on every pick (build_series_gazetteer ->
# build_profile, plus the missing-profile backfill branch; covered in
# tests/overlay_app/test_tray.py). These tests seed the file through
# series_profile's own write path (build_profile with a stubbed intel client)
# -- the same file shape the palette flow produces -- to stay independent of
# the Qt tray machinery.


class _StubIntelClient:
    """fetch_media_intel stub so build_profile writes a real profile JSON."""

    def __init__(self, intel: dict):
        self._intel = intel

    def fetch_media_intel(self, media_id: int) -> dict:
        return self._intel


_CYBERPUNK_INTEL = {
    "genres": ["Action"],
    "tags": [{
        "name": "Cyberpunk", "rank": 92, "category": "Theme-Sci-Fi",
        "isMediaSpoiler": False, "isGeneralSpoiler": False,
    }],
    "description": "A hacker fights the system. Season two of the saga.",
    "relations": [],
}
# context_hint over _CYBERPUNK_INTEL's ingested profile: top tag, then the
# synopsis first sentence.
_CYBERPUNK_HINT = "Cyberpunk; A hacker fights the system."


class _UserMsgCapturingClient:
    """httpx.AsyncClient stand-in installed on the REAL Translator: records
    every POST body so tests read the composed user message off the wire."""

    def __init__(self):
        self.bodies: list[dict] = []

    async def post(self, url, **kwargs):
        import httpx

        self.bodies.append(kwargs.get("json"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello."}}]},
            request=httpx.Request("POST", url),
        )

    async def aclose(self):
        pass

    def user_messages(self) -> list[str]:
        return [b["messages"][1]["content"] for b in self.bodies]


def _wire_capturing_translator():
    """REAL Translator (real build_user_message composition) over the
    capturing fake wire, so a dropped hint anywhere on the stage ->
    translate() -> prompt path is observable in the captured body."""
    from yamabiko.translation.translator import Translator, TranslationConfig

    t = Translator(TranslationConfig(
        endpoint="http://localhost:8090/v1/chat/completions",
        model="test", max_tokens=40, temperature=0.1,
    ))
    t._client = _UserMsgCapturingClient()
    return t


def _jp_segment(jp: str) -> Segment:
    return Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp=jp,
    )


async def _run_stage_one_segment(state, translator, **stage_kwargs):
    """Drive translation_stage over one segment; stage_kwargs forward as-is
    (omit series_hint_enabled to exercise the stage's own default)."""
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    await transcript_q.put(_jp_segment("こんにちは。"))
    try:
        await asyncio.wait_for(
            translation_stage(
                transcript_q, translator, ContextBuffer(max_lines=2), state,
                LatencyStats(), None, max_segments=1, done=done,
                manager=None, pipeline=None,
                tracker=ErrorTracker(),
                **stage_kwargs,
            ),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        pass


@pytest.mark.asyncio
async def test_series_hint_from_cached_profile_reaches_composed_user_message(
    tmp_path, monkeypatch,
):
    """Flag on + active series + cached profile -> the user message the
    translator sends carries the exact "Series: <hint>" line composed from
    that profile. Runs with pipeline=None: the hint path must not depend on
    the romanization pipeline (gazetteer hints do; this lever must work with
    romanization disabled).

    Seeded defect (one line): stop forwarding the hint into translate()
    (or load the profile from the wrong path / never load it) -> the line is
    absent from the captured body -> red.
    """
    from yamabiko.gazetteer.series_profile import build_profile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    build_profile(_StubIntelClient(_CYBERPUNK_INTEL), 4224)

    state = AppState(genre="variety")
    state.set_series(4224, "Test Series")
    translator = _wire_capturing_translator()

    await _run_stage_one_segment(state, translator, series_hint_enabled=True)

    msgs = translator._client.user_messages()
    assert msgs, "the translator never received a completion request"
    assert f"Series: {_CYBERPUNK_HINT}" in msgs[0], (
        f"the profile-composed series hint did not reach the composed user "
        f"message; got {msgs[0]!r}"
    )


@pytest.mark.asyncio
async def test_series_hint_absent_by_default_even_with_series_and_profile(
    tmp_path, monkeypatch,
):
    """translation.series_context_hint defaults OFF (A/B gate pending): with
    the stage's default flag, an active series with a cached profile must
    still produce a hint-free prompt.

    Seeded defect (one line): default the stage's series_hint_enabled
    parameter to True (or ignore the flag when composing) -> a Series line
    appears -> red.
    """
    from yamabiko.gazetteer.series_profile import build_profile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    build_profile(_StubIntelClient(_CYBERPUNK_INTEL), 4224)

    state = AppState(genre="variety")
    state.set_series(4224, "Test Series")
    translator = _wire_capturing_translator()

    await _run_stage_one_segment(state, translator)  # default flag

    msgs = translator._client.user_messages()
    assert msgs, "the translator never received a completion request"
    assert "Series:" not in msgs[0], (
        f"series hint leaked into the prompt with the flag off (default); "
        f"got {msgs[0]!r}"
    )


@pytest.mark.asyncio
async def test_series_hint_enabled_with_no_series_still_translates_hint_free(
    tmp_path, monkeypatch,
):
    """Flag on but no series ever picked: the segment must still translate
    normally (no crash in the hint branch) and carry no Series line.

    Seeded defect (one line): call context_hint unconditionally (drop the
    profile None-guard) -> AttributeError on None -> the supervisor eats the
    sentence, no completion is sent -> the non-empty assert goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    state = AppState(genre="variety")  # series_media_id stays None
    translator = _wire_capturing_translator()

    await _run_stage_one_segment(state, translator, series_hint_enabled=True)

    msgs = translator._client.user_messages()
    assert msgs, (
        "no completion request was sent; the no-series hint branch broke "
        "normal translation"
    )
    assert "Series:" not in msgs[0], (
        f"a Series line appeared with no active series; got {msgs[0]!r}"
    )


@pytest.mark.asyncio
async def test_series_repick_recomputes_hint_from_freshly_built_profile(
    tmp_path, monkeypatch,
):
    """Within ONE stage lifetime: first pick finds no profile (hint-free
    prompt); the profile is then written and the SAME series re-picked
    (set_series bumps the epoch, id unchanged). The next sentence must carry
    the freshly composed hint -- the epoch is the trigger, exactly like the
    gazetteer reload.

    Seeded defect (one line): gate the recompute on media-id inequality (or
    compute the hint once and never again) -> the re-pick is skipped, the
    second prompt stays hint-free -> red.
    """
    from yamabiko.gazetteer.series_profile import build_profile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    state = AppState(genre="variety")
    state.set_series(7, "X")  # first pick: no profile on disk yet
    translator = _wire_capturing_translator()

    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    stage_task = asyncio.create_task(translation_stage(
        transcript_q, translator, ContextBuffer(max_lines=2), state,
        LatencyStats(), None, max_segments=2, done=done,
        manager=None, pipeline=None,
        tracker=ErrorTracker(),
        series_hint_enabled=True,
    ))
    try:
        await transcript_q.put(_jp_segment("こんにちは。"))
        # Wait for the first completion BEFORE the re-pick so both segments
        # flow through one live stage (epoch state must persist between them).
        deadline = time.perf_counter() + 3.0
        while not translator._client.bodies:
            assert time.perf_counter() < deadline, (
                "first segment never produced a completion request"
            )
            await asyncio.sleep(0.01)

        build_profile(_StubIntelClient(_CYBERPUNK_INTEL), 7)
        state.set_series(7, "X")  # same media_id, new epoch

        await transcript_q.put(_jp_segment("元気ですか。"))
        await asyncio.wait_for(done.wait(), timeout=3.0)
    finally:
        stage_task.cancel()
        try:
            await asyncio.wait_for(stage_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    msgs = translator._client.user_messages()
    assert len(msgs) == 2, f"expected two completions, got {len(msgs)}"
    assert "Series:" not in msgs[0], (
        f"a Series line appeared before any profile existed; got {msgs[0]!r}"
    )
    assert f"Series: {_CYBERPUNK_HINT}" in msgs[1], (
        f"re-picking series 7 after its profile was written did not recompute "
        f"the hint; got {msgs[1]!r}"
    )


@pytest.mark.asyncio
async def test_pipeline_splitter_crash_escalates_via_supervisor():
    """A repeatedly-raising SentenceSplitter trips the supervisor's
    escalation hook in translation_stage.

    NAME-BODY FIX: prior name was test_pipeline_translation_crash_escalates,
    which claimed the Translator (httpx layer) crashing would escalate.
    But the body actually crashes SentenceSplitter (fast-bunkai layer) --
    Translator.translate's exceptions are swallowed inside _one_iter's
    try/except, so they never reach the supervisor. Splitter.split runs
    OUTSIDE that try/except, so its exception does. Different layer,
    different semantics, accurate name now. docs/TESTING.md §3.8.

    The translator-layer failure paths are covered separately:
    test_pipeline_translator_failures_escalate_while_stage_keeps_consuming
    (escalation feed) and
    test_translation_stage_drops_failed_sentence_and_translates_rest_of_segment
    (per-sentence drop-and-continue).

    Seeded-defect check: remove the on_escalate callback wiring in
    translation_stage, or widen the try/except to swallow splitter
    crashes -> this test fails (no escalation recorded).
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()

    from yamabiko.sentencing import splitter as _splitter_mod

    class BoomSplitter:
        def split(self, text):
            raise RuntimeError("splitter boom")

    import yamabiko.main as _main
    orig = _main.SentenceSplitter
    _main.SentenceSplitter = BoomSplitter  # type: ignore[attr-defined]

    escalations: list[str] = []
    tracker = ErrorTracker(escalation_threshold=3, max_notify_per_minute=999)
    translator = _FakeTranslator(en="unused")
    try:
        tr_task = asyncio.create_task(
            translation_stage(
                transcript_q, translator, ctx, state, stats, None,
                max_segments=0, done=done, manager=None,
                tracker=tracker,
                on_escalate=lambda s: escalations.append(s),
            )
        )
        # Feed 5 crashing segments
        for _ in range(5):
            await transcript_q.put(Segment(audio=np.zeros(1, dtype=np.float32),
                                           t_chunk_emit=time.perf_counter(),
                                           jp="こんにちは。"))
        # Let the stage loop process them. The stage blocks on
        # in_q.get() after draining, so cancel rather than signal done.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if escalations:
                break
        tr_task.cancel()
        try:
            await asyncio.wait_for(tr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    finally:
        _main.SentenceSplitter = orig  # type: ignore[attr-defined]

    assert escalations, "escalation hook should have fired after 3+ crashes"
    assert escalations[0] == "translation"


@pytest.mark.asyncio
async def test_pipeline_translator_failures_escalate_while_stage_keeps_consuming():
    """Persistent translator.translate() failures must feed the
    supervisor's ErrorTracker and fire on_escalate, while the stage
    keeps consuming segments (the per-sentence skip is preserved).

    The old per-sentence ``log.warning + continue`` swallowed every
    translate() exception, so the escalation chain (10-in-60s ->
    on_escalate -> done.set -> watchdog restart -> sanity check ->
    exit-43 dialog) was structurally unreachable for the MOST COMMON
    runtime failure class — persistent llama-server 500s, VRAM
    exhaustion, glossary OSError inside process_pre. Subtitles just
    stopped with zero notification.

    Seeded-defect checks:
    - remove the record_failure call from the translate except-block
      in _one_iter -> escalations stays empty -> red.
    - replace the per-sentence skip with a bare ``raise`` -> the
      supervisor counts crashes but each raise abandons the segment
      loop; with it, translator.calls still reaches 5 here, so this
      stays green — the paired guard is that escalation fires AND all
      five segments were attempted.
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()

    escalations: list[str] = []
    tracker = ErrorTracker(escalation_threshold=3, max_notify_per_minute=999)
    translator = _FakeTranslator(raise_exc=RuntimeError("HTTP 500 from llama-server"))

    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=0, done=done, manager=None,
            tracker=tracker,
            on_escalate=lambda s: escalations.append(s),
        )
    )
    for _ in range(5):
        await transcript_q.put(Segment(audio=np.zeros(1, dtype=np.float32),
                                       t_chunk_emit=time.perf_counter(),
                                       jp="こんにちは。"))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if escalations and translator.calls >= 5:
            break
    tr_task.cancel()
    try:
        await asyncio.wait_for(tr_task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    assert escalations, (
        "translator failures never reached the escalation hook — "
        "translate() exceptions are being swallowed without feeding "
        "the ErrorTracker: persistent translator death would stop "
        "subtitles silently"
    )
    assert escalations[0] == "translation"
    assert translator.calls >= 5, (
        f"stage stopped consuming after a translate failure "
        f"({translator.calls} calls); the per-sentence skip must "
        f"survive the escalation wiring"
    )


# --- P1-8b: a hung-but-accepting llama-server must escalate -----------------
# Each hang-class translate() failure costs ~61 s at the client (30 s
# timeout x 2 tries + 1 s backoff), so the ErrorTracker's 10-in-60s
# threshold is mathematically unreachable for timeouts: one dropped line
# per minute forever, /v1/models still answering, no restart. The stage
# tracks CONSECUTIVE timeout-class failures (httpx.TimeoutException
# family) and, at the streak threshold, fires the SAME exit-45 escalation
# directly. Any successful round-trip resets the streak.


class _ScriptedTranslator:
    """translate() pops one scripted item per call: an Exception instance
    raises, anything else is returned as the EN text."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls = 0

    async def translate(self, jp, genre, context):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def _drive_timeout_script(script, target, *, want_calls: int):
    """Feed one single-sentence segment per script item into
    translation_stage wired to the REAL PipelineApp.on_stage_escalate body
    (via `target`); return the translator after `want_calls` calls or the
    target's done event fires, whichever comes first."""
    import httpx  # noqa: F401  (script items are httpx exceptions)

    transcript_q: asyncio.Queue = asyncio.Queue()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    translator = _ScriptedTranslator(script)
    # Default tracker (threshold 10): the ONLY escalation path reachable
    # with < 10 recorded errors is the consecutive-timeout streak.
    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=0, done=target.done, manager=None,
            tracker=ErrorTracker(),
            on_escalate=target.escalate,
        )
    )
    for _ in range(len(script)):
        await transcript_q.put(Segment(
            audio=np.zeros(1, dtype=np.float32),
            t_chunk_emit=time.perf_counter(),
            jp="こんにちは。",
        ))
    for _ in range(300):
        await asyncio.sleep(0.01)
        if target.done.is_set() or translator.calls >= want_calls:
            break
    tr_task.cancel()
    try:
        await asyncio.wait_for(tr_task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    return translator


@pytest.mark.asyncio
async def test_translation_stage_escalates_after_five_consecutive_timeouts(caplog):
    """Five consecutive httpx.ReadTimeout translate() failures must fire
    the exit-45 escalation path directly -- observable as the typed
    stage_escalation_requested flag + done set on the target -- with an
    ERROR log naming the translator as wedged.

    Seeded defect: drop the consecutive-timeout streak counter from
    translation_stage -> 5 timeouts record only 5 tracker errors (below
    the 10-in-60s threshold) -> no escalation -> red.
    """
    import httpx

    caplog.set_level(logging.ERROR, logger="yamabiko")
    target = _EscalationTarget()
    translator = await _drive_timeout_script(
        [httpx.ReadTimeout("read timed out")] * 5, target, want_calls=5,
    )

    assert target.stage_escalation_requested is True, (
        "5 consecutive translator timeouts never armed the typed exit-45 "
        "request; a wedged llama-server drops one line per minute forever"
    )
    assert target.done.is_set()
    assert translator.calls == 5, (
        f"escalation must fire exactly at the streak threshold (5); the "
        f"stage stopped after {translator.calls} calls"
    )
    assert any(
        "wedged" in r.getMessage() and "translator" in r.getMessage()
        for r in caplog.records if r.levelno >= logging.ERROR
    ), "no ERROR log naming the translator as wedged"


@pytest.mark.asyncio
async def test_translator_timeout_streak_resets_on_success():
    """4 timeouts + 1 success + 4 timeouts = 8 total timeouts but never 5
    CONSECUTIVE: no escalation, and the stage keeps consuming to the end.
    Proves the counter resets on any successful round-trip rather than
    accumulating across successes.

    Seeded defect: drop the on-success reset -> the streak crosses 5 on
    the 6th timeout overall -> escalation fires -> red.
    """
    import httpx

    def _rt():
        return httpx.ReadTimeout("read timed out")

    target = _EscalationTarget()
    translator = await _drive_timeout_script(
        [_rt(), _rt(), _rt(), _rt(), "Hello.", _rt(), _rt(), _rt(), _rt()],
        target, want_calls=9,
    )

    assert translator.calls == 9, (
        f"stage stopped consuming early ({translator.calls}/9 calls)"
    )
    assert target.stage_escalation_requested is False, (
        "a success between timeout runs did not reset the streak; "
        "8 non-consecutive timeouts escalated"
    )
    assert not target.done.is_set()


@pytest.mark.asyncio
async def test_translation_stage_reports_each_completed_translation():
    """Decision 19: translation_stage awaits on_translated after a
    translate() call that returned, never after one that raised (the clear
    signal for the translator problem).

    Seeded defects: report before translate(), or from the except branch ->
    red."""
    import contextlib

    import httpx

    transcript_q: asyncio.Queue = asyncio.Queue()
    translator = _ScriptedTranslator([httpx.ReadTimeout("read timed out"), "Hello."])
    reported: list[int] = []

    async def _on_translated() -> None:
        reported.append(translator.calls)

    task = asyncio.create_task(translation_stage(
        transcript_q, translator, ContextBuffer(max_lines=2), AppState(genre="variety"),
        LatencyStats(), None, max_segments=0, done=asyncio.Event(), manager=None,
        tracker=ErrorTracker(), on_translated=_on_translated,
    ))
    for _ in range(2):
        await transcript_q.put(Segment(
            audio=np.zeros(1, dtype=np.float32), t_chunk_emit=time.perf_counter(),
            jp="こんにちは。",
        ))
    for _ in range(300):
        await asyncio.sleep(0.01)
        if translator.calls >= 2 and reported:
            break
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert reported == [2]


@pytest.mark.asyncio
async def test_translator_timeout_streak_resets_on_a_completed_error_response():
    """A llama-server that ALTERNATES ReadTimeouts with COMPLETED error
    responses (an HTTP 500 raised as HTTPStatusError, or the think-leak
    RuntimeError translator._complete raises on a parsed body) is
    answering, not wedged. Every completed round-trip -- success or error
    -- proves a request went out and came back, which is exactly what the
    "accepting but never completing" escalation claims is NOT happening.
    So a completed failure must reset the streak, and only timeout-class
    failures may extend it.

    9 failures, 5 of them timeouts, none consecutive: below the
    ErrorTracker's 10-in-60s threshold, so the streak counter is the only
    escalation path reachable here.

    Seeded defect: make a non-timeout failure neither extend nor reset the
    streak (the pre-fix shape -- only the isinstance(e, TimeoutException)
    branch touches the counter) -> the 5 scattered timeouts accumulate to
    the threshold -> the wedge escalation fires on a server that is
    provably answering -> red.
    """
    import httpx

    def _rt():
        return httpx.ReadTimeout("read timed out")

    def _http_500():
        return httpx.HTTPStatusError(
            "Server error '500 Internal Server Error'",
            request=httpx.Request(
                "POST", "http://127.0.0.1:8090/v1/chat/completions",
            ),
            response=httpx.Response(500),
        )

    target = _EscalationTarget()
    translator = await _drive_timeout_script(
        [
            _rt(), _http_500(), _rt(), _http_500(), _rt(),
            RuntimeError("Thinking-mode block detected in translation output"),
            _rt(), _http_500(), _rt(),
        ],
        target, want_calls=9,
    )

    assert translator.calls == 9, (
        f"stage stopped consuming early ({translator.calls}/9 calls)"
    )
    assert target.stage_escalation_requested is False, (
        "a translator that answered with HTTP 500s between timeouts was "
        "escalated as wedged: a completed response contradicts the "
        "'accepting but never completing' rationale, and the restart it "
        "triggers churns into the anti-flap 'keeps crashing' dialog"
    )
    assert not target.done.is_set()


@pytest.mark.asyncio
async def test_translation_stage_drops_failed_sentence_and_translates_rest_of_segment(
    caplog,
):
    """Per-sentence ride-through: one translate() failure inside a
    multi-sentence segment must drop ONLY that sentence. The remaining
    sentences of the SAME segment still translate and broadcast, done is
    not set, and a single sub-threshold failure does not escalate — a
    routine mid-run llama restart (~10 s of ConnectErrors) must degrade
    to dropped sentences, not pipeline teardown.

    This is the test that distinguishes `continue` from `raise` in the
    translate except-block. The persistent-failure test (every call
    raises) cannot: with `raise`, supervise() absorbs the crash and the
    stage moves to the NEXT segment, so its calls>=5 / escalation asserts
    stay green. Only a mid-segment failure makes the abandoned-rest
    observable.

    Seeded-defect drill: change the except's `continue` to `raise` in
    _one_iter -> sentence 3 never broadcast -> red.
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)

    class _SecondCallFails:
        def __init__(self):
            self.calls: list[str] = []

        async def translate(self, jp, genre, context):
            self.calls.append(jp)
            if len(self.calls) == 2:
                raise RuntimeError("HTTP 500 from llama-server (transient)")
            return f"EN:{jp}"

    translator = _SecondCallFails()
    escalations: list[str] = []
    tracker = ErrorTracker(escalation_threshold=3, max_notify_per_minute=999)
    telemetry = PostASRTelemetry(run_id="legacy-branch-run")
    caplog.set_level(logging.INFO, logger="yamabiko")

    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=0, done=done, manager=manager,
            tracker=tracker,
            on_escalate=lambda s: escalations.append(s),
            post_asr_telemetry=telemetry,
        )
    )
    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp="一文目です。二文目です。三文目です。",
    ))
    # Wait on the terminal OBSERVABLE (sentence 3's terminal fate), not on
    # translate() being entered or on the broadcast count: the terminal is
    # logged after the broadcast's send receipt, so cancelling as soon as
    # the second subtitle is sent races the telemetry assert.
    for _ in range(300):
        await asyncio.sleep(0.01)
        if any(
            "post_asr terminal:" in record.getMessage()
            and "artifact_seq=3 fate=" in record.getMessage()
            for record in caplog.records
        ):
            break
    tr_task.cancel()
    try:
        await asyncio.wait_for(tr_task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    assert translator.calls == ["一文目です。", "二文目です。", "三文目です。"], (
        f"all three sentences of the segment must be ATTEMPTED; got "
        f"{translator.calls!r} — a raise in the except-block abandons "
        f"the rest of the segment"
    )
    broadcast_jps = [m["jp"] for m in ws.sent if m.get("type") == "subtitle"]
    assert broadcast_jps == ["一文目です。", "三文目です。"], (
        f"sentences 1 and 3 must broadcast despite sentence 2's failure; "
        f"got {broadcast_jps!r}"
    )
    assert not done.is_set(), (
        "a single dropped sentence must not shut the pipeline down"
    )
    assert escalations == [], (
        f"one sub-threshold failure must not escalate; got {escalations!r}"
    )

    telemetry_messages = [
        record.getMessage() for record in caplog.records
        if record.name == "yamabiko"
        and record.getMessage().startswith("post_asr ")
    ]
    assert (
        "post_asr root: run_id=legacy-branch-run artifact_seq=0 "
        "kind=legacy_unattributed window_seq=None"
    ) in telemetry_messages
    sentence_edges = [
        message for message in telemetry_messages
        if message.startswith("post_asr edge:") and "kind=sentence" in message
    ]
    assert sentence_edges == [
        "post_asr edge: run_id=legacy-branch-run artifact_seq=1 "
        "parent_artifact_seq=0 kind=sentence ordinal=0",
        "post_asr edge: run_id=legacy-branch-run artifact_seq=2 "
        "parent_artifact_seq=0 kind=sentence ordinal=1",
        "post_asr edge: run_id=legacy-branch-run artifact_seq=3 "
        "parent_artifact_seq=0 kind=sentence ordinal=2",
    ]
    terminal_messages = [
        message for message in telemetry_messages
        if message.startswith("post_asr terminal:")
    ]
    assert any(
        "artifact_seq=1 fate=display_send_completed" in message
        for message in terminal_messages
    )
    assert any(
        "artifact_seq=2 fate=translation_error" in message
        for message in terminal_messages
    )
    assert any(
        "artifact_seq=3 fate=display_send_completed" in message
        for message in terminal_messages
    )
    assert not any("artifact_seq=0 fate=" in message for message in terminal_messages)


@pytest.mark.asyncio
async def test_pipeline_translator_failure_logs_traceback_on_rate_limited_notify(
    caplog,
):
    """When persistent translate() failures finally escalate, the only
    forensic evidence in main-pipeline.log used to be bare 'translation
    failed: <msg>' lines — no stack. For an HTTP 500 the message
    suffices; for a glossary OSError raised inside translate()'s
    process_pre you need the traceback. The rate-limited notify decision
    (1/min by default) doubles as the traceback gate: the notified failure
    logs with exc_info, the rest stay one-line so a 10-failure burst
    doesn't dump 10 stacks.

    Seeded-defect check: drop ``exc_info=notified`` from the warning
    in _one_iter's translate except-block — zero 'translation failed'
    records carry exc_info, the ==1 assert goes red.
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()

    # Default notify rate (1/min) — the gate under test.
    tracker = ErrorTracker(escalation_threshold=100)
    translator = _FakeTranslator(raise_exc=OSError("glossary write failed"))

    with caplog.at_level(logging.WARNING, logger="yamabiko"):
        tr_task = asyncio.create_task(
            translation_stage(
                transcript_q, translator, ctx, state, stats, None,
                max_segments=0, done=done, manager=None,
                tracker=tracker,
            )
        )
        for _ in range(5):
            await transcript_q.put(Segment(audio=np.zeros(1, dtype=np.float32),
                                           t_chunk_emit=time.perf_counter(),
                                           jp="こんにちは。"))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if translator.calls >= 5:
                break
        tr_task.cancel()
        try:
            await asyncio.wait_for(tr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    failures = [r for r in caplog.records
                if "translation failed" in r.getMessage()]
    assert len(failures) == 5, (
        f"expected 5 per-sentence failure warnings, got {len(failures)}"
    )
    with_tb = [r for r in failures if r.exc_info]
    assert len(with_tb) == 1, (
        f"{len(with_tb)} of {len(failures)} failure warnings carry a "
        f"traceback; expected exactly 1 (the rate-limited notified "
        f"failure) — 0 means escalations are untriageable from the "
        f"log, >1 means a failure burst spams stacks"
    )


@pytest.mark.asyncio
async def test_pipeline_post_translate_scaffolding_crash_escalates_via_supervisor():
    """A repeatedly-raising POST-TRANSLATE scaffolding call (here,
    ``stats.record``) must trip the supervisor's escalation hook --
    same contract as the splitter test but exercising a different
    seam inside ``_one_iter``.

    Complements ``test_pipeline_splitter_crash_escalates_via_supervisor``
    (which covers the PRE-translate `splitter.split` seam, upstream
    of the try/except around `translator.translate`). This one
    covers the DOWNSTREAM seam: after translate succeeds, `_one_iter`
    runs `ctx.push(en)`, `stats.record(sub_seg)`, `print(..., flush=True)`,
    and `latency_log.write(...)` -- none of which are wrapped in the
    translate-try. A crash in any of them escapes `_one_iter` and must
    reach the supervisor.

    Note: state lookup via `ctx.snapshot()` is an argument evaluation
    inside the translate-try, so the escapable seams are all
    POST-translate. This test locks that post-translate path.

    Seeded-defect check: widen `_one_iter`'s try/except to cover the
    entire for-loop body (so post-translate exceptions get swallowed
    like translate's) -> stats.record crashes are silently eaten,
    no escalation, this test fails.
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)

    class _BoomStats:
        """Stand-in for LatencyStats whose record() always raises.

        Covers the post-translate crash class: translator returns
        cleanly, but the subsequent state-update call raises. In
        production, this could happen via ctx.push (e.g., a bug in
        ContextBuffer's internal deque) or stats.record (e.g., a
        divide-by-zero in a future percentile calc). Any of them
        escape _one_iter and must reach the supervisor.
        """
        def __init__(self):
            self.calls = 0

        def record(self, seg):
            self.calls += 1
            raise RuntimeError(f"post-translate stats.record boom #{self.calls}")

        def summary(self):
            return {}

    stats = _BoomStats()

    escalations: list[str] = []
    tracker = ErrorTracker(escalation_threshold=3, max_notify_per_minute=999)
    translator = _FakeTranslator(en="Hello.")

    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=0, done=done, manager=None,
            tracker=tracker,
            on_escalate=lambda s: escalations.append(s),
        )
    )
    try:
        # Feed 5 segments. Each one: translator returns "Hello.",
        # then stats.record raises -> escapes _one_iter -> supervise
        # increments error count. After 3 the escalation hook fires.
        for _ in range(5):
            await transcript_q.put(Segment(
                audio=np.zeros(1, dtype=np.float32),
                t_chunk_emit=time.perf_counter(),
                jp="こんにちは。",
            ))
        # Drain loop; each iteration advances one segment + crashes.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if escalations:
                break
        tr_task.cancel()
        try:
            await asyncio.wait_for(tr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    finally:
        pass

    # Observable effects: translator was called (translate ran each
    # iteration), stats.record crashed each time, escalation fired
    # with the expected stage name.
    assert translator.calls >= 3, (
        f"translator.translate must be called for each segment before "
        f"the post-translate crash; got {translator.calls} calls"
    )
    assert stats.calls >= 3, (
        f"stats.record must be reached (and crash) on each translated "
        f"segment; got {stats.calls} calls"
    )
    assert escalations, (
        "escalation hook should have fired after 3+ post-translate "
        "crashes; it did not. Either translation_stage silently "
        "swallowed the stats.record exceptions (check the for-loop "
        "in _one_iter for over-broad try/except) or on_escalate was "
        "not wired."
    )
    assert escalations[0] == "translation", (
        f"escalation fired for wrong stage: {escalations[0]!r}"
    )


# ---------------------------------------------------------------------------
# Privacy: no transcript content in default-level logs (hardening Task 6)
#
# Three INFO lines in translation_stage used to embed raw JP/EN
# snippets unconditionally (main.py:252 / :292-293 / :353). They now
# log LENGTHS at INFO and gate the snippet behind the SAME consent flag
# that gates latency.log's raw text (logging.transcript_log_enabled),
# making bug_report.yml's "logs contain no transcript content" claim
# true by default.
# ---------------------------------------------------------------------------


async def _run_translation_stage_until_marker(
    *, jp: str, translator, caplog, marker: str, transcript_enabled: bool,
):
    """Feed one Segment into translation_stage and wait until `marker`
    shows up in caplog (the log line under test), then cancel the stage.
    Fails the test if the marker never appears -- a driver that can't
    reach its path must not pass silently.
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()
    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=1, done=done,
            tracker=ErrorTracker(),
            transcript_enabled=transcript_enabled,
        )
    )
    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp=jp,
    ))
    try:
        for _ in range(150):
            if any(marker in r.getMessage() for r in caplog.records):
                return
            await asyncio.sleep(0.02)
        raise AssertionError(
            f"marker {marker!r} never logged; the driver did not reach the "
            f"path under test. Records: "
            f"{[r.getMessage() for r in caplog.records]!r}"
        )
    finally:
        tr_task.cancel()
        try:
            await asyncio.wait_for(tr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript_enabled", [False, True])
async def test_dropped_non_ja_log_gates_content_behind_transcript_flag(
    transcript_enabled, caplog,
):
    """The non-JA-drop INFO line (main.py:252) must carry the length
    always, and the raw snippet ONLY when transcript logging is
    consented to. Default (False): NO segment content in the log.

    Seeded defect: revert :252 to the unconditional
    ``log.info("dropped non-JA segment: %r", raw[:60])`` -> the
    flag-off content-absence assert goes red.
    """
    hallucination = "And that concludes today's broadcast."
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="yamabiko"):
        await _run_translation_stage_until_marker(
            jp=hallucination,
            translator=_FakeTranslator(en="unused"),
            caplog=caplog,
            marker="dropped non-JA segment",
            transcript_enabled=transcript_enabled,
        )
    line = next(
        r.getMessage() for r in caplog.records
        if "dropped non-JA segment" in r.getMessage()
    )
    assert f"len={len(hallucination)}" in line, (
        f"length metric missing from the drop line: {line!r}"
    )
    if transcript_enabled:
        assert hallucination[:20] in caplog.text, (
            "consented transcript snippet missing from the log"
        )
    else:
        assert hallucination[:20] not in caplog.text, (
            "segment content leaked into the default-level log"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript_enabled", [False, True])
async def test_empty_en_drop_log_gates_content_behind_transcript_flag(
    transcript_enabled, caplog,
):
    """The empty-EN drop INFO line (main.py:353) must not embed the JP
    sentence unless transcript logging is enabled.

    Seeded defect: restore the unconditional
    ``log.info("dropped empty EN for JP=%r", ...)`` -> the flag-off
    content-absence assert goes red.
    """
    jp = "全然違いますよ。"
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="yamabiko"):
        await _run_translation_stage_until_marker(
            jp=jp,
            translator=_FakeTranslator(en=""),  # empty EN -> drop path
            caplog=caplog,
            marker="dropped empty EN",
            transcript_enabled=transcript_enabled,
        )
    line = next(
        r.getMessage() for r in caplog.records
        if "dropped empty EN" in r.getMessage()
    )
    assert f"len={len(jp)}" in line, f"length metric missing: {line!r}"
    if transcript_enabled:
        assert "全然違います" in caplog.text, (
            "consented transcript snippet missing from the log"
        )
    else:
        assert "全然違います" not in caplog.text, (
            "JP sentence leaked into the default-level log"
        )


# ---------------------------------------------------------------------------
# Ultra-short JA fragment guard (echo-of-previous-line bug)
#
# A live session logged 1-char ASR fragments being "translated" into whole
# fabricated sentences that echoed the PRIOR line:
#   お  -> "Yes, Your Highness."  (the previous line's EN, re-emitted twice)
#   関  -> a 93-char re-run of the previous translation
# Root cause: is_japanese() passes a lone kana/kanji, splitter yields it as a
# one-token "sentence", and translate() was fed the rolling EN context -- so
# the model, given almost no input signal but rich context, regurgitated it.
#
# Two gates in translation_stage._one_iter:
#   1. _is_bare_interjection() DROPS a lone kana (お/え/あ/ん/う) -- filler with
#      no translatable content. Conservative: 2+ char replies (はい/うん) live.
#      _has_no_content() drops punctuation-only sentences (......) the same way.
#   2. _context_for() STARVES context for any remaining ultra-short input:
#      1-char content (a lone kanji like 関) and 2-char ALL-KANA content
#      (ニャ -> "Man, 100-yen lighters!" confabulated from rolling context in
#      a live session; ニャ/おう/はい translate fine context-free). Translate
#      with [] so it cannot echo. The line is still translated and shown.
#      2-char content containing kanji keeps context (kanji carries lexical
#      weight worth disambiguating).
# ---------------------------------------------------------------------------


class _ContextRecordingTranslator:
    """Records (jp, context) for each translate() call so a test can assert
    exactly what context the stage fed the model."""

    def __init__(self, en: str = "hello"):
        self._en = en
        self.calls: list[tuple[str, list]] = []

    async def translate(self, jp, genre, context):
        self.calls.append((jp, list(context)))
        return self._en


# --- pure helper unit tests -------------------------------------------------


@pytest.mark.parametrize("kana", ["お", "え", "あ", "ん", "う", "ア", "ー"])
def test_is_bare_interjection_true_for_single_kana(kana):
    """A lone kana is a bare interjection -> dropped.

    Seeded defect: change the length bound in _is_bare_interjection from
    `== 1` to `== 0` -> every case here goes red.
    """
    from yamabiko.main import _is_bare_interjection
    assert _is_bare_interjection(kana) is True


@pytest.mark.parametrize("text", ["お。", "え？", "あ!", "ア？"])
def test_is_bare_interjection_true_for_single_kana_with_terminal_punctuation(text):
    """A lone kana remains a bare interjection when ASR/splitting keeps a
    terminal mark attached.

    Seeded defect: measure len(text.strip()) directly -> each punctuated
    one-kana fragment is translated with context and can echo the prior line.
    """
    from yamabiko.main import _is_bare_interjection
    assert _is_bare_interjection(text) is True


@pytest.mark.parametrize("text", ["はい", "うん", "ええ", "そう"])
def test_is_bare_interjection_false_for_two_char_replies(text):
    """Valid 2-char replies must survive (the gate is deliberately narrow).

    Seeded defect: widen the bound to `<= 2` -> はい/うん go red here (and
    would be silently dropped in production).
    """
    from yamabiko.main import _is_bare_interjection
    assert _is_bare_interjection(text) is False


@pytest.mark.parametrize("kanji", ["関", "雨", "犬"])
def test_is_bare_interjection_false_for_single_kanji(kanji):
    """A lone kanji still carries lexical content; it is context-starved, not
    dropped.

    Seeded defect: make _is_kana() also match the kanji ranges -> these go red
    and 関 would be dropped instead of shown.
    """
    from yamabiko.main import _is_bare_interjection
    assert _is_bare_interjection(kanji) is False


def test_context_for_starves_single_character_input():
    """A 1-char input (the lone kanji 関) is translated with EMPTY context so
    the model cannot regurgitate the previous line.

    Seeded defect: `return context` unconditionally -> this goes red.
    """
    from yamabiko.main import _context_for
    assert _context_for("関", ["Yes, Your Highness."]) == []


@pytest.mark.parametrize("text", ["関。", "雨?", "犬！"])
def test_context_for_starves_single_character_input_with_terminal_punctuation(text):
    """A one-kanji fragment with only trailing punctuation still gets empty
    context; punctuation must not let the previous EN line leak back in.

    Seeded defect: use len(text.strip()) for the starve check -> all cases
    receive ["Yes, Your Highness."] and this goes red.
    """
    from yamabiko.main import _context_for
    assert _context_for(text, ["Yes, Your Highness."]) == []


@pytest.mark.parametrize("text", ["そうか", "こんにちは。", "全然違いますよ。"])
def test_context_for_keeps_context_for_multichar_input(text):
    """3+ char inputs keep the real rolling context, including all-kana ones
    (そうか is the boundary: one char past the kana starve limit).

    Seeded defect: `return []` unconditionally -> red.
    """
    from yamabiko.main import _context_for
    prev = ["Yes, Your Highness."]
    assert _context_for(text, prev) == prev


@pytest.mark.parametrize("text", ["ニャ", "おう", "はい", "うん", "はい。"])
def test_context_for_starves_two_char_kana_only_input(text):
    """2-char all-kana content is translated with EMPTY context: a live
    session showed ニャ + rich EN context confabulating whole sentences
    ("Man, 100-yen lighters!"), and ニャ/おう/はい translate fine context-free.
    Trailing punctuation (はい。) does not count as content.

    Seeded defect: drop the all-kana length-2 branch from _context_for
    (starve only at len <= 1) -> every case here receives the real context
    and goes red.
    """
    from yamabiko.main import _context_for
    assert _context_for(text, ["Man, 100-yen lighters!"]) == []


@pytest.mark.parametrize("text", ["二人", "雨だ", "犬か"])
def test_context_for_keeps_context_for_two_char_input_containing_kanji(text):
    """2-char content containing a kanji keeps the real rolling context: the
    kana starve extension must not widen to kanji, which carries lexical
    weight worth disambiguating (see _is_kana docstring).

    Seeded defect: starve on len <= 2 unconditionally (drop the all-kana
    check) -> every case here gets [] and goes red.
    """
    from yamabiko.main import _context_for
    prev = ["Yes, Your Highness."]
    assert _context_for(text, prev) == prev


# --- integration through translation_stage ---------------------------------


async def _drive_short_segment(*, jp, ctx, translator, caplog,
                               drop_marker="dropped bare interjection"):
    """Feed one Segment into translation_stage (max_segments=1) with a manager
    wired, and wait until the stage either emits a subtitle (done) or logs the
    drop marker. Returns the _RecordingWS so the caller can assert on emitted
    subtitles.
    """
    transcript_q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    state = AppState(genre="variety")
    stats = LatencyStats()
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)
    tr_task = asyncio.create_task(
        translation_stage(
            transcript_q, translator, ctx, state, stats, None,
            max_segments=1, done=done, manager=manager,
            tracker=ErrorTracker(), transcript_enabled=False,
        )
    )
    await transcript_q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp=jp,
    ))
    try:
        for _ in range(150):
            if done.is_set() or any(
                drop_marker in r.getMessage() for r in caplog.records
            ):
                break
            await asyncio.sleep(0.02)
    finally:
        tr_task.cancel()
        try:
            await asyncio.wait_for(tr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    return ws


@pytest.mark.asyncio
async def test_translation_stage_drops_bare_interjection_o(caplog):
    """A lone お is dropped: the translator is NEVER called and NO subtitle is
    emitted, even though a prior EN line sits in context.

    Seeded defect: delete the `if _is_bare_interjection(...)` drop gate in
    _one_iter -> お is translated, translator.calls becomes non-empty and a
    subtitle is broadcast -> both asserts go red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Yes, Your Highness.")  # the line the model must not echo
    translator = _ContextRecordingTranslator(en="Yes, Your Highness.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        ws = await _drive_short_segment(
            jp="お", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert translator.calls == [], (
        f"bare interjection was translated instead of dropped: {translator.calls!r}"
    )
    assert ws.sent == [], f"bare interjection produced a subtitle: {ws.sent!r}"
    assert any("dropped bare interjection" in r.getMessage()
               for r in caplog.records), "drop was not logged"


@pytest.mark.asyncio
async def test_translation_stage_drops_punctuated_bare_interjection_o(caplog):
    """A lone お followed only by punctuation is still dropped. This is the
    real pipeline path, not just the helper.

    Seeded defect: leave _is_bare_interjection length-based on raw stripped
    text -> translator.calls is non-empty and a subtitle is broadcast.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Yes, Your Highness.")
    translator = _ContextRecordingTranslator(en="Yes, Your Highness.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        ws = await _drive_short_segment(
            jp="お。", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert translator.calls == [], (
        f"punctuated bare interjection was translated: {translator.calls!r}"
    )
    assert ws.sent == [], f"punctuated bare interjection produced: {ws.sent!r}"
    assert any("dropped bare interjection" in r.getMessage()
               for r in caplog.records), "drop was not logged"


@pytest.mark.asyncio
async def test_translation_stage_drops_leading_ellipsis_sentence(caplog):
    """A window that opens with a pause ellipsis splits into ``......`` and
    ``めんどくさい。``. Only the worded sentence is translated and shown; the
    dots-only sentence gets the display-policy fate instead of a "..."
    subtitle.

    Seeded defect: drop the no-content check from the display-policy gate in
    _one_iter -> ``......`` is translated and shown, and its fate is a
    display fate, not display_policy_interjection -> red.
    """
    caplog.set_level(logging.INFO, logger="yamabiko")
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)
    translator = _DrainTranslator(en="...")

    await _translate_until_drained(
        [_jp_segment("......めんどくさい。")],
        translator,
        PostASRTelemetry(run_id="contentless"),
        manager=manager,
    )

    assert translator.calls == ["めんどくさい。"]
    assert [m["jp"] for m in ws.sent] == ["めんどくさい。"]
    assert len(_terminal_lines(caplog, "display_policy_interjection")) == 1


@pytest.mark.asyncio
async def test_translation_stage_translates_the_words_after_an_unterminated_leading_ellipsis(
    caplog,
):
    """A window that opens with a pause ellipsis and ends without 。 splits
    into ``......`` and ``ふーん``. The dots are not a sentence, so ``ふーん``
    is the whole line: it is translated now, not held as an unfinished tail
    for the next line.

    Seeded defect: count the dots-only sentence in _assemble_sentences'
    trailing_partial (len(sentences) > 1) -> ふーん is buffered and joined
    onto the next line, never translated alone -> red.
    """
    caplog.set_level(logging.INFO, logger="yamabiko")
    manager = ConnectionManager()
    ws = _RecordingWS()
    manager.add(ws)
    translator = _DrainTranslator(en="Hmm.")

    await _translate_until_drained(
        [_jp_segment("......ふーん")],
        translator,
        PostASRTelemetry(run_id="leading-ellipsis-tail"),
        manager=manager,
    )

    assert translator.calls == ["ふーん"]


@pytest.mark.asyncio
async def test_translation_stage_starves_context_for_single_kanji(caplog):
    """A lone 関 IS translated and shown, but with EMPTY context so it cannot
    echo the previous line.

    Seeded defect: pass ctx.snapshot() directly at the translate call (revert
    _context_for) -> the translator receives ["Yes, Your Highness."] and the
    empty-context assert goes red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Yes, Your Highness.")
    translator = _ContextRecordingTranslator(en="A barrier.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        ws = await _drive_short_segment(
            jp="関", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert len(translator.calls) == 1, (
        f"single kanji should be translated exactly once: {translator.calls!r}"
    )
    jp_arg, ctx_arg = translator.calls[0]
    assert jp_arg == "関"
    assert ctx_arg == [], (
        f"single-kanji input was fed real context (echo risk): {ctx_arg!r}"
    )
    assert [m["jp"] for m in ws.sent] == ["関"], (
        f"single kanji should still be emitted as a subtitle: {ws.sent!r}"
    )


@pytest.mark.asyncio
async def test_translation_stage_starves_context_for_punctuated_single_kanji(caplog):
    """A lone kanji with punctuation is translated and shown, but with EMPTY
    context so it cannot echo the previous line.

    Seeded defect: count punctuation as content in _context_for -> the
    translator receives ["Yes, Your Highness."] and this goes red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Yes, Your Highness.")
    translator = _ContextRecordingTranslator(en="A barrier.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        ws = await _drive_short_segment(
            jp="関。", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert len(translator.calls) == 1, (
        f"punctuated single kanji should translate once: {translator.calls!r}"
    )
    jp_arg, ctx_arg = translator.calls[0]
    assert jp_arg == "関。"
    assert ctx_arg == [], (
        f"punctuated single kanji was fed real context: {ctx_arg!r}"
    )
    assert [m["jp"] for m in ws.sent] == ["関。"], (
        f"punctuated single kanji should still be emitted: {ws.sent!r}"
    )


@pytest.mark.asyncio
async def test_translation_stage_keeps_context_for_normal_line(caplog):
    """Regression guard: a normal multi-char line still receives the real
    rolling context.

    Seeded defect: starve context unconditionally (_context_for returns []) ->
    ctx_arg is [] and this goes red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Yes, Your Highness.")
    translator = _ContextRecordingTranslator(en="Good morning.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        await _drive_short_segment(
            jp="おはようございます。", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert len(translator.calls) == 1, (
        f"normal line should be translated once: {translator.calls!r}"
    )
    _jp_arg, ctx_arg = translator.calls[0]
    assert ctx_arg == ["Yes, Your Highness."], (
        f"normal line lost its rolling context: {ctx_arg!r}"
    )


@pytest.mark.asyncio
async def test_translation_stage_starves_context_for_two_char_kana(caplog):
    """A 2-char all-kana line (ニャ) IS translated and shown, but with EMPTY
    context so the model cannot confabulate from the rolling EN lines
    (live session: ニャ -> "Man, 100-yen lighters!", 9 occurrences).

    Seeded defect: drop the all-kana length-2 branch from _context_for ->
    the translator receives ["Man, 100-yen lighters!"] and the
    empty-context assert goes red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Man, 100-yen lighters!")  # the line the model must not mine
    translator = _ContextRecordingTranslator(en="Meow.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        ws = await _drive_short_segment(
            jp="ニャ", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert len(translator.calls) == 1, (
        f"2-char kana line should be translated exactly once: {translator.calls!r}"
    )
    jp_arg, ctx_arg = translator.calls[0]
    assert jp_arg == "ニャ"
    assert ctx_arg == [], (
        f"2-char kana input was fed real context (confabulation risk): {ctx_arg!r}"
    )
    assert [m["jp"] for m in ws.sent] == ["ニャ"], (
        f"2-char kana line should still be emitted as a subtitle: {ws.sent!r}"
    )


@pytest.mark.asyncio
async def test_translation_stage_keeps_context_for_two_char_kanji_line(caplog):
    """A 2-char line containing kanji (二人) keeps the real rolling context
    through the stage: the kana starve extension must not widen to kanji.

    Seeded defect: starve on len <= 2 unconditionally in _context_for ->
    ctx_arg is [] and this goes red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Yes, Your Highness.")
    translator = _ContextRecordingTranslator(en="The two of us.")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        ws = await _drive_short_segment(
            jp="二人", ctx=ctx, translator=translator, caplog=caplog,
        )
    assert len(translator.calls) == 1, (
        f"2-char kanji line should be translated exactly once: {translator.calls!r}"
    )
    jp_arg, ctx_arg = translator.calls[0]
    assert jp_arg == "二人"
    assert ctx_arg == ["Yes, Your Highness."], (
        f"2-char kanji line lost its rolling context: {ctx_arg!r}"
    )
    assert [m["jp"] for m in ws.sent] == ["二人"], (
        f"2-char kanji line should still be emitted as a subtitle: {ws.sent!r}"
    )


@pytest.mark.asyncio
async def test_translation_stage_clears_context_when_the_series_changes():
    """A series pick (set_series bumps AppState.series_epoch) starts another
    show: the English lines of the previous one must not reach the
    translator as context for the first line after the pick.

    Seeded defect: drop the ctx.clear() on a series_epoch change in
    _one_iter -> the second call gets the previous show's lines as context
    -> red.
    """
    ctx = ContextBuffer(max_lines=2)
    ctx.push("Line from the previous show.")
    translator = _ContextRecordingTranslator(en="EN line.")
    state = AppState(genre="anime")
    transcript_q: asyncio.Queue = asyncio.Queue()
    stage_task = asyncio.create_task(translation_stage(
        transcript_q, translator, ctx, state, LatencyStats(), None,
        max_segments=2, done=asyncio.Event(), manager=None,
        tracker=ErrorTracker(),
    ))
    try:
        await transcript_q.put(_jp_segment("今日はいい天気ですね。"))
        deadline = time.perf_counter() + 3.0
        while len(translator.calls) < 1:
            assert time.perf_counter() < deadline, "the first line was never translated"
            await asyncio.sleep(0.01)
        state.set_series(7, "X")
        await transcript_q.put(_jp_segment("明日も晴れるでしょう。"))
        await asyncio.wait_for(stage_task, timeout=3.0)
    finally:
        stage_task.cancel()
        try:
            await asyncio.wait_for(stage_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    assert translator.calls[0][1] == ["Line from the previous show."]
    assert translator.calls[1][1] == [], (
        f"the previous show's lines reached the translator after the pick: "
        f"{translator.calls[1][1]!r}"
    )


# --- run_pipeline teardown: one failing step must not eat the rest -----------
# The teardown seams (fake HotkeyManager, stubbed overlay runtime, tmp-path
# config) already exist in test_build_pipeline; reuse them rather than
# rebuilding a second run_pipeline harness. The imported fixture is
# re-exported into this module's namespace, which is how pytest resolves it.

from tests.test_build_pipeline import (  # noqa: E402
    _drive_run_pipeline_with_early_done,
    _pipeline_env,  # noqa: F401  (fixture, resolved by name)
    _write_overrides,
)


def test_run_pipeline_teardown_continues_after_a_failing_glossary_save(
    _pipeline_env, tmp_path, monkeypatch, caplog,
):
    """The glossary save in run_pipeline's finally starts a thread, and
    Thread.start() raises RuntimeError under resource exhaustion. Unguarded,
    that exception escapes the rest of the teardown: the hotkey unregister
    below it never runs (a low-level Windows keyboard hook then outlives the
    process) and neither does the overlay shutdown ladder. The save is the
    lowest-stakes step in the block -- genre data re-persists on the next
    periodic save -- so a failure there must be logged and stepped over.

    Seeded defect: remove the try/except around the _save_glossary_bounded
    await in run_pipeline's finally -> the RuntimeError propagates out of
    run_pipeline, the unregister never runs -> red (the test errors on the
    escaped RuntimeError, and the unregister assertion is unreachable).
    """
    env = _pipeline_env
    _write_overrides(tmp_path, {
        "romanization": {"enabled": True},
        "hotkeys": {"enabled": True, "pause_resume": "Ctrl+Alt+P"},
    })

    async def _failing_save(glossary_mgr, remaining_s):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(env.module, "_save_glossary_bounded", _failing_save)
    caplog.set_level(logging.WARNING, logger="yamabiko")

    app = asyncio.run(_drive_run_pipeline_with_early_done(env, tmp_path))

    assert app.glossary_mgr is not None, (
        "precondition: romanization on, so the teardown reaches the save"
    )
    assert "unregister" in env.fake_hotkey_mgr.calls, (
        "the hotkey unregister that follows the glossary save never ran: a "
        "failing save aborted the remaining teardown, leaking the keyboard "
        "hook and skipping the overlay shutdown ladder"
    )
    assert any(
        "glossary save" in r.getMessage().lower()
        for r in caplog.records if r.levelno >= logging.WARNING
    ), (
        f"a failing glossary save left no WARNING in the log; records: "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )


# Lines lost across capture gaps and the post-ASR fate of every emission site.
# The drain sentinel parks inside translate(), so cancelling the stage after
# it arrives proves every earlier segment finished without giving the
# sentinel itself a terminal fate.

_DRAIN_SENTINEL = "区切りです。"


class _DrainTranslator:
    def __init__(self, en: str = "Line.", on_translate=None):
        self.calls: list[str] = []
        self._en = en
        self._on_translate = on_translate
        self.drained = asyncio.Event()

    async def translate(self, jp, genre, context, **_kwargs):
        if jp == _DRAIN_SENTINEL:
            self.drained.set()
            await asyncio.Event().wait()
        self.calls.append(jp)
        if self._on_translate is not None:
            self._on_translate(jp)
        return self._en


async def _translate_until_drained(
    segments: list[Segment],
    translator: _DrainTranslator,
    telemetry: PostASRTelemetry,
    *,
    manager=None,
    pipeline=None,
    remainder_state=None,
) -> None:
    transcript_q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(translation_stage(
        transcript_q, translator, ContextBuffer(max_lines=2),
        AppState(genre="variety"), LatencyStats(), None,
        max_segments=0, done=asyncio.Event(), manager=manager,
        pipeline=pipeline, tracker=ErrorTracker(),
        post_asr_telemetry=telemetry, remainder_state=remainder_state,
    ))
    for segment in [*segments, _jp_segment(_DRAIN_SENTINEL)]:
        await transcript_q.put(segment)
    try:
        await asyncio.wait_for(translator.drained.wait(), timeout=5.0)
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


def _terminal_lines(caplog, fate: str) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("post_asr terminal:")
        and f" fate={fate} " in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_discontinuity_segment_drops_buffered_tail_instead_of_joining_it(
    caplog,
):
    """A buffered unfinished tail belongs to audio before a capture gap; the
    first segment after the gap must not be glued onto it.

    Seeded defect: drop the ``abandon_for`` call guarded by
    ``seg.discontinuity_before`` in translation_stage -> the stage translates
    ``でも続きです。`` -> red.
    """
    caplog.set_level(logging.INFO, logger="yamabiko")
    translator = _DrainTranslator()
    gap_segment = _jp_segment("続きです。")
    gap_segment.discontinuity_before = True

    await _translate_until_drained(
        [_jp_segment("最初です。でも"), gap_segment],
        translator,
        PostASRTelemetry(run_id="tail-gap"),
    )

    assert translator.calls == ["最初です。", "続きです。"]
    assert len(_terminal_lines(caplog, "tail_discontinuity_cleared")) == 1


@pytest.mark.asyncio
async def test_asr_stage_carries_discontinuity_past_an_empty_transcript():
    """The gap flag rides on the first VAD segment after the gap. When ASR
    drops that segment as empty, the next forwarded segment must carry it.

    Seeded defect: forward ``seg.discontinuity_before`` unchanged instead of
    the pending flag -> the forwarded segment reads False -> red.
    """
    audio = np.zeros(16000, dtype=np.float32)
    t = time.perf_counter()
    out = await _run_asr_stage(
        [
            Segment(audio=audio, t_chunk_emit=t, discontinuity_before=True),
            Segment(audio=audio, t_chunk_emit=t),
        ],
        ["", "次の台詞です"],
    )

    assert [(s.jp, s.discontinuity_before) for s in out] == [
        ("次の台詞です", True),
    ]


class _BroadcastStub:
    def __init__(self, outcome):
        self._outcome = outcome

    async def broadcast(self, payload):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FailingClient:
    async def send_json(self, payload):
        raise ConnectionResetError("client went away")

    async def close(self, code=1000):
        return None


class _RubyPipeline:
    """Minimal romanization seam; build_ruby raises for one sentence."""

    def __init__(self, raise_for: str):
        self._raise_for = raise_for
        self._glossary = SimpleNamespace(current_genre=lambda: "variety")

    def applied_series_epoch(self) -> int:
        return 0

    def build_ruby(self, jp: str):
        if jp == self._raise_for:
            raise RuntimeError("ruby failed")
        return []


class _NoSplit:
    """Sentence splitter seam that yields nothing for one input."""

    def __init__(self, empty_for: str):
        self._empty_for = empty_for

    def split(self, text: str) -> list[str]:
        return [] if text == self._empty_for else [text]


_TRANSLATION_FATES = sorted(
    fate.value for fate in PostASRFate
    if fate not in (
        PostASRFate.DISPLAY_SEND_COMPLETED,
        PostASRFate.TRANSLATION_ERROR,
    )
)


@pytest.mark.asyncio
@pytest.mark.parametrize("fate", _TRANSLATION_FATES)
async def test_each_emission_site_records_exactly_one_terminal(
    fate, caplog, monkeypatch,
):
    """The minimal input reaching each post-ASR emission site in main.py
    produces exactly one terminal line with that fate.

    Seeded defect: delete (or duplicate) the ``terminal`` call at any one
    emission site -> that case counts 0 (or 2) -> red.
    """
    import yamabiko.main as main_module
    from yamabiko.sentencing.remainder import PendingRemainderState

    caplog.set_level(logging.INFO, logger="yamabiko")
    telemetry = PostASRTelemetry(run_id="fate-sites")

    if fate == "dedup_empty":
        from concurrent.futures import ThreadPoolExecutor

        audio = np.zeros(16000, dtype=np.float32)
        t = time.perf_counter()
        chunk_q: asyncio.Queue = asyncio.Queue()
        transcript_q: asyncio.Queue = asyncio.Queue()
        await chunk_q.put(Segment(audio=audio, t_chunk_emit=t, force_split=True))
        await chunk_q.put(Segment(audio=audio, t_chunk_emit=t))
        pool = ThreadPoolExecutor(max_workers=1)
        task = asyncio.create_task(asr_stage(
            chunk_q, transcript_q,
            _SeqASR(["ちょっと待ってください", "ください"]), pool, 16000,
            tracker=ErrorTracker(), post_asr_telemetry=telemetry,
        ))
        try:
            await asyncio.wait_for(_drain(transcript_q, chunk_q), timeout=3.0)
        finally:
            task.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        assert len(_terminal_lines(caplog, fate)) == 1
        return

    translator = _DrainTranslator()
    segments = [_jp_segment("今日は晴れです。")]
    stage_kwargs: dict = {}
    if fate == "language_policy_rejected":
        segments = [_jp_segment("And that concludes the broadcast.")]
    elif fate == "sentence_split_empty":
        monkeypatch.setattr(
            main_module, "SentenceSplitter", lambda: _NoSplit("今日は晴れです。"),
        )
    elif fate == "display_policy_interjection":
        segments = [_jp_segment("お。")]
    elif fate == "tail_expired":
        clock = [1000.0]
        monkeypatch.setattr(main_module, "_monotonic", lambda: clock[0])

        def _age_tail(jp: str) -> None:
            if jp == "最初です。":
                clock[0] += 10_000.0

        translator = _DrainTranslator(on_translate=_age_tail)
        segments = [_jp_segment("最初です。でも"), _jp_segment("続きです。")]
    elif fate == "tail_discontinuity_cleared":
        # A segment emitted before a pause edge drains after it; its tail
        # can no longer be stored for the new epoch.
        stale = _jp_segment("古いです。続き")
        stale.pause_epoch = 0
        segments = [stale]
        stage_kwargs["remainder_state"] = PendingRemainderState(active_epoch=2)
    elif fate == "pipeline_stage_error":
        stage_kwargs["pipeline"] = _RubyPipeline(raise_for="今日は晴れです。")
    elif fate == "translation_empty":
        translator = _DrainTranslator(en="")
    elif fate == "display_no_clients":
        stage_kwargs["manager"] = ConnectionManager()
    elif fate == "display_all_sends_failed":
        manager = ConnectionManager()
        manager.add(_FailingClient())
        stage_kwargs["manager"] = manager
    elif fate == "display_receipt_unavailable":
        stage_kwargs["manager"] = _BroadcastStub(None)
    elif fate == "display_broadcast_error":
        stage_kwargs["manager"] = _BroadcastStub(RuntimeError("socket gone"))
    else:
        assert fate == "display_not_configured", f"no trigger for {fate}"

    await _translate_until_drained(
        segments, translator, telemetry, **stage_kwargs,
    )

    assert len(_terminal_lines(caplog, fate)) == 1, [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("post_asr terminal:")
    ]
