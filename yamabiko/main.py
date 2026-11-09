"""Pipeline entrypoint: capture -> SileroChunker -> ASR -> translator -> stdout.

Per-stage timings are logged to latency.log for p50/p95 analysis.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from yamabiko import hf_env  # noqa: F401 — side-effect: pin HF_HOME + HF_XET_HIGH_PERFORMANCE
from yamabiko import log_config, loopback_http, paths, wizard_state, hotkeys as hk_mod
from yamabiko import problems
from yamabiko.supervisor import supervise, record_failure, ErrorTracker

# Force UTF-8 stdout/stderr on Windows so Hepburn macrons (ū, ō, ā…) don't
# crash the pipeline when the default console codepage is cp932.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
import uvicorn

from yamabiko.display.broadcast import (
    WS_TOKEN_ENV,
    BroadcastReceipt,
    ConnectionManager,
    make_notice_frame,
    make_show_series_cmd,
    make_toggle_overlay_cmd,
)
from yamabiko.display.browser_server import build_app
from yamabiko.state import AppState, PauseState
from yamabiko.asr.asr_client import ASROutcome, build_asr
from yamabiko.asr.incremental import (
    IncrementalEmitter,
    OpenWindowSource,
    StepJob,
    StepLane,
    StepOutcome,
)
from yamabiko.capture.audio import AudioConfig, LoopbackCapture
from yamabiko.capture.default_endpoint import active_render_endpoint_names
from yamabiko.lifecycle.exit_codes import (
    EXIT_CHILD_GIVEUP,
    EXIT_STAGE_ESCALATION,
    EXIT_STARTUP_FAILURE,
    EXIT_STREAM_DEATH,
)
from yamabiko.jp_text import is_japanese
from yamabiko.translation.prompt import ContextBuffer, fuzzy_suffix_dedup
from yamabiko.translation.translator import Translator, TranslationConfig, health_says_ready
from yamabiko.vad.chunker import chunker_config_from_config
from yamabiko.vad.silero_chunker import FrameGateEmitReason
from yamabiko.romanization.glossary import GlossaryManager
from yamabiko.romanization.romanizer import Romanizer
from yamabiko.romanization.pipeline import RomanizationPipeline
from yamabiko.gazetteer.gazetteer import (
    SeriesGazetteer,
    feature_enabled,
    gazetteer_path,
    load_active_series,
)
from yamabiko.gazetteer.series_profile import context_hint, load_profile
from yamabiko.sentencing.splitter import SentenceSplitter, has_terminator
from yamabiko.sentencing.remainder import PendingRemainderState
from yamabiko.post_asr_telemetry import (
    ArtifactKind,
    ArtifactRef,
    Decision as PostASRDecision,
    Fate as PostASRFate,
    Phase as PostASRPhase,
    PostASRTelemetry,
    Stage as PostASRStage,
)

_PARTIAL_REMAINDER_CAP = 300  # chars; force-flush cap for accumulated partial sentences
_VAD_EMIT_REASON_VALUES = frozenset(r.value for r in FrameGateEmitReason)

# Age bound for a buffered trailing partial (pending_remainder). The
# buffer deliberately SURVIVES dropped non-JA chunks (see the non-JA
# guard in translation_stage), which also meant it survived forever: an
# unterminated tail like 「でも」 could be glued onto the next JP chunk
# minutes or hours later, stitching one subtitle across unrelated
# dialogue. 30 s comfortably covers the real mid-sentence-cut case (the
# continuation arrives within a chunk or two, i.e. seconds) while
# guaranteeing a stale tail can never cross a scene change.
_PENDING_REMAINDER_MAX_AGE_S = 30.0

# Clock seam for the remainder age check. Tests monkeypatch THIS name to
# advance a fake clock; monkeypatching time.monotonic itself would also
# retune the event loop's own timer and corrupt every await in the test.
_monotonic = time.monotonic

# Upper bound for ONE asr.transcribe call on the cuda_pool worker. Warm
# transcribe of a <=16 s chunk takes single-digit seconds; the one
# legitimately slow call -- the FIRST inference after model load -- runs in
# build_pipeline's warmup (asr.warmup, same pool) BEFORE any stage starts,
# so at steady state 120 s only trips on a genuine CUDA/CT2 wedge. Without
# the bound, a hung transcribe left the ASR stage pending FOREVER with
# /livez green (the event loop is fine) and the supervisor blind (a hung
# thread raises nothing), so subtitles stopped permanently with no restart.
_ASR_TRANSCRIBE_TIMEOUT_S = 120.0

# Incremental emission: how often the ASR stage logs its step counters
# (steps_run, steps_dropped, step_overruns, steps_failed). The main log ships
# in diagnostics zips, which is how the counters reach diagnostics.
_STEP_COUNTS_LOG_S = 60.0

# P1-8b: consecutive timeout-class translate() failures that prove the
# translator wedged. Each hang-class failure costs ~61 s at the client
# (30 s httpx timeout x 2 tries + 1 s backoff in Translator._complete),
# so the ErrorTracker's 10-in-60s escalation threshold is mathematically
# unreachable for this failure class: one dropped line per minute forever,
# /v1/models still answering, no restart. Five consecutive timeouts
# (~5 min of a llama-server that accepts connections but never completes)
# escalate directly through the SAME exit-45 path; any successful
# round-trip resets the streak.
_TRANSLATOR_TIMEOUT_ESCALATION_STREAK = 5

# Ultra-short JA fragment guard. A live session showed 1-char ASR fragments
# being "translated" into whole fabricated sentences that echoed the PRIOR
# line (お -> "Yes, Your Highness."; 関 -> a 93-char re-run of the previous
# translation), and 2-char kana fragments confabulating from the rolling EN
# context (ニャ -> "Man, 100-yen lighters!"). is_japanese() passes these, so
# the fragment reaches translate() with the rolling EN context and the model,
# given almost no input signal, mines that context. Two gates below defuse it.
_CONTEXT_STARVE_MAX_LEN = 1  # chars; <= this feeds the translator NO context
_KANA_CONTEXT_STARVE_MAX_LEN = 2  # chars; all-kana content <= this also gets NO context


def _ultra_short_content(text: str) -> str:
    """Transcript content counted by the ultra-short fragment guard.

    Sentence splitting can leave punctuation attached (お。 / 関。). Those marks
    do not add translation signal, so ignore Unicode punctuation when deciding
    whether a segment is a one-character fragment.
    """
    return "".join(
        ch for ch in text.strip()
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _is_kana(ch: str) -> bool:
    """True iff ``ch`` is a single hiragana/katakana (incl. halfwidth) code
    point. Kanji are deliberately EXCLUDED: a lone kanji still carries lexical
    content (雨 -> "Rain") and is handled by the context-starve path, not
    dropped. Ranges mirror yamabiko.jp_text so the two can't diverge."""
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F      # hiragana
        or 0x30A0 <= code <= 0x30FF   # katakana (incl. ー、ヽヾ)
        or 0xFF66 <= code <= 0xFF9F   # halfwidth katakana
    )


def _is_bare_interjection(text: str) -> bool:
    """True for a single bare kana with nothing else (お/え/あ/ん/う …).

    These are filler that carries no translatable content; feeding them to the
    translator (with rich EN context) makes the model echo the previous line.
    Deliberately narrow: 2+ char utterances (はい, うん) never match, so valid
    short replies survive."""
    content = _ultra_short_content(text)
    return len(content) == 1 and _is_kana(content)


def _has_no_content(text: str) -> bool:
    """True when nothing is left once punctuation and whitespace are removed
    (......, ……).

    ASR can open a window with an ellipsis for a pause or a breath
    (......めんどくさい。) and the sentence splitter cuts it off as a sentence
    of its own. Translated, it comes back as "..." and shows as a subtitle of
    dots, so it is dropped with the bare interjections. One character of
    anything else keeps the line (そうか......)."""
    return not _ultra_short_content(text)


def _context_for(text: str, context: list[str]) -> list[str]:
    """Context lines to feed the translator for ``text``.

    Ultra-short inputs get an EMPTY context so the model cannot regurgitate
    or confabulate from the previous EN lines: 1-char content (a lone kanji
    like 関) and all-kana content up to 2 chars (ニャ/おう/はい translate fine
    context-free). 2-char content containing kanji keeps the real rolling
    context -- kanji carries lexical weight worth disambiguating (see the
    _is_kana docstring). The short line is still translated and shown -- it
    just can't echo."""
    content = _ultra_short_content(text)
    if len(content) <= _CONTEXT_STARVE_MAX_LEN:
        return []
    if len(content) <= _KANA_CONTEXT_STARVE_MAX_LEN and all(map(_is_kana, content)):
        return []
    return context

# uvicorn's graceful-drain bound and the outer wait that lets it
# finish are coupled here. The outer wait MUST exceed the drain bound,
# or run_pipeline's finally cancels the server task mid-drain (dropping
# WebSocket close frames -- exactly what timeout_graceful_shutdown
# exists to prevent); and the sum must stay inside
# lifecycle.grace_s.pipeline (10 s) so the watchdog doesn't terminate
# the pipeline mid-teardown.
_UV_GRACEFUL_DRAIN_S = 5
_SERVER_DRAIN_WAIT_S = _UV_GRACEFUL_DRAIN_S + 1.0

# Plan phase 4 (problems): the exit codes the watchdog answers with a
# restart; how long such an exit waits after raising a problem so the
# notification and the overlay line are on screen before teardown stops the
# overlay (R1), inside the budget for the whole announcement; and how long
# a restarted pipeline tells new clients which problems the restart
# resolved.
_RESTART_EXIT_CODES = (EXIT_STREAM_DEATH, EXIT_STAGE_ESCALATION, EXIT_CHILD_GIVEUP)
_ANNOUNCE_HOLD_S = 4.0
_ANNOUNCE_BUDGET_S = 6.0
_CARRIED_RECOVERY_S = 30.0

log = logging.getLogger("yamabiko")


def _reset_stage_tracker() -> ErrorTracker:
    """Return a fresh ErrorTracker.  Called at the top of amain() so each
    invocation of the pipeline gets an independent error-count baseline."""
    return ErrorTracker()


# Module-level sentinel used only outside amain() (e.g. direct stage tests).
# Production code always passes the tracker created inside amain().
_stage_tracker = ErrorTracker()


def _notify_stage_error(stage: str, msg: str) -> None:
    """Logs a supervised stage's error. What a user can act on is reported
    as a problem notice (``PipelineApp.raise_problem``: translator, audio
    device); other stage errors stay in the log."""
    log.warning("stage %s: %s", stage, msg)


@dataclass
class Segment:
    audio: "object"                  # np.ndarray
    t_chunk_emit: float
    t_asr_done: float | None = None
    t_translate_done: float | None = None
    jp: str = ""
    en: str = ""
    jp_ruby: list = field(default_factory=list)  # list[dict] serialized in broadcast
    # True when the chunker force-emitted this chunk at max_chunk_ms
    # (mid-speech cut). Only such a boundary legitimately re-emits the
    # prior tail into the NEXT chunk, so asr_stage gates its transcript
    # dedup on this flag.
    force_split: bool = False
    # True only on the first VAD segment emitted after capture time was lost
    # (including a pause/resume transition). It breaks transcript adjacency
    # and abandons the unfinished Japanese tail buffered before the gap,
    # without affecting translation context. asr_stage carries it onto the
    # next forwarded segment when it drops the flagged one.
    discontinuity_before: bool = False
    # Typed FrameGate reason serialized as its stable string value. Hand-made
    # or third-party Segment producers may leave this unset; SileroChunker
    # always stamps it when the gate emits audio.
    vad_emit_reason: str | None = None
    # Content-free retained-state summary immediately after the VAD emit.
    # A dip cut leaves its right piece open; all true closes retain zero.
    vad_gate_open_after_emit: bool = False
    vad_retained_frames_after_emit: int = 0
    # Content-free correlation coordinate for post-ASR lineage telemetry.
    # None preserves compatibility with hand-built and third-party segments.
    post_asr_artifact_seq: int | None = None
    # Capture/pause discontinuity epoch. Producers that do not participate in
    # the pause-boundary contract retain legacy behavior via None.
    pause_epoch: int | None = None
    # The chunker's id of the VAD window this text came from; the ASR stage
    # matches a window's final decode to its step decodes by it
    # (yamabiko/asr/incremental.py). None for producers that do not number
    # windows.
    window_id: int | None = None
    # Incremental emission. early: a sentence committed while its window was
    # still open. follows_early: a window's final remainder after early
    # sentences of the same window; _assemble_sentences then holds its
    # unfinished last sentence as it would for a multi-sentence window.
    early: bool = False
    follows_early: bool = False


@dataclass
class LatencyStats:
    asr_ms: list[float] = field(default_factory=list)
    translate_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    def record(self, seg: Segment) -> None:
        if seg.t_asr_done is not None:
            self.asr_ms.append((seg.t_asr_done - seg.t_chunk_emit) * 1000)
        if seg.t_translate_done is not None and seg.t_asr_done is not None:
            self.translate_ms.append((seg.t_translate_done - seg.t_asr_done) * 1000)
            self.total_ms.append((seg.t_translate_done - seg.t_chunk_emit) * 1000)

    def summary(self) -> dict:
        def pct(xs: list[float], p: float) -> float:
            if not xs:
                return 0.0
            s = sorted(xs)
            k = int(round((len(s) - 1) * p))
            return s[k]
        return {
            "count": len(self.total_ms),
            "asr_p50_ms": pct(self.asr_ms, 0.5),
            "asr_p95_ms": pct(self.asr_ms, 0.95),
            "translate_p50_ms": pct(self.translate_ms, 0.5),
            "translate_p95_ms": pct(self.translate_ms, 0.95),
            "total_p50_ms": pct(self.total_ms, 0.5),
            "total_p95_ms": pct(self.total_ms, 0.95),
            "asr_mean_ms": statistics.fmean(self.asr_ms) if self.asr_ms else 0.0,
            "translate_mean_ms": statistics.fmean(self.translate_ms) if self.translate_ms else 0.0,
            "total_mean_ms": statistics.fmean(self.total_ms) if self.total_ms else 0.0,
        }


def _load_config(explicit: Path | None) -> dict:
    """Load the effective merged config.

    ``explicit`` is a dev/repro escape hatch. When provided, that YAML
    file is layered over the shipped defaults with the SAME unknown-key
    and type validation as a normal load, but the ``user_overrides.yaml``
    layer is DELIBERATELY SKIPPED — the point of ``--config`` is an
    isolated, reproducible config that does not pick up ambient user
    state. A missing file or parse error surfaces as ``RuntimeError``, so
    build_pipeline's startup handler shows a tailored dialog + exit 43
    instead of the watchdog restart-looping an uncaught crash. In
    production (``explicit=None``) the standard layered load (shipped
    defaults + sparse user overrides) runs.
    """
    from .config import load_full_config
    return load_full_config(explicit_path=explicit)


async def asr_stage(in_q: asyncio.Queue, out_q: asyncio.Queue, asr, cuda_pool: ThreadPoolExecutor, sample_rate: int,
                    *, tracker: ErrorTracker | None = None,
                    on_escalate: "Callable[[str], None] | None" = None,
                    transcribe_timeout_s: float = _ASR_TRANSCRIBE_TIMEOUT_S,
                    post_asr_telemetry: PostASRTelemetry | None = None,
                    open_windows: "OpenWindowSource | None" = None,
                    incremental_enabled: "Callable[[], bool]" = lambda: False) -> None:
    """Decode closed VAD windows (finals) and, with ``open_windows``, step
    decodes of the open window (yamabiko/asr/incremental.py).

    One dispatcher drives the single-worker ``cuda_pool``: a waiting final
    always runs first, then the step lane's job. ``incremental_enabled`` is
    read when a window is first seen, so the switch takes effect at the next
    window. Without ``open_windows`` the stage decodes finals only, exactly as
    before incremental emission.
    """
    loop = asyncio.get_running_loop()
    prev_transcript = ""
    # True when the PREVIOUS chunk was force-split at max_chunk_ms. Only
    # that boundary genuinely re-emits the prior tail; running the dedup
    # across a silence-terminated boundary compares textually-unrelated
    # utterances and strips coincidental aizuchi/echo prefixes.
    prev_forced = False
    # A dropped segment (empty transcript, dedup-empty, transcribe failure)
    # must not swallow its capture-gap flag: the next forwarded segment is
    # the first text after the gap.
    pending_discontinuity = False
    _tracker = tracker if tracker is not None else _stage_tracker
    emitter = (
        IncrementalEmitter(SentenceSplitter(), sample_rate)
        if open_windows is not None else None
    )
    lane = StepLane(emitter.step_samples) if emitter is not None else None
    step_counts: Counter[str] = Counter()
    counts_logged_at = time.monotonic()
    counts_last = (0, 0, 0, 0)
    # Post-ASR telemetry window_seq of each window that already sent an item.
    window_seqs: dict[int, int] = {}

    async def _one_final(seg: Segment) -> None:
        nonlocal prev_transcript, prev_forced, pending_discontinuity
        had_commits = emitter is not None and emitter.has_commits(seg.window_id)
        if seg.discontinuity_before and not had_commits:
            prev_transcript = ""
            prev_forced = False
        # Bound the transcribe await (P1-8a). A CUDA/CT2 wedge leaves the
        # executor future pending FOREVER: /livez stays green (the event
        # loop is healthy), the supervisor sees nothing (a hung thread
        # raises nothing, so the 10-in-60s tracker can never fire), and
        # because cuda_pool is single-worker every queued transcribe is
        # poisoned behind the hung one -- in-process recovery is
        # impossible. wait_for cancels only the asyncio WRAPPER at the
        # deadline (the worker thread itself is unstoppable and is
        # abandoned in the pool), which is exactly why the only correct
        # response is prompt escalation to the watchdog restart path.
        t_asr_start = time.perf_counter()
        window_seq = None
        if post_asr_telemetry is not None:
            window_seq = window_seqs.pop(seg.window_id, None)
            if window_seq is None:
                window_seq = post_asr_telemetry.new_window()
        if seg.window_id is not None:
            # Windows abandoned at a pause or capture gap never get a final.
            for key in [key for key in window_seqs if key <= seg.window_id]:
                del window_seqs[key]
        fut = loop.run_in_executor(
            cuda_pool, asr.transcribe, seg.audio, sample_rate,
        )
        try:
            result = await asyncio.wait_for(fut, timeout=transcribe_timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            # Both spellings on purpose: identical on 3.11+ (the bundle
            # ships 3.12) but distinct classes on the 3.10 floor.
            # Failed chunk either way: the NEXT boundary must skip the
            # dedup rather than compare against the chunk TWO back (the
            # non-adjacent-boundary corruption the prev_forced gate
            # exists to prevent). Same reset as the except below.
            prev_transcript = ""
            prev_forced = False
            # Wedge-vs-quick-error discrimination (same trap as
            # _stop_capture_bounded: on 3.11+ asyncio.TimeoutError IS the
            # builtin, and wait_for re-raises worker exceptions). On a
            # genuine deadline hit wait_for CANCELLED the wrapper future;
            # a fut that is DONE instead means asr.transcribe itself
            # raised TimeoutError quickly -- a finished call, no hung
            # thread, no poisoned pool -- so treat it as an ordinary
            # stage error (re-raise into supervise) rather than
            # restarting the whole pipeline over one chunk.
            if not fut.cancelled():
                # A DONE future carries the worker's OWN outcome: the
                # deadline handle fired after asr.transcribe had already
                # finished, so re-raising the deadline here would both
                # mislabel the failure and leave the worker's exception
                # unretrieved (the loop's never-retrieved handler then
                # logs it out of context). Re-raise what actually
                # happened; fall back to the deadline when the future
                # holds no exception.
                worker_exc = fut.exception() if fut.done() else None
                if worker_exc is not None:
                    raise worker_exc
                raise
            log.critical(
                "asr transcribe exceeded %.0fs -- CUDA/CT2 call presumed "
                "wedged. cuda_pool is single-worker, so every queued ASR "
                "call is now poisoned behind the hung thread; in-process "
                "recovery is impossible. Escalating directly for a "
                "watchdog RESTART (exit %d) -- the 10-in-60s tracker is "
                "unreachable here because a hung thread produces no "
                "further error events.",
                transcribe_timeout_s, EXIT_STAGE_ESCALATION,
            )
            if on_escalate is not None:
                on_escalate("asr")
            return
        except Exception:
            # P2 (2026-07-12 review): a transcribe exception used to be
            # swallowed by supervise() BEFORE the prev_forced /
            # prev_transcript update below ran, so the NEXT chunk was
            # deduped against the chunk TWO back and could lose a
            # coincidentally-matching >=3-char prefix. Reset the dedup
            # state so the next boundary skips dedup instead of using
            # stale state, then re-raise into the supervisor as before.
            prev_transcript = ""
            prev_forced = False
            raise
        seg.t_asr_done = time.perf_counter()
        reported_outcome = getattr(result, "outcome", None)
        outcome = (
            reported_outcome.value
            if isinstance(reported_outcome, ASROutcome)
            else "untyped_backend_result"
        )
        reported_vad_reason = seg.vad_emit_reason
        vad_emit_reason = (
            reported_vad_reason
            if (
                isinstance(reported_vad_reason, str)
                and reported_vad_reason in _VAD_EMIT_REASON_VALUES
            )
            else "unavailable"
        )
        reported_retained_frames = seg.vad_retained_frames_after_emit
        retained_frames_after = (
            reported_retained_frames
            if (
                type(reported_retained_frames) is int
                and reported_retained_frames >= 0
            )
            else -1
        )
        log.info(
            "asr window: run_id=%s window_seq=%s outcome=%s "
            "input_ms=%d stage_ms=%d "
            "force_split=%s vad_emit_reason=%s gate_open_after=%s "
            "retained_frames_after=%d discontinuity_before=%s queue_ms=%d",
            (
                post_asr_telemetry.run_id
                if post_asr_telemetry is not None
                else "unavailable"
            ),
            window_seq if window_seq is not None else "unavailable",
            outcome,
            round(len(seg.audio) * 1000 / sample_rate),
            round((seg.t_asr_done - t_asr_start) * 1000),
            str(bool(seg.force_split)).lower(),
            vad_emit_reason,
            str(bool(seg.vad_gate_open_after_emit)).lower(),
            retained_frames_after,
            str(bool(seg.discontinuity_before)).lower(),
            # How long the closed window waited for the executor.
            round((t_asr_start - seg.t_chunk_emit) * 1000),
        )
        rest = result.text
        if emitter is not None:
            rest, had_commits = emitter.on_final(seg.window_id, result.text)
        # A raw empty ASR result breaks textual adjacency. In particular,
        # a force-split empty chunk must not arm dedup against the last
        # non-empty transcript on the following chunk.
        if not result.text.strip():
            prev_transcript = ""
            prev_forced = False
            return
        if had_commits:
            # This window's first sentences went out early and took the
            # force-split dedup with them; only the rest of the final goes
            # on, marked so _assemble_sentences may hold its trailing partial.
            prev_transcript = result.text
            prev_forced = seg.force_split
            if not rest.strip():
                return
            if post_asr_telemetry is not None and window_seq is not None:
                artifact_ref = post_asr_telemetry.root(window_seq)
                post_asr_telemetry.decision(
                    artifact_ref, PostASRStage.DEDUP, PostASRDecision.NOT_APPLICABLE,
                )
                seg.post_asr_artifact_seq = artifact_ref.artifact_seq
            seg.jp = rest
            seg.follows_early = True
            seg.discontinuity_before = pending_discontinuity
            pending_discontinuity = False
            await out_q.put(seg)
            return
        artifact_ref = (
            post_asr_telemetry.root(window_seq)
            if post_asr_telemetry is not None and window_seq is not None
            else None
        )
        dedup_applied = prev_forced
        try:
            if dedup_applied:
                text = fuzzy_suffix_dedup(prev_transcript, result.text)
            else:
                text = result.text
        except Exception:
            if post_asr_telemetry is not None and artifact_ref is not None:
                post_asr_telemetry.terminal(
                    artifact_ref,
                    PostASRFate.PIPELINE_STAGE_ERROR,
                    PostASRPhase.DEDUP,
                )
            raise
        # If dedup alone emptied the text, keep the prior transcript: it
        # still contains the duplicated tail needed at the next boundary.
        prev_forced = seg.force_split
        if not text.strip():
            if post_asr_telemetry is not None and artifact_ref is not None:
                post_asr_telemetry.terminal(
                    artifact_ref, PostASRFate.DEDUP_EMPTY,
                )
            return
        if post_asr_telemetry is not None and artifact_ref is not None:
            if not dedup_applied:
                dedup_decision = PostASRDecision.NOT_APPLICABLE
            elif text == result.text:
                dedup_decision = PostASRDecision.UNCHANGED
            else:
                dedup_decision = PostASRDecision.REDUCED
            post_asr_telemetry.decision(
                artifact_ref, PostASRStage.DEDUP, dedup_decision,
            )
        prev_transcript = result.text
        seg.jp = text
        if artifact_ref is not None:
            seg.post_asr_artifact_seq = artifact_ref.artifact_seq
        seg.discontinuity_before = pending_discontinuity
        pending_discontinuity = False
        await out_q.put(seg)

    async def _process_final(seg: Segment) -> None:
        nonlocal pending_discontinuity
        if lane is not None:
            # A lane job must never outlive its window's step state (on_step
            # raises KeyError for a dropped window). This final drops the
            # state of its window and every earlier one, so their jobs go
            # first; the dispatcher offers none until this returns.
            lane.discard_through(seg.window_id)
        # A window that sent early sentences already carried its capture-gap
        # flag on the first of them.
        if not (emitter is not None and emitter.has_commits(seg.window_id)):
            pending_discontinuity |= bool(seg.discontinuity_before)
        await supervise(
            "asr", lambda: _one_final(seg),
            notify=_notify_stage_error,
            tracker=_tracker,
            on_escalate=on_escalate,
        )
        if emitter is not None:
            emitter.drop_through(seg.window_id)

    def _step_failed(error: BaseException) -> None:
        # Counts and the exception type only: no transcript text, no message.
        step_counts["steps_failed"] += 1
        log.warning(
            "asr step failed: error=%s steps_failed=%d",
            type(error).__name__, step_counts["steps_failed"],
        )

    async def _run_step(job: StepJob) -> None:
        t_start = time.perf_counter()
        fut = loop.run_in_executor(cuda_pool, asr.transcribe, job.audio, sample_rate)
        try:
            result = await asyncio.wait_for(fut, timeout=transcribe_timeout_s)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            if fut.cancelled():
                # The same wedge as a final's (see _one_final): the
                # single-worker pool is poisoned, so escalate for a restart.
                log.critical(
                    "asr step transcribe exceeded %.0fs -- CUDA/CT2 call "
                    "presumed wedged; escalating for a watchdog RESTART "
                    "(exit %d)",
                    transcribe_timeout_s, EXIT_STAGE_ESCALATION,
                )
                if on_escalate is not None:
                    on_escalate("asr")
                return
            # A DONE future: the decode finished, so report (and retrieve)
            # what the worker raised, as _one_final does.
            worker_exc = fut.exception() if fut.done() else None
            _step_failed(worker_exc if worker_exc is not None else exc)
            return
        except Exception as exc:
            _step_failed(exc)
            return
        t_done = time.perf_counter()
        step_counts["steps_run"] += 1
        # on_step is applied before the dispatcher's next due_step, and only
        # for the latest job issued for its window (the lane keeps the
        # latest; no due_step runs while a step decodes): an older job
        # applied after a newer one was issued would move the window's due
        # time backwards. A failure applying or forwarding the step's result
        # is logged and skips the step. If _forward_early fails partway, the
        # window's final is affected: on_step has already recorded the
        # commits, so on_final strips the sentences not yet forwarded and
        # they are lost, with any capture-gap flag the first of them carried.
        # Nothing in its loop can raise today (string dedup, telemetry
        # argument checks).
        try:
            outcome = emitter.on_step(job, result.text)
            if (t_done - t_start) * sample_rate > outcome.interval_samples:
                step_counts["step_overruns"] += 1
            await _forward_early(job, outcome, t_done)
        except Exception as exc:
            _step_failed(exc)

    async def _forward_early(job: StepJob, outcome: StepOutcome, t_done: float) -> None:
        """Put a step's committed sentences on out_q, one item each."""
        nonlocal prev_transcript, prev_forced, pending_discontinuity
        discontinuity = pending_discontinuity or (
            job.discontinuity_before and outcome.first_commits
        )
        dedup_first = False
        if outcome.first_commits:
            if job.discontinuity_before:
                # Same reset as a final after a capture gap: no dedup across it.
                prev_transcript = ""
                prev_forced = False
            dedup_first = prev_forced
        for index, sentence in enumerate(outcome.commits):
            artifact_ref = None
            if post_asr_telemetry is not None:
                window_seq = window_seqs.get(job.window_id)
                if window_seq is None:
                    window_seq = post_asr_telemetry.new_window()
                    window_seqs[job.window_id] = window_seq
                artifact_ref = post_asr_telemetry.root(window_seq)
            dedup_applied = outcome.first_commits and index == 0 and dedup_first
            text = fuzzy_suffix_dedup(prev_transcript, sentence) if dedup_applied else sentence
            if not text.strip():
                if artifact_ref is not None:
                    post_asr_telemetry.terminal(artifact_ref, PostASRFate.DEDUP_EMPTY)
                continue
            if artifact_ref is not None:
                if not dedup_applied:
                    decision = PostASRDecision.NOT_APPLICABLE
                elif text == sentence:
                    decision = PostASRDecision.UNCHANGED
                else:
                    decision = PostASRDecision.REDUCED
                post_asr_telemetry.decision(artifact_ref, PostASRStage.DEDUP, decision)
            await out_q.put(Segment(
                audio=job.audio,
                t_chunk_emit=job.t_snapshot,
                t_asr_done=t_done,
                jp=text,
                discontinuity_before=discontinuity,
                post_asr_artifact_seq=(
                    artifact_ref.artifact_seq if artifact_ref is not None else None
                ),
                pause_epoch=job.pause_epoch,
                window_id=job.window_id,
                early=True,
            ))
            discontinuity = False
            pending_discontinuity = False

    def _log_step_counts() -> None:
        nonlocal counts_logged_at, counts_last
        now = time.monotonic()
        if now - counts_logged_at < _STEP_COUNTS_LOG_S:
            return
        counts_logged_at = now
        current = (
            step_counts["steps_run"], lane.steps_dropped,
            step_counts["step_overruns"], step_counts["steps_failed"],
        )
        if current != counts_last:
            counts_last = current
            log.info(
                "asr steps: steps_run=%d steps_dropped=%d step_overruns=%d "
                "steps_failed=%d",
                *current,
            )

    # The dispatcher runs outside supervise(): an exception here would end
    # the stage silently, so it escalates for a watchdog restart instead
    # (same shape as _supervised_chunker).
    try:
        while True:
            if emitter is not None:
                # due_step reads open_window_id and the window's snapshot on
                # this event loop with no await between them, so the two agree.
                job = emitter.due_step(open_windows, incremental_enabled)
                if job is not None:
                    lane.offer(job)
                _log_step_counts()
            if not in_q.empty():
                await _process_final(in_q.get_nowait())
                continue
            if lane is not None:
                # Due times count decided samples from the window's opening
                # frame (after a dip cut, from where speech came back); the
                # stale check reads the same clock and never re-bases them.
                job = lane.take(open_windows.decided_samples, open_windows.open_window_id)
                if job is not None:
                    await _run_step(job)
                    continue
            if emitter is None:
                seg = await in_q.get()
            else:
                try:
                    seg = await asyncio.wait_for(in_q.get(), emitter.seconds_until_due(open_windows))
                except (TimeoutError, asyncio.TimeoutError):
                    continue
            await _process_final(seg)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("asr dispatcher crashed; escalating for a watchdog restart")
        if on_escalate is not None:
            on_escalate("asr")


@dataclass(frozen=True)
class AssembledSegment:
    """One transcript segment after tail joining, language policy and split.

    ``sentences`` is empty when the segment was dropped (its fate is already
    recorded); ``sentence_refs`` holds each sentence's lineage reference.
    The tail flags report a force-flushed or newly buffered trailing partial.
    """

    sentences: tuple[str, ...]
    sentence_refs: tuple[ArtifactRef | None, ...]
    current_ref: ArtifactRef | None
    force_flushed_tail: bool = False
    tail_buffered: bool = False


def _assemble_sentences(
    seg: Segment, remainder_state: PendingRemainderState,
    splitter: SentenceSplitter, now: float, *,
    telemetry: PostASRTelemetry | None, incoming_ref: ArtifactRef | None,
    transcript_enabled: bool = False,
) -> AssembledSegment:
    """Join the unfinished tail onto ``seg`` and split it into sentences.

    Synchronous on purpose: the tail's capture-gap and expiry rules, the
    Japanese-language policy, the split and the trailing-partial decision
    run without yielding, so a pause edge cannot interleave with them.
    """

    def _terminal(ref, fate, phase=None) -> None:
        if telemetry is not None and ref is not None:
            telemetry.terminal(ref, fate, phase)

    segment_epoch = (
        seg.pause_epoch
        if seg.pause_epoch is not None
        else remainder_state.active_epoch
    )
    if seg.discontinuity_before:
        # Capture time was lost before this segment. The buffered
        # tail belongs to speech before the gap; joining it to text
        # after the gap would stitch a subtitle across missing audio.
        _terminal(
            remainder_state.abandon_for(segment_epoch),
            PostASRFate.TAIL_DISCONTINUITY_CLEARED,
        )
    prefix_len = (
        len(remainder_state.text)
        if segment_epoch == remainder_state.active_epoch
        else 0
    )
    prefix_age = now - remainder_state.buffered_at if prefix_len else 0.0
    tail_prefix = remainder_state.prefix_for(
        segment_epoch, now, _PENDING_REMAINDER_MAX_AGE_S,
    )
    if tail_prefix.expired:
        # The buffered tail survived the non-JA guard below ON PURPOSE (a
        # dropped hallucination chunk must not delete a valid JP partial),
        # so without an age bound it survived FOREVER: 「でも」 buffered at
        # minute 3 would be glued onto the first JP chunk of minute 40.
        # Past the bound the sentence it belonged to is long gone --
        # discard it rather than stitch a subtitle across unrelated
        # dialogue. Length only; no transcript content in default logs.
        log.debug(
            "discarding stale pending remainder (age %.1fs > %.0fs, "
            "len=%d)",
            prefix_age,
            _PENDING_REMAINDER_MAX_AGE_S, prefix_len,
        )
        _terminal(tail_prefix.artifact_ref, PostASRFate.TAIL_EXPIRED)
    raw = tail_prefix.text + seg.jp

    try:
        japanese_accepted = is_japanese(raw)
    except Exception:
        _terminal(
            incoming_ref,
            PostASRFate.PIPELINE_STAGE_ERROR,
            PostASRPhase.LANGUAGE_POLICY,
        )
        raise
    if not japanese_accepted:
        # The combined text is predominantly non-JA (an English ASR
        # hallucination in seg.jp). Do NOT clear the shared remainder:
        # a buffered JP partial already passed this guard when it was
        # stored, so discarding it silently loses a valid sentence. Keep
        # it buffered for the next chunk; only the new non-JA chunk is
        # dropped. (When pending_remainder is empty this is the original
        # behaviour.) The legitimate short-Latin-continuation case still
        # works because the guard runs on the COMBINED text, not seg.jp.
        #
        # Length always; the raw snippet only behind the SAME consent
        # flag that gates latency.log's raw text
        # (logging.transcript_log_enabled). Default logs must carry
        # no transcript content -- bug_report.yml promises it.
        log.info(
            "dropped non-JA segment (len=%d)%s", len(raw),
            f": {raw[:60]!r}" if transcript_enabled else "",
        )
        _terminal(incoming_ref, PostASRFate.LANGUAGE_POLICY_REJECTED)
        return AssembledSegment((), (), incoming_ref)
    current_ref = incoming_ref
    if (
        tail_prefix.text
        and telemetry is not None
        and tail_prefix.artifact_ref is not None
        and incoming_ref is not None
    ):
        telemetry.decision(
            tail_prefix.artifact_ref,
            PostASRStage.TAIL,
            PostASRDecision.TAIL_CONSUMED,
        )
        current_ref = telemetry.merge(tail_prefix.artifact_ref, incoming_ref)
    if telemetry is not None and current_ref is not None:
        telemetry.decision(
            current_ref,
            PostASRStage.LANGUAGE_POLICY,
            PostASRDecision.LANGUAGE_POLICY_ACCEPTED,
        )
    remainder_state.clear_for(segment_epoch)

    # Only buffer a trailing partial when the chunk also carried at least
    # one complete sentence -- that's the real mid-sentence-cut case. A
    # window's final remainder after early sentences of the same window
    # (follows_early) is such a chunk too, even when it holds one sentence.
    # If the whole chunk is a single partial, translate it as-is: Whisper
    # often omits the terminal 。 even on complete utterances, and
    # buffering those would stall the pipeline indefinitely. A dots-only
    # sentence (a leading ......) doesn't count as a sentence here.
    try:
        sentences = splitter.split(raw)
        trailing_partial = (
            sentences
            and (
                sum(1 for s in sentences if not _has_no_content(s)) > 1
                or seg.follows_early
            )
            and not has_terminator(sentences[-1])
        )
    except Exception:
        _terminal(
            current_ref,
            PostASRFate.PIPELINE_STAGE_ERROR,
            PostASRPhase.SENTENCE_SPLIT,
        )
        raise
    if not sentences:
        _terminal(current_ref, PostASRFate.SENTENCE_SPLIT_EMPTY)
        return AssembledSegment((), (), current_ref)
    tail_to_store: str | None = None
    force_flushed_tail = False
    if trailing_partial:
        tail = sentences.pop()
        if (
            len(tail) > _PARTIAL_REMAINDER_CAP
            and segment_epoch == remainder_state.active_epoch
        ):
            # Force-flush: prefix with [...] and translate as-is.
            # Only for the active epoch: a stale segment's tail is
            # speech from before a pause edge, so it goes through
            # store_for below, which rejects it.
            sentences.append("[...] " + tail)
            force_flushed_tail = True
        else:
            tail_to_store = tail

    if telemetry is not None and current_ref is not None:
        sentence_refs: list[ArtifactRef | None] = [
            telemetry.child(current_ref, ArtifactKind.SENTENCE, ordinal)
            for ordinal in range(len(sentences))
        ]
        if force_flushed_tail and sentence_refs[-1] is not None:
            telemetry.decision(
                sentence_refs[-1],
                PostASRStage.TAIL,
                PostASRDecision.TAIL_FORCE_FLUSHED,
            )
    else:
        sentence_refs = [None] * len(sentences)
    tail_buffered = False
    if tail_to_store is not None:
        # The tail's lineage reference travels with its text, so the
        # path that ends the tail (consume, expiry, pause, capture
        # gap) can record its fate.
        tail_ref = (
            telemetry.child(current_ref, ArtifactKind.TAIL, len(sentences))
            if telemetry is not None and current_ref is not None
            else None
        )
        tail_buffered = remainder_state.store_for(
            tail_to_store, segment_epoch, now, artifact_ref=tail_ref,
        )
        if not tail_buffered:
            _terminal(tail_ref, PostASRFate.TAIL_DISCONTINUITY_CLEARED)
        elif telemetry is not None and tail_ref is not None:
            telemetry.decision(
                tail_ref, PostASRStage.TAIL, PostASRDecision.TAIL_BUFFERED,
            )
    return AssembledSegment(
        tuple(sentences),
        tuple(sentence_refs),
        current_ref,
        force_flushed_tail=force_flushed_tail,
        tail_buffered=tail_buffered,
    )


class _SegmentFates:
    """Records each sentence's terminal post-ASR fate at most once."""

    def __init__(self, telemetry: PostASRTelemetry | None,
                 sentence_refs: tuple[ArtifactRef | None, ...]) -> None:
        self._telemetry = telemetry
        self.sentence_refs = sentence_refs
        self._recorded: set[int] = set()

    def terminal_once(self, i: int, fate: PostASRFate,
                      phase: PostASRPhase | None = None, **counts: int) -> None:
        ref = self.sentence_refs[i]
        if i in self._recorded or self._telemetry is None or ref is None:
            return
        self._telemetry.terminal(ref, fate, phase, **counts)
        self._recorded.add(i)

    def fail_remaining(self, i: int, phase: PostASRPhase) -> None:
        """Sentences after ``i`` are never processed once ``i`` fails."""
        for later in range(i + 1, len(self.sentence_refs)):
            self.terminal_once(later, PostASRFate.PIPELINE_STAGE_ERROR, phase)


@dataclass
class _SeriesHint:
    """The series-context prompt line ("" = no Series line) and the series
    epoch it was composed for. The epoch starts None (not 0) so a
    boot-restored series -- whose set_series already bumped the epoch
    before the stage started -- still gets its hint on the first sentence.
    """

    text: str = ""
    epoch: int | None = None


async def _enrich_sentence(
    sub_seg: Segment, index: int, fates: _SegmentFates, *, state: AppState,
    pipeline: RomanizationPipeline | None, series_hint: _SeriesHint | None,
) -> None:
    """Apply the genre glossary, series gazetteer, furigana and series hint
    (``series_hint`` None = disabled). A failure records the stage error
    for sentence ``index`` and every later one, then propagates."""
    phase = PostASRPhase.ROMANIZATION_GLOSSARY
    try:
        if pipeline is not None:
            current = pipeline._glossary.current_genre()
            if current != state.genre:
                # Disk I/O and furigana work stay off the loop.
                await asyncio.to_thread(pipeline._glossary.set_genre, state.genre)
            if state.series_epoch != pipeline.applied_series_epoch():
                # A set_series command (or boot restore) changed the active
                # series -- or RE-picked the same one, which must also reload
                # (the cache may have just been built). Epoch, not media-id
                # inequality, is the trigger. Missing cache -> clear hints;
                # the epoch is recorded either way so this can't retry every
                # sentence. Mirrors the genre reload above.
                _epoch = state.series_epoch
                _new_sid = state.series_media_id
                _new_gaz = None
                if _new_sid is not None:
                    _new_gaz = await asyncio.to_thread(
                        SeriesGazetteer.load, gazetteer_path(_new_sid),
                    )
                pipeline.set_gazetteer(_new_gaz, _new_sid, epoch=_epoch)
            ruby = await asyncio.to_thread(pipeline.build_ruby, sub_seg.jp)
            sub_seg.jp_ruby = [
                {"text": r.text, **({"reading": r.reading} if r.reading else {})}
                for r in ruby
            ]
        phase = PostASRPhase.SERIES_CONTEXT
        if series_hint is not None and state.series_epoch != series_hint.epoch:
            # Recompute the series prompt hint on the SAME trigger as the
            # gazetteer reload above: epoch change, never media-id inequality
            # (a re-pick of the same series must pick up a profile written
            # since). Independent of `pipeline` -- the hint must work with
            # romanization disabled. The epoch is recorded even on a miss so
            # this can't re-read disk every sentence. profile-<id>.json is
            # written by the palette pick flow; a series without one hints
            # nothing.
            series_hint.epoch = state.series_epoch
            series_hint.text = ""
            if state.series_media_id is not None:
                # Disk read -- off the event loop, like the gazetteer load
                # above.
                _profile = await asyncio.to_thread(
                    load_profile, state.series_media_id,
                )
                if _profile is not None:
                    series_hint.text = context_hint(_profile)
    except Exception:
        fates.terminal_once(index, PostASRFate.PIPELINE_STAGE_ERROR, phase)
        fates.fail_remaining(index, phase)
        raise


def _display_fate(receipt: object) -> tuple[PostASRFate, dict[str, int]]:
    """Terminal fate and client counts for one subtitle broadcast receipt."""
    if not isinstance(receipt, BroadcastReceipt):
        return PostASRFate.DISPLAY_RECEIPT_UNAVAILABLE, {}
    counts = {"attempted": receipt.attempted, "accepted": receipt.accepted,
              "failed": receipt.failed}
    if receipt.accepted > 0:
        return PostASRFate.DISPLAY_SEND_COMPLETED, counts
    if receipt.attempted == 0:
        return PostASRFate.DISPLAY_NO_CLIENTS, counts
    return PostASRFate.DISPLAY_ALL_SENDS_FAILED, counts


async def translation_stage(
    in_q: asyncio.Queue,
    translator: Translator,
    ctx: ContextBuffer,
    state: AppState,
    stats: LatencyStats,
    latency_log: "log_config._SizeRotatingTextIO | None",
    max_segments: int,
    done: asyncio.Event,
    manager: ConnectionManager | None = None,
    pipeline=None,
    *,
    tracker: ErrorTracker | None = None,
    on_escalate: "Callable[[str], None] | None" = None,
    transcript_enabled: bool = False,
    series_hint_enabled: bool = False,
    post_asr_telemetry: PostASRTelemetry | None = None,
    remainder_state: PendingRemainderState | None = None,
    on_translated: "Callable[[], Awaitable[None]] | None" = None,
) -> None:
    splitter = SentenceSplitter()
    if remainder_state is None:
        pause_probe = getattr(state, "pause_snapshot", None)
        active_epoch = pause_probe().epoch if callable(pause_probe) else 0
        remainder_state = PendingRemainderState(active_epoch=active_epoch)
    processed = 0
    # P1-8b: consecutive timeout-class translate() failures. See
    # _TRANSLATOR_TIMEOUT_ESCALATION_STREAK for why the ErrorTracker's
    # 10-in-60s window can never catch a wedged-but-accepting llama-server
    # and this dedicated streak escalates the same exit-45 path instead.
    consecutive_timeouts = 0
    # Series-context prompt hint (translation.series_context_hint, default
    # OFF -- the A/B gate is pending).
    series_hint = _SeriesHint() if series_hint_enabled else None
    # Series epoch the context lines belong to. A series pick, re-pick or
    # Clear bumps it, and the next segment starts with an empty context.
    ctx_series_epoch = state.series_epoch
    _tracker = tracker if tracker is not None else _stage_tracker
    while True:
        # Check the exit sentinel BEFORE blocking on the queue so a done signal
        # that was set inside _one_iter (even if _one_iter later raised) is never
        # silently swallowed.
        if done.is_set():
            return
        seg: Segment = await in_q.get()

        async def _one_iter() -> bool:
            """Process one incoming transcript segment.

            Returns True when max_segments has been reached and the
            translation stage should exit. Mutates `processed` and the
            timeout streak via nonlocal. The unfinished tail and its
            lineage reference are owned by `remainder_state`.
            """
            nonlocal processed, consecutive_timeouts, ctx_series_epoch
            if state.series_epoch != ctx_series_epoch:
                ctx.clear()
                ctx_series_epoch = state.series_epoch
            if post_asr_telemetry is not None:
                incoming_ref = (
                    ArtifactRef(seg.post_asr_artifact_seq)
                    if seg.post_asr_artifact_seq is not None
                    else post_asr_telemetry.adopt_legacy()
                )
            else:
                incoming_ref = None
            assembled = _assemble_sentences(
                seg, remainder_state, splitter, _monotonic(),
                telemetry=post_asr_telemetry,
                incoming_ref=incoming_ref,
                transcript_enabled=transcript_enabled,
            )
            fates = _SegmentFates(post_asr_telemetry, assembled.sentence_refs)

            for sentence_index, (sentence, sentence_ref) in enumerate(
                zip(assembled.sentences, assembled.sentence_refs),
            ):
                # The step in progress. A failure while it is set records a
                # stage error for this sentence and every later one; None
                # marks a step that records its own fate or none at all.
                phase: PostASRPhase | None = PostASRPhase.DISPLAY
                try:
                    no_content = _has_no_content(sentence)
                    bare_interjection = _is_bare_interjection(sentence)
                    phase = None
                    if no_content or bare_interjection:
                        # Nothing to translate: punctuation only (......) or
                        # a lone kana (お/え/あ/ん/う). Translating either
                        # shows a subtitle of dots or makes the model echo
                        # the previous line. Length always; the raw snippet
                        # only behind the transcript-consent flag -- same
                        # contract as the non-JA / empty-EN drop lines.
                        log.info(
                            "dropped %s (len=%d)%s",
                            "contentless line" if no_content
                            else "bare interjection",
                            len(sentence),
                            f": {sentence[:60]!r}" if transcript_enabled else "",
                        )
                        fates.terminal_once(
                            sentence_index,
                            PostASRFate.DISPLAY_POLICY_INTERJECTION,
                        )
                        continue
                    sub_seg = Segment(
                        audio=seg.audio,
                        t_chunk_emit=seg.t_chunk_emit,
                        t_asr_done=seg.t_asr_done,
                        jp=sentence,
                        post_asr_artifact_seq=(
                            sentence_ref.artifact_seq
                            if sentence_ref is not None
                            else None
                        ),
                        pause_epoch=seg.pause_epoch,
                    )
                    await _enrich_sentence(
                        sub_seg, sentence_index, fates,
                        state=state, pipeline=pipeline, series_hint=series_hint,
                    )
                    hint_text = series_hint.text if series_hint is not None else ""
                    try:
                        en = await translator.translate(
                            sub_seg.jp, state.genre,
                            _context_for(sub_seg.jp, ctx.snapshot()),
                            # Kwarg only when a hint exists: the translator
                            # seam is duck-typed, and the hint-free call must
                            # stay byte-identical for translate()
                            # implementations that predate the
                            # series_context parameter.
                            **({"series_context": hint_text} if hint_text else {}),
                        )
                    except Exception as e:
                        # Keep the per-sentence skip (one bad sentence
                        # must not abandon the segment), but feed the
                        # supervisor's tracker: a bare warning would leave
                        # the 10-in-60s escalation chain (on_escalate ->
                        # stage_escalation_requested + done.set -> exit 45
                        # -> watchdog restart -> sanity check -> exit-43
                        # dialog) unreachable for persistent translator
                        # death, so subtitles would stop silently.
                        # The notify decision doubles as the traceback
                        # gate: one full stack per notify window keeps a
                        # glossary-OSError class triageable from the log
                        # without a 10-failure burst dumping 10 stacks.
                        notified = record_failure(
                            "translation", e,
                            notify=_notify_stage_error,
                            on_escalate=on_escalate,
                            tracker=_tracker,
                        )
                        log.warning("translation failed: %s", e,
                                    exc_info=notified)
                        # P1-8b: the tracker above can NEVER escalate for
                        # the hang class -- each timeout-family failure
                        # takes ~61 s end-to-end, so 10-in-60s is
                        # unreachable and a wedged llama-server (accepting
                        # connections, never completing; /v1/models still
                        # answering) drops one line per minute forever.
                        # Count CONSECUTIVE timeout-class failures and fire
                        # the SAME exit-45 escalation directly at the
                        # streak threshold. ONLY timeout-class failures
                        # extend the streak: any COMPLETED response -- a
                        # success (below), an error status, or a body that
                        # failed to parse -- proves a request
                        # round-tripped, which is exactly what "accepting
                        # but never completing" claims did not happen. A
                        # server alternating 500s and read timeouts under
                        # GPU contention is answering, so its scattered
                        # timeouts must reset rather than accumulate into a
                        # bogus wedge restart (which churns on into the
                        # anti-flap "keeps crashing" dialog). Those
                        # completed failures stay with the tracker above,
                        # whose 10-in-60s window they fail fast enough to
                        # reach.
                        if isinstance(e, httpx.TimeoutException):
                            consecutive_timeouts += 1
                            if consecutive_timeouts >= \
                                    _TRANSLATOR_TIMEOUT_ESCALATION_STREAK:
                                log.error(
                                    "translator wedged: %d consecutive "
                                    "timeout-class failures (llama-server "
                                    "accepting but never completing); "
                                    "escalating for a watchdog RESTART "
                                    "(exit %d)",
                                    consecutive_timeouts, EXIT_STAGE_ESCALATION,
                                )
                                consecutive_timeouts = 0
                                if on_escalate is not None:
                                    on_escalate("translation")
                        else:
                            consecutive_timeouts = 0
                        fates.terminal_once(
                            sentence_index, PostASRFate.TRANSLATION_ERROR,
                        )
                        continue
                    # Any completed round-trip -- even one whose EN is
                    # empty -- proves the translator is answering again:
                    # reset the wedge streak.
                    consecutive_timeouts = 0
                    if on_translated is not None:
                        # The translator answered: clears a translator
                        # problem (plan phase 4, Decision 19).
                        await on_translated()
                    if not en or not en.strip():
                        # Snippet gated on transcript consent -- same
                        # contract as the non-JA drop line above.
                        log.info(
                            "dropped empty EN for JP (len=%d)%s",
                            len(sub_seg.jp),
                            f": {sub_seg.jp[:60]!r}" if transcript_enabled else "",
                        )
                        fates.terminal_once(
                            sentence_index, PostASRFate.TRANSLATION_EMPTY,
                        )
                        continue
                    if post_asr_telemetry is not None and sentence_ref is not None:
                        post_asr_telemetry.decision(
                            sentence_ref,
                            PostASRStage.TRANSLATION,
                            PostASRDecision.TRANSLATION_COMPLETED,
                        )
                    phase = PostASRPhase.POST_TRANSLATION_SCAFFOLDING
                    sub_seg.t_translate_done = time.perf_counter()
                    sub_seg.en = en
                    ctx.push(en)
                    stats.record(sub_seg)
                    asr_ms = (
                        (sub_seg.t_asr_done - sub_seg.t_chunk_emit) * 1000
                        if sub_seg.t_asr_done else 0
                    )
                    tr_base = (
                        sub_seg.t_asr_done
                        if sub_seg.t_asr_done else sub_seg.t_chunk_emit
                    )
                    tr_ms = (sub_seg.t_translate_done - tr_base) * 1000
                    total_ms = (
                        sub_seg.t_translate_done - sub_seg.t_chunk_emit
                    ) * 1000
                    print(
                        f"[JA] {sub_seg.jp}\n[EN] {sub_seg.en}\n"
                        f"[ms] asr={asr_ms:.0f} tr={tr_ms:.0f} "
                        f"total={total_ms:.0f}\n",
                        flush=True,
                    )
                    phase = None
                    if manager is None:
                        fates.terminal_once(
                            sentence_index, PostASRFate.DISPLAY_NOT_CONFIGURED,
                        )
                    else:
                        try:
                            receipt = await manager.broadcast({
                                "type": "subtitle",
                                "jp": sub_seg.jp,
                                "jp_ruby": sub_seg.jp_ruby,
                                "en": sub_seg.en,
                                "t": time.time(),
                            })
                        except Exception as e:
                            # Preserve the existing ride-through and
                            # diagnostic; the new fate line itself receives
                            # no exception data.
                            log.warning("broadcast failed: %s", e)
                            fates.terminal_once(
                                sentence_index,
                                PostASRFate.DISPLAY_BROADCAST_ERROR,
                            )
                        else:
                            phase = PostASRPhase.DISPLAY
                            fate, counts = _display_fate(receipt)
                            fates.terminal_once(sentence_index, fate, **counts)
                    phase = PostASRPhase.POST_TRANSLATION_SCAFFOLDING
                    if latency_log is not None:
                        record = {
                            "ts": time.time(),
                            "asr_ms": asr_ms,
                            "translate_ms": tr_ms,
                            "total_ms": total_ms,
                            "jp_len": len(sub_seg.jp),
                            "en_len": len(sub_seg.en),
                        }
                        if transcript_enabled:
                            # Opt-in only; post-ASR telemetry never receives it.
                            record["jp"] = sub_seg.jp
                            record["en"] = sub_seg.en
                        latency_log.write(
                            json.dumps(record, ensure_ascii=False) + "\n",
                        )
                        latency_log.flush()
                    processed += 1
                    if max_segments > 0 and processed >= max_segments:
                        done.set()
                        return True
                except Exception:
                    if phase is not None:
                        fates.terminal_once(
                            sentence_index,
                            PostASRFate.PIPELINE_STAGE_ERROR,
                            phase,
                        )
                        fates.fail_remaining(sentence_index, phase)
                    raise
            return False

        should_exit = await supervise(
            "translation", _one_iter,
            notify=_notify_stage_error,
            tracker=_tracker,
            on_escalate=on_escalate,
        )
        # Check done.is_set() directly — don't rely solely on the return value.
        # If _one_iter raised after setting done, supervise returns None but the
        # exit signal must still be honoured.
        if should_exit or done.is_set():
            return


@dataclass
class HealthProbes:
    """Lazily-evaluated readiness checks for the pipeline's /healthz route.

    The browser_server /healthz handler reads each property on every
    request; keep the evaluation cheap. `asr_alive` and
    `broadcaster_accepting` are O(1) attribute reads on PipelineApp;
    `translator_reachable` is cached on PipelineApp with ~10s TTL by
    the `_refresh_translator_health` background task in run_pipeline
    (an active GET would block the request).
    """
    app: "PipelineApp"

    @property
    def asr_alive(self) -> bool:
        task = getattr(self.app, "_asr_task", None)
        if task is None:
            return False
        # A cancelled task was TOLD to stop -- it is not alive, and
        # it is not in an error state either. Check cancelled() BEFORE
        # exception(): per the asyncio contract, .exception() on a
        # cancelled task RAISES CancelledError (a BaseException that
        # Starlette's `except Exception` does not catch), which used to
        # blow up any /healthz poll landing in the shutdown drain window
        # between the stage-task cancellation and uvicorn's exit.
        if task.cancelled():
            return False
        # "done without exception" treated as nominal shutdown completion;
        # stage loops never return in steady-state operation.
        return (not task.done()) or (task.exception() is None)

    @property
    def translator_reachable(self) -> bool:
        # Default True so /healthz doesn't report "unhealthy" in the
        # 10s window before the first refresh completes. The refresher
        # overwrites this on every tick.
        return bool(getattr(self.app, "_translator_reachable_cached", True))

    @property
    def broadcaster_accepting(self) -> bool:
        mgr = getattr(self.app, "manager", None)
        return mgr is not None


@dataclass
class PipelineApp:
    """Wired-but-not-running pipeline. Composition Root output.

    Fields mirror what run_pipeline() needs. Unit tests build one via
    build_pipeline() and introspect it -- replaces the monkeypatch
    harness in tests/test_main_entry.py for wiring coverage.

    Hotkey setup lives in run_pipeline (not build_pipeline), next to
    the browser server that assigns app.manager. The hotkey adapters
    read app.manager at fire time, so a press before the server exists
    only skips its broadcast instead of holding a None manager for good.
    """
    args: Any                               # argparse.Namespace
    cfg: dict
    app_state: AppState
    audio_cfg: AudioConfig
    capture: LoopbackCapture
    chunker: Any                            # SileroChunker
    asr: Any
    translator: Translator
    ctx: ContextBuffer
    stats: LatencyStats
    latency_log: Any
    glossary_mgr: Optional[GlossaryManager]
    romanization_pipeline: Optional[RomanizationPipeline]
    cuda_pool: ThreadPoolExecutor
    max_segments: int
    audio_q: "asyncio.Queue"
    chunk_q: "asyncio.Queue"
    transcript_q: "asyncio.Queue"
    stage_tracker: ErrorTracker
    loop: "asyncio.AbstractEventLoop"
    done: "asyncio.Event"
    explicit_config_path: Path | None
    # Unfinished Japanese belongs to the current pause epoch. Shared with the
    # pause controller so an edge clears it even when ASR emits no transcript.
    remainder_state: PendingRemainderState = field(
        default_factory=PendingRemainderState,
    )
    # logging.transcript_log_enabled: when true, run_pipeline passes this to
    # translation_stage so latency.log records carry raw jp/en transcript text.
    transcript_enabled: bool = False
    # translation.series_context_hint: when true, run_pipeline passes this to
    # translation_stage, which composes the per-series "Series: ..." prompt
    # line from the cached AniList profile. Default OFF (A/B gate pending).
    series_hint_enabled: bool = False
    # Post-ASR lineage sink shared by the stages. run_pipeline sets it right
    # after creating it, so a pause edge can record the fate of the tail it
    # abandons. None during build and in unit-built apps.
    post_asr_telemetry: Optional[PostASRTelemetry] = None
    # Previous translator health-probe state. note_translator_health logs
    # on FLIPS only, so a down llama doesn't write one warning per 10 s
    # poll. Starts True to mirror HealthProbes' optimistic pre-first-probe
    # default (so the very first failing probe counts as a flip and warns).
    _translator_health_prev: bool = True
    # Populated by run_pipeline (None during build).
    hotkey_mgr: Optional[hk_mod.HotkeyManager] = None
    manager: Optional[ConnectionManager] = None
    uv_server: Optional[uvicorn.Server] = None
    # Set when the USER asks to quit (tray / overlay quit command). The
    # pipeline then exits STOP_ALL (43) so the watchdog stops instead of
    # treating the quit's teardown as a crash and restarting it (an
    # overlay quit whose teardown runs slow could otherwise loop into a
    # "keeps crashing" give-up). Distinct from `done` alone: the recoverable
    # failures that ALSO fire `done` (capture death, stage escalation) arm
    # their own restart flags below, so _exit_code_for can tell them apart.
    stop_all_requested: bool = False
    # Set when the capture stream dies (device unplug / suspend / active-but-
    # silent). amain then returns EXIT_STREAM_DEATH (44), which classify()
    # routes to RESTART -- the watchdog rebuilds the pipeline and re-resolves
    # the device. Capture death previously returned 0 (STOP_SELF) and the
    # watchdog stopped. If a user quit (stop_all) races it, stop_all wins.
    stream_death_requested: bool = False
    # Set when a pipeline stage crosses its supervisor escalation threshold
    # (repeated crashes, e.g. translation failing 10x in 60 s). amain then
    # returns EXIT_STAGE_ESCALATION (45), which classify() routes to RESTART.
    # Mirrors stream_death_requested: escalation previously set `done` alone,
    # so amain returned 0 (STOP_SELF) and the watchdog read a stage crash-loop
    # as a clean quit -- the app silently died with no dialog and no restart.
    # Precedence: stop_all(43) > startup_failure(46) > stream_death(44) >
    # stage_escalation(45) > overlay_giveup(47).
    stage_escalation_requested: bool = False
    # Set when the overlay HTTP/WebSocket server can't bind its port at
    # startup (_serve_overlay catches uvicorn's SystemExit/OSError). amain
    # then returns EXIT_STARTUP_FAILURE (46), which classify() routes to
    # STOP_ALL -- the watchdog gives up immediately (a bind conflict does
    # not self-heal in seconds) and shows a port-specific dialog. Bind
    # failure previously reached _exit_code_for with no armed flag => exit
    # 0 (STOP_SELF): the watchdog read the failed startup as a clean quit
    # and the whole app silently disappeared.
    startup_failure_requested: bool = False
    # Set when the overlay supervisor exhausts its respawn budget (the
    # overlay crash-looped past the anti-flap cap). run_pipeline arms this
    # from the supervisor's gave_up flag after its wait; amain then returns
    # EXIT_CHILD_GIVEUP (47), which classify() routes to RESTART -- a
    # pipeline restart grants a fresh overlay budget. Give-up previously
    # set the same overlay_stop event as a normal overlay exit => exit 0
    # (STOP_SELF): a flapped-out overlay was indistinguishable from a quit.
    overlay_giveup_requested: bool = False
    # True when Yamabiko really restarts the translator: a watchdog
    # supervises the bundled loopback llama-server (plan lead decision 4).
    # Picks the translator problem's copy. Set by build_pipeline.
    translator_restarts: bool = False
    # The first stage that crossed its escalation threshold ("asr",
    # "translation", "vad"); a "translation" escalation raises the
    # translator problem before the exit and restarts llama-server.
    escalated_stage: Optional[str] = None
    # time.perf_counter() of the last completed translation, and of the
    # previous health-refresher tick: a failing probe right after a
    # translation completed raises no translator problem (Decision 19).
    # perf_counter, not monotonic(): both are monotonic, but two events in
    # one 15 ms monotonic() tick must stay ordered.
    last_translation_ok: float = 0.0
    _translator_tick_prev: float = 0.0
    # What the last health probe saw of llama-server restarting on its own
    # (Decision 21): its connection failed (crashed; note_translator_health),
    # or /v1/models answered while /health was not ready (loading;
    # track_translator_problem). Either way no second restart is asked for.
    _translator_down: bool = False
    _translator_loading: bool = False
    # /v1/models answered at least once in this process (Decision 71): only
    # then does a failing probe raise the translator problem.
    _models_answered: bool = False
    # A restart exit is being announced (Decision 70): nothing is lifted
    # before the restart, the next pipeline reports it through the carry.
    _announcing_restart: bool = False
    # Shortcuts announced as taken, config key -> combo (Decision 72). A key
    # leaves when it binds or its combo changes. A restart exit carries them,
    # and run_pipeline seeds them from the carry, so the next pipeline does
    # not announce an unchanged one again.
    announced_hotkeys: dict = field(default_factory=dict)

    def request_stop_all(self) -> None:
        """User-initiated quit (tray / overlay). Mark the stop-all intent AND
        fire `done`. ``amain`` reads the flag and returns exit code 43
        (STOP_ALL), so the watchdog tears everything down rather than
        restarting the pipeline."""
        log.info("user quit requested (tray/overlay); stop_all armed -> exit 43")
        self.stop_all_requested = True
        self.done.set()

    def request_stream_death(self) -> None:
        """Capture stream died (device unplug / suspend / active-but-silent).
        Mark the restart intent AND fire `done`. amain reads the flag and
        returns EXIT_STREAM_DEATH, which classify() routes to RESTART so the
        watchdog rebuilds the pipeline and re-resolves the device. Mirrors
        request_stop_all; if both race, stop_all wins (a user quit must STOP).

        Called (marshalled onto the loop) from the capture monitor's foreign
        thread via the on_stream_death hook wired in run_pipeline.
        """
        log.info(
            "capture stream death signalled; stream_death armed -> exit %d (RESTART)",
            EXIT_STREAM_DEATH,
        )
        self.stream_death_requested = True
        self.done.set()

    def on_stage_escalate(self, stage: str) -> None:
        """Supervisor escalation callback: a stage has crossed its
        escalation threshold (repeated crashes) and should take the
        pipeline down with it. Arms stage_escalation_requested AND fires
        `done`, so amain returns EXIT_STAGE_ESCALATION (45) and the
        watchdog RESTARTS the pipeline (classify(45) -> RESTART). With
        `done` alone this exited 0 (STOP_SELF): the watchdog read the
        crash-loop as a clean quit and the app silently died. Mirrors
        request_stream_death; if a user quit races it, stop_all wins.

        Extracted from the _escalate_stage_died closure that used to
        live inside run_pipeline, so unit tests can invoke it
        directly instead of reaching into supervisor internals. Must
        NOT block (the supervisor calls this synchronously from the
        asyncio event loop).
        """
        log.error(
            "stage %s crashed repeatedly (threshold hit); "
            "stage_escalation armed -> exit %d (watchdog RESTART). "
            "See %%LOCALAPPDATA%%\\Yamabiko\\logs\\main-pipeline.log "
            "for details.",
            stage,
            EXIT_STAGE_ESCALATION,
        )
        if getattr(self, "escalated_stage", None) is None:
            self.escalated_stage = stage
        self.stage_escalation_requested = True
        self.done.set()

    async def set_paused(self, paused: bool) -> PauseState | None:
        """Apply one ordered pause edge and broadcast authoritative state.

        Same-state commands remain useful acknowledgements and therefore still
        broadcast, but only a real edge inserts a capture marker or advances
        unfinished-Japanese ownership.
        """

        edge = self.app_state.set_paused(paused)
        if edge is not None:
            self.capture.apply_pause_edge(edge)
            abandoned_ref = self.remainder_state.advance_epoch(edge.epoch)
            if (
                abandoned_ref is not None
                and self.post_asr_telemetry is not None
            ):
                self.post_asr_telemetry.terminal(
                    abandoned_ref,
                    PostASRFate.TAIL_DISCONTINUITY_CLEARED,
                )
        if self.manager is not None:
            await self.manager.broadcast(self.app_state.state_frame())
        return edge

    async def handle_pause_hotkey(self) -> None:
        """Hotkey Ctrl+Alt+P body: toggle pause state, then broadcast
        the new state to any connected overlay.

        Extracted from the _on_pause / _toggle_and_broadcast closure
        that used to live inside run_pipeline so unit tests can call
        this coroutine directly (so the hotkey-closure logic is unit-
        testable instead of locked inside a local closure).

        The `_on_pause_hotkey` adapter wraps this with dispatch_hotkey
        (`asyncio.run_coroutine_threadsafe`) to bridge the `keyboard`
        library's background thread into the asyncio loop. The adapter
        owns thread crossing; this method owns the state-change +
        broadcast logic.
        """
        await self.set_paused(not self.app_state.paused)

    async def handle_toggle_overlay_hotkey(self) -> None:
        """Hotkey Ctrl+Alt+O body: send a toggle-overlay command to
        any connected overlay client.

        Extracted from _on_toggle_overlay for the same reason as
        handle_pause_hotkey. If `manager` is None (no browser
        server running), logs debug and returns -- no broadcast
        because there's no one to broadcast to.
        """
        if self.manager is None:
            log.debug(
                "toggle_overlay hotkey pressed; no broadcast manager yet",
            )
            return
        await self.manager.broadcast(make_toggle_overlay_cmd())

    async def handle_show_series_hotkey(self) -> None:
        """Hotkey (Ctrl+Alt+S) body: ask any connected overlay to show the
        series-search palette. No-op if no broadcast manager yet — mirrors
        handle_toggle_overlay_hotkey."""
        if self.manager is None:
            log.debug("show_series hotkey pressed; no broadcast manager yet")
            return
        await self.manager.broadcast(make_show_series_cmd())

    def _on_pause_hotkey(self) -> None:
        self.dispatch_hotkey(self.handle_pause_hotkey())

    def _on_toggle_overlay_hotkey(self) -> None:
        self.dispatch_hotkey(self.handle_toggle_overlay_hotkey())

    def _on_show_series_hotkey(self) -> None:
        self.dispatch_hotkey(self.handle_show_series_hotkey())

    def register_hotkeys(self, announce: frozenset[str] | None = None) -> None:
        """Bind the global shortcuts in ``cfg["hotkeys"]`` through
        ``hotkey_mgr``. The callbacks are thin thread-bridge adapters: the
        `keyboard` library fires them on its own thread and dispatch_hotkey
        posts the async body onto the pipeline loop. The series-palette
        shortcut belongs to the opt-in series feature: with the feature off
        its combo is passed empty, which HotkeyManager.register skips.

        ``announce`` picks which conflicts reach the user. None (startup)
        announces every conflict except one on a key already announced with
        the same combo (``announced_hotkeys``, carried over a restart:
        Decision 72). Otherwise only a conflict on one of the listed config
        keys ("pause_resume", "toggle_overlay", "show_series") is announced;
        any other conflict is only logged, so a shortcut the user did not
        touch does not say "Shortcut unavailable" again."""
        if self.hotkey_mgr is None:
            return
        hk_cfg = self.cfg.get("hotkeys", {}) or {}
        combos = {
            "pause_resume": hk_cfg.get("pause_resume", "Ctrl+Alt+P"),
            "toggle_overlay": hk_cfg.get("toggle_overlay", "Ctrl+Alt+O"),
            "show_series": (hk_cfg.get("show_series", "Ctrl+Alt+S")
                            if feature_enabled(self.cfg) else ""),
        }
        if announce is None:
            announce = frozenset(key for key, combo in combos.items()
                                 if self.announced_hotkeys.get(key) != combo)
        failed: set[str] = set()

        def _on_conflict(combo: str, exc: Exception) -> None:
            # HotkeyManager passes the combo exactly as given above.
            keys = {key for key, value in combos.items() if value == combo}
            failed.update(keys)
            if keys & announce:
                self.announced_hotkeys.update(dict.fromkeys(keys, combo))
                self.notify_hotkey_conflict(combo, exc)
            else:
                log.info("shortcut %s still unavailable (not announced): %s", combo, exc)

        self.hotkey_mgr.register(
            pause=combos["pause_resume"],
            toggle_overlay=combos["toggle_overlay"],
            show_series=combos["show_series"],
            on_pause=self._on_pause_hotkey,
            on_toggle_overlay=self._on_toggle_overlay_hotkey,
            on_show_series=self._on_show_series_hotkey,
            on_conflict=_on_conflict,
        )
        # A key that bound, or whose combo changed, is no longer announced.
        for key, combo in combos.items():
            if key not in failed or self.announced_hotkeys.get(key) != combo:
                self.announced_hotkeys.pop(key, None)

    def apply_hotkeys(self, hotkeys: dict, replay: bool = False) -> None:
        """Settings changed shortcuts (the set_hotkeys command, after it
        persisted them): adopt them and re-bind at once. No-op binding while
        hotkeys are disabled or the manager failed to start.

        A conflict is announced only for the shortcuts this payload names,
        so re-binding after one shortcut changed does not re-announce the
        others. ``replay`` marks the tray's reconnect replay of shortcuts
        saved while offline: then only a shortcut whose combo changed is
        announced."""
        old = dict(self.cfg.get("hotkeys") or {})
        changed = {key for key, value in hotkeys.items()
                   if hk_mod.canonical_combo(value) != hk_mod.canonical_combo(old.get(key, ""))}
        announce = changed if replay else set(hotkeys)
        merged = dict(old)
        merged.update(hotkeys)
        self.cfg["hotkeys"] = merged
        if self.hotkey_mgr is None:
            return
        self.hotkey_mgr.unregister()
        self.register_hotkeys(announce=frozenset(announce))

    def notify_hotkey_conflict(self, combo: str, exc: Exception) -> None:
        """HotkeyManager on_conflict body: tell the user a shortcut couldn't
        be bound, and where to fix it.

        Broadcasts a notice frame (the tray shows it as a toast) when the Qt
        overlay is connected. Otherwise, which is the normal case because
        hotkeys register before the WS server starts, the notice waits on
        AppState and the /ws handler delivers it to the first native client.
        A browser/OBS page can't show notices, so it doesn't count.
        """
        log.warning("hotkey %s unavailable: %s", combo, exc)
        shown = hk_mod.canonical_combo(combo) or combo
        if isinstance(exc, hk_mod.HotkeyTaken):
            body = f"{shown} is used by another app. Pick a new one in Settings."
        else:
            body = f"Yamabiko couldn't set up {shown}. Pick a new one in Settings."
        notice = make_notice_frame("Shortcut unavailable", body, settings_page="shortcuts")
        if self.manager is not None and self.manager.native_client_count() > 0:
            self.dispatch_hotkey(self.manager.broadcast(notice))
        else:
            self.app_state.queue_notice(notice)

    def note_translator_health(
        self, reachable: bool, detail: object = None,
    ) -> None:
        """Record one translator health-probe result for /healthz and log
        state FLIPS only: one WARNING on healthy->unhealthy, one INFO on
        recovery -- never one line per poll (a down llama would otherwise
        write ~6 identical warnings a minute for its whole outage).

        Extracted from the _refresh_translator_health closure in
        run_pipeline so the transition logic is directly unit-testable
        (testing charter §3.9: invoke the body, not just the wiring).
        ``detail`` is the failure evidence (exception or "HTTP <code>");
        only rendered on the warning line.
        """
        prev = self._translator_health_prev
        self._translator_reachable_cached = reachable
        self._translator_health_prev = reachable
        # Nothing accepted the probe's connection: llama-server is down and
        # its own supervisor restarts it (Decision 21). A refused loopback
        # connect takes about 2 s on Windows, so it can arrive as a connect
        # timeout; a ReadTimeout (connected, no answer) is a hung server.
        self._translator_down = not reachable and isinstance(
            detail, (httpx.ConnectError, httpx.ConnectTimeout),
        )
        if not reachable and prev:
            log.warning(
                "translator health probe failing%s; /healthz reports "
                "unhealthy until it recovers",
                f" ({detail})" if detail is not None else "",
            )
        elif reachable and not prev:
            log.info("translator health probe recovered")

    async def raise_problem(self, problem: dict) -> bool:
        """Make ``problem`` active and tell every client (plan lead decision
        1). A repeat of the active entry sends nothing. True when the set
        changed."""
        if not self.app_state.set_problem(problem):
            return False
        log.warning("problem raised: %s", problem["kind"])
        await self._broadcast_problems()
        return True

    async def lift_problem(self, kind: str) -> None:
        """Clear ``kind`` and tell every client; the clients show its
        recovery line. Nothing is lifted once a restart exit is being
        announced (Decision 70): neither a health tick nor a completed
        translation during the hold shows the recovery before the restart."""
        if self._announcing_restart:
            return
        if not self.app_state.clear_problem(kind):
            return
        log.info("problem cleared: %s", kind)
        await self._broadcast_problems()

    async def _broadcast_problems(self) -> None:
        if self.manager is not None:
            await self.manager.broadcast(self.app_state.problems_frame())

    async def track_translator_problem(self, models_ok: bool, client) -> None:
        """One health-refresher tick (plan lead decision 2, Decisions 19 and
        71): a failing /v1/models probe raises the translator problem once
        /v1/models has answered in this process (an endpoint that never
        serves the route, a remote server without it, is left to the
        translation escalations), unless a translation completed since the
        previous tick (the translator answers; only the probe failed, e.g. it
        timed out). When /v1/models answered, /health tells whether
        llama-server serves or still loads its model (it answers /v1/models
        during the load): ready clears an active problem, not ready is
        remembered for Decision 21."""
        translated = self.last_translation_ok > self._translator_tick_prev
        self._translator_tick_prev = time.perf_counter()
        if not models_ok:
            self._translator_loading = False
            if (
                self._models_answered
                and not translated
                and not self.app_state.has_problem(problems.TRANSLATOR)
            ):
                # A restart is promised only when one really happens (lead
                # decision 4): the server could not be reached, so its
                # supervisor respawns it. A hung one is restarted only by a
                # translation escalation, which upgrades this entry.
                await self.raise_problem(
                    problems.translator_problem(
                        self.translator_restarts and self._translator_down,
                    ),
                )
            return
        self._models_answered = True
        ready = await health_says_ready(client, self.translator.readiness_url)
        self._translator_loading = not ready
        if ready and self.app_state.has_problem(problems.TRANSLATOR):
            await self.lift_problem(problems.TRANSLATOR)

    async def note_translation_ok(self) -> None:
        """translation_stage's on_translated: a completed translation proves
        the translator answers again (Decision 19)."""
        self.last_translation_ok = time.perf_counter()
        if self.app_state.has_problem(problems.TRANSLATOR):
            await self.lift_problem(problems.TRANSLATOR)

    async def announce_restart_problems(self) -> None:
        """Before an exit the watchdog answers with a restart (44, 45, 47):
        broadcast the problems behind it, leave the carry for the watchdog and
        the next pipeline (plan Decision 18; it also holds the shortcuts
        announced as taken, Decision 72), and, when this raised a problem
        while a client is connected, hold the exit _ANNOUNCE_HOLD_S so the
        notification and the overlay line are on screen before teardown
        stops the overlay (R1). Once a restart exit is announced nothing is
        lifted (Decision 70). A capture death whose endpoint is still an
        active render endpoint (the default output switched, a glitch), or
        whose name or the endpoint list is unknown, restarts without a
        notice, as before."""
        if _exit_code_for(self) not in _RESTART_EXIT_CODES:
            return
        self._announcing_restart = True
        raised = False
        if self.stream_death_requested:
            name = self.capture.endpoint_name()
            present = (
                await asyncio.to_thread(active_render_endpoint_names)
                if name else None
            )
            if not name:
                log.info(
                    "capture stream died and its endpoint name is unknown; "
                    "restarting without a notice",
                )
            elif present is None:
                log.info(
                    "capture stream died and the active render endpoints "
                    "could not be listed; restarting without a notice",
                )
            elif name not in present:
                raised = await self.raise_problem(problems.audio_problem(name)) or raised
            else:
                log.info(
                    "capture stream died with its endpoint still listed; "
                    "restarting without a notice",
                )
        if self.escalated_stage == "translation":
            raised = await self.raise_problem(
                problems.translator_problem(self.translator_restarts),
            ) or raised
        kinds = self.app_state.problem_kinds()
        # Restart llama-server only when it is not already restarting on its
        # own (Decision 21): the last probe could not connect (it crashed;
        # its supervisor respawns it) or found it loading. A hung server
        # (connected, no answer) or one that fails translations is restarted.
        restart_translator = (
            self.escalated_stage == "translation"
            and self.translator_restarts
            and not self._translator_down
            and not self._translator_loading
        )
        hotkeys = dict(self.announced_hotkeys)
        if kinds or restart_translator or hotkeys:
            await asyncio.to_thread(
                problems.write_carry, kinds, restart_translator, hotkeys,
            )
        if raised and self.manager is not None and self.manager.client_count() > 0:
            await asyncio.sleep(_ANNOUNCE_HOLD_S)

    def dispatch_hotkey(self, coro) -> None:
        """Bridge a `keyboard`-thread callback onto the pipeline loop, guarded.

        A keystroke arriving after the loop has closed (teardown race:
        unregister() may have thrown so the kernel hook is still live)
        would make run_coroutine_threadsafe raise RuntimeError on the keyboard
        daemon thread and print a stderr traceback. Swallow it and close the
        un-scheduled coroutine (no 'coroutine was never awaited' warning)."""
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        except RuntimeError as exc:
            log.debug("hotkey dispatch skipped (loop closed): %s", exc)
            coro.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the pipeline CLI flags.

    Extracted from main() so tests can construct args without driving
    sys.argv / SystemExit. The --run-main flag is accepted and ignored
    (belt-and-suspenders; the watchdog strips it before calling us).
    """
    parser = argparse.ArgumentParser(prog="yamabiko")
    parser.add_argument(
        "--max-segments", type=int, default=0,
        help="Exit after N translated segments (0 = run forever)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Dev/repro escape hatch: layer this YAML file over the shipped "
            "defaults, SKIPPING user_overrides.yaml so the run is isolated "
            "from ambient user state. Omit for the normal layered config "
            "(defaults + user_overrides.yaml)."
        ),
    )
    parser.add_argument(
        "--run-main", action="store_true", help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _seed_user_glossaries(dest: Path) -> None:
    """Seed the user-writable glossaries dir from the bundle, copying each
    bundled file ONLY if it isn't already present in ``dest``.

    The bundle glossaries dir (``paths.bundle_data_dir()/glossaries``) is
    read-only and re-extracted on every install, so the runtime must read
    AND write the genre JSONs from a writable location instead. Per-file
    copy-if-absent means: first run gets the shipped
    glossaries; a file already in ``dest`` (with accumulated learned
    entries) is NEVER overwritten; and a newly-NAMED shipped file reaches
    existing installs on the next launch. ``dest`` is always created, so a
    missing bundle source or a copy error can't leave the dir absent and
    FileNotFound the first learned-entry write.

    Underscore-prefixed files (_seed.yaml, _denylist.txt) are NOT copied:
    they are read-only data the runtime reads directly from the bundle, and
    a copy-if-absent copy in app-data would be frozen at first-install
    version, hiding every future shipped revision (the same staleness
    class that affects seed entries). Only genre JSONs and user.yaml
    belong in the writable dir.
    """
    import shutil
    try:
        dest.mkdir(parents=True, exist_ok=True)
        src = paths.bundle_data_dir() / "glossaries"
        if not src.is_dir():
            return
        for f in src.iterdir():
            if f.name.startswith("_"):
                continue
            target = dest / f.name
            if f.is_file() and not target.exists():
                shutil.copy2(f, target)
    except OSError as exc:
        log.warning("could not seed user glossaries from bundle: %s", exc)


def _flatten_config_keys(block: dict, prefix: str = "") -> list[str]:
    """Dotted leaf-key paths of a nested config dict -- keys ONLY, values
    never leave this function (banner privacy contract)."""
    keys: list[str] = []
    for k, v in block.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            keys.extend(_flatten_config_keys(v, dotted + "."))
        else:
            keys.append(dotted)
    return keys


def _applied_override_keys() -> list[str]:
    """Dotted keys from user_overrides.yaml that load_full_config
    ACTUALLY applied to the merged config.

    Replays the fallback validation load_full_config applies to unknown,
    wrong-typed, and invalid-null values. Numeric semantic violations are
    fatal during config loading, so they can never reach the fingerprint.
    """
    from .config import (
        _read_yaml, _strip_keys, _walk_null_violations,
        _walk_type_mismatches, _walk_unknown_keys,
    )

    defaults = _read_yaml(paths.shipped_defaults_path())
    overrides = _read_yaml(paths.user_overrides_path())
    unknown = _walk_unknown_keys(overrides, defaults)
    if unknown:
        overrides = _strip_keys(overrides, unknown)
    bad_types = _walk_type_mismatches(overrides, defaults)
    if bad_types:
        overrides = _strip_keys(overrides, [p for p, _, _ in bad_types])
    null_bad = _walk_null_violations(overrides, defaults)
    if null_bad:
        overrides = _strip_keys(overrides, null_bad)
    return sorted(_flatten_config_keys(overrides))


def _config_digest_for_banner(
    cfg: dict, explicit: Path | None = None,
) -> str | None:
    """Config fingerprint for the ``config:`` INFO line:
    ``digest=<sha256(canonical-JSON of the merged cfg)[:12]>`` plus
    either the dotted override KEYS the user applied (layered mode) or
    ``source=--config:<basename>`` (explicit-file mode).

    KEYS ONLY, never values: override values carry endpoints, machine
    paths, and device names -- PII once the log is attached to a public
    bug report. The digest still lets two reports be compared for "same
    effective config?" without reading either.

    ``explicit`` is args.config: under --config the file is layered over
    the shipped defaults with the user_overrides.yaml layer SKIPPED, so
    there are no user overrides to list -- naming keys from
    user_overrides.yaml there would fingerprint a file the run never
    applied. Never raises: the fingerprint must not be able to break
    startup.
    """
    try:
        import hashlib
        digest = hashlib.sha256(
            json.dumps(
                cfg, sort_keys=True, ensure_ascii=False, default=repr,
            ).encode("utf-8")
        ).hexdigest()[:12]
        if explicit is not None:
            return f"digest={digest} source=--config:{Path(explicit).name}"
        try:
            keys = _applied_override_keys()
        except Exception:
            keys = ["<overrides unreadable>"]
        return f"digest={digest} overrides=[{','.join(keys)}]"
    except Exception:
        return None


async def build_pipeline(args: argparse.Namespace) -> PipelineApp:
    """Load config + construct every pipeline collaborator.

    Async because ASR warmup and translator sanity_check both require
    the event loop. Does NOT: start audio capture,
    build the browser server, register hotkeys, or create any stage
    task -- those are run_pipeline()'s job.

    On startup failure (VAD init / ASR init / translator sanity check),
    shows a tailored dialog via show_startup_error and exits with code 43
    -- same behaviour as pre-refactor amain(). The watchdog's exit-43
    sentinel path is preserved.
    """
    from .config import ConfigValidationError, load_overlay_config

    loop = asyncio.get_running_loop()
    # Fresh tracker per invocation so repeated build_pipeline() calls
    # (e.g. tests) don't accumulate error counts from previous runs.
    stage_tracker = _reset_stage_tracker()
    try:
        cfg = _load_config(args.config)
    except (RuntimeError, ConfigValidationError) as exc:
        # Missing/empty bundled default_config.yaml: a broken
        # install (spec datas regression). Same tailored-dialog +
        # exit-43 treatment as ASR/translator init failures, so the
        # watchdog shows one clear dialog instead of restarting a
        # permanently broken pipeline 3x and falling back to the
        # generic min-lifetime message.
        from yamabiko.ui.error_dialog import (
            classify_startup_exception, show_startup_error,
        )
        log.exception("config load failed")
        show_startup_error(classify_startup_exception("config load", exc))
        sys.exit(43)
    # Config fingerprint line. The process identity banner is emitted
    # in main() right after setup_main_log (so wizard-phase logs carry a
    # version line); this SECOND line is emitted here because the digest
    # needs the MERGED config, which only exists post-load (and
    # post-wizard on a first run). Override KEYS only, never values --
    # see _config_digest_for_banner.
    log.info(
        "config: %s",
        _config_digest_for_banner(cfg, args.config) or "digest unavailable",
    )
    # Junk logging.level warning: load_full_config normalizes the merged
    # value and STASHES the warning text instead of emitting it (in the
    # WATCHDOG process the load runs before any handler is attached, so
    # an in-load emission dies on the frozen bundle's discarded stderr).
    # Each role re-emits after its handler attach; for this pipeline
    # child, main() wired main-pipeline.log onto the root logger before
    # build_pipeline ran, so the warning lands in the role file here.
    from .config import consume_level_warning
    _level_warn = consume_level_warning()
    if _level_warn:
        log.warning("%s", _level_warn)
    # config_path selects which file load_overlay_config reads the
    # initial overlay style from -- the user_overrides file under the
    # layered model, or the explicit file when running with --config.
    # (The browser server's set_style writer targets user_overrides
    # directly via the sparse config writer; it no longer takes a path.)
    config_path = args.config if args.config is not None else paths.user_overrides_path()
    app_state = AppState(genre=cfg.get("genre", "variety"))
    overlay_cfg = load_overlay_config(
        None if args.config is None else config_path,
    )
    app_state.set_style(overlay_cfg.style)
    # overlay.reduce_motion for the browser page: the style frame carries it.
    app_state.set_reduce_motion(overlay_cfg.reduce_motion)
    # asr.incremental.enabled ("Faster subtitles"): the ASR stage reads it at
    # each new window; Settings changes it live through set_incremental.
    app_state.set_incremental(
        bool(cfg["asr"].get("incremental", {}).get("enabled", True))
    )
    log.info(
        "faster subtitles: enabled=%s",
        str(app_state.incremental_enabled()).lower(),
    )

    audio_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    transcript_q: asyncio.Queue = asyncio.Queue(maxsize=16)

    # Yamabiko's Silero wrapper contract is fixed at 16 kHz with
    # 512-sample external frames. The wrapper prepends 64 samples of
    # rolling context, so the raw ONNX graph receives 576 samples. An
    # audio.sample_rate override is a valid int so the config type
    # validator passes it through -- coerce it here. Safe because the
    # capture layer resamples device audio to the configured rate.
    _audio_block = cfg.get("audio")
    if (
        isinstance(_audio_block, dict)
        and _audio_block.get("sample_rate") != 16000
    ):
        log.warning(
            "audio.sample_rate=%r is unsupported: Yamabiko's Silero "
            "wrapper requires 16000 (512-sample external frames; "
            "576-sample raw graph input); forcing 16000",
            _audio_block.get("sample_rate"),
        )
        _audio_block["sample_rate"] = 16000

    audio_cfg = AudioConfig(
        device=cfg["audio"]["device"],
        sample_rate=cfg["audio"]["sample_rate"],
        block_ms=cfg["audio"]["block_ms"],
        # .get(): belt-and-suspenders from the era when an explicit
        # --config file was read verbatim and direct indexing raised an
        # uncaught KeyError -> watchdog fast-crash give-up. The
        # defaults<-explicit merge now restores the key at load time;
        # the fallback stays as AudioConfig's own field default
        # (DEFAULT_IDLE_DEATH_S) so the two can't drift.
        idle_death_s=cfg["audio"].get("idle_death_s", AudioConfig.idle_death_s),
    )
    capture = LoopbackCapture(
        audio_cfg,
        loop,
        audio_q,
        pause_state_probe=app_state.pause_snapshot,
    )

    chunk_q: asyncio.Queue = asyncio.Queue(maxsize=8)
    chunker_cfg, vad_threshold = chunker_config_from_config(cfg)
    from yamabiko.vad.silero_chunker import SileroChunker
    try:
        chunker = SileroChunker(
            chunker_cfg,
            threshold=vad_threshold,
            state=app_state,
        )
    except Exception as exc:
        # A stale or corrupt cached Silero ONNX fails its digest check
        # here, and a re-run cannot heal it. Same tailored dialog + exit 43
        # as ASR init: any other exit makes the watchdog restart into the
        # same failure until it gives up with a generic dialog.
        from yamabiko.ui.error_dialog import (
            classify_startup_exception, show_startup_error,
        )
        log.exception("VAD startup failed")
        err = classify_startup_exception("VAD init", exc)
        show_startup_error(err)
        sys.exit(43)
    log.info(
        "chunker: SileroChunker (threshold=%s, min_silence_ms=%s)",
        vad_threshold,
        cfg["vad"]["min_silence_ms"],
    )

    cuda_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda")
    log.info("loading ASR backend=faster_whisper ...")
    try:
        asr = await loop.run_in_executor(cuda_pool, build_asr, cfg["asr"])
        log.info("ASR loaded; warming up ...")
        await loop.run_in_executor(cuda_pool, asr.warmup)
    except Exception as exc:
        # Classify the failure and show a tailored dialog.
        # ASR init covers multiple failure classes: missing CUDA/GPU
        # (cublas/cudnn errors), missing model files (FileNotFoundError),
        # and disk-full (errno 28). classify_startup_exception pattern-
        # matches on exception message to pick the right StartupError
        # factory.
        from yamabiko.ui.error_dialog import (
            classify_startup_exception, show_startup_error,
        )
        log.exception("ASR startup failed")
        err = classify_startup_exception("ASR init", exc)
        show_startup_error(err)
        # Exit code 43 tells the watchdog "user has been shown a
        # tailored startup-error dialog; don't show the generic
        # crash-loop give-up on top of it."
        sys.exit(43)

    romanization_pipeline = None
    glossary_mgr = None
    # Fallback True matches default_config.yaml (romanization.enabled:
    # true): a dropped key must not silently turn the shipped-ON feature
    # OFF (no furigana, no glossary learning, gazetteer dead).
    #
    # Keep the section literal adjacent to the leaf it reads: the preflight
    # consumer gate anchors a collision-prone leaf ("enabled" names two
    # different keys) to its parent within a short window, so prose wedged
    # between the two reads it as an orphan.
    rom_cfg = cfg.get("romanization") or {}
    if rom_cfg.get("enabled", True):
        # Resolve glossaries_dir:
        #   1. If config explicitly overrides, use as-is (user knows best).
        #   2. Otherwise, default to the bundled glossaries path computed
        #      via paths.bundle_data_dir() -- frozen-aware, works in both
        #      dev mode and the installed bundle.
        # Learned proper-nouns are written back to the genre JSONs, so the
        # glossaries dir must be USER-WRITABLE and survive upgrades. The
        # bundle dir (bundle_data_dir) is read-only + re-extracted on every
        # install, so writing there loses all learned entries. Default to
        # app_data and seed it from the bundle on first
        # run; an explicit override is still honoured.
        default_gdir = paths.app_data_dir() / "glossaries"
        override = rom_cfg.get("glossaries_dir")
        if override:
            gdir = Path(override)
            if not gdir.is_absolute():
                # A relative override is NEVER honored: cwd differs between
                # dev (repo root) and the frozen exe ($INSTDIR), so the same
                # relative string resolves to two different directories.
                # Only an absolute path is unambiguous. Fall back to the
                # user-writable bundled default (matches the glossaries_dir
                # comment in default_config.yaml).
                log.warning(
                    "glossaries_dir override %r is not absolute; a relative "
                    "path is ambiguous (cwd differs between dev and the "
                    "installed app). Falling back to %s. Set an absolute "
                    "path to override.", override, default_gdir,
                )
                gdir = default_gdir
        else:
            gdir = default_gdir
        if gdir == default_gdir:
            _seed_user_glossaries(default_gdir)
            # Read the seed layer directly from the read-only bundle.
            # A copy in app-data would be frozen at first-install version
            # (copy-if-absent never refreshes it), so a revised shipped
            # _seed.yaml would never reach existing installs.
            seed_path = paths.bundle_data_dir() / "glossaries" / "_seed.yaml"
        else:
            # Explicit override dir is fully user-managed, seed included.
            seed_path = gdir / "_seed.yaml"
        # Degrade to romanization-disabled (rather than crash the whole app)
        # if the glossary store can't be built -- e.g. an unwritable dir or a
        # glossary file the loaders can't recover. Corrupt genre JSON is
        # already handled gracefully inside GlossaryManager; this
        # is the belt-and-suspenders for anything else.
        try:
            glossary_mgr = GlossaryManager(
                gdir, genre=app_state.genre, seed_path=seed_path,
            )
            # Restore the active series' gazetteer + seed runtime state from the
            # persisted pointer, so a freshly-connected overlay shows the series
            # and the hot loop doesn't redundantly reload on the first sentence.
            _ptr = load_active_series() if feature_enabled(cfg) else None
            # Fuzzy katakana recovery config (gazetteer v1.5). Read here so
            # the pipeline stays config-free; fallbacks match
            # default_config.yaml (fuzzy_enabled: true) so a dropped key
            # can't silently turn the shipped-ON recovery off.
            _gaz_cfg = cfg.get("gazetteer") or {}
            romanization_pipeline = RomanizationPipeline(
                glossary_mgr,
                Romanizer(),
                fuzzy_enabled=bool(_gaz_cfg.get("fuzzy_enabled", True)),
                fuzzy_max_distance=float(_gaz_cfg.get("fuzzy_max_distance", 0.25)),
            )
            if _ptr is not None:
                _sid, _slabel, _snative = _ptr
                # SeriesGazetteer.load already returns None for a missing file,
                # so the is_file() pre-check was redundant.
                _restored_gaz = SeriesGazetteer.load(gazetteer_path(_sid))
                # set_series bumps the epoch; recording that same epoch on the
                # pipeline stops the hot loop from redundantly reloading on the
                # first sentence. set_gazetteer is the single write path.
                app_state.set_series(_sid, _slabel, _snative)
                romanization_pipeline.set_gazetteer(
                    _restored_gaz, _sid, epoch=app_state.series_epoch,
                )
            log.info(
                "romanization pipeline enabled: glossaries=%s, genre=%s",
                gdir, app_state.genre,
            )
        except Exception:
            log.exception(
                "glossary/romanization init failed; disabling romanization",
            )
            glossary_mgr = None
            romanization_pipeline = None

    t_cfg = TranslationConfig(
        endpoint=cfg["translation"]["endpoint"],
        model=cfg["translation"]["model"],
        max_tokens=cfg["translation"]["max_tokens"],
        temperature=cfg["translation"]["temperature"],
        startup_readiness_timeout_s=cfg["translation"].get("startup_readiness_timeout_s", 90.0),
        startup_readiness_poll_s=cfg["translation"].get("startup_readiness_poll_s", 2.0),
        # Fallback True matches default_config.yaml (polarity_guard: true).
        # TranslationConfig's own dataclass default stays False (the class
        # is opt-in); THIS call site mirrors the shipped yaml so a dropped
        # key can't silently disable the shipped-ON guard.
        polarity_guard=cfg["translation"].get("polarity_guard", True),
    )
    # The Translator reuses the romanization pipeline's fugashi/UniDic tagger for
    # the polarity guard (no second dictionary load); when romanization is off it
    # lazily builds its own only if the guard actually fires.
    translator = Translator(t_cfg, pipeline=romanization_pipeline)
    try:
        # Wait for llama-server to finish loading the GGUF (it returns 503
        # "loading model" while loading) BEFORE the sanity translate, so a cold
        # load is no longer a spurious exit-43 STOP_ALL. A genuine never-ready
        # llama raises LlamaNotReadyError -> the same tailored-error path below.
        log.info("waiting for llama-server to finish loading the model ...")
        await translator.wait_until_ready(
            timeout_s=t_cfg.startup_readiness_timeout_s,
            poll_s=t_cfg.startup_readiness_poll_s,
        )
        log.info("translator sanity check ...")
        await translator.sanity_check(genre=cfg.get("genre", "variety"))
    except Exception as exc:
        # Show the tailored "The translator didn't start" dialog
        # rather than exit 1 into a generic watchdog
        # give-up message. classify_startup_exception matches on
        # ConnectionError / "localhost:8090" / "llama-server" keywords.
        from yamabiko.ui.error_dialog import (
            classify_startup_exception, show_startup_error,
        )
        log.error(
            "translator sanity check failed (is llama-server running?): %s",
            exc,
        )
        err = classify_startup_exception("translator sanity check", exc)
        show_startup_error(err)
        sys.exit(43)

    ctx = ContextBuffer(max_lines=cfg["translation"]["context_lines"])
    remainder_state = PendingRemainderState(
        active_epoch=app_state.pause_snapshot().epoch,
    )
    stats = LatencyStats()
    # Same process as main()'s root-logger call, so the same
    # main-pipeline.log filename (role convention). Normally the
    # ancestor dedup skips the attach; the filename matters when
    # build_pipeline runs standalone (tests, embedding).
    log_config.setup_main_log(
        log, level=cfg.get("logging", {}).get("level", "INFO"),
        filename="main-pipeline.log",
    )
    latency_enabled = cfg.get("logging", {}).get("latency_log_enabled", True)
    latency_log = log_config.setup_latency_log(enabled=latency_enabled)
    if latency_log is None:
        log.info("latency log: disabled via config")
    else:
        log.info("latency log: %s", paths.logs_dir() / "latency.log")
    transcript_enabled = cfg.get("logging", {}).get("transcript_log_enabled", False)
    if transcript_enabled:
        log.info("transcript logging ON: latency.log carries raw jp/en text")
    series_hint_enabled = bool(
        cfg.get("translation", {}).get("series_context_hint", False)
    )
    if series_hint_enabled:
        log.info(
            "series-context prompt hint ON (translation.series_context_hint)"
        )

    # "Yamabiko is restarting it" only when the watchdog supervises the
    # bundled loopback llama-server (plan lead decision 4): the watchdog
    # hands the pipeline its shutdown event, and _llama_endpoint_plan is the
    # test build_watchdog uses to decide whether to spawn llama-server.
    from yamabiko import watchdog as _wd
    from yamabiko.lifecycle.shutdown_event import ENV_VAR_BY_ROLE
    translator_restarts = bool(
        os.environ.get(ENV_VAR_BY_ROLE["pipeline"])
    ) and _wd._llama_endpoint_plan(cfg)[0]

    return PipelineApp(
        args=args,
        cfg=cfg,
        app_state=app_state,
        audio_cfg=audio_cfg,
        capture=capture,
        chunker=chunker,
        asr=asr,
        translator=translator,
        ctx=ctx,
        stats=stats,
        latency_log=latency_log,
        glossary_mgr=glossary_mgr,
        romanization_pipeline=romanization_pipeline,
        cuda_pool=cuda_pool,
        max_segments=args.max_segments,
        audio_q=audio_q,
        chunk_q=chunk_q,
        transcript_q=transcript_q,
        stage_tracker=stage_tracker,
        loop=loop,
        done=asyncio.Event(),
        explicit_config_path=(
            args.config.resolve(strict=False) if args.config is not None else None
        ),
        remainder_state=remainder_state,
        transcript_enabled=transcript_enabled,
        series_hint_enabled=series_hint_enabled,
        translator_restarts=translator_restarts,
    )


def _overlay_ws_url(cfg: dict) -> str:
    """WS endpoint for the spawned Qt overlay.

    overlay.websocket_port is honored by the pipeline's uvicorn and the
    watchdog health probe; the overlay child must receive the same port
    or it silently reconnect-loops against the 8765 argparse default —
    no subtitles, tray Pause/Genre/Quit dropped. Same int() coercion
    and 8765 fallback as the uvicorn bind in run_pipeline.
    """
    port = int((cfg.get("overlay") or {}).get("websocket_port", 8765))
    return f"ws://127.0.0.1:{port}/ws"


def _start_capture_or_exit(app: PipelineApp) -> None:
    """Start audio capture, converting an unrecoverable failure into a
    tailored exit-43 startup dialog.

    LoopbackCapture.start() recovers a stale saved endpoint by following the
    Windows default for that session. A failure reaching this boundary means
    capture still has no usable source (or failed for another reason).
    capture.start() runs before run_pipeline's try/finally, so letting that
    exception propagate would make the watchdog reload Whisper and retry 3x
    before a generic give-up dialog. Mirror the ASR-init pattern: show one
    tailored dialog and exit 43 (STOP_ALL) on the first attempt.
    """
    try:
        app.capture.start()
    except Exception as exc:
        from yamabiko.ui.error_dialog import (
            classify_startup_exception, show_startup_error,
        )
        # Mirror the ASR-init exit-43 path: log the full traceback to
        # main-pipeline.log BEFORE showing the dialog so a
        # device-not-found failure is triageable (the dialog itself
        # only carries the short details blob).
        log.exception("audio capture startup failed")
        err = classify_startup_exception("audio capture", exc)
        show_startup_error(err)
        sys.exit(43)


def _exit_code_for(app: PipelineApp) -> int:
    """The process exit code for an app end-state. Precedence:

        stop_all(43) > startup_failure(46) > stream_death(44) >
        stage_escalation(45) > overlay_giveup(47) > 0 (clean stop).

    A user quit (stop_all) always wins: a deliberate quit must STOP, never
    RESTART, and needs no error dialog. An overlay-server port-bind failure
    (46, STOP_ALL) outranks the recoverable RESTART codes (44/45) -- a bind
    conflict won't self-heal, so surfacing the port dialog beats a silent
    restart. A capture death (44) outranks a stage escalation (45): both
    RESTART, but 44 carries the device-specific give-up copy in the
    watchdog. An overlay give-up (47) is the lowest-priority nonzero code
    (itself a RESTART); any more-specific reason wins.

    Single source for amain's return code (clean path AND its
    except-teardown path), main()'s asyncio-cleanup hard-exit, and the
    capture-stop wedge hard-exit, so the precedence can't drift between them.

    Flag access is getattr-with-False-default on purpose: the two exception
    paths call this with whatever landed in app_box, and tests exercise them
    with partial stubs that only define the flags they arm -- a missing flag
    must read as "not requested", never AttributeError.
    """
    if getattr(app, "stop_all_requested", False):
        return 43
    if getattr(app, "startup_failure_requested", False):
        return EXIT_STARTUP_FAILURE
    if getattr(app, "stream_death_requested", False):
        return EXIT_STREAM_DEATH
    if getattr(app, "stage_escalation_requested", False):
        return EXIT_STAGE_ESCALATION
    if getattr(app, "overlay_giveup_requested", False):
        return EXIT_CHILD_GIVEUP
    return 0


# Log labels for _exit_code_for's nonzero codes, shared by amain's
# except-teardown return and main()'s asyncio-cleanup hard-exit so both
# log lines keep naming the intent, not just the number.
_INTENT_LABELS = {
    43: "user quit",
    EXIT_STARTUP_FAILURE: "overlay startup failure",
    EXIT_STREAM_DEATH: "capture death",
    EXIT_STAGE_ESCALATION: "stage escalation",
    EXIT_CHILD_GIVEUP: "overlay give-up",
}


async def _stop_capture_bounded(app: PipelineApp, remaining_s: float) -> None:
    """Stop audio capture within the teardown budget.

    capture.stop() makes synchronous native WASAPI/PortAudio calls
    (stop_stream / close / _pa.terminate) that can wedge on a stuck driver, so
    it runs on a worker thread under a timeout. A TIMEOUT means that worker is
    abandoned on asyncio's DEFAULT executor -- and when amain returns,
    asyncio.run's Runner.close() re-joins the default executor via
    shutdown_default_executor(THREAD_JOIN_TIMEOUT == 300 s on Python 3.13)
    BEFORE main() can reach its own _hard_exit. That hangs a tray-Quit ~5 min on
    a stuck driver, defeating the hard-exit (issue #55). So on a timeout we
    _hard_exit NOW with the correct code, skipping the join entirely. A QUICK
    exception means the worker already finished (no abandoned thread, no join
    risk) -- log and return so the rest of teardown still runs.

    We use asyncio.wait (not wait_for) to tell those two cases apart by the
    task's STATE, not by exception type: on 3.11+ asyncio.TimeoutError IS the
    builtin TimeoutError, and wait_for re-raises a worker exception, so a
    capture.stop() that itself raised TimeoutError would be indistinguishable
    from a genuine wedge under `except asyncio.TimeoutError`. With wait, a wedge
    leaves the task PENDING (hard-exit) while any quick return -- success or
    exception -- leaves it DONE (log + continue).
    """
    task = asyncio.ensure_future(asyncio.to_thread(app.capture.stop))
    done, _pending = await asyncio.wait({task}, timeout=remaining_s)
    if task not in done:
        # Still running past the budget = wedged on the native call. Its worker
        # is stuck on the default executor; _hard_exit before amain returns so
        # asyncio.run's Runner.close() can't re-join it for ~300s (issue #55).
        log.warning(
            "teardown: capture.stop wedged past budget; hard-exiting to skip "
            "the 300s default-executor re-join (issue #55)"
        )
        _hard_exit(_exit_code_for(app))
    exc = task.exception()
    if exc is not None:
        # Finished quickly with an error (worker already done -- no abandoned
        # thread, no join hazard). Log and let the rest of teardown run.
        log.warning("teardown: capture.stop raised %s; abandoning", type(exc).__name__)


async def _save_glossary_bounded(glossary_mgr, remaining_s: float) -> None:
    """Save the glossary within the teardown budget, off the DEFAULT executor.

    save() is fsync'd disk I/O that can wedge on a dying disk / hung
    filter driver. The previous shape -- wait_for(shield(to_thread(save)))
    -- returned at the deadline but ABANDONED the wedged worker on
    asyncio's DEFAULT executor, and asyncio.run's Runner.close() re-joins
    that executor via shutdown_default_executor (THREAD_JOIN_TIMEOUT ==
    300 s) BEFORE main() can reach its _hard_exit: the same issue-#55
    stall class _stop_capture_bounded hard-exits around, keeping
    yamabiko.exe alive minutes after tray-Quit. The save is lower-stakes
    than capture.stop (nothing later in teardown depends on it), so
    instead of a hard-exit: run it on a DEDICATED daemon thread and
    bounded-join. join(timeout) always returns promptly, so the to_thread
    worker below always completes (the default executor stays clean), and
    on a wedge the DAEMON flag lets the process exit with the save thread
    still stuck. Genre data persists on the next periodic save anyway.
    Save errors are logged in the worker; they must not abort the rest of
    teardown.
    """
    def _save() -> None:
        try:
            glossary_mgr.save()
        except Exception:
            log.exception("glossary save failed")

    t = threading.Thread(target=_save, name="glossary-save", daemon=True)
    t.start()
    # The join (not the save) is what runs on the default executor: it
    # returns at the deadline even when the save doesn't, so this task
    # can never leave a stuck worker behind for Runner.close() to re-join.
    await asyncio.to_thread(t.join, max(remaining_s, 0.0))
    if t.is_alive():
        log.warning(
            "glossary save still running at the teardown deadline; "
            "abandoning the daemon save thread (issue-#55 class: it must "
            "not be joinable at interpreter exit)",
        )


@dataclass
class _OverlaySupervision:
    """The overlay child's inner job, shutdown event, stop signal and
    supervisor thread, as run_pipeline's teardown needs them."""

    inner_job: Any
    writer: Any
    stop: threading.Event
    supervisor: Any
    thread: threading.Thread


def _start_overlay_supervisor(
    app: PipelineApp, ws_token: str, lifecycle_cfg: dict,
) -> _OverlaySupervision:
    """Create the inner Job Object and start the overlay supervisor thread.

    Overlay is pipeline's child (hybrid design): pipeline death -> inner
    job closes -> overlay killed by kernel. Shutdown ping via named event
    scoped to this pipeline's PID.

    Nesting forms when overlay is assigned to inner_job: overlay
    already inherits the outer watchdog job via child-of-in-job
    inheritance (Windows kernel rule), and explicitly assigning it
    to inner_job then creates the nested-jobs hierarchy. Pipeline
    itself is NOT a member of inner_job -- pipeline's natural death
    closes its last inner_job handle, which fires KILL_ON_JOB_CLOSE
    on the overlay via the nest; pipeline is unaffected because it
    isn't a member. This keeps the cascade invariant without the
    pipeline-self-kill side effect that an inner_job.assign(None)
    would introduce.
    """
    import subprocess as _subprocess
    from yamabiko.lifecycle.jobs import create_inner_job, spawn_in_job
    from yamabiko.lifecycle.shutdown_event import (
        ShutdownEventWriter, ENV_VAR_BY_ROLE,
    )
    from yamabiko.lifecycle.pipeline_supervisor import PipelineSupervisor
    from yamabiko import watchdog as _wd

    inner_job = create_inner_job("yamabiko-pipeline-inner")
    overlay_writer = ShutdownEventWriter.create(
        "overlay", parent_pid=os.getpid(),
    )

    def _overlay_cmd_env() -> "tuple[list, dict]":
        env = os.environ.copy()
        env[ENV_VAR_BY_ROLE["overlay"]] = overlay_writer.name
        env[WS_TOKEN_ENV] = ws_token
        # Forward overlay.websocket_port so the overlay connects
        # where uvicorn actually serves, and logging.level as
        # --log-level so `logging.level: DEBUG` raises overlay
        # verbosity too. Both come from APP.CFG -- the config this
        # pipeline actually runs with (honouring an explicit --config
        # file) -- NOT a fresh load_full_config() disk read inside
        # _build_overlay_cmd, which silently ignored --config (the
        # same documented bug class as run_pipeline's lifecycle read).
        # _build_overlay_cmd normalizes the value (INFO fallback), so
        # a bad config level can't crash-loop the overlay child.
        return _wd._build_overlay_cmd(
            url=_overlay_ws_url(app.cfg),
            level=((app.cfg or {}).get("logging") or {}).get("level"),
            config_path=app.explicit_config_path,
        ), env

    def _spawn_overlay() -> _subprocess.Popen:
        """Legacy two-step spawn (Popen, then the supervisor assigns the
        proc to the inner job). Kept only as PipelineSupervisor's
        required ``spawn=`` fallback seam; production routes through
        _spawn_overlay_in_job."""
        cmd, env = _overlay_cmd_env()
        return _subprocess.Popen(cmd, env=env, stdin=_subprocess.DEVNULL)

    def _spawn_overlay_in_job(job) -> _subprocess.Popen:
        """Atomic spawn -- jobs.spawn_in_job assigns the overlay to
        the inner job while it is still CREATE_SUSPENDED, so a pipeline
        death mid-spawn (taskkill /F between Popen and assign) can't
        leave a running overlay outside the KILL_ON_JOB_CLOSE cascade
        (pipeline owns overlay via the inner job)."""
        cmd, env = _overlay_cmd_env()
        return spawn_in_job(cmd, job, env=env, stdin=_subprocess.DEVNULL)

    overlay_stop = threading.Event()
    # Expose the overlay supervisor's stop signal so the pipeline's
    # wait loop (and tests) can observe a definitive overlay give-up and
    # tear the pipeline down instead of running headless.
    app.overlay_stop = overlay_stop
    # Overlay grace window + anti-flap knobs wired from lifecycle.*
    # (these are the consumers for those YAML keys; without this, the
    # keys have no effect and user_overrides.yaml changes silently
    # vanish).
    _overlay_grace_s = float(
        (lifecycle_cfg.get("grace_s") or {}).get("overlay", 3.0)
    )
    # Overlay anti-flap, lifecycle.overlay.*.
    _overlay_cfg = lifecycle_cfg.get("overlay") or {}
    # Call-site half (the supervisor also guarantees the first spawn on
    # its own): max_restarts < 1 is a user_overrides foot-gun that would
    # mean "never spawn the overlay at all". Clamp to 1 and say so.
    _max_restarts = int(_overlay_cfg.get("max_restarts", 3))
    if _max_restarts < 1:
        log.warning(
            "lifecycle.overlay.max_restarts=%d is below the minimum of "
            "1; clamping to 1 so the overlay still gets its initial "
            "spawn", _max_restarts,
        )
        _max_restarts = 1
    overlay_supervisor = PipelineSupervisor(
        spawn=_spawn_overlay,
        # Atomic spawn path -- overlay joins the inner job while
        # still CREATE_SUSPENDED, closing the taskkill-mid-spawn window
        # in which a running overlay isn't yet a job member and would
        # survive KILL_ON_JOB_CLOSE. ``spawn=`` stays as the
        # supervisor's required legacy fallback seam.
        spawn_in_job=_spawn_overlay_in_job,
        inner_job=inner_job,
        shutdown_writer=overlay_writer,
        stop_event=overlay_stop,
        grace_s=_overlay_grace_s,
        max_restarts=_max_restarts,
        restart_window_s=float(_overlay_cfg.get("restart_window_s", 60.0)),
    )
    # Expose the supervisor so the pipeline's wait loop (and tests) can
    # read its definitive give-up state (gave_up) after overlay_stop
    # fires -- overlay_stop alone conflates a respawn-budget give-up with
    # a normal overlay exit and the teardown's own set().
    app.overlay_supervisor = overlay_supervisor
    overlay_thread = threading.Thread(
        target=overlay_supervisor.run, daemon=True, name="overlay-supervisor",
    )
    overlay_thread.start()
    return _OverlaySupervision(
        inner_job, overlay_writer, overlay_stop, overlay_supervisor,
        overlay_thread,
    )


async def _teardown_pipeline(
    app: PipelineApp, pipeline_tasks: list[asyncio.Task],
    server_task: asyncio.Task, deadline: float, *,
    overlay: _OverlaySupervision, shutdown_listener: Any,
) -> None:
    """Stop everything run_pipeline started; every wait spends only the
    budget left before ``deadline`` (time.monotonic seconds)."""

    def _td_remaining() -> float:
        return max(deadline - time.monotonic(), 0.5)

    # Start the overlay shutdown ladder FIRST: it has no dependency on the
    # pipeline stages -- it's pure overlay teardown -- so it overlaps the
    # stage drain AND the server drain below. With the ladder created after
    # the stage drain, a wedged stage would serialize drain + ladder past the
    # budget: in production (grace 10 -> budget 8) the ladder would start at
    # t=8 and the watchdog would hard-kill at t=10 mid-ladder. The stop event
    # is set BEFORE the ladder runs so the supervisor loop can't respawn the
    # overlay mid-shutdown; stop() is idempotent and thread-safe (it is also
    # the supervisor thread's own give-up path).
    overlay.stop.set()
    _overlay_ladder = asyncio.create_task(
        asyncio.to_thread(overlay.supervisor.stop)
    )

    # Cancel pipeline stages; they're short-lived coroutines.
    for t in pipeline_tasks:
        t.cancel()
    # Retrieve cancelled/crashed stage exceptions deterministically here,
    # rather than letting them surface at GC time as "Task exception was
    # never retrieved". Bounded: a stage that doesn't yield to cancellation
    # (drain semantics, or an executor call whose await defers the cancel)
    # would eat the watchdog grace without being charged to the budget.
    # NOTE: asyncio.wait, NOT wait_for(gather(...)) -- wait_for waits for the
    # inner future's CANCELLATION to complete, which a stubborn child can
    # hold up indefinitely; asyncio.wait never cancels and genuinely returns
    # at the timeout.
    _done_stages, _pending_stages = await asyncio.wait(
        pipeline_tasks, timeout=_td_remaining(),
    )
    for _t in _done_stages:
        if _t.cancelled():
            continue
        _t_exc = _t.exception()
        if _t_exc is not None:
            # The supervised stage bodies log + swallow their own
            # failures (supervise / _supervised_chunker), so any
            # exception that reaches HERE was never logged anywhere --
            # the health refresher was the live case: it died silently,
            # /healthz served stale data until exit, and the bare
            # `.exception()` retrieval consumed the only evidence.
            # Log-only on purpose: exit-code semantics are unchanged.
            log.error(
                "pipeline task %r died with an unlogged exception; "
                "retrieved at teardown",
                _t.get_name(), exc_info=_t_exc,
            )
    if _pending_stages:
        log.warning(
            "%d pipeline stage(s) still draining at the teardown "
            "deadline; proceeding -- late stage exceptions may "
            "surface as 'never retrieved' warnings",
            len(_pending_stages),
        )

    # Ask uvicorn to exit gracefully and wait for it, instead of cancelling
    # mid-drain (which produces "Task was destroyed while pending" and drops
    # WebSocket close frames). uv_server and server_task are always set.
    # The wait must outlast uvicorn's own drain bound
    # (timeout_graceful_shutdown) -- the old 3.0 s literal cancelled the
    # server task mid-drain whenever a connected WS client was slow to
    # complete its close handshake.
    app.uv_server.should_exit = True
    try:
        await asyncio.wait_for(
            server_task,
            timeout=min(_SERVER_DRAIN_WAIT_S, _td_remaining()),
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        server_task.cancel()
    except SystemExit as exc:
        # uvicorn calls sys.exit(1) inside serve() when the overlay port
        # can't be bound (e.g. :8765 already in use by a second launch or an
        # unrelated app). Awaiting the finished task re-raises that
        # SystemExit -- a BaseException that would otherwise escape this
        # teardown, skipping ALL remaining cleanup (glossary save, capture
        # stop, translator + CUDA close) and misleading the watchdog into a
        # generic "configuration error" dialog. Swallow it so cleanup runs;
        # log the real cause.
        log.error(
            "overlay server exited abnormally -- the overlay port is "
            "likely already in use: %s", exc,
        )
    except Exception:
        log.exception("overlay server task raised during shutdown")
    if app.glossary_mgr is not None:
        # save() is fsync'd disk I/O; keep it off the loop even at teardown
        # and bound it by the shared budget -- a contended disk must not
        # starve the overlay ladder-await/join downstream. Runs on a
        # dedicated DAEMON thread, never the default executor: the old
        # shield-and-abandon shape left a wedged save worker for
        # asyncio.run's Runner.close() to re-join for up to 300 s after
        # tray-Quit (issue-#55 class); see _save_glossary_bounded.
        try:
            await _save_glossary_bounded(app.glossary_mgr, _td_remaining())
        except Exception:
            # Thread.start() raises under resource exhaustion, and the
            # save is the lowest-stakes step in this block (genre data
            # re-persists on the next periodic save). It must never take
            # the rest of teardown with it: the hotkey unregister below
            # would be skipped, leaving a low-level keyboard hook alive
            # past process exit, and the overlay shutdown ladder would
            # never run.
            log.warning("glossary save at teardown failed", exc_info=True)
    if app.hotkey_mgr is not None:
        try:
            app.hotkey_mgr.unregister()
        except Exception:
            log.exception("hotkey unregister failed")
    if shutdown_listener is not None:
        try:
            shutdown_listener.close()
        except Exception:
            log.exception("shutdown listener close failed")
    # Overlay supervisor cleanup. The shutdown ladder (stop()) was started
    # before the server drain and has been running in a worker thread
    # since; kernel-enforced death via inner job closing remains the
    # belt-and-suspenders backstop. Await the ladder within the shared
    # teardown budget -- the to_thread task is shielded because a thread
    # can't be cancelled; on timeout it finishes in the background while
    # teardown proceeds.
    try:
        await asyncio.wait_for(
            asyncio.shield(_overlay_ladder), timeout=_td_remaining(),
        )
    except asyncio.TimeoutError:
        log.warning(
            "overlay shutdown ladder still running at the teardown "
            "deadline; proceeding with handle close",
        )
    except Exception:
        log.exception("overlay supervisor stop failed")
    finally:
        # Let the supervisor thread finish its shutdown ladder
        # (writer.signal -> wait grace_s -> terminate +2s -> kill +2s)
        # against LIVE handles before we close them. Without the join, this
        # thread can close overlay_writer/inner_job in the window between the
        # supervisor releasing its lock and calling writer.signal() -- the
        # graceful SetEvent ping then silently no-ops (handle=None) and the
        # overlay is always terminate/killed. The join spends the remaining
        # teardown budget; on timeout we proceed (same behaviour as before
        # the join) rather than hang teardown into the watchdog's hard-kill.
        overlay.thread.join(timeout=_td_remaining())
        if overlay.thread.is_alive():
            log.warning(
                "overlay supervisor thread still alive at the "
                "teardown deadline; closing its handles anyway",
            )
        try:
            overlay.writer.close()
        except Exception:
            log.exception("overlay writer close failed")
        try:
            overlay.inner_job.close()
        except Exception:
            log.exception("inner job close failed")
    # Bound the tail. capture.stop() (PyAudio terminate) and
    # translator.aclose() (httpx pool drain) with NO timeout after the
    # deadline-bounded section let a stuck audio driver / in-flight HTTP
    # push teardown well past the budget. Cap each at the remaining budget.
    # capture.stop is special: a TIMEOUT abandons a native worker on the
    # default executor that asyncio.run's Runner.close() would then re-join
    # for ~300 s (issue #55), so _stop_capture_bounded hard-exits on that
    # timeout instead of returning into that join. translator.aclose simply
    # abandons on timeout (its drain has no such re-join hazard).
    log.info("teardown: stopping capture")
    await _stop_capture_bounded(app, _td_remaining())
    log.info("teardown: closing translator")
    try:
        await asyncio.wait_for(app.translator.aclose(), timeout=_td_remaining())
    except (asyncio.TimeoutError, Exception):
        log.warning("teardown: translator.aclose did not finish in budget; abandoning")
    app.cuda_pool.shutdown(wait=False, cancel_futures=True)
    if app.latency_log is not None:
        app.latency_log.close()
    log.info("teardown: complete (stop_all=%s)", app.stop_all_requested)
    print("\n=== latency summary ===")
    print(json.dumps(app.stats.summary(), indent=2))


async def run_pipeline(app: PipelineApp) -> None:
    """Register hotkeys, start capture + stage tasks + browser server;
    block until done.set() or any task completes; run cleanup in finally.

    Hotkey registration lives here (not in build_pipeline), next to the
    browser server that populates app.manager; the hotkey adapters
    (PipelineApp._on_*_hotkey) read app.manager at fire time.
    """
    post_asr_telemetry = PostASRTelemetry()
    app.post_asr_telemetry = post_asr_telemetry
    # Tier A shutdown wiring: if our parent (the watchdog) created a
    # named Win32 Event and passed its name via
    # YAMABIKO_SHUTDOWN_EVENT, open it and start the daemon watcher
    # thread. On signal, the thread calls
    # loop.call_soon_threadsafe(app.done.set) -- which already drives
    # the existing graceful-shutdown path in the outer finally block.
    # Install this BEFORE hotkey registration so an early shutdown
    # (e.g. watchdog cancels startup) still has a live listener.
    #
    # When no parent set the env var, from_env() returns None and this
    # is a no-op; direct invocations (python -m yamabiko.main) and the
    # current watchdog path both continue to work unchanged.
    from yamabiko.lifecycle.shutdown_event import ShutdownEventListener
    shutdown_listener = ShutdownEventListener.from_env(role="pipeline")
    if shutdown_listener is not None:
        shutdown_listener.start_watcher(asyncio.get_running_loop(), app.done)

    import secrets

    # The overlay knobs and the teardown budget below read APP.CFG --
    # already loaded once by build_pipeline via _load_config(args.config)
    # -- NOT a fresh load_full_config() disk read, which silently ignored
    # an explicit --config file's lifecycle block (and could disagree with
    # the rest of run_pipeline, e.g. the hotkeys block below, which has
    # always read app.cfg). The `or {}` chain tolerates a sparse config, so
    # missing keys still fall back to the PipelineSupervisor defaults.
    _cfg_lifecycle = (app.cfg or {}).get("lifecycle") or {}
    # Per-launch /ws session token. Generated before the overlay
    # supervisor starts (its first spawn reads it) and handed to build_app
    # below; every WS client, the Qt overlay included, must present it.
    ws_token = secrets.token_urlsafe(16)
    overlay = _start_overlay_supervisor(app, ws_token, _cfg_lifecycle)

    # Register global hotkeys. Keep the manager alive for the whole
    # asyncio run; _teardown_pipeline unregisters it. The bodies live on
    # PipelineApp (handle_*_hotkey), the keyboard-thread adapters too
    # (_on_*_hotkey); Settings re-binds live through apply_hotkeys.
    hk_cfg = app.cfg.get("hotkeys", {}) or {}
    if hk_cfg.get("enabled", True):
        try:
            app.hotkey_mgr = hk_mod.HotkeyManager()
            # A pipeline the watchdog restarted keeps quiet about shortcuts
            # the previous one already announced as taken (Decision 72).
            app.announced_hotkeys = problems.hotkeys_from_env(
                os.environ.get(problems.CARRY_HOTKEYS_ENV),
            )
            app.register_hotkeys()
        except Exception as exc:
            log.warning("hotkey manager unavailable: %s", exc)
            app.hotkey_mgr = None

    # If the capture stream dies (device unplugged / WASAPI reset), signal
    # a graceful shutdown instead of stalling forever with no audio. The
    # monitor fires this from its own thread, so marshal onto the loop.
    _capture_loop = asyncio.get_running_loop()
    # A capture death (unplug / suspend / active-but-silent) must RESTART and
    # re-resolve the device, not silently stop: route it through
    # request_stream_death so amain returns EXIT_STREAM_DEATH (RESTART), not 0
    # (STOP_SELF). The monitor fires this from its own thread, so marshal it.
    app.capture.on_stream_death = lambda: _capture_loop.call_soon_threadsafe(
        app.request_stream_death
    )
    _start_capture_or_exit(app)
    log.info(
        "capture started -- play JP audio through your default output device",
    )
    # A pipeline the watchdog restarted on problems (problems.CARRY_ENV)
    # resolved them before it accepts clients: the translator answered a
    # real translation in build_pipeline and capture is running. Tell the
    # clients that connect in the next seconds (the overlay process is new
    # and never saw the problem; plan Decision 18).
    carried = problems.kinds_from_env(os.environ.get(problems.CARRY_ENV))
    if carried:
        app.app_state.set_carried_recoveries(
            carried, until=time.monotonic() + _CARRIED_RECOVERY_S,
        )
        log.info("restarted after: %s; new clients hear it recovered", ",".join(carried))
    if app.max_segments > 0:
        log.info("will exit after %d segments", app.max_segments)

    # The uvicorn/FastAPI server is the display + control bus:
    # /ws feeds the Qt overlay's subtitles and the tray's
    # pause/genre/quit commands, /healthz + /livez feed the watchdog
    # probes. It must start unconditionally. overlay.browser gates
    # ONLY whether the browser overlay page (/ + /static) is served;
    # gating the whole server on it used to kill subtitles, tray Quit
    # (pipeline + llama left running), and the health probes in one
    # silent stroke. Inline default True matches the shipped default.
    port = int(app.cfg["overlay"].get("websocket_port", 8765))
    serve_browser_ui = bool(app.cfg["overlay"].get("browser", True))
    # Quit callback: when the overlay (tray) sends
    # {"type":"command","action":"quit"}, browser_server.py calls this. It
    # fires `done` (driving the existing teardown in the outer finally block,
    # block) AND marks stop_all_requested so amain returns STOP_ALL (43)
    # -- a user quit must STOP the app, never be read as a crash and
    # restarted.
    fastapi_app = build_app(
        state=app.app_state,
        on_quit=app.request_stop_all,
        on_pause_change=app.set_paused,
        on_hotkeys_change=app.apply_hotkeys,
        health_probes=HealthProbes(app=app),
        serve_browser_ui=serve_browser_ui,
        # Pin the /ws Origin guard to THIS server's port: build_app
        # requires bound_port, and a browser origin is allowed only on
        # exactly this loopback port, so a co-resident localhost
        # service's page can't connect to the subtitle bus.
        bound_port=port,
        # The same token the overlay child received in its environment.
        ws_token=ws_token,
        # Reject set_series at the wire when the series feature is off, so
        # no WS client can drive a disabled feature. Same helper (both
        # flags) every other entry point gates on.
        gazetteer_enabled=feature_enabled(app.cfg),
    )
    app.manager = fastapi_app.state.manager
    uv_cfg = uvicorn.Config(
        fastapi_app, host="127.0.0.1", port=port,
        log_level="warning", access_log=False,
        loop="asyncio",
        http="h11",
        ws="websockets",
        lifespan="on",
        # Bounds drain inside the parent grace window. Shared constant:
        # the finally's outer wait must exceed this so uvicorn's own
        # drain isn't cancelled mid-flight.
        timeout_graceful_shutdown=_UV_GRACEFUL_DRAIN_S,
    )
    app.uv_server = uvicorn.Server(uv_cfg)
    # Prevent uvicorn from installing its own SIGINT/SIGTERM handlers;
    # the pipeline's outer KeyboardInterrupt path handles shutdown.
    app.uv_server.install_signal_handlers = lambda: None
    log.info(
        "overlay server starting on http://127.0.0.1:%d/ (browser ui: %s)",
        port, "on" if serve_browser_ui else "off",
    )

    # on_escalate callback: stage crashed past threshold -> exit for a
    # watchdog RESTART (arms stage_escalation_requested -> exit 45).
    # The supervisor calls this synchronously from the event loop,
    # so it must not block. Extracted to PipelineApp.on_stage_escalate
    # so unit tests can invoke it directly.
    on_escalate = app.on_stage_escalate

    async def _supervised_chunker() -> None:
        # The chunker (Silero VAD) must not run UNSUPERVISED. A crash in
        # onnxruntime inference / np.concatenate escaped run(), completed the
        # task silently, asyncio.wait returned, the finally cancelled the
        # rest, and the whole app exited 0 -- the watchdog saw a clean quit
        # (no restart, no dialog) and the exception surfaced only as a
        # GC-time "Task exception was never retrieved". Surface it like the
        # other stages: log + escalate (sets done) so shutdown is intentional
        # and explained. A dead VAD means no audio chunks reach ASR, so the
        # whole pipeline is useless -- escalate-on-crash is correct here
        # (vs the recoverable stages' 10-in-60s threshold).
        try:
            await app.chunker.run(app.audio_q, app.chunk_q)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("VAD/chunker stage crashed; shutting down pipeline")
            on_escalate("vad")

    # Task names feed the teardown drain's died-with-unlogged-exception
    # ERROR line; the default "Task-N" would make that line untriageable.
    pipeline_tasks = [
        asyncio.create_task(_supervised_chunker(), name="vad-chunker"),
        asyncio.create_task(asr_stage(
            app.chunk_q, app.transcript_q, app.asr, app.cuda_pool,
            app.audio_cfg.sample_rate,
            tracker=app.stage_tracker,
            on_escalate=on_escalate,
            post_asr_telemetry=post_asr_telemetry,
            open_windows=app.chunker,
            incremental_enabled=app.app_state.incremental_enabled,
        ), name="asr-stage"),
        asyncio.create_task(translation_stage(
            app.transcript_q, app.translator, app.ctx, app.app_state,
            app.stats, app.latency_log, app.max_segments, app.done,
            app.manager,
            pipeline=app.romanization_pipeline,
            tracker=app.stage_tracker,
            on_escalate=on_escalate,
            transcript_enabled=app.transcript_enabled,
            series_hint_enabled=app.series_hint_enabled,
            post_asr_telemetry=post_asr_telemetry,
            remainder_state=app.remainder_state,
            on_translated=app.note_translation_ok,
        ), name="translation-stage"),
    ]
    # Expose the ASR task handle so HealthProbes.asr_alive can observe it.
    # Runtime attribute assignment on PipelineApp (a plain @dataclass,
    # not slots/frozen) is fine; mirrors the existing hotkey_mgr /
    # manager / uv_server pattern where run_pipeline populates fields
    # that build_pipeline leaves unset.
    app._asr_task = pipeline_tasks[1]

    # Periodic translator reachability probe, 10 s between ticks: slow
    # enough that a busy llama-server isn't hammered. Timeouts keep a hung
    # llama-server from stalling a tick: 2 s on the /v1/models GET, and
    # health_says_ready's own 5 s on the /health GET that follows an
    # answered one. So a tick takes up to about 7 s before its sleep, and
    # the watchdog's 5 s /healthz poll sees data up to about 17 s old.
    async def _refresh_translator_health() -> None:
        url = app.translator.health_check_url
        async with loopback_http.async_client(url, timeout=2.0) as client:
            while True:
                models_ok = False
                try:
                    r = await client.get(url)
                    models_ok = r.status_code == 200
                    # note_translator_health owns the cache write AND the
                    # flip-only WARNING/recovery logging (unit-testable
                    # method rather than closure state).
                    app.note_translator_health(models_ok, f"HTTP {r.status_code}")
                except Exception as exc:
                    # CancelledError inherits from BaseException (not Exception),
                    # so shutdown cancellation propagates correctly through here.
                    app.note_translator_health(False, exc)
                # The translator problem (plan phase 4): raised on a failing
                # probe once /v1/models has answered in this process, with no
                # completed translation since the last tick; cleared once
                # /health says ready.
                await app.track_translator_problem(models_ok, client)
                await asyncio.sleep(10.0)

    async def _serve_overlay() -> None:
        # uvicorn's serve() calls sys.exit(1) when it can't bind the
        # overlay port (e.g. :8765 already in use). A SystemExit raised
        # inside a task propagates to the event loop and cancels the whole
        # run_pipeline coroutine -- skipping cleanup and (via the watchdog's
        # exit-code classify) surfacing a misleading "configuration error"
        # dialog. Catch it HERE so it never reaches the loop: log the real
        # cause and signal a clean shutdown via app.done, so the finally
        # runs its full teardown (glossary save, capture stop, translator +
        # CUDA close) like any other shutdown.
        try:
            await app.uv_server.serve()
        except asyncio.CancelledError:
            raise
        except (SystemExit, OSError) as exc:
            log.error(
                "overlay server failed to start -- the overlay port (%d) is "
                "likely already in use: %s", port, exc,
            )
            # A bind failure must surface as a typed startup failure, not
            # masquerade as a user quit: arm startup_failure so amain
            # returns EXIT_STARTUP_FAILURE (46 -> watchdog give-up + port
            # dialog). Without this the exit is 0 (STOP_SELF) and the
            # watchdog silently cascades a full shutdown.
            app.startup_failure_requested = True
            app.done.set()

    # The server is built unconditionally (app.uv_server is ALWAYS set a
    # few lines up), so no `if app.uv_server is not None:` guard is
    # needed here or in the finally below. The /healthz refresher and the
    # serve task always run.
    pipeline_tasks.append(
        asyncio.create_task(
            _refresh_translator_health(), name="translator-health-refresher",
        ),
    )
    server_task = asyncio.create_task(_serve_overlay())
    tasks = pipeline_tasks + [server_task]
    # Positive "fully wired, event loop about to run" sentinel. Tests
    # (E2E + installer-smoke) assert this fires exactly once during
    # default-launch.
    log.info("pipeline.ready")
    try:
        # Wait until either (a) done is set (quit command from tray;
        # max_segments reached; stage-escalation shutdown) or (b) any
        # task ends on its own (crash / normal completion). First to
        # finish wins; we then fall through to cleanup. Note:
        # previously max_segments=0 used asyncio.gather which only
        # unblocked on task completion, so done.set from the quit
        # command didn't actually exit the pipeline.
        done_task = asyncio.create_task(app.done.wait())
        # Forward a definitive overlay-supervisor give-up (it sets
        # overlay_stop when the overlay flaps past the anti-flap cap, or
        # exits cleanly -- see PipelineSupervisor.run) to the pipeline's
        # shutdown. Without this, a give-up left the pipeline running
        # headless: ASR + translation burning CPU/GPU with no overlay, no
        # tray, and /livez still green (the event loop is alive, just
        # headless) so the watchdog couldn't recover it.
        overlay_giveup_task = asyncio.create_task(asyncio.to_thread(overlay.stop.wait))
        try:
            wait_set = set(tasks) | {done_task, overlay_giveup_task}
            _, pending = await asyncio.wait(
                wait_set, return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for _wt in (done_task, overlay_giveup_task):
                if not _wt.done():
                    _wt.cancel()
        # A definitive supervision failure (respawn budget exhausted, or
        # the supervisor loop itself crashed) is the only overlay_stop
        # cause that must NOT read as a clean quit: the pipeline is about
        # to tear down, and without a typed reason amain returns 0
        # (STOP_SELF) and the watchdog silently stops the whole tree.
        # gave_up is set ONLY on the supervisor's two failure paths (a
        # normal overlay exit or a user quit leaves it False), and the
        # supervisor sets it BEFORE overlay_stop, so it is already visible
        # once the wait above returns. Arm EXIT_CHILD_GIVEUP (47 ->
        # watchdog RESTART, fresh overlay budget); stop_all still wins in
        # _exit_code_for if a user quit raced. getattr-with-False so a
        # stub supervisor (teardown-focused tests) reads as "no give-up".
        if getattr(overlay.supervisor, "gave_up", False):
            app.overlay_giveup_requested = True
        # Before a restart exit: tell the clients why, leave the carry and,
        # after raising a problem, hold the exit so the notification and the
        # overlay line show before teardown stops the overlay (plan Decision
        # 18, R1). Bounded: a stalled client must not hold the exit; this
        # runs before the teardown budget below starts.
        try:
            await asyncio.wait_for(
                app.announce_restart_problems(), timeout=_ANNOUNCE_BUDGET_S,
            )
        except Exception:
            log.warning("restart problem announcement failed", exc_info=True)
    finally:
        # Teardown spends out of ONE budget derived from the watchdog's
        # pipeline grace. Run sequentially against independent timeouts
        # the steps could sum past the grace (stage drain + 6 s server
        # drain + up to 7 s ladder) and the watchdog would hard-kill the
        # pipeline mid-cleanup -- so the deadline clock starts FIRST (the
        # watchdog's grace clock is already running), the overlay ladder
        # runs concurrent with BOTH the stage drain and the server drain,
        # and every wait uses the remaining budget. The -2.0 reserves room
        # for capture/translator/CUDA close after the joins.
        _pipeline_grace_s = float(
            (_cfg_lifecycle.get("grace_s") or {}).get("pipeline", 10.0)
        )
        _teardown_deadline = time.monotonic() + max(_pipeline_grace_s - 2.0, 2.5)
        await _teardown_pipeline(
            app, pipeline_tasks, server_task, _teardown_deadline,
            overlay=overlay, shutdown_listener=shutdown_listener,
        )


def _log_asyncio_exception(loop, context: dict) -> None:
    """asyncio loop-level exception handler: put the failure ("Task
    exception was never retrieved", callback crashes) into the role log,
    then chain to the loop's default handler so dev-console stderr
    output is unchanged. The default handler alone prints to a stderr
    the frozen bundle discards -- these failures were invisible in the
    field. Module-level (not a closure in amain) so tests invoke it
    directly.
    """
    log.error(
        "asyncio: %s", context.get("message"),
        exc_info=context.get("exception"),
    )
    loop.default_exception_handler(context)


async def amain(args: argparse.Namespace, app_box: dict | None = None) -> int:
    """Thin composition: build + run. Returns the process exit code: 43
    (STOP_ALL) on a user-initiated quit, 0 on a clean stop.

    `app_box`, when given, receives the built PipelineApp under key "app" so
    main() can read stop_all_requested even if asyncio.run's post-amain task
    cleanup raises AFTER this coroutine returns (the exit must still be 43 on a
    user quit). Optional so existing tests that call amain(args) keep working.

    Retained as a named symbol so existing tests that monkeypatch
    yamabiko.main.amain continue to work through the transition.
    New code should call build_pipeline / run_pipeline directly.
    """
    # Route loop-level failures into the log for the WHOLE lifetime of
    # this loop -- build phase included (fire-and-forget task crashes
    # during warmup would otherwise only surface at GC time).
    asyncio.get_running_loop().set_exception_handler(_log_asyncio_exception)
    app = await build_pipeline(args)
    if app_box is not None:
        app_box["app"] = app
    try:
        await run_pipeline(app)
    except KeyboardInterrupt:
        # Ctrl+C: let main()'s handler print + return cleanly. Never coerce
        # to 43 -- a console interrupt is not a tray/overlay quit.
        raise
    except BaseException as exc:  # noqa: BLE001 -- see below
        # A user quit (tray/overlay) set stop_all_requested before teardown.
        # If the teardown then raised, the exit must STILL be STOP_ALL (43) so
        # the watchdog stops rather than reading the user's quit as a crash and
        # restarting it. The catch is BaseException (not Exception) ON PURPOSE:
        # the dirty teardowns that loop into "keeps crashing" are precisely the
        # BaseException kinds the old `except Exception` missed -- asyncio.
        # CancelledError (a BaseException since 3.8, raised by task cancellation
        # during shutdown) and uvicorn's SystemExit. Those escaped, main() exited
        # code 1, and the watchdog restarted. A genuine failure with no stop-all intent
        # re-raises, surfacing as a non-zero exit -> watchdog restart, as before.
        # Precedence (43 > 44 > 45 > 0) lives in _exit_code_for, not here.
        _code = _exit_code_for(app)
        if _code:
            log.info("%s: teardown raised %s; returning exit code %d",
                     _INTENT_LABELS[_code], type(exc).__name__, _code)
            return _code
        raise
    # stop_all (user quit) wins over a racing stream death: a deliberately-quit
    # app must STOP, never restart. (Precedence lives in _exit_code_for.)
    _code = _exit_code_for(app)
    log.info("amain returning exit code %d (stop_all=%s)", _code, app.stop_all_requested)
    return _code


def _hard_exit(code: int) -> None:
    """Exit the process IMMEDIATELY with `code`, skipping interpreter finalization.

    sys.exit raises SystemExit and lets the interpreter shut down, which JOINS
    non-daemon threads. Native-extension threads (onnxruntime / CTranslate2 /
    WASAPI / keyboard) can MANGLE the delivered OS exit code during that join,
    so the watchdog reads a non-43 code and RESTARTS instead of honoring the
    quit's STOP_ALL (amain can return 43 and main can log "exited code
    43", yet the watchdog still restart-loops because the OS code it reads
    is not 43). os._exit delivers the exact code with no finalization.
    Same pattern as the wizard-cancel exit_wizard_cancelled() os._exit(42).
    Logs + latency
    summary are already flushed by run_pipeline's finally; flush logging once more
    since os._exit skips atexit.
    """
    logging.shutdown()
    os._exit(code)


def main() -> None:
    """Synchronous entry point for the pipeline. Thin shell:
    composition lives in build_pipeline.
    """
    args = parse_args()

    # Wire file logging BEFORE the wizard runs. Wizard download
    # failures need forensic data for users to send us -- without
    # this, a beta user hitting a 401 or a stall has nothing to
    # share. Attach to the ROOT logger so every module-level logger
    # inherits. Per-process file: this process (wizard +
    # pipeline) owns main-pipeline.log; watchdog and overlay write
    # their own files because sharing one breaks Windows rotation.
    log_config.setup_main_log(
        logging.getLogger(), level="INFO", filename="main-pipeline.log",
    )
    # Identity banner immediately after log setup and BEFORE the wizard
    # gate: a wizard-phase failure log must still carry the version /
    # frozen / PID / OS line (bug_report.yml points reporters at it).
    # No config digest here -- the merged config doesn't exist yet;
    # build_pipeline emits the "config: digest=..." line once it does.
    log_config.log_startup_banner(log, "pipeline")
    # Crash evidence immediately after log setup, BEFORE the wizard and
    # the heavy imports: faulthandler -> crash-pipeline.log for native
    # faults (CUDA/CT2/onnx), excepthooks -> main-pipeline.log for
    # unhandled Python exceptions. The wizard runs inside this process,
    # so this one call covers it too. Never raises.
    log_config.install_crash_evidence("pipeline")
    logging.getLogger(__name__).info(
        "yamabiko startup — HF_HOME=%s HF_XET_HIGH_PERFORMANCE=%s",
        os.environ.get("HF_HOME"),
        os.environ.get("HF_XET_HIGH_PERFORMANCE"),
    )

    # First-run wizard gate. Runs BEFORE pipeline startup. The wizard
    # writes state.json plus any user overrides to user_overrides.yaml;
    # under the layered-config model there is no config.yaml to bootstrap.
    if wizard_state.needs_wizard():
        if args.config is not None:
            # The wizard persists its picks to user_overrides.yaml, a
            # layer a --config run deliberately skips, and a dev/repro run
            # must not mutate state.json either. Refuse instead of running
            # setup inside an isolated run; exit 43 is the tailored-dialog
            # sentinel, so the watchdog stops instead of restarting.
            from yamabiko.ui.error_dialog import StartupError, show_startup_error
            show_startup_error(StartupError(
                title="Setup required",
                message=(
                    "First-run setup has not finished. Start Yamabiko once "
                    "without --config to complete setup, then retry with "
                    "--config."
                ),
                suggested_fix=(
                    "Start Yamabiko from the Start menu shortcut and finish "
                    "the setup wizard."
                ),
                details=f"--config {args.config}: first-run setup is incomplete",
            ))
            sys.exit(43)
        from yamabiko.wizard import exit_wizard_cancelled, run_wizard
        run_wizard()
        if wizard_state.needs_wizard():
            # User cancelled. Exit 42 is the sentinel the watchdog
            # uses to distinguish wizard-cancel from a clean run
            # (it triggers the shared stop_event so the overlay and
            # llama-server supervisors drop too). The helper hard-exits
            # via os._exit when a download thread is still draining, so
            # interpreter teardown can't qFatal on the running QThread
            # and turn the cancel into a watchdog RESTART.
            exit_wizard_cancelled()

    _mlog = logging.getLogger(__name__)
    app_box: dict = {}
    try:
        code = asyncio.run(amain(args, app_box))
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        return
    except BaseException as exc:
        # asyncio.run's OWN post-amain cleanup (cancelling leftover tasks /
        # closing async generators) raised AFTER amain returned -- this is
        # OUTSIDE amain's try/except, so a user quit whose teardown leaked a
        # task surfaces here as a non-zero crash exit -> watchdog restart
        # loop ("exit refuses to close"). If the user asked to quit,
        # the exit is STILL STOP_ALL(43); only a genuine non-quit crash re-raises.
        # Precedence (43 > 44 > 45) lives in _exit_code_for. Hard-exit ONLY on a
        # nonzero code (an armed intent flag); with no intent, fall through to
        # the issue-#56 startup-failure backstop below -- never hard-exit 0 here.
        _app = app_box.get("app")
        if _app is not None:
            _code = _exit_code_for(_app)
            if _code:
                _mlog.info("%s: asyncio.run cleanup raised %s; hard-exit %d",
                           _INTENT_LABELS[_code], type(exc).__name__, _code)
                _hard_exit(_code)
        # Startup-failure backstop (issue #56). build_pipeline /
        # _start_capture_or_exit show a tailored dialog then sys.exit(43); on a
        # build_pipeline failure app_box is EMPTY (the app was never returned),
        # so the guards above don't fire. Re-raising that SystemExit(43) runs
        # interpreter finalization, which joins the never-shut-down cuda_pool
        # CTranslate2/onnxruntime threads and can mangle the 43 -> the watchdog
        # restart-loops the tailored dialog instead of stopping. os._exit
        # delivers an unmangled 43 and skips the join. Specific to code 43 (the
        # tailored-dialog sentinel): any other SystemExit is a genuine abnormal
        # exit and must propagate.
        if isinstance(exc, SystemExit) and exc.code == 43:
            _mlog.info("startup failure SystemExit(43); hard-exit STOP_ALL(43)")
            _hard_exit(43)
        _mlog.error("asyncio.run raised %s with no quit intent; propagating",
                    type(exc).__name__)
        raise
    # Hard-exit unconditionally (issue #61): os._exit skips interpreter
    # finalization and the default-executor join, so non-daemon native-extension
    # threads (onnxruntime / CTranslate2 / WASAPI) can neither mangle the code
    # nor stall the process. This covers code 0 (named-event graceful shutdown:
    # SIGINT/SIGTERM, tray-Quit cascade, frozen-E2E graceful-shutdown test) as
    # well as 43 (user quit). os._exit(0) is identical to a normal return for
    # both the OS and the watchdog (classify(0) == STOP_SELF).
    _mlog.info("main: amain exited code %s", code)
    _hard_exit(code)


if __name__ == "__main__":
    main()
