"""Incremental emission: commit whole sentences while a VAD window is open.

The chunker hands a window to ASR only when it closes, so a sentence that
ends early in a long window waits for the rest of the window. While a window
is open, the ASR stage also decodes it as it grows (step decodes), and this
module decides which sentences are stable: LocalAgreement-2 over complete
sentences. A sentence ending in 。！？?! commits when it, and every sentence
before it, reads the same in the window's last two step decodes, both read
after the text the window has already committed. Step decodes of a window's
first second or two sometimes hear a lone うん。 or はい。 that the final
decode does not write, so such a sentence (``_held``) and every sentence
after it wait for the final decode instead. At the close the window's final
decode runs as before, and only the part after the committed text goes on
(``strip_committed``).

The ASR bench's ``incremental_emit`` strategy (scripts/asr_bench/
incremental.py) imports these functions, so the bench measures this rule.

Times are samples of the chunker's decided audio
(``SileroChunker.decided_samples``, the frames its gate has admitted). A
window's first step is due ``STEP_MS`` after the gate opened it; each later
one ``STEP_MS`` after the previous step's due time, or ``CONFIRM_MS`` after
it when that decode left a complete sentence uncommitted. A step whose audio
did not grow since the window's last step is skipped.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import numpy as np

from yamabiko.sentencing.splitter import SentenceSplitter, has_terminator

# From the bench winner inc-s1000-c250. Module constants, not config keys.
STEP_MS = 1000
CONFIRM_MS = 250

# One chunker frame: the shortest wait between two looks at the clock.
_FRAME_SAMPLES = 512

# A sentence whose content is one of these is held for the final decode
# (``_held``). They are the written forms of three entries of the ASR bench's
# interjection lexicon (scripts.asr_bench.lexicon.INTERJECTIONS: ウン "うん,
# うーん", ハイ "はい, はーい", エエ "ええ"), which production cannot import.
_HELD_INTERJECTION = re.compile(r"うー*ん|はー*い|ええ")

# Katakana ァ..ヶ (U+30A1..U+30F6) to hiragana ぁ..ゖ.
_TO_HIRAGANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


class OpenWindow(NamedTuple):
    """The window the chunker has open now (``open_window_snapshot()``)."""

    window_id: int
    # SileroChunker.decided_samples when the gate opened the window.
    opened_at_sample: int
    # A copy of the frames the gate has admitted to the window so far.
    audio: np.ndarray
    # Pause epoch of the window's frames; None for producers without epochs.
    pause_epoch: int | None
    # True when capture time was lost before this window (the Segment field
    # of the same name).
    discontinuity_before: bool


class OpenWindowSource(Protocol):
    """What the ASR stage reads from the chunker (SileroChunker)."""

    @property
    def decided_samples(self) -> int: ...

    @property
    def open_window_id(self) -> int | None: ...

    def open_window_snapshot(self) -> OpenWindow | None: ...


@dataclass(frozen=True)
class StepJob:
    """One step decode of an open window, with the audio it decodes."""

    window_id: int
    due_sample: int
    audio: np.ndarray
    pause_epoch: int | None
    discontinuity_before: bool
    # time.perf_counter() when the audio was copied.
    t_snapshot: float


@dataclass(frozen=True)
class StepOutcome:
    """What one step decode decided."""

    commits: tuple[str, ...]
    # Samples from this step's due time to the window's next step.
    interval_samples: int
    # True when these are the first sentences the window commits.
    first_commits: bool


def _is_content(char: str) -> bool:
    return not char.isspace() and unicodedata.category(char)[0] not in "PZS"


def content(text: str) -> tuple[str, list[int]]:
    """``text`` without punctuation, symbols and spaces, and the raw index of
    each kept character."""
    indices = [index for index, char in enumerate(text) if _is_content(char)]
    return "".join(text[index] for index in indices), indices


def strip_committed(committed: str, decoded: str) -> tuple[int, int]:
    """Where the part of ``decoded`` not yet committed starts, and the cost.

    The content characters of ``committed`` are aligned (edit distance)
    with every prefix of the content characters of ``decoded``; the
    shortest prefix with the lowest distance is the committed part. Returns
    the raw offset in ``decoded`` just after that prefix and the punctuation
    or spaces that follow it, and the distance.
    """
    wanted, _ = content(committed)
    if not wanted:
        return 0, 0
    got, indices = content(decoded)
    if not got:
        return len(decoded), len(wanted)
    target = np.frombuffer(got.encode("utf-32-le"), dtype=np.uint32)
    source = np.frombuffer(wanted.encode("utf-32-le"), dtype=np.uint32)
    steps = np.arange(target.size + 1, dtype=np.int32)
    row = steps.copy()
    for index in range(source.size):
        previous = row
        row = np.empty_like(previous)
        row[0] = index + 1
        np.minimum(previous[:-1] + (target != source[index]), previous[1:] + 1, out=row[1:])
        # Insertions run along the row: a running minimum of row - j, plus j.
        row = np.minimum.accumulate(row - steps) + steps
    prefix = int(np.argmin(row))
    cut = 0 if prefix == 0 else indices[prefix - 1] + 1
    while cut < len(decoded) and not _is_content(decoded[cut]):
        cut += 1
    return cut, int(row[prefix])


def split_sentences(splitter: SentenceSplitter, text: str) -> list[str]:
    """``text`` split by the production splitter; sentences stripped, empty
    ones dropped."""
    return [sentence.strip() for sentence in splitter.split(text) if sentence.strip()]


def agreed_sentences(previous: Sequence[str], current: Sequence[str]) -> list[str]:
    """LocalAgreement-2 on sentences: the longest common prefix of complete
    sentences."""
    agreed: list[str] = []
    for before, now in zip(previous, current):
        if before != now or not has_terminator(now):
            break
        agreed.append(now)
    return agreed


def complete_prefix(current: Sequence[str]) -> list[str]:
    """The leading complete sentences of one decode."""
    complete: list[str] = []
    for sentence in current:
        if not has_terminator(sentence):
            break
        complete.append(sentence)
    return complete


def _held(sentence: str) -> bool:
    """True when ``sentence``, after NFKC with katakana folded to hiragana,
    has no content or its content is a lone うん, はい or ええ (うーん and
    はーい too): step decodes sometimes hear one of these that the final
    decode does not write, so it never commits early."""
    text = unicodedata.normalize("NFKC", sentence).translate(_TO_HIRAGANA)
    found, _ = content(text)
    return found == "" or _HELD_INTERJECTION.fullmatch(found) is not None


def step_commits(
    splitter: SentenceSplitter, committed: str, previous: str | None, decoded: str,
) -> tuple[list[str], bool]:
    """The sentences step decode ``decoded`` commits, and whether it leaves a
    complete sentence uncommitted (the next step then comes CONFIRM_MS later).

    ``committed`` is the window's committed text so far, stripped from the
    front of ``decoded`` and of the window's ``previous`` step decode before
    both are split; without a previous decode nothing commits. A held
    sentence (``_held``) and every sentence after it wait for the final
    decode's remainder, so line order holds, and count as not complete, so
    they bring no CONFIRM_MS step.
    """
    current = split_sentences(splitter, decoded[strip_committed(committed, decoded)[0]:])
    complete = complete_prefix(current)
    # complete_prefix itself is not guarded: the bench's gap path calls it
    # directly and does not get the guard. The bench's production path
    # (agree) comes through here.
    held = next((index for index, sentence in enumerate(complete) if _held(sentence)), None)
    if held is not None:
        complete = complete[:held]
    if previous is None:
        return [], bool(complete)
    before = split_sentences(splitter, previous[strip_committed(committed, previous)[0]:])
    found = agreed_sentences(before, current)[:len(complete)]
    return found, len(complete) > len(found)


@dataclass
class _WindowState:
    # asr.incremental.enabled as read when the window was first seen.
    steps: bool
    next_due: int
    interval: int
    offered_samples: int = 0
    previous: str | None = None
    committed: list[str] = field(default_factory=list)


class IncrementalEmitter:
    """The step state of every window the ASR stage has seen and not yet
    finalized: the last step decode, the committed sentences and the next
    step's due time. One per pipeline."""

    def __init__(self, splitter: SentenceSplitter, sample_rate: int) -> None:
        self._splitter = splitter
        self._sample_rate = sample_rate
        self.step_samples = sample_rate * STEP_MS // 1000
        self._confirm_samples = sample_rate * CONFIRM_MS // 1000
        self._windows: dict[int, _WindowState] = {}
        self._current: int | None = None

    def due_step(
        self, source: OpenWindowSource, enabled: Callable[[], bool],
    ) -> StepJob | None:
        """The open window's step job when one is due, else None.

        ``enabled`` (asr.incremental.enabled) is read once, when a window is
        first seen, so the switch takes effect at the next window. Synchronous
        on purpose: it reads ``open_window_id`` and the snapshot on the
        chunker's event loop with no await between them, so the two agree.
        """
        open_id = source.open_window_id
        if open_id is None:
            self._current = None
            return None
        now = source.decided_samples
        state = self._windows.get(open_id)
        if state is None:
            self._current = open_id
            if not enabled():
                # Read before any snapshot: with the switch off no window is
                # ever copied, so the off path does what it did before
                # incremental emission.
                self._windows[open_id] = _WindowState(
                    steps=False, next_due=0, interval=self.step_samples,
                )
                return None
        elif not state.steps or now < state.next_due:
            return None
        snapshot = source.open_window_snapshot()
        if state is None:
            state = _WindowState(
                steps=True,
                # The window's opening frame (after a dip cut, the frame where
                # speech came back). Every due time of the window derives
                # from it; nothing re-bases it.
                next_due=snapshot.opened_at_sample + self.step_samples,
                interval=self.step_samples,
            )
            self._windows[open_id] = state
            if now < state.next_due:
                return None
        if snapshot.audio.size <= state.offered_samples:
            # No new audio since the window's last step: skip this one.
            state.next_due += state.interval
            return None
        state.offered_samples = snapshot.audio.size
        due = state.next_due
        # Until this step's decode says otherwise (on_step), the next step
        # follows a full STEP_MS later.
        state.next_due = due + self.step_samples
        return StepJob(
            window_id=open_id,
            due_sample=due,
            audio=snapshot.audio,
            pause_epoch=snapshot.pause_epoch,
            discontinuity_before=snapshot.discontinuity_before,
            t_snapshot=time.perf_counter(),
        )

    def seconds_until_due(self, source: OpenWindowSource) -> float:
        """How long the ASR stage may wait before it looks again: until the
        current window's next step, or one STEP_MS while no window with
        steps is known (a window that opens meanwhile is due no sooner)."""
        state = self._windows.get(self._current) if self._current is not None else None
        if state is None or not state.steps:
            return STEP_MS / 1000
        remaining = state.next_due - source.decided_samples
        return max(remaining, _FRAME_SAMPLES) / self._sample_rate

    def on_step(self, job: StepJob, text: str) -> StepOutcome:
        """Apply one step decode of ``job``'s window; return what it commits."""
        # A window keeps its state until a final for it, or for a later
        # window, runs; the ASR stage drops such a window's pending jobs first.
        state = self._windows[job.window_id]
        found, pending = step_commits(
            self._splitter, "".join(state.committed), state.previous, text,
        )
        first = bool(found) and not state.committed
        state.previous = text
        state.committed.extend(found)
        state.interval = self._confirm_samples if pending else self.step_samples
        state.next_due = job.due_sample + state.interval
        return StepOutcome(tuple(found), state.interval, first)

    def has_commits(self, window_id: int | None) -> bool:
        state = self._windows.get(window_id) if window_id is not None else None
        return state is not None and bool(state.committed)

    def on_final(self, window_id: int | None, final_text: str) -> tuple[str, bool]:
        """The part of a window's final decode after its committed sentences,
        and whether it had any. Drops the state of this window and of every
        earlier one (a window abandoned at a pause or capture gap never gets
        a final)."""
        state = self._windows.get(window_id) if window_id is not None else None
        self.drop_through(window_id)
        if state is None or not state.committed:
            return final_text, False
        cut, _cost = strip_committed("".join(state.committed), final_text)
        return final_text[cut:], True

    def drop_through(self, window_id: int | None) -> None:
        """Forget windows ``window_id`` and earlier."""
        if window_id is None:
            return
        for key in [key for key in self._windows if key <= window_id]:
            del self._windows[key]
        if self._current is not None and self._current <= window_id:
            self._current = None


class StepLane:
    """The step lane of the ASR stage's dispatcher (finals always run first).

    At most one pending step job per window: a newer job replaces an older
    one. When taken, a job whose window is no longer open, or whose due time
    is more than one STEP_MS in the past, is dropped. ``steps_dropped``
    counts every job that never ran.
    """

    def __init__(self, step_samples: int) -> None:
        self._step = step_samples
        self._jobs: dict[int, StepJob] = {}
        self.steps_dropped = 0

    def offer(self, job: StepJob) -> None:
        if job.window_id in self._jobs:
            self.steps_dropped += 1
        self._jobs[job.window_id] = job

    def take(self, now_sample: int, open_window_id: int | None) -> StepJob | None:
        for window_id in sorted(self._jobs):
            job = self._jobs.pop(window_id)
            if window_id != open_window_id or now_sample - job.due_sample > self._step:
                self.steps_dropped += 1
                continue
            return job
        return None

    def discard_through(self, window_id: int | None) -> None:
        """Drop the jobs of window ``window_id`` and earlier: its final runs now."""
        if window_id is None:
            return
        for key in [key for key in self._jobs if key <= window_id]:
            del self._jobs[key]
            self.steps_dropped += 1
