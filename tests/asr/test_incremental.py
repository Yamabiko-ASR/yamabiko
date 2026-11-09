"""Incremental emission's commit rule and final strip
(yamabiko/asr/incremental.py)."""
from __future__ import annotations

import numpy as np
import pytest

from yamabiko.asr.incremental import (
    CONFIRM_MS,
    STEP_MS,
    IncrementalEmitter,
    OpenWindow,
    strip_committed,
)
from yamabiko.sentencing.splitter import SentenceSplitter

_SR = 16000
_STEP = _SR * STEP_MS // 1000
_CONFIRM = _SR * CONFIRM_MS // 1000


class _Window:
    """Window 0, opened at sample 0, whose decided audio is the first
    ``decided_samples`` samples of ten seconds of silence."""

    def __init__(self) -> None:
        self.decided_samples = 0
        self.open_window_id = 0
        self._audio = np.zeros(10 * _SR, dtype=np.float32)

    def open_window_snapshot(self) -> OpenWindow:
        return OpenWindow(0, 0, self._audio[: self.decided_samples].copy(), None, False)


def _run_steps(
    texts: list[str], emitter: IncrementalEmitter | None = None,
) -> tuple[list[tuple[str, ...]], list[int]]:
    """One step decode per text, each run as soon as it is due (the clock
    advances one chunker frame at a time). Returns what each step committed
    and each step's due sample."""
    emitter = emitter or IncrementalEmitter(SentenceSplitter(), _SR)
    window = _Window()
    commits: list[tuple[str, ...]] = []
    dues: list[int] = []
    for text in texts:
        for _ in range(1000):
            window.decided_samples += 512
            job = emitter.due_step(window, lambda: True)
            if job is not None:
                break
        else:
            pytest.fail("no step became due")
        dues.append(job.due_sample)
        commits.append(emitter.on_step(job, text).commits)
    return commits, dues


def test_a_sentence_commits_when_the_last_two_decodes_agree_on_it():
    """Seeded defect: compare each decode with the first one, or with itself,
    instead of the previous one -> 今日は飴です。 never commits, or commits at
    the second step (red)."""
    commits, _ = _run_steps(["今日は雨です。", "今日は飴です。", "今日は飴です。"])
    assert commits == [(), (), ("今日は飴です。",)]


def test_an_unconfirmed_complete_sentence_brings_the_next_step_to_confirm_ms():
    """The first decode ends a sentence nothing has confirmed yet, so the
    next step is due CONFIRM_MS later, and that decode commits it.

    Seeded defect: always wait STEP_MS -> the second due is first + STEP (red)."""
    commits, dues = _run_steps(["今日は雨です。明日は", "今日は雨です。明日は晴れ"])
    assert commits == [(), ("今日は雨です。",)]
    assert dues == [_STEP, _STEP + _CONFIRM]


def test_an_incomplete_sentence_is_never_committed_even_when_decodes_agree():
    """Seeded defect: drop the has_terminator check from agreed_sentences ->
    明日は commits with 今日は雨です。 (red)."""
    commits, _ = _run_steps(["今日は雨です。明日は"] * 3)
    assert commits == [(), ("今日は雨です。",), ()]


def test_a_lone_interjection_two_decodes_agree_on_waits_for_the_final():
    """Step decodes of a window's first seconds sometimes hear a lone うん。
    that the final decode does not write, so it is not sent early, and the
    sentence after it waits with it to keep the line order.

    Seeded defect: no held-sentence cut in step_commits -> both sentences
    commit at the second step, which came CONFIRM_MS after the first (red)."""
    emitter = IncrementalEmitter(SentenceSplitter(), _SR)
    commits, dues = _run_steps(["うん。今日は晴れです。"] * 3, emitter)
    assert commits == [(), (), ()]
    # Nothing complete is pending, so no step comes CONFIRM_MS early.
    assert dues == [_STEP, 2 * _STEP, 3 * _STEP]
    assert emitter.on_final(0, "うん。今日は晴れです。") == ("うん。今日は晴れです。", False)


@pytest.mark.parametrize(("committed", "final", "rest"), [
    pytest.param("今日は雨です。", "今日は雨です。明日は晴れ。", "明日は晴れ。", id="exact"),
    pytest.param("今日は雨です。", "今日は飴です。明日は晴れ。", "明日は晴れ。", id="near-match"),
    pytest.param("", "今日は雨です。明日は晴れ。", "今日は雨です。明日は晴れ。", id="nothing-committed"),
])
def test_strip_committed_leaves_only_the_text_after_the_committed_sentences(
    committed, final, rest,
):
    """Seeded defect: an exact-prefix match instead of the edit-distance
    alignment -> near-match keeps the whole final (red); a cut that does not
    skip the punctuation after the prefix -> 。明日は晴れ。 (red)."""
    cut, _cost = strip_committed(committed, final)
    assert final[cut:] == rest
