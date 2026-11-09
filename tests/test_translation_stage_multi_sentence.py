"""translation_stage must emit every complete sentence in a multi-sentence chunk.

A single ASR chunk can carry more than one complete sentence. The stage
splits with fast-bunkai and emits each; none may be silently dropped or
merged. (This path used to interact with the removed turn detector; it is
now the plain emission behaviour.)
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from yamabiko.main import Segment, LatencyStats, translation_stage
from yamabiko.state import AppState
from yamabiko.supervisor import ErrorTracker
from yamabiko.translation.prompt import ContextBuffer


class _OkTranslator:
    def __init__(self):
        self.calls: list[str] = []

    async def translate(self, jp, genre, context):
        self.calls.append(jp)
        return "EN"


@pytest.mark.asyncio
async def test_translation_stage_emits_all_sentences_in_multi_sentence_chunk():
    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    translator = _OkTranslator()
    state = AppState(genre="variety")
    ctx = ContextBuffer(max_lines=2)
    stats = LatencyStats()

    # Two complete sentences in one chunk: both must be split out and
    # translated, none dropped or merged.
    await q.put(Segment(
        audio=np.zeros(1, dtype=np.float32),
        t_chunk_emit=time.perf_counter(),
        jp="最初の文です。二番目の文です。",
    ))

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

    # Both sentences must have been translated -- no silent drops or merges.
    assert len(translator.calls) == 2
    assert "最初の文です。" in translator.calls
    assert "二番目の文です。" in translator.calls
