"""Real tier for incremental emission: production faster-whisper decodes the
open window as it grows (step decodes), and the ASR stage commits the first
of two sentences before the window closes; the final decode then adds only
the second.

The window is scripted (the clip's decided audio grows in real time) so the
two sentences are exactly one window: everything after the chunker is real.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from yamabiko import hf_env  # noqa: F401  # pin the model cache first
from yamabiko import models, paths
from scripts.e0_evidence.normalize_ja import normalize_minimal
from tests.asr.test_real_faster_whisper import _cuda_available, _model_in_cache

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"
_PCM_PATH = _FIXTURE_DIR / "cv17_ja_8lines.s16le"
_META_PATH = _FIXTURE_DIR / "cv17_ja_8lines.json"
_MIN_SIMILARITY = 0.75


def _similarity(ref: str, out: str) -> float:
    """Share of ``ref`` characters found in matching blocks of ``out``."""
    matcher = difflib.SequenceMatcher(None, ref, out, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(ref)


class _RealTimeWindow:
    """Window 0 of ``audio``, decided as fast as it would play: at wall time
    t after construction the first t * sample_rate samples are decided."""

    def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.started = time.monotonic()
        self.open = True

    @property
    def decided_samples(self) -> int:
        elapsed = time.monotonic() - self.started
        return min(self.audio.size, int(elapsed * self.sample_rate))

    @property
    def open_window_id(self):
        return 0 if self.open else None

    def open_window_snapshot(self):
        from yamabiko.asr.incremental import OpenWindow

        if not self.open:
            return None
        return OpenWindow(0, 0, self.audio[: self.decided_samples].copy(), 0, False)


@pytest.mark.real
async def test_first_sentence_commits_before_the_window_closes_and_the_final_adds_only_the_second():
    """Seeded defect: forward the whole final decode (no strip_committed) ->
    the remainder holds both sentences and is longer than 1.5 x the second
    (red); steps never scheduled -> nothing early before the close (red)."""
    if not _cuda_available():
        pytest.skip("no NVIDIA GPU detected; skipping real incremental emission")
    if not _model_in_cache(models.WHISPER_REPO, "model.bin", revision=models.WHISPER_REVISION):
        pytest.skip(f"{models.WHISPER_REPO}/model.bin not in HF cache")

    from yamabiko.asr.asr_client import build_asr
    from yamabiko.main import Segment, asr_stage
    from yamabiko.supervisor import ErrorTracker

    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    sample_rate = meta["sample_rate"]
    pcm = np.fromfile(_PCM_PATH, dtype="<i2").astype(np.float32) / 32767.0
    first, second = meta["lines"][1], meta["lines"][3]
    clip = np.concatenate([
        pcm[first["start_sample"]:first["end_sample"]],
        np.zeros(int(0.3 * sample_rate), dtype=np.float32),
        # Speech end plus about 0.1 s: decided audio stays below the 7.0 s
        # step, so the final carries sentence 2, and under the 8 s cap.
        pcm[second["start_sample"]:second["start_sample"] + int(2.9 * sample_rate)],
    ])

    cfg = yaml.safe_load(paths.shipped_defaults_path().read_text(encoding="utf-8"))
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda")
    asr = await loop.run_in_executor(pool, build_asr, cfg["asr"])
    await loop.run_in_executor(pool, asr.warmup)

    chunk_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    window = _RealTimeWindow(clip, sample_rate)
    task = asyncio.create_task(asr_stage(
        chunk_q, out_q, asr, pool, sample_rate, tracker=ErrorTracker(),
        open_windows=window, incremental_enabled=lambda: True,
    ))
    early_before_close = []
    late_early = []
    rest = None
    crashed = None
    try:
        while window.decided_samples < clip.size:
            await asyncio.sleep(0.05)
        while not out_q.empty():
            early_before_close.append(out_q.get_nowait())
        window.open = False
        chunk_q.put_nowait(Segment(
            audio=clip, t_chunk_emit=time.perf_counter(), window_id=0, pause_epoch=0,
        ))
        deadline = time.monotonic() + 30.0
        while rest is None and time.monotonic() < deadline and not task.done():
            await asyncio.sleep(0.05)
            while not out_q.empty():
                item = out_q.get_nowait()
                if item.early:
                    late_early.append(item)
                else:
                    rest = item
        if task.done() and not task.cancelled():
            crashed = task.exception()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pool.shutdown(wait=True)

    assert crashed is None, f"asr_stage died: {crashed!r}"
    ref1 = normalize_minimal(first["sentence"])
    ref2 = normalize_minimal(second["sentence"])
    early_text = normalize_minimal("".join(item.jp for item in early_before_close))
    print(f"early before close: {[item.jp for item in early_before_close]!r}")
    print(f"final remainder: {rest.jp if rest is not None else None!r}")
    assert early_before_close and all(item.early for item in early_before_close)
    assert _similarity(ref1, early_text) >= _MIN_SIMILARITY, early_text
    assert rest is not None and rest.follows_early, (
        "final remainder missing or not follows_early; "
        f"early items after the close: {[item.jp for item in late_early]!r}"
    )
    rest_text = normalize_minimal(rest.jp)
    assert _similarity(ref2, rest_text) >= _MIN_SIMILARITY, rest_text
    assert len(rest_text) <= 1.5 * len(ref2), "the final remainder repeats the committed sentence"
