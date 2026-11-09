"""Real-path line recall: every spoken Common Voice line reaches translation.

The committed CC0 fixture (tests/fixtures/audio/cv17_ja_8lines.*) holds
eight clean Japanese read-speech lines separated by silence. It is fed in
capture-sized blocks through the production SileroChunker, the real
asr_stage (the default whisper model via build_asr on CUDA) and the real
translation_stage. A stub translator records every Japanese string it is
asked to translate; each fixture line must appear in that record.

A line held back anywhere between VAD and the translator (a chunk that
never closes, a transcript dropped by a stage, a sentence tail buffered
with no later chunk to release it) makes its similarity collapse and the
test fail.
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
_BLOCK_SAMPLES = 480
_QUIET_S = 60.0
_CAP_S = 180.0
_MIN_SIMILARITY = 0.75


class _RecordingTranslator:
    """Translator seam stub: records every Japanese input it receives."""

    def __init__(self) -> None:
        self.jp: list[str] = []
        self.last_call: float | None = None

    async def translate(self, jp, genre, context, **kw):
        self.jp.append(jp)
        self.last_call = time.monotonic()
        return "translated"


def _similarity(ref: str, out: str) -> float:
    """Share of ``ref`` characters found in matching blocks of ``out``.

    autojunk is off: its popularity heuristic treats frequent kana as junk
    once ``out`` reaches 200 characters, which would make the score depend
    on output length instead of content.
    """
    matcher = difflib.SequenceMatcher(None, ref, out, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(ref)


@pytest.mark.real
async def test_cv_ja_lines_all_reach_translation():
    if not _cuda_available():
        pytest.skip("no NVIDIA GPU detected; skipping real pipeline recall")
    if not _model_in_cache(
        models.WHISPER_REPO, "model.bin", revision=models.WHISPER_REVISION,
    ):
        pytest.skip(f"{models.WHISPER_REPO}/model.bin not in HF cache")
    if not _model_in_cache(
        models.SILERO_REPO, models.SILERO_FILE, revision=models.SILERO_REVISION,
    ):
        pytest.skip(f"{models.SILERO_REPO}/{models.SILERO_FILE} not in HF cache")

    from yamabiko.asr.asr_client import build_asr
    from yamabiko.capture.envelope import CaptureBlock
    from yamabiko.main import LatencyStats, asr_stage, translation_stage
    from yamabiko.state import AppState
    from yamabiko.supervisor import ErrorTracker
    from yamabiko.translation.prompt import ContextBuffer
    from yamabiko.vad.chunker import chunker_config_from_config
    from yamabiko.vad.silero_chunker import SileroChunker

    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    sample_rate = meta["sample_rate"]
    pcm = np.fromfile(_PCM_PATH, dtype="<i2").astype(np.float32) / 32767.0

    # Shipped defaults, read only; components built as build_pipeline does.
    cfg = yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    assert cfg["audio"]["sample_rate"] == sample_rate
    app_state = AppState(genre=cfg.get("genre", "variety"))
    chunker_cfg, threshold = chunker_config_from_config(cfg)
    chunker = SileroChunker(chunker_cfg, threshold=threshold, state=app_state)

    loop = asyncio.get_running_loop()
    cuda_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda")
    asr = await loop.run_in_executor(cuda_pool, build_asr, cfg["asr"])
    await loop.run_in_executor(cuda_pool, asr.warmup)

    audio_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    chunk_q: asyncio.Queue = asyncio.Queue(maxsize=8)
    transcript_q: asyncio.Queue = asyncio.Queue(maxsize=16)
    translator = _RecordingTranslator()
    done = asyncio.Event()

    async def _feed() -> None:
        for start in range(0, pcm.size, _BLOCK_SAMPLES):
            await audio_q.put(CaptureBlock(
                pcm=pcm[start:start + _BLOCK_SAMPLES].copy(),
                stream_generation=0,
                start_sample=start,
            ))

    tasks = [
        asyncio.create_task(_feed(), name="feed"),
        asyncio.create_task(chunker.run(audio_q, chunk_q), name="vad-chunker"),
        asyncio.create_task(asr_stage(
            chunk_q, transcript_q, asr, cuda_pool, sample_rate,
            tracker=ErrorTracker(),
        ), name="asr-stage"),
        asyncio.create_task(translation_stage(
            transcript_q, translator,
            ContextBuffer(max_lines=cfg["translation"]["context_lines"]),
            app_state, LatencyStats(), None, 0, done,
            manager=None,
            tracker=ErrorTracker(),
        ), name="translation-stage"),
    ]
    started = time.monotonic()
    try:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            last = translator.last_call or started
            if now - last >= _QUIET_S or now - started >= _CAP_S:
                break
        crashed = [
            f"{task.get_name()}: {task.exception()!r}"
            for task in tasks
            if task.done() and not task.cancelled()
            and task.exception() is not None
        ]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        cuda_pool.shutdown(wait=True)
    assert not crashed, f"pipeline task died: {crashed}"

    out = normalize_minimal("".join(translator.jp))
    report = []
    for index, line in enumerate(meta["lines"]):
        ref = normalize_minimal(line["sentence"])
        similarity = _similarity(ref, out)
        report.append((index, line["clip"], similarity, ref))
        print(
            f"line {index} {line['clip']} similarity={similarity:.3f} "
            f"ref={ref}"
        )
    missed = [row for row in report if row[2] < _MIN_SIMILARITY]
    assert not missed, (
        "lines not recalled at translation (similarity < "
        f"{_MIN_SIMILARITY}):\n"
        + "\n".join(
            f"  line {i} {clip}: {sim:.3f} ref={ref}"
            for i, clip, sim, ref in report
        )
        + f"\ntranslator inputs: {translator.jp!r}"
    )
