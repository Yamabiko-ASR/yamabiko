"""Real-path test: E0 replays the VAD production ships, on the shipped models.

E0 is the release gate for ASR changes, so its replay must gate audio exactly
as production's ``SileroChunker`` does: the bundled sound tagger, run through
production's ``SoundTagAssist`` delay line, raises Silero's probability, and
``FrameGate`` receives ``max(silero, tag term)``. This test runs both paths
on the committed CV17 speech fixture with the pinned Silero graph and the
bundled tagger on ONNX Runtime, nothing mocked, and requires identical
windows, identical per-frame gate probabilities and the same final gate state.

It keeps the committed contract pins (``scripts/e0_evidence/contracts.py``),
so it also fails when ``SileroChunker`` or ``yamabiko/vad/sound_tagger.py``
changed without E0's pins being re-approved.

Skipped only when the pinned Silero graph is not in the HF cache; the tagger
ships in the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

# Pin the production HF cache before any huggingface_hub import.
from yamabiko import hf_env  # noqa: F401
from yamabiko import paths

from scripts.e0_evidence.replay import (
    load_production_model,
    load_production_tagger,
    production_chunker_config,
    replay_master,
)
from tests.e0_evidence.test_replay import (
    _assert_replay_matches_production,
    _capture,
    _master,
    _production_gate,
    _RecordingSink,
)

pytestmark = pytest.mark.real

_PCM_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "cv17_ja_8lines.s16le"
)


def test_replay_reproduces_the_production_chunker_on_real_speech_with_the_shipped_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    from yamabiko import models

    cached = try_to_load_from_cache(
        models.SILERO_REPO,
        models.SILERO_FILE,
        revision=models.SILERO_REVISION,
    )
    if cached is None or cached is _CACHED_NO_EXIST:
        pytest.skip("the pinned Silero ONNX model is not cached")

    pcm = np.fromfile(_PCM_PATH, dtype="<i2").astype(np.float32) / 32767.0
    capture = _capture(_master(pcm.size, samples=pcm))
    shipped = yaml.safe_load(paths.shipped_defaults_path().read_text(encoding="utf-8"))
    config, threshold = production_chunker_config(shipped)

    result = replay_master(
        capture,
        config,
        threshold,
        load_production_model(config),
        load_production_tagger(),
        _RecordingSink(),
    )
    production = _production_gate(capture.samples, config, threshold, monkeypatch)

    assert result.windows, "the CV17 lines emitted no window"
    _assert_replay_matches_production(result, None, production, None)
