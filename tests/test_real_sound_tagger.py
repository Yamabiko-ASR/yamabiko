"""Real-path test for the sound tagger's onnxruntime.InferenceSession seam.

The VAD assist (``yamabiko/vad/sound_tagger.py``) runs the bundled EfficientSED
graph on ONNX Runtime's CPU provider. This test runs the shipped bytes through
the production loader and streaming layout, with nothing mocked, and checks
them against the assist term the development bench computed with its own
code. A drift in onnxruntime, a wrong group column, weight, cap or window
alignment, or a changed graph makes the terms disagree.

The model ships in the repository, so the test never skips.

``cv17_ja_8lines.tag_assist.npy`` holds, per 512-sample chunker frame of the
committed CV17 fixture (scaled by 1/32767 and cut to whole frames), the term
``min(1, 1.5 * (voice + shout))`` from the bench's path in ``scripts/asr_bench``:
``tagger_onnx.OnnxTagger`` (4 threads) and ``tag_padded`` with hop 12,
look-ahead 6 and batch 1 over ``pad_item``, ``group_columns``/``group_probs``
with ``tag_sounds.GROUPS``, then ``tag_sounds.silero_frames``. Regenerate it by
running those calls on the same audio and saving the result with ``np.save``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.real

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"


async def test_shipped_tagger_reproduces_the_benched_assist_term_on_real_speech():
    from yamabiko.vad.sound_tagger import (
        FRAME_SAMPLES,
        SoundTagAssist,
        SoundTagger,
        default_model_path,
    )

    pcm = np.fromfile(_FIXTURE_DIR / "cv17_ja_8lines.s16le", dtype="<i2")
    audio = pcm.astype(np.float32) / 32767.0
    frames = audio[: audio.size // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    expected = np.load(_FIXTURE_DIR / "cv17_ja_8lines.tag_assist.npy")
    assert expected.shape == (frames.shape[0],)

    assist = SoundTagAssist(SoundTagger(default_model_path()))
    tagged = []
    for ordinal, frame in enumerate(frames):
        tagged.extend(await assist.push(frame, ordinal))
    tagged.extend(await assist.drain())

    assert [meta for _, _, meta in tagged] == list(range(frames.shape[0]))
    terms = np.array([term for _, term, _ in tagged], dtype=np.float32)
    worst = int(np.argmax(np.abs(terms - expected)))
    assert np.allclose(terms, expected, rtol=0.0, atol=1e-4), (
        f"frame {worst}: production term {terms[worst]:.6f}, "
        f"bench term {expected[worst]:.6f}"
    )
