"""Real-path integration tests for the onnxruntime.InferenceSession seam.

Yamabiko runs one ONNX model at inference time:

- **Pinned community Silero VAD v5-era ONNX graph**
  (``onnx-community/silero-vad``) — ~2 MB, CPU-only. The production wrapper
  accepts a 512-sample frame and prepends 64 retained samples, so session
  construction and the raw 576-sample graph call must both work.

It was fully mocked in earlier versions of the test suite. A drift in
onnxruntime's API, an ONNX-opset change on the upstream model, or a
``SessionOptions`` incompatibility would ship to users silently. This
file catches that by constructing a real session against the cached
ONNX file and running one real ``.run()``.

Skipped when the ONNX file isn't in the HF cache — run the wizard or
preflight first.
"""
from __future__ import annotations

import numpy as np
import pytest

# Pin HF cache before any huggingface_hub import reaches across.
from yamabiko import hf_env  # noqa: F401

pytestmark = pytest.mark.real


def _file_in_cache(repo: str, filename: str, revision: str | None = None) -> bool:
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    result = try_to_load_from_cache(repo, filename, revision=revision)
    return result is not None and result is not _CACHED_NO_EXIST


def _official_v5_reference(model_path, frames: list[np.ndarray]) -> list[float]:
    """Run the cached graph with upstream v5's 16 kHz context protocol."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"],
    )
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)
    probabilities: list[float] = []

    for frame in frames:
        graph_input = np.concatenate((context, frame[np.newaxis, :]), axis=1)
        prob, state = session.run(
            None, {"input": graph_input, "state": state, "sr": sr},
        )
        probabilities.append(float(prob[0][0]))
        context = frame[np.newaxis, -64:].copy()

    return probabilities


# ---------------------------------------------------------------------------
# Pinned community Silero VAD v5-era graph used by silero_chunker.py
# ---------------------------------------------------------------------------

def test_silero_vad_session_constructs_and_runs() -> None:
    """SileroVAD constructs, exposes the expected I/O contract, and
    produces a scalar probability for one external 512-sample float32 frame.

    This is the exact path the chunker takes at pipeline startup. A
    regression in onnxruntime's providers, session options, or the
    Silero model's opset would fail here — none of which the old
    ``test_silero_chunker.py`` (mocked via ``_FakeOrtSession``) could
    catch.

    The skip guard probes the PINNED revision (the snapshot the
    wizard warms), and the resolved path must live inside that snapshot
    — the runtime must not silently resolve a floating `main` copy.
    """
    from yamabiko import models

    if not _file_in_cache(
        models.SILERO_REPO, models.SILERO_FILE,
        revision=models.SILERO_REVISION,
    ):
        pytest.skip(
            "silero-vad ONNX not cached at the pinned revision — run the "
            "wizard or preflight first"
        )

    from yamabiko.vad.silero_chunker import SileroVAD, _default_model_path

    model_path = _default_model_path()
    assert models.SILERO_REVISION in str(model_path), (
        f"resolved silero path {model_path} is outside the pinned "
        f"snapshot {models.SILERO_REVISION} — revision pin not wired "
        f"through _default_model_path"
    )
    vad = SileroVAD(model_path, sample_rate=16000)

    # The chunker pumps 512 samples; SileroVAD supplies the graph's 64-sample
    # retained context internally.
    frame = np.zeros(512, dtype=np.float32)
    prob = vad(frame)

    assert isinstance(prob, float), f"expected float, got {type(prob)}"
    assert 0.0 <= prob <= 1.0, f"VAD probability {prob} out of range"


def test_silero_vad_matches_official_v5_context_reference() -> None:
    """Two consecutive calls and reset must match upstream v5's 64+512 rule."""
    from yamabiko import models

    if not _file_in_cache(
        models.SILERO_REPO, models.SILERO_FILE,
        revision=models.SILERO_REVISION,
    ):
        pytest.skip("silero-vad ONNX not cached at the pinned revision")

    from yamabiko.vad.silero_chunker import SileroVAD, _default_model_path

    model_path = _default_model_path()
    vad = SileroVAD(model_path, sample_rate=16000)

    rng = np.random.default_rng(seed=42)
    frames = [
        (rng.standard_normal(512) * scale).astype(np.float32)
        for scale in (0.01, 0.05)
    ]
    expected = _official_v5_reference(model_path, frames)

    actual = [vad(frame) for frame in frames]

    vad.reset_states()
    actual_after_reset = vad(frames[0])

    assert actual == expected
    assert actual_after_reset == _official_v5_reference(
        model_path, [frames[0]],
    )[0]
