"""Real-path integration test for the FasterWhisperASR seam.

Verifies that the entire ASR stack (WhisperModel construction + transcribe
+ warmup) works correctly by actually constructing a WhisperModel against
the real library and running a real transcription against numpy-generated
audio. Catches regressions in CTranslate2 compute_type, Blackwell sm_120
kernel support, faster-whisper API shape, and model-path resolution.

Skipped gracefully when:
- CUDA isn't available (no GPU)
- The model file isn't in the HF cache yet (don't force a 1.6 GB
  download from within a test run)

Both of those skip conditions are real-world "not ready to test" signals
and should not fail. The release gate verifies the happy path by
ensuring the wizard pre-populates the cache before this test runs.
"""
from __future__ import annotations

import numpy as np
import pytest

# Pin HF cache before any huggingface_hub import reaches across.
from yamabiko import hf_env  # noqa: F401
from yamabiko import models

pytestmark = pytest.mark.real


def _cuda_available() -> bool:
    """Best-effort GPU detect without pulling torch in."""
    from yamabiko.gpu import detect_vram_mib
    return detect_vram_mib() is not None


def _model_in_cache(repo: str, filename: str, *, revision: str) -> bool:
    """Check HF cache at ``revision`` without triggering a download.

    The wizard downloads at the pinned commit, which writes only
    snapshots/<sha>/ and no refs/main, so an unpinned probe misses a
    wizard-warmed cache and the real tests skip.
    """
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    result = try_to_load_from_cache(repo, filename, revision=revision)
    return result is not None and result is not _CACHED_NO_EXIST


# Source the repo from yamabiko.models — the single pin source the
# wizard, runtime, and preflight all share. A hardcoded literal could
# silently drift from models.py; sourcing it here means a models.py
# repo change reaches this test automatically.
_WHISPER_REPO = models.WHISPER_REPO


def _wizard_warmed() -> bool:
    """True iff every file the wizard downloads for whisper is cached.

    Gated on the WIZARD's download list (the producer), not on the
    runtime's probe (``models.whisper_files_cached``, the consumer) —
    importing the consumer here would make the offline-gate assertion
    below tautological.
    """
    from yamabiko.wizard_pages.downloads import (
        _WHISPER_FILES, _WHISPER_REPO as _wiz_repo, _WHISPER_REVISION,
    )
    # No repo-drift assert: both _WHISPER_REPO above and the wizard's
    # _WHISPER_REPO alias resolve to models.WHISPER_REPO, the single pin
    # source — comparing them would be a tautology.
    return all(
        _model_in_cache(_wiz_repo, f, revision=_WHISPER_REVISION)
        for f in _WHISPER_FILES
    )


@pytest.fixture(scope="module")
def asr():
    """Construct a real ASR via the PRODUCTION composition (build_asr).

    Skipped if GPU or cache missing. Uses build_asr — not direct
    FasterWhisperASR construction — so the offline branch
    (``WhisperModel(..., local_files_only=True)`` on a warmed cache)
    executes against the real library + real cache.
    """
    if not _cuda_available():
        pytest.skip("no NVIDIA GPU detected — skipping real ASR test")

    if not _model_in_cache(
        _WHISPER_REPO, "model.bin", revision=models.WHISPER_REVISION,
    ):
        pytest.skip(
            f"{_WHISPER_REPO}/model.bin not in HF cache — run the wizard "
            f"first or let preflight warm it. Not triggering a 1.6 GB "
            f"download from a test."
        )

    from yamabiko.asr.asr_client import build_asr

    return build_asr({
        "faster_whisper": {
            "model": _WHISPER_REPO,
            "compute_type": "float16",
            "device": "cuda",
        },
    })


def test_faster_whisper_construction_and_warmup(asr):
    """Constructing WhisperModel + running warmup must not raise.

    This is the seam that would have caught a CTranslate2 compute_type
    regression or a faster-whisper version pin drift. The warmup call
    issues a transcribe against 1 second of silence — exercises the
    full tensor path through CUDA.
    """
    asr.warmup()


def test_faster_whisper_transcribe_returns_asrresult(asr):
    """Transcribing real audio returns an ASRResult with a string + duration.

    We use 2 seconds of silence — whisper won't crash on silence and
    the test is about the plumbing (construction, tensor dtypes, CUDA
    kernel invocation), not transcription quality.
    """
    from yamabiko.asr.asr_client import ASRResult

    sample_rate = 16000
    silence = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
    result = asr.transcribe(silence, sample_rate)

    assert isinstance(result, ASRResult)
    assert isinstance(result.text, str)
    assert result.duration_s == pytest.approx(2.0, abs=0.01)


def test_faster_whisper_handles_short_chunk(asr):
    """A 500 ms chunk — within the normal VAD-gated range — should not crash.

    The pipeline hands short chunks through when speech + pad + silence
    lands below the 1-second mark. faster-whisper has historically had
    edge cases with very short audio; this guards against that.
    """
    sample_rate = 16000
    chunk = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    result = asr.transcribe(chunk, sample_rate)
    assert isinstance(result.text, str)


# ---------------------------------------------------------------------------
# Offline-load seam: build_asr passes local_files_only=True iff
# _whisper_cached() says the cache is warm. Two distinct real-path
# failure modes need coverage:
#   (a) the runtime probe under-triggers on a wizard-warmed cache
#       (the probe set in models.whisper_files_cached drifts to a file
#       the wizard never downloads) -> every warmed install silently
#       degrades to online startup re-checks, re-opening the original
#       offline-load bug;
#   (b) the offline load itself crashes on a warmed cache
#       (models.WHISPER_FILES misses a file faster-whisper actually
#       requires, or the library drops/renames local_files_only) ->
#       hard startup crash on a warmed install with no online fallback.
# ---------------------------------------------------------------------------


def test_whisper_offline_gate_fires_on_wizard_warmed_cache():
    """On a wizard-warmed cache the runtime probe MUST return True, so
    build_asr takes the offline (local_files_only=True) path. Real
    probe, real cache, no GPU needed.

    Seeded-defect check: probe a filename the wizard never downloads
    (extra entry in models.whisper_files_cached's loop) -> probe
    returns False on every warmed install -> this fails.
    """
    if not _wizard_warmed():
        pytest.skip(
            f"{_WHISPER_REPO} not wizard-warmed (some of the wizard's "
            f"whisper files missing from HF cache) — run the wizard or "
            f"preflight first"
        )

    from yamabiko.asr.asr_client import _whisper_cached

    assert _whisper_cached(_WHISPER_REPO) is True, (
        "wizard-warmed cache but _whisper_cached() returned False — "
        "the probe (models.whisper_files_cached) has drifted from the "
        "wizard download list; every warmed install silently degrades to "
        "online startup re-checks despite the model being cached"
    )


def test_whisper_offline_load_constructs_and_warms_on_warmed_cache():
    """``WhisperModel(..., local_files_only=True)`` against the REAL
    library and REAL warmed cache must construct and survive a real
    warmup transcription. This is the call test_build_asr.py stubs out;
    if the required-file list misses something faster-whisper needs
    offline, or the library rejects/renames the kwarg, a warmed install
    hard-crashes at startup with no online fallback.

    Seeded-defect check: typo the kwarg at the WhisperModel boundary in
    asr_client.py (``local_files_only`` -> ``local_files_onlyX``) ->
    real library raises TypeError -> this fails.
    """
    if not _cuda_available():
        pytest.skip("no NVIDIA GPU detected — skipping real ASR test")
    if not _wizard_warmed():
        pytest.skip(
            f"{_WHISPER_REPO} not wizard-warmed — run the wizard or "
            f"preflight first"
        )

    from yamabiko.asr.asr_client import FasterWhisperASR

    # Same call shape as build_asr for the default repo: the pinned
    # revision, because the wizard warms snapshots/<sha>/ with no refs/main.
    asr_offline = FasterWhisperASR(
        model=_WHISPER_REPO,
        compute_type="float16",
        device="cuda",
        local_files_only=True,
        revision=models.WHISPER_REVISION,
    )
    asr_offline.warmup()


def test_junk_gate_drops_and_keeps_real_faster_whisper_segments(monkeypatch):
    """The junk gate reads ``no_speech_prob`` / ``avg_logprob`` /
    ``compression_ratio`` off faster-whisper Segment objects and decides
    keep-vs-drop. The unit tests (tests/asr/test_asr_junk_gates.py) fake
    the Segment with a SimpleNamespace, so they are blind to two real
    regressions: (a) a production attribute-read typo, and (b) a
    faster-whisper version bump renaming or dropping a field.

    This feeds REAL faster_whisper Segment instances through the REAL
    ``FasterWhisperASR.transcribe`` gate and asserts the confident
    no-speech segment is dropped and the clean one kept. The GPU model is
    the only thing stubbed (the subject is the Segment field contract +
    the gate, not CUDA inference), so it needs no GPU or cache.

    Seeded-defect (a): rename a field read in asr_client.transcribe
    (``seg.no_speech_prob`` -> ``seg.no_speechX``) -> AttributeError on
    the real Segment during the gate -> red. Seeded-defect (b): if the
    library renames the field, constructing the real Segment below raises
    TypeError -> red.
    """
    from types import SimpleNamespace

    from yamabiko.asr import faster_whisper_audio_compat

    faster_whisper_audio_compat.install_audio_compat()
    from faster_whisper.transcribe import Segment as RealSegment

    def _real_seg(text, no_speech, logprob, compression):
        return RealSegment(
            id=0, seek=0, start=0.0, end=1.0, text=text, tokens=[],
            avg_logprob=logprob, compression_ratio=compression,
            no_speech_prob=no_speech, words=None, temperature=0.0,
        )

    # Junk: confident no-speech (no_speech>0.6 AND avg_logprob<-1.0).
    # Clean: confident speech. Both short, so the length-floored
    # compression gate is not in play here.
    junk = _real_seg("ゴミ", no_speech=0.9, logprob=-1.5, compression=1.0)
    clean = _real_seg("本物のセリフ", no_speech=0.02, logprob=-0.2, compression=1.0)

    class _StubModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            return iter([junk, clean]), SimpleNamespace(language="ja")

    # Stub ONLY the GPU model construction; the real Segment dataclass and
    # the real gate logic stay in play.
    monkeypatch.setattr(
        faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _StubModel,
    )

    from yamabiko.asr.asr_client import FasterWhisperASR

    asr = FasterWhisperASR("m", "float16", "cuda")
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "本物のセリフ", (
        f"gate must drop the confident-no-speech real Segment and keep "
        f"the clean one; got {out.text!r}"
    )


def test_no_speech_only_gate_drops_real_confident_phantom_segment(monkeypatch):
    """The no-speech-only gate reads no_speech_prob off REAL faster-whisper
    Segment objects. Seeded defect: read a renamed field -> AttributeError
    on the real Segment -> red; ignore the threshold -> phantom kept -> red."""
    from types import SimpleNamespace

    from yamabiko.asr import faster_whisper_audio_compat

    faster_whisper_audio_compat.install_audio_compat()
    from faster_whisper.transcribe import Segment as RealSegment

    def _real_seg(text, no_speech, logprob):
        return RealSegment(
            id=0, seek=0, start=0.0, end=1.0, text=text, tokens=[],
            avg_logprob=logprob, compression_ratio=1.0,
            no_speech_prob=no_speech, words=None, temperature=0.0,
        )

    phantom = _real_seg("おはようございます。", no_speech=0.97, logprob=-0.3)
    clean = _real_seg("本物のセリフ", no_speech=0.02, logprob=-0.2)

    class _StubModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            return iter([phantom, clean]), SimpleNamespace(language="ja")

    monkeypatch.setattr(faster_whisper_audio_compat, "import_whisper_model", lambda: _StubModel)
    from yamabiko.asr.asr_client import FasterWhisperASR

    asr = FasterWhisperASR("m", "float16", "cuda", no_speech_only_threshold=0.95)
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "本物のセリフ"
