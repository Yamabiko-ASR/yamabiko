"""Unit tests for ASR boundary silence padding (measured CER win, 2026-07-05).

Whisper models truncate the first/last syllables of SHORT clips -- a
documented boundary artifact whose community fix is to pad the clip with a
little leading + trailing silence. Measured on our harness on kotoba-whisper
(the default model then; 3 titles, 1530 chunks) chunk-CER improved
47.70->46.20 (digimon), 30.56->27.34 (turn-a), 32.03->30.01 (chainsaw) with
latency and junk-gate-empty counts unchanged. The pad is kept for the current
default model.

These are unit tier: the whisper model is faked so no weights load. The pad is
applied inside FasterWhisperASR.transcribe, BEFORE the model call, so a
capturing fake model sees the padded array.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _seg(text, no_speech=0.0, logprob=-0.1, compression=1.0):
    return SimpleNamespace(
        text=text, no_speech_prob=no_speech, avg_logprob=logprob,
        compression_ratio=compression,
    )


def _capturing_asr(monkeypatch, *, boundary_pad_ms, segments=None):
    """Patch the project seam with a model that CAPTURES the audio array
    handed to transcribe(), and return (asr, captured) where captured['audio']
    is that array after construction with the given pad."""
    captured: dict = {}
    segs = segments if segments is not None else [_seg("x")]

    class _CapturingModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            captured["audio"] = audio
            return iter(list(segs)), SimpleNamespace(language="ja")

    from yamabiko.asr import asr_client

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _CapturingModel,
    )
    FasterWhisperASR = asr_client.FasterWhisperASR
    asr = FasterWhisperASR("m", "float16", "cuda", boundary_pad_ms=boundary_pad_ms)
    return asr, captured


def test_transcribe_pads_clip_with_silence_on_both_sides(monkeypatch):
    """boundary_pad_ms=500 at 16 kHz prepends AND appends 8000 silent
    samples, so the model receives len(clip) + 2*8000 samples with the
    original clip untouched in the middle.

    One-line seeded defect: delete the padding block in transcribe -> the
    model receives len(clip) -> the length assertion fails -> red.
    """
    asr, captured = _capturing_asr(monkeypatch, boundary_pad_ms=500)
    clip = np.full(16000, 0.2, dtype=np.float32)  # 1.0 s of nonzero "speech"

    asr.transcribe(clip, 16000)

    pad = 8000  # 500 ms @ 16 kHz
    got = np.asarray(captured["audio"])
    assert len(got) == len(clip) + 2 * pad
    assert np.all(got[:pad] == 0.0), "leading pad is not silence"
    assert np.all(got[-pad:] == 0.0), "trailing pad is not silence"
    assert np.array_equal(got[pad:-pad], clip), "original clip was mutated"
    assert got.dtype == clip.dtype, "pad changed the sample dtype"


def test_pad_sample_count_scales_with_sample_rate(monkeypatch):
    """The pad is expressed in MILLISECONDS and converted using the clip's
    own sample rate, so it stays 500 ms of silence at any rate (task: the pad
    must be in the model's sample rate, applied after any resampling).

    One-line seeded defect: hardcode the pad sample count (e.g. 8000) instead
    of deriving it from sample_rate -> at 8 kHz the model gets the wrong count
    -> red.
    """
    asr, captured = _capturing_asr(monkeypatch, boundary_pad_ms=500)
    clip = np.full(4000, 0.2, dtype=np.float32)  # 0.5 s @ 8 kHz

    asr.transcribe(clip, 8000)

    pad = 4000  # 500 ms @ 8 kHz
    assert len(np.asarray(captured["audio"])) == len(clip) + 2 * pad


def test_pad_zero_passes_clip_through_byte_identical(monkeypatch):
    """boundary_pad_ms=0 disables padding: the model receives the EXACT same
    array object, unmodified -- byte-identical to the pre-padding behavior.

    One-line seeded defect: pad unconditionally (drop the `if pad > 0` guard,
    concatenating zero-length silence) -> a NEW array is passed and the
    identity assertion fails -> red.
    """
    asr, captured = _capturing_asr(monkeypatch, boundary_pad_ms=0)
    clip = np.full(16000, 0.2, dtype=np.float32)

    asr.transcribe(clip, 16000)

    assert captured["audio"] is clip, (
        "pad=0 must pass the original array through untouched (same object)"
    )


def test_duration_s_reflects_original_clip_not_padding(monkeypatch):
    """The reported ASRResult.duration_s must be the ORIGINAL clip length, not
    the padded length -- padding is an internal decode aid and must not shift
    any value a caller consumes (requirement 3: no reported timestamp shift).

    One-line seeded defect: compute duration_s from the padded `audio`
    (len == 2.0 s) instead of the original 1.0 s clip -> red.
    """
    asr, _ = _capturing_asr(monkeypatch, boundary_pad_ms=500)
    clip = np.zeros(16000, dtype=np.float32)  # exactly 1.0 s

    out = asr.transcribe(clip, 16000)

    assert out.duration_s == 1.0


def test_padding_does_not_newly_reject_realspeech_near_no_speech_threshold(monkeypatch):
    """CRITICAL interaction: padding adds ~1 s of silence to the decode
    window, which pushes the WINDOW-level no_speech_prob upward (faster-whisper
    copies that stat onto every segment). A real-speech chunk whose
    no_speech_prob is thereby pushed OVER the 0.6 gate must still survive,
    because the no_speech gate is PAIRED with avg_logprob < -1.0 and real
    speech keeps a confident logprob. avg_logprob and compression_ratio are
    TEXT-derived, so silence does not move them. This pins that the padding win
    does not newly gate real lines (measured junk-gate empties 0->0).

    One-line seeded defect: change the no_speech gate from AND to OR in
    transcribe -> the padded chunk (no_speech 0.99 OR logprob -0.3) trips the
    gate and the real line is dropped -> red.
    """
    class _SilenceSensitiveModel:
        """Fakes the real coupling: the more silence in the window, the higher
        the no_speech_prob it reports (padding raises the silent fraction)."""

        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            arr = np.asarray(audio)
            silent_frac = float(np.mean(np.abs(arr) < 1e-4))
            no_speech = min(0.99, 0.4 + silent_frac)
            seg = SimpleNamespace(
                text="本物のセリフ", no_speech_prob=no_speech,
                avg_logprob=-0.3, compression_ratio=1.2,
            )
            return iter([seg]), SimpleNamespace(language="ja")

    from yamabiko.asr import asr_client

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _SilenceSensitiveModel,
    )
    FasterWhisperASR = asr_client.FasterWhisperASR

    clip = np.full(8000, 0.1, dtype=np.float32)  # 0.5 s of speech, no silence

    unpadded = FasterWhisperASR("m", "float16", "cuda", boundary_pad_ms=0)
    padded = FasterWhisperASR("m", "float16", "cuda", boundary_pad_ms=500)

    # Baseline: no padding -> silent_frac 0 -> no_speech 0.4 < 0.6 -> kept.
    assert unpadded.transcribe(clip, 16000).text == "本物のセリフ"
    # With padding -> silent_frac ~0.67 -> no_speech 0.99 > 0.6, but the
    # confident logprob (-0.3, not < -1.0) keeps the AND-paired gate open.
    assert padded.transcribe(clip, 16000).text == "本物のセリフ", (
        "padding pushed a real-speech chunk over the no_speech gate; the "
        "avg_logprob pairing failed to protect it"
    )
