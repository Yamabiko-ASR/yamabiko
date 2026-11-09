"""Unit tests for the ASR junk gates (design 2026-07-02 §4.2).

The subject is FasterWhisperASR's per-segment filtering; the whisper
model itself is faked (unit tier). The real-path seam is exercised by
scripts/vad_lab.py --asr in the measurement gate.
"""
from __future__ import annotations

import logging
import zlib
from types import SimpleNamespace

import numpy as np
import pytest


def _fake_whisper_model(segments):
    """Return a model double whose ``transcribe`` yields ``segments``."""
    class _FakeWhisperModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            return iter(segments), SimpleNamespace(language="ja")

    return _FakeWhisperModel


def _seg(text, no_speech=0.0, logprob=-0.1, compression=1.0):
    return SimpleNamespace(
        text=text, no_speech_prob=no_speech, avg_logprob=logprob,
        compression_ratio=compression,
    )


def _asr(monkeypatch, segments, **kwargs):
    from yamabiko.asr import asr_client

    fake_model = _fake_whisper_model(segments)
    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: fake_model,
    )
    return asr_client.FasterWhisperASR("m", "float16", "cuda", **kwargs)


@pytest.mark.parametrize(
    ("segments", "expected_text", "expected_outcome"),
    [
        ([], "", "dependency_no_segments"),
        ([_seg("junk", no_speech=0.9, logprob=-1.5)], "", "all_segments_confidence_rejected"),
        ([_seg("   ")], "", "assembled_text_empty"),
        ([_seg("loop" * 10, compression=3.0)], "", "compression_loop_rejected"),
        ([_seg(" spoken ")], "spoken", "text_available"),
    ],
)
def test_asr_result_classifies_every_wrapper_outcome(
    monkeypatch, segments, expected_text, expected_outcome,
):
    from yamabiko.asr.asr_client import ASROutcome

    result = _asr(monkeypatch, segments).transcribe(
        np.zeros(16000, dtype=np.float32), 16000,
    )

    assert result.text == expected_text
    assert result.outcome is ASROutcome(expected_outcome)
    assert (result.outcome is ASROutcome.TEXT_AVAILABLE) == bool(result.text.strip())


def test_junk_gate_drops_confident_no_speech_segment(monkeypatch):
    """no_speech_prob > 0.6 AND avg_logprob < -1.0 -> dropped.

    One-line seeded defect: change the AND to OR -- the
    high-no-speech-but-confident segment (kept case below) is dropped
    too and test_junk_gate_keeps_* goes red; or drop the gate entirely
    -- this test goes red.
    """
    asr = _asr(monkeypatch, [
        _seg("こんにちは"),
        _seg("ゴミ", no_speech=0.9, logprob=-1.5),
    ])
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "こんにちは"


def test_junk_gate_keeps_segment_failing_only_one_heuristic(monkeypatch):
    """Whisper's heuristics are PAIRED: either signal alone is
    unreliable (design §4.2). High no_speech with confident logprob
    stays; low logprob with low no_speech stays."""
    asr = _asr(monkeypatch, [
        _seg("あ", no_speech=0.9, logprob=-0.2),
        _seg("い", no_speech=0.1, logprob=-1.8),
    ])
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "あい"


def test_no_speech_only_gate_drops_confident_phantom(monkeypatch):
    """A confident phantom (no_speech 0.97, avg_logprob -0.3) passes the
    paired gate; with no_speech_only_threshold=0.95 it is dropped.

    Seeded defect: ignore the threshold -> the phantom is kept -> red."""
    asr = _asr(monkeypatch, [
        _seg("こんにちは"),
        _seg("おはようございます。", no_speech=0.97, logprob=-0.3),
    ], no_speech_only_threshold=0.95)
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "こんにちは"


def test_no_speech_only_gate_keeps_segment_under_threshold(monkeypatch):
    """Quiet real speech at no_speech 0.9 stays under a 0.95 threshold.

    Seeded defect: compare against the paired gate's 0.6 -> red."""
    asr = _asr(monkeypatch, [
        _seg("はい", no_speech=0.9, logprob=-0.2),
    ], no_speech_only_threshold=0.95)
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "はい"


def test_junk_gate_drops_repetition_loop_split_across_short_segments(monkeypatch):
    """The core of finding #1's fix: faster-whisper emits a repetition
    loop as MANY short segments that all share one whole-window
    compression_ratio. Each segment is individually below the 30-char
    floor, so a PER-SEGMENT length floor would let the whole loop through.
    The chunk-level gate judges the ASSEMBLED text (56 chars) and drops
    the whole thing.

    One-line seeded defect: gate per segment (`len(seg.text)` instead of
    the assembled `len(text)`) -> each 14-char repeat clears the floor
    and escapes -> the loop survives concatenated -> red.
    """
    phrase = "ご視聴ありがとうございました"  # 14 chars, below the 30-char floor
    assert len(phrase) < 30
    # 4 segments, all carrying the SAME window ratio (real fw behavior).
    segs = [_seg(phrase, compression=3.9) for _ in range(4)]
    asr = _asr(monkeypatch, segs)
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "", (
        f"a loop split into short segments must be dropped whole; got "
        f"{out.text!r}"
    )


def test_junk_gate_keeps_short_emphatic_repetition_split_across_segments(monkeypatch):
    """The other side of the chunk-level gate: legit emphatic repetition
    that whisper split into short segments (all sharing a high window
    ratio) survives, because the ASSEMBLED text (14 chars) is below the
    floor -- where the old per-segment gate would have dropped every
    2.4+ segment and lost the line.

    One-line seeded defect: remove the length-floor from the chunk gate
    -> the assembled 14-char line trips 2.4 and is dropped -> red.
    """
    segs = [_seg("行け", compression=2.47) for _ in range(7)]  # assembled 14
    asr = _asr(monkeypatch, segs)
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == "行け" * 7, "legit emphatic repetition was dropped"


def test_junk_gate_keeps_short_high_ratio_single_segment_below_length_floor(monkeypatch):
    """The length-floor guard (finding #1) for the single-segment case: a
    short high-compression line (「行け」x7, ratio 2.47, 14 chars) is legit
    emphatic dialogue, not a hallucination loop, and must survive even
    though its ratio is above 2.4.

    One-line seeded defect: remove the `len(text) >=
    _MIN_COMPRESSION_GATE_LEN` guard from the chunk gate -> the short
    emphatic line is dropped -> red.
    """
    from yamabiko.asr.asr_client import _MIN_COMPRESSION_GATE_LEN

    line = "行け" * 7  # 14 chars, real compression_ratio ~2.47 (> 2.4 gate)
    assert len(line) < _MIN_COMPRESSION_GATE_LEN
    asr = _asr(monkeypatch, [_seg(line, compression=2.47), _seg("続き")])
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == line + "続き", "short emphatic repetition was dropped"


def test_junk_gate_keeps_short_single_mora_stutter_below_floor(monkeypatch):
    """Pins the DOCUMENTED residual of the length-floor design: a short
    single-mora stutter loop (「わ」x20, ratio ~4.29, 20 chars) is below the
    30-char floor and is KEPT -- at that length it is indistinguishable
    from a real short scream, and the safety-net gate errs toward keeping
    ambiguous short text. Long single-mora loops (>=30 chars) are still
    dropped (see test_junk_gate_drops_repetition_loop_split_across_short_segments).

    This makes the trade-off a tested contract, not just a code comment, so
    a later floor change can't move it silently. One-line seeded defect:
    lower _MIN_COMPRESSION_GATE_LEN to 10 -> the 20-char stutter now trips
    the gate and is dropped -> red.
    """
    from yamabiko.asr.asr_client import (
        _COMPRESSION_RATIO_THRESHOLD, _MIN_COMPRESSION_GATE_LEN,
    )

    stutter = "わ" * 20
    assert _whisper_compression_ratio(stutter) > _COMPRESSION_RATIO_THRESHOLD
    assert len(stutter) < _MIN_COMPRESSION_GATE_LEN
    asr = _asr(monkeypatch, [
        _seg(stutter, compression=_whisper_compression_ratio(stutter)),
    ])
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == stutter, "short single-mora stutter was dropped"


def test_junk_gate_returns_empty_when_all_segments_are_no_speech(monkeypatch):
    """A chunk whose every segment fails the no_speech/logprob gate yields
    empty text -- the chunk-level compression gate handles the
    no-survivors case (empty ratios) without raising.

    One-line seeded defect: invert the no_speech gate (`>` -> `<` on
    no_speech_prob) -> the junk segments are kept -> text is non-empty -> red.
    """
    asr = _asr(monkeypatch, [
        _seg("ゴミ", no_speech=0.9, logprob=-1.5),
        _seg("ノイズ", no_speech=0.8, logprob=-2.0),
    ])
    out = asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert out.text == ""


def test_junk_gate_drop_logs_lengths_never_text(monkeypatch, caplog):
    """Privacy: drop logs carry lengths/scores, never the segment text
    (PR #69 contract).

    One-line seeded defect: log seg.text in the drop line -- red.
    """
    asr = _asr(monkeypatch, [_seg("ひみつのことば", no_speech=0.9, logprob=-1.5)])
    with caplog.at_level(logging.INFO, logger="yamabiko.asr.asr_client"):
        asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    drop_lines = [r.getMessage() for r in caplog.records
                  if "junk gate" in r.getMessage()]
    assert drop_lines, "junk-gate drop was not logged"
    assert all("ひみつのことば" not in m for m in drop_lines), (
        "recognized text leaked into the log"
    )


def test_junk_gate_repetition_drop_logs_lengths_never_text(monkeypatch, caplog):
    """Privacy on the SECOND gate: a compression-ratio drop must also log
    lengths/scores, never the segment text. This path had no privacy
    test before the drop-logging was centralized in _log_junk_drop.

    One-line seeded defect: pass the text (not len) into _log_junk_drop,
    or log seg.text in the repetition branch -- red.
    """
    asr = _asr(monkeypatch, [_seg("ひみつのループ" * 5, compression=3.1)])
    with caplog.at_level(logging.INFO, logger="yamabiko.asr.asr_client"):
        asr.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    drop_lines = [r.getMessage() for r in caplog.records
                  if "junk gate" in r.getMessage()]
    assert drop_lines, "repetition drop was not logged"
    # Tie the path check to the logged score (the actual gate signal),
    # not the cosmetic reason label, so a benign reason-string rename
    # doesn't flip this test.
    assert all("compression_ratio" in m for m in drop_lines)
    assert all("ひみつのループ" not in m for m in drop_lines), (
        "recognized text leaked into the repetition-drop log"
    )


def _whisper_compression_ratio(text: str) -> float:
    """Byte-for-byte replica of faster_whisper.transcribe.get_compression_ratio
    (verified against the installed faster-whisper source): the ratio of
    UTF-8 byte length to zlib-compressed length. The gate compares
    ``seg.compression_ratio`` -- produced by that exact function -- to
    ``_COMPRESSION_RATIO_THRESHOLD``, so this calibrates the threshold
    against the number the gate actually sees for a given string.
    """
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


@pytest.mark.real
def test_compression_ratio_replica_matches_faster_whisper():
    """The calibration suite below uses _whisper_compression_ratio, a
    hand-copied replica of faster_whisper.get_compression_ratio. If a
    library release changes that algorithm the replica silently desyncs
    and every calibration number becomes meaningless. Pin them together
    against the REAL function (real tier: needs faster_whisper installed).

    Seeded-defect: change the replica's zlib call (e.g. pass level=9) ->
    outputs diverge from the library -> red.
    """
    from yamabiko.asr.faster_whisper_audio_compat import install_audio_compat

    install_audio_compat()
    from faster_whisper.transcribe import get_compression_ratio

    for s in ["行け" * 7, "ご視聴ありがとうございました" * 4, "本物のセリフ",
              "あ" * 50, "だめだめだめ"]:
        assert _whisper_compression_ratio(s) == get_compression_ratio(s), repr(s)


class TestCompressionGateCalibration:
    """Finding #1 (review 2026-07-03): the compression-ratio gate is
    STANDALONE -- no ``avg_logprob`` pairing -- because repetition-loop
    hallucinations are confident, so a logprob guard would miss them.
    That makes the 2.4 threshold the ONLY thing separating a
    hallucination loop from legitimate emphatic repetition. These tests
    pin where that boundary falls relative to real Japanese dialogue so a
    threshold change (or a fix) can't move the false-positive line
    unnoticed.
    """

    def test_moderate_emphatic_repetition_clears_the_gate(self):
        """Everyday emphatic / onomatopoeic dialogue -- a short word
        repeated up to ~6x, e.g. 「だめだめだめ」 or 「行け」x6 -- stays
        UNDER 2.4 and is kept.

        Seeded-defect: drop _COMPRESSION_RATIO_THRESHOLD to 1.5 -> these
        cross it -> red.
        """
        from yamabiko.asr.asr_client import _COMPRESSION_RATIO_THRESHOLD

        for line in ["だめだめだめ", "やだやだやだやだ", "行け" * 6,
                     "ドキドキドキドキ", "ちょっと待って、今どこにいるの？"]:
            ratio = _whisper_compression_ratio(line)
            assert ratio < _COMPRESSION_RATIO_THRESHOLD, (
                f"legit line {line!r} has compression_ratio {ratio:.2f} "
                f">= gate {_COMPRESSION_RATIO_THRESHOLD}"
            )

    def test_sustained_hallucination_loop_exceeds_the_gate(self):
        """The true positive: the classic JP whisper hallucination
        (「ご視聴ありがとうございました」looped) and long single-token loops
        must exceed 2.4 so they ARE dropped.

        Seeded-defect: cut the phrase loop to 3 repeats -> 2.38 < 2.4 ->
        red (the gate would stop catching it).
        """
        from yamabiko.asr.asr_client import _COMPRESSION_RATIO_THRESHOLD

        assert (_whisper_compression_ratio("ご視聴ありがとうございました" * 8)
                > _COMPRESSION_RATIO_THRESHOLD)
        assert _whisper_compression_ratio("わ" * 40) > _COMPRESSION_RATIO_THRESHOLD

    def test_heavy_emphatic_repetition_crosses_2p4_but_is_short(self):
        """Pins the false-positive premise finding #1 rests on, OUTSIDE
        the behavioral test so it can never be silently swallowed: legit
        intense-scene repetition (「行け」x7) has compression_ratio ABOVE
        2.4 yet stays BELOW the length floor. If either half stops
        holding, the length-floor guard is no longer protecting a real
        case and this goes red.

        Seeded-defect: raise _COMPRESSION_RATIO_THRESHOLD to 3.0 -> the
        ratio no longer exceeds it -> red.
        """
        from yamabiko.asr.asr_client import (
            _COMPRESSION_RATIO_THRESHOLD, _MIN_COMPRESSION_GATE_LEN,
        )

        line = "行け" * 7
        assert _whisper_compression_ratio(line) > _COMPRESSION_RATIO_THRESHOLD
        assert len(line) < _MIN_COMPRESSION_GATE_LEN
