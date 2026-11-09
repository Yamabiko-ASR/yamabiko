"""ASR wrapper for faster-whisper (default model Kodama-1, Yamabiko's Japanese
fine-tune of whisper-ja-760M; repo pinned in yamabiko/models.py)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from yamabiko.asr import faster_whisper_audio_compat

_log = logging.getLogger(__name__)

# Whisper's own paired heuristics (its published defaults); either
# signal alone is unreliable -- hallucinations often carry HIGH
# confidence and LOW no_speech_prob, so these gates only catch the
# clear cases; the VAD deep-silence floor handles the rest upstream.
# Constants, not config, until the vad_lab measurement gate says
# otherwise (design 2026-07-02 §9).
_NO_SPEECH_THRESHOLD = 0.6
_LOGPROB_THRESHOLD = -1.0
_COMPRESSION_RATIO_THRESHOLD = 2.4
# Minimum assembled-text length before the compression gate fires. The gate
# is standalone (a confident repetition loop can't be caught by the
# no_speech/logprob pair), so the 2.4 ratio is the ONLY thing separating a
# hallucination loop from legitimate emphatic repetition -- and the two
# overlap in the 2.4-3.0 band on SHORT text. Real emphatic dialogue crosses
# 2.4 at 7 repeats of a 2-char token (「行け」x7 = 2.47, 14 chars) or 5 of a
# 4-char token (「こないで」x5 = 2.50, 20 chars); the hallucination loops that
# matter run long (the ご視聴ありがとうございました loop needs 3+ repeats = 42+
# chars to even reach 2.4). So only drop above a length floor -- short
# high-ratio text is intense-scene dialogue, not a loop. Applied to the
# ASSEMBLED chunk text, not per segment: see the gate below. Boundary
# measured in tests/asr/test_asr_junk_gates.py; revisit via the vad_lab
# measurement gate, not by feel (design 2026-07-02 §9).
_MIN_COMPRESSION_GATE_LEN = 30


def _log_junk_drop(reason: str, text_len: int, **scores: float) -> None:
    """Log a junk-gate drop with the text LENGTH and decision scores
    only -- never the recognized text itself (PR #69 privacy contract).

    Every gate routes its drop through here so the never-log-text
    invariant lives in one place and a new gate can't reintroduce a
    text leak by copy-pasting a log line.
    """
    detail = ", ".join(f"{name}={value:.2f}" for name, value in scores.items())
    _log.info("asr junk gate: dropped %s (len=%d, %s)", reason, text_len, detail)


class ASROutcome(str, Enum):
    DEPENDENCY_NO_SEGMENTS = "dependency_no_segments"
    ALL_SEGMENTS_CONFIDENCE_REJECTED = "all_segments_confidence_rejected"
    ASSEMBLED_TEXT_EMPTY = "assembled_text_empty"
    COMPRESSION_LOOP_REJECTED = "compression_loop_rejected"
    TEXT_AVAILABLE = "text_available"


@dataclass
class ASRResult:
    text: str
    duration_s: float
    outcome: ASROutcome


class ASRBackend(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult: ...
    def warmup(self) -> None: ...


class FasterWhisperASR:
    """faster-whisper CT2 backend (default model: Kodama-1, Yamabiko's Japanese
    fine-tune of whisper-ja-760M; repo pinned in models.py).

    NOTE: CTranslate2 INT8 kernels crash on Blackwell (sm_120) with
    CUBLAS_STATUS_NOT_SUPPORTED. Pass compute_type='float16' until
    CTranslate2 ships sm_120 INT8 support.
    """

    def __init__(self, model: str, compute_type: str, device: str,
                 local_files_only: bool = False, beam_size: int = 5,
                 initial_prompt: str | None = None,
                 revision: str | None = None, boundary_pad_ms: int = 500,
                 no_speech_only_threshold: float | None = None):
        WhisperModel = faster_whisper_audio_compat.import_whisper_model()

        # Silence padding (ms) prepended AND appended to every clip before it
        # reaches the model. Whisper models truncate the first/last syllables
        # of SHORT clips -- a documented boundary artifact whose community fix
        # is a little leading + trailing silence. Measured 2026-07-05 on
        # kotoba-whisper (the default model then) on real episodes vs JA CC
        # ground truth (3 titles, 1530 chunks): chunk-CER 47.70->46.20 /
        # 30.56->27.34 / 32.03->30.01, latency and junk-gate-empty counts
        # unchanged; kept at the same value for the current model, which was
        # benchmarked with it. 0 disables (the clip is passed through
        # byte-identical). Mirrors asr.boundary_pad_ms in default_config.yaml.
        # Padding raises the window no_speech_prob. The paired gate ignores
        # that (it also needs a low avg_logprob); the no-speech-only threshold
        # does not, so re-check it before changing the pad.
        self._boundary_pad_ms = int(boundary_pad_ms)

        # beam_size: faster-whisper's default is 5. Greedy decoding (beam_size=1)
        # is the main driver of high-frequency-word substitution on short clips
        # (悪い -> 終わり): greedy commits to the single top token so the
        # acoustically-correct rare word never gets reconsidered, while a beam
        # keeps several hypotheses alive. 5 restores that diversity at a modest
        # latency cost on the short clips this pipeline feeds. Tunable via
        # asr.faster_whisper.beam_size for A/B without a rebuild.
        self._beam_size = beam_size
        # Empty string / None: no prompt (Kodama-1 is decoded without one).
        self._initial_prompt = initial_prompt or None
        # no_speech_only_threshold drops a segment on no_speech_prob alone
        # above this value, before the paired gate below. The shipped model,
        # Kodama-1, runs at 0.95: it was trained on no-speech windows and
        # reaches no_speech_prob ~0.99 on non-speech, but can still write
        # confident text there, which the paired gate keeps. 1.0 turns it off
        # (no probability is above 1.0), and so does None here; a config
        # cannot say null, because config.py keeps the numeric shipped
        # default over a null override. Mirrors
        # asr.faster_whisper.no_speech_only_threshold in default_config.yaml.
        if no_speech_only_threshold is not None and not 0.0 < no_speech_only_threshold <= 1.0:
            raise ValueError("no_speech_only_threshold must be in (0, 1]")
        self._no_speech_only_threshold = no_speech_only_threshold
        # local_files_only=True skips faster-whisper's per-startup online
        # revision re-check when the weights are already cached, so a
        # warmed install launches offline. revision pins the exact snapshot
        # the wizard warmed (models.WHISPER_REVISION); WhisperModel resolves
        # the repo against the HF cache at that revision, so a fresh install
        # warmed only at snapshots/<sha>/ (no refs/main) still loads offline.
        self._model = WhisperModel(
            model, device=device, compute_type=compute_type,
            local_files_only=local_files_only, revision=revision,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        # duration_s is the ORIGINAL clip length: the boundary pad is an
        # internal decode aid and must not shift any value a caller reads
        # (the pipeline consumes .text; .duration_s is the only reported
        # length). Compute it before padding.
        original_samples = len(audio)
        # Pad AFTER any resampling so the silence is in the model's sample
        # rate. faster-whisper receives audio already at sample_rate (the
        # capture layer resamples), so the pad count derives from it directly.
        pad = int(round(sample_rate * self._boundary_pad_ms / 1000))
        if pad > 0:
            silence = np.zeros(pad, dtype=audio.dtype)
            audio = np.concatenate([silence, audio, silence])
        segments, _ = self._model.transcribe(
            audio,
            language="ja",
            beam_size=self._beam_size,
            compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=_LOGPROB_THRESHOLD,
            no_speech_threshold=_NO_SPEECH_THRESHOLD,
            vad_filter=False,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=False,
        )
        # faster-whisper already skips every decode window whose
        # no_speech_prob > 0.6 and avg_logprob <= -1.0 (the thresholds passed
        # above), so the paired per-segment check below is a backstop for
        # dependency drift. The no-speech-only check before it is not:
        # faster-whisper has no such rule, and with the shipped model it is
        # the check that drops confident text written over non-speech.
        # compression_ratio is a WHOLE-DECODE-WINDOW statistic that
        # faster-whisper copies onto every timestamp-split segment of a
        # window (transcribe.py: compression_ratio=compression_ratio for all
        # of them), so a repetition loop arrives as many SHORT segments each
        # carrying the same high ratio. Gating it per segment would let a
        # long loop escape a per-segment length floor; collect the survivors
        # and apply the compression gate ONCE to the assembled text.
        kept: list[str] = []
        ratios: list[float] = []
        raw_segment_count = 0
        for seg in segments:
            raw_segment_count += 1
            if (self._no_speech_only_threshold is not None
                    and seg.no_speech_prob > self._no_speech_only_threshold):
                _log_junk_drop(
                    "no-speech segment (no-speech-only gate)", len(seg.text),
                    no_speech_prob=seg.no_speech_prob,
                    avg_logprob=seg.avg_logprob,
                )
                continue
            if (seg.no_speech_prob > _NO_SPEECH_THRESHOLD
                    and seg.avg_logprob < _LOGPROB_THRESHOLD):
                _log_junk_drop(
                    "no-speech segment", len(seg.text),
                    no_speech_prob=seg.no_speech_prob,
                    avg_logprob=seg.avg_logprob,
                )
                continue
            kept.append(seg.text)
            ratios.append(seg.compression_ratio)
        text = "".join(kept).strip()
        # Chunk-level compression gate. VAD chunks are always well under one
        # 30 s whisper decode window (max_chunk_ms defaults to 8000), so a
        # transcribe call is a single window with one shared ratio; max()
        # over the survivors' ratios is that value. A LONG, highly
        # compressible result is a hallucination loop -- drop it whole; a
        # SHORT one (even above the ratio) is legit emphatic repetition
        # (「行け」x7 = 2.47 at 14 chars).
        # Known residual: a short single-mora stutter loop (「わ」x20 ~= 4.3 at
        # 20 chars) is kept -- it is indistinguishable from a real short
        # scream at this length, and the gate is a safety net, so the floor
        # errs toward keeping ambiguous short text. Long single-mora loops
        # (>= 30 chars) are still caught.
        compression_rejected = bool(
            len(text) >= _MIN_COMPRESSION_GATE_LEN and ratios
            and max(ratios) > _COMPRESSION_RATIO_THRESHOLD
        )
        if compression_rejected:
            _log_junk_drop(
                "repetition loop", len(text), compression_ratio=max(ratios),
            )
            text = ""
        if raw_segment_count == 0:
            outcome = ASROutcome.DEPENDENCY_NO_SEGMENTS
        elif not kept:
            outcome = ASROutcome.ALL_SEGMENTS_CONFIDENCE_REJECTED
        elif not text and not compression_rejected:
            outcome = ASROutcome.ASSEMBLED_TEXT_EMPTY
        elif compression_rejected:
            outcome = ASROutcome.COMPRESSION_LOOP_REJECTED
        else:
            outcome = ASROutcome.TEXT_AVAILABLE
        return ASRResult(
            text=text,
            duration_s=original_samples / sample_rate,
            outcome=outcome,
        )

    def warmup(self) -> None:
        silent = np.zeros(16000, dtype=np.float32)
        self.transcribe(silent, 16000)


# The probe requires ALL of models.WHISPER_FILES before taking the offline
# path (keying on model.bin alone makes a partial cache -- model.bin landed
# but a metadata file did not, e.g. a mid-download abort -- fatal:
# local_files_only=True would then hard-crash at startup with no online
# fallback). The probe body lives in models.whisper_files_cached, shared with
# wizard_state's required_models_cached() gate, so the two can't drift.
# The revision pin is CONDITIONAL (models.whisper_revision_for): pinned for
# the default repo only -- a user override of asr.faster_whisper.model to any
# other repo (anime-whisper etc.) or a local directory must load unpinned,
# because the default-repo-only sha guarantees RevisionNotFoundError
# everywhere else.
from yamabiko.models import whisper_files_cached as _whisper_files_cached
from yamabiko.models import whisper_revision_for as _whisper_revision_for


def _whisper_cached(model: str) -> bool:
    """True only if EVERY whisper file is already in the local HF cache
    (pinned revision for the default repo, unpinned for an override), so
    an offline (local_files_only=True) load can't fail on a partial cache.
    Any probe failure falls back to the safe online path."""
    try:
        return _whisper_files_cached(model)
    except Exception:
        return False


def build_asr(cfg: dict) -> ASRBackend:
    w = cfg["faster_whisper"]
    return FasterWhisperASR(
        w["model"], w["compute_type"], w["device"],
        local_files_only=_whisper_cached(w["model"]),
        beam_size=int(w.get("beam_size", 5)),
        initial_prompt=w.get("initial_prompt", ""),
        revision=_whisper_revision_for(w["model"]),
        # boundary_pad_ms lives at the asr level (sibling of faster_whisper),
        # so it's read off cfg, not w. Default 500 keeps the measured CER win
        # on even if an override omits the key; 0 disables. Consumer literal
        # for the preflight config-key check lives right here.
        boundary_pad_ms=int(cfg.get("boundary_pad_ms", 500)),
        # Consumer literal for the preflight config-key check.
        no_speech_only_threshold=w.get("no_speech_only_threshold"),
    )
