"""Pinned community Silero VAD v5-era ONNX chunker.

`FrameGate` is the pure state machine (tested in isolation).
`SileroChunker` owns the Silero model and ties it to an asyncio pipeline.
In production it also runs the sound-tag assist (``vad/sound_tagger.py``),
which raises each frame's speech probability with the bundled sound tagger.

The model is run via ``onnxruntime`` directly against the raw Silero VAD
ONNX weights the first-run wizard downloads into the pinned HF cache
(``onnx-community/silero-vad`` -> ``onnx/model.onnx``); the path is
resolved at load time via ``hf_hub_download`` (see ``_default_model_path``).
We used to depend
on the ``silero-vad`` PyPI package, but that package declares ``torch``
as a hard runtime dep — ~3 GB of CUDA DLLs land in the PyInstaller
bundle for no reason (the model itself is ~2 MB of ONNX). Bypassing the
Python wrapper lets us drop torch entirely from the installer.

Wrapper and raw-graph I/O contract (16 kHz):
  external wrapper input:
    audio  float32 [512]        one new mono frame
  inputs:
    input  float32 [1, 576]     64 retained + 512 new samples
    state  float32 [2, 1, 128]  hidden state, caller-owned across calls
    sr     int64   scalar       16000
  outputs:
    output float32 [1, 1]       speech probability
    stateN float32 [2, 1, 128]  next hidden state
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from yamabiko.asr.incremental import OpenWindow
from yamabiko.models import (
    ModelArtifactCorrupt,
    SILERO_FILE,
    SILERO_REPO,
    SILERO_REVISION,
    SILERO_SHA256,
)

from .chunker import ChunkerConfig
from .sound_tagger import SoundTagAssist, SoundTagger
from .sound_tagger import default_model_path as default_tagger_path

_log = logging.getLogger(__name__)

# Seconds without a capture block, while the sound-tag assist holds frames,
# before those frames are tagged without their look-ahead and gated. Blocks
# normally arrive every few tens of milliseconds; WASAPI loopback sends none
# while the endpoint is silent.
STALL_DRAIN_S = 1.0


class FrameGateEmitReason(str, Enum):
    """Exhaustive reasons why :class:`FrameGate.push` returned audio."""

    SILENCE_CLOSE = "endpoint_silence_close"
    GLUE_TIMEOUT = "endpoint_glue_timeout"
    FORCE_SPLIT = "endpoint_force_split"
    DIP_CUT = "endpoint_dip_cut"


@dataclass(frozen=True, slots=True)
class FrameGateState:
    """Content-free snapshot of audio still owned by the endpoint gate.

    ``retained_frames`` counts the unique, not-yet-emitted payload: the live
    right piece while speech is open, otherwise the pending pre-roll buffer.
    The counters describe the current utterance and let evidence consumers
    verify a close reason against the recorded gate configuration without
    reading private PCM.
    """

    in_speech: bool
    retained_frames: int
    speech_frames: int
    consecutive_silence_frames: int
    silence_since_speech_frames: int
    frames_since_open: int


@dataclass(frozen=True, slots=True)
class FrameGateEmission:
    """Atomic reason and retained-state transition for one emitted payload."""

    reason: FrameGateEmitReason
    before: FrameGateState
    after: FrameGateState


@dataclass
class FrameGate:
    """Stateful speech-boundary detector.

    Fed frame-by-frame with (audio_frame, speech_probability). Returns a
    float32 audio array when a chunk should be emitted, else None.

    Emission policy:
      - Open a chunk on the first frame with prob >= threshold.
      - While open, SUSTAIN speech down to ``max(threshold - 0.15, 0.01)``
        (upstream silero-vad's neg_threshold hysteresis). Background
        music dips speech probability below the open threshold
        mid-sentence; without the lower sustain bar those dips read as
        silence and shred real-media dialogue into sub-second fragments
        (2026-07-02 episode measurement: 81.3% -> 90.8% speech coverage
        with hysteresis + the 0.30 default threshold).
      - Close on `min_silence_ms` of consecutive frames below the
        sustain threshold.
      - If accumulated speech < `min_chunk_ms`, keep the chunk open across
        the silence instead of emitting (JP filler over-segmentation fix).
      - If speech duration >= `max_chunk_ms`, force-emit and start fresh.
      - Prepend `speech_pad_ms` of pre-speech audio (from a ring buffer).
      - Append up to `min_silence_ms` of post-speech silence (the silence
        frames accumulated before end-of-speech is triggered are included
        in the emitted chunk).

    Dip-split (optional, `cfg.dip_split_enabled`): a rapid A/B exchange has
    no inter-turn silence so the gate never closes, but the per-frame Silero
    probability shows a valley at the turn boundary. While open the gate
    tracks a maximal run of frames below `dip_split_threshold`; when the run
    recovers it cuts the CURRENT piece at the run's argmin frame, emitting
    the left piece WITHOUT closing the gate or resetting Silero state. The
    cut fires only once the left piece spans `dip_split_min_piece_ms`; the
    right side can't be measured at the valley, so it isn't gated -- the
    right piece keeps accumulating and emits on the eventual close (a short
    final piece is a normal chunk, never suppressed). Disabled => the piece
    boundary never advances and every emit is the whole buffer, byte-for-
    byte identical to the pre-feature gate.
    """

    cfg: ChunkerConfig
    threshold: float = 0.5
    frame_samples: int = 512

    def __post_init__(self) -> None:
        sr = self.cfg.sample_rate
        self._frame_ms = 1000 * self.frame_samples / sr
        # Upstream formula: neg_threshold = max(threshold - 0.15, 0.01).
        self._neg_threshold = max(self.threshold - 0.15, 0.01)
        # Timing keys arrive unvalidated from user config. Below one
        # frame (~32 ms at 16 kHz), int() floors the counts to 0 and the
        # gate degenerates: min_silence_frames=0 makes the silence check
        # (0 >= 0) true on EVERY in-speech frame, emitting fixed
        # ~min_chunk-sized chunks mid-word with no silence ever observed;
        # max_chunk_frames=0 force-emits on the second frame of every
        # utterance. Clamp to sane floors and warn instead of silently
        # mangling every chunk boundary.
        min_silence = int(self.cfg.min_silence_ms / self._frame_ms)
        if min_silence < 1:
            _log.warning(
                "vad.min_silence_ms=%r is below one %.0f ms frame; "
                "clamping to one frame",
                self.cfg.min_silence_ms, self._frame_ms,
            )
            min_silence = 1
        self._min_silence_frames = min_silence
        self._min_chunk_frames = int(self.cfg.min_chunk_ms / self._frame_ms)
        max_chunk = int(self.cfg.max_chunk_ms / self._frame_ms)
        # Floor at min_chunk_frames (not min_chunk+1: max == min is a legal
        # boundary config the test suite pins), and at one frame minimum.
        max_chunk_floor = max(1, self._min_chunk_frames)
        if max_chunk < max_chunk_floor:
            _log.warning(
                "vad.max_chunk_ms=%r (%d frames) does not exceed "
                "min_chunk_ms=%r (%d frames); clamping to %d frames",
                self.cfg.max_chunk_ms, max_chunk,
                self.cfg.min_chunk_ms, self._min_chunk_frames,
                max_chunk_floor,
            )
            max_chunk = max_chunk_floor
        self._max_chunk_frames = max_chunk
        # Cap how long a sub-min-chunk utterance glues across trailing
        # silence. Same sub-frame floor + warn as the sibling timing keys;
        # below min_silence is legal (it just emits at the first glue point).
        glue_max_silence = int(self.cfg.glue_max_silence_ms / self._frame_ms)
        if glue_max_silence < 1:
            _log.warning(
                "vad.glue_max_silence_ms=%r is below one %.0f ms frame; "
                "clamping to one frame",
                self.cfg.glue_max_silence_ms, self._frame_ms,
            )
            glue_max_silence = 1
        self._glue_max_silence_frames = glue_max_silence
        self._pad_frames = int(self.cfg.speech_pad_ms / self._frame_ms)
        # The legacy speech_pad_ms ring size, floored at one frame. Both
        # the attach buffer's maxlen and the deep-silence trim collapse to
        # this floor, so compute it once.
        self._pad_floor = max(self._pad_frames, 1)
        # Attach-don't-extend: _pending holds EVERYTHING since the last
        # emit (bounded), not a speech_pad_ms ring. See the design doc
        # DESIGN-VAD-SEGMENTER-FILTER-SPLIT-2026-07-02. The pad floor
        # keeps degenerate configs (attach < pad) behaving like the
        # legacy ring.
        attach_frames = int(self.cfg.attach_max_ms / self._frame_ms)
        self._attach_max_frames = max(attach_frames, self._pad_floor)
        self._pending: deque[np.ndarray] = deque(maxlen=self._attach_max_frames)
        drop_frames = int(self.cfg.drop_floor_ms / self._frame_ms)
        self._drop_floor_frames = max(drop_frames, 1)
        # Dip-splitter config (see the class docstring's dip section).
        self._dip_enabled = self.cfg.dip_split_enabled
        self._dip_threshold = self.cfg.dip_split_threshold
        # Min frames per split piece. Floor at one frame so a sub-frame
        # config can't degenerate the cut rule (every valley recovery
        # would satisfy 0 >= 0 and cut). Same frame-floor rationale as the
        # timing keys above; a dip config this small is a deliberate stress
        # config, so no warn.
        self._dip_min_piece_frames = max(
            1, int(self.cfg.dip_split_min_piece_ms / self._frame_ms))
        self._deep_silence_frame_count = 0
        self._in_speech = False
        self._buf: list[np.ndarray] = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._total_frame_count = 0
        # Silence frames seen since speech last resumed. Unlike
        # _silence_frame_count (reset to 0 every glue), this only resets
        # when speech actually resumes — so a sub-min utterance trailed by
        # pure silence can't glue forever. _last_speech_frame_index is the
        # index within _buf of the most recent prob>=threshold frame, used
        # to trim trailing silence at the glue-cap emit.
        self._silence_since_speech_frames = 0
        self._last_speech_frame_index = 0
        # Why the last chunk was emitted: True = max-chunk force-split
        # (mid-speech cut; the next chunk continues the same utterance),
        # False = silence-terminated. Consumed by SileroChunker.run to
        # stamp Segment.force_split.
        self.last_emit_forced = False
        # True when the last emit was a dip-split cut (mid-utterance valley
        # split, gate still open). SileroChunker.run reads this to SKIP the
        # per-emit Silero reset on a cut -- the utterance's prob stream must
        # stay continuous for parity with the offline dip spike (PR#71's
        # reset stays at true closes only).
        self.last_emit_was_dip_cut = False
        # Dip piece-tracking state (prob-frame indexing == offline spike).
        self._reset_dip_state()
        # Atomic typed observation for consumers that need more than the two
        # legacy flags above. It is replaced only when push() emits audio and
        # cleared when reset() discards state without emitting.
        self.last_emission: FrameGateEmission | None = None

    def _reset_dip_state(self) -> None:
        """Clear per-utterance dip tracking. Called at construction, at each
        speech open, and at each emit (cut or close)."""
        # prob-frame index where the current not-yet-emitted piece began
        # (0 = utterance open; = the previous cut frame after a cut).
        self._piece_start_k = 0
        # prob-frame index of the frame currently being scored (0 = open).
        self._dip_i = 0
        # _buf offset of prob-index 0: the pre-roll pad/attach frames that
        # sit in _buf before the opening frame. Piece slices map prob-index
        # k to _buf index _dip_pad_len + k.
        self._dip_pad_len = 0
        # In-progress sub-threshold run: its running minimum and the
        # prob-index of that minimum (the pending cut point).
        self._in_dip = False
        self._dip_run_min = 1.0
        self._dip_run_argmin_k = 0

    def push(self, audio_frame: np.ndarray, prob: float) -> np.ndarray | None:
        if not self._in_speech:
            if prob >= self.threshold:
                self._in_speech = True
                # No _deep_silence_frame_count reset here: it is only read
                # while NOT in speech, and _close_and_emit already zeros it
                # before the gate returns to that state.
                self._buf = list(self._pending)
                self._buf.append(audio_frame)
                self._last_speech_frame_index = len(self._buf) - 1
                self._speech_frame_count = 1
                self._silence_frame_count = 0
                self._silence_since_speech_frames = 0
                # New utterance: the opening frame is prob-index 0, and
                # everything already in _buf (the pad/attach pre-roll) sits
                # before it. Dip tracking indexes from this frame.
                self._reset_dip_state()
                self._dip_pad_len = len(self._buf) - 1
                return None
            self._pending.append(audio_frame)
            if prob < self.cfg.drop_floor_prob:
                self._deep_silence_frame_count += 1
                if self._deep_silence_frame_count >= self._drop_floor_frames:
                    # Sustained ring-mode (I5): while deep silence holds,
                    # pending is exactly the legacy speech_pad_ms ring --
                    # a one-shot clear would let post-clear silence
                    # re-grow the buffer and break byte-parity.
                    while len(self._pending) > self._pad_floor:
                        self._pending.popleft()
            else:
                self._deep_silence_frame_count = 0
            return None

        # In speech. Sustain bar is _neg_threshold, NOT threshold:
        # see the hysteresis note in the class docstring.
        self._buf.append(audio_frame)
        self._total_frame_count += 1
        if prob >= self._neg_threshold:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
            self._silence_since_speech_frames = 0
            self._last_speech_frame_index = len(self._buf) - 1
        else:
            self._silence_frame_count += 1
            self._silence_since_speech_frames += 1

        # Dip-split cut (see the class docstring). Evaluated on the raw
        # prob, independent of the neg_threshold sustain bar, before the
        # close checks: a cut emits the left piece while the gate stays
        # open. No Silero reset and no close here -- the utterance
        # continues. Disabled => this whole branch is skipped.
        if self._dip_enabled:
            cut = self._track_dip(prob)
            if cut is not None:
                return cut

        # Force-emit at max chunk.
        # _total_frame_count excludes the opening frame, so the emitted
        # chunk can run one frame (~32 ms) past max_chunk_ms. Deliberate
        # slack, within tolerance.
        if self._total_frame_count >= self._max_chunk_frames:
            return self._close_and_emit(FrameGateEmitReason.FORCE_SPLIT)

        # Silence-terminated end
        if self._silence_frame_count >= self._min_silence_frames:
            if self._speech_frame_count >= self._min_chunk_frames:
                return self._close_and_emit(FrameGateEmitReason.SILENCE_CLOSE)
            # Brief filler gap; keep open, reset the silence counter to glue.
            self._silence_frame_count = 0

        # Sub-minimum chunk. Normally keep open and glue across a brief
        # filler gap. Evaluate the glue cap on every silent frame rather than
        # only when min_silence elapses; otherwise a cap that is not a
        # multiple of it (the default 46 frames) waits for the next multiple.
        if (self._speech_frame_count < self._min_chunk_frames
                and self._silence_since_speech_frames
                >= self._glue_max_silence_frames):
            # This is an isolated short utterance, not a filler gap — emit now
            # rather than gluing to the 8 s max-chunk cap (which would hand
            # Whisper a mostly-silence chunk). Trim trailing silence to at
            # most min_silence first.
            keep = self._last_speech_frame_index + 1 + self._min_silence_frames
            # Clamp the trim to the current (live) piece's start: after a
            # dip cut _last_speech_frame_index can still point into the
            # already-emitted LEFT piece (the right piece never rose back
            # to neg_threshold), so an unclamped keep would slice _buf
            # below buf_start and _close_and_emit would emit an EMPTY
            # chunk. Only reachable when dip_split_threshold >=
            # neg_threshold (threshold > 0.30) AND min_piece < min_chunk;
            # unreachable at shipped defaults (dip 0.15 < neg 0.15).
            piece_start = (0 if self._piece_start_k == 0
                           else self._dip_pad_len + self._piece_start_k)
            if piece_start < keep < len(self._buf):
                self._buf = self._buf[:keep]
            return self._close_and_emit(FrameGateEmitReason.GLUE_TIMEOUT)
        return None

    def _track_dip(self, prob: float) -> np.ndarray | None:
        """Advance the dip run for one in-speech frame; return the left
        piece if this frame closes a valley that warrants a cut, else None.

        A valley = a maximal run of frames below _dip_threshold. It is
        pending while sub-threshold frames arrive (tracking the argmin);
        it resolves the moment a frame recovers to >= _dip_threshold. The
        cut fires at the argmin only if the left piece (piece start ..
        argmin) already spans _dip_min_piece_frames -- the online left-only
        rule (PLAN point 3)."""
        self._dip_i += 1
        if prob < self._dip_threshold:
            if not self._in_dip:
                self._in_dip = True
                self._dip_run_min = prob
                self._dip_run_argmin_k = self._dip_i
            elif prob < self._dip_run_min:
                # Strict < keeps the FIRST minimum, matching np.argmin.
                self._dip_run_min = prob
                self._dip_run_argmin_k = self._dip_i
            return None
        # prob >= dip_threshold: a valley in progress recovers here.
        if not self._in_dip:
            return None
        self._in_dip = False
        cut_k = self._dip_run_argmin_k
        if cut_k - self._piece_start_k < self._dip_min_piece_frames:
            return None                       # left piece too short: no cut
        return self._emit_dip_left(cut_k)

    def _emit_dip_left(self, cut_k: int) -> np.ndarray:
        """Emit _buf's current piece up to prob-index ``cut_k``, advance the
        piece start to the cut, and leave the gate OPEN."""
        before = self.state_snapshot()
        # First piece (start still at open) keeps the original left pad at
        # _buf[0]; a later piece starts exactly at the previous cut frame,
        # no pad. Prob-index k maps to _buf index _dip_pad_len + k.
        buf_start = 0 if self._piece_start_k == 0 else self._dip_pad_len + self._piece_start_k
        buf_end = self._dip_pad_len + cut_k
        left = self._buf[buf_start:buf_end]
        self._piece_start_k = cut_k
        # A dip cut separates two different turns; downstream treats the
        # pieces as unrelated (no force-split dedup), same as a silence
        # close. Silero state is NOT reset here (the caller checks this
        # flag) -- the prob stream stays continuous, matching the spike.
        self.last_emit_forced = False
        self.last_emit_was_dip_cut = True
        self.last_emission = FrameGateEmission(
            reason=FrameGateEmitReason.DIP_CUT,
            before=before,
            after=self.state_snapshot(),
        )
        chunk = np.concatenate(left) if left else np.zeros(0, dtype=np.float32)
        return chunk.astype(np.float32, copy=False)

    def _close_and_emit(self, reason: FrameGateEmitReason) -> np.ndarray:
        if reason not in (
            FrameGateEmitReason.SILENCE_CLOSE,
            FrameGateEmitReason.GLUE_TIMEOUT,
            FrameGateEmitReason.FORCE_SPLIT,
        ):
            raise ValueError(f"invalid close reason: {reason!r}")
        before = self.state_snapshot()
        self.last_emit_forced = reason is FrameGateEmitReason.FORCE_SPLIT
        self.last_emit_was_dip_cut = False
        # Emit only the final (not-yet-cut) piece: any earlier pieces of
        # this utterance were already emitted by dip cuts. With dip disabled
        # _piece_start_k never advances, so buf_start is 0 and this is the
        # whole buffer -- byte-for-byte the pre-feature close.
        buf_start = 0 if self._piece_start_k == 0 else self._dip_pad_len + self._piece_start_k
        seg = self._buf[buf_start:]
        chunk = np.concatenate(seg) if seg else np.zeros(0, dtype=np.float32)
        self._buf = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._total_frame_count = 0
        self._silence_since_speech_frames = 0
        self._last_speech_frame_index = 0
        self._in_speech = False
        # Pending restarts after each emit: emitted audio is never
        # re-attached (I3 -- the fuzzy dedup downstream assumes
        # force-split adjacency is the only overlap case).
        self._pending.clear()
        self._deep_silence_frame_count = 0
        self._reset_dip_state()
        self.last_emission = FrameGateEmission(
            reason=reason,
            before=before,
            after=self.state_snapshot(),
        )
        return chunk.astype(np.float32, copy=False)

    def state_snapshot(self) -> FrameGateState:
        """Return a content-free view of the gate's retained endpoint state."""

        if self._in_speech:
            start = (
                0
                if self._piece_start_k == 0
                else self._dip_pad_len + self._piece_start_k
            )
            retained_frames = len(self._buf) - start
            frames_since_open = self._total_frame_count + 1
        else:
            retained_frames = len(self._pending)
            frames_since_open = 0
        return FrameGateState(
            in_speech=self._in_speech,
            retained_frames=retained_frames,
            speech_frames=self._speech_frame_count,
            consecutive_silence_frames=self._silence_frame_count,
            silence_since_speech_frames=self._silence_since_speech_frames,
            frames_since_open=frames_since_open,
        )

    def open_audio(self) -> np.ndarray | None:
        """A copy of the open window's audio, or None while no window is open.

        The copy holds every frame pushed into the window so far, its pre-roll
        included (the piece _close_and_emit would start from); it is empty,
        never None, while the open window holds no frame. Nothing in the gate
        changes.
        """
        if not self._in_speech:
            return None
        start = 0 if self._piece_start_k == 0 else self._dip_pad_len + self._piece_start_k
        piece = self._buf[start:]
        if not piece:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(piece).astype(np.float32, copy=False)

    def reset(self) -> None:
        """Abandon all buffered audio and boundary state after a capture gap."""
        self._pending.clear()
        self._deep_silence_frame_count = 0
        self._in_speech = False
        self._buf = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._total_frame_count = 0
        self._silence_since_speech_frames = 0
        self._last_speech_frame_index = 0
        self.last_emit_forced = False
        self.last_emit_was_dip_cut = False
        self._reset_dip_state()
        self.last_emission = None


class SileroVAD:
    """Strict 16 kHz wrapper for the pinned v5-era community ONNX graph.

    ``__call__`` accepts one external 512-sample frame, prepends the prior
    frame's final 64 samples, and sends a 576-sample input to the raw graph.
    Recurrent state and retained context advance together only after a fully
    valid inference. The tiny model (~2 MB) stays on CPU — GPU is reserved
    for ASR + translator — so provider selection is fixed.

    ``expected_sha256`` defaults to the shipped pin. Only the E0 evidence
    harness passes another digest, to benchmark a declared candidate graph
    with the same I/O contract.
    """

    _HIDDEN_SHAPE = (2, 1, 128)
    _FRAME_SAMPLES = 512
    _CONTEXT_SHAPE = (1, 64)

    def __init__(
        self,
        model_path: Path,
        sample_rate: int = 16000,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError(
                f"SileroVAD only supports 16000 Hz; got {sample_rate}"
            )

        verified_model_path = _verify_model_artifact(
            model_path, expected_sha256=expected_sha256,
        )

        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress info + warning chatter
        self._session = ort.InferenceSession(
            str(verified_model_path),
            providers=["CPUExecutionProvider"],
            sess_options=sess_opts,
        )
        self._state = np.zeros(self._HIDDEN_SHAPE, dtype=np.float32)
        self._context = np.zeros(self._CONTEXT_SHAPE, dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    def __call__(self, audio_frame: np.ndarray) -> float:
        frame = np.asarray(audio_frame)
        if frame.ndim != 1 or frame.shape != (self._FRAME_SAMPLES,):
            raise ValueError(
                "SileroVAD requires exactly 512 samples in one 1-D frame; "
                f"got shape {frame.shape}"
            )
        try:
            frame = frame.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("SileroVAD frame must be float32-compatible") from exc
        if not np.isfinite(frame).all():
            raise ValueError("SileroVAD frame must contain only finite samples")

        graph_input = np.concatenate(
            (self._context, frame[np.newaxis, :]), axis=1,
        )
        state_input = self._state.copy()
        inputs = {
            "input": graph_input,
            "state": state_input,
            "sr": self._sr,
        }
        outputs = self._session.run(None, inputs)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            raise ValueError("Silero VAD graph must return probability and state")

        raw_prob, raw_state = outputs
        prob = np.asarray(raw_prob)
        next_state = np.asarray(raw_state)
        if prob.shape != (1, 1) or prob.dtype != np.float32:
            raise ValueError(
                "Silero VAD probability output must be float32 with shape (1, 1)"
            )
        if next_state.shape != self._HIDDEN_SHAPE or next_state.dtype != np.float32:
            raise ValueError(
                "Silero VAD state output must be float32 with shape (2, 1, 128)"
            )

        probability = float(prob[0, 0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Silero VAD probability output must be finite in [0, 1]")
        if not np.isfinite(next_state).all():
            raise ValueError("Silero VAD state output must contain only finite values")

        # Commit both temporal state forms only after every operation above
        # succeeds. Copies also prevent the session or caller from mutating the
        # wrapper's committed transaction after return.
        committed_state = np.ascontiguousarray(next_state).copy()
        committed_context = frame[np.newaxis, -self._CONTEXT_SHAPE[1]:].copy()
        self._state = committed_state
        self._context = committed_context
        return probability

    def reset_states(self) -> None:
        self._state = np.zeros(self._HIDDEN_SHAPE, dtype=np.float32)
        self._context = np.zeros(self._CONTEXT_SHAPE, dtype=np.float32)


def _verify_model_artifact(
    model_path: Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Fail closed unless ``model_path`` is the exact expected ONNX artifact.

    ``None`` means the shipped pin, read at call time.
    """
    expected = SILERO_SHA256 if expected_sha256 is None else expected_sha256
    path = Path(model_path)
    # Report the real file: in a symlinked HF cache the snapshot entry links
    # to a blob, and deleting only the link lets hub relink the same bad blob.
    real_file = path.resolve()
    digest = hashlib.sha256()
    with real_file.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ModelArtifactCorrupt(
            "Silero VAD cached artifact is stale or corrupt: expected "
            f"SHA-256 {expected}, got {actual} for {real_file}",
            path=str(real_file),
        )
    return path


def _silero_cached() -> bool:
    """True only if the PINNED-revision silero ONNX is already in the HF
    cache, so an offline (``local_files_only=True``) load can't miss —
    the same pattern as ``asr_client._whisper_cached``. Any probe
    failure falls back to the safe online path."""
    import os
    try:
        # Force HF_HOME to be pinned before any hub call.
        from yamabiko import hf_env  # noqa: F401
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(
            repo_id=SILERO_REPO,
            filename=SILERO_FILE,
            revision=SILERO_REVISION,
        )
        return isinstance(hit, str) and os.path.isfile(hit)
    except Exception:
        return False


def _default_model_path() -> Path:
    """Resolve the Silero ONNX path via the HF cache (warmed by the first-run wizard).

    Uses ``hf_hub_download`` at the PINNED revision: the wizard
    pre-downloads exactly this snapshot into ``HF_HOME`` (pinned by
    ``yamabiko.hf_env``), so in production this is an instant offline
    cache hit (``local_files_only=True`` skips the per-startup HEAD
    revalidation). On a cold runtime (dev mode, no wizard run) it falls
    back to downloading the file now.
    """
    # Force HF_HOME + HF_XET_HIGH_PERFORMANCE to be set before import.
    from yamabiko import hf_env  # noqa: F401
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=SILERO_REPO,
        filename=SILERO_FILE,
        revision=SILERO_REVISION,
        local_files_only=_silero_cached(),
    ))


class SileroChunker:
    """Owns the Silero model, pumps frames through `FrameGate`, emits `Segment`s.

    Optional `state` parameter: epoch-stamped blocks are admitted only while
    their epoch is the current unpaused epoch. Every pause/resume edge abandons
    the accumulator, all gate audio, and Silero temporal state so waveform
    content can never be concatenated across a user pause boundary. Legacy
    bare arrays and Boolean-only state objects retain sampled-level support.

    ``model``: optional injection point for tests. Anything callable that
    takes a 512-sample float32 frame and returns a probability works; it
    must also expose ``reset_states()``. Production code leaves this
    ``None`` and the chunker constructs a real :class:`SileroVAD`.

    ``tagger``: the sound tagger of the VAD assist (``vad/sound_tagger.py``),
    a callable from one 10 s window to the assist term of its 250 tag
    frames. Production (no ``model``) loads the bundled
    :class:`SoundTagger` when none is given. With an injected ``model`` the
    assist runs only when a tagger is injected too, so model-only tests keep
    Silero-only behaviour.

    With the assist, each frame waits in :class:`SoundTagAssist` until its
    tag exists; only then does Silero score it and the gate receive
    ``max(silero, tag term)``. Silero thus sees the frames, and its resets at
    true window closes, in the same order as without the assist. Frames
    still held at a discontinuity are tagged with zeros for the missing
    look-ahead and pushed through the gate before the reset. The same
    happens, without a reset, when no capture block arrives for
    ``STALL_DRAIN_S`` (WASAPI loopback delivers nothing while the endpoint
    is silent), so a line that just ended does not wait for the next audio.
    """

    def __init__(
        self,
        cfg: ChunkerConfig,
        threshold: float = 0.5,
        state=None,
        model: Optional[object] = None,
        model_path: Optional[Path] = None,
        tagger: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._cfg = cfg
        if model is not None:
            self._model = model
        else:
            self._model = SileroVAD(
                model_path or _default_model_path(),
                sample_rate=cfg.sample_rate,
            )
            if tagger is None:
                tagger = SoundTagger(default_tagger_path())
        self._assist = None if tagger is None else SoundTagAssist(tagger)
        self._gate = FrameGate(cfg, threshold=threshold, frame_samples=512)
        self._accum = np.zeros(0, dtype=np.float32)
        self._state = state
        self._discontinuity_before_next_segment = False
        # Count of discontinuity resets per attributed reason.
        self.discontinuity_resets: Counter[str] = Counter()
        # Incremental emission (yamabiko/asr/incremental.py): samples the gate
        # has decided so far (the step clock) and the window open now.
        self._decided_samples = 0
        self._next_window_id = 0
        self._open_window_id: Optional[int] = None
        self._opened_at_sample = 0
        self._open_window_epoch: Optional[int] = None

    @property
    def decided_samples(self) -> int:
        """Samples the gate has decided since this chunker started: the clock
        of incremental emission's step decodes. It never goes back, not even
        at a discontinuity."""
        return self._decided_samples

    @property
    def open_window_id(self) -> Optional[int]:
        """The id of the window open now, or None.

        Not None exactly while the gate holds an open window: while it is not
        None, ``open_window_snapshot()`` returns that window under this id.
        """
        return self._open_window_id

    def open_window_snapshot(self) -> Optional[OpenWindow]:
        """The window open now, or None.

        Returns None only while ``open_window_id`` is None; otherwise an
        ``OpenWindow`` with that same ``window_id``, also for a window with no
        admitted audio yet (its ``audio`` is then empty). The chunker changes
        its window and its gate together, never across an await, so a caller
        that reads ``open_window_id`` and then this on the chunker's event
        loop, with no await between, always sees them agree.

        ``audio`` is a copy of the frames the gate has admitted to the window
        (decided audio): frames still waiting in the sound-tag delay line or
        in the 512-sample accumulator are not in it. Reads only; never
        mutates or drains the chunker.
        """
        if self._open_window_id is None:
            return None
        audio = self._gate.open_audio()
        if audio is None:
            raise RuntimeError("SileroChunker has an open window but its gate is closed")
        return OpenWindow(
            window_id=self._open_window_id,
            opened_at_sample=self._opened_at_sample,
            audio=audio,
            pause_epoch=self._open_window_epoch,
            discontinuity_before=self._discontinuity_before_next_segment,
        )

    def _open_window(self, pause_epoch: Optional[int]) -> None:
        self._open_window_id = self._next_window_id
        self._next_window_id += 1
        self._opened_at_sample = self._decided_samples
        self._open_window_epoch = pause_epoch

    def _reset_for_discontinuity(self, reason: str) -> None:
        """Abandon every temporal owner and arm the next emitted segment.

        Abandoned audio is discarded, never emitted. The reset is logged at
        INFO when it drops an open utterance or any buffered audio, otherwise
        at DEBUG, and counted under ``reason``.
        """

        state = self._gate.state_snapshot()
        accum_samples = int(self._accum.size)
        _log.log(
            logging.INFO
            if state.in_speech or state.retained_frames or accum_samples
            else logging.DEBUG,
            "vad discontinuity reset reason=%s abandoned_in_speech=%s "
            "retained_frames=%d speech_frames=%d accum_samples=%d",
            reason,
            state.in_speech,
            state.retained_frames,
            state.speech_frames,
            accum_samples,
        )
        self.discontinuity_resets[reason] += 1
        self._accum = np.zeros(0, dtype=np.float32)
        self._gate.reset()
        # The gate's open window is abandoned with its audio; it gets no final.
        self._open_window_id = None
        self._model.reset_states()
        if self._assist is not None:
            self._assist.reset()
        self._discontinuity_before_next_segment = True

    async def _discontinuity(self, reason: str, out_queue: asyncio.Queue) -> None:
        """Let every frame admitted so far reach the gate, then reset.

        Without the assist every admitted frame is already through the gate.
        With it, held frames are tagged with zeros for the look-ahead that
        will never come (the audio after a gap belongs to another stream),
        so a line that closed just before a pause or gap still emits.
        """
        if self._assist is not None:
            await self._gate_frames(await self._assist.drain(), out_queue)
        self._reset_for_discontinuity(reason)

    async def run(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue) -> None:
        from yamabiko.capture.envelope import (
            CaptureDiscontinuity,
            CaptureGapReason,
            CaptureSequenceTracker,
        )
        from yamabiko.state import PauseState

        sequence_tracker = CaptureSequenceTracker()
        pause_probe = getattr(self._state, "pause_snapshot", None)
        has_epoch_probe = callable(pause_probe)
        if has_epoch_probe:
            initial_pause = pause_probe()
            if not isinstance(initial_pause, PauseState):
                raise TypeError("pause_snapshot() must return PauseState")
            observed_pause_epoch = initial_pause.epoch
            legacy_paused = initial_pause.paused
        else:
            observed_pause_epoch = 0
            legacy_paused = bool(
                self._state is not None
                and getattr(self._state, "paused", False)
            )
        # Queued blocks dropped before reaching the accumulator, per reason;
        # reported once when the next block is admitted.
        skipped_blocks: Counter[str] = Counter()
        skipped_samples: Counter[str] = Counter()

        while True:
            if self._assist is not None and self._assist.holding:
                try:
                    block = await asyncio.wait_for(in_queue.get(), STALL_DRAIN_S)
                except TimeoutError:
                    # Capture went quiet with frames still waiting for their
                    # look-ahead. Tag them with zeros for it and let the gate
                    # see them now; nothing is reset.
                    await self._gate_frames(await self._assist.drain(), out_queue)
                    continue
            else:
                block = await in_queue.get()
            if isinstance(block, CaptureDiscontinuity):
                await self._discontinuity(block.reason.value, out_queue)
                if block.pause_epoch is not None:
                    observed_pause_epoch = block.pause_epoch
                if not has_epoch_probe:
                    if block.reason is CaptureGapReason.USER_PAUSE:
                        legacy_paused = True
                    elif block.reason is CaptureGapReason.USER_RESUME:
                        legacy_paused = False
                continue

            observation = sequence_tracker.observe(block)
            reset_applied = False

            if has_epoch_probe:
                pause_snapshot = pause_probe()
                if not isinstance(pause_snapshot, PauseState):
                    raise TypeError("pause_snapshot() must return PauseState")
                if pause_snapshot.epoch != observed_pause_epoch:
                    await self._discontinuity("pause_epoch_changed", out_queue)
                    reset_applied = True
                    observed_pause_epoch = pause_snapshot.epoch
                if pause_snapshot.paused:
                    skipped_blocks["paused"] += 1
                    skipped_samples["paused"] += observation.pcm.size
                    continue
                expected_pause_epoch = pause_snapshot.epoch
            else:
                paused = (
                    legacy_paused
                    if self._state is None
                    else bool(getattr(self._state, "paused", False))
                )
                if paused != legacy_paused:
                    await self._discontinuity("pause_state_changed", out_queue)
                    reset_applied = True
                    legacy_paused = paused
                    observed_pause_epoch += 1
                if paused:
                    skipped_blocks["paused"] += 1
                    skipped_samples["paused"] += observation.pcm.size
                    continue
                expected_pause_epoch = observed_pause_epoch

            # None is the additive migration value for old CaptureBlock and
            # bare-array producers. A stamped block, however, must belong to
            # the exact current unpaused epoch; both stale and future work are
            # discarded before a single frame reaches Silero.
            if (
                observation.pause_epoch is not None
                and observation.pause_epoch != expected_pause_epoch
            ):
                if not reset_applied:
                    await self._discontinuity("pause_epoch_changed", out_queue)
                skipped_blocks["stale_epoch"] += 1
                skipped_samples["stale_epoch"] += observation.pcm.size
                continue

            if (
                observation.generation_changed
                or observation.exact_gap_samples not in (None, 0)
            ):
                if not reset_applied:
                    await self._discontinuity(
                        "stream_generation_changed"
                        if observation.generation_changed
                        else "sample_gap",
                        out_queue,
                    )
            if skipped_blocks:
                for reason, blocks in skipped_blocks.items():
                    _log.info(
                        "vad skipped queued audio reason=%s blocks=%d "
                        "samples=%d",
                        reason,
                        blocks,
                        skipped_samples[reason],
                    )
                skipped_blocks.clear()
                skipped_samples.clear()
            self._accum = np.concatenate([self._accum, observation.pcm])
            while len(self._accum) >= 512:
                frame = self._accum[:512]
                self._accum = self._accum[512:]
                if self._assist is None:
                    ready = ((frame, None, observation.pause_epoch),)
                else:
                    ready = await self._assist.push(frame, observation.pause_epoch)
                await self._gate_frames(ready, out_queue)

    async def _gate_frames(
        self,
        frames: Iterable[tuple[np.ndarray, Optional[float], Optional[int]]],
        out_queue: asyncio.Queue,
    ) -> None:
        """Score each frame with Silero, raise it to its tag term when the
        assist runs, and push it through the gate, emitting any segment.

        Each entry is ``(frame, tag term or None, pause epoch of its block)``.
        """
        for frame, tag_term, pause_epoch in frames:
            prob = float(self._model(frame))
            if tag_term is not None:
                prob = max(prob, tag_term)
            emit = self._gate.push(frame, prob)
            self._decided_samples += frame.size
            if emit is None:
                if self._open_window_id is None and self._gate.state_snapshot().in_speech:
                    self._open_window(pause_epoch)
                continue
            from yamabiko.main import Segment

            emission = self._gate.last_emission
            if emission is None:
                raise RuntimeError(
                    "FrameGate emitted audio without emission metadata"
                )
            window_id = self._open_window_id
            if self._gate.last_emit_was_dip_cut:
                # The gate stays open: the piece after the cut is a new window.
                self._open_window(pause_epoch)
            else:
                self._open_window_id = None
            # A dip cut leaves the gate open mid-utterance: the
            # Silero hidden state must NOT reset, or the right
            # piece is scored against a cold state and its own
            # dip boundaries drift from the offline spike. Reset
            # only at true closes (silence / max-chunk), as before.
            if not self._gate.last_emit_was_dip_cut:
                self._model.reset_states()
            segment = Segment(
                audio=emit,
                t_chunk_emit=time.perf_counter(),
                force_split=self._gate.last_emit_forced,
                discontinuity_before=(
                    self._discontinuity_before_next_segment
                ),
                vad_emit_reason=emission.reason.value,
                vad_gate_open_after_emit=emission.after.in_speech,
                vad_retained_frames_after_emit=(
                    emission.after.retained_frames
                ),
                pause_epoch=pause_epoch,
                window_id=window_id,
            )
            # Cleared before the put, which waits while the queue is full: a
            # dip cut's window is open meanwhile and its snapshot must not
            # report this segment's discontinuity as its own.
            self._discontinuity_before_next_segment = False
            await out_queue.put(segment)
