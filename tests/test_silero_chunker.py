from pathlib import Path

import numpy as np
import pytest

from yamabiko.vad.chunker import ChunkerConfig
from yamabiko.vad.silero_chunker import (
    FrameGate,
    FrameGateEmitReason,
    SileroChunker,
    SileroVAD,
)


def make_gate(
    min_chunk_ms: int = 1000,
    max_chunk_ms: int = 8000,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 200,
    glue_max_silence_ms: int = 1500,
    threshold: float = 0.5,
    attach_max_ms: int = 6000,
    drop_floor_prob: float = 0.10,
    drop_floor_ms: int = 2000,
    dip_split_enabled: bool = False,
    dip_split_threshold: float = 0.15,
    dip_split_min_piece_ms: int = 1000,
) -> FrameGate:
    cfg = ChunkerConfig(
        sample_rate=16000,
        min_chunk_ms=min_chunk_ms,
        max_chunk_ms=max_chunk_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        glue_max_silence_ms=glue_max_silence_ms,
        attach_max_ms=attach_max_ms,
        drop_floor_prob=drop_floor_prob,
        drop_floor_ms=drop_floor_ms,
        dip_split_enabled=dip_split_enabled,
        dip_split_threshold=dip_split_threshold,
        dip_split_min_piece_ms=dip_split_min_piece_ms,
    )
    return FrameGate(cfg, threshold=threshold, frame_samples=512)


def frame(sr: int = 16000) -> np.ndarray:
    return np.zeros(512, dtype=np.float32)


class TestFrameGate:
    def test_starts_silent_no_emit(self):
        g = make_gate()
        for _ in range(10):
            emit = g.push(frame(), prob=0.01)
            assert emit is None

    def test_single_brief_speech_plus_silence_emits(self):
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0)
        # 20 frames of speech (~640 ms at 512/16000 = 32 ms/frame)
        for _ in range(20):
            out = g.push(frame(), prob=0.9)
            assert out is None
        # 4 frames silence (~128 ms) > 100 ms min_silence
        emits: list = []
        for _ in range(4):
            out = g.push(frame(), prob=0.01)
            if out is not None:
                emits.append(out)
        assert len(emits) == 1
        samples = emits[0]
        assert len(samples) >= 16000 * 0.2  # >= 200 ms

    def test_sub_min_chunk_does_not_emit_too_early(self):
        """Filler-only 300 ms of speech + silence must not emit (under 1000 ms floor)."""
        g = make_gate(min_chunk_ms=1000, min_silence_ms=100, speech_pad_ms=0)
        # ~9 frames speech = ~290 ms
        for _ in range(9):
            assert g.push(frame(), prob=0.9) is None
        emits: list = []
        for _ in range(10):
            out = g.push(frame(), prob=0.01)
            if out is not None:
                emits.append(out)
        assert emits == [], "sub-minimum-chunk speech must not emit standalone"

    def test_force_emit_at_max_chunk(self):
        """Continuous speech past max_chunk_ms force-emits."""
        g = make_gate(max_chunk_ms=1000, min_silence_ms=500, speech_pad_ms=0)
        emits: list = []
        for _ in range(60):  # >> 31 frames at 32 ms/frame = 1000 ms
            out = g.push(frame(), prob=0.9)
            if out is not None:
                emits.append(out)
        assert len(emits) >= 1
        assert len(emits[0]) <= int(16000 * 1.0) + 512

    def test_speech_pad_prepended(self):
        """Pre-roll samples included when speech starts."""
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=100)
        for _ in range(10):
            g.push(frame(), prob=0.01)
        for _ in range(10):
            g.push(frame(), prob=0.9)
        emits: list = []
        for _ in range(5):
            out = g.push(frame(), prob=0.01)
            if out is not None:
                emits.append(out)
        assert len(emits) == 1
        assert len(emits[0]) >= int(16000 * 0.42)  # 320 ms speech + 100 ms pre-roll

    # --- Hysteresis (2026-07-02 anime-dropout fix). Upstream silero-vad
    # sustains an open segment down to neg_threshold = threshold - 0.15;
    # a single open==sustain threshold closed chunks mid-sentence
    # whenever background music dipped the speech prob below threshold,
    # shredding real-episode dialogue into sub-second fragments (measured
    # 81.3% -> 90.8% coverage on a full episode via scripts/vad_lab.py).

    def test_gate_sustains_speech_through_probs_between_neg_and_main_threshold(self):
        """Frames in [threshold-0.15, threshold) while IN speech must not
        count as silence.

        One-line seeded defect: compare against ``self.threshold`` instead
        of the derived neg threshold in push()'s in-speech branch -- the
        0.40-prob frames close the chunk early and the first assert goes
        red.
        """
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0,
                      threshold=0.5)
        for _ in range(10):
            assert g.push(frame(), prob=0.9) is None
        # 10 frames (~320 ms) at 0.40: above neg (0.35), below threshold.
        # min_silence is 100 ms (~4 frames): a non-hysteresis gate emits here.
        for _ in range(10):
            assert g.push(frame(), prob=0.40) is None, (
                "gate closed mid-speech on a music-dip prob in [neg, thr)"
            )
        # True silence still closes the chunk...
        emits = [g.push(frame(), prob=0.01) for _ in range(5)]
        chunks = [e for e in emits if e is not None]
        assert len(chunks) == 1
        # ...and the sustained frames are IN the chunk (20 frames >= 640 ms).
        assert len(chunks[0]) >= int(16000 * 0.64)

    def test_gate_closes_on_probs_below_neg_threshold(self):
        """Hysteresis must not over-sustain: prob below threshold-0.15 is
        silence and closes the chunk after min_silence_ms as before.

        Seeded defect: derive the neg threshold as threshold - 0.5 (too
        low) -- 0.30-prob frames sustain forever, no emit, red.
        """
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0,
                      threshold=0.5)
        for _ in range(10):
            assert g.push(frame(), prob=0.9) is None
        emits = [g.push(frame(), prob=0.30) for _ in range(5)]
        assert sum(e is not None for e in emits) == 1, (
            "prob below the neg threshold must terminate the chunk"
        )

    def test_gate_does_not_open_on_probs_between_neg_and_main_threshold(self):
        """Opening still requires the FULL threshold; the lowered bar only
        sustains already-open speech.

        Seeded defect: use the neg threshold in the not-in-speech branch
        of push() -- 0.40-prob frames open a chunk, the later silence
        emits it, red.
        """
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0,
                      threshold=0.5)
        for _ in range(30):
            assert g.push(frame(), prob=0.40) is None
        emits = [g.push(frame(), prob=0.01) for _ in range(10)]
        assert all(e is None for e in emits), (
            "sub-threshold probs must never open a chunk"
        )

    # --- The gate must record WHY it emitted. Only a max-chunk
    # force-split boundary legitimately re-emits the prior tail into the
    # next chunk; silence-terminated boundaries separate unrelated
    # utterances, and asr_stage uses this flag to gate the transcript
    # dedup. Seeded defect: hardcode forced=False in _close_and_emit.

    def test_force_emit_at_max_chunk_sets_forced_flag(self):
        g = make_gate(max_chunk_ms=1000, min_silence_ms=500, speech_pad_ms=0)
        emits: list = []
        for _ in range(60):
            out = g.push(frame(), prob=0.9)
            if out is not None:
                emits.append(out)
                break
        assert len(emits) == 1
        assert g.last_emit_forced is True
        assert g.last_emission is not None
        assert g.last_emission.reason is FrameGateEmitReason.FORCE_SPLIT
        assert g.last_emission.before.in_speech is True
        assert g.last_emission.after.retained_frames == 0

    def test_silence_terminated_emit_clears_forced_flag(self):
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0)
        for _ in range(20):
            g.push(frame(), prob=0.9)
        emitted = False
        for _ in range(4):
            if g.push(frame(), prob=0.01) is not None:
                emitted = True
                break
        assert emitted
        assert g.last_emit_forced is False
        assert g.last_emission is not None
        assert g.last_emission.reason is FrameGateEmitReason.SILENCE_CLOSE

    def test_glue_path_across_sub_min_chunk_silence(self):
        """Filler -> brief silence -> more speech -> emit as one chunk."""
        # frame_ms = 32 ms. min_chunk_ms=400 -> 12 frames. min_silence_ms=128 -> 4 frames.
        g = make_gate(min_chunk_ms=400, min_silence_ms=128, speech_pad_ms=0)
        emits: list = []

        # 6 frames of speech (~192 ms) -- below 12-frame floor
        for _ in range(6):
            out = g.push(frame(), prob=0.9)
            if out is not None:
                emits.append(out)
        # 4 frames of silence (~128 ms) -- hits min_silence_ms, but speech is sub-floor
        for _ in range(4):
            out = g.push(frame(), prob=0.01)
            if out is not None:
                emits.append(out)
        # 10 more frames of speech (~320 ms), total speech >> 12 frames
        for _ in range(10):
            out = g.push(frame(), prob=0.9)
            if out is not None:
                emits.append(out)
        # 4 frames of silence -- should emit now
        for _ in range(4):
            out = g.push(frame(), prob=0.01)
            if out is not None:
                emits.append(out)

        assert len(emits) == 1, f"expected exactly one emission via glue path, got {len(emits)}"
        # Emitted audio should span all 24 frames (6 speech + 4 glue silence + 10 speech + 4 end silence)
        assert len(emits[0]) == 24 * 512

    def test_isolated_sub_min_utterance_emits_at_glue_cap_not_max_chunk(self):
        """A sub-min-chunk utterance trailed by pure silence must emit at
        the glue cap (~min_chunk + glue_max_silence), NOT be held to the
        8 s max-chunk force-cap inside a mostly-silence chunk that feeds
        Whisper its hallucination zone.

        frame_ms = 32 ms. min_chunk_ms=1000 -> 31 frames (the 3-frame
        utterance is well under it). min_silence_ms=500 -> 15 frames.
        max_chunk_ms=8000 -> 250 frames. glue_max_silence_ms=1500 ->
        46 frames. So the gate must emit once _silence_since_speech hits
        46 — long before _total_frame_count would reach 250.

        Regression: nesting the glue-cap check under the 15-frame
        min-silence check evaluates it only at multiples of 15, so the
        configured 46-frame cap does not fire until frame 60.
        """
        g = make_gate(min_chunk_ms=1000, min_silence_ms=500,
                      max_chunk_ms=8000, glue_max_silence_ms=1500,
                      speech_pad_ms=0)
        # 3 speech frames: opens a sub-min (3 < 31 frame floor) chunk.
        for _ in range(3):
            assert g.push(frame(), prob=0.9) is None

        # Drive exactly the configured 46-frame silence cap. Nothing may
        # emit early, and the 46th frame must emit the isolated utterance.
        for _ in range(45):
            assert g.push(frame(), prob=0.0) is None
        emit = g.push(frame(), prob=0.0)
        assert emit is not None, (
            "isolated sub-min utterance did not emit on the configured "
            "46th silent frame"
        )
        # The emission is silence-terminated, not a forced max-chunk split.
        assert g.last_emit_forced is False
        assert g.last_emission is not None
        assert g.last_emission.reason is FrameGateEmitReason.GLUE_TIMEOUT

        # Trailing silence is bounded: 3 speech + at most min_silence (15)
        # frames = 18 frames. NOT 8 s of appended silence.
        assert len(emit) == (3 + 15) * 512, (
            f"emitted chunk must be trimmed to speech + min_silence "
            f"({(3 + 15) * 512} samples); got {len(emit)} — trailing "
            f"silence not bounded"
        )

    def test_default_config_yaml_carries_attach_keys_matching_chunkerconfig(self):
        """The shipped YAML and the dataclass defaults must agree, and
        main.py's plumb (Task 1) reads these exact keys.

        One-line seeded defect: delete `attach_max_ms:` from
        default_config.yaml -- red.
        """
        import yaml
        from pathlib import Path
        import yamabiko

        raw = yaml.safe_load(
            (Path(yamabiko.__file__).parent / "default_config.yaml")
            .read_text(encoding="utf-8")
        )
        cfg = ChunkerConfig()
        assert raw["vad"]["attach_max_ms"] == cfg.attach_max_ms == 6000
        assert raw["vad"]["drop_floor_prob"] == cfg.drop_floor_prob == 0.10
        assert raw["vad"]["drop_floor_ms"] == cfg.drop_floor_ms == 2000


def indexed_frame(i: int) -> np.ndarray:
    return np.full(512, float(i), dtype=np.float32)


class TestFrameGateAttach:
    def test_gate_attaches_inter_chunk_audio_to_next_chunk(self):
        """Ambiguous (sub-threshold, above-floor) audio between two
        utterances must ride into the SECOND chunk, not be discarded.

        One-line seeded defect: seed `_buf` from only the last
        `speech_pad_ms` of pending at open (the legacy ring) -- the
        ambiguous frames vanish, red.
        """
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0)
        i = 0
        # Utterance 1: 10 speech frames, then true-silence frames until
        # the gate closes it (min_silence 100 ms; capped at 6 pushes).
        for _ in range(10):
            assert g.push(indexed_frame(i), prob=0.9) is None; i += 1
        first = None
        for _ in range(6):
            out = g.push(indexed_frame(i), prob=0.01); i += 1
            if out is not None:
                first = out
                break
        assert first is not None
        # EVERYTHING pushed from here until the next open is inter-chunk
        # audio and must ride into the next chunk (design 4.1) --
        # including further silence frames, not just ambiguous ones.
        attach_start = i
        # 20 ambiguous frames (~640 ms): above drop floor, below open
        # threshold. Legacy behavior discarded these.
        for _ in range(20):
            assert g.push(indexed_frame(i), prob=0.20) is None; i += 1
        # Utterance 2 opens, then closes on silence.
        for _ in range(10):
            assert g.push(indexed_frame(i), prob=0.9) is None; i += 1
        second = None
        for _ in range(6):
            out = g.push(indexed_frame(i), prob=0.01); i += 1
            if out is not None:
                second = out
                break
        assert second is not None
        # The second chunk starts with the first post-emit frame.
        assert second[0] == float(attach_start), (
            "inter-chunk audio was not attached to the following chunk"
        )

    def test_gate_never_reemits_audio_already_emitted(self):
        """I3: pending restarts empty after every emit.

        One-line seeded defect: skip the pending clear in
        _close_and_emit -- the pre-utterance frames already emitted
        inside chunk 1 reappear at the front of chunk 2, red.
        """
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0)
        i = 0
        # Two sub-threshold frames FIRST: they ride into chunk 1 via
        # pending. Open copies pending without consuming it, so ONLY
        # the emit-time clear removes them -- which is exactly what
        # makes the seeded defect (skipped clear) re-emit them into
        # chunk 2 and go red.
        for _ in range(2):
            assert g.push(indexed_frame(i), prob=0.20) is None; i += 1
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        chunks = []
        for _ in range(6):
            out = g.push(indexed_frame(i), prob=0.01); i += 1
            if out is not None:
                chunks.append(out)
                break
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        for _ in range(6):
            out = g.push(indexed_frame(i), prob=0.01); i += 1
            if out is not None:
                chunks.append(out)
                break
        assert len(chunks) == 2
        first_ids = set(np.unique(chunks[0]).tolist())
        second_ids = set(np.unique(chunks[1]).tolist())
        assert not (first_ids & second_ids), (
            f"samples emitted twice: {sorted(first_ids & second_ids)}"
        )

    def test_gate_pending_overflow_drops_oldest_frames(self):
        """I4 aging: pending is bounded by attach_max_ms; overflow drops
        the OLDEST audio.

        One-line seeded defect: construct _pending without maxlen --
        the chunk grows past the attach bound and its first sample is
        frame 0, red.
        """
        # attach_max_ms=320 -> 10 frames of 32 ms.
        g = make_gate(min_chunk_ms=200, min_silence_ms=100,
                      speech_pad_ms=0, attach_max_ms=320)
        i = 0
        for _ in range(30):  # 3x the attach window, all ambiguous
            assert g.push(indexed_frame(i), prob=0.20) is None; i += 1
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        out = None
        for _ in range(4):
            r = g.push(indexed_frame(i), prob=0.01); i += 1
            if r is not None:
                out = r
        assert out is not None
        # Oldest surviving pending frame is 30-10=20.
        assert out[0] == 20.0, (
            f"expected pending capped at 10 frames (first sample 20.0), "
            f"got first sample {out[0]}"
        )

    def test_gate_discards_only_after_sustained_deep_silence(self):
        """I4: sub-floor audio shorter than drop_floor_ms is NOT
        discarded; a sustained run is.

        One-line seeded defect: compare against the OPEN threshold
        instead of drop_floor_prob when counting deep silence -- the
        0.20 refresher frames stop counting as pending-keepers, red.
        """
        # drop_floor_ms=320 -> 10 frames; pad=0 -> trim floor 1 frame.
        g = make_gate(min_chunk_ms=200, min_silence_ms=100,
                      speech_pad_ms=0, drop_floor_ms=320)
        i = 0
        # 9 sub-floor frames (< 10): must all survive in pending.
        for _ in range(9):
            assert g.push(indexed_frame(i), prob=0.01) is None; i += 1
        # One above-floor frame resets the run.
        assert g.push(indexed_frame(i), prob=0.20) is None; i += 1
        # 9 more sub-floor: still under the sustained limit.
        for _ in range(9):
            assert g.push(indexed_frame(i), prob=0.01) is None; i += 1
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        out = None
        for _ in range(4):
            r = g.push(indexed_frame(i), prob=0.01); i += 1
            if r is not None:
                out = r
        assert out is not None and out[0] == 0.0, (
            "pending was trimmed although deep silence never sustained "
            f"drop_floor_ms (first sample {out[0]}, expected 0.0)"
        )

    def test_gate_deep_silence_trim_reproduces_legacy_ring(self):
        """I5: after sustained deep silence, pending == the legacy
        speech_pad_ms ring, byte-for-byte, and STAYS trimmed while the
        silence continues.

        One-line seeded defect: make the trim a one-shot clear instead
        of sustained ring-mode -- post-trim silence re-grows pending and
        the chunk's first sample is too old, red.
        """
        # pad=64 ms -> 2 frames; drop_floor_ms=320 -> 10 frames.
        g = make_gate(min_chunk_ms=200, min_silence_ms=100,
                      speech_pad_ms=64, drop_floor_ms=320)
        i = 0
        for _ in range(30):  # 3x the sustained limit of deep silence
            assert g.push(indexed_frame(i), prob=0.01) is None; i += 1
        open_start = i
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        out = None
        for _ in range(4):
            r = g.push(indexed_frame(i), prob=0.01); i += 1
            if r is not None:
                out = r
        assert out is not None
        # Legacy ring: exactly 2 pad frames precede the opening frame.
        assert out[0] == float(open_start - 2), (
            f"expected legacy 2-frame pre-roll (first sample "
            f"{float(open_start - 2)}), got {out[0]}"
        )

    def test_deep_silence_counter_resets_at_emit_so_next_cycle_not_pretrimmed(self):
        """Cross-cycle invariant the open-reset removal relies on:
        _close_and_emit zeros _deep_silence_frame_count, so a SECOND
        not-in-speech period after an emit counts deep silence from zero
        and does NOT prematurely trim its pre-roll. (This is what makes
        the reset at speech-open redundant and safe to drop.)

        One-line seeded defect: delete `self._deep_silence_frame_count =
        0` from _close_and_emit -> cycle-2's first sub-floor frame
        inherits cycle-1's stale count, trips the drop floor on frame one,
        and trims the pre-roll to the pad ring -> the second chunk loses
        its early pending frames and starts too late -> red.
        """
        # drop_floor_ms=320 -> 10 frames; pad=0 -> pad_floor 1 frame.
        g = make_gate(min_chunk_ms=200, min_silence_ms=100,
                      speech_pad_ms=0, drop_floor_ms=320)
        i = 0
        # Cycle 1: sustained deep silence (>=10) drives the counter high,
        # then speech opens and closes -> emit (which must zero it).
        for _ in range(12):
            assert g.push(indexed_frame(i), prob=0.01) is None; i += 1
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        first = None
        for _ in range(6):
            out = g.push(indexed_frame(i), prob=0.01); i += 1
            if out is not None:
                first = out
                break
        assert first is not None
        # Cycle 2: only 5 sub-floor frames (< the 10-frame drop floor) --
        # NOT sustained, so all five must survive in pending as the next
        # chunk's pre-roll. With a stale counter they'd be trimmed away.
        cycle2_start = i
        for _ in range(5):
            assert g.push(indexed_frame(i), prob=0.01) is None; i += 1
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        second = None
        for _ in range(6):
            out = g.push(indexed_frame(i), prob=0.01); i += 1
            if out is not None:
                second = out
                break
        assert second is not None
        assert second[0] == float(cycle2_start), (
            f"second chunk lost its pre-roll: expected first sample "
            f"{float(cycle2_start)} (all 5 non-sustained sub-floor frames "
            f"survive), got {second[0]} -- deep-silence counter was not "
            f"reset at the previous emit"
        )

    def test_deep_silence_trim_keeps_one_pad_floor_frame_when_pad_is_zero(self):
        """The _pad_floor = max(_pad_frames, 1) floor: with speech_pad_ms=0
        a sustained-deep-silence trim must still leave exactly ONE pre-roll
        frame (the legacy maxlen=max(pad,1) ring), not zero.

        One-line seeded defect: drop the `, 1` from the floor
        (self._pad_floor = self._pad_frames) -> at pad=0 the trim's
        `while len > 0` empties pending -> the chunk starts at the opening
        frame with no pre-roll -> red.
        """
        # pad=0 -> pad_floor must be 1; drop_floor_ms=320 -> 10 frames.
        g = make_gate(min_chunk_ms=200, min_silence_ms=100,
                      speech_pad_ms=0, drop_floor_ms=320)
        i = 0
        for _ in range(30):  # sustained deep silence -> trim to pad floor
            assert g.push(indexed_frame(i), prob=0.01) is None; i += 1
        open_start = i
        for _ in range(10):
            g.push(indexed_frame(i), prob=0.9); i += 1
        out = None
        for _ in range(4):
            r = g.push(indexed_frame(i), prob=0.01); i += 1
            if r is not None:
                out = r
        assert out is not None
        assert out[0] == float(open_start - 1), (
            f"pad floor must keep exactly one pre-roll frame at pad=0 "
            f"(first sample {float(open_start - 1)}), got {out[0]}"
        )


class TestFrameGateBoundaryInvariance:
    """I1/I2/I6: attach must not move chunk boundaries, close timing,
    or force_split values. Emit push-indices are hand-derived from the
    legacy state machine; if attach state ever leaks into boundary
    logic these numbers drift and the tests go red.
    """

    def test_emit_indices_unchanged_with_ambiguous_preroll(self):
        """One-line seeded defect: count pending frames into
        _total_frame_count at open -- the force-emit fires early, red.
        """
        # min_chunk 200ms=6f, min_silence 100ms=3f (100/32 floors to 3),
        # pad 0. 15 ambiguous preroll frames, speech from index 15, gate
        # opens at push 15. 10 speech frames (15..24), silence from 25;
        # close needs 3 sub-sustain frames -> emit at push index 27.
        # max_chunk_ms=416 -> max_chunk_frames=13: chosen so the seeded
        # defect can fire. Unseeded, _total_frame_count peaks at 12 by
        # the silence-close (frames 16..27), 12 < 13 so force-split never
        # fires and the close stays at 27 / forced False. WITH the seed
        # (_total_frame_count = len(_buf) = 15 at open), the first
        # in-speech frame pushes total to 16 >= 13 -> force-emit at push
        # 16, so `emits` is no longer [27] and the test goes red.
        g = make_gate(min_chunk_ms=200, min_silence_ms=100, speech_pad_ms=0,
                      max_chunk_ms=416)
        emits = []
        for i in range(40):
            if i < 15:
                prob = 0.20
            elif i < 25:
                prob = 0.9
            else:
                prob = 0.01
            if g.push(indexed_frame(i), prob=prob) is not None:
                emits.append(i)
        assert emits == [27], f"close boundary moved: emits at {emits}"
        assert g.last_emit_forced is False

    def test_force_split_index_and_flag_unchanged_by_attach(self):
        """One-line seeded defect: include attached preroll in the
        max-chunk frame budget -- force-split fires early, red.
        """
        # max_chunk 320ms=10f. 20 ambiguous frames, speech from 20.
        # _total_frame_count counts post-open frames; force-emit when
        # it reaches 10 -> at push index 30.
        g = make_gate(min_chunk_ms=200, max_chunk_ms=320,
                      min_silence_ms=100, speech_pad_ms=0)
        emits = []
        for i in range(35):
            prob = 0.20 if i < 20 else 0.9
            if g.push(indexed_frame(i), prob=prob) is not None:
                emits.append(i)
        assert emits and emits[0] == 30, (
            f"force-split boundary moved: first emit at {emits[:1]}"
        )
        assert g.last_emit_forced is True


class TestFrameGateTimingValidation:
    """VAD timing keys reach FrameGate unvalidated. Below one 32 ms
    frame, ``int(ms / frame_ms)`` floors to 0 and the gate degenerates:
    ``min_silence_frames=0`` makes the silence check (``0 >= 0``) true on
    EVERY in-speech frame, so the gate emits fixed ~min_chunk-sized
    chunks mid-word with no silence ever observed; ``max_chunk_frames=0``
    force-emits on the second frame of every utterance. __post_init__
    must clamp both to sane floors and warn."""

    def test_sub_frame_min_silence_ms_clamped_to_one_frame(self, caplog):
        """min_silence_ms=0 must NOT let the gate emit mid-speech.

        Seeded defect: remove the ``max(1, ...)`` floor on
        _min_silence_frames -> the gate emits at the min_chunk boundary
        (frame ~2) with zero silence frames observed -> first assert red.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="yamabiko.vad.silero_chunker"):
            g = make_gate(min_chunk_ms=64, min_silence_ms=0, speech_pad_ms=0)
        # 50 frames of continuous speech: no silence -> no emission.
        for i in range(50):
            assert g.push(frame(), prob=0.9) is None, (
                f"gate emitted mid-speech at frame {i} with zero silence "
                f"frames observed — min_silence_frames degenerated to 0"
            )
        # One genuine silence frame (the clamped 1-frame floor) closes it.
        emit = g.push(frame(), prob=0.01)
        assert emit is not None, (
            "after the clamp, one full silence frame must still close "
            "the chunk (floor is 1 frame, not more)"
        )
        assert any("min_silence_ms" in rec.message for rec in caplog.records), (
            "clamping a sub-frame min_silence_ms must WARN — silent "
            "rewriting of user config hides the misconfiguration"
        )

    def test_sub_frame_max_chunk_ms_clamped_to_min_chunk_floor(self, caplog):
        """max_chunk_ms=0 must not force-emit on the second frame of
        every utterance; the floor is max(1, min_chunk_frames) -- so a
        force-split chunk is never below the configured minimum. (Not
        min_chunk+1: ``max == min`` is a legal boundary config that
        ``test_force_emit_at_max_chunk`` pins.)

        Seeded defect: remove the floor on _max_chunk_frames ->
        ``_total_frame_count (1) >= 0`` force-emits on frame 2 -> red.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="yamabiko.vad.silero_chunker"):
            g = make_gate(min_chunk_ms=64, max_chunk_ms=0,
                          min_silence_ms=64, speech_pad_ms=0)
        # min_chunk=2 frames -> clamped max = 2 frames. Frames 1-2 of
        # continuous speech must not emit (pre-fix: frame 2 force-emits
        # because _total_frame_count (1) >= 0).
        assert g.push(frame(), prob=0.9) is None  # opens the chunk
        assert g.push(frame(), prob=0.9) is None, (
            "max_chunk_frames degenerated to 0 — force-emit on the "
            "second frame of the utterance"
        )
        # Frame 3 hits the clamped 2-frame ceiling -> force emit, and the
        # emitted chunk (3 frames) is >= the configured min_chunk.
        emit = g.push(frame(), prob=0.9)
        assert emit is not None and g.last_emit_forced is True
        assert len(emit) >= 2 * 512
        assert any("max_chunk_ms" in rec.message for rec in caplog.records)

    def test_sane_timing_values_do_not_warn(self, caplog):
        """The clamp must not fire on the shipped defaults — a WARN on
        every healthy startup trains users to ignore the log.

        Seeded defect: clamp unconditionally (e.g. ``max(1, ...)``
        replaced by a warn-always branch) -> red.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="yamabiko.vad.silero_chunker"):
            make_gate()  # shipped defaults: 1000/8000/500/200 ms
        assert not caplog.records, (
            f"default timing config must not trigger clamp warnings; "
            f"got {[r.message for r in caplog.records]}"
        )


def drive_dip(gate, probs):
    """Push one indexed 512-sample frame per prob and collect emits with
    the gate flags captured at emit time. indexed_frame(i) fills the frame
    with the float i, so a piece's boundary is readable from its samples."""
    emits = []
    for i, p in enumerate(probs):
        out = gate.push(indexed_frame(i), prob=p)
        if out is not None:
            emits.append({
                "audio": out,
                "frames": len(out) // 512,
                "dip_cut": gate.last_emit_was_dip_cut,
                "forced": gate.last_emit_forced,
                "in_speech_after": gate._in_speech,
                "emission": gate.last_emission,
            })
    return emits


class TestFrameGateDipSplit:
    """Valley-cut splitting of glued exchanges. frame_ms = 32; the shipped
    dip config is threshold 0.15 / min_piece 1000 ms (31 frames).
    speech_pad_ms=0 keeps the pre-roll out of the frame math (no pending)."""

    def test_dip_splits_glued_runs_into_two_pieces(self):
        """Two speech runs joined by a short sub-0.15 valley emit as two
        pieces: the left piece cut at the valley's argmin frame while the
        gate never closes between them.

        Seeded defect: never advance _piece_start_k in _emit_dip_left (or
        never call _track_dip) -> a single close emit -> len(emits)==1, red.
        """
        g = make_gate(dip_split_enabled=True, speech_pad_ms=0,
                      min_chunk_ms=1000, min_silence_ms=500,
                      dip_split_min_piece_ms=1000)
        # run A: prob-index 0..39 (40 frames). valley: 40,41,42 (0.05, the
        # argmin is the first, index 40). run B: 43..82. close: 20 silence.
        probs = [0.9] * 40 + [0.05] * 3 + [0.9] * 40 + [0.01] * 20
        emits = drive_dip(g, probs)

        assert len(emits) == 2, f"expected a cut + a close, got {len(emits)}"
        # The cut is at prob-index 40 (valley argmin): left piece = frames
        # 0..39 = 40 frames, and the gate stayed open across the cut.
        assert emits[0]["dip_cut"] is True
        assert emits[0]["frames"] == 40, (
            f"cut fired at the wrong frame: {emits[0]['frames']} != 40"
        )
        assert emits[0]["audio"][0] == 0.0 and emits[0]["audio"][-1] == 39.0, (
            "left piece must be exactly the run-A frames 0..39"
        )
        assert emits[0]["in_speech_after"] is True, (
            "a dip cut must NOT close the gate"
        )
        assert emits[0]["emission"].reason is FrameGateEmitReason.DIP_CUT
        assert emits[0]["emission"].after.retained_frames > 0
        # The right piece begins at the cut frame (index 40) and is a
        # normal silence-terminated close.
        assert emits[1]["dip_cut"] is False and emits[1]["forced"] is False
        assert emits[1]["audio"][0] == 40.0, (
            "right piece must start exactly at the cut frame"
        )

    def test_no_cut_when_left_piece_below_min_piece(self):
        """A valley whose left piece is under min_piece must NOT cut: only
        the eventual close emits.

        Seeded defect: drop the `cut_k - _piece_start_k < min_piece` guard
        in _track_dip -> the 10-frame left piece cuts -> 2 emits, red.
        """
        g = make_gate(dip_split_enabled=True, speech_pad_ms=0,
                      min_chunk_ms=1000, min_silence_ms=500,
                      dip_split_min_piece_ms=1000)
        # run A only 10 frames (< 31), then a valley, then run B, then close.
        probs = [0.9] * 10 + [0.05] * 3 + [0.9] * 40 + [0.01] * 20
        emits = drive_dip(g, probs)

        assert len(emits) == 1, (
            f"left piece was under min_piece; must not cut, got {len(emits)}"
        )
        assert emits[0]["dip_cut"] is False

    def test_valley_running_into_close_is_the_close_not_a_cut(self):
        """A valley that never recovers (runs straight into a silence
        close) is the close, not a cut: exactly one emit, and it is
        byte-identical to the dip-DISABLED gate on the same input (trailing
        silence trimmed the same way).

        Seeded defect: emit a cut when the gate closes while _in_dip -> a
        spurious extra emit, red on the length/equality assert.
        """
        probs = [0.9] * 40 + [0.05] * 30      # 40 speech, then never recovers
        on = make_gate(dip_split_enabled=True, speech_pad_ms=0,
                       min_chunk_ms=1000, min_silence_ms=500,
                       dip_split_min_piece_ms=1000)
        off = make_gate(dip_split_enabled=False, speech_pad_ms=0,
                        min_chunk_ms=1000, min_silence_ms=500)
        e_on = drive_dip(on, probs)
        e_off = drive_dip(off, probs)

        assert len(e_on) == 1 and len(e_off) == 1
        assert e_on[0]["dip_cut"] is False and e_on[0]["forced"] is False
        # 40 speech + 15 (min_silence) trailing silence frames.
        assert e_on[0]["frames"] == 55, (
            f"trailing silence not trimmed to min_silence: {e_on[0]['frames']}"
        )
        assert np.array_equal(e_on[0]["audio"], e_off[0]["audio"]), (
            "enabled gate diverged from the disabled gate on a no-cut input"
        )

    def test_disabled_is_lossless_repartition_of_enabled(self):
        """Regression guarantee: on an input the enabled gate splits, the
        DISABLED gate emits one chunk, and the enabled gate's pieces
        concatenate back to that exact chunk (a loss-less repartition, no
        audio dropped or duplicated at a cut).

        Seeded defect: forget the `if self._dip_enabled` guard around the
        dip branch -> the disabled gate also splits -> len(off)!=1, red.
        Or a wrong buf_start in _close_and_emit -> concat mismatch, red.
        """
        probs = [0.9] * 40 + [0.05] * 3 + [0.9] * 40 + [0.01] * 20
        on = make_gate(dip_split_enabled=True, speech_pad_ms=0,
                       min_chunk_ms=1000, min_silence_ms=500,
                       dip_split_min_piece_ms=1000)
        off = make_gate(dip_split_enabled=False, speech_pad_ms=0,
                        min_chunk_ms=1000, min_silence_ms=500)
        e_on = drive_dip(on, probs)
        e_off = drive_dip(off, probs)

        assert len(e_off) == 1, "disabled gate must not split"
        assert len(e_on) == 2, "enabled gate must split this input"
        rejoined = np.concatenate([e["audio"] for e in e_on])
        assert np.array_equal(rejoined, e_off[0]["audio"]), (
            "split pieces do not concatenate back to the un-split chunk"
        )

    def test_enabled_two_cut_three_piece_is_lossless_repartition(self):
        """Two valleys -> three pieces: the enabled gate's three pieces
        concatenate back to the DISABLED gate's single chunk, byte-for-byte
        (loss-less across MULTIPLE cuts, not just one).

        Seeded defect: a wrong buf_start for the SECOND piece in
        _emit_dip_left (e.g. reuse the first-piece 0 start) -> the middle
        piece overlaps or gaps the boundary -> concat mismatch, red. Or drop
        the `if self._dip_enabled` guard -> the disabled gate also splits ->
        len(e_off) != 1, red.
        """
        # run A (40) | valley (3, argmin frame 40) | run B (40) |
        # valley (3, argmin frame 83) | run C (40) | close (20 silence).
        probs = ([0.9] * 40 + [0.05] * 3 + [0.9] * 40
                 + [0.05] * 3 + [0.9] * 40 + [0.01] * 20)
        on = make_gate(dip_split_enabled=True, speech_pad_ms=0,
                       min_chunk_ms=1000, min_silence_ms=500,
                       dip_split_min_piece_ms=1000)
        off = make_gate(dip_split_enabled=False, speech_pad_ms=0,
                        min_chunk_ms=1000, min_silence_ms=500)
        e_on = drive_dip(on, probs)
        e_off = drive_dip(off, probs)

        assert len(e_off) == 1, "disabled gate must not split"
        assert len(e_on) == 3, f"enabled gate must make two cuts, got {len(e_on)}"
        assert [e["dip_cut"] for e in e_on] == [True, True, False], (
            "the first two emits must be dip cuts and the last the close"
        )
        rejoined = np.concatenate([e["audio"] for e in e_on])
        assert np.array_equal(rejoined, e_off[0]["audio"]), (
            "three split pieces do not concatenate back to the un-split chunk"
        )

    def test_dip_cut_then_silent_right_piece_glue_cap_not_empty(self):
        """Non-default-config guard: after a dip cut, when the right piece
        never rises back to neg_threshold (dip_split_threshold >=
        neg_threshold, i.e. threshold > 0.30) and then glue-caps as a
        sub-min piece, the trailing-silence trim must NOT cut below the live
        piece's start -- otherwise _close_and_emit emits an EMPTY chunk.

        Seeded defect: revert the piece-start clamp to a bare
        `if keep < len(self._buf)` -> the glue-cap trims _buf below the
        right piece's buf_start -> the second emit is zero-length, red.

        Unreachable at shipped defaults (dip 0.15 < neg 0.15 is false and
        min_piece == min_chunk); this deliberately stresses threshold 0.5 +
        min_piece < min_chunk.
        """
        # threshold 0.5 -> neg 0.35 > dip 0.15: the band [0.15, 0.35)
        # recovers the valley (no more cut) but still counts as SILENCE, so
        # the right piece has no speech frame and _last_speech_frame_index
        # stays pinned in the already-emitted left piece.
        #   run A: 35 speech (frames 0..34, last speech = frame 34)
        #   valley: 36-frame slow descent, unique argmin at frame 70 (the cut)
        #   right piece: 8 frames at 0.20 (recover-but-silent) -> glue-caps
        valley = [round(0.14 - 0.003 * j, 4) for j in range(36)]  # frames 35..70
        probs = [0.9] * 35 + valley + [0.20] * 8
        g = make_gate(dip_split_enabled=True, speech_pad_ms=0,
                      threshold=0.5,
                      min_chunk_ms=2000,          # 62 frames > 35 speech -> sub-min
                      min_silence_ms=160,         # 5 frames
                      glue_max_silence_ms=1280,   # 40 frames
                      dip_split_threshold=0.15,
                      dip_split_min_piece_ms=1000)  # 31 frames < min_chunk
        emits = drive_dip(g, probs)

        assert len(emits) == 2, (
            f"expected a dip cut + a glue-cap close, got {len(emits)}"
        )
        assert emits[0]["dip_cut"] is True, "first emit must be the dip cut"
        assert emits[1]["dip_cut"] is False, "second emit must be the close"
        assert emits[1]["frames"] > 0, (
            "glue-cap close emitted an EMPTY chunk: the trailing-silence "
            "trim cut below the live piece's start"
        )
        assert emits[1]["audio"][0] == 70.0, (
            "right piece must start at the cut frame (70), not earlier"
        )

    def test_default_config_yaml_ships_dip_split_disabled_with_keys_present(self):
        """The shipped YAML ships the splitter DISABLED -- the initial
        thresholds regressed overall subtitle quality in a blind
        full-distribution A/B -- while still carrying the tuned threshold
        keys so the feature stays available to enable by hand. The dataclass
        also defaults it OFF, so direct-constructed gates stay byte-identical
        to the pre-feature gate unless explicitly enabled.

        Seeded defect: flip `dip_split_enabled:` back to true in the YAML ->
        the shipped-default `is False` assert goes red. Or delete
        `dip_split_threshold:` from the YAML -> KeyError, red.
        """
        import yaml
        from pathlib import Path
        import yamabiko

        raw = yaml.safe_load(
            (Path(yamabiko.__file__).parent / "default_config.yaml")
            .read_text(encoding="utf-8")
        )
        cfg = ChunkerConfig()
        assert raw["vad"]["dip_split_enabled"] is False, (
            "shipped config must ship the splitter DISABLED (initial "
            "thresholds regressed overall quality on a full-distribution A/B)"
        )
        assert cfg.dip_split_enabled is False, (
            "dataclass default must be OFF for byte-identical direct gates"
        )
        assert raw["vad"]["dip_split_threshold"] == cfg.dip_split_threshold == 0.15
        assert raw["vad"]["dip_split_min_piece_ms"] == cfg.dip_split_min_piece_ms == 1000


# ---------------------------------------------------------------------------
# Offline-startup contract of _default_model_path.
#
# Patches the hub seams (hf_hub_download + try_to_load_from_cache) so the
# subject -- the revision pin and the warm-cache offline switch -- runs for
# real.
# ---------------------------------------------------------------------------


class TestDefaultModelPath:
    def _patch_hub(self, monkeypatch, tmp_path, cache_warm: bool):
        import huggingface_hub

        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"onnx")
        probes: list[dict] = []
        downloads: list[dict] = []

        def fake_probe(repo_id, filename, revision=None, **kw):
            probes.append({"repo_id": repo_id, "filename": filename,
                           "revision": revision})
            return str(onnx) if cache_warm else None

        def fake_download(repo_id, filename, revision=None,
                          local_files_only=False, **kw):
            downloads.append({
                "repo_id": repo_id, "filename": filename,
                "revision": revision, "local_files_only": local_files_only,
            })
            return str(onnx)

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_probe)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        return probes, downloads, onnx

    def test_loads_offline_at_pinned_revision_when_cache_warm(
        self, monkeypatch, tmp_path,
    ):
        """A warmed cache must resolve the silero ONNX with
        local_files_only=True at the PINNED revision -- no per-startup
        HEAD revalidation, no floating with upstream `main`.

        Seeded defects (each one line in silero_chunker.py):
          - drop ``local_files_only=_silero_cached()`` -> red.
          - drop ``revision=SILERO_REVISION`` -> red.
        """
        import hashlib

        import onnxruntime

        from yamabiko import models
        from yamabiko.vad import silero_chunker as _m

        probes, downloads, onnx = self._patch_hub(
            monkeypatch, tmp_path, cache_warm=True,
        )
        expected_bytes = onnx.read_bytes()
        monkeypatch.setattr(
            _m, "SILERO_SHA256", hashlib.sha256(expected_bytes).hexdigest(),
        )
        constructed: list[Path] = []

        def fake_session(path, **_kwargs):
            constructed.append(Path(path))
            return object()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)

        path = _m._default_model_path()
        _m.SileroVAD(path)

        assert path == Path(str(onnx))
        assert constructed == [onnx]
        [d] = downloads
        assert d["repo_id"] == models.SILERO_REPO
        assert d["filename"] == models.SILERO_FILE
        assert d["revision"] == models.SILERO_REVISION, (
            "the download must target the pinned snapshot, not main"
        )
        assert d["local_files_only"] is True, (
            "warm cache must take the offline path"
        )
        assert probes and probes[0]["revision"] == models.SILERO_REVISION, (
            "the warmth probe must check the SAME pinned revision"
        )

    def test_rejects_warm_cache_sha_mismatch_without_session_or_repair(
        self, monkeypatch, tmp_path,
    ):
        import hashlib

        import onnxruntime

        from yamabiko.vad import silero_chunker as _m

        _, downloads, onnx = self._patch_hub(
            monkeypatch, tmp_path, cache_warm=True,
        )
        actual = hashlib.sha256(onnx.read_bytes()).hexdigest()
        expected = hashlib.sha256(b"different pinned artifact").hexdigest()
        monkeypatch.setattr(_m, "SILERO_SHA256", expected)

        def fail_session(*_args, **_kwargs):
            pytest.fail("hash mismatch reached InferenceSession construction")

        monkeypatch.setattr(onnxruntime, "InferenceSession", fail_session)

        with pytest.raises(RuntimeError) as exc_info:
            _m.SileroVAD(_m._default_model_path())

        message = str(exc_info.value)
        assert expected in message
        assert actual in message
        assert "stale or corrupt" in message
        assert [call["local_files_only"] for call in downloads] == [True], (
            "a corrupt warm-cache artifact must fail closed, not retry online"
        )

    def test_falls_back_online_when_cache_cold(self, monkeypatch, tmp_path):
        """A cold cache keeps the online fallback (dev mode, no wizard run)
        -- still at the pinned revision.

        Seeded defect: hardcode local_files_only=True -> red.
        """
        from yamabiko import models
        from yamabiko.vad import silero_chunker as _m

        _, downloads, _ = self._patch_hub(
            monkeypatch, tmp_path, cache_warm=False,
        )

        _m._default_model_path()

        [d] = downloads
        assert d["local_files_only"] is False
        assert d["revision"] == models.SILERO_REVISION


class _FakeOrtSession:
    """Mimics ``ort.InferenceSession`` for the ONNX contract we use."""

    def __init__(self, prob: float = 0.9) -> None:
        self._prob = prob
        self.failure: str | None = None
        self.calls: list[dict] = []

    def run(self, _outputs, inputs):
        self.calls.append({k: v.copy() if hasattr(v, "copy") else v for k, v in inputs.items()})
        if self.failure == "session":
            inputs["state"].fill(99.0)
            raise RuntimeError("injected session failure")
        prob = np.array([[self._prob]], dtype=np.float32)
        # Mutate the hidden state so we can verify it flows round-trip.
        next_state = inputs["state"] + 1.0
        if self.failure == "output":
            next_state = np.zeros((1,), dtype=np.float32)
        return [prob, next_state]


class TestSileroVAD:
    """SileroVAD owns only ONNX I/O — no torch, no silero_vad pkg."""

    def _make_vad(self, prob: float = 0.9) -> SileroVAD:
        vad = SileroVAD.__new__(SileroVAD)  # bypass __init__ (no real ONNX file)
        vad._session = _FakeOrtSession(prob=prob)
        vad._state = np.zeros((2, 1, 128), dtype=np.float32)
        vad._context = np.zeros((1, 64), dtype=np.float32)
        vad._sr = np.array(16000, dtype=np.int64)
        return vad

    def test_call_prepends_context_and_threads_hidden_state(self):
        vad = self._make_vad(prob=0.73)
        first = np.arange(512, dtype=np.float32)
        second = np.arange(512, dtype=np.float32) + 1000.0

        prob = vad(first)
        vad(second)

        assert isinstance(prob, float)
        assert prob == pytest.approx(0.73)
        first_input = vad._session.calls[0]["input"]
        second_input = vad._session.calls[1]["input"]
        assert first_input.shape == second_input.shape == (1, 576)
        assert np.array_equal(first_input[0, :64], np.zeros(64, dtype=np.float32))
        assert np.array_equal(first_input[0, 64:], first)
        assert np.array_equal(second_input[0, :64], first[-64:])
        assert np.array_equal(second_input[0, 64:], second)
        assert np.allclose(vad._state, 2.0)
        assert np.allclose(vad._session.calls[1]["state"], 1.0)

    def test_reset_states_replaces_state_and_context_and_replays_first_call(self):
        vad = self._make_vad()
        frame = np.arange(512, dtype=np.float32)
        first_prob = vad(frame)
        first_input = vad._session.calls[0]["input"].copy()
        assert not np.allclose(vad._state, 0.0)
        assert not np.allclose(vad._context, 0.0)
        prior_state = vad._state
        prior_context = vad._context

        vad.reset_states()

        assert vad._state is not prior_state
        assert vad._context is not prior_context
        assert np.allclose(vad._state, 0.0)
        assert np.allclose(vad._context, 0.0)
        replay_prob = vad(frame)
        assert replay_prob == first_prob
        assert np.array_equal(vad._session.calls[1]["input"], first_input)
        assert np.allclose(vad._session.calls[1]["state"], 0.0)

    def test_call_auto_casts_non_float32(self):
        vad = self._make_vad()
        frame = np.zeros(512, dtype=np.float64)
        vad(frame)  # must not raise; implementation casts internally
        assert vad._session.calls[0]["input"].dtype == np.float32

    @pytest.mark.parametrize("shape", [(511,), (513,), (1, 512)])
    def test_call_rejects_non_frame_shape_without_advancing_state(
        self, shape,
    ):
        vad = self._make_vad()
        prior_state = vad._state.copy()
        prior_context = vad._context.copy()

        with pytest.raises(ValueError, match="exactly 512 samples"):
            vad(np.zeros(shape, dtype=np.float32))

        assert np.array_equal(vad._state, prior_state)
        assert np.array_equal(vad._context, prior_context)
        assert vad._session.calls == []

    @pytest.mark.parametrize(
        ("failure", "error_type"),
        [("session", RuntimeError), ("output", ValueError)],
    )
    def test_failed_inference_or_output_validation_rolls_back(
        self, failure, error_type,
    ):
        vad = self._make_vad()
        vad(np.arange(512, dtype=np.float32))
        prior_state = vad._state.copy()
        prior_context = vad._context.copy()
        vad._session.failure = failure

        with pytest.raises(error_type):
            vad(np.full(512, 7.0, dtype=np.float32))

        assert np.array_equal(vad._state, prior_state)
        assert np.array_equal(vad._context, prior_context)

    def test_init_rejects_non_16khz_before_touching_model(self, tmp_path):
        with pytest.raises(ValueError, match="only supports 16000 Hz"):
            SileroVAD(tmp_path / "missing.onnx", sample_rate=8000)


class TestSileroChunker:
    """SileroChunker accepts an injected model — no ONNX file needed in tests."""

    def test_constructor_accepts_injected_model(self):
        cfg = ChunkerConfig(
            sample_rate=16000, min_chunk_ms=1000, max_chunk_ms=8000,
            min_silence_ms=500, speech_pad_ms=200,
        )

        class FakeModel:
            def __init__(self):
                self.resets = 0

            def __call__(self, frame):
                return 0.0

            def reset_states(self):
                self.resets += 1

        model = FakeModel()
        chunker = SileroChunker(cfg, threshold=0.5, model=model)
        assert chunker._model is model


class _ScriptedModel:
    """Returns a scripted probability per 512-sample frame."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self.resets = 0

    def __call__(self, frame_arr) -> float:
        return self._probs.pop(0)

    def reset_states(self) -> None:
        self.resets += 1


class TestSileroChunkerRunForceSplit:
    """run() must stamp Segment.force_split from the gate so asr_stage
    can distinguish a mid-speech force-split boundary (dedup legitimate)
    from a silence-terminated one (unrelated utterances).
    Seeded defect: drop force_split= from the Segment construction in
    run() (default False) -> first assertion goes red."""

    @pytest.mark.asyncio
    async def test_run_stamps_force_split_then_clears_on_silence_emit(self):
        import asyncio

        cfg = ChunkerConfig(
            sample_rate=16000, min_chunk_ms=64, max_chunk_ms=320,
            min_silence_ms=64, speech_pad_ms=0,
        )
        # 32 ms frames: max=10, min_silence=2, min_chunk=2.
        # Frames 1-11: continuous speech -> force emit at max-chunk.
        # Frames 12-14: speech reopens; 15-16: silence -> silence emit.
        probs = [0.9] * 14 + [0.01] * 2
        chunker = SileroChunker(cfg, threshold=0.5, model=_ScriptedModel(probs))
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        try:
            await in_q.put(np.zeros(16 * 512, dtype=np.float32))
            seg1 = await asyncio.wait_for(out_q.get(), timeout=2.0)
            seg2 = await asyncio.wait_for(out_q.get(), timeout=2.0)
        finally:
            task.cancel()
        assert seg1.force_split is True
        assert seg2.force_split is False
        assert seg1.vad_emit_reason == FrameGateEmitReason.FORCE_SPLIT.value
        assert seg2.vad_emit_reason == FrameGateEmitReason.SILENCE_CLOSE.value
        assert seg1.vad_gate_open_after_emit is False
        assert seg2.vad_retained_frames_after_emit == 0


class TestSileroChunkerRunLoop:
    """The force-split test drives run() but only asserts the force_split
    flag. The loop's other contracts — reset_states() per emit, the
    paused-discard path, and the emitted audio payload — require separate
    invocation coverage: a regression in any of them (VAD hidden state
    poisoning every subsequent utterance; pause feeding audio anyway;
    wrong audio handed to ASR) would ship with the suite green.
    """

    @pytest.mark.asyncio
    async def test_run_resets_vad_state_once_per_emitted_chunk(self):
        """Silero's recurrent hidden state must be zeroed at every chunk
        boundary; without it, state from utterance N poisons the
        probabilities for utterance N+1 and chunk boundaries silently
        degrade on the default shipped path.

        Seeded defect: delete the `self._model.reset_states()` call in
        run() -> resets stays 0 -> red.
        """
        import asyncio

        cfg = ChunkerConfig(
            sample_rate=16000, min_chunk_ms=64, max_chunk_ms=320,
            min_silence_ms=64, speech_pad_ms=0,
        )
        # 32 ms frames: min_silence=2, min_chunk=2, max=10 frames.
        # Two silence-terminated utterances: 4 speech + 2 silence, twice.
        probs = ([0.9] * 4 + [0.01] * 2) * 2
        model = _ScriptedModel(probs)
        chunker = SileroChunker(cfg, threshold=0.5, model=model)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        try:
            await in_q.put(np.zeros(12 * 512, dtype=np.float32))
            await asyncio.wait_for(out_q.get(), timeout=2.0)
            await asyncio.wait_for(out_q.get(), timeout=2.0)
        finally:
            task.cancel()
        assert model.resets == 2, (
            f"reset_states() must run once per emitted chunk (2 emits); "
            f"got {model.resets}. Stale hidden state poisons every "
            f"subsequent utterance's VAD probabilities."
        )

    @pytest.mark.asyncio
    async def test_pause_abandons_active_gate_before_resumed_segment(self):
        """A pause edge abandons active pre-pause speech instead of gluing it
        to the first resumed utterance. Only resumed audio may emit, and that
        first segment carries both the discontinuity and resumed epoch.
        """
        import asyncio

        from yamabiko.capture.envelope import (
            CaptureBlock,
            CaptureDiscontinuity,
            CaptureGapReason,
        )
        from yamabiko.state import AppState

        cfg = ChunkerConfig(
            sample_rate=16000, min_chunk_ms=64, max_chunk_ms=320,
            min_silence_ms=64, speech_pad_ms=0,
        )

        class _ContentModel:
            """Prob keyed off frame content so the script can't desync:
            0.5-filled frames are speech, zeros are silence."""
            def __init__(self):
                self.calls = 0
                self.resets = 0

            def __call__(self, frame) -> float:
                self.calls += 1
                return 0.9 if frame[0] > 0.4 else 0.01

            def reset_states(self) -> None:
                self.resets += 1

        state = AppState()
        model = _ContentModel()
        chunker = SileroChunker(cfg, threshold=0.5, state=state, model=model)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        try:
            # Four pre-pause frames open a gate but do not emit.
            await in_q.put(CaptureBlock(
                np.full(4 * 512, 0.5, dtype=np.float32),
                0,
                0,
                pause_epoch=0,
            ))
            deadline = asyncio.get_running_loop().time() + 2.0
            while model.calls < 4:
                assert asyncio.get_running_loop().time() < deadline, (
                    f"chunker never processed the speech block "
                    f"(model.calls={model.calls})"
                )
                await asyncio.sleep(0.01)

            paused = state.pause()
            resumed = state.resume()
            assert paused is not None and resumed is not None
            await in_q.put(CaptureDiscontinuity(
                CaptureGapReason.USER_PAUSE,
                pause_epoch=paused.epoch,
            ))
            await in_q.put(CaptureDiscontinuity(
                CaptureGapReason.USER_RESUME,
                pause_epoch=resumed.epoch,
            ))

            # A complete resumed utterance closes independently.
            resumed_audio = np.concatenate([
                np.full(4 * 512, 0.7, dtype=np.float32),
                np.zeros(2 * 512, dtype=np.float32),
            ])
            await in_q.put(CaptureBlock(
                resumed_audio,
                0,
                4 * 512,
                pause_epoch=resumed.epoch,
            ))
            seg = await asyncio.wait_for(out_q.get(), timeout=2.0)
        finally:
            task.cancel()

        audio = seg.audio
        assert len(audio) == 6 * 512
        assert np.count_nonzero(audio == np.float32(0.7)) == 4 * 512
        assert not np.any(audio == np.float32(0.5))
        assert seg.force_split is False
        assert seg.discontinuity_before is True
        assert seg.pause_epoch == resumed.epoch
        assert model.resets == 3

    @pytest.mark.asyncio
    async def test_overflow_marker_reset_of_open_gate_logs_abandoned_speech(
        self, caplog,
    ):
        """A queue-overflow marker that abandons an open utterance must say
        so at INFO, naming the capture reason, so lost speech is visible in
        a default-level log instead of vanishing silently.

        Seeded defect: log every discontinuity reset at DEBUG, or pass a
        fixed label instead of the marker's reason -> no matching record, red.
        """
        import asyncio
        import logging

        from yamabiko.capture.envelope import (
            CaptureBlock,
            CaptureDiscontinuity,
            CaptureGapReason,
        )

        cfg = ChunkerConfig(
            sample_rate=16000, min_chunk_ms=64, max_chunk_ms=320,
            min_silence_ms=64, speech_pad_ms=0,
        )
        # Four speech frames open the gate without emitting (max is 10).
        model = _ScriptedModel([0.9] * 4)
        chunker = SileroChunker(cfg, threshold=0.5, model=model)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        with caplog.at_level(logging.INFO, logger="yamabiko.vad.silero_chunker"):
            task = asyncio.create_task(chunker.run(in_q, out_q))
            try:
                await in_q.put(CaptureBlock(
                    np.full(4 * 512, 0.5, dtype=np.float32), 0, 0,
                ))
                await in_q.put(CaptureDiscontinuity(
                    CaptureGapReason.AUDIO_QUEUE_OVERFLOW,
                ))
                # No segment closes, so the only model reset is the marker's.
                deadline = asyncio.get_running_loop().time() + 2.0
                while model.resets < 1:
                    assert asyncio.get_running_loop().time() < deadline, (
                        "chunker never processed the overflow marker"
                    )
                    await asyncio.sleep(0.01)
            finally:
                task.cancel()

        assert out_q.empty(), "abandoned speech must not be emitted"
        resets = [
            rec for rec in caplog.records
            if rec.name == "yamabiko.vad.silero_chunker"
            and "vad discontinuity reset" in rec.getMessage()
        ]
        assert [rec.levelno for rec in resets] == [logging.INFO], (
            f"expected one INFO reset record; got "
            f"{[(rec.levelname, rec.getMessage()) for rec in resets]}"
        )
        assert (
            "reason=audio_queue_overflow abandoned_in_speech=True"
            in resets[0].getMessage()
        )


class _ScriptedVAD:
    """prob per frame from a list; the production SileroVAD protocol."""
    def __init__(self, probs):
        self._probs = list(probs)
        self._i = 0
    def __call__(self, frame):
        p = self._probs[min(self._i, len(self._probs) - 1)]
        self._i += 1
        return p
    def reset_states(self):
        pass


class _PauseState:
    def __init__(self):
        self.paused = False


def test_unpause_clears_prepause_pending_audio():
    """I7: audio buffered before a pause must not surface in a chunk
    emitted after unpause.

    One-line seeded defect: drop the was_paused clear in
    SileroChunker.run -- the pre-pause ambiguous frames appear at the
    front of the post-unpause chunk, red.
    """
    import asyncio

    async def scenario():
        cfg = ChunkerConfig(min_chunk_ms=200, min_silence_ms=100,
                            speech_pad_ms=0)
        state = _PauseState()
        # probs: 5 ambiguous (buffered), then pause, then speech + close.
        model = _ScriptedVAD([0.20] * 5 + [0.9] * 10 + [0.01] * 6)
        chunker = SileroChunker(cfg, threshold=0.5, state=state, model=model)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        # 5 ambiguous frames arrive unpaused -> land in pending.
        for i in range(5):
            await in_q.put(np.full(512, float(i), dtype=np.float32))
        await asyncio.sleep(0.05)
        # Pause; a block arriving while paused is discarded.
        state.paused = True
        await in_q.put(np.full(512, 99.0, dtype=np.float32))
        await asyncio.sleep(0.05)
        state.paused = False
        # Speech + closing silence after unpause.
        for i in range(5, 21):
            await in_q.put(np.full(512, float(i), dtype=np.float32))
        seg = await asyncio.wait_for(out_q.get(), timeout=5)
        task.cancel()
        return seg

    seg = asyncio.run(scenario())
    assert seg.audio[0] >= 5.0, (
        f"pre-pause audio leaked into the post-unpause chunk "
        f"(first sample {seg.audio[0]})"
    )


def test_unpause_resets_silero_hidden_state():
    """Both sampled Boolean pause edges are hard discontinuities, so Silero
    resets once when paused intake is rejected and once before resumed input.

    The resets are observed before any emit: 4 speech frames open a chunk,
    the paused block is discarded, and one post-unpause silence frame cannot
    close the freshly reset gate.
    """
    import asyncio
    from types import SimpleNamespace

    async def scenario():
        cfg = ChunkerConfig(min_chunk_ms=64, min_silence_ms=64,
                            speech_pad_ms=0)

        class _ContentModel:
            def __init__(self):
                self.calls = 0
                self.resets = 0

            def __call__(self, frame) -> float:
                self.calls += 1
                return 0.9 if frame[0] > 0.4 else 0.01

            def reset_states(self) -> None:
                self.resets += 1

        state = SimpleNamespace(paused=False)
        model = _ContentModel()
        chunker = SileroChunker(cfg, threshold=0.5, state=state, model=model)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        try:
            # 4 speech frames open a chunk; no closing silence yet, so no
            # emit -> resets stays 0 (only an emit resets, pre-fix).
            await in_q.put(np.full(4 * 512, 0.5, dtype=np.float32))
            deadline = asyncio.get_running_loop().time() + 2.0
            while model.calls < 4:
                assert asyncio.get_running_loop().time() < deadline, (
                    f"chunker never processed the speech block "
                    f"(model.calls={model.calls})"
                )
                await asyncio.sleep(0.01)
            assert model.resets == 0, "no emit yet -> no reset yet"

            # Pause: one block discarded whole, no model call.
            state.paused = True
            await in_q.put(np.full(512, 9.9, dtype=np.float32))
            while not in_q.empty():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            assert model.calls == 4, "paused block must not reach the model"

            # Unpause: the first post-unpause block triggers the second edge
            # reset. One silence frame cannot emit from the freshly reset gate.
            state.paused = False
            await in_q.put(np.zeros(512, dtype=np.float32))
            deadline = asyncio.get_running_loop().time() + 2.0
            while model.calls < 5:
                assert asyncio.get_running_loop().time() < deadline, (
                    f"chunker never processed the post-unpause block "
                    f"(model.calls={model.calls})"
                )
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            return model.resets, out_q.empty()
        finally:
            task.cancel()

    resets, out_empty = asyncio.run(scenario())
    assert out_empty, "the still-open chunk must not have emitted"
    assert resets == 2, (
        f"pause and unpause must each reset Silero exactly once; got {resets}"
    )


class TestSileroChunkerDipCutModelReset:
    """run() must reset the Silero hidden state on a true close but NOT on
    a dip cut. Resetting at a cut would score the right piece against a cold
    state, drifting its dip boundaries away from the offline spike (which
    ran one continuous prob stream per chunk). This is the load-bearing
    parity invariant behind the acceptance gate.
    """

    @pytest.mark.asyncio
    async def test_dip_cut_does_not_reset_model_close_does(self):
        """One glued utterance that dip-splits into two pieces: two Segments
        emit, but the model resets ONCE (the close), not twice.

        Seeded defect: drop the `if not last_emit_was_dip_cut` guard in
        run() and always reset -> resets == 2, red. Or make the gate return
        last_emit_was_dip_cut=False for a cut -> resets == 2, red.
        """
        import asyncio

        # frame_ms=32. min_piece 64ms=2f, min_chunk 64ms=2f,
        # min_silence 200ms=6f (so the 3-frame valley does NOT silence-close
        # before it recovers), max_chunk large.
        cfg = ChunkerConfig(
            sample_rate=16000, min_chunk_ms=64, max_chunk_ms=8000,
            min_silence_ms=200, speech_pad_ms=0,
            dip_split_enabled=True, dip_split_threshold=0.15,
            dip_split_min_piece_ms=64,
        )
        # run A 5f, valley 3f (recovers), run B 5f, close 6f silence.
        probs = [0.9] * 5 + [0.05] * 3 + [0.9] * 5 + [0.01] * 6
        model = _ScriptedModel(probs)
        chunker = SileroChunker(cfg, threshold=0.5, model=model)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        try:
            await in_q.put(np.zeros(len(probs) * 512, dtype=np.float32))
            seg_cut = await asyncio.wait_for(out_q.get(), timeout=2.0)
            seg_close = await asyncio.wait_for(out_q.get(), timeout=2.0)
        finally:
            task.cancel()

        assert seg_cut.force_split is False and seg_close.force_split is False
        assert model.resets == 1, (
            f"model must reset only at the true close, not the dip cut; "
            f"got {model.resets} resets for 2 emits"
        )


class TestOpenWindowSnapshot:
    """SileroChunker.open_window_snapshot() for incremental emission
    (yamabiko/asr/incremental.py). The sound-tag assist runs with a silent
    tagger, so the last fed frames wait in its delay line as in production."""

    async def test_snapshot_neither_mutates_nor_drains_the_chunker(self, monkeypatch):
        """Two chunkers get the same audio; one is snapshotted after every
        block and every snapshot's audio is overwritten. Both must emit the
        same window, and that closed window keeps its snapshots' id.

        Seeded defect: return a view of the gate's buffer, or clear it, in
        open_window_snapshot -> the snapshotted chunker's segment differs (red)."""
        import asyncio

        from tests.vad.test_sound_tag_assist import (
            _ContentSilero, _blocks, _frames, _silent_tagger, _until,
        )
        from yamabiko.vad import silero_chunker

        monkeypatch.setattr(silero_chunker, "STALL_DRAIN_S", 30.0)
        audio = _frames([(0.3, 0.0), (2.0, 0.5), (2.5, 0.0)])
        results = []
        for snapshot in (False, True):
            chunker = SileroChunker(
                ChunkerConfig(), threshold=0.3, model=_ContentSilero(), tagger=_silent_tagger,
            )
            in_q: asyncio.Queue = asyncio.Queue()
            out_q: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(chunker.run(in_q, out_q))
            seen_ids: set[int] = set()
            try:
                for block in _blocks(audio):
                    in_q.put_nowait(block)
                    await _until(in_q.empty)
                    await asyncio.sleep(0)
                    if snapshot:
                        snap = chunker.open_window_snapshot()
                        if snap is not None:
                            seen_ids.add(snap.window_id)
                            snap.audio[:] = 1.0
                segment = await asyncio.wait_for(out_q.get(), timeout=5.0)
            finally:
                task.cancel()
            results.append((segment, seen_ids))
        (plain, _), (snapped, seen_ids) = results
        assert np.array_equal(snapped.audio, plain.audio)
        assert snapped.vad_emit_reason == plain.vad_emit_reason
        assert seen_ids == {snapped.window_id} == {0}

    async def test_snapshot_holds_only_the_frames_the_gate_admitted(self, monkeypatch):
        """Behind the sound-tag delay line the last fed frames wait for their
        look-ahead; the snapshot stops at the frames the gate has pushed.

        Seeded defect: build the snapshot from the fed audio, or include the
        assist's held frames -> it is longer than model.calls * 512 (red)."""
        import asyncio

        from tests.vad.test_sound_tag_assist import (
            _ContentSilero, _blocks, _frames, _silent_tagger, _until,
        )
        from yamabiko.vad import silero_chunker

        monkeypatch.setattr(silero_chunker, "STALL_DRAIN_S", 30.0)
        audio = _frames([(1.5, 0.5)])
        model = _ContentSilero()
        chunker = SileroChunker(ChunkerConfig(), threshold=0.3, model=model, tagger=_silent_tagger)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(chunker.run(in_q, out_q))
        try:
            for block in _blocks(audio):
                in_q.put_nowait(block)
            await _until(in_q.empty)
            await asyncio.sleep(0.2)
            snap = chunker.open_window_snapshot()
        finally:
            task.cancel()
        admitted = model.calls * 512
        assert 0 < admitted < audio.size  # the delay line still holds frames
        assert snap.audio.size == admitted == chunker.decided_samples
        assert np.array_equal(snap.audio, audio[:admitted])
