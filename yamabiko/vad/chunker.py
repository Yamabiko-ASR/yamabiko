"""Shared VAD/chunker configuration."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class ChunkerConfig:
    sample_rate: int = 16000
    min_chunk_ms: int = 1000
    max_chunk_ms: int = 8000
    min_silence_ms: int = 500
    speech_pad_ms: int = 200
    # Cap on how long a sub-min-chunk utterance keeps gluing across
    # trailing silence before it is emitted anyway. Genuine JP
    # filler gaps resume speech well within this window; an isolated
    # short utterance followed by pure silence emits here instead of
    # waiting for the max_chunk force-cap.
    glue_max_silence_ms: int = 1500
    # Attach-don't-extend (design doc 2026-07-02): audio between chunks
    # is buffered and attached to the NEXT chunk so the ASR hears it.
    # attach_max_ms bounds that buffer; audio is only discarded after
    # sitting below drop_floor_prob for drop_floor_ms continuously.
    attach_max_ms: int = 6000
    drop_floor_prob: float = 0.10
    drop_floor_ms: int = 2000
    # Dip-splitter: within an OPEN chunk (gate never closes on a rapid
    # A/B exchange, which has no inter-turn silence), cut at internal
    # Silero probability valleys so glued speaker turns become separate
    # subtitles. dip_split_enabled defaults OFF here, matching the
    # shipped default_config.yaml (the splitter regressed quality in a
    # full-distribution A/B, so it ships disabled). A valley is
    # a run of frames below dip_split_threshold; the cut is the run's
    # argmin frame, taken only when the left piece already spans
    # dip_split_min_piece_ms (the right piece keeps accumulating and
    # emits on the eventual close). See silero_chunker.FrameGate.
    dip_split_enabled: bool = False
    dip_split_threshold: float = 0.15
    dip_split_min_piece_ms: int = 1000


def chunker_config_from_config(cfg: Mapping[str, object]) -> tuple[ChunkerConfig, float]:
    """Build the pipeline's chunker config and VAD threshold from the full config.

    Production (``main.build_pipeline``) and the E0 replay both construct the
    gate here, so they cannot drift apart.
    """
    vad = cfg["vad"]
    config = ChunkerConfig(
        sample_rate=cfg["audio"]["sample_rate"],
        min_chunk_ms=vad["min_chunk_ms"],
        max_chunk_ms=vad["max_chunk_ms"],
        min_silence_ms=vad["min_silence_ms"],
        speech_pad_ms=vad["speech_pad_ms"],
        glue_max_silence_ms=vad.get("glue_max_silence_ms", 1500),
        attach_max_ms=vad.get("attach_max_ms", 6000),
        drop_floor_prob=vad.get("drop_floor_prob", 0.10),
        drop_floor_ms=vad.get("drop_floor_ms", 2000),
        dip_split_enabled=vad.get("dip_split_enabled", False),
        dip_split_threshold=vad.get("dip_split_threshold", 0.15),
        dip_split_min_piece_ms=vad.get("dip_split_min_piece_ms", 1000),
    )
    # Fallback 0.3 matches default_config.yaml's shipped produced-media
    # tuning, NOT Silero's generic 0.5. Dead in practice (the layered load
    # always restores the key), but a contradicting fallback would silently
    # re-tune VAD -- skipping whole reaction lines -- if the key were ever
    # dropped from the merge.
    return config, vad.get("threshold", 0.3)
