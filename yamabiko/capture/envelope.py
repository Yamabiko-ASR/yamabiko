"""Content-free coordinates for PCM crossing the capture/asyncio boundary."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

import numpy as np


class GapExtent(StrEnum):
    EXACT_OUTPUT_SAMPLES = "exact_output_samples"
    UNKNOWN_SOURCE_DURATION = "unknown_source_duration"


class CaptureGapReason(StrEnum):
    AUDIO_QUEUE_OVERFLOW = "audio_queue_overflow"
    INPUT_STATUS_GAP = "input_status_gap"
    TRANSFORM_FAILURE = "transform_failure"
    STREAM_REOPENED = "stream_reopened"
    USER_PAUSE = "user_pause"
    USER_RESUME = "user_resume"


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


@dataclass(frozen=True, slots=True)
class CaptureBlock:
    """One owned block on a generation-local 16 kHz output clock."""

    pcm: np.ndarray
    stream_generation: int
    start_sample: int
    pause_epoch: int | None = None

    def __post_init__(self) -> None:
        pcm = self.pcm
        if not isinstance(pcm, np.ndarray):
            raise TypeError("pcm must be a numpy.ndarray")
        if pcm.ndim != 1:
            raise ValueError("pcm must be one-dimensional mono PCM")
        if pcm.dtype != np.float32:
            raise TypeError("pcm must have dtype float32")
        if not pcm.flags.c_contiguous:
            raise ValueError("pcm must be C-contiguous")
        if not pcm.flags.owndata:
            raise ValueError("pcm must own its sample buffer")
        if not np.isfinite(pcm).all():
            raise ValueError("pcm must contain only finite values")
        object.__setattr__(
            self,
            "stream_generation",
            _non_negative_integer(self.stream_generation, "stream_generation"),
        )
        object.__setattr__(
            self,
            "start_sample",
            _non_negative_integer(self.start_sample, "start_sample"),
        )
        if self.pause_epoch is not None:
            object.__setattr__(
                self,
                "pause_epoch",
                _non_negative_integer(self.pause_epoch, "pause_epoch"),
            )

    @property
    def end_sample(self) -> int:
        return self.start_sample + int(self.pcm.size)


@dataclass(frozen=True, slots=True)
class CaptureDiscontinuity:
    """Ordered marker for a boundary whose source duration is not inferred."""

    reason: CaptureGapReason
    extent: GapExtent = GapExtent.UNKNOWN_SOURCE_DURATION
    stream_generation: int | None = None
    at_sample: int | None = None
    pause_epoch: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", CaptureGapReason(self.reason))
        object.__setattr__(self, "extent", GapExtent(self.extent))
        if self.stream_generation is not None:
            object.__setattr__(
                self,
                "stream_generation",
                _non_negative_integer(
                    self.stream_generation,
                    "stream_generation",
                ),
            )
        if self.at_sample is not None:
            object.__setattr__(
                self,
                "at_sample",
                _non_negative_integer(self.at_sample, "at_sample"),
            )
        if self.pause_epoch is not None:
            object.__setattr__(
                self,
                "pause_epoch",
                _non_negative_integer(self.pause_epoch, "pause_epoch"),
            )


@dataclass(frozen=True, slots=True)
class CaptureObservation:
    pcm: np.ndarray
    stream_generation: int | None
    start_sample: int | None
    exact_gap_samples: int | None
    generation_changed: bool
    pause_epoch: int | None = None


class CaptureSequenceTracker:
    """Validate envelope order without inferring across legacy payloads."""

    def __init__(self) -> None:
        self._generation: int | None = None
        self._end_sample: int | None = None

    def _clear_baseline(self) -> None:
        self._generation = None
        self._end_sample = None

    def observe(self, payload: CaptureBlock | np.ndarray) -> CaptureObservation:
        if isinstance(payload, np.ndarray):
            self._clear_baseline()
            return CaptureObservation(payload, None, None, None, False, None)
        if not isinstance(payload, CaptureBlock):
            raise TypeError("capture payload must be CaptureBlock or numpy.ndarray")

        generation_changed = False
        exact_gap_samples: int | None = None
        if self._generation is not None:
            if payload.stream_generation < self._generation:
                raise ValueError("capture generation arrived out of order")
            if payload.stream_generation != self._generation:
                generation_changed = True
            else:
                assert self._end_sample is not None
                if payload.start_sample < self._end_sample:
                    raise ValueError("capture block overlaps or arrived out of order")
                exact_gap_samples = payload.start_sample - self._end_sample

        self._generation = payload.stream_generation
        self._end_sample = payload.end_sample
        return CaptureObservation(
            payload.pcm,
            payload.stream_generation,
            payload.start_sample,
            exact_gap_samples,
            generation_changed,
            payload.pause_epoch,
        )


@dataclass(frozen=True, slots=True)
class CaptureContinuityEvent:
    """Stable content-free continuity telemetry emitted on the loop thread."""

    event_sequence: int
    event_kind: str
    extent: GapExtent | None
    reason: CaptureGapReason | None
    stream_generation: int | None
    start_sample: int | None
    sample_count: int | None
    pause_epoch: int | None = None
