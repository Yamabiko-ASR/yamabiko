"""Continuous, band-limited sample-rate conversion for live capture."""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import soxr


class StreamingResampler:
    """A mono float32 wrapper around one continuous SoXR stream.

    ``finish`` exists for finite sequences and synthetic verification. Live
    capture deliberately discards a stream generation without flushing it.
    """

    def __init__(
        self,
        in_rate: Real,
        out_rate: Real,
        quality: str = "MQ",
    ) -> None:
        self.in_rate = self._validated_rate(in_rate, "in_rate")
        self.out_rate = self._validated_rate(out_rate, "out_rate")
        self.quality = quality
        self._stream: soxr.ResampleStream
        self.input_samples_seen = 0
        self.output_samples_emitted = 0
        self._finished = False
        self._new_stream()

    @staticmethod
    def _validated_rate(value: Real, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite positive number")
        rate = float(value)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        return rate

    def _new_stream(self) -> None:
        self._stream = soxr.ResampleStream(
            self.in_rate,
            self.out_rate,
            1,
            dtype="float32",
            quality=self.quality,
        )

    @staticmethod
    def _validate_samples(samples: np.ndarray) -> None:
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if samples.ndim != 1:
            raise ValueError("samples must be one-dimensional mono PCM")
        if samples.dtype != np.float32:
            raise TypeError("samples must have dtype float32")
        if not samples.flags.c_contiguous:
            raise ValueError("samples must be C-contiguous")
        if not np.isfinite(samples).all():
            raise ValueError("samples must contain only finite values")

    @staticmethod
    def _normalized_output(output: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(output, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._finished:
            raise RuntimeError("resampler is finished; call reset() before process()")
        try:
            self._validate_samples(samples)
            output = self._stream.resample_chunk(samples, last=False)
        except Exception:
            self.reset()
            raise
        output = self._normalized_output(output)
        self.input_samples_seen += len(samples)
        self.output_samples_emitted += len(output)
        return output

    def finish(self) -> np.ndarray:
        if self._finished:
            raise RuntimeError("resampler is already finished; call reset() first")
        output = self._normalized_output(
            self._stream.resample_chunk(np.empty(0, dtype=np.float32), last=True),
        )
        self.output_samples_emitted += len(output)
        self._finished = True
        return output

    def reset(self) -> None:
        """Discard all filter/phase state and begin a fresh generation."""
        self._new_stream()
        self.input_samples_seen = 0
        self.output_samples_emitted = 0
        self._finished = False

    @property
    def pending_output_samples(self) -> float:
        """Output samples retained in SoXR's current filter state."""
        return float(self._stream.delay())
