from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yamabiko.capture.audio import AudioConfig, LoopbackCapture
from yamabiko.capture.resampler import StreamingResampler


def _convert(
    samples: np.ndarray,
    in_rate: int,
    chunks: list[int],
) -> tuple[np.ndarray, StreamingResampler]:
    converter = StreamingResampler(in_rate, 16_000)
    outputs: list[np.ndarray] = []
    start = 0
    chunk_index = 0
    while start < len(samples):
        size = chunks[chunk_index % len(chunks)]
        outputs.append(converter.process(samples[start : start + size]))
        start += size
        chunk_index += 1
    outputs.append(converter.finish())
    return np.concatenate(outputs), converter


@pytest.mark.parametrize("in_rate", [44_100, 48_000, 96_000])
def test_streaming_output_is_chunk_invariant_and_has_exact_final_count(in_rate):
    rng = np.random.default_rng(105)
    samples = rng.normal(0.0, 0.1, in_rate // 3 + 17).astype(np.float32)

    whole, _ = _convert(samples, in_rate, [len(samples)])
    streamed, converter = _convert(samples, in_rate, [1, 997, 31, 4096, 7])

    expected_count = int(np.floor(len(samples) * 16_000 / in_rate + 0.5))
    assert len(streamed) == expected_count
    assert np.array_equal(streamed, whole)
    assert converter.input_samples_seen == len(samples)
    assert converter.output_samples_emitted == expected_count
    assert converter.pending_output_samples == pytest.approx(0.0)


def test_mq_resampling_preserves_passband_and_rejects_alias():
    in_rate = 48_000
    # Construct phase in float64, then cast the PCM. A float32 timebase adds
    # enough phase jitter to manufacture an apparent stopband floor.
    t = np.arange(in_rate, dtype=np.float64) / in_rate

    passband, _ = _convert(np.sin(2 * np.pi * 4_000 * t).astype(np.float32), in_rate, [1440])
    stopband, _ = _convert(np.sin(2 * np.pi * 12_000 * t).astype(np.float32), in_rate, [1440])

    # Ignore finite-sequence edge transients when measuring steady-state gain.
    trim = 1_600
    pass_rms = float(np.sqrt(np.mean(np.square(passband[trim:-trim]))))
    stop_rms = float(np.sqrt(np.mean(np.square(stopband[trim:-trim]))))
    pass_gain_db = 20 * np.log10(pass_rms / np.sqrt(0.5))
    alias_relative_db = 20 * np.log10(stop_rms / pass_rms)
    assert abs(pass_gain_db) <= 0.2
    assert alias_relative_db <= -80.0


def test_reset_matches_a_fresh_converter_and_reopens_finished_stream():
    rng = np.random.default_rng(17)
    discarded = rng.normal(size=9_600).astype(np.float32)
    kept = rng.normal(size=9_600).astype(np.float32)
    converter = StreamingResampler(48_000, 16_000)
    converter.process(discarded)
    converter.finish()

    converter.reset()
    reset_output = np.concatenate([converter.process(kept), converter.finish()])
    fresh_output, _ = _convert(kept, 48_000, [len(kept)])

    assert np.array_equal(reset_output, fresh_output)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float64),
        np.zeros(8, dtype=np.float32)[::2],
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
    ],
    ids=["rank", "dtype", "contiguity", "nan", "inf"],
)
def test_invalid_pcm_clears_native_state_without_poisoning_recovery(invalid):
    valid = np.linspace(-0.5, 0.5, 9_600, dtype=np.float32)
    converter = StreamingResampler(48_000, 16_000)
    converter.process(valid)

    with pytest.raises((TypeError, ValueError)):
        converter.process(invalid)

    recovered = np.concatenate([converter.process(valid), converter.finish()])
    fresh, _ = _convert(valid, 48_000, [len(valid)])
    assert np.array_equal(recovered, fresh)


def test_process_after_finish_requires_reset():
    converter = StreamingResampler(48_000, 16_000)
    converter.finish()
    with pytest.raises(RuntimeError, match="reset"):
        converter.process(np.zeros(1, dtype=np.float32))


def _open_callback(cap: LoopbackCapture, device: dict):
    captured: dict[str, object] = {}
    stream = MagicMock()
    cap._pa = MagicMock()
    cap._device_info = device

    def open_stream(**kwargs):
        captured["callback"] = kwargs["stream_callback"]
        return stream

    cap._pa.open.side_effect = open_stream
    with patch("yamabiko.capture.audio.pyaudio.paFloat32", 1):
        cap._open_stream()
    return captured["callback"]


def test_capture_16khz_bypass_enqueues_an_owned_copy():
    loop = MagicMock()
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=16_000),
        loop,
        MagicMock(spec=asyncio.Queue),
    )
    callback = _open_callback(
        cap,
        {"index": 0, "defaultSampleRate": 16_000, "maxInputChannels": 1},
    )
    source = bytearray(np.arange(480, dtype=np.float32).tobytes())

    callback(source, 480, None, 0)
    enqueued = loop.call_soon_threadsafe.call_args.args[1]
    before = enqueued.pcm.copy()
    source[:] = bytes(len(source))

    assert enqueued.pcm.dtype == np.float32
    assert enqueued.pcm.flags.c_contiguous
    assert enqueued.pcm.flags.owndata
    assert np.array_equal(enqueued.pcm, before)
