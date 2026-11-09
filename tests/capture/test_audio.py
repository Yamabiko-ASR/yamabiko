"""LoopbackCapture tests. Mocks pyaudiowpatch — no real WASAPI stack."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yamabiko.capture.audio import AudioConfig, LoopbackCapture
from yamabiko.capture.envelope import CaptureBlock, CaptureGapReason, GapExtent


def _fake_pa(devices: list[dict], default_output: dict | None = None,
             default_loopback: dict | None = None):
    """Build a mock pyaudio.PyAudio instance exposing the given device table."""
    pa = MagicMock()
    pa.get_device_count.return_value = len(devices)
    pa.get_device_info_by_index.side_effect = lambda i: devices[i]
    if default_loopback is not None:
        pa.get_default_wasapi_loopback.return_value = default_loopback
    else:
        pa.get_default_wasapi_loopback.side_effect = RuntimeError("no default loopback")
    if default_output is not None:
        pa.get_default_output_device_info.return_value = default_output
    return pa


def test_resolve_device_uses_get_default_wasapi_loopback():
    """`device="default"` must prefer the dedicated helper."""
    target = {"index": 7, "name": "Speakers (Realtek) [Loopback]",
              "isLoopbackDevice": True,
              "defaultSampleRate": 48000, "maxInputChannels": 2}
    pa = _fake_pa([], default_loopback=target)
    cap = LoopbackCapture(AudioConfig(device="default"), asyncio.new_event_loop(), asyncio.Queue())
    cap._pa = pa
    assert cap._resolve_device() is target
    pa.get_default_wasapi_loopback.assert_called_once()


def test_substring_match_picks_correct_loopback_among_duplicates():
    """device="Speakers" must match the Speakers loopback, not Headset or Mic."""
    devs = [
        {"index": 0, "name": "Microphone (USB)", "isLoopbackDevice": False,
         "defaultSampleRate": 16000, "maxInputChannels": 1},
        {"index": 1, "name": "Headset Earphone [Loopback]", "isLoopbackDevice": True,
         "defaultSampleRate": 48000, "maxInputChannels": 2},
        {"index": 2, "name": "Speakers (Realtek) [Loopback]", "isLoopbackDevice": True,
         "defaultSampleRate": 48000, "maxInputChannels": 2},
    ]
    pa = _fake_pa(devs)
    cap = LoopbackCapture(AudioConfig(device="Speakers"), asyncio.new_event_loop(), asyncio.Queue())
    cap._pa = pa
    picked = cap._resolve_device()
    assert picked["index"] == 2
    assert "Speakers" in picked["name"]


@pytest.mark.parametrize("device_name", [
    "Speakers (Realtek) [Loopback]",   # no quote characters
    "Dave's AirPods [Loopback]",       # apostrophe -> repr() switches to "..."
    'Bob\'s "Good" Buds [Loopback]',   # both quote styles -> repr escapes
])
def test_missing_pinned_device_raises_typed_error_carrying_the_exact_name(
    device_name,
):
    """The pinned-device-gone raise site must carry the device name as a FIELD,
    byte-exact, for every name a real Windows endpoint can have.

    This is the PRODUCER half of the round trip that the startup dialog depends
    on (the CONSUMER half is
    tests/test_error_dialog.py::test_classify_typed_loopback_gone_names_device_for_any_quote_style).
    Both halves exist because deriving the name by re-parsing str(exc) is a trap:
    the message is built with !r, and repr() switches to double quotes for a name
    containing an apostrophe, which is exactly how Windows names a personal
    Bluetooth headset.

    Seeded-defect drill: revert the raise to
    `RuntimeError(f"Loopback device not found: {self.config.device!r}")` ->
    the exception has no .device_name attribute -> AttributeError -> red.
    """
    from yamabiko.capture._audio_constants import LoopbackDeviceNotFound

    # A device table where the pinned device is absent (it vanished).
    other = {"index": 0, "name": "Microphone (USB)", "isLoopbackDevice": False,
             "defaultSampleRate": 16000, "maxInputChannels": 1}
    cap = LoopbackCapture(
        AudioConfig(device=device_name),
        asyncio.new_event_loop(),
        asyncio.Queue(),
    )
    cap._pa = _fake_pa([other])

    with pytest.raises(LoopbackDeviceNotFound) as ei:
        cap._resolve_device()

    assert ei.value.device_name == device_name, (
        f"the raise site must carry the pinned name verbatim; got "
        f"{ei.value.device_name!r} for {device_name!r}"
    )
    # Still a RuntimeError: existing callers and the classifier's keyword
    # fallback match on that.
    assert isinstance(ei.value, RuntimeError)


def test_start_falls_back_to_default_when_saved_device_is_missing():
    """An unplugged saved endpoint must not abort startup.

    The saved preference lives outside ``AudioConfig``; this runtime copy can
    follow the Windows default for the current session. Removing the fallback
    makes ``start()`` raise ``LoopbackDeviceNotFound`` here.
    """
    default = {
        "index": 7,
        "name": "Speakers (Realtek) [Loopback]",
        "isLoopbackDevice": True,
        "defaultSampleRate": 48000,
        "maxInputChannels": 2,
    }
    other = {
        "index": 0,
        "name": "Microphone (USB)",
        "isLoopbackDevice": False,
        "defaultSampleRate": 16000,
        "maxInputChannels": 1,
    }
    pa = _fake_pa([other], default_loopback=default)
    stream = MagicMock()
    pa.open.return_value = stream
    cap = LoopbackCapture(
        AudioConfig(device="Headphones (WH-1000XM5) [Loopback]"),
        asyncio.new_event_loop(),
        asyncio.Queue(),
    )
    cap._device_grace_attempts = 0
    cap._read_default_output_id = lambda: "default-endpoint-id"
    cap._start_monitor = MagicMock()

    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=pa), \
         patch("yamabiko.capture.audio.pyaudio.paFloat32", 1):
        cap.start()

    assert cap.config.device == "default"
    assert cap._device_info is default
    assert cap._default_output_id == "default-endpoint-id"
    stream.start_stream.assert_called_once()


def test_endpoint_name_is_the_core_audio_friendly_name():
    # Lead decision 2: PyAudio names a WASAPI loopback "<friendly name>
    # [Loopback]"; the problem copy and the presence check use the friendly
    # name. Seeded defects: keep the suffix, or raise before start() -> red.
    cap = LoopbackCapture(AudioConfig(device="default"), asyncio.new_event_loop(), asyncio.Queue())
    assert cap.endpoint_name() is None
    cap._device_info = {"index": 3, "name": "Headphones (USB Audio) [Loopback]"}
    assert cap.endpoint_name() == "Headphones (USB Audio)"


def _run_start(cap: LoopbackCapture, device: dict,
               endpoint_id: str = "{0.0.0.00000000}.{unit-stub-endpoint}"):
    """Run LoopbackCapture.start() with pyaudio AND the F59 Core Audio
    default-endpoint seam mocked; return the captured stream callback.

    Stubbing current_default_render_endpoint_id keeps the UNIT tier off the
    real Core Audio COM path (hermeticity -- start() snapshots the default
    endpoint when device=='default') AND lets a test observe exactly what
    start() captures into _default_output_id.
    """
    captured = {}
    pa = _fake_pa([device], default_loopback=device)
    stream = MagicMock()

    def fake_open(**kwargs):
        captured["cb"] = kwargs["stream_callback"]
        return stream
    pa.open.side_effect = fake_open

    # Override the Core Audio default-output seam as an INSTANCE attribute
    # (not just a patch around start()): start() snapshots _default_output_id
    # from it AND any monitor a test starts afterward polls the SAME stub --
    # so the unit tier never touches real Core Audio COM, and a
    # start()-then-monitor test never sees a spurious switch from a stubbed
    # baseline vs a live poll.
    cap._read_default_output_id = lambda: endpoint_id

    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=pa), \
         patch("yamabiko.capture.audio.pyaudio.paFloat32", 1), \
         patch("yamabiko.capture.audio.pyaudio.paContinue", 0), \
         patch("yamabiko.capture.audio.pyaudio.paComplete", 2):
        cap.start()
    return captured.get("cb")


def _start_captures_callback(cap: LoopbackCapture, device: dict):
    """Back-compat shim: run start() (pyaudio + the Core Audio default-output
    seam both mocked) and return the captured stream callback."""
    return _run_start(cap, device)


def test_callback_bridges_pcm_to_asyncio_queue_via_call_soon_threadsafe():
    """Foreign-thread callback must put np.float32 mono frames on the queue via loop.call_soon_threadsafe."""
    loop = MagicMock()
    q = MagicMock(spec=asyncio.Queue)
    cap = LoopbackCapture(AudioConfig(device="default", sample_rate=48000),
                          loop, q)
    cb = _start_captures_callback(cap, {
        "index": 0, "name": "X [Loopback]",
        "defaultSampleRate": 48000, "maxInputChannels": 1, "isLoopbackDevice": True,
    })

    samples = np.arange(480, dtype=np.float32)
    cb(samples.tobytes(), 480, None, 0)
    loop.call_soon_threadsafe.assert_called_once()
    args = loop.call_soon_threadsafe.call_args[0]
    # The callback schedules the back-pressure-safe _enqueue wrapper, not
    # raw put_nowait (which would raise asyncio.QueueFull on the loop).
    assert args[0] == cap._enqueue  # bound-method equality (fresh obj per access)
    enqueued = args[1]
    assert enqueued.pcm.dtype == np.float32
    assert enqueued.pcm.shape == (480,)
    assert np.allclose(enqueued.pcm, samples)
    assert (enqueued.stream_generation, enqueued.start_sample) == (0, 0)


def test_stereo_downmix_averages_channels_before_enqueue():
    """2-channel float32 input must be mean-mixed to mono before put_nowait."""
    loop = MagicMock()
    q = MagicMock(spec=asyncio.Queue)
    cap = LoopbackCapture(AudioConfig(device="default", sample_rate=48000),
                          loop, q)
    cb = _start_captures_callback(cap, {
        "index": 0, "name": "Stereo [Loopback]",
        "defaultSampleRate": 48000, "maxInputChannels": 2, "isLoopbackDevice": True,
    })

    # 4 stereo samples — left=[1,2,3,4], right=[3,4,5,6], mean=[2,3,4,5]
    stereo = np.array([[1, 3], [2, 4], [3, 5], [4, 6]], dtype=np.float32)
    cb(stereo.tobytes(), 4, None, 0)
    enqueued = loop.call_soon_threadsafe.call_args[0][1]
    assert enqueued.pcm.shape == (4,)
    assert np.allclose(enqueued.pcm, [2.0, 3.0, 4.0, 5.0])


def test_callback_streams_48k_device_audio_to_16k_config_rate():
    """48 kHz capture must accumulate into a continuous 16 kHz stream.

    SoXR intentionally emits variable-sized batches and may retain an entire
    callback while its filter delay accumulates. The cumulative sample clock,
    not a fixed 160-output-samples-per-callback shape, is the contract.
    """
    loop = MagicMock()
    q = MagicMock(spec=asyncio.Queue)
    cap = LoopbackCapture(AudioConfig(device="default", sample_rate=16000),
                          loop, q)
    cb = _start_captures_callback(cap, {
        "index": 0, "name": "X [Loopback]",
        "defaultSampleRate": 48000, "maxInputChannels": 1, "isLoopbackDevice": True,
    })

    samples = np.zeros(20 * 480, dtype=np.float32)
    for block in samples.reshape(-1, 480):
        cb(block.tobytes(), len(block), None, 0)

    batches = [call.args[1] for call in loop.call_soon_threadsafe.call_args_list]
    enqueued = np.concatenate([batch.pcm for batch in batches])
    ideal_count = round(len(samples) * 16_000 / 48_000)
    assert ideal_count - 800 <= len(enqueued) <= ideal_count
    assert all(
        call.args[0] == cap._enqueue
        for call in loop.call_soon_threadsafe.call_args_list
    )
    assert all(batch.pcm.dtype == np.float32 for batch in batches)


def test_callback_streams_44k1_device_audio_yields_16k_sample_count():
    """44.1 kHz callbacks must share one rational-phase conversion state."""
    loop = MagicMock()
    q = MagicMock(spec=asyncio.Queue)
    cap = LoopbackCapture(AudioConfig(device="default", sample_rate=16000),
                          loop, q)
    cb = _start_captures_callback(cap, {
        "index": 0, "name": "X [Loopback]",
        "defaultSampleRate": 44100, "maxInputChannels": 1, "isLoopbackDevice": True,
    })

    samples = np.zeros(20 * 441, dtype=np.float32)
    for block in samples.reshape(-1, 441):
        cb(block.tobytes(), len(block), None, 0)

    batches = [call.args[1] for call in loop.call_soon_threadsafe.call_args_list]
    enqueued = np.concatenate([batch.pcm for batch in batches])
    ideal_count = round(len(samples) * 16_000 / 44_100)
    assert ideal_count - 800 <= len(enqueued) <= ideal_count
    assert all(batch.pcm.dtype == np.float32 for batch in batches)


def test_wasapi_adc_time_jump_emits_suspected_source_gap_and_keeps_every_block():
    """WASAPI never sets PortAudio's underflow/overflow status flags, so an
    input ADC time that jumps past three frame durations is the only sign of
    lost source time. It must surface as telemetry while every block still
    reaches the queue: no marker, no dropped PCM.

    Seeded defect: delete the ADC-gap branch in the callback -> no
    suspected_source_gap event -> red.
    """
    loop = MagicMock()
    q: asyncio.Queue = asyncio.Queue()
    events = []
    cap = LoopbackCapture(AudioConfig(device="default", sample_rate=16000),
                          loop, q, on_capture_event=events.append)
    cb = _start_captures_callback(cap, {
        "index": 0, "name": "X [Loopback]",
        "defaultSampleRate": 16000, "maxInputChannels": 1, "isLoopbackDevice": True,
    })

    frame = np.zeros(480, dtype=np.float32)  # 30 ms at 16 kHz
    for adc in (1.00, 1.03, 1.20):
        cb(frame.tobytes(), 480, {"input_buffer_adc_time": adc}, 0)
    for call in loop.call_soon_threadsafe.call_args_list:
        call.args[0](*call.args[1:])

    assert [event.event_kind for event in events] == ["suspected_source_gap"]
    gap = events[0]
    assert gap.extent is GapExtent.UNKNOWN_SOURCE_DURATION
    assert gap.reason is CaptureGapReason.INPUT_STATUS_GAP
    # 0.17 s between ADC stamps minus one 30 ms frame = 140 ms at 16 kHz.
    assert (gap.stream_generation, gap.start_sample, gap.sample_count) == (
        0, 960, 2240,
    )
    queued = [q.get_nowait() for _ in range(q.qsize())]
    assert all(isinstance(item, CaptureBlock) for item in queued)
    assert [block.start_sample for block in queued] == [0, 480, 960]



@pytest.mark.asyncio
async def test_queue_overflow_marks_gap_before_pcm_and_chunker_resets_state():
    """A dropped capture frame is a stream discontinuity, not silence.

    The marker must enter the full queue after already-buffered PCM and before
    any later PCM.  Consuming it must abandon partial samples, an open gate,
    and Silero's recurrent state so pre-gap speech cannot emit with post-gap
    silence.
    """
    from yamabiko.vad.chunker import ChunkerConfig
    from yamabiko.vad.silero_chunker import SileroChunker

    class _ContentModel:
        def __init__(self) -> None:
            self.calls = 0
            self.resets = 0

        def __call__(self, frame) -> float:
            self.calls += 1
            return 0.9 if frame[0] > 0.4 else 0.01

        def reset_states(self) -> None:
            self.resets += 1

    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=1)
    out_q: asyncio.Queue = asyncio.Queue()
    cap = LoopbackCapture(AudioConfig(), loop, audio_q)
    model = _ContentModel()
    chunker = SileroChunker(
        ChunkerConfig(
            min_chunk_ms=64,
            max_chunk_ms=320,
            min_silence_ms=64,
            speech_pad_ms=0,
        ),
        threshold=0.5,
        model=model,
    )

    # Four speech frames open the gate; the extra half-frame remains in
    # SileroChunker._accum.  Filling this one-slot queue forces the next frame
    # to drop and therefore schedules the discontinuity marker behind it.
    pre_gap = np.full(4 * 512 + 256, 0.5, dtype=np.float32)
    audio_q.put_nowait(pre_gap)
    cap._enqueue(CaptureBlock(np.full(512, 9.9, dtype=np.float32), 0, 0))
    assert cap._dropped == 1

    task = asyncio.create_task(chunker.run(audio_q, out_q))
    try:
        deadline = loop.time() + 2.0
        while model.resets < 1:
            assert loop.time() < deadline, "discontinuity marker was not consumed"
            await asyncio.sleep(0.01)

        # If the marker arrived late, or failed to clear the half-frame
        # accumulator/open gate, these silence frames deterministically close
        # and emit the pre-gap speech.
        cap._enqueue(CaptureBlock(
            np.zeros(3 * 512, dtype=np.float32),
            0,
            512,
        ))
        deadline = loop.time() + 2.0
        while model.calls < 7:
            assert loop.time() < deadline, "post-gap PCM was not processed"
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

        # The first complete utterance after the gap carries the boundary to
        # downstream ASR, where force-split dedup state must be abandoned.
        cap._enqueue(CaptureBlock(
            np.concatenate([
                np.full(4 * 512, 0.5, dtype=np.float32),
                np.zeros(2 * 512, dtype=np.float32),
            ]),
            0,
            4 * 512,
        ))
        post_gap_seg = await asyncio.wait_for(out_q.get(), timeout=2.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert out_q.empty(), "pre-gap audio leaked into a post-gap chunk"
    assert post_gap_seg.discontinuity_before is True
    assert model.resets == 2


def test_stop_releases_resources_even_if_stop_stream_raises():
    """stop() must close the stream + terminate PyAudio even if an earlier
    teardown step raises, and must never propagate (it runs inside
    run_pipeline's finally, where a raise would skip the rest of cleanup).

    Seeded-defect drill: revert stop() to the unguarded form -> stop_stream's
    RuntimeError propagates, close()/terminate() are skipped -> this fails.
    """
    cap = LoopbackCapture(AudioConfig(), MagicMock(), MagicMock(spec=asyncio.Queue))
    stream = MagicMock()
    stream.stop_stream.side_effect = RuntimeError("device gone")
    cap._stream = stream
    pa = MagicMock()
    cap._pa = pa

    cap.stop()  # must NOT raise

    stream.close.assert_called_once()
    pa.terminate.assert_called_once()


# --- Persistent transform failure must throttle logging and ---
# --- escalate to stream death instead of stalling invisibly  ---

_DEV_48K_MONO = {
    "index": 0, "name": "X [Loopback]",
    "defaultSampleRate": 48000, "maxInputChannels": 1,
    "isLoopbackDevice": True,
}

# Any byte length that isn't a multiple of 4 makes
# np.frombuffer(dtype=float32) raise ValueError inside the callback.
_BAD_FRAME = b"\x00\x01\x02"


def test_leaked_capture_monitor_is_stopped_by_capture_tier_guard(stop_capture_monitors_fn):
    """A monitor a test never stopped itself must be stoppable by the
    capture-tier leak guard (tests/capture/conftest.py::stop_capture_monitors).

    start() spawns the ``audio-stream-monitor`` daemon thread inside the
    ``_run_start`` helper's ``with patch("...PyAudio", ...)`` block. If that
    thread outlives the block, its 1 s idle-reopen poll eventually calls the
    now-real ``pyaudiowpatch.PyAudio()`` and access-violates, killing the whole
    pytest process (exit 139, root-caused 2026-07-08). The guard must terminate
    the thread so that can never happen.

    Seeded-defect drill: delete the ``cap.stop()`` inside
    ``stop_capture_monitors`` -> the monitor keeps polling -> the 5 s join times
    out -> ``still_alive`` is non-empty and ``mon.is_alive()`` stays True -> red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    # Start the capture the same way the callback tests do -- pyaudio is mocked
    # only for the duration of _start_captures_callback -- and DELIBERATELY do
    # not call cap.stop() ourselves: the guard must do it.
    _start_captures_callback(cap, dict(_DEV_48K_MONO))
    mon = cap._monitor
    assert mon is not None and mon.is_alive(), "start() must spawn a live monitor thread"

    still_alive = stop_capture_monitors_fn([cap])

    assert still_alive == [], "guard must join the monitor thread dead, not leak it"
    assert not mon.is_alive(), "monitor thread must be terminated after the guard runs"


def test_transform_failure_tracebacks_throttled_to_first_and_every_100th(caplog):
    """The per-frame transform except must not log a full traceback on every
    failure — PortAudio's realtime thread can fire ~33x/s, flooding the
    rotating file handler. Full tracebacks must fire on the 1st and every
    100th failure only.

    Seeded defect: revert the except block to unconditional _log.exception(...)
    -> 150 failures log 150 tracebacks -> red.
    """
    import logging

    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    cb = _start_captures_callback(cap, dict(_DEV_48K_MONO))

    with caplog.at_level(logging.DEBUG, logger="yamabiko.capture.audio"):
        # Drives past the death threshold (the callback returns
        # paComplete at #100 and real PortAudio would stop calling it);
        # frames 101-150 deliberately exercise the post-death state to
        # pin that the throttle holds there too (next full log at #200).
        for _ in range(150):
            cb(_BAD_FRAME, 1, None, 0)

    tracebacks = [r for r in caplog.records if r.exc_info]
    assert len(tracebacks) == 2, (
        f"expected full tracebacks on the 1st and 100th transform failure "
        f"only; got {len(tracebacks)} traceback log records over 150 failures"
    )


def test_persistent_transform_failure_signals_stream_death_and_stops_stream():
    """A callback that keeps raising but returns paContinue leaves
    stream.is_active()==True, so the stream-inactive monitor never fires and
    the pipeline stalls silently forever. After 100 CONSECUTIVE transform
    failures the capture must invoke on_stream_death (once — the pipeline
    wires it to graceful shutdown -> watchdog restart) and return paComplete
    so the dead-transform firehose stops at the source.

    Seeded defect: drop the consecutive-failure escalation from the
    except path -> deaths stays empty after 100+ failures -> red.
    """
    import pyaudiowpatch as pyaudio

    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)
    cb = _start_captures_callback(cap, dict(_DEV_48K_MONO))

    for _ in range(99):
        rc = cb(_BAD_FRAME, 1, None, 0)
        assert rc == (None, pyaudio.paContinue), (
            "callback must keep the stream alive below the death threshold"
        )
    assert deaths == [], "death signalled before the 100-failure threshold"

    rc = cb(_BAD_FRAME, 1, None, 0)  # 100th consecutive failure
    assert deaths == [True], (
        "on_stream_death not invoked after 100 consecutive transform "
        "failures; pipeline would stall invisibly"
    )
    assert rc == (None, pyaudio.paComplete), (
        "callback must stop the stream once it declared the transform dead"
    )

    cb(_BAD_FRAME, 1, None, 0)
    assert deaths == [True], "_signal_death must stay idempotent"


def test_successful_transform_resets_consecutive_failure_count():
    """Transient failure bursts must NOT accumulate to a stream death.
    99 failures, one good frame, 99 more failures — no death, because
    the success resets the consecutive counter (only the total counter
    keeps growing for the log throttle).

    Seeded defect: drop the consecutive-counter reset from the success
    path -> the second burst crosses 100 cumulative -> death fires ->
    red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)
    cb = _start_captures_callback(cap, dict(_DEV_48K_MONO))

    good = np.zeros(480, dtype=np.float32).tobytes()
    for _ in range(99):
        cb(_BAD_FRAME, 1, None, 0)
    cb(good, 480, None, 0)
    for _ in range(99):
        cb(_BAD_FRAME, 1, None, 0)

    assert deaths == [], (
        "two sub-threshold failure bursts separated by a successful "
        "frame must not add up to a stream death"
    )


def test_monitor_does_not_signal_death_on_deliberate_stop(caplog):
    """stop() makes the stream inactive BY DESIGN (stop_stream() runs before
    ``self._stream`` is cleared), so a monitor tick landing inside that window
    must not log the device-unplugged ERROR or fire on_stream_death during a
    perfectly deliberate shutdown.

    The interleaving is forced deterministically: the mock's is_active()
    blocks until the main thread has completed cap.stop() — emulating a
    monitor that captured the stream ref just before stop() nulled it and
    then observed the stopped stream.

    Seeded-defect drill: revert the monitor guard to ``if not active:`` ->
    _signal_death fires -> deaths == [True] -> red.
    """
    import logging
    import threading

    cap = LoopbackCapture(AudioConfig(), MagicMock(), MagicMock(spec=asyncio.Queue))
    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)

    monitor_inside_check = threading.Event()
    stop_completed = threading.Event()
    stream = MagicMock()

    def _is_active() -> bool:
        # Monitor passed its 1 s wait (stop not yet set) and grabbed the
        # stream ref. Hold it here while the main thread runs the
        # deliberate stop() to completion, then report the (now stopped)
        # stream as inactive — the exact F60 race window.
        monitor_inside_check.set()
        assert stop_completed.wait(timeout=5.0), "stop() never completed"
        return False

    stream.is_active.side_effect = _is_active
    cap._stream = stream

    cap._start_monitor()
    try:
        assert monitor_inside_check.wait(timeout=5.0), (
            "monitor never reached the is_active() check"
        )
        with caplog.at_level(logging.ERROR, logger="yamabiko.capture.audio"):
            cap.stop()  # deliberate: sets _stop, stops/closes the stream
            stop_completed.set()
            cap._monitor.join(timeout=5.0)
            assert not cap._monitor.is_alive(), (
                "monitor thread failed to exit after stop()"
            )
    finally:
        stop_completed.set()
        cap._stop.set()

    assert deaths == [], (
        "deliberate stop() was reported as a stream death"
    )
    spurious = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "dead" in r.getMessage()
    ]
    assert spurious == [], (
        f"deliberate stop() logged a spurious stream-death ERROR: "
        f"{[r.getMessage() for r in spurious]}"
    )


def test_monitor_signals_death_when_stream_goes_inactive():
    """An unexpectedly-inactive stream (device unplug) must be detected and
    surfaced via on_stream_death, not stall the pipeline silently forever.

    Seeded-defect drill: remove the is_active() check / the _start_monitor
    call -> death is never signalled -> deaths stays empty -> this fails.
    """
    import time as _t

    cap = LoopbackCapture(AudioConfig(), MagicMock(), MagicMock(spec=asyncio.Queue))
    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)
    stream = MagicMock()
    stream.is_active.return_value = False  # stream has died
    cap._stream = stream

    cap._start_monitor()
    try:
        deadline = _t.monotonic() + 5.0
        while _t.monotonic() < deadline and not deaths:
            _t.sleep(0.05)
    finally:
        cap._stop.set()  # end the monitor thread

    assert deaths == [True], (
        "monitor did not signal death on an inactive stream -- pipeline would "
        "stall silently"
    )


# --- A runtime switch of the OS DEFAULT OUTPUT device leaves the old WASAPI ---
# --- loopback is_active()==True but silent. The monitor must notice the     ---
# --- default-output change and signal death so the watchdog restarts.       ---

# The loopback device the capture stream is opened against. is_active()
# stays True throughout -- the dead-air condition is INVISIBLE to the
# existing is_active() probe, hence the separate default-output poll.
_LOOPBACK_SPEAKERS = {
    "index": 7, "name": "Speakers (Realtek) [Loopback]",
    "isLoopbackDevice": True, "defaultSampleRate": 48000,
    "maxInputChannels": 2,
}
# The OS default-render endpoint ID captured at start() (Windows Core Audio
# format -- what current_default_render_endpoint_id() returns). The monitor
# compares these opaque strings; nothing parses them.
_ENDPOINT_SPEAKERS = "{0.0.0.00000000}.{a1a1a1a1-0000-0000-0000-000000000001}"
# A DIFFERENT default-render endpoint after the user switches outputs.
_ENDPOINT_HEADPHONES = "{0.0.0.00000000}.{b2b2b2b2-0000-0000-0000-000000000002}"


def _live_stream() -> MagicMock:
    """A stream that always reports active — the silent-death state where
    is_active()==True but the device delivers no audio."""
    stream = MagicMock()
    stream.is_active.return_value = True
    return stream


def _drive_monitor_until(cap: LoopbackCapture, predicate, timeout: float = 3.0) -> None:
    """Start the real monitor thread and poll until predicate() or timeout."""
    import time as _t

    cap._start_monitor()
    try:
        deadline = _t.monotonic() + timeout
        while _t.monotonic() < deadline and not predicate():
            _t.sleep(0.02)
    finally:
        cap._stop.set()
        if cap._monitor is not None:
            cap._monitor.join(timeout=5.0)


def test_monitor_signals_death_when_default_output_device_changes():
    """When the OS default OUTPUT device switches at runtime, the old loopback
    stream stays is_active()==True but delivers no audio, so subtitles
    silently stop. The monitor polls the LIVE default-render endpoint ID
    (Windows Core Audio, via _read_default_output_id) and, on divergence from
    the ID captured at start(), signals death so the watchdog re-resolves the
    new default. The 1 Hz poll means detection lands ~2 s in.

    Seeded-defect drill: delete the default-output comparison from _watch()
    -> the switch is never noticed -> deaths stays empty -> this fails.
    (is_active() returns True the whole time, so the stream-inactive probe
    could never catch this.)
    """
    cap = LoopbackCapture(AudioConfig(device="default"),
                          MagicMock(), MagicMock(spec=asyncio.Queue))
    cap._stream = _live_stream()
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    # Baseline endpoint captured at start(); the live endpoint now differs.
    cap._default_output_id = _ENDPOINT_SPEAKERS
    cap._read_default_output_id = lambda: _ENDPOINT_HEADPHONES

    deaths: list[str | None] = []
    cap.on_stream_death = lambda: deaths.append(cap._last_death_reason)

    _drive_monitor_until(cap, lambda: bool(deaths), timeout=3.0)

    assert deaths, (
        "monitor did not signal death after the default OUTPUT device "
        "switched -- old loopback is silent but is_active()==True, so "
        "subtitles die with no error"
    )


def test_monitor_death_reason_names_the_default_output_change():
    """The death reason must distinguish a default-output switch from a device
    unplug / WASAPI reset, so the log explains why the pipeline bounced.
    Strengthens the previous test against a Liar that fires death for the
    wrong reason.
    """
    cap = LoopbackCapture(AudioConfig(device="default"),
                          MagicMock(), MagicMock(spec=asyncio.Queue))
    cap._stream = _live_stream()
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    cap._default_output_id = _ENDPOINT_SPEAKERS
    cap._read_default_output_id = lambda: _ENDPOINT_HEADPHONES

    reasons: list[str | None] = []
    cap.on_stream_death = lambda: reasons.append(cap._last_death_reason)

    _drive_monitor_until(cap, lambda: bool(reasons), timeout=3.0)

    assert reasons, "no death signalled on default-output switch"
    assert reasons[0] and "default output" in reasons[0].lower(), (
        f"death reason must name the default-output change; got {reasons[0]!r}"
    )


def test_monitor_does_not_signal_death_while_default_output_unchanged():
    """When the default OUTPUT device is stable, the poll must NOT fire a
    spurious death. Without this, a naive 'always differs' bug would bounce
    the pipeline once per second forever.

    Seeded-defect drill: make _watch() signal death unconditionally, or flip
    the `current != self._default_output_id` comparison to `==` -> deaths
    becomes non-empty -> this fails.
    """
    import time as _t

    cap = LoopbackCapture(AudioConfig(device="default"),
                          MagicMock(), MagicMock(spec=asyncio.Queue))
    cap._stream = _live_stream()
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    # Live endpoint reads the SAME id as the start() baseline, every poll.
    cap._default_output_id = _ENDPOINT_SPEAKERS
    cap._read_default_output_id = lambda: _ENDPOINT_SPEAKERS

    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)

    # Run the monitor long enough for >=2 poll ticks (1 Hz) to elapse.
    cap._start_monitor()
    try:
        _t.sleep(2.4)
    finally:
        cap._stop.set()
        cap._monitor.join(timeout=5.0)

    assert deaths == [], (
        "monitor signalled a spurious death while the default output "
        "device was unchanged"
    )


def test_monitor_ignores_default_output_when_capturing_a_named_device():
    """The default-output poll applies ONLY when capturing the system default
    device. A user who pinned a SPECIFIC loopback (config.device='Speakers')
    must not have their capture bounced just because the OS default OUTPUT
    moved elsewhere -- their explicit choice is independent of the system
    default.

    Seeded-defect drill: drop the `config.device == 'default'` gate ->
    a named-device capture fires death on any output switch -> red.
    """
    import time as _t

    cap = LoopbackCapture(AudioConfig(device="Speakers"),
                          MagicMock(), MagicMock(spec=asyncio.Queue))
    cap._stream = _live_stream()
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    # Deliberately populate a baseline that DIFFERS from the (changed) live
    # endpoint. If the device-gate is dropped, the body finds a divergence
    # and fires -- so the only thing keeping this green is the
    # `config.device == "default"` gate. (start() leaves this None for a
    # named device; we set it so the gate is the sole guard under test.)
    cap._default_output_id = _ENDPOINT_SPEAKERS
    cap._read_default_output_id = lambda: _ENDPOINT_HEADPHONES

    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)

    cap._start_monitor()
    try:
        _t.sleep(2.4)
    finally:
        cap._stop.set()
        cap._monitor.join(timeout=5.0)

    assert deaths == [], (
        "monitor bounced a pinned named-device capture on a default-output "
        "switch; the default-output poll must be gated on device=='default'"
    )


def test_monitor_treats_unreadable_default_output_as_no_change():
    """_read_default_output_id() returns None when the live endpoint can't be
    read (no output device, or a transient COM failure mid-switch). A None
    poll must be treated as 'can't tell' -- NOT as a divergence from the
    baseline -- so it never spuriously bounces the stream.

    Seeded-defect drill: drop the `current is not None` guard in _watch()
    -> `None != self._default_output_id` is True -> a spurious death fires
    on the first unreadable poll -> deaths becomes non-empty -> this fails.
    (Real-switch detection after the endpoint settles is covered by
    test_monitor_signals_death_when_default_output_device_changes.)
    """
    import time as _t

    cap = LoopbackCapture(AudioConfig(device="default"),
                          MagicMock(), MagicMock(spec=asyncio.Queue))
    cap._stream = _live_stream()
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    cap._default_output_id = _ENDPOINT_SPEAKERS
    cap._read_default_output_id = lambda: None  # unreadable every poll

    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)

    cap._start_monitor()
    try:
        _t.sleep(2.4)
    finally:
        cap._stop.set()
        cap._monitor.join(timeout=5.0)

    assert deaths == [], (
        "monitor signalled a spurious death when the live default-output "
        "endpoint was unreadable (None) -- None must be treated as no-change"
    )


def test_start_snapshots_default_output_endpoint_when_following_default():
    """start() coverage: the _watch() tests set _default_output_id by hand, so
    the actual snapshot inside start() -- `if config.device == "default":
    self._default_output_id = self._read_default_output_id()` -- would pass
    all the _watch() tests even if deleted. Drive the real start() and assert
    it captures the live default-render endpoint ID.

    Seeded-defect drill: delete the snapshot line in start() (or read
    _device_info instead) -> cap._default_output_id is None after start() ->
    red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    _run_start(cap, {
        "index": 0, "name": "X [Loopback]", "isLoopbackDevice": True,
        "defaultSampleRate": 48000, "maxInputChannels": 1,
    }, endpoint_id=_ENDPOINT_SPEAKERS)

    assert cap._default_output_id == _ENDPOINT_SPEAKERS, (
        f"start() did not snapshot the default-render endpoint ID for a "
        f"device='default' capture; the monitor has no baseline to compare "
        f"against and default-output detection silently never fires; "
        f"got {cap._default_output_id!r}"
    )


def test_start_does_not_snapshot_default_output_for_a_named_device():
    """A pinned named device must NOT snapshot the default output at start() --
    _default_output_id stays None, so _watch() never polls or bounces a
    capture whose device is independent of the system default. Pairs with the
    _watch() gate test; this pins the start() side.

    Seeded-defect drill: drop the `config.device == 'default'` gate in start()
    -> a named device snapshots an endpoint ID -> _default_output_id is not
    None -> red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="Speakers", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    _run_start(cap, {
        "index": 0, "name": "Speakers (Realtek) [Loopback]",
        "isLoopbackDevice": True, "defaultSampleRate": 48000,
        "maxInputChannels": 2,
    }, endpoint_id=_ENDPOINT_HEADPHONES)

    assert cap._default_output_id is None, (
        f"start() snapshotted a default-output endpoint for a PINNED named "
        f"device; the snapshot must be gated on config.device == 'default'; "
        f"got {cap._default_output_id!r}"
    )


# --- Idle-audio re-open (issue #54): an endpoint that produced audio then     ---
# --- goes silent (suspended / render session moved) must be re-attached by    ---
# --- re-opening the capture stream IN PLACE -- not a process restart. On       ---
# --- WASAPI loopback a silent endpoint delivers NO callbacks, so a benign     ---
# --- pause and a dead endpoint look identical; the re-open is harmless on a    ---
# --- pause and recovering on a real stall. Gated on _ever_audible so cold     ---
# --- start / never-played never fires (the false-positive the review caught). ---

def _idle_cap(device: str = "Speakers", idle_death_s: int = 10) -> LoopbackCapture:
    return LoopbackCapture(
        AudioConfig(device=device, idle_death_s=idle_death_s),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )


def test_idle_reopen_does_not_fire_before_audio_ever_played():
    """The P1 fix: a stream that has NEVER produced above-floor audio (cold
    start -- app launched before playback, or a source paused at launch) must
    NOT trigger a re-open. On WASAPI loopback a not-yet-playing endpoint
    delivers no callbacks, so without this gate the idle timer would run from
    the start() seed and fire on every healthy idle machine (the review's P1).

    Seeded defect: drop the `not self._ever_audible` gate in _check_idle_reopen
    -> a never-audible stream past the window fires -> red.
    """
    cap = _idle_cap(idle_death_s=10)
    cap._ever_audible = False
    cap._last_audible_monotonic = 1000.0 - 30.0  # 30 s "idle" from the seed
    assert cap._check_idle_reopen(1000.0) is None


def test_idle_reopen_fires_after_audio_played_then_went_silent():
    """A stream that produced audio (_ever_audible) and then had no above-floor
    audio for the window is a re-open candidate -- the endpoint may have
    suspended or its render session moved.

    Seeded defect: drop the `now - audible >= idle_death_s` check -> never fires
    -> red.
    """
    cap = _idle_cap(idle_death_s=10)
    cap._ever_audible = True
    cap._last_audible_monotonic = 1000.0 - 30.0
    reason = cap._check_idle_reopen(1000.0)
    assert reason is not None and "no audible audio" in reason.lower(), (
        f"a played-then-silent stream must be a re-open candidate; got {reason!r}"
    )


def test_idle_reopen_does_not_fire_while_audible_within_window():
    """Quiet-but-present audio (above the floor within the window) keeps the
    stream alive -- no re-open.

    Seeded defect: invert the `< idle_death_s` early return -> fires on live
    audio -> red.
    """
    cap = _idle_cap(idle_death_s=10)
    cap._ever_audible = True
    cap._last_audible_monotonic = 1000.0 - 1.0
    assert cap._check_idle_reopen(1000.0) is None


@pytest.mark.parametrize("device", ["default", "Speakers"])
def test_idle_reopen_not_gated_on_device_type(device):
    """The idle re-open runs for both pinned and default devices, unlike the
    default-output-switch poll.

    Seeded defect: add a `config.device == 'default'` gate to _check_idle_reopen
    -> one device type stops triggering -> red.
    """
    cap = _idle_cap(device=device, idle_death_s=10)
    cap._ever_audible = True
    cap._last_audible_monotonic = 1000.0 - 30.0
    assert cap._check_idle_reopen(1000.0) is not None


def test_idle_reopen_suppressed_and_reset_while_paused():
    """While Yamabiko is paused the check must not fire (re-open churn guard)
    AND must reset the audible window so resume starts a fresh countdown.

    Seeded defect 1: drop the pause guard -> fires while paused -> red.
    Seeded defect 2: drop the reset -> the stale idle survives the pause and
    fires on the first post-resume check -> the post-resume assert red.
    """
    cap = _idle_cap(idle_death_s=10)
    cap._ever_audible = True
    paused = [True]
    cap._is_paused = lambda: paused[0]
    cap._last_audible_monotonic = 1000.0 - 30.0
    assert cap._check_idle_reopen(1000.0) is None, "must not fire while paused"
    paused[0] = False
    assert cap._check_idle_reopen(1000.0) is None, (
        "post-resume must start a fresh window, not fire on the stale pre-pause idle"
    )


def test_callback_sets_ever_audible_only_on_above_floor_audio():
    """cb() must set the sticky _ever_audible (and the audible stamp) only when
    a frame's RMS clears the floor; a silent (zeros) frame advances only the
    frame stamp.

    Seeded defect: make the _ever_audible write unconditional -> a silent frame
    sets it -> the 'still False after silence' assert red.
    """
    loop = MagicMock()
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        loop, MagicMock(spec=asyncio.Queue),
    )
    cb = _start_captures_callback(cap, dict(_DEV_48K_MONO))
    cap._ever_audible = False
    cb(np.zeros(480, dtype=np.float32).tobytes(), 480, None, 0)  # silence
    assert cap._ever_audible is False, "a silent frame must not set _ever_audible"
    cb((np.ones(480, dtype=np.float32) * 0.1).tobytes(), 480, None, 0)  # RMS 0.1
    assert cap._ever_audible is True, "an above-floor frame must set _ever_audible"


def test_callback_treats_non_finite_rms_as_silence():
    """A frame with a non-finite sample (inf) must NOT count as audio: inf > floor
    is True, so without an isfinite guard an inf frame would set _ever_audible
    and refresh the audible stamp, masking a real stall.

    Seeded defect: drop the np.isfinite(rms) guard -> the inf frame sets
    _ever_audible -> red.
    """
    loop = MagicMock()
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        loop, MagicMock(spec=asyncio.Queue),
    )
    cb = _start_captures_callback(cap, dict(_DEV_48K_MONO))
    cap._ever_audible = False
    cb(np.full(480, np.inf, dtype=np.float32).tobytes(), 480, None, 0)
    assert cap._ever_audible is False, (
        "a non-finite (inf) frame must count as silence, not audio"
    )


def test_start_seeds_idle_window_and_clears_ever_audible():
    """start() -> _open_stream() must seed the audible/frame stamps (the window
    begins at stream-open) and clear _ever_audible (a fresh stream hasn't proven
    audio yet, so the idle re-open won't fire until audio plays and then stops).

    Seeded defect: drop the seed / _ever_audible reset in _open_stream -> stamps
    None or _ever_audible stale after a fresh start -> red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="default", sample_rate=48000),
        MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    cap._ever_audible = True  # pretend a prior stream had audio
    _run_start(cap, dict(_DEV_48K_MONO))
    assert (
        cap._last_audible_monotonic is not None
        and cap._last_frame_monotonic is not None
    ), "start() must seed both idle stamps"
    assert cap._ever_audible is False, "a fresh stream must clear _ever_audible"


def test_reopen_stream_tears_down_and_rebuilds_via_open_stream():
    """_reopen_stream must close the old stream + PyAudio, create a fresh
    PyAudio (pyaudiowpatch can't refresh its frozen device list in place),
    re-resolve, and re-open via _open_stream -- all against the same out-queue.

    Seeded defect: drop the self._open_stream() call -> no fresh stream after a
    re-open -> opened stays empty -> red.
    """
    cap = _idle_cap()
    old_stream = MagicMock()
    cap._stream = old_stream
    old_pa = MagicMock()
    cap._pa = old_pa
    cap._resolve_device = lambda: dict(_LOOPBACK_SPEAKERS)
    opened = []
    cap._open_stream = lambda: opened.append(True)
    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=MagicMock()):
        cap._reopen_stream()
    old_stream.close.assert_called_once()
    old_pa.terminate.assert_called_once()
    assert opened == [True], "_reopen_stream did not re-open via _open_stream"


def test_reopen_stream_is_a_noop_when_stopping():
    """A re-open racing a shutdown (stop() already set _stop) must NOT rebuild --
    it would resurrect a stream the teardown is tearing down.

    Seeded defect: drop the `if self._stop.is_set(): return` guard -> _open_stream
    runs during shutdown -> opened non-empty -> red.
    """
    cap = _idle_cap()
    cap._stream = MagicMock()
    cap._pa = MagicMock()
    cap._stop.set()
    opened = []
    cap._open_stream = lambda: opened.append(True)
    cap._reopen_stream()
    assert opened == [], "_reopen_stream rebuilt the stream during shutdown"


def test_monitor_reopens_stream_in_place_on_idle_not_process_death():
    """Integration / wiring: when a live stream goes idle (audio played then
    stopped for the window), the monitor must RE-OPEN the stream in place --
    NOT fire on_stream_death (a process restart). A re-open is harmless on a
    benign pause; a process restart would loop the watchdog on every pause
    (the P1 the review caught).

    Seeded defect: have _watch call _signal_death instead of _reopen_stream on
    the idle branch -> on_stream_death fires (deaths non-empty) -> red.
    """
    import time as _t

    cap = _idle_cap(idle_death_s=1)
    cap._stream = _live_stream()              # is_active()==True throughout
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    cap._monitor_poll_s = 0.02
    cap._idle_death_s = 0.05
    cap._ever_audible = True                  # audio DID play
    cap._last_audible_monotonic = _t.monotonic() - 1.0  # then silent ~1 s
    reopens: list[bool] = []
    cap._reopen_stream = lambda: reopens.append(True)   # stub the rebuild
    deaths: list[bool] = []
    cap.on_stream_death = lambda: deaths.append(True)

    _drive_monitor_until(cap, lambda: bool(reopens), timeout=2.0)

    assert reopens, "the monitor did not re-open the idle stream in place"
    assert deaths == [], (
        "an idle (played-then-silent) stream must trigger an in-place re-open, "
        "NOT a process restart (on_stream_death) -- a restart would loop the "
        "watchdog on every pause"
    )


def test_monitor_escalates_to_signal_death_when_reopen_fails():
    """If the in-place re-open fails (device genuinely gone -> resolve raises),
    the monitor must escalate to on_stream_death (a full restart that
    re-resolves), not swallow it.

    Seeded defect: drop the except -> _signal_death in _watch's idle branch ->
    a device-gone re-open failure is swallowed -> deaths stays empty -> red.
    """
    import time as _t

    cap = _idle_cap(idle_death_s=1)
    cap._stream = _live_stream()
    cap._device_info = dict(_LOOPBACK_SPEAKERS)
    cap._monitor_poll_s = 0.02
    cap._idle_death_s = 0.05
    cap._ever_audible = True
    cap._last_audible_monotonic = _t.monotonic() - 1.0

    def _boom():
        raise RuntimeError("Loopback device not found: 'Speakers'")

    cap._reopen_stream = _boom
    deaths: list[str | None] = []
    cap.on_stream_death = lambda: deaths.append(cap._last_death_reason)

    _drive_monitor_until(cap, lambda: bool(deaths), timeout=2.0)

    assert deaths and "re-open failed" in (deaths[0] or ""), (
        f"a failed in-place re-open must escalate to on_stream_death; got {deaths!r}"
    )


# --- Device-gone grace (issue #54 recovery): after a stream death the      ---
# --- watchdog restarts; the fresh start() may race a device that is        ---
# --- mid-reconnect (unplug/replug, or a suspended endpoint settling). A    ---
# --- short bounded grace lets it reappear instead of failing straight to   ---
# --- the no-audio dialog. pyaudiowpatch freezes its device list at         ---
# --- PyAudio() init, so each retry re-creates the instance.                ---

_DEV_SPEAKERS_LB = {
    "index": 1, "name": "Speakers (Realtek) [Loopback]",
    "isLoopbackDevice": True, "defaultSampleRate": 48000, "maxInputChannels": 2,
}


def test_resolve_device_grace_retries_until_device_reappears():
    """A device that is absent on the first resolve but reappears within the
    grace window must be picked up (re-creating PyAudio each attempt, since
    pyaudiowpatch can't refresh its frozen device list in place).

    Seeded defect: drop the retry loop (resolve once) -> the first
    RuntimeError propagates before the device reappears -> red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="Speakers"), MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    cap._device_grace_poll_s = 0.005
    cap._device_grace_attempts = 5

    calls = {"n": 0}

    def _resolve():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Loopback device not found: 'Speakers'")
        return dict(_DEV_SPEAKERS_LB)

    cap._resolve_device = _resolve
    cap._pa = MagicMock()
    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=MagicMock()):
        info = cap._resolve_device_with_grace()

    assert info["index"] == 1, "grace retry did not return the reappeared device"
    assert calls["n"] == 3, (
        f"expected two failed resolves then success; got {calls['n']} attempts"
    )


def test_resolve_device_grace_gives_up_after_bounded_attempts():
    """A genuinely-absent device must still surface promptly (re-raise the
    RuntimeError -> the no-audio dialog) after the bounded grace, never loop
    forever.

    Seeded defect: make the grace loop unbounded (while True) -> the
    always-failing resolve never returns -> the test hangs (timeout) -> red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="Gone"), MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    cap._device_grace_poll_s = 0.005
    cap._device_grace_attempts = 3

    def _resolve():
        raise RuntimeError("Loopback device not found: 'Gone'")

    cap._resolve_device = _resolve
    cap._pa = MagicMock()
    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=MagicMock()):
        with pytest.raises(RuntimeError, match="not found"):
            cap._resolve_device_with_grace()


def test_resolve_device_grace_aborts_on_stop():
    """If the pipeline is shutting down (stop set) during the grace window, the
    retry must abort promptly instead of burning the full attempt budget.

    Seeded defect: drop the `self._stop.wait(...)`-is-set abort -> a shutdown
    mid-grace burns the full attempt budget -> _resolve_device call count is
    51, not 1 -> red.
    """
    cap = LoopbackCapture(
        AudioConfig(device="Gone"), MagicMock(), MagicMock(spec=asyncio.Queue),
    )
    cap._device_grace_poll_s = 0.005
    cap._device_grace_attempts = 50  # would be a long loop if stop is ignored

    calls = {"n": 0}

    def _resolve():
        calls["n"] += 1
        raise RuntimeError("Loopback device not found: 'Gone'")

    cap._resolve_device = _resolve
    cap._pa = MagicMock()
    cap._stop.set()  # deliberate shutdown in progress
    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=MagicMock()):
        with pytest.raises(RuntimeError, match="not found"):
            cap._resolve_device_with_grace()

    assert calls["n"] == 1, (
        f"a shutdown mid-grace must abort immediately after the first resolve; "
        f"_resolve_device was called {calls['n']}x (the grace ignored _stop)"
    )


# --- _stream_lock serialization (#54): the monitor's in-place re-open and    ---
# --- stop() both tear down the PortAudio stream + instance; the lock must     ---
# --- keep them from overlapping (a concurrent double-close/terminate of the   ---
# --- C handles is a use-after-free). These pin the temporal contract that     ---
# --- removing `with self._stream_lock:` from either site would silently break.---

def test_stop_blocks_while_stream_lock_held():
    """stop() must not tear down the PortAudio handles while _stream_lock is
    held by an in-flight re-open: hold the lock and show stop() blocks (does not
    terminate _pa) until it's released.

    Seeded defect: remove `with self._stream_lock:` from stop() -> stop()
    terminates _pa while the lock is held (no longer blocks) -> red.
    """
    import threading

    cap = _idle_cap()
    cap._pa = MagicMock()
    cap._stream = MagicMock()
    done = threading.Event()

    cap._stream_lock.acquire()
    try:
        t = threading.Thread(target=lambda: (cap.stop(), done.set()))
        t.start()
        # stop() sets _stop (outside the lock) then must BLOCK on _stream_lock
        # before terminating _pa.
        assert not done.wait(0.3), (
            "stop() tore down the PortAudio handles while _stream_lock was held "
            "by a (simulated) re-open -- the two can double-free the C handles"
        )
        assert cap._pa is not None, "stop() terminated _pa while the lock was held"
    finally:
        cap._stream_lock.release()

    assert done.wait(2.0), "stop() never completed after the lock released"
    t.join(2.0)
    assert cap._pa is None, "stop() did not tear down after the lock released"


def test_reopen_stream_blocks_while_stream_lock_held():
    """_reopen_stream must not rebuild the stream while _stream_lock is held by
    stop() (or another re-open): hold the lock and show it blocks (does not
    reach _open_stream) until released.

    Seeded defect: remove `with self._stream_lock:` from _reopen_stream -> it
    rebuilds while the lock is held (no longer blocks) -> red.
    """
    import threading

    cap = _idle_cap()
    cap._pa = MagicMock()
    cap._stream = MagicMock()
    cap._resolve_device = lambda: dict(_LOOPBACK_SPEAKERS)
    opened = threading.Event()
    cap._open_stream = lambda: opened.set()
    done = threading.Event()

    with patch("yamabiko.capture.audio.pyaudio.PyAudio", return_value=MagicMock()):
        cap._stream_lock.acquire()
        try:
            t = threading.Thread(target=lambda: (cap._reopen_stream(), done.set()))
            t.start()
            assert not done.wait(0.3), (
                "_reopen_stream rebuilt the stream while _stream_lock was held"
            )
            assert not opened.is_set(), (
                "_open_stream ran while the lock was held by another teardown"
            )
        finally:
            cap._stream_lock.release()
        assert done.wait(2.0), "_reopen_stream never completed after release"
        t.join(2.0)
    assert opened.is_set(), "_reopen_stream did not rebuild after the lock released"
