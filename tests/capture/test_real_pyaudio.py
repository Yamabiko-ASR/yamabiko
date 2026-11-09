"""Real-path checks for the default-output-device-switch detector.

When the OS default OUTPUT device switches at runtime, the old WASAPI loopback
stream stays ``is_active() == True`` but yields only silence, so subtitles
silently stop. ``LoopbackCapture._watch`` polls the LIVE default-render
endpoint ID once a second (Windows Core Audio, via
``current_default_render_endpoint_id`` / ``_read_default_output_id``) and
signals stream death on divergence from the ID captured at ``start()``.

Why Core Audio and not PyAudio: PortAudio freezes its default device at
``Pa_Initialize()`` and pyaudiowpatch ships no ``Pa_RefreshDeviceList`` -- so
``get_default_output_device_info()`` would return a value cached at startup and
NEVER notice a runtime switch. This file exercises the REAL Core Audio query
(``IMMDeviceEnumerator::GetDefaultAudioEndpoint`` -> ``GetId``) on real
hardware -- the GUIDs, vtable indices, and string marshalling must hold against
the real OS, not just a mock.

The unit suite (``tests/capture/test_audio.py``) fakes the seam to exercise the
switch-detection LOGIC. This file is the real-path companion for the external
Core Audio call site, verifying that the query plumbing resolves the current
default endpoint and that an idle system reads back a stable ID.

NOTE: full switch DETECTION still requires a hardware device-switch (we do not
mutate the operator's default device in a test). What IS verified here: the
query plumbing and the false-positive guard.
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest

from yamabiko.capture import default_endpoint
from yamabiko.capture.audio import AudioConfig, LoopbackCapture
from yamabiko.capture.default_endpoint import (
    active_render_endpoint_names,
    current_default_render_endpoint_id,
)
from yamabiko.capture.envelope import CaptureBlock

pytestmark = pytest.mark.real


def _host_has_default_output() -> bool:
    """True iff this host has a working audio subsystem with a default output.

    Used to tell a legitimate environment skip (headless CI, no audio) apart
    from a genuine Core Audio failure: on a host that HAS audio, a None
    endpoint ID is a BUG, not a skip.
    """
    pyaudiowpatch = pytest.importorskip("pyaudiowpatch")
    try:
        pa = pyaudiowpatch.PyAudio()
    except OSError:
        return False
    try:
        try:
            pa.get_default_output_device_info()
            return True
        except Exception:
            return False
    finally:
        pa.terminate()


def test_real_core_audio_resolves_a_default_render_endpoint_id():
    """The Core Audio query must return the current default render endpoint's ID
    on real hardware. Pins the real COM path end to end:
    CoCreateInstance(MMDeviceEnumerator) -> GetDefaultAudioEndpoint(eRender,
    eConsole) -> IMMDevice::GetId -> a Windows endpoint-ID string.

    Crucially this is NOT skip-on-None: we first confirm (via pyaudiowpatch)
    that the host HAS a default output, so a None result here means the COM
    query itself is broken (wrong GUID / vtable slot / marshalling) and
    default-output detection cannot work -- the test goes red, it does not skip.

    Seeded-defect drill: corrupt _GET_DEFAULT_AUDIO_ENDPOINT's vtable index,
    or the CLSID/IID, in default_endpoint.py -> the query returns None on a
    host that has audio -> this fails.
    """
    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")

    eid = current_default_render_endpoint_id()

    assert isinstance(eid, str) and eid, (
        "current_default_render_endpoint_id() returned no endpoint ID on a "
        "host that HAS a default output device -- the Core Audio COM query is "
        "broken (GUIDs / vtable indices / string marshalling) and "
        "default-output switch-detection cannot work"
    )
    assert eid.startswith("{") and "}." in eid, (
        f"endpoint ID is not in the Windows MMDevice format ('{{...}}.{{...}}'): "
        f"{eid!r}"
    )


def test_real_default_render_endpoint_stable_across_reads():
    """The detector declares a switch only when the endpoint ID DIVERGES. On a
    steady system the real default-render endpoint must read back the SAME ID
    twice in a row, or the detector would bounce the pipeline every second on a
    machine no one touched. Drives the production seam
    (``LoopbackCapture._read_default_output_id``) against real Core Audio --
    the subject is the real OS query, not a mock.

    Seeded-defect drill: if _read_default_output_id keyed off a volatile field
    or the query returned fresh garbage each call, the two reads would differ
    and this goes red.
    """
    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")

    cap = LoopbackCapture(AudioConfig(device="default"), object(), object())
    first = cap._read_default_output_id()
    if first is None:
        pytest.skip("no default render endpoint resolvable on this host")
    second = cap._read_default_output_id()

    assert second == first, (
        "the real default-render endpoint ID changed between two back-to-back "
        "reads on an idle system -- F59 would bounce the pipeline once a second "
        "(false positive)"
    )


def test_real_active_render_endpoint_names_match_the_loopback_devices():
    """Lead decision 2: a fresh Core Audio enumeration (EnumAudioEndpoints
    eRender/ACTIVE + PKEY_Device_FriendlyName) lists every endpoint PyAudio
    offers as a loopback, under the loopback name minus " [Loopback]". That
    equality is what lets a dead stream ask "is my endpoint still there?".

    Seeded defects: a wrong vtable slot, the wrong PROPERTYKEY pid, or a
    PROPVARIANT that reads the wrong field -> None or garbage names -> red.
    """
    pyaudiowpatch = pytest.importorskip("pyaudiowpatch")
    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")
    pa = pyaudiowpatch.PyAudio()
    try:
        loopbacks = [d["name"] for d in pa.get_loopback_device_info_generator()]
    finally:
        pa.terminate()
    names = active_render_endpoint_names()
    assert names is not None, "the Core Audio enumeration failed on a host with audio"
    assert loopbacks, "PyAudio offers no loopback device on a host with audio"
    for loopback in loopbacks:
        assert loopback.removesuffix(" [Loopback]") in names, (loopback, names)


def test_real_active_render_endpoint_names_is_none_when_one_name_is_unreadable(
    monkeypatch,
):
    """A list missing one endpoint would report a present device as
    disconnected, so one unreadable friendly name makes the whole answer
    "can't tell" (None), which the pipeline treats as a silent restart.
    The enumeration is real; only device 0's name read fails.

    Seeded defect: skip the unreadable device instead of returning None ->
    a (shorter) list -> red.
    """
    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")
    assert active_render_endpoint_names(), "the real enumeration found no endpoint"
    read_name = default_endpoint._friendly_name
    monkeypatch.setattr(
        default_endpoint, "_friendly_name",
        lambda collection, index, ole32: (
            None if index == 0 else read_name(collection, index, ole32)
        ),
    )
    assert active_render_endpoint_names() is None



def test_real_capture_start_resolves_and_seeds_idle_timestamps():
    """A real LoopbackCapture.start() must resolve the default loopback through
    the new _resolve_device_with_grace path AND seed the issue-#54 idle-
    heartbeat timestamps -- on the REAL pyaudiowpatch capture stack, not a mock
    (hard rule #3: the grace wrapper + cb timestamp writes are new interactions
    with the external audio dep).

    The seed assertion is robust on any host with audio. The stronger
    "callback advances the frame stamp" check is OPPORTUNISTIC: WASAPI loopback
    on a silent, event-driven endpoint delivers no callbacks at all (exactly
    the frame-stall the #54 frame-arrival gate exists to catch), so we only
    assert advancement IF frames actually arrived. When they did, the loop
    runs briefly and the out-queue must hold converted float32 CaptureBlocks:
    real frames that never reach the queue are lost speech. The cb advancement
    logic itself is unit-tested against driven frames in
    tests/capture/test_audio.py.

    Seeded-defect drill: delete the seed lines in start() -> both timestamps
    are None after a real start() -> red.
    """
    import asyncio
    import time

    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        out_queue: asyncio.Queue = asyncio.Queue()
        cap = LoopbackCapture(AudioConfig(device="default"), loop, out_queue)
        cap._device_grace_attempts = 0  # no-device hosts skip fast, no 6 s grace
        try:
            cap.start()
        except RuntimeError:
            pytest.skip("no WASAPI loopback resolvable on this host")
        try:
            seeded = cap._last_frame_monotonic
            assert seeded is not None and cap._last_audible_monotonic is not None, (
                "real start() must seed both #54 idle-heartbeat timestamps so "
                "the window begins at stream-open and the first monitor tick "
                "never reads a None timestamp"
            )
            # Opportunistic: if real frames arrive (audio playing, or a driver
            # that emits silence frames), the callback must move the frame stamp
            # strictly forward -- never backward.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and cap._last_frame_monotonic == seeded:
                time.sleep(0.02)
            if cap._last_frame_monotonic != seeded:
                assert cap._last_frame_monotonic > seeded, (
                    "the callback moved the frame-liveness timestamp backwards"
                )
                loop.run_until_complete(asyncio.sleep(0.5))
                queued = [out_queue.get_nowait() for _ in range(out_queue.qsize())]
                assert any(
                    isinstance(item, CaptureBlock)
                    and item.pcm.dtype == np.float32
                    and item.pcm.size > 0
                    for item in queued
                ), (
                    "real frames arrived but no non-empty float32 CaptureBlock "
                    "reached the out-queue (queued types: "
                    f"{[type(item).__name__ for item in queued]!r})"
                )
        finally:
            cap.stop()
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_real_silent_capture_does_not_fire_death_or_reopen():
    """The review's P1 guard, on real hardware: a capture against a SILENT
    default endpoint (nothing playing -- the common idle state) must NOT fire
    on_stream_death OR re-open the stream within the idle window. On WASAPI
    loopback a silent endpoint delivers no callbacks, so _ever_audible stays
    False and the idle re-open is correctly gated off; a healthy idle machine
    must never be treated as a stalled device (which would restart-loop the
    watchdog). Subject is the real pyaudiowpatch capture stack (hard rule #3).

    Requires no audio playing (the idle state); skips if the device can't be
    resolved. Re-open is stubbed so a fire is observed without rebuilding.

    Seeded-defect drill: drop the `not self._ever_audible` gate in
    _check_idle_reopen -> the silent endpoint's idle timer runs from the
    start() seed and trips within idle_death_s -> reopens non-empty -> red.
    """
    import asyncio
    import time

    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        cap = LoopbackCapture(
            AudioConfig(device="default", idle_death_s=1), loop, asyncio.Queue()
        )
        cap._device_grace_attempts = 0
        cap._monitor_poll_s = 0.05
        deaths: list[bool] = []
        reopens: list[bool] = []
        cap.on_stream_death = lambda: deaths.append(True)
        cap._reopen_stream = lambda: reopens.append(True)
        try:
            cap.start()
        except RuntimeError:
            pytest.skip("no WASAPI loopback resolvable on this host")
        try:
            time.sleep(1.5)  # > idle_death_s, many monitor ticks
            assert deaths == [], (
                "a silent (idle) real endpoint fired a process death -- a "
                "healthy idle machine would restart-loop the watchdog"
            )
            assert reopens == [], (
                "a silent (never-played) real endpoint triggered a re-open; the "
                "_ever_audible gate must suppress it until audio actually plays"
            )
        finally:
            cap.stop()
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# Child script for the re-open test below: hold an ACTIVE render session on the
# default output by playing pure silence (zero float32 samples -- inaudible).
# WASAPI loopback delivers NO callbacks while the endpoint has no active render
# session, so without this the capture under test would see no frames at all.
# It runs as a SUBPROCESS, not an in-process output stream, deliberately: a
# second in-process PyAudio would hold PortAudio's Pa_Initialize refcount above
# zero across _reopen_stream's terminate(), turning the full
# Pa_Terminate -> Pa_Initialize cycle under test into a mere refcount
# decrement/increment and letting the "fresh" instance inherit the old frozen
# device list -- exactly NOT the production condition the re-open relies on.
_SILENCE_WRITER_SRC = """\
import sys
import time

import pyaudiowpatch as pyaudio

pa = pyaudio.PyAudio()
dev = pa.get_default_output_device_info()
rate = int(dev.get("defaultSampleRate") or 48000)
channels = 2 if int(dev.get("maxOutputChannels") or 2) >= 2 else 1
chunk = max(256, rate // 50)  # ~20 ms blocks
stream = pa.open(
    format=pyaudio.paFloat32,
    channels=channels,
    rate=rate,
    output=True,
    output_device_index=int(dev["index"]),
    frames_per_buffer=chunk,
)
silence = bytes(chunk * channels * 4)  # zeroed paFloat32 frames == inaudible
deadline = time.monotonic() + float(sys.argv[1])
while time.monotonic() < deadline:
    stream.write(silence)  # blocking write paces itself at realtime
"""


def _frame_stamp_advances(cap: LoopbackCapture, timeout_s: float) -> bool:
    """True iff the capture's REAL PortAudio callback stamps a new frame
    (advances cap._last_frame_monotonic past its current value) within
    timeout_s.

    _last_frame_monotonic is the callback-written liveness stamp that
    capture/audio.py explicitly reserves for real-tier frame-flow observation
    (it advances on every enqueue, before the asyncio bridge, so it works
    without a running event loop -- same technique as
    test_real_capture_start_resolves_and_seeds_idle_timestamps above).
    """
    baseline = cap._last_frame_monotonic
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cap._last_frame_monotonic != baseline:
            return True  # stamps are time.monotonic() -> any change is forward
        time.sleep(0.02)
    return False


def test_real_reopen_stream_reinitializes_pyaudio_and_frames_resume():
    """The issue-#54 in-place recovery, on the REAL pyaudiowpatch stack: with a
    live capture running, _reopen_stream() must stop+close the old stream,
    terminate the old PyAudio, re-init a FRESH PyAudio in the same process,
    re-resolve the default loopback, re-open -- and real frames must RESUME
    through the new stream's callback. Every one of those is an external-dep
    call site (hard rule #3) that the mocked tier
    (tests/capture/test_audio.py::test_reopen_stream_tears_down_and_rebuilds_
    via_open_stream) exercises only against MagicMock; if the
    terminate-then-reinit sequence breaks against the real library, recovery
    ships broken and every quiet stretch becomes a watchdog restart loop.

    Calling _reopen_stream() directly is the monitor thread's exact entry
    point into recovery, minus the idle heuristics (whose not-called side
    test_real_silent_capture_does_not_fire_death_or_reopen covers above). The
    monitor is quiesced via the _monitor_poll_s seam because in production the
    CALLER of _reopen_stream IS the monitor thread, so it never probes
    is_active() concurrently with the rebuild; a direct call from the test
    thread must not introduce a probe-during-teardown race the production
    path cannot have.

    Skips only on environment limits (no audio subsystem, no loopback, or no
    frames observable BEFORE the subject runs); once frames are proven
    flowing, every recovery failure is a red, never a skip.

    Seeded-defect drill: delete the `self._open_stream()` call at the end of
    _reopen_stream (audio.py:509) -> no new stream is built -> the new-stream
    assertion and the frames-resume assertion go red. Likewise deleting the
    `self._pa = pyaudio.PyAudio()` re-init (audio.py:507) leaves _pa None ->
    _resolve_device's `assert pa is not None` raises out of the direct call ->
    red.
    """
    import asyncio

    import pyaudiowpatch

    if not _host_has_default_output():
        pytest.skip("no default output device on this host (no audio subsystem)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    writer: subprocess.Popen | None = None
    try:
        writer = subprocess.Popen(
            [sys.executable, "-c", _SILENCE_WRITER_SRC, "60"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        cap = LoopbackCapture(AudioConfig(device="default"), loop, asyncio.Queue())
        cap._device_grace_attempts = 0  # no-device hosts skip fast, no 6 s grace
        cap._monitor_poll_s = 3600.0  # quiesce the monitor; see docstring
        deaths: list[str | None] = []
        cap.on_stream_death = lambda: deaths.append(cap._last_death_reason)
        try:
            cap.start()
        except RuntimeError:
            pytest.skip("no WASAPI loopback resolvable on this host")
        try:
            # Precondition, not subject: frames must be observable at all on
            # this host before the re-open's outcome can be judged.
            if not _frame_stamp_advances(cap, 8.0):
                pytest.skip(
                    "default loopback delivered no frames even with an active "
                    f"silence render session (writer exit: {writer.poll()!r}); "
                    "frame flow is not observable on this host"
                )
            old_pa, old_stream = cap._pa, cap._stream

            # THE SUBJECT: issue-#54 in-place recovery against the real
            # library. Must not raise -- the monitor escalates a raise to
            # _signal_death, i.e. a full restart on every quiet stretch.
            cap._reopen_stream()

            assert cap._pa is not old_pa and isinstance(
                cap._pa, pyaudiowpatch.PyAudio
            ), (
                "_reopen_stream must terminate the old PyAudio and re-init a "
                "FRESH real instance (pyaudiowpatch freezes its device list "
                "at Pa_Initialize, so reusing the old instance can never "
                "re-resolve a changed device)"
            )
            assert cap._stream is not None and cap._stream is not old_stream, (
                "_reopen_stream did not build a new stream over the "
                "re-resolved device"
            )
            assert cap._stream.is_active(), (
                "the re-opened stream is not active -- the rebuilt capture "
                "would deliver no frames"
            )
            # Decisive: real frames RESUME through the NEW stream's callback
            # (the silence render session is still active, so frames MUST
            # arrive if the rebuilt stream actually captures).
            assert _frame_stamp_advances(cap, 8.0), (
                "no frames arrived after _reopen_stream() although the "
                "silence render session is still active -- the in-place "
                "recovery rebuilt a stream that does not capture"
            )
            assert deaths == [], (
                f"in-place recovery fired on_stream_death ({deaths!r}) -- "
                "recovery must not tear the pipeline down"
            )
        finally:
            cap.stop()
    finally:
        if writer is not None:
            try:
                writer.terminate()
                writer.wait(timeout=5)
            except Exception:
                pass
        asyncio.set_event_loop(None)
        loop.close()
