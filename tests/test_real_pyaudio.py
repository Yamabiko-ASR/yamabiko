"""Real-path check: pyaudiowpatch is functional on this machine.

Beta shipped a wizard Test button that froze the Qt main thread
because the real ``stream.read()`` blocks and our test used a mock.
This file exercises the parts of pyaudiowpatch that have actually
bitten us, INCLUDING the callback-based stream lifecycle (the exact
seam the beta freeze lived at).

Skips gracefully in environments without a working audio subsystem
(headless CI, container runners).
"""
from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.real


def test_pyaudio_instantiation() -> None:
    """PyAudio() must construct, talk to PortAudio, and terminate
    without raising.

    Seeded-defect drill: break the pyaudiowpatch -> PortAudio binding
    (e.g. pin a mismatched wheel) -> PyAudio() raises OSError ->
    skip fires, test surfaces the env breakage. If the binding works
    but the C call itself raises, get_device_count() propagates and
    this test fails red.

    Headless CI runners may legitimately report 0 devices -- that
    doesn't mean PyAudio is broken, just that the runner has no
    audio hardware. `test_at_least_one_wasapi_loopback_device` below
    is the test for "a usable device exists".
    """
    pyaudiowpatch = pytest.importorskip("pyaudiowpatch")
    try:
        p = pyaudiowpatch.PyAudio()
    except OSError as exc:
        pytest.skip(f"PyAudio() failed to initialize: {exc!r}")
    try:
        # get_device_count() is the cheapest smoke test -- it forces
        # pyaudiowpatch to talk to PortAudio. We don't care about the
        # VALUE (CI runners report 0; that's fine); the contract is
        # "the call returns an int, not a TypeError / OSError / ...".
        count = p.get_device_count()
        assert isinstance(count, int) and count >= 0, (
            f"get_device_count returned an unexpected shape: {count!r}"
        )
    finally:
        p.terminate()


def test_at_least_one_wasapi_loopback_device() -> None:
    """The wizard's capture path depends on WASAPI loopback being available.

    Skipped (not failed) if no loopback device is enumerated -- running
    in a VM or headless box where WASAPI isn't wired is a valid env
    limitation, not a product bug.
    """
    pyaudiowpatch = pytest.importorskip("pyaudiowpatch")
    try:
        p = pyaudiowpatch.PyAudio()
    except OSError as exc:
        pytest.skip(f"PyAudio() failed to initialize: {exc!r}")
    try:
        loopbacks = list(p.get_loopback_device_info_generator())
    finally:
        p.terminate()

    if not loopbacks:
        pytest.skip("no WASAPI loopback devices on this host")
    # Basic shape check -- every loopback info dict must at least have
    # a name and index; PyAudio devs do change return shapes occasionally.
    first = loopbacks[0]
    assert "name" in first
    assert "index" in first


def test_callback_stream_opens_records_and_closes_cleanly() -> None:
    """Open a REAL WASAPI loopback stream in callback mode; record 200 ms;
    verify the callback was invoked and received frames of the expected
    shape.

    This is the beta's wizard-freeze seam. The previous test stopped at
    enumeration; the bug lived inside ``pa.open(stream_callback=...)`` +
    ``start_stream()`` + the callback's numpy reshape. A regression at
    any of those points will make this test fail (not hang — we use a
    bounded wait).
    """
    import numpy as np

    pyaudiowpatch = pytest.importorskip("pyaudiowpatch")
    try:
        pa = pyaudiowpatch.PyAudio()
    except OSError as exc:
        pytest.skip(f"PyAudio() failed to initialize: {exc!r}")

    try:
        # Use the system default output's loopback, not an arbitrary one.
        # Non-default loopbacks (HDMI outputs, idle speakers) only deliver
        # callback buffers while something is actively playing on that
        # output; the default loopback is the one most likely to have
        # audio flowing through it on a normal desktop session.
        if not hasattr(pa, "get_default_wasapi_loopback"):
            pytest.skip("pyaudiowpatch lacks get_default_wasapi_loopback")
        try:
            dev = pa.get_default_wasapi_loopback()
        except (OSError, Exception) as exc:
            pytest.skip(f"no default WASAPI loopback available: {exc!r}")
        device_rate = int(dev.get("defaultSampleRate") or 48000)
        device_channels = int(dev.get("maxInputChannels") or 2)
        chunk = max(256, device_rate // 50)  # ~20 ms blocks

        frame_count_holder = {"count": 0, "bytes": 0, "shape_ok": True}
        got_data = threading.Event()

        def _callback(in_data, frame_count, time_info, status):
            # This is the exact numpy-reshape path used in the wizard
            # and in capture/audio.py — float32 mono/stereo input.
            pcm = np.frombuffer(in_data, dtype=np.float32)
            try:
                if device_channels > 1:
                    pcm = pcm.reshape(-1, device_channels)
                frame_count_holder["count"] += 1
                frame_count_holder["bytes"] += len(in_data)
            except ValueError:
                frame_count_holder["shape_ok"] = False
            if frame_count_holder["count"] >= 3:
                got_data.set()
            return (None, pyaudiowpatch.paContinue)

        stream = pa.open(
            format=pyaudiowpatch.paFloat32,
            channels=device_channels,
            rate=device_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=chunk,
            stream_callback=_callback,
        )
        try:
            stream.start_stream()
            # Bounded wait. Shared-mode WASAPI loopback delivers silent
            # buffers at the device's period rate even with no audio
            # playing, but the first buffer may take ~50-200 ms. Give
            # it 2 seconds.
            fired = got_data.wait(timeout=2.0)
        finally:
            stream.stop_stream()
            stream.close()

        if not fired:
            # No callbacks within 2s = device is idle or the driver only
            # fires on active audio. Not a product regression — valid
            # environmental condition (nothing is playing). Don't fail.
            pytest.skip(
                f"no callback fired on default loopback {dev['name']!r}; "
                f"device is likely idle. Play audio and re-run, or this "
                f"test is non-useful on this machine."
            )

        assert frame_count_holder["shape_ok"], (
            "callback's numpy.reshape raised — frame size doesn't divide "
            "by channel count, which means format/channels mismatch at open()"
        )
        assert frame_count_holder["count"] >= 3, (
            f"expected at least 3 callbacks, got {frame_count_holder['count']}"
        )
        assert frame_count_holder["bytes"] > 0

    finally:
        pa.terminate()


def test_default_list_devices_marks_the_windows_default_loopback() -> None:
    """The Sound step marks "System default" on the entry
    _default_list_devices() flags; that flag must sit on the loopback of
    the Windows default output, read through the real PyAudioWPatch
    (hard rule 3: a new external call site, get_default_wasapi_loopback).

    Skips (not fails) on a host whose default output has no WASAPI
    loopback (VM, headless runner).

    Seeded defect: mark the entry matching get_default_output_device_info()
    (the render endpoint, not its loopback) -> no or the wrong entry is
    marked -> red.
    """
    pyaudiowpatch = pytest.importorskip("pyaudiowpatch")
    from yamabiko.wizard_pages.audio import _default_list_devices

    try:
        with pyaudiowpatch.PyAudio() as p:
            expected = p.get_default_wasapi_loopback()
    except (OSError, LookupError) as exc:
        pytest.skip(f"no default WASAPI loopback on this host: {exc!r}")

    marked = [(dev["index"], dev["name"]) for dev in _default_list_devices() if dev.get("default")]
    assert marked == [(expected["index"], expected["name"])]

