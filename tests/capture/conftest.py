"""Capture-tier fixtures + the audio-stream-monitor leak guard.

Why this exists
---------------
``LoopbackCapture.start()`` spawns the ``audio-stream-monitor`` daemon
thread (a 1 s poll loop). The unit tier mocks pyaudiowpatch only for the
duration of a ``with patch("...pyaudio.PyAudio", ...)`` block inside the
``_run_start`` test helper -- but ``start()`` spawns the monitor thread
*inside* that block, and the thread outlives it. Once a test feeds an
above-floor frame to the captured callback, ``_ever_audible`` latches True
and, ``_idle_death_s`` later, the monitor's idle-reopen path
(``_watch`` -> ``_reopen_stream``) calls the now-UNMOCKED
``pyaudiowpatch.PyAudio()`` -> real PortAudio / WASAPI init on the
headless CI runner -> access violation that kills the whole pytest
process. The crash is asynchronous, so it lands in whatever *later* test
file happens to be running (exit 139 / no summary / random location), and
it corrupts the process exit code even though every test passed.

Root cause captured 2026-07-08 via ``faulthandler(all_threads=False)``::

    Windows fatal exception: access violation
      File "...\\pyaudiowpatch\\__init__.py", line 716 in __init__
      File "...\\yamabiko\\capture\\audio.py", line 507 in _reopen_stream
      File "...\\yamabiko\\capture\\audio.py", line 381 in _watch
      File "...\\threading.py", line 995 in run   (audio-stream-monitor)

Stopping every monitor a capture test starts, at that test's teardown --
within milliseconds of the body, comfortably inside the >= 1 s idle timer
-- eliminates the leak deterministically.
"""
from __future__ import annotations

import pytest

from yamabiko.capture.audio import LoopbackCapture


def stop_capture_monitors(caps):
    """Stop each capture and join its ``audio-stream-monitor`` thread.

    Returns the monitor threads still alive after a 5 s join (an empty list
    on success). Idempotent: a capture already stopped, or one that never
    started a monitor, is a safe no-op. ``stop()`` is fully guarded and does
    not touch the event loop, so it is safe on mock-backed unit captures.
    """
    still_alive = []
    for cap in caps:
        try:
            cap.stop()  # sets _stop -> the monitor's poll loop exits next tick
        except Exception:
            pass
        mon = getattr(cap, "_monitor", None)
        if mon is not None:
            mon.join(timeout=5)
            if mon.is_alive():
                still_alive.append(mon)
    return still_alive


@pytest.fixture
def stop_capture_monitors_fn():
    """Expose ``stop_capture_monitors`` to tests that assert on the guard."""
    return stop_capture_monitors


@pytest.fixture(autouse=True)
def _no_leaked_capture_monitors(monkeypatch, request):
    """Stop every ``audio-stream-monitor`` a capture test starts, at teardown.

    See the module docstring for the full root cause. Tracks captures by
    wrapping ``LoopbackCapture._start_monitor`` (the only path that spawns
    the thread), so exactly the captures that leaked a monitor get stopped.

    A monitor that is STILL alive after the 5 s join did not actually stop --
    which is the exact precondition for the asynchronous WASAPI access violation
    the module docstring describes. Discarding that fact (the pre-M-16 behaviour)
    let the leak sail through green here and detonate a later test file as an
    unattributable exit-139 segfault. So the surviving-thread list is surfaced
    as a ``pytest.fail`` naming the threads and the owning test, pinning the
    anomaly to its culprit instead of a random downstream victim.

    Seeded-defect drill: make ``LoopbackCapture.stop()`` a no-op (so ``_stop``
    is never set and the poll loop never exits) -> the join times out, the
    thread stays alive, and this teardown fails the owning test by name.
    """
    started: list[LoopbackCapture] = []
    real_start_monitor = LoopbackCapture._start_monitor

    def _tracking_start_monitor(self):
        started.append(self)
        return real_start_monitor(self)

    monkeypatch.setattr(LoopbackCapture, "_start_monitor", _tracking_start_monitor)
    yield
    still_alive = stop_capture_monitors(started)
    if still_alive:
        identities = ", ".join(
            f"{t.name}(ident={t.ident}, daemon={t.daemon})" for t in still_alive
        )
        pytest.fail(
            f"{len(still_alive)} audio-stream-monitor thread(s) survived the 5 s "
            f"join at teardown of {request.node.nodeid}: {identities}. A live "
            f"monitor's idle-reopen path calls the now-UNMOCKED "
            f"pyaudiowpatch.PyAudio() -> a real WASAPI init that access-violates "
            f"and segfaults a LATER test file (exit 139, no summary, random "
            f"location). The leak belongs to THIS test -- stop the capture it "
            f"started before it detonates a downstream victim."
        )
