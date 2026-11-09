"""Real-tier test: pipeline stand-in exits cleanly when the named event fires.

This exercises the listener + asyncio.Event wake path end-to-end, not
the full run_pipeline (which is too heavy for unit testing). Full
frozen-exe wiring is covered by the E2E tests.

Fails red if:
- ShutdownEventListener.from_env() stops honouring YAMABIKO_SHUTDOWN_EVENT
  (returns None when the env var is set + handle is openable).
- The daemon watcher thread fails to wake the asyncio.Event on SetEvent
  (loop.call_soon_threadsafe not scheduled; event name mismatch between
  writer and listener; WaitForSingleObject returns non-WAIT_OBJECT_0).
- ShutdownEventWriter.signal() no longer flips the underlying Win32 event.

Seeded-defect drill: replace ``listener.start_watcher(loop, done)`` with
``pass`` in the child code; the child hits asyncio.wait_for timeout,
prints "TIMEOUT", and exits 3. The test then fails on the ``rc == 0``
assertion. Confirmed red-able.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.skipif(
        not sys.platform.startswith("win"),
        reason="Named Win32 events are Windows-only",
    ),
    pytest.mark.real,
]

from yamabiko.lifecycle.shutdown_event import ENV_VAR, ShutdownEventWriter


# Child stand-in: opens listener, starts watcher, awaits an asyncio event,
# exits cleanly. Mirrors the pattern used in run_pipeline.
_CHILD_CODE = '''
import asyncio, os, sys
from yamabiko.lifecycle.shutdown_event import ShutdownEventListener

async def main():
    done = asyncio.Event()
    listener = ShutdownEventListener.from_env()
    if listener is None:
        print("NO_LISTENER", flush=True)
        sys.exit(2)
    loop = asyncio.get_running_loop()
    listener.start_watcher(loop, done)
    print("READY", flush=True)
    try:
        await asyncio.wait_for(done.wait(), timeout=10.0)
        print("DONE", flush=True)
    except asyncio.TimeoutError:
        print("TIMEOUT", flush=True)
        sys.exit(3)
    finally:
        listener.close()

asyncio.run(main())
'''


def test_pipeline_exits_on_shutdown_event():
    """The listener+watcher path wakes app.done end-to-end.

    Observable effect: child process exits 0 after the parent signals
    the named event. No mock assertions -- only PID-level outcomes
    (READY sentinel on stdout, exit code 0).

    Test-pollution hardening (order-dependent vacuity): the event role
    here was previously "pipeline" -- the SAME kernel-object name
    (``Local\\yamabiko.shutdown.pipeline.{pytest-pid}``) the
    test_build_watchdog tests create. Leaked handles from those tests
    pinned the manual-reset event alive in the SET state, so in
    full-suite order this child's watcher woke instantly at start and
    the test passed even with ``ShutdownEventWriter.signal()`` broken --
    the exact defect it exists to catch. Two layers of defence:

    1. A role string unique to this test, so no other test in the
       session can ever create/signal the same kernel object.
    2. A hard precondition that the freshly-created event is
       NON-signalled (WaitForSingleObject(handle, 0) == WAIT_TIMEOUT);
       if a leaked SET object is ever inherited again, this fails
       loudly with a pollution diagnosis instead of passing vacuously.

    Seeded-defect drill: stub ``ShutdownEventWriter.signal`` to
    ``return True`` without calling SetEvent -> child hits the 10 s
    wait_for timeout, prints TIMEOUT, exits 3 -> the ``rc == 0``
    assertion goes red.
    """
    import ctypes
    from ctypes import wintypes

    # Unique role: the name becomes
    # Local\yamabiko.shutdown.pipeline-roundtrip-test.{pid} -- no other
    # test creates it. The role string only feeds the event NAME; the
    # child's from_env() reads the name from the env var, so the
    # listener path is identical to production.
    w = ShutdownEventWriter.create(
        "pipeline-roundtrip-test", parent_pid=os.getpid(),
    )
    # Sanity: on Windows CreateEventW should succeed; if the handle
    # is None the real Win32 primitive is unavailable and the rest of
    # the test is meaningless.
    assert w.handle is not None, "CreateEventW returned NULL; test cannot proceed"
    # Pollution precondition: the event must start NON-signalled. A signalled
    # state here means a leaked handle from an earlier test pinned a SET
    # kernel object under this name -- the round-trip below would then
    # pass without signal() doing anything.
    WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
    WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    WaitForSingleObject.restype = wintypes.DWORD
    WAIT_TIMEOUT = 0x102
    assert WaitForSingleObject(w.handle, 0) == WAIT_TIMEOUT, (
        f"event {w.name!r} is ALREADY SIGNALLED at creation: a leaked "
        f"handle from an earlier test is polluting the kernel object. "
        f"The writer->listener round-trip below would be vacuous. "
        f"Find and close the leaking handle."
    )
    proc: subprocess.Popen | None = None
    try:
        env = os.environ.copy()
        env[ENV_VAR] = w.name
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_CODE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        # Wait for READY -- confirms listener opened the event and
        # started the watcher thread before we signal.
        assert proc.stdout is not None
        ready_line = proc.stdout.readline().strip()
        assert ready_line == "READY", f"unexpected first line: {ready_line!r}"
        # Signal the event: the daemon thread should unblock and
        # loop.call_soon_threadsafe(done.set).
        assert w.signal() is True, "SetEvent failed"
        # Wait for clean exit (bounded so a stuck wait fails loudly).
        rc = proc.wait(timeout=5.0)
        stdout_tail = proc.stdout.read() if proc.stdout else ""
        assert rc == 0, f"exit code {rc}; tail={stdout_tail!r}"
        # Belt-and-braces: the child only prints "DONE" after
        # awaited-event resolved; rules out a coincidental exit.
        assert "DONE" in stdout_tail, f"missing DONE sentinel; tail={stdout_tail!r}"
    finally:
        w.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
