"""Unit tests for yamabiko.lifecycle.shutdown_event."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Named Win32 Events are Windows-only",
)

from yamabiko.lifecycle.shutdown_event import (
    ENV_VAR,
    ENV_VAR_BY_ROLE,
    ShutdownEventWriter,
    ShutdownEventListener,
)


def test_env_var_by_role_returns_distinct_names_per_role():
    """Fails red if: pipeline + overlay share a single env-var name.
    A future process with two ShutdownEventListeners in one env (e.g.
    a side-car observing both the pipeline and the overlay) couldn't
    disambiguate, which is the L3 ambiguity.
    """
    assert ENV_VAR_BY_ROLE["pipeline"] != ENV_VAR_BY_ROLE["overlay"], (
        "role-scoped env vars must differ; got "
        f"pipeline={ENV_VAR_BY_ROLE['pipeline']!r}, "
        f"overlay={ENV_VAR_BY_ROLE['overlay']!r}"
    )
    # Sanity: names are still recognisably Yamabiko shutdown events.
    for role, env in ENV_VAR_BY_ROLE.items():
        assert "YAMABIKO_SHUTDOWN_EVENT" in env, (
            f"env var for role={role!r} lost the YAMABIKO_SHUTDOWN_EVENT "
            f"namespace: {env!r}"
        )


def test_from_env_reads_role_scoped_var(monkeypatch):
    """Fails red if: from_env(role='overlay') reads the pipeline env var
    (or vice versa). Seeded-defect drill: reverting to a single
    ENV_VAR lookup in from_env makes the overlay listener's .name
    equal the pipeline listener's .name, assert goes red.
    """
    monkeypatch.setenv(ENV_VAR_BY_ROLE["pipeline"], "pipeline-only-name")
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], "overlay-only-name")
    pipeline = ShutdownEventListener.from_env(role="pipeline")
    overlay = ShutdownEventListener.from_env(role="overlay")
    assert pipeline is not None and pipeline.name == "pipeline-only-name"
    assert overlay is not None and overlay.name == "overlay-only-name"


def test_env_var_legacy_alias_points_at_pipeline_role():
    """Fails red if: the legacy ``ENV_VAR`` constant drifts from
    ``ENV_VAR_BY_ROLE['pipeline']``. The alias is the back-compat shim
    for any out-of-tree integration that read the pre-L3 single
    ``YAMABIKO_SHUTDOWN_EVENT`` name -- silently repointing it to a
    different role (or a renamed value) would break those integrations
    without a visible breakage at import time.
    """
    assert ENV_VAR == ENV_VAR_BY_ROLE["pipeline"], (
        f"legacy ENV_VAR drifted from pipeline role: "
        f"ENV_VAR={ENV_VAR!r} but ENV_VAR_BY_ROLE['pipeline']="
        f"{ENV_VAR_BY_ROLE['pipeline']!r}"
    )


def test_from_env_unknown_role_raises():
    """Fails red if: from_env silently swallows a typo'd role, returning
    None because no env var matched. A silent None means 'no parent set
    the event' which is indistinguishable from a typo; raise instead."""
    with pytest.raises(ValueError, match="unknown shutdown-event role"):
        ShutdownEventListener.from_env(role="not-a-real-role")


def test_open_existing_round_trips_signal_to_self_created_writer():
    """Fails red if: ShutdownEventWriter.open_existing can't find an
    event created by another writer with the same (role, pid) pair.

    This exercises the watchdog self-shutdown cascade: run_watchdog
    calls ShutdownEventWriter.create("watchdog", pid), and any other
    process (the frozen-E2E test, a future tray-Quit cascade) calls
    ShutdownEventWriter.open_existing(name_for("watchdog", pid)) +
    signal() to trigger graceful shutdown. If open_existing silently
    returns a handle-less writer (our fallback on OpenEventW failure),
    signal() returns False and no wake occurs -- test catches that.

    Seeded-defect drill: remove the OpenEventW call from open_existing
    (stub to `return cls(name=name, handle=None)`) -> opener.handle is
    None and opener.signal() returns False, assertion fails.
    """
    import asyncio as _aio
    import time as _time

    owner = ShutdownEventWriter.create("watchdog", parent_pid=os.getpid())
    assert owner.handle is not None, "CreateEventW failed; test can't proceed"

    try:
        name = ShutdownEventWriter.name_for("watchdog", os.getpid())
        assert name == owner.name, (
            f"name_for drift: helper produced {name!r}, "
            f"create produced {owner.name!r}"
        )
        opener = ShutdownEventWriter.open_existing(name)
        try:
            assert opener.handle is not None, (
                "open_existing returned a handle-less writer; "
                "OpenEventW on a freshly-created event should succeed"
            )
            # Listen on the event via the original owner's handle and
            # wire a ShutdownEventListener so we can assert the signal
            # actually wakes waiters -- same semantics as production.
            os.environ[ENV_VAR_BY_ROLE["watchdog"]] = name
            listener = None
            try:
                listener = ShutdownEventListener.from_env(role="watchdog")
                assert listener is not None

                async def _run() -> float:
                    loop = _aio.get_running_loop()
                    event = _aio.Event()
                    listener.start_watcher(loop, event)
                    await _aio.sleep(0.05)
                    t0 = _time.monotonic()
                    assert opener.signal()
                    await _aio.wait_for(event.wait(), timeout=2.0)
                    return _time.monotonic() - t0

                elapsed = _aio.run(_run())
                assert elapsed < 0.5, f"signal -> wake too slow: {elapsed}s"
            finally:
                os.environ.pop(ENV_VAR_BY_ROLE["watchdog"], None)
                if listener is not None:
                    listener.close()
        finally:
            opener.close()
    finally:
        owner.close()


def test_event_name_includes_role_and_parent_pid():
    """Fails red if: event name is a hardcoded string, causing
    collisions between concurrent watchdogs (e.g. two dev instances)."""
    w = ShutdownEventWriter.create("pipeline", parent_pid=12345)
    try:
        assert w.name == r"Local\yamabiko.shutdown.pipeline.12345"
        assert w.handle is not None
    finally:
        w.close()


def test_two_writers_with_different_pids_do_not_collide():
    """Fails red if: event name doesn't include PID -> two writers
    share a handle and signalling one fires both listeners."""
    w1 = ShutdownEventWriter.create("pipeline", parent_pid=11111)
    w2 = ShutdownEventWriter.create("pipeline", parent_pid=22222)
    try:
        assert w1.name != w2.name
    finally:
        w1.close()
        w2.close()


def test_listener_from_env_returns_none_when_env_absent(monkeypatch):
    """Fails red if: from_env silently fabricates a handle when env
    var is missing (would cause child to crash on signal wait)."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    listener = ShutdownEventListener.from_env()
    assert listener is None


def test_listener_from_env_opens_event_when_env_present(monkeypatch):
    """Fails red if: env var is read but OpenEventW isn't called."""
    w = ShutdownEventWriter.create("pipeline", parent_pid=os.getpid())
    try:
        monkeypatch.setenv(ENV_VAR, w.name)
        listener = ShutdownEventListener.from_env()
        try:
            assert listener is not None
            assert listener.handle is not None
        finally:
            if listener is not None:
                listener.close()
    finally:
        w.close()


def test_close_terminates_blocked_watcher_within_join_bound(monkeypatch):
    """close() must wake a watcher still blocked on its wait and join it,
    not just CloseHandle the named handle out from under a pending wait.

    The named event is NEVER signalled, so the watcher stays blocked in
    its wait -- the tray-Quit / max-segments / give-up shutdown paths,
    where teardown runs while the wait is still pending. Closing a handle
    under a pending wait is undefined behavior (Microsoft docs), and the
    pre-fix close() did exactly that (CloseHandle, no wake, no join),
    leaving the daemon thread blocked forever -- a leaked native-wait
    thread of the kind that segfaults the process at interpreter exit.

    RED before the fix: close() returns while the captured watcher thread
    is still alive -> the liveness assert fails. GREEN after: close()
    SetEvent()s a private cancel event, the WaitForMultipleObjects returns,
    the thread exits, join() succeeds, and is_alive() is False.

    Uses the asyncio start_watcher API (unchanged signature) so the RED
    run exercises the real pre-fix code path.
    """
    w = ShutdownEventWriter.create("overlay-close-blocked", parent_pid=os.getpid())
    assert w.handle is not None, "CreateEventW returned NULL; test cannot proceed"
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w.name)
    listener = ShutdownEventListener.from_env(role="overlay")
    assert listener is not None and listener.handle is not None

    loop = asyncio.new_event_loop()
    thread = None
    try:
        fired = asyncio.Event()
        # Watcher blocks on the never-signalled named event. call_soon_threadsafe
        # is only reached if the NAMED event fires (it never does here), so a
        # non-running loop is fine.
        listener.start_watcher(loop, fired)
        thread = listener._thread
        assert thread is not None
        # Let the daemon thread reach its blocking wait.
        time.sleep(0.15)
        assert thread.is_alive(), "watcher thread should be blocked on the wait"

        t0 = time.monotonic()
        listener.close()
        elapsed = time.monotonic() - t0

        assert not thread.is_alive(), (
            "watcher thread still alive after close(): close() closed the "
            "handle without waking + joining the pending wait (undefined "
            "behavior); the blocked thread leaks into interpreter shutdown"
        )
        assert elapsed < 2.5, f"close() exceeded the join bound: {elapsed}s"
    finally:
        loop.close()
        # Safety net for the pre-fix RED run: if close() failed to join, the
        # thread is still blocked -- signal the event so it can't leak past
        # this test into interpreter shutdown, then join it.
        if thread is not None and thread.is_alive():
            w.signal()
            thread.join(timeout=2.0)
        w.close()


def test_signal_wakes_listener_daemon_thread(monkeypatch):
    """Fails red if: listener's daemon thread isn't started, OR
    the wake callback isn't invoked via loop.call_soon_threadsafe.

    This test shares the kernel-object name
    ``Local\\yamabiko.shutdown.pipeline.{pytest-pid}`` with the
    test_build_watchdog suite. If a leaked handle from an earlier test
    pins the manual-reset event alive in the SET state, the watcher
    wakes during the arming sleep regardless of whether w.signal()
    works -- the same order-dependent vacuity seen in the real-tier
    tests. The precondition below fails loudly on pollution instead.
    """
    import ctypes as _ct
    from ctypes import wintypes as _wt

    w = ShutdownEventWriter.create("pipeline", parent_pid=os.getpid())
    _WFSO = _ct.windll.kernel32.WaitForSingleObject
    _WFSO.argtypes = [_wt.HANDLE, _wt.DWORD]
    _WFSO.restype = _wt.DWORD
    assert _WFSO(w.handle, 0) == 0x102, (  # WAIT_TIMEOUT == non-signalled
        f"event {w.name!r} already signalled at creation -- a leaked "
        f"handle from an earlier test is polluting the kernel object; "
        f"the signal->wake assertion below would be vacuous."
    )
    try:
        monkeypatch.setenv(ENV_VAR, w.name)
        listener = ShutdownEventListener.from_env()
        assert listener is not None
        try:
            async def run() -> float:
                loop = asyncio.get_running_loop()
                event = asyncio.Event()
                listener.start_watcher(loop, event)
                # Give watcher thread a moment to enter WaitForSingleObject.
                await asyncio.sleep(0.1)
                start = time.monotonic()
                w.signal()
                await asyncio.wait_for(event.wait(), timeout=2.0)
                return time.monotonic() - start

            elapsed = asyncio.run(run())
            assert elapsed < 0.5, f"signal too slow: {elapsed}s"
        finally:
            listener.close()
    finally:
        w.close()


def test_start_flag_watcher_fires_callback_once_on_named_event_signal(monkeypatch):
    """start_flag_watcher wakes a plain (non-asyncio) callback when the
    named event is signalled -- the seam the Qt overlay uses instead of
    an asyncio.Event, since a Qt loop has no call_soon_threadsafe.

    Observable effect: the recorded callback runs exactly once after the
    writer signals. Synchronised via a threading.Event so the assertion
    can't race the daemon thread.

    Seeded-defect drill: change the WAIT_OBJECT_0 arm in _start_watcher to
    `return` (drop the on_fire() call) -> the callback never runs -> the
    wait_for(fired) below times out and calls stays [] -> red.
    """
    w = ShutdownEventWriter.create("overlay-flagfire", parent_pid=os.getpid())
    assert w.handle is not None, "CreateEventW returned NULL; test cannot proceed"
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w.name)
    listener = ShutdownEventListener.from_env(role="overlay")
    assert listener is not None and listener.handle is not None

    calls: list[int] = []
    fired = threading.Event()

    def _on_fire() -> None:
        calls.append(1)
        fired.set()

    try:
        listener.start_flag_watcher(_on_fire)
        time.sleep(0.1)  # let the watcher reach its wait
        assert w.signal() is True, "SetEvent failed"
        assert fired.wait(timeout=2.0), "callback never fired after signal"
        assert calls == [1], f"callback fired {len(calls)} times; expected once"
    finally:
        listener.close()
        w.close()


def test_close_does_not_fire_callback_when_named_event_never_signaled(monkeypatch):
    """The cancel wake must let the watcher EXIT without firing the
    shutdown callback: close() cancelling the wait is not a shutdown.

    The named event is never signalled; close() SetEvent()s only the
    private cancel event. The watcher must return via the cancel arm
    (WAIT_OBJECT_0 + 1) WITHOUT calling on_fire, and be joined (no leaked
    thread).

    Seeded-defect drill: make the cancel arm fall through to on_fire()
    (delete the `elif ... : return` guard) -> the callback fires on a
    plain teardown -> calls becomes [1] -> red.
    """
    w = ShutdownEventWriter.create("overlay-nocancelfire", parent_pid=os.getpid())
    assert w.handle is not None, "CreateEventW returned NULL; test cannot proceed"
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w.name)
    listener = ShutdownEventListener.from_env(role="overlay")
    assert listener is not None and listener.handle is not None

    calls: list[int] = []
    thread = None
    try:
        # Zero-arg recorder: _start_watcher invokes on_fire() with no
        # args, so a bare list.append (which needs one arg) would raise
        # and be swallowed -- masking a cancel-arm-fires-on_fire defect.
        listener.start_flag_watcher(lambda: calls.append(1))
        thread = listener._thread
        assert thread is not None
        time.sleep(0.15)
        listener.close()  # cancel wake, not a shutdown signal
        assert not thread.is_alive(), "watcher not joined after close()"
        assert calls == [], (
            f"cancel wake fired the shutdown callback ({calls}); a plain "
            f"teardown must not look like a STOP_ALL"
        )
    finally:
        if thread is not None and thread.is_alive():
            w.signal()
            thread.join(timeout=2.0)
        w.close()


def test_close_is_idempotent_before_and_after_start(monkeypatch):
    """close() must be safe and non-hanging when called: (a) before any
    watcher started, and (b) twice in a row after a watcher started but the
    named event never fired. No hang, both handle attributes released AND
    nulled, second call a no-op.

    The released-and-NULLED asserts carry the idempotence claim: ctypes
    CloseHandle has no errcheck, so a double-close never raises -- the
    only observable of a broken close() is a handle attribute left
    non-None after the call, which makes the SECOND close() re-close a
    stale handle value the OS may have already recycled onto an
    unrelated kernel object.

    Fails red if: close() closes a handle without nulling its attribute
    (self.handle or self._cancel_handle).

    Seeded-defect drill: drop the ``self.handle = None`` reset from
    close()'s finally -> the first close() leaves the closed handle
    visible -> the ``listener.handle is None`` assert in (a) goes red.
    Companion: dropping ``self._cancel_handle = None`` reds the
    both-attributes-None assert in (b).
    """
    # (a) close() before start_watcher: no thread, just an open handle.
    w1 = ShutdownEventWriter.create("overlay-idem-nostart", parent_pid=os.getpid())
    assert w1.handle is not None
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w1.name)
    try:
        listener = ShutdownEventListener.from_env(role="overlay")
        assert listener is not None and listener.handle is not None
        listener.close()
        assert listener.handle is None, "close() did not release the handle"
        listener.close()  # second call must be a no-op, not a crash
    finally:
        w1.close()

    # (b) double close() after start, event never signalled.
    w2 = ShutdownEventWriter.create("overlay-idem-started", parent_pid=os.getpid())
    assert w2.handle is not None
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w2.name)
    thread = None
    try:
        listener = ShutdownEventListener.from_env(role="overlay")
        assert listener is not None and listener.handle is not None
        listener.start_flag_watcher(lambda: None)
        thread = listener._thread
        time.sleep(0.1)
        listener.close()
        listener.close()  # idempotent second close
        assert thread is not None and not thread.is_alive()
        assert listener.handle is None and listener._cancel_handle is None, (
            f"close() must null both handle attributes (handle="
            f"{listener.handle!r}, _cancel_handle="
            f"{listener._cancel_handle!r}); a stale non-None value makes "
            f"the next close() double-CloseHandle a possibly-recycled "
            f"handle"
        )
    finally:
        if thread is not None and thread.is_alive():
            w2.signal()
            thread.join(timeout=2.0)
        w2.close()
