"""run_overlay's shutdown-event waiter must not close the parent's named
Win32 event out from under its blocked waiter thread.

The overlay opens the pipeline's ``overlay``-role named event and starts a
daemon thread that blocks waiting on it, so a parent SetEvent quits the
overlay. On the common shutdown paths (tray Quit, app.quit) the named event
is NEVER signalled, so that thread is still blocked when run_overlay tears
down. The pre-fix code CloseHandle'd the event there -- closing a handle
while a wait on it is pending is undefined behavior (Microsoft docs), and it
left the daemon thread blocked forever (a leaked native-wait thread of the
kind that segfaults the process at interpreter exit).

The fix routes the overlay through the shared ShutdownEventListener's
cancellable watcher (start_flag_watcher + close()), which waits on the named
event AND a private cancel event via WaitForMultipleObjects; close() wakes
and joins the thread before closing any handle.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import types

import pytest
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Named Win32 Events are Windows-only",
)

from yamabiko.lifecycle.shutdown_event import ENV_VAR_BY_ROLE, ShutdownEventWriter


def _watcher_threads() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate()
        if t.name == "shutdown-event-watcher" and t.is_alive()
    ]


def _fake_overlay(app):
    """Minimal OverlayApp stand-in: run_overlay only touches .app + .client.
    Using a real QApplication (QTimer needs a real QObject parent) and a
    no-op client keeps this off the heavy build_overlay path.
    """
    client = types.SimpleNamespace(start=lambda: None, stop=lambda: None)
    return types.SimpleNamespace(app=app, client=client)


def test_run_overlay_joins_shutdown_waiter_when_event_never_fires(monkeypatch):
    """On a plain quit (named event never signalled), run_overlay must wake
    and join its shutdown-waiter thread before returning -- not leave it
    blocked on a wait whose handle was closed.

    exec() is mocked to return immediately WITHOUT signalling the event, so
    the waiter is still blocked when teardown runs. After run_overlay
    returns, no shutdown-event-watcher thread created during the call may
    remain alive.

    Seeded-defect drill: revert ShutdownEventListener.close() to a bare
    CloseHandle (drop the SetEvent(cancel) + join) -> the waiter stays
    blocked -> a new live shutdown-event-watcher thread survives the call
    -> the leaked-thread assert goes red.
    """
    import yamabiko.overlay_app.__main__ as overlay_main

    w = ShutdownEventWriter.create("overlay-runoverlay-nofire", parent_pid=os.getpid())
    assert w.handle is not None, "CreateEventW returned NULL; test cannot proceed"
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w.name)

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    # Never let a stray detached worker trip run_overlay's os._exit path in
    # the test process.
    monkeypatch.setattr(overlay_main.thread_keeper, "any_running", lambda: False)

    before = {id(t) for t in threading.enumerate()}
    try:
        rc = overlay_main.run_overlay(_fake_overlay(app))
        assert rc == 0
        leaked = [t for t in _watcher_threads() if id(t) not in before]
        assert leaked == [], (
            f"run_overlay left a blocked shutdown-waiter thread alive: "
            f"{leaked}; close() closed the handle without waking + joining "
            f"the pending wait"
        )
    finally:
        # Safety net: unblock + join anything still alive so the pre-fix RED
        # run can't leak a native-wait thread into interpreter shutdown.
        if any(id(t) not in before for t in _watcher_threads()):
            w.signal()
            for t in _watcher_threads():
                if id(t) not in before:
                    t.join(timeout=2.0)
        w.close()


def test_run_overlay_quits_when_named_shutdown_event_fires(monkeypatch, caplog):
    """When the parent SetEvents the overlay's named event, the waiter trips
    the poll flag, the QTimer sees it, and app.quit() ends exec() -- so
    run_overlay returns because of the shutdown event, logging its
    'overlay.shutdown_event.fired' marker.

    Drives the REAL Qt event loop (exec is not mocked). A QTimer signals the
    named event shortly after the loop starts; a second QTimer is a hard
    guard so a broken waiter turns a hang into an assertion, never a stuck
    suite.

    Seeded-defect drill: drop the start_flag_watcher(...) call in
    run_overlay -> the named event never trips the flag -> only the guard
    quits -> guard_fired is True and the fired-marker assert goes red.
    """
    from PySide6.QtCore import QTimer
    import yamabiko.overlay_app.__main__ as overlay_main

    w = ShutdownEventWriter.create("overlay-runoverlay-fire", parent_pid=os.getpid())
    assert w.handle is not None, "CreateEventW returned NULL; test cannot proceed"
    monkeypatch.setenv(ENV_VAR_BY_ROLE["overlay"], w.name)

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(overlay_main.thread_keeper, "any_running", lambda: False)

    signaller = QTimer(app)
    signaller.setSingleShot(True)
    signaller.timeout.connect(lambda: w.signal())
    signaller.start(200)

    guard_fired = {"v": False}

    def _guard() -> None:
        guard_fired["v"] = True
        app.quit()

    guard = QTimer(app)
    guard.setSingleShot(True)
    guard.timeout.connect(_guard)
    guard.start(5000)

    before = {id(t) for t in threading.enumerate()}
    try:
        with caplog.at_level(logging.INFO, logger="yamabiko.overlay_app.__main__"):
            rc = overlay_main.run_overlay(_fake_overlay(app))
    finally:
        guard.stop()
        signaller.stop()
        if any(id(t) not in before for t in _watcher_threads()):
            w.signal()
            for t in _watcher_threads():
                if id(t) not in before:
                    t.join(timeout=2.0)
        w.close()

    assert rc == 0
    assert not guard_fired["v"], (
        "run_overlay did not quit on the shutdown event within 5s; the "
        "5s safety guard had to force the quit -> the waiter never tripped"
    )
    assert any(
        "overlay.shutdown_event.fired" in r.getMessage() for r in caplog.records
    ), (
        "run_overlay quit without logging the shutdown-event fired marker; "
        f"records: {[r.getMessage() for r in caplog.records]!r}"
    )
    leaked = [t for t in _watcher_threads() if id(t) not in before]
    assert leaked == [], f"waiter thread not joined after quit: {leaked}"
