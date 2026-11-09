"""Quit-time detach for the overlay's update-check worker.

The update-check QThread does a blocking httpx GET (5 s timeout,
update_check.py). If the user quits inside that window, build_overlay must
detach the running QThread into thread_keeper on aboutToQuit instead of
letting QApplication teardown destroy it -- destroying a running QThread is
a Qt6 qFatal abort ("QThread: Destroyed while thread is still running").
run_overlay's existing thread_keeper.any_running() guard then hard-exits
past interpreter finalization.

Mirrors the palette / gazetteer aboutToQuit detach tests in
test_build_overlay.py; kept in its own module because the update-check seam
lives in overlay_app.__main__, not tray.
"""
from __future__ import annotations

import threading

import pytest
from PySide6.QtWidgets import QApplication

pytest.importorskip("pytestqt")


@pytest.fixture
def _update_env(monkeypatch, tmp_path):
    """build_overlay harness that starts a REAL _UpdateCheckThread blocked on
    a gated fetch, so the aboutToQuit release path can be exercised end to
    end.

    Mirrors test_build_overlay._overlay_env's process-global stubs (file log,
    crash evidence, Qt-message routing, the QApplication singleton proxy) but
    deliberately does NOT stub _schedule_update_check -- that would hide the
    very thread under test. load_full_config is pinned to enable the update
    check and disable the gazetteer feature, so the only in-flight worker is
    the update-check thread.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    import yamabiko.overlay_app.__main__ as overlay_main

    monkeypatch.setattr(overlay_main, "setup_main_log", lambda *a, **kw: None)
    monkeypatch.setattr(overlay_main, "install_crash_evidence", lambda role: None)
    monkeypatch.setattr(overlay_main, "install_qt_message_handler", lambda: None)
    monkeypatch.setattr(
        overlay_main, "load_full_config",
        lambda: {
            "updates": {"check_on_startup": True},
            "gazetteer": {"enabled": False},
            "romanization": {"enabled": False},
        },
    )

    # QApplication is a process-global singleton; reuse any existing instance
    # rather than constructing a second one (which would raise).
    app_singleton = QApplication.instance() or QApplication([])

    class _AppProxy:
        def __new__(cls, argv):
            return app_singleton

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy):
            pass

    monkeypatch.setattr(overlay_main, "QApplication", _AppProxy)
    return overlay_main


def test_about_to_quit_detaches_running_update_check_worker(
    _update_env, qtbot, monkeypatch,
):
    """build_overlay wires _release_update_thread to app.aboutToQuit, so a
    quit while the update check is mid-GET DETACHES the running QThread into
    thread_keeper instead of letting QApplication teardown destroy it (Qt6
    qFatal). Invocation-style (§3.9): emits aboutToQuit and asserts the
    observable detach + keeper state + that the updateAvailable result slot
    was disconnected -- not that a callable is registered.

    Seeded-defect drill: drop the
    `app.aboutToQuit.connect(lambda: _release_update_thread(update_thread))`
    line in build_overlay -> the worker stays parented to the app and
    thread_keeper stays empty -> the parent()/any_running asserts go red.
    Drop the `thread.updateAvailable.disconnect()` line in
    _release_update_thread -> the detached worker still fires its result slot
    after unblock -> the `notified == []` assertion goes red.
    """
    overlay_main = _update_env
    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    notified: list = []
    # Block the worker inside the fetch so it is still running at aboutToQuit,
    # then return an available update so the result slot WOULD fire if it were
    # left connected.
    monkeypatch.setattr(
        overlay_main, "check_for_update",
        lambda current: (gate.wait(timeout=10), ("9.9.9", "https://example/rel"))[1],
    )

    overlay = overlay_main.build_overlay(["--url", "ws://127.0.0.1:1"])
    thread = overlay.update_thread
    assert thread is not None, (
        "build_overlay did not start the update-check thread on the enabled path"
    )
    # Observe the result slot: after release disconnects updateAvailable, this
    # must NOT fire even though the worker completes with an available update.
    thread.updateAvailable.connect(lambda v, u: notified.append((v, u)))
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    overlay.app.aboutToQuit.emit()

    try:
        assert thread.parent() is None, (
            "aboutToQuit did not detach the running update-check worker; the "
            "_release_update_thread wiring in build_overlay is missing"
        )
        assert thread_keeper.any_running(), (
            "the detached update-check worker was not parked in thread_keeper"
        )
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)
    qtbot.wait(100)  # drain any queued updateAvailable delivery
    assert notified == [], (
        "a detached update-check worker still delivered updateAvailable after "
        "release disconnected the slot"
    )


def test_release_update_thread_is_noop_when_absent_or_finished(_update_env):
    """_release_update_thread must no-op cleanly when there is no thread (the
    check was disabled by config -> update_thread is None) or the worker has
    already finished, rather than raising -- aboutToQuit fires on every quit
    path, including the common case where no check is in flight.

    Seeded-defect drill: drop the `thread is None or not thread.isRunning()`
    guard -> the None call raises AttributeError on `.isRunning()` -> red.
    """
    overlay_main = _update_env
    # None (feature disabled) must be a silent no-op.
    overlay_main._release_update_thread(None)

    # A constructed-but-never-started thread reports not-running; releasing it
    # must also be a no-op (nothing to detach).
    thread = overlay_main._UpdateCheckThread("0.0.0")
    assert not thread.isRunning()
    overlay_main._release_update_thread(thread)
    from yamabiko.ui import thread_keeper
    assert not thread_keeper.any_running(), (
        "_release_update_thread parked a thread that was never running"
    )
