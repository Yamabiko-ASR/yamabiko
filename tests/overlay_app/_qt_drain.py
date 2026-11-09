"""Deterministic Qt-object teardown helper for the real-QThread tests.

Several overlay tests spin real ``QThread``s that drive the download worker
(``tests/overlay_app/test_model_fetch.py``, the quant-download cases in
``test_settings_window.py``, and the tray quit-release case in
``test_tray.py``). Each leaves Qt wrappers (the finished ``QThread``, its
worker, the modal dialog) that Python's garbage collector reclaims at an
arbitrary later moment.

If that moment lands during ANOTHER test's pytest-qt teardown event pump --
notably next to ``test_run_overlay_shutdown_waiter``'s native waiter thread --
the wrapper destruction races Qt's event processing and the interpreter takes
a native access violation (a frameless-thread crash surfacing in
``pytestqt.plugin._process_events``). The overlay+wizard suite already runs
near that edge; a single extra real-QThread test tips it.

Calling :func:`drain_qt_debris` in each such test's teardown collects the
garbage and pumps DeferredDelete NOW, so every test's Qt debris is destroyed
inside its own scope rather than leaking into a more fragile later one.
"""
from __future__ import annotations

import gc

# QEvent.DeferredDelete numeric value (52). Referenced numerically so the
# helper needs no QtCore enum import at module load.
_DEFERRED_DELETE = 52


def drain_qt_debris() -> None:
    """Force pending Qt-object destruction to happen now.

    Pump queued events, run a GC pass (dropping the last Python refs to any
    finished QThread / worker / dialog wrappers), then pump again plus an
    explicit DeferredDelete flush so their C++ objects are destroyed within the
    calling test's teardown rather than a later one's.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    gc.collect()
    if app is not None:
        app.processEvents()
        app.sendPostedEvents(None, _DEFERRED_DELETE)
        app.processEvents()
