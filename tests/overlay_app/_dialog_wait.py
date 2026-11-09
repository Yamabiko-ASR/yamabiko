"""Bounded waits for tests that drive a modal dialog from inside its exec loop.

A test that clicks through a dialog from timer callbacks must never hang the
run when the dialog misbehaves. :func:`when` polls for a condition until a
time deadline; :func:`exec_with_watchdog` rejects a dialog still open after
its watchdog. Keep the polling deadline shorter than the watchdog so no poll
outlives the dialog.
"""
from __future__ import annotations

import time


def when(ready, action, *, within_s: float = 8.0, _deadline: float | None = None) -> None:
    """Run ``action`` once ``ready()`` holds, polling every 20 ms for at most
    ``within_s`` seconds from this call."""
    from PySide6.QtCore import QTimer

    deadline = time.monotonic() + within_s if _deadline is None else _deadline
    if ready():
        action()
    elif time.monotonic() < deadline:
        QTimer.singleShot(20, lambda: when(ready, action, _deadline=deadline))


def exec_with_watchdog(dialog, *, watchdog_ms: int = 10_000) -> int:
    """``dialog.exec()``, rejected after ``watchdog_ms`` if still open, so a
    regression fails the test instead of hanging the run. The timer is the
    dialog's child: a pending ``QTimer.singleShot(ms, dialog.reject)`` would
    hold the dialog alive into a later test."""
    from PySide6.QtCore import QTimer

    watchdog = QTimer(dialog)
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(dialog.reject)
    watchdog.start(watchdog_ms)
    try:
        return dialog.exec()
    finally:
        watchdog.stop()
