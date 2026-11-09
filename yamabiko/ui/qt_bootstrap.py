"""HiDPI-aware QApplication bootstrap shared by the wizard and the startup
error dialog, plus Qt-diagnostics routing for every Yamabiko Qt process.

The HiDPI scale-factor rounding policy MUST be set BEFORE the first
QApplication is constructed -- Qt ignores it afterward. Centralizing it
here guarantees every Yamabiko Qt entry point applies PassThrough.
Without it, only the overlay would, so the first-run wizard and the
startup error dialog would render with Qt's default rounding on
fractional-DPI (125% / 150%) Windows laptops, common hardware for this
product.

Qt's own diagnostics (qWarning / qCritical / plugin and platform
complaints) default to stderr, which is a null stream in the windowed
frozen bundle -- they vanish exactly when a beta user hits them.
``install_qt_message_handler`` reroutes them to the ``yamabiko.qt``
logger so they land in the per-process ``main-*.log`` (2026-07-01
logging audit, Task 4). It is importable on its own so entry points
that build their QApplication directly (the overlay) can adopt the
routing with a single call, without switching to
``ensure_qapplication``.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

# Qt severity -> stdlib logging level. Fatal maps to CRITICAL; any
# QtMsgType member this map has never heard of (a future Qt addition)
# falls back to WARNING in the handler so it is surfaced, never dropped.
_QT_LEVEL = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}

# Module-global so install_qt_message_handler() is idempotent: every Qt
# entry point calls it (some more than once per process, e.g. wizard
# rerun after the error dialog), and the handler must be installed
# exactly once.
_qt_handler_installed = False
# The handler this module displaced (None = Qt's default writer). Kept so
# a test that must undo the process-global install can restore the EXACT
# previous handler -- under pytest-qt, that is the plugin's per-test
# capture handler, whose own save/restore bookkeeping a mid-test reset to
# None would otherwise invalidate.
_displaced_qt_handler = None


def _qt_message_handler(mode, context, message):
    """Route one Qt diagnostic to the yamabiko.qt logger at the mapped
    severity. ``context`` (file/line of the emitting C++ call site) is
    unused: it is compiled out of release Qt builds anyway."""
    logging.getLogger("yamabiko.qt").log(
        _QT_LEVEL.get(mode, logging.WARNING), "%s", message
    )


def install_qt_message_handler() -> None:
    """Install the qInstallMessageHandler routing exactly once per
    process. Safe to call before or after QApplication construction and
    from every Qt entry point; repeat calls are no-ops."""
    global _qt_handler_installed, _displaced_qt_handler
    if _qt_handler_installed:
        return
    _displaced_qt_handler = qInstallMessageHandler(_qt_message_handler)
    _qt_handler_installed = True


def ensure_qapplication() -> QApplication:
    """Return the process QApplication, creating one with the PassThrough
    HiDPI rounding policy applied if none exists yet. The policy is set
    before construction (Qt requires this) and is a no-op when an app
    already exists. Qt-diagnostics routing is installed first in either
    case, so messages Qt emits during construction -- and on re-entry
    paths where another entry point already built the app -- are logged."""
    install_qt_message_handler()
    app = QApplication.instance()
    if app is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication([])
    return app
