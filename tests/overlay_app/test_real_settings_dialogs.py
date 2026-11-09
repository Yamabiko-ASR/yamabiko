"""Real Windows platform: Settings' compact dialogs (@pytest.mark.real).

The new call sites: a CompactDialog that grows while shown (the download
dialog's failed state), Copy details on the native clipboard and the
dialog's placement on Settings (native frame margins and work area). The
offscreen QPA has neither a native window nor the OS clipboard. SKIPS unless
the native windows QPA is available (run this file in its own invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_settings_dialogs.py -m real -v

Each clipboard test keeps the user's clipboard: every format it holds is
saved before and put back after.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes

import pytest

from tests.overlay_app._dialog_wait import exec_with_watchdog, when
from tests.overlay_app._qt_drain import drain_qt_debris

pytestmark = pytest.mark.real

_CF_UNICODETEXT = 13
_SENTINEL = "yamabiko test: clipboard not written"


@pytest.fixture(scope="module")
def real_app():
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None and sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    app = QApplication.instance() or QApplication([])
    if app.platformName() != "windows":
        pytest.skip("native windows QPA unavailable; run this file in its own invocation")
    yield app


@pytest.fixture
def clipboard(real_app):
    """The Qt clipboard, seeded with a sentinel. The user's clipboard, every
    format it holds (not only text), is put back after the test."""
    from PySide6.QtCore import QMimeData

    board = real_app.clipboard()
    saved = QMimeData()
    current = board.mimeData()
    if current is not None:
        for fmt in current.formats():
            saved.setData(fmt, current.data(fmt))
        if current.hasImage():
            saved.setImageData(current.imageData())
    board.setText(_SENTINEL)
    yield board
    board.setMimeData(saved)


def _native_clipboard_text() -> str:
    """CF_UNICODETEXT straight from the Win32 clipboard (not Qt's copy)."""
    user32 = ctypes.WinDLL("user32")  # private handles: argtypes stay local
    kernel32 = ctypes.WinDLL("kernel32")
    user32.OpenClipboard.argtypes = (wintypes.HWND,)
    user32.GetClipboardData.argtypes = (wintypes.UINT,)
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    for _ in range(50):  # another process may hold the clipboard briefly
        if user32.OpenClipboard(None):
            break
        time.sleep(0.02)
    else:
        pytest.fail("the clipboard stayed open in another process")
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return ""
        pointer = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _client_height(widget) -> float:
    """The Win32 client height of ``widget``'s window in logical px."""
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(wintypes.HWND(int(widget.winId())), ctypes.byref(rect))
    return rect.bottom / widget.devicePixelRatioF()


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def _visible_bounds(widget) -> wintypes.RECT:
    """The visible window rect (DWMWA_EXTENDED_FRAME_BOUNDS), device px."""
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(int(widget.winId())), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    return rect


def _monitor_api():
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
    user32.MonitorFromPoint.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo))
    return user32


def _monitor_of(user32, widget) -> _MonitorInfo:
    """The monitor holding most of ``widget``'s window (device px)."""
    info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    user32.GetMonitorInfoW(user32.MonitorFromWindow(wintypes.HWND(int(widget.winId())), 2),
                           ctypes.byref(info))  # MONITOR_DEFAULTTONEAREST
    return info


def test_real_download_failure_grows_the_dialog_and_copies_the_details(real_app, clipboard):
    """The download fails while the dialog is on screen: the native window
    grows to hold the two failure lines and the buttons, Copy details puts
    the error on the OS clipboard, and Cancel from there settles "failed".

    Seeded defects: drop the refit in _show_status -> the window keeps its
    pre-failure height -> the status lines fall outside the client -> red;
    leave Copy details unwired -> the clipboard keeps the sentinel -> red."""
    from PySide6.QtCore import QPoint, QRect, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialog

    from yamabiko.overlay_app.model_fetch import QuantDownloadDialog, spec_for_tier
    from yamabiko.overlay_app.theme import MIST

    release = threading.Event()

    class _FailsOnRelease:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            release.wait(timeout=10.0)
            raise RuntimeError("404 Client Error: file not found")

    dialog = QuantDownloadDialog(spec_for_tier("small"), "small",
                                 downloader=_FailsOnRelease(), palette=MIST)
    seen: dict = {}

    def _rect_in_dialog(widget) -> QRect:
        return QRect(widget.mapTo(dialog, QPoint(0, 0)), widget.size())

    def _before_failure() -> None:
        seen["before"] = dialog.height()
        seen["client_before"] = _client_height(dialog)
        release.set()
        when(dialog.retry_button.isVisible, lambda: QTimer.singleShot(300, _in_failed_state),
             within_s=8.0)

    def _in_failed_state() -> None:
        try:
            seen["after"] = dialog.height()
            seen["client_after"] = _client_height(dialog)
            seen["dialog_rect"] = dialog.rect()
            seen["status_rect"] = _rect_in_dialog(dialog.status)
            seen["status_needs"] = dialog.status.heightForWidth(dialog.status.width())
            seen["retry_rect"] = _rect_in_dialog(dialog.retry_button)
            QTest.mouseClick(dialog.copy_button, Qt.LeftButton)
            seen["copied"] = _native_clipboard_text()
        finally:
            QTest.mouseClick(dialog.cancel_button, Qt.LeftButton)

    # Polling (5 s, then 8 s, plus two 300 ms settles) ends before the watchdog.
    when(dialog.isVisible, lambda: QTimer.singleShot(300, _before_failure), within_s=5.0)
    try:
        result = exec_with_watchdog(dialog, watchdog_ms=15_000)
    finally:
        release.set()
        dialog.release_threads()
        outcome = dialog.outcome
        dialog.deleteLater()
        drain_qt_debris()

    assert "copied" in seen, f"the failed state never showed: {seen}"
    assert seen["after"] > seen["before"], f"the dialog did not grow: {seen}"
    assert seen["client_after"] > seen["client_before"], f"the native window did not grow: {seen}"
    status, retry = seen["status_rect"], seen["retry_rect"]
    assert seen["dialog_rect"].contains(status) and seen["dialog_rect"].contains(retry), seen
    assert status.height() >= seen["status_needs"], f"the failure lines are clipped: {seen}"
    assert retry.bottom() < seen["client_after"], f"the buttons sit below the client: {seen}"
    assert "RuntimeError: 404 Client Error: file not found" in seen["copied"], seen["copied"]
    assert result == QDialog.Rejected and outcome == "failed"


def test_real_save_failed_copy_details_puts_the_hidden_path_on_the_clipboard(real_app, clipboard):
    """Copy details puts the path and the error on the OS clipboard with the
    user's %LOCALAPPDATA% folder written as the variable, in both the plain
    path and the error's repr of it.

    Seeded defects: copy the raw path -> the user folder is on the clipboard
    -> red; leave Copy details unwired -> the clipboard keeps the sentinel ->
    red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog
    from yamabiko.overlay_app.theme import MIST

    local = os.environ["LOCALAPPDATA"]
    path = os.path.join(local, "Yamabiko", "user_overrides.yaml")
    dialog = SaveFailedDialog(MIST, path, PermissionError(13, "Access is denied", path))
    try:
        dialog.show()
        QTest.qWaitForWindowExposed(dialog)
        QTest.mouseClick(dialog.copy_button, Qt.LeftButton)
        copied = _native_clipboard_text()
    finally:
        dialog.close()
        dialog.deleteLater()
        drain_qt_debris()

    assert copied.startswith("%LOCALAPPDATA%\\Yamabiko\\user_overrides.yaml"), copied
    assert "PermissionError: [Errno 13] Access is denied: '%LOCALAPPDATA%\\\\Yamabiko" in copied, copied
    lowered = copied.lower()
    assert local.lower() not in lowered and local.replace("\\", "\\\\").lower() not in lowered, copied


def test_real_compact_dialog_opens_centered_on_settings_inside_the_work_area(real_app):
    """Native frame margins. Settings' expanded client area reports a
    negative left frame margin, so QDialog guessed a 10 x 40 px frame and
    opened the dialogs about 40 px above Settings' center; and Qt's frame
    rect of an expanded-client dialog lies inside its visible window. The
    two visible windows share their center, and with Settings pushed past a
    monitor edge that no other monitor touches, the dialog's visible window
    stays inside that monitor's work area, give or take 3 device px (the
    2 device px strip inside the DWM border, and rounding).

    Seeded defects: drop _center_on_parent from CompactDialog.showEvent ->
    about (-12, -40) logical px -> red; drop its clamp -> the dialog hangs
    off the edge -> red; clamp Qt's frame rect alone -> 5 device px past the
    left edge -> red."""
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog
    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.theme import MIST

    user32 = _monitor_api()
    avail = QGuiApplication.primaryScreen().availableGeometry()
    w = SettingsWindow(dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       hotkey_probe=lambda combo: False)
    dialog = SaveFailedDialog(MIST, "C:/x/user_overrides.yaml",
                              PermissionError(13, "Access is denied"), w)
    past = None
    try:
        w.move(avail.left() + 40, avail.top() + 40)
        w.show()
        QTest.qWait(300)
        dialog.show()
        QTest.qWait(300)
        outer, inner = _visible_bounds(w), _visible_bounds(dialog)
        ratio = w.devicePixelRatioF()
        offset = ((inner.left + inner.right - outer.left - outer.right) / 2 / ratio,
                  (inner.top + inner.bottom - outer.top - outer.bottom) / 2 / ratio)
        dialog.close()
        screen = _monitor_of(user32, w).rcMonitor
        mid = wintypes.POINT((screen.left + screen.right) // 2, (screen.top + screen.bottom) // 2)
        edges = (  # a point just past each edge; where Settings goes to hang past it
            (wintypes.POINT(screen.left - 1, mid.y), (avail.left() - 798, avail.top() + 100)),
            (wintypes.POINT(screen.right, mid.y), (avail.right() - 200, avail.top() + 100)),
            (wintypes.POINT(mid.x, screen.bottom), (avail.left() + 100, avail.bottom() - 100)),
            (wintypes.POINT(mid.x, screen.top - 1), (avail.left() + 100, avail.top() - 600)),
        )
        past = next((pos for point, pos in edges
                     if not user32.MonitorFromPoint(point, 0)), None)  # MONITOR_DEFAULTTONULL
        if past is not None:
            w.move(*past)
            QTest.qWait(300)
            dialog.show()
            QTest.qWait(300)
            bounds = _visible_bounds(dialog)
            work = _monitor_of(user32, w).rcWork
    finally:
        dialog.close()
        dialog.deleteLater()
        w.close()
        drain_qt_debris()

    assert abs(offset[0]) <= 2 and abs(offset[1]) <= 2, f"dialog center - Settings center: {offset}"
    if past is None:
        pytest.skip("every edge of this monitor touches another monitor")
    slack = 3
    inside = (bounds.left >= work.left - slack and bounds.top >= work.top - slack
              and bounds.right <= work.right + slack and bounds.bottom <= work.bottom + slack)
    assert inside, ((bounds.left, bounds.top, bounds.right, bounds.bottom),
                    (work.left, work.top, work.right, work.bottom))
