"""Real Windows platform: the setup window (@pytest.mark.real).

The new call sites: run_wizard's dialog with Qt's ExpandedClientAreaHint /
CustomizeWindowHint flags, the minimize-only window menu (SetWindowLongW
after the native show), DWM's frame (DWMWA_EXTENDED_FRAME_BOUNDS) and the
painted caption's minimize button (QWidget.showMinimized on a native
window). The offscreen QPA has none of them. SKIPS unless the native windows
QPA is available (run this file in its own invocation):

    .venv\\Scripts\\python.exe -m pytest tests/test_real_wizard_window.py -m real -v
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

import pytest

from tests.overlay_app._dialog_wait import when
from tests.overlay_app._qt_drain import drain_qt_debris

pytestmark = pytest.mark.real


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
def own_app_data(monkeypatch, tmp_path):
    """Real tier: the app-data isolation exempts it. The wizard reads its
    theme and interim state and writes state.json under this test's folder,
    never the user's."""
    root = tmp_path / "Yamabiko"
    monkeypatch.setenv("YAMABIKO_APPDATA", str(root))
    return root


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def _user32():
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetSystemMenu.argtypes = (wintypes.HWND, wintypes.BOOL)
    user32.GetSystemMenu.restype = wintypes.HMENU
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
    user32.MonitorFromPoint.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo))
    return user32


def _visible_bounds(widget) -> wintypes.RECT:
    """The visible window rect (DWMWA_EXTENDED_FRAME_BOUNDS), device px."""
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(int(widget.winId())), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    return rect


def _primary_work_area() -> wintypes.RECT:
    user32 = _user32()
    info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    user32.GetMonitorInfoW(user32.MonitorFromPoint(wintypes.POINT(0, 0), 1),  # DEFAULTTOPRIMARY
                           ctypes.byref(info))
    return info.rcWork


def test_real_setup_window_is_880_by_600_with_a_minimize_only_caption(
    real_app, own_app_data, monkeypatch,
):
    """L14: run_wizard shows "Yamabiko setup" 880 x 600 logical px (the
    visible window at the primary screen's ratio, within 2 device px),
    centered on the primary work area (within 6 logical px, Decision 31;
    top-aligned on a work area under 600 px); Win32 sees the system menu
    and the minimize box and no maximize box; the painted caption ends with
    minimize then close at the client's right edge, and its minimize button
    minimizes the window. Cancel then ends run_wizard.

    Seeded defects: drop setFixedSize -> QDialog's own size -> red; keep the
    old title -> red; call add_window_menu(min_max=True) -> the maximize
    box -> red; drop ClientAreaFit -> close ends past the client -> red;
    leave the minimize button unwired -> never iconic -> red; skip
    center_on_work_area -> QDialog's placement, (-14.7, -43.3) px off at
    150 % -> red."""
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko import gpu
    from yamabiko import wizard as wizard_mod
    from yamabiko.overlay_app.window_chrome import client_width

    monkeypatch.setattr(gpu, "detect_vram_mib", lambda: None)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: None)
    seen: dict = {}
    held: list = []

    def shown():
        return next((w for w in QApplication.topLevelWidgets()
                     if isinstance(w, wizard_mod.YamabikoWizard) and w.isVisible()), None)

    def measure(window) -> None:
        held.append(window)
        user32 = _user32()
        hwnd = wintypes.HWND(int(window.winId()))

        def read() -> None:
            title = ctypes.create_unicode_buffer(64)
            user32.GetWindowTextW(hwnd, title, 64)
            seen["title"] = title.value
            seen["frame"] = _visible_bounds(window)
            seen["work"] = _primary_work_area()
            seen["style"] = user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
            seen["menu"] = bool(user32.GetSystemMenu(hwnd, False))
            minimize = window.caption.caption_button("min")
            close = window.caption.caption_button("close")
            seen["close_right"] = close.mapTo(window, QPoint(close.width(), 0)).x()
            seen["client"] = client_width(window)
            seen["adjacent"] = (minimize.mapTo(window, QPoint(minimize.width(), 0)).x()
                                == close.mapTo(window, QPoint(0, 0)).x())
            QTest.mouseClick(minimize, Qt.LeftButton)
            QTimer.singleShot(400, restore_and_cancel)

        def restore_and_cancel() -> None:
            seen["iconic"] = bool(user32.IsIconic(hwnd))
            window.showNormal()
            QTimer.singleShot(300, window.cancel_button.click)

        QTimer.singleShot(300, read)

    def close_everything() -> None:
        seen["watchdog"] = True
        window = shown()
        if window is not None:
            window.reject()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(close_everything)
    watchdog.start(20_000)
    when(lambda: shown() is not None, lambda: measure(shown()), within_s=18)
    try:
        wizard_mod.run_wizard()
    finally:
        watchdog.stop()
        held.clear()
        drain_qt_debris()
    assert "watchdog" not in seen, f"Cancel did not end run_wizard: {seen}"
    assert seen["title"] == "Yamabiko setup"
    ratio = QGuiApplication.primaryScreen().devicePixelRatio()
    frame, work = seen["frame"], seen["work"]
    width, height = frame.right - frame.left, frame.bottom - frame.top
    assert abs(width - round(880 * ratio)) <= 2 and abs(height - round(600 * ratio)) <= 2, (
        f"the window is {width} x {height} device px, not 880 x 600 at {ratio:g}x")
    dx = (frame.left + frame.right - work.left - work.right) / 2 / ratio
    if work.bottom - work.top >= height:
        dy = (frame.top + frame.bottom - work.top - work.bottom) / 2 / ratio
    else:
        dy = (frame.top - work.top) / ratio
    assert abs(dx) <= 6 and abs(dy) <= 6, f"off the work area's center by ({dx:.1f}, {dy:.1f})"
    style = seen["style"]
    assert style & 0x00080000 and style & 0x00020000 and seen["menu"], (
        f"no system menu or minimize box: style {style:#x}")
    assert not style & 0x00010000, f"the fixed-size window has a maximize box: {style:#x}"
    assert seen["close_right"] == seen["client"] and seen["adjacent"]
    assert seen["iconic"], "the caption's minimize button did not minimize the window"
