"""Real Windows: the tray icon's OS inputs (@pytest.mark.real).

The new call sites: user32 GetSystemMetricsForDpi(SM_CXSMICON), the winreg
read of SystemUsesLightTheme, and the QAbstractNativeEventFilter that sees
WM_SETTINGCHANGE on a real native window. The offscreen QPA delivers no
native messages. SKIPS unless the native windows QPA is available (run this
file in its own invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_tray_icons.py -m real -v
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes

import pytest

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


def test_real_small_icon_metric_follows_the_dpi(real_app):
    """GetSystemMetricsForDpi(SM_CXSMICON) is the size Windows shows tray
    icons at: 16 px at 100 %, 20 at 125 %, 24 at 150 %, 28 at 175 %, 32 at
    200 %.

    Seeded defects: query SM_CXICON (32 at 96 dpi), or ignore the dpi
    (GetSystemMetrics) -> red."""
    from yamabiko.overlay_app.tray_icons import small_icon_metric

    assert [small_icon_metric(dpi) for dpi in (96, 120, 144, 168, 192)] == [16, 20, 24, 28, 32]


def test_real_taskbar_theme_reads_system_uses_light_theme(real_app):
    """The taskbar color comes from SystemUsesLightTheme (Qt follows
    AppsUseLightTheme, which can differ). Checked against reg.exe's own read
    of the value; a missing value reads dark, as the code does.

    Seeded defects: invert the mapping, or read the wrong value on a
    machine where the two differ -> red."""
    from yamabiko.overlay_app.tray_icons import taskbar_theme

    out = subprocess.run(
        ["reg", "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
         "/v", "SystemUsesLightTheme"],
        capture_output=True, text=True,
    )
    expected = "light" if out.returncode == 0 and out.stdout.split()[-1] == "0x1" else "dark"
    assert taskbar_theme() == expected


def test_real_setting_change_reaches_the_watcher_through_a_native_window(real_app):
    """A real top-level Qt window (like the pre-created flyout) receives
    WM_SETTINGCHANGE and the watcher's native event filter re-reads the
    taskbar theme for "ImmersiveColorSet" only. The messages go to our own
    window, never broadcast, so the user's desktop is untouched.

    Seeded defects: compare the event type to the wrong name, read wParam
    instead of the lParam string, or never install the filter -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    from yamabiko.overlay_app.tray_icons import TaskbarThemeWatcher

    reads: list[str] = []

    def read() -> str:
        reads.append("read")
        return "dark" if len(reads) == 1 else "light"

    watcher = TaskbarThemeWatcher(read=read)
    seen: list[str] = []
    watcher.changed.connect(seen.append)
    window = QWidget(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    try:
        user32 = ctypes.WinDLL("user32")
        user32.SendMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                        wintypes.LPARAM)
        user32.SendMessageW.restype = wintypes.LPARAM
        hwnd = int(window.winId())
        for area in ("Policy", "ImmersiveColorSet"):
            text = ctypes.create_unicode_buffer(area)
            user32.SendMessageW(hwnd, 0x001A, 0, ctypes.addressof(text))
        assert len(reads) == 2 and seen == ["light"]
    finally:
        watcher.uninstall()
        window.deleteLater()
