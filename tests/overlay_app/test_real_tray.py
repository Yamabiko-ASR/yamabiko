"""Real Windows platform: the tray's flyout and short menu (@pytest.mark.real).

The new call sites: QSystemTrayIcon.activated opening the flyout at
tray_placement.place_flyout(QSystemTrayIcon.geometry(), ...) and the short
menu's QMenu.popup at tray_placement.place_menu(...), the real cursor read
against the icon's geometry for the Trigger swallow, the short menu's QSS
rows and Lit selection under the native windows11 style, DWM's corner
preference on both menus (window_chrome.SquareCorners), and Task 13's
QSystemTrayIcon.showMessage(title, body, QIcon, ms) with the shipped app icon
and its messageClicked routing, and the Settings window a notice's click
opens taking the foreground from another process
(window_chrome.take_foreground). The offscreen QPA has no tray icon rect, no
cursor and not the native style.
SKIPS unless the native windows QPA is available (run this file in its own
invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_tray.py -m real -v

The swallow and right-click tests move the mouse pointer onto the tray icon
and put it back; the right-click test also clicks a window of its own; the
notice test's helper process clicks its own window and puts the pointer back.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

from tests.overlay_app._foreground_owner import foreground_hwnd, foreground_owner
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


@pytest.fixture(autouse=True)
def _drain_qt_objects():
    yield
    drain_qt_debris()


def _wait_for(predicate, timeout_ms: int) -> bool:
    from PySide6.QtTest import QTest

    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if predicate():
            return True
        QTest.qWait(20)
    return bool(predicate())


def _tray(**kwargs):
    """A real Tray (real QSystemTrayIcon, real flyout) on a recording client
    with real signals. Returns (tray, client); keep both referenced."""
    from PySide6.QtCore import QObject, Signal

    from yamabiko.overlay_app.tray import Tray

    class _Client(QObject):
        stateReceived = Signal(str, bool)
        styleReceived = Signal(dict)
        connectedChanged = Signal(bool)
        seriesReceived = Signal(object, str, str)
        noticeReceived = Signal(str, str, str)

        def __init__(self):
            super().__init__()
            self.sent: list[dict] = []

        def send_command(self, cmd: dict) -> bool:
            self.sent.append(cmd)
            return True

    client = _Client()
    return Tray(client, on_quit=lambda: None, **kwargs), client


def test_real_left_click_opens_the_flyout_at_the_icon_and_its_closing_click_stays_closed(real_app):
    """Native tray: Trigger opens the flyout at place_flyout(icon rect), inside
    the work area of the icon's screen; with the real cursor on the icon, a
    hide (the click on the icon that closes the flyout) swallows that
    click's Trigger.

    Seeded defects: drop the activated connect, place from the cursor instead
    of the icon rect, or test the cursor against the icon rect in device
    pixels -> red."""
    from PySide6.QtGui import QCursor, QGuiApplication
    from PySide6.QtWidgets import QSystemTrayIcon

    from yamabiko.overlay_app import tray_placement

    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("no notification area in this session")
    tray, _client = _tray()
    flyout = tray._flyout
    saved = QCursor.pos()
    try:
        assert _wait_for(lambda: tray._tray.geometry().isValid(), 5000), (
            "the shell never reported the tray icon's rect")
        icon = tray._tray.geometry()
        tray._tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)
        assert _wait_for(flyout.isVisible, 2000), "Trigger did not open the flyout"
        assert flyout.pos() == tray_placement.place_flyout(icon, flyout.size())
        screen = QGuiApplication.screenAt(icon.center()) or QGuiApplication.primaryScreen()
        assert screen.availableGeometry().contains(flyout.frameGeometry()), (
            f"flyout {flyout.frameGeometry()} leaves the work area {screen.availableGeometry()}")
        QCursor.setPos(icon.center())
        assert _wait_for(lambda: tray_placement.cursor_in(icon), 1000)
        flyout.hide()
        tray._tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)
        _wait_for(lambda: False, 200)
        assert not flyout.isVisible(), "the click that closed the flyout reopened it"
    finally:
        QCursor.setPos(saved)
        flyout.hide()
        tray._tray.hide()


def test_real_short_menu_rows_are_30_px_and_the_hovered_row_is_lit(real_app):
    """Native windows11 style under the menu QSS (Decision 62): 30 px rows, at
    least 224 px wide, and the active row filled with the accent.

    Seeded defects: drop min-height, the minimum width, or the Lit
    :selected rule -> red."""
    from PySide6.QtCore import QPoint

    from yamabiko.overlay_app.theme import MIST

    tray, _client = _tray(palette=MIST, shortcuts={"pause_resume": "Ctrl+Alt+P"})
    menu = tray._menu
    try:
        menu.popup(QPoint(300, 300))
        assert _wait_for(menu.isVisible, 2000), "the short menu did not show"
        menu.setActiveAction(tray._pause_action)
        _wait_for(lambda: False, 100)
        row = menu.actionGeometry(tray._pause_action)
        assert abs(row.height() - 30) <= 1, f"menu row is {row.height()} px, the design is 30"
        assert menu.width() >= 224, f"menu is {menu.width()} px wide"
        image = menu.grab().toImage()
        dpr = image.devicePixelRatio()
        sample = image.pixelColor(int((row.left() + 6) * dpr), int(row.center().y() * dpr))
        assert sample.name() == MIST["accent"], (
            f"the active row is {sample.name()}, not the accent {MIST['accent']}")
    finally:
        menu.hide()
        tray._tray.hide()


def test_real_right_click_pops_the_short_menu_on_the_taskbar_line_and_a_click_elsewhere_closes_it(
    real_app,
):
    """Native tray (screenshots 13 and 14): with the real cursor on the icon,
    Context pops the short menu up at place_menu(icon rect, the size it
    shows at), inside the work area of the icon's screen, its
    taskbar-facing side on the taskbar's line; a click on another of the
    app's windows (one the test owns) closes it. A click on another
    process's window closing it was checked by hand on the real shell.

    Seeded defects: pop the menu at the cursor (Qt's own placement, over the
    taskbar), place it for a size it does not show at, or drop the Context
    popup -> red."""
    import ctypes
    from ctypes import wintypes

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QCursor, QGuiApplication
    from PySide6.QtWidgets import QSystemTrayIcon, QWidget

    from yamabiko.overlay_app import tray_placement

    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("no notification area in this session")
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    tray, _client = _tray()
    menu = tray._menu
    # The window the click lands on: the test's own, shown before the menu.
    other = QWidget(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    saved = QCursor.pos()
    try:
        assert _wait_for(lambda: tray._tray.geometry().isValid(), 5000), (
            "the shell never reported the tray icon's rect")
        icon = tray._tray.geometry()
        screen = QGuiApplication.screenAt(icon.center()) or QGuiApplication.primaryScreen()
        whole, work = screen.geometry(), screen.availableGeometry()
        other.setGeometry(work.x() + 80, work.y() + 80, 240, 120)
        other.show()
        assert _wait_for(other.isVisible, 2000)
        QCursor.setPos(icon.center())
        assert _wait_for(lambda: tray_placement.cursor_in(icon), 1000)
        tray._tray.activated.emit(QSystemTrayIcon.ActivationReason.Context)
        assert _wait_for(menu.isVisible, 2000), "Context did not pop the short menu up"
        placed = menu.geometry()
        assert placed.topLeft() == tray_placement.place_menu(icon, menu.size()), (
            f"menu at {placed}, not where place_menu puts a menu of its size")
        assert work.contains(placed), f"menu {placed} leaves the work area {work}"
        edge = tray_placement.taskbar_edge(
            (whole.x(), whole.y(), whole.width(), whole.height()),
            (work.x(), work.y(), work.width(), work.height()),
            (icon.center().x(), icon.center().y()))
        hidden = whole == work  # an auto-hidden taskbar: its line is the icon's side
        on_line = {
            "bottom": placed.bottom() == (icon.top() - 1 if hidden else work.bottom()),
            "top": placed.top() == (icon.bottom() + 1 if hidden else work.top()),
            "left": placed.left() == (icon.right() + 1 if hidden else work.left()),
            "right": placed.right() == (icon.left() - 1 if hidden else work.right()),
        }
        assert on_line[edge], f"menu {placed} is off the {edge} taskbar's line (work area {work})"

        QCursor.setPos(other.mapToGlobal(other.rect().center()))
        _wait_for(lambda: False, 100)
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        under = user32.WindowFromPoint(point)
        assert under and int(user32.GetAncestor(under, 2)) == int(other.winId()), (
            "the test's window is not under the cursor; nothing was clicked")
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
        user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
        assert _wait_for(lambda: not menu.isVisible(), 2000), (
            "a click on another of the app's windows left the tray menu open")
    finally:
        QCursor.setPos(saved)
        menu.hide()
        other.close()
        tray._tray.hide()


def test_real_menus_keep_square_corners_on_every_show(real_app):
    """Screenshots 05, 06, 13 and 14: the right-click menu and the flyout's
    Troubleshooting menu are nearly square, but the windows11 style asks DWM
    for round corners on every show. DWM reads back DWMWCP_DONOTROUND for
    both menus on their first and second show.

    Seeded defects: drop SquareCorners from either menu (Qt's DWMWCP_ROUND,
    2, reads back), or set the preference once after the first show (the
    second show reads 2) -> red."""
    import ctypes

    from PySide6.QtCore import QPoint

    tray, _client = _tray()
    menus = {"right-click": tray._menu, "Troubleshooting": tray._flyout.troubleshooting_menu}
    try:
        for name, menu in menus.items():
            for show in ("first", "second"):
                menu.popup(QPoint(300, 300))
                assert _wait_for(menu.isVisible, 2000), f"the {name} menu did not show"
                corner = ctypes.c_int(-1)
                hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                    ctypes.c_void_p(int(menu.winId())), 33,  # DWMWA_WINDOW_CORNER_PREFERENCE
                    ctypes.byref(corner), ctypes.sizeof(corner)) & 0xFFFFFFFF
                assert hr == 0, f"DwmGetWindowAttribute failed: 0x{hr:08x}"
                assert corner.value == 1, (  # DWMWCP_DONOTROUND
                    f"the {name} menu's {show} show has corner preference {corner.value}")
                menu.hide()
                assert _wait_for(lambda m=menu: not m.isVisible(), 2000)
    finally:
        for menu in menus.values():
            menu.hide()
        tray._tray.hide()


def test_real_notification_shows_a_banner_and_its_click_reaches_its_handler(real_app):
    """Native notification (Decisions 16 and 24): show_toast with the
    shipped app icon puts up a Windows banner (the top-level "New
    notification" window uncloaked with a nonempty rect, PlatformProbe Q4),
    and a click on the banner runs that toast's handler.

    Seeded defects: no showMessage call, or drop the messageClicked connect
    -> red. When the click misses, the "Yamabiko test" entry leaves the
    notification center with the icon the test hides (p4_toast_exit.py)."""
    import ctypes
    from ctypes import wintypes

    from PySide6.QtGui import QCursor
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QSystemTrayIcon

    if not (QSystemTrayIcon.isSystemTrayAvailable() and QSystemTrayIcon.supportsMessages()):
        pytest.skip("no notification area or no notifications in this session")
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.FindWindowExW.argtypes = (wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowExW.restype = wintypes.HWND
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    dwmapi = ctypes.WinDLL("dwmapi")
    dwmapi.DwmGetWindowAttribute.argtypes = (
        wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
    dwmwa_cloaked = 14

    def _banners() -> set[int]:
        """Top-level "New notification" windows (the Windows 11 notification
        banner host) showing a banner. The host persists between banners as
        a visible window, cloaked by the shell with an empty rect (measured
        2026-10-31), so a banner is the host uncloaked with a nonempty rect."""
        found: set[int] = set()
        h = None
        while True:
            h = user32.FindWindowExW(None, h, None, "New notification")
            if not h:
                return found
            cloaked = wintypes.DWORD(0)
            dwmapi.DwmGetWindowAttribute(h, dwmwa_cloaked, ctypes.byref(cloaked),
                                         ctypes.sizeof(cloaked))
            rect = wintypes.RECT()
            user32.GetWindowRect(h, ctypes.byref(rect))
            if (user32.IsWindowVisible(h) and not cloaked.value
                    and rect.bottom > rect.top and rect.right > rect.left):
                found.add(int(h))

    tray, _client = _tray()
    clicked: list[bool] = []
    saved = QCursor.pos()
    try:
        assert _wait_for(lambda: tray._tray.geometry().isValid(), 5000), (
            "the shell never reported the tray icon's rect")
        # The banner host is shared and persistent: a banner already up
        # (another app's, or an earlier test's) would pass for this one.
        if not _wait_for(lambda: not _banners(), 10000):
            pytest.skip("another notification banner stayed up for 10 s; "
                        "this test needs the banner host to itself")
        tray.show_toast("Yamabiko test", "A real-tier notification check.", 8000,
                        on_click=lambda: clicked.append(True))
        assert _wait_for(_banners, 8000), (
            "no 'New notification' banner appeared within 8 s: notifications for "
            "Python must be on and Do not disturb off")
        hwnd = next(iter(_banners()))
        # The banner slides in after its host uncloaks; PlatformProbe clicked
        # it about a second after showMessage.
        QTest.qWait(800)
        assert hwnd in _banners(), "the banner left before the test could click it"
        rect = wintypes.RECT()
        user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
        # The banner's text area (left of center, below the header with its
        # buttons), as PlatformProbe clicked it.
        user32.SetCursorPos((rect.left + rect.right) // 2 - 60, (rect.top + rect.bottom) // 2 + 20)
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
        user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
        assert _wait_for(lambda: clicked, 3000), "a click on the banner did not reach its handler"
    finally:
        QCursor.setPos(saved)
        tray._tray.hide()


def test_real_settings_opened_from_a_notice_takes_the_foreground(real_app, monkeypatch, tmp_path):
    """Ledger D5 (tray visual check 2): a click on the "Shortcut
    unavailable" notification reaches the shell, not this process, so the
    tray opens Settings at Shortcuts without the right to the foreground and
    Settings opened behind the app in front. _open_settings takes the
    foreground (window_chrome.take_foreground) between raise_() and
    activateWindow(). Here another process's window holds the foreground (it
    clicked itself last); the tray shows the notice and its toast's click
    handler runs as messageClicked runs it, with no input from this process.
    Settings is then the foreground window, at Shortcuts, with the keyboard.

    Seeded defect: drop take_foreground from _open_settings -> the other app
    keeps the foreground -> red. Skips only when this session never gives
    the helper's window the foreground."""
    from PySide6.QtWidgets import QApplication

    from yamabiko.overlay_app.settings_window import PAGE_NAMES

    # Real tier: the app-data isolation exempts it, so Settings reads this
    # test's config, never the user's.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))
    tray, _client = _tray()
    try:
        with foreground_owner() as owner_hwnd:
            tray._on_pipeline_notice(
                "Shortcut unavailable",
                "Ctrl+Alt+S is used by another app. Pick a new one in Settings.", "shortcuts")
            tray._tray.messageClicked.emit()  # the shell reports the click on the toast
            settings = tray._settings_window
            assert settings is not None, "the notice's click opened no Settings window"
            hwnd = int(settings.winId())
            assert _wait_for(lambda: foreground_hwnd() == hwnd, 2000), (
                f"the foreground is {foreground_hwnd():#x}, not Settings {hwnd:#x} "
                f"(the other app {owner_hwnd:#x})")
            assert settings._stack.currentIndex() == PAGE_NAMES.index("shortcuts"), (
                "Settings did not open at Shortcuts")
            assert _wait_for(lambda: settings.isActiveWindow()
                             and QApplication.focusWindow() is settings.windowHandle(), 2000), (
                "Settings is the foreground window but does not have the keyboard")
    finally:
        if tray._settings_window is not None:
            tray._settings_window.close()
        tray._tray.hide()
