"""Real Windows platform: the tray flyout's window (@pytest.mark.real).

The new call sites: a Qt Tool | Frameless | StaysOnTop window shown and
activated from code, DwmSetWindowAttribute(DWMWA_WINDOW_CORNER_PREFERENCE,
DWMWCP_ROUNDSMALL) with SWP_FRAMECHANGED (window_chrome.set_corner_preference),
DWMWA_BORDER_COLOR on that window, the native WindowDeactivate the panel
hides on, the Qt screen list (tray_flyout.live_screens),
tray_placement.place_flyout's QCursor / QGuiApplication.screenAt /
availableGeometry reads, and the Position list under the native windows11
style, placed by Qt against the whole screen and moved into the work area
before its native show. The offscreen QPA has none of them. SKIPS unless
the native windows QPA is available (run this file in its own invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_tray_flyout.py -m real -v
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

import pytest

from tests.overlay_app._screen_grab import screen_pixel

pytestmark = pytest.mark.real

_S_OK = 0
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWCP_ROUNDSMALL = 3


@pytest.fixture(scope="module")
def real_app():
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None and sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    app = QApplication.instance() or QApplication([])
    if app.platformName() != "windows":
        pytest.skip("native windows QPA unavailable; run this file in its own invocation")
    yield app


def _settle(ms: int = 400) -> None:
    from PySide6.QtTest import QTest

    QTest.qWait(ms)


def _foreground_hwnd() -> int:
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetForegroundWindow.restype = wintypes.HWND
    return user32.GetForegroundWindow() or 0


def _open_flyout(palette=None, top_left=None):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QGuiApplication

    from yamabiko.overlay_app.theme import MIST
    from yamabiko.overlay_app.tray_flyout import TrayFlyout

    flyout = TrayFlyout(palette or MIST)
    work = QGuiApplication.primaryScreen().availableGeometry()
    flyout.open_at(top_left or QPoint(work.x() + 120, work.y() + 80))
    _settle()
    return flyout


def _five_positions():
    from yamabiko.overlay_app.tray_flyout import PositionEntry

    return [PositionEntry("Display 1"),
            PositionEntry("Bottom of Display 1", "DISPLAY-A", 88.0),
            PositionEntry("Middle of Display 1", "DISPLAY-A", 55.0),
            PositionEntry("Bottom of Display 2", "DISPLAY-B", 88.0),
            PositionEntry("Middle of Display 2", "DISPLAY-B", 55.0)]


def test_real_flyout_gets_small_corners_and_the_frame_edge(real_app):
    """After open_at, DWM reports DWMWCP_ROUNDSMALL for the panel (read back
    with DwmGetWindowAttribute; a frameless window's own default reads 2) and
    paints its edge in the Mist frame tone.

    Seeded defects: skip set_corner_preference in open_at, or call it before
    show() -> the readback is the default 2 -> red; skip set_border_color ->
    the edge pixel is the system border -> red. (DWMWA_BORDER_COLOR is
    write-only, so the edge pixel is the observable.)"""
    flyout = _open_flyout()
    try:
        hwnd = ctypes.c_void_p(int(flyout.winId()))
        corner = ctypes.c_int(-1)
        hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, _DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(corner), ctypes.sizeof(corner),
        ) & 0xFFFFFFFF
        assert hr == _S_OK, f"DwmGetWindowAttribute(CORNER_PREFERENCE) failed: 0x{hr:08x}"
        assert corner.value == _DWMWCP_ROUNDSMALL, (
            f"corner preference {corner.value}; set_corner_preference did not take")

        frame = wintypes.RECT()
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(frame), ctypes.sizeof(frame))
        mid = (frame.top + frame.bottom) // 2
        edge = screen_pixel(frame.left, mid)
        assert max(abs(a - b) for a, b in zip(edge, (0xBD, 0xBD, 0xB9))) <= 12, (
            f"flyout edge {edge} is not the frame color #bdbdb9")
    finally:
        flyout.hide()
        flyout.deleteLater()


def test_real_flyout_opens_active_and_hides_when_another_window_activates(real_app):
    """open_at makes the panel the foreground window (Decision 7); when
    another window activates, the native WindowDeactivate hides the panel
    and dismissed fires once.

    Seeded defects: drop activateWindow from open_at -> the foreground stays
    on the other window -> red; drop the WindowDeactivate branch -> the panel
    stays visible -> red. Skips when this session cannot hand the
    foreground to its own windows at all (checked on a plain window first)."""
    from PySide6.QtWidgets import QWidget

    other = QWidget()
    other.resize(240, 120)
    other.move(600, 200)
    other.show()
    other.raise_()
    other.activateWindow()
    _settle()
    if _foreground_hwnd() != int(other.winId()):
        other.close()
        pytest.skip("this session cannot bring a test window to the foreground")
    flyout = _open_flyout()
    dismissed: list = []
    flyout.dismissed.connect(lambda: dismissed.append(True))
    try:
        assert _foreground_hwnd() == int(flyout.winId()), "open_at did not activate the panel"
        other.activateWindow()
        _settle()
        assert not flyout.isVisible(), "the panel stayed open after another window activated"
        assert dismissed == [True]
    finally:
        flyout.hide()
        flyout.deleteLater()
        other.close()


def test_real_live_screens_lists_every_qt_screen_with_one_primary(real_app):
    """The Position picker's screen list on the real platform: every Qt
    screen in order, exactly one marked primary (the `is` check relies on
    PySide6 handing back the same QScreen wrapper from screens() and
    primaryScreen()).
    Seeded defects: compare with `==` against a stale screen list, or flag
    no screen (or every screen) as primary -> red."""
    from PySide6.QtGui import QGuiApplication

    from yamabiko.overlay_app.tray_flyout import live_screens

    screens = live_screens()
    assert [name for name, _primary in screens] == [s.name() for s in QGuiApplication.screens()]
    assert [name for name, primary in screens if primary] == [
        QGuiApplication.primaryScreen().name()]


def test_real_place_flyout_lands_inside_the_icon_screen_work_area(real_app):
    """place_flyout's Qt reads on the real platform: for an icon rect at the
    bottom of each screen's work area the flyout lands on that screen
    (screenAt of its center), 12 px inside the work area; an invalid icon
    rect lands it on the cursor's screen.

    Seeded defects: look the screen up from the cursor instead of the icon
    (a second screen's row lands on the cursor's screen), or mix device
    pixels into the math (devicePixelRatio 1.5 on the main display moves
    the rect off its work area) -> red."""
    from PySide6.QtCore import QRect, QSize
    from PySide6.QtGui import QCursor, QGuiApplication

    from yamabiko.overlay_app.tray_placement import INSET, place_flyout

    size = QSize(360, 426)
    for screen in QGuiApplication.screens():
        work = screen.availableGeometry()
        icon = QRect(work.center().x() - 16, work.bottom() - 47, 32, 48)
        placed = QRect(place_flyout(icon, size), size)
        assert QGuiApplication.screenAt(placed.center()) is screen, (screen.name(), placed)
        assert work.adjusted(INSET, INSET, -INSET, -INSET).contains(placed), (
            screen.name(), placed, work)
    cursor_screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
    placed = QRect(place_flyout(QRect(), size), size)
    assert QGuiApplication.screenAt(placed.center()) is cursor_screen


def test_real_open_position_list_marks_the_current_row_lit(real_app):
    """Native windows11 style: the open Position list marks its current row
    Lit like the menus (Decision 62), the theme's accent fill in Mist and in
    Night. The style's own marker is a system-accent pill on the field.

    Seeded defect: drop the flyout list's ::item:selected rule -> the row's
    right end is the field -> red."""
    from PySide6.QtGui import QColor

    from yamabiko.overlay_app.theme import MIST, NIGHT

    for palette in (MIST, NIGHT):
        flyout = _open_flyout(palette)
        picker = flyout._position
        try:
            flyout.set_positions(_five_positions(), 0)
            picker.showPopup()
            _settle()
            view = picker.view()
            assert view.isVisible(), "the Position list did not open"
            image = view.viewport().grab().toImage()
            dpr = image.devicePixelRatio()
            y = view.visualRect(view.model().index(0, 0)).center().y()
            fill = image.pixelColor(int((view.viewport().width() - 6) * dpr), int(y * dpr))
            assert fill == QColor(palette["accent"]), (
                f"the current row is {fill.name()}, not the accent {palette['accent']}")
        finally:
            picker.hidePopup()
            flyout.hide()
            flyout.deleteLater()


def test_real_position_list_opens_inside_the_work_area(real_app):
    """Qt places a combo list against the whole screen on Windows: from the
    flyout 12 px above the taskbar (where place_flyout puts it) the five-row
    list ran over the taskbar. It opens above the picker instead, inside the
    work area, at the spot of its native window.

    Seeded defects: drop the Show filter's move, or move the list after its
    native show -> its window rect passes the work area -> red. (A screen
    whose work area reaches the screen bottom cannot show the defect.)"""
    import ctypes

    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QGuiApplication

    from yamabiko.overlay_app.tray_flyout import FLYOUT_HEIGHT
    from yamabiko.overlay_app.tray_placement import INSET

    work = QGuiApplication.primaryScreen().availableGeometry()
    flyout = _open_flyout(top_left=QPoint(work.x() + 120, work.bottom() + 1 - INSET - FLYOUT_HEIGHT))
    picker = flyout._position
    try:
        flyout.set_positions(_five_positions(), 0)
        picker.showPopup()
        _settle()
        container = picker.view().parentWidget()
        assert container.isVisible(), "the Position list did not open"
        rect = wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(wintypes.HWND(int(container.winId())), ctypes.byref(rect))
        dpr = container.devicePixelRatioF()
        bottom = rect.bottom / dpr  # exclusive, logical px
        assert bottom <= work.bottom() + 1, (
            f"the list's window ends at y {bottom}, past the work area's {work.bottom() + 1}")
        assert container.geometry().bottom() < picker.mapToGlobal(picker.rect().center()).y(), (
            "the list did not open above the picker")
    finally:
        picker.hidePopup()
        flyout.hide()
        flyout.deleteLater()
