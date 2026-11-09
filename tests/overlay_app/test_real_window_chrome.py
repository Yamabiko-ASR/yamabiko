"""Real Windows platform: Settings' own caption (@pytest.mark.real).

The new call sites: Qt's ExpandedClientAreaHint / CustomizeWindowHint
window flags, QWindow.startSystemMove, Win32 GetClientRect, DWM
DwmSetWindowAttribute(DWMWA_BORDER_COLOR) and the WM_NCPAINT /
WM_NCACTIVATE answers (GetWindowDC, FillRect, DefWindowProc, RedrawWindow).
The offscreen QPA has none of them. SKIPS unless the native windows QPA is
available (run this file in its own invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_window_chrome.py -m real -v
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

import pytest

from tests.overlay_app._screen_grab import grab_screen, screen_pixel

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


def _chrome_window():
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from yamabiko.overlay_app.theme import MIST
    from yamabiko.overlay_app.window_chrome import (
        ClientAreaFit, TitleBand, add_window_menu, expand_client_area, set_border_color,
    )

    w = QWidget()
    expand_client_area(w)
    root = QVBoxLayout(w)
    root.setContentsMargins(0, 0, 0, 0)
    band = TitleBand("Yamabiko settings", "山彦", "Yamabiko settings", MIST)
    root.addWidget(band)
    root.addStretch(1)
    fit = ClientAreaFit(w, root)
    w.resize(1000, 728)
    w.move(160, 120)
    w.show()
    # What SettingsWindow.showEvent does on every show.
    assert set_border_color(w, MIST["frame"])
    assert add_window_menu(w, min_max=True)
    return w, band, fit


def _settle(ms: int = 400) -> None:
    from PySide6.QtTest import QTest

    QTest.qWait(ms)


def _foreground(w) -> None:
    """Raise the test window; skip when the session can't (an overlapping
    window would make the screen grab or the injected input land elsewhere).

    Qt's isActiveWindow() is no proof: under Windows' foreground lock (the
    user typed in another window moments ago) Qt reports the window active
    while it stays behind the foreground window. A topmost / not-topmost
    round trip raises it without foreground rights; then the client's
    corners and center must hit-test to this window."""
    w.raise_()
    w.activateWindow()
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, ctypes.c_uint)
    user32.GetAncestor.restype = wintypes.HWND
    hwnd = wintypes.HWND(int(w.winId()))
    for insert_after in (-1, -2):  # HWND_TOPMOST, then HWND_NOTOPMOST
        user32.SetWindowPos(hwnd, wintypes.HWND(insert_after), 0, 0, 0, 0,
                            0x0001 | 0x0002 | 0x0010)  # NOSIZE | NOMOVE | NOACTIVATE
    _settle()
    client = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    x0, y0 = _client_origin(w)
    for x, y in ((2, 2), (client.right - 3, 2), (client.right // 2, client.bottom // 2),
                 (2, client.bottom - 3), (client.right - 3, client.bottom - 3)):
        hit = user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x0 + x, y0 + y)), 2)  # GA_ROOT
        if hit != int(w.winId()):
            w.close()
            pytest.skip("this session cannot bring a test window to the foreground")


def _client_origin(w) -> tuple[int, int]:
    """Device-pixel screen position of the Win32 client's top-left."""
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(wintypes.HWND(int(w.winId())), ctypes.byref(point))
    return point.x, point.y


def _frame_bounds(w) -> wintypes.RECT:
    """The visible window rect (DWMWA_EXTENDED_FRAME_BOUNDS), device px."""
    frame = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(int(w.winId())), 9, ctypes.byref(frame), ctypes.sizeof(frame))
    return frame


def test_real_chrome_paints_its_own_caption_inside_the_client(real_app):
    """The client starts at the window's visible top (no native caption),
    the screen shows the band's own pixels there (Qt draws nothing over
    it), the close button ends on the Win32 client's right edge, the window
    keeps its system menu and min/max styles, and DWM paints the edge in the
    frame color.

    Seeded defects: drop ExpandedClientAreaHint in expand_client_area -> a
    native caption sits above the client -> the client-top assert is red;
    keep Qt's title and button hints -> Qt paints its own caption over the
    band -> the pixel compare is red; drop ClientAreaFit -> the close button
    ends past the client -> red; skip add_window_menu's SetWindowLongW -> the
    style/system-menu asserts are red; skip the DWM border call -> the edge
    pixel is the system border -> red. (DWMWA_BORDER_COLOR can be set but not
    read back: DwmGetWindowAttribute(34) returns E_INVALIDARG, so the edge
    pixel is the observable.)"""
    from PySide6.QtCore import QPoint, QRect

    from yamabiko.overlay_app.window_chrome import client_width

    w, band, _fit = _chrome_window()
    try:
        _foreground(w)
        dpr = w.devicePixelRatioF()
        hwnd = wintypes.HWND(int(w.winId()))
        x0, y0 = _client_origin(w)
        frame = _frame_bounds(w)
        assert y0 - frame.top <= 2 * dpr, f"client top {y0} vs visible top {frame.top}"

        close = band.caption_button("close")
        assert close.mapTo(w, QPoint(close.width(), 0)).x() == client_width(w)

        style = ctypes.windll.user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
        assert style & 0x00080000 and style & 0x00020000 and style & 0x00010000
        assert ctypes.windll.user32.GetSystemMenu(hwnd, False)

        width, height = int(400 * dpr), int(32 * dpr)
        screen = grab_screen(x0, y0, x0 + width, y0 + height)
        own = w.grab(QRect(0, 0, 400, 32)).toImage()
        diffs = 0
        samples = 0
        for y in range(int(4 * dpr), height - int(2 * dpr), 2):
            for x in range(int(4 * dpr), width, 3):
                r, g, b = screen.pixel(x, y)
                c = own.pixelColor(x, y)
                samples += 1
                diffs += max(abs(r - c.red()), abs(g - c.green()), abs(b - c.blue())) > 40
        assert diffs / samples < 0.02, f"{diffs}/{samples} caption pixels differ from the band"

        mid = (frame.top + frame.bottom) // 2
        edge = screen_pixel(frame.left, mid)
        assert max(abs(a - b) for a, b in zip(edge, (0xBD, 0xBD, 0xB9))) <= 12, (
            f"window edge {edge} is not the frame color #bdbdb9")
    finally:
        w.close()


def test_real_caption_drag_moves_the_window_and_close_closes_it(real_app):
    """Injected mouse input on the native platform: dragging the band's sky
    moves the window with the cursor (startSystemMove), clicking the painted
    close button closes it.

    The move starts once the drag passes the system drag distance, so the
    window travels with the cursor from that point: it must move most of the
    way in the drag direction, and the cursor must end over the same spot of
    the window it held when the move began.

    Seeded defects: drop the startSystemMove call -> the window stays put ->
    red; leave the close button unwired -> the window stays visible -> red."""
    from PySide6.QtCore import QTimer

    w, band, _fit = _chrome_window()
    user32 = ctypes.windll.user32
    try:
        _foreground(w)
        hwnd = wintypes.HWND(int(w.winId()))
        before = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(before))
        dpr = w.devicePixelRatioF()
        x0, y0 = _client_origin(w)
        sx, sy = x0 + int(300 * dpr), y0 + int(50 * dpr)
        user32.SetCursorPos(sx, sy)
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down

        def step(i: int) -> None:
            user32.SetCursorPos(sx + 10 * i, sy + 5 * i)
            if i < 10:
                QTimer.singleShot(30, lambda: step(i + 1))
            else:
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up

        QTimer.singleShot(80, lambda: step(1))
        _settle(1200)
        after = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(after))
        dx, dy = after.left - before.left, after.top - before.top
        assert dx >= 50 and dy >= 25, f"window moved by {dx}, {dy}"
        # The cursor ended (sx + 100, sy + 50); the window moved by (dx, dy)
        # since the move began at (sx + 100 - dx, sy + 50 - dy): that start
        # lies on the drag line, within the drag distance plus one step.
        assert abs((100 - dx) - 2 * (50 - dy)) <= 3, f"window lagged the cursor: {dx}, {dy}"

        from yamabiko.overlay_app.window_chrome import client_width

        x0, y0 = _client_origin(w)
        user32.SetCursorPos(x0 + int((client_width(w) - 23) * dpr), y0 + int(16 * dpr))
        _settle(100)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        _settle()
        assert not w.isVisible()
    finally:
        w.close()


def test_real_dark_settings_shows_no_light_line_inside_its_border(real_app):
    """Qt 6.11 leaves a 1 px non-client strip inside the DWM border (left,
    right, bottom) that nothing painted: a white line in the dark theme. On
    deactivation DefWindowProc also painted an inactive caption over the top
    client rows. After a live switch to dark the strip is dark, and the top
    rows stay dark when another window takes the focus.

    Seeded defects: drop SettingsWindow.nativeEvent -> the strip is white ->
    red; drop repaint_nonclient from apply_palette -> the strip keeps the
    light ground (DWM lightens it to white) -> red; skip the WM_NCACTIVATE
    branch -> the top rows turn white on deactivation -> red."""
    from PySide6.QtWidgets import QWidget

    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.theme import MIST, NIGHT

    w = SettingsWindow(dict(STYLE_DEFAULTS), list_devices=lambda: [])
    other = QWidget()
    try:
        w.apply_palette(MIST)
        w.move(160, 120)
        w.show()
        _foreground(w)
        w.apply_palette(NIGHT)
        _settle()
        frame = _frame_bounds(w)
        img = grab_screen(frame.left, frame.top, frame.right, frame.bottom)
        mid_y, third_x = img.height // 2, img.width // 3
        strip = {"left": img.pixel(1, mid_y), "right": img.pixel(img.width - 2, mid_y),
                 "bottom": img.pixel(third_x, img.height - 2)}
        other.setGeometry(w.x() + w.width() + 60, w.y(), 200, 150)
        other.show()
        other.activateWindow()
        _settle()
        user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
        user32.GetForegroundWindow.restype = wintypes.HWND
        # Under Windows' foreground lock the second window may not take the
        # focus: then Settings never deactivates and the half below proves nothing.
        deactivated = user32.GetForegroundWindow() == int(other.winId())
        top = screen_pixel(frame.left + third_x, frame.top + 1) if deactivated else None
    finally:
        other.close()
        w.close()
    light = {side: rgb for side, rgb in strip.items() if max(rgb) > 0x80}
    assert not light, f"light pixels inside the dark window's border: {light}"
    if not deactivated:
        pytest.skip("this session cannot hand the foreground to a second window")
    assert max(top) <= 0x80, f"the top row turned light on deactivation: {top}"


def test_real_minimize_only_window_menu_has_the_minimize_box_and_never_maximize(real_app):
    """Setup's fixed 880 x 600 window: add_window_menu(minimize=True) gives
    it the system menu and the minimize box (a taskbar click minimizes it)
    and no maximize box (Win+Up must not maximize a fixed-size window), also
    when the window had one before.

    Seeded defects: give minimize=True the min_max styles -> the maximize
    assert is red; skip the minimize box -> red; keep a maximize box set
    earlier -> red."""
    from PySide6.QtWidgets import QWidget

    from yamabiko.overlay_app.window_chrome import add_window_menu, expand_client_area

    w = QWidget()
    expand_client_area(w)
    w.resize(400, 300)
    w.move(160, 120)
    w.show()
    try:
        _settle(200)
        hwnd = wintypes.HWND(int(w.winId()))
        assert add_window_menu(w, min_max=True)
        assert add_window_menu(w, min_max=False, minimize=True)
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
        assert style & 0x00080000 and style & 0x00020000, f"no system menu or minimize box: {style:#x}"
        assert not style & 0x00010000, f"the maximize box stayed: {style:#x}"
        assert ctypes.windll.user32.GetSystemMenu(hwnd, False)
    finally:
        w.close()
