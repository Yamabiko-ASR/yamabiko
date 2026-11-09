"""Real Windows platform: the series bar's window (@pytest.mark.real).

The bar's call sites the offscreen QPA cannot show: a Qt Tool | Frameless |
StaysOnTop window pre-warmed at construction (winId + grab, as
build_overlay does) without showing, then placed on a chosen QScreen and
shown from code; the DWM cloak until its first paint
(window_chrome.set_cloak), DWM's small corners, border color (also on a
live theme change) and soft shadow on it (where Windows draws window
shadows); the foreground taken from another process
(window_chrome.take_foreground: SendInput + SetForegroundWindow), given
back to it when the bar hides from the foreground or the subtitle window,
or else to the topmost app window (foreign_foreground, window_class,
foreground_widget, window_shows, foreground_is_app_window,
topmost_app_window, hand_back_foreground: GetForegroundWindow,
GetWindowThreadProcessId, GetClassNameW, EnumWindows, IsIconic,
GetWindowLongW, GetWindow, GetWindowRect, DwmGetWindowAttribute,
QWidget.find, IsWindow, IsWindowVisible, SetForegroundWindow) and the
native WindowDeactivate it hides on; the height animation (setFixedHeight)
and its reduced-motion jump; the slide-away (move + setWindowOpacity, a
layered window while it fades) and its restore; the monitor-above guard's
native screen rects (QScreen geometry and devicePixelRatio, checked against
GetMonitorInfoW). SKIPS
unless the native windows QPA is available (run this file in its own
invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_series_palette.py -m real -v
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

import pytest

from tests.overlay_app._foreground_owner import (
    foreground_hwnd as _foreground_hwnd,
    foreground_owner as _foreground_owner,
    wait_for as _wait_for,
)
from tests.overlay_app._screen_grab import screen_pixel

pytestmark = pytest.mark.real

_S_OK = 0
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DWMWA_CLOAKED = 14
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWCP_ROUNDSMALL = 3
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
# Windows' "Show shadows under windows" (SystemParametersInfoW).
_SPI_GETDROPSHADOW = 0x1024
# The probe's mid-gray backdrop (p5_probe_report.md Q2): DWM's shadow shows
# as a darkening on it.
_BACKDROP = "#7a8691"


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
def _own_app_data(monkeypatch, tmp_path):
    """Real tier: the app-data isolation exempts it, so the bar's saved-cover
    lookups read this test's folder, never the user's."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))


def _settle(ms: int = 400) -> None:
    from PySide6.QtTest import QTest

    QTest.qWait(ms)


def _user32():
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SystemParametersInfoW.argtypes = (wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                                             wintypes.UINT)
    user32.SystemParametersInfoW.restype = wintypes.BOOL
    return user32


def _window_rect(hwnd: int) -> wintypes.RECT:
    """The native window rect, physical px."""
    rect = wintypes.RECT()
    _user32().GetWindowRect(hwnd, ctypes.byref(rect))
    return rect


def _layered(hwnd: int) -> bool:
    return bool(_user32().GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_LAYERED)


def _dwm_int(hwnd: int, attribute: int) -> int:
    value = ctypes.c_int(-1)
    hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(hwnd), attribute, ctypes.byref(value), ctypes.sizeof(value),
    ) & 0xFFFFFFFF
    assert hr == _S_OK, f"DwmGetWindowAttribute({attribute}) failed: 0x{hr:08x}"
    return value.value


def _frame(hwnd: int) -> wintypes.RECT:
    """DWM's visible frame (edge included, shadow excluded), physical px."""
    rect = wintypes.RECT()
    hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(hwnd), _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect),
        ctypes.sizeof(rect),
    ) & 0xFFFFFFFF
    assert hr == _S_OK, f"DwmGetWindowAttribute(EXTENDED_FRAME_BOUNDS) failed: 0x{hr:08x}"
    return rect


def _near(rgb, hex_color: str, tolerance: int) -> bool:
    from PySide6.QtGui import QColor

    c = QColor(hex_color)
    return max(abs(a - b) for a, b in zip(rgb, (c.red(), c.green(), c.blue()))) <= tolerance


def _physical(screen, x: float, y: float) -> tuple[int, int]:
    """A logical point on ``screen`` in Windows' physical px: Qt keeps each
    screen's top-left fixed and scales from there by its pixel ratio."""
    origin = screen.geometry().topLeft()
    dpr = screen.devicePixelRatio()
    return (round(origin.x() + (x - origin.x()) * dpr),
            round(origin.y() + (y - origin.y()) * dpr))


def _hit():
    from yamabiko.gazetteer.anilist import SeriesHit

    return SeriesHit(154587, "Sousou no Frieren", "Frieren: Beyond Journey's End",
                     "葬送のフリーレン", 2023, "TV", cover_image=None, color="#e4a15d",
                     episodes=28)


class _BarClient:
    """The bar's AniList client without the network: every search finds
    Frieren; no list is read (no username is set)."""

    def search_series(self, query, limit=8):
        return [_hit()]

    def fetch_watching(self, username):
        return []


def _bar(**kw):
    """A bar as build_overlay makes it: constructed, then pre-warmed (its
    native window and one off-screen render, P3) before any summon."""
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import MIST

    kw.setdefault("theme_palette", MIST)
    palette = SeriesPalette(client=_BarClient(), username_provider=lambda: "", **kw)
    palette.prewarm()
    return palette


def _show_results(palette, query: str = "frieren") -> None:
    """A settled search answer for ``query`` in the field: the results view
    (no debounce, no worker)."""
    palette._search.blockSignals(True)
    palette._search.setText(query)
    palette._search.blockSignals(False)
    palette._on_results(query, True, [_hit()])


def _pick(palette) -> None:
    """Enter on the lit first result: the pick, then the status face."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _show_results(palette)
    QTest.keyClick(palette._search, Qt.Key.Key_Return)
    assert palette._mode == "status", "Enter on the lit result did not pick it"


def _backdrop():
    """A mid-gray topmost window over the top of the primary work area,
    shown before the bar (so under it): DWM's shadow darkens it. It takes
    the activation (the bar's summon takes it back): a process whose first
    top-level window shows without activation gets a handled COM exception
    that pytest's faulthandler prints as "Windows fatal exception: code
    0x8001010d"."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QGuiApplication, QPalette
    from PySide6.QtWidgets import QWidget

    backdrop = QWidget(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                       | Qt.WindowType.WindowStaysOnTopHint)
    colors = backdrop.palette()
    colors.setColor(QPalette.ColorRole.Window, QColor(_BACKDROP))
    backdrop.setPalette(colors)
    backdrop.setAutoFillBackground(True)
    work = QGuiApplication.primaryScreen().availableGeometry()
    backdrop.setGeometry(work.x(), work.y(), work.width(), 300)
    backdrop.show()
    backdrop.activateWindow()
    _settle()
    return backdrop


def _drop_shadows_on() -> bool:
    """Windows' "Show shadows under windows" (System Properties >
    Performance Options): DWM draws window shadows only while it is on."""
    on = wintypes.BOOL()
    assert _user32().SystemParametersInfoW(_SPI_GETDROPSHADOW, 0, ctypes.byref(on), 0), (
        "SystemParametersInfoW(SPI_GETDROPSHADOW) failed")
    return bool(on.value)


def _assert_chrome(palette, theme: dict) -> None:
    """The shown bar as DWM composes it: uncloaked, small corners, the
    frame-tone edge, its own opaque ground (no fade left), and, where
    Windows draws window shadows, DWM's soft shadow on the backdrop below
    it."""
    hwnd = int(palette.winId())
    assert _dwm_int(hwnd, _DWMWA_CLOAKED) == 0, "the bar is still cloaked after summon"
    corner = _dwm_int(hwnd, _DWMWA_WINDOW_CORNER_PREFERENCE)
    assert corner == _DWMWCP_ROUNDSMALL, f"corner preference {corner}, not DWMWCP_ROUNDSMALL"
    assert not _layered(hwnd), "the bar is still a layered (fading) window"
    frame = _frame(hwnd)
    # The bar's own ratio, never palette.screen(): PySide parents the QScreen
    # wrapper that QWidget.screen() returns to the widget, so the bar's
    # deletion would invalidate that shared wrapper for every later holder
    # ("Internal C++ object (QScreen) already deleted" in the next test).
    dpr = palette.devicePixelRatio()
    row = frame.top + round(24 * dpr)  # the middle of the 48 px bar row
    edge = screen_pixel(frame.left, row)
    assert _near(edge, theme["frame"], 12), f"bar edge {edge} is not the frame tone {theme['frame']}"
    ground = screen_pixel(frame.left + round(560 * dpr), row)  # the empty right end of the field
    assert _near(ground, theme["raised"], 3), (
        f"bar ground {ground} is not an opaque {theme['raised']}")
    if _drop_shadows_on():
        middle = (frame.left + frame.right) // 2
        shadow = screen_pixel(middle, frame.bottom + 2)
        backdrop = screen_pixel(middle, frame.bottom + round(120 * dpr))
        assert sum(backdrop) - sum(shadow) >= 24, (
            f"no DWM shadow under the bar: {shadow} against the backdrop {backdrop}")
    # Else "Show shadows under windows" is off: DWM draws no window shadow on
    # this PC, so there is none to see; the corner readback above still pins
    # the DWM call that gives the bar its shadow where shadows are on.


def test_real_bar_keeps_its_dwm_chrome_after_summon_and_after_sliding_away(real_app, monkeypatch):
    """P3/P4/L2/L12: pre-warmed, the bar stays hidden; summoned, it is
    cloaked from DWM until its first paint, then uncloaked, with DWM's
    small corners, the frame-tone edge, an opaque ground and (where Windows
    draws window shadows) DWM's soft shadow; a live theme change recolors
    the edge; after "Now watching" slides it away with a fade, the next
    summon brings all of it back at full opacity.

    Seeded defects: show the window in prewarm -> IsWindowVisible -> red;
    cloak with DWMWA_CLOAKED (``_DWMWA_CLOAK = 14``, which DWM refuses) ->
    the bar is never cloaked -> red; never uncloak -> DWMWA_CLOAKED nonzero
    after the settle -> red; skip set_corner_preference in summon (or call
    it before show) -> readback 2 -> red; skip set_border_color in
    apply_palette -> the Night edge check red; leave the opacity below 1
    after the slide -> WS_EX_LAYERED stays and the ground is blended ->
    red."""
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.theme import MIST, NIGHT

    monkeypatch.setattr(sp, "_DONE_HOLD_MS", 100)
    backdrop = _backdrop()
    palette = _bar()
    try:
        hwnd = int(palette.winId())
        assert not _user32().IsWindowVisible(hwnd), "the pre-warm showed the bar"
        palette.summon()
        # No event has run since show(), so the first paint (which queues the
        # uncloak) has not happened yet: DWM keeps the bar out of composition.
        assert _dwm_int(hwnd, _DWMWA_CLOAKED) != 0, "summon showed the bar uncloaked"
        _settle()
        _assert_chrome(palette, MIST)

        palette.apply_palette(NIGHT)
        _settle()
        frame = _frame(int(palette.winId()))
        edge = screen_pixel(frame.left, frame.top + round(24 * palette.devicePixelRatio()))
        assert _near(edge, NIGHT["frame"], 12), (
            f"a live theme change left the bar edge at {edge}, not {NIGHT['frame']}")
        palette.apply_palette(MIST)

        _pick(palette)
        palette.show_built(_hit().media_id)
        assert _wait_for(lambda: not palette.isVisible(), 4000), "the bar did not slide away"
        assert palette.windowOpacity() == 1.0, "the slide-away left the bar faded"

        palette._search.clear()  # the next summon shows the bar alone
        palette.summon()
        _settle()
        _assert_chrome(palette, MIST)
    finally:
        palette.hide()
        palette.deleteLater()
        backdrop.close()
        backdrop.deleteLater()


def test_real_results_unfold_the_native_window_to_their_height(real_app):
    """P1/L12: with motion on, a search answer unfolds the native window
    from the 48 px bar to the dropdown's full height over the 160 ms
    OutCubic animation (setFixedHeight per frame): Windows reports heights
    in between, then the full height. Read from Windows (GetWindowRect,
    physical px).

    Seeded defects: set the height at once (_set_height always takes its
    instant branch) -> no height in between -> red; drop the animation's
    setFixedHeight -> the native window stays 48 px tall -> red."""
    from yamabiko.overlay_app.series_palette_parts import BAR_H, dropdown_height

    palette = _bar()
    try:
        palette.summon()
        _settle()
        hwnd = int(palette.winId())
        dpr = palette.devicePixelRatio()
        bar = round(BAR_H * dpr)
        rect = _window_rect(hwnd)
        assert abs((rect.bottom - rect.top) - bar) <= 2, (
            f"the bar alone is {rect.bottom - rect.top} px tall, not {bar}")

        _show_results(palette)
        full = round((BAR_H + dropdown_height(palette._dropdown.model())) * dpr)
        heights: list[int] = []

        def unfolded() -> bool:
            r = _window_rect(hwnd)
            heights.append(r.bottom - r.top)
            return abs(heights[-1] - full) <= 2

        assert _wait_for(unfolded, 2000), (
            f"the results never reached their height {full}: {heights}")
        assert any(bar + 2 < h < full - 2 for h in heights), (
            f"the window jumped to its height instead of unfolding: {heights}")
    finally:
        palette.hide()
        palette.deleteLater()


def test_real_bar_opens_top_center_on_the_subtitle_screen_of_every_monitor(real_app):
    """L3: the bar opens on the monitor the subtitles are pinned to
    (OverlayWindow.subtitle_screen), centered on its work area, 24 px below
    its top, 640 logical px wide: on each connected monitor in turn (this
    machine: the 150 % primary and the 100 % secondary). Read back from
    Windows (GetWindowRect, physical px).

    Seeded defects: place on the primary screen whatever the provider says
    -> the second monitor's check red; center on the full geometry or drop
    the 24 px -> off by the taskbar or 24 px -> red; size the window in
    another screen's scale -> the width check red."""
    from PySide6.QtGui import QGuiApplication

    from yamabiko.overlay_app.overlay_window import OverlayWindow

    for screen in QGuiApplication.screens():
        window = OverlayWindow(position={"screen": screen.name()})
        palette = _bar(screen_provider=window.subtitle_screen)
        try:
            palette.summon()
            _settle()
            work = screen.availableGeometry()
            left, top = _physical(screen, work.x() + (work.width() - 640) // 2, work.top() + 24)
            rect = _window_rect(int(palette.winId()))
            dpr = screen.devicePixelRatio()
            assert abs(rect.left - left) <= 2 and abs(rect.top - top) <= 2, (
                f"on {screen.name()!r} the bar opened at ({rect.left}, {rect.top}), "
                f"not ({left}, {top})")
            assert abs((rect.right - rect.left) - round(640 * dpr)) <= 2, (
                f"on {screen.name()!r} the bar is {rect.right - rect.left} px wide, "
                f"not {round(640 * dpr)}")
            on = QGuiApplication.screenAt(palette.geometry().center())  # not palette.screen()
            assert on is not None and on.name() == screen.name()
        finally:
            palette.hide()
            palette.deleteLater()
            window.close()
            window.deleteLater()


def test_real_summon_focuses_the_field_and_another_window_hides_the_bar(real_app):
    """L13: summoned while another app owns the foreground (the video player
    the user watches: a separate process that clicked its own window), the
    bar is the foreground window with the search field focused, so typing
    goes straight to the search; when another window activates (the user
    goes back to that app), the bar (in its search face) hides.

    Seeded defects: drop activateWindow and take_foreground from summon ->
    the other app keeps the foreground -> red; drop the field's setFocus ->
    red; drop the WindowDeactivate branch -> the bar stays up -> red. Skips
    only when this session never gives the helper's window the foreground."""
    from PySide6.QtWidgets import QApplication

    with _foreground_owner() as owner_hwnd:
        palette = _bar()
        try:
            palette.summon()
            _settle()
            assert _foreground_hwnd() == int(palette.winId()), (
                "summon did not make the bar the foreground window")
            assert QApplication.focusWidget() is palette._search, (
                "the search field does not have the keyboard focus")
            # The bar's process owns the foreground now, so Windows lets it
            # hand the foreground to the other app, as the user's click would.
            assert _user32().SetForegroundWindow(owner_hwnd), (
                "Windows refused to give the other app the foreground back")
            _settle()
            assert not palette.isVisible(), "the bar stayed open after another window activated"
        finally:
            palette.hide()
            palette.deleteLater()


def test_real_reduced_motion_opens_at_full_height_and_hides_without_sliding(
    real_app, monkeypatch,
):
    """L12: with Reduce motion the results open at their full height at
    once (no 160 ms unfold) and "Now watching" ends with the bar hidden
    where it stood: it never moves and never turns into a fading (layered)
    window. Read from Windows: the native window rect and WS_EX_LAYERED,
    sampled through the hide.

    Seeded defects: animate the height anyway -> 30 ms after the answer the
    native height is short of the target -> red; slide or fade anyway -> a
    sampled top moves up or WS_EX_LAYERED appears -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.series_palette_parts import BAR_H, dropdown_height

    monkeypatch.setattr(sp, "_DONE_HOLD_MS", 100)
    palette = _bar(reduce_motion=True)
    try:
        palette.summon()
        _settle()
        hwnd = int(palette.winId())
        dpr = palette.devicePixelRatio()

        _show_results(palette)
        _settle(30)  # an unfold would be under half way (OutCubic, 160 ms)
        full = round((BAR_H + dropdown_height(palette._dropdown.model())) * dpr)
        rect = _window_rect(hwnd)
        assert abs((rect.bottom - rect.top) - full) <= 2, (
            f"with reduced motion the bar is {rect.bottom - rect.top} px tall 30 ms "
            f"after the results, not {full}")

        QTest.keyClick(palette._search, Qt.Key.Key_Return)
        assert palette._mode == "status"
        top = _window_rect(hwnd).top
        palette.show_built(_hit().media_id)
        seen: list[tuple[int, bool]] = []

        def hidden() -> bool:
            if palette.isVisible():
                seen.append((_window_rect(hwnd).top, _layered(hwnd)))
                return False
            return True

        assert _wait_for(hidden, 4000), "the bar never hid after Now watching"
        assert seen, "no sample while the bar showed Now watching"
        assert {t for t, _layer in seen} == {top}, f"the bar moved while hiding: {seen}"
        assert not any(layer for _t, layer in seen), (
            "the bar faded (layered window) with reduced motion")
    finally:
        palette.hide()
        palette.deleteLater()


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def _monitor_rect(x: int, y: int):
    """The physical rect Windows reports for the monitor at physical (x, y)
    (MonitorFromPoint + GetMonitorInfoW rcMonitor); None off every monitor."""
    from PySide6.QtCore import QRect

    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
    user32.MonitorFromPoint.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MONITORINFO))
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    monitor = user32.MonitorFromPoint(wintypes.POINT(x, y), 0)  # MONITOR_DEFAULTTONULL
    if not monitor:
        return None
    info = _MONITORINFO(cbSize=ctypes.sizeof(_MONITORINFO))
    assert user32.GetMonitorInfoW(monitor, ctypes.byref(info)), "GetMonitorInfoW failed"
    r = info.rcMonitor
    return QRect(r.left, r.top, r.right - r.left, r.bottom - r.top)


def test_real_screen_above_finds_a_monitor_over_a_top_edge(real_app):
    """P2's guard on the real monitor layout, in native pixels (lead ruling
    on the Task 6 review): each screen's native rect from Qt (its
    geometry()'s top-left, size() x devicePixelRatio()) is the monitor rect
    Windows reports, within 1 px of rounding; a rect right under a monitor's
    bottom has that monitor above it; each monitor's own rect has one above
    exactly where one of Windows' monitor rects meets its top edge (within
    1 px) over at least one shared column (this machine: none). The
    mixed-scale cases are the unit test's
    (test_a_monitor_above_is_found_in_native_pixels_at_mixed_scale).

    Seeded defect: compare the screens' bottoms with the rect's bottom
    instead of its top -> no monitor above the rect under each monitor ->
    red. (Dropping the scale from _native_rect goes red only where a
    monitor is scaled.)"""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QGuiApplication

    from yamabiko.overlay_app.series_palette import _native_rect, _screen_above

    rects = []
    for screen in QGuiApplication.screens():
        g = screen.geometry()
        windows = _monitor_rect(g.x(), g.y())
        native = _native_rect(screen)
        assert windows is not None and all(
            abs(a - b) <= 1 for a, b in zip(native.getCoords(), windows.getCoords())), (
            f"{screen.name()!r}: native rect {native.getRect()}, Windows says "
            f"{windows and windows.getRect()}")
        under = QRect(windows.x(), windows.bottom() + 1, windows.width(), 200)
        assert _screen_above(under) is True, (
            f"no monitor found above a rect under {screen.name()!r}")
        rects.append((screen.name(), windows))
    for name, windows in rects:
        expected = any(
            abs(r.bottom() + 1 - windows.top()) <= 1
            and min(r.right(), windows.right()) - max(r.left(), windows.left()) >= 1
            for _other, r in rects)
        assert _screen_above(windows) is expected, (
            f"monitor above {name!r}: guard {not expected}, layout {expected}")


def test_real_hotkey_summon_takes_the_foreground_from_another_process(real_app, monkeypatch):
    """D-FG in the product's steady state: the subtitle window
    (OverlayWindow, as build_overlay shows it at boot) has been up for a
    while, the bar is built, pre-warmed and wired to PipelineClient's
    showSeriesPaletteReceived as build_overlay wires it, and another app (the
    video player: a separate process that clicked its own window) holds the
    foreground. Ctrl+Alt+S reaches the overlay process as the WebSocket frame
    show_series_palette, so this process did not receive the summoning input
    and Windows refuses it the foreground (probe: 0 of 9 with activateWindow
    alone, while Qt still reports the bar active). summon() takes it with one
    empty mouse input first (window_chrome.take_foreground): every frame
    makes the bar the foreground window with the search field focused.

    Then each hide (lead ruling on final review round 3):
    - Esc leaves the foreground with the other app. Windows mostly returns
      it there itself on this route (without the hand-back, 1 run in 3 here
      left it on a window of this process), so this check mostly pins
      Windows' behavior.
    - The foreground left on the click-through subtitle window (where the
      visual check saw Windows put it) goes back to the other app, both
      when the search face hides on deactivation and after a pick's
      slide-away (window_chrome.hand_back_foreground).
    - A Yamabiko window the user activates (Settings here) keeps it.
    This process injects nothing itself and types nothing.

    Run alone (-k hotkey), pytest's faulthandler prints one "Windows fatal
    exception: code 0x8001010d": the subtitle window is this process's first
    window and shows without activation, as at the product's boot, which
    raises a handled COM exception (RPC_E_CANTCALLOUT_ININPUTSYNCCALL), not a
    crash. In the file's order an activated window comes first and nothing is
    printed.

    Seeded defects: _take_foreground returns False without calling
    take_foreground (Task 5's stub) -> the helper keeps the foreground ->
    red; no hand-back -> the subtitle window keeps the foreground -> red;
    hand back whenever this process holds the foreground (round 3) -> the
    other app takes it from Settings -> red. Skips only when this session
    never gives the helper window the foreground."""
    import json

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget

    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.ws_client import PipelineClient

    monkeypatch.setattr(sp, "_DONE_HOLD_MS", 100)
    window = OverlayWindow()
    window.show()
    settings = QWidget()  # a Yamabiko window the user types in, open on the side
    settings.setWindowTitle("Yamabiko test: settings")
    settings.setGeometry(700, 400, 320, 200)
    settings.show()
    palette = _bar(screen_provider=window.subtitle_screen)
    client = PipelineClient()  # never started: the frame enters past the socket
    client.showSeriesPaletteReceived.connect(palette.summon)
    frame = json.dumps({"type": "show_series_palette"})
    subtitles = int(window.winId())

    def summon(which: str) -> None:
        _settle(1000)  # the other app holds the foreground
        client._on_text(frame)
        _settle()
        assert _foreground_hwnd() == int(palette.winId()), (
            f"the {which} summon did not take the foreground from the other process; "
            f"typing would go to that app (foreground {_foreground_hwnd():#x}, other "
            f"app {owner_hwnd:#x})")
        assert palette._search.hasFocus(), "the search field does not have the keyboard focus"

    def back_to_owner() -> bool:
        return bool(_wait_for(lambda: _foreground_hwnd() == owner_hwnd, 2000))

    try:
        _settle(3000)  # the subtitles are up long before any Ctrl+Alt+S
        with _foreground_owner() as owner_hwnd:
            summon("first")
            QTest.keyClick(palette._search, Qt.Key.Key_Escape)
            assert not palette.isVisible(), "Esc did not close the bar"
            assert back_to_owner(), (
                "the foreground did not go back to the other app on Esc (foreground "
                f"{_foreground_hwnd():#x}, Settings {int(settings.winId()):#x}, subtitles "
                f"{subtitles:#x})")

            summon("second")
            assert _user32().SetForegroundWindow(subtitles), "the subtitle window took no foreground"
            assert _wait_for(lambda: not palette.isVisible(), 2000), "the search face stayed up"
            assert back_to_owner(), (
                "the hide left the foreground on this process's window "
                f"{_foreground_hwnd():#x} (subtitles {subtitles:#x}), not the other app")

            summon("third")
            hwnd = int(settings.winId())
            assert _user32().SetForegroundWindow(hwnd), "Settings took no foreground"
            assert _wait_for(lambda: not palette.isVisible(), 2000), "the search face stayed up"
            _settle(600)
            assert _foreground_hwnd() == hwnd, (
                "the bar took the foreground from the Settings window the user activated "
                f"(foreground {_foreground_hwnd():#x}, other app {owner_hwnd:#x})")
            assert _user32().SetForegroundWindow(owner_hwnd), "the other app took no foreground"

            summon("fourth")
            _pick(palette)
            # Where Windows left the foreground in the visual check: the
            # status face stays up (deactivation never hides it).
            assert _user32().SetForegroundWindow(subtitles), "the subtitle window took no foreground"
            _settle()
            assert palette.isVisible() and _foreground_hwnd() == subtitles
            palette.show_built(_hit().media_id)
            assert _wait_for(lambda: not palette.isVisible(), 4000), "the bar did not slide away"
            assert back_to_owner(), (
                "the slide-away left the foreground on this process's window "
                f"{_foreground_hwnd():#x} (subtitles {subtitles:#x}), not the other app")
    finally:
        palette.hide()
        palette.deleteLater()
        settings.close()
        settings.deleteLater()
        window.close()
        window.deleteLater()


def test_real_bar_keeps_no_taskbar_window_to_give_the_foreground_back_to(real_app, monkeypatch):
    """Visual check 2, item 6: window_chrome.window_class reads the real
    class name (GetClassNameW): the taskbar FindWindowW finds by its class
    is "Shell_TrayWnd". A bar summoned while that window is the foreground
    another process reports (as on the short menu's route, with the shell in
    front) keeps nothing to give the foreground back to, so its hide cannot
    reopen the taskbar or the tray overflow.

    Seeded defects: GetClassNameW declared or read wrong (no class name) ->
    red; the summon keeps every foreign window -> red."""
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.window_chrome import window_class

    taskbar = _user32().FindWindowW("Shell_TrayWnd", None) or 0
    assert taskbar, "no taskbar on this desktop"
    assert window_class(taskbar) == "Shell_TrayWnd"
    monkeypatch.setattr(sp, "foreign_foreground", lambda: taskbar)
    palette = _bar()
    try:
        palette.summon()
        _settle()
        assert palette._return_to == 0, "the bar kept the taskbar to give the foreground back to"
    finally:
        palette.hide()
        palette.deleteLater()


def _above(upper: int, lower: int) -> bool:
    """Whether window ``upper`` comes before ``lower`` in the z-order."""
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetWindow.restype = wintypes.HWND
    hwnd = upper
    while hwnd:
        hwnd = user32.GetWindow(hwnd, 2) or 0  # GW_HWNDNEXT
        if hwnd == lower:
            return True
    return False


def _to_top_of_topmost(hwnd: int) -> None:
    """Put window ``hwnd`` at the top of the always-on-top band, without
    activating it."""
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, wintypes.UINT)
    user32.SetWindowPos.restype = wintypes.BOOL
    hwnd_topmost = wintypes.HWND(-1)
    swp_nosize_nomove_noactivate = 0x0001 | 0x0002 | 0x0010
    assert user32.SetWindowPos(hwnd, hwnd_topmost, 0, 0, 0, 0, swp_nosize_nomove_noactivate)


def test_real_topmost_app_window_is_the_ordinary_window_on_top_of_any_process(real_app):
    """Lead rulings on visual checks 3 and 4: window_chrome.topmost_app_window
    walks the real z-order (EnumWindows, its callback kept alive) to the
    first ordinary window of any process. The other app (the helper, which
    clicked itself) is put at the top of the always-on-top band, so no other
    app's window sits above it (lead ruling on final review round 5); above
    it go an ordinary topmost window of this process (as Settings) and, on
    top, a tool window of this process (as the flyout). The walk passes the
    tool window and returns ours; by the rule (_is_app_window) ours and the
    helper count, the tool window and the taskbar do not.

    Seeded defects: skip this process's windows -> the helper is returned ->
    red; take tool windows -> the flyout stand-in is returned -> red; the
    walk keeps the last match instead of the first -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    from yamabiko.overlay_app.window_chrome import _is_app_window, topmost_app_window

    ours = QWidget(None, Qt.WindowType.WindowStaysOnTopHint)
    ours.setWindowTitle("Yamabiko test: settings on top")
    ours.setGeometry(600, 380, 300, 160)
    tool = QWidget(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                   | Qt.WindowType.WindowStaysOnTopHint)
    tool.setWindowTitle("Yamabiko test: flyout on top")
    tool.setGeometry(620, 400, 260, 120)
    with _foreground_owner() as owner_hwnd:
        try:
            _to_top_of_topmost(owner_hwnd)
            ours.show()
            tool.show()
            _settle()
            hwnd, tool_hwnd = int(ours.winId()), int(tool.winId())
            _to_top_of_topmost(hwnd)
            _to_top_of_topmost(tool_hwnd)
            _settle()
            assert _above(tool_hwnd, hwnd) and _above(hwnd, owner_hwnd), (
                "precondition: the tool window over ours over the other app")
            taskbar = _user32().FindWindowW("Shell_TrayWnd", None) or 0
            judged = (_is_app_window(hwnd), _is_app_window(owner_hwnd),
                      _is_app_window(tool_hwnd), _is_app_window(taskbar))
            assert judged == (True, True, False, False), (
                f"the rule misjudged ours, the helper, the tool window or the taskbar: {judged}")
            assert topmost_app_window() == hwnd, (
                f"topmost ordinary window {topmost_app_window():#x}, not ours {hwnd:#x} "
                f"(the helper {owner_hwnd:#x}, the tool window {tool_hwnd:#x})")
        finally:
            for window in (ours, tool):
                window.close()
                window.deleteLater()


def test_real_bar_summoned_from_the_tray_gives_the_foreground_to_the_app_on_top(
    real_app, monkeypatch,
):
    """Lead rulings on visual check 3 (A2) and final review round 5 (I1).
    Summoned from the tray, a tool window of this process (the flyout, the
    menu) or the shell's overflow is in front, so the bar keeps nothing to
    give back. When it hides, Windows can leave the foreground there (in
    the product the open overflow, which then stays open and keeps the
    keyboard); the bar instead gives it to the topmost app window, the other
    app here, after Esc and after a pick's slide-away. Summoned from
    Settings (a plain window of this process), the bar keeps it and Esc
    gives Settings the foreground back; so does a dialog Settings owns, on
    Esc in the status face (lead ruling on the Task 2 review: a kept window
    only has to still show; on the search face the deactivation's nested
    hide hands the foreground on to the app on top after giving it to the
    dialog, reported separately). The stand-ins take the foreground the
    bar's way (take_foreground).

    Seeded defects: no z-order fallback -> the tray stand-in keeps the
    foreground -> red; keep only another app's window -> the other app takes
    it from Settings -> red; hand back only to a kept ordinary app window
    (window_chrome._is_app_window) -> the owned dialog is dropped and the
    other app on top takes the foreground -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialog, QWidget

    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.window_chrome import take_foreground

    monkeypatch.setattr(sp, "_DONE_HOLD_MS", 100)
    tray = QWidget(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    tray.setWindowTitle("Yamabiko test: tray flyout")
    tray.setGeometry(700, 400, 320, 200)
    settings = QWidget()
    settings.setWindowTitle("Yamabiko test: settings")
    settings.setGeometry(1100, 400, 320, 200)
    dialog = QDialog(settings)  # owned by Settings, as its compact dialogs are
    dialog.setWindowTitle("Yamabiko test: dialog")
    palette = _bar()

    def summon_from(window) -> None:
        _settle(1000)  # the other app holds the foreground
        window.show()
        assert take_foreground(window), f"{window.windowTitle()!r} took no foreground"
        _settle()
        palette.summon()
        _settle()
        assert _foreground_hwnd() == int(palette.winId()), "the summon kept no foreground"

    with _foreground_owner() as owner_hwnd:
        try:
            for hide in ("Esc", "the slide-away"):
                summon_from(tray)
                if hide == "Esc":
                    QTest.keyClick(palette._search, Qt.Key.Key_Escape)
                else:
                    _pick(palette)
                    palette.show_built(_hit().media_id)
                assert _wait_for(lambda: not palette.isVisible(), 4000), f"{hide} left the bar up"
                assert _wait_for(lambda: _foreground_hwnd() == owner_hwnd, 2000), (
                    f"after {hide} the foreground is {_foreground_hwnd():#x} (the tray stand-in "
                    f"{int(tray.winId()):#x}), not the app on top {owner_hwnd:#x}")
            tray.hide()

            summon_from(settings)
            QTest.keyClick(palette._search, Qt.Key.Key_Escape)
            assert not palette.isVisible(), "Esc did not close the bar"
            _settle(600)
            assert _foreground_hwnd() == int(settings.winId()), (
                f"after Esc the foreground is {_foreground_hwnd():#x}, not Settings "
                f"{int(settings.winId()):#x} the bar was summoned from (other app {owner_hwnd:#x})")

            summon_from(dialog)
            _pick(palette)
            QTest.keyClick(palette, Qt.Key.Key_Escape)
            assert not palette.isVisible(), "Esc did not close the status"
            _settle(600)
            assert _foreground_hwnd() == int(dialog.winId()), (
                f"after Esc the foreground is {_foreground_hwnd():#x}, not the dialog "
                f"{int(dialog.winId()):#x} the bar was summoned from (other app {owner_hwnd:#x})")
        finally:
            palette.hide()
            palette.deleteLater()
            for window in (tray, dialog, settings):
                window.close()
                window.deleteLater()


def test_real_hand_back_reaches_the_kept_app_while_another_process_holds_the_foreground(
    real_app,
):
    """Lead ruling on final review round 5 (I2): the bar hands the foreground
    back after its native hide, when Windows may already have given it to
    another process (the tray overflow) and this process may not have had
    the last input (a pointer move over the video while the names loaded).
    hand_back_foreground then has neither foreground right of its own; it
    sends take_foreground's empty mouse input first. Here the kept app is
    one helper process and the window in front is a second one, which
    clicked itself last; this process calls the hand-back with no input of
    its own.

    Seeded defect: SetForegroundWindow without the empty input -> Windows
    refuses, the second helper keeps the foreground -> red."""
    from yamabiko.overlay_app.window_chrome import hand_back_foreground

    with _foreground_owner() as kept_hwnd, \
            _foreground_owner("Yamabiko test: overflow stand-in") as front_hwnd:
        assert _foreground_hwnd() == front_hwnd, "precondition: the second helper is in front"
        assert hand_back_foreground(kept_hwnd), "Windows refused the hand-back"
        assert _wait_for(lambda: _foreground_hwnd() == kept_hwnd, 2000), (
            f"the foreground is {_foreground_hwnd():#x}, not the kept app {kept_hwnd:#x} "
            f"(the second helper {front_hwnd:#x})")


def test_real_bar_gives_the_foreground_to_the_app_on_top_when_the_kept_app_was_minimized(
    real_app,
):
    """Ledger D4 as ruled on the Task 2 review (palette-final-review-6):
    summoned over another app (the helper, which clicked itself), the bar
    keeps that app's window. The user picks a series and, while the names
    load, that window is minimized without taking the activation
    (SW_SHOWMINNOACTIVE: SW_MINIMIZE activates the next window, and a bar no
    longer in front hands nothing back). Windows still reports a minimized
    window visible (IsWindowVisible), so handing it the foreground on Esc
    activates a window nobody sees and nothing usable gets the keyboard. The
    bar drops a kept window that no longer shows (window_chrome.window_shows)
    and falls back as when nothing was kept: to Windows' own pick of an app
    window or the topmost one, here an ordinary topmost window of this
    process (as Settings) at the top of the always-on-top band. Esc hides
    the status face: on the search face the deactivation hides the bar a
    second time during its native hide, and that nested hide's fallback
    covered the defect.

    Seeded defect: hand back to the kept window without the check -> the
    minimized helper takes the foreground -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget

    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.ShowWindow.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    sw_showminnoactive = 7
    ours = QWidget(None, Qt.WindowType.WindowStaysOnTopHint)
    ours.setWindowTitle("Yamabiko test: settings on top")
    ours.setGeometry(600, 380, 300, 160)
    ours.show()  # before the helper, whose own click then takes the foreground
    palette = _bar()
    try:
        with _foreground_owner() as owner_hwnd:
            hwnd = int(ours.winId())
            _to_top_of_topmost(hwnd)
            palette.summon()
            _settle()
            assert palette._return_to == owner_hwnd, "precondition: the bar kept the other app"
            assert _foreground_hwnd() == int(palette.winId()), "precondition: the bar is in front"
            _pick(palette)
            user32.ShowWindow(owner_hwnd, sw_showminnoactive)
            assert _wait_for(lambda: user32.IsIconic(owner_hwnd), 2000), (
                "the other app's window did not minimize")
            _settle()
            assert palette.isVisible() and _foreground_hwnd() == int(palette.winId()), (
                "precondition: the bar is still in front after the minimize")
            QTest.keyClick(palette, Qt.Key.Key_Escape)
            assert not palette.isVisible(), "Esc did not close the bar"
            _settle(600)
            assert _foreground_hwnd() == hwnd, (
                f"after Esc the foreground is {_foreground_hwnd():#x}, not the app on top "
                f"{hwnd:#x} (the minimized kept app {owner_hwnd:#x})")
    finally:
        palette.hide()
        palette.deleteLater()
        ours.close()
        ours.deleteLater()
