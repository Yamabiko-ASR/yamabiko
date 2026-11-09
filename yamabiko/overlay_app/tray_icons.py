"""The tray icon, drawn per state on the taskbar's own pixel grid.

Spec Surface 5 > Icon states (mockup opt-tray-live.tpl): Listening is the
vermilion tile with the white 山; Paused turns the tile grey with two white
pause bars; Subtitles hidden draws the 山 as an outline; a problem adds a
small amber dot at the bottom-right, with a 1 px ring in the taskbar's
color, over whichever state is current.

Windows shows tray icons at the small-icon size of the taskbar's DPI
(GetSystemMetricsForDpi(SM_CXSMICON): 16 px at 100 %, 20 at 125 %, 24 at
150 %, 28 at 175 %, 32 at 200 %). Qt asks a QIcon for that size times the
device pixel ratio and Windows shrinks whatever it gets, so a multi-size
icon blurs (PlatformProbe Q5). Each state is therefore painted at exactly
the device size, on a grid drawn for that size (plan Decisions 23 and 44:
integer rects; the 16 px grid is the mockup's), and Qt gets only that one
pixmap. The tile is antialiased; every rect is pixel-aligned without
antialiasing.

The taskbar color follows SystemUsesLightTheme, not the app color scheme Qt
tracks (AppsUseLightTheme): TaskbarThemeWatcher re-reads it when Windows
sends WM_SETTINGCHANGE "ImmersiveColorSet".
"""
from __future__ import annotations

import ctypes
import functools
import logging
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import (
    QAbstractNativeEventFilter, QCoreApplication, QObject, QRect, QRectF, Qt, Signal,
)
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPainter, QPixmap

_log = logging.getLogger(__name__)

LISTENING = "listening"
PAUSED = "paused"
HIDDEN = "hidden"
STATES = (LISTENING, PAUSED, HIDDEN)

SIZES = (16, 20, 24, 28, 32)   # device px with their own grid
TILE = "#d2452f"               # brand vermilion
TILE_PAUSED = "#8a8f8a"
GLYPH = "#ffffff"
DOT = "#e0a030"                # the problem dot
TASKBAR_COLORS = {"light": "#f3f3f3", "dark": "#202020"}   # the dot's ring

_SM_CXSMICON = 49
_WM_SETTINGCHANGE = 0x001A
_COLOR_SET = "ImmersiveColorSet"
_PERSONALIZE = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


@dataclass(frozen=True)
class _Grid:
    """One size's drawing, as (x, y, w, h) rects (plan Decisions 23 and 44)."""
    radius: float      # tile corner
    glyph: tuple       # the 山
    bars: tuple        # the pause bars
    outline: tuple     # the 山 outline: these rects in white ...
    hole: tuple        # ... then these back in the tile color
    dot: tuple         # the amber dot ...
    ring: tuple        # ... over these in the taskbar color (clipped)


_GRIDS: dict[int, _Grid] = {
    16: _Grid(
        radius=3.5,
        glyph=((3, 7, 2, 6), (7, 3, 2, 10), (11, 7, 2, 6), (3, 11, 10, 2)),
        bars=((5, 4, 2, 8), (9, 4, 2, 8)),
        outline=((2, 7, 3, 6), (6, 3, 4, 10), (11, 7, 3, 6), (2, 10, 12, 3)),
        hole=((3, 8, 1, 4), (7, 4, 2, 7), (12, 8, 1, 4), (3, 11, 10, 1)),
        dot=((12, 11, 3, 5), (11, 12, 5, 3)),
        ring=((12, 10, 3, 7), (11, 11, 5, 5), (10, 12, 7, 3)),
    ),
    20: _Grid(
        radius=4.0,
        glyph=((4, 9, 2, 7), (9, 4, 2, 12), (14, 9, 2, 7), (4, 14, 12, 2)),
        bars=((7, 5, 2, 10), (11, 5, 2, 10)),
        outline=((3, 8, 4, 9), (8, 3, 4, 14), (13, 8, 4, 9), (3, 13, 14, 4)),
        hole=((4, 9, 2, 7), (9, 4, 2, 12), (14, 9, 2, 7), (4, 14, 12, 2)),
        dot=((15, 14, 4, 6), (14, 15, 6, 4)),
        ring=((15, 13, 4, 8), (14, 14, 6, 6), (13, 15, 8, 4)),
    ),
    24: _Grid(
        radius=5.0,
        glyph=((4, 10, 3, 10), (10, 4, 4, 16), (17, 10, 3, 10), (4, 17, 16, 3)),
        bars=((7, 6, 3, 12), (14, 6, 3, 12)),
        outline=((3, 9, 5, 12), (9, 3, 6, 18), (16, 9, 5, 12), (3, 16, 18, 5)),
        hole=((4, 10, 3, 10), (10, 4, 4, 16), (17, 10, 3, 10), (4, 17, 16, 3)),
        dot=((18, 16, 4, 8), (17, 17, 6, 6), (16, 18, 8, 4)),
        ring=((18, 15, 4, 10), (17, 16, 6, 8), (16, 17, 8, 6), (15, 18, 10, 4)),
    ),
    28: _Grid(
        radius=6.0,
        glyph=((5, 12, 3, 11), (12, 5, 4, 18), (20, 12, 3, 11), (5, 20, 18, 3)),
        bars=((9, 7, 3, 14), (16, 7, 3, 14)),
        outline=((4, 11, 5, 13), (11, 4, 6, 20), (19, 11, 5, 13), (4, 19, 20, 5)),
        hole=((6, 13, 1, 9), (13, 6, 2, 16), (21, 13, 1, 9), (6, 21, 16, 1)),
        dot=((21, 19, 5, 9), (20, 20, 7, 7), (19, 21, 9, 5)),
        ring=((21, 18, 5, 11), (20, 19, 7, 9), (19, 20, 9, 7), (18, 21, 11, 5)),
    ),
    32: _Grid(
        radius=7.0,
        glyph=((6, 14, 4, 12), (14, 6, 4, 20), (22, 14, 4, 12), (6, 22, 20, 4)),
        bars=((10, 8, 4, 16), (18, 8, 4, 16)),
        outline=((4, 14, 6, 12), (12, 6, 8, 20), (22, 14, 6, 12), (4, 20, 24, 6)),
        hole=((6, 16, 2, 8), (14, 8, 4, 14), (24, 16, 2, 8), (6, 22, 20, 2)),
        dot=((24, 22, 6, 10), (22, 24, 10, 6)),
        ring=((24, 21, 6, 12), (23, 22, 8, 10), (22, 23, 10, 8), (21, 24, 12, 6)),
    ),
}


def icon_state(paused: bool, subtitles_shown: bool) -> str:
    """The icon's state: paused wins over hidden (Decision 14)."""
    if paused:
        return PAUSED
    return LISTENING if subtitles_shown else HIDDEN


@functools.cache
def _user32():
    user32 = ctypes.WinDLL("user32")   # private handle: argtypes stay local
    user32.GetSystemMetricsForDpi.argtypes = (ctypes.c_int, ctypes.c_uint)
    user32.GetSystemMetricsForDpi.restype = ctypes.c_int
    return user32


def small_icon_metric(dpi: int) -> int:
    """SM_CXSMICON at ``dpi``: the device-pixel size Windows shows tray
    icons at on a monitor of that DPI. 16 off Windows or on failure."""
    if sys.platform != "win32":
        return 16
    try:
        value = _user32().GetSystemMetricsForDpi(_SM_CXSMICON, int(dpi))
    except (OSError, AttributeError):
        _log.debug("GetSystemMetricsForDpi unavailable", exc_info=True)
        return 16
    return value if value > 0 else 16


def tray_icon_px(screen=None) -> int:
    """The size to draw the tray icon at for ``screen`` (None: the primary
    screen): the small-icon metric at the screen's DPI, snapped down to a
    drawn grid (Decision 44)."""
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    metric = small_icon_metric(round(96 * dpr))
    return max((size for size in SIZES if size <= metric), default=SIZES[0])


def _fill(painter: QPainter, rects: tuple, color: QColor) -> None:
    for x, y, w, h in rects:
        painter.fillRect(QRect(x, y, w, h), color)


def paint_icon(size: int, state: str, problem: bool, taskbar: str) -> QImage:
    """The tray icon for ``state`` at ``size`` device px (one of SIZES), with
    the problem dot when ``problem``; ``taskbar`` ("light" / "dark") colors
    the dot's ring."""
    grid = _GRIDS.get(size)
    if grid is None:
        raise ValueError(f"no tray icon grid for {size} px")
    if state not in STATES:
        raise ValueError(f"unknown tray icon state {state!r}")
    tile = QColor(TILE_PAUSED if state == PAUSED else TILE)
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(tile)
    painter.drawRoundedRect(QRectF(0, 0, size, size), grid.radius, grid.radius)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    white = QColor(GLYPH)
    if state == LISTENING:
        _fill(painter, grid.glyph, white)
    elif state == PAUSED:
        _fill(painter, grid.bars, white)
    else:
        _fill(painter, grid.outline, white)
        _fill(painter, grid.hole, tile)
    if problem:
        _fill(painter, grid.ring, QColor(TASKBAR_COLORS.get(taskbar, TASKBAR_COLORS["dark"])))
        _fill(painter, grid.dot, QColor(DOT))
    painter.end()
    return image


def tray_icon(screen, state: str, problem: bool, taskbar: str) -> QIcon:
    """The QIcon for QSystemTrayIcon: only the one pixmap drawn at
    tray_icon_px(screen), so Qt never hands Windows a bigger frame."""
    return QIcon(QPixmap.fromImage(paint_icon(tray_icon_px(screen), state, problem, taskbar)))


def taskbar_theme() -> str:
    """"light" or "dark": the taskbar's theme (SystemUsesLightTheme). A
    missing value, a registry error or a non-Windows host read as "dark"."""
    if sys.platform != "win32":
        return "dark"
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PERSONALIZE) as key:
            value, _ = winreg.QueryValueEx(key, "SystemUsesLightTheme")
    except OSError:
        return "dark"
    return "light" if value == 1 else "dark"


class _SettingChangeFilter(QAbstractNativeEventFilter):
    """Calls ``on_color_set`` for WM_SETTINGCHANGE "ImmersiveColorSet"; never
    consumes a message."""

    def __init__(self, on_color_set: Callable[[], None]) -> None:
        super().__init__()
        self._on_color_set = on_color_set

    def nativeEventFilter(self, event_type, message):  # noqa: N802 (Qt override)
        if event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == _WM_SETTINGCHANGE and msg.lParam:
                try:
                    area = ctypes.wstring_at(msg.lParam)
                except OSError:   # another app's malformed broadcast
                    area = None
                if area == _COLOR_SET:
                    self._on_color_set()
        return False, 0


class TaskbarThemeWatcher(QObject):
    """The taskbar theme, kept current: ``changed(theme)`` fires when
    Windows switches its mode. Installs a native event filter on the
    application; a real top-level Qt window (the overlay window, the
    pre-created flyout) must exist to receive the broadcast."""

    changed = Signal(str)   # "light" | "dark"

    def __init__(self, parent: QObject | None = None, *,
                 read: Callable[[], str] = taskbar_theme) -> None:
        super().__init__(parent)
        self._read = read
        self._theme = read()
        self._filter = _SettingChangeFilter(self._reread)
        self._app = QCoreApplication.instance()
        if self._app is not None:
            self._app.installNativeEventFilter(self._filter)
            # Out with the watcher: a filter left behind would call into a
            # deleted QObject on the next WM_SETTINGCHANGE.
            app, native_filter = self._app, self._filter
            self.destroyed.connect(lambda *_: app.removeNativeEventFilter(native_filter))

    def theme(self) -> str:
        return self._theme

    def uninstall(self) -> None:
        if self._app is not None:
            self._app.removeNativeEventFilter(self._filter)
            self._app = None

    def _reread(self) -> None:
        # Windows sends one WM_SETTINGCHANGE per top-level window: only a
        # real change is announced.
        theme = self._read()
        if theme != self._theme:
            self._theme = theme
            self.changed.emit(theme)
