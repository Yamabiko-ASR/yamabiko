"""Window chrome for Settings and its compact dialogs: the caption painted on
the window's own client area.

The window runs its client area under the title bar (Qt's
``ExpandedClientAreaHint``, with ``CustomizeWindowHint`` and no title or
button hints so Qt draws no caption of its own). Yamabiko paints the 16 px
app mark, the 12 px window title and the 46 x 32 caption buttons; dragging an
empty part of the caption moves the window (``QWindow.startSystemMove``) and
a double-click maximizes the Settings window. Qt 6.11 reports such a window
5 to 7 px wider than its real Win32 client, so :class:`ClientAreaFit` insets
the root layout's right edge by the difference. The DWM border takes the
theme's ``frame`` tone; the 1 px non-client strip Qt leaves inside it is
filled with the ground (:func:`nonclient_event`). Probes:
.superpowers/sdd/probes/p3_titlebar_probe*.py. The tray flyout asks DWM for
small rounded corners (:func:`set_corner_preference`), the menus for square
ones (:class:`SquareCorners`). The series bar takes the foreground when it
is summoned (:func:`take_foreground`) and gives it back to the app it took
it from when it hides (:func:`hand_back_foreground`).
"""
from __future__ import annotations

import ctypes
import functools
import logging
import os
import sys
from typing import Callable, NamedTuple, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QDialog, QHBoxLayout, QLabel, QLayout,
    QVBoxLayout, QWidget,
)

from .controls import mincho_font, ui_font
from .landscape import LandscapeBand
from .seal import Seal
from .theme import window_qss

_log = logging.getLogger(__name__)

APP_RED = "#d2452f"
CAPTION_H = 32
BAND_H = 80
_BUTTON_W = 46
_CLOSE_HOVER = "#c42b1c"
_DWMWA_BORDER_COLOR = 34
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWA_CLOAK = 13
# DWM_WINDOW_CORNER_PREFERENCE: small rounded corners (about 4 px) with
# DWM's tighter soft shadow (the tray flyout; probe p4_shadow.py); square
# corners (the menus, screenshots 05, 06, 13 and 14).
DWMWCP_DONOTROUND = 1
DWMWCP_ROUNDSMALL = 3
_WM_NCPAINT = 0x0085
_WM_NCACTIVATE = 0x0086
_RDW_INVALIDATE_FRAME = 0x0001 | 0x0400  # RDW_INVALIDATE | RDW_FRAME
_ALL_FONT_PROPERTIES = 0xFFFFF  # QFont::AllPropertiesResolved (PySide6 has no enum for it)


def paint_app_mark(painter: QPainter, rect: QRectF) -> None:
    """The 16 px app icon (vermilion tile, white 山) scaled into ``rect``."""
    s = rect.width() / 16.0
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.topLeft())
    painter.scale(s, s)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(APP_RED))
    painter.drawRoundedRect(QRectF(0, 0, 16, 16), 3.5, 3.5)
    glyph = QPainterPath()
    glyph.moveTo(3, 7)
    for x, y in ((5, 7), (5, 11), (7, 11), (7, 3), (9, 3), (9, 11), (11, 11),
                 (11, 7), (13, 7), (13, 13), (3, 13)):
        glyph.lineTo(x, y)
    glyph.closeSubpath()
    painter.setBrush(QColor("#ffffff"))
    painter.drawPath(glyph)
    painter.restore()


def paint_app_icon(painter: QPainter, rect: QRectF) -> None:
    """The app icon at its 128 px design (assets/icon.svg: vermilion tile with
    26 px corners, white 山, one thick white ridge line clipped to the tile)
    scaled into ``rect`` (setup's 64 px Welcome icon)."""
    s = rect.width() / 128.0
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.topLeft())
    painter.scale(s, s)
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(0, 0, 128, 128), 26, 26)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(APP_RED))
    painter.drawPath(tile)
    glyph = QPainterPath()
    glyph.moveTo(26, 34)
    for x, y in ((42, 34), (42, 52), (56, 52), (56, 8), (72, 8), (72, 52), (86, 52),
                 (86, 34), (102, 34), (102, 68), (26, 68)):
        glyph.lineTo(x, y)
    glyph.closeSubpath()
    painter.setBrush(QColor("#ffffff"))
    painter.drawPath(glyph)
    ridge = QPainterPath(QPointF(-6, 114))
    for x, y in ((30, 94), (46, 102), (80, 84), (134, 110)):
        ridge.lineTo(x, y)
    pen = QPen(QColor("#ffffff"), 10)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setClipPath(tile)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(ridge)
    painter.restore()


def _native_windows() -> bool:
    return sys.platform == "win32" and QGuiApplication.platformName() == "windows"


def expand_client_area(window: QWidget) -> None:
    """Let ``window`` paint under its title bar with no Qt-drawn caption.
    Call before the window is first shown."""
    window.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)
    kind = window.windowType()
    if kind not in (Qt.WindowType.Window, Qt.WindowType.Dialog):
        kind = Qt.WindowType.Window
    window.setWindowFlags(
        kind | Qt.WindowType.CustomizeWindowHint
        | Qt.WindowType.ExpandedClientAreaHint | Qt.WindowType.NoTitleBarBackgroundHint
    )


def client_width(window: QWidget) -> Optional[int]:
    """The real Win32 client width in logical px; None off the native
    Windows platform or when the query fails."""
    if not _native_windows():
        return None
    from ctypes import wintypes

    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetClientRect(wintypes.HWND(int(window.winId())), ctypes.byref(rect)):
        return None
    return round(rect.right / window.devicePixelRatioF())


class ClientAreaFit(QObject):
    """Keep ``layout``'s right margin equal to the strip of ``window`` that
    lies past its real client, so right-anchored content stays visible.
    ``measure`` returns the client width (None: no correction)."""

    def __init__(self, window: QWidget, layout: QLayout,
                 measure: Callable[[QWidget], Optional[int]] = client_width) -> None:
        super().__init__(window)
        self._window = window
        self._layout = layout
        self._measure = measure
        window.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        # Type first: when the garbage collector breaks the window <-> filter
        # cycle it may clear this object's attributes before the window's
        # C++ teardown delivers its last events (DynamicPropertyChange,
        # ChildRemoved) here.
        if event.type() in (
            QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.WindowStateChange,
        ) and watched is self._window:
            self.refit()
        return False

    def refit(self) -> None:
        width = self._measure(self._window)
        hidden = max(0, self._window.width() - width) if width is not None else 0
        margins = self._layout.contentsMargins()
        if margins.right() != hidden:
            self._layout.setContentsMargins(margins.left(), margins.top(), hidden, margins.bottom())
            # The layout already handled this resize; re-lay out now so the
            # right edge is never clipped for a frame.
            self._layout.activate()


_GWL_STYLE = -16
_WS_SYSMENU = 0x00080000
_WS_MINIMIZEBOX = 0x00020000
_WS_MAXIMIZEBOX = 0x00010000
_SWP_REFRESH_FRAME = 0x0001 | 0x0002 | 0x0004 | 0x0020  # NOSIZE|NOMOVE|NOZORDER|FRAMECHANGED
_SWP_NOACTIVATE = 0x0010


def add_window_menu(window: QWidget, *, min_max: bool, minimize: bool = False) -> bool:
    """Give an expanded-client window back the Win32 styles Qt drops with
    CustomizeWindowHint: the system menu (Alt+Space) and, with ``min_max``,
    minimize/maximize (taskbar-click minimize, Win+arrow snapping). With
    ``minimize`` alone the window gets the minimize box and never the
    maximize box (a fixed-size window: setup), so the taskbar click minimizes
    it and Win+Up does not try to maximize it. DWM still draws no caption
    (probe 3 --styles). Idempotent; call on every show, after the native show
    (ErrorWindow.setVisible). True when the styles are as asked; False off the
    native Windows platform."""
    if not _native_windows():
        return False
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    hwnd = wintypes.HWND(int(window.winId()))
    style = user32.GetWindowLongW(hwnd, _GWL_STYLE) & 0xFFFFFFFF
    if min_max:
        wanted, unwanted = _WS_SYSMENU | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX, 0
    elif minimize:
        wanted, unwanted = _WS_SYSMENU | _WS_MINIMIZEBOX, _WS_MAXIMIZEBOX
    else:
        wanted, unwanted = _WS_SYSMENU, 0
    target = (style | wanted) & ~unwanted
    if target != style:
        user32.SetWindowLongW(hwnd, _GWL_STYLE, ctypes.c_long(target).value)
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, _SWP_REFRESH_FRAME)
    style = user32.GetWindowLongW(hwnd, _GWL_STYLE) & 0xFFFFFFFF
    return style & wanted == wanted and not style & unwanted


def set_border_color(window: QWidget, hex_color: str) -> bool:
    """Draw the DWM window edge in ``hex_color`` (Windows 11). True when DWM
    accepted it; False off the native Windows platform."""
    if not _native_windows():
        return False
    color = QColor(hex_color)
    colorref = ctypes.c_uint(color.red() | (color.green() << 8) | (color.blue() << 16))
    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        ctypes.c_void_p(int(window.winId())), _DWMWA_BORDER_COLOR,
        ctypes.byref(colorref), ctypes.sizeof(colorref),
    )
    if result != 0:
        _log.debug("DWMWA_BORDER_COLOR rejected: 0x%08x", result & 0xFFFFFFFF)
    return result == 0


def set_corner_preference(window: QWidget, preference: int) -> bool:
    """Ask DWM for ``preference`` corners on ``window`` (Windows 11
    DWM_WINDOW_CORNER_PREFERENCE, for example DWMWCP_ROUNDSMALL) and refresh
    its frame (SWP_FRAMECHANGED) so a shown window takes them at once. Call
    it after show, or from the window's Show event (SquareCorners): set
    before the first show it has no effect (probe
    .superpowers/sdd/probes/p4_shadow.py). True when DWM accepted it; False
    off the native Windows platform."""
    if not _native_windows():
        return False
    from ctypes import wintypes

    value = ctypes.c_int(preference)
    hwnd = int(window.winId())
    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        ctypes.c_void_p(hwnd), _DWMWA_WINDOW_CORNER_PREFERENCE,
        ctypes.byref(value), ctypes.sizeof(value),
    )
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    user32.SetWindowPos(wintypes.HWND(hwnd), None, 0, 0, 0, 0,
                        _SWP_REFRESH_FRAME | _SWP_NOACTIVATE)
    if result != 0:
        _log.debug("DWMWA_WINDOW_CORNER_PREFERENCE rejected: 0x%08x", result & 0xFFFFFFFF)
    return result == 0


# INPUT.type for a mouse event (SendInput).
_INPUT_MOUSE = 0


@functools.cache
def _foreground_api():
    """Private user32 handle (argtypes stay local) and the INPUT struct for
    take_foreground and the foreground hand-back."""
    from ctypes import wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t)]  # ULONG_PTR

    class INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]  # MOUSEINPUT is the union's largest member

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", INPUTUNION)]  # 40 bytes on x64

    user32 = ctypes.WinDLL("user32")
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindow.argtypes = (wintypes.HWND,)
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetClassNameW.restype = ctypes.c_int
    return user32, INPUT


def take_foreground(window: QWidget) -> bool:
    """Make the shown ``window`` the foreground window although this process
    did not receive the input that summoned it (Ctrl+Alt+S reaches the
    pipeline's low-level keyboard hook, not this process): one empty mouse
    INPUT (no move, no button, so nothing reaches the window under the
    cursor) makes this process the last input source, which Windows requires
    for SetForegroundWindow (PowerToys' technique; probe
    .superpowers/sdd/probes/p5_probe_report.md Q1: 14 of 14, against 0 of 9
    with activateWindow alone). Call it after show() and before
    activateWindow(). True when Windows accepted it; False off the native
    Windows platform or when Windows refused (the window then shows without
    the keyboard)."""
    if not _native_windows():
        return False
    user32 = _nudge()
    if user32.SetForegroundWindow(int(window.winId())):
        return True
    _log.info("Windows refused the foreground; the window shows without the keyboard")
    return False


def _nudge():
    """Send one empty mouse INPUT so this process is the last input source,
    which SetForegroundWindow then accepts (take_foreground, the hand-back);
    returns the declared user32 handle."""
    user32, input_struct = _foreground_api()
    nudge = input_struct(type=_INPUT_MOUSE)
    if user32.SendInput(1, ctypes.byref(nudge), ctypes.sizeof(input_struct)) != 1:
        # Windows rejected the input. This return value does not report UIPI
        # blocking it toward an elevated (higher-integrity) foreground
        # window; the refused SetForegroundWindow after it is the log line
        # that records that case.
        _log.info("foreground nudge not delivered; the window may not get the keyboard")
    return user32


def _window_pid(hwnd: int) -> int:
    """The id of the process that owns window ``hwnd``."""
    from ctypes import wintypes

    pid = wintypes.DWORD()
    _foreground_api()[0].GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _foreground_window(user32) -> tuple[int, bool]:
    """The foreground window's handle (0 for none) and whether this process
    owns it."""
    hwnd = user32.GetForegroundWindow() or 0
    if not hwnd:
        return 0, False
    return hwnd, _window_pid(hwnd) == os.getpid()


def foreign_foreground() -> int:
    """The foreground window when another process owns it (the video player
    the user watches when Ctrl+Alt+S summons the series bar), else 0: no
    foreground window, one of this process's own, or off the native Windows
    platform. The bar keeps it to hand the foreground back when it hides
    (hand_back_foreground)."""
    if not _native_windows():
        return 0
    hwnd, ours = _foreground_window(_foreground_api()[0])
    return 0 if ours else hwnd


# The shell's taskbars and its notification-area overflow: from the short
# menu's Pick a series… route one of them is in front at the summon, and
# giving the foreground back to it reopens the overflow (visual check 2,
# item 6). The bar keeps no such window.
SHELL_WINDOW_CLASSES = frozenset({
    "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "NotifyIconOverflowWindow",
    "TopLevelWindowForOverflowXamlIsland",
})


def window_class(hwnd: int) -> str:
    """The Win32 class name of ``hwnd``: "" for no window, a handle that is
    gone, or off the native Windows platform."""
    if not _native_windows() or not hwnd:
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    length = _foreground_api()[0].GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value if length else ""


# The desktop's own windows, never an app to give the foreground to.
_DESKTOP_CLASSES = frozenset({"Progman", "WorkerW"})
_DWMWA_CLOAKED = 14
_GWL_EXSTYLE = -20
_GW_OWNER = 4
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_APPWINDOW = 0x00040000
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TRANSPARENT = 0x00000020


def _ordinary_app_styles(exstyle: int, owned: bool) -> bool:
    """The extended-style part of _is_app_window: not a tool window, not a
    window that never activates (WS_EX_NOACTIVATE: an on-screen keyboard),
    not a click-through overlay (WS_EX_TRANSPARENT: a game or GPU overlay),
    and unowned (or marked WS_EX_APPWINDOW)."""
    if exstyle & (_WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE | _WS_EX_TRANSPARENT):
        return False
    return not owned or bool(exstyle & _WS_EX_APPWINDOW)


@functools.cache
def _zorder_api():
    """The foreground handle with the z-order walk's calls declared, a
    private dwmapi handle and the EnumWindows callback type."""
    from ctypes import wintypes

    user32 = _foreground_api()[0]
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = (enum_proc, wintypes.LPARAM)
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetWindow.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetWindowRect.restype = wintypes.BOOL
    dwmapi = ctypes.WinDLL("dwmapi")
    dwmapi.DwmGetWindowAttribute.argtypes = (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p,
                                             wintypes.DWORD)
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long  # HRESULT
    return user32, dwmapi, enum_proc


class _WindowFacts(NamedTuple):
    """What the z-order walk reads of one top-level window."""

    visible: bool
    iconic: bool
    exstyle: int
    owned: bool
    cloaked: bool
    cls: str
    has_area: bool


def _window_facts(hwnd: int) -> _WindowFacts:
    """Window ``hwnd``'s visibility, styles, owner, DWM cloak, class and
    rect, as _is_app_window and window_shows weigh them."""
    from ctypes import wintypes

    user32, dwmapi, _proc = _zorder_api()
    cloaked = ctypes.c_int(0)
    cloak_read = dwmapi.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked),
                                              ctypes.sizeof(cloaked)) == 0
    rect = wintypes.RECT()
    rect_read = bool(user32.GetWindowRect(hwnd, ctypes.byref(rect)))
    return _WindowFacts(
        visible=bool(user32.IsWindowVisible(hwnd)),
        iconic=bool(user32.IsIconic(hwnd)),
        exstyle=user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & 0xFFFFFFFF,
        owned=bool(user32.GetWindow(hwnd, _GW_OWNER)),
        cloaked=cloak_read and bool(cloaked.value),
        cls=window_class(hwnd),
        has_area=rect_read and rect.right > rect.left and rect.bottom > rect.top,
    )


def _shows(facts: _WindowFacts) -> bool:
    """The part of _is_app_window that window_shows checks alone: visible,
    not minimized and not cloaked by DWM."""
    return facts.visible and not facts.iconic and not facts.cloaked


def _is_app_window(hwnd: int) -> bool:
    """Whether ``hwnd`` is an ordinary interactive window of any process,
    this one included (Settings counts; lead ruling on visual check 4): it
    shows (_shows), has the extended styles of _ordinary_app_styles (so not
    the bar, the flyout, a menu or the click-through subtitle window), is not
    the shell's tray or overflow nor the desktop, and has a non-empty rect."""
    facts = _window_facts(hwnd)
    return (_shows(facts)
            and _ordinary_app_styles(facts.exstyle, facts.owned)
            and facts.cls not in SHELL_WINDOW_CLASSES | _DESKTOP_CLASSES
            and facts.has_area)


def window_shows(hwnd: int) -> bool:
    """Whether window ``hwnd`` still shows: it exists, is visible, is not
    minimized and is not cloaked by DWM. False for 0, a window that is gone,
    or off the native Windows platform. The bar checks the window it kept
    with it before giving that window the foreground back: one that has
    closed, hidden or been minimized counts as none kept, while a dialog or
    any other window that still shows gets it back."""
    if not _native_windows() or not hwnd:
        return False
    if not _foreground_api()[0].IsWindow(hwnd):
        return False
    return _shows(_window_facts(hwnd))


def _enum_windows() -> list[int]:
    """Every top-level window, topmost first (EnumWindows order)."""
    user32, _dwmapi, enum_proc = _zorder_api()
    found: list[int] = []

    def visit(hwnd, _lparam):
        found.append(hwnd)
        return True

    callback = enum_proc(visit)  # kept alive for the whole EnumWindows call
    user32.EnumWindows(callback, 0)
    return found


def topmost_app_window() -> int:
    """The topmost ordinary window in the z-order (EnumWindows order; see
    _is_app_window): another app's, or this process's Settings or a dialog;
    0 for none or off the native Windows platform. The bar gives it the
    foreground when it hides with nothing kept (summoned from the tray: the
    menu, the overflow or the flyout were in front), so the window the user
    was in gets the keyboard back and the overflow closes (lead rulings on
    visual checks 3 and 4)."""
    if not _native_windows():
        return 0
    return next((hwnd for hwnd in _enum_windows() if _is_app_window(hwnd)), 0)


def foreground_widget() -> QWidget | None:
    """This process's window that is the foreground window (QWidget.find on
    its handle), else None: another process's window, none, or off the
    native Windows platform. The bar reads it as it starts to hide: only a
    foreground left on the bar itself or on the click-through subtitle window
    is handed back."""
    if not _native_windows():
        return None
    hwnd, ours = _foreground_window(_foreground_api()[0])
    return QWidget.find(hwnd) if ours else None


def foreground_is_app_window() -> bool:
    """Whether the foreground window is an ordinary window (see
    _is_app_window): another app's, or this process's Settings or a dialog;
    False for the bar, the flyout, menus, the subtitle window, the shell's
    tray and overflow, the desktop, none, or off the native Windows
    platform. After a bar that kept nothing hides, such a window was
    Windows' own pick and keeps the foreground; any other pick (the subtitle
    window, the tray overflow) gets the z-order fallback."""
    if not _native_windows():
        return False
    hwnd = _foreground_api()[0].GetForegroundWindow() or 0
    return bool(hwnd) and _is_app_window(hwnd)


def hand_back_foreground(hwnd: int) -> bool:
    """Give the foreground to ``hwnd`` (the window the bar kept, or the
    topmost app window) when it still exists and shows. It runs after the
    bar's native hide, when Windows may already have given the foreground to
    another process (the tray overflow) and this process may not have had the
    last input (a pointer move over the video while the names loaded), so it
    sends take_foreground's empty mouse input first. Without it, hiding the
    bar can leave the foreground on the subtitle window or the overflow, and
    the video player loses the keyboard. True when Windows accepted it; False
    off the native Windows platform."""
    if not _native_windows() or not hwnd:
        return False
    user32 = _foreground_api()[0]
    if not (user32.IsWindow(hwnd) and user32.IsWindowVisible(hwnd)):
        return False
    if _nudge().SetForegroundWindow(hwnd):
        return True
    _log.info("Windows refused to give the foreground back")
    return False


def set_cloak(window: QWidget, cloaked: bool) -> bool:
    """Hide the shown ``window`` from DWM composition (``cloaked``) or show
    it again; Qt and Windows still treat it as shown. The series bar is
    cloaked from show until its first paint so no empty outline flashes
    (probe .superpowers/sdd/probes/p5_probe_report.md, Q2 finding 2). True
    when DWM accepted it; False off the native Windows platform."""
    if not _native_windows():
        return False
    value = ctypes.c_int(1 if cloaked else 0)
    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        ctypes.c_void_p(int(window.winId())), _DWMWA_CLOAK,
        ctypes.byref(value), ctypes.sizeof(value),
    )
    if result != 0:
        _log.debug("DWMWA_CLOAK rejected: 0x%08x", result & 0xFFFFFFFF)
    return result == 0


class SquareCorners(QObject):
    """Keep ``window`` (a menu) square-cornered on every show. Qt's windows11
    style rounds every framed top-level window on each Show (DWMWCP_ROUND,
    from its application event filter); this filter on the window runs after
    that one and before the native show, so DWMWCP_DONOTROUND holds and the
    menu keeps its drop shadow. Set after the show instead, the shadow goes
    and the next show rounds the menu again (probe
    .superpowers/sdd/probes/p4fix_menu_show_probe.py)."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        window.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        # Type first, as in ClientAreaFit (teardown can clear the attributes).
        if event.type() == QEvent.Type.Show and watched is self._window:
            set_corner_preference(self._window, DWMWCP_DONOTROUND)
        return False


@functools.cache
def _frame_api():
    """Private user32/gdi32 handles (argtypes stay local) for the non-client
    strip."""
    from ctypes import wintypes

    user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
    user32.DefWindowProcW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                      wintypes.LPARAM)
    user32.DefWindowProcW.restype = wintypes.LPARAM  # LRESULT
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    user32.GetWindowDC.argtypes = (wintypes.HWND,)
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
    user32.FillRect.argtypes = (wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH)
    user32.RedrawWindow.argtypes = (wintypes.HWND, ctypes.c_void_p, ctypes.c_void_p,
                                    wintypes.UINT)
    gdi32.CreateSolidBrush.argtypes = (wintypes.COLORREF,)
    gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
    gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
    gdi32.ExcludeClipRect.argtypes = (wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int)
    return user32, gdi32


def nonclient_event(event_type, message, ground: str) -> Optional[int]:
    """Answer two non-client messages of an expanded-client window (call from
    ``nativeEvent``): the LRESULT when handled, else None.

    Qt 6.11 leaves a 1 px non-client strip inside the DWM border on the
    left, right and bottom that nothing paints (a white line in the dark
    theme): WM_NCPAINT fills it with ``ground`` (DWM still lightens it a
    little, so it reads as the border's tone). On deactivation
    DefWindowProc paints an inactive caption over the top client rows:
    WM_NCACTIVATE keeps the state change but skips that repaint (lParam -1).
    Probes: .superpowers/sdd/probes/p3_fix_ncpaint*.py."""
    if bytes(event_type) != b"windows_generic_MSG":
        return None
    from ctypes import wintypes

    msg = wintypes.MSG.from_address(int(message))
    if msg.message not in (_WM_NCPAINT, _WM_NCACTIVATE):
        return None
    user32, gdi32 = _frame_api()
    if msg.message == _WM_NCACTIVATE:
        return user32.DefWindowProcW(msg.hWnd, msg.message, msg.wParam, -1)
    window, client, origin = wintypes.RECT(), wintypes.RECT(), wintypes.POINT(0, 0)
    user32.GetWindowRect(msg.hWnd, ctypes.byref(window))
    user32.GetClientRect(msg.hWnd, ctypes.byref(client))
    user32.ClientToScreen(msg.hWnd, ctypes.byref(origin))
    left, top = origin.x - window.left, origin.y - window.top
    color = QColor(ground)
    hdc = user32.GetWindowDC(msg.hWnd)
    gdi32.ExcludeClipRect(hdc, left, top, left + client.right, top + client.bottom)
    brush = gdi32.CreateSolidBrush(color.red() | (color.green() << 8) | (color.blue() << 16))
    whole = wintypes.RECT(0, 0, window.right - window.left, window.bottom - window.top)
    user32.FillRect(hdc, ctypes.byref(whole), brush)
    gdi32.DeleteObject(brush)
    user32.ReleaseDC(msg.hWnd, hdc)
    return 0


def repaint_nonclient(window: QWidget) -> None:
    """Repaint ``window``'s non-client strip (after a theme change)."""
    if _native_windows() and window.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
        _frame_api()[0].RedrawWindow(int(window.winId()), None, None, _RDW_INVALIDATE_FRAME)


def _pin_font(widget: QWidget, font: QFont) -> None:
    """Set ``font`` on ``widget`` with every property marked as the
    widget's own. Under a style sheet Qt 6.11 resolves the unmarked ones
    against ``QApplication.font()``, not the parent's font: a font whose
    family and size were inherited would drop to the app font."""
    pinned = QFont(font)
    pinned.setResolveMask(_ALL_FONT_PROPERTIES)
    widget.setFont(pinned)


def apply_window_qss(window: QWidget, palette: dict) -> None:
    """Set the window QSS for ``palette`` on ``window``, keeping every
    descendant's font.

    Qt 6.11: re-setting a window's style sheet re-polishes its subtree, and
    the style sheet style puts each child's saved font back through
    ``setFont``, which resolves an inherited font against
    ``QApplication.font()`` instead of the window's -- labels, fields and
    buttons drop from 13 px to 9 pt until the window reopens. Each child
    gets back the font it showed, pinned: the primary button marks only its
    QSS weight as its own, so a plain restore still took the app's family
    and size (probes: .superpowers/sdd/probes/p3_vis_theme_font*.py,
    p3_final_fonts.py)."""
    fonts = [(child, child.font()) for child in window.findChildren(QWidget)]
    window.setStyleSheet(window_qss(palette))
    for child, font in fonts:
        if child.font() != font:
            _pin_font(child, font)


class CaptionButton(QAbstractButton):
    """One 46 x 32 caption button: minimize, maximize (restore glyph while
    maximized) or close (red on hover)."""

    _NAMES = {"min": "Minimize", "max": "Maximize", "close": "Close"}

    def __init__(self, kind: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self._palette = palette
        self.setFixedSize(_BUTTON_W, CAPTION_H)
        self.setAttribute(Qt.WA_Hover, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAccessibleName(self._NAMES[kind])

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        glyph = QColor(self._palette["ink"])
        if self.underMouse():
            if self._kind == "close":
                p.fillRect(self.rect(), QColor(_CLOSE_HOVER))
                glyph = QColor("#ffffff")
            else:
                p.fillRect(self.rect(), QColor(self._palette["hover"]))
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(glyph, 1.0))
        p.setBrush(Qt.NoBrush)
        p.translate((self.width() - 10) / 2.0, (self.height() - 10) / 2.0)
        if self._kind == "min":
            p.drawLine(QPointF(0, 5.5), QPointF(10, 5.5))
        elif self._kind == "max":
            maximized = self.window().isMaximized()
            # The accessible name follows the glyph, also when Win+Up or the
            # system menu maximized the window (its caption repaints then).
            name = "Restore" if maximized else self._NAMES["max"]
            if self.accessibleName() != name:
                self.setAccessibleName(name)
            if maximized:
                p.drawRoundedRect(QRectF(0.5, 2.5, 7, 7), 1.0, 1.0)
                back = QPainterPath(QPointF(2.5, 2.0))
                back.lineTo(2.5, 0.5)
                back.lineTo(9.5, 0.5)
                back.lineTo(9.5, 7.5)
                back.lineTo(8.0, 7.5)
                p.drawPath(back)
            else:
                p.drawRoundedRect(QRectF(0.5, 0.5, 9, 9), 1.5, 1.5)
        else:
            p.drawLine(QPointF(0.5, 0.5), QPointF(9.5, 9.5))
            p.drawLine(QPointF(9.5, 0.5), QPointF(0.5, 9.5))
        p.end()


def _toggle_maximized(window: QWidget) -> None:
    """Maximize ``window``; restore it when it is maximized."""
    if window.isMaximized():
        window.showNormal()
    else:
        window.showMaximized()


class _WindowMover(QObject):
    """Event filter on a caption area: a drag past the system drag distance
    on an empty part moves the window; a double-click maximizes it when
    ``maximize`` is set. Presses on child buttons never reach the area."""

    def __init__(self, area: QWidget, *, maximize: bool) -> None:
        super().__init__(area)
        self._area = area
        self._maximize = maximize
        self._press = None
        area.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
            self._press = event.position().toPoint()
        elif kind == QEvent.Type.MouseMove and self._press is not None:
            moved = (event.position().toPoint() - self._press).manhattanLength()
            if event.buttons() & Qt.LeftButton and moved >= QApplication.startDragDistance():
                self._press = None
                handle = self._area.window().windowHandle()
                if handle is not None:
                    handle.startSystemMove()
                return True
        elif kind == QEvent.Type.MouseButtonRelease:
            self._press = None
        elif (kind == QEvent.Type.MouseButtonDblClick and self._maximize
              and event.button() == Qt.LeftButton):
            _toggle_maximized(self._area.window())
            return True
        return False


class TitleBand(LandscapeBand):
    """Settings' 80 px title band: the landscape; the caption on its sky (app
    mark, window title, caption buttons); the BIZ UDPMincho Bold heading with
    the seal; and a right-hand slot (Theme)."""

    def __init__(self, title: str, heading_jp: str, heading: str, palette: dict,
                 parent: QWidget | None = None) -> None:
        super().__init__(palette, parent)
        self._title = title
        self._heading_jp = heading_jp
        self._heading = heading
        self.setFixedHeight(BAND_H)
        self.setAccessibleName(f"{heading_jp} {heading}")
        self._buttons = {kind: CaptionButton(kind, palette, self) for kind in ("min", "max", "close")}
        self._buttons["min"].clicked.connect(lambda: self.window().showMinimized())
        self._buttons["max"].clicked.connect(lambda: _toggle_maximized(self.window()))
        self._buttons["close"].clicked.connect(lambda: self.window().close())
        self._seal = Seal(26, self)
        self._slot: QWidget | None = None
        self._mover = _WindowMover(self, maximize=True)

    def caption_button(self, kind: str) -> CaptionButton:
        return self._buttons[kind]

    def set_slot(self, widget: QWidget) -> None:
        widget.setParent(self)
        self._slot = widget
        widget.show()
        self._place()

    def apply_palette(self, palette: dict) -> None:
        super().apply_palette(palette)
        for button in self._buttons.values():
            button.apply_palette(palette)

    def _place(self) -> None:
        x = self.width()
        for kind in ("close", "max", "min"):
            x -= _BUTTON_W
            self._buttons[kind].move(x, 0)
        metrics = QFontMetrics(mincho_font(21, bold=True))
        seal_x = (16 + metrics.horizontalAdvance(self._heading_jp) + 8
                  + metrics.horizontalAdvance(self._heading) + 12)
        self._seal.move(seal_x, 37)
        if self._slot is not None:
            hint = self._slot.sizeHint()
            self._slot.setGeometry(self.width() - 16 - hint.width(), 38, hint.width(), hint.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._place()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        paint_app_mark(p, QRectF(12, 8, 16, 16))
        ink = QColor(self._palette["ink"])
        p.setPen(ink)
        p.setFont(ui_font(12))
        p.drawText(QRectF(38, 0, 400, CAPTION_H), Qt.AlignLeft | Qt.AlignVCenter, self._title)
        heading = mincho_font(21, bold=True)
        p.setFont(heading)
        line = QRectF(16, 36, self.width() - 32, 28)
        p.drawText(line, Qt.AlignLeft | Qt.AlignVCenter, self._heading_jp)
        jp_w = QFontMetrics(heading).horizontalAdvance(self._heading_jp)
        p.drawText(line.adjusted(jp_w + 8, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, self._heading)
        p.end()


class DialogCaption(QWidget):
    """A compact dialog's 32 px caption: app mark, title and close; with
    ``minimize``, a minimize button left of close (setup)."""

    closeRequested = Signal()

    def __init__(self, title: str, palette: dict, parent: QWidget | None = None, *,
                 minimize: bool = False) -> None:
        super().__init__(parent)
        self._title = title
        self._palette = palette
        self.setFixedHeight(CAPTION_H)
        self._buttons = {"close": CaptionButton("close", palette, self)}
        self._buttons["close"].clicked.connect(self.closeRequested)
        if minimize:
            self._buttons["min"] = CaptionButton("min", palette, self)
            self._buttons["min"].clicked.connect(lambda: self.window().showMinimized())
        self._mover = _WindowMover(self, maximize=False)

    def caption_button(self, kind: str) -> CaptionButton:
        return self._buttons[kind]

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        for button in self._buttons.values():
            button.apply_palette(palette)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # Close first, from the right.
        kinds = [kind for kind in ("close", "min") if kind in self._buttons]
        for index, kind in enumerate(kinds):
            self._buttons[kind].move(self.width() - (index + 1) * _BUTTON_W, 0)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        paint_app_mark(p, QRectF(12, 8, 16, 16))
        p.setPen(QColor(self._palette["ink"]))
        p.setFont(ui_font(12))
        p.drawText(QRectF(38, 0, self.width() - 38 - len(self._buttons) * _BUTTON_W, CAPTION_H),
                   Qt.AlignLeft | Qt.AlignVCenter, self._title)
        p.end()


class CompactDialog(QDialog):
    """Base of Settings' compact dialogs (save failed, model download, color
    picker): the painted caption on the ground, a BIZ UDPMincho heading,
    24 px body padding and an action row; fixed width, height to content."""

    def __init__(self, palette: dict, width: int, parent: QWidget | None = None, *,
                 title: str = "Yamabiko settings") -> None:
        super().__init__(parent)
        self.setObjectName("compactDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        expand_client_area(self)
        self.setFont(ui_font(13))
        self._palette = palette
        self._width = width
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.caption = DialogCaption(title, palette, self)
        self.caption.closeRequested.connect(self.reject)
        root.addWidget(self.caption)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(24, 6, 24, 0)
        self.body.setSpacing(0)
        root.addLayout(self.body)
        self.actions = QHBoxLayout()
        self.actions.setContentsMargins(24, 20, 20, 20)
        self.actions.setSpacing(8)
        root.addLayout(self.actions)
        self._fit = ClientAreaFit(self, root)
        self.setFixedWidth(width)
        self.apply_palette(palette)

    def heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("dialogTitle")
        label.setFont(mincho_font(18))
        label.setWordWrap(True)
        # The mockup's 24 px line box (BIZ UDPMincho's own line is 18 px).
        label.setMinimumHeight(24)
        self.body.addWidget(label)
        return label

    def add_text(self, text: str, *, muted: bool = False, top: int = 6) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        if muted:
            label.setObjectName("help")
        self.body.addSpacing(top)
        self.body.addWidget(label)
        return label

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        apply_window_qss(self, palette)
        self.caption.apply_palette(palette)

    def refit_height(self) -> None:
        """Fix the height to the content at the dialog's width (call after
        showing or hiding a row)."""
        layout = self.layout()
        layout.activate()
        if layout.hasHeightForWidth():
            self.setFixedHeight(layout.totalHeightForWidth(self._width))
        else:
            self.setFixedHeight(layout.totalSizeHint().height())

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._inherit_window_font()
        self.refit_height()
        if not event.spontaneous():
            self._center_on_parent()
        super().showEvent(event)
        set_border_color(self, self._palette["frame"])
        add_window_menu(self, min_max=False)

    def _inherit_window_font(self) -> None:
        """Give every widget in the dialog the font it inherits (its own
        properties kept), pinned. A dialog parented to Settings runs under
        Settings' style sheet style from its construction, and Qt resolves
        the fonts of the rows it and its subclass build against the app font
        instead of the dialog's: 12 px text (probes:
        .superpowers/sdd/probes/p3_final_dialogfont*.py). Parents come
        before their children, so each child inherits a settled font."""
        for child in self.findChildren(QWidget):
            if child.window() is self:
                _pin_font(child, child.font().resolve(child.parentWidget().font()))

    def _center_on_parent(self) -> None:
        """Center the fitted dialog on its parent window, kept inside the work
        area of the parent's screen (a parent dragged partly off it would
        take the dialog's buttons along). QDialog centers it before the
        refit, and with no visible window reporting both frame margins (an
        expanded client area reports a negative left one) it falls back to a
        guessed 10 x 40 px frame: about 40 px too high. The clamp takes the
        client rect too: Qt's frame rect of an expanded client area lies
        inside the visible window, and clamped alone it left the dialog
        5 device px past a screen edge at 150 % (probe:
        .superpowers/sdd/probes/p3_final_native_center.py)."""
        parent = self.parentWidget()
        if parent is None:
            return
        window = parent.window()
        self.move(self.pos() + window.geometry().center() - self.geometry().center())
        screen = window.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        shown = self.frameGeometry().united(self.geometry())
        x = min(max(shown.left(), avail.left()), max(avail.left(), avail.right() + 1 - shown.width()))
        y = min(max(shown.top(), avail.top()), max(avail.top(), avail.bottom() + 1 - shown.height()))
        if (x, y) != (shown.left(), shown.top()):
            self.move(self.pos() + QPoint(x - shown.left(), y - shown.top()))

    def nativeEvent(self, event_type, message):  # noqa: N802 (Qt override)
        result = nonclient_event(event_type, message, self._palette["ground"])
        if result is not None:
            return True, result
        return super().nativeEvent(event_type, message)
