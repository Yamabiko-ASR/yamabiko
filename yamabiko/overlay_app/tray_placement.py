"""Where the tray flyout and the right-click menu open, and the
reopen-race debounce.

Placement (plan phase 4, Decisions 9 and 59; PlatformProbe Q1): the flyout
opens on the screen that holds the tray icon, centered on the icon along the
taskbar, 12 px clear of the taskbar, and clamped inside that screen's work
area with a 12 px inset. Everything is in that screen's logical px: Qt's
QSystemTrayIcon.geometry() already is (measured at 150 %), and the logical
coordinates of two screens of different scale leave a gap between them, so
the math never mixes screens. An invalid icon rect anchors on the cursor.
With an auto-hidden taskbar the work area runs under the taskbar, so the
icon's taskbar-facing side is the reference line when it is nearer. The
right-click menu stands on the same line with no gap (above a bottom
taskbar, as screenshots 13 and 14), at the cursor along the taskbar when
the cursor is on the icon (else at the icon: a keyboard Context leaves the
pointer anywhere), inside the work area: Qt's own placement anchors it at
the cursor, so it covered the taskbar.

``taskbar_edge``, ``flyout_origin`` and ``menu_origin`` are pure on
``(x, y, w, h)`` tuples; ``place_flyout``, ``place_menu`` and ``cursor_in``
are the only Qt reads, and the tray calls them.

Reopen race (PlatformProbe Q2): a click on the tray icon while the flyout is
open deactivates (hides) it on mouse-down, then delivers Trigger on
mouse-up, which would open it again. ``TriggerDebounce`` swallows that
Trigger: when the flyout hides with the cursor over the icon, the next
Trigger within ``SWALLOW_S`` is dropped. The gap between the hide and the
Trigger follows the press length (23 ms to 244 ms measured), so a fixed
time window alone would be fragile. A DoubleClick clears the swallow (the
tray shows the flyout for it), so a leftover swallow never eats a click.
"""
from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QCursor, QGuiApplication

INSET = 12
SWALLOW_S = 1.0

Rect = tuple[int, int, int, int]  # x, y, width, height
Point = tuple[int, int]


def _center(rect: Rect) -> Point:
    """QRect.center(): the probe's measured placement uses it."""
    x, y, w, h = rect
    return x + (w - 1) // 2, y + (h - 1) // 2


def _gaps(screen: Rect, work: Rect) -> dict[str, int]:
    """How far the work area stops short of each screen edge."""
    sx, sy, sw, sh = screen
    wx, wy, ww, wh = work
    return {
        "bottom": (sy + sh) - (wy + wh),
        "top": wy - sy,
        "left": wx - sx,
        "right": (sx + sw) - (wx + ww),
    }


def taskbar_edge(screen: Rect, work: Rect, anchor: Point) -> str:
    """The taskbar's side of ``screen``: the gapped edge whose band (between
    the work area and the screen edge) holds ``anchor``, else where the work
    area is shortest (a docked app bar can be wider than the taskbar); with
    no gap (an auto-hidden taskbar, or a screen without one) the screen edge
    nearest ``anchor``."""
    gaps = _gaps(screen, work)
    wx, wy, ww, wh = work
    ax, ay = anchor
    in_band = {
        "bottom": ay >= wy + wh,
        "top": ay < wy,
        "left": ax < wx,
        "right": ax >= wx + ww,
    }
    holding = [edge for edge in gaps if gaps[edge] > 0 and in_band[edge]]
    if holding:
        return max(holding, key=gaps.get)
    edge = max(gaps, key=gaps.get)
    if gaps[edge] > 0:
        return edge
    sx, sy, sw, sh = screen
    distances = {
        "bottom": (sy + sh) - ay,
        "top": ay - sy,
        "left": ax - sx,
        "right": (sx + sw) - ax,
    }
    return min(distances, key=distances.get)


def _taskbar_line(
    icon: Rect | None, cursor: Point, screen: Rect, work: Rect,
) -> tuple[str, Point, tuple[int, int, int, int]]:
    """``(edge, anchor, (left, top, right, bottom))``: the taskbar's edge for
    the icon (None: the cursor) and the work area's sides, the taskbar's one
    moved to an auto-hidden taskbar's inner line."""
    wx, wy, ww, wh = work
    anchor = _center(icon) if icon is not None else cursor
    edge = taskbar_edge(screen, work, anchor)
    left, top, right, bottom = wx, wy, wx + ww, wy + wh
    if icon is not None and max(_gaps(screen, work).values()) <= 0:
        # Auto-hide: the slid-in taskbar covers the work area's edge; the
        # icon's taskbar-facing side is the taskbar's inner line.
        ix, iy, iw, ih = icon
        if edge == "bottom":
            bottom = min(bottom, iy)
        elif edge == "top":
            top = max(top, iy + ih)
        elif edge == "left":
            left = max(left, ix + iw)
        else:
            right = min(right, ix)
    return edge, anchor, (left, top, right, bottom)


def flyout_origin(
    size: tuple[int, int], icon: Rect | None, cursor: Point, screen: Rect, work: Rect,
) -> Point:
    """Top-left of a flyout of ``size`` for the tray icon rect ``icon`` (None:
    anchor on ``cursor``) on ``screen`` with its work area ``work``."""
    w, h = size
    wx, wy, ww, wh = work
    edge, (ax, ay), (left, top, right, bottom) = _taskbar_line(icon, cursor, screen, work)
    if edge == "bottom":
        x, y = ax - w // 2, bottom - INSET - h
    elif edge == "top":
        x, y = ax - w // 2, top + INSET
    elif edge == "left":
        x, y = left + INSET, ay - h // 2
    else:
        x, y = right - INSET - w, ay - h // 2
    # Inside the work area, 12 px in; a flyout larger than the area pins to
    # its left / top inset.
    x = max(wx + INSET, min(x, wx + ww - INSET - w))
    y = max(wy + INSET, min(y, wy + wh - INSET - h))
    return x, y


def menu_origin(
    size: tuple[int, int], icon: Rect | None, cursor: Point, screen: Rect, work: Rect,
) -> Point:
    """Top-left of the right-click menu of ``size``: on the taskbar's line
    found as for the flyout, its left (top, beside a side taskbar) edge at
    ``cursor`` when the cursor is on the icon, else at the icon's center (a
    keyboard Context, Shift+F10 or the Menu key, leaves the pointer
    anywhere), inside the work area."""
    w, h = size
    wx, wy, ww, wh = work
    edge, anchor, (left, top, right, bottom) = _taskbar_line(icon, cursor, screen, work)
    cx, cy = cursor if icon is None or _contains(icon, cursor) else anchor
    if edge == "bottom":
        x, y = cx, bottom - h
    elif edge == "top":
        x, y = cx, top
    elif edge == "left":
        x, y = left, cy
    else:
        x, y = right - w, cy
    x = max(wx, min(x, wx + ww - w))
    y = max(wy, min(y, wy + wh - h))
    return x, y


def _contains(rect: Rect, point: Point) -> bool:
    """QRect.contains(QPoint) on an ``(x, y, w, h)`` tuple."""
    x, y, w, h = rect
    px, py = point
    return x <= px < x + w and y <= py < y + h


def _rect(rect: QRect) -> Rect:
    return rect.x(), rect.y(), rect.width(), rect.height()


def _place(origin: Callable, icon_rect: QRect, size: QSize) -> QPoint:
    """``origin``'s top-left for the tray icon at ``icon_rect``
    (QSystemTrayIcon.geometry(); invalid: anchor on the cursor) and a window
    of ``size``, on the screen under the anchor (else the primary)."""
    cursor = QCursor.pos()
    icon = _rect(icon_rect) if icon_rect.isValid() else None
    anchor = QPoint(*_center(icon)) if icon is not None else cursor
    screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
    x, y = origin(
        (size.width(), size.height()), icon, (cursor.x(), cursor.y()),
        _rect(screen.geometry()), _rect(screen.availableGeometry()),
    )
    return QPoint(x, y)


def place_flyout(icon_rect: QRect, size: QSize) -> QPoint:
    """The flyout's top-left for the tray icon at ``icon_rect`` (flyout_origin)."""
    return _place(flyout_origin, icon_rect, size)


def place_menu(icon_rect: QRect, size: QSize) -> QPoint:
    """The right-click menu's top-left for the tray icon at ``icon_rect``
    (menu_origin)."""
    return _place(menu_origin, icon_rect, size)


def cursor_in(rect: QRect) -> bool:
    """True when the cursor is inside ``rect``; False for an invalid rect."""
    return rect.isValid() and rect.contains(QCursor.pos())


class TriggerDebounce:
    """Swallows the Trigger a click on the tray icon delivers right after the
    same click's mouse-down hid the flyout (see the module docstring)."""

    def __init__(self, *, window_s: float = SWALLOW_S,
                 now: Callable[[], float] = time.monotonic) -> None:
        self._window_s = window_s
        self._now = now
        self._until: float | None = None

    def flyout_hidden(self, cursor_in_icon: bool) -> None:
        """The flyout just hid; arm the swallow when the cursor is over the
        tray icon, else disarm it."""
        self._until = self._now() + self._window_s if cursor_in_icon else None

    def accept_trigger(self) -> bool:
        """A Trigger arrived: False drops it (once, inside the window); True
        lets the tray open (or close) the flyout. Either way it disarms."""
        until, self._until = self._until, None
        return until is None or self._now() >= until

    def double_click(self) -> None:
        """A DoubleClick arrived (the tray shows the flyout): disarm."""
        self._until = None
