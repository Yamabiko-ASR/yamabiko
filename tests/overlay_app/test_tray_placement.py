"""Tray flyout and right-click menu placement and the reopen-race debounce
(plan phase 4, Task 11).

Table-driven over the taskbar edges, auto-hide, the clamp, an invalid icon
rect and a second monitor, all in logical px. The first row is the probe's
measured placement (PlatformProbe Q1: 150 % main display, bottom taskbar,
Qt icon rect (2334, 1392, 32, 48), flyout at (2169, 954)); the second its
overflow case (clamped to x = 2188).
"""
from __future__ import annotations

import pytest

from yamabiko.overlay_app.tray_placement import (
    TriggerDebounce, flyout_origin, menu_origin, taskbar_edge,
)

SIZE = (360, 426)


@pytest.mark.parametrize(
    "case, icon, cursor, screen, work, edge, origin",
    [
        ("bottom, probe-measured", (2334, 1392, 32, 48), (0, 0),
         (0, 0, 2560, 1440), (0, 0, 2560, 1392), "bottom", (2169, 954)),
        ("bottom, icon in the open overflow, clamped right", (2411, 1215, 40, 40), (0, 0),
         (0, 0, 2560, 1440), (0, 0, 2560, 1392), "bottom", (2188, 954)),
        ("top, clamped right", (1800, 0, 32, 48), (0, 0),
         (0, 0, 1920, 1080), (0, 48, 1920, 1032), "top", (1548, 60)),
        ("left, clamped up", (0, 1000, 48, 32), (0, 0),
         (0, 0, 1920, 1080), (48, 0, 1872, 1080), "left", (60, 642)),
        ("right", (1872, 500, 48, 32), (0, 0),
         (0, 0, 1920, 1080), (0, 0, 1872, 1080), "right", (1500, 302)),
        ("auto-hide bottom, clear of the slid-in taskbar", (1700, 1032, 32, 48), (0, 0),
         (0, 0, 1920, 1080), (0, 0, 1920, 1080), "bottom", (1535, 594)),
        ("invalid icon rect, the cursor anchors", None, (500, 1070),
         (0, 0, 1920, 1080), (0, 0, 1920, 1032), "bottom", (320, 594)),
        ("second monitor left of the primary", (-100, 1040, 32, 40), (0, 0),
         (-1920, 0, 1920, 1080), (-1920, 0, 1920, 1040), "bottom", (-372, 602)),
        ("bottom taskbar under a wider top app bar, icon in the bottom band",
         (1700, 1032, 32, 48), (0, 0),
         (0, 0, 1920, 1080), (0, 300, 1920, 732), "bottom", (1535, 594)),
    ],
)
def test_flyout_origin(case, icon, cursor, screen, work, edge, origin):
    """Decisions 9 and 59. Seeded defects: center on x + w // 2 instead of
    QRect.center (the probe row is off by one), measure from the icon's top
    instead of the work area (the overflow row moves up), no clamp (the
    overflow row stays at 2250), the edge from the smallest gap, the largest
    gap winning over the band holding the icon (the app-bar row opens at the
    top), auto-hide measured from the screen edge (the auto-hide row lands on
    the taskbar), or the clamp against another screen's work area (the
    second-monitor row) -> red."""
    anchor = cursor if icon is None else (icon[0] + (icon[2] - 1) // 2,
                                          icon[1] + (icon[3] - 1) // 2)
    assert taskbar_edge(screen, work, anchor) == edge, case
    assert flyout_origin(SIZE, icon, cursor, screen, work) == origin, case


@pytest.mark.parametrize(
    "case, icon, cursor, screen, work, origin",
    [
        ("bottom: on the taskbar's top, left edge at the cursor", (1700, 1392, 32, 48),
         (1716, 1416), (0, 0, 2560, 1440), (0, 0, 2560, 1392), (1716, 1244)),
        ("bottom, clamped right", (2334, 1392, 32, 48), (2350, 1416),
         (0, 0, 2560, 1440), (0, 0, 2560, 1392), (2336, 1244)),
        ("icon in the open overflow: on the taskbar's top", (2411, 1215, 40, 40), (2431, 1235),
         (0, 0, 2560, 1440), (0, 0, 2560, 1392), (2336, 1244)),
        ("top, clamped right", (1800, 0, 32, 48), (1816, 24),
         (0, 0, 1920, 1080), (0, 48, 1920, 1032), (1696, 48)),
        ("left, clamped up", (0, 1000, 48, 32), (24, 1016),
         (0, 0, 1920, 1080), (48, 0, 1872, 1080), (48, 932)),
        ("right", (1872, 500, 48, 32), (1896, 516),
         (0, 0, 1920, 1080), (0, 0, 1872, 1080), (1648, 516)),
        ("auto-hide bottom, clear of the slid-in taskbar", (1500, 1032, 32, 48), (1516, 1056),
         (0, 0, 1920, 1080), (0, 0, 1920, 1080), (1516, 884)),
        ("invalid icon rect, the cursor anchors", None, (500, 1070),
         (0, 0, 1920, 1080), (0, 0, 1920, 1032), (500, 884)),
        ("keyboard Context, the cursor away from the icon: at the icon", (1700, 1392, 32, 48),
         (300, 500), (0, 0, 2560, 1440), (0, 0, 2560, 1392), (1715, 1244)),
    ],
)
def test_menu_origin(case, icon, cursor, screen, work, origin):
    """Right click: the short menu (224 x 148) stands on the taskbar's line
    of the icon's screen (above a bottom taskbar, as screenshots 13 and 14),
    at the cursor along the taskbar (at the icon when the cursor is off it),
    inside the work area. Seeded defects: the menu's bottom on the click
    point (over the taskbar), no clamp (the clamped rows run off the screen),
    a side taskbar treated as the bottom, auto-hide measured from the screen
    edge (the menu lands on the taskbar), or the cursor taken when it is off
    the icon (the keyboard row opens at x 300) -> red."""
    assert menu_origin((224, 148), icon, cursor, screen, work) == origin, case


def test_trigger_debounce_swallows_one_trigger_after_a_hide_over_the_icon():
    """Decision 9 and PlatformProbe Q2: a hide with the cursor over the icon
    swallows the next Trigger once, for at most 1 s; a hide elsewhere
    swallows nothing; a DoubleClick clears a pending swallow.
    Seeded defects: never expire, swallow more than one Trigger, arm on
    every hide, or keep the swallow through a DoubleClick -> red."""
    clock = [100.0]
    debounce = TriggerDebounce(now=lambda: clock[0])
    assert debounce.accept_trigger() is True

    debounce.flyout_hidden(cursor_in_icon=True)
    clock[0] += 0.25
    assert debounce.accept_trigger() is False   # the click that hid it
    assert debounce.accept_trigger() is True    # the next click opens it

    debounce.flyout_hidden(cursor_in_icon=True)
    clock[0] += 1.0
    assert debounce.accept_trigger() is True    # expired

    debounce.flyout_hidden(cursor_in_icon=False)
    assert debounce.accept_trigger() is True    # hidden by Esc or an outside click

    debounce.flyout_hidden(cursor_in_icon=True)
    debounce.double_click()
    assert debounce.accept_trigger() is True
