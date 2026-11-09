"""tray_icons: the tray icon states, drawn on each size's own pixel grid
(plan Decisions 14, 23, 44, 46)."""
from __future__ import annotations

import ctypes
import re
from ctypes import wintypes

import pytest
from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath

from yamabiko.overlay_app import tray_icons
from yamabiko.overlay_app.tray_icons import HIDDEN, LISTENING, PAUSED

# The 16 px icon row of the approved mockup (opt-tray-live.tpl), as drawn
# there: absolute M/H/V/Z paths, even-odd fill.
MOCKUP_16 = {
    "glyph": "M3 7H5V11H7V3H9V11H11V7H13V13H3Z",
    "bars": "M5 4H7V12H5ZM9 4H11V12H9Z",
    "outline": "M2 7H5V10H6V3H10V10H11V7H14V13H2ZM3 8V12H13V8H12V11H9V4H7V11H4V8Z",
    "ring": "M12 10H15V11H16V12H17V15H16V16H15V17H12V16H11V15H10V12H11V11H12Z",
    "dot": "M12 11H15V12H16V15H15V16H12V15H11V12H12Z",
}

# Decisions 23 and 44: the grids for the other sizes, (x, y, w, h) rects.
DECISION_23 = {
    20: {
        "glyph": ((4, 9, 2, 7), (9, 4, 2, 12), (14, 9, 2, 7), (4, 14, 12, 2)),
        "bars": ((7, 5, 2, 10), (11, 5, 2, 10)),
        "outline": ((3, 8, 4, 9), (8, 3, 4, 14), (13, 8, 4, 9), (3, 13, 14, 4)),
        "hole": ((4, 9, 2, 7), (9, 4, 2, 12), (14, 9, 2, 7), (4, 14, 12, 2)),
        "dot": ((15, 14, 4, 6), (14, 15, 6, 4)),
        "ring": ((15, 13, 4, 8), (14, 14, 6, 6), (13, 15, 8, 4)),
    },
    24: {
        "glyph": ((4, 10, 3, 10), (10, 4, 4, 16), (17, 10, 3, 10), (4, 17, 16, 3)),
        "bars": ((7, 6, 3, 12), (14, 6, 3, 12)),
        "outline": ((3, 9, 5, 12), (9, 3, 6, 18), (16, 9, 5, 12), (3, 16, 18, 5)),
        "hole": ((4, 10, 3, 10), (10, 4, 4, 16), (17, 10, 3, 10), (4, 17, 16, 3)),
        "dot": ((18, 16, 4, 8), (17, 17, 6, 6), (16, 18, 8, 4)),
        "ring": ((18, 15, 4, 10), (17, 16, 6, 8), (16, 17, 8, 6), (15, 18, 10, 4)),
    },
    28: {
        "glyph": ((5, 12, 3, 11), (12, 5, 4, 18), (20, 12, 3, 11), (5, 20, 18, 3)),
        "bars": ((9, 7, 3, 14), (16, 7, 3, 14)),
        "outline": ((4, 11, 5, 13), (11, 4, 6, 20), (19, 11, 5, 13), (4, 19, 20, 5)),
        "hole": ((6, 13, 1, 9), (13, 6, 2, 16), (21, 13, 1, 9), (6, 21, 16, 1)),
        "dot": ((21, 19, 5, 9), (20, 20, 7, 7), (19, 21, 9, 5)),
        "ring": ((21, 18, 5, 11), (20, 19, 7, 9), (19, 20, 9, 7), (18, 21, 11, 5)),
    },
    32: {
        "glyph": ((6, 14, 4, 12), (14, 6, 4, 20), (22, 14, 4, 12), (6, 22, 20, 4)),
        "bars": ((10, 8, 4, 16), (18, 8, 4, 16)),
        "outline": ((4, 14, 6, 12), (12, 6, 8, 20), (22, 14, 6, 12), (4, 20, 24, 6)),
        "hole": ((6, 16, 2, 8), (14, 8, 4, 14), (24, 16, 2, 8), (6, 22, 20, 2)),
        "dot": ((24, 22, 6, 10), (22, 24, 10, 6)),
        "ring": ((24, 21, 6, 12), (23, 22, 8, 10), (22, 23, 10, 8), (21, 24, 12, 6)),
    },
}


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _rgba(image: QImage, x: int, y: int) -> tuple:
    c = image.pixelColor(x, y)
    return (c.red(), c.green(), c.blue(), c.alpha())


def _pixels(image: QImage, color: str) -> set:
    """Pixels exactly ``color``, fully opaque."""
    want = QColor(color)
    rgba = (want.red(), want.green(), want.blue(), 255)
    return {(x, y) for y in range(image.height()) for x in range(image.width())
            if _rgba(image, x, y) == rgba}


def _path_pixels(d: str, size: int) -> set:
    """The pixels an even-odd, non-antialiased fill of the mockup path
    ``d`` covers on a ``size`` canvas."""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    x = y = 0.0
    for cmd, args in re.findall(r"([MHVZ])([^MHVZ]*)", d):
        nums = [float(v) for v in args.split()]
        if cmd == "M":
            x, y = nums
            path.moveTo(x, y)
        elif cmd == "H":
            x = nums[0]
            path.lineTo(x, y)
        elif cmd == "V":
            y = nums[0]
            path.lineTo(x, y)
        else:
            path.closeSubpath()
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.fillPath(path, QColor("#ffffff"))
    painter.end()
    return _pixels(image, "#ffffff")


def _cells(rects, size: int) -> set:
    return {(x, y) for rx, ry, rw, rh in rects
            for x in range(rx, rx + rw) for y in range(ry, ry + rh)
            if 0 <= x < size and 0 <= y < size}


@pytest.mark.parametrize("taskbar", ["light", "dark"])
def test_16px_icon_states_match_the_mockup_paths(qapp, taskbar):
    """Decision 23: the 16 px grid is the mockup's, pixel for pixel: the
    white 山 on the vermilion tile (Listening), two white bars on the grey
    tile (Paused), the 山 outline (Subtitles hidden), and the amber dot with
    its 1 px ring in the taskbar color over any state.

    Seeded defects: antialias the glyph rects, shift a rect by a pixel,
    draw the hidden 山 filled, or paint the ring in a fixed color -> red."""
    ring_color = tray_icons.TASKBAR_COLORS[taskbar]
    glyph = _path_pixels(MOCKUP_16["glyph"], 16)
    outline = _path_pixels(MOCKUP_16["outline"], 16)
    listening = tray_icons.paint_icon(16, LISTENING, False, taskbar)
    assert (listening.width(), listening.height()) == (16, 16)
    assert _pixels(listening, "#ffffff") == glyph
    assert _rgba(listening, 8, 0) == (0xD2, 0x45, 0x2F, 255)
    assert listening.pixelColor(0, 0).alpha() < 255          # the tile's rounded corner
    paused = tray_icons.paint_icon(16, PAUSED, False, taskbar)
    assert _pixels(paused, "#ffffff") == _path_pixels(MOCKUP_16["bars"], 16)
    assert _rgba(paused, 8, 0) == (0x8A, 0x8F, 0x8A, 255)
    hidden = tray_icons.paint_icon(16, HIDDEN, False, taskbar)
    assert _pixels(hidden, "#ffffff") == outline
    assert _rgba(hidden, 7, 5) == (0xD2, 0x45, 0x2F, 255)    # inside the outline
    dot = _path_pixels(MOCKUP_16["dot"], 16)
    ring = _path_pixels(MOCKUP_16["ring"], 16) - dot
    for state, white in ((LISTENING, glyph), (HIDDEN, outline)):
        marked = tray_icons.paint_icon(16, state, True, taskbar)
        assert _pixels(marked, "#e0a030") == dot
        assert _pixels(marked, ring_color) == ring
        assert _pixels(marked, "#ffffff") == white - ring - dot


@pytest.mark.parametrize("size", [20, 24, 28, 32])
def test_each_larger_size_draws_its_own_grid(qapp, size):
    """Decisions 14, 23 and 44: 20, 24, 28 and 32 px are drawn on their own integer
    grids (Qt gets only that pixmap, so Windows never resamples it): every
    white and amber pixel is exactly where the grid puts it.

    Seeded defects: scale the 16 px drawing up, antialias the rects, or
    fill the hidden 山's hole in white -> red."""
    grid = DECISION_23[size]
    dot = _cells(grid["dot"], size)
    expected = {
        LISTENING: _cells(grid["glyph"], size),
        PAUSED: _cells(grid["bars"], size),
        HIDDEN: _cells(grid["outline"], size) - _cells(grid["hole"], size),
    }
    for state, white in expected.items():
        plain = tray_icons.paint_icon(size, state, False, "dark")
        assert (plain.width(), plain.height()) == (size, size)
        assert _pixels(plain, "#ffffff") == white
        marked = tray_icons.paint_icon(size, state, True, "light")
        assert _pixels(marked, "#e0a030") == dot
        assert _pixels(marked, "#f3f3f3") == _cells(grid["ring"], size) - dot


def test_paused_wins_over_hidden_in_the_icon_state():
    """Decision 14: paused wins over hidden. Seeded defect: check the
    overlay's visibility first -> red."""
    assert tray_icons.icon_state(paused=True, subtitles_shown=False) == PAUSED
    assert tray_icons.icon_state(paused=True, subtitles_shown=True) == PAUSED
    assert tray_icons.icon_state(paused=False, subtitles_shown=False) == HIDDEN
    assert tray_icons.icon_state(paused=False, subtitles_shown=True) == LISTENING


def test_tray_icon_size_follows_the_screen_dpi_snapped_to_a_drawn_grid(qapp, monkeypatch):
    """Decision 44: the metric for the tray screen's DPI (96 x its device
    pixel ratio) snaps down to a drawn size, and the QIcon holds only that
    one pixmap.

    Seeded defects: ask for 96 dpi on every screen, round 28 up to 32, or
    add more sizes to the QIcon -> red."""
    metric = {72: 12, 96: 16, 120: 20, 144: 24, 168: 28, 192: 32, 240: 40}
    asked: list[int] = []
    monkeypatch.setattr(tray_icons, "small_icon_metric",
                        lambda dpi: asked.append(dpi) or metric[dpi])

    class _Screen:
        def __init__(self, dpr: float) -> None:
            self._dpr = dpr

        def devicePixelRatio(self) -> float:  # noqa: N802 (QScreen API)
            return self._dpr

    sizes = {dpr: tray_icons.tray_icon_px(_Screen(dpr))
             for dpr in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)}
    assert sizes == {0.75: 16, 1.0: 16, 1.25: 20, 1.5: 24, 1.75: 28, 2.0: 32, 2.5: 32}
    assert asked == [72, 96, 120, 144, 168, 192, 240]
    icon = tray_icons.tray_icon(_Screen(1.5), PAUSED, True, "dark")
    assert icon.availableSizes() == [QSize(24, 24)]


def test_taskbar_watcher_rereads_only_on_immersive_color_set(qapp):
    """Decision 46: WM_SETTINGCHANGE with the "ImmersiveColorSet" area
    re-reads the taskbar theme; other areas do not; one change emits once
    although Windows delivers a copy to every top-level window.

    Seeded defects: re-read on any WM_SETTINGCHANGE, or emit on every
    re-read -> red."""
    values = iter(["dark", "light", "light"])
    reads: list[str] = []

    def read() -> str:
        reads.append("read")
        return next(values)

    watcher = tray_icons.TaskbarThemeWatcher(read=read)
    seen: list[str] = []
    watcher.changed.connect(seen.append)
    try:
        def send(area: str):
            text = ctypes.create_unicode_buffer(area)
            msg = wintypes.MSG()
            msg.message = 0x001A   # WM_SETTINGCHANGE
            msg.lParam = ctypes.addressof(text)
            return watcher._filter.nativeEventFilter(
                QByteArray(b"windows_generic_MSG"), ctypes.addressof(msg))

        assert send("Policy") == (False, 0)
        assert len(reads) == 1                       # the initial read only
        send("ImmersiveColorSet")
        send("ImmersiveColorSet")                    # another window's copy
        assert seen == ["light"] and watcher.theme() == "light" and len(reads) == 3
    finally:
        watcher.uninstall()
