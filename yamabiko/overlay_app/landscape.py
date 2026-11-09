"""The flat layered landscape: four ridgelines, a low mist and three echo arcs.

Drawn from a theme palette (ground, ridge1-4, echo1-3) in the 1000 x 80
design box of the Settings landscape band (opt-mountain.tpl), stretched to
the target rect the way the mockup's preserveAspectRatio="none" does. The arc
strokes stay 1.25 logical px whatever the stretch or display scale. Static:
nothing animates.

``paint_ridgeline`` is the tray flyout header's low ridgeline, in its own
360 x 28 design box (opt-tray-flyout.tpl). ``paint_strip`` is the startup
error window's slim landscape strip behind its caption, in its own 720 x 44
design box (opt-dlg-report.tpl). ``paint_vignette`` is the setup rail's
cropped landscape at its bottom, in its own 220 x 112 design box
(opt-wiz-rail.tpl), with no sky: the rail shows through.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import QWidget

DESIGN_W = 1000.0
DESIGN_H = 80.0

# Outlines in design units: a start point, then segments. Six numbers are a
# cubic (c1x, c1y, c2x, c2y, x, y); two numbers are a straight line to (x, y).
_RIDGE1 = ((0, 65), (
    (100, 64, 200, 63, 300, 61), (380, 59, 440, 54, 488, 47),
    (516, 43, 538, 38, 552, 36), (568, 36),
    (582, 38, 604, 43, 632, 47), (680, 54, 730, 59, 790, 61),
    (870, 63, 940, 64, 1000, 64), (1000, 80), (0, 80),
))
_RIDGE2 = ((0, 70), (
    (80, 69, 180, 68, 260, 66), (320, 64, 368, 53, 418, 50),
    (444, 49, 466, 52, 492, 57), (522, 63, 560, 66, 610, 66),
    (660, 66, 700, 59, 732, 58), (762, 57, 792, 62, 832, 66),
    (890, 69, 950, 69, 1000, 68), (1000, 80), (0, 80),
))
_RIDGE3 = ((0, 74), (
    (100, 73, 200, 72, 300, 73), (380, 74, 450, 73, 520, 71),
    (580, 69, 628, 61, 668, 57), (688, 55, 704, 56, 720, 59),
    (750, 64, 790, 70, 850, 71), (920, 72, 970, 72, 1000, 72),
    (1000, 80), (0, 80),
))
_RIDGE4_WEST = ((250, 80), (
    (290, 76, 330, 70, 370, 67), (395, 65.5, 415, 66, 435, 68),
    (465, 71, 495, 76, 520, 80),
))
_RIDGE4_EAST = ((880, 80), (
    (905, 75, 935, 70, 960, 69), (978, 68.5, 992, 69.5, 1000, 70.5),
    (1000, 80),
))
_MIST = ((0, 77), (
    (120, 76, 220, 77.5, 340, 77), (460, 76.5, 540, 78, 660, 77.5),
    (780, 77, 880, 78, 1000, 77), (1000, 80), (0, 80),
))

# Echo arcs over the central peak, outermost first: (token, chord x0, chord
# x1, chord y, rx, ry). Each is the upper arc of an ellipse through both
# chord ends (the mockup's SVG "A rx ry 0 0 1" arcs).
_ARCS = (
    ("echo3", 489.7, 630.3, 30.0, 72.0, 28.0),
    ("echo2", 513.2, 606.8, 32.0, 48.0, 18.0),
    ("echo1", 535.0, 585.0, 33.5, 26.0, 9.0),
)
_ARC_PEN_PX = 1.25


def _outline(spec) -> QPainterPath:
    (x0, y0), segments = spec
    path = QPainterPath()
    path.moveTo(x0, y0)
    for seg in segments:
        if len(seg) == 6:
            path.cubicTo(*seg)
        else:
            path.lineTo(*seg)
    path.closeSubpath()
    return path


def _upper_arc(x0: float, x1: float, y: float, rx: float, ry: float) -> QPainterPath:
    half = (x1 - x0) / 2.0
    end = math.degrees(math.acos(half / rx))  # chord ends, degrees above the axis
    cx = x0 + half
    cy = y + ry * math.sin(math.radians(end))
    rect = QRectF(cx - rx, cy - ry, 2.0 * rx, 2.0 * ry)
    path = QPainterPath()
    path.arcMoveTo(rect, 180.0 - end)
    path.arcTo(rect, 180.0 - end, -(180.0 - 2.0 * end))
    return path


_LAYERS = tuple(
    (token, _outline(spec))
    for token, spec in (
        ("ridge1", _RIDGE1),
        ("ridge2", _RIDGE2),
        ("ridge3", _RIDGE3),
        ("ridge4", _RIDGE4_WEST),
        ("ridge4", _RIDGE4_EAST),
        ("ground", _MIST),
    )
)
_ARC_PATHS = tuple((token, _upper_arc(*geom)) for token, *geom in _ARCS)

# The tray flyout header's low ridgeline (opt-tray-flyout.tpl .ridge) in its
# own 360 x 28 design box: four ridges, then a mist in the raised tone the
# panel body continues from.
RIDGELINE_W = 360.0
RIDGELINE_H = 28.0
_RIDGELINE = tuple(
    (token, _outline(spec))
    for token, spec in (
        ("ridge1", ((0, 20), (
            (40, 19, 80, 17, 120, 14.5), (150, 12.5, 176, 7.5, 196, 4.5),
            (206, 3, 216, 2.5, 226, 2.6), (240, 2.8, 254, 5.5, 270, 8.5),
            (296, 13, 328, 16.5, 360, 17.5), (360, 28), (0, 28),
        ))),
        ("ridge2", ((0, 23), (
            (28, 22, 54, 19.5, 76, 15), (86, 13, 94, 12, 102, 12.4),
            (114, 13, 128, 17.5, 148, 20), (188, 23.5, 228, 22.5, 268, 18.5),
            (298, 15.5, 330, 14.5, 360, 15.5), (360, 28), (0, 28),
        ))),
        ("ridge3", ((0, 26), (
            (60, 25.5, 120, 25, 170, 25.5), (210, 26, 238, 23, 260, 19.5),
            (270, 18, 278, 18.5, 286, 20), (302, 23, 330, 25, 360, 25),
            (360, 28), (0, 28),
        ))),
        ("ridge4", ((232, 28), (
            (242, 26.2, 252, 24.4, 262, 23.8), (270, 23.4, 278, 24.2, 286, 25.6),
            (292, 26.8, 296, 27.6, 300, 28),
        ))),
        ("raised", ((0, 25.4), (
            (90, 25, 180, 26.2, 270, 25.7), (310, 25.5, 340, 25.9, 360, 25.7),
            (360, 28), (0, 28),
        ))),
    )
)


def _paint_scene(
    painter: QPainter, rect: QRectF, palette: dict, design_w: float, design_h: float,
    layers, arcs, *, sky: str | None = "ground",
) -> None:
    """Paint the ``sky`` token across ``rect`` (nothing when None), then the
    filled ``layers`` and the ``arcs`` strokes ((token, path) pairs in a
    ``design_w`` x ``design_h`` design box), stretched to ``rect`` the way the
    mockups' preserveAspectRatio="none" does."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    if sky is not None:
        painter.fillRect(rect, QColor(palette[sky]))
    # Map the design-unit paths into ``rect`` instead of scaling the painter,
    # so the arc pen stays 1.25 px in the caller's (logical) units whatever
    # the stretch, the display scale or the paint target.
    stretch = QTransform(
        rect.width() / design_w, 0, 0, rect.height() / design_h, rect.x(), rect.y()
    )
    painter.setPen(Qt.NoPen)
    for token, path in layers:
        painter.setBrush(QColor(palette[token]))
        painter.drawPath(stretch.map(path))
    painter.setBrush(Qt.NoBrush)
    for token, path in arcs:
        pen = QPen(QColor(palette[token]), _ARC_PEN_PX)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPath(stretch.map(path))
    painter.restore()


def paint_landscape(
    painter: QPainter, rect: QRectF, palette: dict, *, arcs: bool = True
) -> None:
    """Paint sky, ridges, mist and (optionally) echo arcs into ``rect``."""
    _paint_scene(painter, rect, palette, DESIGN_W, DESIGN_H, _LAYERS,
                 _ARC_PATHS if arcs else ())


def paint_ridgeline(painter: QPainter, rect: QRectF, palette: dict) -> None:
    """Paint the tray flyout header's low ridgeline into ``rect`` (no sky:
    the header's ground shows above it), stretched like the mockup's
    preserveAspectRatio="none"."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    stretch = QTransform(
        rect.width() / RIDGELINE_W, 0, 0, rect.height() / RIDGELINE_H, rect.x(), rect.y()
    )
    painter.setPen(Qt.NoPen)
    for token, path in _RIDGELINE:
        painter.setBrush(QColor(palette[token]))
        painter.drawPath(stretch.map(path))
    painter.restore()


# The startup error window's 44 px strip (opt-dlg-report.tpl .e-mast) in its
# own 720 x 44 design box: four ridges and a mist low in the strip, and three
# echo arcs (full half-ellipses: chord = 2 rx) centered at x 411 that rise
# into the caption row. Stretching the Settings band into 44 px would put
# the ridges at the wrong heights (its ridge1 starts at 81 % of the height,
# the strip's at 89 %).
STRIP_W = 720.0
STRIP_H = 44.0
_STRIP_LAYERS = tuple(
    (token, _outline(spec))
    for token, spec in (
        ("ridge1", ((0, 39), (
            (80, 38.6, 160, 38.2, 240, 37.4), (298, 36.6, 340, 34, 372, 31.4),
            (389, 30, 400, 28.8, 405, 28.6), (417, 28.6), (424, 29, 436, 30.2, 452, 31.8),
            (492, 35.4, 542, 37.2, 602, 38), (652, 38.5, 692, 38.8, 720, 38.8), (720, 44), (0, 44),
        ))),
        ("ridge2", ((0, 40.6), (
            (60, 40.3, 130, 40, 190, 39.6), (230, 39.3, 262, 35.8, 296, 34.6),
            (314, 34, 330, 35, 348, 36.6), (372, 38.6, 402, 39.6, 442, 39.6),
            (482, 39.6, 510, 37, 534, 36.6), (556, 36.2, 578, 38.2, 606, 39.6),
            (650, 41, 690, 41, 720, 40.8), (720, 44), (0, 44),
        ))),
        ("ridge3", ((0, 42.4), (
            (80, 42.1, 160, 41.9, 240, 42.1), (300, 42.3, 360, 42.1, 410, 41.6),
            (450, 41.2, 482, 38.6, 512, 37.4), (526, 36.9, 538, 37.1, 550, 38.3),
            (572, 40.2, 600, 42, 640, 42.3), (680, 42.6, 700, 42.6, 720, 42.5), (720, 44), (0, 44),
        ))),
        ("ridge4", ((160, 44), (
            (188, 42.6, 216, 40.9, 246, 40.2), (264, 39.8, 278, 40.1, 292, 40.9),
            (312, 42, 332, 43.2, 352, 44),
        ))),
        ("ridge4", ((632, 44), (
            (650, 42.5, 670, 41.3, 692, 41), (705, 40.9, 714, 41.1, 720, 41.4), (720, 44),
        ))),
        ("ground", ((0, 43.3), (
            (100, 43, 200, 43.5, 300, 43.3), (400, 43.1, 500, 43.6, 600, 43.4),
            (660, 43.3, 700, 43.5, 720, 43.3), (720, 44), (0, 44),
        ))),
    )
)
_STRIP_ARCS = tuple(
    (token, _upper_arc(*geom))
    for token, *geom in (
        ("echo3", 367.0, 455.0, 25.0, 44.0, 15.0),
        ("echo2", 382.0, 440.0, 26.5, 29.0, 9.5),
        ("echo1", 395.0, 427.0, 27.6, 16.0, 5.0),
    )
)


def paint_strip(painter: QPainter, rect: QRectF, palette: dict) -> None:
    """Paint the startup error window's landscape strip into ``rect``: the
    ground, then the ridges and mist, then the echo arcs (1.25 px strokes),
    stretched like the mockup's preserveAspectRatio="none"."""
    _paint_scene(painter, rect, palette, STRIP_W, STRIP_H, _STRIP_LAYERS, _STRIP_ARCS)


# The setup wizard rail's vignette (opt-wiz-rail.tpl svg.vig) in its own
# 220 x 112 design box: three echo arcs over the flat-topped peak, four
# ridges and a mist in the rail's raised tone. No sky: the rail's raised fill
# shows above the ridges.
VIGNETTE_W = 220.0
VIGNETTE_H = 112.0
_VIGNETTE_LAYERS = tuple(
    (token, _outline(spec))
    for token, spec in (
        ("ridge1", ((0, 94), (
            (28, 92, 56, 86, 80, 77), (93, 72, 104, 65, 111, 60), (129, 60),
            (136, 65, 147, 72, 160, 77), (182, 85, 204, 90, 220, 91), (220, 112), (0, 112),
        ))),
        ("ridge2", ((0, 98), (
            (22, 97, 42, 90, 60, 86), (74, 83, 86, 86, 99, 90), (116, 95, 134, 97, 154, 95),
            (172, 93, 186, 88, 198, 88), (208, 88, 214, 90, 220, 91), (220, 112), (0, 112),
        ))),
        ("ridge3", ((0, 103), (
            (30, 102, 70, 102, 100, 101), (128, 100, 148, 94, 166, 91),
            (177, 89, 187, 91, 196, 94), (206, 98, 214, 101, 220, 101), (220, 112), (0, 112),
        ))),
        ("ridge4", ((26, 112), (
            (42, 108, 57, 103, 71, 101), (83, 100, 93, 102, 103, 105), (113, 108, 121, 110, 128, 112),
        ))),
        ("ridge4", ((188, 112), ((199, 108, 211, 105, 220, 104.5), (220, 112)))),
        ("raised", ((0, 108.5), (
            (40, 108, 80, 109.5, 120, 109), (160, 108.5, 190, 109.5, 220, 108.5), (220, 112), (0, 112),
        ))),
    )
)
_VIGNETTE_ARCS = tuple(
    (token, _upper_arc(*geom))
    for token, *geom in (
        ("echo3", 65.5, 174.5, 54.0, 55.0, 27.0),
        ("echo2", 83.5, 156.5, 56.0, 37.0, 17.0),
        ("echo1", 100.0, 140.0, 57.5, 20.0, 8.0),
    )
)


def paint_vignette(painter: QPainter, rect: QRectF, palette: dict) -> None:
    """Paint the setup rail's vignette into ``rect``: the ridges and the mist
    (raised), then the echo arcs (1.25 px strokes), stretched like the
    mockup's preserveAspectRatio="none". No sky fill."""
    _paint_scene(painter, rect, palette, VIGNETTE_W, VIGNETTE_H, _VIGNETTE_LAYERS,
                 _VIGNETTE_ARCS, sky=None)


class LandscapeBand(QWidget):
    """Paints the landscape across its whole rect; children sit on the sky."""

    def __init__(
        self, palette: dict, parent: QWidget | None = None, *, arcs: bool = True
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)  # paints every pixel of its rect
        self._palette = palette
        self._arcs = arcs

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        # The design box; layouts stretch the width.
        return QSize(int(DESIGN_W), int(DESIGN_H))

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        paint_landscape(painter, QRectF(self.rect()), self._palette, arcs=self._arcs)
        painter.end()
