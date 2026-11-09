"""LandscapeBand paints the landscape band from a palette."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtWidgets import QWidget

from yamabiko.overlay_app.landscape import LandscapeBand
from yamabiko.overlay_app.theme import MIST, NIGHT


def _render(widget, w: int, h: int) -> QImage:
    widget.resize(w, h)
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    # DrawChildren only: no window background, so unpainted pixels stay clear.
    widget.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return img


@pytest.mark.parametrize("palette", [MIST, NIGHT], ids=["mist", "night"])
def test_landscape_band_paints_sky_ridges_and_arcs_from_palette(qtbot, palette):
    # At 1000 x 80 one pixel is one design unit. Top-left is open sky
    # (ground); design (385, 72) sits inside the western near ridge (ridge4);
    # design (560, 8) is on the outer echo arc. Seeded defects: hardcode a
    # ridge colour (ridge pixel wrong under one palette), skip the near ridge
    # (pixel shows ridge2), or drop the arcs (arc pixel is plain sky) -> red.
    band = LandscapeBand(palette)
    qtbot.addWidget(band)
    img = _render(band, 1000, 80)
    assert img.pixelColor(10, 5).name() == palette["ground"]
    assert img.pixelColor(385, 72).name() == palette["ridge4"]
    assert img.pixelColor(560, 8).name() != palette["ground"]


@pytest.mark.parametrize("palette", [MIST, NIGHT], ids=["mist", "night"])
def test_strip_paints_its_own_ridge_heights_and_the_arcs_over_the_peak(palette):
    # The startup error window's 720 x 44 strip, one pixel per design unit.
    # At x 100 the far ridge (ridge1) covers y 39 and the second (ridge2)
    # y 41; the sky above is ground; the inner echo arc's top (x 411,
    # y 22.6) is stroked. Seeded defects: the Settings band stretched into
    # 44 px (x 100, y 41 would be ridge3), a hardcoded tone, or no arcs ->
    # red.
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPainter

    from yamabiko.overlay_app.landscape import STRIP_H, STRIP_W, paint_strip

    img = QImage(int(STRIP_W), int(STRIP_H), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    paint_strip(p, QRectF(0, 0, STRIP_W, STRIP_H), palette)
    p.end()
    assert img.pixelColor(100, 20).name() == palette["ground"]
    assert img.pixelColor(100, 39).name() == palette["ridge1"]
    assert img.pixelColor(100, 41).name() == palette["ridge2"]
    assert img.pixelColor(411, 22).name() != palette["ground"]


@pytest.mark.parametrize("palette", [MIST, NIGHT], ids=["mist", "night"])
def test_vignette_paints_ridges_mist_and_arcs_and_leaves_the_sky_alone(palette):
    # Setup's rail vignette, one pixel per design unit (220 x 112). The sky
    # keeps what was under it (the rail's raised fill shows through); x 120
    # is on the far ridge's flat top (y 60), so y 70 is ridge1; the mist
    # along the bottom is the raised tone; the outer echo arc tops out at
    # y 30.6 over x 120. Seeded defects: fill the sky with ground, paint the
    # mist in ground, or drop the arcs -> red.
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPainter

    from yamabiko.overlay_app.landscape import paint_vignette

    img = QImage(220, 112, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    paint_vignette(p, QRectF(0, 0, 220, 112), palette)
    p.end()
    assert img.pixelColor(10, 10).alpha() == 0
    assert img.pixelColor(120, 70).name() == palette["ridge1"]
    assert img.pixelColor(60, 111).name() == palette["raised"]
    assert img.pixelColor(120, 30).alpha() > 200
