"""The hanko seal paints a vermilion tile carrying white characters."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtWidgets import QWidget

from yamabiko.overlay_app.seal import Seal


def test_seal_paints_vermilion_tile_with_white_characters(qtbot):
    # (5, 13) is tile between the inner rule and the glyphs; the box x 8-18,
    # y 3-22 holds 山 over 彦. Seeded defects: fill the tile in another colour,
    # or skip/recolour the characters (no near-white pixels in the box) -> red.
    seal = Seal(26)
    qtbot.addWidget(seal)
    img = QImage(26, 26, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    seal.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)

    assert img.pixelColor(5, 13).name() == "#c8402c"
    white = sum(
        1
        for x in range(8, 19)
        for y in range(3, 23)
        if img.pixelColor(x, y).lightness() > 230
    )
    assert white >= 12, f"only {white} near-white glyph pixels"
