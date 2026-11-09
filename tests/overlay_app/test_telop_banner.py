"""telop_banner: vertical banner text and painting."""
import pytest

from yamabiko.overlay_app.telop_banner import split_vertical_runs


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_split_vertical_runs_keeps_cjk_upright_and_latin_sideways():
    """Kana, kanji and fullwidth marks are one upright cell each; Latin runs
    stay together to be set sideways (spec: upright kana and kanji, Latin
    sideways).

    Seeded defect: treat every character as upright -> "SPY" splits -> red."""
    assert split_vertical_runs("ニュース") == [
        ("ニ", True), ("ュ", True), ("ー", True), ("ス", True)]
    assert split_vertical_runs("SPY×FAMILY") == [("SPY×FAMILY", False)]
    assert split_vertical_runs("Dr.STONE 新世界") == [
        ("Dr.STONE ", False), ("新", True), ("世", True), ("界", True)]


def test_long_title_and_english_grow_the_banner(qapp):
    """The banner's natural height grows with its content: a long series
    title is taller than a genre, the English part and the pause glyph add
    length.

    Seeded defect: return a fixed height -> red."""
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.telop_banner import banner_content

    register_bundled_fonts()
    genre = banner_content("バラエティ", "Variety", False, False).natural_height
    title = banner_content("葬送のフリーレン", "Frieren", False, False).natural_height
    assert title > genre
    assert banner_content("バラエティ", "Variety", False, True).natural_height > genre
    assert banner_content("一時停止", "Paused", True, False).natural_height > \
        banner_content("一時停止", "Paused", False, False).natural_height


def test_banner_paints_tint_and_pause_glyph(qapp):
    """paint_banner fills its rect with the tint and draws white content (the
    pause glyph bars when paused), all inside the rect: mid-reshape the rect
    can be shorter than the content, which must not spill out of it.

    Seeded defects: skip the fillRect -> no tint pixels; drop the clip ->
    pause glyph above the short rect -> red."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    from yamabiko.overlay_app.telop_banner import banner_content, paint_banner

    image = QImage(38, 160, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    content = banner_content("一時停止", "Paused", True, False)
    painter = QPainter(image)
    paint_banner(painter, QRectF(0, 0, 38, 160), content, QColor("#5b5f66"))
    painter.end()
    corner = image.pixelColor(1, 1)
    assert (corner.red(), corner.green(), corner.blue()) == (0x5B, 0x5F, 0x66)
    top = (160 - content.natural_height) / 2 + 12  # BANNER_PAD
    bar = image.pixelColor(19 - 4, int(top) + 7)   # inside the left pause bar
    assert (bar.red(), bar.green(), bar.blue()) == (255, 255, 255)
    short = QImage(38, 160, QImage.Format.Format_ARGB32_Premultiplied)
    short.fill(Qt.GlobalColor.transparent)
    painter = QPainter(short)
    paint_banner(painter, QRectF(0, 60, 38, 40), content, QColor("#5b5f66"))
    painter.end()
    assert not any(short.pixelColor(x, y).alpha()
                   for y in (*range(60), *range(100, 160)) for x in range(38))


def test_banner_content_takes_the_cell_size(qapp):
    """Setup's tone sample sets the banner at 13 px cells.

    Seeded defect: ignore px -> 18 px cells -> red."""
    from yamabiko.overlay_app.telop_banner import banner_content

    content = banner_content("ニュース", "News", False, False, px=13)
    assert content.upright_font.pixelSize() == 13
    assert [length for _text, _up, length in content.runs] == [13.0] * 4
