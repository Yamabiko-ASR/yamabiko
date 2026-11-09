"""telop_text: line measurement and block rendering."""
import pytest

from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from yamabiko.overlay_app.telop_look import IMAGE_MARGIN, look_from_style

_LONG_EN = ("Ramen at a stall just feels different from a restaurant, so you "
            "end up eating bowl after bowl.")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pixels(image):
    for y in range(image.height()):
        for x in range(image.width()):
            yield image.pixelColor(x, y)


def test_english_block_wraps_at_cap_and_short_line_hugs_text(qapp):
    """A long English line wraps inside the box cap (rows > 1, width <= cap,
    height = rows x row height + padding); a short line stays one row and
    narrower than the cap; whitespace-only English is no English (no empty
    dark bar).

    Seeded defects: skip setLineWidth (no wrap), size the box from the cap
    instead of the text, or test the unstripped text -> red."""
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.telop_text import layout_line

    register_bundled_fonts()
    look = look_from_style({**STYLE_DEFAULTS, "en_font": "BIZ UDPGothic"})
    cap = 420
    long_block = layout_line([], _LONG_EN, look, 0, cap).blocks[0]
    assert long_block.kind == "en" and len(long_block.rows) > 1
    assert long_block.width <= cap
    pad = long_block.pad
    assert long_block.height == pad.top + long_block.row_h * len(long_block.rows) + pad.bottom
    short_block = layout_line([], "Wait.", look, 0, cap).blocks[0]
    assert len(short_block.rows) == 1 and short_block.width < cap
    assert layout_line([], " \t ", look, 0, cap).blocks == ()


def test_echo_block_is_outlined_white_and_plate_block_is_ink(qapp):
    """Age 0 paints Japanese in ink (no white); age 1 paints white fill with
    a black outline around it.

    Seeded defect: render every age in the plate look -> no white pixels
    at age 1 -> red."""
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.telop_text import layout_line, render_block

    register_bundled_fonts()
    look = look_from_style({**STYLE_DEFAULTS, "echo": "both"})
    segs = [{"text": "東京", "reading": "とうきょう"}, {"text": "です"}]

    plate = render_block(layout_line(segs, "", look, 0, 600).blocks[0], look, 1.0)
    colors = list(_pixels(plate))
    assert any(c.alpha() == 255 and abs(c.red() - 0x15) < 6 and abs(c.blue() - 0x1B) < 6 for c in colors)
    assert not any(c.alpha() == 255 and c.red() > 240 and c.green() > 240 for c in colors)

    echo = render_block(layout_line(segs, "", look, 1, 600).blocks[0], look, 1.0)
    colors = list(_pixels(echo))
    assert any(c.alpha() == 255 and c.red() > 240 and c.green() > 240 and c.blue() > 240 for c in colors)
    assert any(c.alpha() > 200 and c.red() < 10 and c.green() < 10 and c.blue() < 10 for c in colors)
    assert plate.width() == layout_line(segs, "", look, 0, 600).blocks[0].width + 2 * IMAGE_MARGIN


def test_learn_furigana_vermilion_on_plate_amber_in_echo(qapp):
    """Learn furigana: vermilion #c23b2e on the paper strip, amber #f0bf5a in
    the echo.

    Rendered at dpr 2: at dpr 1 the 15 px / 13 px Klee One furigana strokes
    are thinner than a pixel, so no pixel is fully covered (and the echo's
    amber always blends with the outline ring).

    Seeded defect: use kana_plate for the echo -> no amber -> red."""
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.telop_text import layout_line, render_block

    register_bundled_fonts()
    look = look_from_style({**STYLE_DEFAULTS, "mode": "learn"})
    segs = [{"text": "町", "reading": "まち"}]

    def has(image, rgb):
        return any(c.alpha() == 255 and all(abs(a - b) < 8 for a, b in zip((c.red(), c.green(), c.blue()), rgb))
                   for c in _pixels(image))

    assert has(render_block(layout_line(segs, "", look, 0, 600).blocks[0], look, 2.0), (0xC2, 0x3B, 0x2E))
    assert has(render_block(layout_line(segs, "", look, 1, 600).blocks[0], look, 2.0), (0xF0, 0xBF, 0x5A))
