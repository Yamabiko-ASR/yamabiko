"""Measure and paint the text of one subtitle line for the telop overlay.

A line has up to two blocks, Japanese (ruby rows) then English (wrapped
rows); which ones show depends on the line's age and the Echo setting
(TelopLook.shows_jp / shows_en). layout_line() measures the blocks for one
age; render_block() paints a block's text into a transparent QImage in one
of two looks:

* plate look (age 0): Japanese in ink for the paper strip, English in the
  text color for the dark bar. SubtitleStack paints the strip and the bar.
* echo look (age 1+): outlined text (eight 2 px black offsets) over a soft
  box-blurred shadow, like the mockup's CSS text-shadow.

Block geometry is the CSS box of the approved mockup: rows of
``round(px * line_height)`` with the em box centered in each row (half
leading), padding around the rows; furigana sit right above the base
glyphs' em box. Every image carries IMAGE_MARGIN of transparent space on
each side so the outline and shadow are never clipped.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QFontMetricsF, QImage, QPainter, QTextLayout, QTextOption,
)

from .ruby_layout import layout_rows
from .telop_look import (
    IMAGE_MARGIN, OUTLINE, OUTLINE_PX, SHADOW_ALPHA, SHADOW_BOX_PX, FontSpec,
    Pad, TelopLook, age_scale, round_half_up,
)

# Floor for a capped text column, about one CJK glyph at a large size: a
# tiny cap (big padding in a narrow preview) still wraps, never collapses.
MIN_TEXT_WIDTH = 40

_OUTLINE_OFFSETS = tuple(
    (dx, dy)
    for dx in (-OUTLINE_PX, 0, OUTLINE_PX)
    for dy in (-OUTLINE_PX, 0, OUTLINE_PX)
    if (dx, dy) != (0, 0)
)


def make_font(spec: FontSpec, scale: float = 1.0) -> QFont:
    font = QFont(spec.family)
    font.setPixelSize(max(1, round_half_up(spec.px * scale)))
    font.setWeight(QFont.Weight(spec.weight))
    # CJK blurriness workaround (QTBUG-58610), kept from the retired ruby row.
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return font


@dataclass(frozen=True)
class RubySeg:
    text: str
    reading: str
    text_w: int
    reading_w: int
    width: int          # max(text_w, reading_w)


@dataclass(frozen=True, eq=False)
class Block:
    """One language block of one line, measured for one age."""
    kind: str                       # "jp" | "en"
    age: int
    pad: Pad
    rows: tuple                     # jp: tuple[tuple[RubySeg, ...], ...]; en: tuple[str, ...]
    row_widths: tuple
    row_h: int
    font: QFont                     # base glyphs (Japanese) or English
    kana_font: QFont | None
    width: int                      # box width including side padding
    height: int
    centered: bool = False          # Center alignment: rows centered in the box


@dataclass(frozen=True, eq=False)
class LineLayout:
    age: int
    blocks: tuple                   # tuple[Block, ...]: jp first, then en

    @property
    def width(self) -> int:
        return max((b.width for b in self.blocks), default=0)

    @property
    def height(self) -> int:
        return sum(b.height for b in self.blocks)


def _ceil(value: float) -> int:
    return int(math.ceil(value - 1e-6))


def _layout_jp(segments: list[dict], look: TelopLook, age: int, max_box_w: int) -> Block | None:
    if not any(seg.get("text") or seg.get("reading") for seg in segments):
        return None
    k = age_scale(age)
    font = make_font(look.jp, k)
    kana_font = make_font(look.kana, k)
    fm = QFontMetricsF(font)
    kfm = QFontMetricsF(kana_font)
    pad = look.jp_pad if age == 0 else look.echo_jp_pad
    cap = max(MIN_TEXT_WIDTH, max_box_w - 2 * pad.side)

    def measure(s: str) -> int:
        return _ceil(fm.horizontalAdvance(s))

    def measure_kana(s: str) -> int:
        return _ceil(kfm.horizontalAdvance(s))

    rows, widths = [], []
    for row in layout_rows(segments, cap, measure_kanji=measure,
                           measure_kana=measure_kana, seg_padding=0):
        segs = []
        for seg in row:
            text = seg.get("text", "")
            reading = seg.get("reading") or ""
            text_w = measure(text)
            reading_w = measure_kana(reading) if reading else 0
            segs.append(RubySeg(text, reading, text_w, reading_w, max(text_w, reading_w)))
        rows.append(tuple(segs))
        widths.append(sum(s.width for s in segs))
    row_h = round_half_up(font.pixelSize() * look.jp_line_height)
    return Block("jp", age, pad, tuple(rows), tuple(widths), row_h, font, kana_font,
                 max(widths) + 2 * pad.side, pad.top + row_h * len(rows) + pad.bottom)


def _layout_en(en: str, look: TelopLook, age: int, max_box_w: int) -> Block | None:
    if not en.strip():   # whitespace-only English is no English (no empty bar)
        return None
    font = make_font(look.en, age_scale(age))
    fm = QFontMetricsF(font)
    pad = look.en_pad if age == 0 else look.echo_en_pad
    cap = max(MIN_TEXT_WIDTH, max_box_w - 2 * pad.side)
    layout = QTextLayout(en, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    # QTextLayout positions are UTF-16 code units; slice the UTF-16 bytes so
    # an emoji or other astral character cannot shift the row text.
    utf16 = en.encode("utf-16-le")
    rows, widths = [], []
    layout.beginLayout()
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(cap)
        start, length = line.textStart(), line.textLength()
        text = utf16[2 * start: 2 * (start + length)].decode("utf-16-le").rstrip()
        rows.append(text)
        widths.append(_ceil(fm.horizontalAdvance(text)))
    layout.endLayout()
    row_h = round_half_up(font.pixelSize() * look.en_line_height)
    return Block("en", age, pad, tuple(rows), tuple(widths), row_h, font, None,
                 max(widths) + 2 * pad.side, pad.top + row_h * len(rows) + pad.bottom)


def layout_line(segments: list[dict], en: str, look: TelopLook, age: int,
                max_box_w: int) -> LineLayout:
    """Measure one line for ``age``. ``max_box_w`` caps each block's box
    (the line cap minus the banner gutter). In Center alignment both blocks
    share the line's full width and center their rows."""
    blocks = []
    if look.shows_jp(age):
        jp = _layout_jp(segments, look, age, max_box_w)
        if jp is not None:
            blocks.append(jp)
    if look.shows_en(age):
        en_block = _layout_en(en, look, age, max_box_w)
        if en_block is not None:
            blocks.append(en_block)
    if look.alignment == "center" and blocks:
        shared = max(b.width for b in blocks)
        blocks = [dataclasses.replace(b, width=shared, centered=True) for b in blocks]
    return LineLayout(age, tuple(blocks))


def _row_x(block: Block, row_w: float) -> float:
    if block.centered:
        return block.pad.side + (block.width - 2 * block.pad.side - row_w) / 2
    return float(block.pad.side)


def _baseline(top: float, row_h: int, fm: QFontMetricsF) -> float:
    return top + (row_h - (fm.ascent() + fm.descent())) / 2 + fm.ascent()


def _em_inset(fm: QFontMetricsF, px: int) -> float:
    """How far the em box sits inside the font's ascent and descent. CSS
    centers the em box in that content area; for BIZ UDPGothic the two
    coincide (inset 0), Klee One's content area is 1.45 em."""
    return (fm.ascent() + fm.descent() - px) / 2


def _draw_glyphs(painter: QPainter, block: Block, main: QColor, kana: QColor,
                 dx: float = 0.0, dy: float = 0.0) -> None:
    fm = QFontMetricsF(block.font)
    if block.kind == "jp":
        kfm = QFontMetricsF(block.kana_font)
        for i, (row, row_w) in enumerate(zip(block.rows, block.row_widths)):
            base = _baseline(block.pad.top + i * block.row_h, block.row_h, fm) + dy
            # Furigana em box right on top of the base em box, as the
            # browser sets ruby (measured on the approved Learn screenshot).
            kana_base = (base - (fm.ascent() - _em_inset(fm, block.font.pixelSize()))
                         - (kfm.descent() - _em_inset(kfm, block.kana_font.pixelSize())))
            x = _row_x(block, row_w) + dx
            for seg in row:
                painter.setFont(block.font)
                painter.setPen(main)
                painter.drawText(QPointF(x + (seg.width - seg.text_w) / 2, base), seg.text)
                if seg.reading:
                    painter.setFont(block.kana_font)
                    painter.setPen(kana)
                    painter.drawText(QPointF(x + (seg.width - seg.reading_w) / 2, kana_base),
                                     seg.reading)
                x += seg.width
    else:
        painter.setFont(block.font)
        painter.setPen(main)
        for i, (row, row_w) in enumerate(zip(block.rows, block.row_widths)):
            base = _baseline(block.pad.top + i * block.row_h, block.row_h, fm) + dy
            painter.drawText(QPointF(_row_x(block, row_w) + dx, base), row)


def _blank(width: int, height: int, dpr: float) -> QImage:
    image = QImage(max(1, _ceil(width * dpr)), max(1, _ceil(height * dpr)),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(Qt.GlobalColor.transparent)
    return image


def _glyph_layer(block: Block, width: int, height: int, dpr: float,
                 main: QColor, kana: QColor) -> QImage:
    """``block``'s glyphs alone on a transparent image of the full size."""
    layer = _blank(width, height, dpr)
    painter = QPainter(layer)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.translate(IMAGE_MARGIN, IMAGE_MARGIN)
    _draw_glyphs(painter, block, main, kana)
    painter.end()
    return layer


def render_block(block: Block, look: TelopLook, dpr: float) -> QImage:
    """Paint ``block``'s text (not its plate) into a transparent image of the
    block's box plus IMAGE_MARGIN on each side, at device pixel ratio
    ``dpr``. Age 0 uses the plate look, older ages the echo look."""
    margin = IMAGE_MARGIN
    width, height = block.width + 2 * margin, block.height + 2 * margin
    image = _blank(width, height, dpr)
    echo = block.age > 0
    if block.kind == "jp":
        main = QColor(*(look.jp_echo if echo else look.jp_plate))
        kana = QColor(*(look.kana_echo if echo else look.kana_plate))
    else:
        main = kana = QColor(*(look.en_echo if echo else look.en_plate))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    if not echo:
        painter.translate(margin, margin)
        _draw_glyphs(painter, block, main, kana)
        painter.end()
        return image
    black = QColor(*OUTLINE)
    # Soft shadow: the black text, box-blurred by a downscale (area average)
    # and a smooth upscale back over the full image.
    shadow = _glyph_layer(block, width, height, dpr, black, black)
    box = SHADOW_BOX_PX * dpr
    small = shadow.scaled(max(1, round(shadow.width() / box)),
                          max(1, round(shadow.height() / box)),
                          Qt.AspectRatioMode.IgnoreAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
    painter.setOpacity(SHADOW_ALPHA)
    painter.drawImage(QRectF(0, 0, width, height), small)
    painter.setOpacity(1.0)
    painter.save()
    painter.translate(margin, margin)
    for dx, dy in _OUTLINE_OFFSETS:
        _draw_glyphs(painter, block, black, black, dx, dy)
    painter.restore()
    # The fill goes on its own transparent layer, laid over the outline.
    # Text drawn straight onto opaque pixels takes the ClearType subpixel
    # path on Windows and fringes orange and blue; on a transparent layer it
    # is grayscale, like the plate text.
    painter.drawImage(QPointF(0, 0), _glyph_layer(block, width, height, dpr, main, kana))
    painter.end()
    return image
