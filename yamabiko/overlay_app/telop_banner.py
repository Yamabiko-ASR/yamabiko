"""The telop overlay's vertical banner: genre or series name, or 一時停止.

Text is set vertically: kana and kanji upright, one per em cell, with the
font's vertical alternates ("vert"), so ー and small kana take their
vertical forms; Latin runs sideways (rotated clockwise, tops to the right).
Content order: the pause glyph (paused only), the Japanese, then the English
(banner setting 日本語 + English). The banner's natural height is its
content plus padding; SubtitleStack keeps it at least as tall as the newest
line and lets a long name grow it upward.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter

from .telop_look import (
    BANNER_EN_FONT, BANNER_EN_FONT_PX, BANNER_EN_FONT_WEIGHT, BANNER_FONT,
    BANNER_FONT_PX, BANNER_FONT_WEIGHT, BANNER_GAP, BANNER_PAD, BANNER_TEXT,
    PAUSE_BAR_GAP, PAUSE_BAR_W, PAUSE_GLYPH_H,
)


def split_vertical_runs(text: str) -> list[tuple[str, bool]]:
    """Split ``text`` into ``(run, upright)`` pieces: each East Asian wide or
    fullwidth character is its own upright run; consecutive other characters
    (Latin, digits, spaces) form one sideways run."""
    runs: list[tuple[str, bool]] = []
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            runs.append((ch, True))
        elif runs and not runs[-1][1]:
            runs[-1] = (runs[-1][0] + ch, False)
        else:
            runs.append((ch, False))
    return runs


def _font(family: str, px: int, weight: int, vertical: bool = False) -> QFont:
    font = QFont(family)
    font.setPixelSize(px)
    font.setWeight(QFont.Weight(weight))
    if vertical:
        font.setFeature(QFont.Tag("vert"), 1)
    return font


@dataclass(frozen=True, eq=False)
class BannerContent:
    jp: str
    en: str
    paused: bool
    show_en: bool
    runs: tuple              # tuple[(text, upright, length_px), ...]
    en_length: float
    upright_font: QFont
    side_font: QFont
    en_font: QFont

    def groups(self) -> list[float]:
        """Lengths of the pause glyph, the Japanese and the English, in
        order, for whichever are present."""
        out: list[float] = []
        if self.paused:
            out.append(float(PAUSE_GLYPH_H))
        if self.runs:
            out.append(sum(length for _, _, length in self.runs))
        if self.show_en and self.en:
            out.append(self.en_length)
        return out

    @property
    def natural_height(self) -> float:
        groups = self.groups()
        return 2 * BANNER_PAD + sum(groups) + BANNER_GAP * max(0, len(groups) - 1)


def banner_content(jp: str, en: str, paused: bool, show_en: bool, *,
                   px: int = BANNER_FONT_PX) -> BannerContent:
    """The banner's measured content; ``px`` is the Japanese cell size (the
    overlay's 18, setup's tone sample 13)."""
    upright = _font(BANNER_FONT, px, BANNER_FONT_WEIGHT, vertical=True)
    side = _font(BANNER_FONT, px, BANNER_FONT_WEIGHT)
    en_font = _font(BANNER_EN_FONT, BANNER_EN_FONT_PX, BANNER_EN_FONT_WEIGHT)
    side_fm = QFontMetricsF(side)
    runs = tuple(
        (text, up, float(px) if up else side_fm.horizontalAdvance(text))
        for text, up in split_vertical_runs(jp or "")
    )
    en_length = QFontMetricsF(en_font).horizontalAdvance(en) if (show_en and en) else 0.0
    return BannerContent(jp or "", en or "", bool(paused), bool(show_en), runs,
                         en_length, upright, side, en_font)


def _draw_sideways(painter: QPainter, font: QFont, text: str, cx: float, y: float) -> None:
    fm = QFontMetricsF(font)
    painter.save()
    painter.setFont(font)
    painter.translate(cx, y)
    painter.rotate(90)
    # Centre the em box on the banner's axis.
    painter.drawText(QPointF(0, (fm.ascent() - fm.descent()) / 2), text)
    painter.restore()


def paint_banner(painter: QPainter, rect: QRectF, content: BannerContent, tint: QColor,
                 text: QColor | None = None) -> None:
    """Fill ``rect`` with ``tint`` and draw the content centered in it in
    ``text`` (white when None), clipped to it (mid-reshape the rect can be
    shorter than the content)."""
    painter.save()
    painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
    painter.fillRect(rect, tint)
    groups = content.groups()
    inner = sum(groups) + BANNER_GAP * max(0, len(groups) - 1)
    y = rect.top() + (rect.height() - inner) / 2
    cx = rect.center().x()
    ink = QColor(*BANNER_TEXT) if text is None else text
    if content.paused:
        x0 = cx - (2 * PAUSE_BAR_W + PAUSE_BAR_GAP) / 2
        painter.fillRect(QRectF(x0, y, PAUSE_BAR_W, PAUSE_GLYPH_H), ink)
        painter.fillRect(QRectF(x0 + PAUSE_BAR_W + PAUSE_BAR_GAP, y,
                                PAUSE_BAR_W, PAUSE_GLYPH_H), ink)
        y += PAUSE_GLYPH_H + BANNER_GAP
    painter.setPen(ink)
    if content.runs:
        fm = QFontMetricsF(content.upright_font)
        for text_run, upright, length in content.runs:
            if upright:
                painter.setFont(content.upright_font)
                base = y + (length - (fm.ascent() + fm.descent())) / 2 + fm.ascent()
                painter.drawText(QPointF(cx - fm.horizontalAdvance(text_run) / 2, base), text_run)
            else:
                _draw_sideways(painter, content.side_font, text_run, cx, y)
            y += length
        y += BANNER_GAP
    if content.show_en and content.en:
        _draw_sideways(painter, content.en_font, content.en, cx, y)
    painter.restore()
