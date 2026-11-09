"""Painted building blocks of the setup wizard's steps (opt-wiz-rail.tpl).

The state marks (done check, current dot, paused dot, waiting ring) the rail
and the step lists share; the Download and Translator status rows; the alert
box with its warning icon; the Sound step's level meter; the Content step's
tone sample (night photo, banner, paper strip with furigana, dark bar); the
Done step's tray drawing; the Welcome step's 64 px icon. Every piece takes
the theme palette (MIST / NIGHT) once (L13: setup never follows a theme
change); sizes are the mockup's CSS px.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Sequence

from PySide6.QtCore import QDateTime, QLocale, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from yamabiko.overlay_app.controls import LinkButton, ui_font
from yamabiko.overlay_app.preview_monitor import SAMPLE_NEWEST, photo_path, photo_source
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from yamabiko.overlay_app.telop_banner import banner_content, paint_banner
from yamabiko.overlay_app.telop_look import (
    BANNER_RED, GENRE_BANNER, IMAGE_MARGIN, FontSpec, Pad, look_from_style,
)
from yamabiko.overlay_app.telop_text import layout_line, render_block
from yamabiko.overlay_app.window_chrome import paint_app_icon, paint_app_mark
from yamabiko.ui.error_window import TextBlock

_log = logging.getLogger(__name__)

STATES = ("done", "now", "paused", "waiting")
MARK = 14


def semibold(px: int) -> QFont:
    """The chrome's label font at ``px``, semibold (the mockup's 600)."""
    font = ui_font(px)
    font.setWeight(QFont.Weight.DemiBold)
    return font


def paint_state_mark(painter: QPainter, origin: QPointF, state: str, palette: dict) -> None:
    """One 14 x 14 state mark at ``origin``: "done" a 1.7 px ink check;
    "now" a filled ink dot; "paused" a filled muted dot; "waiting" a hollow
    muted ring (the dots: 8 px outside a 1.5 px edge)."""
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}")
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(origin)
    if state == "done":
        pen = QPen(QColor(palette["ink"]), 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        check = QPainterPath(QPointF(3, 7.4))
        check.lineTo(5.8, 10.1)
        check.lineTo(11, 4.4)
        painter.drawPath(check)
    else:
        tone = QColor(palette["ink" if state == "now" else "muted"])
        painter.setPen(QPen(tone, 1.5))
        painter.setBrush(Qt.NoBrush if state == "waiting" else tone)
        painter.drawEllipse(QRectF(3.75, 3.75, 6.5, 6.5))
    painter.restore()


class StateMark(QWidget):
    """A 14 x 14 state mark (the Sound step's result line)."""

    def __init__(self, state: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._state = "waiting"
        self.setFixedSize(MARK, MARK)
        self.set_state(state)

    def set_state(self, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self._state = state
        self.update()

    def state(self) -> str:
        return self._state

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        paint_state_mark(p, QPointF(0, 0), self._state, self._palette)
        p.end()


@dataclass(frozen=True)
class Run:
    """One piece of a status cell: text in a font and a palette tone."""

    text: str
    font: QFont
    color: str = "muted"


@dataclass
class _RowState:
    state: str = "waiting"
    runs: tuple = ()
    size: str = ""
    bar: float | None = None
    bar_stopped: bool = False


_COLUMN_GAP = 12
_SIZE_W = 64
_STATUS_W = 168
_RUN_GAP = 10
_BAR_W, _BAR_H = 96, 4
_TEXT_H = 20


class StatusRows(QWidget):
    """The Download and Translator lists: a 1 px rule on top, then one row
    per name, ``row_height`` tall with its own bottom rule. Columns: the
    state mark, the name (14 px ink), with ``sizes`` a 64 px right-aligned
    muted size column, and the status: ``Run`` pieces right-aligned 10 px
    apart, optionally after a 96 x 4 square-ended bar. Each row is an
    accessible element named by ``row_text`` (UI Automation reads it)."""

    def __init__(self, names: Sequence[str], palette: dict, parent: QWidget | None = None, *,
                 row_height: int = 44, sizes: bool = True) -> None:
        super().__init__(parent)
        self._palette = palette
        self._names = list(names)
        self._row_height = row_height
        self._sizes = sizes
        self._rows = [_RowState() for _ in self._names]
        self._access: list[QWidget] = []
        for index in range(len(self._names)):
            item = QWidget(self)
            item.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            item.setAccessibleName(self.row_text(index))
            self._access.append(item)
        self.setFixedHeight(1 + len(self._names) * row_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_row(self, index: int, state: str, runs: Sequence[Run] = (), *, size: str = "",
                bar: float | None = None, bar_stopped: bool = False) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self._rows[index] = _RowState(state, tuple(runs), size,
                                      None if bar is None else max(0.0, min(1.0, float(bar))),
                                      bool(bar_stopped))
        self._access[index].setAccessibleName(self.row_text(index))
        self.update()

    def row_text(self, index: int) -> str:
        row = self._rows[index]
        parts = (self._names[index], row.size, *(run.text for run in row.runs))
        return ", ".join(part for part in parts if part)

    def row_state(self, index: int) -> str:
        return self._rows[index].state

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        for index, item in enumerate(self._access):
            item.setGeometry(0, 1 + index * self._row_height, self.width(), self._row_height - 1)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        pal = self._palette
        width = self.width()
        line = QColor(pal["line"])
        p.fillRect(QRectF(0, 0, width, 1), line)
        name_font = ui_font(14)
        inner = self._row_height - 1
        for index, (name, row) in enumerate(zip(self._names, self._rows)):
            top = 1 + index * self._row_height
            p.fillRect(QRectF(0, top + inner, width, 1), line)
            paint_state_mark(p, QPointF(0, top + (inner - MARK) / 2.0), row.state, pal)
            text_top = top + (inner - _TEXT_H) / 2.0
            widths = [QFontMetricsF(run.font).horizontalAdvance(run.text) for run in row.runs]
            left = width - (sum(widths) + _RUN_GAP * max(0, len(widths) - 1))
            x = left
            for run, run_w in zip(row.runs, widths):
                p.setFont(run.font)
                p.setPen(QColor(pal[run.color]))
                p.drawText(QRectF(x, text_top, run_w + 1, _TEXT_H), Qt.AlignLeft | Qt.AlignVCenter,
                           run.text)
                x += run_w + _RUN_GAP
            status_left = left
            if row.bar is not None:
                bar_x = round(left - (_RUN_GAP if row.runs else 0) - _BAR_W)
                status_left = bar_x
                bar_y = round(top + (inner - _BAR_H) / 2.0)
                p.fillRect(bar_x, bar_y, _BAR_W, _BAR_H, QColor(pal["rail"]))
                p.fillRect(bar_x, bar_y, round(_BAR_W * row.bar), _BAR_H,
                           QColor(pal["muted" if row.bar_stopped else "accent"]))
            p.setFont(name_font)
            if self._sizes:
                size_right = width - _STATUS_W - _COLUMN_GAP
                if row.size:
                    p.setPen(QColor(pal["muted"]))
                    p.drawText(QRectF(size_right - _SIZE_W, text_top, _SIZE_W, _TEXT_H),
                               Qt.AlignRight | Qt.AlignVCenter, row.size)
                name_right = size_right - _SIZE_W - _COLUMN_GAP
            else:
                name_right = status_left - _COLUMN_GAP
            p.setPen(QColor(pal["ink"]))
            name_left = MARK + _COLUMN_GAP
            p.drawText(QRectF(name_left, text_top, max(0.0, name_right - name_left), _TEXT_H),
                       Qt.AlignLeft | Qt.AlignVCenter, name)
        p.end()


class _WarningIcon(QWidget):
    """The alert's 16 px accent warning triangle (opt-wiz-rail.tpl)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setFixedSize(16, 16)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        accent = QColor(self._palette["accent"])
        pen = QPen(accent, 1.5)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        triangle = QPainterPath(QPointF(8, 1.8))
        triangle.lineTo(15, 14.2)
        triangle.lineTo(1, 14.2)
        triangle.closeSubpath()
        p.drawPath(triangle)
        p.drawLine(QPointF(8, 6.2), QPointF(8, 10.2))
        p.setPen(Qt.NoPen)
        p.setBrush(accent)
        p.drawEllipse(QPointF(8, 12.1), 0.9, 0.9)
        p.end()


def set_primary(button: QPushButton, primary: bool) -> None:
    """Make ``button`` the accent-filled primary (or a plain one) under the
    window QSS's ``[primary="true"]`` rule, re-polished at once."""
    button.setProperty("primary", primary)
    style = button.style()
    style.unpolish(button)
    style.polish(button)
    button.update()


class Alert(QFrame):
    """The Download and Translator alert (QSS ``QFrame#wizardAlert``: raised,
    1 px line, 2 px corners): the 16 px warning icon, then a column with the
    message (600 14/20 ink), an optional help line (13/18 muted, 2 px under
    it), the action row (12 px under: buttons 8 px apart, links pushed to the
    right) and an optional line after it (13/18 muted, 10 px under)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("wizardAlert")
        self._palette = palette
        row = QHBoxLayout(self)
        # The QSS border (1 px) already insets the contents: 14 16 14 14 inside it.
        row.setContentsMargins(14, 14, 16, 14)
        row.setSpacing(12)
        icon_column = QVBoxLayout()
        icon_column.setContentsMargins(0, 2, 0, 0)
        icon_column.setSpacing(0)
        self._icon = _WarningIcon(palette)
        icon_column.addWidget(self._icon)
        icon_column.addStretch(1)
        row.addLayout(icon_column)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._message = TextBlock("", semibold(14), 20, palette, wrap="pretty")
        self._help = TextBlock("", ui_font(13), 18, palette, color="muted", wrap="pretty", top_pad=2)
        self._after = TextBlock("", ui_font(13), 18, palette, color="muted", wrap="pretty",
                                top_pad=10)
        self._actions = QHBoxLayout()
        self._actions.setContentsMargins(0, 12, 0, 0)
        self._actions.setSpacing(8)
        self._actions.addStretch(1)
        self._button_count = 0
        column.addWidget(self._message)
        column.addWidget(self._help)
        column.addLayout(self._actions)
        column.addWidget(self._after)
        row.addLayout(column, 1)
        self._help.hide()
        self._after.hide()

    def set_text(self, message: str, help: str = "", after: str = "") -> None:  # noqa: A002
        self._message.set_text(message)
        self._help.set_text(help)
        self._help.setVisible(bool(help))
        self._after.set_text(after)
        self._after.setVisible(bool(after))

    def message(self) -> str:
        return self._message.text()

    def help_text(self) -> str:
        return self._help.text()

    def after_text(self) -> str:
        return self._after.text()

    def add_button(self, text: str, *, primary: bool = False) -> QPushButton:
        """A 32 px button at the end of the left group (never the dialog's
        default: Enter stays with the footer's primary)."""
        button = QPushButton(text)
        button.setFont(ui_font(13))
        button.setAutoDefault(False)
        button.setProperty("primary", primary)
        self._actions.insertWidget(self._button_count, button)
        self._button_count += 1
        return button

    def add_link(self, text: str) -> LinkButton:
        """An underlined 32 px text button at the right end ("Copy details")."""
        link = LinkButton(text, self._palette, height=32)
        link.setFont(ui_font(13))
        self._actions.addWidget(link)
        return link


class LevelMeter(QWidget):
    """The Sound step's level meter: 32 square cells in 420 x 10, 2 px apart
    (cell edges snapped to whole px), lit from the left in ink on the rail."""

    CELLS = 32
    _W, _H, _GAP = 420, 10, 2

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._lit = 0
        self.setFixedSize(self._W, self._H)
        self.setAccessibleName("Sound level")

    def set_level(self, fraction: float) -> None:
        lit = round(max(0.0, min(1.0, float(fraction))) * self.CELLS)
        if lit != self._lit:
            self._lit = lit
            self.update()

    def lit_cells(self) -> int:
        return self._lit

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        step = (self._W + self._GAP) / self.CELLS
        on, off = QColor(self._palette["ink"]), QColor(self._palette["rail"])
        for index in range(self.CELLS):
            x0 = round(index * step)
            x1 = round((index + 1) * step) - self._GAP
            p.fillRect(x0, 0, x1 - x0, self._H, on if index < self._lit else off)
        p.end()


GENRE_ORDER = ("variety", "anime", "drama", "news")
GENRE_SAMPLES: dict[str, tuple[list[dict], str]] = {
    "variety": (SAMPLE_NEWEST[2], SAMPLE_NEWEST[1]),
    "anime": ([{"text": "お"}, {"text": "前", "reading": "まえ"}, {"text": "だけは、"},
               {"text": "絶対", "reading": "ぜったい"}, {"text": "に"},
               {"text": "許", "reading": "ゆる"}, {"text": "さない！"}],
              "You're the one I'll never forgive!"),
    "drama": ([{"text": "本当", "reading": "ほんとう"}, {"text": "のこと、"},
               {"text": "話", "reading": "はな"}, {"text": "してくれない？"}],
              "Won't you just tell me the truth?"),
    "news": ([{"text": "明日", "reading": "あした"}, {"text": "は"},
              {"text": "全国的", "reading": "ぜんこくてき"}, {"text": "に"},
              {"text": "雨", "reading": "あめ"}, {"text": "となるでしょう。"}],
             "Expect rain across the country tomorrow."),
}
# The tone sample's line, scaled down from the overlay (opt-wiz-rail.tpl
# .smp-*): BIZ UDPGothic 700 15 px on a 1.75 row with 8 px furigana, then
# Bahnschrift 600 19 px on a 1.2 row; paper strip padding 2 14 1, dark bar
# 7 14 8. Colors are the overlay's (paper, ink, 60 % furigana, the default
# bar #0e1014 at 237).
_SAMPLE_LOOK = dataclasses.replace(
    look_from_style(STYLE_DEFAULTS),
    alignment="left",
    jp=FontSpec("BIZ UDPGothic", 700, 15), kana=FontSpec("BIZ UDPGothic", 400, 8),
    en=FontSpec("Bahnschrift", 600, 19),
    jp_pad=Pad(2, 14, 1), en_pad=Pad(7, 14, 8),
)
_SAMPLE_W, _SAMPLE_H = 578, 214
_SAMPLE_LEFT, _SAMPLE_BOTTOM = 22, 20
_SAMPLE_BANNER_W = 30
_SAMPLE_BANNER_PX = 13


class ToneSample(QWidget):
    """The Content step's 578 x 214 sample: the night photo (cover crop at
    72 %, 2 px corners) with, 22 px from the left and 20 px from the bottom,
    the genre banner (30 px wide, 13 px cells) beside the sample line on its
    paper strip and dark bar. ``set_genre`` swaps banner and line."""

    def __init__(self, parent: QWidget | None = None, *, genre: str = "variety") -> None:
        super().__init__(parent)
        self.setFixedSize(_SAMPLE_W, _SAMPLE_H)
        self.setAccessibleName("Subtitle sample")
        self._photo = QPixmap(str(photo_path()))
        if self._photo.isNull():
            _log.warning("wizard.tone_sample.photo_missing path=%s", photo_path())
        self._genre = ""
        self.set_genre(genre)

    def set_genre(self, genre: str) -> None:
        if genre not in GENRE_SAMPLES:
            raise ValueError(f"unknown genre {genre!r}")
        self._genre = genre
        segments, en = GENRE_SAMPLES[genre]
        room = _SAMPLE_W - _SAMPLE_LEFT - _SAMPLE_BANNER_W
        self._line = layout_line(segments, en, _SAMPLE_LOOK, 0, room)
        jp, en_label = GENRE_BANNER[genre]
        self._banner = banner_content(jp, en_label, False, False, px=_SAMPLE_BANNER_PX)
        self.update()

    def genre(self) -> str:
        return self._genre

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 2.0, 2.0)
        p.setClipPath(clip)
        if self._photo.isNull():
            p.fillRect(self.rect(), QColor("#000000"))
        else:
            p.drawPixmap(QRectF(self.rect()), self._photo,
                         photo_source(self._photo, self.width(), self.height()))
        top = _SAMPLE_H - _SAMPLE_BOTTOM - self._line.height
        paint_banner(p, QRectF(_SAMPLE_LEFT, top, _SAMPLE_BANNER_W, self._line.height),
                     self._banner, QColor(*BANNER_RED))
        x = _SAMPLE_LEFT + _SAMPLE_BANNER_W
        y = float(top)
        dpr = self.devicePixelRatioF()
        for block in self._line.blocks:
            plate = _SAMPLE_LOOK.paper if block.kind == "jp" else _SAMPLE_LOOK.bar
            p.fillRect(QRectF(x, y, block.width, block.height), QColor(*plate))
            p.drawImage(QPointF(x - IMAGE_MARGIN, y - IMAGE_MARGIN),
                        render_block(block, _SAMPLE_LOOK, dpr))
            y += block.height
        p.end()


def _chord_arc(path: QPainterPath, x0: float, y0: float, x1: float, y1: float, r: float) -> None:
    """The SVG arc "M x0 y0 A r r 0 0 1 x1 y1": the short circular arc of
    radius ``r`` from (x0, y0) to (x1, y1), turning clockwise on screen."""
    dx, dy = x1 - x0, y1 - y0
    chord = math.hypot(dx, dy)
    rise = math.sqrt(max(0.0, r * r - (chord / 2.0) ** 2))
    cx, cy = (x0 + x1) / 2.0 - dy / chord * rise, (y0 + y1) / 2.0 + dx / chord * rise
    start = math.degrees(math.atan2(cy - y0, x0 - cx))
    end = math.degrees(math.atan2(cy - y1, x1 - cx))
    path.moveTo(x0, y0)
    path.arcTo(QRectF(cx - r, cy - r, 2 * r, 2 * r), start, -((start - end) % 360.0))


_TRAY_H = 44
_TILE = 32
_TILE_X = (7, 41, 75, 109)
_CLOCK_X = 143 + 10       # after tile 4, a 2 px gap and the clock's 10 px left padding
_CLOCK_RIGHT_PAD = 8 + 4 + 1  # the clock's padding, the tray's padding, the border


class TrayDrawing(QWidget):
    """The Done step's tray drawing (opt-wiz-rail.tpl .tray): a field box
    with a 1 px line edge, four 32 px tiles (chevron, the Yamabiko icon on a
    highlighted tile, Wi-Fi, speaker) and the clock: the time over the date
    in the system's short formats, as ``set_time`` gave them (not ticking)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._lines = ("", "")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Yamabiko in the notification area")
        self.set_time(QDateTime.currentDateTime())

    def set_time(self, when: QDateTime) -> None:
        locale = QLocale.system()
        self._lines = (locale.toString(when.time(), QLocale.FormatType.ShortFormat),
                       locale.toString(when.date(), QLocale.FormatType.ShortFormat))
        self.setFixedSize(self.sizeHint())
        self.update()

    def clock_lines(self) -> tuple[str, str]:
        return self._lines

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        metrics = QFontMetricsF(ui_font(12))
        clock = max(metrics.horizontalAdvance(line) for line in self._lines)
        return QSize(math.ceil(_CLOCK_X + clock + _CLOCK_RIGHT_PAD), _TRAY_H)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pal = self._palette
        ink = QColor(pal["ink"])
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(pal["line"]), 1.0))
        p.setBrush(QColor(pal["field"]))
        p.drawRoundedRect(box, 2.0, 2.0)
        top = (_TRAY_H - _TILE) / 2.0
        # 1: the overflow chevron (12 px glyph at +10).
        pen = QPen(ink, 1.2)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        chevron = QPainterPath(QPointF(_TILE_X[0] + 12.5, top + 17.5))
        chevron.lineTo(_TILE_X[0] + 16, top + 14)
        chevron.lineTo(_TILE_X[0] + 19.5, top + 17.5)
        p.drawPath(chevron)
        # 2: Yamabiko, on the hover tone with a 1 px edge inside the tile.
        tile = QRectF(_TILE_X[1], top, _TILE, _TILE)
        p.setPen(QPen(QColor(pal["edge"]), 1.0))
        p.setBrush(QColor(pal["hover"]))
        p.drawRoundedRect(tile.adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
        paint_app_mark(p, QRectF(_TILE_X[1] + 8, top + 8, 16, 16))
        # 3: Wi-Fi and 4: speaker, 16 px glyphs at +8.
        round_pen = QPen(ink, 1.2)
        round_pen.setCapStyle(Qt.RoundCap)
        round_pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(round_pen)
        p.setBrush(Qt.NoBrush)
        ox, oy = _TILE_X[2] + 8, top + 8
        wifi = QPainterPath()
        for x0, y, x1, r in ((1.8, 6.2, 14.2, 9), (3.9, 8.4, 12.1, 6), (6, 10.6, 10, 3)):
            _chord_arc(wifi, ox + x0, oy + y, ox + x1, oy + y, r)
        p.drawPath(wifi)
        ox = _TILE_X[3] + 8
        speaker = QPainterPath(QPointF(ox + 2.5, oy + 6))
        for x, y in ((5, 6), (8.5, 3), (8.5, 13), (5, 10), (2.5, 10)):
            speaker.lineTo(ox + x, oy + y)
        speaker.closeSubpath()
        _chord_arc(speaker, ox + 11, oy + 5.5, ox + 11, oy + 10.5, 3.5)
        _chord_arc(speaker, ox + 12.8, oy + 3.8, ox + 12.8, oy + 12.2, 6)
        p.drawPath(speaker)
        p.setPen(Qt.NoPen)
        p.setBrush(ink)
        p.drawEllipse(QPointF(_TILE_X[2] + 8 + 8, oy + 12.8), 1.0, 1.0)
        # The clock: two right-aligned 12/16 lines.
        p.setPen(ink)
        p.setFont(ui_font(12))
        right = self.width() - _CLOCK_RIGHT_PAD
        for index, line in enumerate(self._lines):
            p.drawText(QRectF(_CLOCK_X, 6 + 16 * index, right - _CLOCK_X, 16),
                       Qt.AlignRight | Qt.AlignVCenter, line)
        p.end()


class WelcomeIcon(QWidget):
    """The Welcome step's 64 px app icon (window_chrome.paint_app_icon)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(64, 64)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        paint_app_icon(p, QRectF(0, 0, 64, 64))
        p.end()
