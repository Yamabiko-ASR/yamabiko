"""Painted parts of the series bar (plan phase 5, Decisions 29-34, 40, 41, 50-52).

The bar window (``series_palette.SeriesPalette``) lays these out and owns all
AniList state; nothing here fetches or stores anything. Geometry, copy and
colors follow the approved "Spotlight bar" mockup (opt-pal-spot.tpl):

* text helpers: the display title, the short title (chip and status), the
  meta line's parts and the key hint of each view;
* painters: the search, alert and check icons and the 2 px bar line (the
  focus underline, or the progress line on its rail); covers go through the
  flyout's ``tray_flyout.paint_cover`` (1 px corners, no edge line), snapped
  to whole device pixels;
* the dropdown under the bar (``DropdownModel``, ``dropdown_height``,
  ``Dropdown`` with its ``TryAgainButton``), the current-series chip and the
  status text.

Every widget takes the theme palette and re-skins through ``apply_palette``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, NamedTuple

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor, QCursor, QFont, QFontMetricsF, QPainter, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import QAbstractButton, QWidget

from yamabiko.gazetteer.anilist import SeriesHit

from .controls import LinkButton, mincho_font, ui_font
from .theme import tooltip_qss
from .tray_flyout import paint_cover, series_title

# -- geometry (logical px; Decisions 29-31, 50) -----------------------------------

BAR_W = 640
EDGE = 1              # the outer ring DWM draws its border over
BAR_H = 48            # the bar alone, edge included
BAR_INNER_H = 46
TOP_GAP = 24          # below the subtitle monitor's work-area top
ROW_H = 52
LIST_PAD = 4
SECTION_H = 32
SPLIT_SECTION_H = 39
QUIET_H = 32
MESSAGE_H = 52
KEYS_H = 32
COVER_W, COVER_H = 32, 44
SMALL_COVER_W, SMALL_COVER_H = 20, 28
SEAL_PX = 26
CHIP_TITLE_MAX_W = 180

_DROP_W = BAR_W - 2 * EDGE              # 638: the dropdown spans the bar inside the edge
_LIST_X = LIST_PAD                      # rows and messages: x 4..634
_LIST_W = _DROP_W - 2 * LIST_PAD        # 630
_TEXT_X = 52                            # a message's text, from its left
_TEXT_RIGHT = 12                        # room kept right of a message's text
_ROW_BORDER = 1.5                       # the mockup's transparent row border (.row)
_ROW_COVER_X = _ROW_BORDER + 8          # 9.5: a row's cover, from its left
_ROW_TEXT_X = _ROW_COVER_X + COVER_W + 12   # 53.5: a row's title and meta line
_ROW_TEXT_RIGHT = _ROW_BORDER + 12      # 13.5: room kept right of a row's text
_SIDE_X = 12                            # section title, quiet line and key line
_WHO_RIGHT = _DROP_W - 12               # 626: the section's right label ends here
_RETRY_RIGHT = _LIST_X + _LIST_W - 8    # 626: Try again's right edge
_ICON = 18                              # the message icons' box
_ICON_CELL = 32                         # ... centered in this cell, 8 px into the message
_CHIP_COVER_X = 13                      # 1 px separator + 12 px padding
_CHIP_TITLE_X = _CHIP_COVER_X + SMALL_COVER_W + 8
_CHIP_CLEAR_GAP = 10
_CLEAR_H = 24
_STATUS_COVER_Y = 9                     # StatusText sits at the bar's y 1
_STATUS_TEXT_X = SMALL_COVER_W + 10
_STATUS_CHECK = 16
_STATUS_CHECK_Y = 15
_STATUS_TEXT_Y = 13
_STATUS_TEXT_H = 20

# -- copy (exact; L4, L9, Decision 31) ---------------------------------------------

PLACEHOLDER = "Search for a series"
WATCHING_TITLE = "Currently watching"
WHO = "{username} on AniList"
NO_MATCHES = "No matches for \u201c{query}\u201d."
NO_MATCHES_DETAIL = "Try the Japanese or romaji title."
OFFLINE = "Can't reach AniList right now."
TRY_AGAIN = "Try again"
LIST_UNAVAILABLE = "Your AniList list is unavailable."
LIST_PRIVATE = "Your AniList list is private."
KEYS_ROWS = "\u2191 \u2193 move, Enter picks, Esc closes"
KEYS_OFFLINE = "Enter tries again, Esc closes"
KEYS_CLOSE = "Esc closes"
LOADING = "Loading character names for {title}\u2026"
NOW_WATCHING = "Now watching {title}."
FORMAT_NAMES: dict[str, str] = {
    "TV": "TV", "TV_SHORT": "TV short", "MOVIE": "Movie", "SPECIAL": "Special",
    "OVA": "OVA", "ONA": "ONA", "MUSIC": "Music",
}
_META_SEP = " \u00b7 "

# -- views -------------------------------------------------------------------------

VIEW_BAR = "bar"
VIEW_WATCHING = "watching"
VIEW_RESULTS = "results"
VIEW_NO_MATCHES = "no_matches"
VIEW_OFFLINE = "offline"
VIEW_PRIVATE = "private"
_ROW_VIEWS = (VIEW_WATCHING, VIEW_RESULTS)


@dataclass(frozen=True)
class DropdownModel:
    """What the dropdown shows (Decision 36 picks it): one view, its rows
    (watching / results, at most six), the lit row (-1 none), the no-matches
    query, the section's right label ("" hides it) and, offline, whether the
    watch-list line shows."""

    view: str = VIEW_BAR
    rows: tuple[SeriesHit, ...] = ()
    lit: int = -1
    query: str = ""
    who: str = ""
    list_unavailable: bool = False


def dropdown_height(model: DropdownModel) -> int:
    """The dropdown's height for ``model`` (Decision 29); 0 for the bar alone."""
    if model.view in _ROW_VIEWS:
        section = SECTION_H if model.view == VIEW_WATCHING else 0
        return section + 2 * LIST_PAD + ROW_H * len(model.rows) + KEYS_H
    if model.view == VIEW_NO_MATCHES:
        return 2 * LIST_PAD + MESSAGE_H + KEYS_H
    if model.view == VIEW_OFFLINE:
        split = SPLIT_SECTION_H + QUIET_H if model.list_unavailable else 0
        return 2 * LIST_PAD + MESSAGE_H + split + KEYS_H
    if model.view == VIEW_PRIVATE:
        return SECTION_H + QUIET_H + KEYS_H
    return 0


def key_hint(view: str) -> str:
    """The quiet key line at the bottom of ``view``'s dropdown ("" for the
    bar alone)."""
    return {
        VIEW_WATCHING: KEYS_ROWS, VIEW_RESULTS: KEYS_ROWS, VIEW_NO_MATCHES: KEYS_CLOSE,
        VIEW_OFFLINE: KEYS_OFFLINE, VIEW_PRIVATE: KEYS_CLOSE,
    }.get(view, "")


# -- text ----------------------------------------------------------------------------

_YEAR_TAIL = re.compile(r"\s*\(\d{4}\)$")
_SEASON_TAIL = re.compile(r"\s+(?:Season\s+(\d+)|(\d+)(?:st|nd|rd|th)\s+Season)$")


def display_title(hit: SeriesHit) -> str:
    """The row title: the English title, else romaji."""
    return hit.english or hit.romaji


def short_title(title: str) -> str:
    """The chip's and the status' title (L7, Decision 40), in this order: no
    trailing " (YYYY)"; a trailing " Season N" or " Nth Season" set aside as
    " SN"; no leading "The "; nothing from the first ":" on; then the " SN"
    back on the end. The stripped input when nothing is left. Eliding to the
    room left is the painter's job."""
    stripped = title.strip()
    text = _YEAR_TAIL.sub("", stripped)
    season = ""
    match = _SEASON_TAIL.search(text)
    if match:
        season = f" S{match.group(1) or match.group(2)}"
        text = text[:match.start()]
    if text.startswith("The "):
        text = text[4:]
    text = text.split(":", 1)[0].strip()
    return text + season if text else stripped


class MetaPart(NamedTuple):
    """One segment of a row's muted meta line: ``kind`` "jp" (the native
    title, Japanese face), "plain" or "air" ("Episode N next", ink)."""

    text: str
    kind: str


def meta_parts(hit: SeriesHit) -> list[MetaPart]:
    """The meta line's segments (L6, Decision 41), missing ones skipped:
    native title, year, format, then "Episode N next" while airing, else the
    episode count when known."""
    out: list[MetaPart] = []
    if hit.native:
        out.append(MetaPart(hit.native, "jp"))
    if hit.season_year:
        out.append(MetaPart(str(hit.season_year), "plain"))
    fmt = FORMAT_NAMES.get(hit.fmt or "")
    if fmt:
        out.append(MetaPart(fmt, "plain"))
    if hit.next_airing_episode is not None:
        out.append(MetaPart(f"Episode {hit.next_airing_episode} next", "air"))
    elif isinstance(hit.episodes, int) and hit.episodes > 0:
        out.append(MetaPart("1 episode" if hit.episodes == 1 else f"{hit.episodes} episodes",
                            "plain"))
    return out


def jp_font(px: int = 13) -> QFont:
    """The meta line's Japanese face (the mockup's --jp): Yu Gothic UI, the
    bundled BIZ UDPGothic where it is missing."""
    font = QFont()
    font.setFamilies(["Yu Gothic UI", "BIZ UDPGothic"])
    font.setPixelSize(px)
    return font


# -- painters --------------------------------------------------------------------------


def paint_icon(p: QPainter, rect: QRectF, kind: str, color: QColor) -> None:
    """The mockup's line icons (Decision 34) scaled into ``rect``: "search"
    and "alert" on an 18-unit box, "check" on a 16-unit box."""
    box = 16.0 if kind == "check" else 18.0
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    p.translate(rect.topLeft())
    p.scale(rect.width() / box, rect.height() / box)
    pen = QPen(color)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setBrush(Qt.NoBrush)
    if kind == "search":
        pen.setWidthF(1.4)
        p.setPen(pen)
        p.drawEllipse(QPointF(7.5, 7.5), 5.5, 5.5)
        p.drawLine(QPointF(11.6, 11.6), QPointF(16.0, 16.0))
    elif kind == "alert":
        pen.setWidthF(1.4)
        p.setPen(pen)
        p.drawEllipse(QPointF(9.0, 9.0), 7.0, 7.0)
        pen.setWidthF(1.6)
        p.setPen(pen)
        p.drawLine(QPointF(9.0, 5.0), QPointF(9.0, 10.0))
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawEllipse(QPointF(9.0, 12.8), 1.0, 1.0)
    elif kind == "check":
        pen.setWidthF(1.8)
        p.setPen(pen)
        p.drawPolyline(QPolygonF([QPointF(3.0, 8.5), QPointF(6.2, 11.7), QPointF(13.0, 4.8)]))
    p.restore()


def paint_bar_line(p: QPainter, rect: QRectF, palette: dict, fraction: float | None) -> None:
    """The 2 px line along the bar's bottom (L9, L14): with ``fraction`` None
    the focused field's accent underline, else the progress line, the rail
    with the accent filled to ``fraction``."""
    if fraction is None:
        p.fillRect(rect, QColor(palette["accent"]))
        return
    p.fillRect(rect, QColor(palette["rail"]))
    share = max(0.0, min(1.0, float(fraction)))
    if share > 0.0:
        p.fillRect(QRectF(rect.x(), rect.y(), rect.width() * share, rect.height()),
                   QColor(palette["accent"]))


def _device_snapped(p: QPainter, rect: QRectF) -> QRectF:
    """``rect`` with its edges moved to the nearest device pixels through the
    painter's device transform (the widget's place in the window and the
    screen's scale), so a cover at a fractional position (a row's 9.5 px, or
    any edge at 125 or 150 %) paints crisp instead of blended."""
    to_device = p.deviceTransform()
    device = to_device.mapRect(rect)
    left, top = math.floor(device.left() + 0.5), math.floor(device.top() + 0.5)
    right, bottom = math.floor(device.right() + 0.5), math.floor(device.bottom() + 0.5)
    to_logical, _invertible = to_device.inverted()
    return to_logical.mapRect(QRectF(left, top, right - left, bottom - top))


def _paint_bar_cover(p: QPainter, rect: QRectF, cover: "QPixmap | QColor | None") -> None:
    """A row's, the chip's or the status' cover: the flyout's painter with
    1 px corners and no edge line, on whole device pixels."""
    paint_cover(p, _device_snapped(p, rect), cover, radius=1.0)


def _draw_runs(p: QPainter, runs: list[tuple[str, QFont, QColor]], rect: QRectF) -> None:
    """Draw ``runs`` left to right on one baseline (the ui-13 one, centered in
    ``rect``); the first run that does not fit is elided and ends the line
    (Decision 52)."""
    base = QFontMetricsF(ui_font(13))
    baseline = rect.top() + (rect.height() - base.height()) / 2.0 + base.ascent()
    x = rect.left()
    for text, font, color in runs:
        metrics = QFontMetricsF(font)
        width = metrics.horizontalAdvance(text)
        room = rect.right() - x
        last = width > room
        if last:
            text = metrics.elidedText(text, Qt.ElideRight, max(0.0, room))
        if text:
            p.setFont(font)
            p.setPen(color)
            p.drawText(QPointF(x, baseline), text)
        if last:
            return
        x += width


# -- widgets ---------------------------------------------------------------------------


class TryAgainButton(QAbstractButton):
    """The offline view's "Try again" (L4, Decision 31): 30 px tall, at least
    88 wide, a 1 px edge border with 2 px corners on the field tone (the
    hover tone under the pointer). It never takes focus: Enter in the field
    retries."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setText(TRY_AGAIN)
        self.setAccessibleName(TRY_AGAIN)
        self.setFont(ui_font(13))
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self.setFixedSize(self.sizeHint())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        text_w = math.ceil(QFontMetricsF(self.font()).horizontalAdvance(self.text()))
        return QSize(max(88, text_w + 32), 30)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        hover = self.isEnabled() and self.underMouse()
        p.setPen(QPen(QColor(self._palette["edge"]), 1.0))
        p.setBrush(QColor(self._palette["hover" if hover else "field"]))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
        p.setPen(QColor(self._palette["ink"]))
        p.drawText(QRectF(self.rect()), Qt.AlignCenter, self.text())
        p.end()


class Dropdown(QWidget):
    """The dropdown under the bar: one view of a ``DropdownModel`` (Decisions
    29-31, 50). The lit row fills with the accent, the row under the pointer
    takes the hover tone (hover never moves the lit row); a left click
    released on the row it pressed reports that row, Try again a retry.
    Covers come from the bar's ``cover_for``; the bar calls ``update()`` when
    one changes."""

    rowClicked = Signal(int)
    retryClicked = Signal()

    def __init__(self, palette: dict, cover_for: Callable[[int], "QPixmap | QColor"],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._cover_for = cover_for
        self._model = DropdownModel()
        self._hover = -1
        self._pressed = -1
        self.setMouseTracking(True)
        self.setFixedSize(_DROP_W, 0)
        self._retry = TryAgainButton(palette, self)
        self._retry.clicked.connect(lambda _checked=False: self.retryClicked.emit())
        self._retry.hide()

    # -- model ---------------------------------------------------------------------

    def set_model(self, model: DropdownModel) -> None:
        """Show ``model``: resize to it, place or hide Try again, repaint. New
        rows or a new view forget a press (its release picks nothing; the
        same index may hold another series now) and the hover is the row now
        under the pointer, which need not have moved."""
        if (model.view, model.rows) != (self._model.view, self._model.rows):
            self._pressed = -1
        self._model = model
        self.setFixedHeight(dropdown_height(model))
        if model.view == VIEW_OFFLINE:
            self._retry.move(_RETRY_RIGHT - self._retry.width(), LIST_PAD + 11)
            self._retry.show()
        else:
            self._retry.hide()
        self._hover = (self._row_at(self.mapFromGlobal(QCursor.pos()))
                       if self.underMouse() else -1)
        self.update()

    def model(self) -> DropdownModel:
        return self._model

    def row_rect(self, index: int) -> QRect:
        """Row ``index``'s rect in this widget (rows views)."""
        top = SECTION_H + LIST_PAD if self._model.view == VIEW_WATCHING else LIST_PAD
        return QRect(_LIST_X, top + index * ROW_H, _LIST_W, ROW_H)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self._retry.apply_palette(palette)
        self.update()

    # -- mouse ---------------------------------------------------------------------

    def _row_at(self, point) -> int:
        if self._model.view not in _ROW_VIEWS:
            return -1
        for index in range(len(self._model.rows)):
            if self.row_rect(index).contains(point):
                return index
        return -1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        row = self._row_at(event.position().toPoint())
        if row != self._hover:
            self._hover = row
            self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._hover != -1:
            self._hover = -1
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self._pressed = self._row_at(event.position().toPoint())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            row = self._row_at(event.position().toPoint())
            pressed, self._pressed = self._pressed, -1
            if row >= 0 and row == pressed:
                self.rowClicked.emit(row)
        event.accept()

    # -- painting ------------------------------------------------------------------

    def _color(self, token: str) -> QColor:
        return QColor(self._palette[token])

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        model = self._model
        if model.view == VIEW_BAR:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.fillRect(self.rect(), self._color("raised"))
        if model.view in (VIEW_WATCHING, VIEW_PRIVATE):
            self._paint_section(p, 0, split=False)
        if model.view in _ROW_VIEWS:
            for index, hit in enumerate(model.rows):
                self._paint_row(p, index, hit)
        elif model.view == VIEW_NO_MATCHES:
            self._paint_message(p, "search", NO_MATCHES.format(query=model.query),
                                NO_MATCHES_DETAIL, _LIST_X + _LIST_W - _TEXT_RIGHT)
        elif model.view == VIEW_OFFLINE:
            self._paint_message(p, "alert", OFFLINE, "", self._retry.x() - _TEXT_RIGHT)
            if model.list_unavailable:
                top = 2 * LIST_PAD + MESSAGE_H
                self._paint_section(p, top, split=True)
                self._paint_quiet(p, top + SPLIT_SECTION_H, LIST_UNAVAILABLE)
        else:  # VIEW_PRIVATE
            self._paint_quiet(p, SECTION_H, LIST_PRIVATE)
        self._paint_keys(p, self.height() - KEYS_H, key_hint(model.view))
        p.end()

    def _paint_row(self, p: QPainter, index: int, hit: SeriesHit) -> None:
        rect = QRectF(self.row_rect(index))
        lit = index == self._model.lit
        if lit or index == self._hover:
            p.setPen(Qt.NoPen)
            p.setBrush(self._color("accent" if lit else "hover"))
            p.drawRoundedRect(rect, 2.0, 2.0)
        _paint_bar_cover(p, QRectF(rect.x() + _ROW_COVER_X, rect.y() + 4, COVER_W, COVER_H),
                         self._cover_for(hit.media_id))
        text_x = rect.x() + _ROW_TEXT_X
        text_w = rect.width() - _ROW_TEXT_X - _ROW_TEXT_RIGHT
        title_font = ui_font(14)
        title = QFontMetricsF(title_font).elidedText(display_title(hit), Qt.ElideRight, text_w)
        p.setFont(title_font)
        p.setPen(self._color("on_accent" if lit else "ink"))
        p.drawText(QRectF(text_x, rect.y() + 8, text_w, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   title)
        _draw_runs(p, self._meta_runs(hit, lit), QRectF(text_x, rect.y() + 27, text_w, 17))

    def _meta_runs(self, hit: SeriesHit, lit: bool) -> list[tuple[str, QFont, QColor]]:
        plain = ui_font(13)
        muted = self._color("on_accent" if lit else "muted")
        runs: list[tuple[str, QFont, QColor]] = []
        for part in meta_parts(hit):
            if runs:
                runs.append((_META_SEP, plain, muted))
            if part.kind == "jp":
                runs.append((part.text, jp_font(13), muted))
            elif part.kind == "air":
                runs.append((part.text, plain, self._color("on_accent" if lit else "ink")))
            else:
                runs.append((part.text, plain, muted))
        return runs

    def _paint_section(self, p: QPainter, top: int, *, split: bool) -> None:
        """The section title in BIZ UDPMincho with the muted right label; the
        split variant (offline) has a line rule above it and no label."""
        if split:
            p.fillRect(QRectF(0, top + 4, self.width(), 1), self._color("line"))
        title_font = mincho_font(14)
        title_metrics = QFontMetricsF(title_font)
        box_top = top + (15 if split else 8)
        baseline = box_top + (20 - title_metrics.height()) / 2.0 + title_metrics.ascent()
        p.setFont(title_font)
        p.setPen(self._color("ink"))
        p.drawText(QPointF(_SIDE_X, baseline), WATCHING_TITLE)
        who = self._model.who
        if split or not who:
            return
        who_font = ui_font(13)
        who_metrics = QFontMetricsF(who_font)
        room = _WHO_RIGHT - (_SIDE_X + title_metrics.horizontalAdvance(WATCHING_TITLE) + 12)
        text = who_metrics.elidedText(who, Qt.ElideRight, max(0.0, room))
        p.setFont(who_font)
        p.setPen(self._color("muted"))
        p.drawText(QPointF(_WHO_RIGHT - who_metrics.horizontalAdvance(text), baseline), text)

    def _paint_quiet(self, p: QPainter, top: int, text: str) -> None:
        font = ui_font(13)
        width = self.width() - 2 * _SIDE_X
        p.setFont(font)
        p.setPen(self._color("muted"))
        p.drawText(QRectF(_SIDE_X, top + 2, width, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetricsF(font).elidedText(text, Qt.ElideRight, width))

    def _paint_message(self, p: QPainter, icon: str, title: str, detail: str,
                       text_right: float) -> None:
        top = LIST_PAD
        icon_x = _LIST_X + 8 + (_ICON_CELL - _ICON) / 2.0
        paint_icon(p, QRectF(icon_x, top + (MESSAGE_H - _ICON) / 2.0, _ICON, _ICON), icon,
                   self._color("accent" if icon == "alert" else "muted"))
        x = _LIST_X + _TEXT_X
        width = max(0.0, text_right - x)
        title_font = ui_font(14)
        title = QFontMetricsF(title_font).elidedText(title, Qt.ElideRight, width)
        p.setFont(title_font)
        p.setPen(self._color("ink"))
        if not detail:
            p.drawText(QRectF(x, top, width, MESSAGE_H), Qt.AlignLeft | Qt.AlignVCenter, title)
            return
        p.drawText(QRectF(x, top + 8, width, 18), Qt.AlignLeft | Qt.AlignVCenter, title)
        detail_font = ui_font(13)
        p.setFont(detail_font)
        p.setPen(self._color("muted"))
        p.drawText(QRectF(x, top + 27, width, 17), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetricsF(detail_font).elidedText(detail, Qt.ElideRight, width))

    def _paint_keys(self, p: QPainter, top: int, text: str) -> None:
        p.fillRect(QRectF(0, top, self.width(), 1), self._color("line"))
        p.setFont(ui_font(12))
        p.setPen(self._color("muted"))
        p.drawText(QRectF(_SIDE_X, top + 1, self.width() - 2 * _SIDE_X, KEYS_H - 1),
                   Qt.AlignLeft | Qt.AlignVCenter, text)


class SeriesChip(QWidget):
    """The active series at the bar's right (L8, Decisions 30, 57): a 1 px
    line separator, the 20 x 28 cover, the short title (elided at
    CHIP_TITLE_MAX_W) and an underlined "Clear"; the full title is its
    tooltip. Its width follows the title; the bar right-aligns it."""

    clearClicked = Signal()

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._cover: QPixmap | QColor | None = None
        self._title = ""
        self._title_w = 0
        self.setFont(ui_font(13))
        # The full title's tooltip in the theme, as the flyout's (L15).
        self.setStyleSheet(tooltip_qss(palette))
        self._clear = LinkButton("Clear", palette, self, height=_CLEAR_H)
        self._clear.setFont(ui_font(13))
        self._clear.setFocusPolicy(Qt.TabFocus)
        self._clear.clicked.connect(lambda _checked=False: self.clearClicked.emit())
        self._relayout()

    def set_series(self, label: str, cover: "QPixmap | QColor | None") -> None:
        """Show ``label`` (a format_hit label or a restored one) as its short
        title, the full title as the tooltip, and ``cover``."""
        self._title = short_title(label)
        self._cover = cover
        self.setToolTip(series_title(label))
        self._relayout()
        self.update()

    def set_cover(self, cover: "QPixmap | QColor | None") -> None:
        self._cover = cover
        self.update()

    def title(self) -> str:
        return self._title

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.setStyleSheet(tooltip_qss(palette))
        self._clear.apply_palette(palette)
        self.update()

    def _relayout(self) -> None:
        advance = QFontMetricsF(ui_font(13)).horizontalAdvance(self._title)
        self._title_w = min(math.ceil(advance), CHIP_TITLE_MAX_W)
        clear_w = self._clear.sizeHint().width()
        clear_x = _CHIP_TITLE_X + self._title_w + _CHIP_CLEAR_GAP
        self._clear.setGeometry(clear_x, (SMALL_COVER_H - _CLEAR_H) // 2, clear_w, _CLEAR_H)
        self.setFixedSize(clear_x + clear_w, SMALL_COVER_H)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.fillRect(QRectF(0, 0, 1, self.height()), QColor(self._palette["line"]))
        _paint_bar_cover(p, QRectF(_CHIP_COVER_X, 0, SMALL_COVER_W, SMALL_COVER_H), self._cover)
        font = ui_font(13)
        p.setFont(font)
        p.setPen(QColor(self._palette["ink"]))
        text = QFontMetricsF(font).elidedText(self._title, Qt.ElideRight, self._title_w)
        p.drawText(QRectF(_CHIP_TITLE_X, 0, self._title_w, self.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, text)
        p.end()


class StatusText(QWidget):
    """The status face's cover, check (once done) and text (L9, Decisions 30,
    63), right of the seal; the text is elided to the room left."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._cover: QPixmap | QColor | None = None
        self._text = ""
        self._done = False

    def set_status(self, cover: "QPixmap | QColor | None", text: str, *, done: bool) -> None:
        self._cover = cover
        self._text = text
        self._done = bool(done)
        self.update()

    def set_cover(self, cover: "QPixmap | QColor | None") -> None:
        self._cover = cover
        self.update()

    def text(self) -> str:
        return self._text

    def is_done(self) -> bool:
        return self._done

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        _paint_bar_cover(p, QRectF(0, _STATUS_COVER_Y, SMALL_COVER_W, SMALL_COVER_H),
                         self._cover)
        text_x = _STATUS_TEXT_X
        if self._done:
            paint_icon(p, QRectF(text_x, _STATUS_CHECK_Y, _STATUS_CHECK, _STATUS_CHECK), "check",
                       QColor(self._palette["accent"]))
            text_x += _STATUS_CHECK + 10
        font = ui_font(14)
        width = max(0, self.width() - text_x)
        p.setFont(font)
        p.setPen(QColor(self._palette["ink"]))
        p.drawText(QRectF(text_x, _STATUS_TEXT_Y, width, _STATUS_TEXT_H),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetricsF(font).elidedText(self._text, Qt.ElideRight, width))
        p.end()
