"""The tray flyout: the panel a left click on the tray icon opens.

Surface 5 of the UI redesign ("Flyout panel": opt-tray-flyout.tpl and the
screenshots 5 Tray/03-10). A 360 x 426 frameless panel in the app look: the
header carries the seal, "Yamabiko", the status line and the big Pause /
Resume button over a low ridgeline; the body the Subtitles and Mode tiles,
the Genre segments, the series row, the Position picker, Alignment and
"Move subtitles…"; the footer Settings, Troubleshooting (Open log folder,
Save diagnostics) and Quit. Without the series feature the series row goes
and the panel is 358 px tall.

The widget shows state and reports intent; it does no I/O. The tray feeds
it through the setters and acts on its signals. Closing actions hide the
panel before they emit (plan phase 4, Decision 50). It hides when it loses
activation unless its own popup is open, and on Esc (Decision 54). DWM
draws the small rounded corners, the soft shadow and the frame-tone border
once it is shown (window_chrome.set_corner_preference, set_border_color).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from PySide6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPalette,
    QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QHBoxLayout, QLabel, QLayout, QMenu, QPushButton,
    QSizePolicy, QSpacerItem, QStyle, QStyleOptionComboBox, QVBoxLayout, QWidget,
)

from .controls import ChevronCombo, LinkButton, mincho_font, ui_font
from .landscape import paint_ridgeline
from .position_config import DEFAULT_POSITION, MIDDLE_V_PCT
from .seal import Seal
from .segmented import KeyboardRingMixin, SegmentedControl, SegmentOption
from .theme import menu_qss
from .window_chrome import (
    DWMWCP_ROUNDSMALL, SquareCorners, apply_window_qss, set_border_color, set_corner_preference,
)

FLYOUT_WIDTH = 360
FLYOUT_HEIGHT = 426
FLYOUT_HEIGHT_NO_SERIES = 358
# (key, label) in display order; the keys are the pipeline's genres.
GENRE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("variety", "Variety"), ("anime", "Anime"), ("drama", "Drama"), ("news", "News"),
)
_ALIGNMENT_OPTIONS = (("left", "Left"), ("center", "Center"))
_MODES = ("watch", "learn")

# Section heights (Decisions 34 and 51): header 84; top 12 + tiles 56 + 12 +
# Genre 32 + 12 + series 56 + 12; divider 1; bottom 12 + Position 32 + 12 +
# Alignment 32 + 14; footer rule 1; footer 46.
_HEADER_H = 84
_TOP_H = 192
_TOP_H_NO_SERIES = 124
_BOTTOM_H = 102
_FOOTER_H = 46
_ROW_H = 32
_TILE_H = 56
_LABEL_W = 72
_ALIGN_W = 116
_BIG_H = 38
_BIG_MIN_W = 112
_RADIUS = 4.0
_COVER_W = 28
_COVER_H = 40
_PROBLEM_DOT = "#e0a030"
# The cover tone of a series with no poster and no AniList color: the
# flyout's series row and the series bar (paint_cover, and
# SeriesPalette.cover_for's last fallback).
NEUTRAL_COVER = "#2b2b33"
_MATCH_TOLERANCE = 0.05

# Segoe Fluent Icons code points (Segoe MDL2 Assets carries the same ones).
_GLYPH_FAMILIES = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]
_G_PAUSE = "\ue769"
_G_PLAY = "\ue768"
_G_SUBTITLES = "\ue7f0"
_G_WATCH = "\ue890"
_G_LEARN = "\ue7be"
_G_SWITCH = "\ue8ab"
_G_SETTINGS = "\ue713"
_G_REPAIR = "\ue90f"
_G_CHEVRON_UP = "\ue70e"
_G_POWER = "\ue7e8"

_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")


def series_title(label: str) -> str:
    """The series label without its trailing " (year)" ("Frieren (2023)" ->
    "Frieren"); a label without one comes back stripped."""
    return _YEAR.sub("", label).strip()


def _glyph_font(px: int) -> QFont:
    font = QFont()
    font.setFamilies(_GLYPH_FAMILIES)
    font.setPixelSize(px)
    return font


def _semibold(px: int) -> QFont:
    font = ui_font(px)
    font.setWeight(QFont.Weight.DemiBold)
    return font


def _with_combo(text: str, combo: str | None) -> str:
    return f"{text} ({combo})" if combo else text


@dataclass(frozen=True)
class PositionEntry:
    """One row of the Position picker. ``screen`` and ``v_offset_pct`` are
    what the preset writes to overlay.position; the current-value row
    ("Display N") has neither and does nothing when picked."""

    label: str
    screen: str | None = None
    v_offset_pct: float | None = None


def _is_preset(position: Mapping, v_pct: float) -> bool:
    """True when ``position`` is the preset at ``v_pct``; a preset also
    writes the default horizontal anchors of both alignments."""
    try:
        return (
            abs(float(position.get("v_offset_pct")) - v_pct) < _MATCH_TOLERANCE
            and abs(float(position.get("h_offset_pct"))
                    - DEFAULT_POSITION["h_offset_pct"]) < _MATCH_TOLERANCE
            and abs(float(position.get("left_offset_pct"))
                    - DEFAULT_POSITION["left_offset_pct"]) < _MATCH_TOLERANCE
        )
    except (TypeError, ValueError):
        return False


def position_entries(
    screens: Sequence[tuple[str, bool]], position: Mapping,
) -> tuple[list[PositionEntry], int]:
    """Today's presets for ``screens`` (``(QScreen.name(), is_primary)`` in
    Qt screen order; N counts from 1) and the index of the one ``position``
    (the overlay's current position) matches. The screen resolves like
    OverlayWindow._pin_to_screen: a name match, else the primary. With no
    match a first "Display N" row for that screen is the current one."""
    names = [name for name, _primary in screens]
    primary = next((i for i, (_name, is_primary) in enumerate(screens) if is_primary), 0)
    want = position.get("screen")
    on = names.index(want) if want in names else primary
    entries: list[PositionEntry] = []
    current = -1
    for index, name in enumerate(names):
        for word, v_pct in (("Bottom", DEFAULT_POSITION["v_offset_pct"]),
                            ("Middle", MIDDLE_V_PCT)):
            if index == on and _is_preset(position, v_pct):
                current = len(entries)
            entries.append(PositionEntry(f"{word} of Display {index + 1}", name, v_pct))
    if current >= 0:
        return entries, current
    return [PositionEntry(f"Display {on + 1}")] + entries, 0


def live_screens() -> list[tuple[str, bool]]:
    """``(QScreen.name(), is_primary)`` for every screen, in Qt order."""
    primary = QGuiApplication.primaryScreen()
    return [(screen.name(), screen is primary) for screen in QGuiApplication.screens()]


class _FlyButton(KeyboardRingMixin, QAbstractButton):
    """Base of the flyout's painted buttons: palette, hover, Return/Enter
    click, the keyboard-only focus ring. Esc is left to the panel."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setAttribute(Qt.WA_Hover, True)
        self.setFocusPolicy(Qt.StrongFocus)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def _color(self, token: str) -> QColor:
        return QColor(self._palette[token])

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().keyPressEvent(event)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            self.click()

    def _paint_ring(self, painter: QPainter, body: QRectF, token: str = "accent") -> None:
        if self.ring_visible():
            painter.setPen(QPen(self._color(token), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(body.adjusted(1.5, 1.5, -1.5, -1.5), 3.0, 3.0)


class _PauseButton(_FlyButton):
    """The header's big button: "Pause" on the field, or "Resume" filled with
    the accent while paused; 38 px tall, at least 112 px wide, one width for
    both faces."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(palette, parent)
        self._paused = False
        self.setFont(_semibold(14))
        self.setText("Pause")
        self.setAccessibleName("Pause")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        glyph_w = QFontMetrics(_glyph_font(14)).horizontalAdvance(_G_PAUSE)
        text_w = max(self.fontMetrics().horizontalAdvance(t) for t in ("Pause", "Resume"))
        return QSize(max(_BIG_MIN_W, 16 + glyph_w + 9 + text_w + 16), _BIG_H)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.setText("Resume" if paused else "Pause")
        self.setAccessibleName(self.text())
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        hover = self.isEnabled() and self.underMouse()
        body = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._paused:
            fill = self._color("accent_hover" if hover else "accent")
            p.setPen(QPen(fill, 1.0))
            p.setBrush(fill)
            ink = self._color("on_accent")
        else:
            p.setPen(QPen(self._color("edge"), 1.0))
            p.setBrush(self._color("hover" if hover else "field"))
            ink = self._color("ink")
        p.drawRoundedRect(body, _RADIUS, _RADIUS)
        glyph = _G_PLAY if self._paused else _G_PAUSE
        glyph_font = _glyph_font(14)
        glyph_w = QFontMetrics(glyph_font).horizontalAdvance(glyph)
        text_w = self.fontMetrics().horizontalAdvance(self.text())
        x = (self.width() - (glyph_w + 9 + text_w)) / 2.0
        p.setPen(ink)
        p.setFont(glyph_font)
        p.drawText(QRectF(x, 0, glyph_w, self.height()), Qt.AlignCenter, glyph)
        p.setFont(self.font())
        p.drawText(QRectF(x + glyph_w + 9, 0, text_w + 1, self.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, self.text())
        self._paint_ring(p, body, "on_accent" if self._paused else "accent")
        p.end()


class _Tile(_FlyButton):
    """A 56 px body tile: a glyph, a semibold title over its value and an
    optional trailing glyph; accent-filled while lit. A click does not flip
    it: the panel sets it from the state it shows (set_state)."""

    def __init__(self, title: str, palette: dict, parent: QWidget | None = None,
                 *, trailing: str = "") -> None:
        super().__init__(palette, parent)
        self._title = title
        self._trailing = trailing
        self._glyph = ""
        self._value = ""
        self._lit = False
        self.setCheckable(True)
        self.setFixedHeight(_TILE_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def nextCheckState(self) -> None:  # noqa: N802 (Qt override)
        """Keep the check state on click; set_state owns it."""

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(160, _TILE_H)

    def set_state(self, glyph: str, value: str, *, checked: bool, lit: bool) -> None:
        self._glyph = glyph
        self._value = value
        self._lit = lit
        self.setChecked(checked)
        self.setAccessibleDescription(value)
        self.update()

    def value(self) -> str:
        return self._value

    def is_lit(self) -> bool:
        return self._lit

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        hover = self.isEnabled() and self.underMouse()
        body = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._lit:
            fill = self._color("accent_hover" if hover else "accent")
            p.setPen(QPen(fill, 1.0))
            p.setBrush(fill)
            ink = sub = self._color("on_accent")
        else:
            p.setPen(QPen(self._color("edge"), 1.0))
            p.setBrush(self._color("hover" if hover else "field"))
            ink, sub = self._color("ink"), self._color("muted")
        p.drawRoundedRect(body, _RADIUS, _RADIUS)
        h = self.height()
        p.setPen(ink)
        p.setFont(_glyph_font(18))
        p.drawText(QRectF(12, 0, 20, h), Qt.AlignCenter, self._glyph)
        right = self.width() - 12
        if self._trailing:
            trailing_font = _glyph_font(13)
            trailing_w = QFontMetrics(trailing_font).horizontalAdvance(self._trailing)
            p.setFont(trailing_font)
            p.setPen(sub)
            p.drawText(QRectF(right - trailing_w, 0, trailing_w, h), Qt.AlignCenter,
                       self._trailing)
            right -= trailing_w + 10
        text_x = 12 + 20 + 10
        width = max(0, right - text_x)
        top = (h - 36) / 2.0
        title_font = _semibold(13)
        p.setFont(title_font)
        p.setPen(ink)
        p.drawText(QRectF(text_x, top, width, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(title_font).elidedText(self._title, Qt.ElideRight, width))
        p.setFont(self.font())
        p.setPen(sub)
        p.drawText(QRectF(text_x, top + 18, width, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   self.fontMetrics().elidedText(self._value, Qt.ElideRight, width))
        self._paint_ring(p, body, "on_accent" if self._lit else "accent")
        p.end()


class _FooterButton(_FlyButton):
    """A 32 px footer button: muted glyph, ink label, an optional trailing
    chevron; the hover tone on hover and while its menu is open."""

    def __init__(self, glyph: str, text: str, palette: dict, parent: QWidget | None = None,
                 *, chevron: bool = False) -> None:
        super().__init__(palette, parent)
        self._glyph = glyph
        self._chevron = chevron
        self._open = False
        self.setText(text)
        self.setAccessibleName(text)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _widths(self) -> tuple[int, int, int]:
        glyph_w = QFontMetrics(_glyph_font(14)).horizontalAdvance(self._glyph)
        text_w = self.fontMetrics().horizontalAdvance(self.text())
        chevron_w = (QFontMetrics(_glyph_font(9)).horizontalAdvance(_G_CHEVRON_UP)
                     if self._chevron else 0)
        return glyph_w, text_w, chevron_w

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        glyph_w, text_w, chevron_w = self._widths()
        extra = 6 + chevron_w if self._chevron else 0
        return QSize(10 + glyph_w + 8 + text_w + extra + 10, _ROW_H)

    def set_open(self, open_: bool) -> None:
        self._open = open_
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        body = QRectF(self.rect())
        if self._open or (self.isEnabled() and self.underMouse()):
            p.setPen(Qt.NoPen)
            p.setBrush(self._color("hover"))
            p.drawRoundedRect(body, _RADIUS, _RADIUS)
        glyph_w, text_w, chevron_w = self._widths()
        h = self.height()
        p.setFont(_glyph_font(14))
        p.setPen(self._color("muted"))
        p.drawText(QRectF(10, 0, glyph_w, h), Qt.AlignCenter, self._glyph)
        x = 10 + glyph_w + 8
        p.setFont(self.font())
        p.setPen(self._color("ink"))
        p.drawText(QRectF(x, 0, text_w + 1, h), Qt.AlignLeft | Qt.AlignVCenter, self.text())
        if self._chevron:
            p.setFont(_glyph_font(9))
            p.setPen(self._color("muted"))
            p.drawText(QRectF(x + text_w + 6, 0, chevron_w, h), Qt.AlignCenter, _G_CHEVRON_UP)
        self._paint_ring(p, body)
        p.end()


class _StatusLine(QWidget):
    """The header status: a 7 px indicator (a dot, or the muted pause bars)
    and the status text on an 18 px line."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._kind = "listening"
        self._text = "Listening"
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(self._text)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def set_status(self, kind: str, text: str) -> None:
        self._kind = kind
        self._text = text
        self.setAccessibleName(text)
        self.update()

    def kind(self) -> str:
        return self._kind

    def text(self) -> str:
        return self._text

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        muted = QColor(self._palette["muted"])
        if self._kind == "paused":
            p.fillRect(QRectF(0, 4, 2, 9), muted)
            p.fillRect(QRectF(5, 4, 2, 9), muted)
        else:
            dot = {"problem": QColor(_PROBLEM_DOT), "hidden": muted}.get(
                self._kind, QColor(self._palette["accent"]))
            p.setPen(Qt.NoPen)
            p.setBrush(dot)
            p.drawEllipse(QRectF(0, 5.5, 7, 7))
        p.setPen(QColor(self._palette["ink"]))
        text_rect = QRectF(14, 0, max(0, self.width() - 14), self.height())
        p.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter,
                   self.fontMetrics().elidedText(self._text, Qt.ElideRight,
                                                 int(text_rect.width())))
        p.end()


class _Header(QWidget):
    """The 84 px header on the ground: the seal, "Yamabiko" over the status
    line, the big button, and the low ridgeline along the bottom edge."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setFixedHeight(_HEADER_H)
        self.seal = Seal(30, self)
        self.title = QLabel("Yamabiko", self)
        self.title.setFont(mincho_font(18, bold=True))
        self.title.setFixedHeight(22)
        self.status = _StatusLine(palette, self)
        self.button = _PauseButton(palette, self)
        seal_box = QVBoxLayout()
        seal_box.setContentsMargins(0, 2, 0, 0)
        seal_box.addWidget(self.seal)
        seal_box.addStretch(1)
        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(3)
        text_box.addWidget(self.title)
        text_box.addWidget(self.status)
        text_box.addStretch(1)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 16, 16, 0)
        row.setSpacing(12)
        row.addLayout(seal_box)
        row.addLayout(text_box, 1)
        row.addWidget(self.button, 0, Qt.AlignTop)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.status.apply_palette(palette)
        self.button.apply_palette(palette)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._palette["ground"]))
        paint_ridgeline(p, QRectF(0, self.height() - 28, self.width(), 28), self._palette)
        p.end()


class _Rule(QWidget):
    """A 1 px rule in the line tone (the body divider, the footer's top)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setFixedHeight(1)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._palette["line"]))
        p.end()


def paint_cover(p: QPainter, rect: QRectF, cover: "QPixmap | QColor | None", *,
                radius: float, edge: QColor | None = None) -> None:
    """A series cover in ``rect`` with ``radius`` corners: the poster cropped
    to fill (CSS object-fit: cover), else the color (a hit's AniList color),
    else the neutral tone; with ``edge``, a 1 px line of that color just
    inside the corners. The one cover painter of the flyout's series row and
    the series bar's rows, chip and status (series_palette_parts)."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    clip = QPainterPath()
    clip.addRoundedRect(rect, radius, radius)
    p.setClipPath(clip, Qt.ClipOperation.IntersectClip)
    if isinstance(cover, QPixmap) and not cover.isNull():
        scale = max(rect.width() / cover.width(), rect.height() / cover.height())
        width, height = cover.width() * scale, cover.height() * scale
        p.drawPixmap(QRectF(rect.center().x() - width / 2.0, rect.center().y() - height / 2.0,
                            width, height), cover, QRectF(cover.rect()))
    else:
        valid = isinstance(cover, QColor) and cover.isValid()
        p.fillRect(rect, cover if valid else QColor(NEUTRAL_COVER))
    p.restore()
    if edge is not None:
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(edge, 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius - 0.5, radius - 0.5)
        p.restore()


class _Cover(QWidget):
    """The 28 x 40 series cover: the poster cropped to fill (CSS object-fit:
    cover) or a color tile, with 2 px corners and a faint 1 px edge."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._cover: QPixmap | QColor | None = None
        self.setFixedSize(_COVER_W, _COVER_H)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def set_cover(self, cover: "QPixmap | QColor | None") -> None:
        self._cover = cover
        self.update()

    def cover(self) -> "QPixmap | QColor | None":
        return self._cover

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._cover is None:
            return
        edge = (QColor(0, 0, 0, 31) if self._palette["name"] == "mist"
                else QColor(255, 255, 255, 26))
        p = QPainter(self)
        paint_cover(p, QRectF(self.rect()), self._cover, radius=2.0, edge=edge)
        p.end()


class _SeriesText(QWidget):
    """The series row's two elided lines: the title (semibold) over the
    Japanese title (muted); one muted "No series" line without a series."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._title = "No series"
        self._subtitle = ""
        self._muted = True
        self.setFixedHeight(35)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def set_text(self, title: str, subtitle: str, *, muted: bool) -> None:
        self._title = title
        self._subtitle = subtitle
        self._muted = muted
        self.setAccessibleName(f"{title}, {subtitle}" if subtitle else title)
        self.update()

    def title(self) -> str:
        return self._title

    def subtitle(self) -> str:
        return self._subtitle

    def muted(self) -> bool:
        return self._muted

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        width = self.width()
        title_font = ui_font(13) if self._muted else _semibold(13)
        top = 0.0 if self._subtitle else (self.height() - 18) / 2.0
        p.setFont(title_font)
        p.setPen(QColor(self._palette["muted" if self._muted else "ink"]))
        p.drawText(QRectF(0, top, width, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(title_font).elidedText(self._title, Qt.ElideRight, width))
        if self._subtitle:
            sub_font = ui_font(13)
            p.setFont(sub_font)
            p.setPen(QColor(self._palette["muted"]))
            p.drawText(QRectF(0, 18, width, 17), Qt.AlignLeft | Qt.AlignVCenter,
                       QFontMetrics(sub_font).elidedText(self._subtitle, Qt.ElideRight, width))
        p.end()


class _SeriesRow(QWidget):
    """The 56 px series row, a ground field with a line border: cover, the
    two title lines, Change… and Clear."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setFixedHeight(_TILE_H)
        self.cover = _Cover(palette, self)
        self.text = _SeriesText(palette, self)
        self.change = LinkButton("Change…", palette, self, height=28)
        self.clear = LinkButton("Clear", palette, self, height=28)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 7, 13, 7)
        self._layout.setSpacing(0)
        self._layout.addWidget(self.cover)
        self._gap = QSpacerItem(12, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self._layout.addItem(self._gap)
        self._layout.addWidget(self.text, 1)
        self._layout.addSpacing(11)
        self._layout.addWidget(self.change)
        self._layout.addSpacing(18)
        self._layout.addWidget(self.clear)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        for part in (self.cover, self.text, self.change, self.clear):
            part.apply_palette(palette)
        self.update()

    def show_cover(self, shown: bool) -> None:
        self.cover.setVisible(shown)
        self._gap.changeSize(12 if shown else 0, 0, QSizePolicy.Policy.Fixed,
                             QSizePolicy.Policy.Minimum)
        self._layout.invalidate()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(self._palette["line"]), 1.0))
        p.setBrush(QColor(self._palette["ground"]))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), _RADIUS, _RADIUS)
        p.end()


class _PositionPicker(KeyboardRingMixin, ChevronCombo):
    """The Position display picker; reports when its open list closes.

    Only a pick from the open list applies (Decision 74): a closed
    QComboBox changes its row and emits ``activated`` on a wheel notch, the
    navigation keys or typed text, which would move and save the subtitles
    and close the panel. The wheel is ignored; those keys open the list.

    Its focus ring is keyboard-only, like the panel's painted controls: the
    window QSS keeps the plain edge on :focus and the picker paints the
    accent edge while ``ring_visible()``. A list opened with the mouse
    leaves the focus here, and the reopened panel hands it back; neither is
    keyboard use.

    Its open list stays inside the work area: Qt places a combo list against
    the whole screen on Windows, so below a picker near the taskbar it ran
    over the taskbar. As the list shows, it moves above the picker when its
    bottom passes the work area (Qt's own placement when the screen runs
    out)."""

    popupClosed = Signal()

    _OPEN_KEYS = frozenset((Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End,
                            Qt.Key_PageUp, Qt.Key_PageDown))

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        event.ignore()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        text = event.text()
        if event.key() in self._OPEN_KEYS or (text and text[0].isprintable()):
            event.accept()
            self._ring = True  # keyboard use, like the activation keys
            self.showPopup()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        if self.ring_visible():
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(QColor(self._palette["accent"]), 1.0))
            p.setBrush(Qt.NoBrush)
            # Over the QSS edge: 1 px, 4 px corners.
            p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 3.5, 3.5)
            p.end()

    def showPopup(self) -> None:  # noqa: N802 (Qt override)
        # The list's Show (eventFilter) comes before its native show. Qt's
        # roll animation would play at Qt's spot first, so it is off for this
        # list.
        self.view().parentWidget().installEventFilter(self)
        animate = QApplication.isEffectEnabled(Qt.UI_AnimateCombo)
        QApplication.setEffectEnabled(Qt.UI_AnimateCombo, False)
        try:
            super().showPopup()
        finally:
            QApplication.setEffectEnabled(Qt.UI_AnimateCombo, animate)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Type.Show and watched is self.view().parentWidget():
            self._keep_in_work_area(watched)
        return super().eventFilter(watched, event)

    def _keep_in_work_area(self, container: QWidget) -> None:
        """Move the list above the picker when it passes the bottom of the
        work area: its bottom on the top of Qt's list anchor, as Qt opens it
        upward itself."""
        rect = container.geometry()
        work = self.screen().availableGeometry()
        if rect.bottom() <= work.bottom():
            return
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        anchor = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, opt, QStyle.SubControl.SC_ComboBoxListBoxPopup, self)
        rect.moveBottom(self.mapToGlobal(anchor.topLeft()).y())
        rect.moveTop(max(rect.top(), work.top()))
        container.move(rect.topLeft())

    def hidePopup(self) -> None:  # noqa: N802 (Qt override)
        # QComboBox also calls hidePopup from its own hideEvent, so every hide
        # of the panel lands here: report only a list that was open.
        was_open = self.view().isVisible()
        super().hidePopup()
        if was_open:
            self.popupClosed.emit()


class TrayFlyout(QWidget):
    """The tray flyout panel: signals are intents for the tray, setters show
    the state the tray hands it (plan phase 4, p4_iface_B.md)."""

    pauseRequested = Signal(bool)              # True: pause, False: resume
    subtitlesToggleRequested = Signal()
    modeChosen = Signal(str)                   # "watch" | "learn"
    genreChosen = Signal(str)                  # a GENRE_OPTIONS key
    alignmentChosen = Signal(str)              # "left" | "center"
    changeSeriesRequested = Signal()
    clearSeriesRequested = Signal()
    positionPresetChosen = Signal(str, float)  # overlay.position screen, v_offset_pct
    moveSubtitlesRequested = Signal()
    settingsRequested = Signal()
    openLogFolderRequested = Signal()
    saveDiagnosticsRequested = Signal()
    quitRequested = Signal()
    dismissed = Signal()                       # every hide

    def __init__(self, palette: dict, *, show_series: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._palette = palette
        self._paused = False
        self._shown = True
        self._mode_value = "watch"
        self._problem: str | None = None
        self._has_series = False
        self._labels: dict[str, str] = {}
        self._entries: list[PositionEntry] = []
        self.setObjectName("trayFlyout")
        self.setAccessibleName("Yamabiko")
        self.setFont(ui_font(13))
        self.setFixedSize(FLYOUT_WIDTH, FLYOUT_HEIGHT if show_series else FLYOUT_HEIGHT_NO_SERIES)

        self._header = _Header(palette, self)
        self._status = self._header.status
        self._pause = self._header.button
        self._pause.clicked.connect(
            lambda _checked=False: self.pauseRequested.emit(not self._paused))

        top = QWidget(self)
        top.setFixedHeight(_TOP_H if show_series else _TOP_H_NO_SERIES)
        top_box = QVBoxLayout(top)
        top_box.setContentsMargins(16, 12, 16, 12)
        top_box.setSpacing(12)
        self._subtitles = _Tile("Subtitles", palette, top)
        self._subtitles.setAccessibleName("Subtitles")
        self._subtitles.clicked.connect(
            lambda _checked=False: self.subtitlesToggleRequested.emit())
        self._mode = _Tile("Mode", palette, top, trailing=_G_SWITCH)
        self._mode.setAccessibleName("Learn mode")
        self._mode.clicked.connect(lambda _checked=False: self._on_mode_clicked())
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        tiles.addWidget(self._subtitles)
        tiles.addWidget(self._mode)
        top_box.addLayout(tiles)
        self._genre = SegmentedControl(
            [SegmentOption(key, label) for key, label in GENRE_OPTIONS], palette, top,
            value="variety")
        self._genre.valueChanged.connect(self.genreChosen)
        top_box.addLayout(self._row("Genre", self._genre))
        self._series_row: _SeriesRow | None = None
        self._series_text = self._cover = self._change = self._clear = None
        if show_series:
            self._series_row = _SeriesRow(palette, top)
            self._series_text = self._series_row.text
            self._cover = self._series_row.cover
            self._change = self._series_row.change
            self._clear = self._series_row.clear
            self._change.clicked.connect(
                lambda _checked=False: self._act(self.changeSeriesRequested))
            self._clear.clicked.connect(
                lambda _checked=False: self._act(self.clearSeriesRequested))
            top_box.addWidget(self._series_row)

        self._divider = _Rule(palette, self)

        bottom = QWidget(self)
        bottom.setFixedHeight(_BOTTOM_H)
        bottom_box = QVBoxLayout(bottom)
        bottom_box.setContentsMargins(16, 12, 16, 14)
        bottom_box.setSpacing(12)
        self._position = _PositionPicker(palette, bottom)
        self._position.setFixedHeight(_ROW_H)
        self._position.setAccessibleName("Position")
        self._position.activated.connect(self._on_position_activated)
        self._position.popupClosed.connect(self._after_popup)
        bottom_box.addLayout(self._row("Position", self._position))
        self._alignment = SegmentedControl(
            [SegmentOption(value, label) for value, label in _ALIGNMENT_OPTIONS], palette,
            bottom, value="left")
        self._alignment.setFixedWidth(_ALIGN_W)
        self._alignment.valueChanged.connect(self.alignmentChosen)
        self._move = QPushButton("Move subtitles…", bottom)
        self._move.setFixedHeight(_ROW_H)
        self._move.clicked.connect(
            lambda _checked=False: self._act(self.moveSubtitlesRequested))
        alignment_row = QHBoxLayout()
        alignment_row.setContentsMargins(0, 0, 0, 0)
        alignment_row.setSpacing(8)
        alignment_row.addWidget(self._alignment)
        alignment_row.addWidget(self._move, 1)
        bottom_box.addLayout(self._row("Alignment", alignment_row))

        self._footer_rule = _Rule(palette, self)
        footer = QWidget(self)
        footer.setFixedHeight(_FOOTER_H)
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(8, 7, 8, 7)
        footer_row.setSpacing(2)
        self._settings = _FooterButton(_G_SETTINGS, "Settings", palette, footer)
        self._troubleshooting = _FooterButton(
            _G_REPAIR, "Troubleshooting", palette, footer, chevron=True)
        self._quit = _FooterButton(_G_POWER, "Quit", palette, footer)
        self._settings.clicked.connect(lambda _checked=False: self._act(self.settingsRequested))
        self._troubleshooting.clicked.connect(lambda _checked=False: self._open_troubleshooting())
        self._quit.clicked.connect(lambda _checked=False: self._act(self.quitRequested))
        footer_row.addWidget(self._settings)
        footer_row.addWidget(self._troubleshooting)
        footer_row.addStretch(1)
        footer_row.addWidget(self._quit)

        self.troubleshooting_menu = QMenu(self)
        self.troubleshooting_menu.setMinimumWidth(200)
        SquareCorners(self.troubleshooting_menu)  # nearly square, as screenshots 05 and 06
        self.troubleshooting_menu.addAction("Open log folder").triggered.connect(
            lambda _checked=False: self._act(self.openLogFolderRequested))
        self.troubleshooting_menu.addAction("Save diagnostics").triggered.connect(
            lambda _checked=False: self._act(self.saveDiagnosticsRequested))
        self.troubleshooting_menu.aboutToShow.connect(
            lambda: self._troubleshooting.set_open(True))
        self.troubleshooting_menu.aboutToHide.connect(self._on_menu_hidden)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        for part in (self._header, top, self._divider, bottom, self._footer_rule, footer):
            root.addWidget(part)

        self.apply_palette(palette)
        self.set_subtitles_shown(True)
        self.set_mode("watch")
        self.set_series(None)
        self._sync_tooltips()
        # The native window exists before the first open: the first show is
        # fast (PlatformProbe Q3) and the tray's WM_SETTINGCHANGE filter has
        # a top-level window to hear it (plan Decision 14).
        self.winId()

    # -- setters (none emits an intent) ----------------------------------------

    def set_paused(self, paused: bool) -> None:
        """The Pause / Resume face and the header status."""
        self._paused = bool(paused)
        self._pause.set_paused(self._paused)
        self._sync_tooltips()
        self._sync_status()

    def set_subtitles_shown(self, shown: bool) -> None:
        """The Subtitles tile ("Shown", accent-filled, or "Hidden") and the
        header status; the truth is the overlay window's visibility."""
        self._shown = bool(shown)
        self._subtitles.set_state(_G_SUBTITLES, "Shown" if self._shown else "Hidden",
                                  checked=self._shown, lit=self._shown)
        self._sync_status()

    def set_mode(self, mode: str) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self._mode_value = mode
        learn = mode == "learn"
        self._mode.set_state(_G_LEARN if learn else _G_WATCH, "Learn" if learn else "Watch",
                             checked=learn, lit=False)

    def set_genre(self, genre: str) -> None:
        self._genre.setValue(genre)  # ValueError for an unknown key

    def set_alignment(self, alignment: str) -> None:
        self._alignment.setValue(alignment)  # ValueError for anything but left / center

    def set_series(self, label: str | None, native: str = "",
                   cover: "QPixmap | QColor | None" = None) -> None:
        """The series row. ``label`` is the state frame's "Title (year)"
        (shown without the year), ``native`` the Japanese title, ``cover``
        SeriesPalette.cover_for(); None shows "No series" with no cover and
        Clear disabled."""
        if self._series_row is None:
            return
        self._has_series = label is not None
        if label is None:
            self._series_text.set_text("No series", "", muted=True)
        else:
            self._series_text.set_text(series_title(label), native or "", muted=False)
        self._clear.setEnabled(self._has_series)
        self.set_cover(cover)

    def set_cover(self, cover: "QPixmap | QColor | None") -> None:
        """Replace the cover (SeriesPalette.coverChanged); none without a
        series."""
        if self._series_row is None:
            return
        shown = cover if self._has_series else None
        self._cover.set_cover(shown)
        self._series_row.show_cover(shown is not None)

    def set_problem(self, title: str | None) -> None:
        """The first active problem's title, or None."""
        self._problem = title or None
        self._sync_status()

    def set_positions(self, entries: Sequence[PositionEntry], current: int) -> None:
        """The Position picker rows (position_entries) and the current one."""
        self._entries = list(entries)
        self._position.clear()
        for entry in self._entries:
            self._position.addItem(entry.label)
        if 0 <= current < len(self._entries):
            self._position.setCurrentIndex(current)

    def set_shortcut_labels(self, labels: Mapping[str, str]) -> None:
        """The user's combos by hotkeys key (pause_resume, toggle_overlay,
        show_series); an empty or missing one drops the parenthetical."""
        self._labels = {key: value for key, value in labels.items()
                        if isinstance(value, str) and value}
        self._sync_tooltips()

    def apply_palette(self, palette: dict) -> None:
        """Re-skin the panel live for a new theme palette."""
        self._palette = palette
        apply_window_qss(self, palette)
        qpalette = self.palette()
        # The windows11 style marks the selected row of the picker's list with
        # the Accent/Highlight role: make it the theme's accent (Lit).
        for role in (QPalette.ColorRole.Highlight, QPalette.ColorRole.Accent):
            qpalette.setColor(role, QColor(palette["accent"]))
        qpalette.setColor(QPalette.ColorRole.HighlightedText, QColor(palette["on_accent"]))
        self.setPalette(qpalette)
        self.troubleshooting_menu.setStyleSheet(menu_qss(palette))
        for part in self._themed():
            part.apply_palette(palette)
        if self.isVisible():
            set_border_color(self, palette["frame"])
        self.update()

    def open_at(self, top_left: QPoint) -> None:
        """Show the panel with its top-left at ``top_left`` (logical px,
        tray_placement.place_flyout), take activation, then apply the DWM
        chrome (it has no effect before the window is shown)."""
        self.move(top_left)
        self.show()
        self.raise_()
        self.activateWindow()
        set_corner_preference(self, DWMWCP_ROUNDSMALL)
        set_border_color(self, self._palette["frame"])

    # -- internals ---------------------------------------------------------------

    def _row(self, text: str, field: "QWidget | QLayout") -> QHBoxLayout:
        """A body row: the 72 px label column, then ``field``."""
        label = QLabel(text)
        label.setFixedWidth(_LABEL_W)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(label)
        if isinstance(field, QLayout):
            row.addLayout(field, 1)
        else:
            row.addWidget(field, 1)
        return row

    def _themed(self) -> list:
        parts = [self._header, self._subtitles, self._mode, self._genre, self._divider,
                 self._position, self._alignment, self._footer_rule, self._settings,
                 self._troubleshooting, self._quit]
        if self._series_row is not None:
            parts.append(self._series_row)
        return parts

    def _sync_status(self) -> None:
        """Decision 26: problem > Paused > Subtitles hidden > Listening."""
        if self._problem:
            self._status.set_status("problem", self._problem)
        elif self._paused:
            self._status.set_status("paused", "Paused")
        elif not self._shown:
            self._status.set_status("hidden", "Subtitles hidden")
        else:
            self._status.set_status("listening", "Listening")

    def _sync_tooltips(self) -> None:
        """Decision 13: the user's live combos, no parenthetical when unset."""
        self._pause.setToolTip(_with_combo("Resume" if self._paused else "Pause",
                                           self._labels.get("pause_resume")))
        self._subtitles.setToolTip(_with_combo("Show or hide subtitles",
                                               self._labels.get("toggle_overlay")))
        if self._change is not None:
            self._change.setToolTip(_with_combo("Pick a series",
                                                self._labels.get("show_series")))

    def _act(self, signal, *args) -> None:
        """A closing action: hide first (dismissed fires while the cursor is
        over the panel, not the tray icon), then report the intent."""
        self.hide()
        signal.emit(*args)

    def _on_mode_clicked(self) -> None:
        mode = "learn" if self._mode_value == "watch" else "watch"
        self.set_mode(mode)
        self.modeChosen.emit(mode)

    def _on_position_activated(self, index: int) -> None:
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries[index]
        if entry.screen is None or entry.v_offset_pct is None:
            return  # the current-value row
        self._act(self.positionPresetChosen, entry.screen, float(entry.v_offset_pct))

    def _open_troubleshooting(self) -> None:
        """The small menu above its button: left edges aligned, its bottom
        3 px above the button (screenshot 05)."""
        menu = self.troubleshooting_menu
        anchor = self._troubleshooting.mapToGlobal(QPoint(0, -3))
        menu.popup(QPoint(anchor.x(), anchor.y() - menu.sizeHint().height()))

    def _on_menu_hidden(self) -> None:
        self._troubleshooting.set_open(False)
        self._after_popup()

    def _after_popup(self) -> None:
        # Decision 54: a deactivation while the popup was open did not hide
        # the panel; settle it once the popup is gone.
        QTimer.singleShot(0, self._hide_if_inactive)

    def _hide_if_inactive(self) -> None:
        if (self.isVisible() and not self.isActiveWindow()
                and QApplication.activePopupWidget() is None):
            self.hide()

    # -- Qt overrides --------------------------------------------------------------

    def event(self, event) -> bool:  # noqa: D401 (Qt override)
        handled = super().event(event)
        if (event.type() == QEvent.Type.WindowDeactivate
                and QApplication.activePopupWidget() is None):
            self.hide()
        return handled

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() == Qt.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self.dismissed.emit()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._palette["raised"]))
        p.end()
