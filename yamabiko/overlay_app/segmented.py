"""Lit segmented control: a set of options where the chosen one fills with
the accent (the approved "Lit" selection style).

Five looks share one control:

* ``group`` (default): options side by side in a framed group, 32 px tall,
  44 px when an option carries a muted second line ("Watch" over "English
  priority"). The Overlay switcher rows.
* ``list``: options stacked as 36 px rows in the framed group, the label at
  the left and the second line at the right. The translation model list;
  setup's sound devices.
* ``text``: bare text options, 24 px tall, no frame. Theme in the title band.
* ``tiles``: one row of equal tiles in a 60 px group, the label over a BIZ
  UDPMincho second line. Setup's Content step.
* ``cards``: stacked 62 px cards, a semibold label with an optional note at
  the right over a muted description. Setup's Quality step.

``wizard`` selects setup's look (opt-wiz-rail.tpl): the group on the field
tone with 2 px corners and a 13.5 px list text inset. ``framed=False`` leaves
the edge and the tone to a host that frames the group (setup's device list:
its scroll area, so the frame stays put while the rows scroll); the 2 px rim
stays. Arrow keys move the choice; the segment focused from the keyboard
shows an inset ring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

_PAD = 2            # group padding and the gap between segments
_HEIGHT = 32        # one-line group
_TALL_HEIGHT = 44   # group whose options carry a sublabel
_ROW_HEIGHT = 36    # one row of the list variant
_TEXT_HEIGHT = 24   # text variant
_TILES_HEIGHT = 60  # tiles variant group
_CARD_HEIGHT = 62   # one card of the cards variant
_ROW_INSET = 12     # list row label/sublabel inset
_WIZARD_ROW_INSET = 13.5  # setup's list rows: 12 px padding + the 1.5 px border
_CARD_INSET = 15.5  # cards: 14 px padding + the 1.5 px border
_TEXT_PAD_X = 8     # text variant side padding
_GROUP_RADIUS = 4.0
_SEGMENT_RADIUS = 3.0
_WIZARD_RADIUS = 2.0
_SUBLABEL_PX = 13
_TILE_LABEL_H, _TILE_SUBLABEL_H, _TILE_GAP = 19, 17, 1
_CARD_LABEL_H, _CARD_TEXT_H, _CARD_GAP = 19, 17, 3
ACTIVATION_KEYS = (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter)
# Window activation, popups and programmatic moves (a page switch) hand focus
# over without the user choosing it: they keep whatever ring state the widget
# had (like CSS :focus-visible).
_NEUTRAL_FOCUS = (Qt.ActiveWindowFocusReason, Qt.PopupFocusReason, Qt.OtherFocusReason)
# A key a shortcut takes reaches the application only as the ShortcutOverride
# Qt sends ahead of every key press's shortcut lookup, never as a KeyPress.
_KEY_PRESSES = (QEvent.KeyPress, QEvent.ShortcutOverride)
_POINTER_PRESSES = (QEvent.MouseButtonPress, QEvent.TouchBegin)
VARIANTS = ("group", "list", "text", "tiles", "cards")
_STACKED = ("list", "cards")


def _unknown_value(value: str) -> ValueError:
    return ValueError(f"unknown segment value {value!r}")


@dataclass(frozen=True)
class SegmentOption:
    value: str
    label: str
    sublabel: str = ""
    note: str = ""  # cards: the muted text at the right of the label row


class _LastInput(QObject):
    """The application's event filter that records whether its latest press
    was a key (``key``) or a mouse button or touch. KeyboardRingMixin
    installs it once, when the first widget using it is built."""

    def __init__(self, app: QCoreApplication) -> None:
        super().__init__(app)  # lives as long as the application
        self.key = False
        app.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        kind = event.type()
        if kind in _KEY_PRESSES:
            self.key = True
        elif kind in _POINTER_PRESSES:
            self.key = False
        return False


_last_input: _LastInput | None = None


class KeyboardRingMixin:
    """The keyboard-only focus ring (like CSS :focus-visible) for a QWidget
    subclass; list it before the Qt base class. A focus move that follows a
    key press lights the ring (Tab, Shift+Tab, an arrow, a shortcut, and the
    hand-over when Space or Enter turns the focused widget off), one that
    follows a mouse press or a touch clears it, neutral hand-overs keep it,
    and the activation keys light it too. Paint the ring while
    ``ring_visible()``."""

    _ring = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        global _last_input
        if _last_input is None:
            _last_input = _LastInput(QCoreApplication.instance())

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.reason() not in _NEUTRAL_FOCUS:
            # The reason alone can't tell: Qt gives TabFocusReason to the
            # widget it hands the focus on to when the focused one is
            # disabled or hidden, after a click as after a key. The latest
            # press decides.
            self._ring = event.reason() != Qt.MouseFocusReason and _last_input.key
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.reason() not in _NEUTRAL_FOCUS:
            self._ring = False
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() in ACTIVATION_KEYS:
            self._ring = True  # keyboard use, like the arrow keys
            self.update()
        super().keyPressEvent(event)

    def ring_visible(self) -> bool:
        return self._ring and self.hasFocus() and self.isEnabled()

    def hand_ring_to(self, target: "KeyboardRingMixin") -> None:
        """An arrow key moved the choice to ``target``: the ring and the focus
        go with it (arrow keys are keyboard use)."""
        self._ring = False
        self.update()
        target._ring = True
        target.setFocus(Qt.OtherFocusReason)
        target.update()


class _Segment(KeyboardRingMixin, QAbstractButton):
    def __init__(self, option: SegmentOption, owner: "SegmentedControl") -> None:
        super().__init__(owner)
        self.option = option
        self.note = option.note
        self._owner = owner
        self.setCheckable(True)
        self.setAttribute(Qt.WA_Hover, True)
        self._name()

    def _name(self) -> None:
        parts = (self.option.label, self.option.sublabel, self.note)
        self.setAccessibleName(", ".join(part for part in parts if part))

    def _sublabel_font(self) -> QFont:
        if self._owner._variant == "tiles":
            from .controls import mincho_font  # controls imports this module

            return mincho_font(_SUBLABEL_PX)
        font = QFont(self.font())
        font.setPixelSize(_SUBLABEL_PX)
        if self._owner._variant == "cards":
            font.setWeight(QFont.Weight.Normal)
        return font

    def _label_font(self) -> QFont:
        font = QFont(self.font())
        if self._owner._variant == "cards":
            font.setWeight(QFont.Weight.DemiBold)
        return font

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        variant = self._owner._variant
        width = QFontMetrics(self._label_font()).horizontalAdvance(self.option.label)
        if variant == "text":
            return QSize(width + 2 * _TEXT_PAD_X, _TEXT_HEIGHT)
        sub = 0
        if self.option.sublabel:
            sub = QFontMetrics(self._sublabel_font()).horizontalAdvance(self.option.sublabel)
        if variant == "list":
            inset = self._owner._row_inset
            return QSize(math.ceil(width + sub + 2 * inset + (inset if sub else 0)), _ROW_HEIGHT)
        if variant == "cards":
            return QSize(max(width, sub) + 2 * round(_CARD_INSET), _CARD_HEIGHT)
        outer = _TILES_HEIGHT if variant == "tiles" else (_TALL_HEIGHT if self.option.sublabel else _HEIGHT)
        return QSize(max(width, sub) + 24, outer - 2 * (_PAD + 1))

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() in (Qt.Key_Left, Qt.Key_Up):
            self._owner._step(self, -1)
        elif event.key() in (Qt.Key_Right, Qt.Key_Down):
            self._owner._step(self, +1)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = self._owner._palette
        variant = self._owner._variant
        radius = self._owner._segment_radius
        enabled = self.isEnabled()
        checked = self.isChecked()
        hover = enabled and self.underMouse()
        lit = checked and enabled
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        body = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        if lit:
            painter.setPen(QPen(QColor(p["accent"]), 1.5))
            painter.setBrush(QColor(p["accent"]))
            painter.drawRoundedRect(body, radius, radius)
        elif checked:  # disabled: the choice stays readable without the accent
            painter.setPen(QPen(QColor(p["edge"]), 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        elif hover and variant != "text":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(p["hover"]))
            painter.drawRoundedRect(body, radius, radius)
        if self.ring_visible():
            ring = QColor(p["on_accent"] if checked else p["accent"])
            painter.setPen(QPen(ring, 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(body.adjusted(1.5, 1.5, -1.5, -1.5), 2.0, 2.0)

        if lit:
            label_color, sublabel_color = p["on_accent"], p["on_accent"]
        elif not enabled:
            label_color, sublabel_color = p["muted"], p["muted"]
        elif variant == "text":
            label_color = sublabel_color = p["ink"] if hover else p["muted"]
        else:
            label_color, sublabel_color = p["ink"], p["muted"]
        painter.setPen(QColor(label_color))
        rect = QRectF(self.rect())
        if variant == "list":
            row = rect.adjusted(self._owner._row_inset, 0, -self._owner._row_inset, 0)
            painter.drawText(row, Qt.AlignLeft | Qt.AlignVCenter, self.option.label)
            if self.option.sublabel:
                painter.setFont(self._sublabel_font())
                painter.setPen(QColor(sublabel_color))
                painter.drawText(row, Qt.AlignRight | Qt.AlignVCenter, self.option.sublabel)
        elif variant == "tiles":
            top = (rect.height() - (_TILE_LABEL_H + _TILE_GAP + _TILE_SUBLABEL_H)) / 2.0
            painter.drawText(QRectF(0, top, rect.width(), _TILE_LABEL_H), Qt.AlignCenter,
                             self.option.label)
            painter.setFont(self._sublabel_font())
            painter.setPen(QColor(sublabel_color))
            painter.drawText(QRectF(0, top + _TILE_LABEL_H + _TILE_GAP, rect.width(), _TILE_SUBLABEL_H),
                             Qt.AlignCenter, self.option.sublabel)
        elif variant == "cards":
            self._paint_card(painter, rect, sublabel_color)
        elif self.option.sublabel:
            half = rect.height() / 2.0
            painter.drawText(rect.adjusted(0, 0, 0, -half), Qt.AlignHCenter | Qt.AlignBottom,
                             self.option.label)
            painter.setFont(self._sublabel_font())
            painter.setPen(QColor(sublabel_color))
            painter.drawText(rect.adjusted(0, half, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                             self.option.sublabel)
        else:
            painter.drawText(rect, Qt.AlignCenter, self.option.label)
        painter.end()

    def _paint_card(self, painter: QPainter, rect: QRectF, muted: str) -> None:
        """Label (semibold) with the note at the right on its baseline, the
        description under it; both rows centered in the card as a column."""
        top = (rect.height() - (_CARD_LABEL_H + _CARD_GAP + _CARD_TEXT_H)) / 2.0
        text = rect.adjusted(_CARD_INSET, 0, -_CARD_INSET, 0)
        label_font = self._label_font()
        metrics = QFontMetricsF(label_font)
        baseline = top + (_CARD_LABEL_H - (metrics.ascent() + metrics.descent())) / 2.0 + metrics.ascent()
        painter.setFont(label_font)
        painter.drawText(QPointF(text.left(), baseline), self.option.label)
        small = self._sublabel_font()
        painter.setFont(small)
        painter.setPen(QColor(muted))
        if self.note:
            note_w = QFontMetricsF(small).horizontalAdvance(self.note)
            painter.drawText(QPointF(text.right() - note_w, baseline), self.note)
        painter.drawText(QRectF(text.left(), top + _CARD_LABEL_H + _CARD_GAP, text.width(), _CARD_TEXT_H),
                         Qt.AlignLeft | Qt.AlignVCenter, self.option.sublabel)


class SegmentedControl(QWidget):
    """A one-of-few choice; the chosen segment fills with the accent."""

    valueChanged = Signal(str)

    def __init__(
        self,
        options: Sequence[SegmentOption],
        palette: dict,
        parent: QWidget | None = None,
        *,
        value: str | None = None,
        variant: str = "group",
        wizard: bool = False,
        framed: bool = True,
    ) -> None:
        if not options:
            raise ValueError("SegmentedControl needs at least one option")
        if value is not None and all(option.value != value for option in options):
            raise _unknown_value(value)
        if variant not in VARIANTS:
            raise ValueError(f"unknown segmented variant {variant!r}")
        super().__init__(parent)
        self._palette = palette
        self._variant = variant
        self._wizard = wizard
        self._framed = framed
        self._group_radius = _WIZARD_RADIUS if wizard else _GROUP_RADIUS
        self._segment_radius = _WIZARD_RADIUS if wizard else _SEGMENT_RADIUS
        self._row_inset = _WIZARD_ROW_INSET if wizard else _ROW_INSET
        self._segments: list[_Segment] = []
        layout = QVBoxLayout(self) if variant in _STACKED else QHBoxLayout(self)
        # The rim, plus the 1 px edge when the group draws its own frame.
        margin = 0 if variant == "text" else _PAD + (1 if framed else 0)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(_PAD)
        for option in options:
            segment = _Segment(option, self)
            segment.clicked.connect(lambda _checked=False, s=segment: self._choose(s))
            layout.addWidget(segment, 0 if variant == "text" else 1)
            self._segments.append(segment)
        count = len(options)
        if variant in _STACKED:
            row = _ROW_HEIGHT if variant == "list" else _CARD_HEIGHT
            height = count * row + (count - 1) * _PAD + 2 * margin
        elif variant == "text":
            height = _TEXT_HEIGHT
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        elif variant == "tiles":
            height = _TILES_HEIGHT
        else:
            height = _TALL_HEIGHT if any(o.sublabel for o in options) else _HEIGHT
        self.setFixedHeight(height)
        self._current = self._segments[0] if value is None else self._segment_for(value)
        self._sync_checks()

    def value(self) -> str:
        return self._current.option.value

    def setValue(self, value: str) -> None:  # noqa: N802 (Qt naming)
        segment = self._segment_for(value)
        if segment is None:
            raise _unknown_value(value)
        self._current = segment
        self._sync_checks()

    def segment(self, value: str) -> QAbstractButton:
        """The segment of ``value`` (an objectName for UI Automation, tests)."""
        segment = self._segment_for(value)
        if segment is None:
            raise _unknown_value(value)
        return segment

    def set_note(self, value: str, note: str) -> None:
        """Show ``note`` at the right of ``value``'s label row (cards; "" clears
        it); the segment's accessible name follows."""
        segment = self.segment(value)
        segment.note = note
        segment._name()
        segment.update()

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()
        for segment in self._segments:
            segment.update()

    def _segment_for(self, value: str | None) -> _Segment | None:
        return next((s for s in self._segments if s.option.value == value), None)

    def _sync_checks(self) -> None:
        for segment in self._segments:
            chosen = segment is self._current
            segment.setChecked(chosen)
            # One Tab stop per control: Tab lands on the choice, arrows move it.
            segment.setFocusPolicy(Qt.StrongFocus if chosen else Qt.ClickFocus)

    def _choose(self, segment: _Segment) -> None:
        changed = segment is not self._current
        self._current = segment
        self._sync_checks()  # also re-checks a segment its own click toggled off
        if changed:
            self.valueChanged.emit(segment.option.value)

    def _step(self, segment: _Segment, delta: int) -> None:
        index = self._segments.index(segment) + delta
        target = self._segments[max(0, min(len(self._segments) - 1, index))]
        self._choose(target)
        segment.hand_ring_to(target)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._variant == "text" or not self._framed:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(self._palette["edge"]), 1.0))
        painter.setBrush(QColor(self._palette["field" if self._wizard else "ground"]))
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), self._group_radius, self._group_radius
        )
        painter.end()
