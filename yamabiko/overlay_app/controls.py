"""Small painted controls for the Settings window and its dialogs.

The switch, text buttons, BIZ UDPMincho text tabs, raised panels, the color
field, fields with drawn chevrons and the thin progress line of the approved
mockups ("Studio under the mountain"). Every painted control takes the theme
palette and re-skins through ``apply_palette``; focus rings show for keyboard
focus only (like CSS :focus-visible).
"""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractButton, QComboBox, QDoubleSpinBox, QFontComboBox, QFrame,
    QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from yamabiko.overlay_app.segmented import KeyboardRingMixin

MINCHO = "BIZ UDPMincho"
FIELD_H = 30


def mincho_font(px: int, *, bold: bool = False) -> QFont:
    """BIZ UDPMincho (bundled) at ``px``; Yu Mincho where it isn't registered."""
    font = QFont()
    font.setFamilies([MINCHO, "Yu Mincho"])
    font.setPixelSize(px)
    font.setWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
    return font


def ui_font(px: int = 13) -> QFont:
    """The chrome's label font: Segoe UI Variable Text."""
    font = QFont()
    font.setFamilies(["Segoe UI Variable Text", "Segoe UI"])
    font.setPixelSize(px)
    return font


class _PaintedButton(KeyboardRingMixin, QAbstractButton):
    """Base for the painted buttons: palette, hover, keyboard-only ring."""

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


class _ClickOnEnter:
    """Return/Enter click the focused button and stop there, as a
    QPushButton's autoDefault does, so the key never reaches a window's or
    dialog's default-button handling. Space keeps QAbstractButton's own
    press/release click; the ring lights through KeyboardRingMixin. List it
    before the button base."""

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().keyPressEvent(event)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            self.click()


class Switch(_PaintedButton):
    """The on/off switch: a 40 x 20 track with a 12 px knob, the label right
    of it; the whole row toggles."""

    _TRACK_W = 40
    _TRACK_H = 20
    _KNOB = 12
    _GAP = 10

    def __init__(self, text: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(palette, parent)
        self.setText(text)
        self.setCheckable(True)
        self.setAccessibleName(text)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        text_w = self.fontMetrics().horizontalAdvance(self.text())
        return QSize(self._TRACK_W + self._GAP + text_w + 2, FIELD_H)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        on, enabled = self.isChecked(), self.isEnabled()
        top = (self.height() - self._TRACK_H) / 2.0
        track = QRectF(0.5, top + 0.5, self._TRACK_W - 1.0, self._TRACK_H - 1.0)
        radius = track.height() / 2.0
        if on and enabled:
            p.setPen(QPen(self._color("accent"), 1.0))
            p.setBrush(self._color("accent"))
        elif enabled:
            p.setPen(QPen(self._color("muted"), 1.0))
            p.setBrush(self._color("field"))
        else:
            p.setPen(QPen(self._color("edge"), 1.0))
            p.setBrush(self._color("ground"))
        p.drawRoundedRect(track, radius, radius)
        knob_x = 24.0 if on else 4.0
        p.setPen(Qt.NoPen)
        p.setBrush(self._color("on_accent" if on and enabled else "muted"))
        p.drawEllipse(QRectF(knob_x, top + 4.0, self._KNOB, self._KNOB))
        if self.ring_visible():
            p.setPen(QPen(self._color("accent"), 1.5))
            p.setBrush(Qt.NoBrush)
            ring = track.adjusted(-2.5, -2.5, 2.5, 2.5)
            p.drawRoundedRect(ring, ring.height() / 2.0, ring.height() / 2.0)
        p.setPen(self._color("ink" if enabled else "muted"))
        text_x = self._TRACK_W + self._GAP
        p.drawText(QRectF(text_x, 0, self.width() - text_x, self.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, self.text())
        p.end()


class LinkButton(_ClickOnEnter, _PaintedButton):
    """A text button. Underlined by default ("Reset to defaults", "Copy
    details"); with ``underline=False`` it is padded and shows the hover tone
    behind its text. ``quiet`` keeps the text muted until hover. The
    per-shortcut "Reset" is ``quiet=True, underline=False``."""

    def __init__(self, text: str, palette: dict, parent: QWidget | None = None, *,
                 quiet: bool = False, underline: bool = True, height: int = FIELD_H) -> None:
        super().__init__(palette, parent)
        self.setText(text)
        self.setAccessibleName(text)
        self._quiet = quiet
        self._underline = underline
        self._height = height
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        pad = 0 if self._underline else 12
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + pad + 2, self._height)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        hover = self.isEnabled() and self.underMouse()
        if not self._underline and hover:
            p.setPen(Qt.NoPen)
            p.setBrush(self._color("hover"))
            p.drawRoundedRect(QRectF(self.rect()), 3.0, 3.0)
        if not self.isEnabled():
            color = self._color("muted")
        elif self._quiet:
            color = self._color("ink" if hover else "muted")
        else:
            color = self._color("ink")
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text())
        x0 = 0.0 if self._underline else (self.width() - text_w) / 2.0
        top = (self.height() - fm.height()) / 2.0
        baseline = top + fm.ascent()
        p.setPen(color)
        p.drawText(QPointF(x0, baseline), self.text())
        if self._underline:
            line = self._color("ink" if hover else "muted")
            p.setPen(QPen(line, 1.0))
            y = round(baseline + 3) + 0.5
            p.drawLine(QPointF(x0, y), QPointF(x0 + text_w, y))
        if self.ring_visible():
            p.setPen(QPen(self._color("accent"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75), 3.0, 3.0)
        p.end()


class _Tab(_PaintedButton):
    def __init__(self, en: str, jp: str, owner: "TextTabs") -> None:
        super().__init__(owner._palette, owner)
        self._en = en
        self._jp = jp
        self._owner = owner
        self.setCheckable(True)
        self.setFont(mincho_font(15))
        self.setAccessibleName(f"{en} {jp}")

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self._en) + 6 + fm.horizontalAdvance(self._jp), 40)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() in (Qt.Key_Left, Qt.Key_Up):
            self._owner._step(self, -1)
        elif event.key() in (Qt.Key_Right, Qt.Key_Down):
            self._owner._step(self, +1)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(self.font())
        current = self.isChecked()
        hover = self.isEnabled() and self.underMouse()
        p.setPen(self._color("ink" if current or hover else "muted"))
        rect = QRectF(self.rect())
        p.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, self._en)
        jp_x = self.fontMetrics().horizontalAdvance(self._en) + 6
        p.drawText(rect.adjusted(jp_x, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, self._jp)
        if current:
            p.fillRect(QRectF(0, self.height() - 2, self.width(), 2), self._color("accent"))
        if self.ring_visible():
            p.setPen(QPen(self._color("accent"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(0.75, 8, -0.75, -8), 3.0, 3.0)
        p.end()


class TextTabs(QWidget):
    """BIZ UDPMincho text tabs ("Subtitles 字幕"); the current one is ink with
    a 2 px accent bar over the row's bottom rule. One Tab stop; arrow keys
    move between tabs."""

    currentChanged = Signal(int)

    def __init__(self, tabs: Sequence[tuple[str, str]], palette: dict,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._tabs: list[_Tab] = []
        self._current = 0
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(28)
        for en, jp in tabs:
            tab = _Tab(en, jp, self)
            tab.clicked.connect(lambda _checked=False, t=tab: self.setCurrentIndex(self._tabs.index(t)))
            row.addWidget(tab)
            self._tabs.append(tab)
        row.addStretch(1)
        self.setFixedHeight(40)
        self._sync()

    def count(self) -> int:
        return len(self._tabs)

    def currentIndex(self) -> int:  # noqa: N802 (Qt naming)
        return self._current

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 (Qt naming)
        index = max(0, min(len(self._tabs) - 1, index))
        changed = index != self._current
        self._current = index
        self._sync()
        if changed:
            self.currentChanged.emit(index)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        for tab in self._tabs:
            tab.apply_palette(palette)
        self.update()

    def _step(self, tab: _Tab, delta: int) -> None:
        self.setCurrentIndex(self._tabs.index(tab) + delta)
        tab.hand_ring_to(self._tabs[self._current])

    def _sync(self) -> None:
        for index, tab in enumerate(self._tabs):
            current = index == self._current
            tab.setChecked(current)
            tab.setFocusPolicy(Qt.StrongFocus if current else Qt.ClickFocus)
            tab.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(0, self.height() - 1, self.width(), 1, QColor(self._palette["line"]))
        p.end()


class Panel(QFrame):
    """A raised panel: BIZ UDPMincho title (a muted note beside it) above its
    rows. Add rows to ``rows``; ``finish()`` keeps them at the top."""

    def __init__(self, title: str, note: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.rows = QVBoxLayout(self)
        self.rows.setContentsMargins(14, 11, 14, 14)
        self.rows.setSpacing(11)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self.title = QLabel(title)
        self.title.setObjectName("panelTitle")
        self.title.setFont(mincho_font(15))
        # The mockup's .pnl h3 is 20 px tall (BIZ UDPMincho 15 px measures 15).
        self.title.setFixedHeight(20)
        head.addWidget(self.title, 0, Qt.AlignBottom)
        self.note: QLabel | None = None
        if note:
            self.note = QLabel(note)
            self.note.setObjectName("panelNote")
            head.addWidget(self.note, 0, Qt.AlignBottom)
        head.addStretch(1)
        self.rows.addLayout(head)

    def finish(self) -> None:
        self.rows.addStretch(1)


def labeled(text: str, field: QWidget) -> QVBoxLayout:
    """A form field with its label above it (4 px apart); the label's buddy
    is the field."""
    box = QVBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(4)
    label = QLabel(text)
    label.setBuddy(field)
    box.addWidget(label)
    box.addWidget(field)
    return box


class ColorField(_ClickOnEnter, _PaintedButton):
    """A color setting as a field: a 16 px chip and the hex value. Click opens
    the picker (the owner connects ``clicked``)."""

    def __init__(self, color: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(palette, parent)
        self._value = color
        self.setAccessibleName(color)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def color(self) -> str:
        return self._value

    def set_color(self, color: str) -> None:
        self._value = color
        self.setAccessibleName(color)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(120, FIELD_H)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        enabled = self.isEnabled()
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self.ring_visible():
            border = "accent"
        elif enabled and self.underMouse():
            border = "muted"
        else:
            border = "edge" if enabled else "line"
        p.setPen(QPen(self._color(border), 1.0))
        p.setBrush(self._color("field" if enabled else "ground"))
        p.drawRoundedRect(box, 4.0, 4.0)
        chip = QRectF(10.5, (self.height() - 16) / 2.0 + 0.5, 15.0, 15.0)
        p.setPen(QPen(self._color("edge"), 1.0))
        p.setBrush(QColor(self._value))
        p.drawRoundedRect(chip, 2.0, 2.0)
        p.setPen(self._color("ink" if enabled else "muted"))
        p.drawText(QRectF(34, 0, self.width() - 40, self.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, self._value)
        p.end()


def _draw_chevron(widget: QWidget, palette: dict, kind: str) -> None:
    """Paint the mockup's muted chevron at the field's right edge: ``down``
    (10 x 10, combos) or ``updown`` (8 x 14, spin boxes)."""
    p = QPainter(widget)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(palette["muted"]), 1.2))
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    right = widget.width() - 9.0
    if kind == "down":
        x, y = right - 10.0, (widget.height() - 10) / 2.0
        path.moveTo(x + 2, y + 3.5)
        path.lineTo(x + 5, y + 6.5)
        path.lineTo(x + 8, y + 3.5)
    else:
        x, y = right - 8.0, (widget.height() - 14) / 2.0
        path.moveTo(x + 1.5, y + 5.5)
        path.lineTo(x + 4, y + 3)
        path.lineTo(x + 6.5, y + 5.5)
        path.moveTo(x + 1.5, y + 8.5)
        path.lineTo(x + 4, y + 11)
        path.lineTo(x + 6.5, y + 8.5)
    p.drawPath(path)
    p.end()


class ChevronCombo(QComboBox):
    """QComboBox with the mockup's chevron (QSS hides the native arrow)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        _draw_chevron(self, self._palette, "down")


class ChevronFontCombo(QFontComboBox):
    """QFontComboBox with the mockup's chevron."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        _draw_chevron(self, self._palette, "down")


class StepSpin(QSpinBox):
    """QSpinBox with the mockup's up/down chevron (QSS hides the arrows; the
    transparent step buttons stay clickable under it)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        _draw_chevron(self, self._palette, "updown")


class StepDoubleSpin(QDoubleSpinBox):
    """QDoubleSpinBox with the mockup's up/down chevron."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        _draw_chevron(self, self._palette, "updown")


class ProgressLine(QWidget):
    """The download dialog's 4 px progress track: accent fill on the rail,
    muted once the download stopped. ``radius`` rounds both ends (2: Settings'
    dialog; 0: setup's square bars)."""

    def __init__(self, palette: dict, parent: QWidget | None = None, *,
                 radius: float = 2.0) -> None:
        super().__init__(parent)
        self._palette = palette
        self._radius = radius
        self._fraction = 0.0
        self._stopped = False
        self.setFixedHeight(4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def set_fraction(self, fraction: float) -> None:
        self._fraction = max(0.0, min(1.0, float(fraction)))
        self.update()

    def set_stopped(self, stopped: bool) -> None:
        self._stopped = bool(stopped)
        self.update()

    def fraction(self) -> float:
        return self._fraction

    def stopped(self) -> bool:
        return self._stopped

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._palette["rail"]))
        p.drawRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        if self._fraction > 0:
            p.setBrush(QColor(self._palette["muted" if self._stopped else "accent"]))
            p.drawRoundedRect(QRectF(0, 0, self.width() * self._fraction, self.height()),
                              self._radius, self._radius)
        p.end()
