"""Flat key caps for a shortcut such as Ctrl+Alt+P.

Display only (the Shortcuts tab's recording field and setup's Done page put
these inside their own controls). Each cap: a face over a 1 px edge with a
2 px bottom, 4 px between caps. ``size="small"`` (the Shortcuts tab): 20 px
tall, hover-tone face, 3 px corners, 12 px semibold label. ``size="large"``
(setup's Done page, opt-wiz-rail.tpl kbd): 26 px tall, at least 28 wide,
field-tone face, 2 px corners, 13 px regular label.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget


@dataclass(frozen=True)
class _CapSize:
    height: int
    min_width: int
    pad_x: int
    px: int
    semibold: bool
    face: str      # palette token
    radius: float


_SIZES = {
    "small": _CapSize(20, 22, 6, 12, True, "hover", 3.0),
    "large": _CapSize(26, 28, 7, 13, False, "field", 2.0),
}
_CAP_EDGE = 1  # border-box, as the mockup: padding sits inside the edge
_CAP_GAP = 4
_NAMED_KEYS = {
    "ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift",
    "win": "Win", "windows": "Win", "esc": "Esc", "escape": "Esc",
}


def key_labels(sequence: str) -> list[str]:
    """Split a shortcut ("ctrl+alt+p") into cap labels (["Ctrl", "Alt", "P"])."""
    labels: list[str] = []
    for part in sequence.split("+"):
        part = part.strip()
        if not part:
            continue
        named = _NAMED_KEYS.get(part.lower())
        if named is not None:
            labels.append(named)
        elif len(part) == 1:
            labels.append(part.upper())
        elif any(ch.isupper() for ch in part):
            labels.append(part)
        else:
            labels.append(part.capitalize())
    return labels


class KeyCaps(QWidget):
    """Paints one flat cap per key of ``sequence``."""

    def __init__(self, sequence: str, palette: dict, parent: QWidget | None = None, *,
                 size: str = "small") -> None:
        super().__init__(parent)
        self._palette = palette
        self._size = _SIZES[size]
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._labels: list[str] = []
        self.set_sequence(sequence)

    def set_sequence(self, sequence: str) -> None:
        self._labels = key_labels(sequence)
        self.setAccessibleName(" ".join(self._labels))
        self.updateGeometry()
        self.update()

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def _cap_font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(self._size.px)
        font.setWeight(QFont.Weight.DemiBold if self._size.semibold else QFont.Weight.Normal)
        return font

    def _cap_widths(self) -> list[int]:
        metrics = QFontMetrics(self._cap_font())
        size = self._size
        return [
            max(size.min_width, metrics.horizontalAdvance(label) + 2 * (size.pad_x + _CAP_EDGE))
            for label in self._labels
        ]

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        widths = self._cap_widths()
        return QSize(sum(widths) + _CAP_GAP * max(0, len(widths) - 1), self._size.height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setFont(self._cap_font())
        size = self._size
        x = 0.0
        for label, width in zip(self._labels, self._cap_widths()):
            outer = QRectF(x, 0, width, size.height)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._palette["edge"]))
            painter.drawRoundedRect(outer, size.radius, size.radius)
            painter.setBrush(QColor(self._palette[size.face]))
            face = outer.adjusted(_CAP_EDGE, _CAP_EDGE, -_CAP_EDGE, -2 * _CAP_EDGE)
            painter.drawRoundedRect(face, size.radius - 1, size.radius - 1)
            painter.setPen(QColor(self._palette["ink"]))
            painter.drawText(outer.adjusted(0, 0, 0, -1), Qt.AlignCenter, label)
            x += width + _CAP_GAP
        painter.end()
