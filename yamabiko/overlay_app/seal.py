"""The 山彦 hanko seal from the approved Settings title.

A vermilion rounded square with a thin white inner rule and the two
characters stacked in the bundled BIZ UDPMincho Bold (falling back to Yu
Mincho, then Klee One, where the bundled fonts aren't registered), laid out
on the mockup's 32-unit grid and scaled to any square size. The same in both
themes.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

SEAL_RED = "#c8402c"
SEAL_WHITE = "#ffffff"
_GRID = 32.0


def paint_seal(painter: QPainter, rect: QRectF) -> None:
    """Paint the seal into the square ``rect``."""
    s = rect.width() / _GRID
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.translate(rect.topLeft())

    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(SEAL_RED))
    painter.drawRoundedRect(QRectF(0, 0, _GRID * s, _GRID * s), 5.8 * s, 5.8 * s)

    painter.setPen(QPen(QColor(SEAL_WHITE), 1.0 * s))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRectF(2.5 * s, 2.5 * s, 27 * s, 27 * s), 3.6 * s, 3.6 * s)

    font = QFont()
    font.setFamilies(["BIZ UDPMincho", "Yu Mincho", "Klee One"])
    font.setPixelSize(max(1, round(11 * s)))
    font.setWeight(QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor(SEAL_WHITE))
    for char, center_y in (("山", 10.6), ("彦", 21.6)):
        box = QRectF(0, (center_y - 5.5) * s, _GRID * s, 11 * s)
        painter.drawText(box, Qt.AlignCenter, char)
    painter.restore()


class Seal(QWidget):
    """A fixed-size square widget showing the seal."""

    def __init__(self, size: int = 26, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setAccessibleName("Yamabiko seal")

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        paint_seal(painter, QRectF(0, 0, self.width(), self.height()))
        painter.end()
