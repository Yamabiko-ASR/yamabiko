"""Settings' compact dialogs: "Couldn't save your settings" and the color
picker (the model download dialog lives with its worker in model_fetch.py).
Both use the CompactDialog look: the painted caption, a BIZ UDPMincho
heading, Cancel and the accent action at the right."""
from __future__ import annotations

import os
import re
from typing import Optional

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QWidget,
)

from .controls import LinkButton, _PaintedButton
from .style_defaults import HEX_COLOR_RE
from .window_chrome import CompactDialog

# The overlay's own colors (top row): the dark bar, Japanese ink, the paper
# strip, the English text, the banner and black; then the mockup's swatches.
OVERLAY_SWATCHES = ("#0e1014", "#15171b", "#f3eee3", "#ffffff", "#d2452f", "#000000")
MORE_SWATCHES = (
    "#1c1f24", "#2b2f35", "#3d424a", "#5a6069", "#8b9199", "#c3c7cc",
    "#3a1410", "#4a3112", "#1d3a28", "#12304a", "#29254f", "#3f1834",
    "#f5d76e", "#f2e3c6", "#cfe3c4", "#c2d9ec", "#dcd0ec", "#f2cfc6",
)
SAMPLE_TEXT = "This is what the town is famous for."


def display_path(text: str) -> str:
    """``text`` with the user's %LOCALAPPDATA% folder written as the variable,
    so a shown or copied path carries no user name."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return text
    # Errors quote the path through repr, with doubled backslashes: hide both.
    for form in (local, local.replace("\\", "\\\\")):
        text = re.sub(re.escape(form), lambda _m: "%LOCALAPPDATA%", text, flags=re.IGNORECASE)
    return text


def normalize_hex(text: str) -> Optional[str]:
    """``#rrggbb`` (lowercase) from ``#rrggbb`` or ``rrggbb``; None otherwise."""
    value = text.strip().lower()
    if not value.startswith("#"):
        value = "#" + value
    return value if HEX_COLOR_RE.match(value) else None


class SaveFailedDialog(CompactDialog):
    """"Couldn't save your settings": the file Yamabiko couldn't write, a
    quiet Copy details, Cancel and Retry (Retry accepts). ``title`` is the
    caption (setup passes "Yamabiko setup")."""

    def __init__(self, palette: dict, path: str, error, parent: QWidget | None = None, *,
                 title: str = "Yamabiko settings") -> None:
        super().__init__(palette, 440, parent, title=title)
        self._details = display_path(f"{path}\n{type(error).__name__}: {error}"
                                     if isinstance(error, BaseException) else f"{path}\n{error}")
        self.heading("Couldn't save your settings")
        self.add_text("Yamabiko couldn't write to this file:")
        self.path_label = QLabel(display_path(str(path)))
        self.path_label.setObjectName("code")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.addSpacing(12)
        self.body.addWidget(self.path_label, 0, Qt.AlignLeft)
        self.copy_button = LinkButton("Copy details", palette, quiet=True, height=24)
        self.copy_button.clicked.connect(self._copy_details)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.retry_button = QPushButton("Retry")
        self.retry_button.setProperty("primary", True)
        self.retry_button.setDefault(True)
        self.retry_button.clicked.connect(self.accept)
        self.actions.addWidget(self.copy_button)
        self.actions.addStretch(1)
        self.actions.addWidget(self.cancel_button)
        self.actions.addWidget(self.retry_button)

    @staticmethod
    def ask(parent: QWidget | None, palette: dict, path: str, error, *,
            title: str = "Yamabiko settings") -> bool:
        """Show the dialog modally; True when the user chose Retry."""
        dialog = SaveFailedDialog(palette, path, error, parent, title=title)
        try:
            return dialog.exec() == QDialog.Accepted
        finally:
            dialog.deleteLater()

    def _copy_details(self) -> None:
        """Copy with error_window's flushing copy: setup can end in os._exit
        after this dialog closes (Cancel while a download drains), which
        drops a clipboard Qt has not rendered yet. Imported at call time, so
        this module stays free of the error window's Win32 clipboard code
        until a copy."""
        from yamabiko.ui.error_window import copy_to_clipboard

        copy_to_clipboard(self._details)


def _luminance(color: QColor) -> float:
    return (0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF())


def _tile_border(palette: dict) -> QColor:
    """The mockup's swatch edge: 14% black on Mist, 16% white on Night."""
    return QColor(255, 255, 255, 41) if palette["name"] == "night" else QColor(0, 0, 0, 36)


class _Swatch(_PaintedButton):
    """One 32 px color tile in a 4 px margin (the margin holds the ring; two
    margins make the grid's 8 px gap); the chosen one gets a ground gap, a
    2 px accent ring and a check."""

    _MARGIN = 4

    def __init__(self, color: str, owner: "ColorPickerDialog") -> None:
        super().__init__(owner._palette, owner)
        self.color = color
        self.setCheckable(True)
        self.setAccessibleName(color)
        self.setFixedHeight(32 + 2 * self._MARGIN)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(48 + 2 * self._MARGIN, 32 + 2 * self._MARGIN)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        tile = QRectF(self.rect()).adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
        if self.isChecked() or self.ring_visible():
            p.setPen(Qt.NoPen)
            p.setBrush(self._color("accent"))
            p.drawRoundedRect(QRectF(self.rect()), 4.0, 4.0)
            p.setBrush(self._color("ground"))
            p.drawRoundedRect(tile.adjusted(-2, -2, 2, 2), 3.0, 3.0)
        fill = QColor(self.color)
        p.setPen(QPen(_tile_border(self._palette), 1.0))
        p.setBrush(fill)
        p.drawRoundedRect(tile.adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
        if self.isChecked():
            check = QColor("#ffffff") if _luminance(fill) < 0.5 else QColor("#1f2421")
            pen = QPen(check, 1.8)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            x, y = tile.center().x() - 7, tile.center().y() - 7
            path = QPainterPath()
            path.moveTo(x + 2.5, y + 7.5)
            path.lineTo(x + 5.5, y + 10.5)
            path.lineTo(x + 11.5, y + 3.5)
            p.drawPath(path)
        p.end()


class _Chip(QWidget):
    """The 18 px "Current" chip."""

    def __init__(self, color: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self._palette = palette
        self.setFixedSize(18, 18)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(self._palette["edge"]), 1.0))
        p.setBrush(QColor(self._color))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
        p.end()


class _Sample(QWidget):
    """A 40 px sample subtitle line: the English text color on the
    background color, Bahnschrift SemiBold 16 px."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._text_color = "#ffffff"
        self._background = "#0e1014"
        self.setFixedHeight(40)
        self.setAccessibleName("Preview of an English line on this color")

    def set_colors(self, text_color: str, background: str) -> None:
        self._text_color = text_color
        self._background = background
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(_tile_border(self._palette), 1.0))
        p.setBrush(QColor(self._background))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
        font = QFont()
        font.setFamilies(["Bahnschrift SemiBold", "Bahnschrift", "Segoe UI"])
        font.setPixelSize(16)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QColor(self._text_color))
        p.drawText(QRectF(self.rect()).adjusted(14, 0, -14, 0), Qt.AlignLeft | Qt.AlignVCenter, SAMPLE_TEXT)
        p.end()


class ColorPickerDialog(CompactDialog):
    """The compact color picker: a 6 x 4 swatch grid (the overlay's own
    colors on top), a hex field, the current color, a sample line, Cancel
    and OK. ``role`` is "text" or "background"; ``other`` is the color of
    the other role, so the sample shows text on background."""

    def __init__(self, palette: dict, role: str, initial: str, other: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(palette, 384, parent)
        # The swatch tiles sit flush with the mockup's 24 px padding: each
        # swatch keeps a 4 px margin for its ring, so the body runs at 20 px
        # and every other row takes the 4 px back.
        inset = _Swatch._MARGIN
        self.body.setContentsMargins(24 - inset, 6, 24 - inset, 0)
        self._role = role
        self._other = other
        self._chosen = normalize_hex(initial) or OVERLAY_SWATCHES[0]
        title = self.heading("Text color" if role == "text" else "Background color")
        title.setContentsMargins(inset, 0, inset, 0)
        grid = QGridLayout()
        grid.setContentsMargins(0, 8, 0, 0)
        grid.setSpacing(0)
        self.swatches: list[_Swatch] = []
        for index, color in enumerate(OVERLAY_SWATCHES + MORE_SWATCHES):
            swatch = _Swatch(color, self)
            swatch.clicked.connect(lambda _checked=False, s=swatch: self._choose(s.color, from_hex=False))
            grid.addWidget(swatch, index // 6, index % 6)
            self.swatches.append(swatch)
        self.body.addLayout(grid)
        legend = self.add_text("The top row has the overlay's own colors.", muted=True, top=6)
        legend.setContentsMargins(inset, 0, inset, 0)
        hex_row = QHBoxLayout()
        hex_row.setContentsMargins(inset, 0, inset, 0)
        hex_row.setSpacing(10)
        hex_label = QLabel("Hex")
        self.hex_edit = QLineEdit(self._chosen)
        self.hex_edit.setObjectName("mono")
        self.hex_edit.setFixedWidth(132)
        hex_label.setBuddy(self.hex_edit)
        self.hex_edit.textEdited.connect(self._on_hex_edited)
        current = QLabel("Current")
        current.setObjectName("muted")
        was = QHBoxLayout()
        was.setSpacing(8)
        was.addWidget(current)
        was.addWidget(_Chip(self._chosen, palette))
        hex_row.addWidget(hex_label)
        hex_row.addWidget(self.hex_edit)
        hex_row.addStretch(1)
        hex_row.addLayout(was)
        self.body.addSpacing(14)
        self.body.addLayout(hex_row)
        self.sample = _Sample(palette)
        sample_row = QHBoxLayout()
        sample_row.setContentsMargins(inset, 0, inset, 0)
        sample_row.addWidget(self.sample)
        self.body.addSpacing(12)
        self.body.addLayout(sample_row)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QPushButton("OK")
        self.ok_button.setProperty("primary", True)
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        self.actions.addStretch(1)
        self.actions.addWidget(self.cancel_button)
        self.actions.addWidget(self.ok_button)
        self._choose(self._chosen, from_hex=True)

    def chosen(self) -> str:
        return self._chosen

    def _choose(self, color: str, *, from_hex: bool) -> None:
        self._chosen = color
        for swatch in self.swatches:
            swatch.setChecked(swatch.color == color)
        if not from_hex:
            self.hex_edit.setText(color)
        self.ok_button.setEnabled(True)
        if self._role == "text":
            self.sample.set_colors(color, self._other)
        else:
            self.sample.set_colors(self._other, color)

    def _on_hex_edited(self, text: str) -> None:
        color = normalize_hex(text)
        if color is None:
            self.ok_button.setEnabled(False)
            return
        self._choose(color, from_hex=True)

    @staticmethod
    def pick(parent: QWidget | None, palette: dict, role: str, initial: str,
             other: str) -> Optional[str]:
        """Show the picker modally; the chosen ``#rrggbb`` or None on Cancel."""
        dialog = ColorPickerDialog(palette, role, initial, other, parent)
        try:
            return dialog.chosen() if dialog.exec() == QDialog.Accepted else None
        finally:
            dialog.deleteLater()
