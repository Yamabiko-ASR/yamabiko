"""The setup wizard's step rail (opt-wiz-rail.tpl nav.rail).

A 220 px column on the raised tone (a 1 px line on its top and right): the
hanko seal and "Yamabiko setup" in BIZ UDPMincho Bold, one 38 px row per step
and the landscape vignette at the bottom. Marks follow the position only:
rows before the current one are done (ink check; click, Enter or Space jumps
back), the current row has the 3 px accent bar, the ground tone, a filled dot
and a semibold label, rows after it are upcoming (muted ring and label,
inert). While a download is running or stopped unfinished, a 2 px line under
the Download row shows its progress (accent; muted once stopped) and lifts
that row's mark and label 4 px.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QWidget

from yamabiko.overlay_app.controls import _ClickOnEnter, mincho_font, ui_font
from yamabiko.overlay_app.landscape import paint_vignette
from yamabiko.overlay_app.seal import Seal
from yamabiko.overlay_app.segmented import KeyboardRingMixin
from yamabiko.wizard_pages.parts import paint_state_mark, semibold

DOWNLOAD_LABEL = "Download"
RAIL_W = 220          # the rail's width, its 1 px right line included
STEP_H = 38
BRAND = "Yamabiko setup"  # the rail's heading, and the setup window's title
_INNER_W = RAIL_W - 1
_STEPS_TOP = 73       # 20 padding + 28 seal + 24 gap, under the 1 px top line
_MARK_X, _LABEL_X = 22, 48
_LINE = QRectF(48, 29, 149, 2)  # the download line inside the Download row
_LIFT = 4
_VIGNETTE_H = 112
_MARKS = {"done": "done", "current": "now", "upcoming": "waiting"}


class _StepRow(_ClickOnEnter, KeyboardRingMixin, QAbstractButton):
    def __init__(self, label: str, rail: "StepRail") -> None:
        super().__init__(rail)
        self._rail = rail
        self.label = label
        self.state = "upcoming"
        self.line: tuple[float | None, bool] = (None, False)
        self.setText(label)
        self.setAccessibleName(label)
        self.setAttribute(Qt.WA_Hover, True)
        self.setFixedSize(_INNER_W, STEP_H)

    def set_state(self, state: str) -> None:
        self.state = state
        done = state == "done"
        self.setEnabled(done)
        self.setFocusPolicy(Qt.StrongFocus if done else Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor if done else Qt.ArrowCursor)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        pal = self._rail._palette
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        if self.state == "current":
            p.fillRect(self.rect(), QColor(pal["ground"]))
            p.fillRect(QRectF(0, 0, 3, STEP_H), QColor(pal["accent"]))
        elif self.state == "done" and self.underMouse():
            p.fillRect(self.rect(), QColor(pal["hover"]))
        fraction, stopped = self.line
        lift = _LIFT if fraction is not None else 0
        paint_state_mark(p, QPointF(_MARK_X, 12 - lift), _MARKS[self.state], pal)
        p.setFont(semibold(14) if self.state == "current" else ui_font(14))
        p.setPen(QColor(pal["muted" if self.state == "upcoming" else "ink"]))
        p.drawText(QRectF(_LABEL_X, 9 - lift, _INNER_W - _LABEL_X - 8, 20),
                   Qt.AlignLeft | Qt.AlignVCenter, self.label)
        if fraction is not None:
            p.fillRect(_LINE, QColor(pal["rail"]))
            p.fillRect(QRectF(_LINE.x(), _LINE.y(), round(_LINE.width() * fraction), _LINE.height()),
                       QColor(pal["muted" if stopped else "accent"]))
        if self.ring_visible():
            p.setPen(QPen(QColor(pal["accent"]), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75))
        p.end()


class StepRail(QWidget):
    """The step rail (see the module docstring). Rows come from
    ``add_step``; ``set_current`` sets the marks; ``stepClicked(index)``
    reports a click on a done row."""

    stepClicked = Signal(int)

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._rows: list[_StepRow] = []
        self._current = -1
        self._download: tuple[float | None, bool] = (None, False)
        self.setFixedWidth(RAIL_W)
        self.setAccessibleName("Setup steps")
        self._seal = Seal(28, self)
        self._seal.move(20, 21)

    def add_step(self, label: str) -> int:
        index = len(self._rows)
        row = _StepRow(label, self)
        row.move(0, _STEPS_TOP + STEP_H * index)
        row.clicked.connect(lambda _checked=False, i=index: self._clicked(i))
        if label == DOWNLOAD_LABEL:
            row.line = self._download
        self._rows.append(row)
        row.set_state(self._state(index))
        row.show()
        return index

    def count(self) -> int:
        return len(self._rows)

    def set_current(self, index: int) -> None:
        self._current = index
        for i, row in enumerate(self._rows):
            row.set_state(self._state(i))

    def current(self) -> int:
        return self._current

    def set_download(self, fraction: object, stopped: bool) -> None:
        line = (None if fraction is None else max(0.0, min(1.0, float(fraction))), bool(stopped))
        self._download = line
        for row in self._rows:
            if row.label == DOWNLOAD_LABEL:
                row.line = line
                row.update()

    def download_line(self) -> tuple[float | None, bool]:
        return self._download

    def step_button(self, index: int) -> QAbstractButton:
        return self._rows[index]

    def _state(self, index: int) -> str:
        if index < self._current:
            return "done"
        return "current" if index == self._current else "upcoming"

    def _clicked(self, index: int) -> None:
        if index < self._current:
            self.stepClicked.emit(index)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        pal = self._palette
        line = QColor(pal["line"])
        p.fillRect(QRectF(0, 0, _INNER_W, self.height()), QColor(pal["raised"]))
        p.fillRect(QRectF(0, 0, _INNER_W, 1), line)
        p.fillRect(QRectF(_INNER_W, 0, 1, self.height()), line)
        paint_vignette(p, QRectF(0, self.height() - _VIGNETTE_H, _INNER_W, _VIGNETTE_H), pal)
        p.setPen(QColor(pal["ink"]))
        p.setFont(mincho_font(16, bold=True))
        p.drawText(QRectF(58, 24, _INNER_W - 58, 22), Qt.AlignLeft | Qt.AlignVCenter, BRAND)
        p.end()
