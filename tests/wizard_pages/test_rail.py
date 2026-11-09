"""The setup wizard's step rail (plan phase 7, Task 4), offscreen."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QImage, QRegion  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from yamabiko.overlay_app.theme import MIST  # noqa: E402
from yamabiko.wizard_pages.rail import StepRail  # noqa: E402

_LABELS = ("Welcome", "Sound", "Content", "Quality", "Download", "Translator", "Done")


def _rail(qtbot) -> StepRail:
    rail = StepRail(MIST)
    qtbot.addWidget(rail)
    for label in _LABELS:
        rail.add_step(label)
    rail.resize(220, 566)
    rail.show()
    return rail


def _render(widget: QWidget) -> QImage:
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def test_rows_before_the_current_step_jump_back_and_later_rows_do_nothing(qtbot):
    """L2: rows before the current one are done and clickable; the current
    and upcoming rows are inert. Accessible names are the step labels.

    Seeded defects: let an upcoming row emit, keep the current row
    clickable, or mark by "reached" instead of position -> red."""
    rail = _rail(qtbot)
    seen: list[int] = []
    rail.stepClicked.connect(seen.append)
    rail.set_current(3)
    for index in (1, 3, 4):
        QTest.mouseClick(rail.step_button(index), Qt.LeftButton)
    assert seen == [1]
    rail.set_current(1)
    QTest.mouseClick(rail.step_button(2), Qt.LeftButton)
    assert seen == [1]
    assert [rail.step_button(i).accessibleName() for i in range(rail.count())] == list(_LABELS)


def test_return_on_a_done_row_jumps_back(qtbot):
    """D25 / Interfaces: a done row jumps on Enter as on a click.

    Seeded defect: _ClickOnEnter dropped from the row's bases -> red."""
    rail = _rail(qtbot)
    seen: list[int] = []
    rail.stepClicked.connect(seen.append)
    rail.set_current(3)
    QTest.keyClick(rail.step_button(1), Qt.Key_Return)
    assert seen == [1]


def test_rail_paints_the_current_bar_the_download_line_and_the_vignette(qtbot):
    """design §4: the current row's 3 px accent bar; under the Download row
    a 149 px line at (48, row + 29), accent while running, muted once
    stopped, gone when the queue finished; the vignette's far ridge at the
    bottom (design x 110, y 70).

    Seeded defects: no accent bar, a line that ignores ``stopped``, a line
    left after (None, False), or no vignette -> red."""
    rail = _rail(qtbot)
    rail.set_current(4)
    download_top = 73 + 4 * 38
    rail.set_download(0.5, False)
    image = _render(rail)
    assert image.pixelColor(1, download_top + 19).name() == MIST["accent"]
    assert image.pixelColor(58, download_top + 29).name() == MIST["accent"]
    assert image.pixelColor(188, download_top + 29).name() == MIST["rail"]
    assert image.pixelColor(110, 566 - 112 + 70).name() == MIST["ridge1"]
    rail.set_download(0.5, True)
    assert _render(rail).pixelColor(58, download_top + 29).name() == MIST["muted"]
    rail.set_download(None, False)
    assert rail.download_line() == (None, False)
    assert _render(rail).pixelColor(58, download_top + 29).name() == MIST["ground"]
