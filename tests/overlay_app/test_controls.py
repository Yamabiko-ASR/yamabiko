"""Painted Settings controls: switch, tabs, color field, progress line."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from yamabiko.overlay_app.controls import ColorField, ProgressLine, Switch, TextTabs
from yamabiko.overlay_app.theme import MIST


def _render(widget) -> QImage:
    img = QImage(widget.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    widget.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return img


def test_switch_click_toggles_and_lights_the_track(qtbot):
    # Seeded defects: not checkable, or paint the checked track unlit -> red.
    switch = Switch("Reduce motion", MIST)
    qtbot.addWidget(switch)
    switch.resize(switch.sizeHint())
    seen = []
    switch.toggled.connect(seen.append)
    QTest.mouseClick(switch, Qt.LeftButton)
    assert seen == [True] and switch.isChecked()
    track_mid = _render(switch).pixelColor(8, switch.height() // 2)
    assert track_mid.name() == MIST["accent"]


def test_tabs_click_and_arrow_keys_change_the_current_tab(qtbot):
    # Seeded defects: no signal on click, no arrow-key move, or no accent bar
    # under the current tab -> red.
    tabs = TextTabs([("Subtitles", "字幕"), ("Position", "位置"), ("Timing", "間")], MIST)
    qtbot.addWidget(tabs)
    tabs.resize(600, 40)
    tabs.show()
    seen = []
    tabs.currentChanged.connect(seen.append)
    buttons = tabs._tabs
    QTest.mouseClick(buttons[2], Qt.LeftButton)
    QTest.keyClick(buttons[2], Qt.Key_Left)
    assert seen == [2, 1] and tabs.currentIndex() == 1
    tab = buttons[1]
    bar = _render(tabs).pixelColor(tab.x() + tab.width() // 2, tabs.height() - 1)
    assert bar.name() == MIST["accent"]


def test_color_field_shows_its_color_chip(qtbot):
    # Seeded defect: paint the chip from the initial color only -> red.
    field = ColorField("#ffffff", MIST)
    qtbot.addWidget(field)
    field.resize(170, 30)
    field.set_color("#0e1014")
    assert field.color() == "#0e1014"
    assert _render(field).pixelColor(18, 15).name() == "#0e1014"


def test_progress_line_fill_turns_muted_when_stopped(qtbot):
    # The failed download state greys the bar. Seeded defect: keep the accent
    # fill when stopped -> red.
    line = ProgressLine(MIST)
    qtbot.addWidget(line)
    line.resize(200, 4)
    line.set_fraction(0.5)
    assert _render(line).pixelColor(50, 2).name() == MIST["accent"]
    line.set_stopped(True)
    assert _render(line).pixelColor(50, 2).name() == MIST["muted"]
    assert _render(line).pixelColor(150, 2).name() == MIST["rail"]


def test_progress_line_with_radius_zero_has_square_ends(qtbot):
    # Setup's bars are square-ended. Seeded defect: ignore radius -> the
    # fill's corner pixel is antialiased, not the accent -> red.
    line = ProgressLine(MIST, radius=0)
    qtbot.addWidget(line)
    line.resize(96, 4)
    line.set_fraction(1.0)
    img = _render(line)
    assert img.pixelColor(0, 0).name() == MIST["accent"]
    assert img.pixelColor(95, 3).name() == MIST["accent"]
