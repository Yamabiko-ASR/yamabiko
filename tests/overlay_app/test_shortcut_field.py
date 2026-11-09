"""Shortcut field: click to record, Esc cancels, a full combo is recorded."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from yamabiko.overlay_app.shortcut_field import ShortcutField, combo_from_key_event
from yamabiko.overlay_app.theme import MIST

_CTRL_ALT = Qt.ControlModifier | Qt.AltModifier


def test_combo_from_key_event_needs_a_main_key_and_a_modifier():
    # Seeded defects: accept a bare letter, drop the native virtual key
    # (AltGr layouts report another Qt key under Ctrl+Alt), or order the
    # modifiers as pressed -> red.
    assert combo_from_key_event(int(Qt.Key_K), _CTRL_ALT) == "Ctrl+Alt+K"
    assert combo_from_key_event(int(Qt.Key_At), _CTRL_ALT, 0x51) == "Ctrl+Alt+Q"
    assert combo_from_key_event(int(Qt.Key_F19), Qt.ShiftModifier) == "Shift+F19"
    assert combo_from_key_event(int(Qt.Key_K), Qt.ShiftModifier) is None
    assert combo_from_key_event(int(Qt.Key_Control), _CTRL_ALT) is None
    assert combo_from_key_event(int(Qt.Key_Space), _CTRL_ALT) is None


def test_click_records_a_combo_and_esc_cancels(qtbot):
    # Seeded defects: Esc records "Esc" or keeps recording; a modifier-only
    # press ends recording; the recorded combo isn't emitted -> red.
    field = ShortcutField("Ctrl+Alt+P", MIST)
    qtbot.addWidget(field)
    field.show()
    recorded = []
    field.recorded.connect(recorded.append)
    QTest.mouseClick(field, Qt.LeftButton)
    assert field.is_recording()
    QTest.keyClick(field, Qt.Key_Escape)
    assert not field.is_recording() and field.combo() == "Ctrl+Alt+P"
    QTest.mouseClick(field, Qt.LeftButton)
    QTest.keyClick(field, Qt.Key_Control, Qt.ControlModifier)
    assert field.is_recording()
    QTest.keyClick(field, Qt.Key_K, _CTRL_ALT)
    assert not field.is_recording()
    assert recorded == ["Ctrl+Alt+K"] and field.combo() == "Ctrl+Alt+K"
    # Reset sets the combo while the caps are on screen: they must resize,
    # not clip. Seeded defect: no adjustSize in set_combo -> the caps keep
    # the "Ctrl+Alt+K" width -> red.
    field.set_combo("Ctrl+Alt+Shift+F12")
    assert field._caps.width() == field._caps.sizeHint().width()


def test_warning_and_recording_draw_their_outlines(qtbot):
    # The taken state's amber edge and the recording state's accent edge.
    # Seeded defects: paint the warning edge in the accent, or no recording
    # edge -> red.
    field = ShortcutField("Ctrl+Alt+S", MIST)
    qtbot.addWidget(field)
    field.show()

    def edge():
        img = QImage(field.size(), QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        field.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        return img.pixelColor(field.width() // 2, 1).name()

    field.set_warning(True)
    assert edge() == MIST["warn"]
    field.start_recording()
    assert edge() == MIST["accent"]
