"""Settings' compact dialogs: save failed and the color picker."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication, QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QWidget

from yamabiko.overlay_app.settings_dialogs import (
    OVERLAY_SWATCHES, ColorPickerDialog, SaveFailedDialog,
)
from yamabiko.overlay_app.theme import MIST


def test_save_failed_retry_accepts_and_details_hide_the_user_folder(qtbot, monkeypatch, tmp_path):
    # The footer promise of the report card holds here too: a copied path
    # carries %LOCALAPPDATA%, not the user's folder. Seeded defects: copy the
    # raw path, or wire Retry to reject -> red.
    local = tmp_path / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    path = local / "Yamabiko" / "user_overrides.yaml"
    dialog = SaveFailedDialog(MIST, str(path), PermissionError(13, "Access is denied", str(path)))
    qtbot.addWidget(dialog)
    assert dialog.path_label.text() == "%LOCALAPPDATA%\\Yamabiko\\user_overrides.yaml"
    QTest.mouseClick(dialog.copy_button, Qt.LeftButton)
    copied = QGuiApplication.clipboard().text()
    assert copied.startswith("%LOCALAPPDATA%\\Yamabiko")
    # The OSError text repeats the path with doubled backslashes (repr).
    assert str(local) not in copied and str(local).replace("\\", "\\\\") not in copied
    dialog.show()
    QTest.mouseClick(dialog.retry_button, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted


def test_compact_dialog_opens_centered_on_its_parent_window(qtbot):
    # QDialog centers on a guessed 10 x 40 px frame when neither the dialog
    # nor a visible window reports usable frame margins (an expanded client
    # area reports a negative left one): the dialogs opened about 40 px
    # above Settings' center. Frameless windows report none on the offscreen
    # QPA, which reproduces the guess. A parent dragged partly off its
    # screen (x = -798, review I2) still gets the whole dialog inside the
    # work area. Seeded defects: drop _center_on_parent from
    # CompactDialog.showEvent -> offset (-10, -39) -> red; drop its clamp ->
    # the dialog hangs off the screen's left edge -> red.
    parent = QWidget(None, Qt.FramelessWindowHint)
    qtbot.addWidget(parent)
    parent.setGeometry(50, 40, 700, 560)  # inside the offscreen QPA's 800 x 800 screen
    parent.show()
    avail = parent.screen().availableGeometry()
    dialog = SaveFailedDialog(MIST, "C:/x/user_overrides.yaml",
                              PermissionError(13, "Access is denied"), parent)
    dialog.setWindowFlag(Qt.FramelessWindowHint)
    try:
        dialog.show()
        offset = dialog.geometry().center() - parent.geometry().center()
        dialog.close()
        parent.setGeometry(avail.left() - 798, 100, 1000, 500)
        dialog.show()
        frame = dialog.frameGeometry()
    finally:
        dialog.close()
    assert abs(offset.x()) <= 1 and abs(offset.y()) <= 1, offset
    assert avail.contains(frame), (frame, avail)


def test_compact_dialogs_on_settings_show_the_window_font(qtbot):
    # Settings parents every compact dialog. The dialog then runs under
    # Settings' style sheet style from its construction, and Qt resolved the
    # fonts of its rows (the subclasses' too) against the app font: 12 px
    # text and a regular-weight primary button, where the mockup has the
    # window's 13 px and Settings' DemiBold OK. Seeded defect: drop the font
    # step from CompactDialog.showEvent -> the rows show the app font -> red.
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit

    from yamabiko.overlay_app.model_fetch import QuantDownloadDialog, spec_for_tier
    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    settings = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    qtbot.addWidget(settings)
    palette, window_font = settings._palette, settings.font()
    dialogs = (
        ColorPickerDialog(palette, "background", "#0e1014", "#ffffff", settings),
        QuantDownloadDialog(spec_for_tier("small"), "small", parent=settings, palette=palette),
        SaveFailedDialog(palette, "C:/x/user_overrides.yaml",
                         PermissionError(13, "Access is denied"), settings),
    )
    wrong, primaries = [], []
    for dialog in dialogs:
        dialog.show()
        for widget in dialog.findChildren(QWidget):
            texted = isinstance(widget, (QLabel, QLineEdit)) or (
                isinstance(widget, QAbstractButton) and widget.text())
            # The heading (BIZ UDPMincho) and the path (Consolas 12 px) are
            # meant to differ; the hex field takes Consolas at the window size.
            if not texted or widget.objectName() in ("dialogTitle", "code"):
                continue
            font = widget.font()
            family_ok = widget.objectName() == "mono" or font.families() == window_font.families()
            if not family_ok or font.pixelSize() != window_font.pixelSize():
                wrong.append((type(dialog).__name__, widget.objectName() or widget.text(),
                              font.families(), font.pixelSize(), font.pointSizeF()))
            if widget.property("primary"):
                primaries.append((type(dialog).__name__, widget.text(), font.weight()))
        dialog.close()
    assert not wrong, wrong
    assert [weight for _d, _t, weight in primaries] == [QFont.Weight.DemiBold] * 3, primaries


def test_color_picker_swatches_and_hex_choose_the_color(qtbot):
    # Seeded defects: put the overlay colors anywhere but the top row, let a
    # swatch click leave the hex field stale, or keep OK enabled on a bad hex
    # -> red.
    dialog = ColorPickerDialog(MIST, "background", "#0e1014", "#ffffff")
    qtbot.addWidget(dialog)
    assert [s.color for s in dialog.swatches[:6]] == list(OVERLAY_SWATCHES)
    assert dialog.swatches[0].isChecked()
    target = next(s for s in dialog.swatches if s.color == "#12304a")
    QTest.mouseClick(target, Qt.LeftButton)
    assert dialog.hex_edit.text() == "#12304a" and dialog.chosen() == "#12304a"
    dialog.hex_edit.clear()
    QTest.keyClicks(dialog.hex_edit, "F5D76E")
    assert dialog.chosen() == "#f5d76e" and dialog.ok_button.isEnabled()
    QTest.keyClicks(dialog.hex_edit, "Z")
    assert not dialog.ok_button.isEnabled()


def test_color_picker_sample_puts_the_text_on_the_chosen_background(qtbot):
    # The sample line shows the English text color on the picked background.
    # Seeded defect: paint the sample from the initial color -> red.
    dialog = ColorPickerDialog(MIST, "background", "#0e1014", "#ffffff")
    qtbot.addWidget(dialog)
    QTest.mouseClick(next(s for s in dialog.swatches if s.color == "#3f1834"), Qt.LeftButton)
    sample = dialog.sample
    img = QImage(sample.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    sample.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert img.pixelColor(4, sample.height() // 2).name() == "#3f1834"


def test_save_failed_copy_details_goes_through_the_flushing_copy(qtbot, monkeypatch):
    """Decision 146: setup can end in os._exit after this dialog closed
    (Cancel while a download drains), which drops a clipboard Qt has not
    rendered yet (phase 6), so Copy details uses error_window's flushing
    copy_to_clipboard, in Settings too.

    Seeded defect: keep QGuiApplication.clipboard().setText -> nothing
    reaches copy_to_clipboard -> red.
    """
    copied: list[str] = []
    monkeypatch.setattr("yamabiko.ui.error_window.copy_to_clipboard", copied.append)
    dialog = SaveFailedDialog(MIST, "C:/x/user_overrides.yaml", PermissionError(13, "Access is denied"))
    qtbot.addWidget(dialog)

    QTest.mouseClick(dialog.copy_button, Qt.LeftButton)

    assert copied == [dialog._details]
