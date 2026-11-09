"""The setup wizard's painted parts (plan phase 7, Task 4): state marks,
status rows, the alert, the level meter, the tone sample and the tray
drawing, offscreen."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QDateTime, QLocale, QPoint, QPointF, QRect, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QRegion  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from yamabiko.overlay_app.controls import ui_font  # noqa: E402
from yamabiko.overlay_app.theme import MIST  # noqa: E402
from yamabiko.wizard_pages import parts  # noqa: E402


def _render(widget: QWidget) -> QImage:
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def _has_ink(image: QImage, xs: range, ys: range) -> bool:
    return any(image.pixelColor(x, y).alpha() for x in xs for y in ys)


def test_state_marks_draw_a_check_a_filled_dot_and_a_hollow_ring(qtbot):
    """design §4: done = ink check, now = filled ink dot, paused = filled
    muted dot, waiting = hollow muted ring.

    Seeded defects: fill the waiting ring, paint "now" muted, or draw a dot
    for "done" -> red."""
    def mark(state: str) -> QImage:
        image = QImage(14, 14, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        parts.paint_state_mark(painter, QPointF(0, 0), state, MIST)
        painter.end()
        return image

    assert mark("now").pixelColor(7, 7).name() == MIST["ink"]
    assert mark("paused").pixelColor(7, 7).name() == MIST["muted"]
    waiting = mark("waiting")
    assert waiting.pixelColor(7, 7).alpha() == 0
    assert waiting.pixelColor(7, 4).alpha() > 0
    done = mark("done")
    assert done.pixelColor(6, 6).alpha() == 0
    assert done.pixelColor(8, 7).name() == MIST["ink"]


def test_status_rows_put_size_bar_and_status_in_their_columns(qtbot):
    """design §6 Download: rows 44 tall under a top rule; the 96 px bar ends
    10 px left of the status text, accent, muted once stopped; each row's
    accessible element is named by its text.

    Seeded defects: 43 px rows, the bar painted over the text, the stopped
    flag ignored, no size text in its column, or a stale accessible name ->
    red."""
    rows = parts.StatusRows(["Speech recognition", "Translation model", "Voice detection"], MIST)
    qtbot.addWidget(rows)
    rows.resize(578, rows.height())
    percent = parts.Run("42%", ui_font(14), "ink")
    rows.set_row(1, "now", [percent], size="5.1 GB", bar=0.5)
    assert rows.height() == 1 + 3 * 44
    assert rows.row_text(1) == "Translation model, 5.1 GB, 42%"
    assert rows.row_state(1) == "now"
    assert [c.accessibleName() for c in rows.findChildren(QWidget)][1] == rows.row_text(1)
    bar_right = 578 - QFontMetricsF(percent.font).horizontalAdvance("42%") - 10
    center_y = 45 + 21
    image = _render(rows)
    assert image.pixelColor(round(bar_right - 90), center_y).name() == MIST["accent"]
    assert image.pixelColor(round(bar_right - 10), center_y).name() == MIST["rail"]
    size_right = 578 - 168 - 12
    assert _has_ink(image, range(size_right - 64, size_right), range(57, 77))
    rows.set_row(1, "paused", [parts.Run("42%", ui_font(14))], size="5.1 GB", bar=0.5,
                 bar_stopped=True)
    assert _render(rows).pixelColor(round(bar_right - 90), center_y).name() == MIST["muted"]


def test_status_rows_without_sizes_keep_the_name_clear_of_the_bar(qtbot):
    """A row with a bar but no size column: a long name stops 12 px before
    the bar instead of running under it.

    Seeded defect: the name's right edge ignores the bar -> red."""
    rows = parts.StatusRows(["Translation model for the setup check with a long name"], MIST,
                            sizes=False)
    qtbot.addWidget(rows)
    rows.resize(300, rows.height())
    percent = parts.Run("42%", ui_font(14), "ink")
    rows.set_row(0, "now", [percent], bar=0.0)
    bar_x = round(300 - QFontMetricsF(percent.font).horizontalAdvance("42%") - 10 - 96)
    image = _render(rows)
    assert _has_ink(image, range(26, bar_x - 12), range(13, 33))
    beside_bar = range(bar_x - 12, bar_x + 96)
    assert not _has_ink(image, beside_bar, range(13, 20))  # above the bar's 4 px (y 20-23)
    assert not _has_ink(image, beside_bar, range(24, 33))  # below it


def test_alert_sizes_to_its_lines_and_hides_the_empty_ones(qtbot):
    """design §6: the disk-full alert (message + actions) is 94 px tall, the
    translator alert (message, help, actions, after line) 142, under the
    setup window QSS (32 px buttons, the 1 px border).

    Seeded defects: keep an empty help line visible, or lose the 12 px gap
    above the actions -> red."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.window_chrome import apply_window_qss

    register_bundled_fonts()  # offscreen has no system fonts: one-line message
    dialog = QDialog()
    dialog.setObjectName("setupWizard")
    qtbot.addWidget(dialog)
    column = QVBoxLayout(dialog)
    column.setContentsMargins(0, 0, 0, 0)
    alert = parts.Alert(MIST)
    column.addWidget(alert)
    column.addStretch(1)
    alert.set_text("Not enough space on drive C:. Free up 2.1 GB, then click Retry.")
    alert.add_button("Delete old Yamabiko models (4.4 GB)", primary=True)
    alert.add_button("Retry")
    alert.add_link("Copy details")
    apply_window_qss(dialog, MIST)
    dialog.resize(578, 400)
    dialog.show()
    column.activate()
    assert alert.height() == 94
    alert.set_text("The translator didn't answer yet.",
                   "The first start can take a minute while the model loads.",
                   "If you skip, Yamabiko tries again when it starts.")
    column.activate()
    assert alert.height() == 142
    assert alert.after_text() == "If you skip, Yamabiko tries again when it starts."


def test_level_meter_lights_cells_from_the_left(qtbot):
    """design §6 Sound: 32 cells, ink when lit, rail when not.

    Seeded defects: light from the right, or never repaint -> red."""
    meter = parts.LevelMeter(MIST)
    qtbot.addWidget(meter)
    meter.set_level(0.5)
    assert meter.lit_cells() == 16
    image = _render(meter)
    step = 422 / 32
    assert image.pixelColor(round(15 * step) + 2, 5).name() == MIST["ink"]
    assert image.pixelColor(round(16 * step) + 2, 5).name() == MIST["rail"]


def test_tone_sample_banner_and_line_follow_the_genre(qtbot):
    """L9 / design §6 Content: the vermilion banner at x 22 and the paper
    strip from x 52, as wide as the genre's line (Variety's strip is about
    282 px, Anime's 232), the block 20 px above the bottom.

    Seeded defects: a fixed sample line, a banner kept from the first genre,
    set_genre without a repaint, or an unknown genre accepted -> red."""
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.telop_look import BANNER_RED

    register_bundled_fonts()
    paper = QColor(*parts._SAMPLE_LOOK.paper).name()
    banner_column = QRect(22, 0, 30, 214)
    sample = parts.ToneSample(genre="variety")
    qtbot.addWidget(sample)
    image = _render(sample)
    assert image.pixelColor(23, 128).name() == QColor(*BANNER_RED).name()
    assert image.pixelColor(52 + 256, 128).name() == paper
    sample.set_genre("anime")
    assert sample.genre() == "anime"
    anime = _render(sample)
    assert anime.pixelColor(52 + 256, 128).name() != paper
    assert anime.copy(banner_column) != image.copy(banner_column)
    with pytest.raises(ValueError):
        sample.set_genre("sports")


def test_tray_drawing_highlights_the_app_tile_and_shows_the_given_time(qtbot):
    """L11 / design §6 Done: the second tile is the hover tone with the app
    icon; the clock shows the time and date it was given, in the system's
    short formats.

    Seeded defects: no highlight tile, or a hard-coded "9:41 PM" -> red."""
    tray = parts.TrayDrawing(MIST)
    qtbot.addWidget(tray)
    when = QDateTime.fromString("2026-10-27T21:41:00", Qt.ISODate)
    tray.set_time(when)
    locale = QLocale.system()
    assert tray.clock_lines() == (locale.toString(when.time(), QLocale.FormatType.ShortFormat),
                                  locale.toString(when.date(), QLocale.FormatType.ShortFormat))
    image = _render(tray)
    assert image.pixelColor(44, 9).name() == MIST["hover"]
    assert image.pixelColor(10, 9).name() == MIST["field"]
