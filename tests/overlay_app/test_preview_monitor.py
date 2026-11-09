"""Settings preview monitor: the night photo behind the real subtitle stack."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtWidgets import QWidget

from yamabiko.overlay_app.preview_monitor import (
    MONITOR_H, MONITOR_W, PreviewMonitor, photo_path,
)
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS


def _render(widget) -> QImage:
    img = QImage(widget.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    widget.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return img


def test_monitor_paints_the_bundled_photo(qtbot):
    # Seeded defects: a wrong photo path (falls back to the flat fill), or a
    # missing asset -> red.
    assert photo_path().is_file()
    monitor = PreviewMonitor(dict(STYLE_DEFAULTS), screen_width=1920)
    qtbot.addWidget(monitor)
    monitor.resize(MONITOR_W, MONITOR_H)
    monitor.show()
    assert monitor.photo_loaded()
    img = _render(monitor)
    top_row = {img.pixelColor(x, 6).name() for x in range(20, MONITOR_W - 20, 7)}
    assert len(top_row) > 20, "the top of the monitor is a flat fill, not the photo"


def test_samples_sit_on_the_mockup_anchor_and_respect_the_screen_cap(qtbot):
    # The mockup's stack: 28 px from the left, 22 px from the bottom; a line
    # never runs wider than the overlay's own cap (screen width x max
    # width). Seeded defects: anchor at x 0 or on the bottom edge, or drop
    # the screen-width term from the cap -> red. (Offscreen font metrics are
    # not real, so the test pins geometry, not the wrap.)
    narrow = {**STYLE_DEFAULTS, "max_width_pct": 50}
    monitor = PreviewMonitor(narrow, screen_width=500)
    qtbot.addWidget(monitor)
    monitor.resize(MONITOR_W, MONITOR_H)
    monitor.show()
    monitor.rebuild(narrow)
    newest = monitor.stack.frame().lines[-1]
    assert newest.rect.left() == 28
    assert newest.rect.bottom() + 1 == MONITOR_H - 22
    assert newest.rect.right() < 28 + 250
