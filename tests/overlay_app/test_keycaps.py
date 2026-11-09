"""Key caps show a shortcut as one flat cap per key."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtWidgets import QWidget

from yamabiko.overlay_app.keycaps import KeyCaps, key_labels
from yamabiko.overlay_app.theme import MIST


def test_key_labels_normalize_config_spelling():
    # Config and the keyboard library spell shortcuts in either case; caps
    # must read "Ctrl", "Alt", "P". Seeded defect: return parts verbatim ->
    # "ctrl" shows on a cap, red.
    assert key_labels("ctrl+alt+p") == ["Ctrl", "Alt", "P"]
    assert key_labels("Ctrl+Shift+F5") == ["Ctrl", "Shift", "F5"]


def test_keycaps_paints_one_cap_per_key(qtbot):
    # Across the middle row, each cap is one opaque run separated by
    # transparent gaps. Seeded defects: paint one box for the whole sequence,
    # or drop the gap -> the run count is not 3, red.
    caps = KeyCaps("Ctrl+Alt+P", MIST)
    qtbot.addWidget(caps)
    size = caps.sizeHint()
    caps.resize(size)
    img = QImage(size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    caps.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)

    runs, inside = 0, False
    for x in range(size.width()):
        opaque = img.pixelColor(x, size.height() // 2).alpha() > 0
        if opaque and not inside:
            runs += 1
        inside = opaque
    assert runs == 3


def test_large_caps_are_26_tall_on_the_field_face_with_a_2_px_bottom_edge(qtbot):
    # Setup's Done caps (opt-wiz-rail.tpl kbd). Seeded defects: keep the
    # 20 px size, the hover face, or a 1 px bottom edge -> red.
    caps = KeyCaps("Ctrl+Alt+O", MIST, size="large")
    qtbot.addWidget(caps)
    caps.resize(caps.sizeHint())
    assert caps.height() == 26
    img = QImage(caps.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    caps.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert img.pixelColor(3, 3).name() == MIST["field"]
    assert img.pixelColor(3, 23).name() == MIST["field"]
    assert img.pixelColor(3, 24).name() == MIST["edge"]
