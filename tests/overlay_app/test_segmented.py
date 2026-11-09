"""SegmentedControl: one lit choice, picked by click or arrow keys."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QWidget

from yamabiko.overlay_app.segmented import SegmentedControl, SegmentOption
from yamabiko.overlay_app.theme import MIST


def _control(qtbot):
    seg = SegmentedControl(
        [SegmentOption("left", "Left"), SegmentOption("center", "Center")],
        MIST,
        value="left",
    )
    qtbot.addWidget(seg)
    seen: list[str] = []
    seg.valueChanged.connect(seen.append)
    return seg, seg.findChildren(QAbstractButton), seen


def test_segment_click_selects_one_value_and_emits_it(qtbot):
    # Seeded defects: leave the old segment checked, let a second click on
    # the chosen segment un-check it, or emit on a click that changes
    # nothing -> red.
    seg, (left, center), seen = _control(qtbot)
    center.click()
    assert seg.value() == "center"
    assert center.isChecked() and not left.isChecked()
    center.click()
    assert center.isChecked()
    assert seen == ["center"]


def test_arrow_keys_move_the_selection_between_segments(qtbot):
    # Seeded defects: ignore arrow keys, wrap past the last segment, or move
    # without emitting -> red.
    seg, (left, center), seen = _control(qtbot)
    QTest.keyClick(left, Qt.Key_Right)
    assert seg.value() == "center"
    QTest.keyClick(center, Qt.Key_Right)
    assert seg.value() == "center"
    QTest.keyClick(center, Qt.Key_Left)
    assert seg.value() == "left"
    assert seen == ["center", "left"]


def test_only_the_chosen_segment_is_a_tab_stop(qtbot):
    # Tab enters the control once, on the chosen segment; arrow keys move
    # inside it. Seeded defects: leave every segment a Tab stop, or set the
    # policies only at construction -> red after the click moves the choice.
    seg, (left, center), _seen = _control(qtbot)
    center.click()
    assert center.focusPolicy() == Qt.StrongFocus
    assert left.focusPolicy() == Qt.ClickFocus


def test_disabled_control_paints_no_accent(qtbot):
    # A disabled control must not look live. Seeded defects: keep the accent
    # fill, accent outline or a ring on the chosen segment while disabled ->
    # accent pixels in the render, red.
    seg, _segments, _seen = _control(qtbot)
    seg.setEnabled(False)
    seg.resize(160, seg.height())
    img = QImage(seg.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    seg.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)

    accent = sum(
        1
        for x in range(img.width())
        for y in range(img.height())
        if img.pixelColor(x, y).name() == MIST["accent"]
    )
    assert accent == 0, f"{accent} accent pixels on a disabled control"


def _render(widget) -> QImage:
    img = QImage(widget.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    widget.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return img


def test_list_variant_stacks_rows_and_down_arrow_moves_the_choice(qtbot):
    # The translation model list: rows stacked, Down moves the choice.
    # Seeded defects: lay the list out in a row, or drop Down from the key
    # handling -> red.
    seg = SegmentedControl(
        [SegmentOption("default", "Standard", "For 10 GB+ cards"),
         SegmentOption("small", "Smaller", "For 8 GB cards")],
        MIST, value="default", variant="list",
    )
    qtbot.addWidget(seg)
    seg.resize(360, seg.height())
    seg.show()
    first, second = seg.findChildren(QAbstractButton)
    assert second.y() >= first.y() + first.height()
    QTest.keyClick(first, Qt.Key_Down)
    assert seg.value() == "small"


def test_text_variant_draws_no_frame_and_lights_the_choice(qtbot):
    # Theme in the title band: bare text options, the chosen one accent
    # filled. Seeded defects: paint the group frame for the text variant, or
    # skip the lit fill -> red.
    seg = SegmentedControl(
        [SegmentOption("system", "System"), SegmentOption("light", "Light"),
         SegmentOption("dark", "Dark")],
        MIST, value="system", variant="text",
    )
    qtbot.addWidget(seg)
    seg.resize(seg.sizeHint())
    system, light, _dark = seg.findChildren(QAbstractButton)
    img = _render(seg)
    assert img.pixelColor(light.x() + 2, 2).alpha() == 0
    assert img.pixelColor(system.x() + 4, system.height() - 4).name() == MIST["accent"]


def test_a_programmatic_focus_move_keeps_the_ring_and_a_click_clears_it(qtbot):
    # A page switch moves focus with OtherFocusReason: the segment keeps the
    # ring state it had, lit by a Tab even after a mouse press elsewhere, and
    # unlit after a click on it (the ring: the on-accent inset line on the lit
    # segment). Seeded defects: drop OtherFocusReason from _NEUTRAL_FOCUS
    # (the latest press decides that move too) -> the press on the frame
    # unlights the Tab's ring -> red; let a mouse click keep an earlier ring
    # -> on-accent pixels on the ring line -> red.
    seg, (left, _center), _seen = _control(qtbot)
    seg.resize(160, seg.height())
    seg.show()
    # Focus events only reach an active window (offscreen too).
    seg.activateWindow()
    assert QTest.qWaitForWindowActive(seg)
    left.clearFocus()
    QTest.keyClick(seg, Qt.Key_Tab)  # keyboard use: the ring is on
    assert left.hasFocus() and left.ring_visible()
    QTest.mouseClick(seg, Qt.LeftButton, Qt.NoModifier, QPoint(1, 1))  # the frame: no focus move
    left.clearFocus()
    left.setFocus(Qt.OtherFocusReason)
    assert left.hasFocus() and left.ring_visible()
    left.clearFocus()
    QTest.mouseClick(left, Qt.LeftButton)
    assert left.hasFocus()
    left.clearFocus()
    left.setFocus(Qt.OtherFocusReason)
    assert left.hasFocus()
    img = _render(seg)
    ring_x = left.x() + 2
    column = {img.pixelColor(ring_x, y).name() for y in range(6, left.height() - 6)}
    assert MIST["on_accent"] not in column


def test_a_shortcut_after_a_click_lights_the_ring_on_the_segment_it_focuses(qtbot):
    # A key a shortcut takes is keyboard use too, though Qt never delivers
    # its KeyPress (only the ShortcutOverride sent ahead of the lookup):
    # after a click on the frame, F2 moves the focus to a segment and the
    # segment rings. No modifier: a Ctrl or Alt press is delivered and would
    # count by itself. Seeded defect: count only KeyPress as a key -> the
    # click decides -> no ring -> red.
    from PySide6.QtGui import QKeySequence, QShortcut

    seg, (left, _center), _seen = _control(qtbot)
    seg.resize(160, seg.height())
    shortcut = QShortcut(QKeySequence(Qt.Key_F2), seg)
    shortcut.activated.connect(lambda: left.setFocus(Qt.ShortcutFocusReason))
    seg.show()
    seg.activateWindow()
    assert QTest.qWaitForWindowActive(seg)
    QTest.mouseClick(seg, Qt.LeftButton, Qt.NoModifier, QPoint(1, 1))  # the frame: no focus move
    left.clearFocus()

    QTest.keyClick(seg, Qt.Key_F2)

    assert left.hasFocus() and left.ring_visible()


def test_wizard_list_sits_on_the_field_tone(qtbot):
    # Setup's device list: the group fills with the field tone (Settings'
    # with the ground). Seeded defect: ignore wizard=True -> the gap between
    # rows shows the ground -> red.
    seg = SegmentedControl(
        [SegmentOption("a", "Speakers", "System default"), SegmentOption("b", "Headphones")],
        MIST, value="a", variant="list", wizard=True,
    )
    qtbot.addWidget(seg)
    seg.resize(578, seg.height())
    seg.show()
    first, second = seg.findChildren(QAbstractButton)
    gap_y = first.y() + first.height()
    assert second.y() == gap_y + 2
    assert _render(seg).pixelColor(300, gap_y).name() == MIST["field"]


def test_tiles_share_one_60_px_row_equally_and_carry_their_japanese_in_the_name(qtbot):
    # Setup's Content step: four equal tiles in one row. Seeded defects:
    # stack the tiles, give them their text widths, or drop the sublabel
    # from the accessible name -> red.
    seg = SegmentedControl(
        [SegmentOption("variety", "Variety", "バラエティ"), SegmentOption("anime", "Anime", "アニメ"),
         SegmentOption("drama", "Drama", "ドラマ"), SegmentOption("news", "News", "ニュース")],
        MIST, value="variety", variant="tiles", wizard=True,
    )
    qtbot.addWidget(seg)
    seg.resize(578, seg.height())
    seg.show()
    tiles = seg.findChildren(QAbstractButton)
    assert seg.height() == 60
    assert {t.y() for t in tiles} == {3}
    assert max(t.width() for t in tiles) - min(t.width() for t in tiles) <= 1
    assert [t.accessibleName() for t in tiles] == [
        "Variety, バラエティ", "Anime, アニメ", "Drama, ドラマ", "News, ニュース"]


def test_cards_stack_62_px_rows_and_set_note_names_the_marked_card(qtbot):
    # Setup's Quality step: two stacked cards; "Recommended for your card"
    # lands on the fitting one and in its accessible name; segment() hands
    # out the row for its UI Automation id. Seeded defects: 36 px rows, a
    # note that never reaches the name, or segment() returning the wrong
    # row -> red.
    seg = SegmentedControl(
        [SegmentOption("default", "Standard", "Best translations. For cards with 10 GB or more."),
         SegmentOption("small", "Smaller", "Needs less video memory. For 8 GB cards.")],
        MIST, value="default", variant="cards", wizard=True,
    )
    qtbot.addWidget(seg)
    seg.resize(578, seg.height())
    seg.show()
    standard, smaller = seg.segment("default"), seg.segment("small")
    assert seg.height() == 132 and smaller.y() - standard.y() == 64
    seg.set_note("small", "Recommended for your card")
    assert smaller.accessibleName() == (
        "Smaller, Needs less video memory. For 8 GB cards., Recommended for your card")
    assert standard.accessibleName() == "Standard, Best translations. For cards with 10 GB or more."
