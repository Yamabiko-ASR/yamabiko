"""The tray flyout widget (plan phase 4, Task 10), offscreen.

Sizes, the status precedence and tooltips the setters drive, the intents
the controls emit (and which of them close the panel), the series row, the
Position presets, the deactivate / Esc / popup rules and the live theme.
The DWM chrome and the native activation are in test_real_tray_flyout.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPixmap, QRegion, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from yamabiko.overlay_app.theme import MIST, NIGHT
from yamabiko.overlay_app.tray_flyout import PositionEntry, TrayFlyout, position_entries


def _flyout(qtbot, **kwargs) -> TrayFlyout:
    flyout = TrayFlyout(MIST, **kwargs)
    qtbot.addWidget(flyout)
    return flyout


def _open(flyout: TrayFlyout) -> None:
    flyout.open_at(QPoint(40, 40))
    QApplication.processEvents()


def test_flyout_is_360_by_426_or_358_without_the_series_row(qtbot):
    """Decisions 10, 34 and 51: the panel is 360 x 426 and its sections stack
    to it (Move subtitles… at 333, Quit at 387); without the series feature
    the row goes and the panel is 358 tall.
    Seeded defects: a section height or margin off by a pixel, the footer
    rule missing, or the series row kept -> red."""
    full = _flyout(qtbot)
    _open(full)
    assert (full.width(), full.height()) == (360, 426)
    assert full._move.mapTo(full, QPoint(0, 0)).y() == 333
    assert full._quit.mapTo(full, QPoint(0, 0)).y() == 387

    bare = _flyout(qtbot, show_series=False)
    _open(bare)
    assert (bare.width(), bare.height()) == (360, 358)
    assert bare._series_row is None
    assert bare._move.mapTo(bare, QPoint(0, 0)).y() == 265
    assert bare._quit.mapTo(bare, QPoint(0, 0)).y() == 319


def test_status_line_and_big_button_follow_the_state(qtbot):
    """Decision 26: problem > Paused > Subtitles hidden > Listening; the big
    button reads Pause, or Resume while paused; Decision 13: the tooltips
    carry the user's live combos and drop the parenthetical for an empty one.
    Seeded defects: hidden checked before paused, the problem ignored, the
    tile not following set_subtitles_shown, or a hard-coded combo -> red."""
    flyout = _flyout(qtbot)
    status = flyout._status
    assert (status.kind(), status.text()) == ("listening", "Listening")
    assert (flyout._pause.text(), flyout._pause.toolTip()) == ("Pause", "Pause")

    flyout.set_shortcut_labels(
        {"pause_resume": "Ctrl+Alt+K", "toggle_overlay": "", "show_series": "Ctrl+Alt+S"})
    assert flyout._pause.toolTip() == "Pause (Ctrl+Alt+K)"
    assert flyout._subtitles.toolTip() == "Show or hide subtitles"
    assert flyout._change.toolTip() == "Pick a series (Ctrl+Alt+S)"

    flyout.set_subtitles_shown(False)
    assert (status.kind(), status.text()) == ("hidden", "Subtitles hidden")
    assert (flyout._subtitles.value(), flyout._subtitles.is_lit()) == ("Hidden", False)
    flyout.set_paused(True)
    assert (status.kind(), status.text()) == ("paused", "Paused")
    assert (flyout._pause.text(), flyout._pause.toolTip()) == ("Resume", "Resume (Ctrl+Alt+K)")
    flyout.set_problem("Translations stopped")
    assert (status.kind(), status.text()) == ("problem", "Translations stopped")

    flyout.set_problem(None)
    flyout.set_paused(False)
    flyout.set_subtitles_shown(True)
    assert (status.kind(), status.text()) == ("listening", "Listening")
    assert (flyout._subtitles.value(), flyout._subtitles.is_lit()) == ("Shown", True)


def test_toggles_emit_their_choice_and_keep_the_panel_open(qtbot):
    """Decisions 11 and 50: Pause asks for the state its face shows (True
    from "Pause", False from "Resume") without changing the face, the
    Subtitles tile asks for a toggle without flipping itself (its truth is
    the overlay window's visibility), the Mode tile flips and reports the
    new mode, Genre and Alignment report their key; none of them closes the
    panel.
    Seeded defects: pauseRequested(False) from the Pause face, Pause or the
    Subtitles tile flipping locally, the Mode tile reporting the old mode,
    the genre order differing from the mockup, or a toggle that hides the
    panel -> red."""
    flyout = _flyout(qtbot)
    _open(flyout)
    seen: list = []
    flyout.pauseRequested.connect(lambda paused: seen.append(("pause", paused)))
    flyout.subtitlesToggleRequested.connect(lambda: seen.append("toggle"))
    flyout.modeChosen.connect(lambda mode: seen.append(("mode", mode)))
    flyout.genreChosen.connect(lambda genre: seen.append(("genre", genre)))
    flyout.alignmentChosen.connect(lambda value: seen.append(("alignment", value)))

    flyout._pause.click()
    assert flyout._pause.text() == "Pause"
    flyout.set_paused(True)
    flyout._pause.click()
    flyout._subtitles.click()
    flyout._mode.click()
    flyout._genre._segments[2].click()
    flyout._alignment._segments[1].click()

    assert seen == [("pause", True), ("pause", False), "toggle", ("mode", "learn"),
                    ("genre", "drama"), ("alignment", "center")]
    assert flyout._pause.text() == "Resume"
    assert flyout._subtitles.value() == "Shown"
    assert flyout._mode.value() == "Learn"
    assert flyout.isVisible()


def test_actions_close_the_panel_before_they_emit(qtbot):
    """Decision 50: Clear, Change…, a Position preset, Move subtitles…,
    Settings, both Troubleshooting items and Quit each hide the panel, then
    emit; the "Display N" row does nothing. Decision 74: only a pick from the
    open Position list applies; a wheel notch on the closed picker changes
    nothing, and a Down key or a typed letter opens the list.
    Seeded defects: an action that leaves the panel open or emits before
    hiding, the current-value row emitting, or the closed picker following
    the wheel, the Down key or typed text (a preset applied, the panel
    hidden) -> red."""
    flyout = _flyout(qtbot)
    flyout.set_series("Frieren (2023)", "葬送のフリーレン", QColor("#e4a15d"))
    flyout.set_positions(
        [PositionEntry("Display 2"), PositionEntry("Bottom of Display 1", "DISPLAY-A", 88.0)], 0)
    seen: list = []
    hidden_first: list = []

    def record(name):
        def slot(*args):
            hidden_first.append(not flyout.isVisible())
            seen.append((name, *args))
        return slot

    for name in ("clearSeriesRequested", "changeSeriesRequested",
                 "positionPresetChosen", "moveSubtitlesRequested", "settingsRequested",
                 "openLogFolderRequested", "saveDiagnosticsRequested", "quitRequested"):
        getattr(flyout, name).connect(record(name))
    menu = flyout.troubleshooting_menu.actions()
    for action in (flyout._clear.click, flyout._change.click,
                   lambda: flyout._position.activated.emit(0),
                   lambda: flyout._position.activated.emit(1),
                   flyout._move.click, flyout._settings.click,
                   menu[0].trigger, menu[1].trigger, flyout._quit.click):
        _open(flyout)
        action()

    assert seen == [
        ("clearSeriesRequested",), ("changeSeriesRequested",),
        ("positionPresetChosen", "DISPLAY-A", 88.0), ("moveSubtitlesRequested",),
        ("settingsRequested",), ("openLogFolderRequested",), ("saveDiagnosticsRequested",),
        ("quitRequested",),
    ]
    assert hidden_first == [True] * 8

    _open(flyout)
    picker = flyout._position
    notch = QWheelEvent(QPointF(20, 16), QPointF(picker.mapToGlobal(QPoint(20, 16))),
                        QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    QApplication.sendEvent(picker, notch)
    assert (picker.currentIndex(), flyout.isVisible()) == (0, True)
    qtbot.keyClick(picker, Qt.Key_Down)
    assert picker.view().isVisible(), "Down on the closed picker did not open the list"
    assert (picker.currentIndex(), flyout.isVisible()) == (0, True)
    assert len(seen) == 8, f"the closed picker applied a preset: {seen[8:]}"
    picker.hidePopup()
    qtbot.keyClicks(picker, "B")
    assert picker.view().isVisible(), "typing on the closed picker did not open the list"
    assert (picker.currentIndex(), flyout.isVisible()) == (0, True)
    assert len(seen) == 8, f"typing on the closed picker applied a preset: {seen[8:]}"
    picker.hidePopup()


def test_series_row_shows_the_title_without_year_and_no_series_disables_clear(qtbot):
    """Decision 28: line 1 is the label without "(year)", line 2 the Japanese
    title, the cover the palette's poster; no series: "No series" muted, no
    cover (a late cover update included), Clear disabled.
    Seeded defects: the year kept, the native line dropped, Clear left
    enabled, or a cover shown without a series -> red."""
    flyout = _flyout(qtbot)
    text = flyout._series_text
    assert (text.title(), text.muted(), flyout._clear.isEnabled()) == ("No series", True, False)
    assert flyout._cover.isHidden()

    poster = QPixmap(10, 14)
    poster.fill(QColor("#123456"))
    flyout.set_series("Frieren (2023)", "葬送のフリーレン", poster)
    assert (text.title(), text.subtitle(), text.muted()) == ("Frieren", "葬送のフリーレン", False)
    assert flyout._clear.isEnabled()
    assert flyout._cover.cover() is poster
    assert not flyout._cover.isHidden()

    flyout.set_series(None)
    flyout.set_cover(QColor("#e4a15d"))
    assert (text.title(), flyout._clear.isEnabled()) == ("No series", False)
    assert flyout._cover.isHidden()


def test_position_entries_mark_the_matching_preset_else_a_display_row():
    """Decisions 10, 29 and 55: Bottom and Middle per screen in Qt order,
    "primary" resolves to the primary screen, a preset matches only with its
    default horizontal anchors; otherwise "Display N" of the screen the
    overlay is on comes first and does nothing.
    Seeded defects: "primary" not resolved, the horizontal anchors ignored,
    or N counted from 0 -> red."""
    screens = [("DELL U2720Q", False), ("LG 27GP850", True)]
    entries, current = position_entries(screens, {
        "screen": "primary", "v_offset_pct": 55.0, "h_offset_pct": 50.0, "left_offset_pct": 6.0})
    assert [entry.label for entry in entries] == [
        "Bottom of Display 1", "Middle of Display 1", "Bottom of Display 2", "Middle of Display 2"]
    assert entries[current] == PositionEntry("Middle of Display 2", "LG 27GP850", 55.0)

    moved, current = position_entries(screens, {
        "screen": "DELL U2720Q", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
        "left_offset_pct": 20.0})
    assert (current, moved[0]) == (0, PositionEntry("Display 1"))
    assert moved[1:] == entries


def test_panel_hides_on_deactivate_and_esc_but_not_while_its_menu_is_open(qtbot, monkeypatch):
    """Decisions 7 and 54: a window deactivation hides the panel unless its
    own popup (the Troubleshooting menu or the Position list) is open; when
    that popup closes and the panel is no longer the active window, the
    panel hides; Esc hides it; every hide emits dismissed.
    Seeded defects: no deactivate branch, no popup guard, no hide after the
    menu or after the Position list, or no Esc handling -> red."""
    flyout = _flyout(qtbot)
    dismissed: list = []
    flyout.dismissed.connect(lambda: dismissed.append(True))

    _open(flyout)
    QApplication.sendEvent(flyout, QEvent(QEvent.Type.WindowDeactivate))
    assert not flyout.isVisible()

    _open(flyout)
    flyout._troubleshooting.click()
    menu = flyout.troubleshooting_menu
    qtbot.waitUntil(lambda: QApplication.activePopupWidget() is menu, timeout=2000)
    QApplication.sendEvent(flyout, QEvent(QEvent.Type.WindowDeactivate))
    assert flyout.isVisible()
    monkeypatch.setattr(flyout, "isActiveWindow", lambda: False)
    menu.hide()
    qtbot.waitUntil(lambda: not flyout.isVisible(), timeout=2000)

    flyout.set_positions([PositionEntry("Bottom of Display 1", "DISPLAY-A", 88.0)], 0)
    _open(flyout)
    panel_window = flyout.windowHandle()
    flyout._position.showPopup()
    # The open list takes the focus window (a queued platform event). Wait for
    # it while the list is open: delivered after the list closes, it would
    # deactivate, and so hide, the panel on its own.
    qtbot.waitUntil(lambda: QApplication.activePopupWidget() is not None
                    and QGuiApplication.focusWindow() not in (None, panel_window),
                    timeout=2000)
    QApplication.sendEvent(flyout, QEvent(QEvent.Type.WindowDeactivate))
    assert flyout.isVisible()
    flyout._position.hidePopup()
    qtbot.waitUntil(lambda: not flyout.isVisible(), timeout=2000)

    _open(flyout)
    qtbot.keyClick(flyout, Qt.Key_Escape)
    assert not flyout.isVisible()
    assert dismissed == [True] * 4


def test_position_picker_rings_only_for_keyboard_focus(qtbot):
    """Focus rings are keyboard-only, like the panel's other controls: the
    Position list opened with the mouse and closed leaves the focus on the
    picker, and a reopened panel hands it back; neither rings the picker's
    edge with the accent. Tab focus does.
    Seeded defect: the picker's edge following plain :focus (the window
    QSS's QComboBox:focus accent) -> the reopened panel's picker edge is the
    accent -> red."""
    flyout = _flyout(qtbot)
    flyout.set_series("Frieren (2023)", "葬送のフリーレン", QColor("#e4a15d"))
    flyout.set_positions([PositionEntry("Bottom of Display 1", "DISPLAY-A", 88.0)], 0)
    picker = flyout._position

    def edge() -> QColor:
        image = QImage(picker.size(), QImage.Format_ARGB32)
        image.fill(0)
        picker.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        return image.pixelColor(picker.width() // 2, 0)

    _open(flyout)
    qtbot.mouseClick(picker, Qt.LeftButton)
    qtbot.waitUntil(picker.view().isVisible, timeout=2000)
    picker.hidePopup()
    flyout.hide()
    other = QWidget()  # the window the user went to before reopening the panel
    qtbot.addWidget(other)
    other.show()
    other.activateWindow()
    qtbot.waitUntil(other.isActiveWindow, timeout=2000)
    _open(flyout)
    qtbot.waitUntil(picker.hasFocus, timeout=2000)
    assert edge() != QColor(MIST["accent"]), "the mouse-opened list left a ring on the picker"

    qtbot.keyClick(picker, Qt.Key_Backtab)
    qtbot.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    assert picker.hasFocus()
    assert edge() == QColor(MIST["accent"]), "Tab focus did not ring the picker"


def test_apply_palette_reskins_the_open_panel(qtbot):
    """Decision 10: the panel follows the app theme live. After
    apply_palette(NIGHT) the header sky, its ridgeline's mist, a field tile
    and the body show Night's tones.
    Seeded defects: apply_palette not reaching the header or the tiles, or
    the ridgeline's last layer not in the raised tone -> red."""
    flyout = _flyout(qtbot)
    flyout.set_subtitles_shown(False)  # the tile on its field tone
    _open(flyout)
    flyout.apply_palette(NIGHT)
    image = QImage(flyout.size(), QImage.Format_ARGB32)
    image.fill(0)
    flyout.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert image.pixelColor(4, 4) == QColor(NIGHT["ground"])      # header sky
    assert image.pixelColor(180, 83) == QColor(NIGHT["raised"])   # ridgeline mist
    assert image.pixelColor(30, 100) == QColor(NIGHT["field"])    # Subtitles tile
    assert image.pixelColor(4, 300) == QColor(NIGHT["raised"])    # body margin
