"""Setup's Done step (plan phase 7, Task 6), offscreen. Its guidance must
point at things that exist: the tray, the configured shortcuts."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")


def _texts(page) -> list[str]:
    from yamabiko.ui.error_window import TextBlock

    return [block.text() for block in page.findChildren(TextBlock) if not block.isHidden()]


def test_done_shows_the_spec_copy_and_start_yamabiko_and_no_state_json_line(qtbot):
    """Spec Surface 3 Done: "You're all set." (ASCII apostrophe, Decision
    21) / "Yamabiko runs from the tray, next to the clock.", primary
    "Start Yamabiko"; the state.json rerun line is dropped (README).

    Seeded defects: keep the old "Setup complete" copy or the state.json
    paragraph, or leave the Finish text -> red."""
    from PySide6.QtWidgets import QWizard

    from yamabiko.wizard_pages.done import DonePage

    page = DonePage()
    qtbot.addWidget(page)
    page.initializePage()
    assert (page.title(), page.subTitle()) == (
        "You're all set.", "Yamabiko runs from the tray, next to the clock.")
    assert page.buttonText(QWizard.WizardButton.FinishButton) == "Start Yamabiko"
    assert not any("state.json" in text for text in _texts(page))
    assert "Shortcuts" in _texts(page)


def test_done_lists_the_configured_shortcuts_and_hides_them_when_hotkeys_are_off(qtbot):
    """L11: the caps show the merged config's hotkeys in the Done order (a
    rerun keeps custom ones; an emptied one is left out); with
    hotkeys.enabled false the Shortcuts section hides.

    Seeded defects: show the shipped defaults, keep an empty row, or ignore
    hotkeys.enabled -> red."""
    from yamabiko import paths
    from yamabiko.overlay_app.keycaps import KeyCaps
    from yamabiko.wizard_pages.done import DonePage

    paths.ensure_dirs()
    overrides = paths.user_overrides_path()
    overrides.write_text(
        "hotkeys:\n  pause_resume: Ctrl+Shift+P\n  show_series: ''\n", encoding="utf-8")
    page = DonePage()
    qtbot.addWidget(page)
    page.initializePage()
    caps = [c.accessibleName() for c in page.findChildren(KeyCaps) if not c.isHidden()]
    assert caps == ["Ctrl Alt O", "Ctrl Shift P"]
    assert [t for t in _texts(page) if t != "Shortcuts"] == ["Show or hide subtitles", "Pause"]
    overrides.write_text("hotkeys:\n  enabled: false\n", encoding="utf-8")
    page.initializePage()
    assert page.shortcuts.isHidden()


def test_done_tray_drawing_reads_the_clock_when_the_page_shows(qtbot):
    """L11 / design Q10: the drawing's clock is the time and date when the
    page shows, in the system's short formats, not when setup started.

    Seeded defect: read the clock in __init__ only -> the earlier time ->
    red."""
    from PySide6.QtCore import QDateTime, QLocale, Qt

    from yamabiko.wizard_pages.done import DonePage

    clock = {"now": QDateTime.fromString("2026-10-27T21:41:00", Qt.ISODate)}
    page = DonePage(now=lambda: clock["now"])
    qtbot.addWidget(page)
    clock["now"] = QDateTime.fromString("2026-10-27T22:05:00", Qt.ISODate)
    page.initializePage()
    locale = QLocale.system()
    assert page.tray.clock_lines() == (
        locale.toString(clock["now"].time(), QLocale.FormatType.ShortFormat),
        locale.toString(clock["now"].date(), QLocale.FormatType.ShortFormat))


def test_done_leaves_out_pick_a_series_while_the_series_feature_is_off(qtbot):
    """Decision 143: the pipeline binds the series shortcut only while the
    series feature is on (gazetteer.feature_enabled: gazetteer and
    romanization both enabled), so Done must not offer Ctrl+Alt+S then.

    Seeded defect: list show_series whatever the feature switch -> red."""
    from yamabiko import paths
    from yamabiko.overlay_app.keycaps import KeyCaps
    from yamabiko.wizard_pages.done import DonePage

    paths.ensure_dirs()
    paths.user_overrides_path().write_text("romanization:\n  enabled: false\n", encoding="utf-8")
    page = DonePage()
    qtbot.addWidget(page)
    page.initializePage()
    caps = [c.accessibleName() for c in page.findChildren(KeyCaps) if not c.isHidden()]
    assert caps == ["Ctrl Alt O", "Ctrl Alt P"]
    assert [t for t in _texts(page) if t != "Shortcuts"] == ["Show or hide subtitles", "Pause"]


def test_done_leaves_out_a_whitespace_only_combo(qtbot):
    """A hand-edited combo of only spaces binds nothing, so Done leaves its
    row out, as the tray does (one rule: hotkeys.shortcut_labels).

    Seeded defect: Done tests the raw combo's truthiness -> a row without
    caps beside "Pick a series" -> red."""
    from yamabiko import paths
    from yamabiko.overlay_app.keycaps import KeyCaps
    from yamabiko.wizard_pages.done import DonePage

    paths.ensure_dirs()
    paths.user_overrides_path().write_text("hotkeys:\n  show_series: '  '\n", encoding="utf-8")
    page = DonePage()
    qtbot.addWidget(page)
    page.initializePage()
    caps = [c.accessibleName() for c in page.findChildren(KeyCaps) if not c.isHidden()]
    assert caps == ["Ctrl Alt O", "Ctrl Alt P"]
    assert [t for t in _texts(page) if t != "Shortcuts"] == ["Show or hide subtitles", "Pause"]
