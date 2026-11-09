"""Setup's Welcome step (plan phase 7, Task 6), offscreen."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")


def test_welcome_shows_the_spec_copy_the_display_version_and_get_started(qtbot):
    """Spec Surface 3 Welcome: "Welcome to Yamabiko 山彦", "Version 0.2.0
    beta" (from "0.2.0-beta"), the two leads, primary "Get started"; no
    page title, so the shell shows no header; always complete.

    Seeded defects: show the raw "0.2.0-beta", keep the old intro or the
    old title (a header would show) -> red."""
    from PySide6.QtWidgets import QWizard

    from yamabiko.wizard_pages.welcome import WelcomePage

    page = WelcomePage(app_version="0.2.0-beta")
    qtbot.addWidget(page)
    assert page.title() == ""
    assert (page.title_label.text(), page.title_jp.text()) == ("Welcome to Yamabiko", "山彦")
    assert page.version.text() == "Version 0.2.0 beta"
    assert [lead.accessibleName() for lead in page.leads] == [
        "Yamabiko listens to Japanese audio on your PC and shows subtitles with furigana and "
        "an English translation. Your audio never leaves your PC.",
        "Setup takes a few minutes, plus a one-time download of up to 7 GB.",
    ]
    assert page.buttonText(QWizard.WizardButton.NextButton) == "Get started"
    assert page.isComplete()


def test_welcome_block_sits_centered_above_the_mockups_bottom_padding(qtbot):
    """design §6 Welcome: the block (icon to the second lead) is centered in
    the 452 px content box less the mockup's extra 40 px bottom padding.

    Seeded defects: top-align the block, or drop the bottom inset -> the
    gaps above and below differ -> red."""
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.wizard_pages.welcome import WelcomePage

    register_bundled_fonts()  # offscreen has no system fonts
    page = WelcomePage(app_version="0.2.0-beta")
    qtbot.addWidget(page)
    page.resize(578, 452)
    page.show()
    page.layout().activate()
    last = page.leads[-1]
    above = page.icon.y()
    below = 452 - 40 - (last.y() + last.height())
    assert above > 40 and abs(above - below) <= 1, (above, below)
