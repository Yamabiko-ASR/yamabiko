"""Content step (ContentPage): the genre choice and its tone sample.
Persistence is the shell's (tests/test_wizard_flow.py)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")


def test_genre_options_follow_the_spec_and_default_to_variety(qtbot):
    """Spec Content step: Variety, Anime, Drama, News in that order with
    their Japanese names (UIA "Variety, バラエティ" ...); Variety is chosen
    on a first run and the sample shows it.

    Seeded defect: build the options in state.VALID_GENRES order (variety,
    drama, news, anime) -> red.
    """
    from PySide6.QtWidgets import QAbstractButton

    from yamabiko.wizard_pages.content import ContentPage

    page = ContentPage()
    qtbot.addWidget(page)

    names = [b.accessibleName() for b in page.genre_control.findChildren(QAbstractButton)]
    assert names == ["Variety, バラエティ", "Anime, アニメ", "Drama, ドラマ", "News, ニュース"]
    assert page.selected_genre() == "variety"
    assert page.sample.genre() == "variety"


def test_picking_a_genre_moves_the_tone_sample(qtbot):
    """Spec: the sample's banner and line follow the choice.

    Seeded defect: leave the sample unconnected to valueChanged -> red.
    """
    from yamabiko.wizard_pages.content import ContentPage

    page = ContentPage()
    qtbot.addWidget(page)
    page.genre_control.segment("anime").click()

    assert page.selected_genre() == "anime"
    assert page.sample.genre() == "anime"


def test_an_unknown_saved_genre_keeps_variety(qtbot):
    """A hand-edited genre the app doesn't know must not break the step
    (the old combo ignored it); a known one moves choice and sample.

    Seeded defect: pass the seed straight to setValue -> ValueError -> red.
    """
    from yamabiko.wizard_pages.content import ContentPage

    page = ContentPage()
    qtbot.addWidget(page)
    page.set_genre("music")
    assert page.selected_genre() == "variety"

    page.set_genre("drama")
    assert page.selected_genre() == "drama"
    assert page.sample.genre() == "drama"
