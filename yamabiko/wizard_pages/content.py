"""Wizard step "Content": what the user watches most (the genre).

Four tiles (Variety, Anime, Drama, News with their Japanese names) and a
small subtitle sample on the night photo whose banner and line follow the
choice. The genre itself is persisted by the wizard (``accept()`` writes
``genre`` to user_overrides.yaml, ``state.json`` records it); this page
only holds the choice.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWizardPage

from yamabiko.overlay_app.controls import ui_font
from yamabiko.overlay_app.segmented import SegmentOption, SegmentedControl
from yamabiko.overlay_app.telop_look import GENRE_BANNER
from yamabiko.overlay_app.theme import MIST
from yamabiko.wizard_pages.parts import GENRE_ORDER, ToneSample


class ContentPage(QWizardPage):
    """The Content step: one genre out of ``parts.GENRE_ORDER``."""

    def __init__(self, *, palette: dict = MIST) -> None:
        super().__init__()
        self.setTitle("What do you watch most?")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 22, 0, 0)
        layout.setSpacing(0)

        # GENRE_BANNER maps a genre to (Japanese, English): the tile shows
        # the English label over the Japanese one (UIA "Variety, バラエティ").
        self.genre_control = SegmentedControl(
            [SegmentOption(g, GENRE_BANNER[g][1], GENRE_BANNER[g][0]) for g in GENRE_ORDER],
            palette, variant="tiles", wizard=True, value="variety",
        )
        self.genre_control.setFont(ui_font(14))
        layout.addWidget(self.genre_control)

        layout.addSpacing(10)
        help_line = QLabel("Helps the translator match the tone.")
        help_line.setObjectName("help")
        help_line.setFont(ui_font(13))
        layout.addWidget(help_line)

        layout.addSpacing(22)
        self.sample = ToneSample(genre="variety")
        layout.addWidget(self.sample)
        layout.addStretch(1)

        self.genre_control.valueChanged.connect(self.sample.set_genre)

    def selected_genre(self) -> str:
        return self.genre_control.value()

    def set_genre(self, genre: Optional[str]) -> None:
        """Seed the choice (cancelled session or merged config). A value the
        step doesn't know (a hand-edited genre) leaves the choice as it is."""
        if genre not in GENRE_ORDER:
            return
        self.genre_control.setValue(genre)
        self.sample.set_genre(genre)
