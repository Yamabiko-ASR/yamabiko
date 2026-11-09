"""Setup's Welcome step (opt-wiz-rail.tpl .welcome; design record Surface 3).

The 64 px app icon, "Welcome to Yamabiko" with a muted 山彦 (BIZ UDPMincho
26 px), the version and two lead paragraphs (a third on an update that
brings a new speech model), centered as one block in the content box above
the mockup's 64 px bottom padding. The page has no title of its own, so the
shell shows no header. No user input: ``isComplete()`` is always True; the
primary reads "Get started".
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWizard, QWizardPage

from yamabiko.overlay_app.controls import mincho_font, ui_font
from yamabiko.overlay_app.theme import MIST
from yamabiko.ui.error_window import TextBlock
from yamabiko.wizard_pages.parts import WelcomeIcon

TITLE = "Welcome to Yamabiko"
TITLE_JP = "山彦"
LEADS = (
    "Yamabiko listens to Japanese audio on your PC and shows subtitles with furigana and "
    "an English translation. Your audio never leaves your PC.",
    "Setup takes a few minutes, plus a one-time download of up to 7\u00a0GB.",
)
# One more lead when setup reopens because an update moved the speech model
# pin (wizard_state.speech_model_update_pending); {size} is the total the
# Download step fetches, for example "6.6 GB".
UPDATE_LEAD = "This update brings a new speech model. The download is {size}."
_LEAD_W = 540
# The mockup's .welcome .pb padding-bottom is 64 px; the shell's content box
# already ends 24 px above the footer.
_BOTTOM_INSET = 64 - 24


def display_version(version: str) -> str:
    """The version as setup shows it: "0.2.0-beta" -> "0.2.0 beta"."""
    return version.replace("-", " ")


class WelcomePage(QWizardPage):
    def __init__(self, *, app_version: str, palette: dict = MIST,
                 update_download: str | None = None) -> None:
        super().__init__()
        self.setButtonText(QWizard.WizardButton.NextButton, "Get started")
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, _BOTTOM_INSET)
        column.setSpacing(0)
        column.addStretch(1)
        self.icon = WelcomeIcon()
        column.addWidget(self.icon, 0, Qt.AlignLeft)
        column.addSpacing(24)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(12)
        self.title_label = TextBlock(TITLE, mincho_font(26), 34, palette)
        self.title_jp = TextBlock(TITLE_JP, mincho_font(26), 34, palette, color="muted")
        title_row.addWidget(self.title_label)
        title_row.addWidget(self.title_jp)
        title_row.addStretch(1)
        column.addLayout(title_row)
        column.addSpacing(4)
        self.version = TextBlock(f"Version {display_version(app_version)}", ui_font(13), 18,
                                 palette, color="muted")
        column.addWidget(self.version)
        texts = LEADS if update_download is None else (
            *LEADS, UPDATE_LEAD.format(size=update_download),
        )
        self.leads: list[TextBlock] = []
        for index, text in enumerate(texts):
            column.addSpacing(12 if index else 22)
            lead = TextBlock(text, ui_font(14), 21, palette, wrap="pretty")
            lead.setMaximumWidth(_LEAD_W)
            column.addWidget(lead)
            self.leads.append(lead)
        column.addStretch(1)

    def isComplete(self) -> bool:  # noqa: N802 — Qt convention
        return True
