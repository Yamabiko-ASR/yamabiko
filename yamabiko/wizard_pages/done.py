"""Setup's Done step (opt-wiz-rail.tpl .done; design record Surface 3).

"You're all set." and "Yamabiko runs from the tray, next to the clock." (the
shell's header), a drawing of the notification area with the Yamabiko icon
and the time and date read when the page shows, then "Shortcuts": the
configured show/hide, pause and series shortcuts as key caps (a shortcut
turned off is left out, the series one also while the series feature is
off; the section hides when hotkeys are off). The primary
reads "Start Yamabiko". Saves nothing on its own: the wizard's accept()
persists. How to run setup again is in the README, not here.
"""
from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget, QWizard, QWizardPage

from yamabiko.hotkeys import shortcut_labels
from yamabiko.overlay_app.controls import mincho_font, ui_font
from yamabiko.overlay_app.keycaps import KeyCaps
from yamabiko.overlay_app.theme import MIST
from yamabiko.ui.error_window import TextBlock
from yamabiko.wizard_pages.parts import TrayDrawing

_log = logging.getLogger(__name__)

TITLE = "You're all set."
LEAD = "Yamabiko runs from the tray, next to the clock."
# (hotkeys.* key, description) in the Done order.
SHORTCUTS = (
    ("toggle_overlay", "Show or hide subtitles"),
    ("pause_resume", "Pause"),
    ("show_series", "Pick a series"),
)
# The shipped combos (default_config.yaml), standing in for a config that
# can't be read (Decision 49).
_SHIPPED_HOTKEYS = {"toggle_overlay": "Ctrl+Alt+O", "pause_resume": "Ctrl+Alt+P",
                    "show_series": "Ctrl+Alt+S"}


def configured_shortcuts() -> list[tuple[str, str]] | None:
    """(combo, description) of each shortcut the pipeline binds, in the Done
    order, from the merged config (a rerun may follow custom shortcuts):
    hotkeys.shortcut_labels, the tray's rule (an empty or whitespace-only
    combo is left out, and so is the series shortcut while the series
    feature is off, Decision 143). None when ``hotkeys.enabled`` is false.
    The shipped defaults stand in when the config can't be read."""
    try:
        from yamabiko.config import load_full_config

        cfg = load_full_config()
    except Exception:
        _log.warning("setup Done: config unreadable; showing the shipped shortcuts",
                     exc_info=True)
        cfg = {"hotkeys": dict(_SHIPPED_HOTKEYS)}
    labels = shortcut_labels(cfg)
    if labels is None:
        return None
    return [(labels[key], text) for key, text in SHORTCUTS if key in labels]


class DonePage(QWizardPage):
    def __init__(self, *, palette: dict = MIST,
                 now: Callable[[], QDateTime] = QDateTime.currentDateTime) -> None:
        super().__init__()
        self.setTitle(TITLE)
        self.setSubTitle(LEAD)
        self.setButtonText(QWizard.WizardButton.FinishButton, "Start Yamabiko")
        self._now = now
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 14, 0, 0)
        column.setSpacing(0)
        self.tray = TrayDrawing(palette)
        column.addWidget(self.tray, 0, Qt.AlignLeft)
        self.shortcuts = QWidget()
        block = QVBoxLayout(self.shortcuts)
        block.setContentsMargins(0, 28, 0, 0)
        block.setSpacing(12)
        block.addWidget(TextBlock("Shortcuts", mincho_font(15), 20, palette))
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        self._rows: list[tuple[KeyCaps, TextBlock]] = []
        for index, (key, text) in enumerate(SHORTCUTS):
            caps = KeyCaps(_SHIPPED_HOTKEYS[key], palette, size="large")
            label = TextBlock(text, ui_font(14), 20, palette)
            grid.addWidget(caps, index, 0, Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(label, index, 1, Qt.AlignVCenter)
            self._rows.append((caps, label))
        grid.setColumnStretch(1, 1)
        block.addLayout(grid)
        column.addWidget(self.shortcuts)
        column.addStretch(1)

    def initializePage(self) -> None:  # noqa: N802 — Qt convention
        """Read the clock and the shortcuts each time the page shows."""
        self.tray.set_time(self._now())
        rows = configured_shortcuts()
        self.shortcuts.setVisible(bool(rows))
        for index, (caps, label) in enumerate(self._rows):
            shown = rows is not None and index < len(rows)
            if shown:
                caps.set_sequence(rows[index][0])
                label.set_text(rows[index][1])
            caps.setVisible(shown)
            label.setVisible(shown)
