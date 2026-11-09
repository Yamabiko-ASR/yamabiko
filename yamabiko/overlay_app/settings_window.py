"""The Settings window: "Studio under the mountain".

A non-modal ``QWidget`` (the tray shows it and keeps a single instance),
1000 x 728 and painted under its own title bar (window_chrome): the
landscape title band with the caption, the 山彦 heading with the seal and
Theme; the always-visible band with the preview monitor and the four
Overlay switcher rows; BIZ UDPMincho text tabs (Subtitles, Position, Timing,
Audio & model, Shortcuts 操作) over panels aligned to the band (two under the
monitor, one under the switcher); the footer with Reset to defaults, Cancel,
Apply and OK. At 1000 x 728 nothing scrolls. A work area too short for that
shrinks the window: the preview monitor gives up height first, then the
panels scroll; the title band and the footer always stay whole.

Signals:

* ``styleChanged(dict)`` -- emitted on Apply / OK with the full style dict.
  The tray forwards it to the pipeline as an ``apply_style`` command.
* ``stylePreview(dict)`` -- emitted on ANY style-control change, debounced
  150 ms, so a live drag doesn't flood the overlay. The tray shows a sample
  subtitle and restyles it live.
* ``previewEnded()`` -- emitted when the window closes, so the tray can
  drop the sample and restore the committed style.
* ``themeChanged(str)`` -- the chrome theme (``system`` / ``light`` /
  ``dark``) after Apply / OK commits it.
* ``autoGenreChanged(bool)`` -- the committed "Switch genre to Anime when
  you pick a series" switch (``gazetteer.auto_genre``); the switch exists
  only when the series feature is enabled.
* ``reduceMotionChanged(bool)`` -- the committed "Reduce motion" switch
  (``overlay.reduce_motion``), emitted BEFORE ``styleChanged``.
* ``positionApplied(dict)`` -- emitted after Apply / OK persists a changed
  Position page; the tray re-pins the live overlay without a second write.
* ``shortcutsChanged(dict)`` -- the ``hotkeys.*`` combos an Apply / OK just
  saved (changed keys only); the tray sends them to the pipeline as
  ``set_hotkeys``, which re-binds them at once.
* ``incrementalChanged(bool)`` -- the committed "Faster subtitles" switch
  (``asr.incremental.enabled``); the tray sends it to the pipeline as
  ``set_incremental``, which uses it from the next speech window.
* ``noticeRequested(str, str)`` -- (title, body) of a notice that must
  outlive the window (the OK flow's deferred recovery-download notice).

Every change is staged: Apply / OK persist one transaction first
(settings_transaction), then emit; Esc and Cancel discard. A failed write
shows "Couldn't save your settings" with Retry; a model switch downloads the
model in the compact download dialog before the transaction is written.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QRect, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QFontComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout,
    QWidget,
)

from yamabiko import hotkeys as _hotkeys

from .controls import (
    ChevronCombo, ChevronFontCombo, ColorField, LinkButton, Panel, StepDoubleSpin,
    StepSpin, Switch, TextTabs, labeled, mincho_font, ui_font,
)
from .model_fetch import TIER_INFO, size_text
from .position_config import DEFAULT_POSITION, V_OFFSET_RANGE, clamp_position
from .preview_monitor import MONITOR_H, MONITOR_W, PreviewMonitor
from .segmented import SegmentedControl, SegmentOption
from .settings_dialogs import ColorPickerDialog, SaveFailedDialog
from .settings_transaction import SettingsCommit
from .shortcut_field import ShortcutField
from .style_defaults import MAX_WIDTH_PCT_RANGE, STYLE_DEFAULTS, coerce_max_width_pct
from .telop_look import GENRE_BANNER
from .theme import palette_for
from .window_chrome import (
    ClientAreaFit, TitleBand, add_window_menu, apply_window_qss, expand_client_area,
    nonclient_event, repaint_nonclient, set_border_color,
)

_log = logging.getLogger(__name__)

WINDOW_W = 1000
WINDOW_H = 728
# The shortest the window gets: a 1920 x 1080 work area at 150% is 672 px,
# the window frame included. Down there the preview monitor is at its
# minimum and the panels scroll (see resizeEvent).
WINDOW_MIN_H = 600
_PANELS_MIN_H = 120  # the panel scroll area never gets shorter than this
_PAD = 16          # window side padding
_GUT = 12          # gutter between the band's and the pages' columns
_COL = 276         # (564 - 12) / 2: the two panel columns under the monitor

# Value controls a wheel turn would edit: in the panel scroll area they take
# the wheel only when focused (see _PanelScroll). QSlider, not
# QAbstractSlider: the combo popups' scroll bars keep their wheel.
_WHEEL_GUARDED = (QAbstractSpinBox, QComboBox, QSlider)

# Debounce window for the live preview signal. A control drag fires
# valueChanged on every step; without debounce the overlay would be
# restyled dozens of times per second.
_PREVIEW_DEBOUNCE_MS = 150

# Chrome theme whitelist -- the ONE copy in this file, shared by the Theme
# choice handler and the seed reader (config.load_overlay_config carries the
# canonical loader-side coercion; keep the two in sync).
_THEME_VALUES = ("system", "light", "dark")

_PAGES = (
    ("subtitles", "Subtitles", "字幕"),
    ("position", "Position", "位置"),
    ("timing", "Timing", "間"),
    ("audio", "Audio & model", "音声"),
    ("shortcuts", "Shortcuts", "操作"),
)
PAGE_NAMES = tuple(name for name, _en, _jp in _PAGES)
_THEME_OPTIONS = (
    SegmentOption("system", "System"),
    SegmentOption("light", "Light"),
    SegmentOption("dark", "Dark"),
)
_MODEL_HELP = ("Switching downloads the model if you don't have it yet.\n"
               "It takes effect the next time Yamabiko starts.")
_PLACEMENT_HELP = "Drag the sample into place. Enter saves, Esc cancels."
_PLACEMENT_UNAVAILABLE_HELP = "You can move the subtitles while Yamabiko is running."
_ANILIST_PLACEHOLDER = "Optional. Shows your watch list in series search"
_INCREMENTAL_HELP = (
    "Show each sentence as soon as it is stable. Off: wait for the speaker "
    "to pause (about 1 s slower, slightly fewer recognition errors)."
)

# The four Overlay choices (spec copy): the band's switcher rows.
_OVERLAY_CHOICES = (
    ("mode", "Mode", (SegmentOption("watch", "Watch", "English priority"),
                      SegmentOption("learn", "Learn", "Japanese priority"))),
    ("alignment", "Alignment", (SegmentOption("left", "Left"),
                                SegmentOption("center", "Center"))),
    ("echo", "Echo", (SegmentOption("priority", "Priority only"),
                      SegmentOption("both", "Both languages"))),
    ("banner", "Banner", (SegmentOption("jp", "日本語"),
                          SegmentOption("jp_en", "日本語 + English"))),
)
_ECHO_HELP = "Which language older lines keep as they fade."

_SHORTCUT_ROWS = (
    ("pause_resume", "Pause and resume"),
    ("toggle_overlay", "Show or hide subtitles"),
    ("show_series", "Pick a series"),
)
# Shipped defaults, used when default_config.yaml can't be read.
_FALLBACK_HOTKEYS = {"pause_resume": "Ctrl+Alt+P", "toggle_overlay": "Ctrl+Alt+O",
                     "show_series": "Ctrl+Alt+S"}
_SHORTCUTS_NOTE = "Work in any app while Yamabiko runs"
_TAKEN_TEXT = "Used by another app. Pick a different one."


def _load_cfg(cfg: Optional[dict]) -> Optional[dict]:
    """Return ``cfg`` as-is, or a FRESH merged-config load when it is None.

    Returns None on a load failure. This is the single swallow point the six
    ``_read_current_*`` seed readers funnel through (each keeps its own
    documented fallback default), and the None sentinel is what lets the
    Apply-time re-read in :meth:`SettingsWindow._collect_position` tells
    "fresh read failed" from "fresh read succeeded" -- a transient IO failure
    must not masquerade as a config full of shipped defaults.
    """
    if cfg is not None:
        return cfg
    try:
        from yamabiko import config as _config
        return _config.load_full_config()
    except Exception:
        return None


def _read_shipped_defaults() -> Optional[dict]:
    """The bundled defaults file itself (NOT the merged view), or None when
    it can't be read: the source of the Reset targets (the shipped endpoint
    and the shipped shortcuts)."""
    try:
        from yamabiko import config as _config
        from yamabiko import paths as _paths
        return _config._read_yaml(_paths.shipped_defaults_path())
    except Exception:
        return None


def _default_list_devices() -> list[dict]:
    """Production default: enumerate WASAPI loopback devices.

    Imported lazily so tests that inject a fake enumerator don't require
    PyAudioWPatch at collection time.
    """
    from yamabiko.wizard_pages.audio import _default_list_devices as _impl
    return _impl()


def _screen_label(index: int, screen, primary: bool) -> str:
    """"Display 2 (2560 × 1440, primary)": the screen's place in Qt's screen
    order (QScreen.name() is the monitor's friendly name on Qt 6.11, not
    \\\\.\\DISPLAYn) and its physical size."""
    ratio = screen.devicePixelRatio()
    size = screen.geometry().size()
    width, height = round(size.width() * ratio), round(size.height() * ratio)
    return f"Display {index + 1} ({width} × {height}{', primary' if primary else ''})"


class _PageStack(QStackedWidget):
    """The tab pages. QStackedWidget sizes itself for its tallest page; this
    one reports the CURRENT page's height (and keeps the widest page's
    width), so the panel scroll area scrolls only a page that does not fit,
    not every page because the tallest one does not."""

    def __init__(self) -> None:
        super().__init__()
        # The scroll area re-reads the hints on the LayoutRequest this posts.
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        hint = super().sizeHint()
        page = self.currentWidget()
        if page is not None:
            hint.setHeight(page.sizeHint().height())
        return hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        hint = super().minimumSizeHint()
        page = self.currentWidget()
        if page is not None:
            hint.setHeight(page.minimumSizeHint().height())
        return hint

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        return max(page.heightForWidth(width), page.minimumSizeHint().height())


class _PanelScroll(QScrollArea):
    """The tab pages' frameless scroll area. At the design size the pages
    fit and no bar shows; on a short work area, once the preview monitor is
    at its minimum, the vertical bar appears (styled by the window QSS). As
    wide as the pages need (the horizontal bar stays off), never shorter
    than _PANELS_MIN_H. A wheel turn over an unfocused spin box, combo box
    or slider scrolls the panels instead of changing the value."""

    def __init__(self, pages: QWidget) -> None:
        super().__init__()
        self.setObjectName("panelScroll")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setWidget(pages)
        # The window's ground shows through: setWidget turned the pages' own
        # fill on, and the viewport fills with the Base role.
        pages.setAutoFillBackground(False)
        self.viewport().setAutoFillBackground(False)
        # The wheel no longer gives a value control focus, and an unfocused
        # one passes its wheel to the panels (eventFilter); a field the user
        # clicked or tabbed into keeps its wheel.
        for kind in _WHEEL_GUARDED:
            for control in pages.findChildren(kind):
                control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                control.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        # Type first: cheap, and safe while the window tears down.
        if (event.type() == QEvent.Type.Wheel
                and isinstance(watched, _WHEEL_GUARDED) and not watched.hasFocus()):
            self.wheelEvent(event)
            return True
        return super().eventFilter(watched, event)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        pages = self.widget()
        width = pages.minimumSizeHint().width() if pages is not None else 0
        return QSize(width, _PANELS_MIN_H)


class SettingsWindow(QWidget):
    styleChanged = Signal(dict)
    stylePreview = Signal(dict)
    previewEnded = Signal()
    themeChanged = Signal(str)
    positionApplied = Signal(dict)
    autoGenreChanged = Signal(bool)
    reduceMotionChanged = Signal(bool)
    noticeRequested = Signal(str, str)  # (title, body) -> tray toast
    shortcutsChanged = Signal(dict)  # changed hotkeys.* combos, after Apply / OK
    incrementalChanged = Signal(bool)  # committed Faster subtitles, after Apply / OK

    def __init__(
        self,
        current: Optional[dict] = None,
        parent=None,
        *,
        list_devices: Optional[Callable[[], list[dict]]] = None,
        config_path: Optional[Path] = None,
        initial_config: Optional[dict] = None,
        read_only: bool = False,
        on_move_subtitles: Optional[Callable[[], None]] = None,
        quant_downloader=None,
        banner: Optional[tuple[str, str]] = None,
        color_picker: Optional[Callable] = None,
        save_failed: Optional[Callable] = None,
        page: Optional[str] = None,
        hotkey_probe: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(parent)
        self.setObjectName("settingsWindow")
        self.setWindowTitle("Yamabiko settings")
        expand_client_area(self)
        self.setFont(ui_font(13))
        # An explicit minimum height (the layout still sets the minimum
        # width): a short work area may shrink the window below the layout's
        # own floor, and resizeEvent decides what gives.
        self.setMinimumHeight(WINDOW_MIN_H)
        if current is None:
            current = dict(STYLE_DEFAULTS)
        # Clamp the seed through the SAME ingestion path the config loader
        # and tray use: a garbage style reaching the numeric seeds below would
        # raise int(None)/setValue(None).
        from ..config import _clamp_style
        self._working = _clamp_style(dict(current))
        self._list_devices = list_devices if list_devices is not None else _default_list_devices
        self._config_path = config_path
        self._read_only = bool(read_only)
        self._on_move_subtitles = on_move_subtitles
        # The overlay's current banner (genre or series), so the preview
        # shows the same banner the live overlay does.
        self._banner = banner or GENRE_BANNER["variety"]
        # Injected _Downloader for the quant-tier acquisition path (fake at
        # the seam in tests); the in-flight download dialog is tracked so the
        # tray's quit-time release_threads can detach its worker.
        self._quant_downloader = quant_downloader
        self._active_quant_dialog = None
        self._last_quant_error: Optional[str] = None
        self._recovery_notice: Optional[str] = None
        # Dialog seams (tests inject fakes; production shows the themed ones).
        self._pick_color_dialog = color_picker or ColorPickerDialog.pick
        self._ask_retry_save = save_failed or SaveFailedDialog.ask
        self._closed = False
        self._fitted = False
        # Every palette-aware painted control, re-skinned by apply_palette.
        self._themed_widgets: list = []

        # One merged-config read per window open, threaded through every
        # _read_current_* seed helper (initial_config comes from
        # explicit-config composition so the window cannot seed from ambient
        # user state).
        self._boot_cfg = _load_cfg(initial_config)
        self._theme_value = self._read_current_theme(self._boot_cfg)
        self._reduce_motion = self._read_current_reduce_motion(self._boot_cfg)
        self._auto_genre = self._read_current_auto_genre(self._boot_cfg)
        self._incremental = self._read_current_incremental(self._boot_cfg)
        self._initial_theme = self._theme_value
        self._initial_reduce_motion = self._reduce_motion
        self._initial_auto_genre = self._auto_genre
        self._initial_incremental = self._incremental
        # The series feature's effective switch (feature_enabled: BOTH
        # gazetteer.enabled and romanization.enabled) gates the Series panel.
        self._gaz_feature_enabled = self._read_current_feature_enabled(self._boot_cfg)
        self._palette = palette_for(self._theme_value)
        # "Used by another app": the Win32 probe, looked up at call time so
        # tests can pin it (tests/overlay_app/conftest.py).
        self._hotkey_probe = hotkey_probe or (lambda combo: _hotkeys.combo_taken(combo))
        self._shortcut_defaults = self._shipped_default_hotkeys()
        self._initial_shortcuts = self._read_current_hotkeys(self._boot_cfg, self._shortcut_defaults)
        # Rows re-recorded since the last persisted commit: sent again even
        # when the combo equals the saved one (FR1-1).
        self._rerecorded: set[str] = set()
        # True while two rows share a combo; Apply and OK stay disabled.
        self._duplicate_shortcut = False

        self._title_band = self._themed(TitleBand(
            "Yamabiko settings", "山彦", "Yamabiko settings", self._palette))
        self._title_band.set_slot(self._build_theme_slot())
        band = self._build_band()
        self._tabs = self._themed(TextTabs([(en, jp) for _n, en, jp in _PAGES], self._palette))
        self._stack = _PageStack()
        self._stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        for widget in self._build_pages():
            self._stack.addWidget(widget)
        footer = self._build_footer()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._title_band)
        root.addWidget(band)
        root.addWidget(self._tabs)
        self._panel_scroll = _PanelScroll(self._stack)
        root.addWidget(self._panel_scroll, 1)
        root.addWidget(footer)
        self._client_fit = ClientAreaFit(self, root)

        # Enter on a focused push button clicks THAT button (QDialog-style
        # autoDefault; the painted LinkButton and ColorField click themselves
        # on Enter too); OK is the default everywhere else.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(True)
        self._ok_btn.setDefault(True)
        if self._read_only:
            self._apply_read_only()

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._emit_preview)
        # Set by an appearance change to defer the monitor rebuild to the
        # debounce; timing controls never set it.
        self._preview_rebuild_pending = False

        # Seed the preview and theme now that every widget exists, THEN wire
        # the change signals so seeding emits nothing.
        self._rebuild_preview()
        self.apply_palette(self._palette)
        self._connect_change_signals()
        self.resize(WINDOW_W, WINDOW_H)
        self._refresh_shortcut_warnings()
        if page is not None:
            self.show_page(page)

    def show_page(self, name: str) -> None:
        """Switch to a page by name ("subtitles", "position", "timing",
        "audio", "shortcuts"); the tray's notices open Settings here."""
        if name in PAGE_NAMES:
            self._tabs.setCurrentIndex(PAGE_NAMES.index(name))

    def adopt_overlay_choice(self, key: str, value: str) -> None:
        """The tray flyout saved an Overlay choice (Mode or Alignment) while
        this window is open: show it here, so the next Apply / OK writes it
        and not the value this window opened with (phase-4 Decision 68).
        Rebuilds the preview strip without a stylePreview: the live overlay
        already shows the real change."""
        control = self._overlay_choices.get(key)
        if control is None or control.value() == value:
            return
        control.setValue(value)
        self._rebuild_preview()

    def adopt_position(self, position: dict) -> None:
        """The overlay was moved from outside this window (a flyout Position
        preset, or a placement save) while it is open: show the monitor and
        the height and make them the baseline, so the next Apply / OK writes
        no position the user did not set here (phase-4 Decision 68)."""
        screen = position.get("screen", "primary")
        index = self._monitor_combo.findData(screen)
        primary = QGuiApplication.primaryScreen()
        if index < 0 and primary is not None and screen == primary.name():
            index = self._monitor_combo.findData("primary")
        if index >= 0:
            self._monitor_combo.setCurrentIndex(index)
        self._voffset.setValue(
            int(round(float(position.get("v_offset_pct", self._voffset.value()))))
        )
        self._initial_position_screen = self._monitor_combo.currentData() or "primary"
        self._initial_position_v = self._voffset.value()
        self._last_good_position = dict(position)

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------

    def _themed(self, widget):
        """Register a palette-aware painted control; returns it."""
        self._themed_widgets.append(widget)
        return widget

    def _build_theme_slot(self) -> QWidget:
        slot = QWidget()
        row = QHBoxLayout(slot)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label = QLabel("Theme")
        label.setObjectName("muted")
        self._theme_choice = self._themed(SegmentedControl(
            _THEME_OPTIONS, self._palette, value=self._theme_value, variant="text"))
        label.setBuddy(self._theme_choice)
        row.addWidget(label)
        row.addWidget(self._theme_choice)
        return slot

    def _band_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("bandTitle")
        label.setFont(mincho_font(15))
        label.setFixedHeight(20)
        return label

    def _build_band(self) -> QWidget:
        """The always-visible band: the preview monitor and the Overlay
        switcher rows beside it."""
        band = QWidget()
        band.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(band)
        grid.setContentsMargins(_PAD, 10, _PAD, 16)
        grid.setHorizontalSpacing(_GUT)
        grid.setVerticalSpacing(7)
        grid.addWidget(self._band_title("Preview"), 0, 0)
        grid.addWidget(self._band_title("Overlay"), 0, 1)
        self._preview = PreviewMonitor(dict(self._working), banner=self._banner)
        # A short window shrinks the monitor (resizeEvent); it stays at the
        # left of its 564 px column, so the switcher and the panels below
        # keep their columns.
        grid.setColumnMinimumWidth(0, MONITOR_W)
        grid.addWidget(self._preview, 1, 0, Qt.AlignLeft | Qt.AlignTop)
        self._switcher = self._build_switcher()
        grid.addWidget(self._switcher, 1, 1)
        grid.setColumnStretch(1, 1)
        return band

    def _build_switcher(self) -> QFrame:
        """Mode, Alignment, Echo and Banner as Lit segmented rows; the rows
        spread over the monitor's height and close up on a short screen."""
        panel = QFrame()
        panel.setObjectName("panel")
        grid = QGridLayout(panel)
        grid.setContentsMargins(14, 12, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(0)
        self._overlay_choices: dict[str, SegmentedControl] = {}
        row = 0
        for index, (key, text, options) in enumerate(_OVERLAY_CHOICES):
            if index:
                grid.setRowStretch(row, 1)
                grid.setRowMinimumHeight(row, 4)
                row += 1
            control = self._themed(SegmentedControl(options, self._palette, value=self._working[key]))
            self._overlay_choices[key] = control
            label = QLabel(text)
            label.setFixedHeight(control.height())
            label.setBuddy(control)
            grid.addWidget(label, row, 0, Qt.AlignTop)
            if key == "echo":
                cell = QVBoxLayout()
                cell.setContentsMargins(0, 0, 0, 0)
                cell.setSpacing(5)
                cell.addWidget(control)
                self._echo_help = QLabel(_ECHO_HELP)
                self._echo_help.setObjectName("help")
                self._echo_help.setWordWrap(True)
                cell.addWidget(self._echo_help)
                grid.addLayout(cell, row, 1)
            else:
                grid.addWidget(control, row, 1, Qt.AlignTop)
            row += 1
        grid.setColumnStretch(1, 1)
        return panel

    @staticmethod
    def _page_grid(page: QWidget) -> QGridLayout:
        grid = QGridLayout(page)
        grid.setContentsMargins(_PAD, 14, _PAD, 14)
        grid.setSpacing(_GUT)
        grid.setColumnMinimumWidth(0, _COL)
        grid.setColumnMinimumWidth(1, _COL)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(1, 1)
        return grid

    @staticmethod
    def _place(grid: QGridLayout, panel: Panel, column: int, span: int = 1) -> None:
        """Put ``panel`` in row 0: columns 0 and 1 are the monitor's two
        276 px columns, column 2 sits under the switcher."""
        panel.finish()
        if column < 2:
            panel.setFixedWidth(_COL * span + _GUT * (span - 1))
        grid.addWidget(panel, 0, column, 1, span)

    def _spin(self, lo: int, hi: int, value: int, suffix: str = "", step: int = 1) -> StepSpin:
        spin = self._themed(StepSpin(self._palette))
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _slider_row(self, lo: int, hi: int, value: int,
                    template: str) -> tuple[QHBoxLayout, QSlider, QLabel]:
        """``[slider | template.format(v)]``; the label stays live here so the
        two slider rows can't drift."""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        slider.setFixedHeight(30)
        label = QLabel(template.format(slider.value()))
        slider.valueChanged.connect(lambda v: label.setText(template.format(v)))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        row.addWidget(slider, 1)
        row.addWidget(label)
        return row, slider, label

    @staticmethod
    def _pair(left, right, *, gap: int = 10) -> QHBoxLayout:
        """Two fields side by side, sharing the width equally: the fields'
        own width hints are ignored (a double spin's wider hint would
        otherwise take more than its half); labels keep theirs."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(gap)
        for item in (left, right):
            fields = [item] if isinstance(item, QWidget) else [
                item.itemAt(i).widget() for i in range(item.count())]
            for field in fields:
                if field is not None and not isinstance(field, QLabel):
                    policy = field.sizePolicy()
                    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
                    field.setSizePolicy(policy)
            if isinstance(item, QWidget):
                row.addWidget(item, 1)
            else:
                row.addLayout(item, 1)
        return row

    def _build_pages(self) -> list[QWidget]:
        return [self._build_subtitles_page(), self._build_position_page(),
                self._build_timing_page(), self._build_audio_page(),
                self._build_shortcuts_page()]

    def _build_subtitles_page(self) -> QWidget:
        page = QWidget()
        grid = self._page_grid(page)

        # (configured family, family the combo shows) per font key: see
        # _seed_font / _font_family.
        self._font_seed: dict[str, tuple[str, str]] = {}
        fonts = Panel("Fonts")
        self._jp_font = self._themed(ChevronFontCombo(self._palette))
        self._seed_font("jp_font", self._jp_font)
        self._jp_size = self._spin(8, 48, self._working["jp_size"], " pt")
        self._en_font = self._themed(ChevronFontCombo(self._palette))
        self._seed_font("en_font", self._en_font)
        self._en_size = self._spin(8, 72, self._working["en_size"], " pt")
        # A point-size copy of the window font (9.75 pt is 13 px at 96 dpi):
        # under the windows11 style a QFontComboBox with a pixel-size font
        # logs "QFont::setPointSize: Point size <= 0 (-1)".
        combo_font = ui_font()
        combo_font.setPointSizeF(9.75)
        for font, size, name in ((self._jp_font, self._jp_size, "Japanese font"),
                                 (self._en_font, self._en_size, "English font")):
            row = QHBoxLayout()
            row.setSpacing(8)
            font.setFont(combo_font)
            row.addLayout(labeled(name, font), 1)
            size.setFixedWidth(84)
            row.addLayout(labeled("Size", size), 0)
            fonts.rows.addLayout(row)
        self._place(grid, fonts, 0)

        colors = Panel("Colors")
        self._text_color_field = self._themed(ColorField(self._working["text_color"], self._palette))
        self._bg_color_field = self._themed(ColorField(self._working["bg_color"], self._palette))
        colors.rows.addLayout(self._pair(labeled("Text color", self._text_color_field),
                                         labeled("Background", self._bg_color_field)))
        # Shown in percent; stored 0..255. An untouched field keeps the exact
        # stored value (see _opacity_value).
        stored_opacity = self._working["bg_opacity"]
        self._opacity_seed = (stored_opacity, round(stored_opacity * 100 / 255))
        self._bg_opacity = self._spin(0, 100, self._opacity_seed[1], " %")
        self._padding = self._spin(0, 40, self._working["padding"], " px")
        colors.rows.addLayout(self._pair(labeled("Background opacity", self._bg_opacity),
                                         labeled("Padding", self._padding)))
        mw_row, self._max_width, self._max_width_label = self._slider_row(
            *MAX_WIDTH_PCT_RANGE, coerce_max_width_pct(self._working.get("max_width_pct")),
            "{}% of screen width")
        max_width = QVBoxLayout()
        max_width.setSpacing(4)
        max_width.addWidget(QLabel("Max width"))
        max_width.addLayout(mw_row)
        colors.rows.addLayout(max_width)
        self._place(grid, colors, 1)

        # Series: Apply/OK-gated like everything else, and built ONLY when the
        # series feature is enabled (with it off the keys control nothing).
        self._auto_genre_cb: Switch | None = None
        self._anilist_username_edit: QLineEdit | None = None
        self._initial_anilist_username = ""
        if self._gaz_feature_enabled:
            series = Panel("Series", note="AniList")
            self._anilist_username_edit = QLineEdit()
            self._anilist_username_edit.setPlaceholderText(_ANILIST_PLACEHOLDER)
            self._initial_anilist_username = self._read_current_anilist_username(self._boot_cfg)
            self._anilist_username_edit.setText(self._initial_anilist_username)
            series.rows.addLayout(labeled("AniList username", self._anilist_username_edit))
            self._auto_genre_cb = self._themed(Switch(
                "Switch genre to Anime when you pick a series", self._palette))
            self._auto_genre_cb.setChecked(self._auto_genre)
            series.rows.addWidget(self._auto_genre_cb)
            self._place(grid, series, 2)
        return page

    def _build_position_page(self) -> QWidget:
        page = QWidget()
        grid = self._page_grid(page)
        current_pos = self._read_current_position(self._boot_cfg)

        screen_panel = Panel("Screen")
        self._monitor_combo = self._themed(ChevronCombo(self._palette))
        primary = QGuiApplication.primaryScreen()
        for index, screen in enumerate(QGuiApplication.screens()):
            is_primary = screen is primary
            self._monitor_combo.addItem(_screen_label(index, screen, is_primary),
                                        userData="primary" if is_primary else screen.name())
        saved_screen = current_pos.get("screen", "primary")
        _midx = self._monitor_combo.findData(saved_screen)
        if _midx < 0 and primary is not None and saved_screen == primary.name():
            _midx = self._monitor_combo.findData("primary")
        if _midx < 0:
            # The saved screen is not connected (monitor unplugged, laptop
            # undocked). Seeding to "primary" would make an UNTOUCHED combo
            # read as a deliberate pick: a later v-only Apply would rewrite
            # the pin. Surface the saved name as its own "(disconnected)" row
            # so only a deliberate monitor change rewrites it.
            self._monitor_combo.addItem(f"{saved_screen} (disconnected)", userData=saved_screen)
            _midx = self._monitor_combo.count() - 1
        self._monitor_combo.setCurrentIndex(_midx)
        vo_row, self._voffset, self._voffset_label = self._slider_row(
            int(V_OFFSET_RANGE[0]), int(V_OFFSET_RANGE[1]),
            int(current_pos.get("v_offset_pct", DEFAULT_POSITION["v_offset_pct"])),
            "{}% down the screen")
        vertical = QVBoxLayout()
        vertical.setSpacing(4)
        vertical.addWidget(QLabel("Vertical position"))
        vertical.addLayout(vo_row)
        screen_panel.rows.addLayout(self._pair(labeled("Monitor", self._monitor_combo),
                                               vertical, gap=20))
        self._place(grid, screen_panel, 0, span=2)

        placement = Panel("Placement")
        self._move_btn = QPushButton("Move subtitles…")
        self._move_btn.clicked.connect(self._on_move_clicked)
        self._move_btn.setEnabled(self._on_move_subtitles is not None)
        placement.rows.addWidget(self._move_btn, 0, Qt.AlignLeft)
        help_label = QLabel(
            _PLACEMENT_HELP if self._on_move_subtitles is not None
            else _PLACEMENT_UNAVAILABLE_HELP
        )
        help_label.setObjectName("help")
        help_label.setWordWrap(True)
        placement.rows.addWidget(help_label)
        self._place(grid, placement, 2)

        # Change guards seeded from the widgets (not the raw read) so an
        # unrelated Apply is a no-op on disk and a disconnected screen's pin
        # survives; see _collect_position.
        self._initial_position_screen = self._monitor_combo.currentData() or "primary"
        self._initial_position_v = self._voffset.value()
        # Last-good position: this boot read, advanced after every successful
        # Apply; the fallback when the Apply-time fresh read fails.
        self._last_good_position = dict(current_pos)
        return page

    def _build_timing_page(self) -> QWidget:
        page = QWidget()
        grid = self._page_grid(page)
        lines = Panel("Lines")
        self._max_lines = self._spin(1, 6, self._working["max_lines"])
        # Shown in seconds; stored in ms (see _hold_value).
        stored_hold = self._working["hold_ms"]
        self._hold_seed = (stored_hold, round(stored_hold / 1000, 1))
        self._hold = self._themed(StepDoubleSpin(self._palette))
        self._hold.setDecimals(1)
        self._hold.setRange(0.5, 15.0)
        self._hold.setSingleStep(0.1)
        self._hold.setValue(self._hold_seed[1])
        self._hold.setSuffix(" s")
        lines.rows.addLayout(self._pair(labeled("Lines on screen", self._max_lines),
                                        labeled("Time on screen", self._hold)))
        self._place(grid, lines, 0)

        fades = Panel("Fades")
        self._fi = self._spin(0, 1000, self._working["fade_in_ms"], " ms", step=10)
        self._fo = self._spin(0, 2000, self._working["fade_out_ms"], " ms", step=10)
        fades.rows.addLayout(self._pair(labeled("Fade in", self._fi), labeled("Fade out", self._fo)))

        motion = Panel("Motion")
        self._reduce_motion_cb = self._themed(Switch("Reduce motion", self._palette))
        self._reduce_motion_cb.setChecked(self._reduce_motion)
        motion.rows.addWidget(self._reduce_motion_cb)
        # Column 1 stacks Fades over Motion: Motion and Speed are too tall for
        # one column at 1000 x 728, and Speed's help line needs the wide one.
        fades.finish()
        motion.finish()
        column = QWidget()
        column.setFixedWidth(_COL)
        stack = QVBoxLayout(column)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(_GUT)
        stack.addWidget(fades)
        stack.addWidget(motion, 1)  # takes the row's spare height: bottoms align
        grid.addWidget(column, 0, 1)

        # Faster subtitles (asr.incremental.enabled): Apply/OK-gated like
        # everything else; the pipeline uses it from its next speech window.
        speed = Panel("Speed")
        self._incremental_cb = self._themed(Switch("Faster subtitles", self._palette))
        self._incremental_cb.setChecked(self._incremental)
        speed.rows.addWidget(self._incremental_cb)
        incremental_help = QLabel(_INCREMENTAL_HELP)
        incremental_help.setObjectName("help")
        incremental_help.setWordWrap(True)
        speed.rows.addWidget(incremental_help)
        self._place(grid, speed, 2)
        return page

    def _build_audio_page(self) -> QWidget:
        page = QWidget()
        grid = self._page_grid(page)
        connections = Panel("Connections")
        self.audio_device_combo = self._themed(ChevronCombo(self._palette))
        try:
            devices = self._list_devices() or []
        except Exception:
            devices = []
        for dev in devices:
            name = dev.get("name", "") if isinstance(dev, dict) else str(dev)
            idx = dev.get("index") if isinstance(dev, dict) else None
            self.audio_device_combo.addItem(name, userData=idx)
        # Seed to the SAVED audio.device and remember the initial index so a
        # no-touch save can't clobber the saved choice (see _collect_commit).
        self._initial_device = self._read_current_audio_device(self._boot_cfg)
        _didx = self.audio_device_combo.findText(self._initial_device)
        # Row of the "<saved> (unavailable)" placeholder, or -1. Its label is
        # never saved: picking it back maps to the saved name itself.
        self._unavailable_device_row = -1
        self._unavailable_device_name = ""
        if _didx >= 0:
            self.audio_device_combo.setCurrentIndex(_didx)
        elif self._initial_device != "default":
            self.audio_device_combo.insertItem(
                0, f"{self._initial_device} (unavailable)", userData=None)
            self._unavailable_device_row = 0
            self._unavailable_device_name = self._initial_device
            self.audio_device_combo.setCurrentIndex(0)
        self._initial_device_index = self.audio_device_combo.currentIndex()
        self._audio_device_warning = QLabel("")
        self._audio_device_warning.setObjectName("help")
        self._audio_device_warning.setTextFormat(Qt.PlainText)
        self._audio_device_warning.setWordWrap(True)
        if _didx < 0 and self._initial_device != "default":
            self._audio_device_warning.setText(
                f'Saved device "{self._initial_device}" is unavailable. '
                "Yamabiko is using the runtime default; choose another "
                "device to replace the saved setting."
            )
        else:
            self._audio_device_warning.hide()
        self.llama_endpoint_edit = QLineEdit()
        self.llama_endpoint_edit.setObjectName("mono")
        self.llama_endpoint_edit.setPlaceholderText("http://localhost:8090/v1/chat/completions")
        # Seed with the endpoint in effect so an unchanged Apply can't re-write it.
        self._initial_endpoint = self._read_current_endpoint(self._boot_cfg)
        if self._initial_endpoint:
            self.llama_endpoint_edit.setText(self._initial_endpoint)
        device = labeled("Audio device", self.audio_device_combo)
        device.addWidget(self._audio_device_warning)
        connections.rows.addLayout(device)
        connections.rows.addLayout(labeled("Translator endpoint", self.llama_endpoint_edit))
        self._place(grid, connections, 0, span=2)

        model = Panel("Translation model")
        current_tier = self._read_current_quant_tier(self._boot_cfg)
        # The tier in effect: the transaction only writes on an actual CHANGE
        # (an unrelated Apply must not rewrite the coupled compute_type).
        self._initial_tier = current_tier
        options = tuple(
            SegmentOption(tier, name, f"{cards}, {size_text(gb)} download")
            for tier, (name, gb, cards) in TIER_INFO.items()
        )
        self._model_choice = self._themed(SegmentedControl(
            options, self._palette, value=current_tier, variant="list"))
        model.rows.addWidget(self._model_choice)
        self._quant_help = QLabel(_MODEL_HELP)
        self._quant_help.setObjectName("help")
        self._quant_help.setWordWrap(True)
        model.rows.addWidget(self._quant_help)
        # Outcome of a model switch or recovery download; hidden while empty.
        self._quant_status = QLabel("")
        self._quant_status.setObjectName("help")
        self._quant_status.setWordWrap(True)
        self._quant_status.hide()
        model.rows.addWidget(self._quant_status)
        self._place(grid, model, 2)
        return page

    def _build_shortcuts_page(self) -> QWidget:
        page = QWidget()
        grid = self._page_grid(page)
        panel = Panel("Keyboard shortcuts", note=_SHORTCUTS_NOTE)
        rows = QVBoxLayout()
        rows.setSpacing(10)
        self._shortcut_fields: dict[str, ShortcutField] = {}
        self._shortcut_messages: dict[str, QLabel] = {}
        self._shortcut_resets: dict[str, LinkButton] = {}
        self._shortcut_names = dict(_SHORTCUT_ROWS)
        for key, name in _SHORTCUT_ROWS:
            if key == "show_series" and not self._gaz_feature_enabled:
                continue  # the pipeline never binds it with the series feature off
            row = QGridLayout()
            row.setHorizontalSpacing(12)
            row.setVerticalSpacing(5)
            label = QLabel(name)
            # The stored spelling. A hand-edited combo Settings can't record
            # (the pipeline may still bind it) shows as saved, so an Apply
            # that doesn't touch the row leaves it alone; one turned off by
            # hand ("", which the pipeline skips) shows an empty field.
            saved = self._initial_shortcuts[key]
            combo = _hotkeys.canonical_combo(saved) or saved
            field = self._themed(ShortcutField(combo, self._palette))
            label.setBuddy(field)
            reset = self._themed(LinkButton("Reset", self._palette, quiet=True, underline=False))
            message = QLabel("")
            message.setObjectName("warnText")
            message.hide()
            row.addWidget(label, 0, 0)
            row.addWidget(field, 0, 1)
            row.addWidget(reset, 0, 2, Qt.AlignLeft)
            row.addWidget(message, 1, 1, 1, 2)
            row.setColumnStretch(0, 1)
            rows.addLayout(row)
            field.recorded.connect(lambda _combo, k=key: self._on_shortcut_recorded(k))
            reset.clicked.connect(lambda _checked=False, k=key: self._reset_shortcut(k))
            self._shortcut_fields[key] = field
            self._shortcut_messages[key] = message
            self._shortcut_resets[key] = reset
        panel.rows.addLayout(rows)
        self._place(grid, panel, 0, span=2)
        return page

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(56)
        row = QHBoxLayout(footer)
        row.setContentsMargins(_PAD, 0, _PAD, 0)
        row.setSpacing(8)
        self._reset_btn = self._themed(LinkButton("Reset to defaults", self._palette))
        self._reset_btn.clicked.connect(self._reset)
        self._read_only_notice = QLabel(
            "Read-only: Yamabiko was started with --config. Edit "
            f"{self._config_path} and restart to make changes."
        )
        self._read_only_notice.setObjectName("muted")
        self._read_only_notice.setTextFormat(Qt.PlainText)
        self._read_only_notice.setWordWrap(True)
        self._read_only_notice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._read_only_notice.setToolTip(self._read_only_notice.text())
        self._read_only_notice.hide()
        self._cancel_btn = QPushButton("Cancel")
        self._apply_btn = QPushButton("Apply")
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setProperty("primary", True)
        self._cancel_btn.clicked.connect(self.close)
        self._apply_btn.clicked.connect(self._apply)
        self._ok_btn.clicked.connect(self._ok)
        row.addWidget(self._reset_btn)
        # Read-only: the notice takes all the free width and wraps (two lines
        # fit the footer); otherwise a stretch pushes the buttons right.
        row.addWidget(self._read_only_notice, 1)
        if not self._read_only:
            row.addStretch(1)
        row.addWidget(self._cancel_btn)
        row.addWidget(self._apply_btn)
        row.addWidget(self._ok_btn)
        return footer

    def _apply_read_only(self) -> None:
        """--config: nothing is editable; the tabs still browse the pages."""
        self._theme_choice.setEnabled(False)
        self._switcher.setEnabled(False)
        for index in range(self._stack.count()):
            self._stack.widget(index).setEnabled(False)
        self._reset_btn.hide()
        self._read_only_notice.show()
        self._apply_btn.setEnabled(False)
        self._ok_btn.setEnabled(False)
        self._cancel_btn.setText("Close")

    # ------------------------------------------------------------------
    # change-signal wiring
    # ------------------------------------------------------------------

    def _connect_change_signals(self) -> None:
        """Wire every live control AFTER the seeding pass so seeding can't
        emit a spurious preview / re-persist."""
        self._tabs.currentChanged.connect(self._stack.setCurrentIndex)
        # Appearance controls change the rendered preview: mark a rebuild
        # pending and (re)arm the debounce.
        self._jp_font.currentFontChanged.connect(self._on_style_control_changed)
        self._en_font.currentFontChanged.connect(self._on_style_control_changed)
        for w in (self._jp_size, self._en_size, self._padding, self._bg_opacity):
            w.valueChanged.connect(self._on_style_control_changed)
        for control in self._overlay_choices.values():
            control.valueChanged.connect(self._on_style_control_changed)
        # Lines on screen decides whether the echo line shows above the sample.
        self._max_lines.valueChanged.connect(self._on_style_control_changed)
        self._max_width.valueChanged.connect(self._on_style_control_changed)
        # Timing controls don't change the rendered preview; they still arm
        # the debounce so stylePreview carries their values to the tray.
        for w in (self._hold, self._fi, self._fo):
            w.valueChanged.connect(self._on_timing_control_changed)
        self._text_color_field.clicked.connect(lambda: self._pick_color("text_color"))
        self._bg_color_field.clicked.connect(lambda: self._pick_color("bg_color"))
        # Draft controls update the window at once but persist and emit their
        # committed signals only after Apply / OK succeeds.
        self._theme_choice.valueChanged.connect(self._on_theme_changed)
        self._reduce_motion_cb.toggled.connect(self._on_reduce_motion_toggled)
        self._incremental_cb.toggled.connect(self._on_incremental_toggled)
        if self._auto_genre_cb is not None:
            self._auto_genre_cb.toggled.connect(self._on_auto_genre_toggled)

    def _on_style_control_changed(self, *args) -> None:
        # Defer the strip rebuild to the debounce (it happens in _emit_preview):
        # a slider drag fires valueChanged dozens of times per second, and each
        # rebuild re-measures the preview stack's sample lines. Mark it
        # pending and (re)arm the timer.
        self._preview_rebuild_pending = True
        # QTimer.start() on an already-running single-shot timer restarts it:
        # two rapid changes collapse into ONE _emit_preview (one rebuild + one emit).
        self._preview_timer.start()

    def _on_timing_control_changed(self, *args) -> None:
        # A timing control changed: (re)arm the debounce for the stylePreview
        # signal but do NOT mark a rebuild pending -- timing values don't change
        # the rendered preview line, so the strip stays as-is.
        self._preview_timer.start()

    def _rebuild_preview(self) -> None:
        self._preview.rebuild(self._collect())

    def _emit_preview(self) -> None:
        # The debounce fired. Rebuild the strip once (if an appearance control
        # changed since the last emit), then push the merged style to the tray.
        if self._preview_rebuild_pending:
            self._rebuild_preview()
            self._preview_rebuild_pending = False
        self.stylePreview.emit(self._collect())

    # ------------------------------------------------------------------
    # theme
    # ------------------------------------------------------------------

    def apply_palette(self, palette: dict) -> None:
        """Re-skin the open window from the GIVEN palette (the manual theme
        pick and the tray's live OS-follow both funnel through here)."""
        self._palette = palette
        apply_window_qss(self, palette)
        qpalette = self.palette()
        # The windows11 style marks the selected row of a combo popup with
        # the Accent/Highlight role: make it the theme's accent.
        for role in (QPalette.ColorRole.Highlight, QPalette.ColorRole.Accent):
            qpalette.setColor(role, QColor(palette["accent"]))
        qpalette.setColor(QPalette.ColorRole.HighlightedText, QColor(palette["on_accent"]))
        self.setPalette(qpalette)
        for widget in self._themed_widgets:
            widget.apply_palette(palette)
        set_border_color(self, palette["frame"])
        repaint_nonclient(self)

    def _on_theme_changed(self, value: str) -> None:
        self._theme_value = value if value in _THEME_VALUES else "system"
        self.apply_palette(palette_for(self._theme_value))

    def _on_reduce_motion_toggled(self, checked: bool) -> None:
        # Persistence is Apply/OK-gated; nothing in the window animates.
        self._reduce_motion = bool(checked)

    def _on_auto_genre_toggled(self, checked: bool) -> None:
        self._auto_genre = bool(checked)

    def _on_incremental_toggled(self, checked: bool) -> None:
        self._incremental = bool(checked)

    def _on_move_clicked(self) -> None:
        if self._on_move_subtitles is not None:
            self._on_move_subtitles()

    def _on_shortcut_recorded(self, key: str) -> None:
        """A row finished recording: mark it re-recorded so Apply sends it
        even when the combo matches the saved one (FR1-1), then re-check
        the warnings."""
        self._rerecorded.add(key)
        self._refresh_shortcut_warnings()

    def _reset_shortcut(self, key: str) -> None:
        self._shortcut_fields[key].set_combo(self._shortcut_defaults[key])
        self._rerecorded.discard(key)
        self._refresh_shortcut_warnings()

    def _refresh_shortcut_warnings(self) -> None:
        """Amber state per row: a combo an earlier row already uses ("Already
        used for …"), or one another app holds (the Win32 probe). The first
        row in page order keeps a shared combo, so only the later rows warn.
        While any duplicate exists, Apply and OK are disabled."""
        combos = {key: _hotkeys.canonical_combo(field.combo()) or field.combo()
                  for key, field in self._shortcut_fields.items()}
        keys = list(combos)
        dup = False
        for i, key in enumerate(keys):
            field = self._shortcut_fields[key]
            twin = next((other for other in keys[:i]
                         if combos[key] and combos[other] == combos[key]), None)
            if twin is not None:
                dup = True
                text = f"Already used for {self._shortcut_names[twin]}."
            elif self._hotkey_probe(field.combo()):
                text = _TAKEN_TEXT
            else:
                text = ""
            field.set_warning(bool(text))
            message = self._shortcut_messages[key]
            message.setText(text)
            message.setVisible(bool(text))
        self._duplicate_shortcut = dup
        if not self._read_only:
            self._apply_btn.setEnabled(not dup)
            self._ok_btn.setEnabled(not dup)

    # ------------------------------------------------------------------
    # config readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_current_audio_device(cfg: Optional[dict] = None) -> str:
        """Return the merged ``audio.device`` (or ``'default'``).

        Same swallow-and-fallback contract as :meth:`_read_current_quant_tier`
        so the window still opens on a bad/partial config. Used to seed the
        device combo so a no-touch save can't clobber the saved choice. ``cfg``
        is the shared boot config; when None, :func:`_load_cfg` loads a fresh
        one (fallback for callers that pass nothing).
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return "default"
        return str((cfg.get("audio", {}) or {}).get("device", "default") or "default")

    @staticmethod
    def _read_current_endpoint(cfg: Optional[dict] = None) -> str:
        """Return the merged ``translation.endpoint`` (or ``''``).

        Same swallow-and-fallback contract as the two readers below so
        the window still opens on a bad/partial config. Used to seed the
        endpoint field so the active value is visible and an
        unchanged Apply can't re-write it. ``cfg`` is the shared boot config;
        when None, :func:`_load_cfg` loads a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return ""
        return str((cfg.get("translation", {}) or {}).get("endpoint", "") or "")

    @staticmethod
    def _shipped_default_endpoint() -> str:
        """Return the shipped-default ``translation.endpoint`` (or ``''``).

        Read from the bundled defaults file directly (NOT the merged
        view) so Reset can offer the true default as the escape hatch
        that prunes a saved override on the next Apply.
        """
        defaults = _read_shipped_defaults()
        if defaults is None:
            return ""
        return str((defaults.get("translation", {}) or {}).get("endpoint", "") or "")

    @staticmethod
    def _shipped_default_hotkeys() -> dict[str, str]:
        """The shipped ``hotkeys.*`` combos (the per-row Reset target), read
        from the bundled defaults file; the documented defaults on failure."""
        defaults = _read_shipped_defaults()
        if defaults is None:
            return dict(_FALLBACK_HOTKEYS)
        section = (defaults.get("hotkeys", {}) or {})
        return {key: str(section.get(key) or fallback) for key, fallback in _FALLBACK_HOTKEYS.items()}

    @staticmethod
    def _shipped_default_incremental() -> bool:
        """The shipped ``asr.incremental.enabled`` (the Reset target), read
        from the bundled defaults file; True (its shipped value) on failure."""
        defaults = _read_shipped_defaults()
        if defaults is None:
            return True
        section = ((defaults.get("asr", {}) or {}).get("incremental", {}) or {})
        return bool(section.get("enabled", True))

    @staticmethod
    def _read_current_quant_tier(cfg: Optional[dict] = None) -> str:
        """Return the merged ``translation.quant_tier`` or ``'default'``.

        Swallow any load failure (missing yaml package, malformed file,
        partial install) and fall back to ``'default'`` so the window
        still opens; persisting from the dropdown only writes the
        sparse override, so a bad read can't clobber unrelated keys. ``cfg``
        is the shared boot config; when None, :func:`_load_cfg` loads a
        fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return "default"
        tier = (cfg.get("translation", {}) or {}).get("quant_tier", "default")
        return "small" if tier == "small" else "default"

    @staticmethod
    def _read_current_theme(cfg: Optional[dict] = None) -> str:
        """Return the merged ``overlay.theme`` (``system`` unless valid).

        ``cfg`` is the shared boot config; when None, :func:`_load_cfg` loads
        a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return "system"
        theme = (cfg.get("overlay", {}) or {}).get("theme", "system")
        return theme if theme in _THEME_VALUES else "system"

    @staticmethod
    def _read_current_reduce_motion(cfg: Optional[dict] = None) -> bool:
        """Return the merged ``overlay.reduce_motion`` bool (default False).

        ``cfg`` is the shared boot config; when None, :func:`_load_cfg` loads
        a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return False
        return bool((cfg.get("overlay", {}) or {}).get("reduce_motion", False))

    @staticmethod
    def _read_current_auto_genre(cfg: Optional[dict] = None) -> bool:
        """Return the merged ``gazetteer.auto_genre`` bool (default True).

        ``cfg`` is the shared boot config; when None, :func:`_load_cfg` loads
        a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return True
        return bool((cfg.get("gazetteer", {}) or {}).get("auto_genre", True))

    @staticmethod
    def _read_current_incremental(cfg: Optional[dict] = None) -> bool:
        """Return the merged ``asr.incremental.enabled`` bool (default True).

        ``cfg`` is the shared boot config; when None, :func:`_load_cfg` loads
        a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return True
        return bool(((cfg.get("asr", {}) or {}).get("incremental", {}) or {}).get("enabled", True))

    @staticmethod
    def _read_current_anilist_username(cfg: Optional[dict] = None) -> str:
        """Return the merged ``gazetteer.anilist_username``, trimmed
        (default ``""`` = watch-list section off).

        Same swallow-and-fallback contract as the sibling readers so the
        window still opens on a bad/partial config. ``cfg`` is the shared
        boot config; when None, :func:`_load_cfg` loads a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return ""
        raw = (cfg.get("gazetteer", {}) or {}).get("anilist_username", "")
        return str(raw or "").strip()

    @staticmethod
    def _read_current_feature_enabled(cfg: Optional[dict] = None) -> bool:
        """Return the series-gazetteer feature's effective on-switch for the
        merged config: :func:`yamabiko.gazetteer.gazetteer.feature_enabled`
        (BOTH gazetteer.enabled and romanization.enabled), the one helper
        every series entry point gates on. Falls back to the shipped default
        (True -- both flags ship true) on a failed load, the same
        swallow-and-fallback contract as the sibling readers. ``cfg`` is the
        shared boot config; when None, :func:`_load_cfg` loads a fresh one.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return True
        from yamabiko.gazetteer.gazetteer import feature_enabled
        return feature_enabled(cfg)

    @staticmethod
    def _read_current_position(cfg: Optional[dict] = None) -> dict:
        """Return the merged ``overlay.position``, clamped (defaults on error).

        ``cfg`` is the shared boot config; when None, :func:`_load_cfg` loads
        a fresh one. The Apply-time re-read (``_collect_position``) calls
        ``_load_cfg(None)`` ITSELF so it can tell a failed fresh read from a
        successful one, and only comes here with the loaded cfg in hand.
        """
        cfg = _load_cfg(cfg)
        if cfg is None:
            return clamp_position({})
        return clamp_position((cfg.get("overlay", {}) or {}).get("position", {}) or {})

    @staticmethod
    def _read_current_hotkeys(cfg: Optional[dict], defaults: dict[str, str]) -> dict[str, str]:
        """The merged ``hotkeys.*`` combos as saved, in the saved spelling
        (the change guard's baseline; a field shows the stored spelling of a
        combo that parses, any other as saved, and an empty one, a shortcut
        turned off by hand, as empty); ``defaults`` fill a missing key or a
        failed load (same swallow-and-fallback contract as the sibling
        readers)."""
        cfg = _load_cfg(cfg)
        section = ((cfg or {}).get("hotkeys", {}) or {})
        return {key: str(section.get(key, defaults[key]) or "") for key in defaults}

    # ------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------

    def selected_quant_tier(self) -> str:
        """The model list's tier (``'default'`` or ``'small'``)."""
        return "small" if self._model_choice.value() == "small" else "default"

    def _opacity_value(self) -> int:
        stored, shown = self._opacity_seed
        value = self._bg_opacity.value()
        return stored if value == shown else round(value * 255 / 100)

    def _hold_value(self) -> int:
        stored, shown = self._hold_seed
        value = self._hold.value()
        return stored if abs(value - shown) < 1e-9 else round(value * 1000)

    def _pick_color(self, key: str) -> None:
        field = self._text_color_field if key == "text_color" else self._bg_color_field
        role = "text" if key == "text_color" else "background"
        other = self._working["bg_color" if key == "text_color" else "text_color"]
        color = self._pick_color_dialog(self, self._palette, role, self._working[key], other)
        if color:
            self._working[key] = color
            field.set_color(color)
            self._on_style_control_changed()

    def _seed_font(self, key: str, combo: QFontComboBox) -> None:
        """Show the working family in ``combo`` and record what it shows.

        A family that is not installed makes QFontComboBox show a
        substitute (a missing family became 'Tahoma' in a probe). Recording
        the shown family lets _font_family tell "the user picked a font"
        from "the combo is still on the substitute it seeded"."""
        family = self._working[key]
        combo.setCurrentFont(QFont(family))
        self._font_seed[key] = (family, combo.currentFont().family())

    def _font_family(self, key: str, combo: QFontComboBox) -> str:
        """The family to save: the configured one while the combo still
        shows the family it was seeded with, else the user's pick."""
        shown = combo.currentFont().family()
        configured, seeded_shown = self._font_seed[key]
        return configured if shown == seeded_shown else shown

    def _collect(self) -> dict:
        return {
            "jp_font": self._font_family("jp_font", self._jp_font),
            "jp_size": self._jp_size.value(),
            "en_font": self._font_family("en_font", self._en_font),
            "en_size": self._en_size.value(),
            "text_color": self._working["text_color"],
            "bg_color": self._working["bg_color"],
            "bg_opacity": self._opacity_value(),
            "padding": self._padding.value(),
            "max_width_pct": self._max_width.value(),
            "max_lines": self._max_lines.value(),
            "hold_ms": self._hold_value(),
            "fade_in_ms": self._fi.value(),
            "fade_out_ms": self._fo.value(),
            **{key: control.value() for key, control in self._overlay_choices.items()},
        }

    # ------------------------------------------------------------------
    # Apply / OK / Cancel / Reset
    # ------------------------------------------------------------------

    def _collect_position(self) -> Optional[dict]:
        """Return a changed position draft while preserving live disk state."""
        if self._config_path is None:
            return None
        combo_screen = self._monitor_combo.currentData() or "primary"
        v_slider = self._voffset.value()
        if (
            combo_screen == self._initial_position_screen
            and v_slider == self._initial_position_v
        ):
            return None
        cfg = _load_cfg(None)
        if cfg is None:
            _log.warning(
                "position re-read failed at Apply; keeping the window's "
                "last-good position instead of shipped defaults",
            )
            saved = dict(self._last_good_position)
        else:
            saved = self._read_current_position(cfg)
        screen = (
            combo_screen
            if combo_screen != self._initial_position_screen
            else saved["screen"]
        )
        v_out = (
            float(v_slider)
            if v_slider != self._initial_position_v
            else saved["v_offset_pct"]
        )
        return {
            "screen": screen,
            "v_offset_pct": v_out,
            "h_offset_pct": saved["h_offset_pct"],
            "left_offset_pct": saved["left_offset_pct"],
        }

    def _collect_commit(self) -> SettingsCommit:
        """Collect one complete, clamped Settings draft without side effects."""
        from ..config import _clamp_style

        style = _clamp_style(self._collect())
        updates = {
            ("overlay", "style", key): value
            for key, value in style.items()
        }

        if self.audio_device_combo.currentIndex() == self._unavailable_device_row:
            device_name = self._unavailable_device_name
        else:
            device_name = self.audio_device_combo.currentText().strip()
        if (
            self.audio_device_combo.currentIndex() != self._initial_device_index
            and device_name
        ):
            updates[("audio", "device")] = device_name

        endpoint = self.llama_endpoint_edit.text().strip()
        if endpoint and endpoint != self._initial_endpoint:
            updates[("translation", "endpoint")] = endpoint
        elif not endpoint and self._initial_endpoint:
            shipped_default = self._shipped_default_endpoint()
            if shipped_default:
                updates[("translation", "endpoint")] = shipped_default
            else:
                _log.warning(
                    "endpoint clear ignored: shipped default endpoint is "
                    "unreadable; kept the existing translation.endpoint override",
                )

        if self._anilist_username_edit is not None:
            username = self._anilist_username_edit.text().strip()
            if username != self._initial_anilist_username:
                updates[("gazetteer", "anilist_username")] = username

        if self._theme_value != self._initial_theme:
            updates[("overlay", "theme")] = self._theme_value
        if self._reduce_motion != self._initial_reduce_motion:
            updates[("overlay", "reduce_motion")] = self._reduce_motion
        if self._auto_genre != self._initial_auto_genre:
            updates[("gazetteer", "auto_genre")] = self._auto_genre
        if self._incremental != self._initial_incremental:
            updates[("asr", "incremental", "enabled")] = self._incremental

        for key, field in self._shortcut_fields.items():
            combo, saved = field.combo(), self._initial_shortcuts[key]
            # A re-recorded row is resent even when unchanged, so a shortcut
            # that failed to bind gets another try (FR1-1). Otherwise compare
            # raw when either is empty (a shortcut turned off by hand): an
            # empty field and an unparseable saved combo both canonicalize
            # to None.
            if key in self._rerecorded or (
                    combo != saved if not (combo and saved)
                    else _hotkeys.canonical_combo(combo) != _hotkeys.canonical_combo(saved)):
                updates[("hotkeys", key)] = combo

        tier = self.selected_quant_tier()
        if tier != self._initial_tier:
            from ..gpu import tier_overrides

            tier_patch = tier_overrides(tier)
            updates[("translation", "quant_tier")] = (
                tier_patch["translation"]["quant_tier"]
            )
            updates[("asr", "faster_whisper", "compute_type")] = (
                tier_patch["asr"]["faster_whisper"]["compute_type"]
            )

        position = self._collect_position()
        remove_paths = ()
        if position is not None:
            for key, value in position.items():
                updates[("overlay", "position", key)] = value
            remove_paths = (("overlay", "geometry"),)

        return SettingsCommit(
            updates=updates,
            remove_paths=remove_paths,
            style=style,
            position=position,
        )

    def _persist_commit(self, commit: SettingsCommit) -> bool:
        """Persist a complete draft before any committed runtime signal. A
        failed write asks "Couldn't save your settings": Retry writes again,
        Cancel keeps the window open with nothing saved."""
        if self._config_path is None:
            return True
        from . import settings_transaction

        while True:
            try:
                settings_transaction.persist_settings(commit)
                return True
            except Exception as exc:
                _log.exception("failed to persist settings")
                if not self._ask_retry_save(self, self._palette, self._overrides_path(), exc):
                    return False

    @staticmethod
    def _overrides_path() -> str:
        try:
            from yamabiko import paths as _paths
            return str(_paths.user_overrides_path())
        except Exception:
            return "user_overrides.yaml"

    def _apply_committed_runtime(self, commit: SettingsCommit) -> None:
        """Apply only state that has crossed the persistence boundary."""
        self._working = dict(commit.style)
        if ("translation", "quant_tier") in commit.updates:
            self._set_quant_status(
                "The model is ready. It takes effect the next time Yamabiko starts."
            )
        if self._reduce_motion != self._initial_reduce_motion:
            # Before styleChanged: the tray's apply_style must carry it.
            self.reduceMotionChanged.emit(self._reduce_motion)
        self.styleChanged.emit(dict(commit.style))
        if commit.position is not None:
            self.positionApplied.emit(dict(commit.position))
        if self._theme_value != self._initial_theme:
            self.themeChanged.emit(self._theme_value)
        if self._auto_genre != self._initial_auto_genre:
            self.autoGenreChanged.emit(self._auto_genre)
        if self._incremental != self._initial_incremental:
            self.incrementalChanged.emit(self._incremental)
        changed = {path[1]: value for path, value in commit.updates.items()
                   if path[0] == "hotkeys"}
        if changed:
            self.shortcutsChanged.emit(changed)

    def _advance_persisted_baselines(self, commit: SettingsCommit) -> None:
        """Advance change guards only after persistence and runtime apply."""
        paths = commit.updates
        if ("audio", "device") in paths:
            self._initial_device = paths[("audio", "device")]
            self._initial_device_index = self.audio_device_combo.currentIndex()
        if ("translation", "endpoint") in paths:
            self._initial_endpoint = self.llama_endpoint_edit.text().strip()
        if ("gazetteer", "anilist_username") in paths:
            self._initial_anilist_username = (
                self._anilist_username_edit.text().strip()
                if self._anilist_username_edit is not None
                else ""
            )
        self._initial_theme = self._theme_value
        self._initial_reduce_motion = self._reduce_motion
        self._initial_auto_genre = self._auto_genre
        self._initial_incremental = self._incremental
        if ("translation", "quant_tier") in paths:
            self._initial_tier = self.selected_quant_tier()
        if commit.position is not None:
            self._initial_position_screen = (
                self._monitor_combo.currentData() or "primary"
            )
            self._initial_position_v = self._voffset.value()
            self._last_good_position = dict(commit.position)
        for path, value in paths.items():
            if path[0] == "hotkeys":
                self._initial_shortcuts[path[1]] = value
        # The commit carried every re-recorded row; start clean.
        self._rerecorded.clear()

    def _apply(self) -> bool:
        """Prepare, persist, then apply one staged Settings transaction."""
        # Duplicate shortcuts block the commit (Apply and OK are disabled too).
        if self._read_only or self._duplicate_shortcut:
            return False
        commit = self._collect_commit()
        if not self._prepare_quant_tier(commit):
            return False
        if not self._persist_commit(commit):
            return False
        self._apply_committed_runtime(commit)
        self._advance_persisted_baselines(commit)
        return True

    def _ok(self) -> None:
        # Do NOT close on a failed save: the save-failed dialog kept the
        # window open, so the user can retry or cancel.
        if self._apply():
            if self._recovery_notice is not None:
                # This close destroys the inline status line before the
                # "offers the download again" note can be read; hand the
                # notice to the tray toast, which outlives the window.
                # Apply never emits this -- the window stays open there.
                self.noticeRequested.emit(
                    "Translation model still missing", self._recovery_notice,
                )
            self.close()

    def _set_quant_status(self, text: str) -> None:
        self._quant_status.setText(text)
        self._quant_status.setVisible(bool(text))

    def _prepare_quant_tier(self, commit: SettingsCommit) -> bool:
        """Acquire the selected model before the Settings commit is written.

        Only acts when ``config_path`` was supplied -- same
        production-vs-unit-test gate as :meth:`_persist_commit`.

        Unlike the previous behaviour (which wrote the tier and told the user
        the model would "re-download on next launch" -- a promise the runtime
        never kept, since ``watchdog._resolve_translator_model_path`` does a cache
        lookup only and silently falls back to whichever tier IS cached), the
        switch now guarantees the model is present BEFORE the transaction:

        * target tier already in the HF cache -> persist the coupled pair and
          confirm it takes effect next launch;
        * not cached -> download it first in the compact download dialog. On
          success, persist + confirm. On cancel or error, persist NOTHING and
          say so (the app's effective state is exactly as before Apply); a
          cancel additionally reverts the model list to the tier in effect,
          while a failure keeps the selection so the next Apply or OK
          genuinely retries the download.

        The cache probe also runs when the tier did NOT change, so an install
        stranded by the old bug (tier persisted, GGUF never downloaded) can
        recover by re-Applying the already-selected tier; that recovery leg
        downloads but skips the write (the pair is already on disk, and a
        rewrite would clobber a hand-set compute_type). A canceled/failed
        RECOVERY download is non-blocking: it logs, notes the outcome in the
        inline status line, and returns True -- the Apply that triggered it
        was about other settings (already committed), and the recovery
        re-attempts on any future Apply. Blocking OK on it soft-locked the
        window on a download the user never asked for.

        Returns True on success or no-op skip (including a deferred
        recovery-leg download); False when a REAL tier change's download was
        canceled or errored. This method never writes configuration.
        """
        if self._config_path is None:
            return True
        # Per-pass reset: the notice must describe THIS apply, not a
        # recovery outcome from an earlier Apply click.
        self._recovery_notice = None
        tier = self.selected_quant_tier()
        tier_changed = ("translation", "quant_tier") in commit.updates

        # The cache probe runs BEFORE the no-change guard: installs stranded
        # by the old persist-without-download bug (quant_tier persisted, GGUF
        # never fetched) open with the list ALREADY on the persisted tier, so
        # an equal-tier Apply is their only lever to trigger the fetch. The
        # probe is a cheap local-disk lookup; a cached tier still short-
        # circuits to a full no-op below.
        from . import model_fetch
        downloaded = False
        # Recovery leg vs real switch: when the list still matches the tier
        # in effect, the probe below is the stranded-install recovery (the
        # pair is already on disk; only the GGUF is missing) piggybacking on
        # an Apply that was about OTHER settings -- and those settings are
        # already committed by the time this runs. A canceled or failed
        # recovery download therefore must NOT fail the whole Apply: OK
        # would refuse to close, soft-locking the window on a download the
        # user never asked for. The probe re-runs on any future Apply, so
        # recovery is only deferred, never lost. A REAL tier change keeps
        # blocking on cancel/failure (persisting a tier whose GGUF is
        # missing is a promise the runtime cannot keep).
        recovery_leg = not tier_changed
        if not model_fetch.tier_cached(tier):
            outcome = self._download_quant_tier(tier)
            if outcome != "success":
                if recovery_leg:
                    _log.warning(
                        "stranded-install recovery download %s (%s); leaving "
                        "the Apply unblocked -- recovery re-runs on a future "
                        "Apply",
                        outcome, self._last_quant_error or "no error detail",
                    )
                    verb = "was canceled" if outcome == "cancelled" else "stopped"
                    self._set_quant_status(
                        "The model file is still missing. Yamabiko offers the "
                        "download again the next time you click Apply or OK."
                    )
                    # The same notice, worded for after the window is gone:
                    # on OK the close hides the status line immediately, so
                    # _ok toasts this through noticeRequested instead.
                    self._recovery_notice = (
                        f"The download {verb}. Yamabiko offers it again the "
                        "next time you open Settings and click Apply or OK."
                    )
                    return True
                # Canceled or failed REAL switch: nothing persisted, distinct
                # message, effective state unchanged. Only a CANCEL reverts
                # the list (the user abandoned the switch, so the UI reflects
                # the tier in effect). A FAILURE keeps the selection: a
                # reverted list would send the next Apply or OK into the
                # no-change guard as a silent no-op -- with OK then closing
                # the window as if the switch had succeeded. The download
                # dialog already showed the failure with Retry and Copy
                # details.
                if outcome == "cancelled":
                    self._restore_model_choice()
                    self._set_quant_status("Download canceled. The model wasn't changed.")
                else:
                    self._set_quant_status("The download stopped. The model wasn't changed.")
                return False
            downloaded = True

        if not tier_changed:
            # No tier change: nothing to (re)write. Only an actual CHANGE
            # writes the coupled pair -- an unconditional write would reset a
            # hand-set asr.faster_whisper.compute_type override on every
            # Apply, contradicting gpu.py's hand-edit-survival contract; on
            # the stranded-recovery path above the pair is already on disk.
            if downloaded:
                self._set_quant_status(
                    "The model is ready. It takes effect the next time Yamabiko starts."
                )
            return True

        return True

    def _download_quant_tier(self, tier: str) -> str:
        """Run the modal download dialog for ``tier``; return its outcome.

        Returns one of ``"success"`` / ``"cancelled"`` / ``"failed"``. The
        dialog is tracked on ``self._active_quant_dialog`` for the duration of
        the modal loop so the tray's quit-time :meth:`release_threads` can
        detach the in-flight worker into ``thread_keeper`` (a running download
        must not abort the process on quit). ``self._last_quant_error`` carries
        the human-readable failure reason for the caller's log.
        """
        from .model_fetch import QuantDownloadDialog, spec_for_tier
        dialog = QuantDownloadDialog(
            spec_for_tier(tier), tier,
            downloader=self._quant_downloader, parent=self, palette=self._palette,
        )
        self._active_quant_dialog = dialog
        try:
            dialog.exec()
        finally:
            self._active_quant_dialog = None
            # QApplication.exit() unwinds ALL stacked event loops, so an app
            # exit fired while this modal loop runs makes exec() return
            # UNSETTLED: no accept/reject ran and the worker may still be
            # blocked inside hf_hub_download. (Qt 6.5+ app.quit() usually
            # settles the dialog first -- its Quit event closes top-level
            # windows, driving closeEvent -> reject(), which parks the thread
            # -- but exit(), a quit whose window-close leg skips the dialog,
            # or an exec() entered after quitNow was already set all unwind
            # unsettled.) Park the worker NOW (idempotent -- a settled dialog
            # already released its thread and this no-ops). Without this, the
            # nulled _active_quant_dialog above makes the aboutToQuit release
            # path (Tray.release_threads -> SettingsWindow.release_threads) a
            # no-op, thread_keeper.any_running() stays False so run_overlay
            # skips its hard-exit, and interpreter teardown destroys the
            # running QThread -- a Qt 6 qFatal abort.
            dialog.release_threads()
        self._last_quant_error = dialog.friendly_error()
        outcome = dialog.outcome
        dialog.deleteLater()
        return outcome

    def _restore_model_choice(self) -> None:
        """Back to the tier in effect after a CANCELED download (a failed one
        keeps the selection so the next Apply retries). The list has no
        connected change signal, so this has no side effects."""
        self._model_choice.setValue(self._initial_tier)

    def release_threads(self) -> None:
        """Detach an in-flight quant-download worker at quit.

        Called from ``Tray.release_threads`` (wired to ``app.aboutToQuit``),
        the same pattern the palette / gazetteer / diagnostics / update workers
        use. Delegates to the active download dialog, which parks its worker in
        ``thread_keeper`` so QApplication teardown / interpreter finalization
        cannot destroy a running QThread (Qt 6 qFatal). No-op when no download
        is in flight."""
        dialog = self._active_quant_dialog
        if dialog is not None:
            dialog.release_threads()

    def _reset(self) -> None:
        self._working = dict(STYLE_DEFAULTS)
        self._seed_font("jp_font", self._jp_font)
        self._jp_size.setValue(self._working["jp_size"])
        self._seed_font("en_font", self._en_font)
        self._en_size.setValue(self._working["en_size"])
        self._text_color_field.set_color(self._working["text_color"])
        self._bg_color_field.set_color(self._working["bg_color"])
        self._opacity_seed = (self._working["bg_opacity"],
                              round(self._working["bg_opacity"] * 100 / 255))
        self._bg_opacity.setValue(self._opacity_seed[1])
        self._padding.setValue(self._working["padding"])
        self._max_width.setValue(self._working["max_width_pct"])
        self._max_lines.setValue(self._working["max_lines"])
        self._hold_seed = (self._working["hold_ms"], round(self._working["hold_ms"] / 1000, 1))
        self._hold.setValue(self._hold_seed[1])
        self._fi.setValue(self._working["fade_in_ms"])
        self._fo.setValue(self._working["fade_out_ms"])
        for key, control in self._overlay_choices.items():
            control.setValue(STYLE_DEFAULTS[key])
        # Escape hatch: the SHIPPED default endpoint; a following Apply
        # writes it and the sparse writer prunes it (the one UI path that
        # clears a saved endpoint override).
        self.llama_endpoint_edit.setText(self._shipped_default_endpoint())
        # Faster subtitles back to its shipped value; the toggled handler
        # updates the draft, and Apply / OK commits it as usual.
        self._incremental_cb.setChecked(self._shipped_default_incremental())
        # Arm the debounce UNCONDITIONALLY: setters on unchanged values stay
        # silent and the color fields have no change signal.
        self._on_style_control_changed()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # Shrink priority on a short window: the preview monitor gives up one
        # px per px the window is under WINDOW_H, keeping its ratio, down to
        # MONITOR_MIN_H. Every row but the panels has a fixed height, so the
        # panel scroll area gets exactly what is left and scrolls only once
        # the monitor is at its minimum. The layout already ran for this
        # resize with the old monitor size: run it again now.
        if self._preview.fit_height(MONITOR_H - max(0, WINDOW_H - self.height())):
            self.layout().activate()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        set_border_color(self, self._palette["frame"])
        add_window_menu(self, min_max=True)
        if not self._fitted:
            self._fitted = True
            QTimer.singleShot(0, self._fit_to_screen)

    def nativeEvent(self, event_type, message):  # noqa: N802 (Qt override)
        result = nonclient_event(event_type, message, self._palette["ground"])
        if result is not None:
            return True, result
        return super().nativeEvent(event_type, message)

    def _fit_to_screen(self, avail: Optional[QRect] = None) -> None:
        """On first show: fit the whole window, frame and footer included,
        inside the work area. A work area shorter than the design height
        shrinks the window (never below WINDOW_MIN_H; resizeEvent takes the
        height from the preview monitor first, then the panels scroll); one
        that holds it keeps the design height. ``avail`` overrides the work
        area for tests."""
        if self._closed or not self.isVisible():
            return
        if avail is None:
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()
        # The frame around the client (Windows' invisible resize borders, the
        # offscreen platform's 2 px) must fit too.
        frame_h = self.frameGeometry().height() - self.height()
        height = max(self.minimumHeight(), min(WINDOW_H, avail.height() - frame_h))
        if height != self.height():
            self.resize(self.width(), height)
        frame = self.frameGeometry()
        x = min(max(frame.left(), avail.left()), max(avail.left(), avail.right() + 1 - frame.width()))
        y = min(max(frame.top(), avail.top()), max(avail.top(), avail.bottom() + 1 - frame.height()))
        if (x, y) != (frame.left(), frame.top()):
            self.move(x, y)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # The QDialog -> QWidget port lost the dialog's built-in key handling
        # (Esc -> reject, Return -> default button). Restore the same
        # semantics: Esc = Cancel (close; un-Applied edits are discarded),
        # Return/Enter = OK (apply-and-close, gated on the persists like the
        # OK button). Handled here rather than via a QShortcut so Return still
        # lands when a QLineEdit has focus -- the line edit ignores Return
        # after emitting returnPressed, so it propagates up exactly as
        # QDialog's auto-default relied on. A focused push button, link
        # button or color field never gets here: each clicks itself on Return
        # (autoDefault, see __init__; controls._ClickOnEnter).
        if event.key() == Qt.Key_Escape:
            event.accept()
            self.close()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            self._ok()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # previewEnded lets the tray drop the sample subtitle and restore the
        # committed style. _closed is the observable "the window actually
        # closed" flag the OK / Cancel tests assert on.
        self._closed = True
        # Stop the live-preview debounce BEFORE announcing the close. A style
        # control change within the debounce window (150 ms) of closing would
        # otherwise leave the single-shot timer armed: it fires _emit_preview
        # AFTER previewEnded, and the tray would re-show an orphaned sample on
        # the live overlay (P2). Stopping it here is the primary guard; the
        # tray's _on_style_preview no-op-when-closed is the belt.
        self._preview_timer.stop()
        self.previewEnded.emit()
        super().closeEvent(event)
