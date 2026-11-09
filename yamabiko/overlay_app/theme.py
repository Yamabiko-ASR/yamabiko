"""Design tokens and QSS builders for Yamabiko's runtime chrome.

Two palettes, from the approved app look ("Studio under the mountain"):

* MIST (mist, light)
* NIGHT (night, dark)

This module styles CHROME only: the settings window, the tray menu and the
placement hint (the series bar paints itself from the same tokens). The
subtitle overlay itself is NOT themed here; it keeps the user-configured
colours.

Purity contract
---------------
Token access, ``resolve_mode`` and the QSS builders are pure and Qt-free,
so their tests run without a QApplication.
The ONLY Qt touch is a lazy import inside ``system_scheme_probe`` (it can be
called before app construction), and the ONLY OS touch is the winreg
fallback inside the same function.

Anti-drift contract
-------------------
Every hex a builder emits is pulled from the palette dict via a shared
``%(token)s`` format template, so the two palettes can never diverge
structurally: only their token values differ. There are no colour literals
in builder output. (The templates carry no literal ``%`` so ``%``-style
formatting is safe; if a percentage is ever added it must be written ``%%``.)
"""
from __future__ import annotations

import sys
from typing import Callable

from .style_defaults import parse_hex_rgb

# ---------------------------------------------------------------------------
# Token palettes. Surface tokens (ground, raised, field), text tokens (ink,
# muted), rules (line for dividers and panels, edge for control borders),
# hover, accent with its on-accent text, and the landscape tones (ridge1-4 far
# to near, echo1-3 inner to outer). accent_hover (primary button hover), rail
# (slider and progress tracks), warn and warn_mark (the taken-shortcut state)
# and frame (the window edge) come from the Settings mockups. ``name`` is a
# display slug only. Both palettes carry the EXACT same key set; a mismatch is
# a latent KeyError in a builder or painter.
# ---------------------------------------------------------------------------

MIST: dict = {
    "name": "mist",
    "ground": "#f4f3ef",
    "raised": "#fbfaf7",
    "field": "#ffffff",
    "ink": "#1f2421",
    "muted": "#646a65",
    "line": "#dedcd4",
    "edge": "#c5c5c2",
    "hover": "#eceae4",
    "accent": "#c8402c",
    "on_accent": "#ffffff",
    "accent_hover": "#b53a27",
    "rail": "#d3d2cc",
    "warn": "#9d5d00",
    "warn_mark": "#f2b33d",
    "frame": "#bdbdb9",
    "ridge1": "#dadad6",
    "ridge2": "#cececa",
    "ridge3": "#b4b5b1",
    "ridge4": "#5f625f",
    "echo1": "#9b9c98",
    "echo2": "#b4b5b1",
    "echo3": "#c8c8c4",
}

NIGHT: dict = {
    "name": "night",
    "ground": "#121614",
    "raised": "#1a1f1c",
    "field": "#151a17",
    "ink": "#e9e6de",
    "muted": "#949a94",
    "line": "#2a302c",
    "edge": "#414440",
    "hover": "#232825",
    "accent": "#e0533c",
    "on_accent": "#121614",
    "accent_hover": "#e8664f",
    "rail": "#3a3f3b",
    "warn": "#e3a33b",
    "warn_mark": "#e3a33b",
    "frame": "#393b38",
    "ridge1": "#2e312e",
    "ridge2": "#3a3d39",
    "ridge3": "#4a4c49",
    "ridge4": "#686965",
    "echo1": "#757671",
    "echo2": "#5b5d59",
    "echo3": "#464844",
}

_TOKEN_KEYS = (
    "ground", "raised", "field", "ink", "muted", "line", "edge", "hover",
    "accent", "on_accent", "accent_hover", "rail", "warn", "warn_mark", "frame",
    "ridge1", "ridge2", "ridge3", "ridge4", "echo1", "echo2", "echo3",
)


def _validate_palette(p: dict) -> None:
    """Anti-drift guard for one palette dict; raises ValueError on drift.

    Checks the canonical key set (a token present in one palette but not the
    other is a latent KeyError in a builder or painter -- see the Anti-drift
    contract above) and that every colour token parses via the shared
    :func:`parse_hex_rgb`.
    Explicit raises (not assert) so the guard survives ``python -O``.
    """
    if set(p) != {"name", *_TOKEN_KEYS}:
        raise ValueError(
            f"theme palette {p.get('name')!r} keys {sorted(p)} drift from "
            f"the canonical _TOKEN_KEYS {sorted(_TOKEN_KEYS)} (+ 'name')"
        )
    for key in _TOKEN_KEYS:
        if parse_hex_rgb(p[key]) is None:
            raise ValueError(
                f"theme palette {p['name']!r} token {key}={p[key]!r} is not "
                f"a #rrggbb hex"
            )


# Fail fast at IMPORT if either palette drifts (see _validate_palette).
for _p in (MIST, NIGHT):
    _validate_palette(_p)
del _p


# ---------------------------------------------------------------------------
# Mode resolution.
# ---------------------------------------------------------------------------

def resolve_mode(cfg_value: str, scheme_probe: Callable[[], str]) -> str:
    """Resolve the effective ``"light"``/``"dark"`` mode.

    ``cfg_value`` of ``"light"`` or ``"dark"`` forces the result and the
    probe is NOT called. ``"system"`` (or any other value) delegates to
    ``scheme_probe``; its return is trusted only for ``"light"``/``"dark"``,
    and anything else falls back to ``"dark"``.
    """
    if cfg_value == "light":
        return "light"
    if cfg_value == "dark":
        return "dark"
    return "light" if scheme_probe() == "light" else "dark"


def _qt_color_scheme() -> str | None:
    """Best-effort Qt colour-scheme read, or None if unavailable.

    PySide6 must import (this can be called before app construction); the
    actual read of whatever ``instance()`` returns is delegated to
    ``_scheme_from_app`` and carries its own guards.
    """
    try:
        from PySide6.QtGui import QGuiApplication
    except Exception:
        return None
    return _scheme_from_app(QGuiApplication.instance())


def _scheme_from_app(app) -> str | None:
    """Read ``app``'s styleHints colour scheme -> 'light'/'dark'/None.

    Guards, in order: ``app`` must be a genuine QGuiApplication --
    ``instance()`` returns whatever QCoreApplication exists, and a bare
    non-GUI QCoreApplication in a headless host has no ``styleHints`` at
    all, so an unguarded read would break system_scheme_probe's
    never-raises contract (the isinstance also covers ``app is None``);
    styleHints must expose ``colorScheme`` (Qt >= 6.5); and
    ``ColorScheme.Unknown`` returns None so the registry fallback runs.
    Split out of ``_qt_color_scheme`` so the guard is testable with a fake
    app object; only reached with PySide6 importable, so the lazy Qt
    imports here cannot fail.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    if not isinstance(app, QGuiApplication):
        return None
    hints = app.styleHints()
    if not hasattr(hints, "colorScheme"):
        return None
    try:
        scheme = hints.colorScheme()
    except Exception:
        return None
    if scheme == Qt.ColorScheme.Light:
        return "light"
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    return None  # Unknown -> defer to the registry fallback


def _registry_color_scheme() -> str:
    """Read HKCU AppsUseLightTheme -> ``"light"``/``"dark"``.

    Any error (missing key, winreg failure) or a non-Windows host -> dark.
    """
    if sys.platform != "win32":
        return "dark"
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        try:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        finally:
            winreg.CloseKey(key)
        return "light" if value == 1 else "dark"
    except Exception:
        return "dark"


def system_scheme_probe() -> str:
    """Probe the OS colour scheme -> ``"light"`` or ``"dark"``.

    Prefers Qt's ``styleHints().colorScheme()`` when a QGuiApplication is
    already running (Qt >= 6.5), else the Windows registry, else ``"dark"``.
    Never raises.
    """
    qt = _qt_color_scheme()
    if qt is not None:
        return qt
    return _registry_color_scheme()


def palette_for(cfg_value: str) -> dict:
    """Resolve ``cfg_value`` -> a concrete token palette (MIST / NIGHT)."""
    return NIGHT if resolve_mode(cfg_value, system_scheme_probe) == "dark" else MIST


# ---------------------------------------------------------------------------
# QSS builders. Each is ``(p: dict) -> str`` and formats a shared template so
# both palettes render identical structure. Radii 4 px for controls, 6 px for
# panels, 2 px inside the compact dialogs (the Settings mockups); 1px borders,
# no gradients, no glow. Menus mark selection Lit, the accent fill with
# on-accent text (Decision 62).
# ---------------------------------------------------------------------------

_WINDOW_QSS = """\
QWidget#settingsWindow, QDialog#compactDialog {
    background-color: %(ground)s;
}
QWidget {
    color: %(ink)s;
}
QLabel {
    background: transparent;
    color: %(ink)s;
}
QLabel:disabled, QLabel#help, QLabel#panelNote, QLabel#muted {
    color: %(muted)s;
}
QLabel#warnText {
    color: %(warn)s;
}
QLabel#code {
    font-family: "Consolas";
    font-size: 12px;
    background-color: %(field)s;
    border: 1px solid %(line)s;
    border-radius: 2px;
    padding: 0 4px;
}
QFrame#panel {
    background-color: %(raised)s;
    border: 1px solid %(line)s;
    border-radius: 6px;
}
QFrame#footer {
    border-top: 1px solid %(line)s;
}
/* The startup error window's Report panel and its Recent log box. */
QFrame#reportPanel {
    background-color: %(raised)s;
    border: 1px solid %(line)s;
    border-radius: 2px;
}
QTextBrowser#recentLog {
    background-color: %(field)s;
    border: 1px solid %(edge)s;
    border-radius: 2px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: %(field)s;
    color: %(ink)s;
    border: 1px solid %(edge)s;
    border-radius: 4px;
    padding: 0 9px 0 10px;
    min-height: 28px;
    selection-background-color: %(accent)s;
    selection-color: %(on_accent)s;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: %(muted)s;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: %(accent)s;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
    background-color: %(ground)s;
    color: %(muted)s;
    border-color: %(line)s;
}
QLineEdit#mono {
    font-family: "Consolas";
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
}
/* The edit field already stops left of the 24 px step buttons (a wider
   right padding clipped "16 pt" in an 84 px field); max-height keeps the
   windows11 base hint (34 px) to the 30 px field height. The windows11
   style insets an embedded line edit by about 5 px more than a plain one,
   so the left padding is smaller to start the value 11 px in (mockup). */
QSpinBox, QDoubleSpinBox {
    padding-left: 5px;
    padding-right: 2px;
    max-height: 28px;
}
QComboBox:editable {
    padding-left: 5px;
}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: none;
    background: transparent;
    width: 24px;
}
QSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
}
QComboBox QAbstractItemView {
    background-color: %(field)s;
    color: %(ink)s;
    border: 1px solid %(edge)s;
    outline: none;
    selection-background-color: %(hover)s;
    selection-color: %(ink)s;
}
/* The tray flyout's Position picker rings only for keyboard focus, like the
   panel's painted controls: it paints that ring itself. */
QWidget#trayFlyout QComboBox:focus {
    border-color: %(edge)s;
}
QWidget#trayFlyout QComboBox:focus:hover {
    border-color: %(muted)s;
}
/* Its open list marks the current row Lit, like the menus (the native style
   draws the system accent there); the font and row metrics keep the native
   list's 32 px rows and text inset. */
QWidget#trayFlyout QComboBox QAbstractItemView {
    font-family: "Segoe UI Variable Text";
    font-size: 13px;
}
QWidget#trayFlyout QComboBox QAbstractItemView::item {
    min-height: 32px;
    padding: 0px 6px;
}
QWidget#trayFlyout QComboBox QAbstractItemView::item:selected {
    background-color: %(accent)s;
    color: %(on_accent)s;
}
QSlider::groove:horizontal {
    background-color: %(rail)s;
    height: 2px;
    border-radius: 1px;
}
QSlider::sub-page:horizontal {
    background-color: %(ink)s;
    height: 2px;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    background-color: %(raised)s;
    border: 2px solid %(ink)s;
    width: 12px;
    height: 12px;
    margin: -7px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal:disabled {
    background-color: %(muted)s;
}
QSlider::handle:horizontal:disabled {
    border-color: %(muted)s;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: %(edge)s;
    min-height: 24px;
    margin: 0 2px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background-color: %(muted)s;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QPushButton {
    background-color: %(field)s;
    color: %(ink)s;
    border: 1px solid %(edge)s;
    border-radius: 4px;
    padding: 0 16px;
    min-height: 28px;
    min-width: 54px;
}
QPushButton:hover, QPushButton:pressed {
    background-color: %(hover)s;
}
QPushButton:disabled {
    background-color: %(ground)s;
    color: %(muted)s;
    border-color: %(line)s;
}
QPushButton[primary="true"] {
    background-color: %(accent)s;
    border-color: %(accent)s;
    color: %(on_accent)s;
    font-weight: 600;
}
QPushButton[primary="true"]:hover, QPushButton[primary="true"]:pressed {
    background-color: %(accent_hover)s;
    border-color: %(accent_hover)s;
}
QPushButton[primary="true"]:disabled {
    background-color: %(ground)s;
    color: %(muted)s;
    border-color: %(line)s;
}
QDialog#compactDialog QPushButton, QDialog#compactDialog QLineEdit {
    border-radius: 2px;
}
/* Setup (opt-wiz-rail.tpl): the ground, 32 px buttons at least 96 wide with
   2 px corners (QSS sizes the content box: 30 = 32 - the borders, 62 = 96 -
   the 16 px paddings and the borders); a disabled button, the primary
   included, sits on the hover face. The alert box is a raised panel. */
QDialog#setupWizard {
    background-color: %(ground)s;
}
QDialog#setupWizard QPushButton {
    min-height: 30px;
    min-width: 62px;
    border-radius: 2px;
}
QDialog#setupWizard QPushButton:disabled {
    background-color: %(hover)s;
    color: %(muted)s;
    border-color: %(line)s;
}
/* A compact dialog opened from setup ("Couldn't save your settings", parented
   to the wizard or a page) keeps its own 30 px buttons and ground disabled
   face: a descendant selector follows the parent chain into the dialog's
   window, so the two setup rules above would reach its buttons. The values
   restate the base QPushButton rule's min-height and min-width and the base
   QPushButton:disabled rule's background (the first button rules above):
   change them together. */
QDialog#setupWizard QDialog#compactDialog QPushButton {
    min-height: 28px;
    min-width: 54px;
}
QDialog#setupWizard QDialog#compactDialog QPushButton:disabled {
    background-color: %(ground)s;
}
QFrame#wizardAlert {
    background-color: %(raised)s;
    border: 1px solid %(line)s;
    border-radius: 2px;
}
/* Setup's Sound device list (opt-wiz-rail.tpl .devs): the scroll area owns
   the closed frame, so its edge stays put and the scroll bar sits inside it
   while more than five rows scroll; the list in it draws no frame. */
QScrollArea#deviceList {
    background-color: %(field)s;
    border: 1px solid %(edge)s;
    border-radius: 2px;
}
"""

# Tooltips: the Settings window's and the flyout's (through window_qss) and
# the series bar chip's.
_TOOLTIP_QSS = """\
QToolTip {
    background-color: %(raised)s;
    color: %(ink)s;
    border: 1px solid %(line)s;
}
"""

_MENU_QSS = """\
QMenu {
    background-color: %(raised)s;
    color: %(ink)s;
    border: 1px solid %(frame)s;
    border-radius: 2px;
    padding: 4px;
    font-family: "Segoe UI Variable Text";
    font-size: 13px;
}
QMenu::item {
    background-color: transparent;
    min-height: 30px;
    padding: 0px 12px;
    border-radius: 3px;
}
QMenu::item:selected {
    background-color: %(accent)s;
    color: %(on_accent)s;
}
QMenu::item:disabled {
    color: %(muted)s;
}
QMenu::separator {
    height: 1px;
    background-color: %(line)s;
    margin: 4px 0px;
}
"""

_HINT_QSS = """\
QLabel {
    background-color: %(raised)s;
    color: %(ink)s;
    border: 1px solid %(line)s;
    border-radius: 10px;
    padding: 8px 18px;
    font-size: 14px;
}
"""


def window_qss(p: dict) -> str:
    """QSS for the Settings window and its compact dialogs (panels, fields,
    slider, buttons, the primary button, the dialogs' 2 px corners and
    tooltips)."""
    return (_WINDOW_QSS + _TOOLTIP_QSS) % p


def tooltip_qss(p: dict) -> str:
    """QSS for a widget's tooltips (raised ground, ink text, line border),
    for widgets outside a window_qss window (the series bar's chip)."""
    return _TOOLTIP_QSS % p


def menu_qss(p: dict) -> str:
    """QSS for every Yamabiko QMenu (the tray's right-click menu, the flyout's Troubleshooting menu): raised surface, frame border, 30 px rows, Lit selection, line separators."""
    return _MENU_QSS % p


def hint_qss(p: dict) -> str:
    """QSS for the placement hint bar (raised bg, ink fg, 1px line border)."""
    return _HINT_QSS % p
