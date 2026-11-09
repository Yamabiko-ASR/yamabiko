"""Tests for the chrome token module (overlay_app/theme.py).

theme.py is a PURE module: token access, mode resolution, and the
QSS builders never need a QApplication. Only ``system_scheme_probe`` touches
Qt (lazy import), and even that is exercised here through its documented
contract (returns a valid mode, never raises) rather than a mock, so the
whole file runs without constructing an app.

The probe passed to ``resolve_mode`` is a genuine dependency-injection seam
(the function takes ``scheme_probe`` as a parameter), so passing a lambda or
a sentinel is legitimate DI, not subject-mocking.
"""
import re

import pytest

from yamabiko.overlay_app.style_defaults import HEX_COLOR_RE
from yamabiko.overlay_app.theme import (
    MIST,
    NIGHT,
    _TOKEN_KEYS,
    _scheme_from_app,
    _validate_palette,
    menu_qss,
    palette_for,
    resolve_mode,
    system_scheme_probe,
    window_qss,
)

_BUILDERS = (window_qss, menu_qss)


def _boom_probe():
    """A scheme probe that fails the test if it is ever consulted."""
    raise AssertionError("scheme_probe must NOT be called when cfg forces a mode")


# --------------------------------------------------------------------------
# resolve_mode
# --------------------------------------------------------------------------

def test_resolve_mode_forced_light_ignores_probe():
    # cfg 'light' forces the result; the probe must not be consulted.
    # Seeded defect: make resolve_mode fall through to the probe for 'light'
    # -> _boom_probe raises and this goes red.
    assert resolve_mode("light", _boom_probe) == "light"


def test_resolve_mode_forced_dark_ignores_probe():
    assert resolve_mode("dark", _boom_probe) == "dark"


def test_resolve_mode_system_routes_to_probe_and_honors_light():
    # 'system' delegates to the probe and trusts a 'light' answer. Seeded
    # defect: hardcode resolve_mode to return 'dark' -> red.
    assert resolve_mode("system", lambda: "light") == "light"


def test_resolve_mode_system_routes_to_probe_and_honors_dark():
    assert resolve_mode("system", lambda: "dark") == "dark"


def test_resolve_mode_garbage_cfg_routes_to_probe():
    # Any non-light/dark cfg (not only the literal 'system') delegates to the
    # probe. Seeded defect: special-case only 'system' for delegation and
    # default everything else to 'dark' -> this returns 'dark' != 'light',
    # red.
    assert resolve_mode("chartreuse", lambda: "light") == "light"


def test_resolve_mode_garbage_probe_return_defaults_dark():
    # The probe is trusted only for 'light'/'dark'; any other return -> dark.
    # Seeded defect: ``return probed`` verbatim -> 'banana' leaks and red.
    assert resolve_mode("system", lambda: "banana") == "dark"


# --------------------------------------------------------------------------
# system_scheme_probe (real path: Qt branch if an app exists, else winreg)
# --------------------------------------------------------------------------

def test_system_scheme_probe_returns_valid_mode_without_raising():
    # Real-path exercise of the actual probe: no mock. Depending on test
    # ordering a QGuiApplication may or may not exist; either the Qt
    # colorScheme branch or the Windows-registry / non-Windows fallback runs.
    # Every path must yield a valid mode and never raise. Seeded defect: make
    # the registry fallback return 'banana' -> red on a machine with no app.
    assert system_scheme_probe() in ("light", "dark")


def test_scheme_from_app_survives_non_gui_app_without_style_hints():
    # QGuiApplication.instance() returns whatever QCoreApplication exists, so
    # in a headless host the "app" handed to _scheme_from_app can be a bare
    # non-GUI object with NO styleHints at all. The Qt branch must treat that
    # as unavailable (None -> registry fallback), because the probe's
    # contract is "never raises". The fake app is legitimate DI: the function
    # takes the app as a parameter precisely so this guard is testable
    # without patching Qt statics. Seeded defect: hoist the styleHints() read
    # above the QGuiApplication type guard -> AttributeError, red.
    class _CoreOnlyApp:
        pass  # deliberately no styleHints attribute

    assert _scheme_from_app(_CoreOnlyApp()) is None  # bows out cleanly
    assert _scheme_from_app(None) is None            # pre-app call, same path
    assert system_scheme_probe() in ("light", "dark")  # fallback still valid


# --------------------------------------------------------------------------
# palette_for (forced modes only -> deterministic, no probe/Qt)
# --------------------------------------------------------------------------

def test_palette_for_forced_dark_is_night():
    assert palette_for("dark") is NIGHT


def test_palette_for_forced_light_is_mist():
    assert palette_for("light") is MIST


# --------------------------------------------------------------------------
# palette integrity
# --------------------------------------------------------------------------

def test_palettes_share_exact_key_set():
    # Copy-paste drift guard: a token present in one palette but not the other
    # is a latent KeyError inside a builder or painter. Seeded defect: drop
    # 'hover' from MIST -> key sets differ, red.
    assert set(NIGHT) == set(MIST)
    assert set(NIGHT) == set(_TOKEN_KEYS) | {"name"}


def test_palette_names_are_the_expected_slugs():
    assert NIGHT["name"] == "night"
    assert MIST["name"] == "mist"


def test_validate_palette_accepts_both_shipped_palettes():
    # The import-time anti-drift guard must pass the shipped literals
    # (if it did not, the module would not even import).
    _validate_palette(NIGHT)
    _validate_palette(MIST)


def test_validate_palette_rejects_key_drift():
    # Seeded defect: loosen the guard to a subset check -> a palette missing
    # 'on_accent' slips through as a latent painter KeyError, red.
    bad = {k: v for k, v in NIGHT.items() if k != "on_accent"}
    with pytest.raises(ValueError):
        _validate_palette(bad)


def test_validate_palette_rejects_unparseable_colour_token():
    # Every colour token must parse through the shared parse_hex_rgb so
    # painters can trust palette-fed inputs. Seeded defect: drop the parse
    # loop from the guard -> a '#f00' shorthand literal survives import and
    # detonates later in a QColor or QSS consumer, red.
    with pytest.raises(ValueError):
        _validate_palette({**NIGHT, "accent": "#f00"})


@pytest.mark.parametrize("palette", [NIGHT, MIST], ids=["night", "mist"])
def test_palette_token_values_are_valid_hex(palette):
    for key in _TOKEN_KEYS:
        assert HEX_COLOR_RE.match(palette[key]), f"{palette['name']}.{key}={palette[key]!r}"


def test_night_and_mist_ground_differ():
    # Tripwire against a copy-paste that left both palettes identical. Seeded
    # defect: set MIST['ground'] = NIGHT['ground'] -> red.
    assert NIGHT["ground"] != MIST["ground"]


def test_new_chrome_tokens_carry_the_mockup_values():
    # The Settings mockups' slider/progress rail, primary hover, warning and
    # window-edge tones. Seeded defect: swap a Mist value -> red.
    assert {k: MIST[k] for k in ("rail", "accent_hover", "warn", "warn_mark", "frame")} == {
        "rail": "#d3d2cc", "accent_hover": "#b53a27", "warn": "#9d5d00",
        "warn_mark": "#f2b33d", "frame": "#bdbdb9"}
    assert {k: NIGHT[k] for k in ("rail", "accent_hover", "warn", "warn_mark", "frame")} == {
        "rail": "#3a3f3b", "accent_hover": "#e8664f", "warn": "#e3a33b",
        "warn_mark": "#e3a33b", "frame": "#393b38"}


def _qss_rules(qss: str) -> dict[str, str]:
    """{selector list: body} for every rule of a QSS template."""
    return {m.group(1).strip(): m.group(2) for m in re.finditer(r"([^{}]+)\{([^}]*)\}", qss)}


def test_window_qss_uses_the_mockup_corner_radii():
    # Mockup corners: 4 px for controls, 6 px for panels, 2 px for the compact
    # dialogs' buttons and fields. Seeded defects: leave panels at 4 px, or
    # drop the dialog override -> red.
    rules = _qss_rules(window_qss(MIST))
    assert "border-radius: 6px" in rules["QFrame#panel"]
    assert "border-radius: 4px" in rules["QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit"]
    assert "border-radius: 4px" in rules["QPushButton"]
    assert "border-radius: 2px" in rules["QDialog#compactDialog QPushButton, QDialog#compactDialog QLineEdit"]


# --------------------------------------------------------------------------
# QSS builders (structure identical; only token values differ)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("builder", _BUILDERS, ids=lambda b: b.__name__)
@pytest.mark.parametrize(
    "palette,other", [(NIGHT, MIST), (MIST, NIGHT)], ids=["night", "mist"]
)
def test_builder_renders_own_palette_and_no_foreign_hex(builder, palette, other):
    # Each builder must render from the passed palette's tokens and leak NONE
    # of the other palette's hexes. Seeded defect: hardcode a MIST hex into a
    # NIGHT builder line -> the 'no foreign hex' assert goes red.
    qss = builder(palette)
    assert palette["raised"] in qss
    assert palette["accent"] in qss
    for key in _TOKEN_KEYS:
        assert other[key] not in qss, f"{builder.__name__} leaked {other['name']}.{key}"


@pytest.mark.parametrize("builder", _BUILDERS, ids=lambda b: b.__name__)
def test_no_builder_uses_gradients(builder):
    assert "gradient" not in builder(NIGHT)
    assert "gradient" not in builder(MIST)


def test_menu_qss_separator_uses_line_token():
    # The separator draws in the line token (not ink/muted). Seeded defect:
    # style the separator with %(muted)s -> red.
    qss = menu_qss(NIGHT)
    separator = r"QMenu::separator\s*\{[^}]*background-color:\s*" + re.escape(NIGHT["line"])
    assert re.search(separator, qss)


def test_menu_qss_lights_the_selected_row_and_sets_30_px_rows():
    # Decision 62: Lit selection (accent fill, on-accent text) and 30 px rows.
    # Seeded defects: the old hover fill with an accent bar, or no row
    # height -> red.
    qss = menu_qss(MIST)
    selected = re.search(r"QMenu::item:selected\s*\{([^}]*)\}", qss).group(1)
    assert f"background-color: {MIST['accent']}" in selected
    assert f"color: {MIST['on_accent']}" in selected
    item = re.search(r"QMenu::item\s*\{([^}]*)\}", qss).group(1)
    assert "min-height: 30px" in item


def test_setup_buttons_are_32_by_96_and_disabled_ones_sit_on_the_hover_face(qtbot):
    # opt-wiz-rail.tpl .btn under the window QSS: 32 px tall, at least 96
    # wide, a disabled primary on the hover face. Seeded defects: leave the
    # Settings 30 px height, the 54 px minimum, or the ground disabled face
    # -> red.
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage, QRegion
    from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QWidget

    from yamabiko.overlay_app.window_chrome import apply_window_qss

    dialog = QDialog()
    dialog.setObjectName("setupWizard")
    qtbot.addWidget(dialog)
    row = QHBoxLayout(dialog)
    back = QPushButton("Back")
    primary = QPushButton("Next")
    primary.setProperty("primary", True)
    primary.setEnabled(False)
    row.addWidget(back)
    row.addWidget(primary)
    apply_window_qss(dialog, MIST)
    dialog.show()
    assert (back.sizeHint().height(), back.sizeHint().width()) == (32, 96)
    primary.resize(primary.sizeHint())
    img = QImage(primary.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    primary.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert img.pixelColor(primary.width() // 2, 4).name() == MIST["hover"]
