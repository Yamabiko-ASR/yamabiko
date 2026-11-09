"""Single source of default overlay style values. ``default_config.yaml``
ships the same values; no test compares the two, so change both together.
Imported by the config loader (fallback), the telop look, and the settings
window's Reset button.
"""
from __future__ import annotations

import math
import re

# Canonical #RRGGBB validator, shared by the config loader's _clamp_style and
# parse_hex_rgb (the telop look's colour parser) so the two never drift on
# what counts as a valid colour. parse_hex_rgb parses 6 hex digits, so #RGB
# shorthand is intentionally NOT accepted.
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
COLOR_STYLE_KEYS = ("text_color", "bg_color")
# Font-family keys. The overlay hands these straight to QFont(), which raises
# on a non-string, so the config boundary only accepts a non-empty string.
FONT_STYLE_KEYS = ("jp_font", "en_font")

STYLE_DEFAULTS: dict = {
    # Sizes are points (1 pt = 4/3 px): 16 pt / 23 pt are the approved
    # design's 21 px Japanese and 31 px English (telop_look.PT_TO_PX).
    "jp_font": "BIZ UDPGothic",
    "jp_size": 16,
    "en_font": "Bahnschrift",
    "en_size": 23,
    "text_color": "#ffffff",
    # The dark bar behind the newest English line: #0e1014 at 93%.
    "bg_color": "#0e1014",
    "bg_opacity": 237,
    "padding": 10,
    "max_width_pct": 70,
    "max_lines": 3,
    "hold_ms": 5000,
    "fade_in_ms": 160,
    "fade_out_ms": 420,
    "mode": "watch",
    "alignment": "left",
    "echo": "priority",
    "banner": "jp",
}

# One-of-few style keys and their allowed values; the first value is the
# shipped default. Single source for config._clamp_style, the Settings
# segmented rows and telop_look (overlay.js keeps a hand copy).
STYLE_CHOICES: dict[str, tuple[str, ...]] = {
    "mode": ("watch", "learn"),
    "alignment": ("left", "center"),
    "echo": ("priority", "both"),
    "banner": ("jp", "jp_en"),
}

# Canonical max_width_pct window. Single source for config._STYLE_RANGES,
# position_config's migration squeeze, the Settings "Max width" slider, and
# OverlayWindow._cap_px. (overlay.js and overlay.css carry their own copies —
# keep them in sync by hand; they cannot import Python constants.)
MAX_WIDTH_PCT_RANGE: tuple[int, int] = (30, 100)


def coerce_max_width_pct(value) -> int:
    """Return ``value`` clamped to :data:`MAX_WIDTH_PCT_RANGE`, or the shipped
    default for anything that is not a real, finite number.

    bool is rejected explicitly (``isinstance(True, int)`` is True in Python,
    so YAML ``true`` would otherwise clamp to the range floor), and NaN/Inf are
    rejected because they pass ``isinstance(v, float)`` yet detonate the
    ``int(width * pct / 100)`` consumers. This is the one sanitizer for every
    Python-side consumer of the key; do not re-implement it inline.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return STYLE_DEFAULTS["max_width_pct"]
    try:
        finite = math.isfinite(value)
    except OverflowError:
        # An int beyond the float ceiling (~1.8e308): isfinite raises
        # instead of returning False; treat it like any other non-finite.
        finite = False
    if not finite:
        return STYLE_DEFAULTS["max_width_pct"]
    lo, hi = MAX_WIDTH_PCT_RANGE
    return int(min(hi, max(lo, value)))


def parse_hex_rgb(value) -> tuple[int, int, int] | None:
    """Parse a ``#rrggbb`` string to an ``(r, g, b)`` tuple, or ``None`` when
    the value is not a valid 6-digit hex colour (wrong type, shorthand,
    garbage). Callers own their fallback/raise policy; this stays pure."""
    if not isinstance(value, str) or not HEX_COLOR_RE.match(value):
        return None
    h = value.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
