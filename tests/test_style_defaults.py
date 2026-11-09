import pytest

from yamabiko.overlay_app.style_defaults import (
    MAX_WIDTH_PCT_RANGE,
    STYLE_DEFAULTS,
    coerce_max_width_pct,
    parse_hex_rgb,
)


def test_defaults_have_all_keys():
    expected = {
        "jp_font", "jp_size", "en_font", "en_size",
        "text_color", "bg_color", "bg_opacity", "padding",
        "max_width_pct",
        "max_lines", "hold_ms", "fade_in_ms", "fade_out_ms",
        "mode", "alignment", "echo", "banner",
    }
    assert set(STYLE_DEFAULTS.keys()) == expected


def test_defaults_match_telop_values():
    """The approved overlay's defaults (spec Surface 1): 21 px / 31 px are
    16 pt / 23 pt, dark bar #0e1014 at 93%, Watch / Left / Priority / 日本語.
    Each default is also the first STYLE_CHOICES entry for its key."""
    from yamabiko.overlay_app.style_defaults import STYLE_CHOICES

    assert STYLE_DEFAULTS["jp_font"] == "BIZ UDPGothic"
    assert STYLE_DEFAULTS["jp_size"] == 16
    assert STYLE_DEFAULTS["en_font"] == "Bahnschrift"
    assert STYLE_DEFAULTS["en_size"] == 23
    assert STYLE_DEFAULTS["text_color"] == "#ffffff"
    assert STYLE_DEFAULTS["bg_color"] == "#0e1014"
    assert STYLE_DEFAULTS["bg_opacity"] == 237
    assert STYLE_DEFAULTS["padding"] == 10
    assert STYLE_DEFAULTS["max_lines"] == 3
    assert STYLE_DEFAULTS["hold_ms"] == 5000
    assert STYLE_DEFAULTS["fade_in_ms"] == 160
    assert STYLE_DEFAULTS["fade_out_ms"] == 420
    assert (STYLE_DEFAULTS["mode"], STYLE_DEFAULTS["alignment"],
            STYLE_DEFAULTS["echo"], STYLE_DEFAULTS["banner"]) == (
        "watch", "left", "priority", "jp")
    for key, values in STYLE_CHOICES.items():
        assert STYLE_DEFAULTS[key] == values[0]


# --------------------------------------------------------------------------
# parse_hex_rgb: the ONE shared #rrggbb parser (theme._validate_palette and
# the overlay's colour handling both route through it -- no duplicate parsers).
# --------------------------------------------------------------------------

def test_parse_hex_rgb_parses_valid_six_digit_hex():
    assert parse_hex_rgb("#ffffff") == (255, 255, 255)
    assert parse_hex_rgb("#000000") == (0, 0, 0)
    assert parse_hex_rgb("#10A5ff") == (0x10, 0xA5, 0xFF)  # case-insensitive


@pytest.mark.parametrize(
    "bad",
    ["#fff", "ffffff", "#gggggg", "#ffffff0", "", None, 0xFFFFFF, ["#ffffff"]],
    ids=["shorthand", "no-hash", "non-hex", "too-long", "empty", "none",
         "int", "list"],
)
def test_parse_hex_rgb_returns_none_for_invalid_input(bad):
    # Callers own their fallback/raise policy, so the parser itself must
    # NEVER raise: every invalid shape (including non-str types) -> None.
    # Seeded defect: lstrip before validating -> None crashes with
    # AttributeError instead of returning None, red on the 'none' case.
    assert parse_hex_rgb(bad) is None


# --------------------------------------------------------------------------
# coerce_max_width_pct: the one sanitizer for every Python-side consumer of
# max_width_pct.
# --------------------------------------------------------------------------

def test_coerce_max_width_pct_clamps_to_the_canonical_range():
    lo, hi = MAX_WIDTH_PCT_RANGE
    assert coerce_max_width_pct(lo - 1) == lo
    assert coerce_max_width_pct(hi + 500) == hi
    assert coerce_max_width_pct(55) == 55
    assert coerce_max_width_pct(55.9) == 55  # int() truncation inside range


@pytest.mark.parametrize(
    "junk",
    [True, False, "70", None, float("nan"), float("inf"), float("-inf"),
     10 ** 309, -(10 ** 309)],
    ids=["true", "false", "str", "none", "nan", "inf", "-inf",
         "huge-int", "-huge-int"],
)
def test_coerce_max_width_pct_rejects_non_finite_and_non_numbers(junk):
    # bool is explicitly rejected (isinstance(True, int) is True, so YAML
    # `true` would otherwise clamp to the range floor); NaN/Inf pass the
    # float isinstance check but detonate int(width * pct / 100) consumers.
    # An int above the float ceiling (~1.8e308) makes math.isfinite raise
    # OverflowError -- unguarded, that escaped as a crash instead of a
    # keep-default (the same huge-integer class rejected by config numeric
    # rules; reachable at render time via a loopback WS set_style big-int).
    assert coerce_max_width_pct(junk) == STYLE_DEFAULTS["max_width_pct"]
