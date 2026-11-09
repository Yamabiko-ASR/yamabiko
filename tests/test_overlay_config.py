import pytest

from yamabiko.config import load_overlay_config, OverlayConfig


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    """Isolated app-data dir for the NO-ARG (production) loader path.

    load_overlay_config() with no argument delegates to load_full_config,
    which reads the SHIPPED defaults + user_overrides.yaml under
    YAMABIKO_APPDATA -- the path overlay_app/__main__.py actually takes
    on launch. The legacy explicit-path tests below never touch the
    shipped defaults, so they cannot catch a bad value shipped in
    default_config.yaml (TESTING.md §3.2 wrong-path).
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    return tmp_path


def test_missing_block_returns_defaults_via_explicit_path_merge(tmp_path):
    """The explicit path= argument now routes through the shared
    defaults<-explicit merge (not a verbatim read): a --config file with
    no overlay section resolves the overlay slice from the SHIPPED
    defaults, so jp_size is the shipped 16 and geometry the shipped null.

    Because the slice comes from default_config.yaml, this DOES depend on
    what the bundle ships (unlike the old verbatim contract); the no-arg
    production path is additionally covered by
    test_fresh_install_no_arg_load_returns_geometry_none."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("asr:\n  model: qwen\n", encoding="utf-8")
    oc = load_overlay_config(cfg)
    assert isinstance(oc, OverlayConfig)
    assert oc.style["jp_size"] == 16
    assert oc.geometry is None


def test_fresh_install_no_arg_load_returns_geometry_none(appdata):
    """On a fresh install (no user_overrides.yaml), the PRODUCTION loader
    path -- load_overlay_config() with no argument, the call
    overlay_app/__main__.py makes -- must yield geometry=None so the
    overlay computes a per-screen bottom-center placement. A concrete
    x/y/width/height block in the shipped default would land the overlay
    at a dev-machine position on most resolutions (the 75%%-visibility
    fallback only rescues screens smaller than that rect).

    Seeded defect (one line): revert default_config.yaml's
    ``geometry: null`` back to a concrete x/y/width/height block ->
    geometry is populated from the shipped default -> red.
    """
    oc = load_overlay_config()
    assert oc.geometry is None, (
        f"fresh install must have no saved geometry (got {oc.geometry!r}); "
        f"a concrete geometry block in default_config.yaml is dev-machine "
        f"state shipping as a default"
    )


def test_user_saved_geometry_round_trips_through_no_arg_load(appdata):
    """A legacy overlay.geometry override in user_overrides.yaml (from an
    older install) must survive the no-arg load. With the shipped
    default now ``geometry: null``, the unknown-key walker must NOT recurse
    a user's saved geometry dict against the null default and strip its
    x/y/width/height as four 'unknown' sub-keys -- that would silently
    reset the overlay position on every launch.

    Seeded defect (one line): in config._walk_unknown_keys, drop the
    ``isinstance(defaults[k], dict)`` recursion guard -> the saved dict's
    sub-keys are flagged unknown and stripped -> geometry is None -> red.
    """
    from yamabiko.config_writer import write_override

    # A legacy overlay.geometry override from an older install, written via
    # the low-level primitive (write_overlay_geometry was retired). It must
    # survive the no-arg load so the __main__ migration can read + prune it.
    saved = {"x": 100, "y": 200, "width": 800, "height": 300}
    write_override(["overlay", "geometry"], saved)
    oc = load_overlay_config()
    assert oc.geometry == saved, (
        f"user-saved geometry did not round-trip through the production "
        f"no-arg loader: got {oc.geometry!r}, wrote {saved!r}"
    )


def test_reads_style_block(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  style:\n"
        "    jp_size: 20\n"
        "    en_size: 28\n",
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.style["jp_size"] == 20
    assert oc.style["en_size"] == 28
    assert oc.style["jp_font"] == "BIZ UDPGothic"


def test_clamps_out_of_range(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  style:\n"
        "    jp_size: 999\n"
        "    hold_ms: -100\n",
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.style["jp_size"] == 48
    assert oc.style["hold_ms"] == 500


def test_max_width_pct_clamped_at_config_boundary():
    """The max_width_pct style key must clamp to 30..100 at the config
    boundary and default to 70 when absent.

    Seeded defects (each one line, each goes red):
    - remove ``"max_width_pct": (30, 100)`` from config._STYLE_RANGES ->
      20 passes through unclamped -> first assertion fails.
    - remove ``"max_width_pct": 70`` from STYLE_DEFAULTS -> _clamp_style
      skips the unknown key (and the {} default) -> KeyError / wrong
      default -> red.
    """
    from yamabiko.config import _clamp_style
    assert _clamp_style({"max_width_pct": 20})["max_width_pct"] == 30
    assert _clamp_style({"max_width_pct": 150})["max_width_pct"] == 100
    assert _clamp_style({})["max_width_pct"] == 70


def test_style_ranges_max_width_pct_is_the_shared_constant():
    """K3: config._STYLE_RANGES must reference style_defaults.
    MAX_WIDTH_PCT_RANGE -- the single source shared with position_config's
    migration squeeze, the Settings "Max width" slider, and _cap_px -- not a
    re-typed (30, 100) literal that can drift when the canonical window moves.
    Identity (``is``), not equality: an equal-but-separate tuple IS the drift
    this pins against.

    Seeded defect (one line): revert the _STYLE_RANGES entry to the literal
    ``(30, 100)`` -> equal but not identical -> red.
    """
    from yamabiko.config import _STYLE_RANGES
    from yamabiko.overlay_app.style_defaults import MAX_WIDTH_PCT_RANGE
    assert _STYLE_RANGES["max_width_pct"] is MAX_WIDTH_PCT_RANGE


def test_non_numeric_ranged_value_keeps_shipped_default():
    """A present-but-non-numeric OR non-finite value on ANY ranged style key
    must keep the shipped default, not fall through unclamped. An unclamped
    None/"bad" reaches OverlayWindow._cap_px()'s int(width * pct / 100) and
    raises TypeError on the next add_subtitle; NaN/Infinity pass isinstance
    but raise ValueError/OverflowError under int(). Either way every subtitle
    is silently killed. Proven across the whole ranged-key class, not just
    max_width_pct.

    Seeded-defect drill: revert the ``isinstance(v, bool) or not
    isinstance(v, (int, float))`` branch in _clamp_style (let the old
    ``rng is not None and isinstance(v, (int, float))`` gate fall through to
    ``out[k] = v``) -> None/"bad" pass through and their assertions read
    None/"bad" instead of the default -> red. Separately, drop the
    ``or not math.isfinite(v)`` clause -> the NaN/Infinity assertions below
    read nan/inf instead of the default -> red.
    """
    from yamabiko.config import _clamp_style
    # max_width_pct: the key that motivated the fix.
    assert _clamp_style({"max_width_pct": None})["max_width_pct"] == 70
    assert _clamp_style({"max_width_pct": "bad"})["max_width_pct"] == 70
    # A SECOND ranged key proves the fix is class-wide, not key-specific.
    assert _clamp_style({"jp_size": None})["jp_size"] == 16
    # bool is not a valid number here (isinstance(True, int) is True in
    # Python); it must be rejected, not clamped to 1.
    assert _clamp_style({"max_width_pct": True})["max_width_pct"] == 70
    # NaN/Infinity ARE floats (they pass isinstance(v, (int, float))) but
    # detonate int(width * nan / 100) with a ValueError downstream, and
    # json.loads accepts the NaN/Infinity literals so a loopback WS set_style
    # frame can carry them. The isfinite guard must reject both -> default.
    assert _clamp_style({"max_width_pct": float("nan")})["max_width_pct"] == 70
    assert _clamp_style({"max_width_pct": float("inf")})["max_width_pct"] == 70
    assert _clamp_style({"jp_size": float("nan")})["jp_size"] == 16


def test_invalid_colors_fall_back_to_defaults(tmp_path):
    """A malformed bg_color/text_color in the config must fall back to
    the shipped default at the config boundary, not flow through to the overlay
    (where the hex parse crashed the render path and silently dropped every
    subtitle).

    Seeded-defect drill: remove the COLOR_STYLE_KEYS branch in _clamp_style ->
    the bad values pass through verbatim -> these assertions fail.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  style:\n"
        "    bg_color: red\n"          # not #RRGGBB
        "    text_color: '#fff'\n",    # 3-digit shorthand (parser needs 6)
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.style["bg_color"] == "#0e1014"
    assert oc.style["text_color"] == "#ffffff"


def test_valid_colors_are_kept(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n  style:\n    bg_color: '#123456'\n    text_color: '#ABCDEF'\n",
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.style["bg_color"] == "#123456"
    assert oc.style["text_color"] == "#ABCDEF"


def test_reads_geometry(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  geometry:\n"
        "    x: 100\n"
        "    y: 200\n"
        "    width: 800\n"
        "    height: 300\n",
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.geometry == {"x": 100, "y": 200, "width": 800, "height": 300}


def test_shipped_theme_and_position_defaults(appdata):
    """The PRODUCTION no-arg loader must surface the shipped overlay.theme
    and overlay.position defaults, clamped.

    This is hard rule #4's verified consumer for the two new
    default_config.yaml keys: load_overlay_config IS the consumer this
    task provides. The position assertion also catches a wiring miss --
    if load_overlay_config forgets ``position=`` the field is {} (the
    dataclass default), which is not the clamped defaults dict -> red.

    Note: the theme assertion here CANNOT catch a theme-wiring miss on its
    own -- OverlayConfig.theme also defaults to "system", so a load that
    never reads block['theme'] still returns "system". That §3.9
    Wired-not-exercised gap is closed by
    test_user_theme_and_position_flow_through_clamped below, which drives
    a NON-default theme end-to-end.

    Seeded defect (one line): change default_config.yaml's ``theme:
    system`` to ``theme: dark`` -> theme assertion red; delete the
    ``position:`` block -> position becomes {} clamped == full defaults,
    still matches, but deleting a single sub-key (e.g. v_offset_pct) makes
    the merged value diverge -> covered by the round-trip test.
    """
    oc = load_overlay_config()
    assert oc.theme == "system"
    assert oc.position == {
        "screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
        "left_offset_pct": 6.0,
    }


def test_user_theme_and_position_flow_through_clamped(appdata):
    """A user-saved NON-default theme and position must survive the no-arg
    production load, with the position clamped at the config boundary.

    This is the strong wiring test: it drives values that DIFFER from the
    dataclass defaults, so it fails if load_overlay_config forgets to read
    theme/position from the merged block, and it drives an OUT-OF-RANGE
    v_offset_pct so it fails if the clamp step is skipped.

    Seeded defects (each one line, each red):
    - drop ``theme=theme`` from the OverlayConfig(...) return -> theme is
      the "system" dataclass default, not "dark" -> red.
    - drop ``position=position`` -> position is {} -> KeyError/mismatch.
    - pass ``block.get("position")`` straight through instead of
      clamp_position(...) -> v_offset_pct stays 200 instead of 98.0 -> red.
    """
    from yamabiko.config_writer import write_override

    write_override(["overlay", "theme"], "dark")
    write_override(["overlay", "position", "screen"], "\\\\.\\DISPLAY2")
    write_override(["overlay", "position", "v_offset_pct"], 200)  # out of range

    oc = load_overlay_config()
    assert oc.theme == "dark"
    assert oc.position["screen"] == "\\\\.\\DISPLAY2"
    assert oc.position["v_offset_pct"] == 98.0   # clamped from 200
    assert oc.position["h_offset_pct"] == 50.0   # untouched shipped default


def test_shipped_reduce_motion_default_is_false(appdata):
    """Hard rule #4 consumer test for the new overlay.reduce_motion key: the
    PRODUCTION no-arg loader must surface the shipped default (False).

    load_overlay_config IS the consumer this task provides for the key
    settings_window.SettingsWindow reads. This alone cannot catch a
    wiring miss (OverlayConfig.reduce_motion also defaults to False), so
    test_user_reduce_motion_flows_through drives a NON-default value.

    Seeded defect (one line): change default_config.yaml's
    ``reduce_motion: false`` to ``reduce_motion: true`` -> red.
    """
    oc = load_overlay_config()
    assert oc.reduce_motion is False


def test_user_reduce_motion_flows_through(appdata):
    """A user-saved overlay.reduce_motion=True must survive the no-arg
    production load. This is the strong wiring test: it drives a value that
    DIFFERS from the dataclass default, so it fails if load_overlay_config
    forgets to read block['reduce_motion'].

    Seeded defect (one line): drop ``reduce_motion=reduce_motion`` from the
    OverlayConfig(...) return -> the field is the False dataclass default,
    not True -> red.
    """
    from yamabiko.config_writer import write_override

    write_override(["overlay", "reduce_motion"], True)
    oc = load_overlay_config()
    assert oc.reduce_motion is True


def test_invalid_theme_falls_back_to_system(tmp_path):
    """An unrecognized overlay.theme must fall back to "system" rather
    than reaching the chrome-theme resolver as garbage.

    Seeded defect (one line): remove the ``if theme not in (...)`` guard
    in load_overlay_config -> "chartreuse" flows through -> red.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text("overlay:\n  theme: chartreuse\n", encoding="utf-8")
    oc = load_overlay_config(cfg)
    assert oc.theme == "system"


def test_non_dict_style_and_position_degrade_to_defaults(tmp_path):
    """A truthy NON-dict overlay.style / overlay.position (a scalar or list)
    in an explicit --config must degrade to the shipped defaults without
    crashing boot -- NOT reach _clamp_style([1,2]).items() /
    clamp_position("garbage").get() and raise AttributeError.

    The explicit path= argument now routes through the shared
    defaults<-explicit merge, so _walk_type_mismatches strips a non-dict
    override sitting on a dict default (overlay.style / overlay.position)
    as a mapping mismatch BEFORE load_overlay_config sees it; the
    load_overlay_config isinstance guards remain as a backstop. Both
    layers must fail for the AttributeError to escape (defense in depth).

    Seeded defect: drop the mapping-mismatch branch in
    _walk_type_mismatches AND the ``if not isinstance(style_raw, dict)`` /
    ``if not isinstance(pos_raw, dict)`` guards in load_overlay_config ->
    the [1,2] / "garbage" values reach _clamp_style/clamp_position and
    raise AttributeError -> red.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        '  position: "garbage-string"\n'
        "  style:\n"
        "  - 1\n"
        "  - 2\n",
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)  # must not raise
    # style falls back to the full shipped defaults
    assert oc.style["jp_size"] == 16
    assert oc.style["max_width_pct"] == 70
    # position falls back to the clamped shipped defaults
    assert oc.position == {
        "screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
        "left_offset_pct": 6.0,
    }


def test_explicit_config_unknown_overlay_key_warns_via_merge(tmp_path, caplog):
    """Routing load_overlay_config's explicit path through the shared
    defaults<-explicit merge means an unknown overlay key in a --config
    file is now validated + warned, exactly like the no-arg production
    path -- the old verbatim read warned about nothing while still
    honoring the valid style value alongside it.

    Seeded defect: revert load_overlay_config's path branch to the
    verbatim yaml.safe_load (skip _validate_and_merge) -> no unknown-key
    warning is emitted -> red.
    """
    import logging

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  bogus_overlay_key: 5\n"
        "  style:\n"
        "    jp_size: 20\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        oc = load_overlay_config(cfg)
    # The valid style value still applies through the merge.
    assert oc.style["jp_size"] == 20
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("overlay.bogus_overlay_key" in m for m in msgs), (
        f"routing through _validate_and_merge must warn on the unknown "
        f"overlay key; got {msgs!r}"
    )


def test_non_dict_overlay_block_degrades_to_defaults(tmp_path):
    """A truthy non-dict ``overlay:`` block itself (e.g. ``overlay: "enabled"``)
    must degrade to defaults, not crash the very next ``block.get("style")``
    with AttributeError.

    Seeded defect (one line): drop the ``if not isinstance(block, dict)`` guard
    -> "enabled".get("style") raises AttributeError -> red.
    """
    cfg = tmp_path / "c.yaml"
    cfg.write_text('overlay: "enabled"\n', encoding="utf-8")
    oc = load_overlay_config(cfg)  # must not raise
    assert oc.style["jp_size"] == 16
    assert oc.theme == "system"


def test_style_choice_outside_its_values_keeps_default(tmp_path):
    """A choice key holding anything but one of its values (a typo in
    user_overrides.yaml, a loopback WS frame) falls back to the shipped
    default instead of reaching the renderers; valid values survive.

    Seeded defect: drop the STYLE_CHOICES branch in _clamp_style ->
    "banana" survives -> red."""
    from yamabiko.config import _clamp_style

    out = _clamp_style({"mode": "banana", "alignment": "centre",
                        "echo": "both", "banner": 7})
    assert out["mode"] == "watch"
    assert out["alignment"] == "left"
    assert out["echo"] == "both"
    assert out["banner"] == "jp"
