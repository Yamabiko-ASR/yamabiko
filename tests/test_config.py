"""Out-of-range and invalid-geometry handling in config.load_overlay_config.

Existing test_overlay_config.py covers the happy paths; this module specifically
targets the degenerate inputs that should clamp (with a WARNING log) or fall
back silently rather than corrupt the overlay.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
import yaml

from yamabiko import paths
from yamabiko.config import (
    ConfigValidationError,
    NumericRule,
    _NUMERIC_RULES,
    load_full_config,
    load_overlay_config,
    validate_config_semantics,
)


R = NumericRule

EXPECTED_NUMERIC_RULES = {
    "overlay.websocket_port": R(1, 65535, integer_only=True),
    "audio.block_ms": R(1, integer_only=True),
    "audio.idle_death_s": R(1, integer_only=True),
    "vad.threshold": R(0.0, 1.0),
    "vad.min_silence_ms": R(1, integer_only=True),
    "vad.speech_pad_ms": R(0, integer_only=True),
    "vad.min_chunk_ms": R(1, integer_only=True),
    "vad.max_chunk_ms": R(1, integer_only=True),
    "vad.glue_max_silence_ms": R(1, integer_only=True),
    "vad.attach_max_ms": R(1, integer_only=True),
    "vad.drop_floor_prob": R(0.0, 1.0),
    "vad.drop_floor_ms": R(1, integer_only=True),
    "vad.dip_split_threshold": R(0.0, 1.0),
    "vad.dip_split_min_piece_ms": R(1, integer_only=True),
    "asr.boundary_pad_ms": R(0, integer_only=True),
    "asr.faster_whisper.beam_size": R(1, integer_only=True),
    "asr.faster_whisper.no_speech_only_threshold": R(0.0, 1.0, minimum_inclusive=False),
    "translation.max_tokens": R(1, integer_only=True),
    "translation.temperature": R(0.0, 2.0),
    "translation.context_lines": R(0, integer_only=True),
    "translation.startup_readiness_timeout_s": R(
        0.0, minimum_inclusive=False,
    ),
    "translation.startup_readiness_poll_s": R(
        0.0, minimum_inclusive=False,
    ),
    "gazetteer.fuzzy_max_distance": R(0.0, 1.0),
    "lifecycle.health.poll_interval_s": R(
        0.0, minimum_inclusive=False,
    ),
    "lifecycle.health.fail_budget_s": R(
        0.0, minimum_inclusive=False,
    ),
    "lifecycle.grace_s.pipeline": R(0.0, minimum_inclusive=False),
    "lifecycle.grace_s.overlay": R(0.0, minimum_inclusive=False),
    "lifecycle.overlay.max_restarts": R(1, integer_only=True),
    "lifecycle.overlay.restart_window_s": R(
        0.0, minimum_inclusive=False,
    ),
}


INVALID_BOUNDARY_CASES = {
    "overlay.websocket_port": (0, 65536, 1.5),
    "audio.block_ms": (0, -1, 1.5),
    "audio.idle_death_s": (0, -1, 1.5),
    "vad.threshold": (-0.01, 1.01),
    "vad.min_silence_ms": (0, -1, 1.5),
    "vad.speech_pad_ms": (-1, 1.5),
    "vad.min_chunk_ms": (0, -1, 1.5),
    "vad.max_chunk_ms": (0, -1, 1.5),
    "vad.glue_max_silence_ms": (0, -1, 1.5),
    "vad.attach_max_ms": (0, -1, 1.5),
    "vad.drop_floor_prob": (-0.01, 1.01),
    "vad.drop_floor_ms": (0, -1, 1.5),
    "vad.dip_split_threshold": (-0.01, 1.01),
    "vad.dip_split_min_piece_ms": (0, -1, 1.5),
    "asr.boundary_pad_ms": (-1, 1.5),
    "asr.faster_whisper.beam_size": (0, -1, 1.5),
    "asr.faster_whisper.no_speech_only_threshold": (0.0, 1.5),
    "translation.max_tokens": (0, -1, 1.5),
    "translation.temperature": (-0.01, 2.01),
    "translation.context_lines": (-1, 1.5),
    "translation.startup_readiness_timeout_s": (0.0, -0.01),
    "translation.startup_readiness_poll_s": (0.0, -0.01),
    "gazetteer.fuzzy_max_distance": (-0.01, 1.01),
    "lifecycle.health.poll_interval_s": (0.0, -0.01),
    "lifecycle.health.fail_budget_s": (0.0, -0.01),
    "lifecycle.grace_s.pipeline": (0.0, -0.01),
    "lifecycle.grace_s.overlay": (0.0, -0.01),
    "lifecycle.overlay.max_restarts": (0, -1, 1.5),
    "lifecycle.overlay.restart_window_s": (0.0, -0.01),
}


def test_out_of_range_style_values_are_clamped_with_warning_log(tmp_path, caplog):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  style:\n"
        "    jp_size: 200\n"       # above max 48
        "    padding: -5\n"         # below min 0
        "    bg_opacity: 999\n"     # above max 255
        "    hold_ms: 100\n"        # below min 500
        ,
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        oc = load_overlay_config(cfg)

    assert oc.style["jp_size"] == 48
    assert oc.style["padding"] == 0
    assert oc.style["bg_opacity"] == 255
    assert oc.style["hold_ms"] == 500

    # At least one WARNING record must mention each clamped key. We don't
    # pin the exact wording — future rewording shouldn't fail this test.
    warnings = " | ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    for key in ("jp_size", "padding", "bg_opacity", "hold_ms"):
        assert key in warnings, f"expected clamp warning for {key!r}; got {warnings!r}"


def test_invalid_geometry_shape_falls_back_to_none(tmp_path):
    """Non-int or missing keys in overlay.geometry -> geometry=None (default repositioning)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  geometry:\n"
        "    x: 100\n"
        "    y: nope\n"     # non-int — should reject the block
        "    width: 800\n"
        "    height: 300\n",
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.geometry is None


def test_partial_geometry_block_falls_back_to_none(tmp_path):
    """Missing a required key also drops back to None rather than half-applying."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "overlay:\n"
        "  geometry:\n"
        "    x: 100\n"
        "    y: 200\n",      # width/height missing
        encoding="utf-8",
    )
    oc = load_overlay_config(cfg)
    assert oc.geometry is None


def test_malformed_yaml_does_not_crash(tmp_path, caplog):
    """A corrupt config.yaml must return defaults + log, not propagate YAMLError."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("overlay: {\n\n  style:  [this is not valid", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        oc = load_overlay_config(cfg)
    # Defaults survive — most importantly, caller doesn't see an exception.
    assert oc.geometry is None
    assert "jp_size" in oc.style


# ---------------------------------------------------------------------------
# Layered-config loader tests (load_full_config)
# ---------------------------------------------------------------------------


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    return tmp_path


def _shipped_defaults() -> dict:
    import yaml as _yaml
    return _yaml.safe_load(paths.shipped_defaults_path().read_text(encoding="utf-8"))


def test_load_full_config_no_overrides_returns_defaults(appdata, caplog):
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()
    assert merged == _shipped_defaults()
    # No warnings on the happy path.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"unexpected warnings: {[r.getMessage() for r in warnings]}"


def test_load_full_config_empty_overrides_returns_defaults(appdata, caplog):
    paths.user_overrides_path().write_text("{}\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()
    assert merged == _shipped_defaults()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []


def test_load_full_config_partial_override_deep_merges(appdata):
    paths.user_overrides_path().write_text(
        "overlay:\n  style:\n    jp_size: 18\n",
        encoding="utf-8",
    )
    merged = load_full_config()
    assert merged["overlay"]["style"]["jp_size"] == 18
    # Unrelated style keys come from defaults.
    defaults = _shipped_defaults()
    assert merged["overlay"]["style"]["en_size"] == defaults["overlay"]["style"]["en_size"]
    assert merged["overlay"]["style"]["hold_ms"] == defaults["overlay"]["style"]["hold_ms"]
    # Unrelated top-level blocks survive.
    assert merged["audio"] == defaults["audio"]


def test_load_full_config_quant_tier_small_overrides_default(appdata):
    """`translation.quant_tier: small` merges in without warnings."""
    paths.user_overrides_path().write_text(
        "translation:\n  quant_tier: small\n",
        encoding="utf-8",
    )
    merged = load_full_config()
    assert merged["translation"]["quant_tier"] == "small"
    # Sibling translation keys still come from defaults.
    defaults = _shipped_defaults()
    assert merged["translation"]["endpoint"] == defaults["translation"]["endpoint"]
    assert merged["translation"]["max_tokens"] == defaults["translation"]["max_tokens"]


def test_load_full_config_gazetteer_enabled_overrides_default(appdata):
    """`gazetteer.enabled: false` merges in via user_overrides.yaml
    without being stripped as an unknown key by the validator.

    Template: test_load_full_config_quant_tier_small_overrides_default.
    Seeded defect: remove the `gazetteer:` block (or just the `enabled`
    key) from shipped default_config.yaml -> _walk_unknown_keys flags
    the override as unknown-to-defaults, _strip_keys drops it from the
    overrides tree, and the merged dict no longer carries the override
    (KeyError or stale-default value) -- red.
    """
    paths.user_overrides_path().write_text(
        "gazetteer:\n  enabled: false\n",
        encoding="utf-8",
    )
    merged = load_full_config()
    assert merged["gazetteer"]["enabled"] is False
    # Unrelated top-level blocks still come from defaults (the override
    # didn't clobber siblings).
    defaults = _shipped_defaults()
    assert merged["hotkeys"] == defaults["hotkeys"]


def test_shipped_defaults_carry_show_series_hotkey_and_drop_gazetteer_hotkey(
    appdata,
):
    """The palette hotkey moved from the orphan `gazetteer.hotkey` (no
    consumer) to `hotkeys.show_series` (consumed by HotkeyManager.register).
    The shipped defaults MUST carry the new key -- else the config validator
    strips any user override of it (romanization.glossaries_dir reasoning) --
    and MUST NOT carry the removed one.

    Seeded defect: re-add `hotkey:` under gazetteer in default_config.yaml ->
    the absence assert goes red. Drop `show_series:` from hotkeys -> the
    presence assert goes red.
    """
    defaults = _shipped_defaults()
    assert defaults["hotkeys"].get("show_series") == "Ctrl+Alt+S", (
        f"shipped hotkeys.show_series missing/changed; got "
        f"{defaults['hotkeys'].get('show_series')!r}. A user override of it "
        f"would be stripped by the config validator."
    )
    assert "hotkey" not in defaults.get("gazetteer", {}), (
        "gazetteer.hotkey must be removed from shipped defaults (orphan key, "
        "no consumer); the palette hotkey lives in hotkeys.show_series now"
    )


def test_shipped_defaults_keep_series_context_hint_off(appdata):
    """translation.series_context_hint ships FALSE: the series-hint lever
    stays off until the blind judged A/B over the bench corpus wins (the
    feature's central gating contract). The key must also be PRESENT --
    with it absent from the shipped defaults, the validator would strip
    any user override of it (hotkeys.show_series reasoning above).

    Seeded defect: flip default_config.yaml's `series_context_hint:` to
    true (one line) -> red. Delete the key -> .get returns None -> red.
    """
    defaults = _shipped_defaults()
    assert defaults["translation"].get("series_context_hint") is False, (
        f"shipped translation.series_context_hint must be False until the "
        f"blind judged A/B wins; got "
        f"{defaults['translation'].get('series_context_hint')!r}"
    )


def test_load_full_config_unknown_key_warns_and_drops(appdata, caplog):
    paths.user_overrides_path().write_text(
        "overlay:\n  legacy_mode: true\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("overlay.legacy_mode" in m for m in msgs), msgs
    # Unknown key doesn't leak into merged dict.
    assert "legacy_mode" not in merged["overlay"]


def test_load_full_config_fails_fast_when_bundled_defaults_missing(
    tmp_path, monkeypatch,
):
    """Fails red if: a missing bundled default_config.yaml silently
    degrades to {}. With defaults={}, every user override is
    flagged unknown and stripped, load_full_config returns {}, and the
    pipeline dies far from the cause with KeyError('audio') behind a
    wall of misleading 'unknown config key' warnings — instead of one
    error naming the real problem (a yamabiko.spec datas regression
    where a bundled data file is absent).

    One-line seeded defect: delete the empty-defaults guard in
    load_full_config — the call returns {} instead of raising, and
    pytest.raises goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    missing = tmp_path / "bundle" / "default_config.yaml"
    monkeypatch.setattr(paths, "shipped_defaults_path", lambda: missing)
    # A user override that would otherwise be stripped with a bogus
    # 'unknown key' warning — the failure must preempt that noise.
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "translation:\n  quant_tier: small\n", encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="bundled defaults") as ei:
        load_full_config()
    assert str(missing) in str(ei.value), (
        "the error must name the resolved defaults path so a broken "
        "install is triageable from the log alone"
    )


def test_load_full_config_fails_fast_when_bundled_defaults_empty(
    tmp_path, monkeypatch,
):
    """Empty-file variant: a zero-byte default_config.yaml
    parses to {} in _read_yaml and must fail fast identically to the
    missing-file case (same guard, same diagnostic).
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    empty = tmp_path / "bundle" / "default_config.yaml"
    empty.parent.mkdir(parents=True)
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "shipped_defaults_path", lambda: empty)
    with pytest.raises(RuntimeError, match="bundled defaults"):
        load_full_config()


def test_null_override_on_non_nullable_string_key_restored_to_default(
    appdata, caplog,
):
    """A null override on translation.endpoint (a non-null string default)
    is stripped + the default restored, not passed through as None.
    Downstream ``TranslationConfig(endpoint=None)`` would break every
    llama request.

    Supersedes the old "explicit null replaces" contract: null is honored
    ONLY where the shipped default is itself null (overlay.geometry) --
    see test_null_override_on_nullable_default_key_is_kept.

    Seeded defect: drop the _walk_null_violations strip in
    _validate_and_merge -> merged['translation']['endpoint'] is None,
    not the default endpoint -> red.
    """
    paths.user_overrides_path().write_text(
        "translation:\n  endpoint: null\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()
    defaults = _shipped_defaults()
    assert (
        merged["translation"]["endpoint"]
        == defaults["translation"]["endpoint"]
    )
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("translation.endpoint" in m for m in msgs), msgs


def test_load_full_config_migration_only_legacy_present(appdata, caplog):
    # Seed legacy config.yaml with a modified jp_size + everything else default.
    defaults = _shipped_defaults()
    legacy = yaml.safe_load(yaml.safe_dump(defaults))  # deep copy via yaml
    legacy["overlay"]["style"]["jp_size"] = 22
    paths.config_path().write_text(yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="yamabiko.config"):
        merged = load_full_config()

    # Legacy renamed to backup; overrides now has the minimal diff.
    assert not paths.config_path().exists()
    assert paths.legacy_config_backup_path().exists()
    overrides = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert overrides == {"overlay": {"style": {"jp_size": 22}}}
    # Merged value reflects the migrated override.
    assert merged["overlay"]["style"]["jp_size"] == 22

    # Second load is a no-op (no re-migration).
    before_backup_mtime = paths.legacy_config_backup_path().stat().st_mtime
    before_overrides = paths.user_overrides_path().read_text(encoding="utf-8")
    merged2 = load_full_config()
    assert merged2["overlay"]["style"]["jp_size"] == 22
    assert paths.user_overrides_path().read_text(encoding="utf-8") == before_overrides
    assert paths.legacy_config_backup_path().stat().st_mtime == before_backup_mtime


def test_load_full_config_both_files_prefers_overrides(appdata, caplog):
    # Legacy sets jp_size=22, overrides sets jp_size=30. Overrides must win.
    defaults = _shipped_defaults()
    legacy = yaml.safe_load(yaml.safe_dump(defaults))
    legacy["overlay"]["style"]["jp_size"] = 22
    paths.config_path().write_text(yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8")
    paths.user_overrides_path().write_text(
        "overlay:\n  style:\n    jp_size: 30\n",
        encoding="utf-8",
    )
    legacy_before = paths.config_path().read_text(encoding="utf-8")
    overrides_before = paths.user_overrides_path().read_text(encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()

    assert merged["overlay"]["style"]["jp_size"] == 30
    # Neither file was touched.
    assert paths.config_path().read_text(encoding="utf-8") == legacy_before
    assert paths.user_overrides_path().read_text(encoding="utf-8") == overrides_before
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("both" in m and "user_overrides.yaml" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# F_F1 regression: wrong-typed overrides are dropped with a warning,
# so a YAML typo in user_overrides.yaml produces a named warning at
# load-time instead of a cryptic TypeError deep in runtime.
# ---------------------------------------------------------------------------


def test_wrong_typed_override_string_instead_of_float(tmp_path, monkeypatch, caplog):
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        'vad:\n  threshold: "not a number"\n'
        'asr:\n  faster_whisper:\n    no_speech_only_threshold: "0.9"\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=config_mod.__name__):
        merged = config_mod.load_full_config()
    # Override dropped; default kept.
    assert merged["vad"]["threshold"] == 0.3
    assert merged["asr"]["faster_whisper"]["no_speech_only_threshold"] == 0.95
    assert any(
        "vad.threshold" in rec.message and "float" in rec.message
        for rec in caplog.records
    ), (
        f"no type-mismatch warning logged; saw: "
        f"{[r.message for r in caplog.records]}"
    )


def test_bool_vs_int_mismatch_flagged(tmp_path, monkeypatch, caplog):
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "vad:\n  min_silence_ms: true\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=config_mod.__name__):
        merged = config_mod.load_full_config()
    defaults = _shipped_defaults()
    assert merged["vad"]["min_silence_ms"] == defaults["vad"]["min_silence_ms"]
    assert any(
        "min_silence_ms" in rec.message and "int" in rec.message
        for rec in caplog.records
    )


def test_int_where_float_expected_is_accepted(tmp_path, monkeypatch):
    """YAML lazy numbers: `threshold: 1` for a float default survives."""
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "vad:\n  threshold: 1\n", encoding="utf-8",
    )
    merged = config_mod.load_full_config()
    assert merged["vad"]["threshold"] == 1


def test_null_override_on_non_nullable_key_restored_to_default_with_warning(
    tmp_path, monkeypatch, caplog,
):
    """A leaf override of null on a key whose shipped default is NOT null
    is stripped and the default restored, with a warning naming the key.
    audio.device ships "default" (a non-null sentinel); a bare ``device:``
    in user_overrides parses as None, which the type walker treats as
    "explicit unset" and lets through -- then AudioConfig(device=None)
    carries a None device into the capture layer.

    Seeded defect: drop the _walk_null_violations strip in
    _validate_and_merge -> merged['audio']['device'] is None, not
    "default" -> red.
    """
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "audio:\n  device: null\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = config_mod.load_full_config()
    defaults = _shipped_defaults()
    assert merged["audio"]["device"] == defaults["audio"]["device"]
    assert merged["audio"]["device"] == "default"
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("audio.device" in m and "null" in m.lower() for m in msgs), msgs


def test_null_override_on_nullable_default_key_is_kept(
    tmp_path, monkeypatch, caplog,
):
    """A null override on a key whose SHIPPED default is itself null
    (overlay.geometry) is kept and does NOT warn -- it is
    nullable-by-default. Without this half of the contract the walker
    would strip a legitimate null and break geometry's
    null-means-recompute-placement semantics.

    The value is None whether kept or (wrongly) stripped-then-restored
    (the default is also None), so the distinguishing observable is the
    warning: an over-eager walker emits one, the correct walker does not.

    Seeded defect: drop the ``dv is not None`` guard in
    _walk_null_violations -> overlay.geometry null is flagged as a
    violation and a warning fires -> red.
    """
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "overlay:\n  geometry: null\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = config_mod.load_full_config()
    assert merged["overlay"]["geometry"] is None
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert not any("geometry" in m for m in msgs), (
        f"a null override on the nullable-by-default overlay.geometry must "
        f"not warn; got {msgs!r}"
    )


def test_null_override_on_websocket_port_restored_to_default_with_warning(
    tmp_path, monkeypatch, caplog,
):
    """``overlay.websocket_port: null`` is stripped + the 8765 default
    restored. Consumers do int(cfg['overlay']['websocket_port']);
    int(None) is a TypeError that crashes the uvicorn bind on launch.

    Seeded defect: drop the _walk_null_violations strip -> websocket_port
    is None -> red.
    """
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "overlay:\n  websocket_port: null\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = config_mod.load_full_config()
    assert merged["overlay"]["websocket_port"] == 8765
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("websocket_port" in m for m in msgs), msgs


def _write_explicit_tree(tmp_path, tree: dict) -> Path:
    path = tmp_path / "numeric-config.yaml"
    path.write_text(
        yaml.safe_dump(tree, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_explicit_numeric(tmp_path, dotted: str, value: object) -> Path:
    root: dict = {}
    cursor = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return _write_explicit_tree(tmp_path, root)


def test_numeric_rule_table_is_complete_and_exact():
    assert _NUMERIC_RULES == EXPECTED_NUMERIC_RULES


@pytest.mark.parametrize(
    ("dotted", "invalid_values"),
    INVALID_BOUNDARY_CASES.items(),
)
def test_every_numeric_rule_rejects_its_invalid_boundaries(
    dotted, invalid_values, tmp_path,
):
    for value in invalid_values:
        path = _write_explicit_numeric(tmp_path, dotted, value)
        with pytest.raises(ConfigValidationError) as exc:
            load_full_config(explicit_path=path)
        assert dotted in str(exc.value)


@pytest.mark.parametrize("dotted", EXPECTED_NUMERIC_RULES)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_every_numeric_rule_rejects_non_finite_values(
    dotted, value, tmp_path,
):
    path = _write_explicit_numeric(tmp_path, dotted, value)
    with pytest.raises(ConfigValidationError) as exc:
        load_full_config(explicit_path=path)
    assert dotted in str(exc.value)


@pytest.mark.parametrize(
    ("tree", "expected_path"),
    [
        (
            {"vad": {"min_chunk_ms": 9000, "max_chunk_ms": 8000}},
            "vad.min_chunk_ms",
        ),
        (
            {"vad": {"speech_pad_ms": 500, "attach_max_ms": 200}},
            "vad.attach_max_ms",
        ),
        (
            {
                "translation": {
                    "startup_readiness_timeout_s": 2.0,
                    "startup_readiness_poll_s": 3.0,
                }
            },
            "translation.startup_readiness_poll_s",
        ),
    ],
)
def test_cross_field_numeric_invariants_are_rejected(
    tree, expected_path, tmp_path,
):
    path = _write_explicit_tree(tmp_path, tree)
    with pytest.raises(ConfigValidationError) as exc:
        load_full_config(explicit_path=path)
    assert expected_path in str(exc.value)


def test_shipped_defaults_satisfy_every_numeric_contract():
    validate_config_semantics(_shipped_defaults())


def test_shipped_audio_config_has_no_ignored_channel_override():
    assert "channels" not in _shipped_defaults()["audio"]


def test_logging_level_lowercase_normalized_to_uppercase_at_load(
    tmp_path, monkeypatch,
):
    """A lowercase logging.level in user_overrides is normalized to the
    stdlib UPPERCASE name in the merged config, so every consumer
    (pipeline setLevel, watchdog setLevel, overlay --log-level) reads a
    value logger.setLevel accepts -- setLevel('debug') raises ValueError.

    Seeded defect: drop the _normalize_merged_log_level call in
    load_full_config -> merged['logging']['level'] stays 'debug' -> red.
    """
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "logging:\n  level: debug\n", encoding="utf-8",
    )
    merged = config_mod.load_full_config()
    assert merged["logging"]["level"] == "DEBUG"


def test_logging_level_unknown_normalizes_to_info_and_stashes_warning_at_load(
    tmp_path, monkeypatch, caplog,
):
    """An unrecognized logging.level ('verbose') is normalized to INFO
    in the merged dict, and the junk warning is STASHED for the caller
    (config.consume_level_warning) instead of being emitted inside
    load_full_config. The watchdog loads config BEFORE setup_main_log
    attaches its file handler, so an in-load emission dies on the
    frozen bundle's discarded stderr with zero trace; each role must
    emit the stashed text itself, AFTER its handler is attached.

    Seeded defects (each one line):
      - drop the _normalize_merged_log_level call -> merged level stays
        'verbose' -> the INFO assert red.
      - re-add ``_log.warning(...)`` inside _normalize_merged_log_level
        -> the no-in-load-emission assert red.
      - drop the stash assignment in load_full_config -> the first
        consume returns None -> the warning-text assert red.
      - make consume_level_warning non-clearing -> the second-consume
        assert red.
    """
    from yamabiko import config as config_mod
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "logging:\n  level: verbose\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = config_mod.load_full_config()
    assert merged["logging"]["level"] == "INFO"
    in_load = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "verbose" in r.getMessage()
    ]
    assert not in_load, (
        f"the junk-level warning was emitted INSIDE load_full_config "
        f"(pre-handler-attach on the watchdog path: the only evidence "
        f"dies on the frozen bundle's discarded stderr): {in_load}"
    )
    warning = config_mod.consume_level_warning()
    assert warning is not None and "verbose" in warning, (
        f"load_full_config did not stash the junk-level warning for the "
        f"caller to emit post-handler-attach; got {warning!r}"
    )
    assert config_mod.consume_level_warning() is None, (
        "consume_level_warning did not clear the stash; a stale warning "
        "would be re-emitted by the next role's load"
    )


def test_null_section_override_keeps_default_section_with_warning(
    appdata, caplog,
):
    """Fails red if: a section key set to null replaces the entire
    defaults section with None. ``overlay:`` with all children
    commented out — a completely natural hand-edit — parses as None;
    _deep_merge then returns None for the whole section and the
    pipeline crashes at startup far from the cause
    (cfg["overlay"].get(...) -> AttributeError on None,
    cfg["audio"]["device"] -> TypeError).

    The leaf-level null semantics ("explicit unset", covered by
    test_null_override_is_accepted) must NOT change — only
    section-level non-dicts are mismatches.

    One-line seeded defect: in _walk_type_mismatches, drop the
    non-dict check before the section recursion — the recursion hits
    the isinstance guard, returns [] instead of a mismatch, the
    section merges to None, and the first assert goes red.
    """
    paths.user_overrides_path().write_text(
        # `overlay:` with every child commented out — YAML parses the
        # section value as None.
        "overlay:\n"
        "#  style:\n"
        "#    jp_size: 18\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()

    defaults = _shipped_defaults()
    assert merged["overlay"] == defaults["overlay"], (
        f"overlay section was not restored to defaults; got "
        f"{merged['overlay']!r}. A None/scalar section override must "
        f"be dropped, not deep-merged."
    )
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("overlay" in m and "mapping" in m for m in msgs), (
        f"expected a type-mismatch warning naming 'overlay' and the "
        f"expected 'mapping' type; got {msgs!r}"
    )


def test_scalar_section_override_keeps_default_section_with_warning(
    appdata, caplog,
):
    """Scalar variant: ``audio: foo`` (typo'd section) must be
    dropped with a mapping-mismatch warning, keeping the default audio
    section, instead of replacing the dict and crashing at
    cfg["audio"]["device"] with TypeError far from the cause.
    """
    paths.user_overrides_path().write_text(
        "audio: foo\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()

    defaults = _shipped_defaults()
    assert merged["audio"] == defaults["audio"], (
        f"audio section was not restored to defaults; got "
        f"{merged['audio']!r}"
    )
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("audio" in m and "mapping" in m for m in msgs), (
        f"expected a type-mismatch warning naming 'audio' and "
        f"'mapping'; got {msgs!r}"
    )


# ---------------------------------------------------------------------------
# Explicit --config escape hatch: load_full_config(explicit_path=...)
#
# The explicit file is layered over shipped defaults (so missing sections
# never KeyError downstream) with the SAME unknown-key / type validation
# as normal loads, but user_overrides.yaml is DELIBERATELY skipped -- the
# point of --config is an isolated, reproducible config that does not pick
# up ambient user state.
# ---------------------------------------------------------------------------


def test_load_full_config_explicit_path_merges_missing_section_from_defaults(
    appdata, tmp_path,
):
    """An explicit --config file that omits whole sections is layered over
    the shipped defaults, so downstream indexing never KeyErrors.

    Seeded defect: have the explicit branch read the file verbatim (skip
    the defaults merge) -> merged['audio'] is absent -> KeyError on the
    audio assertion -> red.
    """
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "overlay:\n  style:\n    jp_size: 21\n", encoding="utf-8",
    )
    merged = load_full_config(explicit_path=explicit)
    defaults = _shipped_defaults()
    # The section present in the explicit file is applied...
    assert merged["overlay"]["style"]["jp_size"] == 21
    # ...and whole sections ABSENT from the explicit file still resolve
    # from defaults (no verbatim-load KeyError far from the cause).
    assert merged["audio"] == defaults["audio"]
    assert (
        merged["translation"]["endpoint"]
        == defaults["translation"]["endpoint"]
    )


def test_load_full_config_explicit_path_strips_and_warns_unknown_key(
    appdata, tmp_path, caplog,
):
    """An unknown key in the explicit file is stripped + warned, exactly
    like an unknown key in user_overrides.yaml.

    Seeded defect: skip the unknown-key walk for the explicit layer ->
    'legacy_mode' survives into the merged dict -> the absence assert
    goes red.
    """
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "overlay:\n  legacy_mode: true\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config(explicit_path=explicit)
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("overlay.legacy_mode" in m for m in msgs), msgs
    assert "legacy_mode" not in merged["overlay"]


def test_load_full_config_explicit_path_drops_wrong_typed_value_with_warning(
    appdata, tmp_path, caplog,
):
    """A wrong-typed value in the explicit file is dropped with a warning
    and the shipped default kept -- the same type walker that guards
    user_overrides.yaml runs against the explicit layer.

    Seeded defect: skip the type-mismatch walk for the explicit layer ->
    the string "not a number" reaches merged['vad']['threshold'] instead
    of the 0.3 default -> red.
    """
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        'vad:\n  threshold: "not a number"\n', encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config(explicit_path=explicit)
    assert merged["vad"]["threshold"] == 0.3
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("vad.threshold" in m and "float" in m for m in msgs), msgs


def test_load_full_config_explicit_empty_file_returns_defaults_with_warning(
    appdata, tmp_path, caplog,
):
    """An empty explicit file yields pure defaults plus a WARNING, never
    an AttributeError from `None.get(...)`.

    Seeded defect (today's behaviour): read the file verbatim ->
    yaml.safe_load("") is None -> downstream `.get` AttributeError. The
    fix maps empty/None to {} + a warning; dropping the warning makes the
    warning assert red, dropping the {}-coercion makes the equality
    assert red.
    """
    explicit = tmp_path / "empty.yaml"
    explicit.write_text("", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config(explicit_path=explicit)
    assert merged == _shipped_defaults()
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("empty" in m.lower() for m in msgs), msgs


def test_load_full_config_explicit_path_ignores_user_overrides(
    appdata, tmp_path,
):
    """The --config escape hatch is isolated from ambient user state:
    user_overrides.yaml is NOT layered under an explicit file.

    Seeded defect: have the explicit branch also read + merge
    user_overrides.yaml -> genre becomes 'anime' (the override) instead
    of the shipped default -> red.
    """
    # A user override that WOULD change genre if it were layered in.
    paths.user_overrides_path().write_text("genre: anime\n", encoding="utf-8")
    # The explicit file does not touch genre at all.
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "overlay:\n  style:\n    jp_size: 19\n", encoding="utf-8",
    )
    merged = load_full_config(explicit_path=explicit)
    defaults = _shipped_defaults()
    assert merged["genre"] == defaults["genre"], (
        "user_overrides.yaml leaked into the explicit --config run; the "
        "escape hatch must be isolated from ambient user state"
    )
    # Sanity: the explicit file's own value DID apply.
    assert merged["overlay"]["style"]["jp_size"] == 19


def test_load_full_config_explicit_missing_path_raises_runtime_error(
    appdata, tmp_path,
):
    """A nonexistent --config path raises RuntimeError -- the same
    fatal-config channel build_pipeline's handler catches to show a
    tailored dialog + exit 43, instead of an uncaught FileNotFoundError
    the watchdog reads as a generic crash and restart-loops.

    Seeded defect: swallow the missing-file case (return {} / defaults) ->
    no RuntimeError -> pytest.raises goes red.
    """
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(RuntimeError) as ei:
        load_full_config(explicit_path=missing)
    assert str(missing) in str(ei.value), (
        "the error must name the missing path so the user can fix it"
    )


def test_load_full_config_explicit_parse_error_raises_runtime_error(
    appdata, tmp_path,
):
    """A malformed explicit file raises RuntimeError (the fatal-config
    channel), not a raw yaml.YAMLError the startup handler misses.

    Seeded defect: let the YAMLError propagate unwrapped -> pytest.raises
    (RuntimeError) does not match -> red.
    """
    explicit = tmp_path / "broken.yaml"
    explicit.write_text("overlay: {\n  style: [unterminated", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_full_config(explicit_path=explicit)


def test_huge_int_style_override_keeps_default_without_exception(
    tmp_path, monkeypatch, caplog,
):
    """Style keys are deliberately excluded from _NUMERIC_RULES because
    _clamp_style owns their clamp-to-range behavior. A 309-digit
    ``overlay.style.bg_opacity`` therefore reaches _clamp_style, where an unguarded
    ``math.isfinite(huge_int)`` raised OverflowError out of
    load_overlay_config (build_pipeline catches RuntimeError only, the
    overlay catches nothing): BOTH children crashed at boot into a
    watchdog restart loop ending in the generic give-up dialog.

    Seeded defect: drop the OverflowError handling from _clamp_style's
    guard -> this test errors with OverflowError -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    huge = 10 ** 309
    paths.user_overrides_path().write_text(
        f"overlay:\n  style:\n    bg_opacity: {huge}\n"
        f"  position:\n    v_offset_pct: {huge}\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        cfg = config_mod.load_overlay_config()

    # Both keys degrade to their shipped defaults instead of crashing.
    assert cfg.style["bg_opacity"] == 237
    assert cfg.position["v_offset_pct"] == 88.0
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("bg_opacity" in m for m in msgs), msgs


# --- stale translation.model override (Qwen3.5 rename upgrade trap) ---


def test_stale_translation_model_override_reset_to_default_with_warning(
    tmp_path, monkeypatch, caplog,
):
    """A v0.1.0-era ``translation.model: shisa-v2.1-qwen3-8b`` override
    (hand-pinned, or written by the legacy-config migration's diff) is a
    known key of valid type, so the walkers apply it verbatim -- but the
    bundled llama-server serves ONLY the current alias, so every launch
    404s the startup sanity translate into the generic exit-43 dialog
    with nothing naming the stale key. When the endpoint is still the
    bundled default, the loader must reset the model to the shipped
    alias and say why.

    Seeded defect: delete the stale-model rescue call from
    load_full_config's layered branch -> merged carries the shisa alias
    -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "translation:\n  model: shisa-v2.1-qwen3-8b\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()

    defaults = config_mod._read_yaml(paths.shipped_defaults_path())
    assert merged["translation"]["model"] == defaults["translation"]["model"]
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("translation.model" in m for m in msgs), msgs


def test_custom_model_with_custom_endpoint_is_kept(tmp_path, monkeypatch):
    """An external-server user (custom endpoint) may legitimately run a
    different model name; the stale-model reset must NOT touch them.

    Seeded defect: drop the endpoint-is-default condition from the reset
    -> the custom model is clobbered -> red.
    """
    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "translation:\n"
        "  model: my-own-model\n"
        "  endpoint: http://192.168.1.50:9000/v1/chat/completions\n",
        encoding="utf-8",
    )
    merged = config_mod.load_full_config()
    assert merged["translation"]["model"] == "my-own-model"


def test_stale_model_rescue_prunes_the_override_on_disk_and_second_load_is_quiet(
    tmp_path, monkeypatch, caplog,
):
    """The rescue must self-heal: reset the model in memory AND prune the
    stale translation.model from user_overrides.yaml through the locked
    writer machinery, so the WARNING fires once instead of on every
    launch in all three processes forever. The model was the only
    override here, so the sparse file is deleted entirely.

    Seeded defect: drop the on-disk prune after the rescue in
    load_full_config's layered branch -> the key survives on disk and
    the second load warns again -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "translation:\n  model: shisa-v2.1-qwen3-8b\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()

    defaults = config_mod._read_yaml(paths.shipped_defaults_path())
    assert merged["translation"]["model"] == defaults["translation"]["model"]
    assert not paths.user_overrides_path().exists(), (
        "the rescued stale translation.model was not pruned from disk "
        "(model was the only override, so the sparse file must be gone)"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        merged2 = config_mod.load_full_config()
    assert merged2["translation"]["model"] == defaults["translation"]["model"]
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("translation.model" in m for m in msgs), (
        f"second load re-fired the stale-model warning; the rescue did "
        f"not self-heal on disk: {msgs!r}"
    )


def test_stale_model_rescue_prune_preserves_sibling_overrides_on_disk(
    tmp_path, monkeypatch, caplog,
):
    """The prune must remove ONLY translation.model: sibling overrides in
    the same file (another translation key, another section) survive
    verbatim on disk and in the merged result.

    Seeded defect: flush an empty tree (or the whole merged dict)
    instead of pruning the single key -> the sibling overrides vanish
    from disk -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "translation:\n"
        "  model: shisa-v2.1-qwen3-8b\n"
        "  context_lines: 7\n"
        "overlay:\n"
        "  style:\n"
        "    jp_size: 18\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()

    defaults = config_mod._read_yaml(paths.shipped_defaults_path())
    assert merged["translation"]["model"] == defaults["translation"]["model"]
    assert merged["translation"]["context_lines"] == 7
    assert merged["overlay"]["style"]["jp_size"] == 18
    on_disk = yaml.safe_load(
        paths.user_overrides_path().read_text(encoding="utf-8")
    )
    assert on_disk == {
        "translation": {"context_lines": 7},
        "overlay": {"style": {"jp_size": 18}},
    }, f"prune touched more than translation.model: {on_disk!r}"


def test_migration_drops_stale_model_from_diff_when_endpoint_is_default(
    appdata, caplog,
):
    """_migrate_legacy_config used to write the stale translation.model
    into the fresh overrides file itself (a legacy full config always
    differs from the renamed default), re-arming the rescue on the very
    first post-upgrade launch. When the legacy endpoint is still the
    bundled default, the model pin can only be the rename trap: it must
    be dropped from the migrated diff, so the merged config carries the
    shipped alias and no rescue warning ever fires for migrated installs.

    Seeded defect: remove the migration-time model drop -> the migrated
    overrides file carries translation.model -> the on-disk assert goes
    red.
    """
    defaults = _shipped_defaults()
    legacy = yaml.safe_load(yaml.safe_dump(defaults))
    legacy["translation"]["model"] = "shisa-v2.1-qwen3-8b"
    legacy["overlay"]["style"]["jp_size"] = 22
    paths.config_path().write_text(
        yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.config"):
        merged = load_full_config()

    overrides = yaml.safe_load(
        paths.user_overrides_path().read_text(encoding="utf-8")
    )
    assert overrides == {"overlay": {"style": {"jp_size": 22}}}, (
        f"migration carried the stale translation.model into the fresh "
        f"overrides file: {overrides!r}"
    )
    assert merged["translation"]["model"] == defaults["translation"]["model"]
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("translation.model" in m for m in msgs), (
        f"the rescue warning fired although migration should have "
        f"dropped the stale key before it reached disk: {msgs!r}"
    )


def test_migration_keeps_model_override_when_endpoint_is_custom(appdata):
    """External-server counterpart: a legacy config pointing at a custom
    endpoint may legitimately pin any model name; migration must carry
    BOTH keys into the overrides file untouched.

    Seeded defect: drop the endpoint gate from the migration-time model
    drop (drop the key unconditionally) -> the custom model vanishes ->
    red.
    """
    defaults = _shipped_defaults()
    legacy = yaml.safe_load(yaml.safe_dump(defaults))
    legacy["translation"]["model"] = "my-own-model"
    legacy["translation"]["endpoint"] = (
        "http://192.168.1.50:9000/v1/chat/completions"
    )
    paths.config_path().write_text(
        yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8",
    )
    merged = load_full_config()

    overrides = yaml.safe_load(
        paths.user_overrides_path().read_text(encoding="utf-8")
    )
    assert overrides == {
        "translation": {
            "model": "my-own-model",
            "endpoint": "http://192.168.1.50:9000/v1/chat/completions",
        }
    }, f"migration mangled a legitimate external-server pin: {overrides!r}"
    assert merged["translation"]["model"] == "my-own-model"


@pytest.mark.parametrize(
    "endpoint_variant",
    [
        "http://localhost:8090/v1/chat/completions/",
        "http://127.0.0.1:8090/v1/chat/completions",
        "HTTP://Localhost:8090/v1/chat/completions",
    ],
    ids=["trailing-slash", "loopback-ip", "scheme-host-case"],
)
def test_stale_model_rescue_fires_on_cosmetic_endpoint_variants(
    endpoint_variant, tmp_path, monkeypatch, caplog,
):
    """A cosmetic re-spelling of the bundled default endpoint (trailing
    slash, scheme/host case, localhost vs 127.0.0.1) must not dodge the
    rescue: the endpoint gate normalizes before comparing. The endpoint
    override itself is left alone on disk; only the model is pruned.

    Seeded defect: revert the endpoint gate to exact string equality ->
    no variant rescues -> the model assert goes red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    defaults = config_mod._read_yaml(paths.shipped_defaults_path())
    # The variants above are hand-spun from this exact default; pin it so
    # a future endpoint change fails here with a clear message instead of
    # silently testing unrelated strings.
    assert defaults["translation"]["endpoint"] == (
        "http://localhost:8090/v1/chat/completions"
    ), "default endpoint moved; update this test's cosmetic variants"
    paths.user_overrides_path().write_text(
        "translation:\n"
        "  model: shisa-v2.1-qwen3-8b\n"
        f"  endpoint: {endpoint_variant}\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()

    assert merged["translation"]["model"] == defaults["translation"]["model"]
    assert merged["translation"]["endpoint"] == endpoint_variant
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("translation.model" in m for m in msgs), msgs
    on_disk = yaml.safe_load(
        paths.user_overrides_path().read_text(encoding="utf-8")
    )
    assert on_disk == {"translation": {"endpoint": endpoint_variant}}, (
        f"prune must drop the model and keep the endpoint override: "
        f"{on_disk!r}"
    )


@pytest.mark.parametrize(
    "endpoint_other",
    [
        "http://localhost:9000/v1/chat/completions",
        "http://localhost:8090/v2/chat/completions",
    ],
    ids=["different-port", "different-path"],
)
def test_custom_model_kept_when_endpoint_differs_beyond_cosmetics(
    endpoint_other, tmp_path, monkeypatch, caplog,
):
    """The normalization stays conservative: a different port or path
    names a genuinely different server, so the model override passes
    through untouched (no rescue, no warning, no prune).

    Seeded defect: normalize the port (or path) away in the endpoint
    equivalence check -> the custom model is clobbered and the file
    pruned -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "translation:\n"
        "  model: my-own-model\n"
        f"  endpoint: {endpoint_other}\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()

    assert merged["translation"]["model"] == "my-own-model"
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("translation.model" in m for m in msgs), msgs
    on_disk = yaml.safe_load(
        paths.user_overrides_path().read_text(encoding="utf-8")
    )
    assert on_disk == {
        "translation": {"model": "my-own-model", "endpoint": endpoint_other}
    }, f"a genuinely different endpoint must not trigger the prune: {on_disk!r}"


# --- reader hardening: corrupt defaults + unreadable overrides ---


def test_corrupt_shipped_defaults_raise_the_tailored_broken_install_error(
    tmp_path, monkeypatch,
):
    """A truncated/mangled (non-empty) default_config.yaml raised a raw
    yaml.YAMLError, missing the RuntimeError channel built for the
    missing/empty case -- so a damaged install crash-looped through the
    watchdog into the GENERIC dialog instead of the broken-install one.

    Seeded defect: remove the YAMLError->RuntimeError wrap around the
    defaults read -> yaml.YAMLError escapes -> red (wrong exception
    type).
    """
    import pytest as _pytest

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    corrupt = tmp_path / "default_config.yaml"
    corrupt.write_text("audio:\n  device: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(paths, "shipped_defaults_path", lambda: corrupt)
    with _pytest.raises(RuntimeError, match="Broken install"):
        config_mod.load_full_config()


def test_unreadable_overrides_degrade_to_defaults_with_warning(
    tmp_path, monkeypatch, caplog,
):
    """An OSError on the overrides read (AV/backup tool holding an
    exclusive open) escaped load_full_config entirely -- generic crash +
    restart loop instead of the degrade-to-defaults contract the
    parse-error branch already honors.

    Simulated portably by pointing the overrides path at a DIRECTORY
    (read_text raises IsADirectoryError, an OSError subclass).

    Seeded defect: narrow the except back to yaml.YAMLError only -> the
    OSError escapes -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    as_dir = tmp_path / "overrides-is-a-dir"
    as_dir.mkdir()
    monkeypatch.setattr(paths, "user_overrides_path", lambda: as_dir)
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()

    defaults = config_mod._read_yaml(paths.shipped_defaults_path())
    assert merged["translation"]["model"] == defaults["translation"]["model"]
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ignoring overrides" in m for m in msgs), msgs


def test_unreadable_legacy_config_skips_migration_with_warning_and_boots_on_defaults(
    tmp_path, monkeypatch, caplog,
):
    """An OSError on the legacy config.yaml read during the one-time
    migration (an AV/backup tool holding it exclusively open on the
    first post-upgrade launch) must degrade like a parse error: warn,
    leave the file in place, skip the migration, boot on defaults. The
    narrow yaml.YAMLError except let the OSError escape and crash-loop
    all three processes generically.

    Simulated portably by planting a DIRECTORY at the legacy path
    (exists() is True; read_text raises an OSError subclass).

    Seeded defect: narrow _migrate_legacy_config's except back to
    yaml.YAMLError only -> the OSError escapes load_full_config -> red.
    """
    import logging

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.config_path().mkdir()
    with caplog.at_level(logging.WARNING):
        merged = config_mod.load_full_config()  # must not raise

    assert merged == _shipped_defaults()
    # Migration skipped: legacy left in place, nothing written.
    assert paths.config_path().exists()
    assert not paths.legacy_config_backup_path().exists()
    assert not paths.user_overrides_path().exists()
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        paths.config_path().name in m and "migration" in m for m in msgs
    ), f"no skip-migration warning naming the legacy file: {msgs!r}"


def test_unreadable_shipped_defaults_raise_the_tailored_broken_install_error(
    tmp_path, monkeypatch,
):
    """An unreadable-but-present bundled default_config.yaml (ACL
    accident, AV exclusive open) must land in the SAME tailored
    broken-install RuntimeError as the corrupt/missing/empty cases -- a
    raw OSError crash-loops through the watchdog into the generic
    dialog instead.

    Simulated portably by pointing shipped_defaults_path at a DIRECTORY.

    Seeded defect: narrow the defaults-read except back to
    yaml.YAMLError only -> the OSError escapes -> red (wrong exception
    type).
    """
    import pytest as _pytest

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    as_dir = tmp_path / "defaults-is-a-dir"
    as_dir.mkdir()
    monkeypatch.setattr(paths, "shipped_defaults_path", lambda: as_dir)
    with _pytest.raises(RuntimeError, match="Broken install"):
        config_mod.load_full_config()


def test_corrupt_shipped_defaults_raise_broken_install_error_via_load_overlay_config(
    tmp_path, monkeypatch,
):
    """load_overlay_config(path=...) is the pipeline's --config
    overlay-style seeding path; its shipped-defaults read must route a
    corrupt bundled file into the same tailored broken-install
    RuntimeError as its siblings. build_pipeline catches RuntimeError
    only, so a raw yaml.YAMLError there is a generic crash + restart
    loop.

    Seeded defect: drop the RuntimeError wrap from load_overlay_config's
    defaults read -> raw yaml.YAMLError escapes -> red (wrong exception
    type).
    """
    import pytest as _pytest

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    corrupt = tmp_path / "default_config.yaml"
    corrupt.write_text("audio:\n  device: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(paths, "shipped_defaults_path", lambda: corrupt)
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("overlay:\n  theme: dark\n", encoding="utf-8")
    with _pytest.raises(RuntimeError, match="Broken install"):
        config_mod.load_overlay_config(explicit)
