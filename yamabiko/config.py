"""Layered configuration loader.

Two files drive Yamabiko's runtime config:

* ``default_config.yaml`` ships inside the package; it is the authoritative
  schema and the source of every default value. Never written to at
  runtime.
* ``%LOCALAPPDATA%/Yamabiko/user_overrides.yaml`` is sparse: it only
  contains keys the user has explicitly overridden. Absent on first
  launch.

At load time the two files are deep-merged (defaults + overrides) and
downstream code sees a single flat dict with the same shape as the old
``config.yaml``. See ``load_full_config`` for the merge semantics and
one-time migration of legacy ``config.yaml`` installations.
"""
from __future__ import annotations
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import yaml

from . import paths
from .overlay_app.style_defaults import (
    STYLE_DEFAULTS, HEX_COLOR_RE, COLOR_STYLE_KEYS, FONT_STYLE_KEYS,
    MAX_WIDTH_PCT_RANGE, STYLE_CHOICES,
)

_log = logging.getLogger(__name__)

_STYLE_RANGES: dict = {
    "jp_size": (8, 48),
    "en_size": (8, 72),
    "bg_opacity": (0, 255),
    "padding": (0, 40),
    # Canonical window from style_defaults, shared with position_config's
    # migration squeeze, the Settings "Max width" slider, and _cap_px --
    # never a re-typed literal that can drift when that window moves.
    "max_width_pct": MAX_WIDTH_PCT_RANGE,
    "max_lines": (1, 6),
    "hold_ms": (500, 15000),
    "fade_in_ms": (0, 1000),
    "fade_out_ms": (0, 2000),
}


@dataclass
class OverlayConfig:
    style: dict = field(default_factory=lambda: dict(STYLE_DEFAULTS))
    geometry: Optional[dict] = None
    theme: str = "system"
    position: dict = field(default_factory=dict)
    reduce_motion: bool = False


# ---------------------------------------------------------------------------
# Deep-merge helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Any, overlay: Any) -> Any:
    """Return ``base`` with ``overlay`` merged in (non-mutating at top level).

    * Dicts merge recursively.
    * Lists replace entirely (no append/union — predictable semantics).
    * Scalars in ``overlay`` replace ``base``.
    * Explicit ``None`` in ``overlay`` replaces the default with ``None``.
      Non-nullable ``None`` leaves are stripped upstream by
      :func:`_walk_null_violations` (inside ``_validate_and_merge``), so a
      ``None`` reaching here is either a nullable-by-default key
      (``overlay.geometry``) or a caller that bypassed validation.

    No type coercion. Downstream code validates types.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for k, v in overlay.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    # Any non-dict overlay (including None, lists, scalars) replaces base.
    return overlay


def _walk_unknown_keys(overrides: Any, defaults: Any, prefix: str = "") -> list[str]:
    """Return dotted key paths present in ``overrides`` but absent in ``defaults``.

    Only examines keys — not values. A user override whose *value* differs
    from the default's type is someone else's problem (downstream
    validation); this function refuses to guess intent.
    """
    unknown: list[str] = []
    if not isinstance(overrides, dict):
        return unknown
    if not isinstance(defaults, dict):
        # Overrides tree descends past a scalar/list in defaults; every
        # leaf below this point is "unknown" schema-wise.
        for k in overrides:
            dotted = f"{prefix}.{k}" if prefix else k
            unknown.append(dotted)
        return unknown
    for k, v in overrides.items():
        dotted = f"{prefix}.{k}" if prefix else k
        if k not in defaults:
            unknown.append(dotted)
            continue
        if isinstance(v, dict) and isinstance(defaults[k], dict):
            # Only recurse when the DEFAULT is also a mapping. A dict
            # override sitting on a non-dict default is not a set of
            # "unknown keys": overlay.geometry's shipped default is null
            # (geometry is user state, not a default), and recursing the
            # user's saved {x,y,width,height} against that null flagged
            # all four as unknown and stripped them on every load,
            # silently resetting the overlay position. Dict-over-scalar
            # type policing stays in _walk_type_mismatches (a dict
            # override on a str/int/list default is dropped there with a
            # mapping-mismatch warning).
            unknown.extend(_walk_unknown_keys(v, defaults[k], dotted))
    return unknown


def _deep_diff(user: Any, defaults: Any) -> Any:
    """Return the minimal subtree of ``user`` that differs from ``defaults``.

    Result is a dict containing only leaves whose value doesn't match the
    shipped default. If everything matches, returns an empty dict.
    Used by the one-time legacy-``config.yaml`` migration to produce a
    sparse overrides file.
    """
    if isinstance(user, dict) and isinstance(defaults, dict):
        out: dict = {}
        for k, v in user.items():
            if k not in defaults:
                # Key unknown to shipped defaults — preserve verbatim so
                # the user doesn't silently lose custom keys during
                # migration. The loader will warn about it on next load.
                out[k] = v
                continue
            sub = _deep_diff(v, defaults[k])
            # _deep_diff returns a sentinel via ``_NOOP`` when there's no
            # diff; otherwise the minimal subtree.
            if sub is _NOOP:
                continue
            out[k] = sub
        return out if out else _NOOP
    if user == defaults:
        return _NOOP
    return user


# Sentinel for _deep_diff "no difference". Can't use None because None is
# a legitimate user value that differs from a non-None default.
_NOOP = object()


def _read_yaml(path: Path) -> dict:
    """Read a YAML file into a dict. Empty/missing -> {}. Parse errors raise.

    Bytes that are not UTF-8 (a UTF-16 file from Windows PowerShell 5.1
    ``Out-File``, a CP932 file from Notepad's ANSI mode) raise
    ``yaml.YAMLError`` too, so every caller's corrupt-file handling covers
    them. A raw ``UnicodeDecodeError`` is a ValueError and escaped those
    handlers, crash-looping the pipeline instead of booting on defaults.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise yaml.YAMLError(f"{path}: not UTF-8 text ({e})") from e
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise yaml.YAMLError(f"{path}: expected top-level mapping, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_and_merge(defaults: dict, overlay: dict, source_name: str) -> dict:
    """Strip unknown keys + wrong-typed values from ``overlay`` (warning
    on each, naming ``source_name``), then deep-merge onto ``defaults``.

    Shared by the layered path (``source_name`` = user_overrides.yaml) and
    the explicit ``--config`` path (``source_name`` = the explicit file's
    basename) so both layers get IDENTICAL strip-and-warn semantics from a
    single implementation. The warning wording is deliberately unchanged
    from the pre-refactor inline version.
    """
    unknown = _walk_unknown_keys(overlay, defaults)
    for dotted in unknown:
        _log.warning(
            "unknown config key in %s: %s (ignoring)",
            source_name, dotted,
        )
    if unknown:
        overlay = _strip_keys(overlay, unknown)

    # Semantic numeric failures are fatal. Evaluate the sparse overlay
    # before type/null fallback stripping so a correctly typed but unsafe
    # value can never be silently replaced with a default.
    numeric_problems = _numeric_rule_problems(overlay)
    if numeric_problems:
        raise ConfigValidationError(numeric_problems)

    # Type-check override values against shipped defaults.
    # Without this, a YAML typo like `vad.threshold: "0.5"` (string
    # instead of float) would propagate to main.py arithmetic and
    # produce a cryptic TypeError far from the root cause. Log a
    # warning naming the file + key + expected type, and drop the
    # override so the default kicks in.
    bad_types = _walk_type_mismatches(overlay, defaults)
    for dotted, expected, got in bad_types:
        _log.warning(
            "config key %r in %s has type %s; expected %s. Using default.",
            dotted, source_name, got, expected,
        )
    if bad_types:
        overlay = _strip_keys(overlay, [p for p, _, _ in bad_types])

    # Nullability: a leaf override of None is valid ONLY where the shipped
    # default is itself None (nullable-by-default, e.g. overlay.geometry).
    # Everywhere else None slips past _walk_type_mismatches (which accepts
    # None as "explicit unset") and then breaks a consumer that coerces it
    # unconditionally -- int(websocket_port), float(grace_s). Strip + warn +
    # restore the default, same convention as a type mismatch.
    null_bad = _walk_null_violations(overlay, defaults)
    for dotted in null_bad:
        _log.warning(
            "config key %r in %s is null but this key has no nullable "
            "default; using default.",
            dotted, source_name,
        )
    if null_bad:
        overlay = _strip_keys(overlay, null_bad)

    merged = _deep_merge(defaults, overlay)
    validate_config_semantics(merged)
    return merged


def _endpoints_equivalent(a: Any, b: Any) -> bool:
    """True when two translation.endpoint values name the SAME server,
    for the stale-model rescue's gate.

    Deliberately conservative: only cosmetic differences are folded --
    trailing slash on the path, scheme/host case, and localhost vs
    127.0.0.1 for the loopback host specifically. A different port or
    path is a genuinely different endpoint and must NOT compare equal
    (an external-server user's model override has to pass through
    untouched). Non-string or unparseable values fall back to the exact
    equality already checked first.
    """
    if a == b:
        return True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False

    def canon(url: str):
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host == "localhost":
            host = "127.0.0.1"
        return (
            parts.scheme.lower(), host, parts.port,
            parts.path.rstrip("/"), parts.query, parts.fragment,
        )

    try:
        return canon(a) == canon(b)
    except ValueError:
        # urlsplit / port parsing rejected one side; exact equality above
        # already said no.
        return False


def _rescue_stale_translation_model(
    merged: dict, defaults: dict, source_name: str,
) -> bool:
    """Stale translation.model rescue (Qwen3.5 rename, 0.2.0). The bundled
    llama-server registers ONLY the shipped alias, so a leftover model
    override from an earlier release 404s the startup sanity translate on
    EVERY launch with nothing naming the stale key. Resets the merged
    model in place and returns True when the rescue fired, so the layered
    loader can also prune the stale key from the overrides file on disk
    (the in-memory reset alone re-fires this warning in all three
    processes on every launch). Only fires while the endpoint is still
    the bundled default (modulo cosmetic spelling -- see
    :func:`_endpoints_equivalent`): an external-endpoint user may
    legitimately serve any model name.
    """
    tr = merged.get("translation")
    dtr = defaults.get("translation")
    if not (
        isinstance(tr, dict) and isinstance(dtr, dict)
        and tr.get("model") != dtr.get("model")
        and _endpoints_equivalent(tr.get("endpoint"), dtr.get("endpoint"))
    ):
        return False
    _log.warning(
        "config key 'translation.model' in %s = %r does not match the "
        "bundled translator alias %r and translation.endpoint is the "
        "bundled default; using the bundled alias. Remove the stale "
        "override from %s to silence this.",
        source_name, tr.get("model"), dtr.get("model"), source_name,
    )
    tr["model"] = dtr.get("model")
    return True


def _prune_stale_model_override(overrides_path: Path) -> None:
    """Self-heal after the stale-model rescue: remove translation.model
    from the overrides file so the warning fires once, not on every
    launch. Runs a fresh read under the cross-process writer lock: a
    concurrent rescuer that got there first finds the key already gone
    and writes nothing. Best-effort -- a failed prune leaves the stale
    override in place (the rescue re-fires next launch) and must never
    fail the config load itself.
    """
    # Lazy import avoids a cycle (config_writer imports us for defaults).
    from . import config_writer
    try:
        with config_writer._locked_overrides(overrides_path):
            overrides = config_writer._read_yaml(overrides_path)
            tr = overrides.get("translation")
            if not (isinstance(tr, dict) and "model" in tr):
                return
            config_writer._remove_path_and_prune(
                overrides, ["translation", "model"],
            )
            config_writer._flush_tree(overrides_path, overrides)
    except OSError as e:
        _log.warning(
            "could not prune stale translation.model from %s: %s",
            overrides_path.name, e,
        )


def _normalize_merged_log_level(merged: dict) -> "str | None":
    """Coerce ``merged['logging']['level']`` to a stdlib level NAME in
    place and RETURN the junk-value warning text (or ``None``) -- it
    must NOT be emitted here.

    Single source: the pipeline's ``setup_main_log`` setLevel, the
    watchdog's setLevel, and the overlay ``--log-level`` forward all read
    this normalized value, so a lowercase ``debug`` / numeric / ``verbose``
    override cannot reach a ``logger.setLevel`` that would raise
    ``ValueError`` and abort startup. Delegates to the same
    :func:`yamabiko.log_config.normalize_log_level` those consumers use so
    the normalization can never drift. Lazy import avoids a load-time
    cycle (log_config imports only ``paths``).

    Emission sequencing is the reason for the return-don't-emit shape:
    the watchdog calls ``load_full_config`` BEFORE ``setup_main_log``
    attaches its file handler, so a warning logged here dies on the
    frozen bundle's discarded stderr -- the exact dialog-with-zero-trace
    class the watchdog's log-before-mutex ordering exists to prevent.
    ``load_full_config`` stashes the returned text; each role consumes
    it via :func:`consume_level_warning` and emits AFTER its handler is
    attached (the watchdog's ``_level_warn`` seam, build_pipeline's
    post-load re-emit).
    """
    from .log_config import normalize_log_level
    logging_block = merged.get("logging")
    if not isinstance(logging_block, dict):
        return None
    name, warning = normalize_log_level(logging_block.get("level"))
    logging_block["level"] = name
    return warning


# Deferred junk-``logging.level`` warning from the most recent
# ``load_full_config`` call. See ``_normalize_merged_log_level`` for the
# sequencing rationale. Reassigned (not or-ed) on EVERY load so a stale
# warning can never leak into a later, clean load's role.
_pending_level_warning: "str | None" = None


def consume_level_warning() -> "str | None":
    """Return and CLEAR the junk ``logging.level`` warning stashed by
    the most recent :func:`load_full_config` call (``None`` if the
    configured level was valid or absent).

    Callers (the watchdog's build, the pipeline's build) invoke this
    right after their config load and emit the text once their role's
    log handler is attached, so the evidence reaches
    main-<role>.log instead of the frozen bundle's discarded stderr.
    """
    global _pending_level_warning
    warning = _pending_level_warning
    _pending_level_warning = None
    return warning


def _read_explicit_config(explicit_path: Path) -> dict:
    """Read a ``--config`` file into the overlay layer.

    Fatal-config channel: a missing file, an unreadable file, a YAML
    parse error, or a non-mapping top level all raise ``RuntimeError`` so
    the pipeline's existing startup handler (build_pipeline's
    ``except RuntimeError`` -> tailored dialog + ``sys.exit(43)``) catches
    it, instead of an uncaught ``FileNotFoundError`` / ``YAMLError`` the
    watchdog reads as a generic crash and restart-loops. An empty/None
    file is treated as an empty mapping (pure defaults) with a WARNING.
    """
    if not explicit_path.exists():
        raise RuntimeError(
            f"--config file not found: {explicit_path} "
            f"(the path passed to --config does not exist)"
        )
    try:
        text = explicit_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise RuntimeError(
            f"--config file could not be read: {explicit_path}: {e}"
        ) from e
    if data is None:
        _log.warning(
            "--config file %s is empty; using shipped defaults",
            explicit_path,
        )
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(
            f"--config file {explicit_path}: expected a top-level mapping, "
            f"got {type(data).__name__}"
        )
    return data


def _read_shipped_defaults() -> dict:
    """Read the bundled default_config.yaml, routing an unreadable or
    unparseable (but present) file into the tailored broken-install
    RuntimeError channel. Corrupt-but-present shipped defaults
    (truncated extraction, disk damage) and unreadable ones (ACL
    accident, AV exclusive open) must land in the SAME channel as the
    missing/empty guard in load_full_config -- a raw YAMLError/OSError
    crash-loops through the watchdog into the generic dialog instead.
    """
    try:
        return _read_yaml(paths.shipped_defaults_path())
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeError(
            f"bundled defaults unreadable or unparseable: "
            f"{paths.shipped_defaults_path()} ({e}). Broken install? "
            f"(default_config.yaml must ship intact inside the bundle; "
            f"check installer/yamabiko.spec datas)"
        ) from e


def load_full_config(explicit_path: Path | None = None) -> dict:
    """Load the effective merged config and return it as one flat dict.

    Two modes:

    * ``explicit_path is None`` (production): shipped defaults + the sparse
      ``user_overrides.yaml`` are deep-merged.

      1. Read shipped defaults from :func:`paths.shipped_defaults_path`.
      2. If legacy ``config.yaml`` exists but ``user_overrides.yaml`` does
         not, run the one-time migration: diff legacy against defaults,
         write the minimal diff to ``user_overrides.yaml``, rename the
         legacy file to ``config.yaml.migrated.bak``.
      3. If both files exist, prefer the overrides file and log a
         WARNING — the legacy file is left alone.
      4. Deep-merge defaults + overrides, warn about unknown / wrong-typed
         keys, return the merged dict.

    * ``explicit_path`` given (dev / repro escape hatch): the layer is
      shipped defaults <- the explicit file. ``user_overrides.yaml`` is
      DELIBERATELY SKIPPED — the point of ``--config`` is an isolated,
      reproducible config that does not pick up ambient user state. The
      same unknown-key and type-mismatch walkers run against the explicit
      layer (identical strip + warn semantics to overrides). An empty /
      ``None`` file is treated as an empty mapping (pure defaults) with a
      WARNING. A missing file or YAML parse error raises ``RuntimeError``
      (the fatal-config channel the pipeline's startup handler catches).
    """
    paths.ensure_dirs()
    defaults = _read_shipped_defaults()
    if not defaults:
        # _read_yaml maps a missing/empty file to {}: fine for the
        # sparse overrides file, catastrophic for the schema source. With
        # defaults={}, every override would be stripped as "unknown", the
        # merge would return {}, and the pipeline would die far from the
        # cause with KeyError('audio') behind a wall of misleading
        # unknown-key warnings. A missing bundled default_config.yaml is
        # a broken install (yamabiko.spec datas regression); fail fast and
        # name the path.
        raise RuntimeError(
            f"bundled defaults missing or empty: "
            f"{paths.shipped_defaults_path()}. Broken install? "
            f"(default_config.yaml must ship inside the bundle; check "
            f"installer/yamabiko.spec datas)"
        )

    global _pending_level_warning

    if explicit_path is not None:
        # Escape-hatch path: layer the explicit file over defaults ONLY.
        # user_overrides.yaml and the legacy-config migration are skipped
        # so the run is isolated from ambient user state.
        explicit = _read_explicit_config(explicit_path)
        merged = _validate_and_merge(defaults, explicit, Path(explicit_path).name)
        # In-memory rescue only: an explicit --config file is not the
        # overrides file and is never rewritten.
        _rescue_stale_translation_model(merged, defaults, Path(explicit_path).name)
        _pending_level_warning = _normalize_merged_log_level(merged)
        return merged

    overrides_path = paths.user_overrides_path()
    legacy_path = paths.config_path()

    if legacy_path.exists() and not overrides_path.exists():
        _migrate_legacy_config(legacy_path, overrides_path, defaults)
    elif legacy_path.exists() and overrides_path.exists():
        _log.warning(
            "both %s and %s present; using %s (legacy file untouched)",
            legacy_path.name, overrides_path.name, overrides_path.name,
        )

    try:
        overrides = _read_yaml(overrides_path)
    except (yaml.YAMLError, OSError) as e:
        # OSError: an AV/backup tool holding an exclusive open (or a
        # permissions accident) must degrade to defaults exactly like a
        # parse error -- the app boots on defaults, not a crash loop.
        _log.warning("%s parse error: %s; ignoring overrides", overrides_path.name, e)
        overrides = {}

    merged = _validate_and_merge(defaults, overrides, overrides_path.name)
    if _rescue_stale_translation_model(merged, defaults, overrides_path.name):
        _prune_stale_model_override(overrides_path)
    _pending_level_warning = _normalize_merged_log_level(merged)
    return merged


def _walk_type_mismatches(
    overrides: Any, defaults: Any, prefix: str = "",
) -> list[tuple[str, str, str]]:
    """Return ``(dotted_path, expected_type, got_type)`` for every leaf
    override whose type differs from the corresponding default.

    Rules:
    * Only LEAF values are compared. Nested dicts recurse.
    * ``int`` where default is ``float`` is accepted (YAML lazy numbers).
    * ``None`` in override is always accepted (explicit unset).
    * ``bool`` vs ``int`` IS flagged (YAML's ``true``/``false`` vs ``1``/``0``).
    * Keys absent in defaults were already stripped by
      ``_walk_unknown_keys``, so we can assume every key here exists
      in ``defaults``.
    """
    if not isinstance(overrides, dict) or not isinstance(defaults, dict):
        return []
    out: list[tuple[str, str, str]] = []
    for k, ov in overrides.items():
        if k not in defaults:
            continue
        path = f"{prefix}.{k}" if prefix else k
        dv = defaults[k]
        if isinstance(dv, dict):
            if not isinstance(ov, dict):
                # A non-dict override at a SECTION key (e.g.
                # `overlay:` with all children commented out, which
                # YAML parses as None, or `audio: foo`) would replace
                # the entire defaults section in _deep_merge and crash
                # startup far from the cause. The old code recursed
                # here; the recursion's isinstance guard returned []
                # and the mismatch vanished. Leaf-level None stays
                # accepted (explicit unset) — this branch only fires
                # where the DEFAULT is a mapping.
                out.append((path, "mapping", type(ov).__name__))
                continue
            out.extend(_walk_type_mismatches(ov, dv, path))
            continue
        if ov is None:
            continue
        if isinstance(dv, bool):
            if not isinstance(ov, bool):
                out.append((path, "bool", type(ov).__name__))
        elif isinstance(dv, int):
            if isinstance(ov, bool) or not isinstance(ov, int):
                out.append((path, "int", type(ov).__name__))
        elif isinstance(dv, float):
            # int accepted here (YAML may write `5` for a float default)
            if not isinstance(ov, (int, float)) or isinstance(ov, bool):
                out.append((path, "float", type(ov).__name__))
        elif isinstance(dv, str):
            if not isinstance(ov, str):
                out.append((path, "str", type(ov).__name__))
        elif isinstance(dv, list):
            if not isinstance(ov, list):
                out.append((path, "list", type(ov).__name__))
        # else: unknown default type — accept anything.
    return out


@dataclass(frozen=True)
class NumericRule:
    minimum: int | float | None = None
    maximum: int | float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    integer_only: bool = False


_NUMERIC_RULES: dict[str, NumericRule] = {
    "overlay.websocket_port": NumericRule(1, 65535, integer_only=True),
    "audio.block_ms": NumericRule(1, integer_only=True),
    "audio.idle_death_s": NumericRule(1, integer_only=True),
    "vad.threshold": NumericRule(0.0, 1.0),
    "vad.min_silence_ms": NumericRule(1, integer_only=True),
    "vad.speech_pad_ms": NumericRule(0, integer_only=True),
    "vad.min_chunk_ms": NumericRule(1, integer_only=True),
    "vad.max_chunk_ms": NumericRule(1, integer_only=True),
    "vad.glue_max_silence_ms": NumericRule(1, integer_only=True),
    "vad.attach_max_ms": NumericRule(1, integer_only=True),
    "vad.drop_floor_prob": NumericRule(0.0, 1.0),
    "vad.drop_floor_ms": NumericRule(1, integer_only=True),
    "vad.dip_split_threshold": NumericRule(0.0, 1.0),
    "vad.dip_split_min_piece_ms": NumericRule(1, integer_only=True),
    "asr.boundary_pad_ms": NumericRule(0, integer_only=True),
    "asr.faster_whisper.beam_size": NumericRule(1, integer_only=True),
    "asr.faster_whisper.no_speech_only_threshold": NumericRule(
        0.0, 1.0, minimum_inclusive=False,
    ),
    "translation.max_tokens": NumericRule(1, integer_only=True),
    "translation.temperature": NumericRule(0.0, 2.0),
    "translation.context_lines": NumericRule(0, integer_only=True),
    "translation.startup_readiness_timeout_s": NumericRule(
        0.0, minimum_inclusive=False,
    ),
    "translation.startup_readiness_poll_s": NumericRule(
        0.0, minimum_inclusive=False,
    ),
    "gazetteer.fuzzy_max_distance": NumericRule(0.0, 1.0),
    "lifecycle.health.poll_interval_s": NumericRule(
        0.0, minimum_inclusive=False,
    ),
    "lifecycle.health.fail_budget_s": NumericRule(
        0.0, minimum_inclusive=False,
    ),
    "lifecycle.grace_s.pipeline": NumericRule(
        0.0, minimum_inclusive=False,
    ),
    "lifecycle.grace_s.overlay": NumericRule(
        0.0, minimum_inclusive=False,
    ),
    "lifecycle.overlay.max_restarts": NumericRule(1, integer_only=True),
    "lifecycle.overlay.restart_window_s": NumericRule(
        0.0, minimum_inclusive=False,
    ),
}


def _walk_null_violations(
    overrides: Any, defaults: Any, prefix: str = "",
) -> list[str]:
    """Return dotted paths where an override LEAF is ``None`` but the
    shipped default is NOT ``None``.

    ``None`` is valid only on nullable-by-default keys (a key whose
    shipped default is itself ``None`` -- ``overlay.geometry`` is the only
    one today). Elsewhere a ``None`` override sails past
    ``_walk_type_mismatches`` (which treats ``None`` as "explicit unset")
    and then detonates a consumer that coerces unconditionally
    (``int(port)``, ``float(grace)``).

    A ``None`` sitting on a dict (section) default is a mapping mismatch
    handled by ``_walk_type_mismatches``; this walker only recurses into
    dict/dict pairs and flags scalar leaves, so it never double-reports
    a section.
    """
    out: list[str] = []
    if not isinstance(overrides, dict) or not isinstance(defaults, dict):
        return out
    for k, ov in overrides.items():
        if k not in defaults:
            continue
        path = f"{prefix}.{k}" if prefix else k
        dv = defaults[k]
        if isinstance(dv, dict):
            if isinstance(ov, dict):
                out.extend(_walk_null_violations(ov, dv, path))
            continue
        if ov is None and dv is not None:
            out.append(path)
    return out


class ConfigValidationError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems))


_MISSING = object()


def _lookup_dotted(config: Mapping[str, Any], dotted: str) -> object:
    current: object = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _numeric_rule_problems(config: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for dotted, rule in _NUMERIC_RULES.items():
        value = _lookup_dotted(config, dotted)
        if value is _MISSING:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not _is_finite_number(value):
            problems.append(f"{dotted} must be finite; got {value!r}")
            continue
        if rule.integer_only and not isinstance(value, int):
            problems.append(f"{dotted} must be an integer; got {value!r}")
            continue
        if rule.minimum is not None and (
            value < rule.minimum
            or (value == rule.minimum and not rule.minimum_inclusive)
        ):
            operator = ">=" if rule.minimum_inclusive else ">"
            problems.append(
                f"{dotted} must be {operator} {rule.minimum}; got {value!r}"
            )
        if rule.maximum is not None and (
            value > rule.maximum
            or (value == rule.maximum and not rule.maximum_inclusive)
        ):
            operator = "<=" if rule.maximum_inclusive else "<"
            problems.append(
                f"{dotted} must be {operator} {rule.maximum}; got {value!r}"
            )
    return problems


def validate_config_semantics(config: Mapping[str, Any]) -> None:
    problems = _numeric_rule_problems(config)
    vad = config.get("vad")
    if isinstance(vad, Mapping) and (
        isinstance(vad.get("min_chunk_ms"), (int, float))
        and not isinstance(vad.get("min_chunk_ms"), bool)
        and isinstance(vad.get("max_chunk_ms"), (int, float))
        and not isinstance(vad.get("max_chunk_ms"), bool)
        and vad["min_chunk_ms"] > vad["max_chunk_ms"]
    ):
        problems.append(
            "vad.min_chunk_ms must be less than or equal to vad.max_chunk_ms"
        )
    if isinstance(vad, Mapping) and (
        isinstance(vad.get("speech_pad_ms"), (int, float))
        and not isinstance(vad.get("speech_pad_ms"), bool)
        and isinstance(vad.get("attach_max_ms"), (int, float))
        and not isinstance(vad.get("attach_max_ms"), bool)
        and vad["attach_max_ms"] < vad["speech_pad_ms"]
    ):
        problems.append(
            "vad.attach_max_ms must be greater than or equal to "
            "vad.speech_pad_ms"
        )
    translation = config.get("translation")
    if isinstance(translation, Mapping) and (
        isinstance(
            translation.get("startup_readiness_timeout_s"), (int, float)
        )
        and not isinstance(translation.get("startup_readiness_timeout_s"), bool)
        and isinstance(
            translation.get("startup_readiness_poll_s"), (int, float)
        )
        and not isinstance(translation.get("startup_readiness_poll_s"), bool)
        and translation["startup_readiness_poll_s"]
        > translation["startup_readiness_timeout_s"]
    ):
        problems.append(
            "translation.startup_readiness_poll_s must be less than or equal "
            "to translation.startup_readiness_timeout_s"
        )
    if problems:
        raise ConfigValidationError(problems)


def _strip_keys(tree: dict, dotted_paths: list[str]) -> dict:
    """Return ``tree`` with the given dotted key paths removed (non-mutating)."""
    import copy
    out = copy.deepcopy(tree)
    for dotted in dotted_paths:
        parts = dotted.split(".")
        cur = out
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                cur = None
                break
            cur = cur[p]
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    return out


def _migrate_legacy_config(legacy_path: Path, overrides_path: Path, defaults: dict) -> None:
    """One-shot: legacy config.yaml -> sparse user_overrides.yaml + .bak."""
    try:
        legacy_data = _read_yaml(legacy_path)
    except (yaml.YAMLError, OSError) as e:
        # OSError: an AV/backup tool holding the legacy file exclusively
        # open on the first post-upgrade launch must degrade exactly like
        # a parse error (skip migration, boot on defaults), not crash-loop
        # all three processes through the watchdog.
        _log.warning(
            "legacy %s could not be read during migration: %s; "
            "leaving file in place", legacy_path.name, e,
        )
        return
    diff = _deep_diff(legacy_data, defaults)
    minimal: dict = {} if diff is _NOOP else diff

    # A legacy translation.model pin on the bundled-default endpoint is
    # the stale-alias rename trap (see _rescue_stale_translation_model);
    # carrying it into the fresh overrides file would 404 the startup
    # sanity translate on every launch until the rescue prunes it. Drop
    # it at migration time so migrated installs never write it at all.
    tr = minimal.get("translation")
    dtr = defaults.get("translation")
    if (
        isinstance(tr, dict) and "model" in tr and isinstance(dtr, dict)
        and _endpoints_equivalent(
            tr.get("endpoint", dtr.get("endpoint")), dtr.get("endpoint"),
        )
    ):
        del tr["model"]
        if not tr:
            del minimal["translation"]

    # Lazy import avoids a cycle (config_writer imports us for defaults).
    from . import config_writer
    if minimal:
        # Under the cross-process filelock like every other overrides
        # writer: watchdog, pipeline and overlay all run this loader on
        # the first post-upgrade launch and can race the migration.
        with config_writer._locked_overrides(overrides_path):
            config_writer._atomic_write(overrides_path, minimal)
    # Always mark the legacy file as migrated so the branch doesn't fire
    # twice, even when the diff was empty (user had exactly the shipped
    # defaults frozen in their copied config.yaml).
    backup = paths.legacy_config_backup_path()
    try:
        import os
        os.replace(str(legacy_path), str(backup))
    except OSError as e:
        _log.warning("could not rename %s to %s: %s", legacy_path.name, backup.name, e)
        return
    _log.info(
        "migrated legacy %s to %s; backup saved at %s",
        legacy_path.name, overrides_path.name, backup,
    )


# ---------------------------------------------------------------------------
# Overlay-specific loader (kept for callers that only want the overlay slice)
# ---------------------------------------------------------------------------


def _isfinite_no_overflow(v) -> bool:
    """math.isfinite that treats an int beyond the float ceiling as
    non-finite instead of raising OverflowError."""
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _clamp_style(style: dict) -> dict:
    out = dict(STYLE_DEFAULTS)
    for k, v in style.items():
        if k not in out:
            continue
        rng = _STYLE_RANGES.get(k)
        if rng is not None:
            # Ranged keys must be real, finite numbers. A present-but-garbage
            # value (None, "bad") would otherwise fall through unclamped and
            # blow up OverlayWindow._cap_px()'s int(width * pct / 100) with a
            # TypeError on the next add_subtitle, killing every subtitle.
            # bool is rejected too: isinstance(True, int) is True in Python,
            # so a YAML `max_width_pct: true` would clamp to 1 instead of
            # being caught. NaN/Infinity are floats (they pass isinstance) but
            # detonate int(width * nan / 100) with a ValueError; they reach
            # here via a loopback WS set_style frame since json.loads accepts
            # NaN/Infinity literals. Mirror the colour-key guard below: keep
            # default. isfinite is guarded by the isinstance clause against
            # str/None (TypeError), but an int above the float ceiling
            # (~1.8e308) still makes it raise OverflowError. Style keys are
            # deliberately outside _NUMERIC_RULES because this path clamps
            # them instead of rejecting the full configuration.
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not _isfinite_no_overflow(v)
            ):
                _log.warning(
                    "style.%s=%r is not numeric; keeping default %s",
                    k, v, out[k],
                )
                continue  # keep the shipped default
            lo, hi = rng
            if v < lo:
                _log.warning("style.%s=%r below min %d; clamping", k, v, lo)
                v = lo
            elif v > hi:
                _log.warning("style.%s=%r above max %d; clamping", k, v, hi)
                v = hi
        elif k in COLOR_STYLE_KEYS:
            # Validate colours at the config boundary so a bad value in
            # user_overrides.yaml can't reach the overlay and crash the render
            # path (which silently dropped every subtitle).
            if not (isinstance(v, str) and HEX_COLOR_RE.match(v)):
                _log.warning(
                    "style.%s=%r is not a #RRGGBB colour; keeping default %s",
                    k, v, out[k],
                )
                continue  # keep the shipped default
        elif k in FONT_STYLE_KEYS:
            # QFont(5, ...) raises TypeError inside every add_subtitle, so a
            # non-string family from a WS set_style frame or a hand-edited
            # override would drop every subtitle. Keep the shipped family.
            if not (isinstance(v, str) and v.strip()):
                _log.warning(
                    "style.%s=%r is not a font family name; keeping default %s",
                    k, v, out[k],
                )
                continue  # keep the shipped default
        elif k in STYLE_CHOICES:
            # Mode / Alignment / Echo / Banner: anything but one of the
            # listed values (a typo, a loopback WS frame) keeps the default.
            if v not in STYLE_CHOICES[k]:
                _log.warning(
                    "style.%s=%r is not one of %s; keeping default %s",
                    k, v, STYLE_CHOICES[k], out[k],
                )
                continue
        out[k] = v
    return out


def load_overlay_config(
    path: Path | Mapping[str, Any] | None = None,
) -> OverlayConfig:
    """Load just the overlay slice.

    ``path`` is retained for backwards compatibility. A mapping is treated
    as an already-merged, already-validated full config so composition roots
    can derive the overlay slice without a second disk read. When a path is
    given (the pipeline reads the initial overlay style this way under
    ``--config``), the file is layered over the shipped defaults through the SAME
    :func:`_validate_and_merge` the full loader uses -- so a ``--config``
    that omits the overlay section (or mistypes a value) resolves the
    slice from defaults instead of an empty/garbage block, and unknown /
    wrong-typed keys are stripped + warned identically. Unlike
    :func:`load_full_config` with ``explicit_path``, this keeps the
    graceful-degrade contract: a missing file or a parse error returns the
    dataclass defaults rather than raising. When ``None``, this delegates
    to :func:`load_full_config` and extracts the overlay block from the
    merged result.
    """
    if isinstance(path, Mapping):
        raw = path
    elif path is None:
        raw = load_full_config()
    else:
        if not path.exists():
            return OverlayConfig()
        try:
            overlay_layer = _read_yaml(path)
        except yaml.YAMLError as e:
            _log.warning("config parse error: %s; using defaults", e)
            return OverlayConfig()
        # Layer the explicit file over shipped defaults with the shared
        # unknown-key / type / nullability / numeric validation, so the
        # overlay process (and the pipeline seeding its style) can't crash
        # on a malformed or partial --config the verbatim read let through.
        # The defaults read shares the broken-install RuntimeError channel
        # (build_pipeline catches configuration failures; a raw YAMLError from
        # this seeding path is a generic crash + restart loop).
        defaults = _read_shipped_defaults()
        raw = _validate_and_merge(defaults, overlay_layer, path.name)
        # In-memory rescue only, matching the --config branch of
        # load_full_config: the explicit file is never rewritten.
        _rescue_stale_translation_model(raw, defaults, path.name)
    block = raw.get("overlay") or {}
    if not isinstance(block, dict):
        # A truthy non-dict overlay block (e.g. `overlay: "on"` in an explicit
        # --config file, which bypasses load_full_config's type checker) would
        # otherwise crash the very next block.get() with an AttributeError.
        _log.warning("overlay=%r is not a mapping; using defaults", block)
        block = {}
    # style / position mirror the geometry + theme graceful-degrade below: a
    # truthy NON-dict override ("garbage-string", [1, 2]) reaches this loader
    # via the --config path (load_full_config's _walk_type_mismatches only
    # guards the layered path), and `... or {}` passes the truthy scalar/list
    # straight through -- _clamp_style('..').items() / clamp_position([..]).get()
    # then raise AttributeError, uncaught through boot. Coerce to {} + warn.
    style_raw = block.get("style") or {}
    if not isinstance(style_raw, dict):
        _log.warning("overlay.style=%r is not a mapping; using defaults", style_raw)
        style_raw = {}
    style = _clamp_style(style_raw)
    geom_raw = block.get("geometry") or None
    geometry = None
    if isinstance(geom_raw, dict) and all(
        isinstance(geom_raw.get(k), int) for k in ("x", "y", "width", "height")
    ):
        geometry = {k: geom_raw[k] for k in ("x", "y", "width", "height")}
    theme = block.get("theme", "system")
    if theme not in ("system", "light", "dark"):
        _log.warning("overlay.theme=%r invalid; using system", theme)
        theme = "system"
    from .overlay_app.position_config import clamp_position
    pos_raw = block.get("position") or {}
    if not isinstance(pos_raw, dict):
        _log.warning("overlay.position=%r is not a mapping; using defaults", pos_raw)
        pos_raw = {}
    position = clamp_position(pos_raw)
    reduce_motion = bool(block.get("reduce_motion", False))
    return OverlayConfig(
        style=style, geometry=geometry, theme=theme,
        position=position, reduce_motion=reduce_motion,
    )
