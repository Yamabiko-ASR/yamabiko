"""Pure-module tests for the overlay position model.

`yamabiko.overlay_app.position_config` has NO Qt imports, so these run
bare. Two functions under test:

* ``clamp_position`` coerces a possibly-garbage position dict into the
  valid anchor model (bounds + type guards + default fill).
* ``geometry_to_position`` is the one-time migration from the legacy
  free-geometry ``{x,y,width,height}`` rect to the anchor model. This is
  the load-bearing math: a user's saved rect must land on the right
  screen at the right height/center after the config schema flips.
"""
from yamabiko.overlay_app.position_config import (
    geometry_to_position,
    clamp_position,
    migrate_legacy_geometry,
)

SCREENS = [("\\\\.\\DISPLAY1", (0, 0, 2560, 1440)),
           ("\\\\.\\DISPLAY2", (2560, 0, 2560, 1440))]


def test_migrates_users_actual_straddling_rect():
    """The real rect from the field: x=40 w=2648 straddles both monitors;
    majority overlap area is DISPLAY1; bottom edge 857/1440 = 59.5%.

    Seeded defect (one line): corrupt the overlap left-edge from
    ``max(x, sx)`` to ``min(x, sx)`` -> DISPLAY2 wins the area contest ->
    screen assertion reads DISPLAY2 -> red. This is the straddle case the
    task's seeded-defect drill targets.
    """
    pos = geometry_to_position({"x": 40, "y": 416, "width": 2648, "height": 441}, SCREENS)
    assert pos["screen"] == "\\\\.\\DISPLAY1"
    assert abs(pos["v_offset_pct"] - 59.5) < 0.2
    assert pos["h_offset_pct"] == 50.0   # center 1364/2560=53.3% snaps to 50


def test_offcenter_rect_keeps_h_offset():
    """An off-center rect must NOT snap its horizontal offset to 50.

    STRENGTHENED beyond the brief (disclosed): the brief asserted only
    ``h_offset_pct != 50.0``, which a wrong h-formula that lands on any
    other value would still satisfy (a weak inequality Liar). We also pin
    the exact expected center (400/2560 = 15.6%) so a broken h-formula
    goes red too, not just an over-eager snap.

    Seeded defects (each one line, each red):
    - widen the snap window so this center snaps to 50 -> first assert red.
    - drop the ``w / 2`` center term in the h-formula -> 15.6 assert red.
    """
    pos = geometry_to_position({"x": 100, "y": 1000, "width": 600, "height": 200}, SCREENS)
    assert pos["h_offset_pct"] != 50.0   # center 400/2560 = 15.6%, no snap
    assert abs(pos["h_offset_pct"] - 15.6) < 0.2


def test_fully_offscreen_rect_returns_none():
    """A VALID rect that overlaps no connected screen must return None, NOT
    invented defaults.

    This is the P0-5 contract: the caller (build_overlay's migration block)
    must be able to tell "the placement is off all current screens" apart from
    "the placement maps here", because the former must NOT persist -- persisting
    would prune the legacy overlay.geometry and destroy an undocked laptop's
    external-monitor placement unrecoverably. None is that signal.

    Seeded defect (one line): revert the no-overlap branch to ``return
    dict(_DEFAULT)`` -> pos is the primary-default dict, not None -> red.
    """
    pos = geometry_to_position({"x": 99999, "y": 99999, "width": 100, "height": 100}, SCREENS)
    assert pos is None


def test_unparseable_rect_falls_back_to_primary_defaults():
    """A rect that can't even be parsed (missing/garbage fields) is corrupt
    config with no placement to preserve, so it returns the shipped defaults
    (the caller then persists defaults + prunes the garbage key). This is a
    DIFFERENT contract from the valid-but-offscreen None case above.

    Seeded defect (one line): change the malformed branch to ``return None``
    -> pos is None, not the default dict -> red.
    """
    pos = geometry_to_position({"x": "nope", "height": 100}, SCREENS)
    assert pos == {"screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
                   "left_offset_pct": 6.0}


def test_zero_area_rect_is_unparseable_not_offscreen():
    """A zero-width/height rect (all ints, so it parses) has zero overlap
    with EVERY screen, so it used to land in the offscreen-retry branch --
    but no monitor topology can ever make a zero-area rect overlap
    anything, so the "retry when the monitor returns" recovery looped a
    warning + clamped defaults on every boot, forever. There is no
    placement to preserve in a degenerate rect: it must take the
    unparseable branch (defaults + prune).

    Seeded defect: drop the w/h > 0 check from _parse_geom -> the rect
    parses, matches nothing, and geometry_to_position returns None
    instead of the defaults dict -> red.
    """
    for geom in (
        {"x": 100, "y": 100, "width": 0, "height": 300},
        {"x": 100, "y": 100, "width": 300, "height": 0},
        {"x": 100, "y": 100, "width": -300, "height": 200},
    ):
        pos = geometry_to_position(geom, SCREENS)
        assert pos == {
            "screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
            "left_offset_pct": 6.0,
        }, f"degenerate rect {geom} must prune, not retry forever"


def test_clamp_position_bounds_and_garbage():
    """clamp_position must bound each offset to its range and reject
    non-string screens / non-numeric offsets by keeping the default.

    Seeded defects (each one line, each red):
    - drop ``min(hi, max(lo, v))`` -> 200 / -4 pass through -> red.
    - drop the ``isinstance(pos.get("screen"), str)`` guard -> screen=3
      is kept -> first assert's screen reads 3 -> red.
    - drop the empty-dict default fill -> clamp_position({}) is missing
      keys -> KeyError / mismatch -> red.
    """
    assert clamp_position({"screen": 3, "v_offset_pct": 200, "h_offset_pct": -4}) == {
        "screen": "primary", "v_offset_pct": 98.0, "h_offset_pct": 10.0,
        "left_offset_pct": 6.0}
    assert clamp_position({}) == {"screen": "primary", "v_offset_pct": 88.0,
                                  "h_offset_pct": 50.0, "left_offset_pct": 6.0}


def test_clamp_position_rejects_bools_keeping_default():
    """A bool offset must be rejected, not treated as a number.

    ``isinstance(True, (int, float))`` is True in Python, so without an
    explicit bool guard ``clamp_position({"v_offset_pct": True})`` clamps
    ``True`` -> 1 -> 5.0 (the low bound) instead of keeping the 88.0 default.
    This is reachable in production: ``load_overlay_config(path=...)`` (the
    ``--config`` CLI path) clamps directly and skips ``_walk_type_mismatches``,
    the only layer that would otherwise strip a bool override.

    Seeded defects (each one line, each red):
    - delete the ``isinstance(v, bool)`` term from the guard -> ``True`` clamps
      to 5.0 -> first assert reads 5.0 != 88.0 -> red.
    - widen the guard to reject all numbers -> the real 3.0 float below stops
      clamping (returns the 88.0 default) -> third assert reads 88.0 != 5.0 ->
      red. Pins that the guard cannot overreach past bools.
    """
    assert clamp_position({"v_offset_pct": True})["v_offset_pct"] == 88.0
    assert clamp_position({"h_offset_pct": False})["h_offset_pct"] == 50.0
    # A genuine float still clamps to its low bound (guard must not overreach).
    assert clamp_position({"v_offset_pct": 3.0})["v_offset_pct"] == 5.0


def test_clamp_position_rejects_non_finite_keeping_default():
    """NaN/Inf offsets must be rejected, not clamped (C13).

    NaN passes ``isinstance(v, (int, float))``, and in the clamp's argument
    order ``min(98.0, max(5.0, nan))`` evaluates to 5.0 -- so a persisted
    ``v_offset_pct: .nan`` (YAML accepts the literal, and a loopback WS frame
    can carry it via json.loads) silently anchored the overlay at the TOP of
    the screen instead of keeping the 88.0 default. config._clamp_style was
    hardened against exactly this in the same PR; clamp_position must match.

    Seeded defects (each one line, each red):
    - delete the ``not math.isfinite(v)`` term from the guard -> nan clamps
      to 5.0 -> first assert reads 5.0 != 88.0 -> red.
    - the +/-inf cases pin that BOTH tails are rejected, not min/max'd to a
      bound (inf would otherwise clamp to 98.0 / 5.0 -- a plausible-looking
      value that hides the garbage input).
    """
    assert clamp_position({"v_offset_pct": float("nan")})["v_offset_pct"] == 88.0
    assert clamp_position({"h_offset_pct": float("inf")})["h_offset_pct"] == 50.0
    assert clamp_position({"v_offset_pct": float("-inf")})["v_offset_pct"] == 88.0


def test_clamp_position_keeps_default_for_int_above_float_ceiling():
    """A 309-digit int passes the (int, float) isinstance gate and then
    ``math.isfinite`` raises OverflowError ("int too large to convert to
    float") -- unguarded, a hand-edited ``v_offset_pct: <huge>`` crashed
    BOTH children at boot (load escapes the RuntimeError-only catch in
    build_pipeline) into a watchdog restart loop. Must keep the default,
    like every other non-finite input.

    Seeded defect: drop the OverflowError handling from the guard ->
    this test errors with OverflowError -> red.
    """
    huge = 10 ** 309
    assert clamp_position({"v_offset_pct": huge})["v_offset_pct"] == 88.0
    assert clamp_position({"h_offset_pct": -huge})["h_offset_pct"] == 50.0


# ---------------------------------------------------------------------------
# migrate_legacy_geometry: (position, max_width_pct, clamped)
# P0-6 width mapping + C9 clamp traceability
# ---------------------------------------------------------------------------


def test_migrate_maps_narrow_legacy_width_clamped_to_floor():
    """A deliberate narrow legacy resize (600 px on a 2560 px monitor) is a
    user choice that must survive the schema flip -- but 600/2560 = 23.4% is
    below the 30% floor, so it clamps UP to 30, not the 70% default.

    Regression (P0-6): the legacy rect's WIDTH was read for the h-center but
    never mapped to max_width_pct, so every migrated overlay snapped to the 70%
    default and a user's narrow strip wrapped to ~1790 px, unrecoverable after
    the migration pruned overlay.geometry.

    Seeded defects (each one line, each red):
    - return a fixed 70 for the pct in migrate_legacy_geometry -> 30 assert red.
    - drop the ``max(_MAX_WIDTH_PCT_LO, ...)`` clamp -> pct reads 23 -> red.
    """
    pos, pct, clamped = migrate_legacy_geometry(
        {"x": 100, "y": 1000, "width": 600, "height": 200}, SCREENS
    )
    assert pos is not None
    assert pct == 30, f"600px on 2560 -> raw 23.4 -> clamp to 30 floor; got {pct}"
    # The 30% floor MOVED the encoded 23.4% width -- the flag must say so, so
    # the caller can leave the recoverable-trace warning (C9).
    assert clamped is True, "width floored 23.4 -> 30 but the clamped flag is False"


def test_migrate_maps_mid_legacy_width_unclamped():
    """A 1500 px legacy width on a 2560 px monitor maps to 1500/2560 = 58.6% ->
    round 59, which sits inside 30..100 and is NOT clamped.

    Seeded defect (one line): floor instead of round (``int`` not ``round``) ->
    pct reads 58, not 59 -> red. Pins the rounding, not just the range.
    """
    pos, pct, clamped = migrate_legacy_geometry(
        {"x": 100, "y": 1000, "width": 1500, "height": 200}, SCREENS
    )
    assert pos is not None
    assert pct == 59, f"1500px on 2560 -> round(58.6) == 59; got {pct}"
    # Nothing was moved (v 83.3, h 33.2, width 59 all in bounds) -- the flag
    # must stay False or every migration would log the C9 warning noise.
    assert clamped is False, "in-bounds migration wrongly flagged as clamped"


def test_migrate_straddling_rect_keeps_position_and_maps_majority_width():
    """The real field rect (x=40 w=2648 straddling both monitors) must keep the
    Task-4 position numbers AND derive max_width_pct from the MAJORITY screen's
    width (DISPLAY1, 2560 px): 2648/2560 = 103.4% -> clamps to the 100 ceiling.

    Seeded defects (each one line, each red):
    - derive the pct from the wrong screen's width -> pct changes -> red.
    - drop the ``min(_MAX_WIDTH_PCT_HI, ...)`` ceiling -> pct reads 103 -> red.
    """
    pos, pct, clamped = migrate_legacy_geometry(
        {"x": 40, "y": 416, "width": 2648, "height": 441}, SCREENS
    )
    assert pos["screen"] == "\\\\.\\DISPLAY1"
    assert abs(pos["v_offset_pct"] - 59.5) < 0.2
    assert pos["h_offset_pct"] == 50.0
    assert pct == 100, f"2648px on 2560 majority screen -> raw 103 -> clamp to 100; got {pct}"
    # The 100 ceiling squeezed the encoded 103% straddle -- flagged for C9.
    assert clamped is True, "width ceilinged 103 -> 100 but the clamped flag is False"


def test_migrate_no_overlap_returns_none_none():
    """A valid-but-offscreen rect propagates the None position AND a None pct,
    so the caller skips BOTH the position write and the style write (keeping
    overlay.geometry on disk for a later boot).

    Seeded defect (one line): in migrate_legacy_geometry return ``position, 70``
    instead of ``None, None`` on the None position -> pct is 70, not None ->
    red (and the caller would wrongly persist a max_width_pct).
    """
    pos, pct, clamped = migrate_legacy_geometry(
        {"x": 99999, "y": 99999, "width": 100, "height": 100}, SCREENS
    )
    assert pos is None
    assert pct is None
    # Nothing was migrated, so nothing was clamped -- a True here would make
    # the caller log a C9 trace warning for a migration that never happened.
    assert clamped is False


def test_migrate_flags_edge_hugging_rect_as_clamped():
    """C9: an edge-hugging rect is a LEGAL legacy placement that the anchor
    bounds cannot represent -- clamp_position moves its center from 3.9% to
    the 10% floor and the width floor lifts 7.8% to 30. The migration itself
    stays lossy by design (review decision: no schema change), but the
    returned ``clamped`` flag must be True so the caller (build_overlay's
    migration block) can quote the original rect in a warning BEFORE the
    position write prunes overlay.geometry -- the log line is the only
    recoverable trace of the destroyed placement.

    Seeded defects (each one line, each red):
    - hardcode ``clamped = False`` in migrate_legacy_geometry -> red.
    - compare the POST-clamp position to itself (instead of pre vs post) ->
      the flag never trips -> red.
    """
    pos, pct, clamped = migrate_legacy_geometry(
        {"x": 0, "y": 1000, "width": 200, "height": 200}, SCREENS
    )
    assert pos["h_offset_pct"] == 10.0   # center 100/2560 = 3.9% -> 10 floor
    assert pct == 30                     # 200/2560 = 7.8% -> 30 floor
    assert clamped is True, (
        "an edge-hugging rect was moved by the h/width floors but the "
        "clamped flag is False; the caller cannot leave the C9 trace warning"
    )


def test_clamp_position_left_offset_range_and_default():
    """left_offset_pct (the Left alignment anchor) is clamped into 0..70
    and defaults to 6.0 for garbage, independently of h_offset_pct.

    Seeded defect: leave left_offset_pct out of clamp_position's key loop ->
    95.0 survives unclamped -> red."""
    from yamabiko.overlay_app.position_config import clamp_position

    assert clamp_position({})["left_offset_pct"] == 6.0
    assert clamp_position({"left_offset_pct": 95.0})["left_offset_pct"] == 70.0
    assert clamp_position({"left_offset_pct": -3})["left_offset_pct"] == 0.0
    assert clamp_position({"left_offset_pct": "x"})["left_offset_pct"] == 6.0
    out = clamp_position({"left_offset_pct": 12.5, "h_offset_pct": 40.0})
    assert (out["left_offset_pct"], out["h_offset_pct"]) == (12.5, 40.0)
