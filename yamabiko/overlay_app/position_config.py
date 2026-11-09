"""Pure position-model helpers: clamping and the one-time migration
from the legacy free-geometry rect. No Qt imports (testable bare)."""
from __future__ import annotations

import math

from .style_defaults import MAX_WIDTH_PCT_RANGE

# Public validation table: the Settings sliders and tray presets derive their
# bounds/defaults from HERE instead of re-typing the literals.
DEFAULT_POSITION = {
    "screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
    "left_offset_pct": 6.0,
}
# The tray flyout's "Middle of Display N" preset (a tray-only choice; Bottom
# is DEFAULT_POSITION's v_offset_pct).
MIDDLE_V_PCT = 55.0
V_OFFSET_RANGE: tuple[float, float] = (5.0, 98.0)
H_OFFSET_RANGE: tuple[float, float] = (10.0, 90.0)
# Left alignment anchors the stack's LEFT edge; past 70 % too little width
# remains for a readable line.
LEFT_OFFSET_RANGE: tuple[float, float] = (0.0, 70.0)

_DEFAULT = DEFAULT_POSITION  # internal alias, kept for the module's own uses
_H_SNAP_PCT = 2.0
# max_width_pct bounds: one source (style_defaults) shared with the Settings
# "Max width" slider and the overlay's _cap_px clamp. A migrated legacy width
# is squeezed into this window so a hair-thin or full-bleed legacy rect can't
# produce an unusable cap.
_MAX_WIDTH_PCT_LO, _MAX_WIDTH_PCT_HI = MAX_WIDTH_PCT_RANGE


def clamp_position(pos: dict) -> dict:
    out = dict(_DEFAULT)
    if isinstance(pos.get("screen"), str) and pos["screen"]:
        out["screen"] = pos["screen"]
    for key, (lo, hi) in (
        ("v_offset_pct", V_OFFSET_RANGE),
        ("h_offset_pct", H_OFFSET_RANGE),
        ("left_offset_pct", LEFT_OFFSET_RANGE),
    ):
        v = pos.get(key)
        # Reject bool before the numeric check: isinstance(True, int) is True
        # in Python, so a bare (int, float) test would clamp True->1->5.0
        # instead of keeping the default. NaN/Infinity are floats (they pass
        # isinstance) but NaN slips through this argument order --
        # min(98.0, max(5.0, nan)) == 5.0 -- silently anchoring the overlay
        # at the TOP of the screen; they reach here via a persisted YAML
        # ``.nan`` or a loopback WS frame (json.loads accepts the literals).
        # Mirror config._clamp_style's guard: keep the shipped default.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue  # keep the shipped default already in out
        try:
            finite = math.isfinite(v)
        except OverflowError:
            # An int beyond the float ceiling (~1.8e308): isfinite raises
            # instead of returning False; same keep-default outcome.
            finite = False
        if not finite:
            continue  # keep the shipped default already in out
        out[key] = float(min(hi, max(lo, v)))
    return out


def _parse_geom(geom: dict) -> tuple[int, int, int, int] | None:
    """Parse a legacy rect dict to ``(x, y, w, h)``; ``None`` if malformed.

    Non-positive width/height counts as malformed: a zero-area rect can
    never overlap any screen, so letting it parse routed it into the
    offscreen-retry branch -- which warns and re-runs the migration on
    EVERY boot forever, since no monitor topology can ever satisfy it.
    """
    try:
        rect = (int(geom["x"]), int(geom["y"]),
                int(geom["width"]), int(geom["height"]))
    except (KeyError, TypeError, ValueError):
        return None
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    return rect


def _match_screen(
    rect: tuple[int, int, int, int],
    screens: list[tuple[str, tuple[int, int, int, int]]],
) -> tuple[str, tuple[int, int, int, int]] | None:
    """Return ``(name, (sx, sy, sw, sh))`` of the connected screen with the
    largest overlap area with ``rect``, or ``None`` when the rect overlaps no
    connected screen."""
    x, y, w, h = rect
    best_name, best_area, best_rect = None, 0, None
    for name, (sx, sy, sw, sh) in screens:
        ox = max(0, min(x + w, sx + sw) - max(x, sx))
        oy = max(0, min(y + h, sy + sh) - max(y, sy))
        if ox * oy > best_area:
            best_name, best_area, best_rect = name, ox * oy, (sx, sy, sw, sh)
    if best_name is None:
        return None
    return best_name, best_rect


def _geometry_to_position_matched(
    geom: dict, screens: list[tuple[str, tuple[int, int, int, int]]],
) -> tuple[
    dict | None,
    tuple[int, int, int, int] | None,
    tuple[str, tuple[int, int, int, int]] | None,
    bool,
]:
    """Single-pass core shared by :func:`geometry_to_position` and
    :func:`migrate_legacy_geometry`: parse the rect ONCE, match a screen ONCE,
    and hand both intermediates to the migration wrapper (which needs the
    rect's width and the matched screen for the max_width_pct mapping).

    Returns ``(position, rect, match, clamped)``:

    * ``(defaults, None, None, False)`` for an UNPARSEABLE rect;
    * ``(None, rect, None, False)`` when the rect overlaps NO connected screen;
    * ``(position, rect, match, clamped)`` otherwise, where ``clamped`` is True
      when the anchor bounds MOVED the encoded placement (v/h clamped) -- a
      legal edge-hugging legacy rect the anchor model cannot represent.
    """
    rect = _parse_geom(geom)
    if rect is None:
        return dict(_DEFAULT), None, None, False
    match = _match_screen(rect, screens)
    if match is None:
        return None, rect, None, False
    x, y, w, h = rect
    name, (sx, sy, sw, sh) = match
    v = (y + h - sy) / sh * 100.0
    hc = (x + w / 2 - sx) / sw * 100.0
    if abs(hc - 50.0) <= _H_SNAP_PCT + 3.5:   # generous: legacy rects were sloppy
        hc = 50.0
    pre = {"screen": name, "v_offset_pct": round(v, 1), "h_offset_pct": round(hc, 1)}
    position = clamp_position(pre)
    # Compare the pre-clamp percentages to the clamped output: any difference
    # means the migration is about to MOVE a legal legacy placement (the caller
    # persists the result and prunes overlay.geometry, so it owes the user a
    # recoverable trace -- see migrate_legacy_geometry's clamped flag). The
    # center snap above is deliberate generosity, not loss, so it is applied
    # BEFORE the comparison and never trips the flag.
    clamped = (
        position["v_offset_pct"] != pre["v_offset_pct"]
        or position["h_offset_pct"] != pre["h_offset_pct"]
    )
    return position, rect, match, clamped


def geometry_to_position(
    geom: dict, screens: list[tuple[str, tuple[int, int, int, int]]],
) -> dict | None:
    """Map a legacy {x,y,width,height} rect to the anchor model.

    Returns:

    * the anchor position dict when the rect overlaps a connected screen.
      Screen = largest overlap area; v = the rect's BOTTOM edge as % of that
      screen's height; h = the rect's center as % of width, snapped to 50 when
      within 5.5% (generous because legacy rects were sloppy; the extra 3.5
      absorbs it).
    * the shipped default position when the rect is UNPARSEABLE (garbage
      config -- there is no user placement to preserve, so the caller may
      persist defaults and prune the legacy key).
    * ``None`` when the rect parses but overlaps NO connected screen (e.g. an
      undocked laptop whose placement was saved on an external monitor). The
      caller MUST NOT persist in this case: keep overlay.geometry on disk and
      retry the migration on a later boot when the monitor returns. Persisting
      here would prune the legacy rect and destroy the placement unrecoverably.
    """
    return _geometry_to_position_matched(geom, screens)[0]


def migrate_legacy_geometry(
    geom: dict, screens: list[tuple[str, tuple[int, int, int, int]]],
) -> tuple[dict | None, int | None, bool]:
    """One-time migration: (position, max_width_pct, clamped) from a legacy rect.

    Returns ``(position, max_width_pct, clamped)`` when the rect maps to a
    screen, where ``max_width_pct`` is the legacy rect's WIDTH as a percent of
    the matched screen's width, clamped to 30..100. This carries a deliberate
    narrow/wide legacy resize (e.g. a 600 px strip on a 2560 px monitor) across
    the schema flip instead of snapping every migrated overlay to the 70%
    default.

    ``clamped`` is True when the migration MOVED the encoded placement -- the
    anchor bounds clamped v/h, or the width was floored/ceilinged into
    MAX_WIDTH_PCT_RANGE. The result is still persisted (review decision: the
    anchor schema stays lossy), but the caller must log the ORIGINAL rect when
    the flag is set: the position write prunes overlay.geometry in the same
    locked write, so that log line is the only recoverable trace.

    Returns ``(defaults, None, False)`` for an unparseable rect (no width to
    map, no placement to trace) and ``(None, None, False)`` when the rect
    overlaps no connected screen -- in the latter case the caller must keep
    overlay.geometry on disk and retry next boot.
    """
    position, rect, match, clamped = _geometry_to_position_matched(geom, screens)
    if position is None:
        return None, None, False
    if match is None:
        # Unparseable rect -> defaults with no width to map.
        return position, None, False
    _name, (_sx, _sy, sw, _sh) = match
    _x, _y, w, _h = rect
    raw_pct = round(w / sw * 100)
    pct = max(_MAX_WIDTH_PCT_LO, min(_MAX_WIDTH_PCT_HI, raw_pct))
    return position, pct, clamped or pct != raw_pct
