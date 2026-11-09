"""Pinned-monitor position model tests for OverlayWindow.

The overlay is a full-monitor transparent click-through strip pinned to
one QScreen; the subtitle stack is placed inside it by percent offsets
(anchor model). These tests pin the anchor arithmetic, the missing-screen
fallback, the sample-subtitle API, and the persistence callback.
"""
from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from yamabiko.overlay_app.overlay_window import OverlayWindow
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeScreen:
    """Minimal QScreen stand-in with DIVERGENT geometry / availableGeometry.

    OverlayWindow._active_screen returns the object stored in _pinned_screen,
    so injecting one here lets the anchor-basis tests be platform-independent:
    on an offscreen/headless platform the real screen has no taskbar, so
    geometry() == availableGeometry() and a "used the wrong basis" defect could
    not go red. A fake whose two rects differ forces the divergence regardless
    of the host display.
    """

    def __init__(self, geom: QRect, avail: QRect):
        self._geom = geom
        self._avail = avail

    def geometry(self) -> QRect:
        return self._geom

    def availableGeometry(self) -> QRect:
        return self._avail


class _DeadScreen:
    """A QScreen stand-in that mimics a C++ object Qt has already deleted:
    every attribute access raises the libshiboken RuntimeError.

    Reproduces the dangling _pinned_screen a display change (monitor
    sleep/wake, in-place DPI/resolution change, RDP reconnect) can leave
    behind WITHOUT emitting screenAdded/screenRemoved/primaryScreenChanged
    -- so _revalidate_screen never fires and the cache is never refreshed.
    A live QScreen (or a duck-typed _FakeScreen) answers .geometry()
    normally; this one raises, which is exactly what _screen_is_live keys on.
    """
    _MSG = "Internal C++ object (PySide6.QtGui.QScreen) already deleted."

    def geometry(self) -> QRect:
        raise RuntimeError(self._MSG)

    def availableGeometry(self) -> QRect:
        raise RuntimeError(self._MSG)

    def name(self) -> str:
        raise RuntimeError(self._MSG)

    def __getattr__(self, name):
        # Everything else too (geometryChanged, signals, ...): a deleted
        # QScreen raises on ANY attribute access, so the stand-in must not
        # quietly answer AttributeError where shiboken raises RuntimeError.
        raise RuntimeError(self._MSG)


def _newest_line_rect(w, jp="字幕", en="subtitle"):
    """Add a subtitle and return the newest line's window-local target box
    (QRect) from the stack frame. The box is placed from the anchor the
    window last handed the stack, so it is the OBSERVABLE effect a dropped
    reposition would leave stale -- unlike _stack_anchor(), which recomputes
    live from self._position and so cannot detect a missing reposition."""
    w.add_subtitle(jp, en)
    return w.stack.frame().lines[-1].rect


def test_window_spans_screen_and_stack_anchor_hits_percent_model(qapp):
    """The subtitle stack is placed at the configured anchor: in Center
    alignment the newest line's bottom edge sits at v_offset_pct of screen
    height, its box centred horizontally at h_offset_pct of screen width.
    Both the _stack_anchor math AND the placed line box are asserted (the box
    proves the stack consumed the anchor).

    Basis note (W4 P0-4): the anchor percentages are measured against the FULL
    screen.geometry() rect, NOT availableGeometry(), matching resolve_drag,
    geometry_to_position and the tray Position presets.

    Uses an ASYMMETRIC h vs v (30%% / 60%%) on purpose: at the symmetric 50%%
    centre a horizontal/vertical swap would be invisible.

    Seeded-defect check: compute anchor_x from v_offset_pct (or bottom_y from
    h_offset_pct) in _stack_anchor -> the anchor coord and the placed line box
    both miss -> red. Skip the set_anchor call in _reposition_stack -> the
    line sits at the stack's origin anchor, its bottom != anchor bottom -> red.
    """
    w = OverlayWindow(style={**STYLE_DEFAULTS, "alignment": "center"},
                      position={"screen": "primary", "v_offset_pct": 60.0,
                                "h_offset_pct": 30.0})
    try:
        sg = w._active_screen().geometry()
        center_x, bottom_y, cap = w._stack_anchor()
        assert center_x == int(sg.width() * 30.0 / 100), (
            f"anchor centre {center_x} != h-offset px {int(sg.width()*30.0/100)}"
        )
        assert bottom_y == int(sg.height() * 60.0 / 100), (
            f"anchor bottom {bottom_y} != v-offset px {int(sg.height()*60.0/100)}"
        )
        assert cap == w._cap_px()
        g = _newest_line_rect(w)
        assert g.y() + g.height() == bottom_y, (
            f"newest line bottom {g.y()+g.height()} != anchor bottom {bottom_y}"
        )
        assert abs((g.x() + g.width() // 2) - center_x) <= 1, (
            f"line centre {g.x()+g.width()//2} != anchor centre {center_x}"
        )
    finally:
        w.close()


def test_stack_bottom_tracks_v_offset_not_h_offset(qapp):
    """The stack's BOTTOM anchor must be derived from v_offset_pct, not
    h_offset_pct. Uses v != h so a copy/paste that fed h_offset_pct into the
    vertical term lands on the wrong pixel, in BOTH the anchor tuple and the
    placed line.

    Seeded-defect check: compute bottom_y from h_offset_pct in _stack_anchor ->
    the anchor bottom (and the placed line's bottom) use 30%% instead of 80%%
    -> red.
    """
    w = OverlayWindow(style=None, position={"screen": "primary",
                      "v_offset_pct": 80.0, "h_offset_pct": 30.0})
    try:
        sg = w._active_screen().geometry()
        _, bottom_y, _ = w._stack_anchor()
        assert bottom_y == int(sg.height() * 80.0 / 100)
        g = _newest_line_rect(w)
        assert g.y() + g.height() == bottom_y
    finally:
        w.close()


def test_sample_survives_trim_and_has_no_hold(qapp):
    """show_sample adds a persistent sample line: it lands at once with no
    hold timer, and max_lines never evicts it even though it is the OLDEST
    line in the stack.

    Seeded-defect check: add the sample with a hold_ms (non-persistent) ->
    the stack trims it as the oldest line once max_lines is exceeded ->
    `w._sample_id in ids` red.
    """
    w = OverlayWindow(style=None, position=None)
    try:
        w.show_sample("サンプル", "Sample")
        for i in range(5):
            w.add_subtitle(f"jp{i}", f"en{i}")
        ids = [box.id for box in w.stack.frame().lines]
        assert w._sample_id in ids, "sample was trimmed despite being persistent"
        # max_lines regular lines survive, plus the untrimmed sample.
        assert len(ids) == w._style["max_lines"] + 1
    finally:
        w.close()


def test_hide_sample_removes_it(qapp):
    """hide_sample removes the sample line and clears the reference so a later
    show_sample starts fresh.

    Seeded-defect check: make hide_sample a no-op -> the sample stays in the
    stack frame and w._sample_id stays set -> red.
    """
    w = OverlayWindow(style=None, position=None)
    try:
        w.show_sample("あ", "a")
        assert w._sample_id is not None
        sample_id = w._sample_id
        w.hide_sample()
        assert w._sample_id is None
        assert sample_id not in [box.id for box in w.stack.frame().lines]
    finally:
        w.close()


def test_missing_screen_falls_back_to_primary(qapp):
    """An unknown pinned-screen name must fall back to the primary screen
    (its geometry less the bottom row), not crash and not leave a stray
    small window.

    Seeded-defect check: change _pin_to_screen's `screen is None` fallback to
    `return` instead of primaryScreen() -> the window keeps its default
    constructed geometry instead of spanning primary -> red.
    """
    w = OverlayWindow(style=None, position={"screen": "NOPE-9",
                      "v_offset_pct": 88.0, "h_offset_pct": 50.0})
    try:
        primary = QGuiApplication.primaryScreen()
        assert w.screen() is not None  # pinned to primary, not crashed
        assert w.geometry() == primary.geometry().adjusted(0, 0, 0, -1), (
            f"window {w.geometry().getRect()} did not span primary "
            f"{primary.geometry().getRect()}"
        )
    finally:
        w.close()


def test_set_position_repins_remargins_and_persists(qapp):
    """set_position updates the stored position, recomputes the anchor
    margins, and fires the persistence callback with the new position.

    Seeded-defect check: drop the _reposition_stack() call from set_position
    -> the bottom margin stays at the default 88%% anchor -> red; drop the
    callback invocation -> captured empty -> red.
    """
    captured: list[dict] = []
    w = OverlayWindow(style=None, position=None,
                      on_position_changed=captured.append)
    try:
        # A live line placed at the construction-time (88%%) anchor; set_position
        # must re-place it at the new 70%% anchor -- that move is the observable
        # proof the reposition ran.
        w.add_subtitle("字幕", "subtitle")
        QApplication.processEvents()
        w.set_position({"screen": "primary", "v_offset_pct": 70.0,
                        "h_offset_pct": 40.0})
        QApplication.processEvents()
        assert w.current_position() == {
            "screen": "primary", "v_offset_pct": 70.0, "h_offset_pct": 40.0,
            "left_offset_pct": 6.0,
        }
        assert captured and captured[-1]["v_offset_pct"] == 70.0, (
            f"on_position_changed not fired with the new position: {captured}"
        )
        sg = w._active_screen().geometry()
        _, bottom_y, _ = w._stack_anchor()
        assert bottom_y == int(sg.height() * 70.0 / 100)
        g = w.stack.frame().lines[-1].rect
        assert g.y() + g.height() == bottom_y, (
            f"line did not re-place to the new anchor: bottom {g.y()+g.height()}"
            f" != {bottom_y}"
        )
    finally:
        w.close()


def test_update_position_preview_repins_without_firing_persist_callback(qapp):
    """update_position_preview re-pins + re-margins the stack to the given
    position but must NOT fire on_position_changed. Placement mode's live
    drag calls it on every mouse move; routing through set_position instead
    would write user_overrides.yaml (and grab the cross-process filelock)
    on every pixel of the drag.

    Seeded-defect check: make update_position_preview call
    self._on_position_changed -> captured becomes non-empty -> red; drop
    the _reposition_stack() call -> the bottom margin stays at the default
    88%% anchor -> red.
    """
    captured: list[dict] = []
    w = OverlayWindow(style=None, position=None,
                      on_position_changed=captured.append)
    try:
        w.add_subtitle("字幕", "subtitle")
        QApplication.processEvents()
        w.update_position_preview({"screen": "primary", "v_offset_pct": 72.0,
                                   "h_offset_pct": 40.0})
        QApplication.processEvents()
        assert w.current_position() == {
            "screen": "primary", "v_offset_pct": 72.0, "h_offset_pct": 40.0,
            "left_offset_pct": 6.0,
        }
        sg = w._active_screen().geometry()
        _, bottom_y, _ = w._stack_anchor()
        assert bottom_y == int(sg.height() * 72.0 / 100)
        g = w.stack.frame().lines[-1].rect
        assert g.y() + g.height() == bottom_y, (
            f"preview did not re-place the line: bottom {g.y()+g.height()} "
            f"!= {bottom_y}"
        )
        assert captured == [], (
            "update_position_preview fired the persist callback; the live "
            "drag would spam user_overrides.yaml on every mouse move"
        )
    finally:
        w.close()


def test_set_position_callback_failure_is_swallowed_and_logged(qapp, caplog):
    """The persist callback routes through config_writer._atomic_write, which
    re-raises OSError (disk full, AV lock). set_position must contain the
    failure with an exception-level .geom log record rather than letting it
    escape into the Qt slot dispatcher (placement mode's drag handler).

    Seeded-defect check: remove the try/except around the callback in
    set_position -> the OSError propagates out of set_position -> this test
    errors instead of passing.
    """
    def _boom(_pos: dict) -> None:
        raise OSError("disk full")

    w = OverlayWindow(style=None, position=None, on_position_changed=_boom)
    try:
        with caplog.at_level(logging.ERROR):
            w.set_position({"screen": "primary", "v_offset_pct": 70.0,
                            "h_offset_pct": 40.0})  # must not raise
        geom_errors = [
            r for r in caplog.records
            if r.name.endswith(".geom") and r.exc_info is not None
        ]
        assert geom_errors, (
            "raising position callback was swallowed without an "
            "exception-level log record"
        )
    finally:
        w.close()


def test_revalidate_screen_repins_after_layout_change(qapp):
    """_revalidate_screen (wired to Qt screen-change signals) re-pins the
    window to its resolved screen. If the window drifted (or its pinned
    screen vanished), re-pinning restores the full-monitor strip.

    Seeded-defect check: make _revalidate_screen a no-op -> the drifted
    geometry set below survives -> red.
    """
    w = OverlayWindow(style=None, position={"screen": "primary",
                      "v_offset_pct": 88.0, "h_offset_pct": 50.0})
    try:
        primary = QGuiApplication.primaryScreen()
        # Simulate drift, then a hotplug re-eval.
        w.setGeometry(5, 5, 120, 90)
        w._revalidate_screen()
        assert w.geometry() == primary.geometry().adjusted(0, 0, 0, -1), (
            "revalidate did not re-pin the window to the primary screen"
        )
    finally:
        w.close()


def test_stack_anchor_realizes_percent_model_on_full_geometry(qapp):
    """Round-trip proof (W4 P0-4): _stack_anchor must realize EXACTLY the
    center_x / bottom_y the percent model implies from the pinned screen's FULL
    geometry -- the same basis resolve_drag uses to turn a drag into offsets. If
    _stack_anchor used availableGeometry, the stack would render offset from
    where the cursor was dragged by the taskbar thickness.

    Platform-independent by construction: we inject a fake pinned screen whose
    geometry() and availableGeometry() DIVERGE (a 60px side + 48px bottom
    taskbar), so the seeded defect goes red even on an offscreen/headless
    platform where the real screen has no taskbar (geometry == availableGeometry
    there, and a wrong-basis defect would be invisible).

    Seeded-defect check: put availableGeometry() into the height term of
    _stack_anchor -> bottom_y uses 1392 instead of 1440 -> red; into the width
    term -> center_x/cap use 2500 instead of 2560 -> red.
    """
    geom = QRect(0, 0, 2560, 1440)
    avail = QRect(0, 0, 2500, 1392)   # 60px left + 48px bottom taskbar
    v, h = 62.5, 50.0
    w = OverlayWindow(style={**STYLE_DEFAULTS, "alignment": "center"},
                      position={"screen": "primary",
                                "v_offset_pct": v, "h_offset_pct": h})
    try:
        w._pinned_screen = _FakeScreen(geom, avail)
        center_x, bottom_y, cap = w._stack_anchor()

        assert bottom_y == int(geom.height() * v / 100), (          # 900
            f"anchor bottom {bottom_y} != full-geometry y {int(geom.height()*v/100)}"
        )
        assert center_x == int(geom.width() * h / 100), (           # 1280
            f"anchor centre {center_x} != full-geometry x {int(geom.width()*h/100)}"
        )
        assert cap == int(geom.width() * 70 / 100), (               # 1792
            f"cap {cap} != full-geometry cap {int(geom.width()*70/100)}"
        )

        # Divergence guard: prove the two bases actually differ here, so the
        # asserts above genuinely distinguish full geometry from availableGeometry
        # (otherwise the test would be toothless on a taskbar-less display).
        assert int(geom.height() * v / 100) != int(avail.height() * v / 100), (
            "height bases coincide; toothless"
        )
        assert int(geom.width() * 70 / 100) != int(avail.width() * 70 / 100), (
            "width bases coincide; toothless"
        )
    finally:
        w.close()


def test_off_centre_anchor_caps_lines_to_the_margin_box(qapp):
    """S1/P1 regression (reproduced live), Center alignment.

    At an off-centre anchor the clamped margins leave a box NARROWER than the
    raw _cap_px(); a line sized to the raw cap and centred on the raw anchor
    hangs off the monitor. Canonical repro: 1920 px screen, h_offset 90%,
    max_width_pct 70 -> raw cap 1344 (center_x 1728) but the margin box is only
    864 px (left 1056 / right 0). Before dev's 60c87bf a long line took the raw
    1344 px and spanned 1056..2400 -- ~480 px clipped off the right edge.

    The stack must (a) cap the line at the margin-box-narrowed _line_cap_px()
    and (b) place the box FULLY on-screen: a long line's box is at most 864 px
    wide and, centred on 1728, slides inward to end flush at the 1920 edge.

    Seeded-defect check: hand the stack the raw _cap_px() instead of
    _line_cap_px() in _reposition_stack -> the long line grows past 864 px ->
    red. Drop the x-clamp in place_lines -> the box centres at 1728 and runs
    off the 1920 edge -> red.
    """
    geom = QRect(0, 0, 1920, 1080)
    w = OverlayWindow(style={**STYLE_DEFAULTS, "max_width_pct": 70,
                             "alignment": "center"},
                      position={"screen": "primary", "v_offset_pct": 85.0,
                                "h_offset_pct": 90.0})
    try:
        w._pinned_screen = _FakeScreen(geom, geom)
        # A screen change always arrives through a re-pin, which re-places.
        w._reposition_stack()
        # Margin-box math (dev's _stack_geometry, ported): raw cap 1344,
        # left 1056 / right 0 -> box 864, so the effective per-line cap narrows.
        assert w._cap_px() == 1344, f"raw cap {w._cap_px()} != 1344"
        assert w._line_cap_px() == 864, (
            f"margin-box cap {w._line_cap_px()} != 864 (dev's 60c87bf math)"
        )
        # Long JP (many wrappable segments) + long EN so the box fills the cap.
        w.add_subtitle("漢字" * 60, "long english subtitle " * 12,
                       jp_ruby=[{"text": "漢字"}] * 60)
        g = w.stack.frame().lines[-1].rect
        # (a) the line is capped at the margin-box cap, never the raw cap.
        assert g.width() <= w._line_cap_px(), (
            f"line box {g.width()} wider than _line_cap_px() {w._line_cap_px()}"
        )
        # (b) the placed box is FULLY on-screen -- the crux of the P1 fix --
        # and slid inward flush to the right edge.
        assert g.x() >= 0 and g.x() + g.width() == geom.width(), (
            f"box {g.x()}..{g.x()+g.width()} not flush inside the "
            f"{geom.width()}px screen at h_offset 90%"
        )
        # The re-place path must agree with the add path.
        w._reposition_stack()
        g2 = w.stack.frame().lines[-1].rect
        assert g2.x() >= 0 and g2.x() + g2.width() <= geom.width(), (
            f"re-placed box {g2.x()}..{g2.x()+g2.width()} off-screen"
        )
    finally:
        w.close()


def test_stack_taller_than_space_above_top_anchor_keeps_lines_and_banner_on_screen(qapp):
    """At a top anchor (clamp_position allows v_offset_pct down to 5) a stack
    growing upward from bottom_y with no vertical clamp lands its older lines
    at negative y, off the monitor. The stack must start low enough that its
    topmost line sits at or below the top of the screen's available geometry
    (here a 40 px top taskbar). The banner rises from the newest line's
    bottom (a series name taller than a short line) or, paused with no
    lines, from the anchor: it must start on screen too.

    Seeded defects: drop the max(bottom_y, top + stack height) clamp in
    place_lines -> the oldest line's y goes negative -> red. Clamp with the
    line heights only, or place the paused-empty banner at bottom_y -
    natural -> the banner's top goes above the taskbar -> red.
    """
    geom = QRect(0, 0, 1920, 1080)
    avail = QRect(0, 40, 1920, 1040)   # taskbar docked at the top
    w = OverlayWindow(style={**STYLE_DEFAULTS, "max_lines": 3,
                             "alignment": "center"},
                      position={"screen": "primary", "v_offset_pct": 5.0,
                                "h_offset_pct": 50.0})
    try:
        w._pinned_screen = _FakeScreen(geom, avail)
        # A screen change always arrives through a re-pin, which re-places.
        w._reposition_stack()
        _, bottom_y, _ = w._stack_anchor()
        for _ in range(3):
            w.add_subtitle("字幕" * 10, "a fairly long english subtitle " * 3)
        rects = [box.rect for box in w.stack.frame().lines]
        assert len(rects) == 3
        stack_height = sum(r.height() for r in rects)
        assert stack_height > bottom_y - avail.top(), (
            "precondition: the stack must be taller than the space above "
            "the anchor, or the clamp is never exercised"
        )
        tops = [r.y() for r in rects]
        assert min(tops) >= avail.top(), (
            f"line tops {tops} rise above the available-geometry top "
            f"{avail.top()} at a top anchor"
        )

        for box in w.stack.frame().lines:
            w.stack.remove_line(box.id, fade=False)
        w.set_state("variety", True)
        banner = w.stack.frame().banner
        assert banner.height() > bottom_y - avail.top(), (
            "precondition: the paused banner must be taller than the space "
            "above the anchor"
        )
        assert banner.top() >= avail.top(), (
            f"paused banner with no lines starts at {banner.top()}, above "
            f"the available-geometry top {avail.top()}"
        )

        w.set_state("variety", False)
        w.set_series(1, "Frieren", "葬送のフリーレン")
        w.add_subtitle("はい。", "Yes.")
        frame = w.stack.frame()
        assert frame.banner.height() > frame.lines[-1].rect.height(), (
            "precondition: the series banner must be taller than the line"
        )
        assert frame.banner.top() >= avail.top(), (
            f"series banner starts at {frame.banner.top()}, above the "
            f"available-geometry top {avail.top()}"
        )
    finally:
        w.close()



def test_update_position_preview_skips_repin_on_same_screen(qapp, monkeypatch):
    """K18: placement drag fires update_position_preview on every mouse move;
    the target screen almost never changes mid-drag, so the _pin_to_screen
    linear QScreen scan + full-screen setGeometry must run only when the
    requested screen differs from the one already pinned. Previewing on the
    construction screen re-pins zero times; two consecutive previews naming
    a NEW screen re-pin exactly once; the skipped calls still re-margin.

    Seeded-defect check: revert update_position_preview to an unconditional
    self._pin_to_screen() -> all three previews re-pin -> pins reads
    ['primary', 'DISPLAY-X', 'DISPLAY-X'] instead of ['DISPLAY-X'] -> red.
    """
    w = OverlayWindow(style=None, position=None)  # pinned for "primary"
    try:
        # A live line so the "still re-places" half is OBSERVABLE (the manual
        # model has no contents-margins; the placed line's bottom is the anchor).
        w.add_subtitle("字幕", "subtitle")
        QApplication.processEvents()
        pins: list[str] = []
        orig = OverlayWindow._pin_to_screen

        def spy(self):
            pins.append(self._position.get("screen"))
            orig(self)

        monkeypatch.setattr(OverlayWindow, "_pin_to_screen", spy)

        # Same screen as the construction pin -> no re-pin.
        w.update_position_preview({"screen": "primary", "v_offset_pct": 40.0,
                                   "h_offset_pct": 50.0})
        assert pins == [], f"same-screen preview re-pinned: {pins}"

        # A new screen NAME (unknown -> _pin_to_screen falls back to primary):
        # the first preview re-pins, the second sees the same name and skips.
        w.update_position_preview({"screen": "DISPLAY-X", "v_offset_pct": 50.0,
                                   "h_offset_pct": 50.0})
        w.update_position_preview({"screen": "DISPLAY-X", "v_offset_pct": 60.0,
                                   "h_offset_pct": 50.0})
        assert pins == ["DISPLAY-X"], (
            f"screen-change previews must re-pin exactly once, got {pins}"
        )

        # The skipped second call still re-placed the stack to v=60: the live
        # line's bottom edge tracks the new v-offset.
        sg = w._active_screen().geometry()
        expected_bottom = int(sg.height() * 60.0 / 100)
        g = w.stack.frame().lines[-1].rect
        assert g.y() + g.height() == expected_bottom, (
            f"skipped-repin preview did not re-place the stack: line bottom "
            f"{g.y() + g.height()} != v=60 px {expected_bottom}"
        )
    finally:
        w.close()


def test_cap_px_uses_pinned_screen_full_geometry_before_show(qapp):
    """Pre-show correctness (W4 P2 boot-screen): _cap_px must derive the cap
    from the STORED pinned QScreen's full geometry() BEFORE the window is ever
    shown -- not from self.screen() (which can report the primary rather than
    the pinned screen pre-show) and not from availableGeometry().

    The fake pinned screen's geometry width (4321) differs from its own
    availableGeometry width (4200) AND from any real monitor, so:
      - if _cap_px read availableGeometry(), the cap would be 4200-based -> red
      - if _cap_px read self.screen(), the cap would track the real primary
        screen's width (never 4321) -> red

    Seeded-defect check: revert _cap_px to self.screen().availableGeometry()
    -> the returned cap no longer equals int(4321 * pct / 100) -> red.
    """
    w = OverlayWindow(style=None, position=None)   # constructed, never shown
    try:
        assert not w.isVisible()
        w._pinned_screen = _FakeScreen(QRect(0, 0, 4321, 2765),
                                       QRect(0, 0, 4200, 2700))
        pct = w._style.get("max_width_pct", 70)
        assert w._cap_px() == int(4321 * pct / 100), (
            f"_cap_px()={w._cap_px()} did not use the pinned screen's full "
            f"geometry width 4321 (pct={pct})"
        )
        assert w._cap_px() != int(4200 * pct / 100), (
            "_cap_px used availableGeometry width, not full geometry"
        )
    finally:
        w.close()


def test_show_event_recomputes_anchor_against_pinned_screen(qapp):
    """showEvent must re-run _reposition_stack so first-paint geometry is
    computed against the ACTUAL pinned screen (W4 P2 boot-screen self-heal:
    Qt may only finalize the screen association after show).

    A live line is placed at the construction-time anchor; we then mutate
    _position directly (bypassing set_position, which would reposition on its
    own) and swap in a divergent fake pinned screen, then show(): if showEvent
    recomputes, the line re-places to the new anchor; if not, it stays frozen.
    Asserting the placed line's bottom (not just _stack_anchor, which recomputes
    live from _position) is what makes a dropped reposition observable.

    Seeded-defect check: delete the self._reposition_stack() call from showEvent
    -> the line stays at the construction-time anchor bottom -> red.
    """
    w = OverlayWindow(style=None, position=None)  # default v=88 on offscreen primary
    try:
        w.add_subtitle("字幕", "subtitle")
        before = w.stack.frame().lines[-1].rect
        before_bottom = before.y() + before.height()
        # A target the construction-time anchor pass never saw.
        w._position = {"screen": "primary", "v_offset_pct": 60.0,
                       "h_offset_pct": 50.0, "left_offset_pct": 6.0}
        geom = QRect(0, 0, 2560, 1440)
        w._pinned_screen = _FakeScreen(geom, QRect(0, 0, 2560, 1392))
        w.show()
        QApplication.processEvents()
        expected_bottom = int(geom.height() * 60.0 / 100)  # 864
        g = w.stack.frame().lines[-1].rect
        assert g.y() + g.height() == expected_bottom, (
            f"showEvent did not recompute the anchor: line bottom "
            f"{g.y()+g.height()} != {expected_bottom}"
        )
        assert expected_bottom != before_bottom, "test setup failed to change the target"
    finally:
        w.close()


# ---------------------------------------------------------------------------
# Same-monitor resolution change (P1-6). A game flipping the pinned monitor's
# display mode updates QScreen.geometry() IN PLACE and emits ONLY that
# QScreen's geometryChanged -- none of the three app-level hotplug signals
# __main__ wires to _revalidate_screen. Without a per-screen connection the
# window keeps its old size while the anchor math reads the NEW screen rect,
# so subtitles are laid out outside the window (e.g. bottom_y 1267 on a
# still-1080-high window after a 1440p switch).
# ---------------------------------------------------------------------------


def _scramble_window_and_line(w) -> None:
    """Put the window + stack anchor into the stale state a resolution change
    leaves behind, so a re-pin/reposition has something observable to fix."""
    w.setGeometry(7, 7, 320, 200)
    w.stack.set_anchor(1, 1, 10, 10, 0, 10)


def test_pinned_screen_geometry_changed_signal_repins_window_and_stack(qapp):
    """Emitting geometryChanged on the PINNED QScreen (the offscreen platform
    cannot actually switch modes) must restore the full-monitor strip AND
    re-place the subtitle stack at the anchor -- end-to-end through the
    connection _pin_to_screen makes, not by calling the handler by hand.

    Seeded defect: drop the geometryChanged.connect from _pin_to_screen ->
    the scrambled window/line geometry survives the emit -> red.
    """
    w = OverlayWindow(style=None, position={"screen": "primary",
                      "v_offset_pct": 88.0, "h_offset_pct": 50.0})
    try:
        primary = QGuiApplication.primaryScreen()
        w.add_subtitle("字幕", "subtitle")
        QApplication.processEvents()
        _scramble_window_and_line(w)

        primary.geometryChanged.emit(primary.geometry())
        QApplication.processEvents()

        assert w.geometry() == primary.geometry().adjusted(0, 0, 0, -1), (
            "geometryChanged on the pinned screen did not re-apply the "
            "pinned screen's geometry -- the window keeps its old size after "
            "a same-monitor resolution change"
        )
        expected_bottom = int(primary.geometry().height() * 88.0 / 100)
        g = w.stack.frame().lines[-1].rect
        assert g.y() + g.height() == expected_bottom, (
            f"stack not re-placed after the geometry change: line bottom "
            f"{g.y() + g.height()} != anchor {expected_bottom}"
        )
    finally:
        w.close()


def test_on_pinned_screen_geometry_changed_handler_restores_geometry_and_anchor(qapp):
    """Charter §3.9 (wired-not-exercised): invoke the handler DIRECTLY and
    assert on its side effects -- window geometry re-applied from the pinned
    screen and the stack re-anchored -- so a no-op handler body cannot hide
    behind a correct-looking connect call.

    Seeded defect: make _on_pinned_screen_geometry_changed a no-op (`pass`)
    -> the scrambled geometry survives the direct call -> red.
    """
    w = OverlayWindow(style=None, position={"screen": "primary",
                      "v_offset_pct": 60.0, "h_offset_pct": 50.0})
    try:
        primary = QGuiApplication.primaryScreen()
        w.add_subtitle("字幕", "subtitle")
        QApplication.processEvents()
        _scramble_window_and_line(w)

        w._on_pinned_screen_geometry_changed()
        QApplication.processEvents()

        assert w.geometry() == primary.geometry().adjusted(0, 0, 0, -1), (
            "handler did not re-apply the pinned screen's geometry"
        )
        expected_bottom = int(primary.geometry().height() * 60.0 / 100)
        g = w.stack.frame().lines[-1].rect
        assert g.y() + g.height() == expected_bottom, (
            f"handler did not trigger the relayout path: line bottom "
            f"{g.y() + g.height()} != anchor {expected_bottom}"
        )
    finally:
        w.close()


def test_repeated_repins_do_not_accumulate_geometry_signal_connections(
    qapp, monkeypatch,
):
    """Every _pin_to_screen must leave exactly ONE geometryChanged
    connection on the pinned screen: it disconnects the previous pin's
    connection before connecting again. _revalidate_screen re-pins on every
    hotplug signal, so a leaked connection per re-pin would fire the
    handler N times per resolution change (N growing for the process
    lifetime).

    The count comes from an instance-level wrapper on THIS window's
    _revalidate_screen (the handler delegates to it once per delivery),
    installed after the re-pins. Earlier tests leave closed-but-alive
    windows connected to the shared primary QScreen, and the emit below
    reaches them too: a class-level patch of the handler would make their
    re-pins disconnect a slot they never connected (a "Failed to
    disconnect" RuntimeWarning per window) and stack connections.

    Seeded defect: drop the geometryChanged.disconnect from _pin_to_screen
    -> three pins leave three connections -> one emit fires the handler
    three times -> red.
    """
    w = OverlayWindow(style=None, position={"screen": "primary",
                      "v_offset_pct": 88.0, "h_offset_pct": 50.0})
    try:
        # Two hotplug-style re-pins on top of the construction pin.
        w._revalidate_screen()
        w._revalidate_screen()

        calls: list[int] = []
        revalidate = w._revalidate_screen

        def counting() -> None:
            calls.append(1)
            revalidate()

        monkeypatch.setattr(w, "_revalidate_screen", counting)
        primary = QGuiApplication.primaryScreen()
        primary.geometryChanged.emit(primary.geometry())
        QApplication.processEvents()

        assert len(calls) == 1, (
            f"one geometry change fired the handler {len(calls)} times -- "
            f"re-pins are leaking signal connections"
        )
    finally:
        w.close()


def test_pin_to_screen_survives_dead_previous_pinned_screen(qapp):
    """Re-pinning over a pin whose QScreen C++ object Qt already deleted
    (the dangling-pin display change) must not raise: the disconnect from
    the previous screen hits the dead object's RuntimeError and is guarded
    with the same idiom as the _screen_is_live probe.

    Seeded defect: remove the try/except around the previous-screen
    geometryChanged.disconnect in _pin_to_screen -> _DeadScreen's
    RuntimeError escapes -> red (error).
    """
    w = OverlayWindow(style=None, position=None)
    try:
        w._pinned_screen = _DeadScreen()
        w._pin_to_screen()  # must not raise
        assert w._pinned_screen is QGuiApplication.primaryScreen(), (
            "re-pin over the dead screen did not land on a live screen"
        )
    finally:
        w.close()


def test_add_subtitle_self_heals_dead_pinned_screen(qapp):
    """A subtitle arriving after the cached QScreen was deleted must re-pin
    the window to a live screen, restore its full-monitor geometry and land
    at that screen's anchor -- not at the dead screen's stale anchor.

    Root cause (2026-07-07 GitS session): a display change deleted the
    overlay's cached _pinned_screen WITHOUT emitting any of the three
    app-level signals _revalidate_screen is wired to (primaryScreenChanged /
    screenAdded / screenRemoved), so nothing re-pinned. Every add_subtitle
    then dereferenced the dead QScreen and threw 'libshiboken: Internal C++
    object (QScreen) already deleted' -- 72 times in a row, one dropped
    subtitle each, while the pipeline stayed healthy. The add path no longer
    reads the screen, so without its own heal the window kept the dead
    screen's geometry and the stack its stale anchor.

    Seeded-defect check: drop the ``self._heal_dead_pin()`` call from
    add_subtitle -> the dead pin survives the add, the scrambled window and
    line stay put -> the ``is primary`` assertion red.
    """
    primary = QGuiApplication.primaryScreen()
    w = OverlayWindow(style=None, position={"screen": "primary",
                      "v_offset_pct": 88.0, "h_offset_pct": 50.0})
    try:
        # The stale state the dead screen leaves: its geometry and anchor,
        # then the dangling reference itself.
        _scramble_window_and_line(w)
        w._pinned_screen = _DeadScreen()
        w.add_subtitle("それは不可能だな荒牧", "That's impossible, Aramaki.")
        QApplication.processEvents()

        # Asserted before any _active_screen() call, which re-pins on its own.
        assert w._pinned_screen is primary, (
            "add_subtitle did not re-pin the dead screen to a live one"
        )
        assert w.geometry() == primary.geometry().adjusted(0, 0, 0, -1), (
            "the window kept the dead screen's geometry"
        )
        sg = primary.geometry()
        g = w.stack.frame().lines[-1].rect
        assert (g.x(), g.y() + g.height()) == (
            int(sg.width() * 6.0 / 100), int(sg.height() * 88.0 / 100)
        ), f"the subtitle landed at the stale anchor: {g}"
    finally:
        w.close()


def test_left_alignment_anchors_stack_at_left_offset(qapp):
    """Left alignment: the newest line starts at left_offset_pct of the
    screen width (h_offset_pct is the Center anchor and is ignored);
    switching to Center centres it on h_offset_pct.

    Seeded defect: anchor Left on h_offset_pct -> line starts at 50% -> red."""
    pos = {"screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
           "left_offset_pct": 10.0}
    w = OverlayWindow(style=dict(STYLE_DEFAULTS), position=pos)
    try:
        sw = w._active_screen().geometry().width()
        w.add_subtitle("字幕", "Subtitle")
        newest = w.stack.frame().lines[-1]
        assert newest.rect.x() == int(sw * 0.10)
        w.apply_style({**STYLE_DEFAULTS, "alignment": "center"})
        newest = w.stack.frame().lines[-1]
        assert abs(newest.rect.center().x() - sw * 0.5) <= 1
    finally:
        w.close()
