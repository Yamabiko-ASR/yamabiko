"""Placement-mode tests: pure drag math + the controller state machine.

`resolve_drag` is the pure cursor->anchor-position math (fake screens, no
Qt). `PlacementController` drives a real offscreen OverlayWindow: enter
enables input + shows the sample and hint, a drag previews the position
WITHOUT persisting, Enter / click-away saves once, Esc restores the entry
position without persisting, and the controller can re-enter after a save.

Each test names the one-line production defect that turns it red.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from yamabiko.overlay_app import placement as placement_mod
from yamabiko.overlay_app.placement import PlacementController, resolve_drag
from yamabiko.overlay_app.overlay_window import OverlayWindow
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from yamabiko.overlay_app.theme import MIST, NIGHT


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeMouseEvent:
    """The window's mouse handlers ignore the event coords (the controller
    reads the global cursor); they only call accept(). A minimal fake keeps
    the delegation test off the fragile QMouseEvent constructor."""

    def __init__(self):
        self.accepted = False

    def accept(self):
        self.accepted = True


class _FakeKeyEvent:
    def __init__(self, key):
        self._key = key
        self.accepted = False

    def key(self):
        return self._key

    def accept(self):
        self.accepted = True


def _screens_snapshot():
    return [
        (s.name(), (s.geometry().x(), s.geometry().y(),
                    s.geometry().width(), s.geometry().height()))
        for s in QGuiApplication.screens()
    ]


# --------------------------------------------------------------------------
# Pure resolve_drag math (no Qt, fake screens)
# --------------------------------------------------------------------------


def test_resolve_drag_snaps_near_center_h_to_exactly_50():
    """A cursor within +-2% of the screen's horizontal centre snaps h to
    50.0 (the common centred case); v is the cursor y as a percent.

    Seeded defect: replace the `* 100.0` percent scale with `* 1000.0`
    (off-by-10x) -> 520/1000 no longer resolves to 52% near centre ->
    the snapped 50.0 result changes -> red.
    """
    screens = [("S1", (0, 0, 1000, 1000))]
    assert resolve_drag(520, 300, screens) == {
        "screen": "S1", "v_offset_pct": 30.0, "h_offset_pct": 50.0,
        "left_offset_pct": 6.0,
    }


def test_resolve_drag_leaves_h_beyond_snap_window_unsnapped():
    """A cursor well off centre keeps its true horizontal percent; only the
    narrow +-2% window snaps.

    Seeded defect: widen the snap window 10x (`<= _H_SNAP_PCT` ->
    `<= _H_SNAP_PCT * 10`) -> 60% (|60-50|=10) wrongly snaps to 50.0 ->
    red.
    """
    screens = [("S1", (0, 0, 1000, 1000))]
    assert resolve_drag(600, 300, screens)["h_offset_pct"] == 60.0
    assert resolve_drag(200, 300, screens)["h_offset_pct"] == 20.0


def test_resolve_drag_picks_screen_containing_cursor():
    """The screen containing the cursor wins, so dragging across a monitor
    boundary jumps the block to that monitor and computes offsets relative
    to it.

    Seeded defect: always use screens[0] instead of the containing screen
    -> a cursor on screen 2 reports screen S1 (and h relative to S1) ->
    red on the screen-name assertion.
    """
    screens = [("S1", (0, 0, 1920, 1080)), ("S2", (1920, 0, 1920, 1080))]
    r = resolve_drag(2112, 540, screens)  # 10% into S2, vertical centre
    assert r["screen"] == "S2"
    assert r["h_offset_pct"] == 10.0
    assert r["v_offset_pct"] == 50.0


def test_resolve_drag_clamps_out_of_bounds_offsets():
    """Offsets past the anchor bounds are clamped (v 5..98, h 10..90) so a
    drag into a screen corner can't push the stack off-screen.

    Seeded defect: return the raw dict instead of routing through
    clamp_position -> h=99 / v=99.5 survive unclamped -> red.
    """
    screens = [("S1", (0, 0, 1000, 1000))]
    hi = resolve_drag(990, 995, screens)
    assert hi["h_offset_pct"] == 90.0 and hi["v_offset_pct"] == 98.0
    lo = resolve_drag(30, 20, screens)
    assert lo["h_offset_pct"] == 10.0 and lo["v_offset_pct"] == 5.0


def test_resolve_drag_empty_screens_returns_default_anchor():
    """No displays to resolve against -> the default anchor, never a crash.

    Seeded defect: index screens[0] unconditionally -> IndexError -> red.
    """
    assert resolve_drag(10, 10, []) == {
        "screen": "primary", "v_offset_pct": 88.0, "h_offset_pct": 50.0,
        "left_offset_pct": 6.0,
    }


# --------------------------------------------------------------------------
# Controller state machine (real offscreen OverlayWindow)
# --------------------------------------------------------------------------


def test_enter_enables_input_shows_sample_and_hint(qapp):
    """enter() flips the window to input-enabled, shows the persistent
    sample subtitle, and shows the hint bar.

    Seeded defect: drop the set_interactive(True) call in enter() ->
    _interactive stays False -> red; drop show_sample -> _sample_id None ->
    red; drop the hint show -> hint hidden -> red.
    """
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None)
        c.enter()
        assert c.active is True
        assert w._interactive is True, "enter did not enable input"
        assert w._sample_id is not None, "enter did not show the sample subtitle"
        assert c._hint is not None and not c._hint.isHidden(), (
            "enter did not show the hint bar"
        )
    finally:
        w.close()


def test_enter_key_saves_current_position_via_on_save_exactly_once(qapp):
    """Enter commits the placement: on_save fires exactly once with the
    window's current position, through the injected callback (not the
    window's own persist callback), then the session tears down.

    Seeded defect (the Enter/Esc-swap drill): route Return to cancel() ->
    on_save never fires -> red.
    """
    save_spy: list[dict] = []
    pos_spy: list[dict] = []
    w = OverlayWindow(on_position_changed=pos_spy.append)
    try:
        c = PlacementController(w, on_save=save_spy.append)
        c.enter()
        w.update_position_preview({"screen": "primary",
                                   "v_offset_pct": 70.0, "h_offset_pct": 40.0})
        target = w.current_position()
        w.keyPressEvent(_FakeKeyEvent(Qt.Key_Return))
        assert save_spy == [target], (
            f"Enter did not save current_position once; save_spy={save_spy}"
        )
        assert c.active is False, "save did not tear the session down"
        assert pos_spy == [], (
            "placement persisted through the window callback; it must "
            "persist only via on_save"
        )
    finally:
        w.close()


def test_escape_key_cancels_restoring_entry_position_without_persisting(qapp):
    """Esc abandons the placement: the window snaps back to the position it
    held on enter, and NOTHING is persisted (not on_save, not the window
    callback).

    Seeded defect (the Enter/Esc-swap drill): route Escape to save() ->
    on_save fires AND the moved position is not restored -> red.
    """
    save_spy: list[dict] = []
    pos_spy: list[dict] = []
    w = OverlayWindow(on_position_changed=pos_spy.append)
    try:
        c = PlacementController(w, on_save=save_spy.append)
        entry = w.current_position()
        c.enter()
        w.update_position_preview({"screen": "primary",
                                   "v_offset_pct": 70.0, "h_offset_pct": 40.0})
        assert w.current_position() != entry  # the drag actually moved it
        w.keyPressEvent(_FakeKeyEvent(Qt.Key_Escape))
        assert w.current_position() == entry, (
            "Esc did not restore the entry position"
        )
        assert save_spy == [], "cancel must not persist via on_save"
        assert pos_spy == [], "cancel must not persist via the window callback"
        assert c.active is False
    finally:
        w.close()


def test_window_deactivate_saves_placement(qapp):
    """Click-away (WindowDeactivate) commits the placement, matching the
    tray-preset one-click feel.

    Seeded defect: remove the WindowDeactivate branch from
    OverlayWindow.event() -> click-away no longer saves -> red.
    """
    save_spy: list[dict] = []
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=save_spy.append)
        c.enter()
        w.update_position_preview({"screen": "primary",
                                   "v_offset_pct": 72.0, "h_offset_pct": 50.0})
        target = w.current_position()
        w.event(QEvent(QEvent.Type.WindowDeactivate))
        assert save_spy == [target], (
            f"click-away did not save the placement; save_spy={save_spy}"
        )
        assert c.active is False
    finally:
        w.close()


def test_save_defers_declick_off_the_current_event_dispatch(qapp):
    """P2 re-entrancy: teardown must NOT run set_interactive(False) synchronously
    inside the key/deactivate handler that triggered the save. set_interactive
    calls setWindowFlags()+show(), which destroys and recreates the native HWND;
    doing that inside the window's own event dispatch is a Windows sporadic-crash
    class. The flag dance is deferred to the next event-loop turn, so right after
    the Enter-key save the window is STILL interactive (active/delegate cleared
    synchronously); one event-loop turn later it flips back to click-through.

    Seeded defect: make the deferral synchronous again (call
    ``self._window.set_interactive(False)`` in _teardown instead of
    ``QTimer.singleShot(0, self._deferred_declick)``) -> _interactive is already
    False immediately after the key event -> the still-True assertion goes red.
    """
    from PySide6.QtTest import QTest

    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None)
        c.enter()
        assert w._interactive is True, "enter did not enable input"

        # Enter commits the placement (keyPressEvent -> save -> _teardown),
        # exactly the on-the-event-dispatch path the deferral protects.
        w.keyPressEvent(_FakeKeyEvent(Qt.Key_Return))

        # Synchronous half of teardown ran: session is torn down...
        assert c.active is False, "save did not clear active synchronously"
        # ...but the HWND-recreating flag flip is deferred: still interactive.
        assert w._interactive is True, (
            "set_interactive(False) ran synchronously inside the key-event "
            "dispatch; the flag dance must defer to the next event-loop turn"
        )

        QTest.qWait(20)  # pump the singleShot(0) deferred declick
        assert w._interactive is False, (
            "the deferred declick never restored click-through after teardown"
        )
    finally:
        w.close()


class _DeclickWindow:
    """Records set_interactive calls; carries a settable ``_placement_delegate``
    so the _deferred_declick guard can be exercised in isolation (a real window
    fires spurious WindowDeactivate saves during a pumped event loop, which would
    tear a re-entered session down for a legitimate reason and mask the guard)."""

    def __init__(self, delegate):
        self._placement_delegate = delegate
        self.declicks: list[bool] = []

    def set_interactive(self, value: bool) -> None:
        self.declicks.append(value)


def test_deferred_declick_skips_when_a_new_session_reclaimed_the_window(qapp):
    """The deferred declick must NOT disable input underneath a NEW placement
    session. A fast Move -> save -> Move installs a fresh delegate on the window
    before the first teardown's singleShot(0) fires; that stale declick must
    no-op (it checks the window's placement delegate is still None) rather than
    flip the re-entered session's window back to click-through mid-drag.

    Seeded defect: drop the ``if ..._placement_delegate is not None: return``
    guard in _deferred_declick -> it calls set_interactive(False) even with a
    delegate present -> the spy records the disable -> red.
    """
    win = _DeclickWindow(delegate=object())  # a NEW session owns the window
    c = PlacementController(win, on_save=lambda p: None)

    c._deferred_declick()

    assert win.declicks == [], (
        "the stale deferred declick disabled input despite a live placement "
        "delegate on the window"
    )


def test_deferred_declick_disables_input_when_no_session_owns_the_window(qapp):
    """The normal path: with no placement delegate on the window (the session
    fully torn down), the deferred declick flips it back to click-through.

    Seeded defect: make _deferred_declick return unconditionally -> the window is
    never restored to click-through -> the spy stays empty -> red.
    """
    win = _DeclickWindow(delegate=None)  # nobody owns the window
    c = PlacementController(win, on_save=lambda p: None)

    c._deferred_declick()

    assert win.declicks == [False], (
        "the deferred declick did not flip the window back to click-through"
    )


def test_mouse_press_and_move_preview_position_without_persisting(qapp, monkeypatch):
    """Press and move on the window both drag the block to the global cursor
    via the preview path (no disk writes).

    Seeded defect: make mousePressEvent/mouseMoveEvent ignore the active
    delegate -> the position stays at the default anchor -> red; make
    on_drag call set_position instead of update_position_preview -> the
    window persist callback fires -> pos_spy non-empty -> red.
    """
    save_spy: list[dict] = []
    pos_spy: list[dict] = []
    w = OverlayWindow(style={**STYLE_DEFAULTS, "alignment": "center"},
                      on_position_changed=pos_spy.append)
    try:
        c = PlacementController(w, on_save=save_spy.append)
        c.enter()
        screens = _screens_snapshot()
        g = QGuiApplication.primaryScreen().geometry()
        a = (g.x() + int(g.width() * 0.2), g.y() + int(g.height() * 0.85))
        b = (g.x() + int(g.width() * 0.7), g.y() + int(g.height() * 0.6))

        monkeypatch.setattr(placement_mod, "_cursor_global_pos", lambda: a)
        w.mousePressEvent(_FakeMouseEvent())
        assert w.current_position() == resolve_drag(a[0], a[1], screens), (
            "mouse press did not drag the block to the cursor"
        )

        monkeypatch.setattr(placement_mod, "_cursor_global_pos", lambda: b)
        w.mouseMoveEvent(_FakeMouseEvent())
        assert w.current_position() == resolve_drag(b[0], b[1], screens), (
            "mouse move did not track the cursor"
        )

        assert save_spy == [], "live drag must not persist via on_save"
        assert pos_spy == [], (
            "live drag persisted via the window callback; the preview path "
            "must write nothing until save"
        )
    finally:
        w.close()


# --------------------------------------------------------------------------
# Hint-bar theming (Task 11)
# --------------------------------------------------------------------------


def test_hint_bar_styled_from_supplied_palette(qapp):
    """The placement hint bar is skinned from the palette the tray passes: a
    MIST (light) palette paints the hint with the mist raised token.

    Seeded defect: revert _ensure_hint to the hardcoded dark HINT_QSS
    constant -> the mist raised hex is absent from the hint stylesheet -> red.
    """
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None, palette=MIST)
        c.enter()
        assert MIST["raised"] in c._hint.styleSheet(), (
            "hint bar not skinned from the mist palette; "
            f"stylesheet={c._hint.styleSheet()!r}"
        )
    finally:
        w.close()


def test_hint_bar_defaults_to_dark_palette(qapp):
    """With no palette supplied the hint bar defaults to the dark (night)
    palette so it still reads on any content.

    Seeded defect: change the ctor default from NIGHT to MIST -> the night
    raised hex is absent from the hint stylesheet -> red.
    """
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None)
        c.enter()
        assert NIGHT["raised"] in c._hint.styleSheet(), (
            "default hint palette is not the dark night palette; "
            f"stylesheet={c._hint.styleSheet()!r}"
        )
    finally:
        w.close()


def test_apply_palette_reskins_active_hint(qapp):
    """A live OS theme flip while placement is active re-skins the hint bar:
    apply_palette swaps the raised token from night to mist.

    Seeded defect: make PlacementController.apply_palette skip the hint
    setStyleSheet -> the old night raised hex survives -> red.
    """
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None)  # dark default
        c.enter()
        assert NIGHT["raised"] in c._hint.styleSheet()

        c.apply_palette(MIST)
        css = c._hint.styleSheet()
        assert MIST["raised"] in css and NIGHT["raised"] not in css, (
            f"apply_palette did not re-skin the active hint; stylesheet={css!r}"
        )
    finally:
        w.close()


# --------------------------------------------------------------------------
# Hint re-centering on a cross-monitor drag (P1-15)
# --------------------------------------------------------------------------


class _FakeDragWindow:
    """Minimal window seam for the on_drag re-center test: on_drag reads
    ``_pinned_screen`` before/after ``update_position_preview`` and re-runs
    _position_hint iff it changed. A real OverlayWindow would need two physical
    monitors to jump between; this fake flips its own pinned screen deterministically
    without a real screen object (whose absent .geometry() would poison a real
    window's later paint)."""

    def __init__(self, pinned, becomes):
        self._pinned_screen = pinned
        self._becomes = becomes
        self.previews: list[dict] = []

    def update_position_preview(self, pos: dict) -> None:
        self.previews.append(pos)
        self._pinned_screen = self._becomes

    def alignment(self) -> str:
        return "center"

    def current_position(self) -> dict:
        return {}


def test_drag_recenters_hint_when_pinned_screen_changes(qapp, monkeypatch):
    """P1-15: a drag that crosses a monitor boundary re-pins the window to a
    possibly narrower screen (update_position_preview -> _pin_to_screen resizes
    it). The hint bar was centred for the ENTRY screen's width in enter(); on a
    wide->narrow jump its old x clips off the narrower window. on_drag re-runs
    _position_hint whenever the pinned screen actually changed.

    Seeded defect: drop the ``if after is not before: self._position_hint()``
    re-center in on_drag -> the hint keeps its entry-screen x after the jump ->
    the re-position spy is never called -> red.
    """
    win = _FakeDragWindow(pinned="SCREEN_A", becomes="SCREEN_B")  # a monitor jump
    c = PlacementController(win, on_save=lambda p: None)
    c.active = True          # on_drag guards on active; skip the real enter()
    c._hint = object()       # non-None so the re-center branch is reachable
    recenters: list[int] = []
    monkeypatch.setattr(c, "_position_hint", lambda: recenters.append(1))
    monkeypatch.setattr(placement_mod, "_cursor_global_pos", lambda: (10, 10))

    c.on_drag()

    assert win.previews, "on_drag did not preview the drag position"
    assert recenters == [1], (
        "on_drag did not re-center the hint after the pinned screen changed"
    )


def test_drag_leaves_hint_when_pinned_screen_unchanged(qapp, monkeypatch):
    """P1-15 guard: an in-screen drag (no monitor jump) must NOT re-run
    _position_hint -- otherwise every mouse-move would spuriously re-center the
    hint. on_drag re-centers ONLY when the pinned screen object changed.

    Seeded defect: re-center unconditionally (drop the ``after is not before``
    guard, always call _position_hint) -> an in-screen drag re-centers -> red.
    """
    # becomes == pinned (SAME object) -> the pinned screen is unchanged.
    same = "SCREEN_A"
    win = _FakeDragWindow(pinned=same, becomes=same)
    c = PlacementController(win, on_save=lambda p: None)
    c.active = True
    c._hint = object()
    recenters: list[int] = []
    monkeypatch.setattr(c, "_position_hint", lambda: recenters.append(1))
    monkeypatch.setattr(placement_mod, "_cursor_global_pos", lambda: (10, 10))

    c.on_drag()

    assert win.previews, "on_drag did not preview the drag position"
    assert recenters == [], (
        "on_drag re-centered the hint on an in-screen drag (no monitor jump)"
    )


def test_drag_short_circuits_when_resolved_position_unchanged(qapp, monkeypatch):
    """Mouse jitter within one snapped anchor resolves to the SAME position on
    every move. on_drag must push that position to the window only ONCE:
    re-applying an unchanged position re-runs update_position_preview
    (_reposition_stack's stack re-anchor + a full-window update()) for no visible
    change. Two moves reading the same cursor point -> exactly one
    update_position_preview call.

    Seeded defect (one line): drop the ``if pos == self._last_applied_pos: return``
    short-circuit in on_drag -> the second identical move calls
    update_position_preview again -> the spy records two calls -> red.
    """
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None)
        c.enter()  # seeds _screens + _last_applied_pos (the entry anchor)

        calls: list[dict] = []
        _real = w.update_position_preview

        def _spy(pos):
            calls.append(pos)
            _real(pos)

        monkeypatch.setattr(w, "update_position_preview", _spy)

        # Both moves read the SAME cursor point -> the same resolved position,
        # off the default entry anchor so the FIRST move actually applies.
        g = QGuiApplication.primaryScreen().geometry()
        pt = (g.x() + int(g.width() * 0.2), g.y() + int(g.height() * 0.35))
        monkeypatch.setattr(placement_mod, "_cursor_global_pos", lambda: pt)

        c.on_drag()
        c.on_drag()

        assert len(calls) == 1, (
            f"two moves to the same position pushed {len(calls)} previews; an "
            f"unchanged drag must short-circuit before update_position_preview"
        )
    finally:
        w.close()


def test_on_drag_resolves_against_the_session_screen_snapshot(qapp, monkeypatch):
    """enter() snapshots the screen topology once; on_drag resolves against that
    cached snapshot (self._screens), not a screen list rebuilt from
    QGuiApplication on every move. Proven by swapping the snapshot for a single
    deterministic fake screen and confirming the drag resolves against IT.

    Seeded defect (one line): revert on_drag to rebuild ``screens`` from
    QGuiApplication.screens() and resolve against that instead of self._screens ->
    the injected FAKE snapshot is ignored, the drag resolves against the real
    offscreen screen -> the resolved-screen assertion red.
    """
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=lambda p: None)
        c.enter()
        c._screens = [("FAKE", (0, 0, 1000, 1000))]  # override the session snapshot
        c._last_applied_pos = None  # force the next move to apply

        seen: list[dict] = []
        _real = w.update_position_preview

        def _spy(pos):
            seen.append(pos)
            _real(pos)

        monkeypatch.setattr(w, "update_position_preview", _spy)
        monkeypatch.setattr(placement_mod, "_cursor_global_pos", lambda: (200, 300))

        c.on_drag()

        assert seen and seen[0]["screen"] == "FAKE", (
            f"on_drag did not resolve against the session screen snapshot; got "
            f"{seen!r}"
        )
    finally:
        w.close()


def test_save_persists_via_on_save_even_when_on_exit_raises(qapp):
    """_teardown calls on_exit (the tray's restore-borrowed-preview callback)
    from inside the Qt key/deactivate dispatch that drives save(). Because
    _teardown runs BEFORE _on_save in save(), an on_exit raise would propagate
    out of _teardown and SKIP persistence entirely: the placement the user just
    committed would silently never be written. on_exit must be contained like the
    sibling _on_save guard three lines away.

    Seeded defect: drop the try/except around self._on_exit() in _teardown -> the
    raise escapes _teardown, on_save is never reached, save_spy stays empty (and
    save() itself raises into the caller) -> red.
    """
    save_spy: list[dict] = []

    def _boom() -> None:
        raise RuntimeError("borrowed-preview restore blew up")

    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=save_spy.append, on_exit=_boom)
        c.enter()
        target = w.current_position()

        c.save()  # must not raise despite on_exit blowing up

        assert save_spy == [target], (
            "on_exit raising inside _teardown skipped persistence; save() must "
            f"still persist via on_save. save_spy={save_spy}"
        )
        assert c.active is False, "save did not tear the session down"
    finally:
        w.close()


def test_re_enter_after_save_shows_sample_again(qapp):
    """After a save the controller can be re-entered for a second placement.

    Seeded defect: drop the `active = False` from _teardown -> enter()'s
    re-entry guard makes the second enter a no-op -> the sample is not
    re-shown -> red.
    """
    save_spy: list[dict] = []
    w = OverlayWindow()
    try:
        c = PlacementController(w, on_save=save_spy.append)
        c.enter()
        c.save()
        assert c.active is False
        QApplication.sendPostedEvents(None)
        assert w._sample_id is None, "sample not torn down on save"
        c.enter()
        assert c.active is True, "controller could not re-enter after save"
        assert w._sample_id is not None, "re-enter did not re-show the sample"
    finally:
        w.close()


def test_repeated_sessions_do_not_accumulate_hint_labels(qapp):
    """C6: the hint QLabel is parented to the overlay window, which OUTLIVES
    the per-session controller ("Move subtitles..." builds a fresh controller
    every time). _teardown used to only hide() the hint, so N sessions leaked
    N hidden QLabels on the window. Teardown must delete the hint instead.

    Seeded defect: revert _teardown to the bare `self._hint.hide()` -> three
    enter/save cycles leave three hint labels parented to the window -> red.
    """
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel

    from yamabiko.overlay_app.placement import _HINT_TEXT

    w = OverlayWindow()
    try:
        for _ in range(3):
            c = PlacementController(w, on_save=lambda p: None)
            c.enter()
            c.save()
            # Pump the loop: the deferred declick singleShot fires and the
            # hint's deleteLater is processed before the next session.
            QTest.qWait(20)
        hints = [
            lbl for lbl in w.findChildren(QLabel) if lbl.text() == _HINT_TEXT
        ]
        assert hints == [], (
            f"{len(hints)} hint label(s) still parented to the window after 3 "
            "placement sessions; _teardown must delete the hint, not hide it"
        )
    finally:
        w.close()


def test_drag_in_left_alignment_moves_left_anchor_and_keeps_center():
    """In Left alignment the drag moves left_offset_pct (no 50% snap) and
    leaves the Center anchor h_offset_pct alone.

    Seeded defect: always write h_offset_pct -> red."""
    screens = [("A", (0, 0, 1000, 800))]
    base = {"screen": "A", "v_offset_pct": 88.0, "h_offset_pct": 40.0, "left_offset_pct": 6.0}
    pos = resolve_drag(495.0, 400.0, screens, alignment="left", base=base)
    assert pos["left_offset_pct"] == 49.5
    assert pos["h_offset_pct"] == 40.0
    assert pos["v_offset_pct"] == 50.0
