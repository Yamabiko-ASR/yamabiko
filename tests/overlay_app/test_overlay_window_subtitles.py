"""Line lifecycle tests for OverlayWindow.add_subtitle on the SubtitleStack.

Every subtitle schedules a hold timer; when max_lines evicts a line early
(more than max_lines subtitles inside hold_ms, routine during active
speech), the evicted line's still-pending timer must die with it instead
of firing later.
"""
from __future__ import annotations

import logging
import sys

import pytest
from PySide6.QtCore import QEvent, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from yamabiko.overlay_app.overlay_window import OverlayWindow
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


_LONG_EN = ("The big question is whether Prime Minister Takai can really "
            "deliver on her policies like cutting the number of lawmakers "
            "and handling the consumption tax, and whether the coalition holds.")


def test_trimmed_line_hold_timer_never_fires_on_deleted_widget(qapp, capfd):
    """Overflow-evicting a line must also kill its pending hold timer.

    The stack parents each hold timer to itself and stops + deletes it when
    max_lines evicts the line. A timer that outlived the eviction would fire
    on a line the stack no longer holds and report through the excepthook /
    stderr, once per overflow.

    Seeded defects (each goes red): drop the eviction from
    SubtitleStack.add_line (skip ``self._trim()``) -> two lines stay -> the
    max_lines count assert red; drop the stop()/deleteLater() of the hold
    timer in SubtitleStack._release and make its callback assume the line is
    still live -> the evicted line's timer fires inside the capture window
    and the excepthook/stderr capture below goes red.
    """
    style = {
        **STYLE_DEFAULTS,
        "hold_ms": 80,
        "max_lines": 1,
        "fade_in_ms": 1,
        "fade_out_ms": 1,
    }
    w = OverlayWindow(style=style)
    hook_errors: list[tuple] = []
    old_hook = sys.excepthook
    sys.excepthook = lambda *exc: hook_errors.append(exc)
    try:
        w.add_subtitle("一行目", "line one")
        w.add_subtitle("二行目", "line two")  # max_lines=1 -> evicts line one
        assert len(w.stack.frame().lines) == style["max_lines"]
        # Flush the deferred delete of the evicted line's timer while its
        # 80 ms hold is still pending.
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        # Drive the event loop well past hold_ms so any surviving timer
        # fires now, inside the capture window.
        QTest.qWait(300)
    finally:
        sys.excepthook = old_hook
        w.close()

    err = capfd.readouterr().err
    assert not hook_errors, (
        f"hold timer fired on the evicted line: {hook_errors}"
    )
    assert "RuntimeError" not in err and "already deleted" not in err, (
        f"PySide6 reported a deleted-object access on stderr:\n{err}"
    )


def test_cap_px_uses_configured_pct_of_full_screen_width(qapp):
    """_cap_px() must return int(full_screen_width * max_width_pct / 100).

    Basis note (W4 P0-4): the cap is a percent of the pinned screen's FULL
    geometry width, NOT availableGeometry, matching _reposition_stack's single
    coordinate basis. Direct coverage of the pixel-cap computation (previously
    exercised only transitively through add_subtitle).

    Seeded defect: change the ``/ 100`` divisor to ``/ 10`` in _cap_px ->
    the returned cap is 10x the expected -> red.
    """
    from PySide6.QtGui import QGuiApplication

    w = OverlayWindow(style={**STYLE_DEFAULTS, "max_width_pct": 50})
    try:
        # Compute the expectation independently of the implementation line.
        full_w = QGuiApplication.primaryScreen().geometry().width()
        expected = int(full_w * 0.50)
        assert w._cap_px() == expected, (
            f"_cap_px()={w._cap_px()} != {expected} "
            f"(50% of full screen width {full_w})"
        )
    finally:
        w.close()


def test_add_subtitle_caps_a_long_line_at_cap_px(qapp):
    """A subtitle whose content is wider than the cap must be bounded by the
    effective line cap: the newest line's box (banner gutter included) is
    never wider than _line_cap_px(). This is what makes the width cap reach
    the rendered line.

    Seeded defect: return a wrong cap in _cap_px (e.g. change the ``/ 100``
    divisor to ``/ 10``) -> the raw cap no longer matches the effective line
    cap -> red; hand the stack a hardcoded cap in _reposition_stack -> the
    long line grows past _line_cap_px() -> red.
    """
    from yamabiko.overlay_app.telop_text import layout_line

    w = OverlayWindow(style={**STYLE_DEFAULTS, "max_width_pct": 50})
    try:
        # The cap fits right of this anchor, so the effective line cap is the
        # raw cap (the narrowing at an off-centre anchor is tested in
        # test_overlay_window_position.py).
        assert w._line_cap_px() == w._cap_px(), (
            "this anchor should leave the per-line cap == the raw cap"
        )
        unwrapped = layout_line([{"text": "字幕"}], _LONG_EN, w.stack.look(), 0, 100000)
        assert unwrapped.width > w._line_cap_px(), (
            "the long EN did not exceed the cap; widen _LONG_EN or lower the cap"
        )
        w.add_subtitle("字幕", _LONG_EN)
        box = w.stack.frame().lines[-1]
        assert box.rect.width() <= w._line_cap_px(), (
            f"long line width {box.rect.width()} not bounded at cap "
            f"{w._line_cap_px()}"
        )
    finally:
        w.close()


@pytest.mark.parametrize("garbage", [None, float("nan")])
def test_non_numeric_max_width_pct_via_ws_path_does_not_kill_subtitles(qapp, garbage):
    """The WS style path (Tray._on_style_received -> OverlayWindow.apply_style)
    bypasses config._clamp_style, so a non-numeric OR non-finite max_width_pct
    can reach the live window. add_subtitle must NOT raise, and the line must
    fall back to the 70% default cap rather than crashing every subsequent
    subtitle. None covers the non-numeric case; NaN covers the finite-float
    case json.loads lets through on a loopback set_style frame (it passes
    isinstance yet raises ValueError under int()).

    Seeded defects (each goes red): make _cap_px read the raw
    ``self._style.get("max_width_pct", 70)`` instead of routing it through
    style_defaults.coerce_max_width_pct -> int(width * None / 100) raises
    TypeError inside add_subtitle for the None case and int(width * nan
    / 100) raises ValueError for the NaN case.
    """
    w = OverlayWindow()
    try:
        # apply_style stores the dict verbatim (no clamp) -- mirrors the WS path.
        w.apply_style({**STYLE_DEFAULTS, "max_width_pct": garbage})
        # A long EN so the line hits the cap.
        w.add_subtitle("字幕", _LONG_EN)  # must not raise
        lines = w.stack.frame().lines
        assert len(lines) == 1, "subtitle was dropped instead of rendered"
        # Independent of _cap_px: the explicit 70%% fallback of the full screen.
        full_w = w._active_screen().geometry().width()
        expected_default_cap = int(full_w * 70 / 100)
        # _cap_px falls back to the 70% default for the garbage pct (dev's
        # coerce_max_width_pct hardening).
        assert w._cap_px() == expected_default_cap, (
            f"garbage max_width_pct {garbage!r} did not fall back to the 70% "
            f"default cap: _cap_px() {w._cap_px()} != {expected_default_cap}"
        )
        # The rendered line is bounded at the effective line cap, which stays
        # positive and within the default cap.
        assert lines[-1].rect.width() <= w._line_cap_px(), (
            f"garbage max_width_pct {garbage!r}: line width "
            f"{lines[-1].rect.width()} > _line_cap_px() {w._line_cap_px()}"
        )
        assert 0 < w._line_cap_px() <= expected_default_cap
    finally:
        w.close()


@pytest.mark.parametrize("pct, want_pct", [(True, 70), (-50, 30), (500, 100), (0, 30)])
def test_cap_px_rejects_bool_and_clamps_out_of_range(qapp, pct, want_pct):
    """_cap_px must reject a bool max_width_pct and clamp any finite
    out-of-range value into 30..100. isinstance(True, int) is True in Python, so
    without the bool guard a ``max_width_pct: true`` frame slivers the cap to 1%
    of the screen; -50 / 0 / 500 pass isfinite but must clamp to 30 / 30 / 100.
    This is a belt for direct apply_style callers that bypass the tray/config
    clamp. Expected: True -> 70% default, -50/0 -> 30%, 500 -> 100%.

    Seeded defect (goes red for every case): make _cap_px read the raw
    ``self._style.get("max_width_pct", 70)`` instead of routing it through
    style_defaults.coerce_max_width_pct -> True slivers the cap to 1% (bool
    passes an (int, float) test) and -50 / 0 / 500 produce a negative / zero
    / oversized cap instead of clamping to 30 / 30 / 100.
    """
    from PySide6.QtGui import QGuiApplication

    w = OverlayWindow(style={**STYLE_DEFAULTS, "max_width_pct": pct})
    try:
        full_w = QGuiApplication.primaryScreen().geometry().width()
        expected = int(full_w * want_pct / 100)
        assert w._cap_px() == expected, (
            f"max_width_pct={pct!r} -> cap {w._cap_px()} != {expected} "
            f"({want_pct}% of {full_w})"
        )
    finally:
        w.close()


def test_apply_style_recomputes_frame_height_on_font_growth(qapp):
    """A restyle that grows the EN font must re-measure the existing lines so
    the (now taller / more-wrapped) EN still fits its box: the newest line's
    box height grows after apply_style with a larger en_size.

    Seeded defect: skip the re-layout on a text-key change (keep the old
    layout_key in SubtitleStack.apply_style) -> the box keeps its old height
    -> red.
    """
    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        w.add_subtitle("あ" * 6, _LONG_EN)
        before = w.stack.frame().lines[-1].rect.height()
        w.apply_style({**STYLE_DEFAULTS, "en_size": STYLE_DEFAULTS["en_size"] * 2})
        after = w.stack.frame().lines[-1].rect.height()
        assert after > before, (
            f"after restyle the line box height {after} did not grow from "
            f"{before}: apply_style did not re-measure the line"
        )
    finally:
        w.close()


def test_zorder_probe_timer_is_parented_and_helper_is_silent_without_hwnd(qapp, caplog):
    """Change B z-order instrumentation: wiring + headless-safety contract.

    (1) showEvent starts a periodic z-order probe QTimer that is a CHILD of
        the window (parent=self) so teardown destroys it. A free-standing
        timer would fire its callback on the deleted C++ window and raise a
        libshiboken RuntimeError through the event loop -- the same teardown
        hazard the showEvent retry timer documents (lines ~513-525).
    (2) The logging helper must be SAFE and SILENT when there is no real
        HWND (offscreen/headless test QPA: winId()==1, IsWindow(1)==0): it
        must not raise and must emit no z-order log record. Per docs/TESTING.md
        3.9 (Wired-not-exercised) this does not merely assert the timer
        exists; it INVOKES the helper and asserts its side-effect contract.

    Seeded defects (each one line, each goes red):
    - drop the ``self`` parent from ``QTimer(self)`` in _start_zorder_probe
      -> the probe timer is no longer a child of the window ->
      ``w._zorder_timer in w.findChildren(QTimer)`` is False -> red.
    - remove the ``if not hwnd or not user32.IsWindow(hwnd): return`` guard
      from _log_zorder_state -> under offscreen QPA it falls through to
      GetWindowLongW (returns 0, no raise) and emits an INFO record -> the
      "no z-order log" assert goes red.
    """
    from PySide6.QtCore import QTimer

    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        w.show()
        QApplication.processEvents()
        # (1) The probe timer exists, is a child of the window, and is running.
        assert w._zorder_timer is not None, "z-order probe timer not created"
        assert w._zorder_timer in w.findChildren(QTimer), (
            "z-order probe timer is not a child of the overlay window "
            "(teardown would not destroy it -> libshiboken RuntimeError)"
        )
        assert w._zorder_timer.isActive(), "z-order probe timer is not running"
        # (2) Invoke the helper directly. Under the offscreen QPA winId() is
        # not a real HWND, so it must return WITHOUT raising and WITHOUT
        # logging anything.
        with caplog.at_level(logging.INFO):
            w._log_zorder_state()  # must not raise
        zorder_records = [
            r for r in caplog.records if "z-order" in r.getMessage()
        ]
        assert not zorder_records, (
            "z-order helper logged despite having no valid HWND: "
            f"{[r.getMessage() for r in zorder_records]}"
        )
    finally:
        w.close()


def test_zorder_probe_never_logs_foreground_window_title(qapp, monkeypatch, caplog):
    """The z-order probe must NEVER emit a foreground window TITLE.

    Window titles are PII (browser tabs, document names, email subjects,
    password-manager entries). This probe logs at INFO into main-overlay.log,
    which ships verbatim in the one-click diagnostics zip, so any title it
    reads leaks. The probe must identify the covering window by class + handle
    only and never read the title.

    RED baseline / seeded defect: with the current title-capture code (or if
    anyone reintroduces the GetWindowTextW ``foreground=%r`` logging), the
    injected SECRET-WINDOW-TITLE sentinel lands in the z-order record and this
    goes red.
    """
    import ctypes
    from yamabiko.overlay_app import overlay_window as overlay_module

    class _FakeWinFunc:
        def __init__(self, result=0, side_effect=None):
            self.argtypes = None
            self.restype = None
            self.result = result
            self.side_effect = side_effect
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            if self.side_effect is not None:
                return self.side_effect(*args)
            return self.result

    class _FakeUser32:
        def __init__(self):
            self.IsWindow = _FakeWinFunc(result=1)
            self.GetWindowLongW = _FakeWinFunc(result=0x00000008)
            self.GetWindow = _FakeWinFunc(result=0)
            self.GetForegroundWindow = _FakeWinFunc(result=0xABCDEF01)
            # If the probe ever reads the TITLE, this writes a sentinel the
            # assertion below hunts for in every logged record.
            self.GetWindowTextW = _FakeWinFunc(
                side_effect=lambda _hwnd, buf, _size: (
                    setattr(buf, "value", "SECRET-WINDOW-TITLE") or 19
                )
            )
            # Class name is non-sensitive metadata (names the app, not its
            # content); the fixed probe reads this instead of the title.
            self.GetClassNameW = _FakeWinFunc(
                side_effect=lambda _hwnd, buf, _size: (
                    setattr(buf, "value", "Chrome_WidgetWin_1") or 18
                )
            )

    class _FakeWindll:
        def __init__(self, user32):
            self.user32 = user32

    fake_user32 = _FakeUser32()
    monkeypatch.setattr(overlay_module.sys, "platform", "win32")
    monkeypatch.setattr(
        overlay_module.ctypes, "windll", _FakeWindll(fake_user32),
        raising=False,
    )

    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        monkeypatch.setattr(w, "winId", lambda: 0x1111)
        with caplog.at_level(logging.INFO):
            w._log_zorder_state()
    finally:
        w.close()

    zorder_msgs = [
        r.getMessage() for r in caplog.records if "z-order" in r.getMessage()
    ]
    # Guard against a silently-crashed probe making the leak check vacuous.
    assert zorder_msgs, "z-order probe emitted no record; leak check would be vacuous"
    leaked = [m for m in zorder_msgs if "SECRET-WINDOW-TITLE" in m]
    assert not leaked, (
        "z-order probe leaked a foreground window TITLE into its log "
        f"(titles are PII): {leaked}"
    )


def test_zorder_probe_configures_pointer_sized_win32_signatures(qapp, monkeypatch, caplog):
    """The win32 z-order probe must configure ctypes signatures before it
    reads HWND values. On 64-bit Windows, default ctypes return conversion can
    truncate pointer-sized handles; that makes the instrumentation lie.

    It identifies the covering window by its CLASS name (GetClassNameW), never
    its title: titles are PII and this probe's log ships in the diagnostics
    zip. GetWindowTextW must not be called at all.

    Seeded defects (each one line, each goes red):
    - remove the argtypes/restype setup in _log_zorder_state -> the fake
      user32 function attributes remain None and the signature asserts go red.
    - swap GetClassNameW back to GetWindowTextW for the foreground read ->
      GetWindowTextW.calls is non-empty and the class name is absent from the
      log -> red.
    """
    import ctypes
    from ctypes import wintypes
    from yamabiko.overlay_app import overlay_window as overlay_module

    class _FakeWinFunc:
        def __init__(self, result=0, side_effect=None):
            self.argtypes = None
            self.restype = None
            self.result = result
            self.side_effect = side_effect
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            if self.side_effect is not None:
                return self.side_effect(*args)
            return self.result

    class _FakeUser32:
        def __init__(self):
            self.IsWindow = _FakeWinFunc(result=1)
            self.GetWindowLongW = _FakeWinFunc(result=0x00000008)
            self.GetWindow = _FakeWinFunc(result=0)
            self.GetForegroundWindow = _FakeWinFunc(result=0x123456789ABC)
            self.GetClassNameW = _FakeWinFunc(
                side_effect=lambda _hwnd, buf, _size: (
                    setattr(buf, "value", "Chrome_WidgetWin_1") or 18
                )
            )
            # Present so the test can assert the probe NEVER calls it (the
            # window title is PII and must not be read).
            self.GetWindowTextW = _FakeWinFunc(
                side_effect=lambda _hwnd, buf, _size: (
                    setattr(buf, "value", "SHOULD-NOT-BE-READ") or 18
                )
            )

    class _FakeWindll:
        def __init__(self, user32):
            self.user32 = user32

    fake_user32 = _FakeUser32()
    monkeypatch.setattr(overlay_module.sys, "platform", "win32")
    monkeypatch.setattr(
        overlay_module.ctypes, "windll", _FakeWindll(fake_user32),
        raising=False,
    )

    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        monkeypatch.setattr(w, "winId", lambda: 0x123456789ABC)
        with caplog.at_level(logging.INFO):
            w._log_zorder_state()
    finally:
        w.close()

    # Pointer-sized handle signatures are configured (else 64-bit truncation).
    assert fake_user32.IsWindow.argtypes == [wintypes.HWND]
    assert fake_user32.GetWindowLongW.argtypes == [wintypes.HWND, ctypes.c_int]
    assert fake_user32.GetWindow.argtypes == [wintypes.HWND, wintypes.UINT]
    assert fake_user32.GetWindow.restype is wintypes.HWND
    assert fake_user32.GetForegroundWindow.argtypes == []
    assert fake_user32.GetForegroundWindow.restype is wintypes.HWND
    # The foreground window is identified by CLASS name, configured + used.
    assert fake_user32.GetClassNameW.argtypes == [
        wintypes.HWND, wintypes.LPWSTR, ctypes.c_int,
    ]
    assert fake_user32.GetClassNameW.restype is ctypes.c_int
    assert fake_user32.GetClassNameW.calls, "probe did not read the fg class name"
    # The title must never be read: GetWindowTextW is not called at all.
    assert fake_user32.GetWindowTextW.calls == [], (
        "z-order probe called GetWindowTextW; window titles are PII and must "
        "not be read"
    )
    # The class name reaches the log (non-sensitive app identity).
    zorder_msgs = [
        r.getMessage() for r in caplog.records if "z-order" in r.getMessage()
    ]
    assert any("Chrome_WidgetWin_1" in m for m in zorder_msgs), (
        f"foreground class name not logged; got: {zorder_msgs}"
    )


def test_add_subtitle_recovers_from_deleted_pinned_screen(qapp):
    """A pinned QScreen whose C++ object Qt deleted (monitor hotplug / DPI /
    resolution change) must not crash the render path: the subtitle is still
    registered, and the add itself re-pins to a live screen and re-places the
    stack there.

    Confirmed from a live session: 67 occurrences of
    ``RuntimeError: libshiboken: Internal C++ object
    (PySide6.QtGui.QScreen) already deleted.`` raised through the add path,
    and ZERO subtitles shown for the whole run. Root cause: _active_screen
    returned the dead _pinned_screen verbatim (a deleted shiboken wrapper is
    still truthy), and every consumer then called .geometry()/.name() on it
    unguarded. The add path no longer reads the screen, so it must heal the
    pin itself or keep the dead screen's geometry and stale anchor.

    Seeded defect / RED baseline: drop the ``self._heal_dead_pin()`` call from
    add_subtitle -> the dead pin survives the add -> the ``is primary``
    assertion red.
    """
    class _DeletedScreen:
        """Stand-in for a QScreen whose underlying C++ object Qt deleted.

        shiboken raises RuntimeError on ANY access to a deleted C++ object;
        reproduce that on every attribute so the overlay must treat it as
        dead and recover, exactly as for a real hotplug QScreen deletion.
        A bare object is truthy (no __bool__/__len__ on the type), matching a
        real deleted QScreen wrapper.
        """

        _MSG = "Internal C++ object (PySide6.QtGui.QScreen) already deleted."

        def __getattr__(self, name):
            raise RuntimeError(self._MSG)

    from PySide6.QtGui import QGuiApplication

    primary = QGuiApplication.primaryScreen()
    w = OverlayWindow(style=dict(STYLE_DEFAULTS))  # default anchor: primary
    try:
        # The stale state the dead screen leaves (its geometry and anchor),
        # then the hotplug deletion: a dead wrapper that raises on any access.
        w.setGeometry(7, 7, 320, 200)
        w.stack.set_anchor(1, 1, 10, 10, 0, 10)
        w._pinned_screen = _DeletedScreen()
        w.add_subtitle("こんにちは", "hello")  # must NOT raise
        assert len(w.stack.frame().lines) == 1, (
            "add_subtitle registered no line after the pinned screen was "
            "deleted (the render path crashed on the dead QScreen)"
        )
        # Asserted before any _active_screen() call, which re-pins on its own.
        assert w._pinned_screen is primary, (
            f"add_subtitle did not re-pin to the live screen: "
            f"{w._pinned_screen!r}"
        )
        sg = primary.geometry()
        assert w.geometry() == sg.adjusted(0, 0, 0, -1), (
            "the window kept the dead screen's geometry")
        pos = w.current_position()
        g = w.stack.frame().lines[-1].rect
        assert (g.x(), g.y() + g.height()) == (
            int(sg.width() * pos["left_offset_pct"] / 100),
            int(sg.height() * pos["v_offset_pct"] / 100),
        ), f"the subtitle landed at the stale anchor: {g}"
    finally:
        w.close()


# --- jp_ruby element-shape guard (the last unguarded WS boundary field) ---
#
# The boundary is already hardened for NaN styles and malformed colours, but
# a jp_ruby entry that is not a dict (a malformed pipeline frame like
# ["田中"] instead of [{"text": "田中"}]) would reach the ruby layout, whose
# seg.get raises AttributeError -- one crashed subtitle slot per frame.


def test_add_subtitle_drops_non_dict_ruby_segments_and_still_renders(qapp, caplog):
    """A jp_ruby list carrying non-dict entries must render: the garbage
    entries are dropped (DEBUG trace), the dict entries survive, and the
    line lands with its Japanese block.

    Seeded defect: remove the _drop_non_dict_ruby_segments call from
    overlay_window._segments -> add_subtitle raises AttributeError ('str'
    object has no attribute 'get') -> red (error).
    """
    w = OverlayWindow(style=None)
    try:
        with caplog.at_level(
            logging.DEBUG, logger="yamabiko.overlay_app.overlay_window",
        ):
            w.add_subtitle(
                "田中さん", "Mr. Tanaka",
                jp_ruby=["田中", {"text": "さん"}, 42, None],
            )
        lines = w.stack.frame().lines
        assert lines, "subtitle was dropped instead of rendered"
        assert "jp" in lines[-1].blocks, "the surviving segment was not laid out"
        drops = [
            r for r in caplog.records if "non-dict jp_ruby" in r.getMessage()
        ]
        assert drops, "dropped segments must leave a DEBUG trace"
    finally:
        w.close()


def test_add_subtitle_with_all_garbage_ruby_list_falls_back_to_plain_jp(qapp):
    """When EVERY jp_ruby entry is malformed, the filtered list is empty and
    the line must degrade exactly like jp_ruby=None: the plain `jp` string
    becomes the single ruby segment. No crash, no empty JP row.

    Seeded defect: drop the plain-jp fallback in overlay_window._segments
    (return the empty filtered list) -> the line has no Japanese block ->
    red.
    """
    w = OverlayWindow(style=None)
    try:
        w.add_subtitle("こんにちは", "hello", jp_ruby=["こん", 3.5, True])
        lines = w.stack.frame().lines
        assert lines, "subtitle was dropped instead of rendered"
        assert "jp" in lines[-1].blocks, (
            "an all-garbage jp_ruby must fall back to the plain jp text"
        )
    finally:
        w.close()


def test_surviving_line_hold_timer_still_expires_the_line(qapp):
    """The hold-timer fix must not break the normal hold->fade->remove path:
    a line that is NOT evicted is removed by its own hold timer.

    Seeded defect: add the line without a hold (``hold_ms=None`` in
    add_subtitle's add_line call) -> the line never fades out -> still
    present after the hold window -> red.
    """
    style = {
        **STYLE_DEFAULTS,
        "hold_ms": 50,
        "max_lines": 3,
        "fade_in_ms": 1,
        "fade_out_ms": 1,
    }
    w = OverlayWindow(style=style)
    try:
        w.add_subtitle("こんにちは", "hello")
        assert len(w.stack.frame().lines) == 1
        # hold (50ms) + fade_out (1ms), with margin.
        QTest.qWait(400)
        assert len(w.stack.frame().lines) == 0, (
            "line was not removed after its hold window expired"
        )
    finally:
        w.close()


def test_timed_out_lines_are_freed_after_deferred_delete(qapp):
    """A line removed by its own hold timer must not outlive its expiry: the
    stack holds no lines and every hold timer is gone once the deferred
    deletes run. A hold timer the stack kept (or never deleted) would pile up
    one QTimer per subtitle for the rest of the session.

    Seeded defect: drop the deleteLater() of the hold timer in
    SubtitleStack._release -> five single-shot timers stay children of the
    stack -> red.
    """
    style = {
        **STYLE_DEFAULTS,
        "hold_ms": 20,
        "max_lines": 5,
        "fade_in_ms": 1,
        "fade_out_ms": 1,
    }

    def hold_timers():
        return [t for t in w.stack.findChildren(QTimer) if t.isSingleShot()]

    w = OverlayWindow(style=style)
    try:
        for i in range(5):
            w.add_subtitle(f"行{i}", f"line {i}")
        assert len(w.stack.frame().lines) == 5
        assert len(hold_timers()) == 5
        # hold (20ms) + fade_out (1ms), with margin, then flush the deletes.
        QTest.qWait(400)
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert w.stack.frame().lines == ()
        assert hold_timers() == [], (
            f"{len(hold_timers())} hold timers outlived their lines"
        )
    finally:
        w.close()


def test_banner_text_follows_pause_series_and_genre(qapp):
    """The banner shows the genre, the series' Japanese title when one is
    picked (English = label without the year), and 一時停止 while paused.

    Seeded defect: ignore set_series -> genre stays -> red."""
    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        w.set_state("news", False)
        assert w.stack.frame().banner_text == ("ニュース", "News")
        w.set_series(154587, "Frieren (2023)", "葬送のフリーレン")
        assert w.stack.frame().banner_text == ("葬送のフリーレン", "Frieren")
        w.set_state("news", True)
        assert w.stack.frame().banner_text == ("一時停止", "Paused")
        assert w.stack.frame().banner is not None   # visible with no lines
        w.set_state("news", False)
        w.set_series(None, "", "")
        assert w.stack.frame().banner_text == ("ニュース", "News")
    finally:
        w.close()


def test_problem_lines_follow_set_problems_and_recovery_lines_end_after_their_time(qapp, monkeypatch):
    """set_problems shows one problem line per active kind, translator
    above audio; show_recovery puts the recovery line in its kind's slot
    for NOTICE_RECOVERY_MS (5 s; shortened here), then it goes. It is
    ignored while that kind's problem shows, and a problem that returns
    during its recovery line takes the slot back for good.

    Seeded defects: no recovery timer (the line never goes), the recovery
    line out of kind order, a recovery shown over an active problem, or a
    recovery line winning over the problem that returns -> red."""
    from yamabiko import problems
    from yamabiko.overlay_app import overlay_window as ow_mod
    from yamabiko.overlay_app.telop_look import NOTICE_RECOVERY_MS

    assert NOTICE_RECOVERY_MS == 5000
    monkeypatch.setattr(ow_mod, "NOTICE_RECOVERY_MS", 80)
    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        def slots():
            return [(n.kind, n.recovery) for n in w.stack.frame().notices]

        translator = problems.translator_problem(True)
        audio = problems.audio_problem("Speakers")
        w.set_problems([audio, translator])
        assert slots() == [("translator", False), ("audio", False)]
        w.set_problems([audio])
        w.show_recovery("translator", "Translations are back.")
        assert slots() == [("translator", True), ("audio", False)]
        QTest.qWait(300)
        assert slots() == [("audio", False)]
        w.show_recovery("audio", "Listening again.")    # audio is active: ignored
        assert slots() == [("audio", False)]
        w.set_problems([])
        w.show_recovery("audio", "Listening again.")
        assert slots() == [("audio", True)]
        w.set_problems([audio])                           # back during its recovery line
        QTest.qWait(300)
        assert slots() == [("audio", False)]
    finally:
        w.close()
