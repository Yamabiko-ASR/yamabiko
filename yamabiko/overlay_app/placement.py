"""On-screen placement mode for the subtitle overlay.

Replaces the retired Unlock/Lock affordance: the user grabs a persistent
sample subtitle and drags it to where subtitles should appear. Preset
spots (the flyout's Position picker and Settings) are one click; free
placement is this controller.

``PlacementController`` drives one placement session against an
``OverlayWindow``. ``enter()`` enables input, shows the sample subtitle and
a hint bar, and remembers the entry position for cancel. Dragging the
sample re-places the whole block, jumping monitors when the cursor crosses
to another screen. Enter (or a click-away) saves; Esc cancels. The live
drag routes through ``OverlayWindow.update_position_preview`` so it never
touches disk; persistence happens exactly once, on save, through the
injected ``on_save`` callback (typically ``OverlayWindow.set_position``).

``resolve_drag`` is the pure drag-to-position math: given a global cursor
point and the screen list it returns the anchor position dict, with no Qt
dependency, so the snap/monitor-jump/clamp logic is testable bare.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .position_config import clamp_position, _H_SNAP_PCT

_log = logging.getLogger(__name__)


def screens_snapshot() -> list[tuple[str, tuple[int, int, int, int]]]:
    """Snapshot the live screen topology as ``(name, (x, y, w, h))`` tuples.

    The single formatter for QGuiApplication.screens() -> the plain tuple
    shape that position_config's pure helpers (migrate_legacy_geometry,
    resolve_drag) consume. Lives here rather than in position_config because
    it is the ONLY Qt touch; position_config stays Qt-free and bare-testable.
    """
    from PySide6.QtGui import QGuiApplication
    return [
        (s.name(), (s.geometry().x(), s.geometry().y(),
                    s.geometry().width(), s.geometry().height()))
        for s in QGuiApplication.screens()
    ]

# Sample subtitle shown during placement. Short, with a couple of ruby
# segments so the user sees the real furigana layout while positioning.
_SAMPLE_JP = "字幕サンプル"
_SAMPLE_EN = "Subtitle placement sample"
_SAMPLE_RUBY = [
    {"text": "字幕", "reading": "じまく"},
    {"text": "サンプル"},
]

# Hint bar copy. No em-dashes (plain hyphens). The hint bar is skinned from
# the active theme palette (see PlacementController._ensure_hint), which the
# tray hands down; callers that pass no palette get the dark default so the
# hint still reads on any content.
_HINT_TEXT = "Drag to position - Enter to save - Esc to cancel"
_HINT_TOP_MARGIN_PX = 48


def _cursor_global_pos() -> tuple[int, int]:
    """Return the global cursor position as ``(x, y)``.

    Wrapped in a module-level function (rather than calling
    ``QCursor.pos()`` inline) so a test can inject a deterministic point
    without monkeypatching the Shiboken ``QCursor`` type.
    """
    from PySide6.QtGui import QCursor
    p = QCursor.pos()
    return p.x(), p.y()


def resolve_drag(
    global_x: float,
    global_y: float,
    screens: list[tuple[str, tuple[int, int, int, int]]],
    *,
    alignment: str = "center",
    base: Optional[dict] = None,
) -> dict:
    """Map a global cursor point to an anchor position dict.

    ``screens`` mirrors position_config's shape: ``(name, (x, y, w, h))``
    per monitor. The screen CONTAINING the cursor wins, so dragging across
    a monitor boundary jumps the whole subtitle block to that monitor. The
    horizontal offset is the cursor's x as a percent of that screen's
    width; the vertical offset is the cursor's y as a percent of height.
    Both are clamped to the anchor bounds via ``clamp_position``.

    Returns ``base`` clamped (the default anchor when it is None) when
    ``screens`` is empty (degenerate: no display to resolve against).

    The cursor sets the ACTIVE alignment's horizontal anchor: ``h_offset_pct``
    (the stack centre, snapped to 50.0 within +-2 % of the width) in Center,
    ``left_offset_pct`` (the stack's left edge, no snap) in Left. Keys the
    drag does not set come from ``base`` (the window's current position), so
    the other alignment's anchor is kept.
    """
    chosen: Optional[tuple[str, tuple[int, int, int, int]]] = None
    for name, (sx, sy, sw, sh) in screens:
        if sx <= global_x < sx + sw and sy <= global_y < sy + sh:
            chosen = (name, (sx, sy, sw, sh))
            break
    if chosen is None:
        if not screens:
            return clamp_position(dict(base or {}))
        # Cursor is off every screen (rare: a dead corner of a
        # non-rectangular virtual desktop). Fall back to the first screen
        # so the block still lands somewhere visible.
        chosen = screens[0]
    name, (sx, sy, sw, sh) = chosen
    x_pct = (global_x - sx) / sw * 100.0 if sw else 50.0
    v_pct = (global_y - sy) / sh * 100.0 if sh else 50.0
    key = "left_offset_pct" if alignment == "left" else "h_offset_pct"
    if key == "h_offset_pct" and abs(x_pct - 50.0) <= _H_SNAP_PCT:
        x_pct = 50.0
    return clamp_position({
        **(base or {}),
        "screen": name,
        "v_offset_pct": round(v_pct, 1),
        key: round(x_pct, 1),
    })


class PlacementController:
    """Drives one on-screen placement session against an ``OverlayWindow``.

    Constructed lazily by the tray's "Move subtitles..." action (and,
    later, Task 10's Position page) with ``on_save`` bound to the window's
    ``set_position``, its single persist path.
    """

    def __init__(
        self,
        window,
        on_save: Callable[[dict], None],
        palette: Optional[dict] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ):
        from .theme import NIGHT
        self._window = window
        self._on_save = on_save
        self.active = False
        self._hint = None
        self._entry_position: dict = {}
        # Screen topology snapshot, taken once per session in enter(). on_drag
        # runs on every mouse-move; rebuilding the screen list from
        # QGuiApplication each move is pure overhead. Empty until enter().
        self._screens: list[tuple[str, tuple[int, int, int, int]]] = []
        # The last position actually pushed to the window. on_drag short-circuits
        # when a move resolves to this same position, so mouse jitter within one
        # snapped anchor doesn't re-run update_position_preview (a stack
        # re-anchor + full-window update()) for no visible change. None => nothing
        # applied yet this session.
        self._last_applied_pos: Optional[dict] = None
        # Hint-bar theme palette. Defaults to the dark (night) palette so the
        # hint reads on any content when no palette is supplied (older callers
        # / tests). The tray passes its current chrome palette here.
        self._palette = palette if palette is not None else NIGHT
        # Fired once on teardown (save / cancel / click-away). The tray supplies
        # it to restore the settings live-preview sample this session borrowed
        # the shared sample slot for (P1-14). None for callers that don't share
        # the slot (older callers / tests).
        self._on_exit = on_exit

    def enter(self) -> None:
        """Begin placement: enable input, show the sample + hint bar, and
        remember the current position so cancel can restore it.

        Re-entry while already active is a no-op (the tray also guards this,
        but the invariant is defended here too).
        """
        if self.active:
            return
        self._entry_position = self._window.current_position()
        # Snapshot the screen topology ONCE for this session. A monitor hotplug
        # mid-drag is rare, and its staleness is acceptable: the next placement
        # session re-snapshots, and _revalidate_screen still re-pins the passive
        # overlay on hotplug. The entry position is what's currently displayed,
        # so seed it as the last-applied position -- a first move to the same
        # spot is then a genuine no-op the on_drag short-circuit skips.
        self._screens = screens_snapshot()
        self._last_applied_pos = dict(self._entry_position)
        self._window.set_interactive(True)
        self._window.show_sample(_SAMPLE_JP, _SAMPLE_EN, jp_ruby=_SAMPLE_RUBY)
        self._ensure_hint()
        self._position_hint()
        self._hint.show()
        # A Tool window with WA_ShowWithoutActivating won't take keyboard
        # focus on show; placement needs Enter/Esc, so grab focus now.
        self._window.activateWindow()
        self._window.setFocus()
        # Register the delegate LAST: the set_interactive show() dance above
        # can emit a spurious WindowDeactivate, and we must not treat that
        # as a click-away save before the session is fully set up.
        self._window._placement_delegate = self
        self.active = True

    def on_drag(self) -> None:
        """Live-drag handler: re-place the block under the global cursor.

        A drag started on the sample can travel anywhere, including onto
        another monitor, so we resolve against the GLOBAL cursor and the
        per-session screen snapshot. The result is previewed WITHOUT
        persisting: persistence is deferred to ``save()`` so a drag doesn't
        spam user_overrides.yaml (and the cross-process filelock) on every
        mouse move.
        """
        if not self.active:
            return
        x, y = _cursor_global_pos()
        pos = resolve_drag(x, y, self._screens, alignment=self._window.alignment(),
                           base=self._window.current_position())
        # Short-circuit an unchanged drag: mouse jitter within one snapped anchor
        # resolves to the SAME position dict. Re-applying it would re-run
        # update_position_preview -> _reposition_stack (a stack re-anchor) plus
        # a full-window update() for no visible change. Only push a position that
        # actually differs from the last one applied.
        if pos == self._last_applied_pos:
            return
        # A cross-monitor drag re-pins the window to a possibly different-width
        # screen (update_position_preview -> _pin_to_screen resizes it). The
        # hint bar was centred for the ENTRY screen's width in enter(); on a
        # wide->narrow jump its old x clips off the narrower window. Re-centre
        # it whenever the pinned screen actually changed (P1-15).
        before = getattr(self._window, "_pinned_screen", None)
        self._window.update_position_preview(pos)
        self._last_applied_pos = pos
        after = getattr(self._window, "_pinned_screen", None)
        if after is not before and self._hint is not None:
            self._position_hint()

    def save(self) -> None:
        """Persist the placed position once, then tear the session down."""
        if not self.active:
            return
        pos = self._window.current_position()
        self._teardown()
        try:
            self._on_save(pos)
        except Exception:
            # on_save routes through config_writer._atomic_write, which
            # re-raises OSError (disk full, AV lock on user_overrides.yaml).
            # save() is driven from a Qt key / deactivate handler, so an
            # escape would reach the event dispatcher: contain it here.
            _log.exception("placement on_save callback failed")

    def cancel(self) -> None:
        """Restore the entry position (no persist) and tear the session
        down. The entry position was already the persisted state, so
        restoring it visually and writing nothing leaves the file
        untouched."""
        if not self.active:
            return
        entry = self._entry_position
        self._teardown()
        self._window.update_position_preview(entry)

    def _teardown(self) -> None:
        # active=False + unregister FIRST (synchronously) so nothing re-enters
        # save()/cancel() while the teardown runs. This ordering from the
        # original design stays load-bearing.
        self.active = False
        if getattr(self._window, "_placement_delegate", None) is self:
            self._window._placement_delegate = None
        if self._hint is not None:
            # Delete rather than just hide: the hint is parented to the
            # overlay window, which OUTLIVES this controller (the tray builds
            # a fresh controller per "Move subtitles..." session), so a hidden
            # hint would leak one QLabel per session. deleteLater defers the
            # C++ teardown past this dispatch; the ref is dropped now so a
            # re-enter builds a fresh hint via _ensure_hint.
            self._hint.hide()
            self._hint.deleteLater()
            self._hint = None
        self._window.hide_sample()
        # DEFER the click-through flag flip. set_interactive(False) calls
        # setWindowFlags()+show(), which destroys and recreates the native HWND
        # on Windows. _teardown runs INSIDE the window's own event dispatch
        # (a keyPressEvent / WindowDeactivate handler routed here via
        # save()/cancel()); recreating the HWND synchronously inside that
        # dispatch is a Windows sporadic-crash class. singleShot(0) runs the
        # flag dance on the next event-loop turn, after the current event
        # unwinds. Only the flag dance defers; active/delegate already cleared
        # above, so the guarded handlers stay inert in the meantime.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._deferred_declick)
        # Restore the settings live-preview sample this session borrowed the
        # shared sample slot for (P1-14). Runs synchronously: it only adds a
        # child widget (no HWND recreation), and the child survives the later
        # deferred flag dance.
        #
        # Guarded like the sibling _on_save call in save(): _teardown runs from
        # inside the window's own key / deactivate dispatch, and in save() it
        # runs BEFORE _on_save, so an unguarded raise here would both escape into
        # the Qt event dispatcher AND skip persistence of the placement the user
        # just committed. Contain it (log + continue) so save() always reaches
        # _on_save.
        if self._on_exit is not None:
            try:
                self._on_exit()
            except Exception:
                _log.exception("placement on_exit callback failed")

    def _deferred_declick(self) -> None:
        """Flip the window back to click-through on the next event-loop turn.

        Scheduled by _teardown (see the HWND-recreate rationale there). Skips
        the flip if a NEW placement session has since claimed the window: a
        fast Move -> save -> Move can install a fresh delegate and re-enable
        input before this fires, and disabling it underneath that session would
        break the new drag. Guards the window still existing (a quit between
        teardown and this tick leaves a deleted C++ object)."""
        if getattr(self._window, "_placement_delegate", None) is not None:
            return
        try:
            self._window.set_interactive(False)
        except RuntimeError:
            # The window's C++ object was deleted between teardown and this
            # tick (app quit): nothing left to declick.
            pass

    def apply_palette(self, palette: dict) -> None:
        """Re-skin the hint bar for a new theme palette.

        Called on a live OS light/dark flip while a placement session is
        active (the tray routes the flip here). A no-op on the hint when it
        has not been built yet; the new palette is still stored so the next
        _ensure_hint uses it.
        """
        self._palette = palette
        if self._hint is not None:
            from .theme import hint_qss
            self._hint.setStyleSheet(hint_qss(palette))

    def _ensure_hint(self) -> None:
        if self._hint is not None:
            return
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel
        from .theme import hint_qss
        lbl = QLabel(_HINT_TEXT, self._window)
        lbl.setTextFormat(Qt.PlainText)
        lbl.setStyleSheet(hint_qss(self._palette))
        # The hint must not eat drag clicks: press+move anywhere on the
        # window (including over the hint) drags the block.
        lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._hint = lbl

    def _position_hint(self) -> None:
        self._hint.adjustSize()
        x = max(0, (self._window.width() - self._hint.width()) // 2)
        self._hint.move(x, _HINT_TOP_MARGIN_PX)
