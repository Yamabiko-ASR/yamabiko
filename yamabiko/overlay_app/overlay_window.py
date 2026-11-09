"""PySide6 always-on-top transparent subtitle overlay window.

A full-monitor transparent click-through window pinned to ONE screen hosts
one SubtitleStack (telop_stack) that paints the subtitles; the stack is
placed inside it by percent offsets (the anchor model in position_config):
left_offset_pct is the stack's left edge in Left alignment, h_offset_pct its
centre in Center.

Problem notices (set_problems, show_recovery) ride the same stack.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import re
import sys
from typing import Callable

from PySide6.QtCore import Qt, QTimer, QEvent, Signal
from PySide6.QtGui import QGuiApplication, QColor, QPainter, QPen, QIcon, QScreen
from PySide6.QtWidgets import QWidget

from .. import paths
from ..problems import KIND_ORDER, is_problem
from .style_defaults import STYLE_DEFAULTS, coerce_max_width_pct
from .position_config import clamp_position
from .telop_look import GENRE_BANNER, NOTICE_RECOVERY_MS, PAUSED_BANNER
from .telop_stack import SubtitleStack


_dwm_log = logging.getLogger(__name__ + ".dwm")
_geom_log = logging.getLogger(__name__ + ".geom")
_zorder_log = logging.getLogger(__name__ + ".zorder")
_log = logging.getLogger(__name__)


def _drop_non_dict_ruby_segments(jp_ruby) -> list:
    """Return ``jp_ruby`` restricted to its dict entries ([] for non-list
    input).

    The WS boundary is already hardened for NaN styles and malformed
    colours; jp_ruby element SHAPE was the last unguarded field. A frame
    like ``{"jp_ruby": ["田中"]}`` would reach the ruby layout, whose
    ``seg.get`` raises AttributeError and kills the directly-connected
    subtitle slot once per frame. Dropping (with a DEBUG trace) beats
    crashing: the JP row is decorative next to the EN line, and _segments
    falls back to the plain ``jp`` string when nothing survives.
    """
    if not isinstance(jp_ruby, (list, tuple)):
        if jp_ruby is not None:
            _log.debug(
                "jp_ruby is %s, not a list; ignoring it",
                type(jp_ruby).__name__,
            )
        return []
    good = [seg for seg in jp_ruby if isinstance(seg, dict)]
    if len(good) != len(jp_ruby):
        _log.debug(
            "dropped %d non-dict jp_ruby segment(s)", len(jp_ruby) - len(good),
        )
    return good


_SERIES_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")


def series_banner_english(label: str) -> str:
    """English half of a series banner: the tray label without its "(year)"
    suffix ("Frieren (2023)" -> "Frieren")."""
    return _SERIES_YEAR.sub("", label or "")


def _segments(jp, jp_ruby) -> list[dict]:
    """Ruby segments to render: dict entries only, text and reading coerced
    to str (a hostile WS frame could carry numbers), or the plain ``jp``
    string when no segment carries text."""
    out = []
    for seg in _drop_non_dict_ruby_segments(jp_ruby):
        text = seg.get("text", "")
        reading = seg.get("reading")
        entry = {"text": text if isinstance(text, str) else str(text)}
        if reading:
            entry["reading"] = reading if isinstance(reading, str) else str(reading)
        out.append(entry)
    if any(s["text"] or s.get("reading") for s in out):
        return out
    text = jp if isinstance(jp, str) else str(jp or "")
    return [{"text": text}] if text else []


def _suppress_windows11_chrome(hwnd: int) -> None:
    """Strip every DWM decoration Windows 11 applies to top-level windows.

    Qt's FramelessWindowHint only removes the title bar; DWM still paints:
      - rounded corners (DWMWA_WINDOW_CORNER_PREFERENCE)
      - accent-colored 1 px border (DWMWA_BORDER_COLOR)
      - Mica / acrylic backdrop (DWMWA_SYSTEMBACKDROP_TYPE)
      - non-client area rendering incl. drop shadow (DWMWA_NCRENDERING_POLICY)

    Also forces a SWP_FRAMECHANGED refresh so the new attributes take
    effect on an already-visible window.

    Logs each attribute call's HRESULT at DEBUG level. No-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    try:
        dwmapi = ctypes.windll.dwmapi
        user32 = ctypes.windll.user32
    except (OSError, AttributeError) as e:
        _dwm_log.debug("dwmapi/user32 unavailable: %s", e)
        return

    def _set(name: str, attr: int, value: int, ctype=ctypes.c_int) -> None:
        v = ctype(value)
        hr = dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
        _dwm_log.debug("DwmSetWindowAttribute %s(%d)=%d -> HRESULT=0x%08x", name, attr, value, hr & 0xFFFFFFFF)

    DWMWA_NCRENDERING_POLICY = 2
    DWMWA_WINDOW_CORNER_PREFERENCE = 33
    DWMWA_BORDER_COLOR = 34
    DWMWA_SYSTEMBACKDROP_TYPE = 38
    DWMNCRP_DISABLED = 1
    DWMWCP_DONOTROUND = 1
    DWMWA_COLOR_NONE = 0xFFFFFFFE
    DWMSBT_NONE = 1

    _set("NCRENDERING_POLICY", DWMWA_NCRENDERING_POLICY, DWMNCRP_DISABLED)
    _set("CORNER_PREFERENCE", DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND)
    _set("BORDER_COLOR", DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE, ctypes.c_uint)
    _set("SYSTEMBACKDROP_TYPE", DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_NONE)

    # Force the window to re-evaluate its frame so the attributes above
    # take effect on an already-shown window.
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOZORDER = 0x0004
    SWP_FRAMECHANGED = 0x0020
    SWP_NOACTIVATE = 0x0010
    ok = user32.SetWindowPos(
        hwnd, 0, 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE,
    )
    _dwm_log.debug("SetWindowPos(SWP_FRAMECHANGED) -> %s", bool(ok))


class OverlayWindow(QWidget):
    # The window's shown state flipped (Ctrl+Alt+O, the tray flyout's
    # Subtitles tile). Non-spontaneous show/hide only, and not placement's
    # flag change (see set_interactive): the tray's icon and flyout follow.
    visibleChanged = Signal(bool)

    def __init__(
        self,
        style: dict | None = None,
        position: dict | None = None,
        on_position_changed: Callable[[dict], None] | None = None,
        transcript_enabled: bool = False,
        reduce_motion: bool = False,
    ):
        super().__init__(None)
        self._style = dict(style or STYLE_DEFAULTS)
        # Transcript consent (logging.transcript_log_enabled, plumbed in by
        # build_overlay; default OFF). Gates the raw EN text in add_subtitle's
        # render sentinel: default logs must carry no transcript content --
        # main-overlay.log ships in the collect-diagnostics zip and
        # bug_report.yml promises it holds no dialogue.
        self._transcript_enabled = bool(transcript_enabled)
        # clamp_position fills the default anchor keys for a None/partial
        # dict, so _reposition_stack can index h_offset_pct/v_offset_pct
        # unconditionally. (load_overlay_config already clamps the config
        # path; this guards direct construction and Task 7 set_position.)
        self._position = clamp_position(position or {})
        # interactive=False == input-disabled == click-through (the shipped
        # default, matching the old locked-by-default behaviour). Placement
        # mode (Task 7) flips this via set_interactive.
        self._interactive: bool = False
        self._on_position_changed = on_position_changed
        # The position as last persisted (or as loaded at boot). The quit-time
        # save writes only when the live position differs from it: an
        # unconditional write would prune a legacy overlay.geometry rect that
        # boot deliberately kept on disk.
        self._saved_position = dict(self._position)
        # The active PlacementController, if any (Task 7). When set, the
        # window's mouse/key/deactivate handlers delegate to it; when None
        # the window is a passive click-through overlay. Duck-typed to keep
        # this module free of a placement import.
        self._placement_delegate = None
        # WindowTransparentForInput makes the whole window click-through. If
        # we ever add an interactive child widget (inline settings panel etc.),
        # remove this flag and gate passthrough per-widget instead — otherwise
        # the child will be unclickable with no indication why.
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.NoDropShadowWindowHint
        )
        # Full transparency belt-and-suspenders: WA_TranslucentBackground alone
        # can still leave a platform-painted background on Windows with the
        # Qt.Tool flag. Adding WA_NoSystemBackground, disabling auto-fill, and
        # a transparent stylesheet ensures the empty window is fully invisible.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        self.setWindowTitle("Yamabiko")
        icon_path = paths.app_icon_path()
        if icon_path.is_file():
            icon = QIcon(str(icon_path))
            if icon.isNull():
                _log.warning("window icon at %s is unreadable (Qt returned null)", icon_path)
            else:
                self.setWindowIcon(icon)

        # One painted widget draws the whole subtitle stack (plates, text,
        # banner) and owns line lifetimes and motion; this window keeps the
        # screen pin, the anchor math, placement and the render sentinel.
        self._stack = SubtitleStack(self, style=self._style)
        self._stack.set_reduce_motion(reduce_motion)
        # The persistent sample (placement / Settings preview): a stack line
        # id with no hold timer; max_lines never evicts it.
        self._sample_id: int | None = None
        self._genre = "variety"
        self._paused = False
        self._series: tuple = (None, "", "")
        # Problem notices (plan phase 4): the problem line of each active
        # kind, and the recovery lines showing (kind -> (text, its timer)).
        self._problem_lines: dict[str, str] = {}
        self._recoveries: dict[str, tuple] = {}
        self._refresh_banner()

        self._pin_to_screen()
        self._reposition_stack()
        self._chrome_suppressed = False
        # Last shown state visibleChanged reported (see _note_shown), and
        # True while set_interactive swaps the window flags: that hide and
        # re-show is not a visibility change (Decision 36).
        self._shown = False
        self._reflagging = False
        # The periodic z-order / topmost instrumentation timer (Change B).
        # Created lazily on first showEvent; kept here so _start_zorder_probe
        # can guard against creating a second one on a re-entered showEvent.
        self._zorder_timer: QTimer | None = None

    def showEvent(self, event):
        super().showEvent(event)
        # Recompute the anchor geometry against the ACTUAL pinned screen now
        # that the window is realized. _pin_to_screen stores the pinned QScreen
        # so the pre-show __init__ pass is already correct, but re-running here
        # is a cheap idempotent self-heal for the case where Qt only finalizes
        # the screen association after first show (P2 boot-screen fix).
        self._reposition_stack()
        # Apply DWM chrome suppression after the window is actually shown;
        # some Win11 attributes (backdrop, NCRENDERING_POLICY) are silently
        # ignored if set before first paint. Do it once now, and again
        # after a short delay to cover any redraw that resets the state.
        _suppress_windows11_chrome(int(self.winId()))
        if not self._chrome_suppressed:
            # The retry timer must be a
            # CHILD of the window, not a free-standing singleShot. If the
            # window dies within the 250 ms (quit right after show), Qt
            # destroys the child timer with the widget, so the timeout can
            # never call self.winId() on the deleted C++ object
            # (libshiboken RuntimeError through the event loop).
            retry = QTimer(self)
            retry.setSingleShot(True)
            retry.timeout.connect(
                lambda: _suppress_windows11_chrome(int(self.winId()))
            )
            retry.start(250)
            self._chrome_suppressed = True
        # Begin the low-frequency z-order / topmost instrumentation (Change B).
        # Idempotent: only the first realized show arms the timer.
        self._start_zorder_probe()
        if not event.spontaneous():
            self._note_shown(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        if not event.spontaneous():
            self._note_shown(False)

    def _note_shown(self, shown: bool) -> None:
        """Report a flip of the shown state once. Muted during
        set_interactive's flag change, which reports only its net change
        afterwards."""
        if self._reflagging:
            return
        if shown != self._shown:
            self._shown = shown
            self.visibleChanged.emit(shown)

    def toggle_visible(self) -> None:
        """Show or hide the subtitles (Ctrl+Alt+O's toggle_overlay frame, or
        the flyout's Subtitles tile while the socket is down)."""
        self.setVisible(not self.isVisible())

    def _start_zorder_probe(self) -> None:
        """Arm the periodic z-order / topmost instrumentation timer (once).

        Instrumentation ONLY: this schedules _log_zorder_state, which reads
        and logs the overlay's topmost state; it NEVER changes a window flag
        or re-asserts topmost. It exists to capture WHEN the WS_EX_TOPMOST bit
        is lost or a normal window climbs above the always-on-top overlay (a
        reported, un-root-caused z-order bug) so the next live session leaves
        a timestamped trail in main-overlay.log.

        The QTimer is a CHILD of the window (parent=self) so Qt destroys it
        with the widget on teardown -- a free-standing singleShot/timer could
        fire its callback on the already-deleted C++ window and raise a
        libshiboken RuntimeError through the event loop (the same teardown
        hazard the showEvent retry timer above documents). setWindowFlags
        (set_interactive) recreates only the native HWND, not the QObject
        tree, so the child timer survives that and reads the current winId on
        each tick; guard against arming a duplicate on the re-entered
        showEvent. No-op off win32 (the probe body is win32-only anyway)."""
        if sys.platform != "win32":
            return
        if self._zorder_timer is not None:
            return
        timer = QTimer(self)
        timer.setInterval(4000)  # every 4 s while shown: cheap, low-noise
        timer.timeout.connect(self._log_zorder_state)
        timer.start()
        self._zorder_timer = timer

    def _log_zorder_state(self) -> None:
        """Log the overlay's topmost / z-order state (win32, INFO, best-effort).

        Instrumentation ONLY -- reads state, never mutates it. Logs, once per
        tick:
          * ``topmost`` -- whether ``GetWindowLongW(hwnd, GWL_EXSTYLE)`` still
            carries ``WS_EX_TOPMOST`` (0x8). A drop from 1 to 0 pinpoints when
            the always-on-top bit was lost.
          * ``frontmost`` -- whether ``GetWindow(hwnd, GW_HWNDPREV)`` is null
            (nothing above us in the z-order). 0 with a non-null prev handle
            means a window climbed on top of the overlay.
          * ``fg_hwnd`` / ``fg_class`` / ``fg_is_self`` -- the foreground
            window's handle, its window CLASS name (e.g.
            "Chrome_WidgetWin_1"), and whether it is the overlay itself, for
            context on what took the front. The window TITLE is deliberately
            never read: titles are PII (browser tabs, document names, email
            subjects, password-manager entries) and this log ships verbatim
            in the one-click diagnostics zip.

        Wrapped so it can NEVER crash the Qt event loop, and a no-op when there
        is no real HWND (offscreen/headless QPA hands back winId()==1, an
        invalid handle: IsWindow==0), so it stays silent and side-effect-free
        under tests and any non-GUI host."""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            user32.IsWindow.argtypes = [wintypes.HWND]
            user32.IsWindow.restype = wintypes.BOOL
            user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
            user32.GetWindow.restype = wintypes.HWND
            user32.GetForegroundWindow.argtypes = []
            user32.GetForegroundWindow.restype = wintypes.HWND
            # Read the foreground window's CLASS name, never its title:
            # titles are PII (browser tabs, document names, email subjects,
            # password-manager entries) and this probe's log ships verbatim
            # in the diagnostics zip. Class names (e.g. "Chrome_WidgetWin_1")
            # identify the covering app without exposing any content.
            user32.GetClassNameW.argtypes = [
                wintypes.HWND, wintypes.LPWSTR, ctypes.c_int,
            ]
            user32.GetClassNameW.restype = ctypes.c_int
            # Offscreen/headless QPA returns a non-window id; do nothing so
            # tests (and any non-GUI host) stay silent and side-effect-free.
            if not hwnd or not user32.IsWindow(hwnd):
                return
            GWL_EXSTYLE = -20
            WS_EX_TOPMOST = 0x00000008
            GW_HWNDPREV = 3
            # EXSTYLE fits in a LONG, so GetWindowLongW (not the Ptr form) is
            # correct for this bit test on both 32- and 64-bit.
            exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            topmost = bool(exstyle & WS_EX_TOPMOST)
            # GW_HWNDPREV walks toward the FRONT of the z-order; the frontmost
            # window has no previous sibling, so 0/None == nothing above us.
            prev = user32.GetWindow(hwnd, GW_HWNDPREV)
            # Identify the covering window by class + handle only. Its TITLE
            # is deliberately NOT read (no GetWindowTextW call): titles are
            # PII and this record ships verbatim in the diagnostics zip.
            fg = user32.GetForegroundWindow()
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(fg, cls_buf, 256)
            fg_is_self = bool(fg) and int(fg) == hwnd
            _zorder_log.info(
                "overlay z-order: topmost=%d frontmost=%d exstyle=0x%08x "
                "prev_hwnd=%s fg_hwnd=%s fg_class=%r fg_is_self=%d",
                int(topmost),
                int(not prev),
                exstyle & 0xFFFFFFFF,
                ("0" if not prev else hex(int(prev))),
                ("0" if not fg else hex(int(fg))),
                cls_buf.value,
                int(fg_is_self),
            )
        except Exception as e:
            # Never let an instrumentation probe kill the event loop.
            _zorder_log.debug("z-order probe failed: %s", e)

    def _pin_to_screen(self) -> None:
        """Pin the overlay to a full-monitor strip on the configured screen.

        Resolves position["screen"] against the live screen list by
        QScreen.name(); "primary", an empty/absent name, or a name no longer
        present (monitor unplugged) all fall back to the primary screen. The
        window takes the screen geometry less its bottom row so the subtitle
        stack can be placed anywhere inside it by percent offsets.
        """
        want = self._position.get("screen")
        screen = None
        if isinstance(want, str) and want and want != "primary":
            for s in QGuiApplication.screens():
                if s.name() == want:
                    screen = s
                    break
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        # Drop the geometry watch on the previously pinned screen before
        # re-wiring: _revalidate_screen re-pins on every hotplug signal, so
        # without the disconnect each re-pin would stack another connection
        # and one resolution change would fire the handler N times. Guarded
        # like the dead-QScreen probe (_screen_is_live): RuntimeError is the
        # shiboken already-deleted raise, TypeError is never-connected, and
        # AttributeError covers duck-typed test screens with no signal.
        prev = getattr(self, "_pinned_screen", None)
        if prev is not None:
            try:
                prev.geometryChanged.disconnect(
                    self._on_pinned_screen_geometry_changed
                )
            except (AttributeError, RuntimeError, TypeError):
                pass
        # Store the resolved QScreen so _reposition_stack / _cap_px derive the
        # anchor rect from it directly. Before the window is shown, Qt's
        # self.screen() association can still report the PRIMARY screen rather
        # than the pinned one, so layout computed off self.screen() pre-show
        # would target the wrong monitor on a multi-head setup until the event
        # loop settles. The stored reference is correct from pin time
        # (P2 boot-screen fix).
        self._pinned_screen = screen
        # Remember which position["screen"] value produced this pin so
        # update_position_preview can skip redundant re-pins mid-drag (the
        # QScreen scan + full-screen setGeometry per mouse move are wasted
        # work while the name is unchanged).
        self._pinned_screen_name = want
        if screen is not None:
            # One row short of the screen: Windows takes a shown topmost
            # window whose rect covers its monitor for a full-screen app
            # (SHQueryUserNotificationState answers QUNS_BUSY), and its
            # default Do not disturb rule then hides every notification
            # while subtitles show. The origin stays the screen's, so window
            # px still equal screen px, and no line reaches the bottom row
            # (v_offset_pct tops out at 98 %).
            self.setGeometry(screen.geometry().adjusted(0, 0, 0, -1))
            # Watch THIS screen's own geometryChanged: a same-monitor
            # resolution/DPI change (a game flipping the display mode)
            # updates the screen rect in place and emits ONLY this
            # per-QScreen signal -- none of the three app-level hotplug
            # signals __main__ wires to _revalidate_screen -- so without it
            # the window keeps its old size while the anchor math reads the
            # NEW screen rect and lays subtitles outside the window
            # (bottom_y 1267 on a still-1080-high window after a 1440p
            # switch).
            screen.geometryChanged.connect(
                self._on_pinned_screen_geometry_changed
            )

    @staticmethod
    def _screen_is_live(screen) -> bool:
        """False iff ``screen`` is a QScreen whose C++ object Qt has already
        deleted. Such an object raises RuntimeError (libshiboken) on ANY
        attribute access; a live QScreen -- or a duck-typed stand-in used in
        tests -- answers ``.geometry()`` normally. This is the "is the cached
        pin still usable" probe _active_screen keys its re-pin on.
        """
        try:
            screen.geometry()
            return True
        except RuntimeError:
            return False

    def _active_screen(self):
        """The QScreen the overlay geometry is pinned to.

        Prefer the reference resolved in _pin_to_screen over self.screen():
        before first show Qt's self.screen() association can lag and report the
        primary screen rather than the pinned one, so metrics taken from
        self.screen() pre-show can target the wrong monitor. The stored
        reference is correct from pin time. Fall back to
        self.screen()/primaryScreen() when the pin never resolved a screen
        (headless CI).

        Self-heal for a dead pin: _revalidate_screen only fires on the three
        app-level signals it is wired to (primaryScreenChanged / screenAdded /
        screenRemoved), but some Windows display changes -- a monitor
        sleeping/waking, an in-place DPI or resolution change, an RDP
        reconnect -- delete and recreate the underlying QScreen C++ object
        WITHOUT emitting any of them. That leaves _pinned_screen dangling with
        nothing to refresh it, and every later .geometry() call throws
        'libshiboken: Internal C++ object (QScreen) already deleted'. The
        subtitle paths heal it fully (_heal_dead_pin: re-pin AND re-place);
        this re-pin is the belt for any other access, so the reference heals
        on the very next read instead of raising until a restart.
        """
        pinned = getattr(self, "_pinned_screen", None)
        if pinned is not None and not self._screen_is_live(pinned):
            self._pin_to_screen()
            pinned = getattr(self, "_pinned_screen", None)
        return (
            pinned
            or self.screen()
            or QGuiApplication.primaryScreen()
        )

    def _heal_dead_pin(self) -> None:
        """Re-pin and re-place the stack when the cached QScreen is dead (see
        _active_screen for the display changes that kill it silently). The
        subtitle paths call this first: they never read the screen
        themselves, so without it the window would keep the dead screen's
        geometry, the stack its stale anchor (bottom 1267 on a 1080 screen),
        and no geometryChanged watch would be on the recreated QScreen until
        some unrelated show / position / hotplug event. One liveness probe on
        the live path."""
        pinned = self._pinned_screen
        if pinned is not None and not self._screen_is_live(pinned):
            self._revalidate_screen()

    def _stack_geometry(self) -> tuple[int, int, int, int]:
        """(left, right, bottom, line_cap) of the Center alignment's box: the
        anchor margins that bound a stack centred on h_offset_pct and the
        effective per-line width cap they leave room for.

        The margins are clamped to the screen edges, so at an off-centre
        anchor the box between them (width - left - right) is NARROWER than
        the raw _cap_px(). The lines must be capped to that box, not the raw
        cap: pushing the full cap into a line (e.g. h_offset 90% with a 70%
        cap on 1920 px -> box 864, cap 1344) would let a long subtitle size
        its box to the raw 1344 px and, centred on the raw anchor, hang
        ~480 px off the right edge. The min() narrows the stack at an
        off-centre anchor instead of clipping off the monitor. _line_cap_px
        hands this cap to the stack in Center alignment.
        """
        sg = self._active_screen().geometry()
        cap = self._cap_px()
        h_px = int(sg.width() * self._position["h_offset_pct"] / 100)
        bottom_px = int(sg.height() * self._position["v_offset_pct"] / 100)
        left = max(0, h_px - cap // 2)
        right = max(0, sg.width() - (h_px + cap // 2))
        bottom = max(0, sg.height() - bottom_px)
        return left, right, bottom, min(cap, sg.width() - left - right)

    @property
    def stack(self) -> SubtitleStack:
        return self._stack

    def alignment(self) -> str:
        """The current style's alignment ("left" or "center"), as the stack's
        look resolved it (one validation, in telop_look)."""
        return self._stack.look().alignment

    def _line_cap_px(self) -> int:
        """Width cap for a line (banner gutter included): _cap_px() bounded
        by the room the anchor leaves on-screen. Left: from the left anchor
        to the right edge. Center: the margin box of _stack_geometry."""
        if self.alignment() == "left":
            sw = self._active_screen().geometry().width()
            left = int(sw * self._position["left_offset_pct"] / 100)
            return max(1, min(self._cap_px(), sw - left))
        return self._stack_geometry()[3]

    def _stack_anchor(self) -> tuple[int, int, int]:
        """(anchor_x, bottom_y, cap), window-local px. anchor_x is the
        stack's left edge (left_offset_pct) in Left alignment and its centre
        (h_offset_pct) in Center; bottom_y is v_offset_pct of the height.
        One coordinate basis (the pinned screen's FULL geometry) shared with
        resolve_drag, the tray presets and the legacy migration: the window
        spans that geometry, so window-local px equal screen px and placement
        is WYSIWYG (an availableGeometry basis would offset the stack from
        the drag by the taskbar thickness)."""
        sg = self._active_screen().geometry()
        key = "left_offset_pct" if self.alignment() == "left" else "h_offset_pct"
        anchor_x = int(sg.width() * self._position[key] / 100)
        bottom_y = int(sg.height() * self._position["v_offset_pct"] / 100)
        return anchor_x, bottom_y, self._cap_px()

    def _reposition_stack(self, *, animate: bool = False) -> None:
        """Size the stack to the window and hand it the anchor. Called on
        construction, show, re-pin, position changes (instant) and style
        changes (animated)."""
        try:
            anchor_x, bottom_y, _ = self._stack_anchor()
            cap = self._line_cap_px()
            screen = self._active_screen()
            sg = screen.geometry()
            ag = screen.availableGeometry()
        except RuntimeError:
            # A deleted QScreen during teardown (a deferred hold timer firing
            # as the app quits): nothing to place.
            return
        self._stack.setGeometry(0, 0, self.width(), self.height())
        self._stack.set_anchor(
            anchor_x, bottom_y, cap, sg.width(),
            max(0, ag.y() - sg.y()), ag.y() + ag.height() - sg.y(),
            animate=animate,
        )

    def _revalidate_screen(self) -> None:
        """Re-pin + re-margin after a screen hotplug (screenAdded /
        screenRemoved / primaryScreenChanged). _pin_to_screen already falls
        back to the primary screen when the pinned one vanished; the pinned
        name is kept so the strip returns to it if the monitor comes back."""
        self._pin_to_screen()
        self._reposition_stack()

    def _on_pinned_screen_geometry_changed(self, *_args) -> None:
        """Slot for the pinned QScreen's geometryChanged (wired in
        _pin_to_screen): a same-monitor resolution change re-pins the window
        (the screen less its last row) and re-places the stack. Delegates
        to _revalidate_screen -- the exact re-pin + re-margin path the
        hotplug signals use -- so the two display-change flows can't drift.
        Contained: a display flip must never crash the Qt event loop (same
        stance as __main__'s screen-change handler)."""
        try:
            self._revalidate_screen()
        except Exception:
            _geom_log.exception("pinned-screen geometry-change re-pin failed")

    def _cap_px(self) -> int:
        # Tray._on_style_received now runs _clamp_style on every WS frame, but
        # this stays a belt for direct apply_style callers that bypass the tray
        # (tests, a future in-process sender): coerce_max_width_pct is the one
        # sanitizer for the key (bool/NaN/non-numeric -> the 70% default,
        # finite values clamped into 30..100).
        pct = coerce_max_width_pct(self._style.get("max_width_pct"))
        # Full screen.geometry() width, matching _stack_anchor's single
        # coordinate basis (NOT availableGeometry -- see _stack_anchor).
        return int(self._active_screen().geometry().width() * pct / 100)

    def add_subtitle(self, jp: str, en: str, jp_ruby=None) -> None:
        self._heal_dead_pin()
        self._stack.add_line(_segments(jp, jp_ruby), en,
                             hold_ms=self._stack.look().hold_ms)
        # Native-renderer sentinel. The pipeline's WS broadcast only proves a
        # subtitle frame LEFT the pipeline; it says nothing about whether the
        # shipped Qt overlay actually rendered it -- the overlay is a separate
        # process that could crash or drop the frame while a test-owned WS
        # client stays green. This is the one observable that the real overlay
        # put a subtitle on screen: it lands in main-overlay.log and the frozen
        # default-launch E2E asserts it (tests/test_real_frozen_e2e.py::
        # test_frozen_default_launch_translates_wav_and_overlay_renders_subtitle).
        #
        # Content-free by default: default logs must carry no transcript
        # content (bug_report.yml promises it -- same contract as main.py's
        # drop lines and latency.log's raw fields). en_len / en_ascii_letters
        # keep the E2E's assertion power (a non-empty, Latin-lettered EN was
        # rendered) without writing dialogue to disk; the raw text rides along
        # only behind the SAME consent flag (logging.transcript_log_enabled).
        # Token contract (filter "overlay.subtitle.rendered en_len=", split on
        # "en_ascii_letters=") is pinned by
        # tests/overlay_app/test_overlay_render_marker.py.
        _log.info(
            "overlay.subtitle.rendered en_len=%d en_ascii_letters=%d%s",
            len(en),
            sum(1 for c in en if c.isascii() and c.isalpha()),
            f" en={en!r}" if self._transcript_enabled else "",
        )

    def show_sample(self, jp: str, en: str, jp_ruby=None) -> None:
        """Show a persistent sample line for placement / the Settings live
        preview: lands at once, no hold timer, never evicted by max_lines.
        Replaces any existing sample."""
        self._heal_dead_pin()
        self.hide_sample()
        self._sample_id = self._stack.add_line(_segments(jp, jp_ruby), en,
                                               hold_ms=None, instant=True)

    def hide_sample(self) -> None:
        if self._sample_id is not None:
            sample_id, self._sample_id = self._sample_id, None
            self._stack.remove_line(sample_id, fade=False)

    def apply_style(self, style: dict) -> None:
        self._style = dict(style)
        # Alignment and max width move the anchor and the cap; hand the stack
        # the new look first, then re-place it with motion so a Mode or
        # Alignment change glides.
        self._stack.apply_style(style, retarget=False)
        self._reposition_stack(animate=True)

    def set_reduce_motion(self, on: bool) -> None:
        self._stack.set_reduce_motion(on)

    def set_state(self, genre: str, paused: bool) -> None:
        """Pipeline state frame: genre and pause drive the banner. While
        paused the banner stays visible even with no lines."""
        self._genre = genre if genre in GENRE_BANNER else "variety"
        self._paused = bool(paused)
        self._refresh_banner()

    def set_series(self, media_id, label: str, native: str = "") -> None:
        """Picked series (state frame): its Japanese title replaces the
        genre in the banner; without a known Japanese title the genre stays."""
        self._series = (media_id, label or "", native or "")
        self._refresh_banner()

    def banner_text(self) -> tuple[str, str]:
        """The banner's (Japanese, English) text when not paused: the series
        when one with a Japanese title is picked, otherwise the genre."""
        media_id, label, native = self._series
        if media_id is not None and native:
            return native, series_banner_english(label)
        return GENRE_BANNER[self._genre]

    def _refresh_banner(self) -> None:
        jp, en = PAUSED_BANNER if self._paused else self.banner_text()
        self._stack.set_banner(jp, en, self._paused)

    def set_problems(self, active: list) -> None:
        """The pipeline's active problems (ProblemTracker.changed): one
        problem line per kind, translator above audio. A kind that is active
        again ends its recovery line."""
        lines = {entry["kind"]: entry["line"] for entry in active if is_problem(entry)}
        for kind in lines:
            self._end_recovery(kind, refresh=False)
        self._problem_lines = lines
        self._refresh_notices()

    def show_recovery(self, kind: str, text: str) -> None:
        """Show ``kind``'s recovery line (ink banner) for NOTICE_RECOVERY_MS
        (ProblemTracker.recovered). Ignored for an unknown kind or while that
        kind's problem line shows; a repeat restarts the time."""
        if kind not in KIND_ORDER or not isinstance(text, str) or kind in self._problem_lines:
            return
        self._end_recovery(kind, refresh=False)
        # Child of the window: it dies with it and can never fire on a
        # deleted window.
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda kind=kind: self._end_recovery(kind))
        timer.start(NOTICE_RECOVERY_MS)
        self._recoveries[kind] = (text, timer)
        self._refresh_notices()

    def _end_recovery(self, kind: str, *, refresh: bool = True) -> None:
        entry = self._recoveries.pop(kind, None)
        if entry is None:
            return
        timer = entry[1]
        timer.stop()
        timer.deleteLater()
        if refresh:
            self._refresh_notices()

    def _refresh_notices(self) -> None:
        notices = []
        for kind in KIND_ORDER:
            if kind in self._problem_lines:
                notices.append((kind, self._problem_lines[kind], False))
            elif kind in self._recoveries:
                notices.append((kind, self._recoveries[kind][0], True))
        self._stack.set_notices(notices)

    def current_position(self) -> dict:
        """The current anchor position (screen name + offset percents)."""
        return dict(self._position)

    def subtitle_screen(self) -> QScreen:
        """The monitor the subtitles are pinned to (overlay.position.screen;
        the primary screen when that name is gone): the series bar opens at
        its top center. Reads the pin, never moves or resizes this window."""
        return self._active_screen()

    def set_position(self, pos: dict, *, persist: bool = True) -> None:
        """Re-pin the window to the given anchor position, re-margin the
        stack, and persist via the on_position_changed callback.

        Used by placement mode and the tray position presets. Settings passes
        ``persist=False`` because its transaction already wrote the position;
        every other caller retains the callback-backed persistence path.

        The re-pin/re-margin half is the PREVIEW path; delegate to
        update_position_preview so the two stay in lockstep, then fire the
        persist callback that makes this the PERSIST path.
        """
        self.update_position_preview(pos)
        if not persist:
            # The caller already wrote this position to disk.
            self._saved_position = dict(self._position)
            return
        try:
            self._persist_position()
        except Exception:
            _geom_log.exception("on_position_changed callback failed")

    def save_position_if_changed(self) -> None:
        """Persist the live position only if it differs from the last saved
        one. The quit path uses this, so a placement drag still in progress is
        kept while an untouched position is not rewritten. Raises whatever the
        persist callback raises; the caller owns error handling.
        """
        if self._position != self._saved_position:
            self._persist_position()

    def _persist_position(self) -> None:
        # No callback means persistence is off (an explicit --config run, or
        # a window built without one): nothing is written.
        if self._on_position_changed is None:
            return
        self._on_position_changed(dict(self._position))
        self._saved_position = dict(self._position)

    def update_position_preview(self, pos: dict) -> None:
        """Re-pin + re-margin to ``pos`` WITHOUT firing the persist callback.

        Placement mode's live drag calls this on every mouse move. Routing
        through set_position instead would fire on_position_changed ->
        write_overlay_position on every pixel of the drag, hammering the
        disk and the cross-process overrides filelock. This is the PREVIEW
        path; set_position is the PERSIST path. Placement persists exactly
        once, on save.
        """
        self._position = clamp_position(pos)
        # The drag fires this per mouse move but the target screen almost
        # never changes mid-drag; re-pin (linear QScreen scan + full-screen
        # setGeometry) only when the requested screen name differs from the
        # one the window is already pinned for. Hotplug while the name is
        # unchanged is _revalidate_screen's job -- it re-pins unconditionally
        # off the QGuiApplication screen signals.
        if self._position.get("screen") != self._pinned_screen_name:
            self._pin_to_screen()
        self._reposition_stack()
        self.update()

    def set_interactive(self, interactive: bool) -> None:
        """Enable / disable input for the window (placement mode).

        interactive=True drops WindowTransparentForInput so the window
        receives mouse events (Task 7 drags the stack); interactive=False
        restores click-through. Same flag/DWM/show dance as the retired
        set_locked, with the lock semantics inverted, and NO persistence
        (placement mode owns persistence via set_position).
        """
        if interactive == self._interactive:
            return
        self._interactive = interactive
        flags = self.windowFlags()
        if interactive:
            flags &= ~Qt.WindowTransparentForInput
        else:
            flags |= Qt.WindowTransparentForInput
        # setWindowFlags() destroys and recreates the native HWND on
        # Windows; the DWM attributes applied to the old HWND die with it.
        # Re-arm the showEvent once-gate so the re-entered showEvent
        # schedules the 250 ms second suppression pass for the recreated
        # window (the immediate pass alone is documented-insufficient,
        # see the showEvent comment).
        self._chrome_suppressed = False
        # setWindowFlags hides the window and show() re-shows it: not a
        # visibility change, so visibleChanged stays quiet (the tray icon and
        # tooltip would flash "Subtitles hidden"); only a net change counts
        # (a hidden window shown for placement).
        self._reflagging = True
        try:
            self.setWindowFlags(flags)
            self.show()
        finally:
            self._reflagging = False
        self._note_shown(self.isVisible())
        self.update()

    # --- placement-mode input delegation (Task 7) ---
    #
    # When a PlacementController is active, the window forwards drags and
    # keys to it. Off placement the window carries WindowTransparentForInput
    # (set_interactive(False)), so these handlers never fire in normal use;
    # the guards keep them inert if they ever do.

    def mousePressEvent(self, event):
        d = self._placement_delegate
        if d is not None and d.active:
            d.on_drag()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        d = self._placement_delegate
        if d is not None and d.active:
            d.on_drag()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        d = self._placement_delegate
        if d is not None and d.active:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                d.save()
                event.accept()
                return
            if key == Qt.Key_Escape:
                d.cancel()
                event.accept()
                return
        super().keyPressEvent(event)

    def event(self, event):
        # Click-away (focus lost to another window) commits the placement,
        # matching the tray-preset "one click and done" feel. Guarded on the
        # delegate being active so a WindowDeactivate emitted during the
        # set_interactive show() dance is ignored.
        if event.type() == QEvent.WindowDeactivate:
            d = self._placement_delegate
            if d is not None and d.active:
                d.save()
        return super().event(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._interactive:
            painter = QPainter(self)
            pen = QPen(QColor(255, 255, 255, 180))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(1, 1, -2, -2))
