"""System tray icon for the overlay app: the controller of the tray UI.

Left click opens the flyout (tray_flyout.TrayFlyout, placed above the icon
by tray_placement): Pause or Resume, the Subtitles and Mode tiles, Genre,
the series row, Position, Alignment, Move subtitles..., and the footer
(Settings, Troubleshooting with Open log folder and Save diagnostics,
Quit). Right click shows the short menu above the taskbar (tray_placement):
Pause or Resume, Pick a series... (series feature on), Settings..., Quit
Yamabiko, each with the user's current shortcut after a tab.

Commands go to the pipeline through a `PipelineClient`. State frames
(genre, pause, series), style frames (mode, alignment), the overlay
window's visibility and the active problems (ProblemTracker) keep the menu,
the flyout, the state icon and the tooltip current. Every notification
goes through show_toast.

The series bar (series_palette.SeriesPalette) shows a pick's build: the
tray forwards the build's progress, success and failure to its status, and
the bar's Clear takes the flyout's Clear path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .. import paths
from ..config_writer import write_overlay_style
from ..diagnostics import collect_diagnostics
from ..gazetteer.anilist import AniListClient, AniListUnavailable
from ..gazetteer.gazetteer import (
    SeriesGazetteer,
    build_series_gazetteer,
    cache_needs_rebuild,
    clear_active_series,
    gazetteer_path,
    save_active_series,
)
from ..gazetteer.series_profile import build_profile, profile_path
from ..hotkeys import HOTKEY_KEYS, combo_label
from ..lifecycle.jobs import show_in_folder_outside_job
from . import tray_icons, tray_placement
from .position_config import DEFAULT_POSITION
from .preview_monitor import SAMPLE_NEWEST
from .settings_window import SettingsWindow
from .style_defaults import STYLE_CHOICES, STYLE_DEFAULTS
from .theme import menu_qss, palette_for
from .tray_flyout import GENRE_OPTIONS, TrayFlyout, live_screens, position_entries, series_title
from .window_chrome import SquareCorners, take_foreground
from .ws_client import PipelineClient

if TYPE_CHECKING:
    from .series_palette import SeriesPalette

# The genre keys the flyout's Genre segments carry.
_GENRES = tuple(key for key, _label in GENRE_OPTIONS)
_log = logging.getLogger(__name__)

# The right-click menu's width (the mockup's 224 px). The shared menu QSS
# leaves widths to each menu: the flyout's Troubleshooting menu is 200 px.
_MENU_MIN_WIDTH = 224

# "No series": _series with none picked, and the _pending_series of a Clear
# made while the socket was down.
_NO_SERIES: tuple = (None, "", "")

# Shell_NotifyIcon keeps 127 UTF-16 units of a tooltip (szTip[128] WCHARs);
# the series title gives way first (Decision 61).
_TIP_LIMIT = 127


def _utf16_len(text: str) -> int:
    """Length in UTF-16 units: a character outside the BMP counts two."""
    return len(text.encode("utf-16-le")) // 2


# The sample subtitle shown on the live overlay behind the open Settings
# window: the Settings monitor's newest line, so both previews read the same.
_PREVIEW_JP, _PREVIEW_EN, _PREVIEW_RUBY = SAMPLE_NEWEST


class _DiagnosticsThread(QThread):
    """Runs collect_diagnostics() off the Qt GUI thread.

    Collection can block for seconds: the snapshot's nvidia-smi probe
    has a 5 s timeout and deflating tens of MB of logs adds more.
    Running it synchronously in the menu handler froze the tray menu
    and the overlay repaint for the duration. Mirrors
    ``_UpdateCheckThread`` in ``overlay_app.__main__``: QThread
    subclass + typed Signal, so the result crosses back to the GUI
    thread as a queued slot call.
    """

    # Emits the returned zip path (as str -- it may not exist on
    # failure; the slot exists()-checks it, same contract as before).
    finishedWithPath = Signal(str)

    def run(self) -> None:  # worker thread
        # collect_diagnostics never raises (its documented contract);
        # no try/except here -- a raise would be a contract bug that
        # must surface loudly, not vanish in a worker thread.
        self.finishedWithPath.emit(str(collect_diagnostics()))


class _GazetteerBuildThread(QThread):
    """Run the full per-pick AniList build off the GUI thread and cache its
    gazetteer: build_series_gazetteer fetches the picked media's characters
    PLUS the relations carry (a related season's cast) and persists the
    series profile (profile-<media_id>.json, the series_context_hint input)
    from the same intel request. The profile build accompanies EVERY pick
    path: a fresh gazetteer cache missing only its profile gets just the
    profile backfilled (one intel request, best-effort).

    The cache is freshness-gated, not existence-gated (P1-9): a cache
    that is pre-versioning (v1 flat map) or older than 24 h reads as a
    rebuild -- an airing series' cast grows, and extractor fixes must
    reach old caches -- with the OLD file kept as the offline fallback
    when the rebuild fails (a stale cache beats the cold-offline "series
    not changed" outcome). Network happens only here, on a user pick;
    there is deliberately no background/startup refresh.

    Emits built(media_id, label) on success (incl. the fresh-cache hit,
    which is instant and offline-friendly, and the rebuild-failure
    fallback onto an existing cache), or buildFailed(media_id, label,
    error) if the build failed with NO usable cache on disk, so the
    series is NOT switched to a hint-less state silently. ``error`` is
    the raised exception; the tray's slot words the notification by its
    class (network-class vs everything else). While a rebuild runs it emits
    progress(media_id, fraction), the build's staged progress (0.15 after
    the intel request, then the cast pages and the relations carry); a
    fresh-cache hit and the profile backfill emit none."""
    built = Signal(int, str)
    buildFailed = Signal(int, str, object)
    progress = Signal(int, float)

    def __init__(self, media_id: int, label: str, parent=None) -> None:
        super().__init__(parent)
        self._media_id = media_id
        self._label = label

    def run(self) -> None:  # worker thread
        try:
            path = gazetteer_path(self._media_id)
            if cache_needs_rebuild(path):
                # Cache absent, pre-versioning, or past the 24 h freshness
                # horizon: (re)build from AniList.
                gaz = None
                try:
                    client = AniListClient()
                    try:
                        gaz = build_series_gazetteer(
                            client, self._media_id,
                            on_progress=self._report_progress,
                        )
                    finally:
                        client.close()
                except Exception:
                    # Only a cache that actually LOADS is a fallback:
                    # gating on the file's existence let a CORRUPT file
                    # emit built and silently switch to a hint-less
                    # series (load() also quarantines the corrupt file
                    # aside, so the next pick refetches).
                    if SeriesGazetteer.load(path) is None:
                        raise
                    # A STALE cache beats a failed pick: keep serving the
                    # old file (an offline re-pick of a cached series must
                    # not degrade to "series not changed"; cold-offline,
                    # with no usable cache, still does -- the raise above).
                    # The rebuild retries on the next pick.
                    # No media_id in these logs: it identifies the series the
                    # user is watching, and main-overlay.log ships in the
                    # diagnostics zip.
                    _log.warning(
                        "gazetteer rebuild failed; keeping the existing "
                        "cache as fallback",
                        exc_info=True)
                # Only cache a NON-EMPTY result. A genuine empty-200 (the
                # series has no characters yet) must not write an empty {}
                # cache: the freshness gate would then short-circuit picks
                # for 24 h without ever retrying the fetch. Still emit
                # built -- "no hints yet" is a valid outcome, just not a
                # poisoned cache. (The series profile is persisted INSIDE
                # the build, unconditionally: it has no freshness gate, so
                # an empty profile heals on the next pick.)
                if gaz is not None and gaz.map:
                    gaz.save(path)
            elif not profile_path(self._media_id).is_file():
                # Missing-profile heal beside a FRESH cache: a corrupt
                # profile load_profile quarantined aside within the cache's
                # freshness window. Without this branch profile-<id>.json
                # would not reappear until the cache goes stale and
                # series_context_hint would stay inert for the series.
                # Backfill JUST the profile (one intel request); never
                # refetch the cast the fresh-cache hit exists to skip.
                # Best-effort: a failed backfill degrades to the plain
                # cache hit (hints still work, the still-missing profile
                # retries on the next pick) rather than failing an
                # offline-friendly pick.
                try:
                    client = AniListClient()
                    try:
                        build_profile(client, self._media_id)
                    finally:
                        client.close()
                except Exception:
                    _log.warning(
                        "series-profile backfill failed; proceeding with the "
                        "cached gazetteer",
                        exc_info=True)
            self.built.emit(self._media_id, self._label)
        except Exception as exc:
            _log.exception("gazetteer build failed")
            self.buildFailed.emit(self._media_id, self._label, exc)

    def _report_progress(self, fraction: float) -> None:  # worker thread
        self.progress.emit(self._media_id, fraction)


def _make_icon() -> QIcon:
    """The shipped Enhanced app icon (assets/icon.ico): the image of every
    notification (Decision 24). A solid-color placeholder when the file is
    missing."""
    icon_path = paths.app_icon_path()
    if icon_path.is_file():
        return QIcon(str(icon_path))
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(40, 120, 220))
    return QIcon(pixmap)


def _with_combo(label: str, combo: str) -> str:
    """A short-menu label with its shortcut column: QMenu right-aligns the
    text after a tab. Text only, no QAction shortcut: the menu binds nothing
    (the pipeline owns the global shortcuts)."""
    return f"{label}\t{combo}" if combo else label


def _style_choice(style: dict, key: str) -> str:
    """style[key] when it is one of the choices, else its default (the
    flyout's setters accept only the choices)."""
    value = style.get(key)
    return value if value in STYLE_CHOICES[key] else STYLE_DEFAULTS[key]


class Tray(QObject):
    def __init__(
        self,
        client: PipelineClient,
        on_quit,
        overlay_window=None,
        on_change_series=None,
        parent: QObject | None = None,
        palette: dict | None = None,
        series_palette: "SeriesPalette | None" = None,
        style_seed: dict | None = None,
        auto_genre: bool = True,
        config_path: Path | None = None,
        settings_config: dict | None = None,
        settings_read_only: bool = False,
        reduce_motion: bool = False,
        shortcuts: dict | None = None,
    ):
        super().__init__(parent)
        self._client = client
        self._overlay_window = overlay_window
        self._config_path = (
            config_path if config_path is not None else paths.user_overrides_path()
        )
        self._settings_config = settings_config
        self._settings_read_only = bool(settings_read_only)
        # The series-search palette (gazetteer feature only). build_overlay
        # hands the reference here so apply_palette can re-skin it on a live
        # theme flip, the same way the open settings window is re-skinned.
        # None when the gazetteer feature is off.
        self._series_palette = series_palette
        # Chrome theme palette. build_overlay resolves overlay.theme once at
        # boot and hands the palette here; a None default resolves the system
        # theme internally so older callers and tests still get a skinned menu.
        self._palette = palette if palette is not None else palette_for("system")
        # Callable that summons the series bar (palette.summon, injected by
        # build_overlay). None in unit tests that don't wire it.
        self._on_change_series = on_change_series
        # The active on-screen placement session (Task 7), constructed
        # lazily by "Move subtitles...". None when not placing.
        self._placement = None
        # The single open SettingsWindow (Task 10), or None. Guarded so a
        # second Settings click raises the existing window instead of
        # stacking a new one.
        self._settings_window: SettingsWindow | None = None
        # True while a live style preview is on screen (a sample subtitle is
        # showing). Set on the first stylePreview, cleared on previewEnded.
        self._preview_active = False
        # The last style dict a stylePreview carried this session, or None.
        # Cached so a placement session that BORROWS the single sample slot
        # (both draw through OverlayWindow.show_sample) can restore the settings
        # live-preview on exit (_on_placement_exit). See P1-14.
        self._preview_style: dict | None = None
        # True when the settings window closed WHILE a placement session was
        # active with a live preview on the overlay: the close couldn't
        # restore the committed style (placement owned the sample slot), so
        # the restore is OWED and _on_placement_exit settles it exactly once.
        self._style_restore_owed = False
        self._paused = False
        self._genre = "variety"
        # Seed the known-applied style from the config-loaded overlay style
        # (build_overlay passes overlay_cfg.style here), so it matches the
        # style the OverlayWindow was actually constructed with. Seeding from
        # STYLE_DEFAULTS instead let a WS-down preview+close (_on_preview_ended)
        # reskin the live overlay to FACTORY defaults, silently clobbering the
        # user's configured style until a WS echo arrived. None (older callers,
        # unit tests) falls back to STYLE_DEFAULTS for compatibility.
        self._current_style: dict = (
            dict(style_seed) if style_seed else dict(STYLE_DEFAULTS)
        )
        # Style applied while the WS was down, awaiting a flush
        # to the pipeline on reconnect (see _send_style_change).
        self._pending_style: dict | None = None
        self._pending_hotkeys: dict | None = None  # shortcuts saved while the socket was down
        self._pending_incremental: bool | None = None  # Faster subtitles saved while the socket was down
        # Genre picked while the WS was down, awaiting the
        # same reconnect flush (see _on_genre_chosen).
        self._pending_genre: str | None = None
        # Monotonic count of MANUAL genre picks this session
        # (_on_genre_chosen). on_series_chosen snapshots it into
        # _series_pick_genre_seq so _on_gazetteer_built can tell whether a
        # manual pick landed DURING the build window (the AniList fetch is
        # multi-second on a slow connection). The invariant is "a manual
        # genre change after a pick sticks until the next explicit pick" --
        # and the pick happens at palette-selection time, not built-signal
        # time, so a manual pick inside that window is chronologically AFTER
        # the pick and must not be clobbered by the auto-genre ride-along.
        self._manual_genre_seq = 0
        self._series_pick_genre_seq = 0
        # When True (gazetteer.auto_genre, threaded in by build_overlay), a
        # series pick rides a set_genre "anime" along with set_series
        # (_on_gazetteer_built); the settings-window checkbox flips it live
        # via set_auto_genre.
        self._auto_genre = bool(auto_genre)
        # overlay.reduce_motion, sent with every apply_style so the browser
        # page follows Settings (_on_reduce_motion_changed keeps it current).
        self._reduce_motion = bool(reduce_motion)
        # In-flight diagnostics worker (see _on_collect_diagnostics).
        # The reference keeps the running QThread alive; it doubles as
        # the "already collecting" re-click guard.
        self._diag_thread: _DiagnosticsThread | None = None
        # Series picked or cleared while the WS was down, awaiting the
        # reconnect flush (see _on_gazetteer_built / _on_series_clear /
        # _on_connected_changed); _NO_SERIES is a clear.
        self._pending_series: tuple[int | None, str, str] | None = None
        # In-flight gazetteer-build worker (see on_series_chosen). The
        # reference keeps the running QThread alive AND doubles as the
        # "already building" re-entrancy guard (mirrors _diag_thread).
        self._gaz_thread: _GazetteerBuildThread | None = None
        # The LATEST palette pick made while a build was still in flight
        # ((media_id, label, native)), or None. Unlike the diagnostics
        # re-click -- where a repeat click means "same request" -- a second
        # pick is a DIFFERENT request, and the series bar already shows that
        # pick's status, so dropping it would look like the pick took and
        # then the finished build would activate the WRONG series (auto-genre
        # ride-along included). on_series_chosen queues it; the running
        # build's completion slots start it and discard their own, now
        # stale, activation (_consume_queued_pick).
        self._queued_pick: tuple[int, str, str] | None = None
        # A Clear landed while a build's outcome was still undelivered: its
        # completion slots discard it (the Clear is the newer intent). A new
        # build's start resets it (_start_gazetteer_build).
        self._build_superseded = False
        # The active series (media_id, label, native) as the pipeline last
        # reported it, or as this tray just picked or cleared it; the
        # flyout's series row shows it.
        self._series: tuple = _NO_SERIES
        # The overlay window's shown state (build_overlay wires
        # OverlayWindow.visibleChanged to set_overlay_visible).
        self._overlay_visible = True
        # The combos the short menu and the flyout tooltips show (Decision
        # 33); None while hotkeys.enabled is false (hotkeys.shortcut_labels).
        self._shortcuts: dict[str, str] | None = (
            dict(shortcuts) if shortcuts is not None else None
        )
        # The active problems (ProblemTracker.changed, display order): the
        # icon's amber dot, the tooltip lines, the flyout's header status.
        self._problems: list[dict] = []
        # The click handler of the toast currently on screen, or None.
        # QSystemTrayIcon.messageClicked does not say WHICH toast was
        # clicked, so show_toast disconnects the previous handler before
        # every new toast (see show_toast).
        self._toast_click_handler = None

        # The icon shows the state (Decisions 14 and 23) and the tooltip the
        # status sentence (Decision 15); _refresh_status sets both before
        # the icon is shown.
        self._tray = QSystemTrayIcon(self)
        self._icon_key: tuple | None = None
        # Every notification shows the shipped app icon (Decision 24).
        self._toast_image = _make_icon()
        # The taskbar's light / dark (the problem dot's ring). The watcher
        # re-reads it on WM_SETTINGCHANGE "ImmersiveColorSet", which reaches
        # the flyout's native top-level window (built below).
        self._taskbar = tray_icons.TaskbarThemeWatcher(self)
        self._taskbar.changed.connect(self._on_taskbar_theme_changed)
        # The screen the icon is drawn for (see _watch_icon_screen).
        self._icon_screen_watched = None

        # Left click: the flyout (Decisions 7 to 11). Built once at boot and
        # kept: its first show is the slow one, and its native window exists
        # from the start (TrayFlyout calls winId()). It hides itself before it
        # emits an action that closes it.
        self._flyout = TrayFlyout(
            self._palette, show_series=on_change_series is not None,
        )
        # Swallows the Trigger of the click that closed the flyout (Decision 9).
        self._debounce = tray_placement.TriggerDebounce()
        flyout = self._flyout
        flyout.pauseRequested.connect(self._send_pause)
        flyout.subtitlesToggleRequested.connect(self._on_toggle_subtitles)
        flyout.modeChosen.connect(self._on_mode_chosen)
        flyout.genreChosen.connect(self._on_genre_chosen)
        flyout.alignmentChosen.connect(self._on_alignment_chosen)
        flyout.changeSeriesRequested.connect(self._on_change_series_clicked)
        flyout.clearSeriesRequested.connect(self._on_series_clear)
        flyout.positionPresetChosen.connect(self._on_position_preset)
        flyout.moveSubtitlesRequested.connect(self._on_move_subtitles)
        flyout.settingsRequested.connect(self._on_settings_requested)
        flyout.openLogFolderRequested.connect(self._on_open_log_folder)
        flyout.saveDiagnosticsRequested.connect(self._on_collect_diagnostics)
        flyout.quitRequested.connect(self._on_quit_requested)
        flyout.dismissed.connect(self._on_flyout_dismissed)
        self._tray.activated.connect(self._on_activated)

        # Right click: the short menu (Decision 12), popped up by
        # _on_activated. The shortcut column is menu text (_refresh_menu_text).
        self._menu = QMenu()
        # Skin the menu from the boot palette; apply_palette() re-runs this
        # on a live theme change.
        self._menu.setStyleSheet(menu_qss(self._palette))
        self._menu.setMinimumWidth(_MENU_MIN_WIDTH)
        SquareCorners(self._menu)  # nearly square, as screenshots 13 and 14
        self._pause_action = QAction("Pause", self)
        self._pause_action.triggered.connect(self._on_pause_toggled)
        self._menu.addAction(self._pause_action)
        # Pick a series... only with the series feature on: on_change_series
        # carries the palette summon (build_overlay passes None when the
        # feature is off, so no series UI dangles).
        self._pick_series_action: QAction | None = None
        if on_change_series is not None:
            self._pick_series_action = QAction("Pick a series\u2026", self)
            self._pick_series_action.triggered.connect(self._on_change_series_clicked)
            self._menu.addAction(self._pick_series_action)
        self._menu.addSeparator()
        settings_action = QAction("Settings\u2026", self)
        settings_action.triggered.connect(self._on_settings_requested)
        self._menu.addAction(settings_action)
        self._menu.addSeparator()
        quit_action = QAction("Quit Yamabiko", self)
        quit_action.triggered.connect(self._on_quit_requested)
        self._menu.addAction(quit_action)
        # Keep a ref to the outer "on_quit" callable -- this is what
        # QApplication.quit bound to when the tray was constructed.
        # Fired after we've sent the pipeline quit command + given it
        # a moment to drain.
        self._on_quit_cb = on_quit

        self._sync_style_controls()
        self._refresh_shortcut_labels()
        self._refresh_status()
        self._tray.show()
        # Redraw the icon at the right size when a screen comes or goes, the
        # primary changes, or the icon's screen changes resolution or scale.
        app = QGuiApplication.instance()
        app.screenAdded.connect(self._on_screens_changed)
        app.screenRemoved.connect(self._on_screens_changed)
        app.primaryScreenChanged.connect(self._on_screens_changed)
        self._watch_icon_screen()

        client.stateReceived.connect(self._on_state)
        client.styleReceived.connect(self._on_style_received)
        client.connectedChanged.connect(self._on_connected_changed)
        # The series listeners only matter when the series UI exists.
        if on_change_series is not None:
            client.seriesReceived.connect(self._on_series_received)
            if series_palette is not None:
                # A better cover (the AniList color, then the poster) for the
                # series the flyout shows (R3).
                series_palette.coverChanged.connect(self._on_cover_changed)
                # The series bar's chip Clear is the flyout's Clear (L8).
                series_palette.clearRequested.connect(self._on_series_clear)
        client.noticeReceived.connect(self._on_pipeline_notice)

    def show_toast(
        self,
        title: str,
        body: str,
        timeout_ms: int = 8000,
        on_click=None,
    ) -> None:
        """Show one Windows notification; every toast in the overlay goes
        through here. Its image is the shipped app icon, problems included
        (Decision 24).

        messageClicked carries no toast identity, so a handler left
        connected by an earlier toast (the update notice that timed out
        unclicked) would fire on a click on any later toast. The previous
        handler is therefore disconnected on EVERY call, and ``on_click``
        (if given) is connected for this toast only, firing at most once:
        the newest toast's handler wins.
        """
        self._clear_toast_click_handler()
        if on_click is not None:
            def _clicked() -> None:
                self._clear_toast_click_handler()
                on_click()

            self._toast_click_handler = _clicked
            self._tray.messageClicked.connect(_clicked)
        self._tray.showMessage(title, body, self._toast_image, timeout_ms)

    def _clear_toast_click_handler(self) -> None:
        handler = self._toast_click_handler
        if handler is None:
            return
        self._toast_click_handler = None
        try:
            self._tray.messageClicked.disconnect(handler)
        except (RuntimeError, TypeError):
            pass

    def _on_pipeline_notice(self, title: str, body: str, settings_page: str = "") -> None:
        """Toast a notice frame the pipeline sent (a shortcut it couldn't
        bind). A notice naming a Settings page opens Settings there when
        clicked."""
        on_click = (lambda: self._open_settings(page=settings_page)) if settings_page else None
        self.show_toast(title, body, on_click=on_click)

    def _on_pause_toggled(self) -> None:
        """The short menu's Pause / Resume: the opposite of the current state."""
        self._send_pause(not self._paused)

    def _send_pause(self, pause: bool) -> None:
        """Pause (True) or resume: the short menu, the flyout's big button.

        The send_command bool is deliberately ignored here (mirrors the
        best-effort rationale on quit). Pause/resume is a TRANSIENT runtime
        command, not durable intent: a disconnected WS means the pipeline
        process is down or restarting, a fresh pipeline starts unpaused, and
        replaying a stale pause on reconnect would silently mute subtitles
        with no fresh user action. The labels never flip locally -- they
        follow authoritative state broadcasts only, and the server's
        on-connect state snapshot re-syncs them -- so a dropped click is
        visible to the user (the label doesn't change) and self-consistent.
        """
        self._client.send_command(
            {"type": "command", "action": "pause" if pause else "resume"}
        )

    def _on_genre_chosen(self, genre: str) -> None:
        # A manual pick: bump the recency counter so an in-flight series
        # pick's auto-genre ride-along (_on_gazetteer_built) knows this pick
        # is newer than the series pick and must stick.
        self._manual_genre_seq += 1
        sent = self._client.send_command(
            {"type": "command", "action": "set_genre", "genre": genre}
        )
        if sent:
            # A successful send supersedes any genre still queued from an
            # earlier offline pick (same invariant as the style
            # supersede): without this, the next reconnect would replay
            # the stale pick over this newer one.
            self._pending_genre = None
            return
        # Unlike pause, a genre pick is DURABLE user intent
        # -- the flyout's Genre segment already shows it as selected, and the
        # server's on-connect state snapshot would otherwise silently
        # revert the radio on reconnect (the silent-no-op class).
        # Queue it for the reconnect flush; latest offline pick wins. No
        # local persistence: tray genre is session-scoped by design (the
        # pipeline's set_genre handler doesn't persist it either).
        self._pending_genre = genre

    def set_auto_genre(self, enabled: bool) -> None:
        # The settings-window checkbox sink (autoGenreChanged); takes effect
        # on the next series pick.
        self._auto_genre = bool(enabled)

    # -- series selection -------------------------------------------------

    def _on_series_received(self, media_id, label: str, native: str = "") -> None:
        # Authoritative series from the pipeline (on connect / after a change).
        self._set_series(media_id, label, native)

    def _set_series(self, media_id, label: str = "", native: str = "") -> None:
        self._series = (
            _NO_SERIES if media_id is None else (media_id, label or "", native or "")
        )
        if self._series_palette is not None:
            # The series bar's chip and its saved poster follow the active
            # series (L8, L11).
            self._series_palette.set_current_series(media_id, label or "")
        self._refresh_series()

    def _series_cover(self, media_id: int):
        """The flyout cover for media_id from the series palette (R3, L11:
        the session poster, else the poster saved for a picked series, else
        the AniList color, else the neutral tile); None without a palette."""
        palette = self._series_palette
        return palette.cover_for(media_id) if palette is not None else None

    def _refresh_series(self) -> None:
        """Show _series in the flyout's series row (a no-op without the row)
        and in the tooltip."""
        media_id, label, native = self._series
        if media_id is None:
            self._flyout.set_series(None)
        else:
            self._flyout.set_series(label, native, self._series_cover(media_id))
        self._refresh_tooltip()

    def _on_cover_changed(self, media_id: int) -> None:
        """The palette learned a better cover; only the shown series' matters."""
        if media_id == self._series[0]:
            self._flyout.set_cover(self._series_cover(media_id))

    def _on_series_clear(self) -> None:
        """Clear from the flyout or the series bar's chip: no series
        (Decision 11, L8). The server deletes the pointer and broadcasts a
        state frame without a series. With the socket down the pointer is
        deleted here and the clear waits for the reconnect flush, like an
        offline pick (Decision 66)."""
        self._set_series(None)
        # Newer than any pick still in flight: drop the queued one and void
        # the running build's activation (or failure toast) ...
        self._queued_pick = None
        self._build_superseded = self._gaz_thread is not None
        if self._series_palette is not None:
            # ... and the bar's loading status for it (L9). A no-op while the
            # bar shows its search, so a Clear from the chip keeps it open.
            self._series_palette.cancel_status()
        if self._client.send_command(
            {"type": "command", "action": "set_series", "media_id": None}
        ):
            self._pending_series = None
            return
        try:
            clear_active_series()
        except OSError:
            _log.warning("offline active-series clear failed", exc_info=True)
        self._pending_series = _NO_SERIES

    def _on_change_series_clicked(self) -> None:
        if self._on_change_series is not None:
            self._on_change_series()

    def on_series_chosen(self, media_id: int, label: str, native: str = "") -> None:
        # Palette selection -> build+cache the gazetteer off-thread, then tell
        # the pipeline (in _on_gazetteer_built).
        # Snapshot the manual-genre recency counter AT PICK TIME -- queued or
        # started alike: the pick is what carries the "watching anime" intent,
        # and the built signal may land seconds later (network fetch; for a
        # queued pick, a whole earlier build later). _on_gazetteer_built
        # compares against this snapshot to skip the auto-genre ride-along
        # when a manual genre pick happened in between.
        self._series_pick_genre_seq = self._manual_genre_seq
        if self._gaz_thread is not None and self._gaz_thread.isRunning():
            # A DIFFERENT request while a build is in flight (unlike the
            # diagnostics re-click, where a repeat means "same request"):
            # queue the LATEST pick -- overwriting any earlier queued one --
            # instead of silently dropping it. The running build's
            # completion slot starts it and discards its own stale
            # activation (_consume_queued_pick).
            self._queued_pick = (media_id, label, native)
            return
        self._start_gazetteer_build(media_id, label, native)

    def _start_gazetteer_build(self, media_id: int, label: str, native: str = "") -> None:
        thread = _GazetteerBuildThread(media_id, label, parent=self)
        self._build_superseded = False
        # The worker only builds the cache; the pick's Japanese title rides
        # the slot binding (a queued delivery to the GUI thread, like the
        # bound method it replaces) into set_series.
        thread.built.connect(lambda mid, lbl, n=native: self._on_gazetteer_built(mid, lbl, n))
        thread.buildFailed.connect(self._on_gazetteer_build_failed)
        # The build's staged progress: the series bar's loading line (L10).
        thread.progress.connect(self._on_gazetteer_progress)
        # Bind THIS thread into the slot: finished is a QUEUED delivery to the
        # GUI thread, so by the time the slot runs self._gaz_thread may already
        # name a newer, running thread (see _on_gaz_thread_finished).
        thread.finished.connect(lambda t=thread: self._on_gaz_thread_finished(t))
        self._gaz_thread = thread
        thread.start()

    def _on_gazetteer_progress(self, media_id: int, fraction: float) -> None:
        """The running build's staged progress (0..1) for the series bar's
        loading line (L10). A build that a queued re-pick or a Clear already
        superseded reports nothing: the bar follows the newest pick, which
        may be the same series again (other media ids the bar ignores
        itself). Queued delivery keeps a finished build's reports ahead of
        its built/failed, so the gate is never stale for the next build."""
        if self._queued_pick is not None or self._build_superseded:
            return
        if self._series_palette is not None:
            self._series_palette.show_build_progress(media_id, fraction)

    def _consume_queued_pick(self) -> bool:
        """A newer palette pick landed while the just-finished build ran:
        start the queued (latest) pick's build now and tell the caller to
        DISCARD the finished build's outcome -- a stale build must not
        flip series/genre over the user's newer intent, and a stale
        failure toast would blame the pick the user already abandoned.
        Returns True when a queued pick superseded the finished build.
        The genre-seq snapshot is NOT retaken here: it was taken when the
        queued pick was made (on_series_chosen), so a manual genre change
        during the wait still wins over the ride-along."""
        queued = self._queued_pick
        if queued is None:
            return False
        self._queued_pick = None
        self._start_gazetteer_build(*queued)
        return True

    def _on_gaz_thread_finished(self, thread: _GazetteerBuildThread) -> None:
        # Delete the thread that ACTUALLY emitted finished, not whatever
        # self._gaz_thread points at now: finished is a queued delivery, so a
        # re-pick can pass on_series_chosen's isRunning() guard (the worker has
        # already flipped isRunning() false before finished is delivered) and
        # install a new RUNNING thread in self._gaz_thread before this slot
        # runs for the old one. Deleting via the attribute would destroy the
        # live replacement (Qt6 qFatal "Destroyed while thread is still
        # running") and orphan its cleanup. Null the ref only if it still names
        # this thread, so the re-entrancy guard on the next pick never calls
        # isRunning() on a deleted C++ object. deleteLater defers destruction
        # until after this slot returns.
        thread.deleteLater()
        if self._gaz_thread is thread:
            self._gaz_thread = None

    def _discard_finished_build(self) -> bool:
        """True when the finished build's outcome must be dropped: a newer
        queued pick (started now, see _consume_queued_pick) or a Clear made
        while it ran supersedes it."""
        if self._consume_queued_pick():
            return True
        superseded, self._build_superseded = self._build_superseded, False
        return superseded

    def _on_gazetteer_built(self, media_id: int, label: str, native: str = "") -> None:
        # A newer pick queued during this build, or a Clear, supersedes it:
        # discard this activation entirely (no set_series, no label flip, no
        # auto-genre ride-along); a queued pick's build starts instead.
        if self._discard_finished_build():
            return
        # Cache is ready on disk (shared app_data). Tell the pipeline; it loads
        # the cache + hot-swaps + persists the pointer. Show it in the flyout
        # and the bar's chip now (_set_series also saves its poster, L11).
        self._set_series(media_id, label, native)
        if self._series_palette is not None:
            # The bar's status for this pick: "Now watching ...", then it slides
            # away (L9); ignored when the user moved on to another pick.
            self._series_palette.show_built(media_id)
        sent = self._client.send_command({
            "type": "command", "action": "set_series",
            "media_id": media_id, "label": label, "native": native or None,
        })
        if sent:
            self._pending_series = None
        else:
            # WS down (pipeline starting / in reconnect backoff). Persist the
            # pointer locally so a starting pipeline restores it, and queue a
            # reconnect flush (mirror _pending_genre / offline
            # _send_style_change).
            try:
                save_active_series(media_id, label, native or None)
            except Exception:
                _log.exception("offline active-series persist failed")
            self._pending_series = (media_id, label, native)
        if self._auto_genre and self._manual_genre_seq == self._series_pick_genre_seq:
            # A series pick is durable "I am watching anime" intent (the
            # palette is AniList, anime-only): ride a set_genre along the SAME
            # persisted path as a manual tray pick. Boot restore and series
            # clear never come through here, so a manual genre change after a
            # pick sticks until the next explicit pick -- INCLUDING one made
            # during the build window: this slot fires when the worker's
            # AniList fetch finishes, seconds after the pick, so the seq
            # comparison skips the ride-along when _on_genre_chosen ran after
            # on_series_chosen snapshotted the counter (that manual pick is
            # chronologically newer than the pick and must stick). Same
            # offline queue + supersede invariants as _on_genre_chosen: a
            # successful send clears any stale queued genre; a failed one
            # queues "anime" for the reconnect flush.
            if self._client.send_command(
                {"type": "command", "action": "set_genre", "genre": "anime"}
            ):
                self._pending_genre = None
            else:
                self._pending_genre = "anime"

    def _on_gazetteer_build_failed(
        self, media_id: int, label: str, error: object = None,
    ) -> None:
        # A newer pick queued during this build, or a Clear, supersedes the
        # failure too: a "Series not changed" toast for the OLD pick would
        # only mislead. A queued pick's build starts instead.
        if self._discard_finished_build():
            return
        if self._series_palette is not None:
            # The bar's loading status goes at once; the toast below is the
            # notice (L9).
            self._series_palette.show_build_failed(media_id)
        if error is None or isinstance(
            error, (AniListUnavailable, httpx.TransportError)
        ):
            # Network-class failure (offline, AniList outage 403, repeated
            # 429, connect/timeout): the connection copy is accurate.
            body = (
                f"Couldn't load character names for {label}. "
                "Check your connection and try again."
            )
        else:
            # Anything else (disk full, parse bug) must not masquerade as
            # a connectivity problem: name the class so a bug report's
            # triage isn't misdirected at the user's router.
            body = f"Couldn't build the series data: {type(error).__name__}"
        self.show_toast("Series not changed", body, 8000)

    def release_threads(self) -> None:
        """Detach still-running gazetteer-build, diagnostics, and quant-download
        workers at quit (same Qt6 destroyed-while-running contract as
        SeriesPalette.release_threads).

        A QThread destroyed by widget teardown / interpreter finalization
        while its OS thread runs is a Qt6 qFatal abort. The gazetteer build
        blocks in httpx (AniList); the diagnostics collection blocks for
        seconds (the snapshot's nvidia-smi probe has a 5 s subprocess
        timeout, plus the zip deflate); an open settings window may be running
        a quant-tier model download blocked inside hf_hub_download (delegated
        to SettingsWindow.release_threads -> the active download dialog).
        Disconnecting a worker's ``finished``
        entirely also drops its queued cleanup slot (_on_gaz_thread_finished
        / _on_diag_thread_finished), which would otherwise deleteLater the
        now-detached thread and leave thread_keeper holding a dead wrapper
        (any_running() would then RuntimeError on it -- the no-prune
        contract). The diagnostics worker additionally carries
        finishedWithPath -> _on_diagnostics_done; dropping it here stops a
        stale UI notification firing mid-teardown. Each block no-ops cleanly
        when its thread is absent or already finished.
        """
        from ..ui import thread_keeper
        # An open settings window may be running a quant-tier model download;
        # detach its worker too (delegates to the active download dialog).
        win = self._settings_window
        if win is not None:
            try:
                win.release_threads()
            except Exception:
                _log.exception("settings-window thread release on quit failed")
        gaz = self._gaz_thread
        if gaz is not None and gaz.isRunning():
            try:
                gaz.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            thread_keeper.release(gaz)
            self._gaz_thread = None
        diag = self._diag_thread
        if diag is not None and diag.isRunning():
            for signal in (diag.finished, diag.finishedWithPath):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
            thread_keeper.release(diag)
            self._diag_thread = None

    def _on_quit_requested(self) -> None:
        """User clicked Tray -> Quit. Send a clean-shutdown command to
        the pipeline over the WS, then quit our own QApplication.

        Without this wiring, Tray->Quit only stopped
        the overlay process. Pipeline + llama-server kept running; user
        had to Task-Manager them. Now we tell the pipeline to exit
        gracefully (its main.py finally block handles glossary save,
        translator aclose, cuda_pool shutdown, latency_log flush, etc.),
        THEN drop our own QApplication. The top-level watchdog sees the
        pipeline's exit 0 -> stops its pipeline supervisor; the shared
        stop_event-based propagation isn't needed because both watchdog
        supervisor threads will see their children exit cleanly.
        """
        from PySide6.QtCore import QTimer
        # Best-effort: send the quit command. If the WS is disconnected
        # the send is a no-op (PipelineClient handles that silently).
        self._client.send_command({"type": "command", "action": "quit"})
        # Give the WS send + pipeline's graceful-exit cleanup a beat
        # before we kill ourselves. 500 ms is enough for uvicorn to
        # flush and for main.py's finally block to start running;
        # waiting longer blocks the UI unnecessarily.
        #
        # Deliberately a free-standing singleShot, NOT the
        # parented-timer pattern: the callback is the injected
        # App-level on_quit callable (typically QApplication.quit),
        # so firing it never dereferences this Tray's C++ object --
        # and this timer IS the quit path, so nothing destroys the
        # Tray inside the 500 ms except the quit it triggers.
        QTimer.singleShot(500, self._on_quit_cb)

    def _on_style_received(self, msg: dict) -> None:
        raw = {k: v for k, v in msg.items() if k != "type"}
        if not raw:
            return
        # The WS style frame is untrusted: a loopback set_style echo can carry
        # padding:None, a non-numeric size, or NaN/Infinity (json.loads accepts
        # the literals). _clamp_style is the SINGLE ingestion clamp for the
        # overlay process -- it seeds every key from STYLE_DEFAULTS and coerces
        # each garbage value to its default, so the render path
        # (setContentsMargins(None) -> TypeError) and the settings-window seed
        # (int(None) -> TypeError) can't detonate downstream. Disk loads already
        # run this in config.load_overlay_config; WS frames did not until now.
        from ..config import _clamp_style
        style = _clamp_style(raw)
        self._current_style = dict(style)
        self._sync_style_controls()
        if self._overlay_window is not None:
            self._overlay_window.apply_style(style)

    def _open_settings(self, page: str | None = None) -> None:
        # Single instance: a second click on an already-open window raises
        # and focuses it rather than stacking a duplicate.
        existing = self._settings_window
        if existing is not None and existing.isVisible():
            if page is not None:
                existing.show_page(page)
            # A minimized top-level still reports isVisible() True, and
            # raise_/activateWindow do NOT clear WindowMinimized on Windows:
            # without showNormal() the "raise" leaves the window in the
            # taskbar and the Settings click looks dead.
            if existing.isMinimized():
                existing.showNormal()
            existing.raise_()
            # A notice's click reaches the shell, not this process: without
            # the foreground Settings stays behind the app in front. The
            # tray's own routes already hold it, so this changes nothing there.
            take_foreground(existing)
            existing.activateWindow()
            return
        win = SettingsWindow(
            dict(self._current_style),
            config_path=self._config_path,
            initial_config=self._settings_config,
            read_only=self._settings_read_only,
            on_move_subtitles=self._on_move_subtitles,
            banner=(self._overlay_window.banner_text()
                    if self._overlay_window is not None else None),
            page=page,
        )
        # Apply/OK: Settings already persisted; this round-trip only applies
        # and broadcasts the committed style.
        win.styleChanged.connect(self._send_style_change)
        # Any control change: debounced live preview on the real overlay.
        win.stylePreview.connect(self._on_style_preview)
        # Close: drop the sample and restore the committed style.
        win.previewEnded.connect(self._on_preview_ended)
        # Chrome theme pick: the window re-skins itself, then we re-skin the
        # rest of the runtime chrome (menu + active placement hint) to match.
        win.themeChanged.connect(self._on_settings_theme_changed)
        # Auto-genre checkbox: flips the series-pick genre ride-along live
        # (the window owns persistence, same split as themeChanged).
        win.autoGenreChanged.connect(self.set_auto_genre)
        # Reduce motion: the live overlay stops (or resumes) motion now, and
        # the apply_style that follows carries it to the browser page.
        win.reduceMotionChanged.connect(self._on_reduce_motion_changed)
        # Notices that must outlive the window (the OK flow closes it
        # right away, taking the inline status line with it) surface as
        # tray toasts, the same channel as the gazetteer notifications.
        win.noticeRequested.connect(self._on_settings_notice)
        # Shortcuts: Settings persisted them; the menu and the flyout show
        # the new combos and the pipeline re-binds live.
        win.shortcutsChanged.connect(self._on_shortcuts_changed)
        # Faster subtitles: Settings persisted it; the pipeline uses it from
        # its next speech window.
        win.incrementalChanged.connect(self._on_incremental_changed)
        # Position is already in the Settings transaction. Re-pin the live
        # overlay without firing its ordinary persistence callback a second
        # time. Guarded because tests may construct a tray without a window.
        overlay_window = self._overlay_window
        if overlay_window is not None:
            win.positionApplied.connect(
                lambda pos: overlay_window.set_position(
                    pos, persist=False,
                )
            )
        self._settings_window = win
        win.show()
        win.raise_()
        take_foreground(win)  # as above
        win.activateWindow()

    def _on_reduce_motion_changed(self, on: bool) -> None:
        """Settings committed overlay.reduce_motion: the overlay and the
        series bar stop (or resume) motion now; the next apply_style carries
        it to the page."""
        self._reduce_motion = bool(on)
        if self._overlay_window is not None:
            self._overlay_window.set_reduce_motion(self._reduce_motion)
        if self._series_palette is not None:
            self._series_palette.set_reduce_motion(self._reduce_motion)

    def _on_settings_notice(self, title: str, body: str) -> None:
        """Toast a notice the settings window could not keep on screen
        (its OK flow closes the window before the inline status line can
        be read). Fired via noticeRequested, so it runs on the GUI thread."""
        self.show_toast(title, body, 8000)

    def _placement_active(self) -> bool:
        """True while an on-screen placement session is live -- it then OWNS
        the single sample slot (the P1-14 arbitration at the call sites).
        getattr keeps the probe safe on doubles that carry no ``active``."""
        return self._placement is not None and getattr(self._placement, "active", False)

    def _on_settings_theme_changed(self, value: str) -> None:
        """The settings window persisted a new chrome theme and ALREADY
        re-skinned itself (its _on_theme_changed ran apply_palette before
        emitting). Re-skin only the REST of the runtime chrome -- repainting
        the originating window here again would repolish every widget in it
        twice per pick."""
        self._apply_chrome_palette(palette_for(value), skip_settings_window=True)

    def apply_palette(self, palette: dict) -> None:
        """Re-skin the runtime chrome for a new theme palette.

        Repaints the short menu and the flyout, and -- when they are live --
        an open settings window (through its public apply_palette, skinned
        from THIS palette) and the active placement hint. This is the OS
        colorScheme-follow entry point (build_overlay routes
        colorSchemeChanged here while in system mode), so unlike the settings
        window's own theme pick it MUST reach the open window: nothing else
        re-skins it on an OS flip.
        """
        self._apply_chrome_palette(palette, skip_settings_window=False)

    def _apply_chrome_palette(self, palette: dict, skip_settings_window: bool) -> None:
        """Shared re-skin body for apply_palette / _on_settings_theme_changed:
        one palette, one repaint per surface. skip_settings_window is True
        only for the window's own theme pick (it has already skinned itself)."""
        self._palette = palette
        self._menu.setStyleSheet(menu_qss(palette))
        self._flyout.apply_palette(palette)
        win = self._settings_window
        if win is not None and not skip_settings_window:
            # Re-skin the open window from the SAME palette so a system-mode OS
            # flip repaints it too. A theme flip is cosmetic: guard it so a
            # window mid-teardown (deleted C++ object) can't take the overlay
            # down -- match the never-crash discipline of the sibling handlers.
            try:
                win.apply_palette(palette)
            except Exception:
                _log.exception("settings-window re-skin on theme flip failed")
        if self._placement_active():
            self._placement.apply_palette(palette)
        sp = self._series_palette
        if sp is not None:
            # Re-skin the series palette (gazetteer feature only). It is a
            # boot-lived singleton, not the ephemeral settings window, so no
            # deleted-C++ mid-teardown guard is needed: the None check
            # ("when present") is the whole guard, mirroring the
            # active-placement forward above.
            sp.apply_palette(palette)

    def _on_style_preview(self, style: dict) -> None:
        """Live-preview a style dict on the real overlay while Settings is open.

        Shows a persistent sample subtitle the first time (so the user sees
        the change even with no live audio), then restyles it. This is the
        PREVIEW path -- it never persists; commit happens via styleChanged.
        """
        if self._overlay_window is None:
            return
        # Belt against a debounce frame that arrives AFTER the settings window
        # closed (P2): _on_preview_ended nulls _settings_window on close, so a
        # late stylePreview here must not resurrect an orphaned sample on the
        # live overlay. In production a preview only ever fires while a window
        # is open (the signal is connected in _open_settings), so this can only
        # trip on the post-close race the settings-side _preview_timer.stop()
        # already guards.
        if self._settings_window is None:
            return
        # Remember the latest previewed style even if this frame is dropped
        # below, so _on_placement_exit can restore the preview after a
        # placement session that borrowed the sample slot.
        self._preview_style = dict(style)
        # Sample-slot arbitration (P1-14): placement OWNS the single sample
        # slot while active (it shows the drag anchor there). Drop this preview
        # frame rather than replace the drag anchor with the preview sample;
        # _on_placement_exit re-shows the preview when placement tears down.
        if self._placement_active():
            return
        if not self._preview_active:
            self._overlay_window.show_sample(_PREVIEW_JP, _PREVIEW_EN, _PREVIEW_RUBY)
            self._preview_active = True
        self._overlay_window.apply_style(style)

    def _on_preview_ended(self) -> None:
        """Settings window closed: drop the sample and restore the committed
        style so an un-applied preview doesn't stick on the overlay.

        The restore only runs when a preview actually ran this session
        (_preview_active). A plain open+close with no control change shows no
        sample, so there is nothing to hide and no reason to reskin the live
        overlay to _current_style -- doing so unconditionally would push the
        seed/committed style onto the overlay on every close for no user action.
        The style it restores is _current_style, which the connected Apply path
        (_send_style_change) updates optimistically, so an OK'd change survives
        the close instead of reverting for a server round-trip.
        """
        # previewEnded is emitted from INSIDE SettingsWindow.closeEvent, before
        # super().closeEvent(event) runs -- and the tray holds the ONLY Python
        # ref to the parentless (Python-owned) window. Nulling the ref here
        # would destroy the C++ widget while its own close dispatch is still
        # unwinding, the same sporadic-crash class placement._teardown defers
        # with singleShot(0). Park the last ref in a one-shot timer instead:
        # the deferred deleteLater keeps the wrapper alive past the dispatch
        # AND tears the closed window down cleanly on the next loop turn.
        from PySide6.QtCore import QTimer
        win = self._settings_window
        self._settings_window = None
        if win is not None:
            QTimer.singleShot(0, win.deleteLater)
        # Sample-slot arbitration (P1-14): while placement is active it owns the
        # single sample slot (the drag anchor). Hiding here would clear the
        # anchor mid-drag; skip the hide/restore and let placement teardown own
        # the sample. Still clear the preview flag so a later stylePreview
        # re-shows the sample -- but the restore this close owed does not
        # vanish: it is left OWED for _on_placement_exit to settle, or the
        # never-committed preview style would stick on the overlay for every
        # subsequent subtitle.
        if self._placement_active():
            if self._preview_active:
                self._style_restore_owed = True
            self._preview_active = False
            return
        if self._preview_active and self._overlay_window is not None:
            self._overlay_window.hide_sample()
            self._overlay_window.apply_style(self._current_style)
        self._preview_active = False

    def _on_placement_exit(self) -> None:
        """Placement torn down -- it borrowed the single sample slot (P1-14).

        Supplied to PlacementController as its ``on_exit`` callback. If a
        settings live-preview was active BEFORE placement started, placement's
        hide_sample cleared it, so re-show the preview sample and re-apply the
        last previewed style (_preview_style, falling back to the committed
        _current_style) so the live preview resumes. Otherwise reset the
        preview flag so the NEXT stylePreview re-shows the sample rather than
        assuming one is still up -- and, when the settings window CLOSED
        during this placement session, settle the restore its close owed
        (_on_preview_ended couldn't touch the borrowed sample slot then):
        re-apply the committed style exactly once, or the never-committed
        preview style would stick on the overlay.
        """
        if (self._settings_window is not None and self._preview_active
                and self._overlay_window is not None):
            self._overlay_window.show_sample(_PREVIEW_JP, _PREVIEW_EN, _PREVIEW_RUBY)
            style = self._preview_style if self._preview_style is not None else self._current_style
            self._overlay_window.apply_style(style)
        else:
            self._preview_active = False
            if self._style_restore_owed and self._overlay_window is not None:
                self._overlay_window.apply_style(self._current_style)
        self._style_restore_owed = False

    def _send_style_change(self, style: dict) -> None:
        # Optimistic local intent: the just-committed style IS the user's
        # latest choice, so adopt it as the known-applied style right away --
        # before the send, so BOTH branches share it. Without this, the
        # previewEnded that fires when the settings window closes on OK
        # (_on_preview_ended) would restore the PRE-OK style for one server
        # round-trip, and permanently if the echo is lost. The server stays
        # authoritative: its broadcast echo (_on_style_received) overwrites
        # _current_style, confirming or correcting this. Mirrors the offline
        # branch below, which already merged into _current_style eagerly.
        merged = {**self._current_style, **style}
        self._current_style = merged
        self._sync_style_controls()
        sent = self._client.send_command({
            "type": "command",
            "action": "apply_style",
            "style": style,
            "reduce_motion": self._reduce_motion,
        })
        if sent:
            # Server round-trip owns apply + broadcast: Settings already
            # wrote user_overrides.yaml atomically. A successful
            # send supersedes any style still queued from an earlier
            # offline apply — without this, a flush that raced the
            # socket closing could replay the stale style over this
            # newer one on the next reconnect.
            self._pending_style = None
            return
        # WS down (pipeline still starting, or inside the
        # up-to-30s reconnect backoff — exactly when a user pokes at
        # Settings). The committed click must still apply locally, then queue
        # a flush so the pipeline's in-memory style converges on reconnect.
        if self._overlay_window is not None:
            self._overlay_window.apply_style(merged)
        self._pending_style = merged

    def _send_hotkeys(self, hotkeys: dict) -> None:
        """Ask the pipeline to re-bind the shortcuts Settings just saved.
        Offline (pipeline starting or reconnecting), keep the latest changes
        and send them on reconnect, like the style."""
        pending = {**(self._pending_hotkeys or {}), **hotkeys}
        if self._client.send_command({"type": "command", "action": "set_hotkeys",
                                      "hotkeys": pending}):
            self._pending_hotkeys = None
        else:
            self._pending_hotkeys = pending

    def _on_incremental_changed(self, enabled: bool) -> None:
        """Send the committed Faster subtitles switch to the pipeline.
        Offline (pipeline starting or reconnecting), keep it and send it on
        reconnect, like the style."""
        if self._client.send_command({"type": "command", "action": "set_incremental",
                                      "enabled": bool(enabled)}):
            self._pending_incremental = None
        else:
            self._pending_incremental = bool(enabled)

    def _on_connected_changed(self, connected: bool) -> None:
        if not connected:
            return
        # Ordering contract: connectedChanged
        # fires on handshake completion, BEFORE the server's on-connect
        # snapshots (state + style) can arrive -- so the flush below is
        # already in flight when the stale snapshot lands. The snapshot
        # may flash the stale style/genre for ~one loopback RTT (cosmetic,
        # accepted), but the server then merges the flush and broadcasts
        # it back: the flush must win. _on_style_received/_on_state never
        # clear the pending values, so a flush that raced the socket
        # closing survives for the next reconnect.
        if self._pending_style is not None:
            # One-shot: cleared on successful send; kept for the next
            # reconnect if this send raced the socket closing again.
            if self._client.send_command({
                "type": "command",
                "action": "apply_style",
                "style": self._pending_style,
                "reduce_motion": self._reduce_motion,
            }):
                self._pending_style = None
        if self._pending_genre is not None:
            # Same one-shot replay for the offline genre pick.
            if self._client.send_command({
                "type": "command",
                "action": "set_genre",
                "genre": self._pending_genre,
            }):
                self._pending_genre = None
        if self._pending_series is not None:
            # Same one-shot replay for the offline series pick or Clear (a
            # pick's gazetteer cache is already on disk from
            # _on_gazetteer_built; _NO_SERIES is a Clear).
            mid, lbl, native = self._pending_series
            command = {"type": "command", "action": "set_series", "media_id": mid}
            if mid is not None:
                command.update(label=lbl, native=native or None)
            if self._client.send_command(command):
                self._pending_series = None
        if self._pending_hotkeys is not None:
            # Same one-shot replay for shortcuts saved while offline. The
            # replay flag keeps unchanged shortcuts from announcing their
            # conflict a second time.
            if self._client.send_command({"type": "command", "action": "set_hotkeys",
                                          "hotkeys": self._pending_hotkeys,
                                          "replay": True}):
                self._pending_hotkeys = None
        if self._pending_incremental is not None:
            # Same one-shot replay for Faster subtitles saved while offline.
            if self._client.send_command({"type": "command", "action": "set_incremental",
                                          "enabled": self._pending_incremental}):
                self._pending_incremental = None

    def _on_open_log_folder(self) -> None:
        """Flyout > Troubleshooting > Open log folder: explorer on paths.logs_dir().

        Best-effort: ensure_dirs() first so a fresh install (nothing
        logged yet) opens an empty folder instead of raising
        FileNotFoundError; any OS failure logs and returns -- a helper
        menu item must never take the overlay down.
        """
        try:
            paths.ensure_dirs()
            os.startfile(paths.logs_dir())
        except OSError:
            _log.warning("could not open the log folder", exc_info=True)

    def _on_collect_diagnostics(self) -> None:
        """Flyout > Troubleshooting > Save diagnostics: build the zip on a worker
        thread (nvidia-smi probe + zip deflate can take ~5-7 s;
        blocking here froze the GUI thread for the duration), then
        announce the result from _on_diagnostics_done, which the
        worker's finishedWithPath signal reaches as a queued call on
        the GUI thread.
        """
        if self._diag_thread is not None and self._diag_thread.isRunning():
            # A collection is already in flight; a second parallel
            # worker would zip the same logs twice and double-notify.
            return
        thread = _DiagnosticsThread(parent=self)
        thread.finishedWithPath.connect(self._on_diagnostics_done)
        # Delete the finished wrapper and clear the ref (mirrors
        # _on_gaz_thread_finished): each collection builds a new thread, so
        # finished wrappers would otherwise pile up as Tray children until
        # exit, and a re-click after a queued finish could read a deleted
        # C++ object. Bind THIS thread into the slot -- finished is a queued
        # delivery, so self._diag_thread may name a newer thread by then.
        thread.finished.connect(lambda t=thread: self._on_diag_thread_finished(t))
        self._diag_thread = thread
        thread.start()

    def _on_diag_thread_finished(self, thread: _DiagnosticsThread) -> None:
        # Delete the thread that ACTUALLY emitted finished, not whatever
        # self._diag_thread points at now: finished is a queued delivery, so a
        # re-click can pass _on_collect_diagnostics' isRunning() guard (the
        # worker has already flipped isRunning() false before finished is
        # delivered) and install a new RUNNING thread in self._diag_thread
        # before this slot runs for the old one. Deleting via the attribute
        # would risk destroying the live replacement (Qt6 qFatal "Destroyed
        # while thread is still running"). Null the ref only if it still names
        # this thread, so the re-entrancy guard on the next click never calls
        # isRunning() on a deleted C++ object. Mirrors _on_gaz_thread_finished.
        thread.deleteLater()
        if self._diag_thread is thread:
            self._diag_thread = None

    def _on_diagnostics_done(self, zip_path: str) -> None:
        """GUI-thread landing slot for the diagnostics worker.

        collect_diagnostics never raises AND removes any partial zip on
        failure, so the exists() check is the success/failure branch.
        """
        if zip_path and Path(zip_path).exists():
            path = Path(zip_path)
            # Decisions 16 and 35: the file name (the zip is on the Desktop),
            # and a click shows it selected in Explorer, which runs outside
            # the Job Objects so a restart or Quit can't close it.
            self.show_toast(
                "Diagnostics saved",
                f"{path.name}, on your desktop.",
                10_000,
                on_click=lambda: show_in_folder_outside_job(path),
            )
        else:
            self.show_toast(
                "Diagnostics collection failed",
                "No zip could be written. Logs are in "
                "%LOCALAPPDATA%\\Yamabiko\\logs.",
                10_000,
            )

    def _on_position_preset(self, screen_name: str, v_pct: float) -> None:
        """A Position preset from the flyout's picker: pin the stack at v_pct
        of the named screen, at the default horizontal anchor of both
        alignments. set_position persists through on_position_changed, so
        the preset is durable; an open Settings window adopts it (Decision
        68)."""
        if self._overlay_window is None:
            return
        self._overlay_window.set_position({
            "screen": screen_name,
            "v_offset_pct": v_pct,
            # Presets write full default anchors for both alignments (the
            # Left edge and the Center), from DEFAULT_POSITION.
            "h_offset_pct": DEFAULT_POSITION["h_offset_pct"],
            "left_offset_pct": DEFAULT_POSITION["left_offset_pct"],
        })
        self._adopt_position_in_settings()

    def _on_move_subtitles(self) -> None:
        """Enter on-screen placement mode (drag a sample subtitle). Lazily
        construct the PlacementController, guarded against re-entry while a
        session is already active."""
        if self._overlay_window is None:
            return
        from .placement import PlacementController
        if not self._placement_active():
            # Save through the window's persist path, like the presets: it is
            # off under an explicit --config and records the saved position
            # so the quit-time save does not write it again.
            self._placement = PlacementController(
                self._overlay_window, on_save=self._on_placement_save,
                palette=self._palette,
                # on_exit fires on teardown so the tray can restore the settings
                # live-preview sample placement borrowed (P1-14 arbitration).
                on_exit=self._on_placement_exit,
            )
            self._placement.enter()

    def _on_placement_save(self, position: dict) -> None:
        """Placement saved (Enter or a click away): persist through the
        window like a preset, and let an open Settings window adopt it
        (Decision 68)."""
        self._overlay_window.set_position(position)
        self._adopt_position_in_settings()

    def _adopt_position_in_settings(self) -> None:
        win = self._settings_window
        if win is not None and self._overlay_window is not None:
            win.adopt_position(self._overlay_window.current_position())

    def _on_state(self, genre: str, paused: bool) -> None:
        self._paused = bool(paused)
        self._genre = genre if genre in _GENRES else "variety"
        self._flyout.set_genre(self._genre)
        self._refresh_status()

    def set_overlay_visible(self, visible: bool) -> None:
        """OverlayWindow.visibleChanged (wired by build_overlay): the
        subtitles were shown or hidden (Ctrl+Alt+O, the Subtitles tile)."""
        self._overlay_visible = bool(visible)
        self._refresh_status()

    def _refresh_status(self) -> None:
        """Show pause, the subtitles' visibility and the problems in the short
        menu, the flyout (it ranks problem > paused > hidden > listening,
        Decision 26), the icon and the tooltip."""
        self._refresh_menu_text()
        self._flyout.set_paused(self._paused)
        self._flyout.set_subtitles_shown(self._overlay_visible)
        self._flyout.set_problem(self._problems[0]["title"] if self._problems else None)
        self._refresh_icon()
        self._refresh_tooltip()

    def _refresh_menu_text(self) -> None:
        combos = self._shortcuts or {}
        self._pause_action.setText(_with_combo(
            "Resume" if self._paused else "Pause", combos.get("pause_resume", ""),
        ))
        if self._pick_series_action is not None:
            self._pick_series_action.setText(_with_combo(
                "Pick a series\u2026", combos.get("show_series", ""),
            ))

    def _refresh_shortcut_labels(self) -> None:
        """The combos in the short menu and the flyout tooltips (Decision 13)."""
        self._refresh_menu_text()
        self._flyout.set_shortcut_labels(self._shortcuts or {})

    def _on_shortcuts_changed(self, hotkeys: dict) -> None:
        """Settings saved changed shortcuts: show the new combos (Decision 33),
        then have the pipeline re-bind them (_send_hotkeys). With shortcuts
        off (None) the pipeline binds nothing, so no combo appears."""
        if self._shortcuts is not None:
            series_on = self._pick_series_action is not None
            for key, combo in hotkeys.items():
                label = combo_label(combo)
                if label and key in HOTKEY_KEYS and (key != "show_series" or series_on):
                    self._shortcuts[key] = label
                else:
                    self._shortcuts.pop(key, None)
            self._refresh_shortcut_labels()
        self._send_hotkeys(hotkeys)

    def _sync_style_controls(self) -> None:
        """Show the current Mode and Alignment in the flyout."""
        self._flyout.set_mode(_style_choice(self._current_style, "mode"))
        self._flyout.set_alignment(_style_choice(self._current_style, "alignment"))

    def _on_mode_chosen(self, mode: str) -> None:
        self._set_overlay_choice("mode", mode)

    def _on_alignment_chosen(self, alignment: str) -> None:
        self._set_overlay_choice("alignment", alignment)

    def _set_overlay_choice(self, key: str, value: str) -> None:
        """The flyout's Mode tile / Alignment segments (Decision 11): persist
        overlay.style.<key> as a Settings Apply would (never under an explicit
        --config, which leaves the user's file alone), then apply it through
        the Apply path. A failed write logs and still applies for this
        session (Decision 27)."""
        if value not in STYLE_CHOICES[key] or self._current_style.get(key) == value:
            return
        if not self._settings_read_only:
            try:
                write_overlay_style({key: value})
            except Exception:
                _log.warning(
                    "could not save overlay.style.%s; applied for this session only",
                    key, exc_info=True,
                )
        self._send_style_change({key: value})
        if self._settings_window is not None:
            # Otherwise the open window writes the value it opened with on
            # its next Apply / OK (Decision 68).
            self._settings_window.adopt_overlay_choice(key, value)

    def _on_toggle_subtitles(self) -> None:
        """The flyout's Subtitles tile does what Ctrl+Alt+O does (Decision
        11): the pipeline broadcasts toggle_overlay to every client, this
        overlay and the browser/OBS page alike. With the socket down only
        this window can follow, so it toggles directly. The tile shows the
        window's real state either way (visibleChanged)."""
        if self._client.send_command({"type": "command", "action": "toggle_overlay"}):
            return
        if self._overlay_window is not None:
            self._overlay_window.toggle_visible()

    def _on_settings_requested(self) -> None:
        """The short menu's Settings... and the flyout's Settings."""
        self._open_settings()

    def _on_activated(self, reason) -> None:
        """Icon clicks (Decision 9). Left click opens the flyout, or closes it
        when it is open; the Trigger of a click whose mouse-down already hid
        the flyout is swallowed. A double click always ends with the flyout
        open. A right click clears the swallow (Decision 63) and pops the
        short menu up on the taskbar's line: Qt's own context menu anchors at
        the cursor, over the taskbar."""
        kinds = QSystemTrayIcon.ActivationReason
        if reason == kinds.Trigger:
            if not self._debounce.accept_trigger():
                return
            if self._flyout.isVisible():
                self._flyout.hide()
                # Closed by this Trigger, not by a mouse-down: nothing to swallow.
                self._debounce.flyout_hidden(False)
            else:
                self._show_flyout()
        elif reason == kinds.DoubleClick:
            self._debounce.double_click()
            self._show_flyout()
        elif reason == kinds.Context:
            # TriggerDebounce has no Context entry: flyout_hidden(False)
            # disarms the swallow the right click's mouse-down armed.
            self._debounce.flyout_hidden(False)
            # The size it shows at: at least its 224 px minimum width.
            size = self._menu.sizeHint().expandedTo(self._menu.minimumSize())
            self._menu.popup(tray_placement.place_menu(self._tray.geometry(), size))

    def _show_flyout(self) -> None:
        """Open the flyout above the icon, its Position picker rebuilt from
        the live screens (Decision 29)."""
        position = (
            self._overlay_window.current_position()
            if self._overlay_window is not None else dict(DEFAULT_POSITION)
        )
        self._flyout.set_positions(*position_entries(live_screens(), position))
        self._flyout.open_at(
            tray_placement.place_flyout(self._tray.geometry(), self._flyout.size())
        )

    def _on_flyout_dismissed(self) -> None:
        """The flyout hid. With the cursor on the icon the hide came from a
        mouse-down on the icon, whose Trigger must not reopen it."""
        self._debounce.flyout_hidden(tray_placement.cursor_in(self._tray.geometry()))

    # -- problems, the icon and the tooltip ----------------------------------

    def set_problems(self, active: list) -> None:
        """ProblemTracker.changed: the active problems, translator first. The
        icon's dot, the tooltip lines and the flyout status follow."""
        self._problems = [dict(entry) for entry in active]
        self._refresh_status()

    def on_problem_started(self, entry: dict) -> None:
        """ProblemTracker.started: one notification per problem start (lead
        decision 5) with the app icon like every toast (Decision 24); its
        click opens the flyout."""
        self.show_toast(entry["title"], entry["body"], on_click=self._show_flyout)

    def _icon_screen(self):
        """Decision 9's screen: the one under the icon, else the primary."""
        rect = self._tray.geometry()
        screen = QGuiApplication.screenAt(rect.center()) if rect.isValid() else None
        return screen or QGuiApplication.primaryScreen()

    def _on_screens_changed(self, *_args) -> None:
        self._watch_icon_screen()

    def _watch_icon_screen(self) -> None:
        """Follow the icon's screen: its geometryChanged and
        logicalDotsPerInchChanged (a resolution or scale change) redraw the
        icon at the new small-icon size (Decision 14). Then redraw."""
        screen = self._icon_screen()
        old = self._icon_screen_watched
        if screen is not old:
            if old is not None:
                try:
                    old.geometryChanged.disconnect(self._on_screens_changed)
                    old.logicalDotsPerInchChanged.disconnect(self._on_screens_changed)
                except (RuntimeError, TypeError):
                    pass  # a removed screen's C++ object may already be gone
            self._icon_screen_watched = screen
            if screen is not None:
                screen.geometryChanged.connect(self._on_screens_changed)
                screen.logicalDotsPerInchChanged.connect(self._on_screens_changed)
        self._refresh_icon()

    def _on_taskbar_theme_changed(self, _theme: str) -> None:
        """The taskbar switched light / dark: the dot's ring follows."""
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        """Draw the state icon: paused, else hidden subtitles, else listening,
        plus the amber dot while a problem is active, at the icon screen's
        small-icon size with the ring in the taskbar's color. setIcon only
        when something visible changed (Decision 67)."""
        screen = self._icon_screen()
        state = tray_icons.icon_state(self._paused, self._overlay_visible)
        key = (state, bool(self._problems), tray_icons.tray_icon_px(screen),
               self._taskbar.theme())
        if key == self._icon_key:
            return
        self._icon_key = key
        self._tray.setIcon(tray_icons.tray_icon(screen, state, key[1], key[3]))

    def _tooltip_text(self) -> str:
        """The status sentence (Decision 15): "Yamabiko is listening (Anime:
        Frieren)" or "Yamabiko is paused", then "Subtitles hidden" and each
        active problem's title, within _TIP_LIMIT (Decision 61)."""
        extra = [] if self._overlay_visible else ["Subtitles hidden"]
        extra.extend(entry["title"] for entry in self._problems)
        if self._paused:
            first = "Yamabiko is paused"
        else:
            genre = dict(GENRE_OPTIONS).get(self._genre, "Variety")
            media_id, label, _native = self._series
            title = series_title(label) if media_id is not None else ""
            if title:
                frame = f"Yamabiko is listening ({genre}: )"
                room = (_TIP_LIMIT - _utf16_len(frame)
                        - sum(_utf16_len(line) + 1 for line in extra))
                if _utf16_len(title) > room:
                    # Shortened by code points (never splitting a surrogate
                    # pair) until it and the ellipsis fit.
                    title = title[:max(room - 1, 0)]
                    while title and _utf16_len(title) > room - 1:
                        title = title[:-1]
                    title = title.rstrip() + "\u2026"
                first = f"Yamabiko is listening ({genre}: {title})"
            else:
                first = f"Yamabiko is listening ({genre})"
        return "\n".join([first, *extra])

    def _refresh_tooltip(self) -> None:
        text = self._tooltip_text()
        if text != self._tray.toolTip():
            self._tray.setToolTip(text)
