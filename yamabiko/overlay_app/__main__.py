"""Overlay app entry point.

Usage:
    python -m yamabiko.overlay_app

Assumes the pipeline is running at ws://127.0.0.1:8765/ws (default from
config.yaml). Use the `--url` flag to point at a different host/port.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QApplication

from .. import __version__, paths
from ..config import load_full_config, load_overlay_config
from ..config_writer import write_overlay_position, write_overlay_style
from ..gazetteer.gazetteer import feature_enabled
from ..hotkeys import shortcut_labels
from ..log_config import (
    MAIN_LOG_FORMAT,
    install_crash_evidence,
    log_startup_banner,
    setup_main_log,
)
from ..lifecycle.jobs import open_url_outside_job
from ..ui import thread_keeper
from ..ui.qt_bootstrap import install_qt_message_handler
from ..update_check import check_for_update
from .fonts import register_bundled_fonts
from .overlay_window import OverlayWindow
from .problem_view import ProblemTracker
from .series_palette import SeriesPalette
from .theme import palette_for
from .tray import Tray
from .ws_client import PipelineClient

log = logging.getLogger(__name__)


class _UpdateCheckThread(QThread):
    """Runs the blocking HTTP update check off the Qt main thread."""

    # Emits (version, release_url) when an update is available.
    updateAvailable = Signal(str, str)

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current_version = current_version

    def run(self) -> None:
        result = check_for_update(self._current_version)
        if result is not None:
            self.updateAvailable.emit(*result)


def _schedule_update_check(
    tray: Tray,
    parent=None,
    *,
    cfg: dict | None = None,
) -> _UpdateCheckThread | None:
    """Fire a background update check and wire the result to a tray toast.

    Returns the thread object (caller should keep a reference so it isn't
    garbage-collected before finishing), or None if the config toggle is off.
    """
    if cfg is None:
        cfg = load_full_config()
    if not cfg.get("updates", {}).get("check_on_startup", True):
        # INFO, not debug: "why did I never hear about vNext?" must be
        # answerable from a default-level log (logging audit 2026-07-01).
        # On the "yamabiko" logger, NOT this module's ``log``: when the
        # overlay is dev-spawned via `python -m yamabiko.overlay_app`,
        # __name__ is "__main__", so module-logger records bypass the
        # main-overlay.log handler entirely (same trap the startup
        # banner call in build_overlay documents).
        logging.getLogger("yamabiko").info("update check disabled by config")
        return None

    thread = _UpdateCheckThread(__version__, parent=parent)

    def _on_update(version: str, url: str) -> None:
        # show_toast binds the click to THIS toast only: a later toast
        # (diagnostics, series, settings notice) disconnects it, so its
        # click can't open the release page. The page opens outside the
        # Job Objects, so a pipeline restart or Quit can't kill the browser.
        tray.show_toast(
            f"Yamabiko {version} is out",
            "Click to see what's new and download it.",
            10_000,
            on_click=lambda: open_url_outside_job(url),
        )

    thread.updateAvailable.connect(_on_update)
    thread.start()
    return thread


def _release_update_thread(thread: _UpdateCheckThread | None) -> None:
    """Detach a still-running update-check worker at quit.

    The blocking httpx GET has a 5 s timeout, so a quit fired inside that
    window would let QApplication teardown destroy the running QThread
    (Qt6 qFatal "Destroyed while thread is still running"). Mirrors
    SeriesPalette.release_threads / Tray.release_threads: disconnect
    updateAvailable first so no tray notification fires mid-teardown, then
    park the thread in thread_keeper -- run_overlay's any_running() guard
    then hard-exits past interpreter finalization. No-op when the check was
    disabled by config (thread is None) or the worker already finished.
    """
    if thread is None or not thread.isRunning():
        return
    try:
        thread.updateAvailable.disconnect()
    except (RuntimeError, TypeError):
        pass
    thread_keeper.release(thread)


@dataclass
class OverlayApp:
    """Wired-but-not-running overlay. Composition Root output.

    Fields mirror what run_overlay() needs to drive the app. Test
    code builds one via build_overlay() and introspects it without
    hitting app.exec().
    """
    args: argparse.Namespace
    app: QApplication
    window: OverlayWindow
    client: PipelineClient
    tray: Tray
    palette: SeriesPalette | None
    screen_change_handler: Callable[..., None]
    color_scheme_handler: Callable[..., None]
    theme_palette: dict
    update_thread: QThread | None


def build_overlay(argv: list[str] | None = None) -> OverlayApp:
    """Parse argv, build the Qt application + overlay window + WS
    client + tray, wire signals, return an OverlayApp.

    Side effects: QApplication singleton is created (idempotent --
    reuses an existing instance in tests), Qt HiDPI rounding policy
    is installed, paths.ensure_dirs() runs, setup_main_log() attaches
    a file handler (tests monkeypatch this to no-op).

    Does NOT: start the WebSocket client, call app.exec(), or run any
    event loop. Those are run_overlay()'s job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws",
                        help="WebSocket URL of the pipeline overlay server")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    full_cfg = (
        load_full_config(explicit_path=args.config)
        if args.config is not None
        else load_full_config()
    )
    overlay_cfg = load_overlay_config(full_cfg)

    logging.basicConfig(
        level=args.log_level,
        # Shared constant so the console layout can't drift from the
        # main-*.log file layout (log_config.MAIN_LOG_FORMAT).
        format=MAIN_LOG_FORMAT,
    )
    # Per-process file: the overlay child owns main-overlay.log.
    # Sharing one main.log across watchdog/pipeline/overlay breaks
    # RotatingFileHandler rotation on Windows (rename of an open file
    # -> PermissionError -> records dropped, backups destroyed). The
    # E2E test tails a main-*.log glob, so overlay.mainwindow.shown
    # and Traceback markers are still asserted from one place.
    # Idempotent -- safe if some other entry point already called it.
    try:
        paths.ensure_dirs()
        setup_main_log(
            logging.getLogger("yamabiko"), level=args.log_level,
            filename="main-overlay.log",
        )
    except Exception:
        # Never let logging setup crash overlay startup. The
        # basicConfig above still works even if the file handler
        # can't attach.
        log.exception("setup_main_log failed; continuing with basicConfig only")

    # Crash evidence + identity banner for the overlay process
    # (faulthandler -> crash-overlay.log, excepthooks -> the log tree).
    # Both on the "yamabiko" logger explicitly: that's where the
    # main-overlay.log handler hangs, and this module's own logger is
    # named "__main__" when launched via `python -m yamabiko.overlay_app`
    # (dev mode), which would bypass it. Both helpers never raise.
    install_crash_evidence("overlay")
    log_startup_banner(logging.getLogger("yamabiko"), "overlay")

    # Route Qt's own diagnostics (qWarning / qCritical / platform-plugin
    # complaints) to the yamabiko.qt logger BEFORE QApplication
    # construction, so anything Qt emits while the app is being built
    # already lands in main-overlay.log instead of the windowed frozen
    # bundle's null stderr (2026-07-01 logging audit, Task 4). The
    # overlay builds its QApplication directly rather than via
    # qt_bootstrap.ensure_qapplication, so it adopts the routing with
    # this one idempotent call.
    install_qt_message_handler()

    # On fractional-DPI systems (125%, 150% Windows scaling),
    # Qt's default rounding policy snaps logical coordinates to whole
    # device pixels, which produces visible geometry shifts when the
    # overlay is resized, moved, or redrawn. PassThrough preserves
    # the fractional values through the Qt pipeline and lets the
    # compositor handle subpixel rendering -- more consistent
    # end-to-end behavior on HiDPI.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Yamabiko")
    app.setApplicationDisplayName("Yamabiko")
    app.setQuitOnLastWindowClosed(False)  # Tray keeps us alive

    # Bundled faces (fonts.FONT_FILES) must exist before any widget builds a
    # QFont from them. Logs overlay.fonts.registered, which the frozen E2E
    # asserts.
    register_bundled_fonts()

    paths.ensure_dirs()

    # Transcript consent for the render sentinel: the SAME opt-in that gates
    # latency.log's raw jp/en fields and main.py's drop-line snippets
    # (logging.transcript_log_enabled, default false). Default logs must
    # carry no transcript content -- bug_report.yml promises it -- so
    # OverlayWindow.add_subtitle keeps its overlay.subtitle.rendered marker
    # content-free unless the user opted in.
    transcript_enabled = bool(
        full_cfg.get("logging", {}).get("transcript_log_enabled", False)
    )

    # Resolve the chrome theme once at boot. The tray menu, settings window,
    # and placement hint all skin from this palette; "system" reads the OS
    # scheme (Qt colorScheme when an app exists, else the Windows registry).
    theme_palette = palette_for(overlay_cfg.theme)

    # One-time migration off the legacy free-geometry rect: map the saved
    # {x,y,width,height} to the anchor model + a max_width_pct derived from the
    # rect's width, persist both (the position write also prunes the legacy
    # overlay.geometry override), and pin the window with the migrated position.
    #
    # Two guards protect a user's placement:
    #  * NO-OVERLAP (migrated is None): the legacy rect overlaps no connected
    #    screen (undocked laptop whose placement was on an external monitor).
    #    Do NOT persist -- write_overlay_position would prune overlay.geometry
    #    and destroy the placement unrecoverably. Keep the rect on disk and use
    #    clamped defaults for THIS session only; the migration retries on a
    #    later boot when the monitor is back.
    #  * PERSIST FAILURE: a disk error during either write must never crash
    #    boot -- log and fall through with the migrated in-memory position.
    #
    # An explicit --config run is isolated from ambient user state, so it
    # never writes the user's overrides file: the migration still maps the
    # rect for this session, and every position write below is switched off.
    persist_position = args.config is None
    from .position_config import clamp_position, migrate_legacy_geometry
    from .placement import screens_snapshot
    position = overlay_cfg.position
    if overlay_cfg.geometry is not None:
        screens = screens_snapshot()
        migrated, max_width_pct, clamped = migrate_legacy_geometry(
            overlay_cfg.geometry, screens
        )
        if migrated is None:
            log.warning(
                "legacy geometry overlaps no connected screen; keeping it for "
                "a later launch and using defaults this session"
            )
            position = clamp_position(overlay_cfg.position)
        else:
            position = migrated
            if clamped:
                # The anchor model cannot represent this placement (edge-
                # hugging center, off-bottom edge, or a width outside
                # 30..100%), so the migration MOVED it -- and the position
                # write below prunes overlay.geometry in the same locked
                # write. Quote the original rect so the pre-migration
                # placement stays recoverable from the logs (they ship in
                # the collect-diagnostics zip).
                log.warning(
                    "legacy geometry %r does not fit the anchor bounds; "
                    "migrated (clamped) to position=%r max_width_pct=%r -- "
                    "this line is the only trace of the original rect once "
                    "overlay.geometry is pruned",
                    overlay_cfg.geometry, migrated, max_width_pct,
                )
            if max_width_pct is not None:
                # Apply in-memory BEFORE the window/tray are built and
                # UNCONDITIONALLY of persist success (mirror the position
                # handling above): overlay_cfg.style was loaded before the
                # migration ran, so without this the first session renders
                # at the stale 70 default -- and the tray's style seed
                # would carry 70 into the next full-style Settings write,
                # where config_writer prunes 70-as-default and silently
                # DELETES the just-migrated value from disk.
                overlay_cfg.style["max_width_pct"] = max_width_pct
            if persist_position:
                try:
                    write_overlay_position(position)   # also prunes overlay.geometry
                    if max_width_pct is not None:
                        write_overlay_style({"max_width_pct": max_width_pct})
                except Exception:
                    log.exception(
                        "geometry->position migration persist failed "
                        "(position + max_width_pct writes)"
                    )

    # Sanitized ready-state fingerprint for frozen acceptance and support
    # logs. Never emit a named monitor: display names can identify a user's
    # hardware, while primary-vs-named is sufficient to verify placement
    # source propagation.
    logging.getLogger("yamabiko").info(
        "overlay.config.applied theme=%s jp_size=%s screen_mode=%s "
        "v_offset_pct=%s",
        overlay_cfg.theme,
        overlay_cfg.style.get("jp_size"),
        "primary" if position.get("screen") == "primary" else "named",
        position.get("v_offset_pct"),
    )

    window = OverlayWindow(
        style=overlay_cfg.style, position=position,
        # None switches off every position write: presets, placement saves
        # (both persist through the window) and the quit-time save.
        on_position_changed=(
            write_overlay_position if persist_position else None
        ),
        transcript_enabled=transcript_enabled,
        # overlay.reduce_motion: every overlay transition instant from boot.
        reduce_motion=overlay_cfg.reduce_motion,
    )

    # F96, position-model edition: every shipped quit path (tray Quit,
    # shutdown event) goes through app.quit(), which exits exec() WITHOUT
    # delivering closeEvent to top-level widgets -- and PlacementController
    # only saves on Enter / click-away / WindowDeactivate, so quitting
    # DURING an active placement drag silently discarded the position.
    # aboutToQuit fires on every quit path while the window is still alive.
    # The save is conditional: it writes only a position that changed since
    # it was last saved (a drag in progress), because write_overlay_position
    # prunes overlay.geometry and an unconditional write would destroy the
    # legacy rect the no-overlap guard above kept on disk. The save must
    # never detonate the quit path, so any config-write failure is logged
    # and swallowed.
    def _save_position_on_quit() -> None:
        try:
            window.save_position_if_changed()
        except Exception:
            log.exception("position save on quit failed")

    app.aboutToQuit.connect(_save_position_on_quit)
    window.show()
    # Positive "overlay window visible to user" sentinel. Tests
    # (E2E + installer-smoke) assert this fires exactly once on
    # successful launch. Paired with pipeline.ready in yamabiko.main
    # -- absence of this line while pipeline.ready is present
    # indicates the overlay crashed between QApplication init and show().
    log.info("overlay.mainwindow.shown")

    # Re-pin the overlay when the primary screen changes or a screen is
    # added/removed (monitor hotplug, docking station, external-display
    # sleep). If the pinned screen vanished, OverlayWindow._revalidate_screen
    # falls back to the primary screen and re-margins the stack.
    #
    # Critical: these signals must be connected on the `app` INSTANCE,
    # not on QGuiApplication (the class). In PySide6, Signal access via
    # the class returns the Signal descriptor (no .connect); access via
    # a bound instance returns the bound signal. Class-level access
    # crashes on every overlay process spawn with AttributeError — see
    # tests/overlay_app/test_main_signal_connects.py and
    # tests/overlay_app/test_build_overlay.py.
    def _on_screens_changed(*_args) -> None:
        try:
            window._revalidate_screen()
        except Exception:
            log.exception("screen-change revalidation failed")
    app.primaryScreenChanged.connect(_on_screens_changed)
    app.screenAdded.connect(_on_screens_changed)
    app.screenRemoved.connect(_on_screens_changed)

    client = PipelineClient(url=args.url)
    client.subtitleReceived.connect(window.add_subtitle)
    # State frames drive the banner: genre, pause (the banner stays visible
    # while paused) and the picked series' Japanese title.
    client.stateReceived.connect(window.set_state)
    client.seriesReceived.connect(window.set_series)
    client.toggleOverlayReceived.connect(window.toggle_visible)

    # The series gazetteer is an opt-in feature. When it is off, the palette,
    # its Ctrl+Alt+S summon, and the tray series UI must all stay absent so the
    # feature is genuinely inert -- not merely unrestored at boot. build_overlay
    # is the composition root, so it gates these user-facing entry points here
    # (same helper as the pipeline: feature_enabled requires BOTH
    # gazetteer.enabled and romanization.enabled).
    _full_cfg = full_cfg
    _gaz_enabled = feature_enabled(_full_cfg)
    palette = None
    if _gaz_enabled:
        # The series bar. Built BEFORE the tray so its summon can be the
        # tray's on_change_series callback. Three summon routes converge on
        # it: the pipeline's Ctrl+Alt+S hotkey (-> showSeriesPaletteReceived),
        # the tray's Pick a series... item and the flyout's Change....
        palette = SeriesPalette(
            theme_palette=theme_palette,
            # The bar opens at the top center of the monitor the subtitles are
            # pinned to (overlay.position.screen), read at every summon.
            screen_provider=window.subtitle_screen,
            # overlay.reduce_motion at boot; a Settings change reaches the bar
            # through Tray._on_reduce_motion_changed.
            reduce_motion=overlay_cfg.reduce_motion,
        )
        # Pay the first render (fonts) now rather than on the first summon,
        # which would otherwise show an empty outline for about half a
        # second (probe p5_probe_report.md, Q2 finding 1).
        palette.prewarm()
        client.showSeriesPaletteReceived.connect(palette.summon)
        # Detach any in-flight search/cover/watch-list worker before widget
        # teardown on quit -- destroying a running QThread is a Qt6 qFatal
        # abort.
        app.aboutToQuit.connect(palette.release_threads)

    # Single Tray construction carrying BOTH feature seams: our themed-chrome
    # palette (theme_palette, the resolved overlay.theme dict) and the series
    # gazetteer's on_change_series summon (None when the feature is off, so the
    # tray series UI stays absent).
    tray = Tray(client, on_quit=app.quit, overlay_window=window,
                palette=theme_palette, series_palette=palette,
                on_change_series=(palette.summon if palette is not None else None),
                style_seed=overlay_cfg.style,
                config_path=(
                    args.config
                    if args.config is not None
                    else paths.user_overrides_path()
                ),
                settings_config=(full_cfg if args.config is not None else None),
                settings_read_only=(args.config is not None),
                # gazetteer.auto_genre: a series pick also switches genre to
                # "anime" (toggleable via the settings-window checkbox).
                auto_genre=bool(
                    (_full_cfg.get("gazetteer", {}) or {}).get("auto_genre", True)
                ),
                # overlay.reduce_motion: rides every apply_style so the browser
                # page follows Settings.
                reduce_motion=overlay_cfg.reduce_motion,
                # The combos the short menu and the flyout tooltips show
                # (None while hotkeys.enabled is false).
                shortcuts=shortcut_labels(_full_cfg))
    # Same quit-time detach for the tray's in-flight gazetteer build worker.
    app.aboutToQuit.connect(tray.release_threads)
    # The flyout, the icon and the tooltip follow the window's real
    # visibility: Ctrl+Alt+O, the Subtitles tile, placement's flag change.
    # The window was shown above, before the tray existed.
    window.visibleChanged.connect(tray.set_overlay_visible)
    tray.set_overlay_visible(window.isVisible())
    # Problems (lead decisions 1 and 5): the tracker diffs the pipeline's
    # problems frames by kind, keeping the last set across reconnects. The
    # overlay window shows a line per active problem and the recovery line
    # when one clears; the tray shows the icon dot, the tooltip lines, the
    # flyout status and one notification per start. Per frame the tracker
    # emits changed, then recovered, then started (Decision 22). Parented
    # to the window it feeds.
    problem_tracker = ProblemTracker(window)
    client.problemsReceived.connect(problem_tracker.apply)
    problem_tracker.changed.connect(window.set_problems)
    problem_tracker.recovered.connect(window.show_recovery)
    problem_tracker.changed.connect(tray.set_problems)
    problem_tracker.started.connect(tray.on_problem_started)
    if palette is not None:
        # A pick in the palette drives the tray's build+cache+set_series flow.
        palette.series_chosen.connect(tray.on_series_chosen)

    # Live OS light/dark follow (system theme mode only). When Windows flips
    # its colour scheme, recompute the palette and re-skin the runtime chrome
    # (tray menu + any open settings window + active placement hint). The
    # handler re-reads the PERSISTED overlay.theme every time -- never a
    # boot-cached value -- so an explicit light/dark pick made after boot
    # stops the OS from overriding it. A theme flip is cosmetic: never let it
    # crash the overlay.
    def _on_color_scheme_changed(*_args) -> None:
        try:
            # Read the COERCED theme via load_overlay_config().theme, NOT the
            # raw dict value. load_overlay_config maps every invalid persisted
            # value ('Dark ', 1, "") to "system" -- the same coercion boot uses
            # to decide OS-following. A raw `cfg["overlay"]["theme"]` read would
            # early-return on such a config (theme: 'Dark ' boots OS-following,
            # yet 'Dark ' != "system"), leaving the follow silently dead. This
            # keeps the runtime follow and the boot decision on ONE reader.
            current_overlay_cfg = (
                load_overlay_config(args.config)
                if args.config is not None
                else load_overlay_config()
            )
            if current_overlay_cfg.theme != "system":
                return
            tray.apply_palette(palette_for("system"))
        except Exception:
            log.exception("colorScheme follow handler failed")

    # Connect on the `app` INSTANCE's styleHints (a bound signal), never the
    # QGuiApplication class -- the same class-vs-instance Signal trap the
    # screen-signal block above documents. Connect whenever the signal exists
    # (colorSchemeChanged exists on Qt >= 6.5 / PySide6 6.11); the HANDLER is
    # the sole gate. It re-reads the persisted overlay.theme on every call and
    # returns unless it is still "system", so booting forced-light and then
    # switching to System in Settings starts OS-following immediately -- no
    # relaunch. A boot-theme gate here would instead leave that switch dead
    # until the next launch.
    style_hints = app.styleHints()
    if hasattr(style_hints, "colorSchemeChanged"):
        style_hints.colorSchemeChanged.connect(_on_color_scheme_changed)

    # Fire-and-forget update check after the tray is visible.
    # Reference retained on OverlayApp so it isn't GC'd mid-flight.
    update_thread = _schedule_update_check(tray, parent=app, cfg=full_cfg)
    # Same quit-time detach as the palette/tray workers: a quit inside the
    # update check's 5 s HTTP window would otherwise destroy the running
    # QThread at QApplication teardown (Qt6 qFatal). Gated on the thread
    # existing (None when the check is disabled by config).
    if update_thread is not None:
        app.aboutToQuit.connect(lambda: _release_update_thread(update_thread))

    return OverlayApp(
        args=args,
        app=app,
        window=window,
        client=client,
        tray=tray,
        palette=palette,
        screen_change_handler=_on_screens_changed,
        color_scheme_handler=_on_color_scheme_changed,
        theme_palette=theme_palette,
        update_thread=update_thread,
    )


def run_overlay(overlay: OverlayApp) -> int:
    """Start the WebSocket client, drive app.exec(), tear down on exit.

    If the pipeline passed us a named Win32 Event via
    YAMABIKO_SHUTDOWN_EVENT, install a Qt-side listener so a parent
    SetEvent cleanly quits the overlay before the inner-job
    KILL_ON_JOB_CLOSE fallback fires. The shared ShutdownEventListener
    runs a daemon thread blocked on WaitForMultipleObjects (the named
    event plus a private cancel event); a QTimer polls the tripped flag
    at 100ms cadence and calls app.quit() once. Qt's event loop is NOT
    asyncio: call_soon_threadsafe isn't an option here, so the flag +
    QTimer is the bridge. On teardown shutdown_listener.close() SetEvents
    the cancel event and joins the thread BEFORE closing any handle --
    closing a handle while a wait on it is pending is undefined behavior.
    """
    from yamabiko.lifecycle.shutdown_event import ShutdownEventListener
    shutdown_listener = ShutdownEventListener.from_env(role="overlay")
    _shutdown_timer = None
    if shutdown_listener is not None and shutdown_listener.handle is not None:
        from PySide6.QtCore import QTimer

        shutdown_flag = {"tripped": False, "fired": False}

        def _poll() -> None:
            if shutdown_flag["tripped"] and not shutdown_flag["fired"]:
                shutdown_flag["fired"] = True
                log.info("overlay.shutdown_event.fired; quitting")
                overlay.app.quit()

        # Keep timer bound to the QApplication so it isn't GC'd while
        # the event loop is running.
        _shutdown_timer = QTimer(overlay.app)
        _shutdown_timer.timeout.connect(_poll)
        _shutdown_timer.start(100)  # 100ms poll -- cheap; UI-idle anyway

        def _on_shutdown_event() -> None:
            # Runs on the listener's daemon watcher thread. Qt's loop isn't
            # asyncio, so we can't call_soon_threadsafe; flip a flag the
            # QTimer above polls and quits on the main thread.
            shutdown_flag["tripped"] = True

        # Reuse the shared listener's cancellable watcher (waits on the
        # named event AND a private cancel event via WaitForMultipleObjects)
        # so the shutdown_listener.close() in teardown can wake and join
        # this thread BEFORE closing any handle. The previous hand-rolled
        # waiter blocked on WaitForSingleObject and had its handle closed
        # out from under the pending wait (undefined behavior).
        shutdown_listener.start_flag_watcher(_on_shutdown_event)

    overlay.client.start()
    rc = overlay.app.exec()
    overlay.client.stop()
    if shutdown_listener is not None:
        try:
            shutdown_listener.close()
        except Exception:
            log.exception("overlay shutdown listener close failed")
    if thread_keeper.any_running():
        # A detached worker (palette search/cover, or the tray gazetteer
        # build) is still blocked in httpx. Interpreter finalization would
        # destroy its QThread wrapper while the OS thread runs (Qt6 qFatal)
        # -- hard-exit with the loop's rc instead (same contract as main.py's
        # wizard-cancel path).
        log.info("overlay exiting via os._exit(%d): detached worker still running", rc)
        logging.shutdown()
        os._exit(rc)
    return rc


def main(argv: list[str] | None = None) -> int:
    return run_overlay(build_overlay(argv))


if __name__ == "__main__":
    sys.exit(main())
