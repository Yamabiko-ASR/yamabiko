"""Wiring tests for overlay_app.__main__ (build_overlay() plus the
module's logging wiring: crash evidence, startup banner, Qt-diagnostics
routing, update-check logger routing).

overlay_app main() is a 5-line shell delegating to
build_overlay(argv) -> OverlayApp and run_overlay(app) -> int.
Tests here call build_overlay directly and introspect the returned
dataclass -- no monkeypatch harness required.

Seeded defects (each listed assertion goes RED if the paired
production change is reverted):

- assert overlay.window is not None:
    delete `window = OverlayWindow(...)` in build_overlay.
- assert overlay.app is not None:
    delete `app = QApplication(...)` in build_overlay.
- assert callable(overlay.screen_change_handler):
    rename `_on_screens_changed` to something else or drop the
    attribute from OverlayApp.
- the "connects without AttributeError" behaviour:
    revert the screen-signal fix (access
    `QGuiApplication.primaryScreenChanged` instead of
    `app.primaryScreenChanged`) and build_overlay raises
    AttributeError during the connect().
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication


class _OverlayEnv:
    """Return value of the _overlay_env fixture. Exposes the module
    under test AND the QApplication singleton the fixture injected,
    so tests can assert identity (`overlay.app is env.app_singleton`)
    rather than `isinstance(overlay.app, QApplication)`.
    """
    def __init__(self, module, app_singleton):
        self.module = module
        self.app_singleton = app_singleton


@pytest.fixture
def _overlay_env(monkeypatch, tmp_path):
    """Isolate LOCALAPPDATA + skip HTTP update check + skip file log.

    QApplication is a process-global singleton; reuse any existing
    instance rather than constructing a second one. The singleton
    reference is returned via _OverlayEnv so wiring tests can
    identity-check the app field rather than fall back to isinstance.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    import yamabiko.overlay_app.__main__ as overlay_main

    # Skip HTTP update check (would hit GitHub in the real path).
    monkeypatch.setattr(
        overlay_main, "_schedule_update_check", lambda *a, **kw: None,
    )
    # Skip rotating file-handler setup (charter: don't leak handlers
    # across tests; see test_main_signal_connects.py for context).
    monkeypatch.setattr(overlay_main, "setup_main_log", lambda *a, **kw: None)
    # Crash-evidence arming is process-global (sys/threading excepthook
    # chains + a process-lifetime faulthandler handle): stub it so
    # build_overlay() calls here never swap the pytest process's real
    # excepthooks (same hygiene as tests/test_build_watchdog.py's
    # _watchdog_env -- but patched on THIS module's binding, because the
    # overlay imports the name directly; patching log_config would miss
    # it). Helper behavior is owned by tests/test_log_config.py +
    # tests/test_real_crash_evidence.py; the call-site wiring is pinned
    # by test_build_overlay_installs_crash_evidence_for_overlay_role,
    # which re-patches this stub with a recorder.
    monkeypatch.setattr(overlay_main, "install_crash_evidence", lambda role: None)
    # Qt-diagnostics routing is process-global too (qInstallMessageHandler
    # + qt_bootstrap's module-level installed flag): stub it by default;
    # the dedicated wiring/behavioral tests re-patch with a recorder /
    # the real helper and restore Qt's default handler afterwards.
    monkeypatch.setattr(overlay_main, "install_qt_message_handler", lambda: None)

    # QApplication singleton: reuse if one exists, else construct.
    # build_overlay's own QApplication(sys.argv) would raise if an
    # instance already exists, so monkeypatch the name in the module.
    app_singleton = QApplication.instance() or QApplication([])

    class _AppProxy:
        def __new__(cls, argv):
            return app_singleton

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy):
            pass
    monkeypatch.setattr(overlay_main, "QApplication", _AppProxy)

    return _OverlayEnv(module=overlay_main, app_singleton=app_singleton)


def test_build_overlay_returns_wired_app_dataclass(_overlay_env):
    """build_overlay([]) returns an OverlayApp wired to the expected
    objects. Combines two strength levels:

      - app: IDENTITY against the fixture's injected QApplication
        singleton. Strongest assertion -- proves build_overlay used
        the proxied constructor seam, not some other path to a
        QApplication.

      - window / client / tray: isinstance against the concrete
        types. No injectable seam for these (they're constructed
        directly inside build_overlay), so identity isn't meaningful;
        isinstance still catches the field-swap class of bug
        (tray=window by mistake).

    Seeded-defect checks:
      - Swap field assignments in the OverlayApp return
        (`window=tray, tray=window`) -> isinstance assertion fails.
      - Route build_overlay's QApplication construction through a
        non-proxied path -> identity assertion fails.
    """
    env = _overlay_env
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert overlay.app is env.app_singleton, (
        "overlay.app is not the QApplication singleton the fixture "
        "injected via _AppProxy; build_overlay must have bypassed "
        "the proxied constructor"
    )
    assert isinstance(overlay.window, env.module.OverlayWindow), (
        f"overlay.window is not an OverlayWindow; got {type(overlay.window).__name__}"
    )
    assert isinstance(overlay.client, env.module.PipelineClient), (
        f"overlay.client is not a PipelineClient; got {type(overlay.client).__name__}"
    )
    assert isinstance(overlay.tray, env.module.Tray), (
        f"overlay.tray is not a Tray; got {type(overlay.tray).__name__}"
    )


def test_build_overlay_seeds_tray_current_style_from_overlay_config(
    _overlay_env, monkeypatch,
):
    """build_overlay must seed Tray._current_style from the config-loaded
    overlay style (overlay_cfg.style), so it matches the style the OverlayWindow
    was constructed with. Without this the tray falls back to factory
    STYLE_DEFAULTS, and a WS-down preview+close (_on_preview_ended) reskins the
    live overlay to defaults -- the P0-3 trigger (a) boot seam.

    Seeded defect: drop the `style_seed=overlay_cfg.style` kwarg from the Tray
    construction in build_overlay -> the tray seeds STYLE_DEFAULTS and
    _current_style['jp_size'] is 14, not 31 -> red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    custom = {**STYLE_DEFAULTS, "jp_size": 31}
    # Feed a config-loaded overlay style through the SAME seam build_overlay
    # reads (overlay_main.load_overlay_config); geometry=None so the boot
    # migration block is skipped.
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *a, **k: OverlayConfig(style=dict(custom)),
    )

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert overlay.tray._current_style["jp_size"] == 31, (
        "build_overlay did not seed the tray's _current_style from the "
        "config-loaded overlay style; a WS-down preview close would reset the "
        f"overlay to factory defaults. Got {overlay.tray._current_style['jp_size']}"
    )
    # The overlay window and the tray's known-applied style must agree at boot,
    # so previewEnded restores exactly what the window is already showing.
    assert overlay.window._style["jp_size"] == 31


def test_state_and_series_frames_reach_the_overlay_banner(_overlay_env):
    """Production wiring: the client's state and series signals drive the
    window's banner (genre, series title, paused).

    Seeded defect: drop either connect in build_overlay -> red."""
    overlay = _overlay_env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    try:
        overlay.client.stateReceived.emit("anime", False)
        assert overlay.window.stack.frame().banner_text == ("アニメ", "Anime")
        overlay.client.seriesReceived.emit(1, "Frieren (2023)", "葬送のフリーレン")
        assert overlay.window.stack.frame().banner_text == ("葬送のフリーレン", "Frieren")
        overlay.client.stateReceived.emit("anime", True)
        assert overlay.window.stack.frame().paused is True
    finally:
        overlay.window.close()


def test_toggle_overlay_frames_flip_the_window_and_the_flyout_follows(_overlay_env):
    """Production wiring (Decisions 11 and 36): a toggle_overlay frame
    (Ctrl+Alt+O or the flyout's Subtitles tile) flips the overlay window, and
    its visibleChanged keeps the flyout's Subtitles tile current.

    Seeded defects: drop either connect in build_overlay -> red."""
    overlay = _overlay_env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    try:
        assert overlay.window.isVisible()
        overlay.client.toggleOverlayReceived.emit()
        assert not overlay.window.isVisible()
        assert overlay.tray._flyout._subtitles.value() == "Hidden"
        overlay.client.toggleOverlayReceived.emit()
        assert overlay.window.isVisible()
        assert overlay.tray._flyout._subtitles.value() == "Shown"
    finally:
        overlay.window.close()
        overlay.tray._tray.hide()


def test_problems_frames_reach_the_tray_and_the_overlay_lines(_overlay_env, monkeypatch):
    """Production wiring (Task 13): client.problemsReceived -> ProblemTracker
    -> the overlay window's notice lines (set_problems, then show_recovery
    when a problem clears) and the tray (notification, tooltip).

    Seeded defects: drop any connect of the problem block in build_overlay
    -> red."""
    from yamabiko import problems

    env = _overlay_env
    calls: list = []
    monkeypatch.setattr(env.module.OverlayWindow, "set_problems",
                        lambda self, active: calls.append(("lines", [p["kind"] for p in active])))
    monkeypatch.setattr(env.module.OverlayWindow, "show_recovery",
                        lambda self, kind, text: calls.append(("recovery", kind, text)))
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    shown: list = []
    monkeypatch.setattr(overlay.tray._tray, "showMessage", lambda *a: shown.append(a[:2]))
    try:
        overlay.client.problemsReceived.emit([problems.translator_problem(True)], [])
        assert shown == [("Translations stopped",
                          "The translator isn't responding. Yamabiko is restarting it.")]
        assert "Translations stopped" in overlay.tray._tray.toolTip()
        overlay.client.problemsReceived.emit([], [])
        assert calls == [("lines", ["translator"]), ("lines", []),
                         ("recovery", "translator", "Translations are back.")]
        assert "Translations stopped" not in overlay.tray._tray.toolTip()
    finally:
        overlay.window.close()
        overlay.tray._tray.hide()


def test_build_overlay_seeds_reduce_motion_into_window(_overlay_env):
    """overlay.reduce_motion from config reaches the overlay window at boot.

    Seeded defect: build OverlayWindow without reduce_motion -> red."""
    from yamabiko import paths

    overrides = paths.user_overrides_path()
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text("overlay:\n  reduce_motion: true\n", encoding="utf-8")
    overlay = _overlay_env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    try:
        assert overlay.window.stack._reduce_motion is True, (
            "overlay.reduce_motion: true did not reach the overlay window at boot"
        )
    finally:
        overlay.window.close()


def test_build_overlay_logs_to_overlay_role_file(
    _overlay_env, monkeypatch, tmp_path, hermetic_log_handlers,
):
    """build_overlay routes file logging to main-overlay.log -- its own
    per-process file -- never the shared main.log (Windows
    RotatingFileHandler rotation breaks when multiple processes hold
    the same file open).

    The fixture stubs setup_main_log for handler hygiene; the subject
    here IS that call site, so restore the real function for this test
    under the hermetic_log_handlers guard.

    Fails red if: the build_overlay call site drops the
    filename="main-overlay.log" argument, or stops calling
    setup_main_log entirely (role file never created -- the
    RotatingFileHandler opens its file at construction, so existence
    proves attachment).
    """
    from yamabiko import log_config as _lc

    env = _overlay_env
    monkeypatch.setattr(env.module, "setup_main_log", _lc.setup_main_log)
    # YAMABIKO_APPDATA wins over the LOCALAPPDATA the fixture set.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    with hermetic_log_handlers("", "yamabiko"):
        env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
        logs_dir = tmp_path / "logs"
        role_log = logs_dir / "main-overlay.log"
        assert role_log.is_file(), (
            f"main-overlay.log missing after build_overlay; logs dir: "
            f"{[p.name for p in logs_dir.iterdir()] if logs_dir.is_dir() else 'absent'}"
        )
        assert not (logs_dir / "main.log").exists(), (
            "build_overlay wrote the shared main.log; cross-process "
            "rotation contention is back"
        )


# --- crash evidence + identity banner + Qt-diagnostics routing ---
# (2026-07-01 logging audit Tasks 1/4/5; call-site wiring for the
# overlay process. Helper BEHAVIOR is owned by tests/test_log_config.py,
# tests/test_real_crash_evidence.py and tests/test_qt_bootstrap.py --
# these tests pin the build_overlay call sites, which the rest of the
# suite exercised but never asserted on: deleting any of the three
# calls used to leave every test green.)


def test_build_overlay_installs_crash_evidence_for_overlay_role(
    _overlay_env, monkeypatch,
):
    """build_overlay must arm the shared crash-evidence primitives
    (faulthandler -> crash-overlay.log, sys/threading excepthooks) for
    the "overlay" role right after log setup.

    Call-site wiring test: the recorder replaces the fixture's no-op
    stub. Asserting on disk state / live hooks here would leak
    process-global excepthooks into the pytest session (install is
    idempotent per-role per-process), which is exactly what the
    fixture stub exists to prevent.

    Seeded-defect drill: delete the ``install_crash_evidence("overlay")``
    call from build_overlay -> recorded stays empty -> red.
    """
    env = _overlay_env
    recorded: list = []
    monkeypatch.setattr(
        env.module, "install_crash_evidence",
        lambda role: recorded.append(role),
    )

    env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert recorded == ["overlay"], (
        f"install_crash_evidence calls: {recorded!r}; the overlay "
        f"process runs without faulthandler/excepthook evidence -- an "
        f"overlay crash leaves nothing in the logs"
    )


def test_build_overlay_logs_startup_banner_for_overlay_role_on_yamabiko_logger(
    _overlay_env, caplog,
):
    """build_overlay must emit the identity banner for the "overlay"
    role, and it must land on the literal "yamabiko" logger -- NOT this
    module's own ``log``. When the overlay is dev-spawned via
    ``python -m yamabiko.overlay_app``, the module logger is named
    "__main__", which sits outside the yamabiko tree and never reaches
    the main-overlay.log file handler. Under pytest import the module
    logger is "yamabiko.overlay_app.__main__", which deceptively DOES
    propagate -- so the logger-name assert here must be exact, or the
    test is a Liar that passes over the dev-spawn trap.

    Seeded-defect drills:
      - delete the ``log_startup_banner(...)`` call from build_overlay
        -> no banner record -> red.
      - change the call's logger to this module's ``log`` -> record
        name becomes "yamabiko.overlay_app.__main__" -> red.
    """
    import logging as _logging
    import re

    caplog.set_level(_logging.INFO, logger="yamabiko")
    env = _overlay_env

    env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    banner_records = [
        r for r in caplog.records
        if re.search(r"yamabiko overlay v\S+ \((frozen|dev)\) pid=\d+",
                     r.getMessage())
    ]
    assert len(banner_records) == 1, (
        f"expected exactly 1 overlay identity banner; got "
        f"{len(banner_records)}. Records: "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
    assert banner_records[0].name == "yamabiko", (
        f"banner logged on {banner_records[0].name!r}; must be the "
        f"literal 'yamabiko' logger so dev-spawn (`python -m`, where "
        f"__name__ == '__main__') still reaches the file handler"
    )


def test_build_overlay_installs_qt_message_handler_before_qapplication(
    _overlay_env, monkeypatch,
):
    """build_overlay must install the shared Qt-diagnostics routing
    (qt_bootstrap.install_qt_message_handler) BEFORE constructing the
    QApplication, so qWarning/qCritical Qt emits during construction
    (DPI complaints, platform-plugin failures) already land in
    main-overlay.log instead of the frozen bundle's null stderr.

    Seeded-defect drills: delete the install_qt_message_handler() call
    from build_overlay -> no 'install' entry -> red; move it after
    QApplication(sys.argv) -> ordering assert red.
    """
    env = _overlay_env
    order: list[str] = []
    monkeypatch.setattr(
        env.module, "install_qt_message_handler",
        lambda: order.append("install"),
    )

    class _OrderingAppProxy:
        def __new__(cls, _argv):
            order.append("create")
            return env.app_singleton

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(_policy):
            pass

    monkeypatch.setattr(env.module, "QApplication", _OrderingAppProxy)

    env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert order.count("install") == 1, (
        f"install_qt_message_handler called {order.count('install')} "
        f"times (order={order!r}); Qt diagnostics from the overlay "
        f"process are lost in the frozen bundle without it"
    )
    assert order.index("install") < order.index("create"), (
        f"Qt message handler installed AFTER QApplication construction "
        f"(order={order!r}); construction-time Qt diagnostics are lost"
    )


def test_build_overlay_installed_handler_routes_qwarning_to_yamabiko_qt_logger(
    _overlay_env, monkeypatch, caplog,
):
    """Behavioral pair for the wiring test above: after build_overlay
    (with the REAL install_qt_message_handler restored over the
    fixture's hygiene stub), a real PySide6 qWarning dispatch lands as
    a WARNING record on the yamabiko.qt logger.

    Seeded-defect drill: delete the install_qt_message_handler() call
    from build_overlay -> qWarning goes to Qt's default stderr writer,
    no record captured -> red.
    """
    import logging as _logging

    import yamabiko.ui.qt_bootstrap as qb
    from PySide6.QtCore import qInstallMessageHandler, qWarning

    env = _overlay_env
    # The fixture stubs the module-local name for hygiene; the subject
    # here IS that call site, so route it back to the real helper and
    # reset the idempotence flag (monkeypatch restores it at teardown).
    monkeypatch.setattr(
        env.module, "install_qt_message_handler",
        qb.install_qt_message_handler,
    )
    monkeypatch.setattr(qb, "_qt_handler_installed", False)
    try:
        env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
        with caplog.at_level(_logging.DEBUG, logger="yamabiko.qt"):
            qWarning("overlay-build-qt-probe")
    finally:
        # Restore the EXACT handler the install displaced (pytest-qt's
        # per-test capture when the plugin is active, else Qt's default).
        # Resetting to None here left pytest-qt's teardown restoring over
        # a chain it no longer owned -- the mid-test churn suspected in a
        # once-seen CI heap corruption (0xC0000374) during the later
        # widget tests' event drain.
        qInstallMessageHandler(qb._displaced_qt_handler)

    records = [
        r for r in caplog.records
        if r.name == "yamabiko.qt"
        and r.getMessage() == "overlay-build-qt-probe"
    ]
    assert len(records) == 1, (
        f"qWarning did not route through the handler build_overlay "
        f"installed; yamabiko.qt records: "
        f"{[r.getMessage() for r in caplog.records if r.name == 'yamabiko.qt']!r}"
    )
    assert records[0].levelno == _logging.WARNING, records[0].levelno


def test_build_overlay_connects_screen_signals_via_instance(_overlay_env):
    """build_overlay([]) must connect screen-change signals via the
    QApplication INSTANCE, not the QGuiApplication class.

    Class-level access returns the Signal descriptor, whose .connect
    is unbound -> AttributeError during build_overlay. This test fails
    at build_overlay() call time, not at overlay.app.exec().

    Seeded-defect check: change `app.primaryScreenChanged` back to
    `_QGuiApp.primaryScreenChanged` inside build_overlay ->
    build_overlay raises AttributeError.
    """
    env = _overlay_env
    # The assertion IS the successful return from build_overlay -- if
    # the signal connect raises, the call above propagates.
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    # Structural probe: the handler must be reachable from the
    # OverlayApp so future wiring changes can't orphan it from the
    # test surface. Behavioural assertion (handler does the right
    # thing) lives in a separate test below.
    assert callable(overlay.screen_change_handler), (
        "OverlayApp.screen_change_handler missing; regression would "
        "silently drop screen-change geometry validation"
    )


def test_build_overlay_screen_change_handler_invokes_revalidate_screen(_overlay_env):
    """The screen_change_handler attribute must be wired to a closure
    that invokes OverlayWindow._revalidate_screen().

    The two earlier overlay tests only check that the handler is
    callable and that build_overlay doesn't raise on connect(). Both
    pass even if someone replaced the real closure with a no-op
    (`screen_change_handler=lambda *a: None`) in the OverlayApp
    return. This test closes that gap: assert on the SIDE EFFECT of
    invoking the handler.

    Seeded-defect check: change
        screen_change_handler=_on_screens_changed
    to
        screen_change_handler=lambda *a: None
    in build_overlay's OverlayApp(...) return -> the handler is
    still callable, but invoking it does not call
    window._revalidate_screen, so the call counter stays at 0, test
    fails with "got 0 invocations".

    When Qt emits primaryScreenChanged / screenAdded / screenRemoved
    (monitor hotplug, dock/undock, sleep), the overlay re-pins to its
    screen (falling back to primary if the pinned one vanished). If the
    handler were swapped for a no-op, every screen event silently skips
    the re-pin and users see the overlay stuck on a now-absent monitor.
    """
    env = _overlay_env
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    # Replace _revalidate_screen on the live window with a counter. The
    # real _on_screens_changed wraps its call in try/except, but our
    # tracker doesn't raise, so the branch we exercise is the happy path.
    invocations: list[int] = []
    overlay.window._revalidate_screen = lambda: invocations.append(1)

    # Invoke as Qt would when a screen changes. Signature is
    # `*_args` so no arguments is fine.
    overlay.screen_change_handler()

    assert len(invocations) == 1, (
        f"screen_change_handler did not invoke "
        f"window._revalidate_screen; got {len(invocations)} "
        f"invocations. Handler likely wired to a no-op, which would "
        f"silently drop screen-change re-pinning."
    )


# --- subtitle wire: client.subtitleReceived -> window.add_subtitle ---


def test_subtitle_received_signal_lands_subtitle_in_overlay_window(_overlay_env):
    """A subtitle frame on the PipelineClient must reach the OverlayWindow:
    build_overlay wires client.subtitleReceived -> window.add_subtitle
    (production __main__.py). Emitting the signal -- exactly as ws_client
    does on a 'subtitle' frame -- must add one line, with its English block,
    to the window's subtitle stack.

    This is the ONLY test that exercises that connect (H-11a). The sibling
    tests cover the other client .connect lines (series pick, palette
    summon, screen-change); the subtitle wire -- the app's primary output
    path -- was previously unexercised end to end.

    Asserts the OBSERVABLE effect (a real line landed in the stack frame),
    not connection state, so a no-op/wrong-target slot cannot pass.

    Seeded-defect drill: delete or retarget the
    ``client.subtitleReceived.connect(window.add_subtitle)`` line in
    build_overlay -> the emission reaches no slot -> the stack frame stays
    empty -> the ``len(added) == 1`` assert goes red.
    """
    env = _overlay_env
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    before = len(overlay.window.stack.frame().lines)

    # Realistic payload for the typed Signal(str, str, list): jp, en, jp_ruby.
    overlay.client.subtitleReceived.emit(
        "水が飲みたい", "I want to drink water", [],
    )

    added = overlay.window.stack.frame().lines[before:]
    assert len(added) == 1, (
        f"subtitleReceived did not reach window.add_subtitle: the overlay "
        f"stack did not grow (before={before}, "
        f"after={len(overlay.window.stack.frame().lines)}). The "
        f"subtitleReceived -> add_subtitle connect is missing or retargeted."
    )
    assert "en" in added[0].blocks, (
        f"the landed line has no English block: {sorted(added[0].blocks)}"
    )


def test_build_overlay_gates_render_marker_text_on_transcript_consent(
    _overlay_env, monkeypatch, caplog,
):
    """build_overlay must plumb logging.transcript_log_enabled (the ONE
    transcript-consent flag; default false) into the OverlayWindow, so the
    ``overlay.subtitle.rendered`` sentinel in add_subtitle carries the raw EN
    text ONLY when the user opted in. Behavioral, end to end across the
    config seam: build with the flag on -> the marker carries the EN text;
    rebuild with the flag absent (resolves to the shipped-default off) ->
    the marker is content-free. Without the plumbing, main-overlay.log --
    which ships in the collect-diagnostics zip -- would carry every rendered
    translation on a default install, breaking the bug_report.yml promise.

    Seeded-defect drills (each one line, each red):
      - drop the ``transcript_enabled=transcript_enabled`` kwarg from
        build_overlay's OverlayWindow(...) construction -> the consent-on
        arm's marker stays content-free -> red.
      - read the wrong config key (e.g. ``latency_log_enabled``) in
        build_overlay -> same.
    """
    import logging as _logging

    _marker_logger = "yamabiko.overlay_app.overlay_window"

    def _marker_after_render(overlay, en_probe: str) -> str:
        with caplog.at_level(_logging.INFO, logger=_marker_logger):
            overlay.window.add_subtitle("水が飲みたい", en_probe, jp_ruby=[])
        markers = [
            r.getMessage() for r in caplog.records
            if r.name == _marker_logger
            and r.getMessage().startswith("overlay.subtitle.rendered")
        ]
        assert len(markers) == 1, (
            f"expected exactly one render marker; got {markers!r}"
        )
        return markers[0]

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_full_config",
        lambda: {"logging": {"transcript_log_enabled": True}},
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    on_marker = _marker_after_render(overlay, "consent-on-probe")
    assert "consent-on-probe" in on_marker, (
        f"logging.transcript_log_enabled=true did not reach the overlay "
        f"render sentinel; marker stayed content-free: {on_marker!r}"
    )

    caplog.clear()
    # Flag absent entirely -> the read must default to off (the shipped
    # default_config.yaml value is false; an empty dict exercises the same
    # .get() default path).
    monkeypatch.setattr(env.module, "load_full_config", lambda: {})
    overlay_off = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    off_marker = _marker_after_render(overlay_off, "consent-off-probe")
    assert "consent-off-probe" not in off_marker, (
        f"default (consent off) build leaked the EN text into the render "
        f"marker: {off_marker!r}"
    )


# --- series palette wiring (summon path + selection path) ---


def _set_gaz_enabled(env, monkeypatch, enabled: bool) -> None:
    """Pin the effective gazetteer feature for build_overlay's gate.
    build_overlay reads config via load_full_config() and gates on
    feature_enabled (BOTH gazetteer.enabled AND romanization.enabled), so
    romanization is pinned on here to isolate the gazetteer flag. The
    romanization-off divergence has its own dedicated test below. Patch the
    module-local binding so the result is deterministic regardless of the
    ambient shipped default."""
    monkeypatch.setattr(
        env.module, "load_full_config",
        lambda: {
            "gazetteer": {"enabled": enabled},
            "romanization": {"enabled": True},
        },
    )


def test_build_overlay_retains_palette_as_series_palette(_overlay_env, monkeypatch):
    """build_overlay must build a SeriesPalette and retain it on the
    OverlayApp so it isn't GC'd between the summon hotkey and the pick.

    Seeded-defect drill: drop `palette=palette` from the OverlayApp return
    (or never build it) -> the field is missing/None -> red.
    """
    env = _overlay_env
    _set_gaz_enabled(env, monkeypatch, True)
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert isinstance(overlay.palette, env.module.SeriesPalette), (
        f"overlay.palette is not a SeriesPalette; got {type(overlay.palette).__name__}"
    )


def test_palette_series_chosen_reaches_tray_on_series_chosen(_overlay_env, monkeypatch):
    """A pick in the palette must drive the tray's build+cache+set_series
    flow: build_overlay wires palette.series_chosen -> tray.on_series_chosen.
    Asserted by EMITTING the signal and observing the handler ran (the real
    on_series_chosen is replaced with a recorder so no gazetteer build /
    network is triggered in this unit test).

    Seeded-defect drill: delete the
    `palette.series_chosen.connect(tray.on_series_chosen)` line in
    build_overlay -> the emission reaches no slot -> calls stays empty -> red.
    """
    env = _overlay_env
    _set_gaz_enabled(env, monkeypatch, True)
    calls: list[tuple] = []
    monkeypatch.setattr(
        env.module.Tray, "on_series_chosen",
        lambda self, media_id, label, native="": calls.append((media_id, label, native)),
    )

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    overlay.palette.series_chosen.emit(182205, "Slime (2024)", "転生したらスライムだった件")

    assert calls == [(182205, "Slime (2024)", "転生したらスライムだった件")], (
        "emitting palette.series_chosen did not reach tray.on_series_chosen; "
        "the build_overlay selection wiring is missing"
    )


def test_show_series_palette_signal_opens_the_palette(_overlay_env, monkeypatch):
    """The pipeline's show_series_palette command reaches the overlay as
    client.showSeriesPaletteReceived; build_overlay must wire it to
    palette.summon so the bar becomes visible.

    Asserted by emitting the client signal and observing the bar's
    visibility flip (the observable effect of summon), not private
    connection state.

    Seeded-defect drill: delete the
    `client.showSeriesPaletteReceived.connect(palette.summon)` line ->
    the bar never shows -> red.
    """
    env = _overlay_env
    _set_gaz_enabled(env, monkeypatch, True)
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert not overlay.palette.isVisible()  # precondition

    overlay.client.showSeriesPaletteReceived.emit()
    try:
        assert overlay.palette.isVisible(), (
            "showSeriesPaletteReceived did not open the bar; the "
            "summon wiring is missing"
        )
    finally:
        overlay.palette.hide()


def test_build_overlay_opens_the_series_bar_on_the_subtitle_screen_with_boot_reduce_motion(
    _overlay_env, monkeypatch,
):
    """Phase-5 L3/L12/P3: build_overlay hands the series bar the overlay
    window's subtitle screen (the bar opens centered 24 px below the top of
    that screen's work area, here through the flyout's Change…), the boot
    overlay.reduce_motion value, and pre-warms it once at boot.

    Seeded defects: build the palette without screen_provider -> the bar
    opens on the primary screen's top -> red; drop
    reduce_motion=overlay_cfg.reduce_motion -> False -> red; drop the
    prewarm call -> nothing recorded -> red.
    """
    from PySide6.QtCore import QPoint, QRect

    env = _overlay_env
    monkeypatch.setattr(env.module, "load_full_config", lambda: {
        "gazetteer": {"enabled": True},
        "romanization": {"enabled": True},
        "overlay": {"reduce_motion": True},
    })
    prewarmed: list = []
    real_prewarm = env.module.SeriesPalette.prewarm
    monkeypatch.setattr(env.module.SeriesPalette, "prewarm",
                        lambda self: (prewarmed.append(self), real_prewarm(self)))

    class _SubtitleScreen:
        """The monitor the subtitles are pinned to, as the bar reads it."""

        def availableGeometry(self):
            return QRect(0, 100, 800, 400)

        def geometry(self):
            return QRect(0, 100, 800, 440)

        def devicePixelRatio(self):
            return 1.0

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    try:
        assert overlay.palette._reduce_motion is True, (
            "overlay.reduce_motion: true did not reach the series bar at boot"
        )
        assert prewarmed == [overlay.palette], "the series bar was not pre-warmed once at boot"

        # The overlay's pinned screen, read through OverlayWindow.subtitle_screen.
        monkeypatch.setattr(overlay.window, "_active_screen", lambda: _SubtitleScreen())
        overlay.tray._flyout.changeSeriesRequested.emit()
        assert overlay.palette.isVisible(), "Change… did not summon the series bar"
        assert overlay.palette.pos() == QPoint(80, 124), (
            f"the bar opened at {overlay.palette.pos()}, not the top center of the "
            f"subtitle screen (80, 124)"
        )
    finally:
        overlay.palette.hide()
        overlay.window.close()


def test_build_overlay_disables_series_ui_when_gazetteer_disabled(
    _overlay_env, monkeypatch,
):
    """gazetteer.enabled=false must make the feature genuinely inert: no
    palette is built or retained, the tray shows NO Pick a series... item and
    NO flyout series row, and an authoritative seriesReceived frame from
    the pipeline has no effect on the (absent) tray series UI.

    Seeded-defect drills:
      - drop the `if _gaz_enabled:` gate in build_overlay so the palette is
        always built -> overlay.palette is a SeriesPalette, not None -> red.
      - drop the `on_change_series is not None` gate on Pick a series... ->
        the item reappears -> red.
    """
    env = _overlay_env
    _set_gaz_enabled(env, monkeypatch, False)
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert overlay.palette is None, (
        f"gazetteer disabled but a palette was built; got {overlay.palette!r}"
    )
    texts = [a.text() for a in overlay.tray._menu.actions()]
    assert not any(t.startswith("Pick a series") for t in texts), (
        f"gazetteer disabled but the tray still offers Pick a series...; menu: {texts}"
    )
    assert overlay.tray._flyout._series_row is None, (
        "gazetteer disabled but the flyout has a series row"
    )
    # An authoritative series frame from the pipeline must be a safe no-op:
    # the listener was never connected and there is no series UI to update.
    overlay.client.seriesReceived.emit(182205, "Slime (2024)", "")
    assert overlay.tray._flyout._series_row is None


def test_build_overlay_disables_series_ui_when_romanization_disabled(
    _overlay_env, monkeypatch,
):
    """Flag divergence: gazetteer.enabled=True but romanization.enabled=False.
    A series hint only reaches the model through RomanizationPipeline, so with
    romanization off the whole series feature is a silent no-op -- build_overlay
    must gate on feature_enabled (both flags), leaving NO palette and NO tray
    series UI, exactly as the fully-disabled case.

    Seeded-defect drill: revert build_overlay's `_gaz_enabled =
    feature_enabled(load_full_config())` to the raw
    `load_full_config().get("gazetteer", {}).get("enabled", False)` -> the
    palette is built despite romanization being off -> the palette-None assert
    goes red.
    """
    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_full_config",
        lambda: {
            "gazetteer": {"enabled": True},
            "romanization": {"enabled": False},
        },
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert overlay.palette is None, (
        f"romanization off must make the series feature inert (no hint can "
        f"flow), but a palette was built; got {overlay.palette!r}"
    )
    texts = [a.text() for a in overlay.tray._menu.actions()]
    assert not any(t.startswith("Pick a series") for t in texts), (
        f"romanization off but the tray still offers Pick a series...; menu: {texts}"
    )
    assert overlay.tray._flyout._series_row is None, (
        "romanization off but the flyout has a series row"
    )


def test_build_overlay_threads_gazetteer_auto_genre_into_the_tray(
    _overlay_env, monkeypatch,
):
    """build_overlay must read gazetteer.auto_genre from the merged config and
    thread it into the Tray -- the config key's runtime consumer (hard rule 4).
    Proven end to end on the non-default value: with auto_genre=false a series
    pick sends NO set_genre; with the key absent the shipped default (on)
    applies and the pick rides set_genre anime.

    Seeded-defect drill: drop the ``auto_genre=...`` kwarg from the Tray
    construction in build_overlay -> the tray defaults True, the
    auto_genre=false pick still sends set_genre -> red.
    """
    env = _overlay_env

    def _cfg(auto_genre):
        gaz = {"enabled": True}
        if auto_genre is not None:
            gaz["auto_genre"] = auto_genre
        return {"gazetteer": gaz, "romanization": {"enabled": True}}

    # auto_genre=false threads through: a pick sends no set_genre.
    monkeypatch.setattr(env.module, "load_full_config", lambda: _cfg(False))
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    sent: list[dict] = []
    overlay.tray._client.send_command = lambda cmd: sent.append(cmd) or True
    overlay.tray._on_gazetteer_built(4321, "Yani Neko")
    assert all(c.get("action") != "set_genre" for c in sent), (
        f"gazetteer.auto_genre=false was not threaded into the tray; the "
        f"pick still sent set_genre: {sent}"
    )

    # Key absent -> the shipped default (on) applies: the pick rides set_genre.
    monkeypatch.setattr(env.module, "load_full_config", lambda: _cfg(None))
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    sent2: list[dict] = []
    overlay.tray._client.send_command = lambda cmd: sent2.append(cmd) or True
    overlay.tray._on_gazetteer_built(4321, "Yani Neko")
    assert [c["action"] for c in sent2] == ["set_series", "set_genre"], (
        f"default auto-genre did not ride set_genre along set_series: {sent2}"
    )


# --- legacy geometry migrates to the anchor position on boot ---


def test_build_overlay_migrates_legacy_geometry_to_position(_overlay_env, monkeypatch):
    """When an older install left an overlay.geometry rect that overlaps a
    connected screen, build_overlay maps it to the anchor model, persists BOTH
    the migrated position (which also prunes the legacy rect) AND the
    width-derived max_width_pct, and pins the window with the position.

    Presets the config's .geometry via a monkeypatched load_overlay_config and
    records write_overlay_position + write_overlay_style. Asserts the persisted
    position is the migrate_legacy_geometry mapping of the preset rect, that the
    style write carried the migrated pct (P0-6: the legacy WIDTH must survive),
    and that the live window's current_position matches the mapping.

    Seeded defects (each one line, each red):
    - delete the write_overlay_position(position) call in build_overlay's
      migration branch -> `saved_pos` stays empty -> red.
    - delete the write_overlay_style({"max_width_pct": ...}) call -> `saved_style`
      stays empty -> the style-write assert red (P0-6: width lost on migration).
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.position_config import migrate_legacy_geometry
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from PySide6.QtGui import QGuiApplication

    env = _overlay_env

    legacy_geom = {"x": 100, "y": 200, "width": 800, "height": 300}
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), geometry=legacy_geom,
                              theme="system"),
    )
    saved_pos: list[dict] = []
    saved_style: list[dict] = []
    monkeypatch.setattr(env.module, "write_overlay_position", saved_pos.append)
    monkeypatch.setattr(env.module, "write_overlay_style", saved_style.append)

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    screens = [(s.name(), (s.geometry().x(), s.geometry().y(),
                s.geometry().width(), s.geometry().height()))
               for s in QGuiApplication.screens()]
    expected_pos, expected_pct, _clamped = migrate_legacy_geometry(legacy_geom, screens)
    assert expected_pos is not None, (
        "test precondition: the legacy rect must overlap the CI screen so the "
        "normal migration path runs"
    )
    assert expected_pos in saved_pos, (
        f"legacy geometry was not migrated + persisted; saved={saved_pos}, "
        f"expected={expected_pos}"
    )
    assert saved_style == [{"max_width_pct": expected_pct}], (
        f"migration did not persist the width-derived max_width_pct; "
        f"saved_style={saved_style}, expected pct={expected_pct}"
    )
    assert overlay.window.current_position() == expected_pos, (
        f"window not pinned with the migrated position; "
        f"got {overlay.window.current_position()}, expected {expected_pos}"
    )


def test_build_overlay_applies_migrated_max_width_pct_in_memory(
    _overlay_env, monkeypatch,
):
    """C2: the migrated max_width_pct must be applied to the IN-MEMORY style
    before the window/tray are constructed, not only persisted to disk.
    build_overlay loads overlay_cfg BEFORE the migration runs, so without the
    in-memory application the OverlayWindow renders the whole first session at
    the stale 70 default -- and worse, the tray's style seed carries 70 too,
    so the next Settings Apply emits the full style dict and
    config_writer._apply_to_tree prunes 70-as-default, silently DELETING the
    just-migrated value from disk.

    The rect's width is derived from the live CI screen (42% of it) so the
    migrated pct is deterministic and != the 70 default on any host.

    Seeded defect (one line): drop the
    ``overlay_cfg.style["max_width_pct"] = max_width_pct`` line from
    build_overlay's migration branch -> the window/tray read 70 -> red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.position_config import migrate_legacy_geometry
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from PySide6.QtGui import QGuiApplication

    env = _overlay_env

    s = QGuiApplication.primaryScreen().geometry()
    legacy_geom = {
        "x": s.x() + s.width() // 4, "y": s.y() + s.height() // 2,
        "width": int(s.width() * 0.42), "height": max(1, s.height() // 10),
    }
    screens = [(sc.name(), (sc.geometry().x(), sc.geometry().y(),
                sc.geometry().width(), sc.geometry().height()))
               for sc in QGuiApplication.screens()]
    _pos, expected_pct, _clamped = migrate_legacy_geometry(legacy_geom, screens)
    assert expected_pct is not None and expected_pct != STYLE_DEFAULTS["max_width_pct"], (
        f"test precondition: 42% of the CI screen must map to a pct that is "
        f"neither None nor the 70 default; got {expected_pct}"
    )

    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), geometry=legacy_geom,
                              theme="system"),
    )
    monkeypatch.setattr(env.module, "write_overlay_position", lambda pos: None)
    monkeypatch.setattr(env.module, "write_overlay_style", lambda style: None)

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    assert overlay.window._style["max_width_pct"] == expected_pct, (
        f"OverlayWindow was constructed with the stale default max_width_pct "
        f"({overlay.window._style['max_width_pct']}) instead of the migrated "
        f"{expected_pct}; the first session ignores the migrated width"
    )
    assert overlay.tray._current_style["max_width_pct"] == expected_pct, (
        f"the tray's style seed missed the migrated max_width_pct; the next "
        f"full-style write would prune the migrated value from disk "
        f"(got {overlay.tray._current_style['max_width_pct']})"
    )


def test_build_overlay_migration_logs_original_rect_when_clamped(
    _overlay_env, monkeypatch, caplog,
):
    """C9: when the migration materially moves a legal legacy placement
    (anchor bounds / width floor), build_overlay must log ONE warning quoting
    the ORIGINAL rect dict and the migrated position/pct. The position write
    prunes overlay.geometry in the same locked write, so this log line is the
    only recoverable trace of the pre-migration placement (this app ships a
    collect-diagnostics zip that carries the logs).

    The rect hugs the live CI screen's left edge at 5% width, so both the
    h-offset floor (center 2.5% -> 10) and the width floor (5% -> 30) trip
    regardless of the host's resolution.

    Seeded defects (each one line, each red):
    - drop the ``if clamped:`` warning from build_overlay's migration branch
      -> no record quotes the rect -> red.
    - log the MIGRATED position only (no original rect) -> the rect-repr
      containment assert red -- the original placement is unrecoverable.
    """
    import logging as _logging

    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from PySide6.QtGui import QGuiApplication

    env = _overlay_env

    s = QGuiApplication.primaryScreen().geometry()
    legacy_geom = {
        "x": s.x(), "y": s.y() + s.height() // 2,
        "width": max(1, s.width() // 20), "height": max(1, s.height() // 10),
    }
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), geometry=legacy_geom,
                              theme="system"),
    )
    monkeypatch.setattr(env.module, "write_overlay_position", lambda pos: None)
    monkeypatch.setattr(env.module, "write_overlay_style", lambda style: None)

    with caplog.at_level(_logging.WARNING):
        overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    traces = [
        r for r in caplog.records
        if r.levelno == _logging.WARNING and repr(legacy_geom) in r.getMessage()
    ]
    assert len(traces) == 1, (
        f"expected exactly 1 clamp-trace warning quoting the original rect "
        f"{legacy_geom!r}; got {len(traces)}. Warning records: "
        f"{[r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]!r}"
    )
    # The migrated result must be in the SAME record, so one grep of the
    # diagnostics zip recovers the before AND the after.
    assert repr(overlay.window.current_position()) in traces[0].getMessage(), (
        f"the clamp-trace warning does not carry the migrated position; "
        f"got: {traces[0].getMessage()!r}"
    )


def test_build_overlay_migration_persist_failure_does_not_crash_boot(
    _overlay_env, monkeypatch,
):
    """A persist failure during the geometry->position migration must never
    crash overlay boot: it is logged and the migrated in-memory position AND
    max_width_pct are used anyway (C2: the in-memory application must not
    depend on persist success -- it happens BEFORE the try block, mirroring
    the position handling).

    Seeded defects (each red):
    - remove the try/except around write_overlay_position in the migration
      branch -> the OSError propagates out of build_overlay -> red.
    - move the ``overlay_cfg.style["max_width_pct"] = ...`` line inside the
      try AFTER the raising write -> the window renders the 70 default on
      exactly the boots where the disk write failed -> the style assert red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.position_config import migrate_legacy_geometry
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from PySide6.QtGui import QGuiApplication

    env = _overlay_env

    legacy_geom = {"x": 10, "y": 20, "width": 800, "height": 300}
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS),
                              geometry=dict(legacy_geom),
                              theme="system"),
    )

    def _boom(_pos: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(env.module, "write_overlay_position", _boom)

    # Must not raise:
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert overlay.window is not None

    screens = [(sc.name(), (sc.geometry().x(), sc.geometry().y(),
                sc.geometry().width(), sc.geometry().height()))
               for sc in QGuiApplication.screens()]
    _pos, expected_pct, _clamped = migrate_legacy_geometry(legacy_geom, screens)
    assert expected_pct is not None, (
        "test precondition: the legacy rect must overlap the CI screen"
    )
    assert overlay.window._style["max_width_pct"] == expected_pct, (
        f"the migrated max_width_pct was not applied in-memory after a "
        f"persist failure; got {overlay.window._style['max_width_pct']}, "
        f"expected {expected_pct}"
    )


def test_build_overlay_no_overlap_geometry_is_kept_not_pruned(
    _overlay_env, monkeypatch,
):
    """P0-5: a legacy rect that overlaps NO connected screen (undocked laptop
    whose placement was on an external monitor) must NOT be migrated-and-pruned.
    build_overlay must SKIP both persist writes -- so write_overlay_position is
    never called (it is what prunes overlay.geometry, which would destroy the
    placement unrecoverably) and write_overlay_style is never called -- and pin
    the window with clamped DEFAULTS for this session only. The rect stays on
    disk for a later boot when the monitor returns.

    Seeded defects (each one line, each red):
    - revert geometry_to_position's no-overlap branch to ``return dict(_DEFAULT)``
      (so migrate returns a non-None position) -> build_overlay takes the persist
      branch -> write_overlay_position records a call -> the not-called assert red.
    - drop the ``if migrated is None:`` skip branch in build_overlay -> same:
      the persist writes fire on a no-overlap boot -> red.
    - make the quit-time save write the unchanged position unconditionally ->
      the aboutToQuit emit below records a call -> red (the first quit of the
      session would prune the rect the boot deliberately kept).
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.position_config import clamp_position
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env

    # Far off any real/offscreen CI monitor -> zero overlap on every screen.
    off_geom = {"x": 999999, "y": 999999, "width": 100, "height": 100}
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), geometry=off_geom,
                              theme="system"),
    )
    pos_writes: list[dict] = []
    style_writes: list[dict] = []
    monkeypatch.setattr(env.module, "write_overlay_position", pos_writes.append)
    monkeypatch.setattr(env.module, "write_overlay_style", style_writes.append)

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    # Quit without touching the position: nothing new to save, so the kept
    # rect must survive the quit path too, not just boot.
    overlay.app.aboutToQuit.emit()

    assert pos_writes == [], (
        f"no-overlap boot persisted overlay.position (which prunes "
        f"overlay.geometry) -- the placement is now unrecoverable: {pos_writes}"
    )
    assert style_writes == [], (
        f"no-overlap boot persisted a max_width_pct on a placement it could not "
        f"map: {style_writes}"
    )
    # Session position is the clamped shipped defaults (overlay_cfg.position is
    # the empty default {}), NOT an invented mapping of the offscreen rect.
    assert overlay.window.current_position() == clamp_position({}), (
        f"no-overlap boot did not fall back to default position; got "
        f"{overlay.window.current_position()}"
    )


# --- position must be saved on the production quit paths (C8) ---


def test_about_to_quit_persists_current_position(_overlay_env, monkeypatch):
    """Every shipped quit path (tray Quit, shutdown event) calls app.quit(),
    which exits exec() WITHOUT delivering closeEvent -- the same F96 trap the
    retired geometry model was hardened against. PlacementController.save()
    only fires on Enter / click-away / WindowDeactivate, so quitting DURING an
    active placement drag silently discarded the position. aboutToQuit fires
    on every quit path while the window is still alive; build_overlay must
    wire the position save to it.

    The test invokes the wired handler end-to-end by emitting aboutToQuit on
    the real QApplication, and asserts the observable effect: the current
    position reached write_overlay_position.

    Seeded defect: drop the app.aboutToQuit.connect(_save_position_on_quit)
    line from build_overlay -> nothing records the unique position -> red.

    (The qapp singleton is shared across this module's tests, so earlier
    build_overlay calls may have their own aboutToQuit handlers recording
    THEIR window's position; assert our unique position is among the recorded
    saves rather than the only one.)
    """
    env = _overlay_env
    saved: list[dict] = []
    monkeypatch.setattr(env.module, "write_overlay_position", saved.append)

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    # Unique offsets so a stale handler from a sibling test can't satisfy the
    # assertion by accident. Set the private field directly: set_position()
    # would ALSO persist via on_position_changed and muddy the recorder.
    unique = {"screen": "primary", "v_offset_pct": 43.7, "h_offset_pct": 21.3}
    overlay.window._position = dict(unique)

    overlay.app.aboutToQuit.emit()

    assert unique in saved, (
        f"aboutToQuit did not persist the current position; saved={saved}"
    )


def test_about_to_quit_position_save_failure_does_not_raise(_overlay_env, monkeypatch):
    """A config-write failure during quit must not detonate the quit path
    (the handler runs inside Qt's signal dispatch on the way out).

    Seeded defect: remove the try/except around the aboutToQuit save
    -> the OSError propagates out of emit() -> red.
    """
    env = _overlay_env

    def _boom(_pos: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(env.module, "write_overlay_position", _boom)
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    # Move the window the way a placement drag does (preview, not persisted),
    # so the quit handler has an unsaved position and actually attempts the
    # failing write.
    overlay.window.update_position_preview(
        {"screen": "primary", "v_offset_pct": 47.3, "h_offset_pct": 33.1}
    )
    # Must not raise:
    overlay.app.aboutToQuit.emit()


def test_explicit_config_preset_and_quit_leave_user_overrides_untouched(
    _overlay_env, tmp_path,
):
    """An explicit --config run is isolated from ambient user state: neither a
    Position preset nor the quit-time save may write the real
    user_overrides.yaml.

    Drives the shipped paths end-to-end: build_overlay with --config, a tray
    preset click, then aboutToQuit on the real QApplication. The observable
    effect is the overrides file's bytes on disk.

    Seeded defect: pass write_overlay_position as on_position_changed even
    when --config is set -> the preset (and the quit save) rewrite the
    user's position -> the byte comparison goes red.
    """
    from yamabiko import paths

    env = _overlay_env
    overrides = paths.user_overrides_path()
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text(
        "overlay:\n  position:\n    v_offset_pct: 61.5\n", encoding="utf-8",
    )
    before = overrides.read_bytes()
    repro = tmp_path / "repro.yaml"
    repro.write_text("overlay: {}\n", encoding="utf-8")

    overlay = env.module.build_overlay(
        ["--url", "ws://127.0.0.1:1", "--config", str(repro)],
    )
    overlay.tray._on_position_preset("primary", 40.0)
    overlay.app.aboutToQuit.emit()

    assert overrides.read_bytes() == before, (
        "an explicit --config run wrote the real user_overrides.yaml:\n"
        f"{overrides.read_text(encoding='utf-8')}"
    )


# --- chrome theme resolved at boot + live OS follow (Task 11) ---


def test_build_overlay_forced_light_theme_yields_mist_palette(_overlay_env, monkeypatch):
    """build_overlay resolves overlay.theme at boot and stores the palette on
    OverlayApp: a forced 'light' theme yields the mist (light) palette.

    Seeded defect: drop the `palette = palette_for(overlay_cfg.theme)` compute
    (or hardcode a palette) -> the stored palette name is wrong -> red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="light"),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert overlay.theme_palette["name"] == "mist", (
        f"forced light theme did not resolve to the mist palette; "
        f"got {overlay.theme_palette.get('name')!r}"
    )


def test_build_overlay_forced_dark_theme_yields_night_palette(_overlay_env, monkeypatch):
    """A forced 'dark' theme yields the night (dark) palette.

    Seeded defect: swap the NIGHT/MIST arms of palette_for -> red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="dark"),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert overlay.theme_palette["name"] == "night", (
        f"forced dark theme did not resolve to the night palette; "
        f"got {overlay.theme_palette.get('name')!r}"
    )


def test_build_overlay_system_theme_follows_probe(_overlay_env, monkeypatch):
    """A 'system' theme delegates to system_scheme_probe: a light probe ->
    mist, a dark probe -> night. The probe is monkeypatched so the assertion
    is deterministic (independent of the CI host's real OS scheme).

    Seeded defect: make palette_for ignore the probe (hardcode a mode) -> one
    of the two branches goes red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app import theme
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="system"),
    )

    monkeypatch.setattr(theme, "system_scheme_probe", lambda: "light")
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert overlay.theme_palette["name"] == "mist", (
        f"system+light-probe did not resolve to mist; got {overlay.theme_palette.get('name')!r}"
    )

    monkeypatch.setattr(theme, "system_scheme_probe", lambda: "dark")
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    assert overlay.theme_palette["name"] == "night", (
        f"system+dark-probe did not resolve to night; got {overlay.theme_palette.get('name')!r}"
    )


def test_color_scheme_handler_reskins_when_persisted_theme_is_system(
    _overlay_env, monkeypatch,
):
    """The colorSchemeChanged handler follows the OS only while the PERSISTED
    theme is 'system': with 'system' persisted it recomputes the palette and
    re-skins the tray via apply_palette. The handler re-reads the COERCED theme
    (load_overlay_config().theme) each call (P1-13).

    Seeded defect: invert the theme guard in the handler (return when
    'system') -> apply_palette is never called -> red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="system"),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    # Persisted theme is 'system' at flip time -> the handler's gate passes. The
    # handler reads load_overlay_config().theme, so patch THAT (the coerced
    # reader), not the raw load_full_config dict.
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(theme="system"),
    )
    calls: list = []
    overlay.tray.apply_palette = lambda p: calls.append(p)
    overlay.color_scheme_handler()
    assert len(calls) == 1, (
        "colorScheme handler did not re-skin the tray while the persisted "
        f"theme is 'system'; apply_palette calls={len(calls)}"
    )


def test_color_scheme_handler_skips_when_persisted_theme_is_explicit(
    _overlay_env, monkeypatch,
):
    """Handler discipline / stale-cache drill: once the user picks an explicit
    light/dark theme, an OS flip must NOT override it. The handler re-reads the
    PERSISTED theme each time (never a boot-cached value); with 'dark'
    persisted it returns without calling apply_palette.

    Seeded-defect drill (a): cache the boot theme ('system') in the handler
    instead of re-reading load_overlay_config -> the stale 'system' passes the
    guard and apply_palette fires -> red.
    """
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    # Booted in system mode (so the handler is even wired), but the user has
    # since switched to an explicit dark theme (now the persisted value).
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="system"),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    # The handler re-reads load_overlay_config().theme each call; the user's
    # explicit 'dark' pick is the current persisted value.
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(theme="dark"),
    )
    calls: list = []
    overlay.tray.apply_palette = lambda p: calls.append(p)
    overlay.color_scheme_handler()
    assert calls == [], (
        "colorScheme handler re-skinned despite an explicit persisted theme; "
        "it must re-read the persisted value, not a boot-cached one"
    )


def test_color_scheme_handler_reads_coerced_theme_not_raw_dict(
    _overlay_env, monkeypatch,
):
    """P1-13: the handler must gate on the COERCED theme
    (load_overlay_config().theme), NOT the raw config dict value. A persisted
    theme of the int 1 (or 'Dark ', '') coerces to 'system' -- boot decides
    OS-following from that same coerced value -- so the handler must follow too.
    A raw-dict read sees 1, finds 1 != 'system', and early-returns, leaving the
    OS-follow silently DEAD on exactly the configs boot put into system mode.

    Both readers are pointed at the same raw {'overlay': {'theme': 1}}, so the
    ONLY thing separating fix from defect is WHICH reader the handler consults:
      * fix (load_overlay_config): 1 -> coerced 'system' -> follows -> calls==1.
      * defect (raw load_full_config dict): 1 != 'system' -> returns -> calls==0.

    Seeded defect: revert _on_color_scheme_changed to
    ``load_full_config().get('overlay',{}).get('theme','system')`` -> it reads
    the raw int 1 -> early-returns -> apply_palette never fires -> red.
    """
    from yamabiko import config as _config
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="system"),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    # Raw persisted theme is the int 1. Hand the handler the REAL coercing
    # loader (config.load_overlay_config reads config.load_full_config
    # internally, which we patch to the raw dict) so the fix path exercises the
    # true 1 -> 'system' coercion. Point the raw-dict loader the DEFECT would
    # read at the same 1, so the two paths differ ONLY in the coercion.
    monkeypatch.setattr(_config, "load_full_config", lambda: {"overlay": {"theme": 1}})
    monkeypatch.setattr(env.module, "load_overlay_config", _config.load_overlay_config)
    monkeypatch.setattr(env.module, "load_full_config", lambda: {"overlay": {"theme": 1}})

    calls: list = []
    overlay.tray.apply_palette = lambda p: calls.append(p)
    overlay.color_scheme_handler()
    assert len(calls) == 1, (
        "colorScheme handler early-returned on a theme that COERCES to 'system' "
        "(raw int 1); it must read load_overlay_config().theme, not the raw "
        f"dict value; apply_palette calls={len(calls)}"
    )


def test_color_scheme_signal_emission_reaches_apply_palette(_overlay_env, monkeypatch):
    """Wiring proof via the REAL signal: emitting styleHints().colorSchemeChanged
    -- as Qt does on an OS light/dark flip -- must reach Tray.apply_palette.
    The two handler tests above call overlay.color_scheme_handler() directly,
    so they stay green even if build_overlay never actually connected the
    handler (or connected the wrong callable). This one emits through the live
    signal, so a missing / wrong-callable connect goes red.

    Emitting the C++ colorSchemeChanged signal from Python is supported under
    the offscreen platform in PySide6 6.11 (verified 2026-07-04), so the wiring
    is exercised end to end here -- no receivers()/connection-count fallback.

    The spy is patched on the Tray CLASS (not this overlay's instance): that
    both records which Tray Qt drives AND neutralises the real menu re-skin for
    any handler an earlier build_overlay left connected on the shared
    QApplication singleton, so the fan-out does no Qt work during the emit.

    Seeded defect: drop the
    `style_hints.colorSchemeChanged.connect(_on_color_scheme_changed)` line in
    build_overlay -> the emission reaches no slot for this overlay,
    apply_palette never fires on its tray -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="system"),
    )
    # Persisted theme is 'system' at flip time -> the handler's gate passes.
    monkeypatch.setattr(
        env.module, "load_full_config",
        lambda: {"overlay": {"theme": "system"}},
    )
    seen: list = []
    monkeypatch.setattr(
        env.module.Tray, "apply_palette", lambda self, p: seen.append(self),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    style_hints = overlay.app.styleHints()
    assert hasattr(style_hints, "colorSchemeChanged"), (
        "styleHints has no colorSchemeChanged; test host predates Qt 6.5"
    )
    style_hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)

    hits = sum(1 for t in seen if t is overlay.tray)
    assert hits == 1, (
        "emitting the real colorSchemeChanged signal did not reach this "
        "overlay's Tray.apply_palette; build_overlay's colorSchemeChanged."
        f"connect is missing or bound to the wrong callable (hits={hits})"
    )


def test_color_scheme_follow_connected_even_when_booted_forced_light(
    _overlay_env, monkeypatch,
):
    """Mid-session switch to System must follow the OS without a relaunch: the
    colorSchemeChanged connect is NOT gated on the boot theme. The handler is
    the sole gate (it re-reads the persisted theme per call), so build_overlay
    connects it regardless of the boot theme.

    Boot forced-light, then emit the REAL colorSchemeChanged signal with the
    persisted theme now 'system' (the user switched to System after boot):
    apply_palette must fire -- proving the handler was connected at a light
    boot, not skipped.

    Seeded drill (a): restore the `overlay_cfg.theme == "system"` clause on the
    connect -> a light boot skips the connect -> the emission reaches no slot
    for this overlay -> apply_palette never fires on its tray -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.config import OverlayConfig
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    env = _overlay_env
    # Boot forced-light: the old boot-gate would skip the connect here.
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(style=dict(STYLE_DEFAULTS), theme="light"),
    )
    seen: list = []
    monkeypatch.setattr(
        env.module.Tray, "apply_palette", lambda self, p: seen.append(self),
    )
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])

    # The user has since switched to System (persisted 'system'); the handler
    # re-reads load_overlay_config().theme per call, so re-point it now. An OS
    # flip must now be followed even though boot was forced-light.
    monkeypatch.setattr(
        env.module, "load_overlay_config",
        lambda *_args, **_kwargs: OverlayConfig(theme="system"),
    )
    overlay.app.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)

    hits = sum(1 for t in seen if t is overlay.tray)
    assert hits == 1, (
        "colorSchemeChanged was not followed after a forced-light boot + "
        "mid-session switch to System; the connect is still boot-gated "
        f"(hits={hits})"
    )


# --- a malformed colour must not crash the render path ---


def test_look_from_style_degrades_on_bad_colors():
    """look_from_style runs inside the directly-connected style and subtitle
    slots; a bad bg_color must degrade to the default dark bar instead of
    raising (a raise there silently drops EVERY subtitle).

    Seeded-defect drill: parse bg_color without the parse_hex_rgb fallback
    (bare ``value.lstrip("#")`` + int()) -> each of these raises -> this
    test errors.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.telop_look import look_from_style

    for bad in ("#fff", "red", "", 16777215, None):
        look = look_from_style({**STYLE_DEFAULTS, "bg_color": bad})
        # Falls back to the default dark bar #0e1014 (14,16,20); never raises.
        assert look.bar[:3] == (14, 16, 20), (
            f"bad bg_color {bad!r} did not fall back: {look.bar}"
        )


# --- the English and the echo fill must honour the configured text_color ---


def test_text_color_drives_english_and_echo_fill():
    """text_color is the English on the dark bar and the echo fill: a red
    text_color must reach both, or the user's colour is silently ignored.

    Seeded-defect drill: hardcode the echo fill to white in look_from_style
    -> jp_echo stays white -> this fails.
    """
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.telop_look import look_from_style

    look = look_from_style({**STYLE_DEFAULTS, "text_color": "#ff0000"})
    assert look.jp_echo[:3] == (255, 0, 0), f"echo fill ignored text_color: {look.jp_echo}"
    assert look.en_plate[:3] == (255, 0, 0), (
        f"English on the bar ignored text_color: {look.en_plate}"
    )


# --- update-check "disabled" line must reach the yamabiko file handler ---


def test_update_check_disabled_line_logs_on_yamabiko_tree_logger(monkeypatch, caplog):
    """The "update check disabled by config" INFO line must be logged on
    a hardcoded yamabiko-tree logger, never a __name__-derived one: when
    the overlay is dev-spawned via ``python -m yamabiko.overlay_app``,
    __name__ is "__main__", so records on the module logger never reach
    the yamabiko file handler and "why did I never hear about vNext?"
    stays unanswerable from the log.

    The name asserts are exact on purpose. Under pytest import the
    module logger is "yamabiko.overlay_app.__main__" -- a bare
    startswith("yamabiko") check passes over the broken __name__-derived
    logger (a Liar); the not-endswith("__main__") clause is what makes
    the __name__-derived logger red in BOTH launch contexts, because any
    logger derived from this file's __name__ ends with "__main__".

    Seeded-defect drill: revert the line to this module's ``log.info``
    -> record name ends with "__main__" -> red.
    """
    import logging as _logging

    import yamabiko.overlay_app.__main__ as overlay_main

    monkeypatch.setattr(
        overlay_main, "load_full_config",
        lambda: {"updates": {"check_on_startup": False}},
    )
    with caplog.at_level(_logging.INFO):
        # The disabled path must not touch the tray icon at all; a bare
        # object() sentinel raises AttributeError if it does.
        result = overlay_main._schedule_update_check(object())
    assert result is None

    disabled = [
        r for r in caplog.records
        if "update check disabled by config" in r.getMessage()
    ]
    assert len(disabled) == 1, (
        f"expected exactly 1 'disabled by config' record; got "
        f"{len(disabled)}. Records: "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
    rec = disabled[0]
    assert rec.name.startswith("yamabiko") and not rec.name.endswith("__main__"), (
        f"'disabled by config' logged on {rec.name!r}; a __name__-derived "
        f"logger bypasses the yamabiko file handler when the overlay is "
        f"spawned via `python -m yamabiko.overlay_app`"
    )
    assert rec.levelno == _logging.INFO, rec.levelno


# --- quit-time worker detach: aboutToQuit wiring + run_overlay hard-exit ---
#
# Quit while an AniList search / cover fetch (palette) or a gazetteer build
# (tray) is in flight destroys a parented running QThread -> Qt6 qFatal.
# build_overlay must detach those workers on aboutToQuit, and run_overlay
# must hard-exit (os._exit) if any detached worker is still draining so
# interpreter finalization can't destroy its still-running QThread wrapper.


def test_about_to_quit_detaches_running_palette_worker(_overlay_env, qtbot, monkeypatch):
    """build_overlay wires palette.release_threads to app.aboutToQuit, so a
    quit while a cover fetch is in flight DETACHES the running QThread
    instead of letting widget teardown destroy it (Qt6 qFatal). Invocation-
    style (§3.9): emits aboutToQuit and asserts the observable detach, not
    that a callable is registered.

    Seeded-defect drill: drop the
    `app.aboutToQuit.connect(palette.release_threads)` line in build_overlay
    -> the worker stays parented / in _cover_threads -> red.
    """
    import threading

    from yamabiko.gazetteer.anilist import SeriesHit
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.ui import thread_keeper

    env = _overlay_env
    _set_gaz_enabled(env, monkeypatch, True)
    gate = threading.Event()
    monkeypatch.setattr(
        sp, "_fetch_cover_bytes", lambda url, **kw: (gate.wait(10), None)[1],
    )

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    palette = overlay.palette
    hit = SeriesHit(1, "Romaji", "English", None, 2011, "TV",
                    cover_image="http://x/c.png", color=None)
    palette._start_cover_fetch(hit)
    (thread,) = palette._cover_threads
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    overlay.app.aboutToQuit.emit()

    try:
        assert thread.parent() is None, (
            "aboutToQuit did not detach the running cover worker; the "
            "palette.release_threads wiring is missing"
        )
        assert thread not in palette._cover_threads
        assert thread_keeper.any_running()
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)


def test_about_to_quit_detaches_running_tray_gazetteer_worker(
    _overlay_env, qtbot, tmp_path, monkeypatch,
):
    """build_overlay wires tray.release_threads to app.aboutToQuit, so a quit
    while the tray's gazetteer build worker is fetching from AniList DETACHES
    the running QThread instead of destroying it at teardown (Qt6 qFatal).

    Seeded-defect drill: drop the
    `app.aboutToQuit.connect(tray.release_threads)` line in build_overlay ->
    the gaz worker stays parented and _gaz_thread stays set -> red.
    """
    import threading

    from yamabiko.overlay_app import tray as tray_mod
    from yamabiko.ui import thread_keeper

    env = _overlay_env
    _set_gaz_enabled(env, monkeypatch, True)
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))  # empty -> cache miss
    gate = threading.Event()

    class _BlockingClient:
        # The full pick build's FIRST wire call is the intel request;
        # blocking there keeps the worker mid-fetch when aboutToQuit fires.
        def fetch_media_intel(self, media_id):
            gate.wait(10)
            return {"genres": [], "tags": [], "description": "",
                    "relations": []}

        def fetch_characters(self, media_id, max_pages: int = 6, on_page=None):
            return []

        def close(self):
            pass

    monkeypatch.setattr(tray_mod, "AniListClient", lambda *a, **k: _BlockingClient())

    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    tray = overlay.tray
    # After gate.set the worker emits built -> _on_gazetteer_built sends
    # set_series over the (unstarted) real client; keep that queued slot a
    # clean no-op so it can't perturb the event loop the finally spins.
    monkeypatch.setattr(overlay.client, "send_command", lambda cmd: True)
    tray.on_series_chosen(182205, "Slime (2024)")
    thread = tray._gaz_thread
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    overlay.app.aboutToQuit.emit()

    try:
        assert thread.parent() is None, (
            "aboutToQuit did not detach the running gazetteer worker; the "
            "tray.release_threads wiring is missing"
        )
        assert tray._gaz_thread is None
        assert thread_keeper.any_running()
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)


def test_run_overlay_hard_exits_when_detached_worker_still_running(
    _overlay_env, monkeypatch,
):
    """If a detached worker is still running after the event loop returns,
    run_overlay must os._exit(rc) rather than return -- letting the
    interpreter finalize would destroy the running QThread wrapper (Qt6
    qFatal). Mirrors main.py's wizard-cancel hard-exit contract.

    Seeded-defect drill: drop the thread_keeper.any_running() guard before
    run_overlay's final return -> os._exit is never called -> red.
    """
    env = _overlay_env
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    monkeypatch.setattr(env.module.PipelineClient, "start", lambda self: None)
    monkeypatch.setattr(env.module.PipelineClient, "stop", lambda self: None)
    monkeypatch.setattr(QApplication, "exec", lambda self: 7)
    monkeypatch.setattr(
        "yamabiko.lifecycle.shutdown_event.ShutdownEventListener.from_env",
        classmethod(lambda cls, role="overlay": None),
    )
    monkeypatch.setattr(env.module.thread_keeper, "any_running", lambda: True)
    monkeypatch.setattr(env.module.logging, "shutdown", lambda: None)
    exits: list[int] = []
    monkeypatch.setattr(env.module.os, "_exit", lambda code: exits.append(code))

    env.module.run_overlay(overlay)

    assert exits == [7], (
        f"run_overlay did not hard-exit with the loop's rc; os._exit calls={exits}"
    )


def test_run_overlay_returns_rc_when_no_detached_worker(_overlay_env, monkeypatch):
    """When no worker is left running after the event loop, run_overlay
    returns the loop's rc normally and does NOT hard-exit -- the guard is
    conditional, not an unconditional os._exit on every overlay shutdown
    (which would kill the whole test process, and skip normal teardown).

    Seeded-defect drill: make the guard unconditional (always os._exit(rc))
    -> os._exit fires here -> the exits-empty assert goes red.
    """
    env = _overlay_env
    overlay = env.module.build_overlay(["--url", "ws://127.0.0.1:1"])
    monkeypatch.setattr(env.module.PipelineClient, "start", lambda self: None)
    monkeypatch.setattr(env.module.PipelineClient, "stop", lambda self: None)
    monkeypatch.setattr(QApplication, "exec", lambda self: 3)
    monkeypatch.setattr(
        "yamabiko.lifecycle.shutdown_event.ShutdownEventListener.from_env",
        classmethod(lambda cls, role="overlay": None),
    )
    monkeypatch.setattr(env.module.thread_keeper, "any_running", lambda: False)
    exits: list[int] = []
    monkeypatch.setattr(env.module.os, "_exit", lambda code: exits.append(code))

    rc = env.module.run_overlay(overlay)

    assert rc == 3, f"run_overlay did not return the loop's rc; got {rc}"
    assert exits == [], (
        "run_overlay hard-exited despite no detached worker still running"
    )
