"""Regression tests for yamabiko.overlay_app.__main__.main().

Bug surfaced on user install after commit b24db82 (F_J5 screen-change
handlers): the three ``QGuiApplication`` signals (``primaryScreenChanged``,
``screenAdded``, ``screenRemoved``) were being connected via the CLASS
rather than the app INSTANCE, which in PySide6 returns the Signal
descriptor — not a bound signal — and ``.connect`` raises
``AttributeError: 'PySide6.QtCore.Signal' object has no attribute 'connect'``
the moment main() reaches line ~127.

The existing ``test_frozen_exe_default_launch_spawns_pipeline_and_overlay``
did not catch this because the watchdog supervisor respawns the crashing
overlay child, and the E2E's single snapshot at t=3s happened to catch
a just-respawned overlay process — satisfying the ``>=2 children``
assertion without the overlay ever surviving.

This test closes the gap: invoke main() directly with the event loop
mocked to exit immediately, and assert it returns without raising.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication


def _write_explicit_overlay_config(
    tmp_path, *, theme, jp_size, screen, v_offset_pct,
):
    path = tmp_path / "explicit-overlay.yaml"
    path.write_text(
        "overlay:\n"
        f"  theme: {theme}\n"
        "  style:\n"
        f"    jp_size: {jp_size}\n"
        "  position:\n"
        f"    screen: {screen}\n"
        f"    v_offset_pct: {v_offset_pct}\n",
        encoding="utf-8",
    )
    return path


def _run_overlay_main(monkeypatch, tmp_path, *, log_level="CRITICAL"):
    """Shared harness: invoke overlay main() end-to-end with the
    blocking event loop + WS client + update check mocked out.

    Returns the rc. Caller asserts on rc and/or captured log output.

    Qt singleton handling: QApplication refuses to construct a
    second instance in the same process. Multiple tests in this
    file exercise main(), which calls ``QApplication(sys.argv)``
    unconditionally. We monkey-patch ``QApplication`` *in the
    module under test* to an idempotent factory: reuse the existing
    instance if one exists, otherwise construct.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["yamabiko-overlay",
                                      "--log-level", log_level])

    import yamabiko.overlay_app.__main__ as overlay_main
    from yamabiko.overlay_app import ws_client

    # Proxy stand-in that main() uses in place of QApplication.
    # Handles both call patterns main() performs on that name:
    #   QApplication.setHighDpiScaleFactorRoundingPolicy(...)  (classmethod)
    #   app = QApplication(sys.argv)                           (constructor)
    # Returns the pre-existing singleton on construction so the Qt
    # "only one QApplication" invariant isn't violated across tests.
    app_singleton = QApplication.instance() or QApplication([])

    class _AppProxy:
        def __new__(cls, argv):
            return app_singleton

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy):
            # No-op after QApplication exists; Qt warns on late calls
            # but they're benign in our test harness.
            pass
    monkeypatch.setattr(overlay_main, "QApplication", _AppProxy)

    # setup_main_log() sets a persistent level on the ``yamabiko``
    # logger and attaches a RotatingFileHandler. Both leak across tests:
    # the level change drops later caplog WARNING records (discovered
    # via test pollution on test_load_malformed_yaml_returns_empty and
    # test_quant_tier_unknown_value_falls_back_to_default); the file
    # handler accumulates handlers across repeated test runs. Skip it
    # entirely in the harness -- this test exercises main() wiring,
    # not production logging.
    monkeypatch.setattr(overlay_main, "setup_main_log", lambda *a, **kw: None)

    # Crash-evidence arming + Qt-diagnostics routing are process-global
    # (sys/threading excepthook chains, a process-lifetime faulthandler
    # handle, qInstallMessageHandler + qt_bootstrap's installed flag).
    # Stub both on the module-under-test's bindings so main() runs here
    # never swap the pytest process's real hooks (same hygiene as
    # tests/test_build_watchdog.py's _watchdog_env). Call-site wiring is
    # pinned by the dedicated tests in test_build_overlay.py.
    monkeypatch.setattr(overlay_main, "install_crash_evidence",
                        lambda role: None)
    monkeypatch.setattr(overlay_main, "install_qt_message_handler",
                        lambda: None)

    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    monkeypatch.setattr(ws_client.PipelineClient, "start", lambda self: None)
    monkeypatch.setattr(ws_client.PipelineClient, "stop", lambda self: None)
    monkeypatch.setattr(overlay_main, "_schedule_update_check",
                        lambda *a, **kw: None)

    return overlay_main.main()


def test_overlay_main_applies_explicit_config_and_keeps_settings_read_only(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = _write_explicit_overlay_config(
        tmp_path,
        theme="dark",
        jp_size=31,
        screen="primary",
        v_offset_pct=73.0,
    )

    import yamabiko.overlay_app.__main__ as overlay_main

    app_singleton = QApplication.instance() or QApplication([])

    class _AppProxy:
        def __new__(cls, argv):
            return app_singleton

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy):
            pass

    monkeypatch.setattr(overlay_main, "QApplication", _AppProxy)
    monkeypatch.setattr(overlay_main, "setup_main_log", lambda *a, **kw: None)
    monkeypatch.setattr(
        overlay_main, "install_crash_evidence", lambda role: None,
    )
    monkeypatch.setattr(
        overlay_main, "install_qt_message_handler", lambda: None,
    )
    monkeypatch.setattr(
        overlay_main, "_schedule_update_check", lambda *a, **kw: None,
    )

    built = overlay_main.build_overlay(["--config", str(path)])
    try:
        assert built.window._style["jp_size"] == 31
        assert built.tray._current_style["jp_size"] == 31
        assert built.theme_palette["name"] == "night"
        assert built.window.current_position()["v_offset_pct"] == 73.0
        assert built.tray._config_path == path
        built.tray._open_settings()
        settings = built.tray._settings_window
        assert settings is not None
        assert settings._theme_value == "dark"
        assert settings._read_only is True
        assert settings._apply_btn.isEnabled() is False
        assert "--config" in settings._read_only_notice.text()
        # Read-only keeps every control non-editable: every page, the Overlay
        # switcher and Theme are disabled, and the footer shows the notice
        # where "Reset to defaults" sits. _open_settings showed the window.
        assert settings._jp_size.isEnabled() is False
        assert [settings._stack.widget(i).isEnabled() for i in range(settings._stack.count())] == [False] * settings._stack.count()
        assert settings._theme_choice.isEnabled() is False
        assert settings._read_only_notice.isVisible() and not settings._reset_btn.isVisible()

        before = path.read_bytes()
        settings._theme_choice.setValue("light")
        assert settings._apply() is False
        assert path.read_bytes() == before

        from yamabiko import paths
        assert not paths.user_overrides_path().exists()
    finally:
        if built.tray._settings_window is not None:
            built.tray._settings_window.close()
        built.window.hide()
        built.tray._tray.hide()


def test_overlay_main_reaches_completion_without_attribute_error(monkeypatch, tmp_path):
    """main() must connect screen-change signals via the app instance.

    Full-path invocation with the event loop + WS client + update
    check mocked out. If any ``.connect()`` in main() is on a Signal
    descriptor (class-level access) rather than a bound signal, this
    test raises AttributeError before ``app.exec()``.

    Seeded-defect check: revert commit 26c9bed (change
    ``app.primaryScreenChanged`` back to ``_QGuiApp.primaryScreenChanged``)
    and this test goes RED with the exact AttributeError from the
    user's 2026-04-16 trace.
    """
    rc = _run_overlay_main(monkeypatch, tmp_path)
    assert rc == 0


def test_overlay_main_emits_mainwindow_shown_sentinel(monkeypatch, tmp_path, caplog):
    """main() must log ``overlay.mainwindow.shown`` after ``window.show()``.

    The sentinel is asserted in the frozen-bundle E2E test
    (``test_frozen_exe_default_launch_spawns_pipeline_and_overlay``)
    to distinguish "overlay started" from "overlay crashed but
    watchdog respawned it." This unit test catches sentinel drift
    at unit-test time so the E2E's signal isn't the first indicator.

    Seeded-defect check: delete the ``log.info("overlay.mainwindow.shown")``
    line from overlay_app/__main__.py and this test goes RED.
    """
    import logging
    with caplog.at_level(logging.INFO, logger="yamabiko.overlay_app.__main__"):
        rc = _run_overlay_main(monkeypatch, tmp_path, log_level="INFO")
    assert rc == 0

    # Count, not just presence: the sentinel must fire EXACTLY once.
    # Two-plus means something is re-entering main(); zero means
    # main() didn't reach window.show().
    sentinels = [r for r in caplog.records
                 if "overlay.mainwindow.shown" in r.getMessage()]
    assert len(sentinels) == 1, (
        f"expected exactly 1 'overlay.mainwindow.shown' log record, "
        f"got {len(sentinels)}. Records: {[r.getMessage() for r in sentinels]}"
    )
