import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _tier_probe_answers_cached(monkeypatch):
    """Determinism guard for the WHOLE overlay test package, not a behaviour
    claim: every test runs with the quant-tier cache probe answering 'cached'
    unless the test itself overrides it (``_force_tier_cached`` / an in-test
    monkeypatch is applied later and wins). ``_persist_quant_tier`` probes the
    REAL HF cache on every Apply (the stranded-install heal), and the tray's
    ``_open_settings`` builds the REAL SettingsWindow with the REAL config
    path -- so without this pin, ANY test driving an Apply/OK path has an
    outcome that depends on the machine's cache state: green on a dev box
    with the GGUF cached, a REAL 5.14 GiB download on a cache-miss runner.
    The 2026-07-11 CI runs proved it twice: first the settings position-apply
    test, then (after a file-scoped pin) the tray position-apply test, each
    pulled gigabytes of GGUF into the unit job before dying on the 60 s
    timeout. The probe's own behaviour stays pinned by the dedicated
    quant-tier tests in test_settings_window.py, which override this both
    ways and inject a recording downloader. The unit CI job additionally
    sets HF_HUB_OFFLINE=1 so any future leak of this class fails fast and
    loud instead of downloading.
    """
    from yamabiko.overlay_app import model_fetch
    monkeypatch.setattr(model_fetch, "tier_cached", lambda tier: True)


@pytest.fixture(autouse=True)
def _shortcut_probe_answers_free(request, monkeypatch):
    """Determinism guard, not a behaviour claim: Settings probes its saved
    shortcuts through Win32 RegisterHotKey when it opens; a machine where
    another app holds Ctrl+Alt+S would otherwise change what every Settings
    test sees. The probe itself is covered by tests/test_real_hotkey_probe.py;
    Settings tests that need a taken combo inject ``hotkey_probe``. Real-tier
    tests keep the real probe (test_real_shortcut_field.py asserts it)."""
    if request.node.get_closest_marker("real") is not None:
        return
    from yamabiko import hotkeys
    monkeypatch.setattr(hotkeys, "combo_taken", lambda combo: False)


@pytest.fixture(autouse=True)
def _drop_application_fonts():
    """Isolation guard, not a behaviour claim: restore the fontless baseline.

    build_overlay() (and the real font test) registers the bundled fonts with
    QFontDatabase for the whole process. The offscreen QPA on Windows otherwise
    has no font families at all, and existing tests assume that baseline (the
    subtitle-stack test relies on the very wide fallback overflowing the
    window; the substituted-font test relies on QFontComboBox showing a
    substitute). Removing the application fonts after each test keeps every
    test's font environment independent of test order. Only touches Qt when a
    test already imported it and built a QGuiApplication.
    """
    yield
    if "PySide6.QtGui" in sys.modules:
        from PySide6.QtGui import QFontDatabase, QGuiApplication

        if QGuiApplication.instance() is not None:
            QFontDatabase.removeAllApplicationFonts()
