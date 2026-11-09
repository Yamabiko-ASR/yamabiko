"""Tests for the shared Qt bootstrap: HiDPI-aware QApplication creation
(F_J4 / audit Wave-4 HiDPI gap) and Qt-diagnostics routing (2026-07-01
logging audit, Task 4). The PassThrough rounding policy must be applied
BEFORE the QApplication is constructed -- Qt ignores it afterward -- and
the Qt message handler must be installed before construction too, so
diagnostics Qt emits DURING construction are already routed to the log.

QApplication is stubbed so no real Qt app is created and the tests can
run headless without a display. The one @pytest.mark.real test exercises
the REAL qInstallMessageHandler + qWarning dispatch (hard rule #3:
PySide6 is an integration seam); Qt's message machinery is QtCore-level
and needs no QApplication, so it too runs headless.
"""
from __future__ import annotations

import logging

import pytest


def test_ensure_qapplication_sets_passthrough_before_create(monkeypatch):
    """When no app exists, the policy is set, then the app is created --
    in that order.

    Seeded defect: move setHighDpiScaleFactorRoundingPolicy after
    QApplication([]) -> order becomes ['create', 'policy'] -> this fails.
    """
    import yamabiko.ui.qt_bootstrap as qb
    from PySide6.QtCore import Qt

    order = []
    captured = {}

    class _FakeApp:
        @staticmethod
        def instance():
            return None

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy):
            order.append("policy")
            captured["policy"] = policy

        def __init__(self, _argv):
            order.append("create")

    monkeypatch.setattr(qb, "QApplication", _FakeApp)
    qb.ensure_qapplication()
    assert order == ["policy", "create"], order
    # And it must be PassThrough specifically (not just any policy).
    assert captured["policy"] == Qt.HighDpiScaleFactorRoundingPolicy.PassThrough


def test_ensure_qapplication_reuses_existing_app(monkeypatch):
    """When an app already exists, return it without setting the policy
    (too late to matter) or creating a second app."""
    import yamabiko.ui.qt_bootstrap as qb

    sentinel = object()
    calls = []

    class _FakeApp:
        @staticmethod
        def instance():
            return sentinel

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(_policy):
            calls.append("policy")

        def __init__(self, _argv):
            calls.append("create")

    monkeypatch.setattr(qb, "QApplication", _FakeApp)
    assert qb.ensure_qapplication() is sentinel
    assert calls == []


# ---------------------------------------------------------------------------
# Qt-diagnostics routing (qInstallMessageHandler -> yamabiko.qt logger)
# ---------------------------------------------------------------------------


def test_qt_message_handler_routes_each_qt_severity_to_matching_level(caplog):
    """_qt_message_handler maps every Qt severity onto the stdlib level
    the audit assigned (Debug->DEBUG, Info->INFO, Warning->WARNING,
    Critical->ERROR, Fatal->CRITICAL) and logs the message text verbatim
    on the yamabiko.qt logger.

    Seeded defect: map QtMsgType.QtCriticalMsg to logging.WARNING in
    _QT_LEVEL -> the QtCriticalMsg iteration fails its levelno assert.
    """
    import yamabiko.ui.qt_bootstrap as qb
    from PySide6.QtCore import QtMsgType

    expected = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    with caplog.at_level(logging.DEBUG, logger="yamabiko.qt"):
        for mode, level in expected.items():
            caplog.clear()
            qb._qt_message_handler(mode, None, f"probe-{mode.name}")
            records = [r for r in caplog.records if r.name == "yamabiko.qt"]
            assert len(records) == 1, (mode, caplog.records)
            assert records[0].levelno == level, (mode, records[0].levelno)
            assert records[0].getMessage() == f"probe-{mode.name}"


def test_qt_message_handler_falls_back_to_warning_for_unknown_mode(caplog):
    """A Qt severity the map has never heard of (a future QtMsgType
    member) must still surface in the log -- at WARNING, never dropped.

    Seeded defect: change _QT_LEVEL.get's default from logging.WARNING
    to logging.DEBUG -> the levelno assert fails.
    """
    import yamabiko.ui.qt_bootstrap as qb

    with caplog.at_level(logging.DEBUG, logger="yamabiko.qt"):
        qb._qt_message_handler(object(), None, "probe-unknown-mode")
    records = [r for r in caplog.records if r.name == "yamabiko.qt"]
    assert len(records) == 1, caplog.records
    assert records[0].levelno == logging.WARNING, records[0].levelno
    assert records[0].getMessage() == "probe-unknown-mode"


def test_ensure_qapplication_installs_qt_handler_before_app_creation(monkeypatch):
    """The message handler is installed before QApplication construction,
    so diagnostics Qt emits while the app is being built (DPI warnings,
    platform-plugin complaints) are already routed to the log.

    Seeded defects: drop the install_qt_message_handler() call from
    ensure_qapplication -> no 'install' entry -> red; move it after
    QApplication([]) -> the ordering assert goes red.
    """
    import yamabiko.ui.qt_bootstrap as qb

    order = []
    captured = {}

    class _FakeApp:
        @staticmethod
        def instance():
            return None

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(_policy):
            order.append("policy")

        def __init__(self, _argv):
            order.append("create")

    def _fake_install(handler):
        order.append("install")
        captured["handler"] = handler

    monkeypatch.setattr(qb, "QApplication", _FakeApp)
    monkeypatch.setattr(qb, "qInstallMessageHandler", _fake_install)
    monkeypatch.setattr(qb, "_qt_handler_installed", False)
    qb.ensure_qapplication()
    assert "install" in order, order
    assert order.index("install") < order.index("create"), order
    # The exact production handler, not some other callable.
    assert captured["handler"] is qb._qt_message_handler


def test_ensure_qapplication_installs_qt_handler_when_app_already_exists(monkeypatch):
    """Re-entry (wizard rerun, startup error dialog after another entry
    point built the app) still gets Qt routing: the handler is installed
    on the reuse path too, not only when this call creates the app.

    Seeded defect: move the install_qt_message_handler() call under the
    `if app is None:` branch -> no install recorded here -> red.
    """
    import yamabiko.ui.qt_bootstrap as qb

    sentinel = object()
    installs = []

    class _FakeApp:
        @staticmethod
        def instance():
            return sentinel

        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(_policy):
            raise AssertionError("policy must not be set on the reuse path")

        def __init__(self, _argv):
            raise AssertionError("no second app on the reuse path")

    monkeypatch.setattr(qb, "QApplication", _FakeApp)
    monkeypatch.setattr(qb, "qInstallMessageHandler", installs.append)
    monkeypatch.setattr(qb, "_qt_handler_installed", False)
    assert qb.ensure_qapplication() is sentinel
    assert installs == [qb._qt_message_handler]


def test_install_qt_message_handler_installs_only_once(monkeypatch):
    """Every Qt entry point calls install_qt_message_handler(), so repeat
    calls must be a no-op: exactly one qInstallMessageHandler call total.

    Seeded defect: delete the _qt_handler_installed guard -> two calls
    recorded -> red.
    """
    import yamabiko.ui.qt_bootstrap as qb

    installs = []
    monkeypatch.setattr(qb, "qInstallMessageHandler", installs.append)
    monkeypatch.setattr(qb, "_qt_handler_installed", False)
    qb.install_qt_message_handler()
    qb.install_qt_message_handler()
    assert len(installs) == 1, installs


@pytest.mark.real
def test_real_qwarning_routes_through_installed_handler_to_yamabiko_qt_logger(
    monkeypatch, caplog,
):
    """Real-path (hard rule #3): the REAL qInstallMessageHandler + REAL
    qWarning dispatch -- no stubs on the subject -- lands as a WARNING
    record on the yamabiko.qt logger. Qt's message machinery needs no
    QApplication (QtCore-level), so this runs headless; it is marked
    real because its subject is the live PySide6 seam.

    Seeded defect: make install_qt_message_handler set its installed
    flag without calling qInstallMessageHandler -> qWarning goes to Qt's
    default stderr writer, no record captured -> red.
    """
    import yamabiko.ui.qt_bootstrap as qb
    from PySide6.QtCore import qInstallMessageHandler, qWarning

    monkeypatch.setattr(qb, "_qt_handler_installed", False)
    try:
        qb.install_qt_message_handler()
        with caplog.at_level(logging.DEBUG, logger="yamabiko.qt"):
            qWarning("qt-bootstrap-probe-warning")
    finally:
        # Restore the EXACT handler the install displaced (pytest-qt's
        # per-test capture when the plugin is active, else Qt's default).
        # Resetting to None here left pytest-qt's teardown restoring over
        # a chain it no longer owned -- the mid-test churn suspected in a
        # once-seen CI heap corruption (0xC0000374) during the later
        # widget tests' event drain.
        qInstallMessageHandler(qb._displaced_qt_handler)
    records = [r for r in caplog.records if r.name == "yamabiko.qt"]
    assert len(records) == 1, caplog.records
    assert records[0].levelno == logging.WARNING, records[0].levelno
    assert records[0].getMessage() == "qt-bootstrap-probe-warning"


def test_fresh_python_process_installs_qt_handler_on_first_call():
    """A brand-new Python process must get Qt routing from the FIRST
    install_qt_message_handler() call, with the module exactly as
    shipped. Every other install-path test in this file monkeypatches
    ``_qt_handler_installed = False`` before exercising the install, so
    none of them can see a regression of the SHIPPED initial value:
    seeding ``_qt_handler_installed = True`` in the module leaves all of
    them green while every real Yamabiko process silently loses its Qt
    diagnostics. This test imports the module fresh in a subprocess --
    no monkeypatching anywhere -- and asserts the first call actually
    routes a real qWarning to the yamabiko.qt logger.

    Seeded-defect drill: flip the module literal
    ``_qt_handler_installed = False`` to ``True`` -> the child's
    install call no-ops, qWarning falls through to Qt's default stderr
    writer, no record lands, the child exits nonzero -> red.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    child_script = (
        "import logging\n"
        "records = []\n"
        "class _Cap(logging.Handler):\n"
        "    def emit(self, record): records.append(record)\n"
        "lg = logging.getLogger('yamabiko.qt')\n"
        "lg.addHandler(_Cap())\n"
        "lg.setLevel(logging.DEBUG)\n"
        "import yamabiko.ui.qt_bootstrap as qb\n"
        "assert qb._qt_handler_installed is False, (\n"
        "    'shipped initial value regressed: %r' % (qb._qt_handler_installed,))\n"
        "qb.install_qt_message_handler()\n"
        "from PySide6.QtCore import qWarning\n"
        "qWarning('fresh-import-probe')\n"
        "assert any(\n"
        "    r.getMessage() == 'fresh-import-probe'\n"
        "    and r.levelno == logging.WARNING for r in records\n"
        "), 'first install call did not route qWarning; records=%r' % (\n"
        "    [r.getMessage() for r in records],)\n"
        "print('QT-HANDLER-FRESH-INSTALL-OK')\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", child_script],
        capture_output=True, text=True, timeout=45,
        cwd=str(repo_root), env=env,
    )
    assert proc.returncode == 0, (
        f"fresh-import child failed (rc={proc.returncode});\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "QT-HANDLER-FRESH-INSTALL-OK" in proc.stdout, proc.stdout
