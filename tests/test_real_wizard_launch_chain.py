"""Real-path test: `--run-main` + fresh state reaches the first-run wizard.

The wizard has no --run-wizard flag and no __main__ -- it is gated two
spawns deep at main.py's `if wizard_state.needs_wizard(): run_wizard()`.
If a refactor moves that gate or breaks the --run-main dispatch, the
wizard silently never runs, models never download, and llama-server waits
forever.

This drives the REAL dispatch in a subprocess: watchdog.main() with
sys.argv = ["yamabiko", "--run-main"] and a fresh YAMABIKO_APPDATA must
route through main.main()'s needs_wizard() gate and construct
YamabikoWizard. The helper rejects the wizard on sight, so a successful
run exits with the wizard-cancel sentinel 42 -- which can only happen if
run_wizard() was actually reached through the gate. (watchdog.main()'s
--run-main branch calls yamabiko.main.main() IN-PROCESS, so this single
subprocess exercises the whole chain end-to-end.)

The frozen-exe flavour (Start-Menu spawn semantics) lives in
tests/test_real_frozen_e2e.py::test_frozen_fresh_state_reaches_wizard.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The marker run_wizard() logs as its FIRST statement (yamabiko/wizard.py,
# issue #25). It lands in <appdata>/logs/main-pipeline.log because
# main.main() wires that role log before the wizard gate. Its presence is
# the ONLY observable that proves run_wizard() was actually entered:
# exit 42 alone is reachable WITHOUT it (a fall-through gate with the wizard
# call removed still leaves needs_wizard() True -> exit_wizard_cancelled()
# -> 42), so the marker is what stops this test from being a Liar.
#
# Imported from lifecycle.launch_markers (Qt-free single source of truth) so
# the producer string and this gate cannot drift: a change to the logged
# marker is picked up here automatically, as a full-line match.
from yamabiko.lifecycle.launch_markers import WIZARD_START_MARKER as _WIZARD_MARKER

# Any of these in the role log or the child's stderr means an uncaught
# exception, an unguarded stdout write (OSError), or a pipeline stage
# error fired during the launch -- even if the marker + exit 42 both
# landed. The log must be asserted clean on the startup/E2E path.
_BAD_STRINGS = (
    "Traceback", "AttributeError", "ImportError", "ModuleNotFoundError",
    "OSError", "ERROR pipeline stage",
)

# Drives the REAL production dispatch: watchdog.main() detects --run-main
# and calls yamabiko.main.main() in-process, which reaches the
# `if wizard_state.needs_wizard(): run_wizard()` gate. With a fresh
# YAMABIKO_APPDATA, needs_wizard() is True, so run_wizard() constructs the
# wizard; the reject-driver rejects it; main.py's cancel branch calls
# exit_wizard_cancelled() -> exit 42. The ONLY fake here is the wizard
# reject-driver (teardown, not the subject) -- the dispatch + gate +
# run_wizard() are all the real production code.
_HELPER = """
import os, sys, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.argv = ["yamabiko", "--run-main"]

# Fail-fast: if the wizard is never reached (gate moved / dispatch broken),
# exit 97 after 60s so the test gets a deterministic non-42 code, not a
# 120s subprocess.run timeout that reads as an infra flake. The timer MUST
# be a daemon: a non-daemon thread keeps the interpreter alive past the
# main thread's sys.exit(42) (exit_wizard_cancelled's non-draining branch),
# which would swallow the real 42 and let this watchdog fire 97 instead.
_failsafe = threading.Timer(60.0, lambda: os._exit(97))
_failsafe.daemon = True
_failsafe.start()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from yamabiko.ui.qt_bootstrap import ensure_qapplication

# Arm the reject-driver against the QApplication that run_wizard() will
# reuse (same trick as tests/test_real_wizard_cancel.py): ensure one
# exists now so QTimer has a live event loop to attach to.
ensure_qapplication()


def drive():
    from yamabiko import wizard as wizard_mod
    for w in QApplication.topLevelWidgets():
        if isinstance(w, wizard_mod.YamabikoWizard):
            w.reject()
            return
    QTimer.singleShot(200, drive)  # wizard not constructed yet -- retry


# 500 ms is relative to when this QTimer's event loop starts running --
# i.e. once run_wizard() calls wiz.exec() below. The wizard may not be
# constructed on the first poll (the gate + ensure_qapplication run first),
# so drive() re-arms itself every 200 ms until YamabikoWizard appears.
QTimer.singleShot(500, drive)

# The production entry: watchdog.main() owns the --run-main dispatch and
# calls yamabiko.main.main() in-process.
from yamabiko import watchdog
watchdog.main()
"""


@pytest.mark.real
def test_run_main_with_fresh_state_reaches_wizard(tmp_path):
    """`--run-main` + a fresh YAMABIKO_APPDATA must drive the real dispatch
    into run_wizard() (proven by the marker run_wizard() logs) and then exit
    with the watchdog's wizard-cancel sentinel (42). Both observables are
    required: the marker proves the launch chain
    watchdog.main() -> main.main() -> needs_wizard() gate -> run_wizard()
    actually reached run_wizard(); the 42 proves the cancel path completed.

    Seeded defects that turn this red (each a real production break):
    - delete just the `run_wizard()` call at the main.py gate -> the wizard
      never constructs, so the marker never logs. (Exit stays 42 because a
      still-True needs_wizard() falls through to exit_wizard_cancelled() --
      which is exactly why the marker assertion, not the exit code alone,
      is load-bearing here. Verified: dropping run_wizard() leaves rc==42
      but the marker absent, and this test goes red on the marker.)
    - delete the whole `if wizard_state.needs_wizard():` gate -> the
      pipeline starts (or crashes on missing models); no marker, and the
      exit code is no longer 42.
    - break watchdog.main()'s `--run-main` dispatch (drop the
      `if "--run-main" in sys.argv[1:]` branch) -> watchdog mode acquires
      the mutex and spawns children instead of running in-process; no
      marker, no 42 (the daemon 60s timer exits 97).
    """
    env = dict(os.environ)
    env["YAMABIKO_APPDATA"] = str(tmp_path)  # fresh state: needs_wizard() True
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", _HELPER],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, cwd=str(_REPO_ROOT), env=env,
    )
    combined = proc.stdout + proc.stderr

    # The marker proves run_wizard() was actually entered. It lands in the
    # role log under the fresh appdata (main.main() wires main-pipeline.log
    # there before the gate); the StreamHandler copy also shows in stderr in
    # dev mode, so check both so a logging-config change can't hide it.
    log_file = tmp_path / "logs" / "main-pipeline.log"
    log_text = (
        log_file.read_text(encoding="utf-8", errors="replace")
        if log_file.is_file() else ""
    )
    marker_seen = _WIZARD_MARKER in log_text or _WIZARD_MARKER in combined
    assert marker_seen, (
        f"run_wizard() was never reached: marker {_WIZARD_MARKER!r} absent "
        f"from {log_file} and from the child's stderr. The needs_wizard() "
        f"gate did not dispatch into run_wizard().\n"
        f"--- main-pipeline.log ---\n{log_text}\n--- stdout+stderr ---\n{combined}"
    )

    # Log asserted clean (pre-merge checklist): even with the marker present
    # and a 42 exit, an uncaught exception / unguarded stdout write / stage
    # error in the launch window is a real defect. Scan BOTH the role log
    # and the child's stdout+stderr for the charter's bad-string set.
    haystack = log_text + "\n" + combined
    bad = [s for s in _BAD_STRINGS if s in haystack]
    assert not bad, (
        f"the launch reached the wizard but the log/stderr shows {bad} in "
        f"the run window -- an uncaught exception or unguarded stdout write "
        f"on the --run-main -> wizard path.\n"
        f"--- main-pipeline.log ---\n{log_text}\n--- stdout+stderr ---\n{combined}"
    )

    assert proc.returncode == 42, (
        f"--run-main with a fresh YAMABIKO_APPDATA reached the wizard but "
        f"did not exit via the wizard-cancel sentinel (expected 42, got "
        f"{proc.returncode}). The cancel path is broken.\n"
        f"{combined}"
    )
