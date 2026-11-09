"""Real-path test: cancelling the wizard mid-download exits 42, no qFatal.

Drives the end-to-end path in a real subprocess with a real QApplication,
the setup window (``YamabikoWizard``) and a QThread (offscreen platform):
the download worker is blocked mid-"transfer" when the wizard is rejected,
run_wizard releases the page threads, and the process exits through
exit_wizard_cancelled. The window outlives run_wizard (the ClientAreaFit
cycle), so its destruction never frees the thread; without the release and
the hard exit, interpreter finalization takes the running QThread down with
Qt 6's "QThread: Destroyed while thread is still running" qFatal — a non-42
exit code that exit_codes.classify maps to RESTART, making the watchdog
relaunch the wizard the user just cancelled.

The downloader is faked (blocking) because the subject here is the
PySide6 thread-teardown / process-exit seam, not the HF transfer — the
real-HF seam is covered by the @real tests in
tests/wizard_pages/test_downloads.py.

Coverage boundary: this test exercises the DEV-mode cancel path only — it
spawns the dev interpreter (``sys.executable``), not the frozen exe. The
frozen setup is automated up to its first download:
``tests/test_real_frozen_e2e.py``'s
``test_frozen_fresh_wizard_download_loads_hf_xet_and_writes_fresh_cache``
clicks from Welcome to Download, then kills the process tree, so no frozen
run cancels the wizard. The frozen cancel is owned by the manual
clean-install E2E in ``docs/RELEASE.md``, a required release-gate step.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Drives the REAL production path: run_wizard() (which must release the
# blocked thread while the wizard is still alive) followed by main.py's
# cancel branch, exit_wizard_cancelled() (which must hard-exit while a
# detached thread is draining).
_HELPER = """
import os, sys, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from yamabiko.wizard_pages import downloads as dl


class _BlockingDownloader:
    def download(self, repo_id, filename, revision=None, progress_callback=None):
        threading.Event().wait(30.0)  # longer than the whole test window
        return Path("/never")


dl._HFDownloader = _BlockingDownloader

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from yamabiko import wizard as wizard_mod
from yamabiko import wizard_state
from yamabiko.ui.qt_bootstrap import ensure_qapplication

# QTimer needs a live QApplication; run_wizard() reuses this instance.
ensure_qapplication()


def drive():
    for w in QApplication.topLevelWidgets():
        if isinstance(w, wizard_mod.YamabikoWizard):
            # Start the download queue (the page normally does this when
            # the user navigates onto it), then cancel mid-download.
            w._downloads_page.initializePage()
            QTimer.singleShot(1500, w.reject)
            return
    QTimer.singleShot(200, drive)  # wizard not constructed yet — retry


QTimer.singleShot(500, drive)
wizard_mod.run_wizard()

if wizard_state.needs_wizard():
    wizard_mod.exit_wizard_cancelled()
else:
    sys.exit(99)  # state unexpectedly saved — the cancel never happened
"""


@pytest.mark.real
def test_real_wizard_cancel_mid_download_exits_42_without_qthread_abort(tmp_path):
    """Reject the wizard while the download QThread is blocked mid-file →
    the process must exit with the watchdog's wizard-cancel sentinel (42)
    and must not print Qt's qFatal message.

    Seeded defects that turn this red:
    - remove run_wizard()'s release_background_threads() call → qFatal when
      the wizard object is destroyed;
    - replace exit_wizard_cancelled()'s os._exit branch with sys.exit →
      qFatal during interpreter finalization.
    Either way the returncode stops being 42.
    """
    env = dict(os.environ)
    env["YAMABIKO_APPDATA"] = str(tmp_path)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", _HELPER],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, cwd=str(_REPO_ROOT), env=env,
    )
    combined = proc.stdout + proc.stderr
    assert "QThread: Destroyed" not in combined, (
        f"Qt aborted on a running thread:\n{combined}"
    )
    assert "Traceback" not in combined, f"unexpected exception:\n{combined}"
    assert proc.returncode == 42, (
        f"expected wizard-cancel sentinel 42, got {proc.returncode};"
        f" output:\n{combined}"
    )
