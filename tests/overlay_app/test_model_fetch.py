"""Tests for the Settings-owned quant-tier model acquisition.

The download machinery is the wizard's ``_DownloadWorker`` (verified in
tests/wizard_pages/test_downloads.py); here we verify the settings-side
seam: the tier -> spec / cache-probe mapping (single-sourced from
yamabiko.models) and the modal :class:`QuantDownloadDialog` that drives a
REAL worker on a REAL QThread with a fake at the ``_Downloader`` seam
(unit tier -- no multi-GB network transfer). Cancel / error / success /
quit-release are exercised end-to-end through the dialog's own slots, not
by asserting wiring.

Uses ``qtbot`` (pytest-qt) so every dialog + its child QThread is torn
down through pytest-qt's managed teardown -- a plain-fixture dialog left
undeleted leaks Qt state that trips the suite's native teardown flake.
"""
import threading
from pathlib import Path

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog

from yamabiko.overlay_app import model_fetch
from yamabiko.overlay_app.model_fetch import QuantDownloadDialog, spec_for_tier, tier_cached
from yamabiko import models
from yamabiko.ui import thread_keeper
from tests.overlay_app._dialog_wait import exec_with_watchdog, when
from tests.overlay_app._qt_drain import drain_qt_debris


@pytest.fixture(autouse=True)
def _drain_qt_objects():
    """Force lingering QThread/worker/dialog wrappers to be destroyed WITHIN
    this test's own teardown, not a later (more fragile) test's -- see
    tests/overlay_app/_qt_drain.py for why."""
    yield
    drain_qt_debris()


# ---------------------------------------------------------------------------
# tier -> spec / cache probe (single-sourced from yamabiko.models)
# ---------------------------------------------------------------------------


def test_spec_for_tier_maps_tiers_to_the_models_pins():
    """spec_for_tier must return each tier pin's repo, filename, AND
    revision from yamabiko.models -- never a duplicated literal.

    Seeded defect: make spec_for_tier ignore the tier argument (always
    build from the default pin) -> the small spec carries the default
    filename -> red.
    """
    default_pin = models.TRANSLATOR_PINS["default"]
    small_pin = models.TRANSLATOR_PINS["small"]
    default_spec = spec_for_tier("default")
    small_spec = spec_for_tier("small")
    assert default_spec.repo_id == default_pin.repo
    assert default_spec.filename == default_pin.filename
    assert small_spec.repo_id == small_pin.repo
    assert small_spec.filename == small_pin.filename
    # Pinned to each tier's own revision, like the wizard's translator
    # download + the runtime resolver -- a Settings re-download must land
    # the same snapshot the resolver probes.
    assert small_spec.revision == small_pin.revision
    assert default_spec.revision == default_pin.revision


def test_tier_cached_delegates_to_the_models_cache_probe(monkeypatch):
    """tier_cached must reuse models._repo_files_cached (the try_to_load_from_cache
    probe) with the tier pin's filename AND its pinned revision -- not invent
    a second probe, and not probe unpinned (which would resolve refs/main and
    drift from the pinned runtime resolver).

    Seeded defect: pass a wrong filename tuple, or revision=None, to
    _repo_files_cached -> recorded args mismatch -> red.
    """
    calls = []

    def _fake_probe(repo_id, filenames, revision):
        calls.append((repo_id, tuple(filenames), revision))
        return True

    monkeypatch.setattr(model_fetch, "_repo_files_cached", _fake_probe)
    assert tier_cached("small") is True
    small_pin = models.TRANSLATOR_PINS["small"]
    assert calls == [
        (small_pin.repo, (small_pin.filename,), small_pin.revision)
    ]


def test_tier_cached_fails_closed_when_the_probe_raises(monkeypatch):
    """A probe error (hub import / cache-layout failure) must be reported as
    NOT cached, so the download path still runs and actually fetches the model.

    Seeded defect: change tier_cached's except to `return True` (fail open) ->
    a probe error would report cached and skip the fetch -> red.
    """
    def _boom(*a, **kw):
        raise RuntimeError("hub import blew up")

    monkeypatch.setattr(model_fetch, "_repo_files_cached", _boom)
    assert tier_cached("small") is False


# ---------------------------------------------------------------------------
# QuantDownloadDialog -- driven end-to-end with a fake downloader
# ---------------------------------------------------------------------------


def test_dialog_success_accepts_with_success_outcome_and_right_spec(qtbot):
    """A successful download resolves the dialog to Accepted / outcome
    'success' AND the worker asked the downloader for the tier's exact
    repo+filename.

    Seeded defect: make QuantDownloadDialog build the spec from the wrong tier
    -> requested filename mismatches -> red; or drop the accept() in the
    success branch of _on_queue_finished -> outcome not 'success' -> red.
    """
    requested = []

    class _Recording:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            requested.append((repo_id, filename))
            if progress_callback is not None:
                progress_callback(10, 10)
            return Path(f"/fake/{filename}")

    dialog = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=_Recording(),
    )
    qtbot.addWidget(dialog)
    result = dialog.exec()

    assert dialog.outcome == "success"
    assert result == QDialog.Accepted
    small_pin = models.TRANSLATOR_PINS["small"]
    assert requested == [(small_pin.repo, small_pin.filename)]
    assert dialog._thread is None and dialog._worker is None


def test_dialog_failure_offers_retry_then_cancel_keeps_the_failed_outcome(qtbot):
    """A failed download keeps the dialog open in its failed state (stopped
    bar, Retry and Copy details shown); Cancel from there settles 'failed'
    with the raw error and a friendly message.

    Seeded defects: reject at once on failure (old behavior) -> the dialog
    is gone before Retry shows -> red; let reject() overwrite the outcome
    with 'cancelled' -> red."""

    class _Failing:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            raise RuntimeError("404 Client Error: file not found")

    dialog = QuantDownloadDialog(spec_for_tier("default"), "default", downloader=_Failing())
    qtbot.addWidget(dialog)
    seen = {}

    def _record_and_cancel():
        seen["stopped"] = dialog.progress.stopped()
        seen["status"] = dialog.status.text()
        dialog.cancel_button.click()

    when(dialog.retry_button.isVisible, _record_and_cancel)
    result = exec_with_watchdog(dialog)

    assert seen == {"stopped": True, "status": "The download stopped.\nCheck your internet connection, then click Retry."}
    assert result == QDialog.Rejected and dialog.outcome == "failed"
    assert isinstance(dialog.error, RuntimeError)
    friendly = dialog.friendly_error()
    assert friendly and "404" not in friendly


def test_dialog_retry_after_a_failure_downloads_again_and_succeeds(qtbot):
    """Retry restarts the download in the same dialog. Seeded defect: Retry
    that only hides the failed state -> no second download -> red."""

    calls = []

    class _FailOnce:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            calls.append(filename)
            if len(calls) == 1:
                raise RuntimeError("connection reset")
            return Path(f"/fake/{filename}")

    dialog = QuantDownloadDialog(spec_for_tier("small"), "small", downloader=_FailOnce())
    qtbot.addWidget(dialog)
    when(dialog.retry_button.isVisible, dialog.retry_button.click)
    assert exec_with_watchdog(dialog) == QDialog.Accepted
    assert dialog.outcome == "success" and len(calls) == 2


def test_dialog_user_cancel_ends_cancelled_with_no_error(qtbot):
    """A cancel issued mid-download (the worker's own cancel path) resolves the
    dialog to 'cancelled' with no error recorded -- cancelled is not failed.

    Seeded defect: in _on_queue_finished, treat the no-error non-ok case as
    'failed' -> outcome 'failed' -> red.
    """
    box = {}

    class _SelfCancel:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            # Simulate the user clicking Cancel while this file is in flight.
            box["dialog"]._worker.cancel()
            return Path(f"/fake/{filename}")

    dialog = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=_SelfCancel(),
    )
    qtbot.addWidget(dialog)
    box["dialog"] = dialog
    dialog.exec()

    assert dialog.outcome == "cancelled"
    assert dialog.error is None
    assert dialog._thread is None and dialog._worker is None


def test_amount_and_time_left_copy():
    # The dialog copy: "1.7 of 4 GB", "About 3 minutes left". Seeded defects:
    # print "4.0 GB", or round minutes down to 2 -> red.
    assert model_fetch.format_amount(1_700_000_000, 4_000_000_000) == "1.7 of 4 GB"
    assert model_fetch.format_amount(0, 5_100_000_000) == "0 of 5.1 GB"
    assert model_fetch.format_time_left(170) == "About 3 minutes left"
    assert model_fetch.format_time_left(70) == "About a minute left"
    assert model_fetch.format_time_left(30) == "Less than a minute left"
    assert model_fetch.size_text(4.0) == "4 GB"


def test_dialog_progress_slot_clamps_and_reports_amount_and_time_left(qtbot):
    """The progress slot renders "N of M GB", a clamped bar, and time left
    once two seconds of progress are in (fake clock).

    Seeded defects: render the raw over-count (xet drift) -> fraction > 1
    -> red; never estimate time left -> red."""
    ticks = iter([0.0, 94.0, 95.0])
    dialog = QuantDownloadDialog(spec_for_tier("small"), "small", clock=lambda: next(ticks))
    qtbot.addWidget(dialog)
    dialog._on_progress(500_000_000, 4_000_000_000)
    dialog._on_progress(1_700_000_000, 4_000_000_000)
    assert dialog.amount.text() == "1.7 of 4 GB"
    assert dialog.time_left.text() == "About 3 minutes left"
    dialog._on_progress(9_000_000_000, 4_000_000_000)
    assert dialog.progress.fraction() == 1.0


def test_dialog_release_threads_parks_blocked_worker_and_fires_no_callback(qtbot):
    """Quit while a download is blocked inside an uninterruptible hf_hub_download:
    release_threads() must detach the running QThread into thread_keeper (not
    destroy it), and no success/accept callback may fire afterwards.

    Seeded defect: replace the thread_keeper.release(...) in
    _release_running_thread with a plain `self._thread = None` -> the running
    thread is dropped without parking -> the keeper assertions go red.
    """
    entered = threading.Event()
    gate = threading.Event()

    class _Blocking:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    dialog = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=_Blocking(),
    )
    qtbot.addWidget(dialog)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.start()  # drive the worker directly, without the modal exec loop
    assert entered.wait(5.0), "worker never entered the blocking download"
    thread = dialog._thread
    try:
        dialog.release_threads()

        assert thread.parent() is None, "released thread kept its widget parent"
        assert any(t is thread for t, _ in thread_keeper._LINGERING), (
            "released thread was not parked in thread_keeper"
        )
        assert thread_keeper.any_running()
        assert dialog._thread is None and dialog._worker is None
        assert dialog.outcome != "success"
        assert dialog._settled is True
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)

    qtbot.wait(50)
    assert accepted == [], "an accept callback fired after release_threads()"
    assert dialog.outcome != "success"


def test_cancel_button_click_settles_cancelled_and_parks_blocked_worker(qtbot):
    """Clicking the dialog's Cancel button mid-download must drive reject():
    the dialog settles as cancelled, the worker is told to cancel, and the
    still-blocked thread is detached into thread_keeper (not destroyed).
    The outcome string alone would be evergreen ('cancelled' is the ctor
    default); _settled / worker-cancel / parked are the load-bearing asserts.

    Seeded defect: delete the `self.cancel_button.clicked.connect(self.reject)`
    wiring in __init__ -> the click no-ops (dialog unsettled, worker never
    cancelled, thread keeps running under the dialog) -> red.
    """
    entered = threading.Event()
    gate = threading.Event()

    class _Blocking:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    dialog = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=_Blocking(),
    )
    qtbot.addWidget(dialog)
    dialog.start()  # drive the worker directly, without the modal exec loop
    assert entered.wait(5.0), "worker never entered the blocking download"
    thread = dialog._thread
    worker = dialog._worker
    try:
        dialog.cancel_button.click()

        assert dialog._settled is True, "Cancel click did not settle the dialog"
        assert dialog.outcome == "cancelled"
        assert dialog.error is None
        assert worker._cancelled is True, (
            "Cancel click did not request the worker cancel"
        )
        assert dialog._thread is None and dialog._worker is None
        assert any(t is thread for t, _ in thread_keeper._LINGERING), (
            "cancelled-while-blocked thread was not parked in thread_keeper"
        )
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_dialog_start_shows_waiting_hint_while_a_parked_download_drains(qtbot):
    """A retry started while a previously canceled worker is still draining
    inside an uninterruptible hf_hub_download (parked in thread_keeper) must
    say 'Waiting for the other download to finish…' instead of sitting
    silent at "0 of 4 GB" -- HF's per-blob lock serializes the fresh worker
    behind the drain with no other explanation. Mirrors the wizard's
    DownloadsPage._start_downloads hint for the identical seam; the first
    progress tick clears it.

    Seeded defect: drop the thread_keeper.any_running() branch from
    QuantDownloadDialog.start() -> the second dialog shows no status -> red.
    """
    first_entered = threading.Event()
    first_gate = threading.Event()

    class _FirstBlocking:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            first_entered.set()
            first_gate.wait(timeout=10.0)
            return Path("/fake/first")

    second_gate = threading.Event()

    class _SecondBlocking:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            second_gate.wait(timeout=10.0)
            return Path("/fake/second")

    first = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=_FirstBlocking(),
    )
    qtbot.addWidget(first)
    first.start()
    assert first_entered.wait(5.0), "first worker never entered its download"
    first_thread = first._thread

    second = None
    second_thread = None
    try:
        first.release_threads()  # quit-style park: the drain lives on
        assert thread_keeper.any_running(), "park precondition not met"

        second = QuantDownloadDialog(
            spec_for_tier("small"), "small", downloader=_SecondBlocking(),
        )
        qtbot.addWidget(second)
        assert second.status.isHidden()  # nothing to say before start
        second.start()
        second_thread = second._thread

        assert second.status.text() == "Waiting for the other download to finish…"
        assert second.amount.text() == "0 of 4 GB"
    finally:
        if second is not None:
            second.release_threads()  # settle + park the blocked worker
        first_gate.set()
        second_gate.set()
        for t in (first_thread, second_thread):
            if t is not None:
                t.wait(5000)
                thread_keeper.discard(t)


# ---------------------------------------------------------------------------
# Real-path: app quit/exit while the MODAL exec loop is running
# ---------------------------------------------------------------------------

# Driven in a SUBPROCESS on purpose, for two structural reasons:
#
# 1. QApplication.exit()/quit() sets the main thread's quitNow flag, which
#    makes every LATER nested QEventLoop::exec in the same process return -1
#    immediately -- calling it inside the pytest process would silently break
#    every subsequent qtbot.waitSignal / dialog.exec in the session, and the
#    only reset path (a full app.exec() round) re-fires aboutToQuit into
#    handlers other tests left connected on the shared QApplication.
# 2. The defect's observable IS process-level: an unparked running QThread is
#    destroyed at interpreter finalization -> Qt 6 qFatal abort -> nonzero
#    exit code. A subprocess asserts on exactly that.
#
# Two modes, matching the two ways an app teardown reaches the modal loop on
# Qt 6.5+ (verified empirically on PySide6 6.11):
#
# * mode=exit -- QApplication.exit(0) unwinds ALL stacked event loops
#   directly: the modal exec() returns UNSETTLED (no accept/reject ran), the
#   worker is still blocked. This is the orphaning mechanism (also reached by
#   a quit whose window-close leg skips the dialog, or an exec() entered
#   after quitNow was already set). The park must happen in
#   _download_quant_tier's own unwind -- by aboutToQuit time
#   _active_quant_dialog is already None.
# * mode=quit -- the REAL production sequence: app.exec() runs underneath
#   (as run_overlay does), the tray-style app.quit() fires while the modal
#   loop runs. Qt's Quit event first closes top-level windows, driving the
#   dialog's closeEvent -> reject(), which must settle cancelled and park the
#   blocked worker.
#
# Ending: after asserting keeper state the script RELEASES the blocked fake
# download and JOINS every parked thread, then exits normally -- exit code 0
# iff the worker was parked. (A finished QThread is safe to destroy at
# teardown -- the thread_keeper contract -- so both outcomes end without a
# live thread; deliberately NOT os._exit, whose ExitProcess DLL-detach was
# observed to access-violate with Qt in the force-unwound state, masking the
# result. run_overlay's own any_running()->os._exit contract is asserted via
# the PARKED marker.)
_QUIT_DURING_DOWNLOAD_SCRIPT = r"""
import os, sys, threading
os.environ["QT_QPA_PLATFORM"] = "offscreen"
appdata = sys.argv[1]
mode = sys.argv[2]  # "exit" or "quit"
os.environ["YAMABIKO_APPDATA"] = appdata
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

app = QApplication([])
app.setQuitOnLastWindowClosed(False)  # the overlay's production setting

from yamabiko.overlay_app import model_fetch
from yamabiko.overlay_app.settings_window import SettingsWindow
from yamabiko.ui import thread_keeper

model_fetch.tier_cached = lambda tier: False  # force the download path

entered = threading.Event()
release_gate = threading.Event()  # held until after the assertions: the
                                  # worker stays blocked, like a real
                                  # multi-GB hf_hub_download


class _Blocking:
    def download(self, repo_id, filename, revision=None, progress_callback=None):
        entered.set()
        release_gate.wait(timeout=15.0)
        return Path("/fake/f")


w = SettingsWindow(
    list_devices=lambda: [],
    config_path=Path(appdata) / "ignored.yaml",
    quant_downloader=_Blocking(),
)
w._model_choice.setValue("small")  # default -> small

_teardown = app.quit if mode == "quit" else (lambda: app.exit(0))

state = {}


def _teardown_once_blocked():
    # Fire the app teardown from inside the modal loop's event dispatch, once
    # the worker is provably blocked inside the download.
    if entered.is_set():
        _teardown()
    else:
        QTimer.singleShot(10, _teardown_once_blocked)


def _run_apply():
    state["apply"] = w._apply()


QTimer.singleShot(10, _teardown_once_blocked)
if mode == "quit":
    # Full production shape: the modal opens INSIDE the running app loop
    # (run_overlay drives app.exec()); the quit closes the dialog (reject)
    # and then unwinds the app loop.
    QTimer.singleShot(0, _run_apply)
    QTimer.singleShot(30000, lambda: os._exit(9))  # hard safety
    app.exec()
else:
    # exit(0) unwinds the bare modal loop directly -- no app.exec needed.
    _run_apply()  # enters QuantDownloadDialog.exec(); exit(0) unwinds it

parked = thread_keeper.any_running()
print(f"PARKED={parked} APPLY={state.get('apply')}", flush=True)
# Drain: let the blocked worker finish and join every parked thread so the
# process ends with no live thread regardless of outcome (a finished QThread
# is safe to destroy at teardown).
release_gate.set()
for _thread, _worker in thread_keeper._LINGERING:
    _thread.wait(10000)
sys.exit(0 if parked else 3)
"""


@pytest.mark.real
@pytest.mark.parametrize("mode", ["exit", "quit"])
def test_real_app_teardown_during_modal_download_parks_thread_and_exits_cleanly(
    tmp_path, mode,
):
    """App teardown fired WHILE the modal download dialog's exec() loop is
    running must park the still-blocked worker in thread_keeper and let the
    process hard-exit cleanly -- not abort at interpreter teardown.

    mode=exit: QApplication.exit(0) unwinds all stacked loops, so the modal
    dialog.exec() inside _download_quant_tier returns UNSETTLED; its finally
    nulls _active_quant_dialog, so the aboutToQuit release path
    (Tray.release_threads -> SettingsWindow.release_threads) has nothing to
    reach -- the park must happen in _download_quant_tier's own unwind.

    mode=quit: the real tray-quit sequence (app.exec() underneath, app.quit()
    while the modal runs). Qt 6.5+ closes the dialog first -> closeEvent ->
    reject() -> the reject leg must park the blocked worker.

    Drives the REAL sequence in a subprocess: _apply ->
    QuantDownloadDialog.exec() -> QApplication quit/exit with a blocked fake
    downloader (see the script comment for why a subprocess and how it ends
    without leaving a live thread).

    Seeded defects (both drilled red): mode=exit goes red when
    dialog.release_threads() is dropped from _download_quant_tier's finally
    (the HEAD d25156e state: PARKED=False + rc=3). mode=quit goes red when
    the release is dropped from QuantDownloadDialog.reject().
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    script = tmp_path / "teardown_during_download.py"
    script.write_text(_QUIT_DURING_DOWNLOAD_SCRIPT, encoding="utf-8")
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(script), str(appdata), mode],
        capture_output=True, text=True, timeout=90, env=env,
        cwd=str(repo_root),
    )

    assert "PARKED=True" in proc.stdout, (
        f"[{mode}] app teardown during the modal download did not park the "
        f"worker in thread_keeper.\nstdout: {proc.stdout!r}\n"
        f"stderr: {proc.stderr[-2000:]!r}"
    )
    assert proc.returncode == 0, (
        f"[{mode}] teardown-during-download subprocess exited rc="
        f"{proc.returncode} (3 = worker not parked; anything else = crash "
        f"during the sequence).\nstdout: {proc.stdout!r}\n"
        f"stderr: {proc.stderr[-2000:]!r}"
    )
