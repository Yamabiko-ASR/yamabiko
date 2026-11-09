"""Settings-owned model acquisition for a quant-tier switch.

When the user changes ``translation.quant_tier`` in Settings, the target
tier's translator GGUF must actually be present before the switch is persisted
(the runtime resolver does a cache lookup only and silently falls back to
whichever tier IS cached -- so persisting a tier whose file is missing is
a promise the runtime cannot keep). This module supplies:

* :func:`spec_for_tier` / :func:`tier_cached` -- the tier -> filename map
  and the local HF cache probe, BOTH sourced from ``yamabiko.models`` so
  the filename literals and the probe cannot drift from the wizard's
  download queue or the runtime resolver.
* :class:`QuantDownloadDialog` -- the compact modal download dialog
  (progress, time left, waiting and failed states with Retry).

The download itself is NOT re-implemented here. The dialog drives the
wizard's own :class:`~yamabiko.wizard_pages.downloads._DownloadWorker`
(a plain ``QObject``, not entangled with ``QWizard``) on a ``QThread``,
through the same pinned HF cache (``yamabiko.hf_env``) the wizard and the
runtime use. One downloader, two callers. The overlay package already
imports ``yamabiko.wizard_pages`` (see ``settings_window._default_list_devices``),
so this import direction introduces no new cycle.

Quit-safety mirrors the palette / gazetteer / diagnostics / update
workers: a download blocked inside an uninterruptible ``hf_hub_download``
is detached into :mod:`yamabiko.ui.thread_keeper` rather than destroyed
with its owning widget (Qt 6 qFatal "QThread: Destroyed while thread is
still running").
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton

# Reuse the wizard's download machinery rather than forking a second HF
# downloader. These are plain, QWizard-free building blocks.
from ..wizard_pages.downloads import (
    DownloadSpec,
    _DownloadWorker,
    _friendly_error,
    _sweep_incomplete_blobs,
)
# Single source of truth for the per-tier translator pins (repo + filename
# + revision) and the local cache probe -- never duplicate the GGUF
# filename literals here.
from ..models import (
    translator_pin,
    _repo_files_cached,
)
from ..ui import thread_keeper
from .controls import LinkButton, ProgressLine
from .settings_dialogs import display_path
from .theme import MIST
from .window_chrome import CompactDialog

_log = logging.getLogger(__name__)

# Display name, approximate download size in GB and the card it suits, per
# tier: the Settings model rows ("For 8 GB cards, 4 GB download") and the
# download dialog ("Downloading the Smaller model", "0 of 4 GB"). Sizes
# match the models.py / gpu.py documentation (default Qwen3.5 UD-Q3_K_XL
# ~5.1 GB; small Qwen3.5 UD-IQ3_XXS ~4 GB).
TIER_INFO = {
    "default": ("Standard", 5.1, "For 10 GB+ cards"),
    "small": ("Smaller", 4.0, "For 8 GB cards"),
}


def _trim(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def size_text(gb: float) -> str:
    """5.1 -> "5.1 GB"; 4.0 -> "4 GB"."""
    return f"{_trim(gb)} GB"


def format_amount(done: int, total: int) -> str:
    """Decimal GB with one decimal: "1.7 of 4 GB"."""
    return f"{_trim(done / 1e9)} of {_trim(total / 1e9)} GB"


def format_time_left(seconds: float) -> str:
    """"Less than a minute left", "About a minute left", "About 3 minutes
    left", "About an hour left", "About 2 hours left"."""
    if seconds < 60:
        return "Less than a minute left"
    minutes = round(seconds / 60)
    if minutes <= 1:
        return "About a minute left"
    if minutes < 90:
        return f"About {minutes} minutes left"
    hours = round(seconds / 3600)
    return "About an hour left" if hours <= 1 else f"About {hours} hours left"


class _TimeLeft:
    """Time left from progress byte counts: the average rate since the first
    sample, smoothed, once at least two seconds of progress are in."""

    MIN_SPAN_S = 2.0

    def __init__(self, clock) -> None:
        self._clock = clock
        self._start: Optional[tuple[float, int]] = None
        self._rate: Optional[float] = None

    def reset(self) -> None:
        self._start = None
        self._rate = None

    def update(self, done: int, total: int) -> Optional[float]:
        now = self._clock()
        if self._start is None:
            self._start = (now, done)
            return None
        t0, d0 = self._start
        span = now - t0
        if span < self.MIN_SPAN_S or done <= d0:
            return None
        rate = (done - d0) / span
        self._rate = rate if self._rate is None else 0.7 * self._rate + 0.3 * rate
        return max(0.0, (total - done) / self._rate)


# How long to wait for the worker thread to join before detaching it into
# thread_keeper. On a normal finish the worker's run() has already returned,
# so the join is immediate; a user cancel may leave the worker blocked inside
# an uninterruptible hf_hub_download, so that path times out fast and detaches.
_FINISH_JOIN_MS = 2000
_CANCEL_JOIN_MS = 100


class _DownloadThread(QThread):
    """Runs a reused ``_DownloadWorker``'s queue on its own OS thread.

    The worker is created on (and keeps its affinity to) the MAIN thread --
    we deliberately do NOT ``moveToThread`` it. ``run()`` just invokes
    ``worker.run()`` on this thread; the worker's ``progress`` /
    ``queue_finished`` signals still cross back to the main-thread dialog slots
    as queued connections (the emit happens on this thread, the receiver lives
    on the main thread). Keeping the worker main-thread-affine means dropping
    its last reference on the main thread is a safe destruction -- a
    ``moveToThread`` worker garbage-collected from the main thread after its
    thread has finished is a PySide cross-thread-destruction hazard (a native
    access violation that surfaces at a later Qt event-processing turn). Mirrors
    the QThread-subclass idiom of ``tray._DiagnosticsThread`` /
    ``_GazetteerBuildThread``.
    """

    def __init__(self, worker: _DownloadWorker, parent=None) -> None:
        super().__init__(parent)
        self._worker = worker

    def run(self) -> None:  # worker thread
        self._worker.run()


def spec_for_tier(tier: str) -> DownloadSpec:
    """The single-file download spec for ``tier`` (that tier's own repo +
    pinned revision).

    Pinned to the tier's revision to match the wizard's translator download
    and the runtime resolver, which both resolve the tier at that revision
    (see models._translator_gguf_cached and
    watchdog._resolve_translator_model_path). A sha-pinned download writes
    only snapshots/<sha>/ (no refs/main), so a Settings-triggered
    re-download must land the SAME snapshot the resolver probes -- an
    unpinned download here would drift from the pinned resolver.
    """
    pin = translator_pin(tier)
    return DownloadSpec(pin.repo, pin.filename, revision=pin.revision)


def tier_cached(tier: str) -> bool:
    """True iff ``tier``'s GGUF is already in the local HF cache.

    Delegates to ``models._repo_files_cached`` (the ``try_to_load_from_cache``
    probe the wizard gate and runtime resolver use) -- not a second probe.

    Fails CLOSED: a probe error (hub import failure, cache-layout error) is
    treated as "not cached" so the download path runs and the model is
    actually fetched. A false "not cached" costs only a fast re-probe -- the
    download no-ops instantly when the file IS present (HF ETag check) --
    whereas a false "cached" would persist a tier whose GGUF may be missing.
    """
    try:
        pin = translator_pin(tier)
        return _repo_files_cached(pin.repo, (pin.filename,), pin.revision)
    except Exception:
        _log.warning(
            "quant-tier cache probe failed for %r; treating as not cached",
            tier, exc_info=True,
        )
        return False


class QuantDownloadDialog(CompactDialog):
    """Modal compact dialog that pulls one quant tier's GGUF.

    "Downloading the Smaller model", "It takes effect the next time Yamabiko
    starts.", the amount and time left over a progress line, Cancel. While
    an earlier download still drains it says "Waiting for the other download
    to finish…". A failure keeps the dialog open: the bar stops, "The
    download stopped." / "Check your internet connection, then click
    Retry.", Copy details and Retry. Drives the wizard's ``_DownloadWorker``
    on a ``QThread``. Outcomes, readable once ``exec`` returns:

    * ``"success"``  -- the file landed; :meth:`accept` was called.
    * ``"cancelled"`` -- the user canceled (or closed the dialog); no error.
    * ``"failed"``   -- the download failed and the user closed the dialog
      from the failed state; ``self.error`` holds the raw exception and
      :meth:`friendly_error` maps it to a human message.

    ``downloader`` is injected in tests (a fake at the ``_Downloader`` seam);
    ``None`` uses the production ``_HFDownloader`` via the worker default.
    ``clock`` feeds the time-left estimate (tests pass a fake).
    """

    def __init__(self, spec: DownloadSpec, tier: str, *, downloader=None, parent=None,
                 palette: Optional[dict] = None, clock=time.monotonic) -> None:
        palette = palette if palette is not None else MIST
        super().__init__(palette, 440, parent)
        self._spec = spec
        self._tier = tier
        self._downloader = downloader
        # Default outcome is "cancelled" so a dialog closed by any path other
        # than an explicit success/failure (X button, external close) reads as
        # a cancel -- the safe non-persisting outcome.
        self.outcome = "cancelled"
        self.error: Optional[Exception] = None
        # Set once the outcome is decided; guards the accept/reject re-entry
        # and a late queue_finished arriving after a user cancel.
        self._settled = False
        # True while the dialog shows a failure (Retry offered): a close from
        # there keeps the "failed" outcome.
        self._failed = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[_DownloadWorker] = None
        self._time_left = _TimeLeft(clock)

        name, size_gb, _cards = TIER_INFO.get(tier, TIER_INFO["default"])
        self._total_hint = int(size_gb * 1e9)
        self.heading(f"Downloading the {name} model")
        self.add_text("It takes effect the next time Yamabiko starts.", muted=True)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.amount = QLabel(format_amount(0, self._total_hint))
        self.time_left = QLabel("")
        self.time_left.setObjectName("muted")
        row.addWidget(self.amount)
        row.addStretch(1)
        row.addWidget(self.time_left)
        self.body.addSpacing(16)
        self.body.addLayout(row)
        self.progress = ProgressLine(palette)
        self.body.addSpacing(7)
        self.body.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setContentsMargins(0, 8, 0, 0)
        self.status.hide()
        self.body.addWidget(self.status)

        self.copy_button = LinkButton("Copy details", palette, quiet=True, height=24)
        self.copy_button.clicked.connect(self._copy_details)
        self.copy_button.hide()
        self.cancel_button = QPushButton("Cancel")
        # Cancel routes through reject() so the X button and the Cancel button
        # share one teardown path (thread detach + outcome).
        self.cancel_button.clicked.connect(self.reject)
        self.retry_button = QPushButton("Retry")
        self.retry_button.setProperty("primary", True)
        self.retry_button.clicked.connect(self._retry)
        self.retry_button.hide()
        self.actions.addWidget(self.copy_button)
        self.actions.addStretch(1)
        self.actions.addWidget(self.cancel_button)
        self.actions.addWidget(self.retry_button)

    def _show_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))
        if self.isVisible():
            self.refit_height()

    def start(self) -> None:
        """Spawn the worker + thread and begin the download.

        The worker stays on the main thread (no ``moveToThread``); the
        ``_DownloadThread`` subclass runs its queue on a separate OS thread and
        its signals reach these slots as queued cross-thread deliveries. See
        ``_DownloadThread`` for why keeping the worker main-thread-affine is
        the safe destruction contract.
        """
        # A previously canceled download may still be draining inside an
        # uninterruptible hf_hub_download (parked in thread_keeper); HF's
        # per-blob lock then serializes this fresh worker behind that drain.
        # Say so -- the first progress tick clears it.
        if thread_keeper.any_running():
            self._show_status("Waiting for the other download to finish…")
        # Partial blobs left by an earlier canceled tier download are never
        # resumed; drop them before this one starts (skipped while a
        # previous download still drains, see _sweep_incomplete_blobs).
        # The pinned-digest check on the landed GGUF lives in the shared
        # _HFDownloader the worker uses by default.
        _sweep_incomplete_blobs([self._spec.repo_id])
        self._worker = _DownloadWorker([self._spec], downloader=self._downloader)
        self._thread = _DownloadThread(self._worker, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.queue_finished.connect(self._on_queue_finished)
        self._thread.start()

    def exec(self) -> int:  # noqa: A003 - Qt override name
        # Auto-start on first exec so callers need only exec() the dialog; a
        # test that called start() itself (to poke at the in-flight worker)
        # already has a thread, so don't restart it.
        if self._thread is None:
            self.start()
        return super().exec()

    def friendly_error(self) -> Optional[str]:
        """Human-readable message for a failed download, or None."""
        if self.error is None:
            return None
        return _friendly_error(self.error)

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            return
        # The xet progress signal can over-count; clamp so the readout never
        # says "4.4 of 4 GB".
        done = max(0, min(done, total))
        if self.status.isVisible() and not self._failed:
            self._show_status("")
        self.amount.setText(format_amount(done, total))
        self.progress.set_fraction(done / total)
        left = self._time_left.update(done, total)
        self.time_left.setText(format_time_left(left) if left is not None else "")

    def _on_queue_finished(self, ok: bool) -> None:
        # A user cancel may already have settled the outcome and detached the
        # worker; a queue_finished still queued behind it must be ignored.
        if self._settled:
            return
        error = self._worker.error if self._worker is not None else None
        if ok:
            self._settled = True
            self.outcome = "success"
            self._release_running_thread(wait_ms=_FINISH_JOIN_MS)
            self.accept()
            return
        if error is None:
            # queue_finished(False) with no error == a cancel the worker
            # detected itself (between files / mid-file abort).
            self._settled = True
            self.outcome = "cancelled"
            self._release_running_thread(wait_ms=_FINISH_JOIN_MS)
            # _settled is already True, so reject() skips a second release.
            self.reject()
            return
        # Failed: keep the dialog open with Retry (read the error BEFORE
        # releasing; the release nulls self._worker).
        self.error = error
        self.outcome = "failed"
        self._failed = True
        self._release_running_thread(wait_ms=_FINISH_JOIN_MS)
        self.progress.set_stopped(True)
        self.time_left.setText("")
        self.copy_button.show()
        self.retry_button.show()
        self.retry_button.setDefault(True)
        self._show_status("The download stopped.\nCheck your internet connection, then click Retry.")

    def _retry(self) -> None:
        self.error = None
        self.outcome = "cancelled"
        self._failed = False
        self.progress.set_stopped(False)
        self.progress.set_fraction(0.0)
        self.amount.setText(format_amount(0, self._total_hint))
        self.copy_button.hide()
        self.retry_button.hide()
        self._show_status("")
        self._time_left.reset()
        self.start()

    def _copy_details(self) -> None:
        if self.error is None:
            return
        QGuiApplication.clipboard().setText(display_path(
            f"{self.friendly_error()}\n{type(self.error).__name__}: {self.error}"))

    def reject(self) -> None:  # noqa: N802 - Qt override
        # User Cancel / X button: settle (as cancelled, or failed when the
        # failure is on screen) and detach the (possibly blocked) worker so
        # widget teardown can't destroy a running QThread. Guarded so a
        # reject() that follows an already-settled finish does not re-run the
        # release.
        if not self._settled:
            self._settled = True
            if not self._failed:
                self.outcome = "cancelled"
            self._release_running_thread(wait_ms=_CANCEL_JOIN_MS)
        super().reject()

    def release_threads(self) -> None:
        """Quit-time detach: park an in-flight worker in ``thread_keeper``.

        Reached from ``Tray.release_threads`` -> ``SettingsWindow.release_threads``
        at ``app.aboutToQuit`` (same pattern as the palette / gazetteer /
        diagnostics / update workers). Detaches without waiting so a download
        blocked inside an uninterruptible ``hf_hub_download`` cannot stall the
        quit; ``run_overlay``'s ``thread_keeper.any_running()`` guard then
        hard-exits past interpreter finalization. Marks the dialog settled so
        no accept/finish callback fires after the detach.
        """
        if self._settled:
            return
        self._settled = True
        self._release_running_thread(wait_ms=0)

    def _release_running_thread(self, wait_ms: int) -> None:
        """Stop-or-detach the worker thread, then null the refs.

        Cancels the worker, quits the thread's event loop, and waits up to
        ``wait_ms``. A thread still running after that is blocked inside an
        uninterruptible ``hf_hub_download`` and is detached into
        ``thread_keeper`` (destroying a running QThread is a Qt 6 qFatal
        abort). The detached worker's signals are disconnected first so the
        doomed queue can't keep driving this dialog's widgets. Mirrors
        ``DownloadsPage._release_running_thread``.
        """
        thread, worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        if thread is None:
            return
        if worker is not None:
            worker.cancel()
        thread.quit()
        if thread.wait(wait_ms):
            return
        if worker is not None:
            for signal, slot in (
                (worker.progress, self._on_progress),
                (worker.queue_finished, self._on_queue_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        thread_keeper.release(thread, worker)
