"""DownloadsPage — sequential model downloads with byte-level progress.

No ``local_dir``, no subdir-flatten hack, no custom retry loop, no own
skip-if-exists check. huggingface_hub handles cache / resume / integrity /
retry; we handle Qt threading, queue ordering, progress signals, and the
step's progress and its alert (Retry, Delete old Yamabiko models, Copy details).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

pytest.importorskip("pytestqt")

from yamabiko.wizard_pages import downloads as downloads_mod
from yamabiko.wizard_pages.downloads import (
    DownloadSpec, DownloadsPage, _DownloadCancelledError, _DownloadWorker,
    _GIB, _HFDownloader, _disk_full_message, _estimated_spec_bytes,
    _friendly_error, _progress_percent, _space_check, _space_shortfall,
)

_MB = 1024 * 1024


@pytest.fixture
def fake_downloader():
    """MagicMock that mimics a successful downloader — returns a plausible Path."""
    mock = MagicMock()

    def _ok(repo_id, filename, revision=None, progress_callback=None):
        return Path(f"/fake/cache/{repo_id}/{filename}")

    mock.download.side_effect = _ok
    return mock


_DEFAULT_GGUF = "Qwen3.5-9B-UD-Q3_K_XL.gguf"


def _default_tier(monkeypatch) -> None:
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": "default"}},
    )


def _drive() -> str:
    """The drive letter the disk-full sentence names for this HF_HOME."""
    import os

    return Path(os.environ["HF_HOME"]).resolve().drive.upper()


class _TickingDownloader:
    """Every file lands at once (a cache hit: no progress) except
    ``filename``, which reports ``ticks`` against its pinned size, then
    runs ``before_error`` and raises ``error``, or blocks on ``gate``."""

    def __init__(self, filename, ticks, *, error=None, before_error=None):
        import threading

        self.filename = filename
        self.ticks = ticks
        self.error = error
        self.before_error = before_error
        self.gate = threading.Event()

    def download(self, repo_id, filename, revision=None, progress_callback=None):
        from yamabiko import models

        if filename == self.filename:
            total = models.pinned_size(repo_id, filename, revision)
            for done in self.ticks:
                progress_callback(done, total)
            if self.error is not None:
                if self.before_error is not None:
                    self.before_error()
                raise self.error
            self.gate.wait(timeout=10.0)
        return Path(f"/fake/{repo_id}/{filename}")


def _rail_record(page) -> list:
    """Every (fraction, stopped) the page sends to the rail line."""
    emitted: list = []
    page.railProgressChanged.connect(
        lambda fraction, stopped: emitted.append((fraction, stopped)),
    )
    return emitted


def _mutable_disk(monkeypatch, free_bytes: int) -> dict:
    """A cache volume whose free bytes the test can change mid-run."""
    import types

    state = {"free": free_bytes}
    monkeypatch.setattr(
        downloads_mod, "shutil",
        types.SimpleNamespace(disk_usage=lambda path: types.SimpleNamespace(
            total=2**40, used=0, free=state["free"],
        )),
    )
    return state


# ---------------------------------------------------------------------------
# Queue construction
# ---------------------------------------------------------------------------

def test_queue_order_is_whisper_translator_silero():
    """L5: the rows fill top to bottom: the five whisper files (Speech
    recognition) first, then the translator GGUF, then silero (Voice
    detection).

    Seeded defect: restore the translator-first order -> red.
    """
    from yamabiko import models

    specs = downloads_mod.download_queue()
    assert [s.filename for s in specs[:5]] == list(models.WHISPER_FILES)
    assert {s.repo_id for s in specs[:5]} == {models.WHISPER_REPO}
    assert specs[5].filename.endswith(".gguf")
    assert specs[6].repo_id == models.SILERO_REPO
    assert len(specs) == 7


def test_queue_pins_silero_revision_and_models_constants():
    """The queue's silero spec must carry the PINNED revision from
    yamabiko.models -- an unpinned spec floats with upstream `main`, so the
    wizard could warm a different snapshot than the runtime resolves offline.
    Whisper specs must likewise come from the models constants (single
    source of truth to prevent repo-URL / filename drift).

    Seeded-defect drill: drop ``revision=`` from the silero
    ``DownloadSpec`` in ``download_queue`` -> revision is None -> red.
    """
    from yamabiko import models

    specs = downloads_mod.download_queue()

    [silero] = [s for s in specs if s.repo_id == models.SILERO_REPO]
    assert silero.filename == models.SILERO_FILE
    assert silero.revision == models.SILERO_REVISION, (
        "the wizard must download the exact snapshot the runtime "
        "resolves offline (vad/silero_chunker.py)"
    )

    whisper_files = {s.filename for s in specs
                     if s.repo_id == models.WHISPER_REPO}
    assert whisper_files == set(models.WHISPER_FILES)


def test_queue_pins_translator_revision():
    """The queue's translator spec must carry the PINNED revision from
    yamabiko.models -- an unpinned spec floats with upstream `main`, so the
    wizard could warm a different snapshot than the runtime resolves offline
    (watchdog._resolve_translator_model_path probes at the tier pin's
    revision, and a sha-pinned download writes only snapshots/<sha>/ with no
    refs/main, so a mismatched or absent revision makes the runtime cache
    probe miss).

    Seeded-defect drill: drop ``revision=`` from the translator
    ``DownloadSpec`` in ``download_queue`` -> revision is None -> red.
    """
    from yamabiko import models

    specs = downloads_mod.download_queue()

    pin = models.TRANSLATOR_PINS["default"]
    [translator] = [s for s in specs if s.repo_id == pin.repo]
    assert translator.revision == pin.revision, (
        "the wizard must download the exact translator snapshot the runtime "
        "resolves offline (watchdog._resolve_translator_model_path)"
    )


def test_quant_tier_default_fetches_ud_q3_k_xl(monkeypatch):
    """Unset / `default` quant_tier keeps the UD-Q3_K_XL filename.

    Filenames are deliberate literals: a pin change in models.py must go
    red HERE so the shipped artifact swap is a conscious, tested act."""
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": "default"}},
    )
    specs = downloads_mod.download_queue()
    tr_specs = [s for s in specs if "Qwen3.5" in s.repo_id]
    assert len(tr_specs) == 1
    assert tr_specs[0].filename == "Qwen3.5-9B-UD-Q3_K_XL.gguf"


def test_quant_tier_small_fetches_ud_iq3_xxs(monkeypatch):
    """`quant_tier: small` swaps in the UD-IQ3_XXS file (~4 GB vs ~5.1 GB)."""
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": "small"}},
    )
    specs = downloads_mod.download_queue()
    tr_specs = [s for s in specs if "Qwen3.5" in s.repo_id]
    assert len(tr_specs) == 1
    assert tr_specs[0].filename == "Qwen3.5-9B-UD-IQ3_XXS.gguf"


def test_quant_tier_unknown_value_falls_back_to_default(monkeypatch, caplog):
    """An unknown quant_tier warns and uses the default filename."""
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": "medium-rare"}},
    )
    with caplog.at_level("WARNING"):
        specs = downloads_mod.download_queue()
    tr_specs = [s for s in specs if "Qwen3.5" in s.repo_id]
    assert tr_specs[0].filename == "Qwen3.5-9B-UD-Q3_K_XL.gguf"
    assert any("medium-rare" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Worker behaviour
# ---------------------------------------------------------------------------

def test_worker_calls_downloader_per_spec(fake_downloader):
    """Each spec in the queue triggers one downloader.download call."""
    specs = [
        DownloadSpec(repo_id="r1", filename="f1"),
        DownloadSpec(repo_id="r2", filename="f2", revision="v1"),
    ]
    worker = _DownloadWorker(specs, downloader=fake_downloader)
    worker.run()
    assert fake_downloader.download.call_count == 2
    fake_downloader.download.assert_any_call(
        "r1", "f1", revision=None, progress_callback=ANY,
    )
    fake_downloader.download.assert_any_call(
        "r2", "f2", revision="v1", progress_callback=ANY,
    )


def test_worker_aborts_on_first_download_failure():
    """HF's internal retry has already exhausted — worker surfaces the error
    and stops the queue. The UI Retry button gives the user a fresh run.
    """
    fake = MagicMock()
    fake.download.side_effect = RuntimeError("network")
    specs = [
        DownloadSpec(repo_id="r1", filename="f1"),
        DownloadSpec(repo_id="r2", filename="f2"),
    ]
    worker = _DownloadWorker(specs, downloader=fake)
    worker.run()
    # Only the first file was attempted; the error is preserved for the UI.
    assert fake.download.call_count == 1
    assert worker.error is not None
    assert worker.error_spec is specs[0]


def test_worker_emits_per_file_finished_signal(fake_downloader, qtbot):
    spec = DownloadSpec(repo_id="r", filename="f")
    worker = _DownloadWorker([spec], downloader=fake_downloader)
    with qtbot.waitSignal(worker.file_finished, timeout=2000) as blocker:
        worker.run()
    assert blocker.args == [0, "f"]


def test_worker_respects_cancel_flag_between_files():
    """Setting _cancelled before run() stops the queue before any download fires."""
    fake = MagicMock()
    specs = [
        DownloadSpec(repo_id="r1", filename="f1"),
        DownloadSpec(repo_id="r2", filename="f2"),
    ]
    worker = _DownloadWorker(specs, downloader=fake)
    worker._cancelled = True
    results: list = []
    worker.queue_finished.connect(results.append)
    worker.run()
    assert fake.download.call_count == 0
    assert results == [False]


def test_worker_cancel_mid_queue_stops_after_current_file():
    """Cancel raised between files — first file runs, second is skipped."""
    calls: list = []

    def record_and_cancel(repo_id, filename, revision=None, progress_callback=None):
        calls.append(filename)
        worker._cancelled = True  # user hits Back mid-queue
        return Path("/fake/path")

    fake = MagicMock()
    fake.download.side_effect = record_and_cancel
    specs = [
        DownloadSpec(repo_id="r1", filename="f1"),
        DownloadSpec(repo_id="r2", filename="f2"),
    ]
    worker = _DownloadWorker(specs, downloader=fake)
    results: list = []
    worker.queue_finished.connect(results.append)
    worker.run()
    assert calls == ["f1"]
    assert results == [False]


def test_progress_callback_emits_during_download(qtbot):
    """A downloader that invokes progress_callback triggers the worker's progress signal."""

    class _FakeProgressDownloader:
        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            if progress_callback is not None:
                progress_callback(500, 1000)
                progress_callback(1000, 1000)
            return Path(f"/fake/{filename}")

    spec = DownloadSpec(repo_id="r", filename="f")
    worker = _DownloadWorker([spec], downloader=_FakeProgressDownloader())
    received: list = []
    worker.progress.connect(lambda done, total: received.append((done, total)))
    with qtbot.waitSignal(worker.queue_finished, timeout=2000):
        worker.run()
    assert (500, 1000) in received
    assert (1000, 1000) in received


def test_progress_signal_carries_byte_counts_past_the_32_bit_boundary(qtbot):
    """The shipped default-tier GGUF is ~5.1 GiB, so bytes_done/bytes_total
    cross 2**31 on every real download of it. The progress signal must deliver
    those counts intact and the download must complete: declared as C++ int
    (4 bytes), the emit itself dies in libshiboken (Overflow RuntimeWarning +
    OverflowError) once a count exceeds 2**31 -- the 2026-07-11 CI run hit
    exactly this ~2.2 GiB into the file, and a real user pulling the standard
    tier through the wizard or the settings quant switch crosses the same
    boundary mid-download.

    Seeded defect: declare the signal ``Signal(int, int)`` again -> the
    >2**31 emits below error instead of delivering -> both asserts red.
    """
    done_a, total = 3 * 2**30, 5_500_000_000  # ~3.0 of ~5.1 GiB: past 2**31
    done_b = 5_400_000_000

    class _FiveGibDownloader:
        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            progress_callback(done_a, total)
            progress_callback(done_b, total)
            return Path(f"/fake/{filename}")

    spec = DownloadSpec(repo_id="r", filename="f")
    worker = _DownloadWorker([spec], downloader=_FiveGibDownloader())
    received: list = []
    worker.progress.connect(lambda done, tot: received.append((done, tot)))
    finished: list = []
    worker.queue_finished.connect(lambda ok: finished.append(ok))
    worker.run()
    assert received == [(done_a, total), (done_b, total)], (
        f"byte counts past 2**31 must arrive intact; got {received}"
    )
    assert finished == [True], (
        f"a ~5 GiB download must complete, not abort on a progress tick: "
        f"{finished}"
    )


# ---------------------------------------------------------------------------
# Page-level behaviour
# ---------------------------------------------------------------------------

def test_initialize_page_does_not_spawn_duplicate_thread(qtbot):
    """Calling initializePage while a thread is running is a no-op."""
    import threading
    gate = threading.Event()

    class BlockingDownloader:
        def download(self, *args, **kwargs):
            gate.wait(timeout=5.0)
            return Path("/fake/path")

    page = DownloadsPage(        downloader=BlockingDownloader(),
    )
    qtbot.addWidget(page)
    page.initializePage()
    first_thread = page._thread
    assert first_thread is not None
    assert first_thread.isRunning()
    # Second call while the first thread is still running must be a no-op.
    page.initializePage()
    assert page._thread is first_thread
    # Let the queue finish on its own (leaving the step no longer cancels).
    gate.set()
    qtbot.waitUntil(page.isComplete, timeout=5000)


def test_other_failure_shows_the_download_stopped_alert(qtbot):
    """L6: any failure but a full disk: "The download stopped." / "Check
    your internet connection, then click Retry.", Retry as the primary,
    Copy details, no Delete; the running row paused, "Paused", the rail line
    grey.

    Seeded defect: show the disk-full sentence for every failure -> red.
    """
    fake = MagicMock()
    fake.download.side_effect = RuntimeError("Connection reset by peer")

    page = DownloadsPage(downloader=fake)
    qtbot.addWidget(page)
    rail = _rail_record(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed, timeout=5000)

    assert not page.alert_box.isHidden()
    assert page.alert.message() == "The download stopped."
    assert page.alert.help_text() == "Check your internet connection, then click Retry."
    assert page.retry_button.property("primary") is True
    assert page.delete_button.isHidden()
    assert page.copy_button.isVisibleTo(page)  # inside the shown alert (Decision 148)
    assert page.rows.row_state(0) == "paused"
    assert page.time_left.text() == "Paused"
    assert rail[-1] == (0.0, True)
    assert not page.isComplete()


def test_retry_button_re_triggers_download(qtbot):
    """Clicking Retry re-launches downloads — fresh queue run."""
    call_count = 0

    def fail_then_succeed(repo_id, filename, revision=None, progress_callback=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first attempt fails")
        return Path(f"/fake/{filename}")

    fake = MagicMock()
    fake.download.side_effect = fail_then_succeed

    page = DownloadsPage(        downloader=fake,
    )
    qtbot.addWidget(page)
    page.initializePage()

    qtbot.waitUntil(lambda: page._completed, timeout=5000)
    assert page.retry_button.isVisibleTo(page)
    assert not page.isComplete()

    page.retry_button.click()
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)
    assert page.isComplete()


def test_friendly_error_401():
    exc = RuntimeError("401 Client Error. (Request ID: Root=1-69e04462-...)")
    msg = _friendly_error(exc)
    assert "not accessible" in msg
    # Raw HTTP status code should not leak through to users.
    assert "Client Error" not in msg


def test_friendly_error_enospc_names_the_incomplete_file_in_hf_cache():
    """The ENOSPC copy must tell the user that the partial *.incomplete blob
    in the hf-cache folder counts toward the space to clear -- after a
    mid-GGUF ENOSPC there are multiple GB of it, and 'free up space' alone
    sends the user hunting everywhere but the one folder that grew.

    Seeded defect: revert the ENOSPC branch to the bare 'Not enough disk
    space. Free up space and try again.' copy -> both mentions gone -> red.
    """
    msg = _friendly_error(OSError(28, "No space left on device"))
    assert "Not enough disk space" in msg
    assert ".incomplete" in msg
    assert "hf-cache" in msg


def test_friendly_error_timeout():
    exc = ConnectionError("Connection timed out after 30s")
    msg = _friendly_error(exc)
    assert "timed out" in msg.lower() or "connection" in msg.lower()
    assert "Check your internet" in msg


def test_copy_details_copies_a_redacted_report_through_the_flushing_copy(qtbot, monkeypatch):
    """L6: Copy details puts the failure report on the clipboard through
    error_window.copy_to_clipboard (it flushes, so the text outlives the
    wizard's os._exit on Cancel) with the user's folders redacted.

    Seeded defects: copy with QClipboard.setText -> the flushing copy is
    never called -> red; skip redact_user -> the LOCALAPPDATA path reaches
    the clipboard -> red.
    """
    import os

    from yamabiko.ui import error_window

    copied: list = []
    monkeypatch.setattr(
        error_window, "copy_to_clipboard", lambda text: copied.append(text) or True,
    )
    local = os.environ["LOCALAPPDATA"]
    fake = MagicMock()
    fake.download.side_effect = RuntimeError(
        f"could not write {local}\\Yamabiko\\hf-cache\\blob",
    )

    page = DownloadsPage(downloader=fake)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed, timeout=5000)
    page.copy_button.click()

    assert len(copied) == 1
    assert local.lower() not in copied[0].lower()
    assert "%LOCALAPPDATA%\\Yamabiko\\hf-cache" in copied[0]
    assert "Yamabiko v" in copied[0]


# ---------------------------------------------------------------------------
# Cancel / re-entry behaviour
# ---------------------------------------------------------------------------

def test_worker_cancel_mid_file_surfaces_as_cancel_not_failure():
    """cancel() mid-file makes the worker's progress callback raise (aborting
    the blocking download on the HTTP path) and run() must classify that as a
    cancel -- queue_finished(False) with error=None -- not a failure.

    Seeded defect: drop the `self._cancelled` re-check in run()'s except
    handler → the abort is recorded as worker.error → red.
    """
    worker_box: list = []

    class MidFileCancelDownloader:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            worker_box[0].cancel()          # user closes the wizard mid-file
            progress_callback(1024, 4096)   # next chunk arrives → must raise
            raise AssertionError("progress_callback should have raised on cancel")

    specs = [
        DownloadSpec(repo_id="r1", filename="f1"),
        DownloadSpec(repo_id="r2", filename="f2"),
    ]
    worker = _DownloadWorker(specs, downloader=MidFileCancelDownloader())
    worker_box.append(worker)
    results: list = []
    worker.queue_finished.connect(results.append)
    worker.run()
    assert results == [False]
    assert worker.error is None, (
        f"mid-file cancel misclassified as failure: {worker.error!r}"
    )


def test_back_then_tier_change_redownloads_new_tier_gguf(monkeypatch, tmp_path, qtbot):
    """After a successful run, re-entering the page with a changed quant tier
    must rebuild the queue and download the new tier's GGUF.

    Seeded defect: restore the bare `if self._completed and self._success:
    return` guard in initializePage → the Q3_K_M file is never requested → red.
    """
    tier = {"value": "default"}
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": tier["value"]}},
    )
    downloaded: list = []

    class Recorder:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            downloaded.append(filename)
            return Path(f"/fake/{filename}")

    page = DownloadsPage(downloader=Recorder())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)
    assert "Qwen3.5-9B-UD-Q3_K_XL.gguf" in downloaded

    tier["value"] = "small"   # user hit Back and flipped the radio
    page.initializePage()     # forward onto the page again
    qtbot.waitUntil(
        lambda: "Qwen3.5-9B-UD-IQ3_XXS.gguf" in downloaded, timeout=5000,
    )
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)


def test_reentry_with_unchanged_queue_does_not_redownload(monkeypatch, tmp_path, qtbot):
    """Re-entry after success with an identical queue stays a no-op -- an
    unchanged queue must not hit the network again.

    Seeded defect: drop the spec comparison (always restart on re-entry) →
    the download call count doubles → red.
    """
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": "default"}},
    )
    downloaded: list = []

    class Recorder:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            downloaded.append(filename)
            return Path(f"/fake/{filename}")

    page = DownloadsPage(downloader=Recorder())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)
    first_run_count = len(downloaded)

    page.initializePage()
    qtbot.wait(200)  # nothing async should start; give it a beat to prove it
    assert len(downloaded) == first_run_count
    assert page.isComplete()


def test_reentry_with_a_changed_tier_restarts_a_running_queue(monkeypatch, qtbot):
    """L3: coming back to Download while the queue runs: the same queue is
    left alone; a tier changed on Quality restarts it (the old worker is
    detached) and the new tier's GGUF is fetched.

    Seeded defect: restore the "a thread is running -> return" guard in
    initializePage -> the small GGUF is never requested -> red.
    """
    import threading

    from yamabiko.ui import thread_keeper

    tier = {"value": "default"}
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": tier["value"]}},
    )
    gate = threading.Event()
    entered = threading.Event()
    requested: list = []

    class BlockFirstCall:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            requested.append(filename)
            if len(requested) == 1:
                entered.set()
                gate.wait(timeout=10.0)
            return Path(f"/fake/{filename}")

    page = DownloadsPage(downloader=BlockFirstCall())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    old_thread = page._thread
    try:
        page.initializePage()  # back on the step, same tier: no-op
        assert page._thread is old_thread

        tier["value"] = "small"  # the user changed the tier on Quality
        page.initializePage()
        assert page._thread is not old_thread
        assert any(t is old_thread for t, _ in thread_keeper._LINGERING)
        qtbot.waitUntil(
            lambda: "Qwen3.5-9B-UD-IQ3_XXS.gguf" in requested, timeout=5000,
        )
    finally:
        gate.set()
        old_thread.wait(5000)
        thread_keeper.discard(old_thread)
    qtbot.waitUntil(page.isComplete, timeout=5000)


def test_leaving_the_step_does_not_cancel_the_download(qtbot):
    """L3: Back or a rail jump (cleanupPage) leaves the download running;
    it completes on its own.

    Seeded defect: keep the cancel in cleanupPage -> the worker's cancel
    flag is set -> red.
    """
    import threading

    gate = threading.Event()
    entered = threading.Event()

    class Blocking:
        def download(self, *args, **kwargs):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    page = DownloadsPage(downloader=Blocking())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    worker, thread = page._worker, page._thread

    page.cleanupPage()

    assert worker._cancelled is False
    assert page._thread is thread and thread.isRunning()
    gate.set()
    qtbot.waitUntil(page.isComplete, timeout=5000)


def test_a_stale_finish_from_the_replaced_queue_leaves_the_new_queue_alone(monkeypatch, qtbot):
    """Decision 130 (R1): a tier change restarts a running queue; a
    queue_finished the old worker queued just before the restart is still
    delivered after the disconnect. It must not mark the new queue done
    (nor join the new queue's thread on the GUI thread), and the old
    queue's late progress and file_finished must not reach the new rows.

    Seeded defects: drop the generation check in _on_queue_finished -> the
    new queue reads complete while its first file still downloads -> red;
    drop it in _on_progress or _on_file_finished -> the stale tick repaints
    the step (the rail line moves) or marks Translation model Done -> red.
    """
    import threading

    from yamabiko.ui import thread_keeper

    tier = {"value": "default"}
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads._config.load_full_config",
        lambda: {"translation": {"quant_tier": tier["value"]}},
    )
    gate = threading.Event()
    calls: list = []

    class Gated:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            calls.append(filename)
            gate.wait(timeout=10.0)
            return Path(f"/fake/{filename}")

    page = DownloadsPage(downloader=Gated())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=5000)
    old_thread, stale = page._thread, page._relay
    tier["value"] = "small"
    page.initializePage()  # the restart detaches the old worker
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=5000)
    qtbot.waitUntil(lambda: page.rows.row_state(0) == "now", timeout=5000)  # its first file
    # Unguarded, the stale finish joins the new thread: let that thread end.
    threading.Timer(0.5, gate.set).start()
    try:
        stale.queue_finished(True)  # what Qt delivers after the disconnect
        assert not page._completed and not page.isComplete()
        rail = _rail_record(page)
        stale.progress(400_000_000, 600_000_000)
        stale.file_finished(5, _DEFAULT_GGUF)  # the old queue's GGUF
        assert rail == [], "a stale tick repainted the new queue's step"
        assert page.rows.row_state(1) == "waiting"
    finally:
        gate.set()
        old_thread.wait(5000)
        thread_keeper.discard(old_thread)
    qtbot.waitUntil(page.isComplete, timeout=5000)


def test_shutdown_thread_detaches_running_thread_into_keeper(monkeypatch, tmp_path, qtbot):
    """shutdown_thread() while the worker is blocked inside a download must not
    block forever and must not leave the running QThread parented to the page
    -- it detaches it into yamabiko.ui.thread_keeper so page destruction can
    never destroy a running thread (Qt 6 qFatal -> abnormal exit -> watchdog
    relaunches the wizard the user just cancelled).

    Seeded defect: remove the thread_keeper.release(...) call (just null the
    page refs) → thread keeps its page parent / registry stays empty → red.
    """
    import threading

    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    entered = threading.Event()

    class Blocking:
        def download(self, *args, **kwargs):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    page = DownloadsPage(downloader=Blocking())
    qtbot.addWidget(page)
    page.initializePage()
    # Wait until the worker is INSIDE the blocking download — isRunning()
    # alone races the worker's between-files cancel check.
    qtbot.waitUntil(entered.is_set, timeout=5000)
    thread = page._thread
    try:
        page.shutdown_thread(wait_ms=100)
        assert thread.isRunning(), "blocked thread should still be draining"
        assert thread.parent() is None
        assert any(t is thread for t, _ in thread_keeper._LINGERING)
        assert thread_keeper.any_running()
        assert page._thread is None and page._worker is None
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_shutdown_thread_joins_finished_thread_without_detach(
    monkeypatch, tmp_path, qtbot, fake_downloader,
):
    """After the queue finished, shutdown_thread() joins the thread and does
    NOT park it in thread_keeper -- no leak on clean teardown.

    Seeded defect: release unconditionally → registry gains an entry → red.
    """
    from yamabiko.ui import thread_keeper

    page = DownloadsPage(downloader=fake_downloader)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)

    before = len(thread_keeper._LINGERING)
    page.shutdown_thread(wait_ms=2000)
    assert len(thread_keeper._LINGERING) == before
    assert page._thread is None and page._worker is None


def test_start_downloads_releases_previous_running_thread(monkeypatch, tmp_path, qtbot):
    """Starting a new queue (Retry) while a cancelled thread is still draining
    must detach the old thread first -- otherwise it stays page-parented and
    its destruction mid-run aborts the process.

    Seeded defect: drop the release-previous-thread step in _start_downloads
    → the old thread keeps the page as parent → red.
    """
    import threading

    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    entered = threading.Event()

    class BlockFirstCall:
        def __init__(self) -> None:
            self.calls = 0

        def download(self, repo_id, filename, revision=None, progress_callback=None):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                gate.wait(timeout=10.0)
            return Path(f"/fake/{filename}")

    page = DownloadsPage(
        downloader=BlockFirstCall(),
    )
    qtbot.addWidget(page)
    page.initializePage()
    # Wait until the worker is INSIDE the blocking download — isRunning()
    # alone races the worker's between-files cancel check.
    qtbot.waitUntil(entered.is_set, timeout=5000)
    old_thread = page._thread
    try:
        page._start_downloads()  # what Retry does while the old queue drains
        assert page._thread is not old_thread
        assert old_thread.parent() is None
        assert any(t is old_thread for t, _ in thread_keeper._LINGERING)
        qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)
    finally:
        gate.set()
        old_thread.wait(5000)
        thread_keeper.discard(old_thread)


def test_start_downloads_shows_waiting_hint_while_previous_thread_drains(
    monkeypatch, tmp_path, qtbot,
):
    """Retry while the previous worker is still draining (detached but blocked
    inside an uninterruptible hf_hub_download) must surface 'Waiting for the
    other download to finish…' in the step's right-hand text instead of
    nothing -- HF's per-blob lock can serialize the new worker behind
    the drain, so the page would otherwise show no explanation until the new
    worker's first file_started.

    The hint is set on page.time_left the moment _start_downloads detaches a
    still-running thread (thread_keeper.any_running() True). It is observed
    on page.time_left.text() -- what the user sees -- BEFORE the new worker can
    emit file_started (the blocking downloader on the new queue never reaches
    its first progress tick here).

    Seeded-defect check: the text stays empty when _start_downloads
    has no waiting-hint branch. A one-line removal of that branch turns this
    red again.
    """
    import threading

    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    entered = threading.Event()

    # Block ONLY the first (old) worker's call so it is still draining when
    # _start_downloads runs. The second (new) worker's calls return at once,
    # so the new page-parented thread finishes and self-joins via
    # _on_queue_finished — leaving no running page-parented QThread for qtbot
    # teardown to qFatal on (Qt 6).
    class BlockFirstCall:
        def __init__(self) -> None:
            self.calls = 0

        def download(self, repo_id, filename, revision=None, progress_callback=None):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                gate.wait(timeout=10.0)
            return Path(f"/fake/{filename}")

    page = DownloadsPage(
        downloader=BlockFirstCall(),
    )
    qtbot.addWidget(page)
    page.initializePage()
    # Wait until the first worker is INSIDE the blocking download so the
    # detach in _start_downloads leaves a genuinely-running thread.
    qtbot.waitUntil(entered.is_set, timeout=5000)
    old_thread = page._thread
    try:
        page._start_downloads()  # Retry while the old queue is still draining
        # The detached old thread is genuinely still executing — this is the
        # real condition that triggers the hint, not a stubbed flag. The new
        # worker hasn't run yet (it's queued on the new thread's event loop),
        # so the hint is still the visible status text.
        assert thread_keeper.any_running()
        assert "Waiting for the other download to finish" in page.time_left.text(), (
            f"expected the draining-thread hint, got {page.time_left.text()!r}"
        )
    finally:
        gate.set()
        old_thread.wait(5000)
        thread_keeper.discard(old_thread)
        # Let the new (page-parented) queue finish and self-join so qtbot's
        # widget teardown never destroys a running QThread.
        qtbot.waitUntil(lambda: page._completed, timeout=5000)


def test_start_downloads_omits_waiting_hint_when_no_previous_thread_drains(
    monkeypatch, tmp_path, qtbot, fake_downloader,
):
    """A first/clean Retry with nothing draining (thread_keeper.any_running()
    False) must NOT show the waiting hint -- the right-hand text stays empty.
    Proves the hint is conditional on a real drain, not
    unconditional decoration.

    any_running is pinned False so the assertion can't be poisoned by a
    thread another test parked in the process-global registry.

    Seeded-defect check: make the hint unconditional (drop the any_running()
    guard) → the waiting hint shows → red.
    """
    monkeypatch.setattr(
        "yamabiko.wizard_pages.downloads.thread_keeper.any_running",
        lambda: False,
    )

    page = DownloadsPage(
        downloader=fake_downloader,
    )
    qtbot.addWidget(page)
    page._start_downloads()
    # _on_file_started/_on_queue_finished may overwrite status asynchronously
    # via queued signals, but the synchronous _start_downloads body has
    # already run; capture the text it set before yielding to the event loop.
    assert "Waiting for the other download" not in page.time_left.text()
    qtbot.waitUntil(lambda: page._completed, timeout=5000)


# ---------------------------------------------------------------------------
# Integration tests — real HuggingFace (Silero VAD ONNX, ~2 MB)
# ---------------------------------------------------------------------------

def _download_or_skip(downloader, **kwargs):
    """Run a real HF download, but ``pytest.skip`` (not fail) when the Hub
    is unreachable / rate-limiting.

    A ``429 Too Many Requests`` (``HfHubHTTPError``) or a
    ``LocalEntryNotFoundError`` ("couldn't reach the Hub and the file
    isn't cached") is an ENVIRONMENT condition, not a wizard-download
    regression. The recurring ``onnx-community/silero-vad`` 429 from
    GitHub-hosted runners has reddened CI repeatedly; failing on it
    punishes a green codebase for HF's rate-limiter.

    A genuine bad repo / filename still raises ``EntryNotFoundError`` /
    ``RepositoryNotFoundError`` (a reachable-Hub 404, NOT
    ``LocalEntryNotFoundError``) or a non-429 ``HfHubHTTPError`` and
    therefore still FAILS -- so the "invented repo string never verified"
    regression class these @real tests exist to catch stays caught.
    Mirrors the self-skip idiom already used in
    tests/test_real_hf_signature.py.
    """
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    try:
        return downloader.download(**kwargs)
    except LocalEntryNotFoundError as exc:
        pytest.skip(f"HF unreachable / rate-limited (file not cached): {exc}")
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            pytest.skip(f"HF rate-limited (429 Too Many Requests): {exc}")
        raise


@pytest.mark.real
def test_real_hf_download_silero_onnx():
    """Actually download Silero VAD ONNX from HuggingFace.

    Catches repo-URL / filename / signature regressions that mocked tests
    can't. ~2 MB — fast enough even on CI. Relies on the HF cache pinned
    by yamabiko.hf_env, so the returned path is under
    %LOCALAPPDATA%\\Yamabiko\\hf-cache (not user's default HF cache).
    Downloads at the pinned revision, as the wizard queue does, so the
    real bytes pass through the downloader's pinned-digest check.

    Skips (does not fail) when HF is rate-limiting / unreachable; a
    genuine bad repo/filename still fails — see _download_or_skip.
    """
    from yamabiko import models

    downloader = _HFDownloader()
    path = _download_or_skip(
        downloader,
        repo_id=models.SILERO_REPO,
        filename=models.SILERO_FILE,
        revision=models.SILERO_REVISION,
    )
    assert path.exists(), f"File not found at {path}"
    size = path.stat().st_size
    assert size > 1_000_000, f"File too small ({size} bytes) — likely corrupt or truncated"


def test_cancel_raises_once_then_post_cancel_ticks_stay_silent():
    """huggingface_hub 1.19's XetSession surfaces EVERY exception raised
    from the progress callback via sys.excepthook (PyErr print) while STILL
    not aborting the transfer -- so a cancelled multi-hundred-tick GGUF
    download would print hundreds of tracebacks to stderr. The raise only
    buys a mid-file abort on the plain-HTTP path, and once is enough for
    that: the first post-cancel tick raises, every later tick returns
    silently (no raise, no progress signal).

    The fake downloader below reproduces the hub-1.19 xet contract at
    the _Downloader protocol seam: callback raises are caught and the
    transfer keeps ticking. The subject — the worker's cancel/raise
    policy — is real.

    Seeded-defect check: drop the worker's raised-once flag (raise on
    every post-cancel tick) — raises_seen becomes 3, the ==1 assert
    goes red.
    """
    worker_box: list = []

    class _Hub119XetDownloader:
        def __init__(self):
            self.raises_seen = 0
            self.post_cancel_progress_emits = 0

        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            for i, done in enumerate((1, 2, 3, 4)):
                if i == 0:
                    worker_box[0].cancel()  # user cancels at first tick
                try:
                    progress_callback(done, 4)
                except _DownloadCancelledError:
                    self.raises_seen += 1  # hub 1.19: report + continue
            return Path("/fake/cache/file")

    dl = _Hub119XetDownloader()
    worker = _DownloadWorker(
        [DownloadSpec("org/repo", "file.onnx")], downloader=dl,
    )
    worker_box.append(worker)
    progress_emits: list = []
    results: list = []
    worker.progress.connect(lambda d, t: progress_emits.append((d, t)))
    worker.queue_finished.connect(results.append)

    worker.run()

    assert dl.raises_seen == 1, (
        f"cancel raised {dl.raises_seen}x from the progress callback; "
        f"must raise exactly once — on hub 1.19's xet path every raise "
        f"is printed to stderr via sys.excepthook, so per-tick raising "
        f"spams a traceback per tick for the rest of the file"
    )
    assert results == [False], f"expected cancelled queue, got {results}"
    assert worker.error is None, (
        f"cancel misclassified as failure: {worker.error!r}"
    )
    assert progress_emits == [], (
        f"post-cancel ticks must not emit progress (the page would keep "
        f"animating a cancelled download); got {progress_emits}"
    )


@pytest.mark.real
@pytest.mark.qt_no_exception_capture
def test_real_cancel_during_first_file_stops_queue_before_second(tmp_path):
    """A cancel issued once the first REAL HF transfer reports progress must
    stop the queue before the second file, with no error recorded (cancelled
    ≠ failed). Deterministic across both transfer paths: the HTTP path aborts
    mid-file via the progress-callback raise; the xet path swallows the raise,
    finishes file 1, and stops at the between-files
    check. A fresh cache_dir forces a real transfer so progress actually
    fires — a cache hit emits no progress and would skip cancellation
    entirely.

    Seeded defect: drop the cancel re-check in run()'s except handler (HTTP
    path records an error) or the between-files check (xet path starts file
    2) → red.

    qt_no_exception_capture: since huggingface_hub 1.19 the XetSession
    reports the (single, by design) post-cancel callback raise via
    sys.excepthook while continuing the transfer -- pytest-qt would
    capture that expected report and fail the test even though every
    behavioural assertion holds. The raise-once policy is pinned by
    test_cancel_raises_once_then_post_cancel_ticks_stay_silent.
    """

    worker_box: list = []

    class _CancelOnFirstProgress(_HFDownloader):
        """Real downloader; flips the worker's cancel flag from inside the
        progress-callback chain — the same execution context the
        production closure runs in (the xet path invokes it from a Rust
        worker thread, so a Qt-signal hop would be queued and never
        delivered during a synchronous run())."""

        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            from huggingface_hub.errors import (
                HfHubHTTPError, LocalEntryNotFoundError,
            )

            def wrapped(done, total):
                worker_box[0].cancel()  # user cancels at first progress
                if progress_callback is not None:
                    progress_callback(done, total)

            try:
                return super().download(
                    repo_id, filename, revision=revision,
                    progress_callback=(
                        wrapped if progress_callback is not None else None
                    ),
                )
            except LocalEntryNotFoundError as exc:
                pytest.skip(f"HF unreachable / rate-limited: {exc}")
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429:
                    pytest.skip(f"HF rate-limited (429): {exc}")
                raise

    downloader = _CancelOnFirstProgress(cache_dir=tmp_path / "hf-cache")
    specs = [
        DownloadSpec("onnx-community/silero-vad", "onnx/model.onnx"),
        DownloadSpec("onnx-community/silero-vad", "onnx/model_fp16.onnx"),
    ]
    worker = _DownloadWorker(specs, downloader=downloader)
    worker_box.append(worker)
    started: list = []
    results: list = []
    worker.file_started.connect(lambda i, fn: started.append(fn))
    worker.queue_finished.connect(results.append)
    worker.run()

    assert results == [False], f"expected cancelled queue, got {results}"
    assert worker.error is None, (
        f"cancel misclassified as failure: {worker.error!r}"
    )
    assert started == ["onnx/model.onnx"], (
        f"second file must never start after a cancel; started={started}"
    )
    assert worker._cancelled, (
        "no progress event ever fired (cache hit?) — the cancel path was "
        "not exercised; the fresh cache_dir should have forced a transfer"
    )


@pytest.mark.real
def test_real_hf_download_is_idempotent():
    """Second download of the same file returns instantly from the HF cache.

    This is the property that replaced our old custom skip-if-exists
    check. HF's ETag verification + cache layout makes repeat calls free.

    Skips (does not fail) when HF is rate-limiting / unreachable so the
    cache can't be primed — see _download_or_skip.
    """
    import time as _time
    downloader = _HFDownloader()
    # Prime the cache (skips the whole test if HF won't serve it).
    _download_or_skip(
        downloader,
        repo_id="onnx-community/silero-vad",
        filename="onnx/model.onnx",
    )
    # Second call should be near-instant.
    t0 = _time.monotonic()
    path = _download_or_skip(
        downloader,
        repo_id="onnx-community/silero-vad",
        filename="onnx/model.onnx",
    )
    elapsed = _time.monotonic() - t0
    assert path.exists()
    # Generous ceiling to avoid CI flakiness; real cache hits are < 50 ms.
    assert elapsed < 3.0, f"Cached download took {elapsed:.2f}s — cache miss?"


# ---------------------------------------------------------------------------
# Progress rendering: the percent must stay in [0, 100] and the byte readout
# coherent even when the xet progress signal drifts.
#
# The xet signal (group_report.total_bytes_completed vs the tqdm
# total=expected_size) is known to drift across resume/retry — huggingface_hub
# #3234/#3266, xet-core #409 — which can render "1368 MB / 802 MB (170%)".
# The fix clamps at the consumer.
# ---------------------------------------------------------------------------

def test_progress_percent_clamps_xet_overcount_to_100():
    """A drifting xet sample (done 1368 MB > total 802 MB) renders as 100.

    Seeded defect: drop the ``min(100, ...)`` in _progress_percent → 170 → red.
    """
    assert _progress_percent(1368 * _MB, 802 * _MB) == 100


def test_progress_percent_floors_negative_to_zero():
    """A negative byte count floors at 0.

    The input is comfortably below -total/100 so the RAW formula is ≤ -1 (here
    int(-100*100/802) == -12); a smaller magnitude like -5 MB would truncate
    to 0 on its own and so would NOT exercise the ``max(0, ...)`` clamp.

    Seeded defect: drop the ``max(0, ...)`` in _progress_percent → -12 → red.
    """
    assert _progress_percent(-100 * _MB, 802 * _MB) == 0


def test_progress_percent_is_faithful_in_range():
    """In-range input is rendered accurately (not hardcoded to 0 or 100).

    Seeded defect: return a constant from _progress_percent → midpoint ≠ 50 → red.
    """
    assert _progress_percent(0, 802 * _MB) == 0
    assert _progress_percent(401 * _MB, 802 * _MB) == 50
    assert _progress_percent(802 * _MB, 802 * _MB) == 100


def test_progress_percent_guards_unknown_total():
    """total ≤ 0 (size not yet known) returns 0, never divides by zero.

    Seeded defect: remove the ``if total <= 0`` guard → ZeroDivisionError → red.
    """
    assert _progress_percent(123, 0) == 0


def test_drifting_xet_sample_never_reads_past_the_file_size(qtbot, monkeypatch):
    """A drifting xet sample (done far above the file size) must not push
    the amount or the row past 100 %: the step clamps the running file's
    bytes to its pinned size.

    Seeded defect: drop the clamp in _on_progress -> "9.5 of 6.6 GB" /
    "157%" -> red.
    """
    _default_tier(monkeypatch)
    dl = _TickingDownloader(_DEFAULT_GGUF, [8_000_000_000])
    page = DownloadsPage(downloader=dl)
    qtbot.addWidget(page)
    page.initializePage()
    try:
        qtbot.waitUntil(lambda: page.rows.row_state(1) == "now", timeout=5000)
        qtbot.waitUntil(lambda: page.amount.text() == "6.6 of 6.6 GB", timeout=5000)
        assert page.rows.row_text(1) == "Translation model, 5.1 GB, 100%"
    finally:
        dl.gate.set()
    qtbot.waitUntil(page.isComplete, timeout=5000)


def test_overall_amount_counts_finished_files_and_the_running_file(qtbot, monkeypatch):
    """L4: the top line counts every finished file plus the running file's
    bytes against the queue's pinned total, the rows read Done / the running
    percent / Waiting; on success every row is Done and the rail line hides.

    Seeded defects: count only the running file -> "2 of 6.6 GB" -> red;
    send the last fraction instead of (None, False) on success -> red.
    """
    _default_tier(monkeypatch)
    dl = _TickingDownloader(_DEFAULT_GGUF, [2_000_000_000])
    page = DownloadsPage(downloader=dl)
    qtbot.addWidget(page)
    rail = _rail_record(page)
    page.initializePage()
    try:
        qtbot.waitUntil(lambda: page.amount.text() == "3.5 of 6.6 GB", timeout=5000)
        assert page.rows.row_text(0) == "Speech recognition, 1.5 GB, Done"
        assert page.rows.row_text(1) == "Translation model, 5.1 GB, 39%"
        assert page.rows.row_state(1) == "now"
        assert page.rows.row_text(2) == "Voice detection, 2 MB, Waiting"
        assert rail[-1] == (pytest.approx(3_512_116_195 / 6_602_796_529), False)
    finally:
        dl.gate.set()
    qtbot.waitUntil(page.isComplete, timeout=5000)
    assert page.amount.text() == "6.6 of 6.6 GB"
    assert [page.rows.row_state(i) for i in range(3)] == ["done", "done", "done"]
    assert rail[-1] == (None, False)


def test_time_left_comes_from_the_overall_byte_rate(qtbot, monkeypatch):
    """L4: time left divides everything still to fetch (not just the
    running file) by the observed rate: 0.5 GB in 100 s while 5.6 GB
    remain -> "About 19 minutes left".

    Seeded defect: feed _TimeLeft the running file's bytes -> "About 2
    minutes left" -> red.
    """
    _default_tier(monkeypatch)
    clock = iter([0.0, 100.0])
    dl = _TickingDownloader("model.bin", [500_000_000, 1_000_000_000])
    page = DownloadsPage(downloader=dl, clock=lambda: next(clock))
    qtbot.addWidget(page)
    page.initializePage()
    try:
        qtbot.waitUntil(
            lambda: page.time_left.text() == "About 19 minutes left", timeout=5000,
        )
    finally:
        dl.gate.set()
    qtbot.waitUntil(page.isComplete, timeout=5000)


# ---------------------------------------------------------------------------
# Version-tolerant progress hook.
#
# huggingface_hub >= 1.1.0 (the bundle pin) exposes ``tqdm_class=`` on
# hf_hub_download; 0.x / 1.0.x do not (the dev venv drifted to 0.36.2 once
# already -- see pyproject.toml). _HFDownloader must report byte progress on
# BOTH: inject tqdm_class when the signature accepts it, otherwise rebind the
# module-global tqdm class in huggingface_hub.utils.tqdm that hub's real
# transfer paths instantiate at call time. These tests drive _HFDownloader
# end-to-end through fake hf_hub_download bodies (no network) whose
# signatures select the branch -- the same predicate production uses.
# ---------------------------------------------------------------------------

import importlib as _importlib
import logging as _logging

import huggingface_hub as _hf


def _hub_tqdm_module():
    # ``huggingface_hub.utils.tqdm`` as an ATTRIBUTE resolves to the class
    # (the package re-exports it); go through the module registry.
    return _importlib.import_module("huggingface_hub.utils.tqdm")


def test_progress_hook_injects_tqdm_class_when_hub_signature_accepts_it(
    monkeypatch, tmp_path,
):
    """When hf_hub_download's signature has ``tqdm_class`` (hub >= 1.1.0),
    the downloader must pass a callback-emitting class through it, and the
    class must tick even on a DISABLED bar (the frozen wizard is windowed:
    no TTY, so hub's disable-on-non-TTY default disables the bar).

    Seeded defects: drop the ``kwargs["tqdm_class"] = ...`` injection in
    _hub_progress_hook (fake sees tqdm_class=None, zero ticks) -> red;
    or revert _make_progress_tqdm to reading ``self.n`` (disabled bar
    never increments it, ticks report done=0) -> red.
    """
    landed = tmp_path / "file.gguf"

    def fake_hf_hub_download(repo_id, filename, *, revision=None,
                             cache_dir=None, tqdm_class=None):
        # Mimic hub >= 1.1.0's transfer internals: instantiate the caller's
        # class for the file transfer and tick it per chunk. disable=True is
        # the windowed-app reality (tqdm disable-on-non-TTY).
        if tqdm_class is None:
            return str(landed)
        bar = tqdm_class(total=4096, disable=True, unit="B")
        try:
            bar.update(1024)
            bar.update(3072)
        finally:
            bar.close()
        return str(landed)

    monkeypatch.setattr(_hf, "hf_hub_download", fake_hf_hub_download)
    received: list = []
    path = _HFDownloader().download(
        "org/repo", "file.gguf",
        progress_callback=lambda done, total: received.append((done, total)),
    )
    assert path == landed
    assert received == [(1024, 4096), (4096, 4096)], (
        f"tqdm_class branch delivered no/garbled byte ticks: {received!r}"
    )


def test_progress_hook_rebinds_hub_tqdm_global_when_kwarg_missing(
    monkeypatch, tmp_path,
):
    """When hf_hub_download's signature LACKS ``tqdm_class`` (hub 0.x /
    1.0.x), the downloader must not pass the kwarg (TypeError) and must
    instead rebind huggingface_hub.utils.tqdm's module-global class -- the
    one hub's http_get/xet_get instantiate at call time -- then restore it.

    Seeded defects: delete the ``module.tqdm = patched`` rebinding in
    _hub_progress_hook -> zero ticks -> red; delete the finally-restore ->
    the identity assert on the global goes red.
    """
    hub_tqdm_mod = _hub_tqdm_module()
    original_cls = hub_tqdm_mod.tqdm
    landed = tmp_path / "model.onnx"

    def fake_hf_hub_download(repo_id, filename, *, revision=None,
                             cache_dir=None):
        # Mimic hub 0.x's transfer internals: resolve the module-global
        # class AT CALL TIME (exactly what _get_progress_bar_context does)
        # and tick it per chunk, disabled like on any non-TTY process.
        bar = _hub_tqdm_module().tqdm(total=4096, disable=True, unit="B")
        try:
            bar.update(1024)
            bar.update(3072)
        finally:
            bar.close()
        return str(landed)

    monkeypatch.setattr(_hf, "hf_hub_download", fake_hf_hub_download)
    received: list = []
    path = _HFDownloader().download(
        "org/repo", "model.onnx",
        progress_callback=lambda done, total: received.append((done, total)),
    )
    assert path == landed
    assert received == [(1024, 4096), (4096, 4096)], (
        f"hub-0.x fallback delivered no/garbled byte ticks: {received!r}"
    )
    assert hub_tqdm_mod.tqdm is original_cls, (
        "patched tqdm class leaked into huggingface_hub.utils.tqdm after "
        "the download returned"
    )


def test_progress_hook_restores_hub_tqdm_global_when_download_raises(
    monkeypatch,
):
    """A download failure on the fallback path must still restore the hub
    tqdm global -- a leaked patch would keep routing progress from LATER
    downloads (or other hub users in-process) into a dead callback.

    Seeded defect: replace _hub_progress_hook's try/finally with
    sequential restore-after-yield -> the raise skips restoration -> red.
    """
    hub_tqdm_mod = _hub_tqdm_module()
    original_cls = hub_tqdm_mod.tqdm

    def fake_hf_hub_download(repo_id, filename, *, revision=None,
                             cache_dir=None):
        raise RuntimeError("mid-transfer failure")

    monkeypatch.setattr(_hf, "hf_hub_download", fake_hf_hub_download)
    with pytest.raises(RuntimeError, match="mid-transfer failure"):
        _HFDownloader().download(
            "org/repo", "f.bin", progress_callback=lambda d, t: None,
        )
    assert hub_tqdm_mod.tqdm is original_cls, (
        "patched tqdm class leaked into huggingface_hub.utils.tqdm after "
        "a failed download"
    )


def test_overlapping_fallback_hooks_serialize_and_do_not_cross_wire_callbacks(
    monkeypatch,
):
    """Two overlapping fallback-branch (hub 0.x) downloads must not corrupt
    the process-global ``huggingface_hub.utils.tqdm.tqdm`` class. The real
    path to overlap: a cancel detaches a still-draining worker via
    thread_keeper (100 ms join timeout), the user re-Applies / Retries, and
    a second worker starts while the first is still inside its
    _hub_progress_hook patch-yield-restore span. Unserialized, the second
    hook captures the first's patched class as "original" and chains its
    patch on top, so every tick of the second (healthy) download runs the
    first (cancelled) worker's callback -- which raises
    _DownloadCancelledError inside the healthy download's stack, turning a
    good download into a spurious "Download failed".

    Choreography (events, no sleeps): worker A enters the fallback hook on
    a real thread and PARKS inside the span; worker B then attempts entry
    on a second real thread. Asserted:

    - serialization: B provably cannot get inside the span while A holds
      it (B's entered-event stays unset while A is parked);
    - (a) after A restores, B's callback receives B's own byte ticks,
      based on the TRUE hub original class;
    - (b) no _DownloadCancelledError (or anything else) escapes into B's
      download path, and cancelled A's callback never sees B's bytes;
    - (c) after both exit, the hub global is the true original class.

    Seeded one-line defect: drop the ``with _HUB_TQDM_PATCH_LOCK:`` span
    from _hub_progress_hook's fallback branch -> B enters within
    milliseconds while A is still parked inside (the blocks-until-restore
    assert goes red), and B's first tick raises _DownloadCancelledError
    out of cancelled-A's chained callback.
    """
    import threading

    from yamabiko.wizard_pages.downloads import _hub_progress_hook

    hub_tqdm_mod = _hub_tqdm_module()
    true_original = hub_tqdm_mod.tqdm

    def fake_hf_hub_download(repo_id, filename, *, revision=None,
                             cache_dir=None):
        # Signature intentionally lacks tqdm_class -> forces the fallback
        # branch. The transfer body itself is never invoked here: the
        # subject is the hook's patch/restore concurrency, driven directly.
        raise AssertionError("transfer body must not run in this test")

    monkeypatch.setattr(_hf, "hf_hub_download", fake_hf_hub_download)

    a_inside = threading.Event()
    a_may_exit = threading.Event()
    b_entered = threading.Event()
    a_ticks: list = []
    b_ticks: list = []
    b_errors: list = []

    def callback_a(done, total):
        # Worker A is a CANCELLED worker: its production _on_progress
        # closure raises on the next tick (see _DownloadWorker). Record
        # first so cross-wired bytes are visible in the failure output.
        a_ticks.append((done, total))
        raise _DownloadCancelledError()

    def callback_b(done, total):
        b_ticks.append((done, total))

    def worker_a():
        with _hub_progress_hook(callback_a, {}):
            a_inside.set()
            a_may_exit.wait(timeout=10.0)

    def worker_b():
        try:
            with _hub_progress_hook(callback_b, {}):
                b_entered.set()
                # Tick the class hub 0.x's http_get/xet_get would
                # instantiate at call time, disabled like any non-TTY
                # process (the windowed wizard).
                bar = _hub_tqdm_module().tqdm(total=4096, disable=True,
                                              unit="B")
                try:
                    bar.update(1024)
                    bar.update(3072)
                finally:
                    bar.close()
        except BaseException as exc:  # captured for the main thread's asserts
            b_errors.append(exc)

    ta = threading.Thread(target=worker_a, name="fallback-worker-A")
    tb = threading.Thread(target=worker_b, name="fallback-worker-B")
    try:
        ta.start()
        assert a_inside.wait(timeout=10.0), "worker A never entered its hook"

        tb.start()
        # THE serialization property: while A holds the patch span, B's
        # hook entry must block. The 1 s bound is headroom for the buggy
        # case only -- unserialized, B enters within milliseconds and
        # this assert goes red; serialized, B cannot enter at all until
        # A restores, so the wait deterministically times out.
        entered_while_a_inside = b_entered.wait(timeout=1.0)
        assert not entered_while_a_inside, (
            "second fallback hook entered its patch span while the first "
            "was still inside patch-yield-restore; overlapping downloads "
            "can cross-wire the process-global hub tqdm class"
        )

        a_may_exit.set()
        ta.join(timeout=10.0)
        tb.join(timeout=10.0)
        assert not ta.is_alive() and not tb.is_alive(), (
            "hook threads did not finish after A was released -- entry "
            "deadlocked"
        )

        assert b_entered.is_set(), (
            "worker B never entered its hook after A restored"
        )
        assert not b_errors, (
            f"the second (healthy) download's ticks raised inside its own "
            f"stack -- cross-wired into the cancelled worker's callback: "
            f"{b_errors!r}"
        )
        assert b_ticks == [(1024, 4096), (4096, 4096)], (
            f"second download's callback did not receive its own byte "
            f"ticks unpolluted: {b_ticks!r}"
        )
        assert a_ticks == [], (
            f"cancelled worker A's callback received the second "
            f"download's bytes: {a_ticks!r}"
        )
        assert hub_tqdm_mod.tqdm is true_original, (
            "after both hooks exited, huggingface_hub.utils.tqdm.tqdm is "
            "not the true original class -- a chained patch/restore leaked"
        )
    finally:
        # Unblock + drain the threads and repair the process-global class
        # no matter how the asserts above went -- a leaked patch would
        # poison every later test in this process.
        a_may_exit.set()
        ta.join(timeout=10.0)
        tb.join(timeout=10.0)
        hub_tqdm_mod.tqdm = true_original


def test_worker_logs_progress_line_through_hook_on_fake_transfer(
    monkeypatch, tmp_path, caplog,
):
    """The whole wizard progress chain -- worker -> _HFDownloader ->
    _hub_progress_hook -> hub tqdm seam -> worker's throttled logger --
    must land a 'progress: <file> X/Y bytes' record in the wizard log.
    That line is the forensic breadcrumb for stalled first-run downloads
    (main-pipeline.log); this pins it end-to-end with a fake, no-network
    hf_hub_download body that lacks tqdm_class (the drifted-hub shape).

    Seeded defects: drop the ``_log.info("  progress: ...")`` call in
    _DownloadWorker.run's closure -> no record -> red; or break any hook
    link above (e.g. stop rebinding the hub global) -> no callback, no
    record -> red.
    """
    landed = tmp_path / "file.gguf"

    def fake_hf_hub_download(repo_id, filename, *, revision=None,
                             cache_dir=None):
        bar = _hub_tqdm_module().tqdm(total=4096, disable=True, unit="B")
        try:
            bar.update(1024)
        finally:
            bar.close()
        return str(landed)

    monkeypatch.setattr(_hf, "hf_hub_download", fake_hf_hub_download)
    worker = _DownloadWorker(
        [DownloadSpec("org/repo", "file.gguf")], downloader=_HFDownloader(),
    )
    results: list = []
    worker.queue_finished.connect(results.append)
    with caplog.at_level(_logging.INFO, logger="yamabiko.wizard_pages.downloads"):
        worker.run()
    assert results == [True]
    assert any(
        "progress: file.gguf 1024/4096 bytes" in rec.getMessage()
        for rec in caplog.records
    ), (
        "no 'progress: <file> X/Y bytes' record reached the wizard log; "
        f"records={[rec.getMessage() for rec in caplog.records]!r}"
    )


def test_hf_downloader_deletes_file_whose_sha256_mismatches_pin_and_fails_item(
    monkeypatch, tmp_path,
):
    """A landed file that has a pinned digest (Silero here) but whose bytes
    do not hash to the pin must be deleted and must fail its queue item with
    "checksum mismatch for <file>". Otherwise a damaged download stays in the
    cache and every later hf_hub_download returns it again.

    Seeded defect: drop the digest check in _HFDownloader.download -> the
    queue succeeds and the file stays -> red.
    """
    from yamabiko import models

    landed = tmp_path / "model.onnx"
    landed.write_bytes(b"not the pinned silero graph")

    def fake_hf_hub_download(repo_id, filename, *, revision=None,
                             cache_dir=None):
        return str(landed)

    monkeypatch.setattr(_hf, "hf_hub_download", fake_hf_hub_download)
    spec = DownloadSpec(
        models.SILERO_REPO, models.SILERO_FILE, revision=models.SILERO_REVISION,
    )
    worker = _DownloadWorker([spec], downloader=_HFDownloader())
    results: list = []
    worker.queue_finished.connect(results.append)

    worker.run()

    assert results == [False]
    assert worker.error_spec is spec
    assert str(worker.error) == f"checksum mismatch for {models.SILERO_FILE}"
    assert not landed.exists(), "the mismatching file must be deleted"


def test_start_downloads_deletes_stale_incomplete_blobs_of_queued_repos_first(
    qtbot, monkeypatch, tmp_path,
):
    """A cancelled or killed earlier run leaves ``blobs/*.incomplete`` in the
    HF cache. They are never resumed and hold space the preflight counts
    against, so the page must delete them for the queued repos before the
    first download starts, and must leave other repos' files alone.

    Seeded defect: drop the _sweep_incomplete_blobs call from
    _start_downloads -> the partial still exists when the downloader runs.
    """
    from huggingface_hub import constants as hf_constants
    from huggingface_hub.file_download import repo_folder_name
    from yamabiko import models

    hub = tmp_path / "hub"
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub))
    # A detached draining thread left by an earlier test would make the
    # sweep skip itself by design; this test is about the sweep running.
    monkeypatch.setattr(downloads_mod.thread_keeper, "any_running", lambda: False)

    def _blobs(repo_id: str) -> Path:
        blobs = hub / repo_folder_name(repo_id=repo_id, repo_type="model") / "blobs"
        blobs.mkdir(parents=True)
        return blobs

    stale = _blobs(models.SILERO_REPO) / ("ab" * 32 + ".incomplete")
    stale.write_bytes(b"x" * 1024)
    foreign = _blobs("someone/else") / ("cd" * 32 + ".incomplete")
    foreign.write_bytes(b"y")
    stale_at_download: list = []

    class _Recorder:
        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            stale_at_download.append(stale.exists())
            return Path(f"/fake/{filename}")

    page = DownloadsPage(downloader=_Recorder())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(page.isComplete, timeout=5000)

    assert stale_at_download, "the queue never ran"
    assert not any(stale_at_download), (
        "the queued repo's stale .incomplete blob still existed when the "
        "downloads started"
    )
    assert foreign.exists(), "a repo outside the queue must not be swept"


# ---------------------------------------------------------------------------
# Disk-space preflight (P2): the queue used to start blind, so an ENOSPC
# only surfaced 20-60 minutes into the multi-GB GGUF. The decision logic is
# pure (_space_shortfall / _disk_full_message / _estimated_spec_bytes); the
# driver (_space_check) probes the cache + disk seams, monkeypatched
# here. The roomy default disk pin comes from conftest's
# deterministic_space_preflight; these tests re-patch the same seam.
# ---------------------------------------------------------------------------


def _own_hf_home(monkeypatch) -> None:
    """HF_HOME = Yamabiko's own cache, the only cache the reclaim may touch."""
    from yamabiko import paths

    monkeypatch.setenv("HF_HOME", str(paths.app_data_dir() / "hf-cache"))


def _tiny_disk(monkeypatch, free_bytes: int) -> None:
    """Point downloads.py's shutil.disk_usage at a volume with free_bytes."""
    import types

    usage = types.SimpleNamespace(total=2**40, used=2**40 - free_bytes,
                                  free=free_bytes)
    monkeypatch.setattr(
        downloads_mod, "shutil",
        types.SimpleNamespace(disk_usage=lambda path: usage),
    )


def test_space_shortfall_counts_the_working_room():
    """Free space above the raw requirement but inside the working room
    still falls short: HF writes through a temp *.incomplete file and a
    drive squeezed to zero mid-install breaks more than the download.

    Seeded defect: compare against the raw requirement (drop the margin)
    -> 0 -> red.
    """
    required = 6 * _GIB
    margin = 1 * _GIB
    assert _space_shortfall(required, required + margin // 2, margin_bytes=margin) == margin // 2


def test_space_shortfall_is_zero_with_headroom_or_nothing_needed():
    """No shortfall when free space clears required+margin, or when nothing
    remains to download (required <= 0)."""
    required = 6 * _GIB
    margin = 1 * _GIB
    assert _space_shortfall(required, required + margin, margin_bytes=margin) == 0
    assert _space_shortfall(0, 0, margin_bytes=margin) == 0


def test_disk_full_message_names_the_drive_and_rounds_the_shortfall_up():
    """L4/L6: "Not enough space on drive C:. Free up N GB, then click
    Retry." with N in decimal GB rounded UP to 0.1 (freeing a rounded-down
    figure would leave the queue blocked).

    Seeded defect: round to the nearest 0.1 -> 2,000,000,001 bytes reads
    "2 GB" -> red.
    """
    assert _disk_full_message("C:", 2_000_000_001) == (
        "Not enough space on drive C:. Free up 2.1 GB, then click Retry."
    )
    assert _disk_full_message("D:", 2_100_000_000) == (
        "Not enough space on drive D:. Free up 2.1 GB, then click Retry."
    )


def test_estimated_queue_size_covers_the_documented_first_run_payload():
    """The full uncached first-run queue must be estimated at >= 6.6 GB
    (documented ~5.1 GB default GGUF + ~1.5 GB whisper CT2 + VAD), so a
    machine with only a few GB free cannot start a doomed multi-GB queue.

    Seeded defect: shrink the GGUF ceiling to the small-file constant ->
    the sum falls below 6.6 GB -> red.
    """
    from yamabiko import models

    default_pin = models.TRANSLATOR_PINS["default"]
    specs = [DownloadSpec(default_pin.repo, default_pin.filename,
                          revision=default_pin.revision)]
    specs += [DownloadSpec(models.WHISPER_REPO, fn,
                           revision=models.WHISPER_REVISION)
              for fn in models.WHISPER_FILES]
    specs += [DownloadSpec(models.SILERO_REPO, models.SILERO_FILE,
                           revision=models.SILERO_REVISION)]
    total = sum(_estimated_spec_bytes(s) for s in specs)
    assert total >= int(6.6 * _GIB), (
        f"queue estimate {total / _GIB:.2f} GiB undershoots the documented "
        f"~6.6 GB first-run payload"
    )


def test_preflight_counts_only_uncached_specs(monkeypatch):
    """Already-cached files no-op via HF's ETag check and need no space, so
    a fully cached queue must pass the preflight even on a full disk; the
    same queue fully UNcached on the same disk must block.

    Seeded defect: drop the per-spec cache probe (count every spec) -> the
    fully-cached branch reports a shortfall -> red.
    """
    from yamabiko import models

    default_pin = models.TRANSLATOR_PINS["default"]
    specs = [DownloadSpec(default_pin.repo, default_pin.filename,
                          revision=default_pin.revision)]
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)

    monkeypatch.setattr(
        downloads_mod, "_repo_files_cached", lambda *a, **kw: True,
    )
    assert _space_check(specs).shortfall == 0

    monkeypatch.setattr(
        downloads_mod, "_repo_files_cached", lambda *a, **kw: False,
    )
    assert _space_check(specs).shortfall > 0


def test_preflight_fails_open_when_the_disk_probe_raises(monkeypatch):
    """A broken preflight must never block an install the downloads
    themselves might complete: any disk_usage error logs and lets the
    queue start.

    Seeded defect: let the OSError propagate (drop the outer try) -> this
    test errors -> red; or return a message from the except -> red.
    """
    import types

    def _boom(path):
        raise OSError("disk_usage exploded")

    monkeypatch.setattr(
        downloads_mod, "shutil", types.SimpleNamespace(disk_usage=_boom),
    )
    monkeypatch.setattr(
        downloads_mod, "_repo_files_cached", lambda *a, **kw: False,
    )
    specs = [DownloadSpec("org/repo", "file.gguf")]
    assert _space_check(specs) is None


def test_page_blocks_queue_start_on_space_shortfall_then_retry_runs_after_space_freed(
    qtbot, monkeypatch, fake_downloader,
):
    """On a preflight shortfall the page must NOT spawn the worker (zero
    downloader calls), must surface the shortfall with Retry offered, and
    must stay incomplete; once space frees, Retry runs the real queue to
    completion.

    Seeded defect: drop the preflight-block return from _start_downloads ->
    the downloader is called on the first initializePage -> red.
    """
    monkeypatch.setattr(
        downloads_mod, "_repo_files_cached", lambda *a, **kw: False,
    )
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)

    page = DownloadsPage(downloader=fake_downloader)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)  # a wrongly-spawned worker would have called by now

    assert fake_downloader.download.call_count == 0, (
        "the queue started despite the disk-space shortfall"
    )
    assert page.alert.message().startswith(f"Not enough space on drive {_drive()}.")
    assert page.retry_button.isVisibleTo(page), "Retry must be offered"
    assert not page.isComplete()
    assert page._thread is None, "no worker thread may exist on a block"

    # Space freed: the SAME Retry affordance re-runs the preflight and
    # then the real queue.
    _tiny_disk(monkeypatch, free_bytes=64 * _GIB)
    page.retry_button.click()
    qtbot.waitUntil(page.isComplete, timeout=5000)
    assert fake_downloader.download.call_count == len(page._last_specs)
    assert page.alert_box.isHidden()


# ---------------------------------------------------------------------------
# Retired-model cache reclaim: ~6.5 GB of dead v0.1.0 blobs (old shisa
# translator + deepdml ASR pins) sit in the SAME HF cache the ~7.9 GB
# preflight measures, so small-SSD upgraders were blocked with the space
# they need sitting right there. When the disk is full AND retired entries
# exist, the alert's "Delete old Yamabiko models (N GB)" button deletes
# exactly those revisions via the hub's delete_revisions strategy, then
# re-runs the space check.
# ---------------------------------------------------------------------------


def test_retired_repo_ids_do_not_overlap_current_pins():
    """The retired set must never contain a repo the current queue
    downloads -- an overlap would let "Delete old Yamabiko models" delete the
    model the user just fetched.

    Seeded defect: add a current pin (e.g. models.WHISPER_REPO) to
    _RETIRED_MODEL_REPOS -> intersection non-empty -> red.
    """
    from yamabiko import models

    current = {pin.repo for pin in models.TRANSLATOR_PINS.values()}
    current.add(models.WHISPER_REPO)
    current.add(models.SILERO_REPO)
    retired = set(downloads_mod._RETIRED_MODEL_REPOS)
    assert retired, "retired-repo set is empty; the reclaim path is dead code"
    assert not (retired & current), (
        f"retired set overlaps current pins: {retired & current}"
    )


def test_preflight_disk_full_offers_delete_old_models_as_the_primary_action(qtbot, monkeypatch):
    """L6: a blocked preflight shows the one disk-full state: the alert
    "Not enough space on drive X:. Free up N GB, then click Retry." (N
    rounded up: 7.4 GB here), "Delete old Yamabiko models (N GB)" as the
    primary action because old models sit in Yamabiko's own cache, Retry,
    Copy details, "Paused", the rail line grey; the queue does not start.

    Seeded defects: drop the retired scan from the preflight path -> no
    Delete button -> red; leave Retry as the primary -> red.
    """
    _default_tier(monkeypatch)  # the 7.4 GB below is the Standard queue's
    _own_hf_home(monkeypatch)
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries",
        lambda cache_dir=None: (("aaa111", "bbb222"), 6_500_000_000),
    )
    fake = MagicMock()

    page = DownloadsPage(downloader=fake)
    qtbot.addWidget(page)
    rail = _rail_record(page)
    page.initializePage()
    qtbot.wait(100)

    assert fake.download.call_count == 0, "queue started despite the block"
    assert page.alert.message() == (
        f"Not enough space on drive {_drive()}. Free up 7.4 GB, then click Retry."
    )
    assert page.delete_button.text() == "Delete old Yamabiko models (6.5 GB)"
    assert page.delete_button.isVisibleTo(page)  # inside the shown alert (Decision 148)
    assert page.delete_button.property("primary") is True
    assert not page.retry_button.property("primary")
    assert page.copy_button.isVisibleTo(page)
    assert page.time_left.text() == "Paused"
    assert rail[-1] == (0.0, True)
    assert not page.isComplete()


def test_delete_old_models_deletes_exactly_retired_revisions_and_restarts_preflight(
    qtbot, monkeypatch, fake_downloader,
):
    """Clicking Delete old Yamabiko models must delete exactly the scanned retired
    revisions through the delete seam, re-run the preflight, and -- with
    the freed space now sufficient -- start the queue to completion.

    Seeded defects: pass anything but the scanned revisions to the delete
    seam -> the exact-tuple assert goes red; drop the _start_downloads
    re-run from the click handler -> the queue never starts -> red.
    """
    import types

    _own_hf_home(monkeypatch)
    state = {"free": 1 * _GIB}
    monkeypatch.setattr(
        downloads_mod, "shutil",
        types.SimpleNamespace(disk_usage=lambda path: types.SimpleNamespace(
            total=2**40, used=0, free=state["free"],
        )),
    )
    monkeypatch.setattr(
        downloads_mod, "_repo_files_cached", lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries",
        lambda cache_dir=None: (
            (("aaa111", "bbb222"), int(6.5 * _GIB))
            if state["free"] < 2 * _GIB else ((), 0)
        ),
    )
    deleted: list = []

    def _fake_delete(revisions, cache_dir=None):
        deleted.append(tuple(revisions))
        state["free"] = 64 * _GIB  # the deletion frees the space

    monkeypatch.setattr(
        downloads_mod, "_delete_retired_cache_revisions", _fake_delete,
    )

    page = DownloadsPage(downloader=fake_downloader)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)
    assert page.delete_button.isVisibleTo(page)
    assert fake_downloader.download.call_count == 0

    page.delete_button.click()

    assert deleted == [("aaa111", "bbb222")], (
        f"delete seam must receive exactly the scanned retired revisions; "
        f"got {deleted!r}"
    )
    # Behavioral: the block clears (preflight re-ran against the freed
    # space) and the real queue runs to completion.
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)
    assert page.isComplete()
    assert page.alert_box.isHidden()


def test_preflight_disk_full_without_old_models_makes_retry_the_primary(qtbot, monkeypatch):
    """L6: no old models -> no Delete button (it would be a dead button) and
    Retry becomes the primary action.

    Seeded defect: show the Delete button unconditionally -> red.
    """
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries", lambda cache_dir=None: ((), 0),
    )
    page = DownloadsPage(downloader=MagicMock())
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)

    assert page.alert.message().startswith(f"Not enough space on drive {_drive()}.")
    assert page.delete_button.isHidden()
    assert page.retry_button.property("primary") is True


def test_preflight_disk_full_hides_delete_for_a_foreign_hf_home(
    qtbot, monkeypatch, tmp_path,
):
    """A user's own HF_HOME (yamabiko.hf_env keeps it) may be a cache shared
    with other tools. "Delete old Yamabiko models" there could delete repos
    those tools still use, so it must not be offered even when retired
    entries exist; Copy details says that cache was not scanned (not "0
    bytes", which reads as none found).

    Seeded defects: drop the _hf_home_is_ours() guard from _show_disk_full
    -> the Delete button appears -> red; report the skipped scan as 0 bytes
    -> red.
    """
    from yamabiko.ui import error_window

    copied: list = []
    monkeypatch.setattr(
        error_window, "copy_to_clipboard", lambda text: copied.append(text) or True,
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "shared-hf"))
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)
    monkeypatch.setattr(
        downloads_mod, "_repo_files_cached", lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries",
        lambda cache_dir=None: (("aaa111", "bbb222"), int(6.5 * _GIB)),
    )
    fake = MagicMock()

    page = DownloadsPage(downloader=fake)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)

    assert page.alert.message().startswith("Not enough space on drive ")
    assert page.delete_button.isHidden(), (
        "Delete old Yamabiko models must not be offered for a cache Yamabiko does not own"
    )
    page.copy_button.click()
    assert "Old Yamabiko models: not scanned (shared cache)" in copied[0]


def test_mid_download_disk_full_shows_the_same_alert_and_pauses_the_running_row(qtbot, monkeypatch):
    """L6: an ENOSPC during the GGUF lands in the same disk-full state as a
    blocked preflight: the shortfall re-measured on the drive, Delete old
    Yamabiko models when there are any, the running row paused at its
    percent, "Paused", the rail line grey.

    Seeded defect: route a mid-download ENOSPC to the generic failure ->
    "The download stopped." -> red.
    """
    _default_tier(monkeypatch)
    _own_hf_home(monkeypatch)
    disk = _mutable_disk(monkeypatch, 64 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries",
        lambda cache_dir=None: (("aaa111",), 4_400_000_000),
    )
    dl = _TickingDownloader(
        _DEFAULT_GGUF, [1_000_000_000],
        error=OSError(28, "No space left on device"),
        before_error=lambda: disk.update(free=1 * _GIB),
    )
    page = DownloadsPage(downloader=dl)
    qtbot.addWidget(page)
    rail = _rail_record(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed, timeout=5000)

    assert page.alert.message() == (
        f"Not enough space on drive {_drive()}. Free up 7.4 GB, then click Retry."
    )
    assert page.delete_button.text() == "Delete old Yamabiko models (4.4 GB)"
    assert page.delete_button.property("primary") is True
    assert page.rows.row_state(1) == "paused"
    assert page.rows.row_text(1) == "Translation model, 5.1 GB, 19%"
    assert page.time_left.text() == "Paused"
    assert rail[-1][1] is True


def test_mid_download_disk_full_sweeps_the_partial_file_before_measuring(qtbot, monkeypatch):
    """Decision 80 / L6: after a mid-download ENOSPC the partial
    *.incomplete file is dropped before the shortfall is measured again, so
    the space it held counts (1 GB here: "Free up 6.4 GB", not 7.4).

    Seeded defect: drop the sweep from _on_queue_finished's disk-full
    branch, or measure before it -> "Free up 7.4 GB" -> red.
    """
    _default_tier(monkeypatch)
    _own_hf_home(monkeypatch)
    disk = _mutable_disk(monkeypatch, 64 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries", lambda cache_dir=None: ((), 0),
    )

    def _sweep(repo_ids, cache_dir=None):
        list(repo_ids)
        disk["free"] += 1_000_000_000  # the partial blob's bytes come back
        return 1, 1_000_000_000

    monkeypatch.setattr(downloads_mod, "_sweep_incomplete_blobs", _sweep)
    dl = _TickingDownloader(
        _DEFAULT_GGUF, [1_000_000_000],
        error=OSError(28, "No space left on device"),
        before_error=lambda: disk.update(free=1 * _GIB),
    )
    page = DownloadsPage(downloader=dl)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.waitUntil(lambda: page._completed, timeout=5000)

    assert page.alert.message() == (
        f"Not enough space on drive {_drive()}. Free up 6.4 GB, then click Retry."
    )


def test_disk_full_and_failed_delete_details_are_redacted_and_keep_the_disk_facts(
    qtbot, monkeypatch, tmp_path,
):
    """L6 / Decision 83: Copy details in the disk-full state names the model
    folder without the user's folders and the size of the v0.1 model
    folders; a failed delete adds its error, also redacted, after those disk
    facts instead of replacing them.

    Seeded defects: skip redact_user in _disk_full_details or in the delete
    branch -> the LOCALAPPDATA path reaches the clipboard -> red; replace
    the details on a failed delete -> the disk facts are gone -> red; drop
    legacy_bytes from the _disk_full_details call -> the v0.1 line reads
    0 bytes -> red.
    """
    from yamabiko.ui import error_window

    local = tmp_path / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("YAMABIKO_APPDATA", str(local / "Yamabiko"))
    _own_hf_home(monkeypatch)  # %LOCALAPPDATA%\Yamabiko\hf-cache
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries",
        lambda cache_dir=None: (("aaa111",), 4_400_000_000),
    )
    blob = local / "Yamabiko" / "hf-cache" / "hub" / "blob"
    v01 = local / "Yamabiko" / "models" / "shisa"
    v01.mkdir(parents=True)
    (v01 / "model.gguf").write_bytes(b"x" * 10)

    def _locked(revisions, cache_dir=None):
        raise PermissionError(13, f"The process cannot access {blob}")

    monkeypatch.setattr(downloads_mod, "_delete_retired_cache_revisions", _locked)
    copied: list = []
    monkeypatch.setattr(
        error_window, "copy_to_clipboard", lambda text: copied.append(text) or True,
    )
    page = DownloadsPage(downloader=MagicMock())
    qtbot.addWidget(page)
    page.initializePage()
    page.copy_button.click()
    page.delete_button.click()
    page.copy_button.click()

    disk_full, after_delete = copied
    assert "Old Yamabiko models (v0.1 folder): 10 bytes" in disk_full
    assert "Model folder: %LOCALAPPDATA%\\Yamabiko\\hf-cache" in disk_full
    assert after_delete.startswith(disk_full), "the failed delete dropped the disk facts"
    assert "Couldn't delete the old Yamabiko models." in after_delete
    assert "%LOCALAPPDATA%\\Yamabiko\\hf-cache\\hub\\blob" in after_delete
    assert all(str(local).lower() not in text.lower() for text in copied)


def test_xet_disk_full_text_counts_as_disk_full_in_any_windows_language():
    """Decisions 79 and 132 (R3): the xet transfer is Rust and reports
    Windows' ERROR_DISK_FULL as "<the system message> (os error 112)",
    wrapped by the hub without an errno; the system message is in the
    Windows display language, so "(os error 112)" is the part that holds
    everywhere. No real condition reproduces the xet text (the real
    disk-full check is the preflight block): this pins the text match.

    Seeded defect: drop "os error 112" from _DISK_FULL_MARKERS -> the German
    text reads as a generic failure -> red.
    """
    from yamabiko.wizard_pages.downloads import _is_disk_full

    assert _is_disk_full(RuntimeError(
        "Auf dem Datentr\u00e4ger ist nicht gen\u00fcgend Speicherplatz vorhanden. (os error 112)"
    ))
    assert _is_disk_full(RuntimeError("There is not enough space on the disk. (os error 112)"))
    assert not _is_disk_full(RuntimeError("Connection reset by peer (os error 10054)"))


def test_delete_label_never_reads_zero_gb():
    """Decision 144: the old models' size on the delete button is decimal GB
    to the nearest 0.1 and at least 0.1, so a small retired set (the 2 MB
    turn-detector alone) never reads "Delete old Yamabiko models (0 GB)".

    Seeded defect: drop the 0.1 floor -> "(0 GB)" -> red.
    """
    from yamabiko.wizard_pages.downloads import _old_models_label

    assert _old_models_label(2_097_152) == "Delete old Yamabiko models (0.1 GB)"
    assert _old_models_label(4_400_000_000) == "Delete old Yamabiko models (4.4 GB)"


def test_a_failed_delete_names_it_and_gives_the_buttons_back(qtbot, monkeypatch):
    """Decision 83: a delete that fails (a locked blob, permissions) keeps
    the step stopped, says "Couldn't delete the old Yamabiko models." /
    "Free up space on the drive yourself, then click Retry.", enables both
    buttons again and starts nothing.

    Seeded defect: return from the except branch without re-enabling the
    buttons -> red.
    """
    _own_hf_home(monkeypatch)
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries",
        lambda cache_dir=None: (("aaa111",), 4_400_000_000),
    )

    def _locked(revisions, cache_dir=None):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(downloads_mod, "_delete_retired_cache_revisions", _locked)
    fake = MagicMock()
    page = DownloadsPage(downloader=fake)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)

    page.delete_button.click()

    assert page.alert.message() == "Couldn't delete the old Yamabiko models."
    assert page.alert.help_text() == "Free up space on the drive yourself, then click Retry."
    assert page.alert_box.isVisibleTo(page)
    assert page.delete_button.isEnabled() and page.retry_button.isEnabled()
    assert fake.download.call_count == 0


@pytest.mark.real
def test_real_hub_cache_scan_and_delete_removes_only_retired_repo(tmp_path):
    """Drives the REAL huggingface_hub cache manager (scan_cache_dir +
    the delete_revisions strategy) against a fabricated cache tree: the
    scan must report exactly the retired repo's revision and size, and
    executing the delete must remove that repo's directory from disk while
    leaving the current-pin repo untouched. scan_cache_dir/delete_revisions
    are new external-dep call sites; the subject here is the real library
    against real disk state (no symlinks -- the Windows fallback layout the
    docs support for scan/delete tooling).

    Seeded defect: filter on the wrong repo-id set (treat the current
    whisper repo as retired) -> the keeper hash is reported / the keeper
    dir deleted -> red.
    """
    from yamabiko import models

    retired_sha = "a" * 40
    keeper_sha = "b" * 40

    def _fabricate(repo_dir_name: str, sha: str, payload: bytes):
        repo = tmp_path / repo_dir_name
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text(sha, encoding="utf-8")
        snap = repo / "snapshots" / sha
        snap.mkdir(parents=True)
        (snap / "model.bin").write_bytes(payload)
        return repo

    retired_dir = _fabricate(
        "models--deepdml--faster-whisper-large-v3-turbo-ct2",
        retired_sha, b"x" * 4096,
    )
    keeper_dir = _fabricate(
        "models--" + models.WHISPER_REPO.replace("/", "--"),
        keeper_sha, b"y" * 2048,
    )

    revisions, total = downloads_mod._scan_retired_cache_entries(
        cache_dir=tmp_path,
    )
    assert revisions == (retired_sha,), (
        f"scan must report exactly the retired repo's revision; got "
        f"{revisions!r}"
    )
    assert total >= 4096, f"reclaimable size undercounts the blob: {total}"

    downloads_mod._delete_retired_cache_revisions(
        revisions, cache_dir=tmp_path,
    )
    assert not retired_dir.exists(), (
        "retired repo directory survived the delete strategy"
    )
    assert (keeper_dir / "snapshots" / keeper_sha / "model.bin").exists(), (
        "current-pin repo was damaged by the retired-cache delete"
    )


# ---------------------------------------------------------------------------
# v0.1 model folders: v0.1 kept its models (about 7 GB) in plain folders
# under %LOCALAPPDATA%\Yamabiko\models, which nothing reads now. The same
# "Delete old Yamabiko models" button removes them.
# ---------------------------------------------------------------------------


def _real_shutil(monkeypatch) -> None:
    """The folder delete needs the real shutil: conftest swaps downloads.py's
    shutil for a disk_usage-only stub."""
    import shutil

    monkeypatch.setattr(downloads_mod, "shutil", shutil)


def test_legacy_model_scan_finds_only_v01_folders(tmp_path, monkeypatch):
    """The scan of <app data>\\models finds only the folders v0.1 wrote,
    with their total size; anything else there is not Yamabiko's to delete.

    Seeded defects: list every folder under models -> "other" is found and
    the total reads 22 -> red; scan another root -> nothing found -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    root = tmp_path / "models"
    for name, filename, size in (
        ("shisa", "a.gguf", 10), ("silero", "b.onnx", 5), ("other", "c", 7),
    ):
        (root / name).mkdir(parents=True)
        (root / name / filename).write_bytes(b"x" * size)

    dirs, total = downloads_mod._scan_legacy_model_dirs()

    assert dirs == (root / "shisa", root / "silero")
    assert total == 15


def test_legacy_model_scan_ignores_a_linked_models_folder(tmp_path, monkeypatch):
    """A models folder that is itself a link (symlink or junction) is never
    scanned, so the delete can't reach through it into the folder it points
    to.

    Seeded defect: drop the models folder link check from
    _scan_legacy_model_dirs -> the scan returns the linked shisa folder, the
    delete empties it -> red.
    """
    _real_shutil(monkeypatch)
    app_data = tmp_path / "Yamabiko"
    app_data.mkdir()
    monkeypatch.setenv("YAMABIKO_APPDATA", str(app_data))
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "shisa").mkdir(parents=True)
    (elsewhere / "shisa" / "a.gguf").write_bytes(b"x" * 10)
    try:
        (app_data / "models").symlink_to(elsewhere, target_is_directory=True)
    except OSError as exc:  # Windows without the symlink privilege
        pytest.skip(f"can't create a symlink here: {exc}")

    scanned = downloads_mod._scan_legacy_model_dirs()
    downloads_mod._delete_legacy_model_dirs(scanned[0])

    assert scanned == ((), 0)
    assert (elsewhere / "shisa" / "a.gguf").read_bytes() == b"x" * 10


def test_legacy_model_delete_removes_folders_and_the_empty_root(tmp_path, monkeypatch):
    """Deleting the v0.1 folders also removes the models folder they leave
    empty.

    Seeded defect: drop the models folder rmdir -> it stays -> red.
    """
    _real_shutil(monkeypatch)
    root = tmp_path / "models"
    for name in ("shisa", "silero"):
        (root / name / "sub").mkdir(parents=True)
        (root / name / "sub" / "f").write_bytes(b"x")

    downloads_mod._delete_legacy_model_dirs((root / "shisa", root / "silero"))

    assert not root.exists()


def test_legacy_model_delete_keeps_a_root_with_other_files(tmp_path, monkeypatch):
    """A models folder that holds anything else stays, with that file.

    Seeded defect: rmtree the models folder instead of rmdir -> notes.txt is
    gone -> red.
    """
    _real_shutil(monkeypatch)
    root = tmp_path / "models"
    (root / "shisa").mkdir(parents=True)
    (root / "shisa" / "f").write_bytes(b"x")
    (root / "notes.txt").write_text("mine", encoding="utf-8")

    downloads_mod._delete_legacy_model_dirs((root / "shisa",))

    assert not (root / "shisa").exists()
    assert (root / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_legacy_model_delete_skips_a_folder_already_gone(tmp_path, monkeypatch):
    """A scanned folder that's gone by the click (deleted by hand while the
    alert was up, or by an earlier partial Delete) doesn't stop the Delete:
    the other folder goes, and so does the emptied models folder.

    Seeded defect: drop the exists check before rmtree -> FileNotFoundError
    on the missing folder, silero and models stay -> red.
    """
    import shutil

    _real_shutil(monkeypatch)
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    root = tmp_path / "models"
    for name in ("shisa", "silero"):
        (root / name).mkdir(parents=True)
        (root / name / "f").write_bytes(b"x")
    dirs, _total = downloads_mod._scan_legacy_model_dirs()
    assert dirs == (root / "shisa", root / "silero"), "test setup: scan missed a folder"
    shutil.rmtree(root / "shisa")  # by hand, while the alert is up

    downloads_mod._delete_legacy_model_dirs(dirs)

    assert not root.exists()


def test_preflight_disk_full_offers_delete_for_v01_folders(qtbot, monkeypatch, fake_downloader):
    """L6: with no retired cache entries, the v0.1 model folders alone make
    "Delete old Yamabiko models (N GB)" the primary action; clicking it
    deletes exactly the folders the alert was built on and starts the queue
    again.

    Seeded defects: build has_old from the retired revisions only -> no
    Delete button -> red; return early from _on_delete_clicked when there
    are no retired revisions -> nothing is deleted and the queue never
    starts -> red.
    """
    from yamabiko import paths

    _default_tier(monkeypatch)
    _own_hf_home(monkeypatch)
    disk = _mutable_disk(monkeypatch, 1 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries", lambda cache_dir=None: ((), 0),
    )
    p = paths.app_data_dir() / "models" / "shisa"
    monkeypatch.setattr(
        downloads_mod, "_scan_legacy_model_dirs", lambda root=None: ((p,), 7_000_000_000),
    )
    retired_deleted: list = []
    monkeypatch.setattr(
        downloads_mod, "_delete_retired_cache_revisions",
        lambda revisions, cache_dir=None: retired_deleted.append(revisions),
    )
    legacy_deleted: list = []

    def _fake_legacy_delete(dirs):
        legacy_deleted.append(dirs)
        disk["free"] = 64 * _GIB  # the deletion frees the space

    monkeypatch.setattr(downloads_mod, "_delete_legacy_model_dirs", _fake_legacy_delete)

    page = DownloadsPage(downloader=fake_downloader)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)

    assert fake_downloader.download.call_count == 0, "queue started despite the block"
    assert page.delete_button.text() == "Delete old Yamabiko models (7 GB)"
    assert page.delete_button.isVisibleTo(page)
    assert page.delete_button.property("primary") is True
    assert not page.retry_button.property("primary")

    page.delete_button.click()

    assert legacy_deleted == [(p,)]
    assert retired_deleted == []
    qtbot.waitUntil(lambda: page._completed and page._success, timeout=5000)
    assert fake_downloader.download.call_count == len(page._last_specs)


def test_preflight_disk_full_skips_v01_folders_on_another_drive(qtbot, monkeypatch):
    """L6: v0.1 folders on another drive than the model cache free nothing
    the check asks for: they aren't scanned, Delete isn't offered for them
    and Retry stays the primary action.

    Seeded defect: drop the drive comparison from _show_disk_full -> "Delete
    old Yamabiko models (7 GB)" is offered -> red.
    """
    from yamabiko import paths

    _default_tier(monkeypatch)
    _own_hf_home(monkeypatch)
    _tiny_disk(monkeypatch, free_bytes=1 * _GIB)
    monkeypatch.setattr(downloads_mod, "_repo_files_cached", lambda *a, **kw: False)
    monkeypatch.setattr(
        downloads_mod, "_scan_retired_cache_entries", lambda cache_dir=None: ((), 0),
    )
    legacy_root = paths.app_data_dir() / "models"
    monkeypatch.setattr(
        downloads_mod, "_drive_letter", lambda path: "D:" if path == legacy_root else "C:",
    )
    scans: list = []

    def _scan(root=None):
        scans.append(root)
        return (legacy_root / "shisa",), 7_000_000_000

    monkeypatch.setattr(downloads_mod, "_scan_legacy_model_dirs", _scan)
    fake = MagicMock()

    page = DownloadsPage(downloader=fake)
    qtbot.addWidget(page)
    page.initializePage()
    qtbot.wait(100)

    assert fake.download.call_count == 0, "queue started despite the block"
    assert page.alert.message().startswith("Not enough space on drive C:.")
    assert page.delete_button.isHidden()
    assert page.retry_button.property("primary") is True
    assert scans == []
    assert "Old Yamabiko models (v0.1 folder): 0 bytes" in page._details


@pytest.mark.real
@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junction")
def test_real_legacy_model_delete_leaves_a_junction_target(tmp_path, monkeypatch):
    """A junction inside a v0.1 folder (a model folder the user linked from
    elsewhere) goes as a link: the delete never enters it, so what it points
    to survives. A real NTFS junction, made with mklink /J.

    Seeded defect: replace shutil.rmtree with an os.walk(topdown=False)
    delete of every file -> the walk enters the junction and deletes the
    sentinel -> red.
    """
    import subprocess

    _real_shutil(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    shisa = tmp_path / "models" / "shisa"
    shisa.mkdir(parents=True)
    (shisa / "f").write_bytes(b"x")
    link = shisa / "link"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        check=True, capture_output=True,
    )
    assert link.is_junction(), "test setup: mklink /J made no junction"

    downloads_mod._delete_legacy_model_dirs((shisa,))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not shisa.exists()
