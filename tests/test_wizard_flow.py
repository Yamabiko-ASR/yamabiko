"""End-to-end wizard integration covering accept vs decline flows.

Focus: what lands in state.json + config.yaml after the user walks the
wizard happy paths. The pages themselves (audio device pick, smoke probe)
have dedicated unit tests in tests/wizard_pages/; this file asserts the
driver (YamabikoWizard) stitches them together and persists correctly.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest
import yaml

pytest.importorskip("pytestqt")


@pytest.fixture
def isolated_appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths
    paths.ensure_dirs()
    yield tmp_path


@pytest.fixture(autouse=True)
def deterministic_wizard_probes(monkeypatch):
    """Pin the wizard's two host-environment probes for every test here.

    * GPU detection: QuantTierPage now starts a detection QThread in
      __init__ (the synchronous nvidia-smi shell-out froze wizard startup);
      stubbing the gpu helpers to instant values keeps that thread's
      lifetime to microseconds so widget teardown can never race a real
      multi-hundred-ms nvidia-smi probe. Tests that exercise the detection
      thread itself re-patch these same helpers.
    * Disk-space preflight: DownloadsPage consults the HOST's real free
      disk before starting a queue; pin a roomy volume so the walk test
      can't flake on a nearly-full runner (mirrors
      tests/wizard_pages/conftest.py).
    """
    import types

    from yamabiko import gpu

    monkeypatch.setattr(gpu, "detect_vram_mib", lambda: None)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: None)

    try:
        from yamabiko.wizard_pages import downloads as _downloads
    except Exception:
        yield
        return
    roomy = types.SimpleNamespace(total=2**40, used=0, free=2**40)
    monkeypatch.setattr(
        _downloads, "shutil",
        types.SimpleNamespace(disk_usage=lambda path: roomy),
    )
    yield


@pytest.fixture(autouse=True)
def release_wizard_threads_before_close(qtbot, monkeypatch):
    """Join every test-built wizard's page threads before Qt destroys it.

    run_wizard() always calls release_background_threads() before the
    wizard dies; tests build YamabikoWizard directly, and qtbot only keeps a
    weak reference. A test-local wizard otherwise outlives the test function
    until a garbage collection (the ClientAreaFit cycle keeps it), and
    destroying it then while a page QThread (GPU detection starts in
    QuantTierPage.__init__) is still running is a Qt qFatal abort that kills
    the whole pytest process. The strong reference held here until teardown
    keeps the release ordered right before qtbot closes it.
    """
    from yamabiko.wizard import YamabikoWizard

    built = []
    real_init = YamabikoWizard.__init__

    def tracking_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)
        qtbot.addWidget(
            self, before_close_func=lambda w: w.release_background_threads()
        )

    monkeypatch.setattr(YamabikoWizard, "__init__", tracking_init)
    yield
    built.clear()


def _run_wizard_flow(qtbot):
    """Construct the wizard, populate each page programmatically, accept."""
    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)

    # Mark smoke as OK (not the focus of this test)
    wiz._smoke_page._smoke_ok = True
    return wiz


def test_accept_records_first_run_complete_in_state(qtbot, isolated_appdata):
    """The terminal accept() path must stamp first_run_complete=True into
    state.json, so needs_wizard() stops routing the user through setup.

    Seeded defect: set first_run_complete=False in accept()'s WizardResult
    -> needs_wizard() would loop forever -> this assertion goes red.
    """
    from yamabiko import paths

    wiz = _run_wizard_flow(qtbot)
    wiz.accept()  # calls wizard_state.save internally

    state = json.loads(paths.state_path().read_text(encoding="utf-8"))
    assert state["first_run_complete"] is True


def test_cancelled_wizard_restores_interim_audio_and_genre_on_relaunch(
    qtbot, isolated_appdata,
):
    """Cancel snapshots the current picks and a new wizard restores both.

    Seeded defect: omit ``audio_device_name`` when seeding AudioPage from
    partial state -> genre restores but the device falls back to index 0 ->
    red.
    """
    from yamabiko.wizard import YamabikoWizard

    devices = [
        {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
        {"index": 1, "name": "CABLE Output"},
    ]
    first = _run_wizard_flow(qtbot)
    first._audio_page._list_devices = lambda: list(devices)
    first._audio_page.initializePage()
    first._audio_page.device_list.segment("CABLE Output").click()
    first._content_page.genre_control.segment("anime").click()
    first.reject()

    second = YamabikoWizard()
    qtbot.addWidget(second)
    second._audio_page._list_devices = lambda: list(devices)
    second._audio_page.initializePage()

    assert second._audio_page.selected_device_name() == "CABLE Output"
    assert second._content_page.selected_genre() == "anime"


def test_cancelled_wizard_never_marks_first_run_complete(
    qtbot, isolated_appdata,
):
    """Cancel persists the exact typed six-field state as incomplete.

    Seeded defect: pass ``first_run_complete=True`` from reject() -> the
    completion marker and timestamp are stamped -> red.
    """
    from yamabiko import __version__, wizard_state

    wizard = _run_wizard_flow(qtbot)
    wizard.reject()

    saved = wizard_state.load()
    assert saved == wizard_state.WizardResult(
        first_run_complete=False,
        wizard_version=__version__,
        wizard_completed_at=None,
        audio_device_name=None,
        genre="variety",
        llama_smoke_ok=False,
    )


@pytest.mark.parametrize(
    "history",
    [
        "fresh install",
        "update",
        "update after a cancelled setup",
        "update with only the speech model missing",
        "update with the speech model cached",
    ],
)
def test_welcome_names_the_speech_model_download_only_on_an_update(
    qtbot, isolated_appdata, monkeypatch, history,
):
    """An update that moves the speech model pin reopens setup on an install
    that finished it before; Welcome then adds "This update brings a new
    speech model. The download is 6.6 GB." with the total the Download step
    fetches: every queued file not cached yet (1.5 GB when only the speech
    model is missing). A fresh install never shows it. A cancelled update
    session must not lose it: setup's interim save keeps the earlier
    completion time.

    The speech model is missing in every case but the last (probe stubbed),
    so there only the install's history decides. The last case is setup
    reopened for another reason (say, a missing translator model): it must
    not announce a speech model.

    Seeded defects: show the line whenever the speech model is missing ->
    the fresh-install case goes red; key it on first_run_complete, or let
    the interim save write wizard_completed_at=None -> the cancelled case
    goes red; never pass it to WelcomePage -> the update cases with the
    speech model missing go red; show it on any finished install, whatever
    the cache -> the speech-model-cached case goes red; show only the speech
    model's size -> the update cases with every model missing go red; count
    cached files too -> the only-the-speech-model case goes red.
    """
    from yamabiko import models, wizard_state
    from yamabiko.wizard import YamabikoWizard
    from yamabiko.wizard_pages import downloads

    speech_model_missing = history != "update with the speech model cached"
    monkeypatch.setattr(models, "default_whisper_missing", lambda: speech_model_missing)
    only_speech = history == "update with only the speech model missing"
    monkeypatch.setattr(
        downloads, "_repo_files_cached",
        lambda repo_id, filenames, revision=None: only_speech and repo_id != models.WHISPER_REPO,
    )
    if history != "fresh install":
        wizard_state.save(
            wizard_state.WizardResult(first_run_complete=True, llama_smoke_ok=True)
        )
    if history == "update after a cancelled setup":
        first = YamabikoWizard()
        qtbot.addWidget(first)
        first.reject()

    wizard = YamabikoWizard()
    qtbot.addWidget(wizard)
    welcome = wizard.page(wizard.pageIds()[0])
    leads = [lead.accessibleName() for lead in welcome.leads]
    if only_speech:
        update_line = "This update brings a new speech model. The download is 1.5 GB."
    else:
        # Everything the default tier queues: the speech model, the
        # Standard translator GGUF and Silero, from their size pins.
        translator = models.TRANSLATOR_PINS["default"]
        pins = [
            *((models.WHISPER_REPO, name, models.WHISPER_REVISION)
              for name in models.WHISPER_FILES),
            (translator.repo, translator.filename, translator.revision),
            (models.SILERO_REPO, models.SILERO_FILE, models.SILERO_REVISION),
        ]
        total = sum(models.pinned_size(*pin) or 0 for pin in pins)
        update_line = (
            f"This update brings a new speech model. The download is "
            f"{downloads._size_text(total)}."
        )
    if history in ("fresh install", "update with the speech model cached"):
        assert update_line not in leads and len(leads) == 2, leads
    else:
        assert leads[-1] == update_line, leads


def test_accept_override_failure_persists_neither_audio_nor_genre(
    qtbot, isolated_appdata, monkeypatch,
):
    """A failed combined choice publication leaves no partial disk state.

    Seeded defect: publish audio and genre through separate writer calls ->
    the audio-only first write lands before the genre-shaped write fails ->
    red.
    """
    from yamabiko import config_writer, paths
    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog

    wizard = _run_wizard_flow(qtbot)
    wizard._audio_page._list_devices = lambda: [
        {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
        {"index": 1, "name": "CABLE Output"},
    ]
    wizard._audio_page.initializePage()
    wizard._audio_page.device_list.segment("CABLE Output").click()
    wizard._content_page.genre_control.segment("anime").click()
    monkeypatch.setattr(SaveFailedDialog, "ask", lambda *_args, **_kwargs: False)

    real_atomic_write = config_writer._atomic_write

    def _fail_when_genre_reaches_disk(path, data):
        if data.get("genre") == "anime":
            raise OSError("locked")
        return real_atomic_write(path, data)

    monkeypatch.setattr(
        config_writer, "_atomic_write", _fail_when_genre_reaches_disk,
    )

    wizard.accept()

    assert not paths.state_path().exists(), (
        "completed state landed although config publication failed"
    )
    assert not paths.user_overrides_path().exists(), (
        "one wizard choice landed without the other"
    )


def test_wizard_persists_non_default_audio_device_to_user_overrides(qtbot, isolated_appdata):
    """User picks a non-default audio device in the wizard → it must land
    in ``user_overrides.yaml`` at ``audio.device``, not just state.json.

    Previously the wizard stored the pick only in ``state.json``; the
    pipeline reads ``cfg["audio"]["device"]`` from the merged layered
    config and never consulted ``state.json``, so the wizard's choice
    was silently ignored at runtime.
    """
    from yamabiko import paths
    from yamabiko.config import load_full_config

    wiz = _run_wizard_flow(qtbot)
    # One enumerated device (bypass PyAudio: CI has no audio).
    wiz._audio_page._list_devices = lambda: [
        {"index": 7, "name": "Speakers (Realtek) [Loopback]"},
    ]
    wiz._audio_page.initializePage()
    wiz.accept()

    overrides = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert overrides.get("audio", {}).get("device") == "Speakers (Realtek) [Loopback]", (
        f"expected audio.device in user_overrides, got {overrides!r}"
    )
    merged = load_full_config()
    assert merged["audio"]["device"] == "Speakers (Realtek) [Loopback]"


def test_accept_blocks_completion_until_the_override_write_succeeds_on_retry(
    qtbot, isolated_appdata, monkeypatch,
):
    """Completion must never be stamped ahead of the choices it summarizes.

    When the override write fails, accept() must NOT save
    first_run_complete and must NOT close the wizard (QDialog.accept not
    reached) before "Couldn't save your settings" names user_overrides.yaml
    and the error; its Retry runs the whole save again, and once the write
    succeeds the audio override lands on disk AND completion is saved.

    Seeded defects: restore the old order (save WizardResult FIRST, then a
    swallowed override write) -> first_run_complete=True is on disk while
    the dialog shows -> red; ignore the dialog's Retry -> the wizard stays
    open with nothing saved -> red.
    """
    from yamabiko import config_writer, paths, wizard_state
    from yamabiko import wizard as wizard_mod
    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog
    from yamabiko.wizard_pages.rail import BRAND

    wiz = _run_wizard_flow(qtbot)
    # Non-default device so the audio override write actually fires.
    wiz._audio_page._list_devices = lambda: [
        {"index": 5, "name": "Speakers (Realtek) [Loopback]"},
    ]
    wiz._audio_page.initializePage()

    # QDialog.accept is the only thing that closes the wizard; spy on it.
    super_calls: list = []
    monkeypatch.setattr(
        wizard_mod.QDialog, "accept",
        lambda self: super_calls.append(True),
    )
    shown: list = []

    def _retry(parent, palette, path, error, *, title):
        assert title == BRAND
        shown.append((path, str(error), super_calls[:], wizard_state.load()))
        return True  # Retry

    monkeypatch.setattr(SaveFailedDialog, "ask", _retry)

    # First write: the override write explodes; the Retry's write succeeds.
    state = {"n": 0}
    real_write = config_writer.write_overrides

    def _flaky(updates, *, remove_paths=()):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("disk full")
        return real_write(updates, remove_paths=remove_paths)

    monkeypatch.setattr(config_writer, "write_overrides", _flaky)

    wiz.accept()

    assert len(shown) == 1, "no dialog shown for the failed override write"
    path, error, closed_before, saved_before = shown[0]
    assert "user_overrides.yaml" in path, (
        f"the choice-write leg must name user_overrides.yaml, got {path!r}"
    )
    assert "disk full" in error
    assert closed_before == [] and saved_before is None, (
        "the wizard closed or saved completion before the choices persisted"
    )
    assert super_calls == [True], "the successful Retry did not close the wizard"
    overrides = yaml.safe_load(
        paths.user_overrides_path().read_text(encoding="utf-8")
    )
    assert overrides.get("audio", {}).get("device") == "Speakers (Realtek) [Loopback]", (
        f"device override missing after successful retry: {overrides!r}"
    )
    saved = wizard_state.load()
    assert saved is not None and saved.first_run_complete is True, (
        "first_run_complete not saved after the successful retry"
    )


def test_accept_blocks_completion_and_warns_when_state_save_fails_then_retry(
    qtbot, isolated_appdata, monkeypatch,
):
    """The wizard_state.save leg of accept(): when the completion save to
    state.json fails, the wizard must NOT close (super().accept() not
    reached) and must surface a warning naming state.json -- the file that
    actually failed -- not user_overrides.yaml. A retry Finish once the
    cause clears must save completion and close.

    Seeded defect: drop the `return` in accept()'s wizard_state.save except
    branch (fall through to super().accept()) -> the wizard closes on the
    failed save -> red.
    """
    from yamabiko import wizard_state
    from yamabiko import wizard as wizard_mod
    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog

    wiz = _run_wizard_flow(qtbot)

    # QDialog.accept is the only thing that closes the wizard; spy on it.
    super_calls: list = []
    monkeypatch.setattr(
        wizard_mod.QDialog, "accept",
        lambda self: super_calls.append(True),
    )
    warnings: list = []
    monkeypatch.setattr(  # Cancel: the wizard stays open for a later retry
        SaveFailedDialog, "ask",
        lambda parent, palette, path, error, *, title: warnings.append((path, error)) or False,
    )

    # First Finish: the completion save explodes (locked state.json); the
    # retry succeeds through the real writer.
    real_save = wizard_state.save
    state = {"n": 0}

    def _flaky_save(result):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("state.json is locked")
        return real_save(result)

    monkeypatch.setattr(wizard_mod.wizard_state, "save", _flaky_save)

    wiz.accept()

    assert super_calls == [], "wizard closed despite the completion save failing"
    assert warnings, "no warning shown on the failed completion save"
    joined = " ".join(str(part) for part in warnings[0])
    assert "state.json is locked" in joined, (
        f"warning does not name the underlying error: {warnings[0]!r}"
    )
    assert "state.json" in joined.replace("state.json is locked", ""), (
        f"warning must name the failing file (state.json), got: {joined!r}"
    )
    assert "user_overrides.yaml" not in joined, (
        f"warning blames user_overrides.yaml but state.json failed: {joined!r}"
    )
    assert wizard_state.load() is None, (
        "a state.json landed although the completion save raised"
    )

    # Retry: the save now succeeds -> completion recorded AND wizard closes.
    wiz.accept()

    assert super_calls == [True], "successful retry did not close the wizard"
    saved = wizard_state.load()
    assert saved is not None and saved.first_run_complete is True, (
        "first_run_complete not saved after the successful retry"
    )


def test_wizard_persists_non_default_genre_to_user_overrides(qtbot, isolated_appdata):
    """User picks a non-default genre → it must land in user_overrides.yaml."""
    from yamabiko import paths
    from yamabiko.config import load_full_config

    wiz = _run_wizard_flow(qtbot)
    # Pick a non-default genre (shipped default is "variety")
    wiz._content_page.genre_control.segment("anime").click()
    wiz.accept()

    overrides = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert overrides.get("genre") == "anime", (
        f"expected genre in user_overrides, got {overrides!r}"
    )
    merged = load_full_config()
    assert merged["genre"] == "anime"


def test_wizard_leaves_defaults_unwritten_to_user_overrides(qtbot, isolated_appdata):
    """Sparse-override discipline: if the user accepts wizard defaults,
    don't write those defaults to user_overrides.yaml (that would defeat
    the layered-config design that lets shipped-default improvements
    propagate on upgrade).
    """
    from yamabiko import paths

    wiz = _run_wizard_flow(qtbot)
    # audio device: the list is never built (no initializePage) so
    # selected_device_name() returns "" — treat as "unset, leave alone".
    # genre: default to "variety" (the shipped default).
    wiz.accept()

    # With nothing to persist, user_overrides.yaml stays absent.
    assert not paths.user_overrides_path().exists(), (
        f"user_overrides.yaml should not exist after accepting all defaults; "
        f"got {paths.user_overrides_path().read_text() if paths.user_overrides_path().exists() else ''}"
    )


def test_wizard_writes_round_trip_through_load_full_config(
    qtbot, isolated_appdata,
):
    """End-to-end producer/consumer alignment check.

    Wizard picks a non-default audio device and genre. Then load via the
    same load_full_config code path main.py uses, and verify every wizard
    pick lands at the exact dotted path main.py reads from. Catches
    writer/reader key drift bugs before they ship.
    """
    from yamabiko.config import load_full_config

    wiz = _run_wizard_flow(qtbot)
    # Inject a selected device (bypass list_devices; CI has no audio).
    wiz._audio_page._list_devices = lambda: [
        {"index": 0, "name": "My Test Loopback Device"},
    ]
    wiz._audio_page.initializePage()
    wiz._content_page.genre_control.segment("anime").click()

    wiz.accept()

    # Reload via the main.py code path and check each reader path.
    merged = load_full_config()

    # audio.device: wizard -> write_audio_and_translation -> ["audio",
    # "device"] -> main.py:379 reads cfg["audio"]["device"]
    assert merged["audio"]["device"] == "My Test Loopback Device", (
        f"audio.device producer/consumer drift; got {merged['audio']['device']!r}"
    )
    # genre: wizard -> write_override(["genre"], ...) -> main.py:330
    # reads cfg.get("genre", "variety")
    assert merged["genre"] == "anime", (
        f"genre producer/consumer drift; got {merged['genre']!r}"
    )


# ---------------------------------------------------------------------------
# Re-run reversal + teardown / cancel-exit
# ---------------------------------------------------------------------------

def test_rerun_genre_back_to_default_removes_genre_override(qtbot, isolated_appdata):
    """A re-run where the user DELIBERATELY moves the genre back to the
    default ('variety') must remove a previously written non-default override.

    The choice seeds to the prior pick ('anime') on construction, so moving it
    back is an explicit user action.

    Seeded defect: restore `if genre and genre != "variety":` in accept() →
    the stale 'anime' override survives → red.
    """
    from yamabiko import paths
    from yamabiko.config import load_full_config

    wiz1 = _run_wizard_flow(qtbot)
    wiz1._content_page.genre_control.segment("anime").click()
    wiz1.accept()
    overrides = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert overrides.get("genre") == "anime"

    wiz2 = _run_wizard_flow(qtbot)
    # User explicitly moves the (anime-seeded) choice back to the default.
    wiz2._content_page.genre_control.segment("variety").click()
    wiz2.accept()
    merged = load_full_config()
    assert merged["genre"] == "variety", (
        f"re-run did not undo the earlier genre pick; got {merged['genre']!r}"
    )
    if paths.user_overrides_path().exists():
        overrides2 = yaml.safe_load(
            paths.user_overrides_path().read_text(encoding="utf-8")
        )
        assert "genre" not in (overrides2 or {}), (
            f"expected the genre override pruned, got {overrides2!r}"
        )


def test_rerun_with_untouched_genre_choice_preserves_prior_pick(qtbot, isolated_appdata):
    """On a completed-wizard re-run, the genre choice must seed from the merged
    config, so an UNTOUCHED choice round-trips the prior pick instead of
    silently resetting it to 'variety' (an unconditional write would otherwise
    prune a deliberate earlier choice the moment the user re-runs setup and
    clicks straight through).

    Seeded defect: remove the merged-config genre seeding in
    YamabikoWizard.__init__ → the choice sits at index 0 ('variety') → the
    untouched accept prunes the anime override → red.
    """
    from yamabiko import paths
    from yamabiko.config import load_full_config

    wiz1 = _run_wizard_flow(qtbot)
    wiz1._content_page.genre_control.segment("anime").click()
    wiz1.accept()

    wiz2 = _run_wizard_flow(qtbot)
    # The choice must already show the effective genre from config...
    assert wiz2._content_page.selected_genre() == "anime", (
        "re-run choice did not seed from the merged config; an untouched "
        f"accept would reset the pick (got {wiz2._content_page.selected_genre()!r})"
    )
    # ...so accepting without touching it preserves the override.
    wiz2.accept()
    merged = load_full_config()
    assert merged["genre"] == "anime", (
        f"untouched re-run reset the genre pick; got {merged['genre']!r}"
    )
    overrides = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert overrides.get("genre") == "anime"


def test_rerun_with_untouched_device_choice_round_trips_saved_device(
    qtbot, isolated_appdata,
):
    """On a forced re-run (upgrade path) where the saved device still
    enumerates, the audio page must preselect it so an untouched
    Next-Next-Finish round-trips the SAME device through accept() --
    instead of writing whatever enumerates at index 0 over a working
    VB-Cable pick (speakers captured, no/garbled subtitles).

    Seeded defect: drop the saved-device seeding from
    AudioPage.initializePage's devices-present branch -> the re-run list
    sits at index 0 -> accept() overwrites the saved pick -> red.
    """
    from yamabiko.config import load_full_config

    cable = "CABLE Output (VB-Audio Virtual Cable) [Loopback]"
    devices = [
        {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
        {"index": 1, "name": cable},
    ]

    wiz1 = _run_wizard_flow(qtbot)
    wiz1._audio_page._list_devices = lambda: devices
    wiz1._audio_page.initializePage()
    wiz1._audio_page.device_list.segment(cable).click()  # user picks VB-Cable
    wiz1.accept()
    assert load_full_config()["audio"]["device"] == cable

    wiz2 = _run_wizard_flow(qtbot)
    wiz2._audio_page._list_devices = lambda: devices
    wiz2._audio_page.initializePage()  # page entry on the forced re-run
    assert wiz2._audio_page.selected_device_name() == cable, (
        "re-run list did not preselect the saved device; an untouched "
        "click-through would overwrite it"
    )
    wiz2.accept()  # untouched click-through
    assert load_full_config()["audio"]["device"] == cable, (
        "untouched re-run overwrote the saved audio device"
    )


def test_release_background_threads_detaches_blocked_download_thread(qtbot, isolated_appdata):
    """Wizard teardown must detach a download thread blocked inside
    hf_hub_download -- page destruction would otherwise destroy a running
    QThread, a Qt 6 qFatal abort whose non-42 exit code makes the watchdog
    relaunch the wizard the user just cancelled.

    Seeded defect: drop the downloads-page shutdown_thread() call from
    release_background_threads → thread keeps its page parent → red.
    """
    import threading
    from pathlib import Path

    from yamabiko.ui import thread_keeper
    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)

    gate = threading.Event()
    entered = threading.Event()

    class Blocking:
        def download(self, *args, **kwargs):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    wiz._downloads_page._downloader = Blocking()
    wiz._downloads_page.initializePage()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    thread = wiz._downloads_page._thread
    assert thread is not None
    try:
        wiz.release_background_threads(wait_ms=100)
        assert thread.isRunning(), "blocked thread should still be draining"
        assert thread.parent() is None
        assert any(t is thread for t, _ in thread_keeper._LINGERING)
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_release_background_threads_detaches_wedged_gpu_detection_thread(
    qtbot, isolated_appdata, monkeypatch,
):
    """Wizard teardown must also release the quant page's GPU-detection
    thread: the wizard can be cancelled while nvidia-smi is wedged (each
    shell-out blocks up to 5 s), and page destruction with that QThread
    still running is the same qFatal-abort / wizard-relaunch loop as the
    downloads case.

    Seeded defect: drop the quant-tier shutdown_thread() call from
    release_background_threads → thread keeps its page parent → red.
    """
    import threading

    from yamabiko import gpu
    from yamabiko.ui import thread_keeper
    from yamabiko.wizard import YamabikoWizard

    gate = threading.Event()
    entered = threading.Event()

    def _wedged_vram():
        entered.set()
        gate.wait(timeout=10.0)
        return None

    # Overrides the module autouse stub for this test only.
    monkeypatch.setattr(gpu, "detect_vram_mib", _wedged_vram)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: None)

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)
    assert entered.wait(5.0), "GPU detection never started"
    thread = wiz._quant_tier_page._detect_thread
    assert thread is not None
    try:
        wiz.release_background_threads(wait_ms=100)
        assert thread.isRunning(), "wedged detection thread should still run"
        assert thread.parent() is None
        assert any(t is thread for t, _ in thread_keeper._LINGERING)
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_release_background_threads_stops_a_running_sound_test(
    qtbot, isolated_appdata, monkeypatch,
):
    """Cancel, the close button or Esc inside the Sound step's 5 s test must
    release its stream and PyAudio before run_wizard returns. Nothing else
    does in time: the window outlives run_wizard (the ClientAreaFit cycle),
    so the page's destroyed hook fires only at garbage collection, after a
    cancelled setup has reached sys.exit(42) with a live WASAPI callback
    stream (native threads in finalization can mangle the exit code into a
    watchdog RESTART).

    Seeded defect: drop the stop_test() call from release_background_threads
    -> the stream stays open -> red.
    """
    from tests.wizard_pages.test_audio import _fake_devices, _install_recording_pyaudio
    from yamabiko.wizard import YamabikoWizard

    rec = _install_recording_pyaudio(monkeypatch)
    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)
    wiz._audio_page._list_devices = _fake_devices
    wiz._audio_page.initializePage()
    wiz._audio_page.test_button.click()
    assert rec.open_calls == 1

    wiz.release_background_threads(wait_ms=100)

    assert (rec.stop_stream_calls, rec.close_calls, rec.terminate_calls) == (1, 1, 1), (
        "the sound test's stream and PyAudio outlived the wizard's release"
    )


def test_exit_wizard_cancelled_hard_exits_42_when_threads_linger(monkeypatch):
    """With a detached download thread still draining, the cancel exit must
    use os._exit(42) -- normal interpreter finalization would destroy the
    running QThread and abort with a non-42 code, which the watchdog
    classifies RESTART (the wizard relaunch loop).

    Seeded defect: drop the any_running() branch (always sys.exit) → no
    os._exit call → red.
    """
    from yamabiko import wizard as wizard_mod
    from yamabiko.ui import thread_keeper

    hard_exits: list = []
    flushed: list = []

    class _HardExit(Exception):
        """Emulates os._exit never returning."""

    def fake_hard_exit(code: int) -> None:
        hard_exits.append(code)
        raise _HardExit()

    monkeypatch.setattr(thread_keeper, "any_running", lambda: True)
    monkeypatch.setattr(wizard_mod.os, "_exit", fake_hard_exit)
    monkeypatch.setattr(
        wizard_mod.logging, "shutdown", lambda: flushed.append(True),
    )
    with pytest.raises(_HardExit):
        wizard_mod.exit_wizard_cancelled()
    assert hard_exits == [42]
    assert flushed, "logs must be flushed before os._exit skips atexit"


def test_exit_wizard_cancelled_sys_exits_42_when_no_threads(monkeypatch):
    """The clean path keeps the ordinary SystemExit(42) sentinel."""
    from yamabiko import wizard as wizard_mod
    from yamabiko.ui import thread_keeper

    monkeypatch.setattr(thread_keeper, "any_running", lambda: False)
    with pytest.raises(SystemExit) as exc_info:
        wizard_mod.exit_wizard_cancelled()
    assert exc_info.value.code == 42


# ---------------------------------------------------------------------------
# Wizard page chain verification.
#
# Each flow test pokes pages directly and calls accept(), so deleting
# addPage(self._downloads_page) or addPage(self._quant_tier_page) from
# YamabikoWizard.__init__ would keep the whole suite green while a fresh
# install silently skips model downloads (app dead on first launch) or
# VRAM-tier selection.
# ---------------------------------------------------------------------------

_EXPECTED_PAGE_CHAIN = [
    "WelcomePage",
    "AudioPage",
    "ContentPage",
    "QuantTierPage",
    "DownloadsPage",
    "SmokePage",
    "DonePage",
]


def test_wizard_page_chain_registers_all_seven_pages_in_order(
    qtbot, isolated_appdata,
):
    """The registered page chain is exactly the seven pages in install
    order — downloads and quant-tier included.

    Seeded defect: delete `self.addPage(self._downloads_page)` (or the
    quant-tier addPage) from YamabikoWizard.__init__ -> red.
    """
    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)
    try:
        names = [type(wiz.page(i)).__name__ for i in wiz.pageIds()]
        assert names == _EXPECTED_PAGE_CHAIN, (
            f"wizard page chain drifted: {names}"
        )
    finally:
        # QuantTierPage starts its detection QThread in __init__. The window
        # is not destroyed when this function returns: the module's autouse
        # fixture holds it until teardown, and the ClientAreaFit cycle keeps
        # it past its last reference until a garbage collection. Whenever
        # that comes, a QThread still running is a Qt 6 fatal abort that
        # kills the whole pytest process (no traceback, no summary), so the
        # test releases its threads here, as run_wizard does after exec().
        wiz.release_background_threads(wait_ms=2000)


def test_download_progress_reaches_the_rail_line(qtbot, isolated_appdata):
    """L3: the Download step's progress drives the rail's line under
    Download, whatever step is current.

    Seeded defect: drop the railProgressChanged connection in
    YamabikoWizard.__init__ -> the line stays hidden -> red.
    """
    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)
    try:
        wiz._downloads_page.railProgressChanged.emit(0.5, True)
        assert wiz.rail.download_line() == (0.5, True)
        wiz._downloads_page.railProgressChanged.emit(None, False)
        assert wiz.rail.download_line() == (None, False)
    finally:
        wiz.release_background_threads(wait_ms=2000)


def test_wizard_next_walk_traverses_quant_tier_and_downloads_to_done(
    qtbot, isolated_appdata,
):
    """Drive the REAL navigation (next() + each page's validatePage /
    initializePage / isComplete gates) from Welcome through to Done. The
    downloads page must actually gate on its queue finishing and the
    smoke page on its probe — both stubbed at the seam (downloader /
    probe), not at the page level, so the page machinery itself runs.
    Done's footer holds only "Start Yamabiko" (no Cancel, no Back), and
    clicking it finishes setup (first_run_complete saved).

    Seeded defects: delete either addPage call -> the visited sequence
    drifts -> red. Break DownloadsPage.initializePage's _start_downloads
    call -> isComplete never goes true -> waitUntil times out -> red. Keep
    Back or Cancel on the last page, or connect the primary to next() ->
    red.
    """
    from pathlib import Path

    from yamabiko import wizard_state
    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)

    # Stub the SEAMS (device enumeration, HF downloader, llama probe) —
    # never the pages: initializePage/validatePage/isComplete must run.
    wiz._audio_page._list_devices = lambda: [
        {"index": 0, "name": "Fake Loopback Device"},
    ]

    class _InstantDownloader:
        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            return Path("C:/fake-cache") / filename  # instant cache hit

    wiz._downloads_page._downloader = _InstantDownloader()

    class _OkProbe:
        def check_health(self) -> bool:
            return True

        def translate(self, text: str) -> str:
            return "Hello"

    wiz._smoke_page._probe = _OkProbe()

    try:
        wiz.restart()  # enter the start page without exec()
        visited = [type(wiz.currentPage()).__name__]

        def advance():
            wiz.next()
            visited.append(type(wiz.currentPage()).__name__)

        advance()  # Welcome -> Sound (initializePage enumerates devices)
        advance()  # Sound -> Content
        advance()  # Content -> Quality
        advance()  # Quality -> Download (validatePage persists the tier)

        assert visited[-1] == "DownloadsPage", f"walk drifted: {visited}"
        # The real gate: Next is only legal once the download queue
        # reports success through the worker-thread signal chain.
        qtbot.waitUntil(wiz._downloads_page.isComplete, timeout=5000)
        assert wiz._downloads_page._success, "download queue did not succeed"

        advance()  # Downloads -> Smoke (initializePage starts the probe)
        assert visited[-1] == "SmokePage", f"walk drifted: {visited}"
        qtbot.waitUntil(wiz._smoke_page.isComplete, timeout=5000)
        assert wiz._smoke_page.smoke_ok(), "smoke probe did not succeed"

        advance()  # Smoke -> Done

        assert visited == _EXPECTED_PAGE_CHAIN, (
            f"navigation did not traverse the install chain in order: {visited}"
        )
        assert not wiz.cancel_button.isVisibleTo(wiz)
        assert not wiz.back_button.isVisibleTo(wiz)
        assert wiz.primary_button.text() == "Start Yamabiko"
        wiz.primary_button.click()
        saved = wizard_state.load()
        assert saved is not None and saved.first_run_complete is True, (
            "Start Yamabiko on Done did not save completion"
        )
    finally:
        wiz.release_background_threads()


def test_skip_on_a_failed_translator_check_goes_to_done(qtbot, isolated_appdata):
    """L7: Skip, offered only in the Translator step's failure alert, goes
    straight to Done (smoke not ok).

    Seeded defect: drop the SmokePage.skipped connection in
    YamabikoWizard.__init__ -> the wizard stays on the Translator step ->
    red.
    """
    from pathlib import Path

    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz)
    wiz._audio_page._list_devices = lambda: [
        {"index": 0, "name": "Fake Loopback Device"},
    ]

    class _InstantDownloader:
        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            return Path("C:/fake-cache") / filename  # instant cache hit

    wiz._downloads_page._downloader = _InstantDownloader()

    class _SilentProbe:
        def check_health(self) -> bool:
            return False

        def translate(self, text: str) -> str:
            raise AssertionError("translate must not run without health")

    wiz._smoke_page._probe = _SilentProbe()
    wiz._smoke_page._max_health_attempts = 1
    try:
        wiz.restart()
        for _ in range(4):
            wiz.next()  # Welcome -> Sound -> Content -> Quality -> Download
        qtbot.waitUntil(wiz._downloads_page.isComplete, timeout=5000)
        wiz.next()  # Download -> Translator
        qtbot.waitUntil(lambda: not wiz._smoke_page.alert_box.isHidden(), timeout=5000)

        wiz._smoke_page.skip_button.click()

        assert type(wiz.currentPage()).__name__ == "DonePage"
        assert wiz._smoke_page.smoke_ok() is False
    finally:
        wiz.release_background_threads()


def test_run_wizard_releases_threads_even_when_exec_raises(
    qtbot, isolated_appdata, monkeypatch,
):
    """An exception escaping exec() (PySide6 propagates some slot exceptions)
    must not skip the thread release -- skipping it causes a qFatal teardown
    on the still-running thread.

    Seeded defect: revert run_wizard's try/finally to sequential statements
    → release never recorded → red.
    """
    from yamabiko import wizard as wizard_mod

    released: list = []
    real_release = wizard_mod.YamabikoWizard.release_background_threads

    def raising_exec(self):
        raise RuntimeError("slot exception escaped exec")

    def recording_release(self, wait_ms=2000):
        released.append(True)
        # Spy, don't replace: run_wizard() builds a REAL wizard whose
        # detection QThread is running, and it is destroyed when run_wizard
        # returns. Recording the call without performing the release leaves
        # that thread alive at destruction, which is the Qt 6 fatal abort
        # this test exists to prevent.
        real_release(self, wait_ms=wait_ms)

    monkeypatch.setattr(wizard_mod.YamabikoWizard, "exec", raising_exec)
    monkeypatch.setattr(
        wizard_mod.YamabikoWizard, "release_background_threads",
        recording_release,
    )
    with pytest.raises(RuntimeError, match="slot exception escaped exec"):
        wizard_mod.run_wizard()
    assert released == [True], (
        "release_background_threads must run even when exec() raises"
    )
