"""QuantTierPage — async VRAM auto-detect + radio override + persistence.

VRAM detection is stubbed by monkeypatching the module-level helpers
in :mod:`yamabiko.gpu`; no nvidia-smi shell-out happens here. Detection
now runs on a worker QThread (the synchronous constructor shell-out
blocked the whole no-window wizard startup for up to ~10 s on a wedged
driver), so tests wait for the queued result before asserting on the
GPU line / suggestion. Persistence is verified by pointing
YAMABIKO_APPDATA at a tmp_path and reading back user_overrides.yaml
after validatePage.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pytestqt")


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def make_page(qtbot):
    """QuantTierPage factory whose teardown stops the detection thread.

    The page starts a GPU-detection QThread in __init__; pytest-qt destroys
    widgets without the wizard's release_background_threads() hook, and
    destroying a page whose child QThread still runs is a Qt 6 qFatal
    process abort — so every page built here gets shutdown_thread() at
    test end (same pattern as the SmokePage fixture).
    """
    from yamabiko.wizard_pages.quant_tier import QuantTierPage

    pages = []

    def _make():
        page = QuantTierPage()
        qtbot.addWidget(page)
        pages.append(page)
        return page

    yield _make
    for page in pages:
        page.shutdown_thread(wait_ms=5000)


def _read_overrides(tmp_path: Path) -> dict:
    p = tmp_path / "user_overrides.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _patch_gpu(monkeypatch, vram_mib, name="NVIDIA GeForce RTX Test"):
    """Stub yamabiko.gpu's detection helpers for deterministic tests."""
    from yamabiko import gpu

    monkeypatch.setattr(gpu, "detect_vram_mib", lambda: vram_mib)
    monkeypatch.setattr(
        gpu, "detect_gpu_name",
        lambda: (name if vram_mib is not None else None),
    )


def _wait_detected(qtbot, page) -> None:
    """Block until the async detection result has replaced "Checking…" on
    the GPU line (queued cross-thread signal processed)."""
    qtbot.waitUntil(lambda: page.gpu_label.text() != "Checking\u2026", timeout=5000)


def test_default_radio_autoselected_on_16gb_card(
    qtbot, monkeypatch, appdata, make_page,
):
    _patch_gpu(monkeypatch, 16303, name="NVIDIA GeForce RTX 5080")

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.default_radio.isChecked()
    assert not page.small_radio.isChecked()
    assert page.selected_tier() == "default"
    assert page.gpu_label.text() == "NVIDIA GeForce RTX 5080, 16 GB"


def test_small_radio_autoselected_on_8gb_card(
    qtbot, monkeypatch, appdata, make_page,
):
    """An 8 GB card can report a little under 8192 MiB (8188 here): it is
    the 8 GB its card line shows, so it meets the Smaller tier's floor.

    Seeded defect: compare raw MiB with the floor (``vram_mib >= 8 * 1024``)
    -> the requirement line shows -> red.
    """
    _patch_gpu(monkeypatch, 8188, name="NVIDIA GeForce RTX 4060")

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.small_radio.isChecked()
    assert page.selected_tier() == "small"
    assert page.gpu_label.text() == "NVIDIA GeForce RTX 4060, 8 GB"
    assert page.gpu_help.isHidden()


def test_vram_detection_failure_shows_message_and_defaults(
    qtbot, monkeypatch, appdata, make_page,
):
    _patch_gpu(monkeypatch, None)

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.default_radio.isChecked()
    assert page.selected_tier() == "default"
    # Issue #27: a no-GPU user's first signal must be the wizard saying
    # plainly that an NVIDIA card is required -- not an OOM/CUDA crash
    # after a 5 GB download.
    assert page.gpu_label.text() == "None found"
    assert not page.gpu_help.isHidden()
    assert page.gpu_help.text() == (
        "Yamabiko needs an NVIDIA graphics card with 8 GB of video memory or more."
    )
    # No card, no recommendation: neither row is marked.
    assert "Recommended" not in page.default_radio.accessibleName()
    assert "Recommended" not in page.small_radio.accessibleName()


def test_a_card_below_the_smaller_tiers_floor_gets_no_recommendation_and_the_requirement_line(
    qtbot, monkeypatch, appdata, make_page,
):
    """Decision 150: a card under the Smaller tier's 8 GB floor fits
    neither tier, so neither row says "Recommended for your card" and the
    requirement line shows, as with no card.

    Seeded defects: mark suggest_quant_tier's tier whenever VRAM was
    detected (Smaller marked, the help line hidden) -> red; return before
    the selection moves when the card is below the floor (Standard stays)
    -> red, since the selection still follows suggest_quant_tier.
    """
    _patch_gpu(monkeypatch, 6144, name="NVIDIA GeForce RTX 2060")

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.gpu_label.text() == "NVIDIA GeForce RTX 2060, 6 GB"
    assert not page.gpu_help.isHidden()
    assert "Recommended" not in page.default_radio.accessibleName()
    assert "Recommended" not in page.small_radio.accessibleName()
    assert page.selected_tier() == "small"


# ---------------------------------------------------------------------------
# P2: the detection shell-outs must never run synchronously on the GUI
# thread inside __init__ — YamabikoWizard constructs this page BEFORE
# exec() shows anything, and gpu.py's two subprocess calls carry 5 s
# timeouts each (~10 s of no-window startup on a wedged driver).
# ---------------------------------------------------------------------------


def test_construction_returns_immediately_with_conservative_default_while_detection_blocks(
    qtbot, monkeypatch, appdata, make_page,
):
    """With the detectors wedged (simulated driver hang), the constructor
    must return promptly with the conservative Standard tier selected, the
    GPU line on "Checking…", and the page already complete — the wizard can
    show and even advance without the probe.

    Seeded defect: call detect_vram_mib()/detect_gpu_name() synchronously
    in __init__ -> construction blocks on the gate until its 10 s timeout
    -> the elapsed-time assertion goes red.
    """
    from yamabiko import gpu

    entered = threading.Event()
    gate = threading.Event()

    def _wedged_vram():
        entered.set()
        gate.wait(timeout=10.0)
        return None

    monkeypatch.setattr(gpu, "detect_vram_mib", _wedged_vram)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: None)

    t0 = time.monotonic()
    try:
        page = make_page()
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, (
            f"QuantTierPage construction blocked {elapsed:.1f}s on GPU "
            f"detection; it must return immediately"
        )
        assert page.default_radio.isChecked(), (
            "conservative default tier must be selected before detection"
        )
        assert page.gpu_label.text() == "Checking\u2026"
        assert page.isComplete()
        assert entered.wait(5.0), "detection never started on the worker"
    finally:
        gate.set()


def test_detection_runs_off_the_gui_thread(qtbot, monkeypatch, appdata, make_page):
    """The nvidia-smi probes execute on a worker thread; a synchronous
    subprocess on the GUI thread is exactly the freeze this page fixes.

    Seeded defect: run the detectors inline on the page -> recorded thread
    id equals the GUI thread's -> red.
    """
    from yamabiko import gpu

    thread_ids: set = set()

    def _recording_vram():
        thread_ids.add(threading.get_ident())
        return 16303

    monkeypatch.setattr(gpu, "detect_vram_mib", _recording_vram)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5080")

    gui_thread_id = threading.get_ident()
    page = make_page()
    _wait_detected(qtbot, page)
    assert thread_ids, "detector never called"
    assert gui_thread_id not in thread_ids, (
        "GPU detection ran on the GUI thread; it must execute on a worker"
    )


def test_late_detection_result_does_not_override_a_user_pick(
    qtbot, monkeypatch, appdata, make_page,
):
    """A user who clicks a tier while detection is still running must keep
    that pick when the (contradicting) result lands.

    Seeded defect: drop the _user_picked guard in _on_gpu_detected -> the
    16 GB suggestion flips the radios back to Standard -> red.
    """
    from yamabiko import gpu

    gate = threading.Event()

    def _slow_vram():
        gate.wait(timeout=10.0)
        return 16303  # suggests 'default', contradicting the user's pick

    monkeypatch.setattr(gpu, "detect_vram_mib", _slow_vram)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5080")

    page = make_page()
    page.small_radio.click()  # USER interaction while the probe is pending
    gate.set()
    _wait_detected(qtbot, page)

    assert page.small_radio.isChecked(), (
        "late detection result overrode the user's explicit tier pick"
    )
    assert page.selected_tier() == "small"
    # The GPU line still updates -- only the suggestion defers to the user.
    assert page.gpu_label.text() == "NVIDIA GeForce RTX 5080, 16 GB"


def test_an_arrow_key_pick_counts_as_a_user_pick_that_late_detection_keeps(
    qtbot, monkeypatch, appdata, make_page,
):
    """Decision 68: an arrow-key move (the control's valueChanged) is a
    user pick like a click, so a late detection result does not move it.

    Seeded defect: drop the valueChanged connection -> the 16 GB
    suggestion moves the choice back to Standard -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from yamabiko import gpu

    gate = threading.Event()

    def _slow_vram():
        gate.wait(timeout=10.0)
        return 16303  # suggests 'default', contradicting the user's pick

    monkeypatch.setattr(gpu, "detect_vram_mib", _slow_vram)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5080")

    page = make_page()
    QTest.keyClick(page.default_radio, Qt.Key_Down)  # USER moves to Smaller
    assert page.selected_tier() == "small"
    gate.set()
    _wait_detected(qtbot, page)

    assert page.selected_tier() == "small", (
        "late detection result overrode the user's arrow-key pick"
    )


def test_shutdown_thread_detaches_wedged_detection_thread_into_keeper(
    qtbot, monkeypatch, appdata, make_page,
):
    """Cancelling the wizard while nvidia-smi is wedged must detach the
    detection thread (no Qt parent, parked in thread_keeper) — never
    destroy or indefinitely join it. The detached thread exits on its own
    once gpu.py's subprocess timeouts fire.

    Seeded defect: remove the thread_keeper.release call → red.
    """
    from yamabiko import gpu
    from yamabiko.ui import thread_keeper

    entered = threading.Event()
    gate = threading.Event()

    def _wedged_vram():
        entered.set()
        gate.wait(timeout=10.0)
        return None

    monkeypatch.setattr(gpu, "detect_vram_mib", _wedged_vram)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: None)

    page = make_page()
    assert entered.wait(5.0), "detection never started"
    thread = page._detect_thread
    assert thread is not None
    try:
        page.shutdown_thread(wait_ms=100)
        assert thread.isRunning(), "wedged detection thread should still run"
        assert thread.parent() is None
        assert any(t is thread for t, _ in thread_keeper._LINGERING)
        assert page._detect_thread is None
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_shutdown_thread_joins_finished_detection_thread_without_detach(
    qtbot, monkeypatch, appdata, make_page,
):
    """With the detection already finished, shutdown_thread() joins the
    worker and adds nothing to thread_keeper (no leak on clean close).

    Seeded defect: release unconditionally → registry gains an entry → red.
    """
    from yamabiko.ui import thread_keeper

    _patch_gpu(monkeypatch, 16303)
    page = make_page()
    _wait_detected(qtbot, page)

    before = len(thread_keeper._LINGERING)
    page.shutdown_thread(wait_ms=5000)
    assert len(thread_keeper._LINGERING) == before
    assert page._detect_thread is None


# ---------------------------------------------------------------------------
# Saved-override seeding: the upgrade path forces completed installs back
# through the wizard, and the async detection suggestion used to flip the
# radios over a deliberate saved `translation.quant_tier: small` on
# click-through (wrong ~5.2 GB download + runtime VRAM pressure on the 8 GB
# card the user overrode for). The seam is the real user_overrides.yaml
# read: these tests write that file into the YAMABIKO_APPDATA tmp dir.
# ---------------------------------------------------------------------------


def _write_overrides(appdata: Path, data: dict) -> None:
    (appdata / "user_overrides.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8",
    )


def test_saved_small_override_preselects_small_before_detection(
    qtbot, monkeypatch, appdata, make_page,
):
    """A saved `translation.quant_tier: small` override seeds the radios at
    construction, before any detection result lands.

    Seeded defect: drop the saved-tier seeding from __init__ (always seed
    the Standard radio) -> small is not checked at construction -> red.
    """
    _write_overrides(appdata, {"translation": {"quant_tier": "small"}})
    _patch_gpu(monkeypatch, 16303, name="NVIDIA GeForce RTX 5080")

    page = make_page()
    assert page.small_radio.isChecked(), (
        "saved small override was not preselected at construction"
    )
    assert page.selected_tier() == "small"


def test_detection_suggestion_does_not_flip_a_saved_small_override(
    qtbot, monkeypatch, appdata, make_page,
):
    """The async detection result may update the GPU line and the note but
    must NOT move the radios off a saved override -- a 16 GB card suggests
    'default', contradicting the deliberate saved 'small'.

    Seeded defect: drop the saved-tier guard in _on_gpu_detected -> the
    16 GB suggestion flips the radios back to Standard -> red.
    """
    _write_overrides(appdata, {"translation": {"quant_tier": "small"}})
    _patch_gpu(monkeypatch, 16303, name="NVIDIA GeForce RTX 5080")

    page = make_page()
    _wait_detected(qtbot, page)

    # The GPU line still informs...
    assert page.gpu_label.text() == "NVIDIA GeForce RTX 5080, 16 GB"
    # ...but the saved selection stands.
    assert page.small_radio.isChecked(), (
        "detection suggestion overrode a saved quant_tier override"
    )
    assert page.selected_tier() == "small"


def test_saved_small_override_round_trips_through_validate_page_on_click_through(
    qtbot, monkeypatch, appdata, make_page,
):
    """Click-through (no user radio interaction) after the detection lands
    must persist the SAME saved tier: validatePage re-writes the coupled
    pair for 'small', not the 16 GB card's 'default' suggestion.

    Seeded defect: drop the seeding OR the suggestion suppression -> the
    untouched validatePage prunes the small override -> red.
    """
    _write_overrides(appdata, {"translation": {"quant_tier": "small"}})
    _patch_gpu(monkeypatch, 16303, name="NVIDIA GeForce RTX 5080")

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.validatePage() is True

    assert _read_overrides(appdata) == {
        "translation": {"quant_tier": "small"},
        "asr": {"faster_whisper": {"compute_type": "int8_float16"}},
    }, "click-through did not round-trip the saved small tier"


def test_saved_default_tier_survives_detection_on_8gb_card(
    qtbot, monkeypatch, appdata, make_page,
):
    """A user who saved Standard (``default``) and re-enters the wizard on an
    8 GB card must keep Standard: the detection suggestion (small) may only
    fill in a tier the user never chose.

    Seeded defects: let write_quant_tier prune ``quant_tier: default``, or
    read the merged config in _saved_quant_tier (where an unsaved and a
    saved default look the same) -> the suggestion flips the radio to small.
    """
    from yamabiko.config_writer import write_quant_tier

    write_quant_tier("default")
    _patch_gpu(monkeypatch, 8192, name="NVIDIA GeForce RTX 4060")

    page = make_page()
    _wait_detected(qtbot, page)

    assert page.gpu_label.text() == "NVIDIA GeForce RTX 4060, 8 GB"
    assert page.default_radio.isChecked(), (
        "the 8 GB detection suggestion replaced a saved Standard tier"
    )
    assert page.selected_tier() == "default"


def test_quant_config_read_failure_falls_back_to_detection_suggestion(
    qtbot, monkeypatch, appdata, make_page,
):
    """A broken overrides read fails open: no claimed saved tier,
    construction does not raise, and the detection suggestion still refines
    the conservative default (8 GB card -> small).

    Seeded defect: let _saved_quant_tier propagate the exception -> page
    construction errors -> red.
    """
    def _boom(_path):
        raise OSError("user_overrides.yaml unreadable")

    monkeypatch.setattr("yamabiko.config._read_yaml", _boom)
    _patch_gpu(monkeypatch, 8192, name="NVIDIA GeForce RTX 4060")

    page = make_page()  # must NOT raise
    assert page.default_radio.isChecked(), "conservative default not seeded"
    _wait_detected(qtbot, page)
    assert page.small_radio.isChecked(), (
        "detection suggestion was suppressed although no saved override "
        "exists (config read failed)"
    )


# ---------------------------------------------------------------------------
# Persistence (validatePage) — unchanged contract, now driven after the
# async detection settles so a late suggestion can't race the assertions.
# ---------------------------------------------------------------------------


def test_validate_page_writes_default_tier(qtbot, monkeypatch, appdata, make_page):
    """User keeps the auto-selected 'default' radio; validatePage persists it.

    The tier key is written even though 'default' equals the shipped
    default, so a later wizard run can tell a saved Standard pick from no
    pick. The compute type (float16, also the shipped default) stays
    pruned.
    """
    _patch_gpu(monkeypatch, 16303)

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.validatePage() is True

    assert _read_overrides(appdata) == {"translation": {"quant_tier": "default"}}


def test_validate_page_writes_small_tier_when_user_overrides(
    qtbot, monkeypatch, appdata, make_page,
):
    """User flips to Smaller; validatePage writes it to user_overrides.yaml."""
    _patch_gpu(monkeypatch, 16303)

    page = make_page()
    # Wait out the async suggestion first so this is a pure user override
    # (the user-click-during-detection race has its own test above).
    _wait_detected(qtbot, page)
    page.small_radio.click()
    assert page.selected_tier() == "small"

    assert page.validatePage() is True

    overrides = _read_overrides(appdata)
    # Coupled write: both translation.quant_tier=small AND
    # asr.faster_whisper.compute_type=int8_float16 land in overrides.
    assert overrides == {
        "translation": {"quant_tier": "small"},
        "asr": {"faster_whisper": {"compute_type": "int8_float16"}},
    }


def test_validate_page_switches_back_to_default_records_default(
    qtbot, monkeypatch, appdata, make_page,
):
    """User lands on small, switches back to default -> the tier records
    'default' and the small tier's compute type is pruned."""
    _patch_gpu(monkeypatch, 8192)

    page = make_page()
    _wait_detected(qtbot, page)
    assert page.small_radio.isChecked(), "8 GB card must auto-suggest small"
    # First pass: persist 'small' (what was auto-selected on the 8 GB card).
    page.validatePage()
    assert _read_overrides(appdata) == {
        "translation": {"quant_tier": "small"},
        "asr": {"faster_whisper": {"compute_type": "int8_float16"}},
    }

    # User backs up and switches to default.
    page.default_radio.click()
    page.validatePage()
    # The compute type matches the shipped default and is pruned; the tier
    # is kept as an explicit choice.
    assert _read_overrides(appdata) == {"translation": {"quant_tier": "default"}}


def test_validate_page_uses_batched_quant_pair_writer(
    qtbot, monkeypatch, appdata, make_page,
):
    """The page must persist the tier through the coupled batched writer
    (write_quant_tier), not two independent write_override calls -- so the
    quant_tier / compute_type pair lands together or not at all.

    Seeded defect: revert validatePage to two write_override calls ->
    write_quant_tier is never invoked -> red.
    """
    _patch_gpu(monkeypatch, 16303)
    from yamabiko import config_writer

    seen: list = []
    real = config_writer.write_quant_tier
    monkeypatch.setattr(
        config_writer, "write_quant_tier",
        lambda tier: seen.append(tier) or real(tier),
    )

    page = make_page()
    _wait_detected(qtbot, page)
    page.small_radio.click()
    assert page.validatePage() is True
    assert seen == ["small"], (
        f"validatePage did not route through write_quant_tier: {seen!r}"
    )


def test_validate_page_blocks_on_a_failed_pair_write_and_retry_writes_again(
    qtbot, monkeypatch, appdata, make_page,
):
    """L12: a failed write of the coupled pair shows "Couldn't save your
    settings" (SaveFailedDialog, titled "Yamabiko setup") and keeps the user
    on Quality (validatePage False, so Download never fetches the wrong
    tier); Retry in that dialog writes again.

    Seeded defects: restore the `except Exception: log; return True` swallow
    -> no dialog, True -> red; ignore the dialog's Retry -> the second
    validatePage stays False -> red.
    """
    _patch_gpu(monkeypatch, 16303)
    from yamabiko import config_writer
    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog

    real_write = config_writer.write_quant_tier

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(config_writer, "write_quant_tier", _boom)
    asked: list = []

    def _ask(parent, palette, path, error, *, title):
        asked.append((str(error), title))
        if len(asked) == 1:
            return False  # Cancel
        monkeypatch.setattr(config_writer, "write_quant_tier", real_write)  # cause cleared
        return True  # Retry

    monkeypatch.setattr(SaveFailedDialog, "ask", _ask)

    page = make_page()
    _wait_detected(qtbot, page)
    page.small_radio.click()  # non-default: the write fires
    assert page.validatePage() is False, (
        "validatePage must block the advance when the pair write fails"
    )
    assert page.validatePage() is True, "Retry did not write again"
    assert asked == [("disk full", "Yamabiko setup"), ("disk full", "Yamabiko setup")]
    assert _read_overrides(appdata)["translation"]["quant_tier"] == "small"


def test_quality_rows_carry_their_uia_handles_and_the_fitting_tier_note(
    qtbot, monkeypatch, appdata, make_page,
):
    """L10: the two rows keep the objectNames the frozen E2E clicks
    (quant_default_radio / quant_small_radio); the tier that fits the
    detected card says "Recommended for your card" (part of its UIA name)
    without moving a saved pick; the primary reads "Start download".

    Seeded defects: drop the objectNames -> findChild returns None -> red;
    put the note on the selected row instead of the suggested one (here a
    saved Standard on an 8 GB card) -> red.
    """
    from PySide6.QtWidgets import QAbstractButton, QWizard

    from yamabiko.config_writer import write_quant_tier

    write_quant_tier("default")  # a saved Standard pick
    _patch_gpu(monkeypatch, 8192, name="NVIDIA GeForce RTX 4060")

    page = make_page()
    _wait_detected(qtbot, page)

    small = page.findChild(QAbstractButton, "quant_small_radio")
    standard = page.findChild(QAbstractButton, "quant_default_radio")
    assert small is page.small_radio and standard is page.default_radio
    assert small.accessibleName().endswith(", Recommended for your card")
    assert "Recommended" not in standard.accessibleName()
    assert page.selected_tier() == "default"
    assert page.buttonText(QWizard.WizardButton.NextButton) == "Start download"
