from __future__ import annotations

import copy

import pytest
from PySide6.QtWidgets import QApplication

from yamabiko import config_writer, paths
from yamabiko.config import load_full_config
from yamabiko.overlay_app import settings_transaction
from yamabiko.overlay_app.overlay_window import OverlayWindow
from yamabiko.overlay_app.settings_transaction import (
    SettingsCommit,
    persist_settings,
)
from yamabiko.overlay_app.settings_window import SettingsWindow
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    return tmp_path


@pytest.fixture(autouse=True)
def _cached_quant_tier(monkeypatch):
    from yamabiko.overlay_app import model_fetch

    monkeypatch.setattr(model_fetch, "tier_cached", lambda _tier: True)


def _production_window(appdata, *, devices=None, **kwargs) -> SettingsWindow:
    if devices is None:
        devices = [
            {"name": "default", "index": 0},
            {"name": "CABLE Output", "index": 1},
        ]
    return SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: devices,
        config_path=appdata / "user_overrides.yaml",
        **kwargs,
    )


def test_persist_settings_writes_complete_commit_in_one_replace(
    appdata, monkeypatch,
):
    writes = []
    real = config_writer._atomic_write
    monkeypatch.setattr(
        config_writer,
        "_atomic_write",
        lambda path, data: (
            writes.append(copy.deepcopy(data)),
            real(path, data),
        )[1],
    )
    commit = SettingsCommit(
        updates={
            ("audio", "device"): "CABLE Output",
            ("genre",): "anime",
            ("overlay", "style", "jp_size"): 20,
        },
        remove_paths=(),
        style={"jp_size": 20},
        position=None,
    )

    persist_settings(commit)

    assert load_full_config()["audio"]["device"] == "CABLE Output"
    assert load_full_config()["overlay"]["style"]["jp_size"] == 20
    assert len(writes) == 1


def test_atomic_save_failure_keeps_window_open_and_emits_no_committed_state(
    qapp, appdata, monkeypatch,
):
    window = _production_window(appdata, save_failed=lambda *args: False)
    styles = []
    positions = []
    themes = []
    auto_genres = []
    window.styleChanged.connect(styles.append)
    window.positionApplied.connect(positions.append)
    window.themeChanged.connect(themes.append)
    window.autoGenreChanged.connect(auto_genres.append)
    monkeypatch.setattr(
        settings_transaction,
        "persist_settings",
        lambda _commit: (_ for _ in ()).throw(OSError(28, "disk full")),
    )

    window._ok()

    assert window._closed is False
    assert styles == []
    assert positions == []
    assert themes == []
    assert auto_genres == []


def test_cancel_after_live_preview_restores_committed_style_without_writing(
    qapp, appdata,
):
    paths.user_overrides_path().write_text("genre: anime\n", encoding="utf-8")
    window = _production_window(appdata)
    previews = []
    ended = []
    window.stylePreview.connect(previews.append)
    window.previewEnded.connect(lambda: ended.append(True))
    before = paths.user_overrides_path().read_bytes()

    window._jp_size.setValue(window._jp_size.value() + 2)
    window._emit_preview()
    window.close()

    assert previews
    assert ended == [True]
    assert paths.user_overrides_path().read_bytes() == before


def test_missing_saved_audio_device_is_visible_and_no_touch_ok_preserves_it(
    qapp, appdata,
):
    config_writer.write_override(["audio", "device"], "Missing DAC")
    window = _production_window(appdata)

    assert "Missing DAC" in window._audio_device_warning.text()
    assert not window._audio_device_warning.isHidden()
    assert window.audio_device_combo.currentText() == "Missing DAC (unavailable)"

    window._ok()

    assert load_full_config()["audio"]["device"] == "Missing DAC"


def test_picking_the_shown_device_after_unavailable_saved_device_is_committed(
    qapp, appdata,
):
    """With the saved device unplugged, the combo used to sit on the
    "default" row while the warning said "choose another device"; picking
    that already-selected row left the index unchanged, so OK saved nothing
    and the runtime kept falling back to the missing device. The missing
    device now has its own "(unavailable)" first row, so any real pick is a
    change.

    Seeded defect: drop the "(unavailable)" row insertion (back to
    selecting "default") -> picking "default" is a no-op -> the saved
    device stays "Missing DAC" -> red.
    """
    config_writer.write_override(["audio", "device"], "Missing DAC")
    window = _production_window(appdata)

    window.audio_device_combo.setCurrentIndex(
        window.audio_device_combo.findText("default")
    )
    window._ok()

    assert load_full_config()["audio"]["device"] == "default"


def test_position_apply_after_transaction_moves_without_second_disk_write(qapp):
    writes = []
    window = OverlayWindow(
        style=None,
        position=None,
        on_position_changed=writes.append,
    )
    try:
        target = {
            "screen": "primary",
            "v_offset_pct": 73.0,
            "h_offset_pct": 44.0,
            "left_offset_pct": 6.0,
        }

        window.set_position(target, persist=False)

        assert window.current_position() == target
        assert writes == []
    finally:
        window.close()
