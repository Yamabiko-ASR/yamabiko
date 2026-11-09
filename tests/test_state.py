"""Tests for the runtime AppState shared across pipeline stages."""
from __future__ import annotations

import pytest

from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from yamabiko.state import AppState, VALID_GENRES


def test_default_values():
    s = AppState()
    assert s.genre == "variety"
    assert s.paused is False


def test_set_genre_valid():
    s = AppState()
    s.set_genre("news")
    assert s.genre == "news"


def test_set_genre_rejects_invalid():
    s = AppState()
    with pytest.raises(ValueError):
        s.set_genre("sports")
    assert s.genre == "variety"  # unchanged


def test_construction_coerces_invalid_genre_to_variety(caplog):
    """A typo'd genre in user_overrides.yaml (AppState built directly from
    the merged config, bypassing set_genre) must coerce to 'variety' with
    a warning, not silently degrade the genre-steering prompt note.

    Seeded defect: remove AppState.__post_init__ -> genre stays 'anme'
    -> this fails.
    """
    import logging
    with caplog.at_level(logging.WARNING):
        s = AppState(genre="anme")
    assert s.genre == "variety"
    assert any("invalid genre" in r.message.lower() for r in caplog.records), (
        "invalid genre coerced but no warning logged"
    )


def test_construction_keeps_valid_genre():
    assert AppState(genre="anime").genre == "anime"


def test_pause_resume_flags():
    s = AppState()
    s.pause()
    assert s.paused is True
    s.resume()
    assert s.paused is False


def test_valid_genres_expose_constants():
    assert "variety" in VALID_GENRES
    assert "drama" in VALID_GENRES
    assert "news" in VALID_GENRES
    assert "anime" in VALID_GENRES


def test_custom_initial_genre():
    s = AppState(genre="anime")
    assert s.genre == "anime"


def test_update_style_merges_and_stores():
    s = AppState()
    snap = s.snapshot()
    assert snap["style"] == STYLE_DEFAULTS
    payload = s.update_style({"jp_size": 22, "bg_opacity": 200})
    assert payload["type"] == "style"
    assert payload["jp_size"] == 22
    assert payload["bg_opacity"] == 200
    assert payload["en_size"] == STYLE_DEFAULTS["en_size"]
    assert s.snapshot()["style"]["jp_size"] == 22


def test_set_style_merges_without_broadcast():
    s = AppState()
    s.set_style({"jp_size": 30})
    assert s.snapshot()["style"]["jp_size"] == 30


def test_set_series_updates_fields_and_snapshot():
    st = AppState()
    assert st.series_media_id is None
    assert st.series_label is None
    assert st.snapshot()["series_media_id"] is None
    st.set_series(182205, "That Time... Slime (2024)")
    assert st.series_media_id == 182205
    assert st.series_label == "That Time... Slime (2024)"
    snap = st.snapshot()
    assert snap["series_media_id"] == 182205
    assert snap["series_label"] == "That Time... Slime (2024)"


def test_set_series_none_clears():
    st = AppState()
    st.set_series(1, "x")
    st.set_series(None, None)
    assert st.snapshot()["series_media_id"] is None
    assert st.snapshot()["series_label"] is None


def test_set_series_bumps_epoch_even_for_the_same_media_id():
    """The pipeline hot loop reloads the gazetteer on an epoch change, not on
    a media-id change. So a re-pick of the ALREADY-active series must still
    bump the epoch: the first pick may have recorded an empty AniList result
    (gazetteer=None), and a later re-pick of that same id has to reload the
    now-built cache. If set_series only changed state when the id differed,
    the re-pick would be a silent no-op and hints would stay absent all
    session.

    Seeded defect: guard `self.series_epoch += 1` behind `if media_id !=
    self.series_media_id:` (or drop the increment) -> the second set_series
    with the same id leaves the epoch unchanged -> red.
    """
    s = AppState()
    s.set_series(7, "X")
    first = s.series_epoch
    s.set_series(7, "X")  # re-pick of the SAME series must still be a change
    assert s.series_epoch == first + 1


def test_state_frame_is_the_single_wire_shape():
    """state_frame() is THE builder every state broadcast routes through
    (command handlers, the on-connect snapshot, the pause hotkey). It must
    carry exactly the five state keys plus type -- crucially the series
    keys -- so a producer can't hand-build a frame that omits the series
    keys and makes ws_client read the tray's Series line as cleared. A
    picked series carries its Japanese title (the overlay banner's text).

    Exact-equality: dropping/renaming any key OR leaking an extra key (e.g.
    the internal ``style`` dict from snapshot) fails.

    Seeded-defect check: drop the ``series_native`` entry from state_frame()
    -> the dict no longer equals the expected shape -> red.
    """
    s = AppState(genre="anime")
    assert s.state_frame()["series_native"] is None
    s.set_series(7, "Frieren (2023)", "葬送のフリーレン")
    s.pause()
    assert s.state_frame() == {
        "type": "state",
        "genre": "anime",
        "paused": True,
        "series_media_id": 7,
        "series_label": "Frieren (2023)",
        "series_native": "葬送のフリーレン",
    }


def test_style_frame_carries_reduce_motion():
    """The style frame tells the browser page whether to animate: seeded
    false, then whatever set_reduce_motion set, on both the on-connect frame
    and the update_style broadcast.

    Seeded defect: build the frame without reduce_motion -> red."""
    from yamabiko.state import AppState

    state = AppState()
    assert state.style_frame()["reduce_motion"] is False
    state.set_reduce_motion(True)
    assert state.style_frame()["reduce_motion"] is True
    assert state.update_style({"jp_size": 20})["reduce_motion"] is True
    assert state.style_frame()["type"] == "style"
