"""Integration tests for /ws command routing into AppState."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yamabiko.state import AppState


# Every /ws client must present the per-launch session token.
_TOKEN = "test-session-token"
_WS = f"/ws?t={_TOKEN}"


def _get_app(state: AppState):
    from yamabiko.display.browser_server import build_app
    return build_app(state=state, bound_port=8765, ws_token=_TOKEN)


def test_pause_command_mutates_state():
    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.send_json({"type": "command", "action": "pause"})
        # Bounce a round-trip so the server processes our message before we assert.
        import anyio
        import asyncio
        # Wait briefly for server to process (TestClient runs both sides in one process).
        # Use an echo command to block until prior command is handled: send a noop and
        # wait for state to settle.
        for _ in range(10):
            if state.paused:
                break
            import time
            time.sleep(0.01)
    assert state.paused is True


def test_resume_command_clears_paused():
    state = AppState()
    state.pause()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.send_json({"type": "command", "action": "resume"})
        import time
        for _ in range(10):
            if not state.paused:
                break
            time.sleep(0.01)
    assert state.paused is False


def test_set_genre_command(tmp_path, monkeypatch):
    # YAMABIKO_APPDATA isolation: the set_genre handler persists the
    # genre via the sparse override writer; without this the test
    # would write the developer's real user_overrides.yaml.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.send_json({"type": "command", "action": "set_genre", "genre": "news"})
        import time
        for _ in range(10):
            if state.genre == "news":
                break
            time.sleep(0.01)
    assert state.genre == "news"


def test_set_genre_command_persists_to_user_overrides(tmp_path, monkeypatch):
    """A genre chosen from the tray/browser UI must survive a
    restart. main.py re-seeds AppState(genre=cfg.get("genre", ...)) from
    the merged config on every launch, so the set_genre handler has to
    write the `genre` override -- in-memory state alone silently reverts
    the user's choice next launch.

    Seeded defect: delete the write_override(["genre"], genre) call in
    _handle_command's set_genre branch -> no overrides file appears ->
    red.
    """
    import time

    import yaml

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths

    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    overrides = paths.user_overrides_path()
    with client.websocket_connect(_WS) as ws:
        ws.send_json({"type": "command", "action": "set_genre", "genre": "news"})
        deadline = time.time() + 2.0
        while not overrides.exists() and time.time() < deadline:
            time.sleep(0.01)
    assert overrides.exists(), (
        "set_genre must persist the genre override so it survives a "
        "restart; no user_overrides.yaml was written"
    )
    data = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    assert data["genre"] == "news"


def test_invalid_genre_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths

    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.send_json({"type": "command", "action": "set_genre", "genre": "sports"})
        import time
        time.sleep(0.05)
    assert state.genre == "variety"  # unchanged
    # A REJECTED genre must not be persisted either.
    assert not paths.user_overrides_path().exists()


def test_unknown_action_is_ignored():
    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.send_json({"type": "command", "action": "self_destruct"})
        import time
        time.sleep(0.05)
    assert state.paused is False
    assert state.genre == "variety"


def test_build_app_no_state_arg_wires_default_app_state_and_manager():
    """build_app() with no args wires BOTH app_state AND manager.

    The prior test only checked ``app_state`` existence; ``manager``
    (the ConnectionManager used by every WebSocket handler in
    browser_server.py) was unasserted.

    If ``manager`` were absent, every real WebSocket connection
    would raise AttributeError at handler time -- invisible to the
    old test. Now asserted explicitly, with specific typed checks
    so the seeded defect is obvious.

    Seeded-defect check: delete the ``app.state.manager = ...``
    assignment in build_app() -> this test fails.
    """
    from yamabiko.display.broadcast import ConnectionManager
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    app = build_app(bound_port=8765, ws_token=_TOKEN)
    # app_state present, defaulted, typed
    assert hasattr(app.state, "app_state")
    assert isinstance(app.state.app_state, AppState)
    assert app.state.app_state.genre == "variety"
    # manager present and typed -- this is the line the old test missed
    assert hasattr(app.state, "manager"), (
        "app.state.manager missing -- WebSocket handlers would raise "
        "AttributeError on connect. build_app() must wire it."
    )
    assert isinstance(app.state.manager, ConnectionManager)


# ---------------------------------------------------------------------------
# toggle_overlay command
# ---------------------------------------------------------------------------

# NOTE: These tests cover command shape + broadcast routing. The
# visibility-toggle behavior itself (Qt `setVisible`, CSS `.hidden`
# class) is exercised by manual / integration testing — we don't have
# a Qt or headless-browser harness in this suite.
def test_make_toggle_overlay_cmd_shape():
    """make_toggle_overlay_cmd() returns the expected dict."""
    from yamabiko.display.broadcast import make_toggle_overlay_cmd
    cmd = make_toggle_overlay_cmd()
    assert cmd == {"type": "toggle_overlay"}


def test_toggle_overlay_broadcast_reaches_connected_client():
    """Broadcast of toggle_overlay reaches all connected WS clients."""
    import asyncio
    import threading

    received: list[dict] = []

    class _FakeWS:
        async def send_json(self, payload: dict) -> None:
            received.append(payload)

    from yamabiko.display.broadcast import ConnectionManager, make_toggle_overlay_cmd

    async def _run():
        mgr = ConnectionManager()
        ws = _FakeWS()
        mgr.add(ws)
        await mgr.broadcast(make_toggle_overlay_cmd())

    asyncio.run(_run())
    assert received == [{"type": "toggle_overlay"}]


# ---------------------------------------------------------------------------
# set_series command (persist active-series pointer + broadcast state)
# ---------------------------------------------------------------------------


def _drain_initial_frames(ws) -> None:
    """The /ws handler sends a state, a style and a problems frame on connect.
    Read all three so the next receive_json() returns the post-command
    broadcast."""
    ws.receive_json()  # initial state
    ws.receive_json()  # initial style
    ws.receive_json()  # initial problems


def test_apply_style_updates_runtime_without_second_disk_write(monkeypatch):
    from yamabiko.display import browser_server

    writes = []
    monkeypatch.setattr(
        browser_server,
        "write_overlay_style",
        lambda style: writes.append(dict(style)),
    )
    state = AppState()
    app = _get_app(state)
    client = TestClient(app)

    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({
            "type": "command",
            "action": "apply_style",
            "style": {"jp_size": 30},
        })
        # Pause is a response boundary: the style broadcast must arrive first.
        ws.send_json({"type": "command", "action": "pause"})
        frame = ws.receive_json()

    assert frame["type"] == "style"
    assert frame["jp_size"] == 30
    assert state.snapshot()["style"]["jp_size"] == 30
    assert writes == []


def test_apply_style_reduce_motion_updates_the_broadcast_style_frame(monkeypatch):
    """apply_style's top-level reduce_motion (sent by the tray after a
    Settings Apply) reaches AppState and rides the broadcast style frame
    to the browser page; a non-bool is ignored (0 would read as false).

    Seeded defects: drop the reduce_motion branch -> frame stays false ->
    red; drop the isinstance(..., bool) guard -> 0 turns it off -> red."""
    from yamabiko.display import browser_server

    monkeypatch.setattr(browser_server, "write_overlay_style", lambda style: None)
    state = AppState()
    app = _get_app(state)
    client = TestClient(app)

    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({
            "type": "command", "action": "apply_style",
            "style": {"jp_size": 18}, "reduce_motion": True,
        })
        frame = ws.receive_json()
        assert frame["type"] == "style"
        assert frame["reduce_motion"] is True
        assert state.style_frame()["reduce_motion"] is True

        ws.send_json({
            "type": "command", "action": "apply_style",
            "style": {"jp_size": 19}, "reduce_motion": 0,
        })
        frame = ws.receive_json()

    assert frame["jp_size"] == 19
    assert frame["reduce_motion"] is True, "a non-bool reduce_motion was applied"
    assert state.style_frame()["reduce_motion"] is True


def test_set_series_command_mutates_state_persists_pointer_and_broadcasts(
    tmp_path, monkeypatch,
):
    """A palette selection ({"action":"set_series","media_id":N,"label":...})
    must (1) mutate AppState.series_media_id/label, (2) persist the
    active_series.json pointer (so it survives a restart), and (3) broadcast a
    'state' frame carrying the series keys to every connected overlay.

    Mirrors the set_genre routing tests. YAMABIKO_APPDATA is redirected so the
    pointer lands in a tmp dir, not the developer's real app-data.

    Seeded defect: delete the save_active_series(...) call in the set_series
    branch -> no active_series.json appears -> the pointer assert goes red.
    Drop the series keys from _broadcast_state -> the frame assert goes red.
    """
    import json

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths

    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({
            "type": "command", "action": "set_series",
            "media_id": 182205, "label": "Slime (2024)",
            "native": "葬送のフリーレン",
        })
        frame = ws.receive_json()  # broadcast state frame

    assert state.series_media_id == 182205
    assert state.series_label == "Slime (2024)"
    assert state.series_native == "葬送のフリーレン"

    pointer = paths.app_data_dir() / "active_series.json"
    assert pointer.exists(), (
        "set_series must persist active_series.json so the choice survives a "
        "restart; no pointer file was written"
    )
    data = json.loads(pointer.read_text(encoding="utf-8"))
    assert data == {
        "media_id": 182205, "label": "Slime (2024)", "native": "葬送のフリーレン",
    }

    assert frame["type"] == "state"
    assert frame["series_media_id"] == 182205, (
        f"broadcast state frame missing series_media_id; got {frame!r}"
    )
    assert frame["series_label"] == "Slime (2024)"
    assert frame["series_native"] == "葬送のフリーレン"


@pytest.mark.parametrize(
    "bad, why",
    [
        ({"media_id": "182205", "label": "X"}, "media_id is a string"),
        ({"media_id": True, "label": "X"}, "media_id is a bool (int subclass)"),
        ({"media_id": 182205, "label": 5}, "label is not a string"),
        ({"media_id": 182205}, "label missing"),
        ({"label": "X"}, "media_id missing"),
    ],
)
def test_set_series_rejects_bad_payload_leaves_state_and_writes_no_file(
    bad, why, tmp_path, monkeypatch,
):
    """A malformed set_series (non-int/bool media_id or non-str label) must
    leave AppState untouched AND write no pointer file.

    Seeded defect: drop the `not isinstance(media_id, bool)` guard -> the
    bool-media_id case mutates state + writes a file -> red. Drop the whole
    isinstance guard -> the string-media_id case does too -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths

    state = AppState()
    app = _get_app(state)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({"type": "command", "action": "set_series", **bad})
        # Deterministic barrier instead of sleep-and-hope: a valid pause is
        # handled AFTER the bad set_series (single in-order command handler)
        # and pause broadcasts a state frame. Awaiting that frame guarantees
        # the bad set_series was already processed (and, correctly, rejected
        # with no broadcast of its own). If the isinstance guard were dropped,
        # the accepted set_series would broadcast FIRST, so the frame we
        # receive would carry its non-None series_media_id -> red.
        ws.send_json({"type": "command", "action": "pause"})
        frame = ws.receive_json()

    assert frame["type"] == "state"
    assert frame["series_media_id"] is None, (
        f"rejected set_series leaked into the broadcast despite {why}"
    )
    assert state.series_media_id is None, f"state mutated despite {why}"
    assert state.series_label is None, f"state mutated despite {why}"
    assert not (paths.app_data_dir() / "active_series.json").exists(), (
        f"a pointer file was written despite {why}"
    )


def test_set_series_rejected_when_gazetteer_disabled(tmp_path, monkeypatch):
    """gazetteer.enabled=false must disable the feature at the WIRE choke
    point, not merely hide the UI: any WS client holding the token can send
    set_series, so build_app(gazetteer_enabled=False) must make the handler
    reject it -- no state mutation, no epoch bump, no pointer write, and no
    broadcast of its own.

    The barrier is the same as the bad-payload test: a valid pause is handled
    AFTER the rejected set_series and broadcasts a state frame. If the reject
    guard were absent, the accepted set_series would broadcast FIRST, so the
    frame we receive would carry its non-None series_media_id -> red.

    Seeded-defect drill: delete the `if not gazetteer_enabled: return` guard in
    _handle_command's set_series branch -> state mutates, epoch bumps, the
    pointer is written, and its state frame is broadcast -> every assert red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths
    from yamabiko.display.browser_server import build_app

    state = AppState()
    app = build_app(
        state=state, gazetteer_enabled=False, bound_port=8765, ws_token=_TOKEN,
    )
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({
            "type": "command", "action": "set_series",
            "media_id": 7, "label": "X",
        })
        ws.send_json({"type": "command", "action": "pause"})
        frame = ws.receive_json()

    assert frame["type"] == "state"
    assert frame["series_media_id"] is None, (
        f"a rejected set_series leaked into the broadcast; got {frame!r}"
    )
    assert state.series_media_id is None, "state mutated despite feature disabled"
    assert state.series_epoch == 0, "series_epoch bumped despite feature disabled"
    assert not (tmp_path / "active_series.json").exists(), (
        "a pointer file was written despite the gazetteer feature being disabled"
    )


def test_set_series_with_null_media_id_clears_state_pointer_and_broadcasts(
    tmp_path, monkeypatch,
):
    """The flyout's Clear sends set_series with "media_id": null (Decision
    11): AppState drops the series (a new series epoch, so the translation
    stage drops the gazetteer), active_series.json is deleted so the next
    boot starts without it, and the state broadcast carries nulls.

    Barrier: a pause after the clear broadcasts a state frame; a rejected
    clear would leave the pause's frame first, still naming series 7.
    Seeded defects: no clear branch (rejected as a non-int media_id), or a
    clear that keeps the pointer file -> red."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths
    from yamabiko.gazetteer.gazetteer import save_active_series

    paths.ensure_dirs()
    save_active_series(7, "Frieren (2023)", "葬送のフリーレン")
    state = AppState()
    state.set_series(7, "Frieren (2023)", "葬送のフリーレン")
    epoch = state.series_epoch
    client = TestClient(_get_app(state))
    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({"type": "command", "action": "set_series", "media_id": None})
        ws.send_json({"type": "command", "action": "pause"})
        frame = ws.receive_json()

    assert frame["type"] == "state"
    assert (frame["series_media_id"], frame["series_label"], frame["series_native"]) == (
        None, None, None)
    assert (state.series_media_id, state.series_label, state.series_native) == (None, None, None)
    assert state.series_epoch == epoch + 1
    assert not (paths.app_data_dir() / "active_series.json").exists(), (
        "Clear left the active-series pointer: the next boot would restore the series")


def test_toggle_overlay_command_broadcasts_the_shortcut_frame():
    """The flyout's Subtitles tile sends {"action": "toggle_overlay"}; the
    server answers every client with the frame Ctrl+Alt+O broadcasts
    (Decision 11), so the Qt overlay and the browser page both toggle.

    Barrier: the pause after it broadcasts a state frame. Seeded defects: no
    toggle_overlay branch (logged as an unknown action) -> the first frame
    after the command is the pause's state frame -> red; reply to the sender
    only -> the other client's first frame is the pause's -> red."""
    state = AppState()
    client = TestClient(_get_app(state))
    with client.websocket_connect(_WS) as ws, client.websocket_connect(_WS) as other:
        _drain_initial_frames(ws)
        _drain_initial_frames(other)
        ws.send_json({"type": "command", "action": "toggle_overlay"})
        ws.send_json({"type": "command", "action": "pause"})
        first = ws.receive_json()
        # Asserted before the next receive: without the branch only the
        # pause's frame arrives, and a second receive would wait for good.
        assert first == {"type": "toggle_overlay"}
        assert other.receive_json() == {"type": "toggle_overlay"}
        second = ws.receive_json()

    assert second["type"] == "state"


def test_set_hotkeys_command_reaches_the_pipeline_rebind_in_canonical_form():
    """set_hotkeys hands the pipeline the new combos (canonical spelling);
    a payload with an unknown key or an unbindable combo is rejected whole.

    Seeded defects: skip the on_hotkeys_change call, pass a bad combo
    through, or drop the replay flag -> red. Barrier: the /ws loop handles
    commands in order, so the pause's state broadcast arrives only after
    all four set_hotkeys ran."""
    from yamabiko.display.browser_server import build_app

    calls = []
    app = build_app(state=AppState(), bound_port=8765, ws_token=_TOKEN,
                    on_hotkeys_change=lambda hk, replay: calls.append((hk, replay)))
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        _drain_initial_frames(ws)
        ws.send_json({"type": "command", "action": "set_hotkeys",
                      "hotkeys": {"pause_resume": "alt+ctrl+k", "show_series": "Ctrl+Alt+S"}})
        ws.send_json({"type": "command", "action": "set_hotkeys",
                      "hotkeys": {"pause_resume": "K"}})
        ws.send_json({"type": "command", "action": "set_hotkeys",
                      "hotkeys": {"volume_up": "Ctrl+Alt+U"}})
        ws.send_json({"type": "command", "action": "set_hotkeys",
                      "hotkeys": {"toggle_overlay": "Ctrl+Alt+J"}, "replay": True})
        ws.send_json({"type": "command", "action": "pause"})
        assert ws.receive_json()["type"] == "state"
    assert calls == [({"pause_resume": "Ctrl+Alt+K", "show_series": "Ctrl+Alt+S"}, False),
                     ({"toggle_overlay": "Ctrl+Alt+J"}, True)]
