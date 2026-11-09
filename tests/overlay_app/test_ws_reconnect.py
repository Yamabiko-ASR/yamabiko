"""_ReconnectBackoff delay-math tests.

These tests exercise the backoff state machine's arithmetic directly.
They do NOT test the QWebSocket signal/slot plumbing nor the
reconnect loop in PipelineClient — those are integration concerns
handled by the frozen-bundle E2E.

The prior name test_backoff_sequence implied "the reconnect sequence"
(i.e. the WS layer actually reconnecting); the body only tests delay
math. The qtbot fixture is retained because the module is imported from
a Qt subpackage; dropping it would require reworking import order.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")


def test_reconnect_backoff_delay_math_doubles_until_cap(qtbot):
    """Seeded-defect check: change the doubling step to 3x -> fails."""
    from yamabiko.overlay_app.ws_client import _ReconnectBackoff

    bo = _ReconnectBackoff(initial_ms=1000, cap_ms=30000)
    assert bo.next_delay_ms() == 1000
    assert bo.next_delay_ms() == 2000
    assert bo.next_delay_ms() == 4000
    assert bo.next_delay_ms() == 8000
    assert bo.next_delay_ms() == 16000
    assert bo.next_delay_ms() == 30000
    assert bo.next_delay_ms() == 30000  # capped


def test_reconnect_backoff_reset_returns_to_initial(qtbot):
    """Seeded-defect check: make reset() a no-op -> this test fails."""
    from yamabiko.overlay_app.ws_client import _ReconnectBackoff

    bo = _ReconnectBackoff(initial_ms=1000, cap_ms=30000)
    bo.next_delay_ms()
    bo.next_delay_ms()
    bo.next_delay_ms()
    bo.reset()
    assert bo.next_delay_ms() == 1000


def test_reconnect_backoff_honors_custom_cap(qtbot):
    """Seeded-defect check: hard-code cap_ms=30000 in production ->
    this test fails (8th delay would be 30000, not 5000)."""
    from yamabiko.overlay_app.ws_client import _ReconnectBackoff

    bo = _ReconnectBackoff(initial_ms=500, cap_ms=5000)
    delays = [bo.next_delay_ms() for _ in range(8)]
    assert delays == [500, 1000, 2000, 4000, 5000, 5000, 5000, 5000]


# --- send_command must report drops to its caller ---


def test_send_command_returns_false_on_disconnected_drop(qtbot):
    """A freshly-built (never-connected) client drops the frame; callers
    (Tray Settings Apply) need that fact to fall back to a local apply +
    persist instead of silently losing the command.

    Seeded defect: `return` instead of `return False` on the drop
    branch -> send_command yields None... still falsy; the REAL seed:
    `return True` on the drop branch (claim sent when dropped) ->
    this assertion goes red. The strict `is False` also catches the
    bare-return regression (None is not False).
    """
    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    assert client.send_command({"type": "command", "action": "pause"}) is False


def test_send_command_returns_true_and_serializes_when_connected(qtbot, monkeypatch):
    """With the socket in ConnectedState the payload is JSON-serialized
    onto the wire and the caller is told it was sent (True -> Tray
    skips the offline fallback).

    Seeded defect: swap the return values (False on the sent branch)
    -> red; or drop the sendTextMessage call -> wire stays empty -> red.
    """
    import json

    from PySide6.QtNetwork import QAbstractSocket

    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    wire: list[str] = []

    class _ConnectedWs:
        def state(self):
            return QAbstractSocket.SocketState.ConnectedState

        def sendTextMessage(self, text: str) -> None:
            wire.append(text)

    monkeypatch.setattr(client, "_ws", _ConnectedWs())

    payload = {"type": "command", "action": "set_style", "style": {"jp_size": 9}}
    assert client.send_command(payload) is True
    assert wire and json.loads(wire[-1]) == payload


def test_disconnected_send_warning_names_action_but_not_series_title(qtbot, caplog):
    """A series pick while the pipeline is down is dropped with a WARNING in
    main-overlay.log, which ships in the diagnostics zip. The line may name
    the command's action but must not carry the anime title or its AniList
    id.

    Seeded defect: log the whole payload again -> the title and id appear in
    the record -> red.
    """
    import logging

    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    payload = {
        "type": "command", "action": "set_series",
        "media_id": 918273, "label": "Secret Show Title",
    }
    with caplog.at_level(logging.WARNING, logger="yamabiko"):
        assert client.send_command(payload) is False

    text = caplog.text
    assert "set_series" in text, f"warning does not name the action: {text!r}"
    assert "Secret Show Title" not in text, f"warning leaks the title: {text!r}"
    assert "918273" not in text, f"warning leaks the media id: {text!r}"


# --- the refused-connect dead-end (P0: reconnect loop must survive it) ---
#
# A REFUSED connect emits BOTH disconnected and errorOccurred on PySide6
# 6.11/Windows (measured; see test_real_ws_reconnect.py's timing-model
# note), so either handler alone can re-arm the loop and the isActive
# guard absorbs the second of the pair. The tests below pin each arm
# independently -- the pairing is a platform detail, and a build that
# reported a refusal as error-only would ride entirely on _on_error.


def test_refused_connect_error_arms_reconnect_timer(qtbot):
    """The error arm re-arms on its own, without any disconnected.

    Seeded defect: revert _on_error to log-only -> timer never armed
    -> red."""
    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    assert not client._reconnect_timer.isActive()
    client._on_error(object())  # refused connect: no disconnected follows
    assert client._reconnect_timer.isActive()
    client._reconnect_timer.stop()


def test_error_then_disconnected_advances_backoff_exactly_once(qtbot):
    """A genuine drop can emit BOTH errorOccurred and disconnected; the
    isActive guard must keep the pair from double-advancing the backoff.

    Seeded defect: remove the isActive guard in _schedule_reconnect ->
    two next_delay_ms() calls consume 1000 and 2000 -> the probe below
    reads 4000 -> red."""
    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    client._on_error(object())
    client._on_disconnected()
    # Exactly one backoff step (1000) was consumed by the single arm.
    assert client._backoff.next_delay_ms() == 2000
    client._reconnect_timer.stop()


def test_stop_prevents_rearm_from_error_and_disconnected(qtbot):
    """stop() closes the socket, which fires disconnected (and often an
    error) -- neither may resurrect the reconnect timer after shutdown.

    Seeded defect: set _stopped AFTER _ws.close() in stop() (or drop the
    _stopped guard in _schedule_reconnect) -> timer re-arms -> red."""
    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    client.stop()
    client._on_error(object())
    client._on_disconnected()
    assert not client._reconnect_timer.isActive()


def test_stop_then_start_reuse_rearms_reconnect_on_refused_error(qtbot):
    """stop() latches _stopped; start() must clear it, or a reused client
    connects once and never auto-reconnects -- a silently one-shot object.

    Seeded defect: drop the _stopped reset in start() -> the refused
    connect after the restart never arms the retry timer -> red."""
    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    client.stop()
    client.start()
    try:
        assert not client._reconnect_timer.isActive()
        client._on_error(object())  # refused connect after the restart
        assert client._reconnect_timer.isActive()
    finally:
        client.stop()


# --- jp_ruby element-shape guard at the trust boundary ---


def test_on_text_drops_non_dict_jp_ruby_segments_before_emit(qtbot):
    """A squatting/regressed producer sending jp_ruby entries that are
    not dicts must not reach layout code (ruby_layout does seg.get and
    would crash the Qt slot dispatcher outside the ruby fallback try).

    Seeded defect: remove the isinstance filter in _on_text -> the
    emitted list still carries the junk entries -> red."""
    import json as _json

    from yamabiko.overlay_app.ws_client import PipelineClient

    client = PipelineClient(url="ws://127.0.0.1:1/ws")
    got: list = []
    client.subtitleReceived.connect(lambda jp, en, ruby: got.append((jp, en, ruby)))
    client._on_text(_json.dumps({
        "type": "subtitle",
        "jp": "テスト",
        "en": "test",
        "jp_ruby": [{"text": "テスト"}, "junk", None, ["x"]],
    }))
    assert got == [("テスト", "test", [{"text": "テスト"}])]
