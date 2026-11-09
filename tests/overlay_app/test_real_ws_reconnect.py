"""Real-tier: PipelineClient re-attaches to a live WS endpoint after loss.

``tests/overlay_app/test_ws_reconnect.py`` exercises only the
``_ReconnectBackoff`` arithmetic and the handlers in isolation -- nothing
stands up a real socket, drops it, and proves the QWebSocket reconnect LOOP
actually re-attaches. These do (M-14): a real uvicorn server hosting the
overlay's own ``browser_server`` ``/ws`` route, a real ``PipelineClient``,
the server taken down, a fresh server brought back on the SAME port, then
an assertion that the client reconnected AND a broadcast frame flows across
the NEW socket.

Timing model, MEASURED against PySide6 6.11 on Windows (not assumed --
probed by driving a real QWebSocket at a dead port and at a killed
server, then pinned by the tests below):

* a clean drop of a CONNECTED socket (uvicorn shutting down) emits
  ``disconnected`` and NO ``errorOccurred`` at all;
* a REFUSED connect (a first or retried connect landing on a dead port)
  emits ``disconnected`` AND ``errorOccurred(ConnectionRefusedError)``,
  in that order, whether or not the socket was ever connected.

So the client's two arms into the reconnect loop are NOT interchangeable:

* ``_on_disconnected`` is the only arm that can carry a clean DROP (no
  error is emitted there at all);
* ``_on_error`` re-arms on a REFUSAL, where it is redundant on this Qt
  build (``disconnected`` fires first and ``_schedule_reconnect``'s
  isActive guard absorbs the second of the pair, so the backoff does not
  double-advance). It is kept, and tested, because the
  disconnected-on-refusal pairing is an undocumented platform detail: on
  a build that reports a refusal as error-only, this arm is the sole path
  back, and without it a refused connect would end the loop for the
  session (live tray, zero subtitles, no recovery).

The three tests split the space:

1. the frictionless restart -- the replacement server is accepting BEFORE
   the Qt loop processes the drop, so no attempt is ever refused;
2. the refused-attempt dead-end -- the drop is processed while the port is
   still DEAD, at least one attempt is asserted genuinely refused
   (ConnectionRefusedError observed on the client's real socket), and only
   then does the replacement appear. Both arms are live, so this pins the
   OUTCOME (a refusal is not terminal), not which arm delivered it;
3. the ERROR ARM ALONE -- a cold boot connect against a not-yet-bound port
   (the real "overlay started before the pipeline's uvicorn" race) with
   the ``disconnected`` slot detached from the real socket, so only
   ``_on_error`` can re-arm. Nothing else in the suite can distinguish
   that arm from dead code.

A fourth test pins the on-connect frames over the same real stack: the
state frame's series Japanese title through the three-argument
``seriesReceived`` and the style frame's ``reduce_motion``. A fifth pins the
shortcut wiring both ways: a queued "Shortcut unavailable" notice reaches
``noticeReceived`` with its Settings page, and a ``set_hotkeys`` command
from the client reaches the pipeline's ``on_hotkeys_change``. A sixth pins
the problems frame on connect and on change.

Real-tier because the connect + retry cycles run past 1 s (a loopback
refusal surfaces after ~4 s on this stack). All server threads are joined
in teardown (no leaks).

Seeded defects that turn this file red:
- delete ``self._reconnect_timer.start(delay)`` in ``_schedule_reconnect``
  -> nothing ever re-arms -> every reconnect ``waitUntil`` times out;
- revert ``_on_error`` to log-only (no re-arm) -> test 3's connect
  ``waitUntil`` times out (tests 1 and 2 stay GREEN: ``_on_disconnected``
  covers for it on this Qt build -- which is exactly why test 3 exists);
- delete ``self.subtitleReceived.emit(...)`` in ``_on_text`` -> the
  post-reconnect frame never arrives -> the frame-flow ``waitUntil``s
  time out.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("uvicorn")

pytestmark = pytest.mark.real

# The server refuses every token-free /ws connection. The real
# PipelineClient picks the token up from YAMABIKO_WS_TOKEN (the variable
# run_pipeline sets for the overlay child), so these tests go through the
# same tokened connect URL production uses.
_TOKEN = "real-ws-session-token"


@pytest.fixture(autouse=True)
def _session_token_env(monkeypatch):
    from yamabiko.display.broadcast import WS_TOKEN_ENV

    monkeypatch.setenv(WS_TOKEN_ENV, _TOKEN)


def _pick_free_port() -> int:
    """Ask the OS for an unused TCP port (closed before the server binds)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_app(port: int):
    """The overlay's real control-bus app (its /ws route is the subject)."""
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    # serve_browser_ui=False keeps the /ws + broadcast path (what the overlay
    # QWebSocket uses) without depending on the static index assets.
    return build_app(
        state=AppState(), serve_browser_ui=False, bound_port=port,
        ws_token=_TOKEN,
    )


class _ServerThread(threading.Thread):
    """Run a uvicorn.Server for the overlay app on its own asyncio loop.

    Mirrors ``tests/test_real_uvicorn.py``: instance-attribute monkey-patch
    of ``install_signal_handlers`` (keep SIGINT/SIGTERM off this thread),
    then ``server.serve()`` under a fresh loop. A bounded graceful-shutdown
    timeout guarantees ``stop()`` returns even with an active WS connection,
    so the SAME port is free for the replacement server.
    """

    def __init__(self, app, port: int) -> None:
        super().__init__(daemon=True)
        import uvicorn

        cfg = uvicorn.Config(
            app, host="127.0.0.1", port=port,
            log_level="warning", access_log=False,
            timeout_graceful_shutdown=3,
        )
        self.server = uvicorn.Server(cfg)
        self.server.install_signal_handlers = lambda: None
        self.app = app
        self.loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(self.server.serve())
        finally:
            self.loop.close()

    def stop(self, timeout: float = 6.0) -> None:
        """Cooperative shutdown; blocks until the thread exits (idempotent)."""
        self.server.should_exit = True
        self.join(timeout=timeout)

    def broadcast_subtitle(self, jp: str, en: str) -> None:
        """Push a subtitle frame to every connected client, from the test
        thread, by scheduling the async broadcast on the server's own loop."""
        if self.loop is None:  # pragma: no cover - guarded by _wait_for_port
            raise AssertionError("server loop not started")
        frame = {"type": "subtitle", "jp": jp, "en": en, "jp_ruby": []}
        fut = asyncio.run_coroutine_threadsafe(
            self.app.state.manager.broadcast(frame), self.loop,
        )
        fut.result(timeout=5.0)


def _wait_for_port(port: int, timeout: float = 8.0) -> bool:
    """Poll until the server accepts TCP connections, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.05)
    return False


def _start_server(port: int) -> _ServerThread:
    """Start uvicorn on ``port`` and block until it is accepting."""
    t = _ServerThread(_make_app(port), port)
    t.start()
    if not _wait_for_port(port):
        t.stop()
        raise AssertionError(f"uvicorn never accepted on 127.0.0.1:{port}")
    return t


def test_pipeline_client_reattaches_to_restarted_ws_server_and_frame_flows(qtbot):
    """server up -> connect -> server down -> server up (same port) ->
    the real PipelineClient re-attaches via its backoff timer and a
    broadcast subtitle flows across the reconnected socket.
    """
    from yamabiko.overlay_app.ws_client import PipelineClient, _ReconnectBackoff

    port = _pick_free_port()
    server1 = _start_server(port)

    connected: list[bool] = []
    subtitles: list[tuple] = []
    client = PipelineClient(url=f"ws://127.0.0.1:{port}/ws")
    # Short backoff: the replacement server is brought up BEFORE the Qt loop
    # processes the disconnect, so the single scheduled reconnect lands on a
    # live server regardless of the exact delay.
    client._backoff = _ReconnectBackoff(initial_ms=150, cap_ms=600)
    client.connectedChanged.connect(connected.append)
    client.subtitleReceived.connect(lambda jp, en, _ruby: subtitles.append((jp, en)))

    server2: "_ServerThread | None" = None
    try:
        # --- server up -> connect ---
        client.start()
        qtbot.waitUntil(lambda: len(connected) > 0 and connected[-1], timeout=8000)
        # Baseline: a frame flows on the first socket, proving the live channel.
        server1.broadcast_subtitle("最初", "first")
        qtbot.waitUntil(lambda: len(subtitles) > 0, timeout=5000)
        assert subtitles[-1] == ("最初", "first")

        # --- server down, then a fresh server up on the SAME port ---
        # Both happen while the Qt loop is NOT spinning (blocking calls), so
        # the client has not yet processed the disconnect: the reconnect it is
        # about to schedule will find server-2 already accepting.
        server1.stop()
        server2 = _start_server(port)

        connected.clear()
        subtitles.clear()

        # --- now let the client notice the drop and re-attach ---
        qtbot.waitUntil(
            lambda: len(connected) > 0 and connected[-1], timeout=15000,
        )
        assert connected[-1] is True, (
            "PipelineClient did not re-attach to the restarted WS server "
            "via its backoff reconnect timer"
        )

        # Reconnection alone could be a half-open socket; a delivered broadcast
        # proves a genuinely live channel on the NEW connection.
        server2.broadcast_subtitle("再接続", "reconnected")
        qtbot.waitUntil(
            lambda: any(en == "reconnected" for _jp, en in subtitles),
            timeout=8000,
        )
        assert ("再接続", "reconnected") in subtitles, (
            "reconnected but no subtitle frame crossed the new socket"
        )
    finally:
        client.stop()
        server1.stop()
        if server2 is not None:
            server2.stop()


def test_reconnect_survives_refused_attempt_then_reattaches_when_server_returns(qtbot):
    """server up -> connect -> server down -> the drop is processed while
    the port is still DEAD (at least one reconnect attempt is genuinely
    REFUSED) -> server up (same port) -> the client re-attaches and a
    broadcast frame flows on the new socket.

    The dead-end the shipped backoff has to survive: an attempt that lands
    on a closed port. Both arms of the loop are live here, so this pins the
    OUTCOME (a refusal is not terminal), not which arm delivered it -- see
    the error-arm-alone test below for that.

    Seeded-defect drill: delete the _reconnect_timer.start(delay) in
    _schedule_reconnect -> nothing re-arms after the refusal -> the
    recovery waitUntil times out -> red.
    """
    from PySide6.QtNetwork import QAbstractSocket

    from yamabiko.overlay_app.ws_client import PipelineClient, _ReconnectBackoff

    port = _pick_free_port()
    server1 = _start_server(port)

    connected: list[bool] = []
    subtitles: list[tuple] = []
    refusals: list = []
    client = PipelineClient(url=f"ws://127.0.0.1:{port}/ws")
    # Short backoff so several refused attempts fit inside the deadlines.
    client._backoff = _ReconnectBackoff(initial_ms=150, cap_ms=600)
    client.connectedChanged.connect(connected.append)
    client.subtitleReceived.connect(lambda jp, en, _ruby: subtitles.append((jp, en)))

    def _record_refusal(err) -> None:
        # Observing (never mocking) the real socket: a reconnect attempt
        # on a dead port surfaces here as ConnectionRefusedError; the drop
        # of the live connection surfaces as RemoteHostClosedError and is
        # deliberately not counted.
        if err == QAbstractSocket.SocketError.ConnectionRefusedError:
            refusals.append(err)

    client._ws.errorOccurred.connect(_record_refusal)

    server2: "_ServerThread | None" = None
    try:
        # --- server up -> connect, with a baseline frame on socket one ---
        client.start()
        qtbot.waitUntil(lambda: len(connected) > 0 and connected[-1], timeout=8000)
        server1.broadcast_subtitle("最初", "first")
        qtbot.waitUntil(lambda: len(subtitles) > 0, timeout=5000)

        # --- server down; the port stays DEAD while the drop is processed ---
        server1.stop()
        connected.clear()
        subtitles.clear()

        # Spin the Qt loop with NO server listening: the client processes
        # the drop, retries, and must get REFUSED at least once. This pins
        # the exact dead-end the frictionless test never reaches (there,
        # server-2 accepts before the drop is even processed).
        qtbot.waitUntil(lambda: len(refusals) >= 1, timeout=15000)
        assert not any(connected), (
            "client reported a connection while no server was listening"
        )

        # --- only now does the replacement appear on the SAME port ---
        server2 = _start_server(port)
        qtbot.waitUntil(
            lambda: len(connected) > 0 and connected[-1], timeout=15000,
        )
        assert connected[-1] is True, (
            "PipelineClient never recovered after its reconnect attempt "
            "was refused; the error-path re-arm is the only way back"
        )

        # A delivered broadcast proves a genuinely live channel on the
        # post-refusal connection, not a half-open socket.
        server2.broadcast_subtitle("再接続", "reconnected")
        qtbot.waitUntil(
            lambda: any(en == "reconnected" for _jp, en in subtitles),
            timeout=8000,
        )
        assert ("再接続", "reconnected") in subtitles, (
            "re-attached after the refusal but no subtitle frame crossed "
            "the new socket"
        )
    finally:
        client.stop()
        server1.stop()
        if server2 is not None:
            server2.stop()


def test_error_arm_alone_retries_a_refused_boot_connect_until_the_server_binds(qtbot):
    """The error arm (_on_error's re-arm) must be able to drive the retry
    loop on its own, with no disconnected to help it.

    Why this test exists: on PySide6 6.11 a refused connect emits
    disconnected as well as errorOccurred, so _on_disconnected masks
    _on_error everywhere else -- revert the error re-arm and every other
    test in this file stays green, which would leave the arm
    indistinguishable from dead code. It is not dead: the
    disconnected-on-refusal pairing is an undocumented platform detail,
    and on a build that reports a refusal as error-only this arm is the
    only way back.

    The scenario is the real boot race the arm was written for: the
    overlay starts BEFORE the pipeline's uvicorn has bound its port, so
    the first connect is refused. Starting cold means no drop is needed
    (a clean drop emits disconnected only -- no error at all -- so the
    drop path can never be carried by _on_error and is not asked to be).
    The socket's disconnected -> _on_disconnected connection is detached,
    leaving _on_error as the ONLY thing that can re-arm the timer; then
    the server binds and the client must be found already retrying.

    Nothing is mocked: a real PipelineClient over a real QWebSocket
    against a real uvicorn server. One of two handlers is unhooked so the
    other is observed in isolation.

    Seeded-defect drill: revert _on_error to log-only -> nothing re-arms
    after the refused boot connect -> the connect waitUntil times out ->
    red. (Confirmed red against that revert.)
    """
    from PySide6.QtNetwork import QAbstractSocket

    from yamabiko.overlay_app.ws_client import PipelineClient, _ReconnectBackoff

    port = _pick_free_port()  # nothing is listening on it yet

    connected: list[bool] = []
    subtitles: list[tuple] = []
    refusals: list = []
    client = PipelineClient(url=f"ws://127.0.0.1:{port}/ws")
    client._backoff = _ReconnectBackoff(initial_ms=150, cap_ms=600)
    client.connectedChanged.connect(connected.append)
    client.subtitleReceived.connect(lambda jp, en, _ruby: subtitles.append((jp, en)))

    def _record_refusal(err) -> None:
        if err == QAbstractSocket.SocketError.ConnectionRefusedError:
            refusals.append(err)

    client._ws.errorOccurred.connect(_record_refusal)
    # Isolate the error arm: any re-arm observed below came from _on_error.
    client._ws.disconnected.disconnect(client._on_disconnected)

    server: "_ServerThread | None" = None
    try:
        # --- boot against a dead port: the first connect IS refused ---
        client.start()
        qtbot.waitUntil(lambda: len(refusals) >= 1, timeout=20000)
        assert not any(connected), (
            "client reported a connection while nothing was listening"
        )

        # --- the pipeline finally binds; only _on_error can have re-armed ---
        server = _start_server(port)
        qtbot.waitUntil(
            lambda: len(connected) > 0 and connected[-1], timeout=20000,
        )
        assert connected[-1] is True, (
            "with only the error arm attached, a refused boot connect was "
            "never retried: _on_error does not re-arm the loop"
        )

        # A delivered broadcast proves a genuinely live channel, not a
        # half-open socket.
        server.broadcast_subtitle("接続", "connected at last")
        qtbot.waitUntil(
            lambda: any(en == "connected at last" for _jp, en in subtitles),
            timeout=8000,
        )
        assert ("接続", "connected at last") in subtitles, (
            "error-arm retry produced a socket that carries no frames"
        )
    finally:
        client.stop()
        if server is not None:
            server.stop()


def test_real_on_connect_frames_carry_series_native_and_reduce_motion(qtbot):
    """Real uvicorn + real QWebSocket: the on-connect state frame delivers
    the picked series' Japanese title through the three-argument
    seriesReceived, and the on-connect style frame carries reduce_motion.

    Seeded defects: drop series_native from AppState.state_frame, or build
    the on-connect style frame without style_frame() -> red."""
    from yamabiko.display.browser_server import build_app
    from yamabiko.overlay_app.ws_client import PipelineClient
    from yamabiko.state import AppState

    port = _pick_free_port()
    state = AppState()
    state.set_series(7, "Frieren (2023)", "葬送のフリーレン")
    state.set_reduce_motion(True)
    server = _ServerThread(
        build_app(state=state, serve_browser_ui=False, bound_port=port, ws_token=_TOKEN),
        port,
    )
    server.start()
    assert _wait_for_port(port), f"uvicorn never accepted on 127.0.0.1:{port}"
    series: list[tuple] = []
    styles: list[dict] = []
    client = PipelineClient(url=f"ws://127.0.0.1:{port}/ws")
    client.seriesReceived.connect(lambda mid, label, native: series.append((mid, label, native)))
    client.styleReceived.connect(styles.append)
    try:
        client.start()
        qtbot.waitUntil(lambda: bool(series) and bool(styles), timeout=8000)
        assert series[-1] == (7, "Frieren (2023)", "葬送のフリーレン")
        assert styles[-1]["reduce_motion"] is True
    finally:
        client.stop()
        server.stop()


def test_real_shortcut_notice_carries_its_page_and_set_hotkeys_reaches_the_pipeline(qtbot):
    """Real uvicorn + real QWebSocket, both directions of the shortcut
    wiring: the queued "Shortcut unavailable" notice reaches the client's
    three-argument noticeReceived with its Settings page, and a set_hotkeys
    command the client sends reaches the pipeline's on_hotkeys_change.

    Seeded defects: drop settings_page in PipelineClient._on_text, or
    misspell the set_hotkeys action in the server's command handler -> red."""
    from yamabiko.display.broadcast import make_notice_frame
    from yamabiko.display.browser_server import build_app
    from yamabiko.overlay_app.ws_client import PipelineClient
    from yamabiko.state import AppState

    body = "Ctrl+Alt+S is used by another app. Pick a new one in Settings."
    port = _pick_free_port()
    state = AppState()
    state.queue_notice(make_notice_frame("Shortcut unavailable", body, settings_page="shortcuts"))
    calls: list[dict] = []
    server = _ServerThread(
        build_app(state=state, serve_browser_ui=False, bound_port=port, ws_token=_TOKEN,
                  on_hotkeys_change=lambda hk, replay: calls.append(hk)),
        port,
    )
    server.start()
    assert _wait_for_port(port), f"uvicorn never accepted on 127.0.0.1:{port}"
    notices: list[tuple] = []
    client = PipelineClient(url=f"ws://127.0.0.1:{port}/ws")
    client.noticeReceived.connect(lambda title, text, page: notices.append((title, text, page)))
    try:
        client.start()
        qtbot.waitUntil(lambda: bool(notices), timeout=8000)
        assert notices == [("Shortcut unavailable", body, "shortcuts")]
        assert client.send_command({"type": "command", "action": "set_hotkeys",
                                    "hotkeys": {"pause_resume": "Ctrl+Alt+K"}}) is True
        qtbot.waitUntil(lambda: calls == [{"pause_resume": "Ctrl+Alt+K"}], timeout=8000)
    finally:
        client.stop()
        server.stop()


def test_real_problems_frame_reaches_the_client_on_connect_and_on_change(qtbot):
    """Real uvicorn + real QWebSocket: the connect frame carries the active
    problem and a carried recovery; a later broadcast clears it.

    Seeded defects: drop the problems send from the /ws handler, or the
    problems branch in PipelineClient._on_text -> red."""
    from yamabiko import problems
    from yamabiko.display.browser_server import build_app
    from yamabiko.overlay_app.ws_client import PipelineClient
    from yamabiko.state import AppState

    port = _pick_free_port()
    state = AppState()
    state.set_problem(problems.audio_problem("Speakers (Test)"))
    state.set_carried_recoveries(["translator"], until=time.monotonic() + 60.0)
    app = build_app(state=state, serve_browser_ui=False, bound_port=port, ws_token=_TOKEN)
    server = _ServerThread(app, port)
    server.start()
    assert _wait_for_port(port), f"uvicorn never accepted on 127.0.0.1:{port}"
    frames: list[tuple] = []
    client = PipelineClient(url=f"ws://127.0.0.1:{port}/ws")
    client.problemsReceived.connect(lambda active, recovered: frames.append((active, recovered)))
    try:
        client.start()
        qtbot.waitUntil(lambda: bool(frames), timeout=8000)
        active, recovered = frames[0]
        assert [p["line"] for p in active] == ["Speakers (Test) was disconnected."]
        assert recovered == [{"kind": "translator", "recovery": "Translations are back."}]
        state.clear_problem("audio")
        asyncio.run_coroutine_threadsafe(
            app.state.manager.broadcast(state.problems_frame()), server.loop,
        ).result(timeout=5.0)
        qtbot.waitUntil(lambda: len(frames) == 2, timeout=8000)
        assert frames[1] == ([], [])
    finally:
        client.stop()
        server.stop()
