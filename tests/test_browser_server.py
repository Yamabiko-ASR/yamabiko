"""FastAPI TestClient integration test for the overlay server."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# Every /ws client must present the per-launch session token, so each app
# is built with this one and the native-client connects carry it as ?t=.
_TOKEN = "test-session-token"
_WS = f"/ws?t={_TOKEN}"


def _get_app():
    # Fresh app per test to avoid cross-test state from the global manager.
    from yamabiko.display.browser_server import build_app
    return build_app(bound_port=8765, ws_token=_TOKEN)


def test_root_serves_html():
    app = _get_app()
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


def test_static_overlay_css_served():
    app = _get_app()
    client = TestClient(app)
    r = client.get("/static/overlay.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]


def test_static_overlay_js_served():
    app = _get_app()
    client = TestClient(app)
    r = client.get("/static/overlay.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"].lower()


def test_overlay_css_font_faces_point_at_served_files():
    # Every @font-face url in overlay.css must be a bundled font the server
    # actually serves. Seeded defects: a typo in a url, or a font file left
    # out of static/fonts -> 404 or set mismatch, red.
    import re
    from pathlib import Path

    import yamabiko.display as display
    from yamabiko.overlay_app.fonts import FONT_FILES

    css = (Path(display.__file__).parent / "static" / "overlay.css").read_text(encoding="utf-8")
    urls = re.findall(r'url\("(/static/fonts/[^"]+)"\)', css)
    assert {url.rsplit("/", 1)[-1] for url in urls} == set(FONT_FILES)

    client = TestClient(_get_app())
    for url in urls:
        r = client.get(url)
        assert r.status_code == 200, url
        assert len(r.content) > 1_000_000, url


# --- The WS/health bus must survive disabling the browser page ---


def test_build_app_without_browser_ui_keeps_ws_and_health_routes():
    """serve_browser_ui=False gates ONLY the browser overlay page.
    /ws (Qt overlay subtitles + tray commands) and /healthz + /livez
    (watchdog probes) are the display/control bus and must stay up —
    gating the whole server on overlay.browser would kill subtitles,
    tray Quit, and the health probes in one silent stroke.

    Seeded defect: gate build_app's route registration wholesale on
    serve_browser_ui -> /healthz 404s -> red.
    """
    from yamabiko.display.browser_server import build_app

    app = build_app(serve_browser_ui=False, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.get("/livez").status_code == 200
    with client.websocket_connect(_WS) as ws:
        # The /ws handler still serves the on-connect state snapshot.
        assert ws.receive_json()["type"] == "state"


def test_build_app_without_browser_ui_hides_page_and_static():
    """With serve_browser_ui=False the index route and the /static mount
    must NOT be registered.

    Seeded defect: leave the StaticFiles mount / index route
    unconditional -> GET / returns 200 -> red.
    """
    from yamabiko.display.browser_server import build_app

    app = build_app(serve_browser_ui=False, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    assert client.get("/").status_code == 404
    assert client.get("/static/overlay.js").status_code == 404


def test_build_app_default_still_serves_browser_page():
    """The default (and serve_browser_ui=True) keeps the expected behavior
    -- page + static + bus all served. The dedicated page/static tests
    above pin the content; this pins the default.

    Seeded defect: flip the serve_browser_ui default to False ->
    GET / 404s for every existing caller -> red.
    """
    from yamabiko.display.browser_server import build_app

    client = TestClient(build_app(bound_port=8765, ws_token=_TOKEN))
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_websocket_accepts_connection_and_registers_client():
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    assert mgr.client_count() == 0
    with client.websocket_connect(_WS) as ws:
        # Handshake finished; manager should have one client.
        # Use ws to keep the connection open for the assertion.
        assert mgr.client_count() == 1
    # After context exit the socket closes; manager must drop the client.
    assert mgr.client_count() == 0


def test_broadcast_reaches_websocket_client():
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    with client:
        with client.websocket_connect(_WS) as ws:
            # Consume the initial state snapshot the server sends on connect.
            state_msg = ws.receive_json()
            assert state_msg["type"] == "state"
            # Also consume the initial style snapshot.
            style_msg = ws.receive_json()
            assert style_msg["type"] == "style"
            assert ws.receive_json()["type"] == "problems"
            # client.portal is the BlockingPortal that drives the app's async loop;
            # calling broadcast through it avoids the NoEventLoopError that
            # anyio.from_thread.run raises when invoked from a plain sync thread.
            client.portal.call(mgr.broadcast, {"type": "subtitle", "jp": "x", "en": "y"})
            msg = ws.receive_json()
            assert msg == {"type": "subtitle", "jp": "x", "en": "y"}


def test_new_client_gets_problems_between_style_and_queued_notices():
    """Lead decision 1: state, style, problems, then the queued notices.
    Seeded defects: no problems frame on connect, or after the notices ->
    red."""
    from yamabiko import problems
    from yamabiko.display.broadcast import make_notice_frame
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    state.set_problem(problems.translator_problem(True))
    state.queue_notice(make_notice_frame("Shortcut unavailable", "x"))
    client = TestClient(build_app(state=state, bound_port=8765, ws_token=_TOKEN))
    with client.websocket_connect(_WS) as ws:
        frames = [ws.receive_json() for _ in range(4)]
    assert [f["type"] for f in frames] == ["state", "style", "problems", "notice"]
    assert frames[2]["active"] == [problems.translator_problem(True)]
    assert frames[2]["recovered"] == []


def test_queued_notices_wait_for_the_qt_client_not_the_browser():
    """A notice queued before any client connects (a shortcut conflict at
    startup) is for the Qt overlay, the only client that shows notices. A
    browser/OBS page that connects first leaves it queued, and the first
    client without an Origin gets it after its snapshot frames.

    Seeded defects: drain the pending notices for every client -> the browser
    takes the notice, so the native client's frame after the pause barrier
    is the pause's state broadcast -> red. Register every client as non-native
    (manager.add(websocket)) or flip the flag (native=bool(origin)) -> a
    native_client_count assert goes red."""
    from yamabiko.display.broadcast import make_notice_frame
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    state.queue_notice(make_notice_frame("Shortcut unavailable", "x"))
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    kw = _browser_connect_kwargs(client, "http://127.0.0.1:8765")
    with client.websocket_connect(kw["url"], headers=kw["headers"]) as browser:
        assert [browser.receive_json()["type"] for _ in range(3)] == [
            "state", "style", "problems",
        ]
        assert app.state.manager.native_client_count() == 0
    with client.websocket_connect(_WS) as ws:
        assert [ws.receive_json()["type"] for _ in range(3)] == [
            "state", "style", "problems",
        ]
        assert app.state.manager.native_client_count() == 1
        # Barrier: the pause broadcast comes after any queued notice, so a
        # missing notice shows up as this state frame instead of a hang.
        ws.send_json({"type": "command", "action": "pause"})
        frame = ws.receive_json()
    assert frame == {"type": "notice", "title": "Shortcut unavailable", "body": "x"}, (
        f"the Qt client did not get the queued notice; got {frame!r}"
    )


def test_set_style_updates_state_and_broadcasts(tmp_path, monkeypatch):
    """Under the layered-config model, persisted style changes land in
    the sparse user_overrides.yaml (targetted via YAMABIKO_APPDATA).
    The browser server's set_style handler persists unconditionally via
    the sparse writer, which always targets user_overrides.yaml."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        # Drain initial state + style + problems messages.
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()
        ws.send_text('{"type":"command","action":"set_style","style":{"jp_size": 30}}')
        msg = ws.receive_json()
        assert msg["type"] == "style"
        assert msg["jp_size"] == 30
    import yaml
    data = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert data["overlay"]["style"]["jp_size"] == 30


def test_set_style_strips_css_injection_from_string_value(tmp_path, monkeypatch):
    """overlay.js interpolates style strings straight into CSS custom
    properties, so a malicious set_style value containing CSS-breaking
    chars (`;{}<>`, `</`) would inject arbitrary CSS/markup into the
    overlay page. The SERVER (which the client trusts) must reject or
    strip such values before broadcasting -- a value with injection chars
    must NOT reach the client verbatim.

    Seeded-defect check: remove the server-side sanitize() call in the
    set_style handler (pass `style` straight to update_style) -> the
    broadcast carries the raw 'x; } body{display:none' jp_font -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    malicious = "x; } body{display:none"
    with client.websocket_connect(_WS) as ws:
        ws.receive_json()  # initial state
        ws.receive_json()  # initial style
        ws.receive_json()  # initial problems
        ws.send_json({
            "type": "command",
            "action": "set_style",
            "style": {"jp_font": malicious, "jp_size": 30},
        })
        msg = ws.receive_json()
    assert msg["type"] == "style"
    # The benign sibling key still applies (sanitize strips per-value,
    # doesn't drop the whole frame).
    assert msg["jp_size"] == 30
    # The injection value must NOT be broadcast verbatim.
    assert ";" not in str(msg["jp_font"]), (
        f"CSS-injection chars reached the client unsanitized: "
        f"jp_font={msg['jp_font']!r}"
    )
    assert "}" not in str(msg["jp_font"]) and "{" not in str(msg["jp_font"])


def test_set_style_rejects_injection_in_color_value(tmp_path, monkeypatch):
    """text_color/bg_color are interpolated into `--text-color` / parsed
    for rgba(); a value carrying `;}<>` chars is a CSS-injection vector
    and must be stripped/rejected server-side, not broadcast verbatim.

    Seeded-defect check: drop the sanitize() call -> the raw
    '#fff; } html{...' text_color is broadcast -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    bad_color = "#fff; } html{background:url(x)"
    with client.websocket_connect(_WS) as ws:
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()
        ws.send_json({
            "type": "command",
            "action": "set_style",
            "style": {"text_color": bad_color},
        })
        msg = ws.receive_json()
    assert msg["type"] == "style"
    assert ";" not in str(msg["text_color"]) and "{" not in str(msg["text_color"]), (
        f"CSS-injection chars in text_color reached the client: "
        f"text_color={msg['text_color']!r}"
    )


def test_set_style_broadcasts_max_width_pct_through_sanitize(tmp_path, monkeypatch):
    """Browser-overlay parity: the max_width_pct subtitle-width cap must
    ride the set_style -> _sanitize_style -> broadcast path unharmed, so the
    browser overlay caps+wraps subtitles identically to the Qt overlay. It
    is a numeric key, so _sanitize_style (which only rejects unsafe STRING
    values) passes it straight through -- no clamping is invented
    server-side, matching how jp_size and the other numeric keys already
    flow.

    Seeded defect (one line): add `if key == "max_width_pct": continue` to
    browser_server._sanitize_style (a strip-list entry) -> the broadcast
    frame no longer carries the 50 the client sent -> red. Same for
    state.update_style dropping the key from the merged payload.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.receive_json()  # initial state
        ws.receive_json()  # initial style
        ws.receive_json()  # initial problems
        ws.send_json({
            "type": "command",
            "action": "set_style",
            "style": {"max_width_pct": 50},
        })
        msg = ws.receive_json()
    assert msg["type"] == "style"
    # A non-default value (50) so the assertion tracks the value flowing
    # through sanitize+broadcast, not merely the STYLE_DEFAULTS 70.
    assert msg["max_width_pct"] == 50, (
        "max_width_pct must survive _sanitize_style + broadcast unharmed "
        f"(browser/Qt parity); got {msg.get('max_width_pct')!r}"
    )


def test_set_style_clamps_out_of_range_values_before_broadcast_and_persist(
    tmp_path, monkeypatch,
):
    """Server-side clamp parity with the Qt tray: tray._on_style_received
    runs config._clamp_style on every WS frame, but the browser overlay
    applies whatever the server broadcasts, and the server persists it to
    user_overrides.yaml -- so an out-of-range set_style (any loopback WS
    client can send one) used to render unclamped in the browser AND
    survive restarts on disk. The set_style handler must clamp the partial
    dict against the same _STYLE_RANGES before merge/broadcast/persist.

    hold_ms=999999999 (range 500..15000) and max_lines=0 (range 1..6)
    straddle both clamp directions; the valid sibling jp_size=30 proves
    clamping is per-key, not frame-dropping.

    Seeded defect (one line): drop the clamp step in _handle_command's
    set_style branch (pass the sanitized dict straight to update_style)
    -> the broadcast carries hold_ms=999999999 -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko import paths
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.receive_json()  # initial state
        ws.receive_json()  # initial style
        ws.receive_json()  # initial problems
        ws.send_json({
            "type": "command",
            "action": "set_style",
            "style": {"hold_ms": 999999999, "max_lines": 0, "jp_size": 30},
        })
        msg = ws.receive_json()
    assert msg["type"] == "style"
    assert msg["hold_ms"] == 15000, (
        f"hold_ms must clamp to the _STYLE_RANGES max (15000); the browser "
        f"overlay would hold every subtitle for {msg['hold_ms']}ms"
    )
    assert msg["max_lines"] == 1, (
        f"max_lines must clamp to the _STYLE_RANGES min (1); got "
        f"{msg['max_lines']!r}"
    )
    assert msg["jp_size"] == 30, "in-range sibling key must apply unclamped"
    # The persisted overrides must carry the CLAMPED values -- out-of-range
    # values on disk would outlive the session and re-enter on every boot.
    import yaml
    data = yaml.safe_load(paths.user_overrides_path().read_text(encoding="utf-8"))
    assert data["overlay"]["style"]["hold_ms"] == 15000
    assert data["overlay"]["style"]["max_lines"] == 1
    assert data["overlay"]["style"]["jp_size"] == 30


def test_set_style_non_string_font_broadcasts_default_font(tmp_path, monkeypatch):
    """A non-string jp_font from any WS client used to survive the clamp, be
    broadcast and persisted; the Qt overlay's QFont(5, ...) then raises in
    every add_subtitle and drops every subtitle until a restart. The clamp
    must replace it with the shipped default family.

    The state is seeded with a non-default family first, so dropping the
    key (instead of replacing it) would still broadcast the seed and fail.

    Seeded defect: remove the font-key branch from config._clamp_style ->
    the broadcast carries jp_font=5 -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko.display.browser_server import build_app
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.state import AppState

    state = AppState()
    state.set_style({"jp_font": "Meiryo"})
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.receive_json()  # initial state
        ws.receive_json()  # initial style
        ws.receive_json()  # initial problems
        ws.send_json({
            "type": "command",
            "action": "set_style",
            "style": {"jp_font": 5},
        })
        msg = ws.receive_json()
    assert msg["type"] == "style"
    assert msg["jp_font"] == STYLE_DEFAULTS["jp_font"], (
        f"non-string jp_font was not replaced by the default family: "
        f"{msg['jp_font']!r}"
    )


def test_initial_style_sent_on_connect():
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        seen_types: list[str] = []
        for _ in range(2):
            msg = ws.receive_json()
            seen_types.append(msg["type"])
        assert "state" in seen_types
        assert "style" in seen_types


def test_quit_command_invokes_on_quit_callback():
    """Tray Quit -> WS `quit` command -> build_app's on_quit callback fires.
    main.py passes done.set here so the pipeline's finally-block cleanup
    (glossaries save, translator aclose, etc.) runs before exit.
    """
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    quit_calls: list[bool] = []
    app = build_app(
        state=state, on_quit=lambda: quit_calls.append(True), bound_port=8765,
        ws_token=_TOKEN,
    )
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        # Drain the initial state + style + problems frames.
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"type": "command", "action": "quit"})
    assert quit_calls == [True], (
        f"on_quit must fire exactly once on quit command; saw {quit_calls}"
    )


def test_quit_command_without_callback_is_noop():
    """If on_quit isn't wired (e.g., tests building a bare app), the
    quit command is logged but doesn't crash. Defensive against
    partial setups.
    """
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()
    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)  # on_quit=None
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()
        # Must not raise
        ws.send_json({"type": "command", "action": "quit"})


# ---------------------------------------------------------------------------
# Cross-site WebSocket hijacking guard.
#
# The overlay WS accepted any connection with no Origin check. A browser is
# free to open ws://127.0.0.1:<port>/ws from ANY page the user visits and
# send command frames -- including {"action":"quit"} (shuts the app down) and
# {"action":"set_style", ...} (writes config.yaml to disk). Browsers always
# attach an Origin header; native clients (the Qt overlay's QWebSocket, CLI
# tools) attach none. The guard rejects a browser Origin that is not a
# loopback origin on the bound port. The rejection tests below carry a VALID
# token, so the Origin guard (not the token gate) is what refuses them.
# ---------------------------------------------------------------------------
from starlette.websockets import WebSocketDisconnect  # noqa: E402


def test_websocket_rejects_cross_site_origin():
    """A connection bearing a non-loopback browser Origin must be refused
    at the handshake (no accept, client never registered), even with the
    valid session token.

    Seeded-defect check: remove the Origin guard in build_app's ws() ->
    the tokened malicious-origin handshake succeeds -> this fails.
    """
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            _WS, headers={"origin": "https://evil.example"}
        ):
            pass
    assert mgr.client_count() == 0, (
        "a cross-site Origin was allowed to register a WS client"
    )


def test_websocket_rejects_cross_site_origin_cannot_quit():
    """A cross-site page must not be able to drive the `quit` command
    (which fires on_quit -> app shutdown). The handshake is rejected
    before any frame is processed.
    """
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    quit_calls: list[bool] = []
    app = build_app(
        state=AppState(), on_quit=lambda: quit_calls.append(True),
        bound_port=8765, ws_token=_TOKEN,
    )
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            _WS, headers={"origin": "http://attacker.test"}
        ) as ws:
            ws.send_json({"type": "command", "action": "quit"})
    assert quit_calls == [], "cross-site origin reached the quit command"


def _browser_connect_kwargs(client: TestClient, origin: str) -> dict:
    """The legitimate browser-overlay flow: the page fetches the per-session
    token from /ws-token (same-origin, so no Origin header on the GET)
    and appends it to the WS URL as ?t=."""
    token = client.get("/ws-token").json()["token"]
    return {"url": f"/ws?t={token}", "headers": {"origin": origin}}


def test_websocket_allows_loopback_origin():
    """The local browser overlay (served by this very server) fetches the
    session token and connects with a loopback Origin; it must be
    accepted."""
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    kw = _browser_connect_kwargs(client, "http://127.0.0.1:8765")
    with client.websocket_connect(kw["url"], headers=kw["headers"]) as ws:
        assert mgr.client_count() == 1
        msg = ws.receive_json()
        assert msg["type"] in ("state", "style")


def test_websocket_allows_localhost_origin():
    """localhost (not just 127.0.0.1) is a valid loopback overlay origin."""
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    kw = _browser_connect_kwargs(client, "http://localhost:8765")
    with client.websocket_connect(kw["url"], headers=kw["headers"]) as ws:
        assert mgr.client_count() == 1


# ---------------------------------------------------------------------------
# Per-session WS auth token, defense-in-depth on top of the Origin port
# pin. The Origin allowlist trusts a loopback origin on the server's own
# bound port; the token closes the residual where a page on that same
# origin tries to connect without ever having loaded from here. A browser
# connection (Origin header present) must also present the per-launch
# random token, which only a page served from THIS server's own origin can
# read (/ws-token sends no CORS headers, so a cross-origin page's fetch
# can't read the body). Native clients (no Origin) must present it too: the
# Qt overlay receives it in its environment, and 127.0.0.1 is shared by
# every logged-in Windows session.
# ---------------------------------------------------------------------------


def test_websocket_rejects_matching_origin_without_session_token():
    """A browser Origin on the server's OWN bound port (so the Origin
    host+port check passes) but WITHOUT the per-session ?t= token must
    still be refused: only a page actually served from this origin could
    have read the token from /ws-token, so a tokenless same-origin
    handshake is a page that never loaded from here. The token gate is
    the load-bearing check once the port check has passed.

    (The DIFFERENT-port loopback edge -- another localhost service's
    page -- is now rejected by the Origin port check itself; see
    test_websocket_rejects_other_loopback_port.)

    Seeded defect: remove the token check in ws() -> the tokenless
    handshake on the matching origin succeeds -> red.
    """
    app = _get_app()  # bound_port=8765
    mgr = app.state.manager
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws", headers={"origin": "http://127.0.0.1:8765"}
        ):
            pass
    assert mgr.client_count() == 0, (
        "a tokenless handshake on the server's own origin registered a "
        "WS client"
    )


def test_websocket_rejects_loopback_origin_with_wrong_token():
    """A guessed/stale token must not pass: the gate is the per-launch
    value, not the mere presence of a ?t= param.

    Seeded defect: replace the compare_digest token comparison with
    `token is not None` -> red.
    """
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws?t=not-the-session-token",
            headers={"origin": "http://127.0.0.1:8765"},
        ):
            pass
    assert mgr.client_count() == 0


def test_websocket_without_origin_or_token_is_closed_with_1008():
    """127.0.0.1 is shared by every logged-in Windows session, so a native
    client (no Origin header) must present the per-launch token too: a
    token-free process in another session could otherwise read every
    subtitle and send quit / set_style.

    Seeded defect: restore the `if origin:` guard around the token check
    in ws() -> the token-free native handshake is accepted -> red.
    """
    app = _get_app()
    mgr = app.state.manager
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws"):
            pass
    assert exc_info.value.code == 1008, (
        f"token-free native handshake closed with {exc_info.value.code}, "
        f"expected 1008 (policy violation)"
    )
    assert mgr.client_count() == 0


def test_ws_token_endpoint_rejects_cross_site_origin():
    """Defense in depth: a cross-site Origin on the token endpoint gets a
    403 (browsers' same-origin policy already hides the body, since the
    response carries no CORS headers)."""
    app = _get_app()
    client = TestClient(app)
    r = client.get("/ws-token", headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_overlay_js_wires_session_token_into_ws_url():
    """Static tripwire (no JS harness in this suite): overlay.js must
    fetch /ws-token and put the token into the /ws URL's ?t= param,
    otherwise the served page can no longer connect to its own server.

    Seeded defect: revert overlay.js to the bare `${location.host}/ws`
    URL -> red.
    """
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    js = (Path(bs.__file__).parent / "static" / "overlay.js").read_text(
        encoding="utf-8"
    )
    assert "/ws-token" in js, "overlay.js must fetch the session token"
    assert "?t=" in js, "overlay.js must append the token to the WS URL"


def test_overlay_js_maps_max_width_pct_to_css_property_in_vw_units():
    """Static tripwire (no JS harness in this suite): overlay.js applyStyle
    must map the max_width_pct style key onto the `--max-width` CSS custom
    property IN `vw` UNITS, so the browser overlay caps subtitle width like
    the Qt overlay (pct x screen width). `vw` resolves against the viewport,
    matching Qt's screen-width basis; a `%` value was the cyclic-reference
    bug when the cap sat on the shrink-to-fit subtitle inside `#stage`
    (it resolved against the subtitle's OWN width, so the cap silently did
    nothing). Behavioral coverage is the
    set_style wire test above plus the live-overlay smoke (Task 13); this
    guards against a silent revert that would re-diverge the two renderers.

    Seeded defects (each one line, each goes red):
    - drop the `--max-width` setProperty line from overlay.js applyStyle.
    - revert the emitted unit from `${...}vw` to `${...}%` -> the `}vw`
      template tail is gone AND the banned `}%` tail reappears.
    """
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    js = (Path(bs.__file__).parent / "static" / "overlay.js").read_text(
        encoding="utf-8"
    )
    assert "max_width_pct" in js, "overlay.js must read the max_width_pct key"
    assert "--max-width" in js, "overlay.js must set the --max-width property"
    assert "}vw`" in js, (
        "overlay.js must emit --max-width in vw units (Qt parity: "
        "pct x screen width)"
    )
    assert "}%`" not in js, (
        "overlay.js emits --max-width in %; emit vw (Qt parity: pct x "
        "screen width; a % cap on the shrink-to-fit lines is cyclic and "
        "caps nothing)"
    )


def test_overlay_js_max_width_guards_non_number_and_recovers_css_default():
    """Static tripwire (no JS harness in this suite -- disclosed limitation):
    overlay.js applyStyle must guard --max-width on ``typeof number`` +
    Number.isFinite, NOT Number(), because Number(null) / Number(false) /
    Number("") all coerce to 0 (finite) and would clamp the browser overlay to
    30vw while Qt falls back to 70. And it must
    ``removeProperty('--max-width')`` on a non-number so the CSS
    ``var(--max-width, 70vw)`` default recovers -- including when a later
    invalid frame must clear a previously-set good value.

    Limitation (same as the sibling tripwires): this asserts the guard SOURCE is
    present, not its runtime behaviour -- there is no JS engine in this suite.
    The Python-side _sanitize_style + broadcast parity is covered by
    test_set_style_broadcasts_max_width_pct_through_sanitize above.

    Seeded defects (each one line, each goes red):
    - revert the guard to ``const cap = Number(msg.max_width_pct); if
      (Number.isFinite(cap))`` -> the ``typeof msg.max_width_pct === "number"``
      substring is gone -> first assertion red.
    - drop the ``else { r.removeProperty('--max-width'); }`` branch -> the
      removeProperty substring is gone -> second assertion red.
    """
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    js = (Path(bs.__file__).parent / "static" / "overlay.js").read_text(
        encoding="utf-8"
    )
    assert 'typeof msg.max_width_pct === "number"' in js, (
        "overlay.js must guard --max-width on typeof number, not Number() "
        "(Number(null/false/'') === 0 would clamp to 30vw not the 70vw default)"
    )
    assert "removeProperty('--max-width')" in js, (
        "overlay.js must removeProperty('--max-width') on a non-number so the "
        "CSS var(--max-width, 70vw) default recovers"
    )


def test_overlay_css_stage_caps_width_with_max_width_vw_and_en_wraps():
    """Static tripwire: the #stage rule must consume `--max-width` with a
    `70vw` fallback (the STYLE_DEFAULTS 70, in the same viewport units
    overlay.js emits) and the English block must wrap overlong unbroken
    text so the width cap is actually visible. A `70%` fallback is the
    cyclic-percentage bug: on the shrink-to-fit lines inside the stack it
    resolves against the line's own width and caps nothing. Pairs with
    overlay.js setting the property.

    Seeded defects (each one line, each goes red):
    - remove `width: var(--max-width, 70vw)` from the #stage rule (or the
      overflow-wrap from the .en rule) in overlay.css.
    - revert the fallback to `70%` -> the vw assert AND the banned-% assert
      both fire.
    """
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    css = (Path(bs.__file__).parent / "static" / "overlay.css").read_text(
        encoding="utf-8"
    )
    assert "var(--max-width, 70vw)" in css, (
        "#stage must consume --max-width with the 70vw fallback"
    )
    assert "var(--max-width, 70%)" not in css, (
        "a % fallback is cyclic on the shrink-to-fit lines inside the stack "
        "(resolves against the line's own width); use 70vw"
    )
    assert "overflow-wrap" in css, ".en must wrap overlong unbroken text"


def test_status_badge_fades_after_connect_reshows_on_change_and_hides_with_stage():
    """Static tripwire (no JS harness in this suite -- same disclosed
    limitation as the --max-width tripwires): the "connected" status pill
    used to render in the page corner FOREVER, so OBS captured a permanent
    green badge, and Ctrl+Alt+O (toggle_overlay) hid only #stage while the
    pill stayed on the stream. Pins the source of three behaviors:

      1. overlay.js fades the healthy badge: a setTimeout adds the `faded`
         class after a confirmation window, and every setStatus clears the
         pending timer + rebuilds className (dropping `faded`), so a
         disconnect/error re-shows the pill immediately.
      2. overlay.css transitions .status on opacity and defines
         .status.faded { opacity: 0 }.
      3. overlay.js's toggle_overlay branch hides the badge together with
         #stage.

    Seeded defects (one line each, each goes red):
    - drop the `classList.add("faded")` scheduling line -> (1) red.
    - drop the `clearTimeout(statusFadeTimer)` line -> (1) red.
    - drop the `.status.faded` rule or the .status opacity transition
      -> (2) red.
    - drop the `status.classList.toggle("hidden", ...)` line in the
      toggle_overlay branch -> (3) red.
    """
    import re
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    static = Path(bs.__file__).parent / "static"
    js = (static / "overlay.js").read_text(encoding="utf-8")
    css = (static / "overlay.css").read_text(encoding="utf-8")

    # (1) fade scheduled, and re-armed/cleared on every status change.
    assert 'classList.add("faded")' in js, (
        "overlay.js must fade the healthy status badge (add the faded "
        "class after a delay) or OBS captures a permanent green pill"
    )
    assert "clearTimeout(statusFadeTimer)" in js, (
        "setStatus must clear the pending fade timer so a disconnect "
        "re-shows the badge immediately instead of fading it mid-error"
    )
    # (2) the CSS side of the fade.
    assert re.search(r"\.status\s*\{[^}]*transition:\s*opacity", css), (
        ".status must transition on opacity so the badge fades rather "
        "than popping out"
    )
    assert re.search(r"\.status\.faded\s*\{[^}]*opacity:\s*0", css), (
        ".status.faded { opacity: 0 } missing -- the faded class would "
        "do nothing"
    )
    # (3) Ctrl+Alt+O blanks the WHOLE capture, badge included.
    assert 'status.classList.toggle("hidden"' in js, (
        "the toggle_overlay branch must hide the status badge together "
        "with #stage"
    )


# ---------------------------------------------------------------------------
# _origin_allowed pins a browser Origin to the server's own bound port
# (main.py passes the uvicorn bind port as a required arg). A page served
# by a DIFFERENT localhost service (dev server, Jupyter, another app's web
# UI) shares the loopback host but not the port, so it no longer passes the
# Origin gate. Native clients (no Origin) stay allowed.
# ---------------------------------------------------------------------------


def test_origin_allowed_rejects_other_loopback_port_when_bound():
    """Same loopback host, different port = a different localhost service's
    page; with bound_port set it must be rejected.

    Seeded defect: drop the port comparison in _origin_allowed -> the
    :9999 origin passes -> red.
    """
    from yamabiko.display.browser_server import _origin_allowed

    assert _origin_allowed("http://127.0.0.1:9999", bound_port=8765) is False
    assert _origin_allowed("http://localhost:3000", bound_port=8765) is False
    # Origin without an explicit port = served from the scheme default
    # (http 80 / https 443) -- a different service when we bind 8765.
    assert _origin_allowed("http://localhost", bound_port=8765) is False


def test_origin_allowed_normalizes_scheme_default_port():
    """A browser omits the port from Origin when it is the scheme default
    (http://localhost:80 -> Origin: http://localhost), so urlsplit yields
    port=None. That must compare as 80/443 -- not as never-equal None --
    otherwise binding overlay.websocket_port to 80 or 443 makes the server
    reject its OWN page's WebSocket.

    Seeded defect (one line): drop the `if port is None:` normalization in
    _origin_allowed -> the portless http origin no longer matches
    bound_port=80 -> red.
    """
    from yamabiko.display.browser_server import _origin_allowed

    # http default: 80.
    assert _origin_allowed("http://localhost", bound_port=80) is True
    assert _origin_allowed("http://127.0.0.1", bound_port=80) is True
    # https default: 443.
    assert _origin_allowed("https://localhost", bound_port=443) is True
    # The defaults must not cross over.
    assert _origin_allowed("http://localhost", bound_port=443) is False
    assert _origin_allowed("https://localhost", bound_port=80) is False
    # An EXPLICIT default port still matches (urlsplit keeps it).
    assert _origin_allowed("http://localhost:80", bound_port=80) is True
    # Normalization must not loosen the host allowlist.
    assert _origin_allowed("http://evil.example", bound_port=80) is False


def test_origin_allowed_accepts_matching_port_and_native_no_origin():
    from yamabiko.display.browser_server import _origin_allowed

    assert _origin_allowed("http://127.0.0.1:8765", bound_port=8765) is True
    assert _origin_allowed("http://localhost:8765", bound_port=8765) is True
    # Native clients (Qt QWebSocket, CLI) send no Origin at all.
    assert _origin_allowed(None, bound_port=8765) is True
    assert _origin_allowed("", bound_port=8765) is True


def test_origin_allowed_rejects_non_loopback_host():
    """A non-loopback browser Origin is refused regardless of port -- the
    host allowlist is checked before the port comparison, so even a
    matching-port origin on a foreign host is rejected.

    Seeded defect: drop the `host not in _LOOPBACK_HOSTS` check in
    _origin_allowed -> evil.example passes -> red.
    """
    from yamabiko.display.browser_server import _origin_allowed

    assert _origin_allowed("https://evil.example", bound_port=8765) is False
    assert _origin_allowed("http://evil.example:8765", bound_port=8765) is False


def test_websocket_rejects_other_loopback_port(monkeypatch):
    """With build_app(bound_port=8765), a handshake bearing Origin
    http://127.0.0.1:9999 (another localhost service's page) is refused
    before accept and never registered.

    Seeded defect: stop threading bound_port from build_app into the
    ws() origin check -> the wrong-port handshake succeeds -> red.
    """
    from yamabiko.display.browser_server import build_app

    app = build_app(bound_port=8765, ws_token=_TOKEN)
    mgr = app.state.manager
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            _WS, headers={"origin": "http://127.0.0.1:9999"}
        ):
            pass
    assert mgr.client_count() == 0, (
        "an off-port loopback Origin was allowed to register a WS client"
    )


# ---------------------------------------------------------------------------
# bound_port must thread END-TO-END from the build_app argument into the
# ws() closure's origin check. Every other wire test in this file builds
# the app at 8765 -- the production default -- so a refactor that ignores
# the parameter and hardcodes 8765 inside build_app would pass all of
# them. These two tests bind a NON-default port (9099) and prove both
# directions at the wire: the matching-port origin is accepted, and the
# production-default-port origin is refused.
# ---------------------------------------------------------------------------


def test_websocket_accepts_tokened_loopback_origin_on_nondefault_port():
    """build_app(bound_port=9099) must accept a tokened browser handshake
    whose Origin is http://127.0.0.1:9099 -- the ws() origin check compares
    against the bound_port ARGUMENT, not a baked-in copy of the production
    default.

    Seeded defect: hardcode 8765 into the ws() closure's _origin_allowed
    call (`bound_port=8765` instead of the build_app parameter) -> this
    matching-port handshake is refused -> red.
    """
    from yamabiko.display.browser_server import build_app

    app = build_app(bound_port=9099, ws_token=_TOKEN)
    mgr = app.state.manager
    client = TestClient(app)
    kw = _browser_connect_kwargs(client, "http://127.0.0.1:9099")
    with client.websocket_connect(kw["url"], headers=kw["headers"]) as ws:
        assert mgr.client_count() == 1
        assert ws.receive_json()["type"] in ("state", "style")
    assert mgr.client_count() == 0


def test_websocket_rejects_production_default_port_origin_when_bound_elsewhere():
    """With build_app(bound_port=9099), a TOKENED handshake bearing Origin
    http://127.0.0.1:8765 (the production default -- here just another
    localhost service's page) must be refused by the ORIGIN check. The
    valid token matters: without it, a defect that hardcodes 8765 into
    the origin check would still see this handshake rejected -- by the
    token gate -- and the test could not tell the two rejections apart.

    Seeded defect: hardcode 8765 into build_app's origin check -> the
    8765-origin handshake passes the origin gate, its valid token passes
    the token gate, the connection registers -> red.
    """
    from yamabiko.display.browser_server import build_app

    app = build_app(bound_port=9099, ws_token=_TOKEN)
    mgr = app.state.manager
    client = TestClient(app)
    kw = _browser_connect_kwargs(client, "http://127.0.0.1:8765")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(kw["url"], headers=kw["headers"]):
            pass
    assert mgr.client_count() == 0, (
        "an Origin pinned to the production default port was allowed to "
        "register on a server bound elsewhere -- bound_port is not "
        "reaching the origin check"
    )


# ---------------------------------------------------------------------------
# The on-connect state/style snapshot send must not be wrapped in a bare
# `except Exception: pass` -- that would silently swallow errors,
# contradicting the "logged at WARNING and dropped" contract and hiding
# snapshot() programming errors entirely.
# ---------------------------------------------------------------------------
import logging  # noqa: E402


def test_initial_send_failure_is_logged_at_debug(caplog, monkeypatch):
    """A client that disconnects between accept() and the initial
    state/style send is an expected race; it must be visible at DEBUG
    in the browser_server log, not silently swallowed.

    Seeded defect: use a bare `except Exception: pass` around the
    initial snapshot send -> no log record -> red.
    """
    from starlette.websockets import WebSocket

    async def _gone(self, data, mode="text"):
        raise WebSocketDisconnect(1006)

    monkeypatch.setattr(WebSocket, "send_json", _gone)
    caplog.set_level(logging.DEBUG, logger="yamabiko.display.browser_server")

    app = _get_app()
    client = TestClient(app)
    with client.websocket_connect(_WS):
        pass

    records = [
        r for r in caplog.records
        if r.name == "yamabiko.display.browser_server"
        and r.levelno == logging.DEBUG
        and "initial send failed" in r.getMessage()
    ]
    assert records, (
        "a failed initial state/style send must produce a DEBUG record; "
        f"got records: {[r.getMessage() for r in caplog.records]}"
    )


def test_unexpected_initial_snapshot_error_is_logged_and_client_removed(
    caplog, monkeypatch,
):
    """An unexpected error in the on-connect snapshot path (e.g. a future
    AppState.snapshot() regression) must hit the handler's WARNING
    exc_info log and still run the finally: manager.remove() -- not
    propagate out of ws() leaving a ghost client registered.

    Seeded defect: move the initial-send block outside the outer
    try/finally -> ValueError escapes the handler, no WARNING record,
    client never removed -> red.
    """
    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    state = AppState()

    def _boom():
        raise ValueError("snapshot regression")

    monkeypatch.setattr(state, "snapshot", _boom)
    caplog.set_level(logging.DEBUG, logger="yamabiko.display.browser_server")

    app = build_app(state=state, bound_port=8765, ws_token=_TOKEN)
    mgr = app.state.manager
    client = TestClient(app)
    with client.websocket_connect(_WS):
        pass

    warnings = [
        r for r in caplog.records
        if r.name == "yamabiko.display.browser_server"
        and r.levelno == logging.WARNING
        and "websocket handler error" in r.getMessage()
        and r.exc_info is not None
    ]
    assert warnings, "snapshot regression must be logged at WARNING with exc_info"
    assert mgr.client_count() == 0, (
        "finally: manager.remove() must still run on an unexpected error"
    )


def test_ws_on_connect_snapshot_sends_are_bounded_by_two_second_timeout(
    monkeypatch,
):
    """The on-connect state, style and problems snapshot sends must each run under
    asyncio.wait_for(..., timeout=2.0) -- the same bound
    ConnectionManager._send_one applies to broadcasts. An unbounded
    send_json parks the handler coroutine until TCP timeout whenever a
    client connects and immediately stops reading, leaking one handler
    per reconnect attempt.

    A recording asyncio.wait_for wrapper installed in browser_server's
    module namespace records every timeout used during a NORMAL ws connect,
    and at least three awaits with timeout=2.0 must occur (the three snapshot
    sends). The snapshot must also still ARRIVE (state, style and problems
    frames received) so the bound can't be satisfied by sends that no longer
    happen.

    Seeded defect: call websocket.send_json directly (drop the three
    wait_for wrappers, or only the problems one) in browser_server's ws
    route -> fewer than three 2.0-timeout awaits recorded -> red.
    """
    import asyncio
    import types

    import yamabiko.display.browser_server as bs

    recorded: list = []
    real_wait_for = asyncio.wait_for

    async def _recording_wait_for(awaitable, timeout=None, **kw):
        recorded.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout, **kw)

    # Scope the recorder to browser_server's own `asyncio` reference so
    # unrelated wait_for callers (TestClient internals, the broadcast
    # manager) can't satisfy the count. TimeoutError is forwarded for
    # the route's except clause, Lock for build_app's persist lock.
    monkeypatch.setattr(
        bs, "asyncio",
        types.SimpleNamespace(
            wait_for=_recording_wait_for,
            TimeoutError=asyncio.TimeoutError,
            Lock=asyncio.Lock,
        ),
    )

    app = bs.build_app(bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    with client.websocket_connect(_WS) as ws:
        seen_types = {ws.receive_json()["type"] for _ in range(3)}
    assert seen_types == {"state", "style", "problems"}, (
        f"on-connect snapshot frames missing; got {seen_types} -- the "
        f"bound is meaningless if the sends themselves are gone"
    )
    bounded = [t for t in recorded if t == 2.0]
    assert len(bounded) >= 3, (
        f"expected >=3 asyncio.wait_for awaits with timeout=2.0 during "
        f"the ws connect (the state, style and problems snapshot sends); "
        f"recorded timeouts: {recorded!r}"
    )


# ---------------------------------------------------------------------------
# _handle_command runs on the pipeline's hot event loop (the same loop that
# drives ASR -> translation -> broadcast). write_override /
# write_overlay_style / save_active_series are fsync-bearing disk writes:
# executed inline they stall EVERY in-flight subtitle for the duration of a
# contended fsync (AV scan, sleeping disk). They must be pushed off-loop via
# asyncio.to_thread -- the same idiom main.py uses for the glossary save.
# ---------------------------------------------------------------------------


def test_handle_command_disk_writers_run_off_the_event_loop_thread(monkeypatch):
    """Each persisting command action (set_genre, set_style, set_series) must
    invoke its disk writer on a worker thread, not the event-loop thread --
    AND the commands must still take observable effect (state mutated,
    writer actually called with the persisted value).

    Pure-asyncio: _handle_command is driven directly with monkeypatched
    writers that record the invoking thread.

    Seeded defect (one line each): revert any one of the three
    `await asyncio.to_thread(<writer>, ...)` calls in _handle_command to
    the bare synchronous call -> that writer records the loop thread ->
    red.
    """
    import asyncio
    import threading

    import yamabiko.display.browser_server as bs
    from yamabiko.display.broadcast import ConnectionManager
    from yamabiko.state import AppState

    writer_threads: dict[str, threading.Thread] = {}
    writer_args: dict[str, tuple] = {}

    def _recorder(name):
        def _rec(*args):
            writer_threads[name] = threading.current_thread()
            writer_args[name] = args
        return _rec

    monkeypatch.setattr(bs, "write_override", _recorder("write_override"))
    monkeypatch.setattr(bs, "write_overlay_style", _recorder("write_overlay_style"))
    monkeypatch.setattr(bs, "save_active_series", _recorder("save_active_series"))

    state = AppState()
    manager = ConnectionManager()

    async def _run() -> threading.Thread:
        loop_thread = threading.current_thread()
        await bs._handle_command(
            '{"type":"command","action":"set_genre","genre":"news"}',
            state, manager,
        )
        await bs._handle_command(
            '{"type":"command","action":"set_style","style":{"jp_size":30}}',
            state, manager,
        )
        await bs._handle_command(
            '{"type":"command","action":"set_series","media_id":7,"label":"X"}',
            state, manager,
        )
        return loop_thread

    loop_thread = asyncio.run(_run())

    # The commands still took effect (behavior first, scheduling second).
    assert state.genre == "news"
    assert state.snapshot()["style"]["jp_size"] == 30
    assert state.series_media_id == 7
    assert writer_args["write_override"] == (["genre"], "news")
    assert writer_args["write_overlay_style"][0]["jp_size"] == 30
    assert writer_args["save_active_series"] == (7, "X", None)

    for name in ("write_override", "write_overlay_style", "save_active_series"):
        assert name in writer_threads, f"{name} was never invoked"
        assert writer_threads[name] is not loop_thread, (
            f"{name} ran ON the event-loop thread -- an fsync there stalls "
            f"every in-flight subtitle; it must go through asyncio.to_thread"
        )


# ---------------------------------------------------------------------------
# The awaited off-loop persist yields the event loop mid-command, so two
# near-simultaneous persisting commands from different clients can
# interleave: the FIRST (older) full-style snapshot finishes its write and
# broadcast AFTER the second, leaving stale style on disk and on every
# client. The merge + persist + broadcast section must run as one ordered
# critical section per app.
# ---------------------------------------------------------------------------


def test_concurrent_set_style_frames_persist_and_broadcast_second_merge_last(
    monkeypatch,
):
    """Two set_style frames sent while the first one's persist is stalled
    on its worker thread must still land in merge order: the last persist
    to COMPLETE and the last style frame broadcast to every client must
    both carry the second merge (jp_size 31), never the older snapshot
    (jp_size 30).

    The persist writer is monkeypatched with an ordering barrier: the
    first write parks until released, while the second command is sent
    and given time to run. Completion order of the recorded persist
    payloads stands in for what would land last on disk.

    Seeded defect: drop the persist-lock serialization around the
    set_style merge/persist/broadcast section in _handle_command -> the
    second command completes during the stall, then the released first
    write lands last and its stale jp_size=30 frame is broadcast last ->
    red.
    """
    import threading

    import yamabiko.display.browser_server as bs
    from yamabiko.state import AppState

    first_write_started = threading.Event()
    release_first_write = threading.Event()
    second_write_done = threading.Event()
    completed: list[dict] = []  # persist payloads, in completion order

    def _barrier_writer(persist: dict) -> None:
        # The first write always reaches its worker thread before the
        # second command is even sent (the test waits on the event), so
        # the branch below cannot race.
        if not first_write_started.is_set():
            first_write_started.set()
            assert release_first_write.wait(timeout=10), "barrier leak"
            completed.append(persist)
        else:
            completed.append(persist)
            second_write_done.set()

    monkeypatch.setattr(bs, "write_overlay_style", _barrier_writer)

    app = bs.build_app(state=AppState(), bound_port=8765, ws_token=_TOKEN)
    client = TestClient(app)
    # `with client:` shares one portal (one event loop) across both WS
    # sessions, matching production where every handler runs on the
    # pipeline loop.
    with client:
        with client.websocket_connect(_WS) as ws1, \
                client.websocket_connect(_WS) as ws2:
            for ws in (ws1, ws2):
                ws.receive_json()  # initial state
                ws.receive_json()  # initial style
                ws.receive_json()  # initial problems
            ws1.send_json({
                "type": "command", "action": "set_style",
                "style": {"jp_size": 30},
            })
            assert first_write_started.wait(timeout=10), (
                "first persist never reached its worker thread"
            )
            ws2.send_json({
                "type": "command", "action": "set_style",
                "style": {"jp_size": 31},
            })
            # Give the second command a window to run while the first
            # persist is stalled. Unordered handling completes it here
            # (the event fires fast); ordered handling parks it behind
            # the first command and this wait times out, as intended.
            second_write_done.wait(timeout=1.0)
            release_first_write.set()
            frames1 = [ws1.receive_json(), ws1.receive_json()]
            frames2 = [ws2.receive_json(), ws2.receive_json()]

    assert [f["type"] for f in frames1] == ["style", "style"]
    assert frames1[-1]["jp_size"] == 31, (
        "the last style frame client 1 saw is the OLDER snapshot: "
        f"{[f['jp_size'] for f in frames1]}"
    )
    assert frames2[-1]["jp_size"] == 31, (
        "the last style frame client 2 saw is the OLDER snapshot: "
        f"{[f['jp_size'] for f in frames2]}"
    )
    assert completed and completed[-1]["jp_size"] == 31, (
        "the last persist to complete is what survives on disk; the older "
        f"snapshot landed last: {[p['jp_size'] for p in completed]}"
    )


# ---------------------------------------------------------------------------
# The 1011 close on the initial-snapshot-timeout path flushes through the
# same stalled transport that just timed out the send, so an unbounded
# close would park the ws handler for the transport's close_timeout on
# every reconnect of a never-reading client.
# ---------------------------------------------------------------------------


def test_snapshot_timeout_close_is_bounded_and_client_deregistered():
    """When the initial state/style snapshot send times out, the handler
    must attempt the 1011 close, give up on it within the same kind of
    bound as the sends, return, and deregister the client -- even when
    the close itself never returns.

    Driven by invoking the /ws route endpoint directly with a fake
    websocket whose send times out and whose close hangs forever; the
    ASGI layer adds nothing to this path.

    Seeded defect: drop the asyncio.wait_for bound around the 1011 close
    in the snapshot-timeout branch -> the endpoint awaits the hung close
    forever -> the harness wait_for below raises TimeoutError -> red.
    """
    import asyncio

    from yamabiko.display.browser_server import build_app

    app = build_app(bound_port=8765, ws_token=_TOKEN)
    mgr = app.state.manager
    endpoint = next(
        r.endpoint for r in app.router.routes
        if getattr(r, "path", None) == "/ws"
    )

    class _FrozenWS:
        """No Origin (native-client path); send times out, close hangs."""

        def __init__(self):
            self.headers: dict = {}
            self.query_params: dict = {"t": _TOKEN}
            self.accepted = False
            self.close_started = asyncio.Event()
            self._never = asyncio.Event()

        async def accept(self) -> None:
            self.accepted = True

        async def send_json(self, payload: dict) -> None:
            # Surface the same TimeoutError the route's 2s wait_for
            # would produce, without spending the 2s of wall clock.
            raise asyncio.TimeoutError()

        async def close(self, code: int = 1000) -> None:
            self.close_started.set()
            await self._never.wait()

    async def _run() -> "_FrozenWS":
        fake = _FrozenWS()
        # 6s ceiling: the bounded path needs ~2s (the close bound) and
        # the unbounded path never returns, so the outcome is separable.
        await asyncio.wait_for(endpoint(fake), timeout=6.0)
        return fake

    fake = asyncio.run(_run())
    assert fake.accepted
    assert fake.close_started.is_set(), "the 1011 close must still be attempted"
    assert mgr.client_count() == 0, (
        "the timed-out client must be deregistered by the finally block"
    )


# ---------------------------------------------------------------------------
# The set_series rejection warning goes to the pipeline log, which ships
# in diagnostics zips. The label is a watch-list series title, so the log
# line must carry its length/type only, never its text (the identity-free
# convention from overlay_app/series_palette.py).
# ---------------------------------------------------------------------------


def test_set_series_rejection_log_omits_label_text(caplog):
    """A rejected set_series frame must produce a WARNING that does NOT
    contain the label text anywhere in the record. media_id plus the
    label's length/type are enough to debug a malformed frame.

    Seeded defect: put `label` back in the rejection warning's format
    args in _handle_command -> the title text lands in the record -> red.
    """
    import asyncio
    import json

    import yamabiko.display.browser_server as bs
    from yamabiko.display.broadcast import ConnectionManager
    from yamabiko.state import AppState

    caplog.set_level(logging.WARNING, logger="yamabiko.display.browser_server")
    secret = "Extremely Private Watch-List Title"
    # media_id=True is a bool (rejected as a non-int) so the rejection
    # branch fires while a perfectly normal str label is in hand.
    raw = json.dumps({
        "type": "command", "action": "set_series",
        "media_id": True, "label": secret,
    })

    async def _run() -> None:
        await bs._handle_command(raw, AppState(), ConnectionManager())

    asyncio.run(_run())

    rejected = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "set_series rejected" in r.getMessage()
    ]
    assert rejected, "the rejection branch must produce a WARNING"
    for r in caplog.records:
        assert secret not in r.getMessage(), (
            f"watch-list series title leaked into the pipeline log: "
            f"{r.getMessage()!r}"
        )


def test_overlay_js_writes_banner_text_as_text_not_html():
    """Static tripwire (no JS harness in this suite): the banner shows
    AniList titles and series labels, which are remote data, so overlay.js
    must set them with textContent and never route them through innerHTML.

    Seeded defect: build the banner with innerHTML -> red."""
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    js = (Path(bs.__file__).parent / "static" / "overlay.js").read_text(encoding="utf-8")
    start = js.index("function retag(")
    body = js[start: js.index("\n  }\n", start)]
    assert "textContent" in body
    assert "innerHTML" not in body


def test_overlay_js_writes_notice_text_as_text_not_html():
    """Static tripwire (no JS harness in this suite, plan Decision 48): the
    notice line carries the audio device's friendly name from the problems
    frame, so overlay.js must set notice text with textContent and never
    route it through innerHTML, and the page must handle the frame at all.

    Seeded defects: build the notice with innerHTML, or drop the problems
    branch from the message handler -> red."""
    from pathlib import Path

    import yamabiko.display.browser_server as bs

    js = (Path(bs.__file__).parent / "static" / "overlay.js").read_text(encoding="utf-8")
    start = js.index("function noticeEl(")
    body = js[start: js.index("\n  }\n", start)]
    assert "textContent" in body
    assert "innerHTML" not in body
    assert 'payload.type === "problems"' in js
