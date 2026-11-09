"""FastAPI overlay server.

Responsibilities:
  - Serve the static browser overlay (HTML/CSS/JS) at `/` and `/static/*`.
  - Accept WebSocket connections at `/ws` and register them with a
    `ConnectionManager` so the pipeline can broadcast subtitles to every
    open tab.
  - Receive JSON command messages from clients and apply them to a
    shared `AppState` instance (`pause`, `resume`, `set_genre`).
  - Run embedded in the pipeline's asyncio loop via uvicorn.Server.

`build_app(state=...)` is a factory; each call returns a fresh FastAPI
app + `ConnectionManager` + default `AppState` if none is passed in.

Wire protocol (server sends):
  {"type": "subtitle", "jp": str, "jp_ruby": [{"text": str, "reading"?: str}, ...], "en": str, "t": float}   on each translation
  {"type": "state", "genre": str, "paused": bool, "series_media_id": int|null, "series_label": str|null, "series_native": str|null}          on connect + after state change
  {"type": "style", "jp_font": str, "jp_size": int, "en_font": str, "en_size": int, "text_color": str, "bg_color": str, "bg_opacity": int, "padding": int, "max_width_pct": int, "max_lines": int, "hold_ms": int, "fade_in_ms": int, "fade_out_ms": int, "mode": "watch"|"learn", "alignment": "left"|"center", "echo": "priority"|"both", "banner": "jp"|"jp_en", "reduce_motion": bool}   on connect + after a style change (AppState.style_frame is the one builder)
  {"type": "problems", "active": [{"kind": "translator"|"audio", "title": str, "body": str, "line": str, "recovery": str}, ...], "recovered": [{"kind": str, "recovery": str}, ...]}   on connect (right after style) + after every change; "recovered" (problems a previous pipeline restarted on) appears only in connect frames within 30 s of the start
  {"type": "notice", "title": str, "body": str, "settings_page"?: str}   a user-facing notice (the Qt tray shows it as a toast; settings_page, for example "shortcuts", makes a click open Settings at that page). Notices raised while the Qt overlay (the native client, which sends no Origin) isn't connected are queued on AppState and sent to the next native client right after its problems frame. Browser clients leave them queued.
  {"type": "toggle_overlay"}   Ctrl+Alt+O, or a toggle_overlay command from the tray flyout: every client shows or hides its subtitles

Wire protocol (client sends):
  {"type": "command", "action": "pause"}
  {"type": "command", "action": "resume"}
  {"type": "command", "action": "set_genre", "genre": "variety"|"drama"|"news"|"anime"}   persisted to user_overrides.yaml so it survives a restart
  {"type": "command", "action": "set_style", "style": {"jp_size": 30, ...}}   partial dict; merged into state, broadcast as a "style" frame, and persisted to user_overrides.yaml.
  {"type": "command", "action": "apply_style", "style": {...}, "reduce_motion"?: bool}   the Qt tray after a Settings Apply: merged and broadcast like set_style but not persisted (Settings already wrote user_overrides.yaml). The optional top-level reduce_motion is runtime only (Settings already persisted it) and rides the broadcast style frame.
  {"type": "command", "action": "set_series", "media_id": int, "label": str, "native"?: str}   records the active-series pointer (active_series.json) and broadcasts a "state" frame; native is the series' Japanese title for the overlay banner (omitted or empty when unknown). The overlay pre-builds the gazetteer cache before sending. "media_id": null (the key present) clears the series instead: AppState drops it, active_series.json is deleted and the "state" frame carries null series fields. Rejected (no mutation, no persist, no broadcast) when the gazetteer feature is disabled.
  {"type": "command", "action": "toggle_overlay"}   the tray flyout's Subtitles tile: broadcast {"type": "toggle_overlay"} to every client, the frame the Ctrl+Alt+O shortcut sends (PipelineApp.handle_toggle_overlay_hotkey). Nothing is persisted.
  {"type": "command", "action": "set_hotkeys", "hotkeys": {"pause_resume"|"toggle_overlay"|"show_series": str}, "replay"?: true}   the Qt tray after a Settings Apply that changed shortcuts: the pipeline re-binds them at once (Settings already persisted them). Rejected whole unless every key is a known shortcut and every combo is bindable; combos are passed on in their stored spelling ("Ctrl+Alt+P"). "replay": true marks the tray's reconnect replay of shortcuts saved while offline: the pipeline then announces a conflict only for a shortcut whose combo changed.
  {"type": "command", "action": "set_incremental", "enabled": bool}   the Qt tray after a Settings Apply that changed Faster subtitles (asr.incremental.enabled; Settings already persisted it): the ASR stage uses it from its next speech window. Rejected unless "enabled" is a bool.

WS auth: run_pipeline generates one random session token per launch and
passes it to build_app(ws_token=...). EVERY connection must pass it as
`/ws?t=<token>`, with or without an Origin header: 127.0.0.1 is shared by
all logged-in Windows sessions, so a token-free native client in another
session could otherwise read transcripts and send commands. The Qt
overlay receives the token in its YAMABIKO_WS_TOKEN environment variable;
the browser overlay page reads it from GET /ws-token, which a cross-origin
local page cannot do (no CORS headers -> the browser's same-origin policy
hides the body). A browser connection must additionally carry a loopback
Origin on the bound port.

Residual: while the browser page is served (overlay.browser, the default),
GET /ws-token returns the token to any loopback client that sends no
Origin, so a local process that asks for it can still connect. Only with
the page off does the token stay inside this user's Yamabiko processes.

Unknown or malformed messages are logged at WARNING and dropped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import _clamp_style
from ..config_writer import write_overlay_style, write_override
from ..gazetteer.gazetteer import clear_active_series, save_active_series
from ..hotkeys import HOTKEY_KEYS, canonical_combo
from ..state import AppState, VALID_GENRES
from .broadcast import ConnectionManager, make_toggle_overlay_cmd

_STATIC_DIR = Path(__file__).parent / "static"
_log = logging.getLogger(__name__)

# Loopback hosts the overlay page can legitimately be served from. The
# browser overlay is served by THIS server, so its Origin is always one of
# these; the user could type any of them in the address bar.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


# overlay.js interpolates style string values straight into CSS
# custom properties (`--jp-font`, `--text-color`, ...) behind the WS token
# gate. The client trusts the server, so the server must reject values
# carrying CSS-breaking / injection characters before broadcasting -- a
# `jp_font` like `x; } body{display:none` would otherwise close the
# custom-property declaration and inject arbitrary CSS into the page.
# Chars that can break out of a CSS value / smuggle markup:
#   ; { }   close a declaration / open a block
#   < >     open an HTML tag (defense for any future innerHTML sink)
#   plus C0 control chars (newlines, etc.)
_CSS_INJECTION_CHARS = frozenset(";{}<>")


def _sanitize_style(style: dict) -> dict:
    """Return a copy of ``style`` with unsafe string values dropped.

    Non-string values (ints for sizes/opacity/timing) pass through
    untouched. A string value containing a CSS-injection char or a C0
    control char is REJECTED (the key is omitted from the result) rather
    than silently mangled -- so the client keeps whatever value it already
    had for that key instead of rendering attacker-controlled CSS. Logged
    at WARNING so a rejected value is visible, not a silent swallow.
    """
    clean: dict = {}
    for key, value in style.items():
        if isinstance(value, str):
            if any(c in _CSS_INJECTION_CHARS for c in value) or any(
                ord(c) < 0x20 for c in value
            ):
                _log.warning(
                    "set_style rejected unsafe value for %r: %r", key, value,
                )
                continue
        clean[key] = value
    return clean


def _origin_allowed(origin: str | None, bound_port: int) -> bool:
    """Cross-site WebSocket hijacking (CSWSH) guard for the overlay WS.

    The WS accepts command frames that shut the app down (``quit``) and
    write config to disk (``set_style``). A browser will happily open
    ``ws://127.0.0.1:<port>/ws`` from ANY page the user visits, so without
    an Origin check any website could drive those commands.

    Policy:
      * No Origin header -> a native client (the Qt overlay's QWebSocket,
        a CLI tool). Browsers always send Origin, so absence means
        not-a-browser; allow. (A malicious *native* process on the box
        already outranks anything the WS could do.)
      * Origin present -> a browser. Allow ONLY a loopback origin on
        exactly ``bound_port``: the one origin the overlay page this
        server itself serves can have. Any other loopback port is a
        DIFFERENT localhost service's page (dev server, Jupyter, another
        app's web UI) and is rejected like any real website.
    """
    if not origin:
        return True
    try:
        parts = urlsplit(origin)
        host = parts.hostname
        port = parts.port  # None when the origin has no explicit port
    except ValueError:
        return False
    if host not in _LOOPBACK_HOSTS:
        return False
    # Browsers omit the port from Origin when it is the scheme default,
    # so normalize None to http 80 / https 443 before comparing --
    # otherwise binding overlay.websocket_port to 80 or 443 would make
    # this server reject its own page (None never equals bound_port).
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return port == bound_port


async def _handle_command(
    raw: str,
    state: AppState,
    manager: ConnectionManager,
    on_quit: "object | None" = None,
    gazetteer_enabled: bool = True,
    persist_lock: asyncio.Lock | None = None,
    on_pause_change: Callable[[bool], Awaitable[object]] | None = None,
    on_hotkeys_change: Callable[[dict, bool], None] | None = None,
) -> None:
    """Parse a text frame and apply it to state. Silently ignore malformed input.

    `set_style` additionally broadcasts the merged style to every connected
    client and persists it to disk via the atomic config writer (the sparse
    writer always targets ``paths.user_overrides_path``).

    `persist_lock` serializes each persisting action's state merge, awaited
    off-loop write, and broadcast as one critical section. The awaited
    persist yields the event loop, so two near-simultaneous commands from
    different clients would otherwise interleave there and the OLDER full
    snapshot could land last on disk and on every client. build_app passes
    one shared per-app lock; when omitted (single-shot callers) a fresh
    lock keeps the single-call path unchanged.

    `gazetteer_enabled` gates `set_series`: when the series-gazetteer feature
    is off, set_series is rejected (no state mutation, no pointer persist, no
    broadcast) so no WS client can drive a feature the user disabled.
    main.py passes feature_enabled(cfg) here.

    `toggle_overlay` needs no state: it broadcasts the shortcut's
    toggle_overlay frame to every client.

    `on_quit` (Callable[[], None] or None) is invoked when the overlay
    sends ``{"type":"command","action":"quit"}``. main.py passes
    ``lambda: done.set()`` so the tray's Quit action triggers a clean
    pipeline shutdown via the existing amain() finally block (glossary
    save, translator aclose, cuda_pool shutdown, etc.).

    `on_pause_change`, when supplied by the production pipeline, owns pause
    mutation, capture/remainder boundaries, and the state broadcast. When it
    is absent, this standalone handler preserves the legacy direct mutation
    and broadcast behavior.

    `on_hotkeys_change` receives a validated `set_hotkeys` payload (config
    key -> canonical combo); the production pipeline passes
    PipelineApp.apply_hotkeys, which re-binds the global shortcuts.
    Its second argument, ``replay``, is True only when the frame carries
    ``"replay": true`` (the tray's reconnect replay).
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    if payload.get("type") != "command":
        return
    action = payload.get("action")
    if persist_lock is None:
        persist_lock = asyncio.Lock()

    async def _broadcast_state() -> None:
        await manager.broadcast(state.state_frame())

    if action == "pause":
        if on_pause_change is not None:
            await on_pause_change(True)
        else:
            state.pause()
            await _broadcast_state()
    elif action == "resume":
        if on_pause_change is not None:
            await on_pause_change(False)
        else:
            state.resume()
            await _broadcast_state()
    elif action == "set_genre":
        genre = payload.get("genre")
        if isinstance(genre, str) and genre in VALID_GENRES:
            async with persist_lock:
                state.set_genre(genre)
                # Persist, or the choice silently reverts next launch
                # (main.py re-seeds AppState from cfg["genre"] on startup;
                # the wizard was the ONLY writer of this key). The sparse
                # writer prunes the key again if the user returns to the
                # shipped default. Same error-handling shape as set_style.
                # Off-loop: the write is fsync-bearing and this handler runs
                # on the pipeline's hot event loop (same idiom as main.py's
                # glossary save).
                try:
                    await asyncio.to_thread(write_override, ["genre"], genre)
                except Exception as e:
                    _log.warning("config write failed: %s", e)
                await _broadcast_state()
        else:
            _log.warning("set_genre rejected: %r", genre)
    elif action in ("set_style", "apply_style"):
        style = payload.get("style")
        if isinstance(style, dict):
            # Strip CSS-injection / control chars server-side
            # before the value is merged + broadcast -- the overlay page
            # interpolates these straight into CSS custom properties and
            # trusts the server to have vetted them.
            style = _sanitize_style(style)
            # Range-clamp BEFORE merge/broadcast/persist, mirroring the Qt
            # tray (_on_style_received -> config._clamp_style): the browser
            # overlay applies broadcast values raw, and the persist below
            # writes them to user_overrides.yaml, so an out-of-range
            # hold_ms from any loopback client would both render unclamped
            # and survive restarts on disk. _clamp_style returns a FULL
            # defaults-seeded dict, so intersect with the keys the client
            # actually sent: present keys are clamped (garbage values fall
            # back to the shipped default), absent keys keep their current
            # merged value, unknown keys are dropped (the config loader
            # strips them at boot anyway).
            clamped = _clamp_style(style)
            style = {k: v for k, v in clamped.items() if k in style}
            # The lock holds merge, optional persistence, and broadcast together:
            # the awaited write yields the loop, and an unordered second
            # set_style landing in that window would let this OLDER full
            # snapshot finish last on disk and on every client.
            async with persist_lock:
                if action == "apply_style" and isinstance(payload.get("reduce_motion"), bool):
                    # Runtime only: Settings already persisted overlay.reduce_motion.
                    state.set_reduce_motion(payload["reduce_motion"])
                full = state.update_style(style)  # {"type": "style", ...full merged...}
                # Settings has already persisted its full transaction before
                # sending apply_style. Other clients use set_style and retain
                # the server-owned persistence path.
                if action == "set_style":
                    persist = {k: v for k, v in full.items() if k != "type"}
                    # Off-loop like set_genre: an inline fsync here stalls
                    # every in-flight subtitle broadcast for its duration.
                    try:
                        await asyncio.to_thread(write_overlay_style, persist)
                    except Exception as e:
                        _log.warning("config write failed: %s", e)
                await manager.broadcast(full)
        else:
            _log.warning("%s rejected: %r", action, style)
    elif action == "set_series":
        if not gazetteer_enabled:
            _log.warning("set_series rejected: gazetteer feature disabled")
            return
        media_id = payload.get("media_id")
        label = payload.get("label")
        native = payload.get("native")
        native = native if isinstance(native, str) and native else None
        if "media_id" in payload and media_id is None:
            # The tray flyout's Clear: no series. The pointer goes too, so a
            # starting pipeline restores none (a pick persists it the same
            # way); the new series epoch makes the translation stage drop
            # the gazetteer and the series hint.
            async with persist_lock:
                state.set_series(None, None, None)
                try:
                    await asyncio.to_thread(clear_active_series)
                except Exception as e:
                    _log.warning("active-series clear failed: %s", e)
                await _broadcast_state()
        # media_id must be a real int (bool is an int subclass, so exclude it);
        # label a str. The overlay builds + saves the gazetteer CACHE itself
        # before sending this; here we only record the pointer + swap-trigger.
        elif isinstance(media_id, int) and not isinstance(media_id, bool) and isinstance(label, str):
            async with persist_lock:
                state.set_series(media_id, label, native)
                # Off-loop like set_genre/set_style: the pointer write fsyncs.
                try:
                    await asyncio.to_thread(save_active_series, media_id, label, native)
                except Exception as e:
                    _log.warning("active-series persist failed: %s", e)
                await _broadcast_state()
        else:
            # The label is a watch-list series title and this log ships in
            # diagnostics zips, so record its length/type, never its text
            # (the identity-free convention from series_palette).
            _log.warning(
                "set_series rejected: media_id=%r label_type=%s label_len=%s",
                media_id,
                type(label).__name__,
                len(label) if isinstance(label, str) else "n/a",
            )
    elif action == "toggle_overlay":
        # The tray flyout's Subtitles tile: answer every client with the
        # frame the Ctrl+Alt+O shortcut broadcasts, so the Qt overlay and
        # the browser page toggle exactly as for the shortcut.
        await manager.broadcast(make_toggle_overlay_cmd())
    elif action == "set_hotkeys":
        # Settings persisted the shortcuts already; the pipeline re-binds.
        # The whole payload is rejected unless every key is a known shortcut
        # and every combo is bindable. "replay": true marks the tray's
        # reconnect replay, so only changed combos announce a conflict.
        hotkeys = payload.get("hotkeys")
        replay = payload.get("replay") is True
        clean = None
        if isinstance(hotkeys, dict) and hotkeys and all(
            key in HOTKEY_KEYS and isinstance(value, str) and canonical_combo(value)
            for key, value in hotkeys.items()
        ):
            clean = {key: canonical_combo(value) for key, value in hotkeys.items()}
        if clean is None or on_hotkeys_change is None:
            _log.warning("set_hotkeys rejected: %r", hotkeys)
        else:
            try:
                on_hotkeys_change(clean, replay)
            except Exception:
                _log.exception("on_hotkeys_change callback raised")
    elif action == "set_incremental":
        # Settings persisted asr.incremental.enabled already; the ASR stage
        # uses the new value from its next window.
        enabled = payload.get("enabled")
        if isinstance(enabled, bool):
            state.set_incremental(enabled)
            _log.info("faster subtitles: enabled=%s", str(enabled).lower())
        else:
            _log.warning("set_incremental rejected: %r", enabled)
    elif action == "quit":
        _log.info("quit command received from overlay; initiating clean shutdown")
        if on_quit is not None:
            try:
                on_quit()
            except Exception:
                _log.exception("on_quit callback raised")
    else:
        _log.warning("unknown command action: %r", action)


def build_app(
    state: AppState | None = None,
    on_quit=None,
    health_probes: "object | None" = None,
    serve_browser_ui: bool = True,
    *,
    on_pause_change: Callable[[bool], Awaitable[object]] | None = None,
    on_hotkeys_change: Callable[[dict, bool], None] | None = None,
    bound_port: int,
    ws_token: str,
    gazetteer_enabled: bool = True,
) -> FastAPI:
    """Build the overlay FastAPI app.

    `bound_port` (required, keyword-only) is the port uvicorn will bind
    this app to (main.py's overlay.websocket_port). The /ws Origin guard
    accepts a browser origin only on exactly that port -- the one origin
    the page this server serves can have; any other loopback port is a
    co-resident localhost service and is refused.

    `ws_token` (required, keyword-only) is the per-launch session token
    run_pipeline generates. Every /ws connection must present it as
    ``?t=<token>``; GET /ws-token hands it to the browser overlay page.
    The caller generates it so the same value can reach the spawned Qt
    overlay through its environment.

    `health_probes` (if supplied) must expose three bool attributes:
    `asr_alive`, `translator_reachable`, `broadcaster_accepting`.
    The /healthz route reads these on every request (cheap: they're
    cached on the PipelineApp) and returns 200 iff all three are
    truthy, else 503 with {"status":"unhealthy","failing":[...]}
    naming the subsystems that are down. When `health_probes` is
    None (e.g., unit tests for other routes) the route degrades to
    an unconditional 200 -- the watchdog poller only runs against
    the pipeline process, which always passes probes in.

    `serve_browser_ui` gates ONLY the browser overlay page:
    the `/` index route and the `/static` mount. The /ws, /healthz,
    and /livez routes are the display + control bus for the Qt
    overlay, the tray, and the watchdog probes; they are registered
    unconditionally. run_pipeline passes overlay.browser here; it
    used to gate the whole server, killing subtitles, tray Quit,
    and the health probes whenever a user set the key to false.

    `gazetteer_enabled` gates the `set_series` wire command (see
    _handle_command). main.py passes feature_enabled(cfg) so a disabled
    series feature cannot be driven over the wire.

    `on_hotkeys_change` receives each validated `set_hotkeys` payload (see
    _handle_command); the production pipeline passes
    PipelineApp.apply_hotkeys. Its second argument, ``replay``, is True
    only when the frame carries ``"replay": true`` (the tray's reconnect
    replay).
    """
    if not (isinstance(ws_token, str) and ws_token):
        # An empty token would lock every client out (the gate requires a
        # non-empty match), which is a wiring bug, not a runtime state.
        raise ValueError("build_app requires a non-empty ws_token")
    app = FastAPI()
    manager = ConnectionManager()
    app.state.manager = manager
    app.state.app_state = state if state is not None else AppState()
    app.state.on_quit = on_quit
    app.state.on_pause_change = on_pause_change
    app.state.on_hotkeys_change = on_hotkeys_change
    app.state.health_probes = health_probes
    app.state.ws_token = ws_token
    app.state.gazetteer_enabled = gazetteer_enabled
    # One lock per app: _handle_command holds it across each persisting
    # command's merge + off-loop write + broadcast so concurrent writers
    # from different WS clients stay ordered (the awaited persist yields
    # the loop mid-command).
    app.state.persist_lock = asyncio.Lock()

    if serve_browser_ui:
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

        @app.get("/ws-token")
        async def ws_token_route(request: Request):
            # The overlay page fetches this same-origin on load (no
            # Origin header on a same-origin GET). A cross-origin local
            # page CAN issue the GET but cannot read the body: no CORS
            # headers are sent. The Origin check is defense in depth on
            # top of that browser guarantee.
            origin = request.headers.get("origin")
            if not _origin_allowed(origin, bound_port=bound_port):
                _log.warning(
                    "rejected /ws-token request from disallowed origin: %r",
                    origin,
                )
                return JSONResponse(
                    status_code=403, content={"detail": "forbidden origin"}
                )
            return {"token": ws_token}

    @app.get("/healthz")
    def healthz():
        probes = app.state.health_probes
        if probes is None:
            # No probes wired (e.g., unit tests for unrelated routes).
            return {"status": "ok"}
        failing: list[str] = []
        if not probes.asr_alive:
            failing.append("asr")
        if not probes.translator_reachable:
            failing.append("translator")
        if not probes.broadcaster_accepting:
            failing.append("broadcaster")
        if failing:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "failing": failing},
            )
        return {"status": "ok"}

    @app.get("/livez")
    def livez():
        # Liveness: 200 iff the event loop can serve a trivial request.
        # A wedged loop fails to respond -> watchdog HealthProbe times
        # out -> force-restart. DELIBERATELY excludes subsystem
        # readiness (translator/asr/broadcaster) so a llama outage or
        # reload window never trips the wedge detector. Readiness is
        # /healthz; liveness is here.
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        # CSWSH guard: refuse a cross-site browser Origin
        # BEFORE accepting the handshake, so a drive-by page can't send
        # quit / set_style commands. Closing before accept() denies the
        # WebSocket upgrade.
        origin = websocket.headers.get("origin")
        if not _origin_allowed(origin, bound_port=bound_port):
            _log.warning("rejected WebSocket from disallowed origin: %r", origin)
            await websocket.close(code=1008)  # policy violation
            return
        # Every connection, browser or native, must present the
        # per-launch session token (?t=). A browser page can only have
        # read it from /ws-token on this server's own origin, and the Qt
        # overlay gets it from its environment. Native clients are not
        # exempt: 127.0.0.1 is shared by every logged-in Windows session,
        # so a token-free native client could be another user's process.
        token = websocket.query_params.get("t")
        if not (token and secrets.compare_digest(token, ws_token)):
            _log.warning(
                "rejected WebSocket: missing or invalid session token "
                "(origin %r)", origin,
            )
            await websocket.close(code=1008)
            return
        await websocket.accept()
        manager.add(websocket, native=not origin)
        try:
            # Send initial state snapshot so the client UI can render
            # current pause/genre. A peer can legitimately vanish between
            # accept() and these sends -- that race is logged at DEBUG
            # (a bare `except Exception: pass` here would be a fully
            # silent swallow). Anything ELSE
            # (e.g. a snapshot() regression) falls through to the outer
            # WARNING+exc_info handler below so it is never masked.
            try:
                # These direct sends get the same 2 s bound as
                # ConnectionManager._send_one -- a client that connects
                # and immediately stops reading would otherwise park this
                # handler coroutine until TCP timeout, leaking one
                # handler per reconnect attempt.
                await asyncio.wait_for(
                    websocket.send_json(app.state.app_state.state_frame()),
                    timeout=2.0,
                )
                await asyncio.wait_for(
                    websocket.send_json(app.state.app_state.style_frame()),
                    timeout=2.0,
                )
                # The problem set (problems.py), then any queued notices.
                # `now` lets a client that connects soon after a restart hear
                # which problems the restart resolved.
                await asyncio.wait_for(
                    websocket.send_json(
                        app.state.app_state.problems_frame(now=time.monotonic())
                    ),
                    timeout=2.0,
                )
                # Notices raised while no native client was connected (a
                # hotkey conflict at startup fires before this server
                # exists) wait for the first client without an Origin, the
                # Qt overlay, and go to it after the snapshot frames it
                # renders from. The browser page has no notice branch, so it
                # leaves them queued.
                if not origin:
                    for notice in app.state.app_state.take_pending_notices():
                        await asyncio.wait_for(
                            websocket.send_json(notice), timeout=2.0,
                        )
            except asyncio.TimeoutError:
                _log.warning(
                    "ws initial snapshot send exceeded 2s (client not "
                    "reading); closing it",
                )
                # Same bound as the sends: the close flushes through the
                # same stalled transport that just timed out, so an
                # unbounded close would park this handler for the
                # transport's close_timeout on every reconnect of a
                # never-reading client.
                try:
                    await asyncio.wait_for(
                        websocket.close(code=1011), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    _log.debug(
                        "ws close of unresponsive client timed out; "
                        "abandoning the socket"
                    )
                return
            except (WebSocketDisconnect, RuntimeError) as e:
                _log.debug(
                    "ws initial send failed (client gone before the "
                    "state/style snapshot was delivered): %s", e,
                )
            while True:
                raw = await websocket.receive_text()
                await _handle_command(
                    raw,
                    app.state.app_state,
                    manager,
                    on_quit=app.state.on_quit,
                    on_pause_change=app.state.on_pause_change,
                    on_hotkeys_change=app.state.on_hotkeys_change,
                    gazetteer_enabled=app.state.gazetteer_enabled,
                    persist_lock=app.state.persist_lock,
                )
        except WebSocketDisconnect:
            pass
        except Exception:
            _log.warning("websocket handler error", exc_info=True)
        finally:
            manager.remove(websocket)

    return app
