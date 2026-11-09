"""QWebSocket wrapper for the Qt overlay app.

Connects to ws://127.0.0.1:PORT/ws. Emits:
  - subtitleReceived(jp: str, en: str, jp_ruby: list)
  - stateReceived(genre: str, paused: bool)
  - seriesReceived(media_id: int|None, label: str, native: str)  # from the state frame
  - styleReceived(dict)  # full style payload including the "type" key
  - connectedChanged(bool)
  - toggleOverlayReceived()      # server asks us to toggle overlay visibility
  - showSeriesPaletteReceived()  # server asks us to open the series palette
  - noticeReceived(title: str, body: str, settings_page: str)  # a user-facing notice for a toast; settings_page "" for none
  - problemsReceived(active: list, recovered: list)  # the pipeline's problem set (problems.py)

The pipeline passes its per-launch /ws session token in the
YAMABIKO_WS_TOKEN environment variable; when set, it rides the connect URL
as ``?t=<token>`` (the server refuses every token-free connection).

Auto-reconnects with exponential backoff (1s, 2s, 4s, 8s, 16s, 30s cap).
"""
from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket

from ..display.broadcast import WS_TOKEN_ENV
from ..problems import KIND_ORDER, is_problem

_log = logging.getLogger(__name__)


def _with_session_token(url: str, token: str | None) -> str:
    """Append the session token as the ``t`` query parameter, if any."""
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={quote(token, safe='')}"


class _ReconnectBackoff:
    """Exponential-backoff helper. Doubles the delay each call, capped."""

    def __init__(self, *, initial_ms: int = 1000, cap_ms: int = 30000) -> None:
        self._initial = initial_ms
        self._cap = cap_ms
        self._current = initial_ms

    def next_delay_ms(self) -> int:
        d = min(self._current, self._cap)
        self._current = min(self._current * 2, self._cap)
        return d

    def reset(self) -> None:
        self._current = self._initial


class PipelineClient(QObject):
    subtitleReceived = Signal(str, str, list)  # jp, en, jp_ruby (list of dicts)
    stateReceived = Signal(str, bool)       # genre, paused
    seriesReceived = Signal(object, str, str)  # (media_id: int|None, label: str, native: str)
    styleReceived = Signal(dict)            # full style message from server
    connectedChanged = Signal(bool)
    toggleOverlayReceived = Signal()        # server requests visibility toggle
    showSeriesPaletteReceived = Signal()    # server asks us to open the palette
    noticeReceived = Signal(str, str, str)  # title, body, Settings page ("" for none)
    problemsReceived = Signal(list, list)   # active entries, recovered entries

    def __init__(self, url: str = "ws://127.0.0.1:8765/ws", parent: QObject | None = None):
        super().__init__(parent)
        # The token-free URL is the one logged: logs ship in the
        # diagnostics zip, and the token is a per-launch secret.
        self._log_url = url
        self._url = _with_session_token(url, os.environ.get(WS_TOKEN_ENV))
        self._ws = QWebSocket()
        self._ws.connected.connect(self._on_connected)
        self._ws.disconnected.connect(self._on_disconnected)
        self._ws.textMessageReceived.connect(self._on_text)
        self._ws.errorOccurred.connect(self._on_error)
        self._backoff = _ReconnectBackoff(initial_ms=1000, cap_ms=30000)
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._open)
        self._stopped = False

    def start(self) -> None:
        # Clear the stop latch: without the reset, a stop() -> start()
        # reuse yields a client that connects once and never
        # auto-reconnects (_schedule_reconnect no-ops while _stopped).
        self._stopped = False
        self._open()

    def stop(self) -> None:
        # Flag first: close() on a connected socket fires disconnected,
        # whose handler must not re-arm the timer we just stopped.
        self._stopped = True
        self._reconnect_timer.stop()
        self._ws.close()

    def send_command(self, payload: dict) -> bool:
        """Send ``payload`` if connected.

        Returns True when the frame was handed to the socket, False
        when it was dropped because the WS is disconnected. Callers
        (Tray Settings Apply) use the False to fall back to a local
        apply + persist instead of silently losing the command.
        """
        if self._ws.state() != self._ws.state().ConnectedState:
            # Action name only: payloads carry user data (a set_series pick
            # holds the anime title and its AniList id), and this log ships
            # in the diagnostics zip.
            _log.warning(
                "send_command called while disconnected: action=%s",
                payload.get("action"),
            )
            return False
        self._ws.sendTextMessage(json.dumps(payload))
        return True

    def _open(self) -> None:
        _log.info("connecting to %s", self._log_url)
        self._ws.open(QUrl(self._url))

    def _on_connected(self) -> None:
        _log.info("websocket connected")
        self._backoff.reset()
        self.connectedChanged.emit(True)

    def _on_disconnected(self) -> None:
        self.connectedChanged.emit(False)
        self._schedule_reconnect("websocket disconnected")

    def _on_error(self, err) -> None:
        # Fallback arm of the reconnect loop, for a connect that is
        # REFUSED (pipeline restart, or uvicorn not yet bound at boot).
        # Measured on PySide6 6.11/Windows, a refusal emits disconnected
        # AND errorOccurred, so _on_disconnected has usually armed the
        # timer by the time we get here and _schedule_reconnect's isActive
        # guard absorbs this call (the backoff must not double-advance on
        # the pair). The re-arm stays because that pairing is a Qt platform
        # detail, not a documented guarantee: on a build that reports a
        # refusal as error-only, this is the ONLY way back, and without it
        # one refusal ends the reconnect loop for the session -- live tray,
        # zero subtitles, no recovery. The error-arm-alone test in
        # tests/overlay_app/test_real_ws_reconnect.py drives a real refused
        # boot connect through this handler with the disconnected arm
        # detached, so it cannot rot into dead code unnoticed.
        _log.warning("websocket error: %s", err)
        if self._ws.state() != self._ws.state().ConnectedState:
            self._schedule_reconnect("websocket connect failed")

    def _schedule_reconnect(self, why: str) -> None:
        """Arm one backoff retry; no-op after stop() or if already armed.

        The isActive guard keeps the error+disconnected pair a genuine
        drop emits from double-advancing the backoff.
        """
        if self._stopped or self._reconnect_timer.isActive():
            return
        delay = self._backoff.next_delay_ms()
        _log.info("%s; retrying in %.1fs", why, delay / 1000)
        self._reconnect_timer.start(delay)

    def _on_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        t = msg.get("type")
        if t == "subtitle":
            jp = msg.get("jp", "")
            en = msg.get("en", "")
            jp_ruby = msg.get("jp_ruby", []) or []
            # jp_ruby is the one payload field whose element SHAPE reaches
            # layout code (ruby_layout does seg.get(...)); a non-dict entry
            # from a squatting/regressed producer would escape the ruby
            # fallback try and crash the Qt slot dispatcher. Drop bad
            # elements here, at the trust boundary, like _clamp_style does
            # for style frames.
            if not isinstance(jp_ruby, list):
                jp_ruby = []
            elif any(not isinstance(seg, dict) for seg in jp_ruby):
                _log.debug("dropping malformed jp_ruby segments in frame")
                jp_ruby = [seg for seg in jp_ruby if isinstance(seg, dict)]
            self.subtitleReceived.emit(jp, en, jp_ruby)
        elif t == "state":
            self.stateReceived.emit(msg.get("genre", "variety"), bool(msg.get("paused", False)))
            # Only a frame that CARRIES the series keys may touch the tray's
            # Series line and the overlay banner: an omitted key is
            # "unspecified" (a legacy or third-party frame), not "cleared". A
            # present key with value None IS an explicit clear
            # (set_series(None, None)) and still emits. This is the
            # consumer-side defense that stops a keyless frame from
            # resetting the Series line to (none). Label and native coerced
            # to "" for the Signal(object, str, str) str slots.
            if "series_media_id" in msg:
                native = msg.get("series_native")
                self.seriesReceived.emit(
                    msg.get("series_media_id"),
                    msg.get("series_label") or "",
                    native if isinstance(native, str) else "",
                )
        elif t == "style":
            self.styleReceived.emit(msg)
        elif t == "toggle_overlay":
            self.toggleOverlayReceived.emit()
        elif t == "show_series_palette":
            self.showSeriesPaletteReceived.emit()
        elif t == "notice":
            title = msg.get("title")
            body = msg.get("body")
            page = msg.get("settings_page")
            if isinstance(title, str) and isinstance(body, str):
                self.noticeReceived.emit(title, body, page if isinstance(page, str) else "")
        elif t == "problems":
            active = msg.get("active")
            recovered = msg.get("recovered")
            self.problemsReceived.emit(
                [dict(entry) for entry in active if is_problem(entry)]
                if isinstance(active, list) else [],
                [{"kind": entry["kind"], "recovery": entry["recovery"]}
                 for entry in recovered
                 if isinstance(entry, dict) and entry.get("kind") in KIND_ORDER
                 and isinstance(entry.get("recovery"), str)]
                if isinstance(recovered, list) else [],
            )
