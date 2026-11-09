"""Connection manager for WebSocket overlay clients.

Pure asyncio: no FastAPI imports, so this is unit-testable against any
object exposing `async send_json(payload)`.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

_log = logging.getLogger(__name__)

# Environment variable carrying the per-launch /ws session token from the
# pipeline to the Qt overlay child it spawns. Lives here (no FastAPI or Qt
# imports) so both sides can import it.
WS_TOKEN_ENV = "YAMABIKO_WS_TOKEN"


class _SupportsSendJson(Protocol):
    async def send_json(self, payload: dict) -> None: ...


@dataclass(frozen=True)
class BroadcastReceipt:
    """Server-side send-call outcomes; not a delivery/render receipt."""

    attempted: int
    accepted: int
    failed: int

    def __post_init__(self) -> None:
        counts = (self.attempted, self.accepted, self.failed)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError(
                "broadcast receipt counts must be non-negative integers",
            )
        if self.accepted + self.failed != self.attempted:
            raise ValueError("broadcast receipt counts must balance")


class ConnectionManager:
    """Holds connected clients and fans out broadcasts.

    Dead clients are reaped automatically when a send fails -- callers do
    not need to explicitly remove a disconnected client. `add` is
    idempotent; `remove` does not raise on unknown clients.

    Every per-client send is bounded by ``send_timeout_s``: a
    client whose TCP buffer is full (frozen tab, suspended process)
    parks ``send_json`` forever under uvicorn's write flow-control,
    and an unbounded send there would stall the translation hot path
    for ALL clients while /healthz stayed green. A timed-out client is
    treated as dead and reaped, so the stall is one-shot and bounded,
    never repeated.
    """

    def __init__(self, send_timeout_s: float = 2.0) -> None:
        self._clients: set[_SupportsSendJson] = set()
        # Clients that connected without an Origin: the Qt overlay. Only
        # these can show notices; a browser/OBS page can't.
        self._native: set[_SupportsSendJson] = set()
        self._send_timeout_s = send_timeout_s

    def add(self, ws: _SupportsSendJson, *, native: bool = False) -> None:
        self._clients.add(ws)
        if native:
            self._native.add(ws)

    def remove(self, ws: _SupportsSendJson) -> None:
        self._clients.discard(ws)
        self._native.discard(ws)

    def client_count(self) -> int:
        return len(self._clients)

    def native_client_count(self) -> int:
        return len(self._native)

    async def broadcast(self, payload: dict) -> BroadcastReceipt:
        if not self._clients:
            return BroadcastReceipt(attempted=0, accepted=0, failed=0)
        # Snapshot clients so the set isn't mutated during send.
        clients = list(self._clients)
        results = await asyncio.gather(
            *(self._send_one(c, payload) for c in clients),
        )
        for c, ok in zip(clients, results):
            if not ok:
                self._clients.discard(c)
                self._native.discard(c)
                # Discarding only forgets the client on OUR side; without
                # an explicit close the peer keeps a dead "connected"
                # session forever (a frozen tab that wakes back up never
                # learns it was dropped). Best-effort: the transport may
                # already be gone, and the protocol only promises
                # `send_json`, so any failure here is irrelevant. The
                # close gets the same budget as a send: it flushes through
                # the same write flow-control, so against a frozen peer an
                # unbounded close would park this hot path for the
                # transport's close_timeout (tens of seconds).
                try:
                    await asyncio.wait_for(
                        c.close(code=1011),  # 1011 = internal error
                        timeout=self._send_timeout_s,
                    )
                except asyncio.TimeoutError:
                    _log.debug(
                        "close of reaped overlay client timed out; "
                        "abandoning the socket"
                    )
                except Exception:
                    pass
        accepted = sum(results)
        return BroadcastReceipt(
            attempted=len(clients),
            accepted=accepted,
            failed=len(clients) - accepted,
        )

    async def _send_one(self, client: _SupportsSendJson, payload: dict) -> bool:
        """Send one payload; return True on success, False on failure.

        The send is bounded by ``self._send_timeout_s``; a timeout
        means the peer stopped reading and the client is reported dead
        so the caller discards it.

        CancelledError (BaseException) is not caught -- cancellation
        propagates out through gather so shutdown works cleanly.
        (asyncio.wait_for translates an OUTER cancel into CancelledError
        and only raises TimeoutError for its own expiry, so this seam
        is cancel-safe.)
        """
        try:
            await asyncio.wait_for(
                client.send_json(payload), timeout=self._send_timeout_s
            )
            return True
        except asyncio.TimeoutError:
            _log.warning(
                "overlay client stopped reading (send exceeded %.1fs); "
                "dropping it so subtitle delivery is not stalled",
                self._send_timeout_s,
            )
            return False
        except Exception as e:
            # Not silent: a non-serializable frame (or any send regression)
            # reaps EVERY client at once, and without a log line that
            # looks identical to all the tabs closing on their own.
            _log.warning(
                "overlay client send failed (%s: %s); dropping it",
                type(e).__name__, e,
            )
            return False


def make_toggle_overlay_cmd() -> dict:
    """Return a toggle_overlay command payload for broadcast."""
    return {"type": "toggle_overlay"}


def make_show_series_cmd() -> dict:
    """Return a show-series-palette command payload for broadcast. Sent when
    the Ctrl+Alt+S hotkey fires (pipeline-owned, like toggle_overlay); the Qt
    overlay's ws_client raises a signal that opens the series-search palette."""
    return {"type": "show_series_palette"}


def make_notice_frame(title: str, body: str, settings_page: str | None = None) -> dict:
    """Return a user-facing notice frame; the Qt tray shows it as a toast.
    ``settings_page`` (for example "shortcuts") makes a click on the toast
    open Settings at that page."""
    frame = {"type": "notice", "title": title, "body": body}
    if settings_page:
        frame["settings_page"] = settings_page
    return frame
