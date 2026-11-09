"""Unit tests for ConnectionManager using a fake WebSocket."""
from __future__ import annotations

import asyncio
import logging

import pytest

from yamabiko.display.broadcast import ConnectionManager


class FakeWS:
    """Minimal WebSocket surface: `send_json` plus a fail flag."""

    def __init__(self, fail_on_send: bool = False):
        self.sent: list[dict] = []
        self.fail_on_send = fail_on_send

    async def send_json(self, payload: dict) -> None:
        if self.fail_on_send:
            raise RuntimeError("send failed")
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_broadcast_with_no_clients_is_noop():
    m = ConnectionManager()
    # Must not raise.
    await m.broadcast({"x": 1})
    assert m.client_count() == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_clients():
    m = ConnectionManager()
    a, b = FakeWS(), FakeWS()
    m.add(a)
    m.add(b)
    await m.broadcast({"type": "subtitle", "jp": "x", "en": "y"})
    assert a.sent == [{"type": "subtitle", "jp": "x", "en": "y"}]
    assert b.sent == [{"type": "subtitle", "jp": "x", "en": "y"}]


@pytest.mark.asyncio
async def test_dead_clients_are_dropped_on_broadcast_failure():
    m = ConnectionManager()
    good = FakeWS()
    bad = FakeWS(fail_on_send=True)
    m.add(good)
    m.add(bad)
    await m.broadcast({"n": 1})
    assert m.client_count() == 1
    # Good still receives.
    await m.broadcast({"n": 2})
    assert good.sent == [{"n": 1}, {"n": 2}]


@pytest.mark.asyncio
async def test_remove_is_idempotent():
    m = ConnectionManager()
    ws = FakeWS()
    m.add(ws)
    m.remove(ws)
    m.remove(ws)  # second remove must not raise
    assert m.client_count() == 0


@pytest.mark.asyncio
async def test_add_same_client_twice_stored_once():
    m = ConnectionManager()
    ws = FakeWS()
    m.add(ws)
    m.add(ws)
    assert m.client_count() == 1


async def test_native_client_count_tracks_only_native_clients():
    """Only clients added with native=True count as native (the Qt overlay
    connects without an Origin; a browser page sends one). A native client
    reaped by a failed broadcast stops counting, and removing a native
    client twice is safe.

    Seeded defect: return len(self._clients) from native_client_count, or
    leave a reaped or removed client in the native set -> a count assert
    goes red."""
    m = ConnectionManager()
    a = FakeWS(fail_on_send=True)
    b = FakeWS()
    m.add(a, native=True)
    m.add(b)
    assert m.native_client_count() == 1
    assert m.client_count() == 2

    await m.broadcast({"n": 1})
    assert m.native_client_count() == 0
    assert m.client_count() == 1

    c = FakeWS()
    m.add(c, native=True)
    assert m.native_client_count() == 1
    m.remove(c)
    m.remove(c)  # second remove must not raise
    assert m.native_client_count() == 0
    assert m.client_count() == 1


# --- A hung client (frozen tab, suspended process) must not stall
# the translation hot path. The existing dead-client test only covers an
# INSTANTLY-raising send; a send that never returns is a distinct failure
# mode (uvicorn write flow-control blocks send_json forever while
# /healthz stays green).


class HangingWS:
    """send_json never returns -- simulates a peer whose TCP buffer is
    full (uvicorn flow-control parks the send coroutine forever)."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        await asyncio.sleep(9999)
        self.sent.append(payload)  # pragma: no cover - never reached


@pytest.mark.asyncio
async def test_hanging_client_does_not_block_broadcast_and_is_reaped():
    """broadcast() must complete within the per-send timeout even when one
    client's send never returns, deliver to the healthy client, and reap
    the hung client so later broadcasts never touch it again.

    Seeded defect: remove the asyncio.wait_for bound around
    client.send_json in _send_one -> broadcast() awaits the hung send
    forever -> the outer wait_for below raises TimeoutError -> red.
    """
    m = ConnectionManager(send_timeout_s=0.2)
    hang = HangingWS()
    fast = FakeWS()
    m.add(hang)
    m.add(fast)

    # Hot path: bounded. 3s ceiling >> 0.2s timeout, so a pass means the
    # per-send bound fired, not that we got lucky.
    await asyncio.wait_for(m.broadcast({"n": 1}), timeout=3.0)

    assert fast.sent == [{"n": 1}], "healthy client must still receive"
    assert m.client_count() == 1, "hung client must be reaped after timeout"

    # Subsequent broadcasts are instant -- the hung client is gone.
    await asyncio.wait_for(m.broadcast({"n": 2}), timeout=1.0)
    assert fast.sent == [{"n": 1}, {"n": 2}]


# --- Reaping a dead client must also CLOSE its socket. Discarding it from
# the fan-out only forgets it on our side: the peer (a frozen browser tab
# that woke back up, a stalled native client) keeps a live TCP session it
# believes is "connected" and never receives another frame, forever. And a
# non-timeout send failure must be logged -- a non-serializable frame used
# to reap EVERY client with zero log evidence.


class ClosableWS:
    """FakeWS plus a recording `close`, mirroring the starlette WebSocket
    surface the production reap path calls."""

    def __init__(self, exc: Exception | None = None, hang: bool = False):
        self.sent: list[dict] = []
        self.close_codes: list[int] = []
        self._exc = exc
        self._hang = hang

    async def send_json(self, payload: dict) -> None:
        if self._hang:
            await asyncio.sleep(9999)
        if self._exc is not None:
            raise self._exc
        self.sent.append(payload)

    async def close(self, code: int = 1000) -> None:
        self.close_codes.append(code)


async def test_failed_send_client_is_closed_with_1011_and_failure_logged(caplog):
    """A client whose send raises must be reaped AND have its socket closed
    (code 1011, internal error) so the peer sees the disconnect instead of
    holding a dead "connected" session forever -- and the failure must land
    in the log naming the exception class, not vanish into a bare
    `except Exception: return False`.

    Seeded defects (each one line, each goes red):
    - comment out the `await c.close(code=1011)` in broadcast()'s reap loop
      -> close_codes stays empty -> red.
    - drop the warning log in _send_one's `except Exception` branch
      -> no WARNING record naming ValueError -> red.
    """
    caplog.set_level(logging.WARNING, logger="yamabiko.display.broadcast")
    m = ConnectionManager()
    bad = ClosableWS(exc=ValueError("circular reference detected"))
    good = ClosableWS()
    m.add(bad)
    m.add(good)

    await m.broadcast({"n": 1})

    assert good.sent == [{"n": 1}], "healthy client must still receive"
    assert m.client_count() == 1, "failing client must be reaped"
    assert bad.close_codes == [1011], (
        f"reaped client's socket must be closed with 1011; got "
        f"{bad.close_codes} -- the peer is left holding a dead session"
    )
    assert good.close_codes == [], "healthy client must NOT be closed"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "ValueError" in r.getMessage()
    ]
    assert warnings, (
        "a non-timeout send failure must be logged at WARNING naming the "
        f"exception class; got: {[r.getMessage() for r in caplog.records]}"
    )


async def test_timed_out_client_is_closed_with_1011():
    """The timeout reap path (frozen tab) must close the socket too --
    same peer-side dead-session leak as the raising path.

    Seeded defect: comment out the `await c.close(code=1011)` in
    broadcast()'s reap loop -> close_codes stays empty -> red.
    """
    m = ConnectionManager(send_timeout_s=0.1)
    hung = ClosableWS(hang=True)
    m.add(hung)

    await asyncio.wait_for(m.broadcast({"n": 1}), timeout=3.0)

    assert m.client_count() == 0, "hung client must be reaped"
    assert hung.close_codes == [1011], (
        f"timed-out client's socket must be closed with 1011; got "
        f"{hung.close_codes}"
    )


async def test_reap_survives_client_without_close():
    """Closing the reaped socket is best-effort: a client object with no
    `close` at all (the ConnectionManager protocol only requires
    `send_json`) must still be reaped without the AttributeError escaping
    broadcast().

    Seeded defect: remove the try/except around the reap-loop close ->
    AttributeError propagates out of broadcast() -> red (error).
    """
    m = ConnectionManager()
    bad = FakeWS(fail_on_send=True)  # no close attribute
    good = FakeWS()
    m.add(bad)
    m.add(good)

    await m.broadcast({"n": 1})  # must not raise

    assert m.client_count() == 1
    assert good.sent == [{"n": 1}]


# --- The reap-path close itself runs through the same write flow-control
# as the send: against a frozen peer with a full TCP send buffer, the WS
# close handshake parks for the transport's close_timeout (default 10s,
# up to ~4x for servers under uvicorn's websockets impl), and it runs
# inline per dead client on the subtitle hot path.


class HangingCloseWS:
    """send_json fails instantly (client is dead) but `close` never
    returns: models the frozen-peer close handshake described above."""

    def __init__(self):
        self.close_started = asyncio.Event()
        self._never = asyncio.Event()

    async def send_json(self, payload: dict) -> None:
        raise RuntimeError("send failed")

    async def close(self, code: int = 1000) -> None:
        self.close_started.set()
        await self._never.wait()


async def test_reap_close_of_frozen_client_does_not_stall_broadcast():
    """broadcast() must complete within the send budget even when the
    reaped client's close never returns: one frozen tab must not park
    every subtitle broadcast for the transport's multi-second close
    handshake. The healthy client is still served and the frozen client
    is reaped.

    Seeded defect: drop the asyncio.wait_for bound around c.close in
    broadcast()'s reap loop -> broadcast() awaits the hung close forever
    -> the harness wait_for below raises TimeoutError -> red.
    """
    m = ConnectionManager(send_timeout_s=0.2)
    frozen = HangingCloseWS()
    good = FakeWS()
    m.add(frozen)
    m.add(good)

    # 3s ceiling >> 0.2s budget: a pass means the close bound fired, not
    # that we got lucky.
    await asyncio.wait_for(m.broadcast({"n": 1}), timeout=3.0)

    assert frozen.close_started.is_set(), "reap must still attempt the close"
    assert good.sent == [{"n": 1}], "healthy client must still receive"
    assert m.client_count() == 1, "frozen client must be reaped"

    # Subsequent broadcasts never touch the reaped client again.
    await asyncio.wait_for(m.broadcast({"n": 2}), timeout=1.0)
    assert good.sent == [{"n": 1}, {"n": 2}]
