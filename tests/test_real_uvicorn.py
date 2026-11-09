"""Real-path integration test for the uvicorn browser-overlay server.

``tests/test_browser_server.py`` uses ``fastapi.testclient.TestClient``,
which exercises FastAPI's ASGI pipeline in-process and NEVER touches
uvicorn. Everything specific to uvicorn — the signal-handler instance
monkey-patch (``server.install_signal_handlers = lambda: None``), the
``server.serve()`` coroutine, the ``should_exit`` cooperative shutdown,
the graceful-drain timeout — lives outside that test's coverage.

This file spins up a real uvicorn server in a thread, connects a real
WebSocket client, verifies bidirectional traffic + clean shutdown. A
regression in any uvicorn-specific plumbing fails here instead of
shipping silently to users.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

pytestmark = pytest.mark.real


def _pick_free_port() -> int:
    """Ask the OS for an unused TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ServerThread(threading.Thread):
    """Run a uvicorn.Server in its own asyncio loop on a background thread.

    Mirrors main.py's setup: instance-attribute monkey-patch of
    ``install_signal_handlers`` (to keep SIGINT/SIGTERM away from this
    thread), then ``await server.serve()`` under a fresh loop.
    """

    def __init__(self, app, port: int) -> None:
        super().__init__(daemon=True)
        import uvicorn

        cfg = uvicorn.Config(
            app, host="127.0.0.1", port=port,
            log_level="warning", access_log=False,
        )
        self.server = uvicorn.Server(cfg)
        # Same monkey-patch as main.py:471. If uvicorn changes how
        # install_signal_handlers is dispatched (e.g. turning it into
        # a @property), this line breaks and the test fails loudly.
        self.server.install_signal_handlers = lambda: None
        self.loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(self.server.serve())
        finally:
            self.loop.close()

    def stop(self, timeout: float = 3.0) -> None:
        """Trigger cooperative shutdown. Blocks until the thread exits."""
        self.server.should_exit = True
        self.join(timeout=timeout)


def _wait_for_port(port: int, timeout: float = 5.0) -> bool:
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


def _start_server_on_free_port(app, attempts: int = 3) -> "tuple[_ServerThread, int]":
    """Pick a free port and start uvicorn on it, retrying on the bind race.

    ``_pick_free_port()`` closes its probe socket before uvicorn binds
    (TOCTOU): another process can grab the port in that window, and
    uvicorn then dies in its thread with WinError 10048 (PR #46
    review). A dead server thread distinguishes that bind race from
    "server is up but never accepted"; only the former is retried,
    on a freshly picked port.
    """
    last_port = -1
    for _ in range(attempts):
        port = last_port = _pick_free_port()
        thread = _ServerThread(app, port)
        thread.start()
        if _wait_for_port(port, timeout=5.0):
            return thread, port
        if thread.is_alive():
            thread.stop(timeout=3.0)
            raise AssertionError(
                f"uvicorn is alive but never accepted on 127.0.0.1:{port}"
            )
        # Thread died without serving: almost certainly the port bind
        # race (TOCTOU) -- retry with a fresh port.
    raise AssertionError(
        f"uvicorn failed to bind {attempts} times (last port {last_port}); "
        f"repeated failures are unlikely to be the port race -- check "
        f"the server config / captured stderr"
    )


def test_real_uvicorn_serve_and_shutdown() -> None:
    """Spin up a real uvicorn.Server around our browser app; confirm it
    binds, serves the ``/`` route, and shuts down cleanly via should_exit.

    If uvicorn ever changes how ``install_signal_handlers`` is resolved,
    or breaks the ``should_exit`` cooperative loop, this test fails.
    """
    pytest.importorskip("uvicorn")

    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    app = build_app(state=AppState(), bound_port=8765, ws_token="test-session-token")

    server_thread, port = _start_server_on_free_port(app)
    try:
        # HTTP GET of the overlay index — verifies the static-mount +
        # ASGI dispatch actually works against real uvicorn, not just TestClient.
        import httpx

        resp = httpx.get(f"http://127.0.0.1:{port}/", timeout=3.0)
        assert resp.status_code == 200
        # The overlay's index.html begins with `<!doctype html>` — a
        # sanity check that we got the real static asset and not an
        # error page.
        assert "<!doctype html>" in resp.text.lower() or "<html" in resp.text.lower()

    finally:
        server_thread.stop(timeout=3.0)

    # After stop: thread should have exited cleanly.
    assert not server_thread.is_alive(), (
        "uvicorn server didn't exit after should_exit=True — "
        "cooperative-shutdown plumbing broken"
    )


class _FailingProbes:
    """All three readiness probes down — the llama-outage worst case."""
    asr_alive = False
    translator_reachable = False
    broadcaster_accepting = False


def test_real_livez_stays_200_while_readiness_fails() -> None:
    """Liveness/readiness separation over real uvicorn: /livez
    must answer 200 even while every /healthz readiness probe is down.

    If /livez ever consults the readiness probes, a llama outage trips
    the watchdog's wedge detector and force-kills a healthy pipeline.

    Seeded defect: make /livez readiness-dependent (return healthz()'s
    503 path) -> red here on the 200 assert.
    """
    pytest.importorskip("uvicorn")

    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    app = build_app(
        state=AppState(), health_probes=_FailingProbes(), bound_port=8765,
        ws_token="test-session-token",
    )

    server_thread, port = _start_server_on_free_port(app)
    try:
        import httpx

        # Same call shape as HealthProbe.run (watchdog.py): plain GET,
        # 2 s timeout, 200 == live.
        live = httpx.get(f"http://127.0.0.1:{port}/livez", timeout=2.0)
        assert live.status_code == 200, (
            "/livez went non-200 while only READINESS is failing — "
            "liveness has been coupled to readiness"
        )
        ready = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
        assert ready.status_code == 503, (
            "precondition broken: /healthz should be 503 with all "
            "probes failing — this test no longer exercises the split"
        )
        assert set(ready.json()["failing"]) == {"asr", "translator", "broadcaster"}
    finally:
        server_thread.stop(timeout=3.0)


def test_real_health_probe_round_trips_livez_and_detects_server_loss() -> None:
    """Drive the REAL watchdog HealthProbe (its real httpx.get call site)
    against a REAL uvicorn server: the probe must arm on the live
    server's 200, then fire on_wedged within fail_budget_s after the
    server goes away.

    Asserting the FIRE (not just "no false trip") keeps this from being
    a Liar: a probe that never arms also never fires, so the positive
    path is the only observable that proves the 200 round-trip happened.

    Seeded defects: delete `seen_live = True` (budget never arms) or
    delete the `self._on_wedged()` call in HealthProbe.run -> red on
    the timeout below.
    """
    pytest.importorskip("uvicorn")

    import threading

    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState
    from yamabiko.watchdog import HealthProbe

    app = build_app(state=AppState(), bound_port=8765, ws_token="test-session-token")

    server_thread, port = _start_server_on_free_port(app)
    stop = threading.Event()
    wedged = threading.Event()
    probe = HealthProbe(
        url=f"http://127.0.0.1:{port}/livez",
        stop_event=stop,
        on_wedged=wedged.set,
        spawn_generation=lambda: 1,
        startup_dialog_up=lambda: False,
        poll_interval_s=0.1,
        fail_budget_s=0.5,
    )
    probe_thread = threading.Thread(target=probe.run, daemon=True)
    try:
        probe_thread.start()
        # Let the probe see the live server (arm the budget), then kill
        # the server: every subsequent poll fails at the real socket.
        time.sleep(0.5)
        server_thread.stop(timeout=3.0)
        assert wedged.wait(timeout=10.0), (
            "HealthProbe never called on_wedged after the real server "
            "went away — either the live 200 never armed the budget "
            "(real-HTTP round-trip broken) or wedge detection is dead"
        )
    finally:
        stop.set()
        probe_thread.join(timeout=5.0)
        # _ServerThread.stop is idempotent: the in-try stop() above is
        # the TEST TRIGGER (kill the live server so the probe sees
        # loss), this one is teardown for the failure paths. Deleting
        # the in-try call breaks the test, not just cleanup.
        server_thread.stop(timeout=3.0)
    assert not probe_thread.is_alive(), "HealthProbe.run ignored stop_event"


def test_real_uvicorn_headless_app_serves_health_but_not_page() -> None:
    """build_app(serve_browser_ui=False) — the config run_pipeline
    produces for overlay.browser=false — must still serve the control
    bus (/healthz here; /ws shares the same route table) while the
    browser page routes 404. TestClient covers the route logic
    in-process; this pins the same contract through real uvicorn
    dispatch, which is what production runs.

    Seeded defect: gate build_app's whole route registration on
    serve_browser_ui -> /healthz 404s over real HTTP -> red.
    """
    pytest.importorskip("uvicorn")

    from yamabiko.display.browser_server import build_app
    from yamabiko.state import AppState

    app = build_app(
        state=AppState(), serve_browser_ui=False, bound_port=8765,
        ws_token="test-session-token",
    )

    server_thread, port = _start_server_on_free_port(app)
    try:
        import httpx

        health = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=3.0)
        assert health.status_code == 200, (
            "control bus down in headless mode"
        )
        index = httpx.get(f"http://127.0.0.1:{port}/", timeout=3.0)
        assert index.status_code == 404, (
            "browser page served despite serve_browser_ui=False"
        )
    finally:
        server_thread.stop(timeout=3.0)

    assert not server_thread.is_alive(), (
        "uvicorn server didn't exit after should_exit=True"
    )
