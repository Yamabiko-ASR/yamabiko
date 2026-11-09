"""Real-path: the translator-problem clear probe over a real loopback HTTP
server, through the shipped loopback httpx client (AGENTS.md rule 3: the
pipeline's health refresher gained this call site in phase 4)."""
from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from yamabiko import loopback_http
from yamabiko.translation.translator import health_says_ready

pytestmark = pytest.mark.real


def test_real_health_probe_reads_loading_ready_absent_and_down():
    """llama-server serves /health 503 while it loads and 200 once ready; a
    server without /health (404) counts as ready; nothing listening is not.

    Seeded defects: treat 503 as ready, or let a connect error raise -> red."""
    statuses = {"/health": 503}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            self.send_response(statuses.get(self.path, 404))
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    async def _probe() -> bool:
        async with loopback_http.async_client(base, timeout=2.0) as client:
            return await health_says_ready(client, f"{base}/health")

    try:
        assert asyncio.run(_probe()) is False
        statuses["/health"] = 200
        assert asyncio.run(_probe()) is True
        del statuses["/health"]
        assert asyncio.run(_probe()) is True
    finally:
        server.shutdown()
        server.server_close()
    assert asyncio.run(_probe()) is False
