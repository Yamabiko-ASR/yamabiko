"""Real-path proof for Translator.wait_until_ready (hard rule #3): a genuine
httpx GET against a local HTTP server that returns 503 ("loading") then 200,
exactly llama.cpp's cold-load shape. The subject is the real httpx round-trip;
the server is a real socket, not a mock.
"""
from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from yamabiko.translation.translator import Translator, TranslationConfig

pytestmark = pytest.mark.real


def _serve_503_then_200(ready_after: int):
    state = {"hits": 0}

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"] += 1
            code = 200 if state["hits"] > ready_after else 503
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):  # silence the default stderr logging
            pass

    srv = HTTPServer(("127.0.0.1", 0), _H)
    return srv, state


@pytest.mark.real
def test_wait_until_ready_polls_real_health_until_200():
    """Seeded defect: treat 503 as ready/fatal -> the assertion on hits>=3
    (polled through both 503s) goes red."""
    srv, state = _serve_503_then_200(ready_after=2)  # 503, 503, 200
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = TranslationConfig(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model="x", max_tokens=1, temperature=0.0,
        )
        tr = Translator(cfg)
        try:
            asyncio.run(tr.wait_until_ready(timeout_s=10.0, poll_s=0.05))
        finally:
            asyncio.run(tr.aclose())
        assert state["hits"] >= 3, f"expected to poll through both 503s, got {state['hits']}"
    finally:
        srv.shutdown()
        thread.join(timeout=5)
