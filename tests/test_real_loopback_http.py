"""Real-path test: loopback HTTP calls never go through a proxy.

httpx defaults to trust_env=True. On Windows that reads the system proxy
from the registry and ignores its ``<local>`` bypass, so a request to the
local llama-server or the pipeline's /livez is sent to the proxy instead:
transcripts leave the machine and the local call fails. The env proxy set
here stands in for that system proxy (env settings take precedence over
the registry on Windows), so the test exercises the same httpx routing.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.real

# Nothing listens on the discard port, so a proxied request cannot connect.
_DEAD_PROXY = "http://127.0.0.1:9"


def test_loopback_get_reaches_local_server_despite_unreachable_env_proxy(
    monkeypatch,
):
    from yamabiko import loopback_http

    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(var, _DEAD_PROXY)

    received: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        response = loopback_http.get(
            f"http://127.0.0.1:{port}/livez", timeout=5.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert response.status_code == 200
    assert received == ["/livez"], (
        f"the local server did not receive the request directly; got "
        f"{received!r} (a proxied request never reaches it)"
    )
