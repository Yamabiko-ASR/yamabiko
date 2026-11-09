"""Unit tests for the pipeline /livez liveness route.

/livez is the watchdog wedge-detector's signal. Unlike /healthz
(readiness: all subsystems up), /livez is liveness: it returns 200
whenever the event loop can serve the request, INDEPENDENT of
translator/asr/broadcaster state. A wedged event loop fails to respond
-> the watchdog HealthProbe times out -> force-restart.

Fails-red seeded-defect drill:
  - Make /livez consult translator_reachable -> test 2 goes red.
  - Drop the route entirely -> 404 breaks test 1.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from yamabiko.display.browser_server import build_app
from yamabiko.state import AppState


class _ProbesTranslatorDown:
    asr_alive = True
    translator_reachable = False   # readiness DOWN
    broadcaster_accepting = True


def _client(probes):
    app = build_app(state=AppState(),
                    on_quit=lambda: None, health_probes=probes,
                    bound_port=8765, ws_token="test-session-token")
    return TestClient(app)


def test_livez_returns_200_with_no_probes():
    """Fails red if: /livez route is absent (404)."""
    r = _client(None).get("/livez")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_livez_returns_200_even_when_translator_unreachable():
    """Fails red if: /livez couples to readiness (translator) -- a llama
    outage would then trip the wedge detector, creating the feedback loop
    this endpoint exists to break."""
    r = _client(_ProbesTranslatorDown()).get("/livez")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
