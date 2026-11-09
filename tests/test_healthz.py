"""Unit tests for pipeline /healthz.

These tests cover the route body in isolation: a `_FakeHealthProbes`
substitutes for the real inspector so each subsystem-down branch is
exercised without needing to spin up an ASR task, a translator
client, or a ConnectionManager. The real probe (`HealthProbes` in
yamabiko.main) is wired into run_pipeline and covered by the
cross-process E2E tests.

Fails-red seeded-defect drill:
  - Drop the /healthz route entirely   -> 404 breaks test 1.
  - Return 200 unconditionally         -> tests 2-4 red.
  - Skip one subsystem check           -> the corresponding 503 test red.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from yamabiko.display.browser_server import build_app
from yamabiko.state import AppState


class _FakeHealthProbes:
    """Substitute for the real health inspectors. Tests flip fields.

    The route reads each attribute as a bool; defaults represent the
    "all subsystems up" baseline so individual tests only have to
    override the one they're driving down.
    """
    asr_alive: bool = True
    translator_reachable: bool = True
    broadcaster_accepting: bool = True


def _make_app(probes):
    state = AppState()
    app = build_app(
        state=state,
        on_quit=lambda: None,
        health_probes=probes,
        bound_port=8765,
        ws_token="test-session-token",
    )
    return app


def test_healthz_returns_200_when_all_subsystems_up():
    """Fails red if: /healthz returns 200 regardless of probe state,
    OR the route isn't registered (404)."""
    app = _make_app(_FakeHealthProbes())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_returns_503_when_asr_stage_dead():
    """Fails red if: /healthz ignores asr_alive and reports ok."""
    probes = _FakeHealthProbes()
    probes.asr_alive = False
    app = _make_app(probes)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unhealthy"
    assert "asr" in body["failing"]


def test_healthz_returns_503_when_translator_unreachable():
    """Fails red if: /healthz skips the translator check."""
    probes = _FakeHealthProbes()
    probes.translator_reachable = False
    app = _make_app(probes)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 503
    assert "translator" in r.json()["failing"]


def test_healthz_returns_503_when_broadcaster_dead():
    """Fails red if: /healthz skips the ws broadcaster check."""
    probes = _FakeHealthProbes()
    probes.broadcaster_accepting = False
    app = _make_app(probes)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 503
    assert "broadcaster" in r.json()["failing"]


# ---------------------------------------------------------------------------
# The REAL HealthProbes.asr_alive property (yamabiko.main).
#
# Every route test above drives _FakeHealthProbes (plain bools), so the
# real property's task-state logic had zero coverage -- the §3.9
# wired-not-exercised gap: a /healthz poll landing in the shutdown drain
# window (pipeline tasks cancelled at run_pipeline's finally, uvicorn
# still draining) hit task.exception() on a CANCELLED task, which RAISES
# CancelledError -- a BaseException that sails through Starlette's
# `except Exception`.
# ---------------------------------------------------------------------------


import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from yamabiko.main import HealthProbes  # noqa: E402


def test_asr_alive_true_while_asr_task_running():
    """Fails red if: asr_alive misreads a healthy running task (e.g. the
    one-line defect of inverting `not task.done()` to `task.done()`)."""
    async def _drive() -> bool:
        async def _forever():
            await asyncio.Event().wait()

        t = asyncio.create_task(_forever())
        await asyncio.sleep(0)  # let the task start
        probes = HealthProbes(app=SimpleNamespace(_asr_task=t))
        alive = probes.asr_alive
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return alive

    assert asyncio.run(_drive()) is True


def test_asr_alive_false_when_asr_task_crashed():
    """Fails red if: asr_alive ignores a stored exception (one-line
    defect: drop the `task.exception() is None` clause)."""
    async def _drive() -> bool:
        async def _boom():
            raise RuntimeError("asr stage died")

        t = asyncio.create_task(_boom())
        try:
            await t
        except RuntimeError:
            pass
        probes = HealthProbes(app=SimpleNamespace(_asr_task=t))
        return probes.asr_alive

    assert asyncio.run(_drive()) is False


def test_asr_alive_false_not_raising_when_asr_task_cancelled():
    """A cancelled _asr_task means the pipeline was TOLD to stop --
    asr_alive must report False, not raise. task.exception() on a
    cancelled task raises asyncio.CancelledError (a BaseException, NOT
    an Exception on this Python), so without the cancelled() guard the
    property explodes out of the /healthz handler past Starlette's
    ServerErrorMiddleware during the shutdown drain window.

    Seeded defect (one line): remove the ``if task.cancelled(): return
    False`` guard in HealthProbes.asr_alive -> this test dies with
    CancelledError (error, not assertion) -> red.
    """
    async def _drive() -> bool:
        async def _forever():
            await asyncio.Event().wait()

        t = asyncio.create_task(_forever())
        await asyncio.sleep(0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        assert t.cancelled(), "harness drift: task did not end cancelled"
        probes = HealthProbes(app=SimpleNamespace(_asr_task=t))
        return probes.asr_alive

    assert asyncio.run(_drive()) is False
