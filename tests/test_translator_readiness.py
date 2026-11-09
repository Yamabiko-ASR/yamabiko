"""Translator.wait_until_ready -- startup readiness gate.

A fake httpx client with a .get() that replays a sequence of statuses /
exceptions (repeating the last once exhausted, for perpetual-503 cases).
The subject is the readiness LOOP logic; the real-network proof lives in
tests/test_real_translator_readiness.py.
"""
from __future__ import annotations

import logging
import time

import httpx
import pytest

from yamabiko.translation.translator import (
    LlamaNotReadyError, Translator, TranslationConfig,
)


class _FakeGetClient:
    def __init__(self, results):
        self._results = list(results)
        self._last = self._results[-1] if self._results else None
        self.calls = 0
        self.get_kwargs: list[dict] = []

    async def get(self, url, **kwargs):
        self.calls += 1
        self.get_kwargs.append(kwargs)
        result = self._results.pop(0) if self._results else self._last
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self):
        pass


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "http://fake/health"))


def _cfg() -> TranslationConfig:
    return TranslationConfig(
        endpoint="http://localhost:8090/v1/chat/completions",
        model="test", max_tokens=40, temperature=0.1,
    )


@pytest.mark.asyncio
async def test_waits_through_503_until_ready():
    """Seeded defect: treat the first non-200 as fatal (return/raise on it)
    instead of polling -> calls==1, ready never reached -> red."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(503), _resp(503), _resp(200)])
    await t.wait_until_ready(timeout_s=5.0, poll_s=0.01)
    assert t._client.calls == 3


@pytest.mark.asyncio
async def test_timeout_raises_llama_not_ready_when_never_serving():
    """Seeded defect: classify 503 as ready (return on any response) ->
    no raise -> red."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(503)])  # repeats forever
    with pytest.raises(LlamaNotReadyError):
        await t.wait_until_ready(timeout_s=0.2, poll_s=0.02)


@pytest.mark.asyncio
async def test_ready_immediately_does_not_pre_sleep(monkeypatch):
    """A 200 on the first poll returns with ZERO prior sleeps. Seeded defect:
    move `await asyncio.sleep(poll_s)` BEFORE the first get in wait_until_ready
    -> a 'sleep' is recorded before the first 'get' -> red. Deterministic
    (no wall-clock / suite-timeout dependency)."""
    import yamabiko.translation.translator as tr_mod
    events: list[str] = []

    async def _spy_sleep(_delay, *a, **k):
        events.append("sleep")

    monkeypatch.setattr(tr_mod.asyncio, "sleep", _spy_sleep)

    class _RecordingClient(_FakeGetClient):
        async def get(self, url, **kwargs):
            events.append("get")
            return await super().get(url, **kwargs)

    t = Translator(_cfg())
    t._client = _RecordingClient([_resp(200)])
    await t.wait_until_ready(timeout_s=5.0, poll_s=2.0)
    assert events == ["get"], f"expected exactly one GET and no pre-sleep, got {events}"


@pytest.mark.asyncio
async def test_missing_health_endpoint_returns_without_blocking():
    """404/405 = this llama build has no /health -> don't block; let
    sanity_check gate. Seeded defect: loop/raise on 404 -> red (LlamaNotReadyError
    or calls>1)."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(404)])
    await t.wait_until_ready(timeout_s=5.0, poll_s=0.01)
    assert t._client.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 504])
async def test_gateway_5xx_then_200_polls_until_ready(status, caplog):
    """Any 5xx is the still-loading class, not just llama's own 503: a gateway
    (nginx/traefik) in front of a cold llama-server returns 502/504 for the
    5-20 s the backend spends loading the GGUF. The gate must keep polling
    through the 5xx window and return once 200 arrives -- silently, with no
    gate-skip WARNING (skipping after one 502 pushed the cold start onto
    sanity_check's 2-attempt/1-s retry -> exit 43 on a setup that worked
    before). Seeded defect: narrow the still-loading branch back to
    `status_code == 503` only (`if r.status_code != 503: ...return`) ->
    the first 502 skips the gate -> calls == 1 and a WARNING is logged ->
    red twice over."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(status), _resp(status), _resp(200)])
    with caplog.at_level(logging.WARNING, logger="yamabiko.translation.translator"):
        await t.wait_until_ready(timeout_s=5.0, poll_s=0.01)
    assert t._client.calls == 3, (
        f"expected to poll through both {status}s to the 200, got {t._client.calls} probe(s)"
    )
    assert not caplog.records, (
        f"expected NO gate-skip warning for HTTP {status} (still-loading class), got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 504])
async def test_persistent_gateway_5xx_raises_llama_not_ready_at_deadline(status):
    """A 5xx that never clears (backend never finishes loading / is dead
    behind the proxy) must exhaust the startup budget and raise
    LlamaNotReadyError -- exactly the perpetual-503 contract, extended to the
    proxy statuses. Seeded defect: same one-liner as above (skip branch
    condition back to `!= 503`) -> wait_until_ready returns after one probe
    instead of raising -> red."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(status)])  # repeats forever
    with pytest.raises(LlamaNotReadyError):
        await t.wait_until_ready(timeout_s=0.2, poll_s=0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 302])
async def test_unexpected_status_skips_gate_with_warning_instead_of_raising(status, caplog):
    """A non-200/404/405/non-5xx status (auth proxy 401/403, redirect 3xx) is
    NOT a loading signal: an external endpoint may lock down or redirect
    /health yet serve chat completions fine. The gate must log a WARNING
    naming the status and return promptly after ONE probe, leaving
    sanity_check as the arbiter -- not poll the full startup budget and raise
    (which surfaced as exit 43 on working external endpoints). Seeded defect:
    widen the still-loading branch to swallow these too (drop the warn+return
    branch so everything non-200/404/405 keeps polling) -> polls to the
    deadline and raises LlamaNotReadyError -> red."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(status)])  # repeats forever
    start = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="yamabiko.translation.translator"):
        await t.wait_until_ready(timeout_s=5.0, poll_s=0.05)
    elapsed = time.monotonic() - start
    assert t._client.calls == 1, f"expected one probe then gate-skip, got {t._client.calls}"
    assert elapsed < 2.0, f"gate must return promptly, took {elapsed:.2f}s of the 5s budget"
    warned = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and str(status) in r.getMessage()
    ]
    assert warned, f"expected a WARNING naming HTTP {status} when skipping the readiness gate"


@pytest.mark.asyncio
async def test_connect_error_treated_as_still_loading():
    """A ConnectError (server not bound yet) is transient. Seeded defect:
    don't catch httpx.RequestError -> the ConnectError propagates -> red."""
    t = Translator(_cfg())
    t._client = _FakeGetClient([httpx.ConnectError("refused"), _resp(200)])
    await t.wait_until_ready(timeout_s=5.0, poll_s=0.01)
    assert t._client.calls == 2


@pytest.mark.asyncio
async def test_readiness_get_uses_bounded_per_poll_timeout():
    """The /health probe must pass the short per-poll timeout (not the client's
    30 s default), so a hung poll can't overshoot the startup budget. Seeded
    defect: revert `get(url, timeout=_READINESS_PROBE_TIMEOUT_S)` to `get(url)`
    in wait_until_ready -> no timeout kwarg recorded -> red."""
    from yamabiko.translation.translator import _READINESS_PROBE_TIMEOUT_S
    t = Translator(_cfg())
    t._client = _FakeGetClient([_resp(200)])
    await t.wait_until_ready(timeout_s=5.0, poll_s=0.01)
    assert t._client.get_kwargs[0].get("timeout") == _READINESS_PROBE_TIMEOUT_S
