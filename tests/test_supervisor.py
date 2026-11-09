"""Supervisor wraps pipeline coroutines with exception capture.

Each pipeline stage (vad, asr, translation, overlay) runs inside
`supervise` so a single bad segment can't kill the process. Errors are
logged, counted, and reported via an injected notifier (tray in prod,
spy in tests). If a stage crashes 10 times in 60 s, the supervisor
escalates — the escalation hook is called once per threshold crossing.
"""
from __future__ import annotations

import time

import pytest

from yamabiko import supervisor


@pytest.mark.asyncio
async def test_supervise_forwards_success():
    async def stage():
        return "ok"
    result = await supervisor.supervise("asr", stage, notify=lambda *a, **k: None)
    assert result == "ok"


@pytest.mark.asyncio
async def test_supervise_captures_exception_and_notifies():
    notes: list[tuple[str, str]] = []

    async def stage():
        raise RuntimeError("boom")

    # Rate-limit off for the test — we want to observe every notification
    tracker = supervisor.ErrorTracker(stage_window_s=60.0, max_notify_per_minute=999)
    result = await supervisor.supervise(
        "asr", stage,
        notify=lambda stage_name, msg: notes.append((stage_name, msg)),
        tracker=tracker,
    )
    assert result is None
    assert len(notes) == 1
    assert notes[0][0] == "asr"
    assert "boom" in notes[0][1]


@pytest.mark.asyncio
async def test_notifications_rate_limited_per_stage_per_minute():
    notes: list[tuple[str, str]] = []
    tracker = supervisor.ErrorTracker(stage_window_s=60.0, max_notify_per_minute=1)

    async def stage():
        raise RuntimeError("boom")

    for _ in range(5):
        await supervisor.supervise(
            "asr", stage,
            notify=lambda s, m: notes.append((s, m)),
            tracker=tracker,
        )
    assert len(notes) == 1


@pytest.mark.asyncio
async def test_escalates_after_threshold():
    escalations: list[str] = []
    tracker = supervisor.ErrorTracker(
        stage_window_s=60.0, escalation_threshold=3, max_notify_per_minute=999,
    )

    async def stage():
        raise RuntimeError("boom")

    for _ in range(3):
        await supervisor.supervise(
            "asr", stage,
            notify=lambda s, m: None,
            on_escalate=lambda s: escalations.append(s),
            tracker=tracker,
        )
    assert escalations == ["asr"]


@pytest.mark.asyncio
async def test_escalation_fires_once_per_threshold_crossing():
    escalations: list[str] = []
    tracker = supervisor.ErrorTracker(
        stage_window_s=60.0, escalation_threshold=2, max_notify_per_minute=999,
    )

    async def stage():
        raise RuntimeError("boom")

    for _ in range(5):
        await supervisor.supervise(
            "asr", stage,
            notify=lambda s, m: None,
            on_escalate=lambda s: escalations.append(s),
            tracker=tracker,
        )
    # 5 errors, threshold 2 — two full threshold crossings: at errors 2 and 4
    assert escalations == ["asr", "asr"]
