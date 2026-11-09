"""Pipeline-stage exception supervisor.

Each pipeline stage is wrapped in `supervise(...)` so exceptions don't
crash the process. Errors are logged in full, counted per-stage, and
summarised to an injected `notify(stage, message)` callback — the tray
icon in production, a spy in tests. Runaway failure (default: 10 in 60 s)
triggers `on_escalate(stage)` so the app can show the structured
startup-error dialog and stop the pipeline.

Not thread-safe by design — assumes a single asyncio event loop. If any
stage ever wraps in ``run_in_executor``, revisit the tracker fields.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, Optional, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ErrorTracker:
    stage_window_s: float = 60.0
    max_notify_per_minute: int = 1
    escalation_threshold: int = 10
    notify_window_s: float = 60.0

    _error_times: dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _notify_times: dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _escalation_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_error(self, stage: str) -> int:
        """Record an error and return the current error count in the window."""
        now = time.monotonic()
        q = self._error_times[stage]
        q.append(now)
        while q and (now - q[0]) > self.stage_window_s:
            q.popleft()
        return len(q)

    def should_notify(self, stage: str) -> bool:
        """True when a notification should fire for this stage right now.

        Enforces max_notify_per_minute within a rolling notify_window_s
        window. Separate from the error-rate window tracked by
        ``record_error`` — we want to surface errors often enough to be
        actionable but not so often we drown the user.
        """
        now = time.monotonic()
        q = self._notify_times[stage]
        while q and (now - q[0]) > self.notify_window_s:
            q.popleft()
        if len(q) >= self.max_notify_per_minute:
            return False
        q.append(now)
        return True

    def should_escalate(self, stage: str, error_count: int) -> bool:
        """Fire once each time error_count crosses a multiple of threshold."""
        crossings = error_count // self.escalation_threshold
        already = self._escalation_count[stage]
        if crossings > already:
            self._escalation_count[stage] = crossings
            return True
        return False


_DEFAULT_TRACKER = ErrorTracker()


def record_failure(
    stage_name: str,
    exc: Exception,
    *,
    notify: Callable[[str, str], None],
    on_escalate: Optional[Callable[[str], None]] = None,
    tracker: Optional[ErrorTracker] = None,
) -> bool:
    """Count one failure against ``stage_name``: record in the tracker,
    rate-limited notify, threshold-crossing escalation.

    This is the shared tail of :func:`supervise`'s except-handler,
    exposed so stages that deliberately swallow per-item exceptions
    (the per-sentence translate skip in translation_stage) still
    feed the SAME escalation contract instead of becoming invisible to
    the supervisor.

    Returns True when the rate-limited notify fired for this failure —
    callers can reuse that decision as a "log the full traceback now"
    gate (one stack per notify window, not one per failure).
    """
    t = tracker if tracker is not None else _DEFAULT_TRACKER
    count = t.record_error(stage_name)
    notified = t.should_notify(stage_name)
    if notified:
        notify(stage_name, f"{type(exc).__name__}: {exc}")
    if on_escalate is not None and t.should_escalate(stage_name, count):
        on_escalate(stage_name)
    return notified


async def supervise(
    stage_name: str,
    coro_fn: Callable[[], Awaitable[T]],
    *,
    notify: Callable[[str, str], None],
    on_escalate: Optional[Callable[[str], None]] = None,
    tracker: Optional[ErrorTracker] = None,
) -> Optional[T]:
    """Run `coro_fn()` under exception supervision.

    Returns the coroutine's result on success; None on exception.
    Exceptions are logged, the notifier is invoked (subject to rate
    limiting), and on_escalate fires when the stage's error count in
    the window crosses the threshold.
    """
    t = tracker if tracker is not None else _DEFAULT_TRACKER
    try:
        return await coro_fn()
    except Exception as exc:
        _log.exception("pipeline stage %s raised: %s", stage_name, exc)
        record_failure(
            stage_name, exc,
            notify=notify, on_escalate=on_escalate, tracker=t,
        )
        return None
