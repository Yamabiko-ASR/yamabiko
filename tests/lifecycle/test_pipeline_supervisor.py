"""Unit tests for PipelineSupervisor.

Each test's docstring carries a "Fails red if:" line per TESTING.md §3.1.
PipelineSupervisor is pipeline's inner supervisor for overlay; these
tests exercise the two load-bearing behaviours: (a) overlay is
actually assigned to the inner Job Object on spawn so pipeline-death
cascades kernel-enforced, and (b) stop() actually fires the shutdown
event so overlay gets its graceful-exit ping.
"""
from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Inner jobs are Windows-only",
)

from yamabiko.lifecycle.pipeline_supervisor import PipelineSupervisor


def test_supervisor_assigns_overlay_to_inner_job():
    """Fails red if: supervisor spawns overlay but forgets to assign it
    to the inner job -- overlay would survive pipeline crash."""
    inner_job = MagicMock()
    inner_job.assign.return_value = True

    fake_proc = MagicMock()
    fake_proc._handle = 0xDEADBEEF
    fake_proc.pid = 12345

    supervisor = PipelineSupervisor(
        spawn=lambda: fake_proc,
        inner_job=inner_job,
        shutdown_writer=MagicMock(),
        stop_event=threading.Event(),
    )
    supervisor.spawn_once()
    inner_job.assign.assert_called_once_with(fake_proc)


def test_spawn_once_prefers_atomic_spawn_in_job_and_skips_reassign():
    """When an atomic ``spawn_in_job`` callable is provided,
    ``spawn_once`` must (a) route the spawn through it, handing it the
    inner job so assignment happens while the child is still
    CREATE_SUSPENDED, (b) NOT call the legacy two-step ``spawn()`` +
    ``inner_job.assign()`` -- a second assign of an already-member
    process fails with ERROR_ACCESS_DENIED and would log a spurious
    cascade warning on every spawn -- and (c) still publish the proc
    so stop() can reach it.

    Fails red if: spawn_once ignores the atomic callable (legacy spawn
    raises here), or double-assigns the returned proc.

    Seeded-defect drill: make spawn_once call ``self._inner_job.assign
    (proc)`` after the atomic spawn -> assert_not_called goes red.
    """
    inner_job = MagicMock()
    inner_job.assign = MagicMock(return_value=True)

    fake_proc = MagicMock()
    fake_proc._handle = 0xF75
    fake_proc.pid = 7575

    received_jobs: list = []

    def _atomic(job):
        received_jobs.append(job)
        return fake_proc

    def _legacy_spawn():
        raise AssertionError(
            "legacy spawn() must not be called when spawn_in_job is "
            "provided -- the two-step path re-opens the race window"
        )

    sv = PipelineSupervisor(
        spawn=_legacy_spawn,
        spawn_in_job=_atomic,
        inner_job=inner_job,
        shutdown_writer=MagicMock(),
        stop_event=threading.Event(),
    )
    proc = sv.spawn_once()

    assert proc is fake_proc
    assert received_jobs == [inner_job], (
        f"spawn_in_job received {received_jobs}; it must be handed the "
        f"supervisor's inner job (and nothing else) so the suspended "
        f"child is assigned to the RIGHT job"
    )
    inner_job.assign.assert_not_called()
    # The proc must still be published for stop().
    sv.stop()
    assert fake_proc.wait.called, (
        "spawn_once did not publish the atomically-spawned proc; "
        "stop() could not reach it (orphaned overlay)"
    )


# The standalone signals-on-stop test that lived here was deleted --
# both escalation-ladder tests below assert
# writer.signal.assert_called_once() WITH ordering, fully subsuming it.


class _OverlayThatIgnoresShutdown:
    """Test double for the escalation-ladder tests: an overlay that
    ignores the named-event shutdown signal. ``wait()`` raises
    TimeoutExpired until the configured escalation method
    (``terminate`` or ``kill``) has been called; after that, ``wait()``
    returns 0. Records the full call sequence so ordering can be
    asserted.
    """

    def __init__(self, dies_on: str = "terminate") -> None:
        assert dies_on in ("terminate", "kill")
        self._dies_on = dies_on
        self._dead = False
        self.calls: list[tuple[str, object]] = []
        self._handle = 0xF27
        self.pid = 2727

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self._dead:
            return 0
        import subprocess as _sp
        raise _sp.TimeoutExpired(cmd="overlay", timeout=timeout)

    def terminate(self) -> None:
        self.calls.append(("terminate", None))
        if self._dies_on == "terminate":
            self._dead = True

    def kill(self) -> None:
        self.calls.append(("kill", None))
        self._dead = True


def test_stop_terminates_overlay_that_ignores_grace():
    """An overlay that ignores the shutdown event past grace_s must be
    force-terminated by stop()'s escalation ladder -- not left lingering
    on screen until the inner job closes.

    Fails red if: ``proc.terminate()`` (or the surrounding
    TimeoutExpired fallback) is deleted from
    PipelineSupervisor._shutdown_proc.

    Seeded-defect drill: replace ``proc.terminate()`` in _shutdown_proc
    with ``pass`` -> the grace wait times out, terminate never fires,
    the post-terminate wait(2.0) also times out, kill ends it -> the
    "no kill" assertion goes red (and under full deletion of the except
    block, the terminate assertion goes red).
    """
    proc = _OverlayThatIgnoresShutdown(dies_on="terminate")
    writer = MagicMock()
    sv = PipelineSupervisor(
        spawn=lambda: proc,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=writer,
        stop_event=threading.Event(),
        grace_s=0.05,  # fake raises instantly; no real sleeping
    )
    sv.spawn_once()
    sv.stop()

    # The graceful signal fired first.
    writer.signal.assert_called_once()
    # Ladder shape: grace-shaped wait, then terminate, then the
    # post-terminate wait that resolves. Kill must NOT fire for a child
    # that dies on terminate.
    assert proc.calls[0] == ("wait", 0.05), (
        f"first escalation step must be the grace-window wait "
        f"(timeout=grace_s=0.05); got {proc.calls[:1]}"
    )
    assert ("terminate", None) in proc.calls, (
        "overlay ignored the shutdown event past grace_s but was never "
        "terminate()d -- a hung overlay lingers on screen until the "
        "pipeline process itself exits."
    )
    assert ("kill", None) not in proc.calls, (
        f"kill() fired for a child that exited on terminate(); the "
        f"ladder is skipping the post-terminate wait. calls={proc.calls}"
    )
    assert proc.calls.index(("terminate", None)) > 0, (
        "terminate() fired before the grace wait -- the graceful window "
        "was skipped"
    )


def test_stop_kills_overlay_that_survives_terminate():
    """Second escalation rung: an overlay that survives terminate()
    (hung hard enough that TerminateProcess didn't take, or a fake that
    swallows it) must be kill()ed after the 2 s post-terminate wait.

    Fails red if: ``proc.kill()`` (or the inner TimeoutExpired branch)
    is deleted from _shutdown_proc.

    Seeded-defect drill: replace ``proc.kill()`` in _shutdown_proc's
    inner except branch with ``pass`` -> kill never recorded -> red.
    """
    proc = _OverlayThatIgnoresShutdown(dies_on="kill")
    writer = MagicMock()
    sv = PipelineSupervisor(
        spawn=lambda: proc,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=writer,
        stop_event=threading.Event(),
        grace_s=0.05,
    )
    sv.spawn_once()
    sv.stop()

    writer.signal.assert_called_once()
    names = [name for name, _ in proc.calls]
    assert "terminate" in names and "kill" in names, (
        f"expected the full terminate->kill ladder for a child that "
        f"survives terminate; got calls={proc.calls}."
    )
    assert names.index("kill") > names.index("terminate"), (
        f"kill() fired before terminate(); escalation order broken. "
        f"calls={proc.calls}"
    )
    # The ladder must resolve: the final wait must return (child dead).
    assert proc.calls[-1] == ("wait", 2.0), (
        f"the ladder must end with the post-kill wait(2.0) resolving; "
        f"got {proc.calls[-3:]}"
    )


def _fake_proc_that_crashes_immediately(pid: int) -> MagicMock:
    """Test double that returns from proc.wait() on first call (simulates
    an already-dead overlay). Used for anti-flap tests below."""
    p = MagicMock()
    p._handle = 0xDEAD + pid
    p.pid = pid
    p.wait = MagicMock(return_value=137)  # SIGKILL-equivalent on Windows
    return p


def test_supervisor_gives_up_after_three_respawns_with_max_restarts_three():
    """Anti-flap (L1) + M-01 budget semantics: ``max_restarts=N`` must
    permit exactly N actual RE-spawns before give-up -- the initial spawn
    is not a respawn and must not consume a budget slot.

    With max_restarts=3 the supervisor spawns once, then respawns three
    times as the overlay crash-loops (4 spawns total), then gives up. The
    OLD off-by-one counted the initial failure's timestamp toward the cap
    (``>=``), so max_restarts=3 yielded only 2 respawns / 3 total spawns.

    Fails red if: overlay respawns unboundedly (cap not enforced), OR the
    budget is off by one.

    Seeded-defect drill: revert the cap check from
    ``len(self._restart_times) > self._max_restarts`` back to ``>=`` ->
    spawn_count == 3 instead of 4 -> the ``== 4`` assertion goes red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        return _fake_proc_that_crashes_immediately(1000 + spawn_count["n"])

    stop_event = threading.Event()
    supervisor = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=3,
        restart_window_s=60.0,
        backoff_s=(0.0,),  # deterministic: no real sleep in tests
    )
    # .run() must terminate once the cap is hit.
    t = threading.Thread(target=supervisor.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), (
        "supervisor.run did not return after hitting max_restarts; "
        "anti-flap cap is not working"
    )
    assert spawn_count["n"] == 4, (
        f"expected 1 initial spawn + max_restarts=3 respawns = 4 spawns "
        f"before give-up, got {spawn_count['n']} (3 == the off-by-one "
        f"that counted the initial failure against the budget)"
    )
    assert stop_event.is_set(), (
        "anti-flap give-up must set stop_event so pipeline observers "
        "(watchdog / downstream supervisors) see the shutdown"
    )
    assert supervisor.gave_up is True, (
        "cap-exhaustion give-up must set the observable gave_up flag so "
        "run_pipeline can arm EXIT_CHILD_GIVEUP (47) instead of exit 0"
    )


def test_supervisor_max_restarts_one_permits_exactly_one_respawn():
    """M-01 boundary: ``max_restarts=1`` permits exactly ONE respawn --
    the initial spawn plus one respawn = 2 spawns total, then give-up.

    This is the discriminating boundary for the off-by-one: the OLD
    ``>=`` check gave up after the very first crash (0 respawns / 1
    spawn), silently killing the session on a single overlay hiccup.

    Fails red if: the budget is off by one (``>=`` gives spawn_count==1).

    Seeded-defect drill: revert the cap check ``>`` -> ``>=`` ->
    spawn_count == 1 -> the ``== 2`` assertion goes red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        return _fake_proc_that_crashes_immediately(1100 + spawn_count["n"])

    stop_event = threading.Event()
    supervisor = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=1,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    t = threading.Thread(target=supervisor.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "supervisor.run did not return after the cap"
    assert spawn_count["n"] == 2, (
        f"max_restarts=1 must permit exactly 1 respawn (2 spawns total); "
        f"got {spawn_count['n']} (1 == the off-by-one)"
    )
    assert stop_event.is_set()
    assert supervisor.gave_up is True


class _GaveUpAtSetEvent(threading.Event):
    """threading.Event that snapshots the supervisor's ``gave_up`` flag
    at the exact moment ``set()`` runs.

    This is the observation point production uses: run_pipeline wakes
    from ``asyncio.to_thread(overlay_stop.wait)`` the instant set()
    lands and immediately reads ``gave_up`` (main.py's post-wait
    give-up branch). Joining the supervisor thread and then asserting
    both flags cannot see a swapped set order -- both end states are
    True either way -- so the ordering must be captured AT set() time.

    Wire-up: construct, pass as ``stop_event=``, then point
    ``read_gave_up`` at ``lambda: sv.gave_up`` before the supervisor
    thread starts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gave_up_at_set: list[bool] = []
        self.read_gave_up = None  # set post-construction to lambda: sv.gave_up

    def set(self) -> None:
        if self.read_gave_up is not None:
            self.gave_up_at_set.append(self.read_gave_up())
        super().set()


def test_supervisor_crash_in_loop_sets_gave_up_before_stop_event():
    """An UNEXPECTED exception inside the supervisor loop (a
    supervisor-code bug, not an overlay crash -- overlay crashes and spawn
    exceptions are caught inside the loop) must mark gave_up BEFORE
    tripping stop_event: run_pipeline reads gave_up the instant
    ``asyncio.to_thread(overlay_stop.wait)`` wakes, so a
    stop_event-first order re-opens the silent-death race -- the
    pipeline reads gave_up=False, returns 0 (STOP_SELF) and the
    watchdog silently cascades a full stop instead of arming
    EXIT_CHILD_GIVEUP (47, watchdog RESTART).

    The ordering is made deterministically observable via
    _GaveUpAtSetEvent: stop_event.set() itself snapshots sv.gave_up, so
    the assertion sees exactly what production's post-wait read sees.

    The crash is injected via a raising ``clock``: _evict_old_restarts
    calls it at the top of every loop iteration, outside any inner
    try/except, so the exception reaches the defensive ``except
    Exception`` handler -- the exact branch under test.

    Fails red if: the except branch sets stop_event before (or without)
    marking gave_up.

    Seeded-defect drill: in run()'s except branch, move ``self.gave_up
    = True`` below ``self._stop_event.set()`` -> set() snapshots False
    -> the ``== [True]`` assertion goes red.
    """
    stop_event = _GaveUpAtSetEvent()

    def _raising_clock() -> float:
        raise RuntimeError("simulated supervisor-code bug")

    sv = PipelineSupervisor(
        spawn=lambda: MagicMock(),  # never reached: clock raises first
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        clock=_raising_clock,
        backoff_s=(0.0,),
    )
    stop_event.read_gave_up = lambda: sv.gave_up
    t = threading.Thread(target=sv.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "supervisor.run did not return after the crash"
    assert stop_event.is_set(), (
        "a supervisor-loop crash must trip stop_event so the pipeline "
        "tears down instead of running headless"
    )
    assert stop_event.gave_up_at_set == [True], (
        f"gave_up snapshots taken inside stop_event.set(): "
        f"{stop_event.gave_up_at_set}. The except branch must mark "
        f"gave_up BEFORE tripping stop_event -- run_pipeline reads "
        f"gave_up immediately after overlay_stop.wait() wakes, so a "
        f"False-at-set-time value means the pipeline exits 0 and the "
        f"watchdog silently cascades a full stop"
    )
    assert sv.gave_up is True, (
        "a supervisor-loop crash must set gave_up: supervision failed and "
        "the overlay will not respawn -- without the flag the pipeline "
        "exits 0 and the watchdog silently cascades a full stop"
    )


def test_supervisor_cap_give_up_sets_gave_up_before_stop_event():
    """The respawn-cap give-up branch must mark gave_up BEFORE tripping
    stop_event -- same load-bearing contract as the crash branch above:
    run_pipeline reads gave_up the instant overlay_stop.wait() wakes,
    so a stop_event-first order lets the pipeline read gave_up=False,
    return 0 (STOP_SELF), and the watchdog cascade a silent full stop
    instead of a RESTART with a fresh overlay budget.

    Paired with the crash-branch test so BOTH failure paths that set
    gave_up have their ordering pinned; a swap in either branch alone
    is caught by exactly one of the pair (verified during the drill).

    Fails red if: the cap branch sets stop_event before (or without)
    marking gave_up.

    Seeded-defect drill: in run()'s cap branch, move ``self.gave_up =
    True`` below ``self._stop_event.set()`` -> set() snapshots False ->
    the ``== [True]`` assertion goes red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        return _fake_proc_that_crashes_immediately(4000 + spawn_count["n"])

    stop_event = _GaveUpAtSetEvent()
    sv = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=1,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    stop_event.read_gave_up = lambda: sv.gave_up
    t = threading.Thread(target=sv.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "supervisor.run did not return after the cap"
    assert stop_event.is_set(), (
        "cap-exhaustion give-up must trip stop_event so pipeline "
        "observers see the shutdown"
    )
    assert stop_event.gave_up_at_set == [True], (
        f"gave_up snapshots taken inside stop_event.set(): "
        f"{stop_event.gave_up_at_set}. The cap branch must mark gave_up "
        f"BEFORE tripping stop_event -- otherwise run_pipeline's "
        f"post-wait read races to False and the give-up masquerades as "
        f"a clean quit (exit 0, silent full cascade)"
    )
    assert sv.gave_up is True, (
        "cap-exhaustion give-up must leave the observable gave_up flag set"
    )


def test_supervisor_normal_overlay_exit_does_not_set_gave_up():
    """The gave_up flag distinguishes a definitive supervision failure
    (respawn-budget exhaustion, supervisor-loop crash) from a normal
    overlay exit. A clean overlay exit (rc==0, e.g. tray Quit)
    sets stop_event but must leave gave_up False -- otherwise run_pipeline
    would arm EXIT_CHILD_GIVEUP (47, watchdog RESTART) on every clean
    quit and the app would loop instead of stopping.

    Fails red if: gave_up is set on ANY stop path rather than only the
    failure paths (cap exhaustion, supervisor-loop crash).

    Seeded-defect drill: move ``self.gave_up = True`` out of the cap
    branch (e.g. into stop()) -> a clean exit sets gave_up -> red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        p = MagicMock()
        p._handle = 0xC0DE
        p.pid = 4343
        p.wait = MagicMock(return_value=0)  # clean exit on first poll
        return p

    stop_event = threading.Event()
    sv = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=3,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    t = threading.Thread(target=sv.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "supervisor.run did not return after a clean exit"
    assert spawn_count["n"] == 1, (
        f"a clean overlay exit must not respawn; spawn_count="
        f"{spawn_count['n']}"
    )
    assert stop_event.is_set(), (
        "a clean overlay exit must still set stop_event (pipeline teardown)"
    )
    assert sv.gave_up is False, (
        "a clean overlay exit must NOT set gave_up -- only cap-exhaustion "
        "does; otherwise every tray-Quit would arm EXIT_CHILD_GIVEUP (47)"
    )


def test_supervisor_max_restarts_zero_still_spawns_once():
    """``max_restarts=0`` must not prevent the FIRST overlay spawn --
    the cap is a budget for *re*-starts, not for the initial start.
    The watchdog's own loop has exactly this semantic (spawn first,
    give up only after crashes); PipelineSupervisor checked the cap at
    the top of the loop, where ``len(deque()) == 0 >= 0`` is True
    before anything ever ran, so the overlay never appeared and the
    supervisor silently set stop_event.

    Fails red if: the cap check runs before the first spawn attempt
    (i.e. without the ``self._restart_times`` non-empty guard), OR the
    initial spawn is followed by a respawn (max_restarts=0 must permit
    ZERO respawns: exactly 1 spawn, then give-up).

    Seeded-defect drill: revert the guard to a bare
    ``if len(self._restart_times) >= self._max_restarts:`` ->
    spawn_count stays 0 -> the == 1 assertion goes red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        return _fake_proc_that_crashes_immediately(3000 + spawn_count["n"])

    stop_event = threading.Event()
    supervisor = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=0,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    t = threading.Thread(target=supervisor.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), (
        "supervisor.run did not return with max_restarts=0; the cap "
        "logic should give up after the first crash"
    )
    assert spawn_count["n"] == 1, (
        f"max_restarts=0 must permit the FIRST overlay spawn but ZERO "
        f"respawns (spawn_count must be exactly 1); got "
        f"{spawn_count['n']}. 0 == the initial spawn was suppressed; "
        f">1 == a respawn was wrongly permitted"
    )
    assert stop_event.is_set(), (
        "after the post-crash give-up the supervisor must still set "
        "stop_event so the pipeline tears down visibly"
    )
    assert supervisor.gave_up is True, (
        "give-up after the first crash must set the observable gave_up flag"
    )


def test_supervisor_retries_spawn_failure_with_backoff():
    """Spawn retry (L4): a transient spawn-time exception doesn't burn
    the session.

    Fails red if: spawn_once's _spawn() raises OSError (transient
    Windows handle exhaustion, a flaky DLL load, etc.) and the old
    run() immediately sets stop_event and exits. Now the supervisor
    should count it as a restart and keep going up to max_restarts,
    then give up if the failure persists past the cap.

    Seeded-defect drill: reverting run() to the pre-L4 'except
    Exception: stop_event.set(); return' shape makes spawn_count stay
    at 1 and stop_event set immediately.
    """
    spawn_count = {"n": 0}
    failures_wanted = 2  # fail twice, succeed third time

    def _fake_spawn():
        spawn_count["n"] += 1
        if spawn_count["n"] <= failures_wanted:
            raise OSError("simulated transient spawn failure")
        # Long-running on the 3rd try: wait never returns until we set stop.
        p = MagicMock()
        p._handle = 0xABCD
        p.pid = 9999
        # Make proc.wait respect the supervisor's stop by blocking on
        # the event. Supervisor polls with timeout=1.0; we simulate a
        # child that stays alive until stop_event is set via returning
        # TimeoutExpired repeatedly, then on the final call returns 0.
        import subprocess as _sp
        wait_calls = {"n": 0}

        def _wait(timeout=None):  # noqa: ARG001
            wait_calls["n"] += 1
            if stop_event.is_set():
                return 0
            raise _sp.TimeoutExpired(cmd="fake", timeout=timeout)

        p.wait = _wait
        return p

    stop_event = threading.Event()
    supervisor = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=5,          # > failures_wanted so we actually succeed
        restart_window_s=60.0,
        backoff_s=(0.0,),        # deterministic
    )
    t = threading.Thread(target=supervisor.run, daemon=True)
    t.start()
    # Give the supervisor time to fail-fail-succeed and enter the wait loop.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and spawn_count["n"] < failures_wanted + 1:
        time.sleep(0.05)
    assert spawn_count["n"] >= failures_wanted + 1, (
        f"supervisor gave up too early: {spawn_count['n']} spawn attempts "
        f"(expected at least {failures_wanted + 1} -- {failures_wanted} "
        f"failures + 1 success)"
    )
    assert not stop_event.is_set(), (
        "stop_event tripped despite the 3rd spawn succeeding -- L4 retry "
        "budget not honoured"
    )
    # Teardown.
    stop_event.set()
    t.join(timeout=5.0)


def test_supervisor_gives_up_when_every_spawn_raises():
    """L4 + L1 interaction: if spawn keeps raising past max_restarts,
    the supervisor must still hit the anti-flap cap and give up --
    not retry forever, not crash silently.

    Fails red if: the spawn-exception branch lacks the restart-time
    append, so the deque never fills and the cap check never fires;
    OR if the cap path doesn't set stop_event on give-up.

    Seeded-defect drill: remove ``self._restart_times.append(...)`` in
    the ``except Exception`` branch in PipelineSupervisor.run -- deque
    stays empty, give-up never triggered, .run blocks past timeout,
    test's thread.join times out and the first assert fires red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        raise OSError(f"persistent failure #{spawn_count['n']}")

    stop_event = threading.Event()
    supervisor = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=3,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    t = threading.Thread(target=supervisor.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), (
        "supervisor.run did not give up despite every spawn raising; "
        "either the spawn-exception branch isn't recording restart "
        "timestamps, or the cap check is wrong"
    )
    assert spawn_count["n"] == 4, (
        f"expected 1 initial + max_restarts=3 respawn attempts = 4 failed "
        f"spawns before give-up, got {spawn_count['n']}"
    )
    assert stop_event.is_set(), (
        "anti-flap give-up must set stop_event even when the failure "
        "path is spawn-side, not crash-side"
    )


def test_supervisor_does_not_respawn_on_clean_overlay_exit():
    """When the overlay exits cleanly (rc==0 -> classify STOP_SELF,
    e.g. the tray Quit handler), the supervisor must NOT respawn it.

    The old inner loop respawned on ANY exit code, so a clean tray-Quit
    exit-0 (which happens BEFORE the pipeline's finally sets overlay_stop)
    made the supervisor spawn a fresh overlay -- a ghost window that
    flashes back on screen for the uvicorn drain (~3s) on every quit, and
    a 'Quit fails to quit' loop when the WS quit command didn't reach the
    pipeline.

    Seeded-defect drill: revert the classify() check so rc==0 respawns ->
    the supervisor respawns up to the anti-flap cap (spawn_count==3) ->
    the spawn_count==1 assertion goes red.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        p = MagicMock()
        p._handle = 0xC0DE
        p.pid = 4242
        p.wait = MagicMock(return_value=0)  # clean exit on first poll
        return p

    stop_event = threading.Event()
    sv = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=3,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    t = threading.Thread(target=sv.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "supervisor.run did not return after a clean overlay exit"
    assert spawn_count["n"] == 1, (
        f"overlay was respawned after a clean exit-0 (ghost-overlay bug); "
        f"spawn_count={spawn_count['n']} (expected exactly 1)"
    )
    assert stop_event.is_set(), (
        "a clean overlay exit must set stop_event so the pipeline tears down "
        "(forwarded to app.done) instead of running headless"
    )


def test_supervisor_respawns_on_crash_exit_code():
    """Guard the clean-exit fix doesn't over-reach: a CRASH (nonzero,
    non-42/43 -> classify RESTART, e.g. 137) must still respawn up to
    the anti-flap cap.

    Seeded-defect drill: make the inner loop treat every code as
    intentional (stop on any rc) -> a crashing overlay never respawns ->
    spawn_count==1 instead of 4 -> this fails.
    """
    spawn_count = {"n": 0}

    def _fake_spawn():
        spawn_count["n"] += 1
        return _fake_proc_that_crashes_immediately(2000 + spawn_count["n"])  # rc=137

    stop_event = threading.Event()
    sv = PipelineSupervisor(
        spawn=_fake_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=3,
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )
    t = threading.Thread(target=sv.run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive()
    assert spawn_count["n"] == 4, (
        f"a crashing overlay (rc=137 -> RESTART) must respawn to the cap "
        f"(1 initial + 3 respawns); spawn_count={spawn_count['n']} "
        f"(expected 4)"
    )


def test_supervisor_shuts_down_overlay_spawned_during_a_pending_stop():
    """If a full external stop() (overlay_stop.set() + stop()) lands DURING
    spawn_once -- so _stopped is already True and stop() read the previous
    _proc -- run() must still tear down the freshly-spawned overlay, not
    orphan it.

    Seeded-defect drill: remove the post-spawn stop_event re-check in
    run() -> the inner loop's self.stop() is a no-op (_stopped already
    True), so the new proc is never shut down -> fake_proc.wait is never
    called -> this fails.
    """
    import subprocess as _sp

    stop_event = threading.Event()
    fake_proc = MagicMock()
    fake_proc._handle = 0xCAFE
    fake_proc.pid = 5555
    fake_proc.wait = MagicMock(return_value=0)  # exits gracefully on the signal

    holder: dict = {}

    def _spawn():
        # Simulate the main thread completing its whole shutdown mid-spawn:
        # set the stop event AND call stop() (which reads the previous _proc,
        # None here, and flips _stopped True) -- BEFORE this fresh proc is
        # published.
        stop_event.set()
        holder["sv"].stop()
        return fake_proc

    sv = PipelineSupervisor(
        spawn=_spawn,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        backoff_s=(0.0,),
    )
    holder["sv"] = sv

    t = threading.Thread(target=sv.run, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "supervisor.run did not return"
    assert fake_proc.wait.called, (
        "the overlay spawned during a pending stop was orphaned -- never "
        "shut down"
    )


def test_supervisor_polls_wait_across_multiple_timeouts_before_exit():
    """Fails red if: the inner wait loop treats a ``TimeoutExpired``
    as a reason to exit early instead of continuing to poll (mutmut
    mutant 178: ``continue`` -> ``break`` in
    ``PipelineSupervisor.run``'s inner wait loop).

    Under the mutation, the very first ``proc.wait(timeout=1.0)``
    TimeoutExpired exits the inner loop with ``proc_exited=False``,
    so the outer ``if not proc_exited: self.stop(); return`` fires
    after ~1 s of normal operation. That would be a silent
    early-shutdown of every overlay session where the process stays
    alive longer than a single 1 s poll -- i.e. all of them.

    Under normal source: each TimeoutExpired continues the loop;
    the supervisor only exits the inner loop when the proc actually
    exits (``wait`` returns a code) or ``stop_event`` is set. We
    prove this by counting how many times ``proc.wait`` is called
    before the supervisor returns: the mutation caps it at 1, normal
    code polls repeatedly.

    Seeded-defect drill: changing ``continue`` to ``break`` at
    pipeline_supervisor.py:222 makes ``wait_calls`` stop at 1 before
    the stop_event-trigger thread runs; the >= 3 assertion goes red.
    """
    import subprocess as _sp

    stop_event = threading.Event()
    # Track wait() calls that come from the OUTER supervisor polling
    # loop specifically, not from ``stop()``'s cleanup (which also
    # calls proc.wait on grace_s and terminate paths). We distinguish
    # by flipping ``in_stop`` once the supervisor has set stop_event
    # (which happens only inside stop() on the give-up path, or in
    # our stopper thread below after the outer loop has polled enough
    # to be observable). Before stop_event fires, every wait() call
    # was from the outer poll loop.
    outer_wait_calls = {"n": 0}

    def fake_wait(timeout=None):
        if not stop_event.is_set():
            outer_wait_calls["n"] += 1
            # Always timeout in the outer loop: we want the supervisor
            # to stay in the poll loop until stop_event fires.
            raise _sp.TimeoutExpired(cmd="fake", timeout=timeout)
        # stop() cleanup path: let it resolve cleanly so the test
        # teardown doesn't hang or contribute noisy wait counts.
        return 0

    fake_proc = MagicMock()
    fake_proc._handle = 0xBEEF
    fake_proc.pid = 9999
    fake_proc.wait = fake_wait

    sv = PipelineSupervisor(
        spawn=lambda: fake_proc,
        inner_job=MagicMock(assign=MagicMock(return_value=True)),
        shutdown_writer=MagicMock(),
        stop_event=stop_event,
        max_restarts=999,  # don't trip anti-flap cap
        restart_window_s=60.0,
        backoff_s=(0.0,),
    )

    # Give the outer loop a brief window to hot-poll fake_wait before
    # setting stop_event. With an immediately-raising fake_wait, the
    # outer loop iterates thousands of times per second; 0.3 s is
    # ample for hundreds+ of iterations under normal source, while
    # still being fast overall. Under the mutation the outer loop exits
    # after exactly 1 outer_wait_call and triggers stop() before this
    # thread even sleeps off.
    def _stop_after_delay() -> None:
        time.sleep(0.3)
        stop_event.set()

    stopper = threading.Thread(target=_stop_after_delay, daemon=True)
    stopper.start()

    t_sv = threading.Thread(target=sv.run, daemon=True)
    t_sv.start()
    t_sv.join(timeout=5.0)
    assert not t_sv.is_alive(), "supervisor didn't exit within 5 s"

    # Under the mutation: outer_wait_calls == 1 (break on first
    # TimeoutExpired, exit inner loop, proc_exited=False, stop()).
    # Under normal source: hundreds+ of TimeoutExpired iterations
    # during the 0.3 s window -- but the threshold only needs to
    # distinguish "polls repeatedly" (the name's claim) from the
    # mutant's exactly-1, and a starved CI runner can be descheduled
    # for most of the window. >= 3 keeps the full discriminating
    # power with zero throughput sensitivity.
    assert outer_wait_calls["n"] >= 3, (
        f"outer-loop wait() was called only {outer_wait_calls['n']} "
        f"time(s) before the supervisor exited; under normal source "
        f"the inner loop polls on TimeoutExpired (continue) and "
        f"iterates many times in the {0.3} s window. Mutant 178 "
        f"('continue' -> 'break') regression: the very first "
        f"TimeoutExpired breaks out of the inner loop, triggering "
        f"premature self.stop() and supervisor return."
    )
