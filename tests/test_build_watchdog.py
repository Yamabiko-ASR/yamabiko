"""Wiring tests for watchdog.build_watchdog().

watchdog.main() delegates supervisor
composition to build_watchdog(argv) -> WatchdogApp and
run_watchdog(app) -> int. Early-exit argv handling (--version,
--help, --run-main, --run-overlay) stays in main() because it
short-circuits before the mutex + supervisor wiring.

Overlay is no longer a watchdog sibling -- it is a pipeline-owned child.
The sibling set is {pipeline, llama}. The paired tests below still assert
the exact name set so that if overlay is accidentally re-added as a
sibling (regression of Hybrid design), the mismatch trips.

Seeded defects (each listed assertion goes RED if the paired
production change is reverted):

- assert len(app.watchdogs) == 2:
    overlay is still built as a watchdog sibling (regression of
    Hybrid design) -- len would become 3 again.
- assert {wd.name for wd in app.watchdogs} == {"pipeline", "llama"}:
    rename any watchdog, or re-add overlay as a sibling.
- assert app.stop_event is not None:
    drop the shared stop_event wiring.
- assert app.llama_stop_event is not None:
    drop the llama-scoped event; the health probe can no longer
    signal llama without tearing pipeline down too.
- assert app.single_instance_handle is not None:
    drop the mutex acquisition.
"""
from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import pytest


class _FakeOuterJob:
    """Stand-in for lifecycle.jobs.create_outer_job's JobObject.

    Without this fake, every build_watchdog() call in this file would
    execute the REAL ``create_outer_job`` + ``job.assign(None)`` --
    joining the pytest process itself to a fresh KILL_ON_JOB_CLOSE Job
    Object ~20x per session and leaking the only handle. Any future
    cleanup that closed those handles would kernel-kill the test runner,
    and every later test's subprocesses silently inherited the nested
    memberships. The real KILL_ON_JOB_CLOSE semantics are covered at the
    right tier by tests/lifecycle/test_jobs.py's real-tier subprocess
    tests; here only the WIRING is under test.
    """
    def __init__(self):
        self.assigned: list = []
        self.closed = False

    def assign(self, proc) -> bool:
        self.assigned.append(proc)
        return True

    def close(self) -> None:
        self.closed = True


class _WatchdogEnv:
    """Return value of the _watchdog_env fixture. Holds the module
    under test PLUS the sentinel mutex-handle the fixture injected
    so tests can assert identity (not just truthiness) on the
    handle stashed on WatchdogApp, PLUS the fake outer jobs created
    (one per create_outer_job call) so tests can assert on job wiring
    without joining pytest to a real KILL_ON_JOB_CLOSE job.
    """
    def __init__(self, module, mutex_sentinel, created_jobs):
        self.module = module
        self.mutex_sentinel = mutex_sentinel
        self.created_jobs = created_jobs


@pytest.fixture
def _watchdog_env(monkeypatch):
    """Skip file logging + fake the Windows mutex acquisition + fake the
    outer Job Object + close every pipeline ShutdownEventWriter the test
    builds.

    The mutex can't be re-acquired by the same process within the
    test run, so swap it for a fixture-owned sentinel object. Tests
    can then assert `app.single_instance_handle is env.mutex_sentinel`,
    which proves the handle came from _acquire_single_instance_mutex
    and was stored on the app -- not just "some non-None object."

    ``create_outer_job`` is stubbed with ``_FakeOuterJob`` (a NEW
    instance per call, so the "all supervisors share ONE job" identity
    test still goes red if build_watchdog ever constructs per-supervisor
    jobs). Without the stub, every test here joined the live pytest
    process to a real KILL_ON_JOB_CLOSE job and leaked the handle.

    Every WatchdogApp built through this fixture gets its
    ``pipeline_writer`` closed at teardown to prevent handle leaks:
    without this, leaked handles to ``Local\\yamabiko.shutdown.pipeline``
    could be left in a SET state, making real-tier round-trip tests
    vacuous in full-suite order.
    """
    import yamabiko.watchdog as wd_mod

    mutex_sentinel = object()
    monkeypatch.setattr(
        wd_mod, "_acquire_single_instance_mutex", lambda: mutex_sentinel,
    )
    # Don't attach a rotating file handler in tests.
    from yamabiko import log_config
    monkeypatch.setattr(log_config, "setup_main_log", lambda *a, **kw: None)
    # Crash-evidence arming is process-global (sys/threading excepthook
    # chains + a process-lifetime faulthandler handle in the REAL logs
    # dir): stub it so the many build_watchdog() calls in this file
    # never touch the developer's %LOCALAPPDATA%. The helper's own
    # behavior is covered in tests/test_log_config.py; the call-site
    # wiring by test_build_watchdog_installs_crash_evidence_for_
    # watchdog_role below (which re-patches this stub with a recorder).
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )

    # Never join pytest to a real KILL_ON_JOB_CLOSE job.
    created_jobs: list[_FakeOuterJob] = []

    def _fake_create_outer_job(name):
        job = _FakeOuterJob()
        created_jobs.append(job)
        return job

    monkeypatch.setattr(wd_mod, "create_outer_job", _fake_create_outer_job)

    # Track every WatchdogApp built so teardown can close its
    # pipeline_writer (close() is idempotent; tests that already close
    # in their own finally are unaffected).
    real_build = wd_mod.build_watchdog
    built_apps: list = []

    def _tracking_build(argv=None):
        app = real_build(argv)
        built_apps.append(app)
        return app

    monkeypatch.setattr(wd_mod, "build_watchdog", _tracking_build)

    yield _WatchdogEnv(
        module=wd_mod,
        mutex_sentinel=mutex_sentinel,
        created_jobs=created_jobs,
    )

    for app in built_apps:
        try:
            app.pipeline_writer.close()
        except Exception:
            pass


def test_build_watchdog_wires_two_supervisors(_watchdog_env):
    """build_watchdog() returns a WatchdogApp with exactly two
    Watchdog siblings: pipeline and llama. Overlay is a pipeline-owned
    child, not a watchdog sibling.

    Fails red if: overlay is still built as a watchdog sibling
    (regression of Hybrid design). The count drops if the pipeline
    or llama Watchdog construction is removed entirely, and jumps
    back to 3 if the old overlay construction is reintroduced.
    """
    wd_mod = _watchdog_env.module
    app = wd_mod.build_watchdog([])

    assert len(app.watchdogs) == 2, (
        f"expected 2 supervisors (pipeline, llama); "
        f"got {len(app.watchdogs)} -- if 3, overlay has been "
        f"re-added as a watchdog sibling (Hybrid regression)"
    )
    for i, w in enumerate(app.watchdogs):
        assert isinstance(w, wd_mod.Watchdog), (
            f"app.watchdogs[{i}] is not a Watchdog; got "
            f"{type(w).__name__}"
        )


def test_build_watchdog_wires_pipeline_and_llama_stop_events(_watchdog_env):
    """Post-Task-8 wiring: pipeline supervisor listens on the shared
    stop_event; llama supervisor listens on llama_stop_event so the
    HealthProbe can tear llama down without first taking pipeline down.

    A stop_event -> llama_stop_event mirror thread in run_watchdog
    forwards shutdowns in the correct direction (SIGINT still stops
    llama); that behaviour is covered by a dedicated test below.

    Seeded-defect checks:
      - Swap the llama Watchdog's stop_event back to the main one ->
        health-probe-triggered stops would also drop pipeline (the
        whole point of splitting the events is lost). Test fails at
        the "distinct events" assertion.
      - Drop the llama_stop_event field from WatchdogApp -> pytest
        errors out on attribute access before the assertions run.
    """
    app = _watchdog_env.module.build_watchdog([])

    assert isinstance(app.stop_event, threading.Event)
    assert isinstance(app.llama_stop_event, threading.Event)
    assert app.stop_event is not app.llama_stop_event, (
        "stop_event and llama_stop_event must be distinct; otherwise "
        "the HealthProbe can't target just the llama supervisor."
    )

    # Each supervisor's stop_event must match the role it plays.
    by_name = {w.name: w for w in app.watchdogs}
    assert "pipeline" in by_name and "llama" in by_name, (
        f"expected supervisors {{pipeline, llama}}; got {set(by_name)}"
    )
    assert by_name["pipeline"].stop_event is app.stop_event, (
        "pipeline supervisor must listen on the shared stop_event "
        "(SIGINT / exit 42 / exit 43 drop it)"
    )
    assert by_name["llama"].stop_event is app.llama_stop_event, (
        "llama supervisor must listen on llama_stop_event so the "
        "HealthProbe can stop it without forcing pipeline teardown"
    )


def test_build_watchdog_stores_acquired_mutex_handle(_watchdog_env):
    """build_watchdog must call _acquire_single_instance_mutex AND
    stash the returned handle on the WatchdogApp.

    The fixture injects a unique sentinel as the mutex-fn return
    value; asserting identity (`is env.mutex_sentinel`) proves the
    stored handle came FROM the mutex fn -- not just "some non-None
    value." A Liar-pattern weakness in the previous `is not None`
    assertion: it would have passed if someone hardcoded
    `handle = object()` inside build_watchdog without calling the
    mutex fn at all.

    Seeded-defect check: change `handle = _acquire_single_instance_mutex()`
    to `handle = object()` in build_watchdog -> identity check
    fails because the stored object is no longer the fixture's
    sentinel.

    Dropping the handle reference lets Python GC close it, which
    releases the Win32 named mutex and breaks single-instance
    enforcement. This test is the unit-level guard for that invariant.
    """
    env = _watchdog_env
    app = env.module.build_watchdog([])
    assert app.single_instance_handle is env.mutex_sentinel, (
        "WatchdogApp.single_instance_handle is not the object returned "
        "by _acquire_single_instance_mutex; either the mutex fn was "
        "not called or the return value is not being stored"
    )


def test_build_watchdog_supervisors_named_pipeline_llama(_watchdog_env):
    """Each Watchdog must be named distinctly so log output and
    per-supervisor give-up dialogs identify which one tripped.

    The supervisor set is {pipeline, llama}; overlay is a pipeline-owned
    child.

    Seeded-defect checks:
      - Drop the `name=` kwarg in either Watchdog() call -> default
        "watchdog" used, one entry of the set becomes "watchdog",
        test fails.
      - Re-add overlay as a sibling -> names set gains "overlay",
        test fails (this is the Hybrid-regression tripwire).
    """
    app = _watchdog_env.module.build_watchdog([])
    names = {w.name for w in app.watchdogs}
    assert names == {"pipeline", "llama"}, (
        f"expected named supervisors {{pipeline, llama}}; got {names}. "
        f"If this set contains 'overlay', the Hybrid-design overlay-as-"
        f"pipeline-child refactor has been reverted."
    )


class _FakeProc:
    """Minimal subprocess.Popen stand-in for the pipeline supervisor
    exit-code tests.

    Records when wait() is called and returns the pre-configured
    exit code. Without this, the real _real_spawn() would shell out
    to `python -m yamabiko.main` or a fake yamabiko.exe, neither of
    which we want running in a unit test.
    """
    def __init__(self, exit_code: int = 42):
        self.exit_code = exit_code
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self.exit_code

    def terminate(self):
        pass


def test_build_watchdog_pipeline_run_sets_stop_event_on_exit_42(
    _watchdog_env, monkeypatch,
):
    """When the pipeline child exits with code 42 (user cancelled the
    first-run wizard), the pipeline supervisor's run-loop must set the
    shared stop_event so the llama-server supervisor drops too --
    otherwise it keeps retrying against a pipeline that will never come
    back.

    Handled uniformly by exit_codes.classify() inside Watchdog.run for
    every STOP_ALL code (42 wizard-cancel, 43 tailored-error). This test
    asserts the END behaviour -- a pipeline exit 42 propagates to the
    shared stop_event -- regardless of which mechanism does the work.

    Behavioural assertion: drive wd_pipeline.run() against a fake
    proc that returns exit 42. After .run() returns, stop_event
    must be set.

    Seeded-defect check: in yamabiko/watchdog.py, delete the
    `if policy is ExitPolicy.STOP_ALL: self._stop_event.set()`
    branch in Watchdog.run -> exit 42 no longer propagates, this
    test fails with "stop_event unset after pipeline exit 42".
    """
    env = _watchdog_env
    wd_mod = env.module

    # _real_spawn normally calls subprocess.Popen. Stub it at the
    # module level so build_watchdog wires our fake proc in instead
    # of spawning anything. The signature accepts an optional env kwarg
    # (same as the env-accepting _real_spawn) so _pipeline_spawn can call it.
    fake_proc = _FakeProc(exit_code=42)
    monkeypatch.setattr(wd_mod, "_real_spawn", lambda env=None: fake_proc)

    app = wd_mod.build_watchdog([])
    wd_pipeline = app.watchdogs[0]
    assert wd_pipeline.name == "pipeline", (
        "supervisor order assumption broken: app.watchdogs[0] is no "
        "longer the pipeline supervisor. Update this test to find "
        "the pipeline supervisor by name."
    )

    # Sanity: stop_event starts unset.
    assert not app.stop_event.is_set()

    # Drive the run loop end-to-end. Exit 42 is STOP_ALL, so run()
    # returns after setting stop_event (no restart loop).
    wd_pipeline.run()
    app.pipeline_writer.close()

    assert app.stop_event.is_set(), (
        "pipeline supervisor did not set stop_event when child exited 42 "
        "(wizard cancelled). Llama-server supervisor will keep retrying "
        "against a pipeline that won't come back."
    )


def test_build_watchdog_pipeline_run_does_not_set_stop_event_on_transient_crash(
    _watchdog_env, monkeypatch,
):
    """Paired negative: a transient exit code (RESTART, not a sentinel
    STOP_ALL 42/43 and not a give-up) must NOT propagate to the shared
    stop_event. A single ordinary pipeline crash (exit 1, segfault, etc.)
    should only schedule a restart -- the llama supervisor stays up on
    its own llama_stop_event and continues its independent restart logic.

    Note: a give-up (max-restart cap or sub-min-lifetime crash) is a
    TERMINAL outcome that cascades, so this test does not force a
    give-up. Instead it exercises a genuine RESTART: the
    child crashes (exit 1) within the restart budget, the supervisor
    schedules a respawn, and at that point stop_event must still be
    unset. The loop is stopped via the SECOND spawn so the RESTART path
    -- not a terminal path -- is what we observe.

    Seeded-defect checks:
      - Broaden exit_codes.classify() so a transient non-zero code
        returns ExitPolicy.STOP_ALL instead of RESTART -> the first
        crash cascades -> stop_event set at restart time -> red.
      - Add self._cascade_stop() to the RESTART/backoff fall-through
        in Watchdog.run -> stop_event set after the first restart is
        scheduled -> red.
    """
    env = _watchdog_env
    wd_mod = env.module

    app = wd_mod.build_watchdog([])
    wd_pipeline = app.watchdogs[0]

    # Two spawns: the first child crashes (exit 1 -> RESTART). The
    # second spawn records whether stop_event was set at the moment the
    # supervisor came back round to respawn (it must NOT be -- a RESTART
    # does not cascade), then sets stop_event so the loop exits cleanly
    # via the top-of-loop stop check (NOT a terminal classify path).
    stop_state_at_respawn: list = []
    spawn_count = {"n": 0}

    def _spawn(env=None):
        spawn_count["n"] += 1
        if spawn_count["n"] == 1:
            return _FakeProc(exit_code=1)  # ordinary crash -> RESTART
        # Second spawn == we reached the respawn after a RESTART.
        stop_state_at_respawn.append(app.stop_event.is_set())
        # End the loop without a terminal classify: trip stop_event so
        # _wait_or_stop returns and run() hits the top-of-loop stop check.
        app.stop_event.set()
        return _FakeProc(exit_code=0)

    monkeypatch.setattr(wd_mod, "_real_spawn", _spawn)

    # elapsed >= min_lifetime so the crash is a RESTART, not a fast-crash
    # give-up; big restart budget so we don't hit the cap; no real sleep.
    wd_pipeline._clock = lambda: 0.0
    wd_pipeline._min_lifetime_s = 0.0
    wd_pipeline._max_restarts = 5
    wd_pipeline._sleep = lambda s: None

    try:
        wd_pipeline.run()
        assert stop_state_at_respawn == [False], (
            "stop_event was set after a single transient (non-sentinel, "
            "non-give-up) exit code; the RESTART path is cascading -- "
            "every ordinary supervisor crash now tears down its peers, "
            "defeating independent restart logic."
        )
    finally:
        app.pipeline_writer.close()


def test_build_watchdog_pipeline_run_restarts_on_stage_escalation_exit_45(
    _watchdog_env, monkeypatch,
):
    """A pipeline exit with EXIT_STAGE_ESCALATION (45: a stage crossed its
    supervisor escalation threshold, e.g. translation failing 10x in 60 s)
    must take the RESTART path in Watchdog.run -- respawn scheduled, shared
    stop_event NOT set -- not the STOP_SELF/STOP_ALL cascade. This is the
    watchdog half of the stage-escalation contract: the pipeline child
    raises 45, classify(45) routes RESTART, and no watchdog.py change is
    needed (unknown non-zero codes restart by design; exit_codes.py is the
    single source of truth).

    Same two-spawn probe as the transient-crash test above: the first
    child exits 45; at the second spawn (the respawn) stop_event must
    still be unset, then the probe sets it so the loop exits via the
    top-of-loop stop check (not a terminal classify path).

    Seeded defects (one line each, in yamabiko/lifecycle/exit_codes.py):
      - add EXIT_STAGE_ESCALATION to the STOP_ALL tuple in classify() ->
        the first exit 45 cascades -> stop_event set at respawn time -> red.
      - map EXIT_STAGE_ESCALATION to 0 (EXIT_CLEAN's value) -> classify
        returns STOP_SELF -> no respawn ever happens ->
        stop_state_at_respawn == [] -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STAGE_ESCALATION

    env = _watchdog_env
    wd_mod = env.module

    app = wd_mod.build_watchdog([])
    wd_pipeline = app.watchdogs[0]
    assert wd_pipeline.name == "pipeline", (
        "supervisor order assumption broken: app.watchdogs[0] is no "
        "longer the pipeline supervisor. Update this test to find "
        "the pipeline supervisor by name."
    )

    stop_state_at_respawn: list = []
    spawn_count = {"n": 0}

    def _spawn(env=None):
        spawn_count["n"] += 1
        if spawn_count["n"] == 1:
            # Stage escalation exit -> must be a RESTART, not a cascade.
            return _FakeProc(exit_code=EXIT_STAGE_ESCALATION)
        # Second spawn == the supervisor came back round to respawn.
        stop_state_at_respawn.append(app.stop_event.is_set())
        app.stop_event.set()
        return _FakeProc(exit_code=0)

    monkeypatch.setattr(wd_mod, "_real_spawn", _spawn)

    # min_lifetime_s zeroed so this test pins the PLAIN restart path
    # (elapsed >= min_lifetime). The within-min-lifetime path -- where the
    # fast-crash config-error heuristic must exempt 45 just as it exempts
    # 44 -- is pinned by the sibling test directly below. Big restart
    # budget so we don't hit the anti-flap cap; no real sleep.
    wd_pipeline._clock = lambda: 0.0
    wd_pipeline._min_lifetime_s = 0.0
    wd_pipeline._max_restarts = 5
    wd_pipeline._sleep = lambda s: None

    try:
        wd_pipeline.run()
        assert stop_state_at_respawn == [False], (
            f"pipeline exit {EXIT_STAGE_ESCALATION} (stage escalation) did "
            f"not take the restart path: got {stop_state_at_respawn} "
            f"(expected [False]). [] means no respawn happened (the watchdog "
            f"read the escalation as a terminal exit -- the silent-death "
            f"bug); [True] means the exit cascaded to peers instead of "
            f"restarting."
        )
    finally:
        app.pipeline_writer.close()


def test_build_watchdog_pipeline_run_restarts_on_stage_escalation_within_min_lifetime(
    _watchdog_env, monkeypatch,
):
    """Same stage-escalation contract as the test above, but with the
    pipeline supervisor's min_lifetime_s left at its REALISTIC build default
    (30 s) and the exit-45 landing INSIDE that window. This is the
    production shape: the stage supervisor escalates at 10 errors in a
    rolling 60 s window (supervisor.py: stage_window_s=60,
    escalation_threshold=10), so a deterministically-failing stage at ~2 s
    segment cadence takes the pipeline down ~20 s after start -- under the
    30 s min-lifetime. The test above zeroes _min_lifetime_s, which is
    exactly what hid this bug: pre-fix, a sub-min-lifetime exit 45 hit the
    fast-crash 'configuration error' heuristic, fired the give-up
    (misleading 'Yamabiko failed to start' copy) and cascade-stopped on the
    FIRST escalation instead of restarting.

    Asserts, in production wiring (build_watchdog's on_give_up ->
    _record_error -> app.pending_errors):
      - the respawn still happens with the shared stop_event unset, and
      - NO give-up was recorded (app.pending_errors stays empty).

    Seeded defect (one line, yamabiko/watchdog.py): revert the fast-crash
    exemption from `code not in (EXIT_STREAM_DEATH, EXIT_STAGE_ESCALATION)`
    back to `code != EXIT_STREAM_DEATH` -> the first fast 45 gives up and
    cascades -> no respawn -> stop_state_at_respawn == [] -> red (and
    pending_errors gains the 'Yamabiko failed to start' entry).
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STAGE_ESCALATION

    env = _watchdog_env
    wd_mod = env.module

    app = wd_mod.build_watchdog([])
    wd_pipeline = app.watchdogs[0]
    assert wd_pipeline.name == "pipeline", (
        "supervisor order assumption broken: app.watchdogs[0] is no "
        "longer the pipeline supervisor. Update this test to find "
        "the pipeline supervisor by name."
    )

    # Vacuousness guard: this test only covers the within-min-lifetime path
    # if the build default is actually positive. If a future change zeroes
    # it, fail loudly here instead of silently degrading into a duplicate
    # of the plain-restart test above.
    assert wd_pipeline._min_lifetime_s > 0.0, (
        f"build_watchdog's pipeline min_lifetime_s is "
        f"{wd_pipeline._min_lifetime_s!r}; this test needs a positive "
        f"min-lifetime so the exit-45 lands inside the fast-crash window"
    )

    stop_state_at_respawn: list = []
    spawn_count = {"n": 0}

    def _spawn(env=None):
        spawn_count["n"] += 1
        if spawn_count["n"] == 1:
            # Stage escalation, landing within min_lifetime (clock frozen
            # below) -> must still be a RESTART, not the fast-crash give-up.
            return _FakeProc(exit_code=EXIT_STAGE_ESCALATION)
        # Second spawn == the supervisor came back round to respawn.
        stop_state_at_respawn.append(app.stop_event.is_set())
        app.stop_event.set()
        return _FakeProc(exit_code=0)

    monkeypatch.setattr(wd_mod, "_real_spawn", _spawn)

    # Clock frozen at 0.0 => elapsed == 0.0 < min_lifetime (30 s default):
    # the escalation lands INSIDE the fast-crash window. Deliberately do
    # NOT touch _min_lifetime_s. Big restart budget; no real sleep.
    wd_pipeline._clock = lambda: 0.0
    wd_pipeline._max_restarts = 5
    wd_pipeline._sleep = lambda s: None

    try:
        wd_pipeline.run()
        assert stop_state_at_respawn == [False], (
            f"pipeline exit {EXIT_STAGE_ESCALATION} (stage escalation) "
            f"within min_lifetime did not take the restart path: got "
            f"{stop_state_at_respawn} (expected [False]). [] means no "
            f"respawn happened -- the fast-crash config-error heuristic "
            f"swallowed the escalation and gave up on the FIRST one; "
            f"[True] means the exit cascaded to peers instead of restarting."
        )
        with app.pending_errors_lock:
            recorded = list(app.pending_errors)
        assert recorded == [], (
            f"a stage escalation within min_lifetime must NOT record a "
            f"give-up error (the misleading 'failed to start / "
            f"configuration error' dialog); got "
            f"{[e.title for e in recorded]!r}"
        )
    finally:
        app.pipeline_writer.close()


def test_pipeline_exit_46_gives_up_with_port_dialog_and_no_restart():
    """A pipeline exit with EXIT_STARTUP_FAILURE (46: the overlay server
    could not bind its port) must be TERMINAL in Watchdog.run: fire the
    give-up dialog exactly once naming the configured overlay port,
    cascade (set the shared stop_event), and NOT respawn. A bind conflict
    does not self-heal, so restart churn would just delay the message.

    Seeded defects (one line each):
      - remove EXIT_STARTUP_FAILURE from classify()'s STOP_ALL tuple ->
        46 routes to RESTART -> the child respawns (spawn_count > 1), no
        dialog -> red.
      - drop the ``code == EXIT_STARTUP_FAILURE`` dialog branch in
        Watchdog.run's STOP_ALL arm -> give_ups stays empty -> red.
      - stop passing overlay_port into the Watchdog -> the message names
        'None' instead of the port -> the port-in-message assert red.
    """
    import threading
    from yamabiko.watchdog import Watchdog
    from yamabiko.lifecycle.exit_codes import EXIT_STARTUP_FAILURE

    stop_event = threading.Event()
    give_ups: list = []
    spawn_count = {"n": 0}

    class _FakeProcess:
        def __init__(self, code):
            self._code = code

        def wait(self, timeout=None):
            return self._code

        def terminate(self):
            pass

    def _spawn():
        spawn_count["n"] += 1
        return _FakeProcess(EXIT_STARTUP_FAILURE)

    wd = Watchdog(
        name="pipeline",
        spawn=_spawn,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        stop_event=stop_event,
        min_lifetime_s=0.0,
        on_give_up=lambda title, detail: give_ups.append((title, detail)),
        overlay_port=9765,
    )
    wd.run()

    assert spawn_count["n"] == 1, (
        f"exit 46 must be terminal (no respawn); spawn_count="
        f"{spawn_count['n']} -- 46 was routed to RESTART"
    )
    assert stop_event.is_set(), (
        "exit 46 must cascade (set the shared stop_event) so peers + mutex "
        "drop instead of lingering"
    )
    assert len(give_ups) == 1, (
        f"exit 46 must fire the give-up dialog exactly once; got {give_ups}"
    )
    _title, detail = give_ups[0]
    assert "9765" in detail, (
        f"the give-up dialog must name the configured overlay port so the "
        f"user knows which port to free/change; got {detail!r}"
    )


def test_build_watchdog_pipeline_exit_46_records_port_giveup_from_config(
    _watchdog_env, monkeypatch,
):
    """Through the production wiring (build_watchdog -> _record_error ->
    app.pending_errors): a pipeline exit 46 records a give-up dialog whose
    text names the overlay port taken FROM THE MERGED CONFIG. This proves
    build_watchdog threads overlay.websocket_port into the pipeline
    Watchdog's overlay_port -- without that wiring the dialog would say
    'None'.

    Seeded defect: stop passing overlay_port to the pipeline Watchdog in
    build_watchdog -> the recorded give-up says 'None' instead of 9911 ->
    red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STARTUP_FAILURE

    env = _watchdog_env
    wd_mod = env.module

    import yamabiko.config as _cfg_mod
    monkeypatch.setattr(
        _cfg_mod, "load_full_config",
        lambda: {"overlay": {"websocket_port": 9911}},
    )

    monkeypatch.setattr(
        wd_mod, "_real_spawn", lambda env=None: _FakeProc(exit_code=EXIT_STARTUP_FAILURE),
    )

    app = wd_mod.build_watchdog([])
    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    wd_pipeline._clock = lambda: 0.0
    wd_pipeline._min_lifetime_s = 0.0
    wd_pipeline._sleep = lambda s: None

    try:
        wd_pipeline.run()
        with app.pending_errors_lock:
            recorded = list(app.pending_errors)
        assert len(recorded) == 1, (
            f"exit 46 must record exactly one give-up error via the "
            f"production _record_error; got {[e.title for e in recorded]!r}"
        )
        assert "9911" in recorded[0].details, (
            f"the recorded port-bind give-up must name the CONFIGURED "
            f"overlay port (9911); got {recorded[0].details!r}"
        )
    finally:
        app.pipeline_writer.close()


def test_build_watchdog_pipeline_exit_47_restarts_within_min_lifetime_without_giveup(
    _watchdog_env, monkeypatch,
):
    """A pipeline exit with EXIT_CHILD_GIVEUP (47: the overlay supervisor
    exhausted its respawn budget) landing INSIDE min_lifetime must take
    the RESTART path -- a pipeline restart grants a fresh overlay budget --
    NOT the fast-crash 'configuration error' give-up. 47 fires only after
    the overlay spawned and crash-looped, so a quick 47 is a runtime
    flap-out, never a pipeline config error; it must be exempt from the
    fast-crash heuristic exactly like 44/45.

    Seeded defect (one line, yamabiko/watchdog.py): drop EXIT_CHILD_GIVEUP
    from the fast-crash exemption tuple ``code not in (EXIT_STREAM_DEATH,
    EXIT_STAGE_ESCALATION, EXIT_CHILD_GIVEUP)`` -> the first fast 47 gives
    up ('Yamabiko failed to start') and cascades -> no respawn ->
    stop_state_at_respawn == [] and pending_errors gains an entry -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_CHILD_GIVEUP

    env = _watchdog_env
    wd_mod = env.module

    app = wd_mod.build_watchdog([])
    wd_pipeline = app.watchdogs[0]
    assert wd_pipeline.name == "pipeline"
    assert wd_pipeline._min_lifetime_s > 0.0, (
        f"build_watchdog's pipeline min_lifetime_s is "
        f"{wd_pipeline._min_lifetime_s!r}; this test needs a positive "
        f"min-lifetime so the exit-47 lands inside the fast-crash window"
    )

    stop_state_at_respawn: list = []
    spawn_count = {"n": 0}

    def _spawn(env=None):
        spawn_count["n"] += 1
        if spawn_count["n"] == 1:
            return _FakeProc(exit_code=EXIT_CHILD_GIVEUP)
        stop_state_at_respawn.append(app.stop_event.is_set())
        app.stop_event.set()
        return _FakeProc(exit_code=0)

    monkeypatch.setattr(wd_mod, "_real_spawn", _spawn)

    # Clock frozen => elapsed == 0 < min_lifetime (30 s default): the 47
    # lands INSIDE the fast-crash window. Big restart budget; no real sleep.
    wd_pipeline._clock = lambda: 0.0
    wd_pipeline._max_restarts = 5
    wd_pipeline._sleep = lambda s: None

    try:
        wd_pipeline.run()
        assert stop_state_at_respawn == [False], (
            f"pipeline exit {EXIT_CHILD_GIVEUP} (overlay give-up) within "
            f"min_lifetime did not take the restart path: got "
            f"{stop_state_at_respawn} (expected [False]). [] means the "
            f"fast-crash config-error heuristic swallowed the give-up and "
            f"cascaded on the FIRST one; [True] means it cascaded to peers."
        )
        with app.pending_errors_lock:
            recorded = list(app.pending_errors)
        assert recorded == [], (
            f"a 47 within min_lifetime must NOT record a give-up error "
            f"(the misleading 'failed to start' dialog); got "
            f"{[e.title for e in recorded]!r}"
        )
    finally:
        app.pipeline_writer.close()


# ---------------------------------------------------------------------------
# run_watchdog driver tests
#
# run_watchdog had no test that actually invoked it.
# The surrounding wiring tests (build_watchdog_*) verified
# app.watchdogs was populated but never drove the driver. This is the
# wired-not-exercised pattern: wiring tested, behaviour not.
#
# The two tests below close that gap: one invokes run_watchdog and
# asserts all supervisors' .run() methods get called + the function
# returns 0; the other installs fake signal.signal and verifies the
# installed SIGINT handler sets app.stop_event.
# ---------------------------------------------------------------------------


def test_run_watchdog_starts_one_thread_per_watchdog_and_returns_zero(
    _watchdog_env, monkeypatch,
):
    """run_watchdog(app) must invoke Watchdog.run() for each
    supervisor in app.watchdogs (via one thread each), block until
    all complete, and return 0 on clean shutdown.

    Post-Task-8 sibling set is {pipeline, llama}. The health probe
    and stop-mirror helper threads are also started by run_watchdog
    but are daemons (the process won't block on them) and get covered
    by their own dedicated tests (tests/test_health_probe.py plus
    the stop-mirror test below).

    Without this test, a regression that dropped the `t.start()`
    loop (supervisors wired but never actually run) or the
    `t.join()` loop (function returns before supervisors exit)
    would ship silently.

    Seeded-defect checks:
      - Replace `for t in threads: t.start()` with `pass` -> no
        supervisor runs, `invoked` set stays empty, this test fails
        with "expected both watchdogs to run; got []".
      - Change `return 0` to `return 1` -> rc assertion fails.
    """
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    # Pre-set stop_event so each real Watchdog.run() would exit at
    # its while-loop top check anyway (prevents real subprocess
    # spawning). We also replace each .run() with a tracker so we
    # know the thread ran (a regression that never started threads
    # would leave `invoked` empty).
    app.stop_event.set()
    invoked: list[str] = []
    for wd in app.watchdogs:
        # Default-arg binds name at function-definition time so the
        # closure captures the current wd.name, not the final one.
        def _tracker(name=wd.name):
            invoked.append(name)
        wd.run = _tracker

    # Stub signal.signal: pytest runs in the main thread, and
    # installing real signal handlers interferes with pytest's
    # own teardown. We test the handler install contract in a
    # separate test below.
    import signal as _sig
    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    rc = wd_mod.run_watchdog(app)

    assert rc == 0, f"run_watchdog returned {rc}, expected 0"
    assert set(invoked) == {"pipeline", "llama"}, (
        f"expected both watchdogs (pipeline, llama) to have .run() "
        f"invoked by run_watchdog; got {sorted(invoked)}. "
        f"Either t.start() was dropped or threads never reached run()."
    )


def test_run_watchdog_reaps_self_shutdown_watcher_thread(
    _watchdog_env, monkeypatch,
):
    """run_watchdog must unblock and join its "watchdog-self-shutdown"
    watcher thread (parked forever on WaitForSingleObject) and close the
    self-shutdown event writer on exit. Without this, every
    run_watchdog-driving test leaks one parked thread + one kernel event
    handle per call.

    Fails red if: the finally block doesn't signal the event and join
    the watcher -- the thread set below keeps the new watcher alive
    after run_watchdog returns.

    Seeded-defect drill: remove the ``watchdog_self_writer.signal()`` /
    ``join`` reap in run_watchdog's finally -> the freshly-started
    watcher outlives the call -> red.
    """
    import threading as _threading
    import time as _time

    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])
    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    import signal as _sig
    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    def _watchers() -> set:
        return {
            t for t in _threading.enumerate()
            if t.name == "watchdog-self-shutdown" and t.is_alive()
        }

    before = _watchers()
    rc = wd_mod.run_watchdog(app)
    assert rc == 0

    # The watcher this call started must be gone (allow a brief join
    # window for scheduling); pre-existing leaked watchers from OTHER
    # tests are excluded via the before-set.
    deadline = _time.monotonic() + 2.0
    while _time.monotonic() < deadline and (_watchers() - before):
        _time.sleep(0.05)
    leaked = _watchers() - before
    assert not leaked, (
        f"run_watchdog leaked its self-shutdown watcher thread(s) "
        f"{leaked}; the finally block must SetEvent the self-shutdown "
        f"writer, join the watcher, and close the handle."
    )


def test_build_watchdog_each_supervisor_shares_the_same_job_object(_watchdog_env):
    """All sibling supervisors must be assigned the SAME Job Object so
    they all get killed together when the watchdog parent dies.

    If each supervisor got its own job, a partial parent death
    scenario (one supervisor thread crashes) would leave sibling
    supervisors' children alive. Also a Job Object per supervisor
    would mean more separate handles, more separate close-on-exit
    events -- more surface for leaks.

    Anti-zombie infrastructure. This test guards the wiring; the Job
    Object's actual kill-on-close behaviour is a Windows OS contract we
    trust, not a unit-testable Python path.

    Seeded-defect check: call `create_outer_job(...)` separately in
    each `Watchdog(..., job=job)` assignment -> each supervisor has
    a distinct job, identity check below fails.
    """
    wd_mod = _watchdog_env.module
    app = wd_mod.build_watchdog([])

    jobs = {id(w._job) for w in app.watchdogs}
    assert len(jobs) == 1, (
        f"all sibling supervisors must share one Job Object; got "
        f"{len(jobs)} distinct jobs. Each watchdog having its own job "
        f"means partial-death scenarios leak children."
    )
    # The fixture's fake records create_outer_job calls: exactly one job
    # must be created, and build_watchdog must self-assign the watchdog
    # process to it (assign(None) -> children auto-join via inheritance).
    # Seeded defect: delete the `job.assign(None)` call in build_watchdog
    # -> assigned stays empty -> red (children would never auto-join the
    # job and would survive watchdog death).
    assert len(_watchdog_env.created_jobs) == 1, (
        f"build_watchdog created {len(_watchdog_env.created_jobs)} outer "
        f"jobs; expected exactly 1 shared job."
    )
    assert _watchdog_env.created_jobs[0].assigned == [None], (
        "build_watchdog did not self-assign the watchdog process to the "
        "outer job (assign(None)); spawned children won't inherit job "
        "membership and will survive watchdog death."
    )


def test_watchdog_run_assigns_spawned_proc_to_job(monkeypatch):
    """Every subprocess.Popen the watchdog spawns must be assigned to
    its Job Object so the OS will force-kill it when the watchdog
    exits.

    The assignment call site is inside the LIVE Watchdog.run() loop.
    This test previously drove Watchdog._run_once -- a method with
    ZERO production callers -- so deleting `job.assign(proc)` from
    run() kept the whole suite green while every spawned child escaped
    KILL_ON_JOB_CLOSE (the zombie-llama bug class). A later refactor
    deleted _run_once and repointed
    this test at run() itself.

    Retires a dead-code anchor: _run_once had zero production callers.

    Seeded-defect check: remove the `self._job.assign(proc)` line from
    Watchdog.run -> fake_job.assigned stays empty, this test fails.
    """
    from yamabiko import watchdog as wd_mod

    class _FakeJob:
        def __init__(self):
            self.assigned: list = []
        def assign(self, proc):
            self.assigned.append(proc)

    # A real subprocess.Popen whose process exits 0 immediately:
    # classify(0) == STOP_SELF ends the run() loop after one iteration,
    # so run() returns without a restart. Using sys.executable with a
    # no-op one-liner mirrors test_real_watchdog_subprocess's sentinel.
    import subprocess, sys
    sentinel_cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]

    spawned: list = []

    def _real_spawn():
        proc = subprocess.Popen(
            sentinel_cmd, stdin=subprocess.DEVNULL,
        )
        spawned.append(proc)
        return proc

    fake_job = _FakeJob()
    wd = wd_mod.Watchdog(
        name="test",
        spawn=_real_spawn,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        job=fake_job,
    )
    wd.run()
    assert len(spawned) == 1, (
        f"expected exactly one spawn for a clean-exit child; got "
        f"{len(spawned)}"
    )
    assert fake_job.assigned == spawned, (
        f"Watchdog.run must call job.assign(proc) exactly once per "
        f"spawn, with the spawned Popen; got "
        f"{len(fake_job.assigned)} assignment(s). Without this, spawned "
        f"subprocesses escape the Job Object and survive parent death "
        f"(the zombie-llama bug class)."
    )
    # And the proc passed to assign must be the real Popen that was
    # waited on (run() guards with isinstance(proc, subprocess.Popen)).
    assert isinstance(fake_job.assigned[0], subprocess.Popen), (
        f"job.assign received a {type(fake_job.assigned[0]).__name__} "
        f"-- expected subprocess.Popen"
    )


def test_run_watchdog_installed_sigint_handler_sets_stop_event(
    _watchdog_env, monkeypatch,
):
    """The SIGINT handler run_watchdog installs must set
    app.stop_event so the supervisors drop. Without the handler or
    with a no-op handler, Ctrl+C can't propagate through the
    threaded supervisor.

    Captures the handler registered via signal.signal(SIGINT, ...),
    clears stop_event, invokes the captured handler with
    synthesised args, and asserts stop_event is set.

    Seeded-defect check: replace `app.stop_event.set()` inside
    _on_signal with `pass` -> invoking the handler doesn't set
    stop_event, this test fails. Or: drop the
    `signal.signal(SIGINT, _on_signal)` call entirely -> no SIGINT
    handler captured, this test fails at the `sigint_handlers`
    emptiness check.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    # Capture every signal.signal call without actually installing
    # anything (pytest's own signal handling stays intact).
    signal_calls: list = []
    monkeypatch.setattr(
        _sig, "signal",
        lambda signum, handler: signal_calls.append((signum, handler)),
    )

    # Short-circuit each supervisor's .run() and pre-set stop_event
    # so run_watchdog returns immediately after signal-handler
    # install.
    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    wd_mod.run_watchdog(app)

    sigint_handlers = [h for sig_, h in signal_calls if sig_ == _sig.SIGINT]
    assert sigint_handlers, (
        f"no SIGINT handler was installed by run_watchdog; "
        f"signal.signal calls captured: {signal_calls}. "
        f"Ctrl+C will be ignored at runtime."
    )

    # Invoke the installed handler. It must set stop_event so the
    # supervisors' run-loops see the request and exit.
    app.stop_event.clear()
    assert not app.stop_event.is_set(), "precondition: stop_event cleared"

    installed_sigint_handler = sigint_handlers[-1]
    installed_sigint_handler(_sig.SIGINT, None)

    assert app.stop_event.is_set(), (
        "SIGINT handler did not set app.stop_event when invoked; "
        "Ctrl+C won't propagate shutdown to supervisors"
    )


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "SIGBREAK"),
    reason="SIGBREAK is Windows-only",
)
def test_run_watchdog_installs_sigbreak_handler_on_windows(
    _watchdog_env, monkeypatch,
):
    """On Windows, CREATE_NEW_PROCESS_GROUP + CTRL_BREAK_EVENT delivers
    SIGBREAK, not SIGINT. Without an explicit signal.signal(SIGBREAK,
    _on_signal) registration, Python's default SIGBREAK handler merely
    raises KeyboardInterrupt on the main thread -- app.stop_event never
    gets set, the supervisor threads keep running, and the watchdog
    hangs well past the Tier A grace window. That's exactly what the
    frozen-E2E test_graceful_shutdown_via_named_event caught on the
    first real bundle run.

    Fails red if: run_watchdog doesn't pass ``signal.SIGBREAK`` to
    ``signal.signal(...)``, OR the handler it passes doesn't set
    app.stop_event when invoked. NOTE this test intercepts
    ``signal.signal`` via monkeypatch (pytest's own teardown would
    be disrupted by a real install), so it verifies the wiring
    contract, not OS-level registration. Real SIGBREAK delivery is
    covered indirectly by tests/test_real_frozen_e2e.py running
    against the frozen bundle.

    Seeded-defect drill: removing the ``signal.signal(signal.SIGBREAK,
    _on_signal)`` call from run_watchdog makes sigbreak_handlers empty,
    first assert goes red.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    signal_calls: list = []
    monkeypatch.setattr(
        _sig, "signal",
        lambda signum, handler: signal_calls.append((signum, handler)),
    )

    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    wd_mod.run_watchdog(app)

    sigbreak_handlers = [
        h for sig_, h in signal_calls if sig_ == _sig.SIGBREAK
    ]
    assert sigbreak_handlers, (
        f"no SIGBREAK handler installed; signal.signal captured: "
        f"{signal_calls}. Ctrl+Break / tray-Quit via CTRL_BREAK_EVENT "
        f"will not run the Tier A graceful path."
    )

    # Invoke the installed handler and verify it sets stop_event (it
    # should be the same _on_signal closure as the SIGINT handler).
    app.stop_event.clear()
    sigbreak_handlers[-1](_sig.SIGBREAK, None)
    assert app.stop_event.is_set(), (
        "SIGBREAK handler did not set stop_event; supervisors will hang"
    )


def test_watchdog_run_stop_all_fires_shared_stop_event_on_exit_43():
    """Fails red if: exit 43 doesn't set the shared stop_event,
    leaving llama (and the overlay under pipeline) alive after a
    tailored-error shutdown."""
    import threading
    from yamabiko.watchdog import Watchdog

    stop_event = threading.Event()

    class _FakeProcess:
        def __init__(self, code: int) -> None:
            self._code = code
            self._waited = False
        def wait(self, timeout=None):
            if not self._waited:
                self._waited = True
                return self._code
            return self._code
        def terminate(self):
            pass

    def _spawn():
        return _FakeProcess(code=43)

    wd = Watchdog(
        name="pipeline",
        spawn=_spawn,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        stop_event=stop_event,
        min_lifetime_s=0.0,
    )
    wd.run()
    assert stop_event.is_set(), "exit 43 must propagate to shared stop_event"


# ---------------------------------------------------------------------------
# HealthProbe + llama_stop_event wiring tests
#
# The two tests below close the wired-not-exercised gap for the
# wiring that run_watchdog introduces: a HealthProbe daemon thread
# and a stop_event -> llama_stop_event mirror thread. Without these,
# the wiring would sit inert and pass silent regression tests.
# ---------------------------------------------------------------------------


def test_run_watchdog_stop_event_mirror_forwards_to_llama_stop(
    _watchdog_env, monkeypatch,
):
    """Fails red if: the stop_event -> llama_stop_event mirror thread
    isn't started (or forwards in the wrong direction). Without this
    forward, a SIGINT / exit-42 / exit-43 shutdown would stop pipeline
    but leave the llama supervisor running because it listens on
    llama_stop_event, not the main stop_event.

    Seeded-defect checks:
      - Delete the _mirror thread start in run_watchdog -> setting
        stop_event doesn't propagate, llama_stop stays unset, wait
        times out and the assertion below fails.
      - Swap the forward direction (`stop_event.set()` inside
        _mirror, `llama_stop_event.wait()` outside) -> same failure
        mode: stop_event can be set without llama_stop reacting.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    # Stub httpx inside the probe so it doesn't hit the network.
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    # Replace supervisor .run() with a stop_event-aware block so the
    # supervisors don't exit until we set stop_event.
    for wd in app.watchdogs:
        wd.run = lambda _ev=app.stop_event: _ev.wait(timeout=5.0)

    # Kick off run_watchdog in a background thread so we can trigger
    # the mirror mid-run.
    import threading as _th
    t = _th.Thread(target=wd_mod.run_watchdog, args=(app,), daemon=True)
    t.start()

    # Precondition: llama_stop is not yet set.
    assert not app.llama_stop_event.is_set(), (
        "llama_stop_event already set before stop_event was triggered"
    )

    # Trigger shutdown via the main stop_event; the mirror should
    # forward to llama_stop_event.
    app.stop_event.set()

    assert app.llama_stop_event.wait(timeout=2.0), (
        "llama_stop_event was not set within 2s of stop_event.set(); "
        "the stop_event -> llama_stop_event mirror thread is absent "
        "or forwards in the wrong direction. A SIGINT would stop the "
        "pipeline supervisor but leave llama-server running."
    )
    t.join(timeout=5.0)


def test_run_watchdog_starts_health_probe_thread(
    _watchdog_env, monkeypatch,
):
    """Fails red if: run_watchdog doesn't start a HealthProbe thread.
    The HealthProbe is the primary exit-code-agnostic cascade trigger
    (spec C2 fix); without it, a crashed / hung pipeline that doesn't
    exit with a STOP_ALL sentinel leaves llama-server alive and holding
    4 GB of VRAM until the user force-quits.

    Asserts by observing that the probe's httpx.get is actually invoked
    while run_watchdog is running. This is a behavioural check (not just
    counting threads or checking a flag): we wait for the probe's side
    effect.

    Also captures the URL the probe polls so the caller can verify it
    derives from config (URL must not be hardcoded).

    Seeded-defect checks:
      - Delete the `threading.Thread(target=probe.run, ...).start()`
        call in run_watchdog -> httpx.get never called, wait times
        out, assertion fails.
      - Construct the HealthProbe but drop the .start() -> same.
      - Hardcode URL to "http://example.com/x" -> captured URL mismatch.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    # Record when httpx.get is called. The probe is our only caller
    # inside run_watchdog, so any call here means the probe ran.
    call_event = threading.Event()
    captured_urls: list = []

    class _RecordingHttpx:
        @staticmethod
        def get(url, timeout=2.0):
            captured_urls.append(url)
            call_event.set()
            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(wd_mod, "loopback_http", _RecordingHttpx())

    # Stub config loading so the test is deterministic regardless of
    # the dev-mode user_overrides.yaml on disk. Feed a custom port so
    # we can assert the probe URL actually reflects config (fix 2).
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {
            "overlay": {"websocket_port": 8765},
            "lifecycle": {"health": {"poll_interval_s": 0.01, "fail_budget_s": 120.0}},
        },
    )

    # Supervisors: block until we signal, so run_watchdog doesn't
    # return before the probe has had a chance to poll.
    for wd in app.watchdogs:
        wd.run = lambda _ev=app.stop_event: _ev.wait(timeout=5.0)

    import threading as _th
    t = _th.Thread(target=wd_mod.run_watchdog, args=(app,), daemon=True)
    t.start()

    assert call_event.wait(timeout=3.0), (
        "HealthProbe thread did not poll httpx.get within 3s; "
        "either the probe thread was not started, or it never "
        "reached its first iteration."
    )

    app.stop_event.set()
    t.join(timeout=5.0)

    # Fix 2 regression guard: URL derived from config, not hardcoded.
    # A hardcoded URL survives config drift but breaks the moment a
    # user moves overlay.websocket_port.
    assert captured_urls, "no URL captured from httpx.get"
    assert captured_urls[0] == "http://127.0.0.1:8765/livez", (
        f"probe URL does not match config-derived value; got "
        f"{captured_urls[0]!r}, expected 'http://127.0.0.1:8765/livez'. "
        f"If URL drifted from config, fix 2 (URL-from-config) regressed. "
        f"The wedge probe polls /livez (liveness), not /healthz."
    )


def test_run_watchdog_health_probe_uses_config_websocket_port(
    _watchdog_env, monkeypatch,
):
    """Fails red if: the HealthProbe URL is hardcoded to port 8765
    instead of deriving from cfg['overlay']['websocket_port'].

    Stubs load_full_config with a non-default port (9999). If the
    probe URL ends up as 8765 anyway, fix 2 is regressed -- the URL
    isn't actually being pulled from config.

    Paired with test_run_watchdog_starts_health_probe_thread (which
    asserts the probe runs at all). This one specifically asserts
    the URL tracks config drift.

    Seeded-defect check: revert `url=f"http://127.0.0.1:{port}/livez"`
    to the old hardcoded `"http://127.0.0.1:8765/livez"` -> the
    captured URL stays 8765 despite config saying 9999, assertion
    fails.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    captured_urls: list = []
    call_event = threading.Event()

    class _RecordingHttpx:
        @staticmethod
        def get(url, timeout=2.0):
            captured_urls.append(url)
            call_event.set()
            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(wd_mod, "loopback_http", _RecordingHttpx())
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"overlay": {"websocket_port": 9999}},
    )

    for wd in app.watchdogs:
        wd.run = lambda _ev=app.stop_event: _ev.wait(timeout=5.0)

    import threading as _th
    t = _th.Thread(target=wd_mod.run_watchdog, args=(app,), daemon=True)
    t.start()

    assert call_event.wait(timeout=3.0), "probe never polled"
    app.stop_event.set()
    t.join(timeout=5.0)

    assert captured_urls[0] == "http://127.0.0.1:9999/livez", (
        f"probe URL did not reflect cfg['overlay']['websocket_port']=9999; "
        f"got {captured_urls[0]!r}. Fix 2 (URL-from-config) regressed -- "
        f"URL is hardcoded or reading the wrong config key. "
        f"The wedge probe polls /livez (liveness), not /healthz."
    )


def test_run_watchdog_health_probe_uses_config_lifecycle_knobs(
    _watchdog_env, monkeypatch,
):
    """Fails red if: HealthProbe's poll_interval_s / fail_budget_s are
    hardcoded instead of deriving from cfg['lifecycle']['health'], OR
    the probe is not wired as the liveness wedge detector (polls /livez,
    on_wedged == the pipeline supervisor's request_restart).

    Captures the constructed HealthProbe by monkey-patching the class
    and reading the construction kwargs. This is the only way to
    observe the knobs/wiring without waiting long enough for the budget
    to expire behaviourally.

    Seeded-defect checks:
      - Drop the `poll_interval_s=float(...)` kwarg on the
        HealthProbe(...) construction -> probe falls back to its own
        default (5.0s) regardless of config -> knob assertion fails.
      - Wire `on_wedged` to anything other than the pipeline
        supervisor's request_restart (e.g. wd_llama.request_restart,
        or app.stop_event.set) -> the on_wedged identity assertion
        fails. Behaviourally that would restart the wrong process (or
        stop the watchdog) on a wedge.
      - Revert the URL to /healthz -> the /livez assertion fails;
        behaviourally /healthz folds translator_reachable into
        liveness, creating a self-kill feedback loop when the
        translator is temporarily unreachable.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    class _RecordingHttpx:
        @staticmethod
        def get(url, timeout=2.0):
            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(wd_mod, "loopback_http", _RecordingHttpx())

    # Capture HealthProbe construction kwargs.
    constructed: list = []
    original_cls = wd_mod.HealthProbe

    class _CapturingProbe(original_cls):
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(wd_mod, "HealthProbe", _CapturingProbe)
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {
            "overlay": {"websocket_port": 8765},
            "lifecycle": {"health": {"poll_interval_s": 7.5, "fail_budget_s": 45.0}},
        },
    )

    # Supervisors exit immediately so run_watchdog returns.
    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    wd_mod.run_watchdog(app)

    assert constructed, "HealthProbe was never constructed by run_watchdog"
    kwargs = constructed[0]
    assert kwargs["poll_interval_s"] == 7.5, (
        f"poll_interval_s did not reflect config value 7.5; got "
        f"{kwargs.get('poll_interval_s')!r}. Fix 3 regressed -- knob "
        f"is hardcoded or reading the wrong config key."
    )
    assert kwargs["fail_budget_s"] == 45.0, (
        f"fail_budget_s did not reflect config value 45.0; got "
        f"{kwargs.get('fail_budget_s')!r}. Fix 3 regressed."
    )
    # Liveness probe, not readiness. /healthz folds in
    # translator_reachable -> a llama outage would create a self-kill loop.
    assert kwargs["url"].endswith("/livez"), (
        f"probe URL must end with /livez (liveness); got "
        f"{kwargs.get('url')!r}. Polling /healthz creates a self-kill "
        f"feedback loop when the translator is temporarily unreachable."
    )
    # The wedge restarts the PIPELINE supervisor specifically. Identity
    # check, not just callable(): wiring on_wedged to wd_llama or to
    # app.stop_event.set would still be callable but restart the wrong
    # process / stop the watchdog.
    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    assert kwargs["on_wedged"] == wd_pipeline.request_restart, (
        "on_wedged must be the pipeline supervisor's request_restart so "
        "a wedge force-restarts the hung pipeline. It must NOT be wired "
        "to llama_stop_event."
    )
    assert "llama_stop_event" not in kwargs, (
        "HealthProbe must not take llama_stop_event; the liveness "
        "wedge detector never touches the llama-scoped event."
    )


def test_run_watchdog_health_probe_falls_back_to_documented_defaults_on_config_failure(
    _watchdog_env, monkeypatch,
):
    """Fails red if: run_watchdog's config-load-failure fallback drifts
    from the documented defaults (default_config.yaml lifecycle.health:
    poll 5.0s / budget 120.0s, overlay.websocket_port 8765).

    The sibling knob test above proves OVERRIDDEN values flow through;
    it can never see the default literals because it supplies every key.
    This test makes load_full_config RAISE (the cold-profile / corrupt-
    config path that run_watchdog explicitly catches) and pins what the
    probe gets in that case. Mutation testing showed all five fallback
    literals were previously unkillable by the suite.

    Seeded-defect checks (each one line):
      - fail_budget_s default 120.0 -> 121.0: budget assertion fails.
      - poll_interval_s default 5.0 -> 6.0: poll assertion fails.
      - websocket_port default 8765 -> 8766: URL assertion fails.
      - `cfg = {}` fallback -> `cfg = None`: AttributeError, test errors.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    constructed: list = []
    original_cls = wd_mod.HealthProbe

    class _CapturingProbe(original_cls):
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(wd_mod, "HealthProbe", _CapturingProbe)

    def _exploding_config_load():
        raise OSError("user_overrides.yaml unreadable (seeded by test)")

    monkeypatch.setattr(
        "yamabiko.config.load_full_config", _exploding_config_load,
    )

    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    wd_mod.run_watchdog(app)

    assert constructed, (
        "HealthProbe was never constructed -- the config-load-failure "
        "fallback must still arm the wedge detector, not skip it"
    )
    kwargs = constructed[0]
    assert kwargs["url"] == "http://127.0.0.1:8765/livez", (
        f"default probe URL drifted: {kwargs.get('url')!r}; the pipeline "
        f"serves 8765 when overlay.websocket_port is absent, so any other "
        f"port means the probe polls a dead socket and force-restarts a "
        f"healthy pipeline every fail_budget_s"
    )
    assert kwargs["poll_interval_s"] == 5.0, (
        f"default poll_interval_s drifted: {kwargs.get('poll_interval_s')!r}"
    )
    assert kwargs["fail_budget_s"] == 120.0, (
        f"default fail_budget_s drifted: {kwargs.get('fail_budget_s')!r}"
    )


def test_build_watchdog_pipeline_grace_defaults_to_10s_on_config_failure(
    _watchdog_env, monkeypatch,
):
    """Fails red if: build_watchdog's lifecycle.grace_s.pipeline fallback
    drifts from the documented 10.0s, or the llama supervisor loses its
    explicit 0.5s grace (llama-server has no graceful handshake).

    Same shape as the probe-defaults test: load_full_config raises, so
    the `_cfg = {}` fallback and the `grace_cfg.get("pipeline", 10.0)`
    default literal are the only path to the value. The grace window is
    the Tier A guarantee that the pipeline's finally (glossary save,
    uvicorn drain, CUDA close) gets to run before terminate(); a
    fat-fingered default ships silently otherwise (the bug class the
    module-defaults pin test closes for the supervision tuple).

    Seeded-defect checks (each one line):
      - grace default 10.0 -> 1.0 in build_watchdog: pipeline assert fails.
      - drop ``grace_s=0.5`` from the llama Watchdog: llama assert fails.
    """
    def _exploding_config_load():
        raise OSError("user_overrides.yaml unreadable (seeded by test)")

    monkeypatch.setattr(
        "yamabiko.config.load_full_config", _exploding_config_load,
    )

    env = _watchdog_env
    app = env.module.build_watchdog([])

    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    wd_llama = next(w for w in app.watchdogs if w.name == "llama")
    assert wd_pipeline._grace_s == 10.0, (
        f"pipeline grace_s fallback drifted: {wd_pipeline._grace_s!r} "
        f"(documented default is 10.0s -- lifecycle.grace_s.pipeline)"
    )
    assert wd_llama._grace_s == 0.5, (
        f"llama grace_s drifted: {wd_llama._grace_s!r}"
    )


class _FakeHttpxAlwaysOk:
    """Stand-in for the httpx module that always returns 200. Used by
    the mirror test where we don't want the probe to influence
    llama_stop_event (any probe-driven set would mask whether the
    mirror actually worked)."""

    @staticmethod
    def get(url, timeout=2.0):
        class _R:
            status_code = 200
        return _R()


# ---------------------------------------------------------------------------
# Pipeline ShutdownEventWriter wiring tests
#
# build_watchdog must create a ShutdownEventWriter scoped to the
# watchdog's pid ("pipeline" role), wrap _real_spawn in a spawn callable
# that injects YAMABIKO_SHUTDOWN_EVENT into the child's env, and stash
# the writer on WatchdogApp so run_watchdog's finally can signal+close
# it. Without this wiring the Tier A graceful-shutdown chain is broken:
# pipeline never receives the named-event ping, falls back to SIGINT-
# only termination, glossary save / uvicorn drain skipped.
#
# Two tests (structural + behavioural pairing):
#   1. Structural: the writer is created, stored, and has the expected
#      name format (Local\yamabiko.shutdown.pipeline.{pid}).
#   2. Behavioural: the pipeline spawn callable ACTUALLY invokes
#      _real_spawn with env[YAMABIKO_SHUTDOWN_EVENT] == writer.name.
#      This pairing ensures a structural test alone couldn't pass with
#      a wrapper that forgot to set the env var.
# ---------------------------------------------------------------------------


def test_build_watchdog_creates_pipeline_shutdown_writer(_watchdog_env):
    """Fails red if: build_watchdog doesn't create a ShutdownEventWriter
    for pipeline, leaving the Tier A graceful-shutdown chain broken.
    Without the writer on WatchdogApp, run_watchdog's finally can't
    signal the pipeline child; it falls back to SIGINT-only shutdown
    and the pipeline's finally-block cleanup (glossary save, uvicorn
    drain) is skipped.

    Asserts:
      - app.pipeline_writer is a real ShutdownEventWriter instance
        (not None, not a stub object).
      - Its name matches the documented format
        ``Local\\yamabiko.shutdown.pipeline.{pid}`` scoped to the
        watchdog's pid. Collision-safety invariant: two concurrent
        watchdogs on one box don't share an event.
      - On Windows, the handle field is non-None (CreateEventW
        succeeded). On non-Windows the handle is None by design and
        we don't assert on it.

    Seeded-defect checks:
      - Delete the ``pipeline_writer = ShutdownEventWriter.create(...)``
        line from build_watchdog -> AttributeError on
        ``app.pipeline_writer`` during dataclass construction.
      - Hardcode the role to "overlay" -> name check fails.
      - Pass ``parent_pid=0`` -> name check fails (scoped to wrong pid).
    """
    import os
    import sys

    from yamabiko.lifecycle.shutdown_event import ShutdownEventWriter

    wd_mod = _watchdog_env.module
    app = wd_mod.build_watchdog([])

    try:
        assert isinstance(app.pipeline_writer, ShutdownEventWriter), (
            f"app.pipeline_writer is not a ShutdownEventWriter; got "
            f"{type(app.pipeline_writer).__name__}. Tier A graceful "
            f"shutdown chain is broken."
        )
        expected_name = rf"Local\yamabiko.shutdown.pipeline.{os.getpid()}"
        assert app.pipeline_writer.name == expected_name, (
            f"pipeline writer name {app.pipeline_writer.name!r} does "
            f"not match expected {expected_name!r}; the event is "
            f"scoped to the wrong pid or the wrong role, and the "
            f"pipeline's listener will fail to OpenEvent it."
        )
        if sys.platform.startswith("win"):
            assert app.pipeline_writer.handle is not None, (
                "CreateEventW failed on Windows; ShutdownEventWriter "
                "has no handle and signal() is a no-op. Check for "
                "collision with a leaked event from a prior run."
            )
    finally:
        # Release the event handle so a subsequent test doesn't collide
        # on the same name (same pid within the pytest session).
        app.pipeline_writer.close()


def test_build_watchdog_pipeline_spawn_sets_env_var(_watchdog_env, monkeypatch):
    """Fails red if: the pipeline spawn wrapper doesn't pass
    YAMABIKO_SHUTDOWN_EVENT via env to _real_spawn -- the pipeline
    child would never install its ShutdownEventListener, and the
    watchdog's signal() calls would land on a handle nobody's
    listening on.

    Behavioural pairing for the structural test above: a writer
    can be created and stored on WatchdogApp while the spawn wrapper
    still fails to inject the env var. This test closes that gap by
    replacing ``_real_spawn`` with a stub that captures the ``env``
    kwarg, driving the pipeline supervisor's spawn callable, and
    asserting the captured env has the right key.

    Seeded-defect checks:
      - Delete ``env[ENV_VAR] = pipeline_writer.name`` inside
        _pipeline_spawn -> captured env lacks YAMABIKO_SHUTDOWN_EVENT,
        assertion fails.
      - Replace _pipeline_spawn with plain _real_spawn in the pipeline
        Watchdog(...) construction -> _real_spawn is called with
        env=None (default), same failure.
      - Typo the env var name (``env["YAMABIKO_SHUTDOWN"] = ...``) ->
        the ENV_VAR key is absent, assertion fails.
    """
    from yamabiko.lifecycle.shutdown_event import ENV_VAR

    wd_mod = _watchdog_env.module

    # Capture what _real_spawn is called with. The pipeline spawn
    # wrapper closes over _real_spawn in the module, so patching at
    # module level is what the wrapper sees.
    captured: list = []

    class _DummyProc:
        """Minimal subprocess.Popen stand-in. _pipeline_spawn returns
        whatever _real_spawn returns; the caller here only cares about
        the env dict, not the proc itself."""
        def wait(self, timeout=None):
            return 0
        def terminate(self):
            pass

    def _fake_real_spawn(env=None):
        captured.append(env)
        return _DummyProc()

    monkeypatch.setattr(wd_mod, "_real_spawn", _fake_real_spawn)

    app = wd_mod.build_watchdog([])
    try:
        wd_pipeline = next(
            w for w in app.watchdogs if w.name == "pipeline"
        )
        # Invoke the spawn callable directly. This is the whole point
        # of the behavioural pairing: not just "does a callable exist"
        # but "does calling it do the right thing."
        wd_pipeline._spawn()

        assert len(captured) == 1, (
            f"_pipeline_spawn must call _real_spawn exactly once; got "
            f"{len(captured)} calls. If 0, the wrapper short-circuited "
            f"without delegating; if >1, the wrapper loops."
        )
        env = captured[0]
        assert env is not None, (
            "_pipeline_spawn must pass env= to _real_spawn (not None); "
            "without it, the child inherits the bare environment and "
            "the shutdown-event name is lost."
        )
        assert ENV_VAR in env, (
            f"_pipeline_spawn did not set {ENV_VAR!r} in the spawn env; "
            f"env keys captured: {sorted(env.keys())[:10]}... The "
            f"pipeline child's ShutdownEventListener has no name to "
            f"OpenEvent, so watchdog.signal() lands on deaf ears."
        )
        assert env[ENV_VAR] == app.pipeline_writer.name, (
            f"env[{ENV_VAR}]={env[ENV_VAR]!r} does not match the "
            f"pipeline writer's name {app.pipeline_writer.name!r}. "
            f"Child will OpenEvent the wrong name and never hear the "
            f"signal."
        )
    finally:
        app.pipeline_writer.close()


def test_build_watchdog_relays_the_problem_carry_to_the_next_pipeline(_watchdog_env, monkeypatch):
    """Decision 18 wiring (TESTING.md 3.9): build_watchdog drops a carry left
    by an earlier run, the pipeline supervisor's on_restart hands a new
    carry to the next spawn's environment, and the relay knows the llama
    supervisor it restarts for a translator carry.

    Seeded defects: drop the start-up take_carry, the on_restart= argument,
    the carry_relay.apply(env) call or the carry_relay.llama assignment
    -> red."""
    from yamabiko import problems

    wd_mod = _watchdog_env.module
    captured: list = []

    class _DummyProc:
        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def _fake_real_spawn(env=None):
        captured.append(env)
        return _DummyProc()

    monkeypatch.setattr(wd_mod, "_real_spawn", _fake_real_spawn)
    problems.write_carry(("audio",), False)
    app = wd_mod.build_watchdog([])
    try:
        assert problems.take_carry() is None, "a fresh launch kept an earlier run's carry"
        wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
        wd_llama = next(w for w in app.watchdogs if w.name == "llama")
        problems.write_carry(("translator",), False)
        wd_pipeline._on_restart(45)
        wd_pipeline._spawn()
        assert captured[-1][problems.CARRY_ENV] == "translator"
        assert wd_pipeline._on_restart.__self__.llama is wd_llama
    finally:
        app.pipeline_writer.close()


def test_run_watchdog_signal_handler_signals_writer_before_stop_event(
    _watchdog_env, monkeypatch,
):
    """Fails red if: the SIGINT handler sets stop_event BEFORE pinging
    pipeline_writer, causing Watchdog._wait_or_stop's grace window to
    start against a pipeline that hasn't yet received the named-event
    shutdown signal -- pipeline hits terminate() before its asyncio
    listener wakes app.done, finally never runs. C1 regression from
    Test shape: replace app.pipeline_writer with a recording stand-in
    that remembers every call to signal() and stop_event.set() in a
    single ordered list. Capture the SIGINT handler installed by
    run_watchdog, invoke it, assert signal() came first.

    Seeded-defect checks:
      - Swap the order inside _on_signal so stop_event.set() runs
        first -> order[0] becomes "stop_event.set", assertion fails.
      - Delete the signal() call entirely from _on_signal -> "signal"
        is absent from order, assertion fails.
      - Move signal() back to the finally-block only (as it was before
        this fix) -> invoking the signal handler produces
        order=["stop_event.set"], signal never fires during handler
        execution, assertion fails.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    # Wrap stop_event.set to log ordering alongside signal().
    order: list[str] = []

    real_stop_set = app.stop_event.set

    def _tracking_stop_set() -> None:
        order.append("stop_event.set")
        real_stop_set()

    # Patch stop_event.set on the instance so the tracking version
    # is visible to _on_signal's closure (which reads app.stop_event).
    monkeypatch.setattr(app.stop_event, "set", _tracking_stop_set)

    class _RecordingWriter:
        def __init__(self, name):
            self.name = name
            self.handle = object()

        def signal(self):
            order.append("signal")
            return True

        def close(self):
            self.handle = None

    # Close real writer first so we don't leak; replace with recording.
    app.pipeline_writer.close()
    app.pipeline_writer = _RecordingWriter(name="test-recording")

    # Capture signal handlers without actually installing them.
    signal_calls: list = []
    monkeypatch.setattr(
        _sig, "signal",
        lambda signum, handler: signal_calls.append((signum, handler)),
    )
    # Probe pokes httpx; stub it so no network calls.
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    # Short-circuit supervisors + pre-set stop_event so run_watchdog
    # returns quickly after signal-handler install. (The signal
    # handler under test is installed synchronously during
    # run_watchdog; its ordering behaviour is orthogonal to the
    # supervisor loop.)
    real_stop_set()
    order.clear()  # discard the precondition set; we care about the
                   # handler's behaviour from this point on.
    for wd in app.watchdogs:
        wd.run = lambda: None

    wd_mod.run_watchdog(app)

    # Find the SIGINT handler that run_watchdog installed.
    sigint_handlers = [h for sig_, h in signal_calls if sig_ == _sig.SIGINT]
    assert sigint_handlers, (
        f"no SIGINT handler was installed by run_watchdog; "
        f"signal.signal calls captured: {signal_calls}"
    )
    installed = sigint_handlers[-1]

    # Clear order and invoke the handler. The handler is what we're
    # testing: does it ping the writer BEFORE setting stop_event?
    order.clear()
    installed(_sig.SIGINT, None)

    assert order, (
        "signal handler had no observable effect: neither "
        "pipeline_writer.signal() nor stop_event.set() was called."
    )
    assert order[0] == "signal", (
        f"C1 regression: signal handler set stop_event BEFORE pinging "
        f"pipeline_writer. Observed order: {order}. Expected: "
        f"['signal', 'stop_event.set']. Watchdog._wait_or_stop's grace "
        f"window now starts against a pipeline that hasn't yet seen "
        f"the named-event shutdown -- finally won't run inside the "
        f"window, pipeline gets terminate()d hard."
    )
    assert "stop_event.set" in order, (
        f"signal handler never set stop_event; observed order {order}. "
        f"Watchdog supervisors will never notice the shutdown request."
    )


def test_build_watchdog_pipeline_uses_config_grace_s(
    _watchdog_env, monkeypatch,
):
    """Fails red if: Watchdog's grace_s doesn't flow through from
    cfg['lifecycle']['grace_s']['pipeline']. A user override in
    user_overrides.yaml would have no effect; the grace_s.pipeline
    config key would have no verified consumer.

    Stubs load_full_config with a non-default grace value (7.5s)
    and asserts build_watchdog propagates it to the pipeline
    Watchdog's _grace_s field.

    Seeded-defect checks:
      - Drop the ``grace_s=pipeline_grace_s`` kwarg in the pipeline
        Watchdog(...) construction -> ctor default (5.0) is used,
        assertion fails (got 5.0, expected 7.5).
      - Revert the ``pipeline_grace_s = float(grace_cfg.get(
        "pipeline", 10.0))`` lookup so it reads from the wrong key
        (e.g. ``.get("overlay", ...)``) -> grace is 3.0 (overlay's
        stubbed value if present) or still default, assertion fails.
    """
    env = _watchdog_env
    wd_mod = env.module

    # Feed a deterministic cfg with a non-default pipeline grace.
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {
            "lifecycle": {"grace_s": {"pipeline": 7.5, "overlay": 2.0}},
        },
    )

    app = wd_mod.build_watchdog([])
    try:
        wd_pipeline = next(
            w for w in app.watchdogs if w.name == "pipeline"
        )
        assert wd_pipeline._grace_s == 7.5, (
            f"pipeline Watchdog._grace_s={wd_pipeline._grace_s!r}; "
            f"expected 7.5 from cfg['lifecycle']['grace_s']['pipeline']. "
            f"grace_s.pipeline key has no verified consumer; "
            f"user_overrides.yaml can't tune shutdown grace."
        )
    finally:
        app.pipeline_writer.close()


def test_run_pipeline_uses_config_overlay_grace_s(monkeypatch):
    """Fails red if: the overlay PipelineSupervisor's grace_s isn't
    read from APP.CFG's ['lifecycle']['grace_s']['overlay'] -- a user
    override (or an explicit --config file's lifecycle block) would
    have no effect; the overlay grace key would have no consumer.

    The lifecycle source of truth is app.cfg (loaded once by
    build_pipeline via _load_config(args.config)). run_pipeline used
    to call load_full_config() FRESH from disk here, which silently
    ignored an explicit --config file's lifecycle keys. This test
    feeds the grace value through fake_app.cfg and pins
    load_full_config to {} so a reintroduced disk reload yields the
    class default (3.0) and goes red.

    Construction-only assertion: capture the PipelineSupervisor(...)
    kwargs when run_pipeline reaches that line. We don't need to run
    the whole pipeline, just verify the grace kwarg is threaded
    through. Stubbing the class lets us capture kwargs without needing
    real subprocesses/job objects.

    Seeded-defect checks:
      - Drop the ``grace_s=_overlay_grace_s`` kwarg in the
        PipelineSupervisor(...) construction -> captured kwargs lack
        grace_s or use the class default 3.0, assertion fails if
        cfg says 2.5 and captured is 3.0.
      - Reintroduce the load_full_config() disk reload as the source
        -> it returns {} here -> grace falls back to 3.0 -> red.
    """
    import asyncio
    import pytest as _pytest

    # Tripwire: a reintroduced load_full_config() disk reload must NOT
    # be able to supply the expected value.
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {},
    )

    # Capture PipelineSupervisor constructor kwargs. We raise a
    # sentinel exception immediately after so run_pipeline stops
    # before reaching the heavy overlay-thread-start / uvicorn path.
    captured_kwargs: list = []

    class _Sentinel(Exception):
        pass

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod

    def _fake_supervisor(**kwargs):
        captured_kwargs.append(kwargs)
        raise _Sentinel("captured")

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _fake_supervisor)

    # Also stub out the inner_job creation + shutdown-event writer so
    # we don't allocate OS handles in a test that's about to raise.
    import yamabiko.lifecycle.jobs as _jobs_mod

    class _FakeJob:
        def assign(self, proc):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        _jobs_mod, "create_inner_job", lambda name: _FakeJob(),
    )
    import yamabiko.lifecycle.shutdown_event as _se_mod

    class _FakeWriter:
        def __init__(self, name):
            self.name = name
            self.handle = None

        def signal(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        _se_mod.ShutdownEventWriter, "create",
        classmethod(lambda cls, role, parent_pid: _FakeWriter(
            name=f"Local\\yamabiko.shutdown.{role}.{parent_pid}",
        )),
    )

    # Also stub from_env so the early ShutdownEventListener block doesn't
    # pick up ambient YAMABIKO_SHUTDOWN_EVENT_* from a parent process.
    # Accepts role kwarg after L3 tenant-scoping.
    monkeypatch.setattr(
        _se_mod.ShutdownEventListener, "from_env",
        classmethod(lambda cls, role="pipeline": None),
    )

    # Build a minimal PipelineApp stub with just what run_pipeline
    # touches before the PipelineSupervisor construction. The real
    # build_pipeline is heavy; we only need app.cfg, app.done, and
    # a few attributes referenced in the early block.
    import types
    from yamabiko import main as _main
    fake_app = types.SimpleNamespace(
        cfg={
            "hotkeys": {"enabled": False},
            # app.cfg is the lifecycle source of truth.
            "lifecycle": {"grace_s": {"pipeline": 7.5, "overlay": 2.5}},
        },
        done=asyncio.Event(),
        loop=None,  # run_pipeline calls asyncio.get_running_loop() inside
    )

    async def _drive() -> None:
        try:
            await _main.run_pipeline(fake_app)
        except _Sentinel:
            pass

    try:
        asyncio.run(_drive())
    except _Sentinel:
        pass
    # If we never got to PipelineSupervisor construction, the test
    # harness has drifted. Fail with a clear message rather than a
    # misleading empty-list error.
    if not captured_kwargs:
        _pytest.fail(
            "run_pipeline exited before reaching PipelineSupervisor "
            "construction. Test harness has drifted from the real "
            "run_pipeline body; update the stubs above."
        )

    kwargs = captured_kwargs[0]
    assert "grace_s" in kwargs, (
        f"PipelineSupervisor was constructed without a grace_s kwarg; "
        f"captured kwargs: {sorted(kwargs.keys())}. Overlay grace is "
        f"using the class default (3.0) regardless of config -- hard "
        f"rule #4 violated for lifecycle.grace_s.overlay."
    )
    assert kwargs["grace_s"] == 2.5, (
        f"PipelineSupervisor got grace_s={kwargs['grace_s']!r}; expected "
        f"2.5 from cfg['lifecycle']['grace_s']['overlay']. The key is "
        f"either hardcoded or reading the wrong config path."
    )


def test_run_pipeline_uses_config_overlay_anti_flap_knobs(monkeypatch):
    """Fails red if: the overlay PipelineSupervisor's max_restarts /
    restart_window_s kwargs aren't read from APP.CFG's
    ['lifecycle']['overlay'] -- a user override (or an explicit
    --config file's lifecycle block) would have no effect; these
    anti-flap config keys would have no verified consumer.

    Same app.cfg-sourcing contract as the grace_s test above --
    load_full_config is pinned to {} so a reintroduced disk reload
    yields the class defaults (3 / 60.0) and goes red.

    Seeded-defect drill: drop either kwarg from the PipelineSupervisor
    construction -> captured kwargs use the class defaults (3 / 60.0)
    instead of the stubbed 7 / 99.0, assertion fails.
    """
    import asyncio
    import pytest as _pytest

    # Tripwire: a reintroduced load_full_config() disk reload must NOT
    # be able to supply the expected values.
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {},
    )

    captured_kwargs: list = []

    class _Sentinel(Exception):
        pass

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod

    def _fake_supervisor(**kwargs):
        captured_kwargs.append(kwargs)
        raise _Sentinel("captured")

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _fake_supervisor)

    import yamabiko.lifecycle.jobs as _jobs_mod

    class _FakeJob:
        def assign(self, proc):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        _jobs_mod, "create_inner_job", lambda name: _FakeJob(),
    )
    import yamabiko.lifecycle.shutdown_event as _se_mod

    class _FakeWriter:
        def __init__(self, name):
            self.name = name
            self.handle = None

        def signal(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        _se_mod.ShutdownEventWriter, "create",
        classmethod(lambda cls, role, parent_pid: _FakeWriter(
            name=f"Local\\yamabiko.shutdown.{role}.{parent_pid}",
        )),
    )
    monkeypatch.setattr(
        _se_mod.ShutdownEventListener, "from_env",
        classmethod(lambda cls, role="pipeline": None),
    )

    import types
    from yamabiko import main as _main
    fake_app = types.SimpleNamespace(
        cfg={
            "hotkeys": {"enabled": False},
            # app.cfg is the lifecycle source of truth.
            "lifecycle": {
                "grace_s": {"pipeline": 10.0, "overlay": 3.0},
                "overlay": {"max_restarts": 7, "restart_window_s": 99.0},
            },
        },
        done=asyncio.Event(),
        loop=None,
    )

    async def _drive() -> None:
        try:
            await _main.run_pipeline(fake_app)
        except _Sentinel:
            pass

    try:
        asyncio.run(_drive())
    except _Sentinel:
        pass

    if not captured_kwargs:
        _pytest.fail(
            "run_pipeline exited before reaching PipelineSupervisor "
            "construction. Test harness has drifted from the real "
            "run_pipeline body; update the stubs above."
        )

    kwargs = captured_kwargs[0]
    assert kwargs.get("max_restarts") == 7, (
        f"PipelineSupervisor got max_restarts={kwargs.get('max_restarts')!r}; "
        f"expected 7 from cfg['lifecycle']['overlay']['max_restarts']. "
        f"user_overrides.yaml can't tune the overlay anti-flap cap "
        f"(key has no verified consumer)."
    )
    assert kwargs.get("restart_window_s") == 99.0, (
        f"PipelineSupervisor got restart_window_s={kwargs.get('restart_window_s')!r}; "
        f"expected 99.0 from cfg['lifecycle']['overlay']['restart_window_s']. "
        f"Key has no verified consumer."
    )


def test_run_pipeline_clamps_overlay_max_restarts_floor_to_one_with_warning(
    monkeypatch, caplog,
):
    """lifecycle.overlay.max_restarts below 1 is a user_overrides.yaml
    foot-gun that used to make the supervisor's cap check fire before
    the FIRST spawn -- no overlay ever appeared. The supervisor itself
    now guarantees the initial spawn (tier-1 guard), so this call-site
    clamp is belt-and-suspenders: run_pipeline must hand the supervisor
    max(1, value) and WARN about the raw value.

    Seeded-defect drill: revert main.py to
    ``max_restarts=int(_overlay_cfg.get("max_restarts", 3))`` (drop the
    clamp) -> captured max_restarts == 0 -> red; drop the warning ->
    the caplog assert goes red.
    """
    import asyncio
    import logging as _logging
    import pytest as _pytest

    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {},
    )

    captured_kwargs: list = []

    class _Sentinel(Exception):
        pass

    import yamabiko.lifecycle.pipeline_supervisor as _ps_mod

    def _fake_supervisor(**kwargs):
        captured_kwargs.append(kwargs)
        raise _Sentinel("captured")

    monkeypatch.setattr(_ps_mod, "PipelineSupervisor", _fake_supervisor)

    import yamabiko.lifecycle.jobs as _jobs_mod

    class _FakeJob:
        def assign(self, proc):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        _jobs_mod, "create_inner_job", lambda name: _FakeJob(),
    )
    import yamabiko.lifecycle.shutdown_event as _se_mod

    class _FakeWriter:
        def __init__(self, name):
            self.name = name
            self.handle = None

        def signal(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        _se_mod.ShutdownEventWriter, "create",
        classmethod(lambda cls, role, parent_pid: _FakeWriter(
            name=f"Local\\yamabiko.shutdown.{role}.{parent_pid}",
        )),
    )
    monkeypatch.setattr(
        _se_mod.ShutdownEventListener, "from_env",
        classmethod(lambda cls, role="pipeline": None),
    )

    import types
    from yamabiko import main as _main
    fake_app = types.SimpleNamespace(
        cfg={
            "hotkeys": {"enabled": False},
            "lifecycle": {"overlay": {"max_restarts": 0}},
        },
        done=asyncio.Event(),
        loop=None,
    )

    async def _drive() -> None:
        try:
            await _main.run_pipeline(fake_app)
        except _Sentinel:
            pass

    with caplog.at_level(_logging.WARNING, logger="yamabiko"):
        try:
            asyncio.run(_drive())
        except _Sentinel:
            pass

    if not captured_kwargs:
        _pytest.fail(
            "run_pipeline exited before reaching PipelineSupervisor "
            "construction. Test harness has drifted from the real "
            "run_pipeline body; update the stubs above."
        )

    kwargs = captured_kwargs[0]
    assert kwargs.get("max_restarts") == 1, (
        f"PipelineSupervisor got max_restarts={kwargs.get('max_restarts')!r} "
        f"for a configured 0; the call site must clamp to a floor of 1 "
        f"(0 used to mean 'never spawn the overlay at all')"
    )
    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno >= _logging.WARNING and "max_restarts" in r.getMessage()
    ]
    assert warnings, (
        "no WARNING naming max_restarts was logged; the clamp must not "
        "be silent (the user's override is being overridden)"
    )


def test_run_watchdog_signals_pipeline_writer_on_exit(
    _watchdog_env, monkeypatch,
):
    """Fails red if: run_watchdog's finally block doesn't signal
    app.pipeline_writer on exit. Without it, the pipeline child never
    receives the Tier A graceful-shutdown ping, falls back to SIGINT-
    only termination (if even that), and skips its finally-block
    cleanup (glossary save, uvicorn drain).

    Replaces the real ShutdownEventWriter on app with a recording
    stand-in so we observe signal() + close() without needing a real
    Win32 handle. Drives run_watchdog end-to-end with short-circuited
    supervisor .run() methods.

    Seeded-defect checks:
      - Delete the ``finally:`` block in run_watchdog -> signal never
        called, signalled stays False, assertion fails.
      - Call ``app.pipeline_writer.close()`` without first calling
        ``.signal()`` -> signalled stays False (close_only), assertion
        fails with "pipeline_writer.signal() not called".
      - Swap the order to close-before-signal -> close() returns early
        (handle is None), then signal() becomes a no-op too; both
        paths fail.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    class _RecordingWriter:
        def __init__(self, name):
            self.name = name
            self.handle = object()  # truthy sentinel for close() to run
            self.signalled = False
            self.closed = False
            self.order: list = []

        def signal(self):
            self.signalled = True
            self.order.append("signal")
            return True

        def close(self):
            self.closed = True
            self.order.append("close")
            self.handle = None

    # Close the real writer first so we don't leak the OS handle.
    real_writer_name = app.pipeline_writer.name
    app.pipeline_writer.close()
    recording = _RecordingWriter(name=real_writer_name)
    app.pipeline_writer = recording

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    # Short-circuit supervisors + pre-set stop_event so the join
    # returns immediately.
    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    rc = wd_mod.run_watchdog(app)

    assert rc == 0, f"run_watchdog returned {rc}; expected 0"
    assert recording.signalled, (
        "pipeline_writer.signal() was not called by run_watchdog's "
        "finally block. Tier A graceful-shutdown chain is broken: "
        "the pipeline child never receives the named-event ping."
    )
    assert recording.closed, (
        "pipeline_writer.close() was not called; the Win32 event "
        "handle leaks across watchdog restarts."
    )
    assert recording.order == ["signal", "close"], (
        f"signal/close order must be signal-then-close; got "
        f"{recording.order}. Closing before signalling makes the "
        f"signal call a silent no-op (handle is already None)."
    )


def test_stop_all_signals_pipeline_writer_before_setting_stop_event(
    _watchdog_env, monkeypatch,
):
    """A terminal supervisor outcome (llama give-up, exit 43) calls the
    wired on_stop_all. It must ping the pipeline's named shutdown event
    BEFORE setting stop_event, or the pipeline skips its graceful finally
    (glossary save) and is terminated at the end of the grace window.

    Seeded defect: drop ``pipeline_writer.signal()`` from _stop_all ->
    the recorder never sees a signal.
    """
    wd_mod = _watchdog_env.module
    seen: list = []

    class _RecordingWriter:
        def __init__(self, name):
            self.name = name
            self.handle = None
            self.stop_event = None

        def signal(self):
            seen.append(("signal", self.stop_event.is_set()))
            return True

        def close(self):
            pass

    writers: list = []

    def _create(role, parent_pid):
        writer = _RecordingWriter(f"{role}.{parent_pid}")
        writers.append(writer)
        return writer

    monkeypatch.setattr(wd_mod.ShutdownEventWriter, "create", staticmethod(_create))
    app = wd_mod.build_watchdog([])
    assert app.pipeline_writer is writers[0]
    writers[0].stop_event = app.stop_event
    by_name = {w.name: w for w in app.watchdogs}

    by_name["llama"]._on_stop_all()

    assert seen == [("signal", False)], (
        f"_stop_all must signal the pipeline writer before stop_event; "
        f"saw {seen}"
    )
    assert app.stop_event.is_set()


# ---------------------------------------------------------------------------
# Generalised orphan sweep
# ---------------------------------------------------------------------------
#
# Runs at the @pytest.mark.real tier: spawns a real subprocess with a
# cmdline the sweep's matcher must recognise, then calls the actual
# ``_sweep_orphaned_children`` (no mocks) and asserts the subprocess
# was killed. This is the complement to the unit-tier tests above
# which all inject fake ``spawn`` callables.


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="_sweep_orphaned_children is Windows-only (no-ops elsewhere)",
)
def test_sweep_suppresses_console_allocation(monkeypatch):
    """Every subprocess.run call in the sweep must pass
    CREATE_NO_WINDOW -- on a windowed frozen bundle (``console=False``)
    a missing flag makes Windows allocate a visible console for each
    console-subsystem child, flashing N windows on every startup.

    The kill path spawns NO subprocess (TerminateProcess through a
    verified handle replaced ``taskkill``), so the assertions are:
    (a) every spawned subprocess (the PowerShell enumerations) has
    CREATE_NO_WINDOW, and (b) the kill path stays subprocess-free -- a
    reintroduced ``taskkill`` would both flash a console AND re-open the
    PID-reuse race.

    Seeded-defect drill (pre-commit verified): drop
    ``creationflags=subprocess.CREATE_NO_WINDOW`` from the
    ``subprocess.run(["powershell", ...])`` call in
    ``_sweep_orphaned_children`` -> the flag assertion goes red.
    """
    import subprocess as _subprocess
    from pathlib import Path

    from yamabiko import watchdog as wd

    scope = (
        str(Path(wd.__file__).resolve().parent.parent)
        .replace("\\", "/")
        .lower()
    )
    needle = wd.YAMABIKO_CHILD_MATCH_SPECS[0][1] or ""
    fake_stdout = f"99999|{scope}/child.exe|cmd {needle}\n"

    calls: list[dict] = []

    def tracking_run(*args, **kwargs):
        calls.append({"argv": args[0] if args else kwargs.get("args"), "kwargs": kwargs})
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = fake_stdout
        return mock

    monkeypatch.setattr(wd.subprocess, "run", tracking_run)
    # The synthetic orphan pid (99999) must reach the kill seam -- and
    # be handled there, NOT via a spawned process.
    kill_calls: list[tuple] = []
    monkeypatch.setattr(
        wd, "_kill_orphan_verified",
        lambda pid, scope_arg: kill_calls.append((pid, scope_arg)) or True,
    )

    wd._sweep_orphaned_children()

    assert calls, "sweep made no subprocess.run calls; test premise broken"
    powershell_calls = [c for c in calls if c["argv"] and c["argv"][0] == "powershell"]
    assert powershell_calls, "sweep skipped PowerShell enumeration; matcher broken"
    assert kill_calls == [(99999, scope)], (
        f"the synthetic orphan never reached the verified-kill seam "
        f"(got {kill_calls}); the test can't vouch for the kill path"
    )
    non_powershell = [c for c in calls if c["argv"] and c["argv"][0] != "powershell"]
    assert not non_powershell, (
        f"sweep spawned non-PowerShell subprocesses {non_powershell}; "
        f"the kill path must use TerminateProcess on a verified "
        f"handle, never a taskkill child (console flash + PID-reuse race)"
    )
    for call in calls:
        argv0 = (call["argv"] or ["<empty>"])[0]
        assert call["kwargs"].get("creationflags") == _subprocess.CREATE_NO_WINDOW, (
            f"subprocess.run({argv0}...) missing "
            f"creationflags=CREATE_NO_WINDOW; would flash a console "
            f"window on every frozen-exe startup. kwargs={call['kwargs']}"
        )


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="_sweep_orphaned_children is Windows-only (no-ops elsewhere)",
)
def test_sweep_filters_self_scope_needle_and_counts_kills(monkeypatch):
    """Drives _sweep_orphaned_children's matcher/filter matrix in one
    synthetic enumeration (mutation testing revealed low coverage in this
    section -- the surviving half was exactly this edge/branch logic):

      kill:  an in-scope llama-server (needle=None spec), and an
             in-scope python.exe whose CommandLine carries the
             yamabiko.main needle;
      skip:  the current pytest process's own pid (self-exclusion --
             a watchdog that just ran `import yamabiko.main` in-process
             must never kill itself),
             an out-of-scope llama-server (ExecutablePath outside the
             project tree -- someone ELSE's llama),
             an in-scope python.exe whose cmdline lacks any yamabiko
             needle (an unrelated dev script in the venv),
             a malformed enumeration line (no pipe separators).

    Asserts on the OBSERVABLE outcomes: which (pid, scope) pairs the
    the verified-kill seam received, and the returned kill count --
    not on internals. (The kill mechanism is now TerminateProcess on a
    verified handle, covered by the test_kill_orphan_verified_*
    real-tier tests.)

    Seeded-defect drill: invert the scope filter
    (``if scope not in exe...`` -> ``if scope in exe...``) -> the
    out-of-scope llama gets killed and the in-scope targets survive ->
    the killed-pid set assertion goes red.
    """
    import os as _os
    import re as _re
    import subprocess as _subprocess
    from pathlib import Path

    from yamabiko import watchdog as wd

    scope = (
        str(Path(wd.__file__).resolve().parent.parent)
        .replace("\\", "/")
        .lower()
    )
    self_pid = _os.getpid()

    rows_by_image = {
        "llama-server.exe": [
            f"101|{scope}/third_party/llama.cpp/llama-server.exe|llama-server --port 8090",
            f"102|c:/somewhere/else/llama-server.exe|llama-server --port 8090",
            f"{self_pid}|{scope}/llama-server.exe|llama-server",
            "garbage line with no separators",
        ],
        "python.exe": [
            f"103|{scope}/.venv/scripts/python.exe|python -m yamabiko.main",
            f"104|{scope}/.venv/scripts/python.exe|python -m totally_unrelated_tool",
        ],
    }

    kill_calls: list[tuple] = []

    def tracking_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        if argv and argv[0] == "powershell":
            cmd_text = argv[-1]
            m = _re.search(r"Name='([^']+)'", cmd_text)
            image = m.group(1) if m else ""
            mock.stdout = "\n".join(rows_by_image.get(image, [])) + "\n"
        return mock

    monkeypatch.setattr(wd.subprocess, "run", tracking_run)
    monkeypatch.setattr(
        wd, "_kill_orphan_verified",
        lambda pid, scope_arg: kill_calls.append((pid, scope_arg)) or True,
    )

    killed = wd._sweep_orphaned_children()

    killed_pids = {pid for pid, _ in kill_calls}
    assert killed_pids == {101, 103}, (
        f"sweep killed pids {sorted(killed_pids)}; expected exactly "
        f"{{101, 103}}. 102=out-of-scope (foreign llama), "
        f"{self_pid}=self, 104=needle mismatch (unrelated venv script) "
        f"-- one of the filters regressed."
    )
    for pid, scope_arg in kill_calls:
        # The verified-kill helper must receive the SAME scope the
        # enumeration filtered by, so its kill-time re-check is anchored
        # to the same tree.
        assert scope_arg == scope, (
            f"_kill_orphan_verified(pid={pid}) got scope {scope_arg!r}; "
            f"expected the sweep scope {scope!r} -- the kill-time "
            f"re-verification would check the wrong tree."
        )
    assert killed == 2, (
        f"_sweep_orphaned_children returned killed={killed}; expected 2 "
        f"(the return-count accounting drifted from the verified kills)."
    )


@pytest.mark.real
@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="_sweep_orphaned_children is Windows-only (no-ops elsewhere)",
)
def test_sweep_orphaned_children_kills_overlay_orphan(tmp_path):
    """Sweeping overlay-cmdline subprocess terminates it.

    Fails red if the sweep extension misses overlay children:
    specifically if ``YAMABIKO_CHILD_MATCH_SPECS`` drops the
    ``("python.exe", "yamabiko.overlay_app")`` entry, or if the
    needle match is broken (e.g. changed to exact-equal), or if the
    taskkill call is skipped. Also fails red if the sweep no longer
    path-scopes correctly and the sentinel's venv-python ExecutablePath
    stops matching.

    Uses a subprocess whose CommandLine contains the literal string
    ``yamabiko.overlay_app`` (passed as an argv token after a
    ``-c "import time; time.sleep(N)"`` body) so Win32_Process's
    CommandLine field carries the needle the sweep looks for. The
    subprocess doesn't actually import ``yamabiko.overlay_app`` (that
    would pull Qt + the whole desktop stack); it just sleeps, which is
    sufficient to reach WMI enumeration before being killed.

    Seeded-defect check: delete the overlay_app match_spec entry from
    ``YAMABIKO_CHILD_MATCH_SPECS`` -> the sentinel survives the
    ``proc.wait(timeout=...)`` deadline below, test fails.

    Implicitly also covers the pipeline branch: the needle matcher
    uses the same code path for ``yamabiko.main`` and
    ``yamabiko.overlay_app``; killing a defective overlay spec would
    mean the pipeline spec must be killed separately, which is
    caught by code review.
    """
    import os
    import subprocess as _subprocess
    import time
    from pathlib import Path

    from yamabiko.watchdog import _sweep_orphaned_children

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # The sweep's scope filter (watchdog.py:905) requires the child's
    # ExecutablePath to be inside the repo tree. On a dev machine with
    # an in-tree venv (`python -m venv .venv` at repo root) this is
    # true by design; on CI runners using actions/setup-python the
    # interpreter is at C:\hostedtoolcache\windows\Python\... , well
    # outside the cloned repo. In that case the sweep correctly refuses
    # to touch the sentinel (kills only yamabiko-scoped processes),
    # and this test has no kill path to observe. Skip with an
    # actionable reason rather than misdiagnosing the sweep.
    repo_root_posix = Path(repo_root).resolve().as_posix().lower()
    exe_posix = Path(sys.executable).resolve().as_posix().lower()
    if not exe_posix.startswith(repo_root_posix):
        pytest.skip(
            f"sys.executable ({sys.executable}) is outside the repo "
            f"tree ({repo_root}); the sweep's scope filter correctly "
            f"rejects it, so there's no observable kill for this test "
            f"to assert on. Run with an in-repo venv "
            f"(`python -m venv .venv && .venv/Scripts/python -m pytest...`) "
            f"to exercise the kill path, or spawn the frozen yamabiko.exe "
            f"bundle (coverage gap tracked for frozen-bundle E2E)."
        )

    # Sentinel subprocess: sleeps 30 s so there's plenty of time for
    # WMI to enumerate it before the sweep kills it, and a natural
    # upper bound if the sweep unexpectedly no-ops (the test's wait
    # timeout is much shorter, so we still fail fast).
    #
    # The token "yamabiko.overlay_app" after the -c script body
    # appears in the child's CommandLine (Win32_Process exposes the
    # full argv-joined string). The script body itself ignores the
    # extra arg.
    proc = _subprocess.Popen(
        [
            sys.executable,
            "-c", "import time; time.sleep(30)",
            "yamabiko.overlay_app",
        ],
        cwd=repo_root,
        stdin=_subprocess.DEVNULL,
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.DEVNULL,
    )
    try:
        # Give the OS a moment to register the new process so WMI sees
        # it. 300 ms is empirically enough on Win11; if flakes appear
        # bump to 500 ms and leave a note.
        time.sleep(0.3)

        # Sanity: the sentinel is still running at the moment of the
        # sweep. If this fails, something other than our sweep killed
        # it and the test result is meaningless.
        assert proc.poll() is None, (
            f"sentinel exited with rc={proc.returncode} BEFORE the sweep "
            f"ran; test can't distinguish sweep-kill from premature exit."
        )

        killed = _sweep_orphaned_children()

        # Observable effect: the sentinel must actually terminate.
        # Use a bounded wait so a broken matcher surfaces as test
        # failure, not hang.
        try:
            rc = proc.wait(timeout=5.0)
        except _subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(
                "sentinel subprocess (cmdline contains 'yamabiko.overlay_app', "
                "ExecutablePath inside project tree) was NOT killed by "
                "_sweep_orphaned_children within 5 s. Sweep's matcher "
                "missed the overlay/pipeline branch -- check "
                "YAMABIKO_CHILD_MATCH_SPECS and the needle-in-cmdline "
                "check in _sweep_orphaned_children."
            )

        # taskkill /F on Windows results in STATUS_CONTROL_C_EXIT
        # (0xC000013A, which Popen.returncode reports as the UNSIGNED
        # 3221225786 -- not 1073741510, the magnitude of its signed twin,
        # as this comment previously claimed) or rc=1; never 0. Guard
        # against a broken test where the sentinel exited on its own
        # (would indicate sleep(30) was skipped, which shouldn't happen).
        assert rc != 0, (
            f"sentinel exited rc=0 after sweep; that would mean the "
            f"sleep(30) loop completed normally -- but that takes 30 s "
            f"and the test ran faster. Something else killed it."
        )

        # killed count should include our sentinel. Allow >= 1 rather
        # than == 1 because a prior flaky test may have left another
        # orphan the sweep rightly kills in the same call.
        assert killed >= 1, (
            f"_sweep_orphaned_children returned killed={killed}; expected "
            f">= 1 because our sentinel matched the overlay branch. "
            f"The return-count accounting is broken even if taskkill ran."
        )
    finally:
        # Defensive cleanup: if any branch above left proc alive
        # (e.g. assertion raised before wait), taskkill it here so
        # we don't poison the conftest baseline for subsequent tests.
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# build_watchdog translation.endpoint handling
#
# An external (non-loopback) endpoint means the user translates elsewhere
# -- spawning the bundled llama-server would burn ~4 GB VRAM for a server
# nothing will ever call. A loopback endpoint on a non-default port must
# move the bundled server to that port so producer (llama) and consumer
# (pipeline) agree.
# ---------------------------------------------------------------------------


def _patch_cfg_endpoint(monkeypatch, endpoint: str) -> None:
    """Route build_watchdog's function-local ``load_full_config`` import
    to a fixed config carrying ``translation.endpoint``."""
    import yamabiko.config as cfg_mod
    monkeypatch.setattr(
        cfg_mod, "load_full_config",
        lambda *a, **kw: {"translation": {"endpoint": endpoint}},
    )


def test_build_watchdog_skips_llama_supervisor_for_external_endpoint(
    _watchdog_env, monkeypatch,
):
    """With a non-loopback ``translation.endpoint`` (the user runs their
    own server elsewhere), build_watchdog must NOT construct the llama
    supervisor at all -- otherwise the bundled llama-server loads ~4 GB
    of VRAM that the pipeline (which honours the endpoint) never uses.

    Seeded-defect drill: ignore the endpoint in build_watchdog (always
    construct wd_llama) -> the name-set assertion goes red.
    """
    env = _watchdog_env
    wd_mod = env.module
    _patch_cfg_endpoint(
        monkeypatch, "http://192.168.1.10:8090/v1/chat/completions",
    )

    app = wd_mod.build_watchdog([])

    names = {wd.name for wd in app.watchdogs}
    assert names == {"pipeline"}, (
        f"build_watchdog built supervisors {sorted(names)} despite an "
        f"external translation.endpoint; the bundled llama-server would "
        f"idle-load ~4 GB VRAM for a server nothing calls."
    )


def test_build_watchdog_moves_llama_to_loopback_endpoint_port(
    _watchdog_env, monkeypatch,
):
    """A LOOPBACK endpoint on a non-default port keeps the bundled
    llama-server (the user still wants local translation) but must
    hand that port to the spawn factory, so server and pipeline meet
    on the same port.

    Fails red if: build_watchdog keeps constructing the factory without
    a port (pinned 8090) while the pipeline dials 9000.

    Seeded-defect drill: drop ``port=llama_port`` from the
    _make_real_spawn_llama call in build_watchdog -> recorded kwargs
    lack the port -> red.
    """
    env = _watchdog_env
    wd_mod = env.module
    _patch_cfg_endpoint(
        monkeypatch, "http://localhost:9000/v1/chat/completions",
    )

    recorded: list[dict] = []

    def _recording_factory(stop_event, **kwargs):
        recorded.append(kwargs)
        return lambda: (_ for _ in ()).throw(
            AssertionError("spawn must not run during build"),
        )

    monkeypatch.setattr(wd_mod, "_make_real_spawn_llama", _recording_factory)

    app = wd_mod.build_watchdog([])

    names = {wd.name for wd in app.watchdogs}
    assert names == {"pipeline", "llama"}, (
        f"a LOOPBACK endpoint must keep the bundled llama-server; got "
        f"supervisors {sorted(names)}"
    )
    assert recorded and recorded[0].get("port") == 9000, (
        f"_make_real_spawn_llama received kwargs {recorded}; expected "
        f"port=9000 from the loopback endpoint so llama-server binds "
        f"where the pipeline connects."
    )


def test_build_watchdog_keeps_llama_on_default_loopback_endpoint(
    _watchdog_env, monkeypatch,
):
    """The shipped default endpoint (http://localhost:8090/...) must keep
    today's behaviour -- llama supervisor present, default port handed to
    the factory.

    Fails red if: the endpoint plan misclassifies loopback as external
    (llama vanishes from a default install -- translation dead).
    """
    env = _watchdog_env
    wd_mod = env.module
    _patch_cfg_endpoint(
        monkeypatch, "http://localhost:8090/v1/chat/completions",
    )

    recorded: list[dict] = []

    def _recording_factory(stop_event, **kwargs):
        recorded.append(kwargs)
        return lambda: (_ for _ in ()).throw(
            AssertionError("spawn must not run during build"),
        )

    monkeypatch.setattr(wd_mod, "_make_real_spawn_llama", _recording_factory)

    app = wd_mod.build_watchdog([])

    names = {wd.name for wd in app.watchdogs}
    assert names == {"pipeline", "llama"}, (
        f"the DEFAULT loopback endpoint must keep the bundled "
        f"llama-server; got {sorted(names)}"
    )
    assert recorded and recorded[0].get("port") == 8090, (
        f"_make_real_spawn_llama received kwargs {recorded}; expected "
        f"port=8090 for the default endpoint"
    )


# ---------------------------------------------------------------------------
# Orphan kill must be PID-reuse safe: open a handle, RE-VERIFY the
# image path against the scope on that pinned handle, and terminate
# through the SAME handle. A raw `taskkill /PID <stale pid>` can kill an
# unrelated process that inherited the recycled PID between the WMI
# enumeration snapshot and the kill.
# ---------------------------------------------------------------------------


@pytest.mark.real
@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="_kill_orphan_verified is Windows-only",
)
def test_kill_orphan_verified_terminates_in_scope_child():
    """Happy path: a live child whose executable sits inside the scope
    is killed through the verified handle and the helper reports True.

    Fails red if: the helper never opens/terminates (child survives),
    or reports False for a genuine in-scope kill (sweep counter would
    silently under-report).
    """
    import subprocess as _sp
    import time as _time
    from pathlib import Path

    from yamabiko import watchdog as wd

    proc = _sp.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=_sp.CREATE_NO_WINDOW,
    )
    try:
        scope = str(Path(sys.executable).parent).replace("\\", "/").lower()
        assert wd._kill_orphan_verified(proc.pid, scope) is True
        rc = proc.wait(timeout=5.0)
        assert rc != 0, (
            f"child exited rc={rc}; TerminateProcess should produce a "
            f"nonzero exit code"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


@pytest.mark.real
@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="_kill_orphan_verified is Windows-only",
)
def test_kill_orphan_verified_spares_pid_whose_image_is_outside_scope():
    """The PID-reuse guard itself: when the process behind the PID does
    NOT live inside the scope at KILL time (the enumeration snapshot is
    stale -- the orphan died and the PID was recycled by an unrelated
    process), the helper must NOT kill it and must report False.

    Fails red if: the kill is fired on the raw PID without re-verifying
    the image path on the pinned handle -- the innocent stand-in process
    below dies.

    Seeded-defect drill: make _kill_orphan_verified skip the
    QueryFullProcessImageNameW scope re-check -> the out-of-scope child
    is terminated -> red.
    """
    import subprocess as _sp
    import time as _time

    from yamabiko import watchdog as wd

    # Stand-in for "unrelated process that inherited the recycled PID":
    # a live child whose executable is genuinely outside the fake scope.
    proc = _sp.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=_sp.CREATE_NO_WINDOW,
    )
    try:
        assert wd._kill_orphan_verified(
            proc.pid, "z:/definitely/not/our/tree",
        ) is False
        _time.sleep(0.2)
        assert proc.poll() is None, (
            "an out-of-scope process was killed: the PID-reuse guard "
            "did not re-verify the image path at kill time"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


@pytest.mark.real
@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="_kill_orphan_verified is Windows-only",
)
def test_kill_orphan_verified_skips_already_exited_pid():
    """An orphan that exits between WMI enumeration and the kill must be
    skipped silently (False), not counted as a kill -- and certainly
    not chased via a raw PID that the OS is now free to recycle.

    Fails red if: the helper reports True for a process that was
    already dead (sweep kill-count drifts), or raises on the
    open/exit-code probe of a terminated process.
    """
    import subprocess as _sp
    from pathlib import Path

    from yamabiko import watchdog as wd

    proc = _sp.Popen(
        [sys.executable, "-c", "pass"],
        creationflags=_sp.CREATE_NO_WINDOW,
    )
    proc.wait(timeout=15.0)  # fully exited; Popen still pins the handle
    scope = str(Path(sys.executable).parent).replace("\\", "/").lower()
    assert wd._kill_orphan_verified(proc.pid, scope) is False, (
        "helper reported a kill for an already-exited PID; the sweep "
        "count would over-report and a recycled PID could be targeted"
    )


# ---------------------------------------------------------------------------
# The llama spawn factory must poll llama_stop_event, not stop_event.
#
# build_watchdog creates two distinct events: stop_event (shared, propagates
# all shutdown paths) and llama_stop_event (llama-scoped, used by the llama
# supervisor and the HealthProbe).  The llama spawn factory
# (_make_real_spawn_llama) must receive llama_stop_event so that a
# HealthProbe-triggered llama_stop trip during the first-run model-wait can
# cancel the non-daemon supervisor thread.  Closing over stop_event means
# llama_stop trips are invisible to the factory, leaving the supervisor join
# to hang forever in run_watchdog.
# ---------------------------------------------------------------------------


def test_build_watchdog_llama_spawn_polls_llama_stop_event(
    _watchdog_env, monkeypatch,
):
    """Fails red if: the llama spawn factory is wired to the MAIN
    stop_event instead of llama_stop_event. If wired to the wrong event,
    a llama_stop trip during the first-run model-wait can never cancel
    the non-daemon supervisor -> run_watchdog join hangs.

    Seeded-defect drill: change build_watchdog to pass ``stop_event``
    instead of ``llama_stop_event`` to ``_make_real_spawn_llama`` ->
    ``captured["event"] is app.stop_event`` (not ``llama_stop_event``)
    -> assertion fails.
    """
    from yamabiko import watchdog as wd_mod

    captured = {}
    orig = wd_mod._make_real_spawn_llama

    def _capturing(event, **kwargs):
        # ``port=...`` is derived from translation.endpoint; this test
        # only cares about the event identity, so pass everything else
        # through untouched.
        captured["event"] = event
        return orig(event, **kwargs)

    monkeypatch.setattr(wd_mod, "_make_real_spawn_llama", _capturing)
    app = wd_mod.build_watchdog()
    try:
        assert "event" in captured, (
            "_make_real_spawn_llama was never called by build_watchdog; "
            "the llama supervisor spawn factory wiring is absent entirely."
        )
        assert captured["event"] is app.llama_stop_event, (
            "llama spawn factory must poll llama_stop_event, not stop_event. "
            "If wired to stop_event, a HealthProbe-triggered llama_stop trip "
            "during the first-run model-wait leaves the non-daemon supervisor "
            "thread spinning indefinitely -> run_watchdog join hang."
        )
    finally:
        app.pipeline_writer.close()


# ---------------------------------------------------------------------------
# Give-up dialogs must render on the MAIN thread.
#
# The supervisors run on worker threads. Building / showing the startup
# error window (a QDialog) off the GUI thread is unsafe. So a TERMINAL give-up now
# records a StartupError into a thread-safe list (pending_errors) on the
# worker thread; run_watchdog drains + renders it on the MAIN thread
# AFTER join. These tests assert (a) the recorder appends to the
# collector, and (b) every show_startup_error call lands on
# threading.main_thread().
# ---------------------------------------------------------------------------


def test_build_watchdog_record_error_appends_to_pending_errors(_watchdog_env):
    """The pipeline supervisor's on_give_up recorder must append a
    StartupError to WatchdogApp.pending_errors under the lock -- NOT
    build/show the startup error window inline on the worker thread.

    Seeded-defect drill: revert _record_error to call show_startup_error
    directly -> pending_errors stays empty AND a Qt dialog is built
    off-thread -> the len(...) == 1 assertion goes red.
    """
    from yamabiko.ui.error_dialog import StartupError

    wd_mod = _watchdog_env.module
    app = wd_mod.build_watchdog([])
    try:
        wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
        # Precondition: nothing queued yet.
        assert app.pending_errors == []
        # Invoke the wired recorder exactly as Watchdog.run would on a
        # min-lifetime give-up. No Qt should be touched.
        wd_pipeline._on_give_up("Yamabiko failed to start", "boom detail line")
        with app.pending_errors_lock:
            queued = list(app.pending_errors)
        assert len(queued) == 1, (
            "pipeline on_give_up did not record a StartupError to "
            "pending_errors; if it called show_startup_error directly it "
            "built the startup error window on a worker thread."
        )
        err = queued[0]
        assert isinstance(err, StartupError)
        assert err.title.startswith("Pipeline: "), (
            f"recorded error title {err.title!r} missing the 'Pipeline: ' "
            f"label that identifies which supervisor gave up."
        )
        assert err.details == "boom detail line"
    finally:
        app.pipeline_writer.close()


def test_no_llama_idle_probe_symbol():
    """Fails red if: LlamaIdleProbe is reintroduced. Legitimate idle must
    NOT kill llama; the idle probe was removed deliberately."""
    from yamabiko import watchdog as wd
    assert not hasattr(wd, "LlamaIdleProbe")


def test_run_watchdog_starts_no_metrics_poller(_watchdog_env, monkeypatch):
    """Fails red if: run_watchdog still polls llama /metrics. The idle
    probe was removed; only /livez should be polled. Captures every
    httpx.get URL during a brief run_watchdog run and asserts none
    target /metrics.

    Mirrors test_run_watchdog_starts_health_probe_thread exactly: starts
    run_watchdog in a background thread, waits for the first probe GET
    to arrive via an event, stops, then inspects all captured URLs.

    Seeded-defect drill: with LlamaIdleProbe still present in run_watchdog,
    a /metrics URL will be captured -> assertion fails. After deletion only
    /livez (HealthProbe) appears -> passes.
    """
    import threading as _th
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    call_event = threading.Event()
    captured_urls: list = []

    class _CapturingHttpx:
        @staticmethod
        def get(url, timeout=2.0):
            captured_urls.append(url)
            call_event.set()
            class _R:
                status_code = 200
                text = ""
            return _R()

    monkeypatch.setattr(wd_mod, "loopback_http", _CapturingHttpx())

    # Deterministic config: fast poll intervals for BOTH the HealthProbe
    # and (if it still exists) the LlamaIdleProbe, so that if /metrics
    # is still being polled it will show up within the test window.
    # The llama poll interval is deliberately set fast (0.01s) so that
    # if LlamaIdleProbe is still wired in run_watchdog, it fires a
    # /metrics GET before call_event.wait() times out -> /metrics
    # appears in captured_urls -> assertion fails (RED before deletion).
    # After deletion: only /livez (HealthProbe) appears -> GREEN.
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {
            "overlay": {"websocket_port": 8765},
            "lifecycle": {
                "health": {"poll_interval_s": 0.01, "fail_budget_s": 120.0},
                # BAIT, not a live consumer: no production code reads
                # lifecycle.llama.idle_* after 5B deleted LlamaIdleProbe.
                # The keys are injected so a re-introduced /metrics
                # poller would pick them up and surface as a /metrics
                # GET in captured_urls -> the assertion below goes red.
                "llama": {"idle_timeout_s": 900.0, "idle_poll_interval_s": 0.01},
            },
        },
    )

    # Supervisors block until stop_event fires so run_watchdog doesn't
    # return before the probe has polled at least once.
    for wd in app.watchdogs:
        wd.run = lambda _ev=app.stop_event: _ev.wait(timeout=5.0)

    t = _th.Thread(target=wd_mod.run_watchdog, args=(app,), daemon=True)
    t.start()

    assert call_event.wait(timeout=3.0), (
        "No httpx.get was observed during run_watchdog in 3s; "
        "either the HealthProbe thread was not started or it "
        "never reached its first iteration -- test harness broken."
    )

    # Allow a further short window for both probes (HealthProbe at
    # /livez and, if still present, LlamaIdleProbe at /metrics) to
    # accumulate multiple GETs. Both are configured at 0.01s poll
    # interval; 200ms gives ~20 iterations each -- sufficient for any
    # /metrics calls to register in captured_urls before we inspect.
    # Best-effort bait window for a NEGATIVE assertion: a starved
    # runner shrinking it can only weaken the bait (missed detection
    # of a reintroduced /metrics poller), never produce a false red --
    # the call_event.wait(3.0) above already guarantees >=1 GET
    # (PR #46 review).
    import time as _time
    _time.sleep(0.2)

    app.stop_event.set()
    t.join(timeout=5.0)

    assert captured_urls, "expected at least one probe GET"
    assert not any("/metrics" in u for u in captured_urls), (
        f"run_watchdog still polls /metrics: {captured_urls}. "
        f"The LlamaIdleProbe was re-introduced -- the idle probe must "
        f"remain removed; only /livez should be polled."
    )


def test_run_watchdog_renders_pending_errors_on_main_thread(
    _watchdog_env, monkeypatch,
):
    """Give-up dialogs must render on the MAIN thread after join, never
    inside a supervisor worker thread. And a run that ended in a
    give-up dialog must exit NONZERO: hard rule #9 consumers (frozen
    E2E, installer smoke) read the watchdog exit code as ground truth,
    so exit 0 after an error dialog reports a broken run as success.

    Seeded-defect drills:
      - move the drain back into _record_error (worker thread) -> the
        recorded render-thread name is a '*-watchdog' worker, not main
        -> red.
      - compute the exit code without consulting the drained queue
        (``return 0`` on this path) -> the rc assert goes red.
    """
    import threading
    from yamabiko import watchdog as wd_mod
    app = wd_mod.build_watchdog([])

    render_threads = []
    # show_startup_error is hoisted to module level in watchdog.py so the
    # drain references the module global and this patch intercepts it.
    monkeypatch.setattr(
        wd_mod, "show_startup_error",
        lambda err: render_threads.append(threading.current_thread()),
        raising=False,
    )

    import signal as _sig
    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    # Force the pipeline supervisor to give up fast (sub-min-lifetime
    # crash) so its real _record_error("Pipeline") recorder queues an
    # error to app.pending_errors -- the same path a config-error crash
    # takes in production. Stub the llama supervisor so the join returns.
    fake_proc = _FakeProc(exit_code=1)  # ordinary crash, not STOP_ALL
    monkeypatch.setattr(wd_mod, "_real_spawn", lambda env=None: fake_proc)

    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    wd_llama = next(w for w in app.watchdogs if w.name == "llama")
    # Make the crash trip the min-lifetime give-up immediately: elapsed
    # (0.0 via the clock below) < min_lifetime, so run() records an error
    # and cascades. Patch the instance clock so we don't rely on wall time.
    wd_pipeline._clock = lambda: 0.0
    wd_pipeline._min_lifetime_s = 5.0
    wd_pipeline._sleep = lambda s: None
    # Llama supervisor: exit cleanly the moment its loop sees stop, so the
    # non-daemon thread joins (the cascade sets llama_stop_event via the
    # _mirror; we short-circuit run() to avoid the real model-wait).
    wd_llama.run = lambda _ev=app.llama_stop_event: _ev.wait(timeout=5.0)

    rc = wd_mod.run_watchdog(app)

    assert rc != 0, (
        f"run_watchdog returned {rc} after a give-up dialog; a run that "
        f"ended in an error dialog must exit nonzero (hard rule #9 "
        f"consumers read this code as success/failure)"
    )
    assert render_threads, (
        "no give-up dialog rendered; the pipeline give-up either didn't "
        "record an error or run_watchdog never drained pending_errors."
    )
    assert all(t is threading.main_thread() for t in render_threads), (
        f"give-up dialog rendered off the main thread: "
        f"{[t.name for t in render_threads]}. "
        f"A startup error window built on a supervisor worker thread is unsafe."
    )


def test_build_watchdog_logs_to_watchdog_role_file(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """build_watchdog routes file logging to main-watchdog.log -- its own
    per-process file -- never the shared main.log. stdlib
    RotatingFileHandler rotation is only safe with a single writer
    process; on Windows a shared file makes doRollover raise
    PermissionError, dropping records and destroying backups.

    Does NOT use _watchdog_env: that fixture stubs setup_main_log, and
    the subject here IS the setup_main_log call site.

    Fails red if: the build_watchdog call site drops the
    filename="main-watchdog.log" argument, or stops calling
    setup_main_log entirely (role file never created).
    """
    import logging as _logging
    import yamabiko.watchdog as wd_mod
    from yamabiko import log_config

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(
        wd_mod, "_acquire_single_instance_mutex", lambda: object(),
    )
    # This test bypasses _watchdog_env (its subject is the
    # setup_main_log call site), so stub the outer job here too --
    # never join the live pytest process to a real KILL_ON_JOB_CLOSE job.
    monkeypatch.setattr(
        wd_mod, "create_outer_job", lambda name: _FakeOuterJob(),
    )
    # The REAL install_crash_evidence would park a process-lifetime
    # faulthandler handle inside tmp_path (Windows then can't delete
    # it) and install global excepthooks; its behavior is owned by
    # tests/test_log_config.py, its call-site wiring by the dedicated
    # recorder test below.
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )

    with hermetic_log_handlers("", "yamabiko"):
        app = wd_mod.build_watchdog([])
        try:
            for h in _logging.getLogger("yamabiko").handlers:
                h.flush()
            logs_dir = tmp_path / "logs"
            role_log = logs_dir / "main-watchdog.log"
            assert role_log.is_file(), (
                f"main-watchdog.log missing after build_watchdog; logs dir: "
                f"{[p.name for p in logs_dir.iterdir()] if logs_dir.is_dir() else 'absent'}"
            )
            assert "watchdog starting" in role_log.read_text(encoding="utf-8"), (
                "role log exists but the 'watchdog starting (pid=...)' line "
                "logged right after setup_main_log never reached it"
            )
            assert not (logs_dir / "main.log").exists(), (
                "build_watchdog wrote the shared main.log; "
                "cross-process rotation contention is back"
            )
        finally:
            # This test bypasses _watchdog_env's writer-closing teardown,
            # so close the pipeline shutdown event here.
            app.pipeline_writer.close()


# ---------------------------------------------------------------------------
# Watchdog self-crash evidence (three layers).
#
# Layer 1: main()'s BaseException guard -- CRITICAL + dialog + exit 1.
# Layer 2: Watchdog.run's internal-error give-up (tests/test_watchdog.py).
# Layer 3: run_watchdog's dead-thread detection -- a supervisor thread
#          that terminated with NEITHER stop_event set NOR a give-up
#          recorded died abnormally; returning 0 there mimics a clean
#          user quit ("silent exit 0").
# ---------------------------------------------------------------------------


def test_main_crash_logs_critical_shows_dialog_and_exits_1(
    monkeypatch, caplog,
):
    """Layer 1: an exception escaping the watchdog composition/run path
    must produce a CRITICAL log with traceback, a best-effort dialog,
    and process exit code 1 -- not an invisible death on the discarded
    frozen stderr.

    Seeded-defect drill: remove main()'s try/except BaseException guard
    -> the seeded RuntimeError propagates out of main() instead of
    SystemExit(1) -> red.
    """
    import logging as _logging
    import sys as _sys
    import yamabiko.watchdog as wd_mod

    caplog.set_level(_logging.CRITICAL, logger="yamabiko.watchdog")
    monkeypatch.setattr(_sys, "argv", ["yamabiko"])

    def _seeded_build(argv=None):
        raise RuntimeError("seeded composition crash")

    monkeypatch.setattr(wd_mod, "build_watchdog", _seeded_build)

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )
    # logging.shutdown() would close pytest's own capture handlers for
    # the rest of the session; record the call instead of running it.
    shutdown_calls: list = []
    monkeypatch.setattr(
        wd_mod.logging, "shutdown", lambda: shutdown_calls.append(True),
    )

    with pytest.raises(SystemExit) as exc_info:
        wd_mod.main()

    assert exc_info.value.code == 1, (
        f"main() exited {exc_info.value.code!r} after a supervisor "
        f"crash; a nonzero code is the only machine-readable crash "
        f"signal the OS keeps"
    )
    assert "watchdog crashed" in caplog.text and "RuntimeError" in caplog.text, (
        "no CRITICAL crash record with traceback was logged; the crash "
        "leaves zero evidence in main-watchdog.log"
    )
    assert any(r.exc_info for r in caplog.records), (
        "the CRITICAL crash record carries no exc_info traceback"
    )
    assert rendered, (
        "no dialog was shown for the watchdog crash; the user sees "
        "Yamabiko simply not start"
    )
    assert shutdown_calls, (
        "logging.shutdown() was not called before exit; buffered "
        "records may never reach the log file"
    )


def test_main_reraises_systemexit_without_crash_dialog(monkeypatch, caplog):
    """The layer-1 guard must NOT convert deliberate ``sys.exit`` paths
    (the mutex 'already running' exit 0, run_watchdog's exit code) into
    the crash dialog + exit 1.

    Seeded-defect drill: drop the ``except SystemExit: raise`` re-raise
    ahead of the BaseException guard -> the exit-0 below becomes a
    crash-dialog exit 1 -> red.
    """
    import logging as _logging
    import sys as _sys
    import yamabiko.watchdog as wd_mod

    caplog.set_level(_logging.CRITICAL, logger="yamabiko.watchdog")
    monkeypatch.setattr(_sys, "argv", ["yamabiko"])

    def _already_running_build(argv=None):
        _sys.exit(0)  # the mutex-held path inside build_watchdog

    monkeypatch.setattr(wd_mod, "build_watchdog", _already_running_build)

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )
    monkeypatch.setattr(wd_mod.logging, "shutdown", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        wd_mod.main()

    assert exc_info.value.code == 0, (
        f"a deliberate exit 0 was rewritten to {exc_info.value.code!r}; "
        f"the crash guard must re-raise SystemExit untouched"
    )
    assert not rendered, (
        "the crash dialog fired for a deliberate sys.exit(0); every "
        "second-instance launch would now show a bogus crash dialog"
    )
    assert "watchdog crashed" not in caplog.text


def test_main_keyboard_interrupt_during_startup_exits_130_without_crash_dialog(
    monkeypatch, caplog,
):
    """The layer-1 guard must not turn Ctrl+C into a crash report. A
    KeyboardInterrupt in the pre-signal-handler startup window (before
    run_watchdog installs _on_signal -- e.g. during build_watchdog's
    config load or the orphan sweep) is a USER action: no 'Yamabiko
    crashed' dialog, no CRITICAL record; one INFO line and exit 130
    (128+SIGINT, the shell convention for killed-by-Ctrl+C).

    Seeded-defect drill: drop the ``except KeyboardInterrupt`` branch
    ahead of the BaseException guard -> the crash dialog renders and
    main() exits 1 -> the 130 and no-dialog asserts go red.
    """
    import logging as _logging
    import sys as _sys
    import yamabiko.watchdog as wd_mod

    caplog.set_level(_logging.INFO, logger="yamabiko.watchdog")
    monkeypatch.setattr(_sys, "argv", ["yamabiko"])

    def _interrupted_build(argv=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(wd_mod, "build_watchdog", _interrupted_build)

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )
    # Guard pytest's capture handlers in the red case (the BaseException
    # branch calls logging.shutdown() for real).
    monkeypatch.setattr(wd_mod.logging, "shutdown", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        wd_mod.main()

    assert exc_info.value.code == 130, (
        f"Ctrl+C during startup exited {exc_info.value.code!r}; expected "
        f"130 (128+SIGINT) -- a user interrupt is not a crash"
    )
    assert not rendered, (
        "the 'Yamabiko crashed' dialog fired for a user's Ctrl+C during "
        "the startup window; an interrupt is not a crash report"
    )
    assert "watchdog crashed" not in caplog.text, (
        "a CRITICAL crash record was logged for a user's Ctrl+C"
    )
    assert any(
        r.levelno == _logging.INFO and "interrupt" in r.getMessage().lower()
        for r in caplog.records
    ), (
        "no INFO record notes the startup interrupt; the early exit "
        "leaves zero trace of WHY the process stopped"
    )


def test_run_watchdog_flags_silent_supervisor_death_with_dialog_and_exit_1(
    _watchdog_env, monkeypatch, caplog,
):
    """Layer 3: every legitimate terminal path either sets stop_event
    (clean exit / STOP_ALL / cascade) or records a pending give-up
    before returning. Supervisor threads that reach join() with NEITHER
    died abnormally; run_watchdog must log CRITICAL, queue a dialog, and
    return nonzero instead of mimicking a clean exit 0.

    Seeded-defect drill: remove the dead-thread detection block after
    the join loop -> rc == 0 and no dialog -> red.
    """
    import logging as _logging
    import signal as _sig

    caplog.set_level(_logging.CRITICAL, logger="yamabiko.watchdog")
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )

    # Silent-death mimic: each supervisor "thread" returns immediately
    # WITHOUT setting stop_event and WITHOUT recording a give-up --
    # exactly what a pre-guard supervisor crash looked like from
    # run_watchdog's join.
    for wd in app.watchdogs:
        wd.run = lambda: None

    rc = wd_mod.run_watchdog(app)

    assert rc != 0, (
        "run_watchdog returned 0 after every supervisor died silently; "
        "the 'silent exit 0' mimic is back -- a crashed supervisor is "
        "indistinguishable from a clean user quit"
    )
    assert rendered, (
        "no dialog was queued/rendered for the silent supervisor death"
    )
    assert any(
        r.levelno == _logging.CRITICAL for r in caplog.records
    ), "no CRITICAL record was logged for the silent supervisor death"


def test_run_watchdog_returns_zero_when_stop_event_set(_watchdog_env, monkeypatch):
    """Negative twin for the dead-thread detection: a supervisor exit
    WITH stop_event set is the normal shutdown path and must stay
    rc == 0, no dialog.

    Seeded-defect drill: make the detection unconditional (drop the
    ``stop_event.is_set()`` check) -> this clean path returns 1 and
    shows a bogus crash dialog on every normal quit -> red.
    """
    import signal as _sig
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )

    app.stop_event.set()
    for wd in app.watchdogs:
        wd.run = lambda: None

    rc = wd_mod.run_watchdog(app)

    assert rc == 0, f"clean stop_event shutdown returned {rc}; expected 0"
    assert not rendered, (
        "a dialog was rendered on the clean shutdown path; every normal "
        "quit would now show a bogus crash dialog"
    )


def test_run_watchdog_returns_nonzero_when_supervisor_internal_error_gives_up(
    _watchdog_env, monkeypatch,
):
    """A layer-2 internal-error give-up (the supervision logic ITSELF
    raised -- seeded here by making ``classify`` explode) logs, dialogs,
    and cascades -- and run_watchdog must exit NONZERO. Before this fix
    the give-up path returned 0: hard rule #9 consumers (frozen E2E,
    installer smoke) read the exit code as ground truth, so a supervisor
    internal error reported the run as a success.

    Seeded-defect drill: derive the exit code only from the silent-death
    flag (ignore recorded give-ups: ``return 1 if silent_death else 0``)
    -> this internal-error give-up returns 0 -> red.
    """
    import signal as _sig

    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )

    def _seeded_classify(code):
        raise RuntimeError("seeded supervision bug in classify")

    monkeypatch.setattr(wd_mod, "classify", _seeded_classify)
    # The pipeline supervisor's spawn returns a fast-exiting fake child
    # so run() reaches classify() promptly; the llama supervisor is
    # short-circuited to unwind on the cascaded stop (via the _mirror).
    monkeypatch.setattr(
        wd_mod, "_real_spawn", lambda env=None: _FakeProc(exit_code=1),
    )
    wd_llama = next(w for w in app.watchdogs if w.name == "llama")
    wd_llama.run = lambda _ev=app.llama_stop_event: _ev.wait(timeout=10.0)

    rc = wd_mod.run_watchdog(app)

    assert rc != 0, (
        "run_watchdog returned 0 after a supervisor internal-error "
        "give-up; frozen E2E and installer smoke would read this broken "
        "run as success (hard rule #9)"
    )
    assert rendered and any(
        "internal error" in err.title for err in rendered
    ), (
        f"no internal-error dialog was drained/rendered; rendered "
        f"titles: {[e.title for e in rendered]!r}"
    )


@pytest.mark.filterwarnings(
    # The seeded SystemExit IS the subject: pytest's threadexception
    # plugin records it from the dying worker thread; that's expected.
    "ignore::pytest.PytestUnhandledThreadExceptionWarning",
)
def test_run_watchdog_detects_flagless_dead_thread_while_sibling_still_runs(
    _watchdog_env, monkeypatch, caplog,
):
    """Layer 3 must not wait for ALL supervisor threads to exit. One
    supervisor dying flag-lessly -- a BaseException past the layer-2
    ``except Exception`` guard, seeded here as SystemExit, which
    threading's excepthook swallows silently -- while its sibling keeps
    running used to park run_watchdog in a blocking join() forever:
    role unsupervised, single-instance mutex held, zero evidence. The
    monitor loop must spot the dead thread within its join-timeout
    budget, log CRITICAL, queue the dialog, CASCADE stop (so the
    healthy sibling unwinds through its normal ladder), and return
    nonzero.

    Seeded-defect drill: revert the monitor loop to a plain blocking
    ``for t in threads: t.join()`` -> nothing cascades stop, so the
    sibling's 20 s wait must elapse before run_watchdog can even reach
    the post-join check -> the elapsed and stop_event asserts go red.
    """
    import logging as _logging
    import signal as _sig
    import time as _time

    caplog.set_level(_logging.CRITICAL, logger="yamabiko.watchdog")
    env = _watchdog_env
    wd_mod = env.module
    app = wd_mod.build_watchdog([])

    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)
    monkeypatch.setattr(wd_mod, "loopback_http", _FakeHttpxAlwaysOk())

    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )

    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    wd_llama = next(w for w in app.watchdogs if w.name == "llama")

    def _flagless_death():
        # SystemExit bypasses layer 2 (except Exception) AND the default
        # threading.excepthook (which returns early for SystemExit):
        # the thread dies with neither stop_event set nor a give-up
        # recorded -- the exact failure mode under test.
        raise SystemExit(1)

    wd_pipeline.run = _flagless_death
    # Healthy sibling: parks until the llama-scoped event fires (the
    # _mirror forwards the cascaded stop_event). The 20 s ceiling only
    # exists so a regressed (blocking-join) implementation fails this
    # test instead of hanging the suite.
    wd_llama.run = lambda _ev=app.llama_stop_event: _ev.wait(timeout=20.0)

    start = _time.monotonic()
    rc = wd_mod.run_watchdog(app)
    elapsed = _time.monotonic() - start

    assert rc != 0, (
        "run_watchdog returned 0 after a supervisor thread died "
        "flag-lessly; the 'silent exit 0' mimic is back"
    )
    assert elapsed < 10.0, (
        f"run_watchdog took {elapsed:.1f}s to notice the dead supervisor "
        f"thread; the monitor loop must detect it within its join-timeout "
        f"budget, not after the surviving sibling happens to exit"
    )
    assert app.stop_event.is_set(), (
        "dead-thread detection did not cascade stop; the surviving "
        "sibling would keep running with its peer role unsupervised and "
        "the single-instance mutex held"
    )
    assert any(
        r.levelno == _logging.CRITICAL for r in caplog.records
    ), "no CRITICAL record was logged for the flag-less dead thread"
    assert rendered, (
        "no dialog was queued/rendered for the flag-less dead thread"
    )


def test_build_watchdog_config_logging_level_reaches_watchdog_logger(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """``logging.level: DEBUG`` in the merged config must set the
    watchdog process's ``yamabiko`` logger to DEBUG. Before this fix the
    call site hardcoded level="INFO", so the documented config key was
    dead for the watchdog role (hard rule #4's producer/consumer class).

    Does NOT use _watchdog_env: that fixture stubs setup_main_log, and
    the subject here IS the level flowing through the real call.

    Seeded-defect drill: hardcode ``level="INFO"`` back at the
    build_watchdog setup_main_log call site -> effective level stays
    INFO -> red.
    """
    import logging as _logging
    import yamabiko.watchdog as wd_mod
    from yamabiko import log_config

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(
        wd_mod, "_acquire_single_instance_mutex", lambda: object(),
    )
    monkeypatch.setattr(
        wd_mod, "create_outer_job", lambda name: _FakeOuterJob(),
    )
    # Keep the process-global excepthook chain + faulthandler handle out
    # of this tmp_path-scoped test (behavior owned by test_log_config).
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"logging": {"level": "DEBUG"}},
    )

    with hermetic_log_handlers("", "yamabiko"):
        app = wd_mod.build_watchdog([])
        try:
            assert (
                _logging.getLogger("yamabiko").getEffectiveLevel()
                == _logging.DEBUG
            ), (
                "logging.level: DEBUG never reached the watchdog's "
                "setup_main_log; the config key is dead for this role"
            )
        finally:
            app.pipeline_writer.close()


def test_build_watchdog_normalizes_lowercase_logging_level_without_crashing(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """``logging.level: debug`` (lowercase -- the natural YAML spelling)
    must NOT abort watchdog startup: ``logger.setLevel('debug')`` raises
    ValueError, which used to escape build_watchdog, turn into the
    layer-1 crash dialog, and crash-loop the overlay -- while the
    call-site comment promised bad config would never block startup.
    The value is normalized case-insensitively to DEBUG.

    Same harness as the DEBUG test above (no _watchdog_env: the subject
    IS the level flowing through the real setup_main_log call).

    Seeded-defect drill: pass the raw config value straight to
    setup_main_log (drop the normalization) -> build_watchdog raises
    ValueError -> red.
    """
    import logging as _logging
    import yamabiko.watchdog as wd_mod
    from yamabiko import log_config

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(
        wd_mod, "_acquire_single_instance_mutex", lambda: object(),
    )
    monkeypatch.setattr(
        wd_mod, "create_outer_job", lambda name: _FakeOuterJob(),
    )
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    monkeypatch.setattr(
        "yamabiko.config.load_full_config",
        lambda: {"logging": {"level": "debug"}},
    )

    with hermetic_log_handlers("", "yamabiko"):
        app = wd_mod.build_watchdog([])  # must NOT raise
        try:
            assert (
                _logging.getLogger("yamabiko").getEffectiveLevel()
                == _logging.DEBUG
            ), (
                "lowercase 'debug' was not normalized to DEBUG; the "
                "user's intent (more verbosity) was silently dropped"
            )
        finally:
            app.pipeline_writer.close()


def test_build_watchdog_falls_back_to_info_on_unknown_logging_level(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """An unknown ``logging.level`` in a REAL config file (typo, numeric
    YAML) must degrade to INFO -- never abort startup -- and must leave
    a WARNING naming the rejected value in main-watchdog.log. The
    warning is asserted from the ROLE FILE (not caplog): build_watchdog
    loads config BEFORE setup_main_log attaches the file handler, so
    only the deferred (stash -> consume -> emit-after-attach) sequencing
    gets the evidence into the file; an emission inside load_full_config
    dies on the frozen bundle's discarded stderr.

    NO load_full_config mock: the REAL loader normalizes the merged
    value to INFO in place, so a mocked raw-'verbose' return is
    production-impossible and previously kept build_watchdog's deferred
    _level_warn seam looking alive while it was dead on every real path.

    Seeded-defect drills:
      - drop the consume_level_warning() call in build_watchdog (the
        deferred seam) -> _level_warn is None on the real path -> the
        role-file assert goes red.
      - emit the warning inside load_full_config / the normalizer
        instead of stashing -> pre-attach emission never reaches the
        role file -> red.
      - degrade silently (drop the post-setup WARNING emission at the
        _level_warn seam) -> red.
    """
    import logging as _logging
    import yamabiko.watchdog as wd_mod
    from yamabiko import log_config

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    # REAL temp config: the junk level flows through the real layered
    # load (shipped defaults <- user_overrides.yaml).
    (tmp_path / "user_overrides.yaml").write_text(
        "logging:\n  level: verbose\n", encoding="utf-8",
    )
    monkeypatch.setattr(
        wd_mod, "_acquire_single_instance_mutex", lambda: object(),
    )
    monkeypatch.setattr(
        wd_mod, "create_outer_job", lambda name: _FakeOuterJob(),
    )
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )

    with hermetic_log_handlers("", "yamabiko"):
        app = wd_mod.build_watchdog([])  # must NOT raise
        try:
            assert (
                _logging.getLogger("yamabiko").getEffectiveLevel()
                == _logging.INFO
            ), (
                "unknown logging.level did not fall back to INFO; the "
                "documented default drifted"
            )
            for h in _logging.getLogger("yamabiko").handlers:
                h.flush()
            role_log = tmp_path / "logs" / "main-watchdog.log"
            assert role_log.is_file() and (
                "logging.level='verbose'"
                in role_log.read_text(encoding="utf-8")
            ), (
                "no WARNING naming the rejected 'verbose' value reached "
                "main-watchdog.log; the fallback is invisible in the "
                "field (a pre-attach warning dies on frozen stderr, so "
                "the deferred _level_warn seam must be live)"
            )
        finally:
            app.pipeline_writer.close()


# ---------------------------------------------------------------------------
# Crash-evidence + identity-banner wiring at the watchdog entry.
# ---------------------------------------------------------------------------


def test_build_watchdog_installs_crash_evidence_for_watchdog_role(
    _watchdog_env, monkeypatch,
):
    """build_watchdog must arm the shared crash-evidence primitives
    (faulthandler -> crash-watchdog.log, sys/threading excepthooks) for
    the "watchdog" role right after log setup.

    Call-site wiring test: the recorder replaces the fixture's stub.
    The helper's BEHAVIOR (hook chaining, crash-file rotation, the
    faulthandler seam) is owned by tests/test_log_config.py and the
    real-tier tests/test_real_crash_evidence.py -- asserting on disk
    state here would be order-dependent (install is idempotent
    per-role per-process).

    Seeded-defect drill: delete the ``install_crash_evidence("watchdog")``
    call from build_watchdog -> recorded stays empty -> red.
    """
    from yamabiko import log_config

    env = _watchdog_env
    recorded: list = []
    monkeypatch.setattr(
        log_config, "install_crash_evidence",
        lambda role: recorded.append(role),
    )

    env.module.build_watchdog([])

    assert recorded == ["watchdog"], (
        f"install_crash_evidence calls: {recorded!r}; the watchdog "
        f"process runs without faulthandler/excepthook evidence -- a "
        f"native or unhandled crash leaves nothing on disk"
    )


def test_build_watchdog_emits_identity_banner_at_startup(
    _watchdog_env, caplog,
):
    """The startup identity banner (role, version, frozen|dev, PID,
    python, OS) must be logged at the watchdog entry -- it is the first
    thing bug_report.yml asks a reporter for.

    Seeded-defect drill: delete the ``log_startup_banner(...)`` call
    from build_watchdog -> no banner line -> red.
    """
    import logging as _logging
    import re

    caplog.set_level(_logging.INFO, logger="yamabiko.watchdog")
    env = _watchdog_env

    env.module.build_watchdog([])

    assert re.search(
        r"yamabiko watchdog v\S+ \((frozen|dev)\) pid=\d+", caplog.text,
    ), (
        f"no identity banner in the watchdog startup records; "
        f"captured: {caplog.text!r}"
    )


# ---------------------------------------------------------------------------
# Dialog text -> log, verbatim (mutex "already running" + give-ups).
# ---------------------------------------------------------------------------


def test_build_watchdog_logs_already_running_dialog_verbatim_to_role_file(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """The "already running" dialog fires on the second-instance path
    and previously ran BEFORE log setup -- the one user-visible dialog
    with zero trace in any log. build_watchdog must now set up
    main-watchdog.log BEFORE the mutex check and write the dialog's
    exact title/body at ERROR, so a bug report quoting the dialog can
    be matched to a log line.

    File-based assertion (not caplog): the point of the ordering move
    is that the text reaches the role FILE.

    Seeded-defect drills:
      - move setup_main_log back below the mutex check -> no
        main-watchdog.log is ever created on this path -> red.
      - drop the ``_log.error(...)`` for the dialog text -> file exists
        but carries no dialog text -> red.
    """
    import logging as _logging
    import yamabiko.watchdog as wd_mod
    from yamabiko import log_config

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    # Second-instance condition: the mutex is already held elsewhere.
    monkeypatch.setattr(
        wd_mod, "_acquire_single_instance_mutex", lambda: None,
    )
    monkeypatch.setattr(
        wd_mod, "create_outer_job", lambda name: _FakeOuterJob(),
    )
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    rendered: list = []
    monkeypatch.setattr(
        wd_mod, "show_startup_error", lambda err: rendered.append(err),
    )

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(SystemExit) as exc_info:
            wd_mod.build_watchdog([])
        assert exc_info.value.code == 0, (
            "the already-running path must stay a friendly exit 0"
        )
        for h in _logging.getLogger("yamabiko").handlers:
            h.flush()
        role_log = tmp_path / "logs" / "main-watchdog.log"
        assert role_log.is_file(), (
            "main-watchdog.log was never created on the already-running "
            "path; setup_main_log still runs after the mutex check"
        )
        content = role_log.read_text(encoding="utf-8")

    assert "Yamabiko is already running" in content, (
        "the dialog TITLE is not in the role log; the one dialog a "
        "double-launching user sees still leaves no trace"
    )
    assert "Only one copy of Yamabiko can run at a time." in content, (
        "the dialog BODY is not in the role log (verbatim copy required "
        "so a bug report quoting the dialog matches a log line)"
    )
    assert rendered and rendered[0].title == "Yamabiko is already running", (
        "the dialog itself must still be shown; logging is in ADDITION "
        "to, not instead of, the user-facing dialog"
    )


def test_build_watchdog_giveup_recorder_logs_dialog_text_at_error(
    _watchdog_env, caplog,
):
    """Every give-up dialog's exact title/body must hit the log at ERROR
    at RECORD time (worker thread), not only at render time -- if the
    process dies before run_watchdog's main-thread drain, the log line
    is the only surviving copy of what the user was told.

    Invokes the wired _on_give_up callable directly (charter §3.9:
    wired callables must be exercised, not just registered).

    Seeded-defect drill: drop the ``_log.error(...)`` from
    _record_error's _on_give_up -> no ERROR record carries the dialog
    text -> red.
    """
    import logging as _logging

    caplog.set_level(_logging.ERROR, logger="yamabiko.watchdog")
    env = _watchdog_env
    app = env.module.build_watchdog([])

    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    detail = "The pipeline process crashed 3 times in the last 10 minutes."
    wd_pipeline._on_give_up("Yamabiko keeps crashing", detail)

    error_text = "\n".join(
        r.getMessage() for r in caplog.records
        if r.levelno >= _logging.ERROR
    )
    assert "Pipeline: Yamabiko keeps crashing" in error_text, (
        f"the give-up dialog TITLE (label-prefixed, exactly as shown) "
        f"is not in the ERROR records; got: {error_text!r}"
    )
    assert detail in error_text, (
        "the give-up dialog BODY is not in the ERROR records; the "
        "verbatim copy is what lets a bug report match a log line"
    )


# ---------------------------------------------------------------------------
# --config unification: the watchdog resolves ITS OWN effective config from
# an explicit --config file (not the layered store) for both the llama
# endpoint plan and the health probe. Without this, a --config that moves
# translation.endpoint or overlay.websocket_port splits the watchdog's view
# from the pipeline it forwards --config to.
# ---------------------------------------------------------------------------


def test_build_watchdog_explicit_config_external_endpoint_skips_llama_supervisor(
    _watchdog_env, tmp_path, monkeypatch,
):
    """build_watchdog's llama endpoint plan must come from --config, not
    the layered store. An explicit --config whose translation.endpoint is
    a NON-loopback host means the bundled llama-server is nothing's peer,
    so no llama supervisor should be built.

    Red today: build_watchdog `del argv`s and loads load_full_config()
    (layered), whose default endpoint is loopback -> llama IS spawned ->
    the supervisor set includes 'llama'.

    Seeded defect: revert build_watchdog to `_cfg = _load_cfg()` (drop the
    explicit_path threading) -> the external endpoint is ignored, llama is
    spawned, and the name set becomes {'pipeline','llama'} -> red.
    """
    # Isolate from the dev machine's user_overrides.yaml so the layered
    # (pre-fix) load is a deterministic default-loopback endpoint.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    env = _watchdog_env
    wd_mod = env.module
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "translation:\n  endpoint: http://example.com:8090/v1\n",
        encoding="utf-8",
    )

    app = wd_mod.build_watchdog(["--config", str(explicit)])

    names = {wd.name for wd in app.watchdogs}
    assert names == {"pipeline"}, (
        f"an external translation.endpoint in --config should suppress the "
        f"bundled llama supervisor; got supervisors {names}. build_watchdog "
        f"is reading the layered config instead of the --config file."
    )


def test_run_watchdog_health_probe_uses_explicit_config_websocket_port(
    _watchdog_env, tmp_path, monkeypatch,
):
    """The health probe must poll the port the pipeline's uvicorn actually
    binds under --config. An explicit --config with a non-default
    overlay.websocket_port must move the probe URL there; otherwise a
    healthy pipeline on the --config port is polled on 8765 forever and
    force-restarted (M-05 restart loop of a healthy pipeline).

    Red today: run_watchdog reloads load_full_config() (layered), which
    ignores --config -> the probe URL stays on the default 8765 despite
    --config saying 9099.

    Seeded defect: drop the explicit_path threading in run_watchdog's
    health-probe config load -> URL reverts to 8765 -> red.
    """
    import signal as _sig

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    env = _watchdog_env
    wd_mod = env.module
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "overlay:\n"
        "  websocket_port: 9099\n"
        "lifecycle:\n"
        "  health:\n"
        "    poll_interval_s: 0.01\n"
        "    fail_budget_s: 120.0\n",
        encoding="utf-8",
    )

    app = wd_mod.build_watchdog(["--config", str(explicit)])
    monkeypatch.setattr(_sig, "signal", lambda signum, handler: None)

    captured_urls: list = []
    call_event = threading.Event()

    class _RecordingHttpx:
        @staticmethod
        def get(url, timeout=2.0):
            captured_urls.append(url)
            call_event.set()

            class _R:
                status_code = 200

            return _R()

    monkeypatch.setattr(wd_mod, "loopback_http", _RecordingHttpx())

    # Keep the supervisors parked so run_watchdog doesn't return before
    # the probe polls (and no real subprocess is spawned).
    for wd in app.watchdogs:
        wd.run = lambda _ev=app.stop_event: _ev.wait(timeout=5.0)

    t = threading.Thread(target=wd_mod.run_watchdog, args=(app,), daemon=True)
    t.start()

    assert call_event.wait(timeout=3.0), "health probe never polled httpx.get"
    app.stop_event.set()
    t.join(timeout=5.0)

    assert captured_urls[0] == "http://127.0.0.1:9099/livez", (
        f"health probe URL did not reflect the --config "
        f"overlay.websocket_port=9099; got {captured_urls[0]!r}. run_watchdog "
        f"is reading the layered config instead of the --config file."
    )


def test_build_watchdog_llama_spawn_resolves_quant_tier_from_explicit_config(
    _watchdog_env, tmp_path, monkeypatch,
):
    """Full chain: build_watchdog(--config) must thread the explicit path
    into the llama spawn closure so the translator model resolver picks the
    tier the --config file says -- observable as the ``--model`` argv the
    closure hands Popen. Without the threading, the resolver does a bare
    layered load and a dev using --config to repro the small-tier path
    gets the default-tier model.

    Both tiers are cached so the self-heal fallback cannot mask a wrong
    tier choice. No load_full_config mock: the resolver's REAL config
    load runs against the real explicit file, with YAMABIKO_APPDATA
    isolated so the layered store says the shipped 'default' tier.

    Red today: the closure calls _resolve_translator_model_path() bare ->
    layered 'default' -> --model is the default GGUF.

    Seeded defects: (a) drop config_path from the _make_real_spawn_llama
    call in build_watchdog, or (b) drop the explicit_path pass-through
    inside the closure -> --model reverts to the default GGUF -> red.
    """
    import subprocess as _sp

    import huggingface_hub

    from yamabiko import models

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    env = _watchdog_env
    wd_mod = env.module
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "translation:\n  quant_tier: small\n", encoding="utf-8",
    )

    small_gguf = tmp_path / "translator-small.gguf"
    small_gguf.write_bytes(b"gguf")
    default_gguf = tmp_path / "translator-default.gguf"
    default_gguf.write_bytes(b"gguf")

    def _cache(repo_id, filename, revision=None):
        if filename == models.TRANSLATOR_PINS["small"].filename:
            return str(small_gguf)
        if filename == models.TRANSLATOR_PINS["default"].filename:
            return str(default_gguf)
        return None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _cache)

    app = wd_mod.build_watchdog(["--config", str(explicit)])
    wd_llama = next(
        (w for w in app.watchdogs if w.name == "llama"), None,
    )
    assert wd_llama is not None, (
        "the default (loopback) endpoint from the merged --config must "
        "keep the bundled llama supervisor"
    )

    spawn_cmds: list[list[str]] = []

    class _FakeProc:
        pid = 424242

    def _capturing_popen(cmd, *a, **kw):
        spawn_cmds.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(_sp, "Popen", _capturing_popen)

    # Invoke the wired spawn closure directly (charter: wired callables
    # must be exercised). Both tiers are cached, so the closure resolves
    # and Popens on its first loop iteration -- no waiting.
    wd_llama._spawn()

    assert spawn_cmds, "llama spawn closure did not Popen a llama-server"
    argv = spawn_cmds[-1]
    assert "--model" in argv, f"llama argv lacks --model: {argv}"
    model_arg = argv[argv.index("--model") + 1]
    assert model_arg == str(small_gguf), (
        f"llama-server --model is {model_arg!r}, not the small-tier GGUF "
        f"the --config file's quant_tier=small selects: the spawn chain "
        f"is resolving quant_tier from the layered config."
    )


def test_build_watchdog_boots_on_defaults_when_explicit_config_path_is_missing(
    _watchdog_env, tmp_path, monkeypatch,
):
    """A NONEXISTENT --config path must fall back to {} + defaults
    THROUGH THE REAL LOADER and never block watchdog startup:
    load_full_config's _read_explicit_config raises RuntimeError for a
    missing file, and build_watchdog's try/except is all that stands
    between that raise and a startup abort. The pipeline child -- which
    gets the same --config forwarded -- owns the tailored fatal-config
    dialog; the watchdog's job is to keep supervising. The pre-existing
    fallback tests monkeypatch load_full_config with a zero-arg lambda
    (a TypeError, not the real RuntimeError, on the explicit path), so
    none of them prove the explicit load sits INSIDE the try.

    NO load_full_config mock here.

    Seeded defect: hoist the ``_cfg = _load_cfg(explicit_path=...)``
    call out of build_watchdog's try/except -> the RuntimeError escapes
    build_watchdog -> this test errors (red).
    """
    # Isolate from the dev machine's layered store so "defaults" is
    # deterministic.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    env = _watchdog_env
    missing = tmp_path / "does-not-exist.yaml"
    assert not missing.exists()

    # Must NOT raise despite the real loader's RuntimeError.
    app = env.module.build_watchdog(["--config", str(missing)])

    # Defaults all the way down: the default loopback endpoint keeps the
    # llama supervisor, and pipeline grace is the documented 10.0 s
    # fallback literal (same pin as the config-failure sibling test).
    names = {wd.name for wd in app.watchdogs}
    assert names == {"pipeline", "llama"}, (
        f"expected the default supervisor set after the bad --config "
        f"fallback; got {names}"
    )
    wd_pipeline = next(w for w in app.watchdogs if w.name == "pipeline")
    assert wd_pipeline._grace_s == 10.0, (
        f"bad --config fallback must land on the documented 10.0s "
        f"pipeline grace default; got {wd_pipeline._grace_s!r}"
    )
    # The bad path still rides WatchdogApp.config_path: the child gets
    # the same --config forwarded and surfaces the tailored dialog.
    assert app.config_path == missing, (
        f"WatchdogApp.config_path must carry the parsed --config even "
        f"when its load failed (the child owns the fatal-config dialog); "
        f"got {app.config_path!r}"
    )
