"""Cross-seam integration tests for the lifecycle wiring.

The old seam (HealthProbe /healthz -> ``llama_stop_event`` -> llama
Watchdog terminate) no longer exists; it was removed because it created
a feedback loop where llama's own outage tripped llama's own kill.
The two seams that REPLACED it are exercised here, each driving the
REAL production code paths end to end:

Seam 1 -- ``test_seam_livez_wedge_restarts_pipeline``
    A wedged pipeline (process alive, ``/livez`` unresponsive) must be
    *restarted*, not killed. The real ``HealthProbe.run`` observes the
    unresponsive ``/livez``, and once past its fail budget calls
    ``on_wedged`` -- wired to the real ``Watchdog.request_restart`` --
    which force-terminates the current child WITHOUT setting stop_event,
    so the real supervisor's anti-flap loop respawns it. This drives
    REAL ``HealthProbe.run`` + REAL ``Watchdog.run`` + REAL
    ``Watchdog.request_restart`` at the seam between them. The observable
    invariants are: the first child was terminated, the supervisor
    respawned (spawn count >= 2), and the MAIN stop_event stayed UNSET
    (a wedge is a restart, never a shutdown).

Seam 1b -- ``test_seam_respawn_after_live_child_gets_cold_boot_grace``
    A respawn the supervisor makes on its own (a live child exited with a
    RESTART code) must reach the probe through ``spawn_generation``, so
    the fresh child gets cold-boot grace instead of the dead child's
    armed fail timer. Drives REAL ``HealthProbe.run`` + REAL
    ``Watchdog.run``; the observable is that the second child is NOT
    terminated within many fail budgets.

Seam 2 -- ``test_seam_pipeline_giveup_cascades_to_llama_stop``
    A TERMINAL pipeline outcome (anti-flap give-up) must cascade to a
    coordinated shutdown that tears llama down too. The real pipeline
    ``Watchdog.run`` hits its max-restart give-up, runs the real
    ``_cascade_stop`` -> ``on_stop_all`` (wired, as in production's
    ``build_watchdog._stop_all``, to set the MAIN stop_event), and the
    real llama ``Watchdog.run`` -- listening on that same event -- runs
    its grace->terminate ladder and stops its child. This drives REAL
    pipeline ``Watchdog.run`` give-up + REAL ``_cascade_stop`` + REAL
    llama ``Watchdog.run`` terminate.

The fakes (``_FakeLlamaProc`` for the llama child, a per-child fake for
the pipeline) honour the ``_ProcessLike`` contract -- ``.wait(timeout)``
returns an exit code or raises ``subprocess.TimeoutExpired``;
``.terminate()`` / ``.kill()`` stop the child -- so no real subprocess is
spawned but the real Watchdog/HealthProbe state machines run unmocked.

§3.9 Wired-not-exercised guard (docs/TESTING.md). A previous seam test
in this file was deleted because it drove its own INLINE COPY of the
production ``_mirror`` closure (stop_event -> llama_stop_event) instead
of the production one -- a Wired-not-exercised Liar: a seeded defect in
the real ``run_watchdog._mirror`` would NOT have turned it red. That
mistake is NOT reintroduced here. The production ``_mirror`` is covered
end-to-end, against the real closure, by
``tests/test_build_watchdog.py::test_run_watchdog_stop_event_mirror_forwards_to_llama_stop``.
So Seam 2 deliberately does NOT hand-roll a ``_mirror`` thread: it wires
the llama supervisor DIRECTLY to the event that ``on_stop_all`` sets, and
asserts the give-up -> stop_all -> stop_event -> llama-terminate seam.
Seeded-defect drill results (documented per docs/TESTING.md §"Iron Law"
-- the evidence these tests are not Liars):

``test_seam_livez_wedge_restarts_pipeline``:
    (a) Make HealthProbe.run's budget branch a no-op (delete the
        ``self._on_wedged()`` call) -> the first child is never
        terminated and spawn count stays at 1 -> RED.
    (b) Wire ``on_wedged`` to something that sets stop_event instead of
        ``request_restart`` -> the "MAIN stop_event stayed unset"
        assertion flips -> RED (and the supervisor stops instead of
        respawning, so spawn count stays at 1 too).

``test_seam_pipeline_giveup_cascades_to_llama_stop``:
    Comment out ``self._cascade_stop()`` in the max-restart give-up
    branch of ``Watchdog.run`` -> ``on_stop_all`` never fires -> the MAIN
    stop_event is never set -> the llama Watchdog never observes a stop
    -> its child is never terminated -> RED. (This is the cascade-stop
    guard at the integration level.) CONFIRMED RED by hand.
"""
from __future__ import annotations

import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

from yamabiko.lifecycle.exit_codes import EXIT_STREAM_DEATH
from yamabiko.watchdog import HealthProbe, Watchdog


class _FakeLlamaProc:
    """Minimal process stand-in for the llama-server child.

    Watchdog._wait_or_stop treats its child via the _ProcessLike
    protocol: .wait(timeout) returns an exit code or raises
    subprocess.TimeoutExpired; .terminate() and .kill() stop the
    child. This fake mirrors those contracts.

    terminate() / kill() sets a shared Event the test uses as the
    observable "seam completed" signal. wait() blocks on that event
    up to ``timeout`` so _wait_or_stop's polling loop (which calls
    proc.wait(self._poll_interval_s) in a tight loop) doesn't spin
    CPU, and so once terminate fires the next wait returns quickly
    with the "child exited after terminate" exit code.

    ``entered_wait`` is set the first time wait() is called, i.e. once
    the llama supervisor has reached _wait_or_stop and is parked there.
    The test blocks on it (a DETERMINISTIC barrier) before kicking the
    pipeline supervisor toward give-up, so the llama Watchdog cannot
    short-circuit on the outer ``while not stop_event.is_set()`` guard
    before it has a child to terminate (a scheduling race where the
    supervisor's outer stop-event guard fires before it spawns).
    """

    def __init__(self, terminate_event: threading.Event) -> None:
        self._terminated = terminate_event
        self.entered_wait = threading.Event()

    def wait(self, timeout=None):
        # Signal that the supervisor is now parked in _wait_or_stop.
        self.entered_wait.set()
        if self._terminated.wait(timeout=timeout):
            return 1  # typical exit code after TerminateProcess
        raise subprocess.TimeoutExpired(cmd="fake-llama", timeout=timeout)

    def terminate(self) -> None:
        self._terminated.set()

    def kill(self) -> None:
        self._terminated.set()


class _FakePipelineProc:
    """Per-child stand-in for a pipeline child the supervisor restarts.

    Honours _ProcessLike: .wait(timeout) raises subprocess.TimeoutExpired
    (the child is "alive and busy") until .terminate()/.kill() is called,
    after which .wait() returns ``exit_code`` (default 1, a RESTART per
    lifecycle.exit_codes.classify). Each instance owns its OWN terminate
    Event so the test can observe the first child's terminate
    independently of any later respawn.
    """

    def __init__(self, exit_code: int = 1) -> None:
        self._exit_code = exit_code
        self.terminate_called = threading.Event()

    def wait(self, timeout=None):
        if self.terminate_called.wait(timeout=timeout):
            return self._exit_code
        raise subprocess.TimeoutExpired(cmd="fake-pipeline", timeout=timeout)

    def terminate(self) -> None:
        self.terminate_called.set()

    def kill(self) -> None:
        self.terminate_called.set()


def test_seam_livez_wedge_restarts_pipeline():
    """Wedged /livez -> HealthProbe budget -> request_restart -> the real
    pipeline supervisor RESTARTS the child (does NOT shut down).

    Drives REAL HealthProbe.run + REAL Watchdog.run + REAL
    Watchdog.request_restart at the seam. Observable invariants:
      1. the first pipeline child's terminate() fired (request_restart
         reached the live child),
      2. the supervisor respawned (spawn count >= 2) -- a RESTART, not a
         stop,
      3. the MAIN stop_event was NOT set (wedge != shutdown).

    Seeded-defect drills (documented in the module docstring):
      (a) delete self._on_wedged() in HealthProbe.run -> child never
          terminated, spawn count stays 1 -> RED.
      (b) wire on_wedged to set stop_event instead of request_restart ->
          invariant #3 (stop_event unset) flips -> RED.
    """
    main_stop_event = threading.Event()  # the MAIN supervisor stop_event
    spawned: list[_FakePipelineProc] = []
    first_spawn_done = threading.Event()  # deterministic barrier (see below)

    def spawn() -> _FakePipelineProc:
        proc = _FakePipelineProc(exit_code=1)
        spawned.append(proc)
        first_spawn_done.set()
        return proc

    wd_pipeline = Watchdog(
        name="pipeline",
        spawn=spawn,
        clock=time.monotonic,
        sleep=lambda s: None,
        stop_event=main_stop_event,
        # min_lifetime_s=0.0 so a request_restart'd child (which "dies"
        # nonzero almost immediately under the test's fast clock) is
        # classified RESTART, not a fast-crash give-up.
        min_lifetime_s=0.0,
        # Immediate respawn: no backoff sleep (sleep is a no-op anyway).
        backoff_s=(0.0,),
        max_restarts=10,
        restart_window_s=600.0,
        poll_interval_s=0.01,
    )
    pipeline_thread = threading.Thread(target=wd_pipeline.run, daemon=True)

    with patch("yamabiko.watchdog.loopback_http") as mock_httpx:
        # HealthProbe arms its budget only AFTER the first 200. Emit one
        # 200 (arm), then a long run of 503s (wedge) so the budget
        # exhausts and on_wedged fires. The count is generous so short
        # poll jitter can't run the side_effect dry before the wedge.
        mock_httpx.get.side_effect = (
            [MagicMock(status_code=200)]
            + [MagicMock(status_code=503)] * 10000
        )
        probe = HealthProbe(
            url="http://127.0.0.1:8765/livez",
            stop_event=main_stop_event,
            on_wedged=wd_pipeline.request_restart,
            spawn_generation=wd_pipeline.spawn_generation,
            startup_dialog_up=lambda: False,
            poll_interval_s=0.01,
            fail_budget_s=0.02,
        )
        probe_thread = threading.Thread(target=probe.run, daemon=True)

        try:
            pipeline_thread.start()
            # Deterministic barrier (matches seam-2's entered_wait, replaces
            # a prior time.sleep poll): block until the supervisor has
            # spawned its first child before starting the probe. Without a
            # live _current_proc, request_restart is a documented no-op; the
            # supervisor assigns _current_proc on the line immediately after
            # spawn() returns, and the probe's first trip is >= fail_budget_s
            # (20 ms) after it starts, so _current_proc is reliably set by
            # the time on_wedged could fire.
            assert first_spawn_done.wait(timeout=5.0), (
                "pipeline supervisor never spawned a child"
            )
            first = spawned[0]

            probe_thread.start()

            # ORDERING IS LOAD-BEARING (do not reorder). Invariant 2's
            # "spawn >= 2" respawn observation MUST be reached before
            # invariant 3 inspects main_stop_event AND before the finally
            # block sets main_stop_event. The supervisor only respawns
            # while stop_event is unset; if the finally (or a moved
            # invariant 3) tripped main_stop_event before the respawn was
            # observed, the supervisor would exit instead of respawn and
            # invariant 2 would flake.

            # Invariant 1: the FIRST child gets terminated by the
            # request_restart the wedge triggers.
            assert first.terminate_called.wait(timeout=5.0), (
                "seam broken: /livez wedge never reached the pipeline "
                "child's terminate(). Check HealthProbe.run calls "
                "on_wedged past its fail budget, AND that on_wedged is "
                "wired to Watchdog.request_restart, AND that "
                "request_restart terminates the current child."
            )

            # Invariant 2: the wedge caused a RESTART -- the supervisor
            # respawned at least once more.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if len(spawned) >= 2:
                    break
                time.sleep(0.005)
            assert len(spawned) >= 2, (
                "the pipeline supervisor did not respawn after the wedge "
                f"(spawn count={len(spawned)}). A wedge must RESTART the "
                "pipeline, not stop it."
            )

            # Invariant 3: a wedge is a restart, not a shutdown. The MAIN
            # stop_event must NOT have been set by this path.
            assert not main_stop_event.is_set(), (
                "MAIN stop_event was set by the wedge path -- a wedge must "
                "RESTART the pipeline (request_restart), never SHUT DOWN "
                "the supervisor (stop_event)."
            )
        finally:
            main_stop_event.set()
            # Unblock any child currently in _wait_or_stop's grace wait so
            # the supervisor thread can exit promptly.
            for p in spawned:
                p.terminate()
            probe_thread.join(timeout=2.0)
            pipeline_thread.join(timeout=2.0)


def test_seam_respawn_after_live_child_gets_cold_boot_grace():
    """A live pipeline that dies on its own (here stream death, 44) is
    respawned by the supervisor, and the fresh child must get its own
    cold-boot grace. It cannot serve /livez until build_pipeline
    finishes, and a tailored startup dialog (no audio source, GPU
    missing) can hold it there. The probe used to carry the dead child's
    "seen live" state over, so its fail budget ran from the OLD child's
    death and it terminated the new child mid-boot: the dialog vanished
    under the user and the exit became 1 instead of 43.

    Drives REAL HealthProbe.run + REAL Watchdog.run, wired the way
    run_watchdog wires them (spawn_generation is the supervisor's).

    Seeded defect: drop the spawn-generation reset in HealthProbe.run ->
    the probe trips fail_budget_s after the first child exits and
    terminates the second child -> red.
    """
    main_stop_event = threading.Event()
    spawned: list[_FakePipelineProc] = []
    first_spawn_done = threading.Event()
    second_spawn_done = threading.Event()
    live_polls = [0]
    first_seen_live = threading.Event()

    def spawn() -> _FakePipelineProc:
        proc = _FakePipelineProc(exit_code=EXIT_STREAM_DEATH)
        spawned.append(proc)
        (first_spawn_done if len(spawned) == 1 else second_spawn_done).set()
        return proc

    def fake_get(*a, **k):
        # Only the first child ever serves /livez; every later child is
        # still booting (or parked on a startup dialog).
        if len(spawned) == 1 and not spawned[0].terminate_called.is_set():
            live_polls[0] += 1
            if live_polls[0] >= 2:
                first_seen_live.set()
            return MagicMock(status_code=200)
        return MagicMock(status_code=503)

    wd_pipeline = Watchdog(
        name="pipeline",
        spawn=spawn,
        clock=time.monotonic,
        sleep=lambda s: None,
        stop_event=main_stop_event,
        min_lifetime_s=0.0,
        backoff_s=(0.0,),
        max_restarts=10,
        restart_window_s=600.0,
        poll_interval_s=0.01,
    )
    pipeline_thread = threading.Thread(target=wd_pipeline.run, daemon=True)

    with patch("yamabiko.watchdog.loopback_http") as mock_httpx:
        mock_httpx.get.side_effect = fake_get
        probe = HealthProbe(
            url="http://127.0.0.1:8765/livez",
            stop_event=main_stop_event,
            on_wedged=wd_pipeline.request_restart,
            spawn_generation=wd_pipeline.spawn_generation,
            startup_dialog_up=lambda: False,
            poll_interval_s=0.01,
            fail_budget_s=0.05,
            startup_budget_s=60.0,
        )
        probe_thread = threading.Thread(target=probe.run, daemon=True)
        try:
            pipeline_thread.start()
            assert first_spawn_done.wait(timeout=5.0), (
                "pipeline supervisor never spawned a child"
            )
            probe_thread.start()
            # Two served 200s: the first one has been fully processed, so
            # the probe has seen this child live and armed its budget.
            assert first_seen_live.wait(timeout=5.0), (
                "probe never polled the live first child"
            )

            # The first child exits on its own with a RESTART code.
            spawned[0].terminate()
            assert second_spawn_done.wait(timeout=5.0), (
                "supervisor did not respawn after the first child exited"
            )
            second = spawned[1]

            # Twenty fail budgets of a non-serving new child, well inside
            # its startup budget: the probe must leave it alone.
            killed = second.terminate_called.wait(timeout=1.0)
            assert not killed, (
                "the probe terminated the freshly respawned pipeline during "
                "its cold boot: the previous child's seen-live state and "
                "fail timer carried over to the new child"
            )
        finally:
            main_stop_event.set()
            for p in spawned:
                p.terminate()
            probe_thread.join(timeout=2.0)
            pipeline_thread.join(timeout=2.0)


def test_seam_pipeline_giveup_cascades_to_llama_stop():
    """A TERMINAL pipeline give-up cascades through the real
    _cascade_stop -> on_stop_all (sets MAIN stop_event) and the real
    llama Watchdog -- listening on that event -- terminates its child.

    Drives REAL pipeline Watchdog.run (max-restart give-up) + REAL
    _cascade_stop + REAL llama Watchdog.run terminate.

    The llama supervisor is wired DIRECTLY to the event that on_stop_all
    sets (NOT via a hand-rolled _mirror thread). In production the llama
    supervisor listens on ``llama_stop_event`` and run_watchdog's
    ``_mirror`` thread forwards main stop_event -> llama_stop_event; that
    mirror leg is separately and fully covered, against the PRODUCTION
    closure, by
    ``tests/test_build_watchdog.py::test_run_watchdog_stop_event_mirror_forwards_to_llama_stop``.
    Hand-rolling an inline _mirror here would be the §3.9
    Wired-not-exercised Liar this file's history warns about, so we wire
    the llama supervisor to the stop_all'd event directly and assert the
    give-up -> stop_all -> stop_event -> llama-terminate seam.

    Seeded-defect drill (documented in the module docstring): comment out
    ``self._cascade_stop()`` in the max-restart give-up branch of
    Watchdog.run -> on_stop_all never fires -> stop_event never set ->
    llama never terminates -> RED. This is the cascade-stop guard at the
    integration level.
    """
    main_stop_event = threading.Event()  # the MAIN stop_event
    give_up_calls: list[tuple[str, str]] = []

    def _stop_all() -> None:
        # Mirrors build_watchdog._stop_all: a terminal outcome sets the
        # MAIN stop_event, which production's _mirror forwards to
        # llama_stop_event. Here the llama supervisor listens on this
        # same event directly (see module docstring rationale).
        main_stop_event.set()

    # The pipeline supervisor gives up via the MAX-RESTART path: each
    # child crashes nonzero but lives >= min_lifetime (so it's a RESTART,
    # not a fast-crash give-up), and the clock advances enough per call
    # that two crashes inside the window blow the max_restarts=1 cap.
    clock_t = [0.0]

    def pipeline_clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    def pipeline_spawn() -> _FakePipelineProc:
        # exit_code=1 (RESTART); terminate-event pre-set so _wait_or_stop's
        # proc.wait() returns 1 immediately -> the supervisor loops fast.
        proc = _FakePipelineProc(exit_code=1)
        proc.terminate_called.set()
        return proc

    wd_pipeline = Watchdog(
        name="pipeline",
        spawn=pipeline_spawn,
        clock=pipeline_clock,
        sleep=lambda s: None,
        # NB: NOT wired to main_stop_event -- it must run to give-up under
        # its own steam; the cascade is what sets main_stop_event.
        on_stop_all=_stop_all,
        on_give_up=lambda t, d: give_up_calls.append((t, d)),
        max_restarts=1,
        restart_window_s=600.0,
        min_lifetime_s=5.0,
        poll_interval_s=0.01,
    )

    # Real llama supervisor, listening on the SAME event on_stop_all sets.
    llama_terminate = threading.Event()
    fake_llama = _FakeLlamaProc(llama_terminate)
    wd_llama = Watchdog(
        name="llama",
        spawn=lambda: fake_llama,
        clock=time.monotonic,
        sleep=lambda s: None,
        stop_event=main_stop_event,
        grace_s=0.1,          # short grace so terminate fires fast
        min_lifetime_s=0.0,
        poll_interval_s=0.01,
    )
    llama_thread = threading.Thread(target=wd_llama.run, daemon=True)
    pipeline_thread = threading.Thread(target=wd_pipeline.run, daemon=True)

    try:
        # Start llama first and let it reach _wait_or_stop (its child is
        # "alive": wait() raises TimeoutExpired until terminate). Block on
        # a DETERMINISTIC barrier until the llama supervisor is actually
        # PARKED in _wait_or_stop before kicking the pipeline toward
        # give-up. Without this barrier the pipeline (sleep=no-op +
        # advancing clock) can set main_stop_event BEFORE the llama
        # supervisor's outer ``while not stop_event.is_set()`` guard runs,
        # so the llama Watchdog would exit without ever spawning a child to
        # terminate and llama_terminate would never fire (the same
        # scheduling race: supervisor exits before spawning a child).
        llama_thread.start()
        assert fake_llama.entered_wait.wait(timeout=2.0), (
            "llama supervisor never reached _wait_or_stop; barrier failed"
        )
        # Now the llama supervisor is committed to a child in its wait
        # loop; driving the pipeline to give-up can safely set the event.
        pipeline_thread.start()

        # Observable 1: the real _cascade_stop -> on_stop_all set the MAIN
        # stop_event.
        assert main_stop_event.wait(timeout=5.0), (
            "seam broken: the pipeline give-up never set the MAIN "
            "stop_event. Check Watchdog.run's max-restart branch calls "
            "self._cascade_stop(), and that _cascade_stop invokes "
            "on_stop_all."
        )
        # And the give-up actually fired (the terminal path was taken, not
        # some other early return).
        assert len(give_up_calls) == 1, (
            f"expected exactly one give-up, got {len(give_up_calls)}; the "
            "pipeline supervisor did not take the max-restart give-up path."
        )

        # Observable 2: the llama supervisor observed the stop_event and
        # ran its grace->terminate ladder on its child.
        assert llama_terminate.wait(timeout=5.0), (
            "seam broken: the MAIN stop_event was set but the llama "
            "Watchdog never terminated its child. Check the llama "
            "supervisor listens on the cascaded event and that "
            "_wait_or_stop falls through to terminate after grace."
        )
    finally:
        main_stop_event.set()
        llama_terminate.set()
        pipeline_thread.join(timeout=2.0)
        llama_thread.join(timeout=2.0)
