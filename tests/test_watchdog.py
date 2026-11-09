"""Watchdog supervises main.py as a subprocess.

Tests use a fake subprocess factory so we don't actually spawn Python.
The supervisor's decision tree is exercised directly.
"""
from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from yamabiko import __version__, watchdog
from yamabiko.config import ConfigValidationError
from yamabiko.lifecycle.exit_codes import STATUS_CONTROL_C_EXIT


class _FakeProcess:
    """Records start/wait/terminate; exits with configured code + elapsed."""

    def __init__(self, exit_code: int = 0, elapsed_s: float = 10.0) -> None:
        self.exit_code = exit_code
        self.elapsed_s = elapsed_s
        self.started_at: float = 0.0
        self.terminated = False

    def wait(self, timeout=None) -> int:
        # Tests don't simulate wait timeout; always return the configured code.
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True


def test_watchdog_policy_calls_spawn_once_per_clean_run():
    """A clean child exit (0 -> STOP_SELF) means run() invokes the spawn
    factory exactly once -- no respawn after an intentional exit.

    POLICY-LAYER test. Uses _FakeProcess, not real subprocess.Popen.
    The OS-level "does subprocess.Popen actually start a child" seam
    is covered by tests/test_real_watchdog_subprocess.py at the
    @pytest.mark.real tier.

    Seeded-defect check: make the STOP_SELF branch
    in Watchdog.run() unreachable (``if policy is ExitPolicy.STOP_SELF``
    -> ``if False``) -> the clean exit is treated as RESTART and the
    supervisor respawns to the anti-flap cap -> len(spawns) == 4 -> red.
    The clock advances past min_lifetime_s per call so the misrouted
    clean exit cannot hide behind the fast-crash give-up.
    """
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=0, elapsed_s=10.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    wd = watchdog.Watchdog(spawn=spawn, clock=clock, sleep=lambda s: None,
                           min_lifetime_s=5.0)
    wd.run()
    assert len(spawns) == 1


def test_watchdog_policy_retries_on_nonzero_exit():
    """After a child exits nonzero beyond min_lifetime_s, the policy
    respawns once more. POLICY-LAYER (_FakeProcess). OS-level behaviour
    is covered by test_real_watchdog_subprocess.py.

    Seeded-defect check: change Watchdog.run() to exit on any nonzero
    return (no retry) -> this test fails.
    """
    exit_codes = deque([1, 0])  # first crash, then clean exit
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=exit_codes.popleft(), elapsed_s=10.0)
        spawns.append(p)
        return p

    clock_t = [0.0]
    def clock() -> float:
        clock_t[0] += 10.0  # each wait advances time 10 s
        return clock_t[0]

    wd = watchdog.Watchdog(spawn=spawn, clock=clock, sleep=lambda s: None,
                           min_lifetime_s=5.0)
    wd.run()
    assert len(spawns) == 2


def test_watchdog_policy_stops_on_fast_crash_flap():
    """If the process dies within 5 s, treat as config-error and stop.

    POLICY-LAYER.

    Seeded-defect check: remove the ``lifetime < min_lifetime_s`` branch
    from Watchdog.run() -> this test respawns N times instead of
    stopping after 1, fails the assertion.
    """
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=1, elapsed_s=1.0)
        spawns.append(p)
        return p

    clock_t = [0.0]
    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    wd = watchdog.Watchdog(spawn=spawn, clock=clock, sleep=lambda s: None,
                           min_lifetime_s=5.0)
    wd.run()
    assert len(spawns) == 1  # no restart on fast crash


def test_watchdog_stream_death_within_min_lifetime_restarts_not_giveup():
    """EXIT_STREAM_DEATH (capture device died) within min_lifetime_s must
    RESTART (so the child re-resolves the device), NOT hit the fast-crash
    'configuration error' give-up. By construction code 44 fires only AFTER
    capture started (the stream-death monitor and idle re-open are wired
    post-start; idle also needs _ever_audible), so a quick death means the
    device dropped, not that the pipeline can't configure -- the min-lifetime
    config-error heuristic mis-attributes it.

    Regression for the review's P1: pre-fix a device unplug in the first ~30 s
    showed 'Yamabiko failed to start / configuration error' and stopped, where
    pre-PR it was a clean exit 0.

    Seeded defect: drop the `and code != EXIT_STREAM_DEATH` exemption from the
    `elapsed < min_lifetime_s` branch -> a fast stream-death gives up after 1
    spawn (and fires the config-error dialog) -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STREAM_DEATH

    # First spawn dies fast with 44; second exits clean (0) so the loop ends.
    codes = deque([EXIT_STREAM_DEATH, 0])
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=codes.popleft(), elapsed_s=1.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(spawns) == 2, (
        f"a fast EXIT_STREAM_DEATH must RESTART (re-resolve the device), not "
        f"give up after 1 spawn; got {len(spawns)} spawns"
    )
    assert giveups == [], (
        f"a fast device death must NOT show the startup config-error give-up; "
        f"got {giveups!r}"
    )


def test_fast_control_c_exit_gives_up_with_external_stop_copy_not_config_error():
    """A child that dies within min_lifetime_s with STATUS_CONTROL_C_EXIT
    (0xC000013A: console close / Ctrl+C / console-group signal) was
    terminated from OUTSIDE Yamabiko -- describing it as "a configuration
    error" misdirects the user at their config. Behavior is unchanged
    (terminal give-up, no restart: restarting would fight whoever stopped
    it); only the copy names the real cause.

    Field regression: llama-server was observed dying at 2.3s and 12.3s with
    raw code 3221225786 (a tester closed its then-visible console window,
    since removed by CREATE_NO_WINDOW) while the dialog blamed "a
    configuration error". The positive 3221225786 in those logs is also the
    proof that Popen.returncode carries the UNSIGNED DWORD on Windows, so
    comparing against 0xC000013A is correct.

    Seeded-defect check: drop the is_external_stop() branch from the
    fast-crash path -> the give-up message contains "configuration error"
    again -> red.
    """
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=STATUS_CONTROL_C_EXIT, elapsed_s=1.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(spawns) == 1, (
        f"an externally-signaled fast death is still a terminal give-up "
        f"(no restart); got {len(spawns)} spawns"
    )
    assert len(giveups) == 1
    _title, message = giveups[0]
    assert "configuration error" not in message.lower(), (
        f"externally-signaled death must not be blamed on configuration; "
        f"got {message!r}"
    )
    assert "closed" in message.lower(), (
        f"give-up copy should say the process was closed from outside "
        f"Yamabiko; got {message!r}"
    )


def test_repeated_external_stops_give_up_without_claiming_yamabiko_crashed():
    """An external stop that arrives AFTER min_lifetime_s is RESTARTED (the
    user may not have meant to kill it), so repeated ones exhaust the anti-flap
    budget and land in the OTHER give-up surface. That surface must not tell the
    user Yamabiko "keeps crashing" and send them hunting for a traceback that
    does not exist -- the process was stopped from outside, exactly as in the
    fast path above.

    Gap found in review: only the fast-crash path special-cased the external
    stop, so a long-lived child killed from outside three times still produced
    the crash copy.

    Seeded-defect check: drop the is_external_stop() branch from the anti-flap
    give-up -> the message says "keeps crashing" / points at tracebacks -> red.
    """
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        # elapsed 60 s >> min_lifetime, so each death is a normal RESTART that
        # accrues against the anti-flap budget rather than a fast-crash give-up.
        p = _FakeProcess(exit_code=STATUS_CONTROL_C_EXIT, elapsed_s=60.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        max_restarts=2, restart_window_s=600.0, min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(giveups) == 1, f"expected one give-up; got {giveups!r}"
    title, message = giveups[0]
    blob = f"{title} {message}".lower()
    assert "crash" not in blob, (
        f"a repeated EXTERNAL stop must not be reported as a crash; got "
        f"title={title!r} message={message!r}"
    )
    assert "closed" in blob, (
        f"give-up copy should say the process was closed from outside "
        f"Yamabiko; got {message!r}"
    )


def test_watchdog_device_specific_giveup_when_stream_death_flap_exhausts_budget():
    """When repeated EXIT_STREAM_DEATH restarts exhaust the anti-flap budget,
    the give-up dialog must be device-specific ('Audio device keeps dropping'),
    not the generic 'Yamabiko keeps crashing' -- a flapping audio device is not
    a code crash, and the design doc listed device-aware copy as a deliverable.

    Seeded defect: drop the `if code == EXIT_STREAM_DEATH` branch in the
    max-restarts give-up -> the generic crash title fires -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STREAM_DEATH

    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        # elapsed 60 s >> min_lifetime so each death is a clean RESTART (not the
        # fast-crash branch) that accrues against the anti-flap budget.
        p = _FakeProcess(exit_code=EXIT_STREAM_DEATH, elapsed_s=60.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        max_restarts=2, restart_window_s=600.0, min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert giveups, "expected a give-up after the device-death flap"
    title, detail = giveups[0]
    assert "device" in title.lower(), (
        f"a give-up after a device-death flap must be device-specific, not the "
        f"generic crash message; got title {title!r}"
    )


def test_watchdog_stage_escalation_within_min_lifetime_restarts_not_giveup():
    """EXIT_STAGE_ESCALATION (45: a pipeline stage crossed its supervisor
    escalation threshold) within min_lifetime_s must RESTART, NOT hit the
    fast-crash 'configuration error' give-up. By construction code 45 fires
    only AFTER the pipeline started (stages run post-start, and failures must
    accumulate to the threshold), so a quick 45 means a runtime stage
    crash-loop, not that the pipeline can't configure.

    This is not a theoretical timing: the supervisor escalates at 10 errors
    in a ROLLING 60 s window (supervisor.py: stage_window_s=60,
    escalation_threshold=10), so a deterministically-failing stage at ~2 s
    segment cadence escalates in ~20 s -- INSIDE the 30 s default
    min-lifetime. Pre-fix the watchdog showed the misleading 'Yamabiko
    failed to start / configuration error' give-up and cascade-stopped on
    the FIRST escalation; the documented restart -> startup-sanity-check ->
    tailored-43 chain never ran.

    Seeded defect: revert the fast-crash exemption in Watchdog.run from
    `code not in (EXIT_STREAM_DEATH, EXIT_STAGE_ESCALATION)` back to
    `code != EXIT_STREAM_DEATH` -> a fast escalation gives up after 1 spawn
    (and fires the config-error dialog) -> red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STAGE_ESCALATION

    # First spawn dies fast with 45; second exits clean (0) so the loop ends.
    codes = deque([EXIT_STAGE_ESCALATION, 0])
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=codes.popleft(), elapsed_s=1.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(spawns) == 2, (
        f"a fast EXIT_STAGE_ESCALATION must RESTART (respawn the pipeline so "
        f"the startup sanity check can run), not give up after 1 spawn; got "
        f"{len(spawns)} spawns"
    )
    assert giveups == [], (
        f"a fast stage escalation must NOT show the startup config-error "
        f"give-up; got {giveups!r}"
    )


def test_watchdog_generic_giveup_when_fast_stage_escalation_flap_exhausts_budget():
    """Repeated fast EXIT_STAGE_ESCALATION exits -- each landing within
    min_lifetime_s, the exact cadence of a deterministically-failing stage
    re-escalating ~20 s after every respawn -- must accrue against the
    anti-flap budget and exhaust it into the GENERIC 'Yamabiko keeps
    crashing' give-up (the correct copy for a runtime stage crash-loop),
    never the 'Yamabiko failed to start / configuration error' copy and
    never the device-specific copy. Pins the documented chain end-state:
    escalation -> restart -> re-escalation -> budget exhausted -> generic
    give-up.

    Seeded defects:
      - revert the fast-crash exemption to `code != EXIT_STREAM_DEATH` ->
        the FIRST fast 45 fires 'Yamabiko failed to start' after 1 spawn ->
        both the spawn-count and title assertions go red.
      - broaden the device-copy branch in the max-restarts give-up from
        `code == EXIT_STREAM_DEATH` to `code in (44, 45)` -> the stage flap
        gets the audio-device title -> title assertion red.
    """
    from yamabiko.lifecycle.exit_codes import EXIT_STAGE_ESCALATION

    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        # elapsed 1 s << min_lifetime 5 s: every escalation lands inside the
        # fast-crash window, the shape of the shipped bug.
        p = _FakeProcess(exit_code=EXIT_STAGE_ESCALATION, elapsed_s=1.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        max_restarts=2, restart_window_s=600.0, min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(spawns) == 3, (
        f"expected max_restarts(2)+1 = 3 spawns before the anti-flap give-up; "
        f"got {len(spawns)} -- 1 means the fast-crash config-error branch "
        f"swallowed the first escalation instead of restarting"
    )
    assert len(giveups) == 1, f"expected exactly one give-up; got {giveups!r}"
    title, _detail = giveups[0]
    assert "keeps crashing" in title.lower(), (
        f"a stage-escalation flap must exhaust the restart budget into the "
        f"generic crash give-up, not the startup/config-error or "
        f"audio-device copy; got title {title!r}"
    )


def test_watchdog_policy_respects_max_restarts_cap():
    """3 restarts in 10 min, then stop.

    POLICY-LAYER.

    Seeded-defect check: change the restart-limit comparison in
    Watchdog.run() from < max_restarts to <= max_restarts -> 5
    spawns instead of 4, this test fails.
    """
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=1, elapsed_s=60.0)  # 60 s lifetime each
        spawns.append(p)
        return p

    clock_t = [0.0]
    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    wd = watchdog.Watchdog(spawn=spawn, clock=clock, sleep=lambda s: None,
                           max_restarts=3, restart_window_s=600.0,
                           min_lifetime_s=5.0)
    wd.run()
    # 1 initial + 3 restarts = 4 total spawns before giving up
    assert len(spawns) == 4


def test_watchdog_backoff_sequence():
    delays: list[float] = []

    def spawn() -> _FakeProcess:
        return _FakeProcess(exit_code=1, elapsed_s=60.0)

    clock_t = [0.0]
    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock,
        sleep=lambda s: delays.append(s),
        max_restarts=3, restart_window_s=600.0,
        min_lifetime_s=5.0, backoff_s=(1.0, 5.0, 15.0),
    )
    wd.run()
    assert delays == [1.0, 5.0, 15.0]


def test_watchdog_stop_during_backoff_exits_immediately():
    """A stop request that lands during the crash-restart backoff
    must wake the supervisor IMMEDIATELY -- not after the full backoff
    tier (up to 15 s in production) has slept off. The single-instance
    mutex is held until run() returns, so an uninterruptible backoff
    blocks a relaunch for the whole delay.

    Drives the PRODUCTION backoff wait (no ``sleep`` injected): the
    fake clock's third tick -- the ``now = self._clock()`` read that
    immediately precedes the backoff -- sets stop_event, exactly like a
    SIGINT arriving as the backoff starts. With the fix the default
    backoff wait is ``stop_event.wait(timeout=delay)``, which returns
    at once; run() then exits at the loop head with no second spawn.

    Fails red if: the backoff branch sleeps via bare ``time.sleep``
    (the unfixed default) -- the 30 s backoff blocks past the join
    timeout below.

    Seeded-defect drill: revert the default ``self._sleep`` to
    ``time.sleep`` in Watchdog.__init__ -> the run thread is still
    alive 5 s after stop_event was set -> red.
    """
    import threading

    stop_event = threading.Event()
    spawns: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=1, elapsed_s=60.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        # Calls per iteration: (1) start, (2) elapsed read, (3) the
        # ``now`` read just before the anti-flap window + backoff.
        clock_t[0] += 100.0
        if clock_t[0] >= 300.0:
            # Third tick: stop request arrives right before the
            # backoff wait begins.
            stop_event.set()
        return clock_t[0]

    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock,
        # No ``sleep`` injected: exercise the PRODUCTION backoff wait.
        max_restarts=3, restart_window_s=600.0,
        min_lifetime_s=5.0, backoff_s=(30.0,),
        stop_event=stop_event,
    )

    t = threading.Thread(target=wd.run, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), (
        "Watchdog.run() was still inside the 30 s backoff 5 s after "
        "stop_event was set; the backoff wait is not interruptible -- "
        "a tray-Quit/SIGINT during backoff holds the single-instance "
        "mutex for the full backoff tier."
    )
    assert len(spawns) == 1, (
        f"expected no respawn after a stop request during backoff; got "
        f"{len(spawns)} spawns"
    )


def test_watchdog_policy_tolerates_spawn_raising():
    """When spawn() raises (e.g. FileNotFoundError for a bad exe path),
    Watchdog.run() logs + returns cleanly instead of propagating.

    POLICY-LAYER.

    Seeded-defect check: remove the try/except around self._spawn()
    in Watchdog.run() -> FileNotFoundError propagates out of
    wd.run() instead of being caught, this test fails with
    unhandled exception.
    """
    def spawn():
        raise FileNotFoundError("no such executable")

    wd = watchdog.Watchdog(
        spawn=spawn,
        clock=lambda: 0.0,
        sleep=lambda s: None,
    )
    # Should NOT raise out of run(); logs and returns cleanly.
    wd.run()


def test_on_give_up_fires_on_crash_loop_cap():
    calls: list[tuple[str, str]] = []

    def spawn() -> _FakeProcess:
        return _FakeProcess(exit_code=1, elapsed_s=60.0)

    clock_t = [0.0]
    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        max_restarts=2, restart_window_s=600.0, min_lifetime_s=5.0,
        on_give_up=lambda title, detail: calls.append((title, detail)),
    )
    wd.run()
    assert len(calls) == 1
    assert "keeps crashing" in calls[0][0].lower()


def test_print_version_and_exit(capsys):
    """The helper prints the version banner and raises SystemExit(0)."""
    with pytest.raises(SystemExit) as exc_info:
        watchdog._print_version_and_exit()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"Yamabiko {__version__}" in captured.out


def test_main_short_circuits_on_version_flag(monkeypatch, capsys):
    """``main()`` handles --version before any heavy imports or spawning.

    If the short-circuit regresses, this test would fail loudly because
    main() would try to set up logging / signals / spawn subprocesses.
    We actively prove this by wiring booby traps: signal.signal and
    yamabiko.log_config.setup_main_log both raise if called.
    """
    import sys as _sys
    import types as _types
    import signal as _signal

    def _boom(*a, **kw):
        raise AssertionError("called before --version short-circuit")

    monkeypatch.setattr(_signal, "signal", _boom)
    monkeypatch.setitem(
        _sys.modules,
        "yamabiko.log_config",
        _types.SimpleNamespace(setup_main_log=_boom),
    )
    monkeypatch.setattr(watchdog.sys, "argv", ["yamabiko.exe", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        watchdog.main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"Yamabiko {__version__}" in captured.out


def test_main_short_circuits_on_run_main_flag(monkeypatch):
    """``main()`` dispatches to yamabiko.main.main() when --run-main is in argv.

    Also proves the normal watchdog path (signal handlers, Watchdog.run)
    is NOT taken by booby-trapping signal.signal — identical pattern to
    the --version short-circuit test.
    """
    import sys as _sys
    import types as _types
    import signal as _signal

    calls: list[str] = []

    def _fake_main() -> None:
        calls.append("main")

    def _boom(*a, **kw):
        raise AssertionError("called before --run-main short-circuit")

    # Stub yamabiko.main in sys.modules with a sentinel so watchdog's
    # `from yamabiko import main as _main_mod` picks up our fake instead
    # of importing the real module (which pulls in heavy audio deps).
    # Also booby-trap signal.signal to prove the normal watchdog path
    # isn't taken.
    #
    # NOTE: `from yamabiko import main` returns the `main` attribute on
    # the `yamabiko` package when it already exists (which it does if any
    # earlier test in the suite imported `yamabiko.main`), NOT
    # sys.modules["yamabiko.main"]. So we must stub BOTH: the sys.modules
    # entry (for the case where yamabiko.main hasn't been imported) AND
    # the attribute on the yamabiko package (for the full-suite case).
    import yamabiko as _yamabiko_pkg
    _fake_main_mod = _types.SimpleNamespace(main=_fake_main)
    monkeypatch.setitem(_sys.modules, "yamabiko.main", _fake_main_mod)
    monkeypatch.setattr(
        _yamabiko_pkg, "main", _fake_main_mod, raising=False
    )
    monkeypatch.setattr(_signal, "signal", _boom)
    monkeypatch.setattr(watchdog.sys, "argv", ["yamabiko.exe", "--run-main"])

    # Should return cleanly (no SystemExit, no signal handler install).
    watchdog.main()
    assert calls == ["main"]


def test_spawn_cmd_frozen_vs_dev(monkeypatch):
    """_build_main_cmd branches on sys.frozen.

    argv is pinned to a bare invocation so the exact-equality asserts
    are hermetic (the forwarding loop reads sys.argv; pytest's own
    argv must not leak into the command under test).
    """
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/python.exe")
    monkeypatch.setattr(watchdog.sys, "argv", ["yamabiko.exe"])

    # Dev mode: no sys.frozen attribute -> -m invocation.
    monkeypatch.delattr(watchdog.sys, "frozen", raising=False)
    assert watchdog._build_main_cmd() == [
        "C:/fake/python.exe", "-m", "yamabiko.main"
    ]

    # Frozen mode: sys.frozen True -> --run-main re-entry.
    monkeypatch.setattr(watchdog.sys, "frozen", True, raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/yamabiko.exe")
    assert watchdog._build_main_cmd() == [
        "C:/fake/yamabiko.exe", "--run-main"
    ]


def test_build_main_cmd_forwards_max_segments_and_config(monkeypatch):
    """--max-segments and --config from the user's top-level
    yamabiko.exe invocation must ride the spawned pipeline child's argv,
    in BOTH dev and frozen branches. Deleting the forwarding from
    _build_main_cmd silently drops the flag -- an untested seam.

    --config is expected in the canonical re-emitted form
    ``["--config", str(parsed_path)]``: the child must receive the SAME
    parsed result the watchdog's own planning uses (_parse_config_path),
    not a raw token copy.

    Seeded-defect drill: delete the --max-segments forwarding loop or
    the --config re-emit from _build_main_cmd -> the exact-equality
    asserts go red.
    """
    from pathlib import Path as _P

    monkeypatch.setattr(
        watchdog.sys, "argv",
        ["yamabiko.exe", "--max-segments", "5", "--config", "C:/x.yaml"],
    )

    # Dev branch.
    monkeypatch.delattr(watchdog.sys, "frozen", raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/python.exe")
    assert watchdog._build_main_cmd() == [
        "C:/fake/python.exe", "-m", "yamabiko.main",
        "--max-segments", "5", "--config", str(_P("C:/x.yaml")),
    ]

    # Frozen branch.
    monkeypatch.setattr(watchdog.sys, "frozen", True, raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/yamabiko.exe")
    assert watchdog._build_main_cmd() == [
        "C:/fake/yamabiko.exe", "--run-main",
        "--max-segments", "5", "--config", str(_P("C:/x.yaml")),
    ]


def test_build_main_cmd_forwards_equals_form_and_abbreviated_config(
    monkeypatch,
):
    """``--config=repro.yaml`` (single-token equals form) and the
    argparse prefix form ``--conf repro.yaml`` are both accepted by
    _parse_config_path (parse_known_args + allow_abbrev), so the
    watchdog plans its llama endpoint and health probe from the explicit
    file. The child spawn must forward the SAME parsed result. The old
    exact-token membership test (``if "--config" in argv``) missed both
    forms: the watchdog planned from the explicit file while the child
    booted on layered defaults -- e.g. a custom overlay.websocket_port
    makes the health probe poll a port the child never binds, and after
    the fail budget the watchdog force-restarts a healthy pipeline
    forever.

    Seeded-defect drill: revert _build_main_cmd's --config re-emit to
    token matching (put "--config" back in the ``for flag in (...)``
    loop and drop the _parse_config_path re-emit) -> both asserts go
    red (child argv carries no --config).
    """
    from pathlib import Path as _P

    monkeypatch.delattr(watchdog.sys, "frozen", raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/python.exe")

    # Equals form: one argv token, invisible to `"--config" in argv`.
    monkeypatch.setattr(
        watchdog.sys, "argv", ["yamabiko.exe", "--config=C:/x.yaml"],
    )
    assert watchdog._build_main_cmd() == [
        "C:/fake/python.exe", "-m", "yamabiko.main",
        "--config", str(_P("C:/x.yaml")),
    ]

    # Abbreviated form: argparse allow_abbrev resolves --conf to
    # --config for the watchdog's own parse; the child must see the
    # canonical flag.
    monkeypatch.setattr(
        watchdog.sys, "argv", ["yamabiko.exe", "--conf", "C:/x.yaml"],
    )
    assert watchdog._build_main_cmd() == [
        "C:/fake/python.exe", "-m", "yamabiko.main",
        "--config", str(_P("C:/x.yaml")),
    ]


def test_build_main_cmd_skips_flag_with_missing_value(monkeypatch):
    """A trailing flag with no value (``yamabiko.exe --max-segments``)
    must NOT be forwarded -- the ``i + 1 < len(argv)`` guard prevents
    forwarding a flag whose value would be stolen from the next unrelated
    token (or crash on index).

    Seeded-defect drill: change ``i + 1 < len(argv)`` to
    ``i + 1 <= len(argv)`` (the classic off-by-one mutant) ->
    IndexError inside _build_main_cmd -> red with traceback.
    """
    monkeypatch.delattr(watchdog.sys, "frozen", raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/python.exe")
    monkeypatch.setattr(
        watchdog.sys, "argv", ["yamabiko.exe", "--max-segments"],
    )
    assert watchdog._build_main_cmd() == [
        "C:/fake/python.exe", "-m", "yamabiko.main"
    ]


def test_wait_or_stop_waits_grace_before_terminate():
    """Fails red if: stop_event causes immediate terminate without
    giving the child its graceful-finally window.

    Tier A contract: a graceful shutdown signal
    (ShutdownEventWriter.signal()) is delivered by the signal handler
    BEFORE stop_event is set. _wait_or_stop must give the child
    grace_s to process that signal + run its finally before falling
    back to terminate(). This test models the hang case where the
    child doesn't exit inside the grace window, so we still expect
    terminate() to fire at the end -- but we also assert the call
    ordering by watching proc.wait() be invoked with the grace_s
    timeout BEFORE terminate() fires.

    Seeded-defect check: revert ``proc.wait(timeout=self._grace_s)``
    in _wait_or_stop to an immediate ``proc.terminate()`` without a
    grace wait -> waits list stays empty / doesn't contain the
    grace-shaped wait, AND terminate fires before any wait, this
    test fails.
    """
    import subprocess
    import threading

    stop_event = threading.Event()

    class _FakeProcess:
        def __init__(self, grace_s: float) -> None:
            self._grace_s = grace_s
            self.wait_timeouts: list[float | None] = []
            self.terminated = False
            self.killed = False
            # Becomes True once terminate() has been called; subsequent
            # wait() calls then return 0 (simulating post-terminate exit).
            self._post_terminate = False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            # First call is the grace-window wait; model the child as
            # hung during grace so we fall through to terminate.
            if not self._post_terminate:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            # Post-terminate wait: child has exited.
            return 0

        def terminate(self):
            self.terminated = True
            self._post_terminate = True

        def kill(self):
            self.killed = True

    proc = _FakeProcess(grace_s=0.05)

    # stop_event is pre-set so _wait_or_stop enters its stop branch
    # immediately (no polling loop).
    stop_event.set()

    wd = watchdog.Watchdog(
        name="test",
        spawn=lambda: proc,  # unused; we pass proc directly
        clock=lambda: 0.0,
        sleep=lambda s: None,
        stop_event=stop_event,
        grace_s=0.05,
    )
    rc = wd._wait_or_stop(proc)

    # Contract assertions:
    # 1. The grace-shaped wait MUST happen, and it MUST be first.
    #    If this list is empty, _wait_or_stop jumped straight to
    #    terminate() without giving the child any window.
    assert proc.wait_timeouts, (
        "proc.wait() was never called; _wait_or_stop skipped the "
        "grace window entirely and went straight to terminate()."
    )
    assert proc.wait_timeouts[0] == 0.05, (
        f"first wait() timeout was {proc.wait_timeouts[0]!r}, expected "
        f"grace_s=0.05. The grace-window wait is missing or uses the "
        f"wrong timeout -- the grace ladder collapsed."
    )
    # 2. After grace timed out, terminate() was called.
    assert proc.terminated, (
        "proc.terminate() was never called after the grace window "
        "elapsed -- hung child would leak."
    )
    # 3. We had to terminate, so rc is None (not the exit code).
    assert rc is None, (
        f"rc should be None when we had to terminate a hung child; got {rc!r}"
    )


def test_wait_or_stop_returns_exit_code_on_graceful_exit_within_grace():
    """Fails red if: a child that exits naturally inside the grace
    window is terminated anyway, losing its STOP_SELF / STOP_ALL exit
    code classification.

    The happy path: the pipeline receives the named-event shutdown ping,
    its listener wakes app.done, the finally block runs, and the pipeline
    exits 0 BEFORE Watchdog._wait_or_stop's grace window elapses. The
    method must return the exit code (not None) so Watchdog.run's
    classify() sees STOP_SELF and returns cleanly.

    Seeded-defect check: change `return proc.wait(timeout=self._grace_s)`
    to a plain `proc.wait(timeout=self._grace_s)` (drop the return) ->
    rc becomes None and Watchdog.run classifies it as "watchdog
    terminated" instead of STOP_SELF, this test fails.
    """
    import threading

    stop_event = threading.Event()

    class _FakeProcess:
        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []
            self.terminated = False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            # Simulate graceful exit inside the grace window: the
            # child's finally ran and it exited 0.
            return 0

        def terminate(self):
            self.terminated = True

    proc = _FakeProcess()
    stop_event.set()

    wd = watchdog.Watchdog(
        name="test",
        spawn=lambda: proc,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        stop_event=stop_event,
        grace_s=0.5,
    )
    rc = wd._wait_or_stop(proc)

    assert rc == 0, (
        f"graceful exit within grace window must return the exit code "
        f"(0); got rc={rc!r}. STOP_SELF classification is lost."
    )
    assert not proc.terminated, (
        "terminate() was called even though the child exited gracefully "
        "inside the grace window -- cooperative shutdown is being "
        "forced to a hard terminate."
    )


def test_wait_or_stop_polls_repeatedly_until_child_exits():
    """Fails red if: a TimeoutExpired in _wait_or_stop's poll loop exits
    the loop instead of continuing to poll (the ``continue`` -> ``break``
    mutant sends a perfectly healthy child into the stop/grace/terminate
    path -- every session would be torn down ~1 poll interval after spawn).

    Distinguishes the mutant by the SHAPE of the wait sequence: normal
    code polls with poll_interval_s on every iteration until the child
    exits naturally (rc returned, no terminate); the mutant's second
    wait would use grace_s and then escalate to terminate.

    Seeded-defect drill: change ``continue`` to ``break`` in
    _wait_or_stop's poll loop -> rc becomes None, the second wait timeout
    reads grace_s (7.7) not poll_interval_s, terminate fires -> all three
    asserts go red.
    """
    import subprocess
    import threading

    class _SlowExitProc:
        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []
            self.terminated = False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) < 3:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            return 5  # natural exit on the 3rd poll

        def terminate(self):
            self.terminated = True

    proc = _SlowExitProc()
    wd = watchdog.Watchdog(
        name="test",
        spawn=lambda: proc,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        stop_event=threading.Event(),  # NOT set: pure poll-loop path
        poll_interval_s=0.05,
        grace_s=7.7,  # distinct so a grace-shaped wait is unmistakable
    )
    rc = wd._wait_or_stop(proc)

    assert rc == 5, (
        f"child's natural exit code was lost (got {rc!r}); the poll "
        f"loop did not return proc.wait()'s value."
    )
    assert proc.wait_timeouts == [0.05, 0.05, 0.05], (
        f"poll-loop wait sequence wrong: {proc.wait_timeouts}. Every "
        f"pre-exit wait must use poll_interval_s (0.05); a grace_s (7.7) "
        f"entry means a TimeoutExpired BROKE the loop and the healthy "
        f"child entered the stop ladder."
    )
    assert not proc.terminated, (
        "terminate() fired on a healthy child that exited naturally -- "
        "the poll loop escalated instead of continuing."
    )


def test_wait_or_stop_kills_child_that_survives_terminate():
    """Fails red if: the kill rung of _wait_or_stop's ladder (terminate
    -> wait(2.0) -> kill -> wait(2.0)) is dropped or reordered. A child
    that shrugs off terminate() would leak past watchdog shutdown.

    Pins the ladder's wait shape [grace_s, 2.0, 2.0] so the hardcoded
    escalation timeouts can't silently drift.

    Seeded-defect drill: delete the ``kill()`` call (keep the getattr
    lookup) in _wait_or_stop -> killed stays False -> red.
    """
    import subprocess
    import threading

    class _UnkillableByTerminate:
        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.killed:
                return 1
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

        def terminate(self):
            self.terminated = True  # has no effect on wait()

        def kill(self):
            self.killed = True

    proc = _UnkillableByTerminate()
    stop_event = threading.Event()
    stop_event.set()  # straight to the stop ladder
    wd = watchdog.Watchdog(
        name="test",
        spawn=lambda: proc,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        stop_event=stop_event,
        grace_s=0.05,
    )
    rc = wd._wait_or_stop(proc)

    assert rc is None, f"expected None for a watchdog-killed child; got {rc!r}"
    assert proc.terminated, "terminate() never fired before kill()"
    assert proc.killed, (
        "child survived terminate() but kill() never fired -- a hung "
        "child outlives the watchdog's shutdown ladder."
    )
    assert proc.wait_timeouts == [0.05, 2.0, 2.0], (
        f"ladder wait shape wrong: {proc.wait_timeouts}; expected "
        f"[grace_s=0.05, post-terminate 2.0, post-kill 2.0]."
    )


def test_run_evicts_stale_restarts_outside_window():
    """Fails red if: Watchdog.run's restart-window eviction loop is
    deleted -- crashes spread far apart in time would accumulate forever
    and a long-running session would hit the max-restart give-up even
    though no flap ever happened.

    Each crash here is 180 clock-seconds from the next (window 100 s,
    cap 2), so the deque never holds more than one in-window timestamp
    and the supervisor must keep restarting until the clean exit.

    Seeded-defect drill: replace the eviction while-loop body with
    ``break`` -> the 3rd crash hits the cap -> give-up fires, spawns
    stop at 3 -> red.
    """
    from collections import deque

    codes = deque([1, 1, 1, 0])  # three slow crashes, then clean exit
    spawns: list[_FakeProcess] = []
    give_up: list = []

    def spawn() -> _FakeProcess:
        p = _FakeProcess(exit_code=codes.popleft(), elapsed_s=60.0)
        spawns.append(p)
        return p

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0  # 3 reads/iteration => 180 s between restarts
        return clock_t[0]

    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        max_restarts=2, restart_window_s=100.0, min_lifetime_s=5.0,
        on_give_up=lambda t, d: give_up.append((t, d)),
    )
    wd.run()
    assert len(spawns) == 4, (
        f"expected 4 spawns (3 out-of-window crashes + clean exit); got "
        f"{len(spawns)}. Stale restart timestamps were not evicted -- slow, "
        f"unrelated crashes accumulated to a bogus give-up."
    )
    assert give_up == [], (
        f"give-up fired despite every crash being outside the rolling "
        f"window: {give_up}"
    )


def test_module_supervision_defaults_are_pinned():
    """Fails red if: any module-level supervision default is
    fat-fingered -- a corrupted backoff tuple or min-lifetime would
    ship silently and change every install's crash-recovery behaviour.

    These are user-facing operational numbers (30.0 for min-lifetime
    is deliberate); pin them exactly.

    Seeded defect: change any constant (e.g. _DEFAULT_MIN_LIFETIME_S
    30.0 -> 5.0) -> red.
    """
    assert watchdog._DEFAULT_BACKOFF_S == (1.0, 5.0, 15.0)
    assert watchdog._DEFAULT_MAX_RESTARTS == 3
    assert watchdog._DEFAULT_RESTART_WINDOW_S == 600.0
    assert watchdog._DEFAULT_MIN_LIFETIME_S == 30.0
    assert watchdog._DEFAULT_POLL_INTERVAL_S == 1.0
    # And the Watchdog constructor actually consumes them as defaults.
    wd = watchdog.Watchdog(spawn=lambda: None)
    assert wd._max_restarts == 3
    assert wd._restart_window_s == 600.0
    assert wd._min_lifetime_s == 30.0
    assert wd._backoff_s == (1.0, 5.0, 15.0)
    assert wd._poll_interval_s == 1.0


def test_on_give_up_fires_on_fast_crash():
    calls: list[tuple[str, str]] = []

    def spawn() -> _FakeProcess:
        return _FakeProcess(exit_code=1, elapsed_s=1.0)

    clock_t = [0.0]
    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None,
        min_lifetime_s=5.0,
        on_give_up=lambda title, detail: calls.append((title, detail)),
    )
    wd.run()
    assert len(calls) == 1
    assert "failed to start" in calls[0][0].lower()


# ---------------------------------------------------------------------------
# Stop-all cascade: every TERMINAL supervisor outcome must funnel through
# the shared stop_all cascade so peer supervisors + the single-instance
# mutex drop immediately. RESTART outcomes must NOT cascade.
#
# The give-up paths (spawn-failure, min-lifetime, max-restart) and the
# clean STOP_SELF path must call the cascade rather than bare-returning --
# otherwise the llama supervisor thread and mutex can linger or hang.
# on_stop_all centralises the cascade; _cascade_stop falls back to the
# local stop_event when no callback is wired (unit tests).
# ---------------------------------------------------------------------------


def test_run_calls_stop_all_on_clean_quit():
    """A clean child exit (STOP_SELF) must cascade so peers + mutex
    drop immediately, not linger.

    Seeded-defect drill: revert the STOP_SELF branch to a bare `return`
    -> stop_all not called -> red.
    """
    from collections import deque
    from yamabiko.watchdog import Watchdog
    called = []
    spawns = deque([_FakeProcess(exit_code=0, elapsed_s=10.0)])
    wd = Watchdog(name="pipeline", spawn=lambda: spawns.popleft(),
                  on_stop_all=lambda: called.append(True),
                  clock=lambda: 0.0, sleep=lambda s: None)
    wd.run()
    assert called == [True]


def test_run_calls_stop_all_and_records_error_on_fast_crash():
    """A sub-min-lifetime give-up must cascade (stop_all) AND surface a
    user-facing error -- not bare-return and hang.

    Seeded-defect drill: remove on_stop_all() from the min-lifetime
    branch -> red.
    """
    from yamabiko.watchdog import Watchdog
    stop_called = []
    give_up = []
    # elapsed is computed from self._clock(); with a clock that advances
    # 1s per call the child's lifetime reads as 1s, which is < the 5s
    # min_lifetime, so run() takes the fast-crash give-up branch.
    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    wd = Watchdog(name="pipeline",
                  spawn=lambda: _FakeProcess(exit_code=1, elapsed_s=1.0),
                  on_stop_all=lambda: stop_called.append(True),
                  on_give_up=lambda t, d: give_up.append((t, d)),
                  clock=clock, sleep=lambda s: None,
                  min_lifetime_s=5.0)
    wd.run()
    assert stop_called == [True]
    assert len(give_up) == 1


def test_run_calls_stop_all_on_spawn_failure():
    """A spawn that raises (bad exe path, model never appears) is a
    TERMINAL outcome -- it must cascade AND surface an error, not
    bare-return leaving the llama supervisor + mutex hung forever.

    Seeded-defect drill: remove self._cascade_stop() from the
    spawn-failure branch -> stop_called stays empty -> red.
    """
    from yamabiko.watchdog import Watchdog
    stop_called = []
    give_up = []

    def _boom():
        raise FileNotFoundError("no such executable")

    wd = Watchdog(name="pipeline", spawn=_boom,
                  on_stop_all=lambda: stop_called.append(True),
                  on_give_up=lambda t, d: give_up.append((t, d)),
                  clock=lambda: 0.0, sleep=lambda s: None)
    wd.run()
    assert stop_called == [True]
    assert len(give_up) == 1


def test_run_calls_stop_all_on_crash_loop_cap():
    """Hitting the max-restart cap is a TERMINAL outcome -- it must
    cascade so peers drop, not bare-return.

    Seeded-defect drill: remove self._cascade_stop() from the max-restart
    branch -> stop_called stays empty -> red.
    """
    from yamabiko.watchdog import Watchdog
    stop_called = []
    give_up = []

    # 60s lifetime each (>= min_lifetime) so every crash is a RESTART
    # until the cap; clock advances 60s per call.
    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    wd = Watchdog(name="pipeline",
                  spawn=lambda: _FakeProcess(exit_code=1, elapsed_s=60.0),
                  on_stop_all=lambda: stop_called.append(True),
                  on_give_up=lambda t, d: give_up.append((t, d)),
                  clock=clock, sleep=lambda s: None,
                  max_restarts=2, restart_window_s=600.0, min_lifetime_s=5.0)
    wd.run()
    assert stop_called == [True]
    assert len(give_up) == 1


def test_run_calls_stop_all_on_stop_all_exit_code():
    """A STOP_ALL exit code (42 wizard-cancel, 43 tailored-error) must
    route through the same cascade as the give-up paths.

    Seeded-defect drill: revert the STOP_ALL branch's self._cascade_stop()
    back to a bare `return` (drop the set entirely) -> stop_called empty
    -> red.
    """
    from yamabiko.watchdog import Watchdog
    stop_called = []
    wd = Watchdog(name="pipeline",
                  spawn=lambda: _FakeProcess(exit_code=42, elapsed_s=10.0),
                  on_stop_all=lambda: stop_called.append(True),
                  clock=lambda: 0.0, sleep=lambda s: None,
                  min_lifetime_s=0.0)
    wd.run()
    assert stop_called == [True]


def test_run_does_not_call_stop_all_on_transient_restart():
    """Fails red if: a transient crash (RESTART, within budget) cascades.
    Only TERMINAL outcomes cascade. stop_all fires exactly once, on the
    final clean exit -- never on the intermediate restart.

    Seeded-defect drill: add self._cascade_stop() to the RESTART/backoff
    fall-through -> stop_called == [True, True] -> red.
    """
    from collections import deque
    from yamabiko.watchdog import Watchdog
    stop_called = []
    codes = deque([1, 0])  # first crash (RESTART), then clean exit (STOP_SELF)
    # min_lifetime_s=0.0 with a 0.0 clock => elapsed 0.0 is NOT < 0.0, so
    # the first crash is a genuine RESTART (not a fast-crash give-up); the
    # second spawn exits 0 -> STOP_SELF -> the single cascade.
    wd = Watchdog(name="pipeline",
                  spawn=lambda: _FakeProcess(exit_code=codes.popleft(), elapsed_s=60.0),
                  on_stop_all=lambda: stop_called.append(True),
                  clock=lambda: 0.0, sleep=lambda s: None,
                  min_lifetime_s=0.0)
    wd.run()
    assert stop_called == [True]  # fires once, on the final clean exit


def test_cascade_stop_falls_back_to_local_stop_event_when_no_callback():
    """When on_stop_all is None (the unit-test / direct-instantiation
    default), _cascade_stop sets the local stop_event so existing
    exit-42/43 tests that assert stop_event.is_set() keep working.

    Seeded-defect drill: change the else-branch of _cascade_stop to a
    no-op (drop self._stop_event.set()) -> stop_event stays unset after
    a STOP_ALL exit -> red.
    """
    import threading
    from yamabiko.watchdog import Watchdog
    stop_event = threading.Event()
    wd = Watchdog(name="pipeline",
                  spawn=lambda: _FakeProcess(exit_code=43, elapsed_s=10.0),
                  stop_event=stop_event,
                  clock=lambda: 0.0, sleep=lambda s: None,
                  min_lifetime_s=0.0)
    wd.run()
    assert stop_event.is_set()


# ---------------------------------------------------------------------------
# _SafeTextIO: makes print() safe in the frozen windowed bundle.
#
# Without this wrapper, print(..., flush=True) on runw.exe's stdout
# raises OSError [Errno 22], the supervisor catches it, and the downstream
# manager.broadcast() that ships subtitles to the overlay never runs --
# pipeline is alive, no subtitles arrive. The wrapper is only installed
# when sys.frozen is set, so dev-mode pytest runs the tests below against
# the wrapper class directly.
# ---------------------------------------------------------------------------


class _BrokenStream:
    """write/flush raise OSError 22, exactly like runw.exe's stdout."""
    encoding = "utf-8"

    def __init__(self) -> None:
        self.write_attempts = 0
        self.flush_attempts = 0

    def write(self, s: str) -> int:
        self.write_attempts += 1
        raise OSError(22, "Invalid argument")

    def flush(self) -> None:
        self.flush_attempts += 1
        raise OSError(22, "Invalid argument")


def test_safe_text_io_swallows_write_oserror() -> None:
    inner = _BrokenStream()
    w = watchdog._SafeTextIO(inner)
    # Must not raise even though the underlying stream errors.
    n = w.write("hello\n")
    assert n == len("hello\n"), "write must pretend to succeed"
    assert inner.write_attempts == 1


def test_safe_text_io_returns_underlying_write_count_on_success() -> None:
    """Fails red if: _SafeTextIO.write() drops the underlying stream's
    return value on the SUCCESS path (TextIO contract: write returns the
    character count; io.TextIOWrapper consumers may rely on it). The
    error-path return is covered above; this pins the happy path the
    Pins the happy path (the `write()` return-value).

    Seeded defect: change ``return self._w.write(s)`` to
    ``self._w.write(s)`` (drop the return) -> None returned -> red.
    """
    class _CountingStream:
        encoding = "utf-8"

        def write(self, s: str) -> int:
            return 7  # deliberately != len(s) to catch a len() fallback

        def flush(self) -> None:
            pass

    w = watchdog._SafeTextIO(_CountingStream())
    assert w.write("hello") == 7, (
        "_SafeTextIO.write() must return the wrapped stream's return "
        "value on success, not swallow or recompute it"
    )


def test_safe_text_io_swallows_flush_oserror() -> None:
    inner = _BrokenStream()
    w = watchdog._SafeTextIO(inner)
    w.flush()  # must not raise
    assert inner.flush_attempts == 1


def test_safe_text_io_makes_print_flush_true_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual failure mode: print(..., flush=True) on broken stdout.

    In the frozen bundle a print with flush=True on the broken stdout
    raises, causing the translation stage's success path to bail and
    broadcast to never fire. With the wrapper the print is a silent no-op.
    """
    import sys
    inner = _BrokenStream()
    monkeypatch.setattr(sys, "stdout", watchdog._SafeTextIO(inner))
    # Exercises the call site that was producing the broken-stdout error.
    print("[JA] ... [EN] ... [ms] ...", flush=True)
    # Got here = the exception was swallowed. Also verify the underlying
    # stream was actually poked (otherwise our wrapper isn't really in
    # line).
    assert inner.write_attempts >= 1
    assert inner.flush_attempts >= 1


def test_safe_text_io_delegates_unrelated_attrs() -> None:
    """reconfigure, isatty, etc. must reach the wrapped stream.

    main.py does sys.stdout.reconfigure(encoding='utf-8', ...) at
    import time — if the wrapper shadowed that, import would fail.
    """
    class _WithExtras:
        encoding = "utf-8"
        def reconfigure(self, **kw):
            self.last_reconfigure = kw
        def isatty(self):
            return False
    inner = _WithExtras()
    w = watchdog._SafeTextIO(inner)
    assert w.encoding == "utf-8"
    assert w.isatty() is False
    w.reconfigure(encoding="utf-8", errors="replace")
    assert inner.last_reconfigure == {"encoding": "utf-8", "errors": "replace"}


def test_frozen_stdio_guard_gives_a_none_stderr_a_sink_a_live_tqdm_bar_writes_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A windowed yamabiko.exe started without std handles (Explorer, a
    shortcut, a DETACHED spawn) has sys.stdout and sys.stderr None. tqdm,
    which hf_hub_download's progress bar uses, defaults its file to
    sys.stderr and writes from its constructor, so every download in such a
    process failed before its first byte with AttributeError("'NoneType'
    object has no attribute 'write'"). The frozen entry's guard gives a None
    stream a sink and keeps wrapping a real one.

    Seeded defects (Decision 148): in _guard_frozen_stdio, leave a None
    stream None (the old ``if sys.stderr is not None`` form) -> the
    isinstance assert is red; wrap a None stream without a sink
    (``_SafeTextIO(sys.stderr)``) -> the isinstance asserts pass and the
    bar raises AttributeError ('NoneType' object has no attribute 'flush';
    probed with tqdm 4.67.3) -> red. A closed-file sink is no drill:
    _SafeTextIO swallows the ValueError it raises.
    """
    import io

    from tqdm.auto import tqdm

    console = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", None)

    watchdog._guard_frozen_stdio()

    assert isinstance(sys.stdout, watchdog._SafeTextIO)
    assert isinstance(sys.stderr, watchdog._SafeTextIO)
    try:
        print("kept", flush=True)
        assert console.getvalue() == "kept\n", "a real stdout must stay the wrapped stream"
        # As hf_hub_download builds our tqdm_class bar: live (disable=False),
        # file=None, so tqdm writes to sys.stderr from its constructor on.
        with tqdm(total=3, disable=False) as bar:
            bar.update(3)
    finally:
        # The os.devnull sink; monkeypatch restores sys.stderr. The no-sink
        # drill has no stream to close: let it fail at the bar, not here.
        with contextlib.suppress(AttributeError):
            sys.stderr.close()


# ---------------------------------------------------------------------------
# --run-overlay dispatch and dual-process supervision.
#
# The shipped bundle only has one executable (yamabiko.exe). The top-level
# watchdog spawns BOTH the pipeline and the overlay as supervised child
# processes, each re-entering yamabiko.exe with its own --run-* flag.
# ---------------------------------------------------------------------------


def test_build_overlay_cmd_frozen_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen exe: overlay child re-enters yamabiko.exe with --run-overlay.

    No ``level`` from the caller -> the --log-level tail is the
    documented INFO default. _build_overlay_cmd takes the level as a
    PARAMETER from the caller's already-loaded config -- it must never
    re-read config from disk itself (see
    test_run_pipeline_overlay_log_level_round_trips_from_explicit_config
    in tests/test_build_pipeline.py for the --config round-trip).
    """
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "executable", r"C:\fake\yamabiko.exe")
    cmd = watchdog._build_overlay_cmd()
    assert cmd == [
        r"C:\fake\yamabiko.exe", "--run-overlay", "--log-level", "INFO",
    ]


def test_build_overlay_cmd_dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev mode: overlay child uses python -m yamabiko.overlay_app."""
    import sys as _sys
    # sys.frozen may or may not exist; delattr safely.
    if hasattr(_sys, "frozen"):
        monkeypatch.delattr(_sys, "frozen", raising=False)
    monkeypatch.setattr(_sys, "executable", r"C:\fake\python.exe")
    cmd = watchdog._build_overlay_cmd()
    assert cmd == [
        r"C:\fake\python.exe", "-m", "yamabiko.overlay_app",
        "--log-level", "INFO",
    ]


def test_build_overlay_cmd_appends_url_in_frozen_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overlay child's argparse defaults to ws://...:8765/ws; when
    the caller passes the configured URL it must ride the argv, AFTER
    --run-overlay (the frozen re-entry strips only its own flag and
    hands the rest to overlay argparse).

    Seeded defect: drop the `cmd.extend(["--url", url])` branch ->
    the URL never reaches the child -> red.
    """
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "executable", r"C:\fake\yamabiko.exe")
    cmd = watchdog._build_overlay_cmd(url="ws://127.0.0.1:9999/ws")
    assert cmd == [
        r"C:\fake\yamabiko.exe", "--run-overlay",
        "--url", "ws://127.0.0.1:9999/ws",
        "--log-level", "INFO",
    ]


def test_build_overlay_cmd_appends_url_in_dev_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dev-mode twin: python -m yamabiko.overlay_app --url <url>."""
    import sys as _sys
    if hasattr(_sys, "frozen"):
        monkeypatch.delattr(_sys, "frozen", raising=False)
    monkeypatch.setattr(_sys, "executable", r"C:\fake\python.exe")
    cmd = watchdog._build_overlay_cmd(url="ws://127.0.0.1:9999/ws")
    assert cmd == [
        r"C:\fake\python.exe", "-m", "yamabiko.overlay_app",
        "--url", "ws://127.0.0.1:9999/ws",
        "--log-level", "INFO",
    ]


def test_build_overlay_cmd_forwards_explicit_config_path(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    cmd = watchdog._build_overlay_cmd(
        url="ws://127.0.0.1:9911/ws",
        level="DEBUG",
        config_path=Path("C:/tmp/repro.yaml"),
    )
    assert cmd[-2:] == ["--config", str(Path("C:/tmp/repro.yaml"))]


def test_main_short_circuits_on_run_overlay_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--run-overlay dispatches to overlay_app.main() and exits.

    The flag must be stripped from sys.argv before delegating so
    overlay_app's argparse doesn't choke on an unknown arg. Same shape
    as --run-main's short-circuit, just a different target.
    """
    import sys as _sys
    calls: list[str] = []

    def _fake_overlay_main() -> int:
        # Capture the argv that reached overlay_app.main() -- the
        # dispatch flag must have been removed by now.
        calls.append("overlay")
        assert "--run-overlay" not in _sys.argv, (
            "watchdog must strip --run-overlay before delegating"
        )
        return 0

    # Synthesize a fake yamabiko.overlay_app.__main__ so main() imports
    # us instead of the real QApplication-launching module.
    import types
    fake_mod = types.ModuleType("yamabiko.overlay_app.__main__")
    fake_mod.main = _fake_overlay_main
    monkeypatch.setitem(_sys.modules, "yamabiko.overlay_app.__main__", fake_mod)

    monkeypatch.setattr(_sys, "argv", ["yamabiko", "--run-overlay"])

    with pytest.raises(SystemExit) as excinfo:
        watchdog.main()
    assert excinfo.value.code == 0
    assert calls == ["overlay"], (
        f"overlay_app.main not called exactly once; saw {calls}"
    )


def test_main_run_overlay_does_not_consume_other_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only --run-overlay is stripped. Other args pass through to
    overlay_app's own argparse.
    """
    import sys as _sys
    seen_argv: list[list[str]] = []

    def _fake_overlay_main() -> int:
        seen_argv.append(list(_sys.argv))
        return 0

    import types
    fake_mod = types.ModuleType("yamabiko.overlay_app.__main__")
    fake_mod.main = _fake_overlay_main
    monkeypatch.setitem(_sys.modules, "yamabiko.overlay_app.__main__", fake_mod)

    monkeypatch.setattr(
        _sys, "argv",
        ["yamabiko", "--run-overlay", "--url", "ws://testhost:1234/ws"],
    )

    with pytest.raises(SystemExit):
        watchdog.main()
    assert seen_argv == [["yamabiko", "--url", "ws://testhost:1234/ws"]], (
        f"--url should survive for overlay_app's argparse; saw {seen_argv}"
    )


# ---------------------------------------------------------------------------
# Single-instance check.
#
# Without this check, double-clicking the Start Menu shortcut launches a
# second yamabiko.exe whose pipeline child crashes on the port 8765 bind
# (already held by the first instance), producing a "Yamabiko keeps
# crashing" dialog. These tests lock down the mutex-based check and the
# friendly-dialog fallback path.
# ---------------------------------------------------------------------------


def test_single_instance_mutex_acquires_fresh() -> None:
    """Fresh mutex acquisition returns a REAL Win32 HANDLE (nonzero int)
    on Windows -- not one of the function's object() failure fallbacks.

    Asserting only ``handle is not None`` is satisfied by EVERY failure
    path in the function (CreateMutexW failure -> object(), exception ->
    object(), non-Windows -> object()) -- a fully broken acquisition
    would pass. CreateMutexW's restype is wintypes.HANDLE, so the success
    path returns a nonzero int -- assert exactly that, and close the
    handle so the GLOBAL named mutex isn't held for the rest of the
    pytest session (which would order-poison any later real-acquire test).

    Seeded-defect drill: change the success path's ``return handle`` to
    ``return object()`` -> isinstance(int) fails -> red.
    """
    import sys as _sys
    handle = watchdog._acquire_single_instance_mutex()
    if not _sys.platform.startswith("win"):
        # Non-Windows: enforcement is a documented no-op placeholder.
        assert handle is not None
        return
    try:
        assert isinstance(handle, int) and handle != 0, (
            f"_acquire_single_instance_mutex returned {handle!r} on a "
            f"fresh acquire; expected a real nonzero int HANDLE from "
            f"CreateMutexW. An object() here means a failure fallback "
            f"fired -- single-instance enforcement is silently dead."
        )
    finally:
        if isinstance(handle, int) and handle:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(handle)


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="named-mutex enforcement is Windows-only",
)
def test_single_instance_mutex_second_acquire_returns_none() -> None:
    """The ERROR_ALREADY_EXISTS enforcement branch actually fires against
    the REAL Win32 mutex: while the first handle is held, a second
    _acquire_single_instance_mutex() must observe ERROR_ALREADY_EXISTS
    and return None -- the double-launch rejection.

    If the constant drifts (183 -> anything else) or the comparison
    inverts, double-launch protection silently dies (second instance's
    pipeline crashes on the port-8765 bind -> "keeps crashing" dialog)
    with the suite green.

    CreateMutexW on an existing name succeeds AND sets last-error 183
    regardless of which process created it, so the same-process second
    call exercises the production branch for real (cross-process
    enforcement is covered by the real-tier subprocess test below).

    Seeded-defect drill: change ``_ERROR_ALREADY_EXISTS`` to 9999 ->
    second acquire returns the handle instead of None -> red.
    """
    import ctypes
    first = watchdog._acquire_single_instance_mutex()
    try:
        assert isinstance(first, int) and first != 0, (
            f"precondition: first acquire must win a real handle; got "
            f"{first!r}"
        )
        second = watchdog._acquire_single_instance_mutex()
        assert second is None, (
            f"second acquire returned {second!r} while the mutex was "
            f"held; expected None (ERROR_ALREADY_EXISTS). "
            f"Double-launch protection is broken: a second yamabiko.exe "
            f"would spawn a pipeline that crashes on the port bind."
        )
    finally:
        if isinstance(first, int) and first:
            ctypes.windll.kernel32.CloseHandle(first)


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="named-mutex enforcement is Windows-only",
)
def test_single_instance_mutex_access_denied_returns_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """CreateMutexW fails with ERROR_ACCESS_DENIED (5) when the Global\\
    mutex is held by an instance in another logon session. That is an
    instance already running, so the launch must be refused (None), not
    waved through as "enforcement unavailable".

    Seeded defect: drop the err=5 branch -> the fallback returns object().
    """
    import ctypes

    def _denied(_attrs, _owner, _name):
        ctypes.set_last_error(5)
        return 0

    monkeypatch.setattr(watchdog, "_CreateMutexW", _denied)
    with caplog.at_level(logging.WARNING, logger="yamabiko.watchdog"):
        handle = watchdog._acquire_single_instance_mutex()

    assert handle is None, f"err=5 must mean already running; got {handle!r}"
    assert "single-instance mutex owned by another session (err=5)" in caplog.text


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="named-mutex enforcement is Windows-only",
)
def test_single_instance_mutex_binding_captures_already_exists_in_tls() -> None:
    """The production CreateMutexW binding must be declared with
    ``use_last_error=True`` so ctypes captures ERROR_ALREADY_EXISTS into
    thread-local storage ATOMICALLY at the FFI call. The unsafe idiom
    (``ctypes.windll.kernel32.CreateMutexW`` + a separate live
    ``GetLastError()`` FFI call) leaves a window in which any
    ctypes/CPython-internal Win32 call clobbers the thread's last-error
    -- and a clobbered 183 means a second yamabiko instance sails past
    the single-instance gate onto the port-8765 bind crash.

    The behaviour twin (``..._second_acquire_returns_none``) cannot
    distinguish the safe from the unsafe idiom (nothing intervenes in
    the test's tight window); this test pins the MECHANISM: after
    calling the module's own binding against an already-held mutex, the
    ctypes TLS slot -- pre-cleared to 0 -- must read 183.

    Seeded-defect drill: rebind ``_CreateMutexW`` from
    ``ctypes.windll.kernel32`` (no use_last_error) -> the TLS slot stays
    0 after the call -> red.
    """
    import ctypes
    first = watchdog._acquire_single_instance_mutex()
    try:
        assert isinstance(first, int) and first != 0, (
            f"precondition: first acquire must win a real handle; got "
            f"{first!r}"
        )
        # Clear the TLS slot so a stale 183 captured by an EARLIER
        # use_last_error call can't fake a pass.
        ctypes.set_last_error(0)
        second = watchdog._CreateMutexW(
            None, False, watchdog._SINGLE_INSTANCE_MUTEX,
        )
        try:
            assert second, (
                "CreateMutexW on an existing mutex must still return a "
                "(non-owning) handle; NULL means the probe itself broke"
            )
            assert ctypes.get_last_error() == watchdog._ERROR_ALREADY_EXISTS, (
                f"ctypes.get_last_error() == "
                f"{ctypes.get_last_error()} after CreateMutexW on an "
                f"already-held mutex; expected ERROR_ALREADY_EXISTS "
                f"(183). The production binding is not capturing "
                f"last-error into TLS (use_last_error=True missing) -- "
                f"the single-instance decision is reading a clobberable "
                f"live GetLastError."
            )
        finally:
            if second:
                ctypes.windll.kernel32.CloseHandle(second)
    finally:
        if isinstance(first, int) and first:
            ctypes.windll.kernel32.CloseHandle(first)


@pytest.mark.real
@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="named-mutex enforcement is Windows-only",
)
def test_real_second_process_mutex_acquire_returns_none() -> None:
    """Cross-PROCESS double-launch enforcement: while this pytest
    process holds the Global\\ mutex, a helper subprocess importing the
    real watchdog module and calling _acquire_single_instance_mutex()
    must get None -- the actual two-yamabiko.exe scenario.

    Subject is the real Win32 named-mutex machinery across a process
    boundary; nothing is mocked.

    NOTE: holds the production Global\\ mutex for the test duration; a
    concurrently-running real yamabiko instance (or a parallel test
    session holding the mutex) would make the FIRST acquire fail. The
    precondition assert below surfaces that as an explicit failure, not
    a silent vacuous pass.

    Seeded-defect drill: invert the ERROR_ALREADY_EXISTS comparison ->
    the child exits 8 -> red.
    """
    import ctypes
    import subprocess as _sp
    import sys as _sys

    first = watchdog._acquire_single_instance_mutex()
    try:
        assert isinstance(first, int) and first != 0, (
            f"precondition: pytest could not win the global mutex (got "
            f"{first!r}). Is a real yamabiko.exe (or a parallel test "
            f"session) running and holding it?"
        )
        child_code = (
            "import sys\n"
            "from yamabiko import watchdog as w\n"
            "h = w._acquire_single_instance_mutex()\n"
            "sys.exit(7 if h is None else 8)\n"
        )
        proc = _sp.run(
            [_sys.executable, "-c", child_code],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 7, (
            f"helper subprocess exited {proc.returncode} (7=rejected as "
            f"second instance, 8=acquired the mutex despite it being "
            f"held, other=import/runtime error). stderr tail: "
            f"{proc.stderr[-500:]!r}. Cross-process double-launch "
            f"enforcement is broken."
        )
    finally:
        if isinstance(first, int) and first:
            ctypes.windll.kernel32.CloseHandle(first)


def test_main_second_instance_shows_friendly_error_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _acquire_single_instance_mutex returns None (signalling
    another instance holds the mutex), main() shows a StartupError
    dialog saying so and exits 0 -- NOT crash-loop, NOT the generic
    crash dialog.
    """
    import sys as _sys
    from yamabiko import log_config
    # build_watchdog now runs file-log setup + crash-evidence arming
    # BEFORE the mutex check (so the "already running" dialog leaves a
    # log line). Stub both here: this test's subject is the dialog +
    # exit-0 contract, and the real pre-mutex logging path is covered
    # by test_build_watchdog_logs_already_running_dialog_verbatim_to_
    # role_file under a tmp_path-scoped YAMABIKO_APPDATA. Without the
    # stubs this test would write into the developer's real
    # %LOCALAPPDATA%\Yamabiko\logs and leak a RotatingFileHandler on
    # the global "yamabiko" logger for the rest of the session.
    monkeypatch.setattr(log_config, "setup_main_log", lambda *a, **kw: None)
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    # Stub out the mutex acquire to simulate "already held"
    monkeypatch.setattr(watchdog, "_acquire_single_instance_mutex", lambda: None)
    # Capture the StartupError that gets shown. show_startup_error is
    # imported at module level in watchdog.py and the "already running"
    # branch calls that module global, so patch it on the watchdog module
    # (patching error_dialog.show_startup_error would not intercept it).
    shown: list = []
    monkeypatch.setattr(watchdog, "show_startup_error", lambda err: shown.append(err))
    # Empty argv (no --version / --run-main / --run-overlay)
    monkeypatch.setattr(_sys, "argv", ["yamabiko"])

    with pytest.raises(SystemExit) as excinfo:
        watchdog.main()

    assert excinfo.value.code == 0
    assert len(shown) == 1
    err = shown[0]
    assert err.title == "Yamabiko is already running"
    assert "tray" in err.message.lower() or "tray" in err.suggested_fix.lower()


def test_main_second_instance_does_not_sweep_first_instances_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SECOND launch (single-instance mutex already held by the first,
    running instance) must NOT run the orphan sweep.

    ``_sweep_orphaned_children`` taskkills EVERY project-tree
    llama-server / overlay / pipeline. Running it before the mutex gate
    means a double-launch force-kills the FIRST (running) instance's
    children, and only then does the mutex reject the second instance --
    so double-clicking the shortcut nukes the app you were using.

    main() must let build_watchdog (which acquires the mutex) run FIRST,
    and sweep only if we actually won the mutex.

    Seeded-defect check: move ``_sweep_orphaned_children()`` back ahead of
    ``build_watchdog()`` in main() -> the spy records a call -> this fails.
    """
    import sys as _sys
    from yamabiko import log_config
    # Pre-mutex log setup + crash-evidence arming are stubbed for the
    # same hermeticity reason as in
    # test_main_second_instance_shows_friendly_error_and_exits above:
    # keep this test out of the developer's real logs dir; the real
    # pre-mutex logging path has its own tmp_path-scoped test.
    monkeypatch.setattr(log_config, "setup_main_log", lambda *a, **kw: None)
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    swept: list[bool] = []
    # Simulate "another instance already holds the mutex".
    monkeypatch.setattr(watchdog, "_acquire_single_instance_mutex", lambda: None)
    monkeypatch.setattr(
        watchdog, "_sweep_orphaned_children",
        lambda: (swept.append(True), 0)[1],
    )
    # Suppress the real Qt "already running" dialog. The call site uses
    # watchdog's module-global show_startup_error, so patch it there
    # (not on error_dialog).
    monkeypatch.setattr(watchdog, "show_startup_error", lambda err: None)
    monkeypatch.setattr(_sys, "argv", ["yamabiko"])

    with pytest.raises(SystemExit) as excinfo:
        watchdog.main()

    assert excinfo.value.code == 0
    assert swept == [], (
        "second instance ran the orphan sweep -> it force-killed the FIRST "
        "instance's children before the mutex rejected the second"
    )


def test_main_sweeps_only_after_winning_the_mutex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering invariant: main() must run build_watchdog (the mutex gate)
    BEFORE the orphan sweep, and sweep BEFORE run_watchdog spawns our own
    children.

    Seeded-defect check: reorder main() so ``_sweep_orphaned_children()``
    runs before ``build_watchdog()`` -> order != ['build','sweep','run'] -> fails.
    """
    import sys as _sys
    order: list[str] = []

    class _FakeApp:
        single_instance_handle = object()

    monkeypatch.setattr(
        watchdog, "build_watchdog",
        lambda argv=None: (order.append("build"), _FakeApp())[1],
    )
    monkeypatch.setattr(
        watchdog, "_sweep_orphaned_children",
        lambda: (order.append("sweep"), 0)[1],
    )
    monkeypatch.setattr(
        watchdog, "run_watchdog",
        lambda app: (order.append("run"), 0)[1],
    )
    monkeypatch.setattr(_sys, "argv", ["yamabiko"])

    with pytest.raises(SystemExit) as excinfo:
        watchdog.main()

    assert excinfo.value.code == 0
    assert order == ["build", "sweep", "run"], (
        f"main() lifecycle order wrong: {order}; the sweep must run only "
        f"after build_watchdog wins the mutex, and before run_watchdog "
        f"spawns our own children"
    )


@pytest.fixture
def isolated_watchdog_log(monkeypatch, tmp_path, hermetic_log_handlers):
    """watchdog.main()'s config-error path attaches the watchdog file handler
    unless the root or ``yamabiko`` logger already has one, and sets the
    ``yamabiko`` level. Clear both loggers for the test (a handler leaked by
    an earlier test would skip the attach), point the log at tmp_path, and
    restore both afterwards."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    with hermetic_log_handlers("", "yamabiko"):
        yield tmp_path


def test_watchdog_main_surfaces_semantic_config_before_orphan_sweep(
    monkeypatch, isolated_watchdog_log,
):
    shown = []
    swept = []

    def fail_build(_argv):
        raise ConfigValidationError([
            "translation.temperature must be between 0.0 and 2.0"
        ])

    monkeypatch.setattr(watchdog, "build_watchdog", fail_build)
    monkeypatch.setattr(watchdog, "_sweep_orphaned_children", swept.append)
    monkeypatch.setattr(watchdog, "show_startup_error", shown.append)
    monkeypatch.setattr(watchdog.sys, "argv", ["yamabiko.exe"])

    with pytest.raises(SystemExit) as exc:
        watchdog.main()

    assert exc.value.code == 43
    assert swept == []
    assert len(shown) == 1
    assert shown[0].title == "Invalid configuration"
    assert "translation.temperature" in shown[0].details


def test_watchdog_main_config_error_before_log_setup_reaches_watchdog_log(
    monkeypatch, isolated_watchdog_log,
):
    """build_watchdog raises ConfigValidationError from its config load,
    before its own setup_main_log call. main() must still attach the
    watchdog file handler so the reason lands in main-watchdog.log; the
    frozen bundle discards stderr, so otherwise no trace survives.

    Seeded defect: remove the setup_main_log call from the
    ConfigValidationError handler -> the log file is never written.
    """

    def fail_build(_argv):
        raise ConfigValidationError([
            "translation.temperature must be between 0.0 and 2.0"
        ])

    monkeypatch.setattr(watchdog, "build_watchdog", fail_build)
    monkeypatch.setattr(watchdog, "_sweep_orphaned_children", lambda: 0)
    monkeypatch.setattr(watchdog, "show_startup_error", lambda err: None)
    monkeypatch.setattr(watchdog.sys, "argv", ["yamabiko.exe"])

    with pytest.raises(SystemExit) as exc:
        watchdog.main()

    assert exc.value.code == 43
    log_path = isolated_watchdog_log / "logs" / "main-watchdog.log"
    assert log_path.is_file(), "main-watchdog.log was never written"
    text = log_path.read_text(encoding="utf-8")
    assert "invalid configuration" in text
    assert "translation.temperature" in text


# ---------------------------------------------------------------------------
# Translator quant-tier resolution self-heals to the cached tier.
#
# A Settings quant-tier switch writes translation.quant_tier but never
# downloads the new GGUF, so the configured tier can be absent from the cache.
# _resolve_translator_model_path must fall back to ANOTHER cached tier rather
# than return None (which makes the spawn loop poll forever -> translation
# silently dead). The genuine first-run case (NO tier cached) still returns
# None so the loop waits for the wizard download.
# ---------------------------------------------------------------------------


def _patch_quant(monkeypatch, tier: str) -> None:
    import yamabiko.config as _cfg
    monkeypatch.setattr(
        _cfg, "load_full_config",
        lambda: {"translation": {"quant_tier": tier}},
    )


def _tier_file(tier: str) -> str:
    from yamabiko import models
    return models.TRANSLATOR_PINS[tier].filename


def test_resolve_translator_falls_back_to_cached_other_tier(monkeypatch, tmp_path):
    """Seeded-defect drill: remove the other-tier fallback in
    _resolve_translator_model_path -> returns None when the configured tier
    is missing -> this fails (and the real spawn loop would poll forever)."""
    import huggingface_hub
    _patch_quant(monkeypatch, "small")  # user switched to small...

    default_gguf = tmp_path / "translator-default.gguf"
    default_gguf.write_bytes(b"gguf")

    def _cache(repo_id, filename, revision=None):
        # small NOT downloaded; default IS present.
        return str(default_gguf) if filename == _tier_file("default") else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _cache)
    result = watchdog._resolve_translator_model_path()
    assert result == default_gguf, (
        "configured-but-missing 'small' tier did not fall back to the cached "
        "'default' GGUF -> translator would hang forever"
    )


def test_resolve_translator_returns_configured_tier_when_present(monkeypatch, tmp_path):
    import huggingface_hub
    _patch_quant(monkeypatch, "small")
    small_gguf = tmp_path / "translator-small.gguf"
    small_gguf.write_bytes(b"gguf")
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache",
        lambda repo_id, filename, revision=None: (
            str(small_gguf) if filename == _tier_file("small") else None
        ),
    )
    assert watchdog._resolve_translator_model_path() == small_gguf


def test_resolve_translator_returns_none_when_no_tier_cached(monkeypatch):
    """First-run before the wizard download: NO tier cached -> None, so
    the spawn loop legitimately waits for the download (not a quant-tier hang)."""
    import huggingface_hub
    _patch_quant(monkeypatch, "default")
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache",
        lambda repo_id, filename, revision=None: None,
    )
    assert watchdog._resolve_translator_model_path() is None


def test_resolve_translator_probes_cache_at_pinned_revision(monkeypatch, tmp_path):
    """The runtime resolver must probe the HF cache at the tier pin's
    revision, not bare `main`. The wizard downloads the tier at that pinned
    sha, which writes only snapshots/<sha>/ (no refs/main), so a bare probe
    defaults to `main` and misses the warm cache -> llama-server waits
    forever for a model already on disk. Mirrors the whisper/silero
    pinned-probe pattern.

    Seeded defect: drop ``revision=pin.revision`` from the ``_cached``
    probe in _resolve_translator_model_path -> the captured revision is
    None -> red.
    """
    import huggingface_hub
    from yamabiko import models
    _patch_quant(monkeypatch, "default")

    gguf = tmp_path / "translator-default.gguf"
    gguf.write_bytes(b"gguf")
    seen_revisions = []

    def _capture(repo_id, filename, revision=None):
        seen_revisions.append(revision)
        return str(gguf) if filename == _tier_file("default") else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _capture)
    assert watchdog._resolve_translator_model_path() == gguf
    assert seen_revisions, "resolver never probed the cache"
    expected = models.TRANSLATOR_PINS["default"].revision
    assert all(r == expected for r in seen_revisions), (
        f"the runtime resolver must probe at the pinned revision; "
        f"got {seen_revisions!r}"
    )


def test_resolve_translator_treats_stale_cache_path_as_missing(monkeypatch, tmp_path):
    """Fails red if: a cache-index hit pointing at a DELETED file is
    returned as if usable. huggingface_hub can return a string path (or
    its _CACHED_NO_EXIST sentinel) for an entry whose blob was pruned;
    the ``Path(result).is_file()`` guard in _resolve_translator_model_path's
    ``_cached`` must treat that as not-cached so the spawn loop keeps
    waiting instead of handing llama-server a dead --model path
    (mutmut survivors in the 54%-kill resolver section).

    Seeded defect: drop the ``not Path(result).is_file()`` half of the
    guard -> the stale path is returned -> red.
    """
    import huggingface_hub
    _patch_quant(monkeypatch, "default")
    ghost = tmp_path / "pruned-blob.gguf"  # never created on disk
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache",
        lambda repo_id, filename, revision=None: str(ghost),
    )
    assert watchdog._resolve_translator_model_path() is None, (
        "a cache hit whose file no longer exists on disk must resolve "
        "to None (keep waiting), not a dead --model path for llama-server"
    )


def test_resolve_translator_explicit_config_quant_tier_picks_small_tier(
    monkeypatch, tmp_path,
):
    """An explicit --config with ``translation.quant_tier: small`` must
    drive the resolver's tier choice; a bare layered load ignores it and
    a dev reproducing the small-tier path gets the default-tier model.

    Both tiers are present in the fake cache so the other-tier self-heal
    fallback cannot mask a wrong tier choice -- the ONLY way to return
    the small GGUF is to read quant_tier from the explicit file.

    Deliberately NO load_full_config mock: the defect lives in exactly
    that call (bare vs explicit_path=...), so the test exercises the
    REAL config load against a real explicit file, with YAMABIKO_APPDATA
    isolated so the layered store resolves to the shipped default tier.

    Seeded defect: drop the explicit_path threading inside
    _resolve_translator_model_path (back to a bare load_full_config()) ->
    layered quant_tier 'default' wins -> the default GGUF is returned ->
    red.
    """
    import huggingface_hub

    # Layered store isolated to tmp: no user_overrides, so the layered
    # quant_tier is the shipped 'default' -- the wrong answer here.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "translation:\n  quant_tier: small\n", encoding="utf-8",
    )

    small_gguf = tmp_path / "translator-small.gguf"
    small_gguf.write_bytes(b"gguf")
    default_gguf = tmp_path / "translator-default.gguf"
    default_gguf.write_bytes(b"gguf")

    def _cache(repo_id, filename, revision=None):
        if filename == _tier_file("small"):
            return str(small_gguf)
        if filename == _tier_file("default"):
            return str(default_gguf)
        return None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _cache)

    result = watchdog._resolve_translator_model_path(explicit_path=explicit)
    assert result == small_gguf, (
        f"--config quant_tier=small did not pick the small-tier GGUF; got "
        f"{result!r}. The resolver is reading the layered config instead "
        f"of the explicit --config file."
    )


# ---------------------------------------------------------------------------
# _build_llama_cmd argv snapshot.
#
# The entire llama-server command line is locked by these snapshot tests
# so any single-flag drop, reorder, or corruption goes red immediately.
# ---------------------------------------------------------------------------


def test_build_llama_cmd_argv_dev_mode_snapshot(monkeypatch):
    """Fails red if: any llama-server flag is dropped/reordered/corrupted.
    This cmdline IS the translation feature.

    Seeded-defect drill: drop '-fa','on' or change '--port' -> red.
    """
    import sys as _sys
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", False, raising=False)
    argv = wd._build_llama_cmd("C:/models/translator.gguf")
    assert argv[0].endswith("llama-server.exe")
    # Literal values, NOT wd._LLAMA_* -- deriving the expectation from
    # the module constants makes those pairs self-referential (a
    # fat-fingered _LLAMA_PORT would ship green). translator.endpoint in
    # default_config.yaml hardcodes 8090 on the consumer side, and the
    # alias literal is cross-checked against translation.model by
    # tests/test_model_constants.py.
    assert argv[1:] == [
        "--model", "C:/models/translator.gguf",
        "--alias", "qwen3.5-9b",
        "--port", "8090",
        "--ctx-size", "2048",
        "-ngl", "99",
        "-fa", "on",
        "--jinja",
        "--models-max", "1",
        "--sleep-idle-seconds", "-1",
        "--no-webui",
    ]


def test_build_llama_cmd_forces_jinja_without_reasoning_format(monkeypatch):
    """--jinja must be passed EXPLICITLY, not relied on as a vendored-binary
    default: translator.py sends chat_template_kwargs={enable_thinking: False}
    to disable Qwen3 thinking, and llama-server only honors that kwarg when the
    jinja engine is active. A future binary refresh to a --jinja-off default
    would silently re-enable thinking with no flag forcing it on (issue #60,
    rule #7: don't depend on an implicit default for a bundled binary).

    --reasoning-format is deliberately LEFT at the default (auto): forcing
    `none` leaves Qwen3's empty non-thinking <think></think> block unparsed in
    message.content, which false-trips translator.py's <think> guard and fails
    the startup sanity check (a frozen-E2E-caught regression). auto parses that
    block away into message.reasoning_content (None on a clean non-thinking
    reply), which translator.py also guards.

    Seeded-defect drill: drop "--jinja" -> first assert red. Re-add
    "--reasoning-format none" -> second assert red.
    """
    import sys as _sys
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", False, raising=False)
    argv = wd._build_llama_cmd("C:/models/shisa.gguf")
    assert "--jinja" in argv, "llama-server spawn must force --jinja explicitly"
    assert "--reasoning-format" not in argv, (
        "llama-server must NOT force --reasoning-format (auto is required; "
        "none false-trips the <think> guard on Qwen3 non-thinking replies)"
    )


def test_build_llama_cmd_port_parameter_overrides_default(monkeypatch):
    """A caller-supplied port must reach the llama-server argv, so a
    loopback ``translation.endpoint`` on a non-default port (e.g.
    ``http://localhost:9000/...``) spawns the server where the pipeline
    will actually connect -- instead of pinning 8090 and leaving the
    pipeline dialling a port nobody serves.

    Seeded-defect drill: ignore the ``port`` parameter inside
    _build_llama_cmd (keep ``str(_LLAMA_PORT)``) -> argv still carries
    8090 -> red.
    """
    import sys as _sys
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", False, raising=False)
    argv = wd._build_llama_cmd("C:/models/shisa.gguf", port=9000)
    port_idx = argv.index("--port")
    assert argv[port_idx + 1] == "9000", (
        f"--port carries {argv[port_idx + 1]!r}; the caller-supplied "
        f"port=9000 never reached the llama-server argv"
    )


def test_make_real_spawn_llama_passes_port_through_to_llama_cmd(
    monkeypatch, tmp_path,
):
    """The spawn factory's ``port`` argument must flow into the spawned
    llama-server cmdline.

    Seeded-defect drill: drop the ``port=port`` pass-through in
    ``_make_real_spawn_llama._spawn`` -> argv pins 8090 -> red.
    """
    import threading
    from yamabiko import watchdog as wd

    # _spawn opens logs_dir()/llama-server.log for capture; keep that
    # write inside tmp_path, not the developer's real %LOCALAPPDATA%.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["argv"] = argv

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        wd, "_resolve_translator_model_path",
        lambda: __import__("pathlib").Path(__file__),  # is_file() True
    )
    ev = threading.Event()
    wd._make_real_spawn_llama(ev, port=9001)()
    argv = captured["argv"]
    port_idx = argv.index("--port")
    assert argv[port_idx + 1] == "9001", (
        f"--port carries {argv[port_idx + 1]!r}; the factory's port=9001 "
        f"never reached the spawned cmdline"
    )


def test_llama_endpoint_plan_schemaless_external_endpoint_skips_spawn():
    """A schema-less EXTERNAL endpoint value ("remotehost:8090/v1" -- the
    shorthand the llama.cpp docs use) must classify as external ->
    (False, default port). Plain urlsplit reads "remotehost" as the URL
    SCHEME (hostname=None, port=None), which would fall through to "spawn
    the bundled server on the default port" -- ~4 GB VRAM burned on a
    server the pipeline never dials.

    Seeded-defect drill: revert the netloc re-parse to
    ``parts = urlsplit(str(endpoint))`` in _llama_endpoint_plan ->
    hostname parses as None -> plan returns (True, _LLAMA_PORT) -> red.
    """
    from yamabiko.watchdog import _LLAMA_PORT, _llama_endpoint_plan

    plan = _llama_endpoint_plan(
        {"translation": {"endpoint": "remotehost:8090/v1"}}
    )
    assert plan == (False, _LLAMA_PORT), (
        f"schema-less external endpoint planned {plan!r}, expected "
        f"(False, {_LLAMA_PORT}); the bundled llama-server would spawn "
        f"for a host the pipeline never dials"
    )


def test_llama_endpoint_plan_schemaless_loopback_moves_port():
    """Schema-less "localhost:9999" must keep the bundled server AND move
    it to port 9999 -- not pin the default port while the pipeline dials
    9999 (a port nobody serves).

    Seeded-defect drill: plain ``urlsplit(str(endpoint))`` -- the port
    parses as None -> plan returns (True, _LLAMA_PORT) instead of
    (True, 9999) -> red.
    """
    from yamabiko.watchdog import _LLAMA_PORT, _llama_endpoint_plan

    plan = _llama_endpoint_plan({"translation": {"endpoint": "localhost:9999"}})
    assert plan == (True, 9999), (
        f"schema-less loopback endpoint on port 9999 planned {plan!r}; "
        f"llama-server would bind {_LLAMA_PORT} while the pipeline "
        f"dials 9999"
    )


def test_build_llama_cmd_argv_frozen_mode_uses_meipass(monkeypatch):
    """Fails red if: frozen exe-path resolution regresses (ships a
    dev-tree path into the bundle)."""
    import sys as _sys
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "_MEIPASS", r"C:\app\_internal", raising=False)
    argv = wd._build_llama_cmd("m.gguf")
    assert argv[0] == r"C:\app\_internal\llama\llama-server.exe"
    assert "--metrics" not in argv


def test_build_llama_cmd_frozen_falls_back_to_exe_dir_without_meipass(
    monkeypatch,
):
    """Fails red if: the frozen-mode fallback (sys.frozen set but
    sys._MEIPASS absent -- e.g. a onefile/bootloader variant) stops
    resolving llama-server.exe relative to the yamabiko.exe directory
    (the previously-untested ``bundle_internal is None`` branch in
    _build_llama_cmd; part of the 34-survivor 0%-kill section).

    Seeded defect: change the fallback to ``Path(sys.executable).parent``
    (drop the ``/ "_internal"``) -> argv[0] loses the _internal segment
    -> red.
    """
    import sys as _sys
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.delattr(_sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(_sys, "executable", r"C:\app\yamabiko.exe")
    argv = wd._build_llama_cmd("m.gguf")
    assert argv[0] == r"C:\app\_internal\llama\llama-server.exe", (
        f"frozen no-_MEIPASS fallback resolved {argv[0]!r}; expected the "
        f"llama exe under <exe-dir>\\_internal\\llama\\"
    )


# ---------------------------------------------------------------------------
# CJK (non-ASCII) model-path workaround for upstream llama.cpp#18571 (OPEN).
# The vendored llama-server's ggml_fopen corrupts a non-ASCII --model path ->
# "failed to open GGUF file", so a JP user whose model lives under a CJK
# %LOCALAPPDATA% never gets translation.
# Fix: for a non-ASCII model path, pass the ASCII basename as --model and
# spawn with cwd=<model parent> (CreateProcessW sets cwd Unicode-correctly,
# and the broken cast is harmless on ASCII argv bytes). ASCII paths are
# unchanged (absolute --model, no special cwd). Both halves key off the
# single _model_path_is_ascii predicate so they can't drift.
#
# The CJK constant below uses a Japanese username segment; the real
# CUDA-loads-it proof lives in
# tests/test_real_cjk_model_paths.py::test_llama_server_loads_gguf_from_cjk_path.
# ---------------------------------------------------------------------------

_CJK_MODEL_PATH = r"C:\Users\日本\models\shisa-v2.1-qwen3-8B-UD-Q4_K_XL.gguf"
_ASCII_MODEL_PATH = r"C:\Users\alice\models\shisa-v2.1-qwen3-8B-UD-Q4_K_XL.gguf"


def test_build_llama_cmd_ascii_path_uses_absolute_model_arg(monkeypatch):
    """ASCII model path -> --model carries the full ABSOLUTE path, and
    _llama_cwd is None (behaviour unchanged for the common case).

    Seeded-defect drill: make _build_llama_cmd ALWAYS emit the basename
    (drop the ``if _model_path_is_ascii(...)`` guard) -> --model becomes
    the bare filename for an ASCII path -> red here. Equally, make
    _llama_cwd always return the parent -> the cwd assertion below goes red.
    """
    import sys as _sys
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", False, raising=False)

    argv = wd._build_llama_cmd(_ASCII_MODEL_PATH)
    model_idx = argv.index("--model")
    assert argv[model_idx + 1] == _ASCII_MODEL_PATH, (
        f"--model carries {argv[model_idx + 1]!r}; an ASCII path must keep "
        f"the absolute path (only the CJK workaround uses the basename)"
    )
    assert wd._llama_cwd(_ASCII_MODEL_PATH) is None, (
        "_llama_cwd must be None for an ASCII path (no special cwd needed)"
    )


def test_build_llama_cmd_cjk_path_uses_basename_and_parent_cwd(monkeypatch):
    """Non-ASCII model path -> --model carries only the ASCII BASENAME and
    _llama_cwd is the model's PARENT directory (the llama.cpp#18571
    workaround). The absolute non-ASCII path must NOT appear in argv (that
    is exactly what corrupts in ggml_fopen).

    Seeded-defect drill: revert _build_llama_cmd to ``--model str(model_path)``
    unconditionally -> the full CJK path reappears in argv -> red on the
    basename assertion AND the "no absolute CJK path in argv" assertion.
    Or make _llama_cwd return None for the CJK path -> red on the cwd
    assertion (and in production the relative basename would fail to
    resolve).
    """
    import sys as _sys
    from pathlib import Path
    from yamabiko import watchdog as wd
    monkeypatch.setattr(_sys, "frozen", False, raising=False)

    argv = wd._build_llama_cmd(_CJK_MODEL_PATH)
    model_idx = argv.index("--model")
    expected_basename = Path(_CJK_MODEL_PATH).name  # ASCII: shisa-...gguf

    assert argv[model_idx + 1] == expected_basename, (
        f"--model carries {argv[model_idx + 1]!r}; a CJK path must pass the "
        f"ASCII basename {expected_basename!r} so llama.cpp#18571's broken "
        f"char->wchar cast can't corrupt it"
    )
    assert _CJK_MODEL_PATH not in argv, (
        "the absolute non-ASCII model path leaked into the argv; it would "
        "be corrupted by the upstream ggml_fopen cast"
    )
    assert wd._llama_cwd(_CJK_MODEL_PATH) == str(Path(_CJK_MODEL_PATH).parent), (
        f"_llama_cwd returned {wd._llama_cwd(_CJK_MODEL_PATH)!r}; the child "
        f"must run with cwd=<model parent> so the relative --model basename "
        f"resolves"
    )


def test_model_path_is_ascii_classifies_str_and_path_inputs():
    """The linchpin predicate that routes the CJK workaround: pure-ASCII
    paths -> True (absolute --model, unchanged); any non-ASCII component
    -> False (basename + cwd workaround). Covers both str and pathlib.Path
    inputs (the function is called with both: str from the snapshot tests,
    Path from _resolve_translator_model_path in production).

    Seeded-defect drill: replace the body with ``return True`` -> the CJK
    str AND the CJK-Path cases below go red (and in production the CJK
    workaround would never trigger -> the bug returns). Replace with
    ``return False`` -> the two ASCII cases go red.
    """
    from pathlib import Path
    from yamabiko import watchdog as wd

    # ASCII str -> True
    assert wd._model_path_is_ascii(r"C:\Users\alice\models\shisa.gguf") is True
    # CJK str -> False
    assert wd._model_path_is_ascii(r"C:\Users\日本\models\shisa.gguf") is False
    # pathlib.Path with a CJK directory component -> False (str(Path)
    # carries the CJK bytes; this is the production input shape).
    assert wd._model_path_is_ascii(Path(r"C:\Users\日本\models\shisa.gguf")) is False
    # Empty str is trivially ASCII -> True (degenerate but must not crash).
    assert wd._model_path_is_ascii("") is True


def test_llama_cmd_and_cwd_derive_from_same_ascii_predicate(monkeypatch):
    """Consistency guard: --model is relative IFF _llama_cwd is set, both
    driven by _model_path_is_ascii. A path where _build_llama_cmd took the
    basename branch but _llama_cwd returned None (or vice versa) would
    spawn llama with a relative basename and NO cwd -> unresolvable -> the
    CJK workaround is half-applied. This pins the two halves together.

    Seeded-defect drill: change _llama_cwd's condition to a DIFFERENT test
    (e.g. ``len(str(model_path)) > 260``) so it disagrees with
    _build_llama_cmd on the CJK case -> the relative/cwd XOR below goes
    red.
    """
    import sys as _sys
    from yamabiko import watchdog as wd

    # Pin sys.frozen=False like the sibling test_build_llama_cmd_* tests:
    # without it a frozen test runner would make _build_llama_cmd resolve
    # a bogus _MEIPASS llama-exe path, and the "--model is relative iff
    # _llama_cwd is set" invariant this test pins is about the --model
    # ARG, which is independent of the exe-path branch -- but the guard
    # keeps the test deterministic across runners.
    monkeypatch.setattr(_sys, "frozen", False, raising=False)

    for path in (_ASCII_MODEL_PATH, _CJK_MODEL_PATH):
        argv = wd._build_llama_cmd(path)
        model_arg = argv[argv.index("--model") + 1]
        used_basename = (model_arg != str(path))  # relative basename branch
        has_cwd = wd._llama_cwd(path) is not None
        assert used_basename == has_cwd, (
            f"for {path!r}: --model-is-relative={used_basename} but "
            f"_llama_cwd-set={has_cwd}; the argv and cwd halves disagree, "
            f"so the CJK workaround is half-applied (would not load)"
        )


def test_make_real_spawn_llama_passes_cjk_cwd_to_popen(monkeypatch, tmp_path):
    """INVOCATION test: the production _spawn closure must actually pass
    cwd=<model parent> to subprocess.Popen for a non-ASCII model path --
    not just the pure helpers being correct in isolation.

    Spawns through the real _make_real_spawn_llama factory with a fake
    Popen capturing kwargs and _resolve_translator_model_path returning a CJK
    path; asserts the captured cwd is the model's parent AND the argv
    --model is the basename.

    Seeded-defect drill: drop ``cwd=_llama_cwd(model_path)`` from the
    Popen call in _make_real_spawn_llama._spawn -> captured cwd is None
    -> red (and a real JP user's translation would never start).
    """
    import os
    import threading
    from pathlib import Path
    from yamabiko import watchdog as wd

    # _spawn opens logs_dir()/llama-server.log for capture; keep that
    # write inside tmp_path, not the developer's real %LOCALAPPDATA%.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    captured = {}

    class _FakeProc:
        def __init__(self, argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw

    class _FakeModelPath(os.PathLike):
        """Real os.PathLike wrapping a CJK path that reports is_file()=True
        without touching disk -- so Path(model_path), str(model_path), and
        model_path.is_file() in production all behave as with a real Path,
        but no CJK file has to exist on the test runner."""
        _p = Path(_CJK_MODEL_PATH)

        def __fspath__(self):
            return _CJK_MODEL_PATH

        def __str__(self):
            return _CJK_MODEL_PATH

        def is_file(self):
            return True

        @property
        def name(self):
            return self._p.name

        @property
        def parent(self):
            return self._p.parent

    monkeypatch.setattr(wd.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(wd, "_resolve_translator_model_path", lambda: _FakeModelPath())

    ev = threading.Event()
    proc = wd._make_real_spawn_llama(ev)()

    assert isinstance(proc, _FakeProc)
    assert captured["kw"].get("cwd") == str(Path(_CJK_MODEL_PATH).parent), (
        f"_spawn passed cwd={captured['kw'].get('cwd')!r} to Popen; for a "
        f"CJK model path it must be the model parent so the relative "
        f"--model basename resolves"
    )
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == Path(_CJK_MODEL_PATH).name, (
        "_spawn's argv --model must be the ASCII basename for a CJK path"
    )


# ---------------------------------------------------------------------------
# _make_real_spawn_llama factory cancel behavior.
#
# The model-wait loop must check the stop event so a llama_stop_event
# trip during the first-run model-wait doesn't hang the non-daemon
# supervisor thread forever (which would prevent run_watchdog's join
# from returning).
#
# Two factory tests cover:
#   1. Fast path: model already present -> Popen called immediately.
#   2. Cancel path: event already set -> RuntimeError raised, not hang.
# ---------------------------------------------------------------------------


def test_make_real_spawn_llama_spawns_when_model_present(monkeypatch, tmp_path):
    """Fails red if: the spawn factory skips the model-present fast path
    and never calls subprocess.Popen.

    Seeded-defect drill: remove the ``subprocess.Popen(...)`` call inside
    ``_spawn()`` (return None instead) -> ``captured["argv"]`` is never
    set -> KeyError, assertion fails.
    """
    import threading
    from yamabiko import watchdog as wd

    # _spawn opens logs_dir()/llama-server.log for capture; keep that
    # write inside tmp_path, not the developer's real %LOCALAPPDATA%.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["argv"] = argv

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        wd, "_resolve_translator_model_path",
        lambda: __import__("pathlib").Path(__file__),  # is_file() True (this .py file)
    )
    ev = threading.Event()
    proc = wd._make_real_spawn_llama(ev)()
    assert isinstance(proc, _FakePopen)
    assert captured["argv"][0].endswith("llama-server.exe"), (
        f"expected first argv element to end with 'llama-server.exe'; "
        f"got {captured['argv'][0]!r}"
    )


def test_make_real_spawn_llama_cancels_on_event_during_model_wait(monkeypatch):
    """Fails red if: the model-wait loop ignores the event it was given
    (closing over the wrong event would spin/hang forever).

    Seeded-defect drill: change ``while not stop_event.is_set()`` to
    ``while True`` inside ``_make_real_spawn_llama._spawn`` -> this test
    hangs / never raises RuntimeError -> red (timeout or no exception).

    The event is pre-set so the very first loop iteration sees it set and
    raises RuntimeError immediately — no sleeping, no real I/O.
    """
    import threading
    from yamabiko import watchdog as wd

    # _resolve_translator_model_path returns None → model never present, loop
    # falls through to the stop_event check on every iteration.
    monkeypatch.setattr(wd, "_resolve_translator_model_path", lambda: None)
    ev = threading.Event()
    ev.set()  # already cancelled before the first loop iteration
    # Match the specific message so a stray RuntimeError from elsewhere
    # in the spawn path can't satisfy this assertion (Iron Law: the test
    # must fail for the right reason).
    with pytest.raises(RuntimeError, match="stop_event set during model wait"):
        wd._make_real_spawn_llama(ev)()


# ---------------------------------------------------------------------------
# Watchdog.request_restart() — wedge-driven child restart.
#
# The HealthProbe wedge detector calls request_restart() when the pipeline
# process is alive but /livez is unresponsive. That call must terminate the
# current child WITHOUT setting stop_event (which would stop the supervisor
# instead of restarting). The existing _wait_or_stop + restart loop in run()
# then sees the child exit and respawns it normally.
# ---------------------------------------------------------------------------


def test_request_restart_terminates_current_child_without_stopping():
    """Fails red if: request_restart sets stop_event (it must NOT --
    that would stop the supervisor instead of restarting the child),
    OR fails to terminate the current child.

    Seeded-defect drill: make request_restart call self._stop_event.set()
    -> assert-not-set goes red. Alternatively, remove the proc.terminate()
    call -> child.terminated stays False -> assert fails.
    """
    from yamabiko.watchdog import Watchdog

    class _Child:
        def __init__(self):
            self.terminated = False
        def terminate(self):
            self.terminated = True

    wd = Watchdog(name="pipeline", spawn=lambda: _Child())
    child = _Child()
    wd._current_proc = child  # simulate "currently supervising"
    wd.request_restart()
    assert child.terminated is True
    assert not wd.stop_event.is_set()


def test_request_restart_tolerates_no_current_child():
    """Fails red if: request_restart raises when no child is running
    (race: probe fires between spawn attempts).

    Seeded-defect drill: change ``if proc is None: return`` to
    ``if proc is None: raise RuntimeError("no child")`` -> this test
    fails with RuntimeError instead of completing cleanly.
    """
    from yamabiko.watchdog import Watchdog
    wd = Watchdog(name="pipeline", spawn=lambda: None)
    wd._current_proc = None
    wd.request_restart()  # must not raise


def test_request_restart_propagates_programming_errors():
    """Fails red if: request_restart's except clause swallows
    AttributeError-class programming bugs (the broad ``except Exception``
    demotes a wrong-attribute or wrong-type bug in the terminate path to a
    routine-looking WARN, so the wedge restart silently never happens).

    Seeded-defect drill: revert ``except OSError`` back to ``except
    Exception`` in request_restart -> the AttributeError below is
    swallowed, pytest.raises gets nothing -> red.
    """
    from yamabiko.watchdog import Watchdog

    class _BrokenChild:
        def terminate(self):
            raise AttributeError("simulated programming bug in terminate path")

    wd = Watchdog(name="pipeline", spawn=lambda: _BrokenChild())
    wd._current_proc = _BrokenChild()
    with pytest.raises(AttributeError):
        wd.request_restart()


def test_request_restart_swallows_oserror_from_terminate():
    """Paired negative for the except-narrowing: an OSError from
    terminate() (the child died in the race window -- an expected
    OS-level outcome, not a programming bug) must stay swallowed so the
    HealthProbe thread survives.

    Seeded-defect drill: delete the try/except around proc.terminate()
    in request_restart -> the OSError below propagates -> red.
    """
    from yamabiko.watchdog import Watchdog

    class _DyingChild:
        def terminate(self):
            raise OSError("process already gone")

    wd = Watchdog(name="pipeline", spawn=lambda: _DyingChild())
    wd._current_proc = _DyingChild()
    wd.request_restart()  # must not raise


def test_run_honors_request_restart_that_lands_during_spawn_publication():
    """There must be NO gap between ``_spawn()`` returning and
    ``_current_proc`` publication in run(). If the HealthProbe's
    on_wedged fires in that window, the restart request must reach the
    child -- not silently no-op (leaving a hung pipeline unserved for
    another full fail_budget_s ~120s).

    The spawn callable blocks at exactly the window instant (child
    "exists", not yet published); a probe thread calls request_restart()
    right then. Under the fixed code the publication happens under the
    same lock acquisition as the spawn, so request_restart blocks until
    the child is visible and then terminates it. Under the broken code
    request_restart reads _current_proc=None and no-ops -> the child is
    never terminated -> red.

    Seeded-defect drill: move the ``self._current_proc = proc``
    publication back OUTSIDE the lock acquisition that wraps
    ``self._spawn()`` -> request_restart no-ops in the window ->
    proc.terminated never set -> red.
    """
    import subprocess as _sp
    import threading
    import time as _time

    class _HangingProc:
        def __init__(self):
            self.terminated = threading.Event()

        def wait(self, timeout=None):
            if self.terminated.wait(timeout=timeout):
                return 0  # clean exit -> STOP_SELF ends run() after this child
            raise _sp.TimeoutExpired(cmd="fake", timeout=timeout)

        def terminate(self):
            self.terminated.set()

    proc = _HangingProc()
    in_spawn = threading.Event()
    release_spawn = threading.Event()

    def _spawn():
        # The publication window: the child is about to exist but run()
        # has not yet stored it on _current_proc.
        in_spawn.set()
        release_spawn.wait(timeout=5.0)
        return proc

    wd = watchdog.Watchdog(
        name="pipeline",
        spawn=_spawn,
        sleep=lambda s: None,
        poll_interval_s=0.01,
        grace_s=0.05,
        min_lifetime_s=0.0,
    )

    def _probe_thread():
        in_spawn.wait(timeout=5.0)
        # Fires while run() is mid-spawn. Fixed code: blocks on the
        # publication lock, then terminates the fresh child. Broken
        # code: reads None and returns -- the request is dropped.
        wd.request_restart()

    probe = threading.Thread(target=_probe_thread, daemon=True)
    runner = threading.Thread(target=wd.run, daemon=True)
    runner.start()
    probe.start()
    assert in_spawn.wait(timeout=5.0), "spawn was never entered"
    # Best-effort scheduling window for the probe thread to reach
    # request_restart while run() is still mid-spawn (parks on the lock
    # under fixed code; no-ops under broken code). NOT a guarantee: if
    # the probe is late, broken code ALSO terminates the published
    # child and this test passes -- i.e. a missed detection, never a
    # false red.
    _time.sleep(0.05)
    release_spawn.set()
    try:
        assert proc.terminated.wait(timeout=2.0), (
            "request_restart landed during the spawn-publication window "
            "and was DROPPED: the freshly-spawned child was never "
            "terminated. The publication-window gap is open -- a wedged "
            "pipeline stays unserved for another full fail_budget_s."
        )
        runner.join(timeout=2.0)
        assert not runner.is_alive(), "run() did not exit after STOP_SELF"
    finally:
        wd.stop_event.set()
        proc.terminated.set()
        release_spawn.set()
        runner.join(timeout=2.0)


def test_requested_restart_of_a_young_child_restarts_instead_of_giving_up():
    """Decision 21: a child the supervisor terminated itself
    (request_restart, here llama-server after a translation escalation) is
    respawned even when it lived under min_lifetime_s.

    Seeded defect: drop the requested-restart exemption -> the terminate
    reads as a fast crash and the supervisor gives up after one spawn -> red."""
    spawns: list = []
    gave_up: list = []
    holder: dict = {}

    class _Child(_FakeProcess):
        def wait(self, timeout=None) -> int:
            if len(spawns) == 1:
                holder["wd"].request_restart()
                return 1
            return 0

    def spawn():
        child = _Child()
        spawns.append(child)
        return child

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 1.0
        return clock_t[0]

    wd = watchdog.Watchdog(spawn=spawn, clock=clock, sleep=lambda s: None,
                           min_lifetime_s=30.0,
                           on_give_up=lambda title, message: gave_up.append(title))
    holder["wd"] = wd
    wd.run()
    assert len(spawns) == 2 and spawns[0].terminated
    assert gave_up == []


def test_restart_hook_sees_the_exit_code_before_the_respawn():
    """Decision 18: the pipeline supervisor hands each restart's exit code to
    on_restart before it spawns the next child, so the carry reaches it.
    Seeded defect: call the hook after the spawn, or not at all -> red."""
    events: list = []
    codes = deque([45, 0])

    def spawn():
        events.append("spawn")
        return _FakeProcess(exit_code=codes.popleft())

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    wd = watchdog.Watchdog(spawn=spawn, clock=clock, sleep=lambda s: None,
                           min_lifetime_s=5.0, on_restart=events.append)
    wd.run()
    assert events == ["spawn", 45, "spawn"]


def test_carry_relay_hands_kinds_to_the_next_pipeline_and_restarts_llama_for_a_translator_carry():
    """Decision 18: the kinds reach exactly the next spawn's environment; a
    translator carry restarts llama-server; no carry clears an inherited
    variable. Seeded defects: leave the variable for every later spawn, or
    restart llama for an audio carry -> red."""
    from yamabiko import problems

    carries = deque([
        problems.Carry(("audio",), False),
        problems.Carry(("translator",), True),
        None,
    ])
    relay = watchdog._CarryRelay(take=carries.popleft)
    restarts: list = []
    relay.llama = SimpleNamespace(request_restart=lambda: restarts.append(True))

    relay.on_pipeline_restart(44)
    env: dict = {}
    relay.apply(env)
    assert env == {problems.CARRY_ENV: "audio"} and restarts == []
    env = {}
    relay.apply(env)
    assert env == {}

    relay.on_pipeline_restart(45)
    env = {problems.CARRY_ENV: "stale"}
    relay.apply(env)
    assert env == {problems.CARRY_ENV: "translator"} and restarts == [True]

    relay.on_pipeline_restart(1)
    env = {problems.CARRY_ENV: "stale"}
    relay.apply(env)
    assert env == {}


def test_carry_relay_hands_announced_shortcuts_to_the_next_pipeline_only():
    """Decision 72: the shortcuts the previous pipeline announced as taken
    reach exactly the next spawn's environment, also with no problem kind.
    Seeded defects: drop them, or leave the variable for every later spawn
    -> red."""
    from yamabiko import problems

    relay = watchdog._CarryRelay(
        take=lambda: problems.Carry((), False, {"pause_resume": "Ctrl+Alt+P"}))
    relay.on_pipeline_restart(45)
    env: dict = {}
    relay.apply(env)
    assert problems.hotkeys_from_env(env.pop(problems.CARRY_HOTKEYS_ENV)) == {
        "pause_resume": "Ctrl+Alt+P"}
    assert env == {}
    env = {problems.CARRY_HOTKEYS_ENV: "stale"}
    relay.apply(env)
    assert env == {}


# ---------------------------------------------------------------------------
# llama-server child-output capture (llama-server.log).
#
# Upstream llama-server logs to stdout only (--log-file is unreliable:
# ggml-org discussions #6786 / #7913), so the watchdog captures the
# child's stdout+stderr at the subprocess seam into
# logs_dir()/llama-server.log. These tests pin the redirect wiring, the
# pre-spawn 10 MB rotation, and the open-failure degrade (never block
# the spawn on logging). The REAL Popen seam -- a child that actually
# writes to stderr landing in the file -- is covered at the real tier
# by tests/test_real_watchdog_subprocess.py.
# ---------------------------------------------------------------------------


def test_llama_spawn_redirects_child_stdout_and_stderr_to_llama_server_log(
    monkeypatch, tmp_path, caplog,
):
    """The production ``_spawn`` closure must hand subprocess.Popen an
    APPEND-mode utf-8 handle to logs_dir()/llama-server.log as stdout,
    fold stderr into it via ``subprocess.STDOUT``, close the PARENT's
    copy of the handle once Popen returns (the child keeps its own
    inherited handle), and log one INFO line naming the file.

    Seeded-defect drills (each one line):
      - drop ``stderr=subprocess.STDOUT`` from the Popen kwargs -> the
        stderr assert goes red (llama's error output -- the part that
        matters for a crash -- would vanish again).
      - open the log with mode "w" -> the append-mode assert goes red
        (each respawn would clobber the previous crash's evidence).
      - drop the INFO line naming the file -> the caplog assert goes red.
    """
    import logging as _logging
    import threading
    from pathlib import Path
    from yamabiko import watchdog as wd

    caplog.set_level(_logging.INFO, logger="yamabiko.watchdog")
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        wd, "_resolve_translator_model_path",
        lambda: Path(__file__),  # is_file() True (this .py file)
    )

    proc = wd._make_real_spawn_llama(threading.Event())()

    assert isinstance(proc, _FakePopen)
    kw = captured["kw"]
    fh = kw.get("stdout")
    assert fh is not None, (
        "Popen received no stdout redirect; llama-server's output is "
        "discarded and a llama crash leaves zero evidence on disk"
    )
    assert Path(fh.name) == tmp_path / "logs" / "llama-server.log", (
        f"stdout redirect targets {fh.name!r}; expected "
        f"logs_dir()/llama-server.log"
    )
    assert fh.mode == "a", (
        f"llama-server.log opened with mode {fh.mode!r}; append mode is "
        f"required so a respawn doesn't clobber the previous crash's tail"
    )
    assert kw.get("stderr") is wd.subprocess.STDOUT, (
        f"stderr={kw.get('stderr')!r}; must be subprocess.STDOUT so "
        f"llama's error output folds into the same capture file"
    )
    assert fh.closed, (
        "the parent's copy of the llama-server.log handle must be closed "
        "once Popen returns (the child holds its own inherited handle); "
        "a lingering parent handle would block the pre-spawn rotation "
        "rename for the rest of the watchdog's life"
    )
    assert "llama-server.log" in caplog.text, (
        "no INFO line names the capture file; a triager reading "
        "main-watchdog.log can't discover where llama's output went"
    )


def test_llama_spawn_rotates_oversize_llama_server_log_before_spawn(
    monkeypatch, tmp_path,
):
    """A llama-server.log over 10 MB at spawn time is ``os.replace``d to
    llama-server.log.1 BEFORE the child gets the handle, so the capture
    file cannot grow without bound across respawns.

    Seeded-defect drill: drop the pre-spawn rotation branch (or flip its
    size comparison) -> the oversize file keeps accumulating and the .1
    backup never appears -> both asserts go red.
    """
    import threading
    from pathlib import Path
    from yamabiko import watchdog as wd

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    oversize = wd._LLAMA_LOG_MAX_BYTES + 1
    # Sparse-seek write: a >10 MB file without 10 MB of actual I/O.
    with open(logs / "llama-server.log", "wb") as f:
        f.seek(oversize - 1)
        f.write(b"x")

    class _FakePopen:
        def __init__(self, argv, **kw):
            pass

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(wd, "_resolve_translator_model_path", lambda: Path(__file__))

    wd._make_real_spawn_llama(threading.Event())()

    rotated = logs / "llama-server.log.1"
    assert rotated.is_file() and rotated.stat().st_size == oversize, (
        f"oversize llama-server.log was not rotated to .1 before spawn; "
        f"logs dir: {[p.name for p in logs.iterdir()]}"
    )
    assert (logs / "llama-server.log").stat().st_size == 0, (
        "post-rotation llama-server.log should start empty; the child "
        "would otherwise keep appending to an unbounded file"
    )


def test_llama_spawn_warns_when_log_rotation_fails_and_degrades_to_append(
    monkeypatch, tmp_path, caplog,
):
    """When the pre-spawn ``os.replace`` rotation of an oversize
    llama-server.log fails, the spawn must degrade to appending to the
    oversize file (llama-server is never blocked on logging) AND leave
    a WARNING naming the capture path and the error. The old bare
    ``except OSError: pass`` was an evidence-free swallow inside the
    very feature that exists to stop evidence-free swallows.

    The failure is real, not a mock of the subject: a DIRECTORY planted
    at llama-server.log.1 makes ``os.replace`` raise OSError.

    Seeded-defect drill: revert the rotation except-branch to a bare
    ``pass`` -> the append degrade still happens but no WARNING record
    exists -> red.
    """
    import logging as _logging
    import threading
    from pathlib import Path
    from yamabiko import watchdog as wd

    caplog.set_level(_logging.WARNING, logger="yamabiko.watchdog")
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    oversize = wd._LLAMA_LOG_MAX_BYTES + 1
    # Sparse-seek write: a >10 MB file without 10 MB of actual I/O.
    with open(logs / "llama-server.log", "wb") as f:
        f.seek(oversize - 1)
        f.write(b"x")
    # A directory squatting on the rotation target makes os.replace
    # raise a REAL OSError (PermissionError on Windows).
    (logs / "llama-server.log.1").mkdir()

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["kw"] = kw

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(wd, "_resolve_translator_model_path", lambda: Path(__file__))

    proc = wd._make_real_spawn_llama(threading.Event())()

    # Degrade-to-append semantics preserved: the child still gets a
    # capture handle to the (oversize) file.
    assert isinstance(proc, _FakePopen), (
        "spawn did not return a child after a rotation failure; "
        "llama-server must never be blocked on logging"
    )
    fh = captured["kw"].get("stdout")
    assert fh is not None and Path(fh.name) == logs / "llama-server.log", (
        f"rotation failure dropped the capture redirect entirely "
        f"(stdout={fh!r}); the degrade contract is append-to-oversize, "
        f"not lose-the-output"
    )
    rotation_warnings = [
        r for r in caplog.records
        if r.levelno >= _logging.WARNING and "rotation" in r.getMessage()
    ]
    assert rotation_warnings, (
        "the failed rotation left no WARNING; the evidence-free "
        "'except OSError: pass' swallow is back"
    )
    msg = rotation_warnings[0].getMessage()
    assert "llama-server.log" in msg, (
        f"the rotation WARNING does not name the capture path: {msg!r}"
    )


def test_llama_spawn_degrades_to_no_redirect_when_log_open_fails(
    monkeypatch, tmp_path, caplog,
):
    """When llama-server.log cannot be opened (read-only / pre-empted
    logs dir), the spawn must proceed WITHOUT the redirect and log a
    warning -- llama-server must never be blocked on logging.

    The logs path is planted as a FILE so ``paths.ensure_dirs()`` raises
    a real OSError (FileExistsError) on the open path -- the failure is
    real, not a mock of the subject.

    Seeded-defect drill: let the OSError escape the log-open helper
    (delete its outer try/except) -> the spawn raises instead of
    returning a child -> red.
    """
    import logging as _logging
    import threading
    from pathlib import Path
    from yamabiko import watchdog as wd

    caplog.set_level(_logging.INFO, logger="yamabiko.watchdog")
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    # logs_dir() resolves to tmp_path/logs; planting a FILE there makes
    # mkdir(parents=True, exist_ok=True) raise FileExistsError (OSError).
    (tmp_path / "logs").write_text("not a directory", encoding="utf-8")

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(wd, "_resolve_translator_model_path", lambda: Path(__file__))

    proc = wd._make_real_spawn_llama(threading.Event())()

    assert isinstance(proc, _FakePopen), (
        "spawn did not return a child when the capture-log open failed; "
        "llama-server must never be blocked on logging"
    )
    kw = captured["kw"]
    assert "stdout" not in kw and "stderr" not in kw, (
        f"Popen still received redirect kwargs {sorted(kw)} after the "
        f"log open failed; a closed/invalid handle would kill the spawn"
    )
    warnings = [
        r for r in caplog.records
        if r.levelno >= _logging.WARNING and "llama-server" in r.getMessage()
    ]
    assert warnings, (
        "no WARNING was logged about the failed llama-server.log open; "
        "the degrade must leave evidence of the missing capture"
    )


def test_llama_spawn_hides_console_window(monkeypatch, tmp_path):
    """The production ``_spawn`` closure must pass CREATE_NO_WINDOW to
    subprocess.Popen: the frozen yamabiko.exe is windowed (console=False),
    so a console-subsystem llama-server spawned without the flag makes
    Windows allocate a visible terminal on the user's desktop (live
    report 2026-07-07; dev mode masks it because the parent already owns
    a console). The frozen-bundle seam -- no visible window on the real
    child -- is asserted in tests/test_real_frozen_e2e.py.

    Seeded-defect drill (one line): drop the ``creationflags`` kwarg from
    the Popen call in ``_make_real_spawn_llama._spawn`` -> the
    no-creationflags assert goes red.
    """
    import threading
    from pathlib import Path
    from yamabiko import watchdog as wd

    # _spawn opens logs_dir()/llama-server.log for capture; keep that
    # write inside tmp_path, not the developer's real %LOCALAPPDATA%.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    captured = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["kw"] = kw

    monkeypatch.setattr(wd.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(wd, "_resolve_translator_model_path", lambda: Path(__file__))

    proc = wd._make_real_spawn_llama(threading.Event())()

    assert isinstance(proc, _FakePopen)
    flags = captured["kw"].get("creationflags")
    assert flags is not None, (
        "Popen received no creationflags; a console-subsystem "
        "llama-server child of the windowed yamabiko.exe gets a visible "
        "terminal window allocated by Windows"
    )
    assert flags & wd.subprocess.CREATE_NO_WINDOW, (
        f"creationflags={flags:#x} lacks CREATE_NO_WINDOW; Windows "
        f"allocates a visible console for the llama-server child"
    )


# ---------------------------------------------------------------------------
# Watchdog self-crash evidence, layer 2: the supervision loop's own
# exceptions must surface as an internal-error give-up + cascade -- a
# silently-dead supervisor thread mimics a clean exit 0.
# ---------------------------------------------------------------------------


def test_watchdog_run_surfaces_internal_supervision_error_as_giveup_and_cascade(
    monkeypatch, caplog,
):
    """An exception thrown by the supervision logic itself (seeded here
    into ``classify``) must NOT propagate out of ``run()`` and silently
    kill the supervisor thread. run() catches it, logs the traceback,
    fires the give-up dialog with the internal-error title, and cascades
    stop so peer supervisors + the single-instance mutex drop.

    Pre-fix behavior (the bug this pins): the exception escaped run(),
    the thread died, run_watchdog's join returned, and the process
    exited 0 -- indistinguishable from a clean user quit.

    Seeded-defect drill: remove the try/except guard in run() (call the
    supervision loop directly) -> the seeded RuntimeError propagates out
    of wd.run() -> red.
    """
    import logging as _logging

    caplog.set_level(_logging.ERROR, logger="yamabiko.watchdog")

    def _seeded_classify(code):
        raise RuntimeError("seeded supervision bug in classify")

    monkeypatch.setattr(watchdog, "classify", _seeded_classify)

    def spawn() -> _FakeProcess:
        return _FakeProcess(exit_code=1, elapsed_s=60.0)

    clock_t = [0.0]

    def clock() -> float:
        clock_t[0] += 60.0
        return clock_t[0]

    giveups: list[tuple[str, str]] = []
    wd = watchdog.Watchdog(
        spawn=spawn, clock=clock, sleep=lambda s: None, min_lifetime_s=5.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )

    wd.run()  # must NOT raise

    assert giveups, (
        "the internal supervision error produced no give-up; the "
        "supervisor died silently (the exact 'silent exit 0' mimic)"
    )
    title, detail = giveups[0]
    assert title == "Yamabiko hit an internal error", (
        f"give-up title {title!r}; an internal supervision bug must be "
        f"labelled as such, not as a child crash"
    )
    assert "seeded supervision bug in classify" in detail, (
        f"give-up detail {detail!r} does not carry the causing error; "
        f"the dialog is undiagnosable"
    )
    assert wd.stop_event.is_set(), (
        "internal-error give-up did not cascade; peer supervisors + the "
        "single-instance mutex would hang forever"
    )
    assert any(r.exc_info for r in caplog.records), (
        "no traceback was logged for the internal supervision error"
    )
    assert "internal error" in caplog.text and "RuntimeError" in caplog.text


# ---------------------------------------------------------------------------
# logging.level -> overlay child argv (--log-level).
#
# _build_overlay_cmd takes the level as a PARAMETER from the caller's
# already-loaded config. It must never call load_full_config() itself:
# a fresh disk read from inside the pipeline process silently ignores
# an explicit --config file (the documented bug class from
# run_pipeline's lifecycle block). The producer->consumer round-trip
# (app.cfg -> spawn closure -> argv) lives in tests/test_build_pipeline
# .py::test_run_pipeline_overlay_log_level_round_trips_from_explicit_config.
# ---------------------------------------------------------------------------


def test_build_overlay_cmd_forwards_caller_supplied_level(monkeypatch):
    """The ``level`` parameter must ride the overlay child's argv as
    ``--log-level``: the overlay parses that flag
    (overlay_app/__main__.py) but nothing forwarded it, so
    ``logging.level: DEBUG`` raised pipeline verbosity while the
    overlay silently stayed at INFO.

    Seeded-defect drill: drop the ``cmd.extend(["--log-level", ...])``
    line from _build_overlay_cmd -> "--log-level" absent from the argv
    -> red.
    """
    import sys as _sys
    if hasattr(_sys, "frozen"):
        monkeypatch.delattr(_sys, "frozen", raising=False)
    cmd = watchdog._build_overlay_cmd(level="DEBUG")
    assert "--log-level" in cmd, f"--log-level missing from overlay argv {cmd}"
    assert cmd[cmd.index("--log-level") + 1] == "DEBUG", (
        f"overlay argv carries --log-level "
        f"{cmd[cmd.index('--log-level') + 1]!r}; the caller's DEBUG "
        f"never reached the overlay child"
    )


def test_build_overlay_cmd_normalizes_lowercase_level_for_the_child(
    monkeypatch,
):
    """A lowercase config value (``logging.level: debug``) must reach
    the child as the stdlib name ``DEBUG``: the overlay's own
    ``setLevel`` rejects lowercase with ValueError, so forwarding it
    verbatim would crash-loop the overlay child.

    Seeded-defect drill: forward the raw value (drop the
    normalization) -> argv carries "debug" -> red.
    """
    cmd = watchdog._build_overlay_cmd(level="debug")
    assert cmd[cmd.index("--log-level") + 1] == "DEBUG", (
        f"lowercase 'debug' was not normalized for the overlay child: "
        f"{cmd}; setLevel('debug') raises ValueError in the child"
    )


def test_build_overlay_cmd_falls_back_to_info_on_unknown_level_with_warning(
    caplog,
):
    """An unknown level value (typo'd user_overrides, numeric YAML) must
    degrade to the documented INFO default WITH a WARNING naming the
    bad value -- never block the overlay spawn, never forward a value
    the child's setLevel would reject.

    Seeded-defect drill: forward the raw value unvalidated -> argv
    carries "verbose" -> red; or degrade silently (drop the WARNING)
    -> the caplog assert goes red.
    """
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="yamabiko.watchdog")
    cmd = watchdog._build_overlay_cmd(level="verbose")
    assert cmd[cmd.index("--log-level") + 1] == "INFO", (
        f"unknown-level fallback drifted: {cmd}"
    )
    warnings = [
        r for r in caplog.records
        if r.levelno >= _logging.WARNING and "verbose" in r.getMessage()
    ]
    assert warnings, (
        "no WARNING names the rejected logging.level value; the user's "
        "typo silently downgrades the overlay to INFO with zero evidence"
    )


def test_build_overlay_cmd_defaults_to_info_when_caller_has_no_level():
    """``level=None`` (caller's config has no logging.level, or the
    caller couldn't load config at all) degrades to the documented INFO
    default -- and does NOT warn, because an absent key is not an error.

    Seeded-defect drill: change the default fallback to DEBUG ->
    the INFO assert goes red.
    """
    cmd = watchdog._build_overlay_cmd()
    assert cmd[cmd.index("--log-level") + 1] == "INFO", (
        f"absent-level default drifted: {cmd}"
    )


# --- stop-driven spawn aborts must not masquerade as startup failures ---
#
# The llama spawn raises RuntimeError BY DESIGN when stop_event is set
# during its model-wait (_make_real_spawn_llama), and that path fires on
# every first-run wizard cancel (pipeline exits 42 -> STOP_ALL cascade
# sets llama_stop_event mid-wait). The supervision loop's spawn-failure
# handler must distinguish that clean cancel from a genuine spawn
# failure.


def test_stop_driven_spawn_abort_records_no_give_up():
    """Wizard-cancel shape: stop_event set, THEN the in-flight spawn
    raises. No "Yamabiko failed to start" dialog may be queued.

    Seeded defect: remove the stop_event.is_set() early-return from the
    spawn-exception handler -> giveups gets the bogus entry -> red.
    """
    from yamabiko import watchdog

    giveups: list[tuple[str, str]] = []
    holder: dict = {}

    def spawn():
        # The cascade lands while the spawn is blocked in its model
        # wait; the spawn then raises its documented clean-cancel error.
        holder["wd"]._stop_event.set()
        raise RuntimeError(
            "llama-server supervisor: stop_event set during model wait"
        )

    wd = watchdog.Watchdog(
        spawn=spawn, clock=lambda: 0.0, sleep=lambda s: None,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    holder["wd"] = wd
    wd.run()

    assert giveups == [], (
        f"a stop-driven spawn abort is a clean cancel, not a startup "
        f"failure; queued give-ups: {giveups!r}"
    )


def test_genuine_spawn_failure_still_records_give_up():
    """Companion guard: with NO stop requested, a spawn exception must
    keep its terminal give-up (bad exe path, model never appeared).

    Seeded defect: over-broaden the new early-return (e.g. return on any
    RuntimeError) -> giveups stays empty -> red.
    """
    from yamabiko import watchdog

    giveups: list[tuple[str, str]] = []

    def spawn():
        raise RuntimeError("bad exe path")

    wd = watchdog.Watchdog(
        spawn=spawn, clock=lambda: 0.0, sleep=lambda s: None,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(giveups) == 1 and "could not be spawned" in giveups[0][1], (
        f"a genuine spawn failure must surface the startup give-up; "
        f"got {giveups!r}"
    )


def test_build_main_cmd_forwards_max_segments_equals_form(monkeypatch):
    """``--max-segments=5`` (one argv token) was silently dropped by the
    old exact-token membership test -- the precise bug class the
    adjacent --config forwarding fix documents. The child must receive
    the flag in canonical re-emitted form.

    Seeded defect: revert _build_main_cmd to the ``if flag in argv``
    token match -> the equals form vanishes from the child argv -> red.
    """
    monkeypatch.setattr(
        watchdog.sys, "argv", ["yamabiko.exe", "--max-segments=5"],
    )
    monkeypatch.delattr(watchdog.sys, "frozen", raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/python.exe")
    assert watchdog._build_main_cmd() == [
        "C:/fake/python.exe", "-m", "yamabiko.main",
        "--max-segments", "5",
    ]


def test_build_main_cmd_warns_when_max_segments_value_is_unparseable(
    monkeypatch, caplog,
):
    """``--max-segments abc`` cannot be forwarded (argparse SystemExits on
    the int conversion), and crashing the top-level process over it would
    be worse than ignoring it. But the drop must not be SILENT: the child
    then runs unbounded, and the user who asked for a bounded run gets no
    signal anywhere -- in the frozen bundle argparse's own usage text goes
    to a discarded stderr. Name the ignored value at WARNING.

    Seeded defect: delete the WARNING from the SystemExit handler in
    _build_main_cmd -> the drop is silent again -> red (the flag-absence
    assertion still passes, which is exactly why the log assertion has to
    be here).
    """
    monkeypatch.delattr(watchdog.sys, "frozen", raising=False)
    monkeypatch.setattr(watchdog.sys, "executable", "C:/fake/python.exe")
    monkeypatch.setattr(
        watchdog.sys, "argv", ["yamabiko.exe", "--max-segments", "abc"],
    )
    caplog.set_level(logging.WARNING, logger="yamabiko.watchdog")

    cmd = watchdog._build_main_cmd()

    assert cmd == ["C:/fake/python.exe", "-m", "yamabiko.main"], (
        f"an unparseable --max-segments must not be forwarded; got {cmd!r}"
    )
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any(
        "max-segments" in m and "abc" in m for m in warnings
    ), (
        f"the dropped --max-segments value was not named in any WARNING; "
        f"the child runs unbounded with zero signal. Warnings: {warnings!r}"
    )


def test_slow_spawn_does_not_consume_the_childs_fast_crash_budget():
    """The lifetime clock must start AFTER _spawn returns. The llama spawn
    blocks in its model-wait (minutes to hours on a first-run download), so
    charging that wait to child lifetime made a broken-DLL insta-crash read
    as a long healthy run: the fast-crash give-up was skipped and the
    supervisor churned a full restart cycle instead of telling the user the
    install is broken.

    The fake clock is advanced INSIDE the spawn seam (a 120 s model wait);
    the child then dies 1 s into its own life, well inside min_lifetime_s.

    Seeded defect: move ``start = self._clock()`` back above the spawn
    try-block -> elapsed becomes 121 s -> the fast-crash branch is skipped
    -> the supervisor restarts to its anti-flap cap and the give-up copy
    becomes the generic crash-loop one -> red on both assertions.
    """
    now = [0.0]
    spawns: list = []
    giveups: list[tuple[str, str]] = []

    class _SlowSpawnChild:
        def wait(self, timeout=None) -> int:
            now[0] += 1.0  # the child itself lives 1 s, then crashes
            return 1

        def terminate(self) -> None:
            pass

    def spawn() -> _SlowSpawnChild:
        # The model-wait: elapses long before the child process exists.
        now[0] += 120.0
        p = _SlowSpawnChild()
        spawns.append(p)
        return p

    wd = watchdog.Watchdog(
        name="llama", spawn=spawn, clock=lambda: now[0],
        sleep=lambda s: None, min_lifetime_s=30.0,
        on_give_up=lambda t, d: giveups.append((t, d)),
    )
    wd.run()

    assert len(spawns) == 1, (
        f"a child that lived 1 s after a 120 s spawn wait was read as "
        f"healthy and restarted ({len(spawns)} spawns): the model-wait is "
        f"being charged to child lifetime"
    )
    assert giveups and giveups[0][0] == "Yamabiko failed to start", (
        f"expected the fast-crash config-error give-up; got {giveups!r}"
    )


# --- per-spawn Job Object membership --------------------------------------
# build_watchdog self-assigns the watchdog process to the outer job, so every
# child inherits membership by kernel rule and no per-child assign runs. That
# removed the only per-spawn evidence that the kill cascade is intact: a
# future breakaway flag (or an external-owner job that refuses nesting) would
# leave children outside the job, and nothing would say so until orphaned
# llama-servers showed up in someone's VRAM.


def test_spawned_child_outside_any_job_object_logs_a_warning(caplog):
    """A spawned child that the OS reports as NOT a job member must produce
    a WARNING naming the child, because the KILL_ON_JOB_CLOSE cascade can no
    longer reach it when the watchdog dies.

    Seeded defect: delete the membership check from the supervision loop ->
    no WARNING -> red.
    """
    caplog.set_level(logging.WARNING, logger="yamabiko.watchdog")

    wd = watchdog.Watchdog(
        name="pipeline",
        spawn=lambda: _FakeProcess(exit_code=0, elapsed_s=10.0),
        clock=lambda: 0.0, sleep=lambda s: None,
        job=None,
        in_job_check=lambda proc: False,
    )
    wd.run()

    warnings = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any(
        "Job Object" in m and "pipeline" in m for m in warnings
    ), (
        f"a child spawned outside the Job Object was not reported; the kill "
        f"cascade is broken and nothing says so. Warnings: {warnings!r}"
    )


def test_spawned_child_inside_a_job_object_logs_no_warning(caplog):
    """Paired guard: the membership check must be a CHECK, not an
    unconditional warning. The happy path (child inherited the outer job)
    stays silent, or the warning is noise on every start and nobody reads
    it when it matters.

    Seeded defect: warn unconditionally (drop the ``is False`` test) -> red.
    """
    caplog.set_level(logging.WARNING, logger="yamabiko.watchdog")

    wd = watchdog.Watchdog(
        name="pipeline",
        spawn=lambda: _FakeProcess(exit_code=0, elapsed_s=10.0),
        clock=lambda: 0.0, sleep=lambda s: None,
        job=None,
        in_job_check=lambda proc: True,
    )
    wd.run()

    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and "Job Object" in r.getMessage()
    ]
    assert warnings == [], (
        f"a child that IS a job member was reported as escaped: {warnings!r}"
    )


@pytest.mark.real
@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Job Objects are a Windows kernel primitive",
)
def test_real_job_membership_query_answers_true_for_an_assigned_child():
    """The membership check is a live Win32 IsProcessInJob call: a seam fake
    cannot tell whether the handle and argtypes are right. Spawn a REAL
    child, assign it to a REAL Job Object, and assert the query answers True
    -- not None (the fail-open "unanswerable" value), which is what a
    botched call would produce.

    Seeded defect: pass proc.pid instead of the process HANDLE to
    IsProcessInJob -> the call fails -> the helper returns None -> red.
    """
    from yamabiko.lifecycle.jobs import create_outer_job

    job = create_outer_job("yamabiko-test-membership")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        assert job.assign(proc) is True, "could not assign the child to a job"
        assert watchdog._child_in_job(proc) is True, (
            "IsProcessInJob did not see a child that was just assigned to a "
            "Job Object; the per-spawn cascade guard is inert"
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)
        job.close()


# --- err=5 spawn noise: the per-child assign is the FALLBACK, not the path --


class _FakeOuterJob:
    """create_outer_job stand-in. Records assign() arguments; the
    self-assign result is configurable so both the inherit path and the
    failed-self-assign fallback can be driven. Never joins the pytest
    process to a real KILL_ON_JOB_CLOSE job."""

    def __init__(self, self_assign_ok: bool = True) -> None:
        self.assigned: list = []
        self._self_assign_ok = self_assign_ok

    def assign(self, proc_or_pid) -> bool:
        self.assigned.append(proc_or_pid)
        return self._self_assign_ok

    def close(self) -> None:
        pass


def _build_watchdog_with_fake_job(monkeypatch, *, self_assign_ok: bool):
    """build_watchdog() with its process-global side effects stubbed (the
    single-instance mutex, the rotating file log, crash-evidence arming, and
    the real outer Job Object). Returns (app, fake_job); the caller closes
    app.pipeline_writer."""
    from yamabiko import log_config

    monkeypatch.setattr(
        watchdog, "_acquire_single_instance_mutex", lambda: object(),
    )
    monkeypatch.setattr(log_config, "setup_main_log", lambda *a, **kw: None)
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    job = _FakeOuterJob(self_assign_ok=self_assign_ok)
    monkeypatch.setattr(watchdog, "create_outer_job", lambda name: job)
    return watchdog.build_watchdog([]), job


def test_build_watchdog_propagates_semantic_config_failure(monkeypatch):
    import yamabiko.config as config_mod

    def fail_semantics(*_args, **_kwargs):
        raise ConfigValidationError([
            "translation.temperature must be finite; got nan"
        ])

    monkeypatch.setattr(config_mod, "load_full_config", fail_semantics)
    app = None
    try:
        with pytest.raises(ConfigValidationError, match="temperature"):
            app, _job = _build_watchdog_with_fake_job(
                monkeypatch,
                self_assign_ok=True,
            )
    finally:
        if app is not None:
            app.pipeline_writer.close()


def test_build_watchdog_skips_the_redundant_per_child_job_assign(monkeypatch):
    """With the watchdog self-assigned to the outer job, children inherit
    membership: re-assigning them fails ERROR_ACCESS_DENIED, which logged an
    ``AssignProcessToJobObject failed: err=5`` WARNING on every pipeline and
    llama start -- a line that reads exactly like a broken kill cascade. No
    supervisor may carry a job to re-assign on the inherit path.

    Seeded defect: pass ``job=job`` to the supervisors again (instead of the
    self-assign-conditional child_job) -> each Watchdog carries the job and
    re-assigns every spawn -> the err=5 noise returns -> red.
    """
    app, job = _build_watchdog_with_fake_job(monkeypatch, self_assign_ok=True)
    try:
        assert job.assigned == [None], (
            f"the outer job must be assigned the watchdog process (None) and "
            f"nothing else; got {job.assigned!r}"
        )
        carried = [w.name for w in app.watchdogs if w._job is not None]
        assert carried == [], (
            f"supervisors {carried} still carry the outer job after a "
            f"successful self-assign; every spawn will re-assign an existing "
            f"member and log a spurious err=5 warning"
        )
    finally:
        app.pipeline_writer.close()


def test_build_watchdog_keeps_per_child_job_assign_when_self_assign_fails(
    monkeypatch,
):
    """The suppression is conditional on the self-assign SUCCEEDING. When it
    fails, children inherit nothing, so the per-child assign is the only
    thing keeping the kill cascade alive and must still be wired.

    Seeded defect: hardcode ``child_job = None`` -> no supervisor carries the
    job -> children escape KILL_ON_JOB_CLOSE with no fallback -> red.
    """
    app, job = _build_watchdog_with_fake_job(monkeypatch, self_assign_ok=False)
    try:
        assert app.watchdogs, "expected at least one supervisor"
        orphaned = [w.name for w in app.watchdogs if w._job is not job]
        assert orphaned == [], (
            f"supervisors {orphaned} carry no job after a FAILED self-assign; "
            f"their children inherit no membership and would survive watchdog "
            f"death (the zombie-llama class)"
        )
    finally:
        app.pipeline_writer.close()
