"""Pipeline-as-supervisor-of-overlay.

Pipeline owns an inner Job Object and spawns overlay. Graceful
shutdown uses a named event scoped to the pipeline PID; on
pipeline death the inner job's KILL_ON_JOB_CLOSE semantics
kernel-enforce overlay teardown.

Architecture context: the watchdog's outer Job Object already
contains the pipeline process, so when pipeline creates+assigns-
itself to an inner job, the Windows kernel automatically forms a
nest. Overlay is then bound to the inner job, which means:
 - Outer job closes (watchdog dies) -> every descendant dies.
 - Inner job closes (pipeline dies) -> overlay dies, outer job
   unaffected (watchdog survives to respawn pipeline).

Spawn ordering is load-bearing: the inner job must exist before the
overlay is spawned, so the overlay is born already inside it. The
pipeline itself is deliberately NOT a member of the inner job (see
main.py's inner-job setup): self-membership meant the pipeline's own
death semantics were tied to its child's job -- the B1 self-kill
blocker the lifecycle overhaul removed. Do not "restore" a pipeline
self-assign here.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Sequence

from yamabiko.lifecycle.exit_codes import ExitPolicy, classify
from yamabiko.lifecycle.jobs import JobObject
from yamabiko.lifecycle.shutdown_event import ShutdownEventWriter

_log = logging.getLogger(__name__)

# Anti-flap defaults. Shorter window than Watchdog's because
# overlay is lighter and more recoverable, and we want fast give-up if
# a user is fighting Task-Manager-kill loops.
_DEFAULT_MAX_RESTARTS = 3
_DEFAULT_RESTART_WINDOW_S = 60.0
_DEFAULT_BACKOFF_S: Sequence[float] = (1.0, 3.0, 10.0)


class PipelineSupervisor:
    """Minimal supervisor for overlay (pipeline's only child).

    Spawn the overlay, assign to inner job (kernel-enforced cascade on
    pipeline death), signal via named event on shutdown. If overlay
    crashes during normal operation, we restart it with fresh job
    membership -- bounded by the anti-flap budget (max_restarts per
    restart_window_s) so a user Task-Manager-killing the overlay
    repeatedly can't burn the pipeline in a respawn loop, and a
    transient spawn exception retries with backoff rather than burning
    the session on the first failure.
    """

    def __init__(
        self,
        *,
        spawn: Callable[[], subprocess.Popen],
        inner_job: JobObject,
        shutdown_writer: ShutdownEventWriter,
        stop_event: threading.Event,
        grace_s: float = 3.0,
        max_restarts: int = _DEFAULT_MAX_RESTARTS,
        restart_window_s: float = _DEFAULT_RESTART_WINDOW_S,
        backoff_s: Sequence[float] = _DEFAULT_BACKOFF_S,
        clock: Callable[[], float] = time.monotonic,
        spawn_in_job: Optional[Callable[[JobObject], subprocess.Popen]] = None,
    ) -> None:
        self._spawn = spawn
        # Optional ATOMIC spawn path. When provided, spawn_once
        # routes through this callable (typically a partial of
        # lifecycle.jobs.spawn_in_job over the overlay cmd), which
        # assigns the child to the inner job while it is still
        # CREATE_SUSPENDED -- closing the window in which a pipeline
        # death (taskkill /F mid-spawn) leaves a running overlay that
        # is not yet a job member and so survives KILL_ON_JOB_CLOSE.
        self._spawn_in_job = spawn_in_job
        self._inner_job = inner_job
        self._writer = shutdown_writer
        self._stop_event = stop_event
        self._grace_s = grace_s
        self._max_restarts = max_restarts
        self._restart_window_s = restart_window_s
        self._backoff_s = tuple(backoff_s) or (0.0,)
        self._clock = clock
        self._restart_times: Deque[float] = deque()
        self._proc: Optional[subprocess.Popen] = None
        self._stopped = False
        self._lock = threading.Lock()
        # Observable give-up state. Set True ONLY on run()'s two
        # supervision-failure paths -- the anti-flap cap branch (overlay
        # crash-looped past the budget) and the defensive crash handler
        # (an unexpected exception in the supervisor loop itself) -- in
        # both cases BEFORE tripping stop_event. A normal overlay exit
        # and a user quit both leave it False. run_pipeline reads it
        # after its wait to tell a definitive supervision failure (arm
        # EXIT_CHILD_GIVEUP -> watchdog RESTART with a fresh overlay
        # budget) apart from a clean quit (exit 0). Without this
        # discriminator both set the same overlay_stop event and a
        # failure masqueraded as a clean quit.
        self.gave_up = False

    def spawn_once(self) -> subprocess.Popen:
        """Spawn overlay, assign to inner job, return the Popen.

        Assignment-failure is logged but not fatal: in that case the
        named-event signal (stop()) still provides a graceful path, and
        the watchdog's outer job still catches everything on its own
        close. The kernel-enforced cascade is belt; this is the
        suspenders.
        """
        if self._spawn_in_job is not None:
            # Atomic path -- the child is assigned to the inner
            # job before its first instruction runs (CREATE_SUSPENDED
            # -> assign -> resume inside the callable). Do NOT assign
            # again here: re-assigning an existing member fails with
            # ERROR_ACCESS_DENIED and would log a spurious warning on
            # every spawn.
            proc = self._spawn_in_job(self._inner_job)
        else:
            proc = self._spawn()
            if not self._inner_job.assign(proc):
                _log.warning(
                    "overlay pid %d not assigned to inner job; pipeline-death "
                    "cascade will rely on SetEvent fallback",
                    proc.pid,
                )
        # Publish _proc inside the lock so stop()'s read can't race the
        # write (a torn read returned the previous proc and orphaned this one).
        with self._lock:
            self._proc = proc
        return proc

    def stop(self) -> None:
        """Signal graceful shutdown to overlay; fall back to terminate
        after grace_s; kill after an additional 2s.

        Idempotent: safe to call multiple times (main thread + supervisor
        thread may both call on normal shutdown).
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            # Read _proc in the SAME critical section as the _stopped flip
            # so we always shut down whatever proc was current at stop time.
            proc = self._proc
        self._shutdown_proc(proc)

    def _shutdown_proc(self, proc) -> None:
        """Graceful named-event signal, then the terminate -> kill ladder.

        Shared by stop() and the post-spawn race guard in run(), so a
        freshly-spawned overlay that a concurrent stop() missed still gets torn
        down rather than orphaned.
        """
        self._writer.signal()
        if proc is None:
            return
        try:
            proc.wait(timeout=self._grace_s)
            return
        except subprocess.TimeoutExpired:
            _log.warning(
                "overlay did not exit within %.1fs; terminating",
                self._grace_s,
            )
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _log.warning("overlay did not exit within 2s of terminate; killing")
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception as exc:
                _log.error("overlay kill failed: %s", exc)

    def _evict_old_restarts(self) -> None:
        """Drop restart timestamps outside the rolling window."""
        now = self._clock()
        while (
            self._restart_times
            and (now - self._restart_times[0]) > self._restart_window_s
        ):
            self._restart_times.popleft()

    def _backoff_for_attempt(self, attempt_idx: int) -> float:
        """Pick the backoff-sequence entry for ``attempt_idx``.

        Clamps to the last entry so a long crash loop keeps waiting at
        the longest backoff rather than tripping IndexError.
        """
        if not self._backoff_s:
            return 0.0
        idx = min(attempt_idx, len(self._backoff_s) - 1)
        return self._backoff_s[idx]

    def run(self) -> None:
        """Supervise overlay: respawn on crash/kill with bounded anti-flap.

        Any exit of the overlay process -- natural crash, external kill
        (e.g. Task Manager), OR a _spawn() exception -- records a
        restart timestamp. Once ``max_restarts`` timestamps fall inside
        ``restart_window_s``, the supervisor gives up: log, set
        stop_event, return. External observers (pipeline, watchdog)
        see stop_event and tear down cleanly instead of the supervisor
        dying as a silent daemon-thread.

        Between restarts the supervisor sleeps ``backoff_s[min(i,
        len-1)]`` seconds, interruptible by stop_event.
        """
        try:
            while not self._stop_event.is_set():
                self._evict_old_restarts()
                # The cap budgets RE-spawns, never the initial start.
                # ``_restart_times`` gains one entry per overlay exit
                # (crash / external kill / spawn exception), so after N
                # crashes it holds N timestamps. Strict ``>`` gives the
                # INITIAL failure a free slot: max_restarts=N then permits
                # exactly N respawns before give-up. (With ``>=`` the
                # initial failure was counted as a respawn, so N allowed
                # only N-1 -- the M-01 off-by-one.) ``_restart_times`` is
                # empty until the first exit and ``[] and ...``
                # short-circuits, so the initial spawn always proceeds
                # even with max_restarts=0. Mirrors Watchdog.run, which
                # spawns first and only consults the cap after a crash.
                if self._restart_times and len(self._restart_times) > self._max_restarts:
                    self.gave_up = True
                    _log.error(
                        "overlay respawn cap reached: %d respawns in %.0fs; "
                        "giving up and signalling pipeline shutdown",
                        self._max_restarts, self._restart_window_s,
                    )
                    # Contract on give-up: we do NOT call self.stop() or
                    # terminate the current ``self._proc`` here. The
                    # last-surviving overlay (if any) is either already
                    # dead (we only reach the cap after it exited, which
                    # is what appended the latest restart timestamp) or
                    # will be cleaned up by the pipeline's outer finally
                    # block as it unwinds in response to stop_event. The
                    # happy-path stop() (hit when stop_event fires during
                    # proc.wait) handles writer.signal() + graceful
                    # terminate; the give-up path trusts pipeline-level
                    # cleanup because there is no live overlay proc to
                    # send a graceful shutdown to.
                    self._stop_event.set()
                    return
                # Backoff: skip on the very first spawn, apply on subsequent.
                # ``len(self._restart_times) - 1`` is intentional: after
                # restart #1, len == 1 and we want backoff_s[0] (the first
                # entry in the sequence) -- that's the delay BEFORE the
                # second spawn attempt. After restart #2, len == 2 and
                # idx 1 picks the second entry. The cap check above fires
                # before idx would exceed the backoff length.
                if self._restart_times:
                    delay = self._backoff_for_attempt(len(self._restart_times) - 1)
                    if delay > 0 and self._stop_event.wait(timeout=delay):
                        return  # stop requested during backoff
                # Spawn: a spawn exception counts as a restart, not as a
                # session-burning supervisor crash.
                try:
                    proc = self.spawn_once()
                except Exception as exc:
                    _log.warning(
                        "overlay spawn failed (%s); counting as restart", exc,
                    )
                    self._restart_times.append(self._clock())
                    continue
                # A stop() that landed DURING spawn_once read the previous
                # _proc (or None) and won't touch this fresh one. If a stop is
                # now pending, tear this proc down directly so it isn't orphaned.
                if self._stop_event.is_set():
                    self._shutdown_proc(proc)
                    return
                # Wait for exit or stop.
                proc_exited = False
                while not self._stop_event.is_set():
                    try:
                        rc = proc.wait(timeout=1.0)
                        if classify(rc) is ExitPolicy.RESTART:
                            _log.info("overlay exited with code %d; respawning", rc)
                            self._restart_times.append(self._clock())
                            proc_exited = True
                            break
                        # Intentional exit -- a clean tray-Quit (rc==0 ->
                        # STOP_SELF) or a fatal stop-all (42/43 -> STOP_ALL).
                        # Do NOT respawn: that flashes a ghost overlay back
                        # on screen during the pipeline's shutdown drain
                        # Signal pipeline teardown and stop; the
                        # pipeline's finally performs the actual cleanup.
                        _log.info(
                            "overlay exited with code %d (intentional); "
                            "signalling pipeline shutdown, not respawning", rc,
                        )
                        self._stop_event.set()
                        return
                    except subprocess.TimeoutExpired:
                        continue
                if not proc_exited:
                    # stop_event fired while waiting on proc; normal shutdown.
                    self.stop()
                    return
            # Outer while exited because stop_event was already set on entry.
            self.stop()
        except Exception:
            _log.exception(
                "overlay supervisor thread crashed; overlay will not respawn"
            )
            # Supervision definitively failed -- a supervisor-code bug,
            # not a cap exhaustion (the log lines distinguish the two:
            # this one carries a traceback, the cap branch says "respawn
            # cap reached"). Mark gave_up BEFORE stop_event, same
            # ordering as the cap branch, so run_pipeline's post-wait
            # read arms EXIT_CHILD_GIVEUP (47, watchdog RESTART) -- a
            # crashed supervisor must not masquerade as a clean quit
            # (exit 0 = STOP_SELF = silent full cascade).
            self.gave_up = True
            # Trip stop_event so external observers can see the crash.
            self._stop_event.set()
