"""Shared pytest fixtures.

Also hosts session-level safeguards against process leaks. See
``_kill_session_zombies_fixture`` below. The root-cause fix for child
process leaks lives in ``yamabiko/watchdog.py`` (Windows Job Object with
KILL_ON_JOB_CLOSE); this fixture is the belt-and-suspenders layer for tests
that bypass the watchdog entirely or whose subprocesses escape through buggy
mutations.

The session teardown shares its match set with
``yamabiko.watchdog._sweep_orphaned_children`` via the imported
``YAMABIKO_CHILD_MATCH_SPECS`` constant, so the conftest kills the same
broadened set the startup sweep does -- overlay and pipeline orphans as well
as llama-server. The conftest does its own enumeration + baseline subtraction
rather than calling the production sweep directly, because the production
sweep has no concept of "baseline" and would happily kill a pre-existing
yamabiko instance the developer had running before pytest started.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Single source of truth for the match set. Importing the constant
# (rather than duplicating it) guarantees the conftest baseline and
# the production sweep agree on which processes count as "ours".
from yamabiko.watchdog import YAMABIKO_CHILD_MATCH_SPECS  # noqa: E402


# ---------------------------------------------------------------------------
# Session-level zombie sweep
# ---------------------------------------------------------------------------

# Only snapshot processes whose executable path sits inside this
# project tree. Prevents the baseline from including unrelated
# pythons (e.g. the user's global Python REPL) and from pre-existing
# llamas elsewhere on the box.
_PROJECT_PATH_MARKER = str(ROOT).replace("\\", "/").lower()


def _snapshot_child_pids() -> set[int]:
    """Return PIDs of currently-running processes that the production
    sweep ``_sweep_orphaned_children`` would consider a match.

    Mirrors the sweep's match set (imported as
    ``YAMABIKO_CHILD_MATCH_SPECS``) so the pre-run baseline correctly
    represents pre-existing children we must not kill later. Empty
    set on non-Windows or on enumeration error.
    """
    if not sys.platform.startswith("win"):
        return set()
    pids: set[int] = set()
    image_names = {image for image, _ in YAMABIKO_CHILD_MATCH_SPECS}
    # One PowerShell call per image. CommandLine is requested so the
    # needle filter can distinguish our python children from unrelated
    # python processes (e.g. a pytest runner itself, or an LSP).
    for image_name in image_names:
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    f"Get-CimInstance Win32_Process "
                    f"-Filter \"Name='{image_name}'\" | "
                    f"Select-Object ProcessId, ExecutablePath, CommandLine | "
                    f"ForEach-Object {{ "
                    f"\"$($_.ProcessId)|$($_.ExecutablePath)|$($_.CommandLine)\" "
                    f"}}"
                ],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.count("|") < 2:
                continue
            pid_s, rest = line.split("|", 1)
            exe, cmdline = rest.split("|", 1)
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            if _PROJECT_PATH_MARKER not in exe.replace("\\", "/").lower():
                continue
            # Match any needle registered for this image; None means
            # "image-name match alone is enough".
            for spec_image, needle in YAMABIKO_CHILD_MATCH_SPECS:
                if spec_image != image_name:
                    continue
                if needle is None or needle in cmdline:
                    pids.add(pid)
                    break
    return pids


@pytest.fixture(scope="session", autouse=True)
def _kill_session_zombies_fixture():
    """Kill any yamabiko child process this session spawned that the
    watchdog's Job Object didn't already reap.

    At session start: snapshot PIDs of project-scoped yamabiko children
    (llama-server, pipeline, overlay, frozen exe re-entry children).
    These are *pre-existing* -- user may have yamabiko running, or
    leftover zombies from a previous stalled run. We do NOT kill
    these at start (would break the user's real yamabiko if running).

    At session end: snapshot again using the same matcher, compute
    the delta, kill only the NEW PIDs. Those are guaranteed to be
    ours (tests don't share PIDs with the developer's pre-existing
    run) and shouldn't have survived.

    Scope: session-wide autouse. Any test suite run picks this up;
    no opt-in required. Safe no-op on non-Windows.
    """
    baseline = _snapshot_child_pids()
    yield
    if not sys.platform.startswith("win"):
        return
    after = _snapshot_child_pids()
    new_pids = sorted(after - baseline)
    if not new_pids:
        return
    # Tests are DONE by the time this fires; killing here is strictly
    # cleanup. Print the PID list first so the post-run logs show
    # exactly what the sweep targeted.
    print(
        f"\n[conftest] killing {len(new_pids)} zombie child process(es) "
        f"that survived the test session: {new_pids}"
    )
    # Kill only the delta (post-session PIDs not in the baseline).
    # We deliberately do NOT call
    # ``yamabiko.watchdog._sweep_orphaned_children`` here, even though
    # that helper uses the same matcher logic: the production sweep
    # has no concept of "baseline", so invoking it would kill a
    # pre-existing yamabiko instance the developer had running before
    # pytest started. The session-autouse contract is "only kill what
    # this session spawned," so we must honour the baseline subtraction.
    for pid in new_pids:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-test app-data isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_yamabiko_appdata(request, monkeypatch, tmp_path):
    """Point YAMABIKO_APPDATA at a per-test tmp dir for every non-``real``
    test.

    Failure class prevented: any test that (directly or transitively)
    reaches ``paths.app_data_dir()`` without setting YAMABIKO_APPDATA
    falls through to the developer's real ``%LOCALAPPDATA%\\Yamabiko`` --
    silently reading the dev machine's config/state and littering it with
    test artifacts (three tray tests shipped exactly this in PR #74).
    ``paths.app_data_dir()`` reads ``os.environ`` at call time (no
    import-time caching), so an env fixture is a reliable seam.

    Why this composes with existing tests:

    * Per-test ``monkeypatch.setenv``/``delenv`` wins: same-scope autouse
      fixtures run BEFORE the fixtures/bodies a test declares, so the
      ~100 tests that pin YAMABIKO_APPDATA themselves (or delenv it, as
      test_paths.py does) overwrite this default for their duration, and
      monkeypatch's undo stack restores the pre-test environment.
    * The target is ``tmp_path / "Yamabiko"`` -- NOT a bare ``tmp_path``
      -- because several suites isolate via ``LOCALAPPDATA=str(tmp_path)``
      and assert on ``tmp_path / "Yamabiko" / ...`` paths
      (tests/test_main_entry.py, tests/test_build_pipeline.py).
      YAMABIKO_APPDATA takes precedence inside ``app_data_dir()``, so it
      must resolve to the identical directory those assertions expect.
    * ``@pytest.mark.real`` tests are exempt: the frozen-E2E tier spawns
      the installed exe with the INHERITED environment and tails the real
      ``%LOCALAPPDATA%\\Yamabiko\\logs`` (test_real_frozen_e2e.py); an
      ambient override would redirect the exe's app data and time out
      every log-marker wait. Real-tier tests manage env explicitly.
    """
    if request.node.get_closest_marker("real") is not None:
        return
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))


@pytest.fixture(autouse=True)
def _no_key_press_recorded_for_the_focus_ring():
    """Isolation guard, not a behaviour claim: the keyboard focus ring's
    record of the latest press (``segmented._last_input``, one application
    event filter for the whole session) starts every test as in a fresh
    process, with no key press, so a ring assertion never depends on what
    the test before it pressed. Covers every user of KeyboardRingMixin (the
    overlay, setup and the error window, in this directory and below). Only
    touches it once a test has built a widget that installs it."""
    segmented = sys.modules.get("yamabiko.overlay_app.segmented")
    if segmented is not None and segmented._last_input is not None:
        segmented._last_input.key = False


@pytest.fixture
def crash_evidence_guard():
    """Snapshot/restore the process-global state install_crash_evidence
    mutates (sys/threading excepthooks, faulthandler, the module-global
    handle + role registries), so tests that intentionally install the
    real crash-evidence hooks can't leak them into the rest of the
    session.

    faulthandler handling is two-sided:

    * Setup calls ``faulthandler.disable()`` so every guard-using test
      starts from a known-off state. Without this, pytest's builtin
      faulthandler plugin (or a leak from an earlier test) pre-enables
      faulthandler and an ``is_enabled()`` assert passes even when
      ``install_crash_evidence`` never calls ``enable`` --
      order-dependent Liar (charter §3.1).
    * Teardown restores the PRIOR enabled/disabled state rather than
      leaving faulthandler pointing at a closed tmp_path handle (or
      unconditionally off, which would strip pytest's own native-crash
      protection for the rest of the session).

    Shared by test_log_config.py and test_main_entry.py (previously two
    inline copies); any new test that installs real hooks must use this
    fixture instead of cloning another copy.
    """
    import faulthandler
    import threading

    from yamabiko import log_config

    prev_sys_hook = sys.excepthook
    prev_thread_hook = threading.excepthook
    prev_roles = set(log_config._INSTALLED_ROLES)
    prev_handles = set(log_config._CRASH_LOG_HANDLES)
    faulthandler_was_enabled = faulthandler.is_enabled()
    log_config._INSTALLED_ROLES.clear()
    faulthandler.disable()
    yield
    # Detach faulthandler from any test-installed tmp_path handle BEFORE
    # closing it (enable-then-close would leave a dangling target fd).
    faulthandler.disable()
    for role in list(log_config._CRASH_LOG_HANDLES):
        if role in prev_handles:
            continue
        try:
            log_config._CRASH_LOG_HANDLES.pop(role).close()
        except Exception:
            pass
    log_config._INSTALLED_ROLES.clear()
    log_config._INSTALLED_ROLES.update(prev_roles)
    sys.excepthook = prev_sys_hook
    threading.excepthook = prev_thread_hook
    if faulthandler_was_enabled:
        # Restore against the ORIGINAL process stderr: the pre-test
        # target fd (pytest's dup'd stderr) isn't recoverable via the
        # public API, and at teardown time sys.stderr is pytest's
        # capture object, which has no usable fileno() and would make
        # a bare enable() raise. sys.__stderr__ keeps the session's
        # native-crash evidence armed.
        try:
            if sys.__stderr__ is not None:
                faulthandler.enable(file=sys.__stderr__)
            else:
                faulthandler.enable()
        except Exception:
            pass


@pytest.fixture
def hermetic_log_handlers():
    """Factory for a save/clear/restore guard around logger handler state.

    Usage::

        def test_x(hermetic_log_handlers, ...):
            with hermetic_log_handlers("", "yamabiko"):
                ...  # attach handlers freely

    Why: tests that exercise the REAL ``log_config.setup_main_log`` attach
    RotatingFileHandlers to process-global loggers. Without isolation,
    (a) a handler leaked from an earlier test dedup-skips the attach under
    test (order-dependent false red), and (b) the handler attached during
    the test holds a file open inside ``tmp_path``, which Windows then can't
    delete. The guard snapshots handlers + propagate + level, clears them for
    the block, and on exit closes only the handlers that were ADDED inside the
    block before restoring the originals.

    This pattern previously existed as four inline copies
    (test_build_overlay / test_build_watchdog / test_main_entry /
    test_log_config); new role-log tests must use this fixture instead of
    cloning another copy.
    """
    import logging
    from contextlib import contextmanager

    @contextmanager
    def _hermetic(*names: str):
        loggers = [logging.getLogger(n) for n in names]
        saved = [
            (lg, list(lg.handlers), lg.propagate, lg.level)
            for lg in loggers
        ]
        for lg in loggers:
            for h in list(lg.handlers):
                lg.removeHandler(h)
            lg.setLevel(logging.NOTSET)
        try:
            yield
        finally:
            for lg, original, propagate, level in saved:
                added = [h for h in lg.handlers if h not in original]
                for h in list(lg.handlers):
                    lg.removeHandler(h)
                for h in added:
                    try:
                        h.close()
                    except Exception:
                        pass
                for h in original:
                    lg.addHandler(h)
                lg.propagate = propagate
                lg.setLevel(level)

    return _hermetic
