"""Entry-point tests for yamabiko.main.main().

The pipeline entry function -- which parses CLI args, attaches file
logging, gates on wizard state, and dispatches to the asyncio event
loop -- is exercised here end-to-end with the asyncio pipeline coroutine
short-circuited, so the composition layer (argparse schema, log wiring,
wizard gate, asyncio dispatch) is covered without running the actual
pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yamabiko import log_config


class _MainHardExit(BaseException):
    """Sentinel standing in for os._exit so a test can observe main()'s
    hard-exit code without terminating the pytest process. main() now hard-exits
    UNCONDITIONALLY at the end (issue #61), so every full main() dispatch ends
    here."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _patch_main_dependencies(monkeypatch, tmp_path: Path, *,
                              needs_wizard: bool = False,
                              on_wizard_run=None):
    """Shared setup: route LOCALAPPDATA to tmp, skip the wizard gate,
    short-circuit the asyncio pipeline, and stub os._exit (main() hard-exits
    unconditionally now -- issue #61 -- so it never returns normally).

    Returns a dict with signal flags so tests can assert on observable
    behaviour (wizard_run, amain_ran, hard_exit_code) rather than on mock
    call counts. ``on_wizard_run`` (needs_wizard=True only) is invoked
    inside the run_wizard stub so ordering tests can snapshot state at
    wizard time.

    main() attaches a RotatingFileHandler to the ROOT logger and sets its
    level; ``logging.shutdown()`` in _hard_exit closes that handler but
    leaves it attached. Every test that drives main() past log setup must
    therefore run it inside ``hermetic_log_handlers("", "yamabiko")``, or
    the leaked root handler dedup-skips every later setup_main_log attach
    in the session (tests/test_log_config.py goes red when run after this
    file).
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    # Import after env setup so module-level paths read our tmp dir.
    import yamabiko.main as _main_mod
    import yamabiko.wizard_state as _ws

    observed = {"wizard_run": False, "amain_ran": False}

    # main() now ends in _hard_exit -> os._exit. Stub it so the runner
    # survives and the delivered code is observable.
    def _fake_os_exit(code):
        observed["hard_exit_code"] = code
        raise _MainHardExit(code)
    monkeypatch.setattr(_main_mod.os, "_exit", _fake_os_exit)

    # Crash-evidence arming is process-global (sys/threading excepthook
    # chains + a process-lifetime faulthandler handle parked in module
    # globals): main()-driving tests must NOT run the real thing, or the
    # session keeps the swapped hooks + a tmp_path crash handle forever
    # after this test's tmp dir is gone. Stub it here (same pattern as
    # tests/test_build_watchdog.py's _watchdog_env). The helper's own
    # behavior is covered in tests/test_log_config.py; the main() call-
    # site wiring by test_main_installs_crash_evidence_for_pipeline_role
    # below, which re-patches the REAL helper under crash_evidence_guard.
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )

    # Wizard gate: default is "not needed." Tests can flip this.
    def _needs_wizard_stub():
        return needs_wizard
    monkeypatch.setattr(_ws, "needs_wizard", _needs_wizard_stub)

    # If needs_wizard is True, the real run_wizard would build a QApplication
    # and block. Replace with a no-op that just flips needs_wizard off via
    # a follow-up stub, so main()'s second needs_wizard() check passes.
    if needs_wizard:
        def _run_wizard_stub():
            observed["wizard_run"] = True
            if on_wizard_run is not None:
                on_wizard_run()
            # After run_wizard returns, main() calls needs_wizard() AGAIN
            # to decide wizard-cancel (exit 42) vs continue. Flip to False.
            monkeypatch.setattr(_ws, "needs_wizard", lambda: False)
        # run_wizard is imported lazily inside main(); patch at source.
        import yamabiko.wizard as _wiz
        monkeypatch.setattr(_wiz, "run_wizard", _run_wizard_stub)

    # The BIG short-circuit: replace amain with a fast no-op coroutine.
    # asyncio.run(no_op_coro) returns instantly; pipeline never starts.
    # Signature: amain(args, app_box=None) -- app_box lets main() read
    # stop_all_requested if asyncio.run's cleanup raises after amain.
    async def _fake_amain(args, app_box=None):
        observed["amain_ran"] = True
        return 0  # amain always returns an int exit code (0 = clean stop)
    monkeypatch.setattr(_main_mod, "amain", _fake_amain)

    return observed, _main_mod


def test_main_parses_args_and_dispatches_to_amain_when_wizard_complete(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """Happy path: wizard already done, argparse succeeds, amain runs.

    This is the path a returning user takes on every app launch. If
    any of these steps regresses -- argparse schema breaks because a
    required argument lacks a default, log_config.setup_main_log is
    renamed, wizard_state.needs_wizard is renamed, asyncio.run is
    broken by a dependency upgrade -- this test goes RED.

    Seeded-defect check #1: rename ``amain`` in yamabiko/main.py to
    anything else -> monkeypatch.setattr raises AttributeError, test
    fails at setup.
    Seeded-defect check #2: add ``parser.add_argument("--required",
    required=True)`` to the argparse schema -> argparse exits nonzero
    with "the following arguments are required", SystemExit is raised,
    test fails.
    """
    observed, _main_mod = _patch_main_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["yamabiko", "--max-segments", "0"])

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(_MainHardExit):
            _main_mod.main()

    assert observed["amain_ran"], (
        "main() completed without dispatching amain -- the argparse/"
        "wizard/asyncio.run chain broke somewhere"
    )
    assert observed["hard_exit_code"] == 0, (
        "a clean amain(0) must hard-exit 0 (issue #61)"
    )
    assert not observed["wizard_run"], (
        "wizard ran when needs_wizard stub returned False"
    )


def test_main_runs_wizard_when_state_missing(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """First-run path: wizard_state.needs_wizard() returns True,
    main() invokes run_wizard, then proceeds to amain.

    Seeded-defect check: remove the ``if wizard_state.needs_wizard():
    run_wizard()`` block from main() -> wizard_run stays False even
    when needs_wizard=True, test fails.
    """
    observed, _main_mod = _patch_main_dependencies(
        monkeypatch, tmp_path, needs_wizard=True,
    )
    monkeypatch.setattr(sys, "argv", ["yamabiko"])

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(_MainHardExit):
            _main_mod.main()

    assert observed["wizard_run"], "main() did not invoke run_wizard"
    assert observed["amain_ran"], (
        "wizard ran but amain was not reached -- the post-wizard "
        "needs_wizard() check is gating the pipeline incorrectly"
    )


def test_main_with_config_and_setup_incomplete_exits_43_with_setup_required(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """A --config run on an install whose first-run setup never finished
    must not run the wizard (its picks would land in user_overrides.yaml,
    the layer --config skips) and must not start the pipeline: it shows
    "Setup required" and exits 43 so the watchdog stops.

    Seeded defect: drop the ``args.config`` check -> the wizard runs and
    the pipeline starts -> red.
    """
    observed, _main_mod = _patch_main_dependencies(
        monkeypatch, tmp_path, needs_wizard=True,
    )
    import yamabiko.ui.error_dialog as _ed

    shown: list = []
    monkeypatch.setattr(_ed, "show_startup_error", shown.append)
    explicit = tmp_path / "repro.yaml"
    explicit.write_text("genre: anime\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["yamabiko", "--config", str(explicit)])

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(SystemExit) as exc_info:
            _main_mod.main()

    assert exc_info.value.code == 43
    assert [err.title for err in shown] == ["Setup required"]
    assert "without --config" in shown[0].message
    assert not observed["wizard_run"], "the wizard ran under --config"
    assert not observed["amain_ran"], "the pipeline started before setup"


def test_main_exits_with_sentinel_42_when_wizard_still_incomplete(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """Wizard-cancel path: user opens wizard, closes it without
    completing. needs_wizard() returns True BOTH before and after
    run_wizard(). main() must exit with code 42 -- the sentinel the
    watchdog keys on to distinguish wizard-cancel from a clean run.

    Seeded-defect check: change sys.exit(42) in main() to sys.exit(0)
    -> SystemExit code changes, this test fails.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    import yamabiko.main as _main_mod
    import yamabiko.wizard_state as _ws
    import yamabiko.wizard as _wiz

    # This test drives main() WITHOUT _patch_main_dependencies (it needs
    # the real wizard-cancel exit path), so it must stub the process-
    # global crash-evidence arming itself: the real thing parks a
    # tmp_path faulthandler handle + the 'pipeline' role in module
    # globals for the rest of the session (verified leak).
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    # needs_wizard() stays True on both invocations (cancel case).
    monkeypatch.setattr(_ws, "needs_wizard", lambda: True)
    monkeypatch.setattr(_wiz, "run_wizard", lambda: None)
    monkeypatch.setattr(sys, "argv", ["yamabiko"])

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(SystemExit) as exc_info:
            _main_mod.main()

    assert exc_info.value.code == 42, (
        f"expected wizard-cancel sentinel 42, got {exc_info.value.code}"
    )


def test_main_propagates_bad_arg_as_systemexit_nonzero(monkeypatch, tmp_path):
    """argparse handles bad args by calling sys.exit(2). main() must
    NOT swallow that -- the watchdog relies on the exit code to
    distinguish "wizard cancel (42)" from "tailored error (43)" from
    "bad invocation (2)".

    Seeded-defect check: wrap parser.parse_args() in
    ``try: ... except SystemExit: return`` -> this test fails because
    main() returns normally instead of raising SystemExit(2).
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    import yamabiko.main as _main_mod

    # Belt-and-suspenders: argparse rejects --bogus-flag BEFORE main()
    # reaches install_crash_evidence today, but if that ordering ever
    # changes this test must not become the file's hook-leak source.
    monkeypatch.setattr(
        log_config, "install_crash_evidence", lambda role: None,
    )
    monkeypatch.setattr(sys, "argv", ["yamabiko", "--bogus-flag"])

    with pytest.raises(SystemExit) as exc_info:
        _main_mod.main()

    assert exc_info.value.code != 0, (
        "argparse bad-arg rejection must produce a nonzero exit code"
    )


def test_main_logs_to_pipeline_role_file(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """main() — the --run-main pipeline/wizard child process — attaches
    file logging to main-pipeline.log, its own per-process file, never
    the shared main.log (Windows RotatingFileHandler rotation breaks when
    multiple processes hold the same file open).

    Fails red if: the main() call site drops the
    filename="main-pipeline.log" argument, or the startup line stops
    reaching the role file.
    """
    import logging as _logging

    observed, _main_mod = _patch_main_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["yamabiko", "--max-segments", "0"])

    with hermetic_log_handlers("", "yamabiko"):
        # main() hard-exits unconditionally now (issue #61); _hard_exit calls
        # logging.shutdown() (flush + close) before os._exit, so the startup
        # line is already on disk by the time the sentinel is raised.
        with pytest.raises(_MainHardExit):
            _main_mod.main()
        logs_dir = tmp_path / "Yamabiko" / "logs"
        role_log = logs_dir / "main-pipeline.log"
        assert role_log.is_file(), (
            f"main-pipeline.log missing after main(); logs dir: "
            f"{[p.name for p in logs_dir.iterdir()] if logs_dir.is_dir() else 'absent'}"
        )
        assert "yamabiko startup" in role_log.read_text(encoding="utf-8"), (
            "role log exists but main()'s 'yamabiko startup' line never "
            "reached it"
        )
        assert not (logs_dir / "main.log").exists(), (
            "main() wrote the shared main.log; cross-process rotation "
            "contention is back"
        )


def test_main_writes_identity_banner_to_role_log_before_wizard_runs(
    monkeypatch, tmp_path, hermetic_log_handlers,
):
    """The pipeline identity banner (role, version, frozen|dev, PID,
    python, OS) must be on disk BEFORE the first-run wizard executes: a
    beta user whose wizard download stalls sends us a log whose only
    content is wizard-phase records, and without the banner that log
    carries no version line at all. So main() emits the banner right
    after setup_main_log -- not in build_pipeline, which only runs after
    the wizard gate.

    Observable effect: at the moment run_wizard is invoked (snapshotted
    inside the wizard stub), main-pipeline.log already contains the
    ``yamabiko pipeline v<version>`` banner line.

    One-line seeded defect: move the log_startup_banner call from
    main() back below the wizard gate (or into build_pipeline) -- the
    wizard-time snapshot no longer contains the banner, red.
    """
    from yamabiko import __version__

    role_log = tmp_path / "Yamabiko" / "logs" / "main-pipeline.log"
    wizard_time_content: list[str] = []

    def _snapshot_role_log():
        wizard_time_content.append(
            role_log.read_text(encoding="utf-8")
            if role_log.is_file() else ""
        )

    observed, _main_mod = _patch_main_dependencies(
        monkeypatch, tmp_path, needs_wizard=True,
        on_wizard_run=_snapshot_role_log,
    )
    # No nvidia-smi shell-out from the banner's gpu line in the unit tier.
    monkeypatch.setattr(log_config, "_gpu_name_cache", "GPU-STUB")
    monkeypatch.setattr(sys, "argv", ["yamabiko"])

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(_MainHardExit):
            _main_mod.main()

    assert observed["wizard_run"], "precondition: the wizard stub never ran"
    assert wizard_time_content, "wizard stub took no role-log snapshot"
    assert f"yamabiko pipeline v{__version__}" in wizard_time_content[0], (
        f"identity banner not in main-pipeline.log at wizard time; a "
        f"wizard-phase failure log would carry no version line. Role log "
        f"content at wizard time: {wizard_time_content[0]!r}"
    )


# ---------------------------------------------------------------------------
# Crash-evidence bootstrap + asyncio exception routing (logging hardening)
# ---------------------------------------------------------------------------


# crash_evidence_guard (snapshot/restore of excepthooks + faulthandler +
# the module-global registries) lives in tests/conftest.py -- one shared
# fixture instead of per-file copies.


def test_main_installs_crash_evidence_for_pipeline_role(
    monkeypatch, tmp_path, hermetic_log_handlers, crash_evidence_guard,
):
    """main() must arm the crash-evidence bootstrap (faulthandler ->
    crash-pipeline.log + excepthooks) right after log setup and BEFORE
    the wizard/pipeline dispatch, so even a startup-path crash leaves
    evidence. Observable effects: crash-pipeline.log exists and
    sys.excepthook was replaced.

    One-line seeded defect: delete the
    ``log_config.install_crash_evidence("pipeline")`` call in main() ->
    crash-pipeline.log is never created, red.
    """
    # _patch_main_dependencies stubs install_crash_evidence for every
    # other test in this file; THIS test verifies main() wires the real
    # thing, so re-patch the real helper back in. crash_evidence_guard
    # restores the hooks/handles it installs.
    real_install = log_config.install_crash_evidence
    observed, _main_mod = _patch_main_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(log_config, "install_crash_evidence", real_install)
    monkeypatch.setattr(sys, "argv", ["yamabiko"])
    hook_before = sys.excepthook

    with hermetic_log_handlers("", "yamabiko"):
        with pytest.raises(_MainHardExit):
            _main_mod.main()

    crash_log = tmp_path / "Yamabiko" / "logs" / "crash-pipeline.log"
    assert crash_log.is_file(), (
        f"crash-pipeline.log missing after main(); logs dir: "
        f"{[p.name for p in crash_log.parent.iterdir()] if crash_log.parent.is_dir() else 'absent'}"
    )
    assert sys.excepthook is not hook_before, (
        "main() did not install the crash-evidence sys.excepthook"
    )


def test_amain_installs_asyncio_exception_handler_before_build(
    monkeypatch, tmp_path,
):
    """amain() must set the loop exception handler BEFORE build_pipeline
    runs, so loop-level failures during build (fire-and-forget task
    crashes) already route into the log.

    One-line seeded defect: delete the
    ``loop.set_exception_handler(_log_asyncio_exception)`` line in
    amain -> the observed handler stays None, red.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    import yamabiko.main as _main_mod

    seen: dict = {}

    async def _fake_build(args):
        seen["handler"] = asyncio.get_running_loop().get_exception_handler()
        return SimpleNamespace(stop_all_requested=False)

    async def _fake_run(app):
        return None

    monkeypatch.setattr(_main_mod, "build_pipeline", _fake_build)
    monkeypatch.setattr(_main_mod, "run_pipeline", _fake_run)

    code = asyncio.run(_main_mod.amain(_main_mod.parse_args([])))

    assert code == 0
    assert seen["handler"] is _main_mod._log_asyncio_exception, (
        f"loop exception handler at build time is {seen['handler']!r}; "
        f"expected main._log_asyncio_exception"
    )


def test_asyncio_exception_handler_logs_error_and_chains_to_default(
    caplog,
):
    """_log_asyncio_exception must write the loop failure (message +
    exc_info) at ERROR into the yamabiko log tree, then chain to the
    loop's default handler so dev-console stderr output is preserved.

    One-line seeded defect: drop the
    ``loop.default_exception_handler(context)`` chain call -> the
    recorded-default assert goes red. Drop the log.error call -> the
    ERROR-record assert goes red.
    """
    import yamabiko.main as _main_mod

    class _FakeLoop:
        def __init__(self):
            self.default_calls: list[dict] = []

        def default_exception_handler(self, context):
            self.default_calls.append(context)

    loop = _FakeLoop()
    exc = RuntimeError("task blew up mid-flight")
    ctx = {"message": "Task exception was never retrieved", "exception": exc}

    with caplog.at_level(logging.ERROR, logger="yamabiko"):
        _main_mod._log_asyncio_exception(loop, ctx)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        "asyncio" in r.getMessage()
        and "Task exception was never retrieved" in r.getMessage()
        for r in errors
    ), f"no asyncio ERROR record; got {[r.getMessage() for r in errors]!r}"
    assert "task blew up mid-flight" in caplog.text, (
        "exc_info (the actual exception) missing from the record"
    )
    assert loop.default_calls == [ctx], (
        "the default loop exception handler was not chained"
    )
