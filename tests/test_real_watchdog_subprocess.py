"""Real-path coverage for yamabiko.watchdog's subprocess seams
(_real_spawn and the llama-server spawn path's output capture).

Every test in ``tests/test_watchdog.py`` injects a fake ``spawn``
callable -- the actual Popen invocation (argv construction, stdin
wiring, kwargs) is uncovered. A regression there would ship silently.

We don't boot the full pipeline -- that'd require CUDA + audio +
llama-server. Instead, monkeypatch ``_build_main_cmd`` to return a
trivial sentinel invocation (``python -c "import sys; sys.exit(0)"``)
and call the real ``_real_spawn()``. That verifies the Popen seam
itself: can we actually spawn a child with the kwargs we pass it,
does it accept our stdin=DEVNULL choice, does it exit cleanly.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.real


def test_real_spawn_uses_popen_with_sentinel_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_real_spawn()`` actually calls ``subprocess.Popen`` with DEVNULL stdin.

    Substitute a one-liner Python command for the main pipeline argv;
    the goal is to exercise the real Popen seam, not the pipeline.
    """
    from yamabiko import watchdog

    sentinel_cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
    monkeypatch.setattr(watchdog, "_build_main_cmd", lambda: sentinel_cmd)

    proc = watchdog._real_spawn()

    # _real_spawn passes stdin=subprocess.DEVNULL; proc.stdin should be
    # None on the Popen side (the DEVNULL handle is consumed internally
    # and not exposed as a pipe).
    assert proc.stdin is None

    rc = proc.wait(timeout=10.0)
    assert rc == 0, f"sentinel subprocess exited rc={rc}, expected 0"


def test_real_spawn_raises_filenotfounderror_on_missing_exe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad argv (nonexistent exe) must produce ``FileNotFoundError``
    at the real ``subprocess.Popen`` boundary, not be silently
    swallowed.

    The unit-tier `test_watchdog_policy_tolerates_spawn_raising` covers
    the policy's RESPONSE to a raising spawn via `_FakeProcess` — but
    it does NOT verify that the REAL OS-level Popen seam actually
    raises on a missing exe. A regression in `_real_spawn` that, say,
    caught + logged + returned a dummy proc would make the policy
    layer see "spawn succeeded" forever and the watchdog loop
    silently no-ops while the user's pipeline never runs.

    Seeded-defect check: wrap _real_spawn's Popen call in
    `try: ... except FileNotFoundError: return _FakeProc()` (or
    similar silent-swallow) -> this test fails (no exception
    propagated, subprocess layer silently broken).
    """
    from yamabiko import watchdog

    # Build an argv whose exe doesn't exist. subprocess.Popen on
    # Windows raises FileNotFoundError with winerror 2 on a missing
    # executable. On POSIX it raises the same error class (ENOENT).
    bad_exe_cmd = [
        r"Z:\does-not-exist-for-yamabiko-ci\fake-exe.exe",
        "--no-op",
    ]
    monkeypatch.setattr(watchdog, "_build_main_cmd", lambda: bad_exe_cmd)

    with pytest.raises(FileNotFoundError):
        watchdog._real_spawn()


def test_watchdog_run_catches_real_popen_filenotfounderror_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `_real_spawn` raises `FileNotFoundError` (real Popen
    failure on a missing exe), `Watchdog.run()` must catch the
    exception, log it, and exit cleanly — not propagate the
    exception up through the thread and crash the supervisor.

    This is the policy-level counterpart to the test above. Together
    they cover both ends of the bug class: the OS boundary DOES
    raise (test above), and the Watchdog DOES handle the raise
    (this test).

    A spawn failure is a TERMINAL outcome and cascades. With no
    on_stop_all callback wired (bare Watchdog), _cascade_stop falls
    back to setting the local stop_event — leaving stop_event unset on
    a give-up is exactly what lets the llama supervisor + single-instance
    mutex hang forever.

    Seeded-defect checks:
      - Narrow Watchdog.run's `except Exception` to `except OSError`
        (or remove the try/except entirely) -> FileNotFoundError
        propagates out of wd.run(), this test fails with the
        propagated exception.
      - Remove `self._cascade_stop()` from the spawn-failure branch ->
        stop_event stays unset, the is_set() assertion below goes red.
    """
    from yamabiko import watchdog

    # Point _build_main_cmd at a nonexistent exe so _real_spawn raises
    # FileNotFoundError when called from inside Watchdog.run's try.
    bad_exe_cmd = [
        r"Z:\does-not-exist-for-yamabiko-ci\fake-exe.exe",
        "--no-op",
    ]
    monkeypatch.setattr(watchdog, "_build_main_cmd", lambda: bad_exe_cmd)

    wd = watchdog.Watchdog(
        name="test-pipeline",
        spawn=watchdog._real_spawn,  # the REAL spawn; not a fake
    )

    # Must not raise. Function returns cleanly after logging the
    # spawn failure. The critical assertion is the absence of a
    # raised exception: if the exception handler in Watchdog.run
    # were removed or narrowed, this call would propagate
    # FileNotFoundError.
    wd.run()

    # Absence of exception above is the primary signal. The
    # spawn-failure branch cascades via _cascade_stop(); with no
    # on_stop_all callback it falls back to the local stop_event, so
    # stop_event MUST be set -- that's what makes peer supervisors +
    # the mutex drop instead of hanging forever.
    assert wd.stop_event.is_set(), (
        "Watchdog.run did NOT cascade on a real spawn failure -- "
        "a give-up that leaves stop_event unset lets the "
        "llama supervisor + single-instance mutex hang forever"
    )


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="cmd.exe echo-to-stderr harness is Windows-only",
)
def test_real_llama_spawn_captures_child_stderr_into_llama_server_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """The REAL Popen redirect seam: a child spawned through the actual
    llama spawn path (``_make_real_spawn_llama._spawn``, real
    ``subprocess.Popen``, real inherited file handle) that writes to
    STDERR must land that output in ``logs_dir()/llama-server.log``.

    The llama binary is substituted with ``cmd /c echo boom 1>&2`` --
    only the cmdline is faked; the spawn path, the append-mode handle,
    the ``stderr=STDOUT`` fold, and the OS-level handle inheritance are
    all real. (Upstream llama-server logs to stdout/stderr only;
    ``--log-file`` is unreliable -- ggml-org discussions #6786/#7913 --
    so this subprocess-seam capture is the version-proof path, and this
    test is its rule-#3 real-tier coverage.)

    Seeded-defect drills:
      - drop ``stderr=subprocess.STDOUT`` from the Popen kwargs in
        ``_spawn`` -> the child's stderr goes to the void -> "boom"
        never appears in llama-server.log -> red.
      - drop the stdout redirect entirely -> no llama-server.log is
        created -> red.
    """
    import threading

    from yamabiko import watchdog

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    # Model presence gate: any real file satisfies the is_file() check.
    sentinel_model = tmp_path / "translator-sentinel.gguf"
    sentinel_model.write_bytes(b"gguf")
    monkeypatch.setattr(
        watchdog, "_resolve_translator_model_path", lambda: sentinel_model,
    )
    # Substitute the llama cmdline only. A plain STRING command line:
    # Popen passes it to CreateProcess verbatim on Windows, so cmd.exe
    # parses the 1>&2 redirect without list2cmdline quoting games.
    monkeypatch.setattr(
        watchdog, "_build_llama_cmd",
        lambda model_path, port=watchdog._LLAMA_PORT: "cmd /c echo boom 1>&2",
    )

    proc = watchdog._make_real_spawn_llama(threading.Event())()
    rc = proc.wait(timeout=15.0)
    assert rc == 0, f"echo child exited rc={rc}, expected 0"

    log_path = tmp_path / "logs" / "llama-server.log"
    assert log_path.is_file(), (
        f"llama-server.log was not created by the spawn path; logs dir: "
        f"{[p.name for p in (tmp_path / 'logs').iterdir()] if (tmp_path / 'logs').is_dir() else 'absent'}"
    )
    content = log_path.read_text(encoding="utf-8", errors="replace")
    assert "boom" in content, (
        f"the child's STDERR output never reached llama-server.log "
        f"(content: {content!r}); the stderr=STDOUT fold is broken -- "
        f"llama's crash/error output would be lost in the field"
    )
