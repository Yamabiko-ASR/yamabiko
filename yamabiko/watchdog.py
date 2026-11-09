"""Subprocess supervisor for the Yamabiko pipeline.

Wraps `python -m yamabiko.main` in a restart-on-failure loop with
exponential backoff and an anti-flap guard. The Start Menu shortcut in
the installer points at this module, not at main.py directly, so a
crashed pipeline auto-recovers without user intervention.

Tests inject ``spawn`` / ``clock`` / ``sleep`` to exercise the state
machine without actually running subprocesses.
"""
from __future__ import annotations

import argparse
import functools
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Optional, Protocol, Sequence


class _SafeTextIO:
    """Wrap a stream so OSError/ValueError on write or flush is swallowed.

    A frozen role spawned by another Yamabiko process with only stdin
    redirected (the watchdog's pipeline, the pipeline's overlay) gets
    ``sys.stdout`` / ``sys.stderr`` on a broken pipe: when the parent has no
    std handles, CPython hands the child a pipe whose read end it closes, so
    writes raise ``OSError [Errno 22] Invalid argument``, notably
    ``print(..., flush=True)``. In the frozen bundle, main.py emits a
    per-segment ``print(..., flush=True)`` for the ``[JA] ... [EN] ...``
    dev-console trace; when that raises, the exception bubbles out of
    _one_iter(), the translation-stage supervisor catches it, and the
    downstream ``manager.broadcast`` call that actually ships subtitles to
    the overlay never runs. Symptom: llama-server returns HTTP 200 for every
    segment, pipeline is alive, but no subtitles reach the overlay. After
    enough failures the supervisor's escalation threshold trips and the
    pipeline shuts down, visible as the tray icon disappearing.

    The frozen windowed bundle has no console for these writes anyway;
    real output flows through the rotating log file and the WebSocket
    broadcast channel. Silently dropping writes is the correct behavior.

    Dev mode (``python -m yamabiko.main``) has a real stdout, so this
    wrapper is only installed when ``sys.frozen`` is true.
    """

    __slots__ = ("_w",)

    def __init__(self, wrapped) -> None:
        self._w = wrapped

    def write(self, s):
        try:
            return self._w.write(s)
        except (OSError, ValueError):
            return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        try:
            self._w.flush()
        except (OSError, ValueError):
            pass

    # Delegate everything else (encoding, reconfigure, buffer, isatty,
    # close, fileno, writable, ...) to the wrapped stream so .reconfigure
    # calls in main.py and other hot paths still find what they expect.
    def __getattr__(self, name):
        return getattr(self._w, name)


def _guard_frozen_stdio() -> None:
    """Make ``sys.stdout`` and ``sys.stderr`` safe to write in the frozen app.

    A windowed yamabiko.exe started without std handles (Explorer, the Start
    Menu, the installer, a DETACHED spawn) gets None for both, as pythonw
    does, and libraries write to ``sys.stderr`` regardless: tqdm, which
    hf_hub_download's progress bar uses, defaults its file to
    ``sys.stderr`` and writes from its constructor, so a download failed
    before its first byte with AttributeError("'NoneType' object has no
    attribute 'write'"). A None stream gets a sink that discards (os.devnull:
    a real file, so fileno and reconfigure keep working), and both streams
    are wrapped in _SafeTextIO for the broken pipes a spawned child gets.
    """
    sys.stdout = _SafeTextIO(
        sys.stdout if sys.stdout is not None
        else open(os.devnull, "w", encoding="utf-8")
    )
    sys.stderr = _SafeTextIO(
        sys.stderr if sys.stderr is not None
        else open(os.devnull, "w", encoding="utf-8")
    )


if getattr(sys, "frozen", False):
    # First, before the Win32 bindings and every yamabiko import below, so
    # nothing imported later can write to a missing or broken stream before
    # the guard is in place.
    _guard_frozen_stdio()


if sys.platform.startswith("win"):
    import ctypes
    from ctypes import wintypes

    # Error-bearing Win32 calls are bound from a
    # ``use_last_error=True`` WinDLL so ctypes captures GetLastError
    # into thread-local storage atomically at each FFI call;
    # ``ctypes.get_last_error()`` then reads that captured slot.
    # Reading the LIVE ``GetLastError()`` as a separate FFI call is
    # racy: any ctypes/CPython-internal Win32 call between the failing
    # API and the read clobbers the thread's last-error value -- for
    # the single-instance mutex that means a clobbered
    # ERROR_ALREADY_EXISTS and a second instance sailing past the gate.
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _CreateMutexW = _k32.CreateMutexW
    _CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    _CreateMutexW.restype = wintypes.HANDLE

    _CloseHandle = _k32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

    # Handle-based orphan kill (open -> re-verify -> terminate
    # through the SAME pinned handle; see _kill_orphan_verified).
    _OpenProcess = _k32.OpenProcess
    _OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _OpenProcess.restype = wintypes.HANDLE

    _QueryFullProcessImageNameW = _k32.QueryFullProcessImageNameW
    _QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _QueryFullProcessImageNameW.restype = wintypes.BOOL

    _GetExitCodeProcess = _k32.GetExitCodeProcess
    _GetExitCodeProcess.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
    ]
    _GetExitCodeProcess.restype = wintypes.BOOL

    _TerminateProcess = _k32.TerminateProcess
    _TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _TerminateProcess.restype = wintypes.BOOL

    # Per-spawn kill-cascade guard (see _child_in_job). A NULL JobHandle
    # asks "is this process in ANY job", which is the question here: the
    # watchdog self-assigns to the outer job, so a child that inherited
    # membership is in a job and a child that broke away is not.
    _IsProcessInJob = _k32.IsProcessInJob
    _IsProcessInJob.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL),
    ]
    _IsProcessInJob.restype = wintypes.BOOL
else:
    # Non-Windows placeholder so module imports cleanly on dev machines
    # (the production target is Windows-only, but the test suite
    # still collects on POSIX). All uses are guarded with
    # sys.platform.startswith("win") checks at runtime.
    ctypes = None  # type: ignore[assignment]
    wintypes = None  # type: ignore[assignment]
    _k32 = None
    _CreateMutexW = None
    _CloseHandle = None
    _OpenProcess = None
    _QueryFullProcessImageNameW = None
    _GetExitCodeProcess = None
    _TerminateProcess = None
    _IsProcessInJob = None

from yamabiko import loopback_http
from yamabiko.lifecycle.exit_codes import (
    EXIT_CHILD_GIVEUP,
    EXIT_STAGE_ESCALATION,
    EXIT_STARTUP_FAILURE,
    EXIT_STREAM_DEATH,
    ExitPolicy,
    classify,
    is_external_stop,
)
from yamabiko.lifecycle.jobs import JobObject, create_outer_job, is_in_job
from yamabiko.lifecycle import startup_dialog
from yamabiko.lifecycle.shutdown_event import ENV_VAR_BY_ROLE, ShutdownEventWriter

# Give-up copy for a child stopped from outside Yamabiko (console close /
# Ctrl+C). Shared by both give-up surfaces below so the two cannot drift.
_EXTERNAL_STOP_TITLE = "Yamabiko was stopped"


def _external_stop_message(name: str) -> str:
    return (
        f"The {name} process was closed from outside Yamabiko (for example a "
        f"Ctrl+C, or its window being closed). Start Yamabiko again to "
        f"continue. If you didn't close anything, check the main-*.log files "
        f"in %LOCALAPPDATA%\\Yamabiko\\logs."
    )


def _installable_signals() -> list[tuple[int, str]]:
    """Return (signum, label) pairs for every shutdown signal this
    watchdog should handle on the current platform.

    SIGINT / SIGTERM are universal. SIGBREAK is Windows-only; guarded
    with hasattr so POSIX doesn't trip an AttributeError at import
    time. SIGBREAK is the signal CREATE_NEW_PROCESS_GROUP children
    receive when their parent sends CTRL_BREAK_EVENT, so registering
    it is what makes Ctrl+Break / tray-Quit-via-CTRL_BREAK run the
    Tier A graceful path instead of raising a default-handler
    KeyboardInterrupt that bypasses pipeline_writer.signal().
    """
    pairs: list[tuple[int, str]] = [
        (signal.SIGINT, "SIGINT"),
        (signal.SIGTERM, "SIGTERM"),
    ]
    if hasattr(signal, "SIGBREAK"):
        pairs.append((signal.SIGBREAK, "SIGBREAK"))
    return pairs


# Pin HF cache + enable high-performance xet before any HF code runs.
# The child process (yamabiko.main) imports yamabiko.hf_env too; duplicating
# here covers `watchdog.py --version` / `--run-main` paths in the frozen exe.
from yamabiko import hf_env  # noqa: F401 — side-effect import
from yamabiko import paths
from yamabiko import problems
from yamabiko.config import ConfigValidationError

# Hoisted to module level so run_watchdog's MAIN-thread
# drain references the module global `show_startup_error`, making it
# patchable in tests (a function-local `from ... import` would not be).
# error_dialog imports cleanly at module load: besides the standard
# library, its module-level deps are paths, models, lifecycle.startup_dialog,
# capture._audio_constants and ui.error_report (which adds gpu), none of
# them Qt and none importing this module. The window (ui.error_window,
# PySide6) is imported lazily inside show_startup_error, so this does not
# pull PySide6 in at watchdog import time and creates no circular import.
from yamabiko.ui.error_dialog import StartupError, show_startup_error

_log = logging.getLogger("yamabiko.watchdog")
# NOT ``getLogger(__name__)`` deliberately. PyInstaller's spec uses
# yamabiko/watchdog.py as the ``scripts=[...]`` entry, so in the frozen
# bundle this module is loaded with ``__name__ == "__main__"`` -- which
# would make every ``_log.*`` call in this file route to the
# __main__ logger, a sibling (not descendant) of the ``yamabiko`` logger
# that log_config.setup_main_log attaches the RotatingFileHandler to.
# Result before the fix: the watchdog process's logs never reached
# its log file, hiding every signal-handler event, health-probe trip,
# anti-flap give-up, and supervisor crash since PyInstaller adoption.
# Hardcoding the logger name makes behaviour identical across the
# unit-test import path (``from yamabiko import watchdog``) and the
# frozen-bundle script entry path.

_DEFAULT_BACKOFF_S = (1.0, 5.0, 15.0)
_DEFAULT_MAX_RESTARTS = 3
_DEFAULT_RESTART_WINDOW_S = 600.0
# A pipeline crash on startup (bad config, missing model, CUDA DLL not
# found, llama-server port conflict, ...) surfaces ~15 s after spawn --
# that's how long it takes to reach the failing stage. With 5 s here,
# a 15 s-to-crash process "looked healthy" to the anti-flap logic and
# got retried, so the user saw three consecutive crash dialogs before
# give_up fired. 30 s is long enough to cover the actual startup path
# (CUDA warmup + llama-server handshake + first audio callback) while
# still catching a real insta-crash in one shot.
_DEFAULT_MIN_LIFETIME_S = 30.0
_DEFAULT_POLL_INTERVAL_S = 1.0


class _ProcessLike(Protocol):
    def wait(self, timeout: Optional[float] = ...) -> int: ...
    def terminate(self) -> None: ...


def _parse_config_path(argv: list[str] | None) -> Path | None:
    """Extract ``--config`` from the watchdog's own argv.

    Mirrors main._parse_args's ``--config`` (``type=Path``) so the
    watchdog resolves the SAME effective config the pipeline child does:
    it drives the llama endpoint plan and the health probe, and
    ``_build_main_cmd`` re-emits this parse's result onto the child's
    argv (so ``--config=X`` / abbreviations forward identically).
    ``parse_known_args`` preserves the historical leniency
    (build_watchdog used to ``del argv``) so an unrelated flag never
    ``SystemExit``s the top-level process. Supports both ``--config X``
    and ``--config=X``.
    """
    if not argv:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    try:
        ns, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        return None
    return ns.config


def _build_main_cmd() -> list[str]:
    """Build the argv for spawning the main pipeline as a subprocess.

    In a PyInstaller-frozen bundle ``sys.executable`` is ``yamabiko.exe``
    (the bootloader), which does not understand ``-m module``. We instead
    re-enter the same exe with ``--run-main``; the watchdog's ``main()``
    detects the flag and calls ``yamabiko.main.main()`` in-process.

    In dev mode ``sys.executable`` is a real ``python`` interpreter, so
    the classic ``-m yamabiko.main`` invocation is preserved.

    Forward --max-segments and --config from the user's top-level
    yamabiko.exe invocation to the spawned pipeline child. Without
    this, `yamabiko.exe --max-segments 5` silently ignores the flag
    (watchdog falls through to default dual/triple-spawn path, child
    is spawned without the flag).

    --config is NOT token-matched: it is re-emitted from the SAME
    ``_parse_config_path`` parse the watchdog's own planning uses
    (build_watchdog's llama endpoint plan, run_watchdog's health
    probe). argparse accepts ``--config=repro.yaml`` (one argv token)
    and allow_abbrev prefixes like ``--conf repro.yaml``; the old
    ``if "--config" in argv`` membership test missed both, so the
    watchdog planned from the explicit file while the child booted on
    layered defaults -- e.g. a custom overlay.websocket_port made the
    health probe poll a port the child never binds, force-restarting a
    healthy pipeline after every fail budget.
    """
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--run-main"]
    else:
        cmd = [sys.executable, "-m", "yamabiko.main"]
    # Pass through --max-segments and --config if the user supplied
    # them to the top-level watchdog invocation. Both are re-emitted
    # from a real argparse parse, not token-matched: the membership
    # test missed ``--max-segments=5`` (one argv token) and
    # abbreviations -- the same class the --config re-emit fixed.
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-segments", type=int, default=None)
    try:
        ns, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        # A non-int value SystemExits argparse. Killing the top-level
        # process over one bad flag is worse than ignoring it, but the drop
        # must not be silent: the child then runs UNBOUNDED, and argparse's
        # own usage text went to stderr, which the frozen windowed bundle
        # discards. Name the value the user actually typed.
        _log.warning(
            "ignoring unparseable --max-segments in %r; the pipeline child "
            "will run unbounded", argv,
        )
        ns = None
    if ns is not None and ns.max_segments is not None:
        cmd.extend(["--max-segments", str(ns.max_segments)])
    config_path = _parse_config_path(argv)
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    return cmd


def _normalize_log_level(
    value: object, *, default: str = "INFO",
) -> tuple[str, "str | None"]:
    """Delegate to :func:`yamabiko.log_config.normalize_log_level` -- the
    single-source level normalizer.

    Keeping the watchdog's own ``setup_main_log`` call and the overlay
    ``--log-level`` forwarding on the same function as the pipeline's
    setLevel path means the three cannot drift. The ``(level_name,
    warning)`` shape is unchanged: the caller still controls warning
    emission timing (after ``setup_main_log`` attaches the file handler,
    so the warning survives the frozen bundle's discarded stderr).
    """
    from yamabiko.log_config import normalize_log_level
    return normalize_log_level(value, default=default)


def _build_overlay_cmd(
    url: str | None = None, *, level: str | None = None,
    config_path: Path | None = None,
) -> list[str]:
    """Build the argv for spawning the Qt desktop overlay as a subprocess.

    Same re-entry trick as ``_build_main_cmd``: in the frozen bundle the
    child is ``yamabiko.exe --run-overlay`` and ``main()`` routes to
    ``yamabiko.overlay_app.__main__.main()`` in-process. In dev mode we
    use ``python -m yamabiko.overlay_app``.

    ``url``: the ws:// endpoint the overlay should connect to.
    The overlay's argparse default is ws://127.0.0.1:8765/ws; uvicorn
    and the watchdog health probe both follow overlay.websocket_port,
    so a caller that knows the merged config MUST forward the port or
    a non-default-port overlay reconnect-loops against 8765 forever
    (no subtitles, tray commands silently dropped). The frozen
    re-entry strips only ``--run-overlay`` from argv, so ``--url``
    passes through to the overlay's argparse in both modes.

    ``level``: the ``logging.level`` value from the CALLER's
    already-loaded config, forwarded as ``--log-level`` (the overlay
    parses that flag but nothing ever forwarded it -- a hard-rule-#4
    producer/consumer gap). It is a parameter ON PURPOSE: this function
    must NOT re-read config from disk. The pipeline spawns the overlay
    with the config it ACTUALLY runs with (app.cfg -- which honours an
    explicit ``--config`` file), whereas a fresh ``load_full_config()``
    here silently ignored ``--config`` -- the same documented bug class
    as run_pipeline's lifecycle block. ``None`` (caller has no level)
    and invalid values degrade to the documented INFO default via
    ``_normalize_log_level``, because the overlay child's own
    ``setLevel`` would reject e.g. lowercase ``"debug"`` and a bad
    config value must not crash-loop the overlay.

    ``config_path``: the explicit config already selected by the pipeline.
    Forwarding the canonical path keeps the overlay's theme, style, and
    position on the same configuration source as the other runtime roles.
    """
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--run-overlay"]
    else:
        cmd = [sys.executable, "-m", "yamabiko.overlay_app"]
    if url is not None:
        cmd.extend(["--url", url])
    level_name, level_warn = _normalize_log_level(level)
    if level_warn:
        # Emitted in the SPAWNING process's log (the pipeline in
        # production), whose handlers are long since attached.
        _log.warning("overlay cmd: %s", level_warn)
    cmd.extend(["--log-level", level_name])
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    return cmd


def _real_spawn(env: Optional[dict] = None) -> subprocess.Popen:
    """Production default: start the pipeline as a child process.

    Accepts an optional ``env`` dict so callers (e.g. the pipeline
    wrapper in ``build_watchdog``) can inject
    ``YAMABIKO_SHUTDOWN_EVENT`` pointing at the parent-owned named
    event. When ``env`` is None, the child inherits the current
    process environment as usual.
    """
    cmd = _build_main_cmd()
    _log.info("spawn: %s", " ".join(cmd))
    spawn_env = os.environ.copy() if env is None else env
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        env=spawn_env,
    )


# ---------------------------------------------------------------------------
# llama-server supervision
#
# The translator GGUF lives in the HF cache; its exact path depends on
# translation.quant_tier (see yamabiko.models.TRANSLATOR_PINS). On first
# run the wizard hasn't downloaded the model yet, so the spawn blocks
# on a model-cache poll until the wizard's DownloadsPage completes.
# The vendored llama-server.exe lives at <bundle>/_internal/llama/
# (copied by installer/build.ps1).
# ---------------------------------------------------------------------------

_LLAMA_PORT = 8090
_LLAMA_CTX_SIZE = 2048
# llama-server.log rotation threshold, checked once per spawn BEFORE the
# child gets the handle (the running child owns the only open handle to
# the file, so there is no in-flight rotation -- only pre-spawn).
_LLAMA_LOG_MAX_BYTES = 10 * 1024 * 1024
# Per-tier repo/file/revision pins AND the served alias from the single
# source of truth (yamabiko.models) so the runtime resolver cannot drift
# from the wizard's download list, and the ``--alias`` handed to
# llama-server cannot drift from the ``translation.model`` name the
# Translator addresses requests to (a producer/consumer mismatch either
# way).
from yamabiko.models import (  # noqa: E402
    TRANSLATOR_ALIAS as _LLAMA_ALIAS,
    TRANSLATOR_PINS as _TRANSLATOR_PINS,
    translator_pin as _translator_pin,
)


def _resolve_translator_model_path(explicit_path: Optional[Path] = None):
    """Return the absolute Path to the translator GGUF in the HF cache, or
    None.

    Reads translation.quant_tier from the merged config to pick the tier
    pin (``default`` ~5.1 GB for 10 GB+ cards, ``small`` ~4 GB for 8 GB
    cards); each tier carries its own repo + filename + revision
    (yamabiko.models.TRANSLATOR_PINS -- the tiers may live in different
    repos). ``explicit_path`` is the watchdog's ``--config`` file (threaded
    from build_watchdog via the llama spawn factory): when set, quant_tier
    resolves from that file layered over defaults -- the same effective
    config every other role sees -- instead of the layered store, so a
    --config repro of the small-tier path actually loads the small model.
    Uses ``huggingface_hub.try_to_load_from_cache`` so we don't trigger a
    download — pure cache lookup.

    Returns None if:
        - The file is not yet cached (first run before wizard download)
        - The quant_tier value is unexpected
        - Any exception during cache lookup
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        from yamabiko.config import load_full_config
    except Exception as exc:
        _log.warning("llama-server model-path resolver import failed: %s", exc)
        return None

    try:
        if explicit_path is not None:
            cfg = load_full_config(explicit_path=explicit_path)
        else:
            cfg = load_full_config()
        quant_tier = cfg.get("translation", {}).get("quant_tier", "default")
    except Exception as exc:
        _log.warning("llama-server could not read quant_tier from config: %s", exc)
        quant_tier = "default"

    def _cached(pin):
        try:
            # Pinned revision: the wizard downloads the tier at its pinned
            # sha, which writes only snapshots/<sha>/ (no refs/main), so a
            # bare probe would default to `main` and miss the warm cache.
            # Mirrors the whisper/silero pinned-probe pattern.
            result = try_to_load_from_cache(
                repo_id=pin.repo, filename=pin.filename, revision=pin.revision,
            )
        except Exception as exc:
            _log.warning("llama-server try_to_load_from_cache raised: %s", exc)
            return None
        # huggingface_hub returns the sentinel ``_CACHED_NO_EXIST`` string
        # for files known-not-to-exist on the remote; treat same as None.
        if result is None or (isinstance(result, str) and not Path(result).is_file()):
            return None
        return Path(str(result))

    pin = _translator_pin(quant_tier)
    path = _cached(pin)
    if path is not None:
        return path

    # The configured tier's GGUF isn't in the cache. The Settings
    # quant-tier switch writes translation.quant_tier but never downloads the
    # new file, so without this fallback the spawn loop below would poll every
    # 5 s FOREVER for a file that will never appear -- translation silently
    # dead. Self-heal to ANOTHER tier if it IS cached (the user keeps
    # translating at the previous quant) and warn; re-running setup downloads
    # the requested quant. If NO tier is cached this returns None, preserving
    # the legitimate first-run "waiting for the wizard download" path.
    for other_tier, other_pin in _TRANSLATOR_PINS.items():
        if other_pin == pin:
            continue
        other_path = _cached(other_pin)
        if other_path is not None:
            _log.warning(
                "configured quant_tier %r model (%s) not in the HF cache; "
                "falling back to the cached %s tier (%s). Re-run first-time "
                "setup to download the %r quant.",
                quant_tier, pin.filename, other_tier, other_pin.filename,
                quant_tier,
            )
            return other_path
    return None


def _model_path_is_ascii(model_path: str | os.PathLike) -> bool:
    """True iff the full model path is pure ASCII.

    The single source of truth for the CJK-path workaround: BOTH
    ``_build_llama_cmd`` (which emits a relative ``--model`` basename
    when this is False) and ``_llama_cwd`` (which sets the child cwd to
    the model's parent when this is False) key off this one predicate,
    so the two halves can never drift apart.
    """
    return str(model_path).isascii()


def _build_llama_cmd(
    model_path: str | os.PathLike, port: int = _LLAMA_PORT,
) -> list[str]:
    """Argv for spawning the vendored llama-server against the translator
    GGUF.

    Args mirror scripts/start-llama-server.ps1 (the canonical dev-mode
    launcher), minus the --no-webui flag we always want. The --models-max
    1 + --sleep-idle-seconds -1 pair pins the model in VRAM indefinitely
    so reloading is a non-issue; see OVERVIEW.md "llama.cpp router mode".

    ``port``: derived from ``translation.endpoint`` by
    build_watchdog when the endpoint is loopback-but-non-default, so
    the server binds where the pipeline will actually connect.

    Upstream ggml-org/llama.cpp#18571: the vendored llama-server's
    ``ggml_fopen`` does a broken char->wchar cast that corrupts
    NON-ASCII ``--model`` paths -- so a JP user whose model lives under a
    CJK ``%LOCALAPPDATA%`` (the modal case) gets "failed to open GGUF
    file" and translation never starts. Workaround: when the model path
    is non-ASCII, pass only the model's BASENAME (ASCII -- models.py
    filenames are ASCII) as ``--model`` and run the child with
    cwd=<model parent> (see ``_llama_cwd``). The broken cast is harmless
    on pure-ASCII argv bytes, and subprocess.Popen sets cwd via
    CreateProcessW (Unicode-correct), so llama's
    ``fopen("<model>.gguf","rb")`` resolves the relative ASCII name
    against the correct Unicode directory. Do NOT "simplify" this back
    to an absolute ``--model`` -- that reintroduces the bug.
    GetShortPathNameW 8.3 was rejected: on this target it returns a
    still-non-ASCII alias for CJK dirs (8dot3 creation disabled), so it
    wouldn't help.
    """
    if getattr(sys, "frozen", False):
        # Frozen: llama-server.exe is bundled under _internal/llama/
        # by installer/build.ps1. sys._MEIPASS points at _internal/.
        bundle_internal = getattr(sys, "_MEIPASS", None)
        if bundle_internal is None:
            # Fallback: alongside yamabiko.exe
            bundle_internal = str(Path(sys.executable).parent / "_internal")
        llama_exe = Path(bundle_internal) / "llama" / "llama-server.exe"
    else:
        # Dev mode: third_party/llama.cpp/
        llama_exe = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "llama-server.exe"

    if _model_path_is_ascii(model_path):
        model_arg = str(model_path)
    else:
        # CJK-path workaround (see docstring): pass the ASCII basename;
        # _llama_cwd() pins the child cwd to the parent so the relative
        # name resolves. Defensive: if the basename itself were non-ASCII
        # (models.py filenames are ASCII, so this should never happen),
        # the cwd trick can't rescue it -- warn and pass the basename
        # anyway rather than the corrupting absolute path.
        model_arg = Path(model_path).name
        if not model_arg.isascii():
            # error, not warning: this means translation WILL fail to
            # load. Log the FULL path too so a log-bisect from the
            # "failed to open GGUF file" symptom finds the cause. Don't
            # raise -- the precondition (ASCII models.py filenames) makes
            # this unreachable, and a hard block would be worse UX than
            # letting the (already-doomed) spawn surface the real error.
            _log.error(
                "llama-server model basename %r is itself non-ASCII "
                "(full path %r); the cwd-relative workaround for upstream "
                "llama.cpp#18571 cannot make --model ASCII, so the GGUF "
                "load will fail",
                model_arg, str(model_path),
            )
    return [
        str(llama_exe),
        "--model", model_arg,
        "--alias", _LLAMA_ALIAS,
        "--port", str(port),
        "--ctx-size", str(_LLAMA_CTX_SIZE),
        "-ngl", "99",
        "-fa", "on",
        # --jinja explicitly: translator.py sends chat_template_kwargs=
        # {enable_thinking: False} to switch Qwen3 to non-thinking mode, and
        # llama-server only honors that kwarg with the jinja engine active.
        # The currently vendored binary defaults --jinja on, but a refresh to
        # a --jinja-off build would silently re-enable thinking -- so force it
        # rather than depend on the default (rule #7). We deliberately do NOT
        # pass --reasoning-format: the default (auto) extracts any stray
        # reasoning into message.reasoning_content and strips it from
        # message.content. --reasoning-format none would instead leave Qwen3's
        # empty non-thinking <think></think> block UNPARSED in message.content,
        # which false-trips translator.py's <think> guard and fails the startup
        # sanity check (a frozen-E2E-caught regression). translator.py guards
        # BOTH message.content (<think>) and message.reasoning_content, so auto
        # is correct and safe.
        "--jinja",
        "--models-max", "1",
        "--sleep-idle-seconds", "-1",
        "--no-webui",
    ]


def _llama_cwd(model_path: str | os.PathLike) -> Optional[str]:
    """Working directory for the spawned llama-server, or None.

    Returns ``str(model_path.parent)`` when the model path is non-ASCII
    (so the relative ASCII ``--model`` basename emitted by
    ``_build_llama_cmd`` resolves against the correct Unicode directory
    -- the llama.cpp#18571 workaround), else None (ASCII paths use
    an absolute ``--model`` and need no special cwd). Keyed off the same
    ``_model_path_is_ascii`` predicate as ``_build_llama_cmd`` so the
    argv and cwd halves cannot drift.
    """
    if _model_path_is_ascii(model_path):
        return None
    return str(Path(model_path).parent)


def _open_llama_log():
    """Append-mode capture handle for the llama child's output, or None.

    Upstream llama-server logs to stdout/stderr only; its ``--log-file``
    flag is unreliable (ggml-org discussions #6786 / #7913), so the
    version-proof capture path is the subprocess seam: the watchdog
    hands the child an inherited handle to
    ``logs_dir()/llama-server.log``. Opened in append so a respawn after
    a crash doesn't clobber the previous run's tail -- the tail IS the
    crash evidence. If the file is over ``_LLAMA_LOG_MAX_BYTES`` at open
    it is ``os.replace``d to ``llama-server.log.1`` first (atomic
    clobber of a stale .1; plain rename raises FileExistsError on
    Windows), so the capture cannot grow without bound across respawns.

    Returns None (after a WARNING) on any open failure: llama-server
    must never be blocked on logging -- a read-only or pre-empted logs
    dir degrades to a redirect-less spawn, not a dead translator.
    """
    try:
        paths.ensure_dirs()
        log_path = paths.logs_dir() / "llama-server.log"
        try:
            if (
                log_path.exists()
                and log_path.stat().st_size > _LLAMA_LOG_MAX_BYTES
            ):
                os.replace(
                    log_path, log_path.with_name(log_path.name + ".1"),
                )
        except OSError as exc:
            # Rotation is best-effort; append to the oversize file
            # rather than lose llama's output entirely. But leave
            # evidence -- once per spawn (this helper runs once per
            # spawn): a bare swallow here would be exactly the
            # evidence-free pattern this capture feature exists to fix.
            _log.warning(
                "llama-server.log rotation failed (%s -> %s): %s; "
                "appending to the oversize file instead",
                log_path, log_path.name + ".1", exc,
            )
        return open(log_path, "a", encoding="utf-8", errors="replace")
    except OSError as exc:
        _log.warning(
            "could not open llama-server.log for output capture (%s); "
            "spawning llama-server WITHOUT redirect -- its stdout/stderr "
            "will be lost for this run", exc,
        )
        return None


def _make_real_spawn_llama(
    stop_event, port: int = _LLAMA_PORT,
    config_path: Optional[Path] = None,
):
    """Factory for the llama-server spawn callable.

    Returns a function that blocks (polling every 5 s) until the translator
    model is present in the HF cache, then spawns the bundled
    llama-server. The wait is cancellable via the provided event (in
    production, ``llama_stop_event``); if it is set during the wait the
    spawn raises instead of blocking — so a clean shutdown during the
    first-run wizard window doesn't leave a zombie supervisor.

    ``port``: forwarded to the llama-server cmdline; derived
    from ``translation.endpoint`` by build_watchdog.

    ``config_path``: the watchdog's ``--config`` file (or None for the
    layered load), threaded from build_watchdog so the model resolver's
    quant_tier comes from the same effective config as the rest of the
    llama plan. The None branch calls the resolver bare so the layered
    path stays byte-identical.
    """
    def _spawn():
        while not stop_event.is_set():
            if config_path is not None:
                model_path = _resolve_translator_model_path(
                    explicit_path=config_path,
                )
            else:
                model_path = _resolve_translator_model_path()
            if model_path is not None and model_path.is_file():
                _log.info("llama-server spawning with model: %s", model_path)
                # Capture the child's stdout+stderr into
                # logs_dir()/llama-server.log (see _open_llama_log for
                # why the subprocess seam, not --log-file). stderr folds
                # into the same file via subprocess.STDOUT -- llama's
                # error output is the part that matters for a crash.
                # On open failure the redirect kwargs are omitted
                # entirely (never hand Popen a dead handle).
                log_handle = _open_llama_log()
                redirect_kwargs = {}
                if log_handle is not None:
                    redirect_kwargs["stdout"] = log_handle
                    redirect_kwargs["stderr"] = subprocess.STDOUT
                    # One INFO line naming the capture file so a triager
                    # reading main-watchdog.log knows where llama's
                    # output went.
                    _log.info(
                        "llama-server output (stdout+stderr) captured "
                        "to %s", log_handle.name,
                    )
                # cwd=_llama_cwd(model_path): None for ASCII paths
                # (absolute --model, behaviour unchanged); the model's
                # parent dir for non-ASCII paths so the relative ASCII
                # --model basename resolves (llama.cpp#18571 -- see
                # _build_llama_cmd). Local to this Popen; does not touch
                # the supervisor's stop-event / Job Object / respawn.
                try:
                    return subprocess.Popen(
                        _build_llama_cmd(model_path, port=port),
                        cwd=_llama_cwd(model_path),
                        stdin=subprocess.DEVNULL,
                        # llama-server.exe is a console-subsystem binary and
                        # yamabiko.exe is windowed: without CREATE_NO_WINDOW
                        # Windows allocates a visible terminal for the child.
                        # Shutdown is terminate()/Job-based, never console
                        # events, so the child needs no console.
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        **redirect_kwargs,
                    )
                finally:
                    # Close the PARENT's copy of the handle once Popen
                    # returns (chosen over supervisor-side close-after-
                    # exit; either is correct, this one is simplest):
                    # the child holds its own OS-duplicated handle, so
                    # writes keep flowing. Keeping a parent copy open
                    # would make the NEXT pre-spawn rotation's
                    # os.replace fail for the watchdog's whole lifetime
                    # (Windows renames fail on open files).
                    if log_handle is not None:
                        try:
                            log_handle.close()
                        except OSError:
                            pass
            _log.info(
                "llama-server waiting for translator model to finish "
                "downloading (run the first-run wizard to download it)"
            )
            # Wait up to 5 s for the model, cancel early on stop.
            stop_event.wait(timeout=5.0)
        # Shutdown requested during wait. Raise so the outer run() loop
        # exits cleanly per its existing "failed to spawn" handler.
        raise RuntimeError("llama-server supervisor: stop_event set during model wait")
    return _spawn


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _llama_endpoint_plan(cfg) -> tuple[bool, int]:
    """Decide from ``translation.endpoint`` whether build_watchdog
    should spawn the bundled llama-server, and on which port.

    Returns ``(spawn_llama, port)``:

    * Non-loopback host -> ``(False, _LLAMA_PORT)``. The user translates
      against their own server; spawning the bundled one would burn
      ~4 GB VRAM on a server nothing calls. The pipeline side
      (main.py's translator) already honours the endpoint; this closes
      the producer half.
    * Loopback host with an explicit port -> ``(True, port)``. The
      bundled server binds where the pipeline will actually connect.
    * Loopback without an explicit port, missing/empty endpoint, or an
      unparseable value -> ``(True, _LLAMA_PORT)`` -- preserve today's
      default behaviour and let the existing config plumbing surface
      the misconfiguration.
    """
    default = (True, _LLAMA_PORT)
    try:
        endpoint = (cfg.get("translation") or {}).get("endpoint")
    except Exception:
        return default
    if not endpoint:
        return default
    try:
        from urllib.parse import urlsplit
        raw = str(endpoint).strip()
        # A schema-less value ("remotehost:8090/v1" -- the shorthand
        # llama.cpp docs use) makes urlsplit read the host as the SCHEME
        # (hostname=None), which fell through to "spawn on the default
        # port" -- the exact misconfiguration this function exists to
        # honor. Re-parse with a netloc prefix when no scheme separator
        # is present.
        parts = urlsplit(raw if "://" in raw else "//" + raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except Exception:
        _log.warning(
            "translation.endpoint %r is unparseable; spawning the bundled "
            "llama-server on the default port %d", endpoint, _LLAMA_PORT,
        )
        return default
    if host and host not in _LOOPBACK_HOSTS:
        _log.info(
            "external translation endpoint configured (%s); the bundled "
            "llama-server will not be started", endpoint,
        )
        return (False, _LLAMA_PORT)
    if port is not None and port != _LLAMA_PORT:
        _log.info(
            "loopback translation endpoint on non-default port %d; the "
            "bundled llama-server will bind there", port,
        )
        return (True, port)
    return default


class _CarryRelay:
    """Hands the problems a pipeline restarted on (problems.write_carry) to
    the next pipeline, and restarts llama-server when the pipeline exited
    for a translation escalation (plan Decisions 18 and 21). The shortcuts
    the pipeline announced as taken go to the next pipeline too, so it does
    not announce them again (Decision 72).

    ``on_pipeline_restart`` is the pipeline supervisor's ``on_restart``
    hook; ``apply`` puts the carried kinds and shortcuts into the next
    spawn's environment, once. ``llama`` is the llama supervisor when
    Yamabiko runs the bundled llama-server, else None."""

    def __init__(self, take: Callable[[], "problems.Carry | None"] = problems.take_carry) -> None:
        self._take = take
        self._env_value: str | None = None
        self._hotkeys_value: str | None = None
        self.llama: Optional["Watchdog"] = None

    def on_pipeline_restart(self, code: int) -> None:
        carry = self._take()
        self._env_value = (
            problems.carry_env_value(carry.kinds)
            if carry is not None and carry.kinds else None
        )
        self._hotkeys_value = (
            problems.carry_hotkeys_env_value(carry.hotkeys)
            if carry is not None and carry.hotkeys else None
        )
        if carry is not None and carry.restart_translator and self.llama is not None:
            _log.info(
                "pipeline exited %d after a translation escalation; "
                "restarting llama-server", code,
            )
            self.llama.request_restart()

    def apply(self, env: dict) -> None:
        for name, value in ((problems.CARRY_ENV, self._env_value),
                            (problems.CARRY_HOTKEYS_ENV, self._hotkeys_value)):
            if value:
                env[name] = value
            else:
                env.pop(name, None)
        self._env_value = None
        self._hotkeys_value = None


def _child_in_job(proc: _ProcessLike) -> "bool | None":
    """Is ``proc`` a member of a Job Object?

    ``True`` / ``False`` when Windows can answer; ``None`` when the question
    is unanswerable (non-Windows, a test fake with no OS handle, a failed
    query). Callers must treat ``None`` as "no signal" and warn only on an
    explicit ``False``: an unanswerable query must not manufacture a scary
    warning on every spawn.

    This is the only PER-SPAWN evidence that the kill cascade is intact.
    build_watchdog self-assigns the watchdog process to the outer job, so
    children inherit membership by kernel rule and no per-child
    AssignProcessToJobObject runs on the happy path (re-assigning an
    existing member fails ERROR_ACCESS_DENIED). Without this check, a
    future breakaway flag -- or an external-owner job that refuses nesting
    -- would drop children out of the job silently, and the first evidence
    would be orphaned llama-servers holding VRAM after a watchdog crash.
    """
    if not sys.platform.startswith("win") or _IsProcessInJob is None:
        return None
    handle = getattr(proc, "_handle", None)
    if handle is None:
        return None
    try:
        out = wintypes.BOOL(False)
        if not _IsProcessInJob(int(handle), None, ctypes.byref(out)):
            _log.debug(
                "IsProcessInJob failed: err=%d", ctypes.get_last_error(),
            )
            return None
        return bool(out.value)
    except Exception as exc:
        _log.debug("job-membership query raised: %s", exc)
        return None


class Watchdog:
    def __init__(
        self,
        *,
        name: str = "watchdog",
        spawn: Callable[[], _ProcessLike] = _real_spawn,
        clock: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], None]] = None,
        max_restarts: int = _DEFAULT_MAX_RESTARTS,
        restart_window_s: float = _DEFAULT_RESTART_WINDOW_S,
        min_lifetime_s: float = _DEFAULT_MIN_LIFETIME_S,
        backoff_s: Sequence[float] = _DEFAULT_BACKOFF_S,
        stop_event: Optional[threading.Event] = None,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        on_give_up: Optional[Callable[[str, str], None]] = None,
        job: Optional[JobObject] = None,
        grace_s: float = 5.0,
        on_stop_all: Optional[Callable[[], None]] = None,
        overlay_port: Optional[int] = None,
        in_job_check: Callable[[_ProcessLike], "bool | None"] = _child_in_job,
        on_restart: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.name = name
        self._spawn = spawn
        self._clock = clock
        self._max_restarts = max_restarts
        self._restart_window_s = restart_window_s
        self._min_lifetime_s = min_lifetime_s
        self._backoff_s = tuple(backoff_s)
        self._restart_times: Deque[float] = deque()
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        # The production backoff wait is stop_event.wait(delay),
        # NOT time.sleep(delay) -- a stop request (SIGINT, tray-Quit,
        # STOP_ALL cascade) arriving mid-backoff wakes the supervisor
        # immediately instead of blocking up to the 15 s backoff tier
        # while the single-instance mutex is still held. Mirrors the
        # interruptible pattern already used by
        # lifecycle/pipeline_supervisor.py's backoff. Tests may still
        # inject a recording ``sleep`` to observe the chosen delays;
        # run()'s loop-head stop_event check preserves the early-exit
        # semantics either way.
        def _interruptible_backoff(delay: float) -> None:
            self._stop_event.wait(timeout=delay)

        self._sleep = sleep if sleep is not None else _interruptible_backoff
        self._poll_interval_s = poll_interval_s
        self._on_give_up = on_give_up
        # Called on EVERY terminal outcome (clean
        # STOP_SELF, STOP_ALL exit code, spawn-failure, min-lifetime
        # give-up, max-restart give-up) to trigger coordinated shutdown
        # of ALL supervisors via the shared stop_all. A RESTART outcome
        # NEVER calls it. When None (unit tests / direct instantiation)
        # _cascade_stop falls back to setting this watchdog's own
        # stop_event, preserving the old single-supervisor semantics.
        self._on_stop_all = on_stop_all
        self._job = job  # None in unit tests; real Job Object in production
        # Grace window the watchdog allows for a child's graceful finally
        # to complete after stop_event fires, before falling back to
        # terminate(). See _wait_or_stop. 5.0 is a safe default for tests
        # and direct instantiation; build_watchdog overrides from config
        # (lifecycle.grace_s.pipeline).
        self._grace_s = grace_s
        # Handle to the child currently being supervised, so an external
        # observer (HealthProbe wedge detector) can force a
        # restart by terminating it WITHOUT setting stop_event. Guarded
        # because run() (supervisor thread) and request_restart()
        # (probe thread) touch it concurrently.
        self._current_proc: Optional[_ProcessLike] = None
        self._proc_lock = threading.Lock()
        # Count of children spawned so far, bumped under _proc_lock as
        # each child is published. HealthProbe reads it on every poll:
        # a new value means a new child, which gets its own cold-boot
        # grace instead of inheriting the previous child's liveness
        # state and fail timer. See spawn_generation().
        self._spawn_generation = 0
        # Configured overlay.websocket_port, wired only for the pipeline
        # supervisor (the sole child that can exit EXIT_STARTUP_FAILURE).
        # Used to name the offending port in the port-bind give-up dialog.
        self._overlay_port = overlay_port
        # Post-spawn kill-cascade guard; injectable so the miss path is
        # testable without a real breakaway child. See _child_in_job.
        self._in_job_check = in_job_check
        # Called with the child's exit code just before each respawn (plan
        # phase 4: the pipeline supervisor's problem carry, _CarryRelay).
        self._on_restart = on_restart
        # The child request_restart terminated, so its exit is a restart we
        # asked for, never a fast crash (Decision 21). Guarded by _proc_lock.
        self._requested_restart_proc: Optional[_ProcessLike] = None

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def spawn_generation(self) -> int:
        """Number of children this supervisor has spawned (0 before the
        first). Read lock-free: the supervisor thread is the only writer
        and an int read is atomic.
        """
        return self._spawn_generation

    def _wait_or_stop(self, proc: _ProcessLike) -> Optional[int]:
        """Wait for ``proc`` to exit OR the stop event to fire.

        Returns the exit code if the child exited naturally (either
        during normal operation or within the graceful-shutdown window
        after stop_event), or None if we had to terminate/kill it.

        Tier A contract: a graceful shutdown signal (e.g.
        ``ShutdownEventWriter.signal()``) is expected to have been
        delivered by the signal handler BEFORE stop_event was set.
        This method gives the child ``self._grace_s`` to notice the
        event and run its ``finally`` cleanup before falling back to
        terminate(), realising the spec's SetEvent -> grace ->
        terminate -> kill ladder (previously this function collapsed
        straight to terminate).
        """
        while not self._stop_event.is_set():
            try:
                return proc.wait(timeout=self._poll_interval_s)
            except subprocess.TimeoutExpired:
                continue
        # Stop requested. Allow the child its graceful-finally window
        # to process the named-event ping that the signal handler fired
        # before tripping stop_event. If it exits naturally within the
        # grace window, return the exit code (preserves STOP_SELF /
        # STOP_ALL classify() semantics); otherwise fall through to
        # terminate/kill.
        _log.info(
            "stop requested; waiting %.1fs for graceful exit before terminate",
            self._grace_s,
        )
        try:
            return proc.wait(timeout=self._grace_s)
        except subprocess.TimeoutExpired:
            _log.info("grace window elapsed; terminating child")
        try:
            proc.terminate()
        except Exception as exc:
            _log.warning("terminate() failed: %s", exc)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _log.warning(
                "child did not exit within 2s of terminate; falling back to kill"
            )
            try:
                # _ProcessLike doesn't formally have kill(); real Popen does.
                kill = getattr(proc, "kill", None)
                if kill is not None:
                    kill()
                    proc.wait(timeout=2.0)
            except Exception as exc:
                _log.error("kill failed: %s", exc)
        return None

    def request_restart(self) -> None:
        """Force-terminate the current child so the restart loop respawns it.

        Used by the HealthProbe wedge detector: a hung pipeline
        (process alive, /livez unresponsive) is force-killed here so
        ``_wait_or_stop``'s ``proc.wait()`` returns, ``run`` classifies the
        terminated child as RESTART, and the existing anti-flap loop
        respawns it. Deliberately does NOT set stop_event (that would
        stop the supervisor instead of restarting). Idempotent / safe if
        no child is currently running.

        A child terminated here is exempt from the min-lifetime give-up: a
        young llama-server restarted after a translation escalation is not
        a configuration error.

        ``run()`` publishes ``_current_proc`` under the SAME lock
        acquisition that calls ``_spawn()``, so a request landing
        mid-spawn parks on the lock here and then terminates the
        freshly-published child instead of silently no-oping for
        another full fail_budget_s.

        Only ``OSError`` is swallowed, and the child is then not marked
        (the child raced us and died -- an expected OS-level outcome).
        Programming bugs (AttributeError-class) propagate to the caller's
        thread so they surface via threading.excepthook instead of hiding
        behind a routine-looking WARN.
        """
        with self._proc_lock:
            proc = self._current_proc
            if proc is None:
                return
            if getattr(proc, "returncode", None) is not None:
                # Already exited and reaped on its own: not a restart we
                # asked for, so it is neither terminated nor marked.
                return
            try:
                proc.terminate()
            except OSError as exc:
                _log.warning("request_restart: terminate() failed: %s", exc)
                return
            # Marked only once terminate() returned, and under the lock the
            # loop takes after the child's exit, so the loop sees the mark:
            # this exit is a restart we asked for and skips the min-lifetime
            # give-up (Decision 21). Accepted race: a child that exited on its
            # own but is not reaped yet still gets marked (Windows terminate()
            # does not raise for it), so it skips the give-up once.
            self._requested_restart_proc = proc

    def _cascade_stop(self) -> None:
        """Trigger coordinated shutdown of ALL supervisors.

        The single rule for every TERMINAL outcome.
        In production ``on_stop_all`` is wired to set the MAIN stop_event,
        which the ``_mirror`` thread forwards to llama_stop_event, so all
        supervisors' ``_wait_or_stop`` see their event and run the
        grace->terminate ladder; run_watchdog's join then returns. When no
        callback is wired (unit tests) it falls back to setting the local
        stop_event so the watchdog still stops itself.
        """
        if self._on_stop_all is not None:
            self._on_stop_all()
        else:
            self._stop_event.set()

    # NOTE: Watchdog used to carry a `_run_once()` helper that duplicated
    # run()'s spawn/assign/wait sequence. It had ZERO production callers
    # but anchored the job-assignment guard tests, so deleting
    # `job.assign(proc)` from the LIVE run() loop kept the suite green
    # (dead-code anchor; removed).
    # the tests now drive run() itself. Do not reintroduce it.

    def run(self) -> None:
        """Supervise the child in a restart loop.

        Returns cleanly when (a) the child exits 0, (b) a crash-loop cap is
        hit, (c) a sub-min-lifetime crash occurs, (d) the stop_event is set
        (via ``main()``'s SIGINT/SIGTERM handlers), or (e) the supervision
        logic ITSELF raises -- surfaced as an internal-error give-up, never
        a silently-dead thread.
        """
        try:
            self._run_supervision_loop()
        except Exception as exc:
            # Layer-2 self-crash evidence: a bug in the supervision
            # logic itself (classify, anti-flap arithmetic, wait/stop
            # plumbing) used to propagate out of this thread and die in
            # threading's default excepthook -- run_watchdog's join then
            # returned and the process exited 0, indistinguishable from
            # a clean user quit. A supervision bug must SURFACE: log the
            # traceback, fire the give-up dialog with an internal-error
            # title (NOT a child-crash title -- the child didn't do
            # anything wrong), and cascade so peer supervisors + the
            # single-instance mutex drop instead of hanging forever.
            _log.exception("supervisor %s: internal error", self.name)
            if self._on_give_up is not None:
                self._on_give_up(
                    "Yamabiko hit an internal error",
                    f"The {self.name} supervisor hit an internal error "
                    f"and stopped: {exc}. Check main-watchdog.log in "
                    f"%LOCALAPPDATA%\\Yamabiko\\logs for the traceback.",
                )
            self._cascade_stop()

    def _run_supervision_loop(self) -> None:
        """The spawn/wait/classify/backoff state machine run() guards."""
        while not self._stop_event.is_set():
            try:
                # Spawn AND publish under one lock acquisition:
                # request_restart() takes _proc_lock, so there is no
                # instant where the child exists but is invisible to it
                # -- a wedge request landing mid-spawn parks on the lock
                # and then terminates the fresh child instead of being
                # dropped. Holding the lock across _spawn() is safe: the
                # only other taker is request_restart, from the health
                # probe on the pipeline supervisor, whose spawn is a
                # prompt Popen. (The llama spawn can block on the
                # model-wait; _CarryRelay calls request_restart on the
                # llama supervisor only after a pipeline exit, when
                # llama-server has long been running.)
                with self._proc_lock:
                    proc = self._spawn()
                    self._current_proc = proc
                    self._spawn_generation += 1
            except Exception as exc:
                # A STOP-DRIVEN spawn abort is not a failure: the llama
                # spawn raises by design when stop_event is set during
                # its model-wait (see _make_real_spawn_llama), and that
                # path fires on every first-run wizard cancel. Recording
                # a give-up here rendered a bogus "Yamabiko failed to
                # start" dialog and flipped the watchdog exit to 1.
                if self._stop_event.is_set():
                    _log.info(
                        "spawn aborted by stop request (%s): %s",
                        self.name, exc,
                    )
                    return
                _log.critical("failed to spawn child: %s", exc)
                # TERMINAL: spawn failed (bad exe path, model never
                # appeared, ...). Surface a user-facing error AND cascade
                # so peer supervisors + the mutex drop instead of hanging
                # forever.
                if self._on_give_up is not None:
                    self._on_give_up(
                        "Yamabiko failed to start",
                        f"The {self.name} process could not be spawned: {exc}",
                    )
                self._cascade_stop()
                return
            # Assign to job so the child dies with the watchdog parent.
            # Test-injected _FakeProcess has no OS handle; skip gracefully.
            # A job is carried ONLY on the fallback path (the watchdog's own
            # self-assign failed); on the normal path the child inherited
            # membership, so verify that rather than re-assign it -- the
            # re-assign of an existing member fails ERROR_ACCESS_DENIED and
            # logged an err=5 warning on every start.
            if self._job is not None and isinstance(proc, subprocess.Popen):
                self._job.assign(proc)
            elif self._in_job_check(proc) is False:
                _log.warning(
                    "spawned %s child (pid=%s) is not a Job Object member; "
                    "it will survive watchdog death instead of being killed "
                    "with the job. Children normally inherit the outer job "
                    "from the watchdog's self-assign.",
                    self.name, getattr(proc, "pid", "?"),
                )
            # Lifetime clock starts AFTER _spawn returns: the llama spawn
            # can block minutes-to-hours in its model-wait (first-run
            # wizard download), and counting that as child lifetime let a
            # broken-DLL insta-crash read as a healthy long run --
            # skipping the fast-crash give-up for one full extra cycle.
            start = self._clock()
            code = self._wait_or_stop(proc)
            with self._proc_lock:
                self._current_proc = None
                requested = self._requested_restart_proc is proc
                self._requested_restart_proc = None
            elapsed = self._clock() - start

            if self._stop_event.is_set():
                _log.info("stop requested; watchdog exiting")
                return

            # _wait_or_stop returns None only when IT terminated the child,
            # which only happens once stop_event is set -- caught above, so
            # code is non-None here. Defensive guard: if a future change ever
            # lets a None reach classify() (which would silently classify as
            # RESTART), treat it as a watchdog-initiated stop instead.
            if code is None:
                _log.info("child terminated by watchdog; supervisor exiting")
                return

            policy = classify(code)
            # Log the RAW OS exit code for every child exit: a user quit
            # must arrive as 43 (STOP_ALL). If finalization mangles it
            # (see main._hard_exit), this line shows the real code the
            # watchdog acted on -- the difference between "honored the
            # quit" and "restart-looped".
            _log.info("child exited: raw code=%d -> policy=%s (lifetime=%.1fs)",
                      code, policy.name, elapsed)
            if policy is ExitPolicy.STOP_SELF:
                _log.info("child exited cleanly; watchdog done")
                # TERMINAL clean exit (tray-Quit, exit 0). Cascade so the
                # llama supervisor + mutex drop immediately instead of
                # lingering ~125s. No give_up -- this is clean.
                self._cascade_stop()
                return
            if policy is ExitPolicy.STOP_ALL:
                _log.info(
                    "child exited %d (STOP_ALL); signalling peer supervisors",
                    code,
                )
                if code == EXIT_STARTUP_FAILURE and self._on_give_up is not None:
                    # The overlay server could not bind its port. A bind
                    # conflict does not self-heal in seconds, so we give up
                    # immediately (no restart) and tell the user which port
                    # to free/change -- restart churn would only delay this
                    # message. Reuses the existing give-up dialog plumbing.
                    self._on_give_up(
                        "Overlay port unavailable",
                        (
                            f"Yamabiko could not start because the overlay "
                            f"port ({self._overlay_port}) is already in use. "
                            f"Another program may be using it, or a previous "
                            f"Yamabiko instance has not fully exited. Change "
                            f"overlay.websocket_port in your settings or "
                            f"overrides, or reboot to release the port, then "
                            f"restart Yamabiko."
                        ),
                    )
                # TERMINAL: route through the same shared cascade as every
                # other terminal outcome (replaces the old bespoke
                # self._stop_event.set()).
                self._cascade_stop()
                return
            # else: RESTART -- fall through to anti-flap / backoff logic.

            # The fast-crash heuristic ("died < min_lifetime -> can't even
            # start -> config error, give up") must NOT apply to
            # EXIT_STREAM_DEATH: by construction code 44 fires only AFTER
            # capture started (the stream-death monitor + idle re-open are
            # wired post-start; idle also needs _ever_audible), so a quick 44
            # means the audio DEVICE dropped, not that the pipeline can't
            # configure. Let it fall through to anti-flap/backoff so the child
            # restarts and re-resolves; a device that's genuinely gone makes the
            # restart's start() exit 43 (the correct no-audio dialog) instead.
            # Same by-construction argument for EXIT_STAGE_ESCALATION: code 45
            # fires only AFTER the pipeline started (stages run post-start,
            # and failures must accumulate to the supervisor's threshold --
            # 10 errors in a ROLLING 60 s window, so a deterministically-
            # failing stage at ~2 s segment cadence escalates in ~20 s,
            # INSIDE the 30 s min-lifetime). A quick 45 therefore means a
            # runtime stage crash-loop, not a config error. Let it fall
            # through to anti-flap/backoff so the documented chain runs:
            # restart -> startup sanity check -> tailored exit 43 for a
            # persistent root cause, or repeated escalations exhaust the
            # restart budget into the generic "crashed N times" give-up
            # (the correct copy) -- never the misleading "failed to start /
            # configuration error" dialog on the first escalation.
            # Same by-construction argument for EXIT_CHILD_GIVEUP: code 47
            # fires only AFTER the overlay spawned and crash-looped past
            # its anti-flap cap, so a quick 47 is a runtime overlay
            # flap-out, never a pipeline config error. Exempt it too, so a
            # fast give-up RESTARTS the pipeline (fresh overlay budget)
            # instead of showing the misleading "failed to start /
            # configuration error" dialog.
            # A restart this supervisor requested (request_restart) is not a crash.
            if elapsed < self._min_lifetime_s and not requested and code not in (
                EXIT_STREAM_DEATH,
                EXIT_STAGE_ESCALATION,
                EXIT_CHILD_GIVEUP,
            ):
                # An externally-signaled death (console close / Ctrl+C) is not a
                # config problem, so it must not send the user config-hunting.
                # It is still a TERMINAL give-up either way: restarting would
                # fight whoever stopped the process.
                if is_external_stop(code):
                    _log.error(
                        "child terminated externally in %.1fs "
                        "(STATUS_CONTROL_C_EXIT); not a config error",
                        elapsed,
                    )
                    title = _EXTERNAL_STOP_TITLE
                    message = _external_stop_message(self.name)
                else:
                    _log.error(
                        "child died in %.1fs (< %.1fs min); likely a config error",
                        elapsed, self._min_lifetime_s,
                    )
                    title = "Yamabiko failed to start"
                    if self.name == "llama":
                        message = (
                            f"The translation service (llama-server) exited after only "
                            f"{elapsed:.1f}s, which usually means port 8090 in use or not "
                            f"enough GPU memory. Check llama-server.log in "
                            f"%LOCALAPPDATA%\\Yamabiko\\logs for the reason."
                        )
                    else:
                        message = (
                            f"The {self.name} process exited after only {elapsed:.1f}s, which usually "
                            f"means a configuration error. Check the main-*.log files in "
                            f"%LOCALAPPDATA%\\Yamabiko\\logs for the traceback."
                        )
                if self._on_give_up is not None:
                    self._on_give_up(title, message)
                # TERMINAL give-up: cascade so peers + mutex drop.
                self._cascade_stop()
                return

            now = self._clock()
            while self._restart_times and (now - self._restart_times[0]) > self._restart_window_s:
                self._restart_times.popleft()

            if len(self._restart_times) >= self._max_restarts:
                _log.error(
                    "child crashed %d times in the last %ds; giving up",
                    self._max_restarts, int(self._restart_window_s),
                )
                if self._on_give_up is not None:
                    if is_external_stop(code):
                        # The death that exhausted the budget came from outside
                        # Yamabiko (console close / Ctrl+C). Same reasoning as
                        # the fast-crash branch above: an external stop is not a
                        # crash, so do not tell the user Yamabiko keeps crashing
                        # and send them hunting for a traceback that isn't there.
                        # (Only the fast-crash path used to special-case this;
                        # a repeated external stop landed here with crash copy.)
                        self._on_give_up(
                            _EXTERNAL_STOP_TITLE,
                            _external_stop_message(self.name),
                        )
                    elif code == EXIT_STREAM_DEATH:
                        # The restarts that exhausted the budget were driven by
                        # the audio capture device dropping (EXIT_STREAM_DEATH),
                        # not a code crash -- give device-specific copy so the
                        # user looks at the device, not a traceback.
                        self._on_give_up(
                            "Audio device keeps dropping",
                            f"The audio capture device became unavailable "
                            f"{self._max_restarts} times in the last "
                            f"{int(self._restart_window_s / 60)} minutes, so "
                            f"Yamabiko stopped retrying. Check that your audio "
                            f"output device is connected and not being "
                            f"reconfigured, then restart Yamabiko.",
                        )
                    else:
                        # "crashed or became unresponsive": a restart can be
                        # crash-driven OR wedge-driven (HealthProbe force-
                        # restarting a hung pipeline), and a wedge leaves a
                        # watchdog WARNING, not a traceback -- so point the user
                        # at both kinds of evidence: the child's traceback lives
                        # in its own role log (main-pipeline.log), the restart
                        # reasons in main-watchdog.log (per-process log split).
                        self._on_give_up(
                            "Yamabiko keeps crashing",
                            f"The {self.name} process crashed or became unresponsive "
                            f"{self._max_restarts} times in the last "
                            f"{int(self._restart_window_s / 60)} minutes. Yamabiko has "
                            f"stopped trying. Check the main-*.log files in "
                            f"%LOCALAPPDATA%\\Yamabiko\\logs for tracebacks and "
                            f"restart reasons.",
                        )
                # TERMINAL give-up: cascade so peers + mutex drop.
                self._cascade_stop()
                return

            if self._on_restart is not None:
                # A hook failure must not end supervision.
                try:
                    self._on_restart(code)
                except Exception:
                    _log.exception("supervisor %s: restart hook failed", self.name)
            self._restart_times.append(now)
            backoff_idx = len(self._restart_times) - 1
            delay = self._backoff_s[min(backoff_idx, len(self._backoff_s) - 1)]
            _log.info("restart #%d in %.1fs", len(self._restart_times), delay)
            # The default _sleep is stop_event.wait(delay), so a
            # stop request arriving mid-backoff returns immediately and
            # the loop-head ``while not self._stop_event.is_set()`` is
            # the shutdown exit -- no respawn, no 15 s mutex-held stall.
            self._sleep(delay)


class HealthProbe:
    """Polls pipeline /livez; force-restarts a HUNG pipeline.

    Wedge detector. A pipeline whose event loop has wedged
    stays alive as a process (so exit-code classification never fires)
    but stops answering ``/livez``. This probe observes that liveness
    actively and, once the pipeline has been unresponsive for longer
    than ``fail_budget_s``, calls ``on_wedged`` -- wired to the pipeline
    supervisor's ``request_restart`` -- to force-kill the hung child so
    the supervisor's restart loop respawns it. It then re-arms and keeps
    polling so a re-wedge of the fresh process is caught again.

    ``/livez`` is DELIBERATELY readiness-independent: it excludes
    translator/asr/broadcaster reachability so a llama outage or reload
    window never trips the wedge detector. (Readiness lives on
    ``/healthz``.) This removes the feedback loop where llama's own
    outage, surfaced via ``/healthz``'s ``translator_reachable``, tripped
    llama's own kill. The probe NEVER touches ``llama_stop_event``.

    Each child is judged on its own: the probe reads the supervisor's
    ``spawn_generation`` on every poll, and a new child goes back to
    cold-boot grace instead of inheriting the previous child's "seen
    live" state and fail timer. A child holding the startup-dialog
    marker (see ``lifecycle.startup_dialog``) is waiting on the user, not
    wedged, so no timer runs while it is up.

    The main watchdog ``stop_event`` stays the primary shutdown trigger
    (SIGINT / exit 42 / exit 43); winding it down stops this probe.
    """

    def __init__(
        self,
        *,
        url: str,
        stop_event: threading.Event,
        on_wedged: Callable[[], None],
        spawn_generation: Callable[[], int],
        startup_dialog_up: Callable[[], bool],
        poll_interval_s: float = 5.0,
        fail_budget_s: float = 120.0,
        startup_budget_s: float = 300.0,
    ) -> None:
        self._url = url
        self._stop_event = stop_event
        self._on_wedged = on_wedged
        self._spawn_generation = spawn_generation
        self._startup_dialog_up = startup_dialog_up
        self._poll_interval_s = poll_interval_s
        self._fail_budget_s = fail_budget_s
        self._startup_budget_s = startup_budget_s

    def run(self) -> None:
        """Run loop -- call in a dedicated thread.

        Polls /livez (liveness, readiness-independent). Budget arms only
        AFTER the first 200, so a cold-boot pipeline never false-trips.
        On fail_budget_s of continuous liveness failure, force-restarts
        the pipeline via on_wedged, then re-arms for the fresh process
        (seen_live reset) and keeps polling so a re-wedge is caught
        again. Any respawn, probe-driven or not, resets to cold-boot
        grace. Never trips llama directly.
        """
        first_fail_at: Optional[float] = None
        seen_live = False
        # Armed only once some child has been live (or the probe has
        # restarted one). The INITIAL cold boot stays unbounded (the
        # first-run wizard runs inside the pipeline child and
        # legitimately takes as long as its downloads); a respawn after a
        # live pipeline is model loads only, so a generous time-to-first-
        # 200 budget applies -- without it, a recurring boot-time wedge
        # (the same CUDA hang recurring in build_pipeline before /livez
        # binds) left the app hung silently forever with the
        # single-instance mutex held.
        startup_deadline: Optional[float] = None
        startup_wedges = 0
        seen_generation: Optional[int] = None
        dialog_noted = False
        while not self._stop_event.is_set():
            # Snapshot BEFORE the poll: a 200 is credited only to the
            # child that was current when the poll started. A respawn
            # landing mid-poll is picked up on the next iteration.
            generation = self._spawn_generation()
            try:
                r = loopback_http.get(self._url, timeout=2.0)
                live = r.status_code == 200
            except Exception:
                live = False

            now = time.monotonic()
            if generation != seen_generation:
                if seen_generation is not None:
                    # A new child. The previous child's seen-live state
                    # would arm the fail budget for a child that is still
                    # booting, and its fail timer would count the old
                    # child's death against the new one. A respawn after a
                    # live child (or a probe restart) gets the startup
                    # budget, counted from this spawn.
                    if seen_live or startup_deadline is not None:
                        startup_deadline = now + self._startup_budget_s
                    seen_live = False
                    first_fail_at = None
                    dialog_noted = False
                    _log.info(
                        "health probe: pipeline child #%d spawned; back to "
                        "cold-boot grace", generation,
                    )
                seen_generation = generation
            if live:
                if not seen_live:
                    _log.info("health probe: first 200 OK seen; budget armed")
                    seen_live = True
                if first_fail_at is not None:
                    _log.info("health probe recovered; resetting fail timer")
                first_fail_at = None
                startup_deadline = None
                startup_wedges = 0
            elif self._startup_dialog_up():
                # Parked on a tailored startup dialog by design: the child
                # exits 43 once the user closes it. Time on screen is not a
                # wedge, so every timer holds while the dialog is up.
                if not dialog_noted:
                    _log.info(
                        "health probe: pipeline is showing a startup "
                        "dialog; liveness timers paused",
                    )
                    dialog_noted = True
                first_fail_at = None
                if startup_deadline is not None:
                    startup_deadline = now + self._startup_budget_s
            else:
                if not seen_live:
                    # Cold-boot grace -- unbounded on the initial boot,
                    # bounded by startup_deadline once a child has been
                    # live or the probe has restarted one.
                    if startup_deadline is not None and now >= startup_deadline:
                        startup_wedges += 1
                        _log.error(
                            "restarted pipeline served no 200 within %.0fs "
                            "(startup wedge #%d); force-restarting again",
                            self._startup_budget_s, startup_wedges,
                        )
                        if startup_wedges >= 2:
                            _log.error(
                                "repeated startup wedges: the pipeline is "
                                "hanging during boot (GPU/driver wedge?); "
                                "restarts will continue but a machine "
                                "reboot is likely needed"
                            )
                        self._on_wedged()
                        startup_deadline = now + self._startup_budget_s
                elif first_fail_at is None:
                    first_fail_at = now
                    _log.warning("health probe failing; fail timer started")
                elif (now - first_fail_at) >= self._fail_budget_s:
                    _log.error(
                        "pipeline unresponsive for %.1fs (budget %.1fs); "
                        "force-restarting pipeline",
                        now - first_fail_at, self._fail_budget_s,
                    )
                    self._on_wedged()
                    # Re-arm for the freshly-restarted pipeline: its
                    # cold boot must not immediately re-trip the budget.
                    # If the restarted pipeline dies/wedges before serving
                    # any 200, the supervisor's anti-flap (min_lifetime /
                    # max_restarts) catches the repeated crash, not this
                    # probe -- which stays in cold-boot grace until the
                    # next first 200.
                    first_fail_at = None
                    seen_live = False
                    startup_deadline = now + self._startup_budget_s

            self._stop_event.wait(timeout=self._poll_interval_s)


from yamabiko import __version__ as _VERSION


def _print_version_and_exit() -> None:
    """Print the short version banner and exit(0).

    Called from ``main()`` before any heavy imports or subprocess spawning,
    so ``yamabiko.exe --version`` stays fast and side-effect-free (the
    build.ps1 sanity smoke relies on this).
    """
    print(f"Yamabiko {_VERSION}")
    sys.exit(0)


_ASR_RUNTIME_SMOKE_FLAG = "--verify-asr-runtime"
_ASR_RUNTIME_OK_PREFIX = "YAMABIKO_ASR_RUNTIME_OK"


def _verify_asr_runtime_and_exit() -> None:
    """Verify the frozen ASR import surface without constructing a model.

    This is a suppressed release-smoke route, not a user-facing command. It
    proves that the artifact contains BOTH the pinned distribution metadata
    and importable faster-whisper package code, then exits before any model,
    GPU, audio device, network, or cache access.
    """
    from importlib import metadata

    from yamabiko.asr import faster_whisper_audio_compat as compat

    actual = metadata.version("faster-whisper")
    if actual != compat.SUPPORTED_FASTER_WHISPER_VERSION:
        raise RuntimeError(
            "frozen ASR runtime version mismatch: expected faster-whisper "
            f"{compat.SUPPORTED_FASTER_WHISPER_VERSION}, got {actual}"
        )

    whisper_model = compat.import_whisper_model()
    if (
        not isinstance(whisper_model, type)
        or whisper_model.__name__ != "WhisperModel"
    ):
        raise RuntimeError(
            "frozen ASR runtime compatibility seam did not return "
            "the WhisperModel class"
        )

    print(f"{_ASR_RUNTIME_OK_PREFIX} faster-whisper={actual}")
    sys.exit(0)


_SINGLE_INSTANCE_MUTEX = "Global\\Yamabiko-SingleInstance-{E4FB5AC1-7A36-4E79-8E56-5C6183F6DDAB}"
_ERROR_ALREADY_EXISTS = 183
# CreateMutexW on a Global\ name fails with ERROR_ACCESS_DENIED when the
# mutex exists but was created in another logon session (fast user
# switching, an elevated instance) whose DACL denies this one.
_ERROR_ACCESS_DENIED = 5


# Match tuples: (image_name, cmdline_needle_or_None). cmdline_needle=
# None means "any command line with this image is ours" (llama-server
# is dedicated to us); a needle enforces an extra signature check to
# distinguish our python/yamabiko children from unrelated processes
# that happen to share the image name. Module-level so the conftest
# teardown fixture can import + mirror it without drifting out of sync.
YAMABIKO_CHILD_MATCH_SPECS: tuple[tuple[str, Optional[str]], ...] = (
    # (1) llama-server: image-name match alone is sufficient once
    # path-scoped.
    ("llama-server.exe", None),
    # (2) Frozen bundle re-entry children.
    ("yamabiko.exe", "--run-main"),
    ("yamabiko.exe", "--run-overlay"),
    # (3) Dev-mode python children. Match on both python.exe and
    # pythonw.exe (windowed variant); ExecutablePath scope check at
    # call site narrows to our venv.
    ("python.exe", "yamabiko.main"),
    ("python.exe", "yamabiko.overlay_app"),
    ("pythonw.exe", "yamabiko.main"),
    ("pythonw.exe", "yamabiko.overlay_app"),
)


_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _kill_orphan_verified(pid: int, scope: str) -> bool:
    """PID-reuse-safe kill for the startup orphan sweep.

    The WMI enumeration snapshot and the kill are separated by up to
    tens of seconds (one 10 s-budget PowerShell per image name); in
    that window an enumerated orphan can exit and the OS can recycle
    its PID for an unrelated process. The old ``taskkill /F /T /PID``
    fired on the raw PID and would kill whatever now answers to it.

    Established idiom -- verify and kill through ONE pinned handle:

      1. ``OpenProcess`` -- a handle refers to the process *object*,
         not the PID, so once opened the target can no longer be
         swapped underneath us (PID recycling cannot retarget an open
         handle).
      2. ``GetExitCodeProcess`` -- skip if the process already exited
         (a dead orphan needs no kill; its PID may already be reused
         by the time we look, which is exactly why step 3 matters).
      3. ``QueryFullProcessImageNameW`` -- re-verify the executable
         still lives inside our ``scope`` (bundle dir / repo root).
         A recycled PID's new owner fails this check and is spared.
      4. ``TerminateProcess`` on the SAME handle -- kills exactly the
         object we just verified.

    Per-image enumeration covers the old ``/T`` tree semantics: every
    yamabiko child image (llama-server, pipeline, overlay) is matched
    and killed individually, so no kill-the-tree call is needed.

    Returns True iff the verified process was terminated. Any
    verification failure (vanished, unqueryable, out of scope) skips
    the kill and returns False. No-op False on non-Windows.
    """
    if not sys.platform.startswith("win"):
        return False
    handle = _OpenProcess(
        _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        pid,
    )
    if not handle:
        _log.info(
            "orphan pid=%d gone before kill (OpenProcess err=%d); skipping",
            pid, ctypes.get_last_error(),
        )
        return False
    try:
        exit_code = wintypes.DWORD(0)
        if not _GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            _log.info(
                "orphan pid=%d exit-code probe failed (err=%d); skipping",
                pid, ctypes.get_last_error(),
            )
            return False
        if exit_code.value != _STILL_ACTIVE:
            _log.info(
                "orphan pid=%d already exited (code=%d); skipping",
                pid, exit_code.value,
            )
            return False
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if not _QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            _log.info(
                "orphan pid=%d image-path re-check failed (err=%d); skipping",
                pid, ctypes.get_last_error(),
            )
            return False
        exe_now = buf.value.replace("\\", "/").lower()
        if scope not in exe_now:
            _log.warning(
                "pid %d now maps to %s -- outside our tree; PID likely "
                "recycled since enumeration. NOT killing it.",
                pid, buf.value,
            )
            return False
        if not _TerminateProcess(handle, 1):
            _log.warning(
                "TerminateProcess(pid=%d) failed: err=%d",
                pid, ctypes.get_last_error(),
            )
            return False
        return True
    finally:
        _CloseHandle(handle)


def _sweep_orphaned_children() -> int:
    """Kill any yamabiko-scoped orphan child process from a prior run.

    Runs once at watchdog startup, AFTER the single-instance mutex is
    acquired (running it before the mutex meant a rejected
    second launch would taskkill the FIRST instance's children). Handles
    the orphan case: prior yamabiko.exe was force-killed (taskkill /F,
    power loss, crash without Job Object), leaving its children alive --
    that crashed instance has already released its mutex, so the new
    instance wins it and is free to sweep. Without the sweep, the new yamabiko
    would collide with the orphans (second llama-server fails to bind
    :8090; zombie overlay keeps a frameless window on top; old
    pipeline keeps its WebSocket; user accumulates VRAM-eating
    zombies across runs).

    Match set (all three scoped by project / bundle path so unrelated
    processes are left untouched):

    1. ``llama-server.exe`` whose ``ExecutablePath`` is inside our
       tree (historical sweep target; covers the ~4 GB VRAM hog).
    2. Frozen ``yamabiko.exe --run-main`` / ``--run-overlay`` children
       whose ``ExecutablePath`` is inside our tree (pipeline + overlay
       re-entry children in a bundled install).
    3. Dev-mode ``python.exe -m yamabiko.main`` /
       ``-m yamabiko.overlay_app`` children whose ``ExecutablePath``
       (the venv interpreter) is inside our tree, matched by the
       ``yamabiko.main`` / ``yamabiko.overlay_app`` substring in the
       ``CommandLine`` (a signature no non-yamabiko process would
       legitimately produce).

    The current process is always excluded (defensive): a watchdog
    that just executed ``import yamabiko.main`` in-process must never
    kill itself.

    Returns the number of zombies killed (0 in the normal case).
    No-op on non-Windows.

    The Job Object with ``KILL_ON_JOB_CLOSE`` handles most cleanup in
    practice; this sweep is belt-and-suspenders for paths that bypass
    it (tests spawning ``Popen`` directly, crashes during startup before
    Job assignment, external ``taskkill`` on the parent only).
    """
    if not sys.platform.startswith("win"):
        return 0
    # Scope: bundle dir in frozen mode, repo root in dev mode. Both
    # paths end up as substrings of a child's ExecutablePath when it
    # was launched from inside our tree (llama-server.exe lives in
    # the bundle; yamabiko.exe re-entry children inherit
    # ExecutablePath = yamabiko.exe path; dev-mode python children
    # have ExecutablePath = venv's python.exe).
    if getattr(sys, "frozen", False):
        scope = str(Path(sys.executable).parent).replace("\\", "/").lower()
    else:
        # Dev mode: repo root (this file is yamabiko/watchdog.py;
        # parent.parent is the repo root).
        scope = str(Path(__file__).resolve().parent.parent).replace("\\", "/").lower()

    match_specs = YAMABIKO_CHILD_MATCH_SPECS

    self_pid = os.getpid()
    killed = 0
    # Enumerate each image name once, then filter in Python. One
    # PowerShell per image keeps the command short; CommandLine is
    # requested alongside so the needle check can run locally.
    enumerated: dict[str, list[tuple[int, str, str]]] = {}
    image_names = {name for name, _ in match_specs}
    for image_name in image_names:
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    # UTF-8 output: the default OEM code page mangles
                    # non-ASCII install paths (a Japanese user folder),
                    # which would make the scope check below miss them.
                    f"[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
                    f"Get-CimInstance Win32_Process "
                    f"-Filter \"Name='{image_name}'\" | "
                    f"Select-Object ProcessId, ExecutablePath, CommandLine | "
                    f"ForEach-Object {{ "
                    f"\"$($_.ProcessId)|$($_.ExecutablePath)|$($_.CommandLine)\" "
                    f"}}"
                ],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception as exc:
            _log.debug(
                "startup zombie sweep: enumeration failed for %s: %s",
                image_name, exc,
            )
            enumerated[image_name] = []
            continue
        if result.returncode != 0:
            enumerated[image_name] = []
            continue

        rows: list[tuple[int, str, str]] = []
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
            rows.append((pid, exe, cmdline))
        enumerated[image_name] = rows

    seen_pids: set[int] = set()
    for image_name, needle in match_specs:
        for pid, exe, cmdline in enumerated.get(image_name, ()):
            if pid == self_pid or pid in seen_pids:
                continue
            if scope not in exe.replace("\\", "/").lower():
                continue
            if needle is not None and needle not in cmdline:
                continue
            seen_pids.add(pid)
            _log.warning(
                "killing orphaned yamabiko child pid=%d image=%s path=%s "
                "cmdline=%s (previous run didn't clean up)",
                pid, image_name, exe, cmdline,
            )
            # Handle-based verify-then-kill instead of a raw
            # ``taskkill /PID`` -- the enumeration snapshot above can be
            # tens of seconds stale, and a recycled PID must never
            # receive the kill. _kill_orphan_verified re-checks the
            # image path on a pinned handle and terminates through that
            # same handle.
            try:
                if _kill_orphan_verified(pid, scope):
                    killed += 1
            except Exception as exc:
                _log.warning("orphan kill pid=%d failed: %s", pid, exc)
    return killed


# Job Object primitives for the watchdog now live in
# `yamabiko/lifecycle/jobs.py` (imported above as `JobObject`,
# `create_outer_job`, `is_in_job`). The watchdog uses a single outer
# job whose KILL_ON_JOB_CLOSE flag guarantees children die when the
# watchdog's process exits -- crash, taskkill, or normal.


def _acquire_single_instance_mutex():
    """Create a named Windows mutex so only one top-level yamabiko.exe
    runs at a time. Returns the handle on success, or None if another
    instance already holds the mutex.

    Without this check, a user who double-clicked the Start
    Menu shortcut launched a second instance whose pipeline child
    would crash on port 8765 bind (already held by the first instance),
    triggering watchdog anti-flap and a "Yamabiko keeps crashing"
    dialog. The first instance was fine; the second was just wrongly
    launched.

    Returns None (and the caller should show a friendly dialog + exit)
    when ``GetLastError`` returns ``ERROR_ALREADY_EXISTS (183)`` after
    CreateMutexW.

    Non-Windows returns None and doesn't enforce single-instance; the
    production target is Windows-only, but this keeps
    dev-mode pytest runs on non-Windows hosts working.
    """
    if not sys.platform.startswith("win"):
        # Return a placeholder object so the caller can just check for
        # None-vs-not-None. Non-Windows: enforcement is a no-op.
        return object()
    try:
        handle = _CreateMutexW(None, False, _SINGLE_INSTANCE_MUTEX)
        # Read the error ctypes captured atomically AT the
        # CreateMutexW call (use_last_error=True on _k32), before any
        # other FFI call can overwrite the TLS slot. The old idiom --
        # a separate live GetLastError() FFI call several Python
        # statements later -- could be clobbered by ctypes/CPython
        # internals, losing ERROR_ALREADY_EXISTS and letting a second
        # instance through the gate.
        last = ctypes.get_last_error()
        if not handle:
            if last == _ERROR_ACCESS_DENIED:
                # The mutex exists in another session: an instance is
                # already running there, so this launch must not start.
                _log.warning(
                    "single-instance mutex owned by another session (err=%d)",
                    last,
                )
                return None
            # CreateMutex itself failed; fall back to not enforcing.
            _log.warning(
                "CreateMutexW failed (err=%d); skipping single-instance check",
                last,
            )
            return object()
        if last == _ERROR_ALREADY_EXISTS:
            # Another instance holds it. Close our (non-owning) handle
            # and return None to signal "already running".
            _CloseHandle(handle)
            return None
        return handle  # caller keeps this alive for process lifetime
    except Exception as exc:
        _log.warning("single-instance mutex check raised: %s; skipping", exc)
        return object()


def _release_single_instance_mutex(handle) -> None:
    """Close the single-instance mutex handle and drop the module keepalive.

    Called before the give-up dialogs render: the supervisors have stopped,
    so a user who relaunches while a dialog is still open must not be told
    Yamabiko is already running. Only real handles (nonzero ints) are
    closed; None and the ``object()`` no-enforcement sentinel are skipped.
    """
    globals()["_single_instance_handle_keepalive"] = None
    if isinstance(handle, int) and handle and _CloseHandle is not None:
        try:
            _CloseHandle(handle)
        except Exception as exc:
            _log.warning("single-instance mutex close raised: %s", exc)


_HELP_TEXT = """yamabiko - Japanese audio -> subtitle pipeline

Usage:
    yamabiko                     Normal launch (watchdog supervises
                                 pipeline + overlay + llama-server).
    yamabiko --version           Print version and exit.
    yamabiko --help              Show this message and exit.
    yamabiko --max-segments N    Dev: exit the pipeline after N
                                 translated segments (0 = run forever).
                                 Forwarded to yamabiko.main.
    yamabiko --config PATH       Dev: load config from PATH instead of
                                 the layered default_config.yaml +
                                 user_overrides.yaml. Forwarded to
                                 yamabiko.main.

Internal dispatch (not typically user-invoked):
    yamabiko --run-main          Drive the pipeline in this process
                                 (watchdog subprocess re-entry).
    yamabiko --run-overlay       Drive the Qt desktop overlay in this
                                 process (watchdog subprocess re-entry).

See docs/RELEASE.md for the release workflow and the main-*.log files
in %LOCALAPPDATA%\\Yamabiko\\logs for runtime logs (one per process:
watchdog, pipeline, overlay).
"""


@dataclass
class WatchdogApp:
    """Wired-but-not-running watchdog supervisors. Composition Root output.

    Holds the Watchdog instances (pipeline + llama; the overlay is owned by
    the pipeline via PipelineSupervisor, not supervised here), the shared
    stop_event, the llama-scoped stop_event (set only via the ``_mirror``
    thread; see below), and the single-instance mutex handle (must be
    retained so GC doesn't release it).
    run_watchdog() drives the thread pool + signal handlers + joins.

    The two events are deliberately asymmetric:

      * ``stop_event`` is the primary shutdown signal. Set by SIGINT /
        SIGTERM handlers, by an exit-policy ``STOP_ALL`` classify()
        result, or by any TERMINAL supervisor outcome via
        ``_cascade_stop`` (clean quit, give-up, spawn-failure). Once set,
        every supervisor (pipeline, llama, overlay-via-pipeline) begins
        its graceful finally.

      * ``llama_stop_event`` is the llama-scoped shutdown signal. As of
        It is set ONLY by the ``_mirror`` thread (stop_event ->
        llama_stop_event), never by an independent probe. The probes no
        longer tear llama down on their own: HealthProbe restarts a hung
        pipeline instead, and the idle probe was removed. The two events
        are kept separate so the pipeline supervisor (stop_event) and the
        llama supervisor (llama_stop_event) remain independently
        restartable; coordinated shutdown flows stop_event -> _mirror ->
        llama_stop_event.

    Propagation is one-way: ``stop_event`` -> ``llama_stop_event`` via
    the ``_mirror`` thread in :func:`run_watchdog`. Never the reverse.
    A llama-only restart MUST NOT cascade into a full watchdog shutdown
    -- that's the whole point of keeping the two events separate. If a
    future change feels the urge to ``_mirror`` both ways, re-read this
    docstring: bidirectional coupling would make a routine llama-server
    restart (a llama crash the llama supervisor handles on its own,
    without the pipeline) cascade into a full watchdog shutdown.
    """
    watchdogs: list["Watchdog"]
    stop_event: threading.Event
    llama_stop_event: threading.Event
    pipeline_writer: ShutdownEventWriter
    single_instance_handle: object
    # The explicit ``--config`` path (or None for the layered load),
    # parsed from argv in build_watchdog. run_watchdog re-reads it for the
    # health probe so the probe resolves the SAME effective config the
    # llama plan did -- an explicit config with a different
    # overlay.websocket_port must move the probe URL, not leave it polling
    # the layered default forever.
    config_path: Optional[Path] = None
    # Worker threads NEVER touch Qt. A TERMINAL give-up
    # records a StartupError here under the lock; run_watchdog's MAIN
    # thread drains + renders them after join.
    pending_errors: list = field(default_factory=list)
    pending_errors_lock: threading.Lock = field(default_factory=threading.Lock)


def build_watchdog(argv: list[str] | None = None) -> WatchdogApp:
    """Build the three supervisors + shared stop_event + mutex handle.

    Does NOT: install signal handlers, start any threads, or block on
    join. Those are run_watchdog()'s job.

    Argv IS parsed for ``--config``: the flag drives the watchdog's own
    llama endpoint plan and (via WatchdogApp.config_path) the health
    probe, so the watchdog resolves the same effective config it forwards
    to the pipeline child. The early-exit flags --version / --help /
    --verify-asr-runtime / --run-main / --run-overlay are handled by
    main() before build_watchdog is called, and they short-circuit.
    """
    explicit_path = _parse_config_path(argv)

    # Merged config loads FIRST: logging.level must be known at
    # setup_main_log time (before this fix the call site hardcoded
    # level="INFO", leaving the documented config key dead for the
    # watchdog role -- hard rule #4's producer/consumer class). The
    # same load feeds grace_s and the llama endpoint plan below. Under
    # --config the layered user_overrides is skipped (dev/repro escape
    # hatch); with no flag the standard layered load runs -- and the
    # no-flag call stays load_full_config() so it matches the pipeline's
    # own resolution exactly. A corrupt/unreadable config (including a
    # bad --config path) falls back to {} + defaults here; it must never
    # block the watchdog's startup -- the pipeline child, which gets the
    # same --config forwarded, surfaces the tailored fatal-config dialog.
    _deferred_level_warn: "str | None" = None
    try:
        from yamabiko.config import (
            consume_level_warning as _consume_level_warning,
            load_full_config as _load_cfg,
        )
        if explicit_path is not None:
            _cfg = _load_cfg(explicit_path=explicit_path)
        else:
            _cfg = _load_cfg()
        # load_full_config normalizes logging.level IN the merged dict
        # and STASHES the junk-value warning instead of emitting it
        # (this load runs before setup_main_log below -- an in-load
        # emission dies on the frozen bundle's discarded stderr).
        # Consume it here; the emission is sequenced after the file
        # handler attach, at the _level_warn seam below.
        _deferred_level_warn = _consume_level_warning()
    except ConfigValidationError:
        raise
    except Exception as exc:
        _log.warning(
            "build_watchdog: config load failed (%s); using defaults "
            "(log level INFO, default grace_s, bundled llama)", exc,
        )
        _cfg = {}

    from yamabiko import log_config
    # Per-process log file: the pipeline child and overlay child
    # write main-pipeline.log / main-overlay.log; sharing one file
    # across processes breaks RotatingFileHandler rotation on Windows.
    #
    # Deliberately BEFORE the single-instance mutex check: the
    # "already running" dialog used to fire pre-logging, making it the
    # one user-visible dialog with zero trace in any log. The cost is a
    # transient second appender on main-watchdog.log for the seconds a
    # rejected double-launch lives -- appends interleave safely
    # (Python's open() shares on Windows); only a rotation rename could
    # collide, and the loser's doRollover failure is swallowed by
    # logging.Handler.handleError. That beats an evidence-free dialog.
    # logging.level flows through _normalize_log_level, NOT raw into
    # setLevel: logger.setLevel('debug') / setLevel('verbose') raise
    # ValueError, and an unvalidated config value aborted the watchdog
    # at the very call site whose comment promised bad config would
    # never block startup (crash-looping the overlay with it). The
    # warning (if any) is emitted AFTER setup_main_log attaches the
    # file handler -- earlier and the only evidence dies on the frozen
    # bundle's discarded stderr. On the real path load_full_config has
    # already normalized the merged value, so THIS call returns no
    # warning; the junk-value text arrives via the deferred stash
    # consumed above. The direct call stays as belt-and-suspenders for
    # the fallback (_cfg = {}) path and keeps the level derivation on
    # the single-source normalizer.
    _level_name, _level_warn = _normalize_log_level(
        (_cfg.get("logging") or {}).get("level"),
    )
    if _level_warn is None:
        _level_warn = _deferred_level_warn
    log_config.setup_main_log(
        logging.getLogger("yamabiko"),
        level=_level_name,
        filename="main-watchdog.log",
    )
    _log.info("watchdog starting (pid=%d)", os.getpid())
    if _level_warn:
        _log.warning("build_watchdog: %s", _level_warn)
    # Crash-evidence primitives (faulthandler -> crash-watchdog.log,
    # sys/threading excepthooks -> CRITICAL records) + the identity
    # banner bug_report.yml points reporters at. Both never raise;
    # both belong immediately after log setup so even a startup that
    # dies at the mutex check below leaves a fingerprint.
    log_config.install_crash_evidence("watchdog")
    log_config.log_startup_banner(_log, "watchdog")

    # Single-instance enforcement. Only applies to the
    # top-level watchdog invocation -- child processes use their own
    # argv flags (--run-main / --run-overlay) and short-circuit before
    # this point, so re-entries don't trip the check.
    handle = _acquire_single_instance_mutex()
    if handle is None:
        # StartupError + show_startup_error are module-level imports.
        # Re-importing them as locals here would rebind them as
        # build_watchdog locals, which Python then treats as free
        # variables in the nested _record_error closure below -- and that
        # closure would hit a NameError because the local is only assigned
        # on this branch. So rely on the module globals.
        err = StartupError(
            title="Yamabiko is already running",
            message="Only one copy of Yamabiko can run at a time.",
            suggested_fix=(
                "Look for its tray icon or its setup window. To restart it, "
                "close Yamabiko setup, or right-click its tray icon and "
                "choose Quit Yamabiko, then start Yamabiko again. If you "
                "can't find either, end Yamabiko in Task Manager first."
            ),
            details=(
                f"Named mutex {_SINGLE_INSTANCE_MUTEX} is already held "
                f"by another process."
            ),
        )
        # Verbatim dialog copy at ERROR (the whole point of moving log
        # setup above this check): a bug report quoting the dialog can
        # now be matched to a main-watchdog.log line.
        _log.error("startup dialog: %s -- %s", err.title, err.message)
        show_startup_error(err)
        sys.exit(0)

    # Terminal supervisor outcomes funnel here. Worker threads
    # NEVER touch Qt; they record a StartupError to a thread-safe
    # list that run_watchdog's MAIN thread drains + renders after join.
    pending_errors: list = []
    pending_errors_lock = threading.Lock()

    def _record_error(label: str):
        def _on_give_up(title: str, detail: str) -> None:
            err = StartupError(
                title=f"{label}: {title}",
                message=detail.split("\n")[0] if "\n" in detail else detail,
                suggested_fix=(
                    "Restart Yamabiko. If the problem persists, share the "
                    "main-*.log files from %LOCALAPPDATA%\\Yamabiko\\logs."
                ),
                details=detail,
            )
            # Verbatim dialog copy at ERROR, at RECORD time (worker
            # thread), not only at render time: if the process dies
            # before run_watchdog's main-thread drain renders the
            # queue, this line is the only surviving copy of what the
            # user was told.
            _log.error("give-up dialog: %s -- %s", err.title, err.details)
            with pending_errors_lock:
                pending_errors.append(err)
        return _on_give_up

    # Shared Job Object: every subprocess we spawn (pipeline, overlay,
    # llama-server) gets assigned to this job. When the watchdog exits
    # -- crash, taskkill, or normal -- Windows closes the job and
    # force-kills every member process. No more orphaned llama-servers
    # eating VRAM after a crashed test run or a force-killed parent.
    #
    # Startup probe: log if we're already inside an external job (NSSM,
    # Task Scheduler, debugger). Nested jobs compose correctly per spec
    # research §2, but we want the context if anything goes wrong.
    if is_in_job():
        _log.info("watchdog starting inside an existing Job Object (external owner)")
    job = create_outer_job("yamabiko-outer")
    # Assign self -> any child we spawn auto-joins via inheritance.
    self_assigned = job.assign(None)
    if not self_assigned:
        _log.warning("failed to assign watchdog to outer job; children won't auto-kill")
    # Per-child assign is the FALLBACK for a failed self-assign only.
    # With self-assign in place every spawn already inherits membership,
    # and re-assigning an existing member fails ERROR_ACCESS_DENIED --
    # a spurious err=5 warning on every pipeline/llama start that reads
    # like a broken kill cascade (same rationale as the overlay's
    # atomic-spawn path in pipeline_supervisor).
    child_job = None if self_assigned else job

    # Shared stop event so one signal drops ALL supervisors: the SIGINT/
    # SIGTERM handlers and any terminal supervisor outcome (via _stop_all)
    # set it, and the _mirror thread forwards it to llama_stop_event. (The
    # llama spawn fn now polls llama_stop_event for its own model-wait
    # cancel, not this event.)
    stop_event = threading.Event()

    def _stop_all() -> None:
        # Coordinated shutdown in _on_signal's Tier A order: graceful
        # ping to the pipeline child FIRST, so it starts its finally
        # (glossary save, uvicorn drain) inside the grace window, THEN
        # the MAIN stop_event. _mirror cascades to llama_stop_event; both
        # supervisors' _wait_or_stop see their event and run the
        # grace->terminate ladder; run_watchdog's join returns and the
        # main thread drains pending_errors. pipeline_writer is bound
        # below, before any supervisor can call this.
        try:
            pipeline_writer.signal()
        except Exception as exc:
            _log.warning("pipeline_writer.signal() in stop_all raised: %s", exc)
        stop_event.set()

    # Separate event for the llama peer -- set ONLY by the stop_event ->
    # llama_stop_event mirror thread in run_watchdog (so SIGINT / STOP_ALL
    # / any terminal supervisor give-up that sets stop_event also stops
    # llama). No probe trips it directly.
    llama_stop_event = threading.Event()

    # Tier A named-event shutdown writer for the pipeline child. The
    # name is scoped to THIS process's pid so parallel test runs or a
    # double-launched watchdog don't collide. run_watchdog.signal() is
    # the graceful ping; pipeline's ShutdownEventListener opens the
    # same event by name (via YAMABIKO_SHUTDOWN_EVENT) and wakes its
    # asyncio app.done on SetEvent.
    pipeline_writer = ShutdownEventWriter.create(
        "pipeline", parent_pid=os.getpid(),
    )

    # Problem carry (plan Decision 18). A file left on disk belongs to a run
    # that never respawned its pipeline; a fresh launch starts without it.
    problems.take_carry()
    carry_relay = _CarryRelay()

    def _pipeline_spawn() -> subprocess.Popen:
        """Inject YAMABIKO_SHUTDOWN_EVENT_PIPELINE into the pipeline
        child's env so its ShutdownEventListener can OpenEvent() the
        handle the watchdog created above. Without this wrapper, the
        child would inherit a bare environment and fall back to
        SIGINT-only shutdown, defeating the Tier A graceful chain.

        Also names the startup-dialog marker the child holds while a
        tailored startup dialog is up, so the health probe (run_watchdog)
        does not kill a child that is waiting on the user.

        Also hands the next pipeline the problems the previous one
        restarted on (YAMABIKO_CARRIED_PROBLEMS) and the shortcuts it
        announced as taken (YAMABIKO_CARRIED_HOTKEYS).
        """
        env = os.environ.copy()
        env[ENV_VAR_BY_ROLE["pipeline"]] = pipeline_writer.name
        env[startup_dialog.ENV_VAR] = startup_dialog.name_for(os.getpid())
        carry_relay.apply(env)
        return _real_spawn(env=env)

    # Grace window for the pipeline's graceful finally after the Tier A
    # named-event ping fires. Wired through from
    # lifecycle.grace_s.pipeline in default_config.yaml so ops can tune
    # without a code change (hard rule #4: every config key has a
    # verified consumer; this is the consumer). ``_cfg`` is the merged
    # config loaded once at the top of this function.
    grace_cfg = (_cfg.get("lifecycle") or {}).get("grace_s") or {}
    pipeline_grace_s = float(grace_cfg.get("pipeline", 10.0))
    # The overlay port the pipeline's uvicorn binds. Threaded into the
    # pipeline Watchdog so an EXIT_STARTUP_FAILURE (46, port-bind failure)
    # give-up dialog can name the exact port the user must free or change.
    overlay_port = int((_cfg.get("overlay") or {}).get("websocket_port", 8765))

    wd_pipeline = Watchdog(
        name="pipeline",
        spawn=_pipeline_spawn,
        stop_event=stop_event,
        on_give_up=_record_error("Pipeline"),
        on_stop_all=_stop_all,
        job=child_job,
        grace_s=pipeline_grace_s,
        overlay_port=overlay_port,
        on_restart=carry_relay.on_pipeline_restart,
    )
    # Overlay is not a watchdog sibling: the process-lifecycle overhaul
    # makes the pipeline own the overlay as a child via PipelineSupervisor,
    # so the watchdog supervises only the pipeline and llama-server here.
    #
    # The llama supervisor exists only when translation.endpoint
    # points at OUR loopback server. An external endpoint means the
    # pipeline (which honours the endpoint -- main.py's translator)
    # never calls the bundled llama-server; spawning it anyway would
    # idle-load ~4 GB VRAM. A loopback endpoint on a non-default port
    # moves the bundled server to that port so producer and consumer
    # agree.
    spawn_llama, llama_port = _llama_endpoint_plan(_cfg)
    watchdogs = [wd_pipeline]
    if spawn_llama:
        wd_llama = Watchdog(
            name="llama",
            # Spawn callable polls the llama-scoped event for its first-run
            # model-wait cancel: the _mirror guarantees stop_event ->
            # llama_stop_event, so the llama event covers both shutdown
            # sources, and a llama_stop trip during the model-wait cancels
            # the supervisor instead of hanging.
            spawn=_make_real_spawn_llama(
                llama_stop_event, port=llama_port,
                config_path=explicit_path,
            ),
            stop_event=llama_stop_event,
            on_give_up=_record_error("Translation service"),
            on_stop_all=_stop_all,
            job=child_job,
            # llama-server has no graceful-shutdown handshake and keeps no
            # state worth flushing; a short grace releases its VRAM and
            # port quickly so a relaunch does not race it.
            grace_s=0.5,
        )
        watchdogs.append(wd_llama)
        carry_relay.llama = wd_llama

    return WatchdogApp(
        watchdogs=watchdogs,
        stop_event=stop_event,
        llama_stop_event=llama_stop_event,
        pipeline_writer=pipeline_writer,
        single_instance_handle=handle,
        config_path=explicit_path,
        pending_errors=pending_errors,
        pending_errors_lock=pending_errors_lock,
    )


def run_watchdog(app: WatchdogApp) -> int:
    """Install signal handlers, start one thread per supervisor,
    block on join. Returns 0 on clean shutdown, nonzero when the
    supervisors died silently (layer-3 self-crash evidence below).

    On ANY exit path (SIGINT-driven stop_event, supervisor natural
    exit, anti-flap give-up) the cleanup in the finally block signals
    ``app.pipeline_writer`` so the pipeline child gets the Tier A
    graceful-shutdown ping; its listener wakes app.done, its finally
    runs (glossary save, uvicorn drain, ...), and it exits 0.
    Watchdog.run classify(0) == STOP_SELF returns cleanly.
    """
    def _on_signal(signum, frame):
        _log.info("received signal %d; initiating clean shutdown", signum)
        # Tier A: ping the pipeline's named shutdown event BEFORE
        # tripping stop_event. Watchdog._wait_or_stop uses stop_event
        # as its "time to die" signal; if we set stop_event first,
        # the grace window in _wait_or_stop opens against a child
        # that has not yet received the named-event ping -- likely
        # hitting terminate() before its asyncio listener wakes
        # app.done and the finally cleanup runs. Signalling first
        # means the child is already unwinding when the grace window
        # starts, so a well-behaved pipeline exits cleanly inside
        # the window and _wait_or_stop returns its exit code via
        # proc.wait(), never reaching terminate().
        try:
            app.pipeline_writer.signal()
        except Exception as exc:
            _log.warning(
                "pipeline_writer.signal() in signal handler raised: %s", exc,
            )
        app.stop_event.set()

    # Install process-control-signal handlers. Per-signal try/except so a
    # failure on one doesn't silently skip the others (SIGBREAK on a
    # bundle without a console, for instance, can raise OSError and we
    # want to know about it via the WARN rather than hide behind a
    # generic DEBUG).
    for signum, label in _installable_signals():
        try:
            signal.signal(signum, _on_signal)
            _log.info("watchdog: installed %s handler", label)
        except (ValueError, OSError) as exc:
            _log.warning(
                "watchdog: could not install %s handler (%s); "
                "graceful shutdown via %s will fall back to OS default",
                label, exc, label,
            )

    # Self-directed shutdown event. The signal handlers above cover the
    # terminal / Ctrl+C path. CTRL_BREAK_EVENT does NOT reach a PyInstaller
    # console=False GUI-subsystem bundle (Python's signal module wires
    # SIGBREAK via SetConsoleCtrlHandler, which requires an attached
    # console -- the shipping Start-Menu launch path doesn't have one).
    # So we ALSO expose a named Win32 Event any other process can open
    # by (deterministic) name and fire SetEvent to initiate Tier A
    # graceful shutdown. Covers:
    #   * The frozen-E2E graceful-shutdown test, which can't rely on
    #     CTRL_BREAK against the bundle we actually ship.
    #   * Tray-Quit cascade: the overlay's "Quit" menu item can
    #     open ``Local\yamabiko.shutdown.watchdog.{ppid}`` and fire it
    #     to shut the whole pipeline+overlay+llama stack down cleanly.
    # Uses the same primitive as the pipeline/overlay shutdown events
    # for architectural consistency (one mechanism, three tiers).
    watchdog_self_writer = ShutdownEventWriter.create(
        "watchdog", parent_pid=os.getpid(),
    )
    # Set in run_watchdog's finally BEFORE the reaping SetEvent, so the
    # watcher can tell "run_watchdog is unwinding and waking me to exit"
    # from a genuine external shutdown request and not re-run the
    # _on_signal sequence.
    runner_done = threading.Event()

    def _watch_self_shutdown() -> None:
        if watchdog_self_writer.handle is None:
            # Non-Windows, or CreateEventW failed (logged at WARN).
            # The signal-handler path above is still wired -- for the
            # cases CTRL_BREAK can reach us, graceful shutdown still
            # works. No console-less path, but that's not a regression.
            return
        try:
            # Bind from the use_last_error WinDLL like every other
            # kernel32 call in this module -- ctypes.windll bindings do
            # not capture GetLastError atomically and can be clobbered.
            WaitForSingleObject = _k32.WaitForSingleObject
            WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            WaitForSingleObject.restype = wintypes.DWORD
            INFINITE = 0xFFFFFFFF
            WAIT_OBJECT_0 = 0x0
            rc = WaitForSingleObject(watchdog_self_writer.handle, INFINITE)
            if runner_done.is_set():
                # Reap ping from run_watchdog's finally; just exit.
                return
            if rc == WAIT_OBJECT_0:
                _log.info(
                    "watchdog: self-shutdown event fired; "
                    "initiating Tier A graceful shutdown",
                )
                # Reuse the same sequence as _on_signal: pipeline-writer
                # signal BEFORE stop_event.set, so the Tier A grace
                # ordering in Watchdog._wait_or_stop is preserved.
                _on_signal(0, None)
            else:
                _log.warning(
                    "watchdog: self-shutdown WaitForSingleObject returned %d", rc,
                )
        except Exception:
            _log.exception("watchdog: self-shutdown watcher crashed")

    self_watch_thread = threading.Thread(
        target=_watch_self_shutdown,
        daemon=True,
        name="watchdog-self-shutdown",
    )
    self_watch_thread.start()

    # Health probe: polls pipeline /livez, restarts the hung pipeline on
    # budget (wedge detector). Daemon thread so it
    # doesn't block the process from exiting once the supervisors are
    # done. The supervisors exiting is our cue to wind down the probe via
    # app.stop_event (see _mirror below).
    #
    # URL + budget + poll interval all derive from the merged config so
    # a user who moves the overlay websocket_port or wants a longer
    # fail budget gets the same numbers in the probe that the pipeline's
    # uvicorn serves on. Honour app.config_path so a --config invocation
    # resolves the SAME effective config the llama plan used in
    # build_watchdog: without this the probe read the layered config and
    # a --config with a different overlay.websocket_port polled the wrong
    # port forever, force-restarting a healthy pipeline. The no-flag path
    # stays load_full_config() so the layered resolution is unchanged.
    try:
        from yamabiko.config import load_full_config
        if app.config_path is not None:
            cfg = load_full_config(explicit_path=app.config_path)
        else:
            cfg = load_full_config()
    except Exception as exc:
        _log.warning("health probe: config load failed (%s); using defaults", exc)
        cfg = {}
    port = int((cfg.get("overlay") or {}).get("websocket_port", 8765))
    health_cfg = (cfg.get("lifecycle") or {}).get("health") or {}
    wd_pipeline = next((w for w in app.watchdogs if w.name == "pipeline"), None)
    if wd_pipeline is None:
        raise RuntimeError("run_watchdog: no 'pipeline' supervisor in app.watchdogs")
    probe = HealthProbe(
        url=f"http://127.0.0.1:{port}/livez",
        stop_event=app.stop_event,
        on_wedged=wd_pipeline.request_restart,
        # Each respawned child gets its own cold-boot grace, and a child
        # holding the startup-dialog marker (named in _pipeline_spawn,
        # same pid) is waiting on the user rather than wedged.
        spawn_generation=wd_pipeline.spawn_generation,
        startup_dialog_up=functools.partial(
            startup_dialog.is_held, startup_dialog.name_for(os.getpid()),
        ),
        poll_interval_s=float(health_cfg.get("poll_interval_s", 5.0)),
        fail_budget_s=float(health_cfg.get("fail_budget_s", 120.0)),
    )
    threading.Thread(
        target=probe.run, daemon=True, name="health-probe",
    ).start()

    # Forward main stop_event -> llama_stop_event so SIGINT / STOP_ALL
    # exit codes also stop llama. The pipeline supervisor listens on
    # stop_event; the llama supervisor listens on llama_stop_event.
    # Without this mirror, Ctrl+C would stop pipeline + probe but leave
    # llama running until its child exited naturally.
    #
    # DELIBERATELY ONE-WAY: stop_event -> llama_stop_event only. The
    # reverse direction is wrong on purpose -- the llama supervisor must
    # be able to restart llama-server on its own (a llama crash) WITHOUT
    # cascading into a full watchdog shutdown. See WatchdogApp's docstring
    # for the full rationale.
    def _mirror() -> None:
        app.stop_event.wait()
        app.llama_stop_event.set()
    threading.Thread(
        target=_mirror, daemon=True, name="stop-mirror",
    ).start()

    try:
        # Run supervisors concurrently. Each manages its own child's
        # restart loop independently -- a llama-server crash doesn't
        # take down the pipeline. Main thread waits for all supervisors
        # to exit.
        threads = [
            threading.Thread(
                target=wd.run, name=f"{wd.name}-watchdog", daemon=False,
            )
            for wd in app.watchdogs
        ]
        for t in threads:
            t.start()
        # Layer-3 self-crash evidence (dead-thread detection), as a
        # MONITOR LOOP rather than a blocking join over all threads.
        # Every LEGITIMATE terminal path in Watchdog.run either sets
        # stop_event (clean exit / STOP_ALL / any _cascade_stop) or
        # records a pending give-up before returning. A thread found
        # dead with NEITHER terminated abnormally -- a BaseException
        # (SystemExit et al.) past layer 2's ``except Exception``, which
        # threading swallows without running any give-up path. The old
        # ``for t in threads: t.join()`` only noticed AFTER every
        # sibling exited: one flag-less death while a sibling kept
        # running parked this function in join() forever -- role
        # unsupervised, single-instance mutex held, zero evidence.
        # The loop cycles join(timeout=1s) over the live threads; on
        # the first flag-less death it logs CRITICAL, queues the
        # dialog, and CASCADES stop so the surviving siblings unwind
        # through their normal grace->terminate ladder and the loop
        # drains promptly.
        silent_death = False
        pending_threads = list(threads)
        while pending_threads:
            for t in list(pending_threads):
                t.join(timeout=1.0)
                if t.is_alive():
                    continue
                pending_threads.remove(t)
                with app.pending_errors_lock:
                    give_up_recorded = bool(app.pending_errors)
                if (
                    silent_death
                    or app.stop_event.is_set()
                    or give_up_recorded
                ):
                    continue
                silent_death = True
                _log.critical(
                    "supervisor thread %s exited with neither stop_event "
                    "set nor a give-up recorded; treating as a watchdog "
                    "internal crash (a clean-exit mimic otherwise) and "
                    "cascading stop to the surviving supervisors",
                    t.name,
                )
                with app.pending_errors_lock:
                    app.pending_errors.append(StartupError(
                        title="Yamabiko stopped unexpectedly",
                        message=(
                            "Yamabiko's supervisor stopped without a clean "
                            "shutdown request or an error report."
                        ),
                        suggested_fix=(
                            "Restart Yamabiko. If the problem persists, "
                            "share the main-*.log files from "
                            "%LOCALAPPDATA%\\Yamabiko\\logs."
                        ),
                        details=(
                            f"run_watchdog: supervisor thread {t.name} "
                            f"terminated with stop_event unset and no "
                            f"give-up recorded."
                        ),
                    ))
                # Cascade in _on_signal's Tier A order: graceful ping to
                # the pipeline child FIRST, so a still-supervised child
                # gets its grace window before _wait_or_stop's
                # terminate ladder, THEN stop_event (the _mirror thread
                # forwards it to llama_stop_event).
                try:
                    app.pipeline_writer.signal()
                except Exception as exc:
                    _log.warning(
                        "pipeline_writer.signal() in dead-thread cascade "
                        "raised: %s", exc,
                    )
                app.stop_event.set()
        # Render any queued give-up dialogs HERE, on the
        # main thread, never inside a supervisor worker thread.
        with app.pending_errors_lock:
            queued = list(app.pending_errors)
            app.pending_errors.clear()
        # The supervisors are done; release the single-instance mutex so
        # a relaunch while a dialog is open is not refused as a second
        # instance.
        _release_single_instance_mutex(app.single_instance_handle)
        app.single_instance_handle = None
        for err in queued:
            show_startup_error(err)
        # Exit code from the drained queue, not only the silent-death
        # flag: EVERY give-up (internal error, anti-flap, spawn failure,
        # flag-less dead thread) means this run ended in an error dialog,
        # and hard rule #9 consumers (frozen E2E, installer smoke) read
        # the watchdog exit code as ground truth. Exit 0 here used to
        # report a supervisor internal error as a successful run.
        return 1 if queued else 0
    finally:
        # Tier A graceful ping. In the SIGINT / SIGTERM path this was
        # already fired by _on_signal BEFORE stop_event was set so the
        # pipeline had its grace window to run its finally (glossary
        # save, uvicorn drain, hotkey unregister) while
        # Watchdog._wait_or_stop was still in proc.wait(grace_s).
        # On supervisor-natural-exit paths (pipeline exited on its own,
        # STOP_ALL via classify(), anti-flap give-up) the signal below
        # is a defensive no-op: child is already dead. signal() is
        # idempotent, so firing again here is safe belt-and-suspenders.
        try:
            app.pipeline_writer.signal()
        except Exception as exc:
            _log.debug(
                "pipeline_writer.signal() in finally raised (expected "
                "if handle already closed): %s", exc,
            )
        shutdown_reason = (
            "stop_event" if app.stop_event.is_set() else "supervisor_exit"
        )
        _log.info("watchdog exiting; reason=%s", shutdown_reason)
        app.pipeline_writer.close()
        # Reap the self-shutdown watcher. It is parked forever on
        # WaitForSingleObject; signal the event (the runner_done flag
        # set first makes the watcher exit without re-running
        # _on_signal), join it, then close the handle. Harmless
        # once-per-process in production; without it every
        # run_watchdog-driving test leaked one parked daemon thread +
        # one kernel event handle per call.
        runner_done.set()
        try:
            watchdog_self_writer.signal()
        except Exception as exc:
            _log.debug("self-shutdown reap signal raised: %s", exc)
        self_watch_thread.join(timeout=2.0)
        if self_watch_thread.is_alive():
            # CloseHandle on a handle with a PENDING WaitForSingleObject
            # is undefined behavior (the same Microsoft-docs rationale
            # ShutdownEventListener.close() quotes -- it leaks instead).
            # Once-per-process at exit; leaking beats UB.
            _log.warning(
                "self-shutdown watcher still parked after 2s; leaking its "
                "event handle instead of closing under a pending wait"
            )
        else:
            watchdog_self_writer.close()


def main() -> None:
    """Entry point — invoked by the Start Menu shortcut.

    Handles early-exit flags (--version / --verify-asr-runtime / --run-main /
    --run-overlay / --help) inline because they short-circuit before any
    mutex acquisition or supervisor composition. Everything else
    delegates to build_watchdog + run_watchdog.
    """
    # Short-circuit --version BEFORE any heavy imports or subprocess work.
    if "--version" in sys.argv[1:]:
        _print_version_and_exit()

    # Suppressed release-artifact smoke. It intentionally precedes --help so
    # ``--verify-asr-runtime --help`` cannot false-green through public help on
    # an artifact that lacks the route or the faster-whisper package code.
    if _ASR_RUNTIME_SMOKE_FLAG in sys.argv[1:]:
        _verify_asr_runtime_and_exit()

    # --run-main: the frozen-bundle re-entry hatch. When the watchdog's
    # subprocess is spawned as `yamabiko.exe --run-main`, skip the
    # supervise/signal/restart logic entirely and drive the pipeline in
    # this process.
    if "--run-main" in sys.argv[1:]:
        from yamabiko import main as _main_mod
        _main_mod.main()
        return

    # --run-overlay: same re-entry trick for the Qt desktop overlay.
    # overlay_app's main() uses argparse, so strip the dispatch flag
    # from sys.argv before delegating.
    if "--run-overlay" in sys.argv[1:]:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--run-overlay"]
        from yamabiko.overlay_app.__main__ import main as _overlay_main
        sys.exit(_overlay_main() or 0)

    # --help: print usage and exit. Caught early before heavy imports.
    # It comes AFTER the re-entry dispatch above: the rest of a --run-main /
    # --run-overlay argv, --help included, belongs to the child's own
    # argparse, so ``--run-main --help`` imports yamabiko.main's whole
    # module tree and prints the pipeline's usage (the frozen-bundle import
    # smoke relies on this). The watchdog never forwards --help to its
    # children, so normal launches are unaffected.
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print(_HELP_TEXT)
        sys.exit(0)

    # Layer-1 self-crash evidence: anything escaping the compose/run
    # path below used to die on the frozen bundle's discarded stderr --
    # process gone, no log line, no dialog, and an exit code nobody
    # keeps. SystemExit is re-raised UNTOUCHED first: sys.exit is how
    # the deliberate paths terminate (the mutex "already running"
    # exit 0 inside build_watchdog, run_watchdog's exit code below),
    # and rewriting those into a crash dialog + exit 1 would turn
    # every second-instance launch into a bogus crash report.
    try:
        # build_watchdog acquires the single-instance mutex FIRST. A second
        # launch (mutex already held by the running instance) shows the
        # "already running" dialog and sys.exit(0)s inside build_watchdog,
        # before the orphan sweep below -- so a double-launch can no longer
        # force-kill the FIRST instance's children.
        app = build_watchdog(sys.argv[1:])
        # Keep the mutex handle alive for the process lifetime by stashing
        # it on the module. Python will close it automatically at exit.
        globals()["_single_instance_handle_keepalive"] = app.single_instance_handle

        # Startup zombie sweep: if a PRIOR yamabiko.exe crashed or was
        # force-killed, its children (llama-server, pipeline, overlay) are
        # likely orphaned and holding :8090 + GPU VRAM + overlay window.
        # Kill them before run_watchdog spawns our own. Only processes from
        # our project / bundle tree are touched. Reached only once we own the
        # mutex (i.e. we are the sole instance), so it never targets a healthy
        # running instance -- a crashed instance has already released its mutex.
        _sweep_orphaned_children()

        rc = run_watchdog(app)
    except SystemExit:
        raise
    except ConfigValidationError as exc:
        # build_watchdog raises this from its config load, BEFORE its
        # setup_main_log call: attach the watchdog's file handler here so
        # the reason reaches main-watchdog.log, not only the discarded
        # stderr of the frozen bundle.
        from yamabiko import log_config as _log_config
        _yamabiko_logger = logging.getLogger("yamabiko")
        if not any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for lg in (_yamabiko_logger, logging.getLogger())
            for h in lg.handlers
        ):
            _log_config.setup_main_log(
                _yamabiko_logger, level="INFO", filename="main-watchdog.log",
            )
        _log.error("invalid configuration", exc_info=True)
        show_startup_error(StartupError(
            title="Invalid configuration",
            message=(
                "Yamabiko could not start because its configuration is "
                "invalid."
            ),
            suggested_fix=(
                "Correct the named values in your explicit config or "
                "%LOCALAPPDATA%\\Yamabiko\\user_overrides.yaml, then "
                "restart Yamabiko."
            ),
            details=str(exc),
        ))
        logging.shutdown()
        sys.exit(43)
    except KeyboardInterrupt:
        # A Ctrl+C landing in the pre-signal-handler startup window
        # (build_watchdog's config load, the orphan sweep -- before
        # run_watchdog installs _on_signal) is a USER action, not a
        # crash: no dialog, no CRITICAL, one INFO line and the shell's
        # killed-by-SIGINT convention (128+2). A literal, not an
        # exit_codes.py entry: the watchdog is the TOP of the process
        # tree, nothing classify()s ITS exit code, and adding 130 to
        # the supervised-child policy table would wrongly make a child
        # 130 mean something.
        _log.info("watchdog interrupted during startup (Ctrl+C); exiting")
        sys.exit(130)
    except BaseException as exc:
        _log.critical("watchdog crashed", exc_info=True)
        # Best-effort dialog -- the user must see SOMETHING beyond
        # "Yamabiko simply didn't start". Guarded: a Qt failure here
        # (headless session, broken display) must not mask exit 1.
        try:
            show_startup_error(StartupError(
                title="Yamabiko crashed",
                message=(
                    "Yamabiko's supervisor process hit an unexpected "
                    "error and had to stop."
                ),
                suggested_fix=(
                    "Restart Yamabiko. If the problem persists, share "
                    "the main-*.log and crash-*.log files from "
                    "%LOCALAPPDATA%\\Yamabiko\\logs."
                ),
                details=f"{type(exc).__name__}: {exc}",
            ))
        except Exception:
            _log.warning("crash dialog could not be shown", exc_info=True)
        # Flush buffered records before dying -- the CRITICAL above is
        # the only evidence of WHY.
        logging.shutdown()
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":  # pragma: no mutate -- the ==-inversion mutant would run main() at import time and wedge the test run
    main()
