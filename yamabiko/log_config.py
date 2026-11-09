"""Rotating log configuration for Yamabiko.

Two log surfaces:
  - latency.log — JSON lines, one per pipeline segment. Written directly
    via an open file handle because main.py's hot path doesn't want
    logging-module overhead per segment. Size rotation is implemented
    by wrapping the open handle in a small size-aware writer.
  - main-<role>.log — standard stdlib logging with RotatingFileHandler.
    One file per OS process: main-watchdog.log, main-pipeline.log,
    main-overlay.log. Never share one file across processes — Windows
    rotation renames fail on open files held by another process.

Both live under paths.logs_dir() (= %LOCALAPPDATA%\\Yamabiko\\logs).
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import tempfile
import threading
from pathlib import Path
from typing import IO, Optional

from yamabiko import paths

_LATENCY_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_LATENCY_DEFAULT_BACKUP = 5
_MAIN_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_MAIN_DEFAULT_BACKUP = 5

# One shared record layout for every process's main-*.log AND the
# overlay's console basicConfig (overlay_app/__main__.py imports this
# constant -- keep them identical). p<pid> + thread name make the E2E
# _LogTail's merged multi-process view triageable: a respawned child
# shows up as a new pid, and a keyboard/supervisor-thread record is
# separable from MainThread.
MAIN_LOG_FORMAT = (
    "%(asctime)s p%(process)d %(threadName)s %(name)s %(levelname)s %(message)s"
)

# crash-<role>.log rotation threshold, checked once at open (crash logs
# are written by a dying process; there is no runtime rotation).
_CRASH_LOG_MAX_BYTES = 5 * 1024 * 1024

# Module-global refs so the faulthandler target handle outlives
# install_crash_evidence (faulthandler needs a live fd at crash time),
# plus per-role idempotence so repeat calls (test harnesses rebuilding
# an app in one process) don't stack excepthook chains.
_CRASH_LOG_HANDLES: dict[str, IO[str]] = {}
_INSTALLED_ROLES: set[str] = set()

_log = logging.getLogger(__name__)


# The six standard stdlib level names ``logger.setLevel`` accepts as
# strings. ``setLevel("debug")`` / ``setLevel("10")`` / ``setLevel("verbose")``
# raise ``ValueError``, so an unvalidated ``logging.level`` config value
# would abort startup -- and the contract everywhere this is consumed is
# that bad config must NEVER block startup.
_STD_LOG_LEVEL_NAMES = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)

# stdlib-registered aliases ``logger.setLevel`` accepts as strings
# (logging registers WARN and FATAL alongside the canonical names).
# The pre-normalization pipeline passed the raw config value straight
# to setLevel, so these were honored; rejecting them here as junk
# silently downgraded a user's WARN config to INFO with only a
# fallback warning. Mapped to the canonical name, no warning.
_LOG_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}


def _best_effort_stderr(message: str) -> None:
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass


def normalize_log_level(
    value: object, *, default: str = "INFO",
) -> tuple[str, "str | None"]:
    """Coerce a config-sourced ``logging.level`` to a stdlib level NAME.

    Single source of truth for level normalization. Case-insensitive;
    the stdlib aliases ``WARN`` / ``FATAL`` map to ``WARNING`` /
    ``CRITICAL`` (setLevel accepts them, so they are valid, not junk);
    anything else that is not one of the six standard names (lowercase
    ``debug`` is fine; ``verbose`` / numeric YAML are not) degrades to
    ``default``. ``None`` (no configured level) -> ``default`` silently.

    Returns ``(level_name, warning)`` where ``warning`` is ``None`` or a
    message the CALLER should log: the caller controls emission timing so
    the watchdog can sequence the warning AFTER ``setup_main_log`` attaches
    the file handler (a warning emitted before that dies on the frozen
    bundle's discarded stderr). ``config.load_full_config`` normalizes the
    merged value through here, ``setup_main_log`` re-applies it as a
    belt-and-suspenders guard on the setLevel call, and the watchdog's
    overlay ``--log-level`` forward reuses it -- so the pipeline, watchdog,
    and overlay level handling can never drift.
    """
    if value is None:
        return default, None
    name = str(value).strip().upper()
    name = _LOG_LEVEL_ALIASES.get(name, name)
    if name in _STD_LOG_LEVEL_NAMES:
        return name, None
    return default, (
        f"logging.level={value!r} is not one of "
        f"{sorted(_STD_LOG_LEVEL_NAMES)}; falling back to {default}"
    )


class _NullWriter:
    """Silent-drop writer used when log-file open fails at startup.

    Matches the subset of the text-file protocol _SizeRotatingTextIO
    exposes so callers don't need to special-case a None handle.
    Lets the app continue running without its latency.log when
    the logs directory is read-only or out of disk.
    """
    def write(self, s: str) -> int:
        return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _SizeRotatingTextIO:
    """Thin wrapper around a text file that rolls over on size.

    main.py writes JSON lines to latency.log directly (no logging module).
    We preserve that interface while adding rotation: on every write,
    check the current size and roll if past max_bytes.
    """

    def __init__(
        self, path: Path, *, max_bytes: int, backup_count: int,
    ) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        # If the parent dir is read-only or the disk is full,
        # mkdir / open would raise OSError and the app would crash at
        # startup with no diagnostic (because error_dialog itself
        # relies on logging being up). Fall back to a silent-drop
        # writer so the app stays alive; tray + pipeline still work,
        # user just loses the log for this session.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh: IO[str] = open(self._path, "a", encoding="utf-8")
        except OSError as exc:
            # Best-effort stderr warning so the user has SOMETHING to
            # triage from if the app misbehaves. Safe even in windowed
            # mode -- watchdog wraps stderr in _SafeTextIO, so writes
            # are no-ops not exceptions.
            _best_effort_stderr(
                f"WARNING: could not open {self._path}: {exc!r}. "
                "Running with logging disabled."
            )
            self._fh = _NullWriter()  # type: ignore[assignment]
            self._bytes_written = 0
            return
        # Seed byte counter from current file size — append-mode tell() is
        # platform-unreliable until the first write.
        try:
            self._bytes_written = self._path.stat().st_size
        except OSError:
            self._bytes_written = 0

    def write(self, s: str) -> int:
        # Once the writer has degraded to _NullWriter (open failed
        # at construction, or a rollover failed below), drop silently
        # with NO byte accounting — counting would re-fire _rollover
        # every max_bytes and re-attempt the exact open() that failed.
        if isinstance(self._fh, _NullWriter):
            return self._fh.write(s)
        encoded_len = len(s.encode("utf-8"))
        if self._bytes_written + encoded_len > self._max_bytes:
            self._rollover()
            if isinstance(self._fh, _NullWriter):
                return self._fh.write(s)
        try:
            n = self._fh.write(s)
        except OSError as exc:
            self._disable("write", exc)
            return len(s)
        self._bytes_written += encoded_len
        return n

    def flush(self) -> None:
        try:
            self._fh.flush()
        except OSError as exc:
            self._disable("flush", exc)

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError as exc:
            self._disable("close", exc)

    def _disable(self, operation: str, exc: OSError) -> None:
        if isinstance(self._fh, _NullWriter):
            return
        _best_effort_stderr(
            f"WARNING: latency log {operation} failed: {exc!r}. "
            "Latency logging disabled for this session."
        )
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = _NullWriter()  # type: ignore[assignment]
        self._bytes_written = 0

    def _rollover(self) -> None:
        try:
            self._fh.close()
            # NOTE: assumes path has a standard final extension (e.g. ".log").
            # Paths with no suffix or with multi-dot suffixes (e.g. "foo.log.gz")
            # would append the index as a new suffix ("foo.log.gz.1") instead
            # of replacing; fine for current callers (latency.log only).
            for i in range(self._backup_count, 0, -1):
                src = (
                    self._path.with_suffix(self._path.suffix + f".{i - 1}")
                    if i > 1 else self._path
                )
                dst = self._path.with_suffix(self._path.suffix + f".{i}")
                if src.exists():
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
            self._fh = open(self._path, "a", encoding="utf-8")
            self._bytes_written = 0
        except OSError as exc:
            # A rollover failure (locked backup file, disk full,
            # logs dir gone read-only) must never escape from the
            # subtitle hot path. Mirror the constructor's fallback:
            # lose this session's latency log, keep delivery alive.
            self._disable("rollover", exc)


def setup_latency_log(
    *,
    enabled: bool = True,
    max_bytes: int = _LATENCY_DEFAULT_MAX_BYTES,
    backup_count: int = _LATENCY_DEFAULT_BACKUP,
) -> Optional[_SizeRotatingTextIO]:
    """Return a write handle for latency.log, or None when disabled.

    When ``enabled`` is False the per-segment handler in main.py is
    omitted entirely (no file created, no writes issued).
    """
    if not enabled:
        return None
    paths.ensure_dirs()
    return _SizeRotatingTextIO(
        paths.logs_dir() / "latency.log",
        max_bytes=max_bytes,
        backup_count=backup_count,
    )


def _ancestor_has_rotating_file_handler(logger: logging.Logger) -> bool:
    """True iff some ancestor in ``logger``'s parent chain has a
    ``RotatingFileHandler`` attached AND records from ``logger`` would
    propagate up to it.

    Guards the double-write case: if ``setup_main_log`` is called on
    BOTH the root logger and the ``yamabiko`` logger in the same process
    (which is what ``main()`` does, via ``build_pipeline``), every
    ``yamabiko.xxx`` log line is written twice to main.log: once by
    ``yamabiko``'s own handler, once by root's handler via Python's
    default ``logger.propagate = True`` hierarchy propagation. Detecting
    an ancestor that already files-to-disk lets the second call skip,
    leaving propagation as the single write path.

    Propagation check: if any step in the chain between ``logger`` and
    the handler-bearing ancestor has ``propagate=False``, records never
    reach the ancestor — in which case we'd be suppressing our own
    handler with no substitute, losing every log line. Bail early in
    that case so the caller attaches a handler on ``logger`` as normal.

    Assumption: every ``setup_main_log`` call within one process targets
    the same ``paths.logs_dir() / filename`` path (the role
    convention: one filename per process, identical across that
    process's calls). Callers writing to a *different* file via a
    different ``RotatingFileHandler`` up the chain would be incorrectly
    suppressed, but we don't have that case.
    """
    cur = logger
    parent = logger.parent
    while parent is not None:
        # Records only propagate from ``cur`` to ``parent`` when
        # ``cur.propagate`` is True. Break the chain and the ancestor's
        # handler never sees ``logger``'s records -- skipping our own
        # attachment would mean logs silently vanish.
        if not cur.propagate:
            return False
        if any(isinstance(h, logging.handlers.RotatingFileHandler)
               for h in parent.handlers):
            return True
        cur = parent
        parent = parent.parent
    return False


def setup_main_log(
    logger: logging.Logger,
    *,
    level: str = "INFO",
    max_bytes: int = _MAIN_DEFAULT_MAX_BYTES,
    backup_count: int = _MAIN_DEFAULT_BACKUP,
    filename: str,
) -> None:
    """Attach a RotatingFileHandler (+ StreamHandler) to ``logger`` if absent.

    ``filename`` is REQUIRED (no default) and MUST be unique per OS
    process: a default of "main.log" would let a future call
    site silently reinstate the shared-file bug this parameter exists
    to prevent. stdlib
    RotatingFileHandler rotation is only safe with a single writer
    process: on Windows, ``doRollover``'s ``os.rename`` raises
    PermissionError while any other process holds the file open, so a
    shared file silently drops every post-rollover record and each
    failed retry shifts the backup chain until all backups are gone.
    Role convention: the watchdog passes ``main-watchdog.log``, the
    pipeline/wizard child ``main-pipeline.log``, the overlay child
    ``main-overlay.log``. Within one process, every call must use the
    same filename (see ``_ancestor_has_rotating_file_handler``).

    Idempotent — safe to call multiple times; only one file handler and
    one stream handler are ever attached per logger. ALSO skips file
    handler attachment if any ancestor in the parent chain already has
    a RotatingFileHandler (see ``_ancestor_has_rotating_file_handler``):
    without that check, calling on both root and the ``yamabiko`` logger
    in the same process (as ``main.main()`` does) produces a 2x
    double-write of every ``yamabiko.xxx`` log line via propagation.
    """
    has_file = (
        any(isinstance(h, logging.handlers.RotatingFileHandler)
            for h in logger.handlers)
        or _ancestor_has_rotating_file_handler(logger)
    )
    has_stream = any(isinstance(h, logging.StreamHandler)
                     and not isinstance(h, logging.FileHandler)
                     for h in logger.handlers)
    # In windowed-bundle mode (runw.exe) sys.stderr is a wrapped
    # handle whose writes are silently discarded. Adding a
    # StreamHandler in that environment means every log call runs
    # format() + emit twice, once for nothing. Skip the StreamHandler
    # entirely when sys.frozen is set -- file handler still captures
    # everything. In dev mode (console), keep the StreamHandler so
    # logs are visible in the terminal.
    want_stream = not getattr(sys, "frozen", False)
    # The level must apply even when handler attachment is fully
    # deduped. In the frozen pipeline child, root gets the file handler
    # at INFO first (main.main()), then build_pipeline configures the
    # ``yamabiko`` logger with the user's logging.level; that second
    # call hits this early-return (has_file via ancestor, no stream
    # wanted when frozen), so a setLevel placed below it would never
    # run and the config key would be dead on the shipped path. Dev
    # mode keeps want_stream=True, which skips the early-return.
    #
    # Normalize here too (config.load_full_config already does at load):
    # this is the single choke point for the setLevel call, so a caller
    # that passes a raw config value or a junk literal ("verbose") can
    # never make logger.setLevel raise ValueError and abort startup.
    level_name, _ = normalize_log_level(level)
    logger.setLevel(level_name)
    if has_file and (has_stream or not want_stream):
        return
    formatter = logging.Formatter(MAIN_LOG_FORMAT)
    if not has_file:
        # Wrap the file handler init in try/except so a
        # read-only or full logs dir doesn't crash the whole app at
        # startup. First OSError triggers ONE retry under
        # %TEMP%\Yamabiko-logs\ -- an unwritable %LOCALAPPDATA%
        # (roaming-profile lockdown, controlled-folder-access deny,
        # volume full) otherwise leaves a beta user with zero file
        # evidence, because the windowed bundle discards the stderr
        # warning below. Only after BOTH locations fail does the
        # process run file-less.
        file_handler: Optional[logging.handlers.RotatingFileHandler] = None
        primary_exc: Optional[OSError] = None
        try:
            paths.ensure_dirs()
            file_handler = logging.handlers.RotatingFileHandler(
                paths.logs_dir() / filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        except OSError as exc:
            primary_exc = exc
            try:
                fallback_dir = Path(tempfile.gettempdir()) / "Yamabiko-logs"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.handlers.RotatingFileHandler(
                    fallback_dir / filename,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            except OSError as exc2:
                try:
                    print(
                        f"WARNING: could not open {filename}: {exc!r}; "
                        f"temp fallback failed too: {exc2!r}. "
                        f"Continuing without file logging.",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
        if file_handler is not None:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            if primary_exc is not None:
                # Recorded into the fallback file itself, so whoever
                # finds it knows why it isn't in the normal logs dir.
                logger.warning(
                    "primary logs dir unavailable (%r); logging "
                    "redirected to %s",
                    primary_exc, file_handler.baseFilename,
                )
    if want_stream and not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def install_crash_evidence(role: str) -> None:
    """Arm the calling process's crash-evidence primitives. Never raises.

    Three independent best-effort layers (a setup failure logs a warning
    and the app keeps starting -- crash evidence must never become the
    crash):

    1. ``faulthandler.enable`` (faulting thread only; the all_threads
       comment at the call site is load-bearing) against an explicit,
       process-lifetime handle to ``logs_dir()/crash-<role>.log`` (append; if over
       ``_CRASH_LOG_MAX_BYTES`` at open it is ``os.replace``d to
       ``crash-<role>.log.1`` first). The explicit handle is REQUIRED:
       ``faulthandler.enable()``'s default ``sys.stderr`` is a
       NullWriter/None in the windowed frozen bundle and raises
       AttributeError there (pyinstaller#4481). The handle is parked in
       ``_CRASH_LOG_HANDLES`` so it outlives this call.
    2. ``sys.excepthook`` -> ``log.critical(..., exc_info=...)`` BEFORE
       chaining to the previous hook, so an unhandled main-thread
       exception reaches main-<role>.log instead of dying on the
       discarded frozen stderr.
    3. ``threading.excepthook`` -> same, for non-main threads (keyboard
       hotkey thread, supervisor threads, capture monitor).

    Idempotent per role within one process: a repeat call is a no-op, so
    harnesses that rebuild an app repeatedly don't stack hook chains.
    Call once per OS process, immediately after ``setup_main_log``.
    Roles: watchdog / pipeline / overlay (wizard runs inside the
    pipeline process and is covered by its call).
    """
    if role in _INSTALLED_ROLES:
        return
    _INSTALLED_ROLES.add(role)

    # Layer 1: faulthandler -> crash-<role>.log.
    try:
        paths.ensure_dirs()
        crash_path = paths.logs_dir() / f"crash-{role}.log"
        try:
            if (
                crash_path.exists()
                and crash_path.stat().st_size > _CRASH_LOG_MAX_BYTES
            ):
                # os.replace: atomic clobber of a stale .1 (plain rename
                # raises FileExistsError on Windows).
                os.replace(
                    crash_path, crash_path.with_name(crash_path.name + ".1"),
                )
        except OSError:
            # Rotation is best-effort; append to the oversize file
            # rather than lose faulthandler entirely.
            pass
        fh = open(crash_path, "a", encoding="utf-8")
        _CRASH_LOG_HANDLES[role] = fh
        # all_threads=False is load-bearing. On Windows this call also
        # installs a vectored exception handler that fires on
        # FIRST-CHANCE exceptions, including benign ones Qt/COM raises
        # and handles internally during window activation (0x8001010D
        # RPC_E_CANTCALLOUT_ININPUTSYNCCALL -- PYSIDE-2501). Walking
        # the OTHER, still-running threads' frames from that handler
        # races their mutation (python/cpython#128400); the resulting
        # access violation inside the handler killed the overlay with
        # 0xC000041D on every launch (2026-07-02 crash loop). Any role
        # can host Qt (overlay always; pipeline via wizard + error
        # dialogs), so every role dumps the faulting thread only.
        faulthandler.enable(file=fh, all_threads=False)
    except Exception:
        _log.warning(
            "crash-evidence: faulthandler setup failed for role %s; "
            "native crashes will leave no stack", role, exc_info=True,
        )

    # Layer 2: sys.excepthook (log first, then chain -- the previous
    # hook may only print to a stderr the frozen bundle discards).
    try:
        prev_sys_hook = sys.excepthook

        def _crash_excepthook(exc_type, exc, tb):
            try:
                _log.critical(
                    "unhandled exception in %s", role,
                    exc_info=(exc_type, exc, tb),
                )
            finally:
                prev_sys_hook(exc_type, exc, tb)

        sys.excepthook = _crash_excepthook
    except Exception:
        _log.warning(
            "crash-evidence: sys.excepthook install failed", exc_info=True,
        )

    # Layer 3: threading.excepthook.
    try:
        prev_thread_hook = threading.excepthook

        def _crash_threading_excepthook(args):
            try:
                _log.critical(
                    "unhandled exception in %s thread %s", role,
                    getattr(args.thread, "name", "?"),
                    exc_info=(args.exc_type, args.exc_value,
                              args.exc_traceback),
                )
            finally:
                prev_thread_hook(args)

        threading.excepthook = _crash_threading_excepthook
    except Exception:
        _log.warning(
            "crash-evidence: threading.excepthook install failed",
            exc_info=True,
        )


# Per-process GPU-name cache for the banner. The probe shells out to
# nvidia-smi; one shell-out per process is plenty (and keeps repeated
# build calls in test harnesses cheap).
_GPU_NAME_UNSET = object()
_gpu_name_cache: object = _GPU_NAME_UNSET


def _gpu_name_bounded(timeout_s: float = 2.0) -> Optional[str]:
    """Best-effort GPU name for the startup banner. Never raises.

    ``gpu.detect_gpu_name`` shells out to nvidia-smi with its own 5 s
    cap; the banner budget is 2 s, so the probe runs on a daemon thread
    that is abandoned at the deadline (a stuck driver must not delay
    startup logging).
    """
    global _gpu_name_cache
    if _gpu_name_cache is not _GPU_NAME_UNSET:
        return _gpu_name_cache  # type: ignore[return-value]
    result: list[Optional[str]] = []

    def _probe() -> None:
        try:
            from yamabiko import gpu
            result.append(gpu.detect_gpu_name())
        except Exception:
            result.append(None)

    try:
        t = threading.Thread(
            target=_probe, daemon=True, name="gpu-banner-probe",
        )
        t.start()
        t.join(timeout_s)
    except Exception:
        pass
    _gpu_name_cache = result[0] if result else None
    return _gpu_name_cache  # type: ignore[return-value]


def log_startup_banner(
    log: logging.Logger, role: str, *, config_digest: Optional[str] = None,
) -> None:
    """One INFO identity line at process startup. Never raises.

    Fields: role, yamabiko __version__, frozen|dev, PID, python version,
    platform.platform() -- everything bug_report.yml asks a reporter
    for, greppable from the first screen of the log. ``config_digest``
    (pipeline only) appends the merged-config fingerprint + override
    KEYS computed by the caller. A second "gpu:" INFO line follows for
    the watchdog + pipeline roles only (best-effort, <2 s, the overlay
    never touches the GPU).
    """
    try:
        from yamabiko import __version__
        line = (
            f"yamabiko {role} v{__version__} "
            f"({'frozen' if getattr(sys, 'frozen', False) else 'dev'}) "
            f"pid={os.getpid()} "
            f"python={platform.python_version()} "
            f"os={platform.platform()}"
        )
        if config_digest is not None:
            line += f" config={config_digest}"
        log.info("%s", line)
        if role in ("watchdog", "pipeline"):
            log.info("gpu: %s", _gpu_name_bounded() or "not detected")
    except Exception:
        try:
            log.warning("startup banner failed", exc_info=True)
        except Exception:
            pass
