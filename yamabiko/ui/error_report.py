"""The startup error window's report: system facts, the recent log and the
report text the "Copy report" button puts on the clipboard.

Qt-free (``error_dialog`` imports it at module level, and the watchdog
imports ``error_dialog`` before any Qt exists). Every string that leaves this
module is passed through :func:`redact_user`, so the window's footer line
("your user name is left out") stays true for the panel, the log view and
the clipboard alike.
"""
from __future__ import annotations

import functools
import getpass
import logging
import logging.handlers
import os
import platform
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple, TypeVar

from yamabiko import __version__, gpu, paths

if TYPE_CHECKING:
    from yamabiko.ui.error_dialog import StartupError

_log = logging.getLogger(__name__)

NO_GPU = "None reported by Windows"
# The logger show_startup_error writes its "startup error dialog:" line to;
# the tail comes from the file its records reach.
_DIALOG_LOGGER = "yamabiko.ui.error_dialog"
_TAIL_BYTES = 64 * 1024
# One path separator as it appears in logs and reprs: a doubled backslash
# (repr), a backslash, or a forward slash.
_SEP = r"(?:\\\\|\\|/)"
# Path-name characters that may continue a folder name: a match must end
# where the folder name ends ("C:\Users\ann" never matches "C:\Users\anne").
_NAME_END = r"(?![\w.~-])"
_T = TypeVar("_T")


class LogTail(NamedTuple):
    """The current log file and its last lines (redacted, raw format)."""

    path: Path | None
    lines: list[str]


class ReportFacts(NamedTuple):
    """The Report panel's facts, collected once when the window opens."""

    version: str
    windows: str
    gpu: str
    python: str
    error: str
    log: LogTail


def _user_name() -> str:
    """The Windows user name: USERNAME, else getpass; "" when unknown."""
    name = os.environ.get("USERNAME", "")
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:
        return ""


def _short_path(path: str) -> str | None:
    """``path``'s 8.3 short form (GetShortPathNameW), or None off Windows,
    for a path that does not exist, or when the call fails."""
    if sys.platform != "win32" or not path:
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")  # private handle: argtypes stay local
    kernel32.GetShortPathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    kernel32.GetShortPathNameW.restype = wintypes.DWORD
    size = kernel32.GetShortPathNameW(path, None, 0)
    if not size:
        return None
    buffer = ctypes.create_unicode_buffer(size)
    if not kernel32.GetShortPathNameW(path, buffer, size):
        return None
    return buffer.value


def _folder_pattern(folder: str) -> str:
    """A regex for ``folder`` written with any of the three separators."""
    parts = [re.escape(part) for part in re.split(r"[\\/]+", folder.rstrip("\\/")) if part]
    return _SEP.join(parts) + _NAME_END


@functools.lru_cache(maxsize=8)
def _rules(local: str, profile: str, user: str) -> tuple[tuple[re.Pattern[str], str], ...]:
    """The ordered (pattern, replacement) pairs for one environment. Cached:
    the 8.3 lookups are OS calls, and the environment rarely changes."""
    rules: list[tuple[re.Pattern[str], str]] = []
    for folder, token in ((local, "%LOCALAPPDATA%"), (profile, "%USERPROFILE%")):
        if not folder:
            continue
        forms = [folder]
        short = _short_path(folder)
        if short and short.lower() != folder.lower():
            forms.append(short)
        for form in forms:
            rules.append((re.compile(_folder_pattern(form), re.IGNORECASE), token))
    if user:
        name = re.escape(user)
        # Anchored by "Users", so even a one-character name is safe here.
        rules.append((
            re.compile(rf"(?P<a>{_SEP})Users(?P<b>{_SEP}){name}{_NAME_END}", re.IGNORECASE),
            r"\g<a>Users\g<b><user>",
        ))
        if len(user) >= 2:  # a bare one-letter "name" would match everywhere
            rules.append((re.compile(rf"(?<![\w<]){name}(?![\w>])", re.IGNORECASE), "<user>"))
    return tuple(rules)


def redact_user(text: str) -> str:
    """``text`` with the Windows user name taken out: the LOCALAPPDATA and
    USERPROFILE folders (long and 8.3 short forms, with backslash, forward
    slash or doubled-backslash separators) become ``%LOCALAPPDATA%`` and
    ``%USERPROFILE%``, any other ``\\Users\\<name>`` segment becomes
    ``\\Users\\<user>``, and the bare user name becomes ``<user>``.
    Case-insensitive; in that order."""
    if not text:
        return text
    rules = _rules(os.environ.get("LOCALAPPDATA", ""), os.environ.get("USERPROFILE", ""),
                   _user_name())
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def windows_version() -> str:
    """The Windows field: "11 26H2, build 26300.9818" from the registry
    (DisplayVersion, CurrentBuild, UBR; ProductName says "Windows 10" on
    Windows 11, so the major version comes from the build number); the
    platform module's release and version when the registry can't say."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            display = winreg.QueryValueEx(key, "DisplayVersion")[0]
            build = winreg.QueryValueEx(key, "CurrentBuild")[0]
            ubr = winreg.QueryValueEx(key, "UBR")[0]
        major = 11 if int(build) >= 22000 else 10
        return f"{major} {display}, build {build}.{ubr}"
    except Exception:
        return f"{platform.release()}, build {platform.version()}"


def gpu_field() -> str:
    """The GPU field: "NVIDIA GeForce RTX 5080, 16 GB, driver 617.42", or
    "None reported by Windows" when nvidia-smi reports no card."""
    summary = gpu.detect_gpu_summary()
    if summary is None:
        return NO_GPU
    return f"{summary.name}, {round(summary.vram_mib / 1024)} GB, driver {summary.driver}"


def _current_log_file() -> Path | None:
    """The file this process logs to: the first RotatingFileHandler on the
    dialog's logger or a parent its records reach; else the newest
    main-*.log in the logs folder; None when there is none."""
    logger: logging.Logger | None = logging.getLogger(_DIALOG_LOGGER)
    while logger is not None:
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                return Path(handler.baseFilename)
        if not logger.propagate:
            break
        logger = logger.parent
    try:
        candidates = [(p.stat().st_mtime, p) for p in paths.logs_dir().glob("main-*.log")]
    except OSError:
        return None
    return max(candidates)[1] if candidates else None


def read_log_tail(max_lines: int = 40) -> LogTail:
    """The last ``max_lines`` non-blank lines of the current log file, read
    from its last 64 KiB (a cut first line is dropped), redacted. Call it
    after the "startup error dialog:" line is written so the tail ends with
    it. Any OSError gives no lines."""
    path = _current_log_file()
    if path is None:
        return LogTail(None, [])
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - _TAIL_BYTES)
            handle.seek(start)
            data = handle.read()
    except OSError:
        _log.warning("could not read the log tail", exc_info=True)
        return LogTail(path, [])
    lines = data.decode("utf-8", errors="replace").splitlines()
    if start > 0:
        lines = lines[1:]
    kept = [redact_user(line) for line in lines if line.strip()]
    return LogTail(path, kept[-max_lines:] if max_lines > 0 else [])


def error_line(err: "StartupError") -> str:
    """The Error field: the last non-empty line of the details (a
    traceback's "Type: message" line), else the message; redacted."""
    for line in reversed(err.details.splitlines()):
        if line.strip():
            return redact_user(line.strip())
    return redact_user(err.message)


def _guarded(fact: str, read: Callable[[], _T], fallback: _T) -> _T:
    """``read()``, or ``fallback`` when it raises: a fact the window can't
    read must never stop it from showing (every caller exits 43 right after
    it, and an exception would crash the pipeline into a restart instead)."""
    try:
        return read()
    except Exception:
        _log.warning("startup report: could not read the %s; showing %r", fact, fallback,
                     exc_info=True)
        return fallback


def collect_facts(err: "StartupError") -> ReportFacts:
    """Collect the Report panel's facts once (nvidia-smi and the registry
    run here, synchronously; an absent nvidia-smi is instant). Each OS read
    is guarded: a failure shows its fallback, never raises."""
    v = sys.version_info
    return ReportFacts(
        version=__version__,
        windows=redact_user(_guarded("Windows version", windows_version, "unknown")),
        gpu=redact_user(_guarded("GPU", gpu_field, NO_GPU)),
        python=f"{v.major}.{v.minor}.{v.micro}",
        error=error_line(err),
        log=_guarded("log tail", read_log_tail, LogTail(None, [])),
    )


def report_text(err: "StartupError", facts: ReportFacts) -> str:
    """The report "Copy report" puts on the clipboard, laid out as the
    window reads, redacted as a whole."""
    lines = [
        "Yamabiko startup error",
        f"Title: {err.title}",
        f"Message: {err.message}",
        f"What to try: {err.suggested_fix}",
        "",
        f"Yamabiko: {facts.version}",
        f"Windows: {facts.windows}",
        f"GPU: {facts.gpu}",
        f"Python: {facts.python}",
        f"Error: {facts.error}",
        "",
        "Recent log:",
        *facts.log.lines,
        "",
        "Details:",
        err.details.rstrip("\n"),
    ]
    return redact_user("\n".join(lines)) + "\n"


# ---------------------------------------------------------------------------
# The window's condensed log
# ---------------------------------------------------------------------------

_LEVELS = "DEBUG|INFO|WARNING|ERROR|CRITICAL"
# log_config.MAIN_LOG_FORMAT with the default asctime:
# "2026-11-01 21:41:06,123 p4242 MainThread yamabiko.capture.audio ERROR text"
_RECORD = re.compile(
    rf"^\d{{4}}-\d{{2}}-\d{{2}} (?P<time>\d{{2}}:\d{{2}}:\d{{2}}),\d{{3}} p\d+ (?P<thread>.+?) "
    rf"(?P<name>\S+) (?P<level>{_LEVELS})(?: (?P<text>.*))?$"
)
_EXCEPTION_NAME = re.compile(
    r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|NotFound\w*)\b.*:")
_TRACEBACK_END = re.compile(r"^[A-Za-z_][\w.]*(?::\s|:?$)")


class LogLine(NamedTuple):
    """One line of the window's log view."""

    kind: str   # "record" | "exception" | "text"
    time: str
    name: str
    level: str
    text: str


def condense_log(lines: list[str]) -> list[LogLine]:
    """Parse raw log lines for the window: a record keeps its time, the last
    part of its logger name, its level and its message; a traceback's final
    "Type: message" line (or any line naming an exception type) is an
    "exception"; every other line (traceback frames, continuations) is
    "text", shown as it is."""
    out: list[LogLine] = []
    in_traceback = False
    for raw in lines:
        record = _RECORD.match(raw)
        if record is not None:
            in_traceback = False
            name = record["name"].rsplit(".", 1)[-1]
            out.append(LogLine("record", record["time"], name, record["level"],
                               record["text"] or ""))
            continue
        if raw.startswith("Traceback (most recent call last):"):
            in_traceback = True
            out.append(LogLine("text", "", "", "", raw))
            continue
        ends_traceback = in_traceback and not raw[:1].isspace() and _TRACEBACK_END.match(raw)
        if ends_traceback or _EXCEPTION_NAME.match(raw):
            in_traceback = False
            out.append(LogLine("exception", "", "", "", raw))
            continue
        out.append(LogLine("text", "", "", "", raw))
    return out
