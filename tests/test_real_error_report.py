"""Real-path tests for the startup error report's OS seams (@pytest.mark.real).

The 8.3 short-name lookup (GetShortPathNameW) on a real folder whose short
name differs, the registry read of the Windows version, and the log tail of
a real RotatingFileHandler's file. No mocks on the subject: real kernel32,
real winreg, real files. Runs in the real-deps phase (no window).
"""
from __future__ import annotations

import ctypes
import logging
import logging.handlers
import re
import sys
from ctypes import wintypes

import pytest

from yamabiko.log_config import MAIN_LOG_FORMAT
from yamabiko.ui import error_report as er

pytestmark = pytest.mark.real


def _short_form(path: str) -> str:
    """GetShortPathNameW, called here independently of the subject."""
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetShortPathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    kernel32.GetShortPathNameW.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(1024)
    assert kernel32.GetShortPathNameW(path, buffer, 1024), f"GetShortPathNameW failed for {path}"
    return buffer.value


def test_real_redaction_replaces_the_8dot3_short_form_of_a_real_profile_folder(monkeypatch, tmp_path):
    """A user name longer than eight characters or holding a space gets an
    8.3 short form (C:\\Users\\CHRIST~1) that tracebacks and %TEMP% paths
    carry. A real folder named like such a profile, on the temp volume:
    its long and short forms both become the variables.

    Seeded defect: drop the GetShortPathNameW lookup (or pass it the wrong
    buffer size) -> the short forms stay in the text -> red."""
    profile = tmp_path / "Yamabiko Redaction Profile"
    local = profile / "AppData" / "Local"
    local.mkdir(parents=True)
    short_profile = _short_form(str(profile))
    short_local = _short_form(str(local))
    assert short_profile.lower() != str(profile).lower(), (
        "8.3 short names are disabled on the temp volume (fsutil 8dot3name query); "
        "this check needs them")
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("USERNAME", "ybtester")
    er._rules.cache_clear()
    try:
        text = (f"{profile}\\notes.txt | {short_profile}\\notes.txt | "
                f"{short_local}\\Yamabiko\\x.log")
        assert er.redact_user(text) == (
            "%USERPROFILE%\\notes.txt | %USERPROFILE%\\notes.txt | %LOCALAPPDATA%\\Yamabiko\\x.log")
    finally:
        er._rules.cache_clear()


def test_real_windows_version_reads_the_display_version_and_build_from_the_registry():
    """The Windows field from HKLM ...\\Windows NT\\CurrentVersion:
    "<10|11> <DisplayVersion>, build <CurrentBuild>.<UBR>", its build the
    one Python reports.

    Seeded defect: a wrong key or value name -> the platform fallback
    ("11, build 10.0.26300") -> red."""
    value = er.windows_version()
    match = re.fullmatch(r"(10|11) (\w+), build (\d+)\.(\d+)", value)
    assert match, f"not the registry form: {value!r}"
    build = int(match[3])
    assert build == sys.getwindowsversion().build
    assert match[1] == ("11" if build >= 22000 else "10")


def test_real_log_tail_reads_the_handlers_file_and_drops_the_cut_first_line(monkeypatch, tmp_path):
    """The tail comes from the file of the RotatingFileHandler that the
    dialog logger's records reach (here on the "yamabiko" logger, as in the
    watchdog), read from its last 64 KiB: every kept line is whole, the last
    is the dialog line, and the user name is out.

    Seeded defects: read the head of the file, keep the cut first line of
    the 64 KiB window, look only at the dialog logger itself, or skip the
    redaction -> red."""
    monkeypatch.setenv("USERNAME", "ybtester")
    er._rules.cache_clear()
    path = tmp_path / "main-watchdog.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=10 * 1024 * 1024,
                                                   backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter(MAIN_LOG_FORMAT))
    parent = logging.getLogger("yamabiko")
    parent.addHandler(handler)
    old_level = parent.level
    parent.setLevel(logging.INFO)
    try:
        writer = logging.getLogger("yamabiko.capture.audio")
        for index in range(1200):  # about 120 KB: the tail starts mid-file
            writer.info("filler line %04d for the tail window, padded out to length", index)
        writer.info("profile at C:\\Users\\ybtester\\x")
        logging.getLogger("yamabiko.ui.error_dialog").error(
            "startup error dialog: %s | %s", "Some model files are missing", "Yamabiko can't start without them.")
        tail = er.read_log_tail(max_lines=100_000)
    finally:
        parent.removeHandler(handler)
        parent.setLevel(old_level)
        handler.close()
        er._rules.cache_clear()
    assert tail.path == path
    assert 300 < len(tail.lines) < 1200
    record = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} p\d+ \S+ \S+ (INFO|ERROR) ")
    assert all(record.match(line) for line in tail.lines), tail.lines[0]
    assert tail.lines[-1].endswith(
        "startup error dialog: Some model files are missing | Yamabiko can't start without them.")
    assert tail.lines[-2].endswith("profile at C:\\Users\\<user>\\x")
