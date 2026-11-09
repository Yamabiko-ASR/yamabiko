"""The startup error window's report data (yamabiko.ui.error_report), Qt-free.

Redaction of every path form and the bare user name, the Windows field's
major version, the GPU field, the log tail's fallback file, the Error line,
the report text and the window's condensed log. The OS seams (registry,
GetShortPathNameW, nvidia-smi, a real log handler's file) have real tests in
test_real_error_report.py and test_real_gpu.py.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import types

import pytest

from yamabiko.gpu import GpuSummary
from yamabiko.ui import error_report as er
from yamabiko.ui.error_dialog import StartupError

_PROFILE = r"C:\Users\Tester Person"
_LOCAL = r"C:\Users\Tester Person\AppData\Local"


@pytest.fixture
def tester_env(monkeypatch):
    """A user "Tester Person" whose profile has the 8.3 short form TESTER~1
    (the short-name lookup is stubbed; the real one is the real test's)."""
    monkeypatch.setenv("LOCALAPPDATA", _LOCAL)
    monkeypatch.setenv("USERPROFILE", _PROFILE)
    monkeypatch.setenv("USERNAME", "Tester Person")
    shorts = {_PROFILE: r"C:\Users\TESTER~1", _LOCAL: r"C:\Users\TESTER~1\AppData\Local"}
    monkeypatch.setattr(er, "_short_path", shorts.get)
    er._rules.cache_clear()
    yield
    er._rules.cache_clear()


def test_redact_user_takes_the_user_out_of_every_path_form_and_the_bare_name(tester_env, monkeypatch):
    """L5: the LOCALAPPDATA folder before the profile folder, each in its
    long and 8.3 short form and with backslash, forward slash or doubled
    backslash separators, case-insensitive; any other \\Users\\<name>
    segment, even for a one-letter name (ruling M3); the bare user name in a
    log line, only for names of 2 or more characters. Another user's folder
    is left alone.

    Seeded defects: skip the short form, the forward slash or the doubled
    backslash (for either form: a separator alternation built for the long
    form only leaves the "short repr" line), replace the profile before
    LOCALAPPDATA (the first line would read
    %USERPROFILE%\\AppData\\Local\\...), drop the bare-name rule, or match
    case-sensitively -> red. For the one-letter name "a": drop the
    Users-segment rule (D:\\Users\\a stays) or the 2-character guard (the
    standalone "a" becomes <user>) -> red."""
    text = "\n".join([
        r"loaded C:\Users\Tester Person\AppData\Local\Yamabiko\models\x.bin",
        r"temp C:\Users\TESTER~1\AppData\Local\Temp\yb.tmp",
        r"short C:\Users\TESTER~1\Desktop\notes.txt",
        r"short repr 'C:\\Users\\TESTER~1\\AppData\\Local\\Temp\\x' and C:/Users/TESTER~1/Desktop/y",
        "desktop c:/users/tester person/Desktop/notes.txt",
        r"repr 'C:\\Users\\Tester Person\\AppData\\Local\\Yamabiko'",
        r"venv D:\Users\Tester Person\venv\python.exe",
        "signed in as tester person today",
        r"other C:\Users\Tester Personal\x",
    ])
    assert er.redact_user(text) == "\n".join([
        r"loaded %LOCALAPPDATA%\Yamabiko\models\x.bin",
        r"temp %LOCALAPPDATA%\Temp\yb.tmp",
        r"short %USERPROFILE%\Desktop\notes.txt",
        r"short repr '%LOCALAPPDATA%\\Temp\\x' and %USERPROFILE%/Desktop/y",
        "desktop %USERPROFILE%/Desktop/notes.txt",
        r"repr '%LOCALAPPDATA%\\Yamabiko'",
        r"venv D:\Users\<user>\venv\python.exe",
        "signed in as <user> today",
        r"other C:\Users\Tester Personal\x",
    ])
    monkeypatch.setenv("USERNAME", "a")
    er._rules.cache_clear()
    assert er.redact_user(r"venv D:\Users\a\venv, a file") == r"venv D:\Users\<user>\venv, a file"


@pytest.mark.parametrize("build, display, expected", [
    ("26300", "26H2", "11 26H2, build 26300.9818"),
    ("19045", "22H2", "10 22H2, build 19045.9818"),
])
def test_windows_version_takes_the_major_version_from_the_build(monkeypatch, build, display, expected):
    """L4: Windows 11's registry ProductName still says "Windows 10", so the
    major version comes from CurrentBuild (22000 and up is 11).

    Seeded defects: read ProductName, or put the threshold elsewhere -> red;
    drop UBR from the build -> red."""
    values = {"DisplayVersion": display, "CurrentBuild": build, "UBR": 9818,
              "ProductName": "Windows 10 Pro"}

    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        OpenKey=lambda root, path: _Key(),
        QueryValueEx=lambda key, name: (values[name], 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert er.windows_version() == expected


def test_gpu_field_names_the_card_vram_and_driver_or_says_none_reported(monkeypatch):
    """L4: "<name>, <GB rounded>, driver <version>"; without a card the
    mockup's "None reported by Windows".

    Seeded defects: MiB shown raw, GB truncated (16303 MiB is 15.9 GB),
    or the driver dropped -> red; no fallback -> red."""
    monkeypatch.setattr(er.gpu, "detect_gpu_summary",
                        lambda: GpuSummary("NVIDIA GeForce RTX 5080", 16303, "617.42"))
    assert er.gpu_field() == "NVIDIA GeForce RTX 5080, 16 GB, driver 617.42"
    monkeypatch.setattr(er.gpu, "detect_gpu_summary", lambda: None)
    assert er.gpu_field() == "None reported by Windows"


def test_read_log_tail_falls_back_to_the_newest_main_log(monkeypatch, tmp_path):
    """L7: with no RotatingFileHandler on the dialog logger's chain (a
    process that logs nowhere yet), the tail comes from the newest main-*.log
    in the logs folder; blank lines are dropped and only the last lines are
    kept.

    Seeded defects: take the oldest file, glob every *.log (the newer
    latency.log would win), or keep blank lines -> red."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("yamabiko.ui.error_dialog")
    while logger is not None:
        monkeypatch.setattr(logger, "handlers", [
            h for h in logger.handlers
            if not isinstance(h, logging.handlers.RotatingFileHandler)])
        logger = logger.parent
    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "main-watchdog.log"
    new = logs / "main-pipeline.log"
    other = logs / "latency.log"
    old.write_text("old 1\nold 2\n", encoding="utf-8")
    new.write_text("one\ntwo\n\nthree\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    os.utime(other, (3_000, 3_000))
    assert er.read_log_tail(max_lines=2) == er.LogTail(new, ["two", "three"])


def test_error_line_is_the_last_details_line_or_else_the_message():
    """L4: the Error field is the traceback's final "Type: message" line;
    with no details it falls back to the message.

    Seeded defects: take the first line ("Stage: ...") or a trailing blank
    line -> red; no fallback -> an empty Error field -> red."""
    err = StartupError(title="T", message="M", suggested_fix="F",
                       details="Stage: ASR init\n\nTraceback (most recent call last):\n"
                               "  File \"x.py\", line 1\nValueError: bad value\n\n")
    assert er.error_line(err) == "ValueError: bad value"
    assert er.error_line(StartupError(title="T", message="M", suggested_fix="F",
                                      details="")) == "M"


def test_report_text_lays_out_the_report_and_takes_the_user_out(tester_env):
    """L4: the report's exact layout (header, the error's texts, the facts,
    the raw log lines, the details), redacted as a whole: a user path in the
    details (or anywhere) never reaches the clipboard.

    Seeded defects: a dropped or renamed line, the log condensed instead of
    raw, the details not redacted, or the details' trailing newline kept
    (a classified traceback ends with one: the report would end with a blank
    line, ruling M4) -> red."""
    err = StartupError(title="Some model files are missing", message="Yamabiko can't start without them.",
                       suggested_fix="Delete it. Start again.",
                       details=r"Stage: ASR init" "\n" r"FileNotFoundError: C:\Users\Tester Person\x.gguf" "\n")
    facts = er.ReportFacts(
        version="0.2.0-beta", windows="11 26H2, build 26300.9818",
        gpu="NVIDIA GeForce RTX 5080, 16 GB, driver 617.42", python="3.12.10",
        error=r"FileNotFoundError: %USERPROFILE%\x.gguf",
        log=er.LogTail(None, ["2026-11-01 21:41:06,001 p1 MainThread yamabiko INFO hello"]),
    )
    assert er.report_text(err, facts) == (
        "Yamabiko startup error\n"
        "Title: Some model files are missing\n"
        "Message: Yamabiko can't start without them.\n"
        "What to try: Delete it. Start again.\n"
        "\n"
        "Yamabiko: 0.2.0-beta\n"
        "Windows: 11 26H2, build 26300.9818\n"
        "GPU: NVIDIA GeForce RTX 5080, 16 GB, driver 617.42\n"
        "Python: 3.12.10\n"
        "Error: FileNotFoundError: %USERPROFILE%\\x.gguf\n"
        "\n"
        "Recent log:\n"
        "2026-11-01 21:41:06,001 p1 MainThread yamabiko INFO hello\n"
        "\n"
        "Details:\n"
        "Stage: ASR init\n"
        "FileNotFoundError: %USERPROFILE%\\x.gguf\n"
    )


def test_collect_facts_shows_a_fallback_for_a_fact_it_cannot_read(monkeypatch):
    """Ruling I3: the facts are read before the window shows, and every
    caller exits 43 right after it; an exception out of a read (nvidia-smi
    output the decoder can't read, a registry or log failure the readers
    don't expect) would crash the pipeline into a restart of the same
    failure with no window at all. Each OS read gives its fallback instead.

    Seeded defect: collect_facts calls one of the readers unguarded ->
    RuntimeError escapes -> red."""
    def broken():
        raise RuntimeError("the read broke")

    monkeypatch.setattr(er.gpu, "detect_gpu_summary", broken)
    monkeypatch.setattr(er, "windows_version", broken)
    monkeypatch.setattr(er, "read_log_tail", broken)
    facts = er.collect_facts(StartupError(title="T", message="M", suggested_fix="F", details=""))
    assert (facts.windows, facts.gpu, facts.log) == ("unknown", er.NO_GPU, er.LogTail(None, []))
    assert facts.error == "M"


def test_condense_log_keeps_time_short_name_level_and_marks_the_exception_line():
    """L8: a record keeps its time, the last part of its logger name, its
    level and its message (a thread name may hold spaces; an empty message
    stays empty); a traceback's final "Type: message" line (even a type not
    ending in Error) and a line naming an exception type are exception
    lines; frames and other lines stay text, as they are.

    Seeded defects: keep the date or the full logger name, misparse a
    spaced thread name, miss "OfflineModeIsEnabled: ..." after a traceback,
    or mark "The above exception was ..." -> red."""
    log_line = er.LogLine
    lines = [
        "2026-11-01 21:41:06,123 p4242 MainThread yamabiko.capture.audio ERROR saved device not found",
        "2026-11-01 21:41:06,124 p4242 Dummy thread 1 yamabiko.ui.error_dialog INFO ",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "OfflineModeIsEnabled: Cannot reach the Hub",
        "The above exception was the direct cause of the following exception:",
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find an appropriate cached snapshot",
    ]
    assert er.condense_log(lines) == [
        log_line("record", "21:41:06", "audio", "ERROR", "saved device not found"),
        log_line("record", "21:41:06", "error_dialog", "INFO", ""),
        log_line("text", "", "", "", "Traceback (most recent call last):"),
        log_line("text", "", "", "", '  File "x.py", line 1, in <module>'),
        log_line("exception", "", "", "", "OfflineModeIsEnabled: Cannot reach the Hub"),
        log_line("text", "", "", "", "The above exception was the direct cause of the following exception:"),
        log_line("exception", "", "", "", lines[-1]),
    ]
