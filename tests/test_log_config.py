"""Rotating log setup for latency.log + the per-process main-*.log files.

Tests hit the real filesystem via tmp_path — same pattern as test_paths.py.
The RotatingFileHandler's size-based rollover is exercised with a small
cap so we can trigger rollover without writing megabytes.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import re
import sys
import threading
from pathlib import Path

import pytest

from yamabiko import log_config, paths


def test_setup_latency_log_returns_handle_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    handle = log_config.setup_latency_log(enabled=True)
    assert handle is not None
    handle.write("{}\n")
    handle.flush()
    handle.close()
    assert (tmp_path / "logs" / "latency.log").is_file()


def test_setup_latency_log_returns_none_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    handle = log_config.setup_latency_log(enabled=False)
    assert handle is None
    assert not (tmp_path / "logs" / "latency.log").exists()


def test_setup_main_log_attaches_rotating_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_main_log")
    logger.handlers.clear()
    log_config.setup_main_log(logger, level="INFO", filename="main.log")
    logger.info("hello")
    for h in logger.handlers:
        h.flush()
    assert (tmp_path / "logs" / "main.log").is_file()
    content = (tmp_path / "logs" / "main.log").read_text(encoding="utf-8")
    assert "hello" in content


def test_setup_main_log_writes_to_filename_kwarg_not_main_log(
    tmp_path, monkeypatch,
):
    """Fails red if: setup_main_log ignores ``filename`` and writes the
    shared main.log. The regression shape: all three processes (watchdog
    / pipeline / overlay) funnel into one file, and on Windows stdlib
    RotatingFileHandler.doRollover's os.rename raises PermissionError
    while any other process holds the file open -- every post-10MB emit
    drops its record and each retry shifts the backup chain until all
    backups are destroyed.

    One-line seeded defect: in setup_main_log, replace ``filename``
    with the literal "main.log" in the RotatingFileHandler path -- the
    role file is never created and the first assert goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_filename_kwarg")
    logger.handlers.clear()
    log_config.setup_main_log(
        logger, level="INFO", filename="main-watchdog.log",
    )
    logger.info("hello-from-role-file")
    for h in logger.handlers:
        h.flush()
    role_log = tmp_path / "logs" / "main-watchdog.log"
    logs_dir = tmp_path / "logs"
    assert role_log.is_file(), (
        f"main-watchdog.log missing; logs dir contains: "
        f"{[p.name for p in logs_dir.iterdir()] if logs_dir.is_dir() else 'nothing'}"
    )
    assert "hello-from-role-file" in role_log.read_text(encoding="utf-8")
    assert not (tmp_path / "logs" / "main.log").exists(), (
        "filename kwarg was given but the shared main.log was written; "
        "cross-process rotation contention is back"
    )


def test_setup_main_log_respects_existing_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_main_log_2")
    logger.handlers.clear()
    # Idempotent: calling twice doesn't add two file handlers.
    log_config.setup_main_log(logger, level="INFO", filename="main.log")
    log_config.setup_main_log(logger, level="INFO", filename="main.log")
    file_handlers = [h for h in logger.handlers
                     if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(file_handlers) == 1


def test_main_log_rolls_over_at_size_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_rollover")
    logger.handlers.clear()
    # Use a tiny cap so we can trigger rollover in the test.
    log_config.setup_main_log(logger, level="INFO", filename="main.log",
                              max_bytes=200, backup_count=2)
    # Write enough content to trigger at least one rollover.
    for i in range(50):
        logger.info("message %d padded with filler text to hit the cap", i)
    for h in logger.handlers:
        h.flush()
    assert (tmp_path / "logs" / "main.log").is_file()
    assert (tmp_path / "logs" / "main.log.1").is_file()


def test_latency_log_size_cap_is_documented_default(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    # The returned handle is a plain file, not a handler, because
    # main.py writes JSON lines directly. Rotation for latency.log
    # happens via a companion RotatingFileHandler if max_bytes is set.
    handle = log_config.setup_latency_log(
        enabled=True, max_bytes=100, backup_count=1,
    )
    assert handle is not None
    for i in range(20):
        handle.write('{"x": "padding padding padding padding"}\n')
    handle.flush()
    handle.close()
    # Rollover should have produced latency.log.1 at the size cap.
    log_dir = tmp_path / "logs"
    rolled = list(log_dir.glob("latency.log.*"))
    assert len(rolled) >= 1, f"no rollover occurred; files: {list(log_dir.iterdir())}"


def test_latency_rollover_failure_disables_log_instead_of_raising(
    tmp_path, monkeypatch,
):
    """Fails red if: an OSError inside _SizeRotatingTextIO._rollover
    escapes write(). latency_log.write sits on the subtitle hot path
    inside _one_iter, BEFORE manager.broadcast — an escaping rollover
    failure kills subtitle delivery for that segment and, after 10
    errors in 60 s, escalates the supervisor and stops the whole
    pipeline while translation is still succeeding.

    Trigger is the real Windows failure mode, no mocks: a backup file
    (latency.log.1) held open by another handle makes the rotation
    rename raise PermissionError (Windows cannot rename an open file).

    One-line seeded defect: remove the try/except around _rollover's
    body — the PermissionError propagates out of write() and the test
    goes red at the bare write call.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    log_path = tmp_path / "logs" / "latency.log"
    writer = log_config._SizeRotatingTextIO(
        log_path, max_bytes=64, backup_count=2,
    )
    # A backup from a "previous session", held open so the rotation
    # chain's rename(latency.log.1 -> latency.log.2) hits WinError 32.
    backup = tmp_path / "logs" / "latency.log.1"
    backup.write_text("old backup\n", encoding="utf-8")
    blocker = backup.open("r", encoding="utf-8")
    try:
        # Cross max_bytes so the next write triggers _rollover. The
        # write must NOT raise — rollover failure degrades to dropping
        # the latency log for this session (the silent-drop fallback).
        writer.write("x" * 100 + "\n")
        writer.write("post-failure write must be accepted too\n")
        writer.flush()
        writer.close()
    finally:
        blocker.close()


def test_null_writer_fallback_skips_rollover_reattempts(tmp_path):
    """Fails red if: the _NullWriter fallback still counts bytes, so
    after max_bytes of dropped writes _rollover fires and re-attempts
    the very open() that failed at construction — raising the OSError
    the fallback exists to survive.

    Real trigger, no mocks: the log path's parent is a regular FILE,
    so the constructor's mkdir raises OSError and installs _NullWriter;
    the same condition makes _rollover's reopen raise if it ever runs.

    One-line seeded defect: drop the _NullWriter short-circuit from
    write() — byte accounting resumes, _rollover fires at the cap,
    open() raises, red.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed", encoding="utf-8")
    writer = log_config._SizeRotatingTextIO(
        blocker / "latency.log", max_bytes=64, backup_count=2,
    )
    assert isinstance(writer._fh, log_config._NullWriter), (
        "constructor fallback did not install _NullWriter; test "
        "precondition broken"
    )
    # Far past max_bytes: every write must be silently dropped with no
    # rollover re-attempt (which would raise — the parent is a file).
    for _ in range(10):
        writer.write("y" * 32 + "\n")
    writer.flush()
    writer.close()


def test_latency_write_enospc_disables_sink_and_future_writes_drop(tmp_path):
    writer = log_config._SizeRotatingTextIO(
        tmp_path / "latency.log", max_bytes=1024, backup_count=1,
    )

    class _WriteFails:
        def write(self, _text):
            raise OSError(28, "disk full")

        def close(self):
            pass

    writer._fh = _WriteFails()
    assert writer.write("first\n") == len("first\n")
    assert isinstance(writer._fh, log_config._NullWriter)
    assert writer.write("second\n") == len("second\n")


def test_latency_flush_enospc_disables_sink_and_future_flushes_drop(tmp_path):
    writer = log_config._SizeRotatingTextIO(
        tmp_path / "latency.log", max_bytes=1024, backup_count=1,
    )

    class _FlushFails:
        def flush(self):
            raise OSError(28, "disk full")

        def close(self):
            pass

    writer._fh = _FlushFails()
    writer.flush()
    assert isinstance(writer._fh, log_config._NullWriter)
    writer.flush()


def test_setup_main_log_no_double_write_via_hierarchy_propagation(
    tmp_path, monkeypatch
):
    """Fails red if: attaching handlers to BOTH a logger and its parent
    (the exact shape of the --run-main child, where main.py:1017
    configures the root logger and main.py:652 configures the
    ``yamabiko`` logger in the same process) causes every
    ``yamabiko.xxx`` log line to be written twice to main.log via
    propagation.

    One-line seeded defect: reintroduce the unguarded ``addHandler``
    branch in ``setup_main_log`` for the child-logger case -- the line
    ``[2026-04-21 ... hello-from-yamabiko-child]`` appears twice, assert
    goes red (count == 2).

    The bug class is 'double handlers across loggers in the hierarchy' --
    NOT covered by test_setup_main_log_respects_existing_handlers (which
    only proves same-logger dedup).
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    loggers = [
        logging.getLogger(name) for name in ("", "yamabiko", "yamabiko.child")
    ]
    # Hermetic: snapshot pre-test handler lists + propagate flags, nuke
    # them for the duration of the test, restore in the finally so
    # subsequent tests in this file aren't affected (charter §3.6).
    saved = [(lg, list(lg.handlers), lg.propagate) for lg in loggers]
    for lg in loggers:
        for h in list(lg.handlers):
            lg.removeHandler(h)
    try:
        root, yamabiko_logger, child = loggers
        # Reproduce the --run-main call order:
        #   main.py:1017  ->  setup_main_log(root)
        #   main.py:652   ->  setup_main_log(yamabiko)  (via build_pipeline)
        log_config.setup_main_log(root, level="INFO", filename="main.log")
        log_config.setup_main_log(yamabiko_logger, level="INFO",
                                  filename="main.log")

        child.info("hello-from-yamabiko-child")
        for lg in (root, yamabiko_logger, child):
            for h in lg.handlers:
                h.flush()

        main_log = tmp_path / "logs" / "main.log"
        assert main_log.is_file(), f"main.log not created under {tmp_path}"
        content = main_log.read_text(encoding="utf-8")
        hit_count = content.count("hello-from-yamabiko-child")
        assert hit_count == 1, (
            f"log line written {hit_count}x to main.log; expected exactly 1. "
            f"Double-write via hierarchy propagation. "
            f"main.log content:\n{content}"
        )
    finally:
        # Restore original handler + propagate state.
        for lg, handlers, propagate in saved:
            for h in list(lg.handlers):
                lg.removeHandler(h)
            for h in handlers:
                lg.addHandler(h)
            lg.propagate = propagate


def test_setup_main_log_attaches_when_ancestor_handler_is_unreachable(
    tmp_path, monkeypatch
):
    """Fails red if: _ancestor_has_rotating_file_handler reports True
    even when intermediate ``propagate=False`` breaks the chain, so
    setup_main_log skips attachment and records silently vanish.

    One-line seeded defect: delete the ``if not cur.propagate: return
    False`` branch from _ancestor_has_rotating_file_handler -- the
    handler-less ``yamabiko`` logger swallows the record (root's handler
    is unreachable via propagation), assert goes red (count == 0).

    Named scenario: defensive regression guard for the assumption that
    propagation is intact between the caller and the ancestor-handler
    holder. No production caller flips propagate today, but the check
    keeps setup_main_log safe to use if one ever does.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    loggers = [
        logging.getLogger(name) for name in ("", "yamabiko", "yamabiko.child")
    ]
    saved = [(lg, list(lg.handlers), lg.propagate) for lg in loggers]
    for lg in loggers:
        for h in list(lg.handlers):
            lg.removeHandler(h)
    try:
        root, yamabiko_logger, child = loggers
        log_config.setup_main_log(root, level="INFO", filename="main.log")
        # Break propagation between yamabiko and root. Now yamabiko's
        # records would NOT reach root's handler. setup_main_log on
        # yamabiko MUST therefore attach a local handler rather than
        # skip.
        yamabiko_logger.propagate = False
        log_config.setup_main_log(yamabiko_logger, level="INFO",
                                  filename="main.log")

        import logging.handlers as _lh
        yamabiko_handlers = [
            h for h in yamabiko_logger.handlers
            if isinstance(h, _lh.RotatingFileHandler)
        ]
        assert len(yamabiko_handlers) == 1, (
            f"yamabiko logger has {len(yamabiko_handlers)} file handler(s); "
            f"expected 1. _ancestor_has_rotating_file_handler skipped the "
            f"attach despite propagate=False breaking the chain."
        )

        child.info("hello-via-yamabiko-only")
        for lg in (root, yamabiko_logger, child):
            for h in lg.handlers:
                h.flush()

        main_log = tmp_path / "logs" / "main.log"
        content = main_log.read_text(encoding="utf-8") if main_log.is_file() else ""
        hit_count = content.count("hello-via-yamabiko-only")
        assert hit_count == 1, (
            f"log line written {hit_count}x; expected exactly 1. "
            f"With propagate=False on yamabiko, only yamabiko's own "
            f"handler should fire."
        )
    finally:
        for lg, handlers, propagate in saved:
            for h in list(lg.handlers):
                lg.removeHandler(h)
            for h in handlers:
                lg.addHandler(h)
            lg.propagate = propagate


def test_setup_main_log_applies_level_when_frozen_dedup_skips_handlers(
    tmp_path, monkeypatch, hermetic_log_handlers,
):
    """Fails red if: the frozen-bundle early-return in setup_main_log
    fires BEFORE logger.setLevel(level), silently ignoring the user's
    ``logging.level`` override.

    Frozen pipeline child call order (main.py main() then
    build_pipeline): root gets the file handler at level=INFO first,
    then the ``yamabiko`` logger is configured with the user's level.
    With sys.frozen set, want_stream=False, and root's handler visible
    via _ancestor_has_rotating_file_handler, the second call's ONLY
    remaining job is the setLevel — the dedup early-return must not
    skip it. Dev mode masked this because want_stream=True avoids the
    early-return, the exact dev-vs-frozen divergence.

    One-line seeded defect: move ``logger.setLevel(level)`` back below
    the ``if has_file and (...): return`` early-return — the yamabiko
    logger stays NOTSET, inherits root's INFO, and the DEBUG assert
    goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    with hermetic_log_handlers("", "yamabiko"):
        root = logging.getLogger()
        yamabiko_logger = logging.getLogger("yamabiko")
        log_config.setup_main_log(root, level="INFO",
                                  filename="main-pipeline.log")
        log_config.setup_main_log(yamabiko_logger, level="DEBUG",
                                  filename="main-pipeline.log")

        assert yamabiko_logger.getEffectiveLevel() == logging.DEBUG, (
            f"yamabiko logger effective level is "
            f"{logging.getLevelName(yamabiko_logger.getEffectiveLevel())}, "
            f"expected DEBUG. The frozen-mode early-return skipped "
            f"setLevel, so the user's logging.level override is dead in "
            f"the shipped bundle."
        )
        # The dedup behaviour itself must survive the fix: still no
        # second file handler on the yamabiko logger.
        own_file_handlers = [
            h for h in yamabiko_logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert not own_file_handlers, (
            "the level fix must not regress the dedup: yamabiko got "
            "its own RotatingFileHandler despite root's reachable one "
            "(double-write returns)"
        )


def test_setup_main_log_attaches_stream_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_stream_handler")
    logger.handlers.clear()
    log_config.setup_main_log(logger, level="INFO", filename="main.log")
    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 1
    # Idempotent on the stream handler too:
    log_config.setup_main_log(logger, level="INFO", filename="main.log")
    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 1


# ---------------------------------------------------------------------------
# Log-level normalization: single-source coercion so no consumer's
# logger.setLevel ever sees a raw config value it would reject.
# ---------------------------------------------------------------------------


def test_normalize_log_level_uppercases_known_lowercase_name():
    """'debug' -> ('DEBUG', None): a valid stdlib level in any case is
    accepted and returns no warning.

    Seeded defect: drop the ``.upper()`` in normalize_log_level -> 'debug'
    is not in the name set -> falls back to ('INFO', <warning>) -> red.
    """
    assert log_config.normalize_log_level("debug") == ("DEBUG", None)


def test_normalize_log_level_unknown_falls_back_to_info_with_warning():
    """An unknown level name returns INFO plus a warning message naming
    the offending value (the caller controls emission timing).

    Seeded defect: return ``(name, None)`` unconditionally -> the warning
    is None -> red.
    """
    name, warning = log_config.normalize_log_level("verbose")
    assert name == "INFO"
    assert warning is not None and "verbose" in warning


def test_normalize_log_level_none_falls_back_to_info_silently():
    """None (no configured level) -> INFO with no warning: an absent key
    is not an error."""
    assert log_config.normalize_log_level(None) == ("INFO", None)


def test_normalize_log_level_maps_stdlib_aliases_warn_and_fatal():
    """``warn`` -> WARNING and ``FATAL`` -> CRITICAL with NO junk
    warning. Both are stdlib-registered aliases ``logger.setLevel``
    accepts, and the pre-normalization pipeline honored them (the raw
    config value went straight to setLevel) -- so rejecting them as
    junk silently downgraded a user's WARN config to INFO.

    Seeded defect: drop the ``_LOG_LEVEL_ALIASES`` lookup in
    normalize_log_level -> 'warn' degrades to ('INFO', <junk warning>)
    -> both asserts red.
    """
    assert log_config.normalize_log_level("warn") == ("WARNING", None)
    assert log_config.normalize_log_level("FATAL") == ("CRITICAL", None)


def test_setup_main_log_coerces_junk_level_to_info_without_raising(
    tmp_path, monkeypatch,
):
    """setup_main_log must coerce a junk config level to a real stdlib
    level BEFORE logger.setLevel, so a ``logging.level: verbose`` override
    cannot abort startup: setLevel('verbose') raises ValueError.

    Seeded defect: pass ``level`` straight to logger.setLevel (drop the
    normalize_log_level coercion) -> setLevel('verbose') raises
    ValueError -> the call below raises -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_junk_level")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    # Must not raise:
    log_config.setup_main_log(
        logger, level="verbose", filename="main-junk.log",
    )
    assert logger.level == logging.INFO, (
        f"junk level did not coerce to INFO; logger.level={logger.level}"
    )


# ---------------------------------------------------------------------------
# Record format: pid + thread name (logging-hardening Task 9)
# ---------------------------------------------------------------------------


def test_main_log_records_carry_pid_and_thread_name(tmp_path, monkeypatch):
    """Fails red if: the main-log record layout drops the process id or
    the thread name. Multi-process triage (watchdog + pipeline + overlay
    logs read side by side, respawned PIDs after a crash-loop, keyboard/
    supervisor threads vs MainThread) depends on both fields being on
    every record.

    One-line seeded defect: revert MAIN_LOG_FORMAT to the pre-hardening
    "%(asctime)s %(name)s %(levelname)s %(message)s" -- the regex below
    finds no p<pid>/thread fields, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logger = logging.getLogger("test_fmt_pid_thread")
    logger.handlers.clear()
    log_config.setup_main_log(logger, level="INFO", filename="main-fmt.log")
    logger.info("fmt-probe")
    for h in logger.handlers:
        h.flush()
    content = (tmp_path / "logs" / "main-fmt.log").read_text(encoding="utf-8")
    assert re.search(
        rf"p{os.getpid()} MainThread test_fmt_pid_thread INFO fmt-probe",
        content,
    ), f"pid/thread fields missing from record: {content!r}"


def test_overlay_build_passes_main_log_format_to_basic_config(monkeypatch):
    """The overlay's console basicConfig and the file handlers must use
    ONE record layout (the E2E _LogTail merges main-*.log across
    processes; two layouts break line-oriented triage). Structural
    "module namespace HAS the constant" checks are a Liar here (charter
    §3.5): the import can be wired while the basicConfig CALL inlines a
    different literal. So this test invokes the overlay's real
    logging-setup path -- build_overlay up to and including its
    basicConfig call -- with logging.basicConfig swapped for a recorder
    that captures the kwargs and then short-circuits the rest of
    build_overlay (Qt construction, file handlers, crash hooks: not this
    test's subject).

    One-line seeded defect: replace the overlay's
    ``format=MAIN_LOG_FORMAT`` with an inline copy of the OLD format
    string ("%(asctime)s %(name)s %(levelname)s %(message)s") -- the
    identity assert on the recorded kwarg goes red.
    """
    overlay_main = pytest.importorskip("yamabiko.overlay_app.__main__")

    class _StopAfterBasicConfig(Exception):
        pass

    recorded: list[dict] = []

    def _recording_basic_config(**kwargs):
        recorded.append(kwargs)
        raise _StopAfterBasicConfig

    monkeypatch.setattr(logging, "basicConfig", _recording_basic_config)

    with pytest.raises(_StopAfterBasicConfig):
        overlay_main.build_overlay([])

    assert recorded, "build_overlay never called logging.basicConfig"
    assert recorded[0].get("format") is log_config.MAIN_LOG_FORMAT, (
        f"overlay basicConfig format kwarg is {recorded[0].get('format')!r}, "
        f"not the shared log_config.MAIN_LOG_FORMAT constant; the overlay "
        f"console layout can drift from the main-*.log file layout"
    )


# ---------------------------------------------------------------------------
# Unwritable-logs-dir fallback (logging-hardening Task 9)
# ---------------------------------------------------------------------------


def test_setup_main_log_falls_back_to_temp_dir_when_logs_dir_unwritable(
    tmp_path, monkeypatch, hermetic_log_handlers,
):
    """Fails red if: an OSError from the primary logs-dir open leaves the
    process with NO file handler at all (the pre-hardening behaviour:
    stderr warning only, which the windowed bundle discards). The retry
    must land the handler under %TEMP%\\Yamabiko-logs\\<filename> and
    record the redirection into the fallback file itself, so whoever
    finds the file knows why it isn't in the normal logs dir.

    One-line seeded defect: delete the fallback retry inside
    setup_main_log's except OSError branch (restore the print-and-give-up
    body) -- no fallback file is created, first assert red.
    """
    import tempfile

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    fake_temp = tmp_path / "faketemp"
    fake_temp.mkdir()
    # tempfile.gettempdir() reads (and caches into) tempfile.tempdir;
    # patching the attribute is the reliable per-test redirect.
    monkeypatch.setattr(tempfile, "tempdir", str(fake_temp))

    def _unwritable_logs_dir():
        raise OSError("logs dir unwritable (simulated ACL deny)")

    monkeypatch.setattr(log_config.paths, "logs_dir", _unwritable_logs_dir)

    with hermetic_log_handlers("", "test_fallback_dir"):
        logger = logging.getLogger("test_fallback_dir")
        log_config.setup_main_log(
            logger, level="INFO", filename="main-fallback.log",
        )
        logger.info("post-fallback-probe")
        for h in logger.handlers:
            h.flush()

        fallback_file = fake_temp / "Yamabiko-logs" / "main-fallback.log"
        assert fallback_file.is_file(), (
            f"no fallback log under {fake_temp / 'Yamabiko-logs'}; "
            f"an unwritable logs dir still means zero file evidence"
        )
        content = fallback_file.read_text(encoding="utf-8")
        assert "post-fallback-probe" in content, (
            "fallback handler attached but records don't reach it"
        )
        assert "redirected" in content, (
            "the redirection itself must be logged into the fallback "
            f"file; content: {content!r}"
        )


# ---------------------------------------------------------------------------
# Crash-evidence bootstrap (logging-hardening Task 1)
# ---------------------------------------------------------------------------


# crash_evidence_guard (snapshot/restore of excepthooks + faulthandler +
# the module-global registries) lives in tests/conftest.py -- one shared
# fixture instead of per-file copies.


def test_install_crash_evidence_excepthook_logs_critical_and_chains(
    tmp_path, monkeypatch, caplog, crash_evidence_guard,
):
    """The installed sys.excepthook must (a) write a CRITICAL record with
    the traceback into the logging tree (that's the whole point: an
    unhandled main-thread exception currently dies onto the discarded
    frozen stderr), and (b) chain to the PREVIOUS hook so nothing an
    earlier layer installed (pytest, dev console printing) is lost.

    One-line seeded defect: delete the ``prev_sys_hook(...)`` chain call
    inside the installed hook -- the prior-hook assert goes red. Second
    seeded defect: drop the log.critical call -- the CRITICAL-record
    assert goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    prev_calls: list[tuple] = []
    monkeypatch.setattr(
        sys, "excepthook", lambda *a: prev_calls.append(a),
    )

    log_config.install_crash_evidence("unittest-hook")
    exc = ValueError("boom-excepthook")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        sys.excepthook(ValueError, exc, None)

    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert crit, "no CRITICAL record logged by the installed sys.excepthook"
    assert "unhandled exception" in crit[0].getMessage()
    assert "boom-excepthook" in caplog.text, (
        "the exception itself (exc_info) is missing from the record"
    )
    assert prev_calls == [(ValueError, exc, None)], (
        f"previous sys.excepthook not chained; got {prev_calls!r}"
    )


def test_install_crash_evidence_threading_excepthook_logs_and_chains(
    tmp_path, monkeypatch, caplog, crash_evidence_guard,
):
    """A crash on a non-main thread (keyboard hotkey thread, overlay
    supervisor, capture monitor) must land a CRITICAL record in the log
    AND still reach the previous threading.excepthook. Exercised through
    a REAL thread raising, not by calling the hook by hand.

    One-line seeded defect: delete the ``prev_thread_hook(args)`` chain
    call -- the prior-hook assert goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    prev_calls: list = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: prev_calls.append(args),
    )

    log_config.install_crash_evidence("unittest-thook")

    def _boom():
        raise RuntimeError("boom-thread-hook")

    with caplog.at_level(logging.INFO, logger="yamabiko"):
        t = threading.Thread(target=_boom, name="boom-thread")
        t.start()
        t.join(5.0)

    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert crit, "no CRITICAL record logged by the installed threading.excepthook"
    assert "boom-thread" in crit[0].getMessage(), (
        f"record does not name the crashing thread: {crit[0].getMessage()!r}"
    )
    assert "boom-thread-hook" in caplog.text
    assert len(prev_calls) == 1 and prev_calls[0].exc_type is RuntimeError, (
        f"previous threading.excepthook not chained; got {prev_calls!r}"
    )


def test_install_crash_evidence_creates_crash_role_log_and_enables_faulthandler(
    tmp_path, monkeypatch, crash_evidence_guard,
):
    """faulthandler must be armed against an explicit crash-<role>.log
    handle (default sys.stderr is a NullWriter in the windowed bundle
    and raises -- pyinstaller#4481), so a native access violation has
    somewhere to write its stack.

    Self-sufficiency: pytest's builtin faulthandler plugin pre-enables
    faulthandler for the whole session, so without the explicit
    disable() below the is_enabled assert would pass even if
    install_crash_evidence never called enable (order-dependent Liar --
    it only went red when an earlier test's teardown happened to
    disable it). Start from a known-off state; the conftest
    crash_evidence_guard restores the prior state at teardown.

    One-line seeded defect: drop the ``faulthandler.enable(file=fh)``
    call -- the is_enabled assert goes red, including when this node
    runs in isolation.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    faulthandler.disable()
    assert not faulthandler.is_enabled(), "precondition: faulthandler off"
    log_config.install_crash_evidence("roleA")
    assert (tmp_path / "logs" / "crash-roleA.log").is_file(), (
        "crash-roleA.log was not created"
    )
    assert faulthandler.is_enabled(), (
        "faulthandler is not enabled after install_crash_evidence"
    )
    assert "roleA" in log_config._CRASH_LOG_HANDLES, (
        "the crash-log handle is not parked module-globally; it can be "
        "GC-closed while faulthandler still points at it"
    )


def test_install_crash_evidence_rotates_oversize_crash_log_at_open(
    tmp_path, monkeypatch, crash_evidence_guard,
):
    """A crash-log over 5 MB at open must be os.replace'd to
    crash-<role>.log.1 first, so a crash-looping install can't grow the
    file without bound (crash logs have no runtime rotation -- the
    process is dead when they're written).

    One-line seeded defect: delete the size check / os.replace branch --
    the .1 file never appears, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    crash = tmp_path / "logs" / "crash-roleB.log"
    crash.parent.mkdir(parents=True, exist_ok=True)
    crash.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    log_config.install_crash_evidence("roleB")

    rolled = tmp_path / "logs" / "crash-roleB.log.1"
    assert rolled.is_file(), "oversize crash log was not rotated to .1"
    assert rolled.stat().st_size > 5 * 1024 * 1024
    assert crash.stat().st_size < 1024, (
        "crash-roleB.log should be fresh (near-empty) after rotation"
    )


def test_install_crash_evidence_survives_unwritable_logs_dir(
    tmp_path, monkeypatch, caplog, crash_evidence_guard,
):
    """install_crash_evidence must NEVER raise: crash evidence can't be
    allowed to become the crash. With logs_dir unavailable the
    faulthandler layer degrades (warning logged) but the excepthooks --
    which need no file -- still install.

    One-line seeded defect: remove the try/except around the
    faulthandler layer -- the simulated OSError escapes, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    def _boom_logs_dir():
        raise OSError("logs dir gone (simulated)")

    monkeypatch.setattr(log_config.paths, "logs_dir", _boom_logs_dir)
    hook_before = sys.excepthook
    monkeypatch.setattr(sys, "excepthook", hook_before)  # restore point

    with caplog.at_level(logging.DEBUG, logger="yamabiko"):
        log_config.install_crash_evidence("roleC")  # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("faulthandler" in r.getMessage() for r in warnings), (
        "faulthandler setup failure was swallowed without a warning"
    )
    assert sys.excepthook is not hook_before, (
        "sys.excepthook was not installed although only the faulthandler "
        "layer failed"
    )


def test_install_crash_evidence_is_idempotent_per_role(
    tmp_path, monkeypatch, crash_evidence_guard,
):
    """A repeat call for the same role must be a no-op (no hook
    stacking): test harnesses rebuild the overlay/pipeline repeatedly in
    one process, and stacked hooks would log one CRITICAL per layer for
    a single crash.

    One-line seeded defect: remove the _INSTALLED_ROLES early return --
    sys.excepthook is replaced again on the second call, red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    log_config.install_crash_evidence("roleD")
    hook_after_first = sys.excepthook
    log_config.install_crash_evidence("roleD")
    assert sys.excepthook is hook_after_first, (
        "second install for the same role stacked another excepthook layer"
    )


# ---------------------------------------------------------------------------
# Startup identity banner (logging-hardening Task 5)
# ---------------------------------------------------------------------------


def test_log_startup_banner_emits_role_version_pid_python_platform(caplog):
    """One INFO line with every identity field a bug report needs:
    role, yamabiko version, frozen|dev, PID, python version, OS. The
    bug_report.yml template points reporters at this line for the
    version field.

    One-line seeded defect: drop any single field (e.g. the pid= part)
    from the banner f-string -- the regex goes red.
    """
    from yamabiko import __version__

    logger = logging.getLogger("yamabiko.test_banner")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        log_config.log_startup_banner(logger, "overlay")

    banners = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("yamabiko overlay")
    ]
    assert banners, "no banner line logged"
    line = banners[0]
    assert __version__ in line, f"version missing from banner: {line!r}"
    assert re.search(
        r"^yamabiko overlay v\S+ \((dev|frozen)\) pid=\d+ "
        r"python=\d+\.\d+\.\d+ os=.+",
        line,
    ), f"banner fields malformed: {line!r}"


def test_log_startup_banner_appends_config_digest_when_given(caplog):
    """The pipeline passes a config digest; it must appear on the SAME
    banner line (one grep hit per process identity).

    One-line seeded defect: drop the ``config=`` append branch -- red.
    """
    logger = logging.getLogger("yamabiko.test_banner_digest")
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        log_config.log_startup_banner(
            logger, "pipeline", config_digest="abcdef123456 overrides=[genre]",
        )
    banners = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("yamabiko pipeline")
    ]
    assert banners, "no banner line logged"
    assert "config=abcdef123456 overrides=[genre]" in banners[0]


def test_log_startup_banner_gpu_line_for_pipeline_and_watchdog_not_overlay(
    monkeypatch, caplog,
):
    """The GPU line is emitted for the watchdog + pipeline roles only
    (they own CUDA work; the overlay doesn't touch the GPU and its
    banner must stay one line + zero nvidia-smi shell-outs).

    One-line seeded defect: change the role gate to include "overlay"
    (or drop the gate) -- the overlay-absence assert goes red.
    """
    import yamabiko.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "detect_gpu_name", lambda: "FakeRTX 9090")
    logger = logging.getLogger("yamabiko.test_banner_gpu")

    for role in ("pipeline", "watchdog"):
        caplog.clear()
        monkeypatch.setattr(
            log_config, "_gpu_name_cache", log_config._GPU_NAME_UNSET,
        )
        with caplog.at_level(logging.INFO, logger="yamabiko"):
            log_config.log_startup_banner(logger, role)
        assert any(
            r.getMessage() == "gpu: FakeRTX 9090" for r in caplog.records
        ), f"gpu line missing for role {role}"

    caplog.clear()
    monkeypatch.setattr(
        log_config, "_gpu_name_cache", log_config._GPU_NAME_UNSET,
    )
    with caplog.at_level(logging.INFO, logger="yamabiko"):
        log_config.log_startup_banner(logger, "overlay")
    assert not any(
        r.getMessage().startswith("gpu:") for r in caplog.records
    ), "overlay banner must not emit a gpu line"
