"""collect_diagnostics() tests.

The diagnostics zip is the one artifact a beta user can hand over when
something breaks, so these tests exercise the REAL zipfile + filesystem
path against planted log files (observable disk state, not mock call
counts) and pin the two properties the plan makes non-negotiable:

  * privacy -- override KEYS land in the redacted snapshot, override
    VALUES never do (values can hold endpoints/paths = PII);
  * resilience -- the collector never raises into its caller (the tray
    click handler must not detonate the overlay because the Desktop was
    unwritable or one log was locked mid-rotation).

App-data isolation is via YAMABIKO_APPDATA (the paths.py test seam).
"""
from __future__ import annotations

import logging
import os
import platform
import re
import sys
import zipfile
from pathlib import Path

import pytest

from yamabiko import __version__


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    """Isolated %LOCALAPPDATA%\\Yamabiko stand-in with planted logs.

    Plants one of each artifact class the zip must carry (live log,
    rotated backup, faulthandler crash log, llama-server capture), one
    file that must NOT be shipped, and a sparse user_overrides.yaml
    whose values are distinctive sentinels so a privacy leak is
    unambiguous in the assertions.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    # newline="" so Windows text mode doesn't translate \n -> \r\n;
    # the byte-faithful round-trip assertion depends on exact content.
    (logs / "main-watchdog.log").write_text(
        "watchdog line\n", encoding="utf-8", newline="",
    )
    (logs / "main-pipeline.log.1").write_text(
        "rotated pipeline line\n", encoding="utf-8"
    )
    (logs / "crash-pipeline.log").write_text(
        "Windows fatal exception: access violation\n", encoding="utf-8"
    )
    (logs / "llama-server.log").write_text("llama boot line\n", encoding="utf-8")
    (logs / "notes.txt").write_text("not a log\n", encoding="utf-8")
    (tmp_path / "user_overrides.yaml").write_text(
        "translation:\n"
        "  endpoint: http://sekrit-host:9999\n"
        "overlay:\n"
        "  jp_size: 424242\n",
        encoding="utf-8",
    )
    return tmp_path


def test_collect_diagnostics_zips_all_planted_log_files(appdata, tmp_path):
    """The bundle carries every logs_dir() artifact class -- live logs,
    rotated ``.log.N`` backups, ``crash-*.log`` faulthandler evidence,
    and the ``llama-server.log`` child capture -- and nothing else.

    Seeded defect: narrow the glob from ``*.log*`` to ``*.log`` (drops
    rotated backups) -> the main-pipeline.log.1 assertion goes red.
    """
    from yamabiko.diagnostics import collect_diagnostics

    out = collect_diagnostics(dest_dir=tmp_path / "out")
    assert out.is_file(), f"no zip written at {out}"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "main-watchdog.log" in names
        assert "main-pipeline.log.1" in names   # rotated backups included
        assert "crash-pipeline.log" in names    # faulthandler evidence
        assert "llama-server.log" in names      # child-output capture
        assert "notes.txt" not in names         # only log files are shipped
        # Content round-trip, not just a name listing.
        assert zf.read("main-watchdog.log").decode("utf-8") == "watchdog line\n"


def test_snapshot_carries_override_keys_and_env_info_but_never_values(appdata, tmp_path):
    """The redacted snapshot names the version, banner-style env info,
    and the dotted override KEYS -- and none of the override VALUES.

    Seeded defect: dump the override values (not just keys) into the
    snapshot -> the sekrit-host / 424242 assertions go red.
    """
    from yamabiko.diagnostics import collect_diagnostics

    out = collect_diagnostics(dest_dir=tmp_path / "out")
    with zipfile.ZipFile(out) as zf:
        text = zf.read("diagnostics.txt").decode("utf-8")
    assert __version__ in text
    assert platform.python_version() in text
    assert platform.platform() in text
    assert "translation.endpoint" in text
    assert "overlay.jp_size" in text
    assert "sekrit-host" not in text, f"override VALUE leaked:\n{text}"
    assert "424242" not in text, f"override VALUE leaked:\n{text}"


def test_collect_diagnostics_defaults_to_desktop_with_timestamped_name(
    appdata, tmp_path, monkeypatch,
):
    """dest_dir=None resolves via the Desktop helper, and the zip name is
    ``yamabiko-diagnostics-<ts>.zip`` so repeated collections never
    clobber each other.

    Seeded defect: hardcode the filename to ``yamabiko-diagnostics.zip``
    -> the timestamp-pattern assertion goes red.
    """
    from yamabiko import diagnostics

    fake_desktop = tmp_path / "Desktop"
    fake_desktop.mkdir()
    monkeypatch.setattr(diagnostics, "_desktop_dir", lambda: fake_desktop)
    out = diagnostics.collect_diagnostics()
    assert out.parent == fake_desktop
    assert re.fullmatch(r"yamabiko-diagnostics-\d{8}-\d{6}\.zip", out.name), (
        f"zip name not timestamped: {out.name}"
    )
    assert out.is_file()


def test_collect_diagnostics_swallows_unwritable_dest_and_logs_it(
    appdata, tmp_path, caplog,
):
    """Never raises into the caller: an unwritable destination (here a
    dest_dir nested under a regular FILE) is logged at ERROR and the
    function still returns the intended path (which the caller can
    ``exists()``-check).

    Seeded defect: delete the outer try/except in collect_diagnostics ->
    the OSError propagates out of the call and this test errors (red).
    """
    from yamabiko.diagnostics import collect_diagnostics

    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="yamabiko.diagnostics"):
        out = collect_diagnostics(dest_dir=blocker / "nested")
    assert not out.exists()
    assert any(
        "diagnostics collection failed" in r.getMessage() for r in caplog.records
    ), "failure was swallowed WITHOUT log evidence"


def test_collect_diagnostics_skips_unreadable_log_and_keeps_the_rest(
    appdata, tmp_path, monkeypatch, caplog,
):
    """A single locked/unreadable log (AV lock, mid-rotation rename) must
    not abort the whole bundle: the offending file is skipped with a
    warning; every other log plus the snapshot still lands.

    Seeded defect: remove the per-file try/except inside the zip loop ->
    the injected OSError aborts the loop before diagnostics.txt is
    written -> the snapshot assertion goes red.
    """
    orig_write = zipfile.ZipFile.write

    def _locked_write(self, filename, arcname=None, *args, **kwargs):
        if Path(filename).name == "main-watchdog.log":
            raise OSError("file locked by another process")
        return orig_write(self, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", _locked_write)
    from yamabiko.diagnostics import collect_diagnostics

    with caplog.at_level(logging.WARNING, logger="yamabiko.diagnostics"):
        out = collect_diagnostics(dest_dir=tmp_path / "out")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "main-watchdog.log" not in names
    assert "llama-server.log" in names, "healthy log lost to a sibling's lock"
    assert "diagnostics.txt" in names, "snapshot lost to a locked log file"
    assert any(
        "main-watchdog.log" in r.getMessage() for r in caplog.records
    ), "skipped file left no log evidence"


def test_collect_diagnostics_skips_pre_1980_mtime_log_and_keeps_the_rest(
    appdata, tmp_path, caplog,
):
    """zipfile (strict_timestamps default) raises ValueError -- not
    OSError -- for any file with an mtime before 1980 (dead CMOS
    battery, bogus file-copy metadata). One such log must not abort
    the whole bundle: it is skipped with a warning; every other log
    plus the snapshot still lands.

    Seeded defect: narrow the per-file guard back to ``except OSError``
    -> the ValueError escapes the loop before diagnostics.txt is
    written -> the snapshot assertion goes red.
    """
    ancient = appdata / "logs" / "ancient-crash.log"
    ancient.write_text("old evidence\n", encoding="utf-8")
    # 100_000_000 s epoch = 1973 -- before the ZIP format's 1980 floor.
    os.utime(ancient, (100_000_000, 100_000_000))

    from yamabiko.diagnostics import collect_diagnostics

    with caplog.at_level(logging.WARNING, logger="yamabiko.diagnostics"):
        out = collect_diagnostics(dest_dir=tmp_path / "out")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "diagnostics.txt" in names, "snapshot lost to a pre-1980 mtime log"
    assert "main-watchdog.log" in names, (
        "healthy log lost to a sibling's bad mtime"
    )
    assert "ancient-crash.log" not in names
    assert any(
        "ancient-crash.log" in r.getMessage() for r in caplog.records
    ), "skipped file left no log evidence"


def test_collect_diagnostics_removes_partial_zip_when_write_fails_midway(
    appdata, tmp_path, monkeypatch, caplog,
):
    """A failure AFTER the ZipFile was created (disk full while
    deflating; here a seeded ENOSPC on the snapshot writestr) must not
    leave the partially-written zip on disk: the contract callers rely
    on is "on failure the returned path won't exist", and the tray's
    ``exists()`` check would otherwise announce a corrupt half-bundle
    as success.

    Seeded defect: drop the ``zip_path.unlink(missing_ok=True)`` from
    the outer except -> the partial zip survives -> the not-exists
    assertion goes red.
    """
    def _enospc_writestr(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", _enospc_writestr)
    from yamabiko.diagnostics import collect_diagnostics

    with caplog.at_level(logging.ERROR, logger="yamabiko.diagnostics"):
        out = collect_diagnostics(dest_dir=tmp_path / "out")
    assert not out.exists(), (
        "partial zip left on disk after a mid-write failure; the tray's "
        "exists() check would announce a corrupt bundle as success"
    )
    assert any(
        "diagnostics collection failed" in r.getMessage()
        for r in caplog.records
    ), "failure was swallowed WITHOUT log evidence"


@pytest.mark.real
@pytest.mark.skipif(
    sys.platform != "win32",
    reason="SHGetKnownFolderPath is a Windows shell32 seam",
)
def test_real_desktop_dir_resolves_existing_folder_via_shell32(
    caplog, monkeypatch,
):
    """``_desktop_dir()`` against the REAL shell32
    ``SHGetKnownFolderPath`` (rule #3: no mocks of the subject): it
    must return an existing absolute directory WITHOUT tripping the
    exception fallback -- i.e. the ctypes GUID/signature actually
    satisfies the live API on this machine.

    ``Path.home`` is pointed at a nonexistent sentinel (home is NOT the
    subject) so the naive ``~/Desktop`` fallback cannot satisfy the
    assertions: only a genuinely successful shell32 resolution passes.
    Without this, a silent API failure (e.g. corrupted FOLDERID -> hr
    0x80070002 with no exception) fell back to the real ~/Desktop and
    kept the test green under exactly the defect class it exists to
    catch.

    Seeded defects: corrupt the FOLDERID Data1 field -> hr != 0 -> the
    new hr-warning fires (no-warning assertion red) AND the sentinel
    fallback fails ``is_dir()``; corrupt the returned path
    (``Path(str(buf.value) + "-x")``) -> ``is_dir()`` goes red.
    """
    from yamabiko.diagnostics import _desktop_dir

    monkeypatch.setattr(
        "pathlib.Path.home",
        classmethod(lambda cls: cls("C:/nonexistent-home-sentinel")),
    )
    with caplog.at_level(logging.WARNING, logger="yamabiko.diagnostics"):
        desktop = _desktop_dir()
    assert desktop.is_absolute()
    assert desktop.is_dir(), (
        f"_desktop_dir() returned a non-directory: {desktop}"
    )
    assert not any(
        "SHGetKnownFolderPath" in r.getMessage() for r in caplog.records
    ), "real shell32 call fell back to ~/Desktop; ctypes seam is broken"
