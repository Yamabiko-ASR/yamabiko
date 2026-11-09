"""Path helpers live here — verify they resolve under %LOCALAPPDATA%\\Yamabiko\\
on Windows and respect the YAMABIKO_APPDATA env override used by tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from yamabiko import paths


def test_app_data_dir_uses_localappdata(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("YAMABIKO_APPDATA", raising=False)
    assert paths.app_data_dir() == tmp_path / "Yamabiko"


def test_app_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "custom"))
    assert paths.app_data_dir() == tmp_path / "custom"


def test_config_path_under_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert paths.config_path() == tmp_path / "config.yaml"


def test_state_path_under_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert paths.state_path() == tmp_path / "state.json"


def test_logs_dir_under_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert paths.logs_dir() == tmp_path / "logs"


def test_ensure_dirs_creates_all(tmp_path, monkeypatch):
    """ensure_dirs creates the app-data and logs folders, and no longer the
    v0.1 models folder (nothing uses it; Delete old Yamabiko models removes
    it).

    Seeded defect: put root / "models" back in ensure_dirs -> red.
    """
    target = tmp_path / "appdata"  # does not exist yet
    monkeypatch.setenv("YAMABIKO_APPDATA", str(target))
    paths.ensure_dirs()
    assert target.is_dir()
    assert not (target / "models").exists()
    assert (target / "logs").is_dir()


def test_app_data_dir_fallback_without_env_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("YAMABIKO_APPDATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert paths.app_data_dir() == tmp_path / ".local" / "share" / "Yamabiko"


def test_user_overrides_path_under_app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert paths.user_overrides_path() == tmp_path / "user_overrides.yaml"


def test_shipped_defaults_path_ships_with_package():
    p = paths.shipped_defaults_path()
    assert p.is_file()
    assert p.name == "default_config.yaml"


def test_app_icon_path_ships_with_package():
    p = paths.app_icon_path()
    assert p.is_file(), f"expected app icon at {p}"
    assert p.suffix == ".ico"


def test_bundle_data_dir_dev_mode_resolves_to_package_dir(monkeypatch):
    """bundle_data_dir() must be frozen-aware.

    In dev mode (sys.frozen not set), it returns the yamabiko/ package
    dir, computed via Path(__file__). The glossaries/ subdir must
    exist under it.
    """
    import sys
    # Ensure sys.frozen is NOT set
    monkeypatch.delattr(sys, "frozen", raising=False)
    result = paths.bundle_data_dir()
    assert result.is_dir(), f"bundle_data_dir {result} is not a directory"
    glossaries = result / "glossaries"
    assert glossaries.is_dir(), (
        f"glossaries should exist under bundle_data_dir; got {glossaries}"
    )
    # Verify it actually points at yamabiko/ by checking for a
    # known file
    assert (result / "paths.py").is_file()


def test_bundle_data_dir_frozen_uses_meipass(monkeypatch, tmp_path):
    """In frozen mode, bundle_data_dir uses sys._MEIPASS.

    PyInstaller sets sys.frozen = True and sys._MEIPASS = <bundle>/_internal.
    Our yamabiko package sits at <bundle>/_internal/yamabiko. That's
    what the helper must return.
    """
    import sys
    # Create a fake _MEIPASS layout
    fake_meipass = tmp_path / "_internal"
    (fake_meipass / "yamabiko").mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)
    result = paths.bundle_data_dir()
    assert result == fake_meipass / "yamabiko", (
        f"expected {fake_meipass}/yamabiko, got {result}"
    )


def test_bundle_data_dir_frozen_fallback_when_meipass_missing(monkeypatch, tmp_path):
    """If _MEIPASS somehow isn't set, fall back to sys.executable's dir.

    Defensive: some non-standard PyInstaller configs might not set
    _MEIPASS. Ensure we still land at the right place via the
    ``<exe_dir>/_internal/yamabiko`` convention.
    """
    import sys
    fake_exe_dir = tmp_path / "appdir"
    fake_exe = fake_exe_dir / "yamabiko.exe"
    fake_exe_dir.mkdir()
    fake_exe.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    # Explicitly clear _MEIPASS
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    result = paths.bundle_data_dir()
    assert result == fake_exe_dir / "_internal" / "yamabiko"


def test_shipped_defaults_path_logs_warning_when_resources_probe_raises(
    monkeypatch, caplog,
):
    """A failed importlib.resources probe must leave WARNING evidence (with
    traceback) in the log while STILL falling back to the module-relative
    path -- the bare ``except Exception: pass`` erased the only evidence
    that a frozen install's resources tree is broken (2026-07-01 logging
    audit, Task 9).

    Seeded defect: re-silence the except (revert paths.py's resources
    probe to ``except Exception: pass``) -> no WARNING record -> red.
    """
    import importlib.resources
    import logging

    def _boom(*args, **kwargs):
        raise RuntimeError("resources tree unreadable")

    # shipped_defaults_path does `from importlib.resources import files`
    # at call time, so patching the module attribute intercepts it.
    monkeypatch.setattr(importlib.resources, "files", _boom)

    with caplog.at_level(logging.WARNING, logger="yamabiko.paths"):
        p = paths.shipped_defaults_path()

    # Swallow semantics KEPT: no raise, and the dev fallback still resolves.
    assert p == Path(paths.__file__).parent / "default_config.yaml"

    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.paths" and r.levelno == logging.WARNING
    ]
    assert recs, "resources-probe failure left no WARNING evidence"
    assert recs[0].exc_info and recs[0].exc_info[0] is RuntimeError, (
        "the WARNING must carry the swallowed traceback (exc_info=True)"
    )
