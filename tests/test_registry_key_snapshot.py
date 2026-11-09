"""Behavioural guards for installer/snapshot-registry-key.ps1 -- the
save/restore helper ci-smoke.ps1 uses to protect the real install's
HKCU uninstall registration.

ci-smoke.ps1 installs to %TEMP%\\yamabiko-ci-smoke, which makes the
NSIS installer overwrite the SINGLE shared HKCU\\...\\Uninstall\\Yamabiko
key (InstallLocation now points at the temp dir); a successful run then
DELETES the key on uninstall (unregistering any real install from
Add/Remove Programs), and a failed or -KeepInstall run leaves the temp
path in InstallDirRegKey so every subsequent interactive install resolves
$INSTDIR to %TEMP% -- violating the install-dir == %LOCALAPPDATA%
invariant yamabiko.nsi declares mandatory.

The helper is exercised for REAL against a scratch key under
HKCU:\\Software\\YamabikoTests (never the actual uninstall key), via
``powershell.exe -File`` -- same pattern as test_vendor_vcredist.py.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPER = _REPO_ROOT / "installer" / "snapshot-registry-key.ps1"
_CI_SMOKE = _REPO_ROOT / "installer" / "ci-smoke.ps1"


def _ps(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-Command", command],
        capture_output=True, text=True, timeout=60,
    )


def _run_helper(action: str, key_path: str, backup: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(_HELPER), "-Action", action,
         "-KeyPath", key_path, "-BackupFile", str(backup)],
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def scratch_key():
    """A unique scratch key under HKCU:\\Software\\YamabikoTests; removed
    (with its parent, if empty) on teardown so test runs leave no residue."""
    name = f"snapshot-test-{uuid.uuid4().hex[:8]}"
    key = rf"HKCU:\Software\YamabikoTests\{name}"
    yield key
    _ps(f"if (Test-Path '{key}') {{ Remove-Item '{key}' -Recurse -Force }}")


def _get_value(key: str, value_name: str) -> str:
    proc = _ps(
        f"(Get-ItemProperty -Path '{key}' -Name '{value_name}' "
        f"-ErrorAction SilentlyContinue).'{value_name}'"
    )
    return proc.stdout.strip()


def _key_exists(key: str) -> bool:
    proc = _ps(f"Test-Path '{key}'")
    return proc.stdout.strip() == "True"


def test_restore_recovers_preexisting_values(scratch_key, tmp_path):
    """The real-install-protection path: a pre-existing key (the user's
    actual install registration) survives a smoke run that overwrote and
    polluted it -- restore brings back the original values and drops the
    smoke run's additions. Seeded defect: make restore skip the
    delete-before-import (the polluted extra value would survive) or
    import nothing (InstallLocation stays poisoned)."""
    backup = tmp_path / "snap.reg"
    _ps(f"New-Item -Path '{scratch_key}' -Force | Out-Null; "
        f"Set-ItemProperty -Path '{scratch_key}' -Name InstallLocation "
        f"-Value 'C:\\real\\install'")

    save = _run_helper("save", scratch_key, backup)
    assert save.returncode == 0, save.stdout + save.stderr

    # Simulate the smoke install poisoning the key.
    _ps(f"Set-ItemProperty -Path '{scratch_key}' -Name InstallLocation "
        f"-Value 'C:\\Temp\\yamabiko-ci-smoke'; "
        f"Set-ItemProperty -Path '{scratch_key}' -Name SmokeJunk -Value 'x'")

    restore = _run_helper("restore", scratch_key, backup)
    assert restore.returncode == 0, restore.stdout + restore.stderr
    assert _get_value(scratch_key, "InstallLocation") == r"C:\real\install"
    assert _get_value(scratch_key, "SmokeJunk") == ""


def test_restore_deletes_key_that_did_not_preexist(scratch_key, tmp_path):
    """The no-real-install path: if no key existed at save time, restore
    must remove whatever the smoke run created -- otherwise a failed or
    -KeepInstall run leaves InstallDirRegKey pointing at %TEMP% and every
    future interactive install lands there. Seeded defect: drop the
    .absent marker handling so restore leaves the smoke-created key in
    place."""
    backup = tmp_path / "snap.reg"
    assert not _key_exists(scratch_key)

    save = _run_helper("save", scratch_key, backup)
    assert save.returncode == 0, save.stdout + save.stderr

    # Smoke run creates the key.
    _ps(f"New-Item -Path '{scratch_key}' -Force | Out-Null; "
        f"Set-ItemProperty -Path '{scratch_key}' -Name InstallLocation "
        f"-Value 'C:\\Temp\\yamabiko-ci-smoke'")

    restore = _run_helper("restore", scratch_key, backup)
    assert restore.returncode == 0, restore.stdout + restore.stderr
    assert not _key_exists(scratch_key)


def test_restore_without_prior_save_fails_loudly(scratch_key, tmp_path):
    """Calling restore with neither a .reg snapshot nor an .absent marker
    means save never ran -- silently doing nothing would defeat the
    guarantee, so it must error. Seeded defect: return 0 from the
    no-snapshot branch."""
    backup = tmp_path / "never-saved.reg"
    restore = _run_helper("restore", scratch_key, backup)
    assert restore.returncode != 0


# ---------------------------------------------------------------------------
# ci-smoke.ps1 wiring (the helper is only protection if ci-smoke calls it
# around the install/uninstall cycle; full behavioural coverage of the
# cycle itself runs in installer-smoke.yml on every release-cut PR)
# ---------------------------------------------------------------------------


def test_ci_smoke_saves_before_install_and_restores_in_finally():
    """Seeded defect: delete the finally block (a mid-run throw would
    again leave the poisoned key), or move the save below the
    Start-Process install (the snapshot would capture the poisoned
    state)."""
    text = _CI_SMOKE.read_text(encoding="utf-8")
    save_at = text.index("-Action save")
    install_at = text.index('@("/S", "/D=$DestDir")')  # the install call
    restore_at = text.index("-Action restore")
    assert save_at < install_at < restore_at
    assert "finally" in text
    finally_at = text.index("finally")
    assert restore_at > finally_at, (
        "the restore call must live inside the finally block so throws "
        "and -KeepInstall exits still restore the real install's key"
    )
