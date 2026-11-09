"""Behavioural guards for installer/assert-llama-vendor.ps1 -- the shared
manifest assert used by build.ps1 + rebuild-fast.ps1 -- and the CI-only
guard on installer/ci-prepare-vendor-stubs.ps1.

Every guard between a CI text stub and a shipped installer was
existence-only. If ci-prepare-vendor-stubs.ps1 was ever run locally
while third_party/llama.cpp was missing a manifest file (partial
re-vendor), the 27-byte stub it created satisfied build.ps1's Test-Path
assert, vendor-vcredist's non-clobber, and preflight's name-membership
check -- and would ship in a real release build where llama-server (or a
sidecar DLL) fails to load and translation is dead on the user's machine.

Two production changes under test, both exercised for real via
``powershell.exe -File`` (same pattern as tests/test_vendor_vcredist.py):

* assert-llama-vendor.ps1 rejects manifest .exe/.dll entries that are
  smaller than 4 KB or lack the MZ magic, unless stubs are explicitly
  allowed (-AllowStubs / YAMABIKO_ALLOW_STUB_LLAMA=1 -- the CI escape
  hatch installer-smoke.yml uses, since its builds stub deliberately).
* ci-prepare-vendor-stubs.ps1 refuses to run outside CI (GITHUB_ACTIONS
  / CI env) so the stub-creation path can't be triggered on a dev box
  by accident.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSERT = _REPO_ROOT / "installer" / "assert-llama-vendor.ps1"
_STUBS = _REPO_ROOT / "installer" / "ci-prepare-vendor-stubs.ps1"

# Plausible PE content: MZ magic + padding past the 4 KB stub floor.
_FAKE_PE = b"MZ" + b"\x00" * 8192


def _write_manifest(tmp_path: Path, entries: list[str]) -> Path:
    manifest = tmp_path / "llama-manifest.txt"
    manifest.write_text(
        "# test manifest\n" + "\n".join(entries) + "\n", encoding="utf-8",
    )
    return manifest


def _run_ps1(script: Path, *args: str, env_extra: dict | None = None,
             env_drop: tuple = ()) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in env_drop}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(script), *args],
        capture_output=True, text=True, timeout=120, env=env,
    )


# ---------------------------------------------------------------------------
# assert-llama-vendor.ps1
# ---------------------------------------------------------------------------


def test_assert_passes_and_emits_file_list_for_real_pe_vendor(tmp_path):
    """Happy path: every manifest entry present with PE-plausible content
    -> exit 0 and the manifest names on stdout (callers drive their copy
    loop from this output). Seeded defect: make the stub check reject
    valid MZ content (invert the magic comparison)."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    for name in ("llama-server.exe", "ggml.dll"):
        (vendor / name).write_bytes(_FAKE_PE)
    manifest = _write_manifest(tmp_path, ["llama-server.exe", "ggml.dll"])

    proc = _run_ps1(_ASSERT, "-VendorDir", str(vendor),
                    "-ManifestPath", str(manifest),
                    env_drop=("YAMABIKO_ALLOW_STUB_LLAMA",))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "llama-server.exe" in proc.stdout
    assert "ggml.dll" in proc.stdout


def test_assert_fails_on_missing_manifest_file(tmp_path):
    """An absent manifest file fails the build loudly.
    Seeded defect: drop the Test-Path missing check."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "llama-server.exe").write_bytes(_FAKE_PE)
    manifest = _write_manifest(tmp_path, ["llama-server.exe", "ggml-cuda.dll"])

    proc = _run_ps1(_ASSERT, "-VendorDir", str(vendor),
                    "-ManifestPath", str(manifest))
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "ggml-cuda.dll" in combined


def test_assert_fails_on_text_stub_without_allowstubs(tmp_path):
    """A CI text stub (the exact 27-byte content ci-prepare-vendor-stubs.ps1
    writes) must fail the assert on a non-CI build.
    Seeded defect: revert to the existence-only check."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "llama-server.exe").write_text(
        "CI stub (not a real binary)\n", encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, ["llama-server.exe"])

    proc = _run_ps1(_ASSERT, "-VendorDir", str(vendor),
                    "-ManifestPath", str(manifest),
                    env_drop=("YAMABIKO_ALLOW_STUB_LLAMA",))
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "llama-server.exe" in combined
    assert "stub" in combined.lower()


def test_assert_allowstubs_switch_admits_stubs(tmp_path):
    """The CI escape hatch: -AllowStubs skips ONLY the content check
    (presence is still enforced) so installer-smoke's deliberate stub
    build keeps working. Seeded defect: make -AllowStubs skip the
    presence assert too (drop a manifest entry -> should still fail)."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "llama-server.exe").write_text(
        "CI stub (not a real binary)\n", encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, ["llama-server.exe"])

    ok = _run_ps1(_ASSERT, "-VendorDir", str(vendor),
                  "-ManifestPath", str(manifest), "-AllowStubs")
    assert ok.returncode == 0, ok.stdout + ok.stderr

    # Presence must still be enforced under -AllowStubs.
    manifest2 = _write_manifest(tmp_path, ["llama-server.exe", "ggml.dll"])
    still_missing = _run_ps1(_ASSERT, "-VendorDir", str(vendor),
                             "-ManifestPath", str(manifest2), "-AllowStubs")
    assert still_missing.returncode != 0


def test_assert_env_var_admits_stubs(tmp_path):
    """YAMABIKO_ALLOW_STUB_LLAMA=1 is how installer-smoke.yml opts in
    without touching the build.ps1 call site. Seeded defect: read the
    wrong env var name."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "ggml.dll").write_text("CI stub (not a real binary)\n",
                                     encoding="utf-8")
    manifest = _write_manifest(tmp_path, ["ggml.dll"])

    proc = _run_ps1(_ASSERT, "-VendorDir", str(vendor),
                    "-ManifestPath", str(manifest),
                    env_extra={"YAMABIKO_ALLOW_STUB_LLAMA": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# ci-prepare-vendor-stubs.ps1 CI-only guard
# ---------------------------------------------------------------------------


def test_stub_script_refuses_to_run_outside_ci(tmp_path):
    """The stub creator must refuse on a dev box (no GITHUB_ACTIONS / CI env),
    so the stub-creation path can't be triggered by accident.
    Seeded defect: delete the env guard."""
    vendor = tmp_path / "vendor"
    proc = _run_ps1(_STUBS, "-VendorDir", str(vendor),
                    env_drop=("GITHUB_ACTIONS", "CI"))
    assert proc.returncode != 0
    assert not vendor.exists() or not any(vendor.iterdir())


def test_stub_script_creates_stubs_when_ci_env_set(tmp_path):
    """With CI=true the script works as before, creating a stub for
    every manifest entry in the target dir."""
    vendor = tmp_path / "vendor"
    proc = _run_ps1(_STUBS, "-VendorDir", str(vendor),
                    env_extra={"CI": "true"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (vendor / "llama-server.exe").is_file()
