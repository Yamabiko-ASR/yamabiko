"""Guards for installer/vendor-vcredist.ps1 -- the auto-sourcer for the
MSVC C/C++ runtime + OpenMP redistributable DLLs the vendored
llama-server.exe statically imports.

WHY this exists: llama-server.exe (built with MSVC) has load-time imports
on msvcp140.dll / vcruntime140.dll / vcruntime140_1.dll / vcomp140.dll.
A clean Windows install does NOT ship these, and -- critically -- a child
process spawned by the watchdog searches ITS OWN directory (_internal/llama)
first, NOT the PyInstaller _internal/ root where PyInstaller drops its own
copy. So llama-server fails to start for users who lack the VC++ redist
unless we vendor these four DLLs next to it (see installer/llama-manifest.txt).
vendor-vcredist.ps1 pulls them from the build host's installed Visual Studio
redist directory so the re-vendor is never a manual chore and the manifest
assert in build.ps1 / rebuild-fast.ps1 finds them.

Two layers of guard:
  * STRUCTURAL (always run): the helper sources EXACTLY the VC++ subset the
    manifest lists (a name drift is a build-day bug); both build scripts invoke it
    BEFORE the manifest assert (else the assert throws first); it is
    non-clobbering and x64-targeted.
  * BEHAVIOURAL (skip if no VS redist on this host): actually run the helper
    against a temp dir and assert it produced real x64 PE DLLs, and that it
    leaves a pre-existing file untouched.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPER = _REPO_ROOT / "installer" / "vendor-vcredist.ps1"
_MANIFEST = _REPO_ROOT / "installer" / "llama-manifest.txt"
_BUILD = _REPO_ROOT / "installer" / "build.ps1"
_REBUILD = _REPO_ROOT / "installer" / "rebuild-fast.ps1"

# Names matching the VC++ runtime / OpenMP redistributable convention:
# msvcp140*.dll, vcruntime140*.dll, vcomp140.dll, concrt140.dll. Matches
# the redist DLLs and NOT the CUDA/ggml/llama sidecars (cublas64_13.dll,
# ggml-base.dll, llama.dll, llama-server.exe).
_VCPP_RE = re.compile(r"^(?:msvcp|vcruntime|vcomp|concrt)\d+.*\.dll$", re.I)


def _manifest_vcpp_dlls() -> set[str]:
    entries = [
        ln.strip()
        for ln in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return {e for e in entries if _VCPP_RE.match(e)}


def _helper_dll_array(variable: str) -> tuple[str, ...]:
    """Parse one named ``$...Dlls = @(...)`` array out of the helper."""
    text = _HELPER.read_text(encoding="utf-8")
    m = re.search(rf"\${re.escape(variable)}\s*=\s*@\((.*?)\)", text, re.S)
    assert m, f"vendor-vcredist.ps1 has no ${variable} = @( ... ) array"
    return tuple(re.findall(r'"([^"]+\.dll)"', m.group(1)))


def _helper_sourced_dlls() -> set[str]:
    return set(_helper_dll_array("VcRuntimeDlls"))


def _helper_application_runtime_dlls() -> tuple[str, ...]:
    return _helper_dll_array("AppRuntimeDlls")


def _executable_vendor_helper_call(text: str, script: Path) -> re.Match[str]:
    call = re.search(
        r'(?m)^[ \t]*&[ \t]+\(Join-Path\s+\$PSScriptRoot\s+'
        r'"vendor-vcredist\.ps1"\)'
        r'(?P<arguments>.*?)(?=\r?\n[ \t]*if\s*\(\$LASTEXITCODE)',
        text,
        re.S,
    )
    assert call, f"{script.name} has no executable vendor-vcredist.ps1 call"
    return call


# ---------------------------------------------------------------------------
# Structural guards (always run)
# ---------------------------------------------------------------------------


def test_helper_exists() -> None:
    assert _HELPER.is_file(), f"missing {_HELPER}"


def test_helper_sources_exactly_the_manifest_vcpp_subset() -> None:
    """The DLLs the helper auto-sources must equal the VC++ subset the
    manifest demands. If the manifest lists vcomp140.dll but the helper
    doesn't source it, a clean build box's manifest assert throws with a
    confusing 'missing vcomp140.dll' and the build dies; if the helper
    sources a name the manifest doesn't ship, it's dead work.

    Seeded defect: add a 5th VC++ DLL (e.g. msvcp140_1.dll) to
    llama-manifest.txt but not to $VcRuntimeDlls (or vice versa) -> the
    set equality breaks -> this fails.
    """
    manifest = _manifest_vcpp_dlls()
    helper = _helper_sourced_dlls()
    assert manifest, "manifest lists no VC++ runtime DLLs -- pattern drift?"
    assert helper == manifest, (
        f"vendor-vcredist.ps1 sources {sorted(helper)} but the manifest "
        f"demands {sorted(manifest)}; a clean build box would fail the "
        f"manifest assert on the difference (missing-DLL drift)"
    )


def test_helper_declares_only_msvcp140_1_as_application_runtime() -> None:
    """ONNX Runtime imports the VC dot library from the application's main
    DLL search domain.  It is not part of llama's four-file manifest subset,
    so the helper must declare it in a separate, exact output contract.

    Seeded defect: remove ``MSVCP140_1.dll`` from ``$AppRuntimeDlls`` -> the
    tuple is empty -> this fails.
    """
    assert _helper_application_runtime_dlls() == ("MSVCP140_1.dll",)


def test_helper_filters_to_x64() -> None:
    """Sourcing the x86 redist DLL would let the build + manifest assert
    pass yet ship a 32-bit msvcp140.dll next to a 64-bit llama-server.exe,
    which fails to load at runtime on the user's machine -- a silent ship.

    Seeded defect: drop the `\\x64\\` path filter from the Get-ChildItem
    pipeline -> the helper could grab the x86 copy -> this fails.
    """
    text = _HELPER.read_text(encoding="utf-8")
    selection = re.search(
        r"\$hit\s*=\s*Get-ChildItem\b.*?Select-Object\s+-First\s+1",
        text,
        re.S,
    )
    assert selection, "vendor-vcredist.ps1 has no redist selection pipeline"
    assert re.search(
        r"Where-Object\s*\{\s*\$_\.FullName\s+-match\s+'\\\\x64\\\\'\s*\}",
        selection.group(0),
    ), (
        "vendor-vcredist.ps1's executable redist selection pipeline does not "
        "constrain FullName to the x64 directory; comments mentioning x64 "
        "cannot satisfy this guard"
    )


def test_helper_is_non_clobbering() -> None:
    """A real re-vendor may drop version-matched VC++ DLLs into
    third_party/llama.cpp/ deliberately, and CI pre-stubs them; the helper
    must not overwrite an existing file.

    Seeded defect: remove the `Test-Path` skip-if-present guard -> the
    helper clobbers real drops / CI stubs -> this fails.
    """
    text = _HELPER.read_text(encoding="utf-8")
    assert "Test-Path" in text, (
        "vendor-vcredist.ps1 lacks a skip-if-present guard; it would "
        "clobber deliberately-vendored or CI-stubbed DLLs"
    )


@pytest.mark.parametrize("script", [_BUILD, _REBUILD], ids=["build", "rebuild-fast"])
def test_build_scripts_invoke_helper_after_pyinstaller_before_manifest_assert(
    script: Path,
) -> None:
    """Both build paths run vendor-vcredist.ps1 after PyInstaller creates
    dist and before the shared llama manifest assert.

    Running it before PyInstaller can lose the published application runtime
    when COLLECT recreates the tree. Running it after the manifest assert (or
    not at all) makes a clean vendor cache fail before auto-sourcing runs.

    Seeded defects: move the call before PyInstaller or below the manifest
    assert -> the executable ordering bound fails.
    """
    text = script.read_text(encoding="utf-8")
    call = _executable_vendor_helper_call(text, script)
    pyinstaller = re.search(
        r'(?m)^[ \t]*&[ \t]+\$Python[ \t]+-m[ \t]+PyInstaller[ \t]+'
        r'\$Spec[ \t]+--noconfirm[ \t]+`[ \t]*$',
        text,
    )
    manifest_assert = re.search(
        r'(?m)^[ \t]*\$LlamaFiles\s*=\s*@\(&\s*\(Join-Path\s+'
        r'\$PSScriptRoot\s+"assert-llama-vendor\.ps1"\)\)',
        text,
    )
    assert pyinstaller, f"{script.name} has no executable PyInstaller call"
    assert manifest_assert, f"{script.name} has no executable llama manifest assert"
    assert pyinstaller.start() < call.start() < manifest_assert.start(), (
        f"{script.name} must invoke vendor-vcredist.ps1 after PyInstaller "
        f"and before the manifest assert; got offsets PyInstaller="
        f"{pyinstaller.start()}, helper={call.start()}, "
        f"manifest={manifest_assert.start()}"
    )


@pytest.mark.parametrize("script", [_BUILD, _REBUILD], ids=["build", "rebuild-fast"])
def test_build_scripts_publish_application_runtime_to_main_dll_root(
    script: Path,
) -> None:
    """Both build paths must publish the ONNX dot library to the exact
    PyInstaller main DLL search root after Analysis has built the tree.

    Seeded defect: omit ``-AppRuntimeDir`` from either helper invocation ->
    that script's parameterized case fails.
    """
    text = script.read_text(encoding="utf-8")
    call = _executable_vendor_helper_call(text, script)
    assert re.search(
        r'-AppRuntimeDir\s+\(Join-Path\s+\$Dist\s+"_internal"\)',
        call.group("arguments"),
    ), (
        f"{script.name} does not pass dist/yamabiko/_internal as "
        "-AppRuntimeDir; ONNX would not find MSVCP140_1.dll on a clean host"
    )


# ---------------------------------------------------------------------------
# Behavioural guards (skip when this host has no VS C++ redist)
# ---------------------------------------------------------------------------


def _vswhere() -> Path | None:
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    p = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return p if p.is_file() else None


def _has_vcpp_redist() -> bool:
    vw = _vswhere()
    if vw is None:
        return False
    try:
        out = subprocess.run(
            [str(vw), "-latest", "-products", "*", "-property", "installationPath"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    vs = out.stdout.strip()
    if not vs:
        return False
    return (Path(vs) / "VC" / "Redist" / "MSVC").is_dir()


@pytest.fixture
def require_vcpp_redist() -> None:
    """Skip real helper execution unless this host has a VS C++ redist.

    Detection is fixture-scoped rather than module-scoped so selecting the
    structural unit tier never executes ``vswhere.exe`` during collection.
    """
    if not (os.name == "nt" and _has_vcpp_redist()):
        pytest.skip("no Visual Studio C++ redistributable on this host")


def _run_helper(
    vendor_dir: Path, app_runtime_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    command = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(_HELPER), "-VendorDir", str(vendor_dir),
    ]
    if app_runtime_dir is not None:
        command.extend(["-AppRuntimeDir", str(app_runtime_dir)])
    return subprocess.run(
        command,
        capture_output=True, text=True, timeout=180,
    )


def _assert_real_x64_pe(path: Path) -> bytes:
    data = path.read_bytes()
    assert data[:2] == b"MZ", f"{path.name} is not a PE image (no MZ header)"
    assert len(data) > 10_000, f"{path.name} is implausibly small ({len(data)} B)"
    pe_offset = int.from_bytes(data[0x3C:0x40], "little")
    assert data[pe_offset:pe_offset + 4] == b"PE\0\0", (
        f"{path.name} has no PE signature at its declared header offset"
    )
    machine = int.from_bytes(data[pe_offset + 4:pe_offset + 6], "little")
    assert machine == 0x8664, (
        f"{path.name} has PE machine 0x{machine:04x}, expected x64 0x8664"
    )
    return data


@pytest.mark.real
def test_helper_produces_real_x64_runtime_dlls(
    tmp_path: Path, require_vcpp_redist: None,
) -> None:
    """Run the helper against an empty dir and assert it actually copied
    real x64 PE DLLs (MZ header, non-trivial size) for every name it
    claims to source -- not empty files, not the wrong arch.

    Seeded defect: break the redist path resolution (e.g. point at a
    non-existent folder) or strip the x64 filter so it copies a stub ->
    the produced files are missing / 0-byte / non-PE -> this fails.
    """
    proc = _run_helper(tmp_path)
    assert proc.returncode == 0, (
        f"vendor-vcredist.ps1 exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    for name in _helper_sourced_dlls():
        dll = tmp_path / name
        assert dll.is_file(), f"{name} was not produced: {proc.stdout}"
        _assert_real_x64_pe(dll)


@pytest.mark.real
def test_helper_preserves_existing_file(
    tmp_path: Path, require_vcpp_redist: None,
) -> None:
    """Non-clobber, verified behaviourally: a pre-existing file with the
    name of a sourced DLL must survive a run untouched (real re-vendor
    drops + CI stubs depend on this).

    Seeded defect: remove the Test-Path skip in the helper -> the sentinel
    is overwritten with a real DLL -> the content assertion fails.
    """
    names = sorted(_helper_sourced_dlls())
    sentinel_name = names[0]
    sentinel = tmp_path / sentinel_name
    sentinel.write_bytes(b"SENTINEL-not-a-real-dll")

    proc = _run_helper(tmp_path)
    assert proc.returncode == 0, (
        f"vendor-vcredist.ps1 exited {proc.returncode}\n{proc.stderr}"
    )
    assert sentinel.read_bytes() == b"SENTINEL-not-a-real-dll", (
        f"{sentinel_name} was clobbered; helper is not non-clobbering"
    )


@pytest.mark.real
def test_helper_publishes_cached_application_runtime_to_requested_root(
    tmp_path: Path, require_vcpp_redist: None,
) -> None:
    """The helper copies a real x64 dot library from its non-clobbering
    vendor cache to the requested application root on every invocation.

    Running twice is causal pressure on the complete-cache path: after the
    first run fills the cache, deleting only the published copy and invoking
    again must restore it without re-sourcing or returning early.

    Seeded defect: omit the application-root copy loop -> the first output is
    absent.  Restore the loop but return when the cache is complete -> the
    second output remains absent.
    """
    vendor_dir = tmp_path / "vendor"
    app_runtime_dir = tmp_path / "application-root"

    first = _run_helper(vendor_dir, app_runtime_dir)
    assert first.returncode == 0, (
        f"vendor-vcredist.ps1 exited {first.returncode}\n"
        f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"
    )
    cached = vendor_dir / "MSVCP140_1.dll"
    published = app_runtime_dir / "MSVCP140_1.dll"
    cached_bytes = _assert_real_x64_pe(cached)
    assert _assert_real_x64_pe(published) == cached_bytes

    published.unlink()
    second = _run_helper(vendor_dir, app_runtime_dir)
    assert second.returncode == 0, (
        f"complete-cache invocation exited {second.returncode}\n"
        f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
    )
    assert _assert_real_x64_pe(published) == cached_bytes
