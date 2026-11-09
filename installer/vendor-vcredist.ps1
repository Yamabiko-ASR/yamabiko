# Auto-source the MSVC C/C++ runtime + OpenMP redistributable DLLs imported by
# Yamabiko's native binaries, copying them from the build host's installed
# Visual Studio redist directory into third_party/llama.cpp/ (or a
# caller-supplied -VendorDir). Application-root DLLs can also be published to
# a caller-supplied -AppRuntimeDir after the vendor cache is populated.
#
# WHY this exists
# ---------------
# llama-server.exe is built with MSVC, so it has LOAD-TIME imports on
# msvcp140.dll / vcruntime140.dll / vcruntime140_1.dll (C/C++ runtime) and
# vcomp140.dll (OpenMP). Two facts make these a trap:
#   1. A clean Windows 11 install does NOT ship them -- they only exist if
#      some other app already installed the VC++ redistributable.
#   2. PyInstaller DOES bundle the three CRT DLLs, but into _internal/ (the
#      onedir root). llama-server.exe lives in _internal/llama/, and a
#      freshly-spawned child process searches its OWN directory first, then
#      System32 -- NOT the parent's _internal/ root. So PyInstaller's copy is
#      invisible to llama-server. The DLLs must sit next to llama-server.exe.
# Hence they are part of installer/llama-manifest.txt and must be vendored
# into third_party/llama.cpp/. This script removes the manual "go grab the
# DLLs" step of a re-vendor: it pulls them from the local MSVC toolchain.
#
# App-local deployment of these redist DLLs is supported + licensed for VS
# users, and the v14 runtime is forward-compatible (a newer runtime
# satisfies an older-built binary), so the build host's VS copy is correct.
#
# Idempotent + non-clobbering: a DLL already present in the vendor dir (a real
# re-vendor drop, or a CI stub from ci-prepare-vendor-stubs.ps1) is left
# untouched. Safe to run on every build.
#
# Usage (normally invoked by build.ps1 / rebuild-fast.ps1, before the manifest
# assert):
#   .\installer\vendor-vcredist.ps1
#   .\installer\vendor-vcredist.ps1 -VendorDir C:\some\dir   (tests)
#   .\installer\vendor-vcredist.ps1 -AppRuntimeDir C:\app\_internal
param(
    [string]$VendorDir,
    [string]$AppRuntimeDir
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $VendorDir) {
    $VendorDir = Join-Path $Repo "third_party\llama.cpp"
}

# The VC++ redistributable DLL subset of installer/llama-manifest.txt that we
# can auto-source from the local MSVC toolchain. (llama-server.exe + the
# CUDA/ggml sidecars come from the llama re-vendor, not from VS.) Keep this in
# lockstep with the "MSVC C/C++ runtime" block of llama-manifest.txt --
# tests/test_vendor_vcredist.py asserts the two sets are identical.
$VcRuntimeDlls = @(
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcomp140.dll"
)

# Application-root VC dot libraries are deliberately separate from llama's
# authoritative four-file subset above. ONNX Runtime imports MSVCP140_1.dll by
# basename from the main process domain, so package-private PySide6/shiboken6
# copies cannot satisfy it on a clean host. Cache it beside the llama runtime
# sources, then publish it to PyInstaller's main _internal/ DLL search root.
$AppRuntimeDlls = @(
    "MSVCP140_1.dll"
)
$AllRuntimeDlls = @($VcRuntimeDlls + $AppRuntimeDlls | Select-Object -Unique)

New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null

# Non-clobber: only source DLLs that aren't already present (real drops / CI
# stubs win).
$needed = @($AllRuntimeDlls | Where-Object { -not (Test-Path (Join-Path $VendorDir $_)) })
if ($needed.Count -eq 0) {
    Write-Host "==> VC++ runtime DLLs already present in $VendorDir -- nothing to source." -ForegroundColor Green
} else {
    # Locate the Visual Studio install via vswhere (ships with VS 2017+).
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Cannot auto-source the VC++ runtime DLLs ($($needed -join ', ')): vswhere.exe not found at $vswhere. Install Visual Studio with the 'Desktop development with C++' workload, or drop the DLLs into $VendorDir manually."
    }

    $vsPath = (& $vswhere -latest -products * -property installationPath | Select-Object -First 1)
    if (-not $vsPath -or -not (Test-Path $vsPath)) {
        throw "vswhere found no Visual Studio installation. Install the 'Desktop development with C++' workload, or drop the VC++ runtime DLLs ($($needed -join ', ')) into $VendorDir manually."
    }

    $redistRoot = Join-Path $vsPath "VC\Redist\MSVC"
    if (-not (Test-Path $redistRoot)) {
        throw "No VC redist under $redistRoot -- the 'Desktop development with C++' workload (which ships the redistributable DLLs) is not installed. Add it via the VS Installer, or drop the DLLs into $VendorDir manually."
    }

    foreach ($dll in $needed) {
        # Search the redist tree for this DLL, constrain to the x64 redist
        # (an x86 copy would load-fail beside these 64-bit native binaries),
        # and prefer the newest file version.
        $hit = Get-ChildItem -Path $redistRoot -Recurse -Filter $dll -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\' } |
            Sort-Object { [version]::new($_.VersionInfo.FileMajorPart, $_.VersionInfo.FileMinorPart, $_.VersionInfo.FileBuildPart, $_.VersionInfo.FilePrivatePart) } -Descending |
            Select-Object -First 1
        if (-not $hit) {
            throw "Could not find an x64 $dll under $redistRoot. Ensure the VS 'Desktop development with C++' workload (with the latest MSVC v143 redistributable) is installed, or drop $dll into $VendorDir manually."
        }
        Copy-Item $hit.FullName (Join-Path $VendorDir $dll)
        Write-Host ("==> Sourced {0} (x64) from {1}" -f $dll, $hit.FullName) -ForegroundColor Cyan
    }
}

# Publishing is intentionally outside the sourcing branch: a complete vendor
# cache must still repopulate a freshly-created PyInstaller dist tree.
if ($AppRuntimeDir) {
    New-Item -ItemType Directory -Force -Path $AppRuntimeDir | Out-Null
    foreach ($dll in $AppRuntimeDlls) {
        $cached = Join-Path $VendorDir $dll
        if (-not (Test-Path $cached)) {
            throw "Application runtime DLL missing from vendor cache after sourcing: $cached"
        }
        Copy-Item $cached (Join-Path $AppRuntimeDir $dll) -Force
        Write-Host ("==> Published {0} to application runtime root {1}" -f $dll, $AppRuntimeDir) -ForegroundColor Cyan
    }
}
