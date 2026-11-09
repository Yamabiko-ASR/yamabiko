# Yamabiko fast-rebuild script for dev iteration.
#
# Full build (installer\build.ps1) takes ~15 minutes:
#   * Full PyInstaller analysis   (~2.5 min)
#   * NSIS LZMA compression       (~10 min)
#
# This fast path skips both expensive steps:
#   * Keeps build/ cache intact so PyInstaller reuses the module-graph
#     analysis and only recompiles modules whose .py files changed.
#   * Skips NSIS entirely. For testing frozen-exe behavior against
#     yamabiko.exe --run-main / --version you don't need the packaged
#     .exe installer -- you run dist/yamabiko/yamabiko.exe directly.
#
# Expected wall time on a warm cache:
#   * ~20-30s if only yamabiko/*.py changed
#   * ~60-90s if the spec changed (forces deeper re-analysis)
#
# Usage (from repo root):
#   .\installer\rebuild-fast.ps1 [-Venv .venv-release]
#
# -Venv picks the build venv: a folder name under the repo root or an
# absolute path (default .venv). Unlike build.ps1 this script does not check
# the venv against constraints-release.txt.
#
# When to use full build.ps1 instead:
#   * Before tagging a release
#   * Whenever you actually need installer\yamabiko-<version>-setup.exe
#     to hand to a user
#   * If dependency versions changed (pip install happened)
#   * If you suspect the PyInstaller cache is stale (rare, but if you
#     get weird "ModuleNotFoundError" on something that should be
#     bundled, nuke build/ and run build.ps1)

param(
    # Build venv: a folder name under the repo root, or an absolute path.
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$VenvDir = if ([System.IO.Path]::IsPathRooted($Venv)) { $Venv } else { Join-Path $Repo $Venv }
$Dist = Join-Path $Repo "dist\yamabiko"
$Vendor = Join-Path $Repo "third_party\llama.cpp"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$SitePackages = Join-Path $VenvDir "Lib\site-packages"
$Inventory = Join-Path $Repo "build\yamabiko\bundle-inventory.json"
$ProvenanceConfig = Join-Path $Repo "installer\bundle-provenance.toml"
$ProvenanceReport = Join-Path $Repo "build\yamabiko\bundle-provenance.json"
$LicenseOverrides = Join-Path $Repo "installer\licenses-manifest.txt"
$LicenseIndex = Join-Path $Dist "_internal\licenses\INDEX.json"
$ClosureAudit = Join-Path $Repo "build\yamabiko\license-closure-audit.json"
$Spec = Join-Path $Repo "installer\yamabiko.spec"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "no Python at $Python -- create the build venv (see installer\build.ps1's Prereqs)"
}

Write-Host "==> fast-rebuild: wiping dist\ only (keeping build\ cache)" -ForegroundColor Cyan
if (Test-Path (Join-Path $Repo "dist")) {
    Remove-Item -Recurse -Force (Join-Path $Repo "dist")
}
# NOTE: we deliberately do NOT wipe build/. PyInstaller's module-graph
# analysis is cached there; preserving it is the whole point of this
# script.

Write-Host "==> fast-rebuild: PyInstaller (incremental from build\ cache)" -ForegroundColor Cyan
& $Python -m PyInstaller $Spec --noconfirm `
    --workpath (Join-Path $Repo "build") `
    --distpath (Join-Path $Repo "dist")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Auto-source the VC++ runtime DLLs (same as build.ps1) before the manifest
# assert, so a clean box doesn't fail on a missing msvcp140/vcomp140 etc.
# Non-clobbering: real drops + CI stubs are left untouched.
& (Join-Path $PSScriptRoot "vendor-vcredist.ps1") `
    -AppRuntimeDir (Join-Path $Dist "_internal")
if ($LASTEXITCODE -ne 0) { throw "vendor-vcredist.ps1 failed" }

# Assert the vendored llama.cpp is complete + non-stub per the shared
# manifest -- same guard as build.ps1.
$LlamaFiles = @(& (Join-Path $PSScriptRoot "assert-llama-vendor.ps1"))

Write-Host "==> fast-rebuild: copying vendored llama.cpp (per manifest)" -ForegroundColor Cyan
$LlamaTarget = Join-Path $Dist "_internal\llama"
New-Item -ItemType Directory -Force -Path $LlamaTarget | Out-Null
foreach ($f in $LlamaFiles) {
    Copy-Item (Join-Path $Vendor $f) $LlamaTarget
}

Write-Host "==> fast-rebuild: auditing generated bundle inventory" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\audit_bundle_inventory.py") `
    --inventory $Inventory `
    --dist $Dist
if ($LASTEXITCODE -ne 0) { throw "audit_bundle_inventory.py failed" }

Write-Host "==> fast-rebuild: resolving final-tree bundle provenance" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\bundle_provenance.py") resolve `
    --inventory $Inventory `
    --dist $Dist `
    --policy-config $ProvenanceConfig `
    --site-packages $SitePackages `
    --output $ProvenanceReport
if ($LASTEXITCODE -ne 0) { throw "bundle_provenance.py failed" }

# Derive the exact payload from bundle ownership and the reviewed overrides,
# matching the full release build rather than copying a static basename list.
Write-Host "==> fast-rebuild: collecting bundle-derived licenses" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\collect_licenses.py") `
    --dist $Dist `
    --inventory $Inventory `
    --provenance-report $ProvenanceReport `
    --override-policy $LicenseOverrides `
    --site-packages $SitePackages
if ($LASTEXITCODE -ne 0) { throw "collect_licenses.py failed" }

Write-Host "==> fast-rebuild: auditing independent bundle and license closure" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\audit_license_closure.py") `
    --dist $Dist `
    --inventory $Inventory `
    --license-index $LicenseIndex `
    --provenance-policy-config $ProvenanceConfig `
    --license-override-policy $LicenseOverrides `
    --site-packages $SitePackages `
    --output $ClosureAudit
if ($LASTEXITCODE -ne 0) { throw "audit_license_closure.py failed" }

Write-Host "==> fast-rebuild: smoke test (--version)" -ForegroundColor Cyan
# yamabiko.exe is a windowed (console=False) PyInstaller bootloader.
# The `&` call operator cannot reliably capture exit code or stdout for
# windowed apps -- the bootloader detaches from the parent console, so
# the old `&` + $LASTEXITCODE smoke discarded the version string and
# could miss a crashed child. Use
# Start-Process with explicit redirection + a hard timeout, mirroring
# ci-smoke.ps1's Invoke-InstalledExe.
$smokeOut = New-TemporaryFile
$smokeErr = New-TemporaryFile
try {
    $smokeProc = Start-Process -FilePath (Join-Path $Dist "yamabiko.exe") `
        -ArgumentList @("--version") `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput $smokeOut.FullName `
        -RedirectStandardError $smokeErr.FullName
    # Touch .Handle before waiting; without it .NET may not cache the
    # process handle and .ExitCode stays $null for a fast-exiting child.
    $null = $smokeProc.Handle
    if (-not $smokeProc.WaitForExit(30 * 1000)) {
        $smokeProc.Kill()
        throw "yamabiko.exe --version still running after 30s -- killed (frozen-import deadlock?)"
    }
    $smokeStdout = (Get-Content $smokeOut.FullName -Raw -ErrorAction SilentlyContinue | Out-String).Trim()
    $smokeStderr = (Get-Content $smokeErr.FullName -Raw -ErrorAction SilentlyContinue | Out-String).Trim()
    if ($smokeProc.ExitCode -ne 0) {
        throw "yamabiko.exe --version exited $($smokeProc.ExitCode). Stdout:`n$smokeStdout`nStderr:`n$smokeStderr"
    }
    if ($smokeStdout -notmatch "Yamabiko") {
        throw "yamabiko.exe --version stdout missing 'Yamabiko' marker. Stdout:`n$smokeStdout`nStderr:`n$smokeStderr"
    }
    Write-Host "    $smokeStdout"
}
finally {
    Remove-Item $smokeOut.FullName, $smokeErr.FullName -ErrorAction SilentlyContinue
}

Write-Host "==> fast-rebuild: done. Run tests/test_real_frozen_e2e.py for E2E verification." -ForegroundColor Green
Write-Host "    (Full installer .exe NOT built. Use installer\build.ps1 for that.)" -ForegroundColor Yellow
