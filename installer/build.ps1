# Yamabiko installer build orchestrator.
#
# Usage (from repo root):
#   .\installer\build.ps1 -Venv .venv-release
#
# -Venv picks the build venv: a folder name under the repo root or an
# absolute path (default .venv, which CI uses). Its `pip freeze` must match
# constraints-release.txt exactly (checked below), so release builds use the
# dedicated .venv-release (docs/RELEASE.md, "Release venv"); a dev venv with
# extra packages is refused.
#
# Prereqs (Python 3.12 build venv, version-locked for a reproducible
# release build -- see constraints-release.txt):
#   <venv>\Scripts\python.exe -m pip install -r requirements-dev.txt -c constraints-release.txt
#   <venv>\Scripts\python.exe -m pip install -e .                    -c constraints-release.txt
#   <venv>\Scripts\python.exe -m pip install "pyinstaller>=6.10"     -c constraints-release.txt
# - third_party/llama.cpp/llama-server.exe + DLLs present
# - NSIS Unicode build on PATH (makensis.exe)

param(
    # Build venv: a folder name under the repo root, or an absolute path.
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is installer/; its parent is the repo root. This makes
# the script location-independent (don't rely on the user's cwd).
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
$ReleaseGates = Join-Path $Repo "build\release-gates"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "no Python at $Python -- create the build venv (see the Prereqs above and docs\RELEASE.md, 'Release venv')"
}

# Bind the installer to the commit it is built from. Read before the build
# writes anything, so its own outputs cannot mark the tree dirty. Outside a
# git checkout both stay null and preflight --release section 16 fails.
$GitHead = $null
$GitDirty = $null
if (Get-Command git -ErrorAction SilentlyContinue) {
    $headLines = @(& git -C $Repo rev-parse HEAD)
    if ($LASTEXITCODE -eq 0 -and $headLines.Count -eq 1) {
        $GitHead = "$($headLines[0])".Trim()
        $statusLines = @(& git -C $Repo status --porcelain)
        if ($LASTEXITCODE -eq 0) { $GitDirty = $statusLines.Count -gt 0 }
    }
}
if (-not $GitHead -or $null -eq $GitDirty) {
    Write-Host "    git HEAD or tree state unreadable; installer.json will not bind a commit" -ForegroundColor Yellow
}

Write-Host "==> Verifying build venv against constraints-release.txt" -ForegroundColor Cyan
# The reproducibility claim ("two builds of the same version resolve the
# same package set") only holds if the venv this script builds FROM
# actually matches the lock. The Prereqs commands above install with
# `-c constraints-release.txt`, but nothing re-checked the venv at build
# time -- one ad-hoc `pip install` after setup would silently diverge
# the installer users receive. Compare `pip freeze` (the same command
# that generates the lock; see the lock's header) against the lock and
# fail loudly, with the full diff, on any drift. Runs FIRST so a stale
# env can't burn ~5 min of PyInstaller before failing.
function Get-PinnedVersions([string[]] $Lines) {
    $map = @{}
    foreach ($line in $Lines) {
        $trimmed = "$line".Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        $parts = $trimmed -split "==", 2
        if ($parts.Count -ne 2) { continue }
        # PEP 503 canonical name so PyYAML/pyyaml, hf_xet/hf-xet compare equal.
        $name = ($parts[0].Trim() -replace "[-_.]+", "-").ToLowerInvariant()
        $map[$name] = $parts[1].Trim()
    }
    return $map
}
$LockPath = Join-Path $Repo "constraints-release.txt"
$Locked = Get-PinnedVersions (Get-Content $LockPath)
if ($Locked.Count -eq 0) { throw "no pins parsed from $LockPath" }
$FrozenLines = & $Python -m pip freeze --exclude-editable
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed for $Python -- is the release venv installed?" }
$Installed = Get-PinnedVersions $FrozenLines
$Drift = @()
foreach ($name in ($Locked.Keys | Sort-Object)) {
    if (-not $Installed.ContainsKey($name)) {
        $Drift += "MISSING   $name==$($Locked[$name]) (locked but not installed)"
    } elseif ($Installed[$name] -ne $Locked[$name]) {
        $Drift += "MISMATCH  ${name}: locked $($Locked[$name]), installed $($Installed[$name])"
    }
}
foreach ($name in ($Installed.Keys | Sort-Object)) {
    if (-not $Locked.ContainsKey($name)) {
        # An extra package changes the artifact too: PyInstaller bundles
        # whatever is importable (a stray torch install once re-shaped
        # the CUDA DLL set), so unlocked extras are drift, not noise.
        $Drift += "UNPINNED  $name==$($Installed[$name]) (installed but not in the lock)"
    }
}
if ($Drift.Count -gt 0) {
    Write-Host "Build venv diverges from constraints-release.txt:" -ForegroundColor Red
    foreach ($entry in $Drift) { Write-Host "  $entry" -ForegroundColor Red }
    throw "build venv $VenvDir diverges from constraints-release.txt ($($Drift.Count) difference(s)); build from a venv that matches the lock (-Venv .venv-release, see docs\RELEASE.md 'Release venv'), reinstall with -c constraints-release.txt, or regenerate the lock per its header if the change is intentional"
}
Write-Host "    venv matches the lock ($($Locked.Count) pins verified)" -ForegroundColor Green

Write-Host "==> Cleaning previous build" -ForegroundColor Cyan
# No SilentlyContinue: let "file in use" errors surface so the user
# knows to close VS Code / Explorer / etc. holding a handle on dist/.
if (Test-Path (Join-Path $Repo "dist")) {
    Remove-Item -Recurse -Force (Join-Path $Repo "dist")
}
# Keep build/release-gates: phase evidence recorded before this build (unit,
# real-deps) stays valid for the same clean HEAD, and preflight --release
# rejects frozen-phase evidence whose bundle inventory no longer matches.
# The previous installer record is dropped so a failed build cannot leave it.
if (Test-Path (Join-Path $Repo "build")) {
    Get-ChildItem -Force (Join-Path $Repo "build") |
        Where-Object { $_.Name -ne "release-gates" } |
        Remove-Item -Recurse -Force
}
if (Test-Path (Join-Path $ReleaseGates "installer.json")) {
    Remove-Item -Force (Join-Path $ReleaseGates "installer.json")
}

Write-Host "==> Running PyInstaller" -ForegroundColor Cyan
& $Python -m PyInstaller $Spec --noconfirm `
    --workpath (Join-Path $Repo "build") `
    --distpath (Join-Path $Repo "dist")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Host "==> Auto-sourcing VC++ runtime DLLs (vendor-vcredist.ps1)" -ForegroundColor Cyan
# llama-server.exe statically imports msvcp140/vcruntime140/vcruntime140_1/
# vcomp140; a clean Windows box lacks them and PyInstaller's copies land in
# _internal/ (not _internal/llama/, which is where a child process searches).
# Pull them from the local VS redist into third_party/llama.cpp/ so the
# manifest assert below finds them. Non-clobbering; runs before the assert.
& (Join-Path $PSScriptRoot "vendor-vcredist.ps1") `
    -AppRuntimeDir (Join-Path $Dist "_internal")
if ($LASTEXITCODE -ne 0) { throw "vendor-vcredist.ps1 failed" }

Write-Host "==> Verifying vendored llama.cpp against manifest (assert-llama-vendor.ps1)" -ForegroundColor Cyan
# installer/llama-manifest.txt is the single source of truth for the files
# llama-server.exe needs at runtime. The shared
# assert checks presence AND that each .exe/.dll is a real PE image, so
# neither a forgotten sidecar nor a leftover CI text stub can
# ship an installer whose translator dies at startup on the user's machine.
# CI smoke builds set YAMABIKO_ALLOW_STUB_LLAMA=1 to permit deliberate stubs.
$LlamaFiles = @(& (Join-Path $PSScriptRoot "assert-llama-vendor.ps1"))

Write-Host "==> Copying vendored llama.cpp into dist/_internal/llama/" -ForegroundColor Cyan
# Place under _internal/ so it's part of the PyInstaller-6.x bundle tree.
# This lets the NSIS uninstaller clean it up via a single RMDir /r on
# $INSTDIR\_internal. Copy each manifest file explicitly (not a *.dll
# glob) so the bundle matches the manifest exactly.
$LlamaTarget = Join-Path $Dist "_internal\llama"
New-Item -ItemType Directory -Force -Path $LlamaTarget | Out-Null
foreach ($f in $LlamaFiles) {
    Copy-Item (Join-Path $Vendor $f) $LlamaTarget
}

Write-Host "==> Auditing generated bundle inventory" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\audit_bundle_inventory.py") `
    --inventory $Inventory `
    --dist $Dist
if ($LASTEXITCODE -ne 0) { throw "audit_bundle_inventory.py failed" }

Write-Host "==> Resolving final-tree bundle provenance" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\bundle_provenance.py") resolve `
    --inventory $Inventory `
    --dist $Dist `
    --policy-config $ProvenanceConfig `
    --site-packages $SitePackages `
    --output $ProvenanceReport
if ($LASTEXITCODE -ne 0) { throw "bundle_provenance.py failed" }

Write-Host "==> Collecting bundle-derived license payload" -ForegroundColor Cyan
# The collector derives the required component set from the inventory and
# provenance report, then applies only the reviewed overrides in the manifest.
& $Python (Join-Path $Repo "scripts\collect_licenses.py") `
    --dist $Dist `
    --inventory $Inventory `
    --provenance-report $ProvenanceReport `
    --override-policy $LicenseOverrides `
    --site-packages $SitePackages
if ($LASTEXITCODE -ne 0) { throw "collect_licenses.py failed" }

Write-Host "==> Auditing independent bundle and license closure" -ForegroundColor Cyan
& $Python (Join-Path $Repo "scripts\audit_license_closure.py") `
    --dist $Dist `
    --inventory $Inventory `
    --license-index $LicenseIndex `
    --provenance-policy-config $ProvenanceConfig `
    --license-override-policy $LicenseOverrides `
    --site-packages $SitePackages `
    --output $ClosureAudit
if ($LASTEXITCODE -ne 0) { throw "audit_license_closure.py failed" }

Write-Host "==> Sanity smoke: launching dist/yamabiko/yamabiko.exe --version" -ForegroundColor Cyan
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

Write-Host "==> Running NSIS" -ForegroundColor Cyan
# Resolve makensis robustly: it is NOT on PATH on the dev box (it lives under
# Program Files\NSIS), so a bare `makensis` call fails the whole build at the
# very last step after ~5 min of PyInstaller (beta 2026-06-17). Prefer PATH,
# then the standard install locations.
$makensis = $null
$cmd = Get-Command makensis -ErrorAction SilentlyContinue
if ($cmd) { $makensis = $cmd.Source }
if (-not $makensis) {
    foreach ($p in @("${env:ProgramFiles(x86)}\NSIS\makensis.exe",
                     "$env:ProgramFiles\NSIS\makensis.exe")) {
        if (Test-Path $p) { $makensis = $p; break }
    }
}
if (-not $makensis) {
    throw "makensis not found on PATH or in Program Files\NSIS -- install NSIS (https://nsis.sourceforge.io) or add it to PATH"
}
& $makensis (Join-Path $PSScriptRoot "yamabiko.nsi")
if ($LASTEXITCODE -ne 0) { throw "NSIS failed" }

# Derive the output path from the file NSIS actually produced (its name
# comes from APP_VERSION) instead of a hardcoded version literal that
# silently breaks the success banner on the next version bump (audit build-1).
$Installer = Get-ChildItem (Join-Path $PSScriptRoot "yamabiko-*-setup.exe") |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Installer) { throw "NSIS reported success but no yamabiko-*-setup.exe was produced" }

Write-Host "==> Recording installer evidence (build\release-gates\installer.json)" -ForegroundColor Cyan
# preflight --release section 16 requires this record to match the installer
# bytes and the current HEAD, so an installer from an earlier build or commit
# cannot be published as the tested candidate.
$InstallerRecord = [ordered]@{
    version = ($Installer.BaseName -replace '^yamabiko-(.+)-setup$', '$1')
    installer_file = "installer/$($Installer.Name)"
    installer_sha256 = (Get-FileHash -Algorithm SHA256 $Installer.FullName).Hash.ToLowerInvariant()
    git_head = $GitHead
    git_dirty = $GitDirty
}
New-Item -ItemType Directory -Force -Path $ReleaseGates | Out-Null
[System.IO.File]::WriteAllText(
    (Join-Path $ReleaseGates "installer.json"),
    (($InstallerRecord | ConvertTo-Json) + "`n"),
    (New-Object System.Text.UTF8Encoding $false))
Write-Host "    sha256 $($InstallerRecord.installer_sha256) at $($InstallerRecord.git_head)"
Write-Host "==> Done. Installer at: $($Installer.FullName)" -ForegroundColor Green
