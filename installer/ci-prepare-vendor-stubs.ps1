# Yamabiko CI: stub the vendored llama-server.exe + CUDA DLLs so
# installer\build.ps1 completes on a GitHub Actions runner without
# needing the real binaries (which are ~200 MB and not redistributed
# by the yamabiko repo).
#
# The stubs are empty files with the EXACT filenames the build
# expects. build.ps1 copies them into dist\yamabiko\_internal\llama\
# verbatim. CI tests that check bundle STRUCTURE pass; tests that
# actually launch llama-server would fail (but those require
# llama-server to be reachable at :8090, which CI skips).
#
# Usage (from repo root):
#   .\installer\ci-prepare-vendor-stubs.ps1
#
# Idempotent: re-running over existing stubs or real binaries is safe
# -- we only create files that don't already exist.
#
# CI-ONLY: refuses to run unless GITHUB_ACTIONS or CI is
# set. A stub created on a dev box after a partial re-vendor used to
# satisfy every downstream existence-only guard and could ship in a
# real release installer with a dead translator. To iterate on CI
# behavior locally, set $env:CI = 'true' explicitly -- and prefer
# -VendorDir pointing at a scratch dir, not the real vendor tree.

[CmdletBinding()]
param(
    [string]$VendorDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "third_party\llama.cpp")
)

$ErrorActionPreference = "Stop"

if ($env:GITHUB_ACTIONS -ne "true" -and $env:CI -ne "true") {
    throw ("ci-prepare-vendor-stubs.ps1 creates NON-FUNCTIONAL text stubs and refuses to run outside CI. " +
        "Set `$env:CI = 'true' to override for deliberate local CI iteration.")
}

$Vendor = $VendorDir

# Single source of truth for the vendored llama.cpp file set:
# installer/llama-manifest.txt (shared with build.ps1 + rebuild-fast.ps1
# so the stub list can never drift from what the build asserts/copies).
$stubFiles = Get-Content (Join-Path $PSScriptRoot "llama-manifest.txt") |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith("#") }

Write-Host "==> Ensuring vendor stub directory: $Vendor" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $Vendor | Out-Null

$created = 0
$skipped = 0
foreach ($name in $stubFiles) {
    $target = Join-Path $Vendor $name
    if (Test-Path $target) {
        # Preserve existing (possibly real) binary. Local dev boxes
        # drop real llama-server.exe + real DLLs here; we must not
        # clobber those when CI locally tries to iterate.
        $skipped++
        continue
    }
    # Non-empty byte so Windows doesn't treat the file as 0-length and
    # NSIS File /r doesn't produce zero-size archive entries that the
    # installer extracts as empty stubs. The exact text is a contract:
    # scripts/audit_bundle_dlls.py (is_allowed_ci_llama_stub) exempts
    # only files with this content from the PE import audits under
    # YAMABIKO_ALLOW_STUB_LLAMA=1. Keep the two in sync.
    Set-Content -Path $target -Value "CI stub (not a real binary)" -Encoding UTF8
    $created++
}

Write-Host ("==> Stub summary: created={0}, skipped(already-present)={1}" -f $created, $skipped) -ForegroundColor Green
