# Shared assert for the vendored llama.cpp file set.
#
# Verifies third_party\llama.cpp\ against installer\llama-manifest.txt:
#   1. PRESENCE -- every manifest file exists (the pre-existing guard
#      that lived inline in build.ps1 + rebuild-fast.ps1).
#   2. CONTENT -- every manifest .exe/.dll is a plausible PE image
#      (>= 4 KB and starts with the MZ magic). This catches the case
#      where ci-prepare-vendor-stubs.ps1 creates 27-byte TEXT stubs with the
#      exact manifest names, and every downstream guard was
#      existence-only -- so a stub created on a dev box after a partial
#      re-vendor could ship verbatim in a real release installer, where
#      llama-server (or a sidecar DLL) fails to load and translation is
#      dead on the user's machine.
#
# CI escape hatch: installer-smoke.yml builds with stubs DELIBERATELY
# (the real binaries are ~200 MB and not in the repo). Pass -AllowStubs
# or set YAMABIKO_ALLOW_STUB_LLAMA=1 to skip check 2 only -- presence is
# always enforced.
#
# On success, writes the manifest file names (one per line) to the
# output stream so callers can drive their copy loop from the same
# parse:   $LlamaFiles = @(& installer\assert-llama-vendor.ps1)
# On failure, throws -- which propagates to a calling script under
# $ErrorActionPreference = "Stop", and exits non-zero under -File.

[CmdletBinding()]
param(
    [string]$VendorDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "third_party\llama.cpp"),
    [string]$ManifestPath = (Join-Path $PSScriptRoot "llama-manifest.txt"),
    [switch]$AllowStubs
)

$ErrorActionPreference = "Stop"

if ($env:YAMABIKO_ALLOW_STUB_LLAMA -eq "1") {
    $AllowStubs = $true
}

$LlamaFiles = @(Get-Content $ManifestPath |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith("#") })
if ($LlamaFiles.Count -eq 0) {
    throw "$ManifestPath lists no files (empty or all-comments) -- refusing to build a bundle with no llama runtime."
}

# 1. Presence (always enforced, -AllowStubs included).
$missing = @($LlamaFiles | Where-Object { -not (Test-Path (Join-Path $VendorDir $_)) })
if ($missing.Count -gt 0) {
    throw "Vendored llama.cpp incomplete -- missing: $($missing -join ', '). See third_party/README.md + installer/llama-manifest.txt"
}

# 2. Content: manifest .exe/.dll entries must be plausible PE images.
if (-not $AllowStubs) {
    $stubs = @()
    foreach ($name in $LlamaFiles) {
        if ($name -notmatch '\.(exe|dll)$') { continue }
        $path = Join-Path $VendorDir $name
        $item = Get-Item $path
        if ($item.Length -lt 4096) {
            $stubs += $name
            continue
        }
        $magic = [byte[]]::new(2)
        $fs = [System.IO.File]::OpenRead($path)
        try {
            [void]$fs.Read($magic, 0, 2)
        }
        finally {
            $fs.Dispose()
        }
        if (-not ($magic[0] -eq 0x4D -and $magic[1] -eq 0x5A)) {  # 'MZ'
            $stubs += $name
        }
    }
    if ($stubs.Count -gt 0) {
        throw ("Vendored llama.cpp contains stub/non-PE binaries: $($stubs -join ', '). " +
            "A CI text stub (ci-prepare-vendor-stubs.ps1) or corrupt drop would ship a dead translator. " +
            "Re-vendor real binaries, or set YAMABIKO_ALLOW_STUB_LLAMA=1 / -AllowStubs for a deliberate stub build (CI smoke only).")
    }
}

# Success: emit the file list for the caller's copy loop.
$LlamaFiles
