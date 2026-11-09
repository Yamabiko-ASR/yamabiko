# Save/restore one HKCU registry key subtree around a destructive
# operation.
#
# WHY: the NSIS installer writes the SINGLE shared uninstall key
# HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko
# (yamabiko.nsi APP_REGKEY) on every install, and DeleteRegKey's it on
# every uninstall. ci-smoke.ps1 installs to %TEMP%\yamabiko-ci-smoke,
# so without snapshot/restore a smoke run on a dev box overwrites the
# real install's InstallLocation (poisoning InstallDirRegKey: future
# interactive installs resolve $INSTDIR to %TEMP%) and, on success,
# unregisters the real install from Add/Remove Programs entirely.
#
# Usage:
#   snapshot-registry-key.ps1 -Action save    -KeyPath HKCU:\... -BackupFile x.reg
#   ... destructive work ...
#   snapshot-registry-key.ps1 -Action restore -KeyPath HKCU:\... -BackupFile x.reg
#
# save:    exports the key to BackupFile if it exists; otherwise writes
#          a "<BackupFile>.absent" marker recording that the key was not
#          there (so restore knows to delete, not import).
# restore: removes the key's CURRENT state, then re-imports the snapshot
#          (or, with an .absent marker, leaves it deleted). Errors if
#          neither snapshot nor marker exists -- restore without save is
#          a wiring bug, not a no-op.

[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet("save", "restore")][string]$Action,
    [Parameter(Mandatory)][string]$KeyPath,
    [Parameter(Mandatory)][string]$BackupFile
)

$ErrorActionPreference = "Stop"

# reg.exe takes HKCU\Sub\Key, PowerShell providers take HKCU:\Sub\Key.
$RegExeKey = $KeyPath -replace "^HKCU:\\", "HKCU\"
$AbsentMarker = "$BackupFile.absent"

function Invoke-RegExe {
    param([string[]]$RegArgs)
    # reg.exe writes its success line to stdout and errors to stderr;
    # under $ErrorActionPreference=Stop a redirected native stderr can
    # surface as a terminating error in Windows PowerShell, so relax
    # around the call and gate on the exit code instead.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & reg.exe @RegArgs 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "reg.exe $($RegArgs -join ' ') failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

if ($Action -eq "save") {
    if (Test-Path $KeyPath) {
        Invoke-RegExe @("export", $RegExeKey, $BackupFile, "/y")
        Write-Host "snapshot: exported $KeyPath -> $BackupFile"
    }
    else {
        Set-Content -Path $AbsentMarker -Value "key absent at save time" -Encoding ASCII
        Write-Host "snapshot: $KeyPath absent; marker at $AbsentMarker"
    }
}
else {
    if (-not (Test-Path $BackupFile) -and -not (Test-Path $AbsentMarker)) {
        throw "no snapshot at $BackupFile (and no .absent marker) -- 'save' never ran; refusing a blind restore"
    }
    # Drop whatever state the destructive operation left behind.
    if (Test-Path $KeyPath) {
        Remove-Item -Path $KeyPath -Recurse -Force
    }
    if (Test-Path $BackupFile) {
        Invoke-RegExe @("import", $BackupFile)
        Remove-Item $BackupFile
        Write-Host "snapshot: restored $KeyPath from $BackupFile"
    }
    else {
        Remove-Item $AbsentMarker
        Write-Host "snapshot: $KeyPath was absent before; left deleted"
    }
}
