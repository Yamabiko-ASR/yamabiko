# Yamabiko installer silent-install → launch → uninstall smoke cycle.
#
# Runs locally and in CI. Assumes installer\build.ps1 has already
# produced installer\yamabiko-<version>-setup.exe. Exercises:
#
#   1. Silent install to a temp dir (-DestDir or $env:TEMP\yamabiko-ci-smoke)
#   2. yamabiko.exe --version from the INSTALLED location (catches
#      install-layout bugs that dist\yamabiko\ smokes miss, e.g.
#      missing NSIS File entries, wrong SetOutPath target)
#   3. yamabiko.exe --help from the installed location (exercises the
#      watchdog help path without touching the wizard)
#   3b. Reinstall over the existing install (stale _internal sentinel
#       must be purged by yamabiko.nsi's pre-clean; exe still answers
#       --version)
#   3c. Running-process guards (EnsureNotRunning): a fake yamabiko.exe
#       causes both installer and uninstaller to abort with NSIS script-
#       abort exit code 2; a fake llama-server.exe under $INSTDIR also
#       blocks the installer; a llama-server.exe OUTSIDE $INSTDIR (LM
#       Studio / ollama) does NOT block (path-scoping negative control)
#   3d. User-data sentinels planted before uninstall (user_overrides.yaml,
#       hf-cache blob, logs file); asserted to survive uninstall
#       (issue #23(a): the RMDir-without-/r data-preservation contract)
#   4. Silent uninstall via $DestDir\Uninstall.exe /S
#   5. Assertion that $DestDir\yamabiko.exe is gone (installer actually
#      removed its files, not just unregistered itself); and that user
#      data (config overrides, hf-cache, logs) was NOT deleted
#
# Does NOT exercise:
#   - yamabiko.exe --run-main full startup (needs GPU + llama-server,
#     neither of which exists on a windows-latest CI runner)
#   - The first-run wizard (GUI)
#   - Audio capture (needs real WASAPI loopback)
#
# These belong to the manual wizard E2E step in docs/RELEASE.md and
# to tests/test_real_frozen_e2e.py on GPU dev boxes.
#
# REGISTRY AND START MENU SAFETY: the installer writes the single shared
# HKCU uninstall key (yamabiko.nsi APP_REGKEY) and the Start Menu
# shortcuts ($SMPROGRAMS\Yamabiko) regardless of -DestDir, and the
# uninstaller deletes both, so a smoke run on a dev box would
# poison/destroy a real install's registration and shortcuts. The key is
# snapshotted and the Start Menu folder copied aside (or its absence
# recorded) before the install; both are restored in a finally block --
# on success, on throw, and on -KeepInstall -- so the machine's state is
# identical before and after a run. A stop (Ctrl+C) doesn't stop a
# running installer or uninstaller, so the finally block waits for it
# first (up to 10 min); if it is still running then, nothing is restored,
# the snapshots stay and the script prints the commands that restore
# them.
#
# Usage (from repo root):
#   .\installer\ci-smoke.ps1                        # uses default
#   .\installer\ci-smoke.ps1 -DestDir C:\tmp\test   # custom install dir
#   .\installer\ci-smoke.ps1 -KeepInstall           # skip uninstall
#                                                   # (inspection only;
#                                                   # registry and Start
#                                                   # Menu are still
#                                                   # restored)
#
# Exit codes:
#   0 = all steps passed
#   1 = install, launch, or uninstall failure (prints details)

[CmdletBinding()]
param(
    [string]$DestDir = (Join-Path $env:TEMP "yamabiko-ci-smoke"),
    [switch]$KeepInstall
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$InstallerGlob = Join-Path $Repo "installer\yamabiko-*-setup.exe"

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------

Write-Host "==> Locating built installer" -ForegroundColor Cyan
$Installer = Get-ChildItem $InstallerGlob -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $Installer) {
    throw "No installer found matching $InstallerGlob. Run installer\build.ps1 first."
}
Write-Host "    found: $($Installer.FullName) ($([math]::Round($Installer.Length / 1MB, 1)) MB)"

# Snapshot the shared HKCU uninstall key BEFORE the installer can touch
# it. Must match APP_REGKEY in installer\yamabiko.nsi.
$AppUninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko"
$RegSnapshot = Join-Path $env:TEMP "yamabiko-ci-smoke-uninstallkey.reg"
$SnapshotHelper = Join-Path $PSScriptRoot "snapshot-registry-key.ps1"
# The Start Menu folder is copied aside too (the per-user Programs folder
# is NSIS's $SMPROGRAMS for this RequestExecutionLevel user installer).
# Without a folder, a marker records that, so the smoke's own shortcuts
# that a killed run leaves behind are never taken for real ones.
$StartMenuDir = Join-Path ([Environment]::GetFolderPath("Programs")) "Yamabiko"
$StartMenuBackup = Join-Path $env:TEMP "yamabiko-ci-smoke-startmenu"
$StartMenuAbsent = "$StartMenuBackup.absent"

function ConvertTo-PsLiteral([string]$Text) {
    # $Text as a single-quoted PowerShell literal, whatever it holds: each
    # quote character in it is doubled (', as in C:\Users\O'Brien, and the
    # typographic ones PowerShell reads as quotes too).
    "'" + [System.Management.Automation.Language.CodeGeneration]::EscapeSingleQuotedStringContent($Text) + "'"
}

function Get-RestoreSteps {
    # For each snapshot there is now: the PowerShell command that puts the
    # state it holds back and deletes it. (-Action:restore is the helper's
    # restore action; the finally block below makes the script's own call.)
    $helperLit = ConvertTo-PsLiteral $SnapshotHelper
    $keyLit = ConvertTo-PsLiteral $AppUninstallKey
    $regLit = ConvertTo-PsLiteral $RegSnapshot
    $dirLit = ConvertTo-PsLiteral $StartMenuDir
    $backupLit = ConvertTo-PsLiteral $StartMenuBackup
    $absentLit = ConvertTo-PsLiteral $StartMenuAbsent
    if ((Test-Path $RegSnapshot) -or (Test-Path "$RegSnapshot.absent")) {
        "& $helperLit -Action:restore -KeyPath $keyLit -BackupFile $regLit"
    }
    if (Test-Path $StartMenuBackup) {
        "if (Test-Path $dirLit) { Remove-Item -Recurse -Force $dirLit -ErrorAction Stop }; Copy-Item -Recurse $backupLit $dirLit -ErrorAction Stop; Remove-Item -Recurse -Force $backupLit"
    }
    if (Test-Path $StartMenuAbsent) {
        "if (Test-Path $dirLit) { Remove-Item -Recurse -Force $dirLit -ErrorAction Stop }; Remove-Item $absentLit"
    }
}

# A run stopped before its finally block (a closed console, a tool
# deadline), or one whose installer outlived the finally block's wait,
# leaves its snapshots behind, holding the real state; saving over them
# would lose it. Refuse, before anything changes, and say how to restore.
$Leftovers = @(@($RegSnapshot, "$RegSnapshot.absent", $StartMenuBackup, $StartMenuAbsent) | Where-Object { Test-Path $_ })
if ($Leftovers.Count -gt 0) {
    throw ("An earlier smoke run left its snapshots: $($Leftovers -join ', '). " +
        "Once no installer or uninstaller of that run is still running, put the state they hold back and delete them (PowerShell):`n" +
        ((Get-RestoreSteps) -join "`n") + "`nThen run this script again.")
}

# Pre-clean any prior test install. `/D=` does NOT overwrite in place
# on NSIS; leftover files from a previous run would confuse the
# uninstall-verification assertion at the end.
if (Test-Path $DestDir) {
    Write-Host "==> Cleaning prior test install: $DestDir" -ForegroundColor Cyan
    Remove-Item -Recurse -Force $DestDir
}

Write-Host "==> Snapshotting $AppUninstallKey" -ForegroundColor Cyan
& $SnapshotHelper -Action save -KeyPath $AppUninstallKey -BackupFile $RegSnapshot

$StartMenuExisted = Test-Path $StartMenuDir
if ($StartMenuExisted) {
    Write-Host "==> Copying aside $StartMenuDir" -ForegroundColor Cyan
    Copy-Item -Recurse $StartMenuDir $StartMenuBackup
}
else {
    Set-Content -Path $StartMenuAbsent -Value "Start Menu folder absent at snapshot time" -Encoding ASCII
}

# Every installer and uninstaller run goes through Invoke-SmokeChild, which
# records the process here before waiting on it. A stop (Ctrl+C) ends the
# wait but not the child, so the finally block waits for the recorded
# children before restoring: a child still running would overwrite or
# delete what came back, after the snapshots were deleted.
$SmokeChildren = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

function Invoke-SmokeChild {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList
    )
    $proc = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -PassThru
    $SmokeChildren.Add($proc)
    # Touch .Handle before waiting (as in Invoke-InstalledExe below) so
    # .ExitCode is set; short waits let a stop through.
    $null = $proc.Handle
    while (-not $proc.WaitForExit(250)) {}
    $proc
}

function Wait-SmokeChildren {
    # Waits up to 10 minutes in all for the recorded children; returns the
    # ones still running.
    $deadline = (Get-Date).AddMinutes(10)
    foreach ($child in $SmokeChildren) {
        if (-not $child.HasExited) {
            Write-Host "==> Waiting for this smoke's PID $($child.Id) to exit before restoring" -ForegroundColor Yellow
        }
        while (-not $child.HasExited -and (Get-Date) -lt $deadline) {
            $null = $child.WaitForExit(250)
        }
    }
    @($SmokeChildren | Where-Object { -not $_.HasExited })
}

try {
    # -----------------------------------------------------------------------
    # 1. Silent install
    # -----------------------------------------------------------------------

    Write-Host "==> Silent install to: $DestDir" -ForegroundColor Cyan
    # NSIS gotcha: /D=<path> must be the LAST argument, must be absolute,
    # and MUST NOT be quoted (NSIS parses everything after /D= as a raw
    # literal path, including any quotes).
    $installProc = Invoke-SmokeChild -FilePath $Installer.FullName `
        -ArgumentList @("/S", "/D=$DestDir")
    if ($installProc.ExitCode -ne 0) {
        throw "Installer exited $($installProc.ExitCode); expected 0"
    }

    $InstalledExe = Join-Path $DestDir "yamabiko.exe"
    if (-not (Test-Path $InstalledExe)) {
        $contents = (Get-ChildItem $DestDir -ErrorAction SilentlyContinue | Out-String)
        throw "Installer reported success but $InstalledExe is missing. Install layout broken. Contents of ${DestDir}:`n$contents"
    }
    $InstalledUninst = Join-Path $DestDir "Uninstall.exe"
    if (-not (Test-Path $InstalledUninst)) {
        throw "Installer reported success but $InstalledUninst is missing. `
NSIS WriteUninstaller step failed or was skipped."
    }

    Write-Host ("    installed yamabiko.exe ({0:N0} bytes), Uninstall.exe ({1:N0} bytes)" -f `
        (Get-Item $InstalledExe).Length, (Get-Item $InstalledUninst).Length)

    # -----------------------------------------------------------------------
    # 2. Installed-exe --version smoke
    # -----------------------------------------------------------------------
    #
    # yamabiko.exe is a PyInstaller windowed bootloader (console=False in
    # installer/yamabiko.spec). PowerShell's `&` operator cannot reliably
    # capture exit code or stdout for windowed apps because the bootloader
    # detaches from the parent console. Use Start-Process -PassThru with
    # explicit RedirectStandardOutput/Error so the pipe setup is
    # guaranteed and $proc.ExitCode reflects the actual child's exit,
    # then enforce -TimeoutSec via WaitForExit.

    function Start-FakeProcess {
        # Copy cmd.exe under a target image name and keep it alive ~120 s
        # via ping (non-interactive; `timeout` rejects redirected stdin).
        # tasklist/Get-Process match on IMAGE NAME, so a renamed cmd.exe
        # is indistinguishable from the real thing for the nsi probes.
        param(
            [Parameter(Mandatory)][string]$ImagePath
        )
        New-Item -ItemType Directory -Force (Split-Path -Parent $ImagePath) | Out-Null
        $CmdExe = Join-Path ([Environment]::SystemDirectory) "cmd.exe"
        Copy-Item $CmdExe $ImagePath -Force
        Start-Process -FilePath $ImagePath `
            -ArgumentList @("/c", "ping -n 120 127.0.0.1 >nul") `
            -WindowStyle Hidden -PassThru
    }

    function Stop-FakeProcess {
        # Kill the fake and remove its directory. Retry the removal: the
        # image-file lock can outlive WaitForExit by a beat (AV on-access
        # scans), and a leftover fake under _internal\ would poison the
        # later scenarios with confusing failures -- so throw rather than
        # leave debris behind.
        param(
            [Parameter(Mandatory)]$Proc,
            [Parameter(Mandatory)][string]$ImagePath
        )
        # Kill() takes only the renamed-cmd.exe parent; its ping child is
        # deliberately left to expire on its own (~2 min) -- it matches no
        # probe image name and holds no lock in the fake dir. Kill($true)
        # would reap it but is pwsh-only; this script stays WinPS 5.1-safe.
        if (-not $Proc.HasExited) { $Proc.Kill() }
        if (-not $Proc.WaitForExit(10000)) {
            throw "fake process $ImagePath did not exit within 10 s of Kill()"
        }
        $fakeDir = Split-Path -Parent $ImagePath
        for ($i = 0; $i -lt 10; $i++) {
            Remove-Item -Recurse -Force $fakeDir -ErrorAction SilentlyContinue
            if (-not (Test-Path $fakeDir)) { return }
            Start-Sleep -Milliseconds 300
        }
        throw "Could not remove fake-process dir $fakeDir after 10 attempts; a leftover fake exe would corrupt later scenarios."
    }

    function Invoke-InstalledExe {
        # $Args is a PowerShell automatic variable in functions -- don't
        # shadow it. Use $CmdArgs for the exe argument list.
        param(
            [Parameter(Mandatory)][string]$ExePath,
            [Parameter(Mandatory)][string[]]$CmdArgs,
            [int]$TimeoutSec = 30
        )
        $outFile = New-TemporaryFile
        $errFile = New-TemporaryFile
        try {
            # Deliberately NOT the blocking wait switch on Start-Process:
            # that blocks with no timeout, turning a frozen-import
            # deadlock in the exe into a hung CI job, and made
            # $TimeoutSec dead code. Enforce the budget
            # ourselves via WaitForExit below.
            $proc = Start-Process -FilePath $ExePath `
                -ArgumentList $CmdArgs `
                -PassThru -NoNewWindow `
                -RedirectStandardOutput $outFile.FullName `
                -RedirectStandardError $errFile.FullName
            # Touch .Handle before waiting: without it .NET may never
            # cache the process handle and .ExitCode stays $null for a
            # child that exits quickly.
            $null = $proc.Handle
            if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
                $proc.Kill()
                throw "$ExePath $($CmdArgs -join ' ') still running after ${TimeoutSec}s -- killed (frozen-import deadlock?)"
            }
            $stdout = Get-Content $outFile.FullName -Raw -ErrorAction SilentlyContinue
            $stderr = Get-Content $errFile.FullName -Raw -ErrorAction SilentlyContinue
            [PSCustomObject]@{
                ExitCode = $proc.ExitCode
                Stdout   = ($stdout | Out-String).Trim()
                Stderr   = ($stderr | Out-String).Trim()
            }
        }
        finally {
            Remove-Item $outFile.FullName, $errFile.FullName -ErrorAction SilentlyContinue
        }
    }

    Write-Host "==> Smoke: installed yamabiko.exe --version" -ForegroundColor Cyan
    $versionResult = Invoke-InstalledExe -ExePath $InstalledExe -CmdArgs @("--version")
    if ($versionResult.ExitCode -ne 0) {
        throw "Installed yamabiko.exe --version exited $($versionResult.ExitCode). Stdout:`n$($versionResult.Stdout)`nStderr:`n$($versionResult.Stderr)"
    }
    if ($versionResult.Stdout -notmatch "Yamabiko") {
        throw "--version stdout missing 'Yamabiko' marker. Stdout:`n$($versionResult.Stdout)`nStderr:`n$($versionResult.Stderr)"
    }
    Write-Host "    $($versionResult.Stdout)"

    # -----------------------------------------------------------------------
    # 3. Installed-exe --help smoke
    # -----------------------------------------------------------------------

    Write-Host "==> Smoke: installed yamabiko.exe --help" -ForegroundColor Cyan
    $helpResult = Invoke-InstalledExe -ExePath $InstalledExe -CmdArgs @("--help")
    if ($helpResult.ExitCode -ne 0) {
        throw "Installed yamabiko.exe --help exited $($helpResult.ExitCode). Stdout:`n$($helpResult.Stdout)`nStderr:`n$($helpResult.Stderr)"
    }
    # The help text in yamabiko\watchdog.py mentions the Start-Menu launch
    # and the internal --run-main / --run-overlay dispatch flags. Assert
    # at least two of those are present so a stripped help string can't
    # silently pass.
    $expectedHelpMarkers = @("--version", "--help", "--run-main", "--run-overlay")
    $missingHelpMarkers = @($expectedHelpMarkers | Where-Object { $helpResult.Stdout -notmatch [regex]::Escape($_) })
    if ($missingHelpMarkers.Count -gt 0) {
        throw "--help stdout missing markers: $($missingHelpMarkers -join ', '). Stdout:`n$($helpResult.Stdout)"
    }
    Write-Host "    ok (found markers: $($expectedHelpMarkers -join ', '))"

    # -----------------------------------------------------------------------
    # 3b. Scenario A: reinstall over the existing install (issues #14/#23).
    #
    # Plant a stale file inside _internal\ first: yamabiko.nsi's SecMain
    # pre-clean (RMDir /r + FileExists postcondition) must purge it, or an
    # upgrade would ship a mixed-version _internal where a stale DLL
    # shadows a new one (the DLL-shadowing class via the upgrade path).
    # -----------------------------------------------------------------------

    Write-Host "==> Scenario A: reinstall over existing install" -ForegroundColor Cyan
    $StaleSentinel = Join-Path $DestDir "_internal\stale-from-previous-version.dll"
    Set-Content -Path $StaleSentinel -Value "stale" -Encoding ascii
    $reinstallProc = Invoke-SmokeChild -FilePath $Installer.FullName `
        -ArgumentList @("/S", "/D=$DestDir")
    if ($reinstallProc.ExitCode -ne 0) {
        throw "Reinstall over an existing install exited $($reinstallProc.ExitCode); expected 0. Upgrades are broken."
    }
    if (Test-Path $StaleSentinel) {
        throw "Reinstall left a stale previous-version file in _internal ($StaleSentinel). The yamabiko.nsi pre-clean regressed -- upgrades would ship mixed-version bundles (DLL-shadowing class)."
    }
    $reVersion = Invoke-InstalledExe -ExePath $InstalledExe -CmdArgs @("--version")
    if ($reVersion.ExitCode -ne 0 -or $reVersion.Stdout -notmatch "Yamabiko") {
        throw "Installed exe broken after reinstall-over-existing. Exit $($reVersion.ExitCode). Stdout:`n$($reVersion.Stdout)`nStderr:`n$($reVersion.Stderr)"
    }
    Write-Host "    reinstall ok; stale _internal file purged; --version still good"

    # -----------------------------------------------------------------------
    # 3c. Scenario B: running-process guards (issues #14/#21/#23).
    # B1: a process named yamabiko.exe blocks BOTH install and uninstall
    #     (nsi EnsureNotRunning tasklist probe), exit code 2 = NSIS
    #     script Abort. The silent MessageBox /SD IDCANCEL takes the
    #     Cancel -> Abort path with no UI.
    # B2: a llama-server.exe running FROM UNDER the install dir blocks
    #     the installer (path-scoped powershell probe).
    # B3: negative control -- a llama-server.exe running OUTSIDE the
    #     install dir must NOT block (LM Studio / ollama run binaries
    #     with the same generic image name).
    # -----------------------------------------------------------------------

    Write-Host "==> Scenario B1: install+uninstall refused while yamabiko.exe runs" -ForegroundColor Cyan
    $FakeYamabiko = Join-Path $env:TEMP "yamabiko-ci-fake\yamabiko.exe"
    $fakeProc = Start-FakeProcess -ImagePath $FakeYamabiko
    try {
        if ($fakeProc.HasExited) { throw "fake yamabiko.exe died before the blocked-install attempt; scenario invalid" }
        $blockedInstall = Invoke-SmokeChild -FilePath $Installer.FullName `
            -ArgumentList @("/S", "/D=$DestDir")
        if ($blockedInstall.ExitCode -ne 2) {
            throw "Installer ran to exit $($blockedInstall.ExitCode) while yamabiko.exe was running; expected NSIS abort (2). EnsureNotRunning regressed (issues #14/#21)."
        }
        if ($fakeProc.HasExited) { throw "fake yamabiko.exe died before the blocked-uninstall attempt; scenario invalid" }
        $blockedUninst = Invoke-SmokeChild -FilePath $InstalledUninst `
            -ArgumentList @("/S", "_?=$DestDir")
        if ($blockedUninst.ExitCode -ne 2) {
            throw "Uninstaller ran to exit $($blockedUninst.ExitCode) while yamabiko.exe was running; expected NSIS abort (2). un.EnsureNotRunning regressed (issue #21)."
        }
        if (-not (Test-Path $InstalledExe)) {
            throw "Blocked uninstall still deleted yamabiko.exe -- the un.onInit guard aborted too late."
        }
    }
    finally {
        Stop-FakeProcess -Proc $fakeProc -ImagePath $FakeYamabiko
    }
    Write-Host "    both refused with exit 2; install intact"

    Write-Host "==> Scenario B2: install refused while OUR llama-server.exe runs" -ForegroundColor Cyan
    $FakeLlamaOurs = Join-Path $DestDir "_internal\llama\ci-fake\llama-server.exe"
    $fakeLlama = Start-FakeProcess -ImagePath $FakeLlamaOurs
    try {
        if ($fakeLlama.HasExited) { throw "fake llama-server.exe (under install dir) died before the blocked-install attempt; scenario invalid" }
        $blockedByLlama = Invoke-SmokeChild -FilePath $Installer.FullName `
            -ArgumentList @("/S", "/D=$DestDir")
        if ($blockedByLlama.ExitCode -ne 2) {
            throw "Installer ran to exit $($blockedByLlama.ExitCode) while a llama-server.exe under `$INSTDIR was running; expected NSIS abort (2). The path-scoped probe regressed (issue #21)."
        }
        # Discriminator vs the SecMain pre-clean backstop (which ALSO
        # exits 2, but only after gutting _internal): a probe abort in
        # .onInit leaves the existing install untouched.
        if (-not (Test-Path (Join-Path $DestDir "_internal\llama\llama-server.exe"))) {
            throw "Blocked install still removed files from _internal -- the abort came from the SecMain pre-clean backstop, not the .onInit llama probe. The path-scoped probe is dead AND a blocked upgrade now breaks the existing install (issue #21)."
        }
    }
    finally {
        Stop-FakeProcess -Proc $fakeLlama -ImagePath $FakeLlamaOurs
    }
    Write-Host "    refused with exit 2"

    Write-Host "==> Scenario B3: install NOT blocked by an unrelated llama-server.exe" -ForegroundColor Cyan
    $FakeLlamaOther = Join-Path $env:TEMP "yamabiko-ci-other-llama\llama-server.exe"
    $fakeOther = Start-FakeProcess -ImagePath $FakeLlamaOther
    try {
        if ($fakeOther.HasExited) { throw "fake llama-server.exe (outside install dir) died before the negative-control install attempt; scenario invalid" }
        $ctrl = Invoke-SmokeChild -FilePath $Installer.FullName `
            -ArgumentList @("/S", "/D=$DestDir")
        if ($ctrl.ExitCode -ne 0) {
            throw "Installer refused (exit $($ctrl.ExitCode)) because of a llama-server.exe running OUTSIDE the install dir. The probe lost its path scoping -- every LM Studio / ollama user would be unable to install."
        }
    }
    finally {
        Stop-FakeProcess -Proc $fakeOther -ImagePath $FakeLlamaOther
    }
    Write-Host "    negative control passed (unrelated llama-server ignored)"

    # -----------------------------------------------------------------------
    # 4. Silent uninstall
    # -----------------------------------------------------------------------

    if ($KeepInstall) {
        Write-Host "==> -KeepInstall set; leaving install in place at $DestDir" -ForegroundColor Yellow
        Write-Host "    (the HKCU uninstall key and the Start Menu folder are still restored by the finally block)" -ForegroundColor Yellow
        exit 0
    }

    # -----------------------------------------------------------------------
    # 3d. Scenario C setup: plant user data that the uninstall MUST keep
    # (issue #23). yamabiko.nsi's Section Uninstall removes the bundle
    # (yamabiko.exe, _internal\, ...) then tries `RMDir "$INSTDIR"`
    # WITHOUT /r -- user data must make that final RMDir fail-soft and
    # survive in place.
    #
    # Deliberately AFTER the -KeepInstall early-exit (despite the 3d
    # numbering): planting is only needed on the path that actually
    # uninstalls, and a -KeepInstall inspection tree must not be
    # littered with behavior-valid override/cache/log debris.
    # -----------------------------------------------------------------------

    Write-Host "==> Scenario C: planting user-data sentinels before uninstall" -ForegroundColor Cyan
    $SentinelOverrides = Join-Path $DestDir "user_overrides.yaml"
    $SentinelCache     = Join-Path $DestDir "hf-cache\sentinel\model-blob.bin"
    $SentinelLog       = Join-Path $DestDir "logs\main-pipeline.log"
    Set-Content -Path $SentinelOverrides -Value "translation:`n  quant_tier: small" -Encoding utf8
    New-Item -ItemType Directory -Force (Split-Path -Parent $SentinelCache) | Out-Null
    Set-Content -Path $SentinelCache -Value "pretend-gguf-bytes" -Encoding ascii
    New-Item -ItemType Directory -Force (Split-Path -Parent $SentinelLog) | Out-Null
    Set-Content -Path $SentinelLog -Value "2026-06-12 sentinel log line" -Encoding utf8

    Write-Host "==> Silent uninstall" -ForegroundColor Cyan
    # Spawn the uninstaller via its COPY in the install dir. NSIS's
    # uninstaller self-relocates on launch (copies itself to a temp dir
    # and re-execs) so the original can be deleted, which means we
    # can't block on the uninstaller process directly -- it terminates
    # before the temp copy finishes. Poll for the uninstaller being
    # actually gone + the exe file being gone.
    $uninstProc = Invoke-SmokeChild -FilePath $InstalledUninst `
        -ArgumentList @("/S", "_?=$DestDir")
    # _?=$DestDir tells NSIS to NOT self-relocate, so the wait above works
    # and leftover files (Uninstall.exe itself) can be cleaned up by this
    # script. Without it, uninstall returns instantly and we race the
    # self-relocated process.
    if ($uninstProc.ExitCode -ne 0) {
        throw "Uninstaller exited $($uninstProc.ExitCode); expected 0"
    }

    # -----------------------------------------------------------------------
    # 5. Verify cleanup
    # -----------------------------------------------------------------------

    Write-Host "==> Verifying uninstall cleanup" -ForegroundColor Cyan
    # With _?=$DestDir, NSIS leaves Uninstall.exe + $DestDir behind
    # (the user is expected to delete them; we do that here). But the
    # bundle contents -- yamabiko.exe and the whole _internal\ tree --
    # must be gone. That's what the uninstaller's Section -Un is
    # responsible for. LICENSE / NOTICE ship UNDER _internal\ (the spec
    # datas' "." destination lands in PyInstaller's contents dir), never
    # at $DestDir root, so the _internal check covers them; root-level
    # LICENSE/NOTICE asserts here would pass on any uninstaller.
    $shouldBeGone = @(
        (Join-Path $DestDir "yamabiko.exe"),
        (Join-Path $DestDir "_internal")
    )
    $stillThere = @($shouldBeGone | Where-Object { Test-Path $_ })
    if ($stillThere.Count -gt 0) {
        throw "Uninstall left files behind: $($stillThere -join ', '). `
Contents of $DestDir (post-uninstall):`n" + (
            (Get-ChildItem $DestDir -Recurse -ErrorAction SilentlyContinue | Out-String)
        )
    }

    # Data-preservation contract: the uninstaller must NOT have touched
    # the user's config / model cache / logs (issue #23 -- the branch a
    # clean %TEMP% install could never exercise).
    $mustSurvive = @($SentinelOverrides, $SentinelCache, $SentinelLog)
    $lost = @($mustSurvive | Where-Object { -not (Test-Path $_) })
    if ($lost.Count -gt 0) {
        throw "Uninstall DELETED user data it must preserve: $($lost -join ', '). The yamabiko.nsi data-preservation contract (final RMDir without /r) regressed -- real users would lose their multi-GB model cache and settings."
    }
    Write-Host "    user-data sentinels survived uninstall"

    # Clean up the _?=$DestDir leftovers (Uninstall.exe + possibly empty $DestDir).
    if (Test-Path $DestDir) {
        Remove-Item -Recurse -Force $DestDir -ErrorAction SilentlyContinue
    }
}
finally {
    # Runs on success, on any throw above, on the -KeepInstall `exit 0`
    # (PowerShell executes finally before process exit) and on a stop
    # (Ctrl+C). The smoke install/uninstall mutated the shared HKCU
    # uninstall key; put back exactly what was there before (or delete it
    # if nothing was) so a real install's registration survives any smoke
    # run. A stop can land here while the smoke's installer or
    # uninstaller still runs; restore only once none does.
    $StillRunning = @(Wait-SmokeChildren)
    if ($StillRunning.Count -eq 0) {
        try {
            Write-Host "==> Restoring $AppUninstallKey" -ForegroundColor Cyan
            & $SnapshotHelper -Action restore -KeyPath $AppUninstallKey -BackupFile $RegSnapshot
        }
        finally {
            # Same for the Start Menu shortcuts, even when the key restore
            # throws: the smoke's folder goes, the copied one comes back.
            Write-Host "==> Restoring $StartMenuDir" -ForegroundColor Cyan
            if (Test-Path $StartMenuDir) { Remove-Item -Recurse -Force $StartMenuDir }
            if ($StartMenuExisted) {
                Copy-Item -Recurse $StartMenuBackup $StartMenuDir
                Remove-Item -Recurse -Force $StartMenuBackup
            }
            else {
                Remove-Item $StartMenuAbsent
            }
        }
    }
    else {
        Write-Host "==> Process $($StillRunning.Id -join ', ') that this smoke started is still running after 10 minutes: nothing was restored and the snapshots stay." -ForegroundColor Red
        Write-Host "    Once it has exited, put the state the snapshots hold back and delete them (PowerShell):" -ForegroundColor Red
        Get-RestoreSteps | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    }
}

Write-Host "==> All smoke steps passed." -ForegroundColor Green
