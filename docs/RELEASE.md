# Yamabiko release checklist

Follow this before every `v*` tag. Every step is blocking: if a
step fails, fix it and restart from the top.

The checklist is heavy on real-path verification because Yamabiko
ships a frozen PyInstaller bundle that talks to a GPU, an audio
device, a vendored `llama-server.exe`, and the HuggingFace network.
Mocked unit tests pass while any of those seams drift, so the gate
below exercises the real ones.

## Release venv

The release build and every release phase below run in a dedicated Python
3.12 venv, `.venv-release`, that holds exactly the pins in
`constraints-release.txt` (plus pip and the editable `yamabiko`).
`installer\build.ps1` compares the build venv's `pip freeze` with the lock
and stops on any difference, so a dev `.venv` with extra packages is
refused, and a test that passes only because the dev venv has an unlocked
package fails here instead. `.gitignore` covers it (`.venv*/`).

Create it once, from the Python 3.12 the lock's header names, and install
nothing else into it:

```powershell
py -3.12 -m venv .venv-release
.venv-release\Scripts\python.exe -m pip install -r requirements-dev.txt -c constraints-release.txt
.venv-release\Scripts\python.exe -m pip install -e . -c constraints-release.txt
.venv-release\Scripts\python.exe -m pip install "pyinstaller>=6.10" -c constraints-release.txt
```

Don't upgrade pip in it (pip is outside the lock). After a lock change,
delete the folder and create it again. To check it without a build,
`.venv-release\Scripts\python.exe -m pip freeze --exclude-editable` must list
exactly the lock's pins.

`installer\build.ps1 -Venv .venv-release` and
`installer\rebuild-fast.ps1 -Venv .venv-release` build from it (`-Venv`
takes a folder name under the repo root or an absolute path; the default is
`.venv`, which CI uses). `scripts\preflight.py --venv .venv-release` runs its
real tier and DLL audit with that venv's Python and checks the license
audit against that venv's packages; with `--release` it must also be
started from that venv's Python. The release phase runner uses the Python
that starts it.

## Model repositories are public

Why: setup and the runtime download every model without a token, so a
private or gated repository fails every fresh install: the app gets the
same 401 a misnamed repository gives. A developer machine can hide that: an
`HF_TOKEN` in the environment authenticates every Hugging Face call, the
frozen app's included. (An `hf auth login` token does not reach the app:
the app moves `HF_HOME`, and the hub reads its token file from there.)

- [ ] If a pinned repository of ours is still private (Kodama-1,
      `Yamabiko-ASR/kodama-1`, until its first release), make it public
      now, at the start of the release session, before any phase below:
      on its Hugging Face page, Settings, Change repository visibility,
      Public. Or set `$env:HF_TOKEN` to a write token for this one command
      (the next item clears it; prefer the variable to `hf auth login`,
      whose stored token outlives the session), then run
      `.venv-release\Scripts\python.exe -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('Yamabiko-ASR/kodama-1', private=False)"`
- [ ] Clear the token variables in the shell you release from, keep them
      cleared for the whole session, and check every pinned repository
      anonymously:

  ```powershell
  Remove-Item Env:HF_TOKEN, Env:HF_TOKEN_PATH, Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue
  @'
  from huggingface_hub import HfApi
  from yamabiko import models
  pins = {(pin.repo, pin.revision) for pin in models.TRANSLATOR_PINS.values()}
  pins |= {(models.WHISPER_REPO, models.WHISPER_REVISION),
           (models.SILERO_REPO, models.SILERO_REVISION)}
  for repo, revision in sorted(pins):
      info = HfApi().model_info(repo, revision=revision, token=False)
      assert not info.private and not info.gated, (repo, info.private, info.gated)
      print("public", repo, info.sha)
  '@ | & .venv-release\Scripts\python.exe -
  ```

  -> one `public` line per repository (three today: the translator, the
  speech model, Silero). An error names the repository; do not go on.

## Pre-flight (automated)

Run every step below from a clean tree at the candidate commit. Each release
phase records `git rev-parse HEAD`, whether `git status --porcelain` was
empty, and the SHA-256 of `build/yamabiko/bundle-inventory.json`.
`installer\build.ps1` records the same commit and tree state for the
installer. `scripts/preflight.py --release` (section 16) rejects evidence
from another commit, from a dirty tree, or from an earlier bundle, so commit
first and re-run any phase whose evidence no longer matches. The ASR quality
gate below records its results in the private release-gate record
(`docs/_internal/RELEASE-GATE-v<version>.md`), so it doesn't change the
candidate.

- [ ] `origin/main` is an ancestor of the candidate:
      `git merge-base --is-ancestor origin/main HEAD; $LASTEXITCODE` -> 0.
      If not, `git merge origin/main`, check that `git diff HEAD^1 HEAD`
      prints nothing, push, and start over.
- [ ] CHANGELOG's first versioned section is `## [<version>] - <date>`
      under an empty `## [Unreleased]`, and
      `docs/release-notes/v<version>.md` holds the owner-approved notes.

- [ ] `.venv-release\Scripts\python.exe scripts\run_pytest_release_phase.py unit`
      -> exit 0
- [ ] `.venv-release\Scripts\python.exe scripts\run_pytest_release_phase.py real-deps`
      -> exit 0. Run this on the NVIDIA dev box: it exercises the real GPU,
      audio, Hugging Face, and uvicorn seams. The release runner records an
      independent collection and fails on skips (including a whole module
      skipped while it is collected, such as a module-level
      `pytest.importorskip`), xfails, missing reports, or collection drift.
      Evidence is written to `build/release-gates/`. The speech-evidence seam
      test (`tests/e0_evidence/test_real_e0_seam.py`) is not part of this
      phase: it writes a run into the private evidence root, so it runs only
      on request, with `YAMABIKO_E0_RUN_SEAM=1`, `YAMABIKO_E0_PRIVATE_ROOT`
      and `YAMABIKO_E0_REAL_FIXTURE_ID` set.
- [ ] `.venv-release\Scripts\python.exe scripts\run_pytest_release_phase.py real-gui`
      -> exit 0. Run this on the dev box at an unlocked desktop and leave
      mouse and keyboard alone while it runs: it opens real windows on the
      native Windows Qt platform (caption, DWM, clipboard, keyboard hook,
      layout with real fonts). These tests need the native platform from the
      first QApplication of their process, so they run here and not in
      `real-deps`, whose first Qt test starts the offscreen platform. The
      phase lists `test_real_shortcut_field.py` first: it types into a window
      that must hold the foreground, and Windows lets a new process take the
      foreground only before its earlier windows hand it to another app. It
      ends with `tests/lifecycle/test_real_show_in_folder.py`, which opens an
      Explorer window with the saved file selected; the GitHub-hosted runner
      opens that window without the selection, so CI and `real-deps` leave
      it to this phase.
- [ ] `.venv-release\Scripts\python.exe scripts\preflight.py --venv .venv-release` -> exit 0. ONE command, must be green. Its 16
      sections:
      1. Hugging Face repo and file existence
      2. `hf_hub_download` signature
      3. Version consistency (strings and binary VERSIONINFO)
      4. Lazy imports
      5. Vendored llama-server
      6. Audio devices
      7. PyInstaller spec parse
      8. Pre-build-safe real tier (`pytest -m "real and not frozen_artifact"`)
      9. Bundle DLL deep audit (`scripts/audit_bundle_dlls.py`, the
         missing-CUDA-DLL class)
      10. Config key producer/consumer match
      11. Bundle module to launcher coupling
      12. CHANGELOG migration-notes gate
      13. No subject-mocks in the `@pytest.mark.real` tier
      14. Bundled llama-server DLL closure (`installer/llama-manifest.txt`)
      15. Bundle license closure audit evidence
          (`build/yamabiko/license-closure-audit.json` must describe the
          current bundle and every audit input)
      16. Release evidence binding (every phase's
          `build/release-gates/<phase>-evidence.json` passed at HEAD from a
          clean tree, the frozen phases tested the current bundle inventory,
          and `build/release-gates/installer.json` matches the installer
          bytes and HEAD)
- [ ] `installer\build.ps1 -Venv .venv-release` -> exit 0 (it first checks the venv against the lock), produces
      `installer/yamabiko-<version>-setup.exe` and records its SHA-256, HEAD
      and tree state in `build/release-gates/installer.json`. The build
      clears the rest of `build/` but keeps the phase evidence already in
      `build/release-gates/`.
- [ ] `.venv-release\Scripts\python.exe scripts\run_pytest_release_phase.py frozen-bundle`
      -> exit 0 against that exact fresh candidate
- [ ] With the external llama server available on port 8090, run
      `.venv-release\Scripts\python.exe scripts\run_pytest_release_phase.py frozen-external`
      -> exit 0
- [ ] Stop the external llama server so Yamabiko can own port 8090, then run
      `.venv-release\Scripts\python.exe scripts\run_pytest_release_phase.py frozen-owned`
      -> exit 0
- [ ] Re-run `.venv-release\Scripts\python.exe scripts\preflight.py --release --venv .venv-release` after the build and the frozen
      phases -> exit 0. `--release` turns sections 8, 9, 14, 15 and 16 into
      hard requirements: section 8 must execute at least one real-tier test;
      sections 9 and 14 FAIL instead of skipping when `dist/yamabiko/` is
      absent; section 15 FAILs unless the license closure audit matches the
      exact candidate; section 16 FAILs unless every phase above and the
      installer are bound to HEAD from a clean tree and the frozen phases
      tested the current bundle. Without `--release`, a pre-build run may
      skip these sections and prints "NOT a release-grade run", so the gate
      cannot be passed by accident before the bundle is built. The pass
      banner deliberately does not say "safe to release": preflight checks
      bundle presence, static gates and the recorded evidence, while the
      frozen E2E phases above are the only checks that catch
      silently-disabled romanization from a namespace-shadowed dictionary
      package in the frozen bundle.
- [ ] `dist\yamabiko\yamabiko.exe --version` -> prints matching version

## ASR quality gate (informational)

This step measures speech recognition with the speech-evidence tool in
`scripts/e0_evidence/`. Its operating runbook holds private fixture
details, so it is kept outside the public repository; the section numbers
below refer to it.

For now the step is informational: it runs on the clips whose human
references are frozen at release time, its numbers go in the private
release-gate record, and it does not block the tag. Real pass criteria are
set once the locked-confirmation clips L1 to L4 are annotated.

The tool replays audio through the source tree with the release `.venv`
(runbook §9), so it is source-tree evidence. Confirming the same
measurement against the frozen bundle is a separate future gate that this
checklist does not cover yet.

- [ ] At the candidate commit, run every clip with a frozen reference once,
      P0 arm, with an empty config file (the shipped defaults), under a new
      cohort ID, and verify each run (runbook §5):

  ```powershell
  .venv\Scripts\python.exe -m scripts.e0_evidence run --request <request.json>
  .venv\Scripts\python.exe -m scripts.e0_evidence verify-run --run-dir <private-root>\runs\<RUN-ID>
  ```

- [ ] Score the cohort with one `--run-id` per clip, then render the
      private report (runbook §7):

  ```powershell
  .venv\Scripts\python.exe -m scripts.e0_evidence score --run-id <RUN-ID-1> --run-id <RUN-ID-2> ...
  .venv\Scripts\python.exe -m scripts.e0_evidence report --report-dir <private-root>\reports\<REPORT-ID>
  ```

- [ ] When the release changes the speech model or its decode settings,
      run the same clips for the previous default as its own cohort (a
      config file that names its model, `initial_prompt` and
      `no_speech_only_threshold`; runbook §5, "Benchmarking a candidate CT2
      Whisper model"), score it the same way, and compare:
      `.venv\Scripts\python.exe -m scripts.e0_evidence compare --baseline-report <PREVIOUS-REPORT-ID> --candidate-report <REPORT-ID>`
- [ ] In the private release-gate record, record the number of clips (never
      their titles), each report ID and its `scorer_source_sha256`, the P0
      values of `micro.lexically_usable_recall`, `micro.cer_minimal`,
      `micro.cer_words` and `micro.hallucinations.count`, and, when
      compared, the `comparison_id` with each contrast and its interval.
      The release notes only say, in one plain sentence, that the model was
      checked against hand-made transcripts and compared with the previous
      model.

## Fast dev loop (for iterating on frozen-exe bugs)

When fixing a bug that only reproduces in the bundle, the full
`installer\build.ps1` round trip is about 29 min (2.5 min PyInstaller,
about 24 min NSIS). Most of that is NSIS LZMA compression, which you
don't need for running the exe.

Use `installer\rebuild-fast.ps1` instead:

- Keeps `build/` (PyInstaller's module-graph cache) intact
- Wipes only `dist/yamabiko/` and rebuilds there
- Skips NSIS entirely (no installer `.exe` produced)
- Runs the `--version` smoke

Expected wall time on a warm cache: **20-30 s** (vs about 29 min full).

Iterate on frozen-exe bugs like this:

```powershell
# 1. edit code
# 2. fast rebuild
.\installer\rebuild-fast.ps1

# 3. run the bundle surface against the fresh bundle
.venv\Scripts\python.exe scripts\run_pytest_release_phase.py frozen-bundle

# 4. run both E2E partitions (external llama first, then stop it)
.venv\Scripts\python.exe scripts\run_pytest_release_phase.py frozen-external
.venv\Scripts\python.exe scripts\run_pytest_release_phase.py frozen-owned
```

`rebuild-fast.ps1` builds from `.venv` unless you pass `-Venv`; it does not
check the venv against the lock, so a dev bundle can carry packages the
release build refuses.

That fast loop is what lets you converge on frozen-env-only bugs
without a half-hour per-iteration tax. When the fix lands, run the full
`installer\build.ps1 -Venv .venv-release` once at the end to produce the
release installer.

## Seeded-defect drill (mandatory before every tag)

Operationalises Fowler's principle: *"If my tests never failed,
either my programming is perfect or my tests are inadequate."* See
`docs/TESTING.md §6.1` for the full rationale.

On a throwaway branch, pick one commit from the rotation below. Revert it.
Run the relevant phase through `scripts/run_pytest_release_phase.py`, plus
preflight when the rotation row names it:
`.venv-release\Scripts\python.exe scripts\preflight.py --venv .venv-release`
or `.venv-release\Scripts\python.exe scripts\preflight.py --release --venv .venv-release`.
Expected: the row's detector is RED. If it stays green, it is not actually
exercising the defect: **do not tag** until the detector is fixed.

Rows 8 and 9 are NSIS-**edit** drills, not commit reverts, and their red
signal is the installer smoke (for row 9, `tests/test_installer_nsi.py`
also turns red; the smoke must still throw). Copy the candidate installer
aside, seed the edit, and rebuild the installer on the candidate's `dist\`
(about 24 min; `build.ps1` would also rewrite
`build\release-gates\installer.json`). From the repo root, with makensis
found as `build.ps1` finds it (PATH, then Program Files):

```powershell
$makensis = @((Get-Command makensis -ErrorAction SilentlyContinue).Source, "${env:ProgramFiles(x86)}\NSIS\makensis.exe", "$env:ProgramFiles\NSIS\makensis.exe") | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1; if (-not $makensis) { throw "makensis not found on PATH or in Program Files\NSIS" }; & $makensis installer\yamabiko.nsi
```

Run `installer\ci-smoke.ps1` and expect the named throw. Then revert the
edit and copy the candidate back (`Copy-Item` keeps its bytes and time, so
`installer.json` still matches it) instead of rebuilding.
`ci-smoke.ps1` saves and restores the uninstall registry key and the Start
Menu's Yamabiko folder itself, and refuses to start over a snapshot an
earlier stopped run left behind, so a local run leaves an installed
Yamabiko's registration and shortcuts as they were. It deletes its copy
of the folder as soon as it has put it back, so keep one of your own until
the end of the release session: the last step of Tag + publish checks the
shortcuts against it, then deletes it. Before the smoke:

```powershell
$Key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko"; $Menu = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Yamabiko"
$Kept = "C:\tmp\yamabiko-drill-$(Get-Date -f yyyyMMdd-HHmm)"; New-Item -ItemType Directory $Kept | Out-Null
if (Test-Path $Menu) { Copy-Item -Recurse $Menu "$Kept\startmenu-backup" }
@("InstallLocation: $((Get-ItemProperty $Key -ErrorAction SilentlyContinue).InstallLocation)") +
    @(Get-ChildItem $Menu -Filter *.lnk -ErrorAction SilentlyContinue | ForEach-Object { $l = (New-Object -ComObject WScript.Shell).CreateShortcut($_.FullName); "$($_.Name): $($l.TargetPath) | $($l.Arguments) | $($l.WorkingDirectory)" }) |
    Set-Content "$Kept\before.txt"
```

Rotation (advance by one per release; wrap at the end):

| # | Revert | Test that must go red |
|---|--------|----------------------|
| 1 | `26c9bed` signal-on-class fix | `tests/overlay_app/test_main_signal_connects.py` |
| 2 | `c3f41a7` unidic_lite + cutlet (`git revert c3f41a7` no longer applies: by hand, set `installer/yamabiko.spec` `unidic_datas = collect_data_files("unidic_lite")`, delete the `cutlet_datas = ...` line and `*cutlet_datas,` from `datas`) | `tests/test_real_frozen_bundle.py` |
| 3 | `10f3e91` cuBLAS/cuDNN/NVRTC flatten | preflight §9 |
| 4 | `502a335` frozen-aware glossary path | `test_frozen_exe_starts_without_errors_for_45s` |
| 5 | `1d24021` `_SafeTextIO` wrapper (moved to the top of `yamabiko/watchdog.py` in phase 7, so `git revert` no longer applies: in `_guard_frozen_stdio` keep the `os.devnull` sink and drop the `_SafeTextIO(...)` around both streams by hand) | `tests/test_real_frozen_e2e.py::test_frozen_exe_start_menu_launch_translates_and_log_stays_clean` |
| 6 | `36c5aee` watchdog triple-spawn | `test_frozen_exe_default_launch_spawns_pipeline_and_overlay` |
| 7 | `2fcaa18` Silero/Whisper path resolve | `test_real_onnxruntime.py` + bundle E2E |
| 8 | Edit `installer/yamabiko.nsi`: change final `RMDir "$INSTDIR"` → `RMDir /r "$INSTDIR"` | `installer\ci-smoke.ps1` Scenario C must throw ("DELETED user data it must preserve") |
| 9 | Edit `installer/yamabiko.nsi` `.onInit`: comment out the `Call EnsureNotRunning` inside `${If} ${Silent}` | `installer\ci-smoke.ps1` Scenario B1 must throw (installer exit 0 instead of 2 while fake yamabiko.exe runs). The smoke is silent, so it never reaches the interactive call in `WelcomeLeave`; the "Installer pages" step below covers that one. |
| 10 | `228145b` (#49; the original commit was squashed) llama CJK-path workaround: revert `_build_llama_cmd` to unconditional `--model str(model_path)` (drop the basename branch) | unit: `tests/test_watchdog.py::test_build_llama_cmd_cjk_path_uses_basename_and_parent_cwd`; bundle proof: `tests/test_real_cjk_model_paths.py::test_llama_server_loads_gguf_from_cjk_path` (real CUDA llama-server "failed to open GGUF file"). No NSIS/live drill; unit + real-probe only. |
| 11 | Edit `yamabiko/watchdog.py` `_guard_frozen_stdio`: leave a `None` stream `None` (wrap only the streams that exist, the pre-phase-7 form) | `tests/test_real_frozen_e2e.py::test_frozen_no_stderr_pipeline_wizard_download_reports_progress` (`hf_hub_download FAILED` with `'NoneType' object has no attribute 'write'`); unit: `tests/test_watchdog.py::test_frozen_stdio_guard_gives_a_none_stderr_a_sink_a_live_tqdm_bar_writes_to` |

The drill takes ~10 min (row 8 or 9: one `makensis` build, about 24 min,
plus the smoke, about 10 min) and must pass before the tag. If a rotation
entry's test fails to go red when its fix is reverted, the test has
drifted: rewrite the test first, then re-run the drill.

- [ ] Drill complete; reverted commit's test went RED; branch discarded.

## Clean-install wizard E2E

Why: mocked tests pass while the real setup still freezes, crashes, or
downloads at 3 MB/s. This is the release-blocker step.

It installs the candidate into a new folder so the Yamabiko already on the
box stays as it is: the installer reuses an installed folder unless `/D=`
names another, rewrites the one uninstall registry key, overwrites the
Start Menu shortcuts, and ignores `YAMABIKO_APPDATA` (only the app reads
it, at runtime). Before you start: no `yamabiko.exe` and no
`llama-server.exe` running (`tasklist`), port 8090 free, Hugging Face
reachable and every pinned model repository public (Model repositories
are public, above), an audio output you can play Japanese audio to, and
about 12 GB free on the drive you use below.

1. Back up what the installer touches, then install into a new folder
   (PowerShell, repo root; `<version>` as in the installer's file name;
   the path must have no spaces):

   ```powershell
   $V = "<version>"; $Root = "C:\tmp\yamabiko-release-test-$V-$(Get-Date -f yyyyMMdd-HHmm)"
   $Dest = "$Root\app"; $AppData = "$Root\appdata"; New-Item -ItemType Directory -Force $AppData | Out-Null
   $Key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko"
   $Menu = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Yamabiko"
   @("InstallLocation: $((Get-ItemProperty $Key -ErrorAction SilentlyContinue).InstallLocation)") +
       @(Get-ChildItem $Menu -Filter *.lnk -ErrorAction SilentlyContinue | ForEach-Object { $l = (New-Object -ComObject WScript.Shell).CreateShortcut($_.FullName); "$($_.Name): $($l.TargetPath) | $($l.Arguments) | $($l.WorkingDirectory)" }) |
       Set-Content "$Root\before.txt"
   & .\installer\snapshot-registry-key.ps1 -Action save -KeyPath $Key -BackupFile "$Root\uninstall-key.reg"
   if (Test-Path $Menu) { Copy-Item -Recurse $Menu "$Root\startmenu-backup" }
   (Start-Process ".\installer\yamabiko-$V-setup.exe" -ArgumentList @("/S", "/D=$Dest") -Wait -PassThru).ExitCode   # 0
   ```

   (`/D=` comes last, absolute and unquoted. Without `"/S"` the
   installer's own pages show, and `/D=` still applies. Its last page's
   Start Yamabiko then starts the candidate with the installer's
   environment, which has no `YAMABIKO_APPDATA` yet, so it would run on
   the real data. Untick it, or run step 2's `$env:` and `Remove-Item`
   lines in this shell before you start the installer and let that page
   do step 2's start.)
2. Start the installed app from this shell with its own data folder (a
   Start Menu click would not see `YAMABIKO_APPDATA`):

   ```powershell
   $env:YAMABIKO_APPDATA = $AppData
   Remove-Item Env:HF_HOME, Env:HF_HUB_CACHE, Env:HUGGINGFACE_HUB_CACHE, Env:HF_XET_CACHE, Env:HF_HUB_OFFLINE, Env:HF_TOKEN, Env:HF_TOKEN_PATH, Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue
   Start-Process "$Dest\yamabiko.exe"
   ```

- [ ] "Yamabiko setup" opens, 880 × 600, with the steps on the left.
- [ ] Welcome: the version matches, and there is no line about an update
      (a fresh install); Get started.
- [ ] Sound: the Windows default output is marked System default and
      selected. Play Japanese audio, click Test for 5 seconds: the meter
      moves, it says "Picking up sound." beside the meter's loudest level, nothing freezes.
- [ ] Content: pick a genre; the sample's banner and line follow it.
- [ ] Quality: the graphics card line names your card and its memory, and
      "Recommended for your card" is on the tier that fits it.
- [ ] Download: Start download. The overall line counts GB ("1.2 of 6.6
      GB") with the time left, the rows fill from the top, more than
      8 MB/s on gigabit; Next turns on when all three rows are done.
- [ ] Translator: "Starting the translator", then "Test translation" with
      こんにちは and its English; Next.
- [ ] Done: "You're all set." and your shortcuts. Start Yamabiko: setup
      closes and Yamabiko starts (tray icon).
- [ ] Run 30 sec of Japanese audio from any source (YouTube, VOD), verify overlay renders with furigana + translation
- [ ] Kill the **pipeline** `yamabiko.exe` via Task Manager and confirm the watchdog respawns it within about 15 s (the overlay comes back with it). Three roles share the `yamabiko.exe` image name when frozen; kill the one whose command line contains `--run-main` (Task Manager Details tab, add the *Command line* column). The overlay child carries `--run-overlay`; the bare watchdog has neither. Killing the watchdog instead just stops everything and is not this test.
- [ ] Only when this release moves `WHISPER_REPO` or `WHISPER_REVISION`:
      the upgrade path. Quit from the tray, delete the speech model from
      this test's cache, and start the app again as in step 2:

  ```powershell
  $Speech = (& .venv-release\Scripts\python.exe -c "from yamabiko.models import WHISPER_REPO; print(WHISPER_REPO)").Replace('/', '--')
  Remove-Item -Recurse -Force "$AppData\hf-cache\hub\models--$Speech"
  Start-Process "$Dest\yamabiko.exe"
  ```

  Setup opens again although `state.json` says it finished (this is what
  an upgraded install sees). Welcome: when this release has Welcome's
  update line, it adds "This update brings a new speech model. The
  download is 1.5 GB." (the size is the total still to download, here only
  the speech model); without it, Welcome is
  as on a fresh install. Sound has the output you picked selected; Quality
  keeps your tier; Download fetches only Speech recognition (about
  1.5 GB), while Translation model and Voice detection finish without
  downloading; Translator and Done as above; Yamabiko starts, and 30 sec
  of Japanese audio get subtitles.
- [ ] Quit from the tray, then uninstall (`Settings > Apps` now lists this
      test install, or silently):
      `(Start-Process "$Dest\Uninstall.exe" -ArgumentList @("/S", "_?=$Dest") -Wait -PassThru).ExitCode` -> 0.
      `$AppData` still holds `hf-cache`, `logs`, `state.json` and
      `user_overrides.yaml`.
- [ ] Install again into `$Dest` (step 1's `Start-Process` line) and start
      it as in step 2: Yamabiko starts without setup. Quit from the tray
      and uninstall again.
- [ ] Put the backups back:

  ```powershell
  & .\installer\snapshot-registry-key.ps1 -Action restore -KeyPath $Key -BackupFile "$Root\uninstall-key.reg"
  if (Test-Path $Menu) { Remove-Item -Recurse -Force $Menu }
  if (Test-Path "$Root\startmenu-backup") { Copy-Item -Recurse "$Root\startmenu-backup" $Menu }
  Remove-Item Env:YAMABIKO_APPDATA
  ```

  Then delete everything in `$Root` except `startmenu-backup` and
  `before.txt`. Keep those two until the end of the release session: the
  last step of Tag + publish checks the shortcuts against them.

## Installer pages

Why: the installer smoke runs silently, so it never shows a page. This
walks every page of the candidate installer and uninstaller by hand:
Welcome and the Still running box raised from it, License, Installing,
Finish and its Start Yamabiko, Update, Remove, Removed with the size it
measures, and Cancel.

Like the wizard E2E it installs into new folders and backs up the
uninstall registry key and the Start Menu's Yamabiko folder first. It also
removes the key for the run: while the key shows an installed version, the
installer opens on its Update page instead of Welcome. Before you start: no
`yamabiko.exe` and no `llama-server.exe` running (`tasklist`), about 15 GB
free on the drive you use below, and the previous release's installer
(`yamabiko-<previous>-setup.exe`) for the update. v0.1.0-beta:
`gh release download v0.1.0-beta -R MeowCatboyMeow/yamabiko -p '*setup.exe'`
(the private development repository); later releases are on
`Yamabiko-ASR/yamabiko`.

1. Back up, remove the key, and start a stand-in `yamabiko.exe` (a renamed
   `cmd.exe`, as the smoke uses) for the Still running box. Then start the
   installer with its pages (PowerShell, repo root; the path must have no
   spaces):

   ```powershell
   . {
       $ErrorActionPreference = "Stop"   # a failed backup must stop everything below
       $V = "<version>"; $PrevSetup = "<path>\yamabiko-<previous>-setup.exe"
       $Root = "C:\tmp\yamabiko-pages-test-$V-$(Get-Date -f yyyyMMdd-HHmm)"
       $Dest = "$Root\app"; $Old = "$Root\old"
       New-Item -ItemType Directory -Force "$Root\appdata", "$Root\fake" | Out-Null
       $Key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko"
       $Menu = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Yamabiko"
       # The state the last step must find again: the key's install folder
       # and what each shortcut opens (not its bytes: see the last step of
       # Tag + publish).
       @("InstallLocation: $((Get-ItemProperty $Key -ErrorAction SilentlyContinue).InstallLocation)") +
           @(Get-ChildItem $Menu -Filter *.lnk -ErrorAction SilentlyContinue | ForEach-Object { $l = (New-Object -ComObject WScript.Shell).CreateShortcut($_.FullName); "$($_.Name): $($l.TargetPath) | $($l.Arguments) | $($l.WorkingDirectory)" }) |
           Set-Content "$Root\before.txt"
       Get-Content "$Root\before.txt"
       & .\installer\snapshot-registry-key.ps1 -Action save -KeyPath $Key -BackupFile "$Root\uninstall-key.reg"
       if (Test-Path $Menu) { Copy-Item -Recurse $Menu "$Root\startmenu-backup" }
       if ((Test-Path $Menu) -and -not (Test-Path "$Root\startmenu-backup")) { throw "no Start Menu backup; nothing changed" }
       if (Test-Path "$Root\uninstall-key.reg") { Remove-Item -Recurse $Key }
       elseif (-not (Test-Path "$Root\uninstall-key.reg.absent")) { throw "no key backup; nothing changed" }
       $env:YAMABIKO_APPDATA = "$Root\appdata"
       Remove-Item Env:HF_HOME, Env:HF_HUB_CACHE, Env:HUGGINGFACE_HUB_CACHE, Env:HF_XET_CACHE, Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
       Copy-Item "$([Environment]::SystemDirectory)\cmd.exe" "$Root\fake\yamabiko.exe"
       $Fake = Start-Process "$Root\fake\yamabiko.exe" -ArgumentList '/c', 'ping -n 600 127.0.0.1 >nul' -WindowStyle Hidden -PassThru
       Start-Process ".\installer\yamabiko-$V-setup.exe" -ArgumentList "/D=$Dest"
   }
   ```

   Paste the block whole: `. { }` makes it one command, so its guards
   stop everything below them even in a console that sends a paste a line
   at a time, and its variables stay in the session. The first lines it
   prints are the starting state, also saved to `$Root\before.txt`.

   Start Yamabiko starts the app with the installer's environment, so it
   uses `$Root\appdata`. Run every step below from this same PowerShell
   window (it holds `$Root`, `$Key`, `$Menu` and `YAMABIKO_APPDATA`). If the
   window closes before the last step, open a new one, set `$Root`, `$Key`
   and `$Menu` again and run the last step: the backups and
   `$Root\before.txt` are still there.

- [ ] Welcome: caption "Install Yamabiko <version>" (the version with
      spaces for dashes), the light landscape with the seal and the echo
      arcs, "Welcome to Yamabiko" and its two lines; Next and Cancel; no
      text on the line above the buttons.
- [ ] Next: "Yamabiko is still running. Close Yamabiko setup, or right-click
      its tray icon and choose Quit Yamabiko, then click Retry." over the
      window. Cancel: Welcome stays, its text unchanged. Stop the stand-in
      (`taskkill /F /T /PID $Fake.Id`) and click Next again: the License
      page comes within about 3 seconds (the running check runs while the
      window waits; report a longer wait, and Windows marks the window Not
      Responding after 5 seconds).
- [ ] License: "License" / "Free for personal, noncommercial use." with
      the ridgeline and seal at the right; the license as formatted text
      (bold headings, no `#`, `**` or angle brackets, the URLs plain text,
      not links); I Agree.
- [ ] Installing: "Installing Yamabiko" / "This takes a minute or two.";
      the status line reads "Copying files…" until the end (Show details
      lists the files).
- [ ] Finish: the landscape at dusk, "Yamabiko is installed.", its line,
      Start Yamabiko ticked; Back and Cancel are off. Finish: setup closes
      and Yamabiko starts with its first-run setup ("Yamabiko setup"). This
      is the real-path check of the Finish page's `Exec`: in Task Manager's
      Details tab (add the Command line column) or with
      `Get-CimInstance Win32_Process -Filter "Name='yamabiko.exe'" | Select ProcessId, ExecutablePath`,
      the process runs from `$Dest`, and `$Root\appdata\logs` is written
      (it got `YAMABIKO_APPDATA`).
- [ ] With that Yamabiko running, start the installer again
      (`Start-Process ".\installer\yamabiko-$V-setup.exe" -ArgumentList "/D=$Dest"`):
      the Update page ("Yamabiko <version> is installed. This updates it
      to <version>."). Update: the Still running box; Cancel keeps the
      page; Cancel closes the installer. Close Yamabiko's setup with its
      Cancel button (Yamabiko quits).
- [ ] Update from the previous release:
      `(Start-Process $PrevSetup -ArgumentList @("/S", "/D=$Old") -Wait -PassThru).ExitCode`
      -> 0, then `Start-Process ".\installer\yamabiko-$V-setup.exe" -ArgumentList "/D=$Old"`:
      "Update Yamabiko", "Yamabiko <previous> is installed. This updates
      it to <version>." (versions with spaces for dashes), "Your settings
      and downloaded models stay."; the button reads Update and leads
      straight to the progress page (no license). Untick Start Yamabiko,
      Finish: Yamabiko does not start. Then
      `(Start-Process "$Old\Uninstall.exe" -ArgumentList @("/S", "_?=$Old") -Wait -PassThru).ExitCode`
      -> 0.
- [ ] Uninstall with models kept: give `$Dest` a 5 GB model file (a sparse
      file, no disk space used), start the stand-in again and the
      uninstaller:

      ```powershell
      $Big = "$Dest\hf-cache\release-test\model.bin"
      New-Item -ItemType Directory -Force (Split-Path $Big) | Out-Null
      New-Item -ItemType File $Big | Out-Null
      fsutil sparse setflag $Big | Out-Null
      $f = [IO.File]::OpenWrite($Big); $f.SetLength(5GB); $f.Close()
      $Fake = Start-Process "$Root\fake\yamabiko.exe" -ArgumentList '/c', 'ping -n 600 127.0.0.1 >nul' -WindowStyle Hidden -PassThru
      Start-Process "$Dest\Uninstall.exe"
      ```

      Remove: caption "Uninstall Yamabiko", "Remove Yamabiko?", "This
      removes the app and its Start menu shortcuts.", "Your downloaded
      models and settings stay on your PC."; Uninstall and Cancel.
      Uninstall: the Still running box; Cancel keeps Remove. Stop the
      stand-in (`taskkill /F /T /PID $Fake.Id`), Uninstall: "Removing
      Yamabiko" / "This takes a moment.", then Removed: "Yamabiko has been
      removed.", "Your models and settings (5.0 GB) are still in:", the
      folder, "Delete the folder to free up the space, or keep it for a
      quicker reinstall.", and Open the folder (vermilion, underlined, a
      hand cursor) opens `$Dest` in File Explorer (close it). Close. The
      size checks the 64-bit measure (5.0 GB, not a figure without the 5 GB
      file) and the folder window checks the link's `ExecShell`.
- [ ] Uninstall with nothing kept: delete `$Dest`, install silently
      (`(Start-Process ".\installer\yamabiko-$V-setup.exe" -ArgumentList @("/S", "/D=$Dest") -Wait -PassThru).ExitCode`
      -> 0), start `$Dest\Uninstall.exe`, Uninstall: Removed shows only
      "Yamabiko has been removed." and `$Dest` is gone.
- [ ] Cancel: start the installer with `/D=$Root\cancel`, Cancel on
      Welcome: it closes and `$Root\cancel` does not exist.
- [ ] Sharp at the display's scale: the pictures, the seal and the header
      art look as crisp as the text at your scale (the installer picks the
      100, 150 or 200 % art for the system DPI); compare with the approved
      screenshots.
- [ ] Put the backups back and check them:

  ```powershell
  & .\installer\snapshot-registry-key.ps1 -Action restore -KeyPath $Key -BackupFile "$Root\uninstall-key.reg"
  if (Test-Path $Menu) { Remove-Item -Recurse -Force $Menu }
  if (Test-Path "$Root\startmenu-backup") { Copy-Item -Recurse "$Root\startmenu-backup" $Menu }
  Remove-Item Env:YAMABIKO_APPDATA
  $After = @("InstallLocation: $((Get-ItemProperty $Key -ErrorAction SilentlyContinue).InstallLocation)") +
      @(Get-ChildItem $Menu -Filter *.lnk -ErrorAction SilentlyContinue | ForEach-Object { $l = (New-Object -ComObject WScript.Shell).CreateShortcut($_.FullName); "$($_.Name): $($l.TargetPath) | $($l.Arguments) | $($l.WorkingDirectory)" })
  Compare-Object (Get-Content "$Root\before.txt") $After
  ```

  `Compare-Object` prints nothing: the key's install folder is as it was
  in step 1, and each shortcut opens what it opened then. Then delete
  everything in `$Root` except `startmenu-backup` and `before.txt`
  (`$Root\fake` only after `tasklist` shows no `yamabiko.exe`). Keep those
  two until the end of the release session: the last step of Tag +
  publish checks the shortcuts against them.

## Tag + publish

- [ ] `git tag -a v<version> -m "Yamabiko v<version>"`, then
      `git push origin v<version>` and
      `git push origin "v<version>^{commit}:refs/heads/main"` (fast-forward
      only; `origin` is the private development repository,
      `MeowCatboyMeow/yamabiko`).
- [ ] Download the exact python-soxr source the bundle ships (its LGPL-2.1
      source offer; the version must equal the `soxr==` pin in
      `constraints-release.txt`) and check its SHA-256:

  ```powershell
  Invoke-WebRequest https://files.pythonhosted.org/packages/ed/11/27cebce4a108f77afea7c80545115536b45e3f11ebfb914f638fdd9ba847/soxr-1.1.0.tar.gz -OutFile build\soxr-1.1.0.tar.gz
  (Get-FileHash build\soxr-1.1.0.tar.gz).Hash  # 9F228AE21C78FA9359CA98D8A5E8E91F30639E438E574133DACE62C5B5309E44
  ```

- [ ] Public snapshot (PowerShell): one commit on the public repo whose tree
      equals the tag's tree.

  ```powershell
  $V="<version>"; $Sha = git rev-parse "v$V^{commit}"; $Pub = Join-Path $env:TEMP "yamabiko-public-$V"
  git clone https://github.com/Yamabiko-ASR/yamabiko.git $Pub; git -C $Pub config core.autocrlf false
  git -C $Pub rm -rq --ignore-unmatch .
  git -c core.autocrlf=false archive --format=tar $Sha | tar -x -C $Pub
  git -C $Pub add -A
  if ((git -C $Pub write-tree) -ne (git rev-parse "$Sha^{tree}")) { throw "snapshot tree differs from v$V" }
  git -C $Pub commit -s -m "Yamabiko v$V" -m "Source-Commit: $Sha"
  git -C $Pub tag -a "v$V" -m "Yamabiko v$V"
  git -C $Pub push origin HEAD:main "v$V"
  ```

- [ ] `gh release create v<version> installer/yamabiko-<version>-setup.exe build/soxr-1.1.0.tar.gz -R Yamabiko-ASR/yamabiko --title "Yamabiko <version as the app shows it, with a space for the dash, for example 0.2.0 beta>" --notes-file docs/release-notes/v<version>.md`
      (uploads `soxr-1.1.0.tar.gz` as a release asset next to the installer).
      The title is the one the owner approved with the notes.
- [ ] Last, check the Start Menu shortcuts against the copies kept by the
      drill (rows 8 and 9), the wizard E2E and Installer pages, then delete
      the copies. Windows can rewrite a restored shortcut's bytes seconds
      after the restore (it refreshes the link tracking data stored in the
      `.lnk`) while the shortcut still opens the same program. So the check
      reads what each shortcut opens (target, arguments, working folder),
      never its bytes or a hash, and the copies stay until now in case a
      shortcut really changed:

  ```powershell
  . {
      $ErrorActionPreference = "Stop"   # a failed listing stops the check
      $Ran = 2   # the wizard E2E and Installer pages; 3 when the drill ran row 8 or 9
      $Key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko"; $Menu = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Yamabiko"
      $Now = @("InstallLocation: $((Get-ItemProperty $Key -ErrorAction SilentlyContinue).InstallLocation)") +
          @(Get-ChildItem $Menu -Filter *.lnk -ErrorAction SilentlyContinue | ForEach-Object { $l = (New-Object -ComObject WScript.Shell).CreateShortcut($_.FullName); "$($_.Name): $($l.TargetPath) | $($l.Arguments) | $($l.WorkingDirectory)" })
      $Kept = @(Get-ChildItem C:\tmp\yamabiko-*\before.txt)
      if ($Kept.Count -ne $Ran) { throw "found $($Kept.Count) before.txt, not $($Ran): $($Kept.FullName -join ', '). Find the missing (or extra) folder before you delete anything." }
      foreach ($File in $Kept) {
          $Before = @(Get-Content $File.FullName)
          if ($Before[0] -notlike "InstallLocation:*") { throw "$($File.FullName) holds no listing. Find out what the shortcuts opened before you delete anything." }
          $File.FullName
          Compare-Object $Before $Now | ForEach-Object { "  $($_.SideIndicator) $($_.InputObject)" }
      }
  }
  ```

  Set `$Ran` first. The check stops with a message when it finds another
  number of `before.txt` files or one without a listing. Otherwise each
  `before.txt` it names has no `=>` or `<=` line under it; then delete
  those `C:\tmp\yamabiko-*` folders. If a shortcut line shows under one,
  put that folder's copy back and run the check again (`<folder>` is the
  folder whose `before.txt` showed the line):

  ```powershell
  $Copy = "<folder>\startmenu-backup"; if (Test-Path $Copy) { if (Test-Path $Menu) { Remove-Item -Recurse -Force $Menu }; Copy-Item -Recurse $Copy $Menu } else { throw "no copy at $Copy; nothing changed" }
  ```

  A folder has no copy when there was no Start Menu folder at its start
  (its `before.txt` lists no shortcut; then remove `$Menu` instead) or
  when its copy failed (use another folder's copy). An `InstallLocation`
  line means the uninstall key changed: find out why before you delete
  anything.

## Known-broken-test workarounds (if any)

None currently. If a test regresses to an order-dependent hang or
platform-specific flake, record it here with the invocation that works
around it and a short note on the root cause.

## The testing charter

`docs/TESTING.md` is the binding charter for every test file in
this repo. Read it before editing anything under `tests/`. It states
the Iron Law (a test has value if and only if it can fail when the
code is broken in the way its name claims to verify), a conditional
nine-row pre-merge checklist (applicable rows block), and a catalogue of
named test anti-patterns, each tied to a bug that shipped because a test
was structured so the failure mode was unreachable.

The catalogue grows in place: the bar for adding a new entry is "a
real bug that a test should have caught but didn't."

## Why the real-path suite exists

The frozen bundle is the product, not the dev venv. A suite that is
mostly mocks passes happily while the real dependency drifts: an HF
repo rename, an `hf_hub_download` signature change, an audio-stack
regression, a PyInstaller spec omission, a missing CUDA DLL. Several
of those have bitten us. Seams with zero real coverage, mocked-only
coverage of bug classes that had already shipped, and the absence of
any frozen-bundle smoke test let regressions through unchecked.

The `@pytest.mark.real` suite covers:

- FasterWhisperASR construction + warmup + transcribe
- PyAudio callback-stream lifecycle (open + deliver frames + close)
- onnxruntime.InferenceSession for Silero VAD
- uvicorn.Server.serve() + shutdown (not TestClient/ASGI-in-process)
- PyInstaller frozen-bundle `--version` + `_internal/` layout
- HuggingFace download real path (Silero ~2 MB)
- GitHub releases API (update-check plumbing)

Release is NOT safe to publish if any of these fail. Contributor runs may
skip when a prerequisite is absent; the release runner treats that as an
unmet gate and blocks the release.
