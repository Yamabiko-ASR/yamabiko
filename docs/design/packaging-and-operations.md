# Packaging & Operations

How Yamabiko gets from a git checkout to something a non-developer can
install, run, and recover from when it breaks.

## Installer

`yamabiko-<version>-setup.exe` is an NSIS installer that drops a
PyInstaller onedir bundle into `%LOCALAPPDATA%\Yamabiko\`. Per-user
install, so no admin rights and no UAC prompt. It adds a Start Menu
shortcut, writes an uninstaller, and registers under
`HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Yamabiko` so
the app shows up in Add/Remove Programs.

### Installer pages

The installer keeps NSIS Modern UI 2's own window, buttons and progress
bar, with Segoe UI text and the app's light colors. Its first page,
Welcome, shows a landscape picture; when the Apps entry already lists a
version it becomes Update ("Yamabiko 0.1.0 beta is installed. This updates
it to …") and the license page is skipped. The license page shows
`installer/LICENSE.rtf`, a formatted copy of `LICENSE`. When Next is
clicked on Welcome, the installer checks that Yamabiko and its
`llama-server.exe` aren't running and asks the user to quit them if they
are; silent installs (`/S`) check before anything else and exit with code
2 instead. The last page, at dusk, has Start Yamabiko ticked: it starts
`yamabiko.exe` from the install folder, as the Start Menu shortcut does
(silent installs never start it). The uninstaller asks on a Remove page and
runs the same running check when Uninstall is clicked (silent uninstalls:
before anything, exit code 2). Its last page shows what it kept (models,
settings, logs), their measured size and the folder, with a link that
opens it; when nothing was kept, that page shows only its title.

The pictures are BMPs drawn from the SVGs in `installer/art/`
(`build_art.py`) at 100, 150 and 200 %; the installer is DPI aware and
picks the scale for the system DPI. They and the license RTF are
committed, so the build needs no renderer.

### What's bundled

- Python runtime (frozen by PyInstaller)
- The app itself, plus its Python dependencies
- `llama-server.exe` and its CUDA DLLs, from a pinned llama.cpp commit
  (added to the bundle by `installer/build.ps1` after PyInstaller runs)
- The CUDA runtime DLLs CTranslate2 needs (cuBLAS, cuDNN, NVRTC), from
  the pinned `nvidia-*-cu12` wheels
- `unidic-lite`, `sudachidict_core`, and `cutlet` dictionary/exception
  data (pinned)
- `default_config.yaml`, the seed glossaries, the browser-overlay static
  assets, the app icon, LICENSE, NOTICE, README, CHANGELOG

The installer is around 1.5 GB.

### What's **not** bundled

Model weights are never shipped inside the installer. The first-run
wizard downloads them into the HuggingFace cache that
`yamabiko.hf_env` pins to `%LOCALAPPDATA%\Yamabiko\hf-cache\`, and the
runtime reads them straight back from that cache:

- **Translator GGUF**: the Standard tier (`default` in config) fetches `Qwen3.5-9B` at
  `UD-Q3_K_XL` (around 5.1 GB); 8 GB cards select the `small` tier,
  which fetches the same model's `UD-IQ3_XXS` quant (around 4 GB)
  instead.
- **faster-whisper CT2 model**: Kodama-1, `Yamabiko-ASR/kodama-1` (around
  1.5 GB), Yamabiko's Japanese fine-tune of whisper-ja-760M. Its pruned
  vocabulary needs CTranslate2 4.7.1 or newer.
- **Silero VAD ONNX weights**: small, fetched in the same wizard step.

Pinning every repo, file, and revision in one place (`yamabiko/models.py`)
keeps the wizard's download list, the runtime resolvers, and the
preflight release gate from drifting apart.

There is no PyTorch in the bundle. The VAD loads its ONNX model through
`onnxruntime` directly, so `torch`, `torchvision`, `torchaudio`, and
`transformers` are excluded outright. Dropping torch is what brought the
bundle down to a size NSIS can build; the CUDA DLLs CTranslate2 still
needs come from the `nvidia-*-cu12` wheels rather than from `torch/lib/`.

### Start Menu target

The Start Menu shortcut, and the installer's Start Yamabiko, point at
`yamabiko.exe`, which is the watchdog stub, not the main pipeline. Crashes
get caught and restarted without the user noticing (see § Watchdog).

### Known gaps

- **Code signing**: unsigned for the initial beta. Windows SmartScreen
  flags the installer as an unrecognized app; the README troubleshooting
  explains the "More info, then Run anyway" path. A signing certificate
  is planned for a later release.
- **Distribution channel**: the installer ships as a GitHub release asset;
  generated installers are not committed to the repository.

## First-run setup

A Qt window (`wizard.py`, "Yamabiko setup", 880 × 600) runs on first
launch, in the pipeline process before the pipeline starts, gated on
`%LOCALAPPDATA%\Yamabiko\state.json` (`wizard_state.needs_wizard()`, which
also starts it again when a pinned model is missing from the cache). Its
steps are listed in a rail on the left: finished steps carry a check and
can be clicked to go back; later steps open only through the primary
button, which runs the current step's check first. It uses the app look
(Light or Dark from `overlay.theme`, read once when it opens) and the
bundled fonts.

1. **Welcome**: icon, version, what Yamabiko does, the download size. When
   setup opens on an install that finished it before and the speech model
   is not cached (an update moved its pin), one more line says so with its
   download size.
2. **Sound**: the WASAPI loopback outputs, the Windows default output
   marked "System default" and preselected on a first run. "Test for 5
   seconds" records with a live level meter, then says whether it picked
   up sound.
3. **Content**: the genre (Variety, Anime, Drama, News) with a sample
   subtitle that follows the choice.
4. **Quality**: the translator tier (Standard for cards with 10 GB or
   more, Smaller for 8 GB cards), with the detected graphics card and the
   tier recommended for it. A card under 8 GB gets no recommendation and,
   as with no card, the line naming the 8 GB requirement. Start download
   saves the tier.
5. **Download**: pulls the faster-whisper CT2 files, the translator GGUF
   for the chosen tier and the Silero VAD model, in that order, from
   Hugging Face into the cache, each model file checked against its
   pinned SHA-256. The total and the row sizes come from the size pins in
   `models.py`, time left from the progress byte counts. Leaving the step
   does not stop the download (a line under Download in the rail shows
   its progress); only canceling setup does, and coming back after a tier
   change starts the queue for the new tier. Files already in the cache
   no-op. Before starting, it checks the free space on the cache's drive;
   when it is short, or a download runs out of space, it names the drive
   and the amount to free, offers to delete models from older Yamabiko
   versions when Yamabiko's own cache holds any, and Retry. Other
   failures offer Retry; Copy details puts the details, with the user
   name left out, on the clipboard.
6. **Translator**: waits for the translator's `/health`, then sends one
   `こんにちは` request. The watchdog starts `llama-server` once the GGUF is
   in the cache; setup only talks HTTP to it. If it doesn't answer in
   time, or the test translation fails, Retry or Skip (Yamabiko checks
   again when it starts); an empty or unreadable answer gets a plain line
   (the log keeps the error), and another program answering on the port
   is named.
7. **Done**: where Yamabiko sits in the tray and the shortcuts. Start
   Yamabiko writes the choices (`user_overrides.yaml`, then
   `state.json`) and the pipeline starts. Cancel, Esc or the close button
   end setup with exit code 42, and the watchdog stops instead of
   restarting.

`state.json` records the wizard's own bookkeeping (`first_run_complete`,
`wizard_version`, `wizard_completed_at`, `audio_device_name`, `genre`,
and `llama_smoke_ok`). Anything user-mutable
lives in the sparse `user_overrides.yaml`. The wizard writes keys there
only when the user's choice diverges from the shipped default (for
example `translation.quant_tier: small` for an 8 GB card).

**One-time legacy-config migration.** The wizard isn't involved in this,
but installs that predate the layered-config split carry a
fully-populated `%LOCALAPPDATA%\Yamabiko\config.yaml` from the old
copy-on-first-run model. On the next launch the loader diffs that file
against the shipped defaults, writes the minimal diff to
`user_overrides.yaml`, and renames the original to
`config.yaml.migrated.bak` so nothing is lost if the migration needs
forensic review.

To run setup again, delete `%LOCALAPPDATA%\Yamabiko\state.json`; the next
launch starts it (the README says so for users).

## Watchdog

`yamabiko.exe` is a small PyInstaller-frozen stub that launches the
watchdog, not the main pipeline. The watchdog:

- Spawns `yamabiko.main` as a subprocess and supervises it.
- Spawns `llama-server` and supervises it the same way; the pipeline
  process in turn owns the browser/desktop overlay child through an
  inner Job Object, so a dead parent never leaves orphaned children.
  Both jobs allow explicit breakaway, which two notification clicks use:
  the update notification opens the release page, and "Diagnostics
  saved" opens Explorer with the zip selected, each through a process
  started outside the jobs, so a pipeline restart or Quit never closes
  the user's browser or that Explorer window.
- Each process logs to its own role file (`main-watchdog.log`,
  `main-pipeline.log`, `main-overlay.log`) rather than sharing one file,
  which Windows file locking makes unsafe for concurrent writers.
- On a crash exit: restarts with exponential backoff (1 s, then 5 s,
  then 15 s), up to 3 restarts in any rolling 10-minute window.
- **Anti-flap**: a run that lived only a few seconds before exiting is
  almost certainly a startup config error that more restarts won't fix,
  so it escalates to the startup error window immediately.
- On 3 failed restarts within the window: gives up and shows the
  startup error window, whose Open log folder opens the logs.
- **Typed exit codes** steer all of this. The pipeline exits with a code
  that says why it stopped (a user quit, a capture death, a stage
  escalation, an overlay give-up, or an overlay-port bind failure), and
  the watchdog classifies each into restart, stop, or give-up rather
  than reading every non-zero exit as a crash. A user quit stops
  cleanly; an audio device that keeps dropping gets a device-specific
  give-up dialog; an overlay that crash-loops past its own cap triggers
  a pipeline restart so the overlay gets a fresh budget.
- **Problem carry.** When the pipeline exits for a restart (44, 45 or
  47), it first writes `%LOCALAPPDATA%\Yamabiko\problem-carry.json` with
  the problems active then and the shortcuts it announced as taken (no
  file when there are none). From then on it clears no problem, so no
  recovery line shows before the restart. When it raised a problem itself
  with a client connected, it then waits 4 s so the notification and the
  overlay line show before the overlay closes with it (Windows drops that
  notification from the notification center once the overlay's tray icon
  is gone). The watchdog takes the file before the respawn and passes
  the kinds to the next pipeline in `YAMABIKO_CARRIED_PROBLEMS`, which
  reports them as recovered to clients that connect within 30 s, and
  the shortcuts in `YAMABIKO_CARRIED_HOTKEYS`, so the next pipeline does
  not announce a taken shortcut again while its combo is unchanged. It
  also restarts the bundled llama-server when a translation escalation
  caused the exit, unless llama-server is already restarting on its own
  (the pipeline's last health probe could not connect, or found it
  loading its model). A requested restart is exempt from the fast-crash
  give-up but counts toward the restart budget. A fresh launch drops a
  leftover file.
- **Overlay port conflict**: if the overlay's WebSocket server cannot
  bind its port, the pipeline reports a startup-failure exit and the
  watchdog shows a dialog naming the occupied port and pointing at
  `overlay.websocket_port`, without restarting. A bind conflict will not
  self-heal, so restart churn would only delay that message.
- Forwards a clean shutdown from tray "Quit" to the children and exits
  after they do.

The watchdog also runs a health probe that polls the pipeline's `/livez`
endpoint. The probe only arms its fail budget after the first healthy
response, so a cold-booting pipeline never false-trips; once armed, a
sustained run of failures means the pipeline is wedged rather than busy,
and the probe force-restarts it through the same restart loop.

The watchdog never reads stdin (Windows subprocess stdin piping is
quirky; keeping the child silent on stdin avoids that bug class).

## Service reconnects

Two subprocesses that the pipeline talks to can fail independently of the
main process.

### llama-server

The watchdog spawns `llama-server` and supervises it with the same
restart-on-failure loop it uses for the pipeline: exponential backoff
(1 s, then 5 s, then 15 s) and a cap of 3 restarts in a rolling
10-minute window. Past the cap it stops retrying and shows its give-up
message in the startup error window, whose Open log folder opens the
logs.

On the pipeline side, the translator classifies each request's errors
and logs them; a translator that stops answering raises the
"Translations stopped" notice (see Problem notices):

- **Network error** (connection refused, timeout) or **HTTP 5xx**:
  retry once after 1 s. If it still fails, log it and drop the segment.
  A dead `llama-server` is the watchdog's to respawn, not the
  translator's.
- **HTTP 4xx**: log it and drop the segment. A 4xx is usually a config
  problem a respawn would not fix.

The server starts in llama.cpp router mode (`--models-max 1
--sleep-idle-seconds -1`), which pins the model in VRAM indefinitely so
reloads are a non-issue. Because the pipeline is spawned concurrently
with the server, the translator polls `/health` until the multi-GB GGUF
finishes loading before declaring a real failure, so a slow cold load
no longer reads as a crash.

### Browser overlay WebSocket

Client-side exponential reconnect: 1 s, 2 s, 4 s, 8 s, capped at 30 s. No
server-side state recovery, so subtitles past the disconnect are dropped
as stale.

## Error surface

Two layers.

### Startup error window (critical failures)

A failure that stops startup becomes a `StartupError` (title, a
one-sentence message, what to try, details): the pipeline classifies its
exception by type and message (`classify_startup_exception` in
`ui/error_dialog.py`), and the watchdog and the setup check build theirs
by hand ("Yamabiko is already running", "Invalid configuration", its
give-ups, "Setup required"). `show_startup_error` logs the title and
message (`startup error dialog: ...`) and shows the startup error window
(`ui/error_window.py`, the redesign's "Report card"):

- 720 px wide with its height following the content, centered on the
  primary screen's work area (it gets its system menu only after it
  shows: at 150% Qt widened a window by 3 px when that frame change came
  before the show applied its size), in the app look:
  Light or Dark from `overlay.theme`, read straight from
  `user_overrides.yaml` because the full config load can be what failed.
  A background process shows it, so it takes the foreground itself
  (`window_chrome.take_foreground`).
- Left: the title, the message and "What to try", one item per sentence
  (the message and the items with the user name left out, as in the
  report: a message Yamabiko builds itself can quote a path).
  For a missing audio device, Open Settings opens Settings at Audio &
  model in the same process, as its own window the way the tray opens it
  (unowned, seeded with the saved style); the new device applies the
  next time Yamabiko starts. Nothing is modal: the window is shown with
  `show()` and runs in a local event loop (shown by `exec()` instead, it
  would be application-modal and block input to Settings), and the
  process waits until both windows are closed.
- Right, the Report panel (`ui/error_report.py`): the Yamabiko version,
  the Windows version and build (from the registry), the GPU name, video
  memory and driver (one nvidia-smi query), the error line (shown with
  the exception's type name but not its module path, on three lines at
  most) and the last 40 lines of this process's log (condensed, scrolled
  to the newest), with Open log folder. Copy report puts the report on
  the clipboard: those facts (the error line whole), the Python version,
  the raw log lines and the details, with the `%LOCALAPPDATA%` and
  `%USERPROFILE%` folders and the Windows user name replaced in every
  line (long and 8.3 short paths, every slash style), then flushes it
  (`OleFlushClipboard`) so it outlives the pipeline's hard exit. A
  clipboard listener (the text input host, Explorer, the clipboard
  service) can hold the clipboard open right after the copy, so the
  flush is retried for up to half a second; if it still fails, the
  failure is logged and the button keeps its normal state. Text copied
  out of the log box is flushed the same way.
- Close, the caption's close button or Esc end it. In the pipeline the
  process then exits with code 43, so the watchdog stops instead of
  restarting. While the window (or the Settings window it opened) is up,
  the pipeline holds the startup-dialog marker
  (`lifecycle/startup_dialog.held()`), so the watchdog's liveness probe
  does not take the waiting process for a hung one.
- If Qt can't load, or the window itself fails (logged with its
  traceback), a plain Windows message box shows the same title, message
  and what to try, where the logs are, and the details, with the user name
  left out of all but the title.

| Case | Title |
| --- | --- |
| No CUDA / GPU unavailable | No NVIDIA graphics card found |
| No WASAPI loopback output | No speakers or headphones found |
| The audio device Yamabiko is set to is gone | `<device>` isn't connected |
| llama-server did not become ready | The translator didn't start |
| Model files missing: a `FileNotFoundError` anywhere in the exception chain (a speech model set by hand that isn't downloaded, started offline) or a missing-file message; a DLL that fails to load stays generic, except a CUDA library (its error names cuda, cublas, cudnn or nvrtc), which the GPU case, checked first, shows as No NVIDIA graphics card found | Some model files are missing |
| A cached model file fails its integrity check | A model file is damaged |
| A file shipped with the install is missing or damaged | A Yamabiko file is damaged |
| Out of disk space mid-download | Not enough disk space |
| Anything else | Yamabiko couldn't start |

### Runtime stage errors

Each pipeline stage wraps its loop in a broad except and logs the
traceback; the stage supervisor routes repeated failures through a
log-only notifier. Other stage failures stay in the log; the two a user
can act on are Problem notices (below).

If a stage hits a sustained run of consecutive errors, the supervisor
escalates: the pipeline exits with a stage-escalation code and the
watchdog restarts it, then falls back to the give-up dialog once the
restart cap is hit.

### Problem notices

Two failures a user can act on are reported with a Windows notification
when they start and, while they last, an amber お知らせ line in the Qt
overlay and on the browser/OBS page and the tray icon's amber dot:

- **Translations stopped**: the pipeline's health probe of the translator
  fails, or a translation escalation fires (5 timeouts in a row, or 10
  in 60 s). The probe raises it only once `/v1/models` has answered in
  this run and no translation completed since the previous probe, so an
  endpoint that never serves `/v1/models` (a remote server without the
  route) is left to the escalations. It clears on the first `/health`
  200 (a server without `/health` counts when `/v1/models` answers) or
  the first completed translation. The copy promises a restart only when
  one happens under the watchdog with the bundled loopback llama-server:
  the probe could not connect (its supervisor respawns it), or a
  translation escalation fired (it restarts a hung server and upgrades
  the entry). Otherwise, and for a remote endpoint, it reads "The
  translator isn't responding."
- **Can't hear your audio**: the capture stream dies and its endpoint is
  no longer an active render endpoint (a fresh Core Audio query); a death
  with the endpoint still present, or when Yamabiko can't tell, restarts
  silently as before. The restart carries it, and the next pipeline
  reports it recovered once its capture runs.

On clear the line shows the recovery text ("Translations are back.",
"Listening again.") for 5 s; no notification.

## Config paths

| Path | Contents | Written by |
| --- | --- | --- |
| `<bundle>/yamabiko/default_config.yaml` | shipped defaults; authoritative schema, read-only at runtime | release build |
| `%LOCALAPPDATA%\Yamabiko\user_overrides.yaml` | sparse user overrides (only keys that differ from the shipped default); absent on a fresh install | user via Settings, wizard, or hand edit |
| `%LOCALAPPDATA%\Yamabiko\config.yaml.migrated.bak` | one-shot backup of the pre-split legacy config, written once during migration | config loader |
| `%LOCALAPPDATA%\Yamabiko\state.json` | wizard bookkeeping | wizard |
| `%LOCALAPPDATA%\Yamabiko\logs\main-*.log` | per-process logs (watchdog / pipeline / overlay), rotating | each process |
| `%LOCALAPPDATA%\Yamabiko\logs\latency.log` | per-segment JSON, rotating | pipeline |
| `%LOCALAPPDATA%\Yamabiko\problem-carry.json` | problems and announced-taken shortcuts a restarting pipeline hands to the next one; deleted when the watchdog takes it | pipeline |
| `%LOCALAPPDATA%\Yamabiko\hf-cache\` | first-run model downloads | wizard |

Config is layered: the bundled `default_config.yaml` carries the
authoritative schema, and `user_overrides.yaml` holds only the keys a
user has changed. Both are parsed with PyYAML's `safe_load`. The merged
result drives the runtime; the translator endpoint defaults to
`http://localhost:8090`, the browser overlay WebSocket to port 8765.

Every log file uses size-based rotation at 10 MB across 5 backups, so
each caps at 60 MB. The dev/repro escape hatch `python -m yamabiko.main
--config <path>` layers the given file over the shipped defaults and
skips `user_overrides.yaml`, so the run is isolated from ambient user
state.

Yamabiko does not send telemetry or diagnostics automatically. To share
diagnostics, Save diagnostics (the tray flyout's Troubleshooting menu)
zips the logs plus a keys-only settings snapshot to the Desktop in one
click; the notification it shows opens Explorer with the zip selected.

## Global hotkeys

Three hotkeys ship by default, and each can be rebound or disabled:

- **Pause / resume**: default `Ctrl+Alt+P`. Pause stops new audio intake
  and abandons any incomplete waveform or unfinished Japanese sentence.
  Completed English translation context is retained. A subtitle already
  emitted downstream may still finish and appear while paused; Yamabiko
  does not hide or retract it. Resume starts from a discontinuity rather
  than joining new audio or Japanese text to pre-pause partial state.
- **Show / hide overlay**: default `Ctrl+Alt+O`. Toggles overlay
  visibility without stopping the pipeline, which is useful on streams
  where subtitles would cover a key moment.
- **Pick a series**: default `Ctrl+Alt+S`. Opens the series bar at the top
  of the subtitle monitor for character-name hints. The pipeline's hook
  catches the keys, so the overlay process takes the foreground for the
  bar itself (`window_chrome.take_foreground`).

These are implemented with the MIT-licensed `keyboard` library (no admin
rights needed for non-modifier-blocking hotkeys). The library fires its
callbacks on a background thread, so each handler is bridged onto the
pipeline's asyncio event loop via `run_coroutine_threadsafe` rather than
touching pipeline state directly off-thread.

Settings > Shortcuts rebinds them live: Settings saves the `hotkeys.*`
combos, the tray sends a `set_hotkeys` command, and
`PipelineApp.apply_hotkeys` clears and re-adds the bindings without a
restart. `hotkeys.enabled: false` turns all three off; an empty
`hotkeys.<key>` turns one off (Settings shows it empty, and recording or
Reset there turns it back on).

A `keyboard` hook never fails for a combination another app registered
with `RegisterHotKey`, so before binding, `hotkeys.combo_taken` registers
the combination itself and releases it at once;
`ERROR_HOTKEY_ALREADY_REGISTERED` means it is taken. A taken combination
is skipped and reported through the tray as "Shortcut unavailable"
(clicking the notification opens Settings at Shortcuts), while the other
bindings remain available. Each taken combination is announced once: a
later re-bind (another shortcut changed, or the tray's reconnect resend
of shortcuts saved while offline) announces it again only when its own
combination changed or, after a Settings Apply, it was recorded again,
which also retries it. A pipeline the watchdog restarted does not
announce again the ones the previous pipeline announced
(`YAMABIKO_CARRIED_HOTKEYS`, see "Problem carry" under Watchdog).
Settings runs the same probe to mark a taken combination "Used by
another app", and won't save two actions on one combination.
