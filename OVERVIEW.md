# Yamabiko: Technical Overview

A live Japanese speech-to-subtitle pipeline for Windows. Captures
system audio, transcribes it, translates it, and renders the result with
furigana on a frameless always-on-top overlay. Passive across any audio
source (livestream, VOD, game, browser) without per-content setup.

- **Platform**: Windows 11 with an NVIDIA GPU at Compute Capability 7.5 or newer
- **ASR**: `faster-whisper` running Kodama-1 (`Yamabiko-ASR/kodama-1`),
  Yamabiko's Japanese fine-tune of whisper-ja-760M (float16 on GPU)
- **Translation**: `Qwen3.5-9B` GGUF on local llama.cpp router mode;
  context-aware prompting with genre presets. The Smaller (8 GB) tier
  runs the same model at a lighter quant (UD-IQ3_XXS).

Subsystem-level design rationale lives in [docs/design/](docs/design/).
This document is a structural tour: the pipeline, the modules, and the
Windows-specific gotchas.

---

## Pipeline

```text
 [WASAPI loopback]
         │
         ▼
 [Silero VAD] ─► frames gated into the ASR queue
         │
         ▼
 [faster-whisper] ─► transcript chunks
         │
         ▼
 [non-JA guard] ─► drop English passthrough
         │
         ▼
 [fast-bunkai] ─► one sentence per broadcast
         │
         ▼
 [cutlet + fugashi/sudachi] ─► furigana + romaji segments
         │
         ├─► [Qwen3.5-9B on llama.cpp] ─► EN translation
         │
         ▼
 [ConnectionManager broadcast]
         │
         ├─► Qt overlay (separate supervised process, custom QPainter ruby)
         └─► Browser overlay (FastAPI + WebSocket, HTML <ruby>)
```

Each stage is an async coroutine communicating via bounded `asyncio`
queues. The WASAPI callback thread bridges in via
`loop.call_soon_threadsafe`; CUDA work runs on a dedicated
`ThreadPoolExecutor(max_workers=1)` so CTranslate2's non-reentrant CUDA
state stays linear. No stage can block another.

The pipeline always spawns the Qt overlay as a separate process
(`python -m yamabiko.overlay_app`) and supervises it through an inner
Job Object, so a dead pipeline never leaves an orphaned overlay. The
desktop and browser overlays both consume the same WebSocket stream
from `display.broadcast.ConnectionManager`.

One optional side input feeds the translator: when the user picks a
series (the `Ctrl+Alt+S` series bar, or the tray), a per-series gazetteer
built from AniList character names is loaded and its JA-to-EN name
mappings ride along as soft prompt hints. The Japanese source text is
never rewritten. Gazetteers are cached on disk per series, with the
picked series' cover beside its cache, and the active pick is restored
at boot via `active_series.json`.

---

## Module layout

```text
yamabiko/
├── main.py                     # Pipeline entrypoint; wires stages + queues
├── watchdog.py                 # Top-level supervisor: pipeline + llama-server (the pipeline owns the overlay)
├── default_config.yaml         # Shipped defaults; read in place, never copied to user space
├── paths.py                    # %LOCALAPPDATA%\Yamabiko + bundled-data path helpers
├── log_config.py               # Per-process rotating logs (main-{watchdog,pipeline,overlay}.log) + latency.log
├── models.py                   # Single source for HF repo / file / revision pins
├── fsutil.py                   # Atomic text-file write (temp + os.replace + fsync)
├── diagnostics.py              # Tray diagnostics bundle: logs + settings names + versions to zip
├── hf_env.py                   # HF_HOME / HF_XET_HIGH_PERFORMANCE env-var pin (side-effect import)
├── hotkeys.py                  # Global shortcuts: combo parsing, `keyboard` binding, RegisterHotKey "used by another app" probe
├── supervisor.py               # Stage-level retry loop + escalation hook
├── state.py                    # AppState: genre, paused, style, active series, problems (shared across stages)
├── problems.py                 # The two problem notices (translator, audio device): copy, problems frame, restart carry
├── config.py                   # load_overlay_config + OverlayConfig dataclass + load_full_config
├── config_writer.py            # Atomic YAML writer preserving unrelated keys
├── gpu.py                      # nvidia-smi VRAM, name and the error report's GPU summary (no torch / pynvml dep)
├── jp_text.py                  # is_japanese() non-JA guard helper
├── update_check.py             # GitHub Releases startup version-check probe
├── wizard.py                   # First-run setup window (step rail, pages, footer) and run_wizard()
├── wizard_state.py             # state.json round-trip + first_run_complete flag
├── wizard_pages/               # Setup steps: welcome, audio (Sound), content, quant_tier (Quality), downloads, smoke (Translator), done; rail and shared parts
│
├── capture/
│   ├── audio.py                # WASAPI loopback capture, device selection, silent-stall recovery
│   ├── default_endpoint.py     # Core Audio queries: the OS default render endpoint, the active endpoints' names
│   └── _audio_constants.py     # Dependency-free audio constants (importable without PyAudio)
├── vad/
│   ├── chunker.py              # ChunkerConfig dataclass
│   ├── silero_chunker.py       # FrameGate state machine + SileroChunker
│   ├── sound_tagger.py         # bundled EfficientSED sound tagger + streaming VAD assist
│   └── data/                   # the tagger's ONNX graph and its MIT license
├── asr/
│   ├── asr_client.py           # faster-whisper backend wrapper
│   └── incremental.py          # incremental emission: commit rule, step cadence, final strip, step lane
├── sentencing/
│   └── splitter.py             # fast-bunkai wrapper with partial-sentence buffer
├── romanization/
│   ├── romanizer.py            # cutlet + Hepburn macron post-processing
│   ├── extractor_fugashi.py    # UniDic 固有名詞 filter + denylist
│   ├── extractor_sudachi.py    # SudachiPy split-mode C for katakana brands
│   ├── glossary.py             # GlossaryEntry + three-layer GlossaryManager
│   ├── glossary_yaml.py        # Shared YAML loader (user + seed)
│   └── pipeline.py             # RomanizationPipeline, dual-extractor union
├── glossaries/
│   ├── user.yaml               # Hand-curated, never machine-written
│   ├── _seed.yaml              # Time-stable shipped seed
│   ├── _denylist.txt           # Common-noun terms excluded from NER
│   └── {variety,drama,news,anime}.json  # Read-only legacy compatibility data
├── gazetteer/
│   ├── anilist.py              # AniList GraphQL client (public API, no auth): series + character search
│   └── gazetteer.py            # SeriesGazetteer: JA name surfaces to official EN spellings; disk cache
├── translation/
│   ├── translator.py           # llama.cpp OpenAI-endpoint client + warmup probe
│   ├── prompt.py               # System prompt, genre notes, context buffer
│   └── postprocess.py          # Quote/label/stage-direction stripping
├── display/
│   ├── broadcast.py            # ConnectionManager, pure-asyncio WS fan-out
│   ├── browser_server.py       # FastAPI + WebSocket overlay server
│   └── static/                 # index.html, overlay.css, overlay.js; fonts/ (bundled BIZ UDPGothic / BIZ UDPMincho / Klee One + OFL texts)
├── lifecycle/
│   ├── exit_codes.py           # ExitPolicy classifier (STOP_SELF / STOP_ALL / RESTART)
│   ├── jobs.py                 # Windows Job Object helpers (outer + inner nesting)
│   ├── launch_markers.py       # Qt-free launch-chain log markers (launch tests assert on them)
│   ├── pipeline_supervisor.py  # Overlay supervisor (pipeline's inner job; anti-flap respawn)
│   └── shutdown_event.py       # Named Win32 Event primitive (role-scoped graceful-shutdown signal)
├── overlay_app/
│   ├── __main__.py             # `python -m yamabiko.overlay_app` entry
│   ├── ws_client.py            # QWebSocket to Qt signals
│   ├── overlay_window.py       # Frameless transparent window pinned to one monitor; hosts SubtitleStack; anchor placement; DWM chrome suppression
│   ├── telop_look.py           # Qt-free subtitle look: style dict to fonts, px sizes, colors, paddings (TelopLook)
│   ├── telop_text.py           # Measures a line's Japanese/English blocks; renders them in the plate or echo look
│   ├── telop_banner.py         # Vertical banner text (genre, series, 一時停止) and its painter
│   ├── telop_stack.py          # SubtitleStack: plates, banner, newest line, echo and notice lines, layout and motion
│   ├── ruby_layout.py          # Pure cap-aware segment-wrap layout for the ruby row (Qt-free, unit-testable)
│   ├── position_config.py      # Pure anchor-position clamp + one-time legacy-geometry migration (Qt-free)
│   ├── placement.py            # On-screen placement controller: drag a sample subtitle, monitor-jump, Esc/Enter
│   ├── series_palette.py       # The series bar: search, Currently watching, pick status, saved cover
│   ├── series_palette_parts.py # The series bar's painted rows, messages, key line, chip and status
│   ├── problem_view.py         # ProblemTracker: problems frames to started / changed / recovered
│   ├── settings_window.py      # Settings window: title band, preview monitor + Overlay switcher, five tabs of panels, footer
│   ├── window_chrome.py        # Caption painted under the title bar, title band, compact-dialog frame, client-area fit; DWM corners, border and cloak; take_foreground
│   ├── controls.py             # Painted Settings controls: switch, text tabs, panels, color field, chevron fields, progress line
│   ├── preview_monitor.py      # Settings preview: the real SubtitleStack over a night street photo
│   ├── shortcut_field.py       # Click-to-record shortcut field (Shortcuts tab)
│   ├── settings_dialogs.py     # "Couldn't save your settings" dialog and the color picker
│   ├── theme.py                # Chrome palettes (Mist light / Night dark) + QSS builders + system-scheme probe
│   ├── fonts.py                # Registers the bundled BIZ UDPGothic / BIZ UDPMincho / Klee One fonts with Qt
│   ├── keycaps.py              # Painted key caps for shortcuts
│   ├── landscape.py            # Painted ridges + echo arcs (app look), and the error window's 720 x 44 strip
│   ├── seal.py                 # Painted 山彦 hanko seal
│   ├── segmented.py            # Lit segmented control
│   ├── style_defaults.py       # STYLE_DEFAULTS (single source)
│   ├── tray.py                 # Tray controller: flyout, short menu, state icon, tooltip, notifications, problems
│   ├── tray_flyout.py          # The flyout panel (left click) and the cover painter the series bar shares (paint_cover)
│   ├── tray_icons.py           # Painted tray icon states, small-icon size, taskbar color
│   └── tray_placement.py       # Flyout and right-click menu placement at the icon + the Trigger debounce
└── ui/
    ├── error_dialog.py         # StartupError cases + classifier; show_startup_error (error window, Win32 fallback)
    ├── error_report.py         # The startup error report: redaction, Windows and GPU facts, log tail, text (Qt-free)
    ├── error_window.py         # The startup error window ("Report card"): what to try, Report panel, Copy report
    ├── qt_bootstrap.py         # Shared HiDPI QApplication bootstrap + Qt message routing
    └── thread_keeper.py        # Keeps QThreads alive past widget teardown (Qt6 abort guard)

third_party/
└── llama.cpp/                  # Vendored llama-server.exe + DLLs; manifest-committed, binaries gitignored

installer/                      # PyInstaller spec + NSIS Unicode installer (art/, LICENSE.rtf) + build.ps1 orchestrator
```

Tests live under `tests/` mirroring the module layout; integration tests
sit at the repo top-level of `tests/`.

---

## Configuration

Configuration uses a two-file layered model: `default_config.yaml`
ships in the app bundle and is the authoritative schema;
`%LOCALAPPDATA%\Yamabiko\user_overrides.yaml` is sparse: it contains
only keys the user has explicitly overridden (via Settings, the
first-run wizard, or a hand edit). At load time the two are
deep-merged. This lets shipped-default improvements propagate
automatically to existing installs while user intent persists across
upgrades. Installs that predate this split are migrated once on first
launch: the old `config.yaml` is diffed against the shipped defaults,
only the diff lands in `user_overrides.yaml`, and the original is
kept as `config.yaml.migrated.bak`. In dev mode, `python -m yamabiko.main
--config <path>` validates and layers the explicit file over shipped defaults
while skipping `user_overrides.yaml`, isolating the run from ambient user
state.

Representative excerpt:

```yaml
audio:
  device: default                # device name substring, or "default"

vad:
  threshold: 0.3                 # produced-media default; raise toward 0.5 for clean speech
  min_silence_ms: 400
  speech_pad_ms: 200
  min_chunk_ms: 500
  max_chunk_ms: 8000
  glue_max_silence_ms: 1500      # no `enabled` flag; the VAD is load-bearing
  attach_max_ms: 6000            # between-chunk audio is attached to the next chunk...
  drop_floor_prob: 0.10          # ...and only discarded after sitting below this
  drop_floor_ms: 2000            # probability for this long

asr:
  faster_whisper:
    model: Yamabiko-ASR/kodama-1
    compute_type: float16        # Standard tier; Smaller uses int8_float16
    device: cuda
    beam_size: 5
    initial_prompt: ""           # Kodama-1 runs without a prompt
    no_speech_only_threshold: 0.95   # 1.0 turns it off

translation:
  endpoint: http://localhost:8090/v1/chat/completions
  model: qwen3.5-9b
  quant_tier: default            # default | small (8 GB cards)
  max_tokens: 80
  temperature: 0.1

genre: variety                   # variety | anime | news | drama

gazetteer:
  enabled: true                  # series-aware character names (AniList lookup)

overlay:
  theme: system                  # system | light | dark  (chrome for settings, tray, series bar)
  reduce_motion: false           # subtitles (Qt and browser page) and the series bar switch instantly
  position:                      # subtitle anchor; replaces the legacy free-geometry rect
    screen: primary              # "primary" or a QScreen name
    v_offset_pct: 88.0           # stack bottom edge, % of screen height
    h_offset_pct: 50.0           # Center alignment: stack center, % of screen width
    left_offset_pct: 6.0         # Left alignment: stack left edge, % of screen width
  geometry: null                 # legacy rect; migrated to `position` once on first boot, then pruned
  style:
    jp_font: "BIZ UDPGothic"     # Watch mode faces; Learn uses Klee One and Segoe UI Variable Text
    jp_size: 16                  # points (1 pt = 4/3 px): 21 px
    en_font: "Bahnschrift"
    en_size: 23                  # 31 px
    text_color: "#ffffff"        # English on the dark bar, and the echo text
    bg_color: "#0e1014"          # the dark bar behind the newest English
    bg_opacity: 237              # 93%
    padding: 10                  # scales the strip and bar padding (10 = the design)
    max_width_pct: 70            # subtitle wrap cap, % of the overlay monitor width
    max_lines: 3                 # the newest line plus the echo
    hold_ms: 5000
    fade_in_ms: 160
    fade_out_ms: 420
    mode: watch                  # watch (English priority) | learn (Japanese priority)
    alignment: left              # left | center
    echo: priority               # priority (one language in the echo) | both
    banner: jp                   # jp | jp_en
```

---

## Settings and shortcuts

The tray's flyout and short menu open one Settings window
(`overlay_app/settings_window.py`, 1000 × 728) that paints under its own
title bar (`window_chrome.py`).
From top to bottom: a landscape title band with the 山彦 heading, the
seal and the Theme choice; a band shown on every tab with the preview
monitor (`preview_monitor.py`, the real `SubtitleStack` over a night
street photo) and the four Overlay choices (Mode, Alignment, Echo,
Banner) beside it; five text tabs (Subtitles, Position, Timing, Audio &
model, Shortcuts) over a grid of panels; and a footer with Reset to
defaults, Cancel, Apply and OK. The painted controls live in
`controls.py`; the save-failed dialog and the color picker in
`settings_dialogs.py`; the model download dialog in `model_fetch.py`.
Every change is staged: Apply and OK write `user_overrides.yaml` in one
transaction and then push the change live, while Esc and Cancel discard
it. On a work area shorter than the window, the window fits the work
area: the preview shrinks first, then the current tab's panels scroll,
and the title band and the footer stay whole.

Shortcuts ride the same transaction. Settings saves the `hotkeys.*`
combos and emits `shortcutsChanged`; the tray sends a `set_hotkeys`
command (kept and sent on reconnect while the socket is down); the
pipeline validates it and `PipelineApp.apply_hotkeys` re-binds the
`keyboard` hooks at once, with no restart. A `keyboard` hook never fails
for a combo another app holds, so `hotkeys.combo_taken` probes the combo
with Win32 `RegisterHotKey` and releases it at once. Settings uses the
probe to mark a taken combo "Used by another app" (at open and after each
recording). The pipeline probes before binding, skips a taken combo and
sends a "Shortcut unavailable" notice for the combos that changed or were
recorded again (recording one again also retries it); any other re-bind
only logs a taken combo, so an unchanged one is announced once. At
startup every taken combo is announced, except that a restarted pipeline
skips the ones the previous pipeline announced (carried in
`YAMABIKO_CARRIED_HOTKEYS`); the tray's reconnect resend of shortcuts
saved while offline announces only the combos that changed (Settings
already marked a re-recorded one). Clicking the notification opens
Settings at Shortcuts. Settings blocks Apply and OK while two actions
share a combo. While a field records, `shortcut_field.py` takes the
completing key in a low-level keyboard hook, so a combo another app
holds can still be recorded.

---

## Tray and problem notices

The tray (`overlay_app/tray.py`) lives in the overlay process and is the
controller of the tray UI. A left click opens the flyout
(`tray_flyout.py`), a frameless panel that `tray_placement.py` puts
12 px above the taskbar over the icon (the click that closes it never
reopens it); a right click shows the short menu (Pause, Pick a series…,
Settings…, Quit Yamabiko) with the current shortcuts. The flyout sends
the commands the old menu sent; Mode and Alignment are saved like a
Settings Apply, and the Subtitles tile asks the pipeline for the same
`toggle_overlay` broadcast as `Ctrl+Alt+O`. `tray_icons.py` paints the
icon per state (listening, paused, subtitles hidden, plus an amber dot
for a problem) at the taskbar's small-icon size, and the tooltip carries
the status sentence.

Two problems are reported: the translator stopped answering, and the
captured audio device went away. The pipeline owns the problem set
(`AppState`, copy in `problems.py`) and sends it as a `problems` frame on
every change and to every new client. In the overlay process
`problem_view.ProblemTracker` diffs the frames by kind: a new kind shows
one Windows notification, the icon's dot and an お知らせ line above the
newest subtitle in the Qt overlay (`SubtitleStack.set_notices`); a kind
that leaves the set turns into its recovery line for 5 s. The
browser/OBS page diffs the frames itself and shows the same lines. A
problem that ends in a pipeline restart is written to
`problem-carry.json`; from then on nothing is cleared before the
restart. The watchdog takes the file, restarts the bundled llama-server
for a translation escalation, and hands the kinds to the next pipeline
(`YAMABIKO_CARRIED_PROBLEMS`), which tells clients connecting within
30 s that they recovered. The same file carries the shortcuts the
pipeline announced as taken, so the next pipeline doesn't announce them
again.

---

## VRAM

On a 16 GB GPU with Qwen3.5-9B at UD-Q3_K_XL and Kodama-1
in float16:

| Component | Approx. VRAM |
|---|---|
| Kodama-1 (float16, faster-whisper) | ~3.5 GB |
| Qwen3.5-9B (UD-Q3_K_XL, llama.cpp) | ~5.1 GB |
| Silero VAD (CPUExecutionProvider) | 0 GB |
| KV cache + working memory | ~0.5 GB |
| **Total** | **~9.1 GB** |

The runtime keeps both models resident while the pipeline is running.

---

## llama.cpp router mode

`llama-server` runs as a child process with router-mode flags that give
the "pinned forever" semantics an earlier llama-swap sidecar used to
provide:

```powershell
llama-server `
  --model .\models\Qwen3.5-9B-UD-Q3_K_XL.gguf `
  --alias qwen3.5-9b `
  --port 8090 `
  --ctx-size 2048 `
  -ngl 99 -fa on `
  --models-max 1 `
  --sleep-idle-seconds -1 `
  --no-webui
```

`scripts/start-llama-server.ps1` parameterizes this. **Do not** pass
`--chat-template-kwargs '{"enable_thinking":true}'`. A silently-enabled
reasoning block burns the `max_tokens: 80` cap and returns nothing
visible. The `no-thinking` instruction in the system prompt is a
belt-and-suspenders second layer.

---

## Windows gotchas

**WASAPI loopback device naming.** PyAudioWPatch exposes loopback
endpoints as virtual input devices with `isLoopbackDevice=True` and a
suffix on the name. Prefer `get_default_wasapi_loopback()` when
targeting the default output; otherwise substring-match rather than
equality. Exact names vary across driver updates.

**Blackwell (sm_120) INT8 kernels.** Pure INT8 CTranslate2 kernels can fail
with `CUBLAS_STATUS_NOT_SUPPORTED` on RTX 50-series hardware. Yamabiko uses
`float16` for Standard and `int8_float16` for Smaller rather than pure INT8.

**faster-whisper cuDNN DLLs.** faster-whisper needs cuBLAS and cuDNN
DLLs on `PATH` or co-located with the package. Missing DLLs produce a
cryptic import error. The installer bundles the necessary DLLs into the
`yamabiko.exe` directory; dev-mode users need a CUDA toolkit on PATH.

**Windows 11 DWM chrome on frameless windows.** Qt's
`FramelessWindowHint` doesn't defeat DWM's rounded-corner clip, 1 px
system border, backdrop shadow, or hit-test caption region.
`overlay_window._suppress_windows11_chrome()` makes four
`DwmSetWindowAttribute` calls; all four are required, and removing any
brings back a visible artifact on specific backgrounds. Detailed
breakdown in [docs/design/overlay.md](docs/design/overlay.md).

**Qt CJK font rendering.** Qt6 on Windows has
[long-standing CJK blurriness](https://bugreports.qt.io/browse/QTBUG-58610)
with the default font substitution. Pin an explicit CJK font
(`BIZ UDPGothic`, bundled, by default; `Noto Sans CJK JP` is a reasonable
alternative) and set `QFont.HintingPreference` rather than relying on
Qt's defaults.

**Console codepage.** Default cp932 on Japanese Windows installs will
crash on the Hepburn macrons (`ū`, `ō`, `ā`, `ē`) in the romaji output.
`main.py` forces `sys.stdout.reconfigure(encoding="utf-8")` at startup;
don't remove it.

**Chunk-boundary artifacts.** ASR models occasionally re-emit the last
few characters of the previous chunk at the start of the next. Exact
string matching misses these because punctuation and whitespace
normalize differently. Use fuzzy suffix-overlap matching, as the
transcript-buffer code already does.

**llama-server port conflicts.** Default port 8090 may collide with
other local tooling. The endpoint URL is configurable in Settings or
`user_overrides.yaml`; setup's Translator step notices another program
answering there and names the model that answered.

**Qt global-hotkey race.** `keyboard`-library callbacks fire on a
non-Qt thread. Any handler that touches UI state must marshal back to
the Qt thread via `QMetaObject.invokeMethod`, or you'll hit random
crashes that only reproduce when the hotkey fires while the overlay is
rendering.

---

## What this document is not

- **Not user-facing documentation.** Install, troubleshooting, hotkeys,
  and genre picker live in the README.
- **Not subsystem design rationale.** Why the NER stack is dual, how
  the settings live-apply works: those live under
  [docs/design/](docs/design/).
- **Not a roadmap.** Upcoming work lives in
  [docs/ROADMAP.md](docs/ROADMAP.md).
