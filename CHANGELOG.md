# Changelog

All notable changes to Yamabiko are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0-beta] - 2026-11-08

### Added

- **python-soxr (libsoxr) is now bundled.** Yamabiko uses it to resample
  captured audio. It's licensed under LGPL-2.1-or-later. Its exact source,
  `soxr-1.1.0.tar.gz`, is published with each GitHub release, and its Python
  wrapper ships as replaceable `.py` files in the install folder.
- **Bundled fonts.** BIZ UDPGothic, BIZ UDPMincho, and Klee One (SIL Open
  Font License 1.1) now ship with Yamabiko. You can pick BIZ UDPGothic and
  Klee One as subtitle fonts, including on the browser/OBS page. BIZ
  UDPMincho sets window titles. The install folder grows by about 31 MB.
- **New subtitle look.** Subtitles now sit on a broadcast-style caption: the
  Japanese on a paper strip, the English on a dark bar, and a vertical
  vermilion banner beside them that shows the genre or the Japanese title of
  the series you picked. Older lines rise above as outlined text and fade
  out. The browser/OBS page shows the same design.
- **Mode, Alignment, Echo, and Banner settings.** They're in Settings,
  beside the preview. **Mode** is **Watch** (English first) or **Learn**
  (Japanese first, with colored readings). **Alignment** is **Left** or
  **Center**. **Echo** sets which language older lines keep. **Banner**
  shows Japanese, or Japanese and English.
- **Pause shows on screen.** While Yamabiko is paused, the banner turns gray
  and reads 一時停止, even when no subtitles are showing.
- **Shortcuts tab in Settings.** To change a shortcut, click it and press
  the new keys. Press Esc to cancel. Changes work right away, without a
  restart. You can still record keys that another app uses. They're marked
  "Used by another app", and recording the same keys again retries the
  shortcut. Two actions can't share a shortcut: Settings marks the later one
  "Already used for" the other and saves nothing until you change it. If you
  added a shortcut to `user_overrides.yaml` by hand and Settings can't
  record it, Settings shows it as you wrote it and keeps it.
- **Tray flyout.** Click the tray icon to open a panel in the new look. It
  has **Pause** or **Resume**, subtitles shown or hidden, **Watch** or
  **Learn**, the genre, the series you picked with its cover (**Change** and
  **Clear**), the subtitle position and alignment, **Move subtitles**,
  **Settings**, **Troubleshooting** (**Open log folder** and **Save
  diagnostics**), and **Quit**. Point to a button to see its shortcut.
- **The tray icon shows what Yamabiko is doing.** It turns gray with two
  pause bars while Yamabiko is paused, shows 山 as an outline while the
  subtitles are hidden, and gets a small amber dot when something is wrong.
  It works on light and dark taskbars. Point to it to read the status, for
  example "Yamabiko is listening (Anime: Frieren)".
- **Yamabiko tells you when translations stop or your audio device goes
  away.** A Windows notification says so ("Translations stopped" or "Can't
  hear your audio"). An お知らせ line above the subtitles stays while the
  problem lasts, on the desktop overlay and on the browser/OBS page. When
  the problem is fixed, the line says "Translations are back." or "Listening
  again." for a few seconds. Click the notification to open the flyout.
- **Clear a series.** Click **Clear** in the flyout or the series search bar
  to go back to the genre alone.
- **See your series load.** After you pick a series, the search bar shows
  "Loading character names for …" with a line along its bottom edge that
  fills as the names arrive. Then it shows "Now watching …" for two seconds
  and slides away. If the names can't be loaded, the "Series not changed"
  notification tells you, as before.
- **The series you picked keeps its cover.** Yamabiko saves the cover of the
  series you watch, so after a restart the flyout and the search bar show it
  right away, even offline.
- **Faster subtitles.** Each sentence now appears as soon as it's stable,
  instead of when the speaker pauses. It's on by default, in **Settings** >
  **Timing**. A lone うん, はい, or ええ that the early guesses hear at the start
  of a line waits for the pause, together with the lines after it, so it
  appears only when the full line has it. Turn the setting off to wait for
  the pause again, with slightly fewer recognition errors.
- **Setup opens again when an update needs new models.** It downloads only
  what's missing, and its first page says so and how big the download is. A
  new install doesn't show this.
- **Choose the monitor and drag the subtitles into place.** Long lines wrap
  at a width you set instead of running off the screen.
- **Series-aware character names.** Press Ctrl+Alt+S, or right-click the
  tray icon and click **Pick a series**, and then find the show you're
  watching. Yamabiko looks up its characters on AniList and helps the
  translator keep their names the same from line to line. To turn it off,
  set `gazetteer.enabled: false`.
- **Crashes leave a trace.** A crash in any part of Yamabiko, including the
  translator, is written to the log folder, so a bug report can show what
  happened.
- **Save diagnostics.** In the flyout, **Troubleshooting** > **Save
  diagnostics** saves your logs, version details, and the names of your
  settings to a zip file on your desktop, ready to attach to a bug report.
  **Open log folder** is next to it.
- **Support for 8 GB graphics cards.** Setup reads your card's video memory
  and suggests the **Smaller** tier for 8 GB cards. You can change the tier
  later in Settings.
- **Transcript logging.** When you turn on `logging.transcript_log_enabled`,
  the log records the Japanese and English of each line, which helps tell a
  mishearing from a mistranslation. It's off by default, and without it the
  logs don't contain what was said.

### Changed

- **New speech recognition model: Kodama-1.** The default speech model is
  now Kodama-1 (`Yamabiko-ASR/kodama-1`), Yamabiko's own Japanese model:
  whisper-ja-760M by efwkjn with a fine-tuned decoder, in place of Whisper
  large-v3-turbo. Interjections and short replies come through more often,
  and fewer lines appear where nobody speaks (such as a lone 「うん。」). It runs
  without a decoder prompt and skips what it recognizes as non-speech. The
  download (about 1.5 GB) and its use of graphics memory stay about the
  same. Kodama-1 is licensed under CC BY-SA 4.0; see NOTICE.
- **New translation model: Qwen3.5-9B.** Both tiers use it, in place of
  shisa-v2.1.
- **Voice detection keeps up with anime and drama.** Dialogue under music or
  sound effects is no longer skipped or cut into pieces.
- **Voice detection also listens with a sound tagger.** A small sound event
  model (EfficientSED, bundled with the installer, about 21 MB) runs on the
  CPU beside Silero VAD, and audio counts as speech when either of them
  hears a voice. Lines are caught more often, especially short lines and
  interjections. The cost is delay: the tagger listens 0.24 seconds ahead,
  so subtitles appear about a quarter to three quarters of a second later
  than they would without it. If the tagger's file is missing or damaged,
  Yamabiko asks you to reinstall instead of starting without it.
- **Lines end sooner after a pause.** Voice detection now ends a line after
  0.4 seconds of silence instead of 0.5 seconds. Only a line with less than
  0.5 seconds of speech (before, 1 second) is still held open, for up to 1.5
  seconds of silence, to join the words after it. This wins back part of the
  sound tagger's delay, with no measurable change in recognition.
- **Fewer broken-up lines.** Audio between lines is carried into the next
  one instead of being dropped, and sound that the speech model rates as
  non-speech is filtered out.
- **Subtitles show only the translation.** They never include the
  translator's notes to itself.
- **Lost audio is easier to diagnose.** The log now says why voice detection
  discarded speech it was holding (a full queue, a pause, or a capture
  reset), how much queued audio was skipped after a pause, and when the
  Windows audio clock shows a jump that suggests missing input.
- **Capture resampling.** Loopback audio is converted to 16 kHz by a
  streaming soxr resampler that carries its state from one capture block to
  the next, in place of plain decimation.
- **Voice detection restarts at capture gaps.** When capture reports a gap
  (audio dropped because a queue was full, an input gap or failed
  conversion, a reopened audio stream, or a pause or resume), voice
  detection resets instead of joining audio from both sides of the gap. A
  line that was still open at the gap is dropped.
- **Silero VAD sees the context it expects.** Each 512-sample window now
  reaches the model together with the last 64 samples of the window before
  it. Before, the model saw each window alone.
- **Pausing drops queued audio.** Audio captured before you paused that was
  still waiting to be processed is discarded, instead of being transcribed
  after you resume.
- **New colors.** Settings, the tray menu, series search, and the placement
  hint use the new Mist (light) and Night (dark) colors. The theme still
  follows Windows unless you pick **Light** or **Dark**.
- **New app icon.** It's a vermilion tile with a bold 山 and a ridge line. At
  the smallest size (16 pixels), only the 山 shows.
- **New subtitle defaults.** The defaults are BIZ UDPGothic Bold for
  Japanese, Bahnschrift SemiBold for English, and a dark bar at 93% opacity.
  Settings you saved before keep their values.
- **Subtitles are left-aligned by default.** Where you placed subtitles
  before still sets their height, but their left-to-right position now
  applies only when **Alignment** is **Center**. **Left** uses its own
  position, 6% from the left edge of the screen. **Move subtitles** in the
  flyout sets it.
- **Reduce motion now covers the subtitles.** Lines and plates switch
  instantly instead of sliding and fading, on the desktop overlay and the
  browser/OBS page.
- **The browser/OBS page uses the same text sizes as the desktop overlay.**
  Sizes are in points on both now. The page sits at the default anchor: its
  left edge 6% from the left of the screen, or centered, 12% above the
  bottom.
- **Settings has a new look.** It has a landscape title band with the 山彦
  seal, the subtitle preview over a night street photo with the overlay
  choices beside it on every tab, and the tabs **Subtitles**, **Position**,
  **Timing**, **Audio & model**, and **Shortcuts**. **Background opacity**
  is now shown in percent and **Time on screen** in seconds. Your saved
  values don't change.
- **Settings fits short screens.** At full size, nothing in Settings
  scrolls. On a screen too short for the whole window (such as a 1080p
  screen at 150% scale), the preview gets smaller first, and then the
  current tab's settings scroll. **Reset to defaults**, **Cancel**,
  **Apply**, and **OK** always stay on screen. Scrolling the mouse wheel
  over a field you haven't clicked scrolls the settings instead of changing
  the value.
- **Choosing a color opens a compact picker.** It has the overlay's own
  colors at the top, a hex field, and a sample line.
- **When saving fails, Settings says which file it couldn't write.** It
  offers **Retry**.
- **The translation model download in Settings shows how much is done and
  about how long is left.** If it stops, you can retry from the same window.
- **Right-click the tray icon for a short menu.** It has **Pause**, **Pick a
  series**, **Settings**, and **Quit Yamabiko**, with your current shortcuts
  beside them. The rest of the old menu moved to the flyout.
- **Notifications have new wording and show the Yamabiko icon.** They appear
  under the name "Yamabiko". An update reads "Yamabiko 0.3.0 is out".
  "Diagnostics saved" names the zip file, and clicking it shows the file in
  its folder.
- **The tray tooltip shows the status instead of the version.** To see your
  version, go to Windows **Settings** > **Apps** > **Installed apps** and
  find Yamabiko.
- **Series search is now a slim bar at the top of your subtitle screen.**
  Press Ctrl+Alt+S, or click **Pick a series** or **Change**, to open it at
  the top center of the monitor your subtitles are on, away from the
  subtitles. The series you're watching and **Clear** are beside the search
  box. It shows up to six matches, each with its cover, Japanese title,
  year, format, and number of episodes. If you add your AniList username in
  Settings, the empty search lists the shows on your AniList Currently
  watching list, most recently updated first, and shows "Episode 6 next" on
  a show that's still airing. If AniList can't be reached, the bar says so
  and offers **Try again**. Use the Up arrow and Down arrow keys to move,
  Enter to pick, and Esc to close. With **Reduce motion** on, the bar opens
  and closes without animation.
- **A new window when Yamabiko can't start.** It replaces the old message
  box and says what went wrong and what to try in plain words, one step per
  line, for example "No NVIDIA graphics card found" or "Some model files are
  missing". Beside it, a **Report** panel lists what the developer needs:
  the Yamabiko version, your Windows version and build, your graphics card
  with its video memory and driver, the error, and the last lines of the
  log, with **Open log folder**. **Copy report** copies all of it to the
  clipboard with your Windows user name removed everywhere, even in paths
  and log lines, ready to paste into a message to the developer. Nothing is
  sent automatically. Where Settings can fix the problem, **Open Settings**
  takes you there. The window follows your Light or Dark theme and opens in
  front of other windows.
- **Clearer message when Yamabiko is already running.** It now asks you to
  close Yamabiko setup, or to quit Yamabiko from its tray icon, before you
  start it again, and to end it in Task Manager if you can't find either.
- **A new first-run setup.** Setup has the app's new look in one window,
  with its steps listed on the left (**Welcome**, **Sound**, **Content**,
  **Quality**, **Download**, **Translator**, and **Done**) and checked off
  as you go. Click a finished step to go back to it. **Sound** lists your
  outputs with the Windows default marked, and **Test** shows a level meter
  for 5 seconds. **Content** asks what you watch most and shows a sample
  subtitle that follows your pick. **Quality** names your graphics card and
  marks the model size recommended for it. A card with less than 8 GB of
  video memory gets no mark, and setup says Yamabiko needs 8 GB.
  **Download** shows how much of the total is done, in GB, and about how
  long is left, with a row each for speech recognition, the translation
  model, and voice detection. A thin line under **Download** in the step
  list keeps showing its progress while you look at other steps, and going
  back no longer stops the download. When the disk is full, setup names the
  drive and how much space to free up, and it can delete models from older
  Yamabiko versions. When the translator doesn't answer yet, or its test
  answer is empty or unreadable, setup says so in plain words, and you can
  retry or skip. The last step shows where Yamabiko sits in the tray and
  lists your shortcuts. Setup uses your Light or Dark theme and opens in
  front of other windows.
- **A new installer and uninstaller.** They have the app's new look: a
  landscape picture on the first page, the license as formatted text, and a
  last page with **Start Yamabiko** selected, so Yamabiko starts right after
  it installs. When Yamabiko is already installed, the first page says which
  version you have and offers to update it, and updates skip the license. If
  Yamabiko is still running, the installer now asks you to quit it after you
  click **Next**, instead of before its window opens. Uninstalling shows how
  much space your models and settings take up and where they are, with a
  link that opens the folder.

### Removed

- **The turn detector.** It barely changed where lines ended, so it's gone,
  along with its license prompt and download in setup.
- **The `lifecycle.llama.*` and `translation.stream` settings.** If they're
  still in `user_overrides.yaml`, Yamabiko ignores them and logs a warning.

### Fixed

- **Lines made only of dots (……) are no longer shown as subtitles.** A pause
  at the start of speech could come through as a line of dots on its own and
  show as "...". Lines with any words in them, dots included, are still
  translated and shown, and the words after the dots no longer wait for the
  next line.
- **A subtitle no longer repeats the line before it.** The translator could
  restate the previous subtitle at the start of a new one and then translate
  the new sentence. The earlier lines it reads for context are now marked as
  already translated and shown, so it uses them only to understand the new
  line.
- **Yamabiko's notifications now show while subtitles are on screen.**
  Windows took the subtitle overlay for a full-screen app, and its default
  Do not disturb rule hid them.
- **Local translation traffic no longer goes through a system proxy.** With
  a Windows proxy set up, requests to the bundled `llama-server` and the
  pipeline's liveness check went to the proxy, so Japanese transcripts left
  your PC, translation failed, and the hang detector couldn't see the
  translator. Calls to local services now always connect directly. A remote
  `translation.endpoint` still uses your proxy.
- **Clear message for a damaged voice detection model file.** A cached
  Silero VAD file that fails its integrity check now shows one message that
  names the file to delete, instead of the app restarting repeatedly and
  ending with a generic error. Running setup again doesn't replace a damaged
  cached file; deleting it does.
- **Furigana reads verbs and adjectives correctly.** Kanji are now read in
  the context of their whole word, so 行きました shows 行 as い instead of こう, and
  the furigana sits over the kanji only (思う no longer shows おもう over 思
  followed by a second う).
- **Furigana for names with 々 or small kana.** These names are read
  correctly now, so 佐々木 reads ささき.
- **Real dialogue is no longer cut down as a "here is the translation"
  preamble.** Lines such as "This is my English teacher." or "So this is the
  output?" are now shown in full instead of being cut to "teacher." or "?".
- **The overlay no longer keeps old subtitles in memory.** Every subtitle
  that faded out normally stayed in memory until you quit, so the overlay
  slowly grew during long sessions.
- **Quitting no longer forgets your subtitle position for a missing
  monitor.** If your subtitles were placed on an external monitor that isn't
  connected, Yamabiko keeps that placement for when the monitor comes back.
  Quitting used to erase it. Quitting now saves the position only when you
  actually moved the subtitles.
- **Starting with `--config` no longer changes your saved settings.**
  Position presets, **Move subtitles**, and quitting used to write the
  subtitle position to your own settings file during a `--config` run.
- **Enter on a focused button in Settings presses that button.** Pressing
  Enter while **Cancel**, **Apply**, **Reset to defaults**, or a color
  button had focus used to act like **OK**: it saved your changes, closed
  the window, and could start a model download. Enter elsewhere in Settings
  still means **OK**.
- **The overlay log no longer records the series you pick.** If you picked a
  series while the translator wasn't connected, the log line (which goes
  into the diagnostics zip file) included the anime title. It now records
  only which command was dropped.
- **A settings file saved in the wrong text encoding no longer stops
  Yamabiko from starting.** A `user_overrides.yaml` written by Windows
  PowerShell 5.1 (UTF-16) or by Notepad in ANSI mode made Yamabiko restart
  over and over and kept setup from saving. Yamabiko now starts with default
  settings, and the next save moves the unreadable file aside as
  `user_overrides.yaml.corrupt.bak` and writes a new one.
- **Startup error messages stay open until you close them.** When Yamabiko
  restarted itself after a problem and then stopped at a message such as "No
  audio source" or "GPU not detected", it could close that message by itself
  and try again every few minutes. The message now stays until you close it,
  and closing it stops Yamabiko as intended.
- **When the audio device you picked is gone, Yamabiko says which one.** It
  offers to open Settings so you can pick another.
- **Unfinished lines are no longer glued across lost audio.** When audio was
  lost (a full queue, a capture stream reset, or a pause), the start of an
  unfinished sentence from before the gap could be joined to the first line
  after it, so one subtitle mixed two unrelated lines. The unfinished part
  is now dropped at the gap, including when speech recognition hears nothing
  in the first stretch after it, and a very long unfinished line from before
  a pause is no longer shown after you resume.
- **Quieter logs while paused.** Pausing used to write one log line for
  every 30 milliseconds of discarded audio (about 10,000 lines for a
  5-minute pause). The log now records one line when a stretch of discarded
  or dropped audio starts and one with its totals when it ends.
- **Lines with two stage directions keep their dialogue.** A translation
  such as "*(laughs)* Really? *(sighs)*" used to disappear entirely. Now
  only the trailing direction is removed.
- **Parenthetical dialogue made of everyday words is no longer hidden.**
  Lines such as "(It's over. Drop it. Nothing else.)" were mistaken for the
  translator repeating its instructions. Now only a parenthetical line that
  talks about the translation task (translating, subtitles, honorifics, or
  register) is treated that way.
- **A slow translator no longer holds a finished subtitle for a minute.**
  When the negation check asks the translator for a second try, that try now
  gets one request of at most 5 seconds. If it doesn't answer in time, the
  first translation is shown.
- **Curated furigana readings win over leftover learned ones.** The bundled
  genre glossaries held readings recorded during development, and they
  overrode the curated list (日本 showed にっぽん instead of にほん). They now ship
  empty, and the curated list also outranks readings that an older version
  learned on your PC.
- **Picking a series works when AniList's series details fail to load.** If
  the request for the show's related seasons and description failed, the
  pick failed or fell back to an older saved copy of the names. The pick now
  keeps the show's own character names and skips the related seasons.
- **Kanji character names are hinted by family or given name alone.** After
  you pick a series, a line that says only 竈門 or 炭治郎 now gets the name hint
  (Kamado, Tanjirou). Before, only the full name 竈門炭治郎 matched.
- **Place names and common words are no longer turned into character
  names.** The near-match against the series' characters ran before the
  built-in list of places, so アメリカ could be hinted as Amelia, and ordinary
  words such as ライター (a lighter) as Heiter. Known places now win, and the
  near-match runs only for proper nouns.
- **No partial furigana for names the dictionary can't read.** 禰豆子 showed こ
  over 子 alone, which read as the furigana for the whole name. Such names
  now show no furigana.
- **Quit closes everything.** Quitting from the tray also stops the
  translator, so its video memory is freed and you can start Yamabiko again
  right away.
- **Yamabiko shuts down cleanly when the translation service stops for
  good.** The pipeline wasn't asked to shut down and was force-closed after
  its wait, which skipped its glossary save. It now gets the normal shutdown
  request first, and the translation service itself is stopped after half a
  second instead of 5 seconds.
- **The translator's slow first start is no longer a failure.** Yamabiko
  waits for the translator to finish loading before it reports a startup
  error.
- **You can start Yamabiko again while its error message is open.** The
  "already running" check stayed locked until the last error message was
  closed, so a relaunch in the meantime was refused.
- **A Yamabiko already running in another Windows session is detected.**
  Starting a second copy from another user session, or from an administrator
  prompt, skipped the "already running" check instead of stopping.
- **Leftover processes from a crash are cleaned up in folders with
  non-English names.** The startup cleanup misread install paths with
  Japanese or other non-ASCII characters and left those processes running.
- **A translation service that stops right after starting points to the
  right log.** The message now names `llama-server.log` and the usual causes
  (port 8090 in use or not enough GPU memory), instead of sending you to the
  `main-*.log` files.
- **An invalid configuration is recorded in `main-watchdog.log`.** The
  "Invalid configuration" message used to leave no trace in the logs.
- **Startup errors still show when Yamabiko's Qt components are damaged.**
  Yamabiko now falls back to a plain Windows message box instead of closing
  without a message.
- **Changing the translation tier in Settings switches the model.** Before,
  the old model kept loading. Settings now downloads the new one first if
  you don't have it yet.
- **Damaged model downloads are caught.** Setup and Settings check the
  speech, translation, and voice detection model files they download against
  known checksums. A damaged file is deleted and its download fails with an
  error, so **Retry** gets it again instead of Yamabiko starting with a
  broken model.
- **Interrupted downloads no longer leave gigabytes behind.** Canceling or
  closing setup during a download left partial files in the model cache that
  were never reused. The next download now deletes them first.
- **Delete old Yamabiko models is offered only for Yamabiko's own model
  cache.** If you point `HF_HOME` at a cache shared with other tools, setup
  no longer offers to delete old models from it.
- **Setup lists only devices Yamabiko can capture from.** A microphone or
  virtual input with "Loopback" in its name was offered and then failed when
  Yamabiko started.
- **Setup keeps your Standard translation model choice.** Running setup
  again on an 8 GB card replaced a saved **Standard** choice with
  **Smaller**.
- **Saving settings survives a briefly locked settings file.** When an
  antivirus scanner or backup tool had `user_overrides.yaml` open for a
  moment, the save failed. Yamabiko now retries for up to half a second.
- **Starting with `--config` before setup has finished shows "Setup
  required".** Setup used to run inside the `--config` run, which then
  ignored the choices it saved. Start Yamabiko once without `--config` to
  finish setup.
- **Setup's translation check notices another program on the translator's
  port.** When LM Studio, Ollama, or another llama-server answered there,
  the check passed even though Yamabiko's own translator wasn't the one
  answering. It now names the model that answered and points to
  `translation.endpoint`.
- **Opening the update page no longer ties your browser to Yamabiko.** A
  browser started by clicking "Update available" was closed together with
  Yamabiko on the next pipeline restart or on **Quit**, possibly in the
  middle of downloading the installer. It now runs on its own.
- **Clicking a later notification no longer opens the update page.** If an
  "Update available" notification timed out without a click, a click on any
  later Yamabiko notification opened the GitHub release page.
- **Local programs can no longer connect to the subtitle server without the
  per-launch key.** Every connection to the local subtitle server, including
  the desktop overlay's own, now needs it. Before, any program that sent no
  browser origin, such as one run by another signed-in Windows user, could
  read every subtitle and send commands such as **Quit**. While the browser
  overlay page is on (`overlay.browser`, the default), a local program can
  still ask the server for the page's key. Set `overlay.browser: false` if
  other people sign in to this PC.
- **A broken font setting no longer hides every subtitle.** A font name that
  isn't text, sent by a local program or typed into `user_overrides.yaml`,
  made the desktop overlay drop every subtitle. It now falls back to the
  default font.
- **Subtitles placed near the top of the screen stay on screen.** With a
  high position, older lines were stacked above the top edge of the monitor.
  The stack now moves down until every line is visible.
- **Settings no longer replaces a missing font with a substitute.** If your
  font isn't installed, Settings showed a substitute and saved it over your
  choice when you applied any other change. Your font is now kept unless you
  pick another one.
- **Picking an audio device after the saved one was unplugged now saves
  it.** Settings showed the default device as selected while warning that
  your saved device was unavailable, and choosing that device saved nothing.
  The missing device now has its own "(unavailable)" entry, so any device
  you pick is saved.
- **A shortcut that can't be used is now reported.** If Ctrl+Alt+P or
  another Yamabiko shortcut is already used by another app, or can't be set
  up, a "Shortcut unavailable" notification names it. Click it to pick a new
  one in **Settings** > **Shortcuts**. Each taken shortcut is announced
  once, and again only when you change it. Before, a shortcut that failed to
  set up was only written to the log, and one that another app held went
  unnoticed.
- **A translator that stops answering is restarted.** When the bundled
  translator hung, Yamabiko restarted only its own pipeline, and the hung
  translator ended the session with "Translator service didn't start".
  Yamabiko now restarts the translator too.
- **Typing after Ctrl+Alt+S goes into the series search.** The search opened
  without keyboard focus, so what you typed went to the video or game behind
  it. It now takes keyboard focus when it opens.
- **A copied startup error stays on the clipboard.** The details copied from
  a startup error message were gone as soon as Yamabiko closed, and **Copy
  details** closed the message. **Copy report** keeps the window open, and
  what it copied stays after you close it.
- **Setup's model downloads no longer fail right away when Yamabiko runs
  without a console.** When Yamabiko started without one, for example
  `yamabiko.exe --run-main` from a shortcut, setup stopped every model
  download before its first byte with "'NoneType' object has no attribute
  'write'".
- **Switching Watch and Learn, alignment, or text options no longer makes
  the line on screen disappear a moment later.** The line gets a fresh full
  display time in its new layout.

### Security

- **Bundled libraries updated for known advisories.** The updated libraries
  are starlette 1.3.1, anyio 4.14.2, idna 3.15, certifi 2026.10.19, and
  fsspec 2026.6.0.
- **Speech recognition engine updated to CTranslate2 4.8.3.** It was 4.7.2.
  The new version validates model files more strictly as it loads them: it
  checks sizes and indexes before it allocates or reads memory, which closes
  a heap overflow and out-of-bounds reads that a damaged or malicious model
  download could trigger. It also fixes a crash on Windows with CUDA in
  Whisper word alignment, which Yamabiko doesn't use today.
- **File locking library updated to filelock 4.2.4.** It was 3.25.2. The
  update fixes three advisories: an oplock on Windows that could stall a
  soft lock past its timeout, and two in its SQLite-backed read/write lock.

### Migration

- Install over Yamabiko 0.1.0 beta. Your settings stay. When Yamabiko
  starts, setup opens again and downloads Kodama-1 and Qwen3.5-9B: about 6.6
  GB, or 5.6 GB on the **Smaller** tier.
- The models that 0.1.0 beta downloaded aren't used anymore. They take about
  7 GB in `%LOCALAPPDATA%\Yamabiko\models`. If setup runs short of space,
  **Delete old Yamabiko models** removes them, along with older speech
  models from development builds. You can also delete the folder yourself.
- Settings that 0.1.0 beta saved keep working. If you edited a value by hand
  and it's now out of range, Yamabiko doesn't start, and its message names
  the setting to fix. Yamabiko ignores the `audio.device_name` and
  `sentencing.turn_detector` settings.
- If `translation.model` still names the old shisa model, Yamabiko switches
  back to the bundled translator and removes that line from
  `user_overrides.yaml`. This happens only while `translation.endpoint` is
  the default.
- Logs are now one file per part of Yamabiko: `main-watchdog.log`,
  `main-pipeline.log`, and `main-overlay.log`. You can delete an old
  `main.log` file.
- The setting `overlay.browser: false` now turns off only the browser/OBS
  page. Yamabiko always uses its WebSocket port (`overlay.websocket_port`,
  8765 by default), so if another app uses that port, pick another one.
- A small `user_overrides.yaml.lock` file sits next to your settings. You
  can delete it while Yamabiko isn't running.
- The `vad.threshold` setting now defaults to 0.3. If you lowered it to stop
  lines from being dropped, you can remove it. For clean speech, 0.5 gives
  the old behavior.


## 0.1.0-beta - 2026-04-15

First shareable beta. Private distribution only; public release is a
later goal.

### Added

- WASAPI loopback audio capture via PyAudioWPatch
- Pinned community Silero VAD v5-era ONNX graph chunking with
  Japanese-tuned silence thresholds
- ASR via faster-whisper large-v3-turbo (float16 on GPU)
- Japanese sentence splitting via fast-bunkai
- End-of-turn detection via LiveKit multilingual turn-detector
  (Qwen2.5-0.5B distilled ONNX); license-gated first-run download
- Translation via shisa-v2.1-qwen3-8b on local llama.cpp with
  genre-aware prompting (variety / anime / news / drama)
- Stacked furigana rendered by a custom Qt QPainter
- Romaji (cutlet + UniDic) with glossary overrides
- Qt overlay with frameless always-on-top window and configurable styling
- Browser overlay (FastAPI + WebSocket) for OBS-style stream setups
- System tray with pause/resume, genre switching, live settings
- First-run wizard: audio-device picker, license acceptance, unified
  model-download page (shisa GGUF, faster-whisper CT2, LiveKit
  turn-detector, Silero VAD pulled sequentially into
  `%LOCALAPPDATA%\Yamabiko\models\` via a QThread worker with cancel
  flag and re-entry guard), and llama-server smoke test
- Global hotkeys: pause/resume (Ctrl+Alt+P), toggle overlay (Ctrl+Alt+O)
- Pipeline-stage exception supervisor with rate-limited notifications
  and escalation to a user-facing error dialog
- Rotating log files under `%LOCALAPPDATA%\Yamabiko\logs`
- Subprocess watchdog (Start Menu target) with exponential backoff
  and single-exe argv dispatch: the frozen `yamabiko.exe --run-main`
  re-enters the same PyInstaller bundle and calls
  `yamabiko.main.main()` directly
- `yamabiko.exe --version` flag on the watchdog entrypoint
- Translator retry-once on transient network errors and 5xx responses
- Exponential-backoff WS reconnect on the Qt and browser overlays
- Vendored-binary manifest directory `third_party/llama.cpp/` with
  `COMMIT.txt` template, upstream MIT `LICENSE.txt`, and a README
  documenting the refresh recipe; the binaries themselves stay out of
  git
- PyInstaller onedir spec at `installer/yamabiko.spec`, PowerShell build
  orchestrator `installer/build.ps1`, NSIS Unicode installer script
  `installer/yamabiko.nsi`, and an installer build README
- Legal scaffolding: `LICENSE` (PolyForm Noncommercial 1.0.0), `NOTICE`,
  `CLA.md`, `CONTRIBUTING.md`

### Changed

- Python package renamed from `realtimejpsubs/` to `yamabiko/`;
  `yamabiko/__init__.py`'s `__version__` is now the single source of
  truth for the version string, and `yamabiko.main` exports a `main()`
  entry function
- **Config is now layered.** The single
  `%LOCALAPPDATA%\Yamabiko\config.yaml` is replaced by a shipped
  `default_config.yaml` (authoritative schema, read in place from the
  app bundle) plus a sparse `user_overrides.yaml` at
  `%LOCALAPPDATA%\Yamabiko\`. The two are deep-merged at load time.
  This prevents shipped-default improvements from being silently
  pinned by stale copied values in the old flat file, and stops
  removed-from-schema keys from accumulating in the user's config
  forever
- On first launch after the upgrade, existing installs migrate once:
  the legacy `config.yaml` is diffed against shipped defaults, only
  the diff is written to `user_overrides.yaml`, and the original is
  renamed to `config.yaml.migrated.bak`. Both-files-present is
  non-destructive (overrides wins; legacy left alone with a warning)
- The sparse writer collapses a key back to the shipped default when
  the user's new value matches it, which makes "reset to defaults"
  fall out of the data model for free
- First-run wizard `LicensePage` is now a pure acceptance-gate
  (license display + accept/decline radios); all model-download logic
  was folded into the new unified `DownloadsPage`
- shisa GGUF source switched to
  `dahara1/shisa-v2.1-qwen3-8b-UD-japanese-imatrix` Q4_K_XL
  (JP-imatrix + Unsloth Dynamic 2.0) for better JP quality at the
  ~5.5 GB quant
- User-facing strings renamed from RealtimeJPSubs / JPFluxTranslate to
  Yamabiko, including the Autostart Task Scheduler entry description
  ("Yamabiko autostart")
- Overview file renamed from `RealtimeJPSubs_Overview.md` to `OVERVIEW.md`

### Removed

- Inline LiveKit-only download logic from `LicensePage` (folded into
  the unified `DownloadsPage`)

### Known issues

None open.

### Deferred

- Code-signed installer (Azure Trusted Signing)
- Auto-update mechanism
- Public GitHub repo

[Unreleased]: https://github.com/Yamabiko-ASR/yamabiko/compare/v0.2.0-beta...HEAD
[0.2.0-beta]: https://github.com/Yamabiko-ASR/yamabiko/releases/tag/v0.2.0-beta
