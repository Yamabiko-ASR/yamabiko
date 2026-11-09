# Overlay

Yamabiko ships two overlays that render the same subtitle stream: a
native Qt window that floats over fullscreen video, and a browser
overlay intended for OBS scenes. They share a single settings surface
and a single source of styling truth.

## Qt overlay

### Frameless always-on-top window

`overlay_app/overlay_window.py` defines `OverlayWindow`, a `QWidget` with:

- `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.NoDropShadowWindowHint`
- `Qt.WA_TranslucentBackground` (plus `WA_NoSystemBackground`, disabled
  auto-fill, and a transparent stylesheet) so the empty window is fully
  invisible; the `Qt.Tool` flag alone can otherwise leave a
  platform-painted backdrop on Windows
- `Qt.WindowTransparentForInput` in the default **locked** state, so
  clicks pass through to the game/video underneath

Locked mode is the default; the overlay behaves like a static HUD.

### Windows 11 DWM quirk

Windows 11's Desktop Window Manager reintroduces subtle window chrome
(a rounded-corner clip, an accent-colored border, a Mica/acrylic
backdrop, and a non-client drop shadow) even on windows that never had
a title bar. Qt's `FramelessWindowHint` only removes the title bar; it
does not address this platform-specific behavior.

`_suppress_windows11_chrome(hwnd)` makes four `DwmSetWindowAttribute`
calls:

- `DWMWA_NCRENDERING_POLICY` = `DWMNCRP_DISABLED` (disables non-client
  rendering, including the drop shadow)
- `DWMWA_WINDOW_CORNER_PREFERENCE` = `DWMWCP_DONOTROUND`
- `DWMWA_BORDER_COLOR` = `DWMWA_COLOR_NONE` (`0xFFFFFFFE`)
- `DWMWA_SYSTEMBACKDROP_TYPE` = `DWMSBT_NONE`

It then issues a `SetWindowPos` with `SWP_FRAMECHANGED` so the new
attributes take effect on an already-visible window. All four
attributes are required; removing any of them brings back a visible
artifact on specific backgrounds.

The suppression runs from `showEvent`: once immediately, then again
from a short single-shot retry timer, because some attributes
(backdrop, non-client rendering) are silently ignored if set before
the first paint. Toggling interactive mode for free placement
recreates the native window handle and discards the attributes applied
to the old one, so the re-entered `showEvent` re-arms the same two-pass
suppression for the new handle.

### Pinned-monitor model

The window is not a free-floating draggable rect. It is a full-monitor
transparent click-through strip pinned to ONE screen, hosting one
full-window `SubtitleStack` (see Rendering) placed by percent offsets
(the anchor model in `position_config`). `_pin_to_screen()` resolves
`position["screen"]` against the live screen list by `QScreen.name()`
(falling back to the primary screen for `"primary"`, an unknown name, or
an unplugged monitor) and sets the window geometry to the whole screen
less its bottom row: Windows takes a shown topmost window whose rect
covers its monitor for a full-screen app (`SHQueryUserNotificationState`
answers `QUNS_BUSY`), and its default Do not disturb rule "When using an
app in full-screen mode" would then hide every Yamabiko notification
while subtitles show. The origin stays the screen's, so window px equal
screen px, and no line reaches that row (`v_offset_pct` tops out at 98).
`_reposition_stack()` sizes the stack to the window and hands it the
anchor. The newest line's bottom sits at `v_offset_pct` of screen
height. The horizontal anchor follows the Alignment setting: in Left
(the default) it is the stack's left edge at `left_offset_pct` of screen
width (default 6, range 0 to 70); in Center it is the stack's center at
`h_offset_pct` (default 50). Both keys are stored, so switching
alignment flips between two remembered spots. A line's width cap
(banner gutter included) is `max_width_pct` of the screen width,
bounded by the room the anchor leaves on screen. `place_lines` keeps
every line on screen: x is clamped, and a stack taller than the space
above the anchor is pushed down until its top is visible (below a
top-docked taskbar). The banner stays on screen the same way: a series
name taller than the stack pushes the stack down, and the paused banner
with no lines is pushed down on its own.

Position is USER STATE persisted via `write_overlay_position` (an older
install's legacy `overlay.geometry` rect is migrated to the anchor
model once on boot, then pruned). The window tracks the last saved
position, and the quit-time save writes only when the live position
differs from it, so a quit never prunes a legacy rect that boot kept on
disk. An explicit `--config` run builds the window without a persist
callback, so it writes no position at all. On screen add/remove or
primary-screen change, `_revalidate_screen()` re-pins and re-margins so
the strip tracks the current display layout.

Some Windows display changes (a monitor sleeping and waking, an in-place
DPI or resolution change, an RDP reconnect) delete and recreate the
pinned `QScreen` without emitting any of those signals. `add_subtitle`
and `show_sample` first call `_heal_dead_pin()`, which re-pins and
re-places the stack when the stored `QScreen` is dead, so the next
subtitle lands at the new anchor; `_active_screen()` also re-pins on any
other read of a dead pin. A same-monitor resolution change arrives
through the pinned screen's `geometryChanged` and takes the same re-pin
path.

### Interactive (placement) mode

`set_interactive(True)` drops `WindowTransparentForInput` so the window
receives mouse events (a 2 px dashed outline is drawn in `paintEvent`
while interactive); `set_interactive(False)` restores click-through.
The flag flip recreates the native HWND, so the re-entered `showEvent`
re-arms the two-pass DWM suppression. `set_interactive` persists
nothing; position persistence flows through `set_position`.

Placement replaces the old Unlock/Lock affordance. There are two ways in:
the one-click presets in the tray flyout's Position picker, and Move
subtitles… (in the flyout and in Settings > Position). The presets pin
the stack at a fixed vertical percentage of a named screen (Bottom, the
default 88%, or Middle, 55%; the flyout lists them as "Bottom of Display
N" and "Middle of Display N"), reset both horizontal anchors to their
defaults, and persist immediately. Free placement is
`placement.PlacementController`: it enters interactive mode, shows a
persistent sample subtitle and a hint bar, and lets the user drag the
sample to where subtitles should appear, jumping monitors when the
cursor crosses to another screen. The live drag routes through
`OverlayWindow.update_position_preview`, which never touches disk; Enter
(or a click away) saves once through the injected `on_save` callback
(the window's `set_position`), and Esc cancels back to the entry
position. `resolve_drag` is the pure cursor-to-anchor math (screen pick,
snap, clamp), separated out with no Qt dependency so it is testable
bare. The drag moves the active alignment's anchor: in Center the cursor
sets `h_offset_pct`, snapped to 50 within 2% of the screen width; in
Left it sets `left_offset_pct`, with no snap. The other alignment's
anchor is kept.

### Rendering

The subtitles are the "Telop, echoing" design (spec
`docs/superpowers/specs/2026-10-27-ui-redesign-design.md`, Surface 1),
painted by one widget, `SubtitleStack` (`overlay_app/telop_stack.py`).
The same class serves the full-screen overlay window and the Settings
preview monitor; the owner supplies the anchor (`set_anchor`), and text,
style, banner, hold timers and motion live in the stack. Three helper
modules feed it:

- `telop_look.py` (Qt-free): `look_from_style` turns the `overlay.style`
  dict into a `TelopLook` (fonts, pixel sizes, colors, paddings, the four
  choices). Garbage values fall back to the shipped defaults.
- `telop_text.py`: `layout_line` measures a line's Japanese block (ruby
  rows through `ruby_layout.layout_rows`, see
  [romanization-and-glossary.md](romanization-and-glossary.md)) and
  English block (wrapped with `QTextLayout`) for one echo age;
  `render_block` paints a block's text into a cached transparent
  `QImage` in the plate look or the echo look.
- `telop_banner.py`: the vertical banner text (`split_vertical_runs`,
  `banner_content`) and its painter (`paint_banner`).

**Two layers.** The anchored layer belongs to the stack, never to a
line: the paper strip `#f3eee3` behind the newest Japanese (ink
`#15171b`), the dark bar behind the newest English (`bg_color` at
`bg_opacity`, default `#0e1014` at 93%), and the vertical banner at the
left of the newest line. On every new line the plates and banner
reshape (position and size) to fit the newest line over 350 ms with an
ease-out curve; they never travel with a line. The text layer has one
entry per line: the newest line's text sits on the plates and wipes in
from the left (280 ms, the English 40 ms behind). When a new line
arrives, the previous line's text rises off the plates into the echo,
crossfading into its echo look as it glides.

**Sizes and faces.** `jp_size` and `en_size` are points, converted with
`px = round(pt × 4/3)`; the defaults 16 and 23 give 21 px and 31 px at
100% scale. Watch mode (English priority) uses the user's faces, bold
Japanese (default BIZ UDPGothic) with furigana at 0.5 em in ink at 60%,
and semibold English (default Bahnschrift) in `text_color`. Learn mode
(Japanese priority) makes the Japanese the headline in fixed faces:
Klee One SemiBold at `en_size × 33/31` with vermilion furigana at
0.46 em, over Segoe UI Variable Text at `jp_size` on a slimmer bar. The
`padding` key scales the design's strip and bar paddings (exact at the
default 10).

**Echo.** Older lines lose their plates and become outlined text: fill
`text_color`, a 2 px black outline (the text drawn at eight 2 px
offsets) over a soft shadow (the black text box-blurred at 85% alpha).
Age 1 is 84% size at 80% opacity, age 2 and older 70% size at 55%
opacity. Echo "Priority only" keeps the English in Watch and the
Japanese in Learn; "Both languages" keeps both. In Learn the echo
furigana turn amber `#f0bf5a`. The stack holds `max_lines` lines, the
newest included.

**Banner.** 38 px wide, and every line keeps a left gutter that wide so
the echo stays aligned with the newest line. Text is set vertically
(kana and kanji upright with vertical alternates, Latin sideways) in
BIZ UDPGothic Bold 18 px, white on vermilion `#d2452f`. It shows the
picked series' Japanese title (`series_native`) when one is known,
otherwise the genre (バラエティ, ニュース, アニメ, ドラマ). With Banner
"日本語 + English" it adds the English (the series' tray label without
its "(year)", or the genre name) sideways. The banner is at least as
tall as the newest line; a long name grows it upward into the gutter
beside the echo instead of shrinking the text. Paused: the tint fades to
gray `#5b5f66` (300 ms) and the content becomes a pause glyph and
一時停止. While paused the banner stays at the anchor even with no lines
on screen; after an unpause with no lines it fades out still showing the
gray 一時停止.

**Alignment.** Left: lines are left-aligned on the anchor with ragged
right edges. Center: each line is centered on the anchor, its strip and
bar share the line's full width, and the text is centered. Plates and
banner glide sideways to stay under the newest line.

**Lifetimes and timing keys.** Each line starts its own `hold_ms` timer.
`fade_in_ms`: the anchored layer fades in when the stack appears from
empty (the text wipes in). `fade_out_ms`: a line whose hold ends fades
out in place; when the last line goes, the plates and banner fade out
with it (the banner stays while paused). A line pushed past `max_lines`
fades out over 350 ms. When a new line arrives, the older lines and any
lines still fading out rise together by its height. If it arrives while
the plates and banner are still fading out, they fade back in and
reshape from where they are. The placement and preview sample appears
and goes at once (no wipe, no fade), and so do the plates and banner
when it is the newest line, or when hiding it leaves no line (while
paused the banner stays). Other lines keep their motion.

**Reduced motion.** `overlay.reduce_motion` reaches the window at boot
and on Settings Apply (`OverlayWindow.set_reduce_motion`). With it on,
every change lands at once: no wipe, glide, reshape, crossfade or fade,
and anything in flight when it turns on lands immediately.

**Painting.** Motion is tweened on one clock and repainted by a 16 ms
timer that only runs while something moves; each tick repaints only the
union of the previous and current painted bounds, so the full-screen
translucent window does not re-upload itself every frame. Each block's
rendered image is cached and re-rendered only when its layout (age,
text look, wrap cap), the text colors or the device pixel ratio change.

A long line wraps rather than running off the screen. The English wraps
on word boundaries and the ruby Japanese wraps a whole kanji-plus-reading
segment at a time. `ruby_layout` holds that segment-wrap math as a pure,
Qt-free module (widths arrive through measurer callables), so the layout
is unit-testable without a running `QApplication`. A single segment
wider than the cap is character-broken, with its reading kept on the
first fragment. The browser overlay applies the same cap through a CSS
custom property, so both overlays break at the same point.

**Notices.** A problem notice is its own entry kind in `SubtitleStack`
(`set_notices`, carried in `StackFrame.notices`), not an echo line, and
it does not age. From the anchor up the stack holds the newest line (or
the paused banner when paused with no lines), then the notices (audio,
then translator above it, 8 px apart and 8 px above the newest line or
its banner when that rises past the line), then the echo. A notice has
its own 38 px banner reading お知らせ, ink on amber `#e0a030` for a
problem and paper `#f3eee3` on ink `#1f2421` for a recovery, and its
text sits on the English bar's color in the English face
(`TelopLook.notice_family`, the Watch English family in both modes),
semibold 24 px on a 30 px line, white, never wrapped (elided at the line
cap). `OverlayWindow.set_problems` shows one line per active problem;
`show_recovery` swaps a cleared problem's line for its recovery line for
5 s. A slot whose text changes fades the old entry out while the new one
fades in. Qt fades notices in and out (instant with reduced motion); the
Settings preview never shows them.

## Browser overlay

A FastAPI server (`display/browser_server.py`) broadcasts subtitles
over a WebSocket. Loading `http://localhost:<port>/` in any browser (or
as an OBS Browser Source) renders the same telop design as the Qt
overlay (`display/static/index.html`, `overlay.css`, `overlay.js`): the
anchored plates and banner, the echo, and the Mode, Alignment, Echo and
Banner choices, with HTML `<ruby>` for the furigana and CSS transitions
for the motion. `overlay.js` `applySizes` computes and rounds the pixel
sizes (fonts, furigana and row heights per echo age, paddings) exactly
as `telop_look.py` and `telop_text.py` do, so sizes are points on both
renderers and both draw the same pixels; the numbers are kept in sync
by hand. Position is Qt-only: the page sits at the default anchors, the
stack's left edge at 6vw (Left) or its center at 50vw (Center), with the
newest line's bottom 12vh above the bottom edge, and a centered stack
follows window resizes. Motion follows the Qt rules: a line whose hold
ends fades in place, older and leaving lines rise together when a new
line arrives, and after an unpause with no lines the banner fades out
still gray and reading 一時停止. Reduced motion (the style frame's
`reduce_motion`, or the browser's `prefers-reduced-motion`) switches
lines and plates instantly. Banner text is set as text, never parsed as
HTML. Problem notices show as the same お知らせ lines in a `.notices`
column between the echo and the newest line (CSS `order`); the page
keeps the last problem set across reconnects, fades a notice in and
removes it at once, and sets its text as text. The notices glide with
the lines whenever the stack moves (a line arrives or expires, the
paused banner comes or goes, a notice comes or goes), and a carried
recovery shows once per page load.

The same server also serves the Qt overlay's control bus and the
watchdog health probes, so the WebSocket and `/healthz` endpoints run
even when the browser page is disabled. The default port is 8765
(`overlay.websocket_port`).

### Wire protocol

The WebSocket lives at `/ws`. Messages are JSON objects with a `type`
discriminator. Server-to-client frames:

- `{"type": "subtitle", "jp": "...", "jp_ruby": [...], "en": "...", "t": ...}`,
  one per translated sentence. `jp_ruby` is a list of
  `{text, reading?}` segments, the same shape the Qt overlay consumes;
  `jp` is the flat fallback string.
- `{"type": "style", "jp_font": ..., "jp_size": ..., "en_font": ...,
   "en_size": ..., "text_color": ..., "bg_color": ..., "bg_opacity": ...,
   "padding": ..., "max_width_pct": ..., "max_lines": ..., "hold_ms": ...,
   "fade_in_ms": ..., "fade_out_ms": ..., "mode": "watch"|"learn",
   "alignment": "left"|"center", "echo": "priority"|"both",
   "banner": "jp"|"jp_en", "reduce_motion": bool}`, sent on connect and
   on every settings change (`AppState.style_frame` is the one builder).
   The four choices are `overlay.style` keys; values outside
   `style_defaults.STYLE_CHOICES` fall back to the default.
   `reduce_motion` is `overlay.reduce_motion`, not a style key: the
   pipeline seeds it from config at start, and the tray's `apply_style`
   updates it. `max_width_pct` caps the subtitle box width so the
   browser overlay wraps at the same point the Qt overlay does; the
   browser client clamps it to the same 30 to 100 range the config layer
   enforces and falls back to a 70 percent CSS default for a non-numeric
   value.
- `{"type": "state", "genre": ..., "paused": ..., "series_media_id": ...,
  "series_label": ..., "series_native": ...}`, a snapshot sent on
  connect so a new client doesn't need to wait for the next event, and
  again whenever pause/resume, the genre or the series changes.
  `series_label` is the tray label ("Frieren (2023)"); `series_native` is
  the series' Japanese title for the banner, null when unknown (no pick,
  or a pointer saved before the field existed).
- `{"type": "notice", "title": ..., "body": ..., "settings_page"?: ...}`,
  a user-facing notice the Qt tray shows as a toast (today: a shortcut
  the pipeline couldn't bind, "Shortcut unavailable"). `settings_page`
  (for example `"shortcuts"`) makes a click on the toast open Settings at
  that tab, in front with the keyboard: the click goes to the shell, so
  the tray runs `window_chrome.take_foreground` on Settings, as the series
  bar does on its summon. A notice raised while the Qt overlay isn't
  connected, which is the normal case for a shortcut conflict at startup,
  is queued and sent to the next native client (one without an Origin, the
  Qt overlay) right after its problems frame. A browser client leaves it
  queued, and the browser page ignores notices.
- `{"type": "problems", "active": [{"kind": "translator"|"audio",
  "title": ..., "body": ..., "line": ..., "recovery": ...}, ...],
  "recovered": [{"kind": ..., "recovery": ...}, ...]}`, the pipeline's
  problem set (copy from `problems.py`), sent on connect right after the
  style frame and on every change. Clients keep the last set and diff it
  by kind: a new kind is a problem start, a kind that disappears a
  recovery. `recovered` lists problems a previous pipeline restarted on;
  it rides only connect frames within 30 s of the restarted pipeline's
  start.
- `{"type": "toggle_overlay"}`: show or hide the subtitles, sent to every
  client for `Ctrl+Alt+O` and for the tray's `toggle_overlay` command.

Clients send `{"type": "command", "action": ...}` frames:

- `pause`, `resume`.
- `set_genre` with `"genre"`, persisted so it survives a restart.
- `set_style` with a partial `"style"` dict: merged, broadcast as a
  `style` frame and persisted.
- `apply_style` with `"style"` and an optional top-level
  `"reduce_motion"` bool: the Qt tray after a Settings Apply or a flyout
  Mode or Alignment change. Merged and broadcast like `set_style` but not
  persisted (Settings or the tray already wrote the file);
  `reduce_motion` is runtime only and rides the broadcast style frame to
  the browser page.
- `toggle_overlay`: the tray flyout's Subtitles tile; answered with the
  `toggle_overlay` frame to every client, exactly like `Ctrl+Alt+O`.
  Never persisted.
- `set_series` with `"media_id"`, `"label"` and an optional `"native"`
  (the Japanese title, omitted or empty when unknown): records the
  active-series pointer (`active_series.json`, which keeps `native`) and
  broadcasts a `state` frame. `"media_id": null` (the key present)
  clears the series: the pointer file is deleted and a `state` frame
  without a series goes out. Rejected when the gazetteer feature is off.
- `set_hotkeys` with `"hotkeys"`, a dict of the changed or re-recorded
  `pause_resume`, `toggle_overlay` and `show_series` combos: the Qt tray
  after a Settings Apply that changed or re-recorded shortcuts (kept and
  sent on reconnect while the socket is down). A re-recorded combo is
  sent even when it equals the saved one, so the pipeline retries it.
  Not persisted (Settings already wrote the file); the pipeline re-binds
  the shortcuts at once (`PipelineApp.apply_hotkeys`).
  The whole payload is rejected unless every key is a known shortcut and
  every combo is bindable. The tray's reconnect resend adds
  `"replay": true`: the pipeline then announces a taken shortcut only
  when its combo changed.
- `set_incremental` with an `"enabled"` bool: the Qt tray after a
  Settings Apply that changed Faster subtitles (`asr.incremental.enabled`;
  kept and sent on reconnect while the socket is down). Not persisted
  (Settings already wrote the file); the ASR stage uses it from its next
  speech window. Rejected unless `"enabled"` is a bool.
- `quit`.

### Connection auth

The WebSocket accepts commands that write config and shut the app down,
and it carries every subtitle, so every connection must present the
per-launch session token as `/ws?t=<token>`. The pipeline generates the
token at startup. The Qt overlay receives it in its `YAMABIKO_WS_TOKEN`
environment variable; the browser page reads it from the same-origin
`GET /ws-token` route, which a cross-origin local page cannot read
because no CORS headers are sent. Native clients are not exempt:
127.0.0.1 is shared by every logged-in Windows session. A browser
connection must also come from a loopback origin on exactly the bound
port. Style values are sanitized server-side before broadcast: a string
carrying CSS-injection or control characters is dropped rather than
interpolated into a CSS custom property, and a font family that is not a
non-empty string falls back to the shipped default.

### Auto-reconnect

The browser client reconnects with exponential backoff: the delay
starts at 1 s and doubles each attempt (1, 2, 4, 8, 16 s) up to a 30 s
cap, resetting on a successful connect. There is no replay buffer, so
subtitles produced while the socket was down are not redelivered; new
ones arrive as they're produced.

## Chrome theme

The settings window, the tray flyout and its menus, the series bar and
the startup error window share one themed chrome.
`overlay_app/theme.py` defines two
token palettes from the approved app look: **Mist** (`mist`, light)
and **Night** (`night`, dark). Painted pieces that later surfaces share
(`landscape.py`, `segmented.py`, `keycaps.py`) take the same palette
dicts; `seal.py` paints the fixed vermilion seal in both themes. The
`overlay.theme` config key selects between them: `dark` and `light`
force a palette, while `system` follows the Windows app light/dark
setting.
`resolve_mode` reads the config value first and only consults the OS
when it is `system`; the probe prefers the running `QApplication` colour
scheme (Qt 6.5+) and falls back to the `AppsUseLightTheme` registry
value, defaulting to dark on any error or non-Windows host. When the
config is `system`, the app also re-themes live if the user flips the OS
setting while it is running. The startup error window reads
`overlay.theme` once when it opens, straight from `user_overrides.yaml`
(the full config load can be what failed), and keeps that palette.
The theme only paints chrome; the subtitle overlay's own colors stay
under the user's `overlay.style` control.

## Tray

`overlay_app/tray.py` is the controller. Left click (`QSystemTrayIcon`
Trigger) opens `TrayFlyout` (`tray_flyout.py`): a 360 × 426 frameless
`Qt.Tool` panel (358 tall without the series feature) in the app look,
with DWM's small corners, shadow and frame-tone border, built once at
boot and placed by `tray_placement.place_flyout` over the icon, 12 px
above the taskbar and clamped into the work area. It hides on
deactivation and Esc; a click on the icon hides it on mouse-down, so
`TriggerDebounce` swallows that click's Trigger (a double click always
ends open, a right click clears the swallow). Its controls send the
existing commands (pause/resume, `set_genre`, `set_series` with a null
media id for Clear, the Position presets through
`OverlayWindow.set_position`); Mode and Alignment write
`overlay.style.mode` / `.alignment` and go through the same
`apply_style` path as a Settings Apply. An open Settings window adopts a
flyout Mode, Alignment or Position change and a placement save, so its
next Apply does not undo them. With the socket down, Clear deletes
`active_series.json` itself and queues the clear for the reconnect, like
an offline pick. Only a pick from the Position picker's open list
applies: the wheel does nothing over it, and the arrow keys or typing
open the list. The open list marks its current row Lit and stays inside
the work area: Qt places it against the whole screen, so when it would
run over the taskbar it opens above the picker. The Subtitles tile sends
`toggle_overlay` (with the socket down it toggles the window itself); the
tile shows the Qt window's real visibility
(`OverlayWindow.visibleChanged`). Pause/Resume,
the tiles and the segments keep the flyout open; every other action
closes it. Right click shows the short menu on the taskbar's line
(`tray_placement.place_menu`: above a bottom taskbar, at the cursor when
it is on the icon, else at the icon, inside the work area); its shortcut
column and the flyout's tooltips show the user's current combos (none
while `hotkeys.enabled` is false). `theme.menu_qss` styles both menus
(the short menu and the flyout's Troubleshooting menu) with Lit
selection; `window_chrome.SquareCorners` keeps their corners square
against the windows11 style's rounding.

`tray_icons.py` paints the icon for each state (listening; paused, which
wins over hidden; subtitles hidden) and the amber problem dot with a
1 px ring in the taskbar color (`SystemUsesLightTheme`, re-read on
`WM_SETTINGCHANGE` "ImmersiveColorSet"), at the tray screen's
`GetSystemMetricsForDpi(SM_CXSMICON)` size snapped down to a drawn grid
(16, 20, 24, 28 or 32 px; 225% and up draw 32), each on its own pixel
grid, and hands Qt only that one pixmap, so from 100% to 200% Windows
shows it without resampling. The icon is re-rendered on a screen or DPI
change and set only when what it shows changes. The tooltip is the
status sentence plus "Subtitles hidden" and each problem title, kept
within Windows' 127 characters by shortening the series title. Every
notification goes through `Tray.show_toast` with the shipped app icon
(`assets/icon.ico`, no dot); `messageClicked` does not say which
notification was clicked, so only the newest one's handler stays
connected. The diagnostics notification opens Explorer with the zip
selected (`lifecycle.jobs.show_in_folder_outside_job`, outside the Job
Objects). `problem_view.ProblemTracker` turns `problems` frames into
`changed` (the dot, the tooltip, the flyout status, the overlay's notice
lines), then `recovered` (the recovery lines), then `started` (one
notification each, whose click opens the flyout).

## Series bar

`overlay_app/series_palette.py` (`SeriesPalette`) is the series search,
the "Spotlight bar" of the redesign (spec Surface 4); its painted pieces
(rows, section header, messages, key line, chip, status text and the
line along its bottom edge) are in `series_palette_parts.py`. It is a
640 px wide frameless `Qt.Tool` window in the app look, 48 px tall for the
bar alone and growing downward with its dropdown, with DWM's small
corners, shadow and frame-tone border. `build_overlay` builds it once at
boot and renders it once off screen (`prewarm`), so the first summon
paints at once. Three routes summon it: `Ctrl+Alt+S` (the pipeline's
hotkey broadcasts `show_series_palette`), the short menu's Pick a
series… and the flyout's Change…. Every summon places it at the top
center of the subtitle monitor's work area, 24 px down
(`OverlayWindow.subtitle_screen`, the screen `overlay.position.screen`
pins), keeps it cloaked from DWM until its first paint so no empty outline
flashes, and gives it the foreground: the hotkey reaches the pipeline's
low-level keyboard hook, not the overlay process, so Windows would refuse
the overlay `SetForegroundWindow`; `window_chrome.take_foreground` first
sends one empty mouse input (no move, no button), which makes the overlay
process the last input source. If Windows still refuses (an elevated
window in front may block the input; untested), the bar shows without the
keyboard and `main-overlay.log` says so. The summon also keeps the window
that had the foreground: another app's (not the taskbar or the tray
overflow, which the short menu's route leaves in front), or Settings or a
dialog of the overlay process. If the foreground is on the bar itself or on
the click-through subtitle window as the bar starts to hide (Esc, the
slide-away, a failure), `window_chrome.hand_back_foreground` gives the
foreground back to the kept window while it still shows
(`window_chrome.window_shows`: not closed, hidden or minimized), whatever
Windows picked, so the video player (or Settings) gets the keyboard again;
it sends the same empty mouse input first, because by then another process
(the overflow) may hold the foreground. With nothing kept (a summon from the
tray menu, the overflow or the flyout), or a kept window that no longer
shows, Windows' own pick of an ordinary window stands; otherwise the
foreground goes to the topmost ordinary window of any process
(`window_chrome.topmost_app_window`: Settings counts; tool, non-activating
and click-through windows do not), which also closes an open tray overflow.
A window the user activated (Settings, the flyout, a menu, another app)
keeps the foreground. The bar never touches the overlay window's
one-row-short geometry.

The layout follows the approved mockup: the field starts at x 48 so its
text begins at the mockup's x 50 (`QLineEdit` draws its text 2 px inside
its edge), row contents sit inside the mockup's 1.5 px row border (cover
at +9.5, text at +53.5), covers snap to whole device pixels, and the 2 px
focus underline and the progress line are drawn at y 45 to 46, so the
DWM border, which covers the window's outermost pixel, never hides them
on the 48 px bar.

With the search empty it shows the user's AniList Currently watching list
(`gazetteer.anilist_username`; the six most recently updated entries),
else up to six search results; no matches, an AniList failure (with Try
again) and a private list each have their own message. Their key line is
"Esc closes", except the failure's: "Enter tries again, Esc closes". In
its search face it hides on Esc or when another window activates. A
pick emits `series_chosen`; the tray builds the gazetteer
(`_GazetteerBuildThread`) and drives the bar's status: the build's staged
progress (`progress(media_id, fraction)`: the intel request, the cast
pages, the related seasons) fills the line, success shows "Now watching
…" for 2 s before the bar slides up and fades, and failure hides it at
once beside the "Series not changed" notification. When a monitor sits
directly above the subtitle monitor, the bar fades in place instead of
sliding up. "Above" is decided in native pixels: Qt on Windows keeps each
screen's native top-left and divides only its size by the scale, so a
logical point just above a screen can miss a scaled monitor above it.
A newer pick or a Clear supersedes the status; deactivation does not
hide it. The chip shows the active series with its cover, and its Clear
takes the flyout's Clear path.

The active series' network poster is saved once per session as
`cover-<media_id>.png` beside its gazetteer cache
(`gazetteer.cover_path`, written through `QSaveFile`): when the pick's
build succeeds, or when the poster arrives later. `cover_for` reads the
file back before the AniList color, so a restored series shows its
cover in the chip and the flyout at boot, offline too; a poster read
from the file is never written back, and Clear keeps the file. Reduce
motion switches off the unfold, fold, slide and fill animations; the
theme follows `Tray.apply_palette`, including the field's right-click menu
(`menu_qss`, square corners and no icons, as the tray's menus) and the chip's tooltip
(`tooltip_qss`, the rule the flyout's tooltips get through `window_qss`).

## Settings window

`overlay_app/settings_window.py` defines `SettingsWindow`, a non-modal
`QWidget` the tray shows and keeps to a single instance: "Studio under
the mountain" (spec
`docs/superpowers/specs/2026-10-27-ui-redesign-design.md`, Surface 2).
It opens at 1000 × 728 and paints under its own title bar
(`window_chrome.py`): Qt's `ExpandedClientAreaHint` runs the client area
under the caption, and Yamabiko paints the app mark, the title and the
caption buttons on the sky of the title band. Dragging an empty part of
the band moves the window, and a double-click maximizes it. From top to
bottom:

- **Title band** (`TitleBand`, 80 px, static): the landscape ridgelines
  with echo arcs, "山彦 Yamabiko settings" in BIZ UDPMincho Bold with the
  seal, and Theme (System / Light / Dark) as a text-style choice at the
  right.
- **Preview band**, the same on every tab: the preview monitor and,
  beside it, the **Overlay** switcher: Mode (Watch / Learn, "English
  priority" / "Japanese priority"), Alignment (Left / Center), Echo
  (Priority only / Both languages, with the help "Which language older
  lines keep as they fade.") and Banner (日本語 / 日本語 + English) as Lit
  segmented rows.
- **Tabs**: BIZ UDPMincho text tabs (the active one with an accent bar)
  over a grid of panels aligned to the band, two under the monitor and
  one under the switcher:
  - **Subtitles 字幕**: Fonts (Japanese and English font and size),
    Colors (text color, background, background opacity in percent,
    padding, max width), and Series (AniList username and the "Switch
    genre to Anime when you pick a series" switch) when the series
    feature is on.
  - **Position 位置**: Screen (monitor, vertical position) and Placement
    ("Move subtitles…", which enters on-screen placement mode).
  - **Timing 間**: Lines (lines on screen, time on screen in seconds),
    Fades (fade in, fade out) over Motion (Reduce motion), and Speed (the
    "Faster subtitles" switch, `asr.incremental.enabled`, with its help
    line).
  - **Audio & model 音声**: Connections (audio device, translator
    endpoint) and Translation model (the two tiers as a stacked list). A
    tier change downloads and validates the selected model before
    committing the new setting. Cancellation or failure leaves the
    current tier unchanged.
  - **Shortcuts 操作**: one Keyboard shortcuts panel (see Shortcuts below).
- **Footer**: Reset to defaults as an underlined text button at the left;
  Cancel, Apply and OK at the right.

At 1000 × 728, and on any work area that holds it, nothing scrolls and
the layout follows the design's height budget (title band 80, preview
band 288, tabs 40, panels 266, footer 56). A shorter work area (1920 ×
1080 at 150% leaves 672 px) is the one exception: on first show the
window fits the work area, frame included. `resizeEvent` takes the
missing height from the preview monitor first
(`PreviewMonitor.fit_height`, ratio kept, down to 200 px); every other
row has a fixed height, so the panel area (`_PanelScroll`, a frameless
scroll area around the tab pages) gets what is left and shows a scroll
bar only once the preview has shrunk as far as it can, and only for a
tab whose panels don't fit. The title band, the tabs and the footer
always stay whole; the window never gets shorter than 600 px. A wheel
turn over a spin box, combo box or slider the user hasn't clicked or
tabbed into scrolls the panels instead of changing its value; a focused
field keeps its wheel.

The preview monitor (`preview_monitor.py`) is a 1:1 crop of a screen's
lower-left corner: a night street photo
(`overlay_app/assets/preview-night.jpg`, "Nakasu Yatai Stalls" by
Yoshikazu TAKADA, CC BY 2.0) under a real `SubtitleStack` at the
overlay's real sizes, anchored 28 px from the left and 22 px from the
bottom (centered with Center alignment). It shows an echo line above the
newest line, with the plates and the current genre or series banner,
never animated, restyled as controls change. Lines on screen decides
whether the echo line shows (at 1 the newest line shows alone). A line
wraps at the overlay's real cap (screen width × `max_width_pct`),
bounded by the room the crop leaves.

The preview is **dual**: any style-control change also emits a debounced
`stylePreview` signal that the tray mirrors onto the real overlay, so the
user sees the effect on the actual subtitle surface, not just the sample.
Closing the window emits `previewEnded`, and the tray drops the sample
and restores the committed style. Applying emits `styleChanged` with the
full style dict (the same contract the old settings dialog used, plus
`max_width_pct` and the four Overlay choices). The Theme choice re-skins
the window at once but is saved by Apply / OK like every other setting,
which then emits `themeChanged`.

The painted controls (`controls.py`: switch, text buttons, text tabs,
panels, color field, fields with drawn chevrons, progress line) take the
theme palette and show a focus ring for keyboard focus only. Enter on a
focused push button, text button or color field clicks it; Enter
anywhere else means OK, and Esc means Cancel.

**Shortcuts.** Three rows: Pause and resume, Show or hide subtitles, and
Pick a series (only when the series feature is on). Each row has a
click-to-record field (`shortcut_field.py`) drawn as key caps and a
quiet Reset back to the shipped default. Recording takes one main key
(A to Z, 0 to 9, F1 to F24) with Ctrl, Alt or Win, or an F-key alone or
with Shift, stored as "Ctrl+Alt+Shift+Win+P"; Esc cancels. Windows hands
a combo another app holds with `RegisterHotKey` to that app, so on
Windows a recording field also takes the completing key in a low-level
keyboard hook: such a combo can still be recorded, and neither app's
shortcut fires while recording. A combo another app holds
(`hotkeys.combo_taken`, checked at open and after each recording) gets
an amber outline and "Used by another app. Pick a different one."; that
warning is advisory: Apply saves what the field shows. When two rows
share a combo, the row that comes first on the page keeps it and each
later row gets "Already used for {row}." (for example "Already used for
Pause and resume."); while any duplicate remains, Apply and OK are
disabled and Enter saves nothing either. A saved combo Settings can't
record (a hand edit) shows as saved and is kept by an Apply that
doesn't touch its row; one turned off by hand (an empty
`hotkeys.<key>`) shows an empty field, and recording or Reset turns it
back on. Apply / OK saves changed combos and emits `shortcutsChanged`
with them and every re-recorded row, even one recorded with its saved
combo, so a taken shortcut gets another try; the tray sends
`set_hotkeys`, and the pipeline re-binds at once.

**Dialogs.** Three compact themed dialogs (`window_chrome.CompactDialog`,
with the same painted caption) replace the stock ones:

- "Couldn't save your settings" (`settings_dialogs.SaveFailedDialog`):
  the file Yamabiko couldn't write (the user's `%LOCALAPPDATA%` folder
  written as the variable, so no user name shows), Copy details, Cancel
  and Retry. Retry writes again and Apply / OK continue on success;
  Cancel keeps the window open with nothing saved.
- The color picker (`settings_dialogs.ColorPickerDialog`, in place of
  `QColorDialog`): a 6 × 4 swatch grid whose top row holds the overlay's
  own colors, a hex field (OK is disabled while it is invalid), the
  current color, and a sample line in the text color on the background
  color.
- The model download (`model_fetch.QuantDownloadDialog`): "Downloading
  the Smaller model", the amount ("1.7 of 4 GB") and the time left over a
  progress line, and Cancel. While an earlier download still drains it
  reads "Waiting for the other download to finish…". A failure keeps the
  dialog open with Copy details and Retry, which restarts the download in
  place; Cancel after a failure keeps the selection, so the next Apply
  retries.

**Read-only.** Started with `--config`, every control is disabled, the
footer shows the read-only notice where Reset to defaults sits, and
Cancel reads Close. The tabs still switch pages.

### Live-apply flow

Apply / OK persists the change and pushes it live:

1. Writes `user_overrides.yaml` via the shared `config_writer` helper:
   tempfile plus flush/fsync plus `os.replace`, so a process kill or
   power loss mid-save never leaves a half-written or truncated config.
   The writer is sparse: it stores only values that differ from the
   shipped defaults, prunes a key (and any now-empty parents) when it
   returns to its default, and deletes the file entirely once it's
   empty. The read-modify-write cycle is serialized with a cross-process
   lock, because the overlay process and the pipeline process can both
   write the file on a single Apply.
2. Calls `OverlayWindow.apply_style(style)` in-process, which hands the
   stack the new look and re-places it with motion, so a Mode or
   Alignment change glides. A changed Reduce motion setting emits
   `reduceMotionChanged`; the tray calls
   `OverlayWindow.set_reduce_motion` and `SeriesPalette.set_reduce_motion`
   and sends the value with every `apply_style`, so the browser page
   follows it too.
3. Broadcasts a `style` message on the WebSocket, so every connected
   browser overlay restyles live too.

When the WebSocket is down (the pipeline is still starting, or inside
the reconnect backoff), the tray applies the style locally, persists it
directly, and queues it for a flush to the pipeline on reconnect, so a
click is never a silent no-op.

Cancel discards the working copy. Reset to defaults loads the ship
defaults from `style_defaults.py` (the style, the Overlay choices and
the timings), the shipped translator endpoint and the shipped Faster
subtitles switch (`asr.incremental.enabled`) into the working copy;
each shortcut row has its own Reset. The user still has to click Apply
to commit, so there are no surprise mutations.

Config is layered: a bundled `default_config.yaml` holds every shipping
default, and `user_overrides.yaml` holds only the user's deltas. The
first-run wizard writes overrides only for choices that differ from the
defaults; it does not copy the defaults file. The overlay's first paint
therefore reads its style straight from the bundled defaults and matches the
source-checkout appearance.

## What this is not

- **Not a video compositor.** The overlay draws on top of whatever the
  desktop is showing; it doesn't intercept, re-encode, or compose video.
- **Not platform-portable yet.** DWM-specific calls, WASAPI loopback,
  and the `keyboard` hotkey library pin the current build to Windows 11.
