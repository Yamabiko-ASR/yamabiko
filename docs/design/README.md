# Design notes

Subsystem-level design docs for Yamabiko. Each doc covers what the
subsystem does, the decisions behind the current approach, and the
tradeoffs worth knowing if you're changing it.

- [romanization-and-glossary.md](romanization-and-glossary.md):
  proper-noun extraction (fugashi + UniDic, Sudachi split-mode C),
  stacked-furigana rendering, Hepburn romaji, the three-layer
  glossary (user / genre / seed), and the AniList series gazetteer.
- [sentence-boundaries.md](sentence-boundaries.md): Silero VAD as an
  audio gate, fast-bunkai as a multi-sentence splitter, the
  partial-sentence buffer, and the non-JA drop.
- [overlay.md](overlay.md): Qt frameless always-on-top overlay,
  Windows 11 DWM chrome suppression, custom-paint stacked furigana,
  browser overlay wire protocol, and the settings live-apply path.
- [packaging-and-operations.md](packaging-and-operations.md): NSIS
  installer, first-run wizard, subprocess watchdog, service reconnect
  logic, error dialogs, log rotation, global hotkeys.

## What's not here

- **Per-feature task breakdowns, test plans, and rollout orderings**
  live in the development history, not the design docs; they rot
  fast and the code is the source of truth.
- **User-facing documentation** (install walk-through, troubleshooting
  FAQ, hotkey reference) belongs in the README, not here.
- **The phase-by-phase roadmap** lives in
  [docs/ROADMAP.md](../ROADMAP.md).
