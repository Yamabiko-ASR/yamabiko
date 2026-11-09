# Romanization & Glossary

How Yamabiko produces furigana and proper-noun hints for the translator.

## Goals

- **Furigana** over every kanji run in the JP subtitle line, using correct
  readings for proper nouns (not the default dictionary fallback).
- **Name hints** for the translator: a confirmed English form for a
  proper noun is handed over with the line, so the model keeps the
  speaker's reference instead of inventing a translation.
- **Stable output**: the same proper noun reads the same way across a
  session, even if the underlying dictionary's lemma-level guess drifts.

## NER stack (proper-noun extraction)

Two tokenizers run in parallel and their results are unioned:

- **fugashi + UniDic** (`yamabiko/romanization/extractor_fugashi.py`)
  is the primary. We keep tokens whose UniDic POS chain matches
  `名詞 → 固有名詞 → *` (人名 / 地名 / 組織 / 一般). This rejects
  the common-noun false positives a PROPN/NOUN-confusing tagger
  produces (`日本`-as-普通名詞, `代表`, `昼`, `客` etc.).
- **SudachiPy split-mode C** (`yamabiko/romanization/extractor_sudachi.py`)
  runs as a secondary pass. UniDic fragments long katakana brand names
  into separate tokens (`ソフト` + `バンク`); Sudachi's mode-C
  longest-match keeps them whole and tags them 固有名詞.

The split is deliberate: UniDic is stricter on short tokens and
name-entity sub-categories; Sudachi is better on multi-token katakana
compounds. Their union covers both cases without weakening either filter.

### Guards

Three filters run on both extractors' output:

- **Single-kanji tokens rejected.** UniDic tags `国` / `人` / `月` as
  固有名詞 in some contexts; the single-kanji false-positive rate is
  high enough to skip them wholesale.
- **Short kana-only tokens rejected.** `ワイド` / `ショー` drift through
  固有名詞 tagging in broadcast-news transcripts.
- **Deny list** (`yamabiko/glossaries/_denylist.txt`): plain text,
  one term per line, UTF-8, loaded at startup. A last-resort override for
  specific surfaces the filters miss. Users can edit it; the list is the
  release valve instead of loosening the POS filter.

## Furigana rendering

Qt has no native `<ruby>` and its `QTextDocument` rich-text ruby support
is version-dependent, so the Qt overlay paints stacked furigana itself.
The telop overlay's text module (`yamabiko/overlay_app/telop_text.py`)
lays a line's `{text, reading?}` segments out with
`ruby_layout.layout_rows`, the Qt-free segment-wrap math that keeps a
kanji run and its reading together on one row, and paints each row with
the furigana right above the base glyphs: 0.5 of the Japanese size in
Watch mode (ink at 60% on the paper strip), 0.46 in Learn mode
(vermilion on the paper strip, amber once the line moves into the
echo).

Stacked ruby is the only rendering. There is no parenthetical fallback
(`田中(たなか)さん`) any more; it was legible but visually noisy and
wrecked line-length estimation.

The browser overlay uses real HTML `<ruby>` tags; no custom rendering
needed there.

## Readings and name hints

[cutlet](https://github.com/polm/cutlet) is used only to load its
configured fugashi/UniDic tagger; Yamabiko produces no romaji. Furigana
come from UniDic's kana reading of each token, read in the context of the
whole line. A kanji run the tagger reads only in part (UniDic has no
reading for 禰豆 in 禰豆子) gets no ruby, rather than a reading for part
of the name.

Name hints are looked up in this order: `user.yaml`, the active series
gazetteer, `_seed.yaml`, and finally (with `gazetteer.fuzzy_enabled`) a
strict fuzzy match against the gazetteer that recovers a katakana name
ASR garbled by one sound. The fuzzy match runs only for proper nouns, so
a common word one sound away from a character name (ライター and
ハイター) is never hinted as that name. The English forms are used as
written in those sources; nothing is romanized at runtime.

## Glossary layers

Three priority layers, highest-wins:

1. **`user.yaml`**: user-authored overrides. The user edits this; the
   app never writes to it.
2. **`_seed.yaml`**: curated, read-only, shipped with the app. Covers
   time-stable entries only: the 47 prefectures, major cities, major
   countries, long-standing institutions (`国会`, `NHK`,
   `富士山`). It explicitly excludes political figures, athletes,
   celebrities, and current-event names, which go stale quickly and are
   better handled by the series gazetteer or user overrides.
3. **Genre JSONs** (`variety.json`, `anime.json`, and so on): read-only
   compatibility data retained for upgrades from versions that learned
   readings at runtime. They ship empty, and current releases do not add
   entries. They rank below the seed so a machine-learned reading
   (`日本` as にっぽん) cannot override a curated one (にほん).

The two curated layers (`user.yaml` and `_seed.yaml`) are YAML and share
one loader (`glossary_yaml.load_yaml_glossary`, PyYAML). The legacy genre
layer is JSON. All three carry the same `{surface: {en, reading}}` entry
shape. Current releases do not learn new genre entries, although the manager
may reserialize existing compatibility data during a genre switch or shutdown.
Those writes are atomic, and a corrupt genre file is quarantined rather than
crashing startup.

### Seed authorship

Every seed entry carries a source comment:

```yaml
東京都:
  en: Tōkyō
  reading: とうきょうと  # ref: BGN/PCGN 2024
```

A test guard (`tests/romanization/test_seed_glossary.py`) rejects
hand-authoring typos: every reading must be hiragana-only
(`^[ぁ-ゖー]+$`), and each word in an `en` value is checked for the
unromanized digraphs (`Ou`, `Oo`, `Uu`, `Aa`, `Ee`) at word start that
would signal a missing macron. The same suite pins all 47 prefectures
present and rejects duplicate surfaces.

## Series gazetteer

Character names are the proper nouns that matter most when watching a
show, and the bundled dictionaries do not know them up front. When the
user picks a series (in the series bar, opened with `Ctrl+Alt+S`, the
tray menu's "Pick a series…" or the flyout's "Change…"), Yamabiko
fetches that series' character list from the public AniList GraphQL API
and builds a per-series gazetteer (`yamabiko/gazetteer/`) mapping
Japanese surfaces (native names, kana alternatives, individual name
parts) to the romanized names AniList lists (its `full` name, such as
Tanjirou Kamado), which are not necessarily the spellings an official
English release uses. AniList stores a kanji name without a separator
(竈門炭治郎); when UniDic tags it as a family name followed by a given
name, both parts are added, so 竈門 and 炭治郎 are hinted on their own.

The gazetteer is a soft hint, not a rewrite. Matched names are handed
to the translator as prompt hints alongside the untouched Japanese source.
Keeping enforcement on the prompt side avoids corrupting the source line when
a name match is wrong.

Fetched series are cached on disk, one file per series, with atomic
writes; a corrupt cache is quarantined so re-picking the series
refetches instead of crashing. The build reports its progress in stages
(the series intel request, the picked cast's pages, each related
season's cast), which the series bar shows as its loading line. Beside a
picked series' cache the series bar saves that series' poster
(`cover-<media_id>.png`, written through `QSaveFile`), so a series
restored at boot shows its cover in the tray flyout and the bar, offline
too; the diagnostics zip collects only the logs folder, never these
files. The active pick is remembered in `active_series.json` and
restored at boot. The feature is on by default and controlled by
`gazetteer.enabled`; disabling it removes the series bar, the tray
entries, and the prompt hints. The AniList search, the Currently watching
list, the covers and the character fetch are the network calls for this
feature; they happen when the user opens the series bar, searches, or
picks a series.

## Non-JA guard

ASR occasionally returns English-language passthrough when the source
audio is English (ads mid-stream, English interviewees on JP news), and
sometimes hallucinates a mostly-English line with a stray katakana
fragment when fed English audio under a Japanese language hint. Sending
either through a JP-tuned translator wastes a llama.cpp call and puts
confusing output on screen.

`yamabiko/jp_text.py::is_japanese` runs before translation and accepts a
transcript only when it is predominantly Japanese: it must contain at
least one kanji or kana character (kanji include CJK Extension A, the
basic block, and the compatibility-ideograph block used by name variants
like `山﨑`; kana cover both hiragana and katakana, half-width katakana,
and the iteration marks), and the proportion of Latin letters must stay
below 25 percent of all letters. So a mostly-Japanese line with a few
Latin abbreviations (`AI`, `NHK`, `LDP`) passes, while a full English
sentence carrying one katakana token (`オッケー、I will let him know.`)
is rejected. Rejected segments are dropped and logged at INFO.
