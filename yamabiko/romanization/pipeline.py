"""RomanizationPipeline: glossary + extractor + series gazetteer composed.

Lifecycle per translation:
  pre  -> extract JA name candidates, look each up in hint-precedence order
          user.yaml (authoritative) > active series gazetteer > shipped seed
          > fuzzy gazetteer match (proper nouns only), and return a soft
          prompt-hint string naming the confirmed (jp=en) pairs. The source
          is never mutated, so it is not returned -- the translator sends
          its own jp_text unchanged. The writable learned genre layer is
          NEVER hinted (on upgraded installs it holds stale machine-romanized
          entries the removed learning loop auto-added). Only confirmed
          names are hinted; an unknown candidate is left for the LLM to
          translate freely. No substitution.
  ruby -> split JP text into (text, optional reading) segments for the
          subtitle payload's jp_ruby field.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from yamabiko.gazetteer import fuzzy as _fuzzy

from .extractor_fugashi import ProperNounExtractor, load_denylist
from .extractor_sudachi import SudachiExtractor
from .glossary import GlossaryManager
from .romanizer import Romanizer, contains_kanji, is_kanji_run_mark, split_reading

if TYPE_CHECKING:
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer


@dataclass
class RubySegment:
    text: str
    reading: str | None = None


def _flag_spans(flags: list[bool], start: int, end: int) -> list[tuple[int, int, bool]]:
    """Cut [start, end) into maximal (s, e, flag) spans of equal flag."""
    spans: list[tuple[int, int, bool]] = []
    s = start
    for i in range(start + 1, end + 1):
        if i == end or flags[i] != flags[s]:
            spans.append((s, i, flags[s]))
            s = i
    return spans


class RomanizationPipeline:
    def __init__(
        self,
        glossary: GlossaryManager,
        romanizer: Romanizer,
        extractor: ProperNounExtractor | None = None,
        sudachi: SudachiExtractor | None = None,
        fuzzy_enabled: bool = False,
        fuzzy_max_distance: float = 0.25,
    ):
        self._glossary = glossary
        self._romanizer = romanizer
        # Fuzzy katakana recovery (gazetteer v1.5): OFF by default so a
        # pipeline built without the config (unit tests, embedding) is
        # byte-identical to exact-only. main.build_pipeline passes the shipped
        # gazetteer.fuzzy_enabled / fuzzy_max_distance.
        self._fuzzy_enabled = bool(fuzzy_enabled)
        self._fuzzy_max_distance = float(fuzzy_max_distance)
        # The gazetteer trio (gazetteer, its media_id, and the series_epoch
        # it was applied under) is NEVER set at construction -- set_gazetteer
        # is the single write path, so no caller can leave the trio half-set.
        # Starts empty: no active series until the boot restore or a palette
        # selection calls set_gazetteer.
        self._gazetteer: "SeriesGazetteer | None" = None
        self._gazetteer_media_id: int | None = None
        # Echo of AppState.series_epoch the loaded gazetteer was applied under.
        # The hot loop reloads when state.series_epoch differs from this.
        self._gazetteer_epoch: int = 0
        denylist_path = Path(__file__).parent.parent / "glossaries" / "_denylist.txt"
        if extractor is None:
            extractor = ProperNounExtractor(
                tagger=romanizer.tagger,
                denylist=load_denylist(denylist_path),
            )
        self._extractor = extractor
        if sudachi is None:
            sudachi = SudachiExtractor(denylist=load_denylist(denylist_path))
        self._sudachi = sudachi

    def set_gazetteer(
        self,
        gazetteer: "SeriesGazetteer | None",
        media_id: int | None,
        epoch: int = 0,
    ) -> None:
        """Swap the active series gazetteer. The ONLY write path (the ctor
        takes none): gazetteer, media_id, and epoch always move together,
        so no caller can leave the trio half-set.

        The stored gazetteer also maps the family and given parts of each
        all-kanji character name (SeriesGazetteer.with_name_parts, using
        the romanizer's tagger), so 竈門 and 炭治郎 are hinted on their own.
        Every load path (boot restore, hot swap) goes through here.

        Passing gazetteer=None clears hints (better none than wrong-series
        names). epoch echoes AppState.series_epoch; it is recorded even on
        a failed load so the hot loop does not re-attempt every sentence.
        """
        if gazetteer is not None:
            gazetteer = gazetteer.with_name_parts(self._romanizer.tagger)
        self._gazetteer = gazetteer
        self._gazetteer_media_id = media_id
        self._gazetteer_epoch = epoch

    @property
    def tagger(self):
        """The shared fugashi / UniDic tagger (owned by the romanizer).

        Exposed so the translator's polarity guard can reuse the tagger that is
        already loaded for romanization instead of loading UniDic a second time
        per process. Both consumers run on the single translation_stage loop, so
        the shared tagger is never touched concurrently.
        """
        return self._romanizer.tagger

    def current_series_id(self) -> int | None:
        return self._gazetteer_media_id

    def applied_series_epoch(self) -> int:
        return self._gazetteer_epoch

    # ---- pre-translation ----

    def process_pre(self, jp_text: str) -> str:
        """Find JA name candidates, look each up in hint-precedence order
        user.yaml (authoritative) > active series gazetteer EXACT > shipped
        seed EXACT > active series gazetteer FUZZY, and return the soft
        glossary-hint string. The source is never mutated, so it is not
        returned -- the translator sends its own jp_text unchanged.

        The FUZZY rung (gazetteer v1.5, only when fuzzy_enabled) recovers a
        name whose katakana surface ASR garbled just enough to miss the exact
        lookup (ライラン->Lilamon). It runs LAST and only for proper nouns
        (the fugashi 固有名詞 subset plus every Sudachi candidate): any exact
        hit is unambiguous, and a common word or a seed place name one edit
        from a character name (ライター->Heiter, アメリカ->Amelia) must not
        be rewritten into that name. Deliberately strict -- a wrong-name hint
        steers the model worse than none. Kanji candidates are compared via
        their katakana reading (rare; keeps cross-kanji rewrites like
        牧場様->マキマ out).

        The writable learned genre layer is NEVER consulted for hints: on
        upgraded installs it holds machine-romanized entries the removed
        learning loop auto-added (incl. macron-corrupted forms), so hinting
        them as exact English names would corrupt translations. build_ruby
        still uses lookup() (all layers) for readings.

        Only CONFIRMED (jp=en) pairs are hinted; an unknown candidate is left
        for the LLM to translate freely, so a common word can never be forced
        into a name. The hint is prompt-only.
        """
        if self._gazetteer is None and self._glossary is None:
            return ""
        cands, fugashi_propn = self._extractor.extract_all(jp_text)
        proper_nouns = set(fugashi_propn)
        # Merge Sudachi candidates too: UniDic/fugashi and Sudachi disagree on
        # some katakana brand/name splits, so the union catches more surfaces.
        # Sudachi only returns 固有名詞, so each one is a proper noun.
        for name in self._sudachi.extract(jp_text):
            proper_nouns.add(name)
            if name not in cands:
                cands.append(name)
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for jp in cands:
            if jp in seen:
                continue
            # Hint precedence: user.yaml (authoritative) > series gazetteer
            # EXACT > shipped seed > series gazetteer FUZZY. The writable
            # learned genre layer is never hinted (and the fuzzy layer reads
            # ONLY the gazetteer, never the glossary, so it cannot leak it).
            en: str | None = None
            if self._glossary is not None:
                u = self._glossary.lookup_user(jp)
                en = u.en if u else None
            if en is None and self._gazetteer is not None:
                en = self._gazetteer.lookup(jp)
            if en is None and self._glossary is not None:
                s = self._glossary.lookup_seed(jp)
                en = s.en if s else None
            if (
                en is None
                and self._fuzzy_enabled
                and self._gazetteer is not None
                and jp in proper_nouns
            ):
                # Kanji candidate: compare via its katakana reading. The
                # katakana-only fuzzy index never holds a kanji key, so the
                # raw kanji surface would never match; the reading (e.g.
                # ボクジョウサマ) is what stays honestly far from マキマ.
                surface = jp
                if contains_kanji(jp):
                    surface = _fuzzy.hiragana_to_katakana(
                        self._romanizer.reading(jp)
                    )
                en = self._gazetteer.fuzzy_lookup(
                    surface, max_distance=self._fuzzy_max_distance
                )
            if en:
                seen.add(jp)
                pairs.append((jp, en))
        if not pairs:
            return ""
        body = "; ".join(f"{jp}={en}" for jp, en in pairs)
        return "Use these exact English names for the proper nouns: " + body + "."

    # ---- furigana segmentation ----

    def build_ruby(self, jp_text: str) -> list[RubySegment]:
        """Split JP text into RubySegments; kanji get a hiragana reading.

        The line is tagged once, so each kanji is read inside its word:
        行きました gives 行=い (the verb), where a lone 行 reads as the
        noun こう. The kana a token spells out (okurigana, a leading
        prefix) is cut from its reading and stays plain text, so ruby
        covers only the kanji. When a token's kana cannot be placed
        unambiguously, the whole token gets one ruby rather than a wrong
        per-kanji split.

        Run marks (々ヶ...) extend an open kanji run (佐々木, not 佐+々+木).
        A curated glossary reading for an exact kanji run outranks the
        tagger (the documented user.yaml `reading:` contract). A kanji run
        the tagger splits into several tokens stays plain when any of those
        tokens has no reading (禰豆子: UniDic does not know 禰豆): ruby over
        only 子 would read as the furigana for the whole name.
        Segments always recombine to `jp_text`.
        """
        if not jp_text:
            return []
        n = len(jp_text)
        in_run = [False] * n
        for i, ch in enumerate(jp_text):
            in_run[i] = contains_kanji(ch) or (
                i > 0 and in_run[i - 1] and is_kanji_run_mark(ch)
            )
        runs = _flag_spans(in_run, 0, n)
        # start -> (end, reading); the spans never overlap.
        rubies: dict[int, tuple[int, str]] = {}
        curated = [False] * n
        needs_tagger = False
        for start, end, is_kanji in runs:
            if not is_kanji:
                continue
            entry = self._glossary.lookup(jp_text[start:end])
            if entry is not None and entry.reading:
                rubies[start] = (end, entry.reading)
                curated[start:end] = [True] * (end - start)
            else:
                needs_tagger = True

        if needs_tagger:
            tokens = self._romanizer.token_readings(jp_text)
            # Positions the tagger must not annotate: curated runs, and runs
            # it can read only in part.
            blocked = list(curated)
            for start, end, is_kanji in runs:
                if not is_kanji or curated[start]:
                    continue
                readings = [r for s, e, r in tokens if s < end and e > start]
                if len(readings) > 1 and not all(readings):
                    blocked[start:end] = [True] * (end - start)
            for start, end, reading in tokens:
                if not reading or not any(
                    in_run[i] and not blocked[i] for i in range(start, end)
                ):
                    continue
                pieces = _flag_spans(in_run, start, end)
                split = split_reading(
                    [(jp_text[s:e], is_kanji) for s, e, is_kanji in pieces], reading
                )
                if split is not None:
                    kanji = [(s, e) for s, e, is_kanji in pieces if is_kanji]
                    for (s, e), piece_reading in zip(kanji, split):
                        if not blocked[s]:
                            rubies[s] = (e, piece_reading)
                elif not any(blocked[start:end]):
                    rubies[start] = (end, reading)

        segs: list[RubySegment] = []
        pos = 0
        for start in sorted(rubies):
            end, reading = rubies[start]
            if pos < start:
                segs.append(RubySegment(text=jp_text[pos:start]))
            segs.append(RubySegment(text=jp_text[start:end], reading=reading))
            pos = end
        if pos < n:
            segs.append(RubySegment(text=jp_text[pos:]))
        return segs
