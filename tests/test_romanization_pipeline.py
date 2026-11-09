"""Tests for RomanizationPipeline: gazetteer/glossary soft-hint pre pass
and ruby building.

The pipeline never mutates the source line: `process_pre` returns the soft
glossary-hint string ONLY (the source is unchanged by contract, so it is
not returned) and there is no post pass. Only CONFIRMED (jp=en) pairs
(found in the active series gazetteer or the user-curated glossary) become
a soft prompt hint; unknown candidates are left for the LLM to translate
freely.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from yamabiko.gazetteer.anilist import Character
from yamabiko.gazetteer.gazetteer import SeriesGazetteer
from yamabiko.romanization.glossary import GlossaryManager
from yamabiko.romanization.pipeline import RomanizationPipeline, RubySegment
from yamabiko.romanization.romanizer import Romanizer
from yamabiko.romanization.extractor_fugashi import ProperNounExtractor


@pytest.fixture
def tmp_glossaries(tmp_path: Path) -> Path:
    gdir = tmp_path / "glossaries"
    gdir.mkdir()
    for g in ("variety", "drama", "news", "anime"):
        (gdir / f"{g}.json").write_text(
            json.dumps({"version": 1, "entries": {}}), encoding="utf-8"
        )
    (gdir / "user.yaml").write_text(
        yaml.safe_dump({"entries": {}}, allow_unicode=True), encoding="utf-8"
    )
    return gdir


@pytest.fixture(scope="module")
def _rom() -> Romanizer:
    return Romanizer()


@pytest.fixture(scope="module")
def _ext(_rom: Romanizer) -> ProperNounExtractor:
    return ProperNounExtractor(tagger=_rom.tagger)


@pytest.fixture
def pipeline(tmp_glossaries: Path, _rom: Romanizer, _ext: ProperNounExtractor) -> RomanizationPipeline:
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    return RomanizationPipeline(mgr, _rom, _ext)


def _gaz() -> SeriesGazetteer:
    """A one-name series gazetteer: リムル -> Rimuru."""
    return SeriesGazetteer.from_characters([Character("リムル", "Rimuru", [])])


# ---- process_pre: hint-only string, gazetteer/glossary sourced ----

def test_process_pre_returns_hint_string_only(tmp_glossaries: Path, _rom, _ext):
    """process_pre returns the soft-hint STRING, not a (source, hint) tuple.
    The source is unchanged by contract, so it is no longer returned; the
    translator uses its own jp_text arg directly.

    Seeded defect: revert process_pre to `return jp_text, hint` (the tuple
    vestige) -> the return is a tuple, not a str -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    hint = pipe.process_pre("こんにちは")
    assert isinstance(hint, str), (
        f"process_pre must return the hint string only; got {type(hint).__name__} {hint!r}"
    )


def test_process_pre_hints_gazetteer_name(tmp_glossaries: Path, _rom, _ext):
    """A line name found in the gazetteer becomes a soft (jp=en) hint.

    Seeded defect: drop the gazetteer lookup branch in process_pre -> リムル
    is never confirmed -> 'Rimuru'/'リムル' missing from the hint -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    pipe.set_gazetteer(_gaz(), 1, epoch=1)
    jp = "リムルさんが来た。"
    hint = pipe.process_pre(jp)
    assert "Rimuru" in hint and "リムル" in hint, (
        f"gazetteer name must surface as a soft (jp=en) hint; got {hint!r}"
    )


def test_process_pre_does_not_hint_non_gazetteer_or_common_word(
    tmp_glossaries: Path, _rom, _ext
):
    """A common katakana noun / a name not in the gazetteer produces NO hint,
    so the LLM stays free and a common word can never become a forced name.

    Seeded defect: hint every extracted candidate (drop the lookup gate) ->
    リンゴ appears in the hint -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    pipe.set_gazetteer(_gaz(), 1, epoch=1)
    hint = pipe.process_pre("リンゴを食べた。")
    assert hint == "", (
        f"a candidate with no gazetteer/glossary hit must NOT be hinted; got {hint!r}"
    )


def test_process_pre_empty_hint_when_no_proper_nouns(pipeline: RomanizationPipeline):
    """Noun-free input returns an EMPTY hint.

    The contract is exact: `if not pairs: return ""`. A weaker assertion
    stays green even if the guard is dropped and the dangling template
    'Use these exact English names for the proper nouns: .' leaks into
    every noun-free prompt.

    Seeded defect: remove the `if not pairs` guard in process_pre -> hint ==
    'Use these ...: .' -> red.
    """
    jp = "今日は寒いですね。"
    hint = pipeline.process_pre(jp)
    assert hint == "", (
        f"noun-free input must yield an EMPTY hint (a dangling template leaks "
        f"instructions into the translator prompt); got {hint!r}"
    )


def test_learned_genre_entries_are_never_hinted(tmp_glossaries: Path, _rom, _ext):
    """The writable per-genre JSON is NEVER served as a prompt hint. On
    upgraded v0.1.0-beta installs it holds machine-romanized entries the
    REMOVED learning loop auto-added -- including macron-corrupted forms
    (e.g. 'Gōgle') the deleted heal pass used to repair; hinting those as
    exact English names would actively steer the LLM wrong.

    (Replaces test_process_pre_reuses_existing_entry, which asserted the
    now-prohibited opposite: that a genre-layer entry reaches the hint.)

    Seeded defect: consult the genre layer on the hint path (call lookup()
    instead of lookup_user/lookup_seed in process_pre) -> the learned genre
    entry 'Gōgle' lands in the hint -> red.
    """
    # Seed the writable genre JSON on disk, the way an upgraded install
    # carries entries the removed learning loop auto-added.
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {"グーグル": {"en": "Gōgle"}}},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    # Guard against a vacuous pass: the entry must really be in the genre
    # layer (not user/seed, which have none), so lookup() must surface it.
    assert mgr.lookup("グーグル") is not None and mgr.lookup("グーグル").en == "Gōgle"
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    jp = "グーグルで検索した。"
    hint = pipe.process_pre(jp)
    assert "Gōgle" not in hint, (
        f"a learned genre entry must never be hinted (legacy installs hold "
        f"macron-corrupted machine forms); got {hint!r}"
    )


def test_user_yaml_curated_english_form_is_hinted(tmp_glossaries: Path, _rom, _ext):
    """A user.yaml-curated English form is the one served as the soft hint,
    and the source is returned unchanged. (Renamed from
    test_user_override_beats_auto: the soft-hint design has no 'auto'
    romanization layer for it to beat -- the body only asserts the curated
    user.yaml form reaches the hint.)

    Seeded defect: emit the raw candidate instead of entry.en -> 'Mr. Tanaka'
    absent -> red.
    """
    (tmp_glossaries / "user.yaml").write_text(
        yaml.safe_dump({"entries": {
            "田中": {"en": "Mr. Tanaka", "reading": "たなか"}
        }}, allow_unicode=True),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    jp = "田中さんは先に帰りました。"
    hint = pipe.process_pre(jp)
    assert "Mr. Tanaka" in hint, (
        f"a curated user.yaml form must be the hinted English; got {hint!r}"
    )


def test_user_yaml_override_beats_gazetteer(tmp_glossaries: Path, _rom, _ext):
    """Spec contract: user.yaml is authoritative. An active series gazetteer
    must never silently shadow a hand-curated spelling -- the hint path checks
    user.yaml BEFORE the gazetteer.

    Seeded defect: swap the two lookups so the gazetteer is consulted first ->
    the curated 'Limule' is shadowed by the gazetteer's 'Rimuru' -> red.
    """
    (tmp_glossaries / "user.yaml").write_text(
        yaml.safe_dump({"entries": {"リムル": {"en": "Limule"}}}, allow_unicode=True),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    pipe.set_gazetteer(SeriesGazetteer({"リムル": "Rimuru"}), 1, epoch=1)
    hint = pipe.process_pre("リムルが来た")
    assert "リムル=Limule" in hint, (
        f"user.yaml must win over the gazetteer; got {hint!r}"
    )
    assert "Rimuru" not in hint, (
        f"the gazetteer spelling must not shadow the curated user.yaml form; "
        f"got {hint!r}"
    )


def test_seed_entries_still_hinted_on_gazetteer_miss(
    tmp_glossaries: Path, _rom, _ext, tmp_path: Path
):
    """The shipped seed layer is still hinted when the active gazetteer misses
    the surface: seed is the curated fallback below the series gazetteer (only
    the writable learned genre layer is excluded from hints).

    Seeded defect: drop the seed branch in process_pre (make lookup_seed
    unused) -> 'リムル=Rimuru' absent from the hint -> red.
    """
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(
        yaml.safe_dump({"entries": {"リムル": {"en": "Rimuru"}}}, allow_unicode=True),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="anime", seed_path=seed_path)
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    # A gazetteer that does NOT contain リムル: the seed must fill the gap.
    pipe.set_gazetteer(SeriesGazetteer({"ナントカ": "Nantoka"}), 1, epoch=1)
    hint = pipe.process_pre("リムルが来た")
    assert "リムル=Rimuru" in hint, (
        f"a seed entry must be hinted on a gazetteer miss; got {hint!r}"
    )


def test_process_pre_merges_sudachi_only_candidate(tmp_glossaries: Path, _rom):
    """process_pre unions Sudachi candidates with the fugashi extractor's, so a
    name only Sudachi surfaces (fugashi misses the split) still gets hinted
    from the gazetteer. Forced deterministically: a fugashi stub returns NO
    candidates and a Sudachi stub returns the name, so the hint can ONLY come
    from the Sudachi-merge branch. (Constructing a real fugashi-miss /
    Sudachi-hit input is brittle across dictionary versions, so injected fakes
    pin the branch instead.)

    Seeded defect: delete the `for name in self._sudachi.extract(...)` merge
    loop in process_pre -> the Sudachi-only name never becomes a candidate ->
    no hint -> red.
    """
    class _NoFugashi:
        def extract_all(self, text):
            return ([], set())

    class _OneSudachi:
        def extract(self, text):
            return ["リムル"]

    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = RomanizationPipeline(
        mgr, _rom, extractor=_NoFugashi(), sudachi=_OneSudachi(),
    )
    pipe.set_gazetteer(SeriesGazetteer({"リムル": "Rimuru"}), 1, epoch=1)
    hint = pipe.process_pre("リムルが来た")
    assert "リムル=Rimuru" in hint, (
        f"a Sudachi-only candidate must be merged and hinted from the "
        f"gazetteer; got {hint!r}"
    )


# ---- build_ruby ----

def test_build_ruby_kanji_gets_reading(pipeline: RomanizationPipeline):
    segs = pipeline.build_ruby("田中さん")
    kanji = [s for s in segs if s.text == "田中"]
    assert len(kanji) == 1
    assert kanji[0].reading == "たなか"


def test_build_ruby_kana_no_reading(pipeline: RomanizationPipeline):
    segs = pipeline.build_ruby("さくらさん")
    # All segments should have None for reading (no kanji anywhere).
    for s in segs:
        assert s.reading is None


def test_build_ruby_segments_recombine_to_input(pipeline: RomanizationPipeline):
    text = "田中さんは東京に行きました。"
    segs = pipeline.build_ruby(text)
    assert "".join(s.text for s in segs) == text


def test_build_ruby_empty_input(pipeline: RomanizationPipeline):
    assert pipeline.build_ruby("") == []


def test_build_ruby_reads_verb_kanji_in_context_without_okurigana(
    pipeline: RomanizationPipeline,
):
    """A kanji is read in its word, and the ruby covers only the kanji.

    Tagging a lone kanji reads it as a noun (行 -> こう) and keeps the
    okurigana inside the reading (思う -> 思(おもう)う).
    Seeded defect: tag each bare kanji run instead of the whole line -> red.
    """
    segs = [(s.text, s.reading) for s in pipeline.build_ruby("行きました")]
    assert segs == [("行", "い"), ("きました", None)]
    segs = [(s.text, s.reading) for s in pipeline.build_ruby("そう思う")]
    assert segs == [("そう", None), ("思", "おも"), ("う", None)]


# --- 々/ヶ must stay inside the kanji run so the tagger sees the whole
# word. The old per-char contains_kanji predicate split 佐々木 into
# 佐+々+木 and annotated the fragments with wrong readings (佐=すけ).
# Seeded defect: drop the run-mark branch from build_ruby's predicate.

def test_build_ruby_keeps_iteration_mark_in_sasaki_run(pipeline: RomanizationPipeline):
    segs = pipeline.build_ruby("佐々木さん")
    assert segs[0].text == "佐々木"
    assert segs[0].reading == "ささき"


def test_build_ruby_keeps_iteration_mark_in_hitobito_run(pipeline: RomanizationPipeline):
    segs = pipeline.build_ruby("人々")
    assert [s.text for s in segs] == ["人々"]
    assert segs[0].reading == "ひとびと"


def test_build_ruby_keeps_small_ke_in_kasumigaseki_run(pipeline: RomanizationPipeline):
    segs = pipeline.build_ruby("霞ヶ関")
    assert [s.text for s in segs] == ["霞ヶ関"]
    assert segs[0].reading == "かすみがせき"


def test_build_ruby_leading_iteration_mark_stays_unannotated(pipeline: RomanizationPipeline):
    # 々 with no preceding kanji (ASR noise) must not open a kanji run.
    segs = pipeline.build_ruby("々あ")
    assert "".join(s.text for s in segs) == "々あ"
    assert all(s.reading is None for s in segs)


# --- Curated glossary readings must reach the furigana. user.yaml
# documents `reading: <hiragana-for-furigana>  # used above kanji in JP
# line`, but build_ruby always recomputed via the tagger and no code path
# read GlossaryEntry.reading. Sentinel readings (そらまめ) are deliberately
# distinct from the tagger's output so preference is observable.
# Seeded defect: remove the glossary lookup from build_ruby's flush.

def test_build_ruby_prefers_curated_user_yaml_reading_over_tagger(
    tmp_glossaries: Path, _rom, _ext
):
    (tmp_glossaries / "user.yaml").write_text(
        yaml.safe_dump(
            {"entries": {"田中": {"en": "Tanaka", "reading": "そらまめ"}}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    segs = pipe.build_ruby("田中さん")
    assert segs[0].text == "田中"
    assert segs[0].reading == "そらまめ"


def test_build_ruby_prefers_genre_json_reading_over_tagger(
    tmp_glossaries: Path, _rom, _ext,
):
    """A reading carried in the genre JSON (an upgraded install may hold one)
    outranks the tagger for furigana: build_ruby consults every glossary
    layer via lookup(), not just user.yaml.

    Seeded defect: remove the glossary lookup from build_ruby's kanji flush
    -> the tagger reading たなか wins over the sentinel そらまめ -> red.
    """
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {
            "田中": {"en": "Tanaka", "reading": "そらまめ"}
        }}, ensure_ascii=False),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    segs = pipe.build_ruby("田中さん")
    assert segs[0].reading == "そらまめ"


def test_build_ruby_falls_back_to_tagger_when_glossary_reading_empty(
    tmp_glossaries: Path, _rom, _ext,
):
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {
            "田中": {"en": "Tanaka", "reading": ""}
        }}, ensure_ascii=False),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    pipe = RomanizationPipeline(mgr, _rom, _ext)
    segs = pipe.build_ruby("田中さん")
    assert segs[0].reading == "たなか"


# --- 日本 lives in BOTH _denylist.txt and _seed.yaml. That overlap is a
# DOCUMENTED carve-out, not a conflict: the denylist suppresses the
# surface on the extraction/prompt-hint path (the LLM needs no "日本=Japan"
# hint), while build_ruby looks up every kanji span WITHOUT consulting the
# denylist, so the seed reading にほん still reaches the furigana — and
# matters, because the tagger alone reads 日本 as にっぽん. Any OTHER overlap
# is dead seed data and must fail the data-contract test below.

# Mirrors RomanizationPipeline.__init__'s own denylist-path resolution.
from yamabiko.romanization import pipeline as _pipeline_mod  # noqa: E402

_SHIPPED_GLOSSARIES_DIR = Path(_pipeline_mod.__file__).parent.parent / "glossaries"
_SHIPPED_SEED = _SHIPPED_GLOSSARIES_DIR / "_seed.yaml"
_SHIPPED_DENYLIST = _SHIPPED_GLOSSARIES_DIR / "_denylist.txt"
_DOCUMENTED_CARVE_OUTS = {"日本"}


def test_seed_denylist_overlap_is_exactly_the_documented_carve_out():
    """Data contract: a seed surface that is also denylisted can never
    produce a prompt hint (extraction drops it first), so its `en:` field
    is dead data. 日本 is the one deliberate exception (kept for its
    curated furigana reading; see the comments in both files).

    Seeded defect: add any denylisted surface (e.g. 政府) to _seed.yaml,
    or add a seed surface to _denylist.txt -> overlap != carve-out -> red.
    Also red if the carve-out itself is removed from either file without
    updating this contract.
    """
    from yamabiko.romanization.extractor_fugashi import load_denylist
    denylist = load_denylist(_SHIPPED_DENYLIST)
    seed = yaml.safe_load(_SHIPPED_SEED.read_text(encoding="utf-8"))["entries"]
    overlap = denylist & set(seed)
    assert overlap == _DOCUMENTED_CARVE_OUTS, (
        f"seed/denylist overlap {overlap!r} != documented carve-outs "
        f"{_DOCUMENTED_CARVE_OUTS!r}; a denylisted seed surface's en: field "
        f"is unreachable dead data — remove one side or document the "
        f"carve-out here and in both data files"
    )


@pytest.fixture
def shipped_data_pipeline(tmp_glossaries: Path, _rom: Romanizer) -> RomanizationPipeline:
    """Production-shaped pipeline: REAL shipped seed + REAL shipped
    denylist (extractor/sudachi built by RomanizationPipeline itself,
    exactly as in yamabiko.main), with a writable tmp genre dir."""
    mgr = GlossaryManager(tmp_glossaries, genre="news", seed_path=_SHIPPED_SEED)
    return RomanizationPipeline(mgr, _rom)


def test_denylisted_seed_surface_emits_no_prompt_hint(shipped_data_pipeline):
    """日本 is denylisted in BOTH extractors, so it never becomes a candidate
    and its seed en: Japan never lands in a prompt hint.

    Seeded defect: remove 日本 from _denylist.txt -> both extractors emit it
    (verified live: fugashi AND sudachi tag 日本 as 固有名詞) -> the seed
    entry's en: Japan lands in the hint -> red.
    """
    jp = "日本が発表した。"
    hint = shipped_data_pipeline.process_pre(jp)
    assert "日本" not in hint, f"denylisted surface must not reach the hint; got {hint!r}"
    assert "Japan" not in hint, f"denylisted seed's en must not reach the hint; got {hint!r}"


def test_denylisted_seed_surface_keeps_curated_furigana_reading(shipped_data_pipeline):
    """The carve-out is load-bearing: build_ruby bypasses the denylist and
    must serve the seed's curated reading にほん, not the tagger's にっぽん.

    Seeded defect: delete the 日本 entry from _seed.yaml -> the tagger
    fallback annotates にっぽん -> red."""
    segs = shipped_data_pipeline.build_ruby("日本")
    assert segs[0].text == "日本"
    assert segs[0].reading == "にほん"


# ---- set_gazetteer: runtime series-gazetteer swap ----

def test_set_gazetteer_swaps_active_hints_and_records_id(pipeline: RomanizationPipeline):
    """set_gazetteer must both swap the active gazetteer (so process_pre's
    hints change immediately) and record the series media_id for the hot
    loop's change-detection.

    `pipeline` is built with gazetteer=None and an empty tmp glossary (no
    リムル entry), so hint0 is empty. Seeded defect: if set_gazetteer didn't
    reassign self._gazetteer, hint1 would stay empty -> red. If it didn't
    store media_id, current_series_id() would stay None/stale -> red.
    """
    from yamabiko.gazetteer.gazetteer import SeriesGazetteer

    assert pipeline.current_series_id() is None
    # No gazetteer, no glossary entry -> no hint for the name.
    hint0 = pipeline.process_pre("リムルが来た")
    assert "リムル" not in hint0

    # Swap in a gazetteer -> the same name is now hinted.
    pipeline.set_gazetteer(SeriesGazetteer({"リムル": "Rimuru"}), 182205)
    assert pipeline.current_series_id() == 182205
    hint1 = pipeline.process_pre("リムルが来た")
    assert "リムル=Rimuru" in hint1

    # Clearing to None removes hints again but keeps the recorded id.
    pipeline.set_gazetteer(None, 999)
    assert pipeline.current_series_id() == 999
    hint2 = pipeline.process_pre("リムルが来た")
    assert "リムル" not in hint2


# ---- fuzzy katakana matching (gazetteer v1.5) ----
#
# process_pre's hint precedence gains one rung: user.yaml > gazetteer EXACT >
# gazetteer FUZZY > seed. The fuzzy rung recovers ASR-garbled katakana name
# surfaces (on the kotoba-whisper output the fixtures were mined from, the
# exact lookup missed ~95% of lines). Candidates are injected via fakes here
# so the fuzzy branch is pinned deterministically -- the fold-distance
# behaviour itself is covered unit-side in tests/gazetteer/test_fuzzy.py
# against REAL mined garbles.


class _FugashiReturns:
    """A fugashi-extractor stub that yields exactly the given candidates, all
    tagged 固有名詞 (as the real extractor reports a proper-noun hit), so the
    fuzzy rung, which runs only for proper nouns, is reachable."""

    def __init__(self, cands):
        self._cands = list(cands)

    def extract_all(self, text):
        return (list(self._cands), set(self._cands))


class _NoSudachi:
    def extract(self, text):
        return []


def _pipe_with_candidate(mgr, _rom, cand, *, fuzzy_enabled, max_distance=0.25):
    return RomanizationPipeline(
        mgr, _rom,
        extractor=_FugashiReturns([cand]),
        sudachi=_NoSudachi(),
        fuzzy_enabled=fuzzy_enabled,
        fuzzy_max_distance=max_distance,
    )


def test_fuzzy_recovers_garbled_gazetteer_name(tmp_glossaries: Path, _rom):
    """A real kotoba-whisper garble (ライラン for ライラモン, from
    digimon-beatbreak/episode-jp.json t=227.03) is recovered to the exact
    English name and hinted, once fuzzy matching is on.

    Seeded defect: drop the fuzzy branch in process_pre (never call
    gazetteer.fuzzy_lookup) -> ライラン is never confirmed -> 'Lilamon' absent
    -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    gaz = SeriesGazetteer({"ライラモン": "Lilamon", "ムラサメモン": "Murasamemon"})
    pipe = _pipe_with_candidate(mgr, _rom, "ライラン", fuzzy_enabled=True)
    pipe.set_gazetteer(gaz, 188388, epoch=1)
    hint = pipe.process_pre("ライランが現れた。")
    assert "ライラン=Lilamon" in hint, (
        f"a garbled katakana name must be fuzzy-recovered and hinted; got {hint!r}"
    )


def test_fuzzy_disabled_is_byte_identical_exact_only(tmp_glossaries: Path, _rom):
    """With fuzzy_enabled False the pipeline is byte-identical to exact-only:
    the same garble that fuzzy would recover produces NO hint, and an exact
    surface produces the SAME hint either way.

    Seeded defect: make the fuzzy branch ignore the enabled flag (always run)
    -> the disabled garble now yields 'Lilamon' -> the equality below breaks
    -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    gaz_map = {"ライラモン": "Lilamon", "ムラサメモン": "Murasamemon"}

    off = _pipe_with_candidate(mgr, _rom, "ライラン", fuzzy_enabled=False)
    off.set_gazetteer(SeriesGazetteer(dict(gaz_map)), 188388, epoch=1)
    assert off.process_pre("ライランが現れた。") == "", (
        "fuzzy off must leave a garble unhinted (exact-only behaviour)"
    )

    on = _pipe_with_candidate(mgr, _rom, "ライラン", fuzzy_enabled=True)
    on.set_gazetteer(SeriesGazetteer(dict(gaz_map)), 188388, epoch=1)
    assert on.process_pre("ライランが現れた。") != "", "fuzzy on must recover it"

    # An EXACT surface: enabling fuzzy must not change the exact result.
    off_exact = _pipe_with_candidate(mgr, _rom, "ライラモン", fuzzy_enabled=False)
    off_exact.set_gazetteer(SeriesGazetteer(dict(gaz_map)), 188388, epoch=1)
    on_exact = _pipe_with_candidate(mgr, _rom, "ライラモン", fuzzy_enabled=True)
    on_exact.set_gazetteer(SeriesGazetteer(dict(gaz_map)), 188388, epoch=1)
    assert (
        off_exact.process_pre("ライラモンが現れた。")
        == on_exact.process_pre("ライラモンが現れた。")
        == "Use these exact English names for the proper nouns: ライラモン=Lilamon."
    ), "an exact hit must be byte-identical with fuzzy on vs off"


def test_exact_gazetteer_hit_wins_over_ambiguous_fuzzy(tmp_glossaries: Path, _rom):
    """Precedence: EXACT is consulted before FUZZY. ビーム and ビイム fold to
    the same katakana (the ー expands to イ), so the fuzzy layer -- seeing two
    DIFFERENT-EN keys tie -- would REFUSE ビーム. The exact map knows it
    unambiguously, so exact-first still hints Beam.

    Seeded defect: reorder process_pre to run fuzzy before the exact lookup ->
    ビーム trips the ambiguity guard -> no hint -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    gaz = SeriesGazetteer({"ビーム": "Beam", "ビイム": "Beem"})
    pipe = _pipe_with_candidate(mgr, _rom, "ビーム", fuzzy_enabled=True)
    pipe.set_gazetteer(gaz, 171627, epoch=1)
    hint = pipe.process_pre("ビームが来た。")
    assert "ビーム=Beam" in hint, (
        f"an exact gazetteer hit must win before the fuzzy layer runs; got {hint!r}"
    )


def test_user_yaml_beats_fuzzy_gazetteer(tmp_glossaries: Path, _rom):
    """user.yaml stays authoritative on the fuzzy path too: a curated spelling
    for the garbled surface must win over the fuzzy gazetteer match.

    Seeded defect: consult the gazetteer fuzzy layer before user.yaml -> the
    gazetteer's 'Lilamon' shadows the curated 'Custom Name' -> red.
    """
    (tmp_glossaries / "user.yaml").write_text(
        yaml.safe_dump({"entries": {"ライラン": {"en": "Custom Name"}}},
                       allow_unicode=True),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = _pipe_with_candidate(mgr, _rom, "ライラン", fuzzy_enabled=True)
    pipe.set_gazetteer(SeriesGazetteer({"ライラモン": "Lilamon"}), 188388, epoch=1)
    hint = pipe.process_pre("ライランが現れた。")
    assert "ライラン=Custom Name" in hint, f"user.yaml must win over fuzzy; got {hint!r}"
    assert "Lilamon" not in hint, f"fuzzy must not shadow user.yaml; got {hint!r}"


def test_kanji_candidate_does_not_cross_fuzzy_match_a_katakana_name(
    tmp_glossaries: Path, _rom
):
    """The honest negative: 牧場様 (reading ぼくじょうさま) must NOT fuzzy-match
    マキマ (Makima). Cross-kanji rewrites are out of scope for v1.5 -- the
    kanji path compares via the READING (ボクジョウサマ), which is nowhere near
    マキマ, so no hint is produced.

    Seeded defect: feed the raw kanji surface to fuzzy_lookup instead of its
    katakana reading (drop the contains_kanji reading conversion) -> a
    cross-script matcher could snap 牧 -> マキ -> Makima -> red. Or loosen the
    threshold far enough that ボクジョウサマ reaches マキマ -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = _pipe_with_candidate(mgr, _rom, "牧場様", fuzzy_enabled=True)
    pipe.set_gazetteer(SeriesGazetteer({"マキマ": "Makima", "デンジ": "Denji"}), 171627, 1)
    hint = pipe.process_pre("牧場様が来た。")
    assert hint == "", f"a kanji word must not cross-match a katakana name; got {hint!r}"


def test_kanji_candidate_matches_via_its_katakana_reading(tmp_glossaries: Path, _rom):
    """The reading-conversion line is live: a kanji-surfaced candidate (岸辺)
    is compared via its katakana reading (キシベ) and recovers a katakana
    gazetteer key of the same name. This is the load-bearing counterpart to
    the 牧場様 negative -- together they show the reading path fires and stays
    strict.

    Seeded defect: drop the `contains_kanji` reading conversion in process_pre
    (feed the raw kanji surface to fuzzy_lookup) -> 岸辺 is not katakana-
    dominant, the fuzzy gate rejects it, no hint -> 'Kishibe' absent -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = _pipe_with_candidate(mgr, _rom, "岸辺", fuzzy_enabled=True)
    pipe.set_gazetteer(SeriesGazetteer({"キシベ": "Kishibe"}), 171627, epoch=1)
    hint = pipe.process_pre("岸辺が現れた。")
    assert "Kishibe" in hint, (
        f"a kanji candidate must match a katakana key via its reading; got {hint!r}"
    )


def test_learned_genre_never_hinted_even_on_fuzzy_path(tmp_glossaries: Path, _rom):
    """The writable genre layer stays excluded from hints with fuzzy ON: the
    fuzzy layer reads ONLY the series gazetteer, never the glossary genre
    JSON, so a learned 'Gōgle' can never leak even when a gazetteer is active.

    Seeded defect: point the fuzzy lookup (or a new fallback) at the glossary
    genre layer -> 'Gōgle' leaks into the hint -> red.
    """
    # Seed the writable genre JSON on disk (legacy machine-romanized entry).
    (tmp_glossaries / "variety.json").write_text(
        json.dumps({"version": 1, "entries": {"グーグル": {"en": "Gōgle"}}},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="variety")
    # Guard against a vacuous pass: the entry must really be in the genre layer.
    assert mgr.lookup("グーグル") is not None and mgr.lookup("グーグル").en == "Gōgle"
    pipe = _pipe_with_candidate(mgr, _rom, "グーグル", fuzzy_enabled=True)
    # A gazetteer that does NOT contain グーグル so the fuzzy layer is exercised
    # (and still must not fall through to the learned genre layer).
    pipe.set_gazetteer(SeriesGazetteer({"ライラモン": "Lilamon"}), 188388, epoch=1)
    hint = pipe.process_pre("グーグルで検索した。")
    assert "Gōgle" not in hint, f"learned genre must never be hinted; got {hint!r}"


def test_set_gazetteer_records_epoch_and_id(pipeline: RomanizationPipeline):
    """set_gazetteer is the single write path: gazetteer, media_id, and the
    AppState.series_epoch echo all move together. applied_series_epoch()
    surfaces the recorded epoch so the hot loop can gate reloads on epoch
    change (letting a re-pick of the same id reload a freshly built cache).

    Seeded defect: drop `self._gazetteer_epoch = epoch` in set_gazetteer
    (leave applied_series_epoch returning its 0 init) -> applied_series_epoch()
    stays 0 -> red.
    """
    pipe = pipeline
    assert pipe.applied_series_epoch() == 0  # untouched pipeline starts at 0
    pipe.set_gazetteer(_gaz(), 7, epoch=3)
    assert pipe.current_series_id() == 7
    assert pipe.applied_series_epoch() == 3


def test_set_gazetteer_hints_kanji_name_parts(tmp_glossaries: Path, _rom):
    """AniList's native name for a kanji character is one unbroken string
    (竈門炭治郎), but dialogue uses the family or given name alone. After
    set_gazetteer the tagger-split parts are hinted with their own English
    token: family 竈門 -> Kamado, given 炭治郎 -> Tanjirou.

    Seeded defect: drop the name-part derivation from set_gazetteer -> only
    the full-name key exists -> neither part is hinted -> red.
    """
    mgr = GlossaryManager(tmp_glossaries, genre="anime")
    pipe = RomanizationPipeline(
        mgr, _rom, extractor=_FugashiReturns(["炭治郎", "竈門"]), sudachi=_NoSudachi(),
    )
    pipe.set_gazetteer(SeriesGazetteer({"竈門炭治郎": "Tanjirou Kamado"}), 101922, epoch=1)
    hint = pipe.process_pre("炭治郎、竈門家の長男だ。")
    assert "炭治郎=Tanjirou" in hint, hint
    assert "竈門=Kamado" in hint, hint


def test_seed_name_beats_fuzzy_and_common_nouns_get_no_fuzzy_hint(
    tmp_glossaries: Path, _rom, _ext, tmp_path: Path
):
    """The seed's exact entry outranks a fuzzy gazetteer match, and the fuzzy
    layer only runs for proper nouns. アメリカ is one edit from the character
    アメリア but is a seed country (America); ライター (a lighter) is one edit
    from ハイター but is a common noun, so it gets no hint at all.

    Seeded defect: run the fuzzy rung before the seed -> アメリカ=Amelia ->
    red; or fuzzy-match every candidate -> ライター=Heiter -> red.
    """
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(
        yaml.safe_dump({"entries": {"アメリカ": {"en": "America"}}}, allow_unicode=True),
        encoding="utf-8",
    )
    mgr = GlossaryManager(tmp_glossaries, genre="anime", seed_path=seed_path)
    pipe = RomanizationPipeline(mgr, _rom, _ext, fuzzy_enabled=True)
    pipe.set_gazetteer(
        SeriesGazetteer({"アメリア": "Amelia", "ハイター": "Heiter"}), 154587, epoch=1,
    )
    assert pipe.process_pre("アメリカに行った。") == (
        "Use these exact English names for the proper nouns: アメリカ=America."
    )
    assert pipe.process_pre("ライターを貸して。") == ""


def test_build_ruby_gives_no_reading_to_a_run_with_an_unknown_kanji_token(
    pipeline: RomanizationPipeline,
):
    """UniDic has no reading for 禰豆 but reads the trailing 子 as こ. A
    partial reading over 禰豆子 is wrong furigana, so the whole kanji run
    stays plain while the rest of the line keeps its ruby.

    Seeded defect: keep the per-token reading for 子 -> a ruby segment starts
    inside 禰豆子 -> red.
    """
    segs = pipeline.build_ruby("禰豆子、行くぞ")
    assert "".join(s.text for s in segs) == "禰豆子、行くぞ"
    pos = 0
    for s in segs:
        if s.reading:
            assert pos >= len("禰豆子"), f"ruby inside 禰豆子: {segs!r}"
        pos += len(s.text)
    assert RubySegment(text="行", reading="い") in segs
