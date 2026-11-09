"""Fuzzy katakana matching for ASR-garbled proper-noun surfaces (v1.5).

Gazetteer v1 fires only on an EXACT katakana surface, but the ASR model
garbles name surfaces often enough that the exact lookup misses most lines.
`yamabiko.gazetteer.fuzzy` recovers a name when the garble is close enough,
and REFUSES when it is not -- a wrong-name hint steers the translator worse
than no hint at all, so the accept bar is deliberately tight.

Every ACCEPT / REJECT fixture below is a REAL surface mined from the
2026-07-05 eval artifacts, not an invented garble. Provenance is cited per
fixture: `<title>/episode-jp.json` (kotoba-whisper ASR output) for the
garbled surface + start-time, and `gazetteers/<media_id>.json` for the key.
The exact fold-distances were measured before these tests were written (see
the mining pass recorded in the PR notes); the thresholds are chosen to sit
on the honest side of the measured garbles, not the other way round.
"""
from __future__ import annotations

import pytest

from yamabiko.gazetteer import fuzzy
from yamabiko.gazetteer.anilist import Character
from yamabiko.gazetteer.gazetteer import SeriesGazetteer


# --- Real key subsets (verbatim from the shipped-shape gazetteer caches the
# tray writes; values are the official AniList English spellings). ----------

# gazetteers/188388.json (Digimon Beatbreak). Decoys included so a match is
# never trivially single-key.
_DIGIMON = {
    "ライラモン": "Lilamon",
    "ムラサメモン": "Murasamemon",
    "パンダモン": "Pandamon",
    "グラニット": "Granit",
    "ギガスモン": "Gigasmon",
}

# gazetteers/171627.json (Chainsaw Man - Reze arc).
_CHAINSAW = {
    "マキマ": "Makima",
    "デンジ": "Denji",
    "パワー": "Power",
    "ビーム": "Beam",
    "ポチタ": "Pochita",
}

# gazetteers/95.json (Turn A Gundam).
_TURN_A = {
    "ディアナ": "Dianna",
    "ロラン": "Loran",
    "フラン": "Fran",
    "ヨット": "Joseph Yaht",
    "ドール": "Doll",
}

# gazetteers/177637.json (Sayonara Lara).
_SAYONARA = {
    "ローワン": "Rowan",
    "グレイス": "Grace",
    "ナレーター": "Narrator",
    "ララ": "Lara",
    "リサ": "Lisa",
}

_MAX = 0.25  # the shipped gazetteer.fuzzy_max_distance default


def _match(mapping: dict, surface: str) -> "str | None":
    return fuzzy.match(surface, fuzzy.build_index(mapping), max_distance=_MAX)


# ---------------------------------------------------------------------------
# fold(): the normalization every comparison runs through. Load-bearing --
# a wrong fold silently changes which garbles are recoverable.
# ---------------------------------------------------------------------------

def test_fold_choonpu_becomes_preceding_vowel():
    """The long-vowel mark folds to the vowel of the mora before it, so a
    surface that spells the vowel out (or drops the bar) compares equal.
    ローワン -> ロオワン (the ー after ロ, an o-row mora, becomes オ).

    Seeded defect: make fold() drop ー instead of expanding it (return "" in
    the ch == 'ー' branch) -> fold('ローワン') == 'ロワン' != 'ロオワン' -> red.
    """
    assert fuzzy.fold("ローワン") == "ロオワン"
    assert fuzzy.fold("キース") == "キイス"


def test_fold_small_kana_and_middle_dots_and_width():
    """Small vowel kana fold to full, middle dots are stripped, and NFKC
    folds half-width katakana to full-width -- so ﾃﾞィアナ, ディアナ and
    ディ・アナ all fold to one comparison form.

    Seeded defect: drop the NFKC pass (or the middle-dot strip) -> the
    half-width / dotted forms no longer fold equal -> red.
    """
    assert fuzzy.fold("ディ・アナ") == fuzzy.fold("ディアナ")
    assert fuzzy.fold("ﾃﾞｨｱﾅ") == fuzzy.fold("ディアナ")
    # small ィ folds to full イ
    assert fuzzy.fold("ァィゥェォ") == "アイウエオ"


def test_fold_expands_choonpu_after_small_glide_to_long_vowel_kana():
    """ー after a small glide (ャ/ュ/ョ) folds to the long-vowel kana the
    spelled-out orthography uses -- リョーマ and リョウマ must fold to ONE
    comparison form (the equivalence the fold promises), and ジョー must
    keep its third mora instead of shrinking below the fuzzy min length.
    ャ's long vowel is spelled ア (ジャー ~ ジャア); ュ/ョ take ウ (ジュウ /
    ジョウ, the kana ou-convention).

    Seeded defect: leave ャ/ュ/ョ out of the ー expansion map (the pre-fix
    code: _VOWEL_OF had no small-glide entries, so the bar was silently
    DROPPED) -> fold('リョーマ') == 'リョマ' != fold('リョウマ') -> red.
    (Confirmed red against exactly that pre-fix fold.)
    """
    assert fuzzy.fold("リョーマ") == fuzzy.fold("リョウマ") == "リョウマ"
    assert fuzzy.fold("ジョー") == "ジョウ"
    assert fuzzy.fold("ジュー") == "ジュウ"
    assert fuzzy.fold("ジャー") == "ジャア"


def test_fold_expands_choonpu_after_small_wa_ka_ke():
    """ー after the remaining small kana (ヮ/ヵ/ヶ) folds to that mora's
    column vowel (ヮ and ヵ sit in the ア column, ヶ in the エ column)
    instead of being silently dropped -- the same artificial-edit hazard
    the small glides had.

    Seeded defect: remove any one of the ヮ/ヵ/ヶ entries from _VOWEL_OF ->
    that line's ー is dropped and the folded string loses a mora -> red.
    (Confirmed red against the entry-less pre-fix map.)
    """
    assert fuzzy.fold("クヮーン") == "クヮアン"
    assert fuzzy.fold("イチヵー") == "イチヵア"
    assert fuzzy.fold("ツヶーン") == "ツヶエン"


def test_choonpu_final_key_reaching_three_mora_only_via_the_glide_stays_indexable():
    """A key like ジョー reaches 3 folded mora (ジョウ) and enters the fuzzy
    index. Pre-fix it folded to 2 (ジョ), fell below _MIN_LEN, and silently
    left the index -- the spelled-out surface ジョウ could then never match
    the key at any threshold.

    Seeded defect: the pre-fix ー drop after ョ -> ジョー is excluded from
    the index and the spelled-out surface returns None -> both asserts red.
    """
    mapping = {"ジョー": "Joe", "パワー": "Power", "ポチタ": "Pochita"}
    idx = fuzzy.build_index(mapping)
    assert ("ジョウ", "Joe") in idx.entries
    assert _match(mapping, "ジョウ") == "Joe"


def test_small_glide_long_vowel_garble_stays_within_the_bar():
    """The fold equivalence keeps a genuine 1-edit garble of a small-glide
    long-vowel name recoverable: リョーバ (a last-mora garble of リョウマ)
    folds to リョウバ, d = 1 edit / 4 folded mora = 0.25 <= the shipped
    bar. Pre-fix the dropped ー made the same garble 2 edits of the 4-mora
    key (0.5) -- an artificial distance exactly where the fold promises
    equivalence.

    Seeded defect: the pre-fix ー drop after ョ -> d=0.5 > 0.25 -> None ->
    red.
    """
    mapping = {"リョウマ": "Ryoma", "パンダモン": "Pandamon", "グレイス": "Grace"}
    assert _match(mapping, "リョーバ") == "Ryoma"


# ---------------------------------------------------------------------------
# ACCEPT: real ASR garbles that MUST be recovered (measured d <= 0.25 with a
# comfortable runner-up gap, so no ambiguity).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mapping, garble, expected, provenance",
    [
        # digimon-beatbreak/episode-jp.json t=227.03  (fold d=0.200)
        (_DIGIMON, "ライラン", "Lilamon", "digimon t=227.03"),
        # digimon-beatbreak/episode-jp.json t=712.63  (fold d=0.200)
        (_DIGIMON, "ライロモン", "Lilamon", "digimon t=712.63"),
        # turn-a-gundam-e06/episode-jp.json t=1322.52 (voiced->unvoiced first
        # mora, fold d=0.250)
        (_TURN_A, "ティアナ", "Dianna", "turn-a t=1322.52"),
        # sayonara-lara/episode-jp.json t=176.18  (Grace, fold d=0.250)
        (_SAYONARA, "クレイス", "Grace", "sayonara t=176.18"),
        # sayonara-lara/episode-jp.json t=134.36  (dropped ー, fold d=0.250)
        (_SAYONARA, "ロワン", "Rowan", "sayonara t=134.36"),
        # sayonara-lara/episode-jp.json t=546.97  (ワ->ア + spelled vowel)
        (_SAYONARA, "ローアン", "Rowan", "sayonara t=546.97"),
    ],
)
def test_recovers_real_asr_garble(mapping, garble, expected, provenance):
    """A real kotoba-whisper garble of a katakana name resolves to the exact
    English spelling of the intended character.

    Seeded defect: raise the accept bar to `< max_distance` (strict less-than)
    -> the d==0.250 fixtures (ティアナ / クレイス / ロワン / ローアン) fall out
    -> red. Or drop the whole fuzzy accept branch -> every row red.
    """
    assert _match(mapping, garble) == expected, provenance


# ---------------------------------------------------------------------------
# REJECT: real garbles that land just ABOVE the bar. Recovering these would
# require loosening the threshold past the point where common words start
# matching names, so they are correctly LEFT for the LLM (honest strictness
# cost -- a real Denji/Makima garble is missed, not silently mis-hinted).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mapping, garble, provenance",
    [
        # chainsaw-man-reze/episode-jp.json t=2935.45  (テンジ vs デンジ, d=0.333)
        (_CHAINSAW, "テンジ", "chainsaw t=2935.45 (Denji garble, over bar)"),
        # chainsaw-man-reze/episode-jp.json t=1855.06  (マヒマ vs マキマ, d=0.333)
        (_CHAINSAW, "マヒマ", "chainsaw t=1855.06 (Makima garble, over bar)"),
    ],
)
def test_over_bar_garble_is_refused(mapping, garble, provenance):
    """A 1-edit garble of a SHORT (3-mora) name normalizes to d=0.333, above
    the 0.25 bar, so it is refused rather than hinted.

    Seeded defect: loosen fuzzy_max_distance to >=0.34 (or normalize by the
    LONGER of the two strings) -> these start matching -> red.
    """
    assert _match(mapping, garble) is None, provenance


# ---------------------------------------------------------------------------
# COMMON WORD near a name: a real ASR common noun that sits close to a name
# key must NOT be forced into that name.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "garble, meaning, provenance",
    [
        # chainsaw-man-reze/episode-jp.json t=4038.3  (タイム = "time" vs ビーム)
        ("タイム", "time", "chainsaw t=4038.3"),
        # chainsaw-man-reze/episode-jp.json t=4185.66 (チーム = "team" vs ビーム)
        ("チーム", "team", "chainsaw t=4185.66"),
    ],
)
def test_common_word_near_name_is_not_forced(garble, meaning, provenance):
    """A common katakana loanword (time / team) sits at d=0.333 from ビーム
    (Beam) -- a real sub-threshold-adjacent pair. It must stay unhinted so a
    common word can never be forced into a name.

    Seeded defect: raise the bar to 0.34 -> 'タイム'/'チーム' resolve to Beam,
    corrupting a subtitle that just said "time" -> red.
    """
    assert _match(_CHAINSAW, garble) is None, f"{garble} ({meaning}) {provenance}"


# ---------------------------------------------------------------------------
# Ambiguity guard: when the two nearest keys name DIFFERENT characters and
# are within 0.10 of each other, refuse -- a near-tie is a coin flip and a
# wrong name is worse than none. (Constructed: no real garble in the eval set
# was ambiguous, so a controlled two-close-keys gazetteer pins the guard.)
# ---------------------------------------------------------------------------

def test_ambiguous_between_two_different_names_is_refused():
    """A surface one edit from EACH of two different 5-mora names (both
    d=0.200, gap 0.0) is refused.

    ライリモン is one substitution from ライラモン (Lilamon) and one from
    ライロモン (a different character) -> two DIFFERENT-EN keys tie -> None.

    Seeded defect: delete the ambiguity guard (return best_en unconditionally)
    -> ライリモン resolves to whichever key iterated first -> red.
    """
    mapping = {"ライラモン": "Lilamon", "ライロモン": "Rothmon"}
    assert _match(mapping, "ライリモン") is None


def test_near_tie_to_the_same_name_still_resolves():
    """Two near keys that map to the SAME character are NOT ambiguous: both
    spellings yield one English name, so the hint is served. (Digimon's
    ローズ / ウッドヴィル -> 'Rose Woodville' shape.)

    Seeded defect: make the ambiguity guard compare the second-best key
    regardless of its English value (drop the `en != best_en` filter) -> two
    same-name keys look like a tie -> None -> red.
    """
    mapping = {"ローズウッドヴィル": "Rose Woodville", "ローズ": "Rose Woodville"}
    # A garble one edit from the full form; the short key maps to the same EN.
    assert _match(mapping, "ロースウッドヴィル") == "Rose Woodville"


# ---------------------------------------------------------------------------
# Gates: Latin/kanji-heavy noise and too-short surfaces are never attempted.
# ---------------------------------------------------------------------------

def test_non_katakana_surface_is_not_attempted():
    """A surface that is not katakana-dominant (Latin ASR noise) is refused
    before any distance is computed, however close it looks.

    Seeded defect: drop the katakana-dominance gate in match() -> a Latin
    surface can score against a key -> a spurious hint -> red here.
    """
    assert _match(_CHAINSAW, "Beem") is None


def test_too_short_surface_is_not_attempted():
    """A 2-mora surface is below the min length: too little signal to fuzzy
    on without matching half the gazetteer.

    Seeded defect: drop the len >= 3 gate -> a 2-mora surface fuzzes -> red.
    """
    assert _match(_CHAINSAW, "デン") is None


def test_kanji_key_is_excluded_from_the_fuzzy_index():
    """Kanji-containing keys (e.g. 早川アキ) are NOT katakana-dominant, so they
    never enter the fuzzy index -- v1.5 recovers katakana surfaces only; a
    garbled kanji name is out of scope, not silently mismatched.

    Seeded defect: index every key regardless of script -> the kanji key is
    present and a katakana garble could snap to it -> the entries count is
    wrong -> red.
    """
    idx = fuzzy.build_index({"早川アキ": "Aki Hayakawa", "ビーム": "Beam"})
    assert idx.entries == (("ビイム", "Beam"),)


# ---------------------------------------------------------------------------
# SeriesGazetteer.fuzzy_lookup: the instance seam the pipeline calls. Folded
# index built once and cached on the instance.
# ---------------------------------------------------------------------------

def test_series_gazetteer_fuzzy_lookup_recovers_garble():
    """SeriesGazetteer.fuzzy_lookup wraps fuzzy.match over the instance map.

    Seeded defect: have fuzzy_lookup return None unconditionally -> red.
    """
    g = SeriesGazetteer(dict(_DIGIMON))
    assert g.fuzzy_lookup("ライラン", max_distance=_MAX) == "Lilamon"


def test_fuzzy_index_is_built_once_and_cached_on_the_instance():
    """The folded index is computed on first fuzzy_lookup and reused, so a
    hot loop does not re-fold 100 keys per sentence.

    Seeded defect: rebuild the index every call (never cache) -> the identity
    assertion below fails because a fresh tuple is built each time -> red.
    """
    g = SeriesGazetteer(dict(_DIGIMON))
    g.fuzzy_lookup("ライラン", max_distance=_MAX)
    first = g._fuzzy_index
    assert first is not None
    g.fuzzy_lookup("ライロモン", max_distance=_MAX)
    assert g._fuzzy_index is first, "index must be cached, not rebuilt per call"


def test_fuzzy_lookup_exact_surface_still_resolves():
    """An exact katakana surface trivially folds to distance 0 and resolves
    (fuzzy is a superset of exact for katakana keys).

    Seeded defect: skip the d==0 case (start the accept range above 0) -> an
    exact katakana surface passed to fuzzy_lookup misses -> red.
    """
    g = SeriesGazetteer.from_characters([Character("ベニマル", "Benimaru", [])])
    assert g.fuzzy_lookup("ベニマル", max_distance=_MAX) == "Benimaru"


def test_ambiguity_gap_of_exactly_the_margin_is_accepted():
    """Boundary pin (review M1): the ambiguity guard refuses only when the
    different-name runner-up is STRICTLY closer than _AMBIGUITY_MARGIN, so a
    gap of exactly 0.10 resolves to the nearer key. Synthetic 10-mora keys
    make the arithmetic exact: best = 1 sub / 10 = 0.10, runner = 2 subs /
    10 = 0.20, gap = 0.10 exactly.

    Seeded defect: relaxing the guard to `<=` refuses this match -> red.
    """
    index = fuzzy.build_index({
        "アアアアアアアコココ": "Beta",   # 1 substitution from the surface
        "アアアアアアアアアア": "Alpha",  # 2 substitutions from the surface
    })
    assert fuzzy.match("アアアアアアアアココ", index, max_distance=0.25) == "Beta"
