"""Fuzzy katakana matching for ASR-garbled proper-noun surfaces (gazetteer
v1.5). Pure: no I/O, no romanizer, no config -- callers hand in surfaces and
a prebuilt index, and get back an exact English name or None.

Gazetteer v1 hints only on an EXACT katakana surface, but the ASR model
mangles name surfaces often enough that the exact lookup misses most lines
(クレイス for グレイス: one devoiced mora on a 4-kana name, 1 edit / 4 folded
mora = 0.25, exactly the shipped 0.25 bar; ライラン for ライラモン: 1 edit /
5 = 0.20). This layer recovers a name when the
garble is close enough AND refuses when it is not: a wrong-name hint steers
the translator worse than no hint at all, so the accept bar is deliberately
tight and any near-tie between two DIFFERENT names is refused.

Scope (v1.5): katakana surfaces only. Kanji names are compared via their
reading (converted to katakana by the caller) but almost never fire -- the
strict bar keeps cross-kanji rewrites (牧場様 -> マキマ) out.

Distance is normalized Levenshtein over the FOLDED katakana strings, divided
by the folded KEY length (a short 3-mora name therefore tolerates no edits;
a longer name tolerates one). Inputs are short and keys are <= ~100, so the
O(len^2) DP is trivial and rapidfuzz is not needed (nor is it a runtime dep).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# A surface/key must reach this many folded mora to be fuzzy-eligible: below
# it there is too little signal to match without snapping to half the map.
_MIN_LEN = 3
# A surface must be at least this fraction katakana (after folding) to be
# attempted; Latin / kanji-heavy ASR noise is not a garbled katakana name.
_KATAKANA_DOMINANCE = 0.6
# If a second key naming a DIFFERENT character is within this normalized
# distance of the best, refuse: a near-tie is a coin flip and a wrong name is
# worse than none.
_AMBIGUITY_MARGIN = 0.10

# Katakana block (゠-ヿ) covers full katakana incl. ー and small kana; after
# folding, ー is gone and small vowels are full, so a folded katakana name is
# entirely within this range.
_KATAKANA_CHAR = re.compile(r"[゠-ヿㇰ-ㇿ]")

# Choonpu (ー) folds to the vowel kana matching the preceding mora's vowel, so
# a spelled-out long vowel (ローワン ~ ロオワン) and a dropped bar compare close.
_VOWEL_KANA = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}
_VOWEL_OF: dict[str, str] = {}
for _v, _chars in {
    "a": "アカサタナハマヤラワガザダバパ",
    "i": "イキシチニヒミリギジヂビピ",
    "u": "ウクスツヌフムユルグズヅブプヴ",
    "e": "エケセテネヘメレゲゼデベペ",
    "o": "オコソトノホモヨロヲゴゾドボポ",
}.items():
    for _c in _chars:
        _VOWEL_OF[_c] = _v
# Small glides: ー after ャ/ュ/ョ expands to the long-vowel kana the
# spelled-out orthography uses (ジャー ~ ジャア, ジュー ~ ジュウ, ジョー ~
# ジョウ -- the ou-convention, so ョ maps to ウ, not オ). Without these
# entries the bar was silently DROPPED: リョーマ folded to リョマ while
# リョウマ kept its ウ (an artificial edit exactly where the fold promises
# equivalence), and a ジョー-class key shrank below _MIN_LEN and left the
# index.
_VOWEL_OF["ャ"] = "a"
_VOWEL_OF["ュ"] = "u"
_VOWEL_OF["ョ"] = "u"
# The remaining small kana: ヮ/ヵ sit in the ア column, ヶ in the エ column.
# Without them a following ー is silently dropped (the same artificial-edit
# hazard the glides had).
_VOWEL_OF["ヮ"] = "a"
_VOWEL_OF["ヵ"] = "a"
_VOWEL_OF["ヶ"] = "e"

# Small vowel kana fold to their full form (the brief's ァィゥェォ set). ャ/ュ/ョ
# and ッ are intentionally left as-is: they carry structural information and
# appear on both sides of a genuine match.
_SMALL_VOWEL = {"ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ"}

# Name separators: katakana middle dot, ASCII '=' (NFKC-folded from ＝), and
# the middle dot U+00B7. Stripped so ディ・アナ and ディアナ compare equal.
_SEPARATORS = re.compile(r"[・=·]")


def fold(s: str) -> str:
    """Normalize a katakana surface for comparison ONLY (never emitted):
    NFKC (half-width -> full), strip name separators, expand ー into the
    preceding vowel, fold small vowel kana to full."""
    s = unicodedata.normalize("NFKC", s).strip()
    s = _SEPARATORS.sub("", s)
    out: list[str] = []
    for ch in s:
        if ch == "ー":
            if out:
                v = _VOWEL_OF.get(out[-1])
                if v:
                    out.append(_VOWEL_KANA[v])
            # A leading ー, or one after a char with no known vowel, is dropped.
            continue
        out.append(_SMALL_VOWEL.get(ch, ch))
    return "".join(out)


def hiragana_to_katakana(s: str) -> str:
    """Map hiragana to katakana (used by the pipeline to feed a kanji
    candidate's reading through the katakana-only fuzzy path). Non-hiragana
    characters pass through unchanged."""
    out: list[str] = []
    for ch in s:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            out.append(chr(cp + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _katakana_ratio(folded: str) -> float:
    if not folded:
        return 0.0
    n = sum(1 for c in folded if _KATAKANA_CHAR.match(c))
    return n / len(folded)


def _levenshtein(a: str, b: str) -> int:
    """Plain edit distance (insert/delete/substitute). Inputs are short."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class FuzzyIndex:
    """Prebuilt (folded_key, english) pairs for the katakana-dominant keys of
    one gazetteer. Built once per gazetteer (folding is not free); the hot
    path only runs Levenshtein against these."""

    entries: tuple[tuple[str, str], ...]


def build_index(mapping: dict[str, str]) -> FuzzyIndex:
    """Fold every eligible (katakana-dominant, >= 3 mora) key once. Kanji and
    Latin-heavy keys are skipped -- v1.5 recovers katakana surfaces only.
    Pure-HIRAGANA surfaces are likewise out of scope by design: they fold to
    hiragana, fail the dominance check, and never fuzzy-match (a hiragana
    spelling of a name is an ASR *style* choice, not a garble)."""
    entries: list[tuple[str, str]] = []
    for key, en in mapping.items():
        folded = fold(key)
        if len(folded) < _MIN_LEN:
            continue
        if _katakana_ratio(folded) < _KATAKANA_DOMINANCE:
            continue
        entries.append((folded, en))
    return FuzzyIndex(tuple(entries))


def match(surface: str, index: FuzzyIndex, *, max_distance: float) -> "str | None":
    """Return the exact English name for a garbled katakana `surface`, or None.

    Refuses (returns None) when the surface is too short, not katakana-
    dominant, no key is within `max_distance` (normalized by folded key
    length), or the two nearest keys name DIFFERENT characters within
    `_AMBIGUITY_MARGIN` of each other.
    """
    if not index.entries:
        return None
    folded = fold(surface)
    if len(folded) < _MIN_LEN:
        return None
    if _katakana_ratio(folded) < _KATAKANA_DOMINANCE:
        return None

    scored = [
        (_levenshtein(folded, fkey) / len(fkey), en) for fkey, en in index.entries
    ]
    best_dist, best_en = min(scored, key=lambda t: t[0])
    if best_dist > max_distance:
        return None

    # Ambiguity guard: the nearest key naming a DIFFERENT character must be
    # clearly farther. Keys that map to the SAME English name are not rivals
    # (both yield one hint), so they are excluded from the runner-up.
    runner = min((d for d, en in scored if en != best_en), default=None)
    if runner is not None and runner - best_dist < _AMBIGUITY_MARGIN:
        return None
    return best_en
