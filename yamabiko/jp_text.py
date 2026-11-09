"""Japanese-character detection for the non-JA ASR guard.

Accepts text that is predominantly Japanese. Rejects:
  - Pure non-JP text (no kanji/kana at all).
  - ASR hallucinations that intersperse English sentences with a token
    or two of JP, e.g. `"オッケー、I will let him know."` — ASR
    sometimes emits these when fed English audio and a JP language hint.

Rule: require at least one kanji/kana AND Latin-letter density below
`_MAX_LATIN_RATIO` of total letter-ish chars. A few Latin abbreviations
(AI, NHK, LDP) inside a mostly-JP line pass fine; a full English sentence
with a katakana fragment does not.
"""
from __future__ import annotations

_MAX_LATIN_RATIO = 0.25

# Kanji codepoint ranges, shared with romanization.romanizer.contains_kanji
# so the two classifiers can't diverge again (PR #42 review: 﨑 U+FA11 got
# correct furigana but still counted toward the Latin ratio here).
KANJI_RANGES = (
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (basic block)
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs (山﨑-style names)
)


def _is_jp_char(ch: str) -> bool:
    code = ord(ch)
    return (
        any(lo <= code <= hi for lo, hi in KANJI_RANGES)
        or 0x3040 <= code <= 0x309F  # hiragana (incl. ゝゞ iteration marks)
        or 0x30A0 <= code <= 0x30FF  # katakana (incl. ヽヾ iteration marks)
        or code in (0x3005, 0x3006, 0x303B)  # 々 〆 〻 — alphabetic
        # (category Lm/Lo) so they'd otherwise count as Latin
        or 0xFF66 <= code <= 0xFF9F  # halfwidth katakana
    )


def is_japanese(text: str) -> bool:
    if not text:
        return False
    jp_count = 0
    latin_count = 0
    for ch in text:
        if _is_jp_char(ch):
            jp_count += 1
        elif ch.isalpha():  # any alphabetic that isn't JP → Latin/other
            latin_count += 1
    if jp_count == 0:
        return False
    total_letters = jp_count + latin_count
    if total_letters == 0:
        return False
    latin_ratio = latin_count / total_letters
    return latin_ratio <= _MAX_LATIN_RATIO
