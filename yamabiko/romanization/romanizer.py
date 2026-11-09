"""Hiragana readings for Japanese text, using the UniDic tagger cutlet loads.

`Romanizer.reading` reads a whole word (a name candidate) and
`Romanizer.token_readings` reads every token of a line for furigana.
cutlet is kept for its configured fugashi/UniDic tagger (`Romanizer.tagger`),
which the proper-noun extractor and the translator's polarity guard share.
No romaji is produced.
"""
from __future__ import annotations

import cutlet

# Kanji codepoint ranges: the basic block alone misses CJK
# Extension A and the compatibility ideographs used in name variants
# (山﨑 U+FA11), so those characters fell out of kanji runs. Shared with
# jp_text._is_jp_char so the two classifiers can't diverge.
from yamabiko.jp_text import KANJI_RANGES as _KANJI_RANGES

# Not kanji themselves, but they extend a kanji run when they FOLLOW one:
# iteration marks 々 (U+3005) / 〻 (U+303B), the closing mark 〆 (U+3006),
# and small ヵ/ヶ (U+30F5/U+30F6) used in counters and place names
# (佐々木, 霞ヶ関, 一ヶ月). Splitting the run at these characters hands the
# tagger isolated fragments with wrong readings. Kept separate from
# contains_kanji so a leading mark never opens a run of its own.
_KANJI_RUN_MARKS = frozenset("々〻〆ヵヶ")


def contains_kanji(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _KANJI_RANGES):
            return True
    return False


def is_kanji_run_mark(ch: str) -> bool:
    return ch in _KANJI_RUN_MARKS


def _katakana_to_hiragana(text: str) -> str:
    """Convert each katakana character to its hiragana equivalent."""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            out.append(chr(cp - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _token_kana(tok) -> str:
    """A tagger token's katakana reading: UniDic kana, then pron, else ""."""
    feature = tok.feature
    return getattr(feature, "kana", None) or getattr(feature, "pron", None) or ""


def split_reading(pieces: list[tuple[str, bool]], reading: str) -> list[str] | None:
    """Split one token's hiragana `reading` across its kanji pieces.

    `pieces` is the token surface in order, cut into kanji runs (True) and
    the stretches between them (False). Each non-kanji stretch must appear
    verbatim in the reading, which pins okurigana (行き/いき -> 行=い) and
    leading kana prefixes in place. Returns each kanji piece's reading
    when exactly one split fits, else None: a kana stretch that could sit
    in more than one place, or one the reading does not spell, leaves the
    split unknown, and the caller should not guess.
    """
    found: list[list[str]] = []
    acc: list[str] = []

    def walk(i: int, pos: int) -> None:
        if len(found) > 1:
            return
        if i == len(pieces):
            if pos == len(reading):
                found.append(acc.copy())
            return
        text, is_kanji = pieces[i]
        if not is_kanji:
            kana = _katakana_to_hiragana(text)
            if reading.startswith(kana, pos):
                walk(i + 1, pos + len(kana))
            return
        for end in range(pos + 1, len(reading) + 1):
            acc.append(reading[pos:end])
            walk(i + 1, end)
            acc.pop()

    walk(0, 0)
    return found[0] if len(found) == 1 else None


class Romanizer:
    """Reading extraction over cutlet's fugashi/UniDic tagger."""

    def __init__(self) -> None:
        self._katsu = cutlet.Cutlet(ensure_ascii=False)
        # tagger: shared fugashi.Tagger; callers may reuse to avoid re-loading UniDic.
        self.tagger = self._katsu.tagger

    def reading(self, text: str) -> str:
        """Return the hiragana reading of a whole word (a name candidate).

        Pure-kana input (hiragana or katakana) is returned as hiragana.
        Kanji input is tokenized via the underlying MeCab tagger and each
        token's kana reading is concatenated as hiragana. Returns "" when
        any token has no dictionary reading (an unknown kanji such as 禰 in
        禰豆子): a reading that keeps the raw kanji is not a reading.

        Furigana uses token_readings on the whole line instead: a kanji
        cut out of its sentence is read out of context (行 -> こう).
        """
        if not text:
            return ""
        if not contains_kanji(text):
            return _katakana_to_hiragana(text)
        parts: list[str] = []
        for tok in self.tagger(text):
            kana = _token_kana(tok)
            if not kana:
                return ""
            parts.append(_katakana_to_hiragana(kana))
        return "".join(parts)

    def token_readings(self, text: str) -> list[tuple[int, int, str]]:
        """Tag `text` once; return each token's (start, end, reading).

        Offsets index `text` exactly (fugashi reports the whitespace it
        skipped before each token). The reading is hiragana, or "" when
        the dictionary has none (unknown words, symbols).
        """
        spans: list[tuple[int, int, str]] = []
        pos = 0
        for tok in self.tagger(text):
            pos += len(tok.white_space)
            end = pos + len(tok.surface)
            spans.append((pos, end, _katakana_to_hiragana(_token_kana(tok))))
            pos = end
        return spans
