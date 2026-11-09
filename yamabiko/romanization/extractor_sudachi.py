"""Katakana-brand extraction via SudachiPy split-mode C.

Runs alongside the fugashi+UniDic extractor. Split-mode C keeps
multi-token katakana brand names as a single 固有名詞 token — the
complementary coverage UniDic's split mode misses (e.g. ソフトバンク
which fugashi splits into [ソフト, バンク]).

Same guards as the fugashi extractor: denylist, single-kanji skip,
short kana-only skip.
"""
from __future__ import annotations

from typing import Iterable

from sudachipy import dictionary, tokenizer


_KANJI_LO = 0x4E00
_KANJI_HI = 0x9FFF


def _is_kanji(ch: str) -> bool:
    return _KANJI_LO <= ord(ch) <= _KANJI_HI


def _has_kanji(s: str) -> bool:
    return any(_is_kanji(c) for c in s)


class SudachiExtractor:
    def __init__(self, denylist: Iterable[str] = ()) -> None:
        self._tokenizer = dictionary.Dictionary().create()
        self._mode = tokenizer.Tokenizer.SplitMode.C
        self._denylist = set(denylist)

    def extract(self, text: str) -> list[str]:
        if not text:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for tok in self._tokenizer.tokenize(text, self._mode):
            surface = tok.surface()
            if surface in seen or surface in self._denylist:
                continue
            pos = tok.part_of_speech()
            # pos is a 6-tuple (pos1, pos2, pos3, pos4, conjType, conjForm)
            if pos[0] != "名詞":
                continue
            if "固有名詞" not in pos[1:4]:
                continue
            if len(surface) == 1 and _is_kanji(surface):
                continue
            if len(surface) < 2 and not _has_kanji(surface):
                continue
            seen.add(surface)
            out.append(surface)
        return out
