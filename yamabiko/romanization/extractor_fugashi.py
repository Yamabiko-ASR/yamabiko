"""Proper-noun extraction via fugashi + UniDic.

Two passes feed `extract_all`:

  Pass 1 (固有名詞 filter): pos1 == '名詞' and pos2 == '固有名詞'.
  UniDic's POS hierarchy cleanly separates proper nouns (人名/地名/組織/
  一般) from common nouns, avoiding nagisa's PROPN/NOUN confusion.
  Additional guards on this pass:
    - Skip single-kanji tokens (high common-noun false-positive rate).
    - Skip katakana/hiragana-only tokens shorter than 2 chars.
    - Honor a user-supplied denylist.

  Pass 2 (katakana-run scan): UniDic frequently POS-splits multi-token
  katakana brand/name strings (e.g. ソフトバンク → [ソフト, バンク], both
  普通名詞) so Pass 1 alone misses them. Pass 2 rescans the raw text for
  contiguous katakana runs of length >= 2 via `_KATA_RUN` and adds any
  run not already captured. The middle dot `・` (U+30FB) is excluded from
  the run's character class so name-separator dots split the run instead
  of merging it (スバル・ナツキ -> スバル, ナツキ, never a single
  スバル・ナツキ candidate).

`extract_all(text)` returns `(all candidates, the 固有名詞 subset)`:
Pass-1 hits land in both; Pass-2-only hits (e.g. a raw katakana run that
UniDic tags as common noun) land in the candidate list only, not in the
propn set. `extract(text)` is `extract_all(text)[0]`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import fugashi


_KANJI_LO = 0x4E00
_KANJI_HI = 0x9FFF

# Contiguous katakana run, length >= 2. Range is U+30A0-U+30FA (゠..ヺ)
# plus U+30FC-U+30FF (ー..ヿ), which deliberately excludes U+30FB (・)
# so the middle dot splits a run instead of being absorbed into it.
_KATA_RUN = re.compile(r"[゠-ヺー-ヿ]{2,}")


def _is_kanji(ch: str) -> bool:
    return _KANJI_LO <= ord(ch) <= _KANJI_HI


def _has_kanji(s: str) -> bool:
    return any(_is_kanji(c) for c in s)


class ProperNounExtractor:
    def __init__(self, tagger: fugashi.Tagger, denylist: Iterable[str] = ()) -> None:
        self._tagger = tagger
        self._denylist = set(denylist)

    def extract(self, text: str) -> list[str]:
        return self.extract_all(text)[0]

    def extract_all(self, text: str) -> tuple[list[str], set[str]]:
        """Return (all candidates, the subset UniDic tags 固有名詞). Pass 1 =
        固有名詞; Pass 2 = raw katakana runs (>=2, split on ・)."""
        if not text:
            return [], set()
        seen: set[str] = set()
        out: list[str] = []
        propn: set[str] = set()
        for tok in self._tagger(text):
            surface = tok.surface
            if surface in seen or surface in self._denylist:
                continue
            feat = tok.feature
            if getattr(feat, "pos1", None) != "名詞":
                continue
            if getattr(feat, "pos2", None) != "固有名詞":
                continue
            if len(surface) == 1 and _is_kanji(surface):
                continue
            if len(surface) < 2 and not _has_kanji(surface):
                continue
            seen.add(surface)
            out.append(surface)
            propn.add(surface)
        for m in _KATA_RUN.finditer(text):
            run = m.group()
            if run in seen or run in self._denylist:
                continue
            seen.add(run)
            out.append(run)
        return out, propn


def load_denylist(path: Path) -> set[str]:
    """Load one-term-per-line UTF-8 denylist; ignore blanks and `#`-prefixed comments."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        out.add(term)
    return out
