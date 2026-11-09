"""Post-ASR sentence splitter wrapping fast-bunkai.

Splits Japanese ASR transcripts into sentences on 。！？ with JP-aware
edge cases: decimals (3.14), parentheticals ((笑)), URLs, ellipses,
and trailing particles without 。.

The caller is responsible for buffering any trailing partial sentence
across chunks.
"""
from __future__ import annotations

_TERMINATORS = ("。", "！", "？", "?", "!")


def has_terminator(sentence: str) -> bool:
    """True iff the sentence ends in one of 。！？?!."""
    if not sentence:
        return False
    return sentence.rstrip().endswith(_TERMINATORS)


class SentenceSplitter:
    """Lazy-initialized FastBunkai wrapper.

    First call loads fast-bunkai's rule-based splitter. Rule-only by
    default (the ML linebreak model would add ~20 MB download + latency).
    """

    def __init__(self) -> None:
        self._bunkai = None

    def _ensure_loaded(self) -> None:
        if self._bunkai is None:
            from fast_bunkai import FastBunkai
            self._bunkai = FastBunkai()

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        raw = [s for s in self._bunkai(text) if s.strip()]
        # FastBunkai can emit a bare terminator (e.g. "。") as its own
        # segment when it trails a parenthetical like "すごい(笑)。".
        # Merge any such orphan terminator back onto the previous segment.
        merged: list[str] = []
        for seg in raw:
            stripped = seg.strip()
            if merged and stripped and all(c in "".join(_TERMINATORS) for c in stripped):
                merged[-1] = merged[-1] + stripped
            else:
                merged.append(seg)
        return merged
