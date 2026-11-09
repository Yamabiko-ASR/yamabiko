"""Meaningful Japanese interjections for the ASR bench.

A short utterance such as はい, ううん, えっ or おい changes the meaning of a
scene, so the bench tracks these apart from non-lexical sounds (grunts,
gasps, cries, laughter). ``INTERJECTIONS`` maps each entry's normalized form
to its usual written forms and its category; ``docs/ASR-BENCH.md`` gives the
reason for every entry.

Normalized form: the katakana reading from the MeCab/UniDic tagger the app
already ships (``fugashi`` with ``unidic-lite``), long-vowel marks removed
and small vowels enlarged, so はーい reads ハイ and ねぇ reads ネエ. ッ is
kept: えっ and ええっ (surprise) differ from ええ (yes), ああっ (a cry) from
ああ. Long-vowel marks are not: うん and うーん share ウン.

An occurrence is one utterance as the text separates it: a chunk between
spaces, punctuation or symbols (the characters ``ja-minimal-1.1`` removes),
tagged on its own.

- A chunk whose tokens all have UniDic ``pos1`` 感動詞, 補助記号, 記号 or
  空白 is an occurrence when its whole reading is an entry, or repeats a
  multi-kana entry (はいはい, いやいや). Cries written as runs of one vowel
  (あああ, うわあああ) are not entries.
- In any other chunk, a first token with ``pos1`` 感動詞 whose reading is
  an entry is an occurrence (はいわかりました, ありがとうございます, ねえ聞いて).

The ``pos1`` gate keeps 嫌 (いや, "dislike"), the adverb ああ (ああいう) and
the sentence-final particle ねえ out. ``reading`` tags the whole normalized
text, as the scorer's reading form does, and each occurrence's span indexes
that reading.
"""

from __future__ import annotations

import functools
import unicodedata
from dataclasses import dataclass

from scripts.e0_evidence.normalize_ja import normalize_minimal

CATEGORIES = ("affirm", "deny", "surprise", "call", "hesitation", "greeting")
# Normalized form -> (usual written forms, category).
INTERJECTIONS = {
    "ハイ": ("はい, はーい", "affirm"),
    "ハイッ": ("はいっ", "affirm"),
    "ウン": ("うん, うーん", "affirm"),
    "エエ": ("ええ", "affirm"),
    "アア": ("ああ", "affirm"),
    "オウ": ("おう", "affirm"),
    "ウム": ("うむ", "affirm"),
    "ヨシ": ("よし", "affirm"),
    "ヨシッ": ("よしっ", "affirm"),
    "イイエ": ("いいえ", "deny"),
    "イエ": ("いえ", "deny"),
    "ウウン": ("ううん", "deny"),
    "イヤ": ("いや", "deny"),
    "イヤッ": ("いやっ", "deny"),
    "エ": ("え, えー", "surprise"),
    "エッ": ("えっ", "surprise"),
    "エエッ": ("ええっ", "surprise"),
    "ア": ("あ, あー", "surprise"),
    "アッ": ("あっ", "surprise"),
    "アレ": ("あれ", "surprise"),
    "ヘエ": ("へえ", "surprise"),
    "ホウ": ("ほう", "surprise"),
    "オヤ": ("おや", "surprise"),
    "アラ": ("あら", "surprise"),
    "オイ": ("おい, おーい", "call"),
    "オイッ": ("おいっ", "call"),
    "ネエ": ("ねえ", "call"),
    "ナア": ("なあ", "call"),
    "ホラ": ("ほら", "call"),
    "コラ": ("こら", "call"),
    "コラッ": ("こらっ", "call"),
    "モシモシ": ("もしもし", "call"),
    "ヤア": ("やあ", "call"),
    "エット": ("えっと, えーっと", "hesitation"),
    "エト": ("えーと", "hesitation"),
    "エエト": ("ええと", "hesitation"),
    "アノ": ("あの, あのー", "hesitation"),
    "アノウ": ("あのう", "hesitation"),
    "サア": ("さあ", "hesitation"),
    "アリガトウ": ("ありがとう", "greeting"),
    "アリガト": ("ありがと", "greeting"),
    "オハヨウ": ("おはよう", "greeting"),
    "コンニチハ": ("こんにちは", "greeting"),
    "コンバンハ": ("こんばんは", "greeting"),
    "サヨウナラ": ("さようなら", "greeting"),
}
INTERJECTION_POS1 = "感動詞"
# UniDic pos1 values of a vocalization-only text (every token one of them).
VOCAL_POS1 = frozenset({"感動詞", "補助記号", "記号", "空白"})
_SMALL_VOWELS = str.maketrans("ァィゥェォ", "アイウエオ")
_TAGGER = None


def katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "\u3041" <= ch <= "\u3096" else ch for ch in text)


def interjection_key(reading: str) -> str | None:
    """The ``INTERJECTIONS`` entry a katakana reading (long-vowel marks
    already dropped) stands for, or None. A reading repeating a multi-kana
    entry (ハイハイ) stands for the entry; runs of one kana (アアア) never do."""
    key = reading.translate(_SMALL_VOWELS)
    if key in INTERJECTIONS:
        return key
    for size in range(2, len(key) // 2 + 1):
        unit = key[:size]
        if (
            len(key) % size == 0
            and unit * (len(key) // size) == key
            and unit in INTERJECTIONS
            and len(set(unit)) > 1
        ):
            return unit
    return None


def _tag(normalized: str) -> list[tuple[str, str, str]]:
    """(surface, katakana reading without long-vowel marks, pos1) per token."""
    global _TAGGER
    if _TAGGER is None:
        import fugashi

        _TAGGER = fugashi.Tagger()
    tokens = []
    for word in _TAGGER(normalized):
        kana = getattr(word.feature, "kana", None)
        piece = katakana(kana if kana and kana != "*" else word.surface).replace("ー", "")
        tokens.append((word.surface, piece, word.feature.pos1))
    if "".join(surface for surface, _, _ in tokens) != normalized:
        raise ValueError(f"tagger surfaces do not cover {normalized!r}")
    return tokens


@functools.lru_cache(maxsize=None)
def reading(text: str) -> str:
    """Katakana reading of ``normalize_minimal(text)``, long-vowel marks dropped."""
    return "".join(piece for _, piece, _ in _tag(normalize_minimal(text)))


def _separator(character: str) -> bool:
    """Whitespace, punctuation or a symbol: a character ``ja-minimal-1.1`` removes."""
    return character.isspace() or unicodedata.category(character)[0] in ("P", "S")


def _chunks(text: str) -> list[str]:
    """The minimal-normalized pieces ``text`` separates by whitespace,
    punctuation or symbols; they concatenate to ``normalize_minimal(text)``."""
    chunks: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", text):
        if _separator(character):
            if current:
                chunks.append("".join(current))
            current = []
        else:
            current.append(character)
    if current:
        chunks.append("".join(current))
    return chunks


@functools.lru_cache(maxsize=None)
def _chunk_tags(chunk: str) -> tuple[tuple[str, str, str], ...]:
    return tuple(_tag(chunk))


def chunk_tags(text: str) -> list[tuple[str, tuple[tuple[str, str, str], ...]]]:
    """Each chunk of ``text`` (split as occurrences are, at whitespace,
    punctuation and symbols) with its (surface, reading, pos1) tokens."""
    return [(chunk, _chunk_tags(chunk)) for chunk in _chunks(text)]


def _chunk_interjection(chunk: str) -> tuple[int, str] | None:
    """(surface length, entry) of the chunk's interjection occurrence, or None."""
    tokens = _tag(chunk)
    if all(pos1 in VOCAL_POS1 for _, _, pos1 in tokens):
        key = interjection_key("".join(piece for _, piece, _ in tokens))
        return None if key is None else (len(chunk), key)
    surface, piece, pos1 = tokens[0]
    key = interjection_key(piece) if pos1 == INTERJECTION_POS1 else None
    return None if key is None else (len(surface), key)


def pieces(text: str) -> list[tuple[str, str | None]]:
    """``text`` after NFKC, cut into consecutive pieces, each with the
    ``INTERJECTIONS`` key of the occurrence it is, or None. An occurrence
    piece is a whole chunk with the separators after it, or the first token
    of a longer chunk (the rest of the chunk and its separators follow as a
    piece of their own); separators opening the text are a piece of their
    own. The pieces concatenate to the NFKC text; occurrences are detected
    per chunk as ``analyze`` detects them."""
    text = unicodedata.normalize("NFKC", text)
    result: list[tuple[str, str | None]] = []
    position = 0
    while position < len(text):
        start = position
        while position < len(text) and not _separator(text[position]):
            position += 1
        chunk_end = position
        while position < len(text) and _separator(text[position]):
            position += 1
        found = _chunk_interjection(text[start:chunk_end]) if chunk_end > start else None
        if found is None:
            result.append((text[start:position], None))
        elif start + found[0] == chunk_end:
            result.append((text[start:position], found[1]))
        else:
            result.append((text[start:start + found[0]], found[1]))
            result.append((text[start + found[0]:position], None))
    return result


@dataclass(frozen=True)
class Analysis:
    # Katakana reading of the minimal-normalized text, long-vowel marks dropped.
    reading: str
    # Every token a vocalization or a symbol (UniDic pos1 in VOCAL_POS1).
    vocal: bool
    # (start, end, INTERJECTIONS key) in reading characters, per occurrence.
    interjections: tuple[tuple[int, int, str], ...]


@functools.lru_cache(maxsize=None)
def analyze(text: str) -> Analysis:
    """Tag ``text``: its reading, whether it is vocalization-only, and where
    in the reading its lexicon interjections are."""
    normalized = normalize_minimal(text)
    tokens = _tag(normalized)
    # Surface offset -> reading offset, exact at token boundaries and inside
    # a token whose reading is as long as its surface (kana interjections).
    bounds: list[tuple[int, int, int, int]] = []
    surface_at = reading_at = 0
    for surface, piece, _ in tokens:
        bounds.append((surface_at, len(surface), reading_at, len(piece)))
        surface_at += len(surface)
        reading_at += len(piece)

    def reading_offset(position: int) -> int | None:
        if position == surface_at:
            return reading_at
        for start, size, piece_start, piece_size in bounds:
            if start <= position < start + size:
                if position == start:
                    return piece_start
                return piece_start + position - start if size == piece_size else None
        return None

    interjections: list[tuple[int, int, str]] = []
    chunk_start = 0
    for chunk in _chunks(text):
        found = _chunk_interjection(chunk)
        if found is not None:
            size, key = found
            start = reading_offset(chunk_start)
            end = reading_offset(chunk_start + size)
            if start is not None and end is not None and end > start:
                interjections.append((start, end, key))
        chunk_start += len(chunk)
    return Analysis(
        "".join(piece for _, piece, _ in tokens),
        all(pos1 in VOCAL_POS1 for _, _, pos1 in tokens),
        tuple(interjections),
    )


def is_meaningful_interjection(text: str) -> bool:
    """True when ``text`` is vocalization-only and holds at least one lexicon
    interjection (the bench's ``kind=intj`` line): はい, えっ!?, あっ はい;
    not ハァハァ, くっ, あああ or はい わかりました."""
    analysis = analyze(text)
    return analysis.vocal and bool(analysis.interjections)
