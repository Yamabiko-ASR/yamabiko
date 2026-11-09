"""Frozen Japanese normalization policies for E0 scoring."""

from __future__ import annotations

import functools
import importlib.metadata
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .wire import canonical_sha256

if TYPE_CHECKING:
    import fugashi


_MINIMAL_TRANSFORMS = (
    "unicode_nfkc",
    "remove_str_isspace",
    "remove_unicode_category_prefix_p_s",
)
_LENIENT_TRANSFORMS = (
    *_MINIMAL_TRANSFORMS,
    "ascii_uppercase_to_lowercase",
    "katakana_u+30a1-u+30f6_and_u+30fd-u+30fe_to_hiragana",
)
# The guard's token boundaries and parts of speech come from this tokenizer and
# dictionary, so their installed versions are part of the policy identity.
_FUGASHI_VERSION = importlib.metadata.version("fugashi")
_UNIDIC_LITE_VERSION = importlib.metadata.version("unidic-lite")
_WORDS_TRANSFORMS = (
    *_MINIMAL_TRANSFORMS,
    "remove_laughter_breath_and_interjection_chain_repeats_on_token_boundaries_"
    f"without_content_words_fugashi-{_FUGASHI_VERSION}_unidic-lite-{_UNIDIC_LITE_VERSION}",
)


def _require_nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _freeze_transforms(values: Iterable[object]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("transforms must be an iterable of strings")
    try:
        transforms = tuple(values)
    except TypeError as exc:
        raise TypeError("transforms must be an iterable of strings") from exc
    if not transforms:
        raise ValueError("transforms must not be empty")
    for index, transform in enumerate(transforms):
        _require_nonempty_string(transform, f"transforms[{index}]")
    return transforms  # type: ignore[return-value]


def _policy_sha256(version: str, transforms: tuple[str, ...]) -> str:
    return canonical_sha256(
        {
            "version": version,
            "transforms": transforms,
            "unicodedata_version": unicodedata.unidata_version,
        }
    )


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """A versioned transform sequence bound to the runtime Unicode tables."""

    version: str
    transforms: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        version = _require_nonempty_string(self.version, "version")
        transforms = _freeze_transforms(self.transforms)
        object.__setattr__(self, "transforms", transforms)
        _require_nonempty_string(self.sha256, "sha256")
        expected = _policy_sha256(version, transforms)
        if self.sha256 != expected:
            raise ValueError(
                "sha256 does not match the normalization policy and unicodedata version"
            )


def _make_policy(version: str, transforms: tuple[str, ...]) -> NormalizationPolicy:
    return NormalizationPolicy(
        version=version,
        transforms=transforms,
        sha256=_policy_sha256(version, transforms),
    )


JA_MINIMAL_1_1 = _make_policy("ja-minimal-1.1", _MINIMAL_TRANSFORMS)
JA_LENIENT_1_1 = _make_policy("ja-lenient-1.1", _LENIENT_TRANSFORMS)
JA_WORDS_1_0 = _make_policy("ja-words-1.0", _WORDS_TRANSFORMS)


def _require_text(text: object) -> str:
    if type(text) is not str:
        raise TypeError("text must be a string")
    return text


def normalize_raw(text: str) -> str:
    """Return validated text without applying a normalization transform."""

    return _require_text(text)


def normalize_minimal(text: str) -> str:
    """Apply the frozen primary Japanese scoring normalization."""

    normalized = unicodedata.normalize("NFKC", _require_text(text))
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and unicodedata.category(character)[0] not in ("P", "S")
    )


def normalize_lenient(text: str) -> str:
    """Apply the frozen kana/case sensitivity-analysis normalization."""

    characters: list[str] = []
    for character in normalize_minimal(text):
        codepoint = ord(character)
        if 0x41 <= codepoint <= 0x5A:
            codepoint += 0x20
        if 0x30A1 <= codepoint <= 0x30F6 or codepoint in (0x30FD, 0x30FE):
            codepoint -= 0x60
        characters.append(chr(codepoint))
    return "".join(characters)


@dataclass(frozen=True, slots=True)
class VocalSoundSplit:
    """Minimal-normalized text split into its words and its wordless sounds."""

    words: str
    sound_characters: int
    chains: int


# Matching runs on katakana folded to hiragana; removal uses the original text.
_LAUGHTER_RE = re.compile(r"[あえういお]?([はひふへほ])\1+[っー]?")
# One breath unit is す/は/ふ/ひ/ほ drawn out by ー, by its own vowel (small or
# full; ほ also takes う), or by both; units may repeat and mix (すーはー).
_BREATH_RE = re.compile(
    r"(?:す(?:ー+[うぅ]*|[うぅ]+ー*)"
    r"|は(?:ー+[あぁ]*|[あぁ]+ー*)"
    r"|ふ(?:ー+[うぅ]*|[うぅ]+ー*)"
    r"|ひ(?:ー+[いぃ]*|[いぃ]+ー*)"
    r"|ほ(?:ー+[おぉう]*|[おぉう]+ー*))+"
)
_SOUND_CHARACTERS = frozenset("あいうえおぁぃぅぇぉはひふへほすっー")
_CONTENT_POS1 = frozenset({"名詞", "動詞", "形容詞", "形状詞", "代名詞", "連体詞"})
_INTERJECTION_POS1 = "感動詞"
# A drawn-out particle (今日はー) is speech, so no breath starts on one.
_PARTICLE_POS1 = "助詞"
# UniDic splits a drawn-out です/ます as で/ま + すう, so no breath starts there.
_COPULA_STEMS = ("で", "ま")
_CHAIN_MINIMUM = 3


@functools.cache
def _tagger() -> fugashi.Tagger:
    # Loaded on first use so importing the policies does not load a dictionary.
    import fugashi
    import unidic_lite

    tagger = fugashi.Tagger()
    loaded = Path(tagger.dictionary_info[0]["filename"]).resolve().parent
    if loaded != Path(unidic_lite.DICDIR).resolve():
        raise RuntimeError("the words policy requires the unidic-lite dictionary")
    return tagger


def _fold_katakana(text: str) -> str:
    return "".join(
        chr(ord(character) - 0x60)
        if 0x30A1 <= ord(character) <= 0x30F6
        else character
        for character in text
    )


def split_vocal_sounds(text: str) -> VocalSoundSplit:
    """Remove laughter, breaths, and interjection-chain repeats from a line.

    Sounds are found in the minimal-normalized text, so texts with the same
    minimal form always split the same way. A laugh or breath is removed only
    when it starts and ends on token boundaries and covers no content word; a
    breath never starts on a particle or on the す of a drawn-out です/ます. Of
    three or more identical interjection tokens in a row, the first stays.
    """

    minimal = normalize_minimal(text)
    if not minimal:
        return VocalSoundSplit(words="", sound_characters=0, chains=0)
    tokens: list[tuple[int, int, str, str]] = []
    cursor = 0
    for word in _tagger()(minimal):
        start = cursor + len(word.white_space)
        end = start + len(word.surface)
        if minimal[start:end] != word.surface:
            raise RuntimeError("a token does not match the minimal-normalized text")
        tokens.append((start, end, _fold_katakana(word.surface), word.feature.pos1))
        cursor = end

    removed = [False] * len(tokens)
    index = 0
    while index < len(tokens):
        longest = None
        span = ""
        breath_may_start = tokens[index][3] != _PARTICLE_POS1 and not (
            tokens[index][2].startswith("す")
            and index > 0
            and tokens[index - 1][2].endswith(_COPULA_STEMS)
        )
        for last in range(index, len(tokens)):
            _, _, folded, pos1 = tokens[last]
            if pos1 in _CONTENT_POS1 or not set(folded) <= _SOUND_CHARACTERS:
                break
            span += folded
            if _LAUGHTER_RE.fullmatch(span) or (
                breath_may_start and _BREATH_RE.fullmatch(span)
            ):
                longest = last
        if longest is None:
            index += 1
            continue
        removed[index : longest + 1] = [True] * (longest + 1 - index)
        index = longest + 1

    chains = 0
    index = 0
    while index < len(tokens):
        last = index
        if not removed[index] and tokens[index][3] == _INTERJECTION_POS1:
            while (
                last + 1 < len(tokens)
                and not removed[last + 1]
                and tokens[last + 1][2:] == tokens[index][2:]
            ):
                last += 1
        if last + 1 - index >= _CHAIN_MINIMUM:
            # The first instance stays a word; each repeat after it is sound.
            removed[index + 1 : last + 1] = [True] * (last - index)
            chains += 1
        index = last + 1

    pieces: list[str] = []
    cursor = 0
    for (start, end, _, _), is_removed in zip(tokens, removed, strict=True):
        if is_removed:
            pieces.append(minimal[cursor:start])
            cursor = end
    pieces.append(minimal[cursor:])
    words = "".join(pieces)
    return VocalSoundSplit(
        words=words,
        sound_characters=len(minimal) - len(words),
        chains=chains,
    )


def normalize_words(text: str) -> str:
    """Apply the minimal normalization, then remove wordless vocal sounds."""

    return split_vocal_sounds(text).words


__all__ = (
    "JA_LENIENT_1_1",
    "JA_MINIMAL_1_1",
    "JA_WORDS_1_0",
    "NormalizationPolicy",
    "VocalSoundSplit",
    "normalize_lenient",
    "normalize_minimal",
    "normalize_raw",
    "normalize_words",
    "split_vocal_sounds",
)
