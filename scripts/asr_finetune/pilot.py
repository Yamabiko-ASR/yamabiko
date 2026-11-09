"""Paths, source registry and text helpers of the ASR fine-tuning pilot
``ft-pilot-v1``.

``<pilot>`` is ``<bench root>\\ft-pilot-v1`` (``pilot_root()``): the pilot's
private files (outbox, hygiene lists, candidates, jobs, models).

Sources are pinned here: Hugging Face repos by revision, files on the
ReazonSpeech corpus server by path and byte size. ``SOURCES`` is the training
data the pods download (``scripts.asr_finetune.fetch``), each with its byte
size (and the Common Voice archive and every ReazonSpeech mirror file with
their SHA-256), checked on 2026-10-27 and 2026-10-28 by HEAD or public (for
the gated mirror, token-read) metadata; the OOPPEENN archives come from the
galgame allow-list with their sizes.
ReazonSpeech comes from the corpus server (``reazon-small-tsv``,
``reazon-NNN``, the default) or from its Hugging Face mirror
(``reazon-mirror-all-tsv``, ``reazon-mirror-NNN``): the shards land as the
same ``NNN.tar`` either way, the captions as ``small.tsv`` or ``all.tsv``.

Every hygiene comparison uses ``norm`` (``ja-minimal-1.1``: NFKC, then
whitespace, punctuation and symbols removed); ``BenchLineFilter`` is rule 4.2
of the data survey. ``clean_text`` is the label rule of every training text.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.asr_bench.lexicon import (
    INTERJECTION_POS1,
    INTERJECTIONS,
    VOCAL_POS1,
    analyze,
    chunk_tags,
    interjection_key,
)
from scripts.asr_bench.sets import bench_root
from scripts.e0_evidence.normalize_ja import JA_MINIMAL_1_1, normalize_minimal

PILOT = "ft-pilot-v1"
NORMALIZATION = JA_MINIMAL_1_1.version


def pilot_root() -> Path:
    return bench_root() / PILOT


# Held-out evaluation sources (scripts.asr_finetune.eval_sets); never training data.
# IngCrowd dialogue corpora (ungated, CC BY-NC 4.0): (repo id, revision).
INGCROWD = (("IngCrowd/japanese-themed-dialogue", "e2eed8769400269f20821deb0e47e68b40012e8b"),
            ("IngCrowd/japanese-roleplay-travel-agency", "c213f3e042ba65a05b18b8f32c32d8d17c2766e4"))

# ReazonSpeech v2 on its corpus server (no credentials): (path under the base, bytes).
REAZON_BASE = "https://corpus.reazon-research.org/reazonspeech-v2/"
REAZON_SMALL_TSV = ("tsv/small.tsv", 6_049_679)
REAZON_ALL_TSV = ("tsv/all.tsv", 2_138_723_608)
REAZON_TEST_SHARD = ("data/fff.tar", 604_078_080)

# Common Voice splits that never train (scripts.asr_finetune.hygiene, text only).
# Mirrors of later releases whose ja dev.tsv and test.tsv are dropped, beside
# the bench's own 17.0 pin (scripts.asr_bench.build_synthetic): (repo id, revision).
CV_SPLIT_MIRRORS = (("fsicoli/common_voice_19_0", "590c8abec6cf7c8d06e650f1438e60332a796e11"),
                    ("fsicoli/common_voice_21_0", "6516e5e8eef5f4dee99f431f502959ac78821909"),
                    ("fsicoli/common_voice_22_0", "ae911de250fe9375d08e2daa2105b671d659d446"))
# Common Voice 8.0 ja test, the 760M's test set: (repo id, revision, rows).
CV8_TEST = ("japanese-asr/ja_asr.common_voice_8_0", "bf8819e8d9a5feb51b0c718686bd20ea67a3c729", 4483)

# Training source: the OOPPEENN visual-novel set, the original repo (its name is
# hex for VisualNovel_Dataset), not a copy: (repo id, revision).
OOPPEENN = ("OOPPEENN/56697375616C4E6F76656C5F44617461736574", "f69468368df70700f7abb8d73c957d4f8cf0274d")
# The archives' decompression password, base64-encoded as README.md at that
# revision publishes it.
OOPPEENN_PASSWORD_B64 = "OWxsOUtlNGlxMGpxeXEzZ1MxV3k="


@dataclass(frozen=True)
class Source:
    """One training download: its name, the file it becomes under the pods'
    data folder, its byte size, and where it comes from: a public URL; a
    file of a gated Hugging Face dataset repo (``hub``: repo id, revision,
    path in the repo), fetched with the task's ``HF_TOKEN``; or (both None)
    the presigned Mozilla Data Collective URL a fetch task receives as its
    ``MDC_URL`` secret."""

    name: str
    file: str
    bytes: int
    url: str | None
    sha256: str | None = None
    hub: tuple[str, str, str] | None = None


# Shards 000-007 of ReazonSpeech v2 ``small`` (12 shards, 000-00b): name -> bytes.
REAZON_SHARDS = {"000": 605_306_880, "001": 633_180_160, "002": 608_174_080, "003": 602_951_680,
                 "004": 595_783_680, "005": 622_428_160, "006": 616_529_920, "007": 612_249_600}
# ReazonSpeech v2 mirrored on the Hugging Face Hub (gated, automatic), used while
# the corpus server is down: (repo id, revision). Its data/NNN.tar have the
# corpus server's byte sizes and its all.tsv the size of REAZON_ALL_TSV; it has
# no small.tsv. all.tsv holds the captions of every shard: for shards 000-007
# the same 43,166 keys with the same captions as small.tsv (checked 2026-10-28).
REAZON_MIRROR = ("litagin/reazon-speech-v2-clone", "ddff3829fb26a84b2e10c352779e8f013073ece4")
# LFS SHA-256 of the mirror's files at that revision (Hugging Face API, read
# with the token on 2026-10-28); shard 000's equals the data survey's download
# from the corpus server.
REAZON_MIRROR_ALL_TSV_SHA256 = "47feaa21ea4cd50f3a8d27f5aa7a101c82d7582cb9dc04fca1e14c68172db1f5"
REAZON_MIRROR_SHA256 = {
    "000": "8deb1c9d7dcc49bfd2880b4fb3c0a930e39803d4dc84063be71c29ab4c621162",
    "001": "5ca6ce2f0127a28afd220cb9cf0b994406313a92a61910029f8310c007732b14",
    "002": "ad5b2173e1460ce687c3a6ca3025d785b5df90a3cb2f228b314ae23b502043d9",
    "003": "b1660d72981cb7cace4e5de04f72cec9dd38edd75e30a184d3bbba7645faa501",
    "004": "b1be07b99890550d58bc94b40e2746bb92349be89df797572734bcd3c9bbc689",
    "005": "091b462918d74c664fa6b6774e59458bde9ade6d89bd7d52debe4bb8e15be570",
    "006": "7b74ee5c3190674fa7cc3f5275444caf0f5a434cad66a192c6321d48a5735720",
    "007": "e802c0f2aedc75a829baebc11f7968033e292ce4b32af533e1a7de7ecc5b2ab7",
}
# Common Voice ja 27.0 on the Mozilla Data Collective (public dataset record).
CV_DATASET = "cmu5xbhnf00fmnq07z9p3aa70"
# MUSAN (OpenSLR SLR17), non-speech audio for the no-speech windows (round 4;
# scripts.asr_finetune.musan): CC BY 4.0 as a corpus, its music and noise
# sources under their own per-clip terms. The original musan.tar.gz on a
# Hugging Face mirror: (repo id, revision); its byte count equals OpenSLR's,
# and a second mirror (EaseZh/musan) has the same LFS SHA-256 (2026-10-28).
MUSAN_MIRROR = ("huseinzol05/musan-mirror", "f52353cc256ad55582fd64674c2bb568e65d5f7e")
MUSAN_BYTES = 11_086_114_085
MUSAN_SHA256 = "86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1"
SOURCES = {source.name: source for source in (
    Source("reazon-small-tsv", "small.tsv", REAZON_SMALL_TSV[1], REAZON_BASE + REAZON_SMALL_TSV[0]),
    *(Source(f"reazon-{shard}", f"{shard}.tar", size, f"{REAZON_BASE}data/{shard}.tar")
      for shard, size in REAZON_SHARDS.items()),
    Source("reazon-mirror-all-tsv", "all.tsv", REAZON_ALL_TSV[1], None, REAZON_MIRROR_ALL_TSV_SHA256,
           hub=(*REAZON_MIRROR, "all.tsv")),
    *(Source(f"reazon-mirror-{shard}", f"{shard}.tar", size, None, REAZON_MIRROR_SHA256[shard],
             hub=(*REAZON_MIRROR, f"data/{shard}.tar"))
      for shard, size in REAZON_SHARDS.items()),
    Source("cv", "common-voice-scripted-speech-27-0-japane-ea5ad4e9.tar.gz", 15_417_014_005, None,
           "dba2e11b7b1365360a5f1f31a97cd83410760b9071734c0376d896428c299882"),
    Source("jvnv", "jvnv_ver1.zip", 1_778_635_432, "https://ss-takashi.sakura.ne.jp/corpus/jvnv/jvnv_ver1.zip"),
    Source("jnv", "jnv_corpus_ver3.zip", 325_502_055, "https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip"),
    Source("musan", "musan.tar.gz", MUSAN_BYTES, None, MUSAN_SHA256, hub=(*MUSAN_MIRROR, "musan.tar.gz")),
)}
# Where ReazonSpeech comes from: origin -> (its caption source, its shard
# source pattern). A plan names at most one caption source.
REAZON_ORIGINS = {"official": ("reazon-small-tsv", "reazon-{shard}"),
                  "mirror": ("reazon-mirror-all-tsv", "reazon-mirror-{shard}")}
REAZON_CAPTION_SOURCES = tuple(captions for captions, _ in REAZON_ORIGINS.values())
_REAZON_SHARD = re.compile(r"reazon-(?:mirror-)?([0-9a-f]{3})")


def reazon_shard(name: str) -> str | None:
    """The shard (``000``) a ``reazon-NNN`` or ``reazon-mirror-NNN`` source holds, else None."""
    match = _REAZON_SHARD.fullmatch(name)
    return match[1] if match else None


def reazon_sources(origin: str, shards: Iterable[str]) -> list[str]:
    """The sources of ``origin`` (``REAZON_ORIGINS``) for ``shards``: the captions, then each shard."""
    captions, pattern = REAZON_ORIGINS[origin]
    return [captions, *(pattern.format(shard=shard) for shard in shards)]


# The gated loader script of ReazonSpeech on Hugging Face, a 4,858-byte probe
# of a fine-grained token: (repo id, revision, file name, bytes).
REAZON_LOADER = ("reazon-research/reazonspeech", "0df78f991f6afa051aacb7c63e3fe2b496c13c2c", "reazonspeech.py", 4858)

# Seed of every sampling step of the pilot.
SEED = 20261027

# Training-text sources (``clean_text``).
TEXT_SOURCES = ("reazon", "cv", "galgame", "jvnv", "jnv")
# Drop reasons of ``clean_reason``, in the order they are checked.
CLEAN_REASONS = ("markup", "empty", "vocal", "phantom")
# The family the 760M invents on breaths (acceptance A6); its likely source
# is galgame training data [INFERENCE], so galgame and JNV lines holding it
# never train, word lines included.
PHANTOM_FAMILY = re.compile(r"[ちぢじ][ゅュ][るっぱぷ]?[っッ]?|ぢゅ|んぢゅ|じゅる")
_PHANTOM_SOURCES = frozenset({"galgame", "jnv"})
_MARKUP = frozenset("<>[]{}\\")
# NFKC turns （） into ().
_WRAPPING = (("「", "」"), ("『", "』"), ("(", ")"))


def norm(text: str) -> str:
    """``ja-minimal-1.1`` (``normalize_minimal``): NFKC, then whitespace,
    punctuation and symbols removed."""
    return normalize_minimal(text)


def sentence_hash(text: str) -> str:
    """SHA-256 (hex) of the UTF-8 bytes of ``norm(text)``."""
    return hashlib.sha256(norm(text).encode()).hexdigest()


def _unwrap(text: str) -> str:
    """``text`` without one pair of ``_WRAPPING`` brackets that encloses all of it."""
    for opener, closer in _WRAPPING:
        if len(text) < 2 or text[0] != opener or text[-1] != closer:
            continue
        depth = 0
        for index, char in enumerate(text):
            depth += (char == opener) - (char == closer)
            if depth == 0:
                return text[1:-1].strip() if index == len(text) - 1 else text
        return text
    return text


def _clean(source: str, text: str) -> tuple[str | None, str | None]:
    if source not in TEXT_SOURCES:
        raise ValueError(f"source must be one of {', '.join(TEXT_SOURCES)}, not {source!r}")
    cleaned = _unwrap(unicodedata.normalize("NFKC", text).strip())
    if any(char in _MARKUP for char in cleaned):
        return None, "markup"
    if not norm(cleaned):
        return None, "empty"
    if source == "galgame":
        analysis = analyze(cleaned)
        if analysis.vocal and not analysis.interjections:
            return None, "vocal"
    if source in _PHANTOM_SOURCES and PHANTOM_FAMILY.search(cleaned):
        return None, "phantom"
    return cleaned, None


def clean_text(source: str, text: str) -> str | None:
    """A training label from a source text (``source`` one of
    ``TEXT_SOURCES``): NFKC, stripped, without one pair of 「」, 『』 or （）
    around all of it; None when ``clean_reason`` names a reason."""
    return _clean(source, text)[0]


def clean_reason(source: str, text: str) -> str | None:
    """Why ``clean_text`` drops a text, or None when it keeps it. In order:
    ``markup`` (a character of ``<>[]{}\\`` after NFKC), ``empty`` (nothing
    left after ``norm``), ``vocal`` (galgame only: vocalization-only without a
    lexicon interjection, ``scripts.asr_bench.lexicon.analyze``: moans,
    grunts and NSFW-only sounds), ``phantom`` (galgame and JNV: a match of
    ``PHANTOM_FAMILY`` anywhere in the text)."""
    return _clean(source, text)[1]


# Loop hygiene (round 4): a label the model could learn to loop on.
LOOP_RUN = 4
LOOP_RATIO = 2.0


def loop_reason(text: str) -> str | None:
    """Why a cleaned label could teach a loop, or None: ``loop_run`` (a run of
    ``LOOP_RUN`` or more identical characters, punctuation and whitespace
    aside, so ``……`` and ``！！！！`` pass while ``ああああ``, ``ーーーー`` or
    ``～～～～`` do not) or ``loop_ratio`` (zlib compression ratio of its UTF-8
    bytes over ``LOOP_RATIO``, Whisper's measure of a repeated text)."""
    run, previous = 0, ""
    for char in text:
        if unicodedata.category(char)[0] in "PZC":
            run, previous = 0, ""
            continue
        run = run + 1 if char == previous else 1
        previous = char
        if run >= LOOP_RUN:
            return "loop_run"
    data = text.encode("utf-8")
    return "loop_ratio" if data and len(data) / len(zlib.compress(data)) > LOOP_RATIO else None


# Adult-scene filter (round 4, v4c; galgame only). The fine-tune learned to
# write these sound effects and moans over noisy, breathy or loud audio.
# (a) Sound-effect vocabulary, matched on the NFKC text with katakana folded
# to hiragana: wet and kissing sounds, slaps, the moan forms あんっ, んんっ,
# はぁんっ, ひぎ, おほぉ, and climax and ejaculation words.
ADULT_SFX = re.compile(
    "ぢゅ|じゅぶ|じゅぷ|じゅる|じゅぽ|ちゅぷ|ちゅぱ|ちゅぽ|ちゅる|れろ|くちゅ|ぬちゅ|ぬぷ|ずちゅ|ぐちゅ|"
    "ずぶ|ずぷ|ぱんっ|あんっ|んんっ|はぁんっ|ひぎ|おほぉ|いくっ|びゅる|びゅく|どぴゅ|ぴゅる")
# ... and katakana spellings matched as written (the folded いっちゃ is also
# ordinary speech, 行っちゃう).
ADULT_SFX_KATAKANA = re.compile("イクッ|イくっ|イっちゃ|イッちゃ")
# Round 6 (v6a; applied with --vocal-cap, so earlier outboxes rebuild as
# they were): more sound effects (フェラ; ヂュ, ヂュポ, チュポ and ジュポ are
# already ぢゅ, ちゅぽ and じゅぽ once folded, as is every katakana spelling
# of the kana list above) and explicit words, on the same folded text.
ADULT_SFX_R6 = re.compile(
    "ふぇら|ちゅっ|ぺろぺろ|くぱぁ|ずぽ|ぬぽ|ぴゅっ|ぬちゃ|ぐちょ|しこしこ|"
    "ちんぽ|ちんこ|まんこ|ま◯こ|ま○こ|ま●こ|おちんちん|せーえき")
# (b) A moan-dense line: at least ADULT_MOAN_CHUNKS vocalization chunks, and
# vocalizations at least ADULT_MOAN_SHARE of its reading. A chunk (a piece
# between punctuation, symbols or spaces) is a vocalization when all its
# tokens are UniDic interjections or symbols, or it is at most 8 characters
# of MOAN_KANA (はぁ, ふぁ, くぅん ...). A chunk that is a lexicon interjection
# (はい, うん, えっ, あの, あっ, ああ ...) is never one, and a line holding one
# is not moan-dense: those lines are the positive ones the pilot trains on
# (the moans among them still meet (a) when they carry its vocabulary).
# Counting あっ and ああ as moans dropped 13% of the selected positive lines.
ADULT_MOAN_CHUNKS = 2
ADULT_MOAN_SHARE = 0.40
MOAN_KANA = frozenset("あいうえおぁぃぅぇぉんっはひふへほゃゅょやゆよくぐー〜～")
_KATAKANA_TO_HIRAGANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


def _vocal_chunk(chunk: str, tokens: Iterable[tuple[str, str, str]], cap: int | None = 8) -> bool:
    if all(pos1 in VOCAL_POS1 for _, _, pos1 in tokens):
        return True
    folded = chunk.translate(_KATAKANA_TO_HIRAGANA)
    return (cap is None or len(folded) <= cap) and all(char in MOAN_KANA for char in folded)


def adult_reason(text: str, round6: bool = False) -> str | None:
    """Why a cleaned galgame label belongs to an adult scene, or None:
    ``adult_sfx`` (``ADULT_SFX`` or ``ADULT_SFX_KATAKANA``, with ``round6``
    also ``ADULT_SFX_R6``) or ``adult_moan`` (moan-dense, see
    ``ADULT_MOAN_CHUNKS``)."""
    normalized = unicodedata.normalize("NFKC", text)
    folded = normalized.translate(_KATAKANA_TO_HIRAGANA)
    if (ADULT_SFX.search(folded) or ADULT_SFX_KATAKANA.search(normalized)
            or (round6 and ADULT_SFX_R6.search(folded))):
        return "adult_sfx"
    vocal = total = chunks = 0
    for chunk, tokens in chunk_tags(text):
        if interjection_key("".join(piece for _, piece, _ in tokens)) in INTERJECTIONS:
            return None
        size = sum(len(piece) for _, piece, _ in tokens)
        total += size
        if _vocal_chunk(chunk, tokens):
            vocal += size
            chunks += 1
    return "adult_moan" if chunks >= ADULT_MOAN_CHUNKS and vocal >= ADULT_MOAN_SHARE * total else None


# Long vocal chains (round 6, v6a, select --vocal-cap; every source). The
# fine-tune inserted long vocal chains (ウオオオォォッ..., エヘヘッ x10,
# panting) into speech. The vocal chunks of a line, lexicon interjections
# of vocalization kana included (あっ, ああ, うん, はい): a chunk of
# VOCAL_KANA only (MOAN_KANA plus す, for panting; no length cap) unless
# UniDic finds a word in it (VOCAL_WORD_POS1: いいよ, ぜんぜん), else each
# run of consecutive UniDic symbol tokens and interjections of VOCAL_KANA
# holding such an interjection (a chain followed by クソッ is still a
# chain; word-like interjections such as あの, えっと, ありがとう or よし are
# none); a lone cut-off っ is none. A vocal chunk's size is its reading in
# kana (UniDic, long-vowel marks dropped).
# A line is dropped as
# ``long_vocal_total`` (a) when its vocal chunks hold more than
# VOCAL_CAP_TOTAL kana, ``long_vocal_dense`` (b) with VOCAL_CAP_CHUNKS or
# more vocal chunks holding VOCAL_CAP_DENSE_KANA or more kana and
# ADULT_MOAN_SHARE or more of its reading (えっ?あっ?はい! stays),
# ``long_vocal_chunk`` (c) when one vocal chunk holds VOCAL_CAP_CHUNK or
# more kana, ``long_vocal_repeat`` (d) when one vocal reading comes
# VOCAL_CAP_REPEAT or more times. はい, うん, えっ, あっ、そうなんだ, うん、うん
# and ええっ!? stay.
VOCAL_CAP_TOTAL = 8
VOCAL_CAP_CHUNKS = 3
VOCAL_CAP_DENSE_KANA = 7
VOCAL_CAP_CHUNK = 6
VOCAL_CAP_REPEAT = 3
VOCAL_KANA = MOAN_KANA | frozenset("す")
VOCAL_WORD_POS1 = frozenset({"動詞", "形容詞", "形状詞", "助詞", "助動詞", "代名詞", "副詞", "接続詞", "連体詞"})


def _vocal_token(token: tuple[str, str, str]) -> bool:
    _, piece, pos1 = token
    if pos1 != INTERJECTION_POS1:
        return pos1 in VOCAL_POS1
    return all(char in VOCAL_KANA for char in piece.translate(_KATAKANA_TO_HIRAGANA))


def _vocal_runs(chunk: str, tokens: Sequence[tuple[str, str, str]]) -> list[str]:
    """The readings of a chunk's vocal chunks (see VOCAL_CAP_TOTAL)."""
    reading = "".join(piece for _, piece, _ in tokens)
    if not reading.strip("ッ"):
        return []
    if (all(char in VOCAL_KANA for char in chunk.translate(_KATAKANA_TO_HIRAGANA))
            and not any(pos1 in VOCAL_WORD_POS1 for _, _, pos1 in tokens)):
        return [reading]
    runs: list[str] = []
    for vocal, run in itertools.groupby(tokens, key=_vocal_token):
        run = list(run)
        if vocal and any(pos1 == INTERJECTION_POS1 for _, _, pos1 in run):
            runs.append("".join(piece for _, piece, _ in run))
    return runs


def vocal_reason(text: str) -> str | None:
    """Why a cleaned label holds a long vocal chain (``long_vocal_*``), or None."""
    readings: list[str] = []
    total = 0
    for chunk, tokens in chunk_tags(text):
        total += sum(len(piece) for _, piece, _ in tokens)
        readings += _vocal_runs(chunk, tokens)
    vocal = sum(map(len, readings))
    if vocal > VOCAL_CAP_TOTAL:
        return "long_vocal_total"
    if len(readings) >= VOCAL_CAP_CHUNKS and vocal >= max(VOCAL_CAP_DENSE_KANA, ADULT_MOAN_SHARE * total):
        return "long_vocal_dense"
    if readings and max(map(len, readings)) >= VOCAL_CAP_CHUNK:
        return "long_vocal_chunk"
    if readings and max(readings.count(reading) for reading in readings) >= VOCAL_CAP_REPEAT:
        return "long_vocal_repeat"
    return None


class BenchLineFilter:
    """Rule 4.2 of the data survey: a training text is a hit when its
    normalized form equals or contains the normalized form of a bench
    reference line of at least ``min_chars`` characters. Shorter lines (stock
    phrases) are ignored."""

    def __init__(self, lines: Iterable[str], min_chars: int = 12) -> None:
        if min_chars < 1:
            raise ValueError("min_chars must be at least 1")
        self.min_chars = min_chars
        # Lines by their first min_chars characters: a hit is a text position
        # where one of these shingles starts the line.
        self._by_head: dict[str, set[str]] = {}
        for line in map(norm, lines):
            if len(line) >= min_chars:
                self._by_head.setdefault(line[:min_chars], set()).add(line)

    def __len__(self) -> int:
        """The number of distinct lines the filter matches."""
        return sum(map(len, self._by_head.values()))

    def hit(self, text: str) -> bool:
        normalized = norm(text)
        width = self.min_chars
        for start in range(len(normalized) - width + 1):
            lines = self._by_head.get(normalized[start:start + width])
            if lines and any(normalized.startswith(line, start) for line in lines):
                return True
        return False
