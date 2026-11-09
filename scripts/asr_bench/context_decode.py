"""Context decoding: the ``context_decode`` window strategy and its timing check.

Each VAD window ``[s, e)`` is decoded from a longer span of the item's own
audio, and only the text aligned inside the window is kept:

- **Left context** ``left_s``: real audio from ``s - left_s`` (clipped to
  the item start). With ``text: prefix`` the span instead starts at the end
  of an earlier window that ``s - left_s`` falls inside, so the forced text
  matches the audio.
- **Right context** ``right_s``: real audio up to ``e + right_s`` (clipped
  to the end of the item and the runner's zero tail). The window's
  ``emitted_at_s`` becomes the later of the chunker's emission and the end
  of the real audio the decode used, so line delay includes the wait.
- **Pad** ``pad``: the span's outer edges get ``asr.boundary_pad_ms`` of
  digital silence (``silence``, production's pad) or of the neighbouring
  real audio (``audio``; zeros only where the item has none), which also
  counts toward the wait.
- **Text** ``text``: the system's own earlier output in the same item, never
  a reference line. ``none``: the style prompt alone (production's
  ``initial_prompt``). ``prompt``: the style prompt followed by the texts of
  the last ``prompt_windows`` earlier windows with text, after
  ``<|startofprev|>``. ``prefix``: the texts of the earlier windows starting
  inside the span are forced as decoder tokens after the start sequence
  (encoded without a leading space), at most ``PREFIX_MAX_TOKENS`` of them
  (the oldest windows are dropped first, and the span starts at the first
  window kept).

Decode. The production ``FasterWhisperASR`` the runner built is reused (one
copy of the weights): its faster-whisper ``WhisperModel`` computes the
features, its CTranslate2 ``Whisper`` runs ``encode``, ``generate`` (through
faster-whisper's ``generate_with_fallback``: production's beam size,
temperature fallback and its compression, log-probability and no-speech
thresholds) and ``align``. The span is processed in 30 s encoder windows the
way faster-whisper's ``generate_segments`` does with
``condition_on_previous_text=False``: a pass whose no-speech probability is
above 0.6 with an average log-probability of -1.0 or less is skipped, and
after a pass that ends on two timestamps the next pass starts at the last
one; passes after the first carry no prompt and no prefix. ``none`` and
``prompt`` decode with timestamps, as production does; ``prefix`` decodes
without them, because with timestamps the model closes the forced segment
and stops (seen on silver-v4). The text tokens of each pass are aligned to
the audio by ``Whisper.align`` (faster-whisper's ``find_alignment``:
alignment heads, median filter of 7, DTW), giving each word (a token group
of whole characters) a start and an end time.

Keep rule. A word is kept when its aligned end time ``t`` (item seconds)
satisfies ``lower < t <= upper``. On a side where the view heard real audio
beyond the window, the bound is the window edge widened by ``margin_s``
(``lower = s - margin_s``, ``upper = e + margin_s``); on a side where it
heard only its silence pad, there is no bound, since only the window's own
words can lie there (DTW slop around its first or last sound). ``lower`` is
then raised to the claim of the item's previous window where that claim
reaches past ``s``; a window's claim is ``e + margin_s`` capped at the end
of the real audio its committed view heard (``e`` itself after a silence
pad without right context). So a word the previous window heard in its
right context and kept is not kept again, while a word cut by a split
between adjacent windows (DTW often ends it just before ``s``) stays with
the later window, at the risk of a repeated character. Forced prefix words
are never output. The end time is used because DTW assigns a silence to
the word after it, so a word's start can lie a whole pause before its
sound while its end lands on it. Then production's compression gate (30
characters or more and a zlib compression ratio above 2.4, here of the
kept text) applies. A view without context (``left_s`` and ``right_s`` 0,
silence pad) therefore keeps every word, like production.

Continuation. When a view keeps no word and its last decoded word ends at
or before ``s`` (the model ended the span before a short last utterance,
which Whisper does often), the view decodes once more from that word's end
to the same right edge, with a silence pad in front, timestamps, and its
text as a prompt (``prefix`` text becomes a prompt), and the keep rule
applies to that decode.

Agreement (``agreement: [view, ...]``, at least three views). Each view is
decoded and cut as above. Two views agree when their kept texts have the
same reading (``lexicon.reading`` without ッ) up to ``similarity`` (one
minus the edit distance over the longer reading; two empty texts agree, an
empty and a non-empty one do not). The committed view is the one agreeing
with the most other views, the later view on ties; when no two views agree,
the last view (meant to be the one with the most context) is committed.
Every view is waited for, so ``emitted_at_s`` is the latest of their audio
ends.

Timing check (``python -m scripts.asr_bench.context_decode timing``):
decodes fixed spans of a few items the way a ``text: none`` view does,
aligns the decoded characters with the reference lines (one minimum-edit
alignment per item, ``align.alignment_trace`` on minimal-normalized text)
and reports how far the aligned word and segment times fall from the
lines' start and end times.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench.align import alignment_trace
from scripts.asr_bench.sets import SAMPLE_RATE_HZ

MODE = "context_decode"
TEXT_MODES = ("none", "prompt", "prefix")
PAD_MODES = ("silence", "audio")
MAX_LEFT_S = 20.0
MAX_RIGHT_S = 5.0
# The aligned end of a window's first character falls up to about 0.44 s
# from the line onset at the 90th percentile (timing check on silver-v4).
DEFAULT_MARGIN_S = 0.4
DEFAULT_PROMPT_WINDOWS = 2
DEFAULT_SIMILARITY = 0.75
MIN_AGREEMENT_VIEWS = 3
# Whisper's decoder holds 448 tokens; faster-whisper keeps at most half of
# them for the previous-text prompt.
_MAX_LENGTH = 448
_MAX_PROMPT_TOKENS = _MAX_LENGTH // 2 - 1
# A forced prefix leaves room for the style prompt and the window's text.
PREFIX_MAX_TOKENS = 160
# Outcome of a view whose decode had words but none inside the window.
OUTSIDE_WINDOW = "outside_window"
_VIEW_KEYS = frozenset({"left_s", "right_s", "text", "pad"})
_COMMON_KEYS = frozenset({"mode", "margin_s", "prompt_windows"})


@dataclass(frozen=True)
class View:
    """One way of decoding a window: its context audio and text."""

    left_s: float = 0.0
    right_s: float = 0.0
    text: str = "none"
    pad: str = "silence"

    @property
    def label(self) -> str:
        return f"L{self.left_s:g}-{self.text}-R{self.right_s:g}-{self.pad}"

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ContextDecodeSpec:
    """The ``strategy`` of a ``context_decode`` candidate."""

    views: tuple[View, ...]
    agreement: bool = False
    margin_s: float = DEFAULT_MARGIN_S
    prompt_windows: int = DEFAULT_PROMPT_WINDOWS
    similarity: float = DEFAULT_SIMILARITY
    mode: str = MODE

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode}
        if self.agreement:
            payload["agreement"] = [view.to_json() for view in self.views]
            payload["similarity"] = self.similarity
        else:
            payload.update(self.views[0].to_json())
        payload["margin_s"] = self.margin_s
        payload["prompt_windows"] = self.prompt_windows
        return payload


def _number(raw: Mapping[str, Any], key: str, default: float, low: float, high: float, where: str) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}.{key} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{where}.{key} must lie in [{low}, {high}]")
    return float(value)


def _view(raw: object, where: str) -> View:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object")
    unknown = set(raw) - _VIEW_KEYS
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    text = raw.get("text", "none")
    if text not in TEXT_MODES:
        raise ValueError(f"{where}.text must be one of {TEXT_MODES}")
    pad = raw.get("pad", "silence")
    if pad not in PAD_MODES:
        raise ValueError(f"{where}.pad must be one of {PAD_MODES}")
    return View(
        left_s=_number(raw, "left_s", 0.0, 0.0, MAX_LEFT_S, where),
        right_s=_number(raw, "right_s", 0.0, 0.0, MAX_RIGHT_S, where),
        text=text,
        pad=pad,
    )


def parse_strategy(raw: Mapping[str, Any], where: str) -> ContextDecodeSpec:
    """Parse a candidate's ``{"mode": "context_decode", ...}`` strategy:
    one view's keys (``left_s``, ``right_s``, ``text``, ``pad``) at the top
    level, or ``agreement``: a list of at least three view objects (with
    ``similarity``); ``margin_s`` and ``prompt_windows`` in both forms."""
    if "agreement" in raw:
        unknown = set(raw) - _COMMON_KEYS - {"agreement", "similarity"}
        if unknown:
            raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
        entries = raw["agreement"]
        if not isinstance(entries, list) or len(entries) < MIN_AGREEMENT_VIEWS:
            raise ValueError(f"{where}.agreement must list at least {MIN_AGREEMENT_VIEWS} views")
        views = tuple(_view(entry, f"{where}.agreement[{index}]") for index, entry in enumerate(entries))
        if len(set(views)) != len(views):
            raise ValueError(f"{where}.agreement lists the same view twice")
        similarity = _number(raw, "similarity", DEFAULT_SIMILARITY, 0.0, 1.0, where)
        agreement = True
    else:
        unknown = set(raw) - _COMMON_KEYS - _VIEW_KEYS
        if unknown:
            raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
        views = (_view({key: raw[key] for key in _VIEW_KEYS & set(raw)}, where),)
        similarity = DEFAULT_SIMILARITY
        agreement = False
    prompt_windows = raw.get("prompt_windows", DEFAULT_PROMPT_WINDOWS)
    if isinstance(prompt_windows, bool) or not isinstance(prompt_windows, int) or not 1 <= prompt_windows <= 8:
        raise ValueError(f"{where}.prompt_windows must be an int in [1, 8]")
    return ContextDecodeSpec(
        views=views,
        agreement=agreement,
        margin_s=_number(raw, "margin_s", DEFAULT_MARGIN_S, 0.0, 1.0, where),
        prompt_windows=prompt_windows,
        similarity=similarity,
    )


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_s: float
    end_s: float
    # Index into the span's segments (timestamp segments; one per pass
    # without timestamps).
    segment: int


@dataclass(frozen=True)
class TimedSegment:
    start_s: float
    end_s: float
    text: str


@dataclass
class Span:
    """One decoded span: its words and segments in item seconds."""

    words: list[TimedWord] = field(default_factory=list)
    segments: list[TimedSegment] = field(default_factory=list)
    passes: list[dict[str, Any]] = field(default_factory=list)
    encoder_calls: int = 0


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data))


class SpanDecoder:
    """Drives a production ``FasterWhisperASR``'s model over arbitrary spans."""

    def __init__(self, backend: Any) -> None:
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens

        from yamabiko.asr import asr_client

        model = backend._model
        self._model = model
        self.tokenizer = Tokenizer(
            model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language="ja",
        )
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(model.transcribe).parameters.items()
        }
        # Production's transcribe call: its own beam, prompt and thresholds,
        # faster-whisper's defaults for the rest.
        self.options = TranscriptionOptions(
            beam_size=backend._beam_size,
            best_of=defaults["best_of"],
            patience=defaults["patience"],
            length_penalty=defaults["length_penalty"],
            repetition_penalty=defaults["repetition_penalty"],
            no_repeat_ngram_size=defaults["no_repeat_ngram_size"],
            log_prob_threshold=asr_client._LOGPROB_THRESHOLD,
            no_speech_threshold=asr_client._NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=asr_client._COMPRESSION_RATIO_THRESHOLD,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=defaults["prompt_reset_on_temperature"],
            temperatures=list(defaults["temperature"]),
            initial_prompt=None,
            prefix=None,
            suppress_blank=defaults["suppress_blank"],
            suppress_tokens=get_suppressed_tokens(self.tokenizer, list(defaults["suppress_tokens"])),
            without_timestamps=False,
            max_initial_timestamp=defaults["max_initial_timestamp"],
            word_timestamps=False,
            prepend_punctuations=defaults["prepend_punctuations"],
            append_punctuations=defaults["append_punctuations"],
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps=[],
            hallucination_silence_threshold=None,
            hotwords=None,
        )
        self.style_prompt: str | None = backend._initial_prompt
        # faster-whisper encodes a string initial_prompt with a leading space.
        self._style_tokens = (
            self.tokenizer.encode(" " + self.style_prompt.strip()) if self.style_prompt else []
        )
        self.pad_samples = SAMPLE_RATE_HZ * int(backend._boundary_pad_ms) // 1000
        self.min_compression_len = asr_client._MIN_COMPRESSION_GATE_LEN
        self.compression_threshold = asr_client._COMPRESSION_RATIO_THRESHOLD

    def identity(self) -> dict[str, Any]:
        options = self.options
        return {
            "engine": "faster-whisper WhisperModel features and helpers over CTranslate2 Whisper encode, generate and align",
            "beam_size": options.beam_size,
            "best_of": options.best_of,
            "temperatures": list(options.temperatures),
            "compression_ratio_threshold": options.compression_ratio_threshold,
            "log_prob_threshold": options.log_prob_threshold,
            "no_speech_threshold": options.no_speech_threshold,
            "max_initial_timestamp": options.max_initial_timestamp,
            "suppress_tokens": list(options.suppress_tokens),
            "style_prompt": self.style_prompt,
            "boundary_pad_samples": self.pad_samples,
            "timestamps": {"none": True, "prompt": True, "prefix": False},
            "prefix_max_tokens": PREFIX_MAX_TOKENS,
            "prompt_max_tokens": _MAX_PROMPT_TOKENS,
            "alignment": "Whisper.align via find_alignment, median filter 7",
        }

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text) if text else []

    def decode(
        self,
        audio: np.ndarray,
        origin_s: float,
        *,
        prompt_text: str = "",
        prefix_tokens: Sequence[int] = (),
        timestamps: bool = True,
        cache: dict[Any, Any] | None = None,
        cache_key: Any = None,
    ) -> Span:
        """Decode ``audio`` (its first sample at item time ``origin_s``)."""
        from faster_whisper.audio import pad_or_trim

        model = self._model
        tokenizer = self.tokenizer
        if prefix_tokens and timestamps:
            raise ValueError("a forced prefix decodes without timestamps")
        previous = list(self._style_tokens) + self.encode_text(prompt_text)
        previous = previous[-_MAX_PROMPT_TOKENS:]
        frames_per_window = model.feature_extractor.nb_max_frames
        time_per_frame = model.feature_extractor.time_per_frame
        key = None if cache_key is None else ("features", cache_key)
        if cache is not None and key in cache:
            features = cache[key]
        else:
            features = model.feature_extractor(audio)
            if cache is not None and key is not None:
                cache[key] = features
        content_frames = features.shape[-1] - 1
        span = Span()
        seek = 0
        first = True
        while seek < content_frames:
            time_offset = seek * time_per_frame
            size = min(frames_per_window, content_frames - seek)
            encoder_key = None if cache_key is None else ("encoder", cache_key, seek)
            if cache is not None and encoder_key in cache:
                encoder_output = cache[encoder_key]
            else:
                encoder_output = model.encode(pad_or_trim(features[:, seek:seek + size]))
                span.encoder_calls += 1
                if cache is not None and encoder_key is not None:
                    cache[encoder_key] = encoder_output
            # faster-whisper with condition_on_previous_text=False: only the
            # first pass carries the prompt and the prefix.
            start = [tokenizer.sot_prev, *previous] if first and previous else []
            start += tokenizer.sot_sequence
            if not timestamps:
                start.append(tokenizer.no_timestamps)
            forced = list(prefix_tokens) if first else []
            first = False
            result, avg_logprob, temperature, ratio = model.generate_with_fallback(
                encoder_output, start + forced, tokenizer, self.options,
            )
            skipped = bool(
                result.no_speech_prob > self.options.no_speech_threshold
                and not avg_logprob > self.options.log_prob_threshold
            )
            span.passes.append({
                "offset_s": round(origin_s + time_offset, 3),
                "frames": size,
                "temperature": temperature,
                "avg_logprob": round(float(avg_logprob), 4),
                "no_speech_prob": round(float(result.no_speech_prob), 4),
                "compression_ratio": round(float(ratio), 3),
                "skipped": skipped,
            })
            if skipped:
                seek += size
                continue
            tokens = list(result.sequences_ids[0])
            # CTranslate2 returns the forced tokens after the start sequence.
            if tokens[:len(forced)] != forced:
                raise RuntimeError("generate did not return the forced prefix")
            if timestamps:
                segments, next_seek, _ = model._split_segments_by_timestamps(
                    tokenizer=tokenizer, tokens=tokens, time_offset=time_offset,
                    segment_size=size, segment_duration=size * time_per_frame, seek=seek,
                )
            else:
                segments = [{"start": time_offset, "end": time_offset + size * time_per_frame, "tokens": tokens}]
                next_seek = seek + size
            # Guard: a pass that ends on <|0.00|><|0.00|> would not advance.
            seek = next_seek if next_seek > seek else seek + size
            text_tokens: list[int] = []
            owner: list[int] = []
            for segment in segments:
                text = [token for token in segment["tokens"] if token < tokenizer.eot]
                index = len(span.segments)
                span.segments.append(TimedSegment(
                    origin_s + segment["start"], origin_s + segment["end"],
                    tokenizer.decode(text[len(forced):] if not timestamps else text),
                ))
                text_tokens.extend(text)
                owner.extend([index] * len(text))
            if not text_tokens:
                continue
            alignment = model.find_alignment(tokenizer, [text_tokens], encoder_output, size)[0]
            consumed = 0
            for word in alignment:
                begin = consumed
                consumed += len(word["tokens"])
                if begin < len(forced) or not word["word"]:
                    continue
                span.words.append(TimedWord(
                    word["word"],
                    origin_s + time_offset + float(word["start"]),
                    origin_s + time_offset + float(word["end"]),
                    owner[begin],
                ))
        return span


@dataclass(frozen=True)
class _Committed:
    start_sample: int
    end_sample: int
    # The keep region's end, but never past the real audio its committed
    # view heard: a decode that heard silence after the window end cannot
    # hold the next window's first words there.
    claimed_s: float
    text: str


@dataclass(frozen=True)
class _ViewResult:
    view: View
    text: str
    outcome: str
    audio_end_sample: int
    detail: dict[str, Any]


def agreement_key(text: str) -> str:
    """Reading used to compare views: ``lexicon.reading`` without ッ."""
    from scripts.asr_bench.lexicon import reading

    return reading(text).replace("ッ", "") if text else ""


def views_agree(first: str, second: str, similarity: float) -> bool:
    """Two agreement keys agree: both empty, or both non-empty with one
    minus their edit distance over the longer at least ``similarity``."""
    if not first or not second:
        return first == second
    distance = alignment_trace(first, second)[2]
    return 1.0 - distance / max(len(first), len(second)) >= similarity


def choose_view(keys: Sequence[str], similarity: float) -> tuple[int, list[int], bool]:
    """The committed view index, each view's agreement count, and whether
    the choice fell back to the last view (no two views agree)."""
    votes = [
        sum(1 for j, other in enumerate(keys) if j != i and views_agree(key, other, similarity))
        for i, key in enumerate(keys)
    ]
    best = max(range(len(keys)), key=lambda index: (votes[index], index))
    if votes[best] == 0:
        return len(keys) - 1, votes, True
    return best, votes, False


class ContextDecodeStrategy:
    """The runner's ``context_decode`` window strategy."""

    def __init__(self, context: Any) -> None:
        spec = context.candidate.strategy
        if not isinstance(spec, ContextDecodeSpec):
            raise TypeError("context_decode needs a ContextDecodeSpec strategy")
        if context.backend is None:
            raise ValueError("context_decode drives the faster_whisper backend only")
        self._spec = spec
        self._decoder = SpanDecoder(context.backend)
        self._audio = np.zeros(0, np.float32)
        self._committed: list[_Committed] = []
        self._outcomes: dict[str, Counter] = {view.label: Counter() for view in spec.views}
        self._passes: Counter = Counter()
        self._continuations: Counter = Counter()
        self._encoder_calls = 0
        self._commits: Counter = Counter()
        self._waits: list[float] = []

    def start_item(self, item: Any, asr_audio: np.ndarray, probabilities: np.ndarray) -> None:
        self._audio = asr_audio
        self._committed = []

    def _prompt_text(self) -> str:
        texts = [window.text for window in self._committed if window.text]
        return "".join(texts[-self._spec.prompt_windows:])

    def _prefix(self, left: int, start: int) -> tuple[int, str, list[int], int]:
        """Span start, forced text and tokens, and window count for ``text: prefix``."""
        for window in self._committed:
            if window.start_sample < left < window.end_sample:
                left = window.end_sample
        inside = [window for window in self._committed if window.start_sample >= left]
        while True:
            text = "".join(window.text for window in inside)
            tokens = self._decoder.encode_text(text)
            if len(tokens) <= PREFIX_MAX_TOKENS:
                break
            inside = inside[1:]
            left = inside[0].start_sample if inside else start
        return left, text, tokens, len(inside)

    def _keep(
        self, span: Span, begin: int, finish: int, start: int, end: int,
        previous_claim_s: float | None, upper_s: float,
    ) -> tuple[list[TimedWord], float, float]:
        """The words of ``span`` (real audio ``begin`` to ``finish``) kept for
        the window ``start`` to ``end``, and the bounds used."""
        rate = SAMPLE_RATE_HZ
        # A side where the view heard only its silence pad holds no other
        # window's words, so it is not cut. The previous window's claim
        # raises the lower bound only where it reaches into this window
        # (that window heard real audio past its end and kept its words).
        lower_s = start / rate - self._spec.margin_s if begin < start else -math.inf
        if previous_claim_s is not None and previous_claim_s > start / rate:
            lower_s = max(lower_s, previous_claim_s)
        view_upper_s = upper_s if finish > end else math.inf
        kept = [word for word in span.words if lower_s < word.end_s <= view_upper_s]
        return kept, lower_s, view_upper_s

    def _decode_view(
        self, view: View, start: int, end: int, previous_claim_s: float | None,
        upper_s: float, cache: dict[Any, Any],
    ) -> _ViewResult:
        rate = SAMPLE_RATE_HZ
        pad = self._decoder.pad_samples
        total = self._audio.size
        left = max(0, start - round(view.left_s * rate))
        prefix_text = ""
        prefix_tokens: list[int] = []
        prefix_windows = 0
        if view.text == "prefix":
            left, prefix_text, prefix_tokens, prefix_windows = self._prefix(left, start)
        right = min(total, end + round(view.right_s * rate))
        lead_real = tail_real = 0
        if view.pad == "audio":
            lead_real = min(pad, left)
            tail_real = min(pad, total - right)
        begin, finish = left - lead_real, right + tail_real
        audio = np.concatenate([
            np.zeros(pad - lead_real, np.float32),
            self._audio[begin:finish],
            np.zeros(pad - tail_real, np.float32),
        ])
        prompt_text = self._prompt_text() if view.text == "prompt" else ""
        span = self._decoder.decode(
            audio,
            (begin - (pad - lead_real)) / rate,
            prompt_text=prompt_text,
            prefix_tokens=prefix_tokens,
            timestamps=view.text != "prefix",
            cache=cache,
            cache_key=(begin, finish, pad - lead_real, pad - tail_real),
        )
        kept, lower_s, view_upper_s = self._keep(span, begin, finish, start, end, previous_claim_s, upper_s)
        decoded = "".join(word.text for word in span.words)
        last_end_s = span.words[-1].end_s if span.words else None
        passes, encoder_calls = len(span.passes), span.encoder_calls
        skipped = sum(1 for entry in span.passes if entry["skipped"])
        continued_from_s = None
        continuation = None
        if not kept and last_end_s is not None and last_end_s <= start / rate:
            # The decode stopped before the window (Whisper often ends a
            # long span before a short last utterance): decode on from its
            # last word, as faster-whisper's loop seeks after a pass that
            # ends on two timestamps, with the view's text as a prompt.
            resume = max(begin, min(start, round(last_end_s * rate)))
            span = self._decoder.decode(
                np.concatenate([
                    np.zeros(pad, np.float32),
                    self._audio[resume:finish],
                    np.zeros(pad - tail_real, np.float32),
                ]),
                (resume - pad) / rate,
                prompt_text=prompt_text or prefix_text,
                cache=cache,
                cache_key=(resume, finish, pad, pad - tail_real),
            )
            kept, lower_s, view_upper_s = self._keep(span, resume, finish, start, end, previous_claim_s, upper_s)
            passes += len(span.passes)
            encoder_calls += span.encoder_calls
            skipped += sum(1 for entry in span.passes if entry["skipped"])
            continued_from_s = round(resume / rate, 3)
            continuation = "".join(word.text for word in span.words)
            self._continuations[view.label] += 1
        text = "".join(word.text for word in kept).strip()
        if not span.words:
            outcome = "dependency_no_segments"
        elif not kept:
            outcome = OUTSIDE_WINDOW
        elif not text:
            outcome = "assembled_text_empty"
        elif len(text) >= self._decoder.min_compression_len and (
            compression_ratio(text) > self._decoder.compression_threshold
        ):
            text, outcome = "", "compression_loop_rejected"
        else:
            outcome = "text_available"
        self._passes[view.label] += passes
        self._encoder_calls += encoder_calls
        self._outcomes[view.label][outcome] += 1
        detail = {
            "view": view.label,
            "text": text,
            "outcome": outcome,
            # The whole span's text and where its last word ends, for
            # telling a window the decode stopped before from one it cut.
            "decoded": decoded,
            "last_end_s": None if last_end_s is None else round(last_end_s, 3),
            "continued_from_s": continued_from_s,
            "continuation": continuation,
            "audio_s": [round(begin / rate, 3), round(finish / rate, 3)],
            "keep_s": [
                round(lower_s, 3) if math.isfinite(lower_s) else None,
                round(view_upper_s, 3) if math.isfinite(view_upper_s) else None,
            ],
            "prompt_chars": len(prompt_text),
            "prefix_tokens": len(prefix_tokens),
            "prefix_windows": prefix_windows,
            "passes": passes,
            "skipped_passes": skipped,
            "words": len(span.words),
            "kept_words": len(kept),
        }
        return _ViewResult(view, text, outcome, finish, detail)

    def decode_window(self, window: Any) -> dict[str, Any]:
        spec = self._spec
        rate = SAMPLE_RATE_HZ
        start, end = window.start_sample, window.end_sample
        upper_s = end / rate + spec.margin_s
        previous_claim_s = self._committed[-1].claimed_s if self._committed else None
        cache: dict[Any, Any] = {}
        results = [
            self._decode_view(view, start, end, previous_claim_s, upper_s, cache)
            for view in spec.views
        ]
        fields: dict[str, Any] = {}
        if spec.agreement:
            chosen, votes, fallback = choose_view(
                [agreement_key(result.text) for result in results], spec.similarity,
            )
            rule = "fallback" if fallback else "agreement"
            self._commits[f"{rule}:{results[chosen].view.label}"] += 1
            fields["commit"] = {"rule": rule, "view": chosen, "votes": votes}
        else:
            chosen = 0
        committed = results[chosen]
        emitted = max(window.emitted_at_sample, *(result.audio_end_sample for result in results))
        wait_s = (emitted - window.emitted_at_sample) / rate
        self._waits.append(wait_s)
        claimed_s = min(upper_s, committed.audio_end_sample / rate)
        self._committed.append(_Committed(start, end, claimed_s, committed.text))
        fields.update({
            "text": committed.text,
            "asr_outcome": committed.outcome,
            "emitted_at_s": emitted / rate,
            "context_wait_s": wait_s,
            "context": [result.detail for result in results],
        })
        return fields

    def record(self) -> dict[str, Any]:
        waits = np.asarray(self._waits) if self._waits else np.zeros(1)
        return {
            "mode": MODE,
            "spec": self._spec.to_json(),
            "decode": self._decoder.identity(),
            "keep_rule": (
                "word aligned end t kept when lower < t <= upper; upper = e + margin_s if the view heard"
                " real audio after e, else none; lower = s - margin_s if it heard real audio before s,"
                " else none, raised to the previous window's claim where the claim passes s;"
                " claim = min(e + margin_s, end of the real audio the committed view heard)"
            ),
            "windows": len(self._waits),
            "outcomes": {label: dict(counts) for label, counts in self._outcomes.items()},
            "passes": dict(self._passes),
            "continuations": dict(self._continuations),
            "encoder_calls": self._encoder_calls,
            "commits": dict(self._commits),
            "context_wait_s": {
                "mean": float(waits.mean()),
                "p90": float(np.percentile(waits, 90)),
                "max": float(waits.max()),
            },
        }


def make_strategy(context: Any) -> ContextDecodeStrategy:
    return ContextDecodeStrategy(context)


# --- Timing check ---------------------------------------------------------

_TIMING_BINS = ((1, 3), (4, 6), (7, 12), (13, None))


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    magnitude = np.abs(array)
    return {
        "n": int(array.size),
        "median": round(float(np.median(array)), 3),
        "median_abs": round(float(np.median(magnitude)), 3),
        "p90_abs": round(float(np.percentile(magnitude, 90)), 3),
    }


def _bin(length: int) -> str:
    for low, high in _TIMING_BINS:
        if high is None or length <= high:
            return f"{low}+" if high is None else f"{low}-{high}"
    raise AssertionError("bins end open-ended")


def timing_check(
    decoder: SpanDecoder, bench_set: Any, item_ids: Sequence[str], span_s: float,
    margins: Sequence[float] = (0.0, 0.2, 0.5),
) -> dict[str, Any]:
    """Decode each item in consecutive ``span_s`` spans (silence pads, style
    prompt, timestamps: a ``text: none`` view) and measure the aligned word
    and segment times against the reference lines of the characters they
    align with."""
    from scripts.asr_bench.lexicon import reading
    from scripts.e0_evidence.normalize_ja import normalize_minimal

    rate = SAMPLE_RATE_HZ
    pad = np.zeros(decoder.pad_samples, np.float32)
    measures: dict[str, dict[str, list[float]]] = {}
    inside = {margin: [0, 0] for margin in margins}

    def add(measure: str, length_bin: str, value: float) -> None:
        groups = measures.setdefault(measure, {})
        groups.setdefault("all", []).append(value)
        groups.setdefault(length_bin, []).append(value)

    items = {item.item_id: item for item in bench_set.items}
    lines_used = 0
    for item_id in item_ids:
        item = items[item_id]
        pcm = bench_set.load_audio(item)
        step = round(span_s * rate)
        words: list[TimedWord] = []
        segments: list[TimedSegment] = []
        for begin in range(0, pcm.size, step):
            chunk = pcm[begin:begin + step]
            span = decoder.decode(np.concatenate([pad, chunk, pad]), (begin - pad.size) / rate)
            base = len(segments)
            segments.extend(span.segments)
            words.extend(dataclasses.replace(word, segment=base + word.segment) for word in span.words)
        # Hypothesis characters with their word.
        hyp_chars: list[str] = []
        hyp_word: list[int] = []
        for index, word in enumerate(words):
            normalized = normalize_minimal(word.text)
            hyp_chars.extend(normalized)
            hyp_word.extend([index] * len(normalized))
        first_char_of_segment: dict[int, int] = {}
        last_char_of_segment: dict[int, int] = {}
        for position, word_index in enumerate(hyp_word):
            segment = words[word_index].segment
            first_char_of_segment.setdefault(segment, position)
            last_char_of_segment[segment] = position
        ordered = sorted(item.lines, key=lambda line: (line.start_s, line.end_s))
        texts = [normalize_minimal(line.text) for line in ordered]
        ref_chars = "".join(texts)
        matched, positions, _ = alignment_trace(ref_chars, "".join(hyp_chars))
        offset = 0
        for index, line in enumerate(ordered):
            length = len(texts[index])
            if not length:
                continue
            flags = matched[offset:offset + length]
            where = positions[offset:offset + length]
            offset += length
            if flags.mean() < 0.5:
                continue
            lines_used += 1
            length_bin = _bin(len(reading(line.text)) or length)
            for flag, position in zip(flags, where):
                if flag:
                    end = words[hyp_word[position]].end_s
                    for margin in margins:
                        inside[margin][1] += 1
                        if line.start_s - margin < end <= line.end_s + margin:
                            inside[margin][0] += 1
            if flags[0]:
                word = words[hyp_word[where[0]]]
                add("word_start_minus_line_start", length_bin, word.start_s - line.start_s)
                add("word_end_minus_line_start", length_bin, word.end_s - line.start_s)
                if first_char_of_segment.get(word.segment) == where[0]:
                    add("segment_start_minus_line_start", length_bin, segments[word.segment].start_s - line.start_s)
            if flags[-1]:
                word = words[hyp_word[where[-1]]]
                add("word_end_minus_line_end", length_bin, word.end_s - line.end_s)
                if last_char_of_segment.get(word.segment) == where[-1]:
                    add("segment_end_minus_line_end", length_bin, segments[word.segment].end_s - line.end_s)
    return {
        "items": list(item_ids),
        "span_s": span_s,
        "lines_used": lines_used,
        "rule": "lines with at least half their characters aligned; onsets need the first character aligned, offsets the last",
        "measures": {
            measure: {group: _summary(values) for group, values in groups.items()}
            for measure, groups in measures.items()
        },
        "aligned_chars_with_word_end_inside_line": {
            f"{margin:g}": {
                "inside": count,
                "chars": total,
                "share": round(count / total, 4) if total else None,
            }
            for margin, (count, total) in inside.items()
        },
        "decode": decoder.identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Context decoding: timing check")
    commands = parser.add_subparsers(dest="command", required=True)
    timing = commands.add_parser("timing", help="aligned word and segment times against reference lines")
    timing.add_argument("--set", required=True, type=Path, help="bench set directory")
    timing.add_argument("--candidate", required=True, type=Path, help="candidate JSON (faster_whisper)")
    timing.add_argument("--name", help="candidate to select from a candidates list")
    timing.add_argument("--items", required=True, help="comma-separated item ids")
    timing.add_argument("--span-s", type=float, default=24.0, help="decoded span length in seconds")
    timing.add_argument("--out", required=True, type=Path, help="result JSON (must not exist)")
    args = parser.parse_args(argv)

    from scripts.asr_bench.candidates import load_candidate
    from scripts.asr_bench.run import _shipped_whisper_decoder
    from scripts.asr_bench.sets import load_set

    if args.out.exists():
        raise FileExistsError(f"{args.out} exists")
    bench_set = load_set(args.set)
    candidate = load_candidate(args.candidate, args.name)
    if candidate.asr.backend != "faster_whisper":
        raise ValueError("the timing check decodes with faster_whisper")
    _, identity, backend = _shipped_whisper_decoder(candidate.asr_overrides, candidate.asr)
    result = timing_check(SpanDecoder(backend), bench_set, args.items.split(","), args.span_s)
    result["set_id"] = bench_set.set_id
    result["model"] = identity["model"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for measure, groups in result["measures"].items():
        print(measure, json.dumps(groups))
    print("inside", json.dumps(result["aligned_chars_with_word_end_inside_line"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
