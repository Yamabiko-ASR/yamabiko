"""Island recovery: a bench second pass for short utterances, above all the
meaningful interjections (はい, ううん, えっ) the recognizer drops when they
share a VAD window with other speech.

Each emitted window is decoded as usual (A, the first-pass text). The
per-frame Silero probabilities the chunker computed for the window give its
speech islands: runs that reach the VAD threshold, held while the
probability stays at or above the gate's sustain bar (threshold - 0.15, at
least 0.01) and ended by a dip below that bar lasting at least
``min_gap_ms``. A window with two or more islands has every island decoded
alone (B_i), widened by ``island_pad_ms`` on each side but never past the
window or the middle of the gap to a neighbouring island. A and the
time-ordered concatenation of the B_i are aligned once
(``align.alignment_trace`` over UniDic katakana readings). A B_i of which
fewer than half the reading characters align as exact matches is absent
from A; when it is also insertable (a lexicon interjection, or any text of
at most ``max_chars`` characters; optionally only from islands of at most
``max_island_ms`` of speech) its text is inserted into A's text at its
aligned position. Everything else stays A.

Only the window's own audio and the probabilities the chunker already
computed for it are used, so the pass could run on a live stream as soon as
a window is emitted.

Usage, to measure the merge alone::

    python -m scripts.asr_bench.island --run <island run dir> --out <new run dir>

writes the run's first-pass texts (``text_a``) as a run of their own, named
``<candidate>-first-pass``; scoring it as the baseline of the island run
pairs every window with its own first decode, so decode nondeterminism
between runs cannot pass for an effect of the merge.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.asr_bench import lexicon
from scripts.asr_bench.align import alignment_trace
from scripts.asr_bench.candidates import IslandRecoverSpec
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, Item
from scripts.e0_evidence.normalize_ja import normalize_minimal
from scripts.e0_evidence.wire import canonical_sha256

if TYPE_CHECKING:
    from scripts.asr_bench.run import StrategyContext, Window

# A B_i is absent from A when fewer than this share of its reading
# characters align as exact matches (the silver recall threshold).
ABSENT_SHARE = 0.5
# The keys of IslandRecover.record that add up over items; the runner sums
# them over its worker processes (run._ADDITIVE_STRATEGY_KEYS).
ADDITIVE_RECORD_KEYS = frozenset({
    "windows", "windows_with_islands", "island_decodes", "insertions",
    "first_pass_asr_seconds", "island_asr_seconds",
})
_TAGGER = None


def sustain_threshold(threshold: float) -> float:
    """The FrameGate sustain bar (upstream silero-vad's neg_threshold)."""
    return max(threshold - 0.15, 0.01)


def gap_frames(min_gap_ms: int, frame_ms: float) -> int:
    """Frames a dip must last to split islands, counted as the gate counts
    ``min_silence_ms`` (floored, at least one frame)."""
    return max(1, int(min_gap_ms / frame_ms))


def find_islands(
    probabilities: np.ndarray, threshold: float, min_gap_frames: int
) -> list[tuple[int, int]]:
    """Speech islands of one window as ``[start, end)`` frame ranges: each
    opens at a frame at or above ``threshold``, ends after its last frame at
    or above the sustain bar, and is followed by at least
    ``min_gap_frames`` frames below that bar (or by the window's end)."""
    sustain = sustain_threshold(threshold)
    islands: list[tuple[int, int]] = []
    start: int | None = None
    last = below = 0
    for frame, probability in enumerate(probabilities):
        if start is None:
            if probability >= threshold:
                start, last, below = frame, frame, 0
        elif probability >= sustain:
            last, below = frame, 0
        else:
            below += 1
            if below >= min_gap_frames:
                islands.append((start, last + 1))
                start = None
    if start is not None:
        islands.append((start, last + 1))
    return islands


def island_spans(
    islands: Sequence[tuple[int, int]], window_samples: int, frame_samples: int, pad_samples: int
) -> list[tuple[int, int]]:
    """Sample spans, relative to the window, of the islands' decode audio:
    each island widened by ``pad_samples`` on both sides, clipped to the
    window and to the middle of the gap to each neighbour."""
    spans = []
    for index, (start, end) in enumerate(islands):
        low = start * frame_samples - pad_samples
        high = end * frame_samples + pad_samples
        if index:
            low = max(low, (islands[index - 1][1] + start) * frame_samples // 2)
        if index + 1 < len(islands):
            high = min(high, (end + islands[index + 1][0]) * frame_samples // 2)
        spans.append((max(low, 0), min(high, window_samples)))
    return spans


def is_insertable(text: str, speech_ms: float, spec: IslandRecoverSpec) -> bool:
    """Whether an island's text may be inserted into the first-pass text."""
    if spec.max_island_ms is not None and speech_ms > spec.max_island_ms:
        return False
    if spec.insert == "lexicon":
        return lexicon.is_meaningful_interjection(text)
    return 0 < len(normalize_minimal(text)) <= spec.max_chars


@dataclass(frozen=True)
class _Token:
    # Offsets in the raw text.
    start: int
    end: int
    # Katakana reading, long-vowel marks dropped; empty for punctuation.
    reading: str


def _tokens(text: str) -> list[_Token]:
    global _TAGGER
    if _TAGGER is None:
        import fugashi

        _TAGGER = fugashi.Tagger()
    tokens = []
    position = 0
    for word in _TAGGER(text):
        start = position + len(word.white_space)
        end = start + len(word.surface)
        if text[start:end] != word.surface:
            raise ValueError(f"tagger surfaces do not tile {text!r}")
        surface = normalize_minimal(word.surface)
        reading = ""
        if surface:
            kana = getattr(word.feature, "kana", None)
            reading = "".join(
                chr(ord(ch) + 0x60) if "\u3041" <= ch <= "\u3096" else ch
                for ch in (kana if kana and kana != "*" else surface)
            ).replace("ー", "")
        tokens.append(_Token(start, end, reading))
        position = end
    return tokens


def _raw_offset(tokens: Sequence[_Token], text: str, index: int) -> int:
    """Offset in the raw text for reading index ``index``: the start of the
    first worded token reading from there (punctuation before it stays
    before), the nearer edge of a token it falls inside, else the end."""
    position = 0
    for token in tokens:
        if not token.reading:
            continue
        start, position = position, position + len(token.reading)
        if index <= start:
            return token.start
        if index < position:
            return token.start if index - start <= position - index else token.end
    return len(text.rstrip())


@dataclass(frozen=True)
class Merge:
    text: str
    # Per island: the share of its reading aligned as exact matches in A
    # (None for an island without reading), and whether it was inserted.
    matched: tuple[float | None, ...]
    inserted: tuple[bool, ...]


def find_insertions(
    first_text: str, texts: Sequence[str], insertable: Sequence[bool]
) -> tuple[tuple[float | None, ...], tuple[int | None, ...]]:
    """Align ``first_text`` with the concatenated ``texts`` once. Per text:
    the share of its reading aligned as exact matches (None for a text
    without reading), and the offset in ``first_text`` where it is to be
    inserted when it is insertable and absent (share below
    ``ABSENT_SHARE``), else None."""
    tokens = _tokens(first_text)
    readings = ["".join(token.reading for token in _tokens(text)) for text in texts]
    matched, positions, _ = alignment_trace(
        "".join(readings), "".join(token.reading for token in tokens)
    )
    shares: list[float | None] = []
    offsets: list[int | None] = []
    start = 0
    for index, reading in enumerate(readings):
        end = start + len(reading)
        share = float(matched[start:end].mean()) if reading else None
        absent = share is not None and share < ABSENT_SHARE and insertable[index]
        offsets.append(_raw_offset(tokens, first_text, int(positions[start])) if absent else None)
        shares.append(share)
        start = end
    return tuple(shares), tuple(offsets)


def insert(first_text: str, insertions: Sequence[tuple[int, str]]) -> str:
    """``first_text`` with each ``(offset, text)`` inserted at its offset;
    texts at one offset keep their order."""
    pieces = []
    cursor = 0
    for offset, text in sorted(insertions, key=lambda insertion: insertion[0]):
        pieces.extend((first_text[cursor:offset], text))
        cursor = offset
    pieces.append(first_text[cursor:])
    return "".join(pieces)


def merge(first_text: str, island_texts: Sequence[str], insertable: Sequence[bool]) -> Merge:
    """Insert into ``first_text`` every insertable island text absent from it."""
    shares, offsets = find_insertions(first_text, island_texts, insertable)
    text = insert(first_text, [
        (offset, island_texts[index].strip())
        for index, offset in enumerate(offsets)
        if offset is not None
    ])
    return Merge(text, shares, tuple(offset is not None for offset in offsets))


class IslandRecover:
    """The ``island_recover`` window strategy of the bench runner."""

    def __init__(self, context: StrategyContext) -> None:
        spec = context.candidate.strategy
        if not isinstance(spec, IslandRecoverSpec):
            raise TypeError("island_recover needs an IslandRecoverSpec")
        self._spec = spec
        self._decode = context.decode
        self._threshold = context.threshold
        self._frame_samples = context.frame_samples
        self._frame_ms = 1000 * context.frame_samples / SAMPLE_RATE_HZ
        self._gap_frames = gap_frames(spec.min_gap_ms, self._frame_ms)
        self._pad_ms = (
            context.chunker.speech_pad_ms if spec.island_pad_ms is None else spec.island_pad_ms
        )
        self._pad_samples = SAMPLE_RATE_HZ * self._pad_ms // 1000
        self._island_decode = context.decode
        self._island_asr: dict[str, Any] | None = None
        if spec.asr is not None:
            self._island_decode, self._island_asr, _ = context.whisper_decoder(
                spec.asr_overrides, spec.asr
            )
        self._sample_count = 0
        self._audio = np.zeros(0, np.float32)
        self._probabilities = np.zeros(0)
        self._windows = self._split_windows = self._island_decodes = self._insertions = 0
        self._first_seconds = self._island_seconds = 0.0

    def start_item(self, item: Item, asr_audio: np.ndarray, probabilities: np.ndarray) -> None:
        self._sample_count = item.sample_count
        self._audio = asr_audio
        self._probabilities = probabilities

    def decode_window(self, window: Window) -> dict[str, Any]:
        begin = time.perf_counter()
        text, outcome = self._decode(self._audio[window.start_sample:window.end_sample])
        first_seconds = time.perf_counter() - begin
        self._windows += 1
        self._first_seconds += first_seconds
        fs = self._frame_samples
        found = find_islands(
            self._probabilities[window.start_sample // fs:window.end_sample // fs],
            self._threshold,
            self._gap_frames,
        )
        fields: dict[str, Any] = {
            "text": text,
            "asr_outcome": outcome,
            "text_a": text,
            "asr_a_s": first_seconds,
            "island_count": len(found),
            "islands": [],
        }
        if len(found) < 2:
            return fields
        self._split_windows += 1
        spans = island_spans(found, window.end_sample - window.start_sample, fs, self._pad_samples)
        texts: list[str] = []
        allowed: list[bool] = []
        islands: list[dict[str, Any]] = []
        for (first, last), (low, high) in zip(found, spans):
            begin = time.perf_counter()
            island_text, island_outcome = self._island_decode(
                self._audio[window.start_sample + low:window.start_sample + high]
            )
            seconds = time.perf_counter() - begin
            self._island_decodes += 1
            self._island_seconds += seconds
            speech_ms = (last - first) * self._frame_ms
            texts.append(island_text)
            allowed.append(is_insertable(island_text, speech_ms, self._spec))
            islands.append({
                "start_s": self._seconds(window.start_sample + low),
                "end_s": self._seconds(window.start_sample + high),
                "speech_ms": speech_ms,
                "text": island_text,
                "asr_outcome": island_outcome,
                "asr_s": seconds,
                "insertable": allowed[-1],
            })
        merged = merge(text, texts, allowed)
        for entry, share, inserted in zip(islands, merged.matched, merged.inserted):
            entry["matched"] = share
            entry["inserted"] = inserted
        self._insertions += sum(merged.inserted)
        fields["text"] = merged.text
        fields["islands"] = islands
        return fields

    def _seconds(self, sample: int) -> float:
        """A sample position as item seconds, clamped to the item like window spans."""
        return min(sample, self._sample_count) / SAMPLE_RATE_HZ

    def record(self) -> dict[str, Any]:
        spec = self._spec
        return {
            **spec.to_json(),
            "threshold": self._threshold,
            "sustain_threshold": sustain_threshold(self._threshold),
            "min_gap_frames": self._gap_frames,
            "island_pad_ms_effective": self._pad_ms,
            "absent_share": ABSENT_SHARE,
            "lexicon_entries_sha256": canonical_sha256(lexicon.INTERJECTIONS),
            # None: the islands went through the candidate's own decoder.
            "island_asr": self._island_asr,
            "windows": self._windows,
            "windows_with_islands": self._split_windows,
            "island_decodes": self._island_decodes,
            "insertions": self._insertions,
            "first_pass_asr_seconds": self._first_seconds,
            "island_asr_seconds": self._island_seconds,
        }


def write_first_pass(run_dir: Path, out_dir: Path) -> None:
    """Write ``run_dir``'s first-pass window texts as a run in ``out_dir``."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if (record.get("strategy") or {}).get("mode") != "island_recover":
        raise ValueError(f"{run_dir} is not an island_recover run")
    out_dir.mkdir(parents=True, exist_ok=False)
    record["candidate"]["name"] = f"{record['candidate']['name']}-first-pass"
    record["first_pass_of"] = str(run_dir.resolve())
    # Timing of the first decodes only, as if the run had had no strategy.
    timing = record["timing"]
    timing["asr_seconds"] = record["strategy"]["first_pass_asr_seconds"]
    timing["rtf"] = (timing["vad_seconds"] + timing["asr_seconds"]) / record["set"]["audio_seconds"]
    with (run_dir / "windows.jsonl").open(encoding="utf-8") as source, (
        out_dir / "windows.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as sink:
        for line in source:
            window = json.loads(line)
            window["text"] = window["text_a"]
            window["asr_s"] = window["asr_a_s"]
            sink.write(json.dumps(window, ensure_ascii=False) + "\n")
    (out_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an island run's first pass as a run.")
    parser.add_argument("--run", required=True, type=Path, help="island_recover run directory")
    parser.add_argument("--out", required=True, type=Path, help="new run directory")
    args = parser.parse_args(argv)
    write_first_pass(args.run, args.out)
    print(f"{args.run} first pass -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
