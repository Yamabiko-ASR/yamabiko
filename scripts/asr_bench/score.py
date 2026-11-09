"""Score bench runs: aligned and collar line recall, chance control, line
kinds, key-word recall, CER, hallucinated windows, line delay, cut and
merged lines, paired deltas.

Usage::

    python -m scripts.asr_bench.score --run <run dir> [--run <run dir> ...]
        --out <report dir>

Writes ``report.json`` and ``report.md``. Runs are grouped by bench set; the
first ``--run`` given for a set is that set's baseline, and every other run
on the same set gets paired deltas against it.

- Aligned recall (the primary recall figure): per item, the reference lines
  in time order and the window texts in time order are concatenated and
  aligned once, as one minimum-edit global alignment (Levenshtein with the
  E0 ``character_trace`` tie-break: match, substitution, deletion,
  insertion). A line is caught when at least 0.75 (synthetic) or 0.5
  (silver) of its characters align as exact matches. The reading form
  compares katakana readings from the MeCab/UniDic tagger the app already
  ships (``fugashi`` with ``unidic-lite``), long-vowel marks dropped, so a
  line heard correctly but spelled differently (kana for kanji, ``Windows``
  as ``ウィンドウズ``) still counts as caught; the text form compares the
  minimal-normalized text.
- Chance recall: each item's reference scored against the hypothesis of
  another item, the one a fixed number of positions later in set order
  (cyclic; the smallest shift that pairs no item with an item holding the
  same reference). It is what a line catches from unrelated speech, for
  both the aligned and the collar rule.
- Reading length bins (1-3, 4-6, 7-12, 13-20, 21+ kana) split recall,
  chance, line delay and the aligned recall and line delay deltas by line
  length.
- Line kinds: a line is vocalization-only when every token of its
  minimal-normalized text has UniDic ``pos1`` 感動詞, 補助記号, 記号 or
  空白. A vocalization-only line holding a token of the ``lexicon.py``
  lexicon (meaningful interjections: affirm, deny, surprise or question,
  call, hesitation that answers, greeting or thanks) is ``kind=intj``, any
  other one ``kind=nonlex`` (grunts, gasps, laughter); every other line is
  ``kind=words``. The kinds are line groups next to the item groups.
- Key-word recall: every 感動詞 token of any reference line whose reading
  is a lexicon entry is one occurrence; it is recalled when all of its
  reading characters except ッ align as exact matches in the aligned
  recall alignment. Pooled and per entry.
- Miss causes, per line the aligned reading rule misses: no window over
  the line; windows over it, all with empty text; its own window (not
  merged) with the wrong text; merged into a window that also holds at
  least half of another line, with the line's text dropped or wrong.
- Collar recall (the earlier rule, kept for continuity): a line is caught
  when the minimal-normalized text (or reading) of the windows that overlap
  its span widened by a 0.5 s collar contains, as ``difflib.SequenceMatcher``
  matching blocks, at least the same share of the line's characters.
- CER (synthetic) and subtitle CER (silver): the edit distance of the same
  alignments over reference characters, text and reading form, pooled over
  items.
- Hallucinated windows (synthetic only): windows lying wholly inside one
  ``speech_free`` span whose minimal-normalized text is not empty.
- Uncued windows (silver only): windows with non-empty minimal-normalized
  text that overlap no reference line, even with the collar. They are
  hallucinations or speech the subtitles leave out (background voices,
  dropped lyric cues), so read them as a relative measure between runs;
  reported per hour of audio.
- Line delay, per line caught by the collar reading rule: ``emitted_at_s`` of
  the last window overlapping the line (the last collar window when none
  overlaps the line itself) minus the line's end, plus that window's
  ``asr_s``. Median, p90 and mean per group and per reading length bin of
  each line kind. Runs without emission timing (run schema 1.0) report n/a.
  Each run names its emission timing: the chunker frame that emitted the
  window, or the frame's release from the sound-tag delay line (run schema
  1.2 and later with tags); schema 1.1 runs with tags left that delay out. A window
  emitted in pieces (``emissions``, each with its own ``emitted_at_s`` and
  ``asr_s``) counts from its first piece after which its later pieces add
  no collar coverage to the line, and the line waits for the latest of the
  windows over it.
- Cut lines: a window boundary falls inside the line more than 0.15 s from
  both of its edges. Merged lines: the window holding the largest share of
  the line also holds at least half of another line.
- Paired deltas: candidate minus baseline, pooled per replicate, 10,000
  bootstrap replicates (seed 105), 95% interval from the E0
  ``percentile_type7`` quantile. The resampled unit is the cluster
  (``condition.cluster``, all its items together) when every item of the
  set has one, otherwise the item. The line delay delta is the mean over
  lines both runs catch, for each group and each reading length bin.
- Timing is judged per GPU model and in one process: line delay deltas,
  and the ASR-seconds ratio and RTF difference against the baseline, exist
  only between runs whose ``gpu.model`` (``remote.gpu_model`` for a remote
  run without it) is the same; runs from before either field count as one
  model, "not recorded". Otherwise they read "not comparable (GPU differs)",
  and "not comparable (parallel workers)" when either run has
  ``timing_valid`` false (run with ``--workers`` above 1, or merged from
  such a run).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench.align import alignment_trace
from scripts.asr_bench.lexicon import INTERJECTIONS, VOCAL_POS1, Analysis, analyze
from scripts.asr_bench.lexicon import reading as _reading
from scripts.asr_bench.run import RUN_SCHEMA_VERSIONS
from scripts.asr_bench.sets import BenchSet, Item, Line, load_set
from scripts.e0_evidence.normalize_ja import JA_MINIMAL_1_1, normalize_minimal
from scripts.e0_evidence.score import percentile_type7

REPORT_SCHEMA_VERSION = "asr-bench-report-1.7"
READING_POLICY = "unidic-lite kana reading, hiragana to katakana, long-vowel marks dropped"
COLLAR_S = 0.5
RECALL_THRESHOLD = {"synthetic": 0.75, "silver": 0.5}
# Reading-length bins in kana (inclusive bounds; None is open-ended).
LENGTH_BINS = ((1, 3), (4, 6), (7, 12), (13, 20), (21, None))
LINE_KINDS = ("words", "intj", "nonlex")
MISS_CAUSES = ("no_window", "empty_window", "own_window", "merged_window")
GROUP_KEYS = {"synthetic": ("snr_db", "speech_source"), "silver": ("fixture_label", "title")}
CUT_MARGIN_S = 0.15
MERGE_SHARE = 0.5
CLUSTER_KEY = "cluster"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 105
_INTERVAL = (0.025, 0.975)
_UNTIMED_RUN_SCHEMA = "asr-bench-run-1.0"
# The GPU of runs from before run.json recorded one (all local runs).
_UNRECORDED_GPU = "not recorded"
_ALL = "all"
# Line groups select lines, not items.
_LINE_GROUPS: dict[str, Callable[[_LineScore], bool]] = {
    f"kind={kind}": (lambda line, kind=kind: line.kind == kind) for kind in LINE_KINDS
}
# Recall rules: (metric prefix, form suffix); the _LineScore field is the
# two joined, the metrics are <prefix>_caught<suffix> and <prefix>_recall<suffix>.
_RULES = (
    ("aligned", "_reading"),
    ("aligned", ""),
    ("collar", "_reading"),
    ("collar", ""),
    ("chance_aligned", "_reading"),
    ("chance_collar", "_reading"),
)
_BIN_RULES = ("aligned", "chance_aligned", "collar", "chance_collar")


def _line_kind(analysis: Analysis) -> str:
    if not analysis.vocal:
        return "words"
    return "intj" if analysis.interjections else "nonlex"


def _bin_label(low: int, high: int | None) -> str:
    return f"{low}+" if high is None else f"{low}-{high}"


_BIN_LABELS = tuple(_bin_label(low, high) for low, high in LENGTH_BINS)


def _length_bin(length: int) -> str:
    for (_, high), label in zip(LENGTH_BINS, _BIN_LABELS):
        if high is None or length <= high:
            return label
    raise AssertionError("LENGTH_BINS must end open-ended")


@dataclass(frozen=True)
class _Window:
    start_s: float
    end_s: float
    text: str
    emitted_at_s: float | None = None
    asr_s: float | None = None
    # A window emitted in pieces: (text, emitted_at_s + asr_s) per piece, in
    # order; the texts join to ``text``. None: one piece, the whole window.
    emissions: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True)
class _Hypothesis:
    """One item's windows in time order and their concatenated text."""

    windows: tuple[_Window, ...]
    text: str
    reading: str


@dataclass(frozen=True)
class _LineScore:
    line_id: str
    kind: str
    length_bin: str
    aligned_reading: bool
    aligned: bool
    collar_reading: bool
    collar: bool
    # False when the set has no chance pairing (_ItemScore.chance is False).
    chance_aligned_reading: bool
    chance_collar_reading: bool
    cut: bool
    merged: bool
    # Seconds, for a line caught by the collar reading rule in a timed run.
    delay: float | None
    # (lexicon key, recalled) per interjection occurrence in the line.
    keywords: tuple[tuple[str, bool], ...]
    # One of MISS_CAUSES when the aligned reading rule misses the line.
    miss_cause: str | None


@dataclass(frozen=True)
class _ItemScore:
    item_id: str
    groups: tuple[str, ...]
    lines: tuple[_LineScore, ...]
    chance: bool
    timed: bool
    skipped_lines: int
    windows: int
    errors: int
    reference_chars: int
    errors_reading: int
    reference_chars_reading: int
    hallucinations: int | None
    speech_free_s: float | None
    uncued: int | None
    audio_s: float
    missed: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _RunScore:
    label: str
    run_dir: Path
    record: dict[str, Any]
    bench_set: BenchSet
    items: dict[str, _ItemScore]


def _coverage(reference: str, hypothesis: str) -> float:
    """Share of ``reference`` characters inside matching blocks of
    ``hypothesis``. autojunk is off so frequent kana in a long hypothesis are
    never treated as junk (the real recall test makes the same choice)."""
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(reference)


def _alignment(reference: str, hypothesis: str) -> tuple[np.ndarray, int]:
    """Exact-match flag per ``reference`` character in one minimum-edit
    global alignment with ``hypothesis`` (``align.alignment_trace``), and its
    edit distance."""
    matched, _, distance = alignment_trace(reference, hypothesis)
    return matched, distance


def _aligned_counts(
    parts: Sequence[str], hypothesis: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per part, its characters aligned as exact matches when the
    concatenated parts are aligned with ``hypothesis``; the exact-match flag
    per reference character; and the edit distance."""
    matched, distance = _alignment("".join(parts), hypothesis)
    owner = np.repeat(np.arange(len(parts)), [len(part) for part in parts])
    return np.bincount(owner[matched], minlength=len(parts)), matched, distance


def _hypothesis(windows: Sequence[_Window]) -> _Hypothesis:
    ordered = tuple(sorted(windows, key=lambda window: (window.start_s, window.end_s)))
    joined = "".join(window.text for window in ordered)
    return _Hypothesis(ordered, normalize_minimal(joined), _reading(joined))


def _chance_shift(bench_set: BenchSet) -> int | None:
    """Smallest cyclic shift in set order that pairs no item with an item
    holding the same reference (a synthetic scene repeats one reference at
    each SNR); None when no shift does, as in a one-item set."""
    keys = [tuple(line.text for line in item.lines) for item in bench_set.items]
    count = len(keys)
    for shift in range(1, count):
        if all(keys[k] != keys[(k + shift) % count] for k in range(count)):
            return shift
    return None


def _value_label(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _group_order(bench_set: BenchSet) -> list[str]:
    """``all`` first, then the line kinds, then each condition key's values:
    text labels alphabetically, then numbers from high to low (clean, 10, 5,
    0, -5)."""

    def sort_key(value: object) -> tuple[int, float, str]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (1, -float(value), "")
        return (0, 0.0, _value_label(value))

    order = [_ALL, *_LINE_GROUPS]
    for key in GROUP_KEYS[bench_set.kind]:
        values = {
            _value_label(item.condition[key]): item.condition[key]
            for item in bench_set.items
            if key in item.condition
        }
        for label in sorted(values, key=lambda name: sort_key(values[name])):
            order.append(f"{key}={label}")
    return order


def _item_groups(item: Item, kind: str) -> tuple[str, ...]:
    groups = [_ALL]
    for key in GROUP_KEYS[kind]:
        if key in item.condition:
            groups.append(f"{key}={_value_label(item.condition[key])}")
    return tuple(groups)


def _overlap(window: _Window, line: Line) -> float:
    return max(0.0, min(window.end_s, line.end_s) - max(window.start_s, line.start_s))


def _line_delay(line: Line, collar_windows: Sequence[_Window]) -> float:
    """Seconds from the line's end until the last window holding it is decoded.

    When a window over the line was emitted in pieces, each window over the
    line counts from its piece that completes its part of the line
    (``_piece_time``), and the line waits for the latest of them.
    """
    holding = [w for w in collar_windows if w.start_s < line.end_s and w.end_s > line.start_s]
    if all(window.emissions is None for window in collar_windows):
        last = max(holding or collar_windows, key=lambda window: window.emitted_at_s)
        return last.emitted_at_s - line.end_s + last.asr_s
    text, reading = normalize_minimal(line.text), _reading(line.text)
    chosen = [
        index for index, window in enumerate(collar_windows)
        if not holding or (window.start_s < line.end_s and window.end_s > line.start_s)
    ]
    return max(_piece_time(line, text, reading, collar_windows, index) for index in chosen) - line.end_s


def _piece_time(
    line: Line, text: str, reading: str, collar_windows: Sequence[_Window], index: int
) -> float:
    """When window ``index`` has shown its part of the line: the end of its
    decode, or for a window emitted in pieces the end of the first piece
    after which the window's later pieces add no collar coverage (reading
    form, as the collar rule) to the line."""
    window = collar_windows[index]
    assert window.emitted_at_s is not None and window.asr_s is not None
    if window.emissions is None:
        return window.emitted_at_s + window.asr_s
    full = _collar(line, text, reading, collar_windows)[1]
    shown = ""
    trial = list(collar_windows)
    for part, ready_s in window.emissions:
        shown += part
        trial[index] = replace(window, text=shown)
        if _collar(line, text, reading, trial)[1] >= full:
            return ready_s
    raise AssertionError("the pieces of a window join to its text")


def _is_cut(line: Line, boundaries: Sequence[float]) -> bool:
    low, high = line.start_s + CUT_MARGIN_S, line.end_s - CUT_MARGIN_S
    return any(low < boundary < high for boundary in boundaries)


def _is_merged(line: Line, scored: Sequence[Line], windows: Sequence[_Window]) -> bool:
    best = max(windows, key=lambda window: _overlap(window, line), default=None)
    if best is None or _overlap(best, line) <= 0.0:
        return False
    return any(
        other is not line
        and _overlap(best, other) >= MERGE_SHARE * (other.end_s - other.start_s)
        for other in scored
    )


def _collar(
    line: Line, text: str, reading: str, windows: Sequence[_Window]
) -> tuple[float, float, list[_Window]]:
    """Collar coverage of the line's text and reading, and the collar windows."""
    low, high = line.start_s - COLLAR_S, line.end_s + COLLAR_S
    collar_windows = [window for window in windows if window.start_s < high and window.end_s > low]
    overlapping = "".join(window.text for window in collar_windows)
    coverage = _coverage(text, normalize_minimal(overlapping))
    coverage_reading = _coverage(reading, _reading(overlapping)) if reading else coverage
    return coverage, coverage_reading, collar_windows


def _miss_cause(line: Line, windows: Sequence[_Window], merged: bool) -> str:
    """Why a missed line was missed, as one of MISS_CAUSES."""
    over = [window for window in windows if _overlap(window, line) > 0.0]
    if not over:
        return "no_window"
    if not any(normalize_minimal(window.text) for window in over):
        return "empty_window"
    return "merged_window" if merged else "own_window"


def _score_item(
    item: Item,
    hypothesis: _Hypothesis,
    chance: _Hypothesis | None,
    kind: str,
    timed: bool,
) -> _ItemScore:
    ordered = hypothesis.windows
    threshold = RECALL_THRESHOLD[kind]
    skipped = 0
    scored: list[Line] = []
    texts: list[str] = []
    readings: list[str] = []
    for line in sorted(item.lines, key=lambda line: (line.start_s, line.end_s)):
        text = normalize_minimal(line.text)
        if not text:
            skipped += 1
            continue
        scored.append(line)
        texts.append(text)
        readings.append(_reading(line.text))
    # A line whose reading is empty is aligned by its text, as the collar
    # rule falls back to text coverage for it.
    parts = [reading or text for reading, text in zip(readings, texts)]
    aligned_text, _, errors = _aligned_counts(texts, hypothesis.text)
    aligned_reading, matched_reading, errors_reading = _aligned_counts(parts, hypothesis.reading)
    chance_reading = _aligned_counts(parts, chance.reading)[0] if chance is not None else None
    offsets = np.cumsum([0, *(len(part) for part in parts)]).tolist()

    boundaries = sorted({window.start_s for window in ordered} | {window.end_s for window in ordered})
    lines: list[_LineScore] = []
    missed: list[dict[str, Any]] = []
    for index, line in enumerate(scored):
        text, reading, part = texts[index], readings[index], parts[index]
        analysis = analyze(line.text)
        line_kind = _line_kind(analysis)
        # Interjection spans index the reading, so a text-aligned line has none.
        keywords = tuple(
            (key, all(
                matched_reading[offsets[index] + position]
                for position in range(start, end)
                if part[position] != "ッ"
            ))
            for start, end, key in (analysis.interjections if reading else ())
        )
        share_text = int(aligned_text[index]) / len(text)
        share_reading = int(aligned_reading[index]) / len(part)
        coverage, coverage_reading, collar_windows = _collar(line, text, reading, ordered)
        chance_aligned = chance_collar = False
        if chance is not None:
            chance_aligned = int(chance_reading[index]) / len(part) >= threshold
            chance_collar = _collar(line, text, reading, chance.windows)[1] >= threshold
        collar_reading = coverage_reading >= threshold
        merged = _is_merged(line, scored, ordered)
        miss_cause = _miss_cause(line, ordered, merged) if share_reading < threshold else None
        lines.append(_LineScore(
            line_id=line.line_id,
            kind=line_kind,
            length_bin=_length_bin(len(part)),
            aligned_reading=share_reading >= threshold,
            aligned=share_text >= threshold,
            collar_reading=collar_reading,
            collar=coverage >= threshold,
            chance_aligned_reading=chance_aligned,
            chance_collar_reading=chance_collar,
            cut=_is_cut(line, boundaries),
            merged=merged,
            delay=_line_delay(line, collar_windows) if timed and collar_reading else None,
            keywords=keywords,
            miss_cause=miss_cause,
        ))
        if miss_cause is not None:
            missed.append({
                "line_id": line.line_id,
                "kind": line_kind,
                "miss_cause": miss_cause,
                "aligned_reading": round(share_reading, 4),
                "aligned_text": round(share_text, 4),
                "collar_reading": round(coverage_reading, 4),
                "collar_text": round(coverage, 4),
                "keywords_missed": [key for key, hit in keywords if not hit],
            })

    hallucinations = uncued = None
    speech_free_s = None
    if kind == "silver":
        uncued = sum(
            1
            for window in ordered
            if normalize_minimal(window.text)
            and not any(
                line.start_s < window.end_s + COLLAR_S and window.start_s - COLLAR_S < line.end_s
                for line in item.lines
            )
        )
    if kind == "synthetic":
        hallucinations = sum(
            1
            for window in ordered
            if normalize_minimal(window.text)
            and any(start <= window.start_s and window.end_s <= end for start, end in item.speech_free)
        )
        speech_free_s = sum(end - start for start, end in item.speech_free)
    return _ItemScore(
        item_id=item.item_id,
        groups=_item_groups(item, kind),
        lines=tuple(lines),
        chance=chance is not None,
        timed=timed,
        skipped_lines=skipped,
        windows=len(ordered),
        errors=errors,
        reference_chars=sum(len(text) for text in texts),
        errors_reading=errors_reading,
        reference_chars_reading=sum(len(part) for part in parts),
        hallucinations=hallucinations,
        speech_free_s=speech_free_s,
        uncued=uncued,
        audio_s=item.duration_s,
        missed=tuple(missed),
    )


def _window(raw: dict[str, Any], timed: bool, where: str) -> _Window:
    if not timed:
        return _Window(float(raw["start_s"]), float(raw["end_s"]), str(raw["text"]))
    if raw.get("emitted_at_s") is None or raw.get("asr_s") is None:
        raise ValueError(f"{where}: window lacks emitted_at_s or asr_s")
    emissions = None
    if raw.get("emissions") is not None:
        emissions = tuple(
            (str(piece["text"]), float(piece["emitted_at_s"]) + float(piece["asr_s"]))
            for piece in raw["emissions"]
        )
        if not emissions or "".join(part for part, _ in emissions) != raw["text"]:
            raise ValueError(f"{where}: emissions do not join to the window text")
    return _Window(
        float(raw["start_s"]), float(raw["end_s"]), str(raw["text"]),
        float(raw["emitted_at_s"]), float(raw["asr_s"]), emissions,
    )


def _load_run(run_dir: Path) -> tuple[dict[str, Any], BenchSet, dict[str, _ItemScore]]:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record.get("schema_version") not in RUN_SCHEMA_VERSIONS:
        raise ValueError(f"{run_dir}: unsupported run schema {record.get('schema_version')!r}")
    timed = record["schema_version"] != _UNTIMED_RUN_SCHEMA
    bench_set = load_set(Path(record["set"]["path"]))
    if bench_set.set_id != record["set"]["set_id"]:
        raise ValueError(f"{run_dir}: the set at {bench_set.root} is no longer {record['set']['set_id']}")
    windows: dict[str, list[_Window]] = {item.item_id: [] for item in bench_set.items}
    with (run_dir / "windows.jsonl").open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            raw = json.loads(line)
            where = f"{run_dir}/windows.jsonl:{number}"
            if raw["item_id"] not in windows:
                raise ValueError(f"{where}: unknown item {raw['item_id']!r}")
            windows[raw["item_id"]].append(_window(raw, timed, where))
    hypotheses = [_hypothesis(windows[item.item_id]) for item in bench_set.items]
    shift = _chance_shift(bench_set)
    count = len(hypotheses)
    items = {
        item.item_id: _score_item(
            item,
            hypotheses[index],
            hypotheses[(index + shift) % count] if shift is not None else None,
            bench_set.kind,
            timed,
        )
        for index, item in enumerate(bench_set.items)
    }
    return record, bench_set, items


def _emission(record: dict[str, Any]) -> str:
    """How a run timed its windows' emission."""
    if record["schema_version"] == _UNTIMED_RUN_SCHEMA:
        return "untimed"
    layout = (record.get("emission") or {}).get("tag_layout")
    if layout is not None:
        return f"tag release (hop {layout['hop_s']:g} s, look-ahead {layout['lookahead_s']:g} s)"
    if (record.get("vad_assist") or {}).get("tags") is not None:
        return "chunker frame, tag delay left out"
    return "chunker frame"


def _gpu(record: dict[str, Any]) -> str:
    """The GPU model a run's timings come from: run.py's ``gpu.model``, else
    a remote run's ``remote.gpu_model``; runs from before either say so."""
    model = ((record.get("gpu") or {}).get("model")
             or (record.get("remote") or {}).get("gpu_model"))
    return model or _UNRECORDED_GPU


def _timing_valid(record: dict[str, Any]) -> bool:
    """False for a run whose decodes shared the machine with other worker
    processes (run schema 1.3); earlier runs ran in one process."""
    return record.get("timing_valid", True)


def _not_comparable(base: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """Why two runs' timings may not be compared, or None when they may:
    latency is judged per GPU model, on runs made in one process."""
    if not (_timing_valid(base) and _timing_valid(candidate)):
        return "parallel workers"
    return None if _gpu(base) == _gpu(candidate) else "GPU differs"


def _timing_delta(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """A run's ASR seconds (as a ratio) and RTF (as a difference) against
    the baseline's, on the same GPU model only."""
    reason = _not_comparable(base, candidate)
    if reason:
        return {"not_comparable": reason, "asr_seconds_ratio": None, "rtf_delta": None}
    before, after = base["timing"], candidate["timing"]
    return {"not_comparable": None,
            "asr_seconds_ratio": _ratio(after["asr_seconds"], before["asr_seconds"]),
            "rtf_delta": after["rtf"] - before["rtf"]}


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _delay_summary(delays: Sequence[float]) -> dict[str, Any]:
    return {
        "lines": len(delays),
        "median_s": percentile_type7(delays, 0.5),
        "p90_s": percentile_type7(delays, 0.9),
        "mean_s": _ratio(sum(delays), len(delays)),
    }


def _selected(item: _ItemScore, group: str) -> tuple[_LineScore, ...]:
    """The item's lines in ``group`` (all of them for an item group)."""
    keep = _LINE_GROUPS.get(group)
    return item.lines if keep is None else tuple(line for line in item.lines if keep(line))


def _in_group(item: _ItemScore, group: str) -> bool:
    return group in _LINE_GROUPS or group in item.groups


def _line_metrics(lines: Sequence[_LineScore], chance: bool) -> dict[str, Any]:
    metrics: dict[str, Any] = {"lines": len(lines)}
    for prefix, suffix in _RULES:
        caught = (
            None if prefix.startswith("chance") and not chance
            else sum(getattr(line, prefix + suffix) for line in lines)
        )
        metrics[f"{prefix}_caught{suffix}"] = caught
        metrics[f"{prefix}_recall{suffix}"] = None if caught is None else _ratio(caught, len(lines))
    return metrics


def _keyword_metrics(lines: Sequence[_LineScore]) -> dict[str, Any]:
    """Key-word occurrences and recall, pooled and per lexicon entry."""
    counts: dict[str, list[int]] = {}
    for line in lines:
        for key, hit in line.keywords:
            entry = counts.setdefault(key, [0, 0])
            entry[0] += 1
            entry[1] += hit
    occurrences = sum(n for n, _ in counts.values())
    recalled = sum(hit for _, hit in counts.values())
    return {
        "keyword_occurrences": occurrences,
        "keyword_recalled": recalled,
        "keyword_recall": _ratio(recalled, occurrences),
        "keywords": {
            key: {
                "occurrences": counts[key][0],
                "recalled": counts[key][1],
                "recall": _ratio(counts[key][1], counts[key][0]),
            }
            for key in INTERJECTIONS
            if key in counts
        },
    }


def _aggregate(items: Iterable[_ItemScore], group: str, kind: str) -> dict[str, Any]:
    items = [item for item in items if _in_group(item, group)]
    lines = [line for item in items for line in _selected(item, group)]
    chance = all(item.chance for item in items)
    timed = all(item.timed for item in items)

    def delay(selected: Iterable[_LineScore]) -> dict[str, Any] | None:
        if not timed:
            return None
        return _delay_summary(sorted(line.delay for line in selected if line.delay is not None))

    metrics: dict[str, Any] = {"items": len(items), **_line_metrics(lines, chance)}
    metrics["length_bins"] = {}
    for label in _BIN_LABELS:
        binned = [line for line in lines if line.length_bin == label]
        metrics["length_bins"][label] = {
            **{
                key: value for key, value in _line_metrics(binned, chance).items()
                if key == "lines" or key.endswith("_reading")
            },
            "delay": delay(binned),
        }
    cut = sum(line.cut for line in lines)
    merged = sum(line.merged for line in lines)
    metrics.update(
        delay=delay(lines),
        cut=cut,
        cut_rate=_ratio(cut, len(lines)),
        merged=merged,
        merged_rate=_ratio(merged, len(lines)),
    )
    metrics.update(_keyword_metrics(lines))
    causes = [line.miss_cause for line in lines if line.miss_cause is not None]
    metrics["miss_causes"] = {cause: causes.count(cause) for cause in MISS_CAUSES}
    if group in _LINE_GROUPS:
        return metrics

    def total(name: str) -> int:
        return sum(getattr(item, name) for item in items)

    errors, reference_chars = total("errors"), total("reference_chars")
    errors_reading, reference_chars_reading = total("errors_reading"), total("reference_chars_reading")
    metrics.update(
        errors=errors,
        reference_chars=reference_chars,
        cer=_ratio(errors, reference_chars),
        errors_reading=errors_reading,
        reference_chars_reading=reference_chars_reading,
        cer_reading=_ratio(errors_reading, reference_chars_reading),
        skipped_lines=total("skipped_lines"),
        windows=total("windows"),
    )
    if kind == "synthetic":
        metrics.update(
            hallucinations=sum(item.hallucinations or 0 for item in items),
            speech_free_s=sum(item.speech_free_s or 0.0 for item in items),
        )
    if kind == "silver":
        uncued = sum(item.uncued or 0 for item in items)
        audio_h = sum(item.audio_s for item in items) / 3600.0
        metrics.update(uncued=uncued, audio_h=audio_h, uncued_per_h=_ratio(uncued, audio_h))
    return metrics


def _paired_delta(
    base_hits: np.ndarray, candidate_hits: np.ndarray, totals: np.ndarray
) -> dict[str, Any]:
    """Pooled candidate-minus-baseline rate with a paired bootstrap over the
    resampling units (one array entry per unit)."""
    total = int(totals.sum())
    if total == 0:
        return {"delta": None, "lower": None, "upper": None, "valid_replicates": 0}
    point = float(candidate_hits.sum() - base_hits.sum()) / total
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, totals.size, size=(BOOTSTRAP_REPLICATES, totals.size))
    replicate_totals = totals[draws].sum(axis=1)
    valid = replicate_totals > 0
    replicate_deltas = (
        (candidate_hits[draws].sum(axis=1) - base_hits[draws].sum(axis=1))[valid]
        / replicate_totals[valid]
    ).tolist()
    return {
        "delta": point,
        "lower": percentile_type7(replicate_deltas, _INTERVAL[0]),
        "upper": percentile_type7(replicate_deltas, _INTERVAL[1]),
        "valid_replicates": len(replicate_deltas),
    }


def _bootstrap_units(bench_set: BenchSet) -> tuple[str, dict[str, str]]:
    """``("cluster", item -> cluster)`` when every item has a cluster label,
    else ``("item", item -> item)``."""
    if bench_set.items and all(CLUSTER_KEY in item.condition for item in bench_set.items):
        return "cluster", {
            item.item_id: str(item.condition[CLUSTER_KEY]) for item in bench_set.items
        }
    return "item", {item.item_id: item.item_id for item in bench_set.items}


# Paired-delta line metrics: metric -> _LineScore field (rate over lines).
_LINE_DELTAS = {
    "aligned_recall_reading": "aligned_reading",
    "aligned_recall": "aligned",
    "collar_recall_reading": "collar_reading",
    "collar_recall": "collar",
}
# Paired-delta item metrics (item groups only): metric -> (hits, totals).
_ITEM_DELTAS = {
    "cer_reading": ("errors_reading", "reference_chars_reading"),
    "cer": ("errors", "reference_chars"),
}


def _deltas(
    base: _RunScore,
    candidate: _RunScore,
    groups: Sequence[str],
    kind: str,
    unit_of: dict[str, str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    timed = all(item.timed for run in (base, candidate) for item in run.items.values())
    not_comparable = _not_comparable(base.record, candidate.record)
    for group in groups:
        ids = [item_id for item_id, item in base.items.items() if _in_group(item, group)]
        units = list(dict.fromkeys(unit_of[item_id] for item_id in ids))
        positions = np.array([units.index(unit_of[item_id]) for item_id in ids], dtype=np.int64)

        def per_unit(values: Iterable[float], dtype: type = np.int64) -> np.ndarray:
            """Sum per-item values into their bootstrap units."""
            sums = np.zeros(len(units), dtype=dtype)
            np.add.at(sums, positions, np.fromiter(values, dtype=dtype, count=len(ids)))
            return sums

        pairs = [
            (_selected(base.items[item_id], group), _selected(candidate.items[item_id], group))
            for item_id in ids
        ]

        def line_delta(field: str, length_bin: str | None = None) -> dict[str, Any]:
            def hits(lines: Sequence[_LineScore]) -> int:
                return sum(
                    getattr(line, field) for line in lines
                    if length_bin is None or line.length_bin == length_bin
                )

            return _paired_delta(
                per_unit(hits(b) for b, _ in pairs),
                per_unit(hits(c) for _, c in pairs),
                per_unit(
                    sum(length_bin is None or line.length_bin == length_bin for line in b)
                    for b, _ in pairs
                ),
            )

        def keyword_delta(word: str | None = None) -> dict[str, Any]:
            def found(lines: Sequence[_LineScore]) -> list[bool]:
                return [
                    hit for line in lines for key, hit in line.keywords
                    if word is None or key == word
                ]

            return _paired_delta(
                per_unit(sum(found(b)) for b, _ in pairs),
                per_unit(sum(found(c)) for _, c in pairs),
                per_unit(len(found(b)) for b, _ in pairs),
            )

        def delay_delta(length_bin: str | None = None) -> dict[str, Any]:
            """Mean delay change over the lines both runs catch (and time)."""
            if not timed or not_comparable:
                return {"delta": None, "lower": None, "upper": None, "valid_replicates": 0,
                        "lines": 0, "not_comparable": not_comparable}
            common = [
                [
                    (x.delay, y.delay) for x, y in zip(b, c)
                    if x.delay is not None and y.delay is not None
                    and (length_bin is None or x.length_bin == length_bin)
                ]
                for b, c in pairs
            ]
            delta = _paired_delta(
                per_unit((sum(x for x, _ in both) for both in common), np.float64),
                per_unit((sum(y for _, y in both) for both in common), np.float64),
                per_unit(len(both) for both in common),
            )
            delta["lines"] = sum(len(both) for both in common)
            return delta

        entry: dict[str, Any] = {"items": len(ids), "units": len(units)}
        for metric, field in _LINE_DELTAS.items():
            entry[metric] = line_delta(field)
        entry["keyword_recall"] = keyword_delta()
        if group == _ALL:
            present = {key for lines, _ in pairs for line in lines for key, _ in line.keywords}
            entry["keywords"] = {
                key: {"keyword_recall": keyword_delta(key)}
                for key in INTERJECTIONS
                if key in present
            }
        if group == _ALL or group in _LINE_GROUPS:
            entry["length_bins"] = {
                label: {
                    "aligned_recall_reading": line_delta("aligned_reading", label),
                    "delay_mean": delay_delta(label),
                }
                for label in _BIN_LABELS
            }
        if group not in _LINE_GROUPS:
            items = [(base.items[item_id], candidate.items[item_id]) for item_id in ids]
            for metric, (hit, total) in _ITEM_DELTAS.items():
                entry[metric] = _paired_delta(
                    per_unit(getattr(b, hit) for b, _ in items),
                    per_unit(getattr(c, hit) for _, c in items),
                    per_unit(getattr(b, total) for b, _ in items),
                )
            if kind == "silver":
                # Rate per audio second from whole-second totals, scaled to per hour.
                entry["uncued_per_h"] = _paired_delta(
                    per_unit(b.uncued for b, _ in items),
                    per_unit(c.uncued for _, c in items),
                    per_unit(round(b.audio_s) for b, _ in items),
                )
                for bound in ("delta", "lower", "upper"):
                    if entry["uncued_per_h"][bound] is not None:
                        entry["uncued_per_h"][bound] *= 3600.0
        entry["delay_mean"] = delay_delta()
        result[group] = entry
    return result


def _check_paired(base: _RunScore, candidate: _RunScore) -> None:
    if base.items.keys() != candidate.items.keys():
        raise ValueError(f"{candidate.run_dir} and {base.run_dir} cover different items")
    for item_id, item in base.items.items():
        other = candidate.items[item_id]
        if (
            [line.line_id for line in item.lines] != [line.line_id for line in other.lines]
            or item.reference_chars != other.reference_chars
        ):
            raise ValueError(f"{item_id}: references differ between {base.run_dir} and {candidate.run_dir}")


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _fmt_delta(delta: dict[str, Any] | None) -> str:
    if delta is None:
        return "n/a"
    if delta.get("not_comparable"):
        return f"not comparable ({delta['not_comparable']})"
    return (
        f"{_fmt(delta['delta'], signed=True)} "
        f"[{_fmt(delta['lower'], signed=True)}, {_fmt(delta['upper'], signed=True)}]"
    )


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines + [""]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _rate_table(
    metrics: dict[str, dict[str, Any]], groups: Sequence[str], rate: str, hits: str, total: str
) -> list[str]:
    """One row per candidate: ``rate (hits/total)`` per group."""
    return _table(
        ["candidate", *groups],
        (
            [label, *(
                "n/a" if m[g][hits] is None
                else f"{_fmt(m[g][rate])} ({m[g][hits]}/{m[g][total]})"
                for g in groups
            )]
            for label, m in metrics.items()
        ),
    )


def _length_rows() -> list[tuple[str, str, str | None]]:
    """Rows of the length tables: (label, group, length bin or None for the
    whole group). The bins hold word lines; each line kind gets a row."""
    return [
        *((f"words {label}", "kind=words", label) for label in _BIN_LABELS),
        *((kind, f"kind={kind}", None) for kind in LINE_KINDS),
        (_ALL, _ALL, None),
    ]


def _length_cell(per_group: dict[str, Any], group: str, length_bin: str | None) -> dict[str, Any]:
    """A length row's metrics (or deltas): its group's, or its bin's within the group."""
    return per_group[group] if length_bin is None else per_group[group]["length_bins"][length_bin]


def _length_delta_table(deltas: dict[str, dict[str, Any]], metric: str) -> list[str]:
    """One row per candidate: the paired delta of ``metric`` per length row."""
    rows = _length_rows()
    return _table(
        ["candidate", *(row_label for row_label, _, _ in rows)],
        (
            [label, *(
                _fmt_delta(_length_cell(per_group, group, length_bin)[metric])
                for _, group, length_bin in rows
            )]
            for label, per_group in deltas.items()
        ),
    )


def _keyword_tables(entry: dict[str, Any]) -> list[str]:
    """Key-word recall per lexicon entry and pooled, one column per
    candidate; then the intj miss causes per candidate."""
    metrics = entry["metrics"]
    baseline = metrics[entry["baseline"]][_ALL]
    out = ["### Key-word recall (interjection occurrences)", ""]
    if not baseline["keyword_occurrences"]:
        out += ["No lexicon interjection occurs in this set's references.", ""]
    else:
        rows = [
            [INTERJECTIONS[key][0], INTERJECTIONS[key][1], str(counts["occurrences"]), *(
                _fmt(m[_ALL]["keywords"][key]["recall"]) for m in metrics.values()
            )]
            for key, counts in baseline["keywords"].items()
        ]
        rows.append(["pooled", "", str(baseline["keyword_occurrences"]), *(
            _fmt(m[_ALL]["keyword_recall"]) for m in metrics.values()
        )])
        out += _table(["word", "category", "count", *metrics], rows)
        out += ["### Key-word recall by group", ""]
        out += _rate_table(
            metrics, entry["groups"], "keyword_recall", "keyword_recalled", "keyword_occurrences"
        )
    out += ["### Miss causes of `intj` lines (aligned reading rule)", ""]
    rows = []
    for label, m in metrics.items():
        intj = m["kind=intj"]
        causes = intj["miss_causes"]
        missed = sum(causes.values())
        rows.append([label, str(intj["lines"]), str(missed), *(
            f"{causes[cause]} ({_fmt(_ratio(causes[cause], missed))})" for cause in MISS_CAUSES
        )])
    out += _table(
        ["candidate", "intj lines", "missed", "no window", "empty window", "own window wrong",
         "merged, dropped or wrong"],
        rows,
    )
    return out


def _markdown(report: dict[str, Any]) -> str:
    scoring = report["scoring"]
    threshold = scoring["recall_threshold"]
    out = [
        "# ASR bench report",
        "",
        f"Created {report['created_at_utc']}. Development measurement only; the E0 human "
        "references remain the release gate.",
        "",
        "- Aligned recall (primary): each item's reference lines and its window texts, each "
        "concatenated in time order, are aligned once as a minimum-edit global alignment; a "
        f"line is caught when at least {threshold['synthetic']} (synthetic) or "
        f"{threshold['silver']} (silver) of its characters align as exact matches. Reading "
        f"form on katakana readings ({scoring['reading']}), so spelling differences (kana for "
        f"kanji) do not count as misses; text form on {scoring['normalization']} text.",
        "- Chance: the same reference scored against another item's hypothesis (the item "
        "`chance shift` positions later in set order): what a line catches from unrelated "
        "speech, for the aligned and the collar rule (reading form).",
        "- Reading length: the line's kana count, binned "
        f"{', '.join(scoring['length_bins'])}. Line kinds: `words`; vocalization-only lines "
        "(every token an interjection or symbol in UniDic) split into `intj`, holding a "
        "meaningful interjection of the lexicon (affirm, deny, surprise or question, call, "
        "hesitation that answers, greeting or thanks), and `nonlex` (grunts, gasps, "
        "laughter). The `kind=` columns select lines, the other columns select items.",
        "- Key-word recall: each lexicon interjection in any reference line counts as "
        "recalled when all of its reading characters but ッ align as exact matches.",
        "- Miss causes of a line the aligned reading rule misses: no window overlaps it; the "
        "windows over it are all empty; its own window (not merged) has the wrong text; it "
        "is merged into a window holding at least half of another line and its text is "
        "dropped or wrong.",
        f"- Collar recall (the earlier rule): the windows overlapping the line's span plus a "
        f"{scoring['collar_s']} s collar contain, as difflib matching blocks, at least the same "
        "share of its characters. Loose by construction: short lines pass by chance.",
        "- CER (synthetic) and subtitle CER (silver): edits of the same alignment over "
        "reference characters, pooled over items. Subtitle CER is inflated by subtitle "
        "editing; compare it between candidates only.",
        "- Hallucinated windows (synthetic): windows wholly inside a speech-free span with "
        "non-empty text.",
        "- Uncued windows (silver): windows with non-empty text overlapping no reference line "
        "even with the collar (hallucinations or speech the subtitles leave out), per hour "
        "of audio.",
        "- Line delay: for each line caught by the collar reading rule, the audio position at "
        "which the chunker emitted the last window overlapping the line, minus the line's end, "
        "plus that window's decode seconds; median, p90 and mean in seconds (n/a for runs "
        "without emission timing). With sound tags the emission waits for the tag delay line "
        "(the run's `emission` below); runs that say the tag look-ahead was left out "
        "understate it. A window emitted in pieces counts from the first piece after which "
        "its later pieces add no collar coverage to the line.",
        f"- Cut lines: a window boundary falls inside the line more than "
        f"{scoring['cut_margin_s']} s from both edges. Merged lines: the window holding the "
        f"largest share of the line also holds at least {scoring['merge_share']} of another "
        "line. Both over scored lines.",
        f"- Deltas: candidate minus baseline (the first run on the set), paired bootstrap over "
        f"clusters (when every item of the set has `condition.{CLUSTER_KEY}`; all items of a "
        f"cluster resampled together) or items, {scoring['bootstrap']['replicates']:,} "
        f"replicates, seed {scoring['bootstrap']['seed']}, 95% interval. The line delay delta "
        "is the mean over lines both runs catch, in seconds, per group and per length bin.",
        "- RTF: (VAD + ASR seconds) / audio seconds; model load and warmup excluded.",
        "- Timing is judged per GPU model and in one process: line delay deltas and the timing "
        "table against the baseline (ASR seconds as a ratio, RTF as a difference) are given only "
        "between runs on the same GPU model (the runs table's GPU; `not recorded` for runs from "
        "before the field), otherwise `not comparable (GPU differs)`, and only between runs of "
        "one worker process, otherwise `not comparable (parallel workers)`. A run with more "
        "workers (the runs table's `workers`) also shows its own line delay and RTF under the "
        "contention of its processes.",
        "",
    ]
    for entry in report["sets"]:
        groups = entry["groups"]
        item_groups = [g for g in groups if g not in entry["line_groups"]]
        kind = entry["kind"]
        metrics = entry["metrics"]
        shift = entry["chance_shift"]
        length_rows = _length_rows()
        out += [
            f"## Set `{entry['set_id']}` ({kind}, "
            f"{_count(entry['item_count'], 'item')}, {_count(entry['line_count'], 'line')})",
            "",
            f"Baseline: `{entry['baseline']}`. Bootstrap unit: "
            f"{_count(entry['bootstrap']['units'], entry['bootstrap']['unit'])}. Chance shift: "
            + ("none (no other item to pair with)." if shift is None else f"{shift}."),
            "",
            "### Aligned recall (reading)",
            "",
        ]
        out += _rate_table(metrics, groups, "aligned_recall_reading", "aligned_caught_reading", "lines")
        out += ["### Aligned recall by reading length", ""]
        rows = []
        for label, m in metrics.items():
            for row_label, group, length_bin in length_rows:
                cell = _length_cell(m, group, length_bin)
                rows.append([
                    label, row_label, str(cell["lines"]),
                    *(_fmt(cell[f"{rule}_recall_reading"]) for rule in _BIN_RULES),
                ])
        out += _table(
            ["candidate", "lines", "count", "aligned", "aligned chance", "collar",
             "collar chance"],
            rows,
        )
        out += _keyword_tables(entry)
        out += ["### Aligned recall (text)", ""]
        out += _rate_table(metrics, groups, "aligned_recall", "aligned_caught", "lines")
        out += ["### Chance recall (aligned, reading)", ""]
        out += _rate_table(
            metrics, groups, "chance_aligned_recall_reading", "chance_aligned_caught_reading", "lines"
        )
        if kind == "synthetic":
            out += ["### CER", ""]
            out += _rate_table(metrics, item_groups, "cer", "errors", "reference_chars")
            out += ["### Hallucinated windows", ""]
            out += _table(
                ["candidate", *item_groups],
                (
                    [label, *(
                        f"{m[g]['hallucinations']} in {m[g]['speech_free_s']:.1f} s"
                        for g in item_groups
                    )]
                    for label, m in metrics.items()
                ),
            )
        if kind == "silver":
            out += ["### Subtitle CER (reading)", ""]
            out += _rate_table(
                metrics, item_groups, "cer_reading", "errors_reading", "reference_chars_reading"
            )
            out += ["### Subtitle CER (text)", ""]
            out += _rate_table(metrics, item_groups, "cer", "errors", "reference_chars")
        out += ["### Collar recall (reading)", ""]
        out += _rate_table(metrics, groups, "collar_recall_reading", "collar_caught_reading", "lines")
        out += ["### Collar recall (text)", ""]
        out += _rate_table(metrics, groups, "collar_recall", "collar_caught", "lines")
        if kind == "silver":
            out += ["### Uncued windows", ""]
            out += _table(
                ["candidate", *item_groups],
                (
                    [label, *(
                        f"{m[g]['uncued']} ({_fmt(m[g]['uncued_per_h'], 1)}/h)" for g in item_groups
                    )]
                    for label, m in metrics.items()
                ),
            )
        out += ["### Line delay (s, median / p90)", ""]
        out += _table(
            ["candidate", *groups],
            (
                [label, *(
                    "n/a" if m[g]["delay"] is None
                    else f"{_fmt(m[g]['delay']['median_s'], 2)} / "
                    f"{_fmt(m[g]['delay']['p90_s'], 2)} ({m[g]['delay']['lines']})"
                    for g in groups
                )]
                for label, m in metrics.items()
            ),
        )
        out += ["### Line delay by reading length (s)", ""]
        rows = []
        for label, m in metrics.items():
            for row_label, group, length_bin in length_rows:
                cell = _length_cell(m, group, length_bin)
                delay = cell["delay"]
                rows.append([label, row_label, str(cell["lines"]), *(
                    ("n/a", "n/a", "n/a", "n/a") if delay is None else (
                        str(delay["lines"]), _fmt(delay["median_s"], 2), _fmt(delay["p90_s"], 2),
                        _fmt(delay["mean_s"], 2),
                    )
                )])
        out += _table(["candidate", "lines", "count", "timed", "median", "p90", "mean"], rows)
        for title, key in (("Cut lines", "cut"), ("Merged lines", "merged")):
            out += [f"### {title}", ""]
            out += _rate_table(metrics, groups, f"{key}_rate", key, "lines")
        out += ["### Runs", ""]
        out += _table(
            ["candidate", "windows", "audio s", "RTF", "VAD s", "ASR s", "load s", "GPU",
             "workers", "emission", "run"],
            (
                [
                    run["label"],
                    str(run["window_count"]),
                    f"{run['audio_seconds']:.1f}",
                    _fmt(run["timing"]["rtf"]),
                    f"{run['timing']['vad_seconds']:.1f}",
                    f"{run['timing']['asr_seconds']:.1f}",
                    f"{run['timing']['load_seconds']:.1f}",
                    run["gpu"],
                    str(run["workers"]) + ("" if run["timing_valid"] else " (timing not valid)"),
                    run["emission"],
                    f"`{Path(run['run_dir']).name}`",
                ]
                for run in report["runs"]
                if run["set_id"] == entry["set_id"]
            ),
        )
        if entry["timing_deltas"]:
            out += [f"### Timing vs `{entry['baseline']}` (same GPU model, one worker only)", ""]
            out += _table(
                ["candidate", "GPU", "ASR seconds", "RTF"],
                (
                    [label, next(run["gpu"] for run in report["runs"]
                                 if run["set_id"] == entry["set_id"] and run["label"] == label),
                     *((f"not comparable ({timing['not_comparable']})",) * 2
                       if timing["not_comparable"] else
                       (f"{_fmt(timing['asr_seconds_ratio'], 2)}x",
                        _fmt(timing["rtf_delta"], signed=True)))]
                    for label, timing in entry["timing_deltas"].items()
                ),
            )
        if entry["deltas"]:
            out += [f"### Paired deltas vs `{entry['baseline']}`", ""]
            shown = (
                "aligned_recall_reading", "aligned_recall", "keyword_recall",
                *(("cer_reading", "cer") if kind == "silver" else ("cer",)),
                "collar_recall_reading", "collar_recall", "uncued_per_h", "delay_mean",
            )
            name = {"delay_mean": "delay_mean (s)", "uncued_per_h": "uncued (/h)"}
            if kind == "silver":
                name.update(cer_reading="subtitle_cer_reading", cer="subtitle_cer")
            rows = []
            for label, per_group in entry["deltas"].items():
                for metric in shown:
                    if metric not in per_group[_ALL]:
                        continue
                    rows.append([label, name.get(metric, metric), *(
                        _fmt_delta(per_group[g].get(metric)) for g in groups
                    )])
            out += _table(["candidate", "metric", *groups], rows)
            out += [f"### Paired deltas of aligned recall (reading) by length vs `{entry['baseline']}`", ""]
            out += _length_delta_table(entry["deltas"], "aligned_recall_reading")
            out += [f"### Paired deltas of mean line delay (s) by length vs `{entry['baseline']}`", ""]
            out += _length_delta_table(entry["deltas"], "delay_mean")
            keywords = entry["deltas"][next(iter(entry["deltas"]))][_ALL]["keywords"]
            if keywords:
                out += [f"### Paired deltas of key-word recall vs `{entry['baseline']}`", ""]
                baseline = metrics[entry["baseline"]][_ALL]["keywords"]
                out += _table(
                    ["word", "count", *entry["deltas"]],
                    (
                        [INTERJECTIONS[key][0], str(baseline[key]["occurrences"]), *(
                            _fmt_delta(per_group[_ALL]["keywords"][key]["keyword_recall"])
                            for per_group in entry["deltas"].values()
                        )]
                        for key in keywords
                    ),
                )
    return "\n".join(out)


def score(run_dirs: Sequence[Path]) -> dict[str, Any]:
    runs: list[_RunScore] = []
    for run_dir in run_dirs:
        record, bench_set, items = _load_run(run_dir)
        runs.append(_RunScore(record["candidate"]["name"], run_dir, record, bench_set, items))

    set_order: list[str] = []
    by_set: dict[str, list[_RunScore]] = {}
    for run in runs:
        set_id = run.bench_set.set_id
        if set_id not in by_set:
            set_order.append(set_id)
            by_set[set_id] = []
        by_set[set_id].append(run)

    report_runs: list[dict[str, Any]] = []
    report_sets: list[dict[str, Any]] = []
    for set_id in set_order:
        set_runs = by_set[set_id]
        names = [run.label for run in set_runs]
        labelled = [
            run if names.count(run.label) == 1
            else _RunScore(f"{run.label} ({run.run_dir.name})", run.run_dir, run.record,
                           run.bench_set, run.items)
            for run in set_runs
        ]
        base = labelled[0]
        bench_set = base.bench_set
        kind = bench_set.kind
        groups = _group_order(bench_set)
        unit, unit_of = _bootstrap_units(bench_set)
        for run in labelled[1:]:
            _check_paired(base, run)
        report_sets.append({
            "set_id": set_id,
            "kind": kind,
            "path": str(bench_set.root),
            "item_count": len(bench_set.items),
            "line_count": sum(len(item.lines) for item in bench_set.items),
            "groups": groups,
            "line_groups": [group for group in groups if group in _LINE_GROUPS],
            "baseline": base.label,
            "bootstrap": {"unit": unit, "units": len(set(unit_of.values()))},
            "chance_shift": _chance_shift(bench_set),
            "metrics": {
                run.label: {
                    group: _aggregate(run.items.values(), group, kind) for group in groups
                }
                for run in labelled
            },
            "deltas": {
                run.label: _deltas(base, run, groups, kind, unit_of) for run in labelled[1:]
            },
            "timing_deltas": {
                run.label: _timing_delta(base.record, run.record) for run in labelled[1:]
            },
            "missed_lines": {
                run.label: [
                    {"item_id": item.item_id, **line}
                    for item in run.items.values()
                    for line in item.missed
                ]
                for run in labelled
            },
        })
        for run in labelled:
            report_runs.append({
                "label": run.label,
                "run_dir": str(run.run_dir),
                "set_id": set_id,
                "candidate": run.record["candidate"],
                "window_count": run.record["window_count"],
                "audio_seconds": run.record["set"]["audio_seconds"],
                "timing": run.record["timing"],
                "frontend": run.record.get("frontend"),
                "emission": _emission(run.record),
                "gpu": _gpu(run.record),
                "workers": run.record.get("workers", 1),
                "timing_valid": _timing_valid(run.record),
                "git": run.record["git"],
                "packages": run.record["packages"],
            })
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scoring": {
            "normalization": JA_MINIMAL_1_1.version,
            "reading": READING_POLICY,
            "recall_threshold": RECALL_THRESHOLD,
            "aligned_recall": "share of a line's characters aligned as exact matches in one "
                              "minimum-edit global alignment of the item's time-ordered "
                              "reference lines with its time-ordered window texts (tie-break "
                              "match, substitution, deletion, insertion from the start)",
            "chance": "each item's reference against the hypothesis of the item chance_shift "
                      "positions later in set order (cyclic), the smallest shift pairing no "
                      "item with an item holding the same reference",
            "length_bins": list(_BIN_LABELS),
            "length_bin_basis": "kana count of the line's reading (its minimal-normalized text "
                                "when the reading is empty)",
            "line_kinds": "words unless every token of the minimal-normalized line has UniDic "
                          f"pos1 in {sorted(VOCAL_POS1)}; such a vocalization-only line is "
                          "intj when it holds an interjection of the lexicon, else nonlex; "
                          "kind= groups select lines",
            "interjections": {
                key: {"forms": forms, "category": category}
                for key, (forms, category) in INTERJECTIONS.items()
            },
            "keyword_recall": "per 感動詞 token whose reading (ッ removed, small vowels "
                              "enlarged) is an interjections key, or repeats a multi-kana key: "
                              "recalled when all its reading characters but ッ align as exact "
                              "matches in the aligned-recall alignment",
            "miss_causes": "per line missed by the aligned reading rule: no_window (no window "
                           "overlaps it), empty_window (all overlapping windows empty), "
                           "own_window (not merged, wrong text), merged_window (merged, its "
                           "text dropped or wrong)",
            "collar_s": COLLAR_S,
            "collar_recall": "share of a line's characters inside difflib matching blocks of "
                             "the windows overlapping its span widened by collar_s",
            "cer": "edit distance of the aligned-recall alignment over reference characters, "
                   "pooled over items; on silver sets it is subtitle CER",
            "line_delay": "emitted_at_s of the last window overlapping a line caught by the "
                          "collar reading rule minus the line's end_s, plus that window's asr_s; "
                          "for windows emitted in pieces (emissions), the latest over the windows "
                          "overlapping the line of the emitted_at_s + asr_s of each window's first "
                          "piece after which its later pieces add no collar reading coverage; "
                          "median, p90 and mean per group and per length bin",
            "cut_margin_s": CUT_MARGIN_S,
            "merge_share": MERGE_SHARE,
            "timing_comparability": "line delay deltas, asr_seconds_ratio and rtf_delta only "
                                    "between runs with the same GPU model (run.json gpu.model, "
                                    "else remote.gpu_model, else 'not recorded') and both with "
                                    "timing_valid (one worker process; true for runs before "
                                    "run schema 1.3)",
            "uncued_windows": "silver: windows with non-empty minimal-normalized text that "
                              "overlap no reference line within collar_s, per hour of audio",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "interval": list(_INTERVAL),
                "quantile": "Hyndman-Fan type 7",
                "unit": f"cluster when every item of a set has condition.{CLUSTER_KEY}, else item",
            },
        },
        "runs": report_runs,
        "sets": report_sets,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, type=Path,
                        help="run directory; the first run on each set is its baseline")
    parser.add_argument("--out", required=True, type=Path, help="report directory")
    args = parser.parse_args(argv)
    report = score(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    markdown = _markdown(report)
    (args.out / "report.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
