"""Incremental emission: commit whole sentences while a VAD window is open.

The production chunker emits a window only when it closes, so a line that
ends early in a long window waits for the rest of the window. This bench
strategy decodes the open window as it grows and emits a sentence as soon
as it is stable, then emits only the rest at the close.

While a window is open (from ``Window.opened_at_sample``, the moment the
gate held it open, up to its emission), every ``step_ms`` of audio clock
the strategy decodes the window's audio from its start up to that moment
(``audio: "clock"``, the default: all audio fed by then) or up to the
frames the gate has decided by then (``audio: "decided"``). The two differ
only when the gate decides frames after they are fed, as behind the
sound-tag delay line (``StrategyContext.decision_sample``); a clock decode
may then reach past the window's end into the next speech. Decodes go
through the candidate's own decoder (production ``transcribe`` for
faster-whisper).
With ``confirm_ms``, a decode that leaves a complete sentence uncommitted
is followed by the next one after ``confirm_ms`` instead, so agreement can
come sooner. A step whose audio did not grow is skipped. The sentence
split, the commit rule and the final strip are production's
(``yamabiko/asr/incremental.py``: ``split_sentences``, ``step_commits``,
``strip_committed``), so the bench measures the rule that ships. Each
decode is split into sentences by production's ``SentenceSplitter``
(fast-bunkai) after the already committed text is stripped from its front
(see below). A sentence is committed when it ends with sentence-final
punctuation (``has_terminator``: 。！？?!) and:

- ``agree`` (LocalAgreement-2 at sentence level, production's
  ``step_commits``): it and every sentence before it are identical
  (whitespace stripped) in this decode and the previous one, both read
  against the same committed text; or
- ``gap`` (bench only, with ``gap_ms``): the decided frames end with at least
  ``gap_ms`` of speech probability below the gate's sustain bar
  (``max(threshold - 0.15, 0.01)``, what the gate counts as silence) after
  speech inside the window; then every complete sentence of this one decode
  commits.

Committed sentences go out at once: the emission's ``emitted_at_s`` is the
step's audio clock and its ``asr_s`` that decode's wall time. At the close
the whole window is decoded once, exactly as without the strategy (the
final decode). The committed text is aligned to the front of the final
decode (an edit-distance alignment on content characters, punctuation and
spaces ignored, of the committed text against every prefix of the decode;
the shortest prefix with the lowest cost wins) and the final emission is
the rest of the final decode, at the window's own ``emitted_at_s``.

The window's ``text`` is everything emitted (committed sentences, then the
final rest), so recall, CER and uncued windows score all emitted text. Each
record adds ``emissions`` (``text``, ``emitted_at_s``, ``asr_s``, ``rule``)
in order, whose texts join to ``text``; the scorer takes a line's delay from
the first emission after which the window's later emissions add nothing to
the line (``score._line_delay``). ``asr_s`` becomes the final decode's wall
time; ``asr_steps_s`` holds the step decodes'. A record also keeps the
final decode's text and outcome and, for a window with commits, the
commit's cost against the final decode, the emitted text against the final
decode (extra and missing content characters) and the committed sentences
found again in the final rest (duplicates).

Summary of the commit counts of finished runs::

    python -m scripts.asr_bench.incremental --run <run dir> [--run <run dir> ...]
"""

from __future__ import annotations

import argparse
import bisect
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.asr_bench.align import alignment_trace
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, Item
from yamabiko.asr.incremental import (
    complete_prefix,
    content,
    split_sentences,
    step_commits,
    strip_committed,
)
from yamabiko.sentencing.splitter import SentenceSplitter

if TYPE_CHECKING:
    from scripts.asr_bench.run import StrategyContext, Window

MODE = "incremental_emit"
STEP_MS_RANGE = (100, 4000)
GAP_MS_RANGE = (32, 2000)
# A committed sentence counts as duplicated when its content, at least this
# many characters, appears again in the final rest.
DUPLICATE_MIN_CHARS = 2
_KEYS = frozenset({"mode", "step_ms", "gap_ms", "confirm_ms", "audio"})
AUDIO_MODES = ("clock", "decided")


@dataclass(frozen=True)
class IncrementalEmitSpec:
    """The ``strategy`` of an ``incremental_emit`` candidate."""

    step_ms: int
    # None: LocalAgreement-2 only.
    gap_ms: int | None = None
    # None: every decode follows the previous one by step_ms.
    confirm_ms: int | None = None
    # How far each step decodes: the audio fed by then, or the decided frames.
    audio: str = "clock"
    mode: str = MODE

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "step_ms": self.step_ms, "gap_ms": self.gap_ms,
                "confirm_ms": self.confirm_ms, "audio": self.audio}


def _bounded_int(raw: Mapping[str, Any], key: str, bounds: tuple[int, int], where: str) -> int:
    value = raw[key]
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{where}.{key} must be an int in [{low}, {high}]")
    return value


def parse_strategy(raw: Mapping[str, Any], where: str) -> IncrementalEmitSpec:
    """Parse ``{"mode": "incremental_emit", "step_ms": N, "gap_ms": M | null,
    "confirm_ms": C | null, "audio": "clock" | "decided"}`` (``confirm_ms`` at
    most ``step_ms``)."""
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    if "step_ms" not in raw:
        raise ValueError(f"{where}.step_ms is required")
    gap_ms = None if raw.get("gap_ms") is None else _bounded_int(raw, "gap_ms", GAP_MS_RANGE, where)
    step_ms = _bounded_int(raw, "step_ms", STEP_MS_RANGE, where)
    confirm_ms = None
    if raw.get("confirm_ms") is not None:
        confirm_ms = _bounded_int(raw, "confirm_ms", (STEP_MS_RANGE[0], step_ms), where)
    audio = raw.get("audio", "clock")
    if audio not in AUDIO_MODES:
        raise ValueError(f"{where}.audio must be one of {AUDIO_MODES}")
    return IncrementalEmitSpec(step_ms=step_ms, gap_ms=gap_ms, confirm_ms=confirm_ms, audio=audio)


def compare_to_final(emitted: str, final: str) -> dict[str, int]:
    """Content characters of the emitted text against the final decode:
    edit distance, emitted characters without an exact match in the final
    decode (insertions and substitutions) and final characters without one
    in the emitted text (deletions and substitutions)."""
    emitted_content, _ = content(emitted)
    final_content, _ = content(final)
    matched, _, distance = alignment_trace(final_content, emitted_content)
    hits = int(matched.sum())
    return {
        "edit_distance": distance,
        "extra_chars": len(emitted_content) - hits,
        "missing_chars": len(final_content) - hits,
    }


def duplicated(committed: Sequence[str], rest: str) -> list[str]:
    """Committed sentences whose content appears again in the final rest."""
    rest_content, _ = content(rest)
    found: list[str] = []
    for sentence in committed:
        sentence_content, _ = content(sentence)
        if len(sentence_content) >= DUPLICATE_MIN_CHARS and sentence_content in rest_content:
            found.append(sentence)
    return found


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _percentile(values: Sequence[float], share: float) -> float | None:
    return float(np.percentile(values, 100 * share)) if values else None


class IncrementalEmit:
    """The ``incremental_emit`` window strategy of the bench runner."""

    def __init__(self, context: StrategyContext) -> None:
        spec = context.candidate.strategy
        if not isinstance(spec, IncrementalEmitSpec):
            raise TypeError("incremental_emit needs an IncrementalEmitSpec")
        self._spec = spec
        self._decode = context.decode
        self._decision_sample = context.decision_sample
        self._frame_samples = context.frame_samples
        self._threshold = context.threshold
        self._sustain = max(context.threshold - 0.15, 0.01)
        frame_ms = 1000 * context.frame_samples / SAMPLE_RATE_HZ
        self._gap_frames = None if spec.gap_ms is None else max(1, int(spec.gap_ms / frame_ms))
        self._step_samples = SAMPLE_RATE_HZ * spec.step_ms // 1000
        self._confirm_samples = SAMPLE_RATE_HZ * (spec.confirm_ms or spec.step_ms) // 1000
        self._splitter = SentenceSplitter()
        self._audio = np.zeros(0, np.float32)
        self._probabilities = np.zeros(0)
        self._decided: list[int] = []
        self._counts = {
            "windows": 0, "windows_with_commits": 0, "step_decodes": 0, "steps_without_new_audio": 0,
            "confirm_steps": 0, "step_overruns": 0, "commits_agree": 0, "commits_gap": 0,
            "commits_past_window_end": 0,
            "committed_sentences": 0, "committed_chars": 0, "windows_commit_mismatch": 0,
            "commit_cost_chars": 0, "duplicate_sentences": 0, "duplicate_chars": 0,
            "emitted_vs_final_edit_distance": 0, "emitted_extra_chars": 0, "emitted_missing_chars": 0,
        }
        self._step_seconds: list[float] = []
        self._final_seconds = 0.0
        self._lead_seconds: list[float] = []

    def start_item(self, item: Item, asr_audio: np.ndarray, probabilities: np.ndarray) -> None:
        self._audio = asr_audio
        self._probabilities = probabilities
        # Frames the gate has decided by audio position t: bisect_right(decided, t).
        self._decided = [
            self._decision_sample(frame, asr_audio.size) for frame in range(probabilities.size)
        ]

    def _gap_closed(self, start_frame: int, frontier: int) -> bool:
        """Whether the decided frames end with the gap after speech in the window."""
        if self._gap_frames is None:
            return False
        gap_start = frontier - self._gap_frames
        if gap_start <= start_frame:
            return False
        if np.any(self._probabilities[gap_start:frontier] >= self._sustain):
            return False
        return bool(np.any(self._probabilities[start_frame:gap_start] >= self._threshold))

    def _commits(self, committed: str, previous: str | None, decoded: str,
                 gap: bool) -> tuple[list[str], str | None, bool]:
        """The sentences this decode commits, by which rule, and whether it
        leaves a complete sentence uncommitted. ``agree`` is production's rule
        (yamabiko.asr.incremental.step_commits); ``gap`` is bench-only."""
        if gap:
            current = split_sentences(self._splitter, decoded[strip_committed(committed, decoded)[0]:])
            complete = complete_prefix(current)
            if complete:
                return complete, "gap", False
        found, pending = step_commits(self._splitter, committed, previous, decoded)
        return found, "agree" if found else None, pending

    def decode_window(self, window: Window) -> dict[str, Any]:
        fs = self._frame_samples
        start = window.start_sample
        start_frame = start // fs
        committed: list[str] = []
        emissions: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        previous: str | None = None
        last_end = start
        interval = self._step_samples
        clock = window.opened_at_sample + interval
        while clock < window.emitted_at_sample:
            frontier = bisect.bisect_right(self._decided, clock)
            if self._spec.audio == "decided":
                end = min(frontier * fs, window.end_sample)
            else:
                end = min(clock, self._audio.size)
            if end <= last_end:
                self._counts["steps_without_new_audio"] += 1
                clock += interval
                continue
            begin = time.perf_counter()
            decoded, outcome = self._decode(self._audio[start:end])
            seconds = time.perf_counter() - begin
            self._counts["step_decodes"] += 1
            self._step_seconds.append(seconds)
            gap = self._gap_closed(start_frame, frontier)
            found, rule, pending = self._commits("".join(committed), previous, decoded, gap)
            interval = self._confirm_samples if pending else self._step_samples
            self._counts["step_overruns"] += seconds * SAMPLE_RATE_HZ > interval
            self._counts["confirm_steps"] += pending and interval < self._step_samples
            steps.append({
                "at_s": clock / SAMPLE_RATE_HZ, "audio_end_s": end / SAMPLE_RATE_HZ,
                "text": decoded, "asr_outcome": outcome, "asr_s": seconds, "gap": gap,
                "committed": len(found),
            })
            if found:
                self._counts[f"commits_{rule}"] += 1
                self._counts["commits_past_window_end"] += end > window.end_sample
                self._counts["committed_sentences"] += len(found)
                self._counts["committed_chars"] += sum(len(content(s)[0]) for s in found)
                committed.extend(found)
                emissions.append({
                    "text": "".join(found), "emitted_at_s": clock / SAMPLE_RATE_HZ,
                    "asr_s": seconds, "rule": rule,
                })
            previous = decoded
            last_end = end
            clock += interval

        begin = time.perf_counter()
        final, outcome = self._decode(self._audio[start:window.end_sample])
        final_seconds = time.perf_counter() - begin
        self._final_seconds += final_seconds
        committed_text = "".join(committed)
        cut, cost = strip_committed(committed_text, final)
        rest = final[cut:]
        emitted_at_s = window.emitted_at_sample / SAMPLE_RATE_HZ
        emissions.append({"text": rest, "emitted_at_s": emitted_at_s, "asr_s": final_seconds,
                          "rule": "final"})
        self._counts["windows"] += 1
        fields: dict[str, Any] = {
            "text": committed_text + rest,
            "asr_outcome": outcome,
            "asr_s": final_seconds,
            "asr_steps_s": sum(step["asr_s"] for step in steps),
            "emissions": emissions,
            "final_text": final,
            "steps": steps,
        }
        if committed:
            duplicates = duplicated(committed, rest)
            versus = compare_to_final(committed_text + rest, final)
            counts = self._counts
            counts["windows_with_commits"] += 1
            counts["windows_commit_mismatch"] += cost > 0
            counts["commit_cost_chars"] += cost
            counts["duplicate_sentences"] += len(duplicates)
            counts["duplicate_chars"] += sum(len(content(s)[0]) for s in duplicates)
            counts["emitted_vs_final_edit_distance"] += versus["edit_distance"]
            counts["emitted_extra_chars"] += versus["extra_chars"]
            counts["emitted_missing_chars"] += versus["missing_chars"]
            # How much earlier than the window the first sentence went out.
            self._lead_seconds.append(emitted_at_s - emissions[0]["emitted_at_s"])
            fields["commit"] = {
                "cost_chars": cost, "final_cut": cut, "duplicates": duplicates, **versus,
            }
        return fields

    def record(self) -> dict[str, Any]:
        steps = self._step_seconds
        return {
            **self._spec.to_json(),
            "commit_rule_module": "yamabiko.asr.incremental",
            "splitter": {"module": "yamabiko.sentencing.splitter.SentenceSplitter",
                         "fast_bunkai": _package_version("fast-bunkai"),
                         "terminators": "has_terminator: 。！？?!"},
            "clock": "steps every step_ms from Window.opened_at_sample while before "
                     "emitted_at_sample (confirm_ms after a decode that leaves a complete sentence "
                     "uncommitted); each decodes start_sample up to the step (audio clock) or up to "
                     "the frames decided by then (audio decided, StrategyContext.decision_sample); "
                     "a step without new audio is skipped",
            "agree_rule": "LocalAgreement-2: longest common prefix of complete sentences of this "
                          "decode and the previous one, both after stripping the committed text",
            "gap_rule": None if self._gap_frames is None else {
                "frames": self._gap_frames, "below": self._sustain, "after_speech_at": self._threshold,
            },
            "final_rule": "decode the whole window; drop the prefix aligned to the committed text "
                          "(content characters, edit distance, shortest prefix of lowest cost)",
            "duplicate_min_chars": DUPLICATE_MIN_CHARS,
            "counts": dict(self._counts),
            "step_asr_seconds": sum(steps),
            "step_asr_mean_s": sum(steps) / len(steps) if steps else None,
            "step_asr_p90_s": _percentile(steps, 0.9),
            "step_asr_max_s": max(steps) if steps else None,
            "final_asr_seconds": self._final_seconds,
            "first_commit_lead_s": {
                "windows": len(self._lead_seconds),
                "median": _percentile(self._lead_seconds, 0.5),
                "p90": _percentile(self._lead_seconds, 0.9),
                "mean": (sum(self._lead_seconds) / len(self._lead_seconds)
                         if self._lead_seconds else None),
            },
        }


def _cell(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summary(run_dirs: Sequence[Path]) -> str:
    """A markdown table of the commit counts in each run's ``strategy`` section."""
    header = (
        "| run | step / gap / confirm ms, audio | windows | with commits | commits agree / gap "
        "| sentences (chars) "
        "| mismatch windows | commit cost chars | extra / missing chars vs final "
        "| duplicates (chars) | first commit lead s median / p90 | step decodes | overruns "
        "| step decode s mean / p90 | RTF |"
    )
    rows = [header, "|" + "---|" * 15]
    for run_dir in run_dirs:
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        section = record.get("strategy") or {}
        if section.get("mode") != MODE:
            raise ValueError(f"{run_dir} is not an {MODE} run")
        c = section["counts"]
        lead = section["first_commit_lead_s"]
        rows.append(
            f"| {record['candidate']['name']} | {section['step_ms']} / {section['gap_ms'] or '-'} "
            f"/ {section.get('confirm_ms') or '-'}, {section.get('audio', 'decided')} "
            f"| {c['windows']} | {c['windows_with_commits']} "
            f"| {c['commits_agree']} / {c['commits_gap']} "
            f"| {c['committed_sentences']} ({c['committed_chars']}) "
            f"| {c['windows_commit_mismatch']} | {c['commit_cost_chars']} "
            f"| {c['emitted_extra_chars']} / {c['emitted_missing_chars']} "
            f"| {c['duplicate_sentences']} ({c['duplicate_chars']}) "
            f"| {_cell(lead['median'])} / {_cell(lead['p90'])} "
            f"| {c['step_decodes']} | {c['step_overruns']} "
            f"| {_cell(section['step_asr_mean_s'], 3)} / {_cell(section['step_asr_p90_s'], 3)} "
            f"| {record['timing']['rtf']:.3f} |"
        )
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize incremental_emit runs.")
    parser.add_argument("--run", action="append", required=True, type=Path, help="run directory")
    args = parser.parse_args(argv)
    print(summary(args.run), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
