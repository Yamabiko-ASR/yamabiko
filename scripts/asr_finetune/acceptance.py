"""Acceptance verdict of the ASR fine-tuning pilot: gates A1-A6 of a fine-tuned
candidate against the base model, from the two models' bench runs.

    python -m scripts.asr_finetune.acceptance --candidate ftp-v2a [--baseline ftp-base]
        [--runs <bench root>\\runs] [--prefix ft-pilot-v1]
        [--pair <set> <baseline run dir> <candidate run dir> ...] --out <dir>

Input is run directories (``run.json`` and ``windows.jsonl``). A set's pair is
``<runs>\\<prefix>-<set>\\<baseline>`` and ``<runs>\\<prefix>-<set>\\<candidate>``,
or the two directories ``--pair`` names for that set. The gate sets
``silver-v5``, ``synthetic-v1`` and ``probe-cv8`` are required; the reported
sets ``reazon-test-v1`` and ``ingcrowd-v1`` join the tables when both of their
runs exist. Before anything is scored, every pair must be on exactly its set
(``run.json`` ``set.set_id``) and its runs must be ``--baseline`` and
``--candidate`` in that order (``run.json`` ``candidate.name``).

Each pair is scored here by ``scripts.asr_bench.score`` (its ``main`` with
``--run <baseline> --run <candidate> --out <out>\\<set>``, the baseline first),
which writes ``report.json`` and ``report.md`` there; the gates read that
``report.json``. Its deltas are candidate minus baseline with the bench's paired
bootstrap (10,000 replicates, seed 105, clusters or items, 95% interval).

The report has no pooled 13+ band. The pooled delta of A2 comes from the
per-line scores of the same two runs (``score._load_run``): the word lines
(``kind=words``) of the bands 13-20 and 21+ together, caught by the aligned
reading rule, summed per bootstrap unit of the report (``score._bootstrap_units``,
in the report's unit order) and resampled by ``score._paired_delta``, so it
uses the same draws as the report's band deltas. Its point values are the sums
of the two bands' report counts.

Gates (silver-v5 unless named; "lower" and "upper" are the 95% bounds of the
paired delta; "recall" is aligned recall, reading form):

- A1: ``kind=intj`` recall of the candidate above 0.576 (the shipped config's)
  and its delta's lower bound above 0.
- A2: word-line recall, upper bound at least 0 for the band 13-20, the band
  21+ and the pooled 13+ lines.
- A3: uncued windows per hour, lower bound at most 0.
- A4: synthetic-v1 recall (``all``) upper bound at least 0, CER (text form)
  lower bound at most 0, hallucinated windows not above the baseline's;
  probe-cv8 (one item, zero-width interval) caught lines not below and
  reading-form character errors not above the baseline's.
- A5: word-line recall of the bands 1-3, 4-6 and 7-12 and ``kind=nonlex``
  recall, upper bound at least 0; subtitle CER (reading form) lower bound at
  most 0; key-word recall (``all``) upper bound at least 0.
- A6: matches of ``pilot.PHANTOM_FAMILY`` (non-overlapping ``findall`` over
  each window's ``text`` in ``windows.jsonl``, summed) of the candidate not
  above the baseline's.

A bound or value that does not exist (a group without lines) fails its check.
PASS when every gate passes.

Writes ``<out>\\acceptance.json`` (``verdict``; ``gates``, each with ``pass``
and its checks' values, delta and rule; ``sets`` with the runs, report and
pooled delta; ``skipped`` reported sets; ``phantom_family`` per set and run)
and ``acceptance.md`` (the gate table; per set the runs, every word-line band
with the pooled 13+ lines, ``kind=intj`` and ``kind=nonlex`` with paired
intervals, the set metrics and key-word recall per lexicon entry). Exit 0 on
PASS, 1 on FAIL (a gate failed), 2 on anything else: a gate set's run
directory without ``run.json`` or ``windows.jsonl``, a run on another set or
with another name than its side's, runs that cover different items, or any
other error.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench import score
from scripts.asr_bench.score import _bootstrap_units, _load_run, _paired_delta
from scripts.asr_bench.sets import bench_root
from scripts.asr_finetune.pilot import PHANTOM_FAMILY, PILOT

SCHEMA = "ft-pilot-acceptance-1"
# silver-v5 kind=intj aligned recall (reading) of the shipped config, 298/517.
SHIPPED_INTJ_RECALL = 0.576
GATE_SETS = ("silver-v5", "synthetic-v1", "probe-cv8")
REPORTED_SETS = ("reazon-test-v1", "ingcrowd-v1")
SHORT_BANDS = ("1-3", "4-6", "7-12")
LONG_BANDS = ("13-20", "21+")
RECALL = "aligned_recall_reading"
WORDS = "kind=words"
# Ratio metrics of report.json: metric -> (hits, total) fields of the same cell.
_RATIOS = {
    RECALL: ("aligned_caught_reading", "lines"),
    "cer": ("errors", "reference_chars"),
    "cer_reading": ("errors_reading", "reference_chars_reading"),
    "keyword_recall": ("keyword_recalled", "keyword_occurrences"),
}
_RUN_FILES = ("run.json", "windows.jsonl")


class InputError(Exception):
    """An input is missing or does not fit (exit 2)."""


@dataclass(frozen=True)
class SetResult:
    """One set's pair as scored: the ``report.json`` set entry, the report
    labels of the two runs, the pooled 13+ word-line delta and the phantom
    family counts per label."""

    set_id: str
    entry: Mapping[str, Any]
    baseline: str
    candidate: str
    pooled: Mapping[str, Any]
    phantom: Mapping[str, Mapping[str, int]]
    runs: Sequence[Mapping[str, Any]] = ()
    report_dir: Path | None = None

    def cell(self, label: str, group: str, band: str | None = None) -> Mapping[str, Any]:
        cell = self.entry["metrics"][label][group]
        return cell if band is None else cell["length_bins"][band]

    def delta(self, group: str, metric: str, band: str | None = None) -> Mapping[str, Any]:
        cell = self.entry["deltas"][self.candidate][group]
        return (cell if band is None else cell["length_bins"][band])[metric]

    def pooled_value(self, label: str) -> dict[str, Any]:
        """The 13+ word lines of ``label``: the two long bands' counts summed."""
        bands = [self.cell(label, WORDS, band) for band in LONG_BANDS]
        hits, total = sum(b["aligned_caught_reading"] for b in bands), sum(b["lines"] for b in bands)
        return {"rate": hits / total if total else None, "hits": hits, "total": total}


def _value(cell: Mapping[str, Any], metric: str) -> Any:
    if metric not in _RATIOS:
        return cell[metric]
    hits, total = _RATIOS[metric]
    return {"rate": cell[metric], "hits": cell[hits], "total": cell[total]}


def _upper_ok(delta: Mapping[str, Any]) -> bool:
    return delta["upper"] is not None and delta["upper"] >= 0


def _lower_ok(delta: Mapping[str, Any]) -> bool:
    return delta["lower"] is not None and delta["lower"] <= 0


def _check(set_id: str, name: str, rule: str, passed: bool, baseline: Any = None, candidate: Any = None,
           delta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "set": set_id,
        "check": name,
        "baseline": baseline,
        "candidate": candidate,
        "delta": None if delta is None else {key: delta[key] for key in ("delta", "lower", "upper")},
        "rule": rule,
        "pass": bool(passed),
    }


def _metric_check(result: SetResult, name: str, group: str, metric: str, band: str | None = None,
                  rule: str = "upper >= 0") -> dict[str, Any]:
    """A paired-delta check: ``upper >= 0`` (a rate that must not fall) or
    ``lower <= 0`` (one that must not rise)."""
    ok: Callable[[Mapping[str, Any]], bool] = {"upper >= 0": _upper_ok, "lower <= 0": _lower_ok}[rule]
    delta = result.delta(group, metric, band)
    return _check(result.set_id, name, rule, ok(delta), _value(result.cell(result.baseline, group, band), metric),
                  _value(result.cell(result.candidate, group, band), metric), delta)


def gate_a1(silver: SetResult) -> list[dict[str, Any]]:
    base, cand = (_value(silver.cell(label, "kind=intj"), RECALL) for label in (silver.baseline, silver.candidate))
    delta = silver.delta("kind=intj", RECALL)
    return [
        _check(silver.set_id, "kind=intj recall", f"candidate > {SHIPPED_INTJ_RECALL}",
               cand["rate"] is not None and cand["rate"] > SHIPPED_INTJ_RECALL, base, cand),
        _check(silver.set_id, "kind=intj recall, paired delta", "lower > 0",
               delta["lower"] is not None and delta["lower"] > 0, base, cand, delta),
    ]


def gate_a2(silver: SetResult) -> list[dict[str, Any]]:
    return [
        *(_metric_check(silver, f"word lines {band} recall", WORDS, RECALL, band) for band in LONG_BANDS),
        _check(silver.set_id, "word lines 13+ pooled recall", "upper >= 0", _upper_ok(silver.pooled),
               silver.pooled_value(silver.baseline), silver.pooled_value(silver.candidate), silver.pooled),
    ]


def gate_a3(silver: SetResult) -> list[dict[str, Any]]:
    return [_metric_check(silver, "uncued windows per hour", "all", "uncued_per_h", rule="lower <= 0")]


def gate_a4(synthetic: SetResult, probe: SetResult) -> list[dict[str, Any]]:
    hallucinated = [synthetic.cell(label, "all")["hallucinations"] for label in (synthetic.baseline, synthetic.candidate)]
    caught = [_value(probe.cell(label, "all"), RECALL) for label in (probe.baseline, probe.candidate)]
    errors = [probe.cell(label, "all")["errors_reading"] for label in (probe.baseline, probe.candidate)]
    return [
        _metric_check(synthetic, "all lines recall", "all", RECALL),
        _metric_check(synthetic, "CER (text)", "all", "cer", rule="lower <= 0"),
        _check(synthetic.set_id, "hallucinated windows", "candidate <= baseline",
               hallucinated[1] <= hallucinated[0], *hallucinated),
        _check(probe.set_id, "caught lines (aligned, reading)", "candidate >= baseline",
               caught[1]["hits"] >= caught[0]["hits"], *caught, probe.delta("all", RECALL)),
        _check(probe.set_id, "reading-form character errors", "candidate <= baseline",
               errors[1] <= errors[0], *errors, probe.delta("all", "cer_reading")),
    ]


def gate_a5(silver: SetResult) -> list[dict[str, Any]]:
    return [
        *(_metric_check(silver, f"word lines {band} recall", WORDS, RECALL, band) for band in SHORT_BANDS),
        _metric_check(silver, "kind=nonlex recall", "kind=nonlex", RECALL),
        _metric_check(silver, "subtitle CER (reading)", "all", "cer_reading", rule="lower <= 0"),
        _metric_check(silver, "key-word recall", "all", "keyword_recall"),
    ]


def gate_a6(silver: SetResult) -> list[dict[str, Any]]:
    base, cand = (silver.phantom[label] for label in (silver.baseline, silver.candidate))
    return [_check(silver.set_id, "phantom family matches", "candidate <= baseline",
                   cand["matches"] <= base["matches"], dict(base), dict(cand))]


def evaluate(sets: Mapping[str, SetResult]) -> dict[str, dict[str, Any]]:
    """Every gate with its checks; a gate passes when all of its checks pass."""
    silver = sets["silver-v5"]
    checks = {
        "A1": gate_a1(silver),
        "A2": gate_a2(silver),
        "A3": gate_a3(silver),
        "A4": gate_a4(sets["synthetic-v1"], sets["probe-cv8"]),
        "A5": gate_a5(silver),
        "A6": gate_a6(silver),
    }
    return {gate: {"pass": all(check["pass"] for check in found), "checks": found} for gate, found in checks.items()}


# --- inputs -----------------------------------------------------------------------------------


def phantom_family(run_dir: Path) -> dict[str, int]:
    """``PHANTOM_FAMILY`` matches over the run's window texts, and the windows
    holding at least one."""
    matches = windows = 0
    with (run_dir / "windows.jsonl").open(encoding="utf-8") as source:
        for line in source:
            found = len(PHANTOM_FAMILY.findall(json.loads(line)["text"]))
            matches += found
            windows += found > 0
    return {"matches": matches, "windows": windows}


def _long_word(line: Any) -> bool:
    return line.kind == "words" and line.length_bin in LONG_BANDS


def pooled_delta(base_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Paired delta of the aligned reading recall of the 13+ word lines, over
    the report's bootstrap units in the report's order."""
    _, bench_set, base = _load_run(base_dir)
    candidate = _load_run(candidate_dir)[2]
    unit_of = _bootstrap_units(bench_set)[1]
    position = {unit: index for index, unit in enumerate(dict.fromkeys(unit_of[item_id] for item_id in base))}
    hits = np.zeros((2, len(position)), dtype=np.int64)
    totals = np.zeros(len(position), dtype=np.int64)
    for item_id, item in base.items():
        at = position[unit_of[item_id]]
        for run, scored in enumerate((item, candidate[item_id])):
            hits[run, at] += sum(line.aligned_reading for line in scored.lines if _long_word(line))
        totals[at] += sum(_long_word(line) for line in item.lines)
    return _paired_delta(hits[0], hits[1], totals)


def resolve_pairs(runs: Path, prefix: str, baseline: str, candidate: str,
                  explicit: Mapping[str, tuple[Path, Path]]) -> dict[str, tuple[Path, Path]]:
    """Every gate and reported set's (baseline, candidate) run directories."""
    return {
        set_id: explicit.get(set_id) or (runs / f"{prefix}-{set_id}" / baseline, runs / f"{prefix}-{set_id}" / candidate)
        for set_id in (*GATE_SETS, *REPORTED_SETS)
    }


def _missing(pair: tuple[Path, Path]) -> list[str]:
    return [str(run_dir / name) for run_dir in pair for name in _RUN_FILES if not (run_dir / name).is_file()]


def check_pair(set_id: str, pair: tuple[Path, Path], names: tuple[str, str]) -> None:
    """Refuse a pair whose runs are not on ``set_id`` or are not the runs
    ``names`` (baseline, candidate) in that order."""
    for side, run_dir, name in zip(("baseline", "candidate"), pair, names):
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if record["set"]["set_id"] != set_id:
            raise InputError(f"{set_id}: the {side} run {run_dir} is on set {record['set']['set_id']}")
        if record["candidate"]["name"] != name:
            raise InputError(f"{set_id}: the {side} run {run_dir} is {record['candidate']['name']}, not {name}")


def score_pair(set_id: str, base_dir: Path, candidate_dir: Path, out: Path) -> SetResult:
    """Score the pair with ``scripts.asr_bench.score`` into ``<out>\\<set_id>``
    and read its report back."""
    report_dir = out / set_id
    with contextlib.redirect_stdout(io.StringIO()):
        score.main(["--run", str(base_dir), "--run", str(candidate_dir), "--out", str(report_dir)])
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    (entry,) = report["sets"]
    baseline = entry["baseline"]
    (candidate,) = (label for label in entry["metrics"] if label != baseline)
    return SetResult(
        set_id=set_id,
        entry=entry,
        baseline=baseline,
        candidate=candidate,
        pooled=pooled_delta(base_dir, candidate_dir),
        phantom={baseline: phantom_family(base_dir), candidate: phantom_family(candidate_dir)},
        runs=report["runs"],
        report_dir=report_dir,
    )


# --- output -----------------------------------------------------------------------------------


def _digits(*values: Any) -> int:
    rates = [value.get("rate") if isinstance(value, dict) else value for value in values]
    return 2 if any(isinstance(rate, float) and abs(rate) >= 10 for rate in rates) else 4


def _fmt(value: float | None, digits: int, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _fmt_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, dict) and "matches" in value:
        return f"{value['matches']:,} in {value['windows']:,} windows"
    if isinstance(value, dict):
        return f"{_fmt(value['rate'], digits)} ({value['hits']:,}/{value['total']:,})"
    if isinstance(value, int):
        return f"{value:,}"
    return _fmt(value, digits)


def _fmt_delta(delta: Mapping[str, Any] | None, digits: int = 4) -> str:
    if delta is None:
        return ""
    return (f"{_fmt(delta['delta'], digits, True)} "
            f"[{_fmt(delta['lower'], digits, True)}, {_fmt(delta['upper'], digits, True)}]")


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [_row(header), "|" + "---|" * len(header), *(_row(row) for row in rows), ""]


def _compare_row(name: str, base: Any, cand: Any, delta: Mapping[str, Any] | None) -> list[str]:
    digits = _digits(base, cand)
    return [name, _fmt_value(base, digits), _fmt_value(cand, digits), _fmt_delta(delta, digits)]


def _set_section(result: SetResult, role: str, record: Mapping[str, Any]) -> list[str]:
    entry = result.entry
    labels = (result.baseline, result.candidate)
    out = [
        f"## {result.set_id} ({role})",
        "",
        f"Set `{entry['set_id']}` ({entry['kind']}), {entry['item_count']:,} items, {entry['line_count']:,} lines; "
        f"bootstrap over {entry['bootstrap']['units']:,} {entry['bootstrap']['unit']}s. "
        f"Report: `{record['report']}`.",
        "",
    ]
    out += _table(["run", "label", "directory", "GPU", "windows"], [
        [side, run["label"], f"`{run['run_dir']}`", run["gpu"], f"{run['window_count']:,}"]
        for side, run in zip(("baseline", "candidate"), result.runs)
    ])
    rows = [
        ["all", "all", None],
        [WORDS, WORDS, None],
        *([f"{WORDS} {band}", WORDS, band] for band in (*SHORT_BANDS, *LONG_BANDS)),
    ]
    recall = [
        _compare_row(name, *(_value(result.cell(label, group, band), RECALL) for label in labels),
                     result.delta(group, RECALL, band))
        for name, group, band in rows
    ]
    recall.append(_compare_row(f"{WORDS} 13+ pooled", *(result.pooled_value(label) for label in labels),
                               result.pooled))
    recall += [
        _compare_row(group, *(_value(result.cell(label, group), RECALL) for label in labels),
                     result.delta(group, RECALL))
        for group in ("kind=intj", "kind=nonlex")
    ]
    out += ["Aligned recall (reading):", ""]
    out += _table(["lines", result.baseline, result.candidate, "delta [95%]"], recall)
    metrics = [
        _compare_row(name, *(_value(result.cell(label, "all"), metric) for label in labels),
                     result.delta("all", metric))
        for name, metric in (("CER (text)", "cer"), ("CER (reading)", "cer_reading"),
                             ("key-word recall", "keyword_recall"))
    ]
    if entry["kind"] == "silver":
        metrics.append(_compare_row("uncued windows per hour",
                                    *(result.cell(label, "all")["uncued_per_h"] for label in labels),
                                    result.delta("all", "uncued_per_h")))
    else:
        metrics.append(_compare_row("hallucinated windows",
                                    *(result.cell(label, "all")["hallucinations"] for label in labels), None))
    metrics.append(_compare_row("phantom family", *(result.phantom[label] for label in labels), None))
    out += _table(["metric", result.baseline, result.candidate, "delta [95%]"], metrics)
    keywords = result.cell(result.baseline, "all")["keywords"]
    if keywords:
        out += ["Key-word recall per lexicon entry:", ""]
        out += _table(["key word", "occurrences", result.baseline, result.candidate, "delta [95%]"], [
            [key, f"{counts['occurrences']:,}",
             *_compare_row("", *(_keyword(result.cell(label, "all"), key) for label in labels),
                           result.delta("all", "keywords")[key]["keyword_recall"])[1:]]
            for key, counts in keywords.items()
        ])
    return out


def _keyword(cell: Mapping[str, Any], key: str) -> dict[str, Any]:
    entry = cell["keywords"].get(key, {"recall": None, "recalled": 0, "occurrences": 0})
    return {"rate": entry["recall"], "hits": entry["recalled"], "total": entry["occurrences"]}


def gate_table(record: Mapping[str, Any]) -> list[str]:
    """The gate table of ``acceptance.md``: one row per check."""
    rows = []
    for gate, found in record["gates"].items():
        for check in found["checks"]:
            digits = _digits(check["baseline"], check["candidate"])
            rows.append([gate, "PASS" if check["pass"] else "FAIL", check["set"], check["check"],
                         _fmt_value(check["baseline"], digits), _fmt_value(check["candidate"], digits),
                         _fmt_delta(check["delta"], digits), check["rule"]])
    return ["## Gates", "", *_table(["gate", "result", "set", "check", "baseline", "candidate", "delta [95%]",
                                     "rule"], rows)]


def render_markdown(record: Mapping[str, Any], results: Mapping[str, SetResult]) -> str:
    failed = [gate for gate, found in record["gates"].items() if not found["pass"]]
    out = [
        f"# Fine-tuning acceptance: {record['candidate']} against {record['baseline']}",
        "",
        f"Verdict: **{record['verdict']}**" + (f" ({', '.join(failed)} fail)" if failed else ""),
        "",
        "Each pair is scored by `scripts.asr_bench.score` (its report beside this file, one folder per set). "
        "Deltas are candidate minus baseline, paired bootstrap 95% interval; recall is aligned recall, reading "
        "form. The pooled 13+ delta resamples the word lines of the bands 13-20 and 21+ together over the "
        "report's bootstrap units.",
        "",
    ]
    out += gate_table(record)
    for set_id, result in results.items():
        role = "gates A1-A3, A5, A6" if set_id == "silver-v5" else "gate A4" if set_id in GATE_SETS else "reported"
        out += _set_section(result, role, record["sets"][set_id])
    if record["skipped"]:
        out += ["## Reported sets not found", ""]
        out += [f"- {set_id}: missing {', '.join(f'`{path}`' for path in missing)}"
                for set_id, missing in record["skipped"].items()]
        out.append("")
    out += ["## Phantom family", "", f"`{PHANTOM_FAMILY.pattern}`, non-overlapping matches over window texts.", ""]
    out += _table(["set", "run", "matches", "windows"], [
        [set_id, label, f"{counts['matches']:,}", f"{counts['windows']:,}"]
        for set_id, per_run in record["phantom_family"].items()
        for label, counts in per_run.items()
    ])
    return "\n".join(out)


def _record(args: argparse.Namespace, results: Mapping[str, SetResult], skipped: Mapping[str, list[str]]
            ) -> dict[str, Any]:
    gates = evaluate(results)
    return {
        "schema": SCHEMA,
        "verdict": "PASS" if all(found["pass"] for found in gates.values()) else "FAIL",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "gates": gates,
        "sets": {
            set_id: {
                "role": "gate" if set_id in GATE_SETS else "reported",
                "set_id": result.entry["set_id"],
                "kind": result.entry["kind"],
                "report": str(result.report_dir / "report.json") if result.report_dir else None,
                "baseline": result.baseline,
                "candidate": result.candidate,
                "runs": [{key: run[key] for key in ("label", "run_dir", "gpu", "window_count")} for run in result.runs],
                "pooled_13_plus": {
                    "baseline": result.pooled_value(result.baseline),
                    "candidate": result.pooled_value(result.candidate),
                    "delta": dict(result.pooled),
                },
            }
            for set_id, result in results.items()
        },
        "skipped": dict(skipped),
        "phantom_family": {set_id: {label: dict(counts) for label, counts in result.phantom.items()}
                           for set_id, result in results.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.acceptance",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, help="the fine-tuned model's run name (e.g. ftp-v2a)")
    parser.add_argument("--baseline", default="ftp-base", help="the base model's run name (default ftp-base)")
    parser.add_argument("--runs", type=Path, default=bench_root() / "runs",
                        help="run root holding <prefix>-<set> folders (default <bench root>\\runs)")
    parser.add_argument("--prefix", default=PILOT, help=f"run folder prefix (default {PILOT})")
    parser.add_argument("--pair", nargs=3, action="append", default=[],
                        metavar=("SET", "BASELINE_DIR", "CANDIDATE_DIR"),
                        help="the two run directories of one set, instead of <runs>\\<prefix>-<set>\\<name>")
    parser.add_argument("--out", type=Path, required=True, help="folder for acceptance.json, acceptance.md, reports")
    args = parser.parse_args(argv)
    if args.baseline == args.candidate:
        parser.error("--baseline and --candidate must differ")
    explicit: dict[str, tuple[Path, Path]] = {}
    for set_id, base_dir, candidate_dir in args.pair:
        if set_id not in GATE_SETS and set_id not in REPORTED_SETS:
            parser.error(f"--pair: unknown set {set_id!r} (one of {', '.join([*GATE_SETS, *REPORTED_SETS])})")
        if set_id in explicit:
            parser.error(f"--pair: {set_id} given twice")
        explicit[set_id] = (Path(base_dir), Path(candidate_dir))
    pairs = resolve_pairs(args.runs, args.prefix, args.baseline, args.candidate, explicit)
    try:
        return _run(args, pairs)
    except InputError as error:
        print(f"acceptance: input does not fit: {error}", file=sys.stderr)
    except Exception:
        traceback.print_exc()
        print("acceptance: error, no verdict (exit 2)", file=sys.stderr)
    return 2


def _run(args: argparse.Namespace, pairs: Mapping[str, tuple[Path, Path]]) -> int:
    missing = [path for set_id in GATE_SETS for path in _missing(pairs[set_id])]
    if missing:
        print("acceptance: missing input:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    skipped = {set_id: _missing(pairs[set_id]) for set_id in REPORTED_SETS if _missing(pairs[set_id])}
    scored = [set_id for set_id in (*GATE_SETS, *REPORTED_SETS) if set_id not in skipped]
    for set_id in scored:
        check_pair(set_id, pairs[set_id], (args.baseline, args.candidate))
    args.out.mkdir(parents=True, exist_ok=True)
    results = {set_id: score_pair(set_id, *pairs[set_id], args.out) for set_id in scored}
    record = _record(args, results, skipped)
    (args.out / "acceptance.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8")
    markdown = render_markdown(record, results)
    (args.out / "acceptance.md").write_text(markdown + "\n", encoding="utf-8")
    print("\n".join(gate_table(record)), end="")
    print(f"verdict: {record['verdict']}; acceptance.json and acceptance.md in {args.out}")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
