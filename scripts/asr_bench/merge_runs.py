"""Combine recognizers offline: insert into a base run's window texts the
meaningful interjections other runs over the same windows heard, and bound
what such a combination can gain.

Usage::

    python -m scripts.asr_bench.merge_runs build --base <run dir>
        --donor <run dir> [--donor <run dir> ...] [--votes K] [--fill-empty]
        [--max-base-chars N] --name <candidate> --out <new run dir>
    python -m scripts.asr_bench.merge_runs oracle --base <run dir>
        --donor <run dir> [--donor <run dir> ...] --out <report dir>
    python -m scripts.asr_bench.merge_runs insertions --run <merged run dir>
        [--run <merged run dir> ...]

Every run must hold the same windows: the same set and, line by line of
``windows.jsonl``, the same item, window index, start and end. Any other
pairing is refused, since text is only exchanged window by window.

``build`` writes a merged run. Per window, A is the base run's text and B
each donor's. B is cut into pieces (``lexicon.pieces``): every lexicon
interjection occurrence (a whole utterance chunk with its trailing
punctuation, or the first 感動詞 token of a longer chunk) is an insertable
piece, everything else an anchor. A and the concatenated pieces are aligned
once on UniDic katakana readings (``island.find_insertions``, the island
recovery merge): an insertable piece of which fewer than half the reading
characters align as exact matches is absent from A, and the donor proposes
it at the offset in A where its first character aligns. Proposals of the
same entry, ッ dropped (key-word recall drops it too), at the same offset
agree; the entry is inserted as many times as at least ``--votes`` donors
propose it there, as the first of those donors (in ``--donor`` order)
wrote it. With one donor and one vote this inserts every donor
interjection A lacks. Insertion only touches windows whose base text
normalizes to something, so it never turns a window the base left empty
into text and cannot change the uncued window count. With ``--fill-empty``
a window whose base text normalizes to nothing takes the whole text of the
first donor whose text does not, provided at least ``--votes`` donors'
texts do not. With ``--max-base-chars N`` the donors act only on windows
whose base text normalizes to at most N characters (and to at least one,
unless ``--fill-empty``); they need not decode any other window.

The merged run is a run directory the scorer reads. ``run.json`` is the
base's with the candidate renamed and a ``merge`` section naming the source
runs (directory, candidate, ASR identity, digest of ``windows.jsonl``), the
rule, its counts and the donors' cost: the windows they decode (count and
share), the audio seconds of those windows (and their share of all window
audio), and per donor its decode seconds of those windows (and their share
of the donor run's ASR seconds). Timing adds to the base's each donor's
load seconds and decode seconds of those windows (every model decodes in
turn; VAD once). Each window keeps the base's fields and adds ``text_a``
(the base text), ``asr_a_s`` (its decode seconds) and ``merge`` (whether
the donors decoded it, the insertions with their offset in ``text_a``, text
and agreeing donors, and the donor a filled window came from); ``asr_s``
adds the donors' decode seconds of the window when they decode it. A
decoded donor window must carry ``asr_s``.

``oracle`` scores every run with the scorer and reports, per key-word
occurrence and per ``kind=intj`` line, the share each run catches under the
aligned reading rule, and the share caught by the base or one donor and by
the base or any donor: what a perfect choice among these recognizers could
catch over the same windows. The same OR over the scorer's chance verdicts
of the ``intj`` lines shows how much of that a combination would catch from
unrelated speech. It writes ``oracle.json`` and ``oracle.md``.

``insertions`` checks merged runs' insertions against the set's reference
lines: per run, how many inserted texts lie in a window that overlaps,
widened by the scorer's 0.5 s collar, a line holding a lexicon interjection
(near), one holding the same entry with ッ dropped, or no line at all.
Subtitles leave out some interjections that are spoken and an insertion
next to a subtitle interjection may still be wrong, so this counts how
often the subtitles back an insertion, not how often it is right.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench import island, lexicon
from scripts.asr_bench.run import _git_identity
from scripts.asr_bench.score import COLLAR_S, _load_run
from scripts.asr_bench.sets import load_set
from scripts.e0_evidence.normalize_ja import normalize_minimal
from scripts.e0_evidence.wire import canonical_sha256

MERGE_SCHEMA_VERSION = "asr-bench-merge-1.1"
ORACLE_SCHEMA_VERSION = "asr-bench-merge-oracle-1.0"
_SPAN_KEYS = ("item_id", "window_index", "start_s", "end_s")


@dataclass(frozen=True)
class _Source:
    run_dir: Path
    record: dict[str, Any]
    windows: list[dict[str, Any]]
    windows_sha256: str

    @property
    def label(self) -> str:
        return self.record["candidate"]["name"]

    def describe(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir.resolve()),
            "candidate": self.record["candidate"],
            "asr_model": (self.record.get("asr") or {}).get("model"),
            "created_at_utc": self.record.get("created_at_utc"),
            "git": self.record.get("git"),
            "windows_sha256": self.windows_sha256,
        }


def _load(run_dir: Path) -> _Source:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data = (run_dir / "windows.jsonl").read_bytes()
    windows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return _Source(run_dir, record, windows, hashlib.sha256(data).hexdigest())


def _check_same_windows(base: _Source, others: Sequence[_Source]) -> None:
    """Refuse runs whose windows differ from the base's in set or span."""
    for other in others:
        if other.record["set"]["set_id"] != base.record["set"]["set_id"]:
            raise ValueError(
                f"{other.run_dir} is on set {other.record['set']['set_id']}, "
                f"{base.run_dir} on {base.record['set']['set_id']}"
            )
        if len(other.windows) != len(base.windows):
            raise ValueError(
                f"{other.run_dir} has {len(other.windows)} windows, {base.run_dir} {len(base.windows)}"
            )
        for number, (mine, theirs) in enumerate(zip(base.windows, other.windows), 1):
            if any(mine[key] != theirs[key] for key in _SPAN_KEYS):
                span = {key: theirs[key] for key in _SPAN_KEYS}
                expected = {key: mine[key] for key in _SPAN_KEYS}
                raise ValueError(
                    f"{other.run_dir}/windows.jsonl:{number}: window {span} differs from "
                    f"the base window {expected}"
                )


def _entry(key: str) -> str:
    """The lexicon entry with ッ dropped, the form proposals agree on."""
    return key.replace("ッ", "")


@dataclass(frozen=True)
class WindowMerge:
    text: str
    # Per inserted entry: its offset in the base text, the texts inserted
    # there and the indexes of the donors that agreed.
    inserted: tuple[tuple[int, tuple[str, ...], tuple[int, ...]], ...]
    # The donor whose text filled an empty base window, else None.
    filled_from: int | None


def merge_window(
    base_text: str, donor_texts: Sequence[str], votes: int, fill_empty: bool
) -> WindowMerge:
    """One window's merged text (see the module docstring)."""
    if not normalize_minimal(base_text):
        heard = [index for index, text in enumerate(donor_texts) if normalize_minimal(text)]
        if fill_empty and len(heard) >= votes:
            return WindowMerge(donor_texts[heard[0]], (), heard[0])
        return WindowMerge(base_text, (), None)
    # (offset in the base text, entry) -> donor -> texts it proposes there.
    groups: dict[tuple[int, str], dict[int, list[str]]] = {}
    for donor, text in enumerate(donor_texts):
        cut = lexicon.pieces(text)
        _, offsets = island.find_insertions(
            base_text, [piece for piece, _ in cut], [key is not None for _, key in cut]
        )
        for (piece, key), offset in zip(cut, offsets):
            if offset is not None and key is not None:
                proposed = groups.setdefault((offset, _entry(key)), {})
                proposed.setdefault(donor, []).append(piece.strip())
    insertions: list[tuple[int, str]] = []
    inserted: list[tuple[int, tuple[str, ...], tuple[int, ...]]] = []
    for (offset, _), proposed in groups.items():
        counts = sorted((len(texts) for texts in proposed.values()), reverse=True)
        if len(counts) < votes:
            continue
        copies = counts[votes - 1]
        agreeing = tuple(sorted(donor for donor, texts in proposed.items() if len(texts) >= copies))
        texts = tuple(proposed[agreeing[0]][:copies])
        insertions.extend((offset, text) for text in texts)
        inserted.append((offset, texts, agreeing))
    return WindowMerge(island.insert(base_text, insertions), tuple(inserted), None)


def donors_decode(base_text: str, max_base_chars: int | None, fill_empty: bool) -> bool:
    """Whether the donors decode a window: only one whose base text
    normalizes to 1 to ``max_base_chars`` characters (any number without a
    limit), or to nothing with ``fill_empty``, can change."""
    length = len(normalize_minimal(base_text))
    if not length:
        return fill_empty
    return max_base_chars is None or length <= max_base_chars


def build(
    base_dir: Path,
    donor_dirs: Sequence[Path],
    votes: int,
    fill_empty: bool,
    max_base_chars: int | None,
    name: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Write the merged run to ``out_dir`` (which must not exist) and return its ``run.json``."""
    if not donor_dirs:
        raise ValueError("give at least one donor run")
    if not 1 <= votes <= len(donor_dirs):
        raise ValueError(f"--votes must lie between 1 and the {len(donor_dirs)} donors")
    if max_base_chars is not None and max_base_chars < 1:
        raise ValueError("--max-base-chars must be at least 1")
    base = _load(base_dir)
    donors = [_load(path) for path in donor_dirs]
    _check_same_windows(base, donors)
    labels = [donor.label for donor in donors]
    out_dir.mkdir(parents=True, exist_ok=False)

    changed = insertions = filled = decoded_windows = 0
    audio_s = decoded_audio_s = 0.0
    donor_seconds = [0.0] * len(donors)
    with (out_dir / "windows.jsonl").open("w", encoding="utf-8", newline="\n") as sink:
        for position, window in enumerate(base.windows):
            others = [donor.windows[position] for donor in donors]
            span_s = window["end_s"] - window["start_s"]
            audio_s += span_s
            decoded = donors_decode(window["text"], max_base_chars, fill_empty)
            if decoded:
                for donor, other in zip(donors, others):
                    if other.get("asr_s") is None:
                        raise ValueError(
                            f"{donor.run_dir}/windows.jsonl:{position + 1}: no asr_s, "
                            "which the cost of the merge needs"
                        )
                merged = merge_window(window["text"], [other["text"] for other in others], votes, fill_empty)
                decoded_windows += 1
                decoded_audio_s += span_s
                for index, other in enumerate(others):
                    donor_seconds[index] += other["asr_s"]
            else:
                merged = WindowMerge(window["text"], (), None)
            changed += merged.text != window["text"]
            insertions += sum(len(texts) for _, texts, _ in merged.inserted)
            filled += merged.filled_from is not None
            base_s = window.get("asr_s")
            donors_s = sum(other["asr_s"] for other in others) if decoded else 0.0
            record = {
                **window,
                "text": merged.text,
                "asr_s": None if base_s is None else base_s + donors_s,
                "text_a": window["text"],
                "asr_a_s": base_s,
                "merge": {
                    "donors_decoded": decoded,
                    "inserted": [
                        {"offset": offset, "texts": list(texts), "donors": [labels[d] for d in agreeing]}
                        for offset, texts, agreeing in merged.inserted
                    ],
                    "filled_from": None if merged.filled_from is None else labels[merged.filled_from],
                },
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    record = copy.deepcopy(base.record)
    record["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["candidate"] = {**record["candidate"], "name": name}
    timing = record["timing"]
    donor_load_s = sum(donor.record["timing"]["load_seconds"] for donor in donors)
    timing["load_seconds"] += donor_load_s
    timing["asr_seconds"] += sum(donor_seconds)
    timing["wall_seconds"] += donor_load_s + sum(donor_seconds)
    timing["rtf"] = (timing["vad_seconds"] + timing["asr_seconds"]) / record["set"]["audio_seconds"]
    # The merged windows' asr_s add the donors' decodes: comparable only when
    # every run decoded in one process. The base's wall clock leaves the donors out.
    record["timing_valid"] = all(run.record.get("timing_valid", True) for run in (base, *donors))
    record.pop("throughput", None)
    record["merge"] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "rule": "insert donor lexicon interjections absent from the base text",
        "base": base.describe(),
        "donors": [donor.describe() for donor in donors],
        "votes": votes,
        "fill_empty": fill_empty,
        "max_base_chars": max_base_chars,
        "absent_share": island.ABSENT_SHARE,
        "agreement": "same lexicon entry, ッ dropped, at the same offset in the base text",
        "lexicon_entries_sha256": canonical_sha256(lexicon.INTERJECTIONS),
        "timing": (
            "the base's load, ASR and wall seconds plus each donor's load seconds and its "
            "decode seconds of the windows the donors decode; VAD once"
        ),
        "git": _git_identity(),
        "windows": len(base.windows),
        "windows_changed": changed,
        "insertions": insertions,
        "windows_filled": filled,
        "donor_cost": {
            "windows": decoded_windows,
            "window_share": decoded_windows / len(base.windows) if base.windows else None,
            "audio_s": decoded_audio_s,
            "audio_share": decoded_audio_s / audio_s if audio_s else None,
            "asr_seconds": [
                {
                    "candidate": label,
                    "seconds": seconds,
                    "share": seconds / donor.record["timing"]["asr_seconds"]
                    if donor.record["timing"]["asr_seconds"] else None,
                }
                for label, donor, seconds in zip(labels, donors, donor_seconds)
            ],
        },
    }
    (out_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return record


def _flatten(items: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Per key-word occurrence its entry and aligned verdict, and per
    ``kind=intj`` line its aligned reading and chance verdicts, in set order."""
    keys: list[str] = []
    hits: list[bool] = []
    intj: list[bool] = []
    chance: list[bool] = []
    for item in items.values():
        for line in item.lines:
            for key, hit in line.keywords:
                keys.append(key)
                hits.append(hit)
            if line.kind == "intj":
                intj.append(line.aligned_reading)
                chance.append(line.chance_aligned_reading)
    return keys, np.array(hits, dtype=bool), np.array(intj, dtype=bool), np.array(chance, dtype=bool)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def oracle(base_dir: Path, donor_dirs: Sequence[Path]) -> dict[str, Any]:
    if not donor_dirs:
        raise ValueError("give at least one donor run")
    sources = [_load(path) for path in (base_dir, *donor_dirs)]
    _check_same_windows(sources[0], sources[1:])
    keys: list[str] = []
    hits: list[np.ndarray] = []
    intj: list[np.ndarray] = []
    chance: list[np.ndarray] = []
    for source in sources:
        run_keys, run_hits, run_intj, run_chance = _flatten(_load_run(source.run_dir)[2])
        if keys and run_keys != keys:
            raise ValueError(f"{source.run_dir}: key-word occurrences differ from the base run's")
        keys = run_keys
        hits.append(run_hits)
        intj.append(run_intj)
        chance.append(run_chance)
    labels = [source.label for source in sources]
    combos: list[tuple[str, list[int]]] = [(label, [index]) for index, label in enumerate(labels)]
    combos += [(f"{labels[0]} or {labels[d]}", [0, d]) for d in range(1, len(labels))]
    if len(labels) > 2:
        combos.append((f"{labels[0]} or any donor", list(range(len(labels)))))
    entries = np.array(keys)
    rows = []
    for label, members in combos:
        caught = np.logical_or.reduce([hits[m] for m in members])
        lines = np.logical_or.reduce([intj[m] for m in members])
        lines_chance = np.logical_or.reduce([chance[m] for m in members])
        rows.append({
            "label": label,
            "runs": [labels[m] for m in members],
            "keyword_recalled": int(caught.sum()),
            "keyword_recall": _ratio(int(caught.sum()), len(keys)),
            "intj_caught": int(lines.sum()),
            "intj_recall": _ratio(int(lines.sum()), lines.size),
            "intj_chance_caught": int(lines_chance.sum()),
            "intj_chance_recall": _ratio(int(lines_chance.sum()), lines_chance.size),
            "keywords": {
                key: {
                    "occurrences": int((entries == key).sum()),
                    "recalled": int(caught[entries == key].sum()),
                    "recall": _ratio(int(caught[entries == key].sum()), int((entries == key).sum())),
                }
                for key in lexicon.INTERJECTIONS
                if key in keys
            },
        })
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "set_id": sources[0].record["set"]["set_id"],
        "rule": "aligned reading rule of the scorer; a combination catches what any member catches",
        "git": _git_identity(),
        "runs": [source.describe() for source in sources],
        "keyword_occurrences": len(keys),
        "intj_lines": int(intj[0].size),
        "rows": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    out = [
        f"# Merge oracle on `{report['set_id']}`",
        "",
        f"Created {report['created_at_utc']}. {report['rule'].capitalize()}. The intj chance "
        "column ORs each member's chance verdicts (the scorer's chance control: the line scored "
        "against another item's hypothesis), what a combination catches from unrelated speech.",
        "",
        "| runs | key words | intj lines | intj chance |",
        "|---|---|---|---|",
    ]
    for row in rows:
        out.append(
            f"| {row['label']} | {row['keyword_recall']:.3f} "
            f"({row['keyword_recalled']}/{report['keyword_occurrences']}) | "
            f"{row['intj_recall']:.3f} ({row['intj_caught']}/{report['intj_lines']}) | "
            f"{row['intj_chance_recall']:.3f} |"
        )
    out += ["", "Key-word recall per lexicon entry:", ""]
    out.append("| word | count | " + " | ".join(row["label"] for row in rows) + " |")
    out.append("|---|---|" + "---|" * len(rows))
    for key, (written, _) in lexicon.INTERJECTIONS.items():
        first = rows[0]["keywords"].get(key)
        if first is None:
            continue
        cells = [f"{row['keywords'][key]['recall']:.3f}" for row in rows]
        out.append(f"| {written} | {first['occurrences']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _entries(text: str) -> set[str]:
    return {_entry(key) for _, _, key in lexicon.analyze(text).interjections}


def check_insertions(run_dir: Path) -> dict[str, Any]:
    """Count a merged run's insertions near reference interjections (see the module docstring)."""
    source = _load(run_dir)
    if "merge" not in source.record:
        raise ValueError(f"{run_dir} is not a merged run")
    lines = {item.item_id: item.lines for item in load_set(Path(source.record["set"]["path"])).items}
    counts = {"label": source.label, "insertions": 0, "windows": 0, "near": 0, "same_entry": 0, "no_line": 0}
    for window in source.windows:
        texts = [text for group in window["merge"]["inserted"] for text in group["texts"]]
        if not texts:
            continue
        over = [
            line for line in lines[window["item_id"]]
            if line.start_s < window["end_s"] + COLLAR_S and window["start_s"] - COLLAR_S < line.end_s
        ]
        reference = set().union(*(_entries(line.text) for line in over))
        counts["windows"] += 1
        for text in texts:
            counts["insertions"] += 1
            counts["near"] += bool(reference)
            counts["same_entry"] += bool(_entries(text) & reference)
            counts["no_line"] += not over
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="write a merged run")
    bound = commands.add_parser("oracle", help="bound what a combination can catch")
    for command in (make, bound):
        command.add_argument("--base", required=True, type=Path, help="base run directory")
        command.add_argument(
            "--donor", required=True, type=Path, action="append", help="donor run directory"
        )
    make.add_argument("--votes", type=int, default=1, help="donors that must agree (default 1)")
    make.add_argument("--fill-empty", action="store_true", help="fill empty base windows")
    make.add_argument(
        "--max-base-chars", type=int, help="donors act only on windows whose base text has at most N characters"
    )
    make.add_argument("--name", required=True, help="candidate name of the merged run")
    make.add_argument("--out", required=True, type=Path, help="new run directory")
    bound.add_argument("--out", required=True, type=Path, help="report directory")
    check = commands.add_parser("insertions", help="check merged runs' insertions")
    check.add_argument("--run", required=True, type=Path, action="append", help="merged run directory")
    args = parser.parse_args(argv)
    if args.command == "build":
        record = build(
            args.base, args.donor, args.votes, args.fill_empty, args.max_base_chars, args.name, args.out
        )
        merge = record["merge"]
        cost = merge["donor_cost"]
        print(
            f"{args.name}: {merge['windows']} windows, {merge['windows_changed']} changed, "
            f"{merge['insertions']} insertions, {merge['windows_filled']} filled; donors decode "
            f"{cost['windows']} windows ({cost['window_share']:.3f}), {cost['audio_share']:.3f} of "
            f"the window audio -> {args.out}"
        )
        return 0
    if args.command == "insertions":
        print("| run | insertions | windows | near | same entry | no line |")
        print("|---|---|---|---|---|---|")
        for run_dir in args.run:
            counts = check_insertions(run_dir)
            total = counts["insertions"]
            share = {
                key: f"{counts[key]} ({counts[key] / total:.2f})" if total else "0"
                for key in ("near", "same_entry", "no_line")
            }
            print(
                f"| {counts['label']} | {total} | {counts['windows']} | {share['near']} | "
                f"{share['same_entry']} | {share['no_line']} |"
            )
        return 0
    report = oracle(args.base, args.donor)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oracle.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (args.out / "oracle.md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
