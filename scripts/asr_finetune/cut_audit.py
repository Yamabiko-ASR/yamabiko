"""Audit, and filter, the round-3 cut windows against the base model (pod task).

    python -m scripts.asr_finetune.cut_audit --windows <dir> --out <dir> [--limit 200] [--exit-zero]
    python -m scripts.asr_finetune.cut_audit --windows <dir> --out <dir> --all --filter-index [--workers 3] [--exit-zero]
        [--device D] [--compute-type T]

Reads ``audio.i16`` and ``index.jsonl`` that ``scripts.asr_finetune.windows``
wrote to ``--windows``. The windows with a cut (``index.jsonl`` ``cut``) are
decoded as they are, their 0.5 s pads included, by the base model's recognizer
without its prompt (``teacher.teacher_config``, ``build_asr``) with its
production ``transcribe`` arguments (``teacher.production_segments`` without
a second pad: beam 5, no prompt, ``condition_on_previous_text=False``, the
junk gate): the base model, before any training.

A cut side *matches* when the reading form (``scripts.asr_bench.lexicon.reading``:
katakana of the minimal-normalized text, long-vowel marks dropped) of the
label's edge equals the base text's: the first 3 characters of the label's
first segment and of the base text for a head cut, the last 3 of its last
segment and of the base text for a tail cut. An empty base text matches
nothing.

Default mode, a measurement: up to ``--limit`` cut windows drawn with
``random.Random(pilot.SEED)``, one process.

``--all --filter-index``, a filter: every cut window (train and dev) is
decoded in ``--workers`` processes (each with its own model; default 3 on a
24 GB GPU) and a cut window is kept only when every cut side of it matches.
A cut label the base does not confirm at the cut would teach the model to
drop an audible fragment. ``index.jsonl`` is first saved as
``index.unfiltered.jsonl`` (an existing one is the original and is read
instead, so a rerun filters the original again), then rewritten without the
failing cut windows; uncut windows and the audio are untouched. Both files
are written through a temporary file and ``os.replace``. A crash drops every
cut window instead (``drop_all_cuts``: training never sees an unverified cut
label), and ``cut-audit.json`` says so in ``filter``.

``<out>\\cut-audit.json`` (``ft-pilot-cut-audit-1``): the cut windows and
those decoded; per side the windows compared, the matches, the match rate
and the empty base texts; up to 20 mismatching examples (``w``, side, the
label segment and the base text, both reading forms). The filter adds
``filter`` (``applied``, or why not), the windows kept and dropped per
source and cut kind (``head``, ``tail``, ``both``), the index lines before
and after, and the ``w`` of every dropped window. The console shows counts
and rates only.

Exit 0 whatever the rates. A crash exits 1, or 0 with ``--exit-zero`` (the
pod's case), after writing ``cut-audit.json`` with the error type.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from scripts.asr_finetune.pilot import SEED

SCHEMA = "ft-pilot-cut-audit-1"
COMPARED = 3  # reading characters compared at the cut
EXAMPLES = 20
_PROGRESS = 1000

LabelTexts = Callable[[dict[str, Any]], list[str]]


def segment_texts(tokenizer: Any, labels: Sequence[int]) -> list[str]:
    """The texts between the timestamp pairs of a window's label."""
    from scripts.asr_finetune.labels import TS0, TS_MAX, TS_STEP

    last = TS0 + round(TS_MAX / TS_STEP)
    texts: list[str] = []
    current: list[int] = []
    inside = False
    for token in labels:
        if TS0 <= token <= last:
            if inside:
                texts.append(tokenizer.decode(current, skip_special_tokens=False))
                current = []
            inside = not inside
        elif inside:
            current.append(token)
    return texts


def judge(entry: dict[str, Any], labels: list[str], base: str) -> list[dict[str, Any]]:
    """One result per cut side of a window: side, match, the label edge
    segment, and both reading forms."""
    from scripts.asr_bench.lexicon import reading

    base_reading = reading(base) if base else ""
    results = []
    for side in entry["cut"]:
        label = labels[0] if side == "head" else labels[-1]
        label_reading = reading(label)
        edge = (lambda text: text[:COMPARED]) if side == "head" else (lambda text: text[-COMPARED:])
        results.append({"side": side, "match": bool(base_reading) and edge(label_reading) == edge(base_reading),
                        "label": label, "label_reading": label_reading, "base_reading": base_reading})
    return results


def _source(entry: dict[str, Any]) -> str:
    """The source of a cut window: that of its cut utterance."""
    return entry["segments"][0 if "head" in entry["cut"] else -1][2]


def compare(entries: Sequence[dict[str, Any]], bases: Sequence[str], label_texts: LabelTexts,
            ) -> tuple[list[bool], dict[str, Any]]:
    """Whether each cut window matches on every cut side, and the record's
    per-side counts and examples."""
    sides = {side: Counter(compared=0, match=0, base_empty=0) for side in ("head", "tail")}
    examples: list[dict[str, Any]] = []
    keep: list[bool] = []
    for entry, base in zip(entries, bases):
        results = judge(entry, label_texts(entry), base)
        for result in results:
            sides[result["side"]].update(compared=1, match=result["match"], base_empty=not base)
            if not result["match"] and len(examples) < EXAMPLES:
                examples.append({"w": entry["w"], "side": result["side"], "label": result["label"], "base": base,
                                 "label_reading": result["label_reading"], "base_reading": result["base_reading"]})
        keep.append(all(result["match"] for result in results))
    record = {"sides": {side: {**counts, "rate": round(counts["match"] / counts["compared"], 4)
                               if counts["compared"] else None} for side, counts in sides.items()},
              "mismatch_examples": examples}
    return keep, record


CRASHED = "crashed, all cut windows dropped"


def _replace(path: Path, lines: Sequence[str]) -> None:
    """Write ``lines`` to ``path`` through a temporary file and ``os.replace``."""
    staged = path.with_name(path.name + ".writing")
    staged.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    os.replace(staged, path)


def _original(windows: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """The lines and rows of the unfiltered index, saved first as
    ``index.unfiltered.jsonl`` (an existing one is the original)."""
    index, original = windows / "index.jsonl", windows / "index.unfiltered.jsonl"
    lines = [line for line in (original if original.is_file() else index).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not original.is_file():
        _replace(original, lines)
    return lines, [json.loads(line) for line in lines]


def _keep(windows: Path, lines: Sequence[str], entries: Sequence[dict[str, Any]], dropped: set[int]) -> dict[str, int]:
    """Rewrite ``index.jsonl`` without the ``dropped`` windows."""
    _replace(windows / "index.jsonl", [line for line, entry in zip(lines, entries) if entry["w"] not in dropped])
    return {"before": len(entries), "after": len(entries) - len(dropped)}


def drop_all_cuts(windows: Path) -> dict[str, Any]:
    """Rewrite ``index.jsonl`` without any cut window: training never sees an
    unverified cut label."""
    lines, entries = _original(windows)
    dropped = {entry["w"] for entry in entries if entry.get("cut")}
    return {"filter": CRASHED, "cut_windows": len(dropped), "kept": 0, "dropped": len(dropped),
            "index_lines": _keep(windows, lines, entries, dropped)}


def filter_index(windows: Path, decode: Callable[[Sequence[dict[str, Any]]], list[str]],
                 label_texts: LabelTexts) -> dict[str, Any]:
    """Decode every cut window of ``windows``'s index (``decode``: base texts
    in order), then rewrite ``index.jsonl`` without the cut windows that do
    not match, the original kept as ``index.unfiltered.jsonl``. When the
    decode or the comparison fails, every cut window is dropped
    (``drop_all_cuts``) and the record holds the error type."""
    lines, entries = _original(windows)
    cut = [entry for entry in entries if entry.get("cut")]
    print(f"cut filter: {len(entries):,} windows, {len(cut):,} with a cut", flush=True)
    try:
        bases = decode(cut)
        if len(bases) != len(cut):
            raise RuntimeError(f"the decoder returned {len(bases)} of {len(cut)} texts")
        keep, record = compare(cut, bases, label_texts)
    except Exception as error:  # the type only: a message may hold a path
        traceback.print_exc()
        return {**drop_all_cuts(windows), "error": type(error).__name__}
    dropped = {entry["w"] for entry, kept in zip(cut, keep) if not kept}
    kinds: dict[str, dict[str, Counter]] = {}
    for entry, kept in zip(cut, keep):
        kind = "both" if len(entry["cut"]) == 2 else entry["cut"][0]
        kinds.setdefault(_source(entry), {}).setdefault(kind, Counter()).update(kept=kept, dropped=not kept)
    return {"filter": "applied", "cut_windows": len(cut), "decoded": len(cut), "kept": len(cut) - len(dropped),
            "dropped": len(dropped),
            "by_source": {source: {kind: dict(counts) for kind, counts in sorted(by_kind.items())}
                          for source, by_kind in sorted(kinds.items())},
            "index_lines": _keep(windows, lines, entries, dropped), **record, "dropped_w": sorted(dropped)}


# --- decoding (worker processes) ----------------------------------------------------------

_ASR: Any = None
_AUDIO: Any = None


def _init(config: dict[str, Any], audio_path: str) -> None:
    global _ASR, _AUDIO
    import numpy as np

    from yamabiko.asr.asr_client import build_asr

    _ASR = build_asr(config)
    _AUDIO = np.memmap(audio_path, dtype="<i2", mode="r")


def _decode(span: tuple[int, int]) -> str:
    import numpy as np

    from scripts.asr_finetune.teacher import production_segments

    offset, samples = span
    audio = np.asarray(_AUDIO[offset:offset + samples], dtype=np.float32) / 32768
    return "".join(segment.text for segment in production_segments(_ASR, audio, pad=False)).strip()


def base_decoder(windows: Path, config: dict[str, Any], workers: int) -> Callable[[Sequence[dict[str, Any]]], list[str]]:
    """``decode`` for ``filter_index`` and the audit: the base text of each
    window, in order, from ``workers`` processes (one in this process)."""

    def decode(entries: Sequence[dict[str, Any]]) -> list[str]:
        spans = [(entry["offset"], entry["samples"]) for entry in entries]
        started, texts = time.monotonic(), []
        if workers == 1:
            _init(config, str(windows / "audio.i16"))
            return [_decode(span) for span in spans]
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init,
                                 initargs=(config, str(windows / "audio.i16"))) as pool:
            for text in pool.map(_decode, spans, chunksize=8):
                texts.append(text)
                if len(texts) % _PROGRESS == 0:
                    print(f"cut audit: {len(texts):,} of {len(spans):,} decoded "
                          f"({time.monotonic() - started:.0f} s)", flush=True)
        return texts

    return decode


def _label_texts() -> LabelTexts:
    from scripts.asr_finetune.labels import load_tokenizer

    tokenizer = load_tokenizer()
    return lambda entry: segment_texts(tokenizer, entry["labels"])


def sample_audit(windows: Path, limit: int | None, decode: Callable[[Sequence[dict[str, Any]]], list[str]],
                 label_texts: LabelTexts) -> dict[str, Any]:
    """The measurement: up to ``limit`` cut windows (None: all) drawn with ``SEED``."""
    with (windows / "index.jsonl").open(encoding="utf-8") as handle:
        cut = [entry for entry in map(json.loads, filter(str.strip, handle)) if entry.get("cut")]
    count = len(cut) if limit is None else min(limit, len(cut))
    drawn = sorted(random.Random(SEED).sample(cut, count), key=lambda entry: entry["w"])
    print(f"cut audit: {len(cut):,} cut windows, {len(drawn):,} drawn", flush=True)
    _, record = compare(drawn, decode(drawn) if drawn else [], label_texts)
    return {"cut_windows": len(cut), "decoded": len(drawn), **record}


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.cut_audit", description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=Path, required=True, help="audio.i16 and index.jsonl (scripts.asr_finetune.windows)")
    parser.add_argument("--out", type=Path, required=True, help="folder for cut-audit.json")
    parser.add_argument("--limit", type=int, default=200, help="cut windows drawn without --all (default 200)")
    parser.add_argument("--all", action="store_true", help="decode every cut window")
    parser.add_argument("--filter-index", action="store_true",
                        help="with --all: rewrite index.jsonl without the cut windows the base does not match")
    parser.add_argument("--workers", type=int, help="decoding processes (default 3 with --all, else 1)")
    parser.add_argument("--exit-zero", action="store_true", help="exit 0 even when the audit crashes")
    parser.add_argument("--device", help="override the shipped device (e.g. cpu, for a local probe)")
    parser.add_argument("--compute-type", help="override the shipped compute type (e.g. int8 on a CPU)")
    args = parser.parse_args(argv)
    if args.limit < 1 or (args.workers is not None and args.workers < 1):
        parser.error("--limit and --workers must be at least 1")
    if args.filter_index and not args.all:
        parser.error("--filter-index needs --all")
    workers = args.workers or (3 if args.all else 1)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"schema": SCHEMA, "mode": "filter" if args.filter_index else "all" if args.all else "sample",
                              "workers": workers}
    code = 0
    try:
        from scripts.asr_finetune.teacher import _settings, teacher_config

        config = teacher_config(args.device, args.compute_type)
        record["decode"] = {**_settings(config, words=False), "window": "the training window, its pads included"}
        decode, label_texts = base_decoder(args.windows, config, workers), _label_texts()
        if args.filter_index:
            record.update(filter_index(args.windows, decode, label_texts))
        else:
            record.update(sample_audit(args.windows, None if args.all else args.limit, decode, label_texts))
    except Exception as error:  # recorded, the type only (a message may hold a path)
        traceback.print_exc()
        record["error"] = type(error).__name__
        if args.filter_index:
            try:
                record.update(drop_all_cuts(args.windows))
            except Exception as again:
                record["filter"] = (f"crashed, and dropping the cut windows failed too ({type(again).__name__}): "
                                    "index.jsonl may still hold unverified cut labels")
    if "error" in record:
        code = 0 if args.exit_zero else 1
    record["wall_s"] = round(time.monotonic() - started, 1)
    (args.out / "cut-audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if "error" in record:
        print(f"cut audit: FAILED ({record['error']}); " + record.get("filter", "") + f"; exit {code}",
              file=sys.stderr, flush=True)
        return code
    print("cut audit: " + "; ".join(
        f"{side} {counts['match']:,} of {counts['compared']:,} match at the cut"
        + (f" ({counts['rate']:.1%})" if counts["rate"] is not None else "") + f", base empty {counts['base_empty']:,}"
        for side, counts in record["sides"].items())
        + (f"; kept {record['kept']:,}, dropped {record['dropped']:,} cut windows; index {record['index_lines']}"
           if args.filter_index else "") + f"; {record['wall_s']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
