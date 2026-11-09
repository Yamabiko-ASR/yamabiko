"""Check precomputed sound tags against a set's own labels, without ASR.

Usage::

    python -m scripts.asr_bench.tag_check --set <set dir> --tags <tags dir>
        [--tags <tags dir> ...] [--stem <name>=<frontend dir> ...]
        --out <report dir>

Every score is compared frame by frame with the set's reference spans on the
tags' 40 ms grid: frame ``j`` (centre ``(j + 0.5) * 0.04`` s) is inside a
line when its centre lies in the line's span. The pinned Silero VAD runs
over the raw audio (and over each ``--stem`` frontend) as the reference
detector, one stateful pass per item from a reset state, each 40 ms frame
taking the 512-sample Silero frame holding its centre sample. Scores:

- each tag group, ``human`` (voice + shout + vocal), ``voice+shout``,
  ``human-other`` (human minus other);
- ``silero`` (raw audio), ``silero:<stem>`` and their maximum
  ``silero+stems``;
- ``fused`` = max(silero, min(1, voice + shout)), the first ``vad_assist``
  rule.

Silver sets: lines are kinds ``words``, ``intj`` and ``nonlex``, the
scorer's rule (``lexicon.analyze``: a vocalization-only line holding a
lexicon interjection is ``intj``, any other vocalization-only line
``nonlex``); lines whose minimal-normalized text is empty are skipped.
Caption-free frames lie more than 0.5 s (the scorer's collar) from every
line. The report gives per kind the mean over lines of each group's mean
and maximum, the caption-free frame means, the frame AUROC of every score
for each kind's frames (and all lines) against the caption-free frames,
and per kind the share of lines whose peak reaches a threshold.

Synthetic sets: positives are frames inside lines, negatives frames inside
the known speech-free spans; the report gives the frame AUROC of every
score overall and per ``snr_db`` and ``speech_source``.

AUROC is the Mann-Whitney statistic with tied scores at their average
rank. ``report.json`` and ``report.md`` go to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench.candidates import resolve_vad
from scripts.asr_bench.lexicon import analyze
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, BenchSet, Item, load_set
from scripts.asr_bench.tag_sounds import SILERO_FRAME_SAMPLES, TAG_FRAME_SAMPLES, load_tags
from scripts.e0_evidence.normalize_ja import normalize_minimal
from yamabiko.vad.silero_chunker import SileroVAD

SCHEMA_VERSION = "asr-bench-tag-check-1.0"
COLLAR_S = 0.5
KINDS = ("words", "intj", "nonlex")
FRAME_S = TAG_FRAME_SAMPLES / SAMPLE_RATE_HZ
# Peak thresholds of the line-reach table: Silero at the shipped gate
# threshold, tag scores at two levels.
SILERO_THRESHOLD = 0.3
TAG_THRESHOLDS = (0.3, 0.5)


def line_kind(text: str) -> str | None:
    """The scorer's line kind, or None for a line it skips."""
    if not normalize_minimal(text):
        return None
    analysis = analyze(text)
    if not analysis.vocal:
        return "words"
    return "intj" if analysis.interjections else "nonlex"


def auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if positive.size == 0 or negative.size == 0:
        return None
    scores = np.concatenate([positive, negative])
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    n_pos, n_neg = positive.size, negative.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _silero_track(model: SileroVAD, pcm: np.ndarray) -> np.ndarray:
    """Silero probability per 512-sample frame, one stateful pass."""
    model.reset_states()
    frames = math.ceil(pcm.size / SILERO_FRAME_SAMPLES)
    padded = np.zeros(frames * SILERO_FRAME_SAMPLES, dtype=np.float32)
    padded[:pcm.size] = pcm
    return np.array([model(padded[k * SILERO_FRAME_SAMPLES:(k + 1) * SILERO_FRAME_SAMPLES])
                     for k in range(frames)], dtype=np.float32)


def _on_tag_grid(silero: np.ndarray, frames: int) -> np.ndarray:
    """40 ms frame ``j`` takes the Silero frame holding sample ``640 j + 320``."""
    centres = np.arange(frames, dtype=np.int64) * TAG_FRAME_SAMPLES + TAG_FRAME_SAMPLES // 2
    return silero[np.minimum(centres // SILERO_FRAME_SAMPLES, silero.size - 1)]


def _span_mask(frames: int, spans: Iterable[tuple[float, float]], widen: float = 0.0) -> np.ndarray:
    centres = (np.arange(frames) + 0.5) * FRAME_S
    mask = np.zeros(frames, dtype=bool)
    for start, end in spans:
        mask |= (centres >= start - widen) & (centres < end + widen)
    return mask


def _scores(groups: Mapping[str, np.ndarray], silero: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    human = groups["voice"] + groups["shout"] + groups["vocal"]
    voice_shout = groups["voice"] + groups["shout"]
    scores = dict(groups)
    scores.update({"human": human, "voice+shout": voice_shout,
                   "human-other": human - groups["other"]})
    scores.update(silero)
    if len(silero) > 1:
        scores["silero+stems"] = np.max(np.stack(list(silero.values())), axis=0)
    scores["fused"] = np.maximum(silero["silero"], np.minimum(1.0, voice_shout))
    return scores


class _Item:
    """One item's tag groups per tagger and Silero tracks, on the 40 ms grid."""

    def __init__(self, item: Item, frames: int, groups: dict[str, dict[str, np.ndarray]],
                 silero: dict[str, np.ndarray]):
        self.item, self.frames, self.groups, self.silero = item, frames, groups, silero


def _load_items(bench_set: BenchSet, tag_dirs: Mapping[str, Path],
                stems: Mapping[str, Path]) -> tuple[list[_Item], float]:
    artifact = resolve_vad(None)
    model = SileroVAD(artifact.path, SAMPLE_RATE_HZ, expected_sha256=artifact.sha256)
    silero_s = 0.0
    loaded = []
    for index, item in enumerate(bench_set.items, 1):
        frames = math.ceil(item.sample_count / TAG_FRAME_SAMPLES)
        groups = {}
        for name, directory in tag_dirs.items():
            track = load_tags(directory / f"{item.item_id}.npz", item.sample_count)
            groups[name] = {group: track.column(group) for group in track.names}
        begin = time.perf_counter()
        silero = {"silero": _on_tag_grid(_silero_track(model, bench_set.load_audio(item)), frames)}
        for stem, directory in stems.items():
            pcm = np.fromfile(directory / f"{item.item_id}.f32", dtype="<f4")
            if pcm.size != item.sample_count:
                raise ValueError(f"{stem}: {item.item_id} has {pcm.size} samples, expected "
                                 f"{item.sample_count}")
            silero[f"silero:{stem}"] = _on_tag_grid(_silero_track(model, pcm), frames)
        silero_s += time.perf_counter() - begin
        loaded.append(_Item(item, frames, groups, silero))
        if index % 20 == 0:
            print(f"{index}/{len(bench_set.items)} items", flush=True)
    return loaded, silero_s


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _silver(items: Sequence[_Item], tagger: str) -> dict[str, Any]:
    line_means: dict[str, list[dict[str, float]]] = {kind: [] for kind in KINDS}
    line_peaks: dict[str, list[dict[str, float]]] = {kind: [] for kind in KINDS}
    positives: dict[str, dict[str, list[np.ndarray]]] = {kind: {} for kind in (*KINDS, "all")}
    negatives: dict[str, list[np.ndarray]] = {}
    free_frames = 0
    for entry in items:
        scores = _scores(entry.groups[tagger], entry.silero)
        kinds = [(line, line_kind(line.text)) for line in entry.item.lines]
        kinds = [(line, kind) for line, kind in kinds if kind is not None]
        inside_any = _span_mask(entry.frames, ((line.start_s, line.end_s) for line, _ in kinds))
        near = _span_mask(entry.frames, ((line.start_s, line.end_s) for line in entry.item.lines),
                          widen=COLLAR_S)
        free = ~near
        free_frames += int(free.sum())
        for name, values in scores.items():
            negatives.setdefault(name, []).append(values[free])
            positives["all"].setdefault(name, []).append(values[inside_any])
        for kind in KINDS:
            mask = _span_mask(entry.frames, ((line.start_s, line.end_s)
                                             for line, k in kinds if k == kind))
            for name, values in scores.items():
                positives[kind].setdefault(name, []).append(values[mask])
        for line, kind in kinds:
            mask = _span_mask(entry.frames, [(line.start_s, line.end_s)])
            if not mask.any():
                continue
            line_means[kind].append({name: float(values[mask].mean()) for name, values in scores.items()})
            line_peaks[kind].append({name: float(values[mask].max()) for name, values in scores.items()})
    negative = {name: np.concatenate(parts) for name, parts in negatives.items()}
    score_names = list(negative)
    result: dict[str, Any] = {
        "lines": {kind: len(line_means[kind]) for kind in KINDS},
        "caption_free_frames": free_frames,
        "caption_free_hours": round(free_frames * FRAME_S / 3600, 3),
        "mean_of_line_means": {kind: {name: _round(np.mean([m[name] for m in line_means[kind]]))
                                      for name in score_names} for kind in KINDS},
        "mean_of_line_peaks": {kind: {name: _round(np.mean([m[name] for m in line_peaks[kind]]))
                                      for name in score_names} for kind in KINDS},
        "caption_free_mean": {name: _round(float(values.mean())) for name, values in negative.items()},
        "auroc_vs_caption_free": {
            kind: {name: _round(auroc(np.concatenate(positives[kind][name]), negative[name]))
                   for name in score_names}
            for kind in (*KINDS, "all")
        },
    }
    reach: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        peaks = line_peaks[kind]
        row = {f"silero>={SILERO_THRESHOLD}": np.mean([p["silero"] >= SILERO_THRESHOLD
                                                         for p in peaks])}
        for threshold in TAG_THRESHOLDS:
            row[f"voice+shout>={threshold}"] = np.mean([p["voice+shout"] >= threshold for p in peaks])
            row[f"silero miss, voice+shout>={threshold}"] = np.mean(
                [p["silero"] < SILERO_THRESHOLD and p["voice+shout"] >= threshold for p in peaks])
        reach[kind] = {key: _round(float(value)) for key, value in row.items()}
    free_rates = {f"silero>={SILERO_THRESHOLD}": float(np.mean(negative["silero"] >= SILERO_THRESHOLD))}
    for threshold in TAG_THRESHOLDS:
        free_rates[f"voice+shout>={threshold}"] = float(np.mean(negative["voice+shout"] >= threshold))
        free_rates[f"silero miss, voice+shout>={threshold}"] = float(np.mean(
            (negative["silero"] < SILERO_THRESHOLD) & (negative["voice+shout"] >= threshold)))
    result["line_peak_reach"] = reach
    result["caption_free_frame_rate"] = {key: _round(value) for key, value in free_rates.items()}
    return result


def _synthetic(items: Sequence[_Item], tagger: str) -> dict[str, Any]:
    pools: dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray]]]] = {}
    for entry in items:
        scores = _scores(entry.groups[tagger], entry.silero)
        inside = _span_mask(entry.frames, ((line.start_s, line.end_s) for line in entry.item.lines))
        free = _span_mask(entry.frames, entry.item.speech_free)
        condition = entry.item.condition
        snr = condition.get("snr_db")
        keys = ["all", f"snr_db={'clean' if snr is None else snr}",
                f"speech_source={condition.get('speech_source')}"]
        for key in keys:
            group = pools.setdefault(key, {})
            for name, values in scores.items():
                pos, neg = group.setdefault(name, ([], []))
                pos.append(values[inside])
                neg.append(values[free])
    result: dict[str, Any] = {"auroc_lines_vs_speech_free": {}, "mean_in_lines": {},
                              "mean_in_speech_free": {}, "frames": {}}
    for key, group in pools.items():
        first = next(iter(group.values()))
        result["frames"][key] = {"lines": int(sum(p.size for p in first[0])),
                                 "speech_free": int(sum(n.size for n in first[1]))}
        result["auroc_lines_vs_speech_free"][key] = {
            name: _round(auroc(np.concatenate(pos), np.concatenate(neg)))
            for name, (pos, neg) in group.items()}
        result["mean_in_lines"][key] = {name: _round(float(np.concatenate(pos).mean()))
                                        for name, (pos, _) in group.items()}
        result["mean_in_speech_free"][key] = {name: _round(float(np.concatenate(neg).mean()))
                                              for name, (_, neg) in group.items()}
    return result


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _markdown(report: Mapping[str, Any]) -> str:
    out = [f"# Tag check: {report['set_id']} ({report['kind']})", ""]
    for tagger, result in report["taggers"].items():
        out += [f"## {tagger}", ""]
        if report["kind"] == "silver":
            names = list(result["caption_free_mean"])
            groups = [name for name in names if not name.startswith(("silero", "fused"))
                      and name not in ("human", "voice+shout", "human-other")]
            lines = result["lines"]
            out += ["Mean over lines of each line's mean probability; caption-free frames last.", ""]
            out += _table(["kind", "lines", *groups], [
                *([kind, str(lines[kind]), *(_fmt(result["mean_of_line_means"][kind][g]) for g in groups)]
                  for kind in KINDS),
                ["caption-free", f"{result['caption_free_hours']} h",
                 *(_fmt(result["caption_free_mean"][g]) for g in groups)],
            ])
            out += ["Mean over lines of each line's peak probability.", ""]
            out += _table(["kind", *groups], (
                [kind, *(_fmt(result["mean_of_line_peaks"][kind][g]) for g in groups)]
                for kind in KINDS))
            out += ["Frame AUROC against caption-free frames.", ""]
            out += _table(["score", *KINDS, "all"], (
                [name, *(_fmt(result["auroc_vs_caption_free"][kind][name]) for kind in (*KINDS, "all"))]
                for name in names))
            reach_keys = list(result["caption_free_frame_rate"])
            out += ["Share of lines whose peak reaches the threshold; caption-free frame rate last.", ""]
            out += _table(["kind", *reach_keys], [
                *([kind, *(_fmt(result["line_peak_reach"][kind][key]) for key in reach_keys)]
                  for kind in KINDS),
                ["caption-free frames", *(_fmt(result["caption_free_frame_rate"][key])
                                          for key in reach_keys)],
            ])
        else:
            aurocs = result["auroc_lines_vs_speech_free"]
            keys = list(aurocs)
            names = list(aurocs["all"])
            out += ["Frame AUROC, frames inside lines against speech-free frames.", ""]
            out += _table(["score", *keys], ([name, *(_fmt(aurocs[key][name]) for key in keys)]
                                             for name in names))
            out += ["Mean inside lines / inside speech-free spans (all items).", ""]
            out += _table(["score", "lines", "speech-free"], (
                [name, _fmt(result["mean_in_lines"]["all"][name]),
                 _fmt(result["mean_in_speech_free"]["all"][name])] for name in names))
    return "\n".join(out)


def check(bench_set: BenchSet, tag_dirs: Mapping[str, Path], stems: Mapping[str, Path]) -> dict[str, Any]:
    items, silero_s = _load_items(bench_set, tag_dirs, stems)
    analyse = _silver if bench_set.kind == "silver" else _synthetic
    manifests = {}
    for name, directory in tag_dirs.items():
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests[name] = {key: manifest.get(key) for key in ("tagger", "set_id", "tags_digest",
                                                                   "complete", "identity")}
    artifact = resolve_vad(None)
    return {
        "schema_version": SCHEMA_VERSION, "set_id": bench_set.set_id, "kind": bench_set.kind,
        "item_count": len(items), "frame_s": FRAME_S, "collar_s": COLLAR_S,
        "silero": {**artifact.identity(), "stems": {name: str(path) for name, path in stems.items()},
                   "seconds": round(silero_s, 1)},
        "tags": {name: str(path) for name, path in tag_dirs.items()}, "tag_manifests": manifests,
        "taggers": {name: analyse(items, name) for name in tag_dirs},
    }


def _named_dir(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = Path(raw).parent.name
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", required=True, type=Path, help="bench set directory")
    parser.add_argument("--tags", required=True, action="append",
                        help="tags directory, optionally <name>=<dir> (default name: the "
                             "tagger folder)")
    parser.add_argument("--stem", action="append", default=[],
                        help="<name>=<frontend dir> to run Silero over as well")
    parser.add_argument("--out", required=True, type=Path, help="report directory")
    args = parser.parse_args(argv)
    bench_set = load_set(args.set)
    tag_dirs = dict(_named_dir(raw) for raw in args.tags)
    stems = dict(_named_dir(raw) for raw in args.stem)
    report = check(bench_set, tag_dirs, stems)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")
    markdown = _markdown(report)
    (args.out / "report.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
