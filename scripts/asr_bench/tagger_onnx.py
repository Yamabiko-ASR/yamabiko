"""Run an exported EfficientSED sound tagger (``export_tagger_onnx.py``)
with ONNX Runtime and NumPy alone, as the ``efficientsed_onnx`` taggers of
``tag_sounds.py`` do; compare two tags directories; time the live
streaming layout.

Usage::

    python -m scripts.asr_bench.tagger_onnx compare --a <tags dir> --b <tags dir>
        [--items ID[,ID...]] [--out <json>]
    python -m scripts.asr_bench.tagger_onnx timing --onnx <file> --set <set dir>
        --items ID[,ID...] [--threads N] [--hop-frames H] [--lookahead-frames R]
        [--out <json>]

The graph maps ``audio`` (batch x 160000 float32, 10 s windows at 16 kHz)
to ``probs`` (batch x 250 x 447 float32, the sigmoid of the strong head per
40 ms frame); its ``yamabiko_tagger`` metadata entry holds the class list
and the source checkpoint. ``job_events`` lays the windows out exactly as
``tag_sounds_worker.py`` does (block ``b`` keeps frames ``[b H, (b + 1) H)``
of the window ending ``R`` frames after the block) and writes the same
``.npz`` files. Sessions run on the CPU provider with ``threads`` intra-op
threads, one inter-op thread and no spin-waiting between runs.

``compare`` takes the items both directories hold (or ``--items``) and
reports, per item and over all of them, the largest and the mean absolute
difference per 40 ms frame of the group probabilities (float32 in the
files), of the ``vad_assist`` term ``min(1, voice + shout)``, and of the
class probabilities (float16 in the files, so differences up to about 5e-4
are rounding).

``timing`` replays the live layout (production's hop and look-ahead unless
given): once per hop it runs one window (batch 1) ending ``R`` frames after
the hop's last frame, over the given items, and reports the time per hop
(mean, median, 95th percentile, maximum; window assembly and the graph, the
mel frontend included), the real-time factor (time per hop over the hop's
duration) and the process memory (working set and its peak before the
session loads, after it loads and after the run). Run it once per thread
count, so each run has a process of its own.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Iterator, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench.sets import load_set
from scripts.asr_bench.tag_sounds_worker import (
    FRAME_SAMPLES,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    block_windows,
    group_columns,
    group_probs,
    pad_item,
    write_tags,
)
from yamabiko.vad.sound_tagger import HOP_FRAMES, LOOKAHEAD_FRAMES

SCHEMA = "yamabiko-sound-tagger-onnx-1"
METADATA_KEY = "yamabiko_tagger"
SAMPLE_RATE = 16000


def _session(path: Path, threads: int, optimize: bool = True) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if not optimize:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def _checked_metadata(session: Any, path: Path) -> dict[str, Any]:
    raw = session.get_modelmeta().custom_metadata_map.get(METADATA_KEY)
    if raw is None:
        raise ValueError(f"{path}: no {METADATA_KEY} metadata; export it with "
                         f"export_tagger_onnx.py")
    metadata = json.loads(raw)
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"{path}: metadata schema {metadata.get('schema')!r}, expected {SCHEMA}")
    classes = metadata["classes"]
    expected = {"input": ["batch", WINDOW_SAMPLES], "output": ["batch", WINDOW_FRAMES, len(classes)]}
    for key, shape in expected.items():
        if metadata[key]["shape"] != shape:
            raise ValueError(f"{path}: {key} shape {metadata[key]['shape']}, expected {shape}")
    if metadata["input"]["sample_rate"] != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {metadata['input']['sample_rate']}")
    return metadata


def read_metadata(path: Path) -> dict[str, Any]:
    """The graph's ``yamabiko_tagger`` metadata, checked against the layout
    this module expects (the graph is opened without optimization)."""
    return _checked_metadata(_session(path, 1, optimize=False), path)


class OnnxTagger:
    """One exported tagger on the CPU: 10 s windows in, frame probabilities out."""

    def __init__(self, path: Path, threads: int) -> None:
        self.session = _session(path, threads)
        self.metadata = _checked_metadata(self.session, path)
        self.classes: list[str] = list(self.metadata["classes"])

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        """``(batch, 160000)`` float32 windows to ``(batch, 250, classes)``."""
        (probs,) = self.session.run(["probs"], {"audio": windows})
        if probs.shape != (windows.shape[0], WINDOW_FRAMES, len(self.classes)):
            raise ValueError(f"tagger returned shape {probs.shape}")
        return probs


def tag_padded(tagger: OnnxTagger, padded: np.ndarray, frames: int, hop: int, lookahead: int,
               batch: int) -> np.ndarray:
    """``(frames, classes)``: block ``b`` keeps frames ``[b hop, (b + 1) hop)``
    of the window ending ``lookahead`` frames after it (``padded`` from
    ``pad_item``)."""
    blocks = math.ceil(frames / hop)
    keep = WINDOW_FRAMES - lookahead - hop
    kept = []
    for first in range(0, blocks, batch):
        count = min(batch, blocks - first)
        probs = tagger(block_windows(padded, first, count, hop, lookahead, np))
        kept.append(probs[:, keep:keep + hop].reshape(-1, probs.shape[2]))
    return np.concatenate(kept)[:frames]


def runtime_versions() -> dict[str, Any]:
    return {"python": platform.python_version(), "onnxruntime": version("onnxruntime"),
            "numpy": np.__version__}


def job_events(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Tag a ``tag_sounds_worker.py`` style job (``onnx`` in place of the
    checkpoint fields, device ``cpu``) and yield the worker's events."""
    if job["device"] != "cpu":
        raise ValueError("the ONNX tagger runs on the CPU only")
    hop, lookahead, batch = int(job["hop_frames"]), int(job["lookahead_frames"]), int(job["batch"])
    if not (hop >= 1 and lookahead >= 0 and hop + lookahead <= WINDOW_FRAMES and batch >= 1):
        raise ValueError("need hop_frames >= 1, lookahead_frames >= 0, their sum <= 250, batch >= 1")
    threads = int(job["threads"])
    begin = time.perf_counter()
    tagger = OnnxTagger(Path(job["onnx"]), threads)
    load_s = time.perf_counter() - begin
    begin = time.perf_counter()
    tagger(np.zeros((batch, WINDOW_SAMPLES), dtype=np.float32))
    warmup_s = time.perf_counter() - begin
    group_names, columns = group_columns(tagger.classes, job["groups"], np)
    yield {"event": "ready", "versions": runtime_versions(), "device": "cpu",
           "classes": tagger.classes, "intra_op_threads": threads, "load_s": load_s,
           "warmup_s": warmup_s}
    for entry in job["items"]:
        io_begin = time.perf_counter()
        audio = np.fromfile(entry["input"], dtype="<f4")
        if audio.size != entry["sample_count"]:
            raise ValueError(f"{entry['input']}: {audio.size} samples, expected "
                             f"{entry['sample_count']}")
        tail = int(job.get("tail_frames", 0))
        frames = math.ceil(audio.size / FRAME_SAMPLES) + tail
        padded = pad_item(audio, frames, hop, lookahead, np)
        io_s = time.perf_counter() - io_begin
        begin = time.perf_counter()
        probs = tag_padded(tagger, padded, frames, hop, lookahead, batch)
        tag_s = time.perf_counter() - begin
        groups = group_probs(probs, columns, np)
        io_begin = time.perf_counter()
        write_tags(Path(entry["output"]), probs, groups, group_names, np, tail)
        io_s += time.perf_counter() - io_begin
        yield {"event": "item", "item_id": entry["item_id"], "tag_s": tag_s, "io_s": io_s,
               "frames": frames, "windows": math.ceil(frames / hop),
               "vram_peak_allocated_mib": None, "vram_peak_reserved_mib": None}


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        return (np.asarray(archive["probs"], dtype=np.float32),
                np.asarray(archive["groups"], dtype=np.float32),
                tuple(str(name) for name in archive["group_names"]))


def _stats(diff: np.ndarray) -> dict[str, float]:
    return {"max": float(diff.max(initial=0.0)), "mean": float(diff.mean()) if diff.size else 0.0}


def compare(a: Path, b: Path, item_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Absolute differences between the tags of the items in both directories."""
    def stems(directory: Path) -> set[str]:
        return {path.stem for path in directory.glob("*.npz") if not path.stem.endswith(".part")}

    ids = sorted(stems(a) & stems(b)) if item_ids is None else list(item_ids)
    if not ids:
        raise ValueError(f"no items in both {a} and {b}")
    items: dict[str, Any] = {}
    names: tuple[str, ...] | None = None
    totals = {"frames": 0, "groups_sum": 0.0, "groups_max": 0.0, "assist_sum": 0.0,
              "assist_max": 0.0, "probs_sum": 0.0, "probs_max": 0.0, "probs_count": 0,
              "group_values_above_1e-3": 0, "group_max": None}
    for item_id in ids:
        probs_a, groups_a, names_a = _load_npz(a / f"{item_id}.npz")
        probs_b, groups_b, names_b = _load_npz(b / f"{item_id}.npz")
        if names_a != names_b or groups_a.shape != groups_b.shape or probs_a.shape != probs_b.shape:
            raise ValueError(f"{item_id}: the two files differ in layout")
        names = names_a
        group_diff = np.abs(groups_a - groups_b)
        voice, shout = names.index("voice"), names.index("shout")
        assist_a = np.minimum(1.0, groups_a[:, voice] + groups_a[:, shout])
        assist_b = np.minimum(1.0, groups_b[:, voice] + groups_b[:, shout])
        assist_diff = np.abs(assist_a - assist_b)
        probs_diff = np.abs(probs_a - probs_b)
        per_group = group_diff.max(axis=0)
        items[item_id] = {
            "frames": int(groups_a.shape[0]), "groups": _stats(group_diff),
            "group_max": dict(zip(names, (round(float(v), 8) for v in per_group))),
            "assist": _stats(assist_diff), "probs": _stats(probs_diff),
            "group_values_above_1e-3": int((group_diff > 1e-3).sum()),
        }
        totals["frames"] += groups_a.shape[0]
        totals["groups_sum"] += float(group_diff.sum())
        totals["groups_max"] = max(totals["groups_max"], float(group_diff.max(initial=0.0)))
        totals["assist_sum"] += float(assist_diff.sum())
        totals["assist_max"] = max(totals["assist_max"], float(assist_diff.max(initial=0.0)))
        totals["probs_sum"] += float(probs_diff.sum())
        totals["probs_max"] = max(totals["probs_max"], float(probs_diff.max(initial=0.0)))
        totals["probs_count"] += probs_diff.size
        totals["group_values_above_1e-3"] += items[item_id]["group_values_above_1e-3"]
        totals["group_max"] = (per_group if totals["group_max"] is None
                               else np.maximum(totals["group_max"], per_group))
    assert names is not None
    frames = totals["frames"]
    return {
        "a": str(a), "b": str(b), "items": len(ids), "frames": frames,
        "groups": {"max": totals["groups_max"], "mean": totals["groups_sum"] / (frames * len(names))},
        "group_max": dict(zip(names, (float(v) for v in totals["group_max"]))),
        "group_values_above_1e-3": totals["group_values_above_1e-3"],
        "assist": {"max": totals["assist_max"], "mean": totals["assist_sum"] / frames},
        "probs": {"max": totals["probs_max"], "mean": totals["probs_sum"] / totals["probs_count"]},
        "per_item": items,
    }


def memory_mib() -> dict[str, float | None]:
    """This process's working set and its peak (Windows), in MiB."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    return {"working_set_mib": round(info.rss / 2**20, 1),
            "peak_working_set_mib": None if peak is None else round(peak / 2**20, 1)}


def timing(onnx: Path, set_dir: Path, item_ids: Sequence[str], threads: int, hop: int,
           lookahead: int) -> dict[str, Any]:
    """The live layout: one batch-1 window per hop over the given items."""
    bench_set = load_set(set_dir)
    by_id = {item.item_id: item for item in bench_set.items}
    unknown = sorted(set(item_ids) - set(by_id))
    if unknown:
        raise ValueError(f"items not in {bench_set.set_id}: {unknown}")
    memory = {"before_load": memory_mib()}
    begin = time.perf_counter()
    tagger = OnnxTagger(onnx, threads)
    load_s = time.perf_counter() - begin
    memory["after_load"] = memory_mib()
    silence = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    for _ in range(3):
        tagger(silence)
    hop_s: list[float] = []
    audio_s = 0.0
    for item_id in item_ids:
        item = by_id[item_id]
        audio = bench_set.load_audio(item)
        audio_s += item.duration_s
        frames = math.ceil(audio.size / FRAME_SAMPLES)
        padded = pad_item(audio, frames, hop, lookahead, np)
        for block in range(math.ceil(frames / hop)):
            begin = time.perf_counter()
            tagger(block_windows(padded, block, 1, hop, lookahead, np))
            hop_s.append(time.perf_counter() - begin)
    memory["after_run"] = memory_mib()
    ms = sorted(1000 * value for value in hop_s)
    hop_ms = hop * FRAME_SAMPLES * 1000 / SAMPLE_RATE
    mean = statistics.fmean(ms)
    return {
        "onnx": str(onnx), "set_id": bench_set.set_id, "items": list(item_ids),
        "audio_s": round(audio_s, 3), "threads": threads, "hop_frames": hop,
        "lookahead_frames": lookahead, "hop_ms": hop_ms, "hops": len(ms),
        "load_s": round(load_s, 3),
        "ms_per_hop": {"mean": round(mean, 2), "median": round(statistics.median(ms), 2),
                       "p95": round(ms[min(len(ms) - 1, math.ceil(0.95 * len(ms)) - 1)], 2),
                       "max": round(ms[-1], 2)},
        "rtf": round(mean / hop_ms, 4),
        "memory": memory, "versions": runtime_versions(),
        "definition": "one batch-1 window per hop (window assembly and the graph, mel "
                      "frontend included); rtf = mean ms per hop / hop ms",
    }


def _write_or_print(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    if out is None:
        print(text, end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare_parser = commands.add_parser("compare", help="differences between two tags directories")
    compare_parser.add_argument("--a", required=True, type=Path, help="tags directory")
    compare_parser.add_argument("--b", required=True, type=Path, help="tags directory")
    compare_parser.add_argument("--items", help="comma-separated item ids (default: shared)")
    compare_parser.add_argument("--out", type=Path, help="JSON report (default: stdout)")
    timing_parser = commands.add_parser("timing", help="time the live streaming layout")
    timing_parser.add_argument("--onnx", required=True, type=Path, help="exported tagger")
    timing_parser.add_argument("--set", required=True, type=Path, help="bench set directory")
    timing_parser.add_argument("--items", required=True, help="comma-separated item ids")
    timing_parser.add_argument("--threads", type=int, default=4)
    timing_parser.add_argument("--hop-frames", type=int, default=HOP_FRAMES)
    timing_parser.add_argument("--lookahead-frames", type=int, default=LOOKAHEAD_FRAMES)
    timing_parser.add_argument("--out", type=Path, help="JSON report (default: stdout)")
    args = parser.parse_args(argv)
    items = None if args.items is None else [i for i in args.items.split(",") if i]
    if args.command == "compare":
        report = compare(args.a, args.b, items)
        summary = {key: value for key, value in report.items() if key != "per_item"}
        print(json.dumps(summary, indent=1))
        if args.out is not None:
            _write_or_print(report, args.out)
        return 0
    if args.threads < 1 or args.hop_frames < 1 or args.lookahead_frames < 0 or \
            args.hop_frames + args.lookahead_frames > WINDOW_FRAMES:
        parser.error("need threads >= 1, hop_frames >= 1, lookahead_frames >= 0, "
                     "their sum <= 250")
    assert items is not None
    _write_or_print(timing(args.onnx, args.set, items, args.threads, args.hop_frames,
                           args.lookahead_frames), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
