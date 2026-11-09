"""Precompute per-item sound tags for a bench set: run one AudioSet Strong
frame tagger over every item and store its 40 ms class probabilities and
their coarse groups, for ``vad_assist`` candidates and analysis.

Usage::

    python -m scripts.asr_bench.tag_sounds --set <set dir> --tagger <name>
        [--taggers <json>] [--items ID[,ID...]] [--device cuda|cpu]
        [--threads N] [--hop-frames H] [--lookahead-frames R] [--tail-frames T]
        [--out <tags dir>]

``--taggers`` (default ``<bench root>\\taggers.json``, see
``taggers.example.json``) lists the local taggers. Each entry has a
``name``, a ``kind``, ``hop_frames`` and ``lookahead_frames`` (the
streaming layout, in 40 ms frames; see ``tag_sounds_worker.py``),
``batch`` (windows per forward pass) and an ``about`` object (sources and
licences, copied into the manifest). Kind ``efficientsed`` (the
EfficientSED code, which also runs the PretrainedSED transformers) adds
``python`` (the tagger venv's interpreter), ``repo`` (the EfficientSED
checkout), ``checkpoint`` (its ``.pt`` file), ``model`` (``fmn10`` and the
other widths, or ``BEATs``), ``seq_model_type`` and ``seq_model_dim`` (the
sequence model on top of an ``fmn``, or null); the model runs in that venv
through ``tag_sounds_worker.py`` (``--device`` default ``cuda``). Kind
``efficientsed_onnx`` adds ``onnx``, a graph written by
``export_tagger_onnx.py``; it runs in this process on the CPU through
``tagger_onnx.py`` (ONNX Runtime and NumPy only; ``--device`` must be
``cpu``, its default) and writes the same files. ``--hop-frames`` and
``--lookahead-frames`` replace the entry's layout for one run.
``--tail-frames`` (default 0) also tags that many 40 ms frames of the zeros
after the item, as a live tagger would tag the silence a stream ends with;
the bench runner's trailing silence takes them instead of zeros.

The tags directory (default ``<bench root>\\tags\\<tagger>\\<set_id>``, or
``<bench root>\\tags\\<tagger>-h<H>-l<R>[-tail<T>]\\<set_id>`` when the
layout differs from the entry's or a tail is tagged) receives
``<item_id>.npz`` (``probs``: frames x 447 float16 class probabilities;
``groups``: frames x 7 float32 group probabilities (``GROUPS``), each the
maximum over the group's classes; ``group_names``; ``frame_samples``, 640;
``tail_frames``, absent in files from before tails)
plus ``manifest.json``: the tagger's identity (checkpoint or graph
SHA-256, code commit, parameters, group definitions), each run's runtime
versions, the class list and the resolved groups, and per item the file
SHA-256, the tagging time and real-time factor and the CUDA memory peaks.
``tags_digest`` is the canonical digest of the item digests, as a bench run
records for the same files. Frame ``j`` covers item samples
``[640 j, 640 (j + 1))``: an item of ``N`` samples has ``ceil(N / 640)``
frames, then the tail's. ``silero_frames`` maps a track to the chunker's
512-sample frames.

A rerun skips every item whose manifest entry matches its file, so an
interrupted run resumes where it stopped. A manifest written by a different
tagger identity or set is refused; choose another ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.asr_bench import tagger_onnx
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, BenchSet, Item, bench_root, load_set
from scripts.e0_evidence.wire import canonical_sha256

SCHEMA_VERSION = "asr-bench-tags-1.0"
ONNX_KIND = "efficientsed_onnx"
KINDS = ("efficientsed", ONNX_KIND)
DEFAULT_THREADS = 4
TAG_FRAME_SAMPLES = 640
WINDOW_FRAMES = 250
SILERO_FRAME_SAMPLES = 512

# Coarse groups over the 447 AudioSet Strong classes, by exact class name.
# ``ambience`` holds the acoustic-scene and noise-floor classes, which fire
# under any sound (Background noise averages 0.49 over a clean synthetic
# scene), so ``other`` keeps the sound events. ``other`` (None) is every
# class in no other group. A group's probability is the maximum over its
# classes.
GROUPS: dict[str, tuple[str, ...] | None] = {
    "voice": (
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
        "Speech synthesizer", "Whispering", "Human voice",
    ),
    "shout": (
        "Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Children shouting", "Screaming",
    ),
    "vocal": (
        "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
        "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Groan",
        "Grunt", "Gasp", "Breathing", "Pant", "Wheeze", "Snoring", "Snort", "Cough", "Sneeze",
        "Sniff", "Throat clearing", "Hiccup", "Yawn", "Humming", "Respiratory sounds",
    ),
    "music": ("Music",),
    "singing": (
        "Singing", "Male singing", "Female singing", "Child singing", "Synthetic singing",
        "Choir", "Chant", "Mantra", "Yodeling", "Rapping",
    ),
    "ambience": (
        "Background noise", "Noise", "Environmental noise", "Channel, environment and background",
        "Inside, small room", "Inside, large room or hall", "Inside, public space",
        "Outside, urban or manmade", "Echo", "Reverberation", "Static", "Hum", "Mains hum",
        "White noise, pink noise", "Tape hiss", "Distortion", "Wind noise (microphone)",
        "Sound reproduction", "Sound equipment", "Other sourceless", "Unknown sound",
    ),
    "other": None,
}
GROUP_NAMES = tuple(GROUPS)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).resolve().with_name("tag_sounds_worker.py")
_TAGGER_ONNX = Path(tagger_onnx.__file__).resolve()
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_COMMON_FIELDS = {"name", "kind", "hop_frames", "lookahead_frames", "batch", "about"}
_FIELDS = {
    "efficientsed": _COMMON_FIELDS | {"python", "repo", "checkpoint", "model",
                                      "seq_model_type", "seq_model_dim"},
    ONNX_KIND: _COMMON_FIELDS | {"onnx"},
}
DEVICES = {"efficientsed": ("cuda", "cpu"), ONNX_KIND: ("cpu",)}
_TIMING = {
    "efficientsed": "tag_s is the tagger's forward passes over the item (mel, model, sigmoid), "
                    "CUDA-synchronized; it excludes model load, the warm-up pass and file I/O",
    ONNX_KIND: "tag_s is the ONNX Runtime runs over the item (window assembly and the graph: "
               "mel, model, sigmoid); it excludes session load, the warm-up run and file I/O",
}


@dataclass(frozen=True)
class TaggerSpec:
    name: str
    kind: str
    hop_frames: int
    lookahead_frames: int
    batch: int
    about: Mapping[str, Any]
    # Kind efficientsed: the tagger venv's interpreter, the EfficientSED
    # checkout, the checkpoint and the model parameters.
    python: Path | None = None
    repo: Path | None = None
    checkpoint: Path | None = None
    model: str | None = None
    seq_model_type: str | None = None
    seq_model_dim: int | None = None
    # Kind efficientsed_onnx: the graph export_tagger_onnx.py wrote.
    onnx: Path | None = None
    # 40 ms frames of the zeros after the item that are tagged too.
    tail_frames: int = 0


def _positive_int(value: object, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}")
    return value


def _checked_layout(hop: object, lookahead: object, label: str) -> tuple[int, int]:
    checked = (_positive_int(hop, f"{label}.hop_frames"),
               _positive_int(lookahead, f"{label}.lookahead_frames", 0))
    if sum(checked) > WINDOW_FRAMES:
        raise ValueError(f"{label}: hop_frames + lookahead_frames must not exceed {WINDOW_FRAMES}")
    return checked


def with_layout(spec: TaggerSpec, hop_frames: int | None, lookahead_frames: int | None,
                tail_frames: int = 0) -> TaggerSpec:
    """``spec`` with the hop and look-ahead (40 ms frames) replaced where
    given, tagging ``tail_frames`` frames of zeros after each item."""
    hop, lookahead = _checked_layout(
        spec.hop_frames if hop_frames is None else hop_frames,
        spec.lookahead_frames if lookahead_frames is None else lookahead_frames, spec.name)
    tail = _positive_int(tail_frames, f"{spec.name}.tail_frames", 0)
    return replace(spec, hop_frames=hop, lookahead_frames=lookahead, tail_frames=tail)


def load_tagger(path: Path, name: str) -> TaggerSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("taggers")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected {{\"taggers\": [...]}}")
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1:
        known = sorted(str(entry.get("name")) for entry in entries)
        raise ValueError(f"{path}: {len(matches)} taggers named {name!r} (known: {known})")
    entry = matches[0]
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"tagger name {name!r} must match {_NAME_RE.pattern}")
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{name}: kind must be one of {KINDS}")
    fields = _FIELDS[kind]
    if set(entry) != fields:
        raise ValueError(f"{name}: fields must be exactly {sorted(fields)}; missing "
                         f"{sorted(fields - set(entry))}, unknown {sorted(set(entry) - fields)}")
    hop, lookahead = _checked_layout(entry["hop_frames"], entry["lookahead_frames"], name)
    if not isinstance(entry["about"], Mapping):
        raise ValueError(f"{name}: about must be an object")
    common = {"name": name, "kind": kind, "hop_frames": hop, "lookahead_frames": lookahead,
              "batch": _positive_int(entry["batch"], f"{name}.batch"),
              "about": dict(entry["about"])}
    if kind == ONNX_KIND:
        graph = Path(entry["onnx"])
        if not graph.is_absolute() or not graph.is_file():
            raise FileNotFoundError(f"{name}: {graph} must be an existing absolute file")
        return TaggerSpec(**common, onnx=graph)
    model = entry["model"]
    if not isinstance(model, str) or not (model == "BEATs" or re.fullmatch(r"fmn\d\d", model)):
        raise ValueError(f"{name}: model must be fmnNN or BEATs")
    seq_type, seq_dim = entry["seq_model_type"], entry["seq_model_dim"]
    if model == "BEATs" and (seq_type is not None or seq_dim is not None):
        raise ValueError(f"{name}: BEATs takes no sequence model")
    if seq_type is not None:
        if seq_type not in ("tf", "gru", "hybrid", "attn", "tcn"):
            raise ValueError(f"{name}: unsupported seq_model_type {seq_type!r}")
        seq_dim = _positive_int(seq_dim, f"{name}.seq_model_dim")
    elif seq_dim is not None:
        raise ValueError(f"{name}: seq_model_dim needs a seq_model_type")
    python, repo, checkpoint = Path(entry["python"]), Path(entry["repo"]), Path(entry["checkpoint"])
    for file in (python, checkpoint):
        if not file.is_absolute() or not file.is_file():
            raise FileNotFoundError(f"{name}: {file} must be an existing absolute file")
    if not repo.is_absolute() or not (repo / "models" / "prediction_wrapper.py").is_file():
        raise FileNotFoundError(f"{name}: {repo} must be an absolute EfficientSED checkout")
    return TaggerSpec(**common, python=python, repo=repo, checkpoint=checkpoint, model=model,
                      seq_model_type=seq_type, seq_model_dim=seq_dim)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def _git_identity(cwd: Path) -> dict[str, Any]:
    try:
        return {"head": _git(cwd, "rev-parse", "HEAD"),
                "dirty": bool(_git(cwd, "status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _group_spec() -> list[list[Any]]:
    """``GROUPS`` in order as ``[name, classes or None]`` pairs, as the worker
    and the manifest identity take them."""
    return [[name, None if members is None else list(members)] for name, members in GROUPS.items()]


def identity(spec: TaggerSpec, device: str) -> dict[str, Any]:
    """Everything that determines the output: weights, code, parameters, groups."""
    params = {
        "window_frames": WINDOW_FRAMES, "frame_samples": TAG_FRAME_SAMPLES,
        "hop_frames": spec.hop_frames, "lookahead_frames": spec.lookahead_frames,
        "batch": spec.batch, "device": device, "dtype": "float32",
    }
    if spec.tail_frames:
        # Absent at 0, so directories tagged before tails keep their identity.
        params["tail_frames"] = spec.tail_frames
    if spec.kind == ONNX_KIND:
        assert spec.onnx is not None
        # The graph pins the weights, the frontend and the model code; its
        # metadata names the checkpoint and the EfficientSED commit.
        return {
            "kind": spec.kind,
            "files": {"onnx": _file_identity(spec.onnx)},
            "source": tagger_onnx.read_metadata(spec.onnx)["source"],
            "params": {"runtime": "onnxruntime", **params},
            "groups": _group_spec(),
        }
    assert spec.checkpoint is not None and spec.repo is not None
    return {
        "kind": spec.kind,
        "files": {"checkpoint": _file_identity(spec.checkpoint)},
        "code": {"efficientsed": _git_identity(spec.repo)},
        "params": {"model": spec.model, "seq_model_type": spec.seq_model_type,
                   "seq_model_dim": spec.seq_model_dim, **params},
        "groups": _group_spec(),
    }


def resolved_groups(classes: Sequence[str]) -> dict[str, list[str]]:
    """Every group's classes, ``other`` expanded to the remaining classes."""
    named = {cls for members in GROUPS.values() if members is not None for cls in members}
    missing = sorted(named - set(classes))
    if missing:
        raise ValueError(f"group classes not in the tagger's class list: {missing}")
    return {name: (sorted(set(classes) - named) if members is None else list(members))
            for name, members in GROUPS.items()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _open_manifest(path: Path, spec: TaggerSpec, bench_set: BenchSet,
                   ident: Mapping[str, Any]) -> dict[str, Any]:
    set_sha256 = _file_sha256(bench_set.root / "set.json")
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION, "tagger": spec.name, "about": dict(spec.about),
            "set_id": bench_set.set_id, "set_manifest_sha256": set_sha256,
            "identity": dict(ident), "timing": _TIMING[spec.kind],
            "frames": (f"frame j covers item samples [{TAG_FRAME_SAMPLES} j, "
                       f"{TAG_FRAME_SAMPLES} (j + 1)); block b keeps frames [b H, (b + 1) H) "
                       f"from the {WINDOW_FRAMES}-frame window ending R frames after the "
                       f"block (H = hop_frames, R = lookahead_frames); audio outside the "
                       f"item is zeros; the last tail_frames frames (identity params, 0 "
                       f"when absent) tag the zeros after the item"),
            "runs": [], "items": {},
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema {manifest.get('schema_version')!r}")
    expected = {"tagger": spec.name, "set_id": bench_set.set_id,
                "set_manifest_sha256": set_sha256, "identity": dict(ident)}
    differing = sorted(key for key, value in expected.items() if manifest.get(key) != value)
    if differing:
        raise ValueError(
            f"{path} was written for a different {', '.join(differing)}; choose another --out"
        )
    manifest["about"] = dict(spec.about)
    return manifest


def _verified_done(manifest: dict[str, Any], out_dir: Path, bench_set: BenchSet) -> set[str]:
    """Item ids whose file exists with the manifest's digest; other entries go."""
    known = {item.item_id for item in bench_set.items}
    done = set()
    for item_id, entry in list(manifest["items"].items()):
        path = out_dir / f"{item_id}.npz"
        if item_id in known and path.is_file() and _file_sha256(path) == entry.get("sha256"):
            done.add(item_id)
        else:
            del manifest["items"][item_id]
    return done


def _summarize(manifest: dict[str, Any], bench_set: BenchSet) -> None:
    entries = manifest["items"]
    audio_s = sum(entry["duration_s"] for entry in entries.values())
    tag_s = sum(entry["tag_s"] for entry in entries.values())
    manifest["item_count"] = len(entries)
    manifest["set_item_count"] = len(bench_set.items)
    manifest["complete"] = len(entries) == len(bench_set.items)
    manifest["tags_digest"] = canonical_sha256(
        {item_id: entry["sha256"] for item_id, entry in entries.items()})
    vram = [entry["vram_peak_reserved_mib"] for entry in entries.values()
            if entry.get("vram_peak_reserved_mib") is not None]
    manifest["totals"] = {
        "audio_s": round(audio_s, 3), "tag_s": round(tag_s, 3),
        "rtf": round(tag_s / audio_s, 6) if audio_s else None,
        "tag_s_per_audio_hour": round(3600 * tag_s / audio_s, 2) if audio_s else None,
        "vram_peak_reserved_mib": round(max(vram), 1) if vram else None,
    }


def _thread_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    return env


def _job(spec: TaggerSpec, items: list[Item], bench_set: BenchSet, out_dir: Path, device: str,
         threads: int) -> dict[str, Any]:
    """The job ``tag_sounds_worker.py`` (or ``tagger_onnx.job_events``) runs."""
    job: dict[str, Any] = {
        "hop_frames": spec.hop_frames, "lookahead_frames": spec.lookahead_frames,
        "batch": spec.batch, "device": device, "threads": threads,
        "groups": _group_spec(),
        "items": [{"item_id": item.item_id, "input": str(bench_set.audio_path(item)),
                   "sample_count": item.sample_count,
                   "output": str(out_dir / f"{item.item_id}.npz")} for item in items],
    }
    if spec.tail_frames:
        job["tail_frames"] = spec.tail_frames
    if spec.kind == ONNX_KIND:
        job["onnx"] = str(spec.onnx)
    else:
        job.update(repo=str(spec.repo), checkpoint=str(spec.checkpoint), model=spec.model,
                   seq_model_type=spec.seq_model_type, seq_model_dim=spec.seq_model_dim)
    return job


def _record_events(spec: TaggerSpec, events: Iterable[Mapping[str, Any]], items: list[Item],
                   bench_set: BenchSet, out_dir: Path, manifest: dict[str, Any],
                   manifest_path: Path, run: dict[str, Any], run_index: int) -> int:
    """Record the worker protocol's events in the run and the manifest
    (written after every item); returns the number of items tagged."""
    by_id = {item.item_id: item for item in items}
    count = 0
    for event in events:
        if event["event"] == "ready":
            run["runtime"] = event["versions"]
            run.update(model_load_s=round(event["load_s"], 3),
                       warmup_s=round(event["warmup_s"], 3), worker_device=event["device"])
            run.update((key, event[key]) for key in ("torch_threads", "intra_op_threads")
                       if key in event)
            classes = event["classes"]
            groups = resolved_groups(classes)
            if manifest.setdefault("classes", classes) != classes:
                raise ValueError(f"{spec.name}: the class list changed between runs")
            manifest["groups"] = groups
            continue
        item = by_id[event["item_id"]]
        path = out_dir / f"{item.item_id}.npz"
        track = load_tags(path, item.sample_count)
        entry = {
            "sha256": track.sha256, "sample_count": item.sample_count,
            "duration_s": item.duration_s, "frames": event["frames"],
            "windows": event["windows"], "run": run_index,
            "rtf": round(event["tag_s"] / item.duration_s, 6),
            **{key: round(event[key], 4) if isinstance(event[key], float) else event[key]
               for key in ("tag_s", "io_s", "vram_peak_allocated_mib",
                           "vram_peak_reserved_mib")},
        }
        manifest["items"][item.item_id] = entry
        _summarize(manifest, bench_set)
        _write_json(manifest_path, manifest)
        count += 1
        means = track.groups[:track.groups.shape[0] - track.tail_frames].mean(axis=0)
        print(f"[{count}/{len(items)}] {item.item_id}: rtf {entry['rtf']:.5f} "
              f"({entry['tag_s']:.2f} s), mean "
              + " ".join(f"{name} {value:.2f}" for name, value in zip(track.names, means)),
              flush=True)
    return count


def _run_worker(spec: TaggerSpec, job: dict[str, Any], items: list[Item], bench_set: BenchSet,
                out_dir: Path, manifest: dict[str, Any], manifest_path: Path,
                run: dict[str, Any], run_index: int) -> int:
    """Run the job in the tagger's venv through ``tag_sounds_worker.py``."""
    assert spec.python is not None
    work = out_dir.with_name(out_dir.name + ".work")
    work.mkdir(parents=True, exist_ok=True)
    job_path = work / "job.json"
    _write_json(job_path, job)
    process = subprocess.Popen(
        [str(spec.python), str(_WORKER), "--job", str(job_path)],
        stdout=subprocess.PIPE, text=True, encoding="utf-8", env=_thread_env(job["threads"]),
    )
    assert process.stdout is not None
    try:
        count = _record_events(spec, (json.loads(line) for line in process.stdout), items,
                               bench_set, out_dir, manifest, manifest_path, run, run_index)
    except BaseException:
        process.kill()
        raise
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"{spec.name}: tag_sounds_worker.py exited with {code}")
    job_path.unlink()
    work.rmdir()
    return count


def tag(bench_set: BenchSet, spec: TaggerSpec, out_dir: Path, device: str, threads: int,
        item_ids: set[str] | None = None) -> dict[str, Any]:
    unknown = sorted((item_ids or set()) - {item.item_id for item in bench_set.items})
    if unknown:
        raise ValueError(f"items not in {bench_set.set_id}: {unknown}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = _open_manifest(manifest_path, spec, bench_set, identity(spec, device))
    done = _verified_done(manifest, out_dir, bench_set)
    pending = [item for item in bench_set.items
               if item.item_id not in done and (item_ids is None or item.item_id in item_ids)]
    print(f"{spec.name} on {bench_set.set_id}: {len(done)} done, {len(pending)} to run "
          f"({device}, {threads} threads)", flush=True)
    if pending:
        run = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "device": device, "threads": threads, "items": len(pending),
               "code": {"git": _git_identity(_REPO_ROOT),
                        "tag_sounds_py_sha256": _file_sha256(Path(__file__)),
                        "tag_sounds_worker_py_sha256": _file_sha256(_WORKER)},
               "python": sys.version.split()[0]}
        if spec.kind == ONNX_KIND:
            run["code"]["tagger_onnx_py_sha256"] = _file_sha256(_TAGGER_ONNX)
        manifest["runs"].append(run)
        run_index = len(manifest["runs"]) - 1
        job = _job(spec, pending, bench_set, out_dir, device, threads)
        begin = time.perf_counter()
        count = 0
        try:
            if spec.kind == ONNX_KIND:
                count = _record_events(spec, tagger_onnx.job_events(job), pending, bench_set,
                                       out_dir, manifest, manifest_path, run, run_index)
                run["memory"] = tagger_onnx.memory_mib()
            else:
                count = _run_worker(spec, job, pending, bench_set, out_dir, manifest,
                                    manifest_path, run, run_index)
        finally:
            run["wall_s"] = round(time.perf_counter() - begin, 3)
            run["items_done"] = count
            _summarize(manifest, bench_set)
            _write_json(manifest_path, manifest)
    _summarize(manifest, bench_set)
    _write_json(manifest_path, manifest)
    totals = manifest["totals"]
    print(f"{manifest['item_count']}/{manifest['set_item_count']} items in {out_dir}; "
          f"rtf {totals['rtf']} ({totals['tag_s_per_audio_hour']} s per audio hour), "
          f"tags digest {manifest['tags_digest']}", flush=True)
    return manifest


@dataclass(frozen=True)
class TagTrack:
    # (frames, len(names)) float32 group probabilities on the 40 ms grid:
    # the item's frames, then tail_frames frames of the zeros after it.
    groups: np.ndarray
    names: tuple[str, ...]
    sha256: str
    tail_frames: int = 0

    def column(self, name: str, tail: bool = False) -> np.ndarray:
        """One group over the item's frames, and with ``tail`` the tail's."""
        end = self.groups.shape[0] - (0 if tail else self.tail_frames)
        return self.groups[:end, self.names.index(name)]


def load_tags(path: Path, sample_count: int) -> TagTrack:
    """Read and check one item's ``.npz``: 640-sample frames covering
    ``sample_count`` samples and its tail, finite group probabilities in
    [0, 1]."""
    data = Path(path).read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        frame_samples = int(archive["frame_samples"])
        groups = np.asarray(archive["groups"], dtype=np.float32)
        names = tuple(str(name) for name in archive["group_names"])
        tail = int(archive["tail_frames"]) if "tail_frames" in archive.files else 0
    if frame_samples != TAG_FRAME_SAMPLES:
        raise ValueError(f"{path}: frame_samples {frame_samples}, expected {TAG_FRAME_SAMPLES}")
    frames = math.ceil(sample_count / TAG_FRAME_SAMPLES) + tail
    if tail < 0 or groups.shape != (frames, len(names)):
        raise ValueError(f"{path}: groups have shape {groups.shape}, expected "
                         f"({frames}, {len(names)}) for {sample_count} samples and a tail of "
                         f"{tail} frames")
    if not np.all(np.isfinite(groups)) or groups.min(initial=0) < 0 or groups.max(initial=0) > 1:
        raise ValueError(f"{path}: group probabilities must be finite and in [0, 1]")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: group names repeat")
    return TagTrack(groups, names, hashlib.sha256(data).hexdigest(), tail)


def silero_frames(values: np.ndarray, frame_count: int) -> np.ndarray:
    """A 40 ms track on the chunker's grid: chunker frame ``k`` (samples
    ``[512 k, 512 (k + 1))``) takes the 40 ms frame holding its centre
    sample ``512 k + 256``, and 0 past the end of the track."""
    centres = np.arange(frame_count, dtype=np.int64) * SILERO_FRAME_SAMPLES + SILERO_FRAME_SAMPLES // 2
    index = centres // TAG_FRAME_SAMPLES
    out = np.zeros(frame_count, dtype=np.float32)
    inside = index < values.shape[0]
    out[inside] = values[index[inside]]
    return out


@dataclass(frozen=True)
class TagLayout:
    """A track's streaming layout in 40 ms tag frames: block ``b`` keeps
    frames ``[b H, (b + 1) H)`` of the window that ends ``R`` frames after
    the block (``H`` = ``hop_frames``, ``R`` = ``lookahead_frames``); the
    track also holds ``tail_frames`` frames of the zeros after the item.

    ``release_sample`` is the live delay line of production's
    ``SoundTagAssist`` (``yamabiko/vad/sound_tagger.py``) under this layout.
    A chunker frame reaches Silero and the gate only once the window that
    tags its block has run. That window ends at sample
    ``E = 640 ((b + 1) H + R)``, so it can run once the chunker frame that
    holds sample ``E - 1`` arrives; the tagger runs as that frame is pushed
    and releases every held frame of the block. Production's layout (H = R =
    12) gives ``E = 7680 (b + 2)``, the end of its hop ``b + 2``: the window
    ending at hop ``h`` tags block ``h - 2``, and a hop is exactly 15
    chunker frames. A hop of 6 frames (3,840 samples, 7.5 chunker frames)
    has the tagger run on whichever frame completes its window.
    """

    hop_frames: int
    lookahead_frames: int
    # Frames of the zeros after each item the track also holds.
    tail_frames: int = 0

    def release_sample(self, frame: int) -> int:
        """Samples the chunker has received when chunker frame ``frame``
        (samples ``[512 k, 512 (k + 1))``, tagged by the 40 ms frame holding
        sample ``512 k + 256``) is released to Silero and the gate."""
        tag_frame = (frame * SILERO_FRAME_SAMPLES + SILERO_FRAME_SAMPLES // 2) // TAG_FRAME_SAMPLES
        block = tag_frame // self.hop_frames
        window_end = TAG_FRAME_SAMPLES * ((block + 1) * self.hop_frames + self.lookahead_frames)
        return -(-window_end // SILERO_FRAME_SAMPLES) * SILERO_FRAME_SAMPLES

    def to_json(self) -> dict[str, Any]:
        return {
            "hop_frames": self.hop_frames, "lookahead_frames": self.lookahead_frames,
            "tail_frames": self.tail_frames, "frame_samples": TAG_FRAME_SAMPLES,
            "hop_s": self.hop_frames * TAG_FRAME_SAMPLES / SAMPLE_RATE_HZ,
            "lookahead_s": self.lookahead_frames * TAG_FRAME_SAMPLES / SAMPLE_RATE_HZ,
            "tail_s": self.tail_frames * TAG_FRAME_SAMPLES / SAMPLE_RATE_HZ,
        }


def load_layout(tags_dir: Path) -> TagLayout:
    """The layout a tags directory's ``manifest.json`` records."""
    path = Path(tags_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema {manifest.get('schema_version')!r}")
    params = manifest["identity"]["params"]
    if (params["window_frames"], params["frame_samples"]) != (WINDOW_FRAMES, TAG_FRAME_SAMPLES):
        raise ValueError(f"{path}: windows of {params['window_frames']} frames of "
                         f"{params['frame_samples']} samples; the delay line models "
                         f"{WINDOW_FRAMES} of {TAG_FRAME_SAMPLES}")
    hop, lookahead = _checked_layout(params["hop_frames"], params["lookahead_frames"], str(path))
    tail = _positive_int(params.get("tail_frames", 0), f"{path}.tail_frames", 0)
    return TagLayout(hop, lookahead, tail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", required=True, type=Path, help="bench set directory")
    parser.add_argument("--tagger", required=True, help="tagger name")
    parser.add_argument("--taggers", type=Path, help="taggers JSON "
                        "(default <bench root>/taggers.json)")
    parser.add_argument("--items", help="comma-separated item ids (default: all)")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        help="default cuda (efficientsed) or cpu (efficientsed_onnx, CPU only)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="CPU threads for the tagger")
    parser.add_argument("--hop-frames", type=int,
                        help="tag frames (40 ms) per hop; default the tagger entry's")
    parser.add_argument("--lookahead-frames", type=int,
                        help="tag frames (40 ms) of look-ahead; default the tagger entry's")
    parser.add_argument("--tail-frames", type=int, default=0,
                        help="tag frames (40 ms) of the zeros after each item to tag too")
    parser.add_argument("--out", type=Path, help="tags directory (default "
                        "<bench root>/tags/<tagger>/<set_id>, or <tagger>-h<H>-l<R>[-tail<T>] "
                        "for another layout or a tail)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    bench_set = load_set(args.set)
    entry = load_tagger(args.taggers or bench_root() / "taggers.json", args.tagger)
    spec = with_layout(entry, args.hop_frames, args.lookahead_frames, args.tail_frames)
    device = args.device or DEVICES[spec.kind][0]
    if device not in DEVICES[spec.kind]:
        parser.error(f"{spec.kind} taggers run on {' or '.join(DEVICES[spec.kind])}")
    directory = spec.name
    if ((spec.hop_frames, spec.lookahead_frames) != (entry.hop_frames, entry.lookahead_frames)
            or spec.tail_frames):
        directory = f"{spec.name}-h{spec.hop_frames}-l{spec.lookahead_frames}"
        if spec.tail_frames:
            directory += f"-tail{spec.tail_frames}"
    out_dir = args.out or bench_root() / "tags" / directory / bench_set.set_id
    item_ids = None if args.items is None else {i for i in args.items.split(",") if i}
    manifest = tag(bench_set, spec, out_dir, device, args.threads, item_ids)
    return 0 if manifest["item_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
