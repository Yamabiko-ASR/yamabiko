"""Produce a separation frontend for a bench set: run one speech separator
or enhancer over every item and store its speech stem as the ``audio_dir``
of a candidate ``frontend``.

Usage::

    python -m scripts.asr_bench.separate --set <set dir> --separator <name>
        [--separators <json>] [--items ID[,ID...]] [--device cuda|cpu]
        [--threads N] [--out <frontend dir>]

``--separators`` (default ``<bench root>\\separators.json``, see
``separators.example.json``) lists the local separators. Each entry has a
``name``, a ``kind``, the ``sample_rate`` the tool runs at, an ``about``
object (sources and licences, copied into the manifest) and the fields of
its kind:

- ``msst``: ``python`` (the separator venv's interpreter), ``model_type``,
  ``config``, ``checkpoint`` and ``stem``; the model runs in that venv
  through ``separate_msst.py``.
- ``deep_filter``: ``binary`` (the DeepFilterNet ``deep-filter``
  executable), ``model`` (its model ``.tar.gz``), ``args`` (extra
  command-line arguments) and ``pad_end_samples`` (zeros appended to the
  exported input at the tool's rate; ``--compensate-delay`` drops the
  model's delay from the end of the output, so padding by that delay keeps
  the output exactly as long as the item); one process runs per item.

Each item is exported with ``frontend_io.write_wav`` after resampling from
16 kHz to the tool's rate with soxr (VHQ), separated over its whole length
(no voice detection selects what the tool hears), and imported back with
``frontend_io.import_item``. The stem keeps the level the tool gives it; the
driver applies no gain. The frontend directory (default
``<bench root>\\frontends\\<separator>\\<set_id>``) receives
``<item_id>.f32`` plus ``manifest.json``: the separator's identity (the
SHA-256 of every file it loads and its parameters), each run's runtime
versions, and per item the output SHA-256, the separation time and real-time
factor, the CUDA memory peaks and the input and output levels (whole item,
reference lines, and the speech-free spans of synthetic items).
``audio_digest`` equals the ``frontend.manifest_sha256`` a bench run records
for the same files.

A rerun skips every item whose manifest entry matches its file, so an
interrupted run resumes where it stopped. A manifest written by a different
separator identity or set is refused; choose another ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soxr

from scripts.asr_bench.frontend_io import import_item, write_wav
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, BenchSet, Item, bench_root, load_set
from scripts.e0_evidence.wire import canonical_sha256

SCHEMA_VERSION = "asr-bench-frontend-1.0"
KINDS = ("msst", "deep_filter")
DEFAULT_THREADS = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).resolve().with_name("separate_msst.py")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_KIND_FIELDS = {
    "msst": {"python", "model_type", "config", "checkpoint", "stem"},
    "deep_filter": {"binary", "model", "args", "pad_end_samples"},
}
_COMMON_FIELDS = {"name", "kind", "sample_rate", "about"}
_TIMING = {
    "msst": "separate_s is the Separator.separate call, CUDA-synchronized; it "
            "excludes model load, resampling and file I/O",
    "deep_filter": "separate_s is the wall time of one deep-filter process per "
                   "item, including its model initialisation and WAV I/O",
}
_FLOOR_DB = -200.0


@dataclass(frozen=True)
class SeparatorSpec:
    name: str
    kind: str
    sample_rate: int
    about: Mapping[str, Any]
    fields: Mapping[str, Any]

    def files(self) -> dict[str, Path]:
        """The files whose bytes determine the output."""
        roles = ("config", "checkpoint") if self.kind == "msst" else ("binary", "model")
        return {role: Path(self.fields[role]) for role in roles}


def load_separator(path: Path, name: str) -> SeparatorSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("separators")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected {{\"separators\": [...]}}")
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1:
        known = sorted(str(entry.get("name")) for entry in entries)
        raise ValueError(f"{path}: {len(matches)} separators named {name!r} (known: {known})")
    entry = matches[0]
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"separator name {name!r} must match {_NAME_RE.pattern}")
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{name}: kind must be one of {KINDS}, got {kind!r}")
    expected = _COMMON_FIELDS | _KIND_FIELDS[kind]
    if set(entry) != expected:
        raise ValueError(
            f"{name}: fields must be exactly {sorted(expected)}; missing "
            f"{sorted(expected - set(entry))}, unknown {sorted(set(entry) - expected)}"
        )
    rate = entry["sample_rate"]
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError(f"{name}: sample_rate must be a positive int")
    if not isinstance(entry["about"], Mapping):
        raise ValueError(f"{name}: about must be an object")
    fields = {key: entry[key] for key in _KIND_FIELDS[kind]}
    if kind == "deep_filter":
        if not (isinstance(fields["args"], list)
                and all(isinstance(arg, str) for arg in fields["args"])):
            raise ValueError(f"{name}: args must be a list of strings")
        pad = fields["pad_end_samples"]
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise ValueError(f"{name}: pad_end_samples must be a non-negative int")
    spec = SeparatorSpec(name, kind, rate, dict(entry["about"]), fields)
    executable = Path(fields["python"] if kind == "msst" else fields["binary"])
    for path in [executable, *spec.files().values()]:
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(f"{name}: {path} must be an existing absolute file")
    return spec


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(spec: SeparatorSpec, device: str) -> dict[str, Any]:
    """Everything that determines the output: file digests and parameters."""
    files = {
        role: {"name": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for role, path in spec.files().items()
    }
    params: dict[str, Any] = {"sample_rate": spec.sample_rate, "device": device}
    if spec.kind == "msst":
        params.update(model_type=spec.fields["model_type"], stem=spec.fields["stem"])
    else:
        params.update(args=list(spec.fields["args"]),
                      pad_end_samples=spec.fields["pad_end_samples"])
    return {"kind": spec.kind, "files": files, "params": params}


def _git_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        return {
            "head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _db(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    rms = math.sqrt(float(np.mean(np.square(values, dtype=np.float64))))
    return round(max(20 * math.log10(rms), _FLOOR_DB) if rms > 0 else _FLOOR_DB, 2)


def _spans_db(pcm: np.ndarray, spans: Iterable[tuple[float, float]]) -> float | None:
    parts = [
        pcm[int(round(start * SAMPLE_RATE_HZ)):int(round(end * SAMPLE_RATE_HZ))]
        for start, end in spans
    ]
    return _db(np.concatenate(parts)) if parts else None


def levels(raw: np.ndarray, out: np.ndarray, item: Item) -> dict[str, Any]:
    """RMS levels (dBFS) and peaks of the raw item and the frontend output."""
    line_spans = [(line.start_s, line.end_s) for line in item.lines]
    result: dict[str, Any] = {}
    for label, pcm in (("in", raw), ("out", out)):
        result[f"rms_db_{label}"] = _db(pcm)
        result[f"lines_rms_db_{label}"] = _spans_db(pcm, line_spans)
        result[f"speech_free_rms_db_{label}"] = _spans_db(pcm, item.speech_free)
        result[f"peak_{label}"] = round(float(np.max(np.abs(pcm))), 6)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _open_manifest(
    path: Path, spec: SeparatorSpec, bench_set: BenchSet, ident: Mapping[str, Any]
) -> dict[str, Any]:
    set_sha256 = _file_sha256(bench_set.root / "set.json")
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION, "separator": spec.name, "about": dict(spec.about),
            "set_id": bench_set.set_id, "set_manifest_sha256": set_sha256,
            "identity": dict(ident), "stem_gain": "none: the tool's stem at its own level",
            "whole_item": "each item is separated over its whole length; no voice "
                          "detection selects what the tool hears",
            "timing": _TIMING[spec.kind], "runs": [], "items": {},
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema {manifest.get('schema_version')!r}")
    expected = {"separator": spec.name, "set_id": bench_set.set_id,
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
        path = out_dir / f"{item_id}.f32"
        if item_id in known and path.is_file() and _file_sha256(path) == entry.get("sha256"):
            done.add(item_id)
        else:
            del manifest["items"][item_id]
    return done


def _summarize(manifest: dict[str, Any], bench_set: BenchSet) -> None:
    entries = manifest["items"]
    audio_s = sum(entry["duration_s"] for entry in entries.values())
    separate_s = sum(entry["separate_s"] for entry in entries.values())
    manifest["item_count"] = len(entries)
    manifest["set_item_count"] = len(bench_set.items)
    manifest["complete"] = len(entries) == len(bench_set.items)
    manifest["audio_digest"] = canonical_sha256(
        {item_id: entry["sha256"] for item_id, entry in entries.items()})
    vram = [entry["vram_peak_reserved_mib"] for entry in entries.values()
            if entry.get("vram_peak_reserved_mib") is not None]
    manifest["totals"] = {
        "audio_s": round(audio_s, 3), "separate_s": round(separate_s, 3),
        "rtf": round(separate_s / audio_s, 5) if audio_s else None,
        "vram_peak_reserved_mib": round(max(vram), 1) if vram else None,
    }


class _Recorder:
    """Imports one tool output, records its entry and saves the manifest."""

    def __init__(self, bench_set: BenchSet, spec: SeparatorSpec, out_dir: Path,
                 manifest: dict[str, Any], manifest_path: Path, run_index: int, total: int):
        self.bench_set, self.spec, self.out_dir = bench_set, spec, out_dir
        self.manifest, self.manifest_path = manifest, manifest_path
        self.run_index, self.total, self.count = run_index, total, 0

    def record(self, item: Item, output: Path, timing: Mapping[str, Any]) -> None:
        target = self.out_dir / f"{item.item_id}.f32"
        imported = import_item(output, item, target, self.spec.sample_rate)
        data = target.read_bytes()
        out_pcm = np.frombuffer(data, dtype="<f4")
        entry = {
            "sha256": hashlib.sha256(data).hexdigest(), "sample_count": item.sample_count,
            "duration_s": item.duration_s, "run": self.run_index,
            "tool_sample_rate": imported.sample_rate, "tool_channels": imported.channels,
            "length_delta_samples": imported.length_delta_samples,
            "rtf": round(timing["separate_s"] / item.duration_s, 5),
            "levels": levels(self.bench_set.load_audio(item), out_pcm, item),
            **{key: (round(value, 4) if isinstance(value, float) else value)
               for key, value in timing.items()},
        }
        self.manifest["items"][item.item_id] = entry
        _summarize(self.manifest, self.bench_set)
        _write_json(self.manifest_path, self.manifest)
        self.count += 1
        lv = entry["levels"]
        free = ""
        if lv["speech_free_rms_db_in"] is not None:
            free = (f", speech-free {lv['speech_free_rms_db_in']:.1f} -> "
                    f"{lv['speech_free_rms_db_out']:.1f} dB")
        print(f"[{self.count}/{self.total}] {item.item_id}: rtf {entry['rtf']:.4f} "
              f"({entry['separate_s']:.2f} s), level {lv['rms_db_in']:.1f} -> "
              f"{lv['rms_db_out']:.1f} dB{free}", flush=True)


def _export(bench_set: BenchSet, item: Item, path: Path, rate: int, pad_end: int = 0) -> None:
    pcm = bench_set.load_audio(item)
    if rate != SAMPLE_RATE_HZ:
        pcm = soxr.resample(pcm, SAMPLE_RATE_HZ, rate, quality="VHQ")
    if pad_end:
        pcm = np.concatenate([pcm, np.zeros(pad_end, np.float32)])
    write_wav(path, pcm, rate)


def _thread_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    return env


def _run_msst(spec: SeparatorSpec, items: list[Item], bench_set: BenchSet, work: Path,
              device: str, threads: int, recorder: _Recorder, run: dict[str, Any]) -> None:
    entries = []
    for item in items:
        source, output = work / f"{item.item_id}.in.wav", work / f"{item.item_id}.out.wav"
        _export(bench_set, item, source, spec.sample_rate)
        entries.append({"item_id": item.item_id, "input": str(source), "output": str(output)})
    job = {key: spec.fields[key] for key in ("model_type", "config", "checkpoint", "stem")}
    job.update(device=device, threads=threads, items=entries)
    job_path = work / "job.json"
    _write_json(job_path, job)
    by_id = {item.item_id: item for item in items}
    process = subprocess.Popen(
        [spec.fields["python"], str(_WORKER), "--job", str(job_path)],
        stdout=subprocess.PIPE, text=True, encoding="utf-8", env=_thread_env(threads),
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            event = json.loads(line)
            if event["event"] == "ready":
                run["runtime"] = event["versions"]
                run.update(model_load_s=round(event["load_s"], 3),
                           torch_threads=event["torch_threads"], worker_device=event["device"])
                if event["sample_rate"] != spec.sample_rate:
                    raise ValueError(f"{spec.name}: model rate {event['sample_rate']} Hz, "
                                     f"separators file says {spec.sample_rate} Hz")
                continue
            item = by_id[event["item_id"]]
            source, output = work / f"{item.item_id}.in.wav", work / f"{item.item_id}.out.wav"
            recorder.record(item, output, {
                key: event[key] for key in ("separate_s", "io_s", "vram_peak_allocated_mib",
                                            "vram_peak_reserved_mib")
            })
            output.unlink()
            source.unlink()
    except BaseException:
        process.kill()
        raise
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"{spec.name}: separate_msst.py exited with {code}")


def _deep_filter_version(binary: str) -> str:
    return subprocess.run([binary, "--version"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _run_deep_filter(spec: SeparatorSpec, items: list[Item], bench_set: BenchSet,
                     work: Path, recorder: _Recorder, run: dict[str, Any]) -> None:
    binary = spec.fields["binary"]
    run["runtime"] = {"deep_filter": _deep_filter_version(binary)}
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    for item in items:
        source = work / f"{item.item_id}.wav"
        _export(bench_set, item, source, spec.sample_rate, spec.fields["pad_end_samples"])
        command = [binary, *spec.fields["args"], "--model", spec.fields["model"],
                   "--output-dir", str(out_dir), str(source)]
        begin = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - begin
        if result.returncode != 0:
            raise RuntimeError(f"{spec.name}: {item.item_id} failed ({result.returncode}):\n"
                               f"{result.stdout}\n{result.stderr}")
        output = out_dir / source.name
        recorder.record(item, output, {"separate_s": wall})
        output.unlink()
        source.unlink()


def separate(bench_set: BenchSet, spec: SeparatorSpec, out_dir: Path, device: str,
             threads: int, item_ids: set[str] | None = None) -> dict[str, Any]:
    if spec.kind == "deep_filter" and device != "cpu":
        raise ValueError(f"{spec.name}: deep_filter runs on the CPU only; pass --device cpu")
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
               "code": {"git": _git_identity(), "separate_py_sha256": _file_sha256(Path(__file__)),
                        "separate_msst_py_sha256": _file_sha256(_WORKER)},
               "python": sys.version.split()[0]}
        manifest["runs"].append(run)
        recorder = _Recorder(bench_set, spec, out_dir, manifest, manifest_path,
                             len(manifest["runs"]) - 1, len(pending))
        work = out_dir.with_name(out_dir.name + ".work")
        work.mkdir(parents=True, exist_ok=True)
        begin = time.perf_counter()
        try:
            if spec.kind == "msst":
                _run_msst(spec, pending, bench_set, work, device, threads, recorder, run)
            else:
                _run_deep_filter(spec, pending, bench_set, work, recorder, run)
        finally:
            run["wall_s"] = round(time.perf_counter() - begin, 3)
            run["items_done"] = recorder.count
            _summarize(manifest, bench_set)
            _write_json(manifest_path, manifest)
        shutil.rmtree(work)
    _summarize(manifest, bench_set)
    _write_json(manifest_path, manifest)
    totals = manifest["totals"]
    print(f"{manifest['item_count']}/{manifest['set_item_count']} items in {out_dir}; "
          f"rtf {totals['rtf']}, audio digest {manifest['audio_digest']}", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", required=True, type=Path, help="bench set directory")
    parser.add_argument("--separator", required=True, help="separator name")
    parser.add_argument("--separators", type=Path, help="separators JSON "
                        "(default <bench root>/separators.json)")
    parser.add_argument("--items", help="comma-separated item ids (default: all)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="CPU threads for the separator")
    parser.add_argument("--out", type=Path, help="frontend directory "
                        "(default <bench root>/frontends/<separator>/<set_id>)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    bench_set = load_set(args.set)
    spec = load_separator(args.separators or bench_root() / "separators.json", args.separator)
    out_dir = args.out or bench_root() / "frontends" / spec.name / bench_set.set_id
    item_ids = None if args.items is None else {i for i in args.items.split(",") if i}
    manifest = separate(bench_set, spec, out_dir, args.device, args.threads, item_ids)
    return 0 if manifest["item_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
