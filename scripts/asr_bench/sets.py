"""Bench-set manifest: the contract between the set builders and the runner.

A bench set is a directory holding ``set.json`` and one 16 kHz mono
little-endian float32 file per item. Each item lists its reference lines
(text with start and end times) and, for synthetic items, the spans where no
speech was placed.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "asr-bench-set-1.0"
SAMPLE_RATE_HZ = 16000
KINDS = ("synthetic", "silver")
LINE_SOURCES = ("cv17", "voicevox", "subtitle", "verbatim")


def bench_root() -> Path:
    """Private output root: ``YAMABIKO_ASR_BENCH_ROOT`` or
    ``%LOCALAPPDATA%\\Yamabiko\\asr-bench``. Anime-derived audio and
    subtitle text live here, never in the repository."""
    override = os.environ.get("YAMABIKO_ASR_BENCH_ROOT")
    if override:
        return Path(override)
    return Path(os.environ["LOCALAPPDATA"]) / "Yamabiko" / "asr-bench"


@dataclass(frozen=True)
class Line:
    line_id: str
    start_s: float
    end_s: float
    text: str
    source: str
    overlap: bool = False


@dataclass(frozen=True)
class Item:
    item_id: str
    audio: str
    sample_count: int
    condition: Mapping[str, Any]
    lines: tuple[Line, ...]
    speech_free: tuple[tuple[float, float], ...]

    @property
    def duration_s(self) -> float:
        return self.sample_count / SAMPLE_RATE_HZ


@dataclass(frozen=True)
class BenchSet:
    set_id: str
    kind: str
    created_at_utc: str
    build: Mapping[str, Any]
    items: tuple[Item, ...]
    root: Path

    def audio_path(self, item: Item) -> Path:
        return self.root / item.audio

    def load_audio(self, item: Item) -> np.ndarray:
        pcm = np.fromfile(self.audio_path(item), dtype="<f4")
        if pcm.size != item.sample_count:
            raise ValueError(
                f"{item.item_id}: audio has {pcm.size} samples, manifest says "
                f"{item.sample_count}"
            )
        return pcm


def _finite_seconds(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return seconds


def _line(raw: Mapping[str, Any], item_id: str, duration_s: float) -> Line:
    line_id = str(raw["line_id"])
    label = f"{item_id}/{line_id}"
    start = _finite_seconds(raw["start_s"], f"{label} start_s")
    end = _finite_seconds(raw["end_s"], f"{label} end_s")
    if not start < end <= duration_s + 1e-6:
        raise ValueError(f"{label}: span must satisfy start < end <= duration")
    text = raw["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label}: text must be a non-empty string")
    source = raw["source"]
    if source not in LINE_SOURCES:
        raise ValueError(f"{label}: unknown source {source!r}")
    overlap = raw.get("overlap", False)
    if not isinstance(overlap, bool):
        raise ValueError(f"{label}: overlap must be a bool")
    return Line(line_id, start, end, text, source, overlap)


def _speech_free(
    raw: Iterable[object], item_id: str, duration_s: float
) -> tuple[tuple[float, float], ...]:
    spans: list[tuple[float, float]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{item_id}: speech_free[{index}] must be [start, end]")
        start = _finite_seconds(value[0], f"{item_id} speech_free start")
        end = _finite_seconds(value[1], f"{item_id} speech_free end")
        if not start < end <= duration_s + 1e-6:
            raise ValueError(f"{item_id}: speech_free[{index}] is out of bounds")
        if spans and start < spans[-1][1]:
            raise ValueError(f"{item_id}: speech_free spans must be sorted and disjoint")
        spans.append((start, end))
    return tuple(spans)


def _item(raw: Mapping[str, Any], root: Path, kind: str) -> Item:
    item_id = str(raw["item_id"])
    audio = str(raw["audio"])
    resolved = (root / audio).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"{item_id}: audio path must stay inside the set directory")
    sample_count = raw["sample_count"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError(f"{item_id}: sample_count must be a positive int")
    duration_s = sample_count / SAMPLE_RATE_HZ
    condition = raw.get("condition", {})
    if not isinstance(condition, Mapping):
        raise ValueError(f"{item_id}: condition must be an object")
    lines = tuple(_line(line, item_id, duration_s) for line in raw["lines"])
    if len({line.line_id for line in lines}) != len(lines):
        raise ValueError(f"{item_id}: line_id values must be unique")
    speech_free = _speech_free(raw.get("speech_free", ()), item_id, duration_s)
    if kind == "silver" and speech_free:
        raise ValueError(f"{item_id}: silver items have no known speech-free spans")
    for line in lines:
        for start, end in speech_free:
            if line.start_s < end and start < line.end_s:
                raise ValueError(
                    f"{item_id}/{line.line_id}: line overlaps a speech_free span"
                )
    return Item(item_id, audio, sample_count, dict(condition), lines, speech_free)


def load_set(path: Path) -> BenchSet:
    """Load and validate a bench set from its directory or its ``set.json``."""
    path = Path(path)
    manifest_path = path / "set.json" if path.is_dir() else path
    root = manifest_path.parent
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported bench-set schema {raw.get('schema_version')!r}")
    if raw.get("sample_rate_hz") != SAMPLE_RATE_HZ:
        raise ValueError("bench sets are 16 kHz")
    kind = raw["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown bench-set kind {kind!r}")
    items = tuple(_item(item, root, kind) for item in raw["items"])
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("item_id values must be unique")
    return BenchSet(
        set_id=str(raw["set_id"]),
        kind=kind,
        created_at_utc=str(raw["created_at_utc"]),
        build=dict(raw.get("build", {})),
        items=items,
        root=root,
    )


def write_set(
    directory: Path,
    *,
    set_id: str,
    kind: str,
    created_at_utc: str,
    build: Mapping[str, Any],
    items: Iterable[Mapping[str, Any]],
) -> BenchSet:
    """Write ``set.json`` atomically, then reload it through ``load_set`` so a
    builder can never leave an invalid manifest behind."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "set_id": set_id,
        "kind": kind,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "created_at_utc": created_at_utc,
        "build": dict(build),
        "items": list(items),
    }
    target = directory / "set.json"
    temporary = directory / ".set.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        loaded = load_set(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, target)
    return BenchSet(
        loaded.set_id, loaded.kind, loaded.created_at_utc, loaded.build,
        loaded.items, directory,
    )


def write_audio(path: Path, pcm: np.ndarray) -> int:
    """Write 16 kHz mono float32 LE samples; return the sample count."""
    samples = np.ascontiguousarray(pcm, dtype="<f4")
    if samples.ndim != 1:
        raise ValueError("bench audio must be mono")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.tofile(path)
    return int(samples.size)
