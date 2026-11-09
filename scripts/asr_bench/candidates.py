"""Bench candidates: one VAD artifact plus one ASR backend, optionally with
chunker and ASR config overrides and a precomputed audio frontend.

A candidate file holds one candidate object, or ``{"candidates": [...]}``
from which ``--name`` selects one entry::

    {"name": "baseline",
     "vad": null,
     "asr": {"backend": "faster_whisper",
             "model": "kotoba-tech/kotoba-whisper-v2.0-faster",
             "compute_type": "float16", "beam_size": 5},
     "chunker": {"max_chunk_ms": 5000},
     "asr_config": {"boundary_pad_ms": 250,
                    "faster_whisper": {"initial_prompt": ""}},
     "frontend": {"name": "sep", "audio_dir": "<absolute dir>", "feed": "both"}}

``vad: null`` (or a ``vad`` object whose fields are all null) is the shipped
pinned Silero file. Otherwise ``vad`` names an ONNX graph with production's
I/O contract by ``sha256`` plus either an absolute ``path`` or a ``url``; a
URL is downloaded once into the bench cache and verified before use.

``asr.backend`` is ``faster_whisper`` (production ``build_asr``; ``model`` is
a Hugging Face repo id or an absolute local CTranslate2 directory),
``parakeet_onnx`` (the E0 onnx-asr NeMo TDT adapter, fp32 on CPU, run from the
bench venv), ``sherpa_transducer`` (a sherpa-onnx offline transducer such as
ReazonSpeech k2, fp32 on CPU) or ``transformers_asr`` (a transformers-native
Qwen3-ASR or Cohere ASR checkpoint on CUDA, ``compute_type`` a torch dtype
name). For the last three ``model`` is an absolute local directory.

``chunker`` overrides keys of the shipped ``vad`` config section (same keys,
same types); ``vad_threshold`` is an alias for ``chunker.threshold``.
``asr_config`` overrides keys of the shipped ``asr`` section the same way,
nested like it; ``faster_whisper.model`` and ``compute_type`` come from
``asr`` only, ``asr.beam_size`` is an alias for
``asr_config.faster_whisper.beam_size`` (for ``transformers_asr`` it is the
beam width, default greedy) and the other backends take
``boundary_pad_ms`` only. Every other setting is the shipped default.

``frontend`` names a directory of precomputed ``<item_id>.f32`` files
(16 kHz mono float32 LE, one per item, same sample count as the item) made
offline by any tool (``frontend_io`` converts tool output). ``feed`` says who
hears it: ``vad`` (the chunker), ``asr`` (each window's decode, cut from the
same sample span) or ``both``.

``vad_assist`` fuses extra tracks into the speech probability the
production chunker receives, frame by frame, as the maximum of the
candidate's Silero on the chunker's audio and:

- ``silero``: ``{"name", "audio_dir"}``, the same Silero graph run over a
  frontend directory (same file format as ``frontend``) with its own state,
  reset whenever the chunker resets the primary model;
- ``tags``: ``{"name", "tags_dir", "groups", "weight" | "threshold"}``, a
  ``tag_sounds.py`` directory; the listed groups are summed per 40 ms frame,
  then ``min(1, weight * sum)`` or, with ``threshold``, 1 where the sum
  reaches it and 0 elsewhere.

The decode still hears the raw audio (or the ``frontend`` audio with feed
``asr`` or ``both``).

``strategy`` (null or absent: one decode per window) names how each window
is decoded, ``{"mode": ..., ...}``, its keys checked by the mode's parser in
``STRATEGY_PARSERS``. ``island_recover``::

    {"mode": "island_recover", "min_gap_ms": 200,
     "insert": "lexicon",            # or "short" with "max_chars": N
     "max_island_ms": 1500,          # optional; null: any island
     "island_pad_ms": 200,           # optional; null: the chunker's speech_pad_ms
     "asr": {...}, "asr_config": {...}}  # optional faster-whisper island decoder

``context_decode`` decodes each window from a longer span of the item's
audio with the system's own earlier text; its keys are checked by
``context_decode.parse_strategy``. ``incremental_emit`` commits sentences
while a window is still open; its keys are checked by
``incremental.parse_strategy``. ``fw_kwargs`` decodes each window through
production ``transcribe`` with extra faster-whisper keywords; its keys are
checked by ``fw_kwargs.parse_strategy``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from scripts.asr_bench import context_decode, fw_kwargs, incremental
from scripts.asr_bench.sets import bench_root
from scripts.asr_bench.tag_sounds import GROUP_NAMES

BACKENDS = ("faster_whisper", "parakeet_onnx", "sherpa_transducer", "transformers_asr")
# Backends that load a local directory, with the compute types each accepts.
LOCAL_BACKENDS: dict[str, tuple[str, ...]] = {
    "parakeet_onnx": ("float32",),
    "sherpa_transducer": ("float32",),
    "transformers_asr": ("bfloat16", "float16", "float32"),
}
BEAM_BACKENDS = ("faster_whisper", "transformers_asr")
FRONTEND_FEEDS = ("asr", "vad", "both")
# What an island_recover island may insert: a lexicon interjection, or any
# text of at most ``max_chars`` normalized characters.
ISLAND_INSERTS = ("lexicon", "short")

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VadSpec:
    sha256: str
    path: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class AsrSpec:
    backend: str
    model: str
    compute_type: str
    beam_size: int | None = None

    @property
    def local_model_dir(self) -> Path | None:
        """The model directory, or ``None`` for a Hugging Face repo id."""
        path = Path(self.model)
        return path if path.is_absolute() else None


@dataclass(frozen=True)
class FrontendSpec:
    name: str
    audio_dir: str
    feed: str

    @property
    def feeds_vad(self) -> bool:
        return self.feed in ("vad", "both")

    @property
    def feeds_asr(self) -> bool:
        return self.feed in ("asr", "both")

    def audio_path(self, item_id: str) -> Path:
        return Path(self.audio_dir) / f"{item_id}.f32"


class StrategySpec(Protocol):
    """A candidate's window decode strategy: ``mode`` names it in
    ``STRATEGY_PARSERS`` (here) and in the runner's strategy registry."""

    mode: str

    def to_json(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class IslandRecoverSpec:
    """Island recovery (``scripts/asr_bench/island.py``): each window's speech
    islands decoded alone, lexicon interjections or short texts the window's
    own decode lacks inserted into it.

    ``asr`` (a faster-whisper model) and ``asr_config`` give the islands a
    decoder of their own, configured from the shipped ``asr`` section; with
    neither, the islands go through the candidate's own decoder.
    """

    mode: str
    min_gap_ms: int
    insert: str = "lexicon"
    max_chars: int | None = None
    max_island_ms: int | None = None
    # None: the chunker's speech_pad_ms.
    island_pad_ms: int | None = None
    asr: AsrSpec | None = None
    asr_config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def asr_overrides(self) -> dict[str, Any]:
        """Keys to override in the shipped ``asr`` section for the island decoder."""
        overrides = copy.deepcopy(dict(self.asr_config))
        if self.asr is not None and self.asr.beam_size is not None:
            overrides.setdefault("faster_whisper", {})["beam_size"] = self.asr.beam_size
        return overrides

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "min_gap_ms": self.min_gap_ms,
            "insert": self.insert,
            "max_chars": self.max_chars,
            "max_island_ms": self.max_island_ms,
            "island_pad_ms": self.island_pad_ms,
        }
        if self.asr is not None:
            payload["asr"] = _asr_json(self.asr)
        if self.asr_config:
            payload["asr_config"] = copy.deepcopy(dict(self.asr_config))
        return payload


@dataclass(frozen=True)
class AssistSileroSpec:
    """A second Silero track over precomputed frontend audio."""

    name: str
    audio_dir: str

    def audio_path(self, item_id: str) -> Path:
        return Path(self.audio_dir) / f"{item_id}.f32"


@dataclass(frozen=True)
class AssistTagsSpec:
    """A precomputed ``tag_sounds.py`` track: the selected groups summed,
    then scaled by ``weight`` (clipped to 1) or, with ``threshold``, 1 where
    the sum reaches it and 0 elsewhere."""

    name: str
    tags_dir: str
    groups: tuple[str, ...]
    weight: float | None = None
    threshold: float | None = None

    @property
    def rule(self) -> str:
        return "threshold" if self.threshold is not None else "weight"

    def tags_path(self, item_id: str) -> Path:
        return Path(self.tags_dir) / f"{item_id}.npz"


@dataclass(frozen=True)
class VadAssistSpec:
    silero: AssistSileroSpec | None = None
    tags: AssistTagsSpec | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.silero is not None:
            payload["silero"] = {"name": self.silero.name, "audio_dir": self.silero.audio_dir}
        if self.tags is not None:
            tags: dict[str, Any] = {"name": self.tags.name, "tags_dir": self.tags.tags_dir,
                                    "groups": list(self.tags.groups)}
            if self.tags.weight is not None:
                tags["weight"] = self.tags.weight
            if self.tags.threshold is not None:
                tags["threshold"] = self.tags.threshold
            payload["tags"] = tags
        return payload


@dataclass(frozen=True)
class Candidate:
    name: str
    vad: VadSpec | None
    asr: AsrSpec
    vad_threshold: float | None = None
    chunker: Mapping[str, Any] = field(default_factory=dict)
    asr_config: Mapping[str, Any] = field(default_factory=dict)
    frontend: FrontendSpec | None = None
    vad_assist: VadAssistSpec | None = None
    strategy: StrategySpec | None = None

    @property
    def vad_overrides(self) -> dict[str, Any]:
        """Keys to override in the shipped ``vad`` section."""
        overrides = dict(self.chunker)
        if self.vad_threshold is not None:
            overrides["threshold"] = self.vad_threshold
        return overrides

    @property
    def asr_overrides(self) -> dict[str, Any]:
        """Keys to override in the shipped ``asr`` section, nested like it."""
        overrides = copy.deepcopy(dict(self.asr_config))
        if self.asr.beam_size is not None and self.asr.backend == "faster_whisper":
            overrides.setdefault("faster_whisper", {})["beam_size"] = self.asr.beam_size
        return overrides

    def to_json(self) -> dict[str, Any]:
        vad = None
        if self.vad is not None:
            vad = {"path": self.vad.path, "url": self.vad.url, "sha256": self.vad.sha256}
        asr = _asr_json(self.asr)
        payload: dict[str, Any] = {"name": self.name, "vad": vad, "asr": asr}
        if self.vad_threshold is not None:
            payload["vad_threshold"] = self.vad_threshold
        if self.chunker:
            payload["chunker"] = dict(self.chunker)
        if self.asr_config:
            payload["asr_config"] = copy.deepcopy(dict(self.asr_config))
        if self.frontend is not None:
            payload["frontend"] = {
                "name": self.frontend.name,
                "audio_dir": self.frontend.audio_dir,
                "feed": self.frontend.feed,
            }
        if self.vad_assist is not None:
            payload["vad_assist"] = self.vad_assist.to_json()
        if self.strategy is not None:
            payload["strategy"] = self.strategy.to_json()
        return payload


def _asr_json(spec: AsrSpec) -> dict[str, Any]:
    asr: dict[str, Any] = {
        "backend": spec.backend,
        "model": spec.model,
        "compute_type": spec.compute_type,
    }
    if spec.beam_size is not None:
        asr["beam_size"] = spec.beam_size
    return asr


def shipped_config() -> dict[str, Any]:
    """Shipped ``default_config.yaml``, read only; a user's config is never read."""
    from yamabiko import paths

    return yaml.safe_load(paths.shipped_defaults_path().read_text(encoding="utf-8"))


def _optional_str(raw: Mapping[str, Any], key: str, label: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string or null")
    return value


def _vad(raw: object, label: str) -> VadSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}.vad must be an object or null")
    unknown = set(raw) - {"path", "url", "sha256"}
    if unknown:
        raise ValueError(f"{label}.vad has unknown keys {sorted(unknown)}")
    path = _optional_str(raw, "path", f"{label}.vad")
    url = _optional_str(raw, "url", f"{label}.vad")
    sha256 = _optional_str(raw, "sha256", f"{label}.vad")
    if path is None and url is None and sha256 is None:
        return None
    if sha256 is None or not _SHA256_RE.fullmatch(sha256):
        raise ValueError(f"{label}.vad.sha256 must be a lowercase SHA-256 hex digest")
    if (path is None) == (url is None):
        raise ValueError(f"{label}.vad needs exactly one of path or url")
    if path is not None and not Path(path).is_absolute():
        raise ValueError(f"{label}.vad.path must be absolute")
    if url is not None and not url.startswith("https://"):
        raise ValueError(f"{label}.vad.url must be an https URL")
    return VadSpec(sha256=sha256, path=path, url=url)


def _asr(raw: object, label: str) -> AsrSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}.asr must be an object")
    unknown = set(raw) - {"backend", "model", "compute_type", "beam_size"}
    if unknown:
        raise ValueError(f"{label}.asr has unknown keys {sorted(unknown)}")
    backend = raw.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"{label}.asr.backend must be one of {BACKENDS}")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{label}.asr.model must be a non-empty string")
    compute_type = raw.get("compute_type")
    if not isinstance(compute_type, str) or not compute_type.strip():
        raise ValueError(f"{label}.asr.compute_type must be a non-empty string")
    beam_size = raw.get("beam_size")
    if beam_size is not None and (
        isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1
    ):
        raise ValueError(f"{label}.asr.beam_size must be a positive int")
    spec = AsrSpec(backend, model, compute_type, beam_size)
    if backend in LOCAL_BACKENDS:
        if spec.local_model_dir is None:
            raise ValueError(f"{label}.asr.model must be an absolute local model directory")
        allowed = LOCAL_BACKENDS[backend]
        if compute_type not in allowed:
            raise ValueError(f"{label}.asr.compute_type must be one of {allowed} for {backend}")
    if beam_size is not None and backend not in BEAM_BACKENDS:
        raise ValueError(f"{label}.asr.beam_size applies to {' and '.join(BEAM_BACKENDS)} only")
    return spec


def _threshold(value: float, label: str) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{label} must lie in (0, 1)")
    return value


def _section_overrides(raw: object, shipped: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Validate overrides against the keys and value types of a shipped
    config section: a bool stays a bool, an int an int, a float takes any
    finite number, a string stays a string and a nested section takes
    overrides of its own keys."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object")
    unknown = set(raw) - set(shipped)
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)} (known: {sorted(shipped)})")
    overrides: dict[str, Any] = {}
    for key, value in raw.items():
        default = shipped[key]
        label = f"{where}.{key}"
        if isinstance(default, Mapping):
            value = _section_overrides(value, default, label)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a bool")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an int")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        elif not isinstance(value, type(default)):
            raise ValueError(f"{label} must be a {type(default).__name__}")
        overrides[key] = value
    return overrides


def _chunker(raw: object, label: str) -> dict[str, Any]:
    """Overrides of the shipped ``vad`` section."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}.chunker must be an object or null")
    overrides = _section_overrides(raw, shipped_config()["vad"], f"{label}.chunker")
    if "threshold" in overrides:
        _threshold(overrides["threshold"], f"{label}.chunker.threshold")
    return overrides


def _asr_config(raw: object, label: str) -> dict[str, Any]:
    """Overrides of the shipped ``asr`` section, nested like it."""
    if raw is None:
        return {}
    where = f"{label}.asr_config"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object or null")
    overrides = _section_overrides(raw, shipped_config()["asr"], where)
    if overrides.get("boundary_pad_ms", 0) < 0:
        raise ValueError(f"{where}.boundary_pad_ms must be nonnegative")
    whisper = overrides.get("faster_whisper", {})
    for key in ("model", "compute_type"):
        if key in whisper:
            raise ValueError(f"{where}.faster_whisper.{key} is set by asr.{key}")
    if whisper.get("beam_size", 1) < 1:
        raise ValueError(f"{where}.faster_whisper.beam_size must be a positive int")
    if "device" in whisper and not whisper["device"].strip():
        raise ValueError(f"{where}.faster_whisper.device must be a non-empty string")
    threshold = whisper.get("no_speech_only_threshold")
    if threshold is not None and not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"{where}.faster_whisper.no_speech_only_threshold must lie in (0, 1]; 1.0 turns it off"
        )
    return overrides


def _frontend(raw: object, label: str) -> FrontendSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label}.frontend must be an object or null")
    unknown = set(raw) - {"name", "audio_dir", "feed"}
    if unknown:
        raise ValueError(f"{label}.frontend has unknown keys {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"{label}.frontend.name must match [A-Za-z0-9][A-Za-z0-9._-]*")
    audio_dir = raw.get("audio_dir")
    if not isinstance(audio_dir, str) or not Path(audio_dir).is_absolute():
        raise ValueError(f"{label}.frontend.audio_dir must be an absolute directory")
    feed = raw.get("feed")
    if feed not in FRONTEND_FEEDS:
        raise ValueError(f"{label}.frontend.feed must be one of {FRONTEND_FEEDS}")
    return FrontendSpec(name=name, audio_dir=audio_dir, feed=feed)


def _assist_name_dir(raw: Mapping[str, Any], dir_key: str, where: str) -> tuple[str, str]:
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"{where}.name must match [A-Za-z0-9][A-Za-z0-9._-]*")
    directory = raw.get(dir_key)
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise ValueError(f"{where}.{dir_key} must be an absolute directory")
    return name, directory


def _assist_tags(raw: object, where: str) -> AssistTagsSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object")
    unknown = set(raw) - {"name", "tags_dir", "groups", "weight", "threshold"}
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    name, tags_dir = _assist_name_dir(raw, "tags_dir", where)
    groups = raw.get("groups")
    if (not isinstance(groups, list) or not groups
            or any(group not in GROUP_NAMES for group in groups) or len(set(groups)) != len(groups)):
        raise ValueError(f"{where}.groups must be a non-empty list of distinct groups from "
                         f"{GROUP_NAMES}")
    weight, threshold = raw.get("weight"), raw.get("threshold")
    if (weight is None) == (threshold is None):
        raise ValueError(f"{where} needs exactly one of weight or threshold")
    for key, value in (("weight", weight), ("threshold", threshold)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{where}.{key} must be a number")
    if weight is not None:
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{where}.weight must be a positive finite number")
    if threshold is not None:
        threshold = _threshold(float(threshold), f"{where}.threshold")
    return AssistTagsSpec(name=name, tags_dir=tags_dir, groups=tuple(groups),
                          weight=weight, threshold=threshold)


def _vad_assist(raw: object, label: str) -> VadAssistSpec | None:
    if raw is None:
        return None
    where = f"{label}.vad_assist"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object or null")
    unknown = set(raw) - {"silero", "tags"}
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    silero = None
    if raw.get("silero") is not None:
        entry = raw["silero"]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}.silero must be an object or null")
        unknown = set(entry) - {"name", "audio_dir"}
        if unknown:
            raise ValueError(f"{where}.silero has unknown keys {sorted(unknown)}")
        silero = AssistSileroSpec(*_assist_name_dir(entry, "audio_dir", f"{where}.silero"))
    tags = None if raw.get("tags") is None else _assist_tags(raw["tags"], f"{where}.tags")
    if silero is None and tags is None:
        raise ValueError(f"{where} needs silero, tags or both")
    return VadAssistSpec(silero=silero, tags=tags)


def _optional_int(raw: Mapping[str, Any], key: str, where: str, minimum: int) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where}.{key} must be an int of at least {minimum}")
    return value


def _island_recover(raw: Mapping[str, Any], where: str) -> IslandRecoverSpec:
    unknown = set(raw) - {
        "mode", "min_gap_ms", "insert", "max_chars", "max_island_ms", "island_pad_ms",
        "asr", "asr_config",
    }
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    mode = raw["mode"]
    min_gap_ms = _optional_int(raw, "min_gap_ms", where, 1)
    if min_gap_ms is None:
        raise ValueError(f"{where}.min_gap_ms is required")
    insert = raw.get("insert", "lexicon")
    if insert not in ISLAND_INSERTS:
        raise ValueError(f"{where}.insert must be one of {ISLAND_INSERTS}")
    max_chars = _optional_int(raw, "max_chars", where, 1)
    if (insert == "short") != (max_chars is not None):
        raise ValueError(f"{where}.max_chars is required with insert 'short' and only then")
    asr = None
    if raw.get("asr") is not None:
        asr = _asr(raw["asr"], where)
        if asr.backend != "faster_whisper":
            raise ValueError(f"{where}.asr.backend must be faster_whisper")
    asr_config = _asr_config(raw.get("asr_config"), where)
    if asr_config and asr is None:
        raise ValueError(f"{where}.asr_config needs {where}.asr")
    if asr is not None and asr.beam_size is not None and "beam_size" in asr_config.get(
        "faster_whisper", {}
    ):
        raise ValueError(
            f"{where}: give asr.beam_size or asr_config.faster_whisper.beam_size, not both"
        )
    return IslandRecoverSpec(
        mode=mode,
        min_gap_ms=min_gap_ms,
        insert=insert,
        max_chars=max_chars,
        max_island_ms=_optional_int(raw, "max_island_ms", where, 1),
        island_pad_ms=_optional_int(raw, "island_pad_ms", where, 0),
        asr=asr,
        asr_config=asr_config,
    )


# Strategy mode -> parser of its candidate object (``where`` labels errors).
STRATEGY_PARSERS: dict[str, Callable[[Mapping[str, Any], str], StrategySpec]] = {
    "island_recover": _island_recover,
    context_decode.MODE: context_decode.parse_strategy,
    incremental.MODE: incremental.parse_strategy,
    fw_kwargs.MODE: fw_kwargs.parse_strategy,
}


def _strategy(raw: object, label: str) -> StrategySpec | None:
    if raw is None:
        return None
    where = f"{label}.strategy"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object or null")
    parser = STRATEGY_PARSERS.get(raw.get("mode"))
    if parser is None:
        raise ValueError(f"{where}.mode must be one of {sorted(STRATEGY_PARSERS)}")
    return parser(raw, where)


def parse_candidate(raw: object) -> Candidate:
    if not isinstance(raw, Mapping):
        raise ValueError("a candidate must be a JSON object")
    unknown = set(raw) - {
        "name", "vad", "asr", "vad_threshold", "chunker", "asr_config", "frontend", "strategy",
        "vad_assist",
    }
    if unknown:
        raise ValueError(f"candidate has unknown keys {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError("candidate name must match [A-Za-z0-9][A-Za-z0-9._-]*")
    threshold = raw.get("vad_threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"{name}.vad_threshold must be a number")
        threshold = _threshold(float(threshold), f"{name}.vad_threshold")
    chunker = _chunker(raw.get("chunker"), name)
    if threshold is not None and "threshold" in chunker:
        raise ValueError(f"{name}: give vad_threshold or chunker.threshold, not both")
    asr = _asr(raw.get("asr"), name)
    asr_config = _asr_config(raw.get("asr_config"), name)
    if "faster_whisper" in asr_config:
        if asr.backend != "faster_whisper":
            raise ValueError(f"{name}.asr_config.faster_whisper applies to faster_whisper only")
        if asr.beam_size is not None and "beam_size" in asr_config["faster_whisper"]:
            raise ValueError(
                f"{name}: give asr.beam_size or asr_config.faster_whisper.beam_size, not both"
            )
    strategy = _strategy(raw.get("strategy"), name)
    whisper_overrides = asr_config.get("faster_whisper", {})
    no_speech_only = whisper_overrides.get(
        "no_speech_only_threshold",
        shipped_config()["asr"]["faster_whisper"]["no_speech_only_threshold"],
    )
    if (no_speech_only is not None and no_speech_only < 1.0
            and strategy is not None and strategy.mode == context_decode.MODE):
        raise ValueError(
            f"{name}: faster_whisper.no_speech_only_threshold cannot apply to strategy "
            f"{context_decode.MODE!r}: it decodes through its own SpanDecoder, which the "
            "wrapper's gate never sees; set it to 1.0 (off) for this candidate"
        )
    return Candidate(
        name=name,
        vad=_vad(raw.get("vad"), name),
        asr=asr,
        vad_threshold=threshold,
        chunker=chunker,
        asr_config=asr_config,
        frontend=_frontend(raw.get("frontend"), name),
        vad_assist=_vad_assist(raw.get("vad_assist"), name),
        strategy=strategy,
    )


def load_candidate(path: Path, name: str | None = None) -> Candidate:
    """Load one candidate; ``name`` selects an entry of a ``candidates`` list."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "candidates" in raw:
        entries = raw["candidates"]
        if not isinstance(entries, list):
            raise ValueError("candidates must be a list")
        candidates = [parse_candidate(entry) for entry in entries]
        names = [candidate.name for candidate in candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        if name is None:
            raise ValueError(f"{path} lists several candidates; pass --name ({names})")
        for candidate in candidates:
            if candidate.name == name:
                return candidate
        raise ValueError(f"no candidate named {name!r} in {path} ({names})")
    candidate = parse_candidate(raw)
    if name is not None and candidate.name != name:
        raise ValueError(f"{path} holds candidate {candidate.name!r}, not {name!r}")
    return candidate


def _download(url: str, target: Path, sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}")
    os.replace(temporary, target)


def resolve_vad(spec: VadSpec | None):
    """Return the hashed VAD artifact (``scripts.e0_evidence.replay.VadArtifact``).

    ``None`` resolves production's pinned file exactly as production does.
    A URL is fetched once into ``bench_root()/cache/vad/<sha256>/``.
    """
    from scripts.e0_evidence.replay import resolve_vad_artifact

    if spec is None:
        return resolve_vad_artifact()
    if spec.path is not None:
        path = Path(spec.path)
    else:
        assert spec.url is not None
        filename = spec.url.rstrip("/").rsplit("/", 1)[-1] or "model.onnx"
        path = bench_root() / "cache" / "vad" / spec.sha256 / filename
        if not path.is_file():
            _download(spec.url, path, spec.sha256)
    return resolve_vad_artifact(path, spec.sha256)


def model_artifact_sha256(model_dir: Path) -> str:
    """Digest of a local model directory, computed as the E0 harness names it."""
    from scripts.e0_evidence.runtime import _artifact_manifest
    from scripts.e0_evidence.wire import canonical_sha256

    return canonical_sha256(_artifact_manifest(model_dir))
