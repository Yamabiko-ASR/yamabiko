"""Strict loading contracts for private E0 PCM masters and their replay stream."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import soxr

from scripts.e0_evidence._common import (
    file_sha256_and_size,
    require_sha256,
    require_uuid4,
)
from scripts.e0_evidence.assets import (
    resolve_fixture_source,
    resolve_private_locator_asset,
)
from scripts.e0_evidence.private_root import PrivateBoundary
from scripts.e0_evidence.schema import (
    SCHEMA_VERSION,
    JsonValue,
    Partition,
    RightsRecord,
    freeze_json_mapping,
)
from scripts.e0_evidence.wire import canonical_sha256
from yamabiko.capture.resampler import StreamingResampler


def _nonempty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _opaque_ids(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be UUIDv4 values")
    result = tuple(values)  # type: ignore[arg-type]
    for index, value in enumerate(result):
        require_uuid4(value, f"{field_name}[{index}]")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result  # type: ignore[return-value]


def _strings(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be strings")
    result = tuple(values)  # type: ignore[arg-type]
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    for index, value in enumerate(result):
        _nonempty(value, f"{field_name}[{index}]")
    return result  # type: ignore[return-value]


def _artifact_sha256_and_size(path: Path, artifact_name: str) -> tuple[str, int]:
    try:
        return file_sha256_and_size(path)
    except OSError:
        raise ValueError(f"{artifact_name} is unreadable") from None


@dataclass(frozen=True, slots=True)
class TransformStep:
    transform_id: str
    transform_version: str
    transform_sha256: str
    parent_artifact_id: str
    parent_byte_sha256: str
    output_artifact_id: str
    output_byte_sha256: str
    tool: str
    tool_version: str
    command_argv: tuple[str, ...]
    filter_graph: str
    stream_index: int
    excerpt_start_samples: int
    excerpt_end_samples: int
    input_sample_rate_hz: int
    input_channels: int
    input_layout: str
    output_sample_rate_hz: int
    output_channels: int
    output_layout: str
    output_sample_count: int
    channel_operation: str
    resampler: str
    resampler_parameters: Mapping[str, JsonValue]
    dtype_conversion: str

    def __post_init__(self) -> None:
        require_uuid4(self.transform_id, "transform_id")
        _nonempty(self.transform_version, "transform_version")
        require_sha256(self.transform_sha256, "transform_sha256")
        require_uuid4(self.parent_artifact_id, "parent_artifact_id")
        require_sha256(self.parent_byte_sha256, "parent_byte_sha256")
        require_uuid4(self.output_artifact_id, "output_artifact_id")
        require_sha256(self.output_byte_sha256, "output_byte_sha256")
        _nonempty(self.tool, "tool")
        _nonempty(self.tool_version, "tool_version")
        object.__setattr__(
            self,
            "command_argv",
            _strings(self.command_argv, "command_argv"),
        )
        _nonempty(self.filter_graph, "filter_graph")
        _integer(self.stream_index, "stream_index")
        start = _integer(self.excerpt_start_samples, "excerpt_start_samples")
        end = _integer(self.excerpt_end_samples, "excerpt_end_samples", minimum=1)
        if end <= start:
            raise ValueError("excerpt bounds must form a non-empty half-open range")
        _integer(self.input_sample_rate_hz, "input_sample_rate_hz", minimum=1)
        _integer(self.input_channels, "input_channels", minimum=1)
        _nonempty(self.input_layout, "input_layout")
        _integer(self.output_sample_rate_hz, "output_sample_rate_hz", minimum=1)
        _integer(self.output_channels, "output_channels", minimum=1)
        _nonempty(self.output_layout, "output_layout")
        _integer(self.output_sample_count, "output_sample_count", minimum=1)
        _nonempty(self.channel_operation, "channel_operation")
        _nonempty(self.resampler, "resampler")
        object.__setattr__(
            self,
            "resampler_parameters",
            freeze_json_mapping(
                self.resampler_parameters,
                field_name="resampler_parameters",
            ),
        )
        _nonempty(self.dtype_conversion, "dtype_conversion")


def transform_step_sha256(step: TransformStep) -> str:
    if not isinstance(step, TransformStep):
        raise TypeError("step must be TransformStep")
    payload = cast(
        Mapping[str, JsonValue],
        {
            "channel_operation": step.channel_operation,
            "command_argv": list(step.command_argv),
            "dtype_conversion": step.dtype_conversion,
            "excerpt_end_samples": step.excerpt_end_samples,
            "excerpt_start_samples": step.excerpt_start_samples,
            "filter_graph": step.filter_graph,
            "input_channels": step.input_channels,
            "input_layout": step.input_layout,
            "input_sample_rate_hz": step.input_sample_rate_hz,
            "output_artifact_id": step.output_artifact_id,
            "output_byte_sha256": step.output_byte_sha256,
            "output_channels": step.output_channels,
            "output_layout": step.output_layout,
            "output_sample_count": step.output_sample_count,
            "output_sample_rate_hz": step.output_sample_rate_hz,
            "parent_artifact_id": step.parent_artifact_id,
            "parent_byte_sha256": step.parent_byte_sha256,
            "resampler": step.resampler,
            "resampler_parameters": step.resampler_parameters,
            "stream_index": step.stream_index,
            "tool": step.tool,
            "tool_version": step.tool_version,
            "transform_id": step.transform_id,
            "transform_version": step.transform_version,
        },
    )
    return canonical_sha256(payload)


def _validate_transform_semantics(
    source_native: TransformStep,
    canonical: TransformStep,
) -> None:
    if source_native.transform_version not in {
        "ffmpeg-decode-1.0",
        "ffmpeg-decode-1.1",
    }:
        raise ValueError("source-native transform version is not supported")
    if canonical.transform_version != "canonical-pcm-1.0":
        raise ValueError("canonical transform version is not supported")
    for step_name, step in (
        ("source-native", source_native),
        ("canonical", canonical),
    ):
        if step.tool != "ffmpeg":
            raise ValueError(f"{step_name} transform tool must be ffmpeg")
        if not step.tool_version.startswith("ffmpeg version "):
            raise ValueError(f"{step_name} tool version is not an FFmpeg version")
        if step.command_argv[0] != "ffmpeg":
            raise ValueError(f"{step_name} command must invoke ffmpeg")

    expected_channels = {"mono": 1, "stereo": 2}
    if source_native.input_layout not in expected_channels:
        raise ValueError("source-native input layout is not approved")
    if (
        source_native.input_channels
        != expected_channels[source_native.input_layout]
    ):
        raise ValueError("source-native input channels do not match layout")
    if (
        source_native.output_sample_rate_hz
        != source_native.input_sample_rate_hz
        or source_native.output_channels != source_native.input_channels
        or source_native.output_layout != source_native.input_layout
    ):
        raise ValueError("source-native output must preserve rate and channels")
    if source_native.output_sample_count != (
        source_native.excerpt_end_samples - source_native.excerpt_start_samples
    ):
        raise ValueError("source-native sample count does not match excerpt")
    if source_native.channel_operation != "preserve":
        raise ValueError("source-native channel operation must preserve channels")
    if source_native.resampler != "none":
        raise ValueError("source-native transform must not resample")
    if source_native.resampler_parameters:
        raise ValueError("source-native resampler parameters must be empty")
    if source_native.dtype_conversion != "little-endian-float32":
        raise ValueError("source-native output must be little-endian float32")
    expected_source_filter = (
        "anull"
        if source_native.transform_version == "ffmpeg-decode-1.0"
        else (
            "atrim="
            f"start_sample={source_native.excerpt_start_samples}:"
            f"end_sample={source_native.excerpt_end_samples},"
            "asetpts=N/SR/TB"
        )
    )
    if source_native.filter_graph != expected_source_filter:
        if source_native.transform_version == "ffmpeg-decode-1.1":
            raise ValueError(
                "source-native v1.1 filter graph must use the sample-exact trim"
            )
        raise ValueError("source-native filter graph must preserve the stream")
    if source_native.transform_version == "ffmpeg-decode-1.1":
        command = source_native.command_argv
        filter_indexes = tuple(
            index for index, value in enumerate(command) if value == "-af"
        )
        if (
            len(filter_indexes) != 1
            or filter_indexes[0] + 1 >= len(command)
            or command[filter_indexes[0] + 1] != expected_source_filter
            or any(
                option in command
                for option in ("-ss", "-sseof", "-t", "-to")
            )
        ):
            raise ValueError(
                "source-native v1.1 command must execute the sample-exact trim"
            )

    if (
        canonical.input_sample_rate_hz != source_native.output_sample_rate_hz
        or canonical.input_channels != source_native.output_channels
        or canonical.input_layout != source_native.output_layout
    ):
        raise ValueError("canonical input format does not match source-native output")
    if canonical.stream_index != 0:
        raise ValueError("canonical raw PCM stream index must be zero")
    if (
        canonical.excerpt_start_samples != 0
        or canonical.excerpt_end_samples != source_native.output_sample_count
    ):
        raise ValueError("canonical excerpt must consume the source-native master")
    if (
        canonical.output_sample_rate_hz != 16_000
        or canonical.output_channels != 1
        or canonical.output_layout != "mono"
    ):
        raise ValueError("canonical output format must be mono 16000 Hz")

    expected_channel_operation = (
        "mono-pass-through"
        if canonical.input_layout == "mono"
        else "stereo-0.5l-plus-0.5r"
    )
    if canonical.channel_operation != expected_channel_operation:
        raise ValueError("canonical channel operation does not match input layout")
    parameters = canonical.resampler_parameters
    channel_filter = (
        None
        if canonical.input_layout == "mono"
        else "pan=mono|c0=0.5*c0+0.5*c1"
    )
    if canonical.input_sample_rate_hz == 16_000:
        if canonical.resampler != "none" or parameters:
            raise ValueError("canonical 16000 Hz input must not be resampled")
        expected_filters = [channel_filter] if channel_filter is not None else []
    else:
        if canonical.resampler != "soxr":
            raise ValueError("canonical resampler must be soxr")
        expected_parameters: Mapping[str, JsonValue] = {
            "async": 0,
            "cheby": 0,
            "cutoff": 0.91,
            "isr": canonical.input_sample_rate_hz,
            "osr": 16_000,
            "precision": 28,
        }
        if parameters != expected_parameters:
            raise ValueError("canonical soxr parameters do not match policy")
        resample_filter = (
            "aresample="
            f"isr={canonical.input_sample_rate_hz}:"
            "osr=16000:resampler=soxr:precision=28:"
            "cutoff=0.91:cheby=0:async=0"
        )
        expected_filters = (
            [channel_filter, resample_filter]
            if channel_filter is not None
            else [resample_filter]
        )
    expected_filter_graph = ",".join(expected_filters) or "anull"
    if canonical.filter_graph != expected_filter_graph:
        raise ValueError("canonical filter graph does not match transform metadata")
    if canonical.dtype_conversion != "little-endian-float32":
        raise ValueError("canonical dtype conversion must be little-endian float32")


@dataclass(frozen=True, slots=True)
class MasterTransformDag:
    source_native: TransformStep
    canonical: TransformStep

    def __post_init__(self) -> None:
        if not isinstance(self.source_native, TransformStep):
            raise TypeError("source_native must be TransformStep")
        if not isinstance(self.canonical, TransformStep):
            raise TypeError("canonical must be TransformStep")


@dataclass(frozen=True, slots=True)
class MasterManifest:
    schema_version: str
    master_id: str
    fixture_id: str
    rights: RightsRecord
    partition: Partition
    timeline_id: str
    parent_artifact_ids: tuple[str, ...]
    byte_sha256: str
    canonical_pcm_sha256: str
    sample_rate_hz: int
    channels: int
    layout: str
    dtype: str
    sample_count: int
    duration_samples: int
    transform_dag: MasterTransformDag
    provenance_class: str
    rights_class: str
    retention_version: str
    annotation_set_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        require_uuid4(self.master_id, "master_id")
        require_uuid4(self.fixture_id, "fixture_id")
        if not isinstance(self.rights, RightsRecord):
            raise TypeError("rights must be RightsRecord")
        if self.rights.fixture_id != self.fixture_id:
            raise ValueError("rights record belongs to a different fixture")
        if not isinstance(self.partition, Partition):
            raise TypeError("partition must be Partition")
        require_uuid4(self.timeline_id, "timeline_id")
        parents = _opaque_ids(self.parent_artifact_ids, "parent_artifact_ids")
        if not parents:
            raise ValueError("parent_artifact_ids must not be empty")
        object.__setattr__(self, "parent_artifact_ids", parents)
        require_sha256(self.byte_sha256, "byte_sha256")
        require_sha256(self.canonical_pcm_sha256, "canonical_pcm_sha256")
        _integer(self.sample_rate_hz, "sample_rate_hz", minimum=1)
        _integer(self.channels, "channels", minimum=1)
        _nonempty(self.layout, "layout")
        _nonempty(self.dtype, "dtype")
        _integer(self.sample_count, "sample_count", minimum=1)
        _integer(self.duration_samples, "duration_samples", minimum=1)
        if not isinstance(self.transform_dag, MasterTransformDag):
            raise TypeError("transform_dag must be MasterTransformDag")
        _nonempty(self.provenance_class, "provenance_class")
        _nonempty(self.rights_class, "rights_class")
        _nonempty(self.retention_version, "retention_version")
        if self.retention_version != self.rights.retention_version:
            raise ValueError("master retention version does not match rights record")
        require_sha256(self.annotation_set_sha256, "annotation_set_sha256")


@dataclass(frozen=True, slots=True)
class CanonicalMaster:
    manifest: MasterManifest
    samples: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, MasterManifest):
            raise TypeError("manifest must be MasterManifest")
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a NumPy array")
        if self.samples.ndim != 1:
            raise ValueError("canonical samples must be one-dimensional")
        if self.samples.flags.writeable:
            raise ValueError("canonical samples must be immutable")


def load_canonical_master(
    manifest: MasterManifest,
    boundary: PrivateBoundary,
) -> CanonicalMaster:
    if not isinstance(manifest, MasterManifest):
        raise TypeError("manifest must be MasterManifest")
    if not isinstance(boundary, PrivateBoundary):
        raise TypeError("boundary must be PrivateBoundary")

    source_path = resolve_fixture_source(
        manifest.fixture_id,
        manifest.rights,
        boundary,
    )
    source_sha256, _ = _artifact_sha256_and_size(
        source_path,
        f"source artifact for fixture {manifest.fixture_id}",
    )
    source_native = manifest.transform_dag.source_native
    canonical = manifest.transform_dag.canonical
    if source_sha256 != source_native.parent_byte_sha256:
        raise ValueError("source parent byte SHA-256 does not match transform")
    for step_name, step in (
        ("source-native", source_native),
        ("canonical", canonical),
    ):
        if transform_step_sha256(step) != step.transform_sha256:
            raise ValueError(f"{step_name} transform SHA-256 does not match")
    if source_native.transform_id == canonical.transform_id:
        raise ValueError("transform DAG IDs must be distinct")
    artifact_ids = (
        source_native.parent_artifact_id,
        source_native.output_artifact_id,
        canonical.output_artifact_id,
    )
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("transform DAG artifact IDs must be distinct and acyclic")
    _validate_transform_semantics(source_native, canonical)
    if (
        canonical.parent_artifact_id != source_native.output_artifact_id
        or canonical.parent_byte_sha256 != source_native.output_byte_sha256
    ):
        raise ValueError(
            "canonical transform parent does not match source-native output"
        )
    if manifest.parent_artifact_ids != (canonical.parent_artifact_id,):
        raise ValueError("master parent artifacts do not match canonical transform")
    if canonical.output_artifact_id != manifest.master_id:
        raise ValueError("canonical transform output does not match master_id")

    source_native_path = resolve_private_locator_asset(
        manifest.fixture_id,
        manifest.rights,
        "masters",
        source_native.output_artifact_id,
        boundary,
    )
    source_native_sha256, source_native_size = _artifact_sha256_and_size(
        source_native_path,
        f"source-native master {source_native.output_artifact_id}",
    )
    if source_native_sha256 != source_native.output_byte_sha256:
        raise ValueError("source-native output hash does not match bytes")
    if source_native_sha256 != canonical.parent_byte_sha256:
        raise ValueError("canonical parent hash does not match source-native bytes")
    expected_source_native_size = (
        source_native.output_sample_count
        * source_native.output_channels
        * 4
    )
    if source_native_size != expected_source_native_size:
        raise ValueError("source-native byte count does not match format metadata")

    path = resolve_private_locator_asset(
        manifest.fixture_id,
        manifest.rights,
        "masters",
        manifest.master_id,
        boundary,
    )
    raw: bytes | None = None
    try:
        raw = path.read_bytes()
    except OSError:
        pass
    if raw is None:
        raise ValueError(
            f"canonical master {manifest.master_id} is unreadable"
        ) from None
    if len(raw) % 4 != 0:
        raise ValueError("canonical PCM byte count must be divisible by four")

    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != manifest.byte_sha256:
        raise ValueError("canonical master byte SHA-256 does not match manifest")
    if canonical.output_byte_sha256 != raw_sha256:
        raise ValueError("canonical transform output hash does not match bytes")

    if manifest.sample_rate_hz != 16_000:
        raise ValueError("canonical master sample rate must be 16000 Hz")
    if manifest.channels != 1:
        raise ValueError("canonical master must have one channel")
    if manifest.layout != "mono":
        raise ValueError("canonical master layout must be mono")
    if manifest.dtype != "<f4":
        raise ValueError("canonical master dtype must be little-endian float32")
    if (
        canonical.output_sample_rate_hz != manifest.sample_rate_hz
        or canonical.output_channels != manifest.channels
        or canonical.output_layout != manifest.layout
    ):
        raise ValueError("canonical transform output does not match master format")
    if canonical.output_sample_count != manifest.sample_count:
        raise ValueError("canonical transform sample count does not match master")

    samples = np.frombuffer(raw, dtype="<f4")
    if samples.flags.writeable:
        raise ValueError(
            "canonical PCM buffer unexpectedly produced writable samples"
        )
    if samples.size != manifest.sample_count:
        raise ValueError("canonical master sample count does not match manifest")
    if manifest.duration_samples != manifest.sample_count:
        raise ValueError(
            "canonical master duration_samples does not match sample count"
        )
    if not bool(np.isfinite(samples).all()):
        raise ValueError("canonical master contains non-finite samples")

    canonical_sha256 = hashlib.sha256(samples.tobytes(order="C")).hexdigest()
    if canonical_sha256 != manifest.canonical_pcm_sha256:
        raise ValueError("canonical PCM SHA-256 does not match manifest")
    if canonical_sha256 != raw_sha256:
        raise ValueError("canonical PCM hash differs from raw byte hash")
    return CanonicalMaster(manifest=manifest, samples=samples)


CAPTURE_REPLAY_VERSION = "e0-capture-replay-1.0"
_CAPTURE_SAMPLE_RATE_HZ = 16_000
_CAPTURE_ALIGNMENT_METHOD = "cross-correlation"
_CAPTURE_ALIGNMENT_SEARCH_SAMPLES = 64
_CAPTURE_RESAMPLER = "yamabiko.capture.resampler.StreamingResampler"
_CAPTURE_REPLAY_KEYS = frozenset(
    {
        "version",
        "source_native_artifact_id",
        "source_native_byte_sha256",
        "source_native_sample_rate_hz",
        "source_native_channels",
        "downmix",
        "resampler",
        "resampler_quality",
        "soxr_version",
        "stream_end",
        "output_sample_rate_hz",
        "output_sample_count",
        "output_pcm_sha256",
        "timeline_id",
        "canonical_pcm_sha256",
        "alignment",
    }
)
_CAPTURE_ALIGNMENT_KEYS = frozenset(
    {"method", "search_radius_samples", "lag_samples", "zero_lag_correlation"}
)


def capture_replay_provenance_sha256(provenance: Mapping[str, object]) -> str:
    """Validate one recorded production-capture replay identity; return its hash."""

    if not isinstance(provenance, Mapping) or set(provenance) != _CAPTURE_REPLAY_KEYS:
        raise ValueError("capture replay provenance has an invalid shape")
    if provenance["version"] != CAPTURE_REPLAY_VERSION:
        raise ValueError("capture replay provenance version is not supported")
    require_uuid4(provenance["source_native_artifact_id"], "source_native_artifact_id")
    require_uuid4(provenance["timeline_id"], "timeline_id")
    for key in (
        "source_native_byte_sha256",
        "output_pcm_sha256",
        "canonical_pcm_sha256",
    ):
        require_sha256(provenance[key], key)
    rate = _integer(
        provenance["source_native_sample_rate_hz"],
        "source_native_sample_rate_hz",
        minimum=1,
    )
    channels = _integer(
        provenance["source_native_channels"],
        "source_native_channels",
        minimum=1,
    )
    _integer(provenance["output_sample_count"], "output_sample_count", minimum=1)
    if provenance["output_sample_rate_hz"] != _CAPTURE_SAMPLE_RATE_HZ:
        raise ValueError("capture replay output must be 16000 Hz")
    expected_downmix = "mono-pass-through" if channels == 1 else "mean-of-channels"
    if provenance["downmix"] != expected_downmix:
        raise ValueError("capture replay downmix does not match its channel count")
    if rate == _CAPTURE_SAMPLE_RATE_HZ:
        if (
            provenance["resampler"] != "none"
            or provenance["resampler_quality"] is not None
            or provenance["stream_end"] != "none"
        ):
            raise ValueError("16000 Hz capture replay must not resample")
    else:
        _nonempty(provenance["resampler_quality"], "resampler_quality")
        if (
            provenance["resampler"] != _CAPTURE_RESAMPLER
            or provenance["stream_end"] != "finish"
        ):
            raise ValueError("capture replay must use one finished production stream")
    _nonempty(provenance["soxr_version"], "soxr_version")
    alignment = provenance["alignment"]
    if not isinstance(alignment, Mapping) or set(alignment) != _CAPTURE_ALIGNMENT_KEYS:
        raise ValueError("capture replay alignment has an invalid shape")
    if (
        alignment["method"] != _CAPTURE_ALIGNMENT_METHOD
        or alignment["search_radius_samples"] != _CAPTURE_ALIGNMENT_SEARCH_SAMPLES
        or type(alignment["lag_samples"]) is not int
        or alignment["lag_samples"] != 0
    ):
        raise ValueError("capture replay is not aligned with the canonical timeline")
    correlation = alignment["zero_lag_correlation"]
    if correlation is not None and (
        type(correlation) is not float
        or not math.isfinite(correlation)
        or not -1.0 <= correlation <= 1.0
    ):
        raise ValueError("capture replay zero-lag correlation is invalid")
    return canonical_sha256(cast(Mapping[str, JsonValue], provenance))


@dataclass(frozen=True, slots=True)
class CaptureReplayStream:
    """A registered master's source-native PCM after the production capture path.

    Live capture downmixes every WASAPI block with ``mean(axis=1)`` and feeds one
    continuous ``StreamingResampler`` (``yamabiko/capture/audio.py``). P0 and P1
    replay these samples. The canonical master stays the timeline that
    references and annotations are timed against; the stream must share its
    length and show zero lag against it.
    """

    master: CanonicalMaster
    samples: np.ndarray = field(compare=False, repr=False)
    provenance: Mapping[str, JsonValue]
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.master, CanonicalMaster):
            raise TypeError("master must be CanonicalMaster")
        if (
            not isinstance(self.samples, np.ndarray)
            or self.samples.ndim != 1
            or self.samples.dtype != np.dtype("<f4")
            or self.samples.flags.writeable
        ):
            raise ValueError("capture replay samples must be immutable <f4 mono PCM")
        provenance = freeze_json_mapping(self.provenance, field_name="provenance")
        object.__setattr__(self, "provenance", provenance)
        if capture_replay_provenance_sha256(provenance) != self.provenance_sha256:
            raise ValueError("capture replay provenance hash does not match")
        manifest = self.master.manifest
        source_native = manifest.transform_dag.source_native
        if (
            provenance["output_sample_count"] != manifest.sample_count
            or self.samples.size != manifest.sample_count
            or provenance["timeline_id"] != manifest.timeline_id
            or provenance["canonical_pcm_sha256"] != manifest.canonical_pcm_sha256
            or provenance["source_native_artifact_id"]
            != source_native.output_artifact_id
            or provenance["source_native_byte_sha256"]
            != source_native.output_byte_sha256
        ):
            raise ValueError("capture replay provenance does not match its master")
        actual = hashlib.sha256(self.samples.tobytes(order="C")).hexdigest()
        if actual != provenance["output_pcm_sha256"]:
            raise ValueError("capture replay PCM hash does not match provenance")


def _alignment(capture: np.ndarray, canonical: np.ndarray) -> tuple[int, float | None]:
    """Return the capture-to-canonical lag with the largest cross-correlation.

    A positive lag means the capture stream is late. Ties keep the smallest
    absolute lag, so identical or silent streams report zero.
    """

    capture64 = capture.astype(np.float64)
    canonical64 = canonical.astype(np.float64)
    count = capture64.size
    zero_lag = float(np.dot(capture64, canonical64))
    best_lag, best_score = 0, zero_lag
    for magnitude in range(1, min(_CAPTURE_ALIGNMENT_SEARCH_SAMPLES, count - 1) + 1):
        for lag in (magnitude, -magnitude):
            if lag > 0:
                score = float(np.dot(capture64[lag:], canonical64[: count - lag]))
            else:
                score = float(np.dot(capture64[: count + lag], canonical64[-lag:]))
            if score > best_score:
                best_lag, best_score = lag, score
    energy = math.sqrt(
        float(np.dot(capture64, capture64)) * float(np.dot(canonical64, canonical64))
    )
    correlation = (
        None if energy == 0.0 else min(1.0, max(-1.0, zero_lag / energy))
    )
    return best_lag, correlation


def _capture_from_blocks(
    master: CanonicalMaster,
    blocks: Iterable[np.ndarray],
) -> CaptureReplayStream:
    if not isinstance(master, CanonicalMaster):
        raise TypeError("master must be CanonicalMaster")
    manifest = master.manifest
    source_native = manifest.transform_dag.source_native
    rate = source_native.output_sample_rate_hz
    channels = source_native.output_channels
    converter = (
        StreamingResampler(rate, _CAPTURE_SAMPLE_RATE_HZ)
        if rate != _CAPTURE_SAMPLE_RATE_HZ
        else None
    )
    digest = hashlib.sha256()
    frame_count = 0
    outputs: list[np.ndarray] = []
    for block in blocks:
        if (
            not isinstance(block, np.ndarray)
            or block.ndim != 1
            or block.dtype != np.dtype("<f4")
            or block.size % channels
        ):
            raise ValueError(
                "source-native PCM must be whole interleaved little-endian float32 frames"
            )
        digest.update(block.tobytes(order="C"))
        frame_count += block.size // channels
        # The exact WASAPI callback transform in yamabiko/capture/audio.py.
        pcm = block.reshape(-1, channels)
        pcm = pcm.mean(axis=1) if channels > 1 else pcm[:, 0]
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        if not np.isfinite(pcm).all():
            raise ValueError("source-native PCM contains non-finite samples")
        outputs.append(converter.process(pcm) if converter is not None else pcm)
    if converter is not None:
        # Live capture keeps feeding the stream; flushing here yields the
        # samples production would emit for the end of a finite master.
        outputs.append(converter.finish())
    if (
        frame_count != source_native.output_sample_count
        or digest.hexdigest() != source_native.output_byte_sha256
    ):
        raise ValueError("source-native PCM does not match its registered master")
    samples = (
        np.concatenate(outputs).astype("<f4", copy=False)
        if outputs
        else np.zeros(0, dtype="<f4")
    )
    if samples.size != manifest.sample_count:
        raise ValueError(
            "production capture stream length does not match the canonical timeline"
        )
    lag, correlation = _alignment(samples, master.samples)
    if lag != 0:
        raise ValueError(
            "production capture stream is not aligned with the canonical "
            f"timeline (lag {lag} samples)"
        )
    immutable = np.ascontiguousarray(samples, dtype="<f4").copy()
    immutable.setflags(write=False)
    provenance: dict[str, JsonValue] = {
        "version": CAPTURE_REPLAY_VERSION,
        "source_native_artifact_id": source_native.output_artifact_id,
        "source_native_byte_sha256": source_native.output_byte_sha256,
        "source_native_sample_rate_hz": rate,
        "source_native_channels": channels,
        "downmix": "mono-pass-through" if channels == 1 else "mean-of-channels",
        "resampler": _CAPTURE_RESAMPLER if converter is not None else "none",
        "resampler_quality": converter.quality if converter is not None else None,
        "soxr_version": soxr.__version__,
        "stream_end": "finish" if converter is not None else "none",
        "output_sample_rate_hz": _CAPTURE_SAMPLE_RATE_HZ,
        "output_sample_count": int(immutable.size),
        "output_pcm_sha256": hashlib.sha256(immutable.tobytes(order="C")).hexdigest(),
        "timeline_id": manifest.timeline_id,
        "canonical_pcm_sha256": manifest.canonical_pcm_sha256,
        "alignment": {
            "method": _CAPTURE_ALIGNMENT_METHOD,
            "search_radius_samples": _CAPTURE_ALIGNMENT_SEARCH_SAMPLES,
            "lag_samples": lag,
            "zero_lag_correlation": correlation,
        },
    }
    return CaptureReplayStream(
        master=master,
        samples=immutable,
        provenance=provenance,
        provenance_sha256=canonical_sha256(provenance),
    )


def derive_capture_replay_stream(
    master: CanonicalMaster,
    source_native: np.ndarray,
) -> CaptureReplayStream:
    """Run in-memory interleaved source-native PCM through production capture."""

    if not isinstance(master, CanonicalMaster):
        raise TypeError("master must be CanonicalMaster")
    if not isinstance(source_native, np.ndarray) or source_native.ndim != 1:
        raise TypeError("source_native must be one-dimensional interleaved PCM")
    step = master.manifest.transform_dag.source_native
    block_values = step.output_sample_rate_hz * step.output_channels
    blocks = (
        np.ascontiguousarray(source_native[start : start + block_values])
        for start in range(0, source_native.size, block_values)
    )
    return _capture_from_blocks(master, blocks)


def load_capture_replay_stream(
    master: CanonicalMaster,
    boundary: PrivateBoundary,
) -> CaptureReplayStream:
    """Stream a registered source-native master file through production capture."""

    if not isinstance(master, CanonicalMaster):
        raise TypeError("master must be CanonicalMaster")
    if not isinstance(boundary, PrivateBoundary):
        raise TypeError("boundary must be PrivateBoundary")
    manifest = master.manifest
    step = manifest.transform_dag.source_native
    path = resolve_private_locator_asset(
        manifest.fixture_id,
        manifest.rights,
        "masters",
        step.output_artifact_id,
        boundary,
    )
    block_bytes = step.output_sample_rate_hz * step.output_channels * 4

    def blocks() -> Iterator[np.ndarray]:
        with path.open("rb") as handle:
            while chunk := handle.read(block_bytes):
                yield np.frombuffer(chunk, dtype="<f4")

    try:
        return _capture_from_blocks(master, blocks())
    except OSError:
        raise ValueError(
            f"source-native master {step.output_artifact_id} is unreadable"
        ) from None
