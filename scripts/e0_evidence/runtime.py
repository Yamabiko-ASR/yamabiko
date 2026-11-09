"""Auditable run orchestration and best-effort resource instrumentation."""

from __future__ import annotations

import copy
import ctypes
import functools
import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, cast

from scripts.e0_evidence.align import (
    JA_TIME_DP_1_0,
    ScoredEvent,
    ScoredHypothesis,
    align_component,
    build_temporal_components,
)

from scripts.e0_evidence import contracts
from scripts.e0_evidence._common import (
    exclusive_write_bytes,
    file_sha256_and_size,
    require_sha256,
    require_uuid4,
)
from scripts.e0_evidence.decoder import (
    DecoderArmResult,
    DecoderInstrumentationError,
    DecoderParityError,
    OracleComponent,
    bind_model_artifact,
    oracle_component_map_sha256,
    run_p0,
    run_p1,
    run_p2a,
    run_p2b,
)
from scripts.e0_evidence.ledger import LedgerWriter
from scripts.e0_evidence.nemo_tdt import (
    NemoTdtArmResult,
    NemoTdtDecoder,
    decoder_identity as nemo_tdt_decoder_identity,
    installed_runtime_versions as nemo_tdt_runtime_versions,
    load_nemo_tdt,
    require_fp32_artifact,
    run_nemo_p0,
    run_nemo_p1,
)
from scripts.e0_evidence.normalize_ja import (
    JA_LENIENT_1_1,
    JA_MINIMAL_1_1,
    normalize_minimal,
)
from scripts.e0_evidence.pcm import load_canonical_master, load_capture_replay_stream
from scripts.e0_evidence.private_root import PrivateBoundary, require_private_path
from scripts.e0_evidence.references import (
    ACOUSTIC_TAGS,
    ReferenceSet,
    load_reference_set,
)
from scripts.e0_evidence.registry import (
    load_reference_texts,
    load_registered_master,
    load_registered_reference,
)
from scripts.e0_evidence.replay import (
    ReplayResult,
    TaggerArtifact,
    VadArtifact,
    _verify_silero_vad_contract,
    _verify_vad_assist_contract,
    load_production_model,
    load_production_tagger,
    production_chunker_config,
    replay_master,
    resolve_tagger_artifact,
    resolve_vad_artifact,
)
from scripts.e0_evidence.schema import (
    MONOTONIC_CLOCK_ID,
    SCHEMA_VERSION,
    ArmDefinition,
    ArmKind,
    ArmStatus,
    DecoderAdapter,
    DiagnosticCode,
    EvidenceRecord,
    EvidenceRecordDraft,
    ObservationStatus,
    Partition,
    PrivateContentClass,
    RecordKind,
    RunFinalManifest,
    RunStartManifest,
    Stage,
    TerminalOutcome,
    TextRole,
    new_opaque_id,
)
from scripts.e0_evidence.score import (
    JA_EVENT_USABLE_1_1,
    RAW_UNICODE_1_0_SHA256,
    RAW_UNICODE_1_0_VERSION,
    USABILITY_CURVE_POLICY_SHA256,
    USABILITY_CURVE_POLICY_VERSION,
    ArmEvidence,
    derive_event_fates,
    scored_reference_events,
)
from scripts.e0_evidence.wire import (
    arm_definition_sha256,
    canonical_json_bytes,
    canonical_sha256,
)
from yamabiko import hf_env as _hf_env  # noqa: F401 - pin HF_HOME before Hub use
from yamabiko.asr.asr_client import FasterWhisperASR, build_asr
from yamabiko.config import load_full_config
from yamabiko.models import (
    WHISPER_FILES,
    whisper_files_cached,
    whisper_revision_for,
)


RUN_REQUEST_SCHEMA_VERSION = "e0-run-request-6.0"
COHORT_SCHEMA_VERSION = "e0-cohort-5.0"
HARD_STRATUM_VERSION = "e0-hard-stratum-1.0"
# The acoustic conditions the hard stratum covers unless a request declares
# its own subset of ``ACOUSTIC_TAGS``.
DEFAULT_HARD_STRATUM_TAGS = (
    "effects_noise",
    "high_f0_nonmodal",
    "long_endpoint",
    "music",
    "music_effects_noise",
    "overlap",
    "quiet_dialogue",
    "rapid_turns",
    "short_acknowledgements",
)
_DEPENDENCIES = (
    "ctranslate2",
    "faster-whisper",
    "huggingface-hub",
    "numpy",
    "onnxruntime",
    "PyYAML",
    "soxr",
    "tokenizers",
)
_ARM_ORDER = (ArmKind.P0, ArmKind.P1, ArmKind.P2A, ArmKind.P2B)
# One decoder call's evidence, from whichever adapter the request selected.
_ArmResult = DecoderArmResult | NemoTdtArmResult
_LOGGER = logging.getLogger(__name__)


def _absolute(value: object, name: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    try:
        return path.resolve(strict=False)
    except OSError:
        raise ValueError(f"{name} cannot be resolved") from None


@dataclass(frozen=True, slots=True)
class OracleComponentRequest:
    component_id: str
    event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_uuid4(self.component_id, "component_id")
        event_ids = tuple(self.event_ids)
        if not event_ids or len(event_ids) != len(set(event_ids)):
            raise ValueError("oracle component event IDs must be nonempty and unique")
        for event_id in event_ids:
            require_uuid4(event_id, "oracle event_id")
        object.__setattr__(self, "event_ids", event_ids)


@dataclass(frozen=True, slots=True)
class VadArtifactRequest:
    """A candidate Silero ONNX file and the SHA-256 its bytes must have."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        path = _absolute(str(self.path), "vad_artifact.path")
        if not path.is_file():
            raise ValueError("vad_artifact.path must name a regular file")
        object.__setattr__(self, "path", path)
        require_sha256(self.sha256, "vad_artifact.sha256")


@dataclass(frozen=True, slots=True)
class RunRequest:
    request_id: str
    master_id: str
    reference_set_id: str
    config_path: Path
    model_artifact_path: Path
    arms: tuple[ArmKind, ...]
    arm_ids: Mapping[ArmKind, str]
    cohort_id: str
    cohort_fixture_ids: tuple[str, ...]
    oracle_components: tuple[OracleComponentRequest, ...]
    vad_artifact: VadArtifactRequest | None = None
    decoder_adapter: DecoderAdapter = DecoderAdapter.FASTER_WHISPER
    hard_stratum_tags: tuple[str, ...] = DEFAULT_HARD_STRATUM_TAGS

    def __post_init__(self) -> None:
        require_uuid4(self.request_id, "request_id")
        require_uuid4(self.master_id, "master_id")
        require_uuid4(self.reference_set_id, "reference_set_id")
        config_path = _absolute(str(self.config_path), "config_path")
        model_path = _absolute(str(self.model_artifact_path), "model_artifact_path")
        if not config_path.is_file():
            raise ValueError("config_path must name a regular file")
        if not model_path.is_dir():
            raise ValueError("model_artifact_path must be a model snapshot directory")
        object.__setattr__(self, "config_path", config_path)
        object.__setattr__(self, "model_artifact_path", model_path)
        arms = tuple(self.arms)
        if not arms or len(arms) != len(set(arms)):
            raise ValueError("arms must be nonempty and unique")
        if any(not isinstance(arm, ArmKind) for arm in arms):
            raise TypeError("arms must contain ArmKind values")
        ordered = tuple(arm for arm in _ARM_ORDER if arm in arms)
        if any(arm in arms for arm in (ArmKind.P1, ArmKind.P2A, ArmKind.P2B)):
            if ArmKind.P0 not in arms:
                raise ValueError("P1/P2 diagnostic arms require P0")
        if ArmKind.P2B in arms and ArmKind.P2A not in arms:
            raise ValueError("P2b requires P2a")
        object.__setattr__(self, "arms", ordered)
        try:
            arm_ids = dict(self.arm_ids)
        except (TypeError, ValueError):
            raise TypeError(
                "arm_ids must be a mapping from ArmKind to UUIDv4"
            ) from None
        if set(arm_ids) != set(ordered):
            raise ValueError("arm_ids must name exactly the requested arms")
        if any(not isinstance(kind, ArmKind) for kind in arm_ids):
            raise TypeError("arm_ids keys must be ArmKind values")
        for kind, arm_id in arm_ids.items():
            require_uuid4(arm_id, f"arm_ids.{kind.value}")
        if len(arm_ids.values()) != len(set(arm_ids.values())):
            raise ValueError("arm_ids must be globally unique")
        object.__setattr__(self, "arm_ids", MappingProxyType(arm_ids))
        require_uuid4(self.cohort_id, "cohort_id")
        fixture_ids = tuple(self.cohort_fixture_ids)
        if not fixture_ids or len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("cohort_fixture_ids must be nonempty and unique")
        for fixture_id in fixture_ids:
            require_uuid4(fixture_id, "cohort_fixture_id")
        object.__setattr__(self, "cohort_fixture_ids", tuple(sorted(fixture_ids)))
        reserved_ids = {
            self.request_id,
            self.master_id,
            self.reference_set_id,
            self.cohort_id,
            *fixture_ids,
        }
        if reserved_ids.intersection(arm_ids.values()):
            raise ValueError("cohort arm IDs must not alias request or corpus IDs")
        components = tuple(self.oracle_components)
        if any(not isinstance(item, OracleComponentRequest) for item in components):
            raise TypeError("oracle_components must contain component requests")
        component_ids = tuple(item.component_id for item in components)
        event_ids = tuple(
            event_id for item in components for event_id in item.event_ids
        )
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("oracle component IDs must be unique")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("oracle components must not repeat events")
        if ArmKind.P1 in arms and not components:
            raise ValueError("P1 requires predeclared oracle components")
        if ArmKind.P1 not in arms and components:
            raise ValueError("oracle components require P1")
        object.__setattr__(self, "oracle_components", components)
        if self.vad_artifact is not None and not isinstance(
            self.vad_artifact, VadArtifactRequest
        ):
            raise TypeError("vad_artifact must be a VadArtifactRequest")
        if not isinstance(self.decoder_adapter, DecoderAdapter):
            raise TypeError("decoder_adapter must be a DecoderAdapter")
        if self.decoder_adapter is DecoderAdapter.ONNX_ASR_NEMO_TDT:
            if ArmKind.P2A in arms or ArmKind.P2B in arms:
                raise ValueError(
                    "the onnx_asr_nemo_tdt adapter supports only P0 and P1; "
                    "P2a/P2b are faster-whisper gate counterfactuals"
                )
            require_fp32_artifact(model_path)
        tags = tuple(self.hard_stratum_tags)
        if (
            not tags
            or any(type(tag) is not str for tag in tags)
            or tags != tuple(sorted(set(tags)))
        ):
            raise ValueError("hard_stratum_tags must be nonempty, sorted, and unique")
        if not set(tags) <= ACOUSTIC_TAGS:
            raise ValueError("hard_stratum_tags must be acoustic tags")
        object.__setattr__(self, "hard_stratum_tags", tags)


def load_run_request(path: Path, boundary: PrivateBoundary) -> RunRequest:
    request_path = require_private_path(path, boundary)
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("run request is unreadable") from None
    fields = {
        "schema_version",
        "request_id",
        "master_id",
        "reference_set_id",
        "config_path",
        "model_artifact_path",
        "arms",
        "arm_ids",
        "cohort_id",
        "cohort_fixture_ids",
        "oracle_components",
    }
    if not isinstance(value, dict) or not fields <= set(value) <= fields | {
        "vad_artifact",
        "decoder_adapter",
        "hard_stratum_tags",
    }:
        raise ValueError("run request has an invalid wire shape")
    if value["schema_version"] != RUN_REQUEST_SCHEMA_VERSION:
        raise ValueError("run request schema version is unsupported")
    vad_artifact: VadArtifactRequest | None = None
    if "vad_artifact" in value:
        declared = value["vad_artifact"]
        if not isinstance(declared, dict) or set(declared) != {"path", "sha256"}:
            raise ValueError("run request vad_artifact must be {path, sha256}")
        vad_artifact = VadArtifactRequest(
            path=_absolute(declared["path"], "vad_artifact.path"),
            sha256=declared["sha256"],
        )
    try:
        decoder_adapter = DecoderAdapter(
            value.get("decoder_adapter", DecoderAdapter.FASTER_WHISPER.value)
        )
    except (TypeError, ValueError):
        raise ValueError("run request names an unsupported decoder_adapter") from None
    if (
        not isinstance(value["arms"], list)
        or not isinstance(value["arm_ids"], dict)
        or not isinstance(value["cohort_fixture_ids"], list)
        or not isinstance(value["oracle_components"], list)
    ):
        raise ValueError("run request collections must be lists")
    hard_stratum_tags = value.get("hard_stratum_tags", list(DEFAULT_HARD_STRATUM_TAGS))
    if not isinstance(hard_stratum_tags, list):
        raise ValueError("run request hard_stratum_tags must be a list")
    components: list[OracleComponentRequest] = []
    for item in value["oracle_components"]:
        if not isinstance(item, dict) or set(item) != {"component_id", "event_ids"}:
            raise ValueError("oracle component request is invalid")
        if not isinstance(item["event_ids"], list):
            raise ValueError("oracle component event_ids must be a list")
        components.append(
            OracleComponentRequest(
                component_id=item["component_id"],
                event_ids=tuple(item["event_ids"]),
            )
        )
    try:
        arms = tuple(ArmKind(item) for item in value["arms"])
    except (TypeError, ValueError):
        raise ValueError("run request names an unsupported arm") from None
    if any(
        type(key) is not str or type(item) is not str
        for key, item in value["arm_ids"].items()
    ):
        raise ValueError("run request arm_ids must map arm names to UUIDv4 values")
    try:
        arm_ids = {ArmKind(key): item for key, item in value["arm_ids"].items()}
    except ValueError:
        raise ValueError("run request arm_ids name an unsupported arm") from None
    return RunRequest(
        request_id=value["request_id"],
        master_id=value["master_id"],
        reference_set_id=value["reference_set_id"],
        config_path=_absolute(value["config_path"], "config_path"),
        model_artifact_path=_absolute(
            value["model_artifact_path"], "model_artifact_path"
        ),
        arms=arms,
        arm_ids=arm_ids,
        cohort_id=value["cohort_id"],
        cohort_fixture_ids=tuple(value["cohort_fixture_ids"]),
        oracle_components=tuple(components),
        vad_artifact=vad_artifact,
        decoder_adapter=decoder_adapter,
        hard_stratum_tags=tuple(hard_stratum_tags),
    )


def _file_identity(path: Path) -> tuple[str, int]:
    try:
        return file_sha256_and_size(path)
    except OSError:
        raise ValueError("artifact is unreadable") from None


def _artifact_manifest(path: Path) -> dict[str, object]:
    """Return the per-file identity whose canonical hash names the artifact."""

    if path.is_file():
        digest, size = _file_identity(path)
        return {"kind": "file", "byte_sha256": digest, "byte_count": size}
    if not path.is_dir():
        raise ValueError("model artifact must be a file or directory")
    entries: list[dict[str, object]] = []
    try:
        files = sorted(
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(path).as_posix(),
        )
    except OSError:
        raise ValueError("model artifact tree is unreadable") from None
    if not files:
        raise ValueError("model artifact directory is empty")
    for item in files:
        digest, size = _file_identity(item)
        entries.append(
            {
                "relative_path": item.relative_to(path).as_posix(),
                "byte_sha256": digest,
                "byte_count": size,
            }
        )
    return {"kind": "tree", "files": entries}


def _hash_artifact(path: Path) -> str:
    return canonical_sha256(_artifact_manifest(path))


def _configured_whisper_model(full_config: Mapping[str, object]) -> str:
    try:
        asr = full_config["asr"]
        if not isinstance(asr, Mapping):
            raise TypeError
        faster_whisper = asr["faster_whisper"]
        if not isinstance(faster_whisper, Mapping):
            raise TypeError
        model = faster_whisper["model"]
    except (KeyError, TypeError):
        raise ValueError("effective config has no faster-whisper model") from None
    if type(model) is not str or not model:
        raise ValueError("effective faster-whisper model must be a nonempty string")
    return model


def _resolve_configured_model_artifact(full_config: Mapping[str, object]) -> Path:
    """Resolve exactly the local snapshot production will load, without network I/O."""

    configured = _configured_whisper_model(full_config)
    local = Path(configured).expanduser()
    if local.is_dir():
        return local.resolve(strict=True)
    if local.exists():
        raise ValueError("configured faster-whisper model must be a directory")
    try:
        cached = whisper_files_cached(configured)
    except Exception:
        raise ValueError("configured faster-whisper cache cannot be verified") from None
    if not cached:
        raise ValueError("configured faster-whisper model is not fully cached")
    # ``yamabiko.hf_env`` was imported before this lazy Hub import so the
    # resolver and production runtime use the same Yamabiko cache root.
    from huggingface_hub import try_to_load_from_cache

    revision = whisper_revision_for(configured)
    parents: set[Path] = set()
    for filename in WHISPER_FILES:
        hit = try_to_load_from_cache(
            repo_id=configured,
            filename=filename,
            revision=revision,
        )
        if not isinstance(hit, str) or not Path(hit).is_file():
            raise ValueError("configured faster-whisper cache is incomplete")
        parents.add(Path(hit).parent.resolve(strict=True))
    if len(parents) != 1:
        raise ValueError("configured faster-whisper files span multiple snapshots")
    return next(iter(parents))


def _require_loaded_model_artifact(
    requested_path: Path,
    full_config: Mapping[str, object],
) -> Path:
    resolved = _resolve_configured_model_artifact(full_config)
    try:
        same = os.path.samefile(requested_path, resolved)
    except OSError:
        same = False
    if not same:
        raise ValueError(
            "model_artifact_path is not the snapshot selected by effective config"
        )
    return resolved


def _config_for_resolved_model(
    full_config: Mapping[str, object],
    resolved_model_artifact: Path,
) -> dict[str, object]:
    """Preserve production settings while pinning construction to one snapshot."""

    if not isinstance(full_config, dict):
        raise TypeError("effective config must be a dictionary")
    construction_config = copy.deepcopy(full_config)
    try:
        asr = construction_config["asr"]
        if not isinstance(asr, dict):
            raise TypeError
        faster_whisper = asr["faster_whisper"]
        if not isinstance(faster_whisper, dict):
            raise TypeError
    except (KeyError, TypeError):
        raise ValueError("effective config has no faster-whisper settings") from None
    faster_whisper["model"] = str(resolved_model_artifact)
    return construction_config


def _production_boundary_pad_ms(full_config: Mapping[str, object]) -> int:
    """Read the boundary pad exactly as production's ``build_asr`` reads it."""

    asr = full_config.get("asr")
    if not isinstance(asr, Mapping):
        raise ValueError("effective config has no asr settings")
    try:
        pad_ms = int(asr.get("boundary_pad_ms", 500))
    except (TypeError, ValueError):
        raise ValueError("effective asr.boundary_pad_ms is not an integer") from None
    if pad_ms < 0:
        raise ValueError("effective asr.boundary_pad_ms must be nonnegative")
    return pad_ms


def _run_tool(args: Sequence[str], *, cwd: Path | None = None) -> bytes | None:
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _git_identity(repo: Path) -> dict[str, object]:
    commit_bytes = _run_tool(("git", "rev-parse", "HEAD"), cwd=repo)
    status_bytes = _run_tool(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        cwd=repo,
    )
    git_version = _run_tool(("git", "--version"), cwd=repo)
    if commit_bytes is None or status_bytes is None or git_version is None:
        raise RuntimeError("Git identity is unavailable")
    try:
        commit = commit_bytes.decode("ascii").strip()
        version_text = git_version.decode("ascii").strip()
    except UnicodeError:
        raise RuntimeError("Git identity is malformed") from None
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Git commit identity is malformed")
    return {
        "commit": commit,
        "dirty": bool(status_bytes),
        "dirty_digest_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "git_version": version_text,
    }


def _dependency_identity() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in _DEPENDENCIES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not_installed"
    return result


def _distribution_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _verify_release_lock(repo: Path) -> str:
    """Refuse a runtime whose E0 dependencies differ from the release lock.

    Returns the SHA-256 of ``<repo>/constraints-release.txt``.
    """

    try:
        lock_bytes = (repo / "constraints-release.txt").read_bytes()
        lock_text = lock_bytes.decode("utf-8")
    except (OSError, UnicodeError):
        raise ValueError("the release lock is unreadable") from None
    pins: dict[str, str] = {}
    for line in lock_text.splitlines():
        name, separator, pinned = line.split("#", 1)[0].partition("==")
        if separator:
            pins[_distribution_key(name.strip())] = pinned.strip()
    for name, installed in _dependency_identity().items():
        pinned = pins.get(_distribution_key(name), "unpinned")
        if installed != pinned:
            raise ValueError(
                f"runtime dependency {name}=={installed} does not match "
                f"the release lock {pinned}"
            )
    return hashlib.sha256(lock_bytes).hexdigest()


def _gpu_identity() -> dict[str, object]:
    raw = _run_tool(
        (
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    if raw is None:
        return {"status": ObservationStatus.INSTRUMENTATION_MISSING.value}
    try:
        rows = [
            line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()
        ]
    except UnicodeError:
        return {"status": ObservationStatus.INSTRUMENTATION_MISSING.value}
    devices: list[dict[str, str]] = []
    for row in rows:
        fields = [item.strip() for item in row.split(",")]
        if len(fields) != 4 or any(not item for item in fields):
            return {"status": ObservationStatus.INSTRUMENTATION_MISSING.value}
        devices.append(
            {
                "uuid": fields[0],
                "name": fields[1],
                "driver_version": fields[2],
                "compute_capability": fields[3],
            }
        )
    if not devices:
        return {"status": ObservationStatus.INSTRUMENTATION_MISSING.value}
    return {"status": ObservationStatus.OBSERVED.value, "devices": devices}


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


@functools.cache
def _windows_memory_api() -> tuple[Callable[[], int | None], Callable[..., int]]:
    """Return GetCurrentProcess and GetProcessMemoryInfo with declared types.

    Without them ctypes passes the pseudo-handle as a 32-bit int, which
    GetProcessMemoryInfo rejects on 64-bit Windows.
    """

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    current_process = kernel32.GetCurrentProcess
    current_process.argtypes = ()
    current_process.restype = ctypes.c_void_p
    memory_info = psapi.GetProcessMemoryInfo
    memory_info.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    )
    memory_info.restype = ctypes.c_int
    return current_process, memory_info


def _rss_bytes() -> int | None:
    if os.name == "nt":
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        try:
            current_process, memory_info = _windows_memory_api()
            ok = memory_info(current_process(), ctypes.byref(counters), counters.cb)
        except (AttributeError, OSError):
            return None
        value = int(counters.WorkingSetSize) if ok else 0
        return value if value > 0 else None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    multiplier = 1 if sys.platform == "darwin" else 1024
    result = value * multiplier
    return result if result > 0 else None


def _device_vram_bytes() -> int | None:
    """GPU 0's used memory in bytes, as ``nvidia-smi`` reports it."""

    raw = _run_tool(
        (
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        )
    )
    if raw is None:
        return None
    try:
        used_mib = int(raw.decode("ascii").strip())
    except (UnicodeError, ValueError):
        return None
    return used_mib * 1024 * 1024 if used_mib >= 0 else None


class _ResourceSampler:
    """Sample peak RSS and GPU 0's memory once a second.

    VRAM is a device-wide reading, so an arm's ``peak_vram_bytes`` is the
    device peak over the arm minus the reading taken when the arm started.
    The arm-start reading counts toward the peak, so the delta is never
    negative; it is unobserved unless the arm-start reading and at least one
    later reading succeeded.
    """

    _INTERVAL_SECONDS = 1.0

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="e0-resource-sampler",
            daemon=False,
        )
        # A reading taken before a reset must not land in the next arm.
        self._generation = 0
        self.peak_rss_bytes: int | None = None
        self._vram_start: int | None = None
        self._vram_peak: int | None = None
        self._vram_followed = False

    def _sample_once(self) -> None:
        with self._lock:
            generation = self._generation
        rss = _rss_bytes()
        vram = _device_vram_bytes()
        with self._lock:
            if generation != self._generation:
                return
            if rss is not None:
                self.peak_rss_bytes = max(self.peak_rss_bytes or rss, rss)
            if vram is not None and self._vram_peak is not None:
                self._vram_peak = max(self._vram_peak, vram)
                self._vram_followed = True

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._INTERVAL_SECONDS)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        """Start a new arm's window from fresh RSS and VRAM readings."""

        rss = _rss_bytes()
        vram = _device_vram_bytes()
        with self._lock:
            self._generation += 1
            self.peak_rss_bytes = rss
            self._vram_start = vram
            self._vram_peak = vram
            self._vram_followed = False

    def snapshot(self) -> tuple[int | None, int | None]:
        """Return the window's peak RSS and its device VRAM delta."""

        self._sample_once()
        with self._lock:
            if (
                self._vram_followed
                and self._vram_start is not None
                and self._vram_peak is not None
            ):
                vram_delta: int | None = self._vram_peak - self._vram_start
            else:
                vram_delta = None
            return self.peak_rss_bytes, vram_delta

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            raise RuntimeError("resource sampler did not stop")
        self._sample_once()


class _ArmSink:
    def __init__(self, writer: LedgerWriter, arm_id: str) -> None:
        self._writer = writer
        self.arm_id = arm_id

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        return self._writer.emit(draft)

    def emit_private_text(
        self,
        *,
        canonical_record_id: str,
        arm_id: str,
        text_id: str,
        role: TextRole,
        text: str,
    ) -> str:
        return self._writer.emit_private_text(
            canonical_record_id=canonical_record_id,
            arm_id=arm_id,
            text_id=text_id,
            role=role,
            text=text,
        )


def _parent_arm_id(kind: ArmKind, arm_ids: Mapping[ArmKind, str]) -> str | None:
    if kind is ArmKind.P0:
        return None
    if kind in {ArmKind.P1, ArmKind.P2A}:
        return arm_ids[ArmKind.P0]
    return arm_ids[ArmKind.P2A]


def _declared_changes(kind: ArmKind) -> tuple[str, ...]:
    if kind is ArmKind.P1:
        return ("boundary_mode",)
    if kind is ArmKind.P2B:
        return ("decoder.gates.no_speech_threshold",)
    return ()


def _invariant_arm_parameters(
    kind: ArmKind,
    *,
    config_sha256: str,
    model_artifact_sha256: str,
    vad_artifact_sha256: str,
    decoder_adapter: DecoderAdapter = DecoderAdapter.FASTER_WHISPER,
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "config_sha256": config_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "vad_artifact_sha256": vad_artifact_sha256,
        "alignment_policy": {
            "version": JA_TIME_DP_1_0.version,
            "sha256": JA_TIME_DP_1_0.sha256,
        },
        "normalization_policies": (
            {
                "version": RAW_UNICODE_1_0_VERSION,
                "sha256": RAW_UNICODE_1_0_SHA256,
            },
            {
                "version": JA_MINIMAL_1_1.version,
                "sha256": JA_MINIMAL_1_1.sha256,
            },
            {
                "version": JA_LENIENT_1_1.version,
                "sha256": JA_LENIENT_1_1.sha256,
            },
        ),
        "usability_policy": {
            "version": JA_EVENT_USABLE_1_1.version,
            "sha256": JA_EVENT_USABLE_1_1.sha256,
        },
        "usability_curve_policy": {
            "version": USABILITY_CURVE_POLICY_VERSION,
            "sha256": USABILITY_CURVE_POLICY_SHA256,
        },
    }
    if kind is ArmKind.P0:
        parameters.update({"boundary_mode": "production", "call_surface": "wrapper"})
    elif kind is ArmKind.P1:
        parameters.update({"boundary_mode": "oracle", "call_surface": "wrapper"})
    elif kind is ArmKind.P2A:
        parameters.update({"boundary_mode": "production", "call_surface": "direct"})
    else:
        parameters.update(
            {
                "boundary_mode": "production",
                "call_surface": "direct",
                "no_speech_threshold": None,
            }
        )
    if decoder_adapter is DecoderAdapter.FASTER_WHISPER:
        parameters["decoder_random_seed"] = contracts.DECODER_RANDOM_SEED
    else:
        # CTranslate2's sampling seed has no meaning for greedy TDT decoding.
        parameters["decoder_adapter"] = decoder_adapter.value
    return parameters


def _cohort_identity(
    request: RunRequest,
    *,
    partition: Partition,
    config_sha256: str,
    model_artifact_sha256: str,
    vad_artifact: VadArtifact,
    tagger_artifact: TaggerArtifact,
    decoder_adapter: Mapping[str, object],
) -> tuple[str, Mapping[ArmKind, str], Mapping[str, object]]:
    profiles: dict[ArmKind, str] = {}
    arm_payloads: list[dict[str, object]] = []
    for kind in request.arms:
        payload = {
            "arm_id": request.arm_ids[kind],
            "kind": kind.value,
            "parent_arm_id": _parent_arm_id(kind, request.arm_ids),
            "parameters": _invariant_arm_parameters(
                kind,
                config_sha256=config_sha256,
                model_artifact_sha256=model_artifact_sha256,
                vad_artifact_sha256=vad_artifact.sha256,
                decoder_adapter=request.decoder_adapter,
            ),
            "declared_changes": _declared_changes(kind),
        }
        profile_sha256 = canonical_sha256(payload)
        profiles[kind] = profile_sha256
        arm_payloads.append({**payload, "arm_profile_sha256": profile_sha256})
    identity = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "cohort_id": request.cohort_id,
        "partition": partition.value,
        "fixture_ids": request.cohort_fixture_ids,
        "vad_artifact": vad_artifact.identity(),
        "vad_assist": {
            "model_sha256": tagger_artifact.sha256,
            "byte_count": tagger_artifact.byte_count,
            "layout_sha256": canonical_sha256(contracts.SOUND_TAG_ASSIST_LAYOUT),
        },
        "decoder_adapter": decoder_adapter,
        "hard_stratum": {
            "version": HARD_STRATUM_VERSION,
            "tags": request.hard_stratum_tags,
        },
        "arms": tuple(arm_payloads),
    }
    manifest_sha256 = canonical_sha256(identity)
    manifest = {**identity, "cohort_manifest_sha256": manifest_sha256}
    return manifest_sha256, MappingProxyType(profiles), MappingProxyType(manifest)


def _freeze_cohort_manifest(
    manifest: Mapping[str, object],
    boundary: PrivateBoundary,
) -> None:
    cohort_id = require_uuid4(manifest.get("cohort_id"), "cohort_id")
    directory = require_private_path(boundary.root / "cohorts", boundary)
    try:
        directory.mkdir(exist_ok=True)
    except OSError:
        raise ValueError("cohort registry cannot be created") from None
    target = require_private_path(
        directory / f"{cohort_id}.private.json",
        boundary,
    )
    payload = canonical_json_bytes(manifest, newline=True)
    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError:
            raise ValueError("frozen cohort manifest is unreadable") from None
        if existing != payload:
            raise ValueError("cohort identity is immutable once frozen")
        return
    temporary = require_private_path(
        directory / f".{target.name}.{uuid.uuid4()}.tmp",
        boundary,
    )
    try:
        exclusive_write_bytes(temporary, payload)
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise
        except OSError:
            # The volume has no hard links: create the target exclusively.
            exclusive_write_bytes(target, payload)
    except FileExistsError:
        try:
            existing = target.read_bytes()
        except OSError:
            raise ValueError("frozen cohort manifest is unreadable") from None
        if existing != payload:
            raise ValueError("cohort identity is immutable once frozen")
    except OSError:
        raise ValueError("cohort manifest cannot be frozen") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _arm_definitions(
    request: RunRequest,
    *,
    config_sha256: str,
    model_artifact_sha256: str,
    vad_artifact_sha256: str,
    model_runtime_sha256: str,
    cohort_manifest_sha256: str,
    arm_profile_sha256: Mapping[ArmKind, str],
    oracle_components: Sequence[Mapping[str, object]],
) -> tuple[ArmDefinition, ...]:
    definitions: list[ArmDefinition] = []
    for kind in request.arms:
        parameters = _invariant_arm_parameters(
            kind,
            config_sha256=config_sha256,
            model_artifact_sha256=model_artifact_sha256,
            vad_artifact_sha256=vad_artifact_sha256,
            decoder_adapter=request.decoder_adapter,
        )
        parameters.update(
            {
                "model_runtime_sha256": model_runtime_sha256,
                "cohort_id": request.cohort_id,
                "cohort_manifest_sha256": cohort_manifest_sha256,
                "arm_profile_sha256": arm_profile_sha256[kind],
            }
        )
        if kind is ArmKind.P1:
            # Oracle components are reference-specific and intentionally do not
            # participate in the invariant cohort arm-profile hash.
            parameters["oracle_components"] = tuple(oracle_components)
        definitions.append(
            ArmDefinition(
                arm_id=request.arm_ids[kind],
                kind=kind,
                parent_arm_id=_parent_arm_id(kind, request.arm_ids),
                parameters=parameters,
                declared_changes=_declared_changes(kind),
            )
        )
    return tuple(definitions)


class _FateDraftBuffer:
    """Validate a complete fate set before appending any of it to the ledger.

    ``derive_event_fates`` names a derived diagnostic as a parent of the fate
    it explains, so ``emit`` returns a provisional record and ``flush``
    replaces each provisional ID with the one the ledger assigns.
    """

    # Above any clock reading: provisional records sort after every record
    # already written and, among themselves, in the order flush writes them.
    _PROVISIONAL_NS = 1 << 62

    def __init__(self) -> None:
        self.drafts: list[tuple[str, EvidenceRecordDraft]] = []
        self._provisional_session_id = new_opaque_id()

    def emit(self, draft: EvidenceRecordDraft) -> EvidenceRecord:
        if not isinstance(draft, EvidenceRecordDraft):
            raise TypeError("fate sink accepts EvidenceRecordDraft values")
        ordinal = len(self.drafts)
        record = EvidenceRecord(
            schema_version=SCHEMA_VERSION,
            record_id=new_opaque_id(),
            run_id=self._provisional_session_id,
            arm_id=draft.arm_id,
            session_id=self._provisional_session_id,
            seq=ordinal,
            observed_monotonic_ns=self._PROVISIONAL_NS + ordinal,
            clock_id=MONOTONIC_CLOCK_ID,
            stage=draft.stage,
            kind=draft.kind,
            artifact_id=draft.artifact_id,
            parent_ids=draft.parent_ids,
            source_spans=draft.source_spans,
            observation_status=draft.observation_status,
            decision=draft.decision,
            diagnostics=draft.diagnostics,
            classification=draft.classification,
            details=draft.details,
        )
        self.drafts.append((record.record_id, draft))
        return record

    def flush(self, writer: LedgerWriter) -> None:
        written: dict[str, str] = {}
        for provisional_id, draft in self.drafts:
            parents = tuple(written.get(parent, parent) for parent in draft.parent_ids)
            record = writer.emit(replace(draft, parent_ids=parents))
            written[provisional_id] = record.record_id


def _scored_hypotheses(
    results: Sequence[_ArmResult],
) -> tuple[ScoredHypothesis, ...]:
    hypotheses: list[ScoredHypothesis] = []
    for result in results:
        if len(result.raw_records) != len(result.call.raw_texts):
            raise RuntimeError("decoder result text and record counts disagree")
        for record, text in zip(result.raw_records, result.call.raw_texts):
            hypotheses.append(
                ScoredHypothesis(
                    hypothesis_id=record.record_id,
                    source_spans=record.source_spans,
                    text=text,
                    synthetic_anchor_timeline_id=cast(
                        str | None,
                        record.details.get("synthetic_anchor_timeline_id"),
                    ),
                    synthetic_anchor_sample=cast(
                        int | None,
                        record.details.get("synthetic_anchor_sample"),
                    ),
                )
            )
    return tuple(hypotheses)


def _unique_records(
    groups: Sequence[Sequence[EvidenceRecord]],
) -> tuple[EvidenceRecord, ...]:
    result: list[EvidenceRecord] = []
    seen: set[str] = set()
    for group in groups:
        for record in group:
            if record.record_id in seen:
                continue
            seen.add(record.record_id)
            result.append(record)
    return tuple(result)


def _arm_evidence_records(
    kind: ArmKind,
    replay: ReplayResult | None,
    results_by_kind: Mapping[ArmKind, Sequence[_ArmResult]],
) -> tuple[EvidenceRecord, ...]:
    groups: list[Sequence[EvidenceRecord]] = []
    if replay is not None and kind in {ArmKind.P0, ArmKind.P2A, ArmKind.P2B}:
        groups.append(replay.integrity_records)
    dependencies: tuple[ArmKind, ...]
    if kind is ArmKind.P0:
        dependencies = (ArmKind.P0,)
    elif kind is ArmKind.P1:
        dependencies = (ArmKind.P1,)
    elif kind is ArmKind.P2A:
        dependencies = (ArmKind.P0, ArmKind.P2A)
    else:
        dependencies = (ArmKind.P0, ArmKind.P2A, ArmKind.P2B)
    for dependency in dependencies:
        for result in results_by_kind.get(dependency, ()):
            groups.append(result.evidence_records)
    return _unique_records(groups)


def _derive_and_emit_arm_fates(
    writer: LedgerWriter,
    definition: ArmDefinition,
    status: ArmStatus,
    reference_set: ReferenceSet,
    scored_events: Sequence[ScoredEvent],
    replay: ReplayResult | None,
    results_by_kind: Mapping[ArmKind, Sequence[_ArmResult]],
) -> None:
    hypotheses = _scored_hypotheses(results_by_kind.get(definition.kind, ()))
    records = _arm_evidence_records(definition.kind, replay, results_by_kind)
    arm = ArmEvidence(
        arm_id=definition.arm_id,
        kind=definition.kind,
        status=status,
        records=records,
        scored_events=tuple(scored_events),
        raw_hypotheses=hypotheses,
        reference_invalid_event_ids=(),
        alignment_policy=JA_TIME_DP_1_0,
    )
    components = build_temporal_components(
        arm.scored_events,
        arm.raw_hypotheses,
        arm.alignment_policy,
    )
    alignments = tuple(
        align_component(component, arm.alignment_policy, normalize_minimal)
        for component in components
    )
    buffer = _FateDraftBuffer()
    fates = derive_event_fates(reference_set.events, arm, alignments, buffer)
    if len(fates) != len(reference_set.events):
        raise RuntimeError("event-fate derivation did not cover every reference event")
    buffer.flush(writer)


def _measurement(
    writer: LedgerWriter,
    arm_id: str,
    name: str,
    unit: str,
    value: float | int | None,
    *,
    vram_scope: str | None = None,
) -> None:
    observed = value is not None
    if observed and (
        type(value) not in (int, float) or not math.isfinite(float(value))
    ):
        raise ValueError("measurement value must be finite")
    details: dict[str, object] = {"measurement_name": name, "unit": unit}
    if vram_scope is not None:
        details["vram_scope"] = vram_scope
    if observed:
        details["value"] = value
    writer.emit(
        EvidenceRecordDraft(
            arm_id=arm_id,
            stage=Stage.RUN,
            kind=RecordKind.MEASUREMENT,
            artifact_id=new_opaque_id(),
            observation_status=(
                ObservationStatus.OBSERVED
                if observed
                else ObservationStatus.INSTRUMENTATION_MISSING
            ),
            classification=PrivateContentClass.PRIVATE_METADATA,
            details=details,
        )
    )


def _emit_arm_measurements(
    writer: LedgerWriter,
    arm_id: str,
    *,
    wall_seconds: float | None,
    cold_start_seconds: float | None,
    input_samples: int | None,
    peak_rss_bytes: int | None,
    peak_vram_bytes: int | None,
) -> None:
    rtf = None
    if wall_seconds is not None and input_samples is not None and input_samples > 0:
        rtf = wall_seconds / (input_samples / 16_000.0)
    _measurement(writer, arm_id, "wall_time_seconds", "seconds", wall_seconds)
    _measurement(writer, arm_id, "rtf", "ratio", rtf)
    _measurement(writer, arm_id, "cold_start_seconds", "seconds", cold_start_seconds)
    _measurement(writer, arm_id, "peak_rss_bytes", "bytes", peak_rss_bytes)
    _measurement(
        writer,
        arm_id,
        "peak_vram_bytes",
        "bytes",
        peak_vram_bytes,
        vram_scope="device_delta",
    )


def _tool_identity() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    repo = directory.parents[1]
    production_sources = {
        *(repo / "yamabiko" / "asr").glob("*.py"),
        *(repo / "yamabiko" / "vad").glob("*.py"),
        repo / "yamabiko" / "__init__.py",
        repo / "yamabiko" / "capture" / "audio.py",
        repo / "yamabiko" / "capture" / "resampler.py",
        repo / "yamabiko" / "config.py",
        repo / "yamabiko" / "hf_env.py",
        repo / "yamabiko" / "models.py",
        repo / "yamabiko" / "paths.py",
    }
    sources = {*directory.glob("*.py"), *production_sources}
    if not sources or any(not path.is_file() for path in sources):
        raise RuntimeError("E0 tool source is incomplete")
    result: dict[str, str] = {}
    for path in sorted(sources, key=lambda item: item.relative_to(repo).as_posix()):
        relative = path.relative_to(repo).as_posix()
        result[relative] = _file_identity(path)[0]
    return result


def _arm_failure_reason(exc: BaseException) -> str:
    """Map the exception that stopped an arm to its terminal reason."""

    if isinstance(exc, DecoderParityError):
        return DiagnosticCode.DECODER_PARITY_MISMATCH.value
    if isinstance(exc, DecoderInstrumentationError):
        return DiagnosticCode.DECODER_INSTRUMENTATION.value
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return DiagnosticCode.RESOURCE_EXHAUSTED.value
    if isinstance(exc, (TypeError, ValueError)):
        return DiagnosticCode.HARNESS_VALIDATION.value
    return TerminalOutcome.UNKNOWN.value


def _finalize_noncomplete_run(
    writer: LedgerWriter,
    definitions: Sequence[ArmDefinition],
    *,
    active_kind: ArmKind | None,
    status: ArmStatus,
    reason: str,
    exception_class: str | None,
    completed_arm_ids: set[str],
    fate_attempted_arm_ids: set[str],
    reference_set: ReferenceSet,
    scored_events: Sequence[ScoredEvent],
    replay: ReplayResult | None,
    results_by_kind: Mapping[ArmKind, Sequence[_ArmResult]],
    arm_wall: Mapping[ArmKind, float],
    arm_samples: Mapping[ArmKind, int],
    arm_resources: Mapping[ArmKind, tuple[int | None, int | None]],
    cold_start_seconds: float | None,
) -> RunFinalManifest:
    for definition in definitions:
        if definition.arm_id in completed_arm_ids:
            continue
        terminal_status = (
            status if definition.kind is active_kind else ArmStatus.NOT_RUN
        )
        terminal_reason = (
            reason if definition.kind is active_kind else TerminalOutcome.UNKNOWN.value
        )
        if definition.arm_id not in fate_attempted_arm_ids:
            fate_attempted_arm_ids.add(definition.arm_id)
            try:
                _derive_and_emit_arm_fates(
                    writer,
                    definition,
                    terminal_status,
                    reference_set,
                    scored_events,
                    replay,
                    results_by_kind,
                )
            except Exception as exc:
                # Preserve the run final. The modern ledger verifier will make
                # missing/partial fate coverage explicit rather than allowing
                # this run to pass reporting.
                _LOGGER.warning(
                    "E0 %s event-fate derivation failed: %s",
                    definition.kind.value,
                    type(exc).__name__,
                )
        try:
            peak_rss_bytes, peak_vram_bytes = (
                arm_resources.get(definition.kind, (None, None))
                if definition.kind is active_kind
                else (None, None)
            )
            _emit_arm_measurements(
                writer,
                definition.arm_id,
                wall_seconds=arm_wall.get(definition.kind),
                cold_start_seconds=(
                    cold_start_seconds if definition.kind is ArmKind.P0 else None
                ),
                input_samples=arm_samples.get(definition.kind),
                peak_rss_bytes=peak_rss_bytes,
                peak_vram_bytes=peak_vram_bytes,
            )
        except Exception as exc:
            _LOGGER.warning(
                "E0 %s measurements failed: %s",
                definition.kind.value,
                type(exc).__name__,
            )
        try:
            writer.finish_arm(
                definition.arm_id,
                terminal_status,
                reason=terminal_reason,
                exception_class=(
                    exception_class if definition.kind is active_kind else None
                ),
            )
        except Exception as exc:
            _LOGGER.warning(
                "E0 %s terminal status failed: %s",
                definition.kind.value,
                type(exc).__name__,
            )
    return writer.finalize()


@dataclass(frozen=True, slots=True)
class ArmFailure:
    """The arm an exception stopped, its terminal reason, and the exception."""

    arm_kind: ArmKind
    reason: str
    exception_class: str


@dataclass(frozen=True, slots=True)
class RunOutcome:
    final: RunFinalManifest
    arm_failures: tuple[ArmFailure, ...] = ()


def run_request(
    request_path: Path,
    boundary: PrivateBoundary,
    *,
    repo: Path,
    argv: Sequence[str],
) -> RunOutcome:
    """Run a frozen request, preserving a final manifest on runtime failure."""

    # A runtime off the release lock is refused before anything is written.
    constraints_release_sha256 = _verify_release_lock(repo)
    request = load_run_request(request_path, boundary)
    master_manifest = load_registered_master(request.master_id, boundary)
    reference_manifest = load_registered_reference(request.reference_set_id, boundary)
    reference_texts = load_reference_texts(request.reference_set_id, boundary)
    canonical_master = load_canonical_master(master_manifest, boundary)
    reference_set = load_reference_set(reference_manifest, canonical_master)
    # P0 and P1 replay the production capture transform of the source-native
    # master; the canonical master stays the reference timeline.
    capture = load_capture_replay_stream(canonical_master, boundary)
    scored_events = scored_reference_events(reference_set, reference_texts)
    if reference_manifest.fixture_id not in request.cohort_fixture_ids:
        raise ValueError("run fixture is not declared by the frozen cohort")
    if (
        set(event.event_id for event in reference_set.events)
        != set(
            event_id
            for component in request.oracle_components
            for event_id in component.event_ids
        )
        and ArmKind.P1 in request.arms
    ):
        raise ValueError("P1 oracle components must partition every reference event")

    try:
        config_bytes = request.config_path.read_bytes()
    except OSError:
        raise ValueError("run config is unreadable") from None
    config_file_sha256 = hashlib.sha256(config_bytes).hexdigest()
    full_config = load_full_config(request.config_path)
    config_sha256 = canonical_sha256(full_config)
    construction_config: dict[str, object] | None = None
    if request.decoder_adapter is DecoderAdapter.FASTER_WHISPER:
        resolved_model_artifact = _require_loaded_model_artifact(
            request.model_artifact_path,
            full_config,
        )
        construction_config = _config_for_resolved_model(
            full_config,
            resolved_model_artifact,
        )
    else:
        # A candidate decoder is named by the request, not by the effective
        # config; the config still supplies production's VAD and boundary pad.
        resolved_model_artifact = require_fp32_artifact(request.model_artifact_path)
        onnx_asr_version, onnxruntime_version = nemo_tdt_runtime_versions()
        boundary_pad_ms = _production_boundary_pad_ms(full_config)
    model_artifact_manifest = _artifact_manifest(resolved_model_artifact)
    model_artifact_sha256 = canonical_sha256(model_artifact_manifest)
    decoder_adapter_identity = (
        {"name": request.decoder_adapter.value}
        if request.decoder_adapter is DecoderAdapter.FASTER_WHISPER
        else nemo_tdt_decoder_identity(
            model_artifact_manifest,
            onnx_asr_version=onnx_asr_version,
            onnxruntime_version=onnxruntime_version,
        )
    )
    # The system under test's VAD is hashed before any cohort or run record:
    # a declared candidate must match its declared digest, and an undeclared
    # request uses exactly the pinned file production loads.
    vad_artifact = (
        resolve_vad_artifact()
        if request.vad_artifact is None
        else resolve_vad_artifact(
            request.vad_artifact.path,
            request.vad_artifact.sha256,
        )
    )
    # The bundled sound tagger production fuses into Silero's probability is
    # pinned, not selectable. Its bytes and the assist's sources are checked
    # before the cohort is frozen.
    tagger_artifact = resolve_tagger_artifact()
    vad_assist = _verify_vad_assist_contract(tagger_artifact.sha256)
    (
        cohort_manifest_sha256,
        arm_profile_sha256,
        cohort_manifest,
    ) = _cohort_identity(
        request,
        partition=reference_manifest.partition,
        config_sha256=config_sha256,
        model_artifact_sha256=model_artifact_sha256,
        vad_artifact=vad_artifact,
        tagger_artifact=tagger_artifact,
        decoder_adapter=decoder_adapter_identity,
    )
    _freeze_cohort_manifest(cohort_manifest, boundary)
    dependencies = _dependency_identity()
    gpu = _gpu_identity()
    silero_vad_source_sha256 = _verify_silero_vad_contract()
    runtime_identity = {
        "dependencies": dependencies,
        "gpu": gpu,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "silero_chunker_source_sha256": vad_assist["silero_chunker_source_sha256"],
        "silero_vad_source_sha256": silero_vad_source_sha256,
        "sound_tagger_source_sha256": vad_assist["sound_tagger_source_sha256"],
    }
    model_runtime_sha256 = canonical_sha256(
        {
            "model_artifact_sha256": model_artifact_sha256,
            "vad_artifact_sha256": vad_artifact.sha256,
            "decoder_adapter": decoder_adapter_identity,
            "runtime": runtime_identity,
        }
    )
    oracle_component_hashes = {
        component.component_id: oracle_component_map_sha256(
            component.component_id,
            reference_manifest,
            canonical_master,
            component.event_ids,
        )
        for component in request.oracle_components
    }
    oracle_component_declarations = tuple(
        {
            "oracle_component_id": component.component_id,
            "oracle_component_map_sha256": oracle_component_hashes[
                component.component_id
            ],
            "oracle_event_ids": component.event_ids,
            "oracle_reference_set_id": reference_manifest.reference_set_id,
            "oracle_reference_set_sha256": reference_manifest.reference_set_sha256,
        }
        for component in request.oracle_components
    )
    definitions = _arm_definitions(
        request,
        config_sha256=config_sha256,
        model_artifact_sha256=model_artifact_sha256,
        vad_artifact_sha256=vad_artifact.sha256,
        model_runtime_sha256=model_runtime_sha256,
        cohort_manifest_sha256=cohort_manifest_sha256,
        arm_profile_sha256=arm_profile_sha256,
        oracle_components=oracle_component_declarations,
    )
    arm_by_kind = {definition.kind: definition for definition in definitions}
    run_id = new_opaque_id()
    start = RunStartManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        session_id=new_opaque_id(),
        clock_id="perf_counter_ns",
        canonical_stream_id=new_opaque_id(),
        private_text_stream_id=new_opaque_id(),
        arm_definitions=definitions,
        classification=PrivateContentClass.PRIVATE_METADATA,
        details={
            "request_id": request.request_id,
            "master_id": request.master_id,
            "master_pcm_sha256": master_manifest.canonical_pcm_sha256,
            "capture_replay": capture.provenance,
            "fixture_id": reference_manifest.fixture_id,
            "reference_set_id": request.reference_set_id,
            "reference_set_sha256": reference_manifest.reference_set_sha256,
            "reference_text_sha256": reference_manifest.reference_text_sha256,
            "reference_event_ids": tuple(
                event.event_id for event in reference_set.events
            ),
            "approved_contracts": contracts.approved_contracts(),
            "event_fates_required": True,
            "partition": reference_manifest.partition.value,
            "cohort_id": request.cohort_id,
            "cohort_fixture_ids": request.cohort_fixture_ids,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "hard_stratum": cohort_manifest["hard_stratum"],
            "config_file_sha256": config_file_sha256,
            "config_sha256": config_sha256,
            "effective_config": full_config,
            "model_artifact_path": str(resolved_model_artifact),
            "model_artifact_manifest": model_artifact_manifest,
            "model_artifact_sha256": model_artifact_sha256,
            "vad_artifact": vad_artifact.identity(),
            "decoder_adapter": decoder_adapter_identity,
            "model_runtime_sha256": model_runtime_sha256,
            "runtime_identity": runtime_identity,
            "constraints_release_sha256": constraints_release_sha256,
            "git": _git_identity(repo),
            "tool_source_sha256": _tool_identity(),
            "arm_definition_sha256": {
                definition.arm_id: arm_definition_sha256(definition)
                for definition in definitions
            },
            "arm_profile_sha256": {
                request.arm_ids[kind]: arm_profile_sha256[kind] for kind in request.arms
            },
            "command_argv": list(argv),
        },
    )
    run_dir = require_private_path(boundary.root / "runs" / run_id, boundary)
    # This is intentionally before VAD or ASR construction.
    writer = LedgerWriter.create(run_dir, start, boundary=boundary)
    sampler = _ResourceSampler()
    cold_start_seconds: float | None = None
    arm_wall: dict[ArmKind, float] = {}
    arm_samples: dict[ArmKind, int] = {}
    arm_resources: dict[ArmKind, tuple[int | None, int | None]] = {}
    replay: ReplayResult | None = None
    results_by_kind: dict[ArmKind, list[_ArmResult]] = {
        kind: [] for kind in request.arms
    }
    completed_arm_ids: set[str] = set()
    fate_attempted_arm_ids: set[str] = set()
    active_kind: ArmKind | None = request.arms[0]
    sampler_started = False
    try:
        sampler.start()
        sampler_started = True
        cold_start_begin = time.perf_counter()
        chunker_config, threshold = production_chunker_config(full_config)
        vad_model = load_production_model(chunker_config, vad_artifact)
        tagger = load_production_tagger(tagger_artifact)
        nemo_decoder: NemoTdtDecoder | None = None
        if construction_config is not None:
            built = build_asr(construction_config["asr"])
            if not isinstance(built, FasterWhisperASR):
                raise TypeError("production ASR factory did not return FasterWhisperASR")
            if _hash_artifact(resolved_model_artifact) != model_artifact_sha256:
                raise RuntimeError("configured model artifact changed during construction")
            asr = bind_model_artifact(built, model_artifact_sha256)
        else:
            nemo_decoder = load_nemo_tdt(
                resolved_model_artifact,
                model_artifact_sha256,
                boundary_pad_ms=boundary_pad_ms,
            )
            if _hash_artifact(resolved_model_artifact) != model_artifact_sha256:
                raise RuntimeError("candidate model artifact changed during construction")
        cold_start_seconds = time.perf_counter() - cold_start_begin

        for kind in request.arms:
            active_kind = kind
            sampler.reset()
            begin = time.perf_counter()
            sink = _ArmSink(writer, arm_by_kind[kind].arm_id)
            results = results_by_kind[kind]
            if kind is ArmKind.P0:
                replay = replay_master(
                    capture,
                    chunker_config,
                    threshold,
                    vad_model,
                    tagger,
                    writer,
                )
                for window in replay.windows:
                    results.append(
                        run_p0(window, asr, sink)
                        if nemo_decoder is None
                        else run_nemo_p0(window, nemo_decoder, sink)
                    )
            elif kind is ArmKind.P1:
                for component_request in request.oracle_components:
                    component = OracleComponent(
                        oracle_component_id=component_request.component_id,
                        reference_set_manifest=reference_manifest,
                        master=canonical_master,
                        event_ids=component_request.event_ids,
                        component_map_sha256=oracle_component_hashes[
                            component_request.component_id
                        ],
                    )
                    results.append(
                        run_p1(component, capture, asr, sink)
                        if nemo_decoder is None
                        else run_nemo_p1(component, capture, nemo_decoder, sink)
                    )
            elif kind is ArmKind.P2A:
                for parent in results_by_kind[ArmKind.P0]:
                    results.append(run_p2a(parent.call, asr._model, sink))
            else:
                for parent in results_by_kind[ArmKind.P2A]:
                    results.append(run_p2b(parent.call, asr._model, sink))
            arm_wall[kind] = time.perf_counter() - begin
            arm_samples[kind] = sum(
                result.call.decoder_audio.size for result in results
            )
            if kind is request.arms[-1]:
                sampler.stop()
                sampler_started = False
            arm_resources[kind] = sampler.snapshot()
            definition = arm_by_kind[kind]
            fate_attempted_arm_ids.add(definition.arm_id)
            _derive_and_emit_arm_fates(
                writer,
                definition,
                ArmStatus.COMPLETE,
                reference_set,
                scored_events,
                replay,
                results_by_kind,
            )
            _emit_arm_measurements(
                writer,
                definition.arm_id,
                wall_seconds=arm_wall[kind],
                cold_start_seconds=(cold_start_seconds if kind is ArmKind.P0 else None),
                input_samples=arm_samples[kind],
                peak_rss_bytes=arm_resources[kind][0],
                peak_vram_bytes=arm_resources[kind][1],
            )
            writer.finish_arm(definition.arm_id, ArmStatus.COMPLETE)
            completed_arm_ids.add(definition.arm_id)
            active_kind = None
        return RunOutcome(writer.finalize())
    except KeyboardInterrupt:
        if sampler_started:
            try:
                sampler.stop()
            except Exception:
                pass
        if active_kind is not None and active_kind not in arm_resources:
            arm_resources[active_kind] = sampler.snapshot()
        return RunOutcome(
            _finalize_noncomplete_run(
                writer,
                definitions,
                active_kind=active_kind,
                status=ArmStatus.CANCELLED,
                reason=DiagnosticCode.SHUTDOWN_CANCELLED.value,
                exception_class=None,
                completed_arm_ids=completed_arm_ids,
                fate_attempted_arm_ids=fate_attempted_arm_ids,
                reference_set=reference_set,
                scored_events=scored_events,
                replay=replay,
                results_by_kind=results_by_kind,
                arm_wall=arm_wall,
                arm_samples=arm_samples,
                arm_resources=arm_resources,
                cold_start_seconds=cold_start_seconds,
            )
        )
    except Exception as exc:
        if sampler_started:
            try:
                sampler.stop()
            except Exception:
                pass
        if active_kind is not None and active_kind not in arm_resources:
            arm_resources[active_kind] = sampler.snapshot()
        reason = _arm_failure_reason(exc)
        exception_class = type(exc).__name__
        final = _finalize_noncomplete_run(
            writer,
            definitions,
            active_kind=active_kind,
            status=ArmStatus.FAILED,
            reason=reason,
            exception_class=exception_class,
            completed_arm_ids=completed_arm_ids,
            fate_attempted_arm_ids=fate_attempted_arm_ids,
            reference_set=reference_set,
            scored_events=scored_events,
            replay=replay,
            results_by_kind=results_by_kind,
            arm_wall=arm_wall,
            arm_samples=arm_samples,
            arm_resources=arm_resources,
            cold_start_seconds=cold_start_seconds,
        )
        failures = (
            ()
            if active_kind is None
            else (ArmFailure(active_kind, reason, exception_class),)
        )
        return RunOutcome(final, failures)


__all__ = (
    "COHORT_SCHEMA_VERSION",
    "DEFAULT_HARD_STRATUM_TAGS",
    "HARD_STRATUM_VERSION",
    "RUN_REQUEST_SCHEMA_VERSION",
    "ArmFailure",
    "OracleComponentRequest",
    "RunOutcome",
    "RunRequest",
    "VadArtifactRequest",
    "load_run_request",
    "run_request",
)
