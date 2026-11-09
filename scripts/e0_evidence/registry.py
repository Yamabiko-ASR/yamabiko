"""Strict private manifest codecs and registry transactions for E0."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence, cast

from scripts.e0_evidence._common import atomic_write_bytes, require_uuid4
from scripts.e0_evidence.assets import require_local_evaluation
from scripts.e0_evidence.normalize_ja import normalize_minimal
from scripts.e0_evidence.pcm import (
    MasterManifest,
    MasterTransformDag,
    TransformStep,
    load_canonical_master,
)
from scripts.e0_evidence.private_root import (
    PRIVATE_ROOT_ENV,
    ROOT_MANIFEST_NAME,
    PrivateBoundary,
    discover_git_worktrees,
    parse_root_manifest,
    persisted_excluded_roots,
    require_private_path,
    resolve_private_root,
)
from scripts.e0_evidence.references import (
    Adjudication,
    AnnotationInputRole,
    AnnotationPass,
    EditorialSubtitle,
    NonSpeechControl,
    ReferenceEvent,
    ReferenceSetManifest,
    VerbatimReference,
    adjudicated_reference_content_sha256,
    load_reference_set,
    reference_content_sha256,
    reference_set_manifest_sha256,
)
from scripts.e0_evidence.schema import (
    SCHEMA_VERSION,
    CapabilityDecision,
    FixtureManifest,
    Partition,
    RightsRecord,
    SampleSpan,
    new_opaque_id,
)
from scripts.e0_evidence.wire import canonical_json_bytes, canonical_sha256


_ROOT_DIRECTORIES = (
    "cohorts",
    "fixtures",
    "rights",
    "masters",
    "references",
    "runs",
    "reports",
    "receipts",
    "export-reviews",
)
_ZERO_SHA256 = "0" * 64
_RIGHTS_REVIEW_SCHEMA_VERSION = "e0-rights-review-1.0"
_RIGHTS_CAPABILITIES = (
    "local_evaluation",
    "redistribution",
    "training",
    "transcript_publication",
)


def _mapping(
    value: object,
    fields: Iterable[str],
    name: str,
) -> dict[str, object]:
    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has an invalid wire shape")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} has a non-string field")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    items = _list(value, name)
    if any(type(item) is not str or not item for item in items):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(items)  # type: ignore[return-value]


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{name} is unreadable") from None


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    data = canonical_json_bytes(payload, newline=True)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_bytes(path, data)
    except OSError:
        raise ValueError("private registry write failed") from None


def _contains(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _overlaps(first: Path, second: Path) -> bool:
    return _contains(first, second) or _contains(second, first)


def _resolved_absolute(path: Path, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be absolute")
    try:
        return candidate.resolve(strict=False)
    except OSError:
        raise ValueError(f"{name} cannot be resolved") from None


def initialize_private_root(
    repo: Path,
    environ: Mapping[str, str],
    *,
    excluded_roots: Sequence[Path] = (),
) -> PrivateBoundary:
    """Create the fixed private layout only after proving it is outside Git.

    The manifest persists every excluded root ever passed here; a later run
    adds new roots and never removes one.
    """

    raw = environ.get(PRIVATE_ROOT_ENV)
    if type(raw) is not str or not raw.strip():
        raise ValueError(f"{PRIVATE_ROOT_ENV} is required")
    root = _resolved_absolute(Path(raw), "private root")
    repository = _resolved_absolute(repo, "repository")
    worktrees = tuple(dict.fromkeys((repository, *discover_git_worktrees(repo))))
    excluded = tuple(
        _resolved_absolute(item, "excluded root") for item in excluded_roots
    )
    if any(_overlaps(root, worktree) for worktree in worktrees):
        raise ValueError("private root overlaps a Git worktree")
    if any(_overlaps(root, denied) for denied in excluded):
        raise ValueError("private root overlaps an excluded root")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ValueError("private root cannot be created") from None
    manifest_path = require_private_path(
        root / ROOT_MANIFEST_NAME,
        PrivateBoundary(root, worktrees, excluded),
    )
    existing: object = None
    if manifest_path.exists():
        existing = _read_json(manifest_path, "private root manifest")
        root_id, persisted = parse_root_manifest(existing)
    else:
        root_id, persisted = new_opaque_id(), ()
    union = tuple(sorted({*persisted, *excluded}, key=str))
    boundary = PrivateBoundary(root, worktrees, union)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "private_root_id": root_id,
        "excluded_roots": [str(item) for item in union],
    }
    if existing != manifest:
        _write_canonical_json(manifest_path, manifest)
    for name in _ROOT_DIRECTORIES:
        try:
            require_private_path(root / name, boundary).mkdir(exist_ok=True)
        except OSError:
            raise ValueError("private root layout cannot be created") from None
    return boundary


def boundary_from_environment(
    repo: Path,
    environ: Mapping[str, str],
    *,
    excluded_roots: Sequence[Path] = (),
) -> PrivateBoundary:
    """Build the boundary from the environment and every persisted exclusion.

    ``excluded_roots`` adds to the roots ``root-init`` persisted; it cannot
    remove one.
    """

    worktrees = discover_git_worktrees(repo)
    root = resolve_private_root(
        repo,
        environ,
        worktrees=worktrees,
        excluded_roots=excluded_roots,
    )
    excluded = tuple(
        dict.fromkeys(
            (
                *persisted_excluded_roots(root),
                *(_resolved_absolute(item, "excluded root") for item in excluded_roots),
            )
        )
    )
    return PrivateBoundary(
        root=root,
        worktrees=tuple(dict.fromkeys((Path(repo), *worktrees))),
        excluded_roots=excluded,
    )


def rights_to_wire(rights: RightsRecord) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rights_record_id": rights.rights_record_id,
        "fixture_id": rights.fixture_id,
        "local_evaluation": rights.local_evaluation.value,
        "redistribution": rights.redistribution.value,
        "training": rights.training.value,
        "transcript_publication": rights.transcript_publication.value,
        "reviewer_id": rights.reviewer_id or None,
        "evidence_ids": list(rights.evidence_ids),
        "retention_version": rights.retention_version,
    }


def rights_from_wire(value: object) -> RightsRecord:
    item = _mapping(
        value,
        (
            "schema_version",
            "rights_record_id",
            "fixture_id",
            "local_evaluation",
            "redistribution",
            "training",
            "transcript_publication",
            "reviewer_id",
            "evidence_ids",
            "retention_version",
        ),
        "rights record",
    )
    if item["schema_version"] != SCHEMA_VERSION:
        raise ValueError("rights record schema version is unsupported")
    reviewer = item["reviewer_id"]
    if reviewer is not None and type(reviewer) is not str:
        raise ValueError("reviewer_id must be null or a string")
    try:
        return RightsRecord(
            rights_record_id=item["rights_record_id"],  # type: ignore[arg-type]
            fixture_id=item["fixture_id"],  # type: ignore[arg-type]
            local_evaluation=CapabilityDecision(item["local_evaluation"]),
            redistribution=CapabilityDecision(item["redistribution"]),
            training=CapabilityDecision(item["training"]),
            transcript_publication=CapabilityDecision(item["transcript_publication"]),
            reviewer_id=reviewer or "",
            evidence_ids=_strings(item["evidence_ids"], "evidence_ids"),
            retention_version=item["retention_version"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        raise ValueError("rights record is invalid") from None


def _rights_path(rights_record_id: str, boundary: PrivateBoundary) -> Path:
    require_uuid4(rights_record_id, "rights_record_id")
    return require_private_path(
        boundary.root / "rights" / f"{rights_record_id}.private.json",
        boundary,
    )


def _rights_history_path(rights_record_id: str, boundary: PrivateBoundary) -> Path:
    require_uuid4(rights_record_id, "rights_record_id")
    return require_private_path(
        boundary.root / "rights" / f"{rights_record_id}.reviews.private.jsonl",
        boundary,
    )


def _load_rights_history(
    rights_record_id: str,
    fixture_id: str,
    retention_version: str,
    boundary: PrivateBoundary,
) -> tuple[tuple[dict[str, object], ...], str]:
    path = _rights_history_path(rights_record_id, boundary)
    if not path.exists():
        return (), _ZERO_SHA256
    try:
        raw = path.read_bytes()
    except OSError:
        raise ValueError("rights review history is unreadable") from None
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("rights review history is truncated")
    predecessor = _ZERO_SHA256
    seen_capabilities: set[str] = set()
    records: list[dict[str, object]] = []
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("rights review history is invalid") from None
        record = _mapping(
            value,
            (
                "schema_version",
                "review_id",
                "rights_record_id",
                "fixture_id",
                "capability",
                "decision",
                "reviewer_id",
                "evidence_ids",
                "retention_version",
                "predecessor_sha256",
            ),
            "rights review history record",
        )
        if canonical_json_bytes(record, newline=True) != line:
            raise ValueError("rights review history is not canonical")
        require_uuid4(record["review_id"], "review_id")
        if (
            record["schema_version"] != _RIGHTS_REVIEW_SCHEMA_VERSION
            or record["rights_record_id"] != rights_record_id
            or record["fixture_id"] != fixture_id
            or record["retention_version"] != retention_version
            or record["predecessor_sha256"] != predecessor
        ):
            raise ValueError("rights review history identity is invalid")
        capability = record["capability"]
        if capability not in _RIGHTS_CAPABILITIES or capability in seen_capabilities:
            raise ValueError("rights capability review is duplicated or unsupported")
        try:
            decision = CapabilityDecision(record["decision"])
        except (TypeError, ValueError):
            raise ValueError("rights review decision is invalid") from None
        if decision is CapabilityDecision.PENDING:
            raise ValueError("rights review history cannot record pending")
        require_uuid4(record["reviewer_id"], "reviewer_id")
        evidence_ids = _strings(record["evidence_ids"], "evidence_ids")
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("rights review evidence is invalid")
        for evidence_id in evidence_ids:
            require_uuid4(evidence_id, "evidence_id")
        seen_capabilities.add(capability)  # type: ignore[arg-type]
        records.append(record)
        predecessor = hashlib.sha256(line).hexdigest()
    return tuple(records), predecessor


def _effective_rights(
    stored: RightsRecord,
    history: Sequence[Mapping[str, object]],
) -> RightsRecord:
    if not history:
        return stored
    decisions = {
        capability: CapabilityDecision.PENDING for capability in _RIGHTS_CAPABILITIES
    }
    for review in history:
        capability = cast(str, review["capability"])
        decision = CapabilityDecision(review["decision"])
        if decisions[capability] is not CapabilityDecision.PENDING:
            raise ValueError("rights history repeats a capability decision")
        decisions[capability] = decision
    latest = history[-1]
    latest_reviewer_id = cast(str, latest["reviewer_id"])
    latest_evidence_ids = tuple(cast(list[str], latest["evidence_ids"]))
    if (
        any(
            getattr(stored, capability) is not decisions[capability]
            for capability in _RIGHTS_CAPABILITIES
        )
        or stored.reviewer_id != latest_reviewer_id
        or stored.evidence_ids != latest_evidence_ids
    ):
        raise ValueError("rights summary does not match append-only history")
    return RightsRecord(
        rights_record_id=stored.rights_record_id,
        fixture_id=stored.fixture_id,
        local_evaluation=decisions["local_evaluation"],
        redistribution=decisions["redistribution"],
        training=decisions["training"],
        transcript_publication=decisions["transcript_publication"],
        reviewer_id=latest_reviewer_id,
        evidence_ids=latest_evidence_ids,
        retention_version=stored.retention_version,
    )


def _load_rights(rights_record_id: str, boundary: PrivateBoundary) -> RightsRecord:
    stored = rights_from_wire(
        _read_json(_rights_path(rights_record_id, boundary), "rights record")
    )
    history, _ = _load_rights_history(
        stored.rights_record_id,
        stored.fixture_id,
        stored.retention_version,
        boundary,
    )
    return _effective_rights(stored, history)


@contextmanager
def _rights_review_reservation(
    rights_record_id: str,
    boundary: PrivateBoundary,
) -> Iterable[None]:
    lock_path = require_private_path(
        boundary.root / "rights" / f"{rights_record_id}.review.lock",
        boundary,
    )
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("another rights review is already in progress") from None
    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def apply_rights_review(
    review_path: Path,
    boundary: PrivateBoundary,
) -> RightsRecord:
    review = _mapping(
        _read_json(
            require_private_path(review_path, boundary),
            "rights review",
        ),
        (
            "schema_version",
            "rights_record_id",
            "fixture_id",
            "local_evaluation",
            "redistribution",
            "training",
            "transcript_publication",
            "reviewer_id",
            "evidence_ids",
            "retention_version",
        ),
        "rights review",
    )
    proposed = rights_from_wire(review)
    with _rights_review_reservation(proposed.rights_record_id, boundary):
        current = _load_rights(proposed.rights_record_id, boundary)
        if proposed.fixture_id != current.fixture_id:
            raise ValueError("rights review belongs to a different fixture")
        changes: list[str] = []
        for field_name in _RIGHTS_CAPABILITIES:
            before = getattr(current, field_name)
            after = getattr(proposed, field_name)
            if before is not CapabilityDecision.PENDING and after is not before:
                raise ValueError("a reviewed capability decision is immutable")
            if (
                before is CapabilityDecision.PENDING
                and after is not CapabilityDecision.PENDING
            ):
                changes.append(field_name)
        if proposed.retention_version != current.retention_version:
            raise ValueError("retention changes require a new rights record")
        if not changes:
            raise ValueError("rights review does not decide a pending capability")

        history, predecessor = _load_rights_history(
            current.rights_record_id,
            current.fixture_id,
            current.retention_version,
            boundary,
        )
        reviewed_in_history = {cast(str, record["capability"]) for record in history}
        additions: list[bytes] = []

        def append_review(
            capability: str,
            decision: CapabilityDecision,
            reviewer_id: str,
            evidence_ids: Sequence[str],
        ) -> None:
            nonlocal predecessor
            record = {
                "schema_version": _RIGHTS_REVIEW_SCHEMA_VERSION,
                "review_id": new_opaque_id(),
                "rights_record_id": current.rights_record_id,
                "fixture_id": current.fixture_id,
                "capability": capability,
                "decision": decision.value,
                "reviewer_id": reviewer_id,
                "evidence_ids": list(evidence_ids),
                "retention_version": current.retention_version,
                "predecessor_sha256": predecessor,
            }
            line = canonical_json_bytes(record, newline=True)
            additions.append(line)
            predecessor = hashlib.sha256(line).hexdigest()

        # Migrate any pre-history reviewed decisions before the summary's
        # single reviewer/evidence fields are replaced by a later review.
        for capability in _RIGHTS_CAPABILITIES:
            decision = getattr(current, capability)
            if (
                decision is not CapabilityDecision.PENDING
                and capability not in reviewed_in_history
            ):
                append_review(
                    capability,
                    decision,
                    current.reviewer_id,
                    current.evidence_ids,
                )
        for capability in changes:
            append_review(
                capability,
                getattr(proposed, capability),
                proposed.reviewer_id,
                proposed.evidence_ids,
            )

        history_path = _rights_history_path(current.rights_record_id, boundary)
        payload = b"".join(additions)
        try:
            with history_path.open("ab") as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError("short rights review history write")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            raise ValueError("rights review history cannot be appended") from None
        # History is authoritative and is committed first. If this replace is
        # interrupted, the summary/history mismatch makes future loads fail closed.
        _write_canonical_json(
            _rights_path(current.rights_record_id, boundary), rights_to_wire(proposed)
        )
        return _load_rights(current.rights_record_id, boundary)


_TRANSFORM_FIELDS = (
    "transform_id",
    "transform_version",
    "transform_sha256",
    "parent_artifact_id",
    "parent_byte_sha256",
    "output_artifact_id",
    "output_byte_sha256",
    "tool",
    "tool_version",
    "command_argv",
    "filter_graph",
    "stream_index",
    "excerpt_start_samples",
    "excerpt_end_samples",
    "input_sample_rate_hz",
    "input_channels",
    "input_layout",
    "output_sample_rate_hz",
    "output_channels",
    "output_layout",
    "output_sample_count",
    "channel_operation",
    "resampler",
    "resampler_parameters",
    "dtype_conversion",
)


def _transform_from_wire(value: object, name: str) -> TransformStep:
    item = _mapping(value, _TRANSFORM_FIELDS, name)
    parameters = item["resampler_parameters"]
    if not isinstance(parameters, dict):
        raise ValueError(f"{name}.resampler_parameters must be an object")
    try:
        return TransformStep(
            transform_id=item["transform_id"],  # type: ignore[arg-type]
            transform_version=item["transform_version"],  # type: ignore[arg-type]
            transform_sha256=item["transform_sha256"],  # type: ignore[arg-type]
            parent_artifact_id=item["parent_artifact_id"],  # type: ignore[arg-type]
            parent_byte_sha256=item["parent_byte_sha256"],  # type: ignore[arg-type]
            output_artifact_id=item["output_artifact_id"],  # type: ignore[arg-type]
            output_byte_sha256=item["output_byte_sha256"],  # type: ignore[arg-type]
            tool=item["tool"],  # type: ignore[arg-type]
            tool_version=item["tool_version"],  # type: ignore[arg-type]
            command_argv=_strings(item["command_argv"], f"{name}.command_argv"),
            filter_graph=item["filter_graph"],  # type: ignore[arg-type]
            stream_index=item["stream_index"],  # type: ignore[arg-type]
            excerpt_start_samples=item["excerpt_start_samples"],  # type: ignore[arg-type]
            excerpt_end_samples=item["excerpt_end_samples"],  # type: ignore[arg-type]
            input_sample_rate_hz=item["input_sample_rate_hz"],  # type: ignore[arg-type]
            input_channels=item["input_channels"],  # type: ignore[arg-type]
            input_layout=item["input_layout"],  # type: ignore[arg-type]
            output_sample_rate_hz=item["output_sample_rate_hz"],  # type: ignore[arg-type]
            output_channels=item["output_channels"],  # type: ignore[arg-type]
            output_layout=item["output_layout"],  # type: ignore[arg-type]
            output_sample_count=item["output_sample_count"],  # type: ignore[arg-type]
            channel_operation=item["channel_operation"],  # type: ignore[arg-type]
            resampler=item["resampler"],  # type: ignore[arg-type]
            resampler_parameters=parameters,
            dtype_conversion=item["dtype_conversion"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        raise ValueError(f"{name} is invalid") from None


def _transform_to_wire(step: TransformStep) -> dict[str, object]:
    return {
        field_name: (
            list(value)
            if isinstance(value, tuple)
            else dict(value)
            if isinstance(value, Mapping)
            else value
        )
        for field_name in _TRANSFORM_FIELDS
        for value in (getattr(step, field_name),)
    }


_MASTER_FIELDS = (
    "schema_version",
    "master_id",
    "fixture_id",
    "rights",
    "partition",
    "timeline_id",
    "parent_artifact_ids",
    "byte_sha256",
    "canonical_pcm_sha256",
    "sample_rate_hz",
    "channels",
    "layout",
    "dtype",
    "sample_count",
    "duration_samples",
    "transform_dag",
    "provenance_class",
    "rights_class",
    "retention_version",
    "annotation_set_sha256",
)


def master_manifest_from_wire(value: object) -> MasterManifest:
    item = _mapping(value, _MASTER_FIELDS, "master manifest")
    dag = _mapping(
        item["transform_dag"],
        ("source_native", "canonical"),
        "master transform DAG",
    )
    try:
        return MasterManifest(
            schema_version=item["schema_version"],  # type: ignore[arg-type]
            master_id=item["master_id"],  # type: ignore[arg-type]
            fixture_id=item["fixture_id"],  # type: ignore[arg-type]
            rights=rights_from_wire(item["rights"]),
            partition=Partition(item["partition"]),
            timeline_id=item["timeline_id"],  # type: ignore[arg-type]
            parent_artifact_ids=_strings(
                item["parent_artifact_ids"], "parent_artifact_ids"
            ),
            byte_sha256=item["byte_sha256"],  # type: ignore[arg-type]
            canonical_pcm_sha256=item["canonical_pcm_sha256"],  # type: ignore[arg-type]
            sample_rate_hz=item["sample_rate_hz"],  # type: ignore[arg-type]
            channels=item["channels"],  # type: ignore[arg-type]
            layout=item["layout"],  # type: ignore[arg-type]
            dtype=item["dtype"],  # type: ignore[arg-type]
            sample_count=item["sample_count"],  # type: ignore[arg-type]
            duration_samples=item["duration_samples"],  # type: ignore[arg-type]
            transform_dag=MasterTransformDag(
                source_native=_transform_from_wire(
                    dag["source_native"], "source-native transform"
                ),
                canonical=_transform_from_wire(dag["canonical"], "canonical transform"),
            ),
            provenance_class=item["provenance_class"],  # type: ignore[arg-type]
            rights_class=item["rights_class"],  # type: ignore[arg-type]
            retention_version=item["retention_version"],  # type: ignore[arg-type]
            annotation_set_sha256=item["annotation_set_sha256"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        raise ValueError("master manifest is invalid") from None


def master_manifest_to_wire(manifest: MasterManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "master_id": manifest.master_id,
        "fixture_id": manifest.fixture_id,
        "rights": rights_to_wire(manifest.rights),
        "partition": manifest.partition.value,
        "timeline_id": manifest.timeline_id,
        "parent_artifact_ids": list(manifest.parent_artifact_ids),
        "byte_sha256": manifest.byte_sha256,
        "canonical_pcm_sha256": manifest.canonical_pcm_sha256,
        "sample_rate_hz": manifest.sample_rate_hz,
        "channels": manifest.channels,
        "layout": manifest.layout,
        "dtype": manifest.dtype,
        "sample_count": manifest.sample_count,
        "duration_samples": manifest.duration_samples,
        "transform_dag": {
            "source_native": _transform_to_wire(manifest.transform_dag.source_native),
            "canonical": _transform_to_wire(manifest.transform_dag.canonical),
        },
        "provenance_class": manifest.provenance_class,
        "rights_class": manifest.rights_class,
        "retention_version": manifest.retention_version,
        "annotation_set_sha256": manifest.annotation_set_sha256,
    }


def _locator(boundary: PrivateBoundary) -> tuple[Path, dict[str, object]]:
    path = require_private_path(boundary.root / "locator.private.json", boundary)
    value = _mapping(
        _read_json(path, "private locator"),
        ("schema_version", "fixtures"),
        "private locator",
    )
    if value["schema_version"] != SCHEMA_VERSION or not isinstance(
        value["fixtures"], dict
    ):
        raise ValueError("private locator is invalid")
    return path, value


def register_master(
    manifest_path: Path,
    source_native_path: Path,
    canonical_path: Path,
    boundary: PrivateBoundary,
) -> MasterManifest:
    manifest = master_manifest_from_wire(
        _read_json(
            require_private_path(manifest_path, boundary),
            "master manifest",
        )
    )
    current_rights = _load_rights(manifest.rights.rights_record_id, boundary)
    if manifest.rights != current_rights:
        raise ValueError("master manifest does not carry the current rights record")
    require_local_evaluation(current_rights)
    source_native = require_private_path(source_native_path, boundary)
    canonical = require_private_path(canonical_path, boundary)
    if not source_native.is_file() or not canonical.is_file():
        raise ValueError("master inputs must be private regular files")
    locator_path, original = _locator(boundary)
    fixtures = dict(original["fixtures"])  # type: ignore[arg-type]
    entry = fixtures.get(manifest.fixture_id)
    if not isinstance(entry, dict):
        raise ValueError("private locator has no fixture entry")
    updated_entry = dict(entry)
    masters = dict(updated_entry.get("masters", {}))
    bindings = {
        manifest.transform_dag.source_native.output_artifact_id: str(source_native),
        manifest.master_id: str(canonical),
    }
    for artifact_id, path_value in bindings.items():
        existing = masters.get(artifact_id)
        if existing is not None and existing != path_value:
            raise ValueError("master artifact identity is already bound")
        masters[artifact_id] = path_value
    updated_entry["masters"] = masters
    fixtures[manifest.fixture_id] = updated_entry
    updated = {
        "schema_version": SCHEMA_VERSION,
        "fixtures": fixtures,
    }
    _write_canonical_json(locator_path, updated)
    try:
        load_canonical_master(manifest, boundary)
    except BaseException:
        _write_canonical_json(locator_path, original)
        raise
    target = require_private_path(
        boundary.root / "masters" / manifest.master_id / "master.private.json",
        boundary,
    )
    if target.exists():
        existing = master_manifest_from_wire(
            _read_json(target, "registered master manifest")
        )
        if existing != manifest:
            _write_canonical_json(locator_path, original)
            raise ValueError("master manifest is immutable once registered")
        return manifest
    try:
        _write_canonical_json(target, master_manifest_to_wire(manifest))
    except BaseException:
        _write_canonical_json(locator_path, original)
        raise
    return manifest


def load_registered_master(master_id: str, boundary: PrivateBoundary) -> MasterManifest:
    require_uuid4(master_id, "master_id")
    path = require_private_path(
        boundary.root / "masters" / master_id / "master.private.json",
        boundary,
    )
    manifest = master_manifest_from_wire(_read_json(path, "registered master manifest"))
    if manifest.master_id != master_id:
        raise ValueError("registered master identity does not match its path")
    return manifest


_FIXTURE_FIELDS = (
    "schema_version",
    "fixture_id",
    "partition",
    "title_id",
    "rights_record_id",
)


def load_registered_fixture(
    fixture_id: str,
    boundary: PrivateBoundary,
) -> FixtureManifest:
    """Read the fixture record that ``fixture-register`` wrote."""

    require_uuid4(fixture_id, "fixture_id")
    path = require_private_path(
        boundary.root / "fixtures" / fixture_id / "fixture.private.json",
        boundary,
    )
    item = _mapping(
        _read_json(path, "registered fixture manifest"),
        _FIXTURE_FIELDS,
        "registered fixture manifest",
    )
    if item["schema_version"] != SCHEMA_VERSION:
        raise ValueError("registered fixture manifest is invalid")
    try:
        manifest = FixtureManifest(
            fixture_id=item["fixture_id"],  # type: ignore[arg-type]
            partition=Partition(item["partition"]),
            title_id=item["title_id"],  # type: ignore[arg-type]
            rights_record_id=item["rights_record_id"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        raise ValueError("registered fixture manifest is invalid") from None
    if manifest.fixture_id != fixture_id:
        raise ValueError("registered fixture identity does not match its path")
    return manifest


def _span_from_wire(value: object) -> SampleSpan:
    item = _mapping(value, ("timeline_id", "start", "end"), "sample span")
    return SampleSpan(
        item["timeline_id"],  # type: ignore[arg-type]
        item["start"],  # type: ignore[arg-type]
        item["end"],  # type: ignore[arg-type]
    )


def reference_manifest_from_wire(value: object) -> ReferenceSetManifest:
    item = _mapping(
        value,
        (
            "schema_version",
            "reference_set_id",
            "fixture_id",
            "master_id",
            "partition",
            "reference_version",
            "reference_sha256",
            "reference_text_sha256",
            "reference_set_sha256",
            "events",
            "verbatim_references",
            "editorial_subtitles",
            "annotation_passes",
            "adjudication",
            "non_speech_controls",
        ),
        "reference manifest",
    )
    events: list[ReferenceEvent] = []
    for value_event in _list(item["events"], "events"):
        event = _mapping(
            value_event,
            (
                "event_id",
                "master_id",
                "span",
                "intelligibility",
                "verbatim_ref_id",
                "tags",
                "annotation_version",
            ),
            "reference event",
        )
        events.append(
            ReferenceEvent(
                event_id=event["event_id"],  # type: ignore[arg-type]
                master_id=event["master_id"],  # type: ignore[arg-type]
                span=_span_from_wire(event["span"]),
                intelligibility=event["intelligibility"],  # type: ignore[arg-type]
                verbatim_ref_id=event["verbatim_ref_id"],  # type: ignore[arg-type]
                tags=_strings(event["tags"], "event tags"),
                annotation_version=event["annotation_version"],  # type: ignore[arg-type]
            )
        )
    verbatim: list[VerbatimReference] = []
    for value_reference in _list(item["verbatim_references"], "verbatim_references"):
        reference = _mapping(
            value_reference,
            ("verbatim_ref_id", "event_ids", "text_id", "annotation_version"),
            "verbatim reference",
        )
        verbatim.append(
            VerbatimReference(
                verbatim_ref_id=reference["verbatim_ref_id"],  # type: ignore[arg-type]
                event_ids=_strings(reference["event_ids"], "verbatim event_ids"),
                text_id=reference["text_id"],  # type: ignore[arg-type]
                annotation_version=reference["annotation_version"],  # type: ignore[arg-type]
            )
        )
    subtitles: list[EditorialSubtitle] = []
    for value_subtitle in _list(item["editorial_subtitles"], "editorial_subtitles"):
        subtitle = _mapping(
            value_subtitle,
            ("subtitle_id", "event_ids", "text_id", "annotation_version"),
            "editorial subtitle",
        )
        subtitles.append(
            EditorialSubtitle(
                subtitle_id=subtitle["subtitle_id"],  # type: ignore[arg-type]
                event_ids=_strings(subtitle["event_ids"], "subtitle event_ids"),
                text_id=subtitle["text_id"],  # type: ignore[arg-type]
                annotation_version=subtitle["annotation_version"],  # type: ignore[arg-type]
            )
        )
    passes: list[AnnotationPass] = []
    for value_pass in _list(item["annotation_passes"], "annotation_passes"):
        annotation = _mapping(
            value_pass,
            (
                "pass_id",
                "annotator_id",
                "reference_version",
                "content_sha256",
                "input_roles",
            ),
            "annotation pass",
        )
        passes.append(
            AnnotationPass(
                pass_id=annotation["pass_id"],  # type: ignore[arg-type]
                annotator_id=annotation["annotator_id"],  # type: ignore[arg-type]
                reference_version=annotation["reference_version"],  # type: ignore[arg-type]
                content_sha256=annotation["content_sha256"],  # type: ignore[arg-type]
                input_roles=tuple(
                    AnnotationInputRole(role)
                    for role in _strings(
                        annotation["input_roles"], "annotation input_roles"
                    )
                ),
            )
        )
    adjudication_value = item["adjudication"]
    adjudication: Adjudication | None = None
    if adjudication_value is not None:
        unit = _mapping(
            adjudication_value,
            (
                "adjudication_id",
                "adjudicator_id",
                "input_pass_ids",
                "input_roles",
                "output_reference_version",
                "output_content_sha256",
            ),
            "adjudication",
        )
        adjudication = Adjudication(
            adjudication_id=unit["adjudication_id"],  # type: ignore[arg-type]
            adjudicator_id=unit["adjudicator_id"],  # type: ignore[arg-type]
            input_pass_ids=_strings(unit["input_pass_ids"], "input_pass_ids"),
            input_roles=tuple(
                AnnotationInputRole(role)
                for role in _strings(unit["input_roles"], "input_roles")
            ),
            output_reference_version=unit["output_reference_version"],  # type: ignore[arg-type]
            output_content_sha256=unit["output_content_sha256"],  # type: ignore[arg-type]
        )
    controls: list[NonSpeechControl] = []
    for value_control in _list(item["non_speech_controls"], "non_speech_controls"):
        control = _mapping(
            value_control,
            ("region_id", "span", "tags"),
            "non-speech control",
        )
        controls.append(
            NonSpeechControl(
                region_id=control["region_id"],  # type: ignore[arg-type]
                span=_span_from_wire(control["span"]),
                tags=_strings(control["tags"], "non-speech control tags"),
            )
        )
    try:
        return ReferenceSetManifest(
            schema_version=item["schema_version"],  # type: ignore[arg-type]
            reference_set_id=item["reference_set_id"],  # type: ignore[arg-type]
            fixture_id=item["fixture_id"],  # type: ignore[arg-type]
            master_id=item["master_id"],  # type: ignore[arg-type]
            partition=Partition(item["partition"]),
            reference_version=item["reference_version"],  # type: ignore[arg-type]
            reference_sha256=item["reference_sha256"],  # type: ignore[arg-type]
            reference_text_sha256=item["reference_text_sha256"],  # type: ignore[arg-type]
            reference_set_sha256=item["reference_set_sha256"],  # type: ignore[arg-type]
            events=tuple(events),
            verbatim_references=tuple(verbatim),
            editorial_subtitles=tuple(subtitles),
            annotation_passes=tuple(passes),
            adjudication=adjudication,
            non_speech_controls=tuple(controls),
        )
    except (TypeError, ValueError):
        raise ValueError("reference manifest is invalid") from None


def reference_manifest_to_wire(manifest: ReferenceSetManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "reference_set_id": manifest.reference_set_id,
        "fixture_id": manifest.fixture_id,
        "master_id": manifest.master_id,
        "partition": manifest.partition.value,
        "reference_version": manifest.reference_version,
        "reference_sha256": manifest.reference_sha256,
        "reference_text_sha256": manifest.reference_text_sha256,
        "reference_set_sha256": manifest.reference_set_sha256,
        "events": [
            {
                "event_id": event.event_id,
                "master_id": event.master_id,
                "span": {
                    "timeline_id": event.span.timeline_id,
                    "start": event.span.start,
                    "end": event.span.end,
                },
                "intelligibility": event.intelligibility,
                "verbatim_ref_id": event.verbatim_ref_id,
                "tags": list(event.tags),
                "annotation_version": event.annotation_version,
            }
            for event in manifest.events
        ],
        "verbatim_references": [
            {
                "verbatim_ref_id": unit.verbatim_ref_id,
                "event_ids": list(unit.event_ids),
                "text_id": unit.text_id,
                "annotation_version": unit.annotation_version,
            }
            for unit in manifest.verbatim_references
        ],
        "editorial_subtitles": [
            {
                "subtitle_id": unit.subtitle_id,
                "event_ids": list(unit.event_ids),
                "text_id": unit.text_id,
                "annotation_version": unit.annotation_version,
            }
            for unit in manifest.editorial_subtitles
        ],
        "annotation_passes": [
            {
                "pass_id": unit.pass_id,
                "annotator_id": unit.annotator_id,
                "reference_version": unit.reference_version,
                "content_sha256": unit.content_sha256,
                "input_roles": [role.value for role in unit.input_roles],
            }
            for unit in manifest.annotation_passes
        ],
        "adjudication": (
            None
            if manifest.adjudication is None
            else {
                "adjudication_id": manifest.adjudication.adjudication_id,
                "adjudicator_id": manifest.adjudication.adjudicator_id,
                "input_pass_ids": list(manifest.adjudication.input_pass_ids),
                "input_roles": [
                    role.value for role in manifest.adjudication.input_roles
                ],
                "output_reference_version": (
                    manifest.adjudication.output_reference_version
                ),
                "output_content_sha256": (manifest.adjudication.output_content_sha256),
            }
        ),
        "non_speech_controls": [
            {
                "region_id": control.region_id,
                "span": {
                    "timeline_id": control.span.timeline_id,
                    "start": control.span.start,
                    "end": control.span.end,
                },
                "tags": list(control.tags),
            }
            for control in manifest.non_speech_controls
        ],
    }


def _frozen_reference(
    draft: ReferenceSetManifest,
    master: MasterManifest,
    reference_text_sha256: str,
) -> tuple[ReferenceSetManifest, MasterManifest]:
    if draft.master_id != master.master_id:
        raise ValueError("reference draft belongs to a different master")
    content_sha256 = reference_content_sha256(draft)
    adjudication = draft.adjudication
    if adjudication is not None:
        adjudication = replace(
            adjudication,
            output_content_sha256=adjudicated_reference_content_sha256(
                replace(
                    draft,
                    reference_sha256=content_sha256,
                    reference_text_sha256=reference_text_sha256,
                )
            ),
        )
    frozen = replace(
        draft,
        reference_sha256=content_sha256,
        reference_text_sha256=reference_text_sha256,
        adjudication=adjudication,
    )
    frozen = replace(
        frozen,
        reference_set_sha256=reference_set_manifest_sha256(frozen),
    )
    if master.annotation_set_sha256 not in {
        _ZERO_SHA256,
        frozen.reference_set_sha256,
    }:
        raise ValueError("master is already bound to another reference set")
    return frozen, replace(
        master,
        annotation_set_sha256=frozen.reference_set_sha256,
    )


def _reference_text_payload(
    path: Path,
    manifest: ReferenceSetManifest,
    boundary: PrivateBoundary,
) -> dict[str, object]:
    payload = _mapping(
        _read_json(require_private_path(path, boundary), "reference text sidecar"),
        ("schema_version", "reference_set_id", "texts"),
        "reference text sidecar",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["reference_set_id"] != manifest.reference_set_id
        or not isinstance(payload["texts"], dict)
    ):
        raise ValueError("reference text sidecar is invalid")
    expected = {
        unit.text_id: "verbatim_reference" for unit in manifest.verbatim_references
    }
    expected.update(
        {unit.text_id: "editorial_subtitle" for unit in manifest.editorial_subtitles}
    )
    texts = payload["texts"]
    if set(texts) != set(expected):
        raise ValueError("reference text sidecar coverage is incomplete")
    for text_id, role in expected.items():
        require_uuid4(text_id, "text_id")
        record = _mapping(texts[text_id], ("role", "text"), "reference text")
        if (
            record["role"] != role
            or type(record["text"]) is not str
            or not record["text"]
        ):
            raise ValueError("reference text sidecar record is invalid")
    # Scoring aligns only intelligible events with verbatim text; a text that
    # minimal normalization empties would give such an event no characters.
    scored_verbatim_ids = {
        event.verbatim_ref_id
        for event in manifest.events
        if event.intelligibility == "intelligible" and event.verbatim_ref_id is not None
    }
    for unit in manifest.verbatim_references:
        if unit.verbatim_ref_id in scored_verbatim_ids and not normalize_minimal(
            texts[unit.text_id]["text"]
        ):
            raise ValueError("intelligible speech text has no scorable characters")
    payload_sha256 = canonical_sha256(payload)  # type: ignore[arg-type]
    if manifest.reference_text_sha256 not in {_ZERO_SHA256, payload_sha256}:
        raise ValueError("reference text SHA-256 does not match manifest")
    return payload


def validate_reference_draft(
    draft_path: Path,
    text_path: Path,
    master_id: str,
    boundary: PrivateBoundary,
) -> ReferenceSetManifest:
    draft = reference_manifest_from_wire(
        _read_json(
            require_private_path(draft_path, boundary),
            "reference draft",
        )
    )
    master = load_registered_master(master_id, boundary)
    from .annotations import verify_reference_annotation_passes

    verify_reference_annotation_passes(draft, boundary)
    text_payload = _reference_text_payload(text_path, draft, boundary)
    text_sha256 = canonical_sha256(text_payload)  # type: ignore[arg-type]
    frozen, bound_master = _frozen_reference(draft, master, text_sha256)
    canonical = load_canonical_master(bound_master, boundary)
    load_reference_set(frozen, canonical)
    return frozen


def freeze_reference_draft(
    draft_path: Path,
    text_path: Path,
    master_id: str,
    boundary: PrivateBoundary,
) -> ReferenceSetManifest:
    draft = reference_manifest_from_wire(
        _read_json(
            require_private_path(draft_path, boundary),
            "reference draft",
        )
    )
    master = load_registered_master(master_id, boundary)
    from .annotations import verify_reference_annotation_passes

    verify_reference_annotation_passes(draft, boundary)
    text_payload = _reference_text_payload(text_path, draft, boundary)
    text_sha256 = canonical_sha256(text_payload)  # type: ignore[arg-type]
    frozen, bound_master = _frozen_reference(draft, master, text_sha256)
    canonical = load_canonical_master(bound_master, boundary)
    load_reference_set(frozen, canonical)
    reference_dir = require_private_path(
        boundary.root / "references" / frozen.reference_set_id,
        boundary,
    )
    manifest_target = reference_dir / "reference.private.json"
    text_target = reference_dir / "text.private.json"
    master_target = require_private_path(
        boundary.root / "masters" / master_id / "master.private.json",
        boundary,
    )
    locator_path, locator = _locator(boundary)
    fixtures = dict(locator["fixtures"])  # type: ignore[arg-type]
    entry = fixtures.get(frozen.fixture_id)
    if not isinstance(entry, dict):
        raise ValueError("private locator has no fixture entry")
    updated_entry = dict(entry)
    reference_sets = dict(updated_entry.get("reference_sets", {}))
    existing = reference_sets.get(frozen.reference_set_id)
    if existing is not None and existing != str(manifest_target):
        raise ValueError("reference set identity is already bound")
    reference_sets[frozen.reference_set_id] = str(manifest_target)
    updated_entry["reference_sets"] = reference_sets
    fixtures[frozen.fixture_id] = updated_entry
    updated_locator = {
        "schema_version": SCHEMA_VERSION,
        "fixtures": fixtures,
    }
    if manifest_target.exists():
        existing_manifest = reference_manifest_from_wire(
            _read_json(manifest_target, "registered reference manifest")
        )
        if existing_manifest != frozen:
            raise ValueError("reference manifest is immutable once frozen")
    if text_target.exists():
        existing_text = _mapping(
            _read_json(text_target, "registered reference text sidecar"),
            ("schema_version", "reference_set_id", "texts"),
            "registered reference text sidecar",
        )
        if existing_text != text_payload:
            raise ValueError("reference text is immutable once frozen")
    # Locator is committed last: an interruption can leave an unreferenced
    # immutable file, but never a locator pointing at a partial artifact set.
    _write_canonical_json(manifest_target, reference_manifest_to_wire(frozen))
    _write_canonical_json(text_target, text_payload)
    _write_canonical_json(master_target, master_manifest_to_wire(bound_master))
    _write_canonical_json(locator_path, updated_locator)
    return frozen


def load_registered_reference(
    reference_set_id: str,
    boundary: PrivateBoundary,
) -> ReferenceSetManifest:
    require_uuid4(reference_set_id, "reference_set_id")
    path = require_private_path(
        boundary.root / "references" / reference_set_id / "reference.private.json",
        boundary,
    )
    manifest = reference_manifest_from_wire(
        _read_json(path, "registered reference manifest")
    )
    if manifest.reference_set_id != reference_set_id:
        raise ValueError("registered reference identity does not match its path")
    if manifest.reference_text_sha256 == _ZERO_SHA256:
        raise ValueError("registered reference is not bound to its private text")
    return manifest


def load_reference_texts(
    reference_set_id: str,
    boundary: PrivateBoundary,
) -> dict[str, str]:
    manifest = load_registered_reference(reference_set_id, boundary)
    path = require_private_path(
        boundary.root / "references" / reference_set_id / "text.private.json",
        boundary,
    )
    payload = _reference_text_payload(path, manifest, boundary)
    return {
        text_id: record["text"]
        for text_id, record in payload["texts"].items()  # type: ignore[union-attr]
    }


__all__ = (
    "apply_rights_review",
    "boundary_from_environment",
    "freeze_reference_draft",
    "initialize_private_root",
    "load_reference_texts",
    "load_registered_fixture",
    "load_registered_master",
    "load_registered_reference",
    "master_manifest_from_wire",
    "master_manifest_to_wire",
    "reference_manifest_from_wire",
    "reference_manifest_to_wire",
    "register_master",
    "rights_from_wire",
    "rights_to_wire",
    "validate_reference_draft",
)
