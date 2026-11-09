"""Private fixture and rights records for the E0 evidence harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from scripts.e0_evidence._common import atomic_write_bytes, require_uuid4
from scripts.e0_evidence.private_root import (
    ROOT_MANIFEST_NAME,
    PrivateBoundary,
    parse_root_manifest,
    require_private_path,
)
from scripts.e0_evidence.schema import (
    SCHEMA_VERSION,
    CapabilityDecision,
    FixtureManifest,
    Partition,
    RightsRecord,
    new_opaque_id,
)


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    failed = False
    try:
        serialized = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, serialized)
    except (OSError, UnicodeError):
        failed = True
    if failed:
        raise ValueError("private record write failed") from None


def _private_path_exists(path: Path, record_name: str) -> bool:
    failed = False
    exists = False
    try:
        exists = path.exists()
    except OSError:
        failed = True
    if failed:
        raise ValueError(f"{record_name} is inaccessible") from None
    return exists


def _read_private_json(path: Path, record_name: str) -> object:
    failed = False
    payload: object = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        failed = True
    if failed:
        raise ValueError(f"{record_name} is unreadable") from None
    return payload


def require_local_evaluation(rights: RightsRecord) -> None:
    if not isinstance(rights, RightsRecord):
        raise TypeError("rights must be RightsRecord")
    if rights.local_evaluation is not CapabilityDecision.APPROVED:
        raise PermissionError(
            f"local evaluation is not approved for fixture {rights.fixture_id}"
        )
    if not rights.reviewer_id or not rights.evidence_ids:
        raise ValueError("local evaluation approval lacks reviewer or evidence")


def _rights_payload(rights: RightsRecord) -> dict[str, object]:
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


def _load_locator(path: Path) -> dict[str, object]:
    if not _private_path_exists(path, "private locator"):
        return {"schema_version": SCHEMA_VERSION, "fixtures": {}}
    payload = _read_private_json(path, "private locator")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("fixtures"), dict)
        or set(payload) != {"schema_version", "fixtures"}
    ):
        raise ValueError("private locator has an invalid wire shape")
    fixtures = payload["fixtures"]
    for fixture_id, entry in fixtures.items():
        require_uuid4(fixture_id, "locator fixture_id")
        if (
            not isinstance(entry, dict)
            or "source" not in entry
            or not set(entry).issubset(
                {"source", "masters", "reference_sets"}
            )
        ):
            raise ValueError("private locator has an invalid fixture entry")
        _require_absolute_locator_path(entry["source"], "source")
        for collection_name in ("masters", "reference_sets"):
            collection = entry.get(collection_name, {})
            if not isinstance(collection, dict):
                raise ValueError(
                    f"private locator has an invalid {collection_name} entry"
                )
            for artifact_id, artifact_path in collection.items():
                require_uuid4(artifact_id, f"{collection_name} artifact_id")
                _require_absolute_locator_path(
                    artifact_path,
                    collection_name,
                )
    return payload


def _require_absolute_locator_path(value: object, field_name: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"private locator {field_name} path is invalid")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"private locator {field_name} path must be absolute")
    return candidate


def _validate_root_manifest(path: Path) -> None:
    parse_root_manifest(_read_private_json(path, "private root manifest"))


def _source_path(source: Path, boundary: PrivateBoundary) -> Path:
    candidate = Path(source)
    if not candidate.is_absolute():
        raise ValueError("fixture source must be an explicit absolute file")
    resolved: Path | None = None
    regular_file = False
    try:
        resolved = candidate.resolve(strict=False)
        regular_file = resolved.is_file()
    except OSError:
        pass
    if resolved is None or not regular_file:
        raise ValueError("fixture source must be an explicit regular file")
    if any(
        resolved == denied or denied in resolved.parents
        for denied in boundary.excluded_roots
    ):
        raise ValueError("fixture source is under an excluded root")
    return resolved


def register_fixture(
    source: Path,
    partition: Partition,
    boundary: PrivateBoundary,
    *,
    title_id: str | None = None,
) -> FixtureManifest:
    if not isinstance(partition, Partition):
        raise TypeError("partition must be Partition")
    resolved_source = _source_path(source, boundary)
    fixture_id = new_opaque_id()
    resolved_title_id = new_opaque_id() if title_id is None else title_id
    rights_record_id = new_opaque_id()
    manifest = FixtureManifest(
        fixture_id=fixture_id,
        partition=partition,
        title_id=resolved_title_id,
        rights_record_id=rights_record_id,
    )
    rights = RightsRecord(
        rights_record_id=rights_record_id,
        fixture_id=fixture_id,
        local_evaluation=CapabilityDecision.PENDING,
        redistribution=CapabilityDecision.PENDING,
        training=CapabilityDecision.PENDING,
        transcript_publication=CapabilityDecision.PENDING,
        reviewer_id="",
        evidence_ids=(),
        retention_version="retention-1.0",
    )

    root_manifest = require_private_path(
        boundary.root / ROOT_MANIFEST_NAME,
        boundary,
    )
    locator_path = require_private_path(
        boundary.root / "locator.private.json",
        boundary,
    )
    fixture_path = require_private_path(
        boundary.root / "fixtures" / fixture_id / "fixture.private.json",
        boundary,
    )
    rights_path = require_private_path(
        boundary.root / "rights" / f"{rights_record_id}.private.json",
        boundary,
    )

    root_exists = _private_path_exists(root_manifest, "private root manifest")
    if root_exists:
        _validate_root_manifest(root_manifest)
    locator = _load_locator(locator_path)
    fixtures = dict(locator["fixtures"])
    fixtures[fixture_id] = {"source": str(resolved_source)}
    locator["fixtures"] = fixtures

    if not root_exists:
        _write_private_json(
            root_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "private_root_id": new_opaque_id(),
                "excluded_roots": [],
            },
        )
    _write_private_json(
        fixture_path,
        {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": manifest.fixture_id,
            "partition": manifest.partition.value,
            "title_id": manifest.title_id,
            "rights_record_id": manifest.rights_record_id,
        },
    )
    _write_private_json(rights_path, _rights_payload(rights))
    _write_private_json(locator_path, locator)
    return manifest


def resolve_private_locator_asset(
    fixture_id: str,
    rights: RightsRecord,
    collection_name: str,
    artifact_id: str,
    boundary: PrivateBoundary,
) -> Path:
    require_uuid4(fixture_id, "fixture_id")
    if not isinstance(rights, RightsRecord):
        raise TypeError("rights must be RightsRecord")
    if rights.fixture_id != fixture_id:
        raise ValueError("rights record does not belong to the fixture")
    require_local_evaluation(rights)
    require_uuid4(artifact_id, "artifact_id")
    if collection_name not in {"masters", "reference_sets"}:
        raise ValueError("private locator collection is not supported")
    locator_path = require_private_path(
        boundary.root / "locator.private.json",
        boundary,
    )
    locator = _load_locator(locator_path)
    entry = locator["fixtures"].get(fixture_id)
    if not isinstance(entry, dict):
        raise ValueError(f"private locator has no entry for fixture {fixture_id}")
    collection = entry.get(collection_name)
    if not isinstance(collection, dict):
        raise ValueError(
            f"private locator has no {collection_name} for fixture {fixture_id}"
        )
    raw_path = collection.get(artifact_id)
    candidate = _require_absolute_locator_path(raw_path, collection_name)
    return require_private_path(candidate, boundary)


def resolve_fixture_source(
    fixture_id: str,
    rights: RightsRecord,
    boundary: PrivateBoundary,
) -> Path:
    require_uuid4(fixture_id, "fixture_id")
    if rights.fixture_id != fixture_id:
        raise ValueError("rights record does not belong to the fixture")
    require_local_evaluation(rights)

    locator_path = require_private_path(
        boundary.root / "locator.private.json",
        boundary,
    )
    locator = _load_locator(locator_path)
    entry = locator["fixtures"].get(fixture_id)
    if not isinstance(entry, dict):
        raise ValueError(f"private locator has no entry for fixture {fixture_id}")
    raw_source = entry.get("source")
    if type(raw_source) is not str:
        raise ValueError(f"private locator has no entry for fixture {fixture_id}")
    return _source_path(Path(raw_source), boundary)
