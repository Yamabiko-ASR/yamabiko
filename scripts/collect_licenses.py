"""Derive and atomically publish the frozen bundle's license payload.

The generated provenance report decides which components require coverage.
The override policy supplies reviewed texts only where a wheel does not carry
the applicable legal file.  A hand-maintained distribution allowlist is never
used, so adding a bundled dependency also adds a license obligation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sys
import tempfile
import uuid
from typing import Callable, Iterable, Iterator

from packaging.utils import canonicalize_name


# Direct execution puts scripts/ rather than the repository root on sys.path.
if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bundle_inventory import InventoryEntry, load_inventory
from scripts.bundle_provenance import (
    DistributionRef,
    _hash_file_stably,
    _is_reparse,
    _normalized_relative,
    build_record_ownership,
)


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "installer" / "licenses-manifest.txt"
REPO_LICENSES = ROOT / "licenses"
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
PYTHON_LICENSE = Path(sys.base_prefix) / "LICENSE.txt"
DEFAULT_INVENTORY = ROOT / "build" / "yamabiko" / "bundle-inventory.json"
DEFAULT_PROVENANCE_REPORT = (
    ROOT / "build" / "yamabiko" / "bundle-provenance.json"
)

SCHEMA_VERSION = 1
_PROVENANCE_RECORD_FIELDS = frozenset(
    {
        "subject",
        "component_id",
        "component_name",
        "provenance_class",
        "distribution",
        "license_mode",
        "license_source",
        "review_id",
        "review_reason",
    }
)
_SUBJECT_FIELDS = frozenset({"kind", "key", "path"})
_DISTRIBUTION_FIELDS = frozenset({"name", "version", "dist_info_name"})
_BUNDLE_FILE_FIELDS = frozenset(
    {"relative_path", "size", "sha256", "inventory_keys"}
)
_INDEX_FIELDS = frozenset(
    {
        "component_id",
        "component_name",
        "version",
        "provenance_class",
        "license_mode",
        "source_path",
        "output_path",
        "sha256",
        "review_id",
        "review_reason",
    }
)
_LICENSE_MODES = frozenset(
    {
        "project",
        "distribution",
        "bundle-file",
        "python-runtime-file",
        "reviewed-exception",
    }
)
_LEGAL_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")
_LICENSE_BUNDLE_PREFIX = "_internal/licenses/"
_PUBLISH_LOCK_NAME = ".licenses-publish.lock"
_CRASH_ARTIFACT_PREFIXES = (
    ".licenses-staging-",
    ".licenses-backup-",
    ".licenses-quarantine-",
)


@dataclass(frozen=True)
class _Component:
    component_id: str
    component_name: str
    provenance_class: str
    distribution: DistributionRef | None
    license_mode: str
    license_source: str | None
    review_id: str | None
    review_reason: str | None


@dataclass(frozen=True)
class _ProvenanceInputs:
    components: tuple[_Component, ...]
    bundle_files: dict[str, tuple[str, int, str]]


def _canonical_component_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be a non-empty string")
    canonical = canonicalize_name(value)
    if value != canonical or not _COMPONENT_ID_RE.fullmatch(canonical):
        raise ValueError(f"{label} must be PEP 503-canonical: {value!r}")
    return canonical


def _policy_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("license override path must be a non-empty string")
    normalized_input = value.replace("\\", "/")
    posix = PurePosixPath(normalized_input)
    windows = PureWindowsPath(value)
    if (
        value != normalized_input
        or windows.drive
        or posix.is_absolute()
        or normalized_input.startswith("//")
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
    ):
        raise ValueError(f"license override path must stay under licenses/: {value!r}")
    normalized = posix.as_posix()
    if normalized in {"", "."} or normalized != value:
        raise ValueError(f"license override path is not normalized: {value!r}")
    return normalized


def _parse_manifest(src: str) -> list[tuple[str, str, str]]:
    """Parse the reviewed override policy and reject ambiguous records."""

    entries: list[tuple[str, str, str]] = []
    seen_files: set[tuple[str, str]] = set()
    aliases: dict[str, str] = {}
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        kind, separator, remainder = line.partition(":")
        if not separator or kind not in {"file", "alias"}:
            raise ValueError(f"malformed licenses-manifest.txt line: {raw!r}")
        left, value_separator, right = remainder.partition("|")
        if not value_separator or "|" in right:
            raise ValueError(f"malformed licenses-manifest.txt line: {raw!r}")
        owner = _canonical_component_id(
            left.strip(), label=f"{kind} override component"
        )
        if kind == "file":
            relative = _policy_relative_path(right.strip())
            identity = (owner, relative.casefold())
            if identity in seen_files:
                raise ValueError(
                    f"duplicate license file override for {owner}: {relative}"
                )
            seen_files.add(identity)
            entries.append((kind, owner, relative))
            continue

        target = _canonical_component_id(
            right.strip(), label="license alias target"
        )
        if owner in aliases:
            raise ValueError(f"distribution {owner} has more than one license alias")
        aliases[owner] = target
        entries.append((kind, owner, target))
    return entries


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> object:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc


def _optional_nonempty_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{label} must be null or a non-empty string")
    return value


def _distribution(value: object, *, component_id: str) -> DistributionRef | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _DISTRIBUTION_FIELDS:
        raise ValueError(f"component {component_id} has malformed distribution")
    if any(
        not isinstance(value[field], str)
        or not value[field].strip()
        or "\0" in value[field]
        for field in _DISTRIBUTION_FIELDS
    ):
        raise ValueError(f"component {component_id} has malformed distribution")
    return DistributionRef(
        name=value["name"],
        version=value["version"],
        dist_info_name=value["dist_info_name"],
    )


def _component_from_record(row: dict[str, object], index: int) -> _Component:
    component_id = _canonical_component_id(
        row["component_id"], label=f"provenance record {index} component_id"
    )
    component_name = _optional_nonempty_string(
        row["component_name"], label=f"component {component_id} name"
    )
    provenance_class = _optional_nonempty_string(
        row["provenance_class"], label=f"component {component_id} provenance class"
    )
    assert component_name is not None
    assert provenance_class is not None
    mode = row["license_mode"]
    if not isinstance(mode, str) or mode not in _LICENSE_MODES:
        raise ValueError(f"component {component_id} has unknown license mode")
    source = _optional_nonempty_string(
        row["license_source"], label=f"component {component_id} license source"
    )
    review_id = _optional_nonempty_string(
        row["review_id"], label=f"component {component_id} review id"
    )
    review_reason = _optional_nonempty_string(
        row["review_reason"], label=f"component {component_id} review reason"
    )
    distribution = _distribution(row["distribution"], component_id=component_id)

    if mode == "project":
        if component_id != "yamabiko" or any(
            value is not None
            for value in (distribution, source, review_id, review_reason)
        ):
            raise ValueError("project provenance descriptor is malformed")
    elif mode == "distribution":
        source = _canonical_component_id(
            source, label=f"component {component_id} distribution license source"
        )
        if review_id is not None or review_reason is not None:
            raise ValueError(f"distribution component {component_id} has review fields")
    elif mode == "bundle-file":
        if distribution is not None or source is None:
            raise ValueError(f"bundle-file component {component_id} is malformed")
        source = _normalized_relative(source, label="bundle license source")
        if review_id is not None or review_reason is not None:
            raise ValueError(f"bundle-file component {component_id} has review fields")
    elif mode == "python-runtime-file":
        if (
            distribution is not None
            or source != "python://LICENSE.txt"
            or review_id is not None
            or review_reason is not None
        ):
            raise ValueError(f"Python runtime component {component_id} is malformed")
    elif mode == "reviewed-exception":
        if (
            distribution is not None
            or source is not None
            or review_id is None
            or review_reason is None
        ):
            raise ValueError(f"reviewed exception {component_id} is malformed")

    return _Component(
        component_id=component_id,
        component_name=component_name,
        provenance_class=provenance_class,
        distribution=distribution,
        license_mode=mode,
        license_source=source,
        review_id=review_id,
        review_reason=review_reason,
    )


def _validate_bundle_files(
    value: object, inventory_keys: frozenset[str]
) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("provenance bundle_files must be a list")
    result: dict[str, dict[str, object]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != _BUNDLE_FILE_FIELDS:
            raise ValueError(f"provenance bundle_file {index} has malformed fields")
        relative = _normalized_relative(
            row["relative_path"], label=f"provenance bundle_file {index} path"
        )
        size = row["size"]
        digest = row["sha256"]
        keys = row["inventory_keys"]
        if type(size) is not int or size < 0:
            raise ValueError(f"provenance bundle_file {relative} has invalid size")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"provenance bundle_file {relative} has invalid sha256")
        if (
            not isinstance(keys, list)
            or any(not isinstance(key, str) or key not in inventory_keys for key in keys)
            or len(keys) != len(set(keys))
        ):
            raise ValueError(
                f"provenance bundle_file {relative} has invalid inventory keys"
            )
        folded = relative.casefold()
        if folded in result:
            raise ValueError(f"duplicate provenance bundle_file path: {relative}")
        result[folded] = row
    return result


def _load_provenance(
    path: Path, entries: tuple[InventoryEntry, ...]
) -> _ProvenanceInputs:
    payload = _load_json(path, label="bundle provenance report")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "records",
        "bundle_files",
        "errors",
    }:
        raise ValueError("bundle provenance report has malformed top-level fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported bundle provenance schema_version")
    if payload["errors"] != []:
        raise ValueError("bundle provenance report is not a green closure")
    rows = payload["records"]
    if not isinstance(rows, list):
        raise ValueError("bundle provenance records must be a list")

    inventory_by_key = {entry.key: entry for entry in entries}
    inventory_keys = frozenset(inventory_by_key)
    bundle_files = _validate_bundle_files(payload["bundle_files"], inventory_keys)
    analysis_keys: set[str] = set()
    record_bundle_paths: set[str] = set()
    subjects: set[tuple[str, str]] = set()
    descriptors: dict[str, _Component] = {}

    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _PROVENANCE_RECORD_FIELDS:
            raise ValueError(f"provenance record {index} has malformed fields")
        subject = row["subject"]
        if not isinstance(subject, dict) or set(subject) != _SUBJECT_FIELDS:
            raise ValueError(f"provenance record {index} has malformed subject")
        kind = subject["kind"]
        key = subject["key"]
        subject_path = subject["path"]
        if (
            kind not in {"analysis-entry", "bundle-file"}
            or not isinstance(key, str)
            or not key
            or not isinstance(subject_path, str)
            or not subject_path
        ):
            raise ValueError(f"provenance record {index} has malformed subject")
        subject_identity = (kind, key)
        if subject_identity in subjects:
            raise ValueError(f"duplicate provenance subject {kind}:{key}")
        subjects.add(subject_identity)

        if kind == "analysis-entry":
            entry = inventory_by_key.get(key)
            if entry is None or entry.destination != subject_path:
                raise ValueError(
                    f"analysis provenance subject does not match inventory: {key}"
                )
            analysis_keys.add(key)
        else:
            normalized = _normalized_relative(
                subject_path, label="bundle provenance subject path"
            )
            if key != normalized or normalized.casefold() not in bundle_files:
                raise ValueError(
                    f"bundle provenance subject does not match bundle_files: {key}"
                )
            if normalized.casefold() in record_bundle_paths:
                raise ValueError(f"duplicate bundle provenance subject: {normalized}")
            record_bundle_paths.add(normalized.casefold())

        component = _component_from_record(row, index)
        existing = descriptors.get(component.component_id)
        if existing is not None and existing != component:
            raise ValueError(
                f"component {component.component_id} has inconsistent descriptors"
            )
        descriptors[component.component_id] = component

    if analysis_keys != inventory_keys:
        missing = sorted(inventory_keys - analysis_keys)
        extra = sorted(analysis_keys - inventory_keys)
        raise ValueError(
            f"provenance analysis subjects do not match inventory; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if record_bundle_paths != set(bundle_files):
        raise ValueError("provenance bundle subjects do not match bundle_files")

    components = tuple(
        sorted(
            (
                component
                for component in descriptors.values()
                if component.license_mode != "project"
            ),
            key=lambda component: component.component_id,
        )
    )
    if not components:
        raise ValueError("bundle provenance report has no third-party components")
    return _ProvenanceInputs(
        components,
        {
            folded: (
                str(row["relative_path"]),
                int(row["size"]),
                str(row["sha256"]),
            )
            for folded, row in bundle_files.items()
        },
    )


def _override_policy(
    path: Path, components: tuple[_Component, ...]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    try:
        entries = _parse_manifest(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read license override policy: {path}") from exc

    component_by_id = {
        component.component_id: component for component in components
    }
    component_ids = set(component_by_id)
    file_lists: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}
    for kind, owner, value in entries:
        if owner not in component_ids:
            raise ValueError(f"stale license override component: {owner}")
        if kind == "file":
            if component_by_id[owner].license_mode != "distribution":
                raise ValueError(
                    f"license file override for non-distribution component: {owner}"
                )
            file_lists.setdefault(owner, []).append(value)
        else:
            if value not in component_ids:
                raise ValueError(f"stale license alias target: {value}")
            if (
                component_by_id[owner].license_mode != "distribution"
                or component_by_id[value].license_mode != "distribution"
            ):
                raise ValueError(
                    f"license alias must connect distribution components: {owner}"
                )
            aliases[owner] = value

    for owner, target in aliases.items():
        if owner in file_lists:
            raise ValueError(
                f"license alias for {owner} conflicts with its direct file override"
            )
        if target in aliases or target not in file_lists:
            raise ValueError(
                f"license alias {owner} does not resolve to reviewed file overrides"
            )

    return (
        {
            owner: tuple(sorted(paths, key=lambda value: (value.casefold(), value)))
            for owner, paths in file_lists.items()
        },
        dict(sorted(aliases.items())),
    )


def _distribution_refs(
    ownership: dict[str, DistributionRef],
) -> dict[str, DistributionRef]:
    grouped: dict[str, set[DistributionRef]] = {}
    for ref in ownership.values():
        component_id = canonicalize_name(ref.name)
        grouped.setdefault(component_id, set()).add(ref)
    result: dict[str, DistributionRef] = {}
    for component_id, refs in grouped.items():
        if len(refs) != 1:
            versions = ", ".join(
                sorted(f"{ref.name} {ref.version}" for ref in refs)
            )
            raise ValueError(
                f"multiple installed distributions normalize to {component_id}: {versions}"
            )
        result[component_id] = next(iter(refs))
    return result


def _is_legal_relative(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        bool(path.parts)
        and path.parts[0].casefold() == "licenses"
    ) or path.name.upper().startswith(_LEGAL_PREFIXES)


def _legal_files(
    site_root: Path,
    ref: DistributionRef,
    ownership: dict[str, DistributionRef],
) -> tuple[tuple[Path, str, str], ...]:
    dist_info = site_root / ref.dist_info_name
    if dist_info.is_symlink() or _is_reparse(dist_info) or not dist_info.is_dir():
        raise ValueError(
            f"distribution metadata is not a regular directory: {ref.dist_info_name}"
        )
    ownership_folded = {key.casefold(): value for key, value in ownership.items()}
    found: list[tuple[Path, str, str]] = []
    seen: set[str] = set()
    for path in sorted(dist_info.rglob("*"), key=lambda value: value.as_posix().casefold()):
        if path.is_symlink() or _is_reparse(path):
            raise ValueError(f"reparse point in license source: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(dist_info).as_posix()
        if not _is_legal_relative(relative):
            continue
        folded = relative.casefold()
        if folded in seen:
            raise ValueError(
                f"case-colliding legal paths in {ref.dist_info_name}: {relative}"
            )
        seen.add(folded)
        source_path = f"site-packages://{ref.dist_info_name}/{relative}"
        record_owner = ownership_folded.get(source_path.casefold())
        if record_owner != ref:
            raise ValueError(
                f"legal file lacks a complete matching RECORD claim: {source_path}"
            )
        found.append((path, relative, source_path))
    return tuple(found)


def _repo_override(relative: str) -> Path:
    root = Path(REPO_LICENSES)
    if not root.is_dir():
        raise ValueError(f"repository license directory is missing: {root}")
    resolved_root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink() or _is_reparse(current):
            raise ValueError(f"reparse point in repository license path: {relative}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"repository license file is missing: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"repository license file is not regular: {relative}")
    return resolved


def _copy_verified(source: Path, destination: Path) -> str:
    source_size, source_hash = _hash_file_stably(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_size, copied_hash = _hash_file_stably(destination)
    if (copied_size, copied_hash) != (source_size, source_hash):
        raise ValueError(f"copied license file changed: {source.name}")
    return source_hash


def _regular_bundle_file(dist_root: Path, relative: str) -> Path:
    """Resolve a bundle-relative regular file without crossing a reparse."""

    normalized = _normalized_relative(relative, label="bundle license path")
    root = Path(dist_root)
    current = root
    for part in PurePosixPath(normalized).parts:
        current /= part
        if current.is_symlink() or _is_reparse(current):
            raise ValueError(f"reparse point in bundle license path: {relative}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"bundle license source is missing: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"bundle license source is not regular: {relative}")
    return resolved


def _row(
    component: _Component,
    *,
    version: str | None,
    source_path: str | None,
    output_path: str | None,
    sha256: str | None,
) -> dict[str, object]:
    return {
        "component_id": component.component_id,
        "component_name": component.component_name,
        "version": version,
        "provenance_class": component.provenance_class,
        "license_mode": component.license_mode,
        "source_path": source_path,
        "output_path": output_path,
        "sha256": sha256,
        "review_id": component.review_id,
        "review_reason": component.review_reason,
    }


def _stage_output(staging: Path, output_path: str) -> Path:
    if not output_path.startswith(_LICENSE_BUNDLE_PREFIX):
        raise ValueError(f"license staging path is outside license root: {output_path}")
    relative = output_path[len(_LICENSE_BUNDLE_PREFIX) :]
    normalized = _normalized_relative(relative, label="staged license output")
    return staging.joinpath(*PurePosixPath(normalized).parts)


def _collect_rows(
    *,
    staging: Path,
    dist_root: Path,
    components: tuple[_Component, ...],
    file_overrides: dict[str, tuple[str, ...]],
    aliases: dict[str, str],
    site_root: Path,
    ownership: dict[str, DistributionRef],
    refs: dict[str, DistributionRef],
    bundle_claims: dict[str, tuple[str, int, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component in components:
        if component.license_mode == "reviewed-exception":
            rows.append(
                _row(
                    component,
                    version=None,
                    source_path=None,
                    output_path=None,
                    sha256=None,
                )
            )
            continue

        if component.license_mode == "bundle-file":
            assert component.license_source is not None
            claim = bundle_claims.get(component.license_source.casefold())
            if claim is None or claim[0] != component.license_source:
                raise ValueError(
                    f"bundle license source is absent from provenance: "
                    f"{component.license_source}"
                )
            source = _regular_bundle_file(dist_root, component.license_source)
            size, digest = _hash_file_stably(source)
            if (size, digest) != (claim[1], claim[2]):
                raise ValueError(
                    f"bundle license source changed after provenance: "
                    f"{component.license_source}"
                )
            rows.append(
                _row(
                    component,
                    version=None,
                    source_path=component.license_source,
                    output_path=component.license_source,
                    sha256=digest,
                )
            )
            continue

        if component.license_mode == "python-runtime-file":
            source = Path(PYTHON_LICENSE)
            if source.is_symlink() or _is_reparse(source) or not source.is_file():
                raise ValueError(f"Python runtime license is missing: {source}")
            version = platform.python_version()
            output = (
                f"{_LICENSE_BUNDLE_PREFIX}components/{component.component_id}/"
                f"{version}/LICENSE.txt"
            )
            digest = _copy_verified(source, _stage_output(staging, output))
            rows.append(
                _row(
                    component,
                    version=version,
                    source_path="python://LICENSE.txt",
                    output_path=output,
                    sha256=digest,
                )
            )
            continue

        assert component.license_mode == "distribution"
        assert component.license_source is not None
        ref = refs.get(component.license_source)
        if ref is None:
            raise ValueError(
                f"component {component.component_id} references unavailable "
                f"distribution {component.license_source}"
            )
        if (
            component.distribution is not None
            and component.license_source == component.component_id
            and component.distribution != ref
        ):
            raise ValueError(
                f"component {component.component_id} provenance distribution "
                "does not match installed metadata"
            )

        override_owner = component.component_id
        if override_owner not in file_overrides:
            override_owner = aliases.get(component.component_id, "")
        override_files = file_overrides.get(override_owner, ())
        if override_files:
            for relative in override_files:
                source = _repo_override(relative)
                output = (
                    f"{_LICENSE_BUNDLE_PREFIX}components/{component.component_id}/"
                    f"override/{relative}"
                )
                digest = _copy_verified(source, _stage_output(staging, output))
                rows.append(
                    _row(
                        component,
                        version=ref.version,
                        source_path=f"repo://licenses/{relative}",
                        output_path=output,
                        sha256=digest,
                    )
                )
            continue

        legal_files = _legal_files(site_root, ref, ownership)
        if not legal_files:
            raise ValueError(
                f"derived component {component.component_id} has no wheel legal "
                "file or reviewed override"
            )
        for source, relative, source_path in legal_files:
            output = (
                f"{_LICENSE_BUNDLE_PREFIX}components/{component.component_id}/"
                f"{ref.version}/{relative}"
            )
            digest = _copy_verified(source, _stage_output(staging, output))
            rows.append(
                _row(
                    component,
                    version=ref.version,
                    source_path=source_path,
                    output_path=output,
                    sha256=digest,
                )
            )
    return rows


def _canonical_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    canonical = sorted(
        rows,
        key=lambda row: (
            str(row["component_id"]),
            "" if row["output_path"] is None else str(row["output_path"]).casefold(),
            "" if row["output_path"] is None else str(row["output_path"]),
        ),
    )
    output_paths: set[str] = set()
    component_ids: set[str] = set()
    for row in canonical:
        if set(row) != _INDEX_FIELDS:
            raise ValueError("generated license index row has malformed fields")
        component_ids.add(str(row["component_id"]))
        output = row["output_path"]
        if output is None:
            if row["license_mode"] != "reviewed-exception":
                raise ValueError("only reviewed exceptions may omit an output path")
            continue
        normalized = _normalized_relative(output, label="license index output path")
        folded = normalized.casefold()
        if folded in output_paths:
            raise ValueError(f"duplicate license index output path: {normalized}")
        output_paths.add(folded)
        if not isinstance(row["sha256"], str) or not _SHA256_RE.fullmatch(
            row["sha256"]
        ):
            raise ValueError(f"invalid license index sha256: {normalized}")
    if not component_ids:
        raise ValueError("generated license index is empty")
    return canonical


def _index_text(rows: list[dict[str, object]]) -> str:
    lines = [
        "Third-party component and license-text index for this Yamabiko bundle.",
        "Generated from the exact bundle inventory; do not edit by hand.",
        "",
    ]
    for row in rows:
        component = str(row["component_id"])
        version = "" if row["version"] is None else f" {row['version']}"
        if row["output_path"] is None:
            lines.append(
                f"{component}{version} [{row['license_mode']}]: "
                f"{row['review_id']} - {row['review_reason']}"
            )
        else:
            lines.append(
                f"{component}{version} [{row['license_mode']}]: "
                f"{row['output_path']} (sha256:{row['sha256']})"
            )
    return "\n".join(lines) + "\n"


def _payload_file(
    license_root: Path, dist_root: Path, output_path: str
) -> Path:
    if output_path.startswith(_LICENSE_BUNDLE_PREFIX):
        relative = output_path[len(_LICENSE_BUNDLE_PREFIX) :]
        return license_root.joinpath(*PurePosixPath(relative).parts)
    return _regular_bundle_file(dist_root, output_path)


def _verify_payload(
    license_root: Path,
    *,
    dist_root: Path,
    expected_payload: dict[str, object],
    expected_text: str,
) -> None:
    if (
        license_root.is_symlink()
        or _is_reparse(license_root)
        or not license_root.is_dir()
    ):
        raise ValueError("published license root is not a regular directory")
    actual_payload = _load_json(license_root / "INDEX.json", label="license index")
    if actual_payload != expected_payload:
        raise ValueError("published JSON license index changed after write")
    try:
        actual_text = (license_root / "INDEX.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("published text license index cannot be re-read") from exc
    if actual_text != expected_text:
        raise ValueError("published text license index changed after write")

    expected_inside = {"INDEX.json", "INDEX.txt"}
    for row in expected_payload["entries"]:  # type: ignore[index]
        output = row["output_path"]
        if output is None:
            continue
        path = _payload_file(license_root, dist_root, output)
        if path.is_symlink() or _is_reparse(path) or not path.is_file():
            raise ValueError(f"indexed license output is missing: {output}")
        _size, digest = _hash_file_stably(path)
        if digest != row["sha256"]:
            raise ValueError(f"indexed license output hash changed: {output}")
        if output.startswith(_LICENSE_BUNDLE_PREFIX):
            expected_inside.add(output[len(_LICENSE_BUNDLE_PREFIX) :])

    actual_inside: set[str] = set()
    for path in license_root.rglob("*"):
        if path.is_symlink() or _is_reparse(path):
            raise ValueError(f"reparse point in published license payload: {path}")
        if path.is_file():
            actual_inside.add(path.relative_to(license_root).as_posix())
    actual_folded = [path.casefold() for path in actual_inside]
    if len(actual_folded) != len(set(actual_folded)):
        raise ValueError("published license payload has case-colliding files")
    if set(actual_folded) != {path.casefold() for path in expected_inside}:
        raise ValueError("published license payload has missing or extra files")


def _validate_regular_tree(path: Path, *, label: str) -> None:
    if path.is_symlink() or _is_reparse(path) or not path.is_dir():
        raise ValueError(f"{label} is not a regular directory: {path}")
    for descendant in path.rglob("*"):
        if descendant.is_symlink() or _is_reparse(descendant):
            raise ValueError(f"{label} contains a reparse point: {descendant}")


def _remove_tree(path: Path) -> None:
    if path.exists():
        _validate_regular_tree(path, label="refusing to remove unsafe directory tree")
        shutil.rmtree(path)


@contextmanager
def _exclusive_license_collection(internal: Path) -> Iterator[None]:
    """Serialize publishers and fail closed on prior crash artifacts."""

    lock = internal / _PUBLISH_LOCK_NAME
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"license publication lock already exists: {lock}; another collector "
            "is active or a prior run was interrupted"
        ) from exc
    except OSError as exc:
        raise ValueError(f"cannot acquire license publication lock: {lock}") from exc

    try:
        stale = sorted(
            child.name
            for child in internal.iterdir()
            if child.name.casefold().startswith(_CRASH_ARTIFACT_PREFIXES)
        )
        if stale:
            raise ValueError(
                "stale license publication artifacts require review before build: "
                + ", ".join(stale)
            )
        yield
    finally:
        try:
            lock.rmdir()
        except OSError as exc:
            raise RuntimeError(
                f"cannot release license publication lock: {lock}"
            ) from exc


def _publish(
    staging: Path,
    target: Path,
    *,
    verify: Callable[[Path], None],
) -> None:
    backup: Path | None = None
    if target.exists() or target.is_symlink():
        _validate_regular_tree(target, label="existing license target")
        backup = target.parent / f".licenses-backup-{uuid.uuid4().hex}"
        target.rename(backup)

    try:
        staging.rename(target)
    except Exception as publish_error:
        if backup is not None:
            try:
                backup.rename(target)
            except Exception as restore_error:
                raise RuntimeError(
                    f"license publication failed and backup restoration failed; "
                    f"backup preserved at {backup}"
                ) from restore_error
        raise publish_error

    try:
        verify(target)
    except Exception as verify_error:
        quarantine = target.parent / f".licenses-quarantine-{uuid.uuid4().hex}"
        try:
            target.rename(quarantine)
        except Exception as quarantine_error:
            if backup is not None:
                raise RuntimeError(
                    f"published license verification failed; prior payload is "
                    f"preserved at {backup} but invalid target could not be quarantined"
                ) from quarantine_error
            raise
        if backup is not None:
            try:
                backup.rename(target)
            except Exception as restore_error:
                raise RuntimeError(
                    f"published license verification failed and backup restoration "
                    f"failed; backup preserved at {backup}"
                ) from restore_error
        try:
            _remove_tree(quarantine)
        except OSError:
            pass
        raise verify_error

    if backup is not None:
        _remove_tree(backup)


def _collect_locked(
    dist_root: Path,
    *,
    inventory_path: Path,
    provenance_report_path: Path,
    override_policy_path: Path,
    site_packages: Path,
) -> int:
    if dist_root.is_symlink() or _is_reparse(dist_root) or not dist_root.is_dir():
        raise ValueError(f"bundle tree is missing or unsafe: {dist_root}")
    internal = dist_root / "_internal"
    if internal.is_symlink() or _is_reparse(internal) or not internal.is_dir():
        raise ValueError(f"bundle internal directory is missing or unsafe: {internal}")

    entries = load_inventory(inventory_path)
    provenance = _load_provenance(provenance_report_path, entries)
    file_overrides, aliases = _override_policy(
        override_policy_path, provenance.components
    )

    site_root_input = Path(site_packages)
    ownership = dict(build_record_ownership(site_root_input))
    refs = _distribution_refs(ownership)
    site_root = site_root_input.resolve(strict=True)

    target = internal / "licenses"
    staging = Path(
        tempfile.mkdtemp(prefix=".licenses-staging-", dir=target.parent)
    )
    try:
        collected_rows = _collect_rows(
            staging=staging,
            dist_root=dist_root,
            components=provenance.components,
            file_overrides=file_overrides,
            aliases=aliases,
            site_root=site_root,
            ownership=ownership,
            refs=refs,
            bundle_claims=provenance.bundle_files,
        )
        revalidated_ownership = dict(build_record_ownership(site_root_input))
        if revalidated_ownership != ownership:
            raise ValueError(
                "wheel RECORD ownership changed while staging license payload"
            )
        rows = _canonical_rows(collected_rows)
        indexed_components = {str(row["component_id"]) for row in rows}
        required_components = {
            component.component_id for component in provenance.components
        }
        if indexed_components != required_components:
            raise ValueError(
                "generated license index does not cover the derived component set"
            )

        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "entries": rows,
        }
        serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        text_index = _index_text(rows)
        (staging / "INDEX.json").write_text(serialized, encoding="utf-8")
        (staging / "INDEX.txt").write_text(text_index, encoding="utf-8")

        verify = lambda root: _verify_payload(
            root,
            dist_root=dist_root,
            expected_payload=payload,
            expected_text=text_index,
        )
        verify(staging)
        _publish(staging, target, verify=verify)
        staging = Path()
    finally:
        if staging != Path() and staging.exists():
            _remove_tree(staging)

    print(
        f"==> licenses: {len(required_components)} component(s), "
        f"{len(rows)} index row(s) -> {target}"
    )
    return 0


def _collect(
    dist_root: Path,
    *,
    inventory_path: Path,
    provenance_report_path: Path,
    override_policy_path: Path,
    site_packages: Path,
) -> int:
    if dist_root.is_symlink() or _is_reparse(dist_root) or not dist_root.is_dir():
        raise ValueError(f"bundle tree is missing or unsafe: {dist_root}")
    internal = dist_root / "_internal"
    if internal.is_symlink() or _is_reparse(internal) or not internal.is_dir():
        raise ValueError(f"bundle internal directory is missing or unsafe: {internal}")
    with _exclusive_license_collection(internal):
        return _collect_locked(
            dist_root,
            inventory_path=inventory_path,
            provenance_report_path=provenance_report_path,
            override_policy_path=override_policy_path,
            site_packages=site_packages,
        )


def collect(
    dist_root: Path,
    *,
    inventory_path: Path,
    provenance_report_path: Path,
    override_policy_path: Path | None = None,
    site_packages: Path | None = None,
) -> int:
    """Build the exact license payload, returning a process-style status."""

    try:
        return _collect(
            Path(dist_root),
            inventory_path=Path(inventory_path),
            provenance_report_path=Path(provenance_report_path),
            override_policy_path=(
                Path(override_policy_path)
                if override_policy_path is not None
                else Path(MANIFEST)
            ),
            site_packages=(
                Path(site_packages)
                if site_packages is not None
                else Path(SITE_PACKAGES)
            ),
        )
    except Exception as exc:
        print(f"ERROR: license payload collection failed: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dist",
        type=Path,
        default=ROOT / "dist" / "yamabiko",
        help="PyInstaller bundle root (default: dist/yamabiko)",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="analysis inventory JSON",
    )
    parser.add_argument(
        "--provenance-report",
        type=Path,
        default=DEFAULT_PROVENANCE_REPORT,
        help="generated green bundle provenance report",
    )
    parser.add_argument(
        "--override-policy",
        type=Path,
        default=MANIFEST,
        help="reviewed license override policy",
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=SITE_PACKAGES,
        help="build environment site-packages directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return collect(
        args.dist,
        inventory_path=args.inventory,
        provenance_report_path=args.provenance_report,
        override_policy_path=args.override_policy,
        site_packages=args.site_packages,
    )


if __name__ == "__main__":
    raise SystemExit(main())
