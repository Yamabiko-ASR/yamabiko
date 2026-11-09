"""Deterministic, host-path-free inventory for PyInstaller Analysis TOCs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Iterable


SCHEMA_VERSION = 1

_COLLECTIONS = ("scripts", "pure", "binaries", "datas", "zipfiles")
_INTERNAL_COLLECTIONS = frozenset({"binaries", "datas", "zipfiles"})
_ENTRY_FIELDS = frozenset(
    {"collection", "logical_name", "source_path", "destination", "type_code"}
)
_EXTERNAL_PREFIX = "external://sha256/"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, order=True)
class InventoryEntry:
    collection: str
    logical_name: str
    source_path: str
    destination: str
    type_code: str

    @property
    def key(self) -> str:
        payload = "\0".join(
            (
                self.collection,
                self.logical_name,
                self.source_path,
                self.destination,
                self.type_code,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InventoryRoots:
    repo_root: Path
    site_packages: Path
    python_root: Path


def _normalize_relative_path(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"inventory entry {field} must be a non-empty string")

    windows_path = PureWindowsPath(value)
    normalized_input = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized_input)
    if windows_path.drive or posix_path.is_absolute() or normalized_input.startswith("//"):
        raise ValueError(f"inventory entry {field} must be relative: {value!r}")
    if ".." in posix_path.parts:
        raise ValueError(f"inventory entry {field} cannot contain '..': {value!r}")
    if any(":" in part for part in posix_path.parts):
        raise ValueError(f"inventory entry {field} cannot contain ':': {value!r}")

    normalized = posix_path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"inventory entry {field} must name a file or module")
    return normalized


def _destination(collection: str, logical_name: str) -> str:
    if collection == "scripts":
        return f"EXE:{logical_name}"
    if collection == "pure":
        return f"PYZ:{logical_name}"
    if collection in _INTERNAL_COLLECTIONS:
        return f"_internal/{logical_name}"
    raise ValueError(f"inventory entry has unknown collection: {collection!r}")


def _destination_ownership_key(destination: str) -> str:
    """Return the conservative Windows identity for a bundle destination."""

    return destination.casefold()


def _resolved_roots(roots: InventoryRoots) -> tuple[tuple[str, Path, int], ...]:
    declared = (
        ("site-packages", roots.site_packages, 0),
        ("python", roots.python_root, 1),
        ("repo", roots.repo_root, 2),
    )
    resolved: list[tuple[str, Path, int]] = []
    for scheme, root, tie_breaker in declared:
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError(f"inventory root {scheme!r} must be absolute: {root_path}")
        if not root_path.is_dir():
            raise ValueError(
                f"inventory root {scheme!r} must be an existing directory: {root_path}"
            )
        resolved.append((scheme, root_path.resolve(strict=True), tie_breaker))
    return tuple(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_source(source: object, roots: tuple[tuple[str, Path, int], ...]) -> str:
    try:
        source_path = Path(os.fspath(source))
    except TypeError as exc:
        raise ValueError("PyInstaller TOC source path must be path-like") from exc
    if not source_path.is_absolute():
        raise ValueError(f"PyInstaller TOC source path must be absolute: {source_path}")

    try:
        resolved_source = source_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"PyInstaller TOC source must be an existing regular file: {source_path}"
        ) from exc
    if not resolved_source.is_file():
        raise ValueError(
            f"PyInstaller TOC source must be an existing regular file: {source_path}"
        )

    matches: list[tuple[int, int, str, str]] = []
    for scheme, root, tie_breaker in roots:
        try:
            relative = resolved_source.relative_to(root)
        except ValueError:
            continue
        matches.append(
            (
                len(root.parts),
                -tie_breaker,
                scheme,
                _normalize_relative_path(relative.as_posix(), field="source_path"),
            )
        )

    if matches:
        _depth, _tie_breaker, scheme, relative = max(matches)
        return f"{scheme}://{relative}"

    basename = _normalize_relative_path(resolved_source.name, field="source_path")
    return f"{_EXTERNAL_PREFIX}{_sha256_file(resolved_source)}/{basename}"


def _validate_source_path(source_path: str) -> None:
    for scheme in ("repo", "site-packages", "python"):
        prefix = f"{scheme}://"
        if source_path.startswith(prefix):
            relative = source_path[len(prefix) :]
            if _normalize_relative_path(relative, field="source_path") != relative:
                raise ValueError(
                    f"inventory entry source_path is not normalized: {source_path!r}"
                )
            return

    if source_path.startswith(_EXTERNAL_PREFIX):
        remainder = source_path[len(_EXTERNAL_PREFIX) :]
        digest, separator, basename = remainder.partition("/")
        if not separator or not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"inventory entry source_path has malformed external digest: {source_path!r}"
            )
        if (
            _normalize_relative_path(basename, field="source_path") != basename
            or "/" in basename
        ):
            raise ValueError(
                f"inventory entry source_path has malformed external basename: {source_path!r}"
            )
        return

    raise ValueError(
        f"inventory entry source_path must use a canonical source scheme: {source_path!r}"
    )


def _validate_entry(entry: InventoryEntry) -> None:
    values = asdict(entry)
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("inventory entry fields must all be strings")
    if entry.collection not in _COLLECTIONS:
        raise ValueError(
            f"inventory entry has unknown collection: {entry.collection!r}"
        )
    logical_name = _normalize_relative_path(
        entry.logical_name, field="logical_name"
    )
    if logical_name != entry.logical_name:
        raise ValueError(
            f"inventory entry logical_name is not normalized: {entry.logical_name!r}"
        )
    _validate_source_path(entry.source_path)
    if entry.destination != _destination(entry.collection, entry.logical_name):
        raise ValueError(
            f"inventory entry destination does not match collection rule: {entry.destination!r}"
        )
    if not entry.type_code or "\0" in entry.type_code:
        raise ValueError("inventory entry type_code must be a non-empty string")


def analysis_entries(
    analysis: object, *, roots: InventoryRoots
) -> tuple[InventoryEntry, ...]:
    """Convert all supported Analysis TOCs to sorted, canonical entries."""

    canonical_roots = _resolved_roots(roots)
    by_key: dict[str, InventoryEntry] = {}
    by_destination: dict[str, InventoryEntry] = {}
    for collection in _COLLECTIONS:
        try:
            toc: Iterable[object] = getattr(analysis, collection)
        except AttributeError as exc:
            raise ValueError(
                f"PyInstaller Analysis is missing collection {collection!r}"
            ) from exc

        for row in toc:
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise ValueError(
                    f"{collection} must contain a 3-item TOC row, got {row!r}"
                )
            logical_raw, source_raw, type_code = row
            if not isinstance(logical_raw, str):
                raise ValueError(f"{collection} TOC logical name must be a string")
            if not isinstance(type_code, str) or not type_code or "\0" in type_code:
                raise ValueError(f"{collection} TOC type code must be a non-empty string")

            logical_name = _normalize_relative_path(
                logical_raw, field="logical_name"
            )
            entry = InventoryEntry(
                collection=collection,
                logical_name=logical_name,
                source_path=_canonical_source(source_raw, canonical_roots),
                destination=_destination(collection, logical_name),
                type_code=type_code,
            )
            existing = by_key.get(entry.key)
            if existing is not None and existing != entry:
                raise ValueError("inventory entry key collision")
            destination_key = _destination_ownership_key(entry.destination)
            destination_owner = by_destination.get(destination_key)
            if destination_owner is not None and destination_owner != entry:
                raise ValueError(
                    f"inventory destination {entry.destination!r} maps to distinct entries"
                )
            by_key[entry.key] = entry
            by_destination[destination_key] = entry

    return tuple(sorted(by_key.values()))


def write_analysis_inventory(
    analysis: object, output: Path, *, roots: InventoryRoots
) -> tuple[InventoryEntry, ...]:
    """Write deterministic inventory JSON and return its canonical entries."""

    entries = analysis_entries(analysis, roots=roots)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [asdict(entry) for entry in entries],
    }
    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized.encode("utf-8"))
    return entries


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_inventory(path: Path) -> tuple[InventoryEntry, ...]:
    """Load an inventory while failing closed on malformed or unsafe rows."""

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"inventory is not valid UTF-8 JSON: {path}") from exc

    if not isinstance(payload, dict) or set(payload) != {"schema_version", "entries"}:
        raise ValueError("inventory must contain only schema_version and entries")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported inventory schema_version: {payload['schema_version']!r}"
        )
    rows = payload["entries"]
    if not isinstance(rows, list):
        raise ValueError("inventory entries must be a list")

    entries: list[InventoryEntry] = []
    keys: set[str] = set()
    destinations: dict[str, InventoryEntry] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _ENTRY_FIELDS:
            raise ValueError(f"inventory entry {index} has malformed fields")
        if any(not isinstance(row[field], str) for field in _ENTRY_FIELDS):
            raise ValueError(f"inventory entry {index} fields must all be strings")

        entry = InventoryEntry(
            collection=row["collection"],
            logical_name=row["logical_name"],
            source_path=row["source_path"],
            destination=row["destination"],
            type_code=row["type_code"],
        )
        try:
            _validate_entry(entry)
        except ValueError as exc:
            raise ValueError(f"inventory entry {index} is malformed: {exc}") from exc
        if entry.key in keys:
            raise ValueError(f"duplicate inventory entry key: {entry.key}")
        destination_key = _destination_ownership_key(entry.destination)
        destination_owner = destinations.get(destination_key)
        if destination_owner is not None and destination_owner != entry:
            raise ValueError(
                f"inventory destination {entry.destination!r} maps to distinct entries"
            )
        keys.add(entry.key)
        destinations[destination_key] = entry
        entries.append(entry)

    return tuple(sorted(entries))
