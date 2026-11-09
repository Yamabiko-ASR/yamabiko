"""Independently audit frozen-bundle ownership and license closure.

This release boundary deliberately re-derives ownership from immutable inputs
instead of trusting the report used by the license collector.  It writes a
fingerprinted pass/fail report for preflight to revalidate against the exact
candidate bytes.
"""

import argparse
import base64
import binascii
import csv
from dataclasses import dataclass
import datetime
from email.parser import BytesParser
from email.policy import default as email_policy
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import stat
import sys
import tempfile
import tomllib

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_bundle_dlls import (
    is_allowed_ci_llama_stub,
    is_system_dll,
    parse_imports,
)


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent
REPO_LICENSES = ROOT / "licenses"
PYTHON_LICENSE = Path(sys.base_prefix) / "LICENSE.txt"
_COLLECTIONS = frozenset({"scripts", "pure", "binaries", "datas", "zipfiles"})
_LINKED_COLLECTIONS = frozenset({"binaries", "datas", "zipfiles"})
_NATIVE_SUFFIXES = frozenset({".dll", ".exe", ".pyd"})
_LICENSE_ROOT = "_internal/licenses"
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
_POLICY_MODES = frozenset(
    {"distribution", "bundle-file", "python-runtime-file", "reviewed-exception"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[-_.]+")
_COMPONENT_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")
_LEGAL_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")
_PACKAGE_DLL_SEARCH_DIRECTORIES = {
    "pyside6": frozenset({"_internal/pyside6", "_internal/shiboken6"}),
    "fugashi": frozenset({"_internal/fugashi.libs"}),
    "numpy": frozenset({"_internal/numpy.libs"}),
}


@dataclass(frozen=True, order=True)
class InventoryEntry:
    collection: str
    logical_name: str
    source_path: str
    destination: str
    type_code: str

    @property
    def key(self):
        value = "\0".join(
            (
                self.collection,
                self.logical_name,
                self.source_path,
                self.destination,
                self.type_code,
            )
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class DistributionRef:
    name: str
    version: str
    dist_info_name: str


@dataclass(frozen=True)
class PolicyComponent:
    component_id: str
    name: str
    provenance_class: str
    priority: int
    globs: tuple
    license_mode: str
    license_source: str | None
    review_id: str | None
    review_reason: str | None


@dataclass(frozen=True)
class Owner:
    component_id: str
    component_name: str
    provenance_class: str
    distribution: DistributionRef | None
    license_mode: str
    license_source: str | None
    review_id: str | None
    review_reason: str | None


@dataclass(frozen=True)
class TreeFile:
    relative_path: str
    size: int
    sha256: str
    inventory_keys: tuple
    path: Path


@dataclass(frozen=True)
class RecordClaim:
    source_path: str
    distribution: DistributionRef
    hash_spec: str | None
    size: int | None


def _component_id(value):
    return _COMPONENT_RE.sub("-", value.strip()).casefold()


def _reject_duplicate_object_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _decode_json_bytes(raw, *, label):
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _load_json(path, *, label):
    try:
        raw = _read_file_bytes_stably(path)
    except OSError as exc:
        raise ValueError(f"{label} cannot be read") from exc
    return _decode_json_bytes(raw, label=label)


def _normalized_relative(value, *, label):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be a non-empty string")
    normalized_input = value.replace("\\", "/")
    posix = PurePosixPath(normalized_input)
    windows = PureWindowsPath(value)
    if (
        windows.drive
        or posix.is_absolute()
        or normalized_input.startswith("//")
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
    ):
        raise ValueError(f"{label} must be a contained relative path")
    normalized = posix.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{label} must identify a file")
    return normalized


def _is_reparse(path):
    try:
        attributes = getattr(Path(path).lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return Path(path).is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _is_regular_non_reparse_stat(info):
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISREG(info.st_mode) and not bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _file_identity(info):
    return (
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        getattr(info, "st_ctime_ns", None),
        getattr(info, "st_dev", None),
        getattr(info, "st_ino", None),
        getattr(info, "st_file_attributes", 0),
    )


def _hash_file_stably(path, *, algorithm="sha256"):
    path = Path(path)
    before = path.lstat()
    if not _is_regular_non_reparse_stat(before):
        raise ValueError(f"file is not a regular non-reparse input: {path}")
    digest = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    after = path.lstat()
    if not _is_regular_non_reparse_stat(after):
        raise ValueError(f"file became a non-regular or reparse input: {path}")
    if _file_identity(before) != _file_identity(after) or size != after.st_size:
        raise ValueError(f"file changed while hashing: {path}")
    return size, digest.hexdigest()


def _read_file_bytes_stably(path):
    path = Path(path)
    before = path.lstat()
    if not _is_regular_non_reparse_stat(before):
        raise ValueError(f"file is not a regular non-reparse input: {path}")
    chunks = []
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            chunks.append(chunk)
            size += len(chunk)
    after = path.lstat()
    if not _is_regular_non_reparse_stat(after):
        raise ValueError(f"file became a non-regular or reparse input: {path}")
    if _file_identity(before) != _file_identity(after) or size != after.st_size:
        raise ValueError(f"file changed while reading: {path}")
    return b"".join(chunks)


def _sha256_file(path):
    return _hash_file_stably(path)[1]


def _destination(collection, logical_name):
    if collection == "scripts":
        return f"EXE:{logical_name}"
    if collection == "pure":
        return f"PYZ:{logical_name}"
    if collection in _LINKED_COLLECTIONS:
        return f"_internal/{logical_name}"
    raise ValueError(f"unknown inventory collection: {collection!r}")


def _validate_source_path(source):
    if not isinstance(source, str) or not source or "\0" in source:
        raise ValueError("inventory source_path must be a non-empty string")
    for scheme in ("repo", "site-packages", "python"):
        prefix = f"{scheme}://"
        if source.startswith(prefix):
            relative = source[len(prefix) :]
            if _normalized_relative(relative, label="inventory source_path") != relative:
                raise ValueError("inventory source_path is not normalized")
            return
    external = "external://sha256/"
    if source.startswith(external):
        digest, separator, basename = source[len(external) :].partition("/")
        if not separator or not _SHA256_RE.fullmatch(digest) or "/" in basename:
            raise ValueError("inventory external source is malformed")
        normalized = _normalized_relative(
            basename, label="inventory external basename"
        )
        if normalized != basename:
            raise ValueError("inventory external source basename is not normalized")
        return
    raise ValueError("inventory source_path uses an unsupported scheme")


def _load_inventory(path, *, source_bytes=None):
    payload = (
        _load_json(path, label="bundle inventory")
        if source_bytes is None
        else _decode_json_bytes(source_bytes, label="bundle inventory")
    )
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "entries"}:
        raise ValueError("bundle inventory has malformed top-level fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported bundle inventory schema_version")
    rows = payload["entries"]
    if not isinstance(rows, list):
        raise ValueError("bundle inventory entries must be a list")
    entries = []
    keys = set()
    destinations = set()
    fields = {"collection", "logical_name", "source_path", "destination", "type_code"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"inventory entry {index} has malformed fields")
        if any(not isinstance(row[field], str) for field in fields):
            raise ValueError(f"inventory entry {index} fields must be strings")
        collection = row["collection"]
        if collection not in _COLLECTIONS:
            raise ValueError(f"inventory entry {index} has unknown collection")
        logical = _normalized_relative(row["logical_name"], label="inventory logical_name")
        if logical != row["logical_name"]:
            raise ValueError(f"inventory entry {index} logical_name is not normalized")
        _validate_source_path(row["source_path"])
        if row["destination"] != _destination(collection, logical):
            raise ValueError(f"inventory entry {index} destination is inconsistent")
        if not row["type_code"] or "\0" in row["type_code"]:
            raise ValueError(f"inventory entry {index} type_code is empty")
        entry = InventoryEntry(
            collection,
            logical,
            row["source_path"],
            row["destination"],
            row["type_code"],
        )
        if entry.key in keys:
            raise ValueError(f"duplicate inventory entry key: {entry.key}")
        folded_destination = entry.destination.casefold()
        if folded_destination in destinations:
            raise ValueError(f"duplicate inventory destination: {entry.destination}")
        keys.add(entry.key)
        destinations.add(folded_destination)
        entries.append(entry)
    return tuple(sorted(entries))


def _enumerate_tree(dist_root, entries):
    root = Path(dist_root)
    if root.is_symlink() or _is_reparse(root) or not root.is_dir():
        raise ValueError("dist root must be an existing regular directory")
    resolved_root = root.resolve(strict=True)
    destination_keys = {}
    for entry in entries:
        if entry.collection in _LINKED_COLLECTIONS:
            destination_keys.setdefault(entry.destination, []).append(entry.key)
    def walk_error(error):
        raise ValueError(f"cannot enumerate bundle tree: {error}") from error

    files = []
    seen = set()
    for current_raw, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=walk_error
    ):
        current = Path(current_raw)
        for name in directory_names:
            directory = current / name
            if _is_reparse(directory):
                raise ValueError(
                    f"bundle directory is a symlink or reparse point: "
                    f"{directory.relative_to(root).as_posix()}"
                )
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            normalized = _normalized_relative(relative, label="bundle relative path")
            folded = normalized.casefold()
            if folded in seen:
                raise ValueError(f"bundle has duplicate case-insensitive path: {normalized}")
            seen.add(folded)
            if _is_reparse(candidate) or not candidate.is_file():
                raise ValueError(f"bundle path is not a regular file: {normalized}")
            resolved = candidate.resolve(strict=True)
            try:
                contained = resolved.relative_to(resolved_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"bundle file escapes dist root: {normalized}") from exc
            if contained.casefold() != normalized.casefold():
                raise ValueError(f"bundle file resolves to a different path: {normalized}")
            size, digest = _hash_file_stably(resolved)
            files.append(
                TreeFile(
                    normalized,
                    size,
                    digest,
                    tuple(sorted(destination_keys.get(normalized, ()))),
                    resolved,
                )
            )
    return tuple(sorted(files, key=lambda item: (item.relative_path.casefold(), item.relative_path)))


def _canonical_records_digest(records):
    digest = hashlib.sha256()
    for relative, size, file_hash in sorted(records):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest()


def _tree_digest(files):
    return _canonical_records_digest(
        (item.relative_path, item.size, item.sha256) for item in files
    )


def _metadata_field(message, field_name, dist_info_name):
    values = message.get_all(field_name, [])
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise ValueError(
            f"{dist_info_name}/METADATA must contain exactly one {field_name} field"
        )
    return values[0].strip()


def _distribution_from_metadata(message, dist_info_name):
    name = _metadata_field(message, "Name", dist_info_name)
    version = _metadata_field(message, "Version", dist_info_name)
    try:
        metadata_version = Version(version)
    except InvalidVersion as exc:
        raise ValueError(
            f"{dist_info_name}/METADATA Version does not match its directory"
        ) from exc
    if not dist_info_name.casefold().endswith(".dist-info"):
        raise ValueError(f"invalid dist-info directory: {dist_info_name}")
    stem = dist_info_name[: -len(".dist-info")]
    matches = []
    for index, character in enumerate(stem):
        if character != "-":
            continue
        directory_name = stem[:index]
        directory_version = stem[index + 1 :]
        if not directory_name or not directory_version:
            continue
        try:
            parsed_version = Version(directory_version)
        except InvalidVersion:
            continue
        if (
            canonicalize_name(directory_name) == canonicalize_name(name)
            and parsed_version == metadata_version
        ):
            matches.append((directory_name, directory_version))
    if len(matches) != 1:
        raise ValueError(
            f"{dist_info_name}/METADATA Name/Version does not match its directory"
        )
    return DistributionRef(name, version, dist_info_name)


def _record_target(site_root, recorded_path):
    normalized = recorded_path.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(recorded_path)
    if windows.drive or posix.is_absolute() or normalized.startswith("//"):
        candidate = Path(recorded_path)
    else:
        candidate = Path(site_root).joinpath(*posix.parts)
    return Path(os.path.abspath(candidate))


def _reject_record_reparse(site_root, target, source_path):
    relative = Path(target).relative_to(site_root)
    current = Path(site_root)
    for part in relative.parts:
        current /= part
        if _is_reparse(current):
            raise ValueError(f"reparse point in RECORD payload path {source_path}")


def _parse_record_hash(value, source_path):
    if not value:
        return None
    algorithm, separator, encoded = value.partition("=")
    if not separator or not algorithm or not encoded or algorithm != algorithm.casefold():
        raise ValueError(f"malformed RECORD hash for {source_path}")
    if algorithm not in hashlib.algorithms_guaranteed or algorithm in {"md5", "sha1"}:
        raise ValueError(f"unsupported or weak RECORD hash for {source_path}")
    try:
        digest_size = hashlib.new(algorithm).digest_size
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unsupported RECORD hash for {source_path}") from exc
    if digest_size < 32 or not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
        raise ValueError(f"unsupported or malformed RECORD hash for {source_path}")
    try:
        decoded = base64.b64decode(
            encoded.encode("ascii") + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError, binascii.Error) as exc:
        raise ValueError(f"malformed RECORD hash for {source_path}") from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if len(decoded) != digest_size or canonical != encoded:
        raise ValueError(f"malformed RECORD hash for {source_path}")
    return f"{algorithm}={encoded}"


def _verify_record_claim(target, source_path, hash_spec, size):
    target = Path(target)
    if not target.exists():
        return False
    if _is_reparse(target) or not target.is_file():
        raise ValueError(f"actual bytes for RECORD path {source_path} are unavailable")
    algorithm = hash_spec.split("=", 1)[0] if hash_spec is not None else "sha256"
    actual_size, actual_hex = _hash_file_stably(target, algorithm=algorithm)
    if size is not None and actual_size != size:
        raise ValueError(f"actual bytes for RECORD path {source_path} do not match size")
    if hash_spec is not None:
        expected = hash_spec.split("=", 1)[1]
        actual = base64.urlsafe_b64encode(bytes.fromhex(actual_hex)).decode("ascii").rstrip("=")
        if actual != expected:
            raise ValueError(f"actual bytes for RECORD path {source_path} do not match hash")
    return True


def _canonical_shared_owner(source_path, claims):
    relative = source_path.removeprefix("site-packages://")
    top_component = _component_id(PurePosixPath(relative).parts[0])
    distributions = sorted(
        {claim.distribution for claim in claims},
        key=lambda ref: (_component_id(ref.name), ref.name.casefold(), ref.version),
    )
    named = [ref for ref in distributions if _component_id(ref.name) == top_component]
    if len(named) != 1:
        raise ValueError(f"ambiguous shared RECORD owners for {source_path}")
    return named[0]


def _build_record_ownership(site_packages):
    site_root_input = Path(site_packages)
    if not site_root_input.is_dir():
        raise ValueError("site-packages must be an existing directory")
    site_root = site_root_input.resolve(strict=True)
    claims_by_path = {}
    metadata_records = []
    candidates = sorted(
        (
            candidate
            for candidate in site_root_input.iterdir()
            if candidate.name.casefold().endswith(".dist-info")
        ),
        key=lambda candidate: (candidate.name.casefold(), candidate.name),
    )
    for dist_info_input in candidates:
        if _is_reparse(dist_info_input) or not dist_info_input.is_dir():
            raise ValueError(
                f"distribution metadata is not a regular directory: {dist_info_input.name}"
            )
        dist_info = site_root / dist_info_input.name
        metadata_path = dist_info / "METADATA"
        record_path = dist_info / "RECORD"
        raw_metadata = {}
        for name, path in (("METADATA", metadata_path), ("RECORD", record_path)):
            if _is_reparse(path) or not path.is_file():
                raise ValueError(
                    f"{dist_info.name} must contain regular files METADATA and RECORD"
                )
            raw = _read_file_bytes_stably(path)
            raw_metadata[name] = raw
            size = len(raw)
            digest = hashlib.sha256(raw).hexdigest()
            metadata_records.append(
                (path.relative_to(site_root).as_posix(), size, digest)
            )
        try:
            message = BytesParser(policy=email_policy).parsebytes(
                raw_metadata["METADATA"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot read {dist_info.name}/METADATA") from exc
        distribution = _distribution_from_metadata(message, dist_info.name)
        try:
            record_text = raw_metadata["RECORD"].decode("utf-8")
            rows = list(csv.reader(io.StringIO(record_text, newline="")))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"cannot parse {dist_info.name}/RECORD") from exc
        seen = set()
        for row_number, row in enumerate(rows, start=1):
            if len(row) != 3 or not row[0] or "\0" in row[0]:
                raise ValueError(f"malformed {dist_info.name}/RECORD row {row_number}")
            target = _record_target(site_root, row[0])
            try:
                relative = target.relative_to(site_root)
            except ValueError:
                continue
            if relative in {Path(""), Path(".")}:
                raise ValueError(f"malformed {dist_info.name}/RECORD row {row_number}")
            source_path = f"site-packages://{relative.as_posix()}"
            _reject_record_reparse(site_root, target, source_path)
            folded = source_path.casefold()
            if folded in seen:
                raise ValueError(f"duplicate RECORD path in {dist_info.name}: {source_path}")
            seen.add(folded)
            hash_spec = _parse_record_hash(row[1], source_path)
            if row[2]:
                if not re.fullmatch(r"0|[1-9][0-9]*", row[2]):
                    raise ValueError(f"malformed RECORD size for {source_path}")
                claimed_size = int(row[2])
            else:
                claimed_size = None
            if (hash_spec is None) != (claimed_size is None):
                raise ValueError(
                    f"RECORD path {source_path} must provide a complete hash and size"
                )
            if not _verify_record_claim(
                target, source_path, hash_spec, claimed_size
            ):
                continue
            claims_by_path.setdefault(folded, []).append(
                RecordClaim(source_path, distribution, hash_spec, claimed_size)
            )
    ownership = {}
    for folded, claims in sorted(claims_by_path.items()):
        source_path = min(claim.source_path for claim in claims)
        distributions = {claim.distribution for claim in claims}
        if len(distributions) == 1:
            owner = next(iter(distributions))
        else:
            signatures = {(claim.hash_spec, claim.size) for claim in claims}
            if len(signatures) != 1:
                raise ValueError(f"conflicting RECORD owners for {source_path}")
            hash_spec, size = next(iter(signatures))
            if hash_spec is None and size is None:
                continue
            owner = _canonical_shared_owner(source_path, claims)
        ownership[source_path] = owner
    return dict(sorted(ownership.items())), _canonical_records_digest(metadata_records)


def _load_policy(path, *, source_bytes=None):
    try:
        raw = (
            _read_file_bytes_stably(path)
            if source_bytes is None
            else source_bytes
        )
        payload = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("provenance policy is not valid UTF-8 TOML") from exc
    if not isinstance(payload, dict) or set(payload) - {"schema_version", "component"}:
        raise ValueError("provenance policy has unknown fields")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported provenance policy schema_version")
    rows = payload.get("component", [])
    if not isinstance(rows, list):
        raise ValueError("provenance policy components must be a list")
    components = []
    seen = set()
    required = {"id", "name", "class", "priority", "globs", "license_mode"}
    optional = {"license_source", "review_id", "review_reason"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= set(row) or set(row) - required - optional:
            raise ValueError(f"provenance policy component {index} has malformed fields")
        component_id = _component_id(row["id"]) if isinstance(row["id"], str) else ""
        if component_id != row["id"] or component_id in seen:
            raise ValueError(f"provenance policy component id is invalid: {row.get('id')!r}")
        seen.add(component_id)
        if (
            not isinstance(row["name"], str)
            or not row["name"].strip()
            or not isinstance(row["class"], str)
            or not row["class"].strip()
            or type(row["priority"]) is not int
            or not isinstance(row["globs"], list)
            or not row["globs"]
        ):
            raise ValueError(f"provenance policy component {component_id} is malformed")
        globs = tuple(
            _normalized_relative(item, label="component glob")
            for item in row["globs"]
        )
        if any(glob != raw for glob, raw in zip(globs, row["globs"])):
            raise ValueError(
                f"provenance policy component {component_id} has non-canonical globs"
            )
        if len({item.casefold() for item in globs}) != len(globs):
            raise ValueError(f"provenance policy component {component_id} has duplicate globs")
        mode = row["license_mode"]
        if not isinstance(mode, str) or mode not in _POLICY_MODES:
            raise ValueError(f"component {component_id} license_mode is invalid")
        source = row.get("license_source")
        review_id = row.get("review_id")
        review_reason = row.get("review_reason")
        if mode == "reviewed-exception":
            if source is not None or not isinstance(review_id, str) or not review_id.strip() or not isinstance(review_reason, str) or not review_reason.strip():
                raise ValueError(f"reviewed exception {component_id} is malformed")
            review_id = review_id.strip()
            review_reason = review_reason.strip()
        else:
            if not isinstance(source, str) or not source.strip() or review_id is not None or review_reason is not None:
                raise ValueError(f"component {component_id} license descriptor is malformed")
            source = source.strip()
            if mode == "distribution":
                source = _component_id(source)
            elif mode == "bundle-file":
                source = _normalized_relative(source, label="bundle license source")
            else:
                if source != "python://LICENSE.txt":
                    raise ValueError(
                        f"Python license source for {component_id} must be "
                        "python://LICENSE.txt"
                    )
        components.append(
            PolicyComponent(
                component_id,
                row["name"].strip(),
                row["class"].strip(),
                row["priority"],
                globs,
                mode,
                source,
                review_id,
                review_reason,
            )
        )
    return tuple(components)


@lru_cache(maxsize=None)
def _compiled_policy_glob(glob):
    translated = [r"\A"]
    folded = glob.casefold()
    index = 0
    while index < len(folded):
        character = folded[index]
        if character == "*":
            if index + 1 < len(folded) and folded[index + 1] == "*":
                translated.append(".*")
                index += 2
                continue
            translated.append("[^/]*")
        elif character == "?":
            translated.append("[^/]")
        else:
            translated.append(re.escape(character))
        index += 1
    translated.append(r"\Z")
    return re.compile("".join(translated))


def _policy_matches(component, path):
    return any(_compiled_policy_glob(glob).fullmatch(path.casefold()) for glob in component.globs)


def _canonical_component_id(value, *, label):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be a non-empty string")
    canonical = canonicalize_name(value)
    if value != canonical or not _COMPONENT_ID_RE.fullmatch(canonical):
        raise ValueError(f"{label} must be PEP 503-canonical: {value!r}")
    return canonical


def _distribution_index(ownership):
    grouped = {}
    for distribution in ownership.values():
        grouped.setdefault(_component_id(distribution.name), set()).add(distribution)
    result = {}
    for component_id, candidates in sorted(grouped.items()):
        if len(candidates) != 1:
            versions = ", ".join(
                sorted(f"{item.name} {item.version}" for item in candidates)
            )
            raise ValueError(
                f"multiple installed distributions normalize to {component_id}: {versions}"
            )
        result[component_id] = next(iter(candidates))
    return result


def _distribution_owner(distribution):
    component_id = _component_id(distribution.name)
    return Owner(
        component_id,
        distribution.name,
        "distribution",
        distribution,
        "distribution",
        component_id,
        None,
        None,
    )


def _project_owner():
    return Owner("yamabiko", "Yamabiko", "project", None, "project", None, None, None)


def _python_owner():
    return Owner(
        "python-runtime",
        "Python",
        "python-runtime",
        None,
        "python-runtime-file",
        "python://LICENSE.txt",
        None,
        None,
    )


def _policy_owner(component):
    return Owner(
        component.component_id,
        component.name,
        component.provenance_class,
        None,
        component.license_mode,
        component.license_source,
        component.review_id,
        component.review_reason,
    )


def _owner_signature(owner):
    return (
        owner.component_id,
        owner.component_name,
        owner.provenance_class,
        owner.distribution,
        owner.license_mode,
        owner.license_source,
        owner.review_id,
        owner.review_reason,
    )


def _is_license_payload(relative_path):
    return relative_path.casefold().startswith(f"{_LICENSE_ROOT}/")


def _resolve_ownership(entries, files, record_ownership, policy):
    product_files = tuple(item for item in files if not _is_license_payload(item.relative_path))
    ownership_folded = {key.casefold(): value for key, value in record_ownership.items()}
    distributions = _distribution_index(record_ownership)
    errors = set()
    raw_matches = {component.component_id: 0 for component in policy}
    effective_matches = {component.component_id: 0 for component in policy}

    for component in policy:
        if component.component_id in distributions:
            errors.add(
                f"policy component id {component.component_id} collides with automatic distribution id"
            )

    linked_keys = {key for item in product_files for key in item.inventory_keys}
    for entry in entries:
        if entry.collection in _LINKED_COLLECTIONS and entry.key not in linked_keys:
            errors.add(
                f"{entry.collection} inventory destination {entry.destination} has no final bundle file"
            )

    def select_policy(kind, path):
        matches = [component for component in policy if _policy_matches(component, path)]
        for component in matches:
            raw_matches[component.component_id] += 1
        if not matches:
            return None, False
        maximum = max(component.priority for component in matches)
        winners = sorted(
            (component for component in matches if component.priority == maximum),
            key=lambda component: component.component_id,
        )
        if len(winners) != 1:
            ids = ", ".join(component.component_id for component in winners)
            errors.add(f"ambiguous {kind} {path}: priority {maximum} ties between {ids}")
            return None, True
        return winners[0], True

    analysis_owners = {}
    all_owners = []
    for entry in entries:
        distribution = (
            ownership_folded.get(entry.source_path.casefold())
            if entry.source_path.startswith("site-packages://")
            else None
        )
        component, had_policy_match = select_policy("analysis-entry", entry.destination)
        if distribution is not None and had_policy_match:
            errors.add(
                f"policy cannot override RECORD-owned analysis-entry {entry.key} at {entry.destination}"
            )
            owner = _distribution_owner(distribution)
        elif component is not None:
            effective_matches[component.component_id] += 1
            owner = _policy_owner(component)
        elif had_policy_match:
            continue
        elif entry.source_path.startswith("repo://"):
            owner = _project_owner()
        elif entry.source_path.startswith("python://"):
            owner = _python_owner()
        elif distribution is not None:
            owner = _distribution_owner(distribution)
        else:
            errors.add(
                f"unowned analysis-entry {entry.key} at {entry.destination} "
                f"(source {entry.source_path})"
            )
            continue
        analysis_owners[entry.key] = owner
        all_owners.append(owner)

    final_owners = {}
    for bundle_file in product_files:
        linked = [
            analysis_owners[key]
            for key in bundle_file.inventory_keys
            if key in analysis_owners
        ]
        component, had_policy_match = select_policy(
            "bundle-file", bundle_file.relative_path
        )
        if any(owner.distribution is not None for owner in linked) and had_policy_match:
            errors.add(
                f"policy cannot override RECORD-owned bundle-file {bundle_file.relative_path}"
            )
            component = None
            had_policy_match = False
        if component is not None:
            effective_matches[component.component_id] += 1
            owner = _policy_owner(component)
        elif had_policy_match:
            continue
        elif len(linked) != len(bundle_file.inventory_keys) or not linked:
            if PurePosixPath(bundle_file.relative_path).suffix.casefold() in _NATIVE_SUFFIXES:
                errors.add(f"unowned final native binary: {bundle_file.relative_path}")
            else:
                errors.add(f"unowned bundle-file {bundle_file.relative_path}")
            continue
        else:
            signatures = {_owner_signature(owner) for owner in linked}
            if len(signatures) != 1:
                components = ", ".join(sorted(owner.component_id for owner in linked))
                errors.add(
                    f"bundle-file {bundle_file.relative_path} links conflicting owners: {components}"
                )
                continue
            owner = linked[0]
        final_owners[bundle_file.relative_path.casefold()] = owner
        all_owners.append(owner)

    product_paths = {item.relative_path.casefold() for item in product_files}
    for component in policy:
        raw = raw_matches[component.component_id]
        effective = effective_matches[component.component_id]
        if raw == 0:
            errors.add(f"stale policy component '{component.component_id}': no raw match")
        elif effective == 0:
            errors.add(
                f"stale policy component '{component.component_id}': raw matches exist "
                "but it never owns a subject"
            )
        if component.license_mode == "distribution" and component.license_source not in distributions:
            errors.add(
                f"policy component {component.component_id} references unavailable "
                f"distribution {component.license_source}"
            )
        if (
            component.license_mode == "bundle-file"
            and (component.license_source or "").casefold() not in product_paths
        ):
            errors.add(
                f"policy component {component.component_id} references missing bundle "
                f"license {component.license_source}"
            )

    components = {}
    for owner in all_owners:
        if owner.license_mode == "project":
            continue
        existing = components.get(owner.component_id)
        if existing is not None and _owner_signature(existing) != _owner_signature(owner):
            errors.add(
                f"component id {owner.component_id} has inconsistent descriptor signatures"
            )
        else:
            components[owner.component_id] = owner
    return (
        dict(sorted(components.items())),
        final_owners,
        product_files,
        distributions,
        tuple(sorted(errors)),
    )


def _native_domain(relative_path):
    return "llama" if relative_path.casefold().startswith("_internal/llama/") else "main"


def _bundle_directory(relative_path):
    parent = PurePosixPath(relative_path).parent.as_posix()
    return "" if parent == "." else parent.casefold()


def _native_search_directories(importer):
    if _native_domain(importer.relative_path) == "llama":
        return {"_internal/llama"}
    directories = {"", "_internal", _bundle_directory(importer.relative_path)}
    parts = PurePosixPath(importer.relative_path).parts
    if len(parts) >= 2 and parts[0].casefold() == "_internal":
        directories.update(_PACKAGE_DLL_SEARCH_DIRECTORIES.get(parts[1].casefold(), ()))
    return directories


def _owned_import_target(import_name, importer, bundle_files, owned_paths):
    basename = PureWindowsPath(import_name).name.casefold()
    search_directories = _native_search_directories(importer)
    dlls = [
        item
        for item in bundle_files
        if item.relative_path.casefold().endswith(".dll")
        and _bundle_directory(item.relative_path) in search_directories
    ]
    candidates = [
        item
        for item in dlls
        if PurePosixPath(item.relative_path).name.casefold() == basename
    ]
    if not candidates and basename.endswith(".dll"):
        stem = re.escape(basename[:-4])
        hashed = re.compile(rf"^{stem}-[0-9a-f]{{32}}\.dll\Z", re.IGNORECASE)
        candidates = [
            item
            for item in dlls
            if hashed.fullmatch(PurePosixPath(item.relative_path).name)
        ]
    return any(item.relative_path.casefold() in owned_paths for item in candidates)


def _audit_native_closure(product_files, final_owners):
    errors = set()
    owned_paths = set(final_owners)
    for bundle_file in product_files:
        if PurePosixPath(bundle_file.relative_path).suffix.casefold() not in _NATIVE_SUFFIXES:
            continue
        try:
            actual_size, actual_hash = _hash_file_stably(bundle_file.path)
        except (OSError, ValueError):
            errors.add(f"native bundle-file {bundle_file.relative_path} cannot be re-verified")
            continue
        if actual_size != bundle_file.size or actual_hash != bundle_file.sha256:
            errors.add(f"native bundle-file {bundle_file.relative_path} changed after enumeration")
            continue
        if is_allowed_ci_llama_stub(bundle_file.relative_path, bundle_file.path):
            # A deliberate CI text stub has no imports to close over.
            continue
        try:
            imports = parse_imports(bundle_file.path)
        except Exception:
            errors.add(
                f"native bundle-file {bundle_file.relative_path} PE imports could not be parsed"
            )
            continue
        by_name = {
            name.casefold(): name for name in imports if isinstance(name, str) and name
        }
        for folded_name in sorted(by_name):
            import_name = by_name[folded_name]
            if is_system_dll(import_name):
                continue
            if not _owned_import_target(
                import_name, bundle_file, product_files, owned_paths
            ):
                errors.add(
                    f"PE import {import_name} from {bundle_file.relative_path} "
                    "has no present owner"
                )
    return tuple(sorted(errors))


def _policy_relative_path(value):
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


def _parse_manifest(source):
    entries = []
    seen_files = set()
    aliases = {}
    for raw in source.splitlines():
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


def _override_policy(path, components, *, source_bytes=None):
    try:
        raw = (
            _read_file_bytes_stably(path)
            if source_bytes is None
            else source_bytes
        )
        entries = _parse_manifest(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("cannot read license override policy") from exc
    component_ids = set(components)
    file_lists = {}
    aliases = {}
    for kind, owner, value in entries:
        if owner not in component_ids:
            raise ValueError(f"stale license override component: {owner}")
        if kind == "file":
            if components[owner].license_mode != "distribution":
                raise ValueError(
                    f"license file override for non-distribution component: {owner}"
                )
            file_lists.setdefault(owner, []).append(value)
        else:
            if value not in component_ids:
                raise ValueError(f"stale license alias target: {value}")
            if (
                components[owner].license_mode != "distribution"
                or components[value].license_mode != "distribution"
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


def _is_legal_relative(relative):
    path = PurePosixPath(relative)
    return (
        bool(path.parts) and path.parts[0].casefold() == "licenses"
    ) or path.name.upper().startswith(_LEGAL_PREFIXES)


def _legal_files(site_packages, ref, ownership):
    site_root_input = Path(site_packages)
    site_root = site_root_input.resolve(strict=True)
    dist_info_input = site_root_input / ref.dist_info_name
    if _is_reparse(dist_info_input) or not dist_info_input.is_dir():
        raise ValueError(
            f"distribution metadata is not a regular directory: {ref.dist_info_name}"
        )
    dist_info = site_root / ref.dist_info_name
    ownership_folded = {key.casefold(): value for key, value in ownership.items()}
    found = []
    seen = set()

    def walk_error(error):
        raise ValueError(
            f"cannot enumerate legal files for {ref.dist_info_name}: {error}"
        ) from error

    candidates = []
    for current_raw, directory_names, file_names in os.walk(
        dist_info, topdown=True, followlinks=False, onerror=walk_error
    ):
        current = Path(current_raw)
        directory_names.sort(key=lambda value: (value.casefold(), value))
        file_names.sort(key=lambda value: (value.casefold(), value))
        for name in directory_names:
            directory = current / name
            if _is_reparse(directory) or not directory.is_dir():
                raise ValueError(f"reparse or non-directory in license source: {directory}")
        for name in file_names:
            path = current / name
            if _is_reparse(path) or not path.is_file():
                raise ValueError(f"non-regular file in license source: {path}")
            candidates.append(path)

    for path in sorted(
        candidates,
        key=lambda value: (
            value.relative_to(dist_info).as_posix().casefold(),
            value.relative_to(dist_info).as_posix(),
        ),
    ):
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
        if ownership_folded.get(source_path.casefold()) != ref:
            raise ValueError(
                f"legal file lacks a complete matching RECORD claim: {source_path}"
            )
        found.append((path, relative, source_path))
    return tuple(found)


def _regular_repo_license(relative):
    root = Path(REPO_LICENSES)
    if _is_reparse(root) or not root.is_dir():
        raise ValueError("repository license directory is missing")
    resolved_root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if _is_reparse(current):
            raise ValueError(f"reparse point in repository license path: {relative}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"repository license file is missing: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"repository license file is not regular: {relative}")
    return resolved


def _license_row(owner, *, version, source_path, output_path, sha256):
    return {
        "component_id": owner.component_id,
        "component_name": owner.component_name,
        "version": version,
        "provenance_class": owner.provenance_class,
        "license_mode": owner.license_mode,
        "source_path": source_path,
        "output_path": output_path,
        "sha256": sha256,
        "review_id": owner.review_id,
        "review_reason": owner.review_reason,
    }


def _canonical_rows(rows):
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row["component_id"],
                "" if row["output_path"] is None else row["output_path"].casefold(),
                "" if row["output_path"] is None else row["output_path"],
            ),
        )
    )


def _render_index_text(rows):
    lines = [
        "Third-party component and license-text index for this Yamabiko bundle.",
        "Generated from the exact bundle inventory; do not edit by hand.",
        "",
    ]
    for row in rows:
        component = row["component_id"]
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


def _expected_license_rows(
    *,
    components,
    files,
    distributions,
    ownership,
    overrides_path,
    overrides_bytes=None,
    site_packages,
):
    file_overrides, aliases = _override_policy(
        overrides_path, components, source_bytes=overrides_bytes
    )
    file_by_path = {item.relative_path.casefold(): item for item in files}
    rows = []
    for component_id, owner in sorted(components.items()):
        if owner.license_mode == "reviewed-exception":
            rows.append(
                _license_row(
                    owner,
                    version=None,
                    source_path=None,
                    output_path=None,
                    sha256=None,
                )
            )
            continue
        if owner.license_mode == "bundle-file":
            source = owner.license_source
            item = file_by_path.get((source or "").casefold())
            if item is None or item.relative_path != source:
                raise ValueError(
                    f"bundle license source is absent from final tree: {source}"
                )
            rows.append(
                _license_row(
                    owner,
                    version=None,
                    source_path=source,
                    output_path=source,
                    sha256=item.sha256,
                )
            )
            continue
        if owner.license_mode == "python-runtime-file":
            source = Path(PYTHON_LICENSE)
            if _is_reparse(source) or not source.is_file():
                raise ValueError("Python runtime license is missing")
            version = platform.python_version()
            output = (
                f"{_LICENSE_ROOT}/components/{component_id}/{version}/LICENSE.txt"
            )
            rows.append(
                _license_row(
                    owner,
                    version=version,
                    source_path="python://LICENSE.txt",
                    output_path=output,
                    sha256=_sha256_file(source),
                )
            )
            continue
        if owner.license_mode != "distribution" or owner.license_source is None:
            raise ValueError(f"component {component_id} has unsupported license mode")
        ref = distributions.get(owner.license_source)
        if ref is None:
            raise ValueError(
                f"component {component_id} references unavailable distribution "
                f"{owner.license_source}"
            )
        if (
            owner.distribution is not None
            and owner.license_source == owner.component_id
            and owner.distribution != ref
        ):
            raise ValueError(
                f"component {component_id} ownership does not match installed metadata"
            )
        override_owner = component_id
        if override_owner not in file_overrides:
            override_owner = aliases.get(component_id, "")
        override_files = file_overrides.get(override_owner, ())
        if override_files:
            for relative in override_files:
                source = _regular_repo_license(relative)
                output = (
                    f"{_LICENSE_ROOT}/components/{component_id}/override/{relative}"
                )
                rows.append(
                    _license_row(
                        owner,
                        version=ref.version,
                        source_path=f"repo://licenses/{relative}",
                        output_path=output,
                        sha256=_sha256_file(source),
                    )
                )
            continue
        legal_files = _legal_files(site_packages, ref, ownership)
        if not legal_files:
            raise ValueError(
                f"derived component {component_id} has no wheel legal file or reviewed override"
            )
        for source, relative, source_path in legal_files:
            output = (
                f"{_LICENSE_ROOT}/components/{component_id}/{ref.version}/{relative}"
            )
            rows.append(
                _license_row(
                    owner,
                    version=ref.version,
                    source_path=source_path,
                    output_path=output,
                    sha256=_sha256_file(source),
                )
            )
    return _canonical_rows(rows)


def _optional_nonempty_string(value, *, label):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{label} must be null or a non-empty string")
    return value


def _load_license_index(path, *, source_bytes=None):
    payload = (
        _load_json(path, label="license index")
        if source_bytes is None
        else _decode_json_bytes(source_bytes, label="license index")
    )
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "entries"}:
        raise ValueError("license index has malformed top-level fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported license index schema_version")
    raw_rows = payload["entries"]
    if not isinstance(raw_rows, list):
        raise ValueError("license index entries must be a list")
    rows = []
    identities = set()
    outputs = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict) or set(row) != _INDEX_FIELDS:
            raise ValueError(f"license index row {index} has malformed fields")
        component_id = _canonical_component_id(
            row["component_id"], label=f"license index row {index} component_id"
        )
        component_name = _optional_nonempty_string(
            row["component_name"], label=f"license index row {index} component name"
        )
        provenance_class = _optional_nonempty_string(
            row["provenance_class"],
            label=f"license index row {index} provenance class",
        )
        if component_name is None or provenance_class is None:
            raise ValueError(f"license index row {index} has null descriptor fields")
        version = _optional_nonempty_string(
            row["version"], label=f"license index row {index} version"
        )
        source_path = _optional_nonempty_string(
            row["source_path"], label=f"license index row {index} source path"
        )
        output_path = _optional_nonempty_string(
            row["output_path"], label=f"license index row {index} output path"
        )
        digest = _optional_nonempty_string(
            row["sha256"], label=f"license index row {index} sha256"
        )
        review_id = _optional_nonempty_string(
            row["review_id"], label=f"license index row {index} review id"
        )
        review_reason = _optional_nonempty_string(
            row["review_reason"], label=f"license index row {index} review reason"
        )
        mode = row["license_mode"]
        if not isinstance(mode, str) or mode not in _POLICY_MODES:
            raise ValueError(f"license index row {index} license_mode is invalid")
        if output_path is None:
            if (
                mode != "reviewed-exception"
                or source_path is not None
                or digest is not None
                or version is not None
                or review_id is None
                or review_reason is None
            ):
                raise ValueError(f"license index row {index} reviewed exception is malformed")
            identity_output = None
        else:
            normalized = _normalized_relative(
                output_path, label=f"license index row {index} output path"
            )
            if normalized != output_path:
                raise ValueError(f"license index row {index} output path is not normalized")
            if source_path is None or digest is None or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"license index row {index} output descriptor is malformed")
            if review_id is not None or review_reason is not None:
                raise ValueError(f"license index row {index} has unexpected review fields")
            folded_output = output_path.casefold()
            if folded_output in outputs:
                raise ValueError(f"duplicate license index output path: {output_path}")
            outputs.add(folded_output)
            identity_output = folded_output
        identity = (component_id, identity_output)
        if identity in identities:
            raise ValueError(f"duplicate license index row identity: {component_id}")
        identities.add(identity)
        rows.append(dict(row))
    canonical = _canonical_rows(rows)
    if tuple(rows) != canonical:
        raise ValueError("license index entries are not in canonical order")
    return canonical


def _row_identity(row):
    output = row["output_path"]
    return (row["component_id"], None if output is None else output.casefold())


def _audit_license_index(
    *,
    dist_root,
    license_index_path,
    license_index_bytes=None,
    files,
    expected_rows,
    components,
):
    errors = set()
    root = Path(dist_root).resolve(strict=True)
    expected_index = root / "_internal" / "licenses" / "INDEX.json"
    try:
        actual_index = Path(license_index_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("license index is missing") from exc
    if actual_index != expected_index:
        raise ValueError("license index path is not the bundle license index")
    rows = _load_license_index(actual_index, source_bytes=license_index_bytes)
    required_ids = set(components)
    indexed_ids = {row["component_id"] for row in rows}
    for component_id in sorted(required_ids - indexed_ids):
        errors.add(f"missing license index component: {component_id}")
    for component_id in sorted(indexed_ids - required_ids):
        errors.add(f"extra license index component: {component_id}")

    expected_by_identity = {_row_identity(row): row for row in expected_rows}
    actual_by_identity = {_row_identity(row): row for row in rows}
    for identity in sorted(
        expected_by_identity.keys() - actual_by_identity.keys(),
        key=lambda value: (value[0], "" if value[1] is None else value[1]),
    ):
        errors.add(
            f"missing license index row: {identity[0]} "
            f"{identity[1] if identity[1] is not None else '<reviewed-exception>'}"
        )
    for identity in sorted(
        actual_by_identity.keys() - expected_by_identity.keys(),
        key=lambda value: (value[0], "" if value[1] is None else value[1]),
    ):
        errors.add(
            f"extra license index row: {identity[0]} "
            f"{identity[1] if identity[1] is not None else '<reviewed-exception>'}"
        )
    for identity in sorted(
        expected_by_identity.keys() & actual_by_identity.keys(),
        key=lambda value: (value[0], "" if value[1] is None else value[1]),
    ):
        if expected_by_identity[identity] != actual_by_identity[identity]:
            errors.add(
                f"license index row mismatch: {identity[0]} "
                f"{identity[1] if identity[1] is not None else '<reviewed-exception>'}"
            )

    file_by_path = {item.relative_path.casefold(): item for item in files}
    for row in rows:
        output = row["output_path"]
        if output is None:
            continue
        item = file_by_path.get(output.casefold())
        if item is None or item.relative_path != output:
            errors.add(f"indexed license output is missing: {output}")
        elif item.sha256 != row["sha256"]:
            errors.add(f"indexed license output hash mismatch: {output}")

    text_relative = f"{_LICENSE_ROOT}/INDEX.txt"
    text_item = file_by_path.get(text_relative.casefold())
    if text_item is not None:
        if text_item.relative_path != text_relative:
            errors.add(
                "human-readable license index path has non-canonical casing"
            )
        try:
            raw_text = _read_file_bytes_stably(text_item.path)
            verified_hash = hashlib.sha256(raw_text).hexdigest()
            if verified_hash != text_item.sha256:
                errors.add("human-readable license index changed during audit")
            else:
                actual_text = (
                    raw_text.decode("utf-8")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )
                if actual_text != _render_index_text(expected_rows):
                    errors.add(
                        "human-readable license index does not match derived closure"
                    )
        except (OSError, UnicodeDecodeError, ValueError):
            errors.add("human-readable license index is unreadable or unstable")

    expected_inside = {
        f"{_LICENSE_ROOT}/INDEX.json".casefold(),
        f"{_LICENSE_ROOT}/INDEX.txt".casefold(),
    }
    for row in rows:
        output = row["output_path"]
        if isinstance(output, str) and _is_license_payload(output):
            expected_inside.add(output.casefold())
    actual_inside = {
        item.relative_path.casefold(): item.relative_path
        for item in files
        if _is_license_payload(item.relative_path)
    }
    for folded in sorted(expected_inside - set(actual_inside)):
        errors.add(f"indexed license payload file is missing: {folded}")
    for folded in sorted(set(actual_inside) - expected_inside):
        errors.add(f"unindexed license payload file: {actual_inside[folded]}")
    return tuple(sorted(errors))


def _audit_once(
    *,
    dist_root,
    inventory_path,
    license_index_path,
    provenance_policy_config_path,
    license_override_policy_path,
    site_packages,
):
    inventory_bytes = _read_file_bytes_stably(inventory_path)
    index_bytes = _read_file_bytes_stably(license_index_path)
    policy_bytes = _read_file_bytes_stably(provenance_policy_config_path)
    override_bytes = _read_file_bytes_stably(license_override_policy_path)
    entries = _load_inventory(inventory_path, source_bytes=inventory_bytes)
    files = _enumerate_tree(dist_root, entries)
    ownership, metadata_digest = _build_record_ownership(site_packages)
    policy = _load_policy(
        provenance_policy_config_path, source_bytes=policy_bytes
    )
    components, final_owners, product_files, distributions, ownership_errors = (
        _resolve_ownership(entries, files, ownership, policy)
    )
    errors = set(ownership_errors)
    errors.update(_audit_native_closure(product_files, final_owners))
    expected_rows = _expected_license_rows(
        components=components,
        files=files,
        distributions=distributions,
        ownership=ownership,
        overrides_path=license_override_policy_path,
        overrides_bytes=override_bytes,
        site_packages=site_packages,
    )
    errors.update(
        _audit_license_index(
            dist_root=dist_root,
            license_index_path=license_index_path,
            license_index_bytes=index_bytes,
            files=files,
            expected_rows=expected_rows,
            components=components,
        )
    )
    semantic_fingerprints = {
        "bundle_tree_sha256": _tree_digest(files),
        "bundle_file_count": len(files),
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "license_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "provenance_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "license_override_policy_sha256": hashlib.sha256(override_bytes).hexdigest(),
        "distribution_metadata_sha256": metadata_digest,
    }
    return tuple(sorted(errors)), semantic_fingerprints


def _run_audit(**kwargs):
    try:
        return _audit_once(**kwargs)
    except (OSError, ValueError) as exc:
        return (f"license closure audit failed: {exc}",), None


def audit_license_closure(
    *,
    dist_root,
    inventory_path,
    license_index_path,
    provenance_policy_config_path,
    license_override_policy_path,
    site_packages,
):
    """Return sorted closure errors after independently deriving all owners."""

    errors, _semantic_fingerprints = _run_audit(
        dist_root=Path(dist_root),
        inventory_path=Path(inventory_path),
        license_index_path=Path(license_index_path),
        provenance_policy_config_path=Path(provenance_policy_config_path),
        license_override_policy_path=Path(license_override_policy_path),
        site_packages=Path(site_packages),
    )
    return errors


def _distribution_metadata_fingerprint(site_packages):
    site_root_input = Path(site_packages)
    if not site_root_input.is_dir():
        raise ValueError("site-packages must be an existing directory")
    site_root = site_root_input.resolve(strict=True)
    candidates = sorted(
        (
            candidate
            for candidate in site_root_input.iterdir()
            if candidate.name.casefold().endswith(".dist-info")
        ),
        key=lambda candidate: (candidate.name.casefold(), candidate.name),
    )
    seen = set()
    records = []
    for dist_info_input in candidates:
        folded = dist_info_input.name.casefold()
        if folded in seen:
            raise ValueError(
                f"duplicate case-insensitive dist-info directory: {dist_info_input.name}"
            )
        seen.add(folded)
        if _is_reparse(dist_info_input) or not dist_info_input.is_dir():
            raise ValueError(
                f"distribution metadata is not a regular directory: {dist_info_input.name}"
            )
        dist_info = site_root / dist_info_input.name
        for name in ("METADATA", "RECORD"):
            path = dist_info / name
            if _is_reparse(path) or not path.is_file():
                raise ValueError(
                    f"{dist_info.name} must contain regular files METADATA and RECORD"
                )
            size, digest = _hash_file_stably(path)
            records.append((path.relative_to(site_root).as_posix(), size, digest))
    return _canonical_records_digest(records)


def _audit_fingerprints(
    *,
    dist_root,
    inventory_path,
    license_index_path,
    provenance_policy_config_path,
    license_override_policy_path,
    site_packages,
):
    files = _enumerate_tree(dist_root, ())
    return {
        "bundle_tree_sha256": _tree_digest(files),
        "bundle_file_count": len(files),
        "inventory_sha256": _sha256_file(inventory_path),
        "license_index_sha256": _sha256_file(license_index_path),
        "provenance_policy_sha256": _sha256_file(provenance_policy_config_path),
        "license_override_policy_sha256": _sha256_file(license_override_policy_path),
        "distribution_metadata_sha256": _distribution_metadata_fingerprint(
            site_packages
        ),
    }


def _invalidate_output(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        raise ValueError("closure audit output path is a directory")
    if path.exists() or path.is_symlink():
        path.unlink()


def _atomic_write_json(path, payload):
    path = Path(path)
    serialized = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--license-index", type=Path, required=True)
    parser.add_argument("--provenance-policy-config", type=Path, required=True)
    parser.add_argument("--license-override-policy", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        _invalidate_output(args.output)
        before = _audit_fingerprints(
            dist_root=args.dist,
            inventory_path=args.inventory,
            license_index_path=args.license_index,
            provenance_policy_config_path=args.provenance_policy_config,
            license_override_policy_path=args.license_override_policy,
            site_packages=args.site_packages,
        )
    except (OSError, ValueError) as exc:
        print(f"license closure inputs could not be fingerprinted: {exc}", file=sys.stderr)
        return 1

    audit_errors, semantic = _run_audit(
        dist_root=args.dist,
        inventory_path=args.inventory,
        license_index_path=args.license_index,
        provenance_policy_config_path=args.provenance_policy_config,
        license_override_policy_path=args.license_override_policy,
        site_packages=args.site_packages,
    )
    errors = set(audit_errors)
    try:
        after = _audit_fingerprints(
            dist_root=args.dist,
            inventory_path=args.inventory,
            license_index_path=args.license_index,
            provenance_policy_config_path=args.provenance_policy_config,
            license_override_policy_path=args.license_override_policy,
            site_packages=args.site_packages,
        )
    except (OSError, ValueError) as exc:
        errors.add(f"audit inputs changed or became unreadable during audit: {exc}")
        after = before
    for key in sorted(before):
        if before[key] != after[key]:
            errors.add(f"audit input changed during audit: {key}")
        if semantic is not None and (
            semantic[key] != before[key] or semantic[key] != after[key]
        ):
            errors.add(f"semantic audit snapshot mismatch: {key}")

    canonical_errors = sorted(errors)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not canonical_errors else "fail",
        "audited_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        **after,
        "errors": canonical_errors,
    }
    try:
        _atomic_write_json(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"cannot write license closure audit report: {exc}", file=sys.stderr)
        return 1
    for error in canonical_errors:
        print(error, file=sys.stderr)
    return 0 if not canonical_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
