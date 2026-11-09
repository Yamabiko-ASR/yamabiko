"""Resolve wheel, generated, and vendored ownership for a frozen bundle.

Analysis entries and final-tree files are distinct provenance subjects.  Wheel
ownership comes from installed ``RECORD`` files; immutable policy is reserved
for generated or manually copied files whose owner cannot be derived from a
wheel.  The command-line resolver writes a report only after the complete
ownership and native-import closure is green.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_policy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import tempfile
import tomllib
from typing import Iterable, Literal, Mapping, Sequence

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


# Direct execution puts scripts/ rather than the repository root on sys.path.
# Bootstrap before sibling imports so ``python -I scripts/...`` is supported.
if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_bundle_dlls import (
    is_allowed_ci_llama_stub,
    is_system_dll,
    parse_imports,
)
from scripts.bundle_inventory import InventoryEntry, load_inventory


SCHEMA_VERSION = 1

_LINKED_COLLECTIONS = frozenset({"binaries", "datas", "zipfiles"})
_NATIVE_SUFFIXES = frozenset({".dll", ".exe", ".pyd"})
_LICENSE_ROOT = "_internal/licenses"
_PACKAGE_DLL_SEARCH_DIRECTORIES = {
    "pyside6": frozenset({"_internal/pyside6", "_internal/shiboken6"}),
    "fugashi": frozenset({"_internal/fugashi.libs"}),
    "numpy": frozenset({"_internal/numpy.libs"}),
}
_COMPONENT_ID_RE = re.compile(r"[-_.]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_LICENSE_MODES = frozenset(
    {"distribution", "bundle-file", "python-runtime-file", "reviewed-exception"}
)


@dataclass(frozen=True, order=True)
class DistributionRef:
    name: str
    version: str
    dist_info_name: str


@dataclass(frozen=True)
class BundleFile:
    relative_path: str
    size: int
    sha256: str
    inventory_keys: tuple[str, ...]
    # These fields are in-process inspection capabilities only.  They are
    # deliberately excluded from equality/repr and explicitly never serialized.
    _filesystem_path: Path | None = field(default=None, repr=False, compare=False)
    _dist_root: Path | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ProvenanceSubject:
    kind: Literal["analysis-entry", "bundle-file"]
    key: str
    path: str


@dataclass(frozen=True)
class ProvenanceRecord:
    subject: ProvenanceSubject
    component_id: str
    component_name: str
    provenance_class: str
    distribution: DistributionRef | None
    license_mode: str
    license_source: str | None
    review_id: str | None
    review_reason: str | None


@dataclass(frozen=True)
class ClosureReport:
    records: tuple[ProvenanceRecord, ...]
    bundle_files: tuple[BundleFile, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ComponentPolicy:
    component_id: str
    name: str
    provenance_class: str
    priority: int
    globs: tuple[str, ...]
    license_mode: str
    license_source: str | None
    review_id: str | None
    review_reason: str | None


@dataclass(frozen=True)
class ProvenancePolicy:
    schema_version: int
    components: tuple[ComponentPolicy, ...]


@dataclass(frozen=True)
class _RecordClaim:
    source_path: str
    distribution: DistributionRef
    hash_spec: str | None
    size: int | None


def _component_id(value: str) -> str:
    return _COMPONENT_ID_RE.sub("-", value.strip()).casefold()


def _normalized_relative(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be a non-empty string")
    normalized_input = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(normalized_input)
    if windows.drive or posix.is_absolute() or normalized_input.startswith("//"):
        raise ValueError(f"{label} must be relative")
    if ".." in posix.parts:
        raise ValueError(f"{label} cannot contain '..'")
    if any(":" in part for part in posix.parts):
        raise ValueError(f"{label} cannot contain ':'")
    normalized = posix.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{label} must identify a file")
    return normalized


def _normalized_glob(value: str) -> str:
    normalized = _normalized_relative(value, label="component glob")
    if normalized != value:
        raise ValueError(f"component glob must use normalized '/' separators: {value!r}")
    return normalized


def _is_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _hash_file_stably(path: Path, *, algorithm: str = "sha256") -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    after = path.stat()
    identity_before = (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ino", None),
    )
    identity_after = (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ino", None),
    )
    if identity_before != identity_after or size != after.st_size:
        raise ValueError(f"file changed while hashing: {path.name}")
    return size, digest.hexdigest()


def _metadata_field(message: object, field_name: str, dist_info_name: str) -> str:
    values = getattr(message, "get_all")(field_name, [])
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise ValueError(
            f"{dist_info_name}/METADATA must contain exactly one {field_name} field"
        )
    return values[0].strip()


def _record_target(site_root: Path, recorded_path: str) -> Path:
    normalized = recorded_path.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(recorded_path)
    if windows.drive or posix.is_absolute() or normalized.startswith("//"):
        candidate = Path(recorded_path)
    else:
        candidate = site_root.joinpath(*posix.parts)
    # Normalize ``.`` and ``..`` lexically without following a payload
    # symlink/junction.  Reparse inspection must see the path recorded by the
    # distribution, not a resolved target that can silently change ownership.
    return Path(os.path.abspath(candidate))


def _reject_record_payload_reparse(
    site_root: Path,
    target: Path,
    source_path: str,
) -> None:
    """Reject any reparse point below the already-resolved site root."""

    relative = target.relative_to(site_root)
    current = site_root
    for part in relative.parts:
        current /= part
        if current.is_symlink() or _is_reparse(current):
            raise ValueError(f"reparse point in RECORD payload path {source_path}")


def _parse_record_hash(value: str, source_path: str) -> str | None:
    if not value:
        return None
    algorithm, separator, encoded = value.partition("=")
    if not separator or not algorithm or not encoded:
        raise ValueError(f"malformed RECORD hash for {source_path}")
    normalized_algorithm = algorithm.casefold()
    if (
        algorithm != normalized_algorithm
        or normalized_algorithm not in hashlib.algorithms_guaranteed
        or normalized_algorithm in {"md5", "sha1"}
    ):
        raise ValueError(f"unsupported or weak RECORD hash algorithm for {source_path}")
    try:
        digest_size = hashlib.new(normalized_algorithm).digest_size
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unsupported or weak RECORD hash algorithm for {source_path}") from exc
    if digest_size < 32:
        raise ValueError(f"unsupported or weak RECORD hash algorithm for {source_path}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
        raise ValueError(f"malformed RECORD hash for {source_path}")
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
    return f"{normalized_algorithm}={encoded}"


def _verify_record_claim(
    target: Path,
    source_path: str,
    hash_spec: str | None,
    size: int | None,
) -> bool:
    if not target.exists():
        return False
    if target.is_symlink() or _is_reparse(target) or not target.is_file():
        raise ValueError(f"actual bytes for RECORD path {source_path} are unavailable")
    algorithm = hash_spec.split("=", 1)[0] if hash_spec is not None else "sha256"
    actual_size, actual_hex = _hash_file_stably(target, algorithm=algorithm)
    size_matches = size is None or actual_size == size
    hash_matches = True
    if hash_spec is not None:
        _algorithm, encoded = hash_spec.split("=", 1)
        actual_encoded = base64.urlsafe_b64encode(bytes.fromhex(actual_hex)).decode(
            "ascii"
        ).rstrip("=")
        hash_matches = actual_encoded == encoded
    if not size_matches or not hash_matches:
        raise ValueError(
            f"actual bytes for RECORD path {source_path} do not match its claim"
        )
    return True


def _canonical_shared_owner(
    source_path: str, claims: Sequence[_RecordClaim]
) -> DistributionRef:
    relative = source_path.removeprefix("site-packages://")
    top_level = PurePosixPath(relative).parts[0]
    top_component = _component_id(top_level)
    distributions = sorted(
        {claim.distribution for claim in claims},
        key=lambda ref: (
            _component_id(ref.name),
            ref.name.casefold(),
            ref.version,
            ref.dist_info_name.casefold(),
        ),
    )
    path_named = [ref for ref in distributions if _component_id(ref.name) == top_component]
    if len(path_named) != 1:
        names = ", ".join(ref.name for ref in distributions)
        raise ValueError(
            f"ambiguous shared RECORD owners for {source_path}: "
            f"no unique path-named distribution among {names}"
        )
    return path_named[0]


def _distribution_from_metadata(
    message: object,
    dist_info_name: str,
) -> DistributionRef:
    name = _metadata_field(message, "Name", dist_info_name)
    version = _metadata_field(message, "Version", dist_info_name)
    try:
        metadata_version = Version(version)
    except InvalidVersion as exc:
        raise ValueError(
            f"{dist_info_name}/METADATA Version does not match its dist-info directory"
        ) from exc

    suffix = ".dist-info"
    stem = dist_info_name[: -len(suffix)]
    normalized_name = canonicalize_name(name)
    matches = []
    for index, character in enumerate(stem):
        if character != "-":
            continue
        directory_name = stem[:index]
        directory_version = stem[index + 1 :]
        if not directory_name or not directory_version:
            continue
        try:
            parsed_directory_version = Version(directory_version)
        except InvalidVersion:
            continue
        if (
            canonicalize_name(directory_name) == normalized_name
            and parsed_directory_version == metadata_version
        ):
            matches.append((directory_name, directory_version))
    if len(matches) != 1:
        raise ValueError(
            f"{dist_info_name}/METADATA Name/Version does not match its dist-info directory"
        )
    return DistributionRef(
        name=name,
        version=version,
        dist_info_name=dist_info_name,
    )


def build_record_ownership(site_packages: Path) -> Mapping[str, DistributionRef]:
    """Map canonical site-package sources to installed wheel distributions.

    Split distributions may share a verified RECORD row only when digest and
    size claims agree.  A unique path-named distribution owns that legitimate
    claim; otherwise ownership remains ambiguous and is rejected.
    """

    site_root = Path(site_packages)
    # Linked worktrees intentionally point this root at the shared build venv
    # with a Windows junction.  Resolve the caller-supplied root once and keep
    # every RECORD target contained beneath it; child metadata links remain
    # forbidden below.
    if not site_root.is_dir():
        raise ValueError("site-packages must be an existing directory")
    resolved_root = site_root.resolve(strict=True)
    claims_by_path: dict[str, list[_RecordClaim]] = {}

    candidates = sorted(
        (
            candidate
            for candidate in site_root.iterdir()
            if candidate.name.casefold().endswith(".dist-info")
        ),
        key=lambda candidate: candidate.name.casefold(),
    )
    for dist_info in candidates:
        if dist_info.is_symlink() or _is_reparse(dist_info) or not dist_info.is_dir():
            raise ValueError(f"distribution metadata is not a regular directory: {dist_info.name}")
        metadata_path = dist_info / "METADATA"
        record_path = dist_info / "RECORD"
        if any(
            path.is_symlink() or _is_reparse(path) or not path.is_file()
            for path in (metadata_path, record_path)
        ):
            raise ValueError(
                f"{dist_info.name} must contain regular files METADATA and RECORD"
            )
        try:
            message = BytesParser(policy=email_policy).parsebytes(metadata_path.read_bytes())
        except OSError as exc:
            raise ValueError(f"cannot read {dist_info.name}/METADATA") from exc
        distribution = _distribution_from_metadata(message, dist_info.name)
        seen_in_record: set[str] = set()
        try:
            with record_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"cannot parse {dist_info.name}/RECORD as UTF-8 CSV") from exc
        for row_number, row in enumerate(rows, start=1):
            if len(row) != 3 or not row[0] or "\0" in row[0]:
                raise ValueError(f"malformed {dist_info.name}/RECORD row {row_number}")
            target = _record_target(resolved_root, row[0])
            try:
                relative = target.relative_to(resolved_root)
            except ValueError:
                # RECORD legitimately includes Scripts/ entries outside purelib;
                # the analysis inventory represents those under another scheme.
                continue
            if relative in {Path(""), Path(".")}:
                raise ValueError(f"malformed {dist_info.name}/RECORD row {row_number}")
            normalized = relative.as_posix()
            source_path = f"site-packages://{normalized}"
            _reject_record_payload_reparse(resolved_root, target, source_path)
            folded = source_path.casefold()
            if folded in seen_in_record:
                raise ValueError(f"duplicate RECORD path in {dist_info.name}: {source_path}")
            seen_in_record.add(folded)
            hash_spec = _parse_record_hash(row[1], source_path)
            if row[2]:
                if not re.fullmatch(r"0|[1-9][0-9]*", row[2]):
                    raise ValueError(f"malformed RECORD size for {source_path}")
                claimed_size = int(row[2])
            else:
                claimed_size = None
            if (hash_spec is None) != (claimed_size is None):
                raise ValueError(
                    f"RECORD path {source_path} must provide a complete hash and size claim"
                )
            if not _verify_record_claim(target, source_path, hash_spec, claimed_size):
                continue
            claims_by_path.setdefault(folded, []).append(
                _RecordClaim(source_path, distribution, hash_spec, claimed_size)
            )

    ownership: dict[str, DistributionRef] = {}
    for folded in sorted(claims_by_path):
        claims = claims_by_path[folded]
        source_path = min(claim.source_path for claim in claims)
        distributions = {claim.distribution for claim in claims}
        if len(distributions) == 1:
            owner = next(iter(distributions))
        else:
            signatures = {(claim.hash_spec, claim.size) for claim in claims}
            if len(signatures) != 1:
                names = ", ".join(sorted(ref.name for ref in distributions))
                raise ValueError(
                    f"conflicting RECORD owners for {source_path}: {names} have different digest/size claims"
                )
            hash_spec, size = next(iter(signatures))
            # RECORD explicitly permits both fields to be blank, most often
            # for generated pyc files.  Multiple blank claims provide no
            # evidence for selecting a byte owner, so omit the key.  If such a
            # path ever enters Analysis it becomes an explicit unowned error.
            if hash_spec is None and size is None:
                continue
            if hash_spec is None or size is None:
                raise ValueError(
                    f"shared RECORD path {source_path} lacks complete digest/size claims"
                )
            owner = _canonical_shared_owner(source_path, claims)
        ownership[source_path] = owner
    return dict(sorted(ownership.items()))


def load_provenance_policy(path: Path) -> ProvenancePolicy:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("provenance policy is not valid UTF-8 TOML") from exc
    if not isinstance(payload, dict) or set(payload) - {"schema_version", "component"}:
        raise ValueError("provenance policy has unknown top-level fields")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported provenance policy schema_version")
    rows = payload.get("component", [])
    if not isinstance(rows, list):
        raise ValueError("provenance policy component must be an array of tables")

    required = {"id", "name", "class", "priority", "globs", "license_mode"}
    optional = {"license_source", "review_id", "review_reason"}
    components: list[ComponentPolicy] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= set(row) or set(row) - required - optional:
            raise ValueError(f"provenance policy component {index} has malformed fields")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("id", "name", "class", "license_mode")):
            raise ValueError(f"provenance policy component {index} has empty string fields")
        component_id = _component_id(row["id"])
        if component_id != row["id"] or component_id in seen_ids:
            raise ValueError(f"provenance policy component id is duplicate or non-canonical: {row['id']!r}")
        seen_ids.add(component_id)
        if type(row["priority"]) is not int:
            raise ValueError(f"provenance policy component {component_id} priority must be an integer")
        raw_globs = row["globs"]
        if not isinstance(raw_globs, list) or not raw_globs or any(not isinstance(item, str) for item in raw_globs):
            raise ValueError(f"provenance policy component {component_id} globs must be a non-empty string list")
        globs = tuple(_normalized_glob(item) for item in raw_globs)
        if len({glob.casefold() for glob in globs}) != len(globs):
            raise ValueError(f"provenance policy component {component_id} has duplicate globs")
        mode = row["license_mode"]
        if mode not in _ALLOWED_LICENSE_MODES:
            raise ValueError(f"provenance policy component {component_id} has unknown license mode")
        source = row.get("license_source")
        review_id = row.get("review_id")
        review_reason = row.get("review_reason")
        if mode == "reviewed-exception":
            if source is not None:
                raise ValueError(f"reviewed exception {component_id} cannot have a license source")
            if not isinstance(review_id, str) or not review_id.strip():
                raise ValueError(f"reviewed exception {component_id} requires a non-empty review id")
            if not isinstance(review_reason, str) or not review_reason.strip():
                raise ValueError(f"reviewed exception {component_id} requires a non-empty review reason")
            review_id = review_id.strip()
            review_reason = review_reason.strip()
        else:
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"provenance policy component {component_id} requires a license source")
            if review_id is not None or review_reason is not None:
                raise ValueError(f"provenance policy component {component_id} cannot have review fields")
            source = source.strip()
            if mode == "distribution":
                source = _component_id(source)
            elif mode == "bundle-file":
                source = _normalized_relative(source, label="bundle license source")
            else:
                prefix = "python://"
                if not source.startswith(prefix):
                    raise ValueError(f"python runtime license source for {component_id} must use python://")
                relative = _normalized_relative(source[len(prefix):], label="python license source")
                source = f"{prefix}{relative}"
        components.append(
            ComponentPolicy(
                component_id=component_id,
                name=row["name"].strip(),
                provenance_class=row["class"].strip(),
                priority=row["priority"],
                globs=globs,
                license_mode=mode,
                license_source=source,
                review_id=review_id,
                review_reason=review_reason,
            )
        )
    return ProvenancePolicy(1, tuple(components))


def enumerate_bundle_files(
    dist_root: Path, entries: Iterable[InventoryEntry]
) -> tuple[BundleFile, ...]:
    """Hash final product files and link only exact on-disk destinations."""

    root = Path(dist_root)
    if not root.is_dir() or root.is_symlink() or _is_reparse(root):
        raise ValueError("dist root must be an existing regular directory")
    resolved_root = root.resolve(strict=True)
    destination_keys: dict[str, list[str]] = {}
    for entry in entries:
        if entry.collection in _LINKED_COLLECTIONS:
            destination_keys.setdefault(entry.destination, []).append(entry.key)

    files: list[BundleFile] = []
    seen_casefolded: set[str] = set()
    candidates = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().casefold())
    for candidate in candidates:
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink() or _is_reparse(candidate):
            raise ValueError(f"bundle path is a symlink or reparse point: {relative}")
        if relative.casefold() == _LICENSE_ROOT or relative.casefold().startswith(_LICENSE_ROOT + "/"):
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"bundle path is not a regular file: {relative}")
        normalized = _normalized_relative(relative, label="bundle relative path")
        folded = normalized.casefold()
        if folded in seen_casefolded:
            raise ValueError(f"bundle has duplicate case-insensitive path: {normalized}")
        seen_casefolded.add(folded)
        resolved = candidate.resolve(strict=True)
        try:
            contained = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"bundle file resolves outside dist root: {normalized}") from exc
        if contained.casefold() != normalized.casefold():
            raise ValueError(f"bundle file resolves to a different path: {normalized}")
        size, digest = _hash_file_stably(resolved)
        files.append(
            BundleFile(
                relative_path=normalized,
                size=size,
                sha256=digest,
                inventory_keys=tuple(sorted(destination_keys.get(normalized, ()))),
                _filesystem_path=resolved,
                _dist_root=resolved_root,
            )
        )
    return tuple(sorted(files, key=lambda item: (item.relative_path.casefold(), item.relative_path)))


@lru_cache(maxsize=None)
def _compiled_policy_glob(glob: str) -> re.Pattern[str]:
    translated: list[str] = [r"\A"]
    index = 0
    folded = glob.casefold()
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


def _policy_matches(component: ComponentPolicy, path: str) -> bool:
    folded = path.casefold()
    return any(_compiled_policy_glob(glob).fullmatch(folded) for glob in component.globs)


def _record_from_distribution(
    subject: ProvenanceSubject, distribution: DistributionRef
) -> ProvenanceRecord:
    component_id = _component_id(distribution.name)
    return ProvenanceRecord(
        subject=subject,
        component_id=component_id,
        component_name=distribution.name,
        provenance_class="distribution",
        distribution=distribution,
        license_mode="distribution",
        license_source=component_id,
        review_id=None,
        review_reason=None,
    )


def _project_record(subject: ProvenanceSubject) -> ProvenanceRecord:
    return ProvenanceRecord(
        subject=subject,
        component_id="yamabiko",
        component_name="Yamabiko",
        provenance_class="project",
        distribution=None,
        license_mode="project",
        license_source=None,
        review_id=None,
        review_reason=None,
    )


def _python_record(subject: ProvenanceSubject) -> ProvenanceRecord:
    return ProvenanceRecord(
        subject=subject,
        component_id="python-runtime",
        component_name="Python",
        provenance_class="python-runtime",
        distribution=None,
        license_mode="python-runtime-file",
        license_source="python://LICENSE.txt",
        review_id=None,
        review_reason=None,
    )


def _policy_record(
    subject: ProvenanceSubject,
    component: ComponentPolicy,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        subject=subject,
        component_id=component.component_id,
        component_name=component.name,
        provenance_class=component.provenance_class,
        distribution=None,
        license_mode=component.license_mode,
        license_source=component.license_source,
        review_id=component.review_id,
        review_reason=component.review_reason,
    )


def _clone_owner(subject: ProvenanceSubject, owner: ProvenanceRecord) -> ProvenanceRecord:
    return ProvenanceRecord(
        subject=subject,
        component_id=owner.component_id,
        component_name=owner.component_name,
        provenance_class=owner.provenance_class,
        distribution=owner.distribution,
        license_mode=owner.license_mode,
        license_source=owner.license_source,
        review_id=owner.review_id,
        review_reason=owner.review_reason,
    )


def _owner_signature(record: ProvenanceRecord) -> tuple[object, ...]:
    return (
        record.component_id,
        record.component_name,
        record.provenance_class,
        record.distribution,
        record.license_mode,
        record.license_source,
        record.review_id,
        record.review_reason,
    )


def _descriptor_signature(record: ProvenanceRecord) -> tuple[object, ...]:
    return (
        record.component_name,
        record.provenance_class,
        record.distribution,
        record.license_mode,
        record.license_source,
        record.review_id,
        record.review_reason,
    )


def _distribution_index(
    ownership: Mapping[str, DistributionRef]
) -> dict[str, DistributionRef]:
    grouped: dict[str, set[DistributionRef]] = {}
    for distribution in ownership.values():
        grouped.setdefault(_component_id(distribution.name), set()).add(distribution)
    result: dict[str, DistributionRef] = {}
    for component_id, candidates in sorted(grouped.items()):
        if len(candidates) != 1:
            versions = ", ".join(sorted(f"{item.name} {item.version}" for item in candidates))
            raise ValueError(f"multiple installed distributions normalize to {component_id}: {versions}")
        result[component_id] = next(iter(candidates))
    return result


def _native_domain(relative_path: str) -> str:
    return "llama" if relative_path.casefold().startswith("_internal/llama/") else "main"


def _bundle_directory(relative_path: str) -> str:
    parent = PurePosixPath(relative_path).parent.as_posix()
    return "" if parent == "." else parent.casefold()


def _native_search_directories(importer: BundleFile) -> set[str]:
    if _native_domain(importer.relative_path) == "llama":
        return {"_internal/llama"}
    directories = {"", "_internal", _bundle_directory(importer.relative_path)}
    parts = PurePosixPath(importer.relative_path).parts
    if len(parts) >= 2 and parts[0].casefold() == "_internal":
        package = parts[1].casefold()
        directories.update(_PACKAGE_DLL_SEARCH_DIRECTORIES.get(package, ()))
    return directories


def _owned_import_target(
    import_name: str,
    importer: BundleFile,
    bundle_files: Sequence[BundleFile],
    owned_paths: set[str],
) -> bool:
    basename = PureWindowsPath(import_name).name.casefold()
    search_directories = _native_search_directories(importer)
    dlls = [
        bundle_file
        for bundle_file in bundle_files
        if bundle_file.relative_path.casefold().endswith(".dll")
        and _bundle_directory(bundle_file.relative_path) in search_directories
    ]
    direct = [item for item in dlls if PurePosixPath(item.relative_path).name.casefold() == basename]
    candidates = direct
    if not candidates and basename.endswith(".dll"):
        stem = re.escape(basename[:-4])
        hashed = re.compile(rf"^{stem}-[0-9a-f]{{32}}\.dll\Z", re.IGNORECASE)
        candidates = [
            item
            for item in dlls
            if hashed.fullmatch(PurePosixPath(item.relative_path).name)
        ]
    return any(item.relative_path.casefold() in owned_paths for item in candidates)


def resolve_provenance(
    entries: Iterable[InventoryEntry],
    bundle_files: Iterable[BundleFile],
    *,
    site_packages: Path,
    policy_config_path: Path,
) -> ClosureReport:
    """Resolve every analysis and final-tree subject or return closure errors."""

    canonical_entries = tuple(sorted(entries))
    files = tuple(sorted(bundle_files, key=lambda item: (item.relative_path.casefold(), item.relative_path)))
    entry_by_key: dict[str, InventoryEntry] = {}
    for entry in canonical_entries:
        if entry.key in entry_by_key:
            raise ValueError(f"duplicate analysis inventory key: {entry.key}")
        entry_by_key[entry.key] = entry
    file_paths: set[str] = set()
    for bundle_file in files:
        _normalized_relative(bundle_file.relative_path, label="bundle relative path")
        if type(bundle_file.size) is not int or bundle_file.size < 0:
            raise ValueError(f"bundle file has invalid size: {bundle_file.relative_path}")
        if not _SHA256_RE.fullmatch(bundle_file.sha256):
            raise ValueError(f"bundle file has invalid sha256: {bundle_file.relative_path}")
        folded = bundle_file.relative_path.casefold()
        if folded in file_paths:
            raise ValueError(f"duplicate bundle file path: {bundle_file.relative_path}")
        file_paths.add(folded)
        if tuple(sorted(set(bundle_file.inventory_keys))) != bundle_file.inventory_keys:
            raise ValueError(f"bundle file has non-canonical inventory keys: {bundle_file.relative_path}")
        unknown = [key for key in bundle_file.inventory_keys if key not in entry_by_key]
        if unknown:
            raise ValueError(f"bundle file links unknown inventory key: {bundle_file.relative_path}")

    ownership = build_record_ownership(site_packages)
    ownership_folded = {key.casefold(): value for key, value in ownership.items()}
    distributions = _distribution_index(ownership)
    policy = load_provenance_policy(policy_config_path)
    raw_matches = {component.component_id: 0 for component in policy.components}
    effective_matches = {component.component_id: 0 for component in policy.components}
    errors: set[str] = set()
    for component in policy.components:
        if component.component_id in distributions:
            errors.add(
                f"policy component id {component.component_id} collides with automatic distribution id"
            )
    linked_inventory_keys = {
        key for bundle_file in files for key in bundle_file.inventory_keys
    }
    for entry in canonical_entries:
        if entry.collection in _LINKED_COLLECTIONS and entry.key not in linked_inventory_keys:
            errors.add(
                f"{entry.collection} inventory destination {entry.destination} has no final bundle file"
            )
    records: list[ProvenanceRecord] = []
    analysis_owners: dict[str, ProvenanceRecord] = {}

    def select_policy(kind: str, path: str) -> tuple[ComponentPolicy | None, bool]:
        matches = [component for component in policy.components if _policy_matches(component, path)]
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
        winner = winners[0]
        return winner, True

    for entry in canonical_entries:
        subject = ProvenanceSubject("analysis-entry", entry.key, entry.destination)
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
            record = _record_from_distribution(subject, distribution)
        elif component is not None:
            effective_matches[component.component_id] += 1
            record = _policy_record(subject, component)
        elif had_policy_match:
            continue
        elif entry.source_path.startswith("repo://"):
            record = _project_record(subject)
        elif entry.source_path.startswith("python://"):
            record = _python_record(subject)
        elif entry.source_path.startswith("site-packages://"):
            if distribution is None:
                errors.add(
                    f"unowned analysis-entry {entry.key} at {entry.destination} (source {entry.source_path})"
                )
                continue
            record = _record_from_distribution(subject, distribution)
        else:
            errors.add(
                f"unowned analysis-entry {entry.key} at {entry.destination} (source {entry.source_path})"
            )
            continue
        records.append(record)
        analysis_owners[entry.key] = record

    for bundle_file in files:
        subject = ProvenanceSubject(
            "bundle-file", bundle_file.relative_path, bundle_file.relative_path
        )
        linked = [
            analysis_owners[key]
            for key in bundle_file.inventory_keys
            if key in analysis_owners
        ]
        component, had_policy_match = select_policy("bundle-file", bundle_file.relative_path)
        record_owned = any(owner.distribution is not None for owner in linked)
        if record_owned and had_policy_match:
            errors.add(
                f"policy cannot override RECORD-owned bundle-file {bundle_file.relative_path}"
            )
            component = None
            had_policy_match = False
        if component is not None:
            effective_matches[component.component_id] += 1
            record = _policy_record(subject, component)
        elif had_policy_match:
            continue
        else:
            if len(linked) != len(bundle_file.inventory_keys) or not linked:
                errors.add(f"unowned bundle-file {bundle_file.relative_path}")
                continue
            signatures = {_owner_signature(owner) for owner in linked}
            if len(signatures) != 1:
                components = ", ".join(sorted(owner.component_id for owner in linked))
                errors.add(
                    f"bundle-file {bundle_file.relative_path} links conflicting owners: {components}"
                )
                continue
            record = _clone_owner(subject, linked[0])
        records.append(record)

    for component in policy.components:
        raw = raw_matches[component.component_id]
        effective = effective_matches[component.component_id]
        if raw == 0:
            errors.add(f"stale policy component '{component.component_id}': no raw match")
        elif effective == 0:
            errors.add(
                f"stale policy component '{component.component_id}': raw matches exist but it never owns a subject"
            )
        if component.license_mode == "distribution" and component.license_source not in distributions:
            errors.add(
                f"policy component {component.component_id} references unavailable distribution {component.license_source}"
            )
        if (
            component.license_mode == "bundle-file"
            and (component.license_source or "").casefold() not in file_paths
        ):
            errors.add(
                f"policy component {component.component_id} references missing bundle license {component.license_source}"
            )

    descriptors: dict[str, set[tuple[object, ...]]] = {}
    for record in records:
        component_id = _component_id(record.component_id)
        descriptors.setdefault(component_id, set()).add(_descriptor_signature(record))
    for component_id, signatures in sorted(descriptors.items()):
        if len(signatures) != 1:
            errors.add(
                f"component id {component_id} has inconsistent descriptor signatures"
            )

    owned_paths = {
        record.subject.path.casefold()
        for record in records
        if record.subject.kind == "bundle-file"
    }
    for bundle_file in files:
        if PurePosixPath(bundle_file.relative_path).suffix.casefold() not in _NATIVE_SUFFIXES:
            continue
        filesystem_path = bundle_file._filesystem_path
        dist_root = bundle_file._dist_root
        if filesystem_path is None or dist_root is None:
            errors.add(f"native bundle-file {bundle_file.relative_path} has no inspection path")
            continue
        try:
            resolved_root = dist_root.resolve(strict=True)
            resolved_file = filesystem_path.resolve(strict=True)
            contained = resolved_file.relative_to(resolved_root).as_posix()
        except (OSError, ValueError):
            errors.add(
                f"native bundle-file {bundle_file.relative_path} inspection path is outside enumerated dist root"
            )
            continue
        if contained.casefold() != bundle_file.relative_path.casefold():
            errors.add(
                f"native bundle-file {bundle_file.relative_path} inspection path is outside enumerated dist root"
            )
            continue
        try:
            actual_size, actual_hash = _hash_file_stably(resolved_file)
        except (OSError, ValueError):
            errors.add(f"native bundle-file {bundle_file.relative_path} cannot be re-verified")
            continue
        if actual_size != bundle_file.size or actual_hash != bundle_file.sha256:
            errors.add(f"native bundle-file {bundle_file.relative_path} changed after enumeration")
            continue
        if is_allowed_ci_llama_stub(bundle_file.relative_path, resolved_file):
            # A deliberate CI text stub has no imports to close over.
            continue
        try:
            imports = parse_imports(resolved_file)
        except Exception:
            errors.add(f"native bundle-file {bundle_file.relative_path} PE imports could not be parsed")
            continue
        by_name = {name.casefold(): name for name in imports if isinstance(name, str) and name}
        for folded_name in sorted(by_name):
            import_name = by_name[folded_name]
            if is_system_dll(import_name):
                continue
            if not _owned_import_target(import_name, bundle_file, files, owned_paths):
                errors.add(
                    f"PE import {import_name} from {bundle_file.relative_path} has no present owner in {_native_domain(bundle_file.relative_path)} domain"
                )

    canonical_records = tuple(
        sorted(
            records,
            key=lambda record: (
                record.subject.kind,
                record.subject.path.casefold(),
                record.subject.path,
                record.subject.key,
                record.component_id,
            ),
        )
    )
    return ClosureReport(canonical_records, files, tuple(sorted(errors)))


def _record_payload(record: ProvenanceRecord) -> dict[str, object]:
    distribution = None
    if record.distribution is not None:
        distribution = {
            "name": record.distribution.name,
            "version": record.distribution.version,
            "dist_info_name": record.distribution.dist_info_name,
        }
    return {
        "subject": {
            "kind": record.subject.kind,
            "key": record.subject.key,
            "path": record.subject.path,
        },
        "component_id": record.component_id,
        "component_name": record.component_name,
        "provenance_class": record.provenance_class,
        "distribution": distribution,
        "license_mode": record.license_mode,
        "license_source": record.license_source,
        "review_id": record.review_id,
        "review_reason": record.review_reason,
    }


def _report_payload(report: ClosureReport) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "records": [_record_payload(record) for record in report.records],
        "bundle_files": [
            {
                "relative_path": bundle_file.relative_path,
                "size": bundle_file.size,
                "sha256": bundle_file.sha256,
                "inventory_keys": list(bundle_file.inventory_keys),
            }
            for bundle_file in report.bundle_files
        ],
        "errors": list(report.errors),
    }


def _invalidate_output(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        raise ValueError("provenance output path is a directory")
    if path.exists() or path.is_symlink():
        path.unlink()


def write_provenance_report(report: ClosureReport, output: Path) -> None:
    """Atomically publish canonical JSON, but only for a green closure."""

    output = Path(output)
    if report.errors:
        _invalidate_output(output)
        raise ValueError("cannot write provenance report with errors")
    serialized = (json.dumps(_report_payload(report), sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser(
        "resolve",
        help="resolve bundle ownership closure",
        allow_abbrev=False,
    )
    resolve.add_argument("--inventory", type=Path, required=True)
    resolve.add_argument("--dist", type=Path, required=True)
    resolve.add_argument("--policy-config", type=Path, required=True)
    resolve.add_argument("--site-packages", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    return parser


def _preparse_output_paths(argv: Sequence[str]) -> tuple[Path, ...]:
    """Extract valid resolve-output options before argparse can exit."""

    outputs: list[Path] = []
    index = 0
    if index < len(argv) and str(argv[index]) == "--":
        # The root parser's terminator does not terminate the subparser that
        # begins with the following command token.
        index += 1
    if index >= len(argv) or str(argv[index]) != "resolve":
        return ()
    index += 1
    while index < len(argv):
        value = str(argv[index])
        if value == "--":
            # This terminator belongs to the resolve subparser, so subsequent
            # tokens are not output options.
            break
        if value == "--output":
            if index + 1 < len(argv):
                candidate = str(argv[index + 1])
                if not candidate.startswith("-"):
                    outputs.append(Path(candidate))
                    index += 2
                    continue
            index += 1
            continue
        if value.startswith("--output=") and value != "--output=":
            outputs.append(Path(value.split("=", 1)[1]))
        index += 1
    return tuple(dict.fromkeys(outputs))


def _redacted_error(exc: Exception, paths: Mapping[str, Path]) -> str:
    message = str(exc) or type(exc).__name__
    replacements: list[tuple[str, str]] = []
    for label, path in paths.items():
        replacements.append((str(path), f"<{label}>"))
        try:
            replacements.append((str(path.resolve(strict=False)), f"<{label}>"))
        except OSError:
            pass
    for raw, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if raw:
            message = re.sub(re.escape(raw), replacement, message, flags=re.IGNORECASE)
    return message


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    for output in _preparse_output_paths(raw_argv):
        try:
            _invalidate_output(output)
        except Exception as exc:
            print(
                f"bundle provenance error: "
                f"{_redacted_error(exc, {'output': output})}",
                file=sys.stderr,
            )
            return 2
    args = _build_parser().parse_args(raw_argv)
    paths = {
        "inventory": args.inventory,
        "dist": args.dist,
        "policy-config": args.policy_config,
        "site-packages": args.site_packages,
        "output": args.output,
    }
    try:
        _invalidate_output(args.output)
    except Exception as exc:
        print(f"bundle provenance error: {_redacted_error(exc, paths)}", file=sys.stderr)
        return 2
    try:
        entries = load_inventory(args.inventory)
        bundle_files = enumerate_bundle_files(args.dist, entries)
        report = resolve_provenance(
            entries,
            bundle_files,
            site_packages=args.site_packages,
            policy_config_path=args.policy_config,
        )
    except Exception as exc:
        print(f"bundle provenance error: {_redacted_error(exc, paths)}", file=sys.stderr)
        return 2
    if report.errors:
        for error in report.errors:
            print(error, file=sys.stderr)
        return 1
    try:
        write_provenance_report(report, args.output)
    except Exception as exc:
        print(f"bundle provenance error: {_redacted_error(exc, paths)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
