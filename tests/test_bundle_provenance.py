"""Ownership closure for PyInstaller analysis entries and final bundle files.

The resolver is a release boundary: every analysis subject and every shipped
file must have one deterministic owner, and native imports must close over the
owned final tree.  These tests deliberately use complete wheel metadata rather
than import-name guesses so a broken RECORD join is observable.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.bundle_inventory import InventoryEntry, SCHEMA_VERSION as INVENTORY_SCHEMA
from scripts import bundle_provenance as provenance


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "bundle_provenance.py"


def _entry(
    logical_name: str,
    *,
    source_path: str,
    collection: str = "binaries",
    destination: str | None = None,
) -> InventoryEntry:
    if destination is None:
        if collection == "pure":
            destination = f"PYZ:{logical_name}"
        elif collection == "scripts":
            destination = f"EXE:{logical_name}"
        else:
            destination = f"_internal/{logical_name}"
    type_code = {
        "binaries": "BINARY",
        "datas": "DATA",
        "pure": "PYMODULE",
        "scripts": "PYSOURCE",
        "zipfiles": "ZIPFILE",
    }[collection]
    return InventoryEntry(
        collection=collection,
        logical_name=logical_name,
        source_path=source_path,
        destination=destination,
        type_code=type_code,
    )


def _write_distribution(
    site_packages: Path,
    name: str,
    version: str = "1.0",
    files: dict[str, bytes] | None = None,
    *,
    unhashed: frozenset[str] = frozenset(),
) -> Path:
    """Write an installed distribution with valid METADATA and CSV RECORD."""

    site_packages.mkdir(parents=True, exist_ok=True)
    files = files or {}
    for relative, content in files.items():
        target = site_packages / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    metadata = dist_info / "METADATA"
    record = dist_info / "RECORD"
    metadata.write_text(
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n",
        encoding="utf-8",
    )
    rows = []
    for relative in sorted(files):
        content = files[relative]
        if relative in unhashed:
            rows.append((relative.replace("\\", "/"), "", ""))
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode(
            "ascii"
        ).rstrip("=")
        rows.append(
            (relative.replace("\\", "/"), f"sha256={digest}", str(len(content)))
        )
    rows.extend(
        (
            (f"{dist_info.name}/METADATA", "", ""),
            (f"{dist_info.name}/RECORD", "", ""),
        )
    )
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return dist_info


def _record_hash(algorithm: str, content: bytes) -> str:
    digest = hashlib.new(algorithm, content).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{algorithm}={encoded}"


def _rewrite_record_claim(
    dist_info: Path,
    relative_path: str,
    hash_value: str,
    size_value: str,
) -> None:
    record = dist_info / "RECORD"
    with record.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    matching = [row for row in rows if row[0] == relative_path]
    assert len(matching) == 1
    matching[0][1:] = [hash_value, size_value]
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _component(
    component_id: str,
    *,
    name: str | None = None,
    provenance_class: str = "reviewed",
    priority: int,
    globs: tuple[str, ...],
    license_mode: str = "reviewed-exception",
    license_source: str | None = None,
    review_id: str | None = "TEST-REVIEW",
    review_reason: str | None = "Synthetic fixture reviewed for ownership tests.",
) -> str:
    lines = [
        "[[component]]",
        f"id = {json.dumps(component_id)}",
        f"name = {json.dumps(name or component_id)}",
        f"class = {json.dumps(provenance_class)}",
        f"priority = {priority}",
        f"globs = {json.dumps(list(globs))}",
        f"license_mode = {json.dumps(license_mode)}",
    ]
    if license_source is not None:
        lines.append(f"license_source = {json.dumps(license_source)}")
    if review_id is not None:
        lines.append(f"review_id = {json.dumps(review_id)}")
    if review_reason is not None:
        lines.append(f"review_reason = {json.dumps(review_reason)}")
    return "\n".join(lines)


def _write_policy(path: Path, *components: str) -> Path:
    path.write_text(
        "schema_version = 1\n\n" + "\n\n".join(components) + "\n",
        encoding="utf-8",
    )
    return path


def _write_inventory(path: Path, entries: tuple[InventoryEntry, ...]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": INVENTORY_SCHEMA,
                "entries": [asdict(entry) for entry in entries],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _bundle(
    tmp_path: Path,
    files: dict[str, bytes],
    entries: tuple[InventoryEntry, ...] = (),
) -> tuple[Path, tuple[provenance.BundleFile, ...]]:
    dist_root = tmp_path / "dist" / "yamabiko"
    for relative, content in files.items():
        target = dist_root / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return dist_root, provenance.enumerate_bundle_files(dist_root, entries)


def _record(
    report: provenance.ClosureReport, kind: str, path: str
) -> provenance.ProvenanceRecord:
    return next(
        record
        for record in report.records
        if record.subject.kind == kind and record.subject.path == path
    )


def _resolve(
    tmp_path: Path,
    entries: tuple[InventoryEntry, ...],
    bundle_files: tuple[provenance.BundleFile, ...],
    *components: str,
    site_packages: Path | None = None,
) -> provenance.ClosureReport:
    site_packages = site_packages or (tmp_path / "site-packages")
    site_packages.mkdir(parents=True, exist_ok=True)
    policy = _write_policy(tmp_path / "policy.toml", *components)
    return provenance.resolve_provenance(
        entries,
        bundle_files,
        site_packages=site_packages,
        policy_config_path=policy,
    )


def test_record_ownership_maps_canonical_analysis_source_to_distribution(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _write_distribution(
        site_packages,
        "Alpha-Pkg",
        "2.3",
        {"alpha/module.py": b"VALUE = 1\n"},
    )

    ownership = provenance.build_record_ownership(site_packages)

    assert ownership["site-packages://alpha/module.py"] == provenance.DistributionRef(
        name="Alpha-Pkg",
        version="2.3",
        dist_info_name=dist_info.name,
    )


@pytest.mark.parametrize(
    "hash_value",
    (
        "sha256=!!!!",
        "sha256=AAAA",
        _record_hash("md5", b"payload"),
        _record_hash("sha224", b"payload"),
        "not-a-hash=AAAA",
    ),
)
def test_record_ownership_rejects_malformed_weak_or_unsupported_hash(
    tmp_path: Path,
    hash_value: str,
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/data.bin": b"payload"},
    )
    _rewrite_record_claim(dist_info, "alpha/data.bin", hash_value, "7")

    with pytest.raises(ValueError, match="RECORD hash"):
        provenance.build_record_ownership(site_packages)


def test_record_ownership_rejects_noncanonical_size(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    content = b"payload"
    dist_info = _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/data.bin": content},
    )
    _rewrite_record_claim(
        dist_info,
        "alpha/data.bin",
        _record_hash("sha256", content),
        "07",
    )

    with pytest.raises(ValueError, match="RECORD size"):
        provenance.build_record_ownership(site_packages)


@pytest.mark.parametrize(
    ("hash_value", "size_value"),
    (
        (_record_hash("sha256", b"payload"), ""),
        ("", "7"),
    ),
)
def test_record_ownership_rejects_partial_hash_or_size_claim(
    tmp_path: Path,
    hash_value: str,
    size_value: str,
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/data.bin": b"payload"},
    )
    _rewrite_record_claim(dist_info, "alpha/data.bin", hash_value, size_value)

    with pytest.raises(ValueError, match="complete.*hash.*size"):
        provenance.build_record_ownership(site_packages)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("Name", "beta"), ("Version", "2.0")),
)
def test_record_ownership_rejects_metadata_identity_mismatch(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _write_distribution(
        site_packages,
        "alpha",
        "1.0",
        {"alpha/data.bin": b"payload"},
    )
    metadata = dist_info / "METADATA"
    body = metadata.read_text(encoding="utf-8")
    old_value = "alpha" if field == "Name" else "1.0"
    body = body.replace(f"{field}: {old_value}", f"{field}: {replacement}")
    metadata.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=r"METADATA (?:Name|Version).*dist-info"):
        provenance.build_record_ownership(site_packages)


@pytest.mark.parametrize("mismatch", ("hash", "size"))
def test_record_ownership_rejects_unique_owner_claim_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    site_packages = tmp_path / "site-packages"
    content = b"payload"
    dist_info = _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/data.bin": content},
    )
    hash_value = _record_hash(
        "sha256",
        b"different" if mismatch == "hash" else content,
    )
    size_value = str(len(content) + 1 if mismatch == "size" else len(content))
    _rewrite_record_claim(
        dist_info,
        "alpha/data.bin",
        hash_value,
        size_value,
    )

    with pytest.raises(ValueError, match=r"actual bytes.*do not match its claim"):
        provenance.build_record_ownership(site_packages)


def test_record_ownership_omits_missing_record_target(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    target = site_packages / "alpha" / "missing.py"
    _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/missing.py": b"VALUE = 1\n"},
    )
    target.unlink()

    ownership = provenance.build_record_ownership(site_packages)

    assert "site-packages://alpha/missing.py" not in ownership


def test_record_ownership_maps_blank_single_owner_claim(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/generated.pyc": b"generated"},
        unhashed=frozenset({"alpha/generated.pyc"}),
    )

    ownership = provenance.build_record_ownership(site_packages)

    assert ownership["site-packages://alpha/generated.pyc"].name == "alpha"


def test_record_ownership_accepts_resolved_site_packages_junction_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linked worktrees junction site-packages to the shared build venv.

    Seeded defect: reject the caller-supplied root merely because it is a
    reparse point; the required real CLI then cannot inspect its build venv.
    Child metadata reparse points remain forbidden.
    """

    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/module.py": b"VALUE = 1\n"},
    )
    real_is_reparse = provenance._is_reparse
    monkeypatch.setattr(
        provenance,
        "_is_reparse",
        lambda path: Path(path) == site_packages or real_is_reparse(path),
    )

    ownership = provenance.build_record_ownership(site_packages)

    assert ownership["site-packages://alpha/module.py"].name == "alpha"


def test_record_ownership_rejects_reparse_metadata_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/module.py": b"VALUE = 1\n"},
    )
    metadata_path = dist_info / "METADATA"
    real_is_reparse = provenance._is_reparse
    monkeypatch.setattr(
        provenance,
        "_is_reparse",
        lambda path: Path(path) == metadata_path or real_is_reparse(path),
    )

    with pytest.raises(ValueError, match="regular files.*METADATA.*RECORD"):
        provenance.build_record_ownership(site_packages)


def test_record_ownership_rejects_reparse_payload_file_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "alpha",
        files={
            "alpha/link.bin": b"payload",
            "alpha/target.bin": b"payload",
        },
    )
    lexical_link = site_packages / "alpha" / "link.bin"
    resolved_target = site_packages / "alpha" / "target.bin"
    real_resolve = Path.resolve

    def resolve_reparse(path: Path, strict: bool = False) -> Path:
        if Path(path) == lexical_link:
            return resolved_target
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_reparse)
    monkeypatch.setattr(
        provenance,
        "_is_reparse",
        lambda path: Path(path) == lexical_link,
    )

    with pytest.raises(ValueError, match="reparse.*alpha/link.bin"):
        provenance.build_record_ownership(site_packages)


def test_record_ownership_rejects_reparse_payload_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/data.bin": b"payload"},
    )
    payload_parent = site_packages / "alpha"
    monkeypatch.setattr(
        provenance,
        "_is_reparse",
        lambda path: Path(path) == payload_parent,
    )

    with pytest.raises(ValueError, match="reparse.*alpha/data.bin"):
        provenance.build_record_ownership(site_packages)


def test_record_ownership_rejects_two_distributions_claiming_one_file(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    shared = b"shared\n"
    _write_distribution(site_packages, "alpha", files={"shared.py": shared})
    beta = _write_distribution(site_packages, "beta", files={"shared.py": shared})
    _rewrite_record_claim(
        beta,
        "shared.py",
        _record_hash("sha512", shared),
        str(len(shared)),
    )

    with pytest.raises(ValueError, match="conflicting.*shared.py") as caught:
        provenance.build_record_ownership(site_packages)
    assert str(tmp_path) not in str(caught.value)


def test_record_ownership_rejects_identical_shared_claim_without_path_named_owner(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    shared = {"shared/module.py": b"VALUE = 1\n"}
    _write_distribution(site_packages, "alpha-addon", files=shared)
    _write_distribution(site_packages, "beta-addon", files=shared)

    with pytest.raises(
        ValueError,
        match="ambiguous shared RECORD owners.*shared/module.py",
    ):
        provenance.build_record_ownership(site_packages)


def test_record_ownership_selects_path_named_distribution_for_identical_shared_claim(
    tmp_path: Path,
) -> None:
    """PySide6's split wheels legitimately repeat byte-identical RECORD rows.

    Seeded defect: treat every second owner as conflicting; the real
    PySide6/__init__.py analysis entry becomes unresolvable.
    """

    site_packages = tmp_path / "site-packages"
    shared = {"PySide6/__init__.py": b"__version__ = 'test'\n"}
    canonical = _write_distribution(site_packages, "PySide6", files=shared)
    _write_distribution(site_packages, "PySide6-Addons", files=shared)
    _write_distribution(site_packages, "PySide6-Essentials", files=shared)

    ownership = provenance.build_record_ownership(site_packages)

    assert ownership["site-packages://PySide6/__init__.py"] == (
        provenance.DistributionRef(
            name="PySide6",
            version="1.0",
            dist_info_name=canonical.name,
        )
    )


def test_record_ownership_rejects_identical_claim_metadata_over_changed_bytes(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    shared = {"shared.py": b"original\n"}
    _write_distribution(site_packages, "alpha", files=shared)
    _write_distribution(site_packages, "beta", files=shared)
    (site_packages / "shared.py").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="actual bytes.*shared.py") as caught:
        provenance.build_record_ownership(site_packages)
    assert str(tmp_path) not in str(caught.value)


def test_record_ownership_omits_shared_path_when_all_claims_are_unhashed(
    tmp_path: Path,
) -> None:
    """Blank shared pyc rows carry no evidence for choosing a byte owner."""

    site_packages = tmp_path / "site-packages"
    relative = "PySide6/__pycache__/__init__.cpython-312.pyc"
    shared = {relative: b"compiled bytes"}
    blank = frozenset({relative})
    _write_distribution(site_packages, "PySide6", files=shared, unhashed=blank)
    _write_distribution(
        site_packages, "PySide6-Addons", files=shared, unhashed=blank
    )

    ownership = provenance.build_record_ownership(site_packages)

    assert f"site-packages://{relative}" not in ownership


def test_shared_unhashed_record_source_is_unowned_when_inventory_uses_it(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    relative = "PySide6/__pycache__/__init__.cpython-312.pyc"
    shared = {relative: b"compiled bytes"}
    blank = frozenset({relative})
    _write_distribution(site_packages, "PySide6", files=shared, unhashed=blank)
    _write_distribution(
        site_packages, "PySide6-Addons", files=shared, unhashed=blank
    )
    entry = _entry(
        "PySide6.cached_init",
        source_path=f"site-packages://{relative}",
        collection="pure",
    )

    report = _resolve(tmp_path, (entry,), (), site_packages=site_packages)

    assert any(
        "unowned analysis-entry" in error and entry.key in error
        for error in report.errors
    )


def test_shared_record_selection_preserves_each_split_distribution_unique_files(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    shared = b"shared init\n"
    _write_distribution(
        site_packages,
        "PySide6",
        files={"PySide6/__init__.py": shared, "PySide6/core.py": b"core\n"},
    )
    _write_distribution(
        site_packages,
        "PySide6-Addons",
        files={"PySide6/__init__.py": shared, "PySide6/addon.py": b"addon\n"},
    )
    _write_distribution(
        site_packages,
        "PySide6-Essentials",
        files={
            "PySide6/__init__.py": shared,
            "PySide6/essential.py": b"essential\n",
        },
    )
    entries = (
        _entry(
            "PySide6.__init__",
            source_path="site-packages://PySide6/__init__.py",
            collection="pure",
        ),
        _entry(
            "PySide6.addon",
            source_path="site-packages://PySide6/addon.py",
            collection="pure",
        ),
        _entry(
            "PySide6.essential",
            source_path="site-packages://PySide6/essential.py",
            collection="pure",
        ),
    )

    report = _resolve(tmp_path, entries, (), site_packages=site_packages)

    assert report.errors == ()
    assert {record.component_id for record in report.records} == {
        "pyside6",
        "pyside6-addons",
        "pyside6-essentials",
    }


def test_final_bundle_file_links_to_binary_inventory_destination(
    tmp_path: Path,
) -> None:
    entry = _entry("alpha.dll", source_path="repo://vendor/alpha.dll")
    _dist, files = _bundle(
        tmp_path,
        {"_internal/alpha.dll": b"MZ alpha"},
        (entry,),
    )

    assert files == (
        provenance.BundleFile(
            relative_path="_internal/alpha.dll",
            size=len(b"MZ alpha"),
            sha256=hashlib.sha256(b"MZ alpha").hexdigest(),
            inventory_keys=(entry.key,),
        ),
    )


def test_final_bundle_file_uses_exact_destination_not_matching_basename(
    tmp_path: Path,
) -> None:
    entry = _entry("alpha.dll", source_path="repo://vendor/alpha.dll")
    _dist, files = _bundle(
        tmp_path,
        {"_internal/other/alpha.dll": b"MZ other"},
        (entry,),
    )

    assert files[0].inventory_keys == ()


def test_unowned_analysis_entry_is_an_error(tmp_path: Path) -> None:
    entry = _entry(
        "orphan.py",
        source_path="site-packages://orphan.py",
        collection="pure",
    )

    report = _resolve(tmp_path, (entry,), ())

    assert any(
        "unowned analysis-entry" in error and entry.key in error
        for error in report.errors
    )


@pytest.mark.parametrize("collection", ("binaries", "datas", "zipfiles"))
def test_linked_inventory_destination_missing_from_final_tree_is_an_error(
    tmp_path: Path,
    collection: str,
) -> None:
    entry = _entry(
        "dormant.bin",
        source_path="repo://assets/dormant.bin",
        collection=collection,
    )

    report = _resolve(tmp_path, (entry,), ())

    assert any(
        f"{collection} inventory destination {entry.destination}" in error
        and "has no final bundle file" in error
        for error in report.errors
    )


def test_unowned_final_dll_is_an_error(tmp_path: Path) -> None:
    _dist, files = _bundle(tmp_path, {"_internal/orphan.dll": b"MZ orphan"})

    report = _resolve(tmp_path, (), files)

    assert any(
        "unowned bundle-file" in error and "_internal/orphan.dll" in error
        for error in report.errors
    )


def test_non_system_pe_import_without_present_owner_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dist, files = _bundle(tmp_path, {"app.exe": b"MZ app"})
    owner = _component("app", priority=100, globs=("app.exe",))
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["missing_vendor.dll"] if path.name == "app.exe" else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert any(
        "missing_vendor.dll" in error
        and "app.exe" in error
        and "present owner" in error
        for error in report.errors
    )


def test_system_and_driver_pe_imports_do_not_require_bundle_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dist, files = _bundle(tmp_path, {"app.exe": b"MZ app"})
    owner = _component("app", priority=100, globs=("app.exe",))
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["KERNEL32.dll", "nvcuda.dll"] if path.name == "app.exe" else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert report.errors == ()


def test_owned_pyinstaller_hashed_dll_satisfies_unhashed_pe_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed_name = "Foo-ABCDEF0123456789ABCDEF0123456789.DLL"
    _dist, files = _bundle(
        tmp_path,
        {"app.exe": b"MZ app", f"_internal/{hashed_name}": b"MZ foo"},
    )
    owner = _component(
        "runtime",
        priority=100,
        globs=("app.exe", f"_internal/{hashed_name}"),
    )
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["foo.dll"] if path.name == "app.exe" else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert report.errors == ()


def test_unowned_pyinstaller_hashed_dll_does_not_satisfy_pe_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed_name = "foo-abcdef0123456789abcdef0123456789.dll"
    _dist, files = _bundle(
        tmp_path,
        {"app.exe": b"MZ app", f"_internal/{hashed_name}": b"MZ foo"},
    )
    app_owner = _component("app", priority=100, globs=("app.exe",))
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["foo.dll"] if path.name == "app.exe" else [],
    )

    report = _resolve(tmp_path, (), files, app_owner)

    assert any(
        "foo.dll" in error and "present owner" in error
        for error in report.errors
    )


def test_main_tree_pe_import_cannot_use_llama_private_dll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {"app.exe": b"MZ app", "_internal/llama/dep.dll": b"MZ dep"},
    )
    owner = _component(
        "runtime",
        priority=100,
        globs=("app.exe", "_internal/llama/dep.dll"),
    )
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["dep.dll"] if path.name == "app.exe" else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert any(
        "dep.dll" in error and "present owner" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("importer_path", "target_path", "import_name"),
    (
        (
            "_internal/PySide6/plugins/imageformats/qsvg.dll",
            "_internal/PySide6/Qt6Core.dll",
            "Qt6Core.dll",
        ),
        (
            "_internal/PySide6/QtCore.pyd",
            "_internal/shiboken6/shiboken6.abi3.dll",
            "shiboken6.abi3.dll",
        ),
        (
            "_internal/fugashi/fugashi.cp312-win_amd64.pyd",
            "_internal/fugashi.libs/libmecab-0123456789abcdef0123456789abcdef.dll",
            "libmecab-0123456789abcdef0123456789abcdef.dll",
        ),
        (
            "_internal/numpy/_core/_multiarray_umath.cp312-win_amd64.pyd",
            "_internal/numpy.libs/libscipy_openblas64_-0123456789abcdef0123456789abcdef.dll",
            "libscipy_openblas64_-0123456789abcdef0123456789abcdef.dll",
        ),
    ),
)
def test_main_importer_resolves_owned_dll_from_package_registered_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    importer_path: str,
    target_path: str,
    import_name: str,
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {importer_path: b"MZ importer", target_path: b"MZ dependency"},
    )
    owner = _component(
        "runtime",
        priority=100,
        globs=(importer_path, target_path),
    )
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: [import_name] if path.name == Path(importer_path).name else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert report.errors == ()


@pytest.mark.parametrize(
    "nested_name",
    (
        "dep.dll",
        "dep-abcdef0123456789abcdef0123456789.dll",
    ),
)
def test_main_importer_cannot_use_dll_from_unrelated_nested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_name: str,
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {
            "app.exe": b"MZ app",
            f"_internal/plugins/{nested_name}": b"MZ nested dep",
        },
    )
    owner = _component(
        "runtime",
        priority=100,
        globs=("app.exe", f"_internal/plugins/{nested_name}"),
    )
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["dep.dll"] if path.name == "app.exe" else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert any(
        "dep.dll" in error and "present owner" in error
        for error in report.errors
    )


def test_llama_pe_import_resolves_owned_dll_inside_llama_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/llama/server.exe": b"MZ server",
            "_internal/llama/dep.dll": b"MZ dep",
        },
    )
    owner = _component(
        "runtime",
        priority=100,
        globs=("_internal/llama/server.exe", "_internal/llama/dep.dll"),
    )
    monkeypatch.setattr(
        provenance,
        "parse_imports",
        lambda path: ["dep.dll"] if path.name == "server.exe" else [],
    )

    report = _resolve(tmp_path, (), files, owner)

    assert report.errors == ()


def test_higher_priority_vc_rule_wins_over_llama_catchall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/llama/LICENSE.txt": b"MIT\n",
            "_internal/llama/vcruntime140.dll": b"MZ VC",
        },
    )
    vc = _component(
        "microsoft-vc-runtime",
        priority=200,
        globs=("_internal/llama/vcruntime140*.dll",),
    )
    llama = _component(
        "llama-cpp",
        priority=100,
        globs=("_internal/llama/**",),
        license_mode="bundle-file",
        license_source="_internal/llama/LICENSE.txt",
        review_id=None,
        review_reason=None,
    )
    monkeypatch.setattr(provenance, "parse_imports", lambda _path: [])

    low_first_report = _resolve(tmp_path, (), files, llama, vc)
    high_first_report = _resolve(tmp_path, (), files, vc, llama)

    assert low_first_report == high_first_report
    assert low_first_report.errors == ()
    assert (
        _record(
            low_first_report,
            "bundle-file",
            "_internal/llama/vcruntime140.dll",
        ).component_id
        == "microsoft-vc-runtime"
    )
    assert (
        _record(
            low_first_report,
            "bundle-file",
            "_internal/llama/LICENSE.txt",
        ).component_id
        == "llama-cpp"
    )


def test_higher_priority_nvidia_sidecar_rules_win_over_llama_catchall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(site_packages, "nvidia-cublas-cu12")
    _write_distribution(site_packages, "nvidia-cuda-nvrtc-cu12")
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/llama/LICENSE.txt": b"MIT\n",
            "_internal/llama/cublas64_12.dll": b"MZ cublas",
            "_internal/llama/cublasLt64_12.dll": b"MZ cublasLt",
            "_internal/llama/cudart64_12.dll": b"MZ cudart",
        },
    )
    cublas = _component(
        "nvidia-cublas-cu12-sidecars",
        provenance_class="nvidia-runtime",
        priority=300,
        globs=(
            "_internal/llama/cublas64_12.dll",
            "_internal/llama/cublasLt64_12.dll",
        ),
        license_mode="distribution",
        license_source="nvidia-cublas-cu12",
        review_id=None,
        review_reason=None,
    )
    cuda = _component(
        "nvidia-cuda-runtime-cu12-sidecar",
        provenance_class="nvidia-runtime",
        priority=300,
        globs=("_internal/llama/cudart64_12.dll",),
        license_mode="distribution",
        license_source="nvidia-cuda-nvrtc-cu12",
        review_id=None,
        review_reason=None,
    )
    llama = _component(
        "llama-cpp",
        priority=100,
        globs=("_internal/llama/**",),
        license_mode="bundle-file",
        license_source="_internal/llama/LICENSE.txt",
        review_id=None,
        review_reason=None,
    )
    monkeypatch.setattr(provenance, "parse_imports", lambda _path: [])

    low_first_report = _resolve(
        tmp_path,
        (),
        files,
        llama,
        cublas,
        cuda,
        site_packages=site_packages,
    )
    high_first_report = _resolve(
        tmp_path,
        (),
        files,
        cublas,
        cuda,
        llama,
        site_packages=site_packages,
    )

    assert low_first_report == high_first_report
    assert low_first_report.errors == ()
    assert {
        _record(low_first_report, "bundle-file", path).component_id
        for path in (
            "_internal/llama/cublas64_12.dll",
            "_internal/llama/cublasLt64_12.dll",
        )
    } == {"nvidia-cublas-cu12-sidecars"}
    assert (
        _record(
            low_first_report,
            "bundle-file",
            "_internal/llama/cudart64_12.dll",
        ).component_id
        == "nvidia-cuda-runtime-cu12-sidecar"
    )
    assert all(
        _record(low_first_report, "bundle-file", path).distribution is None
        for path in (
            "_internal/llama/cublas64_12.dll",
            "_internal/llama/cublasLt64_12.dll",
            "_internal/llama/cudart64_12.dll",
        )
    )


def test_policy_distribution_license_provider_must_be_installed(
    tmp_path: Path,
) -> None:
    _dist, files = _bundle(tmp_path, {"sidecar.dll": b"MZ sidecar"})
    owner = _component(
        "manual-sidecar",
        priority=100,
        globs=("sidecar.dll",),
        license_mode="distribution",
        license_source="missing-provider",
        review_id=None,
        review_reason=None,
    )

    report = _resolve(tmp_path, (), files, owner)

    assert any(
        "manual-sidecar" in error
        and "unavailable distribution missing-provider" in error
        for error in report.errors
    )
    assert _record(report, "bundle-file", "sidecar.dll").distribution is None


def test_policy_component_id_cannot_collide_with_automatic_distribution(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "Alpha",
        files={"alpha/module.py": b"VALUE = 1\n"},
    )
    entry = _entry(
        "alpha.module",
        source_path="site-packages://alpha/module.py",
        collection="pure",
    )
    _dist, files = _bundle(
        tmp_path,
        {"manual.txt": b"manual"},
    )
    colliding = _component(
        "alpha",
        name="Manual Alpha",
        priority=100,
        globs=("manual.txt",),
    )

    report = _resolve(
        tmp_path,
        (entry,),
        files,
        colliding,
        site_packages=site_packages,
    )

    assert any(
        "policy component id alpha collides with automatic distribution id" in error
        for error in report.errors
    )
    assert any(
        "component id alpha has inconsistent descriptor signatures" in error
        for error in report.errors
    )


def test_equal_highest_priority_matches_are_an_error(tmp_path: Path) -> None:
    _dist, files = _bundle(tmp_path, {"_internal/shared.dll": b"MZ shared"})
    alpha = _component(
        "alpha", priority=200, globs=("_internal/shared.dll",)
    )
    beta = _component(
        "beta", priority=200, globs=("_internal/shared.dll",)
    )

    report = _resolve(tmp_path, (), files, alpha, beta)

    ambiguity = next(error for error in report.errors if "ambiguous" in error)
    assert "alpha" in ambiguity
    assert "beta" in ambiguity
    assert "_internal/shared.dll" in ambiguity


@pytest.mark.parametrize("mode", ("no-raw-match", "shadowed"))
def test_policy_component_with_no_raw_or_effective_match_is_stale(
    tmp_path: Path, mode: str
) -> None:
    if mode == "no-raw-match":
        files: tuple[provenance.BundleFile, ...] = ()
        components = (
            _component("stale", priority=100, globs=("_internal/missing.dll",)),
        )
        expected = "no raw match"
    else:
        _dist, files = _bundle(tmp_path, {"_internal/shared.dll": b"MZ shared"})
        components = (
            _component("stale", priority=100, globs=("_internal/shared.dll",)),
            _component("winner", priority=200, globs=("_internal/shared.dll",)),
        )
        expected = "never owns"

    report = _resolve(tmp_path, (), files, *components)

    assert any(
        "stale policy component 'stale'" in error and expected in error
        for error in report.errors
    )


def test_repo_analysis_source_is_project_owned(tmp_path: Path) -> None:
    entry = _entry(
        "yamabiko.runtime",
        source_path="repo://yamabiko/runtime.py",
        collection="pure",
    )

    report = _resolve(tmp_path, (entry,), ())

    assert report.errors == ()
    record = _record(report, "analysis-entry", entry.destination)
    assert record.component_id == "yamabiko"
    assert record.provenance_class == "project"
    assert record.distribution is None


def test_policy_cannot_override_record_owned_analysis_or_linked_final_file(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/special.dll": b"wheel special"},
    )
    entry = _entry(
        "special.dll",
        source_path="site-packages://alpha/special.dll",
    )
    _dist, files = _bundle(
        tmp_path,
        {"_internal/special.dll": b"wheel special"},
        (entry,),
    )
    explicit = _component(
        "reviewed-special",
        priority=200,
        globs=("_internal/special.dll",),
    )

    report = _resolve(
        tmp_path,
        (entry,),
        files,
        explicit,
        site_packages=site_packages,
    )

    assert any(
        "cannot override RECORD-owned analysis-entry" in error
        for error in report.errors
    )
    assert any(
        "cannot override RECORD-owned bundle-file" in error
        for error in report.errors
    )
    analysis = _record(report, "analysis-entry", entry.destination)
    final = _record(report, "bundle-file", entry.destination)
    assert analysis.component_id == "alpha"
    assert final.component_id == "alpha"
    assert analysis.distribution is not None
    assert final.distribution == analysis.distribution


def test_single_star_policy_glob_does_not_cross_directory_boundary(
    tmp_path: Path,
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {"_internal/nested/dep.dll": b"MZ nested"},
    )
    shallow = _component(
        "shallow",
        priority=100,
        globs=("_internal/*.dll",),
    )

    report = _resolve(tmp_path, (), files, shallow)

    assert any(
        "unowned bundle-file _internal/nested/dep.dll" in error
        for error in report.errors
    )
    assert any(
        "stale policy component 'shallow': no raw match" in error
        for error in report.errors
    )


def test_double_star_policy_glob_can_cross_directory_boundary(
    tmp_path: Path,
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {"_internal/vendor/nested/dep.txt": b"nested"},
    )
    recursive = _component(
        "recursive",
        priority=100,
        globs=("_internal/**/dep.txt",),
    )

    report = _resolve(tmp_path, (), files, recursive)

    assert report.errors == ()
    assert (
        _record(report, "bundle-file", "_internal/vendor/nested/dep.txt").component_id
        == "recursive"
    )


def test_generated_python_and_bootloader_files_have_explicit_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(site_packages, "PyInstaller", "6.19.0")
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/base_library.zip": b"zip data",
            "_internal/python312.dll": b"MZ python",
            "yamabiko.exe": b"MZ bootloader",
        },
    )
    python_runtime = _component(
        "python-runtime",
        provenance_class="python-runtime",
        priority=200,
        globs=("_internal/python3*.dll", "_internal/base_library.zip"),
        license_mode="python-runtime-file",
        license_source="python://LICENSE.txt",
        review_id=None,
        review_reason=None,
    )
    bootloader = _component(
        "pyinstaller-bootloader",
        provenance_class="bootloader",
        priority=200,
        globs=("yamabiko.exe",),
        license_mode="distribution",
        license_source="pyinstaller",
        review_id=None,
        review_reason=None,
    )
    monkeypatch.setattr(provenance, "parse_imports", lambda _path: [])

    report = _resolve(
        tmp_path,
        (),
        files,
        python_runtime,
        bootloader,
        site_packages=site_packages,
    )

    assert report.errors == ()
    assert (
        _record(report, "bundle-file", "_internal/python312.dll").component_id
        == "python-runtime"
    )
    assert (
        _record(report, "bundle-file", "_internal/base_library.zip").component_id
        == "python-runtime"
    )
    assert (
        _record(report, "bundle-file", "yamabiko.exe").component_id
        == "pyinstaller-bootloader"
    )
    assert _record(report, "bundle-file", "yamabiko.exe").distribution is None


@pytest.mark.parametrize("missing", ("review_id", "review_reason"))
def test_reviewed_exception_requires_nonempty_review_fields(
    tmp_path: Path, missing: str
) -> None:
    kwargs = {
        "review_id": "REVIEW-1",
        "review_reason": "Reviewed legal exception.",
    }
    kwargs[missing] = None
    component = _component(
        "exception",
        priority=100,
        globs=("artifact.dll",),
        **kwargs,
    )
    policy = _write_policy(tmp_path / "policy.toml", component)

    with pytest.raises(ValueError, match=missing.replace("_", " ")):
        provenance.load_provenance_policy(policy)


def test_license_payload_is_excluded_from_product_file_enumeration(
    tmp_path: Path,
) -> None:
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/licenses/orphan.dll": b"not product",
            "_internal/licenses/INDEX.json": b"{}",
        },
    )

    assert files == ()


def test_analysis_and_final_tree_are_distinct_provenance_subjects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(
        site_packages,
        "alpha",
        files={"alpha/native.dll": b"wheel native"},
    )
    entry = _entry(
        "alpha/native.dll",
        source_path="site-packages://alpha/native.dll",
    )
    _dist, files = _bundle(
        tmp_path,
        {"_internal/alpha/native.dll": b"frozen native"},
        (entry,),
    )
    monkeypatch.setattr(provenance, "parse_imports", lambda _path: [])

    report = _resolve(
        tmp_path,
        (entry,),
        files,
        site_packages=site_packages,
    )

    assert report.errors == ()
    analysis = _record(report, "analysis-entry", entry.destination)
    final = _record(report, "bundle-file", entry.destination)
    assert analysis.subject.key == entry.key
    assert final.subject.key == final.subject.path == entry.destination
    assert analysis.component_id == final.component_id == "alpha"


def test_native_bundle_file_without_inspection_path_fails_closed(
    tmp_path: Path,
) -> None:
    bundle_file = provenance.BundleFile(
        relative_path="app.exe",
        size=6,
        sha256=hashlib.sha256(b"MZ app").hexdigest(),
        inventory_keys=(),
    )
    owner = _component("app", priority=100, globs=("app.exe",))

    report = _resolve(tmp_path, (), (bundle_file,), owner)

    assert any(
        "app.exe" in error and "inspection path" in error
        for error in report.errors
    )


def test_native_pe_parse_failure_is_a_closure_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dist, files = _bundle(tmp_path, {"app.exe": b"MZ malformed"})
    owner = _component("app", priority=100, globs=("app.exe",))

    def fail_parse(_path: Path) -> list[str]:
        raise ValueError("synthetic PE parse failure")

    monkeypatch.setattr(provenance, "parse_imports", fail_parse)

    report = _resolve(tmp_path, (), files, owner)

    assert any(
        "app.exe" in error and "PE imports could not be parsed" in error
        for error in report.errors
    )


def test_ci_llama_stub_is_exempt_from_pe_parse_only_under_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exact bytes installer/ci-prepare-vendor-stubs.ps1 writes on Windows.
    stub = b"CI stub (not a real binary)\r\n"
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/llama/ggml.dll": stub,
            "_internal/llama/llama.dll": b"MZ corrupt real binary",
        },
    )
    owner = _component(
        "llama",
        priority=100,
        globs=("_internal/llama/ggml.dll", "_internal/llama/llama.dll"),
    )
    parse_error = "native bundle-file _internal/llama/{} PE imports could not be parsed"

    monkeypatch.setenv("YAMABIKO_ALLOW_STUB_LLAMA", "1")
    opted_in = _resolve(tmp_path, (), files, owner)
    monkeypatch.delenv("YAMABIKO_ALLOW_STUB_LLAMA")
    strict = _resolve(tmp_path, (), files, owner)

    assert opted_in.errors == (parse_error.format("llama.dll"),)
    assert parse_error.format("ggml.dll") in strict.errors


def test_native_inspection_path_must_remain_inside_enumerated_dist_root(
    tmp_path: Path,
) -> None:
    _dist, files = _bundle(tmp_path, {"app.exe": b"MZ app"})
    outside = tmp_path / "outside" / "app.exe"
    outside.parent.mkdir()
    outside.write_bytes(b"MZ outside")
    forged = replace(files[0], _filesystem_path=outside)
    owner = _component("app", priority=100, globs=("app.exe",))

    report = _resolve(tmp_path, (), (forged,), owner)

    assert any(
        "app.exe" in error and "outside enumerated dist root" in error
        for error in report.errors
    )
    assert all(str(tmp_path) not in error for error in report.errors)


def test_report_is_canonical_and_never_serializes_machine_paths(
    tmp_path: Path,
) -> None:
    entries = (
        _entry(
            "assets/z.txt",
            source_path="repo://assets/z.txt",
            collection="datas",
        ),
        _entry(
            "assets/a.txt",
            source_path="repo://assets/a.txt",
            collection="datas",
        ),
    )
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/assets/z.txt": b"z payload",
            "_internal/assets/a.txt": b"a payload",
            "_internal/manual/z.txt": b"manual z",
            "_internal/manual/a.txt": b"manual a",
        },
        entries,
    )
    manual_a = _component(
        "manual-a",
        priority=100,
        globs=("_internal/manual/a.txt",),
    )
    manual_z = _component(
        "manual-z",
        priority=100,
        globs=("_internal/manual/z.txt",),
    )
    first_report = _resolve(
        tmp_path,
        entries,
        files,
        manual_a,
        manual_z,
    )
    second_report = _resolve(
        tmp_path,
        tuple(reversed(entries)),
        tuple(reversed(files)),
        manual_z,
        manual_a,
    )
    assert first_report.errors == ()
    assert second_report.errors == ()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    provenance.write_provenance_report(first_report, first)
    provenance.write_provenance_report(second_report, second)

    assert first.read_bytes() == second.read_bytes()
    raw = first.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert raw == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in raw
    assert "_filesystem_path" not in raw
    assert "_dist_root" not in raw
    assert set(payload) == {"schema_version", "records", "bundle_files", "errors"}
    assert set(payload["bundle_files"][0]) == {
        "relative_path",
        "size",
        "sha256",
        "inventory_keys",
    }
    project_record = next(
        row for row in payload["records"] if row["subject"]["kind"] == "analysis-entry"
    )
    assert set(project_record) == {
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
    assert project_record["distribution"] is None
    assert project_record["license_source"] is None
    assert project_record["review_id"] is None
    assert project_record["review_reason"] is None


def test_report_writer_rejects_error_report_without_publishing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    report = provenance.ClosureReport(records=(), bundle_files=(), errors=("bad",))

    with pytest.raises(ValueError, match="errors"):
        provenance.write_provenance_report(report, output)

    assert not output.exists()


def _checked_in_llama_manifest_paths() -> tuple[str, ...]:
    manifest = _REPO_ROOT / "installer" / "llama-manifest.txt"
    paths = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        filename = raw_line.partition("#")[0].strip()
        if filename:
            paths.append(f"_internal/llama/{filename}")
    return tuple(paths)


def test_checked_in_llama_policy_globs_equal_build_manifest() -> None:
    manifest_paths = _checked_in_llama_manifest_paths()
    policy = provenance.load_provenance_policy(
        _REPO_ROOT / "installer" / "bundle-provenance.toml"
    )
    llama = next(
        component
        for component in policy.components
        if component.component_id == "llama-cpp"
    )

    assert len(manifest_paths) == 15
    assert len(set(manifest_paths)) == len(manifest_paths)
    assert set(llama.globs) == set(manifest_paths)


# The checked-in policy owns these exact files; every component must match
# something in a bundle built from it.
_SOUND_TAGGER_FILES = {
    "_internal/yamabiko/vad/data/efficientsed-fmn10-tf256-kd.onnx": b"onnx graph",
    "_internal/yamabiko/vad/data/LICENSE-EfficientSED.txt": b"MIT\n",
}
_FONT_FILES = {
    "_internal/yamabiko/display/static/fonts/BIZUDPGothic-Regular.ttf": b"ttf regular",
    "_internal/yamabiko/display/static/fonts/BIZUDPGothic-Bold.ttf": b"ttf bold",
    "_internal/yamabiko/display/static/fonts/OFL-BIZUDPGothic.txt": b"OFL\n",
    "_internal/yamabiko/display/static/fonts/BIZUDPMincho-Regular.ttf": b"ttf regular",
    "_internal/yamabiko/display/static/fonts/BIZUDPMincho-Bold.ttf": b"ttf bold",
    "_internal/yamabiko/display/static/fonts/OFL-BIZUDPMincho.txt": b"OFL\n",
    "_internal/yamabiko/display/static/fonts/KleeOne-SemiBold.ttf": b"ttf semibold",
    "_internal/yamabiko/display/static/fonts/OFL-KleeOne.txt": b"OFL\n",
}
_PREVIEW_PHOTO_FILES = {
    "_internal/yamabiko/overlay_app/assets/preview-night.jpg": b"jpeg",
    "_internal/yamabiko/overlay_app/assets/LICENSE-preview-night.txt": b"CC BY 2.0\n",
}


def test_checked_in_llama_policy_assigns_unique_highest_priority_owner_per_manifest_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_paths = _checked_in_llama_manifest_paths()
    expected_owners = {
        "_internal/llama/LICENSE.txt": "llama-cpp",
        "_internal/llama/llama-server.exe": "llama-cpp",
        "_internal/llama/cublas64_12.dll": "nvidia-cublas-cu12-sidecars",
        "_internal/llama/cublasLt64_12.dll": "nvidia-cublas-cu12-sidecars",
        "_internal/llama/cudart64_12.dll": "nvidia-cuda-runtime-cu12-sidecar",
        "_internal/llama/ggml-base.dll": "llama-cpp",
        "_internal/llama/ggml-cpu.dll": "llama-cpp",
        "_internal/llama/ggml-cuda.dll": "llama-cpp",
        "_internal/llama/ggml.dll": "llama-cpp",
        "_internal/llama/llama.dll": "llama-cpp",
        "_internal/llama/mtmd.dll": "llama-cpp",
        "_internal/llama/msvcp140.dll": "microsoft-vc-runtime",
        "_internal/llama/vcomp140.dll": "microsoft-vc-runtime",
        "_internal/llama/vcruntime140.dll": "microsoft-vc-runtime",
        "_internal/llama/vcruntime140_1.dll": "microsoft-vc-runtime",
    }
    assert set(expected_owners) == set(manifest_paths)

    site_packages = tmp_path / "site-packages"
    _write_distribution(site_packages, "nvidia-cublas-cu12")
    _write_distribution(site_packages, "nvidia-cuda-nvrtc-cu12")
    _write_distribution(site_packages, "PyInstaller", "6.19.0")
    bundle_payloads = {path: b"MZ llama fixture" for path in manifest_paths}
    bundle_payloads.update(
        {
            "_internal/base_library.zip": b"zip data",
            "_internal/python312.dll": b"MZ python",
            "yamabiko.exe": b"MZ bootloader",
            **_SOUND_TAGGER_FILES,
            **_FONT_FILES,
            **_PREVIEW_PHOTO_FILES,
        }
    )
    _dist, files = _bundle(tmp_path, bundle_payloads)
    monkeypatch.setattr(provenance, "parse_imports", lambda path: [])

    report = provenance.resolve_provenance(
        (),
        files,
        site_packages=site_packages,
        policy_config_path=_REPO_ROOT / "installer" / "bundle-provenance.toml",
    )

    assert report.errors == ()
    assert {
        path: _record(report, "bundle-file", path).component_id
        for path in manifest_paths
    } == expected_owners


def test_default_policy_explicitly_owns_root_and_llama_vc_runtime_copies() -> None:
    policy = provenance.load_provenance_policy(
        _REPO_ROOT / "installer" / "bundle-provenance.toml"
    )

    vc = next(
        component
        for component in policy.components
        if component.component_id == "microsoft-vc-runtime"
    )
    assert "_internal/msvcp140*.dll" in vc.globs
    assert "_internal/vcruntime140*.dll" in vc.globs
    assert "_internal/llama/msvcp140*.dll" in vc.globs
    assert "_internal/llama/vcruntime140*.dll" in vc.globs


def test_checked_in_policy_owns_python_runtime_and_bootloader_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _write_distribution(site_packages, "nvidia-cublas-cu12")
    _write_distribution(site_packages, "nvidia-cuda-nvrtc-cu12")
    _write_distribution(site_packages, "PyInstaller", "6.19.0")
    _dist, files = _bundle(
        tmp_path,
        {
            "_internal/llama/LICENSE.txt": b"MIT\n",
            "_internal/llama/cublas64_12.dll": b"MZ cublas",
            "_internal/llama/cudart64_12.dll": b"MZ cudart",
            "_internal/vcruntime140.dll": b"MZ VC",
            "_internal/base_library.zip": b"zip data",
            "_internal/python312.dll": b"MZ python",
            "yamabiko.exe": b"MZ bootloader",
            **_SOUND_TAGGER_FILES,
            **_FONT_FILES,
            **_PREVIEW_PHOTO_FILES,
        },
    )
    monkeypatch.setattr(provenance, "parse_imports", lambda path: [])

    report = provenance.resolve_provenance(
        (),
        files,
        site_packages=site_packages,
        policy_config_path=_REPO_ROOT / "installer" / "bundle-provenance.toml",
    )

    assert report.errors == ()
    assert (
        _record(report, "bundle-file", "_internal/base_library.zip").component_id
        == "python-runtime"
    )
    assert (
        _record(report, "bundle-file", "_internal/python312.dll").component_id
        == "python-runtime"
    )
    assert (
        _record(report, "bundle-file", "yamabiko.exe").component_id
        == "pyinstaller-bootloader"
    )
    assert _record(report, "bundle-file", "yamabiko.exe").distribution is None


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_direct_cli_bootstraps_repo_and_writes_deterministic_report_only_on_success(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    entry = _entry(
        "assets/data.txt",
        source_path="repo://assets/data.txt",
        collection="datas",
    )
    inventory = _write_inventory(tmp_path / "inventory.json", (entry,))
    dist_root = tmp_path / "dist"
    (dist_root / "_internal" / "assets").mkdir(parents=True)
    (dist_root / "_internal" / "assets" / "data.txt").write_bytes(b"payload")
    policy = _write_policy(tmp_path / "policy.toml")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = (
        "resolve",
        "--inventory",
        str(inventory),
        "--dist",
        str(dist_root),
        "--policy-config",
        str(policy),
        "--site-packages",
        str(site_packages),
    )

    one = _run_cli(*common, "--output", str(first), cwd=tmp_path)
    two = _run_cli(*common, "--output", str(second), cwd=tmp_path)

    assert one.returncode == two.returncode == 0
    assert one.stdout == one.stderr == ""
    assert two.stdout == two.stderr == ""
    assert first.read_bytes() == second.read_bytes()
    assert str(tmp_path).encode() not in first.read_bytes()


@pytest.mark.parametrize("failure", ("resolution", "malformed-inventory"))
def test_direct_cli_fails_closed_without_leaving_a_success_report(
    tmp_path: Path, failure: str
) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    inventory = tmp_path / "inventory.json"
    if failure == "resolution":
        _write_inventory(inventory, ())
    else:
        inventory.write_text("{not-json", encoding="utf-8")
    dist_root = tmp_path / "dist"
    dist_root.mkdir()
    if failure == "resolution":
        (dist_root / "orphan.txt").write_text("orphan", encoding="utf-8")
    policy = _write_policy(tmp_path / "policy.toml")
    output = tmp_path / "report.json"
    output.write_text('{"status":"stale-success"}\n', encoding="utf-8")

    result = _run_cli(
        "resolve",
        "--inventory",
        str(inventory),
        "--dist",
        str(dist_root),
        "--policy-config",
        str(policy),
        "--site-packages",
        str(site_packages),
        "--output",
        str(output),
        cwd=tmp_path,
    )

    assert result.returncode == (1 if failure == "resolution" else 2)
    assert result.stdout == ""
    assert result.stderr
    assert "Traceback" not in result.stderr
    assert str(tmp_path) not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("joined", (False, True))
def test_direct_cli_parse_failure_removes_stale_output(
    tmp_path: Path, joined: bool
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"status":"stale-success"}\n', encoding="utf-8")
    output_args = (
        (f"--output={output}",)
        if joined
        else ("--output", str(output))
    )

    result = _run_cli("resolve", *output_args, cwd=tmp_path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_direct_cli_missing_output_value_does_not_delete_next_option_name(
    tmp_path: Path,
) -> None:
    option_named_file = tmp_path / "--inventory"
    option_named_file.write_text("must survive\n", encoding="utf-8")

    result = _run_cli(
        "resolve",
        "--output",
        "--inventory",
        "missing.json",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert option_named_file.read_text(encoding="utf-8") == "must survive\n"


def test_direct_cli_rejects_abbreviated_output_without_deleting_its_value(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"status":"stale-success"}\n', encoding="utf-8")

    result = _run_cli(
        "resolve",
        "--inventory",
        "missing-inventory.json",
        "--dist",
        "missing-dist",
        "--policy-config",
        "missing-policy.toml",
        "--site-packages",
        "missing-site-packages",
        "--out",
        str(output),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "--out" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"status":"stale-success"}\n'


def test_direct_cli_output_scanner_stops_at_option_terminator(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"status":"stale-success"}\n', encoding="utf-8")

    result = _run_cli(
        "resolve",
        "--",
        "--output",
        str(output),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == '{"status":"stale-success"}\n'


def test_direct_cli_root_terminator_does_not_hide_subcommand_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"status":"stale-success"}\n', encoding="utf-8")

    result = _run_cli(
        "--",
        "resolve",
        "--output",
        str(output),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert not output.exists()
