"""Contract tests for the bundle-derived third-party license payload.

The collector is a release boundary.  These tests construct inventories,
provenance reports, wheel metadata, and RECORD claims together so a synthetic
fixture fails for the same reasons a frozen bundle would fail.
"""
from __future__ import annotations

import base64
import csv
from dataclasses import asdict
import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_licenses  # noqa: E402
from scripts.bundle_inventory import InventoryEntry  # noqa: E402


MANIFEST = ROOT / "installer" / "licenses-manifest.txt"
LICENSES_DIR = ROOT / "licenses"
NOTICE = ROOT / "NOTICE"
LLAMA_MANIFEST = ROOT / "installer" / "llama-manifest.txt"
BUNDLE = ROOT / "dist" / "yamabiko"
INVENTORY = ROOT / "build" / "yamabiko" / "bundle-inventory.json"
PROVENANCE = ROOT / "build" / "yamabiko" / "bundle-provenance.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _entry(component_id: str, ordinal: int) -> InventoryEntry:
    logical_name = f"fixture/{ordinal:03d}-{component_id}.bin"
    return InventoryEntry(
        collection="datas",
        logical_name=logical_name,
        source_path=f"site-packages://fixture/{ordinal:03d}-{component_id}.bin",
        destination=f"_internal/{logical_name}",
        type_code="DATA",
    )


def _distribution_record(
    entry: InventoryEntry,
    component_id: str,
    *,
    name: str | None = None,
    version: str = "1",
    distribution_name: str | None = None,
    license_source: str | None = None,
) -> dict[str, object]:
    distribution_name = distribution_name or name or component_id
    return {
        "subject": {
            "kind": "analysis-entry",
            "key": entry.key,
            "path": entry.destination,
        },
        "component_id": component_id,
        "component_name": name or component_id,
        "provenance_class": "distribution",
        "distribution": {
            "name": distribution_name,
            "version": version,
            "dist_info_name": (
                f"{distribution_name.replace('-', '_')}-{version}.dist-info"
            ),
        },
        "license_mode": "distribution",
        "license_source": license_source or component_id,
        "review_id": None,
        "review_reason": None,
    }


def _special_record(
    *,
    kind: str,
    key: str,
    path: str,
    component_id: str,
    component_name: str,
    provenance_class: str,
    license_mode: str,
    license_source: str | None,
    review_id: str | None = None,
    review_reason: str | None = None,
) -> dict[str, object]:
    return {
        "subject": {"kind": kind, "key": key, "path": path},
        "component_id": component_id,
        "component_name": component_name,
        "provenance_class": provenance_class,
        "distribution": None,
        "license_mode": license_mode,
        "license_source": license_source,
        "review_id": review_id,
        "review_reason": review_reason,
    }


def _inputs(
    tmp_path: Path,
    records: list[dict[str, object]],
    entries: list[InventoryEntry],
    *,
    bundle_files: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    inventory = _write_json(
        tmp_path / "bundle-inventory.json",
        {"schema_version": 1, "entries": [asdict(entry) for entry in entries]},
    )
    provenance = _write_json(
        tmp_path / "bundle-provenance.json",
        {
            "schema_version": 1,
            "records": records,
            "bundle_files": bundle_files or [],
            "errors": [],
        },
    )
    return inventory, provenance


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + encoded.decode("ascii").rstrip("=")


def _wheel(
    site_packages: Path,
    name: str,
    version: str,
    legal_files: dict[str, bytes],
) -> Path:
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        + "".join(f"License-File: {relative}\n" for relative in legal_files)
        + "\n"
    ).encode("utf-8")
    (dist_info / "METADATA").write_bytes(metadata)
    for relative, content in legal_files.items():
        target = dist_info / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    rows: list[list[str]] = []
    for relative in ["METADATA", *legal_files]:
        content = (dist_info / Path(relative)).read_bytes()
        rows.append(
            [
                f"{dist_info.name}/{Path(relative).as_posix()}",
                _record_hash(content),
                str(len(content)),
            ]
        )
    rows.append([f"{dist_info.name}/RECORD", "", ""])
    with (dist_info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return dist_info


def _policy(tmp_path: Path, text: str = "") -> Path:
    path = tmp_path / "licenses-manifest.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _dist_root(tmp_path: Path) -> Path:
    root = tmp_path / "dist" / "yamabiko"
    (root / "_internal").mkdir(parents=True)
    return root


def _collect(
    dist_root: Path,
    *,
    inventory: Path,
    provenance: Path,
    policy: Path,
    site_packages: Path,
) -> int:
    return collect_licenses.collect(
        dist_root,
        inventory_path=inventory,
        provenance_report_path=provenance,
        override_policy_path=policy,
        site_packages=site_packages,
    )


def _index(dist_root: Path) -> list[dict[str, object]]:
    payload = json.loads(
        (dist_root / "_internal" / "licenses" / "INDEX.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload) == {"schema_version", "entries"}
    assert payload["schema_version"] == 1
    return payload["entries"]


def _manifest_entries() -> list[tuple[str, str, str]]:
    return collect_licenses._parse_manifest(MANIFEST.read_text(encoding="utf-8"))


def test_collect_derives_distribution_not_named_in_override_policy(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"licenses/LICENSE.txt": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 0
    output = (
        dist_root
        / "_internal/licenses/components/alpha/1/licenses/LICENSE.txt"
    )
    assert output.read_bytes() == b"alpha license"
    assert {row["component_id"] for row in _index(dist_root)} == {"alpha"}


def test_collect_fails_when_derived_distribution_has_no_license_or_override(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert not (dist_root / "_internal/licenses").exists()


def test_collect_preserves_relative_paths_when_legal_basenames_collide(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    first = b"license notice"
    second = b"vendor notice"
    _wheel(
        site_packages,
        "alpha",
        "1",
        {
            "licenses/NOTICE.md": first,
            "vendor/NOTICE.md": second,
        },
    )
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path),
            site_packages=site_packages,
        )
        == 0
    )

    base = dist_root / "_internal/licenses/components/alpha/1"
    assert (base / "licenses/NOTICE.md").read_bytes() == first
    assert (base / "vendor/NOTICE.md").read_bytes() == second
    by_path = {row["output_path"]: row for row in _index(dist_root)}
    for relative, content in {
        "_internal/licenses/components/alpha/1/licenses/NOTICE.md": first,
        "_internal/licenses/components/alpha/1/vendor/NOTICE.md": second,
    }.items():
        assert by_path[relative]["sha256"] == hashlib.sha256(content).hexdigest()


def test_collect_rejects_stale_file_override(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    (tmp_path / "licenses").mkdir()
    (tmp_path / "licenses" / "ghost.txt").write_bytes(b"ghost license")
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path, "file:ghost|ghost.txt\n"),
            site_packages=site_packages,
        )
        == 1
    )
    assert not (dist_root / "_internal/licenses").exists()


@pytest.mark.parametrize(
    "policy_text",
    [
        "alias:ghost|alpha\n",
        "alias:alpha|ghost\n",
        "alias:alpha|beta\nalias:alpha|gamma\n",
    ],
)
def test_collect_rejects_stale_or_conflicting_alias(
    tmp_path: Path, policy_text: str
) -> None:
    site_packages = tmp_path / "site-packages"
    records: list[dict[str, object]] = []
    entries: list[InventoryEntry] = []
    for ordinal, component in enumerate(("alpha", "beta", "gamma"), start=1):
        _wheel(site_packages, component, "1", {"LICENSE": component.encode()})
        entry = _entry(component, ordinal)
        entries.append(entry)
        records.append(_distribution_record(entry, component))
    inventory, provenance = _inputs(tmp_path, records, entries)

    assert (
        _collect(
            _dist_root(tmp_path),
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path, policy_text),
            site_packages=site_packages,
        )
        == 1
    )


def test_collect_failure_leaves_existing_license_directory_byte_identical(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    _wheel(site_packages, "beta", "1", {})
    entries = [_entry("alpha", 1), _entry("beta", 2)]
    records = [
        _distribution_record(entries[0], "alpha"),
        _distribution_record(entries[1], "beta"),
    ]
    inventory, provenance = _inputs(tmp_path, records, entries)
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    (target / "components/old/1").mkdir(parents=True)
    (target / "components/old/1/LICENSE").write_bytes(b"known good")
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert _tree_hashes(target) == before


def test_collect_publish_rename_failure_restores_existing_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    target.mkdir()
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    original_rename = Path.rename

    def fail_staging_publish(source: Path, destination: Path) -> Path:
        if source.name.startswith(".licenses-staging-") and Path(destination).name == "licenses":
            raise OSError("seeded staging publication failure")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_staging_publish)

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert _tree_hashes(target) == before


def test_collect_post_publish_verification_failure_restores_existing_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    target.mkdir()
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    original_verify = collect_licenses._verify_payload
    calls = 0

    def fail_second_verification(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("seeded post-publication verification failure")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(collect_licenses, "_verify_payload", fail_second_verification)

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert calls == 2
    assert rc == 1
    assert _tree_hashes(target) == before


@pytest.mark.parametrize(
    "artifact_name",
    [
        ".licenses-staging-dead",
        ".licenses-backup-dead",
        ".licenses-quarantine-dead",
    ],
)
def test_collect_rejects_stale_crash_artifact_without_touching_target(
    tmp_path: Path, artifact_name: str
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    target.mkdir()
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    crash_artifact = dist_root / "_internal" / artifact_name
    crash_artifact.mkdir()
    (crash_artifact / "orphan.txt").write_bytes(b"must not ship")

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert _tree_hashes(target) == before
    assert (crash_artifact / "orphan.txt").read_bytes() == b"must not ship"


def test_collect_rejects_existing_publish_lock_without_touching_target(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    target.mkdir()
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    lock = dist_root / "_internal/.licenses-publish.lock"
    lock.mkdir()

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert _tree_hashes(target) == before
    assert lock.is_dir()


def test_collect_binds_bundle_license_to_reported_size_and_hash(tmp_path: Path) -> None:
    dist_root = _dist_root(tmp_path)
    source_path = "_internal/llama/LICENSE.txt"
    source = dist_root / source_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"changed after provenance")
    record = _special_record(
        kind="bundle-file",
        key=source_path,
        path=source_path,
        component_id="llama-cpp",
        component_name="llama.cpp",
        provenance_class="vendored-native",
        license_mode="bundle-file",
        license_source=source_path,
    )
    inventory, provenance = _inputs(
        tmp_path,
        [record],
        [],
        bundle_files=[
            {
                "relative_path": source_path,
                "size": len(b"original"),
                "sha256": hashlib.sha256(b"original").hexdigest(),
                "inventory_keys": [],
            }
        ],
    )
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path),
            site_packages=site_packages,
        )
        == 1
    )
    assert not (dist_root / "_internal/licenses").exists()


def test_collect_revalidates_wheel_record_claim_after_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _wheel(
        site_packages, "alpha", "1", {"LICENSE": b"record-verified license"}
    )
    legal_file = dist_info / "LICENSE"
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    target.mkdir()
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    real_build_ownership = collect_licenses.build_record_ownership
    calls = 0

    def mutate_after_first_verification(path: Path):
        nonlocal calls
        calls += 1
        ownership = real_build_ownership(path)
        if calls == 1:
            legal_file.write_bytes(b"replacement after RECORD verification")
        return ownership

    monkeypatch.setattr(
        collect_licenses,
        "build_record_ownership",
        mutate_after_first_verification,
    )

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert calls == 2
    assert rc == 1
    assert _tree_hashes(target) == before


def test_collect_rejects_override_for_non_distribution_component(
    tmp_path: Path,
) -> None:
    dist_root = _dist_root(tmp_path)
    source_path = "_internal/llama/LICENSE.txt"
    source = dist_root / source_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"llama license")
    record = _special_record(
        kind="bundle-file",
        key=source_path,
        path=source_path,
        component_id="llama-cpp",
        component_name="llama.cpp",
        provenance_class="vendored-native",
        license_mode="bundle-file",
        license_source=source_path,
    )
    inventory, provenance = _inputs(
        tmp_path,
        [record],
        [],
        bundle_files=[
            {
                "relative_path": source_path,
                "size": source.stat().st_size,
                "sha256": _sha256(source),
                "inventory_keys": [],
            }
        ],
    )
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path, "file:llama-cpp|ctranslate2-MIT.txt\n"),
            site_packages=site_packages,
        )
        == 1
    )


def test_collect_rejects_reparse_ancestor_of_bundle_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist_root = _dist_root(tmp_path)
    source_path = "_internal/llama/LICENSE.txt"
    source = dist_root / source_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"llama license")
    record = _special_record(
        kind="bundle-file",
        key=source_path,
        path=source_path,
        component_id="llama-cpp",
        component_name="llama.cpp",
        provenance_class="vendored-native",
        license_mode="bundle-file",
        license_source=source_path,
    )
    inventory, provenance = _inputs(
        tmp_path,
        [record],
        [],
        bundle_files=[
            {
                "relative_path": source_path,
                "size": source.stat().st_size,
                "sha256": _sha256(source),
                "inventory_keys": [],
            }
        ],
    )
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    real_is_reparse = collect_licenses._is_reparse
    monkeypatch.setattr(
        collect_licenses,
        "_is_reparse",
        lambda path: Path(path).name == "llama" or real_is_reparse(path),
    )

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path),
            site_packages=site_packages,
        )
        == 1
    )


def test_collect_rejects_reparse_bundle_root_before_lock_or_target_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    target.mkdir()
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    real_is_reparse = collect_licenses._is_reparse
    monkeypatch.setattr(
        collect_licenses,
        "_is_reparse",
        lambda path: Path(path) == dist_root or real_is_reparse(path),
    )

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert _tree_hashes(target) == before
    assert not (dist_root / "_internal/.licenses-publish.lock").exists()


def test_collect_rejects_reparse_in_existing_target_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1", {"LICENSE": b"alpha license"})
    entry = _entry("alpha", 1)
    inventory, provenance = _inputs(
        tmp_path, [_distribution_record(entry, "alpha")], [entry]
    )
    dist_root = _dist_root(tmp_path)
    target = dist_root / "_internal/licenses"
    linked = target / "linked"
    linked.mkdir(parents=True)
    (linked / "old.txt").write_bytes(b"known good")
    (target / "INDEX.json").write_bytes(b'{"known":"good"}\n')
    (target / "INDEX.txt").write_bytes(b"known good\n")
    before = _tree_hashes(target)
    real_is_reparse = collect_licenses._is_reparse
    monkeypatch.setattr(
        collect_licenses,
        "_is_reparse",
        lambda path: Path(path).name == "linked" or real_is_reparse(path),
    )

    rc = _collect(
        dist_root,
        inventory=inventory,
        provenance=provenance,
        policy=_policy(tmp_path),
        site_packages=site_packages,
    )

    assert rc == 1
    assert _tree_hashes(target) == before
    assert not list((dist_root / "_internal").glob(".licenses-backup-*"))


def test_alias_redirects_text_without_collapsing_provenance_owner_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "pyside6", "6.1", {})
    _wheel(site_packages, "pyside6-addons", "6.2", {})
    entries = [_entry("pyside6", 1), _entry("pyside6-addons", 2)]
    records = [
        _distribution_record(entries[0], "pyside6", version="6.1"),
        _distribution_record(entries[1], "pyside6-addons", version="6.2"),
    ]
    inventory, provenance = _inputs(tmp_path, records, entries)
    overrides = tmp_path / "licenses"
    overrides.mkdir()
    (overrides / "Qt-LGPL.txt").write_bytes(b"reviewed LGPL text")
    monkeypatch.setattr(collect_licenses, "REPO_LICENSES", overrides)
    policy = _policy(
        tmp_path,
        "file:pyside6|Qt-LGPL.txt\n"
        "alias:pyside6-addons|pyside6\n",
    )
    dist_root = _dist_root(tmp_path)

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=policy,
            site_packages=site_packages,
        )
        == 0
    )
    rows = _index(dist_root)
    assert {row["component_id"] for row in rows} == {
        "pyside6",
        "pyside6-addons",
    }
    by_component = {row["component_id"]: row for row in rows}
    assert by_component["pyside6"]["version"] == "6.1"
    assert by_component["pyside6-addons"]["version"] == "6.2"
    assert "/components/pyside6/override/" in by_component["pyside6"]["output_path"]
    assert (
        "/components/pyside6-addons/override/"
        in by_component["pyside6-addons"]["output_path"]
    )


def test_provider_distribution_supplies_text_without_replacing_owner_identity(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(
        site_packages,
        "nvidia-provider",
        "12.9",
        {"licenses/LICENSE.txt": b"provider terms"},
    )
    sidecar = _entry("nvidia-sidecars", 1)
    record = _special_record(
        kind="analysis-entry",
        key=sidecar.key,
        path=sidecar.destination,
        component_id="nvidia-sidecars",
        component_name="NVIDIA sidecars",
        provenance_class="policy-runtime",
        license_mode="distribution",
        license_source="nvidia-provider",
    )
    inventory, provenance = _inputs(tmp_path, [record], [sidecar])
    dist_root = _dist_root(tmp_path)

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path),
            site_packages=site_packages,
        )
        == 0
    )
    row = _index(dist_root)[0]
    assert row["component_id"] == "nvidia-sidecars"
    assert row["component_name"] == "NVIDIA sidecars"
    assert row["version"] == "12.9"
    assert row["source_path"].startswith(
        "site-packages://nvidia_provider-12.9.dist-info/"
    )
    assert "/components/nvidia-sidecars/12.9/" in row["output_path"]


def test_collect_index_records_component_version_paths_modes_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages"
    _wheel(site_packages, "alpha", "1.2", {"licenses/LICENSE.txt": b"alpha"})
    alpha = _entry("alpha", 1)
    python_entry = _entry("python-runtime", 2)
    vc_entry = _entry("microsoft-vc-runtime", 3)
    llama_path = "_internal/llama/LICENSE.txt"
    llama_bytes = b"llama license"
    records = [
        _distribution_record(
            alpha,
            "alpha",
            name="Alpha",
            version="1.2",
            distribution_name="alpha",
        ),
        _special_record(
            kind="analysis-entry",
            key=python_entry.key,
            path=python_entry.destination,
            component_id="python-runtime",
            component_name="Python",
            provenance_class="python-runtime",
            license_mode="python-runtime-file",
            license_source="python://LICENSE.txt",
        ),
        _special_record(
            kind="analysis-entry",
            key=vc_entry.key,
            path=vc_entry.destination,
            component_id="microsoft-vc-runtime",
            component_name="Microsoft Visual C++ Runtime",
            provenance_class="reviewed-runtime",
            license_mode="reviewed-exception",
            license_source=None,
            review_id="VC-REDIST-EULA",
            review_reason="Redistributable runtime terms reviewed.",
        ),
        _special_record(
            kind="bundle-file",
            key=llama_path,
            path=llama_path,
            component_id="llama-cpp",
            component_name="llama.cpp",
            provenance_class="vendored-native",
            license_mode="bundle-file",
            license_source=llama_path,
        ),
    ]
    inventory, provenance = _inputs(
        tmp_path,
        records,
        [alpha, python_entry, vc_entry],
        bundle_files=[
            {
                "relative_path": llama_path,
                "size": len(llama_bytes),
                "sha256": hashlib.sha256(llama_bytes).hexdigest(),
                "inventory_keys": [],
            }
        ],
    )
    dist_root = _dist_root(tmp_path)
    llama = dist_root / llama_path
    llama.parent.mkdir(parents=True)
    llama.write_bytes(llama_bytes)
    python_license = tmp_path / "python-LICENSE.txt"
    python_license.write_bytes(b"python license")
    monkeypatch.setattr(collect_licenses, "PYTHON_LICENSE", python_license)

    assert (
        _collect(
            dist_root,
            inventory=inventory,
            provenance=provenance,
            policy=_policy(tmp_path),
            site_packages=site_packages,
        )
        == 0
    )

    rows = _index(dist_root)
    assert all(
        set(row)
        == {
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
        for row in rows
    )
    by_component: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_component.setdefault(str(row["component_id"]), []).append(row)

    alpha_row = by_component["alpha"][0]
    assert alpha_row["component_name"] == "Alpha"
    assert alpha_row["version"] == "1.2"
    assert alpha_row["license_mode"] == "distribution"
    assert alpha_row["source_path"].endswith(
        "/alpha-1.2.dist-info/licenses/LICENSE.txt"
    )

    python_row = by_component["python-runtime"][0]
    assert python_row["version"] == platform.python_version()
    assert python_row["source_path"] == "python://LICENSE.txt"
    assert python_row["output_path"].endswith(
        f"/python-runtime/{platform.python_version()}/LICENSE.txt"
    )

    llama_row = by_component["llama-cpp"][0]
    assert llama_row["version"] is None
    assert llama_row["output_path"] == llama_path
    assert llama_row["sha256"] == _sha256(llama)

    vc_row = by_component["microsoft-vc-runtime"][0]
    assert vc_row["version"] is None
    assert vc_row["source_path"] is None
    assert vc_row["output_path"] is None
    assert vc_row["sha256"] is None
    assert vc_row["review_id"] == "VC-REDIST-EULA"
    assert vc_row["review_reason"]

    for row in rows:
        output_path = row["output_path"]
        if output_path is not None:
            assert row["sha256"] == _sha256(dist_root / str(output_path))


@pytest.mark.parametrize(
    "line",
    [
        "dist:alpha",
        "file:alpha|../LICENSE",
        "file:alpha|C:/LICENSE",
        "file:alpha|LICENSE\nfile:alpha|LICENSE",
        "alias:alpha|beta\nalias:alpha|gamma",
    ],
)
def test_manifest_parser_rejects_legacy_traversal_and_duplicates(line: str) -> None:
    with pytest.raises(ValueError):
        collect_licenses._parse_manifest(line)


def test_notice_points_to_the_generated_bundle_index() -> None:
    notice = NOTICE.read_text(encoding="utf-8")
    assert (
        "The complete component and license-text index for each binary distribution is\n"
        "generated from its bundle inventory and ships in _internal/licenses/INDEX.txt."
        in notice
    )


def test_every_repo_license_override_is_claimed_by_the_manifest() -> None:
    claimed = {
        value for kind, _component, value in _manifest_entries() if kind == "file"
    }
    on_disk = {
        path.relative_to(LICENSES_DIR).as_posix()
        for path in LICENSES_DIR.rglob("*")
        if path.is_file()
    }
    assert claimed == on_disk


def test_manifest_file_entries_carry_real_license_text() -> None:
    markers = (
        "permission is hereby granted",
        "apache license",
        "gnu lesser general public",
        "gnu general public license",
        "redistribution and use",
    )
    for kind, component, value in _manifest_entries():
        if kind != "file":
            continue
        body = (LICENSES_DIR / value).read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        assert len(body) > 500, f"{value} for {component} is too short"
        assert any(marker in body for marker in markers)


def test_llama_cpp_license_ships_alongside_its_binaries() -> None:
    listed = [
        line.strip()
        for line in LLAMA_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "LICENSE.txt" in listed
    assert (ROOT / "third_party" / "llama.cpp" / "LICENSE.txt").is_file()


@pytest.mark.real
@pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            INVENTORY,
            PROVENANCE,
            BUNDLE / "_internal/licenses/INDEX.json",
        )
    ),
    reason="no generated provenance report and frozen license index",
)
def test_frozen_bundle_license_index_matches_generated_provenance_report() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    index = json.loads(
        (BUNDLE / "_internal/licenses/INDEX.json").read_text(encoding="utf-8")
    )
    required = {
        row["component_id"]
        for row in provenance["records"]
        if row["license_mode"] != "project"
    }
    actual = {row["component_id"] for row in index["entries"]}
    assert actual == required

    outputs: set[str] = set()
    for row in index["entries"]:
        relative = row["output_path"]
        if relative is None:
            assert row["license_mode"] == "reviewed-exception"
            continue
        assert relative.casefold() not in outputs
        outputs.add(relative.casefold())
        shipped = BUNDLE / relative
        assert shipped.is_file()
        assert row["sha256"] == _sha256(shipped)
