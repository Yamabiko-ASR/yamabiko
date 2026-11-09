from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_SOURCE = ROOT / "scripts" / "audit_license_closure.py"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _load_auditor() -> ModuleType:
    assert AUDITOR_SOURCE.is_file(), "independent license closure auditor is absent"
    spec = importlib.util.spec_from_file_location("task_b6_auditor", AUDITOR_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_tree_digest(root: Path) -> tuple[str, int]:
    rows: list[tuple[str, bytes]] = []
    for path in root.rglob("*"):
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), path.read_bytes()))
    digest = hashlib.sha256()
    for relative, data in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(data).encode("ascii"))
    return digest.hexdigest(), len(rows)


def _canonical_metadata_digest(site_packages: Path) -> str:
    rows: list[tuple[str, bytes]] = []
    for name in ("METADATA", "RECORD"):
        for path in site_packages.rglob(name):
            if path.parent.name.endswith(".dist-info"):
                rows.append((path.relative_to(site_packages).as_posix(), path.read_bytes()))
    digest = hashlib.sha256()
    for relative, data in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(data).encode("ascii"))
    return digest.hexdigest()


def _canonical_index_text(rows: list[dict[str, object]]) -> str:
    lines = [
        "Third-party component and license-text index for this Yamabiko bundle.",
        "Generated from the exact bundle inventory; do not edit by hand.",
        "",
    ]
    for row in rows:
        version = "" if row["version"] is None else f" {row['version']}"
        if row["output_path"] is None:
            lines.append(
                f"{row['component_id']}{version} [{row['license_mode']}]: "
                f"{row['review_id']} - {row['review_reason']}"
            )
        else:
            lines.append(
                f"{row['component_id']}{version} [{row['license_mode']}]: "
                f"{row['output_path']} (sha256:{row['sha256']})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class _BundleFixture:
    root: Path
    dist: Path
    site_packages: Path
    inventory: Path
    policy: Path
    overrides: Path
    index: Path
    metadata: Path
    record: Path
    wheel_files: dict[str, bytes] = field(default_factory=dict)
    inventory_rows: list[dict[str, str]] = field(default_factory=list)

    def write_inventory(self) -> None:
        self.inventory.write_text(
            json.dumps(
                {"schema_version": 1, "entries": self.inventory_rows},
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_record(self) -> None:
        metadata_relative = "demo-1.0.dist-info/METADATA"
        metadata_bytes = self.metadata.read_bytes()
        rows = [
            f"{relative},sha256={_record_digest(data)},{len(data)}"
            for relative, data in sorted(self.wheel_files.items())
        ]
        rows.extend(
            [
                f"{metadata_relative},sha256={_record_digest(metadata_bytes)},{len(metadata_bytes)}",
                "demo-1.0.dist-info/RECORD,,",
            ]
        )
        self.record.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")

    def add_owned_product(
        self,
        *,
        source_relative: str,
        destination: str,
        data: bytes,
        type_code: str = "DATA",
    ) -> None:
        source = self.site_packages / Path(source_relative)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        self.wheel_files[source_relative] = data

        output = self.dist / Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        logical = destination.removeprefix("_internal/")
        self.inventory_rows.append(
            {
                "collection": "binaries" if type_code == "BINARY" else "datas",
                "logical_name": logical,
                "source_path": f"site-packages://{source_relative}",
                "destination": destination,
                "type_code": type_code,
            }
        )
        self.write_inventory()
        self.write_record()

    def index_payload(self) -> dict[str, object]:
        return json.loads(self.index.read_text(encoding="utf-8"))

    def write_index(self, payload: dict[str, object]) -> None:
        self.index.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def audit(self, auditor: ModuleType) -> tuple[str, ...]:
        return auditor.audit_license_closure(
            dist_root=self.dist,
            inventory_path=self.inventory,
            license_index_path=self.index,
            provenance_policy_config_path=self.policy,
            license_override_policy_path=self.overrides,
            site_packages=self.site_packages,
        )


@pytest.fixture
def bundle(tmp_path: Path) -> _BundleFixture:
    dist = tmp_path / "dist" / "yamabiko"
    site_packages = tmp_path / "site-packages"
    build = tmp_path / "build"
    installer = tmp_path / "installer"
    dist.mkdir(parents=True)
    site_packages.mkdir()
    build.mkdir()
    installer.mkdir()

    dist_info = site_packages / "demo-1.0.dist-info"
    legal_source = dist_info / "licenses" / "LICENSE"
    legal_source.parent.mkdir(parents=True)
    license_bytes = b"Demo license text\n"
    legal_source.write_bytes(license_bytes)
    metadata = dist_info / "METADATA"
    metadata.write_text(
        "Metadata-Version: 2.4\n"
        "Name: demo\n"
        "Version: 1.0\n"
        "License-File: LICENSE\n",
        encoding="utf-8",
        newline="",
    )
    record = dist_info / "RECORD"

    inventory = build / "bundle-inventory.json"
    policy = installer / "bundle-provenance.toml"
    overrides = installer / "licenses-manifest.txt"
    policy.write_text("schema_version = 1\ncomponent = []\n", encoding="utf-8")
    overrides.write_text("# no reviewed overrides\n", encoding="utf-8")

    license_root = dist / "_internal" / "licenses"
    output_relative = "_internal/licenses/components/demo/1.0/licenses/LICENSE"
    output = dist / Path(output_relative)
    output.parent.mkdir(parents=True)
    output.write_bytes(license_bytes)
    index = license_root / "INDEX.json"
    payload = {
        "schema_version": 1,
        "entries": [
            {
                "component_id": "demo",
                "component_name": "demo",
                "version": "1.0",
                "provenance_class": "distribution",
                "license_mode": "distribution",
                "source_path": "site-packages://demo-1.0.dist-info/licenses/LICENSE",
                "output_path": output_relative,
                "sha256": _sha256(license_bytes),
                "review_id": None,
                "review_reason": None,
            }
        ],
    }
    index.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (license_root / "INDEX.txt").write_text(
        _canonical_index_text(payload["entries"]),
        encoding="utf-8",
        newline="",
    )

    fixture = _BundleFixture(
        root=tmp_path,
        dist=dist,
        site_packages=site_packages,
        inventory=inventory,
        policy=policy,
        overrides=overrides,
        index=index,
        metadata=metadata,
        record=record,
        wheel_files={
            "demo-1.0.dist-info/licenses/LICENSE": license_bytes,
        },
    )
    fixture.add_owned_product(
        source_relative="demo/data.txt",
        destination="_internal/demo/data.txt",
        data=b"owned product data\n",
    )
    return fixture


def test_auditor_source_does_not_import_collector_or_provenance_resolver() -> None:
    assert AUDITOR_SOURCE.is_file(), "independent license closure auditor is absent"
    tree = ast.parse(AUDITOR_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value.casefold())

    prohibited = {
        "collect_licenses",
        "scripts.collect_licenses",
        "bundle_provenance",
        "scripts.bundle_provenance",
    }
    assert imported.isdisjoint(prohibited)
    script_imports = {name for name in imported if name.startswith("scripts.")}
    assert script_imports <= {"scripts.audit_bundle_dlls"}
    assert not any("bundle-provenance.json" in value for value in strings)


def test_audit_detects_distribution_removed_from_license_index(
    bundle: _BundleFixture,
) -> None:
    auditor = _load_auditor()
    payload = bundle.index_payload()
    payload["entries"] = []
    bundle.write_index(payload)

    errors = bundle.audit(auditor)

    assert "missing license index component: demo" in errors


def test_audit_detects_unowned_final_native_binary(
    bundle: _BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = _load_auditor()
    orphan = bundle.dist / "_internal" / "orphan.dll"
    orphan.write_bytes(b"not owned by inventory or policy")
    monkeypatch.setattr(auditor, "parse_imports", lambda _path: [])

    errors = bundle.audit(auditor)

    assert "unowned final native binary: _internal/orphan.dll" in errors


def test_audit_rejects_bundle_walk_enumeration_error(
    bundle: _BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A traversal failure cannot silently remove files from closure."""
    auditor = _load_auditor()

    def failing_walk(_root, *, topdown, followlinks, onerror=None):
        assert topdown is True and followlinks is False
        if onerror is not None:
            onerror(PermissionError("fixture subtree cannot be enumerated"))
        return iter(())

    monkeypatch.setattr(auditor.os, "walk", failing_walk)

    errors = bundle.audit(auditor)

    assert any("cannot enumerate bundle tree" in error for error in errors)


def test_auditor_stable_hash_rejects_reparse_file_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Input hashing itself must never follow a file-level reparse."""
    auditor = _load_auditor()
    target = tmp_path / "candidate.bin"
    target.write_bytes(b"candidate")
    original_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = getattr(
                auditor.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
            )

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def marked_lstat(path):
        result = original_lstat(path)
        return ReparseStat(result) if Path(path) == target else result

    monkeypatch.setattr(Path, "lstat", marked_lstat)

    with pytest.raises(ValueError, match="regular non-reparse"):
        auditor._hash_file_stably(target)


def test_audit_rejects_nested_legal_source_scan_error(
    bundle: _BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable legal-source subtree cannot shrink expected closure."""
    auditor = _load_auditor()
    original_walk = auditor.os.walk

    def fail_dist_info_walk(root, *, topdown, followlinks, onerror=None):
        if Path(root).name.casefold().endswith(".dist-info"):
            if onerror is not None:
                onerror(PermissionError("fixture legal subtree is unreadable"))
            return iter(())
        return original_walk(
            root,
            topdown=topdown,
            followlinks=followlinks,
            onerror=onerror,
        )

    monkeypatch.setattr(auditor.os, "walk", fail_dist_info_walk)

    errors = bundle.audit(auditor)

    assert any("cannot enumerate legal files" in error for error in errors)


def test_audit_detects_non_system_pe_import_without_present_owner(
    bundle: _BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = _load_auditor()
    bundle.add_owned_product(
        source_relative="demo/native.dll",
        destination="_internal/demo.dll",
        data=b"synthetic PE fixture",
        type_code="BINARY",
    )
    monkeypatch.setattr(
        auditor,
        "parse_imports",
        lambda path: ["missing-vendor.dll"] if path.name == "demo.dll" else [],
    )

    errors = bundle.audit(auditor)

    assert (
        "PE import missing-vendor.dll from _internal/demo.dll has no present owner"
        in errors
    )


def test_audit_detects_changed_shipped_license_hash(bundle: _BundleFixture) -> None:
    auditor = _load_auditor()
    output = bundle.dist / "_internal/licenses/components/demo/1.0/licenses/LICENSE"
    output.write_bytes(b"changed after collection\n")

    errors = bundle.audit(auditor)

    assert (
        "indexed license output hash mismatch: "
        "_internal/licenses/components/demo/1.0/licenses/LICENSE"
    ) in errors


def test_audit_detects_changed_human_readable_license_index(
    bundle: _BundleFixture,
) -> None:
    """INDEX.txt is derived evidence, not an existence-only decoration."""
    auditor = _load_auditor()
    text_index = bundle.dist / "_internal" / "licenses" / "INDEX.txt"
    text_index.write_text("corrupted human-readable index\n", encoding="utf-8")

    errors = bundle.audit(auditor)

    assert "human-readable license index does not match derived closure" in errors


def test_audit_rejects_noncanonical_human_index_casing(
    bundle: _BundleFixture,
) -> None:
    """Case-folded presence cannot bypass exact INDEX.txt validation."""
    auditor = _load_auditor()
    text_index = bundle.dist / "_internal" / "licenses" / "INDEX.txt"
    temporary = text_index.with_name("INDEX.rename-tmp")
    lowercase = text_index.with_name("index.txt")
    text_index.replace(temporary)
    temporary.replace(lowercase)

    errors = bundle.audit(auditor)

    assert "human-readable license index path has non-canonical casing" in errors


def test_audit_accepts_windows_crlf_human_index(bundle: _BundleFixture) -> None:
    """Collector text-mode output is semantically canonical with CRLF on Windows."""
    auditor = _load_auditor()
    text_index = bundle.dist / "_internal" / "licenses" / "INDEX.txt"
    canonical = text_index.read_text(encoding="utf-8")
    text_index.write_bytes(canonical.replace("\n", "\r\n").encode("utf-8"))

    errors = bundle.audit(auditor)

    assert errors == ()


def test_audit_detects_extra_stale_index_row(bundle: _BundleFixture) -> None:
    auditor = _load_auditor()
    payload = bundle.index_payload()
    payload["entries"].append(
        {
            "component_id": "ghost-runtime",
            "component_name": "Ghost runtime",
            "version": None,
            "provenance_class": "reviewed-runtime",
            "license_mode": "reviewed-exception",
            "source_path": None,
            "output_path": None,
            "sha256": None,
            "review_id": "GHOST-REVIEW",
            "review_reason": "Synthetic stale index row",
        }
    )
    bundle.write_index(payload)

    errors = bundle.audit(auditor)

    assert "extra license index component: ghost-runtime" in errors


def test_audit_detects_same_component_license_descriptor_drift(
    bundle: _BundleFixture,
) -> None:
    """Membership alone is insufficient; every derived row field is bound."""
    auditor = _load_auditor()
    payload = bundle.index_payload()
    payload["entries"][0]["version"] = "9.9"
    bundle.write_index(payload)

    errors = bundle.audit(auditor)

    assert any("license index row mismatch: demo" in error for error in errors)


def test_audit_detects_extra_row_for_required_component(
    bundle: _BundleFixture,
) -> None:
    """Extra-row rejection is independent of extra-component rejection."""
    auditor = _load_auditor()
    extra_bytes = b"unapproved extra legal row\n"
    extra_output = "_internal/licenses/components/demo/1.0/NOTICE"
    output = bundle.dist / Path(extra_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(extra_bytes)
    payload = bundle.index_payload()
    payload["entries"].append(
        {
            "component_id": "demo",
            "component_name": "demo",
            "version": "1.0",
            "provenance_class": "distribution",
            "license_mode": "distribution",
            "source_path": "site-packages://demo-1.0.dist-info/NOTICE",
            "output_path": extra_output,
            "sha256": _sha256(extra_bytes),
            "review_id": None,
            "review_reason": None,
        }
    )
    bundle.write_index(payload)

    errors = bundle.audit(auditor)

    assert any(
        error == f"extra license index row: demo {extra_output.casefold()}"
        for error in errors
    )


def test_audit_detects_missing_row_for_still_indexed_component(
    bundle: _BundleFixture,
) -> None:
    """A component stays present while one newly derived legal row is absent."""
    auditor = _load_auditor()
    relative = "demo-1.0.dist-info/vendor/NOTICE"
    notice = bundle.site_packages / Path(relative)
    notice.parent.mkdir(parents=True)
    notice_bytes = b"additional wheel notice\n"
    notice.write_bytes(notice_bytes)
    bundle.wheel_files[relative] = notice_bytes
    bundle.write_record()

    errors = bundle.audit(auditor)

    expected_output = (
        "_internal/licenses/components/demo/1.0/vendor/notice"
    )
    assert any(
        error == f"missing license index row: demo {expected_output}"
        for error in errors
    )


def test_audit_independently_derives_reviewed_alias_override(
    bundle: _BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A policy-owned sidecar alias reuses reviewed bytes but keeps its owner."""
    auditor = _load_auditor()
    reviewed_root = bundle.root / "reviewed-licenses"
    reviewed_root.mkdir()
    reviewed_bytes = b"Reviewed demo license override\n"
    (reviewed_root / "demo-reviewed.txt").write_bytes(reviewed_bytes)
    monkeypatch.setattr(auditor, "REPO_LICENSES", reviewed_root)
    bundle.overrides.write_text(
        "file:demo|demo-reviewed.txt\n"
        "alias:demo-sidecar|demo\n",
        encoding="utf-8",
    )

    sidecar_bytes = b"reviewed sidecar payload\n"
    sidecar_destination = "_internal/demo-sidecar.dat"
    sidecar_output = bundle.dist / Path(sidecar_destination)
    sidecar_output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_output.write_bytes(sidecar_bytes)
    bundle.inventory_rows.append(
        {
            "collection": "datas",
            "logical_name": "demo-sidecar.dat",
            "source_path": (
                f"external://sha256/{_sha256(sidecar_bytes)}/demo-sidecar.dat"
            ),
            "destination": sidecar_destination,
            "type_code": "DATA",
        }
    )
    bundle.write_inventory()
    bundle.policy.write_text(
        "schema_version = 1\n\n"
        "[[component]]\n"
        'id = "demo-sidecar"\n'
        'name = "Demo sidecar"\n'
        'class = "sidecar"\n'
        "priority = 100\n"
        'globs = ["_internal/demo-sidecar.dat"]\n'
        'license_mode = "distribution"\n'
        'license_source = "demo"\n',
        encoding="utf-8",
    )

    old_output = (
        bundle.dist
        / "_internal"
        / "licenses"
        / "components"
        / "demo"
        / "1.0"
        / "licenses"
        / "LICENSE"
    )
    old_output.unlink()
    reviewed_outputs = {
        "demo": (
            "demo",
            "distribution",
            "_internal/licenses/components/demo/override/demo-reviewed.txt",
        ),
        "demo-sidecar": (
            "Demo sidecar",
            "sidecar",
            "_internal/licenses/components/demo-sidecar/override/demo-reviewed.txt",
        ),
    }
    entries = []
    for component_id, (component_name, provenance_class, output_path) in sorted(
        reviewed_outputs.items()
    ):
        output = bundle.dist / Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(reviewed_bytes)
        entries.append(
            {
                "component_id": component_id,
                "component_name": component_name,
                "version": "1.0",
                "provenance_class": provenance_class,
                "license_mode": "distribution",
                "source_path": "repo://licenses/demo-reviewed.txt",
                "output_path": output_path,
                "sha256": _sha256(reviewed_bytes),
                "review_id": None,
                "review_reason": None,
            }
        )
    bundle.write_index({"schema_version": 1, "entries": entries})
    (bundle.dist / "_internal" / "licenses" / "INDEX.txt").write_text(
        _canonical_index_text(entries), encoding="utf-8", newline=""
    )

    errors = bundle.audit(auditor)

    assert errors == ()


def test_audit_ignores_license_payload_when_enumerating_product_files(
    bundle: _BundleFixture,
) -> None:
    auditor = _load_auditor()

    errors = bundle.audit(auditor)

    assert errors == ()


def test_audit_report_binds_exact_bundle_tree_and_input_hashes(
    bundle: _BundleFixture,
) -> None:
    auditor = _load_auditor()
    output = bundle.root / "license-closure-audit.json"

    result = auditor.main(
        [
            "--dist",
            str(bundle.dist),
            "--inventory",
            str(bundle.inventory),
            "--license-index",
            str(bundle.index),
            "--provenance-policy-config",
            str(bundle.policy),
            "--license-override-policy",
            str(bundle.overrides),
            "--site-packages",
            str(bundle.site_packages),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report) == {
        "schema_version",
        "status",
        "audited_utc",
        "bundle_tree_sha256",
        "bundle_file_count",
        "inventory_sha256",
        "license_index_sha256",
        "provenance_policy_sha256",
        "license_override_policy_sha256",
        "distribution_metadata_sha256",
        "errors",
    }
    tree_digest, file_count = _canonical_tree_digest(bundle.dist)
    assert report == {
        "schema_version": 1,
        "status": "pass",
        "audited_utc": report["audited_utc"],
        "bundle_tree_sha256": tree_digest,
        "bundle_file_count": file_count,
        "inventory_sha256": _sha256(bundle.inventory.read_bytes()),
        "license_index_sha256": _sha256(bundle.index.read_bytes()),
        "provenance_policy_sha256": _sha256(bundle.policy.read_bytes()),
        "license_override_policy_sha256": _sha256(bundle.overrides.read_bytes()),
        "distribution_metadata_sha256": _canonical_metadata_digest(
            bundle.site_packages
        ),
        "errors": [],
    }
    assert report["audited_utc"].endswith("Z")


def test_audit_cli_reports_malformed_fingerprintable_index(
    bundle: _BundleFixture,
) -> None:
    """Valid JSON with a wrong field type must produce a red report, not crash."""
    auditor = _load_auditor()
    payload = bundle.index_payload()
    payload["entries"][0]["license_mode"] = []
    bundle.write_index(payload)
    output = bundle.root / "malformed-license-closure-audit.json"

    result = auditor.main(
        [
            "--dist",
            str(bundle.dist),
            "--inventory",
            str(bundle.inventory),
            "--license-index",
            str(bundle.index),
            "--provenance-policy-config",
            str(bundle.policy),
            "--license-override-policy",
            str(bundle.overrides),
            "--site-packages",
            str(bundle.site_packages),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["errors"]
    assert any("license_mode" in error for error in report["errors"])


def test_audit_report_rejects_semantic_snapshot_aba_swap(
    bundle: _BundleFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report must bind the bytes inspected by the semantic audit itself."""
    auditor = _load_auditor()
    license_output = (
        bundle.dist
        / "_internal"
        / "licenses"
        / "components"
        / "demo"
        / "1.0"
        / "licenses"
        / "LICENSE"
    )
    valid_bytes = license_output.read_bytes()
    tampered_bytes = b"tampered bytes outside semantic window\n"
    license_output.write_bytes(tampered_bytes)
    output = bundle.root / "aba-license-closure-audit.json"
    original_fingerprints = auditor._audit_fingerprints
    calls = 0

    def swap_around_semantic_audit(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            license_output.write_bytes(tampered_bytes)
        result = original_fingerprints(**kwargs)
        if calls == 1:
            license_output.write_bytes(valid_bytes)
        return result

    monkeypatch.setattr(
        auditor, "_audit_fingerprints", swap_around_semantic_audit
    )

    result = auditor.main(
        [
            "--dist",
            str(bundle.dist),
            "--inventory",
            str(bundle.inventory),
            "--license-index",
            str(bundle.index),
            "--provenance-policy-config",
            str(bundle.policy),
            "--license-override-policy",
            str(bundle.overrides),
            "--site-packages",
            str(bundle.site_packages),
            "--output",
            str(output),
        ]
    )

    assert calls == 2
    assert result == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert any("semantic audit snapshot" in error for error in report["errors"])


def test_auditor_policy_requires_exact_python_license_source(tmp_path: Path) -> None:
    """Python closure is bound specifically to the runtime LICENSE.txt."""
    auditor = _load_auditor()
    policy = tmp_path / "policy.toml"
    policy.write_text(
        "schema_version = 1\n\n"
        "[[component]]\n"
        'id = "python-runtime"\n'
        'name = "Python"\n'
        'class = "python-runtime"\n'
        "priority = 200\n"
        'globs = ["_internal/python3*.dll"]\n'
        'license_mode = "python-runtime-file"\n'
        'license_source = "python://OTHER.txt"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="python://LICENSE.txt"):
        auditor._load_policy(policy)
