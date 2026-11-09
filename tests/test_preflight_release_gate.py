"""Unit tests for preflight release-gate honesty.

Preflight Section 8 (pytest -m real) and Section 9 (bundle DLL audit)
previously SKIP-to-green when no bundle / GPU / warmed cache is present, yet
the final summary still printed "Safe to release" -- a release gate that can
pass without ever exercising the frozen bundle. These tests pin the corrected
behavior:

  * a ``--release`` run FAILs (does not SKIP) when the bundle is absent;
  * the real-tier is FAIL in ``--release`` mode when 0 tests actually ran
    (all skipped, i.e., a vacuous pass);
  * a dev run with skipped sections refuses to claim "Safe to release".

Each test exercises a pure classifier so the subprocess-running sections stay
thin wrappers around tested logic.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import preflight
from scripts.preflight import Section


_CLOSURE_REPORT_FIELDS = {
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_test_bundle_digest(root: Path) -> tuple[str, int]:
    """Independent fixture implementation of Task B6's tree contract."""
    rows = []
    for path in root.rglob("*"):
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), path))
    digest = hashlib.sha256()
    for relative, path in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest(), len(rows)


def _canonical_test_metadata_digest(site_packages: Path) -> str:
    rows = []
    for dist_info in site_packages.glob("*.dist-info"):
        for name in ("METADATA", "RECORD"):
            path = dist_info / name
            if path.is_file():
                rows.append((path.relative_to(site_packages).as_posix(), path))
    digest = hashlib.sha256()
    for relative, path in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _seed_closure_audit_surface(root: Path, venv: str = ".venv") -> tuple[Path, dict]:
    dist = root / "dist" / "yamabiko"
    license_index = dist / "_internal" / "licenses" / "INDEX.json"
    license_index.parent.mkdir(parents=True)
    license_index.write_text(
        '{"entries": [], "schema_version": 1}\n', encoding="utf-8"
    )
    (dist / "yamabiko.exe").write_bytes(b"MZ" + b"release-candidate")

    inventory = root / "build" / "yamabiko" / "bundle-inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(
        '{"entries": [], "schema_version": 1}\n', encoding="utf-8"
    )
    provenance_policy = root / "installer" / "bundle-provenance.toml"
    provenance_policy.parent.mkdir(parents=True)
    provenance_policy.write_text("schema_version = 1\n", encoding="utf-8")
    override_policy = root / "installer" / "licenses-manifest.txt"
    override_policy.write_text("# no fixture overrides\n", encoding="utf-8")

    dist_info = root / venv / "Lib" / "site-packages" / "alpha-1.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: alpha\nVersion: 1\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(
        "alpha-1.dist-info/METADATA,,\nalpha-1.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    site_packages = dist_info.parent

    bundle_digest, bundle_count = _canonical_test_bundle_digest(dist)
    report = {
        "schema_version": 1,
        "status": "pass",
        "audited_utc": "2026-07-15T12:00:00Z",
        "bundle_tree_sha256": bundle_digest,
        "bundle_file_count": bundle_count,
        "inventory_sha256": _sha256(inventory),
        "license_index_sha256": _sha256(license_index),
        "provenance_policy_sha256": _sha256(provenance_policy),
        "license_override_policy_sha256": _sha256(override_policy),
        "distribution_metadata_sha256": _canonical_test_metadata_digest(site_packages),
        "errors": [],
    }
    report_path = root / "build" / "yamabiko" / "license-closure-audit.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    assert set(report) == _CLOSURE_REPORT_FIELDS
    return report_path, report


# ---------------------------------------------------------------------------
# _parse_pytest_counts -- parse passed/failed/skipped from a summary tail line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tail,expected", [
    ("= 12 passed, 3 skipped in 45.2s =", {"passed": 12, "failed": 0, "skipped": 3}),
    ("= 5 passed in 8.0s =", {"passed": 5, "failed": 0, "skipped": 0}),
    ("= 2 failed, 9 passed, 26 skipped, 571 deselected in 82s =",
     {"passed": 9, "failed": 2, "skipped": 26}),
    ("= 37 skipped in 2.0s =", {"passed": 0, "failed": 0, "skipped": 37}),
    ("", {"passed": 0, "failed": 0, "skipped": 0}),
])
def test_parse_pytest_counts(tail, expected):
    assert preflight._parse_pytest_counts(tail) == expected


# ---------------------------------------------------------------------------
# _classify_real_tier -- (returncode, stdout, release) -> (status, message)
# ---------------------------------------------------------------------------

def test_real_tier_all_skipped_fails_under_release():
    """0 tests ran (all skipped) must FAIL a --release run, not vacuously
    PASS. Seeded defect: returning PASS here makes this go red."""
    status, msg = preflight._classify_real_tier(0, "= 37 skipped in 2s =", release=True)
    assert status == "FAIL", (status, msg)
    assert "0" in msg


def test_real_tier_all_skipped_is_skip_not_pass_in_dev_mode():
    """Non-release: all-skipped is an honest SKIP (invisible to the gate),
    never a green PASS. Seeded defect: returning PASS makes this red."""
    status, _ = preflight._classify_real_tier(0, "= 37 skipped in 2s =", release=False)
    assert status == "SKIP"


def test_real_tier_with_passes_reports_counts():
    """A real run that actually executed tests PASSes and surfaces the
    passed/skipped breakdown (so an all-skipped run looks different)."""
    status, msg = preflight._classify_real_tier(
        0, "= 12 passed, 3 skipped in 45s =", release=True,
    )
    assert status == "PASS"
    assert "12 passed" in msg and "3 skipped" in msg


def test_real_tier_nonzero_rc_fails():
    status, msg = preflight._classify_real_tier(
        1, "= 2 failed, 9 passed in 8s =", release=False,
    )
    assert status == "FAIL"
    assert "FAILED" in msg


# ---------------------------------------------------------------------------
# _classify_downloads_tqdm_usage -- section 2's call-site guard
#
# The pre-1.1.0 guard was "downloads.py must NOT pass tqdm_class" (the
# original TypeError workaround). huggingface_hub>=1.1.0 made tqdm_class
# the deliberate design, so the guard inverted: the gate must now FAIL
# when the call site STOPS passing tqdm_class (wizard progress silently
# disconnects). The stale inverted guard FAILed every healthy tree.
# ---------------------------------------------------------------------------

def test_downloads_tqdm_guard_passes_when_call_site_passes_tqdm_class():
    """The healthy-tree shape: downloads.py wires tqdm_class into the
    hf_hub_download call. Seeded defect: restore the old inverted
    condition (FAIL when tqdm_class present) and this goes red."""
    src = (
        'kwargs["tqdm_class"] = _make_progress_tqdm(progress_callback)\n'
        "hf_hub_download(**kwargs)\n"
    )
    status, msg = preflight._classify_downloads_tqdm_usage(src)
    assert status == "PASS", (status, msg)


def test_downloads_tqdm_guard_fails_when_tqdm_class_dropped():
    """The live bug class: a refactor that stops passing tqdm_class
    kills wizard byte-progress with no other test going red. Seeded
    defect: make the classifier return PASS unconditionally."""
    src = "hf_hub_download(repo_id=repo, filename=name)\n"
    status, msg = preflight._classify_downloads_tqdm_usage(src)
    assert status == "FAIL", (status, msg)
    assert "tqdm_class" in msg


def test_downloads_tqdm_guard_fails_when_hf_hub_download_call_gone():
    """If downloads.py no longer calls hf_hub_download at all, the
    guard has lost its target and must say so rather than pass
    vacuously."""
    src = "some_other_downloader()\n"
    status, msg = preflight._classify_downloads_tqdm_usage(src)
    assert status == "FAIL", (status, msg)


def test_section_2_passes_on_the_real_repo_tree():
    """The release gate's section 2 must exit green on a healthy HEAD.
    Runs against the REAL downloads.py + the REAL installed
    huggingface_hub (no network). Seeded defect: re-invert the substring
    guard in check_hf_download_signature."""
    sec = preflight.check_hf_download_signature()
    assert not sec.failed(), sec.lines


# ---------------------------------------------------------------------------
# _classify_bundle_audit_missing -- the no-bundle decision
# ---------------------------------------------------------------------------

def test_bundle_missing_fails_under_release():
    """A --release run must FAIL when there is no bundle to audit -- it
    cannot be release-grade without one. Seeded defect: returning SKIP
    makes this red."""
    status, msg = preflight._classify_bundle_audit_missing(release=True)
    assert status == "FAIL"
    assert "release" in msg.lower()


def test_bundle_missing_skips_in_dev_mode():
    status, msg = preflight._classify_bundle_audit_missing(release=False)
    assert status == "SKIP"
    assert "build.ps1" in msg


# ---------------------------------------------------------------------------
# check_bundle_dll_audit -- rc=1 with an unparseable stdout surfaces stderr
# ---------------------------------------------------------------------------

def test_bundle_audit_rc1_empty_stdout_reports_stderr_reason(monkeypatch, tmp_path):
    """audit_bundle_dlls.py's gutted-bundle FAILs (empty dist, exe-less
    dist) print their reason to STDERR and nothing to stdout, so the
    rc=1 MISSING parse finds no names. The section line must carry the
    stderr reason -- not the useless "could not parse audit output"
    shrug that sends the operator to rerun the audit blind.

    Seeded defect: make the empty-parse branch ignore proc.stderr
    (restore the bare rerun-manually preview) -> both content asserts
    go red."""
    (tmp_path / "dist" / "yamabiko").mkdir(parents=True)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    stderr_reason = (
        "FAIL: bundle at dist/yamabiko contains no native files "
        "(.dll/.pyd/.exe) -- empty or gutted dist; nothing to audit"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr=stderr_reason + "\n",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    sec = preflight.check_bundle_dll_audit()
    assert sec.failed(), sec.lines
    blob = " ".join(msg for _, msg in sec.lines)
    assert "no native files" in blob, sec.lines
    assert "could not parse" not in blob, sec.lines


def test_bundle_audit_rc1_no_parse_and_no_stderr_still_says_rerun(monkeypatch, tmp_path):
    """When BOTH streams are useless (rc=1, no MISSING block, empty
    stderr) the section must still FAIL and keep pointing at a manual
    rerun rather than fabricating a reason. Seeded defect: return PASS
    (or drop the rerun pointer) on the doubly-empty branch."""
    (tmp_path / "dist" / "yamabiko").mkdir(parents=True)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    sec = preflight.check_bundle_dll_audit()
    assert sec.failed(), sec.lines
    blob = " ".join(msg for _, msg in sec.lines)
    assert "could not parse" in blob, sec.lines


# ---------------------------------------------------------------------------
# Section.skipped -- "this section did nothing meaningful"
# ---------------------------------------------------------------------------

def test_section_skipped_true_for_skip_only():
    s = Section(9, "x")
    s.add("SKIP", "no bundle")
    assert s.skipped() is True


def test_section_skipped_false_when_passed():
    s = Section(9, "x")
    s.add("PASS", "ok")
    assert s.skipped() is False


def test_section_skipped_false_when_failed():
    s = Section(9, "x")
    s.add("FAIL", "bad")
    assert s.skipped() is False


# ---------------------------------------------------------------------------
# _final_report -- (sections, release) -> (exit_code, lines)
# ---------------------------------------------------------------------------

def _sec(num, status, msg="m"):
    s = Section(num, f"section-{num}")
    s.add(status, msg)
    return s


def test_final_report_fails_when_any_section_failed():
    code, lines = preflight._final_report(
        [_sec(1, "PASS"), _sec(9, "FAIL", "boom")], release=False,
    )
    assert code == 1
    assert any("FAILED" in ln for ln in lines)


def test_final_report_refuses_passed_banner_when_sections_skipped_in_dev():
    """A dev run with skipped bundle sections must NOT print the
    all-sections-passed banner -- skips don't fail the gate, but the
    run is not release-grade and must say so + point at --release."""
    code, lines = preflight._final_report(
        [_sec(1, "PASS"), _sec(9, "SKIP")], release=False,
    )
    assert code == 0
    assert not any("All preflight sections passed" in ln for ln in lines)
    blob = " ".join(lines).lower()
    assert "release-grade" in blob or "--release" in blob


def test_final_report_all_passed_names_frozen_phases_as_separate_gates():
    """The green banner must claim only what preflight verified
    (bundle-presence + static checks) and point at the frozen E2E phases
    in docs/RELEASE.md as separate required steps -- never a blanket
    'Safe to release' (preflight cannot run the frozen E2E: section 8's
    300 s cap can't fit it and the E2E self-skips inside preflight).

    Seeded defect: restore the old 'All checks passed. Safe to
    release.' copy -> both content asserts go red."""
    code, lines = preflight._final_report(
        [_sec(1, "PASS"), _sec(9, "PASS")], release=False,
    )
    assert code == 0
    assert any("All preflight sections passed" in ln for ln in lines)
    blob = " ".join(lines)
    assert "frozen-external" in blob and "frozen-owned" in blob, lines
    assert "docs/RELEASE.md" in blob, lines
    assert not any("Safe to release" in ln for ln in lines)


# ---------------------------------------------------------------------------
# _stamp_grade -- the .preflight-stamp run-grade marker
# (a bare timestamp makes a skipped-bundle dev run indistinguishable from a
# release-grade run when auditing why a bad build shipped)
# ---------------------------------------------------------------------------

def test_stamp_grade_names_skipped_sections_in_dev_run():
    """A dev run with bundle-dependent sections skipped must be visibly
    NOT release-grade in the stamp. Seeded defect: return 'full'
    unconditionally."""
    grade = preflight._stamp_grade([_sec(1, "PASS"), _sec(9, "SKIP"),
                                    _sec(14, "SKIP")], release=False)
    assert grade == "dev-skipped:9,14"


def test_stamp_grade_release_run_is_release_preflight_not_release_grade():
    """The --release stamp token records that preflight's bundle-presence
    + static checks ran; it must NOT read 'release-grade' -- the frozen
    E2E gate (docs/RELEASE.md) is a separate required step preflight
    cannot attest to. Seeded defect: restore the 'release-grade' token
    -> red."""
    grade = preflight._stamp_grade([_sec(1, "PASS")], release=True)
    assert grade == "release-preflight"


def test_stamp_grade_dev_run_with_no_skips_is_full():
    grade = preflight._stamp_grade([_sec(1, "PASS"), _sec(9, "PASS")],
                                   release=False)
    assert grade == "full"


# ---------------------------------------------------------------------------
# check_bundle_license_closure -- independently bind release evidence to the
# current frozen tree and every stable audit input.
# ---------------------------------------------------------------------------

def test_license_closure_missing_report_fails_release(monkeypatch, tmp_path):
    """A release cannot inherit green license evidence that was never produced.

    Seeded defect: classify a missing report as SKIP under ``--release``.
    """
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    assert "missing" in " ".join(message for _, message in sec.lines).lower()


def test_license_closure_missing_report_is_honest_skip_for_contributor(
    monkeypatch, tmp_path,
):
    """Contributor preflight may lack a bundle, but must not call that PASS.

    Seeded defect: return PASS when the generated closure report is absent.
    """
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=False)

    assert sec.skipped(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "not release" in blob or "--release" in blob


def test_license_closure_red_report_fails_release(monkeypatch, tmp_path):
    """A report that records closure errors must remain a release failure.

    Seeded defect: trust the report's presence without checking ``status``.
    """
    report_path, report = _seed_closure_audit_surface(tmp_path)
    report["status"] = "fail"
    report["errors"] = ["component alpha has no indexed legal text"]
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "alpha" in blob and "no indexed legal text" in blob


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json",
        json.dumps({"schema_version": 1, "status": "pass"}),
        json.dumps(
            {
                **{field: None for field in sorted(_CLOSURE_REPORT_FIELDS)},
                "schema_version": 1,
                "status": "pass",
                "errors": [],
                "unexpected": "field",
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "status": [],
                "audited_utc": "2026-07-15T12:00:00Z",
                "bundle_tree_sha256": "0" * 64,
                "bundle_file_count": 0,
                "inventory_sha256": "0" * 64,
                "license_index_sha256": "0" * 64,
                "provenance_policy_sha256": "0" * 64,
                "license_override_policy_sha256": "0" * 64,
                "distribution_metadata_sha256": "0" * 64,
                "errors": [],
            }
        ),
    ],
)
def test_license_closure_malformed_report_fails_release(
    monkeypatch, tmp_path, payload,
):
    """Malformed evidence is not weaker evidence; it is no evidence.

    Seeded defect: catch a JSON/schema error and downgrade it to SKIP.
    """
    report_path = tmp_path / "build" / "yamabiko" / "license-closure-audit.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    assert "malformed" in " ".join(message for _, message in sec.lines).lower()


def test_license_closure_nominal_pass_fails_after_bundle_tree_changes(
    monkeypatch, tmp_path,
):
    """A nominal pass is stale after any shipped byte changes.

    Seeded defect: trust ``status: pass`` without re-hashing the current tree.
    """
    _seed_closure_audit_surface(tmp_path)
    (tmp_path / "dist" / "yamabiko" / "yamabiko.exe").write_bytes(
        b"MZchanged-after-audit"
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "stale" in blob and "bundle_tree_sha256" in blob


def test_license_closure_rejects_bundle_walk_enumeration_error(
    monkeypatch, tmp_path,
):
    """An unreadable subtree cannot disappear from the candidate digest.

    Seeded defect: call ``os.walk`` without an ``onerror`` callback, which
    silently omits the failed subtree and hashes a partial tree.
    """
    dist = tmp_path / "dist" / "yamabiko"
    dist.mkdir(parents=True)

    def failing_walk(_root, *, topdown, followlinks, onerror=None):
        assert topdown is True and followlinks is False
        if onerror is not None:
            onerror(PermissionError("fixture subtree cannot be enumerated"))
        return iter(())

    monkeypatch.setattr(preflight.os, "walk", failing_walk)

    with pytest.raises(ValueError, match="enumerate bundle tree"):
        preflight._bundle_tree_fingerprint(dist)


def test_license_closure_stable_hash_rejects_reparse_file_stat(
    monkeypatch, tmp_path,
):
    """The hash helper itself must fail closed if a path becomes a reparse."""
    target = tmp_path / "candidate.bin"
    target.write_bytes(b"candidate")
    original_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = getattr(
                preflight.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
            )

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def marked_lstat(path):
        result = original_lstat(path)
        return ReparseStat(result) if Path(path) == target else result

    monkeypatch.setattr(Path, "lstat", marked_lstat)

    with pytest.raises(ValueError, match="regular non-reparse"):
        preflight._sha256_file(target)


def test_license_closure_rejects_reparse_bundle_directory(
    monkeypatch, tmp_path,
):
    """A junction-like bundle directory cannot redirect tree traversal."""
    _seed_closure_audit_surface(tmp_path)
    redirected = tmp_path / "dist" / "yamabiko" / "_internal"
    original_check = preflight._is_link_or_reparse

    def mark_directory(path):
        return Path(path) == redirected or original_check(path)

    monkeypatch.setattr(preflight, "_is_link_or_reparse", mark_directory)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    assert "reparse" in " ".join(message for _, message in sec.lines).lower()


def test_license_closure_rejects_case_variant_incomplete_metadata_directory(
    monkeypatch, tmp_path,
):
    """The metadata universe is case-insensitive and every member is complete.

    Seeded defect: match only lowercase ``.dist-info`` names or silently omit
    a discovered distribution whose METADATA/RECORD pair is incomplete.
    """
    _seed_closure_audit_surface(tmp_path)
    ghost = (
        tmp_path
        / ".venv"
        / "Lib"
        / "site-packages"
        / "ghost-1.DIST-INFO"
    )
    ghost.mkdir()
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "metadata" in blob and ("missing" in blob or "regular" in blob)


def test_license_closure_rejects_metadata_mutation_during_hash(
    monkeypatch, tmp_path,
):
    """Evidence hashing must reject bytes that change inside the snapshot.

    Seeded defect: hash RECORD once without comparing its identity before and
    after the read, allowing a stale digest to be accepted as current.
    """
    _seed_closure_audit_surface(tmp_path)
    record = (
        tmp_path
        / ".venv"
        / "Lib"
        / "site-packages"
        / "alpha-1.dist-info"
        / "RECORD"
    )
    original_hash = preflight._sha256_file
    mutated = False

    def mutate_after_hash(path):
        nonlocal mutated
        digest = original_hash(path)
        if Path(path) == record and not mutated:
            mutated = True
            record.write_bytes(record.read_bytes() + b"changed-during-hash\n")
        return digest

    monkeypatch.setattr(preflight, "_sha256_file", mutate_after_hash)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert mutated
    assert sec.failed(), sec.lines
    assert "revalidate" in " ".join(message for _, message in sec.lines).lower()


def test_license_closure_rejects_file_added_after_tree_enumeration(
    monkeypatch, tmp_path,
):
    """Two complete snapshots catch a file added after the first walk."""
    _seed_closure_audit_surface(tmp_path)
    original_fingerprint = preflight._bundle_tree_fingerprint
    calls = 0

    def add_file_after_first_snapshot(root):
        nonlocal calls
        result = original_fingerprint(root)
        calls += 1
        if calls == 1:
            (Path(root) / "late-added.dll").write_bytes(b"late candidate bytes")
        return result

    monkeypatch.setattr(
        preflight, "_bundle_tree_fingerprint", add_file_after_first_snapshot
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert calls == 2
    assert sec.failed(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "changed between fingerprint snapshots" in blob


def test_license_closure_nominal_pass_fails_on_changed_bundle_file_count(
    monkeypatch, tmp_path,
):
    """Tree digest and file count are independent report bindings."""
    report_path, report = _seed_closure_audit_surface(tmp_path)
    report["bundle_file_count"] += 1
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "stale" in blob and "bundle_file_count" in blob


def test_license_closure_rejects_noncanonical_audit_timestamp(
    monkeypatch, tmp_path,
):
    """A date-only value is not the auditor's canonical UTC timestamp."""
    report_path, report = _seed_closure_audit_surface(tmp_path)
    report["audited_utc"] = "2026-07-15Z"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    assert "malformed" in " ".join(message for _, message in sec.lines).lower()


@pytest.mark.parametrize(
    "field",
    [
        "inventory_sha256",
        "license_index_sha256",
        "provenance_policy_sha256",
        "license_override_policy_sha256",
        "distribution_metadata_sha256",
    ],
)
def test_license_closure_nominal_pass_fails_on_changed_input_hash(
    monkeypatch, tmp_path, field,
):
    """Every audit input fingerprint must still describe current bytes.

    Seeded defect: omit ``field`` from the preflight comparison table.
    """
    report_path, report = _seed_closure_audit_surface(tmp_path)
    report[field] = "f" * 64 if report[field] != "f" * 64 else "e" * 64
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert sec.failed(), sec.lines
    blob = " ".join(message for _, message in sec.lines).lower()
    assert "stale" in blob and field.lower() in blob


def test_license_closure_fresh_report_passes_release(monkeypatch, tmp_path):
    """Exact current evidence produces an observable PASS.

    Seeded defect: perturb the canonical tree record encoding or any input path.
    """
    _seed_closure_audit_surface(tmp_path)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True)

    assert not sec.failed(), sec.lines
    assert any(status == "PASS" for status, _ in sec.lines), sec.lines


def test_license_closure_fingerprints_the_selected_build_venv(monkeypatch, tmp_path):
    """A bundle built with ``build.ps1 -Venv .venv-release`` was audited
    against that venv's packages; ``--venv .venv-release`` makes section 15
    fingerprint the same site-packages.

    Seeded defect: keep ``ROOT / ".venv"`` in
    _current_license_closure_fingerprints -> the fixture has no .venv ->
    "build site-packages is missing" -> red.
    """
    _seed_closure_audit_surface(tmp_path, venv=".venv-release")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    sec = preflight.check_bundle_license_closure(release=True, venv=".venv-release")

    assert not sec.failed(), sec.lines
    assert any(status == "PASS" for status, _ in sec.lines), sec.lines


@pytest.mark.parametrize(
    "argv,expected_release,expected_venv",
    [
        (["--release"], True, ".venv"),
        ([], False, ".venv"),
        (["--release", "--venv", ".venv-release"], True, ".venv-release"),
    ],
)
def test_preflight_main_invokes_license_closure_with_current_mode(
    monkeypatch, tmp_path, argv, expected_release, expected_venv,
):
    """The gate is live in the composition root, not merely callable, and
    the build venv chosen with --venv reaches every section that reads it
    (8 and 9 run its interpreter, 15 fingerprints its site-packages).

    Seeded defects: remove ``check_bundle_license_closure`` from ``main``;
    drop ``venv=venv`` from any of the three calls.
    """
    ordinary_checks = (
        "check_hf_repos",
        "check_hf_download_signature",
        "check_version_consistency",
        "check_lazy_imports",
        "check_llama_server",
        "check_audio_devices",
        "check_pyinstaller_spec",
        "check_llama_bundle_closure",
        "check_config_key_consumers",
        "check_bundle_module_launchers",
        "check_changelog_migration_notes",
        "check_no_subject_mocks_in_real_tier",
    )

    def passing_section(*_args, **_kwargs):
        section = Section(1, "fixture pass")
        section.add("PASS", "fixture")
        return section

    for name in ordinary_checks:
        monkeypatch.setattr(preflight, name, passing_section)

    seen_venvs = {}

    def venv_section(name):
        def section(*, release=False, venv=preflight.DEFAULT_VENV):
            seen_venvs[name] = venv
            return passing_section()
        return section

    for name in ("check_real_path_pytest", "check_bundle_dll_audit"):
        monkeypatch.setattr(preflight, name, venv_section(name))

    observed = []

    def failing_closure(*, release=False, venv=preflight.DEFAULT_VENV):
        observed.append(release)
        seen_venvs["check_bundle_license_closure"] = venv
        section = Section(15, "Bundle license closure audit evidence")
        section.add("FAIL", "fixture closure failure")
        return section

    monkeypatch.setattr(preflight, "check_bundle_license_closure", failing_closure)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    # A --release run proceeds only from the selected venv's interpreter.
    monkeypatch.setattr(preflight.sys, "prefix", str(tmp_path / expected_venv))

    assert preflight.main(argv) == 1
    assert observed == [expected_release]
    assert seen_venvs == dict.fromkeys(
        ("check_real_path_pytest", "check_bundle_dll_audit", "check_bundle_license_closure"),
        expected_venv,
    )


def test_release_preflight_refuses_to_run_outside_the_selected_venv(
    monkeypatch, tmp_path, capsys,
):
    """Sections 2, 4 and 7 import packages into preflight's own interpreter,
    so a --release run must run from the selected build venv, or the dev
    venv's libraries would vouch for the release build. Started from another
    interpreter, it names both and stops before any section runs.

    Seeded defect: drop the interpreter guard from ``main`` -> the sections
    run and main returns 0 -> red.
    """
    ran = []

    def recording_section(*_args, **_kwargs):
        ran.append(True)
        section = Section(1, "fixture pass")
        section.add("PASS", "fixture")
        return section

    for name in (
        "check_hf_repos",
        "check_hf_download_signature",
        "check_version_consistency",
        "check_lazy_imports",
        "check_llama_server",
        "check_audio_devices",
        "check_pyinstaller_spec",
        "check_real_path_pytest",
        "check_bundle_dll_audit",
        "check_llama_bundle_closure",
        "check_bundle_license_closure",
        "check_release_evidence_binding",
        "check_config_key_consumers",
        "check_bundle_module_launchers",
        "check_changelog_migration_notes",
        "check_no_subject_mocks_in_real_tier",
    ):
        monkeypatch.setattr(preflight, name, recording_section)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight.sys, "prefix", str(tmp_path / ".venv"))

    assert preflight.main(["--release", "--venv", ".venv-release"]) == 1
    assert ran == []
    out = capsys.readouterr().out
    assert (
        f"this is {tmp_path / '.venv'}, the selected venv is {tmp_path / '.venv-release'}"
        in out
    ), out


def test_preflight_venv_without_a_value_is_a_usage_error(monkeypatch, tmp_path, capsys):
    """Decision 107: ``--venv`` with nothing after it exits 2 and says what
    it needs, before any section runs (not a run against the default venv).

    Seeded defect: fall back to DEFAULT_VENV when the value is missing ->
    the sections run and main returns 0 -> red.
    """
    ran = []

    def recording_section(*_args, **_kwargs):
        ran.append(True)
        section = Section(1, "fixture pass")
        section.add("PASS", "fixture")
        return section

    for name in [name for name in vars(preflight) if name.startswith("check_")]:
        monkeypatch.setattr(preflight, name, recording_section)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    assert preflight.main(["--venv"]) == 2
    assert ran == []
    assert "--venv needs the build venv" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [["--venv=.venv-release"], ["--venv", "--release"]],
    ids=["equals-form", "flag-as-value"],
)
def test_preflight_venv_equals_form_or_flag_value_is_a_usage_error(
    monkeypatch, tmp_path, capsys, argv
):
    """``--venv=<x>`` (which the hand parser would ignore, running against
    the default venv) and a ``--venv`` value that is another flag (which it
    would take as the venv) exit 2 with the usage line before any section
    runs.

    Seeded defects: drop the ``--venv=`` check -> equals-form runs the
    sections and returns 0 -> red; drop the ``startswith("--")`` check ->
    flag-as-value takes ``--release`` as the venv and stops at the release
    interpreter guard (exit 1, not the usage line) -> red.
    """
    ran = []

    def recording_section(*_args, **_kwargs):
        ran.append(True)
        section = Section(1, "fixture pass")
        section.add("PASS", "fixture")
        return section

    for name in [name for name in vars(preflight) if name.startswith("check_")]:
        monkeypatch.setattr(preflight, name, recording_section)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    assert preflight.main(argv) == 2
    assert ran == []
    assert "--venv needs the build venv" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# check_release_evidence_binding -- section 16: every release phase, the
# frozen bundle and the installer were produced from the current HEAD.
# ---------------------------------------------------------------------------

def test_release_evidence_binding_fails_when_one_phase_ran_at_another_head(
    monkeypatch, tmp_path,
):
    """Phase evidence recorded at an older commit cannot vouch for HEAD.

    Seeded defect: drop the per-phase ``git_head`` comparison; the stale
    ``unit`` evidence then passes beside the current frozen evidence.
    """
    head = "a" * 40
    manifest = tmp_path / "scripts" / "release-gates.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phases": [
                    {"id": "unit", "timeout_seconds": 60, "args": ["tests"]},
                    {
                        "id": "frozen-bundle",
                        "timeout_seconds": 60,
                        "args": ["tests/test_real_frozen_bundle.py"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "build" / "yamabiko" / "bundle-inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text('{"entries": [], "schema_version": 1}\n', encoding="utf-8")
    gates = tmp_path / "build" / "release-gates"
    gates.mkdir()
    for phase_id, phase_head in (("unit", "b" * 40), ("frozen-bundle", head)):
        (gates / f"{phase_id}-evidence.json").write_text(
            json.dumps(
                {
                    "phase_id": phase_id,
                    "final_exit_status": 0,
                    "git_head": phase_head,
                    "git_dirty": False,
                    "bundle_inventory_sha256": _sha256(inventory),
                }
            ),
            encoding="utf-8",
        )
    installer = tmp_path / "installer" / "yamabiko-0.0.0-setup.exe"
    installer.parent.mkdir()
    installer.write_bytes(b"MZ" + b"installer-candidate")
    (gates / "installer.json").write_text(
        json.dumps(
            {
                "version": "0.0.0",
                "installer_file": "installer/yamabiko-0.0.0-setup.exe",
                "installer_sha256": _sha256(installer),
                "git_head": head,
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "_git_head", lambda: head)

    sec = preflight.check_release_evidence_binding(release=True)

    failures = [message for status, message in sec.lines if status == "FAIL"]
    assert len(failures) == 1, sec.lines
    assert "unit" in failures[0] and "git_head" in failures[0], sec.lines
