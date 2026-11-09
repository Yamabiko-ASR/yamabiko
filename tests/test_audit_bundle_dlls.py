"""Unit tests for scripts/audit_bundle_dlls.py's missing-DLL classification.

The bundle DLL audit -- preflight section 9, the missing-DLL release
gate -- had two independent false-negative (missing-DLL masking) paths:

* ``_importers_are_all_pyds`` suppressed ANY missing DLL whose importers
  are all .pyd extension modules. That heuristic was meant for the pyo3
  self-import convention, but that case is already covered by the narrower
  stem-matching ``_is_pyd_self_import``; the blanket rule additionally
  swallowed real deps like onnxruntime.dll (imported only by
  onnxruntime_pybind11_state.pyd) or libffi-8.dll (_ctypes.pyd).
* ``enumerate_bundle_dlls`` populated the ``present`` map from the WHOLE
  bundle tree, so a DLL existing only under _internal/llama/ (a child
  process's private search path, deliberately excluded from the importer
  walk) silently satisfied a main-tree import.

The classification loop is exercised via the pure ``compute_missing``
seam, and tree enumeration via ``enumerate_bundle_dlls`` on a tmp_path
bundle skeleton (presence is name-based; no PE parsing involved).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_bundle_dlls as audit


# ---------------------------------------------------------------------------
# A real dependency DLL imported only by .pyd extension modules must be
# reported missing, not suppressed as a pyo3 false positive.
# ---------------------------------------------------------------------------

def test_dll_imported_only_by_pyds_is_reported_missing():
    """onnxruntime.dll's sole importer is onnxruntime_pybind11_state.pyd.
    If the DLL is dropped from the bundle the frozen app dies on first
    VAD/EOT import -- the audit must say MISSING. Seeded defect: restore
    the all-pyd-importers suppression."""
    pyd = Path("onnxruntime_pybind11_state.pyd")
    reverse = {"onnxruntime.dll": [pyd]}
    missing = audit.compute_missing(reverse, present={}, natives=[pyd])
    assert "onnxruntime.dll" in missing


def test_pyo3_self_import_dll_stays_suppressed():
    """The genuine pyo3/maturin convention: foo.pyd declares an import
    of 'foo.dll' that never exists on disk (the loader resolves it via
    the import system). Stem-matching suppression must survive the fix.
    Seeded defect: delete the _is_pyd_self_import guard."""
    pyd = Path("hf_xet.cp312-win_amd64.pyd")
    reverse = {"hf_xet.dll": [pyd]}
    missing = audit.compute_missing(reverse, present={}, natives=[pyd])
    assert "hf_xet.dll" not in missing


@pytest.mark.parametrize("platform_dll", ("kernel32.dll", "nvcuda.dll"))
def test_public_system_dll_predicate_exempts_os_and_driver_supplied_dlls(
    platform_dll,
):
    """The provenance resolver may consume only the public platform predicate.

    Both stock-Windows DLLs and NVIDIA driver DLLs are supplied by the target
    platform rather than the bundle.  Seeded defect: expose only the old
    Windows-only ``_is_system`` body; nvcuda becomes a false closure error.
    """
    exe = Path("yamabiko.exe")
    reverse = {platform_dll: [exe]}
    assert audit.is_system_dll(platform_dll)
    missing = audit.compute_missing(reverse, present={}, natives=[exe])
    assert missing == []


# ---------------------------------------------------------------------------
# DLLs under the excluded _internal/llama subtree must not count as
# 'present' for main-tree imports.
# ---------------------------------------------------------------------------

def _make_bundle(tmp_path: Path) -> Path:
    """Minimal bundle skeleton: one main-tree DLL, one llama-only DLL."""
    bundle = tmp_path / "dist" / "yamabiko"
    (bundle / "_internal" / "llama").mkdir(parents=True)
    (bundle / "yamabiko.exe").write_bytes(b"MZ fake")
    (bundle / "_internal" / "shiboken6.dll").write_bytes(b"MZ fake")
    (bundle / "_internal" / "llama" / "msvcp140.dll").write_bytes(b"MZ fake")
    return bundle


def test_enumerate_excludes_llama_subtree_from_present_map(tmp_path):
    """A DLL that exists ONLY under _internal/llama/ is not on
    yamabiko.exe's DLL search path and must not appear in the presence
    map. Seeded defect: drop the subtree filter from the rglob loop."""
    bundle = _make_bundle(tmp_path)
    present, natives = audit.enumerate_bundle_dlls(bundle)
    assert "shiboken6.dll" in present
    assert "msvcp140.dll" not in present
    assert all(not audit._is_subprocess_subtree(p, bundle) for p in natives)


def test_llama_only_copy_does_not_satisfy_main_tree_import(tmp_path):
    """End-to-end through both seams: yamabiko.exe imports msvcp140.dll,
    whose only on-disk copy lives under _internal/llama/. The audit must
    report it MISSING -- a same-named child-process copy must not mask a
    missing main-app DLL."""
    bundle = _make_bundle(tmp_path)
    present, natives = audit.enumerate_bundle_dlls(bundle)
    reverse = {"msvcp140.dll": [bundle / "yamabiko.exe"]}
    missing = audit.compute_missing(reverse, present=present, natives=natives)
    assert "msvcp140.dll" in missing


# ---------------------------------------------------------------------------
# Maturin self-export-name string-scan false positive: a maturin-built
# pyd keeps the Rust cdylib's ORIGINAL name in its PE export directory
# (jiter.cp312-win_amd64.pyd exports as 'jiter_python.dll'), and that
# string sits NUL-terminated in the binary where scan_runtime_loaded reads
# it as a LoadLibrary dependency. _is_pyd_self_import can't suppress it
# because the dll stem differs from the pyd stem ('jiter_python' vs
# 'jiter'). A module's own identity string is definitionally not a
# dependency, so the scanner drops exactly that one name -- per-file PE
# metadata, not the class-based importer heuristic.
#
# Subject artifact: fast_bunkai's pyd -- a declared runtime dep
# (yamabiko/requirements.txt), so it exists in every env that can run
# this suite. Real PE, no mocks.
# ---------------------------------------------------------------------------

def _fast_bunkai_pyd() -> Path:
    import fast_bunkai
    pkg_dir = Path(fast_bunkai.__file__).parent
    pyd = pkg_dir / "_fast_bunkai.pyd"
    assert pyd.is_file(), f"fast_bunkai layout changed; no pyd at {pyd}"
    return pyd


@pytest.mark.real
def test_pe_export_name_reads_maturin_cdylib_name():
    """Positive control for the suppression below: the real artifact
    really does carry its pre-rename cdylib name in the export
    directory. If maturin stops embedding it, this goes red and the
    suppression test becomes moot -- loudly, not silently.
    Seeded defect: point _pe_export_name at the IMPORT directory."""
    assert audit._pe_export_name(_fast_bunkai_pyd()) == "fast_bunkai_native.dll"


@pytest.mark.real
def test_string_scan_drops_pyd_own_export_name():
    """The regression pin: the scanner must not report the pyd's own
    identity string as a runtime-loaded dependency -- that's what made
    preflight section 9 fail on a healthy bundle (fast_bunkai_native /
    jiter_python / safetensors_rust 'missing'). Seeded defect: delete
    the own-export-name filter from scan_runtime_loaded."""
    scanned = {audit._norm(n) for n in audit.scan_runtime_loaded(_fast_bunkai_pyd())}
    assert "fast_bunkai_native.dll" not in scanned


@pytest.mark.real
def test_string_scan_still_reports_foreign_dll_strings():
    """Over-suppression guard: dropping the self name must not swallow
    OTHER runtime-loadable names in the same binary -- the
    cublas-via-LoadLibrary class the scanner exists to catch.
    vcruntime140.dll is in this pyd's import directory and therefore in
    its string table. Seeded defect: make the filter drop everything
    (e.g. `if own` -> `found = set()`)."""
    scanned = {audit._norm(n) for n in audit.scan_runtime_loaded(_fast_bunkai_pyd())}
    assert "vcruntime140.dll" in scanned


def test_pe_export_name_none_for_non_pe_file(tmp_path):
    """Robustness pin: a non-PE file yields None (no crash), so the
    scanner's filter degrades to a no-op on data files. Seeded defect:
    let pefile.PEFormatError propagate out of _pe_export_name."""
    junk = tmp_path / "not_a_pe.dll"
    junk.write_bytes(b"this is not a PE file")
    assert audit._pe_export_name(junk) is None


def test_parse_imports_rejects_invalid_pe(tmp_path):
    junk = tmp_path / "not_a_pe.dll"
    junk.write_bytes(b"MZ not actually a PE")

    with pytest.raises(ValueError, match="PE import"):
        audit.parse_imports(junk)


@pytest.mark.parametrize("failing_directory", ("import", "delay-import"))
def test_parse_imports_rejects_directory_parse_failure(
    tmp_path, monkeypatch, failing_directory
):
    native = tmp_path / "native.dll"
    native.write_bytes(b"MZ synthetic")

    class BrokenPE:
        def __init__(self):
            self.calls = 0

        def parse_data_directories(self, *, directories):
            self.calls += 1
            if failing_directory == "import" or self.calls == 2:
                raise RuntimeError(f"broken {failing_directory} directory")

        def close(self):
            return None

    monkeypatch.setattr(audit.pefile, "PE", lambda *_args, **_kwargs: BrokenPE())

    with pytest.raises(ValueError, match="PE import"):
        audit.parse_imports(native)


# ---------------------------------------------------------------------------
# Gutted-bundle guard: an empty or exe-less dist/yamabiko must FAIL the
# audit, not sail through as "0 native files, 0 missing, exit 0" (which
# preflight section 9 then reports as PASS "Native files: 0").
# ---------------------------------------------------------------------------


def test_main_returns_2_when_bundle_dir_absent(tmp_path, monkeypatch, capsys):
    """rc 2 is the 'no bundle' contract preflight section 9 maps to
    SKIP; the gutted-bundle guard below must not repurpose it."""
    monkeypatch.setattr(audit, "BUNDLE", tmp_path / "dist" / "yamabiko")
    assert audit.main() == 2
    assert "no bundle" in capsys.readouterr().err


def test_main_fails_on_empty_bundle_dir(tmp_path, monkeypatch, capsys):
    """An empty dist/yamabiko (wiped mid-build, or gutted by a cleanup
    gone wrong) yields natives=[] -> missing=[] -> the old main()
    exited 0 and preflight printed PASS 'Native files: 0'. Zero native
    files is never a healthy bundle; it must exit with the audit's
    failure code. Seeded defect: drop the zero-natives guard from
    main() -> rc 0 -> red."""
    bundle = tmp_path / "dist" / "yamabiko"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(audit, "BUNDLE", bundle)
    rc = audit.main()
    assert rc == 1
    assert "no native files" in capsys.readouterr().err


def test_main_fails_when_bundle_root_lacks_yamabiko_exe(tmp_path, monkeypatch, capsys):
    """A tree with stray DLLs but no yamabiko.exe at the bundle root is
    a partial/gutted dist -- nothing the installer ships could launch.
    Seeded defect: drop the yamabiko.exe root check -> rc 0 -> red."""
    bundle = tmp_path / "dist" / "yamabiko"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "_internal" / "some.dll").write_bytes(b"MZ not a real PE")
    monkeypatch.setattr(audit, "BUNDLE", bundle)
    rc = audit.main()
    assert rc == 1
    assert "yamabiko.exe" in capsys.readouterr().err


def test_main_passes_on_minimal_bundle_with_exe_at_root(tmp_path, monkeypatch):
    """Positive control for the gutted-bundle guard: a tree with
    yamabiko.exe at the root and a DLL under _internal/ (non-PE
    content, so no imports and no missing deps) exits 0 -- the guard
    must not over-fire on a healthy-shaped bundle."""
    bundle = tmp_path / "dist" / "yamabiko"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "yamabiko.exe").write_bytes(b"MZ fake")
    (bundle / "_internal" / "some.dll").write_bytes(b"MZ fake")
    monkeypatch.setattr(audit, "BUNDLE", bundle)
    monkeypatch.setattr(audit, "parse_imports", lambda _path: [])
    assert audit.main() == 0


def test_main_fails_when_native_pe_cannot_be_parsed(
    tmp_path, monkeypatch, capsys
):
    bundle = tmp_path / "dist" / "yamabiko"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "yamabiko.exe").write_bytes(b"MZ malformed")
    monkeypatch.setattr(audit, "BUNDLE", bundle)

    def fail_parse(_path):
        raise ValueError("synthetic PE parse failure")

    monkeypatch.setattr(audit, "parse_imports", fail_parse)

    assert audit.main() == 1
    captured = capsys.readouterr()
    assert "PE imports" in captured.err
    assert "yamabiko.exe" in captured.err
