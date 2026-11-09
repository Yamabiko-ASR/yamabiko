from __future__ import annotations

import ast
from collections.abc import Iterable
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bundle_inventory import (
    InventoryEntry,
    InventoryRoots,
    analysis_entries,
    load_inventory,
    write_analysis_inventory,
)


COLLECTIONS = ("scripts", "pure", "binaries", "datas", "zipfiles")


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _analysis(**collections: object) -> SimpleNamespace:
    values = {name: () for name in COLLECTIONS}
    values.update(collections)
    return SimpleNamespace(**values)


def _inventory_fixture(tmp_path: Path) -> tuple[SimpleNamespace, InventoryRoots]:
    repo_root = tmp_path / "repo"
    python_root = repo_root / ".venv"
    site_packages = python_root / "Lib" / "site-packages"
    roots = InventoryRoots(
        repo_root=repo_root,
        site_packages=site_packages,
        python_root=python_root,
    )

    watchdog = _write(repo_root / "yamabiko" / "watchdog.py", b"watchdog")
    config = _write(repo_root / "yamabiko" / "default_config.yaml", b"config")
    package_module = _write(site_packages / "sample_pkg" / "module.py", b"module")
    python_dll = _write(python_root / "python312.dll", b"python runtime")
    base_library = _write(python_root / "Lib" / "base_library.zip", b"stdlib")
    external_a = _write(tmp_path / "vendor-a" / "payload.dll", b"same external bytes")
    external_b = _write(tmp_path / "vendor-b" / "payload.dll", b"same external bytes")

    script_row = ("yamabiko", str(watchdog), "PYSOURCE")
    return (
        _analysis(
            scripts=(script_row, script_row),
            pure=(("sample_pkg.module", str(package_module), "PYMODULE"),),
            binaries=(
                ("python312.dll", str(python_dll), "BINARY"),
                ("vendor/payload-a.dll", str(external_a), "BINARY"),
            ),
            datas=(
                ("yamabiko/default_config.yaml", str(config), "DATA"),
                ("vendor/payload-b.dll", str(external_b), "DATA"),
            ),
            zipfiles=(("base_library.zip", str(base_library), "ZIPFILE"),),
        ),
        roots,
    )


def _reverse_analysis(analysis: SimpleNamespace) -> SimpleNamespace:
    return _analysis(
        **{
            collection: tuple(reversed(getattr(analysis, collection)))
            for collection in COLLECTIONS
        }
    )


def _entry(entries: Iterable[InventoryEntry], logical_name: str) -> InventoryEntry:
    return next(entry for entry in entries if entry.logical_name == logical_name)


def test_write_analysis_inventory_is_byte_stable_across_toc_order_and_duplicates(
    tmp_path: Path,
) -> None:
    analysis, roots = _inventory_fixture(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_entries = write_analysis_inventory(analysis, first, roots=roots)
    second_entries = write_analysis_inventory(
        _reverse_analysis(analysis), second, roots=roots
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_entries == second_entries
    assert len(first_entries) == 7
    assert sum(entry.logical_name == "yamabiko" for entry in first_entries) == 1
    assert first.read_bytes().endswith(b"\n")
    assert not first.read_bytes().endswith(b"\n\n")
    assert b"\r\n" not in first.read_bytes()


def test_analysis_entries_canonicalizes_overlapping_roots_and_destinations(
    tmp_path: Path,
) -> None:
    analysis, roots = _inventory_fixture(tmp_path)

    entries = analysis_entries(analysis, roots=roots)

    assert {entry.collection for entry in entries} == set(COLLECTIONS)
    assert all(":\\" not in entry.source_path for entry in entries)
    assert all(not entry.source_path.startswith("/") for entry in entries)
    assert _entry(entries, "yamabiko").source_path == "repo://yamabiko/watchdog.py"
    assert (
        _entry(entries, "sample_pkg.module").source_path
        == "site-packages://sample_pkg/module.py"
    )
    assert _entry(entries, "python312.dll").source_path == "python://python312.dll"
    assert (
        _entry(entries, "base_library.zip").source_path
        == "python://Lib/base_library.zip"
    )
    assert _entry(entries, "sample_pkg.module").destination == "PYZ:sample_pkg.module"
    assert _entry(entries, "yamabiko").destination == "EXE:yamabiko"
    assert (
        _entry(entries, "vendor/payload-a.dll").destination
        == "_internal/vendor/payload-a.dll"
    )


def test_analysis_entries_identifies_equal_external_files_without_leaking_roots(
    tmp_path: Path,
) -> None:
    analysis, roots = _inventory_fixture(tmp_path)

    entries = analysis_entries(analysis, roots=roots)
    external_a = _entry(entries, "vendor/payload-a.dll")
    external_b = _entry(entries, "vendor/payload-b.dll")

    assert external_a.source_path == external_b.source_path
    assert external_a.source_path.startswith("external://sha256/")
    assert external_a.source_path.endswith("/payload.dll")
    assert "vendor-a" not in external_a.source_path
    assert "vendor-b" not in external_a.source_path


@pytest.mark.parametrize("conflict_kind", ["same-collection", "cross-collection"])
def test_analysis_entries_rejects_distinct_entries_for_the_same_destination(
    tmp_path: Path, conflict_kind: str
) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    for root in (repo_root, site_packages, python_root):
        root.mkdir()
    first_source = _write(repo_root / "first.dll", b"first")
    second_source = _write(repo_root / "second.dll", b"second")
    roots = InventoryRoots(repo_root, site_packages, python_root)

    if conflict_kind == "same-collection":
        analysis = _analysis(
            datas=(
                ("shared.dll", str(first_source), "DATA"),
                ("shared.dll", str(second_source), "DATA"),
            )
        )
    else:
        analysis = _analysis(
            binaries=(("shared.dll", str(first_source), "BINARY"),),
            datas=(("shared.dll", str(second_source), "DATA"),),
        )

    with pytest.raises(ValueError, match=r"destination.*_internal/shared\.dll"):
        analysis_entries(analysis, roots=roots)


@pytest.mark.parametrize("conflict_kind", ["same-collection", "cross-collection"])
def test_analysis_entries_rejects_case_only_windows_destination_conflicts(
    tmp_path: Path, conflict_kind: str
) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    for root in (repo_root, site_packages, python_root):
        root.mkdir()
    first_source = _write(repo_root / "first.bin", b"first")
    second_source = _write(repo_root / "second.bin", b"second")
    roots = InventoryRoots(repo_root, site_packages, python_root)

    if conflict_kind == "same-collection":
        analysis = _analysis(
            datas=(
                ("Foo.txt", str(first_source), "DATA"),
                ("foo.txt", str(second_source), "DATA"),
            )
        )
    else:
        analysis = _analysis(
            binaries=(("Foo.dll", str(first_source), "BINARY"),),
            datas=(("foo.dll", str(second_source), "DATA"),),
        )

    with pytest.raises(ValueError, match=r"destination.*_internal/foo\."):
        analysis_entries(analysis, roots=roots)


@pytest.mark.parametrize("namespace_pair", ["exe-pyz", "pyz-internal"])
def test_analysis_entries_allows_same_suffix_in_distinct_bundle_namespaces(
    tmp_path: Path, namespace_pair: str
) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    for root in (repo_root, site_packages, python_root):
        root.mkdir()
    first_source = _write(repo_root / "first.py", b"first")
    second_source = _write(repo_root / "second.py", b"second")
    roots = InventoryRoots(repo_root, site_packages, python_root)

    if namespace_pair == "exe-pyz":
        analysis = _analysis(
            scripts=(("same", str(first_source), "PYSOURCE"),),
            pure=(("same", str(second_source), "PYMODULE"),),
        )
        expected = {"EXE:same", "PYZ:same"}
    else:
        analysis = _analysis(
            pure=(("same", str(first_source), "PYMODULE"),),
            datas=(("same", str(second_source), "DATA"),),
        )
        expected = {"PYZ:same", "_internal/same"}

    entries = analysis_entries(analysis, roots=roots)

    assert {entry.destination for entry in entries} == expected


def test_write_and_load_inventory_round_trip_the_declared_schema(tmp_path: Path) -> None:
    analysis, roots = _inventory_fixture(tmp_path)
    output = tmp_path / "nested" / "bundle-inventory.json"

    written = write_analysis_inventory(analysis, output, roots=roots)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "entries"}
    assert payload["schema_version"] == 1
    assert all(
        set(row)
        == {"collection", "logical_name", "source_path", "destination", "type_code"}
        for row in payload["entries"]
    )
    assert load_inventory(output) == written


@pytest.mark.parametrize("source_kind", ["missing", "directory"])
def test_analysis_entries_rejects_external_sources_without_file_bytes(
    tmp_path: Path, source_kind: str
) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    for root in (repo_root, site_packages, python_root):
        root.mkdir()
    roots = InventoryRoots(repo_root, site_packages, python_root)
    source = tmp_path / "external" / "payload.bin"
    if source_kind == "directory":
        source.mkdir(parents=True)

    analysis = _analysis(datas=(("payload.bin", str(source), "DATA"),))

    with pytest.raises(ValueError, match="TOC source.*regular file"):
        analysis_entries(analysis, roots=roots)


@pytest.mark.parametrize("source_kind", ["missing", "directory"])
def test_analysis_entries_rejects_rooted_sources_without_file_bytes(
    tmp_path: Path, source_kind: str
) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    for root in (repo_root, site_packages, python_root):
        root.mkdir()
    roots = InventoryRoots(repo_root, site_packages, python_root)
    source = repo_root / "payload.bin"
    if source_kind == "directory":
        source.mkdir()

    analysis = _analysis(datas=(("payload.bin", str(source), "DATA"),))

    with pytest.raises(ValueError, match="TOC source.*regular file"):
        analysis_entries(analysis, roots=roots)


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_analysis_entries_rejects_roots_that_are_not_existing_directories(
    tmp_path: Path, root_kind: str
) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    site_packages.mkdir()
    python_root.mkdir()
    if root_kind == "file":
        repo_root.write_bytes(b"not a directory")
    roots = InventoryRoots(repo_root, site_packages, python_root)

    with pytest.raises(ValueError, match="inventory root 'repo'.*existing directory"):
        analysis_entries(_analysis(), roots=roots)


def test_analysis_entries_rejects_malformed_toc_rows(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    site_packages = tmp_path / "site-packages"
    python_root = tmp_path / "python"
    for root in (repo_root, site_packages, python_root):
        root.mkdir()
    roots = InventoryRoots(repo_root, site_packages, python_root)
    analysis = _analysis(datas=(("payload.bin", str(tmp_path / "payload.bin")),))

    with pytest.raises(ValueError, match="3-item TOC row"):
        analysis_entries(analysis, roots=roots)


VALID_ROW = {
    "collection": "datas",
    "logical_name": "yamabiko/default_config.yaml",
    "source_path": "repo://yamabiko/default_config.yaml",
    "destination": "_internal/yamabiko/default_config.yaml",
    "type_code": "DATA",
}


def _write_payload(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_inventory_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = _write_payload(tmp_path / "inventory.json", {"schema_version": 2, "entries": []})

    with pytest.raises(ValueError, match="schema_version"):
        load_inventory(path)


def test_load_inventory_rejects_duplicate_entry_keys(tmp_path: Path) -> None:
    path = _write_payload(
        tmp_path / "inventory.json",
        {"schema_version": 1, "entries": [VALID_ROW, dict(VALID_ROW)]},
    )

    with pytest.raises(ValueError, match="duplicate inventory entry key"):
        load_inventory(path)


@pytest.mark.parametrize("conflict_kind", ["same-collection", "cross-collection"])
def test_load_inventory_rejects_distinct_entries_for_the_same_destination(
    tmp_path: Path, conflict_kind: str
) -> None:
    first = {
        "collection": "datas",
        "logical_name": "shared.dll",
        "source_path": "repo://first.dll",
        "destination": "_internal/shared.dll",
        "type_code": "DATA",
    }
    second = {
        **first,
        "source_path": "repo://second.dll",
    }
    if conflict_kind == "cross-collection":
        first = {**first, "collection": "binaries", "type_code": "BINARY"}

    path = _write_payload(
        tmp_path / "inventory.json",
        {"schema_version": 1, "entries": [first, second]},
    )

    with pytest.raises(ValueError, match=r"destination.*_internal/shared\.dll"):
        load_inventory(path)


@pytest.mark.parametrize("conflict_kind", ["same-collection", "cross-collection"])
def test_load_inventory_rejects_case_only_windows_destination_conflicts(
    tmp_path: Path, conflict_kind: str
) -> None:
    if conflict_kind == "same-collection":
        first = {
            "collection": "datas",
            "logical_name": "Foo.txt",
            "source_path": "repo://first.bin",
            "destination": "_internal/Foo.txt",
            "type_code": "DATA",
        }
        second = {
            **first,
            "logical_name": "foo.txt",
            "source_path": "repo://second.bin",
            "destination": "_internal/foo.txt",
        }
    else:
        first = {
            "collection": "binaries",
            "logical_name": "Foo.dll",
            "source_path": "repo://first.bin",
            "destination": "_internal/Foo.dll",
            "type_code": "BINARY",
        }
        second = {
            "collection": "datas",
            "logical_name": "foo.dll",
            "source_path": "repo://second.bin",
            "destination": "_internal/foo.dll",
            "type_code": "DATA",
        }

    path = _write_payload(
        tmp_path / "inventory.json",
        {"schema_version": 1, "entries": [first, second]},
    )

    with pytest.raises(ValueError, match=r"destination.*_internal/foo\."):
        load_inventory(path)


@pytest.mark.parametrize("namespace_pair", ["exe-pyz", "pyz-internal"])
def test_load_inventory_allows_same_suffix_in_distinct_bundle_namespaces(
    tmp_path: Path, namespace_pair: str
) -> None:
    exe = {
        "collection": "scripts",
        "logical_name": "same",
        "source_path": "repo://script.py",
        "destination": "EXE:same",
        "type_code": "PYSOURCE",
    }
    pyz = {
        "collection": "pure",
        "logical_name": "same",
        "source_path": "repo://module.py",
        "destination": "PYZ:same",
        "type_code": "PYMODULE",
    }
    internal = {
        "collection": "datas",
        "logical_name": "same",
        "source_path": "repo://data.bin",
        "destination": "_internal/same",
        "type_code": "DATA",
    }
    rows = [exe, pyz] if namespace_pair == "exe-pyz" else [pyz, internal]
    path = _write_payload(
        tmp_path / "inventory.json",
        {"schema_version": 1, "entries": rows},
    )

    entries = load_inventory(path)

    assert {entry.destination for entry in entries} == {
        row["destination"] for row in rows
    }


def test_load_inventory_returns_canonical_order_independent_of_json_row_order(
    tmp_path: Path,
) -> None:
    first = {
        **VALID_ROW,
        "logical_name": "a.yaml",
        "source_path": "repo://a.yaml",
        "destination": "_internal/a.yaml",
    }
    last = {
        **VALID_ROW,
        "logical_name": "z.yaml",
        "source_path": "repo://z.yaml",
        "destination": "_internal/z.yaml",
    }
    forward = _write_payload(
        tmp_path / "forward.json",
        {"schema_version": 1, "entries": [last, first]},
    )
    reverse = _write_payload(
        tmp_path / "reverse.json",
        {"schema_version": 1, "entries": [first, last]},
    )

    forward_entries = load_inventory(forward)
    reverse_entries = load_inventory(reverse)

    assert forward_entries == reverse_entries
    assert forward_entries == tuple(sorted(forward_entries))


def test_load_inventory_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        '{"schema_version": 1, "schema_version": 1, "entries": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_inventory(path)


@pytest.mark.parametrize(
    "source_path",
    [
        pytest.param(r"C:\private\module.py", id="windows-absolute"),
        pytest.param("/private/module.py", id="posix-absolute"),
        pytest.param("repo://../private/module.py", id="parent-component"),
        pytest.param("repo://C:/private/module.py", id="drive-in-canonical-path"),
        pytest.param(
            "repo://nested/C:/private/module.py", id="nested-upper-drive-token"
        ),
        pytest.param(
            "repo://nested/c:/private/module.py", id="nested-lower-drive-token"
        ),
    ],
)
def test_load_inventory_rejects_absolute_or_escaping_source_paths(
    tmp_path: Path, source_path: str
) -> None:
    row = {**VALID_ROW, "source_path": source_path}
    path = _write_payload(
        tmp_path / "inventory.json", {"schema_version": 1, "entries": [row]}
    )

    with pytest.raises(ValueError, match="source_path"):
        load_inventory(path)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({}, id="missing-fields"),
        pytest.param({**VALID_ROW, "extra": "field"}, id="extra-field"),
        pytest.param({**VALID_ROW, "collection": "other"}, id="unknown-collection"),
        pytest.param(
            {**VALID_ROW, "logical_name": "../escape.yaml"},
            id="escaping-logical-name",
        ),
        pytest.param(
            {**VALID_ROW, "destination": "yamabiko/default_config.yaml"},
            id="wrong-destination",
        ),
        pytest.param({**VALID_ROW, "type_code": 7}, id="non-string-field"),
        pytest.param(
            {**VALID_ROW, "source_path": "other://module.py"},
            id="unknown-source-scheme",
        ),
    ],
)
def test_load_inventory_rejects_malformed_rows(tmp_path: Path, row: object) -> None:
    path = _write_payload(
        tmp_path / "inventory.json", {"schema_version": 1, "entries": [row]}
    )

    with pytest.raises(ValueError, match="inventory entry"):
        load_inventory(path)


def test_spec_writes_inventory_to_exact_analysis_workpath() -> None:
    spec_path = Path(__file__).parents[1] / "installer" / "yamabiko.spec"
    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))

    analysis_index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "a" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Analysis"
    )
    inventory_calls = [
        (index, node.value)
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "write_analysis_inventory"
    ]

    assert len(inventory_calls) == 1
    inventory_index, call = inventory_calls[0]
    assert inventory_index > analysis_index
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "a"
    assert ast.unparse(call.args[1]) == "Path(workpath) / 'bundle-inventory.json'"
    assert "yamabiko" not in ast.unparse(call.args[1]).casefold()

    roots_keyword = next(keyword for keyword in call.keywords if keyword.arg == "roots")
    assert isinstance(roots_keyword.value, ast.Call)
    assert ast.unparse(roots_keyword.value.func) == "InventoryRoots"
    roots_arguments = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in roots_keyword.value.keywords
    }
    assert roots_arguments == {
        "repo_root": "REPO.resolve()",
        "site_packages": "Path(sysconfig.get_path('purelib'))",
        "python_root": "Path(sys.base_prefix)",
    }

    helper_import_index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.bundle_inventory"
    )
    repo_path_insert_calls = [
        (index, node.value)
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "sys.path.insert"
    ]
    assert len(repo_path_insert_calls) == 1
    repo_path_insert_index, repo_path_insert = repo_path_insert_calls[0]
    assert len(repo_path_insert.args) == 2
    assert (
        isinstance(repo_path_insert.args[0], ast.Constant)
        and type(repo_path_insert.args[0].value) is int
        and repo_path_insert.args[0].value == 0
    )
    assert ast.unparse(repo_path_insert.args[1]) == "str(REPO)"
    assert repo_path_insert_index < helper_import_index < analysis_index

    exe_call = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "exe"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "EXE"
    )
    assert all(keyword.arg != "contents_directory" for keyword in exe_call.keywords)
