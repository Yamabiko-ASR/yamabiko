"""Causal unit and static gates for excluding the unused media stack."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_bundle_inventory import find_forbidden_media
from scripts.bundle_inventory import InventoryEntry, SCHEMA_VERSION


_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit_bundle_inventory.py"

_FORBIDDEN_COMPONENTS = (
    "faster_whisper.audio",
    "faster_whisper/audio.py",
    "av._core",
    "av.libs/avcodec-62.dll",
    "libavcodec-62.dll",
    "avdevice-62.dll",
    "libavdevice-62.dll",
    "avfilter-11.dll",
    "libavfilter-11.dll",
    "avformat-62.dll",
    "libavformat-62.dll",
    "avutil-60.dll",
    "libavutil-60.dll",
    "swresample-6.dll",
    "libswresample-6.dll",
    "swscale-9.dll",
    "libswscale-9.dll",
    "openh264.dll",
    "libopenh264.dll",
    "x264-164.dll",
    "libx264-164.dll",
    "x265.dll",
    "libx265.dll",
    "ffmpeg.exe",
    "ffprobe.exe",
    "ffplay.exe",
)
_INVENTORY_PATH_FIELDS = ("logical_name", "source_path", "destination")
_ALLOWED_COMPONENTS = ("avatar.py", "save.dll", "x2650.dll", "glibx264.dll")


def _inventory_entry() -> InventoryEntry:
    return InventoryEntry(
        collection="datas",
        logical_name="assets/clean.dat",
        source_path="repo://assets/clean.dat",
        destination="_internal/assets/clean.dat",
        type_code="DATA",
    )


@pytest.mark.parametrize("field", _INVENTORY_PATH_FIELDS)
@pytest.mark.parametrize("forbidden", _FORBIDDEN_COMPONENTS)
def test_media_audit_reports_each_forbidden_family_in_each_inventory_path_field(
    field: str,
    forbidden: str,
) -> None:
    entry = replace(_inventory_entry(), **{field: forbidden})

    errors = find_forbidden_media((entry,))

    assert errors
    assert forbidden.casefold() in "\n".join(errors).casefold()


@pytest.mark.parametrize("field", _INVENTORY_PATH_FIELDS)
@pytest.mark.parametrize("allowed", _ALLOWED_COMPONENTS)
def test_media_audit_does_not_match_allowed_substrings_in_inventory_fields(
    field: str,
    allowed: str,
) -> None:
    entry = replace(_inventory_entry(), **{field: allowed})
    assert find_forbidden_media((entry,)) == ()


@pytest.mark.parametrize("forbidden", _FORBIDDEN_COMPONENTS)
def test_media_audit_reports_each_forbidden_family_in_final_tree(
    tmp_path: Path,
    forbidden: str,
) -> None:
    dist = tmp_path / "dist"
    candidate = dist.joinpath(*forbidden.split("/"))
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"sentinel")

    errors = find_forbidden_media((), dist)

    assert errors
    assert forbidden.casefold() in "\n".join(errors).casefold()


def test_media_audit_reports_empty_forbidden_directory(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "_internal" / "av").mkdir(parents=True)

    assert find_forbidden_media((), dist) == (
        "dist: _internal/av",
    )


@pytest.mark.parametrize("allowed", _ALLOWED_COMPONENTS)
def test_media_audit_does_not_match_allowed_substrings_in_final_tree(
    tmp_path: Path,
    allowed: str,
) -> None:
    dist = tmp_path / "dist"
    candidate = dist / "nested" / allowed
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"sentinel")

    assert find_forbidden_media((), dist) == ()


def test_media_audit_normalizes_case_and_backslashes() -> None:
    entry = replace(
        _inventory_entry(),
        destination=r"_INTERNAL\AV.LIBS\LIBX264-165-HASH.DLL",
    )

    assert find_forbidden_media((entry,)) == (
        r"inventory.destination: _INTERNAL\AV.LIBS\LIBX264-165-HASH.DLL",
    )


def test_media_audit_deduplicates_and_sorts_diagnostics(tmp_path: Path) -> None:
    duplicate = replace(_inventory_entry(), logical_name="av._core")
    dist = tmp_path / "dist"
    (dist / "av.libs").mkdir(parents=True)
    (dist / "av.libs" / "libx265.dll").write_bytes(b"sentinel")

    errors = find_forbidden_media((duplicate, duplicate), dist)

    assert errors == tuple(sorted(set(errors)))
    assert len(errors) == 3


def test_media_audit_fails_closed_when_supplied_dist_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="dist"):
        find_forbidden_media((), tmp_path / "missing-dist")


def _write_inventory(path: Path, logical_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "entries": [
                    {
                        "collection": "datas",
                        "logical_name": logical_name,
                        "source_path": f"repo://payload/{logical_name}",
                        "destination": f"_internal/{logical_name}",
                        "type_code": "DATA",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(_AUDIT_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_media_audit_cli_direct_invocation_is_silent_for_clean_inventory(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "inventory.json"
    _write_inventory(inventory, "assets/avatar.py")

    result = _run_cli("--inventory", str(inventory), cwd=tmp_path)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_media_audit_cli_prints_sorted_matches_to_stderr_and_exits_one(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "inventory.json"
    _write_inventory(inventory, "av._core")

    result = _run_cli("--inventory", str(inventory), cwd=tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert lines == sorted(set(lines))
    assert any("av._core" in line.casefold() for line in lines)


@pytest.mark.parametrize("mode", ("missing-inventory", "malformed", "missing-dist"))
def test_media_audit_cli_fails_closed_on_missing_or_malformed_inputs(
    tmp_path: Path,
    mode: str,
) -> None:
    inventory = tmp_path / "inventory.json"
    args = ["--inventory", str(inventory)]
    if mode == "malformed":
        inventory.write_text("{not-json", encoding="utf-8")
    elif mode == "missing-dist":
        _write_inventory(inventory, "assets/avatar.py")
        args.extend(("--dist", str(tmp_path / "missing-dist")))

    result = _run_cli(*args, cwd=tmp_path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("bundle media audit error:")
    assert "Traceback" not in result.stderr


def test_media_audit_cli_requires_inventory_argument(tmp_path: Path) -> None:
    result = _run_cli(cwd=tmp_path)
    assert result.returncode != 0
    assert "--inventory" in result.stderr


def test_analysis_excludes_exact_pyav_import_boundaries() -> None:
    tree = ast.parse(
        (_REPO_ROOT / "installer" / "yamabiko.spec").read_text(encoding="utf-8")
    )
    analysis_calls = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "a" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Analysis"
    ]
    assert len(analysis_calls) == 1
    excludes_keywords = [
        keyword.value
        for keyword in analysis_calls[0].keywords
        if keyword.arg == "excludes"
    ]
    assert len(excludes_keywords) == 1
    excludes_node = excludes_keywords[0]
    assert isinstance(excludes_node, (ast.List, ast.Tuple))
    excludes = [
        element.value
        for element in excludes_node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    assert excludes.count("faster_whisper.audio") == 1
    assert excludes.count("av") == 1
