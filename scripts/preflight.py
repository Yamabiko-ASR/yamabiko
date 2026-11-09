"""Release preflight - exercise every external touchpoint fast.

Single script, no user interaction. Runs every check independently and
prints a pass/fail summary at the end. Exits non-zero on any failure.

Designed to catch the class of bugs that the beta shipped with: mocked
unit tests drifting from reality (HF repo renames, hf_hub_download
signature changes, audio stack regressions, PyInstaller spec omissions).

Run (dev venv):
    .venv\\Scripts\\python.exe scripts\\preflight.py
Release gate, from the release build venv (docs/RELEASE.md):
    .venv-release\\Scripts\\python.exe scripts\\preflight.py --release --venv .venv-release

--venv names the build venv, as installer\\build.ps1 -Venv does (a folder
under the repo root or an absolute path; default .venv): sections 8 and 9
run its interpreter and section 15 fingerprints its site-packages. An
explicit --venv without an interpreter fails them; only the default .venv
falls back to the running Python. With --release, preflight must itself run
from that venv's interpreter.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import importlib
import inspect
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Japanese Windows ships with cp932 as the default console codec; our
# output is ASCII-only but device names from pyaudiowpatch may contain
# non-Latin characters that crash cp932 encoding. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]

# The build venv (--venv; installer\build.ps1 -Venv): a folder name under
# ROOT or an absolute path, resolved at call time so tests can move ROOT.
DEFAULT_VENV = ".venv"

# Make the repo-root importable. When preflight is invoked as
# ``python scripts/preflight.py`` from the repo root, ``sys.path[0]``
# becomes ``scripts/`` (the script's directory), NOT the cwd, so
# ``import yamabiko`` fails. The LAZY_IMPORTS sweep below now includes
# first-party ``yamabiko.lifecycle.*`` modules, so it would spuriously
# fail without this. Inserting ROOT matches how the frozen bundle and
# the dev venv both see the package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ALL repo / file / revision pins come from the single source of truth so
# the release gate verifies the SAME files the wizard downloads and the
# runtime resolves -- otherwise a hardcoded copy re-introduces
# producer/consumer drift in the gate itself.
from yamabiko.models import (  # noqa: E402
    TRANSLATOR_PINS,
    WHISPER_REPO, WHISPER_FILES, WHISPER_REVISION,
    SILERO_REPO, SILERO_FILE, SILERO_REVISION,
)

# --- Configuration ---------------------------------------------------------

# (repo_id, revision, [expected filenames]) -- one entry per translator
# tier pin (the tiers may live in different repos), then whisper + silero.
HF_CHECKS: List[Tuple[str, Optional[str], List[str]]] = [
    *((pin.repo, pin.revision, [pin.filename])
      for pin in TRANSLATOR_PINS.values()),
    (WHISPER_REPO, WHISPER_REVISION, list(WHISPER_FILES)),
    (SILERO_REPO, SILERO_REVISION, [SILERO_FILE]),
]

# hiddenimports declared in the PyInstaller spec.
LAZY_IMPORTS = [
    "huggingface_hub", "hf_xet", "onnxruntime", "tokenizers",
    "keyboard", "cutlet", "fugashi", "sudachipy",
    "fastapi", "uvicorn", "httpx", "yaml", "numpy",
    "PySide6.QtNetwork", "PySide6.QtWebSockets",
    # yamabiko.lifecycle.* — process-lifecycle overhaul modules used
    # across watchdog / main / overlay_app. PyInstaller's static
    # analyzer currently picks them up via the watchdog entrypoint's
    # import chain, but a future refactor that moves to a lazy
    # importlib.import_module call would silently drop them from
    # the frozen bundle. Listing them here makes the bundle-vs-dev
    # divergence loud. Hard rule #7.
    "yamabiko.lifecycle.jobs",
    "yamabiko.lifecycle.shutdown_event",
    "yamabiko.lifecycle.exit_codes",
    "yamabiko.lifecycle.pipeline_supervisor",
]

# Required kwargs our code passes to hf_hub_download. local_files_only
# joined the set so silero_chunker can skip the
# per-startup online revalidation when the cache is warm.
HF_REQUIRED_KWARGS = {"repo_id", "filename", "revision", "local_dir", "local_files_only"}

# Spec must declare these as hiddenimports — without them, the frozen
# bundle fails at runtime on the lazy import sites.
SPEC_REQUIRED_HIDDENIMPORTS = [
    "faster_whisper",
    "hf_xet",
    "huggingface_hub",
    "onnxruntime",
    "tokenizers",
]

# Section 8 runs before a frozen candidate is required. Tests whose contract
# needs an already-built ``dist/yamabiko`` tree live behind the separate
# ``frozen_artifact`` post-build gate in docs/RELEASE.md and CI.
LIGHTWEIGHT_REAL_MARKER_EXPR = "real and not frozen_artifact"


# --- Result collection -----------------------------------------------------

class Section:
    def __init__(self, num: int, title: str) -> None:
        self.num = num
        self.title = title
        self.lines: List[Tuple[str, str]] = []  # (status, message)

    def add(self, status: str, message: str) -> None:
        self.lines.append((status, message))

    def failed(self) -> bool:
        return any(status == "FAIL" for status, _ in self.lines)

    def skipped(self) -> bool:
        """True when this section did nothing meaningful: it has a SKIP
        line and no PASS or FAIL. The final report uses this to refuse
        the all-sections-passed banner when a bundle-dependent section
        skipped."""
        statuses = {status for status, _ in self.lines}
        return "SKIP" in statuses and "PASS" not in statuses and "FAIL" not in statuses

    def emit(self) -> None:
        print(f"[{self.num}] {self.title}")
        for status, message in self.lines:
            print(f"    {status:<5} {message}")
        print()


# --- Checks ----------------------------------------------------------------

def check_hf_repos() -> Section:
    sec = Section(1, "HuggingFace repo + file existence")
    try:
        from huggingface_hub import list_repo_files
    except Exception as exc:  # pragma: no cover — huggingface_hub should be in venv
        sec.add("FAIL", f"cannot import huggingface_hub: {exc!r}")
        return sec

    for repo_id, revision, expected in HF_CHECKS:
        label = f"{repo_id}" + (f"@{revision}" if revision else "")
        try:
            files = set(list_repo_files(repo_id, revision=revision))
        except Exception as exc:
            sec.add("FAIL", f"{label}: list_repo_files raised {type(exc).__name__}: {exc}")
            continue
        missing = [f for f in expected if f not in files]
        if missing:
            sec.add("FAIL", f"{label}: missing {missing}")
        else:
            sec.add("PASS", f"{label} ({len(expected)} files)")
    return sec


def _classify_downloads_tqdm_usage(src: str) -> Tuple[str, str]:
    """Classify downloads.py's hf_hub_download progress wiring.

    huggingface_hub >=1.1.0 made ``tqdm_class=`` the deliberate design
    for custom progress reporting (hubs without the kwarg get the
    huggingface_hub.utils.tqdm rebind fallback instead). The guard mirrors
    tests/test_real_hf_signature.py::test_installed_hub_delivers_progress_through_downloads_hook:
    that test pins the library side delivering byte ticks; this pins the
    call site still wiring tqdm_class for the >=1.1.0 branch (dropping it
    silently disconnects wizard byte-progress on the bundle's hub).
    """
    if "hf_hub_download" not in src:
        return ("FAIL",
                "downloads.py no longer calls hf_hub_download -- section 2's "
                "call-site guard has lost its target; update the guard")
    if "tqdm_class" not in src:
        return ("FAIL",
                "downloads.py calls hf_hub_download without tqdm_class -- "
                "wizard progress reporting is disconnected (see "
                "test_real_hf_signature.py::"
                "test_installed_hub_delivers_progress_through_downloads_hook)")
    return ("PASS", "downloads.py passes tqdm_class to hf_hub_download")


def check_hf_download_signature() -> Section:
    sec = Section(2, "hf_hub_download signature")
    try:
        from huggingface_hub import hf_hub_download
        sig = inspect.signature(hf_hub_download)
    except Exception as exc:
        sec.add("FAIL", f"cannot inspect hf_hub_download: {exc!r}")
        return sec

    present = set(sig.parameters)
    missing = HF_REQUIRED_KWARGS - present
    if missing:
        sec.add("FAIL", f"hf_hub_download missing kwargs: {sorted(missing)}")
    else:
        sec.add("PASS", f"{', '.join(sorted(HF_REQUIRED_KWARGS))} all present")

    downloads_path = ROOT / "yamabiko" / "wizard_pages" / "downloads.py"
    try:
        src = downloads_path.read_text(encoding="utf-8")
    except OSError as exc:
        sec.add("FAIL", f"cannot read {downloads_path}: {exc}")
        return sec

    sec.add(*_classify_downloads_tqdm_usage(src))
    return sec


def _extract(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _numeric_quad(app_version: str) -> Optional[str]:
    """APP_VERSION -> the 4-part binary quad: numeric X.Y.Z prefix + a
    ".0" build part ("0.1.0-beta" -> "0.1.0.0"). None when APP_VERSION
    has no X.Y.Z numeric prefix."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", app_version)
    if m is None:
        return None
    return ".".join(m.groups()) + ".0"


def _binary_versioninfo_lines(nsi_src: str, verinfo_src: str) -> List[Tuple[str, str]]:
    """PASS/FAIL lines for binary VERSIONINFO consistency.

    The PE version resources are 4-numeric quads that cannot carry the
    "-beta" pre-release suffix, so they live apart from the APP_VERSION
    string: APP_VERSION_QUAD in the .nsi (consumed by VIProductVersion /
    VIFileVersion) and the filevers/prodvers tuples in
    installer/version_info.txt (consumed via yamabiko.spec
    EXE(version=...)). build.ps1 passes no /D overrides, so nothing
    derives them at build time -- a version bump that edits only the
    string sites ships a stale binary VERSIONINFO, a string/binary
    version mismatch that AV / reputation tooling reads off the PE.
    Enforced so that drift is unshippable:
      (a) APP_VERSION_QUAD == numeric prefix of APP_VERSION + ".0"
      (b) VIProductVersion / VIFileVersion resolve to that quad
          (define reference or literal -- checked by value)
      (c) version_info.txt filevers + prodvers tuples equal that quad
      (d) version_info.txt FileVersion / ProductVersion strings equal
          APP_VERSION
    """
    lines: List[Tuple[str, str]] = []
    app_version = _extract(r'!define\s+APP_VERSION\s+"([^"]+)"', nsi_src)
    if app_version is None:
        return [("FAIL",
                 "cannot extract APP_VERSION from installer/yamabiko.nsi; "
                 "binary VERSIONINFO consistency unverifiable")]
    expected = _numeric_quad(app_version)
    if expected is None:
        return [("FAIL",
                 f"APP_VERSION {app_version!r} has no X.Y.Z numeric prefix; "
                 f"cannot derive the binary VERSIONINFO quad")]
    expected_tuple = tuple(int(p) for p in expected.split("."))

    quad = _extract(r'!define\s+APP_VERSION_QUAD\s+"([^"]+)"', nsi_src)
    if quad is None:
        lines.append((
            "FAIL",
            f"installer/yamabiko.nsi has no APP_VERSION_QUAD define -- the "
            f"binary VERSIONINFO quad needs a single source of truth next "
            f"to APP_VERSION (expected {expected!r})"))
    elif quad != expected:
        lines.append((
            "FAIL",
            f"installer/yamabiko.nsi APP_VERSION_QUAD {quad!r} != "
            f"{expected!r} (numeric prefix of APP_VERSION {app_version!r} "
            f"+ '.0') -- the installer PE would ship a stale binary version"))

    for key in ("VIProductVersion", "VIFileVersion"):
        val = _extract(rf'(?m)^\s*{key}\s+"([^"]+)"', nsi_src)
        if val is None:
            lines.append(("FAIL", f"installer/yamabiko.nsi has no {key} statement"))
            continue
        resolved = quad if val == "${APP_VERSION_QUAD}" else val
        if resolved is None:
            continue  # references the define; its absence already FAILed above
        if resolved != expected:
            lines.append((
                "FAIL",
                f"installer/yamabiko.nsi {key} resolves to {resolved!r}, "
                f"expected {expected!r}"))

    for name in ("filevers", "prodvers"):
        m = re.search(rf"{name}\s*=\s*\(([^)]*)\)", verinfo_src)
        if m is None:
            lines.append((
                "FAIL",
                f"installer/version_info.txt has no {name}=(...) tuple"))
            continue
        try:
            got = tuple(int(p.strip()) for p in m.group(1).split(","))
        except ValueError:
            lines.append((
                "FAIL",
                f"installer/version_info.txt {name} tuple is not numeric: "
                f"({m.group(1)})"))
            continue
        if got != expected_tuple:
            lines.append((
                "FAIL",
                f"installer/version_info.txt {name}={got} != "
                f"{expected_tuple} (from APP_VERSION {app_version!r}) -- "
                f"yamabiko.exe would ship a stale binary version"))

    for key in ("FileVersion", "ProductVersion"):
        m = re.search(rf"StringStruct\(\s*'{key}'\s*,\s*'([^']*)'\s*\)",
                      verinfo_src)
        if m is None:
            lines.append((
                "FAIL",
                f"installer/version_info.txt has no StringStruct('{key}', ...)"))
            continue
        if m.group(1) != app_version:
            lines.append((
                "FAIL",
                f"installer/version_info.txt {key} {m.group(1)!r} != "
                f"APP_VERSION {app_version!r}"))

    if not lines:
        lines.append((
            "PASS",
            f"binary VERSIONINFO consistent: quad {expected!r} (nsi "
            f"APP_VERSION_QUAD -> VIProductVersion/VIFileVersion; "
            f"version_info.txt filevers/prodvers), strings match "
            f"APP_VERSION {app_version!r}"))
    return lines


def check_version_consistency() -> Section:
    sec = Section(3, "Version consistency (strings + binary VERSIONINFO)")
    versions: List[Tuple[str, Optional[str]]] = []

    init_path = ROOT / "yamabiko" / "__init__.py"
    try:
        init_src = init_path.read_text(encoding="utf-8")
        versions.append(("yamabiko/__init__.py",
                         _extract(r'__version__\s*=\s*"([^"]+)"', init_src)))
    except OSError as exc:
        sec.add("FAIL", f"cannot read {init_path}: {exc}")

    changelog_path = ROOT / "CHANGELOG.md"
    try:
        changelog_src = changelog_path.read_text(encoding="utf-8")
        # Skip [Unreleased]; find the first concrete versioned header.
        matches = re.findall(r'^##\s*\[([^\]]+)\]', changelog_src, flags=re.MULTILINE)
        concrete = next((v for v in matches if v.lower() != "unreleased"), None)
        versions.append(("CHANGELOG.md", concrete))
    except OSError as exc:
        sec.add("FAIL", f"cannot read {changelog_path}: {exc}")

    nsi_path = ROOT / "installer" / "yamabiko.nsi"
    nsi_src: Optional[str] = None
    try:
        nsi_src = nsi_path.read_text(encoding="utf-8")
        versions.append(("installer/yamabiko.nsi",
                         _extract(r'!define\s+APP_VERSION\s+"([^"]+)"', nsi_src)))
    except OSError as exc:
        sec.add("FAIL", f"cannot read {nsi_path}: {exc}")

    inst_readme_path = ROOT / "installer" / "README.md"
    try:
        inst_readme_src = inst_readme_path.read_text(encoding="utf-8")
        # Most-recent concrete `yamabiko-X.Y.Z-tag-setup.exe` reference.
        m = re.search(r'yamabiko-([0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.]+)?)-setup\.exe',
                      inst_readme_src)
        # If only the placeholder `<version>` appears, treat as consistent
        # with whatever the other three agree on (docs deliberately use a
        # placeholder).
        versions.append(("installer/README.md", m.group(1) if m else None))
    except OSError as exc:
        sec.add("FAIL", f"cannot read {inst_readme_path}: {exc}")

    found: List[Tuple[str, str]] = []
    missing_sources: List[str] = []
    for source, version in versions:
        if version is None:
            missing_sources.append(source)
        else:
            found.append((source, version))

    unique = {v for _, v in found}
    if not found:
        sec.add("FAIL", "no version strings could be extracted")
    elif len(unique) == 1:
        single = next(iter(unique))
        label = f"{len(found)}/{len(versions)}" if missing_sources else str(len(found))
        if missing_sources:
            sec.add("PASS",
                    f"{label} locations agree on {single!r} "
                    f"(placeholder in: {', '.join(missing_sources)})")
        else:
            sec.add("PASS", f"all {label} locations agree on {single!r}")
    else:
        detail = ", ".join(f"{s}={v!r}" for s, v in found)
        sec.add("FAIL", f"version mismatch: {detail}")

    # Binary VERSIONINFO quads (see _binary_versioninfo_lines): the
    # string checks above cannot see them, and a bump that edits only
    # the strings would ship a stale PE version resource.
    if nsi_src is not None:
        verinfo_path = ROOT / "installer" / "version_info.txt"
        try:
            verinfo_src = verinfo_path.read_text(encoding="utf-8")
        except OSError as exc:
            sec.add("FAIL", f"cannot read {verinfo_path}: {exc}")
        else:
            for status, msg in _binary_versioninfo_lines(nsi_src, verinfo_src):
                sec.add(status, msg)

    return sec


def check_lazy_imports() -> Section:
    sec = Section(4, "Lazy imports")
    successes: List[str] = []
    failures: List[Tuple[str, str]] = []
    for name in LAZY_IMPORTS:
        try:
            importlib.import_module(name)
            successes.append(name)
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    if not failures:
        sec.add("PASS", f"{len(successes)}/{len(LAZY_IMPORTS)} imported")
    else:
        # Previously this branch also emitted a PASS line before the FAILs,
        # which produced misleading output ("14/14 imported" next to a FAIL).
        # Section.failed() already drives exit code off any FAIL entry; we
        # just need to list the failures.
        for name, detail in failures:
            sec.add("FAIL", f"{name}: {detail}")
        sec.add("INFO", f"{len(successes)}/{len(LAZY_IMPORTS)} imported, {len(failures)} failed")
    return sec


def check_llama_server() -> Section:
    sec = Section(5, "Vendored llama-server")
    exe = ROOT / "third_party" / "llama.cpp" / "llama-server.exe"
    if not exe.exists():
        sec.add("SKIP", f"{exe} not present (dev setup without vendored binary)")
        return sec
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        sec.add("FAIL", "llama-server --version timed out after 5 s")
        return sec
    except OSError as exc:
        sec.add("FAIL", f"could not execute llama-server: {exc}")
        return sec

    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        sec.add("FAIL", f"llama-server exited {result.returncode}: {combined.strip()[:160]}")
        return sec
    version_line = next((ln for ln in combined.splitlines() if "version" in ln.lower()), None)
    if version_line is None:
        sec.add("FAIL", f"no 'version' line in output: {combined.strip()[:160]}")
    else:
        sec.add("PASS", f"version line: {version_line.strip()!r}")
    return sec


def check_audio_devices() -> Section:
    sec = Section(6, "Audio devices")
    try:
        import pyaudiowpatch  # type: ignore
    except Exception as exc:
        sec.add("FAIL", f"cannot import pyaudiowpatch: {exc!r}")
        return sec

    try:
        p = pyaudiowpatch.PyAudio()
    except Exception as exc:
        sec.add("FAIL", f"PyAudio() raised: {exc!r}")
        return sec

    try:
        loopbacks = list(p.get_loopback_device_info_generator())
    except Exception as exc:
        p.terminate()
        sec.add("FAIL", f"get_loopback_device_info_generator raised: {exc!r}")
        return sec
    finally:
        try:
            p.terminate()
        except Exception:
            pass

    if not loopbacks:
        sec.add("FAIL", "no WASAPI loopback devices enumerated")
    else:
        names = [info["name"] for info in loopbacks[:3]]
        sec.add("PASS", f"{len(loopbacks)} loopback devices; sample: {names}")
    return sec


def _is_analysis_call(node: ast.AST) -> bool:
    """Whether ``node`` calls either bare ``Analysis`` or ``*.Analysis``."""
    if not isinstance(node, ast.Call):
        return False
    return (
        isinstance(node.func, ast.Name) and node.func.id == "Analysis"
    ) or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "Analysis"
    )


def _node_binds_name_by_string(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    name: str,
) -> bool:
    """Catch Python binders whose target is stored as text, not ast.Name."""
    if isinstance(node, ast.alias):
        parent = parents.get(node)
        if isinstance(parent, ast.Import):
            bound = node.asname or node.name.split(".", 1)[0]
        elif isinstance(parent, ast.ImportFrom):
            bound = node.asname or node.name
        else:
            return False
        return bound == name
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.ExceptHandler):
        return node.name == name
    if isinstance(node, ast.arg):
        return node.arg == name
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return name in node.names
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return node.name == name
    if isinstance(node, ast.MatchMapping):
        return node.rest == name
    return False


def _node_references_or_binds_name(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    name: str,
) -> bool:
    """Catch direct, attribute, imported, and text-stored uses of ``name``."""
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.Attribute):
        return node.attr == name
    if isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] == name:
        return True
    return _node_binds_name_by_string(node, parents, name)


def _direct_spec_hiddenimports(tree: ast.Module) -> Tuple[Optional[set[str]], Optional[str]]:
    """Extract auditable direct strings from the spec's hiddenimports list.

    PyInstaller specs are executable Python, but preflight must not execute the
    release spec just to inspect it. Therefore the accepted shape is purposely
    narrow: exactly one module-level simple assignment to a list literal. List
    elements may be direct string constants or starred helper collections (the
    current spec expands hook-generated lists that way). Every other shape is
    rejected as unverifiable rather than guessed at. The assigned name may be
    loaded exactly once by the one canonical, executed module-level
    ``a = Analysis(hiddenimports=hiddenimports)`` assignment; every other
    exact-name use anywhere else could mutate, alias, or rebind the audited
    list, so the gate intentionally fails closed without trying to model
    Python's nested scopes.
    """
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "hiddenimports"
    ]

    if not assignments:
        return None, "spec hiddenimports assignment is missing"
    if len(assignments) != 1:
        return None, (
            f"spec has multiple hiddenimports assignments ({len(assignments)}); "
            "membership is unverifiable"
        )

    assignment = assignments[0]
    value = assignment.value
    if not isinstance(value, ast.List):
        return None, (
            "spec hiddenimports assignment must be a list literal containing "
            f"direct string elements; got {type(value).__name__}"
        )

    direct: set[str] = set()
    for element in value.elts:
        if isinstance(element, ast.Starred):
            continue
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            direct.add(element.value)
            continue
        return None, (
            "spec hiddenimports list contains an unverifiable element at "
            f"line {getattr(element, 'lineno', '?')}; only direct string "
            "elements and starred helper collections are allowed"
        )

    assignment_target = assignment.targets[0]
    analysis_calls = [node for node in ast.walk(tree) if _is_analysis_call(node)]
    if len(analysis_calls) != 1:
        return None, (
            "spec must contain exactly one Analysis call: the canonical "
            "module-level a = Analysis(..., hiddenimports=hiddenimports, ...); "
            f"found {len(analysis_calls)} Analysis/.Analysis call(s)"
        )

    analysis_call = analysis_calls[0]
    analysis_assignment = parents.get(analysis_call)
    hiddenimports_keywords = [
        keyword
        for keyword in analysis_call.keywords
        if keyword.arg == "hiddenimports"
    ]
    is_canonical_consumer = (
        isinstance(analysis_assignment, ast.Assign)
        and analysis_assignment in tree.body
        and analysis_assignment.value is analysis_call
        and len(analysis_assignment.targets) == 1
        and isinstance(analysis_assignment.targets[0], ast.Name)
        and analysis_assignment.targets[0].id == "a"
        and isinstance(analysis_call.func, ast.Name)
        and analysis_call.func.id == "Analysis"
        and len(hiddenimports_keywords) == 1
        and isinstance(hiddenimports_keywords[0].value, ast.Name)
        and hiddenimports_keywords[0].value.id == "hiddenimports"
        and isinstance(hiddenimports_keywords[0].value.ctx, ast.Load)
        and analysis_assignment.lineno > assignment.lineno
    )
    if not is_canonical_consumer:
        return None, (
            "spec hiddenimports must be passed to Analysis exactly once by "
            "one direct, module-level "
            "a = Analysis(..., hiddenimports=hiddenimports, ...) assignment "
            "after the audited list; conditional, nested, attribute, renamed, "
            "missing, duplicate, or computed consumers are unverifiable"
        )

    analysis_target = analysis_assignment.targets[0]
    unsafe_a_bindings = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and node.id == "a"
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node is not analysis_target
        )
        or _node_binds_name_by_string(node, parents, "a")
    ]
    if unsafe_a_bindings:
        lines = sorted(
            {getattr(node, "lineno", "?") for node in unsafe_a_bindings}, key=str
        )
        return None, (
            "spec Analysis result a must have exactly one store: the canonical "
            "a = Analysis(...) target; any other exact-name store, delete, or "
            f"text-stored binding is unverifiable (line(s) {lines})"
        )

    noncanonical_analysis_refs = [
        node
        for node in ast.walk(tree)
        if node is not analysis_call.func
        and _node_references_or_binds_name(node, parents, "Analysis")
    ]
    if noncanonical_analysis_refs:
        lines = sorted(
            {getattr(node, "lineno", "?") for node in noncanonical_analysis_refs},
            key=str,
        )
        return None, (
            "spec has a noncanonical Analysis reference, binding, or alias at "
            f"line(s) {lines}; only the direct function of the one module-level "
            "a = Analysis(...) assignment is allowed"
        )

    reflective_names = {
        "exec",
        "eval",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__dict__",
        "__setattr__",
        "__delattr__",
        "f_globals",
        "f_locals",
        "__globals__",
    }
    critical_names = {"hiddenimports", "Analysis", "a"}
    canonical_keyword = hiddenimports_keywords[0]
    namespace_escapes = [
        node
        for node in ast.walk(tree)
        if any(
            _node_references_or_binds_name(node, parents, name)
            for name in reflective_names
        )
        or (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value in critical_names or node.value in reflective_names)
        )
        or (isinstance(node, ast.Attribute) and node.attr in critical_names)
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
        )
        or (
            isinstance(node, ast.keyword)
            and node.arg in critical_names
            and node is not canonical_keyword
        )
    ]
    if namespace_escapes:
        lines = sorted(
            {getattr(node, "lineno", "?") for node in namespace_escapes}, key=str
        )
        return None, (
            "spec hiddenimports has a reflective, wildcard-import, critical-"
            f"keyword, or dynamic namespace escape at line(s) {lines}; static "
            "membership proof is impossible"
        )

    canonical_name = hiddenimports_keywords[0].value
    unsafe_uses = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and node.id == "hiddenimports"
            and node is not assignment_target
            and node is not canonical_name
        )
        or _node_binds_name_by_string(node, parents, "hiddenimports")
    ]

    if unsafe_uses:
        lines = sorted({getattr(node, "lineno", "?") for node in unsafe_uses}, key=str)
        return None, (
            "spec hiddenimports has an unverifiable exact-name mutation, "
            f"rebinding, use, or binding at line(s) {lines}; the audited name "
            "may only be assigned once and passed directly to Analysis"
        )
    return direct, None


def check_pyinstaller_spec() -> Section:
    sec = Section(7, "PyInstaller spec parse")
    spec_path = ROOT / "installer" / "yamabiko.spec"
    try:
        src = spec_path.read_text(encoding="utf-8")
    except OSError as exc:
        sec.add("FAIL", f"cannot read {spec_path}: {exc}")
        return sec

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        sec.add("FAIL", f"ast.parse raised SyntaxError: {exc}")
        return sec

    try:
        compile(tree, str(spec_path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        sec.add("FAIL", f"spec compile raised SyntaxError: {exc}")
        return sec

    direct, error = _direct_spec_hiddenimports(tree)
    if error is not None:
        sec.add("FAIL", error)
        return sec

    assert direct is not None
    missing = [name for name in SPEC_REQUIRED_HIDDENIMPORTS if name not in direct]
    if missing:
        sec.add("FAIL", f"spec hiddenimports missing: {missing}")
    else:
        sec.add(
            "PASS",
            "ast ok, hiddenimports assignment auditable and required direct "
            "strings complete",
        )
    return sec


# --- Main ------------------------------------------------------------------

def _parse_pytest_counts(tail: str) -> dict:
    """Parse a pytest summary tail line into ``{passed, failed, skipped}``.

    Handles the usual shapes -- ``= 12 passed, 3 skipped in 45s =``,
    ``= 2 failed, 9 passed, 26 skipped, 571 deselected in 82s =`` -- with
    missing categories defaulting to 0."""
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for key in counts:
        m = re.search(rf"(\d+) {key}\b", tail)
        if m:
            counts[key] = int(m.group(1))
    return counts


def _classify_real_tier(returncode: int, stdout: str, release: bool) -> Tuple[str, str]:
    """Decide the Section-8 status from a lightweight real-tier result.

    returncode 0 with ZERO tests actually run -- everything skipped for
    lack of GPU / warmed cache / built bundle -- is NOT a pass. In
    --release mode that is a FAIL (a release-grade preflight must
    exercise the real tier); in dev mode it is an honest SKIP (invisible
    to the gate), never a green PASS. A run that executed tests reports
    its passed/skipped counts so a vacuous run is visibly different from
    a real one."""
    tail = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if returncode != 0:
        return ("FAIL",
                f"real-path suite FAILED (rc={returncode}): {tail or '(no output)'}. "
                f"Rerun verbosely: pytest -m \"{LIGHTWEIGHT_REAL_MARKER_EXPR}\" -v")
    counts = _parse_pytest_counts(tail)
    if counts["passed"] == 0:
        detail = (f"0 tests ran ({counts['skipped']} skipped) -- "
                  f"needs GPU + warmed cache + built bundle")
        if release:
            return ("FAIL", f"real-tier {detail}; a --release run must exercise it")
        return ("SKIP", f"real-tier {detail}; not release-grade")
    return ("PASS",
            f"real-path: {counts['passed']} passed, {counts['skipped']} skipped")


def _venv_dir(venv: str) -> Path:
    """The build venv ``venv`` as a path: absolute as given, else under
    ROOT (as build.ps1 resolves -Venv against the repo root)."""
    path = Path(venv)
    return path if path.is_absolute() else ROOT / path


def _venv_python(venv: str) -> Optional[str]:
    """The build venv's interpreter. Only the default .venv falls back to
    this interpreter when it has none (a contributor box without one); an
    explicit ``--venv`` without one is None, which the section reports as a
    FAIL (see ``_missing_venv_python``) rather than testing another venv."""
    candidate = _venv_dir(venv) / "Scripts" / "python.exe"
    if candidate.is_file():
        return str(candidate)
    return sys.executable if venv == DEFAULT_VENV else None


def _missing_venv_python(venv: str) -> str:
    """The FAIL line for an explicit ``--venv`` that has no interpreter."""
    return (
        f"--venv {venv} has no interpreter at "
        f"{_venv_dir(venv) / 'Scripts' / 'python.exe'}; create the venv "
        f"(docs/RELEASE.md) or pass the right --venv"
    )


def check_real_path_pytest(release: bool = False, venv: str = DEFAULT_VENV) -> "Section":
    """Run the pre-build-safe real tier and fold the result into preflight.

    The real-path tests live at ``tests/test_real_*.py`` (plus
    ``tests/asr/test_real_*.py``) and exercise every external-dep seam
    that has already caused a shipped bug — FasterWhisperASR,
    pa.open(stream_callback=), onnxruntime.InferenceSession for both
    Silero + LiveKit, uvicorn.Server.serve(), and the frozen PyInstaller
    bundle. The non-skippable test that requires an already-built candidate is
    marked ``frozen_artifact`` and runs separately after the build.

    Tests skip gracefully when the environment can't exercise them
    (no GPU, no audio, model not cached, no built bundle). In dev mode an
    all-skipped run is reported as a SKIP, not a pass; in ``--release``
    mode it is a FAIL (see ``_classify_real_tier``).

    ``venv`` is the build venv whose interpreter runs the tier (``--venv``).
    """
    sec = Section(
        8,
        f"Real-path integration tests (pytest -m {LIGHTWEIGHT_REAL_MARKER_EXPR!r})",
    )
    py = _venv_python(venv)
    if py is None:
        sec.add("FAIL", _missing_venv_python(venv))
        return sec
    # Fail-fast: this section hard-passes --timeout=180. On a fresh
    # box provisioned with only `pip install -r yamabiko/requirements.txt`
    # without pytest-timeout installed, pytest exits 4 ("unrecognized
    # arguments: --timeout=180") and the classifier below would report
    # "real-path suite FAILED" — indistinguishable from genuine test
    # failures. Probe the plugin in the TARGET interpreter (which may
    # not be the one running preflight) and name the real problem.
    try:
        probe = subprocess.run(
            [py, "-c", "import pytest_timeout"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        sec.add("FAIL", f"pytest-timeout import probe failed to run: {exc}")
        return sec
    if probe.returncode != 0:
        sec.add(
            "FAIL",
            "pytest-timeout is not installed in the test interpreter; "
            "section 8 passes --timeout=180 and pytest would exit 4 "
            "('unrecognized arguments'). Fix: pip install -r "
            "requirements-dev.txt (pins pytest-timeout>=2.4).",
        )
        return sec
    try:
        proc = subprocess.run(
            [
                py,
                "-m",
                "pytest",
                "-m",
                LIGHTWEIGHT_REAL_MARKER_EXPR,
                "--timeout=180",
                "-q",
                "--no-header",
            ],
            cwd=str(ROOT),
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        sec.add(
            "FAIL",
            f"timed out after 5 min running pytest -m "
            f"{LIGHTWEIGHT_REAL_MARKER_EXPR!r}",
        )
        return sec
    except OSError as exc:
        sec.add("FAIL", f"pytest invocation failed: {exc}")
        return sec

    status, msg = _classify_real_tier(proc.returncode, proc.stdout, release)
    sec.add(status, msg)
    return sec


def _classify_bundle_audit_missing(release: bool = False) -> Tuple[str, str]:
    """The Section-9 no-bundle decision. In --release mode the absence of a
    built bundle is a FAIL -- you cannot release-gate a bundle that was
    never built; in dev mode it is a SKIP."""
    if release:
        return ("FAIL",
                "no bundle at dist/yamabiko -- a --release run requires a built "
                "bundle (run installer\\build.ps1 first)")
    return ("SKIP", "no bundle at dist/yamabiko (run installer\\build.ps1 first)")


def check_bundle_dll_audit(release: bool = False, venv: str = DEFAULT_VENV) -> "Section":
    """Run scripts/audit_bundle_dlls.py against dist/yamabiko/.

    Deep companion to section 8's pytest-based frozen-bundle DLL test.
    The pytest test checks presence of a hardcoded required-DLL list;
    this audit walks every native file's PE import table + runtime
    LoadLibrary string references and reports anything imported but not
    bundled / system / driver. Catches the "new transitive dep silently
    added by a wheel upgrade" class that a hardcoded list misses --
    the missing-transitive-dependency class.

    audit_bundle_dlls.py exit codes: 0 = no missing, 1 = missing,
    2 = no bundle.  This section translates those into PASS / FAIL / SKIP.
    """
    sec = Section(9, "Bundle DLL deep audit")
    bundle = ROOT / "dist" / "yamabiko"
    if not bundle.is_dir():
        status, msg = _classify_bundle_audit_missing(release)
        sec.add(status, msg)
        return sec

    audit_script = ROOT / "scripts" / "audit_bundle_dlls.py"
    py = _venv_python(venv)
    if py is None:
        sec.add("FAIL", _missing_venv_python(venv))
        return sec
    try:
        proc = subprocess.run(
            [py, str(audit_script)],
            cwd=str(ROOT),
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        sec.add("FAIL", "audit_bundle_dlls.py timed out after 2 min")
        return sec
    except OSError as exc:
        sec.add("FAIL", f"audit invocation failed: {exc}")
        return sec

    if proc.returncode == 0:
        header = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("Native files:")),
            "",
        )
        sec.add("PASS", f"no missing DLL deps. {header.strip()}")
        return sec
    if proc.returncode == 2:
        # Race: bundle existed at entry but audit reports missing.
        sec.add("SKIP", "audit reports no bundle (build race?)")
        return sec

    # rc=1 (or other nonzero): extract the missing-DLL names for a terse
    # preview so the operator doesn't have to rerun the audit immediately
    # to see the scope. Keep parsing defensive; the audit's human-readable
    # output could drift and we don't want parsing to crash the gate.
    missing_names: List[str] = []
    in_missing = False
    for ln in proc.stdout.splitlines():
        if ln.startswith("MISSING:"):
            in_missing = True
            continue
        if ln.startswith("PRESENT-BUT-DUPLICATE"):
            in_missing = False
        if not in_missing:
            continue
        stripped = ln.strip()
        # Lines inside the MISSING block look like:
        #   cublas64_12.dll  <- ctranslate2.dll, ...
        # Skip category headers like "[CUDA runtime / math libs]" and
        # the "(none)" sentinel.
        if not stripped or stripped.startswith("[") or stripped.startswith("("):
            continue
        first = stripped.split()[0]
        if "." in first and not first.startswith("="):
            missing_names.append(first)

    if missing_names:
        preview = ", ".join(missing_names[:5])
        if len(missing_names) > 5:
            preview += f", +{len(missing_names) - 5} more"
        sec.add(
            "FAIL",
            f"{len(missing_names)} missing DLL dep(s) (rc={proc.returncode}): "
            f"{preview}. Full list: python scripts/audit_bundle_dlls.py",
        )
        return sec
    # No MISSING block on stdout: the audit's gutted-bundle FAILs (empty
    # dist, exe-less dist) print their reason to stderr with nothing on
    # stdout, so surface that reason instead of sending the operator off
    # to rerun the audit blind.
    err_tail = " ".join(
        ln.strip() for ln in proc.stderr.strip().splitlines()[-3:] if ln.strip()
    )
    reason = (f"stderr: {err_tail[:400]}" if err_tail
              else "(could not parse audit output -- rerun manually)")
    sec.add(
        "FAIL",
        f"audit failed (rc={proc.returncode}) with no parseable MISSING "
        f"list; {reason}. Rerun: python scripts/audit_bundle_dlls.py",
    )
    return sec


# Anything below this size cannot be a real PE binary; the CI text stubs
# from ci-prepare-vendor-stubs.ps1 are 27 bytes.
_PE_MIN_SIZE = 4096


def _find_stub_binaries(llama_dir: Path, names: List[str]) -> List[str]:
    """Return manifest .exe/.dll entries present under ``llama_dir`` whose
    content cannot be a real PE image (smaller than 4 KB, or missing the
    MZ magic). Every guard between a CI text stub and a shipped installer
    that is existence-only risks a CI stub created on a dev box shipping
    in a real release with a dead translator. Mirrors the build-time
    check in installer/assert-llama-vendor.ps1; this is the bundle-side
    backstop."""
    stubs: List[str] = []
    for name in names:
        if not name.lower().endswith((".exe", ".dll")):
            continue
        p = llama_dir / name
        if not p.is_file():
            continue  # absence is the missing-files check's job
        try:
            if p.stat().st_size < _PE_MIN_SIZE:
                stubs.append(name)
                continue
            with p.open("rb") as fh:
                if fh.read(2) != b"MZ":
                    stubs.append(name)
        except OSError:
            stubs.append(name)
    return stubs


def check_llama_bundle_closure(release: bool = False) -> "Section":
    """Verify dist/yamabiko/_internal/llama contains every file the vendored
    llama-server needs, per installer/llama-manifest.txt.

    audit_bundle_dlls.py (section 9) deliberately skips the _internal/llama
    subtree -- llama-server runs as a child with its own DLL search path --
    so without this check a re-vendor that drops a CUDA/ggml sidecar ships
    silently and the translator dies at startup on the user's machine.
    build.ps1 asserts the same manifest at build time; this is the
    bundle-side backstop.
    """
    sec = Section(14, "Bundled llama-server DLL closure (manifest)")
    manifest = ROOT / "installer" / "llama-manifest.txt"
    if not manifest.is_file():
        sec.add("FAIL", f"missing manifest at {manifest}")
        return sec
    required = [
        ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not required:
        sec.add("FAIL", f"manifest {manifest.name} lists no files (empty / all-comments)")
        return sec
    bundle = ROOT / "dist" / "yamabiko"
    llama_dir = bundle / "_internal" / "llama"
    if not bundle.is_dir():
        # No bundle at all: SKIP in dev, FAIL under --release (same as section 9).
        status, msg = _classify_bundle_audit_missing(release)
        sec.add(status, msg)
        return sec
    if not llama_dir.is_dir():
        # A real bundle exists but llama was never vendored in -- the exact
        # silent-ship case this check exists to catch. Always FAIL.
        sec.add(
            "FAIL",
            "bundle exists but _internal/llama is missing entirely -- "
            "llama-server was not vendored into the build",
        )
        return sec
    present = {p.name for p in llama_dir.iterdir() if p.is_file()}
    missing = [f for f in required if f not in present]
    stubs = _find_stub_binaries(llama_dir, required)
    if missing:
        sec.add(
            "FAIL",
            f"_internal/llama missing {len(missing)} manifest file(s): "
            f"{', '.join(missing)}. Re-vendor per third_party/README.md + "
            f"installer/llama-manifest.txt",
        )
    if stubs:
        sec.add(
            "FAIL",
            f"_internal/llama contains {len(stubs)} stub/non-PE binarie(s): "
            f"{', '.join(stubs)} -- a CI text stub leaked into the bundle "
            f"Re-vendor real binaries before shipping.",
        )
    if not missing and not stubs:
        sec.add(
            "PASS",
            f"all {len(required)} manifest file(s) present in _internal/llama "
            f"(PE-content verified for .exe/.dll)",
        )
    return sec


_LICENSE_CLOSURE_REPORT_FIELDS = frozenset(
    {
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
)
_LICENSE_CLOSURE_HASH_FIELDS = (
    "bundle_tree_sha256",
    "inventory_sha256",
    "license_index_sha256",
    "provenance_policy_sha256",
    "license_override_policy_sha256",
    "distribution_metadata_sha256",
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_AUDITED_UTC = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether ``path`` redirects filesystem traversal.

    ``Path.is_symlink`` alone misses Windows junctions. The release evidence
    binds a specific tree, so traversing either kind would let bytes outside
    that tree satisfy the recorded digest.
    """
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _sha256_file(path: Path) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect fingerprint input {path}") from exc
    before_attributes = getattr(before, "st_file_attributes", 0)
    if not stat.S_ISREG(before.st_mode) or bool(
        before_attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError(f"fingerprint input is not a regular non-reparse file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ValueError(f"fingerprint input disappeared while hashing: {path}") from exc
    after_attributes = getattr(after, "st_file_attributes", 0)
    if not stat.S_ISREG(after.st_mode) or bool(
        after_attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError(f"fingerprint input is not a regular non-reparse file: {path}")
    before_identity = (
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ctime_ns", None),
        getattr(before, "st_dev", None),
        getattr(before, "st_ino", None),
        before_attributes,
    )
    after_identity = (
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ctime_ns", None),
        getattr(after, "st_dev", None),
        getattr(after, "st_ino", None),
        after_attributes,
    )
    if before_identity != after_identity or size != after.st_size:
        raise ValueError(f"fingerprint input changed while hashing: {path}")
    return digest.hexdigest()


def _canonical_file_records(
    root: Path,
    paths: List[Path],
) -> Tuple[str, int]:
    """Hash path/size/content records using the B6 report encoding."""
    records: List[Tuple[str, int, str]] = []
    identities: dict[str, str] = {}
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("fingerprinted file escaped its declared root") from exc
        identity = relative.casefold()
        previous = identities.get(identity)
        if previous is not None and previous != relative:
            raise ValueError(
                f"duplicate case-insensitive file paths: {previous!r}, {relative!r}"
            )
        identities[identity] = relative
        try:
            file_stat = path.lstat()
        except OSError as exc:
            raise ValueError(f"cannot inspect fingerprint input {relative!r}") from exc
        if _is_link_or_reparse(path) or not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(
                f"fingerprint input is not a regular in-tree file: {relative!r}"
            )
        file_hash = _sha256_file(path)
        try:
            after_hash = path.lstat()
        except OSError as exc:
            raise ValueError(
                f"fingerprint input disappeared after hashing: {relative!r}"
            ) from exc
        before_identity = (
            file_stat.st_size,
            file_stat.st_mtime_ns,
            getattr(file_stat, "st_ctime_ns", None),
            getattr(file_stat, "st_dev", None),
            getattr(file_stat, "st_ino", None),
        )
        after_identity = (
            after_hash.st_size,
            after_hash.st_mtime_ns,
            getattr(after_hash, "st_ctime_ns", None),
            getattr(after_hash, "st_dev", None),
            getattr(after_hash, "st_ino", None),
        )
        if before_identity != after_identity:
            raise ValueError(f"fingerprint input changed after hashing: {relative!r}")
        records.append((relative, file_stat.st_size, file_hash))

    digest = hashlib.sha256()
    for relative, size, file_hash in sorted(records):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest(), len(records)


def _bundle_tree_fingerprint(dist_root: Path) -> Tuple[str, int]:
    if not dist_root.is_dir():
        raise ValueError("bundle tree is missing")
    if _is_link_or_reparse(dist_root):
        raise ValueError("bundle root is a symlink or Windows reparse point")

    def walk_error(error: OSError) -> None:
        raise ValueError(f"cannot enumerate bundle tree: {error}") from error

    files: List[Path] = []
    for current_raw, directory_names, file_names in os.walk(
        dist_root, topdown=True, followlinks=False, onerror=walk_error
    ):
        current = Path(current_raw)
        for name in directory_names:
            directory = current / name
            if _is_link_or_reparse(directory):
                relative = directory.relative_to(dist_root).as_posix()
                raise ValueError(
                    f"bundle directory is a symlink or reparse point: {relative!r}"
                )
        for name in file_names:
            files.append(current / name)
    return _canonical_file_records(dist_root, files)


def _distribution_metadata_fingerprint(site_packages: Path) -> str:
    """Rebuild the auditor's canonical METADATA/RECORD input digest.

    The explicitly supplied site-packages root may itself be a venv junction;
    descendants may not redirect. Every installed ``*.dist-info`` directory
    is scanned because independent ownership derivation reads that universe.
    """
    if not site_packages.is_dir():
        raise ValueError("build site-packages is missing")

    files: List[Path] = []
    try:
        dist_infos = sorted(
            (
                path
                for path in site_packages.iterdir()
                if path.name.casefold().endswith(".dist-info")
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as exc:
        raise ValueError("cannot enumerate build site-packages") from exc

    identities: set[str] = set()
    for dist_info in dist_infos:
        identity = dist_info.name.casefold()
        if identity in identities:
            raise ValueError(
                f"duplicate case-insensitive metadata directory: {dist_info.name!r}"
            )
        identities.add(identity)
        if _is_link_or_reparse(dist_info) or not dist_info.is_dir():
            raise ValueError(
                f"distribution metadata directory is unsafe: {dist_info.name!r}"
            )
        for name in ("METADATA", "RECORD"):
            candidate = dist_info / name
            if _is_link_or_reparse(candidate) or not candidate.is_file():
                raise ValueError(
                    f"distribution metadata file is missing or not regular: "
                    f"{dist_info.name}/{name}"
                )
            files.append(candidate)
    digest, _count = _canonical_file_records(site_packages, files)
    return digest


def _load_license_closure_report(path: Path) -> dict:
    if _is_link_or_reparse(path) or not path.is_file():
        raise ValueError("report is not a regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("report is not valid UTF-8 JSON") from exc

    if not isinstance(payload, dict) or set(payload) != _LICENSE_CLOSURE_REPORT_FIELDS:
        raise ValueError("report fields do not match schema version 1")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("report schema_version must be integer 1")
    if not isinstance(payload["status"], str) or payload["status"] not in {
        "pass",
        "fail",
    }:
        raise ValueError("report status must be 'pass' or 'fail'")
    audited_utc = payload["audited_utc"]
    if not isinstance(audited_utc, str) or _AUDITED_UTC.fullmatch(audited_utc) is None:
        raise ValueError(
            "report audited_utc must be a canonical UTC timestamp ending in Z"
        )
    try:
        parsed_time = datetime.datetime.fromisoformat(audited_utc[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("report audited_utc is not an ISO timestamp") from exc
    if parsed_time.utcoffset() != datetime.timedelta(0):
        raise ValueError("report audited_utc is not UTC")
    count = payload["bundle_file_count"]
    if type(count) is not int or count < 0:
        raise ValueError("report bundle_file_count must be a non-negative integer")
    for field in _LICENSE_CLOSURE_HASH_FIELDS:
        value = payload[field]
        if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f"report {field} must be a lowercase SHA-256 digest")
    errors = payload["errors"]
    if (
        not isinstance(errors, list)
        or any(not isinstance(error, str) or not error for error in errors)
        or errors != sorted(set(errors))
    ):
        raise ValueError("report errors must be sorted unique non-empty strings")
    if (payload["status"] == "pass") != (errors == []):
        raise ValueError("report status and errors disagree")
    return payload


def _current_license_closure_fingerprints(
    inputs: dict[str, Path], venv: str,
) -> dict[str, object]:
    bundle_hash, bundle_count = _bundle_tree_fingerprint(
        ROOT / "dist" / "yamabiko"
    )
    current: dict[str, object] = {
        "bundle_tree_sha256": bundle_hash,
        "bundle_file_count": bundle_count,
    }
    for field, path in inputs.items():
        if _is_link_or_reparse(path) or not path.is_file():
            raise ValueError(f"{field} input is missing or unsafe")
        current[field] = _sha256_file(path)
    current["distribution_metadata_sha256"] = _distribution_metadata_fingerprint(
        _venv_dir(venv) / "Lib" / "site-packages"
    )
    return current


def check_bundle_license_closure(release: bool = False, venv: str = DEFAULT_VENV) -> "Section":
    """Verify that current release bytes still match a green closure audit.

    This gate deliberately does not import the independent auditor. It parses
    the canonical report and rebuilds the artifact/input fingerprints itself,
    so a stale ``status: pass`` file cannot bless changed bundle bytes.
    """
    sec = Section(15, "Bundle license closure audit evidence")
    report_path = ROOT / "build" / "yamabiko" / "license-closure-audit.json"
    if not report_path.is_file() or _is_link_or_reparse(report_path):
        if release:
            sec.add(
                "FAIL",
                "license closure audit report is missing -- a --release run "
                "requires build/yamabiko/license-closure-audit.json from the "
                "exact candidate build",
            )
        else:
            sec.add(
                "SKIP",
                "license closure audit report is missing; contributor run is "
                "not release-ready (build the candidate and rerun --release)",
            )
        return sec

    try:
        report = _load_license_closure_report(report_path)
    except ValueError as exc:
        sec.add("FAIL", f"license closure audit report is malformed: {exc}")
        return sec

    if report["status"] == "fail":
        errors = report["errors"]
        preview = "; ".join(errors[:3])
        if len(errors) > 3:
            preview += f"; +{len(errors) - 3} more"
        sec.add(
            "FAIL",
            f"license closure audit reported {len(errors)} error(s): {preview}",
        )
        return sec

    inputs = {
        "inventory_sha256": ROOT / "build" / "yamabiko" / "bundle-inventory.json",
        "license_index_sha256": (
            ROOT / "dist" / "yamabiko" / "_internal" / "licenses" / "INDEX.json"
        ),
        "provenance_policy_sha256": ROOT / "installer" / "bundle-provenance.toml",
        "license_override_policy_sha256": ROOT / "installer" / "licenses-manifest.txt",
    }
    try:
        current = _current_license_closure_fingerprints(inputs, venv)
        confirmed = _current_license_closure_fingerprints(inputs, venv)
        if current != confirmed:
            changed = sorted(
                field for field in current if current[field] != confirmed[field]
            )
            raise ValueError(
                "audit inputs changed between fingerprint snapshots: "
                + ", ".join(changed)
            )
        current = confirmed
    except (OSError, ValueError) as exc:
        sec.add("FAIL", f"cannot revalidate license closure evidence: {exc}")
        return sec

    mismatches = [
        field
        for field, value in current.items()
        if report[field] != value
    ]
    if mismatches:
        sec.add(
            "FAIL",
            "license closure audit evidence is stale for current candidate: "
            + ", ".join(sorted(mismatches)),
        )
        return sec

    sec.add(
        "PASS",
        f"green closure report matches {current['bundle_file_count']} current bundle file(s) "
        "and every audited input fingerprint",
    )
    return sec


# Release phases whose tests exercise dist/yamabiko. Their evidence must name
# the inventory of the bundle that exists now, not of an earlier build.
_FROZEN_PHASE_PREFIX = "frozen-"


def _git_head() -> Optional[str]:
    """``git rev-parse HEAD`` for ROOT, or None when it cannot be read."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and head else None


def _release_phase_ids() -> List[str]:
    """Phase IDs of the manifest scripts/run_pytest_release_phase.py runs."""
    payload = json.loads(
        (ROOT / "scripts" / "release-gates.json").read_text(encoding="utf-8")
    )
    phases = payload.get("phases") if isinstance(payload, dict) else None
    if not isinstance(phases, list) or not phases:
        raise ValueError("no phases list")
    ids = [phase.get("id") if isinstance(phase, dict) else None for phase in phases]
    if not all(isinstance(phase_id, str) and phase_id for phase_id in ids):
        raise ValueError("every phase needs a non-empty string id")
    return ids


def _load_json_object(path: Path) -> dict:
    if _is_link_or_reparse(path) or not path.is_file():
        raise ValueError("missing")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("not a JSON object")
    return payload


def _release_evidence_problems() -> List[str]:
    problems: List[str] = []
    head = _git_head()
    if head is None:
        problems.append("cannot read git HEAD of the candidate checkout")
    try:
        phase_ids = _release_phase_ids()
    except (OSError, ValueError) as exc:
        return problems + [f"cannot read scripts/release-gates.json: {exc}"]
    try:
        inventory_sha256: Optional[str] = _sha256_file(
            ROOT / "build" / "yamabiko" / "bundle-inventory.json"
        )
    except ValueError:
        inventory_sha256 = None

    gates = ROOT / "build" / "release-gates"
    for phase_id in phase_ids:
        name = f"{phase_id}-evidence.json"
        try:
            evidence = _load_json_object(gates / name)
        except ValueError as exc:
            problems.append(f"build/release-gates/{name} is {exc}")
            continue
        if evidence.get("final_exit_status") != 0:
            problems.append(
                f"{phase_id} evidence records a failed phase "
                f"(final_exit_status {evidence.get('final_exit_status')!r})"
            )
        if evidence.get("git_head") != head:
            problems.append(
                f"{phase_id} evidence git_head {evidence.get('git_head')!r} "
                f"is not HEAD {head!r}"
            )
        if evidence.get("git_dirty") is not False:
            problems.append(
                f"{phase_id} evidence git_dirty is "
                f"{evidence.get('git_dirty')!r}, not false"
            )
        if not phase_id.startswith(_FROZEN_PHASE_PREFIX):
            continue
        if inventory_sha256 is None:
            problems.append(
                f"{phase_id} evidence cannot be bound to a bundle: "
                "build/yamabiko/bundle-inventory.json is missing"
            )
        elif evidence.get("bundle_inventory_sha256") != inventory_sha256:
            problems.append(
                f"{phase_id} evidence bundle_inventory_sha256 does not match "
                "the current build/yamabiko/bundle-inventory.json"
            )

    try:
        record = _load_json_object(gates / "installer.json")
    except ValueError as exc:
        problems.append(f"build/release-gates/installer.json is {exc}")
        return problems
    installer_file = record.get("installer_file")
    candidate = ROOT / installer_file if isinstance(installer_file, str) else None
    if (
        candidate is None
        or not installer_file
        or candidate.parent.resolve() != (ROOT / "installer").resolve()
    ):
        problems.append(
            f"installer.json installer_file {installer_file!r} does not name "
            "a file in installer/"
        )
    else:
        try:
            installer_sha256: Optional[str] = _sha256_file(candidate)
        except ValueError:
            installer_sha256 = None
            problems.append(f"installer {installer_file} is missing")
        if (
            installer_sha256 is not None
            and record.get("installer_sha256") != installer_sha256
        ):
            problems.append(
                f"installer.json installer_sha256 does not match {installer_file}"
            )
    if record.get("git_head") != head:
        problems.append(
            f"installer.json git_head {record.get('git_head')!r} is not "
            f"HEAD {head!r}"
        )
    if record.get("git_dirty") is not False:
        problems.append(
            f"installer.json git_dirty is {record.get('git_dirty')!r}, not false"
        )
    return problems


def check_release_evidence_binding(release: bool = False) -> "Section":
    """Require release evidence produced from the current commit and build.

    Every phase scripts/run_pytest_release_phase.py runs, the frozen bundle
    those phases tested, and the installer build.ps1 produced must all name
    the current HEAD from a clean tree. Otherwise evidence from an earlier
    commit, or an installer from an earlier build, vouches for a candidate
    nothing tested.
    """
    sec = Section(16, "Release evidence binding")
    problems = _release_evidence_problems()
    if not problems:
        sec.add(
            "PASS",
            "every release phase, the frozen bundle inventory and the installer "
            "are bound to HEAD from a clean tree",
        )
        return sec
    if release:
        for problem in problems:
            sec.add("FAIL", problem)
        return sec
    sec.add(
        "SKIP",
        f"release evidence does not bind the current HEAD ({len(problems)} "
        f"problem(s), first: {problems[0]}); --release requires every phase "
        "and the installer at HEAD",
    )
    return sec


def _yaml_leaf_paths(data, prefix=()):
    """Walk a nested dict and yield (dotted_path, leaf_key_name) pairs."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                yield from _yaml_leaf_paths(v, prefix + (k,))
            else:
                yield ".".join(prefix + (k,)), k


def _yaml_segment_counts(data: dict) -> dict:
    """Count how many key nodes (containers AND leaves) carry each name.

    A leaf whose name also appears elsewhere in the tree — as another
    leaf (``device``, ``threshold``) or as a container (``overlay`` is
    both a top-level section and the ``lifecycle.grace_s.overlay``
    leaf) — is collision-prone for the grep in
    ``check_config_key_consumers``: a bare leaf-literal hit can come
    from the OTHER key's consumer.
    """
    counts: dict = {}

    def _walk(d: dict) -> None:
        for k, v in d.items():
            counts[k] = counts.get(k, 0) + 1
            if isinstance(v, dict):
                _walk(v)

    _walk(data)
    return counts


def check_config_key_consumers() -> "Section":
    """Every leaf key in default_config.yaml must have a consumer.

    Retires the producer/consumer key-mismatch class: producer (YAML or
    config_writer) ships a key, but the reader in main.py / asr_client
    / etc. reads a different one, and the pick is silently dropped. By
    walking every YAML leaf and requiring at least one ``"key"`` string
    literal in ``yamabiko/`` (excluding config.py + default_config.yaml
    themselves, since those ARE the producer layer), we catch orphan
    keys at preflight time rather than on a user's machine.

    This is a structural check — grep-based, not semantic. False
    negatives (a key read via dynamic dict traversal) are possible but
    vanishingly rare in practice; the entire read path in yamabiko/
    uses string-literal access. False positives can be silenced via
    ``allowlist`` below with a named reason.
    """
    import yaml
    sec = Section(10, "Config key producer/consumer match")

    # Keys intentionally in the YAML without a direct consumer — each
    # needs a named reason so we don't accumulate cruft over time.
    # Container-only keys whose CHILDREN are iterated (e.g. the whole
    # overlay.style dict is passed to the style system, not each leaf
    # read by literal name) go here.
    allowlist = {
        "style", "geometry", "faster_whisper",
    }

    cfg_path = ROOT / "yamabiko" / "default_config.yaml"
    try:
        cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        sec.add("FAIL", f"cannot read {cfg_path}: {exc}")
        return sec
    except yaml.YAMLError as exc:
        sec.add("FAIL", f"YAML parse error in {cfg_path}: {exc}")
        return sec

    # Collect source text from every .py under yamabiko/ (except
    # config.py: it's the layered-config machinery and treats keys as
    # data, not as literals).
    sources: List[str] = []
    for py in (ROOT / "yamabiko").rglob("*.py"):
        if py.name == "config.py":
            continue
        try:
            sources.append(py.read_text(encoding="utf-8"))
        except OSError:
            continue
    src_blob = "\n".join(sources)

    orphans: List[str] = []
    unanchored: List[str] = []
    checked = 0
    seg_counts = _yaml_segment_counts(cfg_raw)
    for dotted, leaf in _yaml_leaf_paths(cfg_raw):
        checked += 1
        if leaf in allowlist:
            continue
        pattern = re.compile(rf'["\']{re.escape(leaf)}["\']')
        if not pattern.search(src_blob):
            orphans.append(dotted)
            continue
        # Collision tier: when the leaf NAME is shared
        # with any other key in the YAML tree (leaf or container), a
        # bare leaf-literal hit may belong to a different key's
        # consumer — e.g. lifecycle.grace_s.overlay was "consumed" by
        # an unrelated "overlay" role string. For those, require the
        # immediate parent key quoted within 120 chars before the leaf
        # literal — the shape of every literal read path in yamabiko/
        # (cfg["parent"]["leaf"], cfg.get("parent", {}).get("leaf")).
        # Top-level scalars have no parent to anchor on; the allowlist
        # remains the escape hatch for genuinely dynamic reads.
        parts = dotted.split(".")
        if len(parts) >= 2 and seg_counts.get(leaf, 0) >= 2:
            parent = parts[-2]
            anchored = re.compile(
                rf'["\']{re.escape(parent)}["\'].{{0,120}}?["\']{re.escape(leaf)}["\']',
                re.S,
            )
            if not anchored.search(src_blob):
                unanchored.append(dotted)

    if orphans:
        sec.add(
            "FAIL",
            f"{len(orphans)} orphan key(s) in default_config.yaml "
            f"(no consumer in yamabiko/): {orphans}",
        )
    if unanchored:
        sec.add(
            "FAIL",
            f"{len(unanchored)} collision-prone key(s) without a "
            f"parent-anchored consumer (leaf name shared with another "
            f"config key, so a bare leaf match can be satisfied by an "
            f"unrelated consumer): {unanchored}",
        )
    if not orphans and not unanchored:
        sec.add(
            "PASS",
            f"all {checked} leaf key(s) in default_config.yaml have a consumer reference",
        )
    return sec


def check_bundle_module_launchers() -> "Section":
    """Every bundled ``yamabiko.*.__main__`` module needs a launcher.

    The spec doesn't know "launched by whom" — it just knows what to
    pack. Modules bundled without a runtime launch mechanism silently
    ship dead. This check walks the spec and asserts every
    ``yamabiko.X.__main__`` hiddenimport has a matching ``--run-X``
    argv dispatch in main.py or watchdog.py; and that the
    ``llama-server.exe`` copy in build.ps1 pairs with a watchdog spawn
    helper.
    """
    sec = Section(11, "Bundle module → launcher coupling")
    spec_path = ROOT / "installer" / "yamabiko.spec"
    main_path = ROOT / "yamabiko" / "main.py"
    watchdog_path = ROOT / "yamabiko" / "watchdog.py"
    build_path = ROOT / "installer" / "build.ps1"
    try:
        spec_src = spec_path.read_text(encoding="utf-8")
        main_src = main_path.read_text(encoding="utf-8")
        watchdog_src = watchdog_path.read_text(encoding="utf-8")
    except OSError as exc:
        sec.add("FAIL", f"cannot read source: {exc}")
        return sec

    # Every yamabiko.X.__main__ in spec hiddenimports must have a
    # --run-X dispatch somewhere in main.py / watchdog.py.
    # ([\w.]+) so dotted submodules (yamabiko.a.b.__main__) are seen.
    dispatch_blob = main_src + watchdog_src
    main_modules = re.findall(r'["\']yamabiko\.([\w.]+)\.__main__["\']', spec_src)
    if not main_modules:
        # Zero literals means the regex (or the spec's structure) has
        # drifted — e.g. hiddenimports built via f-string/loop. The
        # launcher-coupling guard would otherwise retire silently while this
        # section keeps reporting green off the llama half alone.
        # Fail loudly instead of silently not checking.
        sec.add(
            "FAIL",
            "no yamabiko.*.__main__ literals found in spec — regex or "
            "spec structure drifted; cannot verify launcher coupling",
        )
    ok_mods: List[str] = []
    for mod in main_modules:
        # Normalize: "overlay_app" → "--run-overlay" (strip trailing _app);
        # keep other suffixes literal so a future "yamabiko.foo_bar.__main__"
        # matches either "--run-foo_bar" or "--run-foo-bar".
        candidates = {
            f"--run-{mod}",
            f"--run-{mod.replace('_app', '')}",
            f"--run-{mod.replace('_', '-')}",
        }
        if any(c in dispatch_blob for c in candidates):
            ok_mods.append(mod)
        else:
            sec.add(
                "FAIL",
                f"yamabiko.{mod}.__main__ in spec hiddenimports but no "
                f"matching argv dispatch (tried: {sorted(candidates)})",
            )
    if main_modules and not sec.failed():
        sec.add("PASS", f"{len(ok_mods)} __main__ module(s) ↔ argv dispatch: {ok_mods}")

    # llama-server.exe copy in build.ps1 must pair with a spawn helper
    # in watchdog.py. Both-or-neither.
    try:
        build_src = build_path.read_text(encoding="utf-8")
    except OSError:
        build_src = ""
    build_has_llama = "llama-server.exe" in build_src
    watchdog_has_llama = (
        "_build_llama_cmd" in watchdog_src
        or "_real_spawn_llama" in watchdog_src
    )
    if build_has_llama and not watchdog_has_llama:
        sec.add(
            "FAIL",
            "installer/build.ps1 copies llama-server.exe but watchdog.py "
            "has no _build_llama_cmd / _real_spawn_llama helper — "
            "llama-server would be bundled but never launched",
        )
    elif watchdog_has_llama and not build_has_llama:
        sec.add(
            "FAIL",
            "watchdog.py has llama-server spawn logic but installer/build.ps1 "
            "does not copy the binary into the bundle",
        )
    elif build_has_llama and watchdog_has_llama:
        sec.add("PASS", "llama-server.exe bundle ↔ watchdog spawn")
    else:
        sec.add("SKIP", "no llama-server.exe references in build.ps1 or watchdog.py")

    return sec


def check_changelog_migration_notes() -> "Section":
    """If config keys changed since the last tag, CHANGELOG must flag it.

    Soft gate — WARN only, never FAIL. The goal is to raise producer/
    consumer shift awareness at release prep, not block non-release
    preflight runs. If last v* tag != current HEAD and the diff
    includes default_config.yaml, CHANGELOG's [Unreleased] section
    should have a 'Migration' subsection warning users about key
    changes. Warning is enough; users who intend to cut a release
    will notice.
    """
    sec = Section(12, "CHANGELOG migration-notes gate")
    cfg_path = ROOT / "yamabiko" / "default_config.yaml"
    changelog_path = ROOT / "CHANGELOG.md"

    # Most recent v* tag. No tags → skip (pre-release repo).
    try:
        tag_proc = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        sec.add("SKIP", f"git describe failed: {exc}")
        return sec
    last_tag = tag_proc.stdout.strip()
    if tag_proc.returncode != 0 or not last_tag:
        sec.add("SKIP", "no v* tags found (pre-release repo)")
        return sec

    try:
        diff_proc = subprocess.run(
            ["git", "diff", "--name-only", f"{last_tag}..HEAD", "--",
             str(cfg_path.relative_to(ROOT))],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        sec.add("SKIP", f"git diff failed: {exc}")
        return sec

    config_changed = bool(diff_proc.stdout.strip())
    if not config_changed:
        sec.add("PASS", f"default_config.yaml unchanged since {last_tag}")
        return sec

    try:
        changelog_src = changelog_path.read_text(encoding="utf-8")
    except OSError as exc:
        sec.add("WARN", f"config changed since {last_tag} but cannot read CHANGELOG: {exc}")
        return sec

    # Look for the Migration note under [Unreleased] OR under the section for
    # the version we are about to ship. At a release cut, docs/RELEASE.md has
    # you PROMOTE [Unreleased] to [<version>] -- so a gate that only greps
    # [Unreleased] goes blind at exactly the moment it is supposed to fire, and
    # reports "no Migration subsection" while the note sits right there under
    # the versioned header. Both headers are legitimate homes for the note:
    # mid-cycle it lives in [Unreleased], at a cut it lives in [<version>].
    app_version = None
    try:
        init_src = (ROOT / "yamabiko" / "__init__.py").read_text(encoding="utf-8")
        app_version = _extract(r'__version__\s*=\s*"([^"]+)"', init_src)
    except OSError:
        pass

    wanted = ["Unreleased"] + ([app_version] if app_version else [])
    searched: List[str] = []
    for header in wanted:
        match = re.search(
            rf'##\s*\[{re.escape(header)}\](.*?)(?=^##\s*\[|\Z)',
            changelog_src, flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            continue
        searched.append(header)
        if re.search(r'(?mi)^###?\s*Migration', match.group(1)):
            sec.add(
                "PASS",
                f"config changed since {last_tag}; CHANGELOG [{header}] "
                "has a Migration subsection",
            )
            return sec

    if not searched:
        sec.add(
            "WARN",
            f"config changed since {last_tag} but CHANGELOG.md has neither an "
            f"[Unreleased] nor an [{app_version}] section — add one with a "
            "Migration subsection.",
        )
        return sec

    sec.add(
        "WARN",
        f"config changed since {last_tag} but no 'Migration' subsection under "
        f"{' or '.join('[' + h + ']' for h in searched)} in CHANGELOG.md — add "
        "one before tagging so upgraders know what keys changed.",
    )
    return sec


def check_no_subject_mocks_in_real_tier() -> "Section":
    """Static check: @pytest.mark.real test files must not use MagicMock.

    The Yamabiko testing charter (docs/TESTING.md) forbids mocking the
    subject-under-test at the @pytest.mark.real tier: a MagicMock
    standing in for hf_hub_download accepts any call shape, so a
    real-library TypeError would pass green.

    Enforcement: grep every file with either `@pytest.mark.real` or a
    `pytestmark = pytest.mark.real` module-level declaration for any
    `MagicMock` / `AsyncMock` / `Mock(` / `unittest.mock` references.
    Those indicate a mislabeled unit test and must either be moved to
    the unit tier or rewritten to use the real library.

    A small named allowlist covers legitimate cases where mocking the
    environment (not the subject) is necessary in a real test. Add
    entries only with a one-line rationale inline.
    """
    sec = Section(13, "No subject-mocks in @pytest.mark.real tier")

    # Files where a mock is legitimately around the subject, not
    # replacing it. Rationale required per entry.
    allowlist = {
        # tests/test_real_watchdog_subprocess.py: monkeypatches
        # _build_main_cmd to a sentinel command so the real
        # subprocess.Popen path (the subject) actually runs a
        # fast-exit process. The Popen call is real.
        "tests/test_real_watchdog_subprocess.py",
    }

    tests_dir = ROOT / "tests"
    if not tests_dir.is_dir():
        sec.add("SKIP", "tests/ directory not found")
        return sec

    # A file is "fully-real-tier" if EITHER:
    #   - it has `pytestmark = pytest.mark.real` at module level
    #     (blanket marker: every test in the file is real-tier), OR
    #   - its filename starts with `test_real_` (naming convention:
    #     the whole file's purpose is real-path coverage).
    # Files that simply have some `@pytest.mark.real` decorators
    # alongside unit tests are NOT flagged -- mocks can legitimately
    # live in the unit tests within the same file. The real tests
    # need their own discipline (reviewer + hook + checklist catches
    # that), but a static whole-file grep can't distinguish.
    module_real_marker = re.compile(
        r"^pytestmark\s*=\s*pytest\.mark\.real\b", re.MULTILINE
    )

    offenders: List[Tuple[str, List[str]]] = []
    checked = 0
    for py in tests_dir.rglob("test_*.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except OSError:
            continue

        is_fully_real = (
            module_real_marker.search(src) is not None
            or py.name.startswith("test_real_")
        )
        if not is_fully_real:
            continue

        rel = py.relative_to(ROOT).as_posix()
        if rel in allowlist:
            continue

        checked += 1
        # Look for actual usage sites, not bare word mentions. Docstrings
        # and comments that explain "we don't use MagicMock" must not
        # trigger. Patterns below require call / import / assignment
        # syntax, which regex alone can't fully disambiguate from strings
        # but gets us 95% of the way. The remaining 5% can be allowlisted.
        patterns = [
            # `MagicMock(` / `MagicMock()` — a call
            (r"\bMagicMock\s*\(", "MagicMock()"),
            # `AsyncMock(` / `AsyncMock()` — a call
            (r"\bAsyncMock\s*\(", "AsyncMock()"),
            # `= Mock(` — assignment to a Mock call (covers `mock.Mock()`
            # too via the module-level attribute)
            (r"=\s*Mock\s*\(", "= Mock()"),
            # Import forms
            (r"^\s*from\s+unittest\.mock\s+import", "from unittest.mock import"),
            (r"^\s*from\s+unittest\s+import\s+mock\b", "from unittest import mock"),
            (r"^\s*import\s+unittest\.mock\b", "import unittest.mock"),
            # pytest-mock's mocker fixture
            (r"def\s+\w+\([^)]*\bmocker\b[^)]*\)", "mocker fixture"),
        ]
        hits: List[str] = []
        for pat, label in patterns:
            if re.search(pat, src, flags=re.MULTILINE):
                hits.append(label)
        if hits:
            offenders.append((rel, hits))

    if offenders:
        for rel, hits in offenders:
            sec.add(
                "FAIL",
                f"{rel}: uses {hits} in a @pytest.mark.real file. "
                f"Either move the test out of the real tier or "
                f"rewrite to use the real library. See docs/TESTING.md.",
            )
    else:
        sec.add(
            "PASS",
            f"all {checked} @pytest.mark.real file(s) are free of "
            f"MagicMock / unittest.mock (allowlist: "
            f"{sorted(allowlist)})",
        )
    return sec


# Preflight verifies bundle PRESENCE (files, DLLs, manifests) plus static
# gates; it cannot run the frozen E2E suite itself (section 8's 300 s cap
# can't fit it, and the E2E self-skips inside preflight), so the summary
# must never read as the whole release gate.
_E2E_GATE_REMINDER = (
    "Preflight covers bundle-presence + static checks only. The frozen "
    "release phases (run_pytest_release_phase.py frozen-external and "
    "frozen-owned) are SEPARATE required steps -- see docs/RELEASE.md "
    "before tagging."
)


def _final_report(sections: List["Section"], release: bool) -> Tuple[int, List[str]]:
    """Aggregate sections into ``(exit_code, summary_lines)``.

    Failures exit 1. Otherwise exit 0 -- but a run with skipped sections is
    NOT release-grade and must not claim the passed banner: it points the
    operator at ``--release`` on a built bundle instead. In ``--release``
    mode the bundle-dependent sections FAIL rather than skip, so a green
    ``--release`` run genuinely exercised them (any remaining skips are
    non-blocking, e.g. audio on a headless box). Even the green banner
    only claims what preflight verified: bundle presence + static gates;
    the frozen E2E suite is a separate required step (docs/RELEASE.md)."""
    out: List[str] = []
    failed = [s for s in sections if s.failed()]
    if failed:
        out.append(f"FAILED: {len(failed)} section(s) with failures")
        for s in failed:
            for status, msg in s.lines:
                if status == "FAIL":
                    out.append(f"  [{s.num}] {s.title}: {msg}")
        return 1, out
    skipped = [s for s in sections if s.skipped()]
    if skipped and not release:
        names = ", ".join(f"[{s.num}] {s.title}" for s in skipped)
        out.append(
            f"PASSED, but {len(skipped)} section(s) SKIPPED -- NOT a "
            f"release-grade run: {names}"
        )
        out.append("Re-run on a built bundle with: python scripts/preflight.py --release")
        return 0, out
    if skipped:
        names = ", ".join(f"[{s.num}] {s.title}" for s in skipped)
        out.append(
            f"All preflight sections passed ({len(skipped)} non-blocking "
            f"section(s) skipped: {names})."
        )
        out.append(_E2E_GATE_REMINDER)
        return 0, out
    out.append("All preflight sections passed.")
    out.append(_E2E_GATE_REMINDER)
    return 0, out


def _stamp_grade(sections: List["Section"], release: bool) -> str:
    """One-token grade for the .preflight-stamp: ``release-preflight``
    for a green --release run (bundle-presence + static checks ran; the
    frozen E2E gate is a SEPARATE required step, see docs/RELEASE.md --
    deliberately not ``release-grade``, which overstated what this
    script can verify), ``full`` for a dev run where every section ran,
    else ``dev-skipped:[n,...]`` naming the sections that did nothing."""
    skipped = [s.num for s in sections if s.skipped()]
    if release or not skipped:
        return "release-preflight" if release else "full"
    return "dev-skipped:" + ",".join(str(n) for n in skipped)


def _release_interpreter_problem(venv: str) -> Optional[str]:
    """None when this interpreter belongs to the selected build venv, else
    the message a --release run stops with."""
    selected = _venv_dir(venv)
    running = Path(sys.prefix)
    if os.path.normcase(os.path.realpath(running)) == os.path.normcase(
        os.path.realpath(selected)
    ):
        return None
    return (
        f"--release must run from the selected build venv's Python: this is "
        f"{running}, the selected venv is {selected} (--venv {venv}). Start "
        f"preflight with {selected / 'Scripts' / 'python.exe'}, or pass the "
        f"--venv this interpreter belongs to."
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    release = "--release" in args
    venv = DEFAULT_VENV
    usage = (
        "--venv needs the build venv as the next argument: a folder name "
        "under the repo root or an absolute path (--venv .venv-release)"
    )
    if any(arg.startswith("--venv=") for arg in args):
        print(usage)
        return 2
    if "--venv" in args:
        position = args.index("--venv") + 1
        if position >= len(args) or args[position].startswith("--"):
            print(usage)
            return 2
        venv = args[position]

    header = f"Yamabiko preflight - {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    if release:
        header += "  [--release: bundle-dependent sections must run]"
    print(header)
    print("=" * 40)
    print()

    if release:
        problem = _release_interpreter_problem(venv)
        if problem is not None:
            print(f"FAILED: {problem}")
            return 1

    sections: List[Section] = [
        check_hf_repos(),
        check_hf_download_signature(),
        check_version_consistency(),
        check_lazy_imports(),
        check_llama_server(),
        check_audio_devices(),
        check_pyinstaller_spec(),
        check_real_path_pytest(release=release, venv=venv),
        check_bundle_dll_audit(release=release, venv=venv),
        check_llama_bundle_closure(release=release),
        check_bundle_license_closure(release=release, venv=venv),
        check_release_evidence_binding(release=release),
        check_config_key_consumers(),
        check_bundle_module_launchers(),
        check_changelog_migration_notes(),
        check_no_subject_mocks_in_real_tier(),
    ]

    for s in sections:
        s.emit()

    print("=" * 40)
    code, lines = _final_report(sections, release)
    for ln in lines:
        print(ln)

    if code == 0:
        # Stamp success for the pre-push guard hook. The hook warns if this
        # file is older than ~10 min, surfacing "I forgot to run preflight"
        # before a push lands. .gitignored via scripts/.gitignore so stamp
        # churn doesn't dirty git status. The grade suffix records whether
        # the run was a --release preflight (bundle-presence + static
        # checks; the frozen E2E gate in docs/RELEASE.md remains a
        # separate step) or a dev run with bundle-dependent sections
        # skipped, so the two are distinguishable when auditing why a
        # build was released.
        try:
            stamp = ROOT / "scripts" / ".preflight-stamp"
            stamp.write_text(
                f"{datetime.datetime.now().isoformat(timespec='seconds')} "
                f"{_stamp_grade(sections, release)}\n",
                encoding="utf-8",
            )
        except OSError:
            # Stamp failure must not fail the gate — the stamp is a UX
            # convenience for the hook, not a correctness requirement.
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
