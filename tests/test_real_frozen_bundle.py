"""Real-path smoke test against the PyInstaller-frozen bundle.

PyInstaller's onedir mode relocates every ``Path(__file__).parent``
reference inside the bundled runtime — ``yamabiko/default_config.yaml``,
``yamabiko/display/static/index.html``, ``assets/icon.ico``,
``yamabiko/glossaries/*``. Unfrozen tests assert these paths from the
source checkout; the frozen bundle uses ``_MEIPASS`` + PyInstaller's
relocated ``__file__`` semantics, which those unfrozen tests don't
exercise.

This test spawns the BUILT ``dist/yamabiko/yamabiko.exe`` as a subprocess
and runs cheap, side-effect-free commands against it (``--version``,
the ``--run-main`` short-circuit with immediate exit) to confirm the
bundle at least boots without an ImportError / NoSuchFile / encoding
crash.

Most tests skip cleanly when no built bundle exists. The dedicated ASR
runtime import smoke is intentionally non-skipped because it is a release
artifact gate: run ``installer/rebuild-fast.ps1`` before selecting it.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.audit_bundle_inventory import find_forbidden_media
from scripts.bundle_inventory import load_inventory

pytestmark = pytest.mark.real

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FROZEN_EXE = _REPO_ROOT / "dist" / "yamabiko" / "yamabiko.exe"
_BUNDLE_INVENTORY = _REPO_ROOT / "build" / "yamabiko" / "bundle-inventory.json"
_DIST_ROOT = _REPO_ROOT / "dist" / "yamabiko"


def _skip_if_no_bundle() -> None:
    if not _FROZEN_EXE.is_file():
        pytest.skip(
            f"no built bundle at {_FROZEN_EXE}; run installer/build.ps1 first"
        )


def test_frozen_bundle_version_prints_without_import_error() -> None:
    """``yamabiko.exe --version`` must print the version string and exit 0.

    The ``--version`` path short-circuits BEFORE any heavy import, so
    this test would fail if:
    - The PyInstaller bundle is missing a hidden import that
      ``yamabiko.watchdog`` touches at module load.
    - The bundle's ``sys.path`` / ``__file__`` wiring is broken.
    - Version string import from ``yamabiko.__init__`` fails.
    """
    _skip_if_no_bundle()

    result = subprocess.run(
        [str(_FROZEN_EXE), "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, (
        f"--version exited {result.returncode}; stderr={result.stderr!r}"
    )
    assert "Yamabiko" in result.stdout, (
        f"--version stdout didn't contain 'Yamabiko': {result.stdout!r}"
    )


def test_frozen_bundle_imports_pipeline_module_tree_without_exceptions() -> None:
    """Force the frozen exe to load yamabiko.main's full import chain.

    The ``--version`` path short-circuits before any pipeline module
    imports, so missing-dependency crashes in the pipeline tree slip past
    a version-only smoke test. This test uses ``--run-main --help``:
      1. watchdog.main() dispatches --run-main, ahead of its own --help
         gate, to ``from yamabiko import main as _main_mod``, which runs
         yamabiko.main's module-level imports (capture, vad, asr,
         sentencing, translation, log_config, etc.).
      2. _main_mod.main() runs argparse, sees --help, prints the
         pipeline's usage and exits 0. The pipeline never starts.
      3. That usage text is the proof that step 1 completed: main() is
         unreachable unless every module-level import of yamabiko.main
         succeeded. Exit 0 alone proves nothing, because a watchdog whose
         --help gate shadows --run-main prints its own help, exits 0 and
         imports none of the pipeline.

    Seeded-defect check: add "yamabiko.asr.asr_client" to ``excludes`` in
    installer/yamabiko.spec and rebuild -> importing yamabiko.main raises
    ModuleNotFoundError, the exe exits nonzero, red.
    """
    _skip_if_no_bundle()

    result = subprocess.run(
        [str(_FROZEN_EXE), "--run-main", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    # argparse --help exits 0 after printing usage.
    assert result.returncode == 0, (
        f"yamabiko.exe --run-main --help exited {result.returncode}; "
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
    # Only yamabiko.main.parse_args prints this metavar; the watchdog's
    # help text spells the flag "--max-segments N".
    assert "--max-segments MAX_SEGMENTS" in result.stdout, (
        "yamabiko.exe --run-main --help did not print the pipeline's "
        "argparse usage, so yamabiko.main was never imported; "
        f"stdout head:\n{result.stdout[:1000]}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    # Failure mode: tracebacks from any module-level import in the chain.
    bad_markers = ("Traceback", "ImportError", "ModuleNotFoundError",
                   "AttributeError")
    hits = [m for m in bad_markers if m in combined]
    assert not hits, (
        f"frozen bundle pipeline import chain has {hits}; "
        f"combined output tail:\n{combined[-3000:]}"
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_loads_pinned_whisper_model_through_compat_seam() -> None:
    """The built candidate contains code + metadata for faster-whisper.

    This gate is deliberately non-skipped: data files and dist-info on disk
    do not prove PyInstaller collected the parent package code. The internal
    probe resolves ``metadata.version('faster-whisper')`` and loads the real
    ``WhisperModel`` class through Yamabiko's compatibility seam without
    instantiating a model or touching GPU, audio, network, or model cache.

    ``--help`` is intentional causal pressure: an older artifact without the
    internal route exits zero via public help, but cannot emit the success
    marker and therefore fails this test.
    """
    assert _FROZEN_EXE.is_file(), (
        f"required frozen candidate missing at {_FROZEN_EXE}; "
        "run installer/rebuild-fast.ps1 before this artifact gate"
    )

    result = subprocess.run(
        [str(_FROZEN_EXE), "--verify-asr-runtime", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    marker = "YAMABIKO_ASR_RUNTIME_OK faster-whisper=1.2.1"
    assert result.returncode == 0, (
        f"ASR runtime probe exited {result.returncode}; "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == marker, (
        "frozen ASR runtime probe did not prove the pinned package import; "
        f"expected {marker!r}, got stdout={result.stdout!r}, "
        f"stderr={result.stderr!r}"
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_omits_pyav_ffmpeg_x264_and_x265() -> None:
    """The exact built candidate contains none of the unused media stack.

    This non-skippable artifact gate audits both PyInstaller Analysis output
    and the final onedir tree. Seeded defect: remove the
    ``faster_whisper.audio`` exclusion and rebuild; the inventory audit turns
    red. The companion AST gate pins the exact ``av`` declaration, while this
    artifact scanner catches any future import edge that reintroduces PyAV.
    """
    assert _BUNDLE_INVENTORY.is_file(), (
        f"required bundle inventory missing at {_BUNDLE_INVENTORY}; "
        "run installer/build.ps1 before this artifact gate"
    )
    assert _DIST_ROOT.is_dir(), (
        f"required frozen candidate tree missing at {_DIST_ROOT}; "
        "run installer/build.ps1 before this artifact gate"
    )

    entries = load_inventory(_BUNDLE_INVENTORY)
    matches = find_forbidden_media(entries, _DIST_ROOT)

    assert matches == (), (
        "frozen candidate still contains the unused PyAV/FFmpeg media stack:\n"
        + "\n".join(matches)
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_omits_huggingface_cli_and_faster_whisper_silero_data() -> None:
    """The candidate omits reviewed Hugging Face and duplicate VAD payloads.

    Yamabiko downloads models through ``hf_hub_download`` and ``hf_xet``; it
    does not expose Hugging Face's CLI or inference client.  Yamabiko also
    downloads its pinned Silero model into the shared cache, so the model
    bundled as faster-whisper package data is unused.  Audit both PyInstaller's
    Analysis inventory and the final onedir tree so neither a PYZ module nor a
    copied data file can hide from the gate.

    The Silero filename is version-tolerant because faster-whisper 1.2.1 ships
    ``silero_vad_v6.onnx``.  This deliberately does not reject ``typer``,
    ``shellingham``, or any distribution by name; those need independent
    provenance before removal.
    """
    assert _BUNDLE_INVENTORY.is_file(), (
        f"required bundle inventory missing at {_BUNDLE_INVENTORY}; "
        "run installer/build.ps1 before this artifact gate"
    )
    assert _DIST_ROOT.is_dir(), (
        f"required frozen candidate tree missing at {_DIST_ROOT}; "
        "run installer/build.ps1 before this artifact gate"
    )

    forbidden = (
        re.compile(
            r"(?:^|[./\\])huggingface_hub[./\\]"
            r"(?:cli|inference)(?:[./\\]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:^|[./\\])faster_whisper[./\\]assets[./\\]"
            r"silero_vad(?:_v\d+)?\.onnx$",
            re.IGNORECASE,
        ),
    )

    matches: set[str] = set()
    for entry in load_inventory(_BUNDLE_INVENTORY):
        for field in ("logical_name", "source_path", "destination"):
            value = getattr(entry, field)
            if any(pattern.search(value) for pattern in forbidden):
                matches.add(f"inventory.{field}: {value}")

    for path in _DIST_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(_DIST_ROOT).as_posix()
        if any(pattern.search(relative) for pattern in forbidden):
            matches.add(f"dist: {relative}")

    ordered = sorted(matches)
    preview_limit = 40
    preview = ordered[:preview_limit]
    if len(ordered) > preview_limit:
        preview.append(f"... {len(ordered) - preview_limit} more match(es) omitted")
    assert not matches, (
        "frozen candidate still contains unused Hugging Face CLI/inference "
        f"or faster-whisper Silero data ({len(ordered)} evidence row(s)):\n"
        + "\n".join(preview)
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_omits_fastapi_cli_and_unused_uvicorn_plugins() -> None:
    """The candidate contains only Yamabiko's reviewed ASGI server surface.

    Yamabiko constructs FastAPI and Uvicorn in-process; it exposes neither
    package's command-line entry point, Uvicorn's Gunicorn worker, nor the
    automatic and alternate loop/protocol/lifespan backends.  FastAPI's
    package-local agent instruction files are development material, not
    runtime data.  Audit both the Analysis inventory and final onedir tree.

    Uvicorn 0.44.0's package initializer unconditionally imports its core
    supervisor modules through ``uvicorn.main``.  Those modules are therefore
    deliberately outside this ban: excluding them would break ordinary
    ``import uvicorn``.  Seeded defects: restore either broad
    ``collect_all("fastapi")`` or ``collect_all("uvicorn")`` and rebuild; its
    corresponding forbidden roots return and this gate turns red.
    """
    assert _BUNDLE_INVENTORY.is_file(), (
        f"required bundle inventory missing at {_BUNDLE_INVENTORY}; "
        "run installer/build.ps1 before this artifact gate"
    )
    assert _DIST_ROOT.is_dir(), (
        f"required frozen candidate tree missing at {_DIST_ROOT}; "
        "run installer/build.ps1 before this artifact gate"
    )

    forbidden_roots = tuple(
        re.sub(r"[./\\]+", "/", value.casefold())
        for value in (
            "fastapi.__main__",
            "fastapi.cli",
            "fastapi.agents",
            "agents.skills.fastapi",
            "uvicorn.__main__",
            "uvicorn.workers",
            "uvicorn.loops.auto",
            "uvicorn.loops.uvloop",
            "uvicorn.protocols.http.auto",
            "uvicorn.protocols.http.httptools_impl",
            "uvicorn.protocols.websockets.auto",
            "uvicorn.protocols.websockets.websockets_sansio_impl",
            "uvicorn.protocols.websockets.wsproto_impl",
            "uvicorn.lifespan.off",
        )
    )

    def _forbidden(value: str) -> bool:
        normalized = re.sub(r"[./\\]+", "/", value.casefold()).strip("/")
        padded = f"/{normalized}/"
        return any(f"/{root}/" in padded for root in forbidden_roots)

    matches: set[str] = set()
    for entry in load_inventory(_BUNDLE_INVENTORY):
        for field in ("logical_name", "source_path", "destination"):
            value = getattr(entry, field)
            if _forbidden(value):
                matches.add(f"inventory.{field}: {value}")

    for path in _DIST_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(_DIST_ROOT).as_posix()
        if _forbidden(relative):
            matches.add(f"dist: {relative}")

    ordered = sorted(matches)
    preview_limit = 40
    preview = ordered[:preview_limit]
    if len(ordered) > preview_limit:
        preview.append(f"... {len(ordered) - preview_limit} more match(es) omitted")
    assert not matches, (
        "frozen candidate still contains unused FastAPI/Uvicorn surface "
        f"({len(ordered)} evidence row(s)):\n" + "\n".join(preview)
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_omits_dev_only_windows_automation_payload() -> None:
    """Dev-only UI automation dependencies do not enter the candidate.

    Pywinauto is installed only so the release suite can drive the frozen
    wizard.  Its PyWin32 transitives can otherwise satisfy optional imports in
    ``logging.handlers`` and ``numpy.testing`` during Analysis even though
    Yamabiko never uses the Windows event-log or performance-counter APIs.
    """
    assert _BUNDLE_INVENTORY.is_file(), (
        f"required bundle inventory missing at {_BUNDLE_INVENTORY}; "
        "run installer/build.ps1 before this artifact gate"
    )
    assert _DIST_ROOT.is_dir(), (
        f"required frozen candidate tree missing at {_DIST_ROOT}; "
        "run installer/build.ps1 before this artifact gate"
    )

    forbidden = (
        "pywinauto",
        "comtypes",
        "win32evtlog",
        "win32evtlogutil",
        "win32pdh",
    )

    matches: set[str] = set()
    for entry in load_inventory(_BUNDLE_INVENTORY):
        for field in ("logical_name", "source_path", "destination"):
            value = getattr(entry, field)
            normalized = value.casefold().replace("\\", "/")
            if any(marker in normalized for marker in forbidden):
                matches.add(f"inventory.{field}: {value}")

    for path in _DIST_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(_DIST_ROOT).as_posix()
        normalized = relative.casefold()
        if any(marker in normalized for marker in forbidden):
            matches.add(f"dist: {relative}")

    ordered = sorted(matches)
    assert not matches, (
        "frozen candidate contains dev-only Windows automation payload "
        f"({len(ordered)} evidence row(s)):\n" + "\n".join(ordered)
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_places_onnx_vc_dot_library_in_main_dll_root() -> None:
    """The exact built candidate ships ONNX's VC dot library at _internal/.

    ONNX Runtime imports ``MSVCP140_1.dll`` by basename.  Copies private to
    PySide6 or shiboken6 do not satisfy the application's main DLL search
    domain, while a developer machine's System32 copy masks the clean-install
    failure.  The artifact must therefore contain this exact root path.

    Seeded defect: omit ``-AppRuntimeDir`` from either build helper call and
    rebuild -> this exact path is absent -> this fails.
    """
    runtime = _DIST_ROOT / "_internal" / "MSVCP140_1.dll"
    assert runtime.is_file(), (
        f"required ONNX VC dot library missing at {runtime}; nested package "
        "copies and the build host's System32 copy do not satisfy this gate"
    )


@pytest.mark.frozen_artifact
def test_frozen_bundle_ships_pinned_sound_tagger_where_runtime_resolves_it() -> None:
    """The candidate carries the pinned sound tagger at the path the frozen
    runtime loads it from, with its license beside it.

    Frozen, ``paths.bundle_data_dir()`` is ``_internal/yamabiko``; the path
    below the package dir comes from the production resolver itself, so a
    spec destination that drifts from ``default_model_path()`` (or a file
    the spec never collects) fails here instead of at the user's first
    start, where SileroChunker refuses to run without it.
    """
    import hashlib

    from yamabiko import paths
    from yamabiko.vad.sound_tagger import MODEL_SHA256, default_model_path

    package_dir = paths.bundle_data_dir()
    relative = default_model_path().relative_to(package_dir.parent)
    model = _DIST_ROOT / "_internal" / relative
    assert model.is_file(), f"sound tagger missing from the candidate at {model}"
    with model.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == MODEL_SHA256, f"{model} is not the pinned graph ({digest})"
    license_file = model.with_name("LICENSE-EfficientSED.txt")
    assert license_file.is_file(), f"sound tagger license missing at {license_file}"


def test_frozen_bundle_imports_overlay_module_tree_without_exceptions() -> None:
    """Force PyInstaller to load yamabiko.overlay_app.__main__'s import
    chain. No test previously exercised the overlay import path in the
    bundle, which let overlay-specific missing-dependency crashes go
    undetected.

    Uses ``--run-overlay --help`` via the same pattern as the pipeline
    sibling above: watchdog.py imports yamabiko.overlay_app.__main__ and
    calls its main(), which argparse exits via --help before any Qt
    window is created. The overlay's usage text on stdout proves the
    import completed; exit 0 alone does not (see the pipeline sibling).

    Seeded-defect check: add "yamabiko.overlay_app.ws_client" to
    ``excludes`` in installer/yamabiko.spec and rebuild -> importing
    yamabiko.overlay_app.__main__ raises ModuleNotFoundError, red.
    """
    _skip_if_no_bundle()

    result = subprocess.run(
        [str(_FROZEN_EXE), "--run-overlay", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"yamabiko.exe --run-overlay --help exited {result.returncode}; "
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
    # Only the overlay's argparse has --url; the watchdog's help does not.
    assert "--url URL" in result.stdout, (
        "yamabiko.exe --run-overlay --help did not print the overlay's "
        "argparse usage, so yamabiko.overlay_app.__main__ was never "
        f"imported; stdout head:\n{result.stdout[:1000]}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    bad_markers = ("Traceback", "ImportError", "ModuleNotFoundError",
                   "AttributeError")
    hits = [m for m in bad_markers if m in combined]
    assert not hits, (
        f"frozen bundle overlay import chain has {hits}; "
        f"combined output tail:\n{combined[-3000:]}"
    )


def test_frozen_bundle_version_smoke_exits_cleanly() -> None:
    """Keep the --version smoke as a minimal check. It exercises a
    narrow subset (watchdog + __init__ + hf_env) that the broader
    import tests above already cover, but the version smoke is also
    the canonical "the exe can at least start" sanity test that
    installer/build.ps1 runs. Keeping it here as a regression guard
    with accurate naming.

    Seeded-defect check: delete __version__ from yamabiko/__init__.py
    -> this test fails.
    """
    _skip_if_no_bundle()
    result = subprocess.run(
        [str(_FROZEN_EXE), "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "Yamabiko" in result.stdout
    assert "Traceback" not in result.stderr


def test_frozen_bundle_has_internal_directory_structure() -> None:
    """PyInstaller 6.x onedir puts bundled package + data under
    ``dist/yamabiko/_internal/``. The NSIS installer's uninstaller keys
    off that path; if PyInstaller ever changes the layout convention
    this test fires.
    """
    _skip_if_no_bundle()

    internal = _REPO_ROOT / "dist" / "yamabiko" / "_internal"
    assert internal.is_dir(), (
        f"expected {internal} — PyInstaller onedir 6.x layout. "
        f"If PyInstaller changed the default, update installer/yamabiko.nsi "
        f"(the uninstaller RMDir /r targets \\_internal)."
    )

    # Known-required bundled resources. If any is missing, a runtime
    # import will fail inside the bundle — these are our spec's datas
    # entries that the app depends on.
    required = [
        "yamabiko/default_config.yaml",
        "yamabiko/display/static/index.html",
        "yamabiko/glossaries",
        "assets/icon.ico",
        "assets/icon.png",
        "yamabiko/display/static/fonts/BIZUDPGothic-Regular.ttf",
        "yamabiko/display/static/fonts/BIZUDPGothic-Bold.ttf",
        "yamabiko/display/static/fonts/BIZUDPMincho-Regular.ttf",
        "yamabiko/display/static/fonts/BIZUDPMincho-Bold.ttf",
        "yamabiko/display/static/fonts/KleeOne-SemiBold.ttf",
        "yamabiko/display/static/fonts/OFL-BIZUDPGothic.txt",
        "yamabiko/display/static/fonts/OFL-BIZUDPMincho.txt",
        "yamabiko/display/static/fonts/OFL-KleeOne.txt",
        "yamabiko/overlay_app/assets/preview-night.jpg",
        "yamabiko/overlay_app/assets/LICENSE-preview-night.txt",
        # The JPEG image-format plugin loads the preview photo in the bundle.
        "PySide6/plugins/imageformats/qjpeg.dll",
        "LICENSE",
        "NOTICE",
        # Package-internal files that must land on disk because their
        # consumers read them via Path(__file__).parent at runtime.
        # The unidic_lite and cutlet missing-file crashes were both at
        # this layer; these assertions guard the same surface.
        #
        # unidic_lite: fugashi's Cython try_import_unidic() imports the
        # package at runtime and reads DICDIR. Without __init__.py on
        # disk, the dicdir/ data-files collection creates a PEP 420
        # namespace package that shadows the PYZ-embedded __init__.py,
        # stripping DICDIR. Both .py files must be on disk next to
        # dicdir/. See installer/yamabiko.spec (include_py_files=True).
        "unidic_lite/__init__.py",
        "unidic_lite/unidic.py",
        "unidic_lite/dicdir",
        # cutlet: load_exceptions() opens Path(__file__).parent /
        # 'exceptions.tsv' at Tagger init. The TSV is a data file that
        # cannot live in PYZ — it must be on disk.
        "cutlet/exceptions.tsv",
        # sudachidict_core: sudachipy resolves the dictionary via
        # `import sudachidict_core` + __file__ — None on the PEP 420
        # namespace package a data-only tree creates (the unidic_lite
        # failure shape exactly). __init__.py on disk keeps it a
        # regular package; the .dic payload rides in resources/.
        "sudachidict_core/__init__.py",
        "sudachidict_core/resources",
    ]
    missing = [r for r in required if not (internal / r).exists()]
    assert not missing, (
        f"bundled resources missing from _internal/: {missing}. "
        f"Check installer/yamabiko.spec datas= list."
    )


# ---------------------------------------------------------------------------
# CUDA runtime DLL coverage
#
# The bundle does not ship torch. Torch used to supply
# cublas64_12.dll + the cuDNN 9 sub-DLL set via torch/lib/; the
# ctranslate2 wheel itself ships ONLY cudnn64_9.dll (the front loader)
# and libiomp5md.dll, so without explicit provisioning, CT2.encode()
# raises "Library cublas64_12.dll is not found" on first ASR warmup.
#
# The `@pytest.mark.real` suite runs in the dev venv where torch is
# still installed — so the DLL resolution succeeds via torch/lib/ and
# the gap is invisible to the test runner. The only way to catch it is
# to inspect the FROZEN bundle itself. That's what these tests do.
# ---------------------------------------------------------------------------

# CT2 4.x links dynamically against cuBLAS 12 (verified by scanning the
# ctranslate2.dll import strings — it references exactly this filename).
_REQUIRED_CUBLAS_DLLS = [
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    # cublasLt64_12.dll imports nvrtc64_120_0.dll (verified by scanning
    # the wheel's DLL strings). Loading cublas without NVRTC silently
    # works until the first JIT-compiled kernel path is hit.
    "nvrtc64_120_0.dll",
]

# cuDNN 9 on Windows is multi-DLL: cudnn64_9.dll is just a thin front
# loader that dlopens the sub-DLLs by filename when a specific op kind
# is first invoked. Whisper's encoder uses convolutions, so at minimum
# the ops + cnn + graph + engines sub-DLLs must be present — but we
# require the full set because the heuristic and adv DLLs are loaded
# lazily during beam search and other code paths.
_REQUIRED_CUDNN_DLLS = [
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    # cudnn_engines_tensor_ir64_9.dll is new in cuDNN 9.21 and is
    # imported by cudnn_graph64_9.dll. Older lists that predate 9.21
    # miss this one.
    "cudnn_engines_tensor_ir64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_adv64_9.dll",
]


def _find_dll_in_bundle(internal_dir: Path, name: str) -> Path | None:
    """Find a DLL anywhere under _internal/. PyInstaller may flatten to
    the onedir root or leave it in a package subdirectory depending on
    where collect_dynamic_libs grabbed it from; both are fine at runtime
    because the bootloader adds _internal/ to the DLL search path."""
    matches = list(internal_dir.rglob(name))
    return matches[0] if matches else None


def test_frozen_bundle_has_cuda12_cublas_dlls() -> None:
    """cuBLAS 12 must be bundled — CT2 links against it at encode time.

    Dropping torch from the bundle removed the de facto cuBLAS source;
    the ctranslate2 wheel does NOT supply cublas*. If this test goes
    red, the nvidia-cublas-cu12 wheel either isn't installed or isn't
    being collected in installer/yamabiko.spec.
    """
    _skip_if_no_bundle()
    internal = _REPO_ROOT / "dist" / "yamabiko" / "_internal"

    missing = [d for d in _REQUIRED_CUBLAS_DLLS
               if _find_dll_in_bundle(internal, d) is None]
    assert not missing, (
        f"CUDA 12 cuBLAS DLLs missing from bundle: {missing}. "
        f"Install nvidia-cublas-cu12 (see yamabiko/requirements.txt) "
        f"and ensure installer/yamabiko.spec collects it via "
        f"collect_dynamic_libs('nvidia.cublas')."
    )


def test_frozen_bundle_has_full_cudnn9_sub_dll_set() -> None:
    """All cuDNN 9 sub-DLLs must ship, not just the front loader.

    cudnn64_9.dll alone is a trap: it ships with the ctranslate2 wheel,
    so a naive inspection says 'cuDNN is there.' But cudnn64_9.dll is a
    thin dispatcher that dlopens cudnn_ops64_9.dll, cudnn_cnn64_9.dll,
    etc. by filename when the first convolution runs. Missing any of
    these fails at the FIRST transcription, not at import.
    """
    _skip_if_no_bundle()
    internal = _REPO_ROOT / "dist" / "yamabiko" / "_internal"

    missing = [d for d in _REQUIRED_CUDNN_DLLS
               if _find_dll_in_bundle(internal, d) is None]
    assert not missing, (
        f"cuDNN 9 sub-DLLs missing from bundle: {missing}. "
        f"Install nvidia-cudnn-cu12 (see yamabiko/requirements.txt) "
        f"and ensure installer/yamabiko.spec collects it via "
        f"collect_dynamic_libs('nvidia.cudnn')."
    )


def test_frozen_bundle_ctranslate2_dll_loads_cublas_from_bundle() -> None:
    """Verify the bundle's DLL resolution actually works end-to-end.

    Presence on disk is necessary but not sufficient — a CUDA DLL may
    be there yet still fail to load because ONE of its own dependencies
    is missing. This test spawns a fresh Python subprocess with a
    stripped DLL search path (only _internal/ is added) and does a real
    ``ctypes.WinDLL('cublas64_12.dll')``. If it loads, every transitive
    dep is present. If it fails, we see the actual missing DLL name.

    We run this in a subprocess (not in-process) to keep the test
    runner's DLL directory table clean — otherwise a loaded cuBLAS
    would stay pinned for the remainder of the pytest session.
    """
    _skip_if_no_bundle()
    import sys, textwrap
    internal = _REPO_ROOT / "dist" / "yamabiko" / "_internal"

    script = textwrap.dedent(f"""
        import ctypes, os, sys
        internal = r'{internal}'
        # Clean search path: no PATH, no user dirs. Only the bundled
        # _internal/ directory — same as the frozen bootloader's setup.
        os.add_dll_directory(internal)
        try:
            # Front-door DLLs the bundle directly depends on. If any
            # transitive dep is missing these WinDLL calls will raise
            # with the exact DLL name the loader couldn't resolve.
            ctypes.WinDLL('cublas64_12.dll')
            ctypes.WinDLL('cublasLt64_12.dll')
            ctypes.WinDLL('cudnn64_9.dll')
            ctypes.WinDLL('nvrtc64_120_0.dll')
        except OSError as e:
            print(f'LOAD_FAIL: {{e}}', file=sys.stderr)
            sys.exit(2)
        print('OK')
    """)

    # Clear PATH to force loader into the explicit add_dll_directory.
    clean_env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
                 "PATH": ""}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
        env=clean_env,
    )
    assert result.returncode == 0, (
        f"Bundle's CUDA DLLs don't load cleanly from _internal/:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}\n"
        f"This means a transitive DLL dep is missing even though the "
        f"primary DLL file is present."
    )


# ---------------------------------------------------------------------------
# Long-path awareness (issue #26)
#
# Windows caps paths at 260 chars (MAX_PATH) for any process whose
# embedded application manifest does NOT declare <longPathAware>true.
# A JP user with a long CJK username + the deep HF-cache model path can
# brush against that ceiling; declaring longPathAware in the exe
# manifest is the per-process opt-in that, combined with the system
# LongPathsEnabled registry key, lifts the cap. PyInstaller writes a
# longPathAware manifest by default since PR #5424, but a future spec
# tweak (a hand-rolled manifest=, or disabling it) would silently drop
# it. This test asserts the SHIPPED exe carries the declaration.
# ---------------------------------------------------------------------------

_RT_MANIFEST = 24  # Windows resource type for application manifests

# Match the longPathAware ELEMENT BODY, not just the token. The real
# PyInstaller manifest form is
# ``<longPathAware xmlns="...">true</longPathAware>`` — note the xmlns
# attribute sits between the tag name and ``>``, so a naive
# ``longpathaware>\s*true`` never matches. This regex (applied to the
# lower-cased manifest):
#   * tolerates attributes inside the opening tag ([^>]*),
#   * tolerates an optional namespace prefix on the tag name ([\w:]*),
#   * binds ``true`` to THIS element's body (\b...>\s*true\s*</),
# so a ``<longPathAware>false</longPathAware>`` — or a stripped element
# with a stray ``true`` elsewhere (e.g. a dpiAware element) — does NOT
# satisfy it. A bare ``"true" in manifest.lower()`` would be satisfied
# independent of the longPathAware value, which is why the element body
# is matched explicitly.
_LONG_PATH_AWARE_TRUE_RE = re.compile(
    r"<\s*[\w:]*longpathaware\b[^>]*>\s*true\s*<\s*/",
)


def _extract_manifest(pe) -> str:
    """Return the embedded RT_MANIFEST XML from an open pefile.PE, or "".

    Walks the three-level PE resource tree (type -> name/id -> language)
    under DIRECTORY_ENTRY_RESOURCE, finds the RT_MANIFEST (type 24) leaf,
    and decodes its bytes as UTF-8. Returns "" — never raises — on any
    resource tree that lacks the expected ``.directory`` / ``.data``
    nodes at any level (a malformed or unusual PE), so the caller sees a
    clean "no manifest" rather than an AttributeError.
    """
    if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        return ""
    for type_entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        if getattr(type_entry, "id", None) != _RT_MANIFEST:
            continue
        if not hasattr(type_entry, "directory"):
            continue
        for name_entry in type_entry.directory.entries:
            if not hasattr(name_entry, "directory"):
                continue
            for lang_entry in name_entry.directory.entries:
                if not hasattr(lang_entry, "data"):
                    continue
                rva = lang_entry.data.struct.OffsetToData
                size = lang_entry.data.struct.Size
                return pe.get_data(rva, size).decode("utf-8", errors="replace")
    return ""


def test_frozen_exe_manifest_declares_long_path_aware() -> None:
    """The shipped yamabiko.exe manifest declares <longPathAware>true.

    Without it, the process is MAX_PATH(260)-capped even when the system
    LongPathsEnabled key is set, so a JP user with a long CJK username
    could fail to load a deep-cache model path. PyInstaller writes this
    by default; this guards against a spec change dropping it.

    Asserts on the longPathAware ELEMENT BODY (via
    ``_LONG_PATH_AWARE_TRUE_RE``), not the bare presence of the
    substrings ``longPathAware`` and ``true`` — those are independently
    satisfiable by other manifest elements.

    Seeded-defect check: take the extracted manifest, replace the
    longPathAware body ``true`` with ``false``, and the regex no longer
    matches -> red. Equivalently, a spec change that points
    EXE(manifest=...) at a hand-rolled manifest declaring
    ``<longPathAware>false</longPathAware>`` (or omitting the element)
    and a rebuild turns this red.
    """
    _skip_if_no_bundle()

    import pefile

    pe = pefile.PE(str(_FROZEN_EXE), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
            ],
        )
        manifest = _extract_manifest(pe)
    finally:
        pe.close()

    assert manifest, (
        "yamabiko.exe carries no embedded RT_MANIFEST resource at all — "
        "PyInstaller normally embeds one; a missing manifest also means "
        "no longPathAware opt-in (issue #26)"
    )
    assert _LONG_PATH_AWARE_TRUE_RE.search(manifest.lower()), (
        "yamabiko.exe manifest does not declare <longPathAware>true</longPathAware>; "
        "the process is MAX_PATH(260)-capped, so a JP user with a long "
        "CJK username may fail to load a deep model path even with the "
        "system LongPathsEnabled key set.\n"
        f"manifest:\n{manifest}"
    )
