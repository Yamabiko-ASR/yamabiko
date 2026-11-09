# -*- mode: python ; coding: utf-8 -*-
"""Yamabiko PyInstaller spec.

Onedir build. Entrypoint = yamabiko/watchdog.py (the Start Menu shortcut
target). The watchdog spawns yamabiko/main.py as a subprocess, so the
spec must collect everything main.py imports too.

Run from repo root:
    pyinstaller installer/yamabiko.spec --noconfirm

Output: dist/yamabiko/yamabiko.exe + supporting tree.
"""
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs, copy_metadata,
)

REPO = Path(SPECPATH).parent  # SPECPATH set by PyInstaller
sys.path.insert(0, str(REPO))

from scripts.bundle_inventory import InventoryRoots, write_analysis_inventory

# --- Imports from main.py + watchdog.py we need to force-collect.
# torch is intentionally not collected. The VAD loads the raw ONNX via
# onnxruntime directly, and faster-whisper uses pure `tokenizers`
# (Rust) instead of `transformers.AutoTokenizer` -- which would drag torch
# in even for tokenizer-only workflows. With no Python-level torch
# references left, we skip collection entirely; this removes ~3 GB of CUDA
# DLLs from the bundle.
ct2_binaries = collect_dynamic_libs("ctranslate2")

# CTranslate2 on Windows dynamically loads cublas64_12.dll at encode
# time and cuDNN 9 sub-DLLs (cudnn_ops/cnn/graph/engines_*/heuristic/adv)
# lazily when convolutions run. The ctranslate2 wheel ships only
# cudnn64_9.dll (the front loader) and libiomp5md.dll — no cuBLAS, no
# cuDNN sub-DLLs. Torch used to supply them via torch/lib/ before we
# dropped it from the bundle, which caused "Library cublas64_12.dll is
# not found" in the shipped beta. The fix is the NVIDIA redistributable
# wheels, pinned in yamabiko/requirements.txt. PyInstaller's
# collect_dynamic_libs flattens them into the onedir root next to
# ctranslate2.dll, where the Windows loader finds them without any
# PATH or os.add_dll_directory gymnastics.
# collect_dynamic_libs preserves the wheel's package layout by default,
# which would put the DLLs under _internal/nvidia/cublas/bin/ etc.
# That doesn't work: the PyInstaller bootloader adds _internal/ (root)
# to the Windows DLL search path but NOT its subdirectories, and CT2
# calls LoadLibrary("cublas64_12.dll") without a full path. Flatten
# every NVIDIA DLL to "." so it lands next to ctranslate2.dll's loader
# in _internal/ root where the search path can find it.
def _flatten_nvidia(pkg: str) -> list:
    return [(src, ".") for (src, _dest) in collect_dynamic_libs(pkg)]

cublas_binaries = _flatten_nvidia("nvidia.cublas")
cudnn_binaries = _flatten_nvidia("nvidia.cudnn")
# NVRTC is a transitive dep: cublasLt64_12.dll imports nvrtc64_120_0.dll,
# and cudnn_engines_precompiled64_9.dll imports both. Collecting cublas
# alone is not enough — the DLL would load but throw on first op that
# needs JIT kernel compilation. Verified by scanning the wheels' own
# DLL import strings.
nvrtc_binaries = _flatten_nvidia("nvidia.cuda_nvrtc")
if not cublas_binaries:
    raise SystemExit(
        "collect_dynamic_libs('nvidia.cublas') returned 0 DLLs — "
        "is nvidia-cublas-cu12 installed? See yamabiko/requirements.txt"
    )
if not cudnn_binaries:
    raise SystemExit(
        "collect_dynamic_libs('nvidia.cudnn') returned 0 DLLs — "
        "is nvidia-cudnn-cu12 installed? See yamabiko/requirements.txt"
    )
if not nvrtc_binaries:
    raise SystemExit(
        "collect_dynamic_libs('nvidia.cuda_nvrtc') returned 0 DLLs — "
        "is nvidia-cuda-nvrtc-cu12 installed? See yamabiko/requirements.txt"
    )
# include_py_files=True is LOAD-BEARING. unidic_lite ships its DICDIR
# constant from __init__.py (`from .unidic import DICDIR, VERSION`),
# and fugashi's Cython `try_import_unidic` reads it at runtime. Plain
# collect_data_files grabs the dicdir/ tree but not the .py source, so
# PyInstaller creates `_internal/unidic_lite/dicdir/` on disk and
# Python treats the parent directory as a PEP 420 namespace package
# when fugashi does `import unidic_lite` — producing an empty module
# with no DICDIR. The namespace package SHADOWS the PYZ-embedded
# version. Including the .py source files in the on-disk tree
# reinstates the regular package.
unidic_datas = collect_data_files("unidic_lite", include_py_files=True)
# cutlet ships exceptions.tsv in its package dir and reads it via
# Path(__file__).parent / 'exceptions.tsv' at Tagger init time.
# cutlet's .py files ride into PYZ via the `cutlet` hiddenimport
# below, but data files never enter PYZ — they must be explicitly
# collected onto disk. Without this, cutlet.Cutlet() raises
# FileNotFoundError on startup. include_py_files=True is defensive:
# it also mirrors the PYZ-embedded .py files onto disk, so if any
# future PyInstaller version changes import precedence to prefer
# filesystem over PYZ (as happened with the unidic_lite namespace-
# package shadowing), cutlet keeps working.
cutlet_datas = collect_data_files("cutlet", include_py_files=True)
# include_py_files=True for the SAME namespace-shadowing reason as
# unidic_lite above: sudachipy resolves the dictionary by importing
# sudachidict_core and reading __file__, which is None on the PEP 420
# namespace package a data-only on-disk tree creates. The
# hooks-contrib hook-sudachipy also collects data-only, so this spec
# entry is the belt-and-braces that keeps the on-disk tree a REGULAR
# package. A ctor failure here is swallowed into "disabling
# romanization" at runtime — furigana, romaji and the series-gazetteer
# hints all silently dead — so only the frozen E2E's "romanization
# pipeline enabled" marker can catch a regression.
sudachi_datas = collect_data_files("sudachidict_core", include_py_files=True)

# Yamabiko selects these Uvicorn backends explicitly in main.py. Uvicorn
# resolves the selector strings dynamically, so Analysis needs the matching
# modules named here. Broad collect_all() also ships both CLIs, alternate
# protocols, workers, and FastAPI's package-local agent instructions.
UVICORN_HIDDENIMPORTS = [
    "uvicorn.lifespan.on",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.websockets_impl",
]
WEB_SERVER_EXCLUDES = [
    "fastapi.__main__",
    "fastapi.cli",
    "uvicorn.__main__",
    "uvicorn.workers",
    "uvicorn.lifespan.off",
    "uvicorn.loops.auto",
    "uvicorn.loops.uvloop",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_sansio_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
]

# collect_all preserves hf_xet's distribution metadata and extension hidden
# imports.  Both are load-bearing: Hub checks importlib.metadata.version(
# "hf_xet") before enabling Xet, while hf_xet.hf_xet is the native Rust
# extension.  Losing either silently falls back to single-stream HTTP.
hf_xet_datas, hf_xet_binaries, hf_xet_hiddenimports = collect_all("hf_xet")
# Keep only Yamabiko's reviewed Hub download/Xet surface. Broad
# collect_all("huggingface_hub") also ships its unused CLI and inference
# client trees; ordinary Analysis may still retain any module reachable from
# these roots.
HF_HIDDENIMPORTS = [
    "huggingface_hub.file_download",
    "huggingface_hub._snapshot_download",
    "huggingface_hub.utils._xet",
    "huggingface_hub.utils._xet_progress_reporting",
    "huggingface_hub.utils.tqdm",
    "hf_xet",
    "hf_xet.hf_xet",
]

datas = [
    (str(REPO / "yamabiko" / "default_config.yaml"),
     "yamabiko"),
    (str(REPO / "assets" / "icon.ico"), "assets"),
    (str(REPO / "assets" / "icon.png"), "assets"),
    (str(REPO / "LICENSE"), "."),
    (str(REPO / "NOTICE"), "."),
    (str(REPO / "README.md"), "."),
    (str(REPO / "CHANGELOG.md"), "."),
    (str(REPO / "yamabiko" / "display" / "static"),
     "yamabiko/display/static"),
    (str(REPO / "yamabiko" / "overlay_app" / "assets"), "yamabiko/overlay_app/assets"),
    (str(REPO / "yamabiko" / "glossaries"),
     "yamabiko/glossaries"),
    # The sound tagger graph and its MIT license. Loaded by
    # SileroChunker through vad/sound_tagger.default_model_path(), which
    # resolves paths.bundle_data_dir() / "vad" / "data".
    (str(REPO / "yamabiko" / "vad" / "data"),
     "yamabiko/vad/data"),
    *copy_metadata("faster-whisper"),
    *unidic_datas, *cutlet_datas, *sudachi_datas,
    *hf_xet_datas,
]

binaries = [
    *ct2_binaries,
    *cublas_binaries, *cudnn_binaries, *nvrtc_binaries,
    *hf_xet_binaries,
    # Vendored llama.cpp + DLLs are added by installer/build.ps1
    # AFTER pyinstaller runs, since they live outside the Python
    # package and don't need PyInstaller's import hooks.
]

hiddenimports = [
    *UVICORN_HIDDENIMPORTS,
    *hf_xet_hiddenimports, *HF_HIDDENIMPORTS,
    "yamabiko.main",
    "yamabiko.watchdog",
    # Some PySide6 plugins need explicit collection on Windows:
    "PySide6.QtNetwork",
    "PySide6.QtWebSockets",
    # Lazy / conditional imports PyInstaller's static analyzer misses:
    # - huggingface_hub: lazy in wizard_pages/downloads.py + asr/asr_client.py
    # - hf_xet: native module huggingface_hub dispatches to for Xet-enabled
    #   repos (all HF repos as of 2026). Without it, downloads fall back to
    #   single-stream HTTP capped around 3 MB/s.
    # - onnxruntime: lazy in vad/silero_chunker.py
    # - faster_whisper: imported dynamically through the NumPy audio
    #   compatibility seam after its child audio module is pre-seeded
    # - tokenizers: used by faster-whisper (asr/asr_client.py)
    # - keyboard: lazy in hotkeys.py
    # - cutlet/fugashi/sudachipy: imported in romanization/*.py
    # Keep the two download parents direct (even though the helper lists also
    # reach them) so preflight can prove release-critical membership without
    # executing this spec or trusting opaque starred hook output.
    "huggingface_hub",
    "hf_xet",
    "onnxruntime",
    # Capture imports python-soxr directly; name both the wrapper and native
    # extension so analyzer drift cannot omit libsoxr from a frozen build.
    "soxr",
    "soxr.soxr_ext",
    "faster_whisper",
    "tokenizers",
    "keyboard",
    "cutlet",
    "fugashi",
    "sudachipy",
    # fast_bunkai is imported lazily inside
    # yamabiko.sentencing.splitter:34 via a function-body `from
    # fast_bunkai import FastBunkai`. PyInstaller's static AST
    # analyzer catches that today, but if someone ever refactors
    # _ensure_loaded to use importlib.import_module or string-based
    # lookup, the dep would silently drop. List it explicitly
    # so it can't be lost to a refactor.
    "fast_bunkai",
    # Qt desktop overlay, spawned as a child process by the PIPELINE
    # (main.py `_spawn_overlay` / `_spawn_overlay_in_job`) via
    # `yamabiko.exe --run-overlay`; the argv is built by watchdog.py's
    # `_build_overlay_cmd` helper, but the watchdog does NOT spawn it --
    # the process-lifecycle overhaul gave the pipeline ownership of the
    # overlay so it dies with its parent via the inner Job Object.
    # PyInstaller's static analyzer pulls
    # in yamabiko.overlay_app.style_defaults (imported by main.py) but
    # not the rest of the package -- the other modules are only reached
    # through overlay_app.__main__'s import graph, which isn't visible
    # until runtime because the dispatch happens via an argv check.
    "yamabiko.overlay_app.__main__",
    "yamabiko.overlay_app.overlay_window",
    "yamabiko.overlay_app.segmented",
    "yamabiko.overlay_app.controls",
    "yamabiko.overlay_app.keycaps",
    "yamabiko.overlay_app.landscape",
    "yamabiko.overlay_app.preview_monitor",
    "yamabiko.overlay_app.seal",
    "yamabiko.overlay_app.settings_dialogs",
    "yamabiko.overlay_app.shortcut_field",
    "yamabiko.overlay_app.window_chrome",
    "yamabiko.overlay_app.telop_banner",
    "yamabiko.overlay_app.telop_look",
    "yamabiko.overlay_app.telop_stack",
    "yamabiko.overlay_app.telop_text",
    "yamabiko.overlay_app.series_palette",
    "yamabiko.overlay_app.series_palette_parts",
    "yamabiko.overlay_app.settings_window",
    "yamabiko.overlay_app.theme",
    "yamabiko.overlay_app.fonts",
    "yamabiko.overlay_app.tray",
    "yamabiko.overlay_app.tray_flyout",
    "yamabiko.overlay_app.tray_placement",
    "yamabiko.overlay_app.tray_icons",
    "yamabiko.overlay_app.problem_view",
    "yamabiko.overlay_app.ws_client",
    # The startup error window: reached only through the function-body
    # import in error_dialog.show_startup_error. PyInstaller's analyzer
    # finds that import today; listed (like fast_bunkai above) so a refactor
    # of the lazy import can't drop the window from the bundle.
    "yamabiko.ui.error_window",
]

block_cipher = None

a = Analysis(
    [str(REPO / "yamabiko" / "watchdog.py")],
    pathex=[str(REPO)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Trim the bundle. Add modules here ONLY after verifying via
        # `python -c "import X"` in the bundled exe that they're not
        # needed at runtime.
        "tkinter", "matplotlib",
        # Belt-and-braces: torch is not part of the runtime but may be
        # installed in the dev venv. Exclude it outright so PyInstaller's
        # static analyzer can't rediscover it via any new transitive import.
        "torch", "torchvision", "torchaudio",
        # The frozen wizard E2E drives the candidate from the outside with
        # pywinauto.  Neither that dev-only package nor its COM/PyWin32 stack
        # belongs in Yamabiko.  Once installed in the build venv, PyWin32 also
        # satisfies otherwise-optional imports used only by NTEventLogHandler
        # and numpy.testing's Windows performance helpers.
        "pywinauto", "comtypes",
        "win32evtlog", "win32evtlogutil", "win32pdh",
        # huggingface_hub's generated TYPE_CHECKING block statically imports
        # its CLI and hosted-inference clients.  PyInstaller sees those dead
        # imports even though Yamabiko's runtime lazy loader only resolves the
        # download/cache APIs listed in HF_HIDDENIMPORTS above.
        "huggingface_hub.cli", "huggingface_hub.inference",
        # Transformers drags torch back in at import time — exclude it
        # too. We use pure `tokenizers` (via faster-whisper).
        "transformers",
        # Yamabiko always supplies NumPy audio through its compatibility seam.
        # Exclude faster-whisper's path/file decoder and PyAV itself so the
        # av hook cannot collect PyAV's bundled FFmpeg/x264/x265 DLL stack.
        "faster_whisper.audio", "av",
        # The runtime pins one concrete Uvicorn loop, HTTP, WebSocket, and
        # lifespan backend. Its alternate implementations appear in
        # TYPE_CHECKING imports and otherwise survive static Analysis.
        *WEB_SERVER_EXCLUDES,
        # Transitive bloat — grepped yamabiko/ + tests/, none referenced
        "nltk", "nltk_data",
        "numba", "llvmlite",
        "scipy",  # transitive from numpy; runtime doesn't call it
        "pandas",  # transitive; no yamabiko use
        "sklearn",  # transitive; no yamabiko use
        "PIL",  # transitive; no yamabiko use (icons are pre-built .ico)
        "IPython", "jupyter", "notebook",
    ],
    cipher=block_cipher,
    noarchive=False,
    # python-soxr is LGPL-2.1-or-later. Ship its Python modules as plain .py
    # files under _internal/soxr/ instead of inside the PYZ archive, so a user
    # can replace or modify them; the native soxr_ext extension is a separate
    # file there either way. NOTICE points at the published source.
    module_collection_mode={"soxr": "py"},
)


def _is_host_runtime_dll(dest_name: str) -> bool:
    """DLLs PyInstaller may resolve from the build host's PATH, not from the
    venv or the Python install: the x64-named OpenSSL 3 pair that Qt's TLS
    backend looks for, and the UCRT (``ucrtbase.dll`` plus its
    ``api-ms-win-*`` forwarders). Windows 11 ships UCRT, Yamabiko makes no
    TLS connection through Qt, and CPython's own ``libssl-3.dll`` /
    ``libcrypto-3.dll`` are unaffected. Dropping them keeps a build on a
    tool-laden host (the CI runner) identical to a release build."""
    name = Path(dest_name).name.lower()
    return (
        name in ("libssl-3-x64.dll", "libcrypto-3-x64.dll", "ucrtbase.dll")
        or (name.startswith("api-ms-win-") and name.endswith(".dll"))
    )


_host_runtime_dlls = sorted(
    entry[0] for entry in a.binaries if _is_host_runtime_dll(entry[0])
)
if _host_runtime_dlls:
    print(
        "yamabiko.spec: dropping host-PATH runtime DLLs: "
        + ", ".join(_host_runtime_dlls)
    )
a.binaries = [entry for entry in a.binaries if not _is_host_runtime_dll(entry[0])]
write_analysis_inventory(
    a,
    Path(workpath) / "bundle-inventory.json",
    roots=InventoryRoots(
        repo_root=REPO.resolve(),
        site_packages=Path(sysconfig.get_path("purelib")),
        python_root=Path(sys.base_prefix),
    ),
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="yamabiko",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                 # GUI app (watchdog still writes to logs)
    disable_windowed_traceback=False,
    icon=str(REPO / "assets" / "icon.ico"),
    version=str(REPO / "installer" / "version_info.txt"),  # PE VERSIONINFO resource
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="yamabiko",
)
