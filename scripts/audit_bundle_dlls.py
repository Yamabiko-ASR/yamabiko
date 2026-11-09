"""Audit every native binary in the built bundle for missing DLL deps.

Walks dist/yamabiko/ for every .dll / .pyd / .exe, parses the PE IMPORT
directory of each, and reports every imported DLL that is NOT either:

  - Present in dist/yamabiko/ (anywhere under the tree EXCEPT the
    _internal/llama subprocess subtree, which has its own search path),
  - A Windows system DLL (kernel32, ntdll, api-ms-win-*, etc.),
  - A known NVIDIA driver DLL (nvcuda.dll, nvapi64.dll),
  - The bundle's own Python interpreter DLL.

The goal is to catch a class of regression: a DLL
the bundle transitively depends on, but that nothing in the spec
explicitly collects, which happens to be present on the dev machine
and therefore invisible to tests that run against the dev venv.

Usage:
    python scripts/audit_bundle_dlls.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pefile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "dist" / "yamabiko"


def _norm(name: str) -> str:
    return name.lower().strip()


# --- Classification of "not in bundle but OK to ignore" -------------------

# Windows system DLLs. These ship with Windows; the target platform is
# Windows 11. Expanding this list is fine — false positives (listing a
# Windows DLL as "ignored" when it's also bundled by us) are harmless.
_WINDOWS_SYSTEM = {
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "user32.dll",
    "gdi32.dll", "gdi32full.dll", "advapi32.dll", "shell32.dll",
    "ole32.dll", "oleaut32.dll", "comctl32.dll", "comdlg32.dll",
    "shlwapi.dll", "shcore.dll", "version.dll", "winmm.dll",
    "ws2_32.dll", "wsock32.dll", "wininet.dll", "winhttp.dll",
    "crypt32.dll", "cryptbase.dll", "bcrypt.dll", "bcryptprimitives.dll",
    "ncrypt.dll", "secur32.dll", "sspicli.dll",
    "msvcrt.dll",  # legacy; MSVC code uses vcruntime140
    "rpcrt4.dll", "sechost.dll", "combase.dll", "dwmapi.dll",
    "uxtheme.dll", "imm32.dll", "msctf.dll",
    "iphlpapi.dll", "dnsapi.dll", "mswsock.dll",
    "psapi.dll", "dbghelp.dll", "dbgcore.dll",
    "d3d11.dll", "d3d12.dll", "d3d9.dll", "dxgi.dll",
    "d2d1.dll", "dwrite.dll", "windowscodecs.dll",
    "mf.dll", "mfplat.dll", "mfreadwrite.dll", "mfcore.dll",
    "avrt.dll", "audioses.dll", "mmdevapi.dll",  # audio stack
    "pdh.dll",  # perf counters
    "powrprof.dll", "setupapi.dll", "cfgmgr32.dll", "devobj.dll",
    "winmmbase.dll",  # multimedia base
    "dsound.dll",  # directsound, pyaudio may query
    "userenv.dll", "profapi.dll",
    "msasn1.dll",
    "wtsapi32.dll",
    "urlmon.dll", "propsys.dll",
    "winnsi.dll",  # netshell info
    "gdiplus.dll",
    "activeds.dll", "adsldpc.dll",
    "d3dcompiler_47.dll",  # present on Win10+
    "opengl32.dll", "glu32.dll",
    "netapi32.dll",
    # Additional Windows system DLLs present on stock Windows 10/11.
    "authz.dll",         # security descriptor APIs
    "wintrust.dll",      # code-signing verification
    "mpr.dll",           # multi-provider router (net shares)
    "imagehlp.dll",      # PE inspection (used by Intel OpenMP for self-introspection)
    "icuuc.dll",         # ICU — Qt6Core's Unicode layer
    "uiautomationcore.dll",  # UI Automation, used by qdirect2d/qwindows
    "avicap32.dll",      # legacy AVI capture, ffmpeg queries optionally
    "dxva2.dll",         # DirectX Video Acceleration, queried by ffmpeg
}

# API set forwarders on Windows 10/11. Any name matching api-ms-win-*,
# ext-ms-*, or similar is a virtual DLL resolved by the loader against
# the real OS module.
_API_SET_RE = re.compile(
    r"^(?:api-ms-win|ext-ms|api-ms)-", re.IGNORECASE
)

# NVIDIA driver-supplied DLLs. Ship with the user's GPU driver install,
# NOT our concern. sm_75+ requirement means the user has a driver.
_NV_DRIVER = {
    "nvcuda.dll", "nvapi.dll", "nvapi64.dll",
    "nvml.dll",  # driver-supplied NVML
    "nvcuda64.dll",
}


def is_system_dll(name: str) -> bool:
    """Return whether *name* is supplied by the supported Windows platform.

    This public predicate covers both stock Windows/API-set modules and the
    NVIDIA display-driver DLLs supplied by the required GPU driver.  Bundle
    closure callers therefore need one complete, stable classification API.
    """

    n = _norm(name)
    if n in _WINDOWS_SYSTEM:
        return True
    if _API_SET_RE.match(n):
        return True
    return n in _NV_DRIVER


# Optional backend DLLs: probed via LoadLibraryEx with graceful
# fallback by their caller. Missing these never breaks anything; they
# let the caller pick a faster implementation when present and fall
# back to a slower path when not. Treating them as "missing" produces
# false positives indistinguishable from a genuine missing dependency.
_OPTIONAL_BACKENDS = {
    # Intel MKL dispatchers (numpy linear-algebra backend; graceful
    # fallback to scipy_openblas which we DO bundle).
    "mkl_avx2.2.dll", "mkl_avx512.2.dll", "mkl_def.2.dll", "mkl_mc3.2.dll",
    "mkl_vml_avx2.2.dll", "mkl_vml_avx512.2.dll", "mkl_vml_cmpt.2.dll",
    "mkl_vml_def.2.dll", "mkl_vml_mc3.2.dll",
    # Intel OpenMP runtime internals — libiomp5md.dll self-introspects
    # + tries to find Intel tooling at runtime. All optional.
    "libiomp5ui.dll", "libittnotify.dll", "tbbmalloc.dll",
    "ifcore_msg.dll", "irc_msg.dll", "libicaf.dll", "libmui.dll",
    "utclibiomp5md_db.dll",
    # Visual Studio OpenMP runtime — if missing, code uses Intel OpenMP
    # (which IS bundled). Mutually exclusive alternatives.
    "vcomp140.dll",
    # cuDNN deprecated fallback and internal profiling shims — cuDNN
    # tries to load these and no-ops if absent.
    "zlibwapi.dll", "cask_profile_interface.dll",
    # DirectX 12 optional debug / compiler shims loaded by Qt6Gui for
    # advanced rendering paths we don't use.
    "dxcompiler.dll", "dxgidebug.dll", "dxtn.dll", "amfrt64.dll",
    # NVIDIA video codecs probed by ffmpeg when present. Yamabiko's
    # audio-only pipeline never hits this path.
    "nvcuvid.dll", "nvencodeapi64.dll",
    # libx265 internal bit-depth dispatchers. libx265.dll loads these
    # on demand for 8/10/12-bit encode paths. Yamabiko doesn't encode
    # video; transitive through PyAV.
    "libx265_main.dll", "libx265_main10.dll", "libx265_main12.dll",
    # PortAudio host-API dispatch: libportaudio64bit*.dll dlopen
    # portaudio_x64.dll if present, fall back to their internal impl.
    "portaudio_x64.dll",
    # NVRTC alt / unversioned variants. The bundle has
    # nvrtc-builtins64_129.dll (the real one); these are backwards-
    # compat aliases the wheel's own imports reference with graceful
    # loader resolution to the versioned name.
    "nvrtc-builtins.alt64_129.dll", "nvrtc-builtins.dll",
}


# PyInstaller renames duplicated DLLs with a content-hash suffix for
# deduplication:  foo.dll   ->   foo-<32 hex chars>.dll
# The hashed form is what's on disk; the un-hashed form appears in the
# import tables of sibling DLLs (PyInstaller patches load paths at
# runtime via bootloader shim). Suppress the un-hashed "missing"
# report when the hashed sibling is present.
_HASHED_RENAME_RE = re.compile(r"^(?P<base>.+?)-[0-9a-f]{32}\.dll$")


def _is_pyinstaller_hashed_variant(
    name: str, present: dict[str, Path]
) -> bool:
    """True if bundle contains foo-<32hex>.dll for a missing foo.dll."""
    base = _norm(name)
    if not base.endswith(".dll"):
        return False
    stem = base[:-4]
    target_re = re.compile(
        rf"^{re.escape(stem)}-[0-9a-f]{{32}}\.dll$"
    )
    return any(target_re.match(key) for key in present)


# CPython extension modules (.pyd) appear in their own PE import table
# as "foo.dll" even though the file on disk is "foo.pyd" or
# "foo.cp312-win_amd64.pyd". The loader resolves it via the import
# system, not the DLL search path.
_PYD_TAG_RE = re.compile(r"^(?P<stem>.+?)(?:\.cp\d+-[a-z0-9_-]+)?\.pyd$")


def _is_pyd_self_import(name: str, natives: list[Path]) -> bool:
    """True if a missing 'X.dll' matches a 'X.pyd' / 'X.cp*.pyd' in bundle."""
    base = _norm(name)
    if not base.endswith(".dll"):
        return False
    stem = base[:-4]
    for p in natives:
        if p.suffix.lower() != ".pyd":
            continue
        m = _PYD_TAG_RE.match(_norm(p.name))
        if m and m.group("stem") == stem:
            return True
    return False


def _is_string_scan_artifact(name: str) -> bool:
    """True for names that are obviously PE-string-scan false positives.

    Well-formed DLL names start with a letter or underscore. Names
    starting with punctuation (``.``, ``-``) or a digit almost always
    come from the string scanner tripping over surrounding bytes in
    the binary (e.g. a byte-prefix leaking into the regex match).
    """
    if not name:
        return True
    first = name[0]
    if not first.isascii():
        return True
    return not (first.isalpha() or first == "_")


def _is_stringscan_prefix_of_present(
    name: str, present: dict[str, Path]
) -> bool:
    """True if trimming one leading char yields a present DLL.

    The ``_DLL_STRING_RE`` scanner in this file occasionally absorbs a
    preceding byte into the name (e.g. the scanner sees
    ``"fctranslate2.dll"`` because the real string ``"ctranslate2.dll"``
    was immediately preceded by an ``f`` byte). If the name one
    character shorter is actually present in the bundle, the original
    is almost certainly a scan artifact.
    """
    n = _norm(name)
    if len(n) < 5:  # need at least 'X.dll'
        return False
    trimmed = n[1:]
    return trimmed in present


# NOTE: there was a broader `_importers_are_all_pyds`
# suppression here ("skip any missing DLL whose importers are all
# .pyds") justified by the same pyo3 rationale as _is_pyd_self_import
# above. It was deleted: the stem-matching rule already covers the
# genuine pyo3 convention, and the blanket rule silently swallowed
# real deps imported only by extension modules (onnxruntime.dll via
# onnxruntime_pybind11_state.pyd, libffi-8.dll via _ctypes.pyd,
# libssl-3.dll via _ssl.pyd). If a residual false positive surfaces,
# add it to a NAMED allowlist (_OPTIONAL_BACKENDS) with a rationale,
# not a heuristic.


# Subtrees whose PE imports we don't audit: their contents run in a
# separate subprocess with its own DLL search path, so their transitive
# deps aren't a dep of yamabiko.exe. Currently just the vendored
# llama-server bundle (runs as `llama-server.exe`, a child process).
_SUBPROCESS_SUBTREES = ("_internal/llama", "_internal\\llama")


def _is_subprocess_subtree(p: Path, root: Path) -> bool:
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        return False
    return any(rel.startswith(sub.replace("\\", "/")) for sub in _SUBPROCESS_SUBTREES)


# --- PE parsing -------------------------------------------------------------

_DLL_STRING_RE = re.compile(rb"[A-Za-z0-9_.\-]{4,64}\.(?:dll|DLL)\x00")


def _pe_export_name(path: Path) -> str | None:
    """The PE's own export-directory module name (normalized), or None.

    maturin-built pyds keep the Rust cdylib's ORIGINAL name here:
    jiter.cp312-win_amd64.pyd exports as 'jiter_python.dll',
    _fast_bunkai.pyd as 'fast_bunkai_native.dll'. That name sits
    NUL-terminated in the binary, so the string scanner would read it
    as a LoadLibrary dependency — and _is_pyd_self_import can't catch
    it because the dll stem differs from the on-disk pyd stem.
    """
    try:
        pe = pefile.PE(str(path), fast_load=True)
    except pefile.PEFormatError:
        return None
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY[
                "IMAGE_DIRECTORY_ENTRY_EXPORT"]]
        )
        exp = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
        raw = getattr(exp, "name", None) if exp is not None else None
        if not raw:
            return None
        return _norm(raw.decode("ascii", "ignore"))
    except Exception:
        return None
    finally:
        pe.close()


def scan_runtime_loaded(path: Path) -> list[str]:
    """Scan binary for DLL filename strings referenced via LoadLibrary.

    PE static imports miss runtime-loaded DLLs (CTranslate2 calls
    LoadLibraryA("cublas64_12.dll") and that name only appears as a
    C-string in the data section, not in the IMPORT directory).
    Complement parse_imports() with a string scan.

    Only DLL-suffixed strings terminated by NUL (C-string convention)
    are accepted — that filters out stray name fragments that happen
    to look like DLL names.

    The one string deliberately dropped is the PE's OWN export-directory
    name (see _pe_export_name): a module's identity string is
    definitionally not a dependency. This is exact per-file metadata,
    NOT the class-based all-pyd-importers heuristic that was deleted —
    it cannot mask a real dep like onnxruntime.dll, whose name never
    equals its importer's export name.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return []
    found: set[str] = set()
    for m in _DLL_STRING_RE.finditer(data):
        # Strip the trailing NUL.
        name = m.group(0)[:-1].decode("ascii", "ignore")
        if name:
            found.add(name)
    own = _pe_export_name(path)
    if own:
        found = {n for n in found if _norm(n) != own}
    return sorted(found)


class PEImportParseError(ValueError):
    """A native file could not be inspected as a complete PE image."""


def parse_imports(path: Path) -> list[str]:
    """Return static and delay-loaded imports, failing closed on parse errors."""

    try:
        pe = pefile.PE(str(path), fast_load=True)
    except Exception as exc:
        raise PEImportParseError(
            f"PE imports could not be parsed for {path.name}"
        ) from exc
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY[
                "IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
        names: list[str] = []
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            raw = entry.dll
            if isinstance(raw, bytes):
                try:
                    names.append(raw.decode("ascii"))
                except UnicodeDecodeError:
                    names.append(raw.decode("latin-1"))
            else:
                names.append(str(raw))

        # Also parse the delay-load IMPORT directory. CTranslate2 and some
        # CUDA DLLs use delay-loaded imports for optional backends — these
        # aren't resolved until first call but ARE required when that call
        # runs, so we MUST consider them.
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY[
                "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]]
        )
        for entry in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []) or []:
            raw = entry.dll
            if isinstance(raw, bytes):
                try:
                    names.append(raw.decode("ascii"))
                except UnicodeDecodeError:
                    names.append(raw.decode("latin-1"))
            else:
                names.append(str(raw))
        return names
    except Exception as exc:
        raise PEImportParseError(
            f"PE imports could not be parsed for {path.name}"
        ) from exc
    finally:
        pe.close()


# Exact text installer/ci-prepare-vendor-stubs.ps1 writes into every vendored
# llama.cpp file for CI smoke builds. Keep the two in sync.
_CI_LLAMA_STUB_TEXT = b"CI stub (not a real binary)"
_CI_LLAMA_STUB_CONTENTS = frozenset(
    prefix + _CI_LLAMA_STUB_TEXT + newline
    for prefix in (b"", b"\xef\xbb\xbf")  # Set-Content -Encoding UTF8 BOM varies by shell
    for newline in (b"", b"\n", b"\r\n")
)
_CI_LLAMA_STUB_MAX_BYTES = max(len(content) for content in _CI_LLAMA_STUB_CONTENTS)


def is_allowed_ci_llama_stub(relative_path: str, path: Path) -> bool:
    """True only for an exact CI text stub when YAMABIKO_ALLOW_STUB_LLAMA=1.

    installer-smoke.yml builds on deliberate stubs because the real
    llama.cpp binaries are not in the repo; assert-llama-vendor.ps1 honors
    the same variable. Matching exact content rather than the name keeps a
    corrupt real binary failing in CI, and builds without the variable stay
    strict.
    """
    if os.environ.get("YAMABIKO_ALLOW_STUB_LLAMA") != "1":
        return False
    if not relative_path.casefold().startswith("_internal/llama/"):
        return False
    try:
        with path.open("rb") as fh:
            data = fh.read(_CI_LLAMA_STUB_MAX_BYTES + 1)
    except OSError:
        return False
    return data in _CI_LLAMA_STUB_CONTENTS


# --- Bundle enumeration -----------------------------------------------------

def enumerate_bundle_dlls(
    bundle: Path = BUNDLE,
) -> tuple[dict[str, Path], list[Path]]:
    """Return (dll_name_lower -> first_path, all_native_files).

    all_native_files is every .dll / .pyd / .exe under ``bundle``,
    EXCLUDING the subprocess subtrees (_internal/llama): those run as
    a child process with their own DLL search path, so their files
    neither need auditing as importers nor count as "present" for the
    main tree (a llama-only copy of e.g. msvcp140.dll
    must not mask the same DLL being dropped from _internal/ root,
    where yamabiko.exe actually resolves it).
    """
    present: dict[str, Path] = {}
    natives: list[Path] = []
    for p in bundle.rglob("*"):
        if not p.is_file():
            continue
        if _is_subprocess_subtree(p, bundle):
            continue
        suf = p.suffix.lower()
        if suf in {".dll", ".pyd", ".exe"}:
            natives.append(p)
            if suf in {".dll", ".pyd"}:
                # record under file name; if duplicate, keep first (we
                # only care about presence, not which copy wins).
                key = _norm(p.name)
                if suf == ".pyd":
                    # PYD imports by .pyd name are irrelevant; only
                    # .dll names matter for the "is it present" check.
                    continue
                if key not in present:
                    present[key] = p
    return present, natives


# --- Classification ---------------------------------------------------------

def compute_missing(
    reverse: dict[str, list[Path]],
    present: dict[str, Path],
    natives: list[Path],
) -> list[str]:
    """Classify every imported DLL name into missing / accounted-for.

    Pure function over the maps built by ``enumerate_bundle_dlls`` +
    the import walk, so the suppression rules are unit-testable
    (tests/test_audit_bundle_dlls.py). Order of guards matters only
    for performance; each is independently sufficient to suppress.
    """
    missing: list[str] = []
    for name in sorted(reverse):
        n = _norm(name)
        if n in present:
            continue
        if is_system_dll(n):
            continue
        if _is_string_scan_artifact(n):
            continue
        if _is_stringscan_prefix_of_present(n, present):
            continue
        if n in _OPTIONAL_BACKENDS:
            continue
        if _is_pyinstaller_hashed_variant(n, present):
            continue
        if _is_pyd_self_import(n, natives):
            continue
        missing.append(name)
    return missing


# --- Main -------------------------------------------------------------------

def main() -> int:
    if not BUNDLE.is_dir():
        print(f"FAIL: no bundle at {BUNDLE}", file=sys.stderr)
        return 2

    present, natives = enumerate_bundle_dlls(BUNDLE)
    print(f"Bundle at: {BUNDLE}")
    print(f"Native files: {len(natives)}  "
          f"(DLLs present: {len(present)})")
    print()

    # Gutted-bundle guard: zero native files (empty/wiped dist) or a
    # bundle root without yamabiko.exe means there is nothing shippable
    # to audit -- "no imports, therefore no missing imports" would exit
    # 0 and preflight section 9 would print PASS "Native files: 0" over
    # a bundle that cannot launch. There is deliberately no override:
    # this script audits exactly dist/yamabiko (no partial-tree flags),
    # so an empty or exe-less tree is always a failure.
    if not natives:
        print(f"FAIL: bundle at {BUNDLE} contains no native files "
              f"(.dll/.pyd/.exe) -- empty or gutted dist; nothing to audit",
              file=sys.stderr)
        return 1
    if not (BUNDLE / "yamabiko.exe").is_file():
        print(f"FAIL: bundle at {BUNDLE} lacks yamabiko.exe at its root "
              f"-- partial/gutted dist; rebuild with installer\\build.ps1",
              file=sys.stderr)
        return 1

    # importer -> list of imported dll names
    importers: dict[Path, list[str]] = {}
    # imported dll name (lowercase) -> list of importers
    reverse: dict[str, list[Path]] = defaultdict(list)
    parse_failures: list[str] = []

    # natives already excludes the llama subprocess subtree (its
    # transitive deps aren't deps of yamabiko.exe -- see
    # enumerate_bundle_dlls).
    for p in natives:
        try:
            imports = parse_imports(p)
        except Exception:
            parse_failures.append(p.relative_to(BUNDLE).as_posix())
            continue
        # Extend with runtime-loaded DLLs (LoadLibrary strings). This is
        # how CT2 drags in cublas and how cuDNN drags in its sub-DLLs;
        # PE static analysis alone misses them.
        string_refs = scan_runtime_loaded(p)
        combined = list(dict.fromkeys(imports + string_refs))
        importers[p] = combined
        for dll in combined:
            reverse[_norm(dll)].append(p)

    missing = compute_missing(reverse, present, natives)

    # Group missing by apparent category for readability.
    def _category(n: str) -> str:
        if n.startswith(("vcruntime", "msvcp", "concrt", "vccorlib")):
            return "VC++ runtime"
        if n.startswith(("cudart", "cublas", "cudnn", "cufft", "curand",
                         "cusolver", "cusparse", "nvrtc", "nvjitlink",
                         "nvjpeg", "nppc", "nppi", "npps")):
            return "CUDA runtime / math libs"
        if n.startswith("mkl_"):
            return "Intel MKL"
        if n.startswith("libiomp") or n.startswith("vcomp"):
            return "OpenMP"
        if n.startswith("python"):
            return "Python interpreter"
        if n.startswith(("qt6", "libssl", "libcrypto")):
            return "Qt / OpenSSL"
        return "Other"

    print("=" * 70)
    print("MISSING: DLLs imported but not in bundle / system / driver set")
    print("=" * 70)
    if not missing:
        print("(none)")
    else:
        by_cat: dict[str, list[str]] = defaultdict(list)
        for name in missing:
            by_cat[_category(name)].append(name)
        for cat in sorted(by_cat):
            print(f"\n[{cat}]")
            for name in sorted(by_cat[cat]):
                dependents = sorted({p.name for p in reverse[name]})
                shown = ", ".join(dependents[:4])
                more = "" if len(dependents) <= 4 else f" (+{len(dependents)-4} more)"
                print(f"  {name}  <- {shown}{more}")

    print()
    print("=" * 70)
    print("PRESENT-BUT-DUPLICATE check (same DLL in multiple locations)")
    print("=" * 70)
    dup_paths: dict[str, list[Path]] = defaultdict(list)
    for p in natives:
        if p.suffix.lower() == ".dll":
            dup_paths[_norm(p.name)].append(p)
    any_dup = False
    for name, paths in sorted(dup_paths.items()):
        if len(paths) > 1:
            any_dup = True
            print(f"\n  {name}:")
            for pp in paths:
                print(f"    {pp.relative_to(BUNDLE)}  "
                      f"({pp.stat().st_size:,} bytes)")
    if not any_dup:
        print("(none)")

    if parse_failures:
        print("FAIL: PE imports could not be parsed for:", file=sys.stderr)
        for relative in sorted(parse_failures, key=str.casefold):
            print(f"  {relative}", file=sys.stderr)

    return 0 if not missing and not parse_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
