"""End-to-end tests against the frozen yamabiko.exe bundle.

These run AGAINST THE BUILT BUNDLE under ``dist/yamabiko/``, not the
dev venv. The existing ``@pytest.mark.real`` suite exercises dev-venv
Python paths -- it can't see bugs that only manifest in the
PyInstaller windowed-exe environment, such as missing CUDA DLLs,
missing unidic_lite or cutlet files, or print-to-invalid-stdout
crashes.

Two tests, increasing in strength:

1. ``test_frozen_exe_starts_without_errors_for_45s`` (log-grep)

   Spawn ``yamabiko.exe --run-main``, let it run 45 s, assert the log
   contains the full startup chain and no tracebacks / OSErrors /
   ``ERROR pipeline stage`` lines. Catches init-time + early-startup
   failures. The cuBLAS, unidic_lite, and cutlet missing-dependency
   crashes all happen before ``overlay server starting`` prints and
   would be caught here.

2. ``test_frozen_exe_translates_sample_wav_and_broadcasts_subtitle``
   (true end-to-end)

   Spawn exe, wait for the WebSocket server to come up, connect a
   test WS client, play ``sample_jp.wav`` through the default output
   device so WASAPI loopback captures it, and assert a subtitle
   frame arrives on the WS with non-empty JP + EN. Catches anything
   between capture and broadcast.

   NOTE: the ``print(flush=True)`` OSError under a windowed exe's
   invalid stdout is NOT catchable by this test -- its spawn redirects
   stdout to subprocess.DEVNULL, handing the whole tree a VALID NUL
   handle, so the print succeeds even with the _SafeTextIO guard
   reverted. That class lives with
   ``test_frozen_exe_start_menu_launch_translates_and_log_stays_clean``
   below, which spawns with NO inherited std handles.

Both tests:
  * Skip if no bundle at ``dist/yamabiko/``
  * Skip if ``llama-server`` is not reachable at localhost:8090
    (these run ``--run-main`` -- the pipeline alone, which needs an
    EXTERNAL llama for its startup sanity check). The default-launch
    / lifecycle tests below use the INVERTED contract: the watchdog
    under test spawns its OWN llama-server on 8090, so they skip when
    the port is already taken.
  * Kill the entire process tree on teardown (the frozen exe spawns
    a pipeline child via ``--run-main``; killing just the watchdog
    leaves a zombie child holding the audio device + port)

Scope intentionally narrow: these aren't substitutes for the manual
wizard E2E step in ``docs/RELEASE.md``. That one covers installer,
wizard pages, autostart, and uninstall -- things that require human
click-through. These cover the pipeline-behavior gap between dev
venv and frozen bundle.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from ctypes import wintypes

import psutil
import pytest

from yamabiko.lifecycle.launch_markers import WIZARD_START_MARKER

pytestmark = [pytest.mark.real, pytest.mark.frozen_artifact]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _REPO_ROOT / "dist" / "yamabiko"
_EXE = _BUNDLE / "yamabiko.exe"
_WS_URL = "ws://127.0.0.1:8765/ws"
_LLAMA_URL = "http://127.0.0.1:8090/v1/models"
_SAMPLE_WAV = _REPO_ROOT / "sample_jp.wav"


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _skip_if_no_bundle() -> None:
    if not _EXE.is_file():
        pytest.skip(
            f"no built bundle at {_EXE}. "
            f"Run installer\\rebuild-fast.ps1 or installer\\build.ps1 first."
        )


def _skip_if_no_llama() -> None:
    import httpx
    try:
        r = httpx.get(_LLAMA_URL, timeout=2.0)
    except Exception as exc:
        pytest.skip(
            f"llama-server not reachable at {_LLAMA_URL}: {exc!r}. "
            f"Start it via scripts\\start-llama-server.ps1 first."
        )
    if r.status_code >= 400:
        pytest.skip(f"llama-server returned HTTP {r.status_code}")


def _localhost_port_is_bound(port: int) -> bool:
    """Return whether a TCP listener accepts connections on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _find_unused_loopback_port() -> int:
    """Ask Windows for an unused loopback port, then release it for launch."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _skip_if_overlay_port_busy() -> None:
    """If 127.0.0.1:8765 is already bound, another yamabiko.exe is running.

    The test would otherwise spawn a second instance that crashes on
    ``OSError winerror 10048`` at uvicorn bind, giving a misleading
    test failure. Skip cleanly with a direct instruction instead.
    """
    if _localhost_port_is_bound(8765):
        pytest.skip(
            "overlay WebSocket port 8765 is already in use — another "
            "yamabiko.exe is running. Exit it from the tray (Quit) or "
            "`taskkill /F /IM yamabiko.exe` and re-run this test."
        )


def _skip_if_llama_port_busy() -> None:
    """Default-launch / lifecycle precondition.

    The no-args watchdog under test spawns its OWN llama-server with an
    unconditional ``--port 8090`` and no pre-bind probe. If something
    already holds 8090 (e.g. scripts/start-llama-server.ps1 running for
    the --run-main tests), the spawned llama crash-loops on bind
    failure and every lifecycle invariant is measured against a
    contaminated, never-shipped topology.
    """
    if _localhost_port_is_bound(8090):
        pytest.skip(
            "port 8090 is already in use — the default-launch watchdog "
            "spawns its OWN llama-server there and would crash-loop on "
            "bind failure. Stop the external llama-server (it is only "
            "needed for the --run-main tests) and re-run."
        )


def _skip_if_no_translator_gguf() -> None:
    """Default-launch / lifecycle precondition.

    The watchdog's llama supervisor polls the HF cache for the translator
    GGUF before spawning (yamabiko.watchdog._make_real_spawn_llama);
    if it isn't cached, the spawn blocks forever waiting for the wizard
    and the three-process tree never forms. Gate on the production
    resolver so this skip can't drift from the runtime's own lookup.
    """
    from yamabiko.watchdog import _resolve_translator_model_path
    path = _resolve_translator_model_path()
    if path is None or not path.is_file():
        pytest.skip(
            "translator GGUF not in the HF cache — the watchdog's own "
            "llama-server can't spawn. Run the first-run wizard or "
            "preflight to download it."
        )


# ---------------------------------------------------------------------------
# Process / log helpers
# ---------------------------------------------------------------------------

def _logs_dir() -> Path:
    # Runtime logs live at %LOCALAPPDATA%\Yamabiko\logs\ as per-process
    # files: main-watchdog.log / main-pipeline.log / main-overlay.log
    # (one writer per file so Windows log rotation is safe).
    local_app = os.environ.get("LOCALAPPDATA")
    if not local_app:
        pytest.skip("LOCALAPPDATA unset; can't locate the logs dir")
    return Path(local_app) / "Yamabiko" / "logs"


class _LogTail:
    """Merged tail over every per-process main-*.log.

    Snapshots per-file sizes at construction; ``read_new()`` returns
    everything appended since, across all role files — including files
    born after the snapshot (offset 0). E2E assertions (positive
    sentinels AND Traceback scans) must see watchdog + pipeline +
    overlay output, which live in three separate role log files.
    """

    def __init__(self, logs_dir: Path) -> None:
        self._dir = logs_dir
        self._offsets: dict = (
            {p: p.stat().st_size for p in logs_dir.glob("main-*.log")}
            if logs_dir.is_dir() else {}
        )

    def read_new(self) -> str:
        if not self._dir.is_dir():
            return ""
        parts = []
        for p in sorted(self._dir.glob("main-*.log")):
            off = self._offsets.get(p, 0)
            try:
                # Rotation guard: if the file rotated since the snapshot,
                # its size is now BELOW the stored offset and a bare seek
                # would land past EOF, silently reading "" for the rest of
                # the test. Restart from 0 — over-reading beats a silent
                # empty tail during a failure investigation.
                if p.stat().st_size < off:
                    off = 0
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(off)
                    chunk = f.read()
            except OSError:
                continue
            if chunk:
                parts.append(f"--- {p.name} ---\n{chunk}")
        return "\n".join(parts)


def _spawn_exe(*extra_args: str) -> subprocess.Popen:
    """Launch yamabiko.exe --run-main (plus ``extra_args``) with the child
    process group isolated.

    CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT and also lets
    ``taskkill /T`` cleanly kill the whole tree on teardown.
    """
    return subprocess.Popen(
        [str(_EXE), "--run-main", *extra_args],
        cwd=str(_BUNDLE),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def _kill_tree(pid: int) -> None:
    """Kill a Windows process and all its descendants.

    The frozen exe spawns a pipeline child; plain proc.terminate() leaves
    it running. `taskkill /F /T /PID <pid>` walks the tree.
    """
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True, timeout=10,
    )


def _wait_for_log_marker(
    tail_reader: "_LogTail", marker: str, timeout_s: float
) -> str:
    """Block until ``marker`` appears in any role log, or timeout."""
    tail = ""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        tail = tail_reader.read_new()
        if marker in tail:
            return tail
        time.sleep(0.5)
    raise TimeoutError(
        f"log marker {marker!r} not seen within {timeout_s:.0f}s. "
        f"Last log tail: {tail[-400:]!r}"
    )


# ---------------------------------------------------------------------------
# Helpers: psutil-based process-tree introspection.
#
# The lifecycle E2E tests below need to (a) wait for the watchdog to
# have spawned all three expected children (pipeline + overlay +
# llama-server), (b) classify each PID by role so tests can hard-kill
# a specific one, and (c) assert PID liveness snapshot-style.
#
# These are deliberately separate from ``_kill_tree`` above (which uses
# ``taskkill /T`` and is a blunt teardown tool) because the lifecycle
# tests need targeted, surgical kills at particular process-tree nodes
# to exercise the nested Job Object + named-event design.
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Return True iff a process with ``pid`` is running (and not a zombie).

    psutil.STATUS_ZOMBIE won't normally surface on Windows but we guard
    for it defensively so the helper is portable and explicit about the
    "still a real, running process" contract.
    """
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_for_three_processes(
    watchdog_proc: subprocess.Popen, timeout: float
) -> list[int]:
    """Return the PIDs of the watchdog's three supervised children.

    Polls ``psutil.Process(watchdog_proc.pid).children(recursive=True)``
    until all three expected roles (pipeline + overlay + llama) are
    classifiable from the descendant tree, then returns their PIDs.

    The "classify 3 roles" check (not "count 3 descendants") avoids a
    flake where psutil reports transient non-yamabiko descendants too
    -- conhost.exe, CREATE_NEW_PROCESS_GROUP helpers, and PyInstaller
    bootloaders can all appear briefly. Counting alone caused the helper
    to return early with ``[pipeline, llama, conhost]`` before the
    overlay process was alive, making ``_classify_pids`` miss the overlay
    role. Waiting for ALL three expected roles is the behaviour every
    caller actually wants.

    Three is load-bearing: the watchdog supervises pipeline + overlay +
    llama-server. If this helper times out, the watchdog composition is
    missing a supervisor.
    """
    deadline = time.monotonic() + timeout
    last_pids: list[int] = []
    last_roles: dict[str, int] = {}
    while time.monotonic() < deadline:
        try:
            p = psutil.Process(watchdog_proc.pid)
            children = p.children(recursive=True)
            pids = [c.pid for c in children]
            roles = _classify_pids(pids)
            last_pids = pids
            last_roles = roles
            if {"pipeline", "overlay", "llama"}.issubset(roles.keys()):
                return pids
        except psutil.NoSuchProcess:
            break
        time.sleep(0.5)
    raise TimeoutError(
        f"did not see all 3 roles (pipeline + overlay + llama) as "
        f"descendants of watchdog pid {watchdog_proc.pid} within "
        f"{timeout:.0f}s. Last descendants seen: {last_pids}. "
        f"Classified roles so far: {last_roles}. Missing: "
        f"{sorted({'pipeline', 'overlay', 'llama'} - set(last_roles))}."
    )


def _classify_pids(pids: list[int]) -> dict[str, int]:
    """Map each supervised-child PID to its role ("pipeline" | "overlay" | "llama").

    Classification heuristic (order matters):
      * process name contains "llama-server" -> "llama"
      * cmdline contains "--run-overlay"     -> "overlay"
      * cmdline contains "--run-main"        -> "pipeline"

    Frozen-bundle cmdline pattern is ``yamabiko.exe --run-overlay`` /
    ``yamabiko.exe --run-main`` (see ``yamabiko.watchdog._build_overlay_cmd``
    + ``_build_main_cmd``). Any PID that doesn't classify is silently
    dropped; callers should assert on the roles they care about.
    """
    roles: dict[str, int] = {}
    for pid in pids:
        try:
            p = psutil.Process(pid)
            name = (p.name() or "").lower()
            try:
                cmdline = " ".join(p.cmdline()).lower()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                cmdline = ""
            if "llama-server" in name:
                roles["llama"] = pid
            elif "--run-overlay" in cmdline:
                roles["overlay"] = pid
            elif "--run-main" in cmdline:
                roles["pipeline"] = pid
        except psutil.NoSuchProcess:
            pass
    return roles


def _kill_pid(pid: int) -> None:
    """Force-kill a single PID via psutil. No-op if already gone."""
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        pass


def _visible_windows_of_pid(pid: int) -> list[int]:
    """Return the visible top-level window handles owned by ``pid``.

    Scope caveat -- this only sees CLASSIC conhost console windows.
    user32 attributes a legacy conhost window to the attached console
    process, so a console child that popped an old-style terminal shows
    up here under its own pid. But on machines whose default terminal
    is Windows Terminal (``HKCU\\Console\\%%Startup`` delegation not
    pinned to the conhost CLSID ``{B23D10C0-...}`` -- the Windows 11
    22H2+ default), the defterm handoff makes WindowsTerminal.exe own
    the visible window and this helper returns [] for the child even
    though a terminal popped. Never use it alone as a console-window
    regression guard; pair it with ``_probe_console_window``, which
    asks the child's own console and is host-independent.
    """
    import ctypes
    import ctypes.wintypes

    hwnds: list[int] = []
    user32 = ctypes.windll.user32

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _l):
        owner = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            hwnds.append(hwnd)
        return True

    user32.EnumWindows(_cb, None)
    return hwnds


# Source of the console-window probe. Runs as a throwaway subprocess so
# its FreeConsole/AttachConsole calls cannot disturb pytest's console.
_CONSOLE_PROBE_SRC = '''\
"""Console-window probe: report the window state of <pid>'s console."""
import ctypes
import json
import sys


def main() -> int:
    pid = int(sys.argv[1])
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32.AttachConsole.argtypes = [ctypes.c_uint32]
    kernel32.AttachConsole.restype = ctypes.c_int
    kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    user32.GetClassNameW.argtypes = [
        ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int
    ]
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user32.IsWindowVisible.restype = ctypes.c_int

    report = {"attached": False, "attach_error": 0,
              "hwnd": 0, "cls": "", "visible": False}
    kernel32.FreeConsole()
    if not kernel32.AttachConsole(pid):
        report["attach_error"] = ctypes.get_last_error()
    else:
        report["attached"] = True
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            report["hwnd"] = int(hwnd)
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buf, 256)
            report["cls"] = buf.value
            report["visible"] = bool(user32.IsWindowVisible(hwnd))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _probe_console_window(pid: int) -> dict:
    """Ask ``pid``'s OWN console for its window. Host-independent.

    AttachConsole(pid) + GetConsoleWindow() from a helper subprocess,
    reporting ``{"attached", "attach_error", "hwnd", "cls", "visible"}``.
    Unlike ``_visible_windows_of_pid`` this does not depend on which
    process user32 attributes a window to, so it stays sensitive on
    machines whose default terminal is Windows Terminal. The four
    console-host states (first three validated empirically against real
    OS controls, 2026-07-10):

    * headless classic conhost (CREATE_NO_WINDOW -- the fixed
      behavior, which per the defterm spec BYPASSES the terminal
      handoff on every machine): hwnd == 0, or an INVISIBLE
      "ConsoleWindowClass" window on older builds.
    * visible classic conhost (defect, conhost-pinned machine):
      cls == "ConsoleWindowClass", visible == True.
    * terminal-delegated / ConPTY (defect, Windows-Terminal-default
      machine): cls == "PseudoConsoleWindow" -- the ConPTY fake window
      (class registered in microsoft/terminal
      src/interactivity/base/InteractivityFactory.cpp); the VISIBLE
      window is WindowsTerminal.exe's, invisible to pid attribution.
    * no console at all (e.g. a DETACHED_PROCESS child): attached is
      False with attach_error == 6 (ERROR_INVALID_HANDLE).
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ymb-console-probe-") as td:
        script = Path(td) / "probe_console_window.py"
        script.write_text(_CONSOLE_PROBE_SRC, encoding="utf-8")
        out = subprocess.run(
            [sys.executable, str(script), str(pid)],
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    assert out.returncode == 0 and out.stdout.strip(), (
        f"console-window probe helper failed (rc={out.returncode}): "
        f"stdout={out.stdout!r} stderr={out.stderr!r}"
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _local_appdata() -> Path:
    """Return %LOCALAPPDATA% as a Path. Assumes Windows (this whole file does)."""
    return Path(os.environ["LOCALAPPDATA"])


# ---------------------------------------------------------------------------
# Test 1: log-grep over 45 seconds
# ---------------------------------------------------------------------------

@pytest.mark.frozen_external_llama
def test_frozen_exe_starts_without_errors_for_45s(tmp_path) -> None:
    """Full startup chain appears + no tracebacks in the first 45s.

    This is the cheap sentinel -- catches crashes at module load, HF
    model-fetch, CUDA init, llama-server handshake, etc. Anything that
    crashes before ``overlay server starting`` has left a trail by the
    45s mark.

    Runs with ``--config`` setting a free WebSocket port, so a busy 8765
    (another app) neither skips the test nor collides with it.

    Expected startup chain markers (ordered, all must appear):
      * yamabiko startup
      * ASR loaded; warming up
      * romanization pipeline enabled
      * translator sanity check
      * capture started
      * overlay server starting on http://127.0.0.1:<the free port>/

    Forbidden markers (none may appear):
      * "Traceback"
      * "OSError"
      * "ERROR pipeline stage"
    """
    _skip_if_no_bundle()
    _skip_if_no_llama()

    port = _find_unused_loopback_port()
    config_path = tmp_path / "free-port.yaml"
    config_path.write_text(f"overlay:\n  websocket_port: {port}\n", encoding="utf-8")
    tail_reader = _LogTail(_logs_dir())

    proc = _spawn_exe("--config", str(config_path))
    try:
        # Give the pipeline 45s to either finish startup or fall over.
        # Startup on a warm cache runs in ~10-12s per the observed
        # log timings; 45s is a 4x margin.
        time.sleep(45)
    finally:
        _kill_tree(proc.pid)
        # Give the OS a moment to finish the tree kill before releasing.
        time.sleep(1)

    tail = tail_reader.read_new()

    required = [
        "yamabiko startup",
        "ASR loaded; warming up",
        "romanization pipeline enabled",
        "translator sanity check",
        "capture started",
        f"overlay server starting on http://127.0.0.1:{port}/",
    ]
    missing = [m for m in required if m not in tail]
    assert not missing, (
        f"startup chain incomplete after 45s. Missing markers: {missing}. "
        f"Log tail (last 2k):\n{tail[-2000:]}"
    )

    forbidden = ["Traceback", "OSError", "ERROR pipeline stage"]
    hits = [f for f in forbidden if f in tail]
    assert not hits, (
        f"forbidden marker(s) {hits} appeared in log during startup. "
        f"Log tail (last 2k):\n{tail[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Test 2: end-to-end via real audio + WebSocket
# ---------------------------------------------------------------------------

def _with_session_token(url: str) -> str:
    """The pipeline refuses every token-free /ws client. Read this launch's
    session token from the same server's GET /ws-token (served while
    overlay.browser is on, the shipped default) and append it as ?t=."""
    from urllib.parse import urlsplit

    from yamabiko import loopback_http

    host = urlsplit(url).netloc
    r = loopback_http.get(f"http://{host}/ws-token", timeout=5.0)
    r.raise_for_status()
    return f"{url}?t={r.json()['token']}"


async def _await_first_subtitle(url: str, timeout_s: float) -> dict:
    """Connect to the overlay WS and return the first {"type":"subtitle",...}."""
    # websockets is a uvicorn transitive dep; already in the dev venv.
    from websockets.asyncio.client import connect as ws_connect

    async with ws_connect(_with_session_token(url)) as ws:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"no subtitle frame received within {timeout_s:.0f}s")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("type") == "subtitle":
                return data


async def _await_style_snapshot(url: str, timeout_s: float) -> dict:
    """Connect to the overlay WS and return its initial style snapshot."""
    from websockets.asyncio.client import connect as ws_connect

    async with ws_connect(_with_session_token(url)) as ws:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"no style frame received within {timeout_s:.0f}s"
                )
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("type") == "style":
                return data


def _assert_looks_translated(subtitle: dict) -> None:
    """Assert a captured overlay subtitle is a real JP->EN translation -- not a
    placeholder, a verbatim echo of the source, or a leaked thinking-mode
    block. Exact wording is ASR+model-dependent, so only robust invariants are
    checked. Shared by both translate E2Es so the quality bar runs on whichever
    path executes (the external-llama WAV test or the default-launch
    start-menu test).

    Seeded-defect drills:
      * postprocess stops stripping "<think>" -> a think block reaches EN -> red;
      * translator echoes the JP prompt verbatim -> kana survives in EN -> red;
      * EN coerced to "" / whitespace -> the non-empty / Latin-letter check red.
    """
    assert subtitle.get("type") == "subtitle", subtitle
    jp = (subtitle.get("jp") or "").strip()
    en = (subtitle.get("en") or "").strip()
    assert jp, f"empty JP field: {subtitle}"
    assert en, f"empty EN field: {subtitle}"
    # No thinking-mode block leaked through postprocess (translator.py raises on
    # "<think>", but confirm it end-to-end in the frozen bundle).
    assert "<think>" not in en, f"thinking-mode block leaked into EN: {en!r}"
    # EN actually produced English text, not an empty-but-truthy placeholder.
    assert any(c.isascii() and c.isalpha() for c in en), (
        f"EN subtitle has no Latin letters -- the pipeline produced no English "
        f"text: {en!r}"
    )
    # No kana in EN: hiragana/katakana never legitimately survive a JP->EN
    # translation, so kana means the translator echoed the source instead of
    # translating it. (Kanji is deliberately NOT asserted against -- a rare
    # kept proper-noun could carry one; kana is the clean "untranslated" signal.)
    assert not any("぀" <= c <= "ヿ" for c in en), (
        f"EN subtitle still contains kana -- the translator echoed Japanese "
        f"instead of translating it: {en!r}"
    )


# The ASR stage starts right after the "overlay server starting" line and logs
# its step counters ("asr steps: steps_run=N ...") one minute later
# (yamabiko.main._STEP_COUNTS_LOG_S), and only when steps ran or were dropped.
_STEP_COUNTS_RE = re.compile(r"asr steps: steps_run=(\d+)")
_STEP_COUNTS_WAIT_S = 75.0


@pytest.mark.frozen_external_llama
@pytest.mark.parametrize(
    "faster_subtitles", [True, False],
    ids=["faster-subtitles-on", "faster-subtitles-off"],
)
def test_frozen_exe_translates_sample_wav_and_broadcasts_subtitle(
    tmp_path, faster_subtitles,
) -> None:
    """Play a real JP WAV and verify a subtitle arrives on the overlay WS.

    End-to-end: audio out -> WASAPI loopback -> Silero chunker -> ASR ->
    fast-bunkai -> translator (real llama-server) ->
    broadcast. A print() that raises between the translator response and
    the broadcast would prevent any subtitle from reaching this test.

    Runs once per Faster subtitles state (``asr.incremental.enabled``, set
    through ``--config``). The startup line ``faster subtitles: enabled=...``
    proves which state the frozen pipeline ran, and the ASR stage's step
    counters prove the state took effect: on, step decodes ran; off, none did.
    The same file sets a free WebSocket port, so a busy 8765 (another app)
    neither skips the test nor collides with it.

    Requires the test machine to actually play audio -- the default
    output device must be a real one WASAPI loopback can tap. Skips if
    audio playback isn't available or the sample WAV is missing.
    """
    _skip_if_no_bundle()
    _skip_if_no_llama()
    if not _SAMPLE_WAV.is_file():
        pytest.skip(f"sample JP WAV missing at {_SAMPLE_WAV}")
    try:
        import winsound  # stdlib, Windows-only (which we are)
    except ImportError:
        pytest.skip("winsound unavailable (non-Windows host)")

    tail_reader = _LogTail(_logs_dir())

    state = str(faster_subtitles).lower()
    port = _find_unused_loopback_port()
    config_path = tmp_path / "faster-subtitles.yaml"
    config_path.write_text(
        f"overlay:\n  websocket_port: {port}\n"
        f"asr:\n  incremental:\n    enabled: {state}\n",
        encoding="utf-8",
    )
    ws_url = f"ws://127.0.0.1:{port}/ws"
    proc = _spawn_exe("--config", str(config_path))
    try:
        # Wait for the WS server to bind. "overlay server starting on
        # http://127.0.0.1" is the last startup marker; once it's in
        # the log, the next uvicorn.serve() call in the asyncio loop
        # accepts connections within a few tens of ms.
        _wait_for_log_marker(
            tail_reader,
            f"overlay server starting on http://127.0.0.1:{port}/",
            timeout_s=60,
        )
        # The ASR stage started with that line; its step counters count from here.
        t_stage = time.monotonic()
        # Small grace period so uvicorn's accept loop is actually live.
        time.sleep(1.0)

        # Play the WAV async so this thread keeps running. Loopback on
        # the default output device is what the chunker is listening on.
        winsound.PlaySound(
            str(_SAMPLE_WAV), winsound.SND_FILENAME | winsound.SND_ASYNC
        )
        try:
            # Budget: WAV is ~3s; Silero min_silence=400ms adds a pause;
            # ASR warmup is already done; translator + broadcast add <2s.
            # Give it 45s to cover cold-path variance on slow machines.
            subtitle = asyncio.run(_await_first_subtitle(ws_url, timeout_s=45))
        finally:
            # Stop anything still playing so we don't leave the test
            # machine making noise after teardown.
            winsound.PlaySound(None, 0)

        # Minimum invariants for a real translation result. The exact
        # text is ASR- and model-dependent; we just want to know the
        # full pipeline delivered something plausible.
        # Full JP->EN round trip + translation-quality invariants (helper).
        _assert_looks_translated(subtitle)

        # On, the WAV's speech window ran step decodes, so the counters line
        # comes; off, no step ever runs, so it never does.
        deadline = t_stage + _STEP_COUNTS_WAIT_S
        while True:
            tail = tail_reader.read_new()
            steps_run = [int(n) for n in _STEP_COUNTS_RE.findall(tail)]
            if steps_run or time.monotonic() >= deadline:
                break
            time.sleep(1.0)
        assert f"faster subtitles: enabled={state}" in tail, (
            f"the pipeline did not start with Faster subtitles {state}. "
            f"Tail:\n{tail[-2000:]}"
        )
        if faster_subtitles:
            assert steps_run and steps_run[-1] >= 1, (
                f"Faster subtitles is on but no step decode ran within "
                f"{_STEP_COUNTS_WAIT_S:.0f}s (steps_run={steps_run}). "
                f"Tail:\n{tail[-2000:]}"
            )
        else:
            assert not steps_run, (
                f"Faster subtitles is off but the ASR stage ran step decodes "
                f"(steps_run={steps_run}). Tail:\n{tail[-2000:]}"
            )

        # Defense in depth: even if a subtitle made it through, if a
        # print()-class bug fired on a LATER segment we'd see a
        # traceback in the log. Catch that too.
        bad = [m for m in ("Traceback", "OSError") if m in tail]
        assert not bad, (
            f"subtitle delivered but log shows {bad} during the run. "
            f"Tail:\n{tail[-2000:]}"
        )
    finally:
        _kill_tree(proc.pid)
        time.sleep(1)


# ---------------------------------------------------------------------------
# Default-launch E2E: exercises the top-level watchdog path
# (pipeline + overlay + llama-server spawned together).
#
# This is the test that catches the "bundled but never launched" class
# for both the overlay and llama-server — invisible to the existing
# "--run-main" tests because those bypass the watchdog's top-level
# dispatch.
# ---------------------------------------------------------------------------


def _spawn_exe_default_launch() -> subprocess.Popen:
    """Launch yamabiko.exe with NO args -- exercises the top-level
    watchdog path, which now triple-spawns pipeline + overlay +
    llama-server via three supervisor threads.
    """
    return subprocess.Popen(
        [str(_EXE)],  # NO --run-main; default (dual/triple-spawn) path
        cwd=str(_BUNDLE),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def _spawn_exe_start_menu_equivalent(
    env: "dict | None" = None,
) -> subprocess.Popen:
    """Default launch with NO inherited std handles.

    Every other spawn in this file passes stdin/stdout/stderr =
    subprocess.DEVNULL. That sets STARTF_USESTDHANDLES with a VALID NUL
    device handle, which the watchdog's own ``stdin=DEVNULL`` re-spawn
    then propagates to the pipeline child — so ``print(flush=True)``
    succeeds everywhere and the unguarded-stdout OSError class is
    structurally unreachable. A real Start-Menu / Explorer launch hands
    the windowed exe no usable std handles at all; the pipeline child
    then inherits an INVALID stdout via the watchdog's
    STARTF_USESTDHANDLES spawn, and any unguarded print raises
    OSError [Errno 22].

    No std arguments => CPython does not set STARTF_USESTDHANDLES, and
    DETACHED_PROCESS guarantees no console either way — the closest
    Popen gets to ShellExecute while still returning a handle for
    teardown.

    ``env``: when None the child inherits this process's environment.
    The fresh-state wizard test passes a copy with ``YAMABIKO_APPDATA``
    overridden so the spawned tree's state.json + role logs resolve under
    an isolated tmp dir; the watchdog forwards that env to its
    ``--run-main`` child via ``os.environ.copy()`` in ``_real_spawn``.
    ``env=`` REPLACES rather than merges, so callers must pass a full
    ``dict(os.environ)`` copy (not just the override) to keep PATH /
    HF_HOME / the CUDA-DLL search dirs the frozen exe needs.
    """
    return subprocess.Popen(
        [str(_EXE)],
        cwd=str(_BUNDLE),
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        ),
        env=env,
    )


@pytest.mark.frozen_owned_llama
def test_frozen_exe_default_launch_spawns_pipeline_and_overlay() -> None:
    """yamabiko.exe (no args) = Start Menu click. Must spawn BOTH a
    pipeline child AND an overlay child beyond the watchdog itself --
    AND the overlay must still be the same process some seconds later
    (i.e. not stuck in a crash/respawn loop).

    A PySide6 Signal bug in a past release made the overlay crash
    immediately on every launch; the watchdog supervisor dutifully
    respawned it, so a single child-count snapshot saw 2 yamabiko
    processes and called the install healthy. This test samples twice +
    asserts pid stability + asserts the log has no tracebacks during
    the test window, which is what's actually required to catch a
    crash-loop. It also asserts the spawned llama-server popped no
    terminal on the desktop, via two complementary detectors: no
    visible top-level window attributed to llama's pid (catches a
    classic conhost console) AND llama's own console is headless per
    ``_probe_console_window`` (catches the defterm/ConPTY case on
    Windows-Terminal-default machines, where the visible window belongs
    to WindowsTerminal.exe and pid attribution alone stays green).
    """
    _skip_if_no_bundle()
    # Default launch — the watchdog spawns its own llama-server on 8090.
    # An external llama (the --run-main tests' precondition) is a port
    # CONFLICT here, not a dependency.
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()

    tail_reader = _LogTail(_logs_dir())

    proc = _spawn_exe_default_launch()
    try:
        _wait_for_log_marker(
            tail_reader,
            "overlay server starting on http://127.0.0.1",
            timeout_s=60,
        )
        time.sleep(3)  # give overlay a beat to actually connect

        import psutil
        try:
            top = psutil.Process(proc.pid)
            first_children = top.children(recursive=True)
        except Exception:
            pytest.skip("psutil unavailable; cannot enumerate children")

        first_ymb = {c.pid: c.name() for c in first_children
                     if "yamabiko" in c.name().lower()}
        assert len(first_ymb) >= 2, (
            f"expected >= 2 yamabiko children (pipeline + overlay) "
            f"3s after launch; got {first_ymb}. Log tail:\n"
            f"{tail_reader.read_new()[-2000:]}"
        )

        # --- Crash-loop detection -----------------------------------------
        # The original version of this test stopped here. That's not
        # enough -- a crashing overlay + watchdog auto-respawn presents
        # as "yamabiko child exists" at any snapshot. Sample again 7 s
        # later and verify the SAME pids survived.
        time.sleep(7)
        try:
            second_children = top.children(recursive=True)
        except psutil.NoSuchProcess:
            pytest.fail(
                "top-level watchdog died during the 7 s survival window"
            )
        second_ymb = {c.pid: c.name() for c in second_children
                      if "yamabiko" in c.name().lower()}

        # Every pid that existed at t=3 must still exist at t=10.
        lost = set(first_ymb) - set(second_ymb)
        assert not lost, (
            f"yamabiko child(ren) died and were respawned between "
            f"t=3s and t=10s -- crash-loop signature. Lost pids: "
            f"{[(pid, first_ymb[pid]) for pid in lost]}. "
            f"Surviving: {second_ymb}. Log tail:\n"
            f"{tail_reader.read_new()[-3000:]}"
        )

        # --- No console window (terminal must not pop) ---------------------
        # yamabiko.exe is windowed (console=False) and llama-server.exe
        # is a console-subsystem binary: without CREATE_NO_WINDOW on the
        # watchdog's llama Popen, Windows gives the child a console the
        # user can see (live report 2026-07-07; dev mode masks it because
        # the parent already owns a console). WHAT the user sees depends
        # on the machine's default terminal:
        #   * conhost-pinned: a visible classic console window,
        #     attributed to llama's own pid (_visible_windows_of_pid
        #     catches it).
        #   * Windows Terminal default (Win11 22H2+ out of the box): the
        #     defterm handoff gives the visible window to
        #     WindowsTerminal.exe, so pid attribution sees NOTHING --
        #     but llama's own console window becomes the ConPTY fake
        #     "PseudoConsoleWindow", which the probe catches.
        # CREATE_NO_WINDOW bypasses the defterm handoff entirely (per
        # microsoft/terminal's default-terminal spec), so the fixed
        # behavior is a headless classic conhost on BOTH machine
        # classes. Seeded defect: drop
        # creationflags=subprocess.CREATE_NO_WINDOW from the llama Popen
        # in yamabiko/watchdog.py, rebuild, re-run -> on a conhost
        # machine the visible-window list is non-empty; on a
        # WT-default machine the probe reports "PseudoConsoleWindow".
        # Red either way.
        roles = _classify_pids([c.pid for c in second_children])
        assert "llama" in roles, (
            f"llama-server not found among watchdog descendants "
            f"{[c.pid for c in second_children]}; cannot assert on its "
            f"console window. Roles: {roles}. Log tail:\n"
            f"{tail_reader.read_new()[-2000:]}"
        )
        visible = _visible_windows_of_pid(roles["llama"])
        assert visible == [], (
            f"llama-server (pid {roles['llama']}) owns visible top-level "
            f"window(s) {visible} -- the frozen app popped a terminal "
            f"window on launch. The watchdog's llama Popen must pass "
            f"creationflags=subprocess.CREATE_NO_WINDOW."
        )
        probe = _probe_console_window(roles["llama"])
        if not probe["attached"]:
            # ERROR_INVALID_HANDLE (6) == the target has no console at
            # all, hence no console window: acceptable. Any other attach
            # failure means the probe itself broke -- fail loud instead
            # of letting the guard go evergreen.
            assert probe["attach_error"] == 6, (
                f"console probe could not attach to llama-server "
                f"(pid {roles['llama']}, WinError "
                f"{probe['attach_error']}); the no-console-window claim "
                f"is unverified. Probe report: {probe}"
            )
        else:
            assert probe["cls"] != "PseudoConsoleWindow", (
                f"llama-server (pid {roles['llama']}) is hosted in a "
                f"ConPTY session -- on this machine the default-terminal "
                f"handoff delegated its console to a terminal (Windows "
                f"Terminal popped a tab/window the pid-attribution check "
                f"cannot see). The watchdog's llama Popen must pass "
                f"creationflags=subprocess.CREATE_NO_WINDOW, which "
                f"bypasses the handoff. Probe report: {probe}"
            )
            assert not probe["visible"], (
                f"llama-server (pid {roles['llama']}) has a VISIBLE "
                f"console window (class {probe['cls']!r}) -- the frozen "
                f"app popped a terminal on launch. The watchdog's llama "
                f"Popen must pass "
                f"creationflags=subprocess.CREATE_NO_WINDOW. "
                f"Probe report: {probe}"
            )
            assert probe["cls"] in ("", "ConsoleWindowClass"), (
                f"llama-server's console window has unrecognized class "
                f"{probe['cls']!r}; the console-host taxonomy this guard "
                f"relies on has drifted (new Windows console host?). "
                f"Re-derive the classification -- do not weaken the "
                f"assertion. Probe report: {probe}"
            )

        # --- Log-content assertion ----------------------------------------
        # If anything threw a Python exception on startup or during the
        # 10 s test window, we want to know even if the supervisor kept
        # children alive. AttributeError (the real bug that led to this
        # test) would be hidden by len(children) >= 2 but SHOULD surface
        # as a Traceback in the role logs (main-*.log).
        tail = tail_reader.read_new()
        bad_markers = ("Traceback", "AttributeError", "ImportError",
                       "ModuleNotFoundError")
        hits = [m for m in bad_markers if m in tail]
        assert not hits, (
            f"log contains uncaught exceptions {hits} during default-launch "
            f"window. Tail (last 3 KB):\n{tail[-3000:]}"
        )

        # --- Positive sentinels -------------------------------------------
        # "Nothing is crashing" is weaker than "the thing that should
        # happen did happen." pipeline.ready fires once main.py has
        # wired all stages and is entering the event loop; it signals
        # pipeline composition succeeded end-to-end.
        # overlay.mainwindow.shown fires once the overlay window is
        # visible; it signals overlay_app.__main__.main() reached
        # show() without raising. Absence of overlay.mainwindow.shown
        # while pipeline.ready is present is the crash-loop signature
        # (overlay crashed between QApplication init and show; supervisor
        # respawned it so child-count looked healthy).
        # These are qualitatively different assertions from the
        # PID-stability + traceback checks above.
        assert tail.count("pipeline.ready") >= 1, (
            f"pipeline.ready sentinel missing from log -- pipeline "
            f"main() did not reach event loop. Tail (last 3 KB):\n"
            f"{tail[-3000:]}"
        )
        assert tail.count("overlay.mainwindow.shown") >= 1, (
            f"overlay.mainwindow.shown sentinel missing from log -- "
            f"overlay_app main() did not reach window.show() "
            f"(overlay crashed between QApplication init and show). "
            f"Tail (last 3 KB):\n{tail[-3000:]}"
        )
        # The frozen overlay resolves the fonts through bundle_data_dir();
        # a wrong frozen path registers nothing and the UI silently falls
        # back to system faces.
        fonts_line = next(
            (ln for ln in tail.splitlines() if "overlay.fonts.registered" in ln), ""
        )
        families = {
            f.strip() for f in fonts_line.partition("families=")[2].split(",")
        }
        assert {"BIZ UDPGothic", "BIZ UDPMincho", "Klee One"} <= families, (
            f"overlay.fonts.registered sentinel missing or incomplete -- the "
            f"frozen overlay did not register the bundled fonts from "
            f"_internal/yamabiko/display/static/fonts. Line: {fonts_line!r}. "
            f"Tail (last 3 KB):\n{tail[-3000:]}"
        )
    finally:
        _kill_tree(proc.pid)
        time.sleep(2)


@pytest.mark.frozen_owned_llama
def test_frozen_explicit_config_reaches_pipeline_and_overlay(tmp_path) -> None:
    """A frozen default launch keeps every role on one explicit config.

    The configured WebSocket port and style are observable from the live
    pipeline snapshot; the overlay role emits a sanitized applied-config
    marker for its theme and anchored position. Pipeline and overlay PIDs must
    remain stable across a five-second sample so a crash/respawn loop cannot
    masquerade as propagation success.
    """
    _skip_if_no_bundle()
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()

    port = _find_unused_loopback_port()
    config_path = tmp_path / "frozen-explicit-config.yaml"
    config_path.write_text(
        "overlay:\n"
        f"  websocket_port: {port}\n"
        "  theme: dark\n"
        "  style:\n"
        "    jp_size: 31\n"
        "  position:\n"
        "    screen: primary\n"
        "    v_offset_pct: 73.0\n",
        encoding="utf-8",
    )
    ws_url = f"ws://127.0.0.1:{port}/ws"
    tail_reader = _LogTail(_logs_dir())

    proc = subprocess.Popen(
        [str(_EXE), "--config", str(config_path)],
        cwd=str(_BUNDLE),
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        ),
    )
    try:
        _wait_for_log_marker(tail_reader, "pipeline.ready", timeout_s=120)
        _wait_for_log_marker(
            tail_reader, "overlay.mainwindow.shown", timeout_s=60,
        )

        first_pids = _wait_for_three_processes(proc, timeout=60)
        first_roles = _classify_pids(first_pids)
        assert {"pipeline", "overlay"}.issubset(first_roles), first_roles

        style = asyncio.run(_await_style_snapshot(ws_url, timeout_s=15))
        assert style.get("jp_size") == 31, style

        time.sleep(5)
        try:
            second_pids = [
                child.pid
                for child in psutil.Process(proc.pid).children(recursive=True)
            ]
        except psutil.NoSuchProcess:
            pytest.fail("watchdog exited during the explicit-config sample")
        second_roles = _classify_pids(second_pids)
        assert second_roles.get("pipeline") == first_roles["pipeline"], (
            f"pipeline restarted during the five-second sample: "
            f"{first_roles} -> {second_roles}"
        )
        assert second_roles.get("overlay") == first_roles["overlay"], (
            f"overlay restarted during the five-second sample: "
            f"{first_roles} -> {second_roles}"
        )

        tail = tail_reader.read_new()
        applied = (
            "overlay.config.applied theme=dark jp_size=31 "
            "screen_mode=primary v_offset_pct=73.0"
        )
        assert applied in tail, (
            "overlay did not report the explicit theme/position. "
            f"Tail (last 4 KB):\n{tail[-4000:]}"
        )
        bad = [
            marker
            for marker in (
                "Traceback",
                "AttributeError",
                "ImportError",
                "ModuleNotFoundError",
                "OSError",
                "ERROR pipeline stage",
            )
            if marker in tail
        ]
        assert not bad, (
            f"explicit-config frozen launch logged {bad}. "
            f"Tail (last 4 KB):\n{tail[-4000:]}"
        )
    finally:
        _kill_tree(proc.pid)
        time.sleep(2)


@pytest.mark.frozen_owned_llama
def test_frozen_default_launch_translates_wav_and_overlay_renders_subtitle() -> None:
    """Full shipped topology: yamabiko.exe (no args) spawns pipeline + overlay
    + its OWN llama-server; play sample_jp.wav; the SHIPPED Qt overlay child
    must actually RENDER the translated subtitle the pipeline broadcasts.

    The --run-main sample-WAV test above asserts only a test-owned WS client's
    frame -- it proves the subtitle frame LEFT the pipeline, not that the
    overlay rendered it, and it cannot: --run-main is pipeline-only, there is
    no overlay process in that tree. This closes that gap (H-11b) at the tier
    where the overlay actually exists.

    Detector / assertion split so audio-capture flakiness can NEVER masquerade
    as an overlay bug. A test-owned WS client is the oracle for "did the
    pipeline actually broadcast a subtitle" -- the SAME frame the overlay
    child receives. Two honest outcomes:

      * No frame arrives within the window -> the pipeline never produced a
        subtitle (the played WAV did not loopback-capture into a JA line on
        this host -- e.g. an audio.device override that isn't the winsound
        default output, or a muted output). That is the SAME audio dependency
        the --run-main sample-WAV test carries, and there is nothing to assert
        an overlay render against, so SKIP.
      * A frame DOES arrive -> the pipeline broadcast a real translated
        subtitle to every WS client, INCLUDING the overlay child (connected
        before the broadcast). The overlay's own add_subtitle sentinel
        (``overlay.subtitle.rendered`` in main-overlay.log) MUST now be
        present. Its absence -- a broadcast confirmed but the shipped renderer
        silent -- is exactly the H-11b bug, so FAIL.

    Requires a real audio output device (WASAPI loopback taps it), like the
    sample-WAV test above. Skips when the bundle / shisa GGUF / sample WAV are
    unavailable, or when 8090 / 8765 are already bound (default-launch spawns
    its OWN llama-server + overlay WS server there).

    Seeded defects that turn this red (once a subtitle broadcasts):
    - drop ``client.subtitleReceived.connect(window.add_subtitle)`` in
      build_overlay -> the overlay never renders -> the render marker never
      logs -> the 30 s marker wait fails;
    - remove the ``overlay.subtitle.rendered`` sentinel log line in
      add_subtitle -> same;
    - break the marker's en_ascii_letters stat (e.g. count over "" instead of
      en) -> the broadcast EN is real Latin text but the marker reports 0 ->
      the letters assert fails;
    - a regression that crash-loops the overlay -> add_subtitle never runs ->
      same.
    """
    _skip_if_no_bundle()
    # Default-launch contract: the watchdog spawns its own llama on 8090 and
    # its own overlay WS server on 8765, so BOTH ports must be free.
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()
    if not _SAMPLE_WAV.is_file():
        pytest.skip(f"sample JP WAV missing at {_SAMPLE_WAV}")
    try:
        import winsound  # stdlib, Windows-only (which we are)
    except ImportError:
        pytest.skip("winsound unavailable (non-Windows host)")

    tail_reader = _LogTail(_logs_dir())

    proc = _spawn_exe_default_launch()
    try:
        # The overlay must be up AND connected to the pipeline WS before the
        # broadcast, or it misses the frame (subtitles are not replayed).
        # pipeline.ready covers the watchdog's cold llama spawn + stage wiring.
        _wait_for_log_marker(tail_reader, "pipeline.ready", timeout_s=120)
        _wait_for_log_marker(tail_reader, "overlay.mainwindow.shown", timeout_s=60)
        time.sleep(3)  # let the overlay's QWebSocket finish connecting

        winsound.PlaySound(
            str(_SAMPLE_WAV), winsound.SND_FILENAME | winsound.SND_ASYNC
        )
        try:
            # Oracle: our own WS client sees the broadcast frame if (and only
            # if) the pipeline produced one. Raises TimeoutError otherwise.
            subtitle = asyncio.run(_await_first_subtitle(_WS_URL, timeout_s=60))
        except TimeoutError:
            pytest.skip(
                "pipeline broadcast no subtitle within 60 s -- the sample WAV "
                "did not loopback-capture into a JA subtitle on this host "
                "(audio.device override vs winsound default output, or muted "
                "output). Same audio dependency as the --run-main sample-WAV "
                "test; with no broadcast there is nothing to assert an overlay "
                "render against."
            )
        finally:
            winsound.PlaySound(None, 0)

        # A subtitle DID broadcast to every client (ours + the overlay), and it
        # is a real JP->EN translation (not an echo / empty / <think> leak).
        _assert_looks_translated(subtitle)

        # The overlay child received the SAME frame, so its add_subtitle render
        # sentinel must now be in main-overlay.log. 30 s is generous -- the
        # overlay renders within a frame of receiving. Absence here, with the
        # broadcast confirmed above, is the H-11b renderer gap.
        tail = _wait_for_log_marker(
            tail_reader, "overlay.subtitle.rendered", timeout_s=30,
        )
        rendered = [
            ln for ln in tail.splitlines()
            if "overlay.subtitle.rendered en_len=" in ln
        ]
        assert rendered, (
            f"render marker vanished from the tail after _wait_for_log_marker "
            f"returned it. Tail (last 2 KB):\n{tail[-2000:]}"
        )
        # The marker is CONTENT-FREE on a default install (transcript consent
        # ships off; default logs carry no dialogue), so the render proof
        # rides on the stats the overlay computed from the EN it rendered:
        # en_ascii_letters counts the Latin letters, and > 0 proves a real
        # (non-empty, Latin-lettered) English line hit the screen -- the same
        # power as a letter check on the raw text, without transcript content
        # in main-overlay.log. These exact filter/split tokens are pinned by
        # tests/overlay_app/test_overlay_render_marker.py::
        # test_render_marker_pins_the_frozen_e2e_filter_and_split_tokens, so
        # production drift turns the fast tier red before this bundle run.
        latin_letters = int(
            rendered[-1].split("en_ascii_letters=", 1)[1].split()[0]
        )
        assert latin_letters > 0, (
            f"the overlay rendered a subtitle with no English text (empty / "
            f"non-Latin EN): {rendered[-1]!r}; the shipped renderer put an "
            f"empty frame on screen"
        )

        # Defense in depth: no uncaught exception in any role log during the run
        # (main-overlay.log included -- a renderer that logged the marker then
        # threw would still be a bug).
        full = tail_reader.read_new()
        bad = [m for m in ("Traceback", "AttributeError", "ImportError",
                           "ModuleNotFoundError") if m in full]
        assert not bad, (
            f"overlay rendered but the role logs show {bad} during the run. "
            f"Tail (last 3 KB):\n{full[-3000:]}"
        )
    finally:
        _kill_tree(proc.pid)
        time.sleep(2)


# ---------------------------------------------------------------------------
# Lifecycle E2E suite.
#
# These three tests are the ground-truth guards for the process-lifecycle
# design. They spawn the frozen yamabiko.exe via its default entry (no
# args; Start-Menu-click equivalent) and exercise the three invariants
# the hybrid design must hold:
#
#  1. Tier A graceful shutdown: user-initiated stop propagates
#     watchdog -> pipeline-named-event -> overlay -> llama, finally
#     blocks run, nothing leaks.
#  2. Outer Job Object hard-kill cascade: if the watchdog itself is
#     TerminateProcess'd, the kernel must KILL_ON_JOB_CLOSE every
#     descendant (no orphans).
#  3. Inner Job Object nest: killing pipeline must take overlay with
#     it (inner job close) but MUST NOT take llama-server (llama is
#     a sibling of pipeline under the outer job, not a grandchild
#     under the inner job).
#
# All three reuse the helpers defined above. They're @pytest.mark.slow
# because each spins the real bundle + llama-server + ASR warmup, which
# takes 30-60 s even on a warm HF cache.
#
# These spawn the frozen yamabiko.exe, not dev-venv Python, because the
# lifecycle primitives (JobObject, named events, CREATE_NEW_PROCESS_GROUP
# semantics) interact with the kernel differently under PyInstaller
# windowed-exe hosting than under pythonw.exe.
# ---------------------------------------------------------------------------


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_graceful_shutdown_via_named_event() -> None:
    """Tier A happy path: watchdog signals pipeline shutdown event ->
    pipeline's finally block runs -> all three descendants exit cleanly.

    Fails red if:
      * The watchdog's self-shutdown listener thread isn't wired up, OR
      * The watchdog's finally block doesn't reach
        ``pipeline_writer.signal()``, OR
      * The pipeline's ShutdownEventListener isn't wired to app.done, OR
      * Pipeline's finally block skips the supervisor.stop() /
        inner_job.close() sequence, OR
      * Any of the three descendants remains alive after watchdog exits.

    One-line seeded defect that would make this red: comment out
    ``app.pipeline_writer.signal()`` in ``yamabiko.watchdog.run_watchdog``'s
    finally block -- pipeline would never receive the shutdown event,
    app.done stays unset, the asyncio wait never completes, and the
    pipeline PID survives the watchdog-exit window.

    Trigger: we open the watchdog process's self-directed shutdown
    event (``Local\\yamabiko.shutdown.watchdog.{pid}``) and fire SetEvent.
    CTRL_BREAK_EVENT was the original plan but doesn't reach a
    PyInstaller console=False GUI-subsystem bundle (no console =>
    Python's signal module never sees SIGBREAK). The self-directed
    named event is the production-ready equivalent: it's the same
    mechanism tray-Quit cascades use, and it works regardless of
    console state.

    Expected lifecycle markers in the main-*.log role logs (positive
    sentinels):
      * ``pipeline.ready``            -- pipeline composed + running
      * ``overlay.mainwindow.shown``  -- overlay fully launched
      * ``watchdog exiting; reason=`` -- watchdog finally block ran
    """
    _skip_if_no_bundle()
    # Default launch — the watchdog under test owns 8090.
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()

    tail_reader = _LogTail(_logs_dir())

    proc = subprocess.Popen(
        [str(_EXE)],
        cwd=str(_BUNDLE),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pids = _wait_for_three_processes(proc, timeout=60.0)
        assert len(pids) >= 3, (
            f"expected at least 3 supervised processes, got {len(pids)}: "
            f"{pids}"
        )
        # Give the pipeline a beat to finish wiring (ShutdownEventListener
        # is registered inside build_pipeline; we want the listener's
        # watcher thread alive before we signal).
        _wait_for_log_marker(
            tail_reader,
            "overlay server starting on http://127.0.0.1",
            timeout_s=90.0,
        )
        time.sleep(2.0)

        # Open the watchdog's self-directed shutdown event (created by
        # run_watchdog at startup with a predictable name based on the
        # watchdog's PID) and fire SetEvent. This kicks off the same
        # Tier A ordering _on_signal uses: pipeline_writer.signal()
        # BEFORE stop_event.set(), so the pipeline's graceful finally
        # gets the full grace window before terminate.
        from yamabiko.lifecycle.shutdown_event import ShutdownEventWriter
        wd_event_name = ShutdownEventWriter.name_for("watchdog", proc.pid)
        wd_writer = ShutdownEventWriter.open_existing(wd_event_name)
        assert wd_writer.handle is not None, (
            f"could not open watchdog self-shutdown event {wd_event_name!r}; "
            f"either the watchdog never ran run_watchdog or _watch_self_shutdown "
            f"didn't create the event"
        )
        assert wd_writer.signal(), (
            "SetEvent on watchdog self-shutdown event returned False; "
            "kernel rejected the signal"
        )
        wd_writer.close()

        # Watchdog + pipeline + overlay + llama all need to unwind.
        # Pipeline cleanup has a 3s overlay grace + 2s terminate
        # fallback + stage teardown; give the whole tree 20s to settle.
        try:
            rc = proc.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            pytest.fail(
                "watchdog did not exit within 20s of self-shutdown event. "
                "Graceful-shutdown chain is broken. Last log tail:\n"
                f"{tail_reader.read_new()[-3000:]}"
            )
        # rc is the watchdog's exit code. The self-shutdown path exits
        # via _on_signal -> stop_event -> supervisors' natural unwind
        # -> run_watchdog returns 0, so rc==0 on the happy path. We
        # accept any exit here; what matters is that it actually exited
        # and the children are gone.

        # All three supervised children must be gone. Give the kernel
        # a 3s window to close the outer job (belt+suspenders even on
        # the graceful path).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not any(_pid_alive(p) for p in pids):
                break
            time.sleep(0.2)
        survivors = [p for p in pids if _pid_alive(p)]
        assert not survivors, (
            f"pid(s) {survivors} survived graceful shutdown; "
            f"watchdog exited rc={rc} but descendants leaked. "
            f"Log tail:\n{tail_reader.read_new()[-3000:]}"
        )

        # Positive sentinel: watchdog actually reached its finally block.
        # Without this, a watchdog that crashed before signalling would
        # still pass the "descendants gone" check (kernel cleanup via
        # outer job close), masking a regression where
        # pipeline_writer.signal() never fires.
        tail = tail_reader.read_new()
        assert "watchdog exiting; reason=" in tail, (
            "watchdog exit sentinel missing from log -- watchdog's "
            "finally block did not run (pipeline_writer.signal() never "
            "fired). Tail:\n"
            f"{tail[-3000:]}"
        )
    finally:
        if proc.poll() is None:
            _kill_tree(proc.pid)
            time.sleep(1)


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_watchdog_hard_kill_cascades_via_job() -> None:
    """Outer Job Object invariant: TerminateProcess on the watchdog
    forces KILL_ON_JOB_CLOSE, which the kernel enforces for every
    descendant even when the watchdog had no chance to run finally
    blocks.

    Fails red if:
      * The outer Job Object was created without
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, OR
      * Children weren't assigned to the outer job (spawn-ordering
        regression; JobObject.assign() silently failed), OR
      * Any descendant PID survives >3s after the watchdog is killed.

    One-line seeded defect that would make this red: in
    ``yamabiko.lifecycle.jobs.create_outer_job``, drop the
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` flag. Children would then
    simply detach from the job when the handle closes, leaving
    orphaned llama-server + pipeline + overlay processes.

    3s is the kernel's deadline, not ours: KILL_ON_JOB_CLOSE fires
    synchronously when the last job handle is closed by the kernel
    (CloseHandle on process exit). Anything over ~1s indicates the
    job contract is broken, not that we were impatient.
    """
    _skip_if_no_bundle()
    # Default launch — the watchdog under test owns 8090.
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()

    tail_reader = _LogTail(_logs_dir())

    proc = subprocess.Popen(
        [str(_EXE)],
        cwd=str(_BUNDLE),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    try:
        pids = _wait_for_three_processes(proc, timeout=60.0)
        assert len(pids) >= 3, (
            f"expected at least 3 supervised processes, got {len(pids)}: "
            f"{pids}"
        )
        # Let the tree settle so llama-server is past its warmup and
        # has installed its own descendants (if any). Also ensures
        # JobObject.assign was exercised on the live children, not on
        # half-initialized skeletons.
        _wait_for_log_marker(
            tail_reader,
            "overlay server starting on http://127.0.0.1",
            timeout_s=90.0,
        )
        time.sleep(2.0)

        # Hard-kill the watchdog. proc.kill() on Windows -> TerminateProcess.
        # Kernel closes the outer Job Object handle -> every assigned
        # descendant is terminated synchronously.
        proc.kill()
        proc.wait(timeout=5.0)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not any(_pid_alive(p) for p in pids):
                break
            time.sleep(0.1)
        survivors = [p for p in pids if _pid_alive(p)]
        assert not survivors, (
            f"pid(s) {survivors} escaped the outer Job Object kill. "
            f"KILL_ON_JOB_CLOSE is either not set or children were "
            f"never assigned. Log tail:\n"
            f"{tail_reader.read_new()[-3000:]}"
        )
    finally:
        if proc.poll() is None:
            _kill_tree(proc.pid)
        # Belt-and-suspenders: any escapees still need cleanup so the
        # next test in the module can run. `pids` is initialized to []
        # above so this is safe even if _wait_for_three_processes raised.
        for pid in pids:
            _kill_pid(pid)
        time.sleep(1)


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_pipeline_kill_cascades_to_overlay_only() -> None:
    """Inner Job Object nest invariant: killing the pipeline takes the
    overlay with it (inner job, KILL_ON_JOB_CLOSE) but MUST NOT touch
    llama-server (llama is a sibling of pipeline under the watchdog's
    outer job, not a descendant of pipeline's inner job).

    Fails red if:
      * Pipeline's inner Job Object was never created / assigned, OR
      * Overlay wasn't bound to the inner job (pipeline_supervisor
        regression: ``self._inner_job.assign(proc)`` silently failed), OR
      * llama-server was accidentally bound to the inner job (would
        indicate the nest flattened; every kill of the pipeline would
        also kill llama, defeating the watchdog's ability to restart
        pipeline without paying the llama warmup cost again).

    One-line seeded defect that would make this red (overlay escape):
    remove ``self._inner_job.assign(proc)`` from
    ``PipelineSupervisor.spawn_once``. Overlay would remain in the
    outer job only; killing pipeline would leave overlay running
    (the watchdog's outer job only kills on WATCHDOG close, not on
    pipeline close).

    One-line seeded defect that would make this red (llama pulled in):
    add ``inner_job.assign(llama_proc)`` in watchdog's llama spawn.
    Killing pipeline would then collapse llama too, breaking the
    "restart pipeline without restarting llama" property.
    """
    _skip_if_no_bundle()
    # Default launch — the watchdog under test owns 8090.
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()

    tail_reader = _LogTail(_logs_dir())

    proc = subprocess.Popen(
        [str(_EXE)],
        cwd=str(_BUNDLE),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pids = _wait_for_three_processes(proc, timeout=60.0)
        _wait_for_log_marker(
            tail_reader,
            "overlay server starting on http://127.0.0.1",
            timeout_s=90.0,
        )
        time.sleep(2.0)

        roles = _classify_pids(pids)
        for expected in ("pipeline", "overlay", "llama"):
            assert expected in roles, (
                f"classifier did not find {expected!r} among pids={pids}. "
                f"Roles found: {roles}. "
                f"Does _classify_pids still match the current cmdline "
                f"conventions (see yamabiko.watchdog._build_*_cmd)?"
            )

        pipeline_pid = roles["pipeline"]
        overlay_pid = roles["overlay"]
        llama_pid = roles["llama"]

        # Hard-kill the pipeline. Kernel closes pipeline's inner job
        # handle as the process exits -> overlay (assigned to that
        # inner job) dies synchronously. llama is under the outer job
        # only, so it must survive.
        _kill_pid(pipeline_pid)

        # Overlay: must die within 3s (inner job KILL_ON_JOB_CLOSE).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not _pid_alive(overlay_pid):
                break
            time.sleep(0.1)
        assert not _pid_alive(overlay_pid), (
            f"overlay pid {overlay_pid} escaped the inner Job Object "
            f"cascade -- pipeline_supervisor.spawn_once isn't assigning "
            f"overlay to the inner job, OR the inner job lacks "
            f"KILL_ON_JOB_CLOSE. Log tail:\n"
            f"{tail_reader.read_new()[-3000:]}"
        )

        # llama-server: must still be running. If it died, pipeline's
        # inner job flattened the nest (llama was assigned to inner
        # instead of outer).
        assert _pid_alive(llama_pid), (
            f"llama-server pid {llama_pid} died when pipeline was "
            f"killed -- the outer/inner job nest is broken (llama was "
            f"assigned to the inner job by mistake). This defeats the "
            f"'restart pipeline cheaply without reloading the LLM' "
            f"design property. Log tail:\n"
            f"{tail_reader.read_new()[-3000:]}"
        )
    finally:
        if proc.poll() is None:
            _kill_tree(proc.pid)
        time.sleep(1)


@pytest.mark.frozen_owned_llama
def test_watchdog_restarts_pipeline_after_kill_reusing_warm_llama() -> None:
    """Watchdog restart property: after the pipeline is hard-killed, the
    watchdog must spawn a FRESH pipeline (and a fresh overlay under it) within
    the restart budget, REUSING the already-warm llama-server (same pid) --
    the "restart the pipeline cheaply without reloading the multi-GB GGUF"
    design property.

    The cascade test above (test_pipeline_kill_cascades_to_overlay_only)
    verifies only the KILL side: overlay dies, llama survives. A green cascade
    test would still pass if the watchdog then sat idle and NEVER brought the
    pipeline back -- leaving the user with no subtitles and no recovery. This
    test exercises the RESTART side, which no other E2E covers.

    Seeded-defect drill (no respawn): make ``Watchdog`` treat a killed
    pipeline's exit as STOP_ALL (or return from the supervise loop) instead of
    RESTART -> no new pipeline pid ever appears -> the restart assertion times
    out red.
    Seeded-defect drill (llama reloaded): route llama through the pipeline's
    inner job so it dies with the pipeline -> the watchdog respawns a NEW
    llama with a new pid -> the same-llama-pid assertion goes red.
    """
    _skip_if_no_bundle()
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()

    tail_reader = _LogTail(_logs_dir())
    proc = subprocess.Popen(
        [str(_EXE)],
        cwd=str(_BUNDLE),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pids = _wait_for_three_processes(proc, timeout=60.0)
        _wait_for_log_marker(
            tail_reader,
            "overlay server starting on http://127.0.0.1",
            timeout_s=90.0,
        )
        time.sleep(2.0)

        roles = _classify_pids(pids)
        pipeline_pid = roles["pipeline"]
        llama_pid = roles["llama"]

        # The watchdog treats a child death < _DEFAULT_MIN_LIFETIME_S (30s from
        # spawn, watchdog.py) as "likely a config error" and gives up WITHOUT
        # restarting -- crash-loop protection. The restart-and-reuse property
        # this test asserts only applies to a crash AFTER that threshold, so
        # wait past it before killing; otherwise the watchdog STOP_ALLs and a
        # green "no respawn" would be the CORRECT behaviour, not a bug. The
        # 3-process + marker wait above already burned ~10s of runtime; sleep
        # a margin so the pipeline's total runtime clears 30s.
        time.sleep(32.0)

        # Hard-kill the now-mature pipeline (overlay cascades via the inner
        # job; llama, a sibling under the outer job, survives -- proven by the
        # cascade test). The watchdog's exit-code policy maps a killed child
        # (neither 0 nor 42/43) past min-lifetime to RESTART.
        _kill_pid(pipeline_pid)

        # Wait for the watchdog to bring back a FRESH pipeline (+ overlay)
        # whose pid differs from the one we killed. 25s is a generous budget
        # for process respawn (the new pipeline's ASR warmup happens AFTER the
        # process exists, so the pid appears well before readiness).
        deadline = time.monotonic() + 25.0
        restarted: dict[str, int] = {}
        while time.monotonic() < deadline:
            try:
                children = [
                    c.pid
                    for c in psutil.Process(proc.pid).children(recursive=True)
                ]
            except psutil.NoSuchProcess:
                break
            cur = _classify_pids(children)
            if (
                cur.get("pipeline") not in (None, pipeline_pid)
                and "overlay" in cur
                and "llama" in cur
            ):
                restarted = cur
                break
            time.sleep(0.5)

        assert restarted, (
            f"watchdog did not respawn a fresh pipeline (+overlay+llama) "
            f"within 25s of killing pipeline pid {pipeline_pid}. The "
            f"supervisor is not restarting the pipeline -- a crashed pipeline "
            f"leaves the user with dead subtitles and no recovery. Log tail:\n"
            f"{tail_reader.read_new()[-3000:]}"
        )
        assert restarted["pipeline"] != pipeline_pid  # explicit: it's a NEW one
        assert restarted["llama"] == llama_pid, (
            f"llama-server pid changed ({llama_pid} -> {restarted['llama']}) "
            f"on a pipeline restart -- the warm GGUF was reloaded instead of "
            f"reused, paying the multi-second llama warmup on every pipeline "
            f"bounce. The outer/inner job nest must keep llama independent of "
            f"the pipeline's lifecycle. Log tail:\n{tail_reader.read_new()[-3000:]}"
        )
        assert _pid_alive(restarted["pipeline"]) and _pid_alive(
            restarted["overlay"]
        ), "restarted pipeline/overlay not alive after respawn"
    finally:
        if proc.poll() is None:
            _kill_tree(proc.pid)
        time.sleep(1)


# ---------------------------------------------------------------------------
# The TRUE Start-Menu-equivalent E2E.
#
# Every spawn above redirects std handles to subprocess.DEVNULL, which
# makes the print(flush=True) -> OSError [Errno 22] class structurally
# unreachable: the watchdog gets a valid NUL handle and its
# stdin=DEVNULL re-spawn propagates it to the pipeline child. This
# means a revert of _SafeTextIO would leave the DEVNULL-spawned sibling
# tests green while the real launch breaks. This test restores the real
# launch environment AND drives a real segment through the per-segment
# print(flush=True) in yamabiko/main.py, the exact statement the guard
# was added to protect.
# ---------------------------------------------------------------------------


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_frozen_exe_start_menu_launch_translates_and_log_stays_clean() -> None:
    """Default launch with NO inherited std handles (Start-Menu/Explorer
    equivalent): full audio -> subtitle round trip, then role logs
    asserted clean of the OSError/Traceback class.

    Topology under test: detached watchdog (no std handles) -> watchdog
    spawns pipeline with stdin=DEVNULL, which sets STARTF_USESTDHANDLES
    and hands the child the watchdog's (absent/invalid) stdout -> the
    pipeline's per-segment ``print(flush=True)`` in main.py executes
    against that handle once the first subtitle is produced. With the
    _SafeTextIO guard in place the print is swallowed; without it, the
    OSError tracebacks into main-pipeline.log and this test fails.

    Seeded-defect check: drop the _SafeTextIO guard, rebuild, and run:
        pytest tests/test_real_frozen_e2e.py -m real -k start_menu
    -> the log-clean assertion below goes red. The DEVNULL-spawned
    sibling tests stay green under the same revert because they hand
    the tree a valid NUL handle.
    """
    _skip_if_no_bundle()
    # Default launch — the watchdog under test owns 8090.
    _skip_if_llama_port_busy()
    _skip_if_no_translator_gguf()
    _skip_if_overlay_port_busy()
    if not _SAMPLE_WAV.is_file():
        pytest.skip(f"sample JP WAV missing at {_SAMPLE_WAV}")
    try:
        import winsound
    except ImportError:
        pytest.skip("winsound unavailable (non-Windows host)")

    tail_reader = _LogTail(_logs_dir())

    proc = _spawn_exe_start_menu_equivalent()
    try:
        # Budget: watchdog spawn + llama GGUF load + ASR warmup. The
        # watchdog-owned llama makes this slower than the --run-main
        # tests (which require a pre-warmed external server).
        _wait_for_log_marker(
            tail_reader,
            "overlay server starting on http://127.0.0.1",
            timeout_s=120,
        )
        time.sleep(1.0)

        winsound.PlaySound(
            str(_SAMPLE_WAV), winsound.SND_FILENAME | winsound.SND_ASYNC
        )
        try:
            subtitle = asyncio.run(_await_first_subtitle(_WS_URL, timeout_s=60))
        finally:
            winsound.PlaySound(None, 0)

        # The segment went all the way through — which means the
        # per-segment print at main.py:331 has executed at least once
        # in the no-std-handles environment.
        _assert_looks_translated(subtitle)

        # Give the post-broadcast print/flush a beat to hit the log if
        # it raised, then assert the whole window clean.
        time.sleep(2.0)
        tail = tail_reader.read_new()
        bad = [m for m in ("Traceback", "OSError", "AttributeError",
                           "ImportError", "ModuleNotFoundError") if m in tail]
        assert not bad, (
            f"forbidden marker(s) {bad} in role logs during a "
            f"start-menu-equivalent launch — the no-std-handles "
            f"environment surfaced an unguarded stdout write or a "
            f"startup exception the DEVNULL-spawned tests can't see. "
            f"Tail:\n{tail[-3000:]}"
        )

        # The watchdog itself must still be alive: an unguarded print
        # in the WATCHDOG process would kill the tree after the
        # subtitle was already delivered.
        assert proc.poll() is None, (
            f"watchdog exited rc={proc.poll()} during the test window"
        )
    finally:
        _kill_tree(proc.pid)
        time.sleep(1)


# ---------------------------------------------------------------------------
# Frozen launch-chain assertion (first-run wizard gate).
#
# Launch-chain coverage above stops at the watchdog spawning a running
# pipeline + overlay + llama (the steady state AFTER first-run setup is
# done). Nothing drives a FRESH-STATE install through the first-run
# wizard gate. The chain is: Start-Menu -> yamabiko.exe (watchdog.main)
# -> spawn yamabiko.exe --run-main -> main.main() ->
# wizard_state.needs_wizard() -> run_wizard(). If a refactor moves that
# gate or breaks --run-main dispatch, the wizard silently never runs,
# models never download, llama-server waits forever. The dev-mode
# flavour lives in tests/test_real_wizard_launch_chain.py; this is the
# frozen-exe (true Start-Menu spawn semantics) flavour, run at the
# release gate.
# ---------------------------------------------------------------------------


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_frozen_fresh_state_reaches_wizard(tmp_path) -> None:
    """Real Start-Menu launch: spawn dist yamabiko.exe with ZERO args, NO
    inherited std handles (DETACHED_PROCESS, no STARTF_USESTDHANDLES --
    the Start-Menu/Explorer condition), and a fresh YAMABIKO_APPDATA. The
    watchdog must spawn ``--run-main``, and the pipeline child must reach
    run_wizard() within 60 s -- asserted via the 'first-run wizard
    starting' marker in the spawned tree's main-pipeline.log under the
    fresh appdata. Kill the tree afterwards (the wizard window is a REAL
    GUI window; never click it).

    Because YAMABIKO_APPDATA is overridden, state.json AND the role logs
    resolve under tmp_path, so this test is fully isolated from the real
    install's %LOCALAPPDATA%\\Yamabiko (it neither reads nor clobbers it).
    Reading the log via the fresh appdata is why losing stdout capture to
    DETACHED_PROCESS is fine -- the marker is on disk, not on a pipe.

    The no-std-handles spawn matters beyond authenticity: the pipeline
    child inherits the watchdog's INVALID stdout via STARTF_USESTDHANDLES,
    so any unguarded ``print(flush=True)`` on the wizard path raises
    OSError [Errno 22]. The OSError entry in the bad-string scan below
    guards exactly that surface on this fresh-state path. A
    DEVNULL-spawned variant would hand the tree a valid NUL handle and
    could never see it.

    Seeded defects that turn this red:
    - comment out ``run_wizard()`` at the main.py needs_wizard() gate,
      rebuild, re-run -> the marker never logs, the 60 s poll times out;
    - drop a wizard-page module from the spec so ``--run-main`` dies on
      import before the gate -> ImportError/ModuleNotFoundError in the
      role log (a dev-venv test cannot see that import break);
    - revert the _SafeTextIO guard -> the wizard path's print raises
      OSError [Errno 22] into main-pipeline.log -> the OSError scan
      goes red.

    Skips (never silently) when the bundle is absent, another yamabiko.exe
    is already running (its tray instance would race this fresh-state
    spawn), or port 8090 is busy (the watchdog spawns its OWN llama-server
    there and a conflict would crash-loop the spawned tree).
    """
    _skip_if_no_bundle()
    # A running instance holds the single-instance mutex; our no-args
    # spawn would hit the "already running" dialog and exit before ever
    # reaching the wizard gate. The overlay-port probe is the cheapest
    # proxy for "another yamabiko.exe is live".
    _skip_if_overlay_port_busy()
    # Default launch -- the watchdog spawns its own llama-server on 8090.
    # An external llama (e.g. the --run-main tests' precondition) is a
    # port CONFLICT here, not a dependency: it would crash-loop the
    # spawned llama and contaminate the tree the wizard gate is being
    # measured against.
    _skip_if_llama_port_busy()

    # Logs land under the FRESH appdata, not %LOCALAPPDATA%, because the
    # watchdog propagates YAMABIKO_APPDATA to the --run-main child and
    # log_config writes to paths.logs_dir() = app_data_dir()/logs.
    tail_reader = _LogTail(tmp_path / "logs")

    # Full env copy + appdata override: env= REPLACES the environment, so
    # we must keep PATH / HF_HOME / CUDA-DLL search dirs the frozen exe
    # needs and only swap YAMABIKO_APPDATA.
    spawn_env = dict(os.environ)
    spawn_env["YAMABIKO_APPDATA"] = str(tmp_path)

    proc = _spawn_exe_start_menu_equivalent(env=spawn_env)
    try:
        # Cold frozen start: watchdog spawn + bootloader + Python import
        # of the pipeline child, then the wizard gate. The marker is the
        # first statement of run_wizard(), so it appears as soon as the
        # gate dispatches. 60 s covers cold-cache import variance. The marker
        # is imported at module scope from the Qt-free lifecycle.launch_markers
        # (single source of truth; no PySide6 drag into this frozen module).
        tail = _wait_for_log_marker(
            tail_reader,
            WIZARD_START_MARKER,
            timeout_s=60,
        )

        # The marker proves run_wizard() was entered. Defense in depth:
        # the same window must be free of startup/E2E bad strings.
        # OSError is the unguarded-stdout surface, reachable on this
        # no-std-handles wizard path; ImportError/ModuleNotFoundError
        # catch a spec that drops a wizard-page module so --run-main
        # dies before the gate.
        bad = [m for m in ("Traceback", "AttributeError", "ImportError",
                           "ModuleNotFoundError", "OSError",
                           "ERROR pipeline stage") if m in tail]
        assert not bad, (
            f"the launch chain reached the wizard marker but the role log "
            f"shows {bad} in the same window -- a frozen-only import/wiring "
            f"break or an unguarded stdout write on the --run-main -> "
            f"wizard path. Tail (last 3 KB):\n{tail[-3000:]}"
        )
    except TimeoutError as exc:
        # Re-raise as an explicit failure with the role-log tail so a
        # broken gate / dispatch is debuggable from CI output alone.
        tail = tail_reader.read_new()
        pytest.fail(
            f"frozen fresh-state launch never reached the first-run wizard "
            f"within 60 s: {exc}. The watchdog -> --run-main -> "
            f"needs_wizard() gate -> run_wizard() chain is broken in the "
            f"frozen bundle. Role-log tail (last 3 KB):\n{tail[-3000:]}"
        )
    finally:
        # The wizard window is a REAL GUI window. ALWAYS kill the whole
        # spawned tree (watchdog + pipeline-with-wizard + any llama that
        # spawned); never leave it open for a human to dismiss.
        _kill_tree(proc.pid)
        time.sleep(1)


def _skip_if_no_huggingface_network() -> None:
    """Skip the local E2E when Hugging Face itself is unreachable.

    The release wrapper rejects skips, so this remains a hard release gate
    while giving contributors an honest environment diagnosis instead of a
    product-looking timeout.
    """
    import httpx

    try:
        response = httpx.get(
            "https://huggingface.co",
            follow_redirects=True,
            timeout=10.0,
        )
    except httpx.TransportError as exc:
        pytest.skip(f"Hugging Face network unavailable: {exc!r}")
    if response.status_code >= 500:
        pytest.skip(
            "Hugging Face service unavailable: "
            f"HTTP {response.status_code} from connectivity probe"
        )


def _wait_for_wizard_window(desktop, title: str, timeout_s: float):
    """Return the single visible top-level wizard wrapper with ``title``."""
    deadline = time.monotonic() + timeout_s
    last_count = 0
    while time.monotonic() < deadline:
        windows = desktop.windows(title=title, visible_only=True)
        last_count = len(windows)
        if len(windows) == 1 and windows[0].is_enabled():
            return windows[0]
        time.sleep(0.25)
    pytest.fail(
        f"expected one visible enabled wizard {title!r} within "
        f"{timeout_s:.0f}s; last count={last_count}"
    )


def _wait_for_wizard_control(
    wizard,
    *,
    control_type: str,
    title: str | None = None,
    auto_id_suffix: str | None = None,
    timeout_s: float,
):
    """Find one live Qt UIA control by exact name and/or ID suffix."""
    deadline = time.monotonic() + timeout_s
    last_candidates: list[tuple[str, str, bool, bool]] = []
    while time.monotonic() < deadline:
        candidates = wizard.descendants(control_type=control_type)
        last_candidates = [
            (
                control.window_text(),
                control.element_info.automation_id or "",
                control.is_visible(),
                control.is_enabled(),
            )
            for control in candidates
        ]
        matches = [
            control
            for control in candidates
            if control.is_visible()
            and control.is_enabled()
            and (title is None or control.window_text() == title)
            and (
                auto_id_suffix is None
                or (control.element_info.automation_id or "").endswith(
                    auto_id_suffix
                )
            )
        ]
        if len(matches) == 1:
            return matches[0]
        time.sleep(0.25)
    pytest.fail(
        f"expected one visible enabled {control_type} with title={title!r}, "
        f"auto_id_suffix={auto_id_suffix!r} within {timeout_s:.0f}s; "
        f"last candidates={last_candidates!r}"
    )


_WIZARD_TITLE = "Yamabiko setup"
_LOOPBACK_SUFFIX = " [Loopback]"
_SOUND_OK = "Picking up sound."
_SOUND_NONE = "No sound picked up. Check that something is playing on this device."
_SOUND_OPEN_FAILED = "Couldn't listen to this device. Pick another one."
_CARD_CHECKING = "Checking\u2026"
_PROGRESS_LINE = re.compile(
    r"progress: (?P<file>\S+) (?P<done>\d+)/(?P<total>\d+) bytes"
)


def _click_wizard_button(wizard, text: str) -> None:
    """Invoke the visible, enabled push button labelled ``text`` (the
    setup window's footer and page buttons are real QPushButtons)."""
    _wait_for_wizard_control(
        wizard,
        control_type="Button",
        title=text,
        timeout_s=30,
    ).invoke()


def _visible_wizard_names(wizard) -> tuple[str, ...]:
    """UIA names of the setup window's visible elements, every control type:
    step titles and leads are painted TextBlocks (Pane/Custom, the text as
    their accessible name), result lines are QLabels (Text)."""
    values: list[str] = []
    for control in wizard.descendants():
        if not control.is_visible():
            continue
        value = control.window_text().strip()
        if value:
            values.append(value)
    return tuple(values)


def _foreground_window_title() -> str:
    """The foreground window's title, for a failure message: a physical
    click lands on whatever window is in front."""
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "<no foreground window>"
    title = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, title, 256)
    return title.value


def _wait_for_visible_wizard_text(
    wizard,
    expected: str,
    timeout_s: float,
) -> None:
    """Wait until a visible element of the setup window is named ``expected``."""
    deadline = time.monotonic() + timeout_s
    last_names: tuple[str, ...] = ()
    while time.monotonic() < deadline:
        last_names = _visible_wizard_names(wizard)
        if expected in last_names:
            return
        time.sleep(0.25)
    pytest.fail(
        f"setup never showed {expected!r} within {timeout_s:.0f}s; "
        f"visible names: {last_names!r}"
    )


def _wait_for_wizard_sound_test(wizard, timeout_s: float) -> None:
    """Wait for the Sound step's 5-second test: pass on "Picking up sound.",
    skip only on the hardware-style results (nothing picked up, the device
    can't be opened)."""
    deadline = time.monotonic() + timeout_s
    last_names: tuple[str, ...] = ()
    while time.monotonic() < deadline:
        last_names = _visible_wizard_names(wizard)
        if _SOUND_OK in last_names:
            return
        unavailable = next(
            (name for name in last_names if name in (_SOUND_NONE, _SOUND_OPEN_FAILED)),
            None,
        )
        if unavailable is not None:
            pytest.skip(f"real setup loopback audio unavailable: {unavailable}")
        time.sleep(0.25)
    pytest.fail(
        f"the Sound test never reported {_SOUND_OK!r} within {timeout_s:.0f}s; "
        f"visible names: {last_names!r}"
    )


def _wait_for_interim_audio_device(
    state_path: Path, expected: str, timeout_s: float,
) -> None:
    """Setup saves its interim state on every step change: wait until
    state.json names ``expected`` (the raw device name) as the audio device."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        last = data.get("audio_device_name") if isinstance(data, dict) else None
        if last == expected:
            return
        time.sleep(0.25)
    pytest.fail(
        f"state.json audio_device_name is {last!r} after Sound, expected the "
        f"Windows default loopback {expected!r}"
    )


def _default_wasapi_loopback_name() -> str:
    """Return the loopback that receives ``winsound`` default-device audio."""
    try:
        import pyaudiowpatch as pyaudio

        with pyaudio.PyAudio() as pa:
            info = pa.get_default_wasapi_loopback()
    except Exception as exc:
        pytest.skip(f"no default WASAPI loopback device: {exc!r}")
    name = str(info.get("name") or "").strip()
    if not name:
        pytest.skip("default WASAPI loopback has no device name")
    return name


def _wait_for_hf_xet_mapping(
    pipeline_pid: int,
    tail_reader: "_LogTail",
    timeout_s: float,
) -> str:
    """Require ``hf_xet*.pyd`` to be mapped before the first download ends."""
    deadline = time.monotonic() + timeout_s
    last_paths: tuple[str, ...] = ()
    last_access_error: str | None = None
    while time.monotonic() < deadline:
        tail = tail_reader.read_new()
        try:
            mappings = psutil.Process(pipeline_pid).memory_maps()
            last_paths = tuple(mapping.path for mapping in mappings if mapping.path)
            for mapped in last_paths:
                basename = Path(mapped).name.casefold()
                if basename.startswith("hf_xet") and basename.endswith(".pyd"):
                    return mapped
            last_access_error = None
        except psutil.AccessDenied as exc:
            last_access_error = repr(exc)
        except psutil.NoSuchProcess:
            pytest.fail(
                f"wizard-owning pipeline pid {pipeline_pid} exited during download"
            )
        if "hf_hub_download FAILED" in tail:
            pytest.fail(
                "frozen wizard download failed before hf_xet mapping was "
                f"observed. Log tail:\n{tail[-3000:]}"
            )
        if "hf_hub_download OK:" in tail:
            pytest.fail(
                "first frozen wizard download completed before an active "
                "hf_xet native mapping was observed"
            )
        time.sleep(0.25)

    candidates = [
        path for path in last_paths if "hf" in Path(path).name.casefold()
    ]
    pytest.fail(
        f"no mapped hf_xet*.pyd in pipeline pid {pipeline_pid} within "
        f"{timeout_s:.0f}s; HF-like mappings={candidates!r}; "
        f"last access error={last_access_error}"
    )


def _wait_for_first_hf_download_ok(
    tail_reader: "_LogTail",
    timeout_s: float,
) -> tuple[str, str]:
    """Return the full log window and first successful landed model path."""
    deadline = time.monotonic() + timeout_s
    tail = ""
    landed_re = re.compile(
        r"^.*hf_hub_download OK:.* landed=(?P<path>.+?)\r?$",
        re.MULTILINE,
    )
    while time.monotonic() < deadline:
        tail = tail_reader.read_new()
        match = landed_re.search(tail)
        if match is not None:
            return tail, match.group("path").strip()
        if "hf_hub_download FAILED" in tail:
            pytest.fail(
                "frozen wizard download failed before one file landed. "
                f"Log tail:\n{tail[-3000:]}"
            )
        time.sleep(0.5)
    pytest.fail(
        "no hf_hub_download OK marker within "
        f"{timeout_s:.0f}s. Log tail:\n{tail[-3000:]}"
    )


def _fresh_hf_spawn_env(tmp_path: Path) -> dict:
    """The frozen exe's env for a fresh-cache run: a full copy of this one
    (PATH and the frozen CUDA DLL search path), without the HF cache
    controls, and with ``YAMABIKO_APPDATA`` at ``tmp_path``. HF uses
    setdefault for those controls, so inherited values must be removed to
    make ``YAMABIKO_APPDATA`` the only source of the fresh cache. The token
    variables go too: setup downloads anonymously on a user's machine, and
    an inherited ``HF_TOKEN`` would let a still-private model repository
    pass here."""
    spawn_env = dict(os.environ)
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_XET",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        spawn_env.pop(name, None)
    spawn_env["YAMABIKO_APPDATA"] = str(tmp_path)
    return spawn_env


def _assert_tree_torn_down(
    root_pid: int,
    known_pids: set[int],
    ports: tuple[int, ...],
    what: str,
) -> None:
    """Kill ``root_pid``'s process tree, then fail when a pid of the run
    (``known_pids`` plus the root's descendants now) is still alive, or one
    of ``ports`` is still bound, 10 s later. Survivors are killed before the
    failure so the next test starts clean; ``what`` names the run."""
    pids = set(known_pids)
    try:
        pids.update(
            child.pid for child in psutil.Process(root_pid).children(recursive=True)
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass

    kill_error: str | None = None
    try:
        _kill_tree(root_pid)
    except (OSError, subprocess.SubprocessError) as exc:
        kill_error = repr(exc)

    deadline = time.monotonic() + 10
    survivors: list[int] = []
    bound_ports: list[int] = []
    while time.monotonic() < deadline:
        survivors = sorted(pid for pid in pids if _pid_alive(pid))
        bound_ports = [port for port in ports if _localhost_port_is_bound(port)]
        if not survivors and not bound_ports:
            return
        time.sleep(0.2)

    for survivor in survivors:
        try:
            _kill_tree(survivor)
        except (OSError, subprocess.SubprocessError):
            pass
    pytest.fail(
        f"{what} teardown leaked process(es) or listener(s): "
        f"surviving pids={survivors}, bound ports={bound_ports}, "
        f"taskkill error={kill_error}"
    )


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_frozen_fresh_wizard_download_loads_hf_xet_and_writes_fresh_cache(
    tmp_path,
) -> None:
    """Drive the real frozen setup (Start Menu launch, fresh appdata) through
    one fresh accelerated download.

    Get started; Sound: the Windows default output's row is marked System
    default and preselected, the 5-second test picks up the sample played to
    it, and the interim save names it in state.json; Content; Quality: once
    the card is detected, a real click on the tier row that is not
    preselected (quant_default_radio or quant_small_radio) selects it, and
    Start download saves that tier (translation.quant_tier in
    user_overrides.yaml). The queue starts with the speech model's model.bin, a
    Xet-backed file: hf_xet must be mapped in the pipeline before it lands,
    and it must land in the fresh cache.

    Port 8090 must be free because this starts the no-args watchdog. The
    test stops after the first file lands; it does not need an external
    llama-server and does not wait for the rest of the queue.

    Locally, absent bundle/network/loopback/audio prerequisites are explicit
    skips. The release wrapper inspects JUnit and rejects every skip.
    """
    _skip_if_no_bundle()
    _skip_if_overlay_port_busy()
    _skip_if_llama_port_busy()
    _skip_if_no_huggingface_network()
    if not _SAMPLE_WAV.is_file():
        pytest.skip(f"sample JP WAV missing at {_SAMPLE_WAV}")
    try:
        import winsound
    except ImportError:
        pytest.skip("winsound unavailable (non-Windows host)")
    default_loopback = _default_wasapi_loopback_name()
    default_row = (
        f"{default_loopback.removesuffix(_LOOPBACK_SUFFIX)}, System default"
    )
    try:
        from pywinauto import Desktop
    except ImportError:
        pytest.fail(
            "pywinauto is required by the frozen wizard release gate; "
            "install requirements-dev.txt"
        )

    tail_reader = _LogTail(tmp_path / "logs")
    expected_cache = tmp_path / "hf-cache"
    spawn_env = _fresh_hf_spawn_env(tmp_path)

    proc = _spawn_exe_start_menu_equivalent(env=spawn_env)
    playing = False
    pipeline_pid: int | None = None
    known_tree_pids = {proc.pid}
    try:
        _wait_for_log_marker(tail_reader, WIZARD_START_MARKER, timeout_s=60)

        wizard = _wait_for_wizard_window(
            Desktop(backend="uia"),
            _WIZARD_TITLE,
            timeout_s=60,
        )
        pipeline_pid = wizard.process_id()
        assert _pid_alive(pipeline_pid), (
            f"wizard owner pid {pipeline_pid} is not a live pipeline process"
        )
        try:
            descendant_pids = {
                child.pid
                for child in psutil.Process(proc.pid).children(recursive=True)
            }
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            pytest.fail(
                f"cannot inspect spawned watchdog pid {proc.pid}: {exc!r}"
            )
        known_tree_pids.update(descendant_pids)
        assert pipeline_pid in descendant_pids, (
            f"UIA wizard owner pid {pipeline_pid} is not a descendant of "
            f"spawned watchdog pid {proc.pid}; descendants={descendant_pids}"
        )
        assert _classify_pids([pipeline_pid]).get("pipeline") == pipeline_pid, (
            f"UIA wizard owner pid {pipeline_pid} is not the --run-main "
            "pipeline child"
        )

        _click_wizard_button(wizard, "Get started")

        _wait_for_visible_wizard_text(
            wizard,
            "Where does the sound play?",
            timeout_s=30,
        )
        device_row = _wait_for_wizard_control(
            wizard,
            control_type="CheckBox",
            title=default_row,
            timeout_s=30,
        )
        # A fresh setup preselects the Windows default output. The row is
        # read, not clicked: the list shows five rows and scrolls, so the
        # default output can sit outside the visible part of the list.
        assert device_row.get_toggle_state() == 1, (
            f"the default output's row {default_row!r} is not selected"
        )

        winsound.PlaySound(
            str(_SAMPLE_WAV),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
        )
        playing = True
        _click_wizard_button(wizard, "Test for 5 seconds")
        try:
            _wait_for_wizard_sound_test(wizard, timeout_s=20)
        finally:
            winsound.PlaySound(None, 0)
            playing = False

        _click_wizard_button(wizard, "Next")
        _wait_for_interim_audio_device(
            tmp_path / "state.json", default_loopback, timeout_s=10,
        )

        _wait_for_visible_wizard_text(
            wizard,
            "What do you watch most?",
            timeout_s=30,
        )
        _click_wizard_button(wizard, "Next")

        _wait_for_visible_wizard_text(
            wizard,
            "Pick a model size",
            timeout_s=30,
        )
        # Detection may still move the preselection: wait until the card
        # line leaves "Checking…" before reading which row is preselected.
        detected_deadline = time.monotonic() + 30
        while _CARD_CHECKING in _visible_wizard_names(wizard):
            if time.monotonic() > detected_deadline:
                pytest.fail(f"the Quality step's card line stayed at {_CARD_CHECKING!r}")
            time.sleep(0.25)
        tier_rows = {
            tier: _wait_for_wizard_control(
                wizard,
                control_type="CheckBox",
                auto_id_suffix=f".quant_{tier}_radio",
                timeout_s=30,
            )
            for tier in ("default", "small")
        }
        preselected = [
            tier for tier, row in tier_rows.items() if row.get_toggle_state() == 1
        ]
        assert len(preselected) == 1, f"preselected tier rows: {preselected}"
        # Click the other row, so the selection read back below can only come
        # from the click. The tier rows are painted checkable buttons: a
        # physical click is what a user does, and UIA's Toggle state reads it.
        picked = "small" if preselected == ["default"] else "default"
        # Setup in front first, so another window can't take the click.
        wizard.set_focus()
        tier_rows[picked].click_input()
        selected_deadline = time.monotonic() + 5
        while tier_rows[picked].get_toggle_state() != 1:
            if time.monotonic() > selected_deadline:
                states = {tier: row.get_toggle_state() for tier, row in tier_rows.items()}
                pytest.fail(
                    f"quant_{picked}_radio did not become selected through UIA; "
                    f"toggle states {states}, foreground window "
                    f"{_foreground_window_title()!r}"
                )
            time.sleep(0.1)

        _click_wizard_button(wizard, "Start download")
        _wait_for_visible_wizard_text(
            wizard,
            "Downloading models",
            timeout_s=30,
        )
        # Start download saved the clicked tier: the click above is the
        # user's, so it must reach the config the pipeline reads (Decision 148).
        import yaml

        overrides_path = tmp_path / "user_overrides.yaml"
        assert overrides_path.is_file(), (
            f"Start download saved no {overrides_path}"
        )
        overrides = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
        assert (overrides.get("translation") or {}).get("quant_tier") == picked, (
            f"{overrides_path} does not hold the clicked tier {picked!r}: {overrides}"
        )
        _wait_for_log_marker(
            tail_reader,
            "hf_hub_download start:",
            timeout_s=60,
        )

        mapped_xet = _wait_for_hf_xet_mapping(
            pipeline_pid,
            tail_reader,
            timeout_s=120,
        )
        assert Path(mapped_xet).name.casefold().startswith("hf_xet")

        tail, landed_text = _wait_for_first_hf_download_ok(
            tail_reader,
            timeout_s=2 * 60 * 60,
        )
        landed = Path(landed_text).resolve(strict=True)
        cache_root = expected_cache.resolve(strict=True)
        try:
            landed.relative_to(cache_root)
        except ValueError:
            pytest.fail(
                f"frozen wizard landed {landed} outside fresh Yamabiko HF "
                f"cache {cache_root}"
            )
        assert landed.is_file(), f"logged landed model is not a file: {landed}"
        assert _pid_alive(pipeline_pid), (
            f"wizard-owning pipeline pid {pipeline_pid} exited after download"
        )

        bad = [
            marker
            for marker in (
                "Traceback",
                "AttributeError",
                "ImportError",
                "ModuleNotFoundError",
                "OSError",
                "ERROR pipeline stage",
            )
            if marker in tail
        ]
        assert not bad, (
            f"frozen wizard download log contains {bad}. "
            f"Tail (last 4 KB):\n{tail[-4000:]}"
        )
    finally:
        if playing:
            winsound.PlaySound(None, 0)
        if pipeline_pid is not None:
            known_tree_pids.add(pipeline_pid)
        _assert_tree_torn_down(
            proc.pid, known_tree_pids, (8090, 8765), "frozen wizard",
        )


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_frozen_no_stderr_pipeline_wizard_download_reports_progress(
    tmp_path,
) -> None:
    """The download-running process itself started without std handles
    (``yamabiko.exe --run-main`` from Explorer, a shortcut or a DETACHED
    spawn) has sys.stdout and sys.stderr None. tqdm, which hf_hub_download's
    progress bar uses, writes to sys.stderr from its constructor, so before
    the frozen entry's sink (watchdog._guard_frozen_stdio) every download
    failed before its first byte with AttributeError("'NoneType' object has
    no attribute 'write'"). The default-launch tests cannot see this: the
    watchdog hands its children broken-pipe handles, not None.

    Spawns ``--run-main`` with no stdin/stdout/stderr arguments at all (any
    one of them makes CPython pass STARTF_USESTDHANDLES and valid or
    broken-pipe handles), walks setup to Start download with its defaults,
    and passes on the first progress line of a download that received bytes.

    Seeded defect: in _guard_frozen_stdio leave a None stream None, rebuild
    -> "hf_hub_download FAILED ... AttributeError(...'write')" within
    seconds -> red.
    """
    _skip_if_no_bundle()
    _skip_if_overlay_port_busy()
    _skip_if_no_huggingface_network()
    try:
        from pywinauto import Desktop
    except ImportError:
        pytest.fail(
            "pywinauto is required by the frozen wizard release gate; "
            "install requirements-dev.txt"
        )

    tail_reader = _LogTail(tmp_path / "logs")
    spawn_env = _fresh_hf_spawn_env(tmp_path)

    # The pipeline role itself, no watchdog: no std arguments and DETACHED,
    # so it starts with no std handles at all.
    proc = subprocess.Popen(
        [str(_EXE), "--run-main"],
        cwd=str(_BUNDLE),
        env=spawn_env,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        ),
    )
    known_tree_pids = {proc.pid}
    try:
        _wait_for_log_marker(tail_reader, WIZARD_START_MARKER, timeout_s=60)
        wizard = _wait_for_wizard_window(
            Desktop(backend="uia"),
            _WIZARD_TITLE,
            timeout_s=60,
        )
        assert wizard.process_id() == proc.pid, (
            f"setup is owned by pid {wizard.process_id()}, not the "
            f"handle-less --run-main pid {proc.pid}"
        )
        _click_wizard_button(wizard, "Get started")
        _wait_for_visible_wizard_text(
            wizard, "Where does the sound play?", timeout_s=30,
        )
        _click_wizard_button(wizard, "Next")
        _wait_for_visible_wizard_text(
            wizard, "What do you watch most?", timeout_s=30,
        )
        _click_wizard_button(wizard, "Next")
        _wait_for_visible_wizard_text(wizard, "Pick a model size", timeout_s=30)
        _click_wizard_button(wizard, "Start download")

        tail = ""
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            tail = tail_reader.read_new()
            if "hf_hub_download FAILED" in tail or "download failed for" in tail:
                pytest.fail(
                    "a download in a pipeline started without std handles "
                    f"failed. Log tail (last 3 KB):\n{tail[-3000:]}"
                )
            if any(
                int(match.group("done")) > 0
                for match in _PROGRESS_LINE.finditer(tail)
            ):
                break
            time.sleep(0.5)
        else:
            pytest.fail(
                "no download progress line within 120 s in a pipeline started "
                f"without std handles. Log tail (last 3 KB):\n{tail[-3000:]}"
            )

        bad = [m for m in ("Traceback", "AttributeError", "OSError") if m in tail]
        assert not bad, (
            f"the handle-less pipeline's download log contains {bad}. "
            f"Tail (last 4 KB):\n{tail[-4000:]}"
        )
    finally:
        # The role binds the overlay port (8765) and starts no llama-server,
        # so 8090 (which this test does not need free) is not checked.
        _assert_tree_torn_down(
            proc.pid, known_tree_pids, (8765,), "handle-less pipeline",
        )


# ---------------------------------------------------------------------------
# Startup error window ("Report card"): a seeded startup failure in the
# frozen app shows it, and its Copy report survives the pipeline's hard exit.
# ---------------------------------------------------------------------------

_CF_UNICODETEXT = 13
_CLIPBOARD_SENTINEL = "yamabiko e2e: the report was not copied"
_ERROR_WINDOW_CLASS = "ErrorWindow"
_COPIED_DESCRIPTION = "Copied. Paste it into a message to the developer."
# A hand-set speech model that is not downloaded: the first-run gate skips
# its whisper arm for an override (models.required_models_cached), so with
# HF_HUB_OFFLINE=1 the pipeline gets past the gate and fails at ASR init
# (huggingface_hub LocalEntryNotFoundError, a FileNotFoundError).
_MISSING_MODEL_OVERRIDE = (
    "asr:\n  faster_whisper:\n    model: yamabiko-e2e/model-not-downloaded\n"
)
_MODELS_MISSING_DIALOG_LINE = (
    "startup error dialog: Some model files are missing | "
    "Yamabiko can't start without them."
)
_PIPELINE_HARD_EXIT = "startup failure SystemExit(43); hard-exit STOP_ALL(43)"


def _skip_unless_setup_gate_models_cached() -> str:
    """The report-window seed overrides only the speech model, so the
    first-run gate (models.required_models_cached) still needs the
    translator GGUF and the Silero VAD file in the shared HF cache; without
    them the frozen app opens the setup wizard instead of failing. Returns
    that cache's HF_HOME as the app resolves it (the hub probes import
    yamabiko.hf_env, which pins it)."""
    from yamabiko import models

    gate_models = models._translator_gguf_cached() and models._repo_files_cached(
        models.SILERO_REPO, (models.SILERO_FILE,), models.SILERO_REVISION,
    )
    if not gate_models:
        pytest.skip(
            "the translator GGUF or the Silero VAD file is not in the shared HF "
            "cache, so a completed setup still opens the first-run wizard. Run "
            "the first-run wizard or preflight to download them."
        )
    return os.environ["HF_HOME"]


def _clipboard_api():
    """Private user32/kernel32 handles (argtypes stay local) for the
    CF_UNICODETEXT reads and writes below."""
    user32 = ctypes.WinDLL("user32")
    kernel32 = ctypes.WinDLL("kernel32")
    user32.OpenClipboard.argtypes = (wintypes.HWND,)
    user32.GetClipboardData.argtypes = (wintypes.UINT,)
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalFree.restype = wintypes.HGLOBAL
    return user32, kernel32


def _open_clipboard(user32) -> None:
    for _ in range(50):  # another process may hold the clipboard briefly
        if user32.OpenClipboard(None):
            return
        time.sleep(0.02)
    pytest.fail("the clipboard stayed open in another process")


def _clipboard_text() -> "str | None":
    """CF_UNICODETEXT straight from the Win32 clipboard; None without text."""
    user32, kernel32 = _clipboard_api()
    _open_clipboard(user32)
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return None
        pointer = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _clipboard_report_text() -> "str | None":
    """The clipboard's text with Qt's Windows line ends (CRLF, which Qt
    writes for every "\\n" in CF_UNICODETEXT) read as the report's own LF;
    None without text. The save/restore of the user's clipboard uses
    _clipboard_text, untouched."""
    text = _clipboard_text()
    return None if text is None else text.replace("\r\n", "\n")


def _set_clipboard_text(text: "str | None") -> None:
    """Put ``text`` on the clipboard as CF_UNICODETEXT (None: empty it). The
    test keeps the user's text clipboard this way; other formats are not
    kept (the app's own copy replaces them anyway)."""
    user32, kernel32 = _clipboard_api()
    handle = None
    if text is not None:
        data = ctypes.create_unicode_buffer(text)
        handle = kernel32.GlobalAlloc(0x0002, ctypes.sizeof(data))  # GMEM_MOVEABLE
        pointer = kernel32.GlobalLock(handle)
        ctypes.memmove(pointer, data, ctypes.sizeof(data))
        kernel32.GlobalUnlock(handle)
    _open_clipboard(user32)
    try:
        user32.EmptyClipboard()
        if handle is not None and not user32.SetClipboardData(_CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
    finally:
        user32.CloseClipboard()


def _tree_processes(root_pid: int) -> "dict[int, psutil.Process]":
    """The processes of the tree rooted at ``root_pid``. psutil objects keep
    their start time, so a reused pid never passes for one of them later."""
    try:
        root = psutil.Process(root_pid)
        return {p.pid: p for p in [root, *root.children(recursive=True)]}
    except psutil.Error:
        return {}


def _still_running(process: "psutil.Process") -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _wait_for_error_window(desktop, root_pid: int, timeout_s: float):
    """The one visible startup error window (UIA class ErrorWindow, title
    "Yamabiko") shown by a process of the tree rooted at ``root_pid`` (the
    overlay's window is titled "Yamabiko" too)."""
    deadline = time.monotonic() + timeout_s
    seen: list = []
    while time.monotonic() < deadline:
        pids = set(_tree_processes(root_pid))
        windows = [w for w in desktop.windows(title="Yamabiko", visible_only=True)
                   if w.process_id() in pids]
        seen = [(w.class_name(), w.process_id()) for w in windows]
        matches = [w for w in windows if w.class_name() == _ERROR_WINDOW_CLASS]
        if len(matches) == 1:
            return matches[0]
        time.sleep(0.25)
    pytest.fail(
        f"no visible {_ERROR_WINDOW_CLASS} window titled 'Yamabiko' in the "
        f"spawned tree within {timeout_s:.0f}s; its 'Yamabiko' windows: {seen}"
    )


def _uia_names(window) -> set:
    """Every UI Automation name inside ``window``, whitespace-normalized
    (the painted text blocks name themselves with their text)."""
    return {" ".join(c.window_text().split()) for c in window.descendants()}


def _uia_button(window, name: str, class_name: str):
    """The one Button named ``name`` of Qt class ``class_name`` (the caption's
    close button is also named "Close")."""
    matches = [b for b in window.descendants(control_type="Button")
               if b.window_text() == name and b.class_name() == class_name]
    assert len(matches) == 1, (
        f"expected one {class_name} button {name!r}; buttons: "
        f"{[(b.window_text(), b.class_name()) for b in window.descendants(control_type='Button')]}"
    )
    return matches[0]


def _wait_for_copied_state(button, timeout_s: float) -> None:
    """Wait until Copy report shows its copied state, read through UI
    Automation (the FullDescription Qt fills from the accessible description
    set_copied sets). The test never opens the clipboard while the app owns
    it: a read there makes Qt render its delayed text into the clipboard,
    which then outlives os._exit even without OleFlushClipboard, and it
    races the flush's own OpenClipboard (probe: the flush failed with
    CLIPBRD_E_CANT_OPEN while the test polled)."""
    from pywinauto.uia_defines import IUIA

    element = button.element_info.element.QueryInterface(
        IUIA().UIA_dll.IUIAutomationElement6
    )
    deadline = time.monotonic() + timeout_s
    description = ""
    while time.monotonic() < deadline:
        description = element.CurrentFullDescription
        if description == _COPIED_DESCRIPTION:
            return
        time.sleep(0.2)
    pytest.fail(
        f"Copy report did not show its copied state within {timeout_s:.0f}s "
        f"(description {description!r}); main-pipeline.log says why the copy failed"
    )


@pytest.mark.frozen_owned_llama
@pytest.mark.real
@pytest.mark.slow
def test_frozen_startup_failure_shows_the_report_window_and_copy_report_survives_the_exit(
    tmp_path,
) -> None:
    """Real Start-Menu launch (no args, no std handles) of the frozen app on
    a fresh YAMABIKO_APPDATA whose setup is complete but whose speech model
    (a hand-set override) is not downloaded, offline: the pipeline child
    fails at ASR init and shows the startup error window, the Report card
    (UIA class ErrorWindow, title "Yamabiko"), with the Models missing copy.
    Copy report, invoked through UI Automation, reaches its copied state;
    Close ends the pipeline with exit 43 (the watchdog stops its
    llama-server and exits 0), and the report is on the clipboard
    afterwards (the pipeline leaves through os._exit and the test reads
    nothing before, so OleFlushClipboard keeps it; a clipboard listener
    that has Qt render the text could keep it too, and mask the missing
    flush), with no user name or profile folder in it.
    The role logs are clean: the seeded failure's two chained tracebacks and
    its two ERROR lines only, the bundled fonts registered by the pipeline,
    no Qt warning while the window was up; nothing is left running.

    Seeded defects that turn this red:
    - the bundle built before this phase (QMessageBox, generic copy): the
      Models missing dialog line never logs -> TimeoutError;
    - drop Task 2's FileNotFoundError arm, rebuild -> "Yamabiko couldn't
      start" -> TimeoutError;
    - comment out OleFlushClipboard in copy_to_clipboard, rebuild -> the
      clipboard is empty after the exit -> red;
    - error_window missing from the bundle -> show_startup_error falls back
      to the Win32 box (class #32770, its own title) -> no ErrorWindow -> red.

    Skips (never silently): no bundle; another Yamabiko running; port 8090
    busy (this default launch spawns its own llama-server); the translator
    GGUF or the Silero file missing from the shared HF cache (the gate would
    open the wizard). The release runner turns every skip into a failure.
    """
    import getpass

    _skip_if_no_bundle()
    _skip_if_overlay_port_busy()
    _skip_if_llama_port_busy()
    hf_home = _skip_unless_setup_gate_models_cached()
    try:
        from pywinauto import Desktop
    except ImportError:
        pytest.fail(
            "pywinauto is required by the frozen release gates; "
            "install requirements-dev.txt"
        )
    from yamabiko import wizard_state

    # A completed setup, written by the product's own writer into the fresh
    # appdata, and the speech-model override.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YAMABIKO_APPDATA", str(tmp_path))
        wizard_state.save(wizard_state.WizardResult(first_run_complete=True))
    (tmp_path / "user_overrides.yaml").write_text(_MISSING_MODEL_OVERRIDE, encoding="utf-8")
    tail_reader = _LogTail(tmp_path / "logs")

    # Full env copy (PATH and the CUDA DLL dirs); the HF cache controls are
    # removed so only HF_HOME (the shared cache the gate checks) and offline
    # mode apply.
    spawn_env = dict(os.environ)
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_XET",
        "HF_XET_HIGH_PERFORMANCE",
    ):
        spawn_env.pop(name, None)
    spawn_env.update(YAMABIKO_APPDATA=str(tmp_path), HF_HOME=hf_home, HF_HUB_OFFLINE="1")

    profile = os.environ["USERPROFILE"]
    local = os.environ["LOCALAPPDATA"]
    user = os.environ.get("USERNAME") or getpass.getuser()
    saved_clipboard = _clipboard_text()
    proc = None
    known: dict = {}
    survivors: list = []
    try:
        _set_clipboard_text(_CLIPBOARD_SENTINEL)
        proc = _spawn_exe_start_menu_equivalent(env=spawn_env)
        known.update(_tree_processes(proc.pid))
        _wait_for_log_marker(tail_reader, _MODELS_MISSING_DIALOG_LINE, timeout_s=90)
        window = _wait_for_error_window(Desktop(backend="uia"), proc.pid, timeout_s=30)
        known.update(_tree_processes(proc.pid))

        names = _uia_names(window)
        expected = {
            "Some model files are missing",
            "Yamabiko can't start without them.",
            "To download them again, delete state.json in %LOCALAPPDATA%\\Yamabiko, "
            "then start Yamabiko.",
            "Setup runs again and gets only the missing files.",
            "Recent log",
        }
        assert expected <= names, (
            f"the window does not show the Models missing copy; missing "
            f"{sorted(expected - names)}; it shows {sorted(names)}"
        )

        copy_button = _uia_button(window, "Copy report", "CopyReportButton")
        copy_button.invoke()
        _wait_for_copied_state(copy_button, timeout_s=10)
        _uia_button(window, "Close", "QPushButton").invoke()
        try:
            watchdog_code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail("Yamabiko kept running 60 s after the error window's Close")
        copied = _clipboard_report_text()  # first read, after the exit; LF line ends
        tail = tail_reader.read_new()
    finally:
        if proc is not None:
            known.update(_tree_processes(proc.pid))
            if proc.poll() is None:
                _kill_tree(proc.pid)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            survivors = sorted(pid for pid, p in known.items() if _still_running(p))
            if not survivors:
                break
            time.sleep(0.2)
        for pid in survivors:
            try:
                known[pid].kill()
            except psutil.Error:
                pass
        _set_clipboard_text(saved_clipboard)

    # Survival: the whole tree ended with the window, nothing listens.
    assert not survivors, f"processes of the run still alive after its exit: {survivors}"
    bound = [port for port in (8090, 8765) if _localhost_port_is_bound(port)]
    assert not bound, f"ports still bound after the run: {bound}"
    assert watchdog_code == 0, f"the watchdog exited {watchdog_code}, not 0 (STOP_ALL)"
    assert "child exited: raw code=43 -> policy=STOP_ALL" in tail, (
        f"the pipeline did not exit 43 after Close. Tail (last 3 KB):\n{tail[-3000:]}"
    )

    # The copied report outlived the pipeline and is the redacted report.
    assert copied is not None and copied.startswith("Yamabiko startup error\n"), (
        "the copied report did not survive the pipeline's exit; the clipboard "
        f"now holds {(copied or '')[:200]!r}"
    )
    assert copied.startswith(
        "Yamabiko startup error\nTitle: Some model files are missing\n"
    ), copied[:400]
    assert "\nError: huggingface_hub.errors.LocalEntryNotFoundError: " in copied, copied
    assert _MODELS_MISSING_DIALOG_LINE.split(" | ")[0] in copied, (
        f"the report's recent log lacks the dialog line (read before it was "
        f"logged, or no log tail):\n{copied}"
    )
    # The pipeline logs HF_HOME at startup; with the shared cache under
    # %LOCALAPPDATA% (the default) the report must show it redacted.
    shown_hf_home = hf_home
    for base, name in ((local, "%LOCALAPPDATA%"), (profile, "%USERPROFILE%")):
        if hf_home.lower().startswith(base.lower() + "\\"):
            shown_hf_home = name + hf_home[len(base):]
            break
    assert f"HF_HOME={shown_hf_home} " in copied, (
        f"the pipeline's HF_HOME log line is missing from the report or not "
        f"redacted:\n{copied}"
    )
    lowered = copied.lower()
    leaks = [
        form
        for base in (profile, local)
        for form in (base, base.replace("\\", "/"), base.replace("\\", "\\\\"))
        if form.lower() in lowered
    ]
    assert not leaks, f"user folders in the copied report: {leaks}\n{copied}"
    assert not re.search(
        rf"(?<!\w){re.escape(user)}(?!\w)", copied.replace("<user>", ""), re.IGNORECASE,
    ), f"the user name {user!r} is in the copied report:\n{copied}"

    # Log asserted clean: only the seeded failure, fonts registered, no Qt
    # warning while the window was up.
    errors = [ln for ln in tail.splitlines() if " ERROR " in ln]
    expected_errors = [
        ln for ln in errors
        if " ERROR ASR startup failed" in ln or " ERROR startup error dialog: " in ln
    ]
    assert errors == expected_errors and len(errors) == 2, (
        f"ERROR lines beyond the seeded failure: {errors}"
    )
    assert tail.count("Traceback (most recent call last):") == 2, (
        f"tracebacks beyond the seeded failure's two chained ones. "
        f"Tail (last 4 KB):\n{tail[-4000:]}"
    )
    bad = [
        marker
        for marker in ("AttributeError", "ImportError", "ModuleNotFoundError",
                       "OSError", "ERROR pipeline stage")
        if marker in tail
    ]
    assert not bad, f"role logs show {bad}. Tail (last 4 KB):\n{tail[-4000:]}"
    fonts_line = next(
        (ln for ln in tail.splitlines() if "overlay.fonts.registered" in ln), ""
    )
    families = {f.strip() for f in fonts_line.partition("families=")[2].split(",")}
    assert {"BIZ UDPGothic", "BIZ UDPMincho", "Klee One"} <= families, (
        f"the pipeline did not register the bundled fonts for the window: "
        f"{fonts_line!r}"
    )
    pipeline_log = (tmp_path / "logs" / "main-pipeline.log").read_text(
        encoding="utf-8", errors="replace",
    )
    assert _PIPELINE_HARD_EXIT in pipeline_log, pipeline_log[-3000:]
    while_up, _, at_exit = pipeline_log.partition(_PIPELINE_HARD_EXIT)
    qt_while_up = [ln for ln in while_up.splitlines() if " yamabiko.qt " in ln]
    assert not qt_while_up, f"Qt warnings while the window was up: {qt_while_up}"
    # Qt's static destructors log two QThreadStorage lines at every pipeline
    # hard exit after Qt ran (DLL detach under os._exit; today's bundle too).
    qt_at_exit = [
        ln for ln in at_exit.splitlines()
        if " yamabiko.qt " in ln and "QThreadStorage" not in ln
    ]
    assert not qt_at_exit, f"Qt warnings at the exit: {qt_at_exit}"
