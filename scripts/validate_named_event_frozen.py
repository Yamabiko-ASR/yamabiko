"""Parent: build the probe child, spawn the frozen exe, signal via named event."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "probe"
DIST_DIR = BUILD_DIR / "dist" / "probe_child"
CHILD_EXE = DIST_DIR / "probe_child.exe"
LOG = BUILD_DIR / "probe.log"

# Build the frozen probe child.
def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "PyInstaller",
         "--noconfirm",
         "--distpath", str(BUILD_DIR / "dist"),
         "--workpath", str(BUILD_DIR / "work"),
         str(ROOT / "scripts" / "validate_named_event_child.spec")],
        cwd=ROOT, check=True,
    )

def _create_event(name: str) -> int:
    CreateEventW = ctypes.windll.kernel32.CreateEventW
    CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    CreateEventW.restype = wintypes.HANDLE
    handle = CreateEventW(None, True, False, name)
    if not handle:
        raise OSError(f"CreateEventW failed: {ctypes.GetLastError()}")
    return handle

def _set_event(handle: int) -> None:
    SetEvent = ctypes.windll.kernel32.SetEvent
    SetEvent.argtypes = [wintypes.HANDLE]
    SetEvent.restype = wintypes.BOOL
    if not SetEvent(handle):
        raise OSError(f"SetEvent failed: {ctypes.GetLastError()}")

def main() -> int:
    if sys.platform != "win32":
        print("SKIP: Windows-only probe")
        return 0
    if LOG.exists():
        LOG.unlink()

    if not CHILD_EXE.exists():
        print(f"Building probe child at {CHILD_EXE}...")
        build()

    event_name = r"Local\yamabiko.probe.shutdown"
    parent_handle = _create_event(event_name)
    print(f"parent created event: {event_name}")

    env = os.environ.copy()
    env["PROBE_EVENT"] = event_name
    env["PROBE_LOG"] = str(LOG)

    proc = subprocess.Popen([str(CHILD_EXE)], env=env)
    print(f"spawned probe child pid={proc.pid}")

    time.sleep(2.0)  # give child time to open the event
    print("signalling event...")
    _set_event(parent_handle)

    rc = proc.wait(timeout=10.0)
    print(f"child exited rc={rc}")
    print(f"--- probe log ---\n{LOG.read_text(encoding='utf-8')}")

    if rc == 0:
        print("\nRESULT: PASS — named event reached frozen onedir child.")
        return 0
    print("\nRESULT: FAIL — switch to Plan B (AllocConsole + CTRL_BREAK_EVENT).")
    return 1

if __name__ == "__main__":
    sys.exit(main())
