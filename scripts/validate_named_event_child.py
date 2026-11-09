"""Probe child: opens a named Win32 Event, waits, logs when it wakes."""
from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes

SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0x00000000

def main() -> int:
    log_path = os.environ["PROBE_LOG"]
    event_name = os.environ["PROBE_EVENT"]
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"child_started pid={os.getpid()} frozen={getattr(sys, 'frozen', False)}\n")
        log.flush()

        OpenEventW = ctypes.windll.kernel32.OpenEventW
        OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        OpenEventW.restype = wintypes.HANDLE
        WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
        WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        WaitForSingleObject.restype = wintypes.DWORD

        handle = OpenEventW(SYNCHRONIZE, False, event_name)
        if not handle:
            err = ctypes.GetLastError()
            log.write(f"OpenEventW FAILED err={err}\n")
            return 1
        log.write(f"event_opened handle={handle}\n")
        log.flush()

        start = time.monotonic()
        rc = WaitForSingleObject(handle, 30_000)  # 30 s timeout
        elapsed = time.monotonic() - start
        if rc == WAIT_OBJECT_0:
            log.write(f"event_signalled elapsed_s={elapsed:.3f}\n")
            return 0
        log.write(f"wait_timeout_or_failed rc={rc} elapsed_s={elapsed:.3f}\n")
        return 2

if __name__ == "__main__":
    sys.exit(main())
