"""Startup-dialog marker: a named Win32 Event the pipeline holds while a
tailored startup dialog is on screen.

A pipeline child that hits an unrecoverable startup error (no audio
source, GPU missing, translator not ready) shows a modal dialog and exits
43 once the user dismisses it. The dialog blocks the child before
``/livez`` is served, so to the watchdog's liveness probe a child with a
dialog up looks exactly like one hung during boot. Without this marker the
probe force-restarted it after its startup budget, the dialog vanished
under the user, and the exit became a restart instead of 43.

The watchdog chooses the name (scoped to its own pid, like the Tier A
shutdown event) and passes it to the pipeline in ``ENV_VAR``. The child
creates the event just before the dialog opens and closes its handle when
the dialog closes. The probe only asks whether the event exists. A kernel
object lives exactly as long as some handle to it is open, so a child
that dies with its dialog up leaves no stale marker behind.

Non-Windows: ``held`` is a no-op and ``is_held`` returns False.
"""
from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
from ctypes import wintypes
from typing import Iterator

_log = logging.getLogger(__name__)

ENV_VAR = "YAMABIKO_STARTUP_DIALOG_EVENT"

_SYNCHRONIZE = 0x00100000

# Own WinDLL instance so these argtypes never collide with another
# module's bindings; use_last_error for the TLS-captured error code.
_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform.startswith("win") else None

if _k32 is not None:
    _CreateEventW = _k32.CreateEventW
    _CreateEventW.argtypes = [
        ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR,
    ]
    _CreateEventW.restype = wintypes.HANDLE

    _OpenEventW = _k32.OpenEventW
    _OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _OpenEventW.restype = wintypes.HANDLE

    _CloseHandle = _k32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL


def name_for(watchdog_pid: int) -> str:
    """The marker name for the pipeline supervised by ``watchdog_pid``."""
    return rf"Local\yamabiko.startup_dialog.{watchdog_pid}"


@contextlib.contextmanager
def held() -> Iterator[None]:
    """Hold the marker named in ``ENV_VAR`` for the duration of the block.

    A no-op when the variable is unset (no watchdog probe is watching this
    process) or off Windows. A failure to create the marker never blocks
    the dialog; it only costs the protection, so it is logged and ignored.
    """
    name = os.environ.get(ENV_VAR)
    handle = None
    if name and _k32 is not None:
        handle = _CreateEventW(None, True, False, name)
        if not handle:
            _log.warning(
                "could not create startup-dialog marker %r (err=%d); the "
                "watchdog may restart this process while the dialog is open",
                name, ctypes.get_last_error(),
            )
    try:
        yield
    finally:
        if handle:
            _CloseHandle(handle)


def is_held(name: str) -> bool:
    """True while some process holds the marker ``name``."""
    if _k32 is None:
        return False
    handle = _OpenEventW(_SYNCHRONIZE, False, name)
    if not handle:
        return False
    _CloseHandle(handle)
    return True
