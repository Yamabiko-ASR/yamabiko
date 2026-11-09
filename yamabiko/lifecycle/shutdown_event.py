"""Named Win32 Event — Tier A shutdown signalling primitive.

Parent (watchdog / pipeline) creates an auto-named event and passes
its name to the child via a role-scoped environment variable
(``YAMABIKO_SHUTDOWN_EVENT_PIPELINE`` or
``YAMABIKO_SHUTDOWN_EVENT_OVERLAY``). Child opens the event; a daemon
thread blocks on WaitForSingleObject and wakes an asyncio.Event via
loop.call_soon_threadsafe on signal.

Event name format: ``Local\\yamabiko.shutdown.{role}.{parent_pid}``.
Manual-reset: SetEvent is idempotent; second signal is a no-op.

Role scoping: a single ``YAMABIKO_SHUTDOWN_EVENT`` env var carrying the
name regardless of role fails the moment a
process needs to listen for more than one role at once (a hypothetical
side-car observing pipeline + overlay). ``ENV_VAR_BY_ROLE`` maps each
role to its own env var; ``from_env(role=...)`` reads only that role's
variable. ``ENV_VAR`` remains as a legacy alias for the pipeline role
so older third-party integrations don't break.

Non-Windows: all operations are no-ops.
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import sys
import threading
from ctypes import wintypes
from typing import Callable, Optional

_log = logging.getLogger(__name__)

ENV_VAR_BY_ROLE = {
    "pipeline": "YAMABIKO_SHUTDOWN_EVENT_PIPELINE",
    "overlay":  "YAMABIKO_SHUTDOWN_EVENT_OVERLAY",
    # "watchdog": self-directed. Unlike pipeline/overlay, there is NO
    # production writer that sets this env var -- the watchdog is the
    # top of the process tree and has no parent to inject it. External
    # openers (the frozen-bundle E2E test; a tray-Quit cascade from
    # the overlay) use ``ShutdownEventWriter.open_existing(name_for(
    # "watchdog", ppid))`` directly, deriving the name from the
    # watchdog's PID. The env-var mapping entry exists so
    # ``ShutdownEventListener.from_env(role="watchdog")`` stays symmetric
    # with the other roles (useful for future scenarios where another
    # process wants to listen on the watchdog's own event) and so the
    # unit-test round-trip can exercise the full from_env path.
    "watchdog": "YAMABIKO_SHUTDOWN_EVENT_WATCHDOG",
}
# Legacy alias: old callers using ``ENV_VAR`` see the pipeline var, which
# is what they historically meant (overlay-as-pipeline-child didn't
# exist in the single-var era). New code should use ENV_VAR_BY_ROLE.
ENV_VAR = ENV_VAR_BY_ROLE["pipeline"]

SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0x00000000
INFINITE = 0xFFFFFFFF

# Module-level ctypes signatures hoisted once at import time.
#
# All error-bearing calls are bound from a ``use_last_error=True``
# WinDLL so ctypes captures GetLastError into thread-local storage
# atomically at each FFI call; the warn paths read it back via
# ``ctypes.get_last_error()``. A separate live ``GetLastError()`` FFI
# call can be clobbered by any intervening ctypes/CPython-internal
# Win32 call, logging a misleading error code.
_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform.startswith("win") else None

_CloseHandle = _k32.CloseHandle if _k32 is not None else None
if _CloseHandle is not None:
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

_CreateEventW = _k32.CreateEventW if _k32 is not None else None
if _CreateEventW is not None:
    _CreateEventW.argtypes = [
        ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR,
    ]
    _CreateEventW.restype = wintypes.HANDLE

_OpenEventW = _k32.OpenEventW if _k32 is not None else None
if _OpenEventW is not None:
    _OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _OpenEventW.restype = wintypes.HANDLE

_SetEvent = _k32.SetEvent if _k32 is not None else None
if _SetEvent is not None:
    _SetEvent.argtypes = [wintypes.HANDLE]
    _SetEvent.restype = wintypes.BOOL

_WaitForSingleObject = _k32.WaitForSingleObject if _k32 is not None else None
if _WaitForSingleObject is not None:
    _WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _WaitForSingleObject.restype = wintypes.DWORD

_WaitForMultipleObjects = _k32.WaitForMultipleObjects if _k32 is not None else None
if _WaitForMultipleObjects is not None:
    _WaitForMultipleObjects.argtypes = [
        wintypes.DWORD,                    # nCount
        ctypes.POINTER(wintypes.HANDLE),   # lpHandles
        wintypes.BOOL,                     # bWaitAll
        wintypes.DWORD,                    # dwMilliseconds
    ]
    _WaitForMultipleObjects.restype = wintypes.DWORD


def _name_for(role: str, parent_pid: int) -> str:
    return rf"Local\yamabiko.shutdown.{role}.{parent_pid}"


class ShutdownEventWriter:
    """Parent-side: owns the event handle and signals it."""

    def __init__(self, name: str, handle: Optional[int]) -> None:
        self.name = name
        self.handle = handle

    @classmethod
    def create(cls, role: str, parent_pid: int) -> "ShutdownEventWriter":
        name = _name_for(role, parent_pid)
        handle: Optional[int] = None
        if sys.platform.startswith("win"):
            try:
                # manual_reset=True, initial=False
                h = _CreateEventW(None, True, False, name)
                if h:
                    handle = h
                else:
                    err = ctypes.get_last_error()  # TLS-captured
                    _log.warning("CreateEventW(%r) failed: err=%d", name, err)
            except Exception as exc:
                _log.warning("ShutdownEventWriter.create(%r) raised: %s", name, exc)
        return cls(name=name, handle=handle)

    @classmethod
    def open_existing(cls, name: str) -> "ShutdownEventWriter":
        """Open an existing named event with modify-state access.

        Counterpart to ``create``: when another process already created
        the event, call this to get a writer that can ``signal()`` it.
        Used by the frozen-bundle E2E test (which opens the watchdog
        process's self-directed event by its predictable name + PID) and
        by a tray-Quit cascade (the overlay opens the watchdog's event
        to initiate a process-tree graceful shutdown).
        """
        handle: Optional[int] = None
        if sys.platform.startswith("win"):
            try:
                h = _OpenEventW(EVENT_MODIFY_STATE, False, name)
                if h:
                    handle = h
                else:
                    err = ctypes.get_last_error()  # TLS-captured
                    _log.warning(
                        "OpenEventW(%r, EVENT_MODIFY_STATE) failed: err=%d",
                        name, err,
                    )
            except Exception as exc:
                _log.warning(
                    "ShutdownEventWriter.open_existing(%r) raised: %s",
                    name, exc,
                )
        return cls(name=name, handle=handle)

    @staticmethod
    def name_for(role: str, parent_pid: int) -> str:
        """Expose the canonical event name so callers (tests, the
        tray-Quit cascade) can construct the exact name to pass to ``open_existing``
        without having to duplicate the ``Local\\yamabiko.shutdown.*``
        format string.
        """
        return _name_for(role, parent_pid)

    def signal(self) -> bool:
        if self.handle is None:
            return False
        try:
            if not _SetEvent(self.handle):
                err = ctypes.get_last_error()  # TLS-captured
                _log.warning("SetEvent(%r) failed: err=%d", self.name, err)
                return False
            return True
        except Exception as exc:
            _log.warning("SetEvent(%r) raised: %s", self.name, exc)
            return False

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            _CloseHandle(self.handle)
        finally:
            self.handle = None


class ShutdownEventListener:
    """Child-side: opens the event by name and wakes an asyncio.Event."""

    def __init__(self, name: str, handle: Optional[int]) -> None:
        self.name = name
        self.handle = handle
        self._thread: Optional[threading.Thread] = None
        # Private unnamed cancel event, created when a watcher starts. It
        # lets close() wake the watcher out of an otherwise-INFINITE wait
        # (SetEvent) so the thread can be joined BEFORE any handle is
        # closed. Closing a handle while a wait on it is still pending is
        # undefined behavior (Microsoft's WaitForMultipleObjects /
        # CloseHandle docs).
        self._cancel_handle: Optional[int] = None

    @classmethod
    def from_env(cls, role: str = "pipeline") -> Optional["ShutdownEventListener"]:
        """Open the Win32 Event named in the env var for ``role``.

        Role-scoped lookup: a pipeline process calls
        ``from_env(role="pipeline")`` to open the watchdog's pipeline
        event; an overlay process calls ``from_env(role="overlay")`` to
        open the pipeline's overlay event. Unknown roles raise so typos
        don't silently return None (which is indistinguishable from "no
        parent set the event" -- a real runtime condition we don't want
        masked).
        """
        env_var = ENV_VAR_BY_ROLE.get(role)
        if env_var is None:
            raise ValueError(
                f"unknown shutdown-event role: {role!r}; "
                f"expected one of {sorted(ENV_VAR_BY_ROLE)}"
            )
        name = os.environ.get(env_var)
        if not name:
            return None
        if not sys.platform.startswith("win"):
            return cls(name=name, handle=None)
        try:
            h = _OpenEventW(SYNCHRONIZE, False, name)
            if not h:
                err = ctypes.get_last_error()  # TLS-captured
                _log.warning("OpenEventW(%r) failed: err=%d", name, err)
                return cls(name=name, handle=None)
            return cls(name=name, handle=h)
        except Exception as exc:
            _log.warning("ShutdownEventListener.from_env raised: %s", exc)
            return cls(name=name, handle=None)

    def start_watcher(
        self,
        loop: asyncio.AbstractEventLoop,
        asyncio_event: asyncio.Event,
    ) -> None:
        """Spawn the daemon watcher; wake ``asyncio_event`` on ``loop``
        when the named event fires.

        asyncio adapter over ``_start_watcher``: an asyncio.Event may only
        be set from its loop thread, so the wake is marshalled via
        ``loop.call_soon_threadsafe``.
        """
        self._start_watcher(lambda: loop.call_soon_threadsafe(asyncio_event.set))

    def start_flag_watcher(self, on_fire: Callable[[], None]) -> None:
        """Spawn the daemon watcher; call ``on_fire`` when the named event
        fires.

        ``on_fire`` runs ON the watcher thread, so it must be a cheap,
        thread-safe hand-off (e.g. flipping a flag a Qt QTimer then polls).
        For non-asyncio consumers such as the overlay, which drives a Qt
        event loop rather than an asyncio loop and cannot use
        ``call_soon_threadsafe``.
        """
        self._start_watcher(on_fire)

    def _start_watcher(self, on_fire: Callable[[], None]) -> None:
        """Block on the named event on a daemon thread; call ``on_fire``
        exactly when it is signalled.

        A private unnamed cancel event is created alongside the thread and
        the wait is a ``WaitForMultipleObjects`` over BOTH handles, so
        ``close()`` can wake the thread out of an otherwise-INFINITE wait
        (SetEvent the cancel) and join it before any handle is closed.
        Closing a handle while a wait on it is still pending is undefined
        behavior (Microsoft docs); the cancel event is how we avoid it.
        """
        if self.handle is None:
            _log.info(
                "ShutdownEventListener(%r) has no handle; not starting watcher",
                self.name,
            )
            return
        if self._thread is not None:
            return

        # manual_reset=True, initial=False, unnamed.
        cancel = (
            _CreateEventW(None, True, False, None)
            if _CreateEventW is not None else None
        )
        if not cancel:
            # Extraordinarily rare (handle/resource exhaustion). Fall back
            # to a plain single-object wait; the thread is then NOT
            # cancellable, so close() will LEAK the handle rather than
            # close it under a pending wait (see close()).
            err = ctypes.get_last_error() if _CreateEventW is not None else 0
            _log.warning(
                "CreateEventW(cancel for %r) failed: err=%d; watcher not "
                "cancellable", self.name, err,
            )
            cancel = None
        self._cancel_handle = cancel or None
        named = self.handle

        def _watch() -> None:
            try:
                if cancel:
                    handles = (wintypes.HANDLE * 2)(named, cancel)
                    rc = _WaitForMultipleObjects(2, handles, False, INFINITE)
                    if rc == WAIT_OBJECT_0:
                        on_fire()
                    elif rc == WAIT_OBJECT_0 + 1:
                        # Cancel event: close() asked us to exit. Do NOT
                        # fire the shutdown hand-off.
                        return
                    else:
                        _log.warning(
                            "WaitForMultipleObjects(%r) returned %d",
                            self.name, rc,
                        )
                else:
                    rc = _WaitForSingleObject(named, INFINITE)
                    if rc == WAIT_OBJECT_0:
                        on_fire()
                    else:
                        _log.warning(
                            "WaitForSingleObject(%r) returned %d", self.name, rc,
                        )
            except Exception as exc:
                _log.warning("shutdown-event watcher(%r) raised: %s", self.name, exc)

        self._thread = threading.Thread(
            target=_watch, name="shutdown-event-watcher", daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Wake and join the watcher, then close both handles. Idempotent.

        Ordering is load-bearing: SetEvent(cancel) -> join -> CloseHandle.
        Closing the named (or cancel) handle while the watcher's wait is
        still pending on it is undefined behavior (Microsoft docs). If the
        watcher can't be woken (cancel-event creation had failed) or does
        not exit within the join bound, LEAK both handles rather than close
        them under a live wait -- two handles the OS reclaims at process
        exit is strictly better than UB. Safe if the watcher never started
        or already exited.
        """
        if self.handle is None and self._cancel_handle is None:
            return
        thread = self._thread
        if thread is not None and thread.is_alive():
            if self._cancel_handle is not None:
                _SetEvent(self._cancel_handle)
                thread.join(timeout=2.0)
            if thread.is_alive():
                _log.warning(
                    "shutdown-event watcher(%r) still blocked at close; "
                    "leaking handles to avoid closing under a pending wait",
                    self.name,
                )
                return
        self._thread = None
        if self.handle is not None:
            try:
                _CloseHandle(self.handle)
            finally:
                self.handle = None
        if self._cancel_handle is not None:
            try:
                _CloseHandle(self._cancel_handle)
            finally:
                self._cancel_handle = None
