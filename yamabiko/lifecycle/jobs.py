"""Windows Job Object primitives.

`JobObject` wraps a single job handle. `create_outer_job` and
`create_inner_job` build jobs with KILL_ON_JOB_CLOSE semantics --
identical flags; nesting happens implicitly via assignment order.
Both also set BREAKAWAY_OK, so a child that EXPLICITLY asks for
CREATE_BREAKAWAY_FROM_JOB may leave the job (`open_url_outside_job`,
`show_in_folder_outside_job`); every other child still inherits
membership and dies with the job.

Non-Windows: all operations are no-ops so imports on Linux/macOS CI
don't fail.
"""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from ctypes import wintypes
from typing import Optional

_log = logging.getLogger(__name__)

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
# Lets a child created WITH CREATE_BREAKAWAY_FROM_JOB leave the job. Not
# SILENT_BREAKAWAY_OK (0x1000): that would free every child, including
# the pipeline, overlay and llama-server, from the kill cascade.
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9

# Atomic spawn-into-job support.
CREATE_SUSPENDED = 0x00000004
_TH32CS_SNAPTHREAD = 0x00000004
_THREAD_SUSPEND_RESUME = 0x0002
_RESUME_THREAD_FAILED = 0xFFFFFFFF  # (DWORD)-1

# Process-creation flags for open_url_outside_job and
# show_in_folder_outside_job (literal values so the module imports off
# Windows, where subprocess lacks the constants).
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _EXT_LIMIT(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BASIC_LIMIT),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


# Module-level ctypes signatures hoisted once at import time.
# Every API we call has argtypes/restype declared for x64 safety.
#
# Error-bearing calls are bound from a ``use_last_error=True``
# WinDLL so ctypes captures GetLastError into thread-local storage
# atomically at each FFI call; ``ctypes.get_last_error()`` then reads
# that captured slot. Reading the LIVE ``GetLastError()`` as a separate
# FFI call is racy -- any ctypes/CPython-internal Win32 call between the
# failing API and the read can clobber the thread's last-error value.
_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform.startswith("win") else None

_GetCurrentProcess = _k32.GetCurrentProcess if _k32 is not None else None
if _GetCurrentProcess is not None:
    _GetCurrentProcess.argtypes = []
    _GetCurrentProcess.restype = wintypes.HANDLE

_CloseHandle = _k32.CloseHandle if _k32 is not None else None
if _CloseHandle is not None:
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

_AssignProcessToJobObject = _k32.AssignProcessToJobObject if _k32 is not None else None
if _AssignProcessToJobObject is not None:
    _AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _AssignProcessToJobObject.restype = wintypes.BOOL

_IsProcessInJob = _k32.IsProcessInJob if _k32 is not None else None
if _IsProcessInJob is not None:
    _IsProcessInJob.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL),
    ]
    _IsProcessInJob.restype = wintypes.BOOL

# Job CREATION goes through the same use_last_error binding as every
# other call here -- a CreateJobObjectW / SetInformationJobObject
# failure is the one path that silently disables the overlay kill
# cascade, so its log line must carry the error code to be diagnosable
# from a frozen-bundle log.
_CreateJobObjectW = _k32.CreateJobObjectW if _k32 is not None else None
if _CreateJobObjectW is not None:
    _CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _CreateJobObjectW.restype = wintypes.HANDLE

_SetInformationJobObject = _k32.SetInformationJobObject if _k32 is not None else None
if _SetInformationJobObject is not None:
    _SetInformationJobObject.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
    ]
    _SetInformationJobObject.restype = wintypes.BOOL

# Toolhelp32 thread enumeration -- subprocess.Popen closes the
# child's thread handle inside _execute_child, so resuming a
# CREATE_SUSPENDED child requires re-discovering its (single) initial
# thread via a snapshot. This is the established workaround; the
# alternative (a full ctypes CreateProcessW reimplementation to keep
# the thread handle) re-invents quoting/env/inheritance handling that
# subprocess already gets right.
_CreateToolhelp32Snapshot = _k32.CreateToolhelp32Snapshot if _k32 is not None else None
if _CreateToolhelp32Snapshot is not None:
    _CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _CreateToolhelp32Snapshot.restype = wintypes.HANDLE

_Thread32First = _k32.Thread32First if _k32 is not None else None
if _Thread32First is not None:
    _Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
    _Thread32First.restype = wintypes.BOOL

_Thread32Next = _k32.Thread32Next if _k32 is not None else None
if _Thread32Next is not None:
    _Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
    _Thread32Next.restype = wintypes.BOOL

_OpenThread = _k32.OpenThread if _k32 is not None else None
if _OpenThread is not None:
    _OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _OpenThread.restype = wintypes.HANDLE

_ResumeThread = _k32.ResumeThread if _k32 is not None else None
if _ResumeThread is not None:
    _ResumeThread.argtypes = [wintypes.HANDLE]
    _ResumeThread.restype = wintypes.DWORD


class JobObject:
    """A Windows Job Object handle with KILL_ON_JOB_CLOSE.

    Non-Windows: handle stays None; all methods are no-ops.

    The kernel job is always ANONYMOUS. ``name`` is kept purely as
    a log label. A named ``CreateJobObjectW`` silently *opens* an
    existing job of the same name (the returned handle is non-NULL and
    ERROR_ALREADY_EXISTS is not surfaced by the ``if not handle``
    check), so two processes using the same fixed name -- e.g. two
    dev-mode ``python -m yamabiko.main`` runs both creating
    "yamabiko-pipeline-inner" -- would share ONE kernel job: the second
    instance's overlay joins the first instance's job and the second
    pipeline's death no longer fires KILL_ON_JOB_CLOSE for its own
    overlay. Nothing in the codebase ever opens a job by name, so
    anonymity costs nothing and removes the collision class entirely.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.handle: Optional[int] = None
        self._name = name
        if not sys.platform.startswith("win"):
            return
        try:
            # Anonymous kernel object -- never pass ``name`` here.
            handle = _CreateJobObjectW(None, None)
            if not handle:
                _log.warning(
                    "CreateJobObjectW(%r) failed: err=%d -- overlay "
                    "kill cascade disabled for this job",
                    name, ctypes.get_last_error(),
                )
                return

            info = _EXT_LIMIT()
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_BREAKAWAY_OK
            )
            ok = _SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                _log.warning(
                    "SetInformationJobObject(%r) failed: err=%d -- "
                    "overlay kill cascade disabled for this job",
                    name, ctypes.get_last_error(),
                )
                _CloseHandle(handle)
                return

            self.handle = handle
        except Exception as exc:
            _log.warning("JobObject(%r) setup raised: %s", name, exc)

    def assign(self, proc_or_pid) -> bool:
        """Assign a Popen, a raw HANDLE (int), or current process (None)
        to this job. Returns True on success, False on Windows failure,
        True on non-Windows (no-op).
        """
        if not sys.platform.startswith("win"):
            return True  # non-Windows: no-op treated as success
        if self.handle is None:
            return False
        try:
            if proc_or_pid is None:
                proc_handle = _GetCurrentProcess()
            elif isinstance(proc_or_pid, subprocess.Popen):
                proc_handle = int(proc_or_pid._handle)  # type: ignore[attr-defined]
            elif isinstance(proc_or_pid, int):
                proc_handle = proc_or_pid
            else:
                raise TypeError(f"unsupported: {type(proc_or_pid)}")

            if not _AssignProcessToJobObject(self.handle, proc_handle):
                # Read the TLS slot captured at the FFI call above,
                # not a fresh (clobberable) live GetLastError().
                err = ctypes.get_last_error()
                _log.warning(
                    "AssignProcessToJobObject(%r) failed: err=%d",
                    self._name, err,
                )
                return False
            return True
        except Exception as exc:
            _log.warning("assign(%r) raised: %s", self._name, exc)
            return False

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            _CloseHandle(self.handle)
        finally:
            self.handle = None

    def __enter__(self) -> "JobObject":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def create_outer_job(name: str) -> JobObject:
    """Create the outermost job for a watchdog. Returns a JobObject
    with KILL_ON_JOB_CLOSE and BREAKAWAY_OK; no UI limits."""
    return JobObject(name=name)


def create_inner_job(name: str) -> JobObject:
    """Create an inner (nested) job. Same flags as outer; the kernel
    forms the nest when the creator is already a member of another job.
    """
    return JobObject(name=name)


def _resume_process_threads(pid: int) -> int:
    """Resume every thread of ``pid`` (a CREATE_SUSPENDED child has
    exactly one). Returns the number of threads successfully resumed.

    Raises OSError if the Toolhelp32 snapshot itself fails -- the
    caller must not leave a permanently-suspended child behind.
    """
    snapshot = _CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        raise OSError(
            f"CreateToolhelp32Snapshot failed: err={ctypes.get_last_error()}"
        )
    resumed = 0
    try:
        entry = _THREADENTRY32()
        entry.dwSize = ctypes.sizeof(_THREADENTRY32)
        ok = _Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                thread = _OpenThread(
                    _THREAD_SUSPEND_RESUME, False, entry.th32ThreadID,
                )
                if thread:
                    try:
                        prev = _ResumeThread(thread)
                        if prev != _RESUME_THREAD_FAILED:
                            resumed += 1
                        else:
                            _log.warning(
                                "ResumeThread(tid=%d) failed: err=%d",
                                entry.th32ThreadID, ctypes.get_last_error(),
                            )
                    finally:
                        _CloseHandle(thread)
                else:
                    _log.warning(
                        "OpenThread(tid=%d) failed: err=%d",
                        entry.th32ThreadID, ctypes.get_last_error(),
                    )
            ok = _Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        _CloseHandle(snapshot)
    return resumed


def spawn_in_job(
    cmd,
    job: JobObject,
    *,
    env=None,
    creationflags: int = 0,
    **popen_kwargs,
) -> subprocess.Popen:
    """Spawn ``cmd`` so its FIRST instruction already runs inside
    ``job`` -- no window in which the child exists but is not yet a
    job member.

    The plain two-step ``Popen()`` then ``job.assign(proc)`` leaves a
    gap: if the spawning process is force-killed (taskkill /F, crash)
    between the two calls, the child keeps running OUTSIDE the job and
    survives the job's KILL_ON_JOB_CLOSE cascade -- the duplicate-
    overlay zombie class. Sequence here (the canonical Windows idiom):

      1. ``subprocess.Popen`` with ``CREATE_SUSPENDED`` -- the child's
         initial thread never runs a user-mode instruction yet;
      2. ``AssignProcessToJobObject`` while suspended;
      3. resume the initial thread (re-discovered via Toolhelp32,
         because Popen closes the thread handle internally).

    If step 2 fails, the failure is logged and the child still runs
    (same not-fatal contract as PipelineSupervisor.spawn_once: the
    named-event signal remains the graceful fallback). If step 3
    fails, the child is killed and the exception propagates -- a
    permanently-suspended child is worse than a failed spawn, which
    callers already handle with retry/backoff.

    Non-Windows: a plain Popen + no-op assign, for test collection.
    """
    if not sys.platform.startswith("win"):
        proc = subprocess.Popen(
            cmd, env=env, creationflags=creationflags, **popen_kwargs,
        )
        job.assign(proc)
        return proc

    proc = subprocess.Popen(
        cmd, env=env,
        creationflags=creationflags | CREATE_SUSPENDED,
        **popen_kwargs,
    )
    try:
        if not job.assign(proc):
            _log.warning(
                "spawn_in_job: pid %d not assigned to job %r; job kill-"
                "cascade will rely on the named-event fallback",
                proc.pid, job._name,
            )
        resumed = _resume_process_threads(proc.pid)
        if resumed < 1:
            raise OSError(
                f"spawn_in_job: no threads resumed for pid {proc.pid}; "
                f"child would stay suspended forever"
            )
    except Exception:
        # Never leave a permanently-suspended (or half-initialised)
        # child behind: it would hold its image/handles invisibly and
        # never exit on its own.
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception as cleanup_exc:  # pragma: no cover - best effort
            _log.warning(
                "spawn_in_job: cleanup of suspended pid %d failed: %s",
                proc.pid, cleanup_exc,
            )
        raise
    return proc


def is_in_job() -> bool:
    """Return True iff the current process is already inside a Job
    Object. Used as a startup probe so external-owner cases (NSSM,
    scheduler, debugger) are traceable."""
    if not sys.platform.startswith("win"):
        return False
    try:
        out = wintypes.BOOL(False)
        ok = _IsProcessInJob(
            _GetCurrentProcess(), None, ctypes.byref(out),
        )
        return bool(ok) and bool(out.value)
    except Exception as exc:
        _log.warning("is_in_job() raised: %s", exc)
        return False


def open_url_outside_job(url: str) -> None:
    """Open an https URL in the default browser, outside every Job Object.

    The overlay runs inside the pipeline's inner job and the watchdog's
    outer job, both KILL_ON_JOB_CLOSE. A browser started from here through
    ShellExecute would join both, so the next pipeline restart or a Quit
    would kill the user's browser. rundll32's FileProtocolHandler is
    started with CREATE_BREAKAWAY_FROM_JOB (both jobs set BREAKAWAY_OK),
    so it and the browser it launches run outside the jobs.

    Raises ValueError for anything but an https:// URL. A breakaway refused
    by a job Yamabiko does not own (ERROR_ACCESS_DENIED) falls back to a
    plain launch, which still opens the page; any other launch failure is
    logged, never raised into the Qt slot that called this.
    Non-Windows: no-op.
    """
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"refusing to open a non-https URL: {url!r}")
    if not sys.platform.startswith("win"):
        return
    cmd = ["rundll32.exe", "url.dll,FileProtocolHandler", url]
    try:
        subprocess.Popen(
            cmd,
            creationflags=CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW,
            close_fds=True,
        )
        return
    except PermissionError as exc:
        # A job above ours (a launcher or IDE) forbids breakaway.
        _log.warning(
            "breakaway launch refused (%s); opening the URL inside the job", exc,
        )
    except OSError:
        _log.exception("opening URL outside the job failed")
        return
    try:
        subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW, close_fds=True)
    except OSError:
        _log.exception("opening URL failed")


def show_in_folder_outside_job(path: "str | os.PathLike[str]") -> None:
    """Open an Explorer window on ``path``'s folder with the file selected,
    outside every Job Object (the "Diagnostics saved" toast click).

    Same reason as open_url_outside_job: an Explorer process started inside
    the overlay's jobs would close with the next pipeline restart or a Quit.
    explorer.exe gets ONE command line, ``/select,"<path>"``: as a list
    argument, Python would quote "/select,<path>" whole when the path has a
    space and Explorer would open Documents instead. A breakaway refused by
    a job Yamabiko does not own falls back to a plain launch; any other
    launch failure is logged, never raised into the Qt slot that called
    this. Returns without waiting for Explorer. Non-Windows: no-op.
    """
    if not sys.platform.startswith("win"):
        return
    cmd = f'explorer.exe /select,"{os.path.abspath(os.fspath(path))}"'
    try:
        subprocess.Popen(cmd, creationflags=CREATE_BREAKAWAY_FROM_JOB, close_fds=True)
        return
    except PermissionError as exc:
        # A job above ours (a launcher or IDE) forbids breakaway.
        _log.warning(
            "breakaway launch refused (%s); showing the file inside the job", exc,
        )
    except OSError:
        _log.exception("showing the file outside the job failed")
        return
    try:
        subprocess.Popen(cmd, close_fds=True)
    except OSError:
        _log.exception("showing the file failed")
