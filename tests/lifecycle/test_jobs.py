"""Unit tests for yamabiko.lifecycle.jobs.

Each test's docstring carries a "Fails red if:" line per TESTING.md §3.1.
"""
from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="Job Objects are Windows-only",
)

import ctypes

from yamabiko.lifecycle.jobs import (
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JobObject,
    _BASIC_LIMIT,
    _EXT_LIMIT,
    create_outer_job,
    create_inner_job,
    is_in_job,
)


def test_kill_on_job_close_flag_value():
    """Fails red if: the KILL_ON_JOB_CLOSE constant is typo'd
    to any other value (e.g. BREAKAWAY_OK = 0x00000800)."""
    assert JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x00002000


def test_job_creation_failure_logs_win32_error_code(monkeypatch, caplog):
    """A CreateJobObjectW failure is the one path that silently disables
    the overlay kill cascade, so its warning must carry the Win32 error
    code (use_last_error binding + ctypes.get_last_error) to be
    diagnosable from a frozen-bundle log.

    Fails red if: JobObject.__init__ reverts to a local
    ``ctypes.windll`` binding / drops ``err=%d`` from the warning, or
    stops handling a NULL job handle.
    """
    import logging

    import yamabiko.lifecycle.jobs as jobs_mod

    def _failing_create(_security, _name):
        ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED
        return None

    monkeypatch.setattr(jobs_mod, "_CreateJobObjectW", _failing_create)
    caplog.set_level(logging.WARNING, logger="yamabiko.lifecycle.jobs")

    job = JobObject(name="err-probe")

    assert job.handle is None
    failures = [
        r.getMessage() for r in caplog.records
        if "CreateJobObjectW" in r.getMessage()
    ]
    assert failures, "creation failure produced no warning at all"
    assert "err=5" in failures[0], (
        f"creation-failure warning carries no Win32 error code: "
        f"{failures[0]!r}"
    )


def test_basic_limit_struct_limitflags_write_reaches_c_memory():
    """Fails red if: the ``LimitFlags`` entry in ``_BASIC_LIMIT._fields_``
    is renamed or reordered. ctypes.Structure silently allows setting an
    attribute whose name is NOT in ``_fields_``; the write lands in the
    Python ``__dict__`` instead of C memory, and ``SetInformationJobObject``
    gets a zero-valued buffer. The kill cascade then silently no-ops.

    This reproduces the exact write ``JobObject.__init__`` performs at
    jobs.py:105 and asserts that the DWORD at the expected byte offset
    in the struct's memory buffer carries the written value.

    Kills mutmut mutant 19 fast instead of the 10 s+ timeout that
    ``test_job_kills_child_when_parent_job_handle_closes`` takes when
    the mutation survives behaviourally.
    """
    info = _EXT_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    # LimitFlags lives at offset(BasicLimitInformation) + offsetof(LimitFlags)
    # within _EXT_LIMIT. Resolve offset AND size via ctypes metadata so this
    # test stays correct if the enclosing struct gains new leading fields,
    # and also goes red on a type-swap mutation (DWORD -> c_longlong) since
    # the read width tracks the declared field type.
    basic_offset = _EXT_LIMIT.BasicLimitInformation.offset
    limitflags_offset = _BASIC_LIMIT.LimitFlags.offset
    limitflags_size = _BASIC_LIMIT.LimitFlags.size
    absolute_offset = basic_offset + limitflags_offset

    raw = ctypes.string_at(ctypes.byref(info), ctypes.sizeof(info))
    observed = int.from_bytes(
        raw[absolute_offset : absolute_offset + limitflags_size], "little",
    )
    assert observed == JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, (
        f"LimitFlags write did not reach C memory at offset "
        f"{absolute_offset}; memory reads {observed:#010x}, expected "
        f"{JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE:#010x}. A rename of the "
        f"'LimitFlags' entry in _BASIC_LIMIT._fields_ would produce "
        f"exactly this symptom: ctypes silently stores the write as a "
        f"Python attribute while the C struct stays zero, making "
        f"KILL_ON_JOB_CLOSE a no-op across the whole watchdog."
    )


def test_create_outer_job_returns_object_with_handle():
    """Fails red if: create_outer_job silently no-ops and returns
    a JobObject with None handle on Windows."""
    job = create_outer_job("yamabiko-test-outer")
    try:
        assert job.handle is not None, "Job handle must be non-None on Windows"
    finally:
        job.close()


def test_create_inner_job_returns_object_with_handle():
    """Fails red if: create_inner_job is not implemented or silently
    returns a no-op JobObject."""
    job = create_inner_job("yamabiko-test-inner")
    try:
        assert job.handle is not None
    finally:
        job.close()


def test_two_jobobjects_with_same_name_are_distinct_kernel_objects():
    """Two ``JobObject`` instances constructed with the SAME descriptive
    name must wrap two DISTINCT kernel job objects.

    Named ``CreateJobObjectW`` silently OPENS an existing job of the
    same name (ERROR_ALREADY_EXISTS is not a failure -- the handle is
    non-NULL), so a second yamabiko stack (``python -m yamabiko.main``
    has no single-instance mutex) sharing "yamabiko-pipeline-inner"
    would bind its overlay to the FIRST instance's job: the second
    pipeline's death no longer cascades KILL_ON_JOB_CLOSE to its own
    overlay. The fix passes name=None to the Win32 API (anonymous jobs
    cannot collide); the name stays as a log label only.

    Discriminator: kernelbase!CompareObjectHandles reports whether two
    handles refer to the same kernel object. Same-object -> the
    collision is live -> red.

    Seeded-defect drill: pass ``name`` back through to
    ``CreateJobObjectW(None, name)`` in JobObject.__init__ -> both
    handles alias one kernel job -> CompareObjectHandles returns TRUE
    -> red.
    """
    from ctypes import wintypes

    kernelbase = ctypes.WinDLL("kernelbase")
    CompareObjectHandles = kernelbase.CompareObjectHandles
    CompareObjectHandles.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    CompareObjectHandles.restype = wintypes.BOOL

    job1 = JobObject(name="yamabiko-test-f74-collision")
    job2 = JobObject(name="yamabiko-test-f74-collision")
    try:
        assert job1.handle is not None and job2.handle is not None, (
            "precondition: both job creations must succeed"
        )
        # Sanity: the API itself works (a handle always equals itself).
        # Guards against a CompareObjectHandles failure (returns FALSE)
        # making the real assertion below pass vacuously.
        assert CompareObjectHandles(job1.handle, job1.handle), (
            "CompareObjectHandles self-comparison returned FALSE; the "
            "probe API is broken on this host and the test is vacuous"
        )
        assert not CompareObjectHandles(job1.handle, job2.handle), (
            "two JobObjects built with the same descriptive name alias "
            "ONE kernel job object (named CreateJobObjectW opened the "
            "existing job instead of creating a new one). A second "
            "yamabiko stack would share the first stack's inner job and "
            "lose its own overlay kill-cascade."
        )
    finally:
        job1.close()
        job2.close()


def test_is_in_job_returns_bool():
    """Fails red if: is_in_job raises instead of returning bool."""
    result = is_in_job()
    assert isinstance(result, bool)


def test_assign_none_routes_through_get_current_process(monkeypatch):
    """Fails red if: ``JobObject.assign(None)`` doesn't take the
    ``_GetCurrentProcess()`` branch (mutmut mutant 80:
    ``proc_or_pid is None`` -> ``is not None``).

    Under the mutation ``.assign(None)`` falls through to the ``else:
    raise TypeError(...)`` limb, the outer ``except Exception`` catches,
    ``.assign`` returns False. The existing subprocess-based test
    ``test_nested_job_assignment_places_child_in_both_jobs`` already
    catches this behaviourally, BUT under the mutation ``.assign(proc)``
    ALSO misroutes (the same `is not None` check is true for a real
    Popen, sending it to ``_GetCurrentProcess()`` -- i.e., the pytest
    handle). Closing that job kills pytest via KILL_ON_JOB_CLOSE, which
    makes mutmut mis-classify the mutant as surviving (crash vs. clean
    failure). This test exercises ONLY the None branch and stubs
    ``AssignProcessToJobObject`` so the job has no members -- ``close()``
    can't cascade onto pytest.

    Seeded-defect drill: changing ``is None`` to ``is not None`` in
    jobs.py:141 makes ``job.assign(None)`` return False, first assert
    goes red.
    """
    from unittest.mock import MagicMock
    import yamabiko.lifecycle.jobs as jobs_mod

    current_process_sentinel = 0xDEADBEEF
    get_current_calls = []

    def fake_get_current():
        get_current_calls.append(1)
        return current_process_sentinel

    fake_assign_api = MagicMock(return_value=1)  # BOOL True

    monkeypatch.setattr(jobs_mod, "_GetCurrentProcess", fake_get_current)
    # The binding is at the module-level ``_AssignProcessToJobObject``
    # (bound from the use_last_error WinDLL); patch the module
    # attribute, the seam assign() reads.
    monkeypatch.setattr(
        jobs_mod, "_AssignProcessToJobObject", fake_assign_api,
    )

    # Real JobObject -- needs a real handle for the assign() code path
    # to reach the branch we care about. Our stubbed
    # AssignProcessToJobObject means no process actually gets assigned,
    # so close() is safe (no members to kill).
    job = JobObject(name="yamabiko-test-d1fu-assign-none")
    try:
        assert job.handle is not None, "precondition: real job handle"
        result = job.assign(None)
        assert result is True, (
            f"assign(None) returned {result!r}; under normal source the "
            f"is-None branch should pick up _GetCurrentProcess() and "
            f"call AssignProcessToJobObject which the stub returns True "
            f"from. Mutant 80 regression -- None falls through to the "
            f"TypeError path."
        )
        assert get_current_calls, (
            "_GetCurrentProcess was never called; assign(None) did not "
            "enter the is-None branch. Mutant 80 regression."
        )
        # Confirm the assign API saw our sentinel handle (not some
        # leaked real handle that happens to truthy-pass the test).
        assert fake_assign_api.call_count == 1
        _, proc_arg = fake_assign_api.call_args.args
        assert proc_arg == current_process_sentinel, (
            f"AssignProcessToJobObject got proc={proc_arg!r}; expected "
            f"the _GetCurrentProcess sentinel {current_process_sentinel:#x}"
        )
    finally:
        job.close()


def test_is_in_job_returns_false_when_api_succeeds_but_process_not_in_job(
    monkeypatch,
):
    """Fails red if: ``is_in_job()`` returns True when the Win32 API
    reports ``ok=True, out.value=0`` (process not in a job) -- i.e., the
    ``and`` in ``return bool(ok) and bool(out.value)`` is swapped to
    ``or`` (mutmut mutant 103). Under the mutation the watchdog startup
    probe would log the wrong is-in-job state: claiming to be in a job
    when it isn't, masking the "external-owner" diagnostic case the
    probe exists for.

    Seeded-defect drill: changing ``and`` to ``or`` on jobs.py:206 makes
    the assertion go red (is_in_job returns True).
    """
    from unittest.mock import MagicMock
    import yamabiko.lifecycle.jobs as jobs_mod

    # Fake IsProcessInJob: BOOL True (api succeeded) but do NOT touch
    # the out pointer. The production code inits ``out = wintypes.BOOL(
    # False)`` before the call -- since our stub never writes to it, it
    # stays 0, exactly the "not in a job" state we need to exercise.
    # The binding lives at module-level ``_IsProcessInJob``;
    # patch the module attribute, the seam is_in_job() reads.
    fake_api = MagicMock(return_value=1)

    monkeypatch.setattr(jobs_mod, "_IsProcessInJob", fake_api)

    result = is_in_job()
    assert result is False, (
        f"is_in_job() returned {result!r} when IsProcessInJob reported "
        f"ok=True, out.value=0 (process not in a job). Expected False. "
        f"Mutant 103 ('and' -> 'or') regression: swapping 'and' to 'or' "
        f"makes is_in_job return True when the process is not in a job."
    )
    assert fake_api.call_count == 1, "IsProcessInJob was never invoked"


import os
import subprocess


def test_spawn_in_job_assigns_to_job_before_resuming_threads(monkeypatch):
    """Ordering seam: ``spawn_in_job`` must (1) pass CREATE_SUSPENDED
    to Popen, (2) assign the suspended child to the job, and only THEN
    (3) resume its threads. Any other order re-opens the window in
    which a pipeline killed mid-spawn (taskkill /F) leaves a running
    overlay that is not yet a member of the inner job -- it survives
    KILL_ON_JOB_CLOSE and turns into a duplicate-overlay zombie.

    Fails red if: CREATE_SUSPENDED is dropped from the creationflags,
    or the resume happens before the assign, or the assign is skipped.

    Seeded-defect drill: swap the ``job.assign(proc)`` and
    ``_resume_process_threads(proc.pid)`` lines in spawn_in_job ->
    the order assertion goes red.
    """
    from unittest.mock import MagicMock
    import yamabiko.lifecycle.jobs as jobs_mod

    order: list = []
    popen_kwargs: dict = {}

    fake_proc = MagicMock()
    fake_proc.pid = 4242

    def fake_popen(cmd, **kwargs):
        popen_kwargs.update(kwargs)
        order.append("popen")
        return fake_proc

    fake_job = MagicMock()
    fake_job.assign = MagicMock(
        side_effect=lambda proc: order.append(("assign", proc.pid)) or True,
    )

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        jobs_mod, "_resume_process_threads",
        lambda pid: order.append(("resume", pid)) or 1,
    )

    proc = jobs_mod.spawn_in_job(["dummy-cmd"], fake_job)

    assert proc is fake_proc
    assert popen_kwargs.get("creationflags", 0) & jobs_mod.CREATE_SUSPENDED, (
        f"spawn_in_job did not pass CREATE_SUSPENDED to Popen "
        f"(creationflags={popen_kwargs.get('creationflags', 0):#x}); the "
        f"child starts executing before job assignment -- the race "
        f"window is back."
    )
    assert order == ["popen", ("assign", 4242), ("resume", 4242)], (
        f"spawn_in_job call order was {order}; required "
        f"popen -> assign -> resume. Resuming before assigning lets the "
        f"child run instructions outside the job."
    )


@pytest.mark.real
def test_spawn_in_job_child_joins_job_before_running_and_dies_on_job_close(tmp_path):
    """End-to-end on the real Win32 path: a child spawned via
    ``spawn_in_job`` (a) is a member of the target job while alive,
    (b) actually executes (its threads were resumed -- proven by the
    marker file it writes), and (c) dies when the job handle closes
    (KILL_ON_JOB_CLOSE cascade through the membership established
    before its first instruction).

    Fails red if: the assign step is skipped (membership probe False,
    close() doesn't kill), or the resume step is skipped (marker file
    never appears), or CREATE_SUSPENDED handling leaves the child
    permanently suspended.

    Seeded-defect drill: comment out the ``job.assign(proc)`` call in
    spawn_in_job -> IsProcessInJob returns FALSE and the post-close
    wait times out -> red.
    """
    import time
    from ctypes import wintypes
    import yamabiko.lifecycle.jobs as jobs_mod

    marker = tmp_path / "f75-ran.txt"
    code = (
        "import pathlib, time; "
        f"pathlib.Path(r'{marker}').write_text('RAN'); "
        "time.sleep(60)"
    )
    job = jobs_mod.create_inner_job("yamabiko-test-f75-spawn")
    assert job.handle is not None, "precondition: job creation failed"
    proc = None
    try:
        proc = jobs_mod.spawn_in_job(
            [sys.executable, "-c", code], job,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # (b) resumed: the child reaches its first instruction and
        # writes the marker. A still-suspended child never does.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists(), (
            "child never executed within 10 s -- spawn_in_job left it "
            "CREATE_SUSPENDED (resume step broken)"
        )
        # (a) membership while alive, against THIS job specifically.
        assert proc.poll() is None, "sentinel exited prematurely"
        out = wintypes.BOOL(False)
        ok = jobs_mod._IsProcessInJob(
            int(proc._handle), job.handle, ctypes.byref(out),
        )
        assert ok, "IsProcessInJob probe call failed"
        assert out.value, (
            "child is NOT a member of the job it was spawned into; "
            "spawn_in_job's assign step is broken"
        )
        # (c) kernel cascade through that membership.
        job.close()
        rc = proc.wait(timeout=5.0)
        assert rc is not None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        job.close()


@pytest.mark.real
def test_job_kills_child_when_parent_job_handle_closes(tmp_path):
    """Fails red if: KILL_ON_JOB_CLOSE is not set, or assign()
    doesn't actually place the child in the job."""
    job = JobObject(name="yamabiko-test-kill-cascade")
    try:
        # Spawn a child that would otherwise live forever.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            assert job.assign(proc) is True
            # Close the job handle -> kernel kills assigned children.
            job.close()
            rc = proc.wait(timeout=5.0)
            # Exit code is some nonzero on kill; we just need it dead.
            assert rc is not None
        finally:
            if proc.poll() is None:
                proc.kill()
    finally:
        job.close()


@pytest.mark.real
def test_nested_job_assignment_places_child_in_both_jobs():
    """Fails red if: inner-job assignment either fails (UI limits
    set somewhere) or doesn't form a nest with the outer job.

    The assignment has to happen in a helper subprocess rather than
    the pytest process directly: once pytest is assigned to a job
    with KILL_ON_JOB_CLOSE, the test's finally-clause that closes
    those jobs would terminate pytest itself (it's the last-handle
    rule: closing the sole handle to a KILL_ON_JOB_CLOSE job kills
    all member processes). The helper performs both assignments,
    prints its is_in_job state, and exits 0 on success. If inner-
    assignment failed (e.g. UI limits block nesting) the helper
    exits non-zero and the test goes red.
    """
    helper = (
        "import sys, ctypes;"
        "sys.path.insert(0, r'" + os.getcwd() + "');"
        "from yamabiko.lifecycle.jobs import "
        "JobObject, is_in_job;"
        "outer = JobObject(name='yamabiko-test-nest-outer');"
        "inner = JobObject(name='yamabiko-test-nest-inner');"
        "assert outer.handle is not None, 'outer handle None';"
        "assert inner.handle is not None, 'inner handle None';"
        "assert outer.assign(None) is True, 'outer assign failed';"
        "assert inner.assign(None) is True, 'inner assign failed';"
        "assert is_in_job() is True, 'is_in_job returned False after nest';"
        "print('OK', flush=True);"
        "sys.exit(0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", helper],
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert proc.returncode == 0, (
        f"helper exited {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "OK" in proc.stdout


def test_show_in_folder_breaks_away_and_falls_back_inside_the_job(monkeypatch, tmp_path):
    """Fails red if: Explorer is started without CREATE_BREAKAWAY_FROM_JOB
    (a pipeline restart or Quit would close the window, Decision 35), a job
    that forbids breakaway (PermissionError) gets no window at all, or the
    command is a list (Python quotes "/select,<path with spaces>" as one
    token and Explorer opens Documents instead)."""
    import yamabiko.lifecycle.jobs as jobs_mod

    calls: list[tuple] = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs.get("creationflags", 0)))
        if len(calls) == 1:
            raise PermissionError(5, "Access is denied")
        return object()

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", fake_popen)
    target = tmp_path / "with space" / "yamabiko-diagnostics.zip"
    jobs_mod.show_in_folder_outside_job(target)
    expected = f'explorer.exe /select,"{os.path.abspath(target)}"'
    assert calls == [(expected, jobs_mod.CREATE_BREAKAWAY_FROM_JOB), (expected, 0)]
