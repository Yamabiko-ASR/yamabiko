"""Real-tier crash-evidence test (logging-hardening Task 1, hard rule #3).

install_crash_evidence's faulthandler layer is an external-dep seam
twice over: CPython's native fault handling (a real OS access violation,
not a Python exception) and the explicit-file-handle contract that the
windowed PyInstaller bundle forces (default sys.stderr is a NullWriter
there -- pyinstaller#4481). No mock can verify either, so this test
spawns a REAL child interpreter, arms the real faulthandler through the
production helper, dies on a REAL access violation via ctypes, and reads
the crash log the way a human triaging a beta report would.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.real

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The child arms crash evidence exactly like a production entry point,
# then reads from address 0. ctypes.string_at(0) is a native fault: it
# bypasses Python exception handling entirely, so ONLY faulthandler can
# leave evidence.
_CHILD_SRC = """\
import ctypes
from yamabiko import log_config
log_config.install_crash_evidence("probe")
ctypes.string_at(0)
"""

# Same crash, but with a second thread alive and parked in a
# distinctively-named frame at fault time. Its presence/absence in the
# crash log is the observable contract for the all_threads policy.
_THREADED_CHILD_SRC = """\
import ctypes
import threading
import time
from yamabiko import log_config

parked = threading.Event()

def background_marker_thread_fn():
    parked.set()
    time.sleep(60)

threading.Thread(target=background_marker_thread_fn, daemon=True).start()
assert parked.wait(10), "marker thread never parked"
log_config.install_crash_evidence("probe")
ctypes.string_at(0)
"""


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="asserts the Windows-specific faulthandler banner text",
)
def test_real_access_violation_child_writes_stack_to_crash_probe_log(
    tmp_path,
):
    """A child process that installs crash evidence and then hits a
    native access violation must (a) die nonzero and (b) leave
    crash-probe.log containing the Windows fatal-exception banner plus a
    thread stack naming the faulting frame.

    One-line seeded defect: drop the ``faulthandler.enable(file=fh)``
    call in install_crash_evidence -- the child still dies nonzero but
    crash-probe.log stays empty, red.
    """
    env = os.environ.copy()
    env["YAMABIKO_APPDATA"] = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SRC],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=str(_REPO_ROOT),  # so `import yamabiko` resolves for -c
    )

    assert proc.returncode != 0, (
        f"child survived ctypes.string_at(0)? rc={proc.returncode}, "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )

    crash_log = tmp_path / "logs" / "crash-probe.log"
    assert crash_log.is_file(), (
        f"crash-probe.log missing; logs dir contains "
        f"{[p.name for p in (tmp_path / 'logs').iterdir()] if (tmp_path / 'logs').is_dir() else 'nothing'}; "
        f"child stderr={proc.stderr!r}"
    )
    content = crash_log.read_text(encoding="utf-8", errors="replace")
    assert "Windows fatal exception: access violation" in content, (
        f"no access-violation banner in crash-probe.log: {content!r}"
    )
    # Single-thread dump header (all_threads=False prints "Stack",
    # not "Current thread 0x...").
    assert "Stack (most recent call first):" in content, (
        f"no thread-stack header in crash-probe.log: {content!r}"
    )
    assert 'File "<string>"' in content, (
        f"the faulting frame (the -c child's <string> module) is not in "
        f"the stack: {content!r}"
    )


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="asserts the Windows-specific faulthandler banner text",
)
def test_real_access_violation_dump_names_only_the_faulting_thread(
    tmp_path,
):
    """The crash dump must contain the faulting thread's stack and NOT
    walk other live threads.

    faulthandler's Windows layer is a vectored exception handler that
    fires on FIRST-CHANCE exceptions, including benign Qt/COM-handled
    ones (0x8001010D during window activation, PYSIDE-2501). With the
    default ``all_threads=True`` the handler then walks every RUNNING
    thread's frames -- a documented race (python/cpython#128400) that
    access-violated inside the handler and crash-looped the overlay
    with 0xC000041D on 2026-07-02. The production contract is therefore
    ``all_threads=False``: dump the faulting thread only.

    One-line seeded defect: drop ``all_threads=False`` from the
    ``faulthandler.enable`` call in install_crash_evidence -- the parked
    background thread's ``background_marker_thread_fn`` frame appears in
    crash-probe.log, red.
    """
    env = os.environ.copy()
    env["YAMABIKO_APPDATA"] = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, "-c", _THREADED_CHILD_SRC],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=str(_REPO_ROOT),  # so `import yamabiko` resolves for -c
    )

    assert proc.returncode != 0, (
        f"child survived ctypes.string_at(0)? rc={proc.returncode}, "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )

    crash_log = tmp_path / "logs" / "crash-probe.log"
    assert crash_log.is_file(), (
        f"crash-probe.log missing; logs dir contains "
        f"{[p.name for p in (tmp_path / 'logs').iterdir()] if (tmp_path / 'logs').is_dir() else 'nothing'}; "
        f"child stderr={proc.stderr!r}"
    )
    content = crash_log.read_text(encoding="utf-8", errors="replace")
    assert "Windows fatal exception: access violation" in content, (
        f"no access-violation banner in crash-probe.log: {content!r}"
    )
    assert "Stack (most recent call first):" in content, (
        f"faulting thread's stack missing -- the fix must not cost the "
        f"actual crash evidence: {content!r}"
    )
    assert 'File "<string>"' in content, (
        f"the faulting frame (the -c child's <string> module) is not in "
        f"the stack: {content!r}"
    )
    assert "background_marker_thread_fn" not in content, (
        f"crash dump walked a live non-faulting thread; faulthandler "
        f"must be armed with all_threads=False (see PYSIDE-2501 / "
        f"cpython#128400): {content!r}"
    )
