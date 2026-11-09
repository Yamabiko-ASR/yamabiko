"""Real-path coverage for the ``keyboard`` library registration seam.

``tests/test_hotkeys.py`` uses a FakeBackend that only verifies
our own dispatch logic -- it never exercises the real
``keyboard.add_hotkey`` / ``keyboard.clear_all_hotkeys`` API surface,
which is what ``yamabiko.hotkeys.HotkeyManager`` actually calls in
production.

The first test covers registration + cleanup: it catches API rename,
signature drift, and module-level init failure from a broken install.
The second fires hotkeys with injected key presses (skipped when the
session has no interactive input desktop) to prove a live re-bind: after
unregister() + register() with a new combo, the new combo fires and
the old one does not. The combos (Ctrl+Alt+Shift+F19 and F20) are
ones no app reacts to.

SUBPROCESS ISOLATION:
the first ``add_hotkey`` permanently installs a WH_KEYBOARD_LL hook on
a library daemon thread. NO public API removes it --
``clear_all_hotkeys`` and ``unhook_all`` clear Python handler
registrations only. From that moment, EVERY exit of the host process
is a coin flip: a keystroke whose kernel->user callback dispatch
coincides with interpreter finalization hits the ctypes trampoline of
a dead interpreter -> the process exits 0xC000041D
(STATUS_FATAL_USER_CALLBACK_EXCEPTION) AFTER pytest prints its
summary, failing preflight section 8 while all tests show green.
Running the registration in a child process keeps the hook out of the
pytest process entirely; the child's own exit is allowed to be 0 or
0xC000041D (the library teardown race, benign there), while API-drift
defects (TypeError/AttributeError/ValueError -> traceback, other
nonzero exit, missing marker) still FAIL the test.

The same hook lives in the PRODUCTION pipeline process via
yamabiko.hotkeys; the pipeline's exit code can be corrupted the same
way.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.real

_STATUS_FATAL_USER_CALLBACK_EXCEPTION = 0xC000041D

# The child exercises the exact production API surface
# (HotkeyManager.register -> add_hotkey, unregister -> clear_all_hotkeys)
# and prints a marker only if BOTH calls succeed. Environment-only
# failures (restricted session, missing driver access) print a SKIP
# marker; API-drift defect classes propagate as a traceback + exit 1.
_CHILD = r"""
import sys
import keyboard

assert hasattr(keyboard, "add_hotkey"), "keyboard.add_hotkey missing (API rename)"
combo = "ctrl+alt+shift+f19"
try:
    keyboard.add_hotkey(combo, lambda: None, suppress=False)
except (OSError, PermissionError, ImportError) as exc:
    print(f"KEYBOARD_REAL_SKIP: {type(exc).__name__}: {exc}")
    sys.exit(0)
keyboard.clear_all_hotkeys()
print("KEYBOARD_REAL_OK")
sys.stdout.flush()
"""


def test_keyboard_add_hotkey_registers_and_clears() -> None:
    """``add_hotkey`` accepts our combo-string format; ``clear_all_hotkeys`` unregisters.

    Runs in a subprocess so the keyboard lib's permanent LL hook never
    enters the pytest process (see module docstring). Fails red on
    API rename (child AssertionError/AttributeError),
    signature drift (TypeError), combo-parser drift (ValueError) -- all
    of which exit the child with a traceback and WITHOUT the OK marker.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if "KEYBOARD_REAL_SKIP" in proc.stdout:
        pytest.skip(f"keyboard backend unavailable: {proc.stdout.strip()}")

    # Signed/unsigned views of the same NTSTATUS, depending on how the
    # platform reports it.
    benign_teardown = {
        0,
        _STATUS_FATAL_USER_CALLBACK_EXCEPTION,
        _STATUS_FATAL_USER_CALLBACK_EXCEPTION - 2**32,
    }
    assert "KEYBOARD_REAL_OK" in proc.stdout, (
        f"keyboard registration child never reached the OK marker -- "
        f"API drift or module-level init failure.\n"
        f"exit={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert proc.returncode in benign_teardown, (
        f"keyboard child exited {proc.returncode} (not clean and not the "
        f"documented 0xC000041D LL-hook teardown race) despite printing "
        f"the OK marker.\nstderr={proc.stderr!r}"
    )


_REBIND_CHILD = r"""
import ctypes, sys, threading, time
from yamabiko.hotkeys import HotkeyManager

user32 = ctypes.windll.user32

def press(vks):
    for vk in vks:
        user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), 0, 0)
    for vk in reversed(vks):
        user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), 2, 0)

# Injected keys reach this session's hook only while its desktop takes
# input: a locked workstation or a non-interactive session has no input
# desktop this process can open.
user32.OpenInputDesktop.restype = ctypes.c_void_p
user32.CloseDesktop.argtypes = [ctypes.c_void_p]
desktop = user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
if not desktop:
    print("KEYBOARD_REAL_SKIP: no interactive input desktop")
    sys.exit(0)
user32.CloseDesktop(desktop)

old, new = threading.Event(), threading.Event()
manager = HotkeyManager(probe=lambda combo: False)
try:
    manager.register(pause="Ctrl+Alt+Shift+F19", toggle_overlay="", show_series="",
                     on_pause=old.set, on_toggle_overlay=lambda: None, on_show_series=lambda: None)
except (OSError, PermissionError, ImportError) as exc:
    print(f"KEYBOARD_REAL_SKIP: {type(exc).__name__}: {exc}")
    sys.exit(0)
time.sleep(1.5)  # the library's hook thread installs asynchronously
manager.unregister()
manager.register(pause="Ctrl+Alt+Shift+F20", toggle_overlay="", show_series="",
                 on_pause=new.set, on_toggle_overlay=lambda: None, on_show_series=lambda: None)
time.sleep(0.3)
press((0x11, 0x12, 0x10, 0x82))  # Ctrl+Alt+Shift+F19 (the old binding)
time.sleep(0.5)
press((0x11, 0x12, 0x10, 0x83))  # Ctrl+Alt+Shift+F20 (the new one)
if not new.wait(3.0):
    print("KEYBOARD_REAL_NEW_DID_NOT_FIRE")
    sys.exit(1)
if old.is_set():
    print("KEYBOARD_REAL_OLD_STILL_FIRES")
    sys.exit(1)
manager.unregister()
print("KEYBOARD_REAL_REBIND_OK")
sys.stdout.flush()
"""


def test_keyboard_rebinds_after_unregister() -> None:
    """Settings re-binds live: unregister() then register() with new combos
    must make the new combo fire and the old one stop, through the real
    keyboard hook (injected key presses with scan codes; the library matches
    hotkeys by scan code).

    Seeded defects: skip the unregister in PipelineApp.apply_hotkeys (here:
    in the child) -> the old combo still fires -> red; a backend whose
    clear_all_hotkeys leaves the hook dead -> the new combo never fires ->
    red."""
    import os

    env = dict(os.environ, PYTHONPATH=os.getcwd())
    proc = subprocess.run([sys.executable, "-c", _REBIND_CHILD], capture_output=True,
                          text=True, timeout=60, env=env)
    if "KEYBOARD_REAL_SKIP" in proc.stdout:
        pytest.skip(f"keyboard backend unavailable: {proc.stdout.strip()}")
    benign_teardown = {0, _STATUS_FATAL_USER_CALLBACK_EXCEPTION,
                       _STATUS_FATAL_USER_CALLBACK_EXCEPTION - 2**32}
    assert "KEYBOARD_REAL_REBIND_OK" in proc.stdout, (
        f"exit={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}")
    assert proc.returncode in benign_teardown
