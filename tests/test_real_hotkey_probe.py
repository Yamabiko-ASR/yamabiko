"""Real Win32 RegisterHotKey probe (@pytest.mark.real).

combo_taken is how Settings and the pipeline tell "used by another app":
another thread holds the combo through the same API other apps use, and the
probe must see it, then see it free once released."""
from __future__ import annotations

import ctypes
import sys
import threading

import pytest

from yamabiko import hotkeys

pytestmark = [pytest.mark.real, pytest.mark.skipif(sys.platform != "win32", reason="Win32 API")]

_COMBO = "Ctrl+Alt+Shift+F20"
_MODS = 0x0001 | 0x0002 | 0x0004  # Alt | Ctrl | Shift
_VK_F20 = 0x83


def test_real_combo_taken_sees_a_hotkey_another_thread_holds():
    """Seeded defects: compare GetLastError against the wrong code, or leak
    the probe's own registration (the second probe then reports taken) ->
    red."""
    held = threading.Event()
    release = threading.Event()
    outcome = {}

    def holder():
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        outcome["registered"] = bool(user32.RegisterHotKey(None, 0x2F2F, _MODS, _VK_F20))
        held.set()
        release.wait(10)
        user32.UnregisterHotKey(None, 0x2F2F)

    if hotkeys.combo_taken(_COMBO):
        pytest.skip(f"{_COMBO} is already held on this machine")
    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(5) and outcome["registered"]
        assert hotkeys.combo_taken(_COMBO) is True
    finally:
        release.set()
        thread.join(5)
    assert hotkeys.combo_taken(_COMBO) is False
    assert hotkeys.combo_taken(_COMBO) is False
