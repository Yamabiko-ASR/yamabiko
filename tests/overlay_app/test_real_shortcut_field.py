"""Real Windows platform: native key presses record a shortcut
(@pytest.mark.real).

On the native windows QPA a recording field takes the main key in its
WH_KEYBOARD_LL hook (SetWindowsHookExW, GetForegroundWindow, GetAsyncKeyState
and CallNextHookEx: Windows hands a combo another app holds to that app, so
its main key never reaches Qt); without the hook (the fallback when it can't
install) QKeyEvent.key / modifiers / nativeVirtualKey from a real WM_KEYDOWN
record it. The offscreen QPA has neither. SKIPS unless the native windows
QPA is up and the test window can take the foreground. Run in its own
invocation."""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes

import pytest

pytestmark = pytest.mark.real

_CTRL, _ALT, _SHIFT = 0x11, 0x12, 0x10
_VK_1, _VK_F20 = 0x31, 0x83
_MOD_CTRL_ALT_SHIFT_NOREPEAT = 0x0002 | 0x0001 | 0x0004 | 0x4000
_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001


@pytest.fixture(scope="module")
def real_app():
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None and sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    app = QApplication.instance() or QApplication([])
    if app.platformName() != "windows":
        pytest.skip("native windows QPA unavailable; run this file in its own invocation")
    yield app


def _foreground_field():
    """A shown field and a private user32 handle, or skip.

    Qt's isActiveWindow() is no proof: under Windows' foreground lock Qt
    reports the window active while another app keeps the foreground, and
    the injected keys would type into that app. Only the real foreground
    window counts."""
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.shortcut_field import ShortcutField
    from yamabiko.overlay_app.theme import MIST

    field = ShortcutField("Ctrl+Alt+P", MIST)
    field.move(200, 200)
    field.show()
    field.raise_()
    field.activateWindow()
    QTest.qWait(400)
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetForegroundWindow.restype = wintypes.HWND
    if user32.GetForegroundWindow() != int(field.winId()):
        field.close()
        pytest.skip("this session cannot bring a test window to the foreground")
    return field, user32


def _press(user32, keys) -> None:
    """Key-downs in order, then key-ups in reverse: nothing stays pressed."""
    for vk in keys:
        user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), 0, 0)
    for vk in reversed(keys):
        user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), 2, 0)


def _pump(ms: int) -> None:
    """Run Qt's event loop for ``ms`` in short slices with a sleep between
    them: QTest.qWait holds the GIL, so one long wait starves the holder
    thread and its count lags behind the hotkeys Windows posted."""
    from PySide6.QtTest import QTest

    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        QTest.qWait(10)
        time.sleep(0.005)


class _Holder(threading.Thread):
    """Another app's shortcut: holds Ctrl+Alt+Shift+F20 with RegisterHotKey on
    its own thread and counts the WM_HOTKEY messages Windows posts for it."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.held = threading.Event()
        self.release = threading.Event()
        self.registered = False
        self.hits = 0

    def run(self) -> None:
        user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
        user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT,
                                          wintypes.UINT)
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.UnregisterHotKey.restype = wintypes.BOOL
        user32.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                        wintypes.UINT, wintypes.UINT, wintypes.UINT)
        user32.PeekMessageW.restype = wintypes.BOOL
        self.registered = bool(user32.RegisterHotKey(None, 1, _MOD_CTRL_ALT_SHIFT_NOREPEAT,
                                                     _VK_F20))
        self.held.set()
        if not self.registered:
            return
        msg = wintypes.MSG()
        try:
            while not self.release.wait(0.01):
                while user32.PeekMessageW(ctypes.byref(msg), None, _WM_HOTKEY, _WM_HOTKEY,
                                          _PM_REMOVE):
                    self.hits += 1
        finally:
            user32.UnregisterHotKey(None, 1)


def test_real_native_key_press_records_the_shortcut(real_app, monkeypatch):
    """The key-event fallback: with the hook off, as when it can't install,
    the real QKeyEvent records. Seeded defects: read Qt's key (Shift+1 is
    "!") instead of event.nativeVirtualKey(), or drop the modifiers -> the
    field keeps recording or records the wrong combo -> red."""
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app import shortcut_field

    monkeypatch.setattr(shortcut_field._KeyDownHook, "install", lambda self: None)

    field, user32 = _foreground_field()
    recorded = []
    field.recorded.connect(recorded.append)
    try:
        field.setFocus()
        field.start_recording()
        _press(user32, (_CTRL, _ALT, _SHIFT, _VK_1))
        QTest.qWait(400)
        assert recorded == ["Ctrl+Alt+Shift+1"]
    finally:
        field.close()


def test_real_recording_takes_a_combo_another_app_holds(real_app):
    """Windows hands a held combo to its holder and posts only the modifiers
    to the window. Seeded defects: no hook while recording (nothing is
    recorded), the hook passes the key on (the holder's shortcut fires), or
    the hook stays installed after recording (the keys never reach the
    holder again) -> red."""
    from yamabiko import hotkeys

    field, user32 = _foreground_field()
    holder = _Holder()
    holder.start()
    recorded = []
    field.recorded.connect(recorded.append)
    try:
        assert holder.held.wait(5) and holder.registered
        assert hotkeys.combo_taken("Ctrl+Alt+Shift+F20")
        field.setFocus()
        field.start_recording()
        _press(user32, (_CTRL, _ALT, _SHIFT, _VK_F20))
        _pump(400)
        assert recorded == ["Ctrl+Alt+Shift+F20"] and holder.hits == 0
        # Recording ended and the hook is gone: the keys reach the holder again.
        _press(user32, (_CTRL, _ALT, _SHIFT, _VK_F20))
        _pump(400)
        assert holder.hits == 1 and recorded == ["Ctrl+Alt+Shift+F20"]
    finally:
        holder.release.set()
        holder.join(5)
        field.close()
