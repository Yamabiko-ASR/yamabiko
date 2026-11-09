"""Another process's window in the foreground, for the real GUI tests.

The foreground owner is a separate process (the video player the user
watches) whose window takes the foreground the way a user gives it, with one
real click on itself. That click is the last input, so this process has no
right to the foreground, as in the product when the input that opens a
Yamabiko window reaches another process (Ctrl+Alt+S reaches the pipeline's
keyboard hook; a click on a notification reaches the shell).
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

import pytest

_OWNER_TITLE = "Yamabiko test: foreground owner"
_OWNER_CLICKED = " (clicked)"

# The owner clicks its own window, puts the cursor back, then adds
# _OWNER_CLICKED to its title (its input is done) and waits until the test
# ends it.
_OWNER_SCRIPT = r'''
import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication([])
owner = QWidget()
owner.setWindowTitle(sys.argv[1])
owner.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
owner.setGeometry(160, 360, 360, 200)
owner.show()

user32 = ctypes.WinDLL("user32")
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
user32.mouse_event.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                               wintypes.DWORD, ctypes.c_size_t)


def click_self():
    rect = wintypes.RECT()
    user32.GetWindowRect(int(owner.winId()), ctypes.byref(rect))
    saved = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(saved))
    user32.SetCursorPos((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP

    def done():
        user32.SetCursorPos(saved.x, saved.y)
        owner.setWindowTitle(sys.argv[1] + " (clicked)")

    QTimer.singleShot(150, done)


QTimer.singleShot(600, click_self)
app.exec()
'''


def wait_for(predicate, timeout_ms: int):
    """predicate()'s first truthy value within timeout_ms while Qt events
    keep flowing, else None."""
    from PySide6.QtTest import QTest

    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            return None
        QTest.qWait(20)


def _user32():
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    return user32


def foreground_hwnd() -> int:
    """The foreground window's handle, 0 for none."""
    return _user32().GetForegroundWindow() or 0


def _find_window(title: str) -> int:
    return _user32().FindWindowW(None, title) or 0


@contextlib.contextmanager
def foreground_owner(title: str = _OWNER_TITLE):
    """The foreground owner (_OWNER_SCRIPT, its window titled ``title``)
    running with its window in the foreground, as the user's click on it
    leaves it: yields that window's handle once its own click is done, and
    ends the process afterwards. Windows may give its window the foreground
    as soon as it shows (this process started it); its click 600 ms later
    would then take the foreground back from a window shown in between (fix
    pass 3b). Fails when its window never shows or it never clicks; skips
    only when this session never gives it the foreground (a session blocker,
    never a pass)."""
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER_SCRIPT, title],
        env={**os.environ, "QT_QPA_PLATFORM": "windows"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        owner_hwnd = wait_for(lambda: _find_window(title), 20000)
        if not owner_hwnd or not wait_for(
                lambda: _find_window(title + _OWNER_CLICKED), 10000):
            owner.kill()
            stderr = owner.communicate(timeout=10)[1] or ""
            pytest.fail(f"the foreground owner never showed its window or clicked it: "
                        f"{stderr[-800:]}")
        if not wait_for(lambda: foreground_hwnd() == owner_hwnd, 10000):
            pytest.skip("this session did not give the helper's window the foreground")
        yield owner_hwnd
    finally:
        owner.terminate()
        try:
            owner.wait(10)
        except subprocess.TimeoutExpired:
            owner.kill()
            owner.wait(10)
