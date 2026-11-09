"""Real Windows platform: the startup error window (@pytest.mark.real).

Copy report's text must survive the pipeline's os._exit(43): Qt renders
clipboard text on request (OLE delayed rendering), so without
OleFlushClipboard the text left with the process (probe,
errors-research-flow.md headline 1). The window is built in a child process
on the native windows QPA; this process reads the clipboard through Win32
only. Runs in the real-gui phase:

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_error_window.py -m real -v

The user's clipboard text is put back after each test.

The tests after the child-process one build the window in this process on
the native windows QPA (run_error_window, as show_startup_error calls it):
placement on the primary work area (DWM frame bounds, GetMonitorInfoW), the
foreground taken from another process (SendInput + SetForegroundWindow),
Copy report on the OS clipboard (QClipboard + OleFlushClipboard, read back
through Win32) with the log tail of a real RotatingFileHandler, and Open
Settings opening a real SettingsWindow as a top-level window of its own that
a click injected through Windows reaches while the error window is up
(nothing modal blocks it), the call returning only after both windows
close. They SKIP unless the native windows QPA is available (run this file
in its own invocation):

    .venv\\Scripts\\python.exe -m pytest tests/overlay_app/test_real_error_window.py -m real -v
"""
from __future__ import annotations

import contextlib
import ctypes
import getpass
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from ctypes import wintypes
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from tests.overlay_app._dialog_wait import when
from tests.overlay_app._qt_drain import drain_qt_debris

pytestmark = pytest.mark.real

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
_REPO = Path(__file__).resolve().parents[2]

# The child: the real window and its real Copy report handler on the native
# platform, then the pipeline's exit (os._exit(43), no finalization).
_CHILD = r"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "windows"
sys.path.insert(0, sys.argv[1])
from yamabiko.ui.qt_bootstrap import ensure_qapplication
app = ensure_qapplication()
from yamabiko.overlay_app.theme import MIST
from yamabiko.ui.error_dialog import StartupError
from yamabiko.ui.error_report import LogTail, ReportFacts
from yamabiko.ui.error_window import ErrorWindow
err = StartupError.models_missing(details="FileNotFoundError: " + sys.argv[2])
facts = ReportFacts("0.0.0-test", "11 26H2, build 1.1", "None reported by Windows", "3.12.0",
                    "FileNotFoundError: " + sys.argv[2], LogTail(None, []))
window = ErrorWindow(err, facts, MIST)
window.copy_button.click()
print("copied" if window.copied() else "not copied", flush=True)
os._exit(43)
"""


def _clipboard_api():
    user32 = ctypes.WinDLL("user32")  # private handles: argtypes stay local
    kernel32 = ctypes.WinDLL("kernel32")
    user32.OpenClipboard.argtypes = (wintypes.HWND,)
    user32.GetClipboardData.argtypes = (wintypes.UINT,)
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    return user32, kernel32


def _open_clipboard(user32) -> None:
    for _ in range(50):  # another process may hold the clipboard briefly
        if user32.OpenClipboard(None):
            return
        time.sleep(0.02)
    pytest.fail("the clipboard stayed open in another process")


def _native_clipboard_text() -> str:
    """CF_UNICODETEXT straight from the Win32 clipboard."""
    user32, kernel32 = _clipboard_api()
    _open_clipboard(user32)
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return ""
        pointer = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _set_native_clipboard_text(text: str) -> None:
    user32, kernel32 = _clipboard_api()
    data = ctypes.create_unicode_buffer(text)
    handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, ctypes.sizeof(data))
    pointer = kernel32.GlobalLock(handle)
    ctypes.memmove(pointer, data, ctypes.sizeof(data))
    kernel32.GlobalUnlock(handle)
    _open_clipboard(user32)
    try:
        user32.EmptyClipboard()
        user32.SetClipboardData(_CF_UNICODETEXT, handle)  # the clipboard owns it now
    finally:
        user32.CloseClipboard()


@pytest.fixture
def saved_clipboard():
    """The user's clipboard text, put back after the test (text only)."""
    saved = _native_clipboard_text()
    yield
    _set_native_clipboard_text(saved)


def test_real_copied_report_survives_the_pipelines_os_exit(saved_clipboard):
    """L10: a child builds the real window, clicks Copy report and leaves
    through os._exit(43) like the pipeline; the report is still on the
    native clipboard afterwards.

    Seeded defect: drop the OleFlushClipboard call from copy_to_clipboard ->
    the clipboard holds only Qt's OLE data object, no CF_UNICODETEXT, once
    the child is gone -> red."""
    token = f"yb-{uuid.uuid4().hex}.gguf"
    child = subprocess.run([sys.executable, "-c", _CHILD, str(_REPO), token],
                           capture_output=True, text=True, timeout=60)
    assert child.returncode == 43, child.stderr
    assert child.stdout.strip() == "copied", child.stderr
    text = _native_clipboard_text().replace("\r\n", "\n")  # Qt writes Windows line ends
    assert text.startswith("Yamabiko startup error\n"), text[:200]
    assert f"Error: FileNotFoundError: {token}" in text


# ---------------------------------------------------------------------------
# The window in this process on the native platform (run_error_window, as
# show_startup_error calls it in the pipeline).
# ---------------------------------------------------------------------------

_COPY_SENTINEL = "yamabiko test: the report was not copied"
_OWNER_TITLE = "Yamabiko test: error window foreground owner"
_OWNER_CLICKED = " (clicked)"

# The foreground owner: a separate process (the app the user works in while
# the pipeline fails in the background) whose window takes the foreground the
# way a user gives it, with one real click on itself. It puts the cursor back,
# adds _OWNER_CLICKED to its title (its input is done) and waits until the
# test ends it. The same helper as test_real_series_palette.py.
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


@pytest.fixture(scope="module")
def real_app():
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None and sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    app = QApplication.instance() or QApplication([])
    if app.platformName() != "windows":
        pytest.skip("native windows QPA unavailable; run this file in its own invocation")
    yield app


@pytest.fixture
def own_app_data(monkeypatch, tmp_path):
    """Real tier: the app-data isolation exempts it. The window reads its
    theme (user_overrides.yaml) and its log fallback, and Settings its
    config, from this test's folder, never the user's."""
    root = tmp_path / "Yamabiko"
    monkeypatch.setenv("YAMABIKO_APPDATA", str(root))
    return root


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def _user32():
    user32 = ctypes.WinDLL("user32")  # private handle: argtypes stay local
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetDesktopWindow.restype = wintypes.HWND
    user32.ChildWindowFromPointEx.argtypes = (wintypes.HWND, wintypes.POINT, wintypes.UINT)
    user32.ChildWindowFromPointEx.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
    user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    user32.mouse_event.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.DWORD, ctypes.c_size_t)
    user32.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
    user32.MonitorFromPoint.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo))
    return user32


def _foreground_hwnd() -> int:
    return _user32().GetForegroundWindow() or 0


def _find_window(title: str) -> int:
    return _user32().FindWindowW(None, title) or 0


def _visible_bounds(widget) -> wintypes.RECT:
    """The visible window rect (DWMWA_EXTENDED_FRAME_BOUNDS, the DWM border
    included, the shadow not), device px."""
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        ctypes.c_void_p(int(widget.winId())), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    return rect


def _primary_work_area() -> wintypes.RECT:
    """The primary monitor's work area (rcWork), device px. The primary
    monitor holds the desktop origin."""
    user32 = _user32()
    info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    user32.GetMonitorInfoW(user32.MonitorFromPoint(wintypes.POINT(0, 0), 1),  # DEFAULTTOPRIMARY
                           ctypes.byref(info))
    return info.rcWork


def _cursor_pos() -> wintypes.POINT:
    point = wintypes.POINT()
    _user32().GetCursorPos(ctypes.byref(point))
    return point


def _native_center(widget) -> tuple[int, int]:
    """``widget``'s center on screen in device px: its window's Win32 client
    origin (an expanded client area starts at the visible top-left) plus the
    widget's offset at the window's pixel ratio."""
    window = widget.window()
    origin = wintypes.POINT(0, 0)
    _user32().ClientToScreen(int(window.winId()), ctypes.byref(origin))
    center = widget.mapTo(window, widget.rect().center())
    ratio = window.devicePixelRatioF()
    return origin.x + round(center.x() * ratio), origin.y + round(center.y() * ratio)


def _top_window_at(x: int, y: int) -> int:
    """The topmost visible top-level window at the screen point (x, y),
    device px, disabled ones included: WindowFromPoint skips a window a
    modal dialog disabled, which would read as covered, not as blocked."""
    user32 = _user32()
    return user32.ChildWindowFromPointEx(
        user32.GetDesktopWindow(), wintypes.POINT(x, y),
        0x0001 | 0x0004,  # CWP_SKIPINVISIBLE | CWP_SKIPTRANSPARENT
    ) or 0


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    _user32().GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _real_click(x: int, y: int) -> None:
    """One left click injected through Windows at (x, y), device px: the
    path a user's click takes, which Qt's modality blocks for windows a
    modal dialog does not own (QTest clicks bypass it on this platform)."""
    user32 = _user32()
    user32.SetCursorPos(x, y)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def _wait_for(predicate, timeout_ms: int):
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


@contextlib.contextmanager
def _foreground_owner(title: str = _OWNER_TITLE):
    """The foreground owner (_OWNER_SCRIPT, its window titled ``title``)
    running with its window in the foreground, as the user's click on it
    leaves it: yields that window's handle once its own click is done, and
    ends the process afterwards. Fails when its window never shows or it
    never clicks; skips only when this session never gives it the foreground
    (a session blocker, never a pass)."""
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER_SCRIPT, title],
        env={**os.environ, "QT_QPA_PLATFORM": "windows"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        owner_hwnd = _wait_for(lambda: _find_window(title), 20000)
        if not owner_hwnd or not _wait_for(
                lambda: _find_window(title + _OWNER_CLICKED), 10000):
            owner.kill()
            stderr = owner.communicate(timeout=10)[1] or ""
            pytest.fail(f"the foreground owner never showed its window or clicked it: "
                        f"{stderr[-800:]}")
        if not _wait_for(lambda: _foreground_hwnd() == owner_hwnd, 10000):
            pytest.skip("this session did not give the helper's window the foreground")
        yield owner_hwnd
    finally:
        owner.terminate()
        try:
            owner.wait(10)
        except subprocess.TimeoutExpired:
            owner.kill()
            owner.wait(10)


def _models_missing():
    from yamabiko.ui.error_dialog import StartupError

    return StartupError.models_missing(details=(
        "Stage: ASR init\n\nTraceback (most recent call last):\n"
        '  File "yamabiko\\asr\\asr_client.py", line 139, in __init__\n'
        "FileNotFoundError: model.bin"))


def _no_audio_device():
    from yamabiko.ui.error_dialog import StartupError

    return StartupError.no_audio_device(details=(
        "Stage: audio capture\n\nTraceback (most recent call last):\n"
        '  File "yamabiko\\capture\\audio.py", line 412, in start\n'
        "RuntimeError: No WASAPI loopback device found for default output"))


def _llama_failed():
    from yamabiko.ui.error_dialog import StartupError

    return StartupError.llama_failed(details=(
        "Stage: translator sanity check\n\nTraceback (most recent call last):\n"
        '  File "yamabiko\\translate\\llama_server.py", line 288, in start\n'
        "RuntimeError: llama-server did not become ready at http://127.0.0.1:8090 "
        "within 180s"))


def _model_file_corrupt():
    from yamabiko.ui.error_dialog import StartupError

    path = "%LOCALAPPDATA%\\Yamabiko\\hf-cache\\models--onnx-community--silero-vad\\model.onnx"
    return StartupError.model_file_corrupt(path=path, details=(
        "Stage: VAD init\n\nTraceback (most recent call last):\n"
        '  File "yamabiko\\models.py", line 301, in verify\n'
        f"yamabiko.models.ModelArtifactCorrupt: sha256 mismatch for {path}"))


def _shown_error_window():
    """The visible ErrorWindow of this process, else None."""
    from PySide6.QtWidgets import QApplication

    from yamabiko.ui.error_window import ErrorWindow

    return next((w for w in QApplication.topLevelWidgets()
                 if isinstance(w, ErrorWindow) and w.isVisible()), None)


def _drive(err, start, *, watchdog_ms: int = 20_000) -> bool:
    """run_error_window(err) on the native platform, with ``start(window)``
    run from inside its local event loop once the window shows. True when
    the call returned by itself; False when the watchdog had to close what
    it opened after ``watchdog_ms`` (a regression fails its test instead of
    hanging the run). A call that returns early leaves Settings open: closed
    here too. The shown window (and through settings_window its Settings)
    stays referenced until it is closed and drained here: once the call
    returns, only a reference cycle keeps it, and a garbage collection in
    a later event pump could free a still-shown window mid-paint (the
    window_for fixture of test_error_window.py)."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.ui.error_window import ErrorWindow, run_error_window

    fired: list[bool] = []
    held: list = []

    def close_everything() -> None:
        fired.append(True)
        for w in QApplication.topLevelWidgets():
            if isinstance(w, (ErrorWindow, SettingsWindow)) and w.isVisible():
                w.close()

    def begin(window) -> None:
        held.append(window)
        start(window)

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(close_everything)
    watchdog.start(watchdog_ms)
    when(lambda: _shown_error_window() is not None,
         lambda: begin(_shown_error_window()), within_s=watchdog_ms / 1000 - 2)
    try:
        run_error_window(err)
    finally:
        watchdog.stop()
        for w in QApplication.topLevelWidgets():
            if isinstance(w, (ErrorWindow, SettingsWindow)) and w.isVisible():
                w.close()
        held.clear()
        drain_qt_debris()
    return not fired


@pytest.mark.parametrize("make_error", [
    _models_missing, _no_audio_device, _llama_failed, _model_file_corrupt,
], ids=["models_missing", "no_audio_device", "llama_failed", "model_file_corrupt"])
def test_real_error_window_opens_720_wide_centered_on_the_primary_work_area(
    real_app, own_app_data, make_error,
):
    """L9: the window opens 720 logical px wide (the visible window, DWM
    frame bounds, at the primary screen's pixel ratio: 1080 device px at
    150 %), centered on the primary monitor's work area (GetMonitorInfoW
    rcWork), within 4 logical px (the expanded client area's frame sits
    2.7 px left of Qt's geometry: probe p6_geom_probe4.py). It ends through
    its Close button (the call returns before the watchdog). Four errors fit
    four different heights: the width must not depend on the height.

    Seeded defects: drop the center_on call from ErrorWindow.showEvent ->
    QDialog's own placement, about (-13, -39) px off -> red; make the window
    712 wide -> the width check red; center with move() after show() ->
    1084 px wide -> red; add the window menu in showEvent again (before the
    native show applies the size fitted while hidden) -> 1084 px wide at
    150 % -> red; skip add_window_menu in ErrorWindow.setVisible -> no
    WS_SYSMENU, no system menu (Alt+Space, the taskbar's Close window) ->
    red."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtTest import QTest

    seen: dict = {}

    def measure(window) -> None:
        def read() -> None:
            hwnd = wintypes.HWND(int(window.winId()))
            seen["frame"] = _visible_bounds(window)
            seen["work"] = _primary_work_area()
            seen["height"] = window.height()
            seen["style"] = ctypes.windll.user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
            seen["menu"] = bool(ctypes.windll.user32.GetSystemMenu(hwnd, False))
            QTest.mouseClick(window.close_button, Qt.LeftButton)

        QTimer.singleShot(300, read)

    assert _drive(make_error(), measure), "the error window did not end on Close"
    frame, work = seen["frame"], seen["work"]
    ratio = QGuiApplication.primaryScreen().devicePixelRatio()
    width = frame.right - frame.left
    assert abs(width - round(720 * ratio)) <= 2, (
        f"the window is {width} device px wide, not {round(720 * ratio)} (720 at {ratio:g}x; "
        f"fitted {seen['height']} px tall)")
    dx = (frame.left + frame.right - work.left - work.right) / 2 / ratio
    dy = (frame.top + frame.bottom - work.top - work.bottom) / 2 / ratio
    assert abs(dx) <= 4 and abs(dy) <= 4, (
        f"window center - primary work area center: ({dx:.1f}, {dy:.1f}) logical px")
    assert seen["style"] & 0x00080000 and seen["menu"], (  # WS_SYSMENU
        f"the error window has no system menu (style {seen['style']:#x}, menu {seen['menu']})")


def test_real_error_window_takes_the_foreground_from_another_app(real_app, own_app_data):
    """L9: the pipeline child shows the window while the user works in another
    app, and Windows refuses a plain activateWindow to a process without the
    last input (probe p6_fg_probe.py; the frozen QMessageBox never had the
    foreground). run_error_window takes it (window_chrome.take_foreground)
    after show: the window is the foreground window, so Esc and Enter reach
    it. The other app is a separate process that clicked its own window.

    Seeded defect: drop the take_foreground call from run_error_window -> the
    other app keeps the foreground -> red. Skips only when this session never
    gives the helper's window the foreground."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest

    seen: dict = {}

    def check(window) -> None:
        def read() -> None:
            seen["foreground"] = _foreground_hwnd()
            seen["window"] = int(window.winId())
            QTest.mouseClick(window.close_button, Qt.LeftButton)

        QTimer.singleShot(400, read)

    with _foreground_owner() as owner_hwnd:
        assert _drive(_models_missing(), check), "the error window did not end on Close"
    assert seen["foreground"] == seen["window"], (
        f"the error window is not the foreground window (foreground {seen['foreground']:#x}, "
        f"window {seen['window']:#x}, the other app {owner_hwnd:#x})")


def test_real_copy_report_puts_the_redacted_report_on_the_native_clipboard(
    real_app, own_app_data, saved_clipboard,
):
    """Copy report puts the whole report on the OS clipboard (CF_UNICODETEXT
    read through Win32, not Qt's copy) with the user's name and profile
    folders gone in every form the window received them: the details (a
    traceback path under USERPROFILE, a repr with doubled backslashes) and
    the recent log, read from this process's real RotatingFileHandler file
    (the LOCALAPPDATA path, a forward-slash USERPROFILE path, the repr of the
    pytest folder named after the user, the bare user name); the button
    switches to its copied state.

    Seeded defects: report_text skips redact_user -> the profile folder is on
    the clipboard -> red; copy_to_clipboard returns at once -> the sentinel
    stays -> red; collect_facts reads no log tail -> the sentinel record is
    missing -> red."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest

    from yamabiko import log_config
    from yamabiko.ui.error_dialog import StartupError

    profile = os.environ["USERPROFILE"]
    local = os.environ["LOCALAPPDATA"]
    user = os.environ.get("USERNAME") or getpass.getuser()
    logs = own_app_data / "logs"
    logs.mkdir(parents=True)
    # The pipeline's handler: a RotatingFileHandler in the main log format on
    # an ancestor of yamabiko.ui.error_dialog, as setup_main_log attaches it.
    handler = RotatingFileHandler(logs / "main-pipeline.log", maxBytes=10 * 1024 * 1024,
                                  backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter(log_config.MAIN_LOG_FORMAT))
    owner_logger = logging.getLogger("yamabiko")
    owner_logger.addHandler(handler)
    seen: dict = {}
    try:
        log = logging.getLogger("yamabiko.test_error_window")
        log.warning("p6 sentinel cache=%s", os.path.join(local, "Yamabiko", "hf-cache"))
        log.warning("p6 sentinel video=%s", Path(profile, "Videos", "show.mkv").as_posix())
        log.warning("p6 sentinel temp=%r", str(own_app_data))
        log.warning("p6 sentinel account %s", user)
        err = StartupError.models_missing(details=(
            "Stage: ASR init\n\nTraceback (most recent call last):\n"
            f'  File "{os.path.join(profile, "Yamabiko", "yamabiko", "main.py")}", '
            "line 2011, in build_pipeline\n"
            "FileNotFoundError: [Errno 2] No such file or directory: "
            f"{os.path.join(local, 'Yamabiko', 'hf-cache', 'model.bin')!r}"))
        real_app.clipboard().setText(_COPY_SENTINEL)

        def copy(window) -> None:
            def click() -> None:
                QTest.mouseClick(window.copy_button, Qt.LeftButton)
                seen["clipboard"] = _native_clipboard_text().replace("\r\n", "\n")  # Qt writes CRLF
                seen["copied"] = window.copied()
                QTest.mouseClick(window.close_button, Qt.LeftButton)

            QTimer.singleShot(300, click)

        assert _drive(err, copy), "the error window did not end on Close"
    finally:
        owner_logger.removeHandler(handler)
        handler.close()

    report = seen.get("clipboard", "")
    assert report.startswith("Yamabiko startup error\nTitle: Some model files are missing\n"), (
        f"Copy report did not put the report on the clipboard: {report[:300]!r}")
    assert "p6 sentinel cache=%LOCALAPPDATA%\\Yamabiko\\hf-cache" in report, (
        f"the recent log is missing from the report or not redacted:\n{report}")
    lowered = report.lower()
    leaks = [form for base in (profile, local)
             for form in (base, base.replace("\\", "/"), base.replace("\\", "\\\\"))
             if form.lower() in lowered]
    assert not leaks, f"user folders on the clipboard: {leaks}\n{report}"
    assert not re.search(rf"(?<!\w){re.escape(user)}(?!\w)", report.replace("<user>", ""),
                         re.IGNORECASE), f"the user name {user!r} is on the clipboard:\n{report}"
    assert seen["copied"], "Copy report did not switch to its copied state"


def test_real_open_settings_opens_audio_and_model_and_the_call_waits_for_both_windows(
    real_app, own_app_data,
):
    """L11 with Decisions 35 and 36: for a missing audio device the window
    offers Open Settings, which opens the real Settings window at Audio &
    model as a top-level window of its own (unparented, as the tray opens
    it). Nothing modal blocks it: a click injected through Windows on
    Settings' Shortcuts tab reaches it while the error window is up (Qt's
    modality blocks real input to windows a modal window does not own;
    QTest clicks bypass it on this platform, probe p6_modal_probe.py).
    Closing the error window first leaves Settings open and run_error_window
    still running (the pipeline must not exit under it); the call returns
    only after Settings closes too.

    Seeded defects: self.setModal(True) in ErrorWindow.__init__ -> the
    injected click is blocked -> red; drop wait_for_settings from
    run_error_window -> the call returns while Settings is open -> red; open
    Settings without page= -> the page check red. (window.exec() in place of
    the local loop stayed green: the window is shown before it, and the
    injected click still reached Settings.)"""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.settings_window import PAGE_NAMES, SettingsWindow
    from yamabiko.ui.error_dialog import StartupError

    err = StartupError.audio_device_gone(
        device_name="Headphones (USB Audio) [Loopback]",
        details=("Stage: audio capture\n\nTraceback (most recent call last):\n"
                 "yamabiko.capture._audio_constants.LoopbackDeviceNotFound: Loopback "
                 "device not found: 'Headphones (USB Audio) [Loopback]'"))
    cursor = _cursor_pos()
    seen: dict = {}

    def start(window) -> None:
        QTest.mouseClick(window.settings_button, Qt.LeftButton)
        when(lambda: window.settings_window is not None and window.settings_window.isVisible(),
             lambda: QTimer.singleShot(500, lambda: click_a_tab(window)), within_s=10)

    def click_a_tab(window) -> None:
        settings = window.settings_window
        # The class only: _drive holds and drains the window itself.
        seen["settings"] = type(settings)
        seen["page"] = settings._tabs.currentIndex()
        x, y = _native_center(settings._tabs._tabs[PAGE_NAMES.index("shortcuts")])
        on_top = _top_window_at(x, y)
        # Another app's window over the tab is a session blocker; this
        # process's own window there (the error window over Settings) is a
        # defect the click catches.
        seen["covered"] = (on_top not in (0, int(settings.winId()))
                           and _window_pid(on_top) != os.getpid())
        _real_click(x, y)
        QTimer.singleShot(400, lambda: close_the_error_window(window, settings))

    def close_the_error_window(window, settings) -> None:
        _user32().SetCursorPos(cursor.x, cursor.y)
        seen["page_after_click"] = settings._tabs.currentIndex()
        QTest.mouseClick(window.close_button, Qt.LeftButton)
        QTimer.singleShot(500, lambda: close_settings(window, settings))

    def close_settings(window, settings) -> None:
        # Runs only while run_error_window is still waiting for Settings.
        seen["error_window_visible"] = window.isVisible()
        seen["settings_visible"] = settings.isVisible()
        settings.close()
        seen["settings_closed_at"] = time.monotonic()

    try:
        returned = _drive(err, start, watchdog_ms=25_000)
        seen["returned_at"] = time.monotonic()
    finally:
        _user32().SetCursorPos(cursor.x, cursor.y)

    if seen.get("covered"):
        pytest.skip("this session cannot bring a test window to the foreground")
    assert seen.get("settings") is SettingsWindow, (
        f"Open Settings opened no Settings window: {seen}")
    assert seen["page"] == PAGE_NAMES.index("audio"), (
        f"Settings opened at page {PAGE_NAMES[seen['page']]!r}, not Audio & model")
    assert seen.get("page_after_click") == PAGE_NAMES.index("shortcuts"), (
        "a real click on Settings did not reach it while the error window was up "
        f"(a modal window blocks windows it does not own): {seen}")
    assert seen.get("settings_visible") and not seen.get("error_window_visible"), (
        f"run_error_window returned while Settings was still open: {seen}")
    assert returned and seen["returned_at"] >= seen["settings_closed_at"], (
        f"run_error_window did not return after both windows closed: {seen}")
