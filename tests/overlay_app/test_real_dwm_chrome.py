"""Real-platform DWM chrome-suppression test (@pytest.mark.real).

``_suppress_windows11_chrome()`` issues four ``DwmSetWindowAttribute`` calls to
strip the Win11 decorations Qt's ``FramelessWindowHint`` leaves behind (rounded
corners, accent border, Mica/acrylic backdrop, non-client rendering). Every
existing test of it drives a FAKE ``windll`` and asserts on the ctypes CALL
SHAPE -- none verifies the real Win32 effect, so a wrong attribute id / value /
struct size that the OS silently rejects would pass. This closes that (M-17):
build a REAL ``OverlayWindow`` on the native windows QPA and read the
attributes back with ``DwmGetWindowAttribute``, asserting the effective
post-suppression state. The HRESULT of every get is checked before its value
is trusted.

Readback coverage (measured on Win11 26xxx, PySide6 6.11):
  * DWMWA_WINDOW_CORNER_PREFERENCE (33) -- readable. Suppression sets
    DWMWCP_DONOTROUND (1); an unsuppressed frameless window reads a
    rounded-corner preference (measured 2). Removing the set flips the readback
    back to that default, so the assertion is sensitive to exactly that line.
  * DWMWA_SYSTEMBACKDROP_TYPE (38) -- readable. Suppression sets DWMSBT_NONE
    (1); the unsuppressed default is DWMSBT_AUTO (0). Same sensitivity.

Documented, NOT asserted (partial real coverage beats none):
  * DWMWA_NCRENDERING_POLICY (2) and DWMWA_BORDER_COLOR (34) are WRITE-ONLY on
    this build -- ``DwmGetWindowAttribute`` returns E_INVALIDARG (0x80070057),
    so their effect cannot be read back.
  * DWMWA_NCRENDERING_ENABLED (1), the readable sibling of NCRENDERING_POLICY,
    is already FALSE for ANY frameless window (measured), so it cannot isolate
    the suppression's set from the frameless default -- asserting it would be an
    equivalent-mutation Liar (charter §3.1.4), so it is deliberately excluded.

Skips unless a real (non-offscreen) QPA is available, same gating as
tests/overlay_app/test_real_gui_layout.py; run explicitly on a workstation:

    py -3.13 -m pytest tests/overlay_app/test_real_dwm_chrome.py -m real -v
"""
from __future__ import annotations

import ctypes
import os
import sys

import pytest

pytestmark = pytest.mark.real

# DWM attribute ids + expected values -- mirror the constants in
# yamabiko/overlay_app/overlay_window.py::_suppress_windows11_chrome.
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWA_SYSTEMBACKDROP_TYPE = 38
_DWMWCP_DONOTROUND = 1
_DWMSBT_NONE = 1
_S_OK = 0


@pytest.fixture(scope="module")
def real_app():
    """A QApplication on a REAL (non-offscreen) platform, or skip.

    Same rationale as tests/overlay_app/test_real_gui_layout.py::real_app: the
    QPA platform is a per-process singleton fixed at first QApplication. Only
    when none exists yet (this file run in its own invocation) do we force the
    native windows platform. When the offscreen suite already created one, DWM
    is unavailable and these tests skip -- so headless CI stays green.
    """
    if sys.platform != "win32":
        pytest.skip("DWM is Windows-only")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        os.environ["QT_QPA_PLATFORM"] = "windows"
    app = QApplication.instance() or QApplication([])
    if app.platformName() == "offscreen":
        pytest.skip(
            "real GUI platform unavailable (QApplication is offscreen); run "
            "this file in its own pytest invocation on a workstation"
        )
    yield app


def _dwm_get_int(hwnd: int, attr: int, ctype=ctypes.c_int) -> "tuple[int, int]":
    """Read a DWM attribute off ``hwnd``; return (HRESULT & 0xFFFFFFFF, value)."""
    v = ctype()
    hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        hwnd, attr, ctypes.byref(v), ctypes.sizeof(v),
    )
    return hr & 0xFFFFFFFF, v.value


def test_suppress_windows11_chrome_disables_rounding_and_backdrop_on_real_window(
    real_app,
):
    """After a real OverlayWindow shows, DWM must report the suppressed chrome:
    corners NOT rounded (DWMWCP_DONOTROUND) and NO system backdrop
    (DWMSBT_NONE), read back from the live HWND via DwmGetWindowAttribute. Each
    get's HRESULT is asserted S_OK before its value is trusted.

    These are the two attributes DWM lets us read back on this Win11 build (see
    the module docstring); both differ from a frameless window's defaults, so
    each assertion goes red if its production _set() is removed.

    Seeded-defect checks (each red on this real tier):
    - delete ``_set("CORNER_PREFERENCE", DWMWA_WINDOW_CORNER_PREFERENCE,
      DWMWCP_DONOTROUND)`` in _suppress_windows11_chrome -> the corner
      preference reads back the frameless rounded default (2) -> corner assert
      red.
    - delete ``_set("SYSTEMBACKDROP_TYPE", DWMWA_SYSTEMBACKDROP_TYPE,
      DWMSBT_NONE)`` -> the backdrop reads back DWMSBT_AUTO (0) -> backdrop
      assert red.
    """
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        w.show()
        QApplication.processEvents()
        # showEvent applies suppression immediately AND schedules a 250 ms
        # retry; wait past that so the effective state is settled before
        # readback.
        QTest.qWait(500)
        QApplication.processEvents()

        hwnd = int(w.winId())

        hr_corner, corner = _dwm_get_int(hwnd, _DWMWA_WINDOW_CORNER_PREFERENCE)
        assert hr_corner == _S_OK, (
            f"DwmGetWindowAttribute(CORNER_PREFERENCE) failed: HRESULT="
            f"0x{hr_corner:08x}; cannot verify the corner suppression on this "
            f"Windows build"
        )
        assert corner == _DWMWCP_DONOTROUND, (
            f"DWM corner preference is {corner}, expected DWMWCP_DONOTROUND "
            f"({_DWMWCP_DONOTROUND}); Win11 is still rounding the overlay's "
            f"corners -- _suppress_windows11_chrome's CORNER_PREFERENCE set did "
            f"not take effect on the real window"
        )

        hr_bd, backdrop = _dwm_get_int(hwnd, _DWMWA_SYSTEMBACKDROP_TYPE)
        assert hr_bd == _S_OK, (
            f"DwmGetWindowAttribute(SYSTEMBACKDROP_TYPE) failed: HRESULT="
            f"0x{hr_bd:08x}; cannot verify the backdrop suppression on this "
            f"Windows build"
        )
        assert backdrop == _DWMSBT_NONE, (
            f"DWM system backdrop is {backdrop}, expected DWMSBT_NONE "
            f"({_DWMSBT_NONE}); a Mica/acrylic backdrop is still painting "
            f"behind the overlay -- the SYSTEMBACKDROP_TYPE set did not take "
            f"effect on the real window"
        )
    finally:
        w.close()


def test_shown_overlay_leaves_the_user_notification_state_as_hidden(real_app):
    """Windows treats a visible topmost window whose rect covers its monitor
    as a full-screen app: SHQueryUserNotificationState answers QUNS_BUSY, and
    the default Do not disturb rule "When using an app in full-screen mode"
    then hides every Yamabiko notification while subtitles show. The shown
    overlay must read the same state as with it hidden (read before it
    shows, so a full-screen app running anyway reads BUSY on both sides).

    Seeded defect: pin the window to the whole screen.geometry() in
    _pin_to_screen -> the shown overlay reads QUNS_BUSY (2) where the hidden
    one read QUNS_ACCEPTS_NOTIFICATIONS (5) -> red.
    """
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    def notification_state() -> int:
        state = ctypes.c_int()
        hr = ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state))
        assert hr & 0xFFFFFFFF == _S_OK, (
            f"SHQueryUserNotificationState failed: HRESULT=0x{hr & 0xFFFFFFFF:08x}")
        return state.value

    hidden = notification_state()
    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        w.show()
        shown = []
        for _ in range(4):
            QTest.qWait(400)
            shown.append(notification_state())
    finally:
        w.close()
    assert shown == [hidden] * 4, (
        f"the shown overlay reads user notification state {shown}, hidden it "
        f"read {hidden}: Windows takes the overlay for a full-screen app "
        f"(QUNS_BUSY is 2) and Do not disturb hides Yamabiko's notifications"
    )
