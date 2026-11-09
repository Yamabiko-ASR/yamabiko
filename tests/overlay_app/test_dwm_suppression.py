"""Guard the four-call DWM ceremony in _suppress_windows11_chrome.

The overlay needs four DwmSetWindowAttribute calls beyond Qt's
FramelessWindowHint to fully suppress Windows 11 chrome. If someone
"simplifies" them away, this test catches it. The exact attribute
numbers must match what the Microsoft DWM headers define; see
yamabiko/overlay_app/overlay_window.py for the constants.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

if sys.platform != "win32":
    pytest.skip("DWM suppression is Windows-only", allow_module_level=True)


def test_suppress_windows11_chrome_issues_four_dwm_attribute_calls():
    from yamabiko.overlay_app.overlay_window import _suppress_windows11_chrome

    dwmapi = MagicMock()
    dwmapi.DwmSetWindowAttribute.return_value = 0  # S_OK
    user32 = MagicMock()
    user32.SetWindowPos.return_value = 1  # BOOL success

    fake_windll = MagicMock()
    fake_windll.dwmapi = dwmapi
    fake_windll.user32 = user32

    with patch("ctypes.windll", fake_windll):
        _suppress_windows11_chrome(hwnd=12345)

    # Exactly four DwmSetWindowAttribute calls, in the documented order.
    assert dwmapi.DwmSetWindowAttribute.call_count == 4, (
        "Exactly 4 DWM attribute calls are required to fully suppress Win11 chrome; "
        f"got {dwmapi.DwmSetWindowAttribute.call_count}"
    )

    # Extract (hwnd, attr_id, value) from each call. ctypes.byref is NOT
    # mocked (only ctypes.windll is), so val_ptr is a real CArgObject
    # whose ._obj is the ctypes value instance production constructed --
    # the attribute VALUE is fully recoverable. (An earlier version
    # unpacked val_ptr and dropped it, so flipping DONOTROUND->ROUND or
    # DWMSBT_NONE->Mica kept this test green while Win11 chrome returned.)
    calls = []
    for call in dwmapi.DwmSetWindowAttribute.call_args_list:
        args = call.args
        hwnd, attr_id, val_ptr, val_sz = args
        calls.append((hwnd, attr_id, val_ptr._obj.value, val_sz))

    # Every call must be against the same hwnd, with cbAttribute matching
    # the value's actual size (DWM rejects mismatched sizes with E_INVALIDARG).
    assert all(h == 12345 for h, _, _, _ in calls)
    assert all(sz == 4 for _, _, _, sz in calls), (
        f"cbAttribute must be sizeof(c_int/c_uint)=4; got {[(a, sz) for _, a, _, sz in calls]}"
    )

    # The attribute id -> value mapping (the four-call ceremony):
    # DWMWA_NCRENDERING_POLICY = 2      -> DWMNCRP_DISABLED = 1
    # DWMWA_WINDOW_CORNER_PREFERENCE=33 -> DWMWCP_DONOTROUND = 1
    # DWMWA_BORDER_COLOR = 34           -> DWMWA_COLOR_NONE = 0xFFFFFFFE
    # DWMWA_SYSTEMBACKDROP_TYPE = 38    -> DWMSBT_NONE = 1
    expected = {2: 1, 33: 1, 34: 0xFFFFFFFE, 38: 1}
    got = {attr_id: value for _, attr_id, value, _ in calls}
    assert got == expected, (
        f"DWMWA_* attribute values drifted: expected {expected}, got {got} -- "
        "a wrong VALUE silently re-introduces rounded corners / border / "
        "Mica backdrop (Windows 11 DWM gotcha)"
    )

    # SetWindowPos must be called afterwards with SWP_FRAMECHANGED to force the
    # frame to re-evaluate — without it the attributes are silently ignored on
    # already-visible windows.
    user32.SetWindowPos.assert_called_once()
    swp_flags = user32.SetWindowPos.call_args.args[-1]
    SWP_FRAMECHANGED = 0x0020
    assert swp_flags & SWP_FRAMECHANGED, "SetWindowPos must include SWP_FRAMECHANGED"


def test_suppress_noop_when_dwmapi_missing():
    """No crash if dwmapi/user32 can't be loaded (non-Win or policy-restricted)."""
    from yamabiko.overlay_app.overlay_window import _suppress_windows11_chrome

    class _BadWindll:
        def __getattr__(self, name):
            raise OSError("dwmapi not found")

    with patch("ctypes.windll", _BadWindll()):
        _suppress_windows11_chrome(hwnd=1)  # must not raise


# --- the 250 ms re-suppression timer's lifetime ---


def test_show_then_quick_close_kills_pending_chrome_retry_timer(
    qapp, monkeypatch, capfd,
):
    """showEvent schedules a 250 ms re-suppression pass. If the window
    dies inside those 250 ms (overlay quit right after show), the
    pending timer must die WITH the widget instead of firing
    `self.winId()` on the destroyed C++ object -- a libshiboken
    RuntimeError through the event loop that pollutes E2E stderr scans.

    Seeded defect: revert showEvent to the unparented
    QTimer.singleShot(250, lambda: _suppress_windows11_chrome(...)) ->
    the timer outlives the window, fires on the deleted object, and the
    excepthook/stderr capture below goes red.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko.overlay_app import overlay_window as ow_mod

    calls: list[int] = []
    monkeypatch.setattr(ow_mod, "_suppress_windows11_chrome", calls.append)

    w = ow_mod.OverlayWindow()
    w.show()
    assert len(calls) == 1, "immediate chrome pass missing on show"
    # The event loop hasn't pumped since the monkeypatch, so calls[0] is
    # OUR window's hwnd. Other suite tests' still-alive shown windows
    # have their own pending 250 ms retry timers; the qWait below pumps
    # those too, so per-window assertions must filter on this hwnd.
    my_hwnd = calls[0]

    hook_errors: list[tuple] = []
    old_hook = sys.excepthook
    sys.excepthook = lambda *exc: hook_errors.append(exc)
    try:
        w.close()
        w.deleteLater()
        # Flush the deferred delete so the C++ object is truly destroyed
        # while the 250 ms retry is still pending, then drive the event
        # loop past the retry deadline so a surviving timer fires now,
        # inside the capture window.
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QTest.qWait(450)
    finally:
        sys.excepthook = old_hook

    err = capfd.readouterr().err
    assert not hook_errors, (
        f"chrome retry timer fired on the destroyed window: {hook_errors}"
    )
    assert "RuntimeError" not in err and "already deleted" not in err, (
        f"PySide6 reported a deleted-object access on stderr:\n{err}"
    )
    assert calls.count(my_hwnd) == 1, (
        "retry pass ran against the destroyed window"
    )


def test_dwm_retry_rearms_after_interactive_toggle_recreates_hwnd(qapp, monkeypatch):
    """set_interactive() calls setWindowFlags()+show(), which destroys and
    recreates the native HWND -- the DWM attributes die with the old
    HWND. The re-entered showEvent must re-arm the 250 ms retry pass
    for the recreated window; the `_chrome_suppressed` once-gate used
    to swallow it, so after any interactive toggle the second (load-bearing)
    suppression pass never ran again.

    Seeded defect: remove the `self._chrome_suppressed = False` reset
    from set_interactive() -> no retry timer is scheduled by either showEvent
    of the cycle, so after the post-toggle qWait the call count for the
    final hwnd never grows past the immediate-pass count -> red.
    """
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app import overlay_window as ow_mod

    calls: list[int] = []
    monkeypatch.setattr(ow_mod, "_suppress_windows11_chrome", calls.append)

    w = ow_mod.OverlayWindow()
    try:
        w.show()
        # Drain the first-show retry so no timer from the initial show is
        # pending when the lock cycle starts.
        QTest.qWait(450)

        w.set_interactive(True)
        w.set_interactive(False)
        # The retry lambdas resolve self.winId() at FIRE time, so any
        # retry scheduled during the cycle lands on the final hwnd.
        hwnd_final = int(w.winId())
        n_immediate = calls.count(hwnd_final)

        QTest.qWait(450)  # past the 250 ms retry deadline
        assert calls.count(hwnd_final) > n_immediate, (
            f"no deferred re-suppression pass ran against the re-created "
            f"hwnd {hwnd_final} (immediate={n_immediate}, calls={calls}); "
            "the once-gate swallowed the retry after the HWND was "
            "recreated, so Win11 chrome returns after every interactive toggle."
        )
    finally:
        w.close()


def test_show_schedules_second_chrome_pass_while_window_alive(qapp, monkeypatch):
    """The timer lifetime fix must NOT weaken the ceremony: some Win11
    DWM attributes are silently reset by the first redraw after show,
    so the suppression runs once immediately and once again ~250 ms
    later against the SAME hwnd (the four-call ceremony and its second
    pass are both load-bearing).

    Seeded defect: delete the retry-timer block from showEvent -> only
    the immediate pass ever runs -> red.
    """
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app import overlay_window as ow_mod

    calls: list[int] = []
    monkeypatch.setattr(ow_mod, "_suppress_windows11_chrome", calls.append)

    w = ow_mod.OverlayWindow()
    try:
        w.show()
        assert len(calls) == 1, "immediate chrome pass missing on show"
        hwnd = calls[0]
        QTest.qWait(450)  # past the 250 ms retry
        # Count OUR hwnd only: pumping the loop also fires pending retry
        # timers left by other suite tests' still-alive shown windows.
        assert calls.count(hwnd) >= 2, (
            f"no deferred re-suppression pass ran against hwnd {hwnd} "
            f"(calls={calls}); Win11 chrome would reappear after the "
            f"first redraw"
        )
    finally:
        w.close()
