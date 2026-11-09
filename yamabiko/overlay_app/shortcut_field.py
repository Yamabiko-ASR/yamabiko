"""Click-to-record shortcut field for the Settings Shortcuts tab.

Idle: the shortcut as flat key caps with a pencil mark. Click it (or press
Space or Enter on it) to record: an accent double edge, a still record dot,
"Press the new shortcut…" and an Esc cap with "cancels". The first full
combo (one main key with Ctrl, Alt or Win, or an F-key) is recorded;
modifier-only presses and other keys keep it waiting; Esc, Tab or a click
elsewhere cancels. Warning (the owner sets it): an amber double edge and a
warning mark instead of the pencil.

On the native Windows platform a recording field that has focus in the
foreground window also takes the main key in a low-level keyboard hook:
Windows hands a combo another app holds with RegisterHotKey to that app and
posts only the modifiers to the window. The hook swallows a key-down that
makes a recordable combo, so neither that app's shortcut nor Yamabiko's own
fires while recording; every other key (Esc, the modifiers) goes on to Qt.
Qt's key events stay the fallback.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import weakref
from ctypes import wintypes
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from yamabiko.hotkeys import format_combo, parse_combo

from .controls import FIELD_H
from .keycaps import KeyCaps
from .segmented import ACTIVATION_KEYS, KeyboardRingMixin

WIDTH = 272
_RECORD_TEXT = "Press the new shortcut…"
_CANCEL_TEXT = "cancels"
# Insets: the mockup's 1 px border plus its "padding: 0 10px 0 5px".
_CAPS_X = 6    # the key caps
_DOT_X = 11    # the record dot
_RIGHT = 11    # the pencil, the warning mark and "cancels" end here
_FKEYS = {int(getattr(Qt.Key, f"Key_F{n}")): f"F{n}" for n in range(1, 25)}
_log = logging.getLogger(__name__)

_WH_KEYBOARD_LL = 13
_HC_ACTION = 0
_KEY_DOWNS = (0x0100, 0x0104)  # WM_KEYDOWN, WM_SYSKEYDOWN
# The virtual keys GetAsyncKeyState reads for each Qt modifier.
_MODIFIER_VKS = ((0x11, Qt.ControlModifier), (0x12, Qt.AltModifier), (0x10, Qt.ShiftModifier),
                 (0x5B, Qt.MetaModifier), (0x5C, Qt.MetaModifier))


def _vk_name(vk: int) -> Optional[str]:
    if 0x41 <= vk <= 0x5A or 0x30 <= vk <= 0x39:
        return chr(vk)
    if 0x70 <= vk <= 0x87:
        return f"F{vk - 0x6F}"
    return None


def _qt_key_name(key: int) -> Optional[str]:
    if int(Qt.Key_A) <= key <= int(Qt.Key_Z) or int(Qt.Key_0) <= key <= int(Qt.Key_9):
        return chr(key)
    return _FKEYS.get(key)


def combo_from_key_event(key: int, modifiers, native_vk: int = 0) -> Optional[str]:
    """The stored spelling of a key press ("Ctrl+Alt+K"), or None when the
    press is not a bindable shortcut. The Windows virtual key wins over the
    Qt key: under Ctrl+Alt (AltGr) some layouts report the AltGr character."""
    mods = set()
    if modifiers & Qt.ControlModifier:
        mods.add("Ctrl")
    if modifiers & Qt.AltModifier:
        mods.add("Alt")
    if modifiers & Qt.ShiftModifier:
        mods.add("Shift")
    if modifiers & Qt.MetaModifier:
        mods.add("Win")
    main = _vk_name(native_vk) or _qt_key_name(int(key))
    if main is None:
        return None
    combo = format_combo(mods, main)
    return combo if parse_combo(combo) is not None else None


@functools.cache
def _win32():
    """Private user32 and kernel32 handles (argtypes stay local) and the
    HOOKPROC type."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    hookproc = ctypes.WINFUNCTYPE(wintypes.LPARAM, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.SetWindowsHookExW.argtypes = (ctypes.c_int, hookproc, wintypes.HINSTANCE,
                                         wintypes.DWORD)
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM,
                                      wintypes.LPARAM)
    user32.CallNextHookEx.restype = wintypes.LPARAM  # LRESULT
    user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    return user32, kernel32, hookproc


def _held_modifiers():
    """The modifiers held right now, as Qt flags."""
    user32 = _win32()[0]
    held = Qt.KeyboardModifier.NoModifier
    for vk, flag in _MODIFIER_VKS:
        if user32.GetAsyncKeyState(vk) & 0x8000:  # high bit: down now
            held |= flag
    return held


class _KeyDownHook:
    """A WH_KEYBOARD_LL hook on the GUI thread (its event loop pumps the
    calls). ``take(vk)`` sees each key-down first and returns True to
    swallow it. The hook holds the field weakly, so the field's
    ``destroyed`` connection that keeps the hook forms no cycle."""

    def __init__(self, take: Callable[[int], bool]) -> None:
        self._take = weakref.WeakMethod(take)
        self._handle = None
        self._proc = None  # the HOOKPROC, referenced while Windows may call it

    def install(self) -> None:
        if self._handle:
            return
        user32, kernel32, hookproc = _win32()
        proc = hookproc(self._call)
        handle = user32.SetWindowsHookExW(_WH_KEYBOARD_LL, proc, kernel32.GetModuleHandleW(None), 0)
        if not handle:
            _log.warning("keyboard hook unavailable (error %d): a shortcut another app holds "
                         "can't be recorded", ctypes.get_last_error())
            return
        self._handle, self._proc = handle, proc

    def remove(self) -> None:
        if self._handle:
            _win32()[0].UnhookWindowsHookEx(self._handle)
            self._handle = self._proc = None

    def _call(self, n_code: int, w_param: int, l_param: int) -> int:
        try:
            if n_code == _HC_ACTION and w_param in _KEY_DOWNS:
                take = self._take()
                # l_param points at a KBDLLHOOKSTRUCT; vkCode is its first DWORD.
                if take is not None and take(wintypes.DWORD.from_address(l_param).value):
                    return 1
        except BaseException:
            # A ctypes callback that raises returns garbage, which Windows
            # would read as "swallow": log it and pass the key on.
            _log.exception("shortcut field keyboard hook failed; the key goes on")
        return _win32()[0].CallNextHookEx(None, n_code, w_param, l_param)


class ShortcutField(KeyboardRingMixin, QWidget):
    """One shortcut as a click-to-record field (see the module docstring)."""

    recorded = Signal(str)
    recordingChanged = Signal(bool)

    def __init__(self, combo: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._combo = combo
        self._recording = False
        self._warning = False
        self._hooked: Optional[str] = None  # a combo the hook took, not yet recorded
        hook = self._hook = _KeyDownHook(self._take_hooked_key)
        # Never left installed: a deleted field unhooks too. A lambda, as
        # PySide holds a bound method's object only weakly.
        self.destroyed.connect(lambda: hook.remove())
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self.setFixedSize(WIDTH, FIELD_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._caps = KeyCaps(combo, palette, self)
        self._caps.move(_CAPS_X, (FIELD_H - self._caps.sizeHint().height()) // 2)
        self._esc = KeyCaps("Esc", palette, self)
        # A live theme switch gives the caps their font back after the
        # field's own (window_chrome.apply_window_qss): size them then too.
        self._caps.installEventFilter(self)
        self._esc.installEventFilter(self)
        self._sync()

    def combo(self) -> str:
        return self._combo

    def set_combo(self, combo: str) -> None:
        self._combo = combo
        self._caps.set_sequence(combo)
        self._sync()

    def warning(self) -> bool:
        return self._warning

    def set_warning(self, on: bool) -> None:
        self._warning = bool(on)
        self.update()

    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self) -> None:
        if self._recording or not self.isEnabled():
            return
        self._recording = True
        if self.hasFocus() and QGuiApplication.platformName() == "windows":
            self._hook.install()  # only for keys typed into this field
        self._sync()
        self.recordingChanged.emit(True)

    def cancel_recording(self) -> None:
        if not self._recording:
            return
        self._stop_recording()
        self._sync()
        self.recordingChanged.emit(False)

    def _stop_recording(self) -> None:
        self._recording = False
        self._hooked = None
        self._hook.remove()

    def _record(self, combo: str) -> None:
        self._stop_recording()
        self.set_combo(combo)
        self.recordingChanged.emit(False)
        self.recorded.emit(combo)

    def _take_hooked_key(self, vk: int) -> bool:
        """The hook's test for a key-down (keep it fast): swallow a key that
        makes a recordable combo with the held modifiers and record it from
        the event loop; anything else goes on to Qt. Only keys typed into
        this field count: it must have focus in the foreground window."""
        if _vk_name(vk) is None:
            return False  # a modifier, Esc or another key: keyPressEvent's
        if (not self.hasFocus()
                or _win32()[0].GetForegroundWindow() != int(self.window().winId())):
            return False
        combo = combo_from_key_event(0, _held_modifiers(), vk)
        if combo is None:
            return False
        if self._hooked is None:
            self._hooked = combo
            QTimer.singleShot(0, self, self._record_hooked)
        return True  # an auto-repeat before that runs is swallowed too

    def _record_hooked(self) -> None:
        combo, self._hooked = self._hooked, None
        if combo is not None and self._recording:
            self._record(combo)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self._caps.apply_palette(palette)
        self._esc.apply_palette(palette)
        self.update()

    def _sync(self) -> None:
        self._caps.setVisible(not self._recording)
        self._esc.setVisible(self._recording)
        # A child without a layout is sized only when it is shown: size the
        # caps here so a Reset while they are on screen never clips them.
        self._caps.adjustSize()
        self._esc.adjustSize()
        if self._recording:
            cancel_w = QFontMetrics(self.font()).horizontalAdvance(_CANCEL_TEXT)
            esc = self._esc.sizeHint()
            self._esc.move(self.width() - _RIGHT - cancel_w - 5 - esc.width(),
                           (FIELD_H - esc.height()) // 2)
            self.setAccessibleName(_RECORD_TEXT)
        else:
            self.setAccessibleName(self._combo)
        self.update()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            # The caps and the "cancels" width follow the field's font.
            self._sync()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Type.FontChange and (watched is self._caps
                                                       or watched is self._esc):
            self._sync()  # the caps' size follows their own font
        return False

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            self.start_recording()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if not self._recording:
            # KeyboardRingMixin lights the ring for the activation keys.
            super().keyPressEvent(event)
            if event.key() in ACTIVATION_KEYS:
                self.start_recording()
                event.accept()
            return
        event.accept()
        if event.key() == Qt.Key_Escape:
            self.cancel_recording()
            return
        combo = combo_from_key_event(int(event.key()), event.modifiers(), event.nativeVirtualKey())
        if combo is None:
            return  # a modifier alone, or a key Yamabiko can't bind: keep waiting
        self._record(combo)

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.cancel_recording()
        super().focusOutEvent(event)

    def _paint_warning_mark(self, p: QPainter, x: float, y: float) -> None:
        tri = QPainterPath(QPointF(x + 8, y + 2))
        tri.lineTo(x + 14.5, y + 13.6)
        tri.lineTo(x + 1.5, y + 13.6)
        tri.closeSubpath()
        pen = QPen(QColor(self._palette["warn"]), 1.2)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(QColor(self._palette["warn_mark"]))
        p.drawPath(tri)
        stem = QPen(QColor("#1f2421"), 1.6)
        stem.setCapStyle(Qt.RoundCap)
        p.setPen(stem)
        p.drawLine(QPointF(x + 8, y + 6.4), QPointF(x + 8, y + 9.7))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1f2421"))
        p.drawEllipse(QPointF(x + 8, y + 11.6), 0.95, 0.95)

    def _paint_pencil(self, p: QPainter, x: float, y: float) -> None:
        pen = QPen(QColor(self._palette["muted"]), 1.1)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath(QPointF(x + 7.6, y + 2.1))
        for px, py in ((9.9, 4.4), (4.2, 10.1), (1.6, 10.4), (1.9, 7.8)):
            path.lineTo(x + px, y + py)
        path.closeSubpath()
        p.drawPath(path)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        enabled = self.isEnabled()
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        double = None
        if self._recording:
            double = "accent"
        elif self._warning:
            double = "warn"
        if double is not None:
            border = QColor(self._palette[double])
        elif self.ring_visible():
            border = QColor(self._palette["accent"])
        elif enabled and self.underMouse():
            border = QColor(self._palette["muted"])
        else:
            border = QColor(self._palette["edge" if enabled else "line"])
        p.setPen(QPen(border, 1.0))
        p.setBrush(QColor(self._palette["field" if enabled else "ground"]))
        p.drawRoundedRect(outer, 4.0, 4.0)
        if double is not None:  # the mockup's inset second edge
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(outer.adjusted(1, 1, -1, -1), 3.0, 3.0)
        if self._recording:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self._palette["accent"]))
            p.drawEllipse(QRectF(_DOT_X, (self.height() - 7) / 2.0, 7, 7))
            p.setPen(QColor(self._palette["ink"]))
            p.drawText(QRectF(_DOT_X + 15, 0, self.width() - _DOT_X - 15, self.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, _RECORD_TEXT)
            p.setPen(QColor(self._palette["muted"]))
            p.drawText(QRectF(0, 0, self.width() - _RIGHT, self.height()),
                       Qt.AlignRight | Qt.AlignVCenter, _CANCEL_TEXT)
        elif self._warning:
            self._paint_warning_mark(p, self.width() - _RIGHT - 16, (self.height() - 16) / 2.0)
        else:
            self._paint_pencil(p, self.width() - _RIGHT - 12, (self.height() - 12) / 2.0)
        p.end()
