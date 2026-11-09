"""Screen pixels for the real GUI tests, without Pillow.

The tests measure in Windows' physical virtual-desktop pixels (DWM frame
bounds, GetWindowRect, ClientToScreen). BitBlt from the display device
context, in a thread that is per-monitor DPI aware, reads exactly those on
every screen: probed pixel-identical to Pillow's
``ImageGrab.grab(bbox, all_screens=True)`` on a 150 % and a 100 % screen.
(``QScreen.grabWindow(0, x, y, w, h)`` takes logical coordinates relative to
one screen instead.) Pillow is not a dependency: the release venv, built
from constraints-release.txt, has none.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

_SRCCOPY = 0x00CC0020
_DIB_RGB_COLORS = 0
# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: device context coordinates are
# physical pixels on every screen, whatever the calling thread had.
_PER_MONITOR_AWARE_V2 = -4


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class ScreenGrab:
    """A grabbed screen rectangle, ``width`` x ``height`` pixels; (0, 0) is
    its top-left."""

    def __init__(self, width: int, height: int, bgra: bytes) -> None:
        self.width = width
        self.height = height
        self._bgra = bgra

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        """The (r, g, b) of one pixel."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"pixel ({x}, {y}) is outside {self.width} x {self.height}")
        i = (y * self.width + x) * 4
        return self._bgra[i + 2], self._bgra[i + 1], self._bgra[i]


def _api():
    user32 = ctypes.WinDLL("user32")  # private handles: argtypes stay local
    gdi32 = ctypes.WinDLL("gdi32")
    user32.SetThreadDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
    user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    gdi32.CreateDCW.argtypes = (
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p,
    )
    gdi32.CreateDCW.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = (wintypes.HDC, ctypes.c_int, ctypes.c_int)
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.BitBlt.argtypes = (
        wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
    )
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = (
        wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT,
    )
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
    gdi32.DeleteDC.argtypes = (wintypes.HDC,)
    return user32, gdi32


def grab_screen(left: int, top: int, right: int, bottom: int) -> ScreenGrab:
    """The screen as composed now, from physical virtual-desktop pixel
    ``left``, ``top`` to ``right``, ``bottom`` (both exclusive)."""
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError(f"empty screen rectangle {(left, top, right, bottom)}")
    user32, gdi32 = _api()
    previous = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
    screen = gdi32.CreateDCW("DISPLAY", None, None, None)
    memory = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
    try:
        old = gdi32.SelectObject(memory, bitmap)
        copied = gdi32.BitBlt(memory, 0, 0, width, height, screen, left, top, _SRCCOPY)
        gdi32.SelectObject(memory, old)  # GetDIBits needs the bitmap deselected
        if not copied:
            raise OSError(f"BitBlt of the screen at {(left, top, right, bottom)} failed")
        header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), width, -height, 1, 32, 0)
        pixels = ctypes.create_string_buffer(width * height * 4)
        rows = gdi32.GetDIBits(
            memory, bitmap, 0, height, pixels, ctypes.byref(header), _DIB_RGB_COLORS,
        )
        if rows != height:
            raise OSError(f"GetDIBits returned {rows} of {height} rows")
        return ScreenGrab(width, height, pixels.raw)
    finally:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        gdi32.DeleteDC(screen)
        if previous:
            user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))


def screen_pixel(x: int, y: int) -> tuple[int, int, int]:
    """The (r, g, b) of the physical screen pixel at ``x``, ``y``."""
    return grab_screen(x, y, x + 1, y + 1).pixel(0, 0)
