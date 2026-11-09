"""Window chrome: caption buttons, title band paint, client-area fit, and the
extended-style rule of the series bar's z-order fallback."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QVBoxLayout, QWidget

from yamabiko.overlay_app.theme import MIST
from yamabiko.overlay_app.window_chrome import APP_RED, ClientAreaFit, TitleBand


def _window(qtbot):
    w = QWidget()
    qtbot.addWidget(w)
    root = QVBoxLayout(w)
    root.setContentsMargins(0, 0, 0, 0)
    band = TitleBand("Yamabiko settings", "山彦", "Yamabiko settings", MIST)
    root.addWidget(band)
    root.addStretch(1)
    w.resize(1000, 300)
    w.show()
    return w, band, root


def test_caption_buttons_close_and_double_click_maximizes(qtbot):
    # Seeded defects: leave close unwired, or ignore the double-click on the
    # sky -> red.
    w, band, _root = _window(qtbot)
    QTest.mouseDClick(band, Qt.LeftButton, pos=QPoint(500, 60))
    assert w.isMaximized()
    QTest.mouseClick(band.caption_button("close"), Qt.LeftButton)
    assert not w.isVisible()


def test_title_band_paints_the_app_mark_on_the_sky(qtbot):
    # Seeded defect: forget paint_app_mark -> the caption's top-left corner
    # shows the sky -> red.
    _w, band, _root = _window(qtbot)
    img = QImage(band.size(), QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    band.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert img.pixelColor(20, 9).name() == APP_RED


def test_client_area_fit_insets_the_hidden_right_strip(qtbot):
    # Qt 6.11 sizes an expanded-client window wider than its Win32 client:
    # the root layout's right margin takes up the difference so the close
    # button stays on screen. Seeded defect: never refit on resize -> red.
    w, band, root = _window(qtbot)
    ClientAreaFit(w, root, measure=lambda window: window.width() - 6)
    w.resize(1006, 300)
    assert root.contentsMargins().right() == 6
    close = band.caption_button("close")
    assert close.mapTo(w, QPoint(close.width(), 0)).x() == w.width() - 6


def test_only_activating_unowned_windows_count_as_app_windows():
    """Lead ruling on final review round 5: the z-order fallback gives the
    keyboard only to an ordinary app window. A tool window, a window that
    never activates (WS_EX_NOACTIVATE: an on-screen keyboard) and a
    click-through overlay (WS_EX_TRANSPARENT: a game or GPU overlay) do not
    count, nor an owned window unless it is marked WS_EX_APPWINDOW.

    Seeded defects: drop the WS_EX_NOACTIVATE or the WS_EX_TRANSPARENT test
    -> red; drop the owner test -> red."""
    from yamabiko.overlay_app.window_chrome import _ordinary_app_styles

    plain, tool, appwindow = 0x00000100, 0x00000080, 0x00040000  # WS_EX_WINDOWEDGE, ...
    noactivate, transparent, layered = 0x08000000, 0x00000020, 0x00080000
    assert _ordinary_app_styles(plain, owned=False)
    assert _ordinary_app_styles(plain | appwindow, owned=True)
    assert not _ordinary_app_styles(plain, owned=True)
    assert not _ordinary_app_styles(plain | tool, owned=False)
    assert not _ordinary_app_styles(plain | noactivate, owned=False)
    assert not _ordinary_app_styles(plain | transparent | layered, owned=False)


def test_the_fallback_takes_the_topmost_ordinary_window_of_any_process(monkeypatch):
    """Lead ruling on visual check 4: with Settings open, a summon from the
    tray and Esc gave the foreground to a terminal behind Settings, because
    the z-order fallback skipped this process's windows. The walk takes the
    topmost ordinary window of any process: Settings (a plain window of this
    process) above another app's window wins; the hidden bar, the flyout (a
    tool window) and the click-through subtitle window above it do not count.

    Seeded defect: skip this process's windows -> the other app's window is
    picked -> red."""
    import os

    from yamabiko.overlay_app import window_chrome as wc

    ours, other = os.getpid(), os.getpid() + 1
    windowedge, tool, transparent, layered = 0x100, 0x80, 0x20, 0x80000

    def facts(visible: bool, exstyle: int) -> "wc._WindowFacts":
        return wc._WindowFacts(visible=visible, iconic=False, exstyle=exstyle, owned=False,
                               cloaked=False, cls="Qt6110QWindowIcon", has_area=True)

    windows = {  # topmost first: (facts, owner pid)
        0x1: (facts(False, windowedge | tool), ours),  # the bar, hidden by then
        0x2: (facts(True, windowedge | tool), ours),  # the tray flyout
        0x3: (facts(True, tool | transparent | layered), ours),  # the subtitle window
        0x4: (facts(True, windowedge), ours),  # Settings
        0x5: (facts(True, windowedge), other),  # the terminal behind it
    }
    monkeypatch.setattr(wc, "_native_windows", lambda: True)
    monkeypatch.setattr(wc, "_enum_windows", lambda: list(windows))
    monkeypatch.setattr(wc, "_window_facts", lambda hwnd: windows[hwnd][0])
    monkeypatch.setattr(wc, "_window_pid", lambda hwnd: windows[hwnd][1])
    assert wc.topmost_app_window() == 0x4, "the fallback passed over Settings"


def test_dialog_caption_minimize_sits_left_of_close_and_minimizes_the_window(qtbot):
    # Setup's caption: minimize + close (no maximize). Seeded defects: leave
    # the minimize button unwired, or place it over the close button -> red.
    from yamabiko.overlay_app.window_chrome import DialogCaption

    w = QWidget()
    qtbot.addWidget(w)
    root = QVBoxLayout(w)
    root.setContentsMargins(0, 0, 0, 0)
    caption = DialogCaption("Yamabiko setup", MIST, minimize=True)
    root.addWidget(caption)
    root.addStretch(1)
    w.resize(880, 200)
    w.show()
    minimize, close = caption.caption_button("min"), caption.caption_button("close")
    assert minimize.accessibleName() == "Minimize"
    assert minimize.x() + minimize.width() == close.x() == w.width() - close.width()
    QTest.mouseClick(minimize, Qt.LeftButton)
    assert w.isMinimized()


def test_app_icon_paints_the_rounded_tile_the_glyph_and_the_ridge_at_64_px(qtbot):
    # The Welcome step's 64 px icon from the 128 px design (assets/icon.svg).
    # Seeded defects: square tile corners, no white 山, or no ridge line ->
    # red.
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPainter

    from yamabiko.overlay_app.window_chrome import paint_app_icon

    img = QImage(64, 64, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    paint_app_icon(p, QRectF(0, 0, 64, 64))
    p.end()
    assert img.pixelColor(1, 1).alpha() == 0
    assert img.pixelColor(32, 2).name() == APP_RED
    assert img.pixelColor(32, 15).name() == "#ffffff"
    assert img.pixelColor(20, 49).name() == "#ffffff"
    assert img.pixelColor(20, 40).name() == APP_RED
