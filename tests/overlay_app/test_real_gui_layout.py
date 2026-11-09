"""Real-platform GUI layout tests (@pytest.mark.real).

The offscreen QPA the rest of tests/overlay_app runs under fakes font metrics
and over-allocates aligned layout items, so it CANNOT reproduce two live bugs
that only bite on the real Windows platform:

* Bug A -- OverlayWindow stacked subtitle frames in a QVBoxLayout + addStretch +
  Qt.AlignHCenter. On real Windows an aligned item is sized from its plain
  sizeHint (not heightForWidth), so a frame was drawn narrower than the max cap
  its wrapped-EN height had been computed for: the EN clipped at the frame
  bottom, and adjacent frames could overlap. (Measured pre-fix at en_size 42:
  74 px EN deficit per line.) The telop stack now places every line by its
  measured box.

* Bug B -- the settings preview strip's demand propagates into the settings
  window layout; the window opens at its construction-time sizeHint (smaller)
  and the QFormLayout controls are squeezed below their text height, clipping
  "System / Yu Gothic UI / 30 / #ffffff".
  (Measured pre-fix at en_size 42: controls render 20 px, need 24.)

Both were invisible to the offscreen suite. These tests run against the native
platform. They SKIP unless a real (non-offscreen) QPA is available, so headless
CI stays green; run them explicitly on a workstation:

    py -3.13 -m pytest tests/overlay_app/test_real_gui_layout.py -m real -v
"""
from __future__ import annotations

import os
import re
import sys

import pytest

pytestmark = pytest.mark.real


@pytest.fixture(scope="module")
def real_app():
    """A QApplication on a REAL (non-offscreen) platform, or skip.

    The QPA platform is a per-process singleton fixed when the first
    QApplication is constructed. So: only when NO QApplication exists yet (this
    file run in its own invocation) do we force the native windows platform,
    overriding the tests/overlay_app conftest's offscreen setdefault. When a
    QApplication already exists -- e.g. the full offscreen suite created one
    first -- the platform is already fixed (offscreen) and these tests skip.
    The env override is done HERE, not at module import, precisely so importing
    this module during the offscreen suite's collection cannot flip the whole
    run onto the windows platform.
    """
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None and sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "windows"
    app = QApplication.instance() or QApplication([])
    if app.platformName() == "offscreen":
        pytest.skip(
            "real GUI platform unavailable (QApplication is offscreen); run "
            "this file in its own pytest invocation on a workstation"
        )
    yield app


_LONG_EN = (
    "The big question is whether Prime Minister Takai can really deliver on her "
    "policies like cutting the number of lawmakers and handling the consumption "
    "tax."
)


def test_two_long_lines_at_en42_do_not_overlap_or_clip(real_app):
    """Bug A on the telop stack: two long-EN subtitles at en_size 42 on the
    real platform must stack with NO geometric overlap, and each English
    block must be tall enough for its wrapped rows: rows x row height +
    padding, as layout_line measures it for the same style, cap and age on
    the real font metrics.

    Seeded-defect checks (each goes red on this real tier):
    - drop the ``y -= h`` advance in place_lines -> both lines sit on the
      anchor and intersect -> overlap assert red.
    - size the EN block from one row in telop_text._layout_en (height =
      pad + row_h) -> the wrapped EN clips -> the height assert red.
    """
    from PySide6.QtWidgets import QApplication

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.overlay_window import OverlayWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.telop_look import BANNER_WIDTH
    from yamabiko.overlay_app.telop_text import layout_line

    register_bundled_fonts()
    w = OverlayWindow(style={**STYLE_DEFAULTS, "en_size": 42})
    try:
        w.show()
        QApplication.processEvents()
        w.add_subtitle("こんにちは", _LONG_EN, jp_ruby=[{"text": "こんにちは"}])
        w.add_subtitle("さようなら", _LONG_EN, jp_ruby=[{"text": "さようなら"}])
        QApplication.processEvents()

        lines = w.stack.frame().lines
        assert len(lines) == 2, f"expected 2 stacked lines, got {len(lines)}"
        rects = [box.rect for box in lines]
        assert not rects[0].intersects(rects[1]), (
            f"subtitle lines overlap on the real platform: {rects[0]} "
            f"intersects {rects[1]}"
        )
        look = w.stack.look()
        max_box = w._line_cap_px() - BANNER_WIDTH
        for idx, box in enumerate(lines):
            block = layout_line([{"text": "x"}], _LONG_EN, look, box.age,
                                max_box).blocks[-1]
            assert block.kind == "en"
            # Guard: the EN genuinely wraps here, so the no-clip assert
            # exercises the real wrap path (not a one-row EN that cannot clip).
            assert len(block.rows) > 1, (
                f"line {idx} EN did not wrap at en 42; the clip check is vacuous"
            )
            need = len(block.rows) * block.row_h + block.pad.top + block.pad.bottom
            assert box.blocks["en"].height() == need, (
                f"line {idx} EN box {box.blocks['en'].height()} px != its "
                f"{len(block.rows)} wrapped rows + padding ({need} px)"
            )
    finally:
        w.close()


def _qss_control_v_allowance() -> int:
    """Vertical space the window QSS reserves around a text control: top+bottom
    padding + top+bottom border, DERIVED from theme._WINDOW_QSS (not hardcoded),
    so a padding/border change in theme.py flows into the expectation."""
    from yamabiko.overlay_app import theme

    rule = re.search(
        r"QComboBox,\s*QSpinBox,\s*QDoubleSpinBox,\s*QLineEdit\s*\{([^}]*)\}", theme._WINDOW_QSS
    )
    body = rule.group(1) if rule else ""
    pad = re.search(r"padding:\s*(\d+)px", body)
    border = re.search(r"border:\s*(\d+)px", body)
    return 2 * (int(pad.group(1)) if pad else 0) + 2 * (
        int(border.group(1)) if border else 0
    )


def _assert_page_controls_taller_than_text(page, *, expect_controls: bool = True) -> None:
    """Assert every top-level QComboBox/QSpinBox/QDoubleSpinBox/QLineEdit on
    the page RENDERS at least text height + the QSS vertical allowance (the
    shared no-clip invariant of the Bug-B tests). ``expect_controls`` guards
    against a vacuous pass on a page that must hold text controls."""
    from PySide6.QtWidgets import (
        QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QAbstractSpinBox,
    )
    from PySide6.QtGui import QFontMetrics

    allowance = _qss_control_v_allowance()
    assert allowance > 0, "failed to derive QSS control padding from theme"

    found = (
        page.findChildren(QComboBox)
        + page.findChildren(QSpinBox)
        + page.findChildren(QDoubleSpinBox)
        + page.findChildren(QLineEdit)
    )
    # Drop internal sub-editors: a QSpinBox / editable QComboBox owns a
    # nested QLineEdit whose height is the control's inner area (it does NOT
    # carry the outer QSS padding, so the allowance below doesn't apply to
    # it). Keep only the top-level form fields.
    controls = [
        c for c in found
        if not isinstance(c.parent(), (QComboBox, QAbstractSpinBox))
    ]
    if expect_controls:
        assert controls, "no text controls found on the page"
    for c in controls:
        fm_h = QFontMetrics(c.font()).height()
        need = fm_h + allowance
        assert c.height() >= need, (
            f"{type(c).__name__} rendered {c.height()} px < needed {need} "
            f"(text {fm_h} + QSS allowance {allowance}): the control text "
            f"is vertically clipped"
        )
        assert c.sizeHint().height() >= need, (
            f"{type(c).__name__} sizeHint {c.sizeHint().height()} < {need}"
        )


def test_settings_pages_fit_at_the_design_size_and_keep_the_footer_when_short(real_app):
    """Real fonts. At 1000 x 728 no page scrolls and every page's text
    controls render at least their text height; large subtitle fonts (EN 42,
    JP 30) inflate nothing (the monitor clips its sample instead). At the
    window's minimum height (a short work area, Decision 2) the monitor is
    at its minimum, the panels may scroll and the footer buttons stay inside
    the window.

    Seeded defects: let a page's panels outgrow the 266 px budget -> the
    panel scroll bar has a range at 728 px -> red; drop the explicit window
    minimum (the ~690 px layout floor returns) -> the minimum assert red;
    let the footer shrink with the window -> a button leaves the window ->
    red."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.preview_monitor import MONITOR_H, MONITOR_MIN_H
    from yamabiko.overlay_app.settings_window import WINDOW_H, WINDOW_MIN_H, SettingsWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    register_bundled_fonts()
    w = SettingsWindow({**STYLE_DEFAULTS, "en_size": 42, "jp_size": 30})
    try:
        w.show()
        QTest.qWait(150)
        if w.screen().geometry().height() < WINDOW_H:
            pytest.skip("this screen is shorter than the 728 px design height")
        w.resize(w.width(), WINDOW_H)  # undo a first-show fit to a short work area
        QApplication.processEvents()
        assert w._preview.height() == MONITOR_H
        bar = w._panel_scroll.verticalScrollBar()
        for index in range(w._stack.count()):
            w._tabs.setCurrentIndex(index)
            QApplication.processEvents()
            assert bar.maximum() == 0, f"page {index} scrolls at the design size"
            _assert_page_controls_taller_than_text(
                w._stack.widget(index), expect_controls=index < 4)
        assert w.minimumHeight() == WINDOW_MIN_H
        w.resize(w.width(), WINDOW_MIN_H)
        QApplication.processEvents()
        assert w.height() == WINDOW_MIN_H and w._preview.height() == MONITOR_MIN_H
        window_rect = QRect(QPoint(0, 0), w.size())
        for button in (w._apply_btn, w._ok_btn, w._cancel_btn):
            rect = QRect(button.mapTo(w, QPoint(0, 0)), button.size())
            assert window_rect.contains(rect), f"{button.text()!r} at {rect.getRect()}"
    finally:
        w.close()


def test_real_timing_lines_row_splits_its_width_equally(real_app):
    """Real windows11 metrics. The mockup's Lines row has two equal columns;
    the double spin's wider size hint took more than its half (110 / 125 px).
    The offscreen style's hints fit inside the halves, so it cannot show this.

    Seeded defect: drop the Ignored width policy in SettingsWindow._pair ->
    the Time on screen field is wider -> red."""
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    register_bundled_fonts()
    w = SettingsWindow(dict(STYLE_DEFAULTS), list_devices=lambda: [], page="timing")
    try:
        w.show()
        QTest.qWait(150)
        widths = (w._max_lines.width(), w._hold.width())
    finally:
        w.close()
    assert abs(widths[0] - widths[1]) <= 1, widths


def test_real_shortcut_caps_fit_their_fields_after_a_theme_switch(real_app):
    """Real font metrics. A live theme switch re-polishes the window and
    puts the fonts back field first: each field sized its key caps while
    they still held the app font (Segoe UI), then the caps got the window's
    Segoe UI Variable Text back and painted 2 px wider than their widget,
    so the last cap of every row lost its right edge. The offscreen QPA
    has one fallback face for both families, so it can't show this.

    Seeded defect: ShortcutField ignores its caps' font changes -> each
    row's caps widget is narrower than its caps -> red."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtTest import QTest

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.settings_window import SettingsWindow
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

    register_bundled_fonts()
    w = SettingsWindow(dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       hotkey_probe=lambda combo: False, page="shortcuts")
    try:
        w.show()
        QTest.qWait(150)
        for theme in ("dark", "light"):
            w._theme_choice._segment_for(theme).click()
            QTest.qWait(100)
        rows = {key: (QRect(field._caps.mapTo(field, QPoint(0, 0)), field._caps.size()),
                      field._caps.sizeHint().width(), field.rect())
                for key, field in w._shortcut_fields.items()}
    finally:
        w.close()
    clipped = {key: (caps.getRect(), need) for key, (caps, need, field) in rows.items()
               if caps.width() < need or not field.contains(caps)}
    assert len(rows) == 3 and not clipped, clipped


def test_real_telop_renders_bundled_fonts_and_vertical_banner(real_app):
    """Real Windows platform: the telop look resolves the bundled BIZ
    UDPGothic Bold for Japanese, the banner's vertical alternates turn ー
    upright (taller than wide), and a long English line wraps inside the
    cap. Covers the new PySide6 call sites (pixel sizes, CSS weights, the
    "vert" font feature, QTextLayout wrapping) on the shipping platform.

    Seeded defects: drop setFeature("vert") -> ー stays horizontal -> red;
    drop setWeight -> regular weight -> red."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QFontInfo, QPainterPath

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.telop_banner import banner_content
    from yamabiko.overlay_app.telop_look import look_from_style
    from yamabiko.overlay_app.telop_text import layout_line, make_font

    register_bundled_fonts()
    look = look_from_style(STYLE_DEFAULTS)
    info = QFontInfo(make_font(look.jp))
    assert info.family() == "BIZ UDPGothic" and info.weight() >= 700

    content = banner_content("ニュース", "News", False, False)
    path = QPainterPath()
    path.addText(QPointF(0, 0), content.upright_font, "ー")
    box = path.boundingRect()
    assert box.height() > 2 * box.width(), f"ー not vertical: {box}"

    block = layout_line([], _LONG_EN, look, 0, 600).blocks[0]
    assert len(block.rows) > 1 and block.width <= 600


def test_real_echo_text_has_no_cleartype_color_fringes(real_app):
    """Real Windows platform: an echo line's white text over its black
    outline is grayscale-antialiased. Glyphs drawn straight onto opaque
    pixels take ClearType's subpixel path and leave orange and blue fringes
    on every stem (the approved design and the browser page have none).

    Control: the same text drawn white straight onto opaque black must
    fringe on this machine; when it does not, ClearType is off and the echo
    check would pass vacuously, so the test skips.

    Seeded defect: paint the echo fill straight onto the outlined image in
    telop_text.render_block -> hundreds of fringed pixels -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.telop_look import look_from_style
    from yamabiko.overlay_app.telop_text import layout_line, render_block

    register_bundled_fonts()
    text = "Come on, this is what the town is famous for."
    look = look_from_style(STYLE_DEFAULTS)
    block = layout_line([], text, look, 1, 900).blocks[0]

    def opaque_pixels(image):
        """(fringed, white): opaque pixels with a channel spread >= 50, and
        opaque white ones (the fill was drawn)."""
        fringed = white = 0
        for y in range(image.height()):
            for x in range(image.width()):
                c = image.pixelColor(x, y)
                if c.alpha() < 255:
                    continue
                rgb = (c.red(), c.green(), c.blue())
                fringed += max(rgb) - min(rgb) >= 50
                white += min(rgb) > 240
        return fringed, white

    control = QImage(block.width, block.height, QImage.Format.Format_ARGB32_Premultiplied)
    control.fill(QColor(0, 0, 0))
    painter = QPainter(control)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(block.font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(control.rect(), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    if opaque_pixels(control)[0] == 0:
        pytest.skip("ClearType is off on this machine")

    fringed, white = opaque_pixels(render_block(block, look, 1.0))
    assert white > 0, "the echo block painted no white fill"
    assert fringed == 0, f"{fringed} opaque echo pixels carry ClearType color fringes"


def test_real_notice_line_uses_the_english_face_and_elides_at_the_cap(real_app):
    """Real Windows platform, Learn mode: the notice text resolves the Watch
    English face (Bahnschrift) at semibold 24 px, not Learn's Segoe UI
    Variable Text, and a long line is elided with … inside the line cap
    (banner and padding included) on the real font metrics. Covers the new
    PySide6 call sites: the notice font, QFontMetricsF.elidedText and the
    text layer.

    Seeded defects: take Learn's English face (look.en), drop the weight,
    or elide against the whole cap -> red."""
    from PySide6.QtGui import QFontInfo

    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
    from yamabiko.overlay_app.telop_stack import SubtitleStack

    register_bundled_fonts()
    stack = SubtitleStack(style={**STYLE_DEFAULTS, "mode": "learn"})
    try:
        stack.resize(1920, 1080)
        stack.set_anchor(100, 950, 700, 1920, 0, 1080)
        line = "The translator isn't responding. Restarting it\u2026"
        stack.set_notices([("translator", line * 3, False)])
        notice = stack.frame().notices[0]
        font = stack._notices[0].font
        info = QFontInfo(font)
        assert info.family() == "Bahnschrift" and info.weight() >= 600
        assert font.pixelSize() == 24
        assert notice.text.endswith("\u2026") and len(notice.text) < len(line * 3)
        assert 650 < notice.rect.width() <= 700
    finally:
        stack.deleteLater()
