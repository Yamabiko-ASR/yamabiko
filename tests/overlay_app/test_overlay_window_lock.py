"""OverlayWindow construction, style, interactivity (set_interactive) and visibility (toggle_visible, visibleChanged).

set_interactive replaces the retired set_locked with INVERTED semantics
(interactive=True means input-enabled): it runs the same flag/DWM/show
dance but persists NOTHING (placement mode, Task 7, owns persistence via
set_position). The dashed-border paintEvent keys on self._interactive.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from yamabiko.overlay_app.overlay_window import OverlayWindow
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_overlay_window_constructs_with_default_style(qapp):
    w = OverlayWindow()
    assert w._style["jp_size"] == STYLE_DEFAULTS["jp_size"]
    w.close()


def test_overlay_window_accepts_custom_style(qapp):
    custom = {**STYLE_DEFAULTS, "jp_size": 20, "en_size": 30}
    w = OverlayWindow(style=custom)
    assert w._style["jp_size"] == 20
    assert w._style["en_size"] == 30
    w.close()


def test_apply_style_updates_stored_style(qapp):
    w = OverlayWindow()
    new_style = {**STYLE_DEFAULTS, "en_size": 40, "bg_opacity": 255}
    w.apply_style(new_style)
    assert w._style["en_size"] == 40
    assert w._style["bg_opacity"] == 255
    w.close()


def test_apply_style_restyles_existing_lines(qapp):
    """apply_style re-measures the lines already on screen: a larger jp_size
    grows the newest line's Japanese block.

    Seeded defect: SubtitleStack.apply_style keeps the old look (skip
    ``self._look = look``) -> the block keeps its old height -> red."""
    w = OverlayWindow()
    try:
        w.add_subtitle("テスト", "Test")
        before = w.stack.frame().lines[-1].blocks["jp"]
        w.apply_style({**STYLE_DEFAULTS, "jp_size": 28})
        after = w.stack.frame().lines[-1].blocks["jp"]
        assert after.height() > before.height(), (
            f"jp block {before} did not grow after jp_size 28: {after}"
        )
    finally:
        w.close()


# --- interactivity (set_interactive) ---


def test_default_not_interactive(qapp):
    """Fresh window is click-through by default (input disabled): the
    WindowTransparentForInput flag is present and _interactive is False.

    Seeded-defect check: initialise self._interactive = True in __init__
    -> the default window becomes input-capturing -> flag assertion red.
    """
    w = OverlayWindow()
    assert w._interactive is False
    assert bool(w.windowFlags() & Qt.WindowTransparentForInput)
    w.close()


def test_set_interactive_true_drops_transparent_flag(qapp):
    """set_interactive(True) enables input by clearing
    WindowTransparentForInput (so placement mode can drag the stack).

    Seeded-defect check: leave the flag OR'd in for the interactive branch
    -> the flag stays set -> red.
    """
    w = OverlayWindow()
    w.set_interactive(True)
    assert w._interactive is True
    assert not bool(w.windowFlags() & Qt.WindowTransparentForInput)
    w.close()


def test_set_interactive_round_trip_restores_flag(qapp):
    """Enabling then disabling input restores the click-through flag."""
    w = OverlayWindow()
    w.set_interactive(True)
    w.set_interactive(False)
    assert w._interactive is False
    assert bool(w.windowFlags() & Qt.WindowTransparentForInput)
    w.close()


def test_set_interactive_does_not_persist(qapp):
    """set_interactive must NOT fire the position-persist callback: placement
    mode (Task 7) owns persistence via set_position. Enabling/disabling input
    is transient view state, not durable position intent.

    Seeded-defect check: add an on_position_changed call inside
    set_interactive -> captured becomes non-empty -> red.
    """
    captured: list[dict] = []
    w = OverlayWindow(on_position_changed=captured.append)
    w.set_interactive(True)
    w.set_interactive(False)
    assert captured == [], (
        "set_interactive persisted; placement mode owns persistence"
    )
    w.close()


def test_bg_opacity_only_apply_style_does_not_remeasure_lines(qapp, monkeypatch):
    """K17: the debounced live-preview path re-fires apply_style every ~150 ms
    while e.g. the bg-opacity slider moves. A bg_opacity-only change moves no
    glyph, so it must not re-measure the lines on screen (layout_line is the
    expensive text layout); a jp_size change must.

    Seeded-defect check: drop the layout_key cache check in
    SubtitleStack._retarget (re-measure every line on every retarget) -> the
    spy fires on a bg_opacity-only change -> red.
    """
    from yamabiko.overlay_app import telop_stack

    w = OverlayWindow()
    try:
        w.add_subtitle("田中", "Tanaka", jp_ruby=[{"text": "田中", "reading": "たなか"}])
        calls: list = []
        orig = telop_stack.layout_line

        def spy(*a, **kw):
            calls.append((a, kw))
            return orig(*a, **kw)

        monkeypatch.setattr(telop_stack, "layout_line", spy)

        w.apply_style({**STYLE_DEFAULTS, "bg_opacity": 255})
        assert calls == [], (
            "bg_opacity-only apply_style re-measured the lines on screen"
        )

        # The gate must OPEN for a key the text layout derives from.
        w.apply_style({**STYLE_DEFAULTS, "bg_opacity": 255, "jp_size": 28})
        assert calls, "jp_size change no longer re-measures the lines"
    finally:
        w.close()


def test_toggle_visible_flips_the_window_and_visible_changed_reports_each_flip(qapp):
    """Decisions 36 and 64: toggle_visible shows or hides the window and
    visibleChanged reports every real flip exactly once. set_interactive's
    flag change (setWindowFlags hides the window, show() re-shows it)
    reports nothing, so the tray icon and tooltip never flash "Subtitles
    hidden" around placement; a hidden window that placement shows is one
    real flip.

    Seeded defects: emit on every showEvent, never emit on hide, report the
    flag change's hide and show, drop the net report after it, or make
    toggle_visible always hide -> red."""
    w = OverlayWindow()
    seen: list[bool] = []
    w.visibleChanged.connect(seen.append)
    w.show()
    w.toggle_visible()
    assert not w.isVisible()
    w.toggle_visible()
    assert w.isVisible()
    assert seen == [True, False, True]
    w.set_interactive(True)
    w.set_interactive(False)
    assert seen == [True, False, True], f"placement's flag change was reported: {seen}"
    assert w.isVisible()
    w.toggle_visible()
    w.set_interactive(True)          # placement shows the hidden window
    assert seen == [True, False, True, False, True] and w.isVisible()
    w.set_interactive(False)
    w.close()
