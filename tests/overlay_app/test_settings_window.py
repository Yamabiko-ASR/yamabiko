"""Tests for the SettingsWindow ("Studio under the mountain").

The persist-helper tests are ported from the retired
test_settings_dialog.py (the helpers moved verbatim from settings_dialog.py
to settings_window.py). Construction is adapted -- the subject is now a
non-modal QWidget, so the OK/Cancel "did it close" assertions observe the
closeEvent-driven ``_closed`` flag instead of QDialog.result().

New tests cover the window-only behaviour: tab switching, the 150 ms
preview debounce, the real SubtitleStack preview monitor, the Apply style
dict carrying max_width_pct, the staged Settings transaction, the design
size and the short-work-area fit.
"""
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFrame

from yamabiko.overlay_app.settings_window import SettingsWindow
from yamabiko.overlay_app.telop_stack import SubtitleStack
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from tests.overlay_app._qt_drain import drain_qt_debris


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _drain_qt_objects():
    """Destroy each test's Qt debris (notably the real download QThreads spun by
    the quant-tier cases) within its own teardown -- see
    tests/overlay_app/_qt_drain.py for the native-teardown-flake rationale."""
    yield
    drain_qt_debris()


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    """Isolate the sparse-write target at a tmp_path user_overrides.yaml."""
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def _cancel_download_on_failure(monkeypatch):
    """The download dialog now waits on its failed state (Retry offered);
    a test that drives a failing download closes it from there, as a user
    choosing Cancel would."""
    from PySide6.QtCore import QTimer
    from yamabiko.overlay_app import model_fetch

    real = model_fetch.QuantDownloadDialog._on_queue_finished

    def _finished_then_cancel(self, ok):
        real(self, ok)
        if self._failed:
            QTimer.singleShot(0, self.cancel_button.click)

    monkeypatch.setattr(model_fetch.QuantDownloadDialog, "_on_queue_finished", _finished_then_cancel)


def _read_overrides(tmp_path):
    import yaml
    p = tmp_path / "user_overrides.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Construction + style reflection (ported)
# ---------------------------------------------------------------------------


def test_window_constructs_with_defaults(qapp):
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert w._collect()["jp_size"] == STYLE_DEFAULTS["jp_size"]


def test_window_reflects_current_style(qapp):
    w = SettingsWindow(current={**STYLE_DEFAULTS, "jp_size": 40, "bg_opacity": 128})
    assert w._jp_size.value() == 40
    assert w._bg_opacity.value() == 50  # percent of the stored 128


def test_reset_populates_defaults(qapp):
    w = SettingsWindow(current={**STYLE_DEFAULTS, "jp_size": 40})
    assert w._jp_size.value() == 40
    w._reset()
    assert w._jp_size.value() == STYLE_DEFAULTS["jp_size"]


def test_reset_with_only_color_customized_still_fires_style_preview(qapp):
    """S3: Qt controls don't emit valueChanged when set to their CURRENT value,
    and the colour buttons aren't wired to any change signal -- so when the only
    customization was a colour, _reset's setValue pass fires nothing and the
    live overlay sample kept the pre-Reset style. _reset must arm the preview
    debounce unconditionally so stylePreview always carries the defaults out.

    Seeded defect: drop the unconditional ``self._on_style_control_changed()``
    from _reset (back to the bare ``self._rebuild_preview()``) -> no control
    fires (every numeric is already at its default) -> no stylePreview -> red.
    """
    from PySide6.QtTest import QTest

    w = SettingsWindow(
        current={**STYLE_DEFAULTS, "text_color": "#ff0000"},  # colour-only custom
        list_devices=lambda: [],
    )
    # On a machine with the default family installed, reset's setCurrentFont is
    # a genuine no-op. The offscreen test platform has an EMPTY font-combo
    # model, so setCurrentFont churns (emits) on every call and would mask the
    # defect -- silence the two font combos to reproduce the production case.
    w._jp_font.blockSignals(True)
    w._en_font.blockSignals(True)
    emitted: list[dict] = []
    w.stylePreview.connect(lambda d: emitted.append(d))

    w._reset()
    QTest.qWait(250)  # let the debounce fire

    assert emitted, (
        "Reset with a colour-only customization never fired stylePreview; the "
        "live overlay sample keeps the stale colour"
    )
    assert emitted[-1]["text_color"] == STYLE_DEFAULTS["text_color"]


def test_window_seed_clamps_garbage_style(qapp):
    """The style seed passes through config._clamp_style at construction, so a
    garbage value (max_width_pct:None, padding:"bad") that reaches the numeric
    seeds in the page builders would otherwise raise int(None) /
    QSpinBox.setValue(None). This defends a direct construction from a raw or
    untrusted style dict (belt for the tray's ingestion clamp).

    Seeded defect (one line): revert ``self._working = _clamp_style(dict(current))``
    to ``self._working = dict(current)`` -> int(self._working['max_width_pct'])
    in _build_subtitles_page raises TypeError on None -> construction raises red;
    and the clamped-value assertions below read the garbage -> red.
    """
    w = SettingsWindow(current={**STYLE_DEFAULTS, "max_width_pct": None, "padding": "bad"})
    assert w._max_width.value() == 70
    assert w._padding.value() == 10
    assert w._collect()["max_width_pct"] == 70


def test_apply_emits_signal(qapp):
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    emitted: list[dict] = []
    w.styleChanged.connect(lambda s: emitted.append(s))
    w._jp_size.setValue(30)
    w._apply()
    assert len(emitted) == 1
    assert emitted[0]["jp_size"] == 30


def test_unchanged_substituted_font_is_saved_as_configured_family(qapp):
    """A configured font that is not installed makes QFontComboBox show a
    substitute family. An Apply that only touched another control used to
    save that substitute over the user's configured jp_font / en_font.

    Seeded defect: make _collect read currentFont().family() unconditionally
    (drop the seeded-family check) -> the commit carries the substitute ->
    red.
    """
    w = SettingsWindow(
        current={**STYLE_DEFAULTS, "jp_font": "Yamabiko Missing JP Font",
                 "en_font": "Yamabiko Missing EN Font"},
        list_devices=lambda: [],
    )
    assert w._jp_font.currentFont().family() != "Yamabiko Missing JP Font", (
        "precondition: the combo must show a substitute for a missing family"
    )
    w._jp_size.setValue(w._jp_size.value() + 1)  # an unrelated edit
    updates = w._collect_commit().updates
    assert updates[("overlay", "style", "jp_font")] == "Yamabiko Missing JP Font", (
        f"substitute saved over the configured JP family: "
        f"{updates[('overlay', 'style', 'jp_font')]!r}"
    )
    assert updates[("overlay", "style", "en_font")] == "Yamabiko Missing EN Font", (
        f"substitute saved over the configured EN family: "
        f"{updates[('overlay', 'style', 'en_font')]!r}"
    )


def test_audio_device_picker_present(qapp):
    def fake_list_devices() -> list[dict]:
        return [
            {"index": 0, "name": "Speakers (Realtek) [Loopback]"},
            {"index": 1, "name": "USB Audio [Loopback]"},
        ]

    w = SettingsWindow(list_devices=fake_list_devices)
    assert w.audio_device_combo.count() == 2


# ---------------------------------------------------------------------------
# NEW: window-only behaviour
# ---------------------------------------------------------------------------


def test_tabs_switch_pages(qapp):
    """Selecting a tab must switch the QStackedWidget to that page.

    Seeded defect: drop the ``currentChanged -> setCurrentIndex`` connect
    in _connect_change_signals -> the stack stays on page 0 -> red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert w._stack.currentIndex() == 0
    for i in (1, 2, 3, 0):
        w._tabs.setCurrentIndex(i)
        assert w._stack.currentIndex() == i


def test_style_control_change_debounces_to_single_preview(qapp):
    """Two rapid style-control changes must collapse into exactly ONE
    stylePreview emission carrying the merged (latest) dict.

    Seeded defect (drill a): replace ``self._preview_timer.start()`` with
    ``self._emit_preview()`` in _on_style_control_changed (emit per change)
    -> two emissions -> len(emitted) == 2 -> red.
    """
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    emitted: list[dict] = []
    w.stylePreview.connect(lambda d: emitted.append(d))
    w._jp_size.setValue(20)
    w._jp_size.setValue(30)
    QTest.qWait(250)
    assert len(emitted) == 1, (
        f"expected exactly one debounced preview, got {len(emitted)}"
    )
    assert emitted[0]["jp_size"] == 30


def test_preview_strip_embeds_real_stack_and_rebuilds(qapp):
    """The preview strip must hold a REAL SubtitleStack (not a stand-in), and
    an appearance-control change must rebuild its sample lines once the
    150 ms debounce fires: a bigger jp_size grows the newest line.

    Seeded defect: drop the ``self._rebuild_preview()`` call from
    _emit_preview (the debounced rebuild) -> the newest line keeps its old
    height on a control change -> red.
    """
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert isinstance(w._preview.stack, SubtitleStack)
    before = w._preview.stack.frame().lines[-1].rect.height()
    w._jp_size.setValue(w._jp_size.value() + 5)
    QTest.qWait(250)  # let the debounced rebuild fire
    after = w._preview.stack.frame().lines[-1].rect.height()
    assert after > before, (
        f"an appearance-control change must rebuild the preview lines; the "
        f"newest line stayed {before} -> {after} px tall"
    )


def test_preview_line_is_fully_opaque_not_invisible(qapp):
    """The preview stack must render its sample at FULL opacity at once: the
    paper strip (#f3eee3) behind the newest Japanese is on screen in the
    first render. The preview never animates, so a stack left on the
    overlay's motion path would still be fading its plates in (strip
    background only). (The monitor's photo behind the sample is pinned in
    test_preview_monitor.py.)

    Seeded defects (each goes red): drop the set_anchor call in
    PreviewMonitor._place -> the sample lines sit above the monitor -> no
    paper pixels in the render; put the preview on the motion path (drop
    ``self.stack.set_reduce_motion(True)`` from PreviewMonitor.__init__;
    ``instant=True`` lands only the sample lines, not the plates) -> the
    plates start their fade-in at opacity 0 -> no paper pixels.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage, QRegion
    from PySide6.QtWidgets import QWidget
    from yamabiko.overlay_app.theme import MIST

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    try:
        w.apply_palette(MIST)
        w.show()
        qapp.processEvents()
        stack = w._preview.stack
        assert stack.frame().lines, "the preview holds no sample line"
        img = QImage(stack.size(), QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        stack.render(img, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
        paper = 0xFFF3EEE3
        found = sum(
            1
            for y in range(0, img.height(), 2)
            for x in range(0, img.width(), 2)
            if img.pixel(x, y) == paper
        )
        assert found > 0, "the rendered preview has no paper strip pixels"
    finally:
        w.close()


def test_preview_monitor_clamps_out_of_range_max_width_pct(qapp):
    """K1 (preview side): an out-of-range max_width_pct (e.g. 150 from a
    loopback WS frame) must not render a cap wider than the overlay's own
    clamp (30..100): rebuild routes through
    style_defaults.coerce_max_width_pct (the one sanitizer for the key), and
    the cap stays a positive width inside the monitor's room (564 - 28 px).

    Seeded defects: drop the room bound from PreviewMonitor._cap -> pct=150
    caps at the 1000 px screen width -> red; skip coerce_max_width_pct ->
    pct=5 caps at 50 px instead of the 30 floor's 300 -> red.
    """
    from yamabiko.overlay_app.preview_monitor import MONITOR_H, MONITOR_W, PreviewMonitor

    monitor = PreviewMonitor(dict(STYLE_DEFAULTS), screen_width=1000)
    monitor.resize(MONITOR_W, MONITOR_H)
    monitor.show()
    qapp.processEvents()
    try:
        monitor.rebuild({**STYLE_DEFAULTS, "max_width_pct": 150})
        assert monitor.stack._cap == MONITOR_W - 28, (
            f"pct=150 must clamp to the 100 ceiling and the monitor's room; got "
            f"{monitor.stack._cap}"
        )
        monitor.rebuild({**STYLE_DEFAULTS, "max_width_pct": 5})
        assert monitor.stack._cap == int(1000 * 30 / 100), (
            f"pct=5 must clamp to the 30 floor; got {monitor.stack._cap}"
        )
    finally:
        monitor.close()


def test_settings_window_builds_preview_line_exactly_once(qapp, monkeypatch):
    """SettingsWindow.__init__ must build the preview lines exactly ONCE.

    The old strip built in its own __init__ AND again in
    SettingsWindow._rebuild_preview -- two builds per window open, the first
    immediately discarded.

    Seeded defect: add a ``self.rebuild(style)`` call at the end of
    PreviewMonitor.__init__ -> the count is 2 -> red.
    """
    calls = _count_preview_rebuilds(monkeypatch)
    SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    assert calls["n"] == 1, (
        f"expected exactly one preview build during __init__, got "
        f"{calls['n']}"
    )


def _count_preview_rebuilds(monkeypatch):
    """Count preview rebuild passes (each restyles the stack once via
    SubtitleStack.apply_style); return the counter."""
    calls = {"n": 0}
    orig = SubtitleStack.apply_style

    def _counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(SubtitleStack, "apply_style", _counting)
    return calls


def test_two_rapid_appearance_changes_rebuild_strip_once_after_debounce(
    qapp, monkeypatch,
):
    """The preview strip rebuild is DEBOUNCED: it used to run synchronously on
    every control tick (a slider drag re-measured the sample lines dozens of times
    per second). Two rapid jp_size changes must rebuild the strip exactly ONCE,
    and only after the 150 ms debounce fires -- not on the control ticks.

    Seeded defect: restore the synchronous ``self._rebuild_preview()`` in
    _on_style_control_changed -> the strip rebuilds on each tick, so the count is
    2 immediately (before qWait) -> the ``== 0`` assertion red; or drop the
    ``self._rebuild_preview()`` from _emit_preview -> after qWait the count is 0
    -> the ``== 1`` assertion red.
    """
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    calls = _count_preview_rebuilds(monkeypatch)

    w._jp_size.setValue(w._jp_size.value() + 2)
    w._jp_size.setValue(w._jp_size.value() + 2)
    assert calls["n"] == 0, (
        f"the strip rebuilt synchronously on the control tick; rebuild must be "
        f"debounced. constructions before qWait={calls['n']}"
    )

    QTest.qWait(250)
    assert calls["n"] == 1, (
        f"two rapid appearance changes must debounce to ONE strip rebuild; got "
        f"{calls['n']} rebuilds"
    )


def test_timing_control_change_does_not_rebuild_the_preview_strip(
    qapp, monkeypatch,
):
    """A timing control (hold_ms) does NOT change the rendered preview line, so it
    must not rebuild the strip -- zero rebuilds across the whole
    debounce window.

    Seeded defect: wire hold to _on_style_control_changed (appearance) instead of
    _on_timing_control_changed -> the change marks a rebuild pending -> after qWait
    the strip rebuilds -> the count is 1 -> red.
    """
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    calls = _count_preview_rebuilds(monkeypatch)

    w._hold.setValue(w._hold.value() + 0.5)  # seconds
    QTest.qWait(250)
    assert calls["n"] == 0, (
        f"a timing control (hold_ms) rebuilt the preview strip; timing controls "
        f"don't change the rendered line and must not rebuild it. got {calls['n']}"
    )


def test_construction_reads_merged_config_exactly_once(qapp, monkeypatch):
    """Opening the window must read the merged config exactly ONCE. Six
    _read_current_* seed helpers used to each call load_full_config -> six disk
    reads per open; they now share one boot cfg loaded in __init__.

    Seeded defect: revert any _read_current_* seed call site to pass no cfg (so
    that helper re-loads) -> the count climbs past 1 -> red.
    """
    import yamabiko.config as _config

    calls = {"n": 0}
    real = _config.load_full_config

    def _counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(_config, "load_full_config", _counting)

    SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])

    assert calls["n"] == 1, (
        f"SettingsWindow construction read the merged config {calls['n']} times; "
        f"it must load once and share the boot cfg across the seed helpers"
    )


def test_apply_emits_style_with_max_width_pct(qapp):
    """Apply must emit styleChanged carrying the Max-width slider value under
    ``max_width_pct`` (the key the pipeline + overlay consume).

    Seeded defect (drill b): drop ``max_width_pct`` from _collect -> the key
    is absent from the emitted dict -> KeyError on the assertion -> red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    emitted: list[dict] = []
    w.styleChanged.connect(lambda s: emitted.append(s))
    w._max_width.setValue(55)
    w._apply()
    assert len(emitted) == 1
    assert emitted[0]["max_width_pct"] == 55


def test_pct_slider_rows_share_ranges_and_keep_labels_live(qapp):
    """K10 guard: the two percent rows (Max width / Vertical offset) come from
    one row factory -- their ranges must be the canonical constants (not
    re-typed literals) and their live labels must track the slider."""
    from yamabiko.overlay_app.position_config import V_OFFSET_RANGE
    from yamabiko.overlay_app.style_defaults import MAX_WIDTH_PCT_RANGE

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    assert (w._max_width.minimum(), w._max_width.maximum()) == MAX_WIDTH_PCT_RANGE
    assert (w._voffset.minimum(), w._voffset.maximum()) == tuple(
        int(b) for b in V_OFFSET_RANGE
    )
    w._max_width.setValue(55)
    assert w._max_width_label.text() == "55% of screen width"
    w._voffset.setValue(42)
    assert w._voffset_label.text() == "42% down the screen"


def test_theme_choice_previews_live_then_persists_and_emits_on_apply(qapp, appdata):
    """Theme previews locally, but committed state crosses Apply atomically."""
    from yamabiko.overlay_app import theme as _theme

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    themed: list[str] = []
    w.themeChanged.connect(lambda v: themed.append(v))

    w._theme_choice._segment_for("light").click()
    assert _read_overrides(appdata) == {}
    assert themed == []
    assert _theme.MIST["ground"] in w.styleSheet(), (
        "light theme did not re-skin the window with the MIST ground hex"
    )
    assert w._apply() is True
    assert _read_overrides(appdata).get("overlay", {}).get("theme") == "light"
    assert themed == ["light"]

    w._theme_choice._segment_for("dark").click()
    assert _read_overrides(appdata).get("overlay", {}).get("theme") == "light"
    assert _theme.NIGHT["ground"] in w.styleSheet()
    assert w._apply() is True
    assert _read_overrides(appdata).get("overlay", {}).get("theme") == "dark"
    assert themed == ["light", "dark"]


def test_apply_palette_reskins_from_given_palette_not_persisted_theme(
    qapp, appdata,
):
    """SettingsWindow.apply_palette must re-skin from the PASSED palette, not
    re-derive from the window's own persisted theme -- that is what lets the
    tray's live OS-follow repaint the open window with the OS-resolved palette.

    Persist theme=dark so the window boots NIGHT (its own value would keep
    NIGHT), then apply the MIST (light) palette directly: the window's
    stylesheet must carry MIST's ground hex, not NIGHT's.

    Seeded defect (drill b): make apply_palette ignore its arg and re-derive
    from self._theme_value (dark -> NIGHT) -> MIST ground absent / NIGHT
    ground present -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT
    from yamabiko.config_writer import write_override

    write_override(["overlay", "theme"], "dark")

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert w._theme_value == "dark"
    assert NIGHT["ground"] in w.styleSheet(), "window did not boot the dark palette"

    w.apply_palette(MIST)
    css = w.styleSheet()
    assert MIST["ground"] in css, (
        "apply_palette did not re-skin from the PASSED palette (MIST ground missing)"
    )
    assert NIGHT["ground"] not in css, (
        "apply_palette left the persisted-theme (NIGHT) hex behind; it "
        "re-derived from self._theme_value instead of using the given palette"
    )


def test_theme_switch_keeps_every_widget_font(qapp, appdata):
    """A live theme change re-sets the window QSS, and Qt 6.11's re-polish
    resolves each font property a widget doesn't mark as its own against
    the app font (12 px). Every widget keeps its font across the window's
    own theme picks and the tray's OS-follow re-skin (it calls apply_palette
    with the OS palette).

    Seeded defects: apply_palette calls setStyleSheet(window_qss(...))
    directly -> labels and fields drop to the app font -> red; the restore
    leaves inherited family and size unmarked -> the primary OK button (its
    only own property is the QSS weight) drops to the app font -> red."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QWidget
    from yamabiko.overlay_app.theme import MIST, NIGHT

    def described(widget: QWidget, font: QFont) -> tuple:
        name = widget.objectName() or getattr(widget, "text", lambda: "")()
        return (type(widget).__name__, name, font.families(), font.pixelSize(),
                font.pointSizeF(), font.weight())

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    w.show()
    try:
        widgets = w.findChildren(QWidget)
        before = [QFont(widget.font()) for widget in widgets]
        for theme in ("dark", "light"):
            w._theme_choice._segment_for(theme).click()
        for palette in (NIGHT, MIST):
            w.apply_palette(palette)
        changed = [(described(widget, font), described(widget, widget.font()))
                   for widget, font in zip(widgets, before) if widget.font() != font]
    finally:
        w.close()
    assert len(widgets) > 100 and not changed, changed


def test_reduce_motion_previews_live_then_persists_on_apply(qapp, appdata):
    """Reduce motion persists only on Apply (the window itself is static)."""
    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    w._reduce_motion_cb.setChecked(True)
    assert _read_overrides(appdata) == {}
    assert w._apply() is True
    assert _read_overrides(appdata).get("overlay", {}).get("reduce_motion") is True


def test_theme_change_without_config_path_skips_persist_but_still_reskins(
    qapp, appdata,
):
    """P1-12: a unit-test-constructed window (config_path=None) must NOT persist
    the theme. write_override always targets paths.user_overrides_path(), so an
    ungated write would mutate the developer's REAL overrides file merely by
    constructing the window and nudging the combo. The live reskin + themeChanged
    signal fires only after Apply -- no committed signal is emitted by a draft.

    appdata redirects the write target to tmp_path, so a missing gate leaves the
    stray file THERE (still proving the write ran) instead of the real config.

    Seeded defect: drop the ``if self._config_path is not None:`` gate in
    _on_theme_changed -> write_override runs -> user_overrides.yaml is created
    carrying overlay.theme -> the not-exists assertion red.
    """
    from yamabiko.overlay_app import theme as _theme

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))  # config_path defaults None
    assert w._config_path is None
    themed: list[str] = []
    w.themeChanged.connect(lambda v: themed.append(v))

    w._theme_choice._segment_for("light").click()

    assert not (appdata / "user_overrides.yaml").exists(), (
        "theme change with no config_path wrote user_overrides.yaml; the "
        "config_path production gate is missing from _on_theme_changed"
    )
    # The local preview still runs, but no committed signal fires yet.
    assert _theme.MIST["ground"] in w.styleSheet(), (
        "the live window reskin did not run without a config_path"
    )
    assert themed == []
    assert w._apply() is True
    assert themed == ["light"], "themeChanged did not fire without a config_path"


def test_reduce_motion_without_config_path_emits_only_after_apply(
    qapp, appdata,
):
    """P1-12 sibling: a reduce-motion toggle with config_path=None must NOT write
    to disk, and reduceMotionChanged stays Apply-gated.

    Seeded defect: drop the ``if self._config_path is not None:`` gate in
    _on_reduce_motion_toggled -> write_override runs -> user_overrides.yaml is
    created carrying overlay.reduce_motion -> the not-exists assertion red;
    emit reduceMotionChanged from the toggle -> the pre-Apply emission is
    recorded -> red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))  # config_path defaults None
    assert w._config_path is None
    emitted: list[bool] = []
    w.reduceMotionChanged.connect(emitted.append)
    w._reduce_motion_cb.setChecked(True)
    assert not (appdata / "user_overrides.yaml").exists(), (
        "reduce-motion toggle with no config_path wrote user_overrides.yaml; "
        "the config_path production gate is missing"
    )
    assert emitted == []
    assert w._apply() is True
    assert emitted == [True], "reduceMotionChanged did not fire without a config_path"


def test_auto_genre_toggle_persists_and_emits_only_on_apply(qapp, appdata):
    """Auto-genre remains a draft until the Settings transaction commits."""
    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    assert w._auto_genre_cb.isChecked() is True  # shipped default: on
    emitted: list[bool] = []
    w.autoGenreChanged.connect(lambda v: emitted.append(v))

    w._auto_genre_cb.setChecked(False)

    assert _read_overrides(appdata) == {}
    assert emitted == []
    assert w._apply() is True
    assert _read_overrides(appdata).get("gazetteer", {}).get("auto_genre") is False, (
        "unchecking auto-genre did not persist gazetteer.auto_genre=False"
    )
    assert emitted == [False], (
        f"autoGenreChanged did not carry the toggle to the tray; got {emitted}"
    )


def test_auto_genre_toggle_without_config_path_emits_only_after_apply(
    qapp, appdata,
):
    """P1-12 sibling: an auto-genre toggle with config_path=None must NOT
    write to disk (a unit-test-constructed window would otherwise mutate the
    developer's real user_overrides.yaml), and autoGenreChanged remains
    Apply-gated.

    Seeded defect: drop the ``if self._config_path is not None:`` gate in
    _on_auto_genre_toggled -> write_override runs -> user_overrides.yaml is
    created carrying gazetteer.auto_genre -> the not-exists assertion red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))  # config_path defaults None
    assert w._config_path is None
    emitted: list[bool] = []
    w.autoGenreChanged.connect(lambda v: emitted.append(v))

    w._auto_genre_cb.setChecked(False)

    assert not (appdata / "user_overrides.yaml").exists(), (
        "auto-genre toggle with no config_path wrote user_overrides.yaml; "
        "the config_path production gate is missing"
    )
    assert emitted == []
    assert w._apply() is True
    assert emitted == [False], "autoGenreChanged did not fire without a config_path"


@pytest.mark.parametrize("off_flag", ["gazetteer", "romanization"])
def test_auto_genre_row_absent_when_series_feature_disabled(
    qapp, appdata, off_flag,
):
    """With the series feature off (feature_enabled: gazetteer.enabled AND
    romanization.enabled -- either flag false disables it), the auto-genre
    switch row must not be built: there is no palette, no 'Change series'
    item, and the pipeline rejects set_series at the wire, so the switch
    would persist gazetteer.auto_genre while controlling nothing. Absent, not
    disabled -- the same convention as the tray's series rows.

    Seeded-defect drill: drop the feature gate in _build_subtitles_page
    (build the switch unconditionally) -> the findChildren scan finds it ->
    red (this IS the pre-fix code; confirmed red against it).
    """
    from yamabiko.config_writer import write_override
    from yamabiko.overlay_app.controls import Switch

    write_override([off_flag, "enabled"], False)

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    anime_boxes = [
        cb for cb in w.findChildren(Switch) if "anime" in cb.text().lower()
    ]
    assert anime_boxes == [], (
        f"{off_flag}.enabled=false but the settings window still built the "
        f"auto-genre checkbox: {[cb.text() for cb in anime_boxes]}"
    )
    assert w._auto_genre_cb is None, (
        "the auto-genre checkbox ref must be None when the series feature "
        "is off, so nothing can wire or toggle it"
    )


def test_auto_genre_checkbox_seeds_from_persisted_override(qapp, appdata):
    """A persisted gazetteer.auto_genre=False must seed the checkbox
    unchecked on the next window open. This also pins the default_config.yaml
    key itself: load_full_config STRIPS override keys unknown to shipped
    defaults, so without ``gazetteer.auto_genre`` in default_config.yaml the
    persisted False would be dropped on load and the checkbox would re-seed
    checked.

    Seeded defects (one line each): delete the ``auto_genre: true`` line from
    default_config.yaml -> the override is stripped as unknown -> red;
    hardcode the ctor seed to True (ignore _read_current_auto_genre) -> red.
    """
    from yamabiko.config_writer import write_override

    write_override(["gazetteer", "auto_genre"], False)

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert w._auto_genre_cb.isChecked() is False, (
        "a persisted gazetteer.auto_genre=False did not seed the checkbox "
        "unchecked (override stripped as unknown, or the seed is hardcoded)"
    )


def test_anilist_username_field_seeds_from_persisted_override(qapp, appdata):
    """A persisted gazetteer.anilist_username must seed the "AniList
    username" field on the next window open. This also pins the
    default_config.yaml key itself: load_full_config STRIPS override keys
    unknown to shipped defaults, so without ``anilist_username: ""`` in
    default_config.yaml the persisted value would be dropped on load and
    the field would re-seed empty (same reasoning as the auto_genre seed
    test above).

    Seeded defects (one line each): delete the ``anilist_username: ""``
    line from default_config.yaml -> the override is stripped as unknown
    -> red; hardcode the ctor seed to "" (ignore
    _read_current_anilist_username) -> red.
    """
    from yamabiko.config_writer import write_override

    write_override(["gazetteer", "anilist_username"], "Mand3012")

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert w._anilist_username_edit is not None
    assert w._anilist_username_edit.text() == "Mand3012", (
        "a persisted gazetteer.anilist_username did not seed the field "
        "(override stripped as unknown, or the seed is hardcoded)"
    )


def test_anilist_username_apply_persists_trimmed_value(qapp, appdata):
    """Typing a username and clicking Apply must persist the TRIMMED value
    via write_override(["gazetteer", "anilist_username"], value) -- the
    Apply/OK convention of the sibling text field (translation.endpoint),
    not the immediate persist of the checkbox/combo controls. AniList
    usernames never carry surrounding whitespace, so an untrimmed paste
    would 404 every watch-list fetch.

    Seeded defects (one line each): drop the ``.strip()`` from
    _persist_anilist_username -> the override carries the padded string ->
    red; drop the _persist_anilist_username call from _apply -> nothing is
    written -> red.
    """
    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    w._anilist_username_edit.setText("  Mand3012  ")
    assert w._apply() is True

    assert _read_overrides(appdata).get("gazetteer", {}).get(
        "anilist_username"
    ) == "Mand3012", (
        "Apply did not persist the trimmed gazetteer.anilist_username"
    )


def test_anilist_username_clear_plus_apply_prunes_the_override(qapp, appdata):
    """Clearing the field and clicking Apply must persist "" -- which the
    sparse writer prunes (the shipped default IS ""), so the merged value
    reads "" again and the watch-list section turns off. Unlike the
    endpoint field there is no shipped non-empty default to reset toward,
    so the plain changed-value write IS the off switch.

    Seeded defect: add an ``if not value: return True`` skip-on-empty
    guard to _persist_anilist_username (the endpoint field's shape) ->
    the stale override survives the clear -> red.
    """
    from yamabiko.config_writer import write_override

    write_override(["gazetteer", "anilist_username"], "OldUser")

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    assert w._anilist_username_edit.text() == "OldUser"
    w._anilist_username_edit.setText("")
    assert w._apply() is True

    assert "anilist_username" not in _read_overrides(appdata).get(
        "gazetteer", {}
    ), (
        "clearing the AniList username + Apply left the stale override on "
        "disk; the empty write must prune it (merged value back to '')"
    )


def test_anilist_username_without_config_path_skips_persist(qapp, appdata):
    """P1-12 sibling: an Apply with config_path=None must NOT write the
    username to disk -- a unit-test-constructed window would otherwise
    mutate the developer's real user_overrides.yaml (appdata redirects the
    target to tmp_path, so a missing gate leaves the stray file THERE).

    Seeded defect: drop the ``if self._config_path is None`` gate from
    _persist_anilist_username -> write_override runs -> user_overrides.yaml
    is created carrying gazetteer.anilist_username -> the not-exists
    assertion red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))  # config_path defaults None
    assert w._config_path is None
    w._anilist_username_edit.setText("Mand3012")
    assert w._apply() is True

    assert not (appdata / "user_overrides.yaml").exists(), (
        "AniList username Apply with no config_path wrote "
        "user_overrides.yaml; the config_path production gate is missing"
    )


@pytest.mark.parametrize("off_flag", ["gazetteer", "romanization"])
def test_anilist_username_field_absent_when_series_feature_disabled(
    qapp, appdata, off_flag,
):
    """With the series feature off (feature_enabled: gazetteer.enabled AND
    romanization.enabled -- either flag false disables it), the "AniList
    username" field must not be built: there is no palette to render the
    watch list, so the field would persist a key that controls nothing.
    Absent, not disabled -- the same gate and convention as the auto-genre
    checkbox above.

    Seeded-defect drill: drop the feature gate around the field in
    _build_subtitles_page (build it unconditionally) -> the ref is
    non-None and the placeholder scan finds it -> red.
    """
    from PySide6.QtWidgets import QLineEdit
    from yamabiko.config_writer import write_override

    write_override([off_flag, "enabled"], False)

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml",
    )
    assert w._anilist_username_edit is None, (
        "the AniList username field ref must be None when the series "
        "feature is off, so nothing can wire or persist it"
    )
    # The field is the Series panel's only line edit with the watch-list
    # placeholder ("Optional. Shows your watch list in series search").
    anilist_edits = [
        e for e in w.findChildren(QLineEdit)
        if "watch list" in e.placeholderText().lower()
    ]
    assert anilist_edits == [], (
        f"{off_flag}.enabled=false but the settings window still built the "
        f"AniList username field"
    )


def test_position_apply_emits_positionapplied_with_changed_v_and_preserved_h(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Moving the v-offset slider + Apply must EMIT positionApplied with the new
    v (float of the int the user chose) AND the SAVED off-center h_offset_pct
    carried through unchanged (not the shipped-default 50.0) AND the screen from
    the combo. The tray connects this signal to OverlayWindow.set_position, so
    the emission is what moves the live overlay and persists through the single
    sink.

    Regression (P0-2): _persist_position wrote disk directly and emitted no
    signal, so Apply never moved the running overlay. Regression (P1-16): the
    Position page has no h control, so a hardcoded h_offset_pct=50.0 write
    PRUNED a user's deliberate off-center placement; h is re-read at emit time.

    Seeded defect (P1-16): re-hardcode ``h_offset_pct`` to 50.0 in
    _emit_position_applied -> the h assertion goes red. Seeded defect (P0-2):
    revert to a direct write_overlay_position call with no emit -> no emission
    -> the emission assertion goes red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))

    w._voffset.setValue(42)
    w._apply()

    assert len(emitted) == 1, f"expected one positionApplied emission, got {emitted}"
    assert emitted[0]["v_offset_pct"] == 42.0, (
        f"the v the user chose must be emitted as a float; got "
        f"{emitted[0]['v_offset_pct']!r}"
    )
    assert emitted[0]["h_offset_pct"] == 30.0, (
        f"saved off-center h must survive a v-only Apply; got "
        f"{emitted[0]['h_offset_pct']!r}"
    )
    assert emitted[0]["screen"] == w._monitor_combo.currentData()


def test_position_apply_does_not_write_overlay_position_directly(
    qapp, appdata, tmp_path, monkeypatch,
):
    """The single-sink refactor (P0-2): the settings window must NOT call
    write_overlay_position itself on a position-changing Apply. Persistence now
    happens once, through the tray's set_position sink (driven by the
    positionApplied signal). The window only emits.

    Seeded defect: restore the ``from ...config_writer import
    write_overlay_position; write_overlay_position(pos)`` call in
    _emit_position_applied -> the direct-write spy records a call -> red.
    """
    from yamabiko import config_writer

    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )
    direct_writes: list[dict] = []
    monkeypatch.setattr(
        config_writer, "write_overlay_position", lambda pos: direct_writes.append(pos)
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))
    w._voffset.setValue(42)
    w._apply()

    assert emitted, "position-changing Apply emitted nothing"
    assert direct_writes == [], (
        f"settings window wrote overlay.position directly instead of routing "
        f"through the tray's set_position sink: {direct_writes}"
    )


def test_position_combo_only_apply_preserves_fractional_saved_v(
    qapp, appdata, tmp_path, monkeypatch,
):
    """When ONLY the monitor combo changed (the slider was never touched, so it
    still reads its int-floored construction seed int(87.3)==87), Apply must
    re-read the fractional saved v from disk and emit 87.3 -- NOT floor it to
    87.0 by sending float(slider.value()).

    Seeded defect (P1-16): send float(v_slider) unconditionally in
    _emit_position_applied -> the emitted v is 87.0, not 87.3 -> red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))

    # Change ONLY the monitor: the saved screen "X" is not a live screen so the
    # combo seeded to the "(disconnected)" entry carrying "X" as its data (P0-7).
    # Add a distinct entry and select it so the combo differs from its seed
    # WITHOUT touching the slider.
    assert w._initial_position_screen == "X"
    w._monitor_combo.addItem("Screen 2", userData="screen-2")
    w._monitor_combo.setCurrentIndex(w._monitor_combo.count() - 1)
    assert w._voffset.value() == 87, "slider must still read its int-floored seed"

    w._apply()

    assert len(emitted) == 1, f"combo-only Apply emitted {emitted}"
    assert emitted[0]["screen"] == "screen-2"
    assert emitted[0]["v_offset_pct"] == 87.3, (
        f"combo-only Apply floored the placement-saved fractional v; got "
        f"{emitted[0]['v_offset_pct']!r}"
    )
    assert emitted[0]["h_offset_pct"] == 30.0


def test_unrelated_apply_does_not_emit_position(
    qapp, appdata, tmp_path, monkeypatch,
):
    """An Apply that changed only a Subtitles control (never the Position combo
    or slider) must NOT emit positionApplied -- otherwise set_position would
    re-pin the overlay and the hardcoded h_offset_pct=50.0 would prune a saved
    off-center placement on every unrelated Apply.

    Seeded defect: remove the change-guard from _emit_position_applied (so it
    always builds+emits) -> positionApplied fires on the unrelated Apply -> red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))
    w._jp_size.setValue(w._jp_size.value() + 5)  # a Subtitles-only change
    w._apply()

    assert emitted == [], (
        f"unrelated Apply emitted positionApplied and would re-pin/prune: {emitted}"
    )


def test_position_slider_flip_back_after_apply_emits_the_revert(
    qapp, appdata, tmp_path, monkeypatch,
):
    """The change-guard must advance to the LAST-APPLIED slider value, not stay
    pinned to the window-open snapshot: move the slider + Apply (emit 1), then
    move it back to the seeded value + Apply -- the flip-back must emit again
    (mirrors the device flip-back semantics).

    Seeded defect: drop the ``self._initial_position_v = v_slider`` advance
    after the emit -> the flip-back reads as unchanged -> the second emit is
    skipped -> red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))

    seeded_v = w._voffset.value()  # int(87.3) == 87
    w._voffset.setValue(42)
    w._apply()
    assert len(emitted) == 1, f"first Apply should emit once, got {emitted}"

    w._voffset.setValue(seeded_v)
    w._apply()
    assert len(emitted) == 2, (
        f"flip-back Apply was skipped; the guard did not advance: {emitted}"
    )
    assert emitted[1]["v_offset_pct"] == float(seeded_v)


def test_position_apply_emits_nothing_without_config_path(qapp, appdata):
    """No config_path (unit-test mode) -> Apply does NOT emit positionApplied.

    Seeded defect: drop the ``if self._config_path is None: return True`` gate
    in _emit_position_applied -> the emission fires in unit-test mode -> red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))
    w._voffset.setValue(40)
    w._apply()
    assert emitted == []


def test_disconnected_saved_screen_seeds_combo_to_disconnected_entry(
    qapp, appdata, tmp_path, monkeypatch,
):
    """P0-7: when the saved screen is not currently connected, the monitor combo
    must carry a "(disconnected)" entry whose data IS the saved name, and seed
    to it -- NOT collapse to index 0 ("primary"). Both the combo's current data
    and the change-guard seed must be the saved name, so the pin survives an
    untouched window.

    Seeded defect (one line): revert the combo seed to
    ``setCurrentIndex(_midx if _midx >= 0 else 0)`` with no disconnected entry
    -> findData("X") stays -1, the entry is absent, currentData() is "primary"
    -> the itemText / currentData / _initial_position_screen asserts go red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )

    idx = w._monitor_combo.findData("X")
    assert idx >= 0, "the saved (disconnected) screen was not added to the combo"
    assert w._monitor_combo.itemText(idx) == "X (disconnected)", (
        f"disconnected entry label wrong: {w._monitor_combo.itemText(idx)!r}"
    )
    assert w._monitor_combo.currentData() == "X", (
        f"combo did not seed to the disconnected saved screen; currentData="
        f"{w._monitor_combo.currentData()!r} (a collapse to 'primary' is the bug)"
    )
    assert w._initial_position_screen == "X", (
        f"change-guard seed is not the saved screen; got "
        f"{w._initial_position_screen!r}"
    )


def test_disconnected_screen_v_only_apply_preserves_saved_pin(
    qapp, appdata, tmp_path, monkeypatch,
):
    """P0-7: with the saved screen disconnected, a v-ONLY Apply (user nudged the
    slider, never touched the monitor combo) must emit the SAVED screen name, so
    OverlayWindow.set_position keeps the pin and _pin_to_screen's keep-the-name
    recovery still returns the strip to the monitor when it reconnects.

    Seeded defect (one line): seed the combo to index 0 on a disconnected screen
    (drop the disconnected-entry branch) -> currentData() is "primary" -> the
    v-only Apply emits screen="primary", rewriting the pin -> the emitted-screen
    assert goes red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))

    w._voffset.setValue(42)   # move ONLY the slider; combo untouched
    w._apply()

    assert len(emitted) == 1, f"expected one positionApplied emission, got {emitted}"
    assert emitted[0]["screen"] == "X", (
        f"v-only Apply on a disconnected saved screen rewrote the pin to "
        f"{emitted[0]['screen']!r}; the saved name must be preserved"
    )
    assert emitted[0]["v_offset_pct"] == 42.0
    assert emitted[0]["h_offset_pct"] == 30.0


def test_disconnected_screen_deliberate_monitor_change_rewrites(
    qapp, appdata, tmp_path, monkeypatch,
):
    """P0-7 completeness: a DELIBERATE monitor pick (away from the disconnected
    saved screen) must still rewrite normally -- the pin-preservation only
    covers the untouched combo, not a real user choice.

    Booted with the saved screen "X" disconnected (combo seeds to the "X
    (disconnected)" entry), the user selects "Primary". That is a genuine change
    from the seed, so Apply emits screen="primary".

    Seeded defect (one line): seed the combo to index 0 on a disconnected screen
    -> the seed is already "primary", so selecting Primary is a no-op with the
    slider untouched -> nothing emits -> the len(emitted)==1 assert goes red.
    """
    saved_pos = {"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0,
                 "left_offset_pct": 6.0}
    monkeypatch.setattr(
        SettingsWindow, "_read_current_position", staticmethod(lambda cfg=None: dict(saved_pos))
    )

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))

    # Deliberately pick Primary (index 0), away from the disconnected "X" seed,
    # WITHOUT touching the slider.
    w._monitor_combo.setCurrentIndex(0)
    assert w._monitor_combo.currentData() == "primary"
    w._apply()

    assert len(emitted) == 1, (
        f"a deliberate monitor change did not emit; the seed likely collapsed to "
        f"'primary' so the pick reads as unchanged: {emitted}"
    )
    assert emitted[0]["screen"] == "primary", (
        f"deliberate monitor change emitted the wrong screen: {emitted[0]['screen']!r}"
    )


def test_v_only_apply_carries_move_mode_saved_screen_not_stale_combo_seed(
    qapp, appdata, tmp_path,
):
    """C1: h and fractional v are re-read from disk at Apply time precisely to
    absorb Move-mode saves made while the window is open -- the SCREEN must be
    too. The monitor combo is seeded once at construction; when the user never
    touched it, a v-only Apply must carry the screen from the FRESH disk read
    (Move mode re-pinned to another monitor), not the stale construction seed,
    or the Apply silently reverts the external monitor move.

    Seeded defect (one line): revert to
    ``screen = self._monitor_combo.currentData() or "primary"`` unconditionally
    in _emit_position_applied -> the emission carries the seeded "primary" ->
    the screen assertion red.
    """
    from yamabiko.config_writer import write_overlay_position

    # No saved position at construction -> the combo seeds to "primary".
    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    assert w._initial_position_screen == "primary"

    # Simulate a Move-mode save WHILE the window is open: the overlay was
    # dragged to monitor "X" and persisted through the single sink.
    write_overlay_position({"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0})

    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))
    w._voffset.setValue(42)  # move ONLY the slider; the combo keeps its seed
    w._apply()

    assert len(emitted) == 1, f"expected one positionApplied emission, got {emitted}"
    assert emitted[0]["screen"] == "X", (
        f"v-only Apply reverted the Move-mode monitor move: emitted "
        f"{emitted[0]['screen']!r} instead of the freshly saved 'X'"
    )
    assert emitted[0]["v_offset_pct"] == 42.0
    assert emitted[0]["h_offset_pct"] == 30.0


def test_apply_during_transient_read_failure_emits_seeds_not_defaults(
    qapp, appdata, tmp_path, monkeypatch, caplog,
):
    """C15: when the Apply-time fresh read RAISES (transient IO failure -- AV
    lock, disk hiccup), _emit_position_applied must not ship
    ``clamp_position({})`` defaults: emitting h=50.0/v=88.0 would snap the
    overlay to centre AND the downstream write would prune the user's saved
    off-center h as equal-to-default. It must log a warning and fall back to
    the values the window was seeded with (the last GOOD read).

    Seeded defect: collapse the sentinel read back to the bare
    ``self._read_current_position()`` swallow -> the failure masquerades as a
    config full of shipped defaults -> emitted h is 50.0 -> red.
    """
    import logging

    from yamabiko.config_writer import write_overlay_position

    # A good position is on disk BEFORE the window opens: the seeds hold it.
    write_overlay_position({"screen": "X", "v_offset_pct": 87.3, "h_offset_pct": 30.0})

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.positionApplied.connect(lambda p: emitted.append(p))

    # The disk goes bad AFTER construction: every merged read now raises.
    import yamabiko.config as _config

    def _boom(*a, **kw):
        raise OSError("transient read failure")

    monkeypatch.setattr(_config, "load_full_config", _boom)

    # Change ONLY the combo (the saved "X" is disconnected, so the combo seeded
    # to its "(disconnected)" entry; pick a distinct one) -- v and h must then
    # both come from the fallback, exposing a defaults-shaped fallback fully.
    w._monitor_combo.addItem("Screen 2", userData="screen-2")
    w._monitor_combo.setCurrentIndex(w._monitor_combo.count() - 1)
    with caplog.at_level(logging.WARNING):
        w._apply()

    assert len(emitted) == 1, f"expected one positionApplied emission, got {emitted}"
    assert emitted[0]["h_offset_pct"] == 30.0, (
        f"transient read failure shipped the shipped-default h; got "
        f"{emitted[0]['h_offset_pct']!r} (50.0 means the saved off-center h "
        f"will be pruned downstream)"
    )
    assert emitted[0]["v_offset_pct"] == 87.3, (
        f"transient read failure floored/defaulted the seeded fractional v; "
        f"got {emitted[0]['v_offset_pct']!r}"
    )
    assert emitted[0]["screen"] == "screen-2"
    assert any(
        "position re-read failed" in r.getMessage() for r in caplog.records
    ), "the fallback must leave a warning, not silently swap in stale values"


def test_close_emits_preview_ended(qapp):
    """Closing the window must emit previewEnded so the tray can drop the
    sample subtitle and restore the committed style.

    Seeded defect: drop the previewEnded.emit() from closeEvent -> no
    emission -> red.
    """
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    ended: list[int] = []
    w.previewEnded.connect(lambda: ended.append(1))
    w.close()
    assert ended == [1]


def test_close_stops_pending_preview_debounce_so_no_late_preview_fires(qapp):
    """P2: a style-control change within the 150 ms debounce window of closing
    must NOT let the single-shot _preview_timer fire a stylePreview AFTER the
    window closed -- that late frame reaches the tray and re-shows an orphaned
    sample on the live overlay. closeEvent must stop the timer before emitting
    previewEnded.

    Asserted directly on the SettingsWindow's own signals (not through the
    tray), so the timer-stop behaviour is isolated from the tray's belt guard:
    a stylePreview that arrives after previewEnded is the exact defect. The
    setValue+close happen with no event-loop spin between them, so the only way
    a preview can appear is the armed timer firing during the qWait.

    Seeded defect: remove `self._preview_timer.stop()` from closeEvent -> the
    pending timer fires _emit_preview after close -> a "preview" lands after
    "ended" during the qWait -> red.
    """
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    events: list[str] = []
    w.stylePreview.connect(lambda _d: events.append("preview"))
    w.previewEnded.connect(lambda: events.append("ended"))

    w._jp_size.setValue(40)  # arms the 150 ms preview debounce
    w.close()                # within the window: closeEvent must stop the timer
    QTest.qWait(250)         # let any still-armed debounce fire

    assert events == ["ended"], (
        "a preview fired around close; closeEvent must stop the debounce timer "
        f"before emitting previewEnded. Got event order: {events}"
    )


# ---------------------------------------------------------------------------
# Keyboard handling (C7: lost in the QDialog -> QWidget port)
# ---------------------------------------------------------------------------


def test_escape_closes_and_discards_unapplied_edits(qapp):
    """C7: QDialog gave Esc->reject for free; the QWidget port must restore it.
    Esc is the Cancel path: close the window WITHOUT emitting styleChanged
    (un-Applied edits are discarded).

    Seeded defect: drop the Key_Escape branch from keyPressEvent -> the widget
    default ignores the key -> the window stays open -> the _closed assert red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    emitted: list[dict] = []
    w.styleChanged.connect(lambda s: emitted.append(s))
    w._jp_size.setValue(30)  # an un-Applied edit

    QTest.keyClick(w, Qt.Key_Escape)

    assert w._closed, "Esc did not close the window (Cancel semantics)"
    assert emitted == [], "Esc must discard the un-Applied edit, not Apply it"


def test_return_applies_and_closes_ok_semantics(qapp):
    """C7: Return/Enter is the OK path -- Apply (styleChanged fires with the
    edited value) and close.

    Seeded defect: drop the Key_Return/Key_Enter branch from keyPressEvent ->
    no emission, window stays open -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    emitted: list[dict] = []
    w.styleChanged.connect(lambda s: emitted.append(s))
    w._jp_size.setValue(30)

    QTest.keyClick(w, Qt.Key_Return)

    assert emitted and emitted[0]["jp_size"] == 30, (
        "Return must Apply (OK semantics) before closing"
    )
    assert w._closed, "Return did not close the window after applying"


def test_return_inside_a_line_edit_still_triggers_ok(qapp, appdata):
    """C7: QDialog's auto-default meant Enter INSIDE a QLineEdit clicked OK.
    QLineEdit ignores Return after emitting returnPressed, so the event
    propagates up to the window's keyPressEvent -- the port must catch it
    there (a plain default-button-less QWidget would drop it).

    Seeded defect: drop the Return/Enter branch from keyPressEvent -> the
    line edit's ignored Return bubbles to a window that also ignores it ->
    Enter in the endpoint field does nothing -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    emitted: list[dict] = []
    w.styleChanged.connect(lambda s: emitted.append(s))

    QTest.keyClick(w.llama_endpoint_edit, Qt.Key_Return)

    assert emitted, "Enter inside the endpoint line edit did not trigger OK"
    assert w._closed, "Enter inside a line edit must apply-and-close"


def test_return_on_focused_cancel_button_cancels_without_saving(
    qapp, appdata, tmp_path,
):
    """Enter on a focused push button activates THAT button, as in a dialog.
    Enter on Cancel must close the window and discard the un-Applied edit: no
    styleChanged, nothing written to the overrides file.

    Seeded defect: drop the autoDefault setup for the window's push buttons ->
    the button ignores Return, the key bubbles up to the window's OK branch ->
    the edit is saved and emitted -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    emitted: list[dict] = []
    w.styleChanged.connect(lambda s: emitted.append(s))
    w._jp_size.setValue(30)  # an un-Applied edit

    QTest.keyClick(w._cancel_btn, Qt.Key_Return)

    assert w._closed, "Enter on the focused Cancel button did not close"
    assert emitted == [], "Enter on Cancel applied the edit (ran OK)"
    assert _read_overrides(appdata) == {}, "Enter on Cancel saved settings"


def test_return_on_focused_reset_or_color_field_clicks_it_without_ok(
    qapp, appdata, tmp_path,
):
    """Enter on the focused "Reset to defaults" or a color field clicks THAT
    control, like an autoDefault push button: Reset resets and the color
    field opens the picker; neither applies the draft or closes the window.

    Seeded defect: drop _ClickOnEnter from LinkButton / ColorField -> the
    painted button ignores Return, the key reaches the window's OK branch ->
    the draft is saved and the window closes -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    picked: list[str] = []
    w = SettingsWindow(
        current={**STYLE_DEFAULTS, "jp_size": 40},
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        color_picker=lambda parent, palette, role, initial, other: picked.append(role),
    )
    emitted: list[dict] = []
    w.styleChanged.connect(emitted.append)

    QTest.keyClick(w._reset_btn, Qt.Key_Return)
    assert w._jp_size.value() == STYLE_DEFAULTS["jp_size"], "Enter on Reset did not reset"
    QTest.keyClick(w._text_color_field, Qt.Key_Return)
    assert picked == ["text"], "Enter on the color field did not open the picker"

    assert not w._closed, "Enter on Reset or a color field ran OK and closed the window"
    assert emitted == [], "Enter on Reset or a color field applied the draft"
    assert _read_overrides(appdata) == {}, "Enter on Reset or a color field saved settings"
    w.close()


# ---------------------------------------------------------------------------
# Endpoint persistence (ported)
# ---------------------------------------------------------------------------


def test_endpoint_apply_round_trips_to_translation_endpoint_key(
    qapp, appdata, tmp_path,
):
    """A user-typed translator endpoint + Apply must land in
    user_overrides.yaml at translation.endpoint AND survive the merged
    load_full_config() read main.py uses (cfg["translation"]["endpoint"]).

    Seeded-defect check: change config_writer.write_audio_and_translation
    to write_override(["translation", "endpoint_url"], ...) -> the override
    lands at the wrong key and the merged view keeps the shipped default ->
    both assertions red.
    """
    url = "http://example:1234/v1/chat/completions"
    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w.llama_endpoint_edit.setText(url)
    w._apply()

    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("endpoint") == url, (
        f"translation.endpoint not persisted at the consumer's key path; "
        f"overrides={overrides!r}"
    )
    from yamabiko.config import load_full_config
    merged = load_full_config()
    assert merged["translation"]["endpoint"] == url, (
        f"merged config dropped the endpoint; got "
        f"{merged['translation'].get('endpoint')!r}"
    )


def test_device_only_apply_preserves_saved_endpoint(qapp, appdata, tmp_path):
    """Changing ONLY the audio device and clicking Apply must not clobber a
    previously saved translation.endpoint.

    Seeded-defect check: invert the writer guard in
    config_writer.write_audio_and_translation (`if translation_endpoint:` ->
    `if not translation_endpoint:`) -> the blank endpoint overwrites the
    saved one -> red.
    """
    from yamabiko.config_writer import write_override
    saved = "http://saved-endpoint:9/v1/chat/completions"
    write_override(["translation", "endpoint"], saved)

    w = SettingsWindow(
        list_devices=lambda: [
            {"name": "Speakers (loopback)", "index": 1},
            {"name": "Headphones (loopback)", "index": 2},
        ],
        config_path=tmp_path / "ignored.yaml",
    )
    i = w.audio_device_combo.findText("Headphones (loopback)")
    assert i >= 1
    w.audio_device_combo.setCurrentIndex(i)
    w._apply()

    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("endpoint") == saved, (
        f"device-only Apply clobbered the saved endpoint: {overrides!r}"
    )
    assert overrides.get("audio", {}).get("device") == "Headphones (loopback)"


def test_endpoint_field_seeded_from_saved_override(qapp, appdata):
    """A previously saved translation.endpoint override must appear in the
    endpoint field when the window opens.

    Seeded defect: drop the setText(self._read_current_endpoint()) seed from
    the Audio page builder -> the field opens empty -> red.
    """
    from yamabiko.config_writer import write_override
    write_override(["translation", "endpoint"], "http://custom:9/v1")

    w = SettingsWindow(list_devices=lambda: [])
    assert w.llama_endpoint_edit.text() == "http://custom:9/v1", (
        f"saved endpoint override not reflected; field shows "
        f"{w.llama_endpoint_edit.text()!r}"
    )


def test_endpoint_field_seeds_shipped_default_on_fresh_install(qapp, appdata):
    """With no override saved, the field seeds to the shipped default endpoint
    (merged-config read), not an empty string.

    Seeded defect: make _read_current_endpoint return '' -> field opens blank
    -> red.
    """
    import yaml as _yaml
    from yamabiko import paths as _paths
    shipped = _yaml.safe_load(
        _paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    expected = shipped["translation"]["endpoint"]

    w = SettingsWindow(list_devices=lambda: [])
    assert w.llama_endpoint_edit.text() == expected


def test_reset_then_apply_prunes_saved_endpoint_override(qapp, appdata, tmp_path):
    """Reset loads the shipped-default endpoint; a following Apply writes that
    default, which the sparse writer prunes -- clearing the saved override.

    Seeded defect: drop the endpoint reseed from _reset -> the field keeps the
    saved custom value, Apply sees it unchanged, the override survives -> red.
    """
    from yamabiko.config_writer import write_override
    write_override(["translation", "endpoint"], "http://custom:9/v1")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._reset()
    w._apply()
    overrides = _read_overrides(appdata)
    assert "endpoint" not in overrides.get("translation", {}), (
        f"Reset+Apply did not prune the endpoint override: {overrides!r}"
    )


def test_clearing_endpoint_field_then_apply_prunes_saved_override(
    qapp, appdata, tmp_path,
):
    """Emptying the endpoint field directly + Apply must prune a saved
    non-default override so the merged config falls back to the shipped
    default.

    Seeded-defect check: restore the `bool(endpoint) and ...` guard in
    _persist_pipeline_fields (so an emptied field reads as unchanged) -> the
    stale override survives -> both assertions red.
    """
    import yaml as _yaml
    from yamabiko import paths as _paths
    from yamabiko.config_writer import write_override

    shipped = _yaml.safe_load(
        _paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    default_endpoint = shipped["translation"]["endpoint"]

    write_override(["translation", "endpoint"], "http://custom:9/v1")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    assert w.llama_endpoint_edit.text() == "http://custom:9/v1"
    w.llama_endpoint_edit.setText("")
    w._apply()

    overrides = _read_overrides(appdata)
    assert "endpoint" not in overrides.get("translation", {}), (
        f"clearing the endpoint field did not prune the saved override: "
        f"{overrides!r}"
    )
    from yamabiko.config import load_full_config
    merged = load_full_config()
    assert merged["translation"]["endpoint"] == default_endpoint, (
        f"after clearing the field, the merged endpoint should be the shipped "
        f"default; got {merged['translation'].get('endpoint')!r}"
    )


def test_endpoint_flip_back_after_apply_persists_the_revert(qapp, appdata, tmp_path):
    """The changed-guard must track the LAST PERSISTED value, not the
    window-open snapshot: apply a new endpoint, then type the original back
    and Apply again -- the revert must land on disk.

    Seeded defect: drop the `self._initial_endpoint = endpoint` update after a
    successful persist -> the second Apply is skipped -> red.
    """
    from yamabiko.config_writer import write_override
    write_override(["translation", "endpoint"], "http://custom:9/v1")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w.llama_endpoint_edit.setText("http://other:10/v1")
    w._apply()
    assert _read_overrides(appdata)["translation"]["endpoint"] == "http://other:10/v1"

    w.llama_endpoint_edit.setText("http://custom:9/v1")
    w._apply()
    assert _read_overrides(appdata)["translation"]["endpoint"] == "http://custom:9/v1", (
        "flip-back Apply was skipped; disk kept the superseded endpoint"
    )


def test_reset_clears_endpoint_field_when_shipped_default_unreadable(
    qapp, appdata, monkeypatch,
):
    """When the shipped-default endpoint can't be read
    (_shipped_default_endpoint()==''), Reset must still clear the field
    instead of leaving the stale override on screen.

    Seeded-defect check: restore the `if default_endpoint:` guard around
    setText in _reset -> the field keeps the stale value -> red.
    """
    w = SettingsWindow(list_devices=lambda: [])
    w.llama_endpoint_edit.setText("http://stale-custom:9/v1")
    monkeypatch.setattr(
        SettingsWindow, "_shipped_default_endpoint", staticmethod(lambda: "")
    )

    w._reset()

    assert w.llama_endpoint_edit.text() == "", (
        f"Reset left a stale endpoint when the shipped default was unreadable; "
        f"field shows {w.llama_endpoint_edit.text()!r}"
    )


def test_clearing_endpoint_warns_when_shipped_default_unreadable(
    qapp, appdata, tmp_path, monkeypatch, caplog,
):
    """An in-field clear can only prune toward a READABLE shipped default.
    When the defaults YAML is unreadable, the saved override survives -- but a
    WARNING is logged so the kept override isn't a mystery.

    Seeded-defect check: delete the `else: logging...warning(...)` branch in
    _persist_pipeline_fields -> the clear silently no-ops -> caplog red.
    """
    import logging

    from yamabiko.config_writer import write_override

    write_override(["translation", "endpoint"], "http://custom:9/v1")
    monkeypatch.setattr(
        SettingsWindow, "_shipped_default_endpoint", staticmethod(lambda: "")
    )

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    assert w.llama_endpoint_edit.text() == "http://custom:9/v1"
    w.llama_endpoint_edit.setText("")
    with caplog.at_level(logging.WARNING):
        w._apply()

    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("endpoint") == "http://custom:9/v1", (
        f"override should survive a clear when the shipped default is "
        f"unreadable; got {overrides!r}"
    )
    assert any(
        "endpoint clear ignored" in r.getMessage() for r in caplog.records
    ), (
        "clearing the endpoint with an unreadable shipped default must log a "
        "WARNING, not silently no-op"
    )


# ---------------------------------------------------------------------------
# Audio device persistence (ported)
# ---------------------------------------------------------------------------


def test_apply_does_not_clobber_audio_device_when_untouched(qapp, appdata, tmp_path):
    """Clicking Apply without touching the device combo must NOT persist
    audio.device.

    Seeded-defect drill: remove the combo seeding + the device_changed guard
    -> audio.device lands in overrides -> this fails.
    """
    w = SettingsWindow(
        list_devices=lambda: [{"name": "Speakers (loopback)", "index": 1}],
        config_path=tmp_path / "ignored.yaml",
    )
    w._apply()
    overrides = _read_overrides(appdata)
    assert "device" not in overrides.get("audio", {}), (
        f"audio.device clobbered on a no-touch Apply: {overrides.get('audio')}"
    )


def test_apply_persists_audio_device_when_changed(qapp, appdata, tmp_path):
    """Paired positive: actually selecting a device DOES persist it."""
    w = SettingsWindow(
        list_devices=lambda: [
            {"name": "Speakers (loopback)", "index": 1},
            {"name": "Headphones (loopback)", "index": 2},
        ],
        config_path=tmp_path / "ignored.yaml",
    )
    i = w.audio_device_combo.findText("Headphones (loopback)")
    assert i >= 1
    w.audio_device_combo.setCurrentIndex(i)
    w._apply()
    overrides = _read_overrides(appdata)
    assert overrides.get("audio", {}).get("device") == "Headphones (loopback)"


def test_device_flip_back_after_apply_persists_the_revert(qapp, appdata, tmp_path):
    """Change device A->B, Apply, then back B->A, Apply. The second write must
    fire.

    Seeded defect: drop the `self._initial_device_index = ...` update after a
    successful persist -> second Apply skipped -> red.
    """
    w = SettingsWindow(
        list_devices=lambda: [
            {"name": "Speakers (loopback)", "index": 1},
            {"name": "Headphones (loopback)", "index": 2},
        ],
        config_path=tmp_path / "ignored.yaml",
    )
    i = w.audio_device_combo.findText("Headphones (loopback)")
    w.audio_device_combo.setCurrentIndex(i)
    w._apply()
    assert _read_overrides(appdata)["audio"]["device"] == "Headphones (loopback)"

    j = w.audio_device_combo.findText("Speakers (loopback)")
    w.audio_device_combo.setCurrentIndex(j)
    w._apply()
    assert _read_overrides(appdata).get("audio", {}).get("device") == (
        "Speakers (loopback)"
    ), "flip-back Apply was skipped; disk kept the superseded device"


# ---------------------------------------------------------------------------
# Translation model list + acquisition (download on switch)
# ---------------------------------------------------------------------------

_MODEL_READY = "The model is ready. It takes effect the next time Yamabiko starts."
_STILL_MISSING = ("The model file is still missing. Yamabiko offers the download "
                  "again the next time you click Apply or OK.")


def _force_tier_cached(monkeypatch, cached=True):
    """Pin model_fetch.tier_cached so a tier switch takes the cached (no
    download) path -- the seam the settings persist logic branches on."""
    from yamabiko.overlay_app import model_fetch
    monkeypatch.setattr(model_fetch, "tier_cached", lambda tier: cached)


def test_model_choice_defaults_to_standard_when_no_override(qapp, appdata):
    """Fresh install: no user_overrides.yaml -> the model list shows Standard."""
    w = SettingsWindow(list_devices=lambda: [])
    assert w.selected_quant_tier() == "default"
    assert [s.option.value for s in w._model_choice._segments] == ["default", "small"]


def test_model_choice_reads_existing_override(qapp, appdata):
    """If user_overrides.yaml already has quant_tier=small, reflect it."""
    from yamabiko.config_writer import write_override
    write_override(["translation", "quant_tier"], "small")

    w = SettingsWindow(list_devices=lambda: [])
    assert w.selected_quant_tier() == "small"
    assert w._model_choice.value() == "small"


def test_quant_tier_apply_persists_to_overrides(qapp, appdata, tmp_path, monkeypatch):
    """Switching to Smaller + Apply writes the override (model already cached)."""
    _force_tier_cached(monkeypatch)
    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._model_choice.setValue("small")
    assert w.selected_quant_tier() == "small"
    w._apply()

    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert (
        overrides.get("asr", {})
        .get("faster_whisper", {})
        .get("compute_type")
        == "int8_float16"
    )


def test_quant_tier_persist_batches_pair_into_one_atomic_write(
    qapp, appdata, tmp_path, monkeypatch,
):
    """The coupled quant pair lands inside the one Settings atomic write."""
    import yamabiko.config_writer as cw

    calls: list = []
    real = cw._atomic_write

    def _counting(path, data):
        calls.append(path)
        return real(path, data)

    monkeypatch.setattr(cw, "_atomic_write", _counting)
    _force_tier_cached(monkeypatch)

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._model_choice.setValue("small")  # default -> small: the write fires
    assert w._apply() is True

    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert (
        overrides.get("asr", {})
        .get("faster_whisper", {})
        .get("compute_type")
        == "int8_float16"
    )
    assert len(calls) == 1, (
        f"expected ONE atomic write for the coupled quant pair; got "
        f"{len(calls)} separate RMW cycles"
    )


def test_quant_tier_apply_back_to_default_prunes_both_keys(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Switching back to Standard resets both keys to shipped defaults."""
    _force_tier_cached(monkeypatch)
    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._model_choice.setValue("small")
    w._apply()
    assert _read_overrides(appdata).get("translation", {}).get("quant_tier") == "small"

    w._model_choice.setValue("default")
    w._apply()
    overrides = _read_overrides(appdata)
    assert "translation" not in overrides
    assert "asr" not in overrides


def test_quant_tier_apply_skips_write_without_config_path(qapp, appdata):
    """No config_path (unit-test mode) -> _apply does not touch disk."""
    w = SettingsWindow(list_devices=lambda: [])
    w._model_choice.setValue("small")
    w._apply()

    overrides = _read_overrides(appdata)
    assert "translation" not in overrides


def test_quant_tier_change_with_cached_model_persists_without_download(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Brief test 1: when the target tier's GGUF is ALREADY cached, Apply must
    persist the coupled pair and NOT open the download dialog / start any
    download.

    Seeded defect: drop the ``if not tier_cached(tier)`` guard in
    _persist_quant_tier so it always downloads -> the exploding dialog is
    constructed -> red.
    """
    from yamabiko.overlay_app import model_fetch

    _force_tier_cached(monkeypatch, cached=True)

    def _exploder(*a, **kw):
        raise AssertionError(
            "a cached tier switch constructed the download dialog -- it must not"
        )

    monkeypatch.setattr(model_fetch, "QuantDownloadDialog", _exploder)

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._model_choice.setValue("small")  # default -> small
    assert w._apply() is True

    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert (
        overrides.get("asr", {}).get("faster_whisper", {}).get("compute_type")
        == "int8_float16"
    )
    assert w._quant_status.text() == _MODEL_READY, (
        f"cached switch must confirm it takes effect next launch: "
        f"{w._quant_status.text()!r}"
    )


def test_quant_tier_change_not_cached_downloads_then_persists(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Brief test 2: an uncached tier switch downloads the tier's exact GGUF
    FIRST, then persists the coupled pair only AFTER the download succeeds, and
    confirms it takes effect next launch.

    Drives the REAL download dialog + worker with a fake at the downloader
    seam (records what was requested). Seeded defect: move the write_quant_tier
    call BEFORE the download branch -> the pair persists even on a would-be
    failure; or build the spec from the wrong tier -> requested filename
    mismatch -> red.
    """
    from yamabiko import models
    from yamabiko.overlay_app import model_fetch

    _force_tier_cached(monkeypatch, cached=False)
    requested: list = []
    # Snapshot the on-disk overrides AT download time (on the worker thread) and
    # assert the ordering on the MAIN thread after exec -- an assertion raised
    # inside the faked download() would be swallowed by the worker as a download
    # error, not surfaced as a test failure.
    seen_at_download: list = []

    class _Recording:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            requested.append((repo_id, filename))
            seen_at_download.append(_read_overrides(appdata))
            if progress_callback is not None:
                progress_callback(10, 10)
            return Path(f"/fake/{filename}")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Recording(),
    )
    w._model_choice.setValue("small")  # default -> small
    assert w._apply() is True

    small_pin = models.TRANSLATOR_PINS["small"]
    assert requested == [(small_pin.repo, small_pin.filename)], (
        f"the uncached switch must fetch the small tier's GGUF; got {requested}"
    )
    assert seen_at_download == [{}], (
        f"the tier must not be persisted BEFORE the download finishes; "
        f"overrides at download time were {seen_at_download}"
    )
    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert (
        overrides.get("asr", {}).get("faster_whisper", {}).get("compute_type")
        == "int8_float16"
    )
    assert w._quant_status.text() == _MODEL_READY


def test_equal_tier_apply_with_missing_gguf_downloads_the_model(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Installs stranded by the old persist-without-download bug (quant_tier
    persisted, GGUF never fetched) open Settings with the combo ALREADY on the
    persisted tier, so an Apply without touching the combo is their only lever:
    it must probe the cache and run the download despite the no-change guard.
    The already-persisted pair is NOT rewritten (a rewrite would clobber a
    hand-set compute_type -- gpu.py's hand-edit-survival contract).

    Seeded defect: restore the unconditional `if tier == self._initial_tier:
    return True` ahead of the cache probe -> no download requested -> red.
    """
    from yamabiko import models
    from yamabiko.config_writer import write_override, write_quant_tier

    write_quant_tier("small")  # the stranded persisted state...
    # ...plus a hand-edited compute_type that must survive the recovery.
    write_override(["asr", "faster_whisper", "compute_type"], "float32")
    _force_tier_cached(monkeypatch, cached=False)  # ...and the GGUF is missing
    requested: list = []

    class _Recording:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            requested.append((repo_id, filename))
            if progress_callback is not None:
                progress_callback(10, 10)
            return Path(f"/fake/{filename}")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Recording(),
    )
    assert w.selected_quant_tier() == "small", (
        "combo did not seed from the persisted tier -- test setup stale?"
    )
    assert w._apply() is True

    small_pin = models.TRANSLATOR_PINS["small"]
    assert requested == [(small_pin.repo, small_pin.filename)], (
        f"the stranded equal-tier Apply must fetch the persisted tier's GGUF; "
        f"got {requested}"
    )
    assert w._quant_status.text() == _MODEL_READY, (
        f"recovery download must confirm it takes effect next launch: "
        f"{w._quant_status.text()!r}"
    )
    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert (
        overrides.get("asr", {}).get("faster_whisper", {}).get("compute_type")
        == "float32"
    ), (
        f"equal-tier recovery rewrote the coupled pair, clobbering the "
        f"hand-set compute_type: {overrides!r}"
    )


def test_position_only_apply_with_missing_gguf_heals_via_the_injected_downloader(
    qapp, appdata, tmp_path, monkeypatch,
):
    """The stranded-install heal fires on ANY Apply, not just quant-page ones:
    a user who only moved the v-offset still gets the missing GGUF fetched
    (probe before the no-change guard), and the fetch flows through the
    injected downloader seam -- the suite must never reach the real HF cache
    or network no matter which page drove the Apply. This pins the exact
    2026-07-11 CI regression: the position-apply test on a cache-miss runner
    walked into a REAL 5.1 GiB download inside the unit job.

    Seeded defect: restore the `if tier == self._initial_tier: return True`
    guard ahead of the cache probe -> no download requested -> red.
    """
    from yamabiko import models

    _force_tier_cached(monkeypatch, cached=False)
    requested: list = []

    class _Recording:
        def download(self, repo_id, filename, revision=None,
                     progress_callback=None):
            requested.append((repo_id, filename))
            if progress_callback is not None:
                progress_callback(10, 10)
            return Path(f"/fake/{filename}")

    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS),
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Recording(),
    )
    w._voffset.setValue(42)  # a position-only edit; quant combo untouched
    assert w._apply() is True

    default_pin = models.TRANSLATOR_PINS["default"]
    assert requested == [(default_pin.repo, default_pin.filename)], (
        f"a position-only Apply with the GGUF missing must still heal the "
        f"install through the injected downloader; got {requested}"
    )


def test_quant_tier_download_cancelled_restores_combo_and_persists_nothing(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Brief test 3 (settings side): a cancelled download must persist NOTHING,
    revert the combo to the tier in effect, and show a cancelled message -- the
    effective state is exactly as before Apply.

    The dialog's real cancel mechanics + thread cleanup are covered in
    test_model_fetch; here the worker is cancelled mid-download to exercise the
    full settings restore path through the real dialog.

    Seeded defect: drop the _restore_model_choice() call on a non-success
    outcome -> the combo stays on 'small' though nothing persisted -> red.
    """
    _force_tier_cached(monkeypatch, cached=False)
    box: dict = {}

    class _SelfCancel:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            box["w"]._active_quant_dialog._worker.cancel()  # user hits Cancel
            return Path(f"/fake/{filename}")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_SelfCancel(),
    )
    box["w"] = w
    w._model_choice.setValue("small")  # default -> small
    assert w._apply() is False, "a cancelled tier switch must fail the Apply"

    assert w.selected_quant_tier() == "default", (
        "combo was not reverted to the in-effect tier after cancel"
    )
    assert _read_overrides(appdata) == {}, "cancelled switch persisted something"
    assert w._quant_status.text() == "Download canceled. The model wasn't changed."


@pytest.mark.usefixtures("_cancel_download_on_failure")
def test_quant_tier_download_failed_keeps_selection_and_says_so(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Brief test 4: a failed download must persist NOTHING, KEEP the user's
    selection in the list (the download dialog's failed state offers Retry,
    and Apply/OK retries too, which a reverted list would turn into a
    no-change no-op), and say so in the status line. Distinct from the
    canceled message.

    Seeded defect: return True (not False) on the failed branch of
    _prepare_quant_tier -> the Apply reads as success -> red; or restore the
    _restore_model_choice() call on the failed branch -> the selection
    assert goes red.
    """
    _force_tier_cached(monkeypatch, cached=False)

    class _Failing:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            raise RuntimeError("404 Client Error: file not found")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Failing(),
    )
    w._model_choice.setValue("small")  # default -> small
    assert w._apply() is False, "a failed tier switch must fail the Apply"

    assert w.selected_quant_tier() == "small", (
        "failure reverted the combo; the instructed Apply/OK retry would "
        "no-op on the no-change guard"
    )
    assert _read_overrides(appdata) == {}, "failed switch persisted something"
    assert w._quant_status.text() == "The download stopped. The model wasn't changed."


class _FailOnceDownloader:
    """Fails the first download() with a network-shaped error, succeeds after.

    The shape of the finding-2 scenario: a transient network failure on the
    first Apply, cleared by the time the user clicks Apply or OK again.
    """

    def __init__(self):
        self.requested: list = []

    def download(self, repo_id, filename, revision=None, progress_callback=None):
        self.requested.append((repo_id, filename))
        if len(self.requested) == 1:
            raise RuntimeError("connection reset by peer")
        if progress_callback is not None:
            progress_callback(10, 10)
        return Path(f"/fake/{filename}")


@pytest.mark.usefixtures("_cancel_download_on_failure")
def test_quant_tier_download_failed_retry_via_apply_downloads_again_and_persists(
    qapp, appdata, tmp_path, monkeypatch,
):
    """After a failed download, Apply or OK retries: the retry must actually
    re-probe and re-download the tier the user selected:
    the combo keeps the selection on failure (reverting it would make the
    retry hit the no-change guard as a silent no-op).

    Seeded defect: restore the _restore_model_choice() call on the failed
    branch -> the retry Apply runs against the reverted default tier, never
    re-fetching the small GGUF and persisting nothing -> red.
    """
    from yamabiko import models

    _force_tier_cached(monkeypatch, cached=False)
    dl = _FailOnceDownloader()

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=dl,
    )
    w._model_choice.setValue("small")  # default -> small
    assert w._apply() is False, "the first (failing) tier switch must fail the Apply"
    assert w.selected_quant_tier() == "small", (
        "failure reverted the combo; the instructed retry would no-op on the "
        "no-change guard"
    )
    assert _read_overrides(appdata) == {}, "failed switch persisted something"

    # The instructed retry, through the SAME Apply path.
    assert w._apply() is True, "the retry Apply did not succeed"

    small_pin = models.TRANSLATOR_PINS["small"]
    small = (small_pin.repo, small_pin.filename)
    assert dl.requested == [small, small], (
        f"the retry must re-download the SELECTED tier's GGUF; got {dl.requested}"
    )
    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert (
        overrides.get("asr", {}).get("faster_whisper", {}).get("compute_type")
        == "int8_float16"
    )
    assert w._quant_status.text() == _MODEL_READY


@pytest.mark.usefixtures("_cancel_download_on_failure")
def test_ok_after_failed_download_keeps_window_open_then_second_ok_retries_and_closes(
    qapp, appdata, tmp_path, monkeypatch,
):
    """OK on a failed tier download must NOT close the window (the user would
    walk away believing the switch stuck); a second OK -- the instructed
    retry -- must re-run the download and close only once it succeeds and
    the pair is persisted.

    Seeded defect: return True from _persist_quant_tier's failed-download
    branch -> the first OK closes the window with nothing persisted -> red.
    """
    _force_tier_cached(monkeypatch, cached=False)
    dl = _FailOnceDownloader()

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=dl,
    )
    w._model_choice.setValue("small")  # default -> small
    w._ok()

    assert not w._closed, "window closed although the tier download failed"
    assert _read_overrides(appdata) == {}, "failed switch persisted something"

    # The instructed retry via OK: download succeeds -> persist -> close.
    w._ok()

    assert w._closed, "successful retry OK did not close the window"
    assert _read_overrides(appdata).get("translation", {}).get("quant_tier") == "small"


def test_no_tier_change_apply_with_cached_model_does_not_download_or_write(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Brief test 6: an Apply that does not change the tier, with the tier's
    GGUF present, must neither download nor write the pair. (The cache probe
    itself now RUNS on every Apply -- that is the stranded-install recovery
    lever, see test_equal_tier_apply_with_missing_gguf_downloads_the_model.)

    Seeded defect: drop the ``if not tier_changed`` no-write leg (write
    unconditionally) -> the recorded write fires on an unrelated Apply ->
    red; or drop the tier_cached branch (download unconditionally) -> the
    exploding dialog fires -> red.
    """
    from yamabiko.overlay_app import model_fetch

    _force_tier_cached(monkeypatch, cached=True)

    def _explode_dialog(*a, **kw):
        raise AssertionError("no-change Apply started a download -- it must not")

    monkeypatch.setattr(model_fetch, "QuantDownloadDialog", _explode_dialog)

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._jp_size.setValue(w._jp_size.value() + 5)  # a Subtitles-only change
    assert w._apply() is True
    overrides = _read_overrides(appdata)
    assert "quant_tier" not in overrides.get("translation", {})
    assert "compute_type" not in (
        overrides.get("asr", {}).get("faster_whisper", {})
    )


def test_recovery_leg_cancelled_download_lets_ok_close_and_keeps_the_pair(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Stranded-install recovery hijacking an unrelated OK: the tier did NOT
    change (combo seeds from the persisted tier), only the GGUF is missing,
    and the user cancels the surprise download. The Apply/OK was about other
    settings -- OK must still close the window, nothing on disk changes, and
    the inline status says the recovery will be offered again.

    Seeded defect: return False from _prepare_quant_tier's recovery-leg
    cancel (the pre-fix behaviour) -> _ok refuses to close -> red.
    """
    from yamabiko.config_writer import write_quant_tier

    write_quant_tier("small")  # the stranded persisted state...
    _force_tier_cached(monkeypatch, cached=False)  # ...with the GGUF missing
    box: dict = {}

    class _SelfCancel:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            box["w"]._active_quant_dialog._worker.cancel()  # user hits Cancel
            return Path(f"/fake/{filename}")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_SelfCancel(),
    )
    box["w"] = w
    assert w.selected_quant_tier() == "small", (
        "combo did not seed from the persisted tier -- test setup stale?"
    )
    w._ok()

    assert w._closed, (
        "OK stayed open although the cancelled download was only the "
        "stranded-install recovery leg, not a tier change"
    )
    # Nothing persisted or clobbered: the stranded pair is untouched.
    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert w._quant_status.text() == _STILL_MISSING


@pytest.mark.usefixtures("_cancel_download_on_failure")
def test_recovery_leg_failed_download_lets_ok_close_without_a_modal_warning(
    qapp, appdata, tmp_path, monkeypatch,
):
    """A FAILED recovery-leg download is likewise non-blocking: OK closes
    (the user asked for a font tweak, not a download) and the failure is
    noted in the inline status. Contrast with the real tier-change failure,
    which must still block
    (test_ok_after_failed_download_keeps_window_open_then_second_ok_retries_and_closes).

    Seeded defect: route the recovery-leg failure through the real switch's
    return-False branch -> OK stays open -> red.
    """
    from yamabiko.config_writer import write_quant_tier

    write_quant_tier("small")
    _force_tier_cached(monkeypatch, cached=False)

    class _Failing:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            raise RuntimeError("connection reset by peer")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Failing(),
    )
    assert w.selected_quant_tier() == "small"
    w._ok()

    assert w._closed, (
        "OK stayed open although the failed download was only the "
        "stranded-install recovery leg, not a tier change"
    )
    overrides = _read_overrides(appdata)
    assert overrides.get("translation", {}).get("quant_tier") == "small"
    assert w._quant_status.text() == _STILL_MISSING


def test_ok_recovery_decline_sends_offer_again_notice_through_notice_signal(
    qapp, appdata, tmp_path, monkeypatch,
):
    """OK closes the window immediately, so the inline status line's
    "offers the download again" note is never seen on the OK flow.
    The same notice must leave through ``noticeRequested`` (the tray
    toasts it) before the close. The Apply flow keeps the inline line
    only -- the window stays open there, so a toast would double up.

    Seeded defect: drop the noticeRequested.emit from _ok's deferred-
    recovery branch -> no notice recorded on OK -> red.
    """
    from yamabiko.config_writer import write_quant_tier

    write_quant_tier("small")  # the stranded persisted state...
    _force_tier_cached(monkeypatch, cached=False)  # ...with the GGUF missing
    box: dict = {}

    class _SelfCancel:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            box["w"]._active_quant_dialog._worker.cancel()  # user hits Cancel
            return Path(f"/fake/{filename}")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_SelfCancel(),
    )
    box["w"] = w
    notices: list[tuple[str, str]] = []
    w.noticeRequested.connect(lambda title, body: notices.append((title, body)))

    # Apply flow unchanged: the window stays open and the inline status
    # line carries the notice, so nothing leaves through the signal.
    assert w._apply() is True
    assert notices == [], (
        "the Apply flow raised the offer-again notice; the inline status "
        "line already covers it while the window is open"
    )

    w._ok()

    assert w._closed, "OK no longer closes on a declined recovery download"
    assert len(notices) == 1, (
        f"expected exactly one OK-flow notice; got {notices}"
    )
    assert notices[0] == (
        "Translation model still missing",
        "The download was canceled. Yamabiko offers it again the next time "
        "you open Settings and click Apply or OK.",
    )


def test_quant_help_copy_sources_sizes_from_tier_info(qapp, appdata):
    """The model rows' size copy must match model_fetch.TIER_INFO -- the same
    figures the download dialog's heading shows. A hardcoded figure here
    went stale once already (still claimed the shisa-era 'about 5.5 GB'
    after the Qwen3.5 swap made Standard ~5.1 GB).

    Seeded defect: reinstate a hardcoded size in a row's sublabel -> red.
    """
    from yamabiko.overlay_app.model_fetch import TIER_INFO, size_text

    w = SettingsWindow(list_devices=lambda: [])
    assert [s.option.sublabel for s in w._model_choice._segments] == [
        f"{cards}, {size_text(gb)} download" for _name, gb, cards in TIER_INFO.values()
    ]


def test_settings_release_threads_detaches_active_quant_download(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Brief test 5 (settings side): while a tier download is in flight,
    SettingsWindow.release_threads() must reach the active dialog and detach
    its worker into thread_keeper -- the path the tray wires to aboutToQuit.

    Seeded defect: make release_threads a no-op (drop the delegate call) ->
    the running thread is never parked -> the keeper assertion red.
    """
    import threading

    from yamabiko.ui import thread_keeper

    _force_tier_cached(monkeypatch, cached=False)
    entered = threading.Event()
    gate = threading.Event()

    class _Blocking:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            entered.set()
            gate.wait(timeout=10.0)
            return Path("/fake/f")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        quant_downloader=_Blocking(),
    )
    # Start the dialog WITHOUT the blocking modal exec so the test can reach in.
    from yamabiko.overlay_app.model_fetch import QuantDownloadDialog, spec_for_tier
    dialog = QuantDownloadDialog(
        spec_for_tier("small"), "small", downloader=w._quant_downloader, parent=w,
    )
    w._active_quant_dialog = dialog
    dialog.start()
    assert entered.wait(5.0), "worker never entered the blocking download"
    thread = dialog._thread
    try:
        w.release_threads()
        assert any(t is thread for t, _ in thread_keeper._LINGERING), (
            "release_threads did not park the in-flight worker in thread_keeper"
        )
        assert thread_keeper.any_running()
    finally:
        gate.set()
        thread.wait(5000)
        thread_keeper.discard(thread)


def test_quant_tier_help_label_describes_download_and_next_launch(qapp):
    """The help copy must describe the REAL contract: switching downloads the
    model if needed and the change applies on next launch (no stale
    'restart / re-download on next launch' wording).

    Seeded defect: revert the help text to the old 'Change requires restart;
    Yamabiko will re-download ... on next launch if the file is missing' ->
    red.
    """
    w = SettingsWindow(list_devices=lambda: [])
    assert w._quant_help.text() == (
        "Switching downloads the model if you don't have it yet.\n"
        "It takes effect the next time Yamabiko starts."
    )


def test_unrelated_apply_does_not_clobber_hand_set_compute_type(
    qapp, appdata, tmp_path, monkeypatch,
):
    """A hand-set asr.faster_whisper.compute_type override must survive an
    Apply where the tier was NOT changed. (tier_cached is pinned True so the
    now-unconditional cache probe can't route a test run through a real
    download on a machine without the GGUF cached.)

    Seeded defect: drop the `if not tier_changed` no-write leg from
    _persist_quant_tier -> the unrelated Apply rewrites compute_type -> red.
    """
    from yamabiko.config_writer import write_override
    _force_tier_cached(monkeypatch, cached=True)
    write_override(["asr", "faster_whisper", "compute_type"], "int8_float16")

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    assert w.selected_quant_tier() == "default"
    w._jp_size.setValue(30)
    w._apply()

    overrides = _read_overrides(appdata)
    assert (
        overrides.get("asr", {})
        .get("faster_whisper", {})
        .get("compute_type")
        == "int8_float16"
    ), (
        f"unrelated Apply clobbered the hand-set compute_type override: "
        f"{overrides!r}"
    )


# ---------------------------------------------------------------------------
# Persistence failures must be surfaced, not swallowed (ported; result() ->
# closeEvent-driven _closed flag)
# ---------------------------------------------------------------------------


def test_ok_keeps_window_open_and_warns_when_pipeline_persist_raises(
    qapp, appdata, tmp_path, monkeypatch,
):
    """An OSError out of the settings write on OK must (1) ask "Couldn't
    save your settings" with the error and (2) NOT close the window when the
    user cancels it.

    Seeded defect: swallow the persist error silently (or close() in _ok
    unconditionally) -> nothing asked and the window closes -> red.
    """
    from yamabiko.overlay_app import settings_transaction

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(settings_transaction, "persist_settings", _boom)
    asked: list[tuple] = []

    w = SettingsWindow(
        list_devices=lambda: [
            {"name": "Speakers (loopback)", "index": 1},
            {"name": "Headphones (loopback)", "index": 2},
        ],
        config_path=tmp_path / "ignored.yaml",
        save_failed=lambda *args: asked.append(args) or False,
    )
    i = w.audio_device_combo.findText("Headphones (loopback)")
    w.audio_device_combo.setCurrentIndex(i)
    w._ok()

    assert not w._closed, "window closed although the settings write failed"
    assert len(asked) == 1, f"expected one save-failed dialog; got {asked}"
    assert "disk full" in str(asked[0][3]), (
        f"the dialog does not carry the underlying error: {asked[0]!r}"
    )


def test_ok_keeps_window_open_and_warns_when_quant_persist_raises(
    qapp, appdata, tmp_path, monkeypatch,
):
    """Quant-tier leg: an OSError out of the batched settings write that
    carries the coupled tier pair must ask once and keep the window open.

    Seeded defect: swallow the persist error silently -> nothing asked,
    window closes -> red.
    """
    from yamabiko.overlay_app import settings_transaction

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(settings_transaction, "persist_settings", _boom)
    _force_tier_cached(monkeypatch)  # cached -> reach the write directly
    asked: list[tuple] = []

    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
        save_failed=lambda *args: asked.append(args) or False,
    )
    w._model_choice.setValue("small")  # default -> small: write fires
    w._ok()

    assert not w._closed, "window closed although the quant-tier write failed"
    assert "takes effect" not in w._quant_status.text(), (
        "failed transaction announced the uncommitted tier as active"
    )
    assert len(asked) == 1, f"expected one save-failed dialog; got {asked}"
    assert "disk full" in str(asked[0][3]), (
        f"the dialog does not carry the underlying error: {asked[0]!r}"
    )


def test_ok_closes_after_successful_persist(qapp, appdata, tmp_path, monkeypatch):
    """Paired positive: when every persist succeeds, OK closes the window AND
    the override actually landed on disk.

    Seeded defect: make _prepare_quant_tier return False on success (or _ok
    stop gating on _apply) -> the window never closes -> red.
    """
    _force_tier_cached(monkeypatch)
    w = SettingsWindow(
        list_devices=lambda: [],
        config_path=tmp_path / "ignored.yaml",
    )
    w._model_choice.setValue("small")
    w._ok()

    assert w._closed
    assert _read_overrides(appdata)["translation"]["quant_tier"] == "small"


def _click(control, value):
    """Click the segment holding ``value`` (the segment is a
    QAbstractButton; click() emits clicked exactly as a mouse click does)."""
    control._segment_for(value).click()


def test_overlay_choices_preview_and_persist_on_apply(qapp, appdata):
    """The four Overlay rows: a click previews live (debounced stylePreview)
    and Apply persists overlay.style.mode / alignment / echo / banner.

    Seeded defects: leave a key out of _collect, or forget to wire
    valueChanged -> red."""
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml")
    previews = []
    w.stylePreview.connect(previews.append)
    for key, value in (("mode", "learn"), ("alignment", "center"),
                       ("echo", "both"), ("banner", "jp_en")):
        _click(w._overlay_choices[key], value)
    QTest.qWait(250)
    assert previews and previews[-1]["mode"] == "learn" and previews[-1]["banner"] == "jp_en"
    assert w._apply()
    style = _read_overrides(appdata)["overlay"]["style"]
    assert (style["mode"], style["alignment"], style["echo"], style["banner"]) == (
        "learn", "center", "both", "jp_en")
    w.close()


def test_reset_restores_overlay_choices(qapp):
    """Reset to defaults puts the four rows back to Watch / Left / Priority
    only / 日本語.

    Seeded defect: skip the rows in _reset -> red."""
    w = SettingsWindow(current={**STYLE_DEFAULTS, "mode": "learn", "alignment": "center",
                                "echo": "both", "banner": "jp_en"})
    w._reset()
    assert {k: c.value() for k, c in w._overlay_choices.items()} == {
        "mode": "watch", "alignment": "left", "echo": "priority", "banner": "jp"}
    w.close()


def test_preview_follows_overlay_choices(qapp):
    """The preview strip shows the choice at once: Echo "Both languages"
    gives the echo line both blocks, Center centres the lines.

    Seeded defect: rebuild the preview without the new keys -> red."""
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    w.resize(900, 900)
    w.show()
    QTest.qWait(50)
    stack = w._preview.stack
    assert set(stack.frame().lines[0].blocks) == {"en"}
    _click(w._overlay_choices["echo"], "both")
    _click(w._overlay_choices["alignment"], "center")
    QTest.qWait(250)
    frame = stack.frame()
    assert set(frame.lines[0].blocks) == {"jp", "en"}
    centre = stack.width() / 2
    assert all(abs(box.rect.center().x() - centre) <= 1 for box in frame.lines)
    w.close()


def test_window_opens_at_the_design_size_with_the_mockup_tabs(qapp):
    # 1000 x 728; BIZ UDPMincho tabs Subtitles, Position, Timing, Audio &
    # model, Shortcuts. Seeded defects: keep the old fit-to-content size, or
    # drop a tab -> red.
    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    assert (w.width(), w.height()) == (1000, 728)
    assert [tab.accessibleName() for tab in w._tabs._tabs] == [
        "Subtitles 字幕", "Position 位置", "Timing 間", "Audio & model 音声", "Shortcuts 操作"]
    assert w.windowTitle() == "Yamabiko settings"
    w.close()


def test_panels_line_up_with_the_band(qapp):
    # Two panels under the preview monitor, one under the switcher. Seeded
    # defect: let the first two panels share the width freely -> the second
    # panel's left edge drifts off the column -> red.
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    w.show()
    QTest.qWait(50)
    page = w._stack.widget(0)
    panels = [p for p in page.findChildren(QFrame) if p.objectName() == "panel"]
    lefts = sorted(p.mapTo(w, QPoint(0, 0)).x() for p in panels)
    monitor_x = w._preview.mapTo(w, QPoint(0, 0)).x()
    switcher_x = w._switcher.mapTo(w, QPoint(0, 0)).x()
    assert lefts[:2] == [monitor_x, monitor_x + 276 + 12]
    if len(lefts) == 3:
        assert lefts[2] == switcher_x
    w.close()


def test_untouched_unit_fields_keep_their_exact_stored_values(qapp):
    # Opacity shows percent and Time on screen seconds; an Apply must not
    # round a saved 200 to 199 or 1234 ms to 1200. Seeded defect: always
    # convert back from the shown unit -> red.
    w = SettingsWindow(current={**STYLE_DEFAULTS, "bg_opacity": 200, "hold_ms": 1234})
    assert (w._bg_opacity.value(), w._hold.value()) == (78, 1.2)
    style = w._collect()
    assert (style["bg_opacity"], style["hold_ms"]) == (200, 1234)
    w._bg_opacity.setValue(50)
    w._hold.setValue(2.5)
    style = w._collect()
    assert (style["bg_opacity"], style["hold_ms"]) == (128, 2500)
    w.close()


def test_color_field_opens_the_picker_and_previews_the_choice(qapp):
    # The themed picker replaces QColorDialog; the choice lands in the
    # field, the working style and the debounced preview. Seeded defects:
    # forget to update the field, or skip the preview -> red.
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    asked = []

    def picker(parent, palette, role, initial, other):
        asked.append((role, initial, other))
        return "#12304a"

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), color_picker=picker)
    previews = []
    w.stylePreview.connect(previews.append)
    QTest.mouseClick(w._bg_color_field, Qt.LeftButton)
    QTest.qWait(250)
    assert asked == [("background", STYLE_DEFAULTS["bg_color"], STYLE_DEFAULTS["text_color"])]
    assert w._bg_color_field.color() == "#12304a"
    assert previews and previews[-1]["bg_color"] == "#12304a"
    w.close()


def test_save_failure_offers_retry_and_retry_saves(qapp, appdata, monkeypatch):
    # "Couldn't save your settings" with Retry: the first write fails, Retry
    # writes again and OK closes. Seeded defects: no retry loop (OK stays
    # open), or retry without re-writing -> red.
    from yamabiko.overlay_app import settings_transaction

    real = settings_transaction.persist_settings
    attempts = []

    def flaky(commit):
        attempts.append(commit)
        if len(attempts) == 1:
            raise OSError(28, "No space left on device")
        real(commit)

    monkeypatch.setattr(settings_transaction, "persist_settings", flaky)
    asked = []
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml",
                       save_failed=lambda *args: asked.append(args) or True)
    w._jp_size.setValue(20)
    w._ok()
    assert len(asked) == 1 and len(attempts) == 2
    assert w._closed is True
    assert _read_overrides(appdata)["overlay"]["style"]["jp_size"] == 20


def test_fit_to_screen_keeps_the_whole_window_inside_a_short_work_area(qapp):
    # 1920 x 1080 at 150% leaves a 672 px work area. The window fits it,
    # frame and footer included, with the title band on top; the preview
    # monitor gives up height first and keeps its ratio; only a page taller
    # than the panel area scrolls. Seeded defects: keep the old ~690 px
    # layout floor, fit the client but not the frame, take the height from
    # the panels alone, or size the panel area for the tallest page (a plain
    # QStackedWidget) -> red.
    from PySide6.QtCore import QPoint, QRect

    from yamabiko.overlay_app.preview_monitor import MONITOR_H, MONITOR_W

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    w.show()
    QApplication.processEvents()  # the show-time fit to the offscreen screen runs first
    avail = QRect(0, 0, 1920, 672)
    w._fit_to_screen(avail=avail)
    QApplication.processEvents()
    client = w.geometry()
    assert avail.contains(w.frameGeometry()), f"frame at {w.frameGeometry().getRect()}"
    ok = QRect(client.topLeft() + w._ok_btn.mapTo(w, QPoint(0, 0)), w._ok_btn.size())
    assert avail.contains(ok), f"OK at {ok.getRect()}"
    assert w._title_band.mapTo(w, QPoint(0, 0)) == QPoint(0, 0)
    monitor = w._preview
    assert monitor.height() < MONITOR_H
    assert monitor.width() == round(monitor.height() * MONITOR_W / MONITOR_H)
    # Subtitles is taller than the panel area here, Position is not.
    bar = w._panel_scroll.verticalScrollBar()
    assert bar.maximum() > 0, "Subtitles fits the short window; the page check is vacuous"
    w._tabs.setCurrentIndex(1)
    QApplication.processEvents()
    assert bar.maximum() == 0, "the short Position page scrolls with the tallest page's height"
    w.close()


def test_wheel_over_an_unfocused_spin_box_scrolls_the_short_panels(qapp):
    # At the 672 px fit the Subtitles panels scroll; a wheel turn over the
    # JP size the user never clicked into scrolls them and leaves the value
    # alone. Seeded defect: drop _PanelScroll's wheel filter -> JP size steps
    # down and the panels stay put -> red.
    from PySide6.QtCore import QPoint, QPointF, QRect, Qt
    from PySide6.QtGui import QWheelEvent

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    w.show()
    QApplication.processEvents()
    w._fit_to_screen(avail=QRect(0, 0, 1920, 672))
    QApplication.processEvents()
    bar = w._panel_scroll.verticalScrollBar()
    spin = w._jp_size
    before = spin.value()
    assert not spin.hasFocus() and bar.value() == 0 < bar.maximum()
    at = QPointF(5, spin.height() / 2)
    QApplication.sendEvent(spin, QWheelEvent(
        at, QPointF(spin.mapToGlobal(at.toPoint())), QPoint(0, 0), QPoint(0, -120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    ))
    assert (spin.value(), bar.value() > 0) == (before, True), (
        f"a wheel step over the unfocused JP size left it at {spin.value()} "
        f"(was {before}) and the panels at {bar.value()}"
    )
    w.close()



def test_fit_to_screen_keeps_the_design_size_when_the_work_area_has_room(qapp):
    # A 1040 px work area holds the window: 728 px tall, the monitor at its
    # full size, the panels not scrolled. Seeded defects: fit to the current
    # height (the window stays as short as the last screen made it), or let
    # the panel area scroll at the design size -> red.
    from PySide6.QtCore import QRect

    from yamabiko.overlay_app.preview_monitor import MONITOR_H, MONITOR_W
    from yamabiko.overlay_app.settings_window import WINDOW_H

    w = SettingsWindow(current=dict(STYLE_DEFAULTS))
    w.show()
    QApplication.processEvents()  # the show-time fit to the offscreen screen runs first
    w._fit_to_screen(avail=QRect(0, 0, 1920, 1040))
    QApplication.processEvents()
    assert w.height() == WINDOW_H
    assert (w._preview.width(), w._preview.height()) == (MONITOR_W, MONITOR_H)
    bar = w._panel_scroll.verticalScrollBar()
    assert not bar.isVisible() and bar.value() == 0
    w.close()


def _record(field, key, modifiers):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mouseClick(field, Qt.LeftButton)
    QTest.keyClick(field, key, modifiers)


def test_shortcuts_tab_records_persists_and_emits_the_change(qapp, appdata):
    # Record Ctrl+Alt+K for Pause and resume: Apply writes hotkeys.pause_resume
    # and emits only that key. Seeded defects: leave hotkeys out of the
    # commit, or emit unchanged keys too -> red.
    from PySide6.QtCore import Qt

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml", page="shortcuts")
    w.show()
    assert w._stack.currentIndex() == 4 and w._tabs.currentIndex() == 4
    emitted = []
    w.shortcutsChanged.connect(emitted.append)
    _record(w._shortcut_fields["pause_resume"], Qt.Key_K, Qt.ControlModifier | Qt.AltModifier)
    assert w._apply()
    assert _read_overrides(appdata)["hotkeys"] == {"pause_resume": "Ctrl+Alt+K"}
    assert emitted == [{"pause_resume": "Ctrl+Alt+K"}]
    w.close()


def test_rerecording_an_unchanged_combo_sends_it_again(qapp, appdata):
    # A user who re-records a shortcut that failed to bind must get it sent
    # again even though the combo equals the saved one (FR1-1); the next
    # Apply without a re-record sends nothing. The saved combo is an
    # override (Ctrl+Alt+K): the sparse writer drops a value equal to the
    # shipped default, so re-recording Ctrl+Alt+P would leave no hotkeys key
    # to read back. Seeded defects: drop re-recorded unchanged combos, or
    # never clear _rerecorded -> red.
    from PySide6.QtCore import Qt

    from yamabiko.config_writer import write_override

    write_override(["hotkeys", "pause_resume"], "Ctrl+Alt+K")
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml", page="shortcuts")
    w.show()
    emitted = []
    w.shortcutsChanged.connect(emitted.append)
    _record(w._shortcut_fields["pause_resume"], Qt.Key_K, Qt.ControlModifier | Qt.AltModifier)
    assert w._apply()
    assert emitted == [{"pause_resume": "Ctrl+Alt+K"}]
    assert _read_overrides(appdata)["hotkeys"] == {"pause_resume": "Ctrl+Alt+K"}
    w._apply()
    assert emitted == [{"pause_resume": "Ctrl+Alt+K"}]
    w.close()


def test_reset_after_a_rerecord_sends_nothing_when_it_restores_the_saved_combo(qapp, appdata):
    # Decision 38: Reset to default takes the row out of the re-recorded set,
    # so a re-record undone by Reset (the shipped default is the saved combo)
    # leaves nothing to send. Seeded defect: keep the row re-recorded on
    # Reset -> red.
    from PySide6.QtCore import Qt

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml", page="shortcuts")
    w.show()
    emitted = []
    w.shortcutsChanged.connect(emitted.append)
    _record(w._shortcut_fields["pause_resume"], Qt.Key_K, Qt.ControlModifier | Qt.AltModifier)
    w._shortcut_resets["pause_resume"].click()
    assert w._apply()
    assert emitted == []
    w.close()


def test_shortcut_row_reset_restores_the_shipped_default(qapp, appdata):
    # The per-row Reset. Seeded defect: reset to the saved value instead of
    # the shipped default -> red.
    from yamabiko.config_writer import write_override

    write_override(["hotkeys", "toggle_overlay"], "Ctrl+Alt+J")
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml")
    assert w._shortcut_fields["toggle_overlay"].combo() == "Ctrl+Alt+J"
    w._shortcut_resets["toggle_overlay"].click()
    assert w._shortcut_fields["toggle_overlay"].combo() == "Ctrl+Alt+O"
    assert w._apply()
    assert "hotkeys" not in _read_overrides(appdata)
    w.close()


def test_unparseable_saved_shortcut_survives_an_unrelated_apply(qapp, appdata):
    # A hand-edited combo Settings can't record (the pipeline still binds
    # "ctrl+alt+space" at boot) shows as saved, and an Apply that only
    # changes a style value neither rewrites nor re-sends it. Seeded defect:
    # show the shipped default for it -> the Apply replaces it -> red.
    from yamabiko.config_writer import write_override

    write_override(["hotkeys", "pause_resume"], "ctrl+alt+space")
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml")
    assert w._shortcut_fields["pause_resume"].combo() == "ctrl+alt+space"
    emitted = []
    w.shortcutsChanged.connect(emitted.append)
    w._jp_size.setValue(30)
    assert w._apply()
    overrides = _read_overrides(appdata)
    assert overrides["overlay"]["style"]["jp_size"] == 30
    assert overrides["hotkeys"] == {"pause_resume": "ctrl+alt+space"}
    assert emitted == []
    w.close()


def test_disabled_saved_shortcut_shows_empty_and_row_reset_turns_it_back_on(qapp, appdata):
    # A shortcut turned off by hand (hotkeys.pause_resume: "", which the
    # pipeline skips) shows an empty field, and an unrelated Apply keeps it
    # off; the row Reset then Apply turns it back on. Seeded defect: show the
    # shipped default for it -> the Reset changes nothing, the Apply saves
    # nothing and the shortcut stays off -> red.
    from yamabiko.config_writer import write_override

    write_override(["hotkeys", "pause_resume"], "")
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
                       config_path=appdata / "user_overrides.yaml")
    assert w._shortcut_fields["pause_resume"].combo() == ""
    emitted = []
    w.shortcutsChanged.connect(emitted.append)
    w._jp_size.setValue(30)
    assert w._apply()
    assert _read_overrides(appdata)["hotkeys"] == {"pause_resume": ""}
    assert emitted == []
    w._shortcut_resets["pause_resume"].click()
    assert w._apply()
    # The shipped Ctrl+Alt+P prunes the override (config_writer's sparse write).
    assert "hotkeys" not in _read_overrides(appdata)
    assert emitted == [{"pause_resume": "Ctrl+Alt+P"}]
    w.close()


def test_taken_and_duplicate_shortcuts_show_the_amber_message(qapp):
    # "Used by another app" comes from the probe at open. The first row in
    # page order keeps a shared combo and only the later row warns; Apply
    # and OK stay disabled until the duplicate clears. Seeded defects:
    # never probe at open, miss duplicates, or leave Apply enabled on a
    # duplicate -> red.
    from PySide6.QtCore import Qt

    w = SettingsWindow(current=dict(STYLE_DEFAULTS),
                       hotkey_probe=lambda combo: combo == "Ctrl+Alt+S")
    w.show()
    w.show_page("shortcuts")
    message = w._shortcut_messages["show_series"]
    assert w._shortcut_fields["show_series"].warning()
    assert message.isVisible() and message.text() == "Used by another app. Pick a different one."
    _record(w._shortcut_fields["pause_resume"], Qt.Key_O, Qt.ControlModifier | Qt.AltModifier)
    messages = w._shortcut_messages
    assert messages["pause_resume"].text() == ""
    assert messages["toggle_overlay"].text() == "Already used for Pause and resume."
    assert not w._apply_btn.isEnabled()
    assert not w._ok_btn.isEnabled()
    assert w._apply() is False
    w._reset_shortcut("pause_resume")
    assert messages["pause_resume"].text() == ""
    assert messages["toggle_overlay"].text() == ""
    assert w._apply_btn.isEnabled()
    assert w._ok_btn.isEnabled()
    w.close()


def test_series_shortcut_row_absent_when_series_feature_disabled(qapp, appdata):
    # Same gate as the pipeline, which never binds the series shortcut with
    # the feature off. Seeded defect: build the row unconditionally -> red.
    from yamabiko.config_writer import write_override

    write_override(["gazetteer", "enabled"], False)
    w = SettingsWindow(current=dict(STYLE_DEFAULTS), config_path=appdata / "user_overrides.yaml")
    assert set(w._shortcut_fields) == {"pause_resume", "toggle_overlay"}
    w.close()


def test_move_subtitles_is_disabled_without_a_mover(qapp):
    """Move subtitles… only works when the tray passes a mover. Without one
    the button is disabled and the help says the subtitles move while
    Yamabiko is running; with one, a click calls it.

    Seeded defect: drop the ``self._move_btn.setEnabled(...)`` line from the
    Placement panel -> ``not w._move_btn.isEnabled()`` goes red.
    """
    from PySide6.QtWidgets import QLabel
    from yamabiko.overlay_app import settings_window as settings_window_mod

    w = SettingsWindow(current=dict(STYLE_DEFAULTS), list_devices=lambda: [])
    assert not w._move_btn.isEnabled()
    help_texts = {label.text() for label in w.findChildren(QLabel)}
    assert settings_window_mod._PLACEMENT_UNAVAILABLE_HELP in help_texts
    assert settings_window_mod._PLACEMENT_HELP not in help_texts
    w.close()

    calls: list[int] = []
    w = SettingsWindow(
        current=dict(STYLE_DEFAULTS), list_devices=lambda: [],
        on_move_subtitles=lambda: calls.append(1),
    )
    assert w._move_btn.isEnabled()
    assert settings_window_mod._PLACEMENT_HELP in {
        label.text() for label in w.findChildren(QLabel)
    }
    w._move_btn.click()
    assert calls == [1]
    w.close()
