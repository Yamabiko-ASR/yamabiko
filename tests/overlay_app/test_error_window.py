"""The startup error window ("Report card", plan phase 6), offscreen.

Painted parts (Task 3): the balanced title and the no-lone-word items, code
chips kept whole, Copy report's two states, the condensed log view. The
window (Task 4): its texts and facts, geometry, theme, Copy report, Close,
Open log folder and show_startup_error's Qt branch. Open Settings (Task 5).
The native platform (clipboard after os._exit, foreground, DWM) is
test_real_error_window.py's.
"""
from __future__ import annotations

import logging
import logging.handlers
import math
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QFontMetricsF, QGuiApplication, QImage, QRegion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from yamabiko.log_config import MAIN_LOG_FORMAT
from yamabiko.overlay_app.controls import mincho_font
from yamabiko.overlay_app.fonts import register_bundled_fonts
from yamabiko.overlay_app.theme import MIST, NIGHT
from yamabiko.ui import error_report as er
from yamabiko.ui import error_window as ew
from yamabiko.ui.error_dialog import StartupError
from tests.overlay_app._qt_drain import drain_qt_debris

_LOG = [
    "2026-11-01 21:41:05,001 p4242 MainThread yamabiko INFO Yamabiko 0.2.0-beta starting",
    "2026-11-01 21:41:05,101 p4242 MainThread yamabiko.gpu WARNING nvidia-smi reported no GPU",
    "2026-11-01 21:41:07,101 p4242 cuda_0 yamabiko.asr.asr_client ERROR speech model failed to load",
    "Traceback (most recent call last):",
    '  File "yamabiko\\asr\\asr_client.py", line 96, in load',
    '    self._model = WhisperModel(path, device="cuda")',
    "RuntimeError: CUDA driver version is insufficient for CUDA runtime version",
    "2026-11-01 21:41:07,200 p4242 MainThread yamabiko.ui.error_dialog ERROR startup error dialog: T | M",
]


def _facts(log_path: Path | None = None, lines: list[str] | None = None) -> er.ReportFacts:
    return er.ReportFacts("0.2.0-beta", "11 24H2, build 26100.2161", "RTX 4060, 8 GB, driver 566.36",
                          "3.12.10", "FileNotFoundError: x.gguf",
                          er.LogTail(log_path, _LOG if lines is None else lines))


def _render(widget: QWidget) -> QImage:
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def _at(widget: QWidget, window: QWidget, x: int, y: int) -> QPoint:
    return widget.mapTo(window, QPoint(x, y))


# ---------------------------------------------------------------------------
# Painted parts (Task 3)
# ---------------------------------------------------------------------------


def test_title_balances_its_lines_and_items_never_end_on_a_lone_word(qtbot):
    """L9: the title wraps balanced (CSS text-wrap: balance) on a 28 px line
    box; a What to try item keeps company for its last word (the mockup's
    "Reconnect it, then start / Yamabiko again.").

    Seeded defects: greedy title ("No NVIDIA graphics / card found"), the
    font's own line height instead of 28, no last-word guard, or lines cut
    at Qt's UTF-16 offsets as if they were code points (an emoji in a
    device name shifts every later line) -> red."""
    register_bundled_fonts()
    title = ew.TextBlock("No NVIDIA graphics card found", mincho_font(21), 28, MIST, wrap="balance")
    qtbot.addWidget(title)
    title.resize(240, title.heightForWidth(240))
    assert title.lines() == ["No NVIDIA", "graphics card found"]
    assert title.height() == 56
    headset = ew.TextBlock("\U0001F3A7 Headphones isn't connected", mincho_font(21), 28, MIST,
                           wrap="balance")
    qtbot.addWidget(headset)
    headset.resize(240, headset.heightForWidth(240))
    assert headset.lines() == ["\U0001F3A7 Headphones", "isn't connected"]
    font = QFont()
    font.setFamilies(["BIZ UDPGothic"])
    font.setPixelSize(13)
    text = "Reconnect it, then start Yamabiko again."
    width = 15 + math.ceil(QFontMetricsF(font).horizontalAdvance("Reconnect it, then start Yamabiko")) + 2
    item = ew.TextBlock(text, font, 19, MIST, wrap="pretty", indent=15, marker=True)
    qtbot.addWidget(item)
    item.resize(width, item.heightForWidth(width))
    assert item.lines() == ["Reconnect it, then start", "Yamabiko again."]
    assert item.height() == 38


def test_a_code_chip_moves_to_the_next_line_whole(qtbot):
    """L9: "%LOCALAPPDATA%\\Yamabiko" is one code chip: at a width where the
    line could break inside it (after "%"), the whole chip moves down.

    Seeded defect: no word joiners in the chip -> the line ends with
    "%LOCALAPPDATA%" -> red."""
    register_bundled_fonts()
    font = QFont()
    font.setFamilies(["BIZ UDPGothic"])
    font.setPixelSize(13)
    text = "To download them again, delete state.json in %LOCALAPPDATA%\\Yamabiko, then start Yamabiko."
    item = ew.TextBlock(text, font, 19, MIST, wrap="greedy", chips=ew.CODE_CHIPS)
    qtbot.addWidget(item)
    head = "To download them again, delete state.json in %LOCALAPPDATA%"
    width = math.ceil(QFontMetricsF(font).horizontalAdvance(head)) + 2
    item.resize(width, item.heightForWidth(width))
    assert any(line.startswith("%LOCALAPPDATA%\\Yamabiko,") for line in item.lines()), item.lines()
    assert item.accessibleName() == text


def test_copy_report_turns_to_its_copied_state_and_keeps_its_name(qtbot):
    """L9: Copy report is accent-filled; copied, it shows the field tone
    inside a 1.5 px accent border. Its name stays "Copy report" (UI
    Automation invokes it by name); the copied sentence is its description.

    Seeded defects: no copied state, the copied state still accent-filled,
    or the name changed to the copied sentence -> red."""
    button = ew.CopyReportButton(MIST)
    qtbot.addWidget(button)
    button.resize(382, 36)
    assert _render(button).pixelColor(3, 3) == QColor(MIST["accent"])
    button.set_copied()
    image = _render(button)
    assert button.copied()
    assert image.pixelColor(3, 3) == QColor(MIST["field"])
    assert image.pixelColor(0, 18) == QColor(MIST["accent"])
    assert (button.text(), button.accessibleName()) == ("Copy report", "Copy report")
    assert button.accessibleDescription() == "Copied. Paste it into a message to the developer."


def test_log_view_condenses_colors_hangs_and_opens_on_the_newest_line(qtbot):
    """L8: records condensed to "HH:MM:SS name LEVEL message" (time muted,
    WARNING bold ink, ERROR accent), the traceback's last line accent bold,
    a 2-character hanging indent on 16 px lines, scrolled to the newest.

    Seeded defects: raw lines shown, a level uncolored, no hanging indent
    (the second line starts at 0), or opened at the top -> red."""
    long = "2026-11-01 21:41:08,000 p4242 MainThread yamabiko.capture.audio INFO " + "word " * 40
    view = ew.LogView(MIST)
    qtbot.addWidget(view)
    view.set_lines(_LOG * 2 + [long])
    view.show()
    qtbot.waitExposed(view)
    document = view.document()
    first = document.firstBlock()
    assert first.text() == "21:41:05 yamabiko INFO Yamabiko 0.2.0-beta starting"

    def color_at(block, position: int) -> QColor:
        cursor = view.textCursor()
        cursor.setPosition(block.position() + position + 1)
        return cursor.charFormat().foreground().color()

    assert color_at(first, 0) == QColor(MIST["muted"])            # the time
    assert color_at(first, len("21:41:05 ")) == QColor(MIST["ink"])  # the name
    warning = first.next()
    assert warning.text() == "21:41:05 gpu WARNING nvidia-smi reported no GPU"
    cursor = view.textCursor()
    cursor.setPosition(warning.position() + len("21:41:05 gpu W"))
    assert cursor.charFormat().fontWeight() == QFont.Weight.Bold
    error = warning.next()
    assert color_at(error, len("21:41:07 asr_client ")) == QColor(MIST["accent"])
    exception = error.next().next().next().next()
    assert exception.text().startswith("RuntimeError: CUDA driver version")
    cursor.setPosition(exception.position() + 1)
    assert cursor.charFormat().foreground().color() == QColor(MIST["accent"])
    assert cursor.charFormat().fontWeight() == QFont.Weight.Bold
    last = document.lastBlock()
    layout = last.layout()
    two_ch = 2 * QFontMetricsF(ew.mono_font(12)).horizontalAdvance("0")
    assert layout.lineCount() > 1
    assert layout.lineAt(0).x() == 0
    assert layout.lineAt(1).x() == pytest.approx(two_ch)
    assert layout.lineAt(1).y() - layout.lineAt(0).y() == pytest.approx(16)
    bar = view.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()


def test_log_view_shows_six_whole_lines_under_its_top_strip_at_the_newest(qtbot):
    """Design doc 6: inside its 1 px border the box keeps a 6 px top strip
    in the field tone, then six whole 16 px lines, then 5 px (107 = 6 +
    6 x 16 + 5). Scrolled to the newest line, the text area starts right
    under the strip on a line boundary (every line box is 16 px from the
    document's top), so no part of the line above shows.

    Seeded defect: the text area 2 px taller than six lines (the old
    viewport margins 5 and 4) -> scrolled to the end, its top 14 px hold
    the bottom of the seventh line from the end -> red."""
    view = ew.LogView(MIST)
    qtbot.addWidget(view)
    view.set_lines(_LOG * 2)
    view.show()
    qtbot.waitExposed(view)
    bar = view.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    text_top = view.viewport().mapTo(view, QPoint(0, 0)).y()
    assert (text_top, view.viewport().height(), bar.value() % 16) == (1 + 6, 6 * 16, 0)


# ---------------------------------------------------------------------------
# The window (Task 4)
# ---------------------------------------------------------------------------


@pytest.fixture
def window_for(qtbot):
    """Build an ErrorWindow (fonts registered, height fitted, shown).

    Holds every window it built until its teardown (pytest-qt has closed
    them by then), then closes the Settings windows they opened (pytest-qt
    never tracked those, and a failed assert skips a test's own close) and
    destroys them all. A window's wrapper is in a reference cycle (its
    ClientAreaFit keeps the window), so without this hold only the garbage
    collector frees it once the test returns: a collection triggered inside
    pytest-qt's post-call event pump deleted the still-shown window from
    within its own mast's paint (access violation in
    landscape._paint_scene)."""
    register_bundled_fonts()
    built: list[ew.ErrorWindow] = []

    def build(err: StartupError, facts: er.ReportFacts | None = None, palette: dict = MIST):
        window = ew.ErrorWindow(err, facts or _facts(), palette)
        built.append(window)
        qtbot.addWidget(window)
        window.refit_height()
        window.show()
        qtbot.waitExposed(window)
        return window

    yield build
    for window in built:
        if window.settings_window is not None:
            window.settings_window.close()
    built.clear()
    drain_qt_debris()


def test_window_shows_the_errors_texts_and_the_report_facts(window_for):
    """L9: title, message, What to try (one item per sentence), the panel's
    "Report" / "For the developer", the four fact labels and values, "Recent
    log", the footer note and Close; 720 wide, the Report panel 412 wide at
    the body's right padding, 24 px after the left column, and the height
    fitted to the panel (one-line facts: 445).

    Seeded defects: a missing or renamed label, a fact in the wrong row,
    What to try as one block, or the panel at another width or place ->
    red."""
    err = StartupError.models_missing(details="FileNotFoundError: x.gguf")
    window = window_for(err)
    assert window.windowTitle() == "Yamabiko"
    assert window.title_label.text() == "Some model files are missing"
    assert [item.text() for item in window.findChildren(ew.TextBlock)
            if item.accessibleName() in err.what_to_try()] == err.what_to_try()
    labels = {label.text() for label in window.findChildren(QLabel)}
    assert {"What to try", "Report", "For the developer", "Yamabiko", "Windows", "GPU", "Error",
            "Recent log", ew.FOOTER_NOTE} <= labels
    assert [window.fact_value(name) for name in ("Yamabiko", "Windows", "GPU", "Error")] == [
        "0.2.0-beta", "11 24H2, build 26100.2161", "RTX 4060, 8 GB, driver 566.36",
        "FileNotFoundError: x.gguf"]
    assert window.settings_button is None
    assert (window.open_log_button.text(), window.close_button.text(),
            window.copy_button.text()) == ("Open log folder", "Close", "Copy report")
    panel = window.findChild(QWidget, "reportPanel")
    assert (window.width(), panel.width()) == (720, 412)
    assert panel.geometry().right() + 1 == 720 - 20
    assert window.title_label.x() == 22
    assert panel.x() - (window.title_label.geometry().right() + 1) == 24
    assert panel.height() == 331 and window.height() == 44 + 8 + 331 + 14 + 48


def test_window_shows_the_message_and_what_to_try_without_the_user_name(window_for, monkeypatch):
    """Ruling I2 / the footer's promise: a hand-built message (the
    watchdog's give-ups put the exception text in it) and the What to try
    sentences show with the user name taken out, as the report copies them;
    the title stays as the error gives it (a device name such as "Tester
    Person's headset" is not a path, and the bare-name rule would turn it
    into "<user>'s headset").

    Seeded defects: the message or the items built from err.message /
    err.what_to_try() without redact_user -> the profile folder on screen
    -> red; the title redacted too -> red."""
    local = r"C:\Users\Tester Person\AppData\Local"
    monkeypatch.setenv("LOCALAPPDATA", local)
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Tester Person")
    monkeypatch.setenv("USERNAME", "Tester Person")
    spawn = FileNotFoundError(2, "The system cannot find the file specified",
                              local + r"\Yamabiko\yamabiko.exe")
    window = window_for(StartupError(
        title="Tester Person's headset isn't connected",
        message=f"The pipeline process could not be spawned: {spawn}",
        suggested_fix=r"Restart Yamabiko. Or reinstall it to C:\Users\Tester Person\Apps.",
        details="d"))
    assert window.title_label.text() == "Tester Person's headset isn't connected"
    assert window._message.text() == (
        "The pipeline process could not be spawned: [Errno 2] The system cannot find the "
        r"file specified: '%LOCALAPPDATA%\\Yamabiko\\yamabiko.exe'")
    assert [item.text() for item in window._items] == [
        "Restart Yamabiko.", r"Or reinstall it to %USERPROFILE%\Apps."]


def test_the_error_value_names_the_exception_type_without_its_module(window_for):
    """Lead decision (fix pass 3, amends L4's panel field): the Error value
    shows the exception's own type name, not its dotted module path; a
    message that merely starts with a dotted name stays whole; the copied
    report keeps the whole error line.

    Seeded defects: show facts.error as it is -> the module path on screen
    -> red; shorten facts.error itself -> the copied report loses the path
    -> red; strip any dotted start -> the config message loses "vad." -> red."""
    line = ("huggingface_hub.errors.LocalEntryNotFoundError: Cannot find an appropriate "
            "cached snapshot folder")
    window = window_for(StartupError.models_missing(details=line), _facts()._replace(error=line))
    assert window.fact_value("Error") == (
        "LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder")
    window.copy_button.click()
    assert f"\nError: {line}\n" in QGuiApplication.clipboard().text()
    config = "vad.min_chunk_ms must be less than or equal to vad.max_chunk_ms"
    window = window_for(StartupError("Invalid configuration", "M", "F", config),
                        _facts()._replace(error=config))
    assert window.fact_value("Error") == config


def test_the_error_value_shows_three_lines_at_most_the_third_ending_in_an_ellipsis(window_for):
    """Lead decision (fix pass 3): a long Error value takes three lines at
    most (1 + 3 x 17 px), the third ending in an ellipsis, so one long
    exception line cannot stretch the window; the copied report keeps the
    whole line.

    Seeded defect: no cap -> the value takes all its lines and the window
    grows with them -> red."""
    line = "RuntimeError: " + " ".join(["the model server failed while loading its weights"] * 6)
    err = StartupError(title="T", message="M", suggested_fix="F", details=line)
    window = window_for(err, _facts()._replace(error=line))
    value = next(block for block in window.findChildren(ew.TextBlock)
                 if block.text() == window.fact_value("Error"))
    shown = value.lines()
    assert (len(shown), shown[-1][-1], value.height()) == (3, "\u2026", 1 + 3 * 17)
    window.copy_button.click()
    assert f"\nError: {line}\n" in QGuiApplication.clipboard().text()


@pytest.mark.parametrize("palette", [MIST, NIGHT], ids=["mist", "night"])
def test_window_paints_in_the_theme_it_was_given(window_for, palette):
    """L9 / design doc 1: the ground, the raised Report panel with its line
    border, the field log box with its edge border, the accent Copy report
    and the strip's near ridge, in both palettes.

    Seeded defects: a hardcoded color, the panel's or the log's QSS rule
    missing, or the mast painting the Settings band -> red."""
    window = window_for(StartupError.gpu_unavailable(details="d"), palette=palette)
    image = _render(window)
    panel = window.findChild(QWidget, "reportPanel")
    assert image.pixelColor(5, 200) == QColor(palette["ground"])
    assert image.pixelColor(_at(panel, window, 5, 5)) == QColor(palette["raised"])
    assert image.pixelColor(_at(panel, window, 0, 40)) == QColor(palette["line"])
    assert image.pixelColor(_at(window.log_view, window, 3, 3)) == QColor(palette["field"])
    assert image.pixelColor(_at(window.log_view, window, 0, 40)) == QColor(palette["edge"])
    assert image.pixelColor(_at(window.copy_button, window, 3, 3)) == QColor(palette["accent"])
    assert image.pixelColor(100, 39) == QColor(palette["ridge1"])


def test_copy_report_puts_the_report_on_the_clipboard_on_every_click(window_for):
    """L9 / L4: Copy report copies report_text(err, facts) and turns copied;
    a second click copies again (the user pasted something else meanwhile).

    Seeded defects: copy the old payload or the condensed log, copy only
    once, or never turn copied -> red."""
    err = StartupError.gpu_unavailable(details="Stage: ASR init\n\nRuntimeError: no CUDA")
    facts = _facts()
    window = window_for(err, facts)
    board = QGuiApplication.clipboard()
    window.copy_button.click()
    assert board.text() == er.report_text(err, facts)
    assert window.copied()
    board.setText("something else")
    window.copy_button.click()
    assert board.text() == er.report_text(err, facts)


def test_copy_report_keeps_its_normal_state_when_the_clipboard_flush_fails(window_for, monkeypatch):
    """L10: OleFlushClipboard is still refused (another process holds the
    clipboard open, CLIPBRD_E_CANT_OPEN) when the retry budget runs out, or
    fails with another HRESULT, which is not retried; the report would then
    leave with the pipeline's exit 43, so Copy report never says "Copied."
    and stays accent-filled for another try.

    Seeded defects: set_copied() whatever the copy did -> copied -> red;
    retry every OSError (or every failure until the deadline) -> E_FAIL
    flushed more than once -> red."""
    import types

    class _WindowsQt(QGuiApplication):  # the flush runs on the native platform only
        platformName = staticmethod(lambda: "windows")

    flushes: list[int] = []
    hresult = {"now": -2147221040}  # CLIPBRD_E_CANT_OPEN

    class _FailingOle32:
        def OleFlushClipboard(self):  # noqa: N802 (Win32 name)
            flushes.append(hresult["now"])
            # How OleDLL reports a failed HRESULT.
            raise OSError(None, "OleFlushClipboard failed", None, hresult["now"])

    window = window_for(StartupError.gpu_unavailable(details="d"))
    monkeypatch.setattr(ew, "QGuiApplication", _WindowsQt)
    monkeypatch.setattr(ew, "ctypes", types.SimpleNamespace(OleDLL=lambda name: _FailingOle32()))
    monkeypatch.setattr(ew, "_FLUSH_RETRY_S", 0.05)
    window.copy_button.click()  # busy until the retry budget runs out
    assert not window.copied()
    assert window.copy_button.accessibleDescription() == ""
    assert _render(window.copy_button).pixelColor(3, 3) == QColor(MIST["accent"])
    flushes.clear()
    hresult["now"] = -2147467259  # E_FAIL
    window.copy_button.click()
    assert flushes == [-2147467259]
    assert not window.copied()
    assert window.copy_button.accessibleDescription() == ""


def test_copy_report_retries_the_flush_while_another_process_holds_the_clipboard(
        window_for, monkeypatch, caplog):
    """L10: clipboard listeners (TextInputHost, explorer, the clipboard
    service ...) open the clipboard as soon as a copy lands, so the first
    OleFlushClipboard can fail with CLIPBRD_E_CANT_OPEN (frozen E2E: 2 of 7
    runs). The copy flushes again, logs the tries it took (INFO) and Copy
    report says "Copied."

    Seeded defects: no retry -> one flush, not copied -> red; no log line
    for a retried flush -> red."""
    import types

    class _WindowsQt(QGuiApplication):  # the flush runs on the native platform only
        platformName = staticmethod(lambda: "windows")

    flushes: list[str] = []

    class _BusyOnceOle32:
        def OleFlushClipboard(self):  # noqa: N802 (Win32 name)
            flushes.append("flush")
            if len(flushes) == 1:
                raise OSError(None, "OpenClipboard Failed", None, -2147221040)

    caplog.set_level(logging.INFO, logger=ew.__name__)
    window = window_for(StartupError.gpu_unavailable(details="d"))
    monkeypatch.setattr(ew, "QGuiApplication", _WindowsQt)
    monkeypatch.setattr(ew, "ctypes", types.SimpleNamespace(OleDLL=lambda name: _BusyOnceOle32()))
    window.copy_button.click()
    assert len(flushes) == 2
    assert window.copied()
    assert window.copy_button.accessibleDescription() == ew.CopyReportButton.COPIED_TEXT
    assert [r.getMessage() for r in caplog.records if r.name == ew.__name__] == [
        "clipboard flush succeeded after 2 tries"]


def test_copying_from_the_recent_log_goes_through_the_flushing_copy(window_for, monkeypatch):
    """Ruling M6 / L10: text copied out of the Recent log box (Ctrl+C, or
    the Copy of its context menu) goes through copy_to_clipboard, which
    flushes it, so it survives the pipeline's os._exit(43) like Copy
    report's text; Qt's own copy would leave with the process.

    Seeded defects: no keyPressEvent override (Ctrl+C is Qt's copy) -> one
    copy recorded -> red; the menu's Copy left on Qt's slot -> one copy
    recorded -> red."""
    copied: list[str] = []
    monkeypatch.setattr(ew, "copy_to_clipboard", copied.append)
    window = window_for(StartupError.gpu_unavailable(details="d"))
    view = window.log_view
    view.selectAll()
    view.setFocus()
    QTest.keyClick(view, Qt.Key_C, Qt.KeyboardModifier.ControlModifier)
    view.context_menu(QPoint(1, 1)).findChild(QAction, "edit-copy").trigger()
    selected = view.textCursor().selection().toPlainText()
    assert selected.startswith("21:41:05 yamabiko INFO Yamabiko 0.2.0-beta starting\n")
    assert copied == [selected, selected]


@pytest.mark.parametrize("how", ["close", "caption", "esc"])
def test_close_the_caption_close_and_esc_each_end_the_window(qtbot, window_for, how):
    """L9: Close, the caption's close and Esc end the window: it hides and
    emits finished, which ends run_error_window's loop.

    Seeded defects: an unwired button or Esc swallowed -> the window stays
    visible -> red; an action wired to hide() -> no finished, the loop and
    the pipeline would wait forever -> red."""
    window = window_for(StartupError.gpu_unavailable(details="d"))
    with qtbot.waitSignal(window.finished, timeout=1000):
        if how == "close":
            window.close_button.click()
        elif how == "caption":
            window._mast.caption.closeRequested.emit()
        else:
            QTest.keyClick(window, Qt.Key_Escape)
    assert not window.isVisible()


def test_open_log_folder_opens_the_folder_of_the_tails_file(window_for, monkeypatch, tmp_path):
    """L9: Open log folder opens the folder holding the log the tail came
    from (the logs folder when there was none), creating the app folders
    first.

    Seeded defects: always the logs folder (wrong for the %TEMP% fallback
    log), the file itself, no click wiring, or tail.parent without the None
    guard (AttributeError, nothing opens) -> red."""
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))
    opened: list[Path] = []
    monkeypatch.setattr("os.startfile", lambda path: opened.append(Path(path)), raising=False)
    tail = tmp_path / "Yamabiko-logs" / "main-pipeline.log"
    window = window_for(StartupError.gpu_unavailable(details="d"), _facts(log_path=tail))
    window.open_log_button.click()
    no_tail = window_for(StartupError.gpu_unavailable(details="d"), _facts(log_path=None))
    no_tail.open_log_button.click()
    assert opened == [tail.parent, paths.logs_dir()]
    assert paths.logs_dir().is_dir()


@pytest.mark.timeout(60)
def test_show_startup_error_shows_the_window_with_the_tail_ending_on_its_dialog_line(
        qtbot, monkeypatch, tmp_path):
    """L7-L9 through the real entry point: show_startup_error logs its line,
    then the window opens (marker held, fonts registered, the saved dark
    theme, centered on the primary screen's work area) with the facts and a
    log tail that ends with that very line, although the font registration
    logs to the same file after it (the handler sits on "yamabiko", as the
    watchdog's does; the pipeline's is on the root).

    Seeded defects: read the tail before logging the line, or after the
    fonts register (the fonts line ends the tail), open the window outside
    held(), skip the fonts, ignore overlay.theme, or not center -> red."""
    import uuid

    from PySide6.QtGui import QFontDatabase

    from yamabiko.lifecycle import startup_dialog
    from yamabiko.ui import error_dialog as ed

    marker = rf"Local\yamabiko.test.{uuid.uuid4().hex}"
    monkeypatch.setenv(startup_dialog.ENV_VAR, marker)
    (tmp_path / "Yamabiko").mkdir(exist_ok=True)
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))
    (tmp_path / "Yamabiko" / "user_overrides.yaml").write_text(
        "overlay:\n  theme: dark\n", encoding="utf-8")
    # The system scheme answers the opposite of the saved theme: offscreen Qt
    # reports no scheme, and the registry fallback says dark on a dark-mode
    # machine, which would keep "ignore overlay.theme" green there.
    monkeypatch.setattr("yamabiko.overlay_app.theme.system_scheme_probe", lambda: "light")
    monkeypatch.setattr(er.gpu, "detect_gpu_summary", lambda: None)
    monkeypatch.setattr(er, "windows_version", lambda: "11 26H2, build 26300.9818")
    log_file = tmp_path / "main-pipeline.log"
    handler = logging.handlers.RotatingFileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(MAIN_LOG_FORMAT))
    logger = logging.getLogger("yamabiko")
    level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    seen: dict = {}

    def inspect():
        window = next(w for w in QApplication.topLevelWidgets()
                      if isinstance(w, ew.ErrorWindow) and w.isVisible())
        try:
            seen["held"] = startup_dialog.is_held(marker)
            seen["title"] = window.title_label.text()
            seen["gpu"] = window.fact_value("GPU")
            seen["last"] = window.log_view.document().lastBlock().text()
            seen["ground"] = _render(window).pixelColor(5, 200)
            seen["pos"] = window.pos()
            seen["size"] = window.size()
            seen["mincho"] = "BIZ UDPMincho" in QFontDatabase.families()
        finally:
            window.close_button.click()

    QTimer.singleShot(0, inspect)
    try:
        ed.show_startup_error(StartupError.gpu_unavailable(details="Stage: ASR init\n\nRuntimeError: x"))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        handler.close()
    assert "overlay.fonts.registered" in log_file.read_text(encoding="utf-8")
    area = QGuiApplication.primaryScreen().availableGeometry()
    assert seen["held"] is True
    assert seen["title"] == "No NVIDIA graphics card found"
    assert seen["gpu"] == "None reported by Windows"
    assert seen["last"].endswith(
        "error_dialog ERROR startup error dialog: No NVIDIA graphics card found | "
        "Yamabiko needs an NVIDIA graphics card and couldn't find one it can use.")
    assert seen["ground"] == QColor(NIGHT["ground"])
    assert seen["mincho"]
    assert seen["pos"].x() == area.x() + (area.width() - seen["size"].width()) // 2


# ---------------------------------------------------------------------------
# Open Settings (Task 5)
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_settings(monkeypatch, tmp_path):
    """A user with a customized subtitle size, a saved device that is gone,
    and two loopback devices Settings lists."""
    from yamabiko import paths
    from yamabiko.overlay_app import settings_window

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "audio:\n  device: Gone Buds [Loopback]\noverlay:\n  style:\n    jp_size: 20\n",
        encoding="utf-8")
    monkeypatch.setattr(settings_window, "_default_list_devices", lambda: [
        {"index": 1, "name": "Speakers (Realtek) [Loopback]"},
        {"index": 2, "name": "Headphones (USB Audio) [Loopback]"},
    ])
    return paths.user_overrides_path()


def test_open_settings_shows_only_for_a_settings_page_and_opens_audio_and_model(
        window_for, saved_settings):
    """L11: only the missing-device case has "Open Settings"; it opens
    Settings at Audio & model as its own window, with the fonts it has when
    the tray opens it (owned by the styled error window, its labels fell to
    the 9 pt application font).

    Seeded defects: the button on every case, Settings parented to the
    window (point-sized label fonts), or opened at its first tab -> red."""
    from yamabiko.overlay_app.settings_window import PAGE_NAMES

    assert window_for(StartupError.gpu_unavailable(details="d")).settings_button is None
    window = window_for(StartupError.audio_device_gone(device_name="Gone Buds [Loopback]", details="d"))
    assert window.settings_button.text() == "Open Settings"
    window.settings_button.click()
    settings = window.settings_window
    assert settings is not None and settings.isVisible()
    assert settings.parentWidget() is None
    assert all(label.font().pixelSize() > 0 for label in settings.findChildren(QLabel))
    assert settings._tabs.currentIndex() == PAGE_NAMES.index("audio")
    settings.close()


def test_settings_saves_the_new_device_and_keeps_the_users_style(window_for, saved_settings):
    """L11: OK in Settings opened from the window saves the new device to
    user_overrides.yaml and keeps the user's subtitle style (every Apply
    writes the whole style: seeded with the defaults it would reset
    jp_size 20 to 16, which the sparse file then drops).

    Seeded defects: seed Settings with the defaults (KeyError 'overlay':
    the whole style section pruned), or pass no config path (Settings then
    saves nothing) -> red."""
    import yaml

    window = window_for(StartupError.audio_device_gone(device_name="Gone Buds [Loopback]", details="d"))
    window.settings_button.click()
    settings = window.settings_window
    combo = settings.audio_device_combo
    combo.setCurrentIndex(combo.findText("Headphones (USB Audio) [Loopback]"))
    settings._ok_btn.click()
    saved = yaml.safe_load(saved_settings.read_text(encoding="utf-8"))
    assert saved["audio"]["device"] == "Headphones (USB Audio) [Loopback]"
    assert saved["overlay"]["style"]["jp_size"] == 20


@pytest.mark.timeout(60)
def test_run_error_window_returns_only_after_settings_closes(qtbot, monkeypatch, saved_settings):
    """L11: Settings takes a real click while the error window is open
    (nothing modal blocks it); the window closes first and Settings stays
    open; the call (and with it the startup-dialog marker and the process)
    returns only once Settings is closed too.

    Seeded defects: a modal error window (setModal(True), or exec() on the
    hidden window: the click is blocked) or return without waiting for
    Settings (os._exit would then kill it) -> red. A loop that never ends
    (Close unwired) -> the backstop closes what is open after 5 s -> red,
    instead of pytest-timeout ending the whole session."""
    from yamabiko.overlay_app.settings_window import PAGE_NAMES, SettingsWindow

    monkeypatch.setattr(er.gpu, "detect_gpu_summary", lambda: None)
    monkeypatch.setattr(er, "windows_version", lambda: "11 26H2, build 26300.9818")
    events: list[str] = []
    state: dict = {}

    def error_window():
        return next((w for w in QApplication.topLevelWidgets()
                     if isinstance(w, ew.ErrorWindow) and w.isVisible()), None)

    def open_settings():
        try:
            window = error_window()
            window.settings_button.click()
            settings = window.settings_window
            state["settings"] = settings
            timing = PAGE_NAMES.index("timing")
            tab = settings._tabs._tabs[timing]
            QTest.mouseClick(settings.windowHandle(), Qt.LeftButton, Qt.NoModifier,
                             tab.mapTo(settings, tab.rect().center()))
            events.append("settings took a click" if settings._tabs.currentIndex() == timing
                          else "settings click blocked")
        finally:
            QTimer.singleShot(50, close_window)  # the window always closes

    def close_window():
        window = error_window()
        if window is not None:
            window.close_button.click()
            events.append("window closed")
        QTimer.singleShot(100, close_settings)

    def close_settings():
        settings = state.get("settings")
        if settings is None:
            return
        events.append("settings open" if settings.isVisible() else "settings gone")
        settings.close()
        events.append("settings closed")

    def backstop():
        # Decision 68's watchdog: the call should have returned long ago.
        events.append("backstop closed what was open")
        for w in QApplication.topLevelWidgets():
            if isinstance(w, (ew.ErrorWindow, SettingsWindow)) and w.isVisible():
                w.close()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(backstop)
    watchdog.start(5000)
    QTimer.singleShot(50, open_settings)
    try:
        ew.run_error_window(StartupError.audio_device_gone(device_name="Gone Buds [Loopback]", details="d"))
    finally:
        watchdog.stop()
    events.append("returned")
    assert events == ["settings took a click", "window closed", "settings open",
                      "settings closed", "returned"]


@pytest.mark.timeout(60)
def test_run_error_window_hides_its_windows_when_it_raises_after_show(qtbot, monkeypatch, saved_settings):
    """A failure after show() (here take_foreground, with Settings already
    open) leaves neither window shown: past the call only each window's
    ClientAreaFit cycle keeps it, and a garbage collection inside a later
    paint (the watchdog's next queued error) would free a shown one in the
    middle of painting it.

    Seeded defects: no try/finally around the body, or one that hides the
    error window only -> a window still shown -> red."""
    monkeypatch.setattr(er.gpu, "detect_gpu_summary", lambda: None)
    monkeypatch.setattr(er, "windows_version", lambda: "11 26H2, build 26300.9818")
    windows: list[QWidget] = []

    def foreground_fails(window):
        window.settings_button.click()
        windows.extend([window, window.settings_window])
        raise RuntimeError("seeded failure after show()")

    monkeypatch.setattr(ew, "take_foreground", foreground_fails)
    with pytest.raises(RuntimeError, match="seeded failure"):
        ew.run_error_window(StartupError.audio_device_gone(device_name="Gone Buds [Loopback]", details="d"))
    shown = [type(w).__name__ for w in windows if w.isVisible()]
    for w in windows:
        w.close()
    assert shown == []


def test_text_block_set_text_rewraps_and_renames(qtbot):
    """Setup's live lines (the alert message, the header) change their text
    in place.

    Seeded defects: keep the old wrap cache or the old accessible name ->
    red."""
    from yamabiko.overlay_app.controls import ui_font

    block = ew.TextBlock("Paused", ui_font(14), 20, MIST)
    qtbot.addWidget(block)
    assert block.heightForWidth(300) == 20
    text = "Not enough space on drive C:. Free up 2.1 GB, then click Retry."
    block.set_text(text)
    height = block.heightForWidth(300)
    assert height >= 40
    assert block.accessibleName() == text
    block.resize(300, height)
    assert " ".join(block.lines()) == text
