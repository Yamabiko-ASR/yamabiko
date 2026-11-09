"""The setup window's shell (plan phase 7, Task 5), offscreen: the footer,
the keys (Esc, Enter, Tab) and the caption's close, where the focus goes when
a step turns its primary off, navigation (Next, Back, rail jumps), the header,
the first-show foreground and run_wizard's theme and fonts. What
accept()/reject() persist is test_wizard_flow.py's; the native window (size,
title, system menu, caption buttons) is test_real_wizard_window.py's."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")


@pytest.fixture(autouse=True)
def instant_gpu_probe(monkeypatch):
    """QuantTierPage starts its GPU detection thread in __init__: instant
    probes keep that thread's life to microseconds."""
    from yamabiko import gpu

    monkeypatch.setattr(gpu, "detect_vram_mib", lambda: None)
    monkeypatch.setattr(gpu, "detect_gpu_name", lambda: None)


@pytest.fixture
def wizard(qtbot):
    """A wizard held until teardown, its page threads released before Qt
    destroys it (a running page QThread at destruction is a qFatal abort)."""
    from yamabiko.wizard import YamabikoWizard

    wiz = YamabikoWizard()
    qtbot.addWidget(wiz, before_close_func=lambda w: w.release_background_threads())
    wiz._audio_page._list_devices = lambda: [{"index": 0, "name": "Fake Loopback Device"}]
    yield wiz
    wiz.release_background_threads()


def _footer(wiz) -> tuple[bool, bool, str]:
    return (wiz.cancel_button.isVisibleTo(wiz), wiz.back_button.isVisibleTo(wiz),
            wiz.primary_button.text())


def test_footer_follows_the_page_position_and_the_pages_primary_text(wizard):
    """L2 / design §5: the first page has Cancel and its own primary text
    (QWizardPage.setButtonText) but no Back; a middle page Cancel, Back and
    "Next". (The last page's footer: test_wizard_flow's walk.)

    Seeded defects: show Back on the first page, or ignore the page's
    button text -> red."""
    wizard.restart()
    assert _footer(wizard) == (True, False, "Get started")
    wizard.next()
    assert _footer(wizard) == (True, True, "Next")


def test_back_and_rail_jumps_clean_up_the_page_left_without_reinitializing_the_target(
    wizard,
):
    """L2: Next validates the current page and initializes the next one;
    Back and a rail jump clean up only the page left and show the target as
    it was. Every switch emits currentIdChanged (the interim save). An
    upcoming rail row does nothing.

    Seeded defects: initialize the target of a jump, skip cleanupPage on
    Back, or let an upcoming row jump forward -> red."""
    calls: list[str] = []
    pages = (wizard.page(0), wizard.page(1), wizard.page(2))
    for name, page in zip(("first", "second", "third"), pages):
        real_init, real_cleanup = page.initializePage, page.cleanupPage
        page.initializePage = lambda n=name, f=real_init: (calls.append(f"init {n}"), f())[1]
        page.cleanupPage = lambda n=name, f=real_cleanup: (calls.append(f"cleanup {n}"), f())[1]
    ids: list[int] = []
    wizard.currentIdChanged.connect(ids.append)
    wizard.restart()
    wizard.next()
    wizard.next()
    wizard.rail.step_button(0).click()
    wizard.next()
    wizard.back_button.click()
    wizard.rail.step_button(2).click()
    assert calls == ["init first", "init second", "init third", "cleanup third",
                     "init second", "cleanup second"]
    assert ids == [0, 1, 2, 0, 1, 0]
    assert wizard.currentId() == 0


def test_a_jump_to_the_current_or_a_later_page_and_back_on_the_first_page_do_nothing(wizard):
    """L2: only pages before the current one can be jumped to; Back on the
    first page (its button hidden) does nothing. Neither cleans up a page,
    initializes one or emits currentIdChanged.

    Seeded defects: drop _jump's upper bound (target < current), or the
    whole guard -> red."""
    calls: list[str] = []
    for index in (0, 1, 2):
        page = wizard.page(index)
        real_init, real_cleanup = page.initializePage, page.cleanupPage
        page.initializePage = lambda n=index, f=real_init: (calls.append(f"init {n}"), f())[1]
        page.cleanupPage = lambda n=index, f=real_cleanup: (calls.append(f"cleanup {n}"), f())[1]
    wizard.restart()
    ids: list[int] = []
    wizard.currentIdChanged.connect(ids.append)
    wizard.back()
    assert (wizard.currentId(), ids) == (0, [])
    wizard.next()
    wizard._jump(1)
    wizard._jump(2)
    assert (wizard.currentId(), ids) == (1, [1])
    assert calls == ["init 0", "init 1"]


def test_a_save_failed_dialog_opened_from_setup_keeps_its_30_px_buttons(wizard, monkeypatch):
    """d6b14ef: setup's descendant button rules (32 px, 96 wide) follow the
    parent chain into a compact dialog parented to the wizard; "Couldn't
    save your settings" keeps its own 30 px buttons.

    Seeded defect: drop the QDialog#setupWizard QDialog#compactDialog rules
    -> its buttons are 32 px tall -> red."""
    from yamabiko import wizard as wizard_mod
    from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog
    from yamabiko.wizard_pages.rail import BRAND

    monkeypatch.setattr(wizard_mod, "take_foreground", lambda window: True)
    wizard.show()
    dialog = SaveFailedDialog(wizard._palette, "C:/x/state.json", OSError("disk full"), wizard,
                              title=BRAND)
    dialog.show()
    try:
        assert (dialog.cancel_button.height(), dialog.retry_button.height()) == (30, 30)
    finally:
        dialog.close()


def test_primary_follows_the_current_pages_complete_state(wizard):
    """L2: the primary is enabled exactly while the current page reports
    complete, refreshed on its completeChanged.

    Seeded defect: never connect completeChanged -> the primary keeps the
    state it had when the page showed -> red."""
    wizard.restart()
    welcome = wizard.page(0)
    ready = {"ok": False}
    welcome.isComplete = lambda: ready["ok"]
    welcome.completeChanged.emit()
    assert not wizard.primary_button.isEnabled()
    ready["ok"] = True
    welcome.completeChanged.emit()
    assert wizard.primary_button.isEnabled()


def _shown(wizard, monkeypatch) -> None:
    """Show and activate the window (key events go to the active window's
    focus widget, offscreen too) without taking the real foreground."""
    from PySide6.QtTest import QTest

    from yamabiko import wizard as wizard_mod

    monkeypatch.setattr(wizard_mod, "take_foreground", lambda window: True)
    wizard.show()
    wizard.activateWindow()
    assert QTest.qWaitForWindowActive(wizard)


def test_esc_cancels_setup_and_saves_the_current_picks(wizard, monkeypatch, qtbot):
    """L2: Esc is Cancel: setup closes rejected, and the picks made on the
    current step reach state.json for the next launch (F_N1).

    Seeded defects: ignore Esc (a keyPressEvent that swallows it) -> no
    rejected -> red; drop the interim save from reject() -> state.json keeps
    the device saved on entering Sound -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko import wizard_state

    wizard._audio_page._list_devices = lambda: [
        {"index": 0, "name": "Speakers [Loopback]", "default": True},
        {"index": 1, "name": "Headphones [Loopback]"},
    ]
    _shown(wizard, monkeypatch)
    wizard.next()  # Sound; entering it saves the preselected Speakers
    QTest.mouseClick(wizard._audio_page.device_list.segment("Headphones [Loopback]"),
                     Qt.LeftButton)

    with qtbot.waitSignal(wizard.rejected, timeout=2000):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key_Escape)

    assert not wizard.isVisible()
    assert wizard_state.load().audio_device_name == "Headphones [Loopback]"


def test_enter_presses_the_primary_even_with_cancel_focused(wizard, monkeypatch):
    """L2: Enter is the primary action (the default button); the footer's
    other buttons are never the default, so Enter on a focused Cancel still
    goes on to the next step.

    Seeded defects: drop setDefault(True) on the primary -> Enter does
    nothing -> red; leave autoDefault on for the footer -> Enter clicks
    Cancel -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _shown(wizard, monkeypatch)
    wizard.cancel_button.setFocus(Qt.OtherFocusReason)
    assert wizard.cancel_button.hasFocus()

    QTest.keyClick(wizard.cancel_button, Qt.Key_Return)

    assert wizard.currentId() == 1
    assert wizard.isVisible()


def test_the_caption_close_button_cancels_setup(wizard, monkeypatch, qtbot):
    """L2: the caption's close (X) is Cancel: setup closes rejected.

    Seeded defect: drop the closeRequested -> reject connection -> the
    window stays open -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _shown(wizard, monkeypatch)

    with qtbot.waitSignal(wizard.rejected, timeout=2000):
        QTest.mouseClick(wizard.caption.caption_button("close"), Qt.LeftButton)

    assert not wizard.isVisible()


def _rail_row_ring_on(row) -> bool:
    """Whether the rail row paints its keyboard ring (the accent rectangle
    along its edge)."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage, QRegion
    from PySide6.QtWidgets import QWidget

    from yamabiko.overlay_app.theme import MIST

    image = QImage(row.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    row.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image.pixelColor(row.width() // 2, 0).name() == MIST["accent"]


@pytest.fixture
def start_download_clicked(wizard, monkeypatch, qtbot):
    """Setup shown and walked to Quality, then "Start download" clicked: the
    click focuses the primary first, as a user's does, and the Download step
    turns it off while the queue runs. Every file waits for the yielded gate
    (set at teardown), so the queue succeeds only once the test sets it."""
    import threading
    import types
    from pathlib import Path

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from yamabiko.wizard_pages import downloads

    gate = threading.Event()

    class _Waiting:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            gate.wait(10.0)
            return Path("C:/fake-cache") / filename

    roomy = types.SimpleNamespace(total=2**40, used=0, free=2**40)
    monkeypatch.setattr(downloads, "shutil", types.SimpleNamespace(disk_usage=lambda path: roomy))
    wizard._downloads_page._downloader = _Waiting()
    _shown(wizard, monkeypatch)
    try:
        for _step in ("Sound", "Content", "Quality"):
            wizard.next()
        qtbot.waitUntil(wizard.primary_button.isEnabled, timeout=5000)  # the card check ended
        QTest.mouseClick(wizard.primary_button, Qt.LeftButton)
        assert (wizard.currentId(), wizard.primary_button.isEnabled()) == (4, False)
        yield gate
    finally:
        gate.set()


def test_a_primary_the_next_step_turns_off_leaves_the_focus_on_that_step(
    wizard, start_download_clicked,
):
    """L2: the Download step turns off the "Start download" just clicked;
    the focus goes to the step, not where Qt hands a disabled widget's focus
    on (along the tab chain, past the footer to the rail's first finished
    row, where Enter jumps back to Welcome).

    Seeded defect: drop the shell's focus move on a page switch -> the
    rail's Welcome row has the focus -> red."""
    from PySide6.QtWidgets import QApplication

    focus = QApplication.focusWidget()
    assert focus is wizard.currentPage(), focus


def test_a_focused_primary_that_turns_off_mid_step_leaves_the_focus_on_the_step(
    wizard, monkeypatch,
):
    """L2: the step reporting incomplete again turns off the primary that
    has the focus; the focus goes to the step, not where Qt hands a disabled
    widget's focus on (along the tab chain to the rail's first finished row,
    where Enter jumps back to Welcome).

    Seeded defect: drop _refresh_primary's focus move -> the rail's Welcome
    row has the focus -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    _shown(wizard, monkeypatch)
    wizard.next()  # Sound: the Welcome row is finished
    sound = wizard.currentPage()
    ready = {"ok": True}
    sound.isComplete = lambda: ready["ok"]
    sound.completeChanged.emit()
    wizard.primary_button.setFocus(Qt.OtherFocusReason)
    assert wizard.primary_button.hasFocus()

    ready["ok"] = False
    sound.completeChanged.emit()

    focus = QApplication.focusWidget()
    assert focus is sound, focus


def test_enter_on_the_download_step_goes_on_once_the_queue_succeeds(
    wizard, start_download_clicked, qtbot,
):
    """L2: Enter is the primary. After "Start download" and a queue that
    succeeds, Enter goes on to Translator.

    Seeded defect: drop the shell's focus move on a page switch -> Enter
    reaches the rail's Welcome row, which jumps back to Welcome -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    class _SilentProbe:  # Translator's check: nothing answers, nothing is sent
        def check_health(self) -> bool:
            return False

        def translate(self, text: str) -> str:
            raise AssertionError("translate must not run without health")

    wizard._smoke_page._probe = _SilentProbe()
    start_download_clicked.set()
    qtbot.waitUntil(wizard.primary_button.isEnabled, timeout=5000)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Return)

    assert wizard.currentId() == 5


def test_tab_from_the_step_still_reaches_a_finished_rail_row(wizard, start_download_clicked):
    """Task 4: the focus on the step is no wall: Tab goes on from it through
    the footer into the rail, where a finished row takes the focus and shows
    the keyboard ring (a key moved it).

    Seeded defect: keep the rail's finished rows out of the tab chain (focus
    policy ClickFocus) -> Tab never reaches the rail -> red."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    for _press in range(6):  # the step -> Cancel -> Back -> (Next is off) -> the rail
        QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
        if wizard.rail.isAncestorOf(QApplication.focusWidget()):
            break

    row = QApplication.focusWidget()
    assert wizard.rail.isAncestorOf(row) and row.state == "done", row
    assert _rail_row_ring_on(row)


def test_space_on_retry_keeps_the_focus_in_the_step_so_a_second_space_never_cancels(
    wizard, monkeypatch, qtbot,
):
    """L6 / L2: Tab to Retry in the Download step's stopped state and press
    Space: the alert hides while the queue runs again, and the focus stays in
    the step, not where Qt's tab chain would carry it (Cancel). Once the
    queue stops again, a second Space does not cancel setup.

    Seeded defect: drop the shell's hand-over into the step -> the focus
    lands on Cancel and the second Space closes setup -> red."""
    import types

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yamabiko.wizard_pages import downloads

    class _Offline:
        def download(self, repo_id, filename, revision=None, progress_callback=None):
            raise ConnectionError("the network is down")

    roomy = types.SimpleNamespace(total=2**40, used=0, free=2**40)
    monkeypatch.setattr(downloads, "shutil", types.SimpleNamespace(disk_usage=lambda path: roomy))
    page = wizard._downloads_page
    page._downloader = _Offline()
    _shown(wizard, monkeypatch)
    for _step in ("Sound", "Content", "Quality"):
        wizard.next()
    qtbot.waitUntil(wizard.primary_button.isEnabled, timeout=5000)  # the card check ended
    wizard.next()
    qtbot.waitUntil(page.retry_button.isVisible, timeout=5000)  # stopped
    for _press in range(6):
        if page.retry_button.hasFocus():
            break
        QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
    assert page.retry_button.hasFocus(), QApplication.focusWidget()

    QTest.keyClick(page.retry_button, Qt.Key_Space)

    focus = QApplication.focusWidget()
    assert focus is not None and page.isAncestorOf(focus), focus
    qtbot.waitUntil(page.retry_button.isVisible, timeout=5000)  # stopped again
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Space)
    assert (wizard.isVisible(), wizard.currentId()) == (True, 4)


def test_header_shows_the_pages_title_and_lead_and_hides_without_a_title(wizard):
    """L1 / Decision 20: the shell paints the page's title() and subTitle()
    (the lead) above it, read on every switch; a page without a title has
    no header (Welcome draws its own).

    Seeded defects: read the texts once, or keep an empty header visible ->
    red."""
    wizard.page(0).setTitle("")
    wizard.page(1).setTitle("Where does the sound play?")
    wizard.page(1).setSubTitle("Pick the output your PC plays sound through.")
    wizard.restart()
    assert not wizard.header.isVisibleTo(wizard)
    wizard.next()
    assert wizard.header.isVisibleTo(wizard)
    assert wizard.header.title.text() == "Where does the sound play?"
    assert wizard.header.lead.text() == "Pick the output your PC plays sound through."


def test_each_step_names_itself_with_its_rail_label_for_screen_readers(wizard):
    """L2: every switch puts the focus on the step itself, so a screen
    reader announces the step's accessible name: its rail label.

    Seeded defect: leave the pages unnamed -> the focus lands on an unnamed
    pane -> red."""
    from PySide6.QtGui import QAccessible

    names = [QAccessible.queryAccessibleInterface(wizard.page(i)).text(QAccessible.Name)
             for i in wizard.pageIds()]
    assert names == ["Welcome", "Sound", "Content", "Quality", "Download", "Translator", "Done"]


def test_the_first_show_takes_the_foreground_once(wizard, monkeypatch):
    """Decision 134 (R5): the pipeline child shows setup while the user works
    in another window, so the first show takes the foreground the way the
    startup error window does (window_chrome.take_foreground, after the
    native show); a later show (a restore from the taskbar) does not.

    Seeded defects: drop the call -> red; call it on every show -> red."""
    from yamabiko import wizard as wizard_mod

    taken: list[bool] = []
    monkeypatch.setattr(wizard_mod, "take_foreground",
                        lambda window: taken.append(window.isVisible()) or True)
    wizard.show()
    wizard.hide()
    wizard.show()
    assert taken == [True]


def test_run_wizard_registers_the_fonts_and_builds_in_the_saved_theme(qtbot, monkeypatch):
    """L13: run_wizard registers the bundled fonts and resolves the saved
    theme once (user_overrides.yaml overlay.theme) before it builds the
    window.

    Seeded defects: skip register_bundled_fonts, or build with the default
    palette -> red."""
    from yamabiko import paths
    from yamabiko import wizard as wizard_mod
    from yamabiko.overlay_app import fonts
    from yamabiko.overlay_app.theme import NIGHT

    paths.ensure_dirs()
    paths.user_overrides_path().write_text("overlay:\n  theme: dark\n", encoding="utf-8")
    registered: list[bool] = []
    real_register = fonts.register_bundled_fonts
    monkeypatch.setattr(fonts, "register_bundled_fonts",
                        lambda: registered.append(True) or real_register())
    seen: dict = {}

    def fake_exec(self):
        seen["palette"] = self._palette
        seen["fonts"] = list(registered)
        return 0

    monkeypatch.setattr(wizard_mod.YamabikoWizard, "exec", fake_exec)
    wizard_mod.run_wizard()
    assert seen == {"palette": NIGHT, "fonts": [True]}
