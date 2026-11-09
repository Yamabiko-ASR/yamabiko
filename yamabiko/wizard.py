"""First-run wizard driver: the "Step rail" setup window (design record
Surface 3, opt-wiz-rail.tpl).

``YamabikoWizard`` is an 880 x 600 dialog titled "Yamabiko setup": the
painted caption (app mark, title, minimize, close) over a 220 px step rail
(``wizard_pages.rail.StepRail``) and the content column, whose header shows
the current page's ``title()`` (BIZ UDPMincho 26 px) and ``subTitle()`` (the
lead), then the page, then the footer: Cancel at the left, Back and the
page's primary action at the right. The pages stay ``QWizardPage``
subclasses; the shell calls their hooks the way QWizard does (Next:
``validatePage`` then the next page's ``initializePage``; Back and rail
jumps: ``cleanupPage`` of the page left) and keeps a QWizard-shaped API
(``addPage``, ``page``, ``pageIds``, ``currentId``, ``currentPage``,
``currentIdChanged``, ``restart``, ``next``, ``back``). On accept it writes
the choices to user_overrides.yaml, then a completed ``WizardResult`` via
``wizard_state.save``; every page change saves an interim one.

Usage from main.py:
    if wizard_state.needs_wizard():
        run_wizard()  # blocks until closed
        if wizard_state.needs_wizard():  # user cancelled
            exit_wizard_cancelled()  # 42
    # ... start pipeline
"""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget, QWizard,
    QWizardPage,
)

from yamabiko import __version__, paths, wizard_state
from yamabiko.lifecycle.exit_codes import EXIT_WIZARD_CANCEL
from yamabiko.lifecycle.launch_markers import WIZARD_START_MARKER
from yamabiko.overlay_app.controls import mincho_font, ui_font
from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog
from yamabiko.overlay_app.theme import MIST
from yamabiko.overlay_app.window_chrome import (
    ClientAreaFit, DialogCaption, add_window_menu, apply_window_qss, expand_client_area,
    nonclient_event, set_border_color, take_foreground,
)
from yamabiko.ui import thread_keeper
from yamabiko.ui.error_window import TextBlock
from yamabiko.wizard_pages.audio import AudioPage
from yamabiko.wizard_pages.content import ContentPage
from yamabiko.wizard_pages.done import DonePage
from yamabiko.wizard_pages.downloads import (
    DownloadsPage, _size_text, download_queue, pending_download_bytes,
)
from yamabiko.wizard_pages.quant_tier import QuantTierPage
from yamabiko.wizard_pages.rail import BRAND, StepRail
from yamabiko.wizard_pages.smoke import SmokePage
from yamabiko.wizard_pages.welcome import WelcomePage

_log = logging.getLogger(__name__)

WINDOW_W, WINDOW_H = 880, 600
FOOTER_H = 60
LEAD_W = 540
# Keep QApplication alive for the lifetime of the process if we had to
# construct one ad-hoc. Otherwise Qt's C++ side can reference a deleted
# Python object during wizard teardown.
_owned_app: "object | None" = None


def _update_download_text() -> str | None:
    """What the Download step will fetch, as Welcome's update line shows it
    ("6.6 GB"): the pinned sizes of the queued files not cached yet. None
    when nothing is left to fetch."""
    total = pending_download_bytes(download_queue())
    return _size_text(total) if total > 0 else None


class _Header(QWidget):
    """The step title (BIZ UDPMincho 26 px on a 34 px line) and the lead
    (14/21, at most 540 wide, 8 px under the title); hidden when the page
    has no title (Welcome draws its own)."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.title = TextBlock("", mincho_font(26), 34, palette)
        self.lead = TextBlock("", ui_font(14), 21, palette, wrap="pretty", top_pad=8)
        self.lead.setMaximumWidth(LEAD_W)
        column.addWidget(self.title)
        column.addWidget(self.lead)

    def show_page(self, page: QWizardPage) -> None:
        self.title.set_text(page.title())
        self.lead.set_text(page.subTitle())
        self.lead.setVisible(bool(page.subTitle()))
        self.setVisible(bool(page.title()))


class YamabikoWizard(QDialog):
    currentIdChanged = Signal(int)

    def __init__(self, palette: dict = MIST) -> None:
        super().__init__(None)
        self.setObjectName("setupWizard")  # the window QSS's ground, 32 px buttons
        self.setWindowTitle(BRAND)
        icon_path = paths.app_icon_path()
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        expand_client_area(self)
        self.setFont(ui_font(13))
        self._palette = palette
        self._pages: list[QWizardPage] = []
        self._current = -1
        self._took_foreground = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.caption = DialogCaption(BRAND, palette, self, minimize=True)
        self.caption.closeRequested.connect(self.reject)
        root.addWidget(self.caption)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.rail = StepRail(palette)
        self.rail.stepClicked.connect(self._jump)
        body.addWidget(self.rail)
        main = QVBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        content = QVBoxLayout()
        content.setContentsMargins(40, 30, 40, 24)
        content.setSpacing(0)
        self.header = _Header(palette)
        content.addWidget(self.header)
        self._stack = QStackedWidget()
        content.addWidget(self._stack, 1)
        main.addLayout(content, 1)
        main.addWidget(self._build_footer())
        body.addLayout(main, 1)
        root.addLayout(body, 1)
        self._fit = ClientAreaFit(self, root)
        self.setFixedSize(WINDOW_W, WINDOW_H)

        # The state setup opened with: Welcome's update line, the interim
        # prefill below, and the completion time interim saves keep.
        prior = wizard_state.load()
        self._last_completed_at = (
            prior.wizard_completed_at if prior is not None else None
        )
        update_download = (
            _update_download_text()
            if wizard_state.speech_model_update_pending(prior) else None
        )
        self.addPage(
            WelcomePage(app_version=__version__, palette=palette,
                        update_download=update_download),
            "Welcome",
        )
        self._audio_page = AudioPage(palette=self._palette)
        self.addPage(self._audio_page, "Sound")
        self._content_page = ContentPage(palette=self._palette)
        self.addPage(self._content_page, "Content")

        # F_N1: prefill from partial state.json if an earlier wizard
        # session was cancelled mid-flow. wizard_state.load() returns
        # the persisted WizardResult even when first_run_complete is
        # False. AudioPage applies the name after its lazily-populated
        # device list is ready; genre can be selected immediately.
        interim = (
            prior if prior is not None and not prior.first_run_complete else None
        )
        self._audio_page.set_interim_device_name(
            interim.audio_device_name if interim is not None else None
        )
        # Genre: prefer the interrupted-session pick; otherwise seed
        # from the merged config. The config seed matters on a
        # completed-wizard RE-RUN: accept() writes the genre
        # unconditionally, so if the choice started at the default
        # instead of the user's current effective genre, clicking straight
        # through would silently prune a deliberate earlier pick
        # (PR #43 review finding 1). With the seed, an untouched choice is
        # a true round-trip and a moved one is an explicit reversal.
        genre_seed = interim.genre if interim is not None and interim.genre else None
        if genre_seed is None:
            try:
                from yamabiko.config import load_full_config
                genre_seed = load_full_config().get("genre")
            except Exception:
                genre_seed = None  # unseeded choice beats a crashed wizard
        self._content_page.set_genre(genre_seed)
        # Quality sits right before Download because the tier pick only
        # matters to the download queue: Back from Download lands on it.
        self._quant_tier_page = QuantTierPage(palette=self._palette)
        self.addPage(self._quant_tier_page, "Quality")
        self._downloads_page = DownloadsPage(palette=self._palette)
        self.addPage(self._downloads_page, "Download")
        self._smoke_page = SmokePage(palette=self._palette)
        self.addPage(self._smoke_page, "Translator")
        self.addPage(DonePage(palette=palette), "Done")
        # The Download step's progress drives the rail's line under
        # Download on every step (L3).
        self._downloads_page.railProgressChanged.connect(self.rail.set_download)
        # Skip, in the Translator step's failure alert, goes on to Done (L7).
        self._smoke_page.skipped.connect(self.next)

        # F_N1: save interim state on every page change (Next, Back, a
        # rail jump). We snapshot the current selections (with
        # first_run_complete=False) to state.json. If the user cancels
        # later, the next wizard session restores these picks via the
        # prefill block above.
        self.currentIdChanged.connect(self._persist_interim_state)
        apply_window_qss(self, palette)

    # --- building -------------------------------------------------------

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("footer")  # the window QSS's 1 px line on top
        footer.setFixedHeight(FOOTER_H)
        row = QHBoxLayout(footer)
        row.setContentsMargins(24, 1, 24, 0)
        row.setSpacing(8)
        self.cancel_button = QPushButton("Cancel")
        self.back_button = QPushButton("Back")
        self.primary_button = QPushButton("Next")
        self.primary_button.setProperty("primary", True)
        self.primary_button.setDefault(True)  # Enter = the primary action
        for button in (self.cancel_button, self.back_button, self.primary_button):
            button.setFont(ui_font(13))
            button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        self.back_button.clicked.connect(self.back)
        self.primary_button.clicked.connect(self._primary)
        row.addWidget(self.cancel_button, 0, Qt.AlignVCenter)
        row.addStretch(1)
        row.addWidget(self.back_button, 0, Qt.AlignVCenter)
        row.addWidget(self.primary_button, 0, Qt.AlignVCenter)
        return footer

    # --- the QWizard-shaped API -------------------------------------------

    def addPage(self, page: QWizardPage, label: str) -> int:  # noqa: N802 (QWizard naming)
        """Add ``page`` as the next step, ``label`` its rail row and its
        accessible name (the step has the focus after every switch, so a
        screen reader announces it); its id."""
        page_id = len(self._pages)
        self._pages.append(page)
        page.setAccessibleName(label)
        self._stack.addWidget(page)
        self.rail.add_step(label)
        page.completeChanged.connect(self._refresh_primary)
        return page_id

    def page(self, page_id: int) -> QWizardPage | None:
        return self._pages[page_id] if 0 <= page_id < len(self._pages) else None

    def pageIds(self) -> list[int]:  # noqa: N802 (QWizard naming)
        return list(range(len(self._pages)))

    def currentId(self) -> int:  # noqa: N802 (QWizard naming)
        return self._current

    def currentPage(self) -> QWizardPage | None:  # noqa: N802 (QWizard naming)
        return self.page(self._current)

    def restart(self) -> None:
        """Enter the first page (initializing it), as the first show does.
        Before the first page only: unlike QWizard.restart() it cleans up no
        page, so a call mid-flow would leave the current page running."""
        if self._pages:
            self._enter(0, initialize=True)

    def next(self) -> None:
        """Validate the current page, then initialize and show the next one
        (QWizard: completeness gates the button, not this call)."""
        page = self.currentPage()
        if page is None or self._current + 1 >= len(self._pages) or not page.validatePage():
            return
        self._enter(self._current + 1, initialize=True)

    def back(self) -> None:
        self._jump(self._current - 1)

    # --- navigation -------------------------------------------------------

    def _jump(self, target: int) -> None:
        """Back or a rail jump: clean up the page left; the target shows as
        it was (no initializePage)."""
        if not 0 <= target < self._current:
            return
        self._pages[self._current].cleanupPage()
        self._enter(target, initialize=False)

    def _enter(self, index: int, *, initialize: bool) -> None:
        page = self._pages[index]
        if initialize:
            page.initializePage()
        left, focus = self.currentPage(), self.focusWidget()
        if left is not None and focus is not None and left.isAncestorOf(focus):
            # The stack drops a focus the step left holds itself, but hands one
            # a control in it holds (Skip) on along the tab chain, to Cancel.
            left.setFocus(Qt.OtherFocusReason)
        self._current = index
        self._stack.setCurrentIndex(index)
        self._focus_page()  # before the rail row, Back or the primary holding it turn off
        self.header.show_page(page)
        self.rail.set_current(index)
        first, last = index == 0, index == len(self._pages) - 1
        self.cancel_button.setVisible(not last)
        self.back_button.setVisible(not first and not last)
        which = QWizard.WizardButton.FinishButton if last else QWizard.WizardButton.NextButton
        self.primary_button.setText(page.buttonText(which) or ("Finish" if last else "Next"))
        self._refresh_primary()
        self.currentIdChanged.emit(index)

    def _refresh_primary(self) -> None:
        page = self.currentPage()
        complete = page is not None and page.isComplete()
        if not complete and self.focusWidget() is self.primary_button:
            self._focus_page()
        self.primary_button.setEnabled(complete)

    def _focus_page(self) -> None:
        """Put the focus on the current step (L2: Enter is the primary).

        Qt hands the focus of a widget that turns off (disabled or hidden)
        on along the tab chain, which runs from the footer to the rail's
        first finished row, where Enter jumps back to that step. So on every
        page switch, and before the focused primary turns off, the step
        takes it: Enter, which a page leaves alone, goes on to the dialog's
        default button, and Tab still leads on into the footer and the rail.
        A step that wants a control focused instead makes it its focus proxy
        (one that leaves Enter alone). OtherFocusReason: no key moved the
        focus, so nothing shows a keyboard ring."""
        page = self.currentPage()
        if page is not None:
            page.setFocus(Qt.OtherFocusReason)

    def focusNextPrevChild(self, forward: bool) -> bool:  # noqa: N802 (Qt override)
        """Tab and Shift+Tab move the focus along the tab chain. Qt also
        calls this to hand on the focus of a widget that turns off (disabled
        or hidden), as a step's own buttons do when pressed (Download's Retry
        and Delete old Yamabiko models, Translator's Retry, Sound's Test and
        Look again): the chain would carry it out of the step to Cancel,
        where the next Space cancels setup. It stays in the step instead: on
        the step's next control that takes Tab, else on the step. The reason
        is Qt's own for a hand-over, so the keyboard ring follows the last
        input. (The footer's primary hands its focus to the step before it
        turns off: _refresh_primary, _enter.)"""
        focus, page = self.focusWidget(), self.currentPage()
        if (focus is None or page is None or not page.isAncestorOf(focus)
                or (focus.isEnabled() and focus.isVisibleTo(self))):
            return super().focusNextPrevChild(forward)
        target = focus
        while True:
            target = target.nextInFocusChain() if forward else target.previousInFocusChain()
            if target is focus:
                target = page
                break
            if (page.isAncestorOf(target) and target.focusPolicy() & Qt.TabFocus
                    and target.isEnabled() and target.isVisibleTo(self)
                    and target.focusProxy() is None):
                break
        target.setFocus(Qt.TabFocusReason if forward else Qt.BacktabFocusReason)
        return True

    def _primary(self) -> None:
        if self._current == len(self._pages) - 1:
            self.accept()
        else:
            self.next()

    # --- persistence ------------------------------------------------------

    def _current_result(
        self, *, first_run_complete: bool,
    ) -> wizard_state.WizardResult:
        """Return the exact persisted-state contract for current choices.

        Interim saves keep the last completion time, so a setup reopened by
        an update and then cancelled is still known as an update at the next
        launch (Welcome's update line); completion passes None so
        wizard_state.save stamps the time.
        """
        return wizard_state.WizardResult(
            first_run_complete=first_run_complete,
            wizard_version=__version__,
            wizard_completed_at=None if first_run_complete else self._last_completed_at,
            audio_device_name=self._audio_page.selected_device_name() or None,
            genre=self._content_page.selected_genre(),
            llama_smoke_ok=(
                self._smoke_page.smoke_ok() if first_run_complete else False
            ),
        )

    def _persist_interim_state(self, _new_page_id: int) -> None:
        """F_N1: write a partial WizardResult to state.json on each
        page change. Never marks first_run_complete=True -- only the
        accept() terminal path does that.
        """
        try:
            wizard_state.save(
                self._current_result(first_run_complete=False)
            )
        except Exception:
            # Interim-save failures must NOT block wizard progress.
            # Log at debug; the real save happens on accept().
            _log.debug("interim wizard state save failed", exc_info=True)

    def reject(self) -> None:  # noqa: N802 — Qt convention
        self._persist_interim_state(self.currentId())
        super().reject()

    def accept(self) -> None:  # noqa: N802 — Qt convention
        device_name = self._audio_page.selected_device_name()
        genre = self._content_page.selected_genre()

        # Persist runtime-relevant choices to user_overrides.yaml BEFORE
        # marking the wizard complete. state.json is wizard bookkeeping --
        # the pipeline reads the merged layered config, so any wizard pick
        # that should affect runtime (audio device, genre) must land in the
        # sparse overrides file. needs_wizard() only reopens setup while
        # first_run_complete is False, so completion must NEVER be stamped
        # ahead of the choices it summarizes: a swallowed override failure
        # would otherwise strand first_run_complete=True with the shipped-
        # default device and no path back to the wizard.
        #
        # Sparse-override discipline: only write values that DIFFER from
        # shipped defaults. Users who accept all defaults leave an empty
        # user_overrides.yaml so shipped-default improvements keep
        # propagating on upgrade.
        from yamabiko.config_writer import write_overrides
        updates = {("genre",): genre}
        if device_name:
            updates[("audio", "device")] = device_name
        try:
            # Genre is unconditional so choosing the shipped default prunes
            # a stale override. An empty device selection carries no
            # information, so that key is omitted and a previously good pick
            # survives.
            write_overrides(updates)
        except Exception as exc:
            # A failed choice-write must NOT be silently swallowed: do not
            # save completion, do not close the wizard. Surface the failure;
            # Retry runs the whole save again, Cancel leaves the wizard open
            # so Start Yamabiko can be pressed again once the cause (disk
            # full / locked file) is cleared.
            if self._persist_failed(exc, paths.user_overrides_path, "user_overrides.yaml"):
                self.accept()
            return

        result = self._current_result(first_run_complete=True)
        try:
            wizard_state.save(result)
        except Exception as exc:
            # Completion couldn't be recorded; same rule -- don't close on a
            # half-finished save. This leg writes state.json, so the dialog
            # must name state.json, not the overrides file the choice-write
            # leg targets.
            if self._persist_failed(exc, paths.state_path, "state.json"):
                self.accept()
            return
        super().accept()

    def _persist_failed(self, exc: Exception, target_path, fallback: str) -> bool:
        """Show "Couldn't save your settings" for a failed first-run persist;
        True when the user chose Retry. Completion is NOT saved and the
        wizard is NOT closed. ``target_path`` is the zero-arg paths helper
        for the file the failing leg actually writes (user_overrides.yaml
        for the choice writes, state.json for the completion save) so the
        dialog names the right file; ``fallback`` names it when even the
        path helper fails."""
        _log.error("wizard could not persist first-run choices; leaving setup open",
                   exc_info=(type(exc), exc, exc.__traceback__))
        try:
            target = str(target_path())
        except Exception:
            target = fallback
        return SaveFailedDialog.ask(self, self._palette, target, exc, title=BRAND)

    def release_background_threads(self, wait_ms: int = 2000) -> None:
        """Stop or detach worker threads before the wizard object dies.

        Called by run_wizard() after exec() returns — covering Cancel,
        the title-bar X, Esc, AND accept (the smoke probe may be mid-POST
        after a Skip; the quant page's GPU detection may still be stuck
        on a wedged nvidia-smi). A thread still running after
        ``wait_ms`` is detached into ``thread_keeper``: letting the
        wizard's destruction take a running QThread down is a Qt 6 qFatal
        abort, whose non-42 exit code the watchdog classifies RESTART,
        relaunching the wizard the user just cancelled.

        The Sound step's 5 s test goes first: it has no QThread, but its
        WASAPI callback stream is a native thread too, and nothing else
        stops it before a cancelled setup exits (the window outlives this
        call, so the page's ``destroyed`` hook fires only at garbage
        collection).
        """
        self._audio_page.stop_test()
        self._quant_tier_page.shutdown_thread(wait_ms=wait_ms)
        self._downloads_page.shutdown_thread(wait_ms=wait_ms)
        self._smoke_page.shutdown_thread(wait_ms=wait_ms)

    # --- the window ---------------------------------------------------------

    def center_on_work_area(self) -> None:
        """Center the window on the primary screen's work area, its top-left
        kept inside it: a work area under 600 px tall cuts the bottom off
        instead of the caption (L14). setGeometry, not move: for an expanded
        client area the client rect is the visible window (Decision 31)."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = area.x() + max(0, (area.width() - self.width()) // 2)
        y = area.y() + max(0, (area.height() - self.height()) // 2)
        self.setGeometry(x, y, self.width(), self.height())

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._current < 0:
            self.restart()
        if not event.spontaneous():
            # Before the native show: a move after it would lay the frame
            # out on the window menu's margins (setVisible) and widen it.
            self.center_on_work_area()
        super().showEvent(event)
        set_border_color(self, self._palette["frame"])

    def setVisible(self, visible: bool) -> None:  # noqa: N802 (Qt override)
        super().setVisible(visible)
        if visible:
            # After the native show, never in showEvent (Qt 6.11's two frame
            # margins: ErrorWindow.setVisible). Minimize only: the window has
            # a fixed size.
            add_window_menu(self, min_max=False, minimize=True)
            if not self._took_foreground:
                # The first show only (Decision 134): the pipeline child opens
                # setup while the user is in another window (the installer's
                # last page, a browser), and without the foreground it opens
                # behind them. A restore from the taskbar is a show too and
                # must not take it again. The error window's order: raise,
                # take the foreground, activate.
                self._took_foreground = True
                self.raise_()
                take_foreground(self)
                self.activateWindow()

    def nativeEvent(self, event_type, message):  # noqa: N802 (Qt override)
        result = nonclient_event(event_type, message, self._palette["ground"])
        if result is not None:
            return True, result
        return super().nativeEvent(event_type, message)


def run_wizard() -> None:
    """Launch the wizard synchronously. Blocks until the dialog closes.

    Safe to call before the main QApplication exists — we construct one
    ad-hoc and keep it alive at module scope to prevent Qt's C++ side
    from referencing a deleted Python object during wizard teardown
    (same pattern as ui.error_dialog). Registers the bundled fonts (BIZ
    UDPMincho, BIZ UDPGothic) and reads the saved theme once (L13).
    """
    # Launch-chain marker: the only observable that proves the
    # Start-Menu -> watchdog --run-main -> main.main() -> needs_wizard()
    # gate actually reached run_wizard(). The frozen E2E + dev-mode tests
    # grep the per-process main-pipeline.log for WIZARD_START_MARKER (the
    # tests import the same constant, so the string can't drift).
    _log.info(WIZARD_START_MARKER)
    global _owned_app
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.theme import palette_for
    from yamabiko.ui.error_window import saved_theme
    from yamabiko.ui.qt_bootstrap import ensure_qapplication
    # Applies the PassThrough HiDPI rounding policy before creating the
    # QApplication so the first-run wizard renders correctly on
    # 125%/150% Windows scaling -- previously only the overlay did this.
    app = ensure_qapplication()
    _owned_app = app  # prevent GC (no-op if Qt already owned one)
    register_bundled_fonts()
    wiz = YamabikoWizard(palette_for(saved_theme()))  # after the QApplication
    try:
        wiz.exec()
    finally:
        # Must run while `wiz` is still alive: once the last reference
        # goes, the wizard (and its pages) can be destroyed, and any
        # page-parented QThread still running at that moment is a qFatal
        # abort. finally-guarded because PySide6 propagates some slot
        # exceptions out of exec(), and that path must not skip the
        # release either.
        wiz.release_background_threads()
        # Hidden before the last reference goes: past this call only the
        # ClientAreaFit cycle keeps it, and a shown window freed by a later
        # garbage collection would go mid-paint (phase 6).
        wiz.hide()


def exit_wizard_cancelled() -> None:
    """Exit with the wizard-cancel sentinel (42), hard if necessary.

    main.py calls this instead of a bare ``sys.exit(42)`` after a
    cancelled wizard. If a detached download thread is still draining
    (cancel mid-GGUF — ``hf_hub_download`` cannot be interrupted
    in-process; see ``thread_keeper``), interpreter finalization would
    destroy the running QThread and abort with a non-42 code, which
    ``exit_codes.classify`` maps to RESTART — the watchdog would
    relaunch the wizard the user just cancelled. ``os._exit`` skips
    finalization: the OS reaps the thread and the sentinel reaches the
    watchdog intact. The partial ``*.incomplete`` file is not resumed:
    the next download queue deletes it before it starts (see
    ``downloads._sweep_incomplete_blobs``) and fetches the file again.
    Logs are flushed first because ``os._exit`` skips atexit.
    """
    if thread_keeper.any_running():
        logging.getLogger(__name__).warning(
            "wizard cancelled with a download still draining; hard-exiting "
            "with code %d to preserve the cancel sentinel",
            EXIT_WIZARD_CANCEL,
        )
        logging.shutdown()
        os._exit(EXIT_WIZARD_CANCEL)
    sys.exit(EXIT_WIZARD_CANCEL)
