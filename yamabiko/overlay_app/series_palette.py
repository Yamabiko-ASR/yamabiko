"""The series bar ("Spotlight bar", plan phase 5): search AniList for a
series and pick it.

A 640 px frameless bar at the top center of the subtitle monitor: the seal,
the search field and, while a series is active, a chip with its cover,
short title and Clear. Under it a dropdown shows one view at a time
(``series_palette_parts``): Currently watching (an empty field and a
configured AniList username), up to six results, no matches, offline with
Try again, or a private list, each closed by a key-hint line. A pick emits
``series_chosen(media_id, label, native)`` (``native`` is the Japanese title
for the overlay banner, "" if unknown) and the tray builds the series'
gazetteer; Clear emits ``clearRequested`` and the tray sets no series.

Off-thread contract (mirrors ``overlay_app.tray._DiagnosticsThread``): the
AniList search, the watch list and every poster fetch run on ``QThread``
workers so a slow network round-trip can never freeze the GUI thread. Each
worker emits a typed ``Signal`` whose payload crosses back to the GUI thread
as a queued slot call.

Covers: a row paints its poster once it loaded; until then (and on ANY
fetch failure, or when a hit has no cover URL) the hit's AniList color,
else a neutral tile. The poster is an upgrade, never a blocker.

Session cover cache (the tray flyout's series row, plan phase 4 R3): the
AniList color of every listed hit and every poster decoded this session are
kept by media id; ``cover_for`` hands the best one to the tray and
``coverChanged`` says when a color was first seen or a poster arrived. The
active series' poster (``set_current_series``, from the tray) is also saved
as cover-<media_id>.png beside its gazetteer cache, and ``cover_for`` falls
back to that file, so a series restored at boot shows its cover, offline
too.

"Currently watching" (AniList Component 4): with
``gazetteer.anilist_username`` configured, every summon re-reads the
username from the merged config (a Settings change lands on the next
summon) and refreshes the user's watch list in the background, cached
in-process for 15 minutes per username (injectable clock). With the field
empty the six most recently updated entries show; a failed fetch shows the
offline view or the private-list line, never a traceback.

Window (plan phase 5, Decisions 29-38 and 50-62, probe rulings P1-P5): DWM
draws the small rounded corners, the frame-tone edge and the soft shadow
after show (``window_chrome``); the bar is cloaked from DWM from show until
its first paint; the window height follows the dropdown with a 160 ms
unfold/fold unless reduce motion is on; Esc or losing activation closes it.

Status face (Decisions 39 and 63-67): a pick folds the dropdown and the bar
becomes the pick's status, the seal, its cover and "Loading character names
for <short>…" over a 2 px progress line that the tray feeds with the build's
staged progress (show_build_progress). On show_built the line fills, a check
and "Now watching <short>." hold for two seconds and the bar slides up while
fading (fades in place when a monitor sits right above). show_build_failed,
cancel_status (Clear), Esc or any hide drop the status at once; a summon
returns to the search face; losing activation never hides the status.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import httpx
from PySide6.QtCore import (
    QAbstractAnimation, QEasingCurve, QEvent, QIODevice, QPoint, QRect, QRectF, QSaveFile,
    QSignalBlocker, Qt, QThread, QTimer, QVariantAnimation, Signal,
)
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPixmap, QScreen
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget

from yamabiko.gazetteer.anilist import (
    AniListClient, AniListPrivateUser, SeriesHit,
)
from yamabiko.gazetteer.gazetteer import cover_path
from yamabiko.ui import thread_keeper

from . import theme
from .controls import ui_font
from .seal import Seal
from .series_palette_parts import (
    BAR_H, BAR_INNER_H, BAR_W, EDGE, LOADING, NOW_WATCHING, PLACEHOLDER, SEAL_PX,
    SMALL_COVER_H, TOP_GAP, VIEW_NO_MATCHES, VIEW_OFFLINE, VIEW_PRIVATE, VIEW_RESULTS,
    VIEW_WATCHING, WHO, Dropdown, DropdownModel, SeriesChip, StatusText, display_title,
    dropdown_height, paint_bar_line, short_title,
)
from .tray_flyout import NEUTRAL_COVER
from .window_chrome import (
    DWMWCP_ROUNDSMALL, SHELL_WINDOW_CLASSES, SquareCorners, foreground_is_app_window,
    foreground_widget, foreign_foreground, hand_back_foreground, set_border_color,
    set_cloak, set_corner_preference, take_foreground, topmost_app_window, window_class,
    window_shows,
)

_log = logging.getLogger(__name__)

_DEBOUNCE_MS = 250
# At most six rows under the bar (spec, Surface 4): the search asks AniList
# for six, and Currently watching shows the six most recently updated.
MAX_ROWS = 6
_COVER_TIMEOUT_S = 8.0
# Watch-list cache TTL: a summon inside this window uses the in-process
# cache without a fetch (spec: 15 minutes).
_WATCHING_TTL_S = 15 * 60
# Motion (plan phase 5, Decision 35; probe P1), read at use time so tests can
# change it: the window height follows the dropdown over this long.
_UNFOLD_MS = 160
# DWM cloak from show until the first paint, lifted after this at the latest
# (probe P4): no empty outline flashes on a summon, and a missed paint never
# leaves an invisible window holding the keyboard.
_UNCLOAK_MS = 100
# The status face (Decisions 35, 39, 65, 66), read at use time: the progress
# fill's tween, the hold of "Now watching …." and the slide-away.
_FILL_MS = 150
_DONE_HOLD_MS = 2000
_SLIDE_MS = 200

# The bar row in window coordinates (Decision 30).
_SEAL_X = EDGE + 11                                    # 12
_SEAL_Y = EDGE + (BAR_INNER_H - SEAL_PX) // 2          # 11
# The field starts 2 px left of the mockup's text (x 50): QLineEdit draws its
# text 2 px inside its own left edge (lead ruling on the Task 5 visual check).
_FIELD_X = _SEAL_X + SEAL_PX + 10                      # 48
_BAR_RIGHT = BAR_W - EDGE - 12                         # 627
_CHIP_Y = EDGE + (BAR_INNER_H - SMALL_COVER_H) // 2    # 10
_CHIP_GAP = 12
# The status text right of the seal (Decision 63): x 48..627 inside the bar.
_STATUS_X = _SEAL_X + SEAL_PX + 10                     # 48
_STATUS_RECT = QRect(_STATUS_X, EDGE, _BAR_RIGHT - _STATUS_X, BAR_INNER_H - 2)
# The focus underline and the progress line: 2 px at the bar's bottom, one
# row above the window's outermost row, which the DWM border covers (P5).
_BAR_LINE = QRect(EDGE, BAR_H - 3, BAR_W - 2 * EDGE, 2)
# The field paints only its text, caret and selection (their colors are
# QPalette roles, _apply_field_palette); this style sheet forces the style
# sheet style so the Windows 11 style's own field fill and underline never
# draw (Decision 53).
_FIELD_QSS = "QLineEdit { background: transparent; border: none; padding: 0px; }"
_ROW_VIEWS = (VIEW_WATCHING, VIEW_RESULTS)


def format_hit(hit: SeriesHit) -> str:
    """The pick's label: english title preferred over romaji, with the
    season year appended in parens when known.

    Pure (no Qt) so it can be unit-tested without a QApplication.
    """
    label = display_title(hit)
    if hit.season_year:
        return f"{label} ({hit.season_year})"
    return label


def _read_configured_username() -> str:
    """The merged ``gazetteer.anilist_username``, trimmed; ``""`` on any
    load failure (config trouble must not break the series bar -- the
    watch list is simply absent).

    The default ``username_provider`` seam of :class:`SeriesPalette`:
    reading the merged config at every summon is what lets a Settings
    change take effect without an app restart.
    """
    try:
        from yamabiko import config as _config
        cfg = _config.load_full_config()
    except Exception:
        _log.debug("anilist_username config read failed", exc_info=True)
        return ""
    raw = (cfg.get("gazetteer", {}) or {}).get("anilist_username", "")
    return str(raw or "").strip()


def _fetch_cover_bytes(url: str, *, client: httpx.Client | None = None) -> bytes | None:
    """Fetch poster bytes for ``url``; return None on ANY failure.

    Runs on a worker thread (see ``_CoverThread``). Never raises: a
    network error, timeout, or non-200 must degrade to the color-tile
    fallback, not take down the worker (whose uncaught exception would
    hit ``sys.excepthook``).

    ``client`` is an injection seam for tests (an ``httpx.Client`` backed
    by a ``MockTransport``); production passes nothing and gets a
    short-lived client that is closed before returning.
    """
    own = client is None
    c = client or httpx.Client(timeout=_COVER_TIMEOUT_S, follow_redirects=True)
    try:
        resp = c.get(url)
        if resp.status_code == 200:
            return resp.content
    except Exception as exc:
        # The cover URL embeds the media id of a searched/watched series
        # and this log ships in diagnostics zips with no consent gate, so
        # log the failure class only -- never the URL, and no exc_info
        # (a transport error's message can quote the URL). Same
        # identity-free convention as the search/watching failure lines.
        _log.debug("cover fetch failed (%s)", type(exc).__name__)
    finally:
        if own:
            c.close()
    return None


def save_cover_file(pixmap: QPixmap, path: Path) -> bool:
    """Write ``pixmap`` to ``path`` as a PNG and say whether it landed.

    QSaveFile writes a temporary file and renames it over ``path`` on
    commit, so a crash mid-write never leaves half a PNG behind. Failures
    are logged by kind only: the path names the series and this log ships in
    diagnostics zips.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.warning("saved cover: folder unavailable (%s)", type(exc).__name__)
        return False
    out = QSaveFile(str(path))
    if not out.open(QIODevice.OpenModeFlag.WriteOnly):
        _log.warning("saved cover: could not open the file for writing")
        return False
    if not pixmap.save(out, "PNG"):
        out.cancelWriting()
        _log.warning("saved cover: PNG encoding failed")
        return False
    if not out.commit():
        _log.warning("saved cover: could not commit the file")
        return False
    return True


def load_cover_file(path: Path) -> QPixmap | None:
    """The poster saved at ``path``, or None when there is none or it does
    not decode (the caller falls back to the cover color)."""
    if not path.is_file():
        return None
    pixmap = QPixmap()
    if not pixmap.load(str(path), "PNG"):
        _log.warning("saved cover unreadable; showing the cover color")
        return None
    return pixmap


def _take_foreground(window: QWidget) -> bool:
    """summon()'s foreground step, between raise_() and activateWindow():
    Ctrl+Alt+S reaches this process from the pipeline's keyboard hook, so the
    bar takes the foreground itself (D-FG; window_chrome.take_foreground).
    Every summon route runs it; for the tray's routes this process already
    had the last input and the call changes nothing."""
    return take_foreground(window)


def _types_in(window: QWidget) -> bool:
    """Whether ``window`` (this process's, in front at a summon) is one the
    user types in, so the bar gives it the foreground back: a shown plain
    window or dialog (Settings, a compact dialog). The tray flyout (a tool
    window), menus, tooltips and the click-through subtitle window are not."""
    return (window.isVisible()
            and window.windowType() in (Qt.WindowType.Window, Qt.WindowType.Dialog)
            and not window.windowFlags() & Qt.WindowType.WindowTransparentForInput)


def _finish(anim: QVariantAnimation) -> None:
    """Jump a running animation to its end (reduce motion turned on): its
    last value and its ``finished`` arrive at once."""
    if anim.state() == QAbstractAnimation.State.Running:
        anim.setCurrentTime(anim.duration())


def _native_rect(screen: QScreen) -> QRect:
    """``screen``'s rect in native pixels. Qt on Windows keeps each screen's
    top-left at its native position and divides only its size by the scale
    ("islands of screens"), so the size is scaled back; the rounding of the
    logical size can leave it 1 px long."""
    g = screen.geometry()
    return QRect(g.topLeft(), g.size() * screen.devicePixelRatio())


def _screen_above(native: QRect) -> bool:
    """Whether a monitor sits right above the screen whose native rect is
    ``native``: its native bottom meets that top edge (within 1 px) and it
    shares more than one pixel column with it (1 px is rounding). The
    slide-away then fades in place instead of crossing onto it (probe P2,
    lead ruling on the Task 6 review). Native pixels, because a logical
    point just outside a scaled screen lies on no screen. The screen itself
    never meets its own top, so every screen can be looked at."""
    for screen in QGuiApplication.screens():
        other = _native_rect(screen)
        meets = abs(other.bottom() + 1 - native.top()) <= 1
        shared = min(other.right(), native.right()) - max(other.left(), native.left())
        if meets and shared >= 1:
            return True
    return False


class _SearchThread(QThread):
    """Runs ``AniListClient.search_series`` off the GUI thread.

    Mirrors ``tray._DiagnosticsThread``: QThread subclass + typed Signal.
    Emits ``resultsReady(ok, hits)``. Any error (network, rate limit,
    parse) is ``(False, [])`` rather than a crash in the worker, and the
    bar shows its offline view for it; a search that found nothing is
    ``(True, [])`` ("No matches").
    """

    resultsReady = Signal(bool, list)

    def __init__(self, client: AniListClient, query: str, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._query = query

    def run(self) -> None:  # worker thread
        try:
            hits = self._client.search_series(self._query, limit=MAX_ROWS)
        except Exception:
            # Identity-free log line: the raw query is user-typed content
            # (often the title being watched) and this log ships in
            # diagnostics zips with no consent gate, so log its length,
            # never its text. (The exception chain here never embeds the
            # query -- it travels in the POST body, not the URL.)
            _log.warning(
                "AniList search failed (query length %d)",
                len(self._query), exc_info=True,
            )
            self.resultsReady.emit(False, [])
            return
        self.resultsReady.emit(True, list(hits))


class _CoverThread(QThread):
    """Fetches one poster's bytes off the GUI thread.

    Emits ``(media_id, bytes|None)``; the GUI slot decodes the bytes into
    the session cache or keeps the color tile.
    """

    coverReady = Signal(int, object)

    def __init__(self, media_id: int, url: str, parent=None) -> None:
        super().__init__(parent)
        self._media_id = media_id
        self._url = url

    def run(self) -> None:  # worker thread
        self.coverReady.emit(self._media_id, _fetch_cover_bytes(self._url))


class _WatchingThread(QThread):
    """Runs ``AniListClient.fetch_watching`` off the GUI thread.

    Mirrors ``_SearchThread``. Emits ``watchingReady(status, hits)``
    where status is ``"ok"`` / ``"private"`` / ``"unavailable"``: both
    AniList failure taxa AND any unexpected error degrade to the bar's
    private or offline view rather than crashing the worker (an uncaught
    exception in run() would hit ``sys.excepthook``). The full detail goes
    to the log only -- the user never sees a traceback.
    """

    watchingReady = Signal(str, list)

    def __init__(self, client: AniListClient, username: str, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._username = username

    def run(self) -> None:  # worker thread
        try:
            hits = self._client.fetch_watching(self._username)
        except AniListPrivateUser:
            self.watchingReady.emit("private", [])
            return
        except Exception as exc:
            # AniListUnavailable (403 / repeated 429 / transport) and
            # anything unexpected: the offline view.
            # Identity-free log line: the AniList username is an account
            # identifier and this log ships in diagnostics zips with no
            # consent gate, so log presence (length) and the exception
            # CLASS only -- never the repr or a traceback, either of
            # which can embed the username (an arbitrary error's message
            # can quote it). AniListPrivateUser carries a fixed message
            # with the username on .username, and is caught above, so it
            # never reaches this handler.
            _log.warning(
                "AniList watching fetch failed (username length %d): %s",
                len(self._username), type(exc).__name__,
            )
            self.watchingReady.emit("unavailable", [])
            return
        self.watchingReady.emit("ok", list(hits))


class SeriesPalette(QWidget):
    """The series bar: type -> debounced AniList search -> pick a row ->
    ``series_chosen(media_id, label, native)``; the chip's Clear ->
    ``clearRequested``."""

    series_chosen = Signal(int, str, str)  # media_id, label, Japanese title ("" if unknown)
    coverChanged = Signal(int)  # media_id: a color first seen or a poster arrived
    clearRequested = Signal()  # the chip's Clear: the tray sets no series (L8)

    def __init__(self, client: AniListClient | None = None, parent=None,
                 theme_palette: dict | None = None, *,
                 username_provider: Callable[[], str] | None = None,
                 now: Callable[[], float] = time.monotonic,
                 screen_provider: Callable[[], QScreen | None] | None = None,
                 reduce_motion: bool = False) -> None:
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # A real httpx-backed client by default; the workers use it off the
        # GUI thread, so constructing it here costs nothing until used.
        self._client = client or AniListClient()
        self._search_threads: set[_SearchThread] = set()
        self._cover_threads: set[_CoverThread] = set()
        # Session cover cache (cover_for): media_id -> the decoded poster,
        # and media_id -> the AniList cover color of every hit listed so far.
        self._posters: dict[int, QPixmap] = {}
        self._colors: dict[int, QColor] = {}
        # The active series (set_current_series, from the tray) and its saved
        # poster (cover_path): ids cover_for already looked up on disk this
        # session, ids whose session poster came from that file, and ids
        # whose network poster this session is already written.
        self._current_media_id: int | None = None
        self._current_label = ""
        self._cover_file_tried: set[int] = set()
        self._file_posters: set[int] = set()
        self._saved_covers: set[int] = set()
        # "Currently watching". username_provider is the injection seam for
        # tests (production reads the merged config at every summon, so a
        # Settings change lands on the next summon); ``now`` is the
        # injectable clock for the 15-minute cache TTL.
        self._username_provider = (
            username_provider if username_provider is not None
            else _read_configured_username
        )
        self._now = now
        self._watching_threads: set[_WatchingThread] = set()
        # username -> (now() at fetch success, hits). Errors are never
        # cached, so a transient outage retries on the next summon.
        self._watching_cache: dict[str, tuple[float, list[SeriesHit]]] = {}
        # The username the watch list belongs to ("" = none), its rows (the
        # last successful fetch for that user, kept while a refetch runs)
        # and the username with a fetch in flight (dedupes double summons).
        self._watching_user = ""
        self._watching_hits: list[SeriesHit] = []
        self._watching_inflight: str | None = None
        # This open's answers (Decision 36): a failed watch-list fetch
        # ("private" or "unavailable") and the latest settled search answer
        # (query, ok, hits) for the text the field held then.
        self._watch_failed: str | None = None
        self._answer: tuple[str, bool, list[SeriesHit]] | None = None
        # Chrome theme palette. build_overlay resolves overlay.theme once at
        # boot and hands it here; None falls back to the dark NIGHT palette.
        self._theme_palette = theme_palette if theme_palette is not None else theme.NIGHT
        # The subtitle monitor (OverlayWindow.subtitle_screen in production;
        # no provider or no screen: the primary screen) and Settings' reduce
        # motion (the boot value; set_reduce_motion follows live changes).
        self._screen_provider = screen_provider
        self._reduce_motion = bool(reduce_motion)
        self._cloaked = False
        # The window in front when the bar was summoned, for the hide to give
        # the foreground back to: another app's, or Settings or a dialog of
        # this process (0: none kept, as from the tray or the flyout).
        self._return_to = 0
        # The status face (Decision 39): "search" or "status", the pick it
        # reports on and its short title, the progress line's target and
        # painted fill, the summoning screen's native rect, and the slide-away's
        # start, distance and fade-only choice (Decision 65).
        self._mode = "search"
        self._status_media_id: int | None = None
        self._status_title = ""
        self._fraction_target = 0.0
        self._fraction_shown = 0.0
        self._screen_native: QRect | None = None
        self._slide_from = QPoint()
        self._slide_dy = 0
        self._slide_fade_only = False

        self.setAccessibleName("Series search")
        self.setFont(ui_font(13))
        self.setFixedSize(BAR_W, BAR_H)

        self._seal = Seal(SEAL_PX, self)
        self._seal.move(_SEAL_X, _SEAL_Y)

        self._search = QLineEdit(self)
        self._search.setAccessibleName(PLACEHOLDER)
        self._search.setPlaceholderText(PLACEHOLDER)
        self._search.setStyleSheet(_FIELD_QSS)
        self._search.setFont(ui_font(15))
        self._search.setFixedHeight(BAR_INNER_H)
        self._search.move(_FIELD_X, EDGE)
        # Nav keys (Up/Down/Enter/Esc) and the field's focus changes (the
        # underline) are caught here, so typing keeps the focus in the field.
        self._search.installEventFilter(self)

        self._chip = SeriesChip(self._theme_palette, self)
        self._chip.hide()
        self._chip.clearClicked.connect(self._on_clear_clicked)

        self._dropdown = Dropdown(self._theme_palette, self.cover_for, self)
        self._dropdown.move(EDGE, BAR_H - EDGE)
        self._dropdown.rowClicked.connect(self._pick)
        self._dropdown.retryClicked.connect(self._retry)

        self._status = StatusText(self._theme_palette, self)
        self._status.setGeometry(_STATUS_RECT)
        self._status.hide()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._run_search)
        self._search.textChanged.connect(self._on_text_changed)

        self._height_anim = QVariantAnimation(self)
        self._height_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._height_anim.valueChanged.connect(
            lambda value: self.setFixedHeight(int(round(value))))

        self._uncloak_timer = QTimer(self)
        self._uncloak_timer.setSingleShot(True)
        self._uncloak_timer.timeout.connect(self._uncloak)

        self._fill_anim = QVariantAnimation(self)
        self._fill_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fill_anim.valueChanged.connect(self._set_fraction_shown)
        # 0 -> 1 is the slide's progress. The values are set before the
        # connections: setting them emits valueChanged.
        self._slide_anim = QVariantAnimation(self)
        self._slide_anim.setStartValue(0.0)
        self._slide_anim.setEndValue(1.0)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._slide_anim.valueChanged.connect(self._on_slide_step)
        self._slide_anim.finished.connect(self._on_slide_done)
        self._done_timer = QTimer(self)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self._slide_away)

        self.coverChanged.connect(self._on_cover_changed)
        self._apply_field_palette()
        self._layout_bar()
        # The native window exists before the first summon (the flyout's
        # technique, phase-4 Decision 54); prewarm() also paints it once.
        self.winId()

    # -- theme ------------------------------------------------------------

    def apply_palette(self, palette: dict) -> None:
        """Re-skin the bar for a new theme (Tray.apply_palette: an OS
        color-scheme follow or a manual theme pick), in place."""
        self._theme_palette = palette
        self._apply_field_palette()
        self._chip.apply_palette(palette)
        self._dropdown.apply_palette(palette)
        self._status.apply_palette(palette)
        if self.isVisible():
            set_border_color(self, palette["frame"])
        self.update()

    def _apply_field_palette(self) -> None:
        """The field's ink text and caret, muted placeholder and accent
        selection (Decision 53)."""
        tokens = self._theme_palette
        qpalette = self._search.palette()
        qpalette.setColor(QPalette.ColorRole.Text, QColor(tokens["ink"]))
        qpalette.setColor(QPalette.ColorRole.PlaceholderText, QColor(tokens["muted"]))
        qpalette.setColor(QPalette.ColorRole.Highlight, QColor(tokens["accent"]))
        qpalette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens["on_accent"]))
        self._search.setPalette(qpalette)

    # -- motion -----------------------------------------------------------

    def set_reduce_motion(self, on: bool) -> None:
        """Settings' reduce motion, live (L12): no unfold/fold, fill tween or
        slide from now on, and a running one jumps to its end (a slide ends
        hidden)."""
        self._reduce_motion = bool(on)
        if self._reduce_motion:
            for anim in (self._height_anim, self._fill_anim, self._slide_anim):
                _finish(anim)

    def _set_height(self, height: int, *, animate: bool) -> None:
        """Make the window ``height`` tall: an unfold/fold over _UNFOLD_MS
        (probe P1: setFixedHeight per frame), or at once when not asked,
        with reduce motion, while hidden, or when it already is."""
        anim = self._height_anim
        anim.stop()
        if (not animate or self._reduce_motion or not self.isVisible()
                or self.height() == height):
            self.setFixedHeight(height)
            return
        anim.setDuration(_UNFOLD_MS)
        # Setting the values emits valueChanged at the animation's current
        # time (the old end after a finished run), which would resize the
        # window to the new end and back before the unfold starts.
        with QSignalBlocker(anim):
            anim.setStartValue(self.height())
            anim.setEndValue(height)
        anim.start()

    # -- search -----------------------------------------------------------

    def _on_text_changed(self, text: str) -> None:
        """Typing re-arms the debounced search; an emptied field shows
        Currently watching (or the bar alone) at once (L4)."""
        if text.strip():
            self._debounce.start()
            return
        self._debounce.stop()
        self._refresh_view()

    def _run_search(self) -> None:
        """Debounce fired (or a summon / a retry): start an off-thread
        AniList search for the field's text. An empty field needs none."""
        query = self._search.text().strip()
        if not query:
            self._refresh_view()
            return
        thread = _SearchThread(self._client, query, parent=self)
        # Capture the query so a slow round-trip that resolves after the
        # user has typed more can be dropped as stale.
        thread.resultsReady.connect(
            lambda ok, hits, q=query: self._on_results(q, ok, hits)
        )
        thread.finished.connect(lambda t=thread: self._search_threads.discard(t))
        thread.finished.connect(thread.deleteLater)
        self._search_threads.add(thread)
        thread.start()

    def _on_results(self, query: str, ok: bool, hits: list) -> None:
        """A search answer settled. It decides the view only for the text
        still in the field (Decision 36); a stale one is dropped."""
        if query != self._search.text().strip():
            return  # stale: the field moved on before this returned
        self._answer = (query, ok, list(hits))
        self._refresh_view()

    # -- covers -----------------------------------------------------------

    def _wants_cover(self, hit: SeriesHit) -> bool:
        """A shown row needs its poster fetched (Decision 55): it has a URL,
        this session has no network poster for it (one read from the saved
        file is replaced, Decision 27), and no fetch for it is running."""
        if not hit.cover_image:
            return False
        if hit.media_id in self._posters and hit.media_id not in self._file_posters:
            return False
        return not any(t._media_id == hit.media_id for t in self._cover_threads)

    def _start_cover_fetch(self, hit: SeriesHit) -> None:
        thread = _CoverThread(hit.media_id, hit.cover_image, parent=self)
        thread.coverReady.connect(self._on_cover_ready)
        thread.finished.connect(lambda t=thread: self._cover_threads.discard(t))
        thread.finished.connect(thread.deleteLater)
        self._cover_threads.add(thread)
        thread.start()

    def _on_cover_ready(self, media_id: int, data: object) -> None:
        if not data:
            return  # fetch failed -> keep the color tile
        pm = QPixmap()
        if not pm.loadFromData(data):
            return  # undecodable bytes -> keep the color tile
        self._posters[media_id] = pm
        self._file_posters.discard(media_id)
        self.coverChanged.emit(media_id)
        if media_id == self._current_media_id:
            self._save_current_cover()

    def _on_cover_changed(self, media_id: int) -> None:
        """A color or a poster arrived: repaint the rows, and the chip or the
        status when it shows that series."""
        if media_id == self._current_media_id:
            self._chip.set_cover(self.cover_for(media_id))
        if media_id == self._status_media_id:
            self._status.set_cover(self.cover_for(media_id))
        self._dropdown.update()

    def _note_hit(self, hit: SeriesHit) -> None:
        """Remember a listed hit's cover color for cover_for."""
        color = QColor(hit.color) if hit.color else QColor()
        if not color.isValid() or self._colors.get(hit.media_id) == color:
            return
        self._colors[hit.media_id] = color
        if hit.media_id not in self._posters:
            self.coverChanged.emit(hit.media_id)

    def cover_for(self, media_id: int) -> "QPixmap | QColor":
        """The best cover this session knows for ``media_id``: the decoded
        poster (full size; the caller scales it), else the poster saved for a
        picked series (cover_path; looked up once per id per session, a hit
        joins the session cache), else the AniList color of a listed hit,
        else the neutral tile color."""
        poster = self._posters.get(media_id)
        if poster is None and media_id not in self._cover_file_tried:
            self._cover_file_tried.add(media_id)
            poster = load_cover_file(cover_path(media_id))
            if poster is not None:
                self._posters[media_id] = poster
                self._file_posters.add(media_id)
        if poster is not None:
            return poster
        return QColor(self._colors.get(media_id, QColor(NEUTRAL_COVER)))

    # -- the active series ---------------------------------------------------

    def set_current_series(self, media_id: int | None, label: str = "") -> None:
        """The series the tray shows as active (Tray._set_series: a pick's
        build succeeded, a state frame, Clear) and its label. The chip shows
        it (L8); its poster is saved for the next boot when this session
        decoded one, now or when it arrives."""
        self._current_media_id = media_id
        self._current_label = label if media_id is not None else ""
        self._save_current_cover()
        self._sync_chip()

    def _save_current_cover(self) -> None:
        """Write the active series' network poster to cover_path once per
        session; a poster read back from that file is never written again."""
        media_id = self._current_media_id
        if (media_id is None or media_id in self._saved_covers
                or media_id in self._file_posters):
            return
        poster = self._posters.get(media_id)
        if poster is not None and save_cover_file(poster, cover_path(media_id)):
            self._saved_covers.add(media_id)

    def _sync_chip(self) -> None:
        """The chip shows the active series in the search face (the status
        face replaces it); the field takes the room it leaves."""
        media_id = self._current_media_id
        if media_id is None or self._mode != "search":
            self._chip.hide()
        else:
            self._chip.set_series(self._current_label, self.cover_for(media_id))
            self._chip.show()
        self._layout_bar()

    def _layout_bar(self) -> None:
        """Right-align the chip at x 627 and give the field the rest
        (Decisions 30, 57)."""
        if self._chip.isVisibleTo(self):
            self._chip.move(_BAR_RIGHT - self._chip.width(), _CHIP_Y)
            right = self._chip.x() - _CHIP_GAP
        else:
            right = _BAR_RIGHT
        self._search.setFixedWidth(right - _FIELD_X)

    def _on_clear_clicked(self) -> None:
        """The chip's Clear (Decision 42): the tray sets no series and then
        hides the chip through set_current_series(None). The bar stays open
        in its view; the keyboard goes back to the field."""
        self._search.setFocus()
        self.clearRequested.emit()

    # -- "Currently watching" ----------------------------------------------

    def _refresh_watching(self) -> None:
        """Re-read the configured username and bring its watch list up to
        date. Runs on every summon and every retry.

        * no username -> forget any list, fetch nothing;
        * username changed -> drop the old user's rows now (never show
          another account's list while fetching);
        * cache fresh (within :data:`_WATCHING_TTL_S` of the last
          successful fetch) -> its rows, NO fetch;
        * otherwise -> ONE background fetch (deduped against an in-flight
          fetch for the same username). Until it resolves the last rows
          stay -- stale-but-yours beats a flicker.
        """
        username = (self._username_provider() or "").strip()
        if not username:
            self._watching_user = ""
            self._watching_hits = []
            return
        if username != self._watching_user:
            self._watching_user = username
            self._watching_hits = []
        cached = self._watching_cache.get(username)
        if cached is not None and (self._now() - cached[0]) < _WATCHING_TTL_S:
            self._watching_hits = list(cached[1])
            return
        if self._watching_inflight == username:
            return  # a fetch for this username is already running
        self._watching_inflight = username
        thread = _WatchingThread(self._client, username, parent=self)
        thread.watchingReady.connect(
            lambda status, hits, u=username: self._on_watching_ready(
                u, status, hits
            )
        )
        thread.finished.connect(lambda t=thread: self._watching_threads.discard(t))
        thread.finished.connect(thread.deleteLater)
        self._watching_threads.add(thread)
        thread.start()

    def _on_watching_ready(self, username: str, status: str, hits: list) -> None:
        if self._watching_inflight == username:
            self._watching_inflight = None
        if username != self._watching_user:
            return  # stale: the configured username changed mid-fetch
        if status == "ok":
            self._watching_cache[username] = (self._now(), list(hits))
            self._watching_hits = list(hits)
            self._watch_failed = None
        else:
            self._watch_failed = status  # "private" or "unavailable"
        self._refresh_view()

    # -- the dropdown's view (Decision 36) -----------------------------------

    def _derive_model(self) -> DropdownModel | None:
        """The view this open's answers call for; None keeps the shown one
        (a typed query whose answer has not settled: no loading row)."""
        query = self._search.text().strip()
        username = self._watching_user
        watch_down = self._watch_failed == "unavailable"
        if query:
            answer = self._answer
            if answer is None or answer[0] != query:
                return None
            _query, ok, hits = answer
            if not ok:
                return DropdownModel(view=VIEW_OFFLINE,
                                     list_unavailable=bool(username) and watch_down)
            if hits:
                return DropdownModel(view=VIEW_RESULTS, rows=tuple(hits[:MAX_ROWS]), lit=0)
            return DropdownModel(view=VIEW_NO_MATCHES, query=query)
        if not username:
            return DropdownModel()
        if watch_down:
            return DropdownModel(view=VIEW_OFFLINE, list_unavailable=True)
        who = WHO.format(username=username)
        if self._watch_failed == "private":
            return DropdownModel(view=VIEW_PRIVATE, who=who)
        if self._watching_hits:
            return DropdownModel(view=VIEW_WATCHING,
                                 rows=tuple(self._watching_hits[:MAX_ROWS]), lit=0, who=who)
        return DropdownModel()  # the first fetch still running, or an empty list

    def _refresh_view(self, *, animate: bool = True) -> None:
        """Show the derived view when its content differs from the shown
        one (the same content keeps the lit row, Decision 54)."""
        model = self._derive_model()
        if model is None:
            return
        shown = self._dropdown.model()
        if replace(model, lit=shown.lit) == shown:
            return
        self._show_model(model, animate=animate)

    def _show_model(self, model: DropdownModel, *, animate: bool) -> None:
        for hit in model.rows:
            self._note_hit(hit)
            if self._wants_cover(hit):
                self._start_cover_fetch(hit)
        self._dropdown.set_model(model)
        if self._mode == "search":  # the status face keeps the bar folded
            self._set_height(BAR_H + dropdown_height(model), animate=animate)

    # -- keys, picks, retries ------------------------------------------------

    def _move_lit(self, delta: int) -> None:
        """Up/Down: the lit row moves by one and stops at the ends."""
        model = self._dropdown.model()
        if model.view not in _ROW_VIEWS or not model.rows:
            return
        lit = max(0, min(len(model.rows) - 1, model.lit + delta))
        if lit != model.lit:
            self._dropdown.set_model(replace(model, lit=lit))

    def _on_enter(self) -> None:
        """Enter picks the lit row, or retries in the offline view; nothing
        in the status face."""
        if self._mode != "search":
            return
        model = self._dropdown.model()
        if model.view in _ROW_VIEWS:
            self._pick(model.lit)
        elif model.view == VIEW_OFFLINE:
            self._retry()

    def _pick(self, index: int) -> None:
        """Pick row ``index`` of the shown rows (Enter or a click): the tray
        builds it, the bar shows its status (L9, Decision 67). Nothing while
        the status face folds in."""
        if self._mode != "search":
            return
        model = self._dropdown.model()
        if model.view not in _ROW_VIEWS or not 0 <= index < len(model.rows):
            return
        hit = model.rows[index]
        self.series_chosen.emit(hit.media_id, format_hit(hit), hit.native or "")
        self._enter_status(hit)

    def _retry(self) -> None:
        """Try again (or Enter) in the offline view (L4, Decision 56): the
        search at once when the field holds text, and the watch list. The
        offline view stays until an answer replaces it."""
        if self._search.text().strip():
            self._debounce.stop()
            self._run_search()
        self._refresh_watching()

    # -- the status face (L9, L10, Decisions 39, 63-67) -----------------------

    def show_build_progress(self, media_id: int, fraction: float) -> None:
        """The tray's staged build progress (0..1) for ``media_id``: the line
        moves toward it, never back. Ignored for another pick, after built,
        or without a status."""
        if not self._reports_to(media_id) or self._status.is_done():
            return
        target = max(self._fraction_target, min(float(fraction), 1.0))
        if target == self._fraction_target:
            return
        self._fraction_target = target
        self._tween_fill()

    def show_built(self, media_id: int) -> None:
        """The pick's names are in: the line fills, the check and "Now
        watching <short>." hold for _DONE_HOLD_MS, then the bar leaves."""
        if not self._reports_to(media_id) or self._status.is_done():
            return
        self._fraction_target = 1.0
        self._tween_fill()
        self._status.set_status(self.cover_for(media_id),
                                NOW_WATCHING.format(title=self._status_title), done=True)
        self._done_timer.start(_DONE_HOLD_MS)

    def show_build_failed(self, media_id: int) -> None:
        """The pick's build failed: the bar hides at once (the tray's
        "Series not changed" toast is the notice)."""
        if self._reports_to(media_id):
            self.hide()

    def cancel_status(self) -> None:
        """Clear while a status shows: it hides at once. The search face
        stays open (the chip's Clear)."""
        if self._mode == "status":
            self.hide()

    def _reports_to(self, media_id: int) -> bool:
        return self._mode == "status" and media_id == self._status_media_id

    def _enter_status(self, hit: SeriesHit) -> None:
        """The picked face (L9, Decision 63): the dropdown folds up, the
        field and the chip give way to the pick's cover and "Loading
        character names for <short>…" over an empty progress line; the bar
        keeps the keyboard (Esc)."""
        self._mode = "status"
        self._status_media_id = hit.media_id
        self._status_title = short_title(display_title(hit))
        self._done_timer.stop()
        self._fill_anim.stop()
        self._fraction_target = 0.0
        self._set_fraction_shown(0.0)
        self._status.set_status(self.cover_for(hit.media_id),
                                LOADING.format(title=self._status_title), done=False)
        self._search.hide()
        self._sync_chip()
        self._status.show()
        self.setFocus()
        self._set_height(BAR_H, animate=True)

    def _leave_status(self) -> None:
        """Back to the search face; the status is dropped, so later reports
        for that pick are ignored (Decision 64)."""
        if self._mode != "status":
            return
        self._mode = "search"
        self._status_media_id = None
        self._done_timer.stop()
        self._fill_anim.stop()
        self._slide_anim.stop()
        if self.windowOpacity() < 1.0:
            self.setWindowOpacity(1.0)
        self._fraction_target = 0.0
        self._fraction_shown = 0.0
        self._status.hide()
        self._search.show()
        self._sync_chip()
        self.update()

    def _set_fraction_shown(self, value) -> None:
        self._fraction_shown = float(value)
        self.update(_BAR_LINE)

    def _tween_fill(self) -> None:
        """Move the painted fill to the target over _FILL_MS (Decision 66);
        at once with reduce motion or while hidden."""
        anim = self._fill_anim
        anim.stop()
        if self._reduce_motion or not self.isVisible():
            self._set_fraction_shown(self._fraction_target)
            return
        anim.setDuration(_FILL_MS)
        # Setting the values emits valueChanged at the animation's current
        # time (as in _set_height): an interrupted tween would step the fill
        # to stale in-between values before this one starts.
        with QSignalBlocker(anim):
            anim.setStartValue(self._fraction_shown)
            anim.setEndValue(self._fraction_target)
        anim.start()

    def _slide_away(self) -> None:
        """After the hold (L9, L12, Decision 65): with reduce motion the bar
        hides; otherwise it slides up by its height + TOP_GAP while fading,
        or only fades when a monitor sits right above (probe P2)."""
        if self._reduce_motion:
            self.hide()
            return
        native = self._screen_native
        self._slide_fade_only = native is not None and _screen_above(native)
        self._slide_from = self.pos()
        self._slide_dy = self.height() + TOP_GAP
        anim = self._slide_anim
        anim.stop()
        anim.setDuration(_SLIDE_MS)
        anim.start()

    def _on_slide_step(self, value) -> None:
        progress = float(value)
        if not self._slide_fade_only:
            self.move(self._slide_from.x(),
                      self._slide_from.y() - round(progress * self._slide_dy))
        self.setWindowOpacity(1.0 - progress)

    def _on_slide_done(self) -> None:
        # Back in place while still fully transparent, then hidden; the hide
        # drops the status and restores the opacity (Decision 65).
        self.move(self._slide_from)
        self.hide()

    # -- summon -----------------------------------------------------------

    def summon(self) -> None:
        """Open the bar (Ctrl+Alt+S, the tray's Pick a series… and the
        flyout's Change…; Decision 38): the window in front (another app's,
        or Settings or a dialog) is kept for the hide to give back, a status
        showing gives way to the search face (its later reports are ignored),
        this open's failures are forgotten, the watch list refreshed, the
        view derived at once (and the field's search re-run without the
        debounce), the bar placed top center on the subtitle monitor and
        shown, brought to the foreground with DWM's corners and edge (D-FG
        order), and the field focused with its text selected."""
        holder = foreground_widget()
        if holder is not None and holder is not self and _types_in(holder):
            # Settings or a dialog the user was in gets it back (lead ruling
            # on final review round 5, I1).
            self._return_to = int(holder.winId())
        else:
            kept = foreign_foreground()
            # Not the taskbar or the tray overflow (the short menu's route):
            # handing the foreground back to it reopens the overflow.
            if kept and window_class(kept) not in SHELL_WINDOW_CLASSES:
                self._return_to = kept
        self._leave_status()
        self._watch_failed = None
        if self._answer is not None and not self._answer[1]:
            self._answer = None
        self._refresh_watching()
        self._dropdown.set_model(DropdownModel())
        self._set_height(BAR_H, animate=False)
        self._refresh_view(animate=False)
        if self._search.text().strip():
            self._debounce.stop()
            self._run_search()
        self._place()
        if not self.isVisible():
            self._cloak()
        self.show()
        self.raise_()
        _take_foreground(self)
        self.activateWindow()
        set_corner_preference(self, DWMWCP_ROUNDSMALL)
        set_border_color(self, self._theme_palette["frame"])
        self._search.setFocus()
        self._search.selectAll()

    def prewarm(self) -> None:
        """Pay the bar's cold first paint (470-530 ms: fonts, probe P3) at
        boot instead of at the first summon: the native window plus one
        off-screen render."""
        self.winId()
        self.grab()

    def _place(self) -> None:
        """Top center of the subtitle monitor's work area, TOP_GAP below its
        top (L3); the primary screen without a provider or a screen. The
        screen's native rect is kept for the slide-away (Decision 65)."""
        screen = self._screen_provider() if self._screen_provider is not None else None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self._screen_native = _native_rect(screen)
        self.move(area.x() + (area.width() - BAR_W) // 2, area.top() + TOP_GAP)

    def _cloak(self) -> None:
        """Hide the bar from DWM until its first paint (probe P4): the first
        paintEvent lifts it, the safety timer at the latest."""
        self._cloaked = True
        set_cloak(self, True)
        self._uncloak_timer.start(_UNCLOAK_MS)

    def _uncloak(self) -> None:
        self._uncloak_timer.stop()
        if self._cloaked:
            self._cloaked = False
            set_cloak(self, False)

    # -- teardown ---------------------------------------------------------

    def release_threads(self) -> None:
        """Detach still-running search/cover/watching workers into
        thread_keeper.

        Connected to QApplication.aboutToQuit: widget teardown would
        otherwise cascade-destroy a QThread whose OS thread is still inside
        an httpx call (up to 15 s search / 8 s cover) -- Qt 6 aborts the
        process on that. Finished workers are left to their queued
        deleteLater; a detached worker must NOT deleteLater on finish (the
        registry would then hold a dead wrapper -- see thread_keeper's
        no-prune contract), so that connection is dropped here.
        """
        for t in (
            list(self._search_threads)
            + list(self._cover_threads)
            + list(self._watching_threads)
        ):
            if not t.isRunning():
                continue
            try:
                t.finished.disconnect(t.deleteLater)
            except (RuntimeError, TypeError):
                pass  # already delivered/disconnected
            thread_keeper.release(t)
            self._search_threads.discard(t)
            self._cover_threads.discard(t)
            self._watching_threads.discard(t)

    # -- Qt overrides -------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        if obj is self._search:
            kind = event.type()
            if kind in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
                self.update(_BAR_LINE)  # the focus underline (L14)
            elif kind == QEvent.Type.KeyPress:
                key = event.key()
                if key in (Qt.Key_Down, Qt.Key_Up):
                    self._move_lit(1 if key == Qt.Key_Down else -1)
                    return True
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    self._on_enter()
                    return True
                if key == Qt.Key_Escape:
                    self.hide()
                    return True
            elif kind == QEvent.Type.ContextMenu:
                self._show_field_menu(event.globalPos())
                return True
        elif event.type() == QEvent.Type.Hide:
            # One of the bar's popups closed after a deactivation spared the
            # bar (event below): settle once it is gone.
            obj.removeEventFilter(self)
            QTimer.singleShot(0, self._hide_if_inactive)
        return super().eventFilter(obj, event)

    def _show_field_menu(self, pos: QPoint) -> None:
        """The field's right-click menu: QLineEdit's own (Undo, Cut, Copy,
        Paste, …) in the bar's theme, as the tray's menus (menu_qss, square
        corners, no icons: Windows' white edit icons vanish on the Mist menu
        and open an icon column the tray menu does not have), instead of the
        platform style's."""
        menu = self._search.createStandardContextMenu()
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        for action in menu.actions():
            action.setIcon(QIcon())
        menu.setStyleSheet(theme.menu_qss(self._theme_palette))
        SquareCorners(menu)
        menu.popup(pos)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # When the bar itself (not the field) holds focus: Esc closes,
        # Enter acts as in the field.
        key = event.key()
        if key == Qt.Key_Escape:
            self.hide()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._on_enter()
            return
        super().keyPressEvent(event)

    def event(self, event) -> bool:  # noqa: D401 (Qt override)
        handled = super().event(event)
        if event.type() == QEvent.Type.WindowDeactivate and self._mode == "search":
            # A click elsewhere closes the search face, like the flyout
            # (L13); the status stays until it is done (L9). One of the bar's
            # own popups (the field's right-click menu) can take the focus
            # window: the bar stays while it is open and settles when it
            # closes, as the flyout does (phase-4 Decision 54).
            popup = QApplication.activePopupWidget()
            owner = popup.parentWidget() if popup is not None else None
            if owner is not None and owner.window() is self:
                popup.installEventFilter(self)
            else:
                self.hide()
        return handled

    def _hide_if_inactive(self) -> None:
        """Hide the search face when the bar is not the active window once
        its popup has closed (a menu command keeps it active)."""
        if (self.isVisible() and self._mode == "search" and not self.isActiveWindow()
                and QApplication.activePopupWidget() is None):
            self.hide()

    def setVisible(self, visible: bool) -> None:  # noqa: N802 (Qt override)
        # The hand-back is decided before the native hide: afterwards Windows
        # has already picked the next foreground window. Only a foreground on
        # the bar itself (Esc, the slide-away, a failure) or on the
        # click-through subtitle window (where Windows can leave it) is
        # handed on; a window the user activated (Settings, the flyout, a
        # menu, another app) keeps it.
        hand_back = not visible and self.isVisible()
        if hand_back:
            holder = foreground_widget()
            hand_back = holder is self or (holder is not None and bool(
                holder.windowFlags() & Qt.WindowType.WindowTransparentForInput))
        super().setVisible(visible)
        if visible:
            return
        # The window kept at the summon (another app's, Settings or a dialog)
        # gets the foreground back while it still shows, whatever Windows
        # picked (Settings behind the bar must not take the video player's
        # keyboard). With nothing kept (the tray, the flyout; a kept window
        # that has closed, hidden or been minimized counts as none), Windows'
        # own pick of an ordinary window stands; any other pick (the subtitle
        # window, the tray overflow, nothing) gives way to the topmost
        # ordinary window (lead rulings on visual checks 3 and 4, final review
        # round 5).
        if hand_back:
            target = self._return_to
            if target and not window_shows(target):
                target = 0
            if not target and not foreground_is_app_window():
                target = topmost_app_window()
            if target:
                hand_back_foreground(target)
        self._return_to = 0

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        # Every hide drops the status: Esc, a failure, cancel_status, the
        # slide-away (Decision 64).
        self._leave_status()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._theme_palette["raised"]))
        if self._mode == "status":
            paint_bar_line(p, QRectF(_BAR_LINE), self._theme_palette, self._fraction_shown)
        elif self._search.hasFocus():
            paint_bar_line(p, QRectF(_BAR_LINE), self._theme_palette, None)
        p.end()
        if self._cloaked:
            QTimer.singleShot(0, self._uncloak)
