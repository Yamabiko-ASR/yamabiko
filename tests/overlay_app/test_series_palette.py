"""Tests for the series bar (plan phase 5, the "Spotlight bar").

Two tiers live here:

* Pure ``format_hit`` and ``_fetch_cover_bytes`` unit tests (no window).

* pytest-qt tests that construct the real ``SeriesPalette``, drive its
  debounced off-thread search and watch-list fetch (AniList monkeypatched to
  canned hits, covers off the network) and its keys, mouse and summon, and
  assert what the user sees: the dropdown's view and rows, painted pixels,
  the window's place and height, and the ``series_chosen`` /
  ``clearRequested`` side effects. Charter §3.9: these INVOKE the widget and
  assert the effect, not merely that a handler is wired. The painted parts
  themselves are tested in test_series_palette_parts.py; the native window
  (DWM, foreground) in test_real_series_palette.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")

import httpx

from yamabiko.gazetteer.anilist import SeriesHit
from yamabiko.overlay_app.series_palette import _fetch_cover_bytes, format_hit


def _hit(**kw) -> SeriesHit:
    base = dict(
        media_id=1, romaji="Romaji Title", english="English Title",
        native="ネイティブ", season_year=2011, fmt="TV",
        cover_image=None, color=None,
    )
    base.update(kw)
    return SeriesHit(**base)


def _png_bytes(color: str = "#123456", width: int = 10, height: int = 14) -> bytes:
    """A solid poster as the cover worker would hand it over (PNG bytes)."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(color))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    buf.close()
    return bytes(data)


def _render(widget):
    """``widget`` and its children painted into an image, offscreen."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QRegion
    from PySide6.QtWidgets import QWidget

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def _view(palette) -> str:
    """The view the bar's dropdown shows."""
    return palette._dropdown.model().view


def _ids(palette) -> list[int]:
    """The media ids of the rows the bar's dropdown shows."""
    return [hit.media_id for hit in palette._dropdown.model().rows]


def _type_silently(palette, text: str) -> None:
    """Put ``text`` in the field without arming the debounced search (the
    test hands the answer in through _on_results)."""
    palette._search.blockSignals(True)
    palette._search.setText(text)
    palette._search.blockSignals(False)


class _QuietClient:
    """No network: every search finds nothing and the watch list is empty
    (the status tests hand their answers in through _on_results)."""

    def search_series(self, query, limit=8):
        return []

    def fetch_watching(self, username):
        return []


def _picked(qtbot, hit: SeriesHit, **kwargs):
    """A summoned bar whose only result ``hit`` was just picked with Enter:
    the pick's status face."""
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app.series_palette import SeriesPalette

    palette = SeriesPalette(_QuietClient(), username_provider=lambda: "", **kwargs)
    qtbot.addWidget(palette)
    palette.summon()
    _type_silently(palette, "frieren")
    palette._on_results("frieren", True, [hit])
    qtbot.keyClick(palette._search, Qt.Key_Return)
    return palette


# --- format_hit (pure, no Qt) ------------------------------------------


def test_format_hit_prefers_english_over_romaji():
    """english wins when both titles are present.

    Seeded defect: return ``hit.romaji`` unconditionally -> label reads
    'Romaji Title (2011)' -> red.
    """
    assert format_hit(_hit()) == "English Title (2011)"


def test_format_hit_falls_back_to_romaji_when_english_missing():
    """A null english title falls back to romaji.

    Seeded defect: return ``hit.english`` unconditionally -> None is not
    a str / label is 'None (2011)' -> red.
    """
    assert format_hit(_hit(english=None)) == "Romaji Title (2011)"


def test_format_hit_appends_season_year_when_present():
    """season_year is appended in parens.

    Seeded defect: drop the ' (year)' suffix -> label is just the title
    -> red.
    """
    assert format_hit(_hit(season_year=1999)) == "English Title (1999)"


def test_format_hit_omits_year_when_absent():
    """No season_year -> no trailing parens (not '(None)').

    Seeded defect: append ' (year)' unconditionally -> label reads
    'English Title (None)' -> red.
    """
    assert format_hit(_hit(season_year=None)) == "English Title"


# --- _fetch_cover_bytes (unit tier; httpx.MockTransport, no network) ---
#
# The live-CDN @pytest.mark.real coverage for _fetch_cover_bytes (hard
# rule #3) is deferred to Task 11 alongside the real AniList search test,
# to avoid compounding real-tier network calls across tasks -- same
# rationale test_anilist.py records for search_series.


def test_fetch_cover_bytes_returns_content_on_200():
    """A 200 yields the raw poster bytes.

    Seeded defect: return ``resp.content`` regardless of status, or gate
    on ``status_code == 201`` -> the byte-equality assert goes red.
    """
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG-fake-bytes")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert _fetch_cover_bytes("https://img/x.jpg", client=client) == b"\x89PNG-fake-bytes"


def test_fetch_cover_bytes_returns_none_on_404():
    """A non-200 degrades to None (the row keeps its colour tile).

    Seeded defect: drop the ``status_code == 200`` guard -> a 404 body
    is returned as bytes -> red.
    """
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert _fetch_cover_bytes("https://img/x.jpg", client=client) is None


def test_fetch_cover_bytes_returns_none_on_transport_error():
    """A transport-level failure is swallowed to None -- an uncaught
    raise here would hit sys.excepthook from the cover worker thread.

    Seeded defect: remove the try/except -> the ConnectError propagates
    -> red.
    """
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert _fetch_cover_bytes("https://img/x.jpg", client=client) is None


def test_cover_fetch_failure_logs_class_only_never_the_url(caplog):
    """The cover URL embeds the media id of a searched/watched series and
    this log ships in diagnostics zips with no consent gate, so the
    failure line must carry the exception CLASS only -- no URL, no media
    id, at ANY level. The exception text can embed the URL too (real
    transport errors carry request context), so the exc_info channel is
    covered as well: the mock's message deliberately embeds the URL.

    Seeded-defect drill: put ``url`` back into the format args (the
    pre-fix line) or log with ``exc_info=True`` -> the URL fragment lands
    in caplog.text -> red (confirmed red against the pre-fix line).
    """
    import logging

    url = "https://img.anilist.example/media/anime/cover/b171457-poster.jpg"

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connect failed for {req.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.DEBUG):
        assert _fetch_cover_bytes(url, client=client) is None
    assert any(
        "ConnectError" in r.getMessage() for r in caplog.records
    ), "the failure log no longer names the exception class"
    assert "171457" not in caplog.text, (
        "the media id from the cover URL leaked into the log (and from "
        "there into diagnostics zips)"
    )
    assert "anilist.example" not in caplog.text, (
        "the cover URL host leaked into the log"
    )


# --- SeriesPalette: debounced off-thread search + selection ------------


def test_palette_search_populates_then_enter_emits_series_chosen(qtbot, monkeypatch):
    """The end-to-end bar flow: typing a query runs the AniList search OFF
    the GUI thread (monkeypatched to canned hits, so no network), the
    results view shows one row per hit with the first one lit, and Enter
    FIRES ``series_chosen`` with ``(media_id, format_hit(hit), native)``
    and turns the bar into the pick's status.

    Charter §3.9 (wired-not-exercised): this invokes the widget and
    asserts the signal's payload + the status side effect -- it is not a
    ``callable(...)`` shape check.

    Seeded-defect drills:
      - emit ``hit.romaji`` instead of ``format_hit(hit)`` on the pick
        -> label arg becomes 'Mahou Shoujo Madoka Magica (2011)' -> red.
      - emit the wrong id -> media_id arg wrong -> red.
      - skip the status face after the pick -> _mode stays "search" -> red.
      - never derive the results view from resultsReady -> waitUntil times
        out -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_RESULTS

    canned = [
        SeriesHit(9756, "Mahou Shoujo Madoka Magica",
                  "Puella Magi Madoka Magica", "魔法少女まどか☆マギカ",
                  2011, "TV", cover_image=None, color="#e4a15d"),
        SeriesHit(1, "Second Series", None, None, None, "TV",
                  cover_image=None, color=None),
    ]
    # Off the network: the bar's worker thread calls this in place of the
    # live AniList round-trip.
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.search_series",
        lambda self, query, limit=8: list(canned),
    )

    palette = SeriesPalette()
    qtbot.addWidget(palette)
    palette.summon()

    palette._search.setText("madoka")
    # Debounce (250 ms) + the worker QThread emitting resultsReady back
    # to the GUI thread; waitUntil pumps the event loop meanwhile.
    qtbot.waitUntil(lambda: _view(palette) == VIEW_RESULTS, timeout=4000)
    assert _ids(palette) == [9756, 1]
    assert palette._dropdown.model().lit == 0, "the first result must be lit"

    with qtbot.waitSignal(palette.series_chosen, timeout=2000) as blocker:
        qtbot.keyClick(palette._search, Qt.Key_Return)  # pick the lit row

    assert blocker.args == [
        9756, "Puella Magi Madoka Magica (2011)", "魔法少女まどか☆マギカ",
    ], (
        f"series_chosen carried {blocker.args!r}; expected the top hit's "
        f"media_id + format_hit label + Japanese title"
    )
    assert palette.isVisible() and palette._mode == "status", (
        "the bar must turn into the pick's status"
    )


def test_palette_empty_query_clears_results(qtbot, monkeypatch):
    """Clearing the field back to empty must not fire a search and must
    drop the results at once (L4: no debounce for the empty field) -- an
    empty AniList query is a wasted round-trip and a stale-results footgun.
    Without a username the bar is then alone.

    Seeded defects: run the search on an empty/whitespace query anyway ->
    calls gains an entry -> red; wait for the debounce before re-deriving
    -> the results still show right after the clear -> red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_BAR, VIEW_RESULTS

    calls: list[str] = []

    def _fake_search(self, query, limit=8):
        calls.append(query)
        return [SeriesHit(1, "X", "X", None, 2000, "TV")]

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.search_series", _fake_search,
    )

    palette = SeriesPalette()
    qtbot.addWidget(palette)

    palette._search.setText("x")
    qtbot.waitUntil(lambda: _view(palette) == VIEW_RESULTS, timeout=4000)

    palette._search.setText("   ")  # whitespace-only -> treated as empty
    assert _view(palette) == VIEW_BAR, "clearing the field kept the results"
    qtbot.wait(400)  # past the debounce
    assert calls == ["x"], f"empty/whitespace query should not search; calls={calls}"


def test_search_worker_asks_anilist_for_six_rows(qapp):
    """The bar shows at most six rows (spec, Surface 4), so the search
    worker's AniList request asks for six (the GraphQL variable ``n``)
    instead of the client's default eight.

    Seeded defect: call ``search_series(query)`` without the limit -> the
    request carries n 8 -> red.
    """
    import json

    from yamabiko.gazetteer.anilist import AniListClient
    from yamabiko.overlay_app.series_palette import _SearchThread

    sent: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.read()))
        return httpx.Response(200, json={"data": {"Page": {"media": []}}})

    client = AniListClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    _SearchThread(client, "frieren").run()  # the worker body, on this thread
    assert [body["variables"]["n"] for body in sent] == [6]


# --- The bar window (phase 5 Task 5: views, keys, place, motion, chip) -------


def test_dropdown_view_follows_the_settled_answer_for_the_typed_query(qtbot, monkeypatch):
    """Decision 36: a failed search shows the offline view (not "no
    matches"); an answer for text the field no longer holds is dropped and
    the shown view stays until the answer for the new text settles (no
    loading row); then no matches with the query, or the results capped at
    six with the first row lit, and the window grows to fit.

    Seeded defects: the worker reports a failure as (True, []) -> no matches
    instead of offline -> red; drop the stale-query guard -> the old answer
    replaces the view -> red; clear the view while waiting -> red; no
    MAX_ROWS cap -> eight rows -> red; the window not resized -> red.
    """
    from yamabiko.gazetteer.anilist import AniListUnavailable
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import (
        VIEW_NO_MATCHES, VIEW_OFFLINE, VIEW_RESULTS, dropdown_height,
    )

    def _raise(self, query, limit=8):
        raise AniListUnavailable("outage")

    monkeypatch.setattr("yamabiko.gazetteer.anilist.AniListClient.search_series", _raise)
    palette = SeriesPalette(username_provider=lambda: "", reduce_motion=True)
    qtbot.addWidget(palette)
    palette.summon()
    palette._search.setText("frieren")
    qtbot.waitUntil(lambda: _view(palette) == VIEW_OFFLINE, timeout=4000)
    assert not palette._dropdown.model().list_unavailable, "no username: no list line"

    _type_silently(palette, "frierne")
    palette._on_results("frieren", True, [_hit(media_id=1)])  # stale: the field moved on
    assert _view(palette) == VIEW_OFFLINE

    palette._on_results("frierne", True, [])
    model = palette._dropdown.model()
    assert (model.view, model.query) == (VIEW_NO_MATCHES, "frierne")

    palette._on_results("frierne", True, [_hit(media_id=i) for i in range(1, 9)])
    model = palette._dropdown.model()
    assert model.view == VIEW_RESULTS
    assert (_ids(palette), model.lit) == ([1, 2, 3, 4, 5, 6], 0)
    assert palette.height() == 48 + dropdown_height(model) == 400


def test_up_down_move_the_lit_row_and_enter_picks_it(qtbot):
    """Decision 37: Up/Down in the field move the lit row by one and stop
    at the ends; Enter picks the lit row, not the first one.

    Seeded defects: no clamp (Down past the last row lights nothing and
    Enter no-ops), Up and Down swapped, or Enter picking row 0 -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app.series_palette import SeriesPalette

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    _type_silently(palette, "show")
    palette._on_results(
        "show", True, [_hit(media_id=i, english=f"Show {i}", season_year=2020) for i in (1, 2, 3)])
    assert palette._dropdown.model().lit == 0

    for _ in range(3):
        qtbot.keyClick(palette._search, Qt.Key_Down)
    assert palette._dropdown.model().lit == 2, "Down must stop at the last row"
    qtbot.keyClick(palette._search, Qt.Key_Up)
    assert palette._dropdown.model().lit == 1

    with qtbot.waitSignal(palette.series_chosen, timeout=1000) as blocker:
        qtbot.keyClick(palette._search, Qt.Key_Return)
    assert blocker.args == [2, "Show 2 (2020)", "ネイティブ"]


def test_esc_and_losing_activation_hide_the_bar(qtbot):
    """L13 / Decision 37: Esc in the field closes the bar, and so does a
    click elsewhere (the window losing activation), like the flyout.

    Seeded defects: no Esc branch in the field's filter, or no
    WindowDeactivate handling -> the bar stays -> red.
    """
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QApplication
    from yamabiko.overlay_app.series_palette import SeriesPalette

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.keyClick(palette._search, Qt.Key_Escape)
    assert not palette.isVisible(), "Esc did not close the bar"

    palette.summon()
    QApplication.sendEvent(palette, QEvent(QEvent.Type.WindowDeactivate))
    assert not palette.isVisible(), "losing activation did not close the bar"


def test_the_fields_right_click_menu_keeps_the_bar_open_until_it_closes(qtbot, monkeypatch):
    """L13, as the flyout (phase-4 Decision 54): a deactivation while the
    field's right-click menu (QLineEdit's standard menu, a popup) is open
    does not close the bar, so Paste stays reachable; once the menu closes
    with the bar no longer active (the click went elsewhere), the bar hides.

    Seeded defects: hide on every deactivation -> the bar closes under its
    own menu -> red; no settle when the menu closes -> the bar stays open
    behind the other window -> red.
    """
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QContextMenuEvent, QGuiApplication
    from PySide6.QtWidgets import QApplication
    from yamabiko.overlay_app.series_palette import SeriesPalette

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    field = palette._search
    QApplication.sendEvent(field, QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, QPoint(20, 12), field.mapToGlobal(QPoint(20, 12))))
    menu = QApplication.activePopupWidget()
    assert menu is not None and menu.parentWidget() is field, "precondition: the field's menu is open"
    # The open menu takes the focus window (a queued platform event); let it
    # land while the menu is open, as the flyout's test does.
    qtbot.waitUntil(lambda: QGuiApplication.focusWindow() not in (None, palette.windowHandle()),
                    timeout=2000)
    QApplication.sendEvent(palette, QEvent(QEvent.Type.WindowDeactivate))
    assert palette.isVisible(), "the field's right-click menu closed the bar"

    monkeypatch.setattr(palette, "isActiveWindow", lambda: False)  # the click went elsewhere
    menu.hide()
    qtbot.waitUntil(lambda: not palette.isVisible(), timeout=2000)


def test_summon_places_the_bar_top_center_on_the_subtitle_screen(qtbot):
    """L3 / Decision 38: every summon centers the 640 px bar on the work
    area of the screen the provider names (the overlay's subtitle monitor),
    24 px below its top; without a screen it falls back to the primary one.

    Seeded defects: center on the primary or the bar's own screen (ignore
    the provider), forget TOP_GAP, center on the full geometry, or place
    only on the first summon -> red.
    """
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QGuiApplication
    from yamabiko.overlay_app.series_palette import SeriesPalette

    class _Screen:  # a second monitor right of the primary, taskbar on top
        def availableGeometry(self):
            return QRect(1920, 40, 2560, 1400)

        def geometry(self):
            return QRect(1920, 0, 2560, 1440)

        def devicePixelRatio(self):
            return 1.0

    screens = [_Screen()]
    palette = SeriesPalette(username_provider=lambda: "", screen_provider=lambda: screens[0])
    qtbot.addWidget(palette)
    palette.summon()
    assert (palette.x(), palette.y(), palette.width(), palette.height()) == (2880, 64, 640, 48)

    palette.hide()
    screens[0] = None
    palette.summon()
    area = QGuiApplication.primaryScreen().availableGeometry()
    assert (palette.x(), palette.y()) == (area.x() + (area.width() - 640) // 2, area.top() + 24)


def test_unfold_animates_the_window_height_and_reduce_motion_jumps_to_it(qtbot, monkeypatch):
    """L12 / Decisions 35, 61 (probe P1): new dropdown content grows the
    window to 48 + its height over the unfold; turning reduce motion on
    mid-way jumps to the end, and from then on height changes are instant.

    Seeded defects: no animation (the height is there at once), reduce
    motion ignored (still between the heights), or the window not
    following the view -> red.
    """
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import dropdown_height

    monkeypatch.setattr(sp, "_UNFOLD_MS", 3000)  # slow enough to see it mid-way
    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    _type_silently(palette, "show")
    palette._on_results("show", True, [_hit(media_id=i) for i in (1, 2, 3)])
    target = 48 + dropdown_height(palette._dropdown.model())
    assert target == 244
    qtbot.waitUntil(lambda: 48 < palette.height() < target, timeout=2000)

    palette.set_reduce_motion(True)
    assert palette.height() == target, "reduce motion did not jump to the end"
    palette._on_results("show", True, [_hit(media_id=9)])
    assert palette.height() == 48 + dropdown_height(palette._dropdown.model()) == 140


def test_a_fold_after_a_finished_unfold_starts_from_the_shown_height(qtbot):
    """Decision 61 (probe P1): the next height change after a finished
    unfold animates from the window's height: the window never takes the
    new end height (and the old one again) before the first frame.

    Seeded defect: set the animation's start and end values without the
    signal blocker -> the window jumps to 140 and back to 244 at once ->
    red.
    """
    from PySide6.QtCore import QAbstractAnimation, QEvent, QObject
    from yamabiko.overlay_app.series_palette import SeriesPalette

    class _Heights(QObject):
        """Every height the window is resized to."""

        def __init__(self):
            super().__init__()
            self.seen: list[int] = []

        def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
            if event.type() == QEvent.Type.Resize:
                self.seen.append(event.size().height())
            return False

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    _type_silently(palette, "show")
    palette._on_results("show", True, [_hit(media_id=i) for i in (1, 2, 3)])
    qtbot.waitUntil(lambda: palette._height_anim.state() == QAbstractAnimation.State.Stopped,
                    timeout=2000)
    assert palette.height() == 244, "precondition: the unfold finished at the results' height"

    heights = _Heights()
    palette.installEventFilter(heights)
    palette._on_results("show", True, [_hit(media_id=9)])
    assert heights.seen == [], f"the window jumped before the fold's first frame: {heights.seen}"
    qtbot.waitUntil(lambda: palette.height() == 140, timeout=2000)
    assert heights.seen == sorted(heights.seen, reverse=True), (
        f"the fold did not shrink the window steadily: {heights.seen}")
    palette.removeEventFilter(heights)


def test_try_again_and_enter_retry_the_search_and_the_watch_list(qtbot, monkeypatch):
    """L4 / Decision 56: in the offline view Enter and the Try again button
    both retry the search (the field holds text) and the watch list; the
    offline view stays until an answer replaces it, here the results, and
    the recovered watch list shows once the field is cleared.

    Seeded defects: Enter does nothing offline, Try again not wired to the
    retry, or the retry skips the watch list -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.gazetteer.anilist import AniListUnavailable
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import (
        VIEW_OFFLINE, VIEW_WATCHING, DropdownModel, TryAgainButton,
    )

    online = [False]
    searches: list[str] = []
    fetches: list[str] = []

    def _search(self, query, limit=8):
        searches.append(query)
        if not online[0]:
            raise AniListUnavailable("outage")
        return [_hit(media_id=201)]

    def _watching(self, username):
        fetches.append(username)
        if not online[0]:
            raise AniListUnavailable("outage")
        return [_hit(media_id=101)]

    monkeypatch.setattr("yamabiko.gazetteer.anilist.AniListClient.search_series", _search)
    monkeypatch.setattr("yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _watching)
    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()
    palette._search.setText("madoka")
    offline = DropdownModel(view=VIEW_OFFLINE, list_unavailable=True)
    qtbot.waitUntil(lambda: palette._dropdown.model() == offline, timeout=4000)

    qtbot.keyClick(palette._search, Qt.Key_Return)  # Enter retries, still offline
    qtbot.waitUntil(lambda: (len(searches), len(fetches)) == (2, 2)
                    and palette._search_threads == set()
                    and palette._watching_threads == set(), timeout=4000)
    assert palette._dropdown.model() == offline

    online[0] = True
    qtbot.mouseClick(palette._dropdown.findChild(TryAgainButton), Qt.LeftButton)
    qtbot.waitUntil(lambda: _ids(palette) == [201]
                    and palette._watching_threads == set(), timeout=4000)
    assert (len(searches), len(fetches)) == (3, 3)
    palette._search.setText("")
    assert (_view(palette), _ids(palette)) == (VIEW_WATCHING, [101])


def test_a_failed_search_beside_a_loaded_watch_list_shows_no_list_line(qtbot):
    """Decision 36: the offline view's "Your AniList list is unavailable."
    line reports a failed watch-list fetch. With a username whose list
    loaded, a failed search shows the offline message without it.

    Seeded defect: offline for a failed search with
    ``list_unavailable=bool(username)`` -> the list line shows -> red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import (
        VIEW_OFFLINE, VIEW_WATCHING, DropdownModel,
    )

    class _ListClient:
        def search_series(self, query, limit=8):
            return []

        def fetch_watching(self, username):
            return [_hit(media_id=101)]

    palette = SeriesPalette(_ListClient(), username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING
                    and palette._watching_threads == set(), timeout=4000)
    _type_silently(palette, "madoka")
    palette._on_results("madoka", False, [])
    assert palette._dropdown.model() == DropdownModel(view=VIEW_OFFLINE, list_unavailable=False)


def test_a_new_summon_forgets_the_failures_of_the_last_open(qtbot):
    """Decision 38: the offline view belongs to the open whose request
    failed. Summoned again, the bar shows alone while the watch list is
    fetched anew (empty field) or the field's search re-runs (typed text).

    Seeded defects: drop ``self._watch_failed = None`` from summon -> the
    empty field shows the last open's watch-list failure -> red; drop the
    reset of a failed answer -> the typed text shows the last open's search
    failure -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.gazetteer.anilist import AniListUnavailable
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_BAR, VIEW_OFFLINE

    searches: list[str] = []

    class _DownClient:
        def search_series(self, query, limit=8):
            searches.append(query)
            raise AniListUnavailable("outage")

        def fetch_watching(self, username):
            raise AniListUnavailable("outage")

    palette = SeriesPalette(_DownClient(), username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)

    def _settled_offline() -> bool:
        return (_view(palette) == VIEW_OFFLINE and palette._watching_threads == set()
                and palette._search_threads == set())

    palette.summon()
    qtbot.waitUntil(_settled_offline, timeout=4000)
    qtbot.keyClick(palette._search, Qt.Key_Escape)
    palette.summon()
    assert _view(palette) == VIEW_BAR, "the empty field showed the last open's watch-list failure"
    qtbot.waitUntil(_settled_offline, timeout=4000)  # this open's own failure

    palette._search.setText("madoka")
    qtbot.waitUntil(lambda: searches == ["madoka"] and _settled_offline(), timeout=4000)
    qtbot.keyClick(palette._search, Qt.Key_Escape)
    palette.summon()
    assert _view(palette) == VIEW_BAR, "the typed text showed the last open's search failure"
    qtbot.waitUntil(lambda: len(searches) == 2 and _settled_offline(), timeout=4000)


def test_clear_asks_for_no_series_and_the_chip_follows_the_active_series(qtbot):
    """L8 / Decisions 42, 57: the chip shows only while a series is active,
    right-aligned at x 627 with its short title and the full title as its
    tooltip, the field ending 12 px before it; Clear emits clearRequested
    and the bar stays open; set_current_series(None) hides the chip and the
    field takes the room back.

    Seeded defects: the chip shown without a series, the long title shown,
    Clear not wired, the bar closing on Clear, or the field not resized ->
    red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app.controls import LinkButton
    from yamabiko.overlay_app.series_palette import SeriesPalette

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    assert not palette._chip.isVisible()
    assert palette._search.width() == 627 - 48

    palette.set_current_series(5, "The Apothecary Diaries Season 3 (2025)")
    chip = palette._chip
    assert chip.isVisible()
    assert chip.title() == "Apothecary Diaries S3"
    assert chip.toolTip() == "The Apothecary Diaries Season 3"
    assert (chip.x() + chip.width(), chip.y()) == (627, 10)
    assert palette._search.x() + palette._search.width() == chip.x() - 12

    with qtbot.waitSignal(palette.clearRequested, timeout=1000):
        qtbot.mouseClick(chip.findChild(LinkButton), Qt.LeftButton)
    assert palette.isVisible(), "Clear closed the bar"

    palette.set_current_series(None)
    assert not chip.isVisible()
    assert palette._search.width() == 627 - 48


def test_summon_cloaks_a_hidden_bar_until_it_has_painted(qtbot, monkeypatch):
    """Probe P4 / Decision 59: a summon of the hidden bar cloaks it from DWM
    before show and uncloaks it after its first paint (or the 100 ms safety
    timer), never leaving it cloaked; a summon of the visible bar does not
    cloak it again. (The real DWM call: test_real_series_palette.py.)

    Seeded defects: no uncloak (an invisible window holding the keyboard),
    or a cloak on every summon -> red.
    """
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.series_palette import SeriesPalette

    cloaks: list[bool] = []
    monkeypatch.setattr(sp, "set_cloak",
                        lambda window, cloaked: cloaks.append(cloaked) or True)
    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    assert cloaks == [True], "the bar was shown uncloaked"
    qtbot.waitUntil(lambda: cloaks == [True, False], timeout=1000)

    palette.summon()
    qtbot.wait(200)
    assert cloaks == [True, False], "a summon of the visible bar cloaked it again"


def test_a_hiding_bar_gives_the_foreground_back_only_from_itself_or_the_subtitle_window(
    qtbot, monkeypatch,
):
    """Visual check, observation 1, under the lead's ruling on final review
    round 3: summoned over another app (the video player, Ctrl+Alt+S), the
    bar keeps that app's window. When the foreground is on the bar itself
    (Esc) or on the click-through subtitle window (where Windows left it
    after a pick) as the bar starts to hide, and Windows then leaves it with
    this process, the bar hands it back to that app. A foreground the user
    gave to another Yamabiko window (Settings, the flyout) or to another app
    stays there. (The real Win32 calls: test_real_series_palette.py.)

    Seeded defects: no hand-back -> red; hand back whenever this process
    holds the foreground after the hide (the round-3 behavior) -> the
    Settings check red; the subtitle window not counted -> the slide-away
    check red.
    """
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QApplication, QWidget
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.overlay_window import OverlayWindow

    video = 0x51C0
    holder: list = [None]  # this process's window in the foreground, None for another app's
    handed: list[int] = []
    monkeypatch.setattr(sp, "_DONE_HOLD_MS", 50)
    monkeypatch.setattr(sp, "foreign_foreground", lambda: video)
    monkeypatch.setattr(sp, "foreground_widget", lambda: holder[0])
    # After the hide Windows leaves it with this process; another app's
    # foreground (None) is an app window.
    monkeypatch.setattr(sp, "foreground_is_app_window", lambda: holder[0] is None)
    monkeypatch.setattr(sp, "hand_back_foreground", lambda hwnd: handed.append(hwnd) or True)
    monkeypatch.setattr(sp, "window_shows", lambda hwnd: True)  # the kept window still shows
    subtitles = OverlayWindow()  # click-through: WindowTransparentForInput
    settings = QWidget()  # a Yamabiko window the user types in
    qtbot.addWidget(subtitles)
    qtbot.addWidget(settings)

    palette = sp.SeriesPalette(_QuietClient(), username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    holder[0] = palette
    qtbot.keyClick(palette._search, Qt.Key_Escape)
    assert handed == [video], "Esc left the foreground with this process"

    holder[0] = None
    palette = _picked(qtbot, _hit(media_id=7))
    holder[0] = subtitles  # Windows moved it there while the status showed
    palette.show_built(7)
    qtbot.waitUntil(lambda: not palette.isVisible(), timeout=3000)
    assert handed == [video, video], "the slide-away left the foreground on the subtitle window"

    for chosen in (settings, None):  # the user activated Settings, then another app
        palette.summon()
        holder[0] = chosen
        QApplication.sendEvent(palette, QEvent(QEvent.Type.WindowDeactivate))
        assert not palette.isVisible()
    assert handed == [video, video], "the bar took the foreground from a window the user chose"


def test_a_bar_summoned_over_the_taskbar_or_the_tray_overflow_never_gives_it_the_foreground(
    qtbot, monkeypatch,
):
    """Visual check 2, item 6: summoned from the short menu with the tray
    icon in the overflow, the window in front is Explorer's overflow; handing
    the foreground back to it after the pick reopened the overflow. The bar
    keeps no taskbar or overflow window (window_chrome.SHELL_WINDOW_CLASSES):
    Esc with the bar in front gives the foreground to the topmost app window
    instead (lead ruling on visual check 3), and a kept app's window gets it
    back. (The real class names: test_real_series_palette.py.)

    Seeded defect: keep every foreign window -> the shell windows are handed
    the foreground -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app import series_palette as sp

    classes = {
        0x101: "Shell_TrayWnd", 0x102: "Shell_SecondaryTrayWnd",
        0x103: "NotifyIconOverflowWindow", 0x104: "TopLevelWindowForOverflowXamlIsland",
        0x51C0: "Qt6110QWindowIcon",  # the video player
    }
    top = 0x7070  # the topmost app window
    front = [0]
    handed: list[int] = []
    palette = sp.SeriesPalette(_QuietClient(), username_provider=lambda: "")
    qtbot.addWidget(palette)
    monkeypatch.setattr(sp, "foreign_foreground", lambda: front[0])
    monkeypatch.setattr(sp, "window_class", lambda hwnd: classes[hwnd])
    monkeypatch.setattr(sp, "foreground_widget", lambda: palette)  # the bar, as Esc hides it
    monkeypatch.setattr(sp, "foreground_is_app_window", lambda: False)  # left with this process
    monkeypatch.setattr(sp, "topmost_app_window", lambda: top)
    monkeypatch.setattr(sp, "hand_back_foreground", lambda hwnd: handed.append(hwnd) or True)
    monkeypatch.setattr(sp, "window_shows", lambda hwnd: True)  # the kept window still shows

    for hwnd in classes:
        front[0] = hwnd
        palette.summon()
        qtbot.keyClick(palette._search, Qt.Key_Escape)
    assert handed == [top, top, top, top, 0x51C0], "the bar gave the foreground to a shell window"


def test_a_bar_with_nothing_kept_gives_the_foreground_to_the_topmost_app_window(
    qtbot, monkeypatch,
):
    """Lead ruling on visual check 3 (A2): summoned from the tray (the menu,
    the overflow or the flyout in front: this process's or the shell's
    windows), the bar keeps nothing. When it hides with the foreground on
    itself (or the subtitle window) and Windows leaves it with this process
    (in the product the open overflow took it and kept the keyboard), it
    gives the foreground to the topmost ordinary app window
    (window_chrome.topmost_app_window). A kept window still wins, also over
    Windows' own pick of an ordinary window (Settings open behind the bar:
    lead ruling on visual check 4), and a window the user activated
    (Settings) keeps the foreground.

    Seeded defects: no fallback when nothing is kept -> red; the fallback
    ahead of the kept window -> red; Windows' pick of Settings beating the
    kept window -> red; a fallback over Windows' own pick, or from Settings
    -> red.
    """
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QApplication, QWidget
    from yamabiko.overlay_app import series_palette as sp

    video, top = 0x51C0, 0x7070
    kept = [0]
    holder: list = [None]
    app_after = [False]  # Windows' own pick after the hide is an app window
    handed: list[int] = []
    monkeypatch.setattr(sp, "foreign_foreground", lambda: kept[0])
    monkeypatch.setattr(sp, "window_class", lambda hwnd: "Qt6110QWindowIcon")
    monkeypatch.setattr(sp, "foreground_widget", lambda: holder[0])
    monkeypatch.setattr(sp, "foreground_is_app_window", lambda: app_after[0])
    monkeypatch.setattr(sp, "topmost_app_window", lambda: top)
    monkeypatch.setattr(sp, "hand_back_foreground", lambda hwnd: handed.append(hwnd) or True)
    monkeypatch.setattr(sp, "window_shows", lambda hwnd: True)  # the kept window still shows
    settings = QWidget()  # a Yamabiko window the user types in
    qtbot.addWidget(settings)
    palette = sp.SeriesPalette(_QuietClient(), username_provider=lambda: "")
    qtbot.addWidget(palette)

    def esc() -> None:
        holder[0] = palette
        qtbot.keyClick(palette._search, Qt.Key_Escape)

    palette.summon()  # from the tray: nothing kept
    esc()  # Windows then picks the tray overflow (no app window)
    assert handed == [top], "with nothing kept the bar left the foreground off every app"

    kept[0] = video  # over the video player (Ctrl+Alt+S)
    palette.summon()
    esc()
    assert handed == [top, video], "the topmost app window won over the kept one"

    app_after[0] = True  # Windows picked Settings, open behind the bar
    palette.summon()
    esc()
    assert handed == [top, video, video], "Windows' pick of Settings beat the kept window"

    kept[0] = 0  # from the tray again: Windows picked an ordinary window itself
    palette.summon()
    esc()
    app_after[0] = False
    palette.summon()
    holder[0] = settings  # the user clicked Settings
    QApplication.sendEvent(palette, QEvent(QEvent.Type.WindowDeactivate))
    assert not palette.isVisible()
    assert handed == [top, video, video], (
        "the bar moved a foreground Windows or the user had placed")


def test_a_bar_summoned_from_a_yamabiko_window_gives_the_foreground_back_to_it(
    qtbot, monkeypatch,
):
    """Lead ruling on final review round 5 (I1): summoned while one of this
    process's windows the user types in (Settings, a dialog) is in front, the
    bar keeps that window and gives it the foreground back on Esc, instead of
    the z-order fallback raising another app over Settings. The tray flyout
    (a tool window) and the click-through subtitle window are not kept: from
    them the fallback runs.

    Seeded defect: keep only another app's window -> the topmost app takes
    the foreground from Settings -> red.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.overlay_window import OverlayWindow

    top = 0x7070
    holder: list = [None]
    handed: list[int] = []
    monkeypatch.setattr(sp, "foreign_foreground", lambda: 0)  # this process is in front
    monkeypatch.setattr(sp, "foreground_widget", lambda: holder[0])
    monkeypatch.setattr(sp, "foreground_is_app_window", lambda: False)
    monkeypatch.setattr(sp, "topmost_app_window", lambda: top)
    monkeypatch.setattr(sp, "hand_back_foreground", lambda hwnd: handed.append(hwnd) or True)
    monkeypatch.setattr(sp, "window_shows", lambda hwnd: True)  # the kept window still shows
    settings = QWidget()  # a plain window, as Settings
    flyout = QWidget(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    subtitles = OverlayWindow()
    for window in (settings, flyout, subtitles):
        qtbot.addWidget(window)
        window.show()
    palette = sp.SeriesPalette(_QuietClient(), username_provider=lambda: "")
    qtbot.addWidget(palette)

    for origin in (settings, flyout, subtitles):
        holder[0] = origin
        palette.summon()
        holder[0] = palette  # the bar in front as Esc hides it
        qtbot.keyClick(palette._search, Qt.Key_Escape)
    assert handed == [int(settings.winId()), top, top], (
        "the bar did not give the foreground back to the Yamabiko window it came from")


def test_focused_field_draws_the_accent_underline_inside_the_dwm_edge(qtbot):
    """L14 / probe P5 (Decision 62): while the field has focus the bar
    paints a 2 px accent underline at y 45-46, one row above the window's
    outermost row (which the DWM border covers); without focus it is gone.

    Seeded defects: the underline at y 46-47 (half hidden under the
    border), never painted, or painted without focus -> red.
    """
    from PySide6.QtGui import QColor
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import NIGHT

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(palette._search.hasFocus, timeout=2000)
    raised, accent = QColor(NIGHT["raised"]), QColor(NIGHT["accent"])
    image = _render(palette)
    assert [image.pixelColor(600, y) for y in (44, 45, 46, 47)] == [raised, accent, accent, raised]

    palette._search.clearFocus()
    assert _render(palette).pixelColor(600, 45) == raised


# --- The status face (phase 5 Task 6: L9, L10, Decisions 39, 63-67) ---------


def test_pick_turns_the_bar_into_the_loading_status_that_survives_deactivation(qtbot):
    """L9 / Decisions 39, 63: a pick folds the bar to 48 px and shows the
    status: the field and the chip give way to the picked series' cover
    and "Loading character names for <short>…" over an empty progress
    line; the bar keeps the keyboard, and losing activation does not hide
    it.

    Seeded defects: hide on the pick (Task 5's behavior), the long title,
    the dropdown left open, no cover, the focus underline instead of the
    rail, or a deactivation that hides the status -> red.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication
    from yamabiko.overlay_app.theme import NIGHT

    hit = _hit(media_id=7, english="Frieren: Beyond Journey's End", color="#e4a15d")
    palette = _picked(qtbot, hit, reduce_motion=True)
    assert (palette._mode, palette._status_media_id) == ("status", 7)
    assert palette.isVisible() and palette.height() == 48
    assert palette._status.text() == "Loading character names for Frieren\u2026"
    assert not palette._status.is_done()
    assert palette._status.isVisible() and not palette._search.isVisible()
    qtbot.waitUntil(palette.hasFocus, timeout=1000)
    image = _render(palette)
    assert image.pixelColor(58, 24) == QColor("#e4a15d")       # the status cover
    assert image.pixelColor(320, 45) == QColor(NIGHT["rail"])  # the empty progress line

    QApplication.sendEvent(palette, QEvent(QEvent.Type.WindowDeactivate))
    assert palette.isVisible(), "losing activation hid the status"


def test_build_progress_fills_the_line_monotonically_for_the_picked_series_only(qtbot):
    """L10 / Decision 66: the tray's progress for the picked series tweens
    the line toward its target over 150 ms; a lower value never moves it
    back, another series' progress is ignored, and with reduce motion the
    line jumps.

    Seeded defects: no tween (the shown fill is there at once), a target
    that follows the last value down, another series' progress accepted,
    or the line painted from the wrong value -> red.
    """
    from PySide6.QtGui import QColor
    from yamabiko.overlay_app.theme import NIGHT

    palette = _picked(qtbot, _hit(media_id=7))
    palette.show_build_progress(7, 0.45)
    assert palette._fraction_shown < 0.45, "the fill did not tween"
    palette.show_build_progress(8, 0.9)   # another series' build
    palette.show_build_progress(7, 0.15)  # never backwards
    assert palette._fraction_target == pytest.approx(0.45)
    qtbot.waitUntil(lambda: palette._fraction_shown == pytest.approx(0.45), timeout=2000)
    image = _render(palette)
    assert image.pixelColor(256, 45) == QColor(NIGHT["accent"])
    assert image.pixelColor(320, 45) == QColor(NIGHT["rail"])

    palette.set_reduce_motion(True)
    palette.show_build_progress(7, 0.75)
    assert palette._fraction_shown == pytest.approx(0.75)


@pytest.mark.parametrize("ending", ["slide", "fade_in_place", "reduce_motion"])
def test_built_shows_now_watching_then_leaves_and_restores_the_window(
    qtbot, monkeypatch, ending,
):
    """L9 / L12 / Decisions 65, 66 (probe P2): on built for the pick the
    line fills and the check with "Now watching <short>." holds; then the
    bar slides up while fading (fades in place when a monitor sits right
    above the subtitle screen; just hides with reduce motion) and ends
    hidden, at full opacity, in its place, back in the search face.

    Seeded defects: no check or the loading text kept, no hold, moving up
    over a monitor above, fading with reduce motion, the opacity or the
    position not restored, or the status kept after the hide -> red.
    """
    from yamabiko.overlay_app import series_palette as sp

    monkeypatch.setattr(sp, "_DONE_HOLD_MS", 100)
    monkeypatch.setattr(sp, "_SLIDE_MS", 400)
    monkeypatch.setattr(sp, "_screen_above", lambda geometry: ending == "fade_in_place")
    palette = _picked(qtbot, _hit(media_id=7, english="Frieren: Beyond Journey's End"),
                      reduce_motion=(ending == "reduce_motion"))
    home = palette.pos()
    opacities: list[float] = []
    set_opacity = palette.setWindowOpacity
    monkeypatch.setattr(palette, "setWindowOpacity",
                        lambda value: (opacities.append(value), set_opacity(value)))

    palette.show_built(7)
    assert palette._status.text() == "Now watching Frieren."
    assert palette._status.is_done()
    assert palette._fraction_target == 1.0
    assert palette.isVisible(), "the done status must hold first"
    if ending == "reduce_motion":
        qtbot.waitUntil(lambda: not palette.isVisible(), timeout=2000)
        assert opacities == [], "reduce motion still faded the bar"
    else:
        qtbot.waitUntil(lambda: palette.windowOpacity() < 0.9, timeout=2000)
        assert (palette.y() < home.y()) is (ending == "slide")
        qtbot.waitUntil(lambda: not palette.isVisible(), timeout=2000)
        assert opacities[-1] == 1.0
    assert palette.windowOpacity() == 1.0
    qtbot.waitUntil(lambda: palette.pos() == home, timeout=1000)
    assert palette._mode == "search"


@pytest.mark.parametrize(
    "other, above",
    [
        pytest.param((0, -1440, 2560, 1440), True, id="150-percent-monitor-above"),
        pytest.param((1920, 0, 2560, 1440), False, id="side-by-side"),
        pytest.param((-2560, -1440, 2560, 1440), False, id="diagonal-only"),
    ],
)
def test_a_monitor_above_is_found_in_native_pixels_at_mixed_scale(monkeypatch, other, above):
    """P2 / lead ruling on the Task 6 review: whether the slide-away must
    fade in place is decided in native pixels. Qt on Windows keeps each
    screen's native top-left and divides only its size by the scale, so a
    150 % monitor right above a 100 % subtitle screen ends 480 logical px
    short of it; a monitor beside it, or touching only its corner, is not
    above it.

    Seeded defects: the logical probe point just above the screen
    (screenAt(center.x, top - 1), the first version) or the logical size
    instead of the native one -> the scaled monitor above is missed -> red;
    one shared pixel column counted as overlap -> the diagonal monitor
    (1 px long from the rounded logical size) counts -> red.
    """
    from types import SimpleNamespace

    from PySide6.QtCore import QRect
    from yamabiko.overlay_app import series_palette as sp

    class _Screen:
        """A monitor as Qt on Windows reports it: the native top-left, the
        native size divided by the scale (rounded), and the scale."""

        def __init__(self, native: QRect, dpr: float) -> None:
            self._geometry = QRect(native.topLeft(), native.size() / dpr)
            self._dpr = dpr

        def geometry(self) -> QRect:
            return QRect(self._geometry)

        def devicePixelRatio(self) -> float:
            return self._dpr

    subtitle = QRect(0, 0, 1920, 1080)  # 100 %: native and logical agree
    screens = [_Screen(subtitle, 1.0), _Screen(QRect(*other), 1.5)]
    # Both of Qt's screen lookups, as Qt answers them (logical geometry).
    monkeypatch.setattr(sp, "QGuiApplication", SimpleNamespace(
        screens=lambda: list(screens),
        screenAt=lambda point: next(
            (s for s in screens if s.geometry().contains(point)), None),
    ))
    assert sp._screen_above(subtitle) is above


def test_failure_cancel_and_esc_hide_the_status_at_once_and_drop_it(qtbot):
    """L9 / Decisions 39, 64: a failed build for the pick hides the bar at
    once (the tray's toast is the notice) and drops the status, so a late
    report for it does nothing; another series' failure is ignored;
    cancel_status (Clear) and Esc hide the status too; in the search face
    cancel_status leaves the bar open.

    Seeded defects: any series' failure hiding the bar, a status that
    survives the hide, Esc or cancel_status ignored in the status, or
    cancel_status closing the search face -> red.
    """
    from PySide6.QtCore import Qt

    palette = _picked(qtbot, _hit(media_id=7))
    palette.show_build_failed(8)
    assert palette.isVisible(), "another series' failure hid the status"
    palette.show_build_failed(7)
    assert (palette.isVisible(), palette._mode) == (False, "search")
    palette.show_built(7)  # late, for the dropped status
    assert (palette.isVisible(), palette._mode) == (False, "search")

    palette = _picked(qtbot, _hit(media_id=7))
    palette.cancel_status()
    assert (palette.isVisible(), palette._mode) == (False, "search")

    palette = _picked(qtbot, _hit(media_id=7))
    qtbot.keyClick(palette, Qt.Key_Escape)
    assert (palette.isVisible(), palette._mode) == (False, "search")

    palette.summon()
    palette.cancel_status()
    assert palette.isVisible(), "cancel_status closed the search face"
    # summon re-ran the field's search: let that worker finish before teardown
    qtbot.waitUntil(lambda: palette._search_threads == set(), timeout=2000)


def test_summon_during_the_status_returns_to_the_search_face_and_drops_the_status(qtbot):
    """L9 / Decision 38: a summon while the status shows brings the search
    face back (field focused, status hidden); the finishing build's reports
    then change nothing on the bar, and the active series shows as the chip
    (the tray's set_current_series).

    Seeded defects: summon keeping the status face, the old status still
    taking progress or built, or the chip kept hidden in the search face
    -> red.
    """
    palette = _picked(qtbot, _hit(media_id=7))
    palette.summon()
    assert palette._mode == "search"
    assert palette._search.isVisible() and not palette._status.isVisible()
    qtbot.waitUntil(palette._search.hasFocus, timeout=1000)

    palette.show_build_progress(7, 0.5)
    palette.show_built(7)
    assert (palette._mode, palette._fraction_target) == ("search", 0.0)
    assert palette.isVisible()

    palette.set_current_series(7, "Frieren: Beyond Journey's End (2023)")
    assert palette._chip.isVisible() and palette._chip.title() == "Frieren"
    # summon re-ran the field's search: let that worker finish before teardown
    qtbot.waitUntil(lambda: palette._search_threads == set(), timeout=2000)


def test_status_cover_follows_the_arriving_poster(qtbot):
    """Decision 39: the status shows the picked series' color tile until
    its poster arrives, then the poster (coverChanged reaches the status).

    Seeded defect: the status cover set once at the pick and never updated
    -> the color stays -> red.
    """
    from PySide6.QtGui import QColor

    palette = _picked(qtbot, _hit(media_id=7, color="#e4a15d"), reduce_motion=True)
    assert _render(palette).pixelColor(58, 24) == QColor("#e4a15d")
    palette._on_cover_ready(7, _png_bytes("#123456"))
    assert _render(palette).pixelColor(58, 24) == QColor("#123456")


def test_palette_cover_fetch_upgrades_tile_to_poster(qtbot, monkeypatch):
    """A hit with a cover URL paints its color tile at once, then the
    OFF-THREAD cover fetch upgrades the row's cover to the decoded poster.
    Exercises the full async cover path (start -> _CoverThread ->
    _on_cover_ready -> cover_for -> the dropdown's painted cover) without
    touching the network.

    The observable effect is the PAINTED cover: row 0's cover sits at
    (13.5, 8) in the dropdown, 32 x 44; its center turns from the hit's
    orange #e4a15d to the poster's solid blue #123456.

    Seeded-defect drills:
      - make _on_cover_ready return before recording the poster -> the
        cover stays orange -> waitUntil times out -> red.
      - never start fetches for shown rows -> same timeout -> red.
    """
    from PySide6.QtGui import QColor
    from yamabiko.overlay_app import series_palette as sp
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_RESULTS

    poster = _png_bytes("#123456", 12, 12)
    hit = SeriesHit(9756, "Madoka", "Puella Magi Madoka Magica", None,
                    2011, "TV", cover_image="https://img/madoka.jpg", color="#e4a15d")
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.search_series",
        lambda self, query, limit=8: [hit],
    )
    # Cover fetch off the network: the worker thread calls this instead of
    # httpx.
    monkeypatch.setattr(sp, "_fetch_cover_bytes", lambda url: poster)

    palette = SeriesPalette()
    qtbot.addWidget(palette)
    palette._search.setText("madoka")
    qtbot.waitUntil(lambda: _view(palette) == VIEW_RESULTS, timeout=4000)
    qtbot.waitUntil(
        lambda: _render(palette._dropdown).pixelColor(28, 30) == QColor("#123456"),
        timeout=4000,
    )


def test_cover_for_serves_the_session_poster_else_the_hit_color_else_neutral(qtbot):
    """The flyout's series cover comes from the palette's session cache (lead
    R3): a listed hit gives its AniList color, a decoded poster replaces it,
    an id the palette never listed (a series restored at boot) gets the
    neutral tile; coverChanged names the id each time its cover improves.

    Seeded defects: no color record for listed hits, no poster record in
    _on_cover_ready, or a missing coverChanged emit -> red."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QPixmap
    from yamabiko.overlay_app.series_palette import SeriesPalette

    source = QPixmap(10, 14)
    source.fill(QColor("#123456"))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    source.save(buf, "PNG")
    buf.close()

    palette = SeriesPalette()
    qtbot.addWidget(palette)
    changed: list[int] = []
    palette.coverChanged.connect(changed.append)
    palette._note_hit(_hit(media_id=7, color="#e4a15d"))   # the rows a view listed
    palette._note_hit(_hit(media_id=8, color=None))
    assert palette.cover_for(7) == QColor("#e4a15d")
    assert palette.cover_for(8) == QColor("#2b2b33")
    assert palette.cover_for(99) == QColor("#2b2b33")
    palette._note_hit(_hit(media_id=7, color="#e4a15d"))   # a repeat is no news

    palette._on_cover_ready(7, bytes(data))
    poster = palette.cover_for(7)
    assert isinstance(poster, QPixmap)
    assert (poster.width(), poster.height()) == (10, 14)
    assert poster.toImage().pixelColor(5, 7) == QColor("#123456")
    assert changed == [7, 7]


def test_active_series_poster_is_saved_beside_its_gazetteer_cache(qtbot):
    """Phase-5 L11: the active series' poster is written to cover_path so a
    restored series shows it at boot: at once when the poster is already
    decoded (the pick's build succeeded after its row loaded), or when it
    arrives later. A poster of a series that is not active is not written.

    Seeded defects: no save in set_current_series -> cover 7 missing -> red;
    no save on arrival -> cover 9 missing -> red; save every decoded poster
    -> cover 8 present -> red.
    """
    from yamabiko.gazetteer.gazetteer import cover_path
    from yamabiko.overlay_app.series_palette import SeriesPalette

    palette = SeriesPalette()
    qtbot.addWidget(palette)
    palette._on_cover_ready(7, _png_bytes())
    palette._on_cover_ready(8, _png_bytes())
    palette.set_current_series(7, "Seven (2020)")
    assert cover_path(7).is_file()
    assert not cover_path(8).is_file()

    palette.set_current_series(9, "Nine (2021)")
    assert not cover_path(9).is_file()
    palette._on_cover_ready(9, _png_bytes())
    assert cover_path(9).is_file()


def test_a_poster_read_from_its_file_is_not_rewritten_until_the_network_one_arrives(qtbot):
    """Decision 27: a restored series' poster loaded from cover_path is not
    written straight back when the series turns active; the network poster,
    once it arrives, replaces the file once.

    Seeded defects: drop ``or media_id in self._file_posters`` from
    _save_current_cover -> the file is rewritten at once -> red; drop
    ``self._file_posters.discard(media_id)`` from _on_cover_ready -> the
    network poster never reaches the file -> red.
    """
    import os

    from PySide6.QtGui import QColor, QImage, QPixmap
    from yamabiko.gazetteer.gazetteer import cover_path
    from yamabiko.overlay_app.series_palette import SeriesPalette, save_cover_file

    saved = QPixmap(10, 14)
    saved.fill(QColor("#123456"))
    assert save_cover_file(saved, cover_path(7))  # what an earlier session saved
    os.utime(cover_path(7), ns=(10**18, 10**18))
    palette = SeriesPalette()
    qtbot.addWidget(palette)
    assert isinstance(palette.cover_for(7), QPixmap)

    palette.set_current_series(7, "Seven (2020)")
    assert cover_path(7).stat().st_mtime_ns == 10**18, "the poster read from the file was rewritten"

    palette._on_cover_ready(7, _png_bytes("#654321"))
    assert QImage(str(cover_path(7))).pixelColor(5, 7) == QColor("#654321")


# --- release_threads: detach in-flight workers at quit --------------------
#
# Quit while an AniList search (15 s httpx timeout) or a cover fetch (8 s)
# is in flight destroys a parented running QThread -> Qt 6 qFatal abort.
# release_threads detaches the still-running workers into thread_keeper
# (the repo's established no-prune registry, see ui/thread_keeper.py) so
# widget teardown can't take a running OS thread down with it.


def test_release_threads_detaches_running_cover_worker(qtbot, monkeypatch):
    """release_threads() strips a still-running cover worker off the widget
    and parks it in thread_keeper, so quit-time widget teardown can't
    destroy a QThread whose OS thread is still inside the (up to 8 s) httpx
    cover fetch -- Qt 6 qFatal-aborts the process on that.

    Invokes the real _CoverThread (blocked on a threading.Event in place of
    the network) and asserts the observable detach: no widget parent, gone
    from _cover_threads, present-and-running in the registry.

    Seeded-defect drill: make release_threads skip running workers
    (`if t.isRunning(): continue`) -> the worker stays parented and in
    _cover_threads and nothing lands in the registry -> all three asserts red.
    """
    import threading

    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    monkeypatch.setattr(
        "yamabiko.overlay_app.series_palette._fetch_cover_bytes",
        lambda url, **kw: (gate.wait(10), None)[1],
    )
    palette = SeriesPalette()
    qtbot.addWidget(palette)
    hit = _hit(cover_image="http://x/c.png")
    palette._start_cover_fetch(hit)
    (thread,) = palette._cover_threads
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    palette.release_threads()

    try:
        assert thread.parent() is None, "cover worker was not detached from the widget"
        assert thread not in palette._cover_threads, (
            "detached cover worker still tracked in _cover_threads"
        )
        assert thread_keeper.any_running(), (
            "detached cover worker was not parked in the thread_keeper registry"
        )
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)  # charter: no cross-test registry leak


def test_release_threads_detaches_running_search_worker(qtbot, monkeypatch):
    """release_threads() detaches a still-running AniList SEARCH worker too
    (up to 15 s in httpx), not only cover workers -- the search-thread branch
    of the detach loop has its own quit-time qFatal exposure.

    Seeded-defect drill: drop `list(self._search_threads)` from the
    release_threads loop (detach only cover workers) -> the search worker
    stays parented and in _search_threads -> red.
    """
    import threading

    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.search_series",
        lambda self, query, limit=8: (gate.wait(10), [])[1],
    )
    palette = SeriesPalette()
    qtbot.addWidget(palette)
    # Set the query WITHOUT arming the 250 ms debounce (which would spawn a
    # second worker mid-wait and break the single-thread unpacking); start
    # the search worker directly instead.
    palette._search.blockSignals(True)
    palette._search.setText("madoka")
    palette._search.blockSignals(False)
    palette._run_search()
    (thread,) = palette._search_threads
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    palette.release_threads()

    try:
        assert thread.parent() is None, "search worker was not detached from the widget"
        assert thread not in palette._search_threads, (
            "detached search worker still tracked in _search_threads"
        )
        assert thread_keeper.any_running(), (
            "detached search worker was not parked in the thread_keeper registry"
        )
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)


# --- "Currently watching" (AniList Component 4; phase 5 L4, Decision 36) ----
#
# When gazetteer.anilist_username is configured, every summon kicks a
# background fetch_watching (same QThread discipline as search/cover
# workers); with the field empty the dropdown shows the "Currently watching"
# section and the six newest rows, the first lit. A pick flows through the
# SAME _pick path as a search pick. A failure shows the offline view (or the
# private-list line), never a traceback. Results are cached in-process for
# 15 minutes per username (injectable clock). The live fetch_watching seam is
# covered at the real tier
# (test_real_fetch_watching_public_user_returns_hits_without_raising).


def test_palette_open_shows_currently_watching_for_the_configured_user(qtbot, monkeypatch):
    """Summoning the bar with a configured username fetches the watch list
    OFF the GUI thread (fake client, no network) and shows the watching
    view: the user's rows in order, the first lit, the section's right label
    "<username> on AniList", and the window grown to fit (the airing row's
    "Episode N next" is the meta line's, test_series_palette_parts.py).

    Seeded-defect drills: never derive the view on watchingReady ->
    waitUntil times out -> red; drop the who label -> red; the window not
    following the view -> the height wait times out -> red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_WATCHING, dropdown_height

    watching = [
        _hit(media_id=101, english="Chainsaw Man", season_year=2026,
             next_airing_episode=8, color="#e4a15d"),
        _hit(media_id=102, english="Finished Show", season_year=2025,
             next_airing_episode=None),
    ]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: list(watching),
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()

    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)
    model = palette._dropdown.model()
    assert _ids(palette) == [101, 102]
    assert model.lit == 0
    assert model.who == "Watcher on AniList"
    qtbot.waitUntil(lambda: palette.height() == 48 + dropdown_height(model), timeout=2000)


def test_typing_swaps_currently_watching_for_results_and_clearing_brings_it_back(
    qtbot, monkeypatch,
):
    """One view at a time (L4): while the first answer for the typed query
    is out, Currently watching stays (no loading row); the results replace
    it; clearing the field brings the watching rows back at once, without a
    refetch.

    Seeded defects: clear the view while waiting -> red; mix the watch rows
    into the results -> red; lose the watch rows when results land, or
    wait for the debounce on an emptied field -> the last assert red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_RESULTS, VIEW_WATCHING

    watching = [_hit(media_id=101, english="Watched Show", season_year=2026)]
    search = [_hit(media_id=201, english="Search Hit", season_year=2012)]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: list(watching),
    )
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.search_series",
        lambda self, query, limit=8: list(search),
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)

    palette._search.setText("madoka")
    assert _view(palette) == VIEW_WATCHING, "the shown view must stay until results settle"
    qtbot.waitUntil(lambda: _view(palette) == VIEW_RESULTS, timeout=4000)
    assert _ids(palette) == [201], "the results view must hold only the search hits"

    palette._search.setText("   ")  # the emptied field
    assert (_view(palette), _ids(palette)) == (VIEW_WATCHING, [101]), (
        "clearing the field did not bring Currently watching back at once"
    )


def test_watching_tile_pick_emits_series_chosen_like_a_search_pick(
    qtbot, monkeypatch,
):
    """Enter on the lit watch row flows through the SAME pick as a search
    pick: series_chosen carries ``(media_id, format_hit(hit), native)`` --
    the search label, not the meta line -- and the bar shows its status.

    Seeded-defect drills: emit display text instead of ``format_hit(hit)``
    -> the label arg differs -> red; light no row in the watching view ->
    Enter no-ops, waitSignal times out -> red; skip the status face ->
    _mode stays "search" -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_WATCHING

    watching = [
        _hit(media_id=101, english="Chainsaw Man", season_year=2026,
             next_airing_episode=8),
    ]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: list(watching),
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)

    with qtbot.waitSignal(palette.series_chosen, timeout=2000) as blocker:
        qtbot.keyClick(palette._search, Qt.Key_Return)  # the lit first row
    assert blocker.args == [101, "Chainsaw Man (2026)", "ネイティブ"], (
        f"series_chosen carried {blocker.args!r}; a watch-row pick must "
        f"emit the same (media_id, format_hit label, native) as a search pick"
    )
    assert palette._mode == "status", "a watch-row pick must show its status"


def test_clicking_the_section_header_emits_nothing_and_keeps_the_bar_open(
    qtbot, monkeypatch,
):
    """Only rows pick (Decision 37): a click on the "Currently watching"
    section above them must NOT emit ``series_chosen`` for the lit row the
    user never clicked, and must NOT hide the bar.

    Seeded defect: pick the lit row on any click (ignore where it landed)
    -> the header click emits row 0's pick and hides the bar -> red.
    """
    from PySide6.QtCore import QPoint, Qt
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_WATCHING

    watching = [_hit(media_id=101, english="Chainsaw Man", season_year=2026)]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: list(watching),
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)

    with qtbot.assertNotEmitted(palette.series_chosen, wait=300):
        qtbot.mouseClick(palette._dropdown, Qt.LeftButton, pos=QPoint(100, 16))
    assert palette.isVisible(), (
        "clicking the section header hid the bar (a pick was made)"
    )


def test_clicking_watch_tile_emits_series_chosen_for_the_clicked_row(
    qtbot, monkeypatch,
):
    """The positive half of the mouse path: a click on a watch row emits
    ``series_chosen`` with THAT row's ``(media_id, format_hit)`` -- not the
    lit row's -- and turns the bar into its status, as the keyboard pick.

    Seeded defects: drop the rowClicked connection -> the click emits
    nothing -> waitSignal times out -> red; pick the lit row instead of the
    clicked one -> media id 101 -> red.
    """
    from PySide6.QtCore import Qt
    from yamabiko.overlay_app.series_palette import SeriesPalette

    watching = [
        _hit(media_id=101, english="Chainsaw Man", season_year=2026),
        _hit(media_id=102, english="Finished Show", season_year=2025),
    ]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: list(watching),
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _ids(palette) == [101, 102], timeout=4000)

    row = palette._dropdown.row_rect(1).center()
    with qtbot.waitSignal(palette.series_chosen, timeout=2000) as blocker:
        qtbot.mouseClick(palette._dropdown, Qt.LeftButton, pos=row)
    assert blocker.args == [102, "Finished Show (2025)", "ネイティブ"], (
        f"series_chosen carried {blocker.args!r}; a row click must emit "
        f"the CLICKED row's (media_id, format_hit label, native)"
    )
    assert palette._mode == "status", "a mouse pick must show its status"


def test_watching_cache_skips_refetch_within_ttl_and_refetches_after_expiry(
    qtbot, monkeypatch,
):
    """The in-process watch-list cache (keyed by username, 15 min TTL,
    injectable clock): a re-open inside the TTL starts NO new fetch
    worker; a re-open after expiry fetches again.

    Seeded-defect drills: drop the TTL check (fetch every open) -> the
    no-new-worker assert on the second open -> red; make the cache never
    expire (drop the ``< _WATCHING_TTL_S`` comparison) -> the third open
    never refetches -> waitUntil times out -> red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_WATCHING

    clock = [0.0]
    calls: list[str] = []

    def _fake_fetch(self, username):
        calls.append(username)
        return [_hit(media_id=101, english="Watched Show")]

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _fake_fetch,
    )

    palette = SeriesPalette(
        username_provider=lambda: "Watcher", now=lambda: clock[0],
    )
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=4000)
    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)
    # The finished-thread discard is a queued delivery that can land
    # after the render waitUntil above returns; wait for it here so the
    # no-new-worker assert after the second open cannot race the FIRST
    # worker's teardown on a slow runner.
    qtbot.waitUntil(lambda: palette._watching_threads == set(), timeout=4000)
    palette.hide()

    clock[0] = 800.0  # inside the 900 s TTL
    palette.summon()
    assert palette._watching_threads == set(), (
        "a re-open inside the TTL kicked a fetch worker; the cache was "
        "not consulted"
    )
    qtbot.wait(200)
    assert calls == ["Watcher"], f"cache hit still refetched: {calls}"
    palette.hide()

    clock[0] = 1000.0  # past the 900 s TTL (cache stamped at t=0)
    palette.summon()
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=4000)


@pytest.mark.parametrize(
    "exc, view",
    [
        pytest.param("private", "private", id="private-user"),
        pytest.param("unavailable", "offline", id="unavailable"),
        pytest.param("crash", "offline", id="unexpected-error"),
    ],
)
def test_watching_fetch_failure_shows_the_private_or_offline_view(
    qtbot, monkeypatch, exc, view,
):
    """A failed watch-list fetch degrades to a view, never a traceback (L4,
    Decision 36): AniListPrivateUser -> the private view ("Your AniList
    list is private."); AniListUnavailable AND any unexpected error -> the
    offline view with its list line ("Your AniList list is unavailable.").
    No rows either way.

    Seeded-defect drills: drop the except clauses from the watching
    worker's run() -> the exception escapes the worker, no view ever
    changes -> waitUntil times out -> red; treat private as offline (or the
    reverse) -> red; offline without its list line -> red.
    """
    from yamabiko.gazetteer.anilist import AniListPrivateUser, AniListUnavailable
    from yamabiko.overlay_app.series_palette import SeriesPalette

    exc_obj = {
        "private": AniListPrivateUser("Watcher"),
        "unavailable": AniListUnavailable("outage"),
        "crash": RuntimeError("boom"),
    }[exc]

    def _raise(self, username):
        raise exc_obj

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _raise,
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()

    qtbot.waitUntil(lambda: _view(palette) == view, timeout=4000)
    model = palette._dropdown.model()
    assert model.rows == ()
    assert model.list_unavailable is (view == "offline")


def test_watching_section_shows_the_first_six_rows_and_fetches_only_their_covers(
    qtbot, monkeypatch,
):
    """A long watch list must not bury the bar's lower screen or burst one
    cover fetch (a fresh TLS connect each) per entry: Currently watching
    shows the first six entries (L5, Decision 22; fetch_watching already
    orders them newest-update first), with no overflow line, and cover
    fetches start ONLY for those six.

    Seeded-defect drills: show every hit (drop the MAX_ROWS cap) -> 40 rows
    and 40 fetch starts -> red; fetch covers for the whole list -> red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_WATCHING

    watching = [
        _hit(media_id=1000 + i, english=f"Show {i}", season_year=2020,
             cover_image=f"https://img/{i}.jpg")
        for i in range(40)
    ]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: list(watching),
    )
    fetched: list[int] = []
    monkeypatch.setattr(
        SeriesPalette, "_start_cover_fetch",
        lambda self, hit: fetched.append(hit.media_id),
    )

    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette.summon()

    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)
    assert _ids(palette) == [1000 + i for i in range(6)], (
        "Currently watching must show the FIRST six hits, in order"
    )
    assert fetched == [1000 + i for i in range(6)], (
        f"cover fetches must start only for the shown rows; got "
        f"{len(fetched)} fetch starts"
    )


# --- Privacy: the palette's failure logs must be identity-free -------------
#
# The AniList username and raw search queries are user identity/content;
# these WARNING lines ship in diagnostics zips with no consent gate (unlike
# transcript content, which is consent-gated), so they must never carry the
# username value, the query text, or an exception repr that embeds either.


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("unavailable", id="anilist-unavailable"),
        pytest.param("private", id="private-user-fixed-message"),
        pytest.param("crash-with-username-in-message", id="repr-carries-username"),
    ],
)
def test_watching_fetch_failure_logs_neither_username_nor_exception_repr(
    qtbot, monkeypatch, caplog, failure,
):
    """After ANY watch-list fetch failure, the username appears in NO log
    record (message or traceback): an arbitrary exception's message can
    embed it (the crash row seeds one that does), so the generic handler
    logs the exception CLASS name, never the repr, and identifies the
    account only by length. AniListPrivateUser now carries a FIXED
    message (the username rides on ``exc.username``); its row pins that
    the private-user path stays identity-free end to end. The bar still
    shows its offline or private view (the log change must not eat the UI
    path).

    Seeded-defect drill: revert the handler to
    ``_log.warning("AniList watching fetch failed for %r", self._username,
    exc_info=True)`` -> the username lands in caplog.text on the
    unavailable/crash rows -> red (this IS the pre-fix line; confirmed
    red against it).
    """
    import logging

    from yamabiko.gazetteer.anilist import (
        AniListPrivateUser, AniListUnavailable,
    )
    from yamabiko.overlay_app.series_palette import SeriesPalette

    username = "SecretWatcher99"
    exc = {
        "unavailable": AniListUnavailable("outage"),
        "private": AniListPrivateUser(username),
        "crash-with-username-in-message": RuntimeError(
            f"boom while fetching {username}"),
    }[failure]

    def _raise(self, u):
        raise exc

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _raise,
    )
    palette = SeriesPalette(username_provider=lambda: username)
    qtbot.addWidget(palette)
    with caplog.at_level(logging.DEBUG):
        palette.summon()
        qtbot.waitUntil(lambda: _view(palette) in ("offline", "private"), timeout=4000)
        qtbot.waitUntil(
            lambda: palette._watching_threads == set(), timeout=4000,
        )
    assert username not in caplog.text, (
        "the AniList username leaked into the log (and from there into "
        "diagnostics zips)"
    )
    if failure != "private":
        # The failure taxon must still be diagnosable: class name, no repr.
        assert any(
            type(exc).__name__ in r.getMessage() for r in caplog.records
        ), "the failure log no longer names the exception class"


def test_search_failure_logs_no_query_text(qtbot, monkeypatch, caplog):
    """A failed AniList search logs the failure WITHOUT the raw query --
    the query is user-typed content (often a title the user is watching)
    and this log ships in diagnostics zips with no consent gate.

    Seeded-defect drill: revert the worker's handler to
    ``_log.warning("AniList search failed for %r", self._query, ...)`` ->
    the query text lands in caplog.text -> red (the pre-fix line;
    confirmed red against it).
    """
    import logging

    from yamabiko.gazetteer.anilist import AniListUnavailable
    from yamabiko.overlay_app.series_palette import SeriesPalette

    query = "very-secret-show-title"

    def _raise(self, q, limit=8):
        raise AniListUnavailable("outage")

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.search_series", _raise,
    )
    palette = SeriesPalette()
    qtbot.addWidget(palette)
    with caplog.at_level(logging.DEBUG):
        palette._search.setText(query)
        qtbot.waitUntil(
            lambda: any(
                "AniList search failed" in r.getMessage()
                for r in caplog.records
            ),
            timeout=4000,
        )
        qtbot.waitUntil(lambda: palette._search_threads == set(), timeout=4000)
    assert query not in caplog.text, (
        "the raw search query leaked into the log (and from there into "
        "diagnostics zips)"
    )


def test_palette_without_username_renders_no_section_and_never_fetches(
    qtbot, monkeypatch,
):
    """No configured username -> the summoned bar stays alone (no dropdown,
    48 px tall) and kicks NO fetch worker.

    Seeded defect: kick the fetch unconditionally in _refresh_watching
    (drop the empty-username early return) -> a worker spawns and the
    fake fetch records a call -> red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_BAR

    calls: list[str] = []
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: (calls.append(username), [])[1],
    )

    palette = SeriesPalette(username_provider=lambda: "")
    qtbot.addWidget(palette)
    palette.summon()

    assert palette._watching_threads == set(), (
        "an empty username still spawned a watch-list fetch worker"
    )
    qtbot.wait(200)
    assert (_view(palette), palette.height()) == (VIEW_BAR, 48), (
        "a watching section showed with no username"
    )
    assert calls == [], f"fetch_watching was called with no username: {calls}"


def test_username_removed_between_opens_clears_the_section(qtbot, monkeypatch):
    """The bar re-reads the username each summon: after the user clears it
    in Settings, the next summon drops the stale section (it must not keep
    showing another/former account's list).

    Seeded defect: drop the reset of the watch rows from the empty-username
    branch of _refresh_watching -> the stale rows survive the re-open ->
    red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_BAR, VIEW_WATCHING

    user = ["Watcher"]
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: [_hit(media_id=101, english="Watched Show")],
    )

    palette = SeriesPalette(username_provider=lambda: user[0])
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _view(palette) == VIEW_WATCHING, timeout=4000)
    palette.hide()

    user[0] = ""  # Settings cleared the username while the bar was closed
    palette.summon()
    assert _view(palette) == VIEW_BAR, (
        "re-opening with the username removed kept the stale watching section"
    )


def test_username_change_between_opens_refetches_and_renders_the_new_users_list(
    qtbot, monkeypatch,
):
    """The A-to-B account switch: after the configured username changes
    from UserA to UserB, the next summon drops A's rows immediately (never
    show another account's list while fetching), issues a FRESH fetch for
    B -- A's still-fresh cache entry must not be served, the cache is
    keyed per username -- and shows B's list when it lands.

    Seeded defect: look the watching cache up by the pre-update
    ``self._watching_user`` (A) before the change branch runs -> B's
    summon shows A's fresh cached rows and no fetch for B is ever issued
    -> the A-rows-gone assert goes red.
    """
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.series_palette_parts import VIEW_BAR

    user = ["UserA"]
    calls: list[str] = []
    lists = {
        "UserA": [_hit(media_id=101, english="A Show", season_year=2020)],
        "UserB": [_hit(media_id=202, english="B Show", season_year=2021)],
    }

    def _fake_fetch(self, username):
        calls.append(username)
        return list(lists[username])

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _fake_fetch,
    )

    palette = SeriesPalette(username_provider=lambda: user[0])
    qtbot.addWidget(palette)
    palette.summon()
    qtbot.waitUntil(lambda: _ids(palette) == [101], timeout=4000)
    palette.hide()

    user[0] = "UserB"  # Settings switched accounts while the bar was closed
    palette.summon()
    # B's fetch result is a queued delivery, so nothing can have arrived
    # between summon() returning and this assert: the change branch must
    # already have dropped A's stale rows.
    assert _view(palette) == VIEW_BAR, (
        "A's rows survived the re-open as B (stale account list shown "
        "while B's fetch is in flight)"
    )
    qtbot.waitUntil(lambda: calls == ["UserA", "UserB"], timeout=4000)
    qtbot.waitUntil(lambda: _ids(palette) == [202], timeout=4000)


def test_stale_inflight_fetch_for_old_user_never_replaces_new_users_section(
    qtbot, monkeypatch,
):
    """The stale-result guard in _on_watching_ready: UserA's fetch is
    still in flight when the configured username changes to UserB and B's
    fetch delivers first; when A's SLOW result finally lands (through the
    same queued-signal path the real worker uses) it must be dropped --
    the shown rows stay B's one-row list, not A's two-row one.

    Seeded defect: delete the ``if username != self._watching_user:
    return`` stale guard from _on_watching_ready -> A's late result
    replaces B's rows ([101, 103]) -> red.
    """
    import threading

    from yamabiko.overlay_app.series_palette import SeriesPalette

    gate_a = threading.Event()
    a_started = threading.Event()
    user = ["UserA"]
    a_hits = [
        _hit(media_id=101, english="A Show One", season_year=2020),
        _hit(media_id=103, english="A Show Two", season_year=2019),
    ]
    b_hits = [_hit(media_id=202, english="B Show", season_year=2021)]

    def _fake_fetch(self, username):
        if username == "UserA":
            a_started.set()
            gate_a.wait(10)  # self-releasing: a failed assert can't wedge it
            return list(a_hits)
        return list(b_hits)

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _fake_fetch,
    )

    palette = SeriesPalette(username_provider=lambda: user[0])
    qtbot.addWidget(palette)
    try:
        palette.summon()
        qtbot.waitUntil(a_started.is_set, timeout=4000)  # A's fetch in flight
        palette.hide()

        user[0] = "UserB"  # account switched while A's fetch is still running
        palette.summon()
        qtbot.waitUntil(lambda: _ids(palette) == [202], timeout=4000)

        gate_a.set()  # release the STALE A result so it lands LAST
        # A's worker posts watchingReady, then finished (which discards it
        # from _watching_threads) from the same thread; queued deliveries
        # keep that order, so an empty thread set proves the stale result
        # was already delivered -- and must have been dropped.
        qtbot.waitUntil(lambda: palette._watching_threads == set(), timeout=5000)
        assert _ids(palette) == [202], (
            "the bar shows A's list after B's summon: the stale-result "
            "guard did not drop the late fetch"
        )
    finally:
        gate_a.set()
        qtbot.waitUntil(lambda: palette._watching_threads == set(), timeout=5000)


def test_default_username_provider_reads_persisted_config_key(
    qtbot, monkeypatch, tmp_path,
):
    """The DEFAULT provider closes the config consumer chain (hard rule 4):
    a gazetteer.anilist_username override persisted by the settings window
    reaches fetch_watching on the next palette open, through the merged
    config (isolated at tmp_path via YAMABIKO_APPDATA).

    Seeded defects (one line each): delete ``anilist_username: ""`` from
    default_config.yaml -> the validator strips the override, the provider
    reads "" and no fetch happens -> red; make the provider read a
    different key path -> no fetch / wrong arg -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    from yamabiko.config_writer import write_override
    from yamabiko.overlay_app.series_palette import SeriesPalette

    write_override(["gazetteer", "anilist_username"], "Mand3012")

    calls: list[str] = []

    def _fake_fetch(self, username):
        calls.append(username)
        return [_hit(media_id=101, english="Watched Show")]

    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching", _fake_fetch,
    )

    palette = SeriesPalette()  # DEFAULT provider: reads the merged config
    qtbot.addWidget(palette)
    palette.summon()

    qtbot.waitUntil(lambda: calls == ["Mand3012"], timeout=4000)
    qtbot.waitUntil(lambda: _view(palette) == "watching", timeout=4000)


def test_release_threads_detaches_running_watching_worker(qtbot, monkeypatch):
    """release_threads() detaches a still-running WATCHING fetch worker
    (up to 15 s in httpx) exactly like the search/cover workers -- the
    watching branch of the detach loop has its own quit-time qFatal
    exposure (quit-time widget teardown of a running QThread aborts the
    process on Qt 6).

    Seeded-defect drill: drop ``list(self._watching_threads)`` from the
    release_threads loop -> the watching worker stays parented and in
    _watching_threads and nothing lands in the registry -> red.
    """
    import threading

    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.ui import thread_keeper

    gate = threading.Event()
    monkeypatch.setattr(
        "yamabiko.gazetteer.anilist.AniListClient.fetch_watching",
        lambda self, username: (gate.wait(10), [])[1],
    )
    palette = SeriesPalette(username_provider=lambda: "Watcher")
    qtbot.addWidget(palette)
    palette._refresh_watching()
    (thread,) = palette._watching_threads
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    palette.release_threads()

    try:
        assert thread.parent() is None, (
            "watching worker was not detached from the widget"
        )
        assert thread not in palette._watching_threads, (
            "detached watching worker still tracked in _watching_threads"
        )
        assert thread_keeper.any_running(), (
            "detached watching worker was not parked in the thread_keeper registry"
        )
    finally:
        gate.set()
        qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)
        thread_keeper.discard(thread)


# --- Theme: the bar paints the shared Mist/Night tokens (phase 5 L15) --------
#
# The bar draws its own surfaces (no style sheet colors): the window fills
# with the raised token and the field's text is the ink token. These tests
# pin the default theme, the constructor's theme and the live re-skin path
# that the tray's theme flip drives.


def test_palette_default_theme_is_night(qtbot):
    """A bar built with no theme_palette paints the shared NIGHT (dark)
    chrome: the raised tone behind the seal and ink field text.

    Seeded defect: default to MIST or a hard-coded color -> red.
    """
    from PySide6.QtGui import QColor, QPalette
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import NIGHT

    palette = SeriesPalette()
    qtbot.addWidget(palette)
    assert _render(palette).pixelColor(5, 5) == QColor(NIGHT["raised"])
    assert palette._search.palette().color(QPalette.ColorRole.Text) == QColor(NIGHT["ink"])


def test_palette_constructed_with_mist_uses_light_tokens(qtbot):
    """Constructing with the MIST (light) palette paints the light tokens,
    proving the constructor keyword drives the look rather than a fixed
    theme.

    Seeded defect: ignore theme_palette and always paint NIGHT -> red.
    """
    from PySide6.QtGui import QColor, QPalette
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import MIST

    palette = SeriesPalette(theme_palette=MIST)
    qtbot.addWidget(palette)
    assert _render(palette).pixelColor(5, 5) == QColor(MIST["raised"])
    assert palette._search.palette().color(QPalette.ColorRole.Text) == QColor(MIST["ink"])


def test_apply_palette_swaps_the_theme(qtbot):
    """apply_palette re-skins the live bar in place: after a dark->light
    swap the bar paints MIST's raised tone and the field MIST's ink.

    Seeded-defect drill: make apply_palette a no-op, or skip the field ->
    the NIGHT tones linger -> red.
    """
    from PySide6.QtGui import QColor, QPalette
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import MIST, NIGHT

    palette = SeriesPalette(theme_palette=NIGHT)
    qtbot.addWidget(palette)
    assert _render(palette).pixelColor(5, 5) == QColor(NIGHT["raised"])

    palette.apply_palette(MIST)
    assert _render(palette).pixelColor(5, 5) == QColor(MIST["raised"])
    assert palette._search.palette().color(QPalette.ColorRole.Text) == QColor(MIST["ink"])


def test_the_fields_menu_and_the_chips_tooltip_wear_the_bars_theme(qtbot):
    """Visual check, observation 3 / L15: the field's right-click menu and
    the chip's tooltip take the bar's theme the way the tray's menus
    (menu_qss) and the flyout's tooltips (the QToolTip rule) do: raised
    ground and ink text, in Mist and again after a live switch to Night,
    not the platform style's white menu and yellow tooltip.

    Seeded defects: the field's stock menu (no menu_qss) -> red; no tooltip
    rule on the chip at construction -> the Mist tooltip red; apply_palette
    not re-styling the chip -> the Night tooltip red.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QContextMenuEvent, QPalette
    from PySide6.QtWidgets import QApplication, QToolTip
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import MIST, NIGHT

    palette = SeriesPalette(username_provider=lambda: "", theme_palette=MIST)
    qtbot.addWidget(palette)
    palette.summon()
    palette.set_current_series(5, "Frieren (2023)")
    field, chip = palette._search, palette._chip

    def menu_colors() -> tuple[str, str]:
        QApplication.sendEvent(field, QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, QPoint(20, 12), field.mapToGlobal(QPoint(20, 12))))
        menu = QApplication.activePopupWidget()
        assert menu is not None and menu.parentWidget() is field, "the field's menu did not open"
        ground = menu.grab().toImage().pixelColor(menu.width() // 2, 2)
        ink = menu.palette().color(QPalette.ColorRole.WindowText)
        menu.close()
        return ground.name(), ink.name()

    def tips() -> list:
        return [w for w in QApplication.topLevelWidgets() if w.inherits("QTipLabel")]

    def tooltip_colors() -> tuple[str, str]:
        QToolTip.showText(chip.mapToGlobal(QPoint(4, 4)), chip.toolTip(), chip)
        tip = next(w for w in tips() if w.isVisible())
        ground = tip.grab().toImage().pixelColor(2, 2)
        ink = tip.palette().color(QPalette.ColorRole.WindowText)
        QToolTip.hideText()
        qtbot.waitUntil(lambda: not tips(), timeout=2000)  # the next one is styled anew
        return ground.name(), ink.name()

    for theme in (MIST, NIGHT):  # as constructed, then after a live switch
        if theme is NIGHT:
            palette.apply_palette(NIGHT)
        expected = (QColor(theme["raised"]).name(), QColor(theme["ink"]).name())
        assert menu_colors() == expected, "the field's menu is not in the bar's theme"
        assert tooltip_colors() == expected, "the chip's tooltip is not in the bar's theme"


def test_the_fields_menu_paints_no_icons_like_the_tray_menu(qtbot, tmp_path):
    """Visual check 2, item 5: QLineEdit's standard menu takes the theme's
    icons (Undo, Cut, Copy, Paste, …), which Windows draws white (near
    invisible on the Mist menu) in an icon column the tray menu does not
    have. The field's menu drops them, as plain as the tray menu. An icon
    theme of solid red icons stands in for the platform's.

    Seeded defect: keep the standard actions' icons -> red icon pixels in the
    menu -> red.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QContextMenuEvent, QIcon, QPixmap
    from PySide6.QtWidgets import QApplication
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import MIST

    icons = tmp_path / "yamabikotest" / "16x16"
    icons.mkdir(parents=True)
    (icons.parent / "index.theme").write_text(
        "[Icon Theme]\nName=yamabikotest\nDirectories=16x16\n\n[16x16]\nSize=16\nType=Fixed\n")
    red = QPixmap(16, 16)
    red.fill(QColor("#ff0000"))
    for name in ("edit-undo", "edit-redo", "edit-cut", "edit-copy", "edit-paste",
                 "edit-delete", "edit-select-all"):
        red.save(str(icons / f"{name}.png"))
    search_paths, theme_name = QIcon.themeSearchPaths(), QIcon.themeName()
    QIcon.setThemeSearchPaths([str(tmp_path)])
    QIcon.setThemeName("yamabikotest")
    try:
        palette = SeriesPalette(_QuietClient(), username_provider=lambda: "", theme_palette=MIST)
        qtbot.addWidget(palette)
        palette.summon()
        _type_silently(palette, "frieren")
        field = palette._search
        field.selectAll()  # Cut, Copy, Delete and Select All enabled: their icons in full color
        QApplication.sendEvent(field, QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, QPoint(20, 12), field.mapToGlobal(QPoint(20, 12))))
        menu = QApplication.activePopupWidget()
        assert menu is not None and menu.parentWidget() is field, "the field's menu did not open"
        image = menu.grab().toImage()
        menu.close()
        reds = sum(1 for x in range(image.width()) for y in range(image.height())
                   if (c := image.pixelColor(x, y)).red() > 200 and c.green() < 60 and c.blue() < 60)
        assert reds == 0, f"the field's menu painted icons ({reds} icon pixels)"
    finally:
        QIcon.setThemeSearchPaths(search_paths)
        QIcon.setThemeName(theme_name)


def _theme_fake_client():
    """A MagicMock PipelineClient stand-in for the Tray live-flip tests: its
    Qt-signal attributes expose ``.connect`` and ``send_command`` returns True.
    The tray theme-flip path never emits or sends, so this suffices -- the
    subject under test is the bar's painted look, not the client."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.send_command.return_value = True
    return client


def test_tray_apply_palette_reskins_wired_series_palette(qtbot):
    """A SeriesPalette handed to the Tray joins the live theme-flip flow:
    tray.apply_palette(MIST) re-skins the wired bar in place (it paints the
    MIST raised tone, the NIGHT one gone) -- matching how the open Settings
    window is re-skinned on an OS color-scheme follow.

    Seeded defect: drop the series-palette forward in Tray.apply_palette ->
    the bar keeps its NIGHT look -> red.
    """
    from PySide6.QtGui import QColor
    from yamabiko.overlay_app.series_palette import SeriesPalette
    from yamabiko.overlay_app.theme import MIST, NIGHT
    from yamabiko.overlay_app.tray import Tray

    client = _theme_fake_client()
    palette = SeriesPalette(theme_palette=NIGHT)
    qtbot.addWidget(palette)
    tray = Tray(
        client, on_quit=lambda: None, palette=NIGHT,
        on_change_series=palette.summon, series_palette=palette,
    )
    assert _render(palette).pixelColor(5, 5) == QColor(NIGHT["raised"])

    tray.apply_palette(MIST)
    assert _render(palette).pixelColor(5, 5) == QColor(MIST["raised"]), (
        "tray.apply_palette did not re-skin the wired series bar"
    )


def test_tray_apply_palette_without_series_palette_does_not_crash(qtbot):
    """Gazetteer disabled -> the Tray is built with no series palette
    (series_palette defaults None). A live theme flip must still re-skin the
    menu and must NOT crash on the absent palette.

    Seeded defect: forward to self._series_palette.apply_palette WITHOUT the
    None guard -> AttributeError on None -> red.
    """
    from yamabiko.overlay_app.theme import MIST, NIGHT
    from yamabiko.overlay_app.tray import Tray

    client = _theme_fake_client()
    tray = Tray(client, on_quit=lambda: None, palette=NIGHT)  # no series palette
    tray.apply_palette(MIST)  # must not raise
    assert MIST["raised"] in tray._menu.styleSheet(), (
        "apply_palette did not repaint the menu when no series palette is wired"
    )
