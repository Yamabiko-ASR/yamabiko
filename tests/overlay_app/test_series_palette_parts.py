"""The series bar's painted parts (plan phase 5, Task 4), offscreen.

Pure text helpers (the short title, the meta line, the copy and the key
hints), the dropdown's height per view, and rendered pixels of a lit row, its
cover and the offline view's Try again. The bar window that drives these
parts is tested in test_series_palette.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QRegion
from PySide6.QtWidgets import QWidget

from yamabiko.gazetteer.anilist import SeriesHit
from yamabiko.overlay_app import series_palette_parts as parts
from yamabiko.overlay_app.controls import ui_font
from yamabiko.overlay_app.fonts import register_bundled_fonts
from yamabiko.overlay_app.series_palette_parts import MetaPart
from yamabiko.overlay_app.theme import MIST


def _hit(**kw) -> SeriesHit:
    base = dict(media_id=1, romaji="Sousou no Frieren",
                english="Frieren: Beyond Journey's End", native="葬送のフリーレン",
                season_year=2023, fmt="TV")
    base.update(kw)
    return SeriesHit(**base)


def _render(widget: QWidget, dpr: int = 1) -> QImage:
    image = QImage(widget.size() * dpr, QImage.Format_ARGB32)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


@pytest.mark.parametrize("title, short", [
    ("The Apothecary Diaries Season 3", "Apothecary Diaries S3"),
    ("Frieren: Beyond Journey's End", "Frieren"),
    ("Frieren: Beyond Journey's End (2023)", "Frieren"),
    ("Kusuriya no Hitorigoto Season 2 (2025)", "Kusuriya no Hitorigoto S2"),
    ("Frieren: Beyond Journey's End Season 2 (2026)", "Frieren S2"),
    ("Sousou no Frieren 3rd Season", "Sousou no Frieren S3"),
])
def test_short_title_drops_the_year_the_the_and_the_subtitle_and_shortens_seasons(title, short):
    """L7 / Decision 40: the chip's and the status' short title, from the
    three ruling examples (rows 1, 2, 5), a restored label with its
    " (year)" (rows 3, 4) and the "Nth Season" form (row 6). The order
    matters: the year goes first so the season marker is trailing, and the
    season is set aside before the colon cut so it survives it.

    Seeded defects: keep the " (YYYY)" -> red on rows 3-5; keep a leading
    "The " -> red on row 1; keep the subtitle -> red on rows 2, 3 and 5;
    look for the season before dropping the year, or cut at the colon
    before setting the season aside -> red on row 5; no " Season N" /
    " Nth Season" -> " SN" -> red on rows 1, 4, 5 and 6."""
    assert parts.short_title(title) == short


def test_meta_parts_list_native_year_format_then_airing_or_the_episode_count():
    """L6 / Decision 41: native title (Japanese face), year, format name, then
    "Episode N next" while airing (ink), else "N episodes" / "1 episode";
    missing parts and non-anime formats are skipped.

    Seeded defects: episodes shown although airing, "1 episodes", the raw
    "TV_SHORT", or an empty native title kept as a part -> red."""
    assert parts.meta_parts(_hit(next_airing_episode=5, episodes=24)) == [
        MetaPart("葬送のフリーレン", "jp"), MetaPart("2023", "plain"),
        MetaPart("TV", "plain"), MetaPart("Episode 5 next", "air"),
    ]
    assert parts.meta_parts(_hit(episodes=28))[-1] == MetaPart("28 episodes", "plain")
    assert parts.meta_parts(_hit(native=None, season_year=None, fmt="TV_SHORT", episodes=1)) == [
        MetaPart("TV short", "plain"), MetaPart("1 episode", "plain"),
    ]
    assert parts.meta_parts(_hit(native="", season_year=None, fmt="MANGA", episodes=None)) == []


def test_dropdown_height_stacks_each_view_to_the_mockup_heights():
    """Decision 29's totals with the 48 px bar: watching with four rows 328,
    six results 400, no matches 140, offline 140 / 211 with the list line,
    private 144, the bar alone 48.

    Seeded defects: a missing list padding, key line or split section, or a
    row height off by a pixel -> red."""
    four = tuple(_hit(media_id=i) for i in range(4))
    six = tuple(_hit(media_id=i) for i in range(6))
    Model = parts.DropdownModel
    totals = [
        parts.BAR_H + parts.dropdown_height(model) for model in (
            Model(),
            Model(view=parts.VIEW_WATCHING, rows=four, lit=0, who="mika on AniList"),
            Model(view=parts.VIEW_RESULTS, rows=six, lit=0),
            Model(view=parts.VIEW_NO_MATCHES, query="frierne"),
            Model(view=parts.VIEW_OFFLINE),
            Model(view=parts.VIEW_OFFLINE, list_unavailable=True),
            Model(view=parts.VIEW_PRIVATE, who="mika on AniList"),
        )
    ]
    assert totals == [48, 328, 400, 140, 140, 211, 144]


def test_lit_row_paints_the_accent_and_each_cover_paints_what_cover_for_gives(qtbot):
    """L6 / Decision 31: the lit row fills with the accent; a row's cover is
    whatever the bar's cover_for gives (here the AniList color of a poster
    that has not loaded), and the neutral tile when that is no color; the
    next row is not lit.

    Seeded defects: rows placed without the section above them, the lit row
    not filled, the cover not drawn from cover_for, or no neutral fallback
    -> red."""
    covers = {11: QColor("#d6d678")}
    dropdown = parts.Dropdown(MIST, lambda media_id: covers.get(media_id, QColor()))
    qtbot.addWidget(dropdown)
    dropdown.set_model(parts.DropdownModel(
        view=parts.VIEW_WATCHING, rows=(_hit(media_id=11), _hit(media_id=12)), lit=0,
        who="mika on AniList"))
    assert (dropdown.width(), dropdown.height()) == (638, 32 + 8 + 2 * 52 + 32)
    assert dropdown.row_rect(0) == QRect(4, 36, 630, 52)
    assert dropdown.row_rect(1) == QRect(4, 88, 630, 52)
    image = _render(dropdown)
    assert image.pixelColor(6, 62) == QColor(MIST["accent"])         # lit row, left of its cover
    assert image.pixelColor(28, 62) == QColor("#d6d678")             # its cover: the AniList color
    assert image.pixelColor(6, 114) == QColor(MIST["raised"])        # the next row is not lit
    assert image.pixelColor(28, 114) == QColor("#2b2b33")            # no color: the neutral tone


def test_row_content_sits_inside_the_mockups_row_border_on_whole_device_pixels(qtbot):
    """Decision 31 (lead ruling on the Task 4 visual check): the mockup's
    rows carry a transparent 1.5 px border, so a row's cover starts 9.5 px
    into it and its title 53.5 px; the cover lands on whole device pixels at
    any scale (crisp edges, never a blended column).

    Seeded defects: the old +8 cover or +52 text offset (red on the cover's
    edges or the title's first ink at 200 %, the screenshots' scale where
    9.5 px is exactly 19 device px), or the cover painted at its fractional
    position (red at 100 %: a column blended from the accent and the cover)."""
    register_bundled_fonts()
    lit, cover = QColor(MIST["accent"]), QColor("#d6d678")
    dropdown = parts.Dropdown(MIST, lambda media_id: cover)
    qtbot.addWidget(dropdown)
    dropdown.set_model(parts.DropdownModel(view=parts.VIEW_RESULTS, rows=(_hit(),), lit=0))
    row = dropdown.row_rect(0)
    image = _render(dropdown, dpr=2)
    mid = (row.y() + 26) * 2
    left = int((row.x() + 9.5) * 2)
    assert (image.pixelColor(left - 1, mid), image.pixelColor(left, mid)) == (lit, cover)
    assert (image.pixelColor(left + 63, mid), image.pixelColor(left + 64, mid)) == (cover, lit)
    title = parts.display_title(_hit())
    ink = (row.x() + 53.5 + QFontMetricsF(ui_font(14)).boundingRect(title).left()) * 2
    band = range((row.y() + 8) * 2, (row.y() + 26) * 2)
    first = next(x for x in range(left + 64, image.width())
                 if any(image.pixelColor(x, y) != lit for y in band))
    assert abs(first - ink) <= 1, (first, ink)
    flat = _render(dropdown)
    across = {flat.pixelColor(x, row.y() + 26).name() for x in range(row.x() + 2, row.x() + 50)}
    assert across == {lit.name(), cover.name()}


def test_offline_view_puts_try_again_at_the_right_and_its_click_retries(qtbot):
    """L4 / Decision 31: the offline view shows Try again, 30 px tall, its
    right edge at x 626, 11 px into the message; a click reports
    retryClicked; other views hide it.

    Seeded defects: the button never shown or placed from the left, its
    click not wired to retryClicked, or left visible in another view -> red."""
    dropdown = parts.Dropdown(MIST, lambda media_id: QColor())
    qtbot.addWidget(dropdown)
    dropdown.set_model(parts.DropdownModel(view=parts.VIEW_OFFLINE, list_unavailable=True))
    dropdown.show()
    button = dropdown.findChild(parts.TryAgainButton)
    assert button.isVisible()
    assert (button.x() + button.width(), button.y(), button.height()) == (626, 15, 30)
    assert button.width() >= 88
    with qtbot.waitSignal(dropdown.retryClicked, timeout=1000):
        qtbot.mouseClick(button, Qt.LeftButton)
    dropdown.set_model(parts.DropdownModel(view=parts.VIEW_NO_MATCHES, query="frierne"))
    assert not button.isVisible()


def test_a_press_is_not_picked_on_a_row_that_new_rows_put_under_it(qtbot):
    """L6: a click picks the row it pressed. A search answer that replaces
    the rows between the press and the release puts another series under
    the same index, so that release picks nothing.

    Seeded defect: set_model keeps the pressed index -> the release reports
    the new answer's row 0 -> red."""
    dropdown = parts.Dropdown(MIST, lambda media_id: QColor())
    qtbot.addWidget(dropdown)
    dropdown.set_model(parts.DropdownModel(view=parts.VIEW_RESULTS, rows=(_hit(media_id=1),), lit=0))
    dropdown.show()
    qtbot.waitExposed(dropdown)
    clicked: list[int] = []
    dropdown.rowClicked.connect(clicked.append)
    point = dropdown.row_rect(0).center()
    qtbot.mousePress(dropdown, Qt.LeftButton, pos=point)
    dropdown.set_model(parts.DropdownModel(view=parts.VIEW_RESULTS, rows=(_hit(media_id=2),), lit=0))
    qtbot.mouseRelease(dropdown, Qt.LeftButton, pos=point)
    assert clicked == [], "the release picked a row the press never touched"


def test_the_hover_tone_follows_a_resting_pointer_into_a_new_view(qtbot):
    """L6 / Decision 31: the row under the pointer takes the hover tone, also
    when the view changes under a pointer that does not move (results' rows
    start 32 px higher than Currently watching's, and no move event comes).

    Seeded defect: set_model keeps the old hover index -> results row 1
    takes the hover tone while the pointer is over row 2 -> red."""
    from PySide6.QtGui import QCursor

    hover, raised = QColor(MIST["hover"]), QColor(MIST["raised"])
    rows = tuple(_hit(media_id=i) for i in (1, 2, 3, 4))
    dropdown = parts.Dropdown(MIST, lambda media_id: QColor())
    qtbot.addWidget(dropdown)
    dropdown.set_model(parts.DropdownModel(
        view=parts.VIEW_WATCHING, rows=rows, lit=0, who="mika on AniList"))
    dropdown.show()
    qtbot.waitExposed(dropdown)
    saved = QCursor.pos()
    try:
        QCursor.setPos(dropdown.mapToGlobal(QPoint(100, 120)))  # watching row 1: y 88-140
        qtbot.waitUntil(lambda: _render(dropdown).pixelColor(6, 114) == hover, timeout=1000)

        dropdown.set_model(parts.DropdownModel(view=parts.VIEW_RESULTS, rows=rows, lit=0))
        image = _render(dropdown)  # results rows 1 and 2: y 56-108 and 108-160
        assert (image.pixelColor(6, 82), image.pixelColor(6, 134)) == (raised, hover)
    finally:
        QCursor.setPos(saved)


def test_state_copy_and_key_hints_read_exactly_as_the_spec():
    """Spec Surface 4, L4, L9 and Decision 31: the visible strings (curly
    quotes U+201C/U+201D, the ellipsis U+2026, the arrows U+2191/U+2193,
    ASCII apostrophes) and the key line of each view.

    Seeded defects: straight quotes, "..." for the ellipsis, a reworded line,
    or offline showing the rows' key line -> red."""
    assert parts.PLACEHOLDER == "Search for a series"
    assert parts.NO_MATCHES.format(query="frierne") == "No matches for \u201cfrierne\u201d."
    assert parts.NO_MATCHES_DETAIL == "Try the Japanese or romaji title."
    assert parts.OFFLINE == "Can't reach AniList right now."
    assert parts.TRY_AGAIN == "Try again"
    assert parts.LIST_UNAVAILABLE == "Your AniList list is unavailable."
    assert parts.LIST_PRIVATE == "Your AniList list is private."
    assert parts.LOADING.format(title="Frieren") == "Loading character names for Frieren\u2026"
    assert parts.NOW_WATCHING.format(title="Frieren") == "Now watching Frieren."
    assert {view: parts.key_hint(view) for view in (
        parts.VIEW_BAR, parts.VIEW_WATCHING, parts.VIEW_RESULTS, parts.VIEW_NO_MATCHES,
        parts.VIEW_OFFLINE, parts.VIEW_PRIVATE)} == {
        parts.VIEW_BAR: "",
        parts.VIEW_WATCHING: "\u2191 \u2193 move, Enter picks, Esc closes",
        parts.VIEW_RESULTS: "\u2191 \u2193 move, Enter picks, Esc closes",
        parts.VIEW_NO_MATCHES: "Esc closes",
        parts.VIEW_OFFLINE: "Enter tries again, Esc closes",
        parts.VIEW_PRIVATE: "Esc closes",
    }
