# tests/overlay_app/test_ruby_layout.py
"""Layout engine only; no Qt. Measurers are fakes: kanji chars 10 px,
kana chars 5 px, so expected pixel math is exact."""
from yamabiko.overlay_app.ruby_layout import layout_rows, row_width, seg_measures


def _mk(text):   # fake kanji measurer
    return 10 * len(text)


def _mn(text):   # fake kana measurer
    return 5 * len(text)


def test_seg_measures_is_the_single_width_definition():
    """seg_measures returns (kanji_w, kana_w, seg_w); seg_w is THE definition
    of a segment's painted width, and row_width must sum exactly those seg_w
    values — telop_text paints each row at the widths the wrapper decided
    with, so any drift between the painter's numbers and the wrapper's would
    offset every glyph.

    Seeded defect (one line): change seg_w's formula in seg_measures (drop the
    padding, or sum instead of max) without row_width following -> the exact
    tuples and the row_width consistency check go red.
    """
    ruby = {"text": "田中", "reading": "たなか"}    # kanji 20, kana 15
    plain = {"text": "さん"}                        # kanji 20, no reading
    assert seg_measures(ruby, _mk, _mn) == (20, 15, 24)
    assert seg_measures(plain, _mk, _mn) == (20, 0, 24)  # no reading -> kana_w 0
    assert seg_measures(ruby, _mk, _mn, seg_padding=10) == (20, 15, 30)
    assert row_width([ruby, plain], _mk, _mn) == (
        seg_measures(ruby, _mk, _mn)[2] + seg_measures(plain, _mk, _mn)[2]
    )


SEGS = [
    {"text": "田中", "reading": "たなか"},   # width max(20, 15)+4 = 24
    {"text": "さん"},                        # width 20+4 = 24
    {"text": "議員定数", "reading": "ぎいんていすう"},  # max(40,35)+4 = 44
]


def test_everything_fits_one_row():
    rows = layout_rows(SEGS, max_width_px=200, measure_kanji=_mk, measure_kana=_mn)
    assert rows == [SEGS]


def test_wraps_between_segments_never_inside():
    # 24 + 24 = 48 fits in 50; adding 44 would be 92 -> wraps.
    rows = layout_rows(SEGS, max_width_px=50, measure_kanji=_mk, measure_kana=_mn)
    # exact assertion (48 <= 50):
    assert rows == [SEGS[:2], [SEGS[2]]]


def test_char_break_keeps_reading_on_first_fragment():
    """A segment wider than the cap is character-broken; its reading
    rides the FIRST fragment only, exactly once, and no text is lost.

    Drives the _char_break path unconditionally: 12 kanji * 10 px =
    120 px > cap 60, so the pre-pass splits it. The reading "か"*4
    (20 px) must land on the first fragment and nowhere else.

    Seeded defect that must go red: attach the reading to the LAST
    fragment (out[-1]) or drop it in _char_break.
    """
    seg = {"text": "あ" * 12, "reading": "か" * 4}
    rows = layout_rows([seg], max_width_px=60, measure_kanji=_mk, measure_kana=_mn)

    # Fragments pack one-per-row at cap 60 (each 5-kanji frag = 54 px).
    assert rows == [
        [{"text": "あ" * 5, "reading": "か" * 4}],
        [{"text": "あ" * 5}],
        [{"text": "あ" * 2}],
    ]
    # reading present exactly once, on the first emitted fragment
    readings = [s.get("reading") for row in rows for s in row]
    assert readings == ["か" * 4, None, None]
    # no characters lost across the break
    joined = "".join(s["text"] for row in rows for s in row)
    assert joined == "あ" * 12


def test_reading_wider_than_cap_rides_first_fragment_only_its_row_exceeds():
    """A furigana reading wider than (cap - padding) makes cap-fit
    physically impossible for the fragment carrying it, because
    readings never split (spec). The carrying row may exceed the cap;
    every OTHER row must still respect it, no text is lost, and the
    reading appears exactly once on the FIRST fragment.

    Seeded defect that must go red: attach the reading to a later
    fragment, drop it, or let a non-carrying fragment inherit it.
    """
    # kanji 120 px spans 3 fragments of 4 chars at cap 50; the reading
    # か*20 = 100 px alone exceeds cap - padding (46 px), so its row
    # cannot fit no matter how the kanji is broken.
    seg = {"text": "あ" * 12, "reading": "か" * 20}
    rows = layout_rows([seg], max_width_px=50, measure_kanji=_mk, measure_kana=_mn)

    assert rows == [
        [{"text": "あ" * 4, "reading": "か" * 20}],
        [{"text": "あ" * 4}],
        [{"text": "あ" * 4}],
    ]
    # reading present exactly once, on the first fragment
    readings = [s.get("reading") for row in rows for s in row]
    assert readings == ["か" * 20, None, None]
    # no characters lost across the break
    joined = "".join(s["text"] for row in rows for s in row)
    assert joined == "あ" * 12
    # the carrying row is unavoidably over the cap (100 px reading +
    # 4 pad); every row that does NOT carry the reading stays within it
    assert row_width(rows[0], _mk, _mn) == 104
    for row in rows[1:]:
        assert row_width(row, _mk, _mn) <= 50


def test_pathological_segment_wider_than_cap_char_breaks():
    monster = [{"text": "あ" * 30}]  # 300 px wide, cap 100
    rows = layout_rows(monster, max_width_px=100, measure_kanji=_mk, measure_kana=_mn)
    assert len(rows) >= 3                          # split across rows
    joined = "".join(s["text"] for row in rows for s in row)
    assert joined == "あ" * 30                     # no characters lost
    for row in rows:
        assert row_width(row, _mk, _mn) <= 100


def test_zero_or_negative_cap_degrades_to_one_row():
    rows = layout_rows(SEGS, max_width_px=0, measure_kanji=_mk, measure_kana=_mn)
    assert rows == [SEGS]


def test_empty_segments():
    assert layout_rows([], 100, _mk, _mn) == []


def test_empty_text_segment_with_wide_reading_emits_whole_row():
    """A segment with empty text but a furigana reading wider than the cap must
    be emitted WHOLE as a single row, never char-broken. _char_break iterates
    the (empty) text and returns [], so char-breaking such a segment drops it
    entirely; with every segment dropped, layout_rows returns no rows at all,
    which would crash telop_text's max() over an empty list of row widths
    during add_subtitle.

    Seeded defect (one line): drop the ``text and`` guard in layout_rows'
    pre-pass (revert to char-breaking any over-cap segment) -> the empty-text
    segment's _seg_width (54 px) exceeds the cap (30), it char-breaks to [], and
    rows becomes [] instead of one whole row -> red.
    """
    # kana 5 px/char: "か"*10 = 50 px reading; _seg_width = max(0, 50) + 4 = 54,
    # which exceeds the 30 px cap and would trigger the char-break path.
    seg = {"text": "", "reading": "か" * 10}
    rows = layout_rows([seg], max_width_px=30, measure_kanji=_mk, measure_kana=_mn)
    assert rows == [[seg]], (
        f"empty-text segment must ride one whole row, got {rows!r}"
    )
