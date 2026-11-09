"""Segment-level wrap layout for the ruby JP subtitle row.

Pure module (no Qt imports): widths come in through measurer callables
so unit tests run without a QApplication. telop_text binds the font
metrics for production.

A segment (kanji text + its furigana reading) never splits across rows;
the whole segment moves to the next row when it does not fit. The one
exception is a single segment wider than the cap by itself: it is
character-broken, and the reading stays with the FIRST fragment (spec
section 3).
"""
from __future__ import annotations

from typing import Callable

_SEG_PADDING = 4  # single owner of the per-segment padding; consumers take it via the seg_padding defaults


def seg_measures(seg: dict, measure_kanji: Callable[[str], int],
                 measure_kana: Callable[[str], int],
                 seg_padding: int = _SEG_PADDING) -> tuple[int, int, int]:
    """Measure one segment: (kanji_w, kana_w, seg_w).

    seg_w = max(kanji_w, kana_w) + seg_padding is THE definition of a
    segment's painted width; _seg_width and row_width derive from it, so a
    painter that caches these tuples at layout time reads the same numbers
    the wrap decisions were made with, never a re-derived (or re-invented)
    copy of the formula.
    """
    kanji_w = measure_kanji(seg.get("text", ""))
    reading = seg.get("reading")
    kana_w = measure_kana(reading) if reading else 0
    return kanji_w, kana_w, max(kanji_w, kana_w) + seg_padding


def _seg_width(seg: dict, mk: Callable[[str], int], mn: Callable[[str], int],
               pad: int) -> int:
    return seg_measures(seg, mk, mn, pad)[2]


def row_width(row: list[dict], measure_kanji: Callable[[str], int],
              measure_kana: Callable[[str], int],
              seg_padding: int = _SEG_PADDING) -> int:
    return sum(_seg_width(s, measure_kanji, measure_kana, seg_padding) for s in row)


def _char_break(seg: dict, max_width_px: int, mk, pad) -> list[dict]:
    """Split one oversized segment into cap-fitting fragments.

    Fragments are sized by kanji width, so every fragment's kanji fits
    the cap. The reading is attached to the first fragment only
    (rendering a reading over a partial word is wrong either way;
    first-fragment keeps it visible and adjacent).

    Honest bound: readings never split (spec), so when a reading alone
    is wider than ``max_width_px - pad`` the fragment carrying it
    cannot fit and its row will exceed ``max_width_px``. That overflow
    is physically unavoidable for that one row. Every fragment whose
    reading fits (or has none) stays within the cap, because a
    kanji-sized fragment plus a fitting reading is still
    ``max(kanji, reading) + pad <= max_width_px``.
    """
    text = seg.get("text", "")
    out: list[dict] = []
    cur = ""
    for ch in text:
        candidate = cur + ch
        if cur and mk(candidate) + pad > max_width_px:
            out.append({"text": cur})
            cur = ch
        else:
            cur = candidate
    if cur:
        out.append({"text": cur})
    if out and seg.get("reading"):
        out[0] = {"text": out[0]["text"], "reading": seg["reading"]}
    return out


def layout_rows(segments: list[dict], max_width_px: int,
                measure_kanji: Callable[[str], int],
                measure_kana: Callable[[str], int],
                seg_padding: int = _SEG_PADDING) -> list[list[dict]]:
    """Break ``segments`` into rows no wider than ``max_width_px``.

    max_width_px <= 0 means "no cap" (single row): the caller failed to
    provide a real width; degrading beats crashing or emitting nothing.

    Cap guarantee and its one exception: every emitted row is
    ``<= max_width_px`` EXCEPT a row that carries a char-break fragment
    whose furigana reading alone is wider than
    ``max_width_px - seg_padding``. Readings never split (spec), so
    such a row is unavoidably over the cap; all other rows respect it.
    See ``_char_break`` for the derivation.
    """
    if not segments:
        return []
    if max_width_px <= 0:
        return [list(segments)]

    # Pre-pass: char-break any segment that alone exceeds the cap.
    flat: list[dict] = []
    for seg in segments:
        # An empty-text segment is emitted WHOLE, never char-broken: a wide
        # furigana reading on empty text makes _seg_width exceed the cap, but
        # _char_break iterates the (empty) text and returns [], which would
        # drop the segment. With every segment dropped, layout_rows returns []
        # and telop_text's max() over no row widths raises ValueError
        # mid-add_subtitle, killing the render. Keeping the segment guarantees
        # at least one non-empty row.
        text = seg.get("text", "")
        if text and _seg_width(seg, measure_kanji, measure_kana, seg_padding) > max_width_px:
            flat.extend(_char_break(seg, max_width_px, measure_kanji,
                                    seg_padding))
        else:
            flat.append(seg)

    rows: list[list[dict]] = []
    cur: list[dict] = []
    cur_w = 0
    for seg in flat:
        w = _seg_width(seg, measure_kanji, measure_kana, seg_padding)
        if cur and cur_w + w > max_width_px:
            rows.append(cur)
            cur = [seg]
            cur_w = w
        else:
            cur.append(seg)
            cur_w += w
    if cur:
        rows.append(cur)
    return rows
