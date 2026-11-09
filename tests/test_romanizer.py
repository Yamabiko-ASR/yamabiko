"""Tests for Romanizer (reading extraction over cutlet's tagger)."""
from __future__ import annotations

import pytest

from yamabiko.romanization.romanizer import (
    Romanizer,
    contains_kanji,
    is_kanji_run_mark,
)


@pytest.fixture(scope="module")
def rom() -> Romanizer:
    return Romanizer()


def test_reading_hiragana(rom: Romanizer):
    assert rom.reading("東京") == "とうきょう"


def test_reading_already_hiragana_passthrough(rom: Romanizer):
    assert rom.reading("さくら") == "さくら"


def test_reading_katakana_normalizes_to_hiragana(rom: Romanizer):
    # Pure katakana input returns hiragana equivalent
    assert rom.reading("サクラ") == "さくら"


# ---- contains_kanji helper ----

def test_contains_kanji_true():
    assert contains_kanji("田中") is True


def test_contains_kanji_false_pure_kana():
    assert contains_kanji("さくら") is False


def test_contains_kanji_mixed():
    assert contains_kanji("田中さん") is True


def test_contains_kanji_empty():
    assert contains_kanji("") is False


# --- widened kanji ranges + run-mark classification ---

def test_contains_kanji_cjk_extension_a():
    assert contains_kanji("㐂") is True  # U+3402


def test_contains_kanji_compatibility_ideograph():
    assert contains_kanji("﨑") is True  # U+FA11, 山﨑-style name variants


def test_contains_kanji_false_for_iteration_mark_alone():
    # 々 extends a kanji run but is not standalone kanji: a leading 々
    # must not open a run of its own.
    assert contains_kanji("々") is False


def test_kanji_run_marks_recognized():
    for ch in "々〻〆ヵヶ":
        assert is_kanji_run_mark(ch) is True, ch


def test_kanji_run_mark_rejects_regular_chars():
    for ch in "あカ田Aー":
        assert is_kanji_run_mark(ch) is False, ch


def test_romanizer_exposes_tagger():
    from yamabiko.romanization.romanizer import Romanizer
    r = Romanizer()
    tokens = list(r.tagger("東京"))
    assert len(tokens) == 1
