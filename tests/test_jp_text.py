from yamabiko.jp_text import is_japanese


def test_kanji_is_japanese():
    assert is_japanese("日本語のテキスト")


def test_hiragana_is_japanese():
    assert is_japanese("ありがとう")


def test_katakana_is_japanese():
    assert is_japanese("コンピューター")


def test_pure_english_is_not_japanese():
    assert not is_japanese("Hello, world.")


def test_empty_is_not_japanese():
    assert not is_japanese("")


def test_whitespace_only_is_not_japanese():
    assert not is_japanese("   \n\t")


def test_english_sentence_with_jp_fragment_is_not_japanese():
    # ASR hallucinates things like this on English input.
    assert not is_japanese("オッケー、I will let him know.")


def test_mostly_jp_with_short_ascii_abbreviation_is_japanese():
    # AI / NHK / LDP inside a mostly-JP line should still pass.
    assert is_japanese("AI サーファーに、彼らがただ何でも作ってもいい、ただ、彼らが知っている限り、他のものがある。")


def test_punctuation_only_is_not_japanese():
    assert not is_japanese("... , . !")


def test_numbers_only_is_not_japanese():
    assert not is_japanese("123 456")


# 々 (U+3005, category Lm) and halfwidth katakana (U+FF66-FF9F) are
# alphabetic but fall outside the three checked JP ranges, so they were
# counted as LATIN. For short transcripts (時々, 色々ね — realistic 1-second
# VAD chunks) that pushed latin_ratio over 0.25 and the valid JP chunk was
# silently dropped by translation_stage's guard (main.py:205).
# Seeded defect for each: remove U+3005 / U+FF66-FF9F from _is_jp_char.

def test_iteration_mark_word_tokidoki_is_japanese():
    assert is_japanese("時々")


def test_iteration_mark_word_hitobito_is_japanese():
    assert is_japanese("人々")


def test_iteration_mark_with_trailing_kana_is_japanese():
    # jp=2 latin=1 ratio 0.333 under the broken classifier -> dropped.
    assert is_japanese("色々ね")


def test_halfwidth_katakana_transcript_is_japanese():
    # Entire halfwidth-katakana transcript had jp_count=0 -> rejected.
    assert is_japanese("ｵｯｹｰﾃﾞｽ")


def test_english_sentence_with_iteration_mark_fragment_still_rejected():
    # Widening the JP ranges must not weaken the hallucination guard:
    # a full English sentence with one 々-word fragment stays rejected.
    assert not is_japanese("時々 I will let him know about it.")


# The jp_text and romanizer kanji classifiers had diverged: contains_kanji
# gained Ext A + compatibility ideographs but _is_jp_char still counted
# 﨑 (U+FA11, alphabetic Lo) and 〻 (U+303B, Lm) as Latin. Both modules
# now share jp_text.KANJI_RANGES.
# Seeded defect: revert _is_jp_char to the basic 0x4E00-0x9FFF block.

def test_compatibility_ideograph_name_is_japanese():
    assert is_japanese("﨑さん")


def test_vertical_iteration_mark_word_is_japanese():
    assert is_japanese("時〻")


def test_kana_iteration_marks_already_counted_japanese():
    # ゝ (U+309D) sits inside the hiragana block — regression pin that the
    # range rework keeps it Japanese.
    assert is_japanese("こゝろ")
