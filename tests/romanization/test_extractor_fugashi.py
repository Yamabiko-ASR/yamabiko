"""fugashi+UniDic NER tests.

Task 1 probe observations (for reference; pos2 for each surface):
  田中 → ['固有名詞']
  東京 → ['固有名詞']
  ソフトバンク → ['普通名詞', '普通名詞']  (split — not a proper noun to UniDic)
  日本 → ['固有名詞']  (fugashi tags as proper; denylist in Task 4 handles it)
  代表 → ['普通名詞']
  昼 → ['普通名詞']
  客 → ['普通名詞']
  ワイド → ['普通名詞']
  ショー → ['普通名詞']
"""
import fugashi
import pytest
from pathlib import Path
from yamabiko.romanization.extractor_fugashi import ProperNounExtractor, load_denylist


@pytest.fixture(scope="module")
def extractor():
    return ProperNounExtractor(tagger=fugashi.Tagger(), denylist=set())


def test_extracts_person_name(extractor):
    assert "田中" in extractor.extract("俳優の田中さんが話しました")


def test_extracts_place_name(extractor):
    assert "東京" in extractor.extract("東京の天気")


def test_katakana_brand_now_caught_by_run_pass(extractor):
    """Pass 2 (raw katakana-run scan) catches ソフトバンク as a single run,
    even though UniDic still POS-splits it into [ソフト, バンク] as 普通名詞
    so Pass 1 (固有名詞 filter) never sees it.

    Seeded defect: remove the Pass-2 katakana-run loop from extract_all ->
    ソフトバンク no longer in result, red.
    """
    result = extractor.extract("ソフトバンクが発表した")
    assert "ソフトバンク" in result


def test_extract_splits_katakana_run_on_nakaguro(extractor):
    """・ separates name parts; the run must split (スバル, ナツキ), never merge
    into スバル・ナツキ (which cutlet would render with a literal '/').

    Seeded defect: put ・ (U+30FB) back in _KATA_RUN -> merged run present, red.
    """
    got = extractor.extract("スバル・ナツキ、こっちだ。")
    assert "スバル・ナツキ" not in got
    assert "スバル" in got and "ナツキ" in got


def test_extract_all_flags_propn_but_not_bare_katakana_runs(extractor):
    """extract_all returns (candidates, propn_set). リムル (固有名詞) is in propn;
    テンペスト (普通名詞, Pass-2 run) is a candidate but NOT in propn.

    Seeded defect: put every Pass-2 run into propn_set -> テンペスト in propn, red.
    """
    _, propn = extractor.extract_all("リムルさんが来た")
    assert "リムル" in propn
    cands, propn2 = extractor.extract_all("テンペストを法治国家にする。")
    assert "テンペスト" in cands and "テンペスト" not in propn2


def test_rejects_common_noun_daihyou(extractor):
    assert "代表" not in extractor.extract("日本代表が勝利しました")


def test_rejects_common_noun_hiru(extractor):
    assert "昼" not in extractor.extract("お昼のワイドショー")


def test_rejects_katakana_common_nouns(extractor):
    extracted = extractor.extract("お昼のワイドショー")
    assert "ワイド" not in extracted
    assert "ショー" not in extracted


def test_skips_single_kanji(extractor):
    assert "客" not in extractor.extract("お客さんが来た")


def test_honors_denylist():
    ex = ProperNounExtractor(tagger=fugashi.Tagger(), denylist={"日本"})
    assert "日本" not in ex.extract("日本の政府")


def test_no_duplicates_in_result(extractor):
    result = extractor.extract("田中さんと田中さんが話した")
    assert result.count("田中") == 1


def test_empty_input(extractor):
    assert extractor.extract("") == []


def test_load_denylist_reads_utf8(tmp_path):
    p = tmp_path / "deny.txt"
    p.write_text("日本\n代表\n\n# comment\n昼\n", encoding="utf-8")
    assert load_denylist(p) == {"日本", "代表", "昼"}


def test_load_denylist_missing_returns_empty(tmp_path):
    assert load_denylist(tmp_path / "nonexistent.txt") == set()
