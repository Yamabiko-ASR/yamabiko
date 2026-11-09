"""Sudachi split-mode C POS probe.

Run with `-s` to see the observed POS chain for each surface.
These observations anchor the assertions in Task 1.

Environment: sudachipy 0.6.11, sudachidict_core 20260116, split mode C.

Observed POS chains (surface | pos[0] | pos[1] | pos[2] | pos[3])
-----------------------------------------------------------------
田中        | 名詞     | 固有名詞  | 人名      | 姓
東京        | 名詞     | 固有名詞  | 地名      | 一般
ソフトバンク | 名詞     | 固有名詞  | 一般      | *       <- single token, keeps katakana company intact
マクドナルド | 名詞     | 固有名詞  | 一般      | *       <- single token in mode C
山田        | 名詞     | 固有名詞  | 人名      | 姓
日本        | 名詞     | 固有名詞  | 地名      | 国      <- tags as 固有名詞 (matches fugashi; denylist handles)

Common-noun / non-proper surfaces (should NOT be picked up as 固有名詞):
昼          | 名詞     | 普通名詞  | 副詞可能
ワイド       | 名詞     | 普通名詞  | 形状詞可能           <- mode C splits ワイドショー -> ワイド + ショー
ショー       | 名詞     | 普通名詞  | 一般                <- ワイドショー NOT flagged as 固有名詞 (safe, no denylist needed)
俳優        | 名詞     | 普通名詞  | 一般
代表        | 名詞     | 普通名詞  | サ変可能             <- safe, not 固有名詞
勝利        | 名詞     | 普通名詞  | サ変可能
お客さん     | 名詞     | 普通名詞  | 一般                <- single token, NOT 固有名詞 (safe)
会議        | 名詞     | 普通名詞  | サ変可能
発表        | 名詞     | 普通名詞  | サ変可能

Affixes / function words:
さん        | 接尾辞   | 名詞的    | 一般
お          | 接頭辞   | *         | *                   <- お昼 splits as 接頭辞 + 普通名詞
が/で/に/の  | 助詞     | 格助詞    | *
来/行く/ある | 動詞     | 非自立可能 | *
た          | 助動詞   | *         | *

Key findings for Task 1 assertions:
- Proper-noun detection rule: pos[0] == '名詞' AND pos[1] == '固有名詞'.
- ソフトバンク and マクドナルド survive as single tokens in mode C (good —
  no reassembly needed, unlike fugashi which split them).
- 日本 tags as 固有名詞 (地名/国). Denylist is required for it (same risk as fugashi).
- ワイドショー splits into ワイド (普通名詞) + ショー (普通名詞). Neither is
  flagged as 固有名詞 — no denylist entry needed for these, BUT the compound
  surface "ワイドショー" cannot be recovered as one extracted name from mode C.
- 代表, 昼, 客 (as お客さん), 俳優 are all 普通名詞 — safe, will NOT be picked
  up by the 固有名詞 rule.
- お客さん is a single 普通名詞 token (not split into お + 客 + さん).
"""
import pytest
from yamabiko.romanization.extractor_sudachi import SudachiExtractor


@pytest.fixture(scope="module")
def extractor():
    return SudachiExtractor(denylist=set())


def test_extracts_katakana_brand(extractor):
    assert "ソフトバンク" in extractor.extract("ソフトバンクが発表した")


def test_extracts_mcdonalds(extractor):
    assert "マクドナルド" in extractor.extract("マクドナルドに行く")


def test_extracts_person_name(extractor):
    assert "田中" in extractor.extract("俳優の田中さんが話しました")


def test_extracts_place_name(extractor):
    assert "東京" in extractor.extract("東京で会議がある")


def test_rejects_common_noun(extractor):
    assert "代表" not in extractor.extract("日本代表が勝利しました")


def test_skips_single_kanji(extractor):
    assert "客" not in extractor.extract("お客さんが来た")


def test_honors_denylist():
    ex = SudachiExtractor(denylist={"日本"})
    assert "日本" not in ex.extract("日本の政府")


def test_empty_input(extractor):
    assert extractor.extract("") == []


def test_no_duplicates(extractor):
    result = extractor.extract("田中さんと田中さんが話した")
    assert result.count("田中") == 1
