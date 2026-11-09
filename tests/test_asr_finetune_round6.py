"""Round 6 of the pilot: the vocal cap drops labels with long vocal chains by
each of its four rules while short interjection lines stay, on every source,
and it adds the round-6 adult words to the galgame adult filter."""

from scripts.asr_finetune.pilot import vocal_reason
from scripts.asr_finetune.select import TextRules

CHAIN = "ウオオオオオォォォッウアアアアアアアァァァァッッッヒイイイィィックソッ"


def test_vocal_reason_drops_each_long_vocal_rule_and_keeps_short_interjection_lines() -> None:
    assert vocal_reason(CHAIN) == "long_vocal_total"
    assert vocal_reason("あっ、あっ、あああっ") == "long_vocal_dense"
    assert vocal_reason("ハァスゥハァスゥ") == "long_vocal_chunk"
    assert vocal_reason("はぁ、はぁ、はぁ......そんなの、飲んじゃうだなんて") == "long_vocal_repeat"
    for line in ("はい", "うん", "えっ", "あっ、そうなんだ", "うん、うん", "ええっ!?", "えっ?あっ?はい!",
                 "いいよいいよ、俺が呼んでくる", "はい。ありがとうございます。ふふ"):
        assert vocal_reason(line) is None, line


def test_vocal_cap_applies_to_every_source_and_adds_the_round6_words_to_the_adult_filter() -> None:
    explicit = "あたしのおまんこ......そんなに気持ちいいんだ?"

    assert TextRules(vocal=True).reason("reazon", CHAIN) == "long_vocal_total"
    assert TextRules(adult=True).reason("galgame", explicit) is None
    assert TextRules(adult=True, vocal=True).reason("galgame", explicit) == "adult_sfx"
