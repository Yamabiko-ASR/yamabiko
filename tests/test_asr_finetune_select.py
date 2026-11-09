"""Rule (e) of the pilot's ReazonSpeech teacher filter
(``scripts.asr_finetune.select.reazon_rules``): a caption that leaves out
speech the prompt-free teacher heard, most often the opening words, drops the
clip; a caption that differs from it by at most two kana keeps it."""

from scripts.asr_finetune.pilot import BenchLineFilter
from scripts.asr_finetune.select import TEXT_AVAILABLE, reazon_rules

NO_BENCH_LINES = BenchLineFilter([])


def test_rule_e_drops_a_caption_missing_three_leading_kana_and_keeps_a_two_kana_difference() -> None:
    # The teacher hears ソコデ before the captioned words: three leading
    # teacher kana the caption lacks.
    missing_lead = reazon_rules("さかなをたべたいとおもいます", "そこでさかなをたべたいとおもいます",
                                TEXT_AVAILABLE, NO_BENCH_LINES)
    # The teacher hears タベ where the caption says ノミ: two kana differ.
    two_kana = reazon_rules("さかなをのみたいとおもいます", "さかなをたべたいとおもいます",
                            TEXT_AVAILABLE, NO_BENCH_LINES)

    assert missing_lead["e_caption_deletes"] is True
    assert two_kana["e_caption_deletes"] is False
