"""Round 4 of the pilot: the trainer's loss reaches the first predicted
position (where the decode reads ``no_speech_prob``) for speech and no-speech
windows, loop-prone labels are dropped while repeated punctuation stays, the
no-speech windows are 8% of the plan, never positive or anchor, with dev
windows from dev files only, and the adult-scene filter (v4c) drops
sound-effect and moan-dense galgame lines while dialogue with meaningful
interjections stays."""

from scripts.asr_finetune.labels import EOT, JA, NOSPEECH, NOSPEECH_LABELS, label_mask
from scripts.asr_finetune.mix import Item, Window, nonspeech_windows
from scripts.asr_finetune.pilot import adult_reason, loop_reason
from scripts.asr_finetune.select import TextRules

TRANSCRIBE = 8734


def test_label_mask_trains_the_first_position_of_speech_and_nospeech_windows_only_with_target() -> None:
    speech = [JA, TRANSCRIBE, 8764, 7772, 8800, EOT]

    assert label_mask(speech, TRANSCRIBE, "target") == (1,)
    assert label_mask(speech, TRANSCRIBE, "masked") == (0, 1)
    assert label_mask(NOSPEECH_LABELS, TRANSCRIBE, "target") == ()
    assert label_mask([NOSPEECH, EOT], TRANSCRIBE, "masked") is None


def test_loop_reason_drops_runs_and_repeats_but_keeps_repeated_punctuation() -> None:
    assert loop_reason("きゃああああっ") == "loop_run"
    assert loop_reason("ええーーーー？") == "loop_run"
    assert loop_reason("すごい、すごい、すごい、すごい、すごい、すごい、すごい、すごい") == "loop_ratio"
    assert loop_reason("えっ……本当に！！！！") is None
    assert loop_reason("ああ、そうだね") is None


def test_nonspeech_windows_are_eight_percent_never_flagged_with_dev_from_dev_files_only() -> None:
    files = [{"path": f"musan/{kind}/src/{kind}-{n:04d}.wav", "kind": kind, "seconds": 3.0 + n}
             for kind in ("noise", "music") for n in range(100)]
    unit = Item("reazon", "000/a.flac", "reazon", 5.0, 10, False)
    speech = [Window("reazon", "P", "dev" if n < 10 else "train", [unit], [0.0], -24.0, [0.0], False, True)
              for n in range(920)]

    windows = nonspeech_windows({"files": files}, speech)

    seconds = {f["path"]: f["seconds"] for f in files}
    dev = {w.units[0].id for w in windows if w.split == "dev"}
    train = {w.units[0].id for w in windows if w.split == "train"}
    assert (len(windows), sum(w.split == "dev" for w in windows)) == (80, 1)
    assert not dev & train
    assert not any(w.positive or w.anchor for w in windows)
    assert all(w.shape == "N" and 2.0 <= w.span[1] <= 28.0 and w.span[0] + w.span[1] <= seconds[w.units[0].id]
               for w in windows)


def test_adult_reason_drops_sound_effects_and_moan_dense_lines_and_keeps_dialogue_interjections() -> None:
    assert adult_reason("あっ……ああっ、たっくんのが……あたしの中に……んんんっ……") == "adult_sfx"
    assert adult_reason("ジュルッ、れろ……") == "adult_sfx"
    assert adult_reason("はぁ、はぁ……ふぁ、んっ……ひゃうっ") == "adult_moan"
    for dialogue in ("えっ、あの、はい", "うん、わかった", "あっ、そうだ。忘れてた", "はぁ……疲れた", "いや、だめだって"):
        assert adult_reason(dialogue) is None, dialogue


def test_text_rules_apply_the_adult_filter_to_galgame_lines_only() -> None:
    moans = "はぁ、はぁ……ふぁ、んっ……ひゃうっ"

    assert TextRules(adult=True).reason("galgame", moans) == "adult_moan"
    assert TextRules(adult=True).reason("reazon", moans) is None
    assert TextRules().reason("galgame", moans) is None
