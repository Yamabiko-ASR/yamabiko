"""Round 5 of the pilot: the ``r5`` no-speech windows are 10% of the plan,
mostly short spans and partly low-level, each a window of its own; and the
loss weight of the first predicted position (``<|ja|>`` or ``<|nospeech|>``)
leaves the masked positions out and is the plain token mean at weight 1."""

from scripts.asr_finetune.labels import EOT, JA, NOSPEECH_LABELS, label_mask, loss_weights
from scripts.asr_finetune.mix import NONSPEECH_PROFILES, Item, Window, nonspeech_windows

TRANSCRIBE = 8734


def test_r5_nonspeech_windows_are_ten_percent_mostly_short_partly_low_level_each_its_own_window() -> None:
    files = [{"path": f"musan/{kind}/src/{kind}-{n:04d}.wav", "kind": kind, "seconds": 3.0 + n}
             for kind in ("noise", "music") for n in range(100)]
    unit = Item("reazon", "000/a.flac", "reazon", 5.0, 10, False)
    speech = [Window("reazon", "P", "dev" if n < 30 else "train", [unit], [0.0], -24.0, [0.0], False, True)
              for n in range(2700)]

    windows = nonspeech_windows({"files": files}, speech, NONSPEECH_PROFILES["r5"])

    spans = [w.span[1] for w in windows]
    low_level = [w for w in windows if -50.0 <= w.level <= -35.0 and w.gains == [0.0]]
    assert len(windows) == 300
    assert all(w.shape == "N" and len(w.units) == 1 and not (w.positive or w.anchor) for w in windows)
    assert all(0.3 <= span <= 28.0 for span in spans)
    assert 0.62 <= sum(span <= 2.5 for span in spans) / len(spans) <= 0.78
    assert sum(span < 1.0 for span in spans) >= 40
    assert 0.14 <= len(low_level) / len(windows) <= 0.26


def test_loss_weights_weight_the_trained_first_position_and_weight_one_is_the_token_mean() -> None:
    speech = [JA, TRANSCRIBE, 8764, 7772, EOT]

    assert loss_weights(len(speech), label_mask(speech, TRANSCRIBE, "target"), 4.0) == (4.0, 0.0, 1.0, 1.0, 1.0)
    assert loss_weights(2, label_mask(NOSPEECH_LABELS, TRANSCRIBE, "target"), 4.0) == (4.0, 1.0)
    assert loss_weights(len(speech), label_mask(speech, TRANSCRIBE, "masked"), 4.0) == (0.0, 0.0, 1.0, 1.0, 1.0)
    assert sum(loss_weights(len(speech), (1,), 1.0)) == len(speech) - 1
