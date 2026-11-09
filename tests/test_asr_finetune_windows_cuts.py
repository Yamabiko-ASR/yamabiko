"""Round-3 cut windows of the pilot (``scripts.asr_finetune.windows``,
``scripts.asr_finetune.cut_audit``): a window that opens or closes
mid-utterance is labelled with exactly the caption words the kept audio
holds, a cut is refused where the caption and the teacher disagree next to
it, a cut window's flags follow its cut texts, and the cut filter drops a cut
window whose label edge the base model does not confirm."""

import json

from scripts.asr_finetune.cut_audit import filter_index
from scripts.asr_finetune.windows import Cut, cut_window, resolve_cut

# The teacher heard the caption word for word: (start_s, end_s, word).
CAPTION = "今日は、駅まで歩いて、家に帰ります。"
WORDS = [(0.0, 0.4, "今日"), (0.4, 0.6, "は"), (0.7, 1.0, "駅"), (1.0, 1.3, "まで"), (1.3, 1.9, "歩いて"),
         (2.0, 2.3, "家"), (2.3, 2.4, "に"), (2.4, 3.2, "帰ります")]


def test_resolve_cut_labels_exactly_the_caption_words_the_kept_audio_holds() -> None:
    # Head target 0.2 x 3.4 s = 0.68 s: the gap before 駅 (0.65 s).
    head = resolve_cut(CAPTION, WORDS, 3.4, 0.2, None)
    # Tail target 0.7 x 3.4 s = 2.38 s: the gap before 帰ります (2.40 s).
    tail = resolve_cut(CAPTION, WORDS, 3.4, None, 0.7)
    # The teacher's % normalizes to nothing: a cut on either side of it (the
    # target's gap before it, 0.9 s, or after it) would keep its audio and
    # drop its text, or the reverse; the cut takes the gap before 5 (0.7 s).
    percent = resolve_cut("金利は5%上がったそうです", [
        (0.0, 0.5, "金利"), (0.5, 0.7, "は"), (0.7, 0.9, "5"), (0.9, 1.4, "%"), (1.4, 1.8, "上が"),
        (1.8, 2.1, "った"), (2.1, 2.5, "そう"), (2.5, 3.0, "です")], 3.2, 0.28125, None)

    assert head == Cut(0.65, None, 0.7, 3.2, "駅まで歩いて、家に帰ります。")
    assert tail == Cut(None, 2.4, 0.0, 2.4, "今日は、駅まで歩いて、家に")
    assert percent == Cut(0.7, None, 0.7, 3.0, "5%上がったそうです")


def test_resolve_cut_never_cuts_next_to_a_caption_teacher_mismatch() -> None:
    # The caption says 駅前 where the teacher heard 駅: the head target's
    # gap (before 駅) and the gaps up to 歩いて sit next to the extra 前, so
    # the cut moves to the nearest gap that matches (before 家, 1.95 s).
    moved = resolve_cut("今日は、駅前まで歩いて、家に帰ります。", WORDS, 3.4, 0.2, None)
    # A caption the teacher matches nowhere is refused.
    refused = resolve_cut("明日の天気は晴れのち曇りです。", WORDS, 3.4, 0.2, None)

    assert moved == Cut(1.95, None, 2.0, 3.2, "家に帰ります。")
    assert refused == "mismatch"


def test_cut_window_recomputes_positive_from_the_cut_text_and_never_anchors() -> None:
    # A ReazonSpeech window whose only interjection opens its first line;
    # the head cut removes it.
    first, second = "えっ、駅まで歩いて行こうか", "明日は雨が降るそうです"
    row = {"w": 7, "source": "reazon", "positive": True, "anchor": False,
           "utts": [{"src": "reazon", "id": "000/a.flac", "cut_start": 0.3},
                    {"src": "reazon", "id": "000/b.flac"}]}
    texts = {("reazon", "000/a.flac"): first, ("reazon", "000/b.flac"): second}
    cuts = {(7, 0): Cut(0.5, None, 0.6, 2.0, "駅まで歩いて行こうか")}

    lines, positive, anchor = cut_window(row, texts, cuts)

    assert lines == ["駅まで歩いて行こうか", second]
    assert (positive, anchor) == (False, False)


def test_filter_index_keeps_only_cut_windows_the_base_confirms(tmp_path) -> None:
    # w 0 is uncut; the base hears w 1's head edge as labelled, and at w 2's
    # tail it hears 駅前 where the label ends in 駅まで.
    rows = [{"w": 0, "offset": 0, "samples": 16000, "labels": [], "segments": [[0.5, 1.0, "reazon"]]},
            {"w": 1, "offset": 16000, "samples": 16000, "labels": [], "cut": ["head"],
             "segments": [[0.5, 1.0, "reazon"]]},
            {"w": 2, "offset": 32000, "samples": 16000, "labels": [], "cut": ["tail"],
             "segments": [[0.5, 1.0, "galgame"]]}]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    labels = {1: ["駅まで歩いて行こうか"], 2: ["今日は駅まで"]}
    bases = {1: "駅まで歩いて行こうかな", 2: "今日は駅前"}

    record = filter_index(tmp_path, lambda entries: [bases[entry["w"]] for entry in entries],
                          lambda entry: labels[entry["w"]])

    kept = [json.loads(line)["w"] for line in index.read_text(encoding="utf-8").splitlines()]
    original = [json.loads(line)["w"] for line in (tmp_path / "index.unfiltered.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (kept, original) == ([0, 1], [0, 1, 2])
    assert (record["kept"], record["dropped"], record["dropped_w"]) == (1, 1, [2])
    # A crashing decode confirms nothing: every cut window goes, w 0 stays.
    filter_index(tmp_path, lambda entries: 1 / 0, lambda entry: labels[entry["w"]])
    assert [json.loads(line)["w"] for line in index.read_text(encoding="utf-8").splitlines()] == [0]
