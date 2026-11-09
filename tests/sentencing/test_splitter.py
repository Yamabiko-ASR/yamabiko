from yamabiko.sentencing.splitter import SentenceSplitter, has_terminator


def test_has_terminator_true():
    assert has_terminator("今日はいい天気ですね。")
    assert has_terminator("本当？")
    assert has_terminator("すごい！")


def test_has_terminator_false():
    assert not has_terminator("今日はいい天気")
    assert not has_terminator("")


def test_split_single_sentence():
    s = SentenceSplitter()
    assert s.split("今日はいい天気ですね。") == ["今日はいい天気ですね。"]


def test_split_two_sentences():
    s = SentenceSplitter()
    result = s.split("こんにちは。元気ですか？")
    assert len(result) == 2
    assert result[0].endswith("。")
    assert result[1].endswith("？")


def test_split_trailing_partial():
    s = SentenceSplitter()
    result = s.split("こんにちは。元気")
    assert len(result) == 2
    assert has_terminator(result[0])
    assert not has_terminator(result[1])


def test_split_empty():
    s = SentenceSplitter()
    assert s.split("") == []


def test_split_whitespace_only():
    s = SentenceSplitter()
    assert s.split("   \n\t") == []


def test_split_preserves_terminators():
    s = SentenceSplitter()
    result = s.split("本当？はい。")
    # Both sentences should retain their terminators in some sentence.
    joined = "".join(result)
    assert "？" in joined
    assert "。" in joined


def test_split_handles_decimal_point():
    s = SentenceSplitter()
    # 3.14 contains a period but isn't a sentence boundary.
    result = s.split("円周率は3.14です。")
    assert len(result) == 1


def test_split_handles_parenthetical():
    s = SentenceSplitter()
    result = s.split("すごい(笑)。本当に面白い。")
    assert len(result) == 2
