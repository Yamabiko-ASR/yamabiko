"""Prompt assembly behavior tests.

Intentionally avoids pinning exact wording of SYSTEM_PROMPT — a wording
refactor that preserves the output contract (one line, no decorations)
should not break these. The translator-facing invariants are covered here
and in test_postprocess.py / test_translator_retry.py.
"""
from __future__ import annotations

from yamabiko.translation.prompt import (
    GENRE_NOTES,
    SYSTEM_PROMPT,
    build_user_message,
    fuzzy_suffix_dedup,
)


def test_system_prompt_nonempty_and_mentions_translation():
    """Behavioral smoke: the prompt exists and at least mentions its job."""
    assert SYSTEM_PROMPT.strip()
    low = SYSTEM_PROMPT.lower()
    assert "translat" in low  # translate / translation / translator
    assert "japanese" in low or "japan" in low


def test_system_prompt_embeds_instructions_before_examples():
    """postprocess builds its prompt-echo vocabulary from
    SYSTEM_PROMPT_INSTRUCTIONS only (the few-shot example sentences are
    ordinary dialogue and must stay OUT of that vocabulary), while the model
    still sees the full instructions + examples prompt. Seeded defect:
    rebuild SYSTEM_PROMPT without the example block (or from a different
    instruction text) -> one of these goes red."""
    from yamabiko.translation.prompt import SYSTEM_PROMPT_INSTRUCTIONS

    assert SYSTEM_PROMPT.startswith(SYSTEM_PROMPT_INSTRUCTIONS)
    # The examples still ship to the model (they follow the instructions).
    assert len(SYSTEM_PROMPT) > len(SYSTEM_PROMPT_INSTRUCTIONS)


def test_build_user_message_includes_genre_name():
    msg = build_user_message("テスト。", "drama", [])
    assert "drama" in msg


def test_build_user_message_includes_transcript_line():
    msg = build_user_message("今日はいい天気。", "variety", [])
    # The transcript appears verbatim somewhere in the prompt.
    assert "今日はいい天気。" in msg


def test_build_user_message_includes_prev_lines_when_nonempty():
    prev = ["First previous line.", "Second previous line."]
    msg = build_user_message("今日は。", "variety", prev)
    for line in prev:
        assert line in msg


def test_prompt_tells_the_model_earlier_lines_are_context_never_to_translate_again():
    """Live bleed 2026-11-02: the earlier EN lines reached the model under
    "Previous:" with nothing saying what that section was, and the model
    restated the previous line, a self-introduction, inside the new one, in
    13 of 13 tries on the shipped model. A system
    rule naming that section as already-translated context cut it to 0 of 8.
    Seeded defects: drop the rule, or rename the builder's context label
    without the rule -> red."""
    context = ["I am here to greet you.", "I serve as the deputy head."]
    msg = build_user_message("え、よろしくお願いします。", "drama", context)
    # The label the earlier lines actually reach the model under.
    label = next(
        line for line in msg.splitlines() if context[0] in line
    ).split(":", 1)[0]
    rules = [line for line in SYSTEM_PROMPT.splitlines() if f'"{label}:"' in line]
    assert rules, f"the system prompt never says what the {label!r} lines are"
    assert "never repeat" in rules[0].lower()


# fuzzy_suffix_dedup previously accepted a 1-char overlap, so a coincidental
# suffix/prefix match between UNRELATED utterances silently stripped the start
# of the new one (pervasive with aizuchi: prev ends ね, next starts ねえ).
# A minimum overlap of 3 chars is now required; cross-silence-gap protection
# is the caller's job (asr_stage only runs the dedup across force-split
# boundaries — see test_main_pipeline.py).
# Seeded defect: restore `range(limit, 0, -1)` in the overlap loop.

class TestFuzzySuffixDedup:
    def test_single_char_coincidental_overlap_not_stripped(self):
        # The exact F111 live repro: 'ね' | 'ねえ…' lost its first char.
        assert fuzzy_suffix_dedup("ちょっと待ってね", "ねえ、見て見て") == "ねえ、見て見て"

    def test_two_char_overlap_below_minimum_not_stripped(self):
        assert fuzzy_suffix_dedup("それで言うのです", "ですか") == "ですか"

    def test_exact_minimum_three_char_overlap_strips(self):
        assert fuzzy_suffix_dedup("終わりました", "ましたので") == "ので"

    def test_long_reemitted_tail_strips(self):
        # ≥ min overlap strips at the function level; protecting genuine
        # echoes across silence gaps is the caller's adjacency gate.
        assert fuzzy_suffix_dedup("そうですね", "そうですね、たしかに") == "、たしかに"

    def test_full_duplicate_strips_to_empty(self):
        assert fuzzy_suffix_dedup("そうですね", "そうですね") == ""

    def test_no_overlap_returns_new_unchanged(self):
        assert fuzzy_suffix_dedup("今日は晴れ", "明日は雨") == "明日は雨"

    def test_empty_prev_returns_new_unchanged(self):
        assert fuzzy_suffix_dedup("", "ねえ、見て") == "ねえ、見て"

    def test_empty_new_returns_empty(self):
        assert fuzzy_suffix_dedup("ちょっと待って", "") == ""

    def test_overlap_capped_at_max_overlap(self):
        # Overlaps longer than max_overlap are not considered; the longest
        # candidate is max_overlap chars.
        prev = "あ" * 20
        new = "あ" * 15 + "X"
        assert fuzzy_suffix_dedup(prev, new, max_overlap=12) == "あ" * 3 + "X"


def test_build_user_message_omits_or_stubs_prev_when_empty():
    """Empty context must not emit the previous lines verbatim; either a
    stub token ("(none)") or the section is omitted."""
    msg = build_user_message("今日は。", "variety", [])
    # No stray previous-line content can leak in, and if the section exists
    # it must be the documented placeholder.
    if "Previous:" in msg:
        # The line after "Previous:" must be the stub, not arbitrary text.
        for line in msg.splitlines():
            if line.startswith("Previous:"):
                tail = line.split("Previous:", 1)[1].strip()
                assert tail == "(none)", f"empty-context previous tail must be '(none)'; got {tail!r}"


def test_genre_notes_actually_vary_output():
    """Switching genre must change the prompt — otherwise genre selection is dead code."""
    variety = build_user_message("テスト。", "variety", [])
    drama = build_user_message("テスト。", "drama", [])
    assert variety != drama
    # And each carries its own genre-specific note.
    assert GENRE_NOTES["variety"] in variety
    assert GENRE_NOTES["drama"] in drama


def test_all_shipped_genres_have_notes():
    """Genre list in state.VALID_GENRES must all have prompt notes."""
    from yamabiko.state import VALID_GENRES
    for genre in VALID_GENRES:
        assert genre in GENRE_NOTES
        assert GENRE_NOTES[genre].strip(), f"genre {genre!r} has empty note"


def test_glossary_hints_reach_the_prompt_when_provided():
    msg = build_user_message(
        "田中さん。", "variety", [],
        glossary_hints="Use these exact English forms for proper nouns: 田中=Tanaka.",
    )
    assert "田中=Tanaka" in msg


def test_glossary_hints_absent_when_not_provided():
    msg = build_user_message("こんにちは。", "variety", [])
    # Whatever hint phrasing the code uses, it must not appear when no hints are passed.
    assert "proper nouns:" not in msg
    assert "=" not in msg.split("Translate:")[0] or "Translate:" not in msg  # defensive


# --- series_context (translation.series_context_hint plumbing) --------------
# One prompt-ready line composed by gazetteer.series_profile.context_hint,
# rendered by build_user_message as its own "Series: <hint>" line between the
# genre note and the glossary hints (spec: SPEC-ANILIST-INTEGRATION-2026-07-10
# Component 3 -- NOT piggybacked on glossary_hints).


def test_series_context_renders_series_line_in_user_message():
    """Seeded defect: drop the Series-line append in build_user_message
    -> the line is absent -> red."""
    msg = build_user_message(
        "テスト。", "anime", [],
        series_context="Cyberpunk, Space; A hacker fights the system.",
    )
    assert "Series: Cyberpunk, Space; A hacker fights the system." in msg


def test_series_line_renders_between_genre_note_and_glossary_hints():
    """Position contract: genre note first, then the Series line, then the
    glossary hints. lines.index() also enforces own-line rendering (a hint
    merged into another line raises ValueError -> red).

    Seeded defect: append the Series line after glossary_hints (or before
    the genre note) -> the index ordering flips -> red.
    """
    hints = "Use these exact English forms for proper nouns: 田中=Tanaka."
    msg = build_user_message(
        "テスト。", "anime", [],
        glossary_hints=hints,
        series_context="Cyberpunk; A hacker fights the system.",
    )
    lines = msg.splitlines()
    note_i = lines.index(GENRE_NOTES["anime"])
    series_i = lines.index("Series: Cyberpunk; A hacker fights the system.")
    hints_i = lines.index(hints)
    assert note_i < series_i < hints_i, (
        f"Series line must sit between the genre note and the glossary "
        f"hints; got note={note_i}, series={series_i}, hints={hints_i}"
    )


def test_series_line_absent_when_series_context_empty():
    """An empty hint must render NO Series line at all (callers pass "" when
    the flag is off / no series / no profile -- a bare "Series:" stub would
    feed the model a meaningless steering line on every sentence).

    Seeded defect: render the line unconditionally (move the append out of
    the truthiness guard) -> a bare "Series: " appears -> red.
    """
    msg = build_user_message("テスト。", "anime", [])
    assert "Series:" not in msg
