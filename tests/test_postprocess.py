import pytest

from yamabiko.translation.postprocess import clean


class TestClean:
    def test_passes_through_plain_text(self):
        assert clean("The government announced a new policy.") == "The government announced a new policy."

    def test_strips_surrounding_straight_quotes(self):
        assert clean('"Great weather today."') == "Great weather today."

    def test_strips_surrounding_smart_quotes(self):
        assert clean('\u201cGreat weather today.\u201d') == "Great weather today."

    def test_preserves_internal_quotes(self):
        assert clean('He said "hello" to me.') == 'He said "hello" to me.'

    # --- The surrounding-quote regex must use a non-greedy match so that
    # two-span dialogue does not lose its outer quote of each span
    # ('”Stop!” he said. “Run!”' -> 'Stop!” he said. “Run!'). The wrapper
    # must only be stripped when the interior carries no further quote of
    # the same kind. Seeded defect: restore a greedy `[“”](.*)[“”]` pattern
    # -> these go red.

    def test_preserves_two_dialogue_spans_straight_quotes(self):
        raw = '"Stop!" he said. "Run!"'
        assert clean(raw) == raw

    def test_preserves_two_dialogue_spans_smart_quotes(self):
        raw = "“Stop!” he said. “Run!”"
        assert clean(raw) == raw

    def test_smart_wrapper_with_straight_interior_quotes_is_stripped(self):
        # Kind-aware: a smart-quote wrapper around straight interior quotes
        # is still a wrapper; only same-kind interior quotes block the strip.
        assert clean('“He said "hi" to me.”') == 'He said "hi" to me.'

    def test_strips_trailing_italic_stage_direction(self):
        raw = "The team won!  *(Cheering continues in the background.)*"
        assert clean(raw) == "The team won!"

    def test_strips_trailing_parenthetical_aside(self):
        raw = "The team won! (cheering in background)"
        assert clean(raw) == "The team won!"

    def test_preserves_mid_sentence_parenthetical(self):
        raw = "The team (Japan) won."
        assert clean(raw) == "The team (Japan) won."

    # --- Multi-line bleed puts the PRIOR context line first and the CURRENT
    # translation last (postprocess.py's own docstring: "previous line
    # re-translated above the current"). clean() must select the last
    # surviving line, not the first. Seeded defect: restore `break` on the
    # first surviving candidate in clean()'s line loop -> these return the
    # first line -> red.

    def test_collapses_multiline_to_last_nonempty_line(self):
        raw = "Line one.\nLine two.\nLine three."
        assert clean(raw) == "Line three."

    def test_multiline_bleed_keeps_current_translation_not_stale_prior_line(self):
        assert clean("Nice weather today.\nIt sure is.") == "It sure is."

    def test_trailing_blank_lines_do_not_blank_result(self):
        assert clean("It sure is.\n\n") == "It sure is."

    def test_own_line_trailing_hedge_does_not_drop_translation(self):
        # The hedge decoration can land on its OWN line below the
        # translation; it must strip to nothing and be skipped, not get
        # selected as the "last line" and blank the subtitle.
        raw = 'That\'s tough.\n*(Depending on context, could also be: "That\'s a lot.")*'
        assert clean(raw) == "That's tough."

    def test_skips_blank_leading_lines(self):
        raw = "\n\nReal line."
        assert clean(raw) == "Real line."

    def test_strips_alternative_translation_hedge(self):
        raw = 'That\'s tough.  *(Depending on context, could also be: "That\'s a lot.")*'
        assert clean(raw) == "That's tough."

    def test_handles_empty_string(self):
        assert clean("") == ""

    def test_handles_whitespace_only(self):
        assert clean("   \n\n  ") == ""

    def test_strips_trailing_italic_with_nested_parens(self):
        raw = "She left.  *(See note (subsection a) for details.)*"
        assert clean(raw) == "She left."

    def test_trailing_stage_direction_strip_keeps_earlier_direction_and_dialogue(self):
        # Two italic directions on one line: only the trailing one is
        # decoration. Seeded defect: a greedy `.*` inside the trailing-italic
        # pattern spans from the first "*(" to the last ")*" and blanks the
        # whole subtitle.
        assert clean("*(laughs)* Really? *(sighs)*") == "*(laughs)* Really?"

    def test_preserves_leading_parenthetical(self):
        # A leading stage direction is left alone — not necessarily
        # decoration (could be legitimate content); only trailing ones
        # are stripped.
        raw = "(laughs) She said yes."
        assert clean(raw) == "(laughs) She said yes."

    # --- The translator occasionally emits translator meta-commentary
    # (e.g. "(Note: The translation captures the casual... tone...)")
    # INSTEAD of, or PREPENDED to, the translation. When a line OPENS with a
    # meta signal ("(Note:", "(Translator's note", "(Translation note", ...)
    # we depth-match the leading "(" and strip that whole "(Note ...)" group:
    #   - group closes cleanly -> strip it and keep cleaning the REMAINDER,
    #     so "(Note: casual tone) \"Matte yo!\"" yields the real dialogue;
    #   - group never closes (max-tokens truncation) -> the whole tail is the
    #     note -> the line cleans to "".
    # A line that does NOT open with a meta signal is left for the trailing
    # strips, so a real off-screen / whispered line the model wrapped in
    # parens ("(Are you there?)") is kept verbatim -- silently dropping a
    # real subtitle line is the worse failure (issue #57). Seeded defect:
    # gate the meta strip behind the old single-group `_WHOLE_LINE_PAREN`
    # test again -> the truncated / nested / leading-meta cases below go red.

    def test_strips_whole_line_meta_commentary_note(self):
        raw = (
            "(Note: The translation captures the casual, slightly exasperated "
            "tone of the original Japanese, including the implied reaction to "
            "the speaker's outfit. The phrase is rendered as a natural, "
            "conversational English equivalent.)"
        )
        assert clean(raw) == ""

    def test_keeps_translation_when_meta_paren_on_own_line(self):
        raw = "Speaking of summer clothes, huh?\n(Note: preserves the teasing tone.)"
        assert clean(raw) == "Speaking of summer clothes, huh?"

    def test_strips_whole_line_translation_note_preamble(self):
        # A meta signal other than "Note:" must also drop.
        # Seeded defect: drop "translation note" from _WHOLE_LINE_PAREN_META
        # -> this survives as a subtitle -> red.
        assert clean("(Translation note: the phrase uses keigo.)") == ""

    def test_preserves_wholly_parenthetical_dialogue(self):
        # The reported bug: a real line of dialogue the model wrapped in
        # parens was silently dropped (clean -> ""), so the user saw NO
        # subtitle for that utterance. It must survive (issue #57).
        # Seeded defect: revert to unconditional whole-line-paren drop ->
        # clean returns "" -> red.
        assert clean("(Are you there?)") == "(Are you there?)"

    def test_preserves_wholly_parenthetical_multiword_dialogue(self):
        raw = "(Don't tell anyone, but I saw them.)"
        assert clean(raw) == raw

    def test_preserves_wholly_parenthetical_non_meta_line(self):
        # Policy change (issue #57): a non-meta whole-line paren (e.g. a bare
        # stage direction) is now KEPT, not silently dropped, because it
        # cannot be reliably distinguished from off-screen dialogue and
        # silent loss is the worse failure.
        assert clean("(cheering)") == "(cheering)"

    def test_preserves_translation_lookalike_dialogue(self):
        # A wholly-parenthetical dialogue line that merely OPENS with
        # "this/the translation" must NOT be mistaken for translator
        # meta-commentary and dropped -- the meta signals are label-like
        # ("Note:", "Translator's note", "Translation note"), not a bare
        # "the/this translation" prefix (which a real line can carry).
        # Seeded defect: add `this\\s+translation\\b` back to
        # _WHOLE_LINE_PAREN_META -> this line is dropped to "" -> red.
        raw = "(This translation device is broken.)"
        assert clean(raw) == raw

    def test_meta_opener_plus_dialogue_on_same_line_keeps_dialogue(self):
        # A line that OPENS with a meta note but then carries real dialogue:
        # depth-matching the leading "(" closes "(Note: casual tone)" at its
        # own ")", so only that group is stripped and the remainder is cleaned
        # normally -- the trailing " (shouted)" pass and the surrounding-quote
        # pass then leave just the dialogue. The old code showed the note
        # verbatim; stripping it and keeping the dialogue resolves the case
        # the design comment called "the exact issue-#57 failure". Seeded
        # defect: gate the meta strip behind the single-group `_WHOLE_LINE_PAREN`
        # test again -> the leading meta paren is no longer stripped and the
        # note ships in the subtitle -> red.
        raw = '(Note: casual tone) "Matte yo!" (shouted)'
        assert clean(raw) == "Matte yo!"

    def test_strips_meta_note_with_nested_parens(self):
        # A whole-line meta note that contains INTERNAL parens still balances
        # by depth from the leading "(", so the whole "(Note ...)" group is
        # stripped and the line cleans to "". The old single-group regex
        # (`^\([^()]*\)$`) could not see past the nested parens and shipped
        # the note as a subtitle. Seeded defect: gate the meta strip behind
        # `_WHOLE_LINE_PAREN` again -> the nested note survives -> red.
        assert clean("(Note: casual tone (informal))") == ""

    def test_strips_meta_note_with_nested_parens_and_trailing_text(self):
        # Depth-match must consume BOTH parens of the nested group before it
        # sees the real closing ")", otherwise it would stop early inside the
        # note. Seeded defect: match on the FIRST ")" instead of by depth ->
        # the outer note text leaks into the subtitle -> red.
        assert clean("(Note: blah (nested) blah)") == ""

    def test_strips_truncated_meta_note_oid191_fixture(self):
        # Live leak from the 2026-07-05 judged eval (digimon oid191): a
        # max-tokens-truncated "(Note: ...)" with NO closing paren AND an
        # internal ("shut up") pair. The old single-group regex needed one
        # balanced whole-line paren, so it matched neither condition and the
        # note shipped to the overlay. Depth-match sees the leading "(" never
        # returns to depth 0 -> the whole tail is the note -> "". Seeded
        # defect: gate the meta strip behind `_WHOLE_LINE_PAREN` again -> the
        # verbatim note is shown -> red.
        raw = (
            '(Note: "Tomoro" is an informal, dismissive phrase, and '
            '"黙るのは" ("shut up") is naturally translated as '
            '"You\'re the one who\'s talking" to match the'
        )
        assert clean(raw) == ""

    def test_strips_leading_meta_then_keeps_remainder_dialogue(self):
        # The clean case-3 remainder: a closed "(Note: ...)" group followed by
        # a quoted line. The group is stripped, then the existing
        # surrounding-quote pass unwraps the dialogue. Seeded defect: gate the
        # meta strip behind `_WHOLE_LINE_PAREN` again -> the note is not
        # stripped and the whole line (note included) ships -> red.
        assert clean('(Note: casual tone) "Matte yo!"') == "Matte yo!"

    def test_strips_truncated_meta_note_no_nested_parens(self):
        # A "(Note: ...)" that runs off the end of the token budget with no
        # nested parens at all: depth never returns to 0 -> "". Seeded defect:
        # treat "never balances" as "no meta group, keep text" -> the partial
        # note is shown -> red.
        assert clean("(Note: the tone here is casual and") == ""

    def test_strips_stacked_leading_meta_groups(self):
        # If the remainder AFTER stripping one meta group opens with ANOTHER
        # meta group, the loop strips that too before falling through to the
        # trailing strips. Seeded defect: strip only once (drop the loop) ->
        # the second "(Note ...)" survives as the subtitle -> red.
        raw = '(Translation note: keigo) (Note: teasing tone) "Ohayou!"'
        assert clean(raw) == "Ohayou!"

    def test_multiline_truncated_meta_note_last_line_keeps_real_translation(self):
        # Multi-line output where a truncated meta note lands on the LAST line
        # BELOW the real translation. clean()'s last-survivor rule would pick
        # the note line unless the note strips to "". The old code left the
        # truncated (unbalanced) note intact, so it became the selected
        # subtitle and the real translation was lost. Seeded defect: gate the
        # meta strip behind `_WHOLE_LINE_PAREN` again -> the truncated note is
        # selected as the last survivor -> red.
        raw = "Speaking of summer clothes, huh?\n(Note: the tone here is casual and"
        assert clean(raw) == "Speaking of summer clothes, huh?"

    def test_strips_then_unquotes(self):
        # Quote-wrapped translation on the LAST line: line selection
        # happens first, then the surrounding-quote wrapper is stripped.
        raw = 'Line one.\n"Line two."'
        assert clean(raw) == "Line two."

    def test_strips_leading_english_label(self):
        assert clean("English: Hello there.") == "Hello there."

    def test_strips_leading_translation_label_case_insensitive(self):
        assert clean("Translation:  The team won.") == "The team won."

    def test_does_not_strip_label_like_word_mid_sentence(self):
        assert clean("He said English is hard.") == "He said English is hard."

    def test_strips_leading_translate_label(self):
        """Field regression: the model echoed the prompt's own literal
        "Translate: " label back as the output prefix ('Translate: Sato-nyaa?'),
        observed once in 1421 field utterances. prompt.build_user_message ends
        every user message with '{USER_MESSAGE_LABEL}: {transcript}', so this
        echo shape recurs.

        Seeded-defect check: drop the derived USER_MESSAGE_LABEL arm from
        _LEADING_LABEL -> red (the word is no longer in the static list)."""
        assert clean("Translate: Sato-nyaa?") == "Sato-nyaa?"

    def test_leading_label_guard_tracks_the_prompts_actual_label(self):
        """The echo-strip arm is DERIVED from prompt.USER_MESSAGE_LABEL, so a
        prompt reword can never silently reopen the echo gap: whatever label
        build_user_message actually sends is whatever this guard strips.

        Review finding 2026-07-12: the first fix hardcoded 'translate' next to
        a _PROMPT_VOCAB guard that was already prompt-derived; a reword of the
        prompt label would have left the hardcoded guard behind with no import,
        no shared constant, and no failing test.

        Seeded-defect check: remove the derived arm from _LEADING_LABEL (or
        hardcode a word different from the prompt's label) -> red."""
        from yamabiko.translation.prompt import USER_MESSAGE_LABEL

        echoed = f"{USER_MESSAGE_LABEL}: The team won."
        assert clean(echoed) == "The team won."

    def test_keeps_leading_translate_imperative_without_colon(self):
        """Negative survival: 'translate' as real leading dialogue (no colon)
        must not be stripped -- the label match requires the colon."""
        assert clean("Translate this for me, please.") == (
            "Translate this for me, please."
        )

    def test_leading_label_then_quote(self):
        assert clean('English: "Hello there."') == "Hello there."

    # --- Own-line preamble/label must not drop the whole translation ---
    # The translator occasionally emits the preamble on its OWN line with
    # the real translation on the next line. clean() must strip the preamble
    # before collapsing lines, otherwise line 1 strips to "" and the
    # translation on line 2 is silently dropped (main.py then logs "dropped
    # empty EN" and the user sees nothing for that sentence).

    def test_own_line_preamble_keeps_translation_on_next_line(self):
        """Seeded-defect: revert clean() to collapse-then-strip -> line 1
        ("Here's the translation:") strips to "" and the real text is lost."""
        assert clean("Here's the translation:\nThe team won the championship.") == (
            "The team won the championship."
        )

    def test_own_line_translation_label_keeps_next_line(self):
        assert clean("Translation:\nNice to meet you.") == "Nice to meet you."

    def test_own_line_english_label_keeps_next_line(self):
        assert clean("English:\nGood morning everyone.") == "Good morning everyone."

    def test_same_line_translation_still_works_after_reorder(self):
        # Guard the reorder didn't regress the common same-line case.
        assert clean("Translation: The team won.") == "The team won."

    # --- The preamble regex must match the uncontracted "here is" / "here
    # are" forms as well as "here's". Without them the per-line loop picks
    # the preamble line as the translation and the real translation on the
    # next line is dropped. Seeded defect: remove the
    # `here\s+(?:is|are)` branch from _PREAMBLE.

    def test_here_is_preamble_own_line_keeps_translation_on_next_line(self):
        assert clean("Here is the translation:\nGood morning everyone.") == (
            "Good morning everyone."
        )

    def test_filler_then_here_is_preamble_keeps_translation_on_next_line(self):
        assert clean("Sure, here is the translation:\nGood morning everyone.") == (
            "Good morning everyone."
        )

    def test_here_is_preamble_same_line_stripped(self):
        assert clean("Here is the translation: Good morning everyone.") == (
            "Good morning everyone."
        )

    def test_here_are_plural_preamble_own_line_keeps_translation(self):
        assert clean("Here are the translations:\nGood morning.") == "Good morning."

    def test_here_is_mid_sentence_not_stripped(self):
        # "here is" must only strip as a leading preamble, never mid-content.
        assert clean("The answer here is simple.") == "The answer here is simple."

    def test_this_is_my_english_dialogue_not_stripped_as_preamble(self):
        # The preamble needs a delimiter or end of line after its noun;
        # without one it ate real dialogue down to "teacher.". Seeded
        # defect: make the delimiter after the noun optional again.
        assert clean("This is my English teacher.") == "This is my English teacher."

    def test_here_are_english_translations_plural_leaves_no_residue(self):
        # `english(?:\s+translation)?` admits only the singular, so the
        # plural matches through "translation" and ships a garbage "s:"
        # line as the subtitle. Seeded defect: drop the s? from the
        # english branch.
        assert clean("Here are the English translations:\nGood morning.") == (
            "Good morning."
        )

    # --- Self-describing "style-note" leak (live capture 2026-07-07) ---
    # shisa sometimes emits a line that DESCRIBES its own translation (the
    # register it chose, what it kept, what it dropped) INSTEAD of the
    # dialogue, wholly wrapped in parens with no leading content:
    #   "(Concise and formal, matching the news tone. No added explanation
    #    or alternatives.)"
    # Because the whole line is one paren group with nothing before the "(",
    # _TRAILING_PLAIN_PAREN (which requires leading content) cannot see it,
    # and the note shipped to the overlay as a subtitle. _WHOLE_LINE_PAREN_META
    # now recognises these self-describing shapes by their OPENER and the
    # existing depth-matching strip removes the whole group.
    #
    # The arms are deliberately label-like and evidence-scoped: a register
    # self-label ("Concise"/"Formal"/...) is a meta signal ONLY when a
    # translation-describing word follows it in the same group, and "no added"
    # is a signal only before a meta noun. This is the issue-#57 thesis: a
    # false NEGATIVE merely lets a note show as a (visible) subtitle, while a
    # false POSITIVE silently DROPS real parenthetical dialogue, which is the
    # worse failure -- so the negative-survival tests below are load-bearing.

    def test_strips_self_describing_style_note_leak(self):
        # The exact captured leak. Seeded defect: remove the register-label
        # arm from _WHOLE_LINE_PAREN_META -> the style note ships verbatim.
        raw = (
            "(Concise and formal, matching the news tone. "
            "No added explanation or alternatives.)"
        )
        assert clean(raw) == ""

    def test_strips_quote_wrapped_self_describing_style_note_leak(self):
        # Some model outputs wrap the whole parenthetical note in quotes. The
        # quote strip must not expose the note after the meta-strip pass has
        # already run. Seeded defect: strip quotes only after leading meta and
        # do not re-run meta stripping -> the note ships as a subtitle.
        raw = (
            '"(Concise and formal, matching the news tone. '
            'No added explanation or alternatives.)"'
        )
        assert clean(raw) == ""

    def test_strips_quote_wrapped_note_meta_leak(self):
        # Same ordering bug for the older (Note: ...) family.
        # Seeded defect: quote-unwrapped meta is not reprocessed -> red.
        raw = '"(Note: The translation captures the formal tone.)"'
        assert clean(raw) == ""

    def test_strips_register_label_with_translation_descriptor(self):
        # register self-label ("Casual") + a translation-describing word
        # ("phrasing"/"tone") later in the group. Seeded defect: drop the
        # descriptor requirement's word list -> still passes, but drop the
        # whole register arm -> red.
        assert clean("(Casual and natural phrasing to match the original tone.)") == ""

    def test_strips_matching_the_tone_opener(self):
        # A note that opens directly with "Matching the ... tone". Seeded
        # defect: remove the "matching the ... tone" arm -> red.
        assert clean("(Matching the formal tone of the original.)") == ""

    def test_strips_no_added_meta_opener(self):
        # A note that opens with "No added <meta-noun>". Seeded defect:
        # remove the "no added" arm -> red.
        assert clean("(No added explanation.)") == ""

    # Negative-survival: legitimate wholly-parenthetical DIALOGUE the model
    # wrapped in parens must be kept verbatim. Each of these would go RED if
    # the corresponding new arm were widened past its label-like scope.

    def test_preserves_question_dialogue_in_parens(self):
        assert clean("(Are you there?)") == "(Are you there?)"

    def test_preserves_cold_dialogue_in_parens(self):
        assert clean("(It's cold...)") == "(It's cold...)"

    def test_preserves_whos_there_dialogue_in_parens(self):
        assert clean("(Who's there?)") == "(Who's there?)"

    def test_preserves_register_word_dialogue_without_descriptor(self):
        # "Natural" opens the line but NO translation-describing word follows,
        # so the register-label arm must not fire. Seeded defect: make the
        # register arm fire on the bare register word (drop the required
        # descriptor group) -> this real line is dropped to "" -> red.
        assert clean("(Natural, isn't it?)") == "(Natural, isn't it?)"

    def test_preserves_no_added_dialogue_without_meta_noun(self):
        # "No added sugar" is dialogue, not a translator note. The "no added"
        # arm is scoped to meta nouns. Seeded defect: widen the arm to a bare
        # "no\\s+added" -> this real line is dropped to "" -> red.
        assert clean("(No added sugar, please.)") == "(No added sugar, please.)"

    def test_preserves_matching_dialogue_without_tone(self):
        # "Matching outfits" has no "the ... tone", so the matching arm must
        # not fire. Seeded defect: drop the "\\btone\\b" tail from the matching
        # arm -> this real line is dropped to "" -> red.
        assert clean("(Matching outfits, aren't we?)") == "(Matching outfits, aren't we?)"

    # --- Own-line trailing stage direction (P2, 2026-07-12 review) ---------
    # A bare `(laughs)` on its OWN line below the translation survives
    # per-line cleaning: it has no meta opener (kept verbatim per issue
    # #57) and _TRAILING_PLAIN_PAREN requires leading whitespace, so it
    # never fires on a line that STARTS with "(". Under last-line-wins it
    # then DISPLACED the real translation ("That's amazing!\n(laughs)"
    # returned "(laughs)"). clean() now drops a whole-line bare
    # parenthetical when a PRECEDING line survived cleaning -- while a
    # single-line (or no-surviving-predecessor) non-meta parenthetical
    # still passes through verbatim, preserving issue #57's guarantee.

    def test_own_line_trailing_stage_direction_does_not_displace_translation(self):
        # Seeded defect: revert clean() to unconditional last-line-wins ->
        # returns "(laughs)" -> red.
        assert clean("That's amazing!\n(laughs)") == "That's amazing!"

    def test_own_line_trailing_multiword_stage_direction_dropped(self):
        assert clean("The team won!\n(cheering in background)") == "The team won!"

    def test_single_line_bare_parenthetical_still_passes_verbatim(self):
        # Issue #57's guarantee must survive the own-line fix: with no
        # other line, the parenthetical IS the subtitle. Seeded defect:
        # drop the "a preceding line survived" condition (drop every
        # whole-line paren group) -> this returns "" -> red.
        assert clean("(That's the plan)") == "(That's the plan)"

    def test_paren_line_kept_when_preceding_lines_all_strip_empty(self):
        # Blank / decoration-only lines above are NOT surviving content, so
        # the parenthetical is still the only real line and must ship
        # (issue #57). Seeded defect: count raw lines instead of SURVIVING
        # lines when deciding to drop -> both go "" -> red.
        assert clean("\n(Are you there?)") == "(Are you there?)"
        assert clean("Here's the translation:\n(Are you there?)") == (
            "(Are you there?)"
        )

    def test_paren_line_kept_when_predecessor_was_meta_note(self):
        # A stripped "(Note: ...)" line is not a survivor either -- the
        # paren dialogue below it is the only real content.
        assert clean("(Note: whispered.)\n(Are you there?)") == "(Are you there?)"

    def test_multi_group_paren_line_is_not_a_bare_parenthetical(self):
        # After the quote unwrap, this line opens with "(" and ends with ")"
        # but is NOT one whole-line paren group (the first group closes
        # mid-line) -- it is dialogue, so it keeps last-line-wins. (The
        # quote-wrapped shape is the reachable one: an UNQUOTED trailing
        # " (quietly)" would already be stripped as trailing decoration.)
        # Seeded defect: test only startswith/endswith instead of
        # depth-matching the whole line -> dropped -> red.
        raw = 'Prior line.\n"(Wait!) He said (quietly)"'
        assert clean(raw) == "(Wait!) He said (quietly)"

    def test_own_line_truncated_stage_direction_does_not_displace_translation(self):
        # A max-tokens cut can lose the closing paren of the trailing stage
        # direction; the unbalanced "(laughs" fragment is still decoration
        # and must not displace the translation. Seeded defect: require a
        # balanced group before dropping -> returns "(laughs" -> red.
        assert clean("That's amazing!\n(laughs") == "That's amazing!"

    def test_own_line_stage_direction_with_trailing_punct_dropped(self):
        # A balanced group followed only by punctuation is still a bare
        # stage direction, not dialogue. Seeded defect: require the group to
        # close exactly at the last char -> returns "(laughs)." -> red.
        assert clean("That's amazing!\n(laughs).") == "That's amazing!"

    def test_final_line_with_words_outside_paren_group_keeps_last_line_wins(self):
        # Real words OUTSIDE the leading paren group mark the line as
        # dialogue; it must never be dropped. Seeded defect: drop every
        # final line that merely OPENS with "(" -> returns "Prior line." ->
        # red.
        assert clean("Prior line.\n(Wait!) That was close.") == (
            "(Wait!) That was close."
        )

    def test_single_line_unbalanced_parenthetical_still_passes_verbatim(self):
        # The survivor precondition covers the unbalanced arm too: with no
        # earlier surviving line, even a truncated parenthetical is the only
        # real content and ships verbatim (issue #57). Seeded defect: drop
        # the precondition from the unbalanced arm -> returns "" -> red.
        assert clean("(Are you there") == "(Are you there"


class TestPromptEchoGuard:
    # The regex arms in _WHOLE_LINE_PAREN_META catch observed echo SHAPES;
    # the semantic guard (_is_prompt_echo) catches echo CONTENT: a whole-line
    # paren group whose content words are nearly all drawn from our own
    # SYSTEM_PROMPT / GENRE_NOTES vocabulary is the model restating its
    # instructions, not dialogue.

    # Positive: paraphrased prompt echo, NOT covered by any regex arm's
    # exact shape, must strip. Includes the live 2026-07-07 GITS leak
    # (latency.log:262). Seeded defect: flip the overlap comparison to ">"
    # and raise _ECHO_THRESHOLD to 1.1 so the guard never fires -> the two
    # paraphrase rows (which no regex arm matches) ship verbatim -> red.
    @pytest.mark.parametrize("echo", [
        "(Concise, natural English, preserving the tone and register of anime dialogue.)",
        "(Translated the Japanese dialogue to natural English, preserving register.)",
        "(Preserving the tone and register of the dialogue in plain English.)",
    ])
    def test_prompt_paraphrase_echo_strips(self, echo):
        assert clean(echo) == ""

    # Negative survival: parenthetical DIALOGUE whose words are not prompt
    # vocabulary must ship verbatim (the issue #57 thesis: dropping real
    # dialogue is the worse failure). None of these names the translation
    # task, so the task-word anchor keeps them; the seeded defects below
    # apply with that anchor removed. Seeded defect: remove the stopword
    # exclusion from _content_words -> "(You are the one and only.)" scores
    # 5/6 prompt-vocab function words and is dropped to "" -> red; zeroing
    # _ECHO_MIN_CONTENT_WORDS plus _ECHO_THRESHOLD drops the others -> red.
    @pytest.mark.parametrize("line", [
        "(Are you there?)",
        "(Don't let them hear us... the guards are right outside.)",
        "(You are the one and only.)",   # function-word overlap trap
        "(No added sugar, please.)",
    ])
    def test_parenthetical_dialogue_survives(self, line):
        assert clean(line) == line

    # Dialogue whose words all happen to appear in the prompt instructions
    # ("over", "drop", "nothing", "else") but that never names the
    # translation task. Seeded defect: drop the task-word anchor from
    # _is_prompt_echo -> 4/5 content words overlap -> dropped to "" -> red.
    def test_dialogue_without_task_words_survives(self):
        line = "(It's over. Drop it. Nothing else.)"
        assert clean(line) == line

    # The vocabulary must come from the prompt's INSTRUCTION section only.
    # SYSTEM_PROMPT's few-shot EXAMPLE outputs are themselves ordinary short
    # dialogue ("Nice weather today.", "Uh, yeah, that's a bit tricky.");
    # folding them into _PROMPT_VOCAB put uh/yeah/bit/tricky/nice/weather/
    # today into the vocabulary, so a real whispered line that merely
    # resembles the examples scored as an echo and was silently dropped --
    # the exact issue-#57 failure the rest of this file is built to avoid.
    # Seeded defect (with the task-word anchor removed): build _PROMPT_VOCAB
    # from full SYSTEM_PROMPT instead of SYSTEM_PROMPT_INSTRUCTIONS -> every
    # row here overlaps >= 0.8 via the example-sentence words and drops to
    # "" -> red.
    @pytest.mark.parametrize("line", [
        "(Uh, yeah, that's a bit tricky.)",        # example output, verbatim
        "(Uh, yeah, that's a bit tricky today.)",  # paraphrase over min words
        "(Nice weather today, huh? Bit tricky.)",  # blend of both examples
    ])
    def test_dialogue_resembling_prompt_examples_survives(self, line):
        assert clean(line) == line

    # Quote-wrapped paraphrase echo: quotes can wrap the whole echo just as
    # they wrap the (Note: ...) / (Concise ...) regex families (documented
    # precedent at the quote-unwrap step). The semantic guard must re-run on
    # the unwrapped text or the wrapper defeats it entirely. Seeded defect:
    # re-run only _strip_leading_meta (not the echo guard) after the quote
    # strip -> the unquoted paraphrase ships as a visible subtitle -> red.
    @pytest.mark.parametrize("echo", [
        '"(Translated the Japanese dialogue to natural English, preserving register.)"',
        "“(Preserving the tone and register of the dialogue in plain English.)”",
    ])
    def test_quote_wrapped_paraphrase_echo_strips(self, echo):
        assert clean(echo) == ""

    def test_quote_wrapped_parenthetical_dialogue_survives_unwrap(self):
        # The post-unwrap re-pass must not eat quote-wrapped DIALOGUE in
        # parens: zero prompt-vocab overlap keeps the guard silent and the
        # paren line ships (issue #57). Seeded defect: make the re-pass drop
        # every whole-line paren group (skip _is_prompt_echo) -> red.
        raw = '"(Don\'t let them hear us... the guards are right outside.)"'
        assert clean(raw) == "(Don't let them hear us... the guards are right outside.)"
