"""Post-process LLM translation output.

Handles the specific decorations qwen3-family translators (Qwen3.5-9B,
shisa-v2.1-qwen3-8b) emit despite the
"output only the translation" instruction:
  - wrapping in straight or smart quotes
  - trailing italicized stage directions: ``*(cheering)*``
  - trailing parenthetical asides: ``(cheering)``
  - multi-line bleed (previous line re-translated above the current)
  - trailing alternative translations: ``*(Depending on context, could...)*``
  - leading ``English:`` / ``Translation:`` label leaking from few-shot
    example format

Pure function. No network, no state.
"""
from __future__ import annotations

import re

from yamabiko.translation.prompt import (
    GENRE_NOTES,
    SYSTEM_PROMPT_INSTRUCTIONS,
    USER_MESSAGE_LABEL,
)

# The trailing "*( ... )*" group may not contain ")*": a greedy body spanned
# from the FIRST "*(" to the last ")*", so "*(laughs)* Really? *(sighs)*"
# lost the dialogue along with the trailing direction.
_TRAILING_ITALIC_PAREN = re.compile(r"\s*\*\((?:(?!\)\*).)*\)\*\s*$")
_TRAILING_PLAIN_PAREN = re.compile(r"\s+\([^)]*\)\s*$")
# shisa/qwen3 occasionally answers with translator meta-commentary INSTEAD of,
# or PREPENDED to, the translation -- e.g. "(Note: The translation captures the
# casual... tone...)" (beta testing 2026-06-16). _TRAILING_PLAIN_PAREN cannot
# catch it (it requires leading content), so the note reaches the overlay.
#
# We recognise it by its OPENER (_WHOLE_LINE_PAREN_META, below) and then remove
# the whole "(Note ...)" group by matching parens by DEPTH from the leading "("
# (_strip_leading_meta). An earlier version instead required the line to be a
# SINGLE balanced paren group (`^\([^()]*\)$`) before dropping it, which missed
# the two shapes the model actually emits, so the note shipped as a subtitle:
#   - TRUNCATION: a max-tokens cut leaves "(Note: ..." with NO closing ")", so
#     the single-group test failed (live digimon oid191 leak, 2026-07-05).
#   - NESTED parens: '(Note: ... ("shut up") ...)' carries internal pairs, so
#     the `[^()]` interior rejected it.
# Depth-matching handles both: a group that returns to depth 0 is stripped and
# the REMAINDER keeps cleaning (a leading note no longer swallows the real
# dialogue that follows it); a group that never balances means the whole tail
# IS the note, so the line cleans to "".
#
# This fires ONLY for a leading META opener. A line that opens with a paren but
# carries no meta signal -- off-screen / whispered dialogue the model wrapped
# despite the "no parentheses" instruction, e.g. "(Are you there?)" -- is left
# untouched and kept verbatim: a missing subtitle is a worse failure than a rare
# stage direction slipping through (issue #57).
#
# Meta-commentary openers, matched against the text just inside the leading
# "(". Kept deliberately narrow/precise: a false NEGATIVE only lets a meta line
# through as a (visible) subtitle, while a false POSITIVE silently drops real
# dialogue -- so we match only unambiguous label-like preambles.
_WHOLE_LINE_PAREN_META = re.compile(
    r"^\(\s*(?:"
    r"note\s*[:\-]"                                   # (Note: ...) / (Note - ...)
    r"|notes?\s+that\b"                               # (Note that ...)
    r"|translator(?:'|’)?s?\s+note\b"            # (Translator's note ...)
    r"|tl\s*note\b|tl\s*;?\s*dr\b"                    # (TL note ...) / (TL;DR ...)
    r"|translation\s+note\b"                          # (Translation note: ...)
    # Self-describing "style note" the model emits INSTEAD of the dialogue,
    # e.g. "(Concise and formal, matching the news tone. No added explanation
    # or alternatives.)" (live leak 2026-07-07). These arms stay label-like
    # and evidence-scoped so they cannot swallow a wholly-parenthetical
    # DIALOGUE line (see the negative-survival tests in test_postprocess.py):
    #  - a register self-label at line start is a signal ONLY when a
    #    translation-describing word follows it inside the same group, so
    #    "(Natural, isn't it?)" is left alone;
    #  - "matching the ... tone" and "no added <meta-noun>" are the two other
    #    observed openers, each anchored to a translation-describing tail so
    #    "(Matching outfits, aren't we?)" and "(No added sugar, please.)"
    #    survive verbatim.
    r"|(?:concise|formal|casual|literal|natural|polite|colloquial|idiomatic|neutral)\b"
    r"[^)]*?\b(?:translations?|tones?|phrasing|rendering|wording|nuances?|"
    r"connotations?|equivalents?|idioms?)\b"          # (Concise ... news tone ...)
    r"|matching\s+the\b[^)]*?\btone\b"                # (Matching the ... tone ...)
    r"|no\s+added\s+(?:explanations?|notes?|commentary|context|"
    r"information|detail\w*|alternativ\w*)"           # (No added explanation ...)
    r")",
    re.IGNORECASE,
)
# NOTE: bare "the/this translation ..." openers are deliberately NOT meta
# signals. They would drop a real wholly-parenthetical dialogue line such as
# "(This translation device is broken.)" -- the exact silent-loss failure this
# guard exists to prevent. The only empirically observed whole-line meta is the
# "(Note: ...)" family above (and "(Note: The translation captures...)" still
# matches via the note: arm). An un-caught bare meta line is shown as a visible
# subtitle, which the thesis above prefers over dropping real dialogue.
# One pattern per quote KIND, and the interior may not contain a
# further quote of the same kind. The old single pattern's greedy `(.*)`
# spanned from the FIRST quote to the LAST, so two-span dialogue
# ('"Stop!" he said. "Run!"') lost the outer quote of each span. A
# wrapper is only stripped when the whole line is one same-kind pair.
_SURROUNDING_STRAIGHT_QUOTES = re.compile(r'^\s*"([^"]*)"\s*$')
_SURROUNDING_SMART_QUOTES = re.compile(r"^\s*\u201c([^\u201c\u201d]*)\u201d\s*$")
# The last alternation arm is DERIVED from prompt.USER_MESSAGE_LABEL:
# build_user_message ends every user message with that literal label
# ("Translate: {transcript}"), and the model has been observed echoing it
# back as an output prefix (field log, 2026-07-09: "Translate: Sato-nyaa?").
# Deriving it means a prompt reword cannot silently reopen the gap -- the
# guard follows the prompt with no second edit. The static words are generic
# English output labels, not prompt-derived. The colon requirement keeps
# imperative dialogue ("Translate this for me, please.") out of scope.
_LEADING_LABEL = re.compile(
    r"^\s*(english|translation|translated|en|output|"
    + re.escape(USER_MESSAGE_LABEL.strip().rstrip(":").strip())
    + r")\s*:\s*",
    re.IGNORECASE,
)
# Meta-commentary preambles shisa occasionally emits despite instructions,
# e.g. "Here's the translation:" with nothing after (whole output is just
# the preamble). We strip the prefix; if nothing remains, clean() returns "".
# The noun must be followed by a ":"/"-" delimiter or end the line, so real
# dialogue that merely starts the same way ("This is my English teacher.")
# is left alone.
_PREAMBLE = re.compile(
    # Optional leading filler like "Okay, " / "Well, " / "Sure, " before the preamble.
    r"^\s*(?:(?:okay|ok|well|sure|alright|so|yes|yeah)[,:\s]+)?"
    r"(?:here(?:'|\u2019)?s|here\s+(?:is|are)|this\s+is)\s+(?:the|a|my)\s+"
    r"(?:translations?|english(?:\s+translations?)?|output)\s*(?:[:\-]\s*|$)",
    re.IGNORECASE,
)


# Prompt-echo discriminator. The regex arms above catch observed echo
# SHAPES; this catches echo CONTENT: a whole-line paren group whose
# content words are nearly all drawn from our own prompt/genre-note
# vocabulary describes the translation task, not the show. Function
# words are excluded from the overlap so dialogue built from common
# words ("(You are the one and only.)") cannot inflate the score, and
# a minimum content-word count keeps short whispered lines out of
# scope entirely.
_ECHO_STOPWORDS = frozenset(
    "a an the and or but of to in on at for with as is are am was were be "
    "been it its this that these those you your i we they he she do does "
    "did not no so such if then than there here what which who".split()
)
_ECHO_MIN_CONTENT_WORDS = 4
_ECHO_THRESHOLD = 0.8
# An echo restates the TASK, so it names it. The instruction vocabulary also
# holds everyday words (over, drop, nothing, else, keep, line), and dialogue
# built only from those ("(It's over. Drop it. Nothing else.)") scored as an
# echo. A group is an echo candidate only when some word starts with one of
# these task stems.
_ECHO_TASK_STEMS = (
    "translat", "subtitle", "honorific", "register", "keigo", "preamble",
    "dialogue",
)


def _stem(word: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    # Base and inflected forms must share a stem: suffix stripping alone
    # leaves the vocab's "translate"/"preserve" apart from the echo's
    # "translated"->"translat"/"preserving"->"preserv", so a trailing "e"
    # is dropped from every stem long enough to keep stems distinct.
    if word.endswith("e") and len(word) >= 4:
        word = word[:-1]
    return word


def _content_words(text: str) -> list[str]:
    return [
        _stem(w)
        for w in re.findall(r"[a-z']+", text.lower())
        if w not in _ECHO_STOPWORDS
    ]


# Built from the prompt's INSTRUCTION section only, never the full
# SYSTEM_PROMPT: the few-shot example outputs embedded in the prompt
# ("Nice weather today.", "Uh, yeah, that's a bit tricky.") are ordinary
# short dialogue, and folding them in put uh/yeah/bit/tricky/nice/weather/
# today into the vocabulary -- a real whispered line resembling the
# examples then scored as an echo and was silently dropped, the exact
# issue-#57 failure this file exists to avoid.
_PROMPT_VOCAB = frozenset(_content_words(
    SYSTEM_PROMPT_INSTRUCTIONS
    + " " + " ".join(GENRE_NOTES)
    + " " + " ".join(GENRE_NOTES.values())
))


def _is_prompt_echo(text: str) -> bool:
    """True when ``text`` (a whole-line paren group) is a restatement of
    our own prompt instructions."""
    if not any(
        w.startswith(_ECHO_TASK_STEMS) for w in re.findall(r"[a-z']+", text.lower())
    ):
        return False
    words = _content_words(text)
    if len(words) < _ECHO_MIN_CONTENT_WORDS:
        return False
    overlap = sum(1 for w in words if w in _PROMPT_VOCAB)
    return overlap / len(words) >= _ECHO_THRESHOLD


def _matching_paren_end(text: str) -> int | None:
    """Index of the ")" that closes the "(" at ``text[0]``.

    Returns None when the opener never returns to depth 0 -- i.e. a
    max-tokens truncation cut the closing paren off. Counts nested pairs so a
    note containing its own parens still matches at its OUTER close.

    Precondition: ``text`` starts with "(". On other input the first depth-0
    ")" does not correspond to ``text[0]``, so the result is meaningless;
    every caller must guarantee the leading "(" before calling.
    """
    assert text[:1] == "(", "caller must pass text starting with '('"
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _strip_leading_meta(text: str) -> str:
    """Remove a leading "(Note ...)" translator-note group; return the rest.

    While ``text`` OPENS with a meta signal, depth-match its leading "(":
      - balanced close -> drop that group and keep cleaning the remainder
        (looping so stacked notes are all removed);
      - never balances (truncated note) -> the whole tail is the note -> "".
    """
    while _WHOLE_LINE_PAREN_META.match(text):
        end = _matching_paren_end(text)
        if end is None:
            return ""
        text = text[end + 1:].strip()
    return text


def _strip_paren_meta(text: str) -> str:
    """Leading "(Note ...)" strip plus the whole-line prompt-echo strip.

    Shared by the main paren pass (steps 2b/2c) and the post-quote-unwrap
    re-pass: quotes can wrap a translator note OR a paraphrased prompt
    echo, so BOTH guards must re-run once the wrapper is gone. (A
    quote-wrapped paraphrase previously escaped because only
    _strip_leading_meta was re-applied after the unwrap.)
    """
    # Leading translator meta-commentary group ("(Note: ...)"), matched by
    # paren depth (see _strip_leading_meta / _WHOLE_LINE_PAREN_META).
    text = _strip_leading_meta(text)
    # Semantic echo guard: a WHOLE-LINE paren group (balanced to the last
    # char, or truncated with no close) whose content words are nearly all
    # prompt vocabulary is the model echoing its instructions. Runs below
    # the shape arms so the cheap regex path wins when it can; see
    # _is_prompt_echo for why real parenthetical dialogue cannot fire it.
    if text.startswith("("):
        end = _matching_paren_end(text)
        if (end is None or end == len(text) - 1) and _is_prompt_echo(text):
            return ""
    return text


def _strip_decorations(line: str) -> str:
    """Fully clean ONE line; returns "" when nothing but decoration remains.

    Ordering: italic-paren strip runs before plain-paren strip so that
    *(...)* is gone before we consider \\s+\\(...\\). A legitimate trailing
    parenthetical (e.g., "He scored 20 (a record).") will also be
    stripped -- this is a deliberate trade-off; decoration patterns
    are more common than content parens at end-of-subtitle.
    """
    # 1+2. Leading label ("Translation:") / preamble ("Here's the
    #      translation:") may consume the whole line.
    text = line.strip()
    if not text:
        return ""
    text = _LEADING_LABEL.sub("", text)
    text = _PREAMBLE.sub("", text).strip()
    if not text:
        return ""

    # 2b+2c. Strip a leading translator meta-commentary group ("(Note: ...)")
    #     by depth-matching its parens (see _strip_leading_meta and the module
    #     comment above _WHOLE_LINE_PAREN_META for the truncation / nested-paren
    #     failure modes this repairs), then apply the semantic prompt-echo
    #     guard to a WHOLE-LINE paren group (see _strip_paren_meta /
    #     _is_prompt_echo). A meta-only or echo-only line cleans to "" here; a
    #     meta note prepended to real dialogue keeps the dialogue. A line with
    #     no meta opener and no echo content is untouched -- off-screen
    #     dialogue the model wrapped in parens ("(Are you there?)") is kept
    #     verbatim (issue #57).
    text = _strip_paren_meta(text)
    if not text:
        return ""

    # 3. Strip trailing italic stage direction
    text = _TRAILING_ITALIC_PAREN.sub("", text)

    # 4. Strip trailing plain parenthetical *only if* preceded by whitespace
    #    (so mid-sentence "The team (Japan) won." is preserved)
    text = _TRAILING_PLAIN_PAREN.sub("", text)

    # 5. Strip one layer of surrounding quotes, kind-aware: only
    #    when the whole line is a single same-kind pair.
    m = _SURROUNDING_STRAIGHT_QUOTES.match(text) or _SURROUNDING_SMART_QUOTES.match(text)
    if m:
        # Quotes can wrap the whole meta note or a paraphrased prompt echo.
        # Once unquoted (and stripped, so the guards' line-start anchors see
        # the "("), run the full paren-meta pass again so '"(Note: ...)"',
        # '"(Concise ...)"', and '"(Translated the ...)"' do not ship.
        text = _strip_paren_meta(m.group(1).strip())

    return text.strip()


def clean(raw: str) -> str:
    if not raw or not raw.strip():
        return ""

    # Clean each line independently and keep the LAST line that retains
    # real content: multi-line bleed re-emits the PRIOR context
    # line ABOVE the current translation, so the current translation is
    # the last survivor. Per-line cleaning (not clean-after-select) also
    # covers own-line decorations: a preamble line ("Here's the
    # translation:") or a trailing hedge line ("*(Depending on context,
    # ...)*") strips to empty and is skipped instead of being selected.
    text = ""
    for line in raw.splitlines():
        candidate = _strip_decorations(line)
        if not candidate:
            continue
        if text and candidate.startswith("("):
            # Own-line trailing stage direction: a bare "(laughs)" BELOW the
            # translation survives per-line cleaning (no meta opener -- kept
            # verbatim per issue #57 -- and _TRAILING_PLAIN_PAREN requires
            # leading content on the SAME line), so last-line-wins would
            # DISPLACE the real translation with it ("That's amazing!\n
            # (laughs)" returned "(laughs)"). When an earlier line already
            # survived cleaning, drop a line that is ONE paren group with no
            # real words outside it: balanced (depth-matched), balanced with
            # only trailing punctuation ("(laughs)."), or truncated with no
            # close at all ("(laughs" from a max-tokens cut, the same shape
            # _strip_leading_meta already treats as junk). A line with real
            # words after the group ("(Wait!) He said (quietly)") is
            # multi-group dialogue and keeps last-line-wins. With NO
            # surviving predecessor the parenthetical is the only real
            # content and still ships verbatim -- issue #57's single-line
            # guarantee is untouched.
            end = _matching_paren_end(candidate)
            if end is None or not re.search(r"\w", candidate[end + 1:]):
                continue
        text = candidate

    return text
