"""Prompt templates, genre notes, and rolling context buffer."""
from __future__ import annotations

from collections import deque

# The literal label build_user_message prefixes to every transcript line.
# A separate constant because postprocess._LEADING_LABEL derives an echo-strip
# arm from it: the model has been observed echoing this exact label back as an
# output prefix ("Translate: Sato-nyaa?", field log 2026-07-09), and if the
# label ever rewords, a guard hardcoding the old word would silently reopen
# that gap. No colon here; the message template and the guard each add their
# own delimiter handling.
USER_MESSAGE_LABEL = "Translate"

# The literal label build_user_message puts before the earlier EN lines. One
# constant because SYSTEM_PROMPT_INSTRUCTIONS names it: without a rule saying
# what that section is, the model restated the previous line inside the new
# translation (live log 2026-11-02: it repeated the previous self-introduction
# in 13 of 13 tries), and a label reworded on one side only would silently
# reopen it.
CONTEXT_LABEL = "Previous"

# The instruction section is a separate constant because
# postprocess._PROMPT_VOCAB builds its prompt-echo vocabulary from it. The
# few-shot example OUTPUTS below are themselves ordinary short dialogue
# ("Nice weather today.", "Uh, yeah, that's a bit tricky."), so they must
# stay OUT of that vocabulary or the echo guard silently drops real
# dialogue that merely resembles them (issue #57's failure class).
# SYSTEM_PROMPT -- what the model actually sees -- is unchanged:
# instructions + examples, concatenated verbatim.
SYSTEM_PROMPT_INSTRUCTIONS = (
    "You are a subtitle translator. Do not use chain-of-thought reasoning.\n"
    "Translate the given Japanese TV dialogue to concise, natural English.\n"
    "Preserve tone and register.\n"
    "\n"
    "Output rules:\n"
    "- Output exactly one line of plain English, nothing else.\n"
    "- No quotation marks around the translation.\n"
    "- No parentheses or brackets containing stage directions, speaker labels, or commentary.\n"
    "- No stage directions such as \"(cheering)\" or \"*(laughs)*\".\n"
    "- No alternative translations or hedges such as \"or could be...\".\n"
    "- No preamble, no explanation, no apology.\n"
    f"- Lines under \"{CONTEXT_LABEL}:\" are earlier subtitles, already translated and shown. "
    f"Use them only to understand the \"{USER_MESSAGE_LABEL}:\" line. "
    "Never repeat, restate, or re-translate them.\n"
)

_FEW_SHOT_EXAMPLES = (
    "\n"
    "Examples (input then output, no labels in the real task):\n"
    "\u4eca\u65e5\u306f\u3044\u3044\u5929\u6c17\u3067\u3059\u306d\u3002\n"
    "Nice weather today.\n"
    "\n"
    "\u3048\u30fc\u3063\u3068\u3067\u3059\u306d\u3001\u305d\u308c\u306f\u3061\u3087\u3063\u3068\u96e3\u3057\u3044\u304b\u306a\u3002\n"
    "Uh, yeah, that's a bit tricky.\n"
)

SYSTEM_PROMPT = SYSTEM_PROMPT_INSTRUCTIONS + _FEW_SHOT_EXAMPLES

GENRE_NOTES = {
    "variety": "Fast informal speech. Preserve energy and reaction language. Sentence fragments are fine.",
    "drama": "Preserve politeness register (keigo vs. plain form). Full sentences preferred.",
    "news": "Formal register. Precise over natural.",
    "anime": "Preserve character speech patterns and affect. Drop honorifics. Keep sentence fragments as fragments.",
}


class ContextBuffer:
    def __init__(self, max_lines: int = 2):
        self._lines: deque[str] = deque(maxlen=max_lines)

    def push(self, english_line: str) -> None:
        if english_line.strip():
            self._lines.append(english_line.strip())

    def snapshot(self) -> list[str]:
        return list(self._lines)

    def clear(self) -> None:
        self._lines.clear()


def build_user_message(
    transcript: str, genre: str, context: list[str], glossary_hints: str = "",
    series_context: str = "",
) -> str:
    notes = GENRE_NOTES.get(genre, "")
    prev = "\n".join(context) if context else "(none)"
    lines = [f"Genre: {genre}"]
    if notes:
        lines.append(notes)
    if series_context:
        # Own line between the genre note and the glossary hints, never
        # merged into glossary_hints: the series hint steers domain register
        # while the glossary hints pin exact name spellings, and the model
        # must not read tag words as name candidates.
        lines.append(f"Series: {series_context}")
    if glossary_hints:
        lines.append(glossary_hints)
    lines.append(f"{CONTEXT_LABEL}: {prev}")
    lines.append(f"{USER_MESSAGE_LABEL}: {transcript}")
    return "\n".join(lines)


def fuzzy_suffix_dedup(prev: str, new: str, max_overlap: int = 12,
                       min_overlap: int = 3) -> str:
    """Strip leading N chars of `new` if they equal the trailing N chars of `prev`.

    Chunk-boundary artifacts sometimes re-emit the prior tail. Exact string
    matching misses cases where punctuation/whitespace normalization differs;
    here we just do a simple suffix/prefix overlap check on the raw strings.

    Overlaps shorter than ``min_overlap`` are ignored: a 1-2 char match
    between chunks is far more likely a coincidence (prev ends ね, next
    utterance starts ねえ — pervasive aizuchi) than a re-emitted tail, and
    stripping it silently corrupts both the JP subtitle and the translator
    input. Callers must additionally gate on temporal adjacency:
    only a force-split chunk boundary genuinely re-emits the prior tail;
    see asr_stage.
    """
    if not prev or not new:
        return new
    limit = min(max_overlap, len(prev), len(new))
    for k in range(limit, min_overlap - 1, -1):
        if prev.endswith(new[:k]):
            return new[k:]
    return new
