"""Select the pilot's training items and write the T1 allow-lists (local).

    python -m scripts.asr_finetune.select [--s1 <dir>] [--extra-teacher <dir> ...]
        [--reazon-hours H] [--cv-hours H] [--galgame-hours H] [--outbox <dir>]
        [--reazon-source {official,mirror}] [--loop-hygiene] [--adult-filter] [--vocal-cap] [--nonspeech]

The targets of the drawn sources default to 50, 20 and 45 h; a target of 0
leaves the source out (no allow-list entry, nothing in the fetch plan).
``--outbox`` writes the two outbox files to another folder (default
``<pilot>\\outbox``), e.g. a fallback built beside the current one; its
report is ``selection-report.<folder name>.json``. ``--reazon-source``
(default ``official``, the corpus server) chooses where T1 downloads
ReazonSpeech (``pilot.REAZON_ORIGINS``): it changes ``fetch-t1.json`` only,
never the selection or ``allow.json``.

Reads what pod S1 brought back (``--s1``, default ``<pilot>\\pods\\s1``):
``stage\\stage.json`` with the rows files beside it, and
``teacher\\teacher.jsonl`` with ``teacher\\teacher.json``; then each
``--extra-teacher`` folder (a later teacher run, e.g. ``<pilot>\\pods\\s1b\\teacher``,
which must hold a ``teacher.jsonl``) in the order given. A clip takes the
first teacher line of its key, so S1 wins and an earlier extra run wins over
a later one (``superseded`` counts the later lines). Also the plain private
hygiene lists ``<pilot>\\hygiene\\bench-lines.json`` and
``cv-exclusion.json``; and ``<pilot>\\outbox\\stage-plan.json`` (the archive
sizes, and the plan order that names archives ``galgame-NN``). A source that
``stage.json`` lists in ``failed_sources`` contributes nothing, whatever rows
file it left behind; so does a source S1 did not stage, and ReazonSpeech
when no run has a ``teacher.jsonl``.

Every text goes through ``pilot.clean_reason`` (the label rule of every
training text: markup, empty, galgame vocalization-only without a lexicon
interjection, galgame and JNV ``PHANTOM_FAMILY``), with ``--loop-hygiene``
(round 4) through ``pilot.loop_reason`` on its cleaned label (``loop_run``: a
run of 4 identical characters, punctuation aside; ``loop_ratio``: a zlib
ratio over 2.0), with ``--adult-filter`` (round 4, galgame lines) through
``pilot.adult_reason`` (``adult_sfx``: sound-effect vocabulary, with
``--vocal-cap`` also the round-6 words; ``adult_moan``: moan-dense), with
``--vocal-cap`` (round 6, every source) through ``pilot.vocal_reason``
(``long_vocal_total``, ``_dense``, ``_chunk``, ``_repeat``: long vocal
chains), and through rule 4.2
(``pilot.BenchLineFilter`` on the bench lines, Common Voice included); every
id that reaches a pod (``carried``: the allow-list id, and a galgame line's
``<archive>|<voice>`` window-plan id) must pass the outbox text rule
(``outbox.text_finding``), else the item is dropped as ``outbox_title`` (it
holds a bench, locked or franchise title or alias) or ``outbox_text`` (a
shingle of a bench line: a chance id collision), the last reasons checked. A
dropped item is counted under its first reason, per source:

- ReazonSpeech (``reazon-rows.jsonl``), the teacher filter of survey 5.2,
  keeping a clip only when (a) the multiset of lexicon keys
  (``scripts.asr_bench.lexicon.analyze``) of the teacher text minus that of
  the caption is empty, (b) no teacher reading character follows the
  alignment position of the caption's last reading character
  (``scripts.asr_bench.align.alignment_trace`` on the two readings), (c) the
  teacher outcome is ``text_available``, (d) ``BenchLineFilter`` misses the
  caption, (e) the caption leaves out few of the teacher's reading
  characters: of the per-teacher-character match flags of
  ``alignment_trace(teacher reading, caption reading)``, at most
  ``MAX_CAPTION_DELETED_KANA`` (2) are unmatched and at most
  ``MAX_LEADING_DELETED_KANA`` (1) come before the first matched one (with
  none matched, the whole teacher reading counts as leading; captions that
  omit the tail of the previous line are the usual case), and the caption
  passes ``clean_text``. A clip the teacher did not
  reach is dropped (``no_teacher``). Kept clips are taken in row order (shard
  order) up to 50 h. The report also counts each rule's hits over every
  decoded clip, and the rate of rule (a) over the ``text_available`` ones.
- Common Voice (``cv-pool.jsonl``): rules 1-4 again in plain text (sentence,
  ``client_prefix``, clip, the 17.0, 19.0, 21.0 and 22.0 dev and test clips,
  the CV 8.0 test sentences), rule 4.2, ``clean_text``, then ``up_votes`` at
  least 2 and ``down_votes`` 0. The eligible clips, sorted by path and
  shuffled with ``pilot.SEED``, are drawn with at most 30 clips per
  ``client_prefix``: first every clip whose sentence holds a lexicon
  occurrence (``copies`` 2), then the others (``copies`` 1) up to 20 h,
  copies counted.
- Galgame (``galgame-rows.jsonl``): ``clean_text`` and rule 4.2. Every
  remaining vocalization-only line with a lexicon interjection
  (``lexicon.is_meaningful_interjection``) is in the interjection pool
  (``reuse`` 2, at most two windows); the other lines, in row order shuffled
  with ``pilot.SEED``, fill each archive up to 1.5 h (pool lines included)
  and the source up to 45 h. Hours count each line once.
- JVNV (``jvnv-rows.jsonl``, the regular session): ``text`` through
  ``clean_text`` and rule 4.2. JNV (``jnv-rows.jsonl``, the regular
  session): ``phrase`` likewise.

Each draw takes items while the next one fits its target and stops at the
first that does not (``over_target`` counts the rest). Before anything is
written, every selected item is checked again against freshly read lists:
rule 4.2, ``clean_text``, ``PHANTOM_FAMILY`` (galgame and JNV), the Common
Voice rules 1-4, and the outbox text rule on what it carries and on each
galgame archive. Any hit: exit 1, no outbox file.

Outputs:

- ``<pilot>\\outbox\\allow.json`` (``ft-pilot-allow-1``)::

    {"schema": "ft-pilot-allow-1",
     "reazon": {"keys": ["000/xxxx.flac", ...]},
     "cv": {"dataset": "<pilot.CV_DATASET>", "clips": [{"path": "common_voice_ja_1.mp3", "copies": 2}, ...]},
     "galgame": {"lines": [{"archive": "GalGame/<maker>_<game>.7z",
                            "voice": "<game folder>/<speaker>/<file>.ogg", "reuse": 2}, ...]},
     "jvnv": {"ids": ["F1_anger_regular_01", ...]},
     "jnv": {"files": ["JNV/F1/F1_happy_00_R.wav", ...]}}

  ``reazon.keys`` are tar member names of ``<data>\\<shard>.tar`` (captions in
  ``small.tsv``, or the same lines in the mirror's ``all.tsv``);
  ``cv.clips[].path`` names a clip of the Common Voice archive
  (``validated.tsv`` ``path``), ``copies`` how often it enters the mix.
  ``galgame.lines[].voice`` is the path ``scripts.asr_finetune.stage`` writes:
  the audio file below the archive's extraction folder, in the archive's own
  tree (usually ``<game folder>/<speaker>/<file>.ogg``; ``.opus`` and
  ``.wav`` occur). T1 extracts ``archive`` with ``7z x`` into
  ``<data>\\galgame\\<PurePosixPath(archive).stem>\\`` as stage does and opens
  ``<that folder>/<voice>``; the text is that file's ``index.json`` entry.
  ``reuse`` is 2 for interjection-pool lines and 1 otherwise. ``jvnv.ids``
  are utterance ids (``<speaker>_<emotion>_regular_<nn>``) of
  ``jvnv_v1/<speaker>/<emotion>/regular/<id>.wav``; ``jnv.files`` are member
  names of ``jnv_corpus_ver3.zip``. Lists are in row order (Common Voice by
  path).
- ``<pilot>\\outbox\\fetch-t1.json``: a ``scripts.asr_finetune.fetch`` plan
  (``ft-pilot-fetch-plan-1``) with the ReazonSpeech captions and the shards
  that hold a selected clip (``reazon-small-tsv`` and ``reazon-NNN``, or with
  ``--reazon-source mirror`` ``reazon-mirror-all-tsv`` and
  ``reazon-mirror-NNN``), ``cv``, ``jvnv`` and ``jnv`` when they have a
  selected item, and the archives that hold a selected line (sizes from
  ``stage-plan.json``, plan order). It must load with ``fetch.load_plan``.
- ``<pilot>\\selection\\selection-report.json`` (private): per source the
  status, rows, hours, drop counts per reason (galgame per archive as
  ``galgame-NN``; ``clean_phantom`` counts the lines ``PHANTOM_FAMILY``
  removes), selected counts and hours against the targets, the recheck and
  the outbox checks.

Both outbox files must pass ``check_outbox_file`` (with ``--outbox`` as its
outbox root); on a refusal both are deleted (exit 1). Outbox files of an
earlier run in that folder are deleted first, so a failed run leaves none.
Console and report hold counts and ids only, never a
caption, sentence, line, title or archive name. Exit 0 when both files are
written and pass, 1 on a recheck hit or a refusal, 2 on a missing or
misshapen input.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scripts.asr_bench.align import alignment_trace
from scripts.asr_bench.lexicon import analyze, is_meaningful_interjection
from scripts.asr_finetune.fetch import PLAN_SCHEMA, PlanError, galgame_name, load_plan
from scripts.asr_finetune.outbox import OutboxRefusal, check_outbox_file, hygiene_dir, outbox_dir, text_finding
from scripts.asr_finetune.pilot import (
    CV_DATASET,
    OOPPEENN,
    PHANTOM_FAMILY,
    REAZON_ORIGINS,
    SEED,
    BenchLineFilter,
    adult_reason,
    clean_reason,
    clean_text,
    loop_reason,
    norm,
    pilot_root,
    reazon_sources,
    vocal_reason,
)
from scripts.asr_finetune.stage import OUT_SCHEMA as STAGE_SCHEMA
from scripts.asr_finetune.teacher import OUT_SCHEMA as TEACHER_SCHEMA

ALLOW_SCHEMA = "ft-pilot-allow-1"
REPORT_SCHEMA = "ft-pilot-selection-1"
HOUR = 3600.0
SOURCES = ("reazon", "cv", "galgame", "jvnv", "jnv")
ROWS = {"reazon": "reazon-rows.jsonl", "cv": "cv-pool.jsonl", "galgame": "galgame-rows.jsonl",
        "jvnv": "jvnv-rows.jsonl", "jnv": "jnv-rows.jsonl"}
# Drawn sources: seconds, copies counted. JVNV and JNV take every kept item.
TARGET_S = {"reazon": 50 * HOUR, "cv": 20 * HOUR, "galgame": 45 * HOUR}
# Hours the plan expects of JVNV and JNV (JNV: under), for the console and report.
EXPECTED_H = {"jvnv": 3.4, "jnv": 0.2}
ARCHIVE_CAP_S = 1.5 * HOUR
CV_MIN_UP_VOTES = 2
CV_PER_SPEAKER = 30
CV_INTERJECTION_COPIES = 2
GALGAME_POOL_REUSE = 2
TEXT_AVAILABLE = "text_available"
# The longest content of a window (plan D3, shape D): an item longer than
# this fits no window; counted for the window plan, not dropped here.
WINDOW_CONTENT_S = 28.5
# Survey 5.2's teacher filter (a)-(d), then the round-2 caption-deletion rule
# (e), in the order a drop is counted.
REAZON_RULES = ("a_teacher_interjection", "b_teacher_beyond_caption", "c_teacher_outcome", "d_bench_line",
                "e_caption_deletes")
# Rule (e): teacher reading characters a caption may leave out, in all and
# before the teacher's first character the caption matches.
MAX_CAPTION_DELETED_KANA = 2
MAX_LEADING_DELETED_KANA = 1
CV_RESIDUAL = "CV 20.0 dev and test have no reachable mirror; the 19.0 and 21.0 splits stand in (plan D4, rule 4)"


class SelectError(RuntimeError):
    """A missing or misshapen input (exit 2)."""


@dataclass(frozen=True)
class Item:
    """One selectable item: its allow-list id, its source text and seconds;
    ``copies`` is the Common Voice copy count or the galgame reuse."""

    id: str
    text: str
    seconds: float
    copies: int = 1
    archive: str = ""  # galgame
    prefix: str = ""   # Common Voice client_prefix


@dataclass(frozen=True)
class CvLists:
    """The Common Voice rules of ``cv-exclusion.json`` in plain normalized text."""

    sentences: frozenset[str]
    prefixes: frozenset[str]
    clips: frozenset[str]
    split_clips: frozenset[str]
    cv8_sentences: frozenset[str]

    def rule(self, path: str, prefix: str, sentence: str) -> str | None:
        normalized = norm(sentence)
        return ("rule1_sentence" if normalized in self.sentences
                else "rule2_speaker" if prefix in self.prefixes
                else "rule3_clip" if path in self.clips
                else "rule4_split" if path in self.split_clips
                else "rule4_cv8_test" if normalized in self.cv8_sentences
                else None)


@dataclass(frozen=True)
class Hygiene:
    bench_lines: tuple[str, ...]
    cv: CvLists


# --- inputs ---------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SelectError(f"{path} is missing") from None
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SelectError(f"cannot read {path}: {type(error).__name__}") from None


def _jsonl(path: Path, partial_tail: bool = False) -> tuple[list[dict[str, Any]], int]:
    """The objects of a JSON Lines file, and 1 when ``partial_tail`` allowed
    an unreadable last line (a writer stopped mid-line), else 0."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise SelectError(f"{path} is missing") from None
    except (OSError, UnicodeDecodeError) as error:
        raise SelectError(f"cannot read {path}: {type(error).__name__}") from None
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if partial_tail and number == len(lines):
                return rows, 1
            raise SelectError(f"{path.name} line {number} is not JSON") from None
        if not isinstance(row, dict):
            raise SelectError(f"{path.name} line {number} is not an object")
        rows.append(row)
    return rows, 0


def load_hygiene() -> Hygiene:
    """The plain lists of ``scripts.asr_finetune.hygiene``, read from disk."""
    folder = hygiene_dir()
    lines = _read_json(folder / "bench-lines.json")
    raw = _read_json(folder / "cv-exclusion.json")
    keys = ("sentences", "client_id_prefixes", "clips", "split_clips", "cv8_test_sentences")
    if (not isinstance(lines, dict) or not isinstance(lines.get("lines"), list) or not lines["lines"]
            or not isinstance(raw, dict) or not all(isinstance(raw.get(key), list) for key in keys)):
        raise SelectError("the hygiene lists are not shaped as scripts.asr_finetune.hygiene writes them")
    return Hygiene(tuple(lines["lines"]), CvLists(*(frozenset(raw[key]) for key in keys)))


def galgame_plan() -> dict[str, tuple[int, int]]:
    """``stage-plan.json``'s archives: repo path -> (plan index, bytes)."""
    try:
        plan = load_plan(outbox_dir() / "stage-plan.json")
    except PlanError as error:
        raise SelectError(str(error)) from None
    return {path: (index, size) for index, (path, size) in enumerate(plan.galgame)}


def _run_label(folder: Path) -> str:
    """A teacher folder as the report names it: its path below ``<pilot>``
    (``pods/s1/teacher``), else its folder name."""
    try:
        return folder.resolve().relative_to(pilot_root().resolve()).as_posix()
    except ValueError:
        return folder.name


def _teacher_run(folder: Path, required: bool) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """One run's ``teacher.jsonl`` lines (None without the file, an error
    when ``required``) and its record: what ``teacher.json`` says (stopped,
    shards complete and partial, lines) and a cut last line."""
    record: dict[str, Any] = {"run": _run_label(folder)}
    summary_path = folder / "teacher.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if not isinstance(summary, dict) or summary.get("schema") != TEACHER_SCHEMA:
            raise SelectError(f"{summary_path} is not a {TEACHER_SCHEMA} record")
        record.update({key: summary.get(key) for key in ("stopped", "shards_complete", "shards_partial", "lines")})
    else:
        record["teacher_json"] = "missing"
    if not (folder / "teacher.jsonl").is_file():
        if required:
            raise SelectError(f"{folder / 'teacher.jsonl'} is missing")
        record["teacher_jsonl"] = "missing"
        return None, record
    rows, partial = _jsonl(folder / "teacher.jsonl", partial_tail=True)
    if not all(isinstance(row.get(key), str) for row in rows for key in ("key", "text", "outcome")):
        raise SelectError(f"{record['run']}/teacher.jsonl holds a line without a key, text and outcome")
    record.update(lines_read=len(rows), partial_last_line=partial)
    return rows, record


def load_teacher(folders: Sequence[Path]) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any]]:
    """The teacher lines by key from every run in order (S1's ``teacher\\``
    first, then each ``--extra-teacher``): a key takes its first line, so an
    earlier run wins and a later run's line for that key is ``superseded``.
    None when no run has a ``teacher.jsonl`` (only S1's may lack one). Also
    what the report says: each run's record, the lines used, the superseded
    count and the shards complete in any run."""
    lines: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    found = False
    for index, folder in enumerate(folders):
        rows, record = _teacher_run(folder, required=index > 0)
        found |= rows is not None
        own: set[str] = set()
        repeated = superseded = 0
        for row in rows or ():
            if row["key"] in own:
                repeated += 1
                continue
            own.add(row["key"])
            if row["key"] in lines:
                superseded += 1
                continue
            lines[row["key"]] = row
        if rows is not None:
            record.update(repeated_keys=repeated, superseded=superseded)
        runs.append(record)
    complete = sorted({shard for record in runs for shard in record.get("shards_complete") or ()})
    return (lines if found else None), {
        "runs": runs, "lines_used": len(lines), "superseded": sum(record.get("superseded", 0) for record in runs),
        "shards_complete": complete,
    }


# --- rules ----------------------------------------------------------------------


@dataclass(frozen=True)
class TextRules:
    """The optional label rules: ``loop`` (``--loop-hygiene``, round 4,
    ``pilot.loop_reason`` on every source), ``adult`` (``--adult-filter``,
    round 4, ``pilot.adult_reason`` on galgame lines) and ``vocal``
    (``--vocal-cap``, round 6, ``pilot.vocal_reason`` on every source, and
    the round-6 words in ``adult_reason``), in that order."""

    loop: bool = False
    adult: bool = False
    vocal: bool = False

    def reason(self, source: str, label: str) -> str | None:
        return ((loop_reason(label) if self.loop else None)
                or (adult_reason(label, self.vocal) if self.adult and source == "galgame" else None)
                or (vocal_reason(label) if self.vocal else None))


NO_RULES = TextRules()


def _clean_drop(source: str, text: str, rules: TextRules = NO_RULES) -> str | None:
    """``clean_<reason>`` when ``clean_text`` drops the text, else the
    reason ``rules`` finds in its cleaned label, or None."""
    reason = clean_reason(source, text)
    if reason:
        return f"clean_{reason}"
    return rules.reason(source, clean_text(source, text) or "")


def _text_drop(source: str, text: str, bench: BenchLineFilter, rules: TextRules = NO_RULES) -> str | None:
    """``_clean_drop``'s reason, ``bench_line`` when rule 4.2 hits the text,
    else None."""
    return _clean_drop(source, text, rules) or ("bench_line" if bench.hit(text) else None)


def trailing_insertions(caption: str, teacher: str) -> int:
    """Characters of ``teacher`` after the alignment position of the last
    character of ``caption`` (``alignment_trace``). A minimum-edit alignment
    never deletes that last character while teacher characters remain after
    its position (one substitution is cheaper than a deletion and an
    insertion), so it aligns at its position, or past the end."""
    if not caption:
        return len(teacher)
    if not teacher:
        return 0
    _, positions, _ = alignment_trace(caption, teacher)
    return max(len(teacher) - 1 - int(positions[-1]), 0)


def caption_deletions(teacher: str, caption: str) -> tuple[int, int]:
    """(teacher characters that ``alignment_trace(teacher, caption)`` does not
    align as an exact match, those of them before the first matched one).
    With no matched character the whole teacher reading is leading."""
    matched, _, _ = alignment_trace(teacher, caption)
    if not matched.any():
        return len(teacher), len(teacher)
    return int((~matched).sum()), int(matched.argmax())


def reazon_rules(caption: str, teacher: str, outcome: str, bench: BenchLineFilter) -> dict[str, bool]:
    """Which rule of ``REAZON_RULES`` the clip breaks: survey 5.2's teacher
    filter (a)-(d) and the caption-deletion rule (e)."""
    heard, captioned = analyze(teacher), analyze(caption)
    extra = Counter(key for *_, key in heard.interjections) - Counter(key for *_, key in captioned.interjections)
    deleted, leading = caption_deletions(heard.reading, captioned.reading)
    return {"a_teacher_interjection": bool(extra),
            "b_teacher_beyond_caption": trailing_insertions(captioned.reading, heard.reading) > 0,
            "c_teacher_outcome": outcome != TEXT_AVAILABLE,
            "d_bench_line": bench.hit(caption),
            "e_caption_deletes": deleted > MAX_CAPTION_DELETED_KANA or leading > MAX_LEADING_DELETED_KANA}


def carried(item: Item) -> tuple[str, ...]:
    """The strings of an item that reach a pod: its allow-list id and, for a
    galgame line, the window plan's ``<archive>|<voice>`` id."""
    return (item.id, f"{item.archive}|{item.id}") if item.archive else (item.id,)


def _outbox_reason(finding: str | None) -> str | None:
    """``outbox_title`` for a title or alias the outbox text rule finds,
    ``outbox_text`` for a shingle of a bench line (a chance id collision)."""
    if finding is None:
        return None
    return "outbox_title" if "title" in finding else "outbox_text"


def _outbox_text(item: Item) -> str | None:
    """Why a string the item carries to a pod breaks the outbox text rule (it
    would refuse ``allow.json``): ``outbox_title`` before ``outbox_text``, or None."""
    reasons = {_outbox_reason(text_finding(value)) for value in carried(item)}
    return next((reason for reason in ("outbox_title", "outbox_text") if reason in reasons), None)


def _take(items: Iterable[Item], target_s: float) -> tuple[list[Item], int]:
    """Items in order while the next fits ``target_s``; (taken, the rest's count)."""
    taken: list[Item] = []
    total = 0.0
    rest = 0
    for item in items:
        if rest or total + item.seconds * item.copies > target_s:
            rest += 1
            continue
        taken.append(item)
        total += item.seconds * item.copies
    return taken, rest


def _seconds(items: Iterable[Item], copies: bool = True) -> float:
    return sum(item.seconds * (item.copies if copies else 1) for item in items)


def _h(seconds: float) -> float:
    return round(seconds / HOUR, 3)


def _rows_h(rows: Iterable[dict[str, Any]]) -> float:
    return _h(sum(float(row["duration_s"]) for row in rows))


def _long(items: Iterable[Item]) -> int:
    return sum(item.seconds > WINDOW_CONTENT_S for item in items)


# --- sources ----------------------------------------------------------------------


def select_reazon(rows: Sequence[dict[str, Any]], teacher: dict[str, dict[str, Any]], bench: BenchLineFilter,
                  target_s: float = TARGET_S["reazon"], rules: TextRules = NO_RULES) -> tuple[list[Item], dict[str, Any]]:
    drops: Counter = Counter()
    hits: Counter = Counter()
    available_hits: Counter = Counter()
    outcomes: Counter = Counter()
    kept: list[Item] = []
    for row in rows:
        key, caption, seconds = row["key"], row["caption"], float(row["duration_s"])
        line = teacher.get(key)
        if line is None:
            drops["no_teacher"] += 1
            continue
        outcomes[line["outcome"]] += 1
        teacher_rules = reazon_rules(caption, line["text"], line["outcome"], bench)
        broken = [name for name in REAZON_RULES if teacher_rules[name]]
        hits.update(broken)
        if line["outcome"] == TEXT_AVAILABLE:
            available_hits.update(broken)
        item = Item(key, caption, seconds)
        reason = (broken[0] if broken else None) or _clean_drop("reazon", caption, rules) or _outbox_text(item)
        if reason:
            drops[reason] += 1
            continue
        kept.append(item)
    picked, rest = _take(kept, target_s)
    available = outcomes[TEXT_AVAILABLE]
    rate = available_hits["a_teacher_interjection"] / available if available else None
    return picked, {
        "rows": len(rows), "rows_h": _rows_h(rows),
        "decoded": sum(outcomes.values()), "outcomes": dict(sorted(outcomes.items())),
        "teacher_keys_without_row": len(set(teacher) - {row["key"] for row in rows}),
        "rule_hits": {name: hits[name] for name in REAZON_RULES},
        "rule_a_rate": None if rate is None else round(rate, 4),
        "dropped": dict(sorted(drops.items())),
        "kept": len(kept), "kept_h": _h(_seconds(kept)), "over_target": rest,
        "selected": len(picked), "selected_h": _h(_seconds(picked)),
        "shards": sorted({item.id.split("/", 1)[0] for item in picked}),
        "selected_over_window_s": _long(picked),
    }


def select_cv(rows: Sequence[dict[str, Any]], lists: CvLists, bench: BenchLineFilter,
              target_s: float = TARGET_S["cv"], rules: TextRules = NO_RULES) -> tuple[list[Item], dict[str, Any]]:
    drops: Counter = Counter()
    eligible: list[Item] = []
    for row in rows:
        path, prefix, sentence = row["path"], row["client_prefix"], row["sentence"]
        item = Item(path, sentence, float(row["duration_s"]), prefix=prefix)
        reason = lists.rule(path, prefix, sentence) or _text_drop("cv", sentence, bench, rules)
        if reason is None and (int(row["up_votes"]) < CV_MIN_UP_VOTES or int(row["down_votes"]) != 0):
            reason = "votes"
        reason = reason or _outbox_text(item)
        if reason:
            drops[reason] += 1
            continue
        eligible.append(item)
    eligible.sort(key=lambda item: item.id)
    random.Random(SEED).shuffle(eligible)
    flagged = {item.id for item in eligible if analyze(item.text).interjections}
    speakers: Counter = Counter()
    capped: Counter = Counter()
    picked: list[Item] = []
    for item in eligible:
        if item.id not in flagged:
            continue
        if speakers[item.prefix] >= CV_PER_SPEAKER:
            capped["interjection"] += 1
            continue
        speakers[item.prefix] += 1
        picked.append(replace(item, copies=CV_INTERJECTION_COPIES))
    with_interjection = len(picked)
    total = _seconds(picked)
    rest = 0
    for item in eligible:
        if item.id in flagged:
            continue
        if rest or total + item.seconds > target_s:
            rest += 1
            continue
        if speakers[item.prefix] >= CV_PER_SPEAKER:
            capped["other"] += 1
            continue
        speakers[item.prefix] += 1
        picked.append(item)
        total += item.seconds
    picked.sort(key=lambda item: item.id)
    return picked, {
        "rows": len(rows), "rows_h": _rows_h(rows), "dropped": dict(sorted(drops.items())),
        "eligible": len(eligible), "eligible_with_interjection": len(flagged),
        "speaker_cap_skips": {kind: capped[kind] for kind in ("interjection", "other")}, "over_target": rest,
        "selected": len(picked), "selected_with_interjection": with_interjection,
        "selected_h": _h(_seconds(picked)), "selected_audio_h": _h(_seconds(picked, copies=False)),
        "speakers": len(speakers), "most_clips_per_speaker": max(speakers.values(), default=0),
        "selected_over_window_s": _long(picked),
    }


def select_galgame(rows: Sequence[dict[str, Any]], archives: dict[str, tuple[int, int]], bench: BenchLineFilter,
                   target_s: float = TARGET_S["galgame"], rules: TextRules = NO_RULES) -> tuple[list[Item], dict[str, Any]]:
    per: dict[str, Counter] = {}
    lines: list[Item] = []  # the kept lines in row order, pool lines with reuse 2
    for row in rows:
        archive, voice, text = row["archive"], row["voice"], row["text"]
        if archive not in archives:
            raise SelectError("galgame-rows.jsonl names an archive that stage-plan.json does not")
        counts = per.setdefault(archive, Counter())
        counts["rows"] += 1
        item = Item(voice, text, float(row["duration_s"]), archive=archive)
        reason = _text_drop("galgame", text, bench, rules) or _outbox_text(item)
        if reason:
            counts[reason] += 1
            continue
        pool = is_meaningful_interjection(clean_text("galgame", text) or "")
        lines.append(replace(item, copies=GALGAME_POOL_REUSE) if pool else item)
    archive_s: Counter = Counter()
    chosen: set[int] = set()
    for item in lines:
        if item.copies > 1:
            archive_s[item.archive] += item.seconds
            chosen.add(id(item))
    total = sum(archive_s.values())
    others = [item for item in lines if item.copies == 1]
    random.Random(SEED).shuffle(others)
    capped: Counter = Counter()
    rest = 0
    for item in others:
        if rest or total + item.seconds > target_s:
            rest += 1
            continue
        if archive_s[item.archive] + item.seconds > ARCHIVE_CAP_S:
            capped[item.archive] += 1
            continue
        archive_s[item.archive] += item.seconds
        total += item.seconds
        chosen.add(id(item))
    picked = [item for item in lines if id(item) in chosen]
    by_archive: dict[str, list[Item]] = {}
    for item in picked:
        by_archive.setdefault(item.archive, []).append(item)
    per_archive = {}
    for path in sorted(per, key=lambda path: archives[path][0]):
        mine = by_archive.get(path, [])
        per_archive[galgame_name(archives[path][0])] = {
            **dict(sorted(per[path].items())), "pool": sum(item.copies > 1 for item in mine),
            "archive_cap_skips": capped[path], "selected": len(mine), "selected_h": _h(_seconds(mine, copies=False)),
        }
    totals: Counter = Counter()
    for counts in per.values():
        totals.update(counts)
    pool_items = [item for item in lines if item.copies > 1]
    return picked, {
        "rows": totals.pop("rows", 0), "rows_h": _rows_h(rows), "archives_with_rows": len(per),
        "dropped": dict(sorted(totals.items())),
        "pool": len(pool_items), "pool_h": _h(_seconds(pool_items, copies=False)),
        "archive_cap_skips": sum(capped.values()), "over_target": rest,
        "selected": len(picked), "selected_h": _h(_seconds(picked, copies=False)),
        "selected_h_with_reuse": _h(_seconds(picked)),
        "archives_selected": len(by_archive), "most_h_per_archive": _h(max(archive_s.values(), default=0.0)),
        "selected_over_window_s": _long(picked),
        "per_archive": per_archive,
    }


def select_every(source: str, rows: Sequence[dict[str, Any]], bench: BenchLineFilter,
                 ident: str, field: str, rules: TextRules = NO_RULES) -> tuple[list[Item], dict[str, Any]]:
    """JVNV (``id``, ``text``) and JNV (``file``, ``phrase``): every kept row."""
    drops: Counter = Counter()
    picked: list[Item] = []
    for row in rows:
        item = Item(row[ident], row[field], float(row["duration_s"]))
        reason = _text_drop(source, row[field], bench, rules) or _outbox_text(item)
        if reason:
            drops[reason] += 1
            continue
        picked.append(item)
    return picked, {
        "rows": len(rows), "rows_h": _rows_h(rows), "dropped": dict(sorted(drops.items())),
        "selected": len(picked), "selected_h": _h(_seconds(picked)), "selected_over_window_s": _long(picked),
    }


# --- recheck ------------------------------------------------------------------


def recheck(selected: dict[str, list[Item]], rules: TextRules = NO_RULES) -> dict[str, int]:
    """Every selected item against freshly read lists: problem -> count."""
    hygiene = load_hygiene()
    bench = BenchLineFilter(hygiene.bench_lines)
    problems: Counter = Counter()
    for source, items in selected.items():
        for item in items:
            if bench.hit(item.text):
                problems[f"{source} bench_line"] += 1
            label = clean_text(source, item.text)
            if label is None:
                problems[f"{source} clean_text"] += 1
            elif rules.reason(source, label):
                problems[f"{source} {rules.reason(source, label)}"] += 1
            if source in ("galgame", "jnv") and (PHANTOM_FAMILY.search(item.text)
                                                 or PHANTOM_FAMILY.search(unicodedata.normalize("NFKC", item.text))):
                problems[f"{source} phantom_family"] += 1
            rule = hygiene.cv.rule(item.id, item.prefix, item.text) if source == "cv" else None
            if rule:
                problems[f"cv {rule}"] += 1
            reason = _outbox_text(item)
            if reason:
                problems[f"{source} {reason}"] += 1
    for archive in {item.archive for item in selected["galgame"]}:
        reason = _outbox_reason(text_finding(archive))
        if reason:
            problems[f"galgame archive {reason}"] += 1
    return {name: count for name, count in sorted(problems.items()) if count}


# --- outputs ----------------------------------------------------------------------


def allow_list(selected: dict[str, list[Item]]) -> dict[str, Any]:
    return {
        "schema": ALLOW_SCHEMA,
        "reazon": {"keys": [item.id for item in selected["reazon"]]},
        "cv": {"dataset": CV_DATASET, "clips": [{"path": item.id, "copies": item.copies} for item in selected["cv"]]},
        "galgame": {"lines": [{"archive": item.archive, "voice": item.id, "reuse": item.copies}
                              for item in selected["galgame"]]},
        "jvnv": {"ids": [item.id for item in selected["jvnv"]]},
        "jnv": {"files": [item.id for item in selected["jnv"]]},
    }


def fetch_plan(selected: dict[str, list[Item]], archives: dict[str, tuple[int, int]],
               reazon_source: str = "official", nonspeech: bool = False) -> dict[str, Any]:
    """The T1 fetch plan: what holds a selected item, and the captions;
    ReazonSpeech from ``reazon_source`` (``pilot.REAZON_ORIGINS``); with
    ``nonspeech``, the MUSAN archive of ``mix --nonspeech``'s windows."""
    shards = sorted({item.id.split("/", 1)[0] for item in selected["reazon"]})
    sources = reazon_sources(reazon_source, shards) if shards else []
    sources += [source for source in ("cv", "jvnv", "jnv") if selected[source]]
    sources += ["musan"] if nonspeech else []
    plan: dict[str, Any] = {"schema": PLAN_SCHEMA, "sources": sources}
    used = {item.archive for item in selected["galgame"]}
    if used:
        repo, revision = OOPPEENN
        plan["galgame"] = {"repo": repo, "revision": revision, "archives": [
            {"path": path, "size": size}
            for path, (_, size) in sorted(archives.items(), key=lambda pair: pair[1][0]) if path in used]}
    return plan


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _outbox_root(folder: Path) -> Iterator[None]:
    """``check_outbox_file`` with ``folder`` as its outbox root while the
    block runs (``--outbox``): every rule of the check applies, the location
    rule against ``folder``. A side-by-side outbox is checked again, by
    ``scripts.asr_bench.remote``, once its files are moved into the real one."""
    from scripts.asr_finetune import outbox

    original = outbox.outbox_dir
    outbox.outbox_dir = lambda: folder
    try:
        yield
    finally:
        outbox.outbox_dir = original


def _label(folder: Path) -> str:
    """``folder`` as the console names it: ``<pilot>\\...`` or its name."""
    try:
        return "<pilot>\\" + str(folder.resolve().relative_to(pilot_root().resolve()))
    except ValueError:
        return folder.name


def _write_outbox(selected: dict[str, list[Item]], archives: dict[str, tuple[int, int]],
                  report: dict[str, Any], folder: Path, reazon_source: str = "official",
                  nonspeech: bool = False) -> int:
    allow, plan_path = folder / "allow.json", folder / "fetch-t1.json"
    plan = fetch_plan(selected, archives, reazon_source, nonspeech)
    _write_json(allow, allow_list(selected))
    _write_json(plan_path, plan)
    try:
        with _outbox_root(folder):
            for path in (allow, plan_path):
                check_outbox_file(path)
        load_plan(plan_path)
    except (OutboxRefusal, PlanError) as refusal:
        for path in (allow, plan_path):
            path.unlink(missing_ok=True)
        report["outbox"] = {"refused": str(refusal)}
        print(f"refused, allow.json and fetch-t1.json deleted: {refusal}", file=sys.stderr)
        return 1
    report["outbox"] = {path.name: {"bytes": path.stat().st_size, "check_outbox_file": "passed"}
                        for path in (allow, plan_path)}
    report["outbox"]["folder"] = _label(folder)
    report["outbox"]["fetch-t1.json"].update(sources=plan["sources"],
                                             archives=len((plan.get("galgame") or {}).get("archives", [])))
    print(f"wrote {_label(folder)}\\allow.json ({allow.stat().st_size:,} bytes) and fetch-t1.json "
          f"({plan_path.stat().st_size:,} bytes: {len(plan['sources'])} sources, "
          f"{report['outbox']['fetch-t1.json']['archives']} archives); check_outbox_file passed"
          + ("" if folder.resolve() == outbox_dir().resolve() else " with that folder as the outbox root")
          + ", fetch-t1.json loads as a fetch plan")
    return 0


# --- main ---------------------------------------------------------------------


def _summary(source: str, record: dict[str, Any], expected: float) -> str:
    dropped = ", ".join(f"{reason} {count:,}" for reason, count in record["dropped"].items()) or "none"
    hours = record["selected_h"]
    against = (f"expected under {expected:g} h" if source == "jnv"
               else f"{(hours - expected) / expected:+.1%} from {expected:g} h")
    return (f"{source}: {record['rows']:,} rows ({record['rows_h']:,.2f} h); dropped {dropped}; "
            f"selected {record['selected']:,} ({hours:,.2f} h, {against})")


def _notes(report: dict[str, Any], staged: dict[str, Any]) -> None:
    """Report and console notes: the teacher-filter hits, the Common Voice
    speaker matches and residual, the galgame and JNV phantom drops."""
    sources = report["sources"]
    reazon = sources["reazon"]
    if "rule_hits" in reazon:
        rate = reazon["rule_a_rate"]
        print(f"  reazon rule hits over {reazon['decoded']:,} decoded clips: "
              + ", ".join(f"{name} {count:,}" for name, count in reazon["rule_hits"].items())
              + f"; rule (a) rate {'n/a' if rate is None else f'{rate:.1%}'} of text_available clips; "
              + f"kept {reazon['kept']:,} ({reazon['kept_h']:,.2f} h); shards {', '.join(reazon['shards']) or 'none'}")
    cv = sources["cv"]
    if "eligible" in cv:
        s1_drops = (staged.get("cv") or {}).get("dropped") or {}
        cv["s1_dropped"] = s1_drops
        cv["speaker_prefix_matches"] = {
            "s1": s1_drops.get("speaker", 0), "local_rule2": cv["dropped"].get("rule2_speaker", 0),
            "note": "S1 counts the v27 validated clips whose client_id prefix is a rule-2 prefix (after its split "
                    "and clip rules); 0 would mean client ids changed across releases (rules 1 and 3 do not "
                    "depend on it)"}
        cv["residual"] = CV_RESIDUAL
        print(f"  cv: the CV 17 speaker prefixes matched {cv['speaker_prefix_matches']['s1']:,} v27 clips on S1 "
              f"and {cv['speaker_prefix_matches']['local_rule2']:,} here; rule 4.2 dropped "
              f"{cv['dropped'].get('bench_line', 0):,}; {cv['selected_with_interjection']:,} interjection clips "
              f"(copies {CV_INTERJECTION_COPIES}), speaker-cap skips {cv['speaker_cap_skips']['interjection']:,} "
              f"of them; {cv['selected_audio_h']:,.2f} h of audio")
    phantom: dict[str, Any] = {}
    galgame = sources["galgame"]
    if "per_archive" in galgame:
        per = {name: counts.get("clean_phantom", 0) for name, counts in galgame["per_archive"].items()}
        phantom["galgame"] = {"total": sum(per.values()), "per_archive": per}
        records = (staged.get("galgame") or {}).get("archives") or {}
        failed = sorted(name for name, record in records.items()
                        if "error" in record or record.get("extract_exit") not in (0, 1))
        galgame["s1_failed_archives"] = failed
        print(f"  galgame PHANTOM_FAMILY drops {phantom['galgame']['total']:,} ("
              + ", ".join(f"{name} {count:,}" for name, count in per.items() if count)
              + f"); pool {galgame['pool']:,} lines (reuse {GALGAME_POOL_REUSE}); {galgame['archives_selected']} "
              + f"archives selected, at most {galgame['most_h_per_archive']:.2f} h each; "
              + f"failed on S1: {', '.join(failed) or 'none'}")
        titled = {name: counts["outbox_title"] for name, counts in galgame["per_archive"].items()
                  if counts.get("outbox_title")}
        galgame["outbox_title"] = {"total": sum(titled.values()), "per_archive": titled}
        print(f"  galgame voice paths holding a title or alias (outbox_title) {sum(titled.values()):,}"
              + "".join(f", {name} {count:,}" for name, count in titled.items()))
    if "dropped" in sources["jnv"]:
        phantom["jnv"] = sources["jnv"]["dropped"].get("clean_phantom", 0)
        print(f"  jnv PHANTOM_FAMILY drops {phantom['jnv']:,}")
    report["phantom_family_drops"] = phantom


def select(s1: Path, extra_teachers: Sequence[Path] = (), targets: dict[str, float] | None = None,
           outbox: Path | None = None, reazon_source: str = "official", rules: TextRules = NO_RULES,
           nonspeech: bool = False) -> tuple[int, dict[str, Any]]:
    """The selection, its recheck and the outbox files: (exit code, report).
    ``targets``: seconds per drawn source (default ``TARGET_S``; 0 leaves the
    source out); ``outbox``: the folder of the two files (default the outbox);
    ``reazon_source``: where ``fetch-t1.json`` takes ReazonSpeech from."""
    if reazon_source not in REAZON_ORIGINS:
        raise SelectError(f"reazon_source must be one of {', '.join(REAZON_ORIGINS)}")
    targets = {**TARGET_S, **(targets or {})}
    stage = _read_json(s1 / "stage" / "stage.json")
    if not isinstance(stage, dict) or stage.get("schema") != STAGE_SCHEMA:
        raise SelectError(f"stage.json is not a {STAGE_SCHEMA} record")
    failed = list(stage.get("failed_sources") or [])
    staged = stage.get("sources") or {}
    hygiene = load_hygiene()
    bench = BenchLineFilter(hygiene.bench_lines)
    archives = galgame_plan()
    teacher, teacher_record = load_teacher([s1 / "teacher", *extra_teachers])
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "s1_failed_sources": failed, "stage_errors": stage.get("errors") or [],
        "teacher": teacher_record, "targets_h": {source: _h(seconds) for source, seconds in targets.items()},
        "expected_h": EXPECTED_H, "bench_lines_4_2": len(bench), "loop_hygiene": rules.loop,
        "adult_filter": rules.adult, "vocal_cap": rules.vocal, "sources": {},
    }
    runs = "; ".join(
        f"{record['run']} teacher.jsonl missing" if "lines_read" not in record else
        f"{record['run']} {record['lines_read']:,} lines ({record['partial_last_line']} cut last line), "
        f"stopped {record.get('stopped')}, shards complete {', '.join(record.get('shards_complete') or []) or 'none'}"
        for record in teacher_record["runs"])
    print(f"S1: failed sources {', '.join(failed) or 'none'}; teacher runs: {runs}; "
          f"{teacher_record['lines_used']:,} lines used, {teacher_record['superseded']:,} superseded; "
          f"shards complete in a run: {', '.join(teacher_record['shards_complete']) or 'none'}")
    steps: dict[str, Callable[[list[dict[str, Any]]], tuple[list[Item], dict[str, Any]]]] = {
        "reazon": lambda rows: select_reazon(rows, teacher or {}, bench, targets["reazon"], rules),
        "cv": lambda rows: select_cv(rows, hygiene.cv, bench, targets["cv"], rules),
        "galgame": lambda rows: select_galgame(rows, archives, bench, targets["galgame"], rules),
        "jvnv": lambda rows: select_every("jvnv", rows, bench, "id", "text", rules),
        "jnv": lambda rows: select_every("jnv", rows, bench, "file", "phrase", rules),
    }
    selected: dict[str, list[Item]] = {source: [] for source in SOURCES}
    for source in SOURCES:
        if source in failed:
            status = "failed on S1 (stage.json failed_sources): contributes nothing"
        elif source not in staged:
            status = "not staged on S1: contributes nothing"
        elif targets.get(source, 1) <= 0:
            status = "target 0 h: contributes nothing"
        elif source == "reazon" and teacher is None:
            status = "no teacher output: contributes nothing"
        else:
            rows, _ = _jsonl(s1 / "stage" / ROWS[source])
            try:
                selected[source], record = steps[source](rows)
            except KeyError as error:
                raise SelectError(f"{ROWS[source]}: a row has no {error.args[0]!r}") from None
            except (ValueError, TypeError) as error:  # the type only: a message may hold a text
                raise SelectError(f"{ROWS[source]}: a row has a malformed field ({type(error).__name__})") from None
            analyze.cache_clear()
            report["sources"][source] = {"status": "selected", **record}
            print(_summary(source, record, _h(targets[source]) if source in targets else EXPECTED_H[source]))
            continue
        report["sources"][source] = {"status": status}
        print(f"{source}: {status}")
    _notes(report, staged)

    problems = recheck(selected, rules)
    report["recheck"] = problems
    print("recheck of the selected items: " + (", ".join(f"{name} {count:,}" for name, count in problems.items())
                                               or "0 hits (bench lines, clean_text, PHANTOM_FAMILY, "
                                                  "Common Voice rules 1-4, outbox text)"))
    if problems:
        print("error: selected items hit a hygiene list or PHANTOM_FAMILY; no outbox file written", file=sys.stderr)
        return 1, report
    report["reazon_source"] = reazon_source
    report["nonspeech_fetch"] = nonspeech
    return _write_outbox(selected, archives, report, outbox or outbox_dir(), reazon_source, nonspeech), report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.select", description=__doc__.splitlines()[0])
    parser.add_argument("--s1", type=Path, default=pilot_root() / "pods" / "s1",
                        help="pod S1's fetched outputs (stage\\, teacher\\); default <pilot>\\pods\\s1")
    parser.add_argument("--extra-teacher", type=Path, action="append", default=[], metavar="DIR",
                        help="a later teacher run's folder (teacher.jsonl, teacher.json), read after "
                             "<s1>\\teacher; repeatable, the first line of a key wins")
    for source in TARGET_S:
        parser.add_argument(f"--{source}-hours", type=float, default=TARGET_S[source] / HOUR, metavar="H",
                            help=f"{source} target in hours (default %(default)g; 0 leaves it out)")
    parser.add_argument("--outbox", type=Path, default=outbox_dir(),
                        help="folder for allow.json and fetch-t1.json (default <pilot>\\outbox); another folder "
                             "also gets its own selection-report.<folder name>.json")
    parser.add_argument("--reazon-source", choices=tuple(REAZON_ORIGINS), default="official",
                        help="where T1 downloads ReazonSpeech: the corpus server (official, default) or its "
                             "Hugging Face mirror; fetch-t1.json only")
    parser.add_argument("--loop-hygiene", action="store_true",
                        help="also drop labels that could teach a loop (pilot.loop_reason: a run of 4 identical "
                             "characters, punctuation aside, or a zlib ratio over 2.0; round 4)")
    parser.add_argument("--adult-filter", action="store_true",
                        help="also drop galgame adult-scene lines (pilot.adult_reason: sound-effect vocabulary, "
                             "adult_sfx, or moan-dense lines, adult_moan; round 4, v4c)")
    parser.add_argument("--vocal-cap", action="store_true",
                        help="also drop labels with long vocal chains (pilot.vocal_reason, every source) and add "
                             "the round-6 words to --adult-filter (round 6, v6a)")
    parser.add_argument("--nonspeech", action="store_true",
                        help="add the MUSAN archive to fetch-t1.json, for mix --nonspeech (round 4)")
    args = parser.parse_args(argv)
    targets = {source: getattr(args, f"{source}_hours") * HOUR for source in TARGET_S}
    if any(seconds < 0 for seconds in targets.values()):
        parser.error("hours must be 0 or more")
    default = args.outbox.resolve() == outbox_dir().resolve()
    for name in ("allow.json", "fetch-t1.json"):
        (args.outbox / name).unlink(missing_ok=True)
    try:
        code, report = select(args.s1, args.extra_teacher, targets, args.outbox, args.reazon_source,
                              TextRules(args.loop_hygiene, args.adult_filter, args.vocal_cap), args.nonspeech)
    except (SelectError, OutboxRefusal) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    path = pilot_root() / "selection" / ("selection-report.json" if default
                                         else f"selection-report.{args.outbox.resolve().name}.json")
    _write_json(path, report)
    print(f"wrote <pilot>\\selection\\{path.name} ({path.stat().st_size:,} bytes)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
