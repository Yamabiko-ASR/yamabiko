"""Plan the pilot's training windows (plan D3) from the allow-list (local).

    python -m scripts.asr_finetune.mix [--outbox <dir>] [--allow <outbox\\allow.json>] [--rows <pods\\s1>] [--out <outbox\\windows-plan.jsonl.gz>] [--cuts] [--nonspeech <inventory.json> [--nonspeech-profile {r4,r5}]]
    python -m scripts.asr_finetune.mix --smoke <reazon-rows.jsonl> --windows 64 --out <file>

Reads ``allow.json`` (``scripts.asr_finetune.select``) and, for its items,
the S1 rows (``<rows>\\stage\\<source>-rows.jsonl``, ``cv-pool.jsonl``):
each item's duration and text. The text passes ``pilot.clean_text`` (an
allow-listed item it drops is an input error, exit 2) and gives the item its
label token count (``labels.load_tokenizer``) and its mark: a line is
*marked* when it is a JNV clip or ``lexicon.analyze`` finds an interjection
in it (``line_mark``). Nothing of a text reaches the plan.

Composition (plan D3), seeded with ``pilot.SEED``:

- **Shapes.** P: content at most 8.0 s, 1-4 utterances; D: content at most
  28.5 s, 2-6 utterances; both with 0.5 s of zeros before and after (added by
  ``scripts.asr_finetune.windows``). Content is the utterances and the gaps
  between them. Utterances are packed by their S1 duration plus 0.1 s of
  slack each (a decoder may return a little more than the duration) with every
  gap counted at its 1.0 s maximum, so the Silero-trimmed window always fits.
  Per source and split, a window is P while P holds under 40% of the planned
  content seconds, else D.
- **Packing.** One window takes the first queued utterance that fits, then the
  first ones of the next 64 queued that fit, up to the shape's most. A D
  window that finds no second utterance searches the whole queue, then becomes
  a P window with one utterance when that fits, else its utterance is skipped
  (``unpaired``). An item longer than 28.4 s (28.5 s minus the slack) fits no
  window and is skipped (``over_window``), as is one whose own label exceeds
  440 tokens (``over_tokens``); no window exceeds 440 label tokens.
- **Sources.** A window draws from one source and, for galgame, from one
  archive (title); JVNV forms its own windows; JNV is only spliced.
- **Positive windows.** A window is positive when it holds two or more lines
  and at least one is marked; anchor when its source is ReazonSpeech or Common
  Voice and no line is marked (``window_flags``, which
  ``scripts.asr_finetune.windows`` re-derives from the source texts). The
  window's designated line (a spliced JNV clip, a line drawn to make the window
  positive, else its first marked line) is placed first, middle or last with
  equal probability (first or last with two lines). In half the positive
  windows, the gap after each marked line is U(0.1, 0.4) s; every other gap is
  U(0.1, 1.0) s; the first utterance follows the leading zeros directly.
- **Galgame.** Per archive, when a window must be positive to keep at least
  half of the archive's windows positive and its first line is unmarked, it
  draws a marked line: from the next 64 queued, else from the second uses of
  the ``reuse`` 2 lines (each line in at most two windows, never twice in
  one). Common Voice clips enter the queue ``copies`` times.
- **JNV splices.** Each JNV clip enters at most two windows per epoch, like
  a galgame interjection line. A split's clips x 2 uses are shared between
  ReazonSpeech and Common Voice by their planned seconds (copies counted).
  A window of either gets one JNV clip (the next of the shuffled clips, in
  turn) while its source has uses left, the source's spliced windows stay at
  or under 10% of its windows so far, and a queued line fits beside the clip;
  a window where none fits passes the splice on to the next.
- **Level.** Per window a speech-RMS target U(-30, -18) dBFS
  (``level_dbfs``), per utterance U(-3, +3) dB (``gain_db``).
- **Dev split.** 1% of each source's items (JNV included), drawn with
  ``random.Random(pilot.SEED)``, form dev windows (``split`` ``dev``) apart
  from the train windows, so no dev utterance ever trains.
- **Cuts** (``--cuts``, round 3; ``mark_cuts``). Production windows often
  open or close mid-utterance (the chunker's max-chunk cut). Of the P windows
  of ReazonSpeech and galgame, 30% get a head cut and, drawn apart, 30% a
  tail cut: the first utterance carries ``cut_start`` (the target cut point
  as a fraction of it, U(0.2, 0.6)), the last ``cut_end`` (U(0.4, 0.8)); one
  utterance cut at both ends takes U(0.15, 0.35) and U(0.65, 0.85) and needs
  3.0 s, else it keeps the head cut. Only an utterance of the window's source
  at least 2.0 s long is cut. The marks use their own seeded generator, so
  the windows are those of the plan without ``--cuts``; a cut window is never
  an anchor. ``scripts.asr_finetune.windows`` resolves each mark at a gap
  between the teacher's words (``teacher --plan``) or refuses it.
- **No-speech windows** (``--nonspeech <inventory.json>``, round 4;
  ``nonspeech_windows``, profile ``--nonspeech-profile``, default ``r4``).
  After the speech windows, ``r4``: 8% of all windows are shape ``N``,
  source ``musan``: one span of 2-28 s of one MUSAN noise or
  music-without-vocals file (the private inventory of
  ``scripts.asr_finetune.musan``), noise or music with equal probability.
  10% of them are near-silence or low-level noise (U(-75, -50) dBFS), the
  others take the speech windows' levels. ``r5`` (round 5): 10% of all
  windows, 70% of them short spans of 0.3-2.5 s (the decode's phantom
  windows are short), the others 2-28 s, and 20% of them low-level
  (U(-50, -35) dBFS: windows that reach the decode passed a VAD, so they
  are rarely silent). Dev files (1% of the files of at least 2 s) give the
  dev share; never positive or anchor; the 0.5 s pads as always; each is a
  window of its own, never merged into another. Their own generators
  (``SEED + 2``, then ``SEED + 3`` to shuffle them in) leave the speech
  windows as they are. ``scripts.asr_finetune.windows`` labels them
  ``labels.NOSPEECH_LABELS``.

The windows are shuffled and numbered ``w`` 0..N-1. Each JSON line::

    {"w": 123, "shape": "P", "split": "train", "anchor": false, "positive": true, "source": "galgame",
     "level_dbfs": -24.0,
     "utts": [{"src": "galgame", "id": "GalGame/<maker>_<game>.7z|<voice>", "gap_before_s": 0.0, "gain_db": -2.1}]}

An ``N`` window's one utterance is ``{"src": "musan", "id":
"musan/noise/<source>/<file>.wav", "start_s", "duration_s", "gap_before_s":
0.0, "gain_db"}``.

``id`` is the allow-list id: the ReazonSpeech key, the Common Voice clip
``path``, ``<archive>|<voice>`` for galgame, the JVNV utterance id, the JNV
member name. The output is gzip (no name, mtime 0). Without ``--smoke`` it
must pass ``check_outbox_file``; an earlier plan at ``--out`` is deleted
first and a refused plan is deleted (exit 1). ``--outbox`` (default
``<pilot>\\outbox``) is the outbox folder: ``--allow`` and ``--out`` default to
its ``allow.json`` and ``windows-plan.jsonl.gz``, and the check takes it as
its outbox root (``select._outbox_root``), e.g. for a fallback built beside
the current outbox with ``select --outbox``.

``--smoke <rows>``: the rows of one ReazonSpeech rows file (``key``,
``caption``, ``duration_s``) are the items, no allow-list; ``--windows N``
keeps the first N windows of the shuffled plan; no outbox check.

Console: windows per shape, source and split, positive and anchor shares,
JNV splices, skips, planned hours; counts only. Exit 0, 1 on an outbox
refusal, 2 on a missing or misshapen input.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import random
import sys
import time
from collections import Counter, deque
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.asr_bench.lexicon import analyze
from scripts.asr_finetune.labels import MAX_LABEL_TOKENS, label_length, load_tokenizer
from scripts.asr_finetune.outbox import OutboxRefusal, check_outbox_file, outbox_dir
from scripts.asr_finetune.pilot import SEED, clean_text, pilot_root
from scripts.asr_finetune.select import ALLOW_SCHEMA, ROWS, SOURCES, SelectError, _jsonl, _label, _outbox_root

HOUR = 3600.0
PAD_S = 0.5
SLACK_S = 0.1
GAP_S = (0.1, 1.0)
REACTION_GAP_S = (0.1, 0.4)
LEVEL_DBFS = (-30.0, -18.0)
GAIN_DB = (-3.0, 3.0)
P_SHARE = 0.40
GALGAME_POSITIVE = 0.5
JNV_SPLICE_EVERY = 10  # at most one spliced window in ten
JNV_USES = 2  # each JNV clip in at most two windows, like a galgame interjection line
DEV_SHARE = 0.01
LOOKAHEAD = 64
WINDOW_SOURCES = ("reazon", "cv", "galgame", "jvnv")
ANCHOR_SOURCES = frozenset({"reazon", "cv"})
SPLICED_SOURCES = ("reazon", "cv")
TEXT_FIELD = {"reazon": "caption", "cv": "sentence", "galgame": "text", "jvnv": "text", "jnv": "phrase"}
# Cut windows (--cuts, round 3): P windows of these sources open or close
# mid-utterance, as production windows do after a max-chunk cut.
CUT_SOURCES = ("reazon", "galgame")
CUT_SHARE = 0.30  # of those P windows, drawn apart for the head and the tail
CUT_MIN_S = 2.0  # an utterance shorter than this is never cut
CUT_BOTH_MIN_S = 3.0  # one utterance cut at both ends needs this
CUT_HEAD = (0.2, 0.6)  # target cut point, as a fraction of the utterance
CUT_TAIL = (0.4, 0.8)
CUT_BOTH = ((0.15, 0.35), (0.65, 0.85))


@dataclass(frozen=True)
class Shape:
    cap_s: float
    fewest: int
    most: int


SHAPES = {"P": Shape(8.0, 1, 4), "D": Shape(28.5, 2, 6), "N": Shape(28.5, 1, 1)}
# No-speech windows (--nonspeech, round 4): MUSAN noise and music without
# vocals (scripts.asr_finetune.musan), shape N, label <|nospeech|>.
NONSPEECH_SOURCE = "musan"


@dataclass(frozen=True)
class NonspeechProfile:
    share: float  # of all windows
    short_share: float  # of the no-speech windows
    short_s: tuple[float, float]
    long_s: tuple[float, float]  # long_s[0]: the shortest file used
    quiet_share: float  # of the no-speech windows, at quiet_dbfs without gain
    quiet_dbfs: tuple[float, float]


NONSPEECH_PROFILES = {
    "r4": NonspeechProfile(0.08, 0.0, (0.3, 2.5), (2.0, 28.0), 0.10, (-75.0, -50.0)),
    "r5": NonspeechProfile(0.10, 0.70, (0.3, 2.5), (2.0, 28.0), 0.20, (-50.0, -35.0)),
}


class MixError(RuntimeError):
    """A missing or misshapen input (exit 2)."""


def line_mark(src: str, text: str) -> bool:
    """A marked line: a JNV clip, or a cleaned text holding a lexicon interjection."""
    return src == "jnv" or bool(analyze(text).interjections)


def window_flags(source: str, marks: Sequence[bool], cut: bool = False) -> tuple[bool, bool]:
    """(positive, anchor) of a ``source`` window whose lines have ``marks``;
    a ``cut`` window (one with a cut mark) is never an anchor."""
    return len(marks) >= 2 and any(marks), source in ANCHOR_SOURCES and not any(marks) and not cut


@dataclass(frozen=True)
class Item:
    """One allow-listed item: its plan id, window group (the source, or the
    galgame archive), S1 seconds, label text tokens, mark and uses (Common
    Voice copies, galgame reuse)."""

    src: str
    id: str
    group: str
    seconds: float
    tokens: int
    marked: bool
    uses: int = 1


@dataclass
class Window:
    source: str
    shape: str
    split: str
    units: list[Item]
    gaps: list[float]
    level: float
    gains: list[float]
    positive: bool
    anchor: bool
    cut_start: float | None = None  # head cut target of units[0], a fraction of it
    cut_end: float | None = None  # tail cut target of units[-1]
    span: tuple[float, float] | None = None  # N window: (start_s, duration_s) in its file

    @property
    def speech_s(self) -> float:
        return sum(unit.seconds for unit in self.units)

    @property
    def content_s(self) -> float:
        return self.speech_s + sum(self.gaps)

    @property
    def cut_s(self) -> float:
        """Planned seconds the cuts remove (at the targets, S1 durations)."""
        return ((self.cut_start or 0.0) * self.units[0].seconds
                + (0.0 if self.cut_end is None else (1 - self.cut_end) * self.units[-1].seconds))

    def row(self, number: int) -> dict[str, Any]:
        utts = [{"src": unit.src, "id": unit.id, "gap_before_s": gap, "gain_db": gain}
                for unit, gap, gain in zip(self.units, self.gaps, self.gains)]
        if self.cut_start is not None:
            utts[0]["cut_start"] = self.cut_start
        if self.cut_end is not None:
            utts[-1]["cut_end"] = self.cut_end
        if self.span is not None:
            utts[0].update(start_s=self.span[0], duration_s=self.span[1])
        return {"w": number, "shape": self.shape, "split": self.split, "anchor": self.anchor,
                "positive": self.positive, "source": self.source, "level_dbfs": self.level, "utts": utts}


# --- items ---------------------------------------------------------------------


def make_item(src: str, ident: str, raw: Any, seconds: Any, uses: int, tokenizer: Any) -> Item | None:
    """The item of a source text, or None when ``clean_text`` drops it."""
    if not isinstance(raw, str):
        raise MixError(f"{src}: a row text is not a string")
    text = clean_text(src, raw)
    if text is None:
        return None
    group = ident.split("|", 1)[0] if src == "galgame" else src
    return Item(src, ident, group, float(seconds), len(tokenizer.encode(text, add_special_tokens=False).ids),
                line_mark(src, text), uses)


def allowed_ids(allow: Any) -> dict[str, dict[str, int]]:
    """Source -> plan id -> uses, in allow-list order."""
    try:
        if allow.get("schema") != ALLOW_SCHEMA:
            raise MixError(f"the allow-list is not a {ALLOW_SCHEMA} file")
        return {
            "reazon": {key: 1 for key in allow["reazon"]["keys"]},
            "cv": {clip["path"]: int(clip["copies"]) for clip in allow["cv"]["clips"]},
            "galgame": {f"{line['archive']}|{line['voice']}": int(line["reuse"]) for line in allow["galgame"]["lines"]},
            "jvnv": {ident: 1 for ident in allow["jvnv"]["ids"]},
            "jnv": {name: 1 for name in allow["jnv"]["files"]},
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        raise MixError(f"the allow-list is not shaped as a {ALLOW_SCHEMA} file") from None


def row_id(source: str, row: dict[str, Any]) -> str:
    if source == "galgame":
        return f"{row['archive']}|{row['voice']}"
    return row[{"reazon": "key", "cv": "path", "jvnv": "id", "jnv": "file"}[source]]


def load_items(allowed: dict[str, dict[str, int]], rows_dir: Path, tokenizer: Any) -> dict[str, list[Item]]:
    items: dict[str, list[Item]] = {source: [] for source in SOURCES}
    for source in SOURCES:
        wanted = allowed[source]
        if not wanted:
            continue
        path = rows_dir / "stage" / ROWS[source]
        try:
            rows, _ = _jsonl(path)
            by_id = {row_id(source, row): row for row in rows}
            missing = sum(ident not in by_id for ident in wanted)
            if missing:
                raise MixError(f"{source}: {missing:,} allow-listed items have no row in {ROWS[source]}")
            for ident, uses in wanted.items():
                row = by_id[ident]
                item = make_item(source, ident, row[TEXT_FIELD[source]], row["duration_s"], uses, tokenizer)
                if item is None:
                    raise MixError(f"{source}: clean_text drops an allow-listed item's text; "
                                   "the allow-list and the rows disagree")
                items[source].append(item)
        except SelectError as error:
            raise MixError(str(error)) from None
        except KeyError as error:
            raise MixError(f"{ROWS[source]}: a row has no {error.args[0]!r}") from None
        except (TypeError, ValueError) as error:  # the type only: a message may hold a text
            raise MixError(f"{ROWS[source]}: a row has a malformed field ({type(error).__name__})") from None
        analyze.cache_clear()
    return items


def smoke_items(path: Path, tokenizer: Any) -> tuple[dict[str, list[Item]], int]:
    """Every row of a ReazonSpeech rows file as an item; and the rows ``clean_text`` drops."""
    try:
        rows, _ = _jsonl(path)
        made = [make_item("reazon", row["key"], row["caption"], row["duration_s"], 1, tokenizer) for row in rows]
    except SelectError as error:
        raise MixError(str(error)) from None
    except (KeyError, TypeError, ValueError) as error:
        raise MixError(f"{path.name}: a row is malformed ({type(error).__name__})") from None
    items = {source: [] for source in SOURCES}
    items["reazon"] = [item for item in made if item is not None]
    return items, len(made) - len(items["reazon"])


def split_dev(items: Sequence[Item], rng: random.Random) -> tuple[list[Item], list[Item]]:
    """(train, dev): ``DEV_SHARE`` of the items, by id, drawn with ``rng``."""
    ids = sorted({item.id for item in items})
    dev = set(rng.sample(ids, round(len(ids) * DEV_SHARE)))
    return [item for item in items if item.id not in dev], [item for item in items if item.id in dev]


# --- composition ---------------------------------------------------------------


def fits(units: Sequence[Item], unit: Item, cap_s: float) -> bool:
    """Whether ``unit`` joins ``units`` within ``cap_s`` of content (slack and
    the largest gaps counted) and the label token limit, its id not in them."""
    if any(other.id == unit.id for other in units):
        return False
    count = len(units) + 1
    seconds = sum(other.seconds for other in units) + unit.seconds + SLACK_S * count + GAP_S[1] * (count - 1)
    return (seconds <= cap_s
            and label_length([other.tokens for other in units] + [unit.tokens]) <= MAX_LABEL_TOKENS)


def take(pool: MutableSequence[Item], units: Sequence[Item], cap_s: float, *,
         marked: bool = False, scan: int | None = LOOKAHEAD) -> Item | None:
    """Remove and return the first of the first ``scan`` (None: all) of
    ``pool`` that fits beside ``units`` (and is marked, with ``marked``)."""
    for index, unit in enumerate(itertools.islice(pool, scan)):
        if (unit.marked or not marked) and fits(units, unit, cap_s):
            del pool[index]
            return unit
    return None


class Cycle:
    """The JNV clips of a split, shuffled, handed out in turn, each at most
    ``JNV_USES`` times."""

    def __init__(self, units: Sequence[Item]) -> None:
        self.units, self.next = list(units), 0

    def peek(self) -> Item | None:
        return self.units[self.next % len(self.units)] if self.next < len(self.units) * JNV_USES else None

    def advance(self) -> None:
        self.next += 1


def splice_budgets(chosen: dict[str, Sequence[Item]], clips: int) -> dict[str, int]:
    """The JNV uses of a split (``clips`` x ``JNV_USES``) shared between the
    spliced sources by their planned seconds (Common Voice copies counted)."""
    seconds = {source: sum(item.seconds * (item.uses if source == "cv" else 1) for item in chosen[source])
               for source in SPLICED_SOURCES}
    total, budgets = clips * JNV_USES, {}
    whole = sum(seconds.values())
    for source in SPLICED_SOURCES[:-1]:
        budgets[source] = round(total * seconds[source] / whole) if whole else 0
    budgets[SPLICED_SOURCES[-1]] = total - sum(budgets.values())
    return budgets


class Planner:
    """The windows of one source and split."""

    def __init__(self, source: str, split: str, rng: random.Random, jnv: Cycle | None, splice_budget: int = 0) -> None:
        self.source, self.split, self.rng, self.jnv = source, split, rng, jnv
        self.seconds = {"P": 0.0, "D": 0.0}
        self.windows: list[Window] = []
        self.spliced, self.splice_budget = 0, splice_budget
        self.reused = 0
        self.skipped: Counter = Counter()

    def plan_group(self, items: Sequence[Item]) -> None:
        """Windows from one group's items (one source, or one galgame archive)."""
        rng = self.rng
        usable = []
        for item in items:
            if not fits([], item, SHAPES["D"].cap_s):
                self.skipped["over_window" if item.seconds + SLACK_S > SHAPES["D"].cap_s else "over_tokens"] += 1
            else:
                usable.append(item)
        units = [item for item in usable for _ in range(item.uses if item.src == "cv" else 1)]
        rng.shuffle(units)
        queue = deque(units)
        reserve = [item for item in usable if item.src == "galgame" for _ in range(item.uses - 1)]
        rng.shuffle(reserve)
        count = positives = 0
        while queue:
            want = self.source == "galgame" and positives < GALGAME_POSITIVE * (count + 1)
            window = self._window(queue, reserve, want)
            if window is not None:
                self.windows.append(window)
                count += 1
                positives += window.positive

    def _window(self, queue: deque[Item], reserve: list[Item], want_positive: bool) -> Window | None:
        rng = self.rng
        shape = "P" if self.seconds["P"] < P_SHARE * sum(self.seconds.values()) else "D"
        units: list[Item] = []
        designated: Item | None = None
        if (self.jnv is not None and self.spliced < self.splice_budget
                and (self.spliced + 1) * JNV_SPLICE_EVERY <= len(self.windows) + 1):
            clip = self.jnv.peek()
            head = None if clip is None else take(queue, [clip], SHAPES[shape].cap_s)
            if head is not None:
                self.jnv.advance()
                designated, units = clip, [head, clip]
        if not units:
            head = take(queue, [], SHAPES[shape].cap_s)
            if head is None and shape == "P":
                shape = "D"
                head = take(queue, [], SHAPES[shape].cap_s)
            assert head is not None, "every queued item fits a D window alone"
            units = [head]
            if want_positive:
                if head.marked:
                    designated = head
                else:
                    designated = take(queue, units, SHAPES[shape].cap_s, marked=True)
                    if designated is None:
                        designated = take(reserve, units, SHAPES[shape].cap_s, marked=True)
                        self.reused += designated is not None
                    if designated is not None:
                        units.append(designated)
        cap_s = SHAPES[shape].cap_s
        while len(units) < SHAPES[shape].most:
            unit = take(queue, units, cap_s)
            if unit is None:
                break
            units.append(unit)
        if len(units) < SHAPES[shape].fewest:
            unit = take(queue, units, cap_s, scan=None)
            if unit is not None:
                units.append(unit)
            elif fits([], units[0], SHAPES["P"].cap_s):
                shape = "P"
            else:
                self.skipped["unpaired"] += 1
                return None
        designated = designated or next((unit for unit in units if unit.marked), None)
        if designated is not None and len(units) >= 2:
            at = next(index for index, unit in enumerate(units) if unit is designated)
            others = units[:at] + units[at + 1:]
            place = rng.choice(("first", "last", "middle") if len(others) >= 2 else ("first", "last"))
            index = 0 if place == "first" else len(others) if place == "last" else rng.randint(1, len(others) - 1)
            units = others[:index] + [designated] + others[index:]
        positive, anchor = window_flags(self.source, [unit.marked for unit in units])
        reaction = positive and rng.random() < 0.5
        gaps = [0.0] + [round(rng.uniform(*(REACTION_GAP_S if reaction and before.marked else GAP_S)), 2)
                        for before in units[:-1]]
        level = round(rng.uniform(*LEVEL_DBFS), 1)
        gains = [round(rng.uniform(*GAIN_DB), 1) for _ in units]
        window = Window(self.source, shape, self.split, units, gaps, level, gains, positive, anchor)
        self.seconds[shape] += window.content_s
        self.spliced += any(unit.src == "jnv" for unit in units)
        return window


def compose(items: dict[str, list[Item]]) -> tuple[list[Window], list[Planner]]:
    """Every window of the plan, shuffled; and the planners (for the summary)."""
    dev_rng, rng = random.Random(SEED), random.Random(SEED)
    splits = {source: split_dev(items[source], dev_rng) for source in SOURCES}
    planners: list[Planner] = []
    for index, split in enumerate(("train", "dev")):
        clips = list(splits["jnv"][index])
        rng.shuffle(clips)
        jnv = Cycle(clips) if clips else None
        budgets = splice_budgets({source: splits[source][index] for source in SPLICED_SOURCES}, len(clips))
        for source in WINDOW_SOURCES:
            chosen = splits[source][index]
            if not chosen:
                continue
            spliced = source in SPLICED_SOURCES
            planner = Planner(source, split, rng, jnv if spliced else None, budgets[source] if spliced else 0)
            groups: dict[str, list[Item]] = {}
            for item in chosen:
                groups.setdefault(item.group, []).append(item)
            for group in groups.values():
                planner.plan_group(group)
            planners.append(planner)
    windows = [window for planner in planners for window in planner.windows]
    rng.shuffle(windows)
    return windows, planners


def nonspeech_windows(inventory: dict[str, Any], speech: Sequence[Window],
                      profile: NonspeechProfile = NONSPEECH_PROFILES["r4"]) -> list[Window]:
    """No-speech windows (round 4), ``profile.share`` of all windows, from the
    MUSAN inventory (``scripts.asr_finetune.musan``) with
    ``random.Random(SEED + 2)``.

    Files of at least ``profile.long_s[0]`` seconds; 1% of them (by path) are
    dev files, and the dev windows (the speech windows' dev share of them)
    come only from those. Each window takes noise or music with equal
    probability, a file of that kind, a span at a uniform start of
    U(``short_s``) seconds (``short_share`` of the windows) or U(``long_s``)
    (both capped at the file), and its level: ``quiet_share`` of the windows
    U(``quiet_dbfs``), the others the speech windows' U(-30, -18) dBFS plus
    U(-3, +3) dB. Never positive or anchor. A profile without short windows
    draws no short choice, so ``r4`` keeps its round-4 windows."""
    rng = random.Random(SEED + 2)
    files = sorted((f for f in inventory["files"] if f["seconds"] >= profile.long_s[0]), key=lambda f: f["path"])
    dev_files = {f["path"] for f in rng.sample(files, max(1, round(len(files) * DEV_SHARE)))}
    count = round(profile.share / (1 - profile.share) * len(speech))
    dev_share = sum(window.split == "dev" for window in speech) / max(len(speech), 1)
    dev_count = round(count * dev_share)
    windows: list[Window] = []
    for number in range(count):
        split = "dev" if number < dev_count else "train"
        kind = rng.choice(("noise", "music"))
        pool = [f for f in files if f["kind"] == kind and (f["path"] in dev_files) == (split == "dev")]
        pool = pool or [f for f in files if (f["path"] in dev_files) == (split == "dev")]
        chosen = rng.choice(pool)
        short = profile.short_share > 0 and rng.random() < profile.short_share
        low, high = profile.short_s if short else profile.long_s
        seconds = round(rng.uniform(low, min(high, chosen["seconds"])), 2)
        start = round(rng.uniform(0.0, chosen["seconds"] - seconds), 2)
        quiet = rng.random() < profile.quiet_share
        level = round(rng.uniform(*profile.quiet_dbfs) if quiet else rng.uniform(*LEVEL_DBFS), 1)
        gain = 0.0 if quiet else round(rng.uniform(*GAIN_DB), 1)
        unit = Item(NONSPEECH_SOURCE, chosen["path"], NONSPEECH_SOURCE, seconds, 0, False)
        windows.append(Window(NONSPEECH_SOURCE, "N", split, [unit], [0.0], level, [gain], False, False,
                              span=(start, seconds)))
    return windows


def add_nonspeech(windows: list[Window], inventory: dict[str, Any],
                  profile: NonspeechProfile = NONSPEECH_PROFILES["r4"]) -> list[Window]:
    """``windows`` with the no-speech windows mixed in (``random.Random(SEED + 3)``)."""
    mixed = windows + nonspeech_windows(inventory, windows, profile)
    random.Random(SEED + 3).shuffle(mixed)
    return mixed


def mark_cuts(windows: Sequence[Window]) -> None:
    """Cut marks (round 3) on P windows of ``CUT_SOURCES``, in ``w`` order
    with their own ``random.Random(SEED + 1)``, so the windows themselves
    stay those of the plan without cuts. Each window draws a head cut and a
    tail cut apart, each with ``CUT_SHARE``; a side is marked only when its
    utterance is of the window's source (not a spliced JNV clip) and at
    least ``CUT_MIN_S`` long (``CUT_BOTH_MIN_S`` for one utterance cut at
    both ends, else it keeps the head cut). A marked window is never an anchor."""
    rng = random.Random(SEED + 1)
    for window in windows:
        if window.shape != "P" or window.source not in CUT_SOURCES:
            continue
        head, tail = rng.random() < CUT_SHARE, rng.random() < CUT_SHARE
        first, last = window.units[0], window.units[-1]
        head = head and first.src == window.source and first.seconds >= CUT_MIN_S
        tail = tail and last.src == window.source and last.seconds >= CUT_MIN_S
        if head and tail and len(window.units) == 1:
            if first.seconds >= CUT_BOTH_MIN_S:
                window.cut_start = round(rng.uniform(*CUT_BOTH[0]), 3)
                window.cut_end = round(rng.uniform(*CUT_BOTH[1]), 3)
                window.anchor = False
                continue
            tail = False
        if head:
            window.cut_start = round(rng.uniform(*CUT_HEAD), 3)
        if tail:
            window.cut_end = round(rng.uniform(*CUT_TAIL), 3)
        if head or tail:
            window.anchor = False


# --- output --------------------------------------------------------------------


def write_plan(path: Path, windows: Sequence[Window]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        for number, window in enumerate(windows):
            handle.write((json.dumps(window.row(number), ensure_ascii=False) + "\n").encode("utf-8"))


def _h(seconds: float) -> str:
    return f"{seconds / HOUR:,.2f} h"


def _share(part: float, whole: float) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def summary(windows: Sequence[Window], planners: Sequence[Planner]) -> None:
    content = sum(window.content_s for window in windows)
    speech = sum(window.speech_s for window in windows)
    splits = Counter(window.split for window in windows)
    print(f"windows: {len(windows):,} (train {splits['train']:,}, dev {splits['dev']:,}); planned content "
          f"{_h(content)} ({_h(speech)} of utterances at their S1 durations, {_h(content - speech)} of gaps), "
          f"{_h(content + 2 * PAD_S * len(windows))} with the 0.5 s pads")
    for shape in SHAPES:
        chosen = [window for window in windows if window.shape == shape]
        seconds = sum(window.content_s for window in chosen)
        sizes = Counter(len(window.units) for window in chosen)
        print(f"  {shape}: {len(chosen):,} windows, {_h(seconds)} ({_share(seconds, content)} of content hours), "
              f"mean content {seconds / max(len(chosen), 1):.1f} s, utterances "
              + ", ".join(f"{size}: {sizes[size]:,}" for size in sorted(sizes)))
    for source in WINDOW_SOURCES:
        chosen = [window for window in windows if window.source == source]
        if not chosen:
            continue
        seconds = sum(window.content_s for window in chosen)
        shapes = Counter(window.shape for window in chosen)
        positive = sum(window.positive for window in chosen)
        anchor = sum(window.anchor for window in chosen)
        spliced = sum(any(unit.src == "jnv" for unit in window.units) for window in chosen)
        dev = sum(window.split == "dev" for window in chosen)
        print(f"  {source}: {len(chosen):,} windows (P {shapes['P']:,}, D {shapes['D']:,}, dev {dev:,}), {_h(seconds)}; "
              f"positive {_share(positive, len(chosen))}, anchor {_share(anchor, len(chosen))}"
              + (f", JNV splices {spliced:,} ({_share(spliced, len(chosen))})" if source in SPLICED_SOURCES else ""))
    quiet = [w for w in windows if w.shape == "N" and w.level < LEVEL_DBFS[0] + GAIN_DB[0]]
    if any(w.shape == "N" for w in windows):
        nonspeech = [w for w in windows if w.shape == "N"]
        kinds = Counter(w.units[0].id.split("/")[1] for w in nonspeech)
        print(f"  {NONSPEECH_SOURCE} (no-speech): {len(nonspeech):,} windows ({_share(len(nonspeech), len(windows))} "
              f"of all; dev {sum(w.split == 'dev' for w in nonspeech):,}), {_h(sum(w.content_s for w in nonspeech))}; "
              f"noise {kinds['noise']:,}, music {kinds['music']:,}; near-silence or low-level {len(quiet):,}; "
              f"{len({w.units[0].id for w in nonspeech}):,} files")
        spans = {"under 2 s": (0.0, 2.0), "2-2.5 s": (2.0, 2.5), "2.5 s or more": (2.5, float("inf"))}
        print("    spans: " + ", ".join(
            f"{name} {len(part):,} ({_h(sum(w.content_s for w in part))})"
            for name, (low, high) in spans.items()
            for part in [[w for w in nonspeech if low <= w.content_s < high]]))
    jnv = Counter(unit.id for window in windows for unit in window.units if unit.src == "jnv")
    if jnv:
        print(f"  jnv: {len(jnv):,} clips spliced, {sum(jnv.values()):,} splices, at most {max(jnv.values())} per clip")
    reused = sum(planner.reused for planner in planners)
    skipped: Counter = Counter()
    for planner in planners:
        skipped.update(planner.skipped)
    print(f"  galgame second uses (reuse 2): {reused:,}; skipped items: "
          + (", ".join(f"{reason} {count:,}" for reason, count in sorted(skipped.items())) or "none"))
    for source in CUT_SOURCES:
        chosen = [window for window in windows if window.source == source and window.shape == "P"]
        cut = [window for window in chosen if window.cut_start is not None or window.cut_end is not None]
        if not cut:
            continue
        head = sum(window.cut_start is not None for window in cut)
        tail = sum(window.cut_end is not None for window in cut)
        both = sum(window.cut_start is not None and window.cut_end is not None for window in cut)
        print(f"  cuts, {source}: {len(cut):,} of {len(chosen):,} P windows ({_share(len(cut), len(chosen))}): "
              f"head {head:,} ({_share(head, len(chosen))}), tail {tail:,} ({_share(tail, len(chosen))}), "
              f"both {both:,}; planned {_h(sum(window.cut_s for window in cut))} cut away at the targets")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.mix", description=__doc__.splitlines()[0])
    parser.add_argument("--outbox", type=Path, default=outbox_dir(),
                        help="the outbox folder: default of --allow and --out, root of the outbox check "
                             "(default <pilot>\\outbox)")
    parser.add_argument("--allow", type=Path, help="the allow-list (default <outbox>\\allow.json)")
    parser.add_argument("--rows", type=Path, default=pilot_root() / "pods" / "s1",
                        help="pod S1's fetched outputs, rows under stage\\ (default <pilot>\\pods\\s1)")
    parser.add_argument("--out", type=Path, help="the window plan (default <outbox>\\windows-plan.jsonl.gz)")
    parser.add_argument("--smoke", type=Path, help="plan from this ReazonSpeech rows file only (probe)")
    parser.add_argument("--windows", type=int, help="with --smoke: keep the first N windows of the plan")
    parser.add_argument("--cuts", action="store_true",
                        help="mark cut windows (round 3: P windows that open or close mid-utterance)")
    parser.add_argument("--nonspeech", type=Path, metavar="INVENTORY",
                        help="add no-speech windows from this MUSAN inventory (scripts.asr_finetune.musan; round 4)")
    parser.add_argument("--nonspeech-profile", choices=tuple(NONSPEECH_PROFILES), default="r4",
                        help="the no-speech windows' share, spans and levels (NONSPEECH_PROFILES: r4, round 4, "
                             "default; r5, round 5: 10%% of the windows, 70%% of them 0.3-2.5 s)")
    args = parser.parse_args(argv)
    args.allow = args.allow or args.outbox / "allow.json"
    args.out = args.out or args.outbox / "windows-plan.jsonl.gz"
    if args.windows is not None and (args.smoke is None or args.windows < 1):
        parser.error("--windows takes a positive count and needs --smoke")
    started = time.monotonic()
    if args.smoke is None:
        args.out.unlink(missing_ok=True)
    try:
        tokenizer = load_tokenizer()
        if args.smoke is not None:
            items, dropped = smoke_items(args.smoke, tokenizer)
            print(f"smoke: {len(items['reazon']):,} ReazonSpeech rows as items ({dropped:,} dropped by clean_text)")
        else:
            try:
                allow = json.loads(args.allow.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise MixError(f"cannot read the allow-list: {type(error).__name__}") from None
            items = load_items(allowed_ids(allow), args.rows, tokenizer)
            print("items: " + ", ".join(f"{source} {len(items[source]):,}" for source in SOURCES))
    except MixError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    windows, planners = compose(items)
    if args.cuts:
        mark_cuts(windows)
    if args.nonspeech is not None:
        try:
            inventory = json.loads(args.nonspeech.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"error: cannot read the MUSAN inventory: {type(error).__name__}", file=sys.stderr)
            return 2
        windows = add_nonspeech(windows, inventory, NONSPEECH_PROFILES[args.nonspeech_profile])
    if args.windows is not None:
        print(f"smoke: {len(windows):,} windows planned, the first {min(args.windows, len(windows)):,} kept")
        windows = windows[:args.windows]
    summary(windows, planners)
    write_plan(args.out, windows)
    if args.smoke is None:
        try:
            with _outbox_root(args.outbox):
                check_outbox_file(args.out)
        except OutboxRefusal as refusal:
            args.out.unlink(missing_ok=True)
            print(f"refused, the plan deleted: {refusal}", file=sys.stderr)
            return 1
        print(f"wrote {_label(args.out.parent)}\\{args.out.name} ({args.out.stat().st_size:,} bytes); "
              "check_outbox_file passed"
              + ("" if args.outbox.resolve() == outbox_dir().resolve() else " with that folder as the outbox root"))
    else:
        print(f"wrote {args.out.name} ({args.out.stat().st_size:,} bytes)")
    print(f"mix: {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
