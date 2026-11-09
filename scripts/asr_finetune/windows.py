"""Materialize the pilot's training windows from the window plan (pod task T1).

    python -m scripts.asr_finetune.windows --plan <windows-plan.jsonl.gz> --data <dir> --out <dir> [--audio <dir>]
        [--words <words.jsonl>] [--drop-no-speech] [--workers N]

Reads the plan (``scripts.asr_finetune.mix``) and only what
``scripts.asr_finetune.fetch`` put in ``--data`` for ``fetch-t1.json``:

- ReazonSpeech: the captions of ``small.tsv`` or of the mirror's ``all.tsv``
  (``stage.reazon_labels``; ``--data`` must hold exactly one of the two) and
  the FLAC members ``<shard>/<name>.flac`` of ``<shard>.tar``;
- Common Voice: one pass over the archive for ``validated.tsv`` (the
  ``sentence`` of each ``path``) and the plan's clips (``clips/<path>``),
  held in memory;
- galgame: each archive the plan names, extracted with ``7z x`` into
  ``<data>\\galgame\\<archive stem>\\`` as ``stage`` does (a folder already
  there is replaced, unless ``<stem>.extracted`` beside it records a finished
  extraction of this module, e.g. by ``teacher --plan`` on the same pod), its
  ``index.json`` entries read with ``stage.galgame_lines``: the text and
  audio file of each voice path;
- JVNV: the ``text`` of ``jvnv_v1/transcription.csv`` and the ``regular``
  members of ``jvnv_ver1.zip``; JNV: the phrases of ``JNV/phrases.txt`` and
  the ``_R.wav`` members of ``jnv_corpus_ver3.zip``.

Besides these, the tokenizer (``labels.load_tokenizer``) and the production
Silero graph (``silero_chunker._default_model_path``) come from the Hugging
Face cache, downloaded at their pinned revisions when absent.

Before any audio, every utterance of the plan must be found, its text must
pass ``pilot.clean_text``, and each window's ``positive`` and ``anchor``
recomputed from the cleaned texts (``mix.line_mark``, ``mix.window_flags``)
must equal the plan's (a window with a cut mark is never an anchor). Any
miss: ``windows.json`` with the counts, exit 1, no audio written.

Cuts (round 3, ``mix --cuts``). A plan with ``cut_start`` or ``cut_end``
marks needs ``--words``: the teacher's words of each marked utterance
(``teacher --plan``; ``{"src", "id", "duration_s", "outcome", "words":
[[start_s, end_s, word], ...]}``, times in the clip; a missing file refuses
every mark as ``no_words``, counted, and the run goes on). ``resolve_cut`` cuts
at a gap between two teacher words that removes at least one word and keeps
at least 3 words and 1.0 s from the first kept word to the last, and where
the caption matches the teacher: the cleaned text and the teacher text are
aligned character by character (``scripts.asr_bench.align.alignment_trace``
on each character's ``pilot.norm``), and the 3 characters on each side of
the teacher's word boundary must be exact one-to-one matches. Of those gaps
it takes the one nearest the mark's target (its fraction of the clip) and
cuts the audio at the gap's middle. The label is the cleaned text after
(head) or before (tail) the matching point; punctuation at the cut is
dropped. Otherwise the cut is refused and the utterance stays whole:
``no_words`` (no teacher words), ``no_gap`` (no gap keeps 3 words and 1.0 s)
or ``mismatch`` (no such gap matches); refusals are counted. A cut start begins the audio at
the cut, right after the leading zeros, so its segment starts at the pad end
(0.50 s); a cut end ends it at the cut. The other end of a cut utterance is
Silero's, but never inside the kept teacher words. A cut window's
``positive`` is recomputed from its cut texts; ``anchor`` stays false.

No-speech windows (round 4, ``mix --nonspeech``, shape ``N``, source
``musan``): their MUSAN files are written from the fetched ``musan.tar.gz``
(``pilot.SOURCES`` ``musan``) below ``<data>\\musan-files\\`` in one pass;
each window is its file's span (``start_s``, ``duration_s``) scaled, without
Silero, to ``level_dbfs + gain_db`` dBFS RMS (peak at most 0.99; a digitally
silent span stays silent, counted ``nonspeech_silent``) between the two
0.5 s pads, labelled ``labels.NOSPEECH_LABELS`` with no segments, never
positive or anchor. ``--drop-no-speech`` (v4b) leaves out every speech
window holding an utterance with no Silero speech frame (reason
``no_speech``, counted, chosen rather than failed, so under no limit).

Each window (``--workers`` processes, default the CPU count): every
utterance decoded with ``faster_whisper.decode_audio`` (16 kHz mono float32)
and trimmed with the production Silero graph (``SileroVAD``, states reset per
utterance, 512-sample frames zero-padded at the end, speech where the frame
probability is at least 0.5): from 0.1 s before its first speech frame to
0.1 s after its last, within the clip (the whole clip when no frame is
speech, counted ``no_speech``). It is scaled so the RMS of its speech frames
(of the whole clip without one) is ``level_dbfs + gain_db`` dBFS, then down
so its peak is at most 0.99 (``peak_limited``). The window is 8,000 zero
samples (0.5 s), each utterance after its ``gap_before_s`` of zeros, then
8,000 zero samples. A window is left out and recorded, with the reason, when
an utterance does not decode (``decode_<error type>``), decodes to nothing
(``empty_audio``) or is digital silence (``silent``: no text is trained over
silence), when its content (from the end of the leading zeros to the start
of the trailing ones) exceeds its shape's cap (``mix.SHAPES``: 8.0 s P,
28.5 s D: ``over_cap``), when its label fails (``labels``), or on another
error (its type). More than 1% of the plan's windows left out, overall or
within one source or one split, is exit 1.

Outputs (``--out``; ``audio.i16`` and ``index.jsonl`` in ``--audio`` when
given, the pod's case: the audio stays beside the data while ``--out`` is
fetched, and ``--out`` then gets a copy of ``index.jsonl``):

- ``audio.i16``: 16 kHz int16 little-endian PCM (``round(sample * 32768)``),
  every window back to back in ``w`` order;
- ``index.jsonl``: per window ``{"w", "offset", "samples", "labels",
  "anchor", "positive", "split", "segments": [[start_s, end_s, src], ...]}``,
  with ``"cut": ["head", "tail"]`` (the sides applied) on a cut window;
  ``offset`` and ``samples`` count int16 samples of ``audio.i16``;
  ``labels`` is ``labels.build_labels`` over the segments with the cleaned
  (or cut) texts (timestamps: the trimmed bounds in the window, rounded to
  0.02 s); ``segments`` holds the label's timestamps in seconds and each
  utterance's source;
- ``windows.json`` (``ft-pilot-windows-1``): windows per shape, source and
  split, positive and anchor counts, hours (audio with the zeros, content,
  trimmed utterances, speech frames), ``no_speech`` and ``peak_limited``
  utterances, label token percentiles, the windows left out (count per
  reason and per source and reason, the groups over the limit, and each
  window's ``w``, source, split and reason), the cuts planned, applied and
  refused per reason and the positive flags they changed, wall seconds.

Console: counts and hours only, never a text or id; left-out windows per
source and reason. Exit 0; 1 on a plan disagreement, an item not found, or
more than 1% of the windows left out overall, in a source or in a split; 2
on an invalid plan, a missing input file, or cut marks without ``--words``.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import itertools
import json
import math
import os
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from scripts.asr_bench.align import alignment_trace
from scripts.asr_finetune.labels import NOSPEECH_LABELS, TS0, TS_STEP, build_labels, load_tokenizer, timestamp_token
from scripts.asr_finetune.mix import (
    CUT_SOURCES,
    NONSPEECH_SOURCE,
    PAD_S,
    SHAPES,
    SPLICED_SOURCES,
    WINDOW_SOURCES,
    line_mark,
    window_flags,
)
from scripts.asr_finetune.musan import extract as musan_extract
from scripts.asr_finetune.pilot import (
    OOPPEENN_PASSWORD_B64,
    REAZON_CAPTION_SOURCES,
    SOURCES,
    TEXT_SOURCES,
    clean_text,
    norm,
)
from scripts.asr_finetune.stage import (
    JNV_REGULAR,
    JVNV_REGULAR,
    StageError,
    _extract,
    _tsv_rows,
    galgame_lines,
    jnv_phrases,
    jvnv_script,
    reazon_labels,
)

OUT_SCHEMA = "ft-pilot-windows-1"
SAMPLE_RATE = 16_000
PAD = round(PAD_S * SAMPLE_RATE)
FRAME = 512
THRESHOLD = 0.5
MARGIN = round(0.1 * SAMPLE_RATE)
PEAK = 0.99
LEFT_OUT_LIMIT = 0.01
_TAR_BUFFER = 4 << 20
_PROGRESS = 5000
CUT_MIN_WORDS = 3
CUT_MIN_KEPT_S = 1.0
CUT_MATCH_SPAN = 3  # exact characters required on each side of a cut

Locator = tuple  # ("tar", path, offset, size) | ("zip", path, member) | ("file", path) | ("bytes", data)


class PlanError(ValueError):
    """An invalid plan or a missing input file (exit 2)."""


# --- plan ------------------------------------------------------------------------


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_row(row: Any, number: int) -> bool:
    if not isinstance(row, dict) or row.get("w") != number or row.get("shape") not in SHAPES:
        return False
    if row["shape"] == "N":
        return _valid_nonspeech(row)
    if (row.get("split") not in ("train", "dev") or not isinstance(row.get("positive"), bool)
            or not isinstance(row.get("anchor"), bool) or row.get("source") not in WINDOW_SOURCES
            or not _number(row.get("level_dbfs"))):
        return False
    utts = row.get("utts")
    spliced = ("jnv",) if row["source"] in SPLICED_SOURCES else ()
    return isinstance(utts, list) and bool(utts) and all(
        isinstance(utt, dict) and utt.get("src") in (row["source"], *spliced) and isinstance(utt.get("id"), str)
        and _number(utt.get("gap_before_s")) and utt["gap_before_s"] >= 0 and _number(utt.get("gain_db"))
        and _valid_cut(row, utts, index, utt)
        for index, utt in enumerate(utts))


def _valid_cut(row: dict[str, Any], utts: list[Any], index: int, utt: dict[str, Any]) -> bool:
    """Cut marks only on a P window of ``CUT_SOURCES``: ``cut_start`` on its
    first utterance, ``cut_end`` on its last, each a fraction in (0, 1), on an
    utterance of the window's source, the start before the end."""
    marks = [name for name in ("cut_start", "cut_end") if name in utt]
    if not marks:
        return True
    if row["shape"] != "P" or row["source"] not in CUT_SOURCES or utt["src"] != row["source"]:
        return False
    if "cut_start" in utt and index != 0 or "cut_end" in utt and index != len(utts) - 1:
        return False
    if not all(_number(utt[name]) and 0 < utt[name] < 1 for name in marks):
        return False
    return len(marks) == 1 or utt["cut_start"] < utt["cut_end"]


def cut_marked(row: dict[str, Any]) -> bool:
    return any("cut_start" in utt or "cut_end" in utt for utt in row["utts"])


def _valid_nonspeech(row: dict[str, Any]) -> bool:
    """A no-speech window (round 4): source ``musan``, never positive or
    anchor, one utterance of a ``musan/**.wav`` file with its span."""
    utts = row.get("utts")
    if (row.get("split") not in ("train", "dev") or row.get("positive") is not False
            or row.get("anchor") is not False or row.get("source") != NONSPEECH_SOURCE
            or not _number(row.get("level_dbfs")) or not isinstance(utts, list) or len(utts) != 1
            or not isinstance(utts[0], dict)):
        return False
    utt = utts[0]
    ident = utt.get("id")
    return (utt.get("src") == NONSPEECH_SOURCE and isinstance(ident, str) and ident.startswith("musan/")
            and ident.endswith(".wav") and ".." not in PurePosixPath(ident).parts
            and _number(utt.get("start_s")) and utt["start_s"] >= 0
            and _number(utt.get("duration_s")) and 0 < utt["duration_s"] <= SHAPES["N"].cap_s
            and utt.get("gap_before_s") == 0 and _number(utt.get("gain_db")))


def read_plan(path: Path) -> list[dict[str, Any]]:
    """The plan's windows, numbered ``w`` 0..N-1 in order."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError:
        raise PlanError(f"{path} is missing") from None
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as error:
        raise PlanError(f"cannot read the plan: {type(error).__name__}") from None
    bad = next((number for number, row in enumerate(rows) if not _valid_row(row, number)), None)
    if bad is not None:
        raise PlanError(f"plan line {bad + 1} is not a well-formed window numbered {bad} "
                        "(known shape, split, source and flags; utterances of its source or spliced JNV)")
    if not rows:
        raise PlanError("the plan holds no window")
    return rows


# --- sources ---------------------------------------------------------------------


def _need(data: Path, name: str) -> Path:
    path = data / SOURCES[name].file
    if not path.is_file():
        raise PlanError(f"{SOURCES[name].file} is not in --data")
    return path


def reazon_captions(data: Path) -> Path:
    """The one ReazonSpeech caption file in ``data``: ``small.tsv`` (corpus
    server) or ``all.tsv`` (mirror), whichever ``fetch-t1.json`` named."""
    present = [data / SOURCES[name].file for name in REAZON_CAPTION_SOURCES if (data / SOURCES[name].file).is_file()]
    if len(present) != 1:
        files = " or ".join(SOURCES[name].file for name in REAZON_CAPTION_SOURCES)
        raise PlanError(f"--data must hold one ReazonSpeech caption file ({files}), not {len(present)}")
    return present[0]


def reazon_sources(data: Path, keys: set[str]) -> dict[str, tuple[str, Locator]]:
    shards = sorted({key.split("/", 1)[0] for key in keys if f"reazon-{key.split('/', 1)[0]}" in SOURCES})
    labels, _ = reazon_labels(reazon_captions(data), shards)
    found: dict[str, tuple[str, Locator]] = {}
    for shard in shards:
        path = _need(data, f"reazon-{shard}")
        with tarfile.open(path) as archive:
            for member in archive:
                key = member.name.removeprefix("./")
                if member.isfile() and key in keys and key in labels[shard]:
                    found[key] = (labels[shard][key], ("tar", str(path), member.offset_data, member.size))
    return found


def cv_sources(data: Path, paths: set[str]) -> dict[str, tuple[str, Locator]]:
    sentences: dict[str, str] | None = None
    clips: dict[str, bytes] = {}
    with tarfile.open(_need(data, "cv"), "r|gz", bufsize=_TAR_BUFFER) as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if name.name == "validated.tsv" or (name.parent.name == "clips" and name.name in paths):
                source = archive.extractfile(member)
                assert source is not None
                if name.name == "validated.tsv":
                    sentences = {row["path"]: row["sentence"] for row in _tsv_rows(source.read().decode("utf-8"))}
                else:
                    clips[name.name] = source.read()
            if sentences is not None and len(clips) == len(paths):
                break
    sentences = sentences or {}
    return {path: (sentences[path], ("bytes", clips[path])) for path in paths if path in clips and path in sentences}


def galgame_sources(data: Path, ids: set[str]) -> tuple[dict[str, tuple[str, Locator]], int]:
    """The galgame lines of the plan, and the archives that failed to extract or read."""
    seven = shutil.which("7z")
    if seven is None:
        raise PlanError("7z is not on PATH")
    password = base64.b64decode(OOPPEENN_PASSWORD_B64).decode()
    archives = sorted({ident.split("|", 1)[0] for ident in ids})
    for archive in archives:
        if PurePosixPath(archive).parts[:1] != ("GalGame",) or ".." in PurePosixPath(archive).parts:
            raise PlanError("a galgame id names an archive outside GalGame/")
        if not (data / archive).is_file():
            raise PlanError("a galgame archive of the plan is not in --data")
    folders = [data / "galgame" / PurePosixPath(archive).stem for archive in archives]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        codes = list(pool.map(lambda pair: _extracted(seven, password, data / pair[0], pair[1]), zip(archives, folders)))
    found: dict[str, tuple[str, Locator]] = {}
    failed = 0
    for archive, folder, code in zip(archives, folders, codes):
        if code not in (0, 1):  # 7-Zip: 1 is a warning, 2 and up an error
            failed += 1
            continue
        try:
            lines, _, _ = galgame_lines(folder)
        except Exception:  # counted; the plan's lines of it then count as not found
            failed += 1
            continue
        for voice, text, audio in lines:
            ident = f"{archive}|{voice}"
            if ident in ids:
                found[ident] = (text, ("file", str(audio)))
    return found, failed


def _extracted(seven: str, password: str, archive: Path, folder: Path) -> int:
    """The 7z exit code of ``archive`` extracted into ``folder``: from the
    ``<folder>.extracted`` record of an earlier finished run, else extracted
    now (``stage._extract``) and recorded when 7z succeeded (exit 0 or 1)."""
    marker = folder.parent / (folder.name + ".extracted")
    if marker.is_file() and folder.is_dir():
        return int(marker.read_text(encoding="utf-8"))
    marker.unlink(missing_ok=True)
    code = _extract(seven, password, archive, folder)
    if code in (0, 1):
        marker.write_text(str(code), encoding="utf-8")
    return code


def jvnv_sources(data: Path, ids: set[str]) -> dict[str, tuple[str, Locator]]:
    path = _need(data, "jvnv")
    with zipfile.ZipFile(path) as archive:
        script = jvnv_script(archive)
        found: dict[str, tuple[str, Locator]] = {}
        for name in filter(JVNV_REGULAR.fullmatch, archive.namelist()):
            utterance = PurePosixPath(name).stem
            line = script.get(utterance.split("_", 1)[1])
            if utterance in ids and line is not None:
                found[utterance] = (line[1], ("zip", str(path), name))  # (phrase, text)
    return found


def jnv_sources(data: Path, files: set[str]) -> dict[str, tuple[str, Locator]]:
    path = _need(data, "jnv")
    with zipfile.ZipFile(path) as archive:
        phrases = jnv_phrases(archive)
        names = set(archive.namelist())
    found: dict[str, tuple[str, Locator]] = {}
    for name in files:
        match = JNV_REGULAR.fullmatch(name)
        phrase = phrases.get(match["key"]) if match else None
        if name in names and phrase is not None:
            found[name] = (phrase, ("zip", str(path), name))
    return found


def musan_sources(data: Path, paths: set[str]) -> dict[str, tuple[str, Locator]]:
    """The MUSAN files of the plan, written below ``<data>\\musan-files\\``
    from the fetched archive in one pass (no text)."""
    written = musan_extract(_need(data, NONSPEECH_SOURCE), paths, data / "musan-files")
    return {path: ("", ("file", str(file))) for path, file in written.items()}


def gather(rows: Sequence[dict[str, Any]], data: Path) -> tuple[dict[tuple[str, str], tuple[str, Locator]], int]:
    """(src, id) -> (source text, audio locator) for every utterance of the
    plan that the data holds; and the galgame archives that failed."""
    sources = (*TEXT_SOURCES, NONSPEECH_SOURCE)
    wanted: dict[str, set[str]] = {source: set() for source in sources}
    for row in rows:
        for utt in row["utts"]:
            wanted[utt["src"]].add(utt["id"])
    readers = {"reazon": reazon_sources, "cv": cv_sources, "jvnv": jvnv_sources, "jnv": jnv_sources,
               NONSPEECH_SOURCE: musan_sources}
    found: dict[tuple[str, str], tuple[str, Locator]] = {}
    failed = 0
    for source in sources:
        if not wanted[source]:
            continue
        started = time.monotonic()
        if source == "galgame":
            got, failed = galgame_sources(data, wanted[source])
        else:
            got = readers[source](data, wanted[source])
        found.update(((source, ident), value) for ident, value in got.items())
        print(f"{source}: {len(got):,} of {len(wanted[source]):,} utterances found"
              + (f", {failed} archives failed" if source == "galgame" and failed else "")
              + f" ({time.monotonic() - started:.0f} s)", flush=True)
    return found, failed


def check(rows: Sequence[dict[str, Any]], found: dict[tuple[str, str], tuple[str, Locator]],
          ) -> tuple[dict[tuple[str, str], str], Counter]:
    """The cleaned text of every utterance found, and the disagreements with the plan."""
    texts: dict[tuple[str, str], str] = {}
    problems: Counter = Counter()
    for row in rows:
        if row["shape"] == "N":  # no text; _valid_nonspeech fixed its flags
            problems["not_found"] += (row["utts"][0]["src"], row["utts"][0]["id"]) not in found
            continue
        marks = []
        for utt in row["utts"]:
            key = (utt["src"], utt["id"])
            if key not in texts:
                if key not in found:
                    problems["not_found"] += 1
                    break
                text = clean_text(utt["src"], found[key][0])
                if text is None:
                    problems["clean_text_drops"] += 1
                    break
                texts[key] = text
            marks.append(line_mark(utt["src"], texts[key]))
        else:
            positive, anchor = window_flags(row["source"], marks, cut=cut_marked(row))
            problems["positive_differs"] += positive != row["positive"]
            problems["anchor_differs"] += anchor != row["anchor"]
    return texts, +problems


# --- cuts (round 3) ---------------------------------------------------------------

Word = tuple[float, float, str]  # (start_s, end_s, word) in the clip


@dataclass(frozen=True)
class Cut:
    """A resolved cut of one utterance: where its audio starts and ends (None
    on an uncut side), the span of its kept teacher words, and its label."""

    start_s: float | None
    end_s: float | None
    first_word_s: float
    last_word_s: float
    text: str


def _norm_chars(text: str) -> tuple[str, list[int]]:
    """``text`` normalized character by character (``pilot.norm``), and the
    ``text`` index each normalized character comes from."""
    chars: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        for piece in norm(char):
            chars.append(piece)
            origin.append(index)
    return "".join(chars), origin


def _caption_point(matched: np.ndarray, positions: np.ndarray, boundary: int) -> int | None:
    """The normalized caption index aligned with the teacher's ``boundary``
    (normalized teacher index), or None unless the ``CUT_MATCH_SPAN``
    characters on each side are exact one-to-one matches."""
    point = int(np.searchsorted(positions, boundary))
    span = range(point - CUT_MATCH_SPAN, point + CUT_MATCH_SPAN)
    if span.start < 0 or span.stop > len(matched):
        return None
    if all(matched[k] and positions[k] == boundary + k - point for k in span):
        return point
    return None


def resolve_cut(text: str, words: Sequence[Word], duration_s: float,
                start: float | None, end: float | None) -> Cut | str:
    """The cut of a cleaned ``text`` whose clip (``duration_s``) the teacher
    heard as ``words``, for a head mark ``start`` and/or a tail mark ``end``
    (fractions of the clip), or why it is refused: ``no_words``, ``no_gap`` or
    ``mismatch`` (see the module docstring)."""
    if not words:
        return "no_words"
    pieces = [_norm_chars(word)[0] for _, _, word in words]
    bounds = [sum(map(len, pieces[:index])) for index in range(len(words) + 1)]
    gaps = {index: (words[index - 1][1] + words[index][0]) / 2 for index in range(1, len(words))}
    caption, origin = _norm_chars(text)
    matched, positions, _ = alignment_trace(caption, "".join(pieces))

    def choose(target: float, keep: Callable[[int], bool]) -> tuple[int, int] | str:
        """(gap, caption point) of the gap nearest ``target`` that ``keep``
        allows and where the caption matches the teacher. A gap beside a
        word that normalizes to nothing (``%``, ``〜``) is never taken: its
        two sides map to the same caption point, so the word's audio and
        its caption text would fall on different sides of the cut."""
        allowed = sorted((index for index in gaps if keep(index) and pieces[index - 1] and pieces[index]),
                         key=lambda index: (abs(gaps[index] - target), index))
        for index in allowed:
            point = _caption_point(matched, positions, bounds[index])
            if point is not None:
                return index, point
        return "mismatch" if allowed else "no_gap"

    head = tail = None
    begin, finish = 0, len(text)
    if start is not None:
        chosen = choose(start * duration_s, lambda i: len(words) - i >= CUT_MIN_WORDS
                        and words[-1][1] - gaps[i] >= CUT_MIN_KEPT_S)
        if isinstance(chosen, str):
            return chosen
        head, point = chosen
        begin = origin[point]
    if end is not None:
        first, opening = (head, gaps[head]) if head is not None else (0, words[0][0])
        chosen = choose(end * duration_s, lambda j: j - first >= CUT_MIN_WORDS and gaps[j] - opening >= CUT_MIN_KEPT_S)
        if isinstance(chosen, str):
            return chosen
        tail, point = chosen
        finish = origin[point - 1] + 1
    return Cut(None if head is None else round(gaps[head], 3), None if tail is None else round(gaps[tail], 3),
               words[head or 0][0], words[(tail or len(words)) - 1][1], text[begin:finish].strip())


def read_words(path: Path) -> dict[tuple[str, str], tuple[float, list[Word]]]:
    """``teacher --plan``'s words file: (src, id) -> (clip seconds, words)."""
    words: dict[tuple[str, str], tuple[float, list[Word]]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    words[(row["src"], row["id"])] = (float(row["duration_s"]),
                                                      [(float(s), float(e), str(w)) for s, e, w in row["words"]])
    except FileNotFoundError:
        raise PlanError(f"{path} is missing") from None
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        raise PlanError(f"cannot read the words file: {type(error).__name__}") from None
    return words


def resolve_cuts(rows: Sequence[dict[str, Any]], texts: dict[tuple[str, str], str],
                 words: dict[tuple[str, str], tuple[float, list[Word]]],
                 ) -> tuple[dict[tuple[int, int], Cut], Counter]:
    """Every cut mark of the plan resolved: (w, utterance index) -> cut; and
    the counts (``planned_head``, ``planned_tail``, ``applied_head``,
    ``applied_tail``, ``refused_<reason>`` per marked utterance)."""
    cuts: dict[tuple[int, int], Cut] = {}
    counts: Counter = Counter()
    for row in rows:
        for index, utt in enumerate(row["utts"]):
            start, end = utt.get("cut_start"), utt.get("cut_end")
            if start is None and end is None:
                continue
            counts.update(planned_head=start is not None, planned_tail=end is not None)
            key = (utt["src"], utt["id"])
            duration, heard = words.get(key, (0.0, []))
            cut = resolve_cut(texts[key], heard, duration, start, end)
            if isinstance(cut, str):
                counts[f"refused_{cut}"] += 1
                continue
            cuts[(row["w"], index)] = cut
            counts.update(applied_head=start is not None, applied_tail=end is not None)
    return cuts, counts


def cut_window(row: dict[str, Any], texts: dict[tuple[str, str], str],
               cuts: dict[tuple[int, int], Cut]) -> tuple[list[str], bool, bool]:
    """A window's label texts with its cuts applied, and its positive and
    anchor flags: recomputed from those texts for a window with a cut mark
    (never an anchor), the plan's otherwise."""
    lines = [cuts[(row["w"], index)].text if (row["w"], index) in cuts else texts[(utt["src"], utt["id"])]
             for index, utt in enumerate(row["utts"])]
    if not cut_marked(row):
        return lines, row["positive"], row["anchor"]
    positive, anchor = window_flags(row["source"], [line_mark(utt["src"], line)
                                                    for utt, line in zip(row["utts"], lines)], cut=True)
    return lines, positive, anchor


# --- audio (worker processes) ------------------------------------------------------------

_VAD: Any = None
_ZIPS: dict[str, zipfile.ZipFile] = {}


def _init(model_path: str) -> None:
    global _VAD
    from yamabiko.vad.silero_chunker import SileroVAD

    _VAD = SileroVAD(Path(model_path))


def locator_bytes(locator: Locator) -> bytes:
    """The encoded audio file a locator names."""
    kind = locator[0]
    if kind == "tar":
        with open(locator[1], "rb") as handle:
            handle.seek(locator[2])
            return handle.read(locator[3])
    if kind == "zip":
        archive = _ZIPS.get(locator[1])
        if archive is None:
            archive = _ZIPS[locator[1]] = zipfile.ZipFile(locator[1])
        return archive.read(locator[2])
    if kind == "bytes":
        return locator[1]
    return Path(locator[1]).read_bytes()


def _audio(locator: Locator) -> np.ndarray:
    from faster_whisper import decode_audio

    return decode_audio(io.BytesIO(locator_bytes(locator)), sampling_rate=SAMPLE_RATE)


def trim(vad: Any, audio: np.ndarray) -> tuple[int, int, np.ndarray]:
    """(start, end) of ``audio`` kept around its Silero speech frames, and
    the per-sample speech mask (all False when no frame is speech)."""
    frames = -(-len(audio) // FRAME)
    padded = np.zeros(frames * FRAME, np.float32)
    padded[:len(audio)] = audio
    vad.reset_states()
    speech = np.fromiter((vad(padded[index * FRAME:(index + 1) * FRAME]) >= THRESHOLD for index in range(frames)),
                         dtype=bool, count=frames)
    mask = np.repeat(speech, FRAME)[:len(audio)]
    hits = np.flatnonzero(speech)
    if hits.size == 0:
        return 0, len(audio), mask
    return max(0, int(hits[0]) * FRAME - MARGIN), min(len(audio), (int(hits[-1]) + 1) * FRAME + MARGIN), mask


def _nonspeech_clip(audio: np.ndarray, span: tuple[float, float], level: float, gain: float,
                    stats: Counter) -> np.ndarray:
    """A no-speech window's span of its file at ``level + gain`` dBFS RMS
    (peak at most ``PEAK``; a digitally silent span stays silent)."""
    start = min(round(span[0] * SAMPLE_RATE), len(audio))
    clip = audio[start:start + round(span[1] * SAMPLE_RATE)].astype(np.float64)
    rms = math.sqrt(float(np.mean(np.square(clip)))) if len(clip) else 0.0
    if rms == 0:
        stats["nonspeech_silent"] += 1
        return clip
    factor = 10 ** ((level + gain - 20 * math.log10(rms)) / 20)
    peak = float(np.max(np.abs(clip))) * factor
    if peak > PEAK:
        factor *= PEAK / peak
        stats["peak_limited"] += 1
    return clip * factor


def make_window(task: tuple[int, float, float, list[tuple[Locator, float, float, Cut | None]],
                            tuple[float, float] | None, bool]) -> tuple[int, Any, Any, Any]:
    """(w, int16 PCM bytes, utterance bounds in samples, stats), or
    (w, None, None, reason) for a window left out. ``span`` (start, seconds)
    marks a no-speech window; ``drop_no_speech`` leaves out a speech window
    with an utterance that has no Silero speech frame (``no_speech``)."""
    w, cap_s, level, utts, span, drop_no_speech = task
    pieces: list[tuple[int, np.ndarray]] = []
    stats: Counter = Counter()
    try:  # the reason names the type only: a message may hold a path
        for locator, gap, gain, cut in utts:
            try:
                audio = _audio(locator)
            except Exception as error:
                return w, None, None, f"decode_{type(error).__name__}"
            if not len(audio):
                return w, None, None, "empty_audio"
            if span is not None:
                clip = _nonspeech_clip(audio, span, level, gain, stats)
                if not len(clip):
                    return w, None, None, "empty_audio"
                pieces.append((0, clip))
                continue
            start, end, mask = trim(_VAD, audio)
            if cut is not None:  # the cut side at the cut, the other never inside the kept words
                start = (round(cut.start_s * SAMPLE_RATE) if cut.start_s is not None
                         else min(start, max(0, round(cut.first_word_s * SAMPLE_RATE) - MARGIN)))
                end = (min(len(audio), round(cut.end_s * SAMPLE_RATE)) if cut.end_s is not None
                       else max(end, min(len(audio), round(cut.last_word_s * SAMPLE_RATE) + MARGIN)))
                if end <= start:
                    return w, None, None, "cut_empty"
            clip, kept = audio[start:end].astype(np.float64), mask[start:end]
            speech = clip[kept] if kept.any() else clip
            rms = math.sqrt(float(np.mean(np.square(speech))))
            if rms == 0:  # never train a text over digital silence
                return w, None, None, "silent"
            if drop_no_speech and not kept.any():
                return w, None, None, "no_speech"
            stats["no_speech"] += not kept.any()
            stats["speech_samples"] += int(kept.sum())
            factor = 10 ** ((level + gain - 20 * math.log10(rms)) / 20)
            peak = float(np.max(np.abs(clip))) * factor
            if peak > PEAK:
                factor *= PEAK / peak
                stats["peak_limited"] += 1
            pieces.append((round(gap * SAMPLE_RATE), clip * factor))
    except Exception as error:
        return w, None, None, type(error).__name__
    content = sum(gap + len(clip) for gap, clip in pieces)
    if content > round(cap_s * SAMPLE_RATE):
        return w, None, None, "over_cap"
    window = np.zeros(2 * PAD + content, np.float64)
    position, bounds = PAD, []
    for gap, clip in pieces:
        position += gap
        window[position:position + len(clip)] = clip
        bounds.append((position, position + len(clip)))
        position += len(clip)
    pcm = np.clip(np.rint(window * 32768), -32768, 32767).astype("<i2")
    return w, pcm.tobytes(), bounds, dict(stats)


def _ordered(tasks: Iterable[Any], workers: int, model_path: str) -> Iterator[tuple[int, Any, Any, Any]]:
    """``make_window`` over ``tasks`` in order, at most a few per worker in flight."""
    if workers == 1:
        _init(model_path)
        yield from map(make_window, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_init, initargs=(model_path,)) as pool:
        iterator = iter(tasks)
        pending = deque(pool.submit(make_window, task) for task in itertools.islice(iterator, 4 * workers))
        while pending:
            result = pending.popleft().result()
            following = next(iterator, None)
            if following is not None:
                pending.append(pool.submit(make_window, following))
            yield result


# --- main ---------------------------------------------------------------------------


def _seconds(token: int) -> float:
    return round((token - TS0) * TS_STEP, 2)


def _hours(samples: int) -> float:
    return round(samples / SAMPLE_RATE / 3600, 4)


def materialize(rows: Sequence[dict[str, Any]], found: dict[tuple[str, str], tuple[str, Locator]],
                texts: dict[tuple[str, str], str], out: Path, workers: int,
                cuts: dict[tuple[int, int], Cut] | None = None, drop_no_speech: bool = False) -> dict[str, Any]:
    from yamabiko.vad.silero_chunker import _default_model_path

    cuts = cuts or {}
    tokenizer = load_tokenizer()
    model_path = str(_default_model_path())
    tasks = ((row["w"], SHAPES[row["shape"]].cap_s, float(row["level_dbfs"]),
              [(found[(utt["src"], utt["id"])][1], float(utt["gap_before_s"]), float(utt["gain_db"]),
                cuts.get((row["w"], index)))
               for index, utt in enumerate(row["utts"])],
              (float(row["utts"][0]["start_s"]), float(row["utts"][0]["duration_s"])) if row["shape"] == "N" else None,
              drop_no_speech)
             for row in rows)
    shapes: dict[str, Counter] = {shape: Counter() for shape in SHAPES}
    sources: dict[str, Counter] = {}
    stats: Counter = Counter()
    splits: Counter = Counter()
    tokens: list[int] = []
    left_out: list[dict[str, Any]] = []
    flips: Counter = Counter()
    started = time.monotonic()
    offset = 0
    out.mkdir(parents=True, exist_ok=True)
    with (out / "audio.i16").open("wb") as audio, (out / "index.jsonl").open("w", encoding="utf-8", newline="\n") as index:
        for done, (w, pcm, bounds, info) in enumerate(_ordered(tasks, workers, model_path), 1):
            row = rows[w]
            nonspeech = row["shape"] == "N"
            lines, positive, anchor = ([], False, False) if nonspeech else cut_window(row, texts, cuts)
            segments: list[tuple[float, float, str]] = []
            if pcm is not None and nonspeech:
                labels = list(NOSPEECH_LABELS)
            elif pcm is not None:
                segments = [(start / SAMPLE_RATE, end / SAMPLE_RATE, line) for (start, end), line in zip(bounds, lines)]
                try:
                    labels = build_labels(tokenizer, segments)
                except ValueError:
                    pcm, info = None, "labels"
            if pcm is None:
                left_out.append({"w": w, "source": row["source"], "split": row["split"], "reason": info})
            else:
                samples = len(pcm) // 2
                applied = [side for side, at, attr in (("head", 0, "start_s"), ("tail", len(row["utts"]) - 1, "end_s"))
                           if getattr(cuts.get((w, at)), attr, None) is not None]
                index.write(json.dumps({
                    "w": w, "offset": offset, "samples": samples, "labels": labels, "anchor": anchor,
                    "positive": positive, "split": row["split"],
                    "segments": [[_seconds(timestamp_token(start)), _seconds(timestamp_token(end)), utt["src"]]
                                 for (start, end, _), utt in zip(segments, row["utts"])],
                    **({"cut": applied} if applied else {}),
                }, ensure_ascii=False) + "\n")
                flips.update(cut_windows=bool(applied), positive_lost=row["positive"] and not positive,
                             positive_gained=positive and not row["positive"])
                audio.write(pcm)
                offset += samples
                tokens.append(len(labels))
                stats.update(info)
                if not nonspeech:
                    stats["utterances"] += len(bounds)
                    stats["trimmed_samples"] += sum(end - start for start, end in bounds)
                shape = shapes[row["shape"]]
                shape.update(windows=1, audio_samples=samples, content_samples=samples - 2 * PAD)
                source = sources.setdefault(row["source"], Counter())
                source.update(windows=1, positive=positive, anchor=anchor, dev=row["split"] == "dev",
                              jnv_splices=any(utt["src"] == "jnv" for utt in row["utts"]), cut=bool(applied))
                splits[row["split"]] += 1
            if done % _PROGRESS == 0:
                print(f"windows: {done:,} of {len(rows):,} ({time.monotonic() - started:.0f} s)", flush=True)
    written = sum(shape["windows"] for shape in shapes.values())
    reasons = Counter(entry["reason"] for entry in left_out)
    by_source: dict[str, Counter] = {}
    for entry in left_out:
        by_source.setdefault(entry["source"], Counter())[entry["reason"]] += 1
    percentiles = (dict(zip(("p50", "p90", "p99", "max"),
                            (int(value) for value in np.percentile(tokens, (50, 90, 99, 100), method="lower"))))
                   if tokens else {})
    return {
        "schema": OUT_SCHEMA,
        "plan_windows": len(rows),
        "windows": written,
        "split": dict(splits),
        "positive": sum(source["positive"] for source in sources.values()),
        "anchor": sum(source["anchor"] for source in sources.values()),
        "shapes": {name: {"windows": shape["windows"], "audio_h": _hours(shape["audio_samples"]),
                          "content_h": _hours(shape["content_samples"])} for name, shape in shapes.items()},
        "sources": {name: dict(source) for name, source in sorted(sources.items())},
        "hours": {"audio": _hours(offset), "content": _hours(offset - 2 * PAD * written),
                  "utterances_trimmed": _hours(stats["trimmed_samples"]),
                  "speech_frames": _hours(stats["speech_samples"])},
        "utterances": stats["utterances"],
        "no_speech": stats["no_speech"], "peak_limited": stats["peak_limited"],
        "nonspeech_silent": stats["nonspeech_silent"], "drop_no_speech": drop_no_speech,
        "label_tokens": percentiles,
        "left_out": {"count": len(left_out), "reasons": dict(reasons),
                     "by_source": {name: dict(counts) for name, counts in sorted(by_source.items())},
                     # no_speech (--drop-no-speech) is chosen, not a failure: it counts toward no limit
                     "over_limit": over_limit(rows, [entry for entry in left_out if entry["reason"] != "no_speech"]),
                     "windows": left_out},
        "cuts_written": dict(flips),
        "audio_i16_bytes": 2 * offset,
        "workers": workers,
        "audio_wall_s": round(time.monotonic() - started, 1),
    }


def over_limit(rows: Sequence[dict[str, Any]], left_out: Sequence[dict[str, Any]]) -> list[str]:
    """The groups (all windows, each source, each split) with more than
    ``LEFT_OUT_LIMIT`` of their plan windows left out."""
    planned: Counter = Counter()
    missed: Counter = Counter()
    for windows, entries in ((planned, rows), (missed, left_out)):
        for entry in entries:
            windows.update(("all", f"source {entry['source']}", f"split {entry['split']}"))
    return [group for group in planned if missed[group] > LEFT_OUT_LIMIT * planned[group]]


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.windows",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True, help="windows-plan.jsonl.gz (scripts.asr_finetune.mix)")
    parser.add_argument("--data", type=Path, required=True, help="the fetched sources (scripts.asr_finetune.fetch)")
    parser.add_argument("--out", type=Path, required=True, help="windows.json (and audio.i16, index.jsonl without --audio)")
    parser.add_argument("--audio", type=Path, help="audio.i16 and index.jsonl (default: --out)")
    parser.add_argument("--words", type=Path, help="the teacher's words of the cut utterances (teacher --plan); "
                                                   "needed when the plan has cut marks")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes (default: CPU count)")
    parser.add_argument("--drop-no-speech", action="store_true",
                        help="leave out every speech window with an utterance that has no Silero speech frame "
                             "(reason no_speech, counted, under no limit; round 4, v4b)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    started = time.monotonic()
    audio_dir = args.audio or args.out
    for folder in {args.out, audio_dir}:  # a failed run leaves none; a cut audit's earlier original goes too
        for name in ("audio.i16", "index.jsonl", "windows.json", "index.unfiltered.jsonl"):
            (folder / name).unlink(missing_ok=True)
    try:
        rows = read_plan(args.plan)
        print(f"plan: {len(rows):,} windows, {sum(len(row['utts']) for row in rows):,} utterances", flush=True)
        marked = any(map(cut_marked, rows))
        if marked and args.words is None:
            raise PlanError("the plan has cut marks: --words is needed")
        words, words_file = {}, None
        if marked and args.words.is_file():
            words, words_file = read_words(args.words), "read"
        elif marked:
            words_file = "missing"
            print("warning: the words file is missing: every cut mark is refused (no_words)", flush=True)
        found, failed_archives = gather(rows, args.data)
    except (PlanError, StageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"error: a source file lacks {error.args[0]!r}", file=sys.stderr)
        return 2
    except (OSError, tarfile.TarError, zipfile.BadZipFile, UnicodeDecodeError) as error:  # the type: a message holds a path
        print(f"error: cannot read a source file ({type(error).__name__})", file=sys.stderr)
        return 2
    texts, problems = check(rows, found)
    if problems:
        args.out.mkdir(parents=True, exist_ok=True)
        record = {"schema": OUT_SCHEMA, "plan_windows": len(rows), "plan_disagreements": dict(problems),
                  "galgame_archives_failed": failed_archives}
        (args.out / "windows.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print("error: the data disagrees with the plan: "
              + ", ".join(f"{name} {count:,}" for name, count in problems.items()) + "; no audio written",
              file=sys.stderr)
        return 1
    print(f"check: every utterance found, positive and anchor agree with the plan for {len(rows):,} windows",
          flush=True)
    cuts, cut_counts = resolve_cuts(rows, texts, words)
    if marked:
        print("cuts: " + ", ".join(f"{name} {count:,}" for name, count in sorted(cut_counts.items())), flush=True)
    record = materialize(rows, found, texts, audio_dir, args.workers, cuts, args.drop_no_speech)
    record["cuts"] = dict(sorted(cut_counts.items()))
    record["words_file"] = words_file
    args.out.mkdir(parents=True, exist_ok=True)
    if audio_dir.resolve() != args.out.resolve():
        shutil.copy2(audio_dir / "index.jsonl", args.out / "index.jsonl")
    record["galgame_archives_failed"] = failed_archives
    record["wall_s"] = round(time.monotonic() - started, 1)
    (args.out / "windows.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    shapes = record["shapes"]
    print(f"windows: {record['windows']:,} of {len(rows):,} written ({record['split']}); "
          + "; ".join(f"{name} {shape['windows']:,} ({shape['audio_h']:,.2f} h audio, {shape['content_h']:,.2f} h content)"
                      for name, shape in shapes.items())
          + f"; positive {record['positive']:,}, anchor {record['anchor']:,}")
    print(f"utterances {record['utterances']:,}: no_speech {record['no_speech']:,}, peak_limited "
          f"{record['peak_limited']:,}; label tokens {record['label_tokens']}; "
          f"audio.i16 {record['audio_i16_bytes']:,} bytes; {record['audio_wall_s']:.0f} s of audio work, "
          f"{record['wall_s']:.0f} s in all")
    left = record["left_out"]
    print(f"left out {left['count']:,} windows" + "".join(
        f"; {source}: " + ", ".join(f"{reason} {count:,}" for reason, count in sorted(counts.items()))
        for source, counts in left["by_source"].items()))
    if left["over_limit"]:
        print(f"error: more than {LEFT_OUT_LIMIT:.0%} of the windows left out in: {', '.join(left['over_limit'])}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
