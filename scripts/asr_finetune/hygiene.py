"""Hygiene lists of the ASR fine-tuning pilot (data survey section 4).

    python -m scripts.asr_finetune.hygiene build [--cv17-check] [--labels JSONL [--field NAME]]

Private outputs in ``<pilot>\\hygiene\\`` (never uploaded):

- ``bench-lines.json``: ``{"lines": [...]}``, the normalized text of every
  reference line of every set under ``<bench root>\\sets\\``;
- ``titles.json``: ``{"titles": [...]}``, every item's ``condition.title`` and
  every locked title and alias (``<bench root>\\config\\locked-titles.private.json``),
  normalized;
- ``cv-exclusion.json``: the Common Voice rules in plain normalized text and
  clip names: ``{"sentences", "client_id_prefixes", "clips", "split_clips",
  "cv8_test_sentences"}``.

Common Voice rules (survey 4.1), from the sets that use Common Voice (a
``cv17`` line or a ``cv17:`` voice) and ``tests/fixtures/audio/cv17_ja_8lines.json``:

1. sentence: every line of those sets (Common Voice or VOICEVOX);
2. speaker: the 12-character ``client_id`` prefixes of their ``cv17:`` voices;
3. clip: the fixture's clips, plus the CV 17.0 test rows that carry a
   ``cv17`` line of a set and one of that set's voices (the sets record
   voices, not clip names);
4. splits: the clips of ``dev.tsv`` and ``test.tsv`` of CV 17.0 (the bench's
   pin) and of the 19.0, 21.0 and 22.0 mirrors (TSVs only), and the CV 8.0
   test sentences (the 760M's test), read through the Hugging Face dataset
   viewer's ``/rows`` JSON (``transcription`` only; audio is never fetched).
   The viewer serves the repo's main branch: main must equal the pinned
   revision before and after the pages are read, and the viewer must report
   the split's known row count.

The one pod-bound output, ``<pilot>\\outbox\\cv-exclude.json``, holds all four
rules for the S1 stage: the SHA-256 of every normalized rule-1 and CV 8 test
sentence (``sentence_hash``), the prefixes, and the bench and split clip names.
It must pass ``check_outbox_file``; a refused file is deleted (exit 1).

``--cv17-check`` counts the clips rules 1-3 remove from each CV 17.0 ja TSV
(survey 4.1 table). ``--labels`` counts the texts of a JSON Lines file that
``BenchLineFilter`` hits (survey 4.2). Output is counts only, never a line,
sentence or title.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.asr_bench.build_synthetic import CV_REPO, CV_REVISION
from scripts.asr_bench.sets import BenchSet, bench_root, load_set
from scripts.asr_finetune.outbox import OutboxRefusal, check_outbox_file, hygiene_dir, outbox_dir
from scripts.asr_finetune.pilot import (
    CV8_TEST,
    CV_SPLIT_MIRRORS,
    NORMALIZATION,
    BenchLineFilter,
    norm,
    sentence_hash,
)

OUTBOX_SCHEMA = "ft-pilot-cv-exclude-1"
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "audio" / "cv17_ja_8lines.json"
CV_SOURCE = "cv17"
CV_VOICE = "cv17:"
PREFIX_CHARS = 12
SPLITS = ("dev", "test")
CV17_CHECK_TSVS = ("train", "dev", "test", "validated", "other")
VIEWER_ROWS = "https://datasets-server.huggingface.co/rows"
VIEWER_PAGE = 100
# The viewer answers 429 (rate limit, no Retry-After) or 5xx while it reads a
# page from its parquet copy: pause between pages, back off on those answers.
VIEWER_PAUSE_S = 1.0
VIEWER_BACKOFF_S = (5, 10, 20, 40, 80, 160)


@dataclass(frozen=True)
class CvRules:
    """Raw texts (rule 1 and CV 8 test) and names (rules 2-4)."""

    sentences: frozenset[str]
    prefixes: frozenset[str]
    clips: frozenset[str]
    split_clips: frozenset[str]
    cv8_sentences: frozenset[str]


def bench_sets() -> list[BenchSet]:
    """Every set under ``<bench root>\\sets\\``; a directory without a valid
    ``set.json`` stops the build (the lists must miss no set)."""
    sets = []
    for directory in sorted(path for path in (bench_root() / "sets").iterdir() if path.is_dir()):
        if not (directory / "set.json").is_file():
            raise SystemExit(f"{directory} holds no set.json; every set directory must hold one")
        sets.append(load_set(directory))
    if not sets:
        raise SystemExit(f"no bench set under {bench_root() / 'sets'}")
    return sets


def _normalized(texts: Iterable[str]) -> list[str]:
    return sorted({normalized for normalized in map(norm, texts) if normalized})


def _locked_titles() -> list[str]:
    path = bench_root() / "config" / "locked-titles.private.json"
    if not path.is_file():
        raise SystemExit(f"the locked-title list {path} is missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("titles") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise SystemExit(f"{path} holds no 'titles' list of objects")
    values = [value for entry in entries for value in [entry.get("key"), *(entry.get("aliases") or [])]]
    if not all(isinstance(value, str) for value in values):
        raise SystemExit(f"{path} holds a key or alias that is not a string")
    return values


def _tsv(repo: str, revision: str, split: str) -> list[dict[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=f"transcript/ja/{split}.tsv", repo_type="dataset",
                           revision=revision, cache_dir=str(bench_root() / "cache" / "hf"))
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))


def _require_main(repo: str, revision: str) -> None:
    from huggingface_hub import HfApi

    main = HfApi().dataset_info(repo, revision="main").sha
    if main != revision:
        raise SystemExit(f"{repo}: main is at {main}, not the pinned {revision}; "
                         "the dataset viewer serves main, so its rows are not the pinned ones")


def _viewer_page(query: str) -> dict[str, Any]:
    """One /rows page, retried after a pause on HTTP 429, 5xx or a network error."""
    problem = ""
    for pause in (0, *VIEWER_BACKOFF_S):
        if pause:
            print(f"dataset viewer: {problem}; retrying in {pause} s", file=sys.stderr)
            time.sleep(pause)
        try:
            with urllib.request.urlopen(f"{VIEWER_ROWS}?{query}", timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise SystemExit(f"the dataset viewer refused {query}: HTTP {error.code}") from None
            problem = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            problem = type(error).__name__
    raise SystemExit(f"the dataset viewer failed on {query} ({problem}) after {len(VIEWER_BACKOFF_S) + 1} attempts")


def cv8_test_sentences() -> list[str]:
    """The ``transcription`` of every CV 8.0 ja test row, in row order."""
    repo, revision, rows = CV8_TEST
    _require_main(repo, revision)
    texts: dict[int, str] = {}
    for offset in range(0, rows, VIEWER_PAGE):
        if offset:
            time.sleep(VIEWER_PAUSE_S)
        page = _viewer_page(urllib.parse.urlencode({"dataset": repo, "config": "default", "split": "test",
                                                    "offset": offset, "length": VIEWER_PAGE}))
        if page.get("num_rows_total") != rows or page.get("partial") is not False:
            raise SystemExit(f"{repo}: the dataset viewer reports {page.get('num_rows_total')} rows "
                             f"(partial: {page.get('partial')}), not the test split's {rows}")
        for row in page["rows"]:
            text = row["row"].get("transcription")
            if "transcription" in row.get("truncated_cells", ()) or not isinstance(text, str):
                raise SystemExit(f"{repo}: row {row['row_idx']} has no complete transcription")
            texts[row["row_idx"]] = text
    if sorted(texts) != list(range(rows)):
        raise SystemExit(f"{repo}: the dataset viewer returned {len(texts)} distinct rows, not {rows}")
    _require_main(repo, revision)
    return [texts[index] for index in range(rows)]


def _voices(bench_set: BenchSet) -> set[str]:
    """The client_id prefixes of the set's ``cv17:`` voices."""
    return {str(voice)[len(CV_VOICE):] for item in bench_set.items
            for voice in item.condition.get("voices", ()) if str(voice).startswith(CV_VOICE)}


def cv_rules(sets: Sequence[BenchSet], counts: dict[str, Any]) -> CvRules:
    """Rules 1-4; fills ``counts`` with what the console shows."""
    cv_sets = [bench_set for bench_set in sets
               if _voices(bench_set) or any(line.source == CV_SOURCE for item in bench_set.items for line in item.lines)]
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))["lines"]
    lines = [line for bench_set in cv_sets for item in bench_set.items for line in item.lines]
    sentences = {line.text for line in lines} | {entry["sentence"] for entry in fixture}
    prefixes = set().union(*map(_voices, cv_sets))
    if any(len(prefix) != PREFIX_CHARS for prefix in prefixes):
        raise SystemExit(f"a {CV_VOICE} voice is not a {PREFIX_CHARS}-character client_id prefix")
    test17 = _tsv(CV_REPO, CV_REVISION, "test")
    # A set that records voices, not clips: its clips are the test rows that
    # carry one of its cv17 sentences and one of its voices.
    cv_keys: set[str] = set()
    rebuilt: dict[str, str] = {}
    for bench_set in cv_sets:
        voices = _voices(bench_set)
        if not voices:
            continue
        keys = {norm(line.text) for item in bench_set.items for line in item.lines if line.source == CV_SOURCE}
        cv_keys |= keys
        rebuilt |= {row["path"]: norm(row["sentence"]) for row in test17
                    if row["client_id"][:PREFIX_CHARS] in voices and norm(row["sentence"]) in keys}
    fixture_clips = {entry["clip"] for entry in fixture}
    split_clips: set[str] = set()
    per_release = {}
    for repo, revision in ((CV_REPO, CV_REVISION), *CV_SPLIT_MIRRORS):
        sizes = {}
        for split in SPLITS:
            rows = test17 if (repo, split) == (CV_REPO, "test") else _tsv(repo, revision, split)
            paths = {row["path"] for row in rows}
            sizes[split] = len(paths)
            split_clips |= paths
        per_release[".".join(repo.rsplit("_", 2)[1:])] = sizes
    cv8 = cv8_test_sentences()
    counts.update({
        "cv_sets": [bench_set.set_id for bench_set in cv_sets],
        "sentences_raw": len(sentences), "sentences": len(_normalized(sentences)),
        "cv_sentences": len(cv_keys), "cv_sentences_rebuilt": len(set(rebuilt.values())),
        "prefixes": len(prefixes),
        "prefixes_in_test": len({row["client_id"][:PREFIX_CHARS] for row in test17} & prefixes),
        "fixture_clips": len(fixture_clips), "rebuilt_clips": len(rebuilt),
        "clips": len(fixture_clips | set(rebuilt)),
        "split_releases": per_release, "split_clips": len(split_clips),
        "cv8_rows": len(cv8), "cv8_sentences": len(_normalized(cv8)),
    })
    return CvRules(frozenset(sentences), frozenset(prefixes), frozenset(fixture_clips | set(rebuilt)),
                   frozenset(split_clips), frozenset(cv8))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cv17_check(rules: CvRules) -> None:
    """Clips rules 1-3 remove from each CV 17.0 ja TSV (survey 4.1 table)."""
    keys = {norm(text) for text in rules.sentences}
    clients: dict[str, set[str]] = {}
    print("\nCV 17.0 ja, clips removed by rule 1 (sentence) / 2 (speaker) / 3 (clip) / union:")
    print(f"{'tsv':<10} {'clips':>8} {'sentence':>9} {'speaker':>8} {'clip':>6} {'union':>7}")
    for split in CV17_CHECK_TSVS:
        rows = _tsv(CV_REPO, CV_REVISION, split)
        for row in rows:
            clients.setdefault(row["client_id"][:PREFIX_CHARS], set()).add(row["client_id"])
        by_sentence = {row["path"] for row in rows if norm(row["sentence"]) in keys}
        by_speaker = {row["path"] for row in rows if row["client_id"][:PREFIX_CHARS] in rules.prefixes}
        by_clip = {row["path"] for row in rows if row["path"] in rules.clips}
        union = by_sentence | by_speaker | by_clip
        print(f"{split:<10} {len(rows):>8,} {len(by_sentence):>9,} {len(by_speaker):>8,} "
              f"{len(by_clip):>6,} {len(union):>7,}")
    shared = sum(len(clients.get(prefix, ())) > 1 for prefix in rules.prefixes)
    print(f"rule-2 prefixes shared by more than one client_id across these TSVs: {shared}")


def labels_check(path: Path, field: str, sets: Sequence[BenchSet], lines: Sequence[str]) -> None:
    """Survey 4.2 on a JSON Lines file of training labels; hits by set."""
    texts = [json.loads(line)[field] for line in path.open(encoding="utf-8") if line.strip()]
    rule = BenchLineFilter(lines)
    hits = sum(map(rule.hit, texts))
    stock = {line for line in lines if len(line) >= 8}
    exact = sum(norm(text) in stock for text in texts)
    print(f"\nlabels ({field!r}): {len(texts):,} texts; BenchLineFilter (lines of at least "
          f"{rule.min_chars} characters: {len(rule):,}) hits {hits}; "
          f"texts equal to a bench line of at least 8 characters: {exact}")
    for bench_set in sets:
        own = BenchLineFilter(line.text for item in bench_set.items for line in item.lines)
        own_hits = sum(map(own.hit, texts))
        if own_hits:
            print(f"  hits on lines of {bench_set.set_id}: {own_hits}")


def build(args: argparse.Namespace) -> int:
    sets = bench_sets()
    print(f"{'set':<24} {'items':>6} {'lines':>7} {'distinct':>9}")
    for bench_set in sets:
        texts = [line.text for item in bench_set.items for line in item.lines]
        print(f"{bench_set.set_id:<24} {len(bench_set.items):>6,} {len(texts):>7,} {len(_normalized(texts)):>9,}")
    raw_lines = [line.text for bench_set in sets for item in bench_set.items for line in item.lines]
    lines = _normalized(raw_lines)
    set_titles = [item.condition["title"] for bench_set in sets for item in bench_set.items
                  if isinstance(item.condition.get("title"), str)]
    locked = _locked_titles()
    titles = _normalized([*set_titles, *locked])
    print(f"bench lines: {len(raw_lines):,} in {len(sets)} sets, {len(lines):,} distinct normalized "
          f"({sum(len(line) >= 12 for line in lines):,} of at least 12 characters)")
    print(f"titles: {len(_normalized(set_titles))} from set items, {len(locked)} locked keys and aliases, "
          f"{len(titles)} distinct normalized")
    # The outbox guard's lists depend on local files only: written before any download.
    hygiene = hygiene_dir()
    _write_json(hygiene / "bench-lines.json", {"lines": lines})
    _write_json(hygiene / "titles.json", {"titles": titles})

    counts: dict[str, Any] = {}
    rules = cv_rules(sets, counts)
    print(f"\nCommon Voice rules, from {', '.join(counts['cv_sets'])} and the {FIXTURE.name} fixture:")
    print(f"rule 1 sentences: {counts['sentences']} distinct normalized ({counts['sentences_raw']} distinct texts)")
    print(f"rule 2 client_id prefixes: {counts['prefixes']} ({counts['prefixes_in_test']} found in CV 17.0 test)")
    print(f"rule 3 bench clips: {counts['clips']} ({counts['fixture_clips']} fixture clips, "
          f"{counts['rebuilt_clips']} CV 17.0 test rows rebuilt for {counts['cv_sentences_rebuilt']} "
          f"of {counts['cv_sentences']} Common Voice sentences of the sets that record voices)")
    releases = ", ".join(f"{name} {sizes['dev']:,}/{sizes['test']:,}" for name, sizes in counts["split_releases"].items())
    print(f"rule 4 split clips (dev/test): {releases}; {counts['split_clips']:,} distinct")
    print(f"rule 4 CV 8.0 test: {counts['cv8_rows']:,} rows, {counts['cv8_sentences']:,} distinct normalized sentences")

    _write_json(hygiene / "cv-exclusion.json", {
        "sentences": _normalized(rules.sentences), "client_id_prefixes": sorted(rules.prefixes),
        "clips": sorted(rules.clips), "split_clips": sorted(rules.split_clips),
        "cv8_test_sentences": _normalized(rules.cv8_sentences),
    })
    hashes = {sentence_hash(text) for text in rules.sentences | rules.cv8_sentences if norm(text)}
    outbox = outbox_dir() / "cv-exclude.json"
    _write_json(outbox, {
        "schema": OUTBOX_SCHEMA, "normalization": NORMALIZATION,
        "sentence_sha256": sorted(hashes), "client_id_prefixes": sorted(rules.prefixes),
        "clips": sorted(rules.clips | rules.split_clips),
    })
    try:
        check_outbox_file(outbox)
    except OutboxRefusal as refusal:
        outbox.unlink()
        print(f"refused and deleted: {refusal}", file=sys.stderr)
        return 1
    print("\nwrote <pilot>\\hygiene\\ bench-lines.json, titles.json, cv-exclusion.json")
    print(f"wrote <pilot>\\outbox\\cv-exclude.json ({outbox.stat().st_size:,} bytes): {len(hashes):,} sentence "
          f"hashes, {len(rules.prefixes)} prefixes, {len(rules.clips | rules.split_clips):,} clips; "
          "check_outbox_file passed")

    if args.cv17_check:
        cv17_check(rules)
    if args.labels:
        labels_check(args.labels, args.field, sets, lines)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.hygiene", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="write the hygiene lists and outbox\\cv-exclude.json")
    command.add_argument("--cv17-check", action="store_true",
                         help="count the clips rules 1-3 remove from each CV 17.0 ja TSV")
    command.add_argument("--labels", type=Path, help="JSON Lines file of training labels to run BenchLineFilter on")
    command.add_argument("--field", default="transcription", help="label field of --labels (default: transcription)")
    args = parser.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
