"""Galgame title allow-list of the ASR fine-tuning pilot (data survey 4.3).

    python -m scripts.asr_finetune.titles candidates
    python -m scripts.asr_finetune.titles allow

Both read the public tree listing of the OOPPEENN visual-novel set at its
pinned revision (``pilot.OOPPEENN``, no credentials): 597
``GalGame/<maker>_<game>.7z`` archives with their byte sizes.

A maker or game name matches a title, alias or franchise name when they share
a token of at least 4 characters (a run between whitespace, punctuation and
symbols after NFKC, case-folded) or when their normalized forms (``norm``,
case-folded) have a ``difflib`` ratio of at least 0.6. The titles are every
bench and locked title and alias: ``<pilot>\\hygiene\\titles.json`` (built by
``scripts.asr_finetune.hygiene``) and the raw forms it is built from, whose
words are tokens: every bench set item's ``condition.title`` and the keys and
aliases of ``<bench root>\\config\\locked-titles.private.json``.

``candidates`` writes, all private:

- ``<pilot>\\hygiene\\franchise-candidates.json``: every archive whose maker or
  game name matches a title, with its size and each match (field, title, and
  the shared token or the ratio);
- ``<bench root>\\config\\ft-franchise-exclusions.private.json``
  (``{"schema_version": "1", "archives": [...], "names": [...]}``): created
  with every candidate in ``archives`` and no ``names``; when it exists, new
  candidates join ``archives``, nothing is removed and the owner's ``names``
  stay. Every candidate is thus excluded by default; the owner adds franchise
  names;
- ``<bench root>\\config\\ft-franchise-review.private.md``: one table row per
  match (archive, matched title or alias, reason) for the owner's review.

It reads the exclusion list before writing anything, writes it first and the
candidates record last, so a run that stops early fails closed.

``allow`` writes ``<pilot>\\outbox\\galgame-titles.json``
(``{"schema": "ft-pilot-galgame-titles-1", "repo", "revision", "archives":
[{"path", "size"}]}``). It refuses to run while an archive matches a title
that the last ``candidates`` run did not record. It drops every archive in
``archives``, every archive whose maker or game name matches a franchise name
of ``names``, and every archive whose path the outbox guard flags
(``outbox.text_finding``: the path holds a title, alias or franchise name, or
a shingle of a bench reference line). Of the remaining archives of 100 to
500 MB it takes the first 40 of a ``SEED`` shuffle, skipping any that would
bring the total over 12 GB. The file must pass ``check_outbox_file``; a
refused file is deleted (exit 1).

Console output is counts only, never a title, alias, franchise or archive name.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.asr_bench.sets import bench_root
from scripts.asr_finetune.hygiene import _locked_titles, _write_json, bench_sets
from scripts.asr_finetune.outbox import (
    OutboxRefusal,
    _tokens,
    check_outbox_file,
    hygiene_dir,
    outbox_dir,
    text_finding,
)
from scripts.asr_finetune.pilot import OOPPEENN, SEED, norm

OUTBOX_SCHEMA = "ft-pilot-galgame-titles-1"
CANDIDATES_SCHEMA = "ft-pilot-franchise-candidates-1"
EXCLUSIONS_SCHEMA = "1"
FOLDER = "GalGame"
SUFFIX = ".7z"
ARCHIVES = 597
MIN_TOKEN_CHARS = 4
MIN_RATIO = 0.6
MIN_BYTES = 100_000_000
MAX_BYTES = 500_000_000
SAMPLE = 40
MAX_TOTAL_BYTES = 12_000_000_000


@dataclass(frozen=True, order=True)
class Archive:
    path: str
    size: int

    def names(self) -> tuple[tuple[str, str], ...]:
        """(field, text) of the maker and game names of ``GalGame/<maker>_<game>.7z``."""
        maker, _, game = self.path[len(FOLDER) + 1:-len(SUFFIX)].partition("_")
        return ("maker", maker), ("game", game)


@dataclass(frozen=True)
class _Name:
    text: str
    key: str
    tokens: frozenset[str]


def _key(text: str) -> str:
    return norm(text).casefold()


def _long_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(text) if len(token) >= MIN_TOKEN_CHARS)


class Matcher:
    """Token and ratio matching of an archive's maker and game names against
    ``names`` (titles and aliases, or franchise names). Values with the same
    normalized form are one name: the first text, the union of their tokens."""

    def __init__(self, names: Iterable[str]) -> None:
        texts: dict[str, str] = {}
        tokens: dict[str, set[str]] = {}
        for text in names:
            key = _key(text)
            if key:
                texts.setdefault(key, text)
                tokens.setdefault(key, set()).update(_long_tokens(text))
        self.names = [_Name(texts[key], key, frozenset(tokens[key])) for key in sorted(texts)]

    def __len__(self) -> int:
        return len(self.names)

    def matches(self, archive: Archive) -> list[dict[str, Any]]:
        """Each (field, name) pair that matches, with the shared token or the ratio."""
        found: list[dict[str, Any]] = []
        for field, text in archive.names():
            key, tokens = _key(text), _long_tokens(text)
            if not key:
                continue
            for name in self.names:
                shared = tokens & name.tokens
                if shared:
                    found.append({"field": field, "title": name.text, "reason": "token", "token": min(shared)})
                    continue
                matcher = difflib.SequenceMatcher(None, key, name.key)
                if matcher.real_quick_ratio() < MIN_RATIO or matcher.quick_ratio() < MIN_RATIO:
                    continue
                ratio = matcher.ratio()
                if ratio >= MIN_RATIO:
                    found.append({"field": field, "title": name.text, "reason": "ratio", "ratio": round(ratio, 3)})
        return found


def candidates_path() -> Path:
    return hygiene_dir() / "franchise-candidates.json"


def exclusions_path() -> Path:
    return bench_root() / "config" / "ft-franchise-exclusions.private.json"


def review_path() -> Path:
    return bench_root() / "config" / "ft-franchise-review.private.md"


def listing() -> list[Archive]:
    """The pinned revision's ``GalGame/*.7z`` archives, sorted by path."""
    from huggingface_hub import HfApi, RepoFile

    repo, revision = OOPPEENN
    entries = HfApi().list_repo_tree(repo, path_in_repo=FOLDER, repo_type="dataset", revision=revision, token=False)
    archives = sorted(Archive(entry.path, entry.size) for entry in entries
                      if isinstance(entry, RepoFile) and entry.path.endswith(SUFFIX))
    if len(archives) != ARCHIVES:
        raise SystemExit(f"the listing of {repo}@{revision} holds {len(archives)} {FOLDER}/*{SUFFIX} "
                         f"archives, not {ARCHIVES}")
    sizes = [archive.size for archive in archives]
    print(f"listing: {len(archives)} {FOLDER}/*{SUFFIX} archives at {revision[:12]}, "
          f"{min(sizes):,} to {max(sizes):,} bytes")
    return archives


def title_texts() -> list[str]:
    """Raw bench titles, raw locked keys and aliases, then ``titles.json``.
    Refuses a ``titles.json`` that misses a normalized raw value (stale)."""
    path = hygiene_dir() / "titles.json"
    if not path.is_file():
        raise SystemExit(f"the hygiene list {path} is missing (build it with scripts.asr_finetune.hygiene build)")
    raw = _load_json(path, "hygiene list")
    titles = raw.get("titles") if isinstance(raw, dict) else None
    if not isinstance(titles, list) or not titles or not all(isinstance(title, str) for title in titles):
        raise SystemExit(f"the hygiene list {path} holds no 'titles' list of strings")
    texts = [item.condition["title"] for bench_set in bench_sets() for item in bench_set.items
             if isinstance(item.condition.get("title"), str)]
    texts += _locked_titles()
    stale = {norm(text) for text in texts if norm(text)} - set(titles)
    if stale:
        raise SystemExit(f"the hygiene list {path} misses {len(stale)} normalized bench or locked titles; "
                         "rebuild it with scripts.asr_finetune.hygiene build")
    print(f"titles: {len(titles)} normalized bench and locked titles and aliases "
          f"({len(set(texts))} distinct raw values)")
    return [*texts, *titles]


def read_exclusions() -> tuple[list[str], list[str]] | None:
    """(archives, names) of the owner's exclusion list, or None when absent."""
    path = exclusions_path()
    if not path.is_file():
        return None
    raw = _load_json(path, "franchise exclusion list")
    if not isinstance(raw, dict) or raw.get("schema_version") != EXCLUSIONS_SCHEMA:
        raise SystemExit(f"{path}: schema_version is not {EXCLUSIONS_SCHEMA!r}")
    archives, names = raw.get("archives"), raw.get("names")
    for values in (archives, names):
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise SystemExit(f"{path} needs 'archives' and 'names' lists of strings")
    return archives, names


def _load_json(path: Path, what: str) -> Any:
    """The decoded ``path``; a clean exit when it cannot be read or decoded."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"the {what} {path} cannot be read ({type(error).__name__})") from None


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _review(found: Sequence[tuple[Archive, list[dict[str, Any]]]]) -> str:
    rows = [f"| {_cell(archive.path)} | {archive.size / 1e6:,.0f} | {match['field']} | {_cell(match['title'])} | "
            + (f"token {_cell(match['token'])}" if match["reason"] == "token" else f"ratio {match['ratio']:.3f}")
            + " |"
            for archive, matches in found for match in matches]
    return "\n".join([
        "# Galgame franchise review (private)",
        "",
        "Written by `python -m scripts.asr_finetune.titles candidates`. Never commit, upload or print this file.",
        "",
        f"Each of the {len(found)} archives below matches a bench or locked title or alias and was added to "
        "`ft-franchise-exclusions.private.json` (`archives`), so it stays out of the galgame allow-list. "
        "To exclude a franchise, add its names (series, character or brand names) to `names`: `allow` then also "
        f"drops every archive whose maker or game name shares a token of at least {MIN_TOKEN_CHARS} characters "
        f"with a name or has a difflib ratio of at least {MIN_RATIO} to it, and the outbox guard refuses any "
        "outbox file that holds a name.",
        "",
        "| archive | MB | field | matched title or alias | reason |",
        "|---|---:|---|---|---|",
        *rows,
        "",
    ])


def candidates() -> int:
    """Reads the exclusions before anything is written, then writes them
    first and the candidates record last: a run that stops early leaves the
    record behind the exclusions, so ``allow``'s stale check refuses."""
    repo, revision = OOPPEENN
    existing = read_exclusions()
    archives = listing()
    matcher = Matcher(title_texts())
    found = [(archive, matches) for archive in archives if (matches := matcher.matches(archive))]
    by_token = sum(any(match["reason"] == "token" for match in matches) for _, matches in found)
    in_band = sum(MIN_BYTES <= archive.size <= MAX_BYTES for archive, _ in found)
    paths = {archive.path for archive, _ in found}
    if existing is None:
        _write_json(exclusions_path(), {"schema_version": EXCLUSIONS_SCHEMA, "archives": sorted(paths), "names": []})
        exclusions = f"created with {len(paths)} archives and no names"
    else:
        listed, names = existing
        new = paths - set(listed)
        if new:
            _write_json(exclusions_path(), {"schema_version": EXCLUSIONS_SCHEMA,
                                            "archives": sorted(new | set(listed)), "names": names})
        exclusions = f"{len(new)} archives added, now {len(new | set(listed))} archives and {len(names)} names"
    review_path().write_text(_review(found), encoding="utf-8")
    _write_json(candidates_path(), {
        "schema": CANDIDATES_SCHEMA, "repo": repo, "revision": revision,
        "rule": {"min_token_chars": MIN_TOKEN_CHARS, "min_ratio": MIN_RATIO},
        "candidates": [{"path": archive.path, "size": archive.size, "matches": matches} for archive, matches in found],
    })
    print(f"candidates: {len(found)} archives ({by_token} by a shared token, {len(found) - by_token} by ratio only; "
          f"{in_band} of {MIN_BYTES // 1_000_000} to {MAX_BYTES // 1_000_000} MB), "
          f"{sum(len(matches) for _, matches in found)} matches")
    print(f"exclusions: {exclusions}")
    print("wrote <pilot>\\hygiene\\franchise-candidates.json and <bench root>\\config\\ft-franchise-review.private.md")
    return 0


def _recorded_candidates() -> set[str]:
    path = candidates_path()
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run python -m scripts.asr_finetune.titles candidates first")
    raw = _load_json(path, "candidate list")
    pinned = isinstance(raw, dict) and (raw.get("schema"), raw.get("repo"), raw.get("revision")) == (
        CANDIDATES_SCHEMA, *OOPPEENN)
    entries = raw.get("candidates") if pinned else None
    if not isinstance(entries, list) or not all(
            isinstance(entry, dict) and isinstance(entry.get("path"), str) for entry in entries):
        raise SystemExit(f"{path} is not a candidate list of the pinned listing; run candidates again")
    return {entry["path"] for entry in entries}


def allow() -> int:
    repo, revision = OOPPEENN
    archives = listing()
    recorded = _recorded_candidates()
    matcher = Matcher(title_texts())
    unrecorded = sum(1 for archive in archives if archive.path not in recorded and matcher.matches(archive))
    if unrecorded:
        raise SystemExit(f"{unrecorded} archives match a title but are not in {candidates_path()}; "
                         "run python -m scripts.asr_finetune.titles candidates again")
    exclusions = read_exclusions()
    if exclusions is None:
        raise SystemExit(f"{exclusions_path()} is missing; run python -m scripts.asr_finetune.titles candidates first")
    listed, names = set(exclusions[0]), exclusions[1]
    franchise = Matcher(names)
    by_list = {archive.path for archive in archives if archive.path in listed}
    by_name = {archive.path for archive in archives if franchise.matches(archive)}
    try:
        by_guard = {archive.path for archive in archives if text_finding(archive.path)}
    except OutboxRefusal as refusal:
        print(refusal, file=sys.stderr)
        return 1
    dropped = by_list | by_name | by_guard
    print(f"excluded: {len(dropped)} archives ({len(by_list)} listed in the exclusions, of {len(listed)} listed; "
          f"{len(by_name)} matching one of {len(franchise)} franchise names; {len(by_guard)} flagged by the outbox "
          f"guard); {len(recorded - listed)} recorded candidates the owner allowed")
    pool = [archive for archive in archives if archive.path not in dropped and MIN_BYTES <= archive.size <= MAX_BYTES]
    print(f"pool: {len(pool)} allowed archives of {MIN_BYTES // 1_000_000} to {MAX_BYTES // 1_000_000} MB "
          f"({sum(archive.size for archive in pool):,} bytes); seed {SEED}")
    order = list(pool)
    random.Random(SEED).shuffle(order)
    chosen: list[Archive] = []
    total = 0
    for archive in order:
        if total + archive.size <= MAX_TOTAL_BYTES:
            chosen.append(archive)
            total += archive.size
            if len(chosen) == SAMPLE:
                break
    if len(chosen) < SAMPLE:
        raise SystemExit(f"only {len(chosen)} of {len(pool)} allowed archives of {MIN_BYTES:,} to {MAX_BYTES:,} "
                         f"bytes fit {MAX_TOTAL_BYTES:,} bytes; {SAMPLE} are needed")
    chosen.sort()
    outbox = outbox_dir() / "galgame-titles.json"
    _write_json(outbox, {"schema": OUTBOX_SCHEMA, "repo": repo, "revision": revision,
                         "archives": [{"path": archive.path, "size": archive.size} for archive in chosen]})
    try:
        check_outbox_file(outbox)
    except OutboxRefusal as refusal:
        outbox.unlink()
        print(f"refused and deleted: {refusal}", file=sys.stderr)
        return 1
    sizes = [archive.size for archive in chosen]
    print(f"wrote <pilot>\\outbox\\galgame-titles.json ({outbox.stat().st_size:,} bytes): {len(chosen)} archives, "
          f"{total:,} bytes ({min(sizes):,} to {max(sizes):,} each); check_outbox_file passed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.titles", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("candidates", help="write the private franchise candidates, exclusions and review")
    commands.add_parser("allow", help="write outbox\\galgame-titles.json")
    args = parser.parse_args(argv)
    return candidates() if args.command == "candidates" else allow()


if __name__ == "__main__":
    raise SystemExit(main())
