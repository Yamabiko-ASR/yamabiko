"""Classify mutmut survivors by composition-surface section.

Reads .mutmut-cache (sqlite), joins Mutant -> Line -> SourceFile, then
maps each mutant's line number to its enclosing top-level def / class
(method granularity inside classes), derived from the CURRENT source at
runtime. Reports kill rate per section so we can distinguish "tests
hardened the composition surface" from "the pipeline-stage code is
still largely uncovered".

Why dynamic bounds: hardcoded line-range tables become stale as target
files grow (watchdog.py has grown substantially), causing mutants past
the ceiling to be classified "out-of-bounds" — a bucket that appears in
NO report row and NO aggregate, silently hiding exactly the
composition-surface functions this script exists to measure. Bounds are
now derived per run, composition membership follows the owner's NAME
(stable across refactors), and main() self-audits that every mutant in
the cache lands in exactly one printed bucket, exiting non-zero on any
accounting mismatch.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

CACHE = ".mutmut-cache"

TARGETS = [
    "yamabiko/main.py",
    "yamabiko/watchdog.py",
    "yamabiko/overlay_app/__main__.py",
]

# Composition-surface membership by top-level OWNER name. A section
# belongs to the composition surface iff its owning top-level def /
# class is listed here ("class Foo" and "Foo.method" both resolve to
# owner "Foo"). Everything else — helpers, module setup, pipeline-stage
# loops — is pre-composition.
COMPOSITION_OWNERS = {
    # main.py
    "PipelineApp", "parse_args", "build_pipeline", "run_pipeline",
    "amain", "main",
    # watchdog.py
    "WatchdogApp", "build_watchdog", "run_watchdog",
    # overlay_app/__main__.py
    "OverlayApp", "build_overlay", "run_overlay",
}

# Status -> report column. Anything not listed lands in "other" so the
# accounting can never silently drop a new mutmut status value.
_KILLED = {"ok_killed"}
_TIMEOUT = {"bad_timeout"}
_SURVIVED = {"bad_survived"}
_SUSPICIOUS = {"ok_suspicious"}
_UNTESTED = {"untested"}


def sections_for(source: str) -> list[tuple[int, str]]:
    """Return [(start_line_1based, section_name)] for top-level defs /
    classes, descending into class bodies at method granularity.
    """
    out: list[tuple[int, str]] = [(1, "(module level)")]
    cls: str | None = None
    for i, raw in enumerate(source.splitlines(), 1):
        m = re.match(r"^(class|def|async def)\s+(\w+)", raw)
        if m:
            cls = m.group(2) if m.group(1) == "class" else None
            out.append((i, m.group(2) if cls is None else f"class {cls}"))
            continue
        m = re.match(r"^    (?:async )?def\s+(\w+)", raw)
        if m and cls:
            out.append((i, f"{cls}.{m.group(1)}"))
    return out


def section_at(sections: list[tuple[int, str]], line: int) -> str:
    """Name of the section containing a 1-based editor line."""
    name = sections[0][1]
    for start, n in sections:
        if start <= line:
            name = n
        else:
            break
    return name


def owner_of(section: str) -> str:
    """Top-level owner of a section name.

    "class Foo" -> "Foo"; "Foo.method" -> "Foo"; "bar" -> "bar".
    """
    if section.startswith("class "):
        return section[len("class "):]
    return section.split(".", 1)[0]


def is_composition(section: str) -> bool:
    return owner_of(section) in COMPOSITION_OWNERS


def normalize(fn: str) -> str:
    # Windows paths may use either separator depending on how mutmut
    # wrote them; replace backslash without an escape sequence that
    # triggers SyntaxWarning in some contexts.
    return fn.replace(chr(92), "/")


def calibrate_offset(rows: list[tuple[int, str, str]], src_lines: list[str]) -> int:
    """Vote whether stored line numbers are 0- or 1-based by matching
    stored line text against the file (scratch-script idiom)."""
    votes = {0: 0, 1: 0}
    for ln, txt, _ in rows[:60]:
        for off in (0, 1):
            idx = ln + off - 1  # candidate 1-based editor line -> index
            if 0 <= idx < len(src_lines) and src_lines[idx].strip() == (txt or "").strip():
                votes[off] += 1
    return 1 if votes[1] > votes[0] else 0


def _count(counts: dict, status: str) -> None:
    if status in _KILLED:
        counts["kill"] += 1
    elif status in _TIMEOUT:
        counts["tmout"] += 1
    elif status in _SURVIVED:
        counts["surv"] += 1
    elif status in _SUSPICIOUS:
        counts["susp"] += 1
    elif status in _UNTESTED:
        counts["untst"] += 1
    else:
        counts["other"] += 1
    counts["total"] += 1


def _fmt_row(name: str, c: dict) -> str:
    caught = c["kill"] + c["tmout"]
    run = caught + c["surv"] + c["susp"]
    pct = f"{caught / run * 100:5.1f}%" if run else "   n/a"
    return (
        f"{name:50} {c['kill']:>5} {c['tmout']:>5} {c['surv']:>5}"
        f" {c['susp']:>5} {c['untst']:>5} {c['other']:>5} {pct:>7}"
    )


def main(cache: str = CACHE, root: Path | str = Path(".")) -> int:
    root = Path(root)
    con = sqlite3.connect(cache)
    cur = con.cursor()
    query = """
    SELECT Mutant.status, SourceFile.filename, Line.line_number, Line.line
    FROM Mutant
    JOIN Line ON Mutant.line = Line.id
    JOIN SourceFile ON Line.sourcefile = SourceFile.id
    """
    rows = list(cur.execute(query))
    con.close()
    total = len(rows)
    print(f"Total mutants: {total}")

    by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for status, fname, line_no, line_txt in rows:
        by_file[normalize(fname)].append((line_no, line_txt or "", status))

    # bucket[(fname, section)] -> column counts
    bucket: dict = defaultdict(lambda: defaultdict(int))
    section_order: dict[str, list[str]] = {}
    for fname in TARGETS:
        frows = by_file.pop(fname, [])
        src_path = root / fname
        try:
            src = src_path.read_text(encoding="utf-8")
        except OSError:
            # Source vanished: keep the mutants visible + counted.
            section_order[fname] = ["(source missing)"]
            for _, _, status in frows:
                _count(bucket[(fname, "(source missing)")], status)
            continue
        src_lines = src.splitlines()
        secs = sections_for(src)
        section_order[fname] = [name for _, name in secs]
        off = calibrate_offset(frows, src_lines)
        for line_no, _, status in frows:
            sec = section_at(secs, line_no + off)
            _count(bucket[(fname, sec)], status)

    # Anything left in by_file is outside the target set — keep it in
    # the report + accounting rather than dropping it.
    unknown_files = sorted(by_file)
    for fname in unknown_files:
        for _, _, status in by_file[fname]:
            _count(bucket[(fname, f"UNKNOWN-FILE:{fname}")], status)

    print()
    print(
        f'{"section":50} {"kill":>5} {"tmout":>5} {"surv":>5} {"susp":>5}'
        f' {"untst":>5} {"other":>5} {"KILL%":>7}'
    )
    print("-" * 97)

    counted = 0
    for fname in TARGETS:
        print(f"=== {fname} ===")
        for name in section_order.get(fname, []):
            c = bucket.get((fname, name))
            if not c:
                continue  # section holds no mutants — nothing to drop
            counted += c["total"]
            print(_fmt_row(name, c))
    if unknown_files:
        print("=== files outside the target set ===")
        for fname in unknown_files:
            c = bucket[(fname, f"UNKNOWN-FILE:{fname}")]
            counted += c["total"]
            print(_fmt_row(f"UNKNOWN-FILE:{fname}", c))

    # Aggregates over the target files, by owner-name membership.
    print()
    print("=== AGGREGATES ===")
    for label, want_comp in [
        ("Composition surface", True),
        ("Pre-composition", False),
    ]:
        agg: dict = defaultdict(int)
        for (fname, section), c in bucket.items():
            if fname not in TARGETS:
                continue
            if is_composition(section) is want_comp:
                for k, v in c.items():
                    agg[k] += v
        caught = agg["kill"] + agg["tmout"]
        run = caught + agg["surv"] + agg["susp"]
        pct = f"{caught / run * 100:.1f}%" if run else "n/a"
        print(
            f"{label:40} caught={caught:>3} surv={agg['surv']:>3}"
            f" susp={agg['susp']:>2} untst={agg['untst']:>3} kill%={pct}"
        )

    # Self-audit: every mutant in the cache must be visible in exactly
    # one printed bucket. A mismatch means a classification path is
    # dropping mutants again (the silent-hiding failure mode) — fail loudly.
    if counted != total:
        print(
            f"ERROR: accounting mismatch — report shows {counted} of "
            f"{total} mutants; some mutants are invisible to the report "
            f"(accounting mismatch).",
            file=sys.stderr,
        )
        return 1
    print(f"\naccounting: all {total} mutants reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
