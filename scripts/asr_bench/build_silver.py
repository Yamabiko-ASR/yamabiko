"""Build the silver ASR bench set: development E0 clips referenced by their own
Japanese subtitles.

One item per development or blocker fixture (D1-D8) whose title is not a
locked confirmation title (the private list, see ``titles``): the fixture's
registered canonical 16 kHz master (240 s) as audio, and the cues of its v3
editorial subtitle sidecar as reference lines. Cue text loses its markup and its
parenthesized SDH annotations (speaker names, sound descriptions, readings),
which are not spoken; cues left without text are dropped. Subtitles are
edited text, so the scorer uses a looser catch threshold for this set, and no
speech-free spans are known. Locked confirmation fixtures (L1-L4) are never
read.

    python -m scripts.asr_bench.build_silver --out <bench_root>/sets/silver-v2

The E0 private root comes from ``YAMABIKO_E0_PRIVATE_ROOT`` and is only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e0_evidence.normalize_ja import normalize_minimal

from .sets import bench_root, write_audio, write_set
from .titles import LockedTitles, load_locked_titles

STATE_FILE = "private-rebuild-v2.state.private.json"
SIDECAR_FILE = "editorial-subtitle-v3.private.json"
LOCKED_PARTITION = "locked_confirmation"
TEXT_CLEANING = (
    "E0 subtitle-proposal cleaning (HTML and ASS override tags removed, line "
    "breaks to spaces, whitespace collapsed), Unicode format characters "
    "(category Cf) removed, parenthesized SDH annotations removed"
)
# Japanese SDH puts speaker names, sound descriptions and readings in
# parentheses, possibly nested: "(田中(たなか))いいよなぁ".
_ANNOTATION_RE = re.compile(r"[(（][^()（）]*[)）]")


@dataclass(frozen=True)
class Fixture:
    label: str
    fixture_id: str
    master_id: str
    title_id: str
    partition: str
    source: Path
    audio_stream_ordinal: int
    subtitle_stream_ordinal: int
    strata: tuple[str, ...]

    @property
    def locked(self) -> bool:
        return self.partition == LOCKED_PARTITION or self.label.startswith("L")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def e0_boundary():
    """The read-only E0 private boundary named by the environment."""
    from scripts.e0_evidence.registry import boundary_from_environment

    return boundary_from_environment(repo_root(), os.environ)


def load_fixtures(state_path: Path) -> dict[str, Fixture]:
    """Every fixture of the E0 rebuild state, keyed by label."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    fixtures: dict[str, Fixture] = {}
    for label, raw in state["fixtures"].items():
        fixtures[label] = Fixture(
            label=label,
            fixture_id=raw["fixture_id"],
            master_id=raw["master_id"],
            title_id=raw["title_id"],
            partition=raw["partition"],
            source=Path(raw["source"]),
            audio_stream_ordinal=int(raw["audio_stream_ordinal"]),
            subtitle_stream_ordinal=int(raw["subtitle_stream_ordinal"]),
            strata=tuple(raw["strata"]),
        )
    return fixtures


def locked_titles(fixtures: Mapping[str, Fixture]) -> LockedTitles:
    """The private locked-title list, checked against the E0 locked fixtures."""
    return load_locked_titles(
        (fixture.label, fixture.source) for fixture in fixtures.values() if fixture.locked
    )


def development_fixtures(fixtures: Mapping[str, Fixture]) -> list[Fixture]:
    """The D fixtures (development and blocker partitions), never L1-L4, and
    never a D fixture whose title is a locked confirmation title, in any
    season: scoring candidates on it would tune toward the locked
    confirmation material."""
    locked = locked_titles(fixtures)
    chosen = [
        fixture for label, fixture in sorted(fixtures.items())
        if label.startswith("D") and not fixture.locked
        and locked.path_title(fixture.source) is None
    ]
    if not chosen:
        raise RuntimeError("the E0 state lists no development fixtures")
    return chosen


def strip_annotations(text: str) -> str:
    while (stripped := _ANNOTATION_RE.sub(" ", text)) != text:
        text = stripped
    return " ".join(text.split())


def clean_cue_text(raw_text: str) -> str:
    from scripts.e0_evidence.subtitle_proposals import _clean_editable_text

    cleaned = _clean_editable_text(raw_text)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")
    return strip_annotations(cleaned)


def _sidecar_path(e0_root: Path, fixture: Fixture) -> Path:
    found = sorted((e0_root / "references" / "work").glob(f"*/{fixture.fixture_id}/{SIDECAR_FILE}"))
    if len(found) != 1:
        raise RuntimeError(f"{fixture.label}: expected one {SIDECAR_FILE}, found {len(found)}")
    return found[0]


def mark_overlaps(lines: Sequence[dict]) -> None:
    """Cues shown at the same time usually mean simultaneous speakers."""
    for line in lines:
        line["overlap"] = any(
            other is not line
            and line["start_s"] < other["end_s"]
            and other["start_s"] < line["end_s"]
            for other in lines
        )


def _lines(cues: Sequence[Mapping[str, Any]], duration_s: float) -> tuple[list[dict], dict[str, int]]:
    kept: list[dict] = []
    stats = {"annotated_cues": 0, "dropped_empty_cues": 0}
    for cue in cues:
        if _ANNOTATION_RE.search(cue["raw_text"]):
            stats["annotated_cues"] += 1
        text = clean_cue_text(cue["raw_text"])
        if not normalize_minimal(text):
            stats["dropped_empty_cues"] += 1
            continue
        kept.append({
            "line_id": f"c{cue['source_ordinal']:04d}",
            "start_s": cue["relative_start_ms"] / 1000.0,
            "end_s": min(cue["relative_end_ms"] / 1000.0, duration_s),
            "text": text,
            "source": "subtitle",
            "overlap": False,
        })
    mark_overlaps(kept)
    return kept, stats


def build(out_dir: Path, state_path: Path | None = None) -> None:
    from scripts.e0_evidence.pcm import load_canonical_master
    from scripts.e0_evidence.registry import load_registered_master
    from scripts.e0_evidence.subtitle_proposals import load_editorial_sidecar

    boundary = e0_boundary()
    state_path = state_path or boundary.root / STATE_FILE
    fixtures = development_fixtures(load_fixtures(state_path))
    items: list[dict] = []
    provenance: dict[str, dict] = {}
    for fixture in fixtures:
        if fixture.locked:
            raise AssertionError(f"{fixture.label} is a locked fixture")
        sidecar_path = _sidecar_path(boundary.root, fixture)
        sidecar_bytes = sidecar_path.read_bytes()
        sidecar = load_editorial_sidecar(sidecar_bytes)
        if sidecar["master_id"] != fixture.master_id:
            raise RuntimeError(f"{fixture.label}: sidecar belongs to another master")
        manifest = load_registered_master(fixture.master_id, boundary)
        if manifest.fixture_id != fixture.fixture_id:
            raise RuntimeError(f"{fixture.label}: master belongs to another fixture")
        master = load_canonical_master(manifest, boundary)
        duration_s = master.samples.size / manifest.sample_rate_hz
        lines, stats = _lines(sidecar["cues"], duration_s)
        audio = f"audio/{fixture.label}.f32"
        sample_count = write_audio(out_dir / audio, master.samples)
        items.append({
            "item_id": fixture.label,
            "audio": audio,
            "sample_count": sample_count,
            "condition": {
                "fixture_label": fixture.label,
                "partition": fixture.partition,
                "strata": list(fixture.strata),
                "title_id": fixture.title_id,
            },
            "lines": lines,
            "speech_free": [],
        })
        provenance[fixture.label] = {
            "fixture_id": fixture.fixture_id,
            "master_id": fixture.master_id,
            "canonical_pcm_sha256": manifest.canonical_pcm_sha256,
            "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "cue_count": len(sidecar["cues"]),
            **stats,
        }
        print(
            f"{fixture.label}: {duration_s:.1f} s, {len(lines)} lines, "
            f"{stats['annotated_cues']} cues with annotations, "
            f"{stats['dropped_empty_cues']} empty cues dropped", flush=True,
        )
    written = write_set(
        out_dir,
        set_id=out_dir.name,
        kind="silver",
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        build={
            "builder": "scripts.asr_bench.build_silver",
            "e0_state_sha256": hashlib.sha256(Path(state_path).read_bytes()).hexdigest(),
            "sidecar_file": SIDECAR_FILE,
            "text_cleaning": TEXT_CLEANING,
            "dropped": "cues whose cleaned text is empty after normalize_minimal",
            "excluded_partitions": [LOCKED_PARTITION],
            "fixtures": provenance,
        },
        items=items,
    )
    print(f"wrote {len(written.items)} items to {out_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=bench_root() / "sets" / "silver-v2")
    parser.add_argument("--state", type=Path, help=f"E0 rebuild state (default: <E0 root>/{STATE_FILE})")
    args = parser.parse_args(argv)
    build(args.out, args.state)


if __name__ == "__main__":
    main()
