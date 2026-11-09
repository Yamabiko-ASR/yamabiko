"""Build the held-out evaluation sets of the fine-tuning pilot.

``ingcrowd-v1`` and ``reazon-test-v1`` are bench sets
(``scripts.asr_bench.sets``), written through ``sets.write_set``; neither is
ever training data.

    python -m scripts.asr_finetune.eval_sets ingcrowd --out <bench root>\\sets\\ingcrowd-v1
    python -m scripts.asr_finetune.eval_sets reazon-test --out <bench root>\\sets\\reazon-test-v1 [--clips 1000]
    python -m scripts.asr_finetune.eval_sets subset --set <set dir> --first N --out <set dir>

``ingcrowd`` (kind ``synthetic``: exact, time-aligned verbatim references,
so the 75% aligned rule and CER apply). The two-speaker dialogues of the
``pilot.INGCROWD`` repos at their pinned revisions: per dialogue the mixed
``<dialogue>_audio.wav`` and the ELAN ``<dialogue>.eaf``, each checked
against the revision's LFS SHA-256 (or, for a file kept in git, its blob
id). The audio is decoded to 16 kHz mono float32 with
``faster_whisper.decode_audio``. Lines come from both speaker tiers of the
EAF: each ``ALIGNABLE_ANNOTATION``, its span from ``TIME_SLOT_REF1`` and
``TIME_SLOT_REF2`` (milliseconds). The transcribers' tags ``<...>`` (``<la>``
laughter, ``<?>`` unclear speech) are removed and a line left without text
(nothing after ``normalize_minimal``) is dropped. A line is ``overlap: true``
when it overlaps a kept line of the other tier. Each dialogue is cut into
items of at most 300 s, each cut at the midpoint (whole millisecond) of a
gap of at least 0.5 s between annotations (dropped ones included, so no
annotated sound is split), the cut nearest an equal share of what remains
of the dialogue. Line source ``verbatim``; no speech-free spans (the
recordings hold untranscribed sounds); ``condition.cluster`` is
``<repo short name>/<dialogue number>``. ``build.attribution`` names both
repos, their revisions and the CC BY-NC 4.0 licence.

``reazon-test`` (kind ``silver``: broadcast captions). Labels: the last
2,000,000 bytes of ReazonSpeech v2 ``all.tsv``, read with one HTTP Range
request, rows whose key starts with ``fff/``. The file is sorted by key, so
the range holds every row of the shard when its first complete row belongs
to another shard and no other shard follows the first ``fff/`` row; both
are required. A repeated row must repeat the same text. Audio: ``fff.tar``
streamed once (its byte count checked against ``pilot.REAZON_TEST_SHARD``,
its SHA-256 recorded, nothing written but the chosen clips). Every labelled
key must be a FLAC member, so the FLAC members with a label equal the
distinct labelled keys. ``--clips`` (default 1,000) keys are drawn without
replacement from the sorted distinct keys with ``pilot.SEED``
(``numpy.random.default_rng``) and decoded like ingcrowd: one item per clip,
in key order, with one line ``[0, duration]`` of source ``subtitle``.

``subset``: the first N items of a set, audio files and item entries copied
unchanged, ``build.subset_of`` naming the source set (its attribution kept).

Downloads go to a temporary folder inside ``--out`` that is removed
afterwards, so the set keeps only its own audio. A build replaces the
folder's ``audio`` and ``set.json``. No ASR or VAD model selects or filters
anything.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from scripts.asr_bench.sets import SAMPLE_RATE_HZ, load_set, write_audio, write_set
from scripts.asr_finetune.pilot import INGCROWD, REAZON_ALL_TSV, REAZON_BASE, REAZON_TEST_SHARD, SEED
from scripts.e0_evidence.normalize_ja import normalize_minimal

INGCROWD_LICENSE = "CC BY-NC 4.0"
TAG_RE = re.compile(r"<[^<>]*>")
CUT_GAP_MS = 500
MAX_PIECE_MS = 300_000
REAZON_TAIL_BYTES = 2_000_000
DEFAULT_CLIPS = 1000
_SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000
_CHUNK = 1 << 20
_TIMEOUT_S = 120


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reset(out_dir: Path) -> None:
    """Remove the folder's manifest and audio before a build writes them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "set.json").unlink(missing_ok=True)
    if (out_dir / "audio").exists():
        shutil.rmtree(out_dir / "audio")


def _decoder() -> tuple[Any, str]:
    import faster_whisper

    return faster_whisper.decode_audio, (
        f"faster_whisper.decode_audio {faster_whisper.__version__} (PyAV), 16 kHz mono float32"
    )


# IngCrowd -----------------------------------------------------------------------


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    tier: str
    start_ms: int
    end_ms: int
    value: str


def read_eaf(path: Path) -> list[Annotation]:
    """Every ``ALIGNABLE_ANNOTATION`` of an ELAN file with its tier and span."""
    root = ET.parse(path).getroot()
    header = root.find("HEADER")
    units = "milliseconds" if header is None else header.get("TIME_UNITS", "milliseconds")
    if units != "milliseconds":
        raise ValueError(f"{path.name}: time units {units!r}, expected milliseconds")
    slots = {slot.get("TIME_SLOT_ID"): slot.get("TIME_VALUE") for slot in root.iter("TIME_SLOT")}
    annotations: list[Annotation] = []
    for tier in root.iter("TIER"):
        for node in tier.iterfind("ANNOTATION/ALIGNABLE_ANNOTATION"):
            ident = str(node.get("ANNOTATION_ID"))
            start = slots.get(node.get("TIME_SLOT_REF1"))
            end = slots.get(node.get("TIME_SLOT_REF2"))
            if start is None or end is None:
                raise ValueError(f"{path.name}: annotation {ident} has an unaligned time slot")
            annotations.append(Annotation(
                ident, str(tier.get("TIER_ID")), int(start), int(end),
                node.findtext("ANNOTATION_VALUE") or "",
            ))
    return annotations


def clean_verbatim(value: str) -> str:
    """An annotation's text without the transcribers' ``<...>`` tags."""
    return TAG_RE.sub("", value).strip()


def piece_bounds(spans: Sequence[tuple[int, int]], duration_ms: int) -> list[int]:
    """Item boundaries in ms, 0 first and ``duration_ms`` last. Each inner
    boundary is the midpoint of a gap of at least ``CUT_GAP_MS`` between
    ``spans``, no piece is longer than ``MAX_PIECE_MS``, and each boundary is
    the candidate nearest an equal share of what remains."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    cuts = [
        (left[1] + right[0]) // 2 for left, right in zip(merged, merged[1:])
        if right[0] - left[1] >= CUT_GAP_MS
    ]
    bounds = [0]
    while duration_ms - bounds[-1] > MAX_PIECE_MS:
        start = bounds[-1]
        target = start + (duration_ms - start) / math.ceil((duration_ms - start) / MAX_PIECE_MS)
        options = [cut for cut in cuts if start < cut <= start + MAX_PIECE_MS]
        if not options:
            raise ValueError(
                f"no gap of {CUT_GAP_MS / 1000} s within {MAX_PIECE_MS / 1000:.0f} s after "
                f"{start / 1000:.1f} s"
            )
        bounds.append(min(options, key=lambda cut: abs(cut - target)))
    return [*bounds, duration_ms]


def dialogue_items(
    out_dir: Path, dialogue: str, cluster: str, pcm: np.ndarray, annotations: Sequence[Annotation]
) -> tuple[list[dict], dict[str, Any]]:
    """Write one dialogue's items; return them and the dialogue's counts."""
    duration_ms = pcm.size // _SAMPLES_PER_MS
    tiers = sorted({annotation.tier for annotation in annotations})
    if len(tiers) != 2:
        raise ValueError(f"{dialogue}: {len(tiers)} annotated tiers, expected 2")
    if len({annotation.annotation_id for annotation in annotations}) != len(annotations):
        raise ValueError(f"{dialogue}: annotation ids repeat")
    for annotation in annotations:
        if not 0 <= annotation.start_ms < annotation.end_ms <= duration_ms:
            raise ValueError(
                f"{dialogue}: annotation {annotation.annotation_id} [{annotation.start_ms}, "
                f"{annotation.end_ms}] ms lies outside the {duration_ms} ms recording"
            )
    kept: list[tuple[Annotation, str]] = []
    for annotation in sorted(annotations, key=lambda one: (one.start_ms, one.end_ms, one.tier)):
        text = clean_verbatim(annotation.value)
        if normalize_minimal(text):
            kept.append((annotation, text))
    overlapping = {
        line.annotation_id for line, _ in kept for other, _ in kept
        if line.tier != other.tier and line.start_ms < other.end_ms and other.start_ms < line.end_ms
    }
    bounds = piece_bounds([(annotation.start_ms, annotation.end_ms) for annotation in annotations], duration_ms)
    items: list[dict] = []
    for index, (first_ms, last_ms) in enumerate(zip(bounds, bounds[1:])):
        last_sample = pcm.size if index == len(bounds) - 2 else last_ms * _SAMPLES_PER_MS
        item_id = f"{dialogue}-p{index}"
        audio = f"audio/{item_id}.f32"
        items.append({
            "item_id": item_id,
            "audio": audio,
            "sample_count": write_audio(out_dir / audio, pcm[first_ms * _SAMPLES_PER_MS:last_sample]),
            "condition": {
                "dialogue": dialogue,
                "piece": index,
                "offset_s": first_ms / 1000,
                "cluster": cluster,
            },
            "lines": [
                {
                    "line_id": line.annotation_id,
                    "start_s": (line.start_ms - first_ms) / 1000,
                    "end_s": (line.end_ms - first_ms) / 1000,
                    "text": text,
                    "source": "verbatim",
                    "overlap": line.annotation_id in overlapping,
                }
                for line, text in kept if first_ms <= line.start_ms and line.end_ms <= last_ms
            ],
            "speech_free": [],
        })
    placed = sum(len(item["lines"]) for item in items)
    if placed != len(kept):
        raise AssertionError(f"{dialogue}: {len(kept) - placed} lines cross an item boundary")
    return items, {
        "duration_s": pcm.size / SAMPLE_RATE_HZ,
        "tiers": tiers,
        "annotations": len(annotations),
        "lines": len(kept),
        "dropped_without_text": len(annotations) - len(kept),
        "overlap_lines": len(overlapping),
        "items": [item["item_id"] for item in items],
    }


def _verified_sha256(path: Path, sibling: Any) -> str:
    """The SHA-256 of a downloaded file, after checking it against the
    revision's LFS SHA-256 or, for a file kept in git, its blob id."""
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sibling.lfs is not None:
        matches = sha256 == sibling.lfs.sha256
    else:
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
        matches = blob == sibling.blob_id
    if len(data) != sibling.size or not matches:
        raise SystemExit(f"{sibling.rfilename}: the download does not match the pinned revision")
    return sha256


def build_ingcrowd(out_dir: Path, downloads: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    decode_audio, decoder = _decoder()
    api = HfApi()
    _reset(out_dir)
    items: list[dict] = []
    dialogues: dict[str, dict] = {}
    for repo_id, revision in INGCROWD:
        short = repo_id.split("/", 1)[1]
        siblings = {
            sibling.rfilename: sibling
            for sibling in api.dataset_info(repo_id, revision=revision, files_metadata=True).siblings
        }
        pattern = re.compile(rf"{re.escape(short)}\d+(?=/)")
        names = sorted({match.group(0) for name in siblings if (match := pattern.match(name))})
        if not names:
            raise SystemExit(f"{repo_id}@{revision} holds no {short}NN dialogue folders")
        for name in names:
            number = name[len(short):]
            files: dict[str, tuple[Path, str]] = {}
            for kind, filename in (("audio", f"{name}/{name}_audio.wav"), ("eaf", f"{name}/{name}.eaf")):
                if filename not in siblings:
                    raise SystemExit(f"{repo_id}@{revision} lacks {filename}")
                path = Path(hf_hub_download(
                    repo_id, filename, repo_type="dataset", revision=revision, local_dir=downloads / short,
                ))
                files[kind] = path, _verified_sha256(path, siblings[filename])
            pcm = decode_audio(str(files["audio"][0]))
            cluster = f"{short}/{number}"
            written, counts = dialogue_items(out_dir, name, cluster, pcm, read_eaf(files["eaf"][0]))
            for path, _ in files.values():
                path.unlink()
            items.extend(written)
            dialogues[name] = {
                "repo_id": repo_id,
                "cluster": cluster,
                "audio": f"{name}/{name}_audio.wav",
                "audio_sha256": files["audio"][1],
                "eaf": f"{name}/{name}.eaf",
                "eaf_sha256": files["eaf"][1],
                **counts,
            }
            print(
                f"{cluster}: {counts['duration_s']:.0f} s, {counts['lines']} lines "
                f"({counts['dropped_without_text']} dropped, {counts['overlap_lines']} overlapping), "
                f"{len(written)} items",
                flush=True,
            )
    written_set = write_set(
        out_dir,
        set_id=out_dir.name,
        kind="synthetic",
        created_at_utc=_now(),
        build={
            "builder": "scripts.asr_finetune.eval_sets ingcrowd",
            "role": "held-out evaluation set of the fine-tuning pilot; never training data",
            "attribution": {
                "datasets": [
                    {"repo_id": repo_id, "revision": revision, "url": f"https://huggingface.co/datasets/{repo_id}"}
                    for repo_id, revision in INGCROWD
                ],
                "license": INGCROWD_LICENSE,
            },
            "decoder": decoder,
            "rules": {
                "lines": "every ALIGNABLE_ANNOTATION of both speaker tiers, span TIME_SLOT_REF1-REF2 in ms",
                "text": "tags <...> removed (<la> laughter, <?> unclear), edges stripped; a line left "
                        "without text after normalize_minimal is dropped",
                "overlap": "a line overlapping a kept line of the other tier",
                "cut_gap_s": CUT_GAP_MS / 1000,
                "max_item_s": MAX_PIECE_MS / 1000,
                "cut": "midpoint (whole ms) of a gap between annotations (dropped ones included), "
                       "the candidate nearest an equal share of what remains of the dialogue",
                "speech_free": "none (untranscribed sounds)",
            },
            "dialogues": dialogues,
        },
        items=items,
    )
    hours = sum(item.duration_s for item in written_set.items) / 3600
    lines = sum(len(item.lines) for item in written_set.items)
    clusters = {item.condition["cluster"] for item in written_set.items}
    print(
        f"wrote {len(written_set.items)} items ({hours:.3f} h, {lines} lines, "
        f"{sum(line.overlap for item in written_set.items for line in item.lines)} overlapping) "
        f"from {len(clusters)} clusters to {out_dir}"
    )


# ReazonSpeech -------------------------------------------------------------------


def _get(url: str, byte_range: tuple[int, int] | None = None) -> Any:
    headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=_TIMEOUT_S)


def shard_prefix() -> str:
    return PurePosixPath(REAZON_TEST_SHARD[0]).stem + "/"


def tail_labels(prefix: str) -> tuple[dict[str, str], dict[str, Any]]:
    """The shard's labels (key -> caption) from the tail of ``all.tsv``."""
    name, size = REAZON_ALL_TSV
    first = size - REAZON_TAIL_BYTES
    expected = f"bytes {first}-{size - 1}/{size}"
    with _get(REAZON_BASE + name, (first, size - 1)) as response:
        answer = response.headers.get("Content-Range")
        if response.status != 206 or answer != expected:
            raise SystemExit(f"{name}: the server answered {response.status} {answer!r}, expected 206 {expected!r}")
        data = response.read()
    if len(data) != REAZON_TAIL_BYTES:
        raise SystemExit(f"{name}: read {len(data)} bytes of the range, expected {REAZON_TAIL_BYTES}")
    rows = data[data.index(b"\n") + 1:].decode("utf-8").split("\n")
    if rows and not rows[-1]:
        rows.pop()
    if not rows or rows[0].startswith(prefix):
        raise SystemExit(f"{name}: the range's first complete row is a {prefix} row; it may miss some of the shard")
    start = next((index for index, row in enumerate(rows) if row.startswith(prefix)), None)
    if start is None:
        raise SystemExit(f"{name}: the range holds no {prefix} row")
    if any(not row.startswith(prefix) for row in rows[start:]):
        raise SystemExit(f"{name}: rows of another shard follow the first {prefix} row")
    labels: dict[str, str] = {}
    repeated: list[str] = []
    for row in rows[start:]:
        key, tab, text = row.partition("\t")
        text = text.strip()
        if not tab or not key.endswith(".flac") or not text:
            raise SystemExit(f"{name}: malformed row for {key!r}")
        if key in labels:
            if labels[key] != text:
                raise SystemExit(f"{name}: {key} is labelled twice with different texts")
            repeated.append(key)
        labels[key] = text
    return labels, {
        "file": name,
        "bytes": size,
        "range": [first, size - 1],
        "first_complete_row_shard": rows[0].split("/", 1)[0],
        "rows": len(rows) - start,
        "distinct_keys": len(labels),
        "repeated_rows": repeated,
    }


class _Counted:
    """A byte stream read through SHA-256 and a byte count."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self._digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self.bytes += len(data)
        self._digest.update(data)
        return data

    def drain(self) -> None:
        while self.read(_CHUNK):
            pass

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def reazon_item_id(key: str) -> str:
    return key.removesuffix(".flac").replace("/", "-")


def build_reazon_test(out_dir: Path, clips: int) -> None:
    decode_audio, decoder = _decoder()
    prefix = shard_prefix()
    labels, label_record = tail_labels(prefix)
    print(
        f"labels: {label_record['rows']} {prefix} rows ({label_record['distinct_keys']} distinct keys, "
        f"{len(label_record['repeated_rows'])} repeated rows) in the last {REAZON_TAIL_BYTES:,} bytes "
        f"of {label_record['file']}; first complete row from shard {label_record['first_complete_row_shard']}",
        flush=True,
    )
    keys = sorted(labels)
    if not 1 <= clips <= len(keys):
        raise SystemExit(f"--clips must lie in 1..{len(keys)}")
    chosen = {keys[index] for index in np.random.default_rng(SEED).choice(len(keys), size=clips, replace=False)}
    _reset(out_dir)
    name, size = REAZON_TEST_SHARD
    flac = 0
    labelled: list[str] = []
    sample_counts: dict[str, int] = {}
    with _get(REAZON_BASE + name) as response:
        length = response.headers.get("Content-Length")
        if response.status != 200 or length != str(size):
            raise SystemExit(f"{name}: the server answered {response.status} with {length} bytes, expected {size}")
        stream = _Counted(response)
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if not (member.isfile() and member.name.endswith(".flac")):
                    continue
                flac += 1
                if member.name not in labels:
                    continue
                labelled.append(member.name)
                if member.name in chosen:
                    handle = archive.extractfile(member)
                    assert handle is not None
                    audio = f"audio/{reazon_item_id(member.name)}.f32"
                    sample_counts[member.name] = write_audio(
                        out_dir / audio, decode_audio(io.BytesIO(handle.read()))
                    )
        stream.drain()
    if stream.bytes != size:
        raise SystemExit(f"{name}: streamed {stream.bytes} bytes, expected {size}")
    print(f"{name}: {stream.bytes:,} bytes, {flac} FLAC members, {len(labelled)} with a label", flush=True)
    if len(labelled) != len(set(labelled)) or set(labelled) != set(labels):
        raise SystemExit(
            f"{name}: {len(labels)} labelled keys against {len(labelled)} labelled FLAC members "
            f"({len(set(labels) - set(labelled))} keys without a member)"
        )
    items = [
        {
            "item_id": reazon_item_id(key),
            "audio": f"audio/{reazon_item_id(key)}.f32",
            "sample_count": sample_counts[key],
            "condition": {"key": key},
            "lines": [{
                "line_id": "c0",
                "start_s": 0.0,
                "end_s": sample_counts[key] / SAMPLE_RATE_HZ,
                "text": labels[key],
                "source": "subtitle",
                "overlap": False,
            }],
            "speech_free": [],
        }
        for key in sorted(chosen)
    ]
    written = write_set(
        out_dir,
        set_id=out_dir.name,
        kind="silver",
        created_at_utc=_now(),
        build={
            "builder": "scripts.asr_finetune.eval_sets reazon-test",
            "role": "held-out evaluation set of the fine-tuning pilot; never training data",
            "corpus": f"ReazonSpeech v2 all, shard {prefix[:-1]} ({REAZON_BASE})",
            "labels": label_record,
            "shard": {
                "file": name,
                "bytes": size,
                "sha256": stream.hexdigest(),
                "flac_members": flac,
                "flac_members_with_label": len(labelled),
            },
            "sample": {
                "seed": SEED,
                "clips": clips,
                "rule": "numpy.random.default_rng(seed).choice(len(keys), clips, replace=False) over the "
                        "sorted distinct labelled keys; items in key order",
            },
            "decoder": decoder,
        },
        items=items,
    )
    hours = sum(item.duration_s for item in written.items) / 3600
    print(f"wrote {len(written.items)} items ({hours:.3f} h) to {out_dir}")


# Subsets ------------------------------------------------------------------------


def build_subset(set_dir: Path, first: int, out_dir: Path) -> None:
    source = load_set(set_dir)
    if out_dir.resolve() == source.root.resolve():
        raise SystemExit("--out must name another folder than --set")
    if not 1 <= first <= len(source.items):
        raise SystemExit(f"--first must lie in 1..{len(source.items)}")
    manifest = (source.root / "set.json").read_bytes()
    items = json.loads(manifest)["items"][:first]
    _reset(out_dir)
    for item in items:
        target = out_dir / item["audio"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.root / item["audio"], target)
    build: dict[str, Any] = {
        "builder": "scripts.asr_finetune.eval_sets subset",
        "subset_of": source.set_id,
        "source_set_json_sha256": hashlib.sha256(manifest).hexdigest(),
        "first": first,
        "items": [item["item_id"] for item in items],
        "copied": "audio files and item entries unchanged",
    }
    if "attribution" in source.build:
        build["attribution"] = source.build["attribution"]
    written = write_set(
        out_dir, set_id=out_dir.name, kind=source.kind, created_at_utc=_now(), build=build, items=items,
    )
    lines = sum(len(item.lines) for item in written.items)
    print(f"wrote the first {len(written.items)} items of {source.set_id} ({lines} lines) to {out_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ingcrowd = commands.add_parser("ingcrowd", help="build ingcrowd-v1")
    ingcrowd.add_argument("--out", type=Path, required=True, help="set folder, e.g. <bench root>/sets/ingcrowd-v1")
    reazon = commands.add_parser("reazon-test", help="build reazon-test-v1")
    reazon.add_argument("--out", type=Path, required=True, help="set folder, e.g. <bench root>/sets/reazon-test-v1")
    reazon.add_argument("--clips", type=int, default=DEFAULT_CLIPS, help=f"clips drawn (default {DEFAULT_CLIPS})")
    subset = commands.add_parser("subset", help="copy the first N items of a set")
    subset.add_argument("--set", type=Path, required=True, help="source set folder")
    subset.add_argument("--first", type=int, required=True, help="items copied")
    subset.add_argument("--out", type=Path, required=True, help="new set folder")
    args = parser.parse_args(argv)
    if args.command == "ingcrowd":
        args.out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out, prefix=".downloads-") as downloads:
            build_ingcrowd(args.out, Path(downloads))
    elif args.command == "reazon-test":
        build_reazon_test(args.out, args.clips)
    else:
        build_subset(args.set, args.first, args.out)


if __name__ == "__main__":
    main()
