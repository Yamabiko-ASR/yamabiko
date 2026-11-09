"""Build the Common Voice Japanese line-recall fixture.

Run once from the repository root; the outputs are committed:

    python tests/fixtures/audio/build_cv17_ja_lines.py [--cache-dir DIR]

Source: Mozilla Common Voice 17.0, Japanese test split, read from the
CC0-1.0 mirror ``fsicoli/common_voice_17_0`` at a pinned revision. The
dataset files are downloaded into ``--cache-dir`` (default: a directory
under the system temp dir), never into the repository or Yamabiko's own
model cache.

Selection, in order:

1. Rows with ``up_votes >= 2``, ``down_votes == 0``, an 8 to 30 character
   sentence and no ASCII letters or digits, sorted by ``path``, keeping
   the first row per ``client_id`` (distinct speakers).
2. Each candidate is decoded with ffmpeg to 16 kHz mono float32 and
   transcribed alone by the production ASR (``build_asr`` on the shipped
   default config, CUDA). It is accepted iff its character error rate
   after minimal Japanese normalization is at most 0.25. Selection stops
   at 8 accepted clips.

Assembly: 0.5 s of silence, the accepted clips separated by 1.0 s of
silence, then 1.5 s of silence. Outputs, next to this script:

* ``cv17_ja_8lines.s16le``: 16 kHz mono signed 16-bit little-endian PCM.
* ``cv17_ja_8lines.json``: provenance, license, attribution, sample rate
  and each line's sentence with its ``[start_sample, end_sample)`` span.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATASET_REPO = "fsicoli/common_voice_17_0"
DATASET_REVISION = "8262c16bf297c87a9cd88c51997c4758ed7a8ba2"
TSV_FILE = "transcript/ja/test.tsv"
TAR_FILE = "audio/ja/test/ja_test_0.tar"
TAR_SIZE = 199_857_664
LICENSE = "CC0-1.0"
ATTRIBUTION = (
    "Mozilla Common Voice 17.0, Japanese test split "
    "(via fsicoli/common_voice_17_0)"
)

SAMPLE_RATE = 16000
TARGET_LINES = 8
MIN_LINES = 5
MAX_CER = 0.25
LEAD_SILENCE_S = 0.5
GAP_SILENCE_S = 1.0
TAIL_SILENCE_S = 1.5
OUTPUT_STEM = "cv17_ja_8lines"

_ASCII_ALNUM = frozenset(string.ascii_letters + string.digits)


def _download(filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    ))


def select_candidates(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(
            handle, delimiter="\t", quoting=csv.QUOTE_NONE,
        ))
    eligible = [
        row for row in rows
        if int(row["up_votes"]) >= 2
        and int(row["down_votes"]) == 0
        and 8 <= len(row["sentence"]) <= 30
        and not _ASCII_ALNUM.intersection(row["sentence"])
    ]
    eligible.sort(key=lambda row: row["path"])
    seen_clients: set[str] = set()
    candidates: list[dict[str, str]] = []
    for row in eligible:
        if row["client_id"] in seen_clients:
            continue
        seen_clients.add(row["client_id"])
        candidates.append(row)
    return candidates


def decode_mp3(data: bytes) -> np.ndarray:
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "-",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        input=data, capture_output=True, check=True,
    )
    return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)


def character_error_rate(reference: str, hypothesis: str) -> float:
    from scripts.e0_evidence.align import character_trace
    from scripts.e0_evidence.normalize_ja import normalize_minimal

    ref = normalize_minimal(reference)
    hyp = normalize_minimal(hypothesis)
    trace = character_trace(ref, hyp)
    errors = trace.substitutions + trace.deletions + trace.insertions
    return errors / len(ref)


def _default_asr():
    """Production ASR from the shipped default config, read without writes."""
    import yaml

    from yamabiko import hf_env  # noqa: F401  # pin the model cache first
    from yamabiko import paths
    from yamabiko.asr.asr_client import build_asr

    cfg = yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    return build_asr(cfg["asr"])


def build(cache_dir: Path, out_dir: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not on PATH")
    version = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=True,
    ).stdout.splitlines()[0]
    print(f"ffmpeg: {version}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = _download(TSV_FILE, cache_dir)
    tar_path = _download(TAR_FILE, cache_dir)
    tar_size = tar_path.stat().st_size
    if tar_size != TAR_SIZE:
        raise SystemExit(
            f"{TAR_FILE} is {tar_size} bytes, expected {TAR_SIZE}"
        )

    candidates = select_candidates(tsv_path)
    print(f"candidates: {len(candidates)}")
    asr = _default_asr()

    accepted: list[tuple[dict[str, str], np.ndarray]] = []
    with tarfile.open(tar_path) as archive:
        members = {
            Path(member.name).name: member
            for member in archive.getmembers() if member.isfile()
        }
        for row in candidates:
            if len(accepted) == TARGET_LINES:
                break
            member = members.get(row["path"])
            if member is None:
                print(f"missing   {row['path']}: not in {TAR_FILE}")
                continue
            audio = decode_mp3(archive.extractfile(member).read())
            text = asr.transcribe(audio, SAMPLE_RATE).text
            cer = character_error_rate(row["sentence"], text)
            verdict = "accepted" if cer <= MAX_CER else "rejected"
            print(
                f"{verdict:9} {row['path']} cer={cer:.3f} "
                f"ref={row['sentence']} hyp={text}"
            )
            if cer <= MAX_CER:
                accepted.append((row, audio))

    if len(accepted) < MIN_LINES:
        raise SystemExit(
            f"only {len(accepted)} clips passed the CER screen; "
            f"at least {MIN_LINES} are required"
        )

    pieces = [np.zeros(round(LEAD_SILENCE_S * SAMPLE_RATE), np.float32)]
    cursor = pieces[0].size
    lines = []
    for index, (row, audio) in enumerate(accepted):
        if index:
            gap = np.zeros(round(GAP_SILENCE_S * SAMPLE_RATE), np.float32)
            pieces.append(gap)
            cursor += gap.size
        pieces.append(audio)
        lines.append({
            "clip": row["path"],
            "sentence": row["sentence"],
            "start_sample": cursor,
            "end_sample": cursor + audio.size,
        })
        cursor += audio.size
    pieces.append(np.zeros(round(TAIL_SILENCE_S * SAMPLE_RATE), np.float32))
    stream = np.concatenate(pieces)
    pcm16 = np.rint(np.clip(stream, -1.0, 1.0) * 32767.0).astype("<i2")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{OUTPUT_STEM}.s16le").write_bytes(pcm16.tobytes())
    metadata = {
        "source": f"https://huggingface.co/datasets/{DATASET_REPO}",
        "revision": DATASET_REVISION,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "sample_rate": SAMPLE_RATE,
        "lines": lines,
    }
    if len(accepted) < TARGET_LINES:
        metadata["line_count"] = len(accepted)
    with (out_dir / f"{OUTPUT_STEM}.json").open(
        "w", encoding="utf-8", newline="\n",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"wrote {len(accepted)} lines, {stream.size} samples "
        f"({stream.size / SAMPLE_RATE:.2f} s) to {out_dir}"
    )


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "yamabiko-cv17-ja",
        help="download cache for the dataset files (outside the repository)",
    )
    args = parser.parse_args(argv)
    build(args.cache_dir, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
