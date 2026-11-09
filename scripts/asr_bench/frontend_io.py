"""Move bench audio through an external frontend (speech separation,
enhancement) and back.

Usage::

    python -m scripts.asr_bench.frontend_io export --set <set dir> --out <wav dir>
    python -m scripts.asr_bench.frontend_io import --in <tool output dir>
        --set <set dir> --out <frontend dir> [--sample-rate HZ] [--pattern P]

``export`` writes every item as ``<item_id>.wav`` (16 kHz mono, IEEE float32,
so the round trip is lossless) for a tool to process offline. ``import``
reads the tool's output files (any format FFmpeg decodes), mixes them down to
mono, resamples them to 16 kHz with soxr when needed, checks each length is
within 1 ms of its item, pads or trims it to the item's exact sample count
and writes ``<item_id>.f32``: the ``audio_dir`` of a candidate ``frontend``.

Frontend audio is derived from bench audio: keep it under the bench root,
never in the repository.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soxr

from scripts.asr_bench.sets import SAMPLE_RATE_HZ, Item, load_set, write_audio

# Tool output may differ from its item by at most 1 ms before padding or trimming.
LENGTH_TOLERANCE_SAMPLES = SAMPLE_RATE_HZ // 1000
_ENCODE_FRAME_SECONDS = 10


@dataclass(frozen=True)
class Imported:
    item_id: str
    source: Path
    sample_rate: int
    channels: int
    # Resampled length minus the item's sample count, before pad or trim.
    length_delta_samples: int


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> None:
    """Write mono float32 samples as an IEEE float32 WAV at ``sample_rate``."""
    samples = np.ascontiguousarray(pcm, dtype=np.float32)
    frame_samples = _ENCODE_FRAME_SECONDS * sample_rate
    with av.open(str(path), "w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=sample_rate, layout="mono")
        for start in range(0, samples.size, frame_samples):
            frame = av.AudioFrame.from_ndarray(
                samples[start:start + frame_samples].reshape(1, -1),
                format="flt", layout="mono",
            )
            frame.sample_rate = sample_rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode the first audio stream as float32 ``(channels, samples)`` at its
    own rate; only the sample format is converted."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"{path} has no audio stream")
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="fltp")
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    if not chunks:
        raise ValueError(f"{path} holds no audio")
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False), int(rate)


def import_item(
    source: Path, item: Item, target: Path, sample_rate: int | None = None
) -> Imported:
    """Convert one tool output file into the item's ``.f32`` frontend file:
    mix down to mono, resample to 16 kHz with soxr (VHQ), check the length is
    within 1 ms of the item and the samples are finite, then pad or trim to
    the item's exact sample count. ``sample_rate``, when given, is the rate
    the file must declare."""
    audio, rate = read_audio(source)
    if sample_rate is not None and rate != sample_rate:
        raise ValueError(f"{source} is {rate} Hz, expected {sample_rate} Hz")
    mono = audio[0] if audio.shape[0] == 1 else audio.mean(axis=0, dtype=np.float64)
    if rate != SAMPLE_RATE_HZ:
        mono = soxr.resample(mono, rate, SAMPLE_RATE_HZ, quality="VHQ")
    mono = np.asarray(mono, dtype=np.float32)
    delta = int(mono.size) - item.sample_count
    if abs(delta) > LENGTH_TOLERANCE_SAMPLES:
        raise ValueError(
            f"{source}: {mono.size} samples at 16 kHz, item {item.item_id} has "
            f"{item.sample_count} ({delta / SAMPLE_RATE_HZ * 1000:+.1f} ms, more than 1 ms)"
        )
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"{source} holds non-finite samples")
    if delta < 0:
        mono = np.concatenate([mono, np.zeros(-delta, np.float32)])
    write_audio(target, mono[:item.sample_count])
    return Imported(item.item_id, source, rate, int(audio.shape[0]), delta)


def export_wav(set_dir: Path, out_dir: Path) -> list[Path]:
    """Write each item of the set as ``<item_id>.wav`` (16 kHz mono float32)."""
    bench_set = load_set(Path(set_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for item in bench_set.items:
        target = out_dir / f"{item.item_id}.wav"
        if target.exists():
            raise FileExistsError(f"{target} exists; choose an empty --out")
        write_wav(target, bench_set.load_audio(item))
        written.append(target)
    return written


def import_audio(
    in_dir: Path,
    set_dir: Path,
    out_dir: Path,
    sample_rate: int | None = None,
    pattern: str = "{item_id}.*",
) -> list[Imported]:
    """Convert a tool's per-item output into ``<item_id>.f32`` frontend files.

    ``pattern`` locates each item's file under ``in_dir`` (a glob with
    ``{item_id}``, e.g. ``htdemucs/{item_id}/vocals.wav``); exactly one file
    must match. ``sample_rate``, when given, is the rate the tool writes, and
    every file must declare it.
    """
    bench_set = load_set(Path(set_dir))
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    for item in bench_set.items:
        target = out_dir / f"{item.item_id}.f32"
        if target.exists():
            raise FileExistsError(f"{target} exists; choose an empty --out")
        matches = sorted(in_dir.glob(pattern.format(item_id=glob.escape(item.item_id))))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{item.item_id}: {len(matches)} files match {pattern!r} under {in_dir}"
            )
        imported.append(import_item(matches[0], item, target, sample_rate))
    return imported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write set items as 16 kHz mono float32 WAV")
    export.add_argument("--set", required=True, type=Path, help="bench set directory")
    export.add_argument("--out", required=True, type=Path, help="WAV output directory")
    imp = commands.add_parser("import", help="convert tool output into <item_id>.f32 files")
    imp.add_argument("--in", dest="in_dir", required=True, type=Path, help="tool output directory")
    imp.add_argument("--set", required=True, type=Path, help="bench set directory")
    imp.add_argument("--out", required=True, type=Path, help="frontend audio directory")
    imp.add_argument("--sample-rate", type=int, help="rate the tool writes (checked per file)")
    imp.add_argument("--pattern", default="{item_id}.*", help="glob per item, with {item_id}")
    args = parser.parse_args(argv)
    if args.command == "export":
        written = export_wav(args.set, args.out)
        print(f"wrote {len(written)} WAV files to {args.out}")
        return 0
    imported = import_audio(args.in_dir, args.set, args.out, args.sample_rate, args.pattern)
    rates = sorted({entry.sample_rate for entry in imported})
    channels = sorted({entry.channels for entry in imported})
    worst = max((abs(entry.length_delta_samples) for entry in imported), default=0)
    print(
        f"wrote {len(imported)} frontend files to {args.out} (input rates {rates} Hz, "
        f"channels {channels}, largest length difference {worst} samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
