"""MUSAN non-speech audio for the pilot's no-speech windows (round 4).

    python -m scripts.asr_finetune.musan [--archive <musan.tar.gz>] [--out <inventory.json>]

MUSAN (OpenSLR SLR17, https://www.openslr.org/17/; Snyder, Chen and Povey,
arXiv:1510.08484) is licensed CC BY 4.0 as a corpus; its music and noise
sources carry their own terms, listed per clip in each subcorpus's LICENSE.
The pods fetch the original ``musan.tar.gz`` (11,086,114,085 bytes) from its
Hugging Face mirror ``pilot.MUSAN_MIRROR`` at a pinned revision (source
``musan`` of ``pilot.SOURCES``, SHA-256 pinned). Two mirrors
(``huseinzol05/musan-mirror``, ``EaseZh/musan``) give the same LFS SHA-256,
and OpenSLR serves the same byte count (checked 2026-10-28).

Only non-speech audio is used: every ``musan/noise/**.wav``, and the
``musan/music/<source>/*.wav`` files that the source's ``ANNOTATIONS`` marks
without vocals (third column ``N``). The ``speech`` subset, music with
vocals, and music without an annotation line are never used.

This command builds the local inventory ``mix --nonspeech`` plans from. It
streams the archive (``--archive``, else the mirror's resolve URL, about
7 minutes at 27 MB/s, nothing written but the inventory), verifies its byte
count and SHA-256, and writes ``--out`` (default
``<pilot>\\musan\\inventory.json``, private)::

    {"schema": "ft-pilot-musan-inventory-1", "archive": {"bytes", "sha256"},
     "files": [{"path": "musan/noise/free-sound/noise-free-sound-0000.wav", "kind": "noise", "seconds": 8.2}, ...],
     "excluded": {"speech": N, "music_vocals": N, "music_unannotated": N}}

``extract`` is the pod side: one pass over the fetched archive that writes the
members a window plan names below a folder.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
import wave
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from scripts.asr_finetune.pilot import MUSAN_BYTES, MUSAN_MIRROR, MUSAN_SHA256, pilot_root

INVENTORY_SCHEMA = "ft-pilot-musan-inventory-1"
MUSAN_FILE = "musan.tar.gz"
_BUFFER = 4 << 20


class _Hashing(io.RawIOBase):
    """A readable stream that hashes and counts what passes through it."""

    def __init__(self, source: BinaryIO) -> None:
        self.source, self.digest, self.bytes = source, hashlib.sha256(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self.source.read(len(buffer))
        self.digest.update(data)
        self.bytes += len(data)
        buffer[:len(data)] = data
        return len(data)

    def drain(self) -> None:
        while self.read(_BUFFER):
            pass


def kind(path: str) -> str | None:
    """``noise``, ``music`` or ``speech`` for a ``.wav`` member, else None."""
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[0] != "musan" or not path.endswith(".wav"):
        return None
    return parts[1] if parts[1] in ("noise", "music", "speech") else None


def music_vocals(text: str, folder: str) -> dict[str, bool]:
    """A music ``ANNOTATIONS`` file (``<name> <genre> <Y|N> ...`` per line):
    member path -> whether the clip has vocals."""
    vocals: dict[str, bool] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2] in ("Y", "N"):
            vocals[f"{folder}/{fields[0]}.wav"] = fields[2] == "Y"
    return vocals


def scan(stream: BinaryIO) -> dict[str, Any]:
    """The inventory of a MUSAN archive read from ``stream`` (see the module docstring)."""
    hashing = _Hashing(stream)
    seconds: dict[str, tuple[str, float]] = {}
    vocals: dict[str, bool] = {}
    speech = 0
    with tarfile.open(fileobj=io.BufferedReader(hashing, _BUFFER), mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.removeprefix("./")
            source = archive.extractfile(member)
            assert source is not None
            if PurePosixPath(name).name == "ANNOTATIONS" and PurePosixPath(name).parts[1:2] == ("music",):
                vocals.update(music_vocals(source.read().decode("utf-8", "replace"), str(PurePosixPath(name).parent)))
                continue
            what = kind(name)
            if what == "speech":
                speech += 1
            elif what is not None:
                with wave.open(io.BytesIO(source.read())) as audio:
                    seconds[name] = (what, round(audio.getnframes() / audio.getframerate(), 3))
    hashing.drain()
    files, excluded = [], {"speech": speech, "music_vocals": 0, "music_unannotated": 0}
    for path, (what, length) in sorted(seconds.items()):
        if what == "music" and path not in vocals:
            excluded["music_unannotated"] += 1
        elif what == "music" and vocals[path]:
            excluded["music_vocals"] += 1
        else:
            files.append({"path": path, "kind": what, "seconds": length})
    return {"schema": INVENTORY_SCHEMA, "archive": {"bytes": hashing.bytes, "sha256": hashing.digest.hexdigest()},
            "files": files, "excluded": excluded}


def extract(archive: Path, paths: Iterable[str], folder: Path) -> dict[str, Path]:
    """Write the ``paths`` members of ``archive`` below ``folder`` in one
    pass: path -> the written file (the paths the archive lacks are absent)."""
    wanted, written = set(paths), {}
    with tarfile.open(archive, "r|gz", bufsize=_BUFFER) as source:
        for member in source:
            name = member.name.removeprefix("./")
            if not member.isfile() or name not in wanted:
                continue
            target = folder / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.extractfile(member)
            assert data is not None
            target.write_bytes(data.read())
            written[name] = target
            if len(written) == len(wanted):
                break
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.musan", description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, help="a local musan.tar.gz (default: stream the pinned mirror)")
    parser.add_argument("--out", type=Path, default=pilot_root() / "musan" / "inventory.json",
                        help="the inventory (default <pilot>\\musan\\inventory.json)")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.archive is not None:
        with args.archive.open("rb") as stream:
            inventory = scan(stream)
    else:
        import urllib.request

        repo, revision = MUSAN_MIRROR
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{MUSAN_FILE}"
        with urllib.request.urlopen(url, timeout=120) as stream:
            inventory = scan(stream)
    archive = inventory["archive"]
    if archive != {"bytes": MUSAN_BYTES, "sha256": MUSAN_SHA256}:
        print(f"error: the archive has {archive['bytes']:,} bytes and SHA-256 {archive['sha256']}, "
              f"not the pinned {MUSAN_BYTES:,} and {MUSAN_SHA256}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=1) + "\n", encoding="utf-8")
    hours = {what: round(sum(f["seconds"] for f in inventory["files"] if f["kind"] == what) / 3600, 2)
             for what in ("noise", "music")}
    print(f"musan: {len(inventory['files']):,} files ({hours['noise']} h noise, {hours['music']} h music without "
          f"vocals); excluded {inventory['excluded']}; archive verified; {time.monotonic() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
