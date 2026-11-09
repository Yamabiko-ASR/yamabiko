"""Stage the pilot's downloaded training sources into rows (pod task S1).

    python -m scripts.asr_finetune.stage --plan <stage-plan.json> --data <dir> --out <dir>

Reads what ``scripts.asr_finetune.fetch`` put in ``--data`` for the plan's
sources (same plan file) and writes JSON Lines to ``--out``:

- ``reazon-rows.jsonl``: ``{"key": "000/xxxx.flac", "caption", "duration_s"}``
  for every FLAC member of the plan's shards (``<data>\\<shard>.tar``, from
  the corpus server or the mirror) that the plan's caption file labels
  (``small.tsv``, or the mirror's ``all.tsv``, read for those shards only),
  as the official loader pairs them (a key the TSV repeats takes its last
  caption), in key order. The duration is the STREAMINFO sample count over
  the sample rate (PyAV when a header lacks it). The file is also copied to
  ``<data>\\reazon-rows.jsonl`` for the teacher task: a task's ``{out}`` is
  fetched and removed.
- ``cv-pool.jsonl``: one pass over the Common Voice ``.tar.gz`` reading
  ``validated.tsv``, ``dev.tsv``, ``test.tsv`` and ``clip_durations.tsv``;
  every ``validated`` row whose clip is in neither v27 split nor hit by
  ``cv-exclude.json`` (the outbox file beside the plan: clip name,
  ``client_id`` prefix, ``pilot.sentence_hash`` of the sentence), as
  ``{"path", "sentence", "client_prefix", "up_votes", "down_votes",
  "duration_s"}``.
- ``galgame-rows.jsonl``: each archive of the plan extracted with ``7z x``
  (``pilot.OOPPEENN_PASSWORD_B64``) into ``<data>\\galgame\\<archive name>\\``;
  each ``index.json`` in it read: ``Text``, and the audio from ``FilePath``
  (its longest tail that names a file below the index) or from ``Speaker`` +
  ``Voice`` (``<Speaker>/<Voice>``, with ``.ogg``, ``.opus`` or ``.wav`` when
  ``Voice`` has no suffix). Entries whose speaker is ``？？？`` are dropped, as
  the card of the set's earlier repo (``OOPPEENN/Galgame_Dataset``) asks, and
  so are entries without text or audio and repeats of an audio file. Rows
  ``{"archive": "GalGame/<maker>_<game>.7z", "voice": <audio path below the
  extraction folder>, "text", "duration_s"}``; Ogg durations from the last
  page's granule position, others with PyAV. ``stage.json`` records per
  archive the drop counts, how many index entries carry each set of key
  names and the suffixes of the extracted files. A 7z exit of 1 (a warning)
  is counted and the archive read; an archive 7z fails on (exit 2 or more)
  or that cannot be read is reported and skipped.
- ``jvnv-rows.jsonl``: the regular-session utterances of ``jvnv_ver1.zip``
  (``jvnv_v1/<spk>/<emotion>/regular/<id>.wav``, 1,375), each with its line of
  ``jvnv_v1/transcription.csv`` (``id|phrase|text``, ids without the speaker):
  ``{"id", "path", "phrase", "text", "duration_s"}``.
- ``jnv-rows.jsonl``: the regular-session clips of ``jnv_corpus_ver3.zip``
  (``JNV/<spk>/<spk>_<emotion>_<nn>_R.wav``, 347), each with its line of
  ``JNV/phrases.txt`` (``<emotion>_<nn>_R|phrase``): ``{"file", "phrase",
  "duration_s"}``.
- ``stage.json``: rows, audio seconds, drops and wall seconds per source.

Each source is staged on its own: one that fails gets ``{"error": ...}``
in ``stage.json`` (the message of a shape error, else the exception type
only, as a message may hold a path naming a title), is listed in
``failed_sources``, and the others go on; the exit stays 0, so the pod run
fetches the rows and continues. Audio inside the zip files and the Common
Voice archive stays packed. The console shows counts only; archives are
named ``galgame-NN`` (plan order). Exit 1 only when a JNV regular-session
clip has no phrase (``stage.json`` written first).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
import wave
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from scripts.asr_finetune.fetch import PlanError, galgame_name, load_plan
from scripts.asr_finetune.pilot import (
    NORMALIZATION,
    OOPPEENN_PASSWORD_B64,
    REAZON_CAPTION_SOURCES,
    SOURCES,
    reazon_shard,
    sentence_hash,
)

OUT_SCHEMA = "ft-pilot-stage-1"
EXCLUDE_SCHEMA = "ft-pilot-cv-exclude-1"
EXCLUDE_FILE = "cv-exclude.json"
CV_TSVS = ("validated.tsv", "dev.tsv", "test.tsv", "clip_durations.tsv")
AUDIO_SUFFIXES = (".ogg", ".opus", ".wav")
UNKNOWN_SPEAKER = "???"  # NFKC of ？？？
JVNV_REGULAR = re.compile(r"jvnv_v1/(?P<speaker>[FM]\d)/(?P<emotion>[a-z]+)/regular/"
                          r"(?P=speaker)_(?P=emotion)_regular_\d+\.wav")
JNV_REGULAR = re.compile(r"JNV/(?P<speaker>[FM]\d)/(?P=speaker)_(?P<key>[a-z]+_\d+_R)\.wav")
_TAR_BUFFER = 4 << 20


class StageError(RuntimeError):
    """An input that does not have the expected shape."""


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:,.2f} h"


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# --- durations ------------------------------------------------------------


def flac_duration(head: bytes) -> float | None:
    """Seconds from a FLAC file's first bytes (STREAMINFO), or None."""
    if len(head) < 26 or head[:4] != b"fLaC" or head[4] & 0x7F != 0:
        return None
    value = int.from_bytes(head[18:26], "big")
    rate, samples = value >> 44, value & ((1 << 36) - 1)
    return samples / rate if rate and samples else None


def ogg_duration(path: Path) -> float | None:
    """Seconds of an Ogg Vorbis or Opus file from its first page (codec,
    rate, Opus pre-skip) and the granule position of its stream's last page."""
    with path.open("rb") as source:
        head = source.read(4096)
        if len(head) < 28 or head[:4] != b"OggS":
            return None
        segments = head[26]
        start = 27 + segments
        packet = head[start:start + sum(head[27:start])]
        if packet[:7] == b"\x01vorbis" and len(packet) >= 16:
            rate, skip = int.from_bytes(packet[12:16], "little"), 0
        elif packet[:8] == b"OpusHead" and len(packet) >= 12:
            rate, skip = 48_000, int.from_bytes(packet[10:12], "little")
        else:
            return None
        size = source.seek(0, os.SEEK_END)
        source.seek(max(0, size - 65_536))
        tail = source.read()
    serial = head[14:18]
    end = len(tail)
    while (index := tail.rfind(b"OggS", 0, end)) >= 0:
        if len(tail) - index >= 27 and tail[index + 14:index + 18] == serial:
            granule = int.from_bytes(tail[index + 6:index + 14], "little", signed=True)
            if granule >= 0:
                return max(0, granule - skip) / rate if rate else None
        end = index
    return None


def wave_duration(source: BinaryIO | str) -> float | None:
    try:
        with wave.open(source) as reader:
            return reader.getnframes() / reader.getframerate()
    except (wave.Error, EOFError):
        return None


def av_duration(source: BinaryIO | str) -> float:
    import av

    with av.open(source) as container:
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return container.duration / 1_000_000
        return sum(frame.samples for frame in container.decode(stream)) / stream.rate


def file_duration(path: Path) -> float | None:
    """Seconds of an audio file, or None when it cannot be read."""
    try:
        suffix = path.suffix.lower()
        duration = (ogg_duration(path) if suffix in (".ogg", ".opus")
                    else wave_duration(str(path)) if suffix == ".wav" else None)
        return duration if duration is not None else av_duration(str(path))
    except Exception:  # a broken file is counted, not fatal
        return None


# --- ReazonSpeech -----------------------------------------------------------


def reazon_labels(tsv: Path, shards: Sequence[str]) -> tuple[dict[str, dict[str, str]], Counter]:
    """The captions of ``tsv`` (``small.tsv``, or the mirror's ``all.tsv``,
    the same ``key<TAB>caption`` lines for every shard) for ``shards``:
    shard -> key -> caption. A repeated key takes its last caption, as the
    official loader's mapping does; ``repeated_keys`` and ``recaptioned_keys``
    (another caption) count them. Lines of other shards are skipped unread."""
    labels: dict[str, dict[str, str]] = {shard: {} for shard in shards}
    prefixes = {f"{shard}/" for shard in shards}
    width = len(next(iter(prefixes), ""))
    counts: Counter = Counter(repeated_keys=0, recaptioned_keys=0)
    with tsv.open(encoding="utf-8", newline="\n") as handle:
        for number, line in enumerate(handle, 1):
            if line[:width] not in prefixes:
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise StageError(f"{tsv.name} line {number} has {len(fields)} fields, not 2")
            key, caption = fields
            known = labels[key.split("/", 1)[0]]
            if key in known:
                counts["repeated_keys"] += 1
                counts["recaptioned_keys"] += known[key] != caption
            known[key] = caption
    return labels, counts


def stage_reazon(data: Path, shards: Sequence[str], captions: str, out: Path) -> dict[str, Any]:
    """``reazon-rows.jsonl`` for ``shards`` (``<data>\\<shard>.tar``) with the
    captions of ``<data>\\<captions>``."""
    labels, counts = reazon_labels(data / captions, shards)
    rows: list[dict[str, Any]] = []
    per_shard: dict[str, Any] = {}
    for shard in shards:
        wanted = labels[shard]
        found: list[dict[str, Any]] = []
        with tarfile.open(data / f"{shard}.tar") as archive:
            for member in archive:
                key = member.name.removeprefix("./")
                if not member.isfile() or key not in wanted:
                    continue
                source = archive.extractfile(member)
                assert source is not None
                duration = flac_duration(source.read(64))
                if duration is None:
                    source.seek(0)
                    duration = av_duration(io.BytesIO(source.read()))
                found.append({"key": key, "caption": wanted[key], "duration_s": round(duration, 4)})
        found.sort(key=lambda row: row["key"])
        seconds = sum(row["duration_s"] for row in found)
        per_shard[shard] = {"labelled": len(wanted), "rows": len(found), "missing_audio": len(wanted) - len(found),
                            "audio_s": round(seconds, 1)}
        print(f"reazon {shard}: {len(found):,} rows, {_hours(seconds)} "
              f"({len(wanted) - len(found):,} labelled keys without audio)", flush=True)
        rows += found
    _write_rows(out / "reazon-rows.jsonl", rows)
    shutil.copyfile(out / "reazon-rows.jsonl", data / "reazon-rows.jsonl")
    seconds = sum(row["duration_s"] for row in rows)
    return {"rows": len(rows), "audio_s": round(seconds, 1), "captions": captions, **counts, "shards": per_shard}


# --- Common Voice -------------------------------------------------------------


def _load_exclude(path: Path) -> tuple[frozenset[str], frozenset[str], int, frozenset[str]]:
    """``cv-exclude.json``: sentence hashes, client_id prefixes (and their
    length), clip names."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != EXCLUDE_SCHEMA or raw.get("normalization") != NORMALIZATION:
        raise StageError(f"{path.name} is not a {EXCLUDE_SCHEMA} list in {NORMALIZATION}")
    prefixes = frozenset(raw["client_id_prefixes"])
    widths = {len(prefix) for prefix in prefixes}
    if len(widths) != 1:
        raise StageError(f"{path.name}: the client_id prefixes must share one length")
    return frozenset(raw["sentence_sha256"]), prefixes, widths.pop(), frozenset(raw["clips"])


def _tsv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter="\t", quoting=csv.QUOTE_NONE))


def stage_cv(data: Path, exclude_path: Path, out: Path) -> dict[str, Any]:
    """``_stage_cv``; a missing TSV column (or exclusion-list key) becomes a
    ``StageError`` that names it, so ``stage.json`` shows which."""
    try:
        return _stage_cv(data, exclude_path, out)
    except KeyError as error:
        raise StageError(f"Common Voice: no {error.args[0]!r} column or key") from None


def _stage_cv(data: Path, exclude_path: Path, out: Path) -> dict[str, Any]:
    sentences, prefixes, width, clips = _load_exclude(exclude_path)
    texts: dict[str, str] = {}
    with tarfile.open(data / SOURCES["cv"].file, "r|gz", bufsize=_TAR_BUFFER) as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if not member.isfile() or name not in CV_TSVS:
                continue
            if name in texts:
                raise StageError(f"the Common Voice archive holds more than one {name}")
            source = archive.extractfile(member)
            assert source is not None
            texts[name] = source.read().decode("utf-8")
            if len(texts) == len(CV_TSVS):
                break
    missing = [name for name in CV_TSVS if name not in texts]
    if missing:
        raise StageError(f"the Common Voice archive lacks {', '.join(missing)}")
    splits = {row["path"] for name in ("dev.tsv", "test.tsv") for row in _tsv_rows(texts[name])}
    durations = {row["clip"]: int(row["duration[ms]"]) for row in _tsv_rows(texts["clip_durations.tsv"])}
    validated = _tsv_rows(texts["validated.tsv"])
    drops: Counter = Counter()
    rows: list[dict[str, Any]] = []
    for row in validated:
        path, client = row["path"], row["client_id"]
        reason = ("split" if path in splits else "clip" if path in clips
                  else "speaker" if client[:width] in prefixes
                  else "sentence" if sentence_hash(row["sentence"]) in sentences
                  else "no_duration" if path not in durations else None)
        if reason:
            drops[reason] += 1
            continue
        rows.append({"path": path, "sentence": row["sentence"], "client_prefix": client[:width],
                     "up_votes": int(row["up_votes"] or 0), "down_votes": int(row["down_votes"] or 0),
                     "duration_s": durations[path] / 1000})
    _write_rows(out / "cv-pool.jsonl", rows)
    seconds = sum(row["duration_s"] for row in rows)
    print(f"cv: {len(rows):,} of {len(validated):,} validated rows, {_hours(seconds)}; dropped "
          + (", ".join(f"{reason} {count:,}" for reason, count in sorted(drops.items())) or "none"), flush=True)
    return {"validated": len(validated), "dev_test_clips": len(splits), "rows": len(rows),
            "audio_s": round(seconds, 1), "dropped": dict(sorted(drops.items()))}


# --- OOPPEENN visual-novel archives ------------------------------------------------


def _inside(folder: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def galgame_audio(folder: Path, base: Path, entry: dict[str, Any]) -> Path | None:
    """The audio file of an ``index.json`` entry (``base`` is the index's
    folder), inside ``folder``, or None."""
    file_path = entry.get("FilePath")
    if isinstance(file_path, str) and file_path.strip():
        parts = [part for part in file_path.replace("\\", "/").split("/") if part not in ("", ".")]
        for start in range(len(parts)):
            if ".." in parts[start:]:
                continue
            candidate = base.joinpath(*parts[start:])
            if _inside(folder, candidate) and candidate.is_file():
                return candidate
        return None
    speaker, voice = entry.get("Speaker"), entry.get("Voice")
    if not (isinstance(speaker, str) and isinstance(voice, str) and speaker and voice):
        return None
    stem = base / speaker / voice
    names = [stem] if stem.suffix.lower() in AUDIO_SUFFIXES else [stem.with_name(stem.name + s) for s in AUDIO_SUFFIXES]
    return next((path for path in names if _inside(folder, path) and path.is_file()), None)


def galgame_lines(folder: Path) -> tuple[list[tuple[str, str, Path]], Counter, Counter]:
    """(voice, text, audio path) of every usable entry of the ``index.json``
    files below an extracted archive, in index order; the drop counts; and
    how many entries carry each set of key names (names only, no values)."""
    counts: Counter = Counter()
    keys: Counter = Counter()
    lines: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for index_path in sorted(folder.rglob("index.json")):
        counts["index_files"] += 1
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, ValueError):
            entries = None
        if not isinstance(entries, list):
            counts["bad_index"] += 1
            keys[f"<{type(entries).__name__}>"] += 1
            continue
        for entry in entries:
            keys[",".join(sorted(map(str, entry))) if isinstance(entry, dict) else f"<{type(entry).__name__}>"] += 1
            if not isinstance(entry, dict) or not isinstance(entry.get("Text"), str):
                counts["no_text"] += 1
                continue
            audio = galgame_audio(folder, index_path.parent, entry)
            if audio is None:
                counts["missing_audio"] += 1
                continue
            speaker = entry.get("Speaker")
            speaker = speaker if isinstance(speaker, str) else audio.parent.name
            if unicodedata.normalize("NFKC", speaker).strip() == UNKNOWN_SPEAKER:
                counts["unknown_speaker"] += 1
                continue
            voice = audio.relative_to(folder).as_posix()
            if voice in seen:
                counts["repeated_audio"] += 1
                continue
            seen.add(voice)
            lines.append((voice, entry["Text"], audio))
    return lines, counts, keys


def _extract(seven: str, password: str, archive: Path, folder: Path) -> int:
    if folder.exists():
        shutil.rmtree(folder)
    return subprocess.run([seven, "x", "-y", f"-p{password}", f"-o{folder}", str(archive)],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def _galgame_archive(repo_path: str, folder: Path, pool: ThreadPoolExecutor) -> tuple[list[dict[str, Any]], dict]:
    """The rows of one extracted archive and its record: drop counts, the
    key-name sets of its index entries and its extracted file suffixes."""
    lines, counts, keys = galgame_lines(folder)
    durations = list(pool.map(file_duration, (audio for _, _, audio in lines)))
    found = [{"archive": repo_path, "voice": voice, "text": text, "duration_s": round(duration, 4)}
             for (voice, text, _), duration in zip(lines, durations) if duration is not None]
    counts["bad_audio"] = len(lines) - len(found)
    suffixes = Counter(path.suffix.lower() for path in folder.rglob("*") if path.is_file())
    return found, {"rows": len(found), "audio_s": round(sum(row["duration_s"] for row in found), 1),
                   **dict(sorted(counts.items())), "index_keys": dict(keys.most_common()),
                   "suffixes": dict(suffixes.most_common())}


def stage_galgame(data: Path, archives: Sequence[tuple[str, int]], out: Path) -> dict[str, Any]:
    seven = shutil.which("7z")
    if seven is None:
        raise StageError("7z is not on PATH")
    password = base64.b64decode(OOPPEENN_PASSWORD_B64).decode()
    folders = [data / "galgame" / PurePosixPath(repo_path).stem for repo_path, _ in archives]
    threads = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        codes = list(pool.map(lambda pair: _extract(seven, password, data / pair[0][0], pair[1]),
                              zip(archives, folders)))
        per_archive: dict[str, Any] = {}
        total: Counter = Counter()
        rows: list[dict[str, Any]] = []
        for index, ((repo_path, _), folder, code) in enumerate(zip(archives, folders, codes)):
            name = galgame_name(index)
            # 7-Zip: 1 is a warning (non-fatal), 2 and up an error.
            if code not in (0, 1):
                per_archive[name] = {"extract_exit": code}
                total["failed_archives"] += 1
                print(f"WARNING {name}: 7z exited with {code}; skipped", flush=True)
                continue
            try:
                found, record = _galgame_archive(repo_path, folder, pool)
            except Exception as error:  # the type only: a message may hold a path naming a title
                per_archive[name] = {"extract_exit": code, "error": type(error).__name__}
                total["failed_archives"] += 1
                print(f"WARNING {name}: {type(error).__name__} while reading it; skipped", flush=True)
                continue
            if code:
                total["extract_warnings"] += 1
            per_archive[name] = {"extract_exit": code, **record}
            total.update({key: value for key, value in record.items() if isinstance(value, int) and key != "rows"})
            rows += found
            print(f"{name}: {len(found):,} rows, {_hours(record['audio_s'])}"
                  + (", 7z warning (exit 1)" if code else "") + "; "
                  + ", ".join(f"{key} {value:,}" for key, value in record.items()
                              if isinstance(value, int) and key != "rows" and value), flush=True)
    _write_rows(out / "galgame-rows.jsonl", rows)
    seconds = sum(row["duration_s"] for row in rows)
    print(f"galgame: {len(rows):,} rows from {len(archives) - total['failed_archives']} of {len(archives)} "
          f"archives, {_hours(seconds)}", flush=True)
    return {"rows": len(rows), "audio_s": round(seconds, 1), **dict(sorted(total.items())), "archives": per_archive}


# --- JVNV and JNV -------------------------------------------------------------


def _member_duration(archive: zipfile.ZipFile, name: str) -> float:
    with archive.open(name) as member:
        duration = wave_duration(member)
    if duration is None:
        duration = av_duration(io.BytesIO(archive.read(name)))
    return round(duration, 4)


def jvnv_script(archive: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    """``jvnv_v1/transcription.csv`` (``id|phrase|text``, ids without the
    speaker): line id -> (phrase, text)."""
    script: dict[str, tuple[str, str]] = {}
    for number, line in enumerate(archive.read("jvnv_v1/transcription.csv").decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 3:
            raise StageError(f"transcription.csv line {number} has {len(fields)} fields, not 3")
        script[fields[0]] = (fields[1], fields[2])
    return script


def jnv_phrases(archive: zipfile.ZipFile) -> dict[str, str]:
    """``JNV/phrases.txt`` (``<emotion>_<nn>_R|phrase``): key -> phrase."""
    phrases: dict[str, str] = {}
    for number, line in enumerate(archive.read("JNV/phrases.txt").decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 2:
            raise StageError(f"phrases.txt line {number} has {len(fields)} fields, not 2")
        phrases[fields[0]] = fields[1]
    return phrases


def stage_jvnv(data: Path, out: Path) -> dict[str, Any]:
    with zipfile.ZipFile(data / SOURCES["jvnv"].file) as archive:
        script = jvnv_script(archive)
        rows: list[dict[str, Any]] = []
        unscripted = 0
        for name in sorted(name for name in archive.namelist() if JVNV_REGULAR.fullmatch(name)):
            utterance = PurePosixPath(name).stem
            line_id = utterance.split("_", 1)[1]
            if line_id not in script:
                unscripted += 1
                continue
            phrase, text = script[line_id]
            rows.append({"id": utterance, "path": name, "phrase": phrase, "text": text,
                         "duration_s": _member_duration(archive, name)})
    _write_rows(out / "jvnv-rows.jsonl", rows)
    seconds = sum(row["duration_s"] for row in rows)
    print(f"jvnv: {len(rows):,} regular utterances, {_hours(seconds)}; {unscripted} without a script line", flush=True)
    return {"rows": len(rows), "audio_s": round(seconds, 1), "script_lines": len(script), "unscripted": unscripted}


def stage_jnv(data: Path, out: Path) -> dict[str, Any]:
    with zipfile.ZipFile(data / SOURCES["jnv"].file) as archive:
        phrases = jnv_phrases(archive)
        rows: list[dict[str, Any]] = []
        unphrased = 0
        for name in sorted(name for name in archive.namelist() if name.endswith("_R.wav")):
            match = JNV_REGULAR.fullmatch(name)
            phrase = phrases.get(match["key"]) if match else None
            if phrase is None:
                unphrased += 1
                continue
            rows.append({"file": name, "phrase": phrase, "duration_s": _member_duration(archive, name)})
    _write_rows(out / "jnv-rows.jsonl", rows)
    seconds = sum(row["duration_s"] for row in rows)
    print(f"jnv: {len(rows):,} regular clips, {seconds:,.1f} s; {unphrased} without a phrase", flush=True)
    return {"rows": len(rows), "audio_s": round(seconds, 1), "phrases": len(phrases), "unphrased": unphrased}


# --- main ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.stage", description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True, help="stage-plan.json (cv-exclude.json beside it)")
    parser.add_argument("--data", type=Path, required=True, help="the fetch task's data folder")
    parser.add_argument("--out", type=Path, required=True, help="folder for the rows and stage.json")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
    except PlanError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2
    names = [source.name for source in plan.sources]
    shards = [shard for shard in map(reazon_shard, names) if shard]
    captions = [SOURCES[name].file for name in names if name in REAZON_CAPTION_SOURCES]
    if shards and not captions:
        print(f"error: the plan's ReazonSpeech shards need {' or '.join(REAZON_CAPTION_SOURCES)}",
              file=sys.stderr, flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    summary: dict[str, Any] = {}
    steps = [("reazon", bool(shards), lambda: stage_reazon(args.data, shards, captions[0], args.out)),
             ("cv", "cv" in names, lambda: stage_cv(args.data, args.plan.parent / EXCLUDE_FILE, args.out)),
             ("galgame", bool(plan.galgame), lambda: stage_galgame(args.data, plan.galgame, args.out)),
             ("jvnv", "jvnv" in names, lambda: stage_jvnv(args.data, args.out)),
             ("jnv", "jnv" in names, lambda: stage_jnv(args.data, args.out))]
    failed = []
    for source, planned, step in steps:
        if not planned:
            continue
        started = time.monotonic()
        try:
            result = step()
        except StageError as error:
            result = {"error": str(error)}
        except Exception as error:  # the type only: a message may hold a path naming a title
            result = {"error": type(error).__name__}
        if "error" in result:
            failed.append(source)
            print(f"error: {source}: {result['error']}; its rows are missing, the other sources go on",
                  file=sys.stderr, flush=True)
        summary[source] = {**result, "wall_s": round(time.monotonic() - started, 1)}
    errors = []
    if summary.get("jnv", {}).get("unphrased"):
        errors.append("a JNV regular-session clip has no phrase in phrases.txt")
    (args.out / "stage.json").write_text(json.dumps(
        {"schema": OUT_SCHEMA, "sources": summary, "failed_sources": failed, "errors": errors,
         "wall_s": round(time.monotonic() - begin, 1)}, indent=1) + "\n", encoding="utf-8")
    for error in errors:
        print(f"error: {error}", file=sys.stderr, flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
