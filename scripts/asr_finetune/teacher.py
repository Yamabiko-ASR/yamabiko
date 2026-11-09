"""Prompt-free teacher decode of the pilot's training clips (pod tasks S1 and T).

    python -m scripts.asr_finetune.teacher --rows <reazon-rows.jsonl> --data <dir> --out <dir> --workers 4 --stop-at <epoch> [--limit N]
    python -m scripts.asr_finetune.teacher --plan <windows-plan.jsonl.gz> --data <dir> --out <dir> [--words <file>] --workers 4

``--device`` and ``--compute-type`` override the shipped settings (a CPU
probe); the pods keep the shipped ``cuda`` and ``float16``.

The teacher is the base model's recognizer without its prompt: each worker
process builds ``yamabiko.asr.asr_client.build_asr`` from the ``asr`` section
of ``yamabiko/default_config.yaml`` with ``faster_whisper.model`` set to the
base export's pinned snapshot (``common.base_export_dir``, fetched into the
app's model cache first), ``initial_prompt`` set to "" (the prompt the base
model shipped with adds a phantom はい) and ``no_speech_only_threshold`` set
to 1.0 (off, as the base model shipped), and decodes each clip as
one window (``transcribe`` on the whole clip, the 500 ms boundary pad
included), as the data survey's spot check did. A clip is the member
``key`` (``000/xxxx.flac``) of ``<data>\\<shard>.tar``, decoded to 16 kHz mono
float32 with the steps of ``faster_whisper.decode_audio`` (metadata errors
ignored, undecodable frames skipped, frames grouped, s16 over 32768); that
function itself is unavailable beside the recognizer, which installs a
NumPy-only ``faster_whisper.audio``. Only ReazonSpeech training clips are
ever decoded here.

``<out>\\teacher.jsonl``: ``{"key", "text", "outcome"}`` per clip, appended
in row order (shard order) and flushed per clip. ``outcome`` is the
``ASROutcome`` value, or ``decode_error`` (PyAV cannot open or decode the
clip) or ``empty_audio`` (no samples), each with an empty text. A key
already in the file is skipped, so a rerun resumes. ``--stop-at`` (epoch
seconds, the task's ``{deadline}``): no clip is handed to a worker after it;
clips already handed out finish and are written. ``--limit``: at most N
clips in this run.

``<out>\\teacher.json``: rows, lines, clips decoded in this run and their
outcomes, the shards complete (every row has a line) and partial, audio,
transcribe and decode wall seconds, each worker's load seconds, why the run
stopped (``complete``, ``deadline``, ``limit`` or ``error``; for ``error``
the exception type only, the traceback goes to the console) and the decode
settings. Exit 1 when the run stopped on an error (``teacher.json`` and the
lines written so far stay; the pod run fetches a failed task's output too),
else 0, also for ``deadline`` and ``limit``.

Words mode (``--plan``, round 3): the teacher's word timings of every
utterance that carries a cut mark (``cut_start`` or ``cut_end``) in the
window plan (``mix --cuts``), ReazonSpeech clips and galgame lines, found in
``--data`` as ``scripts.asr_finetune.windows`` finds them (``gather``; it
extracts the galgame archives and records them, so the windows task reuses
the extraction). Each is decoded like the text mode, with the production
``transcribe`` arguments (``asr_client``: beam, the three thresholds, no
VAD filter, no prompt, the 500 ms pads) plus ``word_timestamps=True`` on the
recognizer's own model; segments the production junk gate drops
(``no_speech_prob`` over 0.6 with ``avg_logprob`` under -1.0) contribute no
words. ``--words`` (default ``<out>\\words.jsonl``): ``{"src", "id",
"duration_s", "outcome", "words": [[start_s, end_s, word], ...]}`` per
utterance, times in the clip (the pad taken off, clamped to the clip),
appended in plan order and resumable like ``teacher.jsonl``; ``outcome`` is
``words`` (at least one word), ``no_words``, ``decode_error`` or
``empty_audio``. ``<out>\\words.json`` records the counts, timings and decode
settings, and ``<out>`` gets a copy of the words file when ``--words`` lies
elsewhere (the pod keeps the live file in ``{data}`` for the windows task).
Utterances the data lacks are counted (``not_found``) and get no line; the
windows task then refuses their cuts (``no_words``).
"""

from __future__ import annotations

import argparse
import gc
import io
import itertools
import json
import multiprocessing
import queue
import shutil
import sys
import tarfile
import time
import traceback
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from scripts.asr_finetune.common import BASE_EXPORT_REPO, BASE_EXPORT_REVISION, base_export_dir

OUT_SCHEMA = "ft-pilot-teacher-1"
WORDS_SCHEMA = "ft-pilot-teacher-words-1"
SAMPLE_RATE = 16_000
_POLL_S = 1.0
_GROUP_SAMPLES = 500_000  # faster_whisper.audio's frame grouping


class TeacherError(RuntimeError):
    """A worker failed or a clip is missing; ``kind`` is what teacher.json
    records (an exception type name)."""

    def __init__(self, message: str, kind: str = "TeacherError") -> None:
        super().__init__(message)
        self.kind = kind


def teacher_config(device: str | None = None, compute_type: str | None = None) -> dict[str, Any]:
    """The shipped ``asr`` section decoding with the base export
    (``base_export_dir``, its pinned snapshot, fetched into the app's model
    cache first) with the prompt and the no-speech-only gate off, as the base
    model shipped without its prompt (and the given device or compute type
    instead of the shipped ones)."""
    import yaml

    import yamabiko

    config = yaml.safe_load((Path(yamabiko.__file__).parent / "default_config.yaml").read_text(encoding="utf-8"))
    asr = config["asr"]
    asr["faster_whisper"]["model"] = str(base_export_dir(download=True))
    asr["faster_whisper"]["initial_prompt"] = ""
    asr["faster_whisper"]["no_speech_only_threshold"] = 1.0
    if device:
        asr["faster_whisper"]["device"] = device
    if compute_type:
        asr["faster_whisper"]["compute_type"] = compute_type
    return asr


def _valid_frames(frames: Iterable[Any]) -> Iterator[Any]:
    """``frames`` without the ones FFmpeg cannot decode (faster_whisper.audio)."""
    import av

    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except av.error.InvalidDataError:
            continue


def _grouped(frames: Iterable[Any]) -> Iterator[Any]:
    """``frames`` regrouped into frames of ``_GROUP_SAMPLES`` (faster_whisper.audio)."""
    import av

    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None
        fifo.write(frame)
        if fifo.samples >= _GROUP_SAMPLES:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def decode_clip(data: bytes) -> Any:
    """16 kHz mono float32 samples of an audio file's bytes, empty when
    nothing decodes; the steps of ``faster_whisper.decode_audio``."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(io.BytesIO(data), mode="r", metadata_errors="ignore") as container:
        for frame in itertools.chain(_grouped(_valid_frames(container.decode(audio=0))), [None]):
            chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
    del resampler
    gc.collect()  # as faster_whisper.audio does: resampler objects are otherwise kept
    return np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, np.float32)


def production_segments(asr: Any, audio: Any, *, pad: bool = True, word_timestamps: bool = False) -> list[Any]:
    """The segments the recognizer (``build_asr``) keeps for ``audio``: its
    own model and ``transcribe`` arguments (``asr_client``: beam, the three
    thresholds, no VAD filter, its prompt, no conditioning), with its
    boundary pad around the audio unless ``pad`` is False (audio that
    already carries it), without the segments its junk gate drops
    (``no_speech_prob`` over 0.6 with ``avg_logprob`` under -1.0)."""
    import numpy as np

    from yamabiko.asr import asr_client

    if pad:
        zeros = np.zeros(int(round(SAMPLE_RATE * asr._boundary_pad_ms / 1000)), dtype=np.float32)
        audio = np.concatenate([zeros, audio, zeros])
    segments, _ = asr._model.transcribe(
        audio, language="ja", beam_size=asr._beam_size,
        compression_ratio_threshold=asr_client._COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=asr_client._LOGPROB_THRESHOLD, no_speech_threshold=asr_client._NO_SPEECH_THRESHOLD,
        vad_filter=False, initial_prompt=asr._initial_prompt, condition_on_previous_text=False,
        word_timestamps=word_timestamps)
    return [segment for segment in segments
            if not (segment.no_speech_prob > asr_client._NO_SPEECH_THRESHOLD
                    and segment.avg_logprob < asr_client._LOGPROB_THRESHOLD)]


def transcribe_words(asr: Any, audio: Any) -> list[list[Any]]:
    """``[start_s, end_s, word]`` of every word the recognizer hears in
    ``audio`` (``production_segments`` with ``word_timestamps=True``), the
    boundary pad taken off the times."""
    offset, duration = asr._boundary_pad_ms / 1000, len(audio) / SAMPLE_RATE
    return [[round(min(max(word.start - offset, 0.0), duration), 3),
             round(min(max(word.end - offset, 0.0), duration), 3), word.word]
            for segment in production_segments(asr, audio, word_timestamps=True) for word in segment.words or ()]


def _worker(number: int, config: dict[str, Any], tasks: Any, results: Any, words: bool = False) -> None:
    """One worker process: builds the recognizer, then decodes the
    ``(position, audio file bytes)`` tasks it takes until it takes None:
    the text (``words`` False) or the word timings of each."""
    try:
        import av

        from yamabiko.asr.asr_client import build_asr

        begin = time.perf_counter()
        asr = build_asr(config)
        asr.warmup()
        results.put(("ready", number, time.perf_counter() - begin))
        empty = {"duration_s": 0.0, "words": []} if words else {"text": ""}
        while (task := tasks.get()) is not None:
            position, data = task
            try:
                audio = decode_clip(data)
            except (av.error.FFmpegError, ValueError):  # a clip faster-whisper would refuse too
                results.put(("clip", position, ({**empty, "outcome": "decode_error"}, 0.0, 0.0)))
                continue
            if audio.size == 0:
                results.put(("clip", position, ({**empty, "outcome": "empty_audio"}, 0.0, 0.0)))
                continue
            started = time.perf_counter()
            if words:
                heard = transcribe_words(asr, audio)
                fields = {"duration_s": round(len(audio) / SAMPLE_RATE, 3),
                          "outcome": "words" if heard else "no_words", "words": heard}
            else:
                result = asr.transcribe(audio, SAMPLE_RATE)
                fields = {"text": result.text, "outcome": getattr(result.outcome, "value", str(result.outcome))}
            results.put(("clip", position, (fields, len(audio) / SAMPLE_RATE, time.perf_counter() - started)))
        results.put(("done", number, None))
    except BaseException as error:  # reported to the parent, which stops the run
        results.put(("error", number, (type(error).__name__, traceback.format_exc())))


class _Clips:
    """The FLAC bytes of a key, one shard tar open at a time."""

    def __init__(self, data: Path) -> None:
        self._data = data
        self._shard: str | None = None
        self._archive: tarfile.TarFile | None = None
        self._members: dict[str, tarfile.TarInfo] = {}

    def read(self, key: str) -> bytes:
        shard = key.split("/", 1)[0]
        if shard != self._shard:
            self.close()
            self._archive = tarfile.open(self._data / f"{shard}.tar")
            self._members = {member.name.removeprefix("./"): member
                             for member in self._archive.getmembers() if member.isfile()}
            self._shard = shard
        member = self._members.get(key)
        source = None if member is None or self._archive is None else self._archive.extractfile(member)
        if source is None:
            raise TeacherError(f"{key} is not a file of {shard}.tar")
        return source.read()

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
        self._archive, self._shard, self._members = None, None, {}


class _Run:
    """The parent's side of a run: worker messages in, lines out in row order
    (each line: the ``ident`` fields of its pending entry, then the worker's)."""

    def __init__(self, pending: Sequence[dict[str, Any]], ident: Sequence[str], sink: Any,
                 processes: Sequence[Any], results: Any) -> None:
        self.pending, self.ident, self.sink, self.processes, self.results = pending, ident, sink, processes, results
        self.loaded: dict[int, float] = {}
        self.finished: set[int] = set()
        self.waiting: dict[int, tuple[dict[str, Any], float, float]] = {}
        self.written = 0
        self.outcomes: Counter = Counter()
        self.audio_s = 0.0
        self.asr_s = 0.0
        self._idle_polls = 0

    def _handle(self, kind: str, key: int, payload: Any) -> None:
        if kind == "error":
            error_type, trace = payload
            raise TeacherError(f"teacher worker {key} failed:\n{trace}", kind=error_type)
        if kind == "ready":
            self.loaded[key] = payload
        elif kind == "done":
            self.finished.add(key)
        else:
            self.waiting[key] = payload
            while self.written in self.waiting:
                fields, audio_s, asr_s = self.waiting.pop(self.written)
                entry = self.pending[self.written]
                self.sink.write(json.dumps({**{name: entry[name] for name in self.ident}, **fields},
                                           ensure_ascii=False) + "\n")
                self.sink.flush()
                self.outcomes[fields["outcome"]] += 1
                self.audio_s += audio_s
                self.asr_s += asr_s
                self.written += 1

    def pump(self, block: bool) -> None:
        """Handle the waiting messages (with ``block``, waiting up to a poll
        interval for one); fail when a worker died."""
        try:
            if block:
                self._handle(*self.results.get(timeout=_POLL_S))
            while True:
                self._handle(*self.results.get_nowait())
        except queue.Empty:
            pass
        for process in self.processes:
            if process.exitcode not in (None, 0):
                raise TeacherError(f"{process.name} exited with code {process.exitcode}", kind="WorkerExited")
        # A worker sends its last message before it exits: two idle polls
        # after every worker exited mean one ended without its "done".
        if block and all(process.exitcode is not None for process in self.processes):
            self._idle_polls += 1
            if self._idle_polls > 2:
                raise TeacherError("a teacher worker exited without reporting its end", kind="WorkerExited")

    def put(self, tasks: Any, item: Any) -> None:
        while True:
            try:
                tasks.put(item, timeout=_POLL_S)
                return
            except queue.Full:
                self.pump(block=False)


def decode(pending: Sequence[dict[str, Any]], read: Callable[[dict[str, Any]], bytes], ident: Sequence[str],
           sink: Any, workers: int, stop_at: float | None, stats: dict[str, Any], config: dict[str, Any],
           words: bool = False) -> None:
    """Decode ``pending`` entries (audio file bytes from ``read``) in
    ``workers`` spawned processes, writing each line to ``sink`` in order;
    fills ``stats``; sets ``stats["stopped"]`` to ``deadline`` when
    ``stop_at`` cut the run short."""
    spawn = multiprocessing.get_context("spawn")
    tasks = spawn.Queue(maxsize=2 * workers)
    results = spawn.Queue()
    processes = [spawn.Process(target=_worker, args=(number, config, tasks, results, words),
                               name=f"teacher-worker-{number}", daemon=True) for number in range(workers)]
    run = _Run(pending, ident, sink, processes, results)
    sent = 0
    try:
        for process in processes:
            process.start()
        while len(run.loaded) < workers:
            run.pump(block=True)
        stats["load_s"] = [round(run.loaded[number], 1) for number in range(workers)]
        print(f"teacher: {workers} workers loaded in {max(run.loaded.values()):.0f} s", flush=True)
        begin = time.perf_counter()
        for position, entry in enumerate(pending):
            if stop_at is not None and time.time() >= stop_at:
                stats["stopped"] = "deadline"
                break
            run.put(tasks, (position, read(entry)))
            sent += 1
            run.pump(block=False)
            if sent % 1000 == 0:
                print(f"teacher: {run.written:,} of {len(pending):,} clips", flush=True)
        for _ in processes:
            run.put(tasks, None)
        while len(run.finished) < workers:
            run.pump(block=True)
        for process in processes:
            process.join()
        if run.written != sent:
            raise TeacherError(f"the workers returned {run.written} of {sent} clips")
        stats["decode_wall_s"] = round(time.perf_counter() - begin, 1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=30)
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
        stats.update(decoded=run.written, outcomes=dict(sorted(run.outcomes.items())),
                     audio_s=round(run.audio_s, 1), asr_s=round(run.asr_s, 1))


def _done(path: Path, ident: Sequence[str]) -> set[tuple[Any, ...]]:
    """The ``ident`` values of the complete lines of an earlier run's lines
    file; a last line without its newline (a run cut mid-write) is removed."""
    if not path.is_file():
        return set()
    raw = path.read_bytes()
    end = raw.rfind(b"\n") + 1
    if end < len(raw):
        with path.open("r+b") as handle:
            handle.truncate(end)
    lines = (json.loads(line) for line in raw[:end].decode("utf-8").splitlines() if line)
    return {tuple(line[name] for name in ident) for line in lines}


def _settings(config: dict[str, Any], words: bool) -> dict[str, Any]:
    """The decode settings of a ``teacher_config`` run: the base export's
    pinned identity and its snapshot directory, then the decode options."""
    whisper = config["faster_whisper"]
    return {"model": BASE_EXPORT_REPO, "revision": BASE_EXPORT_REVISION, "model_dir": whisper["model"],
            "compute_type": whisper["compute_type"], "device": whisper["device"],
            "beam_size": whisper.get("beam_size", 5), "boundary_pad_ms": config.get("boundary_pad_ms", 500),
            "initial_prompt": whisper["initial_prompt"],
            "no_speech_only_threshold": whisper["no_speech_only_threshold"],
            "window": "whole clip", "word_timestamps": words}


def _run(pending: Sequence[dict[str, Any]], read: Callable[[dict[str, Any]], bytes], ident: Sequence[str],
         lines: Path, args: argparse.Namespace, config: dict[str, Any], stats: dict[str, Any], words: bool) -> None:
    """``decode`` into ``lines`` (appended); an error is recorded in ``stats``."""
    try:
        if pending and (args.stop_at is None or time.time() < args.stop_at):
            with lines.open("a", encoding="utf-8", newline="\n") as sink:
                decode(pending, read, ident, sink, args.workers, args.stop_at, stats, config, words)
        elif pending:
            stats["stopped"] = "deadline"
    except Exception as error:  # recorded; the lines written so far stay for the fetch
        stats["stopped"], stats["error"] = "error", getattr(error, "kind", type(error).__name__)
        print(f"error: {error}", file=sys.stderr, flush=True)


def _speed(stats: dict[str, Any]) -> str:
    audio_s, wall = stats.get("audio_s", 0.0), stats.get("decode_wall_s")
    return (f"{stats.get('decoded', 0):,} clips decoded, {audio_s:,.1f} s of audio"
            + (f", {wall:,.1f} s decode wall ({10 * wall / audio_s:.2f} s per 10 s of audio)" if wall and audio_s else ""))


def _limit(pending: list[dict[str, Any]], limit: int | None, stats: dict[str, Any]) -> list[dict[str, Any]]:
    if limit is not None and len(pending) > limit:
        stats["stopped"] = "limit"
        return pending[:limit]
    return pending


def rows_mode(args: argparse.Namespace, config: dict[str, Any]) -> int:
    begin = time.monotonic()
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line]
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        print(f"error: {args.rows} repeats a key", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    lines = args.out / "teacher.jsonl"
    done = {key for key, in _done(lines, ("key",))}
    stats: dict[str, Any] = {"stopped": "complete"}
    pending = _limit([row for row in rows if row["key"] not in done], args.limit, stats)
    print(f"teacher: {len(rows):,} rows, {len(done):,} already decoded, {len(pending):,} to decode", flush=True)
    clips = _Clips(args.data)
    try:
        _run(pending, lambda row: clips.read(row["key"]), ("key",), lines, args, config, stats, words=False)
    finally:
        clips.close()
    done = {key for key, in _done(lines, ("key",))}
    shards: dict[str, bool] = {}
    for key in keys:
        shard = key.split("/", 1)[0]
        shards[shard] = shards.get(shard, True) and key in done
    record = {
        "schema": OUT_SCHEMA, "rows": len(rows), "lines": len(done), **stats,
        "shards_complete": [shard for shard, complete in shards.items() if complete],
        "shards_partial": [shard for shard, complete in shards.items() if not complete],
        "workers": args.workers, "stop_at": args.stop_at, "limit": args.limit,
        "wall_s": round(time.monotonic() - begin, 1), "decode": _settings(config, words=False),
    }
    (args.out / "teacher.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"teacher: {_speed(stats)}; stopped: {stats['stopped']}; "
          f"shards complete: {', '.join(record['shards_complete']) or 'none'}", flush=True)
    return 1 if stats["stopped"] == "error" else 0


def words_mode(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from scripts.asr_finetune.stage import StageError
    from scripts.asr_finetune.windows import PlanError, gather, locator_bytes, read_plan

    begin = time.monotonic()
    try:
        plan = read_plan(args.plan)
        wanted = list(dict.fromkeys((utt["src"], utt["id"]) for row in plan for utt in row["utts"]
                                    if "cut_start" in utt or "cut_end" in utt))
        found, failed = (gather([{"utts": [{"src": src, "id": ident} for src, ident in wanted]}], args.data)
                         if wanted else ({}, 0))
    except (PlanError, StageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"error: a source file lacks {error.args[0]!r}", file=sys.stderr)
        return 2
    except (OSError, tarfile.TarError, zipfile.BadZipFile, UnicodeDecodeError) as error:  # the type: a message holds a path
        print(f"error: cannot read a source file ({type(error).__name__})", file=sys.stderr)
        return 2
    lines = args.words or args.out / "words.jsonl"
    lines.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ident = ("src", "id")
    done = _done(lines, ident)
    present = [key for key in wanted if key in found]
    stats: dict[str, Any] = {"stopped": "complete"}
    pending = _limit([{"src": src, "id": name} for src, name in present if (src, name) not in done], args.limit, stats)
    print(f"teacher words: {len(wanted):,} cut utterances in the plan, {len(present):,} found, "
          f"{len(done):,} already decoded, {len(pending):,} to decode", flush=True)
    _run(pending, lambda entry: locator_bytes(found[(entry["src"], entry["id"])][1]), ident, lines, args, config,
         stats, words=True)
    heard = [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines() if line] if lines.is_file() else []
    record = {
        "schema": WORDS_SCHEMA, "planned": len(wanted), "found": len(present), "not_found": len(wanted) - len(present),
        "galgame_archives_failed": failed, "lines": len(heard),
        "outcomes_all_lines": dict(sorted(Counter(line["outcome"] for line in heard).items())),
        "words": sum(len(line["words"]) for line in heard), **stats,
        "workers": args.workers, "stop_at": args.stop_at, "limit": args.limit,
        "wall_s": round(time.monotonic() - begin, 1), "decode": _settings(config, words=True),
    }
    (args.out / "words.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if lines.is_file() and lines.resolve() != (args.out / "words.jsonl").resolve():
        shutil.copyfile(lines, args.out / "words.jsonl")
    print(f"teacher words: {_speed(stats)}; lines {len(heard):,} ({record['outcomes_all_lines']}), "
          f"{record['words']:,} words; not found {record['not_found']:,}; stopped: {stats['stopped']}", flush=True)
    return 1 if stats["stopped"] == "error" else 0


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.teacher",
                                     description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rows", type=Path, help="reazon-rows.jsonl of the stage task (text mode)")
    mode.add_argument("--plan", type=Path, help="windows-plan.jsonl.gz with cut marks (words mode)")
    parser.add_argument("--data", type=Path, required=True, help="the fetched sources (<shard>.tar, archives, ...)")
    parser.add_argument("--out", type=Path, required=True,
                        help="folder for teacher.jsonl and teacher.json (words mode: words.json)")
    parser.add_argument("--words", type=Path, help="words mode: the words file (default <out>\\words.jsonl)")
    parser.add_argument("--workers", type=int, default=1, help="decoding processes (default 1)")
    parser.add_argument("--stop-at", type=float, help="epoch seconds after which no clip starts")
    parser.add_argument("--limit", type=int, help="decode at most this many clips")
    parser.add_argument("--device", help="override the shipped device (e.g. cpu, for a local probe)")
    parser.add_argument("--compute-type", help="override the shipped compute type (e.g. int8 on a CPU)")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be at least 1")
    if args.words is not None and args.plan is None:
        parser.error("--words needs --plan")
    config = teacher_config(args.device, args.compute_type)
    return rows_mode(args, config) if args.rows is not None else words_mode(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
