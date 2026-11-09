"""Download the pilot's training sources (pod tasks S1 and T1).

    python -m scripts.asr_finetune.fetch --plan <stage-plan.json or fetch-t1.json> --data <dir> --out <dir> [--secrets-stdin]
    python -m scripts.asr_finetune.fetch --probe-gated --out <dir> --secrets-stdin

The plan is an outbox file (``ft-pilot-fetch-plan-1``)::

    {"schema": "ft-pilot-fetch-plan-1",
     "sources": ["reazon-small-tsv", "reazon-000", ..., "cv", "jvnv", "jnv"],
     "galgame": {"repo": ..., "revision": ..., "archives": [{"path": "GalGame/<maker>_<game>.7z", "size": N}]}}

``sources`` names distinct entries of ``pilot.SOURCES`` that land in
distinct files, with at most one ReazonSpeech caption source
(``reazon-small-tsv`` or ``reazon-mirror-all-tsv``); ``galgame``
(optional) has the shape of ``outbox\\galgame-titles.json`` and must name
``pilot.OOPPEENN``. Each source lands in ``--data`` as its ``Source.file``
(``small.tsv`` or ``all.tsv``, ``000.tar``, ..., the Common Voice
``.tar.gz``, ``jvnv_ver1.zip``, ``jnv_corpus_ver3.zip``); an archive lands at
its repo path (``<data>/GalGame/<maker>_<game>.7z``). Archives and the
sources with a ``Source.hub`` (the ReazonSpeech mirror) are downloaded with
``hf_hub_download`` at the pinned revision into ``--data``; a mirror file is
then moved from its repo path to its ``Source.file`` (``data/000.tar`` to
``000.tar``).

Four downloads run at once, largest first, each tried up to three times. A
URL download goes to ``<file>.part`` with httpx and resumes it with an HTTP
Range request; ``hf_hub_download`` resumes its own partial file. Each file
is verified against its byte size and, where the registry pins one (Common
Voice, every file of the ReazonSpeech mirror), its SHA-256 before it takes
its name; a file already present and verified is kept.

Secrets (``--secrets-stdin``): one JSON line on stdin,
``{"HF_TOKEN": ..., "MDC_URL": ...}`` (``scripts.asr_bench.remote``).
``HF_TOKEN`` goes to ``hf_hub_download`` as ``token=`` (no login, no token
file); ``MDC_URL`` is the presigned Common Voice URL. Neither is printed or
written, and an error prints its class and HTTP status only: a presigned
URL's query is a credential and an archive path names a title. Sources are
named in the console and in ``fetch.json`` by registry name, archives as
``galgame-NN`` (plan order).

``--probe-gated`` downloads only ``pilot.REAZON_LOADER`` (the 4,858-byte
gated ``reazonspeech.py``) with the token into a temporary folder that is
removed afterwards: a check of a fine-grained token on a pod.

``<out>\\fetch.json``: ``{"schema": "ft-pilot-fetch-1", "sources": [{"name",
"bytes", "seconds", "mb_per_s", "verified"}]}`` in plan order: bytes of the
file (or partial file), wall seconds, MB (10^6 bytes) transferred per second
by this run. Exit 1 when a source is not verified (``fetch.json`` written
first), 2 for an invalid plan or a missing secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from scripts.asr_finetune.pilot import OOPPEENN, REAZON_CAPTION_SOURCES, REAZON_LOADER, SOURCES, Source

# Before huggingface_hub is imported: no progress bars in a masked task log.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

PLAN_SCHEMA = "ft-pilot-fetch-plan-1"
OUT_SCHEMA = "ft-pilot-fetch-1"
PARALLEL = 4
ATTEMPTS = 3
RETRY_PAUSE_S = (5, 20)
GALGAME_FOLDER = "GalGame/"
_CHUNK = 1 << 20
_PROGRESS_BYTES = 2 << 30
_TIMEOUT = httpx.Timeout(120.0, connect=30.0)
_print_lock = threading.Lock()


class PlanError(ValueError):
    """An invalid plan or a source whose secret is missing."""


class DownloadError(RuntimeError):
    """A download that ended short or failed verification (the message holds no URL)."""


@dataclass(frozen=True)
class Plan:
    sources: tuple[Source, ...]
    galgame: tuple[tuple[str, int], ...]  # (repo path, bytes), plan order


@dataclass(frozen=True)
class Job:
    name: str
    path: Path
    bytes: int
    url: str | None = None                        # a public URL
    secret: str | None = None                     # MDC_URL: the URL; HF_TOKEN: a Hub file
    sha256: str | None = None
    hub: tuple[str, str, str] | None = None       # (repo id, revision, file) of a dataset repo


def _say(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def load_plan(path: Path) -> Plan:
    """The plan file, validated against ``pilot.SOURCES`` and ``pilot.OOPPEENN``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PlanError(f"cannot read the plan {path}: {type(error).__name__}") from None
    if not isinstance(raw, dict) or raw.get("schema") != PLAN_SCHEMA:
        raise PlanError(f"{path} is not a {PLAN_SCHEMA} plan")
    extra = set(raw) - {"schema", "sources", "galgame"}
    if extra:
        raise PlanError(f"{path}: unknown keys {sorted(extra)}")
    names = raw.get("sources", [])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise PlanError(f"{path}: sources must be a list of names")
    unknown = [name for name in names if name not in SOURCES]
    if unknown or len(set(names)) != len(names):
        raise PlanError(f"{path}: sources must be distinct names of {', '.join(SOURCES)}")
    if len({SOURCES[name].file for name in names}) != len(names):
        raise PlanError(f"{path}: two sources land in the same file (a shard from the corpus server and the mirror)")
    if sum(name in REAZON_CAPTION_SOURCES for name in names) > 1:
        raise PlanError(f"{path}: name one ReazonSpeech caption source ({' or '.join(REAZON_CAPTION_SOURCES)})")
    archives: list[tuple[str, int]] = []
    galgame = raw.get("galgame")
    if galgame is not None:
        if not isinstance(galgame, dict) or (galgame.get("repo"), galgame.get("revision")) != OOPPEENN:
            raise PlanError(f"{path}: galgame must name the repo and revision of pilot.OOPPEENN")
        for entry in galgame.get("archives") or []:
            repo_path = entry.get("path") if isinstance(entry, dict) else None
            size = entry.get("size") if isinstance(entry, dict) else None
            parts = PurePosixPath(repo_path).parts if isinstance(repo_path, str) else ()
            if (not isinstance(repo_path, str) or not repo_path.startswith(GALGAME_FOLDER)
                    or not repo_path.endswith(".7z") or len(parts) != 2 or ".." in parts
                    or not isinstance(size, int) or isinstance(size, bool) or size <= 0):
                raise PlanError(f"{path}: galgame archive {len(archives) + 1} needs a GalGame/<name>.7z "
                                "path and a positive size")
            archives.append((repo_path, size))
        if not archives or len({repo_path for repo_path, _ in archives}) != len(archives):
            raise PlanError(f"{path}: galgame needs distinct archives")
    return Plan(tuple(SOURCES[name] for name in names), tuple(archives))


def galgame_name(index: int) -> str:
    """The console and ``fetch.json`` name of the plan's ``index``-th archive (0-based)."""
    return f"galgame-{index + 1:02d}"


def plan_jobs(plan: Plan, data: Path) -> list[Job]:
    jobs = [Job(source.name, data / source.file, source.bytes, url=source.url, sha256=source.sha256,
                secret="HF_TOKEN" if source.hub else None if source.url else "MDC_URL", hub=source.hub)
            for source in plan.sources]
    repo, revision = OOPPEENN
    jobs += [Job(galgame_name(index), data / repo_path, size, secret="HF_TOKEN", hub=(repo, revision, repo_path))
             for index, (repo_path, size) in enumerate(plan.galgame)]
    return jobs


def probe_jobs(folder: Path) -> list[Job]:
    repo, revision, filename, size = REAZON_LOADER
    return [Job("probe-gated", folder / filename, size, secret="HF_TOKEN", hub=(repo, revision, filename))]


def _describe(error: BaseException) -> str:
    """An error without its message (which may hold a URL), but with an HTTP status."""
    if isinstance(error, DownloadError):
        return str(error)
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return type(error).__name__ + (f" (HTTP {status})" if status is not None else "")


def _sha256(path: Path, hasher: Any = None) -> Any:
    hasher = hasher or hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_CHUNK * 8):
            hasher.update(block)
    return hasher


def _verified(job: Job, path: Path) -> bool:
    return path.stat().st_size == job.bytes and (job.sha256 is None or _sha256(path).hexdigest() == job.sha256)


def _http(job: Job, url: str, client: httpx.Client) -> int:
    """Download (or resume) ``job`` from ``url``; returns the bytes transferred."""
    part = job.path.with_name(job.path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > job.bytes:
        part.unlink()
        have = 0
    hasher = hashlib.sha256() if job.sha256 else None
    transferred = 0
    if have < job.bytes:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with client.stream("GET", url, headers=headers) as response:
            if have and response.status_code == 206 and response.headers.get(
                    "content-range", "").startswith(f"bytes {have}-"):
                mode = "ab"
            elif response.status_code == 200:
                have, mode = 0, "wb"
            else:
                raise DownloadError(f"HTTP {response.status_code}")
            if hasher is not None and have:
                _sha256(part, hasher)
            mark = have + _PROGRESS_BYTES
            with part.open(mode) as sink:
                for chunk in response.iter_bytes(_CHUNK):
                    sink.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
                    transferred += len(chunk)
                    if have + transferred >= mark:
                        _say(f"    {job.name}: {(have + transferred) / 1e6:,.0f} of {job.bytes / 1e6:,.0f} MB")
                        mark += _PROGRESS_BYTES
    elif hasher is not None:
        _sha256(part, hasher)
    size = part.stat().st_size
    if size != job.bytes:
        if size > job.bytes:
            part.unlink()
        raise DownloadError(f"{size:,} bytes, not {job.bytes:,}")
    if hasher is not None and hasher.hexdigest() != job.sha256:
        part.unlink()
        raise DownloadError("SHA-256 differs from the pinned value")
    part.replace(job.path)
    return transferred


def _hub(job: Job, token: str, local_dir: Path) -> int:
    """Download ``job.hub`` into ``local_dir`` (its repo path below it), move
    it to ``job.path`` when that differs, and verify it; returns its bytes.
    The repo folder below ``local_dir`` stays: other downloads finish into it."""
    from huggingface_hub import hf_hub_download

    assert job.hub is not None
    repo, revision, filename = job.hub
    path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision,
                                token=token, local_dir=local_dir))
    if path.resolve() != job.path.resolve():
        path.replace(job.path)
    size = job.path.stat().st_size
    if size != job.bytes:
        raise DownloadError(f"{size:,} bytes, not {job.bytes:,}")
    if job.sha256 is not None and _sha256(job.path).hexdigest() != job.sha256:
        job.path.unlink()
        raise DownloadError("SHA-256 differs from the pinned value")
    return size


def _on_disk(job: Job) -> int:
    """Bytes of the file, or of its ``.part``, or 0."""
    for path in (job.path, job.path.with_name(job.path.name + ".part")):
        try:
            return path.stat().st_size
        except OSError:
            continue
    return 0


def run_job(job: Job, secrets: Mapping[str, str], client: httpx.Client, local_dir: Path) -> dict[str, Any]:
    """Download and verify one source; never raises (the result says
    ``verified``), and its console lines name the source, never a path."""
    begin = time.monotonic()
    transferred = 0
    verified = False
    try:
        job.path.parent.mkdir(parents=True, exist_ok=True)
        if job.path.is_file():
            if _verified(job, job.path):
                verified = True
                _say(f"{job.name}: present and verified")
            else:
                job.path.unlink()
    except OSError as error:
        _say(f"{job.name}: the present file cannot be checked: {_describe(error)}")
    for attempt in range(ATTEMPTS):
        if verified:
            break
        if attempt:
            time.sleep(RETRY_PAUSE_S[min(attempt, len(RETRY_PAUSE_S)) - 1])
        try:
            if job.hub is not None:
                transferred = _hub(job, secrets["HF_TOKEN"], local_dir)
            else:
                transferred += _http(job, job.url or secrets["MDC_URL"], client)
            verified = True
        except Exception as error:  # reported without its message; retried
            _say(f"{job.name}: attempt {attempt + 1} of {ATTEMPTS} failed: {_describe(error)}")
    seconds = time.monotonic() - begin
    size = _on_disk(job)
    rate = transferred / 1e6 / seconds if seconds > 0 else 0.0
    _say(f"{job.name}: {size:,} bytes, {seconds:.0f} s, {rate:.1f} MB/s, "
         f"{'verified' if verified else 'NOT verified'}")
    return {"name": job.name, "bytes": size, "seconds": round(seconds, 3), "mb_per_s": round(rate, 3),
            "verified": verified}


def fetch(jobs: Sequence[Job], secrets: Mapping[str, str], local_dir: Path) -> list[dict[str, Any]]:
    """Every job, ``PARALLEL`` at once, largest first; results in ``jobs`` order."""
    missing = sorted({job.secret for job in jobs if job.secret and not secrets.get(job.secret)})
    if missing:
        names = ", ".join(job.name for job in jobs if job.secret in missing)
        raise PlanError(f"{' and '.join(missing)} missing for {names}: pass it with --secrets-stdin")
    _say(f"fetching {len(jobs)} sources, {sum(job.bytes for job in jobs) / 1e9:.2f} GB, {PARALLEL} at once")
    # identity: Range offsets and the pinned sizes count the stored bytes.
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, headers={"Accept-Encoding": "identity"}) as client, \
            ThreadPoolExecutor(max_workers=PARALLEL, thread_name_prefix="fetch") as pool:
        order = sorted(range(len(jobs)), key=lambda index: -jobs[index].bytes)
        futures = {index: pool.submit(run_job, jobs[index], secrets, client, local_dir) for index in order}
        return [futures[index].result() for index in range(len(jobs))]


def read_secrets() -> dict[str, str]:
    """The one JSON line ``scripts.asr_bench.remote`` writes to stdin."""
    line = sys.stdin.readline()
    try:
        secrets = json.loads(line) if line.strip() else {}
    except ValueError:
        raise PlanError("stdin does not hold a JSON line of secrets") from None
    if not isinstance(secrets, dict) or not all(isinstance(value, str) for value in secrets.values()):
        raise PlanError("the secrets line must map names to strings")
    return secrets


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_finetune.fetch",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, help="stage-plan.json or fetch-t1.json")
    parser.add_argument("--data", type=Path, help="folder the sources land in")
    parser.add_argument("--out", type=Path, required=True, help="folder for fetch.json")
    parser.add_argument("--secrets-stdin", action="store_true", help="read the secrets JSON line from stdin")
    parser.add_argument("--probe-gated", action="store_true",
                        help="download only the gated ReazonSpeech loader with HF_TOKEN")
    args = parser.parse_args(argv)
    if args.probe_gated and (args.plan or args.data):
        parser.error("--probe-gated takes no --plan or --data")
    if not args.probe_gated and not (args.plan and args.data):
        parser.error("--plan and --data are required (or --probe-gated)")
    try:
        secrets = read_secrets() if args.secrets_stdin else {}
        if args.probe_gated:
            with tempfile.TemporaryDirectory(prefix="fetch-probe-") as folder:
                results = fetch(probe_jobs(Path(folder)), secrets, Path(folder))
        else:
            results = fetch(plan_jobs(load_plan(args.plan), args.data), secrets, args.data)
    except PlanError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fetch.json").write_text(json.dumps({"schema": OUT_SCHEMA, "sources": results}, indent=1) + "\n",
                                         encoding="utf-8")
    bad = [result["name"] for result in results if not result["verified"]]
    total = sum(result["bytes"] for result in results)
    print(f"fetch.json: {len(results) - len(bad)} of {len(results)} sources verified, {total / 1e9:.2f} GB"
          + (f"; not verified: {', '.join(bad)}" if bad else ""), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
