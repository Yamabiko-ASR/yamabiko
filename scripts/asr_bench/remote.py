"""Run bench jobs or pilot pod tasks on a RunPod Secure Cloud GPU pod over SSH.

Usage::

    python -m scripts.asr_bench.remote whoami
    python -m scripts.asr_bench.remote gpus [--all] [--allow-datacenter]
    python -m scripts.asr_bench.remote plan (--jobs FILE | --job SET CANDIDATE NAME OUT ... | --tasks FILE)
    python -m scripts.asr_bench.remote run (--jobs FILE | --job ... | --tasks FILE) [--approve-cost USD]
    python -m scripts.asr_bench.remote run (--jobs FILE | --job ... | --tasks FILE) --host HOST --port PORT
        --pod-id ID [--rate USD_PER_HOUR]
    python -m scripts.asr_bench.remote spend
    python -m scripts.asr_bench.remote list
    python -m scripts.asr_bench.remote terminate POD_ID [POD_ID ...]
    python -m scripts.asr_bench.remote compare RUN_DIR RUN_DIR

A job is one ``scripts.asr_bench.run`` invocation (with a ``seed``, one of
``scripts.asr_finetune.seeded_bench_run``): a set directory, a candidate
file, the candidate's name and a new local run directory. A pod task
(``--tasks``) is one ``scripts.asr_finetune`` module run with uploaded
outbox files and, over stdin, the secrets it names; its output directory
comes back. ``plan`` prices a list of jobs or tasks on one pod; ``run``
carries them out on one pod and exits non-zero on any failure or refusal:

- API mode (the default): creates the pod (one GPU, Secure Cloud, no
  volume), runs the jobs and terminates the pod when they end or fail, on
  Ctrl-C and at the hard wall-clock cap.
- SSH mode (``--host``): the pod exists already; the script runs the jobs on
  it and wipes its data, and something else creates and terminates the pod.

The approval policy, the GPU rule, the data handling and the estimate are
described in ``docs/ASR-BENCH.md`` ("Running on RunPod"). In both modes a
watchdog on the pod wipes the work directory and terminates the pod at the
cap through the pod's own scoped API key, so a lost connection never leaves
a pod billing.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import getpass
import gzip
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import subprocess
import statistics
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any
from urllib.parse import unquote, urlsplit

import filelock
import httpx

from scripts.asr_bench.candidates import AsrSpec, Candidate, load_candidate
from scripts.asr_bench.sets import BenchSet, bench_root, load_set
from scripts.asr_finetune.outbox import OutboxRefusal, check_outbox_file, outbox_dir
from scripts.e0_evidence.private_root import ROOT_MANIFEST_NAME

REST_URL = "https://api.runpod.io/v2"
GRAPHQL_URL = "https://api.runpod.io/graphql"
# RunPod's official base image (Ubuntu 24.04, Python 3.12, uv, sshd started
# from PUBLIC_KEY by /start.sh), CPU variant: the CUDA libraries come from the
# release pins, the driver from the host. Pinned by digest.
IMAGE = ("runpod/base:1.4.1-ubuntu2404"
         "@sha256:88b67089c36fe9fe8e2e37b0655fccd972eb89f8f04613c1f52a989cd5992318")
REMOTE_ROOT = "/ybench"
SSH_KEY = Path.home() / ".ssh" / "runpod_ed25519"
MIN_CUDA = "12.4"
# torch cu128 wheels (transformers_asr) need a CUDA 12.8 driver.
MIN_CUDA_TORCH = "12.8"
MIN_VRAM_GB = 16

# The owner's standing approval policy.
ROUTINE_JOB_USD = 2.00
ROUTINE_DAY_USD = 10.00
# The most ``--day-cap`` (the owner's raised day cap) may be.
DAY_CAP_MAX_USD = 50.00
CUTOFF_FACTOR = 1.5
LINGER_MINUTES = 15

# Estimate model (seconds). The run time of each job comes from a past run.
_PROVISION_S = 300.0       # create to SSH: scheduling, image pull, sshd
_INSTALL_S = 120.0         # uv, release set; 18 s measured on a home link
_INSTALL_TORCH_S = 240.0   # torch cu128 and transformers
_INSTALL_BENCH_S = 30.0    # onnx-asr and sherpa-onnx
_INSTALL_P7ZIP_S = 60.0    # apt-get p7zip-full, pod tasks
_INSTALL_FT_S = 180.0      # the pilot venv: torch cu128, transformers, CTranslate2
_HF_MODEL_S = 90.0         # one Hugging Face model downloaded on the pod
_JOB_START_S = 30.0        # interpreter start and imports, per job
_FETCH_S = 20.0            # per job or pod task
_TEARDOWN_S = 60.0         # wipe and terminate
_INSTALL_ATTEMPTS = 3
_INSTALL_BACKOFF_S = 30.0  # then 60 s
_UPLINK_MBPS = 20.0        # until the spend log holds a measured upload
_UPLINK_SAMPLES = 3        # the default uplink is the median of this many
_UPLINK_MIN_BYTES = 100e6  # smaller uploads time mostly the SSH setup
_DISK_USD_PER_GB_HOUR = 0.10 / 720  # container disk, $0.10 per GB-month
_LOCK_TIMEOUT_S = 900.0    # waiting for another run's approval and create
_STRAY_PAUSE_S = 10.0      # between the pod list checks after a failed create
# An open record past its deadline by this much is settled: its watchdog has
# terminated the pod (or a live client would have closed the record).
_SETTLE_GRACE = timedelta(minutes=15)

# Workers "auto" (accuracy-only jobs): every worker process holds its own
# copy of the ASR model on the GPU. Per worker: the model files plus the
# CUDA context, cuBLAS and cuDNN workspaces and decode activations, plus a
# sound tagger's share kept free though the runner reads precomputed tags;
# the device keeps headroom besides. The 760M (1.51 GB of files) takes
# 4,514 MiB per worker: 4 fit on a 24 GB card, 3 on a 16 GB one.
_WORKER_OVERHEAD_MIB = 2048
_TAGGER_MIB = 1024
_VRAM_HEADROOM_MIB = 2048
# Each worker runs Silero, the chunker and its strategy on the CPU and
# feeds its decodes: two vCPUs each.
_VCPUS_PER_WORKER = 2
_AUTO_WORKERS_MAX = 4
# A pod's vCPU count before it exists, when no past run on the GPU type
# recorded one.
_ASSUMED_VCPUS = 4
# Throughput of N workers over one, assumed until a run with N workers
# exists: 1 + 0.25 (N - 1). On the 4090 one worker spent 18% of its time
# in CPU-only VAD that other workers overlap, so even a GPU that one worker
# already saturated finishes a job within the 1.5 times cut-off.
_ASSUMED_SPEEDUP_PER_WORKER = 0.25
AUTO = "auto"

# Pod tasks (--tasks): modules of the fine-tuning pilot, the secrets they may
# receive over stdin (never on a command line, in a log or in a record) and
# what each comes from on this machine.
SECRET_NAMES = ("HF_TOKEN", "MDC_URL")
TASK_MODULE_PREFIX = "scripts.asr_finetune."
MDC_API = "https://mozilladatacollective.com/api"
DEADLINE_RESERVE_S = 600  # {deadline} = hard cap minus this
_TASK_ENVS = ("release", "ft")
_PLACEHOLDERS = ("in", "out", "data", "repo", "deadline")
_CREDENTIAL_SOURCES = {
    "HF_TOKEN": ("YAMABIKO_FT_HF_TOKEN", "Hugging Face fine-grained read token"),
    "MDC_API_KEY": ("YAMABIKO_MDC_API_KEY", "Mozilla Data Collective API key"),
}

_STOCK_RANK = {"Low": 1, "Medium": 2, "High": 3}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE_PATHS = (
    "yamabiko", "scripts", "constraints-release.txt", "requirements-bench.txt",
    "requirements-bench-transformers.txt",
)
_SPEND_SCHEMA = "asr-bench-runpod-spend-1.3"
_REMOTE_SCHEMA = "asr-bench-remote-1.2"
_POD_TASK_SCHEMA = "asr-bench-pod-task-1.1"

_GPU_QUERY = """query GpuTypes($cuda: String) {
  gpuTypes {
    id displayName memoryInGb secureCloud securePrice manufacturer
    lowestPrice(input: {gpuCount: 1, secureCloud: true, minCudaVersion: $cuda}) { stockStatus }
  }
}"""

# Pod self-termination: at the deadline (epoch seconds in ./deadline, an
# unreadable or missing deadline counts as reached) wipe the work directory
# named in ./work and delete this pod with its own scoped key, retrying until
# the pod is gone. The key and pod id come from /proc/1/environ when the
# shell that started the script lacks them (SSH sessions do).
_WATCHDOG = r"""#!/bin/bash
dir=$(cd "$(dirname "$0")" && pwd)
echo $$ > "$dir/pid"
for name in RUNPOD_POD_ID RUNPOD_API_KEY; do
    if [ -z "${!name}" ] && [ -r /proc/1/environ ]; then
        export "$name=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$name=//p" | head -n 1)"
    fi
done
while true; do
    deadline=$(cat "$dir/deadline" 2>/dev/null)
    case "$deadline" in ''|*[!0-9]*) break ;; esac
    [ "$(date +%s)" -ge "$deadline" ] && break
    sleep 15
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) cap reached, terminating pod $RUNPOD_POD_ID"
work=$(cat "$dir/work" 2>/dev/null)
case "$work" in /?*) rm -rf -- "$work" ;; esac
while true; do
    runpodctl pod delete "$RUNPOD_POD_ID" \
        || runpodctl remove pod "$RUNPOD_POD_ID" \
        || curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
        || curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
    sleep 60
done
"""

# Container command of a pod created here: start the watchdog with the
# absolute deadline the client computed before the create call (billing
# starts at creation; the container disk holding ./deadline is wiped on every
# restart, the pod's env is not), keeping an earlier deadline already on
# disk; then run the image's own start script and stay up if it exits.
_BOOTSTRAP = (
    'w="$YBENCH_WORK-watchdog"; ('
    ' mkdir -p "$w" && printf %s "$YBENCH_WATCHDOG" | base64 -d > "$w/watchdog.sh"'
    ' && printf "%s\\n" "$YBENCH_WORK" > "$w/work"'
    ' && d="$YBENCH_DEADLINE" && c=$(cat "$w/deadline" 2>/dev/null || true)'
    ' && case "$c" in ""|*[!0-9]*) ;; *) if [ "$c" -lt "$d" ]; then d="$c"; fi ;; esac'
    ' && printf "%s\\n" "$d" > "$w/deadline"'
    ' && exec setsid bash "$w/watchdog.sh" > "$w/log" 2>&1 < /dev/null ) &'
    ' if [ -x /start.sh ]; then /start.sh; fi; exec sleep infinity'
)

# $1 work root, $2 deadline (epoch seconds), $3 watchdog script (base64): set
# the deadline (atomically) and start the watchdog unless it already runs.
_ARM_WATCHDOG = r"""set -eu
w="$1-watchdog"
mkdir -p "$w"
printf %s "$3" | base64 -d > "$w/watchdog.sh.tmp" && mv "$w/watchdog.sh.tmp" "$w/watchdog.sh"
printf '%s\n' "$1" > "$w/work"
echo "$2" > "$w/deadline.tmp" && mv "$w/deadline.tmp" "$w/deadline"
if ! { [ -f "$w/pid" ] && kill -0 "$(cat "$w/pid")" 2>/dev/null; }; then
    setsid nohup bash "$w/watchdog.sh" > "$w/log" 2>&1 < /dev/null &
    sleep 1
fi
kill -0 "$(cat "$w/pid")" && echo armed
"""

# $1 work root, $2 deadline (epoch seconds): move the deadline earlier, never later.
_EARLIER_DEADLINE = r"""set -eu
w="$1-watchdog"
new=$2
cur=$(cat "$w/deadline" 2>/dev/null || echo "$new")
if [ "$new" -lt "$cur" ]; then
    echo "$new" > "$w/deadline.tmp" && mv "$w/deadline.tmp" "$w/deadline"
fi
"""

# Facts about the pod, as JSON on stdout.
_PREFLIGHT = r"""
import json, os, re, shutil, subprocess, urllib.request
try:
    raw = open("/proc/1/environ", "rb").read().decode("utf-8", "replace")
    first = dict(entry.split("=", 1) for entry in raw.split("\0") if "=" in entry)
except OSError:
    first = {}
def env(name):
    return os.environ.get(name) or first.get(name) or None
facts = {"pod_id": env("RUNPOD_POD_ID"), "pod_key": bool(env("RUNPOD_API_KEY")),
         "data_center": env("RUNPOD_DC_ID"), "runpodctl": shutil.which("runpodctl"),
         "curl": shutil.which("curl"), "python312": shutil.which("python3.12"), "uv": shutil.which("uv"),
         "vcpus": env("RUNPOD_CPU_COUNT") or os.cpu_count(), "cpu_model": None,
         "gpus": [], "cuda": None, "pod": None, "pod_api": None, "container_seconds": None}
try:
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                facts["cpu_model"] = line.split(":", 1)[1].strip()
                break
except OSError:
    pass
try:
    ticks = os.sysconf("SC_CLK_TCK")
    start = int(open("/proc/1/stat").read().rsplit(")", 1)[1].split()[19])
    facts["container_seconds"] = float(open("/proc/uptime").read().split()[0]) - start / ticks
except (OSError, ValueError, IndexError):
    pass
try:
    query = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"]
    out = subprocess.run(query, capture_output=True, text=True, timeout=60).stdout
    for line in out.splitlines():
        name, memory, driver = (part.strip() for part in line.split(",")) if line.count(",") == 2 else ("",) * 3
        if name:
            facts["gpus"].append({"name": name, "memory_mib": int(float(memory)), "driver": driver})
    head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", head)
    facts["cuda"] = match.group(1) if match else None
except (OSError, subprocess.SubprocessError):
    pass
if facts["pod_id"] and env("RUNPOD_API_KEY"):
    request = urllib.request.Request(
        "https://api.runpod.io/v2/pods/" + facts["pod_id"],
        headers={"Authorization": "Bearer " + env("RUNPOD_API_KEY"),
                 "User-Agent": "yamabiko-asr-bench"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            pod = json.load(response)
        facts["pod"] = {key: pod.get(key) for key in
                        ("image", "cloud", "dataCenterId", "cost", "disk", "createdAt", "mounts")}
        facts["pod"]["gpu"] = (pod.get("gpu") or {}).get("id")
        facts["pod_api"] = "ok"
    except Exception as error:
        facts["pod_api"] = type(error).__name__ + ": " + str(error)[:200]
print(json.dumps(facts))
"""

# $1 work root, $2 1 for torch, $3 1 for the bench extras, $4 1 for the
# pilot venv ($W/venv-ft with env-ft.sh), $5 1 for p7zip (every pod-task
# pod). The release set (yamabiko/requirements.txt under
# constraints-release.txt) without the Qt UI
# and PyAudioWPatch, which is Windows-only: the chunker imports yamabiko.main
# for its Segment type, and yamabiko.main imports the capture module, so
# pyaudiowpatch gets an import stub that fails on use. The bench never
# captures audio. uv (the image's own binary) downloads in parallel and
# retries each request; the caller retries the whole step. The venvs are
# rebuilt on every attempt, the download cache in the work root is kept.
# unsafe-best-match lets the PyTorch index and PyPI serve the same package,
# as pip does. The pilot venv is synced from its Linux lock (torch's own CUDA
# wheels) and gets its own env-ft.sh, whose library path holds those wheels
# instead of the release ones.
_INSTALL_ENV = r"""set -euo pipefail
W=$1; torch=$2; bench=$3; ft=$4; p7zip=$5
export UV_CACHE_DIR="$W/uv-cache" UV_PYTHON_DOWNLOADS=never UV_PYTHON_PREFERENCE=only-system \
    UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=8
if [ "$p7zip" = 1 ] && ! command -v 7z > /dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    { apt-get update -qq && apt-get install -y -qq p7zip-full; } >&2
fi
rm -rf "$W/venv"
uv venv -q --python python3.12 "$W/venv"
grep -v -i -E '^(PyAudioWPatch|PySide6)' "$W/repo/yamabiko/requirements.txt" > "$W/requirements.txt"
set -- -r "$W/requirements.txt"
if [ "$torch" = 1 ]; then
    # torch pins its own CUDA wheels on Linux; the Windows release pins of
    # those would conflict.
    grep -v -i '^nvidia-' "$W/repo/constraints-release.txt" > "$W/constraints.txt"
    set -- "$@" -r "$W/repo/requirements-bench-transformers.txt"
else
    cp "$W/repo/constraints-release.txt" "$W/constraints.txt"
fi
uv=(uv pip install -q --python "$W/venv/bin/python" --index-strategy unsafe-best-match)
"${uv[@]}" -c "$W/constraints.txt" "$@"
if [ "$bench" = 1 ]; then
    "${uv[@]}" --no-deps -r "$W/repo/requirements-bench.txt"
fi
site=$("$W/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')
mkdir -p "$site/pyaudiowpatch"
cat > "$site/pyaudiowpatch/__init__.py" <<'PY'
# Import stub of the Windows-only PyAudioWPatch for ASR bench pods.


def __getattr__(name):
    raise RuntimeError("PyAudioWPatch is Windows-only; the ASR bench pod has an import stub")
PY
libs=$("$W/venv/bin/python" -c 'import glob, os, site; print(":".join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib")))))')
{
    printf 'export HF_HOME=%q YAMABIKO_APPDATA=%q YAMABIKO_ASR_BENCH_ROOT=%q PYTHONUNBUFFERED=1\n' \
        "$W/hf" "$W/appdata" "$W/bench-root"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$libs"
    printf 'unset HF_HUB_ENABLE_HF_TRANSFER\n'
} > "$W/env.sh"
ftp=
if [ "$ft" = 1 ]; then
    rm -rf "$W/venv-ft"
    uv venv -q --python python3.12 "$W/venv-ft"
    uv pip sync -q --python "$W/venv-ft/bin/python" --torch-backend cu128 \
        "$W/repo/scripts/asr_finetune/requirements-ft.linux.lock.txt"
    ftlibs=$("$W/venv-ft/bin/python" -c 'import glob, os, site; print(":".join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib")))))')
    {
        printf 'export HF_HOME=%q YAMABIKO_APPDATA=%q YAMABIKO_ASR_BENCH_ROOT=%q PYTHONUNBUFFERED=1\n' \
            "$W/hf" "$W/appdata" "$W/bench-root"
        printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$ftlibs"
        printf 'unset HF_HUB_ENABLE_HF_TRANSFER\n'
    } > "$W/env-ft.sh"
    ( . "$W/env-ft.sh" && "$W/venv-ft/bin/python" -c 'import torch, transformers, peft, ctranslate2; raise SystemExit(0 if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() else "torch or CTranslate2 sees no CUDA device in the pilot venv")' )
    "$W/venv-ft/bin/python" -c 'import json, importlib.metadata as m; print(json.dumps(dict(sorted((d.metadata["Name"], d.version) for d in m.distributions()))))' > "$W/ft-packages.json"
    ftp="$W/ft-packages.json"
fi
. "$W/env.sh"
cd "$W/repo"
"$W/venv/bin/python" -c 'import yamabiko.main, scripts.asr_bench.run, ctranslate2; n = ctranslate2.get_cuda_device_count(); raise SystemExit(0 if n else "CTranslate2 sees no CUDA device")'
UV_VERSION=$(uv --version) FT_PACKAGES=$ftp "$W/venv/bin/python" -c 'import json, os, sys, importlib.metadata as m; r = {"python": sys.version.split()[0], "installer": os.environ["UV_VERSION"], "stubs": ["pyaudiowpatch"], "packages": dict(sorted((d.metadata["Name"], d.version) for d in m.distributions()))}; p = os.environ["FT_PACKAGES"]; r.update({"ft_packages": json.load(open(p))} if p else {}); print(json.dumps(r))'
"""


class RemoteError(RuntimeError):
    """A pod, SSH or job step failed."""


class PolicyRefusal(RuntimeError):
    """The approval policy or a data guard refuses the request."""


class CapReached(RemoteError):
    """The hard wall-clock cap ran out."""


# --- RunPod API ---------------------------------------------------------


def api_key() -> tuple[str, str] | None:
    """The API key and where it came from: ``RUNPOD_API_KEY``, else the
    ``apikey`` of runpodctl's ``~/.runpod/config.toml``."""
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if value:
        return value, "RUNPOD_API_KEY"
    path = Path.home() / ".runpod" / "config.toml"
    if path.is_file():
        for key, value in tomllib.loads(path.read_text(encoding="utf-8")).items():
            if key.lower() == "apikey" and isinstance(value, str) and value.strip():
                return value.strip(), "~/.runpod/config.toml"
    return None


def _require_key() -> str:
    found = api_key()
    if found is None:
        raise PolicyRefusal("no RunPod API key: set RUNPOD_API_KEY or ~/.runpod/config.toml apikey")
    return found[0]


class RunpodError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"RunPod API {status}: {detail}")
        self.status = status


class RunpodApi:
    """REST API v2 (v1 is retired on 2026-11-15). Errors never carry the key."""

    def __init__(self, key: str):
        self._http = httpx.Client(
            base_url=REST_URL, timeout=30.0,
            headers={"Authorization": f"Bearer {key}", "User-Agent": "yamabiko-asr-bench"},
        )

    def close(self) -> None:
        self._http.close()

    def _send(self, method: str, path: str, *, retry: bool = True, **kwargs: Any) -> httpx.Response:
        for attempt in range(6):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError:
                if not retry or attempt == 5:
                    raise
                time.sleep(2 ** attempt)
                continue
            if retry and attempt < 5 and (response.status_code == 429 or response.status_code >= 500):
                time.sleep(min(30.0, float(response.headers.get("Retry-After") or 2 ** attempt)))
                continue
            return response
        raise AssertionError("unreachable")

    @staticmethod
    def _check(response: httpx.Response, ok: Sequence[int] = (200, 201)) -> Any:
        if response.status_code in ok:
            return response.json() if response.content else None
        try:
            body = response.json()
            detail = body.get("detail") or body.get("title") or body.get("message") or body
        except ValueError:
            detail = response.text[:300]
        raise RunpodError(response.status_code, str(detail))

    def pods(self) -> list[dict[str, Any]]:
        pods: list[dict[str, Any]] = []
        params: dict[str, str] = {}
        while True:
            page = self._check(self._send("GET", "/pods", params=params))
            pods.extend(page.get("pods", []))
            pagination = page.get("pagination") or {}
            if not pagination.get("hasNextPage"):
                return pods
            params = {"cursor": pagination["nextCursor"]}

    def pod(self, pod_id: str) -> dict[str, Any] | None:
        response = self._send("GET", f"/pods/{pod_id}")
        return None if response.status_code == 404 else self._check(response)

    def create_pod(self, body: Mapping[str, Any]) -> dict[str, Any]:
        # Never retried: a lost response may still have created the pod.
        return self._check(self._send("POST", "/pods", retry=False, json=body), ok=(201, 200))

    def terminate(self, pod_id: str) -> None:
        self._check(self._send("DELETE", f"/pods/{pod_id}"), ok=(200, 204, 404))


@dataclasses.dataclass(frozen=True)
class GpuOffer:
    id: str
    name: str
    memory_gb: int
    usd_per_hour: float
    stock: str | None

    @property
    def consumer(self) -> bool:
        """A consumer GeForce card, the class latency is judged on."""
        return "GeForce" in self.id

    @property
    def in_stock(self) -> bool:
        return self.stock in _STOCK_RANK


def secure_gpus(min_cuda: str) -> list[GpuOffer]:
    """Secure Cloud NVIDIA GPU types with list price and one-GPU stock on
    hosts with at least ``min_cuda``, cheapest first (public GraphQL)."""
    response = httpx.post(GRAPHQL_URL, json={"query": _GPU_QUERY, "variables": {"cuda": min_cuda}},
                          timeout=30.0, headers={"User-Agent": "yamabiko-asr-bench"})
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"GPU price query failed: {payload['errors']}")
    offers = []
    for gpu in payload["data"]["gpuTypes"]:
        if not gpu.get("secureCloud") or not gpu.get("securePrice"):
            continue
        if str(gpu.get("manufacturer", "")).lower() != "nvidia":
            continue
        offers.append(GpuOffer(gpu["id"], gpu["displayName"], int(gpu["memoryInGb"]),
                               float(gpu["securePrice"]), (gpu.get("lowestPrice") or {}).get("stockStatus")))
    return sorted(offers, key=lambda o: (o.usd_per_hour, -_STOCK_RANK.get(o.stock or "", 0), o.id))


def choose_gpu(offers: Sequence[GpuOffer], wanted: str | None, min_vram: int,
               allow_datacenter: bool) -> GpuOffer:
    if wanted:
        match = next((o for o in offers if wanted in (o.id, o.name)), None)
        if match is None:
            raise PolicyRefusal(f"{wanted!r} is not a Secure Cloud NVIDIA GPU type (see `gpus --all`)")
        if not match.consumer and not allow_datacenter:
            raise PolicyRefusal(f"{match.id} is not a consumer GeForce card; it needs --allow-datacenter "
                                "and --approve-cost after the owner's OK")
        return match
    pool = [o for o in offers if o.in_stock and o.memory_gb >= min_vram
            and (o.consumer or allow_datacenter)]
    if not pool:
        kind = "NVIDIA" if allow_datacenter else "consumer GeForce"
        raise RemoteError(f"no {kind} Secure Cloud GPU with {min_vram} GB is in stock")
    return pool[0]


# --- Approval policy and spend log ---------------------------------------


@dataclasses.dataclass(frozen=True)
class Authorization:
    basis: str
    cap_usd: float
    cap_hours: float
    day_cap_usd: float = ROUTINE_DAY_USD

    @property
    def day_cap_raised(self) -> bool:
        return self.day_cap_usd > ROUTINE_DAY_USD


def day_cap_value(text: str) -> float:
    """``--day-cap``: a positive amount of at most ``DAY_CAP_MAX_USD``."""
    value = float(text)
    if not 0 < value <= DAY_CAP_MAX_USD:
        raise argparse.ArgumentTypeError(f"the day cap must be above $0 and at most ${DAY_CAP_MAX_USD:.2f}, "
                                         f"not ${value:.2f}")
    return value


def authorize(*, estimate_usd: float, estimate_hours: float, usd_per_hour: float,
              consumer: bool, today_usd: float, approve: float | None,
              day_cap_usd: float = ROUTINE_DAY_USD) -> Authorization:
    """The standing policy: a consumer GPU, at most ``ROUTINE_JOB_USD`` and a
    day total at most the day cap (``ROUTINE_DAY_USD`` unless the owner
    raised it with ``--day-cap``) run without a flag; anything else
    needs ``approve`` (the owner's OK in USD). ``today_usd`` is today's spend
    with every open pod at the most it can cost (``spend_totals(at_most=True)``).
    The cap is the earlier of ``CUTOFF_FACTOR`` times the estimate and the
    time the money runs out."""
    reasons = []
    if not consumer:
        reasons.append("a datacenter GPU")
    if estimate_usd > ROUTINE_JOB_USD:
        reasons.append(f"an estimate of ${estimate_usd:.2f} over the ${ROUTINE_JOB_USD:.2f} job cap")
    if today_usd + estimate_usd > day_cap_usd:
        reasons.append(f"a day total of ${today_usd + estimate_usd:.2f} over the "
                       f"${day_cap_usd:.2f} day cap")
    if approve is None:
        if reasons:
            raise PolicyRefusal("not routine (" + "; ".join(reasons) + "): needs --approve-cost "
                                "after the owner's OK")
        cap_usd = min(ROUTINE_JOB_USD, day_cap_usd - today_usd)
        basis = "routine"
    else:
        if estimate_usd > approve:
            raise PolicyRefusal(f"the estimate ${estimate_usd:.2f} exceeds the approved ${approve:.2f}")
        cap_usd = approve
        basis = f"approved ${approve:.2f}"
    cap_hours = min(CUTOFF_FACTOR * estimate_hours, cap_usd / usd_per_hour)
    return Authorization(basis, cap_usd, cap_hours, day_cap_usd)


def _spend_dir() -> Path:
    return bench_root() / "runpod"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _epoch(moment: datetime) -> int:
    return int(moment.timestamp())


def _whole_second(moment: datetime) -> datetime:
    return datetime.fromtimestamp(_epoch(moment), timezone.utc)


def _open_key(record: Mapping[str, Any]) -> str:
    """API-mode records are keyed by the pod name, written before the create
    call; SSH-mode records and those of spend schema 1.0 by the pod id."""
    return record.get("name") or record["pod_id"]


def _open_path(record: Mapping[str, Any]) -> Path:
    return _spend_dir() / "open" / f"{_open_key(record)}.json"


def _write_open(record: Mapping[str, Any]) -> None:
    """A pod in flight, or being created; it counts toward the day until its
    record is closed."""
    path = _open_path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


def _discard_open(record: Mapping[str, Any]) -> None:
    _open_path(record).unlink(missing_ok=True)


def _same_pod(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if first.get("name") and second.get("name"):
        return first["name"] == second["name"]
    return (bool(first.get("pod_id")) and first.get("pod_id") == second.get("pod_id")
            and first.get("start_utc") == second.get("start_utc"))


def _close_pod(record: Mapping[str, Any]) -> None:
    """Append the pod's one record to the spend log (once, whoever closes it
    first) and drop its open file."""
    closed, _ = _spend_records()
    if not any(_same_pod(record, other) for other in closed):
        with (_spend_dir() / "spend.jsonl").open("a", encoding="utf-8", newline="\n") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
    _discard_open(record)


def _spend_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    log = _spend_dir() / "spend.jsonl"
    closed = []
    if log.is_file():
        closed = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    folder = _spend_dir() / "open"
    opened = []
    if folder.is_dir():
        opened = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]
    return closed, opened


def _bound(record: Mapping[str, Any]) -> datetime:
    """The latest time the pod can bill: its watchdog deadline (start plus
    cap for spend schema 1.0)."""
    if record.get("deadline_utc"):
        return _parse_iso(record["deadline_utc"])
    return _parse_iso(record["start_utc"]) + timedelta(hours=float(record.get("cap_hours") or 0.0))


def _cost(record: Mapping[str, Any], until: datetime) -> float:
    hours = max(0.0, (until - _parse_iso(record["start_utc"])).total_seconds() / 3600)
    return float(record["usd_per_hour"]) * hours


def _open_cost(record: Mapping[str, Any], now: datetime, *, at_most: bool) -> float:
    """An open pod's cost so far, or (``at_most``) the most it can cost: its
    rate up to its deadline, or up to now for a pod past its deadline."""
    bound = _bound(record)
    return _cost(record, max(bound, now) if at_most else min(bound, now))


def spend_totals(now: datetime | None = None, *, at_most: bool = False
                 ) -> tuple[float, float, list[dict[str, Any]]]:
    """Spend today and this month (local calendar, by pod start) and the open
    records. Closed pods count at their actual cost; open pods at their cost
    so far or, with ``at_most`` (what approvals use), at the most they can
    cost, so parallel runs cannot pass the day cap together."""
    now = now or _now()
    local = now.astimezone()
    closed, opened = _spend_records()
    today = month = 0.0
    costs = [float(record["actual_usd"]) for record in closed]
    costs += [_open_cost(record, now, at_most=at_most) for record in opened]
    for record, cost in zip(closed + opened, costs):
        start = _parse_iso(record["start_utc"]).astimezone()
        if (start.year, start.month) == (local.year, local.month):
            month += cost
            if start.date() == local.date():
                today += cost
    return today, month, opened


@contextlib.contextmanager
def _approval_lock():
    """Held around a run's approval and its pod create (or, in SSH mode, its
    open record), so parallel runs each see the others' open records."""
    _spend_dir().mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(_spend_dir() / "approve.lock"))
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout:
        print("waiting for another run's approval and pod create", flush=True)
        try:
            lock.acquire(timeout=_LOCK_TIMEOUT_S)
        except filelock.Timeout:
            raise RemoteError(f"{lock.lock_file} stayed locked for {_LOCK_TIMEOUT_S:.0f} s") from None
    try:
        yield
    finally:
        lock.release()


def _close_at(record: Mapping[str, Any], end: datetime, source: str, outcome: str | None = None) -> None:
    """Close an open record (its latest content on disk) at ``end``."""
    path = _open_path(record)
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
    _close_pod({**record, "end_utc": _iso(end), "end_source": source,
                "outcome": outcome or record.get("outcome") or "unknown",
                "actual_usd": round(_cost(record, end), 4), "actual_source": "runtime x rate"})


def _settle(api: RunpodApi | None) -> None:
    """Close the open records whose pod is gone (call under the approval lock).

    With the API: a record without a pod id is left by a create whose stray
    check failed; a pod listed under its name is an orphan and is terminated.
    A record whose pod the API no longer lists is closed at the earlier of now
    and its deadline. A pod still listed past its deadline is terminated if
    this tool created it, reported otherwise. Without the API, a record past
    its deadline by ``_SETTLE_GRACE`` is closed at its deadline.
    """
    _, opened = _spend_records()
    if not opened:
        return
    now = _now()
    if api is None:
        for record in opened:
            if now > _bound(record) + _SETTLE_GRACE:
                _close_at(record, _bound(record), "deadline passed; not checked against the API")
        return
    pods = api.pods()
    by_id = {pod["id"]: pod for pod in pods}
    by_name = {pod.get("name"): pod for pod in pods if pod.get("name")}
    gone = []
    for record in opened:
        pod_id = record.get("pod_id")
        if not pod_id:
            orphan = by_name.get(record.get("name"))
            if orphan is None:
                gone.append(record)
                continue
            print(f"pod {orphan['id']} ({record['name']}) is left from an interrupted create; "
                  "terminating it", flush=True)
            record = {**record, "pod_id": orphan["id"], "state": "orphan"}
            _write_open(record)
            if _terminate_confirmed(api, orphan["id"]):
                _close_at(record, _now(), "orphan of an interrupted create, terminated", "failed")
            continue
        if pod_id not in by_id:
            gone.append(record)
        elif now > _bound(record) + _SETTLE_GRACE:
            if record.get("mode") == "api" and str(by_id[pod_id].get("name", "")).startswith("ybench-"):
                print(f"pod {pod_id} is still running past its deadline {_iso(_bound(record))}; "
                      "terminating it", flush=True)
                if _terminate_confirmed(api, pod_id):
                    _close_at(record, _now(), "terminated past its deadline")
            else:
                print(f"WARNING: pod {pod_id} is still running past its deadline {_iso(_bound(record))}; "
                      f"terminate it (`remote terminate {pod_id}`)", flush=True)
    if gone:
        # A live client closes its record right after terminating its pod.
        time.sleep(_STRAY_PAUSE_S)
        for record in gone:
            if _open_path(record).is_file():
                _close_at(record, min(now, _bound(record)), "pod no longer listed; end bounded by its deadline")


def _new_record(*, name: str | None, pod_id: str | None, mode: str, gpu: str, compute_usd_per_hour: float,
                disk_gb: int, start: datetime, deadline: datetime, estimated_usd: float,
                auth: Authorization, jobs: Iterable[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": _SPEND_SCHEMA, "name": name, "pod_id": pod_id, "mode": mode,
        "state": "creating" if pod_id is None else "running", "gpu": gpu,
        "usd_per_hour": compute_usd_per_hour + disk_gb * _DISK_USD_PER_GB_HOUR,
        "compute_usd_per_hour": compute_usd_per_hour, "disk_gb": disk_gb,
        "start_utc": _iso(start), "deadline_utc": _iso(deadline), "end_utc": None,
        "estimated_usd": round(estimated_usd, 4), "cap_usd": auth.cap_usd, "cap_hours": auth.cap_hours,
        "approval": auth.basis, "jobs": list(jobs), "actual_usd": None,
    }
    if auth.day_cap_raised:
        record.update(day_cap_usd=auth.day_cap_usd, day_cap_basis="owner-approved --day-cap")
    return record


# --- Jobs and uploads --------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JobSpec:
    id: str | None
    set: Path
    candidate: Path
    name: str
    out: Path
    assume_rtf: float | None = None
    # A positive int or AUTO; None takes --workers.
    workers: int | str | None = None
    # CTranslate2's sampler seed (scripts.asr_finetune.seeded_bench_run);
    # None runs scripts.asr_bench.run unseeded.
    seed: int | None = None


_SEED_MAX = 2**32 - 1


def _seed_refusal(where: str, workers: int | str) -> PolicyRefusal:
    return PolicyRefusal(f"{where}: a seeded job needs workers 1, not {workers!r} (spawned worker "
                         "processes would not inherit the seed)")


def workers_value(value: Any) -> int | str:
    """A job's worker count: a positive int or ``AUTO``."""
    if value == AUTO:
        return AUTO
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"workers must be a positive integer or {AUTO!r}, not {value!r}")
    return value


def read_job_specs(job_args: Sequence[Sequence[str]] | None, jobs_file: Path | None) -> list[JobSpec]:
    specs = [JobSpec(None, Path(s), Path(c), n, Path(o)) for s, c, n, o in job_args or ()]
    if jobs_file is not None:
        raw = json.loads(jobs_file.read_text(encoding="utf-8"))
        entries = raw.get("jobs") if isinstance(raw, Mapping) else raw
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{jobs_file}: expected {{\"jobs\": [...]}} with at least one job")
        base = jobs_file.resolve().parent
        for index, entry in enumerate(entries):
            unknown = set(entry) - {"id", "set", "candidate", "name", "out", "assume_rtf", "workers", "seed"}
            missing = {"set", "candidate", "name", "out"} - set(entry)
            if unknown or missing:
                raise ValueError(f"{jobs_file} job {index}: unknown keys {sorted(unknown)}, "
                                 f"missing keys {sorted(missing)}")
            rtf = entry.get("assume_rtf")
            if rtf is not None and (isinstance(rtf, bool) or not isinstance(rtf, (int, float)) or rtf <= 0):
                raise ValueError(f"{jobs_file} job {index}: assume_rtf must be a positive number")
            try:
                workers = None if entry.get("workers") is None else workers_value(entry["workers"])
            except ValueError as error:
                raise ValueError(f"{jobs_file} job {index}: {error}") from None
            seed = entry.get("seed")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                                     or not 0 <= seed <= _SEED_MAX):
                raise ValueError(f"{jobs_file} job {index}: seed must be an integer from 0 to {_SEED_MAX}")
            if seed is not None and workers not in (None, 1):
                raise _seed_refusal(f"{jobs_file} job {index}", workers)
            specs.append(JobSpec(entry.get("id"), base / entry["set"], base / entry["candidate"],
                                 entry["name"], base / entry["out"], None if rtf is None else float(rtf),
                                 workers, seed))
    if not specs:
        raise ValueError("give --jobs FILE or at least one --job")
    return specs


def forbidden_roots() -> list[Path]:
    """What never leaves this machine by location: the E0 private roots
    (``YAMABIKO_E0_PRIVATE_ROOT`` and ``%LOCALAPPDATA%\\Yamabiko\\e0-private``)
    and the bench ``config`` folder (the private locked-title list). ``Guard``
    adds every directory holding an E0 root manifest and every set derived
    from E0."""
    roots = [bench_root() / "config"]
    if os.environ.get("YAMABIKO_E0_PRIVATE_ROOT"):
        roots.append(Path(os.environ["YAMABIKO_E0_PRIVATE_ROOT"]))
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "Yamabiko" / "e0-private")
    return [root.resolve() for root in roots]


def e0_derivation(set_json: Path, seen: frozenset[Path] = frozenset()) -> str | None:
    """Why a set is derived from E0 data, or None: its manifest records
    ``e0_state_sha256`` (``build_silver.py``) or E0 ``fixture_label``s, or it
    is (through ``build.subset_of``, by set id under the bench root) a subset
    of such a set."""
    text = set_json.read_text(encoding="utf-8")
    if '"e0_state_sha256"' in text:
        return "its set.json records e0_state_sha256 (built from the E0 masters)"
    if '"fixture_label"' in text:
        return "its items carry E0 fixture labels"
    if '"subset_of"' not in text:
        return None
    parent = (json.loads(text).get("build") or {}).get("subset_of")
    if not isinstance(parent, str) or not parent:
        return None
    source = bench_root() / "sets" / parent / "set.json"
    if not source.is_file() or source.resolve() in seen:
        return None
    reason = e0_derivation(source, seen | {set_json.resolve()})
    return None if reason is None else f"it is a subset of {parent}, and {reason}"


_FileKey = tuple[int, int]


def _file_key(path: Path) -> _FileKey | None:
    """A file's identity (device or volume serial, inode or file index), the
    same for every spelling of its path (``\\\\?\\``, admin shares, links);
    None when it cannot be read or its file system has no file index."""
    try:
        status = os.stat(path)
    except (OSError, ValueError):
        return None
    return (status.st_dev, status.st_ino) if status.st_ino else None


class Guard:
    """Refuses every upload of private data, by path spelling and by file
    identity (``os.path.samefile``'s test):

    - a path inside, equal to or holding a forbidden root
      (``forbidden_roots``) or the directory of a set derived from E0
      (``e0_derivation``) under the bench root;
    - a path inside or holding a directory with an E0 root manifest
      (``root.private.json``), wherever it is;
    - a hard link to a file of an E0-derived set;
    - (``check_set``) any set derived from E0, wherever it is.
    """

    def __init__(self) -> None:
        self._e0_sets: dict[Path, str] = {}
        sets_dir = bench_root() / "sets"
        if sets_dir.is_dir():
            for set_json in sorted(sets_dir.glob("*/set.json")):
                reason = e0_derivation(set_json)
                if reason:
                    self._e0_sets[set_json.parent.resolve()] = reason
        self._roots = forbidden_roots() + list(self._e0_sets)
        self._root_keys: dict[_FileKey, Path] = {}
        self._chain_keys: dict[_FileKey, Path] = {}
        for root in self._roots:
            if (key := _file_key(root)) is not None:
                self._root_keys[key] = root
            for directory in (root, *root.parents):
                if (key := _file_key(directory)) is not None:
                    self._chain_keys.setdefault(key, root)
        self._e0_files: dict[_FileKey, Path] = {}
        for directory in self._e0_sets:
            for item in directory.rglob("*"):
                if (key := _file_key(item)) is not None:
                    self._e0_files[key] = directory
        self._dir_keys: dict[Path, _FileKey | None] = {}
        self._manifests: dict[Path, bool] = {}

    def _key(self, directory: Path) -> _FileKey | None:
        if directory not in self._dir_keys:
            self._dir_keys[directory] = _file_key(directory)
        return self._dir_keys[directory]

    def _has_manifest(self, directory: Path) -> bool:
        if directory not in self._manifests:
            self._manifests[directory] = (directory / ROOT_MANIFEST_NAME).is_file()
        return self._manifests[directory]

    def check(self, path: Path) -> Path:
        """``path`` resolved, or ``PolicyRefusal``."""
        resolved = Path(path).resolve()

        def refuse(why: str) -> None:
            raise PolicyRefusal(f"refusing to upload {path}: {why}")

        for root in self._roots:
            if resolved == root or root in resolved.parents or resolved in root.parents:
                refuse(f"it overlaps the private root {root}")
        own = _file_key(resolved)
        if own is not None and own in self._chain_keys:
            refuse(f"it is, or holds, the private root {self._chain_keys[own]}")
        if own is not None and own in self._e0_files:
            refuse(f"it is a link to a file of {self._e0_files[own]}, a set derived from E0")
        for directory in (resolved, *resolved.parents):
            key = own if directory == resolved else self._key(directory)
            if key is not None and key in self._root_keys:
                refuse(f"it lies in the private root {self._root_keys[key]}")
            if self._has_manifest(directory):
                refuse(f"{directory} holds an E0 root manifest ({ROOT_MANIFEST_NAME})")
        if resolved.is_dir():
            manifest = next(resolved.rglob(ROOT_MANIFEST_NAME), None)
            if manifest is not None:
                refuse(f"it holds an E0 root manifest ({manifest})")
        return resolved

    def check_set(self, bench_set: BenchSet) -> None:
        reason = e0_derivation(bench_set.root / "set.json")
        if reason:
            raise PolicyRefusal(f"refusing to upload set {bench_set.set_id} ({bench_set.root}): "
                                f"it is derived from E0 data: {reason}")


@dataclasses.dataclass(frozen=True)
class Upload:
    label: str
    remote: str
    files: tuple[tuple[Path, str], ...]
    size: int


class Uploads:
    """Everything the jobs need on the pod, deduplicated, and the map from
    remote paths back to the local strings they replace."""

    def __init__(self, root: str, guard: Guard):
        self._root = root
        self._guard = guard
        self._by_key: dict[tuple[Any, ...], Upload] = {}
        self.items: list[Upload] = []
        self.paths: dict[str, str] = {}

    def _add(self, key: tuple[Any, ...], label: str, files: list[tuple[Path, str]]) -> Upload:
        if key not in self._by_key:
            slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)[:60]
            upload = Upload(label, f"{self._root}/data/{len(self.items) + 1}-{slug}", tuple(files),
                            sum(path.stat().st_size for path, _ in files))
            self._by_key[key] = upload
            self.items.append(upload)
        return self._by_key[key]

    def add_set(self, bench_set: BenchSet) -> str:
        self._guard.check_set(bench_set)
        check = self._guard.check
        root = check(bench_set.root)
        files = [(check(root / "set.json"), "set.json")]
        files += [(check(bench_set.audio_path(item)), item.audio.replace("\\", "/"))
                  for item in bench_set.items]
        upload = self._add(("set", root), bench_set.set_id, files)
        self.paths[upload.remote] = str(root)
        return upload.remote

    def add_path(self, value: str) -> str:
        """A whole file or directory (a model, a VAD graph)."""
        path = self._guard.check(Path(value))
        if path.is_file():
            upload = self._add(("file", path), path.name, [(path, path.name)])
            remote = f"{upload.remote}/{path.name}"
        elif path.is_dir():
            files = sorted((self._guard.check(item), item.relative_to(value).as_posix())
                           for item in Path(value).rglob("*") if item.is_file())
            if not files:
                raise ValueError(f"{value} is empty")
            upload = self._add(("dir", path), path.name, files)
            remote = upload.remote
        else:
            raise ValueError(f"{value} does not exist")
        self.paths[remote] = value
        return remote

    def add_items(self, value: str, suffix: str, bench_set: BenchSet) -> str:
        """The per-item files of a frontend or tags directory for one set, plus
        the directory's top-level JSON metadata (``manifest.json``: the runner
        reads the tag layout from it)."""
        directory = self._guard.check(Path(value))
        files = []
        for item in bench_set.items:
            path = directory / f"{item.item_id}{suffix}"
            if not path.is_file():
                raise ValueError(f"{path} is missing")
            files.append((self._guard.check(path), path.name))
        files.extend((self._guard.check(path), path.name) for path in sorted(directory.glob("*.json"))
                     if path.is_file())
        upload = self._add(("items", directory, suffix, bench_set.root.resolve()),
                           f"{directory.name}-{bench_set.set_id}", files)
        self.paths[upload.remote] = value
        return upload.remote

    def rewrite(self, raw: Any, bench_set: BenchSet, key: str | None = None) -> Any:
        """The candidate with every absolute local path replaced by its upload."""
        if isinstance(raw, Mapping):
            return {k: self.rewrite(v, bench_set, k) for k, v in raw.items()}
        if isinstance(raw, list):
            return [self.rewrite(v, bench_set, key) for v in raw]
        if isinstance(raw, str) and Path(raw).is_absolute():
            if key == "audio_dir":
                return self.add_items(raw, ".f32", bench_set)
            if key == "tags_dir":
                return self.add_items(raw, ".npz", bench_set)
            return self.add_path(raw)
        return raw

    @property
    def size(self) -> int:
        return sum(upload.size for upload in self.items)


@dataclasses.dataclass(frozen=True)
class PastTiming:
    rtf: float
    load_seconds: float
    source: str


@dataclasses.dataclass
class Job:
    index: int
    id: str
    bench_set: BenchSet
    candidate: Candidate
    out: Path
    remote_set: str
    remote_candidate: dict[str, Any]
    backends: frozenset[str]
    hf_models: frozenset[str]
    timing: PastTiming | None = None
    assume_rtf: float | None = None
    # As given (an int or AUTO); bytes of the ASR model files, None when
    # not on this machine; the choice the estimate used and the one run.
    workers: int | str = 1
    model_bytes: int | None = None
    plan_workers: WorkerChoice | None = None
    run_workers: WorkerChoice | None = None
    seed: int | None = None

    @property
    def audio_seconds(self) -> float:
        return sum(item.duration_s for item in self.bench_set.items)


@dataclasses.dataclass(frozen=True)
class WorkerChoice:
    workers: int
    # Recorded in the run's remote block (workers_choice).
    basis: dict[str, Any]
    text: str


def choose_workers(requested: int | str, *, vram_mib: float | None, vcpus: int | None,
                   model_bytes: int | None, items: int, vcpus_source: str) -> WorkerChoice:
    """A given count as it is; ``AUTO``: as many processes as the GPU's
    memory holds models for (``_WORKER_OVERHEAD_MIB`` and ``_TAGGER_MIB``
    each besides the model files, ``_VRAM_HEADROOM_MIB`` kept free) and its
    vCPUs feed (``_VCPUS_PER_WORKER`` each), at most ``_AUTO_WORKERS_MAX``
    and the job's item count; 1 when a fact is unknown."""
    if requested != AUTO:
        return WorkerChoice(int(requested), {"requested": requested, "workers": requested}, f"{requested} (given)")
    unknown = [name for name, value in (("GPU memory", vram_mib), ("vCPUs", vcpus),
                                        ("model size", model_bytes)) if value is None]
    if unknown:
        return WorkerChoice(1, {"requested": AUTO, "workers": 1, "unknown": unknown},
                            f"1 (auto: {', '.join(unknown)} unknown)")
    assert vram_mib is not None and vcpus is not None and model_bytes is not None
    per_worker = model_bytes / 2**20 + _WORKER_OVERHEAD_MIB + _TAGGER_MIB
    by_vram = int((vram_mib - _VRAM_HEADROOM_MIB) // per_worker)
    by_vcpus = vcpus // _VCPUS_PER_WORKER
    workers = max(1, min(by_vram, by_vcpus, _AUTO_WORKERS_MAX, items))
    basis = {
        "requested": AUTO, "workers": workers, "vram_mib": vram_mib, "headroom_mib": _VRAM_HEADROOM_MIB,
        "per_worker_mib": round(per_worker), "fit_vram": by_vram, "vcpus": vcpus,
        "vcpus_source": vcpus_source, "vcpus_per_worker": _VCPUS_PER_WORKER, "fit_vcpus": by_vcpus,
        "max": _AUTO_WORKERS_MAX, "items": items,
    }
    text = (f"{workers} (auto: {vram_mib:.0f} MiB less {_VRAM_HEADROOM_MIB} fits {by_vram} at "
            f"{per_worker:.0f} MiB each, {vcpus} vCPUs ({vcpus_source}) feed {by_vcpus}, "
            f"at most {_AUTO_WORKERS_MAX} and {items} items)")
    return WorkerChoice(workers, basis, text)


def assumed_speedup(workers: int) -> float:
    """Throughput of ``workers`` processes over one, before a run measured it."""
    return 1.0 + _ASSUMED_SPEEDUP_PER_WORKER * (workers - 1)


def _hf_snapshot(repo: str) -> Path | None:
    """The local Hugging Face cache's snapshot of ``repo`` at the revision
    production pins (else the one ``refs/main`` names), or None."""
    from yamabiko import hf_env  # noqa: F401  # HF_HOME as the app sets it
    from yamabiko.models import whisper_revision_for

    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ["HF_HOME"]) / "hub")
    folder = hub / ("models--" + repo.replace("/", "--"))
    revision = whisper_revision_for(repo)
    if revision is None and (folder / "refs" / "main").is_file():
        revision = (folder / "refs" / "main").read_text(encoding="utf-8").strip()
    snapshot = folder / "snapshots" / revision if revision else None
    return snapshot if snapshot is not None and snapshot.is_dir() else None


def model_bytes(candidate: Candidate) -> int | None:
    """Bytes of the ASR model files a candidate's process loads (its own
    and an island decoder's), from local directories or the local Hugging
    Face cache; None when one is not on this machine."""
    specs = [candidate.asr]
    island_asr = getattr(candidate.strategy, "asr", None)
    if isinstance(island_asr, AsrSpec):
        specs.append(island_asr)
    total = 0
    for spec in specs:
        directory = spec.local_model_dir or _hf_snapshot(spec.model)
        if directory is None or not directory.is_dir():
            return None
        total += sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    return total


def _raw_candidate(path: Path, name: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "candidates" in raw:
        return next(entry for entry in raw["candidates"] if entry.get("name") == name)
    return raw


def _models(raw: Any, backends: set[str], hf: set[str]) -> None:
    if isinstance(raw, Mapping):
        backend = raw.get("backend")
        if isinstance(backend, str):
            backends.add(backend)
            model = raw.get("model")
            if backend == "faster_whisper" and isinstance(model, str) and not Path(model).is_absolute():
                hf.add(model)
        for value in raw.values():
            _models(value, backends, hf)
    elif isinstance(raw, list):
        for value in raw:
            _models(value, backends, hf)


def load_jobs(specs: Sequence[JobSpec], uploads: Uploads, workers: int | str = 1) -> list[Job]:
    """``workers``: the count of a job that gives none."""
    jobs = []
    outs: set[Path] = set()
    for index, spec in enumerate(specs, 1):
        bench_set = load_set(spec.set)
        candidate = load_candidate(spec.candidate, spec.name)
        out = spec.out.resolve()
        if out.exists():
            raise ValueError(f"{out} already exists; choose a new run directory")
        if out in outs:
            raise ValueError(f"two jobs write {out}")
        outs.add(out)
        raw = _raw_candidate(spec.candidate, spec.name)
        backends: set[str] = set()
        hf: set[str] = set()
        _models(raw, backends, hf)
        job_workers = workers if spec.workers is None else spec.workers
        if spec.seed is not None and job_workers != 1:
            raise _seed_refusal(f"job {index}", job_workers)
        jobs.append(Job(
            index=index, id=spec.id or f"{bench_set.set_id}/{candidate.name}", bench_set=bench_set,
            candidate=candidate, out=out, remote_set=uploads.add_set(bench_set),
            remote_candidate=uploads.rewrite(raw, bench_set), backends=frozenset(backends),
            hf_models=frozenset(hf), assume_rtf=spec.assume_rtf, workers=job_workers,
            model_bytes=model_bytes(candidate) if job_workers == AUTO else None, seed=spec.seed,
        ))
    return jobs


def _past_runs() -> list[tuple[Path, dict[str, Any]]]:
    runs = []
    for path in sorted((bench_root() / "runs").rglob("run.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "merge" in record or not isinstance(record.get("timing"), Mapping):
            continue
        runs.append((path.parent, record))
    return runs


def past_timing(candidate: Candidate, set_id: str, gpu_id: str,
                runs: Sequence[tuple[Path, dict[str, Any]]], workers: int = 1) -> PastTiming | None:
    """Timing of the newest past run of this candidate: a remote run on the
    same GPU type first, then a local run, same set first. Remote runs on
    other GPU types are not used. A run with ``workers`` processes gives
    its measured wall-clock throughput (run schema 1.3; the timing of an
    earlier run, which ran in one process); without one, a one-process run's
    divided by ``assumed_speedup(workers)``, its load times ``workers``."""
    target = candidate.to_json()

    def remote_gpu(record: Mapping[str, Any]) -> bool:
        remote = record.get("remote") or {}
        return gpu_id in (remote.get("gpu_type"), remote.get("gpu_model"))

    def matches(record: Mapping[str, Any], exact: bool) -> bool:
        if record.get("remote") is not None and not remote_gpu(record):
            return False
        past = record.get("candidate") or {}
        if exact:
            return past == target
        return past.get("name") == target["name"] and past.get("asr") == target["asr"]

    def newest(count: int) -> tuple[Path, dict[str, Any]] | None:
        same = [(d, r) for d, r in runs if int(r.get("workers") or 1) == count]
        pool = [(d, r) for d, r in same if matches(r, True)] or [(d, r) for d, r in same if matches(r, False)]
        if not pool:
            return None
        return max(pool, key=lambda dr: (
            remote_gpu(dr[1]), dr[1]["set"]["set_id"] == set_id, dr[1].get("created_at_utc", "")))

    def rate(record: Mapping[str, Any]) -> tuple[float, float]:
        throughput = record.get("throughput")
        if throughput:
            return float(throughput["rtf"]), float(throughput["load_wall_seconds"])
        return float(record["timing"]["rtf"]), float(record["timing"]["load_seconds"])

    found = newest(workers)
    if found is not None:
        rtf, load = rate(found[1])
        return PastTiming(rtf, load, f"{found[0]} ({workers} worker(s), measured)")
    found = newest(1) if workers > 1 else None
    if found is None:
        return None
    rtf, load = rate(found[1])
    speedup = assumed_speedup(workers)
    return PastTiming(rtf / speedup, load * workers,
                      f"{found[0]} (1 worker) / assumed speedup {speedup:.2f}x for {workers} workers")


@dataclasses.dataclass(frozen=True)
class Extras:
    torch: bool
    bench: bool

    @staticmethod
    def of(jobs: Iterable[Job]) -> Extras:
        jobs = list(jobs)
        backends = set().union(*(job.backends for job in jobs))
        return Extras(
            torch="transformers_asr" in backends,
            bench=bool(backends & {"parakeet_onnx", "sherpa_transducer"}),
        )

    @property
    def min_cuda(self) -> str:
        return MIN_CUDA_TORCH if self.torch else MIN_CUDA


# --- Pod tasks -----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """One pod task of a ``--tasks`` file: ``python -m module args`` in the
    release venv or the pilot venv (env ``ft``), with ``inputs`` uploaded to
    ``{in}`` and ``secrets`` written to its stdin; ``{out}`` comes back to
    ``out``."""

    id: str
    module: str                  # starts with TASK_MODULE_PREFIX
    env: str                     # "release" or "ft"
    args: tuple[str, ...]        # may hold {in} {out} {data} {repo} {deadline}
    inputs: tuple[Path, ...]     # each passes outbox.check_outbox_file
    secrets: tuple[str, ...]     # subset of SECRET_NAMES
    hours: Mapping[str, float]   # GPU type id -> declared hours; only these GPUs are eligible
    out: Path                    # local directory that receives {out}
    mdc_dataset: str | None = None  # needed when "MDC_URL" is in secrets


@dataclasses.dataclass(frozen=True)
class TaskWork:
    """What does not depend on the GPU: the tasks, the container disk they
    declare, code."""

    tasks: list[TaskSpec]
    disk_gb: int
    commit: str
    archive_bytes: int

    @property
    def extras(self) -> Extras:
        """torch when any task runs in the pilot venv (its torch cu128 needs
        a CUDA 12.8 host); the bench extras never."""
        return Extras(torch=any(task.env == "ft" for task in self.tasks), bench=False)

    @property
    def gpu_ids(self) -> frozenset[str]:
        """The GPU types every task declares hours for: the only eligible ones."""
        return frozenset.intersection(*(frozenset(task.hours) for task in self.tasks))


_TASK_KEYS = {"id", "module", "env", "args", "inputs", "secrets", "hours", "out", "mdc_dataset"}
_TASK_REQUIRED = {"id", "module", "env", "args", "hours", "out"}
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TASK_MODULE = re.compile(re.escape(TASK_MODULE_PREFIX) + r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", re.ASCII)
_MDC_DATASET = re.compile(r"[a-z0-9]+")
_PLACEHOLDER = re.compile(r"\{(?:" + "|".join(_PLACEHOLDERS) + r")\}")


def _string_list(value: Any, where: str, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{where}: {key} must be a list of strings")
    return value


def _input_name(path: Path) -> str:
    """Where an input lies under ``{in}``: its path under the outbox."""
    return path.resolve().relative_to(outbox_dir().resolve()).as_posix()


def _positive(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def read_task_specs(path: Path, commit: str = "HEAD") -> TaskWork:
    """A tasks file, ``{"disk_gb": int, "tasks": [...]}``, with paths relative
    to the file; every input passes ``check_outbox_file`` (else
    ``OutboxRefusal``). ``commit`` is the code revision the pod gets."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"disk_gb", "tasks"}:
        raise ValueError(f'{path}: expected {{"disk_gb": int, "tasks": [...]}}')
    disk_gb, entries = raw["disk_gb"], raw["tasks"]
    if isinstance(disk_gb, bool) or not isinstance(disk_gb, int) or disk_gb < 1:
        raise ValueError(f"{path}: disk_gb must be a positive integer")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: tasks must list at least one task")
    base = path.resolve().parent
    tasks: list[TaskSpec] = []
    for index, entry in enumerate(entries):
        where = f"{path} task {index}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: expected an object")
        unknown, missing = set(entry) - _TASK_KEYS, _TASK_REQUIRED - set(entry)
        if unknown or missing:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        task_id, module, env, out = entry["id"], entry["module"], entry["env"], entry["out"]
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise ValueError(f"{where}: id must be letters, digits, '.', '_' or '-'")
        if any(task.id == task_id for task in tasks):
            raise ValueError(f"{where}: two tasks are named {task_id}")
        if not isinstance(module, str) or not _TASK_MODULE.fullmatch(module):
            raise ValueError(f"{where}: module must name a module under {TASK_MODULE_PREFIX}")
        if env not in _TASK_ENVS:
            raise ValueError(f"{where}: env must be one of {', '.join(_TASK_ENVS)}")
        args = _string_list(entry["args"], where, "args")
        for arg in args:
            if "{" in _PLACEHOLDER.sub("", arg) or "}" in _PLACEHOLDER.sub("", arg):
                raise ValueError(f"{where}: {arg!r} holds a placeholder other than "
                                 + ", ".join("{" + name + "}" for name in _PLACEHOLDERS))
        inputs = []
        for item in _string_list(entry.get("inputs", []), where, "inputs"):
            check_outbox_file(base / item)
            inputs.append((base / item).resolve())
        if len({_input_name(item) for item in inputs}) != len(inputs):
            raise ValueError(f"{where}: an input is listed twice")
        names = _string_list(entry.get("secrets", []), where, "secrets")
        if not set(names) <= set(SECRET_NAMES) or len(set(names)) != len(names):
            raise ValueError(f"{where}: secrets must be distinct names of {', '.join(SECRET_NAMES)}")
        if bool(names) != ("--secrets-stdin" in args):
            raise ValueError(f"{where}: a task with secrets reads them with --secrets-stdin, and only such a task")
        mdc_dataset = entry.get("mdc_dataset")
        if ("MDC_URL" in names) != (mdc_dataset is not None):
            raise ValueError(f"{where}: mdc_dataset goes with the MDC_URL secret, and only with it")
        if mdc_dataset is not None and (not isinstance(mdc_dataset, str) or not _MDC_DATASET.fullmatch(mdc_dataset)):
            raise ValueError(f"{where}: mdc_dataset must be an MDC dataset id")
        hours = entry["hours"]
        if not isinstance(hours, Mapping) or not hours or not all(
                isinstance(gpu, str) and gpu and _positive(value) for gpu, value in hours.items()):
            raise ValueError(f"{where}: hours must map GPU type ids to positive hours")
        if not isinstance(out, str) or not out:
            raise ValueError(f"{where}: out must be a directory path")
        out_dir = (base / out).resolve()
        if out_dir.exists():
            raise ValueError(f"{out_dir} already exists; choose a new output directory")
        if any(task.out == out_dir for task in tasks):
            raise ValueError(f"two tasks write {out_dir}")
        tasks.append(TaskSpec(task_id, module, env, tuple(args), tuple(inputs), tuple(names),
                              {gpu: float(value) for gpu, value in hours.items()}, out_dir, mdc_dataset))
    resolved = _resolve_commit(commit)
    return TaskWork(tasks, disk_gb, resolved, _archive_bytes(resolved))


# --- Estimate ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Estimate:
    rows: tuple[tuple[str, float], ...]
    disk_gb: int

    @property
    def hours(self) -> float:
        return sum(seconds for _, seconds in self.rows) / 3600


def _archive_bytes(commit: str) -> int:
    archive = subprocess.run(["git", "-C", str(_REPO_ROOT), "archive", "--format=tar", commit, "--",
                              *_ARCHIVE_PATHS], capture_output=True, check=True)
    return len(archive.stdout)


def estimate(jobs: Sequence[Job], uploads: Uploads, extras: Extras, archive_bytes: int,
             uplink: tuple[float, str], provision: tuple[str, float] | None = None) -> Estimate:
    """``uplink``: Mbit/s and where the figure comes from; ``provision``:
    the pod time before the install (label, seconds), by default the time a
    new pod takes to come up."""
    hf_models = set().union(*(job.hf_models for job in jobs))
    upload_bytes = uploads.size + archive_bytes
    install = _INSTALL_S + (_INSTALL_TORCH_S if extras.torch else 0) + (_INSTALL_BENCH_S if extras.bench else 0)
    uplink_mbps, uplink_source = uplink
    rows = [
        provision or ("pod start, image pull, SSH", _PROVISION_S),
        ("environment install", install),
        (f"upload {upload_bytes / 1e9:.2f} GB at {uplink_mbps:.0f} Mbit/s ({uplink_source})",
         upload_bytes * 8 / (uplink_mbps * 1e6)),
    ]
    if hf_models:
        rows.append((f"{len(hf_models)} Hugging Face model download(s) on the pod", _HF_MODEL_S * len(hf_models)))
    for job in jobs:
        assert job.timing is not None and job.plan_workers is not None
        seconds = _JOB_START_S + job.timing.load_seconds + job.audio_seconds * job.timing.rtf
        rows.append((f"job {job.index} {job.id}: {job.audio_seconds / 3600:.2f} h audio x RTF "
                     f"{job.timing.rtf:.4f} + load {job.timing.load_seconds:.0f} s, "
                     f"{job.plan_workers.workers} worker(s)", seconds))
    rows.append(("fetch runs", _FETCH_S * len(jobs)))
    rows.append(("wipe and terminate", _TEARDOWN_S))
    data_gb = upload_bytes / 1e9
    need = (8 + (12 if extras.torch else 0) + 3 * len(hf_models) + data_gb + 2) * 1.25
    return Estimate(tuple(rows), max(20, int(math.ceil(need / 5) * 5)))


def task_estimate(work: TaskWork, gpu: GpuOffer, uplink: tuple[float, str],
                  provision: tuple[str, float] | None = None) -> Estimate:
    """Pod tasks on ``gpu``: each task's declared hours for that GPU type
    (``PolicyRefusal`` when a task declares none), the install, the uploads
    at ``uplink`` and the fixed rows; the container disk the file declares."""
    missing = [task.id for task in work.tasks if gpu.id not in task.hours]
    if missing:
        raise PolicyRefusal(f"{gpu.id} is not eligible: task(s) {', '.join(missing)} declare no hours for it "
                            f"(eligible: {', '.join(sorted(work.gpu_ids)) or 'none'})")
    ft = work.extras.torch
    install = _INSTALL_S + _INSTALL_P7ZIP_S + (_INSTALL_FT_S if ft else 0.0)
    input_bytes = sum(path.stat().st_size for task in work.tasks for path in task.inputs)
    upload_bytes = input_bytes + work.archive_bytes
    uplink_mbps, uplink_source = uplink
    rows = [
        provision or ("pod start, image pull, SSH", _PROVISION_S),
        ("environment install (release, p7zip" + (", pilot venv)" if ft else ")"), install),
        (f"upload inputs {input_bytes / 1e6:.1f} MB and code {work.archive_bytes / 1e6:.1f} MB at "
         f"{uplink_mbps:.0f} Mbit/s ({uplink_source})", upload_bytes * 8 / (uplink_mbps * 1e6)),
    ]
    for index, task in enumerate(work.tasks, 1):
        hours = task.hours[gpu.id]
        rows.append((f"task {index} {task.id}: {task.module}, {hours:g} h declared for {gpu.id}", hours * 3600))
    rows.append(("fetch task outputs", _FETCH_S * len(work.tasks)))
    rows.append(("wipe and terminate", _TEARDOWN_S))
    return Estimate(tuple(rows), work.disk_gb)


@dataclasses.dataclass(frozen=True)
class Work:
    """What does not depend on the GPU: jobs, uploads, code."""

    jobs: list[Job]
    uploads: Uploads
    extras: Extras
    commit: str
    archive_bytes: int
    runs: list[tuple[Path, dict[str, Any]]]


@dataclasses.dataclass(frozen=True)
class Plan:
    work: Work | TaskWork
    gpu: GpuOffer
    estimate: Estimate
    offers: tuple[GpuOffer, ...] = ()
    # The container disk an existing pod (SSH mode) bills; else the planned one.
    pod_disk_gb: int | None = None

    @property
    def disk_gb(self) -> int:
        return self.estimate.disk_gb if self.pod_disk_gb is None else self.pod_disk_gb

    @property
    def jobs(self) -> list[Job]:
        """The bench jobs (``Work`` only)."""
        assert isinstance(self.work, Work)
        return self.work.jobs

    @property
    def uploads(self) -> Uploads:
        """The bench uploads (``Work`` only)."""
        assert isinstance(self.work, Work)
        return self.work.uploads

    @property
    def job_ids(self) -> list[str]:
        """The bench job ids or the pod task ids, for the spend record."""
        if isinstance(self.work, TaskWork):
            return [task.id for task in self.work.tasks]
        return [job.id for job in self.work.jobs]

    @property
    def extras(self) -> Extras:
        return self.work.extras

    @property
    def commit(self) -> str:
        return self.work.commit

    @property
    def usd_per_hour(self) -> float:
        """The pod's rate: the GPU's compute rate plus its container disk."""
        return self.gpu.usd_per_hour + self.disk_gb * _DISK_USD_PER_GB_HOUR

    @property
    def usd(self) -> float:
        return self.usd_per_hour * self.estimate.hours


def _resolve_commit(revision: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "--verify", f"{revision}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def prepare(args: argparse.Namespace, root: str) -> Work | TaskWork:
    """The bench jobs, or with ``--tasks`` the pod tasks, and the code."""
    if args.tasks is not None:
        if args.jobs is not None or args.job:
            raise ValueError("--tasks cannot be combined with --jobs or --job")
        return read_task_specs(args.tasks, args.commit)
    uploads = Uploads(root, Guard())
    jobs = load_jobs(read_job_specs(args.job, args.jobs), uploads, args.workers)
    commit = _resolve_commit(args.commit)
    return Work(jobs, uploads, Extras.of(jobs), commit, _archive_bytes(commit), _past_runs())


def _count(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def pod_size(gpu: GpuOffer, facts: Mapping[str, Any] | None,
             runs: Sequence[tuple[Path, dict[str, Any]]]) -> tuple[float | None, int | None, str]:
    """GPU memory (MiB) and vCPUs of the pod, and where the vCPU count comes
    from: the pod's preflight facts; before the pod exists, the catalog's
    memory and the fewest vCPUs a past run on the GPU type recorded, else
    ``_ASSUMED_VCPUS``."""
    if facts is not None:
        gpus = facts.get("gpus") or [{}]
        return _count(gpus[0].get("memory_mib")), _count(facts.get("vcpus")), "the pod"
    recorded = [
        count for _, record in runs
        if (remote := record.get("remote") or {}) and gpu.id in (remote.get("gpu_type"), remote.get("gpu_model"))
        and (count := _count(remote.get("vcpus"))) is not None
    ]
    if recorded:
        return gpu.memory_gb * 1024, min(recorded), f"the fewest of {len(recorded)} past runs on this GPU type"
    return gpu.memory_gb * 1024, _ASSUMED_VCPUS, "assumed; no past run on this GPU type"


def measured_uplink() -> tuple[float, str]:
    """The median rate of the last ``_UPLINK_SAMPLES`` uploads of at least
    ``_UPLINK_MIN_BYTES`` in the spend log (closed and open records), else
    ``_UPLINK_MBPS``."""
    closed, opened = _spend_records()
    measured = sorted(
        (record for record in closed + opened
         if (record.get("upload") or {}).get("bytes", 0) >= _UPLINK_MIN_BYTES),
        key=lambda record: record["start_utc"],
    )[-_UPLINK_SAMPLES:]
    if not measured:
        return _UPLINK_MBPS, "default; no measured upload in the spend log"
    rate = statistics.median(record["upload"]["mbit_per_s"] for record in measured)
    return rate, f"median of the last {len(measured)} measured upload(s)"


def _uplink(args: argparse.Namespace) -> tuple[float, str]:
    return (args.uplink_mbps, "--uplink-mbps") if args.uplink_mbps is not None else measured_uplink()


def price(work: Work | TaskWork, gpu: GpuOffer, args: argparse.Namespace, offers: Sequence[GpuOffer] = (), *,
          provision: tuple[str, float] | None = None, pod_disk_gb: int | None = None,
          facts: Mapping[str, Any] | None = None) -> Plan:
    """The plan on ``gpu`` (the pod's ``facts`` when it exists): each job's
    workers and timing for that GPU type, the estimate; for pod tasks, their
    declared hours on that GPU type."""
    if isinstance(work, TaskWork):
        return Plan(work, gpu, task_estimate(work, gpu, _uplink(args), provision), tuple(offers), pod_disk_gb)
    vram_mib, vcpus, vcpus_source = pod_size(gpu, facts, work.runs)
    for job in work.jobs:
        job.plan_workers = choose_workers(
            job.workers, vram_mib=vram_mib, vcpus=vcpus, model_bytes=job.model_bytes,
            items=len(job.bench_set.items), vcpus_source=vcpus_source)
        workers = job.plan_workers.workers
        job.timing = past_timing(job.candidate, job.bench_set.set_id, gpu.id, work.runs, workers)
        if job.timing is None:
            rtf, source = (job.assume_rtf, "jobs file assume_rtf") if job.assume_rtf is not None \
                else (args.assume_rtf, "--assume-rtf")
            if rtf is None:
                raise ValueError(f"no past run of {job.candidate.name} under {bench_root() / 'runs'}; "
                                 "pass --assume-rtf or set assume_rtf on the job")
            speedup = assumed_speedup(workers)
            if workers > 1:
                source += f" / assumed speedup {speedup:.2f}x for {workers} workers"
            job.timing = PastTiming(rtf / speedup, 30.0 * workers, source)
    est = estimate(work.jobs, work.uploads, work.extras, work.archive_bytes, _uplink(args), provision)
    return Plan(work, gpu, est, tuple(offers), pod_disk_gb)


def make_plan(args: argparse.Namespace, root: str) -> Plan:
    """Jobs, uploads and estimate on the GPU chosen from the catalog; for pod
    tasks, from the GPU types every task declares hours for."""
    work = prepare(args, root)
    offers = secure_gpus(work.extras.min_cuda)
    if isinstance(work, TaskWork):
        eligible = work.gpu_ids
        if args.gpu and not any(args.gpu in (o.id, o.name) and o.id in eligible for o in offers):
            raise PolicyRefusal(f"{args.gpu!r} is not a Secure Cloud GPU type every task declares hours for "
                                f"({', '.join(sorted(eligible))})")
        offers = [o for o in offers if o.id in eligible]
    return price(work, choose_gpu(offers, args.gpu, args.min_vram, args.allow_datacenter), args, offers)


def gpu_candidates(plan: Plan, args: argparse.Namespace) -> list[GpuOffer]:
    """The plan's GPU, then (unless ``--gpu-strict``) the other GPUs the
    default choice admits, cheapest first: the order to try on capacity
    failures."""
    if args.gpu_strict:
        return [plan.gpu]
    others = [o for o in plan.offers if o.id != plan.gpu.id and o.in_stock and o.memory_gb >= args.min_vram
              and (o.consumer or args.allow_datacenter)]
    if isinstance(plan.work, TaskWork):
        others = [o for o in others if o.id in plan.work.gpu_ids]
    return [plan.gpu, *others]


def _print_tasks(work: TaskWork) -> None:
    for index, task in enumerate(work.tasks, 1):
        print(f"task {index} {task.id}: {task.module} ({task.env} venv) -> {task.out}")
        print(f"    args: {' '.join(task.args)}")
        for path in task.inputs:
            print(f"    input {_input_name(path)}: {path.stat().st_size / 1e6:.1f} MB")
        if task.secrets:
            dataset = f" (MDC dataset {task.mdc_dataset})" if task.mdc_dataset else ""
            print(f"    secrets over stdin: {', '.join(task.secrets)}{dataset}")
        print("    hours: " + ", ".join(f"{gpu} {hours:g}" for gpu, hours in sorted(task.hours.items())))


def print_plan(plan: Plan) -> None:
    print(f"code: {plan.commit[:12]} (git archive of {', '.join(_ARCHIVE_PATHS)})")
    if isinstance(plan.work, TaskWork):
        _print_tasks(plan.work)
    else:
        for job in plan.jobs:
            timing = job.timing
            assert timing is not None
            print(f"job {job.index} {job.id}: {job.candidate.asr.backend} {job.candidate.asr.model}, "
                  f"{len(job.bench_set.items)} items -> {job.out}")
            print(f"    timing from {timing.source}")
            assert job.plan_workers is not None
            print(f"    workers {job.plan_workers.text}")
            if job.seed is not None:
                print(f"    seed {job.seed} (scripts.asr_finetune.seeded_bench_run)")
        for upload in plan.uploads.items:
            print(f"upload {upload.label}: {len(upload.files)} files, {upload.size / 1e6:.1f} MB")
    print("estimate:")
    for label, seconds in plan.estimate.rows:
        print(f"    {seconds / 60:6.1f} min  {label}")
    print(f"    {plan.estimate.hours * 60:6.1f} min  total ({plan.estimate.hours:.3f} h)")
    gpu = plan.gpu
    kind = "consumer" if gpu.consumer else "DATACENTER"
    print(f"GPU: {gpu.id} ({gpu.memory_gb} GB, {kind}), Secure Cloud, ${gpu.usd_per_hour:.2f}/h, "
          f"stock {gpu.stock}, host CUDA >= {plan.extras.min_cuda}")
    print(f"container disk: {plan.disk_gb} GB (${plan.disk_gb * _DISK_USD_PER_GB_HOUR:.4f}/h); "
          f"pod rate ${plan.usd_per_hour:.4f}/h")
    print(f"estimated cost: ${plan.usd:.4f}")
    if plan.offers:
        others = [o for o in plan.offers if o.in_stock and o.consumer and o.id != gpu.id][:4]
        if others:
            print("other consumer GPUs in stock: " + ", ".join(
                f"{o.name} ${o.usd_per_hour:.2f}/h ({o.stock})" for o in others))


def print_authorization(plan: Plan, auth: Authorization, committed_usd: float) -> None:
    day_cap = f", day cap ${auth.day_cap_usd:.2f} (owner-approved --day-cap)" if auth.day_cap_raised else ""
    print(f"policy: {auth.basis}{day_cap}; today ${committed_usd:.2f} (spent, open pods at their caps); "
          f"cap ${auth.cap_usd:.2f}, cut off after {auth.cap_hours * 60:.1f} min (the pod costs "
          f"at most ${auth.cap_hours * plan.usd_per_hour:.4f})")


# --- Pod task secrets --------------------------------------------------

# The credentials read on this machine, by name in ``_CREDENTIAL_SOURCES``.
# ``run --tasks`` moves both variables out of the process environment as it
# starts (``hold_credentials``), so no child process (ssh, git) inherits them.
_CREDENTIALS: dict[str, str] = {}
# A presigned URL's query values of at least this length are masked and
# scanned for on their own, raw and decoded.
_URL_PART_CHARS = 16


def hold_credentials() -> None:
    """Move both credential variables from the process environment into
    ``_CREDENTIALS``, whether or not a task needs them."""
    for name, (variable, _) in _CREDENTIAL_SOURCES.items():
        value = os.environ.pop(variable, "").strip()
        if value:
            _CREDENTIALS.setdefault(name, value)


def _credential(name: str) -> str:
    if name not in _CREDENTIALS:
        variable, label = _CREDENTIAL_SOURCES[name]
        value = os.environ.pop(variable, "").strip()
        if not value:
            value = getpass.getpass(f"{label} ({variable} is not set): ").strip()
        if not value:
            raise PolicyRefusal(f"no {label}: set {variable} or type it at the prompt")
        _CREDENTIALS[name] = value
    return _CREDENTIALS[name]


def ask_credentials(work: TaskWork) -> None:
    """Read the credentials the tasks' secrets come from (the environment,
    else a hidden prompt) before the pod runs, so no prompt waits mid-run."""
    names = {name for task in work.tasks for name in task.secrets}
    if "HF_TOKEN" in names:
        _credential("HF_TOKEN")
    if "MDC_URL" in names:
        _credential("MDC_API_KEY")


def mdc_download_url(api_key: str, dataset_id: str) -> tuple[str, datetime]:
    """A presigned download URL of a Mozilla Data Collective dataset and when
    it expires (``POST /datasets/{id}/download``). Counts toward the
    organization's 30 downloads a day; the key never leaves this machine."""
    response = httpx.post(f"{MDC_API}/datasets/{dataset_id}/download", timeout=60.0,
                          headers={"Authorization": f"Bearer {api_key}", "User-Agent": "yamabiko-asr-bench"})
    if response.status_code != 200:
        raise RemoteError(f"MDC download URL for dataset {dataset_id}: HTTP {response.status_code} "
                          f"{mask(response.text[:300], [api_key])}")
    body = response.json()
    url, expires = body.get("downloadUrl"), body.get("expiresAt")
    if not isinstance(url, str) or not url.startswith("https://") or not isinstance(expires, str):
        raise RemoteError(f"MDC download URL for dataset {dataset_id}: no downloadUrl or expiresAt in the answer")
    return url, _parse_iso(expires)


def resolve_secrets(names: Sequence[str], mdc_dataset: str | None) -> dict[str, str]:
    """The values of a task's secrets: ``HF_TOKEN`` as read, ``MDC_URL`` a
    fresh presigned URL of ``mdc_dataset``. Nothing here prints a value."""
    values: dict[str, str] = {}
    for name in names:
        if name == "HF_TOKEN":
            values[name] = _credential("HF_TOKEN")
        elif name == "MDC_URL":
            if not mdc_dataset:
                raise ValueError("the MDC_URL secret needs an mdc_dataset")
            url, expires = mdc_download_url(_credential("MDC_API_KEY"), mdc_dataset)
            print(f"    MDC download URL for dataset {mdc_dataset} issued; it expires at {_iso(expires)}", flush=True)
            values[name] = url
        else:
            raise ValueError(f"unknown secret {name!r} (known: {', '.join(SECRET_NAMES)})")
    return values


def mask(text: str, values: Sequence[str]) -> str:
    """``text`` with every non-empty value replaced by ``***``."""
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        text = text.replace(value, "***")
    return text


def secret_parts(name: str, value: str) -> list[str]:
    """A secret value and what else of it to mask and scan for: for
    ``MDC_URL`` its query and each query value of ``_URL_PART_CHARS`` or
    more characters, raw and decoded, as HTTP library errors print a URL as
    its path and query only."""
    parts = [value]
    if name == "MDC_URL":
        query = urlsplit(value).query
        if query:
            parts.append(query)
        for item in query.split("&"):
            raw = item.partition("=")[2]
            parts += [part for part in (raw, unquote(raw)) if len(part) >= _URL_PART_CHARS]
    return parts


def _holds(stream: IO[bytes], needles: Sequence[bytes]) -> bool:
    keep = max(map(len, needles)) - 1
    tail = b""
    while chunk := stream.read(1 << 20):
        window = tail + chunk
        if any(needle in window for needle in needles):
            return True
        tail = window[len(window) - keep:] if keep else b""
    return False


def scan_for_secrets(directory: Path, values: Sequence[str]) -> list[Path]:
    """The files under ``directory`` (or ``directory`` itself, a file) whose
    bytes, decompressed bytes (``.gz``) or path below it hold a value."""
    needles = sorted({value.encode() for value in values if value}, key=len, reverse=True)
    if not needles:
        return []
    texts = [value for value in values if value]
    files = [directory] if directory.is_file() else sorted(p for p in directory.rglob("*") if p.is_file())
    hits = []
    for path in files:
        relative = path.name if path == directory else path.relative_to(directory).as_posix()
        with path.open("rb") as raw:
            found = any(text in relative for text in texts) or _holds(raw, needles)
        if not found and path.suffix == ".gz":
            try:
                with gzip.open(path, "rb") as unpacked:
                    found = _holds(unpacked, needles)
            except (OSError, EOFError):
                pass
        if found:
            hits.append(path)
    return hits


# --- Deadline and SSH --------------------------------------------------


class Deadline:
    """The hard wall-clock cap on this side: when it fires, every tracked
    SSH process is killed and the next step raises ``CapReached``."""

    def __init__(self, seconds: float):
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen[bytes]] = set()
        self.expired = threading.Event()
        self._at = time.monotonic() + seconds
        self._timer: threading.Timer | None = None
        self._arm()

    def _arm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(max(0.0, self._at - time.monotonic()), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.expired.set()
        with self._lock:
            for proc in self._procs:
                proc.kill()

    def shorten(self, seconds: float) -> None:
        if time.monotonic() + seconds < self._at:
            self._at = time.monotonic() + seconds
            self._arm()

    def remaining(self) -> float:
        return max(0.0, self._at - time.monotonic())

    def check(self) -> None:
        if self.expired.is_set():
            raise CapReached("the hard wall-clock cap was reached")

    def track(self, proc: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._procs.add(proc)
        if self.expired.is_set():
            proc.kill()

    def untrack(self, proc: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._procs.discard(proc)

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()


class _Meter:
    """A write-only stream that counts and reports upload progress."""

    def __init__(self, sink: IO[bytes], total: int, label: str):
        self._sink, self._total, self._label = sink, total, label
        self._done = 0
        self._begin = self._last = time.monotonic()

    def write(self, data: bytes) -> int:
        self._sink.write(data)
        self._done += len(data)
        now = time.monotonic()
        if now - self._last > 15:
            self._last = now
            rate = self._done / max(now - self._begin, 1e-6) / 1e6
            print(f"    {self._label}: {self._done / 1e6:.0f}/{self._total / 1e6:.0f} MB ({rate:.1f} MB/s)",
                  flush=True)
        return len(data)


class Remote:
    """Commands and tar streams over SSH to one pod (root shell, bash)."""

    def __init__(self, host: str, port: int, user: str, key: Path, root: str, deadline: Deadline):
        self.host, self.port, self.user, self.key, self.root = host, port, user, key, root
        self.deadline = deadline
        self._folder = Path(tempfile.mkdtemp(prefix="ybench-ssh-"))

    def close(self) -> None:
        shutil.rmtree(self._folder, ignore_errors=True)

    def argv(self, command: str) -> list[str]:
        options = {
            "BatchMode": "yes", "IdentitiesOnly": "yes", "StrictHostKeyChecking": "accept-new",
            "UserKnownHostsFile": (self._folder / "known_hosts").as_posix(), "ServerAliveInterval": "15",
            "ServerAliveCountMax": "8", "ConnectTimeout": "20", "LogLevel": "ERROR",
        }
        argv = ["ssh", "-i", str(self.key), "-p", str(self.port)]
        for key, value in options.items():
            argv += ["-o", f"{key}={value}"]
        return argv + [f"{self.user}@{self.host}", command]

    def _timeout(self, timeout: float | None, tracked: bool) -> float | None:
        if not tracked:
            return timeout
        return self.deadline.remaining() if timeout is None else min(timeout, self.deadline.remaining())

    def call(self, command: str, data: bytes | None = None, *, timeout: float | None = None,
             tracked: bool = True, what: str | None = None) -> str:
        proc = subprocess.Popen(self.argv(command), stdout=subprocess.PIPE,
                                stdin=subprocess.DEVNULL if data is None else subprocess.PIPE)
        if tracked:
            self.deadline.track(proc)
        try:
            out, _ = proc.communicate(data, timeout=self._timeout(timeout, tracked))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RemoteError(f"{what or command} timed out") from None
        finally:
            self.deadline.untrack(proc)
        if tracked:
            self.deadline.check()
        if proc.returncode:
            raise RemoteError(f"{what or command} failed (exit {proc.returncode})")
        return out.decode("utf-8", "replace")

    def script(self, text: str, *args: str, what: str, timeout: float | None = None,
               tracked: bool = True) -> str:
        command = "bash -s -- " + " ".join(shlex.quote(arg) for arg in args)
        return self.call(command, text.encode(), timeout=timeout, tracked=tracked, what=what)

    def stream(self, command: str, data: bytes | None = None, mask_values: Sequence[str] = ()) -> int:
        """Run with output on this console; the exit code. ``data`` is the
        command's stdin (else it has none); with ``mask_values`` the output,
        stderr included, is piped and printed line by line through ``mask``."""
        piped = subprocess.PIPE if mask_values else None
        proc = subprocess.Popen(self.argv(command), stdin=subprocess.DEVNULL if data is None else subprocess.PIPE,
                                stdout=piped, stderr=subprocess.STDOUT if mask_values else None)
        self.deadline.track(proc)
        try:
            if proc.stdin is not None:
                with contextlib.suppress(OSError):  # a command that ends without reading it
                    proc.stdin.write(data or b"")
                    proc.stdin.close()
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(mask(line.decode("utf-8", "replace").rstrip("\r\n"), mask_values), flush=True)
            return proc.wait(timeout=self.deadline.remaining())
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise CapReached("the hard wall-clock cap was reached") from None
        finally:
            self.deadline.untrack(proc)
            self.deadline.check()

    def wait_ready(self, limit: float) -> None:
        until = time.monotonic() + min(limit, self.deadline.remaining())
        while True:
            try:
                self.call("true", timeout=60, what="SSH check")
                return
            except CapReached:
                raise
            except RemoteError:
                if time.monotonic() > until:
                    raise RemoteError(f"no SSH to {self.host}:{self.port}") from None
                time.sleep(5)

    def put_files(self, upload: Upload) -> float:
        """Upload one ``Upload``; the seconds it took."""
        begin = time.monotonic()
        command = f"mkdir -p {shlex.quote(upload.remote)} && tar -xf - -C {shlex.quote(upload.remote)}"
        proc = subprocess.Popen(self.argv(command), stdin=subprocess.PIPE)
        self.deadline.track(proc)
        try:
            assert proc.stdin is not None
            with tarfile.open(fileobj=_Meter(proc.stdin, upload.size, upload.label), mode="w|",
                              format=tarfile.PAX_FORMAT) as tar:
                for path, arcname in upload.files:
                    info = tar.gettarinfo(str(path), arcname)
                    info.mode, info.uid, info.gid, info.uname, info.gname = 0o644, 0, 0, "", ""
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
            proc.stdin.close()
            code = proc.wait(timeout=self.deadline.remaining())
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            code = -1
        finally:
            self.deadline.untrack(proc)
        self.deadline.check()
        if code:
            raise RemoteError(f"upload of {upload.label} failed (exit {code})")
        seconds = time.monotonic() - begin
        print(f"    {upload.label}: {upload.size / 1e6:.1f} MB in {seconds:.0f} s", flush=True)
        return seconds

    def put_archive(self, commit: str, remote: str) -> None:
        git = subprocess.Popen(["git", "-C", str(_REPO_ROOT), "archive", "--format=tar", commit, "--",
                                *_ARCHIVE_PATHS], stdout=subprocess.PIPE)
        command = f"mkdir -p {shlex.quote(remote)} && tar -xf - -C {shlex.quote(remote)}"
        proc = subprocess.Popen(self.argv(command), stdin=git.stdout)
        assert git.stdout is not None
        git.stdout.close()
        self.deadline.track(proc)
        try:
            code = proc.wait(timeout=self.deadline.remaining())
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -1
        finally:
            self.deadline.untrack(proc)
        self.deadline.check()
        if git.wait() or code:
            raise RemoteError(f"code upload failed (git {git.returncode}, ssh {code})")

    def get_dir(self, remote: str, local: Path) -> None:
        proc = subprocess.Popen(self.argv(f"tar -cf - -C {shlex.quote(remote)} ."), stdout=subprocess.PIPE,
                                stdin=subprocess.DEVNULL)
        self.deadline.track(proc)
        try:
            assert proc.stdout is not None
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                tar.extractall(local, filter="data")
            # tarfile stops at the end-of-archive marker, but tar pads the
            # stream to a whole record; when the marker lands on a record
            # boundary, ssh still holds unread padding and blocks on the full
            # pipe, so wait() would hang until the deadline. Drain to EOF.
            while proc.stdout.read(1 << 16):
                pass
            code = proc.wait(timeout=self.deadline.remaining())
        except (OSError, tarfile.TarError, subprocess.TimeoutExpired) as error:
            proc.kill()
            proc.wait()
            self.deadline.check()
            raise RemoteError(f"download of {remote} failed: {error}") from None
        finally:
            self.deadline.untrack(proc)
        self.deadline.check()
        if code:
            raise RemoteError(f"download of {remote} failed (exit {code})")


# --- One pod's work ----------------------------------------------------


@dataclasses.dataclass
class PodContext:
    mode: str
    pod_id: str
    since: datetime
    compute_usd_per_hour: float
    disk_gb: int
    rate_source: str
    cloud: str | None
    gpu_type: str | None = None
    image: str | None = None
    region: str | None = None
    facts: dict[str, Any] = dataclasses.field(default_factory=dict)
    environment: dict[str, Any] = dataclasses.field(default_factory=dict)
    # The data uploads: bytes, seconds and Mbit/s, for the spend log.
    upload: dict[str, Any] | None = None

    @property
    def usd_per_hour(self) -> float:
        """What the pod bills: the pod's ``cost`` is compute only, the
        container disk bills on top of it."""
        return self.compute_usd_per_hour + self.disk_gb * _DISK_USD_PER_GB_HOUR


def volume_mounts(pod: Mapping[str, Any]) -> dict[str, Any]:
    """The pod's volumes of any kind: host-local ``persistent`` (it survives
    restarts and bills while stopped), ``network`` and ``global``."""
    mounts = pod.get("mounts") or {}
    return {kind: mounts[kind] for kind in ("persistent", "network", "global") if mounts.get(kind)}


def preflight(remote: Remote, expected_pod_id: str) -> dict[str, Any]:
    """Pod facts; refuses a host that is not the expected pod or cannot
    terminate itself."""
    facts = json.loads(remote.call("python3 -", _PREFLIGHT.encode(), what="preflight").strip().splitlines()[-1])
    if facts.get("pod_id") != expected_pod_id:
        raise PolicyRefusal(f"{remote.host}:{remote.port} is pod {facts.get('pod_id')!r}, "
                            f"not {expected_pod_id!r}")
    if not facts.get("pod_key") or not (facts.get("runpodctl") or facts.get("curl")):
        raise PolicyRefusal("the pod has no scoped API key, runpodctl or curl: it cannot terminate itself")
    if not facts.get("python312") or not facts.get("uv"):
        raise RemoteError("the pod image lacks python3.12 or uv")
    return facts


def arm_watchdog(remote: Remote, deadline: datetime, *, tracked: bool = True) -> None:
    """Set the pod watchdog's absolute deadline and start it if needed."""
    encoded = base64.b64encode(_WATCHDOG.encode()).decode()
    out = remote.script(_ARM_WATCHDOG, remote.root, str(_epoch(deadline)), encoded, what="watchdog",
                        timeout=120, tracked=tracked)
    if "armed" not in out:
        raise RemoteError("the pod watchdog did not start")


def localize(value: Any, paths: Mapping[str, str]) -> Any:
    """Replace remote paths (and paths under them) with the local ones."""
    if isinstance(value, Mapping):
        return {k: localize(v, paths) for k, v in value.items()}
    if isinstance(value, list):
        return [localize(v, paths) for v in value]
    if isinstance(value, str):
        for remote in sorted(paths, key=len, reverse=True):
            if value == remote:
                return paths[remote]
            if value.startswith(remote + "/"):
                return str(Path(paths[remote], *value[len(remote) + 1:].split("/")))
    return value


def remote_block(ctx: PodContext, commit: str, paths: Mapping[str, str],
                 workers: WorkerChoice | None = None, seed: int | None = None) -> dict[str, Any]:
    now = _now()
    hours = (now - ctx.since).total_seconds() / 3600
    gpu = (ctx.facts.get("gpus") or [{}])[0]
    return {
        "schema_version": _REMOTE_SCHEMA,
        "provider": "runpod",
        "mode": ctx.mode,
        "pod_id": ctx.pod_id,
        "cloud": ctx.cloud,
        "gpu_type": ctx.gpu_type,
        "gpu_model": gpu.get("name"),
        "gpu_memory_mib": gpu.get("memory_mib"),
        "gpu_driver": gpu.get("driver"),
        "cuda_version": ctx.facts.get("cuda"),
        "cpu_model": ctx.facts.get("cpu_model"),
        "vcpus": ctx.facts.get("vcpus"),
        "image": ctx.image,
        "region": ctx.region,
        "environment": ctx.environment,
        "workers_choice": None if workers is None else workers.basis,
        "seed": seed,
        "code": {"commit": commit, "source": "git archive"},
        "paths": dict(paths),
        "cost": {
            "usd_per_hour": ctx.usd_per_hour,
            "compute_usd_per_hour": ctx.compute_usd_per_hour,
            "disk_gb": ctx.disk_gb,
            "rate_source": ctx.rate_source,
            "since_utc": _iso(ctx.since),
            "at_utc": _iso(now),
            "hours": hours,
            "usd_so_far": ctx.usd_per_hour * hours,
        },
    }


def fetch(remote: Remote, job: Job, remote_out: str, ctx: PodContext, plan: Plan) -> None:
    staging = job.out.with_name(f".{job.out.name}.remote")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        remote.get_dir(remote_out, staging)
        run_path = staging / "run.json"
        record = localize(json.loads(run_path.read_text(encoding="utf-8")), plan.uploads.paths)
        # The pod has no .git; the code that ran is exactly the archived commit.
        record["git"] = {"head": plan.commit, "dirty": False}
        record["remote"] = remote_block(ctx, plan.commit, plan.uploads.paths, job.run_workers, job.seed)
        run_path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if job.out.exists():
            raise RemoteError(f"{job.out} appeared during the run")
        staging.rename(job.out)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"job {job.index} {job.id}: fetched -> {job.out}", flush=True)


def install_environment(remote: Remote, plan: Plan) -> dict[str, Any]:
    """The release environment on the pod (on a pod-task pod also p7zip and,
    when a task runs in it, the pilot venv), the whole step retried after a
    failure (a slow or unreachable package index) with a growing pause."""
    if isinstance(plan.work, TaskWork):
        # The release venv of a task pod takes no extras; torch is the pilot venv's.
        flags = (0, 0, int(plan.extras.torch), 1)
        label = "release pins, p7zip" + (", pilot venv" if plan.extras.torch else "")
    else:
        flags = (int(plan.extras.torch), int(plan.extras.bench), 0, 0)
        label = "release pins"
    for attempt in range(1, _INSTALL_ATTEMPTS + 1):
        print(f"installing the environment ({label}), attempt {attempt} of {_INSTALL_ATTEMPTS}", flush=True)
        begin = time.monotonic()
        try:
            out = remote.script(_INSTALL_ENV, remote.root, *(str(flag) for flag in flags),
                                what="environment install")
        except CapReached:
            raise
        except RemoteError as error:
            if attempt == _INSTALL_ATTEMPTS:
                raise
            pause = _INSTALL_BACKOFF_S * attempt
            print(f"    attempt {attempt} failed after {time.monotonic() - begin:.0f} s ({error}); "
                  f"retrying in {pause:.0f} s", flush=True)
            remote.deadline.expired.wait(pause)
            remote.deadline.check()
            continue
        print(f"    installed in {time.monotonic() - begin:.0f} s", flush=True)
        return json.loads(out.strip().splitlines()[-1])
    raise AssertionError("unreachable")


def _set_up(remote: Remote, plan: Plan, ctx: PodContext) -> None:
    """An empty work root, the code and the environment."""
    root = remote.root
    q = shlex.quote
    print(f"preparing {root} on the pod", flush=True)
    remote.call(f"rm -rf -- {q(root)} && mkdir -p {q(root)}", what="prepare", timeout=300)
    remote.put_archive(plan.commit, f"{root}/repo")
    ctx.environment = install_environment(remote, plan)
    packages = ctx.environment["packages"]
    print(f"    {ctx.environment['installer']}, python {ctx.environment['python']}, "
          f"ctranslate2 {packages.get('ctranslate2')}, faster-whisper {packages.get('faster-whisper')}, "
          f"onnxruntime {packages.get('onnxruntime')}, numpy {packages.get('numpy')}", flush=True)
    if "ft_packages" in ctx.environment:
        ft = ctx.environment["ft_packages"]
        print(f"    pilot venv: torch {ft.get('torch')}, transformers {ft.get('transformers')}, "
              f"peft {ft.get('peft')}, ctranslate2 {ft.get('ctranslate2')}", flush=True)


def execute(remote: Remote, plan: Plan, ctx: PodContext) -> None:
    if isinstance(plan.work, TaskWork):
        execute_tasks(remote, plan, ctx)
        return
    root = remote.root
    q = shlex.quote
    _set_up(remote, plan, ctx)
    print("uploading data", flush=True)
    seconds = sum(remote.put_files(upload) for upload in plan.uploads.items)
    if seconds > 0:
        size = plan.uploads.size
        ctx.upload = {"bytes": size, "seconds": round(seconds, 3),
                      "mbit_per_s": round(size * 8 / seconds / 1e6, 2)}
        print(f"    {size / 1e9:.2f} GB in {seconds:.0f} s: {ctx.upload['mbit_per_s']:.0f} Mbit/s", flush=True)
    vram_mib, vcpus, _ = pod_size(plan.gpu, ctx.facts, ())
    for job in plan.jobs:
        assert job.plan_workers is not None
        job.run_workers = choose_workers(
            job.workers, vram_mib=vram_mib, vcpus=vcpus, model_bytes=job.model_bytes,
            items=len(job.bench_set.items), vcpus_source="the pod")
        if job.run_workers.workers < job.plan_workers.workers:
            print(f"note: job {job.index} runs {job.run_workers.text}, fewer than the estimate's "
                  f"{job.plan_workers.workers}: it takes longer than estimated", flush=True)
        job_dir = f"{root}/jobs/{job.index}"
        remote.call(f"mkdir -p {q(job_dir)} && cat > {q(job_dir)}/candidate.json",
                    json.dumps(job.remote_candidate, ensure_ascii=False).encode(), what="candidate upload")
        remote_out = f"{root}/runs/{job.index}"
        runner = "scripts.asr_bench.run" if job.seed is None else \
            f"scripts.asr_finetune.seeded_bench_run --seed {job.seed}"
        seeded = "" if job.seed is None else f", seed {job.seed}"
        print(f"job {job.index} {job.id}: running, workers {job.run_workers.text}{seeded}", flush=True)
        code = remote.stream(
            f". {q(root)}/env.sh && cd {q(root)}/repo && exec {q(root)}/venv/bin/python -m "
            f"{runner} --set {q(job.remote_set)} --candidate {q(job_dir)}/candidate.json "
            f"--name {q(job.candidate.name)} --out {q(remote_out)} --workers {job.run_workers.workers}")
        if code:
            raise RemoteError(f"job {job.index} {job.id} failed (exit {code})")
        fetch(remote, job, remote_out, ctx, plan)
        remote.call(f"rm -rf -- {q(remote_out)}", what="run cleanup", timeout=120)
    wipe(remote)


def _upload_inputs(remote: Remote, work: TaskWork, ctx: PodContext) -> list[list[dict[str, Any]]]:
    """Each task's inputs into its ``{in}``, every file checked again right
    before; per task, the inputs as ``pod.json`` records them."""
    records: list[list[dict[str, Any]]] = []
    uploads = []
    for index, task in enumerate(work.tasks, 1):
        files, record = [], []
        for path in task.inputs:
            check_outbox_file(path)
            files.append((path, _input_name(path)))
            record.append({"path": _input_name(path), "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append(record)
        if files:
            uploads.append(Upload(f"task {index} {task.id} inputs", f"{remote.root}/tasks/{index}/in",
                                  tuple(files), sum(path.stat().st_size for path, _ in files)))
    print("uploading inputs", flush=True)
    seconds = sum(remote.put_files(upload) for upload in uploads)
    if seconds > 0:
        size = sum(upload.size for upload in uploads)
        ctx.upload = {"bytes": size, "seconds": round(seconds, 3), "mbit_per_s": round(size * 8 / seconds / 1e6, 2)}
        print(f"    {size / 1e6:.1f} MB in {seconds:.0f} s: {ctx.upload['mbit_per_s']:.0f} Mbit/s", flush=True)
    return records


def _fetch_task(remote: Remote, plan: Plan, ctx: PodContext, index: int, task: TaskSpec,
                inputs: list[dict[str, Any]], seconds: float, known: Sequence[str], exit_code: int) -> None:
    """The task's ``{out}`` into ``task.out`` with ``pod.json`` (its exit
    code included), unless a fetched file holds a secret value: then the copy
    is deleted and the run fails."""
    task_dir = f"{remote.root}/tasks/{index}"
    staging = task.out.with_name(f".{task.out.name}.remote")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        remote.get_dir(f"{task_dir}/out", staging)
        packages = ctx.environment.get("packages" if task.env == "release" else "ft_packages") or {}
        record = {
            "schema_version": _POD_TASK_SCHEMA, "task": task.id, "module": task.module, "args": list(task.args),
            "env": {"name": task.env, "packages": packages}, "inputs": inputs, "secrets": list(task.secrets),
            "seconds": round(seconds, 3), "exit_code": exit_code,
            "remote": remote_block(ctx, plan.commit, {f"{task_dir}/out": str(task.out),
                                                      f"{task_dir}/in": str(outbox_dir())}),
        }
        (staging / "pod.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        hits = scan_for_secrets(staging, known)
        if hits:
            names = mask(", ".join(path.relative_to(staging).as_posix() for path in hits), known)
            shutil.rmtree(staging)
            raise RemoteError(f"task {index} {task.id}: {len(hits)} fetched file(s) hold a secret value ({names}); "
                              "the fetched copy is deleted. Revoke the credentials this run read "
                              f"({', '.join(sorted(_CREDENTIALS))}) and create new ones")
        if task.out.exists():
            raise RemoteError(f"{task.out} appeared during the run")
        staging.rename(task.out)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"task {index} {task.id}: fetched -> {task.out}", flush=True)


def execute_tasks(remote: Remote, plan: Plan, ctx: PodContext) -> None:
    """The pod tasks in file order: code, environment and inputs; then each
    task with its secrets on stdin (requested right before it) and its
    console output masked, its ``{out}`` fetched and scanned; then the wipe."""
    work = plan.work
    assert isinstance(work, TaskWork)
    root = remote.root
    q = shlex.quote
    _set_up(remote, plan, ctx)
    inputs = _upload_inputs(remote, work, ctx)
    # Every credential read on this machine, sent to the pod or not, is
    # masked and scanned for, as is every secret value sent and, of a
    # presigned URL, its query parts (``secret_parts``).
    known = list(_CREDENTIALS.values())
    for index, task in enumerate(work.tasks, 1):
        task_dir = f"{root}/tasks/{index}"
        remote.call(f"mkdir -p {q(root)}/data {q(task_dir)}/in {q(task_dir)}/out", what="task directories",
                    timeout=120)
        fill = {"in": f"{task_dir}/in", "out": f"{task_dir}/out", "data": f"{root}/data", "repo": f"{root}/repo",
                "deadline": str(int(time.time() + remote.deadline.remaining()) - DEADLINE_RESERVE_S)}
        argv = [_PLACEHOLDER.sub(lambda match: fill[match.group()[1:-1]], arg) for arg in task.args]
        venv, env_file = ("venv", "env.sh") if task.env == "release" else ("venv-ft", "env-ft.sh")
        values = resolve_secrets(task.secrets, task.mdc_dataset)
        known += [part for name, value in values.items() for part in secret_parts(name, value)]
        sent = f", secrets {', '.join(values)} on stdin" if values else ""
        print(f"task {index} {task.id}: running {task.module} in the {task.env} venv{sent}", flush=True)
        begin = time.monotonic()
        code = remote.stream(
            f". {q(root)}/{env_file} && cd {q(root)}/repo && exec {q(root)}/{venv}/bin/python -m {q(task.module)} "
            + " ".join(q(arg) for arg in argv),
            (json.dumps(values) + "\n").encode() if values else None, mask_values=known)
        seconds = time.monotonic() - begin
        if code:
            print(f"task {index} {task.id}: exit {code}; fetching its output before the run fails", flush=True)
        _fetch_task(remote, plan, ctx, index, task, inputs[index - 1], seconds, known, code)
        if code:
            raise RemoteError(f"task {index} {task.id} failed (exit {code}); its output is in {task.out}")
        remote.call(f"rm -rf -- {q(task_dir)}/out", what="task cleanup", timeout=120)
    wipe(remote)


def wipe(remote: Remote, *, tracked: bool = True) -> None:
    root = shlex.quote(remote.root)
    remote.call(f"rm -rf -- {root} && test ! -e {root}", what="wipe", timeout=300, tracked=tracked)
    print(f"wiped {remote.root} on the pod", flush=True)


# --- run: API mode -------------------------------------------------------


def _terminate_confirmed(api: RunpodApi, pod_id: str) -> bool:
    for attempt in range(12):
        try:
            api.terminate(pod_id)
            pod = api.pod(pod_id)
            if pod is None or pod.get("status") == "TERMINATED":
                print(f"pod {pod_id} terminated", flush=True)
                return True
        except KeyboardInterrupt:
            print("still terminating the pod; wait", flush=True)
            continue
        except (RunpodError, httpx.HTTPError) as error:
            print(f"terminate {pod_id}: {error}", flush=True)
        time.sleep(min(30, 2 ** attempt))
    print(f"WARNING: termination of pod {pod_id} is not confirmed; run `remote terminate {pod_id}`. "
          "Its watchdog terminates it at the cap.", flush=True)
    return False


def _wait_for_pod(api: RunpodApi, pod_id: str, deadline: Deadline) -> dict[str, Any]:
    until = time.monotonic() + min(900.0, deadline.remaining())
    status = None
    while True:
        deadline.check()
        pod = api.pod(pod_id)
        if pod is None or pod.get("status") in ("ERROR", "TERMINATED", "EXITED"):
            raise RemoteError(f"pod {pod_id} is {None if pod is None else pod.get('status')}")
        if pod.get("status") != status:
            status = pod.get("status")
            print(f"pod {pod_id}: {status}", flush=True)
        direct = (pod.get("ssh") or {}).get("direct")
        if status == "RUNNING" and direct:
            return pod
        if time.monotonic() > until:
            raise RemoteError(f"pod {pod_id} has no SSH after 15 min")
        time.sleep(5)


def _pod_body(plan: Plan, args: argparse.Namespace, name: str, deadline: datetime) -> dict[str, Any]:
    return {
        "name": name,
        "cloud": "SECURE",
        "gpu": {"id": plan.gpu.id, "count": 1, "minCudaVersion": plan.extras.min_cuda},
        "image": args.image,
        "disk": plan.disk_gb,
        "ports": ["22/tcp"],
        "env": {
            "PUBLIC_KEY": Path(f"{args.ssh_key}.pub").read_text(encoding="utf-8").strip(),
            "YBENCH_WORK": REMOTE_ROOT,
            "YBENCH_DEADLINE": str(_epoch(deadline)),
            "YBENCH_WATCHDOG": base64.b64encode(_WATCHDOG.encode()).decode(),
        },
        "entrypoint": ["/bin/bash", "-c"],
        "cmd": [_BOOTSTRAP],
    }


def _stray_pod(api: RunpodApi, name: str) -> dict[str, Any] | None:
    """The pod a failed create may have made anyway, found by its unique name
    (checked three times: the list can lag the create)."""
    for attempt in range(3):
        for pod in api.pods():
            if pod.get("name") == name:
                return pod
        if attempt < 2:
            time.sleep(_STRAY_PAUSE_S)
    return None


def _after_failed_create(api: RunpodApi, record: dict[str, Any]) -> None:
    """The create raised (a lost response, Ctrl-C, a server error) and the
    pod may exist anyway: look for it by name and terminate it. The open
    record is dropped when no such pod exists, closed once it is terminated,
    and left open (for ``_settle``) when the check itself fails."""
    try:
        pod = _stray_pod(api, record["name"])
        if pod is None:
            _discard_open(record)
            print(f"no pod named {record['name']} was created", flush=True)
            return
        record.update(pod_id=pod["id"], state="stray")
        _write_open(record)
        print(f"the failed create made pod {pod['id']}; terminating it", flush=True)
        if _terminate_confirmed(api, pod["id"]):
            _close_at(record, _now(), "stray of a failed create, terminated", "failed")
    except BaseException as error:  # the create's own error is re-raised by the caller
        print(f"could not check for a pod named {record['name']} ({type(error).__name__}: {error}); "
              "its spend record stays open until a later run settles it", flush=True)


def approve_and_create(api: RunpodApi, plan: Plan, args: argparse.Namespace
                       ) -> tuple[Plan, Authorization, dict[str, Any], dict[str, Any]]:
    """Under the approval lock: settle the open records, then create the pod
    on the plan's GPU or, when RunPod has no capacity for it (400, or 403 for
    a pool the account cannot use), on the next GPU of ``gpu_candidates``.
    Each GPU is priced at its own rate and authorized against today's spend
    with every open pod at the most it can cost, and its open record (keyed
    by the pod's name, with the watchdog deadline) is written before the
    create call."""
    with _approval_lock():
        _settle(api)
        committed, _, _ = spend_totals(at_most=True)
        failures = []
        for index, gpu in enumerate(gpu_candidates(plan, args)):
            if index:
                plan = price(plan.work, gpu, args, plan.offers)
                print(f"trying {gpu.id} at ${gpu.usd_per_hour:.2f}/h: {plan.estimate.hours * 60:.1f} min, "
                      f"estimated ${plan.usd:.4f}", flush=True)
            try:
                auth = authorize(estimate_usd=plan.usd, estimate_hours=plan.estimate.hours,
                                 usd_per_hour=plan.usd_per_hour, consumer=gpu.consumer, today_usd=committed,
                                 approve=args.approve_cost, day_cap_usd=args.day_cap)
            except PolicyRefusal as refusal:
                if not index:
                    raise
                failures.append(f"{gpu.id}: {refusal}")
                print(f"    skipped: {refusal}", flush=True)
                continue
            print_authorization(plan, auth, committed)
            name = f"ybench-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
            start = _whole_second(_now())  # as recorded, so the deadline stays within the cap
            deadline = _whole_second(start + timedelta(hours=auth.cap_hours))
            record = _new_record(
                name=name, pod_id=None, mode="api", gpu=gpu.id, compute_usd_per_hour=gpu.usd_per_hour,
                disk_gb=plan.disk_gb, start=start, deadline=deadline, estimated_usd=plan.usd, auth=auth,
                jobs=plan.job_ids)
            _write_open(record)
            try:
                pod = api.create_pod(_pod_body(plan, args, name, deadline))
            except RunpodError as error:
                if error.status in (400, 403):
                    _discard_open(record)
                    failures.append(f"{gpu.id}: {error}")
                    print(f"    {gpu.id} not created: {error}", flush=True)
                    continue
                _after_failed_create(api, record)
                raise
            except BaseException:
                _after_failed_create(api, record)
                raise
            record.update(pod_id=pod["id"], state="running")
            _write_open(record)
            return plan, auth, pod, record
        raise RemoteError("no pod could be created: " + "; ".join(failures))


def run_api(args: argparse.Namespace, plan: Plan) -> int:
    api = RunpodApi(_require_key())
    try:
        plan, auth, pod, record = approve_and_create(api, plan, args)
        pod_id = pod["id"]
        start = _parse_iso(record["start_utc"])
        deadline_at = _bound(record)
        deadline = Deadline((deadline_at - _now()).total_seconds())
        outcome = "failed"
        remote = None
        ctx: PodContext | None = None
        try:
            print(f"created pod {pod_id} ({record['name']}) on {plan.gpu.id}; the pod's watchdog "
                  f"terminates it at {_iso(deadline_at)}", flush=True)
            pod = _wait_for_pod(api, pod_id, deadline)
            volumes = volume_mounts(pod)
            if pod.get("cloud") != "SECURE" or volumes:
                raise PolicyRefusal(f"pod {pod_id} is {pod.get('cloud')} with volumes {volumes}")
            compute = float(pod.get("cost") or plan.gpu.usd_per_hour)
            disk_gb = int(pod.get("disk") or plan.disk_gb)
            rate = compute + disk_gb * _DISK_USD_PER_GB_HOUR
            if rate > record["usd_per_hour"] + 0.005:
                usd = rate * plan.estimate.hours
                if usd > auth.cap_usd:
                    raise PolicyRefusal(f"the pod bills ${rate:.3f}/h: estimate ${usd:.2f} over the cap")
                earlier = _whole_second(start + timedelta(hours=auth.cap_usd / rate))
                if earlier < deadline_at:
                    deadline_at = earlier
                    deadline.shorten((earlier - _now()).total_seconds())
            record.update(usd_per_hour=rate, compute_usd_per_hour=compute, disk_gb=disk_gb,
                          deadline_utc=_iso(deadline_at))
            _write_open(record)
            direct = pod["ssh"]["direct"]
            remote = Remote(direct["host"], int(direct["port"]), direct.get("username") or "root",
                            args.ssh_key, REMOTE_ROOT, deadline)
            remote.wait_ready(600)
            facts = preflight(remote, pod_id)
            arm_watchdog(remote, deadline_at)
            region = (pod.get("dataCenterId") or (facts.get("pod") or {}).get("dataCenterId")
                      or facts.get("data_center"))
            print(f"pod {pod_id}: {(facts.get('gpus') or [{}])[0].get('name')}, driver CUDA {facts.get('cuda')}, "
                  f"{facts.get('cpu_model')}; region {region} (REST {pod.get('dataCenterId')}, RUNPOD_DC_ID "
                  f"{facts.get('data_center')}); scoped key {facts.get('pod_api')}; watchdog armed", flush=True)
            ctx = PodContext(mode="api", pod_id=pod_id, since=start, compute_usd_per_hour=compute,
                             disk_gb=disk_gb, rate_source="pod cost plus container disk", cloud=pod.get("cloud"),
                             gpu_type=plan.gpu.id, image=pod.get("image") or args.image, region=region,
                             facts=facts)
            execute(remote, plan, ctx)
            outcome = "ok"
        except KeyboardInterrupt:
            print("interrupted", flush=True)
        except Exception as error:  # every failure ends in teardown and a non-zero exit
            print(f"error: {error}", flush=True)
        finally:
            deadline.cancel()
            if remote is not None:
                if outcome != "ok" and not deadline.expired.is_set():
                    try:
                        wipe(remote, tracked=False)
                    except (RemoteError, OSError, KeyboardInterrupt) as error:
                        print(f"wipe failed ({error}); terminating deletes the container disk", flush=True)
                remote.close()
            terminated = _terminate_confirmed(api, pod_id)
            end = _now()
            record["outcome"] = outcome
            if ctx is not None and ctx.upload is not None:
                record["upload"] = ctx.upload
                _write_open(record)
            if terminated:
                _close_at(record, end, "terminated", outcome)
            else:
                record["state"] = "termination not confirmed"
                _write_open(record)
            print(f"pod {pod_id}: {(end - start).total_seconds() / 60:.1f} min, ${_cost(record, end):.4f}",
                  flush=True)
        return 0 if outcome == "ok" and terminated else 1
    finally:
        api.close()


# --- run: SSH mode -------------------------------------------------------


def run_ssh(args: argparse.Namespace) -> int:
    start = _now()
    probe = Remote(args.host, args.port, args.user, args.ssh_key, args.remote_root, Deadline(900))
    try:
        probe.wait_ready(120)
        facts = preflight(probe, args.pod_id)
    finally:
        probe.deadline.cancel()
        probe.close()
    pod = facts.get("pod") or {}
    if pod.get("cloud") != "SECURE" and not args.confirm_secure_cloud:
        raise PolicyRefusal(f"cannot confirm Secure Cloud from the pod ({facts.get('pod_api')}); "
                            "pass --confirm-secure-cloud only if you created it there")
    volumes = volume_mounts(pod)
    if volumes:
        raise PolicyRefusal(f"pod {args.pod_id} has volumes: {volumes}")
    if pod.get("cost"):
        compute, disk_gb, rate_source = float(pod["cost"]), int(pod.get("disk") or 0), "pod cost plus container disk"
        if not pod.get("disk"):
            print("note: the pod reports no container disk size; the disk is not counted", flush=True)
    elif args.rate is not None:
        compute, disk_gb, rate_source = args.rate, 0, "--rate (the whole rate)"
    else:
        raise PolicyRefusal(f"the pod's rate is unknown ({facts.get('pod_api')}); pass --rate")
    # The pod has billed since its creation: that time counts toward the
    # estimate, the approval and the cap.
    since = _parse_iso(pod["createdAt"]) if pod.get("createdAt") else (
        start - timedelta(seconds=facts.get("container_seconds") or 0.0))
    elapsed_s = max(0.0, (start - since).total_seconds())
    gpu = (facts.get("gpus") or [{}])[0]
    gpu_model = gpu.get("name") or ""
    if "GeForce" not in gpu_model and not args.allow_datacenter:
        raise PolicyRefusal(f"the pod's GPU {gpu_model!r} is not a consumer GeForce card; "
                            "it needs --allow-datacenter and --approve-cost")
    offer = GpuOffer(pod.get("gpu") or gpu_model, gpu_model, int(float(gpu.get("memory_mib") or 0) / 1024),
                     compute, None)
    plan = price(prepare(args, args.remote_root), offer, args,
                 provision=(f"pod time before this run (created {_iso(since)})", elapsed_s), pod_disk_gb=disk_gb,
                 facts=facts)
    print_plan(plan)
    if isinstance(plan.work, TaskWork):
        ask_credentials(plan.work)
    found = api_key()
    api = RunpodApi(found[0]) if found else None
    try:
        with _approval_lock():
            try:
                _settle(api)
            except (RunpodError, httpx.HTTPError) as error:
                print(f"note: open pods not checked against the API ({error})", flush=True)
                _settle(None)
            # This pod's own open record (a previous run on it) is replaced:
            # the new one covers the pod from its creation.
            _, _, opened = spend_totals()
            previous = [record for record in opened if record.get("pod_id") == args.pod_id]
            committed, _, _ = spend_totals(at_most=True)
            committed -= sum(_open_cost(record, _now(), at_most=True) for record in previous
                             if _parse_iso(record["start_utc"]).astimezone().date() == _now().astimezone().date())
            auth = authorize(estimate_usd=plan.usd, estimate_hours=plan.estimate.hours,
                             usd_per_hour=plan.usd_per_hour, consumer="GeForce" in gpu_model,
                             today_usd=committed, approve=args.approve_cost, day_cap_usd=args.day_cap)
            print_authorization(plan, auth, committed)
            deadline_at = _whole_second(since + timedelta(hours=auth.cap_hours))
            if (deadline_at - _now()).total_seconds() < 300:
                raise PolicyRefusal(f"pod {args.pod_id} (created {_iso(since)}) reaches its cap at "
                                    f"{_iso(deadline_at)}: no time left for this run")
            jobs = [job for record in previous for job in record.get("jobs", [])]
            record = _new_record(
                name=None, pod_id=args.pod_id, mode="ssh", gpu=gpu_model, compute_usd_per_hour=compute,
                disk_gb=disk_gb, start=since, deadline=deadline_at, estimated_usd=plan.usd, auth=auth,
                jobs=jobs + plan.job_ids)
            for old in previous:
                _discard_open(old)
            _write_open(record)
    finally:
        if api is not None:
            api.close()
    deadline = Deadline((deadline_at - _now()).total_seconds())
    remote = Remote(args.host, args.port, args.user, args.ssh_key, args.remote_root, deadline)
    outcome = "failed"
    ctx: PodContext | None = None
    try:
        arm_watchdog(remote, deadline_at)
        print(f"pod {args.pod_id}: {gpu_model}, driver CUDA {facts.get('cuda')}, {facts.get('cpu_model')}; "
              f"watchdog armed for {_iso(deadline_at)}", flush=True)
        ctx = PodContext(mode="ssh", pod_id=args.pod_id, since=since, compute_usd_per_hour=compute,
                         disk_gb=disk_gb, rate_source=rate_source, cloud=pod.get("cloud"),
                         gpu_type=pod.get("gpu"), image=pod.get("image"),
                         region=pod.get("dataCenterId") or facts.get("data_center"), facts=facts)
        execute(remote, plan, ctx)
        outcome = "ok"
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    except Exception as error:  # every failure ends in cleanup and a non-zero exit
        print(f"error: {error}", flush=True)
    finally:
        deadline.cancel()
        end = _now()
        linger_at = min(deadline_at, _whole_second(end + timedelta(minutes=args.linger_minutes)))
        if not deadline.expired.is_set():
            try:
                if outcome != "ok":
                    wipe(remote, tracked=False)
                remote.script(_EARLIER_DEADLINE, remote.root, str(_epoch(linger_at)),
                              what="watchdog linger", timeout=120, tracked=False)
                record["deadline_utc"] = _iso(linger_at)
                print(f"the pod's watchdog terminates it at {_iso(linger_at)} unless it is terminated first; "
                      f"after terminating it, run `remote terminate {args.pod_id}` to record its end", flush=True)
            except (RemoteError, OSError, KeyboardInterrupt) as error:
                print(f"cleanup failed ({error}); the watchdog still terminates the pod at "
                      f"{record['deadline_utc']}", flush=True)
        remote.close()
        # The pod bills until it is terminated: the record stays open, at
        # most until its deadline (``spend``, ``terminate``, ``_settle``).
        record.update(state="lingering", outcome=outcome, client_end_utc=_iso(end))
        if ctx is not None and ctx.upload is not None:
            record["upload"] = ctx.upload
        _write_open(record)
    return 0 if outcome == "ok" else 1


# --- Other commands ------------------------------------------------------


def cmd_whoami(_: argparse.Namespace) -> int:
    found = api_key()
    if found is None:
        print("no RunPod API key: set RUNPOD_API_KEY or ~/.runpod/config.toml apikey")
        return 1
    api = RunpodApi(found[0])
    try:
        pods = api.pods()
    finally:
        api.close()
    print(f"API key from {found[1]} accepted by REST v2; {len(pods)} pod(s) on the account")
    return 0


def cmd_gpus(args: argparse.Namespace) -> int:
    offers = secure_gpus(args.min_cuda)
    print(f"Secure Cloud NVIDIA GPU types, 1 GPU, host CUDA >= {args.min_cuda} (list price, stock now):")
    for offer in offers:
        if not args.all and not offer.in_stock:
            continue
        if not args.allow_datacenter and not args.all and not offer.consumer:
            continue
        kind = "consumer" if offer.consumer else "datacenter"
        print(f"  ${offer.usd_per_hour:5.2f}/h  {offer.stock or 'None':<6}  {offer.memory_gb:>3} GB  "
              f"{kind:<10}  {offer.id}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan = make_plan(args, REMOTE_ROOT)
    print_plan(plan)
    committed, _, _ = spend_totals(at_most=True)
    try:
        auth = authorize(estimate_usd=plan.usd, estimate_hours=plan.estimate.hours,
                         usd_per_hour=plan.usd_per_hour, consumer=plan.gpu.consumer, today_usd=committed,
                         approve=args.approve_cost, day_cap_usd=args.day_cap)
    except PolicyRefusal as refusal:
        print(f"policy: {refusal}")
        return 0
    print_authorization(plan, auth, committed)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.tasks is not None:
        hold_credentials()
    ssh_mode = args.host is not None
    if ssh_mode and (args.pod_id is None):
        raise PolicyRefusal("SSH mode needs --pod-id (the pod the host must be)")
    if not args.ssh_key.is_file():
        raise PolicyRefusal(f"no SSH key at {args.ssh_key}")
    status = subprocess.run(["git", "-C", str(_REPO_ROOT), "status", "--porcelain", "--", "yamabiko", "scripts"],
                            capture_output=True, text=True).stdout
    if status.strip():
        print(f"note: uncommitted changes under yamabiko/ or scripts/ are not uploaded ({args.commit})")
    if ssh_mode:
        return run_ssh(args)
    plan = make_plan(args, REMOTE_ROOT)
    print_plan(plan)
    if isinstance(plan.work, TaskWork):
        ask_credentials(plan.work)
    return run_api(args, plan)


def _settle_now() -> None:
    """Settle the open records, against the API when a key is set."""
    found = api_key()
    api = RunpodApi(found[0]) if found else None
    try:
        with _approval_lock():
            try:
                _settle(api)
            except (RunpodError, httpx.HTTPError) as error:
                print(f"note: open pods not checked against the API ({error})")
                _settle(None)
    finally:
        if api is not None:
            api.close()


def cmd_spend(_: argparse.Namespace) -> int:
    _settle_now()
    now = _now()
    today, month, opened = spend_totals(now)
    committed, _, _ = spend_totals(now, at_most=True)
    closed, _ = _spend_records()
    local = now.astimezone()
    for record in closed:
        start = _parse_iso(record["start_utc"]).astimezone()
        if start.date() == local.date():
            print(f"  {start:%H:%M} {record['pod_id']} {record['gpu']} ${record['usd_per_hour']:.3f}/h "
                  f"est ${record['estimated_usd']:.2f} actual ${record['actual_usd']:.4f} "
                  f"{record.get('outcome')} {', '.join(record['jobs'])}")
    for record in opened:
        print(f"  open: {record.get('pod_id') or record.get('name')} {record['gpu']} "
              f"{record.get('state', 'running')} since {record['start_utc']}: "
              f"${_open_cost(record, now, at_most=False):.4f} so far, at most "
              f"${_open_cost(record, now, at_most=True):.4f} (deadline {_iso(_bound(record))})")
    print(f"today ({local:%Y-%m-%d}): ${today:.4f} spent so far; ${committed:.4f} with open pods at their "
          f"caps (the approval figure), of ${ROUTINE_DAY_USD:.2f}")
    print(f"this month ({local:%Y-%m}): ${month:.4f}")
    print(f"log: {_spend_dir() / 'spend.jsonl'}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    api = RunpodApi(_require_key())
    try:
        pods = api.pods()
    finally:
        api.close()
    for pod in pods:
        print(f"{pod['id']}  {pod.get('status'):<12} {(pod.get('gpu') or {}).get('id')}  ${pod.get('cost')}/h  "
              f"{pod.get('cloud')}  {pod.get('dataCenterId')}  {pod.get('name')}  created {pod.get('createdAt')}")
    print(f"{len(pods)} pod(s)")
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    """Terminate pods (also pods already terminated elsewhere) and close
    their open spend records: at now for a pod that was still running, else
    at the earlier of now and the record's deadline."""
    api = RunpodApi(_require_key())
    ok = True
    try:
        for pod_id in args.pod_ids:
            pod = api.pod(pod_id)
            running = pod is not None and pod.get("status") in ("PROVISIONING", "STARTING", "RUNNING")
            if not _terminate_confirmed(api, pod_id):
                ok = False
                continue
            with _approval_lock():
                _, _, opened = spend_totals()
                for record in opened:
                    if record.get("pod_id") == pod_id:
                        now = _now()
                        if running:
                            _close_at(record, now, "terminated by `remote terminate`")
                        else:
                            _close_at(record, min(now, _bound(record)),
                                      "gone at `remote terminate`; end bounded by its deadline")
                        print(f"closed the spend record of pod {pod_id}", flush=True)
        return 0 if ok else 1
    finally:
        api.close()


def _windows(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "windows.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def cmd_compare(args: argparse.Namespace) -> int:
    """Windows and texts of two runs of the same candidate on the same set."""
    first, second = _windows(args.runs[0]), _windows(args.runs[1])
    # emitted_at_s exists from run schema 1.1 on.
    key = [k for k in ("item_id", "window_index", "start_s", "end_s", "emit_reason", "emitted_at_s")
           if first and second and k in first[0] and k in second[0]]
    spans = sum(1 for a, b in zip(first, second) if all(a[k] == b[k] for k in key))
    texts = [(a, b) for a, b in zip(first, second) if a["text"] != b["text"]]
    print(f"windows: {len(first)} vs {len(second)}; same {', '.join(key[2:])}: {spans}; "
          f"different text: {len(texts)}")
    for a, b in texts[:20]:
        print(f"  {a['item_id']} #{a['window_index']}:\n    {a['text']}\n    {b['text']}")
    same = len(first) == len(second) == spans and not texts
    print("identical" if same else "different")
    return 0 if same else 1


# --- CLI ---------------------------------------------------------------


def _job_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", type=Path,
                        help='JSON file: {"jobs": [{"set", "candidate", "name", "out", "id"?, "assume_rtf"?, '
                             '"workers"?, "seed"?}]}')
    parser.add_argument("--job", nargs=4, action="append", metavar=("SET", "CANDIDATE", "NAME", "OUT"))
    parser.add_argument("--tasks", type=Path,
                        help='pod tasks instead of bench jobs, JSON file: {"disk_gb": N, "tasks": [{"id", '
                             '"module", "env", "args", "inputs"?, "secrets"?, "mdc_dataset"?, "hours", "out"}]}')
    parser.add_argument("--gpu", help="GPU type id (default: the cheapest consumer GeForce in stock)")
    parser.add_argument("--allow-datacenter", action="store_true",
                        help="allow datacenter GPUs (not routine: needs --approve-cost)")
    parser.add_argument("--min-vram", type=int, default=MIN_VRAM_GB)
    parser.add_argument("--uplink-mbps", type=float,
                        help="upload rate for the estimate (default: the median of the last "
                             f"{_UPLINK_SAMPLES} measured uploads in the spend log, else {_UPLINK_MBPS:g})")
    parser.add_argument("--workers", type=workers_value, default=1,
                        help=f"runner processes per job without its own workers: N, or {AUTO} for "
                             "accuracy-only jobs (as many as the pod's GPU memory and vCPUs fit); "
                             "above 1 the runs' timing is not comparable (default 1)")
    parser.add_argument("--assume-rtf", type=float, help="RTF for a candidate without a past run")
    parser.add_argument("--commit", default="HEAD", help="code revision to upload (git archive)")
    parser.add_argument("--approve-cost", type=float, metavar="USD",
                        help="the owner's approval for a job outside the routine policy")
    parser.add_argument("--day-cap", type=day_cap_value, default=ROUTINE_DAY_USD, metavar="USD",
                        help=f"the day cap of the routine policy (default ${ROUTINE_DAY_USD:.2f}); a higher "
                             f"one only with the owner's approval, at most ${DAY_CAP_MAX_USD:.2f}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("whoami", help="check the API key")
    gpus = sub.add_parser("gpus", help="Secure Cloud GPU types, price and stock")
    gpus.add_argument("--min-cuda", default=MIN_CUDA)
    gpus.add_argument("--allow-datacenter", action="store_true", help="also list datacenter GPUs in stock")
    gpus.add_argument("--all", action="store_true", help="every type, in stock or not")
    _job_options(sub.add_parser("plan", help="price jobs on one pod"))
    run = sub.add_parser("run", help="run jobs on one pod")
    _job_options(run)
    run.add_argument("--gpu-strict", action="store_true",
                     help="API mode: only the planned GPU, no fallback when RunPod has no capacity")
    run.add_argument("--image", default=IMAGE)
    run.add_argument("--ssh-key", type=Path, default=SSH_KEY)
    run.add_argument("--host", help="SSH mode: the pod's public IP")
    run.add_argument("--port", type=int, default=22, help="SSH mode: the pod's public port for 22/tcp")
    run.add_argument("--user", default="root")
    run.add_argument("--pod-id", help="SSH mode: the pod the host must be")
    run.add_argument("--rate", type=float,
                     help="SSH mode: the pod's whole $/h (compute and disk) if the pod cannot report it")
    run.add_argument("--confirm-secure-cloud", action="store_true",
                     help="SSH mode: the pod is on Secure Cloud though it cannot report it")
    run.add_argument("--remote-root", default=REMOTE_ROOT, help="SSH mode: work directory on the pod")
    run.add_argument("--linger-minutes", type=int, default=LINGER_MINUTES,
                     help="SSH mode: the watchdog terminates the pod this long after the run")
    sub.add_parser("spend", help="today's and this month's spend")
    sub.add_parser("list", help="the account's pods")
    terminate = sub.add_parser("terminate", help="terminate pods and close their spend records")
    terminate.add_argument("pod_ids", nargs="+")
    compare = sub.add_parser("compare", help="windows and texts of two runs")
    compare.add_argument("runs", nargs=2, type=Path)
    args = parser.parse_args(argv)
    commands = {
        "whoami": cmd_whoami, "gpus": cmd_gpus, "plan": cmd_plan, "run": cmd_run, "spend": cmd_spend,
        "list": cmd_list, "terminate": cmd_terminate, "compare": cmd_compare,
    }
    try:
        return commands[args.command](args)
    except PolicyRefusal as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    except OutboxRefusal as refusal:
        print(f"refused: OutboxRefusal: {refusal}", file=sys.stderr)
        return 2
    except (RemoteError, RunpodError, httpx.HTTPError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
