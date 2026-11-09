"""Separator worker for ``separate.py``: runs one MSST model (the
Music-Source-Separation-Training package, ``pip install msst``) over a list
of WAV files inside the separator's own venv.

Usage (``separate.py`` starts it; it is not meant to be run by hand)::

    <separator venv python> scripts/asr_bench/separate_msst.py --job <job.json>
    <separator venv python> scripts/asr_bench/separate_msst.py
        --convert-lightning <ckpt> --out <pt> [--prefix model.]

A job is ``{"model_type", "config", "checkpoint", "stem", "device": "cuda" |
"cpu", "threads", "items": [{"item_id", "input", "output"}]}``. Each input is
a WAV already at the model's sample rate; the worker writes the selected
stem as a float32 WAV at the same rate, unscaled. Progress goes to stdout as
one JSON object per line: a ``ready`` event with the runtime versions and
the model load time, then one ``item`` event per file with its separation
time and the CUDA memory peaks. Everything MSST prints goes to stderr.

``--convert-lightning`` turns a PyTorch Lightning checkpoint (weights under
``state_dict``, prefixed by the Lightning module attribute) into the plain
``{"state_dict": ...}`` form MSST loads, keeping only the prefixed keys.

The worker imports only the standard library, NumPy, SoundFile, PyTorch and
MSST, so it runs without the bench's own packages.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO


def _emit(stream: TextIO, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def _versions(torch: Any) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "msst": version("msst"),
        "numpy": version("numpy"),
        "soundfile": version("soundfile"),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


def convert_lightning(source: Path, target: Path, prefix: str) -> int:
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not weights:
        raise ValueError(f"{source}: no state_dict key starts with {prefix!r}")
    torch.save({"state_dict": weights}, target)
    print(f"kept {len(weights)} of {len(state)} tensors under {prefix!r}; wrote {target}")
    return 0


def run_job(job: dict[str, Any], out: TextIO) -> int:
    import numpy as np
    import soundfile
    import torch
    from msst import Separator

    torch.set_num_threads(int(job["threads"]))
    device = job["device"]
    if device not in ("cuda", "cpu"):
        raise ValueError(f"device must be cuda or cpu, got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda requested but CUDA is not available")
    cuda = device == "cuda"

    begin = time.perf_counter()
    separator = Separator(
        model_type=job["model_type"],
        config_path=job["config"],
        checkpoint_path=job["checkpoint"],
        device_ids=0,
        force_cpu=not cuda,
        detailed_progress=False,
    )
    load_s = time.perf_counter() - begin
    stem = job["stem"]
    if stem not in separator.instruments:
        raise ValueError(f"stem {stem!r} not in the model's stems {separator.instruments}")
    _emit(out, {
        "event": "ready", "versions": _versions(torch), "device": separator.device,
        "sample_rate": separator.sample_rate, "instruments": list(separator.instruments),
        "torch_threads": torch.get_num_threads(), "load_s": load_s,
    })

    for entry in job["items"]:
        io_begin = time.perf_counter()
        audio, rate = soundfile.read(entry["input"], dtype="float32", always_2d=True)
        if rate != separator.sample_rate:
            raise ValueError(f"{entry['input']} is {rate} Hz, the model needs "
                             f"{separator.sample_rate} Hz")
        io_s = time.perf_counter() - io_begin
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        begin = time.perf_counter()
        stems = separator.separate(audio.T)
        if cuda:
            torch.cuda.synchronize()
        separate_s = time.perf_counter() - begin
        estimate = np.asarray(stems[stem], dtype=np.float32)
        if estimate.ndim == 1:
            estimate = estimate[np.newaxis, :]
        io_begin = time.perf_counter()
        soundfile.write(entry["output"], estimate.T, rate, subtype="FLOAT")
        io_s += time.perf_counter() - io_begin
        _emit(out, {
            "event": "item", "item_id": entry["item_id"], "separate_s": separate_s,
            "io_s": io_s, "channels": int(estimate.shape[0]),
            "samples": int(estimate.shape[1]),
            "vram_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if cuda else None),
            "vram_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20 if cuda else None),
        })
    separator.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", type=Path, help="job JSON")
    parser.add_argument("--convert-lightning", type=Path, help="Lightning checkpoint to convert")
    parser.add_argument("--out", type=Path, help="converted checkpoint path")
    parser.add_argument("--prefix", default="model.", help="state_dict key prefix to keep")
    args = parser.parse_args(argv)
    if args.convert_lightning is not None:
        if args.out is None:
            parser.error("--convert-lightning needs --out")
        return convert_lightning(args.convert_lightning, args.out, args.prefix)
    if args.job is None:
        parser.error("give --job or --convert-lightning")
    job = json.loads(args.job.read_text(encoding="utf-8"))
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        return run_job(job, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
