"""Sound-tagger worker for ``tag_sounds.py``: runs one AudioSet Strong frame
tagger from the EfficientSED code (``theMoro/EfficientSED``, which also
holds the PretrainedSED transformers) over bench items inside the tagger's
own venv.

Usage (``tag_sounds.py`` starts it; it is not meant to be run by hand)::

    <tagger venv python> scripts/asr_bench/tag_sounds_worker.py --job <job.json>

A job is ``{"repo", "checkpoint", "model", "seq_model_type",
"seq_model_dim", "hop_frames", "lookahead_frames", "batch", "device":
"cuda" | "cpu", "threads", "groups": [[name, [class, ...] | null], ...], "items":
[{"item_id", "input", "sample_count", "output"}]}``, optionally with
``"tail_frames"`` (default 0). ``repo`` is the
EfficientSED checkout, ``checkpoint`` the weights file, ``model`` ``fmn04``
to ``fmn30`` or ``BEATs``. Each input is a bench item file (16 kHz mono
float32 LE); each output is an ``.npz`` the worker writes:

- ``probs``: ``(frames, 447)`` float16 class probabilities (sigmoid of the
  strong head), class order ``as_strong_train_classes``;
- ``groups``: ``(frames, len(groups))`` float32, per group (in job order;
  a null group holds every class in no other group) the maximum
  probability over its classes, computed from the float32 probabilities;
- ``group_names``: the group names in column order;
- ``frame_samples``: 640 (one 40 ms frame at 16 kHz);
- ``tail_frames``: the job's, when it is not 0.

Frame ``j`` covers item samples ``[640 j, 640 (j + 1))``; an item of ``N``
samples has ``ceil(N / 640)`` frames, followed by ``tail_frames`` frames
of the zeros after it. The model sees exactly 10 s (250
frames) at a time. Block ``b`` keeps frames ``[b H, (b + 1) H)`` (``H`` =
``hop_frames``) from the window that ends ``R`` = ``lookahead_frames``
frames after the block, so every frame is predicted with ``R`` to
``R + H - 1`` frames of future audio, as a streaming tagger that runs
every ``H`` frames and waits ``R`` frames would. Audio outside the item is
zeros.

Progress goes to stdout as one JSON object per line: a ``ready`` event
with the runtime versions, the class list and the model load time, then
one ``item`` event per file with its tagging time and the CUDA memory
peaks. Everything else goes to stderr. The worker imports only the standard
library, NumPy, PyTorch and the EfficientSED modules, so it runs without
the bench's own packages.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

FRAME_SAMPLES = 640
WINDOW_FRAMES = 250
WINDOW_SAMPLES = FRAME_SAMPLES * WINDOW_FRAMES


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
        "numpy": version("numpy"),
        "einops": version("einops"),
        "rotary_embedding_torch": version("rotary-embedding-torch"),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


def load_model(job: dict[str, Any], torch: Any) -> tuple[Any, list[str]]:
    repo = Path(job["repo"])
    sys.path.insert(0, str(repo))
    # prediction_wrapper imports mamba_ssm at module level; the Mamba
    # sequence model is never built here, so a stub stands in for the
    # package (which needs triton and a CUDA build toolchain).
    stub = types.ModuleType("mamba_ssm")

    def _no_mamba(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the Mamba sequence model is not supported")

    stub.Mamba2 = _no_mamba  # type: ignore[attr-defined]
    sys.modules.setdefault("mamba_ssm", stub)

    from data_util.audioset_classes import as_strong_train_classes
    from models import prediction_wrapper
    from models.prediction_wrapper import PredictionsWrapper

    checkpoint = Path(job["checkpoint"])
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} must be an existing .pt file")
    # load_checkpoint reads <RESOURCES_FOLDER>/<name>.pt and only downloads
    # when the file is missing; point it at the checkpoint's own folder.
    prediction_wrapper.RESOURCES_FOLDER = str(checkpoint.parent)
    name = checkpoint.stem
    model_name = job["model"]
    if model_name == "BEATs":
        from models.transformers.beats.BEATs_wrapper import BEATsWrapper

        model = PredictionsWrapper(BEATsWrapper(), checkpoint=name)
    elif model_name.startswith("fmn"):
        from models.efficient_cnns.fmn.fmn_wrapper import FrameMNWrapper
        from models.efficient_cnns.fmn.utils import NAME_TO_WIDTH

        fmn = FrameMNWrapper(NAME_TO_WIDTH(model_name))
        embed_dim = fmn.state_dict()["fmn.features.16.1.bias"].shape[0]
        model = PredictionsWrapper(
            fmn, checkpoint=name, seq_model_type=job["seq_model_type"],
            seq_model_dim=job["seq_model_dim"], embed_dim=embed_dim,
        )
    else:
        raise ValueError(f"unsupported model {model_name!r}")
    model.eval()
    return model, list(as_strong_train_classes)


def pad_item(audio: Any, frames: int, hop: int, lookahead: int, np: Any) -> Any:
    """The item with one window of zeros before it and enough after it for
    the last block's window; item sample 0 sits at ``WINDOW_SAMPLES``."""
    blocks = math.ceil(frames / hop)
    padded = np.zeros(WINDOW_SAMPLES + (blocks * hop + lookahead) * FRAME_SAMPLES,
                      dtype=np.float32)
    padded[WINDOW_SAMPLES:WINDOW_SAMPLES + audio.size] = audio
    return padded


def block_windows(padded: Any, first: int, count: int, hop: int, lookahead: int, np: Any) -> Any:
    """The windows of blocks ``first`` to ``first + count - 1``: block ``b``'s
    window ends ``lookahead`` frames after item frame ``(b + 1) hop``."""
    windows = np.empty((count, WINDOW_SAMPLES), dtype=np.float32)
    for row, block in enumerate(range(first, first + count)):
        end = WINDOW_SAMPLES + ((block + 1) * hop + lookahead) * FRAME_SAMPLES
        windows[row] = padded[end - WINDOW_SAMPLES:end]
    return windows


def group_columns(classes: list[str], groups: list[list[Any]], np: Any) -> tuple[list[str], list[Any]]:
    """The group names in order and each group's class columns. ``groups``
    is ``[[name, [class, ...] | null], ...]``; a null group holds every
    class in no other group."""
    index = {name: position for position, name in enumerate(classes)}
    by_name = {name: members for name, members in groups}
    names = [name for name, _ in groups]
    if len(by_name) != len(names):
        raise ValueError("group names repeat")
    named = {cls for members in by_name.values() if members is not None for cls in members}
    missing = sorted(named - set(index))
    if missing:
        raise ValueError(f"group classes not in the model's class list: {missing}")
    members = {name: [cls for cls in classes if cls not in named] if by_name[name] is None
               else by_name[name] for name in names}
    return names, [np.array([index[cls] for cls in members[name]]) for name in names]


def group_probs(probs: Any, columns: list[Any], np: Any) -> Any:
    """``(frames, len(columns))``: each group's maximum class probability."""
    return np.stack([probs[:, group].max(axis=1) for group in columns], axis=1)


def write_tags(output: Path, probs: Any, groups: Any, group_names: list[str], np: Any,
               tail_frames: int = 0) -> None:
    """Write one item's ``.npz`` (layout in the module docstring) through a
    temporary file."""
    temporary = output.with_name(output.stem + ".part.npz")
    tail = {"tail_frames": np.int64(tail_frames)} if tail_frames else {}
    np.savez(temporary, probs=probs.astype(np.float16), groups=groups.astype(np.float32),
             group_names=np.array(group_names), frame_samples=np.int64(FRAME_SAMPLES), **tail)
    temporary.replace(output)


def run_job(job: dict[str, Any], out: TextIO) -> int:
    import numpy as np
    import torch

    torch.set_num_threads(int(job["threads"]))
    device = job["device"]
    if device not in ("cuda", "cpu"):
        raise ValueError(f"device must be cuda or cpu, got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda requested but CUDA is not available")
    cuda = device == "cuda"
    hop, lookahead, batch = int(job["hop_frames"]), int(job["lookahead_frames"]), int(job["batch"])
    if not (hop >= 1 and lookahead >= 0 and hop + lookahead <= WINDOW_FRAMES and batch >= 1):
        raise ValueError("need hop_frames >= 1, lookahead_frames >= 0, their sum <= 250, batch >= 1")

    begin = time.perf_counter()
    model, classes = load_model(job, torch)
    model.to(device)
    load_s = time.perf_counter() - begin
    # The first forward pass loads CUDA kernels (seconds); keep it out of
    # the per-item timing.
    begin = time.perf_counter()
    with torch.inference_mode():
        model(model.mel_forward(torch.zeros(batch, WINDOW_SAMPLES, device=device)))
    if cuda:
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - begin
    group_names, columns = group_columns(classes, job["groups"], np)
    _emit(out, {
        "event": "ready", "versions": _versions(torch), "device": device,
        "classes": classes, "torch_threads": torch.get_num_threads(), "load_s": load_s,
        "warmup_s": warmup_s,
    })

    keep_begin = WINDOW_FRAMES - lookahead - hop
    for entry in job["items"]:
        io_begin = time.perf_counter()
        audio = np.fromfile(entry["input"], dtype="<f4")
        if audio.size != entry["sample_count"]:
            raise ValueError(f"{entry['input']}: {audio.size} samples, expected "
                             f"{entry['sample_count']}")
        tail = int(job.get("tail_frames", 0))
        frames = math.ceil(audio.size / FRAME_SAMPLES) + tail
        blocks = math.ceil(frames / hop)
        padded = pad_item(audio, frames, hop, lookahead, np)
        io_s = time.perf_counter() - io_begin
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        begin = time.perf_counter()
        kept = []
        with torch.inference_mode():
            for first in range(0, blocks, batch):
                count = min(batch, blocks - first)
                chunk = torch.from_numpy(block_windows(padded, first, count, hop, lookahead, np))
                logits, _ = model(model.mel_forward(chunk.to(device)))
                # (batch, classes, 250) -> the kept frames of each window.
                part = torch.sigmoid(logits[:, :, keep_begin:keep_begin + hop].float())
                kept.append(part.permute(0, 2, 1).reshape(-1, len(classes)).cpu())
        if cuda:
            torch.cuda.synchronize()
        tag_s = time.perf_counter() - begin
        probs = torch.cat(kept).numpy()[:frames]
        groups = group_probs(probs, columns, np)
        io_begin = time.perf_counter()
        write_tags(Path(entry["output"]), probs, groups, group_names, np, tail)
        io_s += time.perf_counter() - io_begin
        _emit(out, {
            "event": "item", "item_id": entry["item_id"], "tag_s": tag_s, "io_s": io_s,
            "frames": frames, "windows": blocks,
            "vram_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if cuda else None),
            "vram_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20 if cuda else None),
        })
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", type=Path, required=True, help="job JSON")
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        return run_job(job, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
