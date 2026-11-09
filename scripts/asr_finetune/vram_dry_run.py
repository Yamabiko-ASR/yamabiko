"""Step 5 of the fine-tuning pilot: peak VRAM of a realistic LoRA training
configuration, from a short dry run on random audio.

Usage::

    python -m scripts.asr_finetune.vram_dry_run [--work-dir <dir>] [--batch 8] [--steps 20]
        [--label-tokens 448] [--precision fp16] [--precision bf16]

Needs step 2's source snapshot. Each precision runs in a fresh process: a
batch of ``--batch`` 30 s windows of Gaussian noise (log-mel features from
the model's own feature extractor), random text-token labels of
``--label-tokens`` tokens each (448, the decoder's maximum, is the worst
case), the pilot adapter (``lora.py``), gradient checkpointing, AdamW and
``--steps`` optimizer steps. ``fp16``: float16 weights and autocast with a
gradient scaler (step 4's configuration); ``bf16``: bfloat16 weights and
autocast. Recorded: PyTorch's peak allocated and reserved memory, the
median step time, the card total and what other processes held at the
start (``torch.cuda.mem_get_info``). A card fits when its size in GiB holds
the peak reserved memory plus ``ALLOWANCE_GIB`` for the CUDA context, the
driver's reserve and a desktop. The record goes to
``results\\vram_dry_run.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import get_peft_model
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from scripts.asr_bench.sets import SAMPLE_RATE_HZ
from scripts.asr_finetune.common import add_common_args, package_versions, require_source, write_result
from scripts.asr_finetune.lora import load_base, lora_config

WINDOW_S = 30
CARDS_GB = (16, 24, 32)
ALLOWANCE_GIB = 1.0
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
_GIB = 2**30


def random_labels(tokenizer: WhisperTokenizerFast, batch: int, tokens: int, rng: np.random.Generator) -> torch.Tensor:
    """``<|ja|><|transcribe|><|notimestamps|>``, random text tokens, ``<|endoftext|>``."""
    prefix = tokenizer.convert_tokens_to_ids(["<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    end = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    text_ids = min(tokenizer.all_special_ids)
    body = rng.integers(0, text_ids, size=(batch, tokens - len(prefix) - 1))
    return torch.tensor([prefix + row.tolist() + [end] for row in body])


def dry_run(source: Path, precision: str, batch: int, steps: int, label_tokens: int) -> dict[str, Any]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    free_at_start, total = torch.cuda.mem_get_info()
    dtype = _DTYPES[precision]
    extractor = WhisperFeatureExtractor.from_pretrained(source)
    tokenizer = WhisperTokenizerFast.from_pretrained(source)
    audio = [0.1 * rng.standard_normal(WINDOW_S * SAMPLE_RATE_HZ).astype(np.float32) for _ in range(batch)]
    features = extractor(audio, sampling_rate=SAMPLE_RATE_HZ, return_tensors="pt").input_features.to("cuda", dtype)
    labels = random_labels(tokenizer, batch, label_tokens, rng).to("cuda")

    base = load_base(source, dtype)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, lora_config()).to("cuda")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model_gib = torch.cuda.memory_allocated() / _GIB

    times: list[float] = []
    losses: list[float] = []
    skipped = 0
    for _ in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            loss = model(input_features=features, labels=labels).loss
        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        losses.append(loss.item())
        skipped += scaler.is_enabled() and scaler.get_scale() < scale
    peak_reserved = torch.cuda.max_memory_reserved() / _GIB
    return {
        "precision": precision,
        "batch": batch,
        "window_s": WINDOW_S,
        "label_tokens": label_tokens,
        "steps": steps,
        "scaler_skipped_steps": int(skipped),
        "losses_finite": all(np.isfinite(losses)),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "step_seconds_median": statistics.median(times[1:] or times),
        "model_and_adapter_gib": model_gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / _GIB,
        "peak_reserved_gib": peak_reserved,
        "fits": {f"{size} GB": peak_reserved + ALLOWANCE_GIB <= size for size in CARDS_GB},
        "card": torch.cuda.get_device_name(0),
        "card_total_gib": total / _GIB,
        "held_by_others_at_start_gib": (total - free_at_start) / _GIB,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser, bench=False)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--label-tokens", type=int, default=448)
    parser.add_argument("--precision", action="append", choices=sorted(_DTYPES),
                        help="repeatable (default: fp16 and bf16)")
    parser.add_argument("--child", choices=sorted(_DTYPES), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source = require_source(args.work_dir)

    if args.child:
        print(json.dumps(dry_run(source, args.child, args.batch, args.steps, args.label_tokens)))
        return 0

    runs = []
    for precision in args.precision or ["fp16", "bf16"]:
        completed = subprocess.run(
            [sys.executable, "-m", "scripts.asr_finetune.vram_dry_run", "--work-dir", str(args.work_dir),
             "--batch", str(args.batch), "--steps", str(args.steps), "--label-tokens", str(args.label_tokens),
             "--child", precision],
            check=True, capture_output=True, text=True,
        )
        runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    path = write_result(args.work_dir, "vram_dry_run", {
        "allowance_gib": ALLOWANCE_GIB, "runs": runs, "packages": package_versions(),
    })
    for run in runs:
        fits = ", ".join(f"{card} {'yes' if ok else 'no'}" for card, ok in run["fits"].items())
        print(f"{run['precision']}: batch {run['batch']} x {run['window_s']} s, {run['label_tokens']} label tokens, "
              f"{run['steps']} steps ({run['scaler_skipped_steps']} skipped), median step {run['step_seconds_median']:.2f} s, "
              f"loss {run['first_loss']:.3f} -> {run['last_loss']:.3f}; peak allocated {run['peak_allocated_gib']:.2f} GiB, "
              f"reserved {run['peak_reserved_gib']:.2f} GiB; fits: {fits}")
    print(f"record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
