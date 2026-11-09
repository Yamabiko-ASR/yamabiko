"""Decoder-only fine-tune of the 760M with the KL anchor (plan D1, D2; pod task T1).

    python -m scripts.asr_finetune.train --config <json> --source <transformers dir> --windows <dir> --out <dir>
        [--checkpoints <dir>] [--max-steps N] [--stop-at EPOCH [--stop-reserve SECONDS]]

Everything that shapes the run comes from ``--config``
(``scripts/asr_finetune/configs/*.json``); a key missing, unknown or set to a
value this trainer does not implement is exit 2. ``--source`` is the
transformers snapshot of ``SOURCE_REPO`` at ``SOURCE_REVISION``, fetched
there first when it holds no weights. ``--windows`` is the output of
``scripts.asr_finetune.windows`` (``audio.i16``, ``index.jsonl``).

Model: the source loads in float16 with SDPA attention and ``use_cache``
off. The encoder stays float16, frozen (``requires_grad_(False)``) and in
eval mode; the decoder (4 layers, token and position embeddings, layer
norms; the output projection is the tied token embedding) becomes the
float32 master the optimizer trains. The teacher is a frozen float32 copy of
that decoder taken before the first step.

A step is ``batch_windows`` train windows; on a card with less than 22 GiB
it is ``grad_accumulation_16gb`` micro-batches of ``batch_windows_16gb``,
the same step. Train windows are shuffled per epoch (``random.Random(seed +
epoch)``); the last step of an epoch may hold fewer windows. Per
micro-batch: the 128-bin log-mel of each window as the shipped decode feeds
the encoder (``log_mel``: the source's ``WhisperFeatureExtractor`` on the
GPU, then every frame from ``samples // 160`` on set to 0.0, which is
faster-whisper's ``content_frames`` slice and ``pad_or_trim`` zero fill);
``encoder_out`` from the encoder under ``torch.no_grad()`` in float16,
cast to bfloat16; student and teacher decoder forwards under bfloat16
autocast on ``<|startoftranscript|>`` followed by the labels shifted right,
with ``encoder_hidden_states=encoder_out``, logits through each decoder's
tied embedding (the teacher only when the micro-batch holds an anchor
window). Labels are each window's ``labels`` with the padding at -100: a
speech window's (``<|ja|>``, ``<|transcribe|>``, checked against the
source's tokenizer) with the first ``labels.MASKED_PREFIX`` positions at
-100 too, or with ``first_position`` ``target`` (optional config key,
default ``masked``; round 4) only ``<|transcribe|>``'s: the first predicted
position, the token after ``<|startoftranscript|>``, is where the decode
reads ``no_speech_prob`` (CTranslate2 takes p(``<|nospeech|>``) there), so
speech windows target ``<|ja|>`` and no-speech windows (``labels``
``[<|nospeech|>, <|endoftext|>]``, accepted only then, never positive or
anchor) target ``<|nospeech|>`` and then ``<|endoftext|>``.

Loss: ``cross_entropy(student, labels) + anchor_lambda(step) *
anchor_kl(student, teacher)``, both token means over the whole step (each
micro-batch's means weighted by its share of the step's label tokens,
anchor label tokens for the KL), so the accumulated layout computes the loss
of one batch. With ``first_position_weight`` w (optional config key, default
1.0; round 5; needs ``first_position`` ``target``), the cross-entropy is the
weighted mean over the step's label positions instead
(``labels.loss_weights``: w at every window's first predicted position,
``<|ja|>`` or ``<|nospeech|>``, 1 elsewhere; the weighted sum over the sum
of the weights); w = 1 runs the token mean above unchanged, and the logged
``ce`` stays the plain token mean. AdamW (fused) on the decoder parameters only, gradients
clipped to ``grad_clip``. ``step`` counts the updates already applied: step
0 runs the unchanged decoder, so its KL is exactly 0. From the first update
on, the KL sits at about 1e-3 or more even for tiny weight moves: under
bfloat16 autocast, a float32 weight that crosses a bfloat16 rounding
boundary changes the student's logits (measured on the smoke windows: about
3e-4 for every weight moved by 5e-8, against 1e-8 in float32). That floor is
not drift. Learning rate:
``lr_peak * (step + 1) / warmup_steps`` during warm-up, then a cosine from
``lr_peak`` to ``lr_final`` at the last step. Lambda: ``anchor_lambda(step,
total)``.

``--max-steps N`` makes the run N steps (the total of both schedules), the
train windows repeated in further epochs when N exceeds them; N = 0 writes
the unchanged decoder. ``--stop-at`` is a deadline in Unix seconds, moved
earlier by ``--stop-reserve`` seconds (on a pod: time left for the export
tasks and their fetch before the pod's cap): before each step and
evaluation, the run stops when the longest step (or the last evaluation) so
far would carry it past that deadline. The weights at the
stop are then written as ``decoder-last.pt`` (when at least one update was
applied; its step and epoch go to ``train.json``), the epoch checkpoints
already written stay, ``decoder-final.pt`` is not written, ``stopped`` is
``deadline``, ``stopped_early`` is true and the exit is 0. A run whose steps all
finished writes ``decoder-final.pt`` first and skips only the final
evaluation (``final_eval_skipped``) when the deadline is that close.

Evaluations, at step 0, after every ``dev_every_steps`` updates and at the
end: the CE of the dev windows (all, positive, anchor and no-speech ones)
and their anchor KL, and the CE of a fixed sample of up to 128 train windows
(``random.Random(seed)``). CE and KL are token means; the KL is
``anchor_kl``'s value (temperature ``tau``, times ``tau**2``).

Outputs (earlier ones are deleted first):

- in ``--checkpoints`` (default ``--out``): ``decoder-epoch<e>.pt`` at the
  end of each epoch e in ``checkpoint_epochs`` below ``epochs``,
  ``decoder-final.pt`` at the end of the run, ``decoder-last.pt`` at a
  ``--stop-at`` stop: the decoder ``state_dict`` in float32, written
  atomically;
- in ``--out``:
  - ``steps.jsonl``: per step its epoch, windows, label and anchor tokens,
    CE, KL, lambda, learning rate, gradient norm before clipping and seconds;
  - ``train.json`` (``ft-pilot-train-1``): the config, window counts, the
    layout, CE, KL, lambda and learning rate per ``dev_every_steps`` steps,
    the evaluations, step seconds p50 and p90 and the milliseconds per
    window at p50, peak GPU memory, the checkpoints, ``state`` (``running``,
    ``done`` or ``stopped``), ``stopped`` (null, ``deadline``,
    ``nonfinite`` or ``error``, with ``error_type``, the exception's type
    name only) and ``stopped_early``; rewritten after each evaluation.

Paid training is never thrown away. A non-finite loss or gradient norm
(that step's update is not applied; ``non_finite`` names the step) or any
other error after a checkpoint was written ends the run with the
checkpoints kept, no ``decoder-last.pt`` (the weights may be broken), the
record written, and exit 0, so the export tasks after it still run (the
final export then falls back to ``decoder-epoch1.pt``). Before any
checkpoint, a non-finite loss is exit 1 and an error propagates (exit 1).

Console: counts and losses only, never a text. Exit 0 (also at a deadline
stop and after a fault with a checkpoint kept); 1 as above; 2 on an invalid
config or windows directory, or without a CUDA device.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperConfig, WhisperFeatureExtractor

from scripts.asr_bench.sets import SAMPLE_RATE_HZ
from scripts.asr_finetune.common import SOURCE_WEIGHTS, fetch_source, package_versions, sha256_file
from scripts.asr_finetune.labels import FIRST_POSITIONS, NOSPEECH, SOT, label_mask, load_tokenizer, loss_weights
from scripts.asr_finetune.lora import load_base

SCHEMA = "ft-pilot-train-1"
FULL_BATCH_MEMORY = 22 * 2**30  # bytes of GPU memory for batch_windows in one micro-batch
TRAIN_SAMPLE = 128
PRINT_EVERY = 25
_RECORDS = ("steps.jsonl", "train.json", "train.json.tmp")
_CHECKPOINTS = ("decoder-*.pt", "decoder-*.pt.tmp")
_FIXED = {"trainable": "decoder", "encoder_dtype": "float16", "decoder_master_dtype": "float32",
          "autocast": "bfloat16", "schedule": "cosine"}
_NUMBERS = ("lr_peak", "lr_final", "grad_clip")
_COUNTS = ("seed", "epochs", "batch_windows", "batch_windows_16gb", "grad_accumulation_16gb", "warmup_steps",
           "dev_every_steps")
_ANCHOR = ("tau", "lambda_start", "lambda_floor", "hold_until", "floor_at")
_ADAMW = ("betas", "eps", "weight_decay")
_KEYS = {"name", "adamw", "anchor", "checkpoint_epochs", *_FIXED, *_NUMBERS, *_COUNTS}
# Optional keys (absent: the first value or the default, the behavior of the configs before them).
_OPTIONAL = {"first_position": FIRST_POSITIONS}
_OPTIONAL_NUMBERS = {"first_position_weight": 1.0}  # > 0; not 1 only with first_position target


class InputError(ValueError):
    """An invalid config or windows directory (exit 2)."""


class NonFinite(RuntimeError):
    """A step's loss or gradient norm is not finite (its update is not applied)."""


# Loss --------------------------------------------------------------------------


def anchor_lambda(step: int, total: int, cfg: Mapping[str, float]) -> float:
    progress = step / total
    if progress <= cfg["hold_until"]:
        return cfg["lambda_start"]
    if progress >= cfg["floor_at"]:
        return cfg["lambda_floor"]
    span = (progress - cfg["hold_until"]) / (cfg["floor_at"] - cfg["hold_until"])
    return cfg["lambda_start"] + span * (cfg["lambda_floor"] - cfg["lambda_start"])


def anchor_kl(student_logits, teacher_logits, labels, anchor_rows, tau):
    """Mean over label positions (labels != -100) of anchor rows of
    KL(softmax(teacher / tau) || softmax(student / tau)) * tau**2."""
    mask = (labels != -100) & anchor_rows[:, None]
    if not mask.any():
        return student_logits.new_zeros(())
    log_p_student = torch.log_softmax(student_logits[mask].float() / tau, dim=-1)
    log_p_teacher = torch.log_softmax(teacher_logits[mask].float() / tau, dim=-1)
    kl = torch.sum(log_p_teacher.exp() * (log_p_teacher - log_p_student), dim=-1)
    return kl.mean() * tau * tau


def learning_rate(step: int, total: int, config: Mapping[str, Any]) -> float:
    warmup = config["warmup_steps"]
    if step < warmup:
        return config["lr_peak"] * (step + 1) / warmup
    progress = min((step - warmup) / max(total - 1 - warmup, 1), 1.0)
    return config["lr_final"] + 0.5 * (config["lr_peak"] - config["lr_final"]) * (1.0 + math.cos(math.pi * progress))


# Config and windows ------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InputError(f"{path}: {error}") from None
    if not isinstance(config, dict):
        raise InputError(f"{path}: not a JSON object")
    problems = [f"missing {key}" for key in sorted(_KEYS - set(config))]
    problems += [f"unknown {key}" for key in sorted(set(config) - _KEYS - set(_OPTIONAL) - set(_OPTIONAL_NUMBERS))]
    if problems:
        raise InputError(f"{path}: {', '.join(problems)}")
    problems += [key for key, values in _OPTIONAL.items() if config.setdefault(key, values[0]) not in values]
    problems += [key for key, default in _OPTIONAL_NUMBERS.items()
                 if not _is_number(config.setdefault(key, default)) or config[key] <= 0]
    if config["first_position_weight"] != 1.0 and config["first_position"] != "target":
        problems.append("first_position_weight without first_position target")
    if not isinstance(config["name"], str) or not config["name"]:
        problems.append("name")
    problems += [key for key, value in _FIXED.items() if config[key] != value]
    problems += [key for key in _NUMBERS if not _is_number(config[key]) or config[key] <= 0]
    problems += [key for key in _COUNTS if not _is_count(config[key]) or config[key] < (0 if key == "seed" else 1)]
    anchor, adamw = config["anchor"], config["adamw"]
    if not isinstance(anchor, dict) or set(anchor) != set(_ANCHOR) or not all(map(_is_number, anchor.values())):
        problems.append("anchor")
    elif not (anchor["tau"] > 0 and 0 <= anchor["hold_until"] < anchor["floor_at"] <= 1):
        problems.append("anchor")
    if (not isinstance(adamw, dict) or set(adamw) != set(_ADAMW) or not isinstance(adamw["betas"], list)
            or len(adamw["betas"]) != 2 or not all(_is_number(beta) and 0 <= beta < 1 for beta in adamw["betas"])
            or not _is_number(adamw["eps"]) or adamw["eps"] <= 0
            or not _is_number(adamw["weight_decay"]) or adamw["weight_decay"] < 0):
        problems.append("adamw")
    epochs = config["epochs"] if _is_count(config["epochs"]) else 0
    checkpoints = config["checkpoint_epochs"]
    if (not isinstance(checkpoints, list) or not all(_is_count(e) and 1 <= e <= epochs for e in checkpoints)
            or epochs not in checkpoints or len(set(checkpoints)) != len(checkpoints)):
        problems.append("checkpoint_epochs")
    if not problems and config["batch_windows_16gb"] * config["grad_accumulation_16gb"] != config["batch_windows"]:
        problems.append("batch_windows_16gb x grad_accumulation_16gb != batch_windows")
    if not problems and config["lr_final"] > config["lr_peak"]:
        problems.append("lr_final above lr_peak")
    if problems:
        raise InputError(f"{path}: invalid {', '.join(problems)}")
    return config


@dataclass(frozen=True)
class Window:
    w: int
    offset: int
    samples: int
    labels: tuple[int, ...]
    anchor: bool
    positive: bool
    split: str
    masked: tuple[int, ...] = ()  # label positions at -100

    @property
    def tokens(self) -> int:
        return len(self.labels) - len(self.masked)

    @property
    def nospeech(self) -> bool:
        return self.labels[0] == NOSPEECH


def load_windows(folder: Path, max_positions: int, transcribe: int,
                 first_position: str = "masked") -> tuple[list[Window], np.ndarray]:
    index, audio_path = folder / "index.jsonl", folder / "audio.i16"
    if not index.is_file() or not audio_path.is_file():
        raise InputError(f"{folder}: index.jsonl or audio.i16 missing")
    audio = np.memmap(audio_path, dtype="<i2", mode="r")
    windows: list[Window] = []
    with index.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                labels = tuple(row["labels"])
                masked = label_mask(labels, transcribe, first_position) if labels else None
                window = Window(int(row["w"]), int(row["offset"]), int(row["samples"]), labels,
                                row["anchor"], row["positive"], row["split"], masked or ())
            except (ValueError, KeyError, TypeError):
                raise InputError(f"{index}: line {number} is not a window row") from None
            if (masked is None or not all(_is_count(token) for token in labels)
                    or len(labels) > max_positions
                    or window.split not in ("train", "dev") or not isinstance(window.anchor, bool)
                    or not isinstance(window.positive, bool) or window.samples <= 0 or window.offset < 0
                    or window.offset + window.samples > len(audio)
                    or (window.nospeech and (window.anchor or window.positive))):
                raise InputError(f"{index}: line {number} (w {window.w}) is invalid")
            windows.append(window)
    if not any(window.split == "train" for window in windows):
        raise InputError(f"{folder}: no train windows")
    return windows, audio


# Model and batches -------------------------------------------------------------


@dataclass
class Models:
    extractor: WhisperFeatureExtractor
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    teacher: torch.nn.Module
    pad_id: int


def load_models(source: Path) -> Models:
    model = load_base(source, torch.float16)
    model.config.use_cache = False
    model.model.encoder.requires_grad_(False)
    model.model.decoder.float()
    model.to("cuda")
    if model.proj_out.weight is not model.model.decoder.embed_tokens.weight:
        raise InputError(f"{source}: the output projection is not tied to the token embedding")
    encoder = model.model.encoder.eval()
    decoder = model.model.decoder.train()
    teacher = copy.deepcopy(decoder).eval().requires_grad_(False)
    return Models(WhisperFeatureExtractor.from_pretrained(source), encoder, decoder, teacher, model.config.pad_token_id)


def decoder_logits(decoder: torch.nn.Module, input_ids: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
    hidden = decoder(input_ids=input_ids, encoder_hidden_states=encoder_out, use_cache=False).last_hidden_state
    return F.linear(hidden, decoder.embed_tokens.weight)


@dataclass
class Batch:
    encoder_out: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    anchor_rows: torch.Tensor
    weights: torch.Tensor  # labels.loss_weights per label position, 0 at the padding


def clip_audio(audio: np.ndarray, window: Window) -> np.ndarray:
    return audio[window.offset:window.offset + window.samples].astype(np.float32) / 32768.0


def log_mel(extractor: WhisperFeatureExtractor, clips: Sequence[np.ndarray]) -> torch.Tensor:
    """Log-mel features of each clip as the shipped decode feeds the encoder.

    faster-whisper computes the log-mel of the clip plus 160 zero samples,
    keeps its first ``content_frames`` = ``len(clip) // 160`` frames and
    zero-fills the rest to 3000 in feature space (``pad_or_trim``). Those
    frames equal the extractor's (the clip zero-padded to 30 s), so the
    frames from ``len(clip) // hop_length`` on are set to 0.0."""
    features = extractor(list(clips), sampling_rate=SAMPLE_RATE_HZ, return_tensors="pt", device="cuda").input_features
    for row, clip in enumerate(clips):
        features[row, :, len(clip) // extractor.hop_length:] = 0.0
    return features


def make_batch(models: Models, audio: np.ndarray, windows: Sequence[Window], first_weight: float = 1.0) -> Batch:
    features = log_mel(models.extractor, [clip_audio(audio, w) for w in windows])
    with torch.no_grad():
        encoder_out = models.encoder(features.to("cuda", torch.float16)).last_hidden_state
    width = max(len(w.labels) for w in windows)
    input_ids = torch.full((len(windows), width), models.pad_id, dtype=torch.long)
    labels = torch.full((len(windows), width), -100, dtype=torch.long)
    weights = torch.zeros((len(windows), width), dtype=torch.float32)
    for row, window in enumerate(windows):
        ids = torch.tensor(window.labels, dtype=torch.long)
        input_ids[row, 0] = SOT
        input_ids[row, 1:len(ids)] = ids[:-1]
        labels[row, :len(ids)] = ids
        labels[row, list(window.masked)] = -100
        weights[row, :len(ids)] = torch.tensor(loss_weights(len(ids), window.masked, first_weight))
    anchor_rows = torch.tensor([w.anchor for w in windows], dtype=torch.bool)
    return Batch(encoder_out.to(torch.bfloat16), input_ids.to("cuda"), labels.to("cuda"), anchor_rows.to("cuda"),
                 weights.to("cuda"))


def forward(models: Models, batch: Batch, *, grad: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Student and (when the batch holds an anchor window) teacher logits."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.set_grad_enabled(grad):
            student = decoder_logits(models.decoder, batch.input_ids, batch.encoder_out)
        teacher = None
        if bool(batch.anchor_rows.any()):
            with torch.no_grad():
                teacher = decoder_logits(models.teacher, batch.input_ids, batch.encoder_out)
    return student, teacher


def chunks(items: Sequence[Window], size: int) -> list[Sequence[Window]]:
    return [items[start:start + size] for start in range(0, len(items), size)]


# Evaluation --------------------------------------------------------------------


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


def evaluate(models: Models, audio: np.ndarray, windows: Sequence[Window], micro: int, tau: float) -> dict[str, Any]:
    """Token-mean CE of all, positive, anchor and no-speech windows and the anchor KL."""
    sums = {"ce": 0.0, "tokens": 0, "positive_ce": 0.0, "positive_tokens": 0,
            "anchor_ce": 0.0, "anchor_tokens": 0, "anchor_kl": 0.0, "nospeech_ce": 0.0, "nospeech_tokens": 0}
    models.decoder.eval()
    for part in chunks(windows, micro):
        batch = make_batch(models, audio, part)
        student, teacher = forward(models, batch, grad=False)
        with torch.no_grad():
            token_ce = F.cross_entropy(student.float().flatten(0, 1), batch.labels.flatten(), ignore_index=-100,
                                       reduction="none").view(batch.labels.shape)
            row_ce = token_ce.sum(dim=1).tolist()
            row_tokens = (batch.labels != -100).sum(dim=1).tolist()
            anchor_tokens = int(((batch.labels != -100) & batch.anchor_rows[:, None]).sum())
            if teacher is not None:
                sums["anchor_kl"] += float(anchor_kl(student, teacher, batch.labels, batch.anchor_rows, tau)) * anchor_tokens
        for window, ce, tokens in zip(part, row_ce, row_tokens):
            sums["ce"] += ce
            sums["tokens"] += tokens
            for flag in ("positive", "anchor", "nospeech"):
                if getattr(window, flag):
                    sums[f"{flag}_ce"] += ce
                    sums[f"{flag}_tokens"] += tokens
    models.decoder.train()
    return {
        "windows": len(windows),
        "ce": _mean(sums["ce"], sums["tokens"]),
        "positive": {"windows": sum(w.positive for w in windows), "ce": _mean(sums["positive_ce"], sums["positive_tokens"])},
        "anchor": {"windows": sum(w.anchor for w in windows), "ce": _mean(sums["anchor_ce"], sums["anchor_tokens"]),
                   "kl": _mean(sums["anchor_kl"], sums["anchor_tokens"])},
        "nospeech": {"windows": sum(w.nospeech for w in windows),
                     "ce": _mean(sums["nospeech_ce"], sums["nospeech_tokens"])},
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# Run ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def save_checkpoint(decoder: torch.nn.Module, path: Path) -> str:
    state = {name: tensor.detach().to("cpu", torch.float32).clone() for name, tensor in decoder.state_dict().items()}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def clear_outputs(out: Path, checkpoints: Path) -> None:
    for folder, patterns in ((out, _RECORDS), (checkpoints, _CHECKPOINTS)):
        folder.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for path in folder.glob(pattern):
                path.unlink()


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    started = time.monotonic()
    torch.manual_seed(config["seed"])
    source = args.source if (args.source / SOURCE_WEIGHTS).is_file() else fetch_source(args.source)
    try:
        transcribe = load_tokenizer(source / "tokenizer.json").token_to_id("<|transcribe|>")
    except Exception as error:  # load_tokenizer: ValueError on wrong ids; tokenizers: Exception on a bad file
        raise InputError(f"{source}: tokenizer.json: {error}") from None
    windows, audio = load_windows(args.windows, WhisperConfig.from_pretrained(source).max_target_positions,
                                  transcribe, config["first_position"])
    train = [w for w in windows if w.split == "train"]
    dev = [w for w in windows if w.split == "dev"]
    sample = sorted(random.Random(config["seed"]).sample(train, min(TRAIN_SAMPLE, len(train))), key=lambda w: w.w)
    clear_outputs(args.out, args.checkpoints)
    models = load_models(source)

    device = torch.cuda.get_device_properties(0)
    per_step = config["batch_windows"]
    if device.total_memory >= FULL_BATCH_MEMORY:
        micro, accumulation = per_step, 1
    else:
        micro, accumulation = config["batch_windows_16gb"], config["grad_accumulation_16gb"]
    steps_per_epoch = math.ceil(len(train) / per_step)
    total = config["epochs"] * steps_per_epoch if args.max_steps is None else args.max_steps
    tau = config["anchor"]["tau"]
    first_weight = config["first_position_weight"]

    parameters = list(models.decoder.parameters())
    adamw = config["adamw"]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr_peak"], betas=tuple(adamw["betas"]), eps=adamw["eps"],
                                  weight_decay=adamw["weight_decay"], fused=True)
    torch.cuda.reset_peak_memory_stats()
    print(f"windows: train {len(train):,} (positive {sum(w.positive for w in train):,}, anchor "
          f"{sum(w.anchor for w in train):,}, no-speech {sum(w.nospeech for w in train):,}), dev {len(dev):,}; "
          f"{micro} x {accumulation} windows per step, {steps_per_epoch:,} steps per epoch, {total:,} steps", flush=True)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "config": config,
        "source": {"weights_sha256": sha256_file(source / SOURCE_WEIGHTS)},
        "windows": {
            "train": len(train), "train_positive": sum(w.positive for w in train),
            "train_anchor": sum(w.anchor for w in train), "train_tokens": sum(w.tokens for w in train),
            "dev": len(dev), "dev_positive": sum(w.positive for w in dev), "dev_anchor": sum(w.anchor for w in dev),
            "train_nospeech": sum(w.nospeech for w in train), "dev_nospeech": sum(w.nospeech for w in dev),
            "first_position": config["first_position"], "first_position_weight": first_weight,
            "train_sample": len(sample),
        },
        "layout": {
            "gpu": device.name, "gpu_memory_gib": round(device.total_memory / 2**30, 2),
            "micro_batch": micro, "accumulation": accumulation, "windows_per_step": per_step,
            "steps_per_epoch": steps_per_epoch, "total_steps": total, "max_steps": args.max_steps,
            "stop_at": args.stop_at, "stop_reserve_s": args.stop_reserve,
        },
        "parameters": {
            "decoder_trainable": sum(p.numel() for p in parameters if p.requires_grad),
            "encoder_trainable": sum(p.numel() for p in models.encoder.parameters() if p.requires_grad),
        },
        "log": [], "evals": [], "checkpoints": {}, "state": "running", "stopped": None, "stopped_early": False,
        "packages": package_versions(),
    }
    seconds: list[float] = []
    trained_windows = 0
    interval = {"ce": 0.0, "tokens": 0, "kl": 0.0, "anchor_tokens": 0, "first": 0}

    def finish(state: str) -> None:
        record["state"] = state
        record["step_seconds"] = {"steps": len(seconds), "p50": _percentile(seconds, 50),
                                  "p90": _percentile(seconds, 90), "sum": sum(seconds)}
        p50 = record["step_seconds"]["p50"]
        record["ms_per_window_p50"] = None if p50 is None else p50 * 1000.0 / per_step
        record["ms_per_window_mean"] = sum(seconds) * 1000.0 / trained_windows if trained_windows else None
        try:
            record["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            record["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        except Exception:  # a broken CUDA context after a crash must not cost the record
            record["peak_allocated_gib"] = record["peak_reserved_gib"] = None
        record["wall_s"] = round(time.monotonic() - started, 1)
        write_json(args.out / "train.json", record)

    def past_deadline(expected: float) -> bool:
        return args.stop_at is not None and time.time() + expected >= args.stop_at - args.stop_reserve

    def run_eval(step: int) -> None:
        begin = time.monotonic()
        entry: dict[str, Any] = {"step": step}
        entry["dev"] = evaluate(models, audio, dev, micro, tau) if dev else None
        entry["train_sample"] = evaluate(models, audio, sample, micro, tau)
        entry["seconds"] = round(time.monotonic() - begin, 2)
        record["evals"].append(entry)
        dev_part = entry["dev"]
        dev_text = ("no dev windows" if dev_part is None else
                    f"dev ce {_fmt(dev_part['ce'])} (positive {_fmt(dev_part['positive']['ce'])}, anchor "
                    f"{_fmt(dev_part['anchor']['ce'])}, anchor kl {_fmt(dev_part['anchor']['kl'], 6)})")
        print(f"eval at step {step}: {dev_text}; train sample ce {_fmt(entry['train_sample']['ce'])} "
              f"({entry['seconds']:.1f} s)", flush=True)
        finish("running")

    def stop_at_deadline(where: str) -> int:
        done = len(seconds)
        if done:
            name = "decoder-last.pt"
            record["checkpoints"][name] = {"step": done, "epoch": math.ceil(done / steps_per_epoch),
                                           "epochs_done": round(done / steps_per_epoch, 4),
                                           "sha256": save_checkpoint(models.decoder, args.checkpoints / name)}
        record["stopped"], record["stopped_early"] = "deadline", True
        finish("stopped")
        print(f"stopped at the --stop-at deadline {where} after {done:,} of {total:,} steps; "
              f"checkpoints: {', '.join(record['checkpoints']) or 'none'}", flush=True)
        return 0

    if past_deadline(0.0):
        return stop_at_deadline("before step 0")
    try:
        run_eval(0)
        orders: dict[int, list[Window]] = {}
        with (args.out / "steps.jsonl").open("w", encoding="utf-8") as steps_log:
            for step in range(total):
                if past_deadline(max(seconds, default=0.0)):
                    return stop_at_deadline(f"before step {step}")
                epoch, position = divmod(step, steps_per_epoch)
                if epoch not in orders:
                    orders.clear()
                    orders[epoch] = random.Random(config["seed"] + epoch).sample(train, len(train))
                windows_now = orders[epoch][position * per_step:(position + 1) * per_step]
                lr = learning_rate(step, total, config)
                lam = anchor_lambda(step, total, config["anchor"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                parts = chunks(windows_now, micro)
                step_tokens = sum(w.tokens for w in windows_now)
                step_anchor_tokens = sum(w.tokens for w in windows_now if w.anchor)
                step_weight = sum(sum(loss_weights(len(w.labels), w.masked, first_weight)) for w in windows_now)
                begin = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                ce_sum = torch.zeros((), device="cuda")
                kl_sum = torch.zeros((), device="cuda")
                for part in parts:
                    batch = make_batch(models, audio, part, first_weight)
                    student, teacher = forward(models, batch, grad=True)
                    tokens = sum(w.tokens for w in part)
                    anchor_tokens = sum(w.tokens for w in part if w.anchor)
                    if first_weight == 1.0:  # the token mean of the configs before round 5, unchanged
                        ce = F.cross_entropy(student.float().flatten(0, 1), batch.labels.flatten(), ignore_index=-100)
                        weighted = ce * (tokens / step_tokens)
                    else:  # the weighted mean over the step's label positions
                        token_ce = F.cross_entropy(student.float().flatten(0, 1), batch.labels.flatten(),
                                                   ignore_index=-100, reduction="none").view(batch.labels.shape)
                        ce = token_ce.sum() / tokens
                        weighted = (token_ce * batch.weights).sum() / step_weight
                    kl = (anchor_kl(student, teacher, batch.labels, batch.anchor_rows, tau) if teacher is not None
                          else student.new_zeros((), dtype=torch.float32))
                    loss = weighted
                    if step_anchor_tokens:
                        loss = loss + lam * kl * (anchor_tokens / step_anchor_tokens)
                    loss.backward()
                    ce_sum += ce.detach() * tokens
                    kl_sum += kl.detach() * anchor_tokens
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"]))
                ce_value = float(ce_sum) / step_tokens
                kl_value = float(kl_sum) / step_anchor_tokens if step_anchor_tokens else 0.0
                if not all(map(math.isfinite, (ce_value, kl_value, grad_norm))):
                    record["non_finite"] = {"step": step, "ce": repr(ce_value), "kl": repr(kl_value),
                                            "grad_norm": repr(grad_norm)}
                    raise NonFinite(f"step {step}")
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                seconds.append(elapsed)
                trained_windows += len(windows_now)
                steps_log.write(json.dumps({
                    "step": step, "epoch": epoch + 1, "windows": len(windows_now), "tokens": step_tokens,
                    "anchor_tokens": step_anchor_tokens, "ce": ce_value, "kl": kl_value, "lambda": lam, "lr": lr,
                    "grad_norm": grad_norm, "seconds": round(elapsed, 4),
                }) + "\n")
                steps_log.flush()
                interval["ce"] += ce_value * step_tokens
                interval["tokens"] += step_tokens
                interval["kl"] += kl_value * step_anchor_tokens
                interval["anchor_tokens"] += step_anchor_tokens
                done = step + 1
                if step == 0 or done % config["dev_every_steps"] == 0 or done == total:
                    record["log"].append({
                        "steps": [interval["first"], step], "ce": interval["ce"] / interval["tokens"],
                        "kl": _mean(interval["kl"], interval["anchor_tokens"]), "lambda": lam, "lr": lr,
                    })
                    interval = {"ce": 0.0, "tokens": 0, "kl": 0.0, "anchor_tokens": 0, "first": done}
                if step % PRINT_EVERY == 0 or done == total:
                    print(f"step {step}/{total} (epoch {epoch + 1}): ce {ce_value:.4f} kl {kl_value:.6f} "
                          f"lambda {lam:.3f} lr {lr:.3g} grad norm {grad_norm:.3f} {elapsed:.2f} s", flush=True)
                if done % steps_per_epoch == 0 and done < total:
                    finished = done // steps_per_epoch
                    if finished in config["checkpoint_epochs"] and finished < config["epochs"]:
                        name = f"decoder-epoch{finished}.pt"
                        record["checkpoints"][name] = {
                            "step": done, "epoch": finished,
                            "sha256": save_checkpoint(models.decoder, args.checkpoints / name)}
                        print(f"wrote {name} after {done:,} steps", flush=True)
                if done % config["dev_every_steps"] == 0 and done < total:
                    if past_deadline(record["evals"][-1]["seconds"]):
                        return stop_at_deadline(f"before the evaluation at step {done}")
                    run_eval(done)
        name = "decoder-final.pt"
        record["checkpoints"][name] = {
            "step": total, "epoch": None if total == 0 else math.ceil(total / steps_per_epoch),
            "sha256": save_checkpoint(models.decoder, args.checkpoints / name)}
        if record["evals"][-1]["step"] != total:
            if past_deadline(record["evals"][-1]["seconds"]):
                record["final_eval_skipped"] = True
                print("final evaluation skipped at the --stop-at deadline", flush=True)
            else:
                run_eval(total)
    except Exception as error:
        nonfinite = isinstance(error, NonFinite)
        record["stopped"] = "nonfinite" if nonfinite else "error"
        record["stopped_early"] = "decoder-final.pt" not in record["checkpoints"]
        if not nonfinite:
            record["error_type"] = type(error).__name__
        finish("stopped")
        what = "a non-finite loss or gradient norm (update not applied)" if nonfinite else type(error).__name__
        if record["checkpoints"]:
            print(f"stopped on {what} after {len(seconds):,} of {total:,} steps; checkpoints kept: "
                  f"{', '.join(record['checkpoints'])}; exit 0 so the exports run", flush=True)
            return 0
        print(f"stopped on {what} after {len(seconds):,} of {total:,} steps, before any checkpoint", flush=True)
        if nonfinite:
            return 1
        raise
    finish("done")
    p50 = record["step_seconds"]["p50"]
    print(f"wrote {name} after {total:,} steps; step seconds p50 {_fmt(p50, 3)}, p90 "
          f"{_fmt(record['step_seconds']['p90'], 3)}; ms per window at p50 {_fmt(record['ms_per_window_p50'], 1)}; "
          f"peak GPU memory {record['peak_allocated_gib']} GiB", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="transformers snapshot of the 760M")
    parser.add_argument("--windows", type=Path, required=True, help="scripts.asr_finetune.windows output")
    parser.add_argument("--out", type=Path, required=True, help="train.json, steps.jsonl")
    parser.add_argument("--checkpoints", type=Path, help="decoder-*.pt (default: --out)")
    parser.add_argument("--max-steps", type=int, help="run exactly N steps (probes)")
    parser.add_argument("--stop-at", type=float, help="deadline in Unix seconds")
    parser.add_argument("--stop-reserve", type=float, default=0.0,
                        help="seconds before --stop-at by which the run stops (time for the tasks after it)")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps must be at least 0")
    if args.stop_reserve < 0:
        parser.error("--stop-reserve must be at least 0")
    args.checkpoints = args.checkpoints or args.out
    try:
        config = load_config(args.config)
        if not torch.cuda.is_available():
            raise InputError("no CUDA device")
        return run(args, config)
    except InputError as error:
        print(f"error: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
