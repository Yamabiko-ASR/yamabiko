"""Step 4 of the fine-tuning pilot: one real LoRA training step on five
Common Voice clips, then the merge, conversion and decode of step 3.

Usage::

    python -m scripts.asr_finetune.lora_train_step --bench-python <release venv python>
        [--work-dir <dir>] [--set <set dir> ...] [--runs-dir <dir>] [--lr 1e-4]

Needs step 2 (``reproduce_export``). Clips: the Common Voice 17.0 ja train
split (CC0) at the snapshot ``build_synthetic`` pins, filtered as that
builder filters its test clips (votes, sentence length) and kept clear of
the held-out bench sets: no sentence a ``synthetic-v1`` or ``probe-cv8``
line speaks, no speaker of a ``synthetic-v1`` scene, no clip of the
committed probe fixture. The first five such clips in path order, one per
speaker, are decoded with FFmpeg into ``data\\cv-train5``.

Training: the source weights in float16 on CUDA, the pilot adapter
(``lora.py``) in float32, float16 autocast with a gradient scaler, gradient
checkpointing and AdamW. The loss over the five clips is measured before and
after one optimizer step (eval mode, no dropout); a step the scaler skips
for non-finite gradients is retried at the lowered scale, as training
would. Every adapter B gradient is recorded. The adapter is saved to
``adapters\\lora-step``, merged into ``hf\\lora-step`` and converted into
``ct2\\lora-step``. Compared: the merged weights with the source (only the
adapter's targets may change), the conversion with ``ct2\\repro`` (only the
variables those targets land in), and the bench decode (label
``lora-step``) with the ``repro`` runs, window by window and through
``scripts.asr_bench.score`` (``reports\\lora_train_step-<set_id>`` in the
work directory). The record goes to ``results\\lora_train_step.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, get_peft_model
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from scripts.asr_bench.build_synthetic import (
    CV_MIN_UP_VOTES,
    CV_REPO,
    CV_REVISION,
    CV_SENTENCE_CHARS,
    CommonVoiceClips,
)
from scripts.asr_bench.sets import SAMPLE_RATE_HZ, bench_root
from scripts.asr_finetune.common import (
    CT2_SEED,
    REPO_ROOT,
    REPRO_LABEL,
    SOURCE_WEIGHTS,
    add_common_args,
    bench_decode,
    bench_score,
    compare_model_dirs,
    compare_run_sets,
    convert_to_ct2,
    ct2_dir,
    existing_runs,
    hf_dir,
    package_versions,
    require_source,
    resolve_bench_args,
    text_summary,
    weight_summary,
    write_result,
)
from scripts.asr_finetune.lora import (
    ct2_target_summary,
    compare_safetensors,
    describe_adapter,
    load_base,
    lora_config,
    lora_group,
    lora_layers,
    merge_adapter,
)

LABEL = "lora-step"
CLIP_COUNT = 5
CV_TRAIN_TSV = "transcript/ja/train.tsv"
CV_TRAIN_TAR = "audio/ja/train/ja_train_0.tar"
CV_ATTRIBUTION = "Mozilla Common Voice 17.0, Japanese train split (via fsicoli/common_voice_17_0)"
HELD_OUT_SETS = ("synthetic-v1", "probe-cv8")
PROBE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "audio" / "cv17_ja_8lines.json"
MAX_ATTEMPTS = 8


# Clips -------------------------------------------------------------------------


def _cv_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=CV_REPO, filename=filename, repo_type="dataset",
        revision=CV_REVISION, cache_dir=str(bench_root() / "cache" / "hf"),
    ))


def held_out() -> dict[str, set[str]]:
    """Sentences, speakers (12-character client id prefixes) and clip names
    the held-out bench sets use."""
    sentences: set[str] = set()
    speakers: set[str] = set()
    for set_id in HELD_OUT_SETS:
        manifest = json.loads((bench_root() / "sets" / set_id / "set.json").read_text(encoding="utf-8"))
        for item in manifest["items"]:
            sentences.update(line["text"] for line in item["lines"])
            speakers.update(
                voice.removeprefix("cv17:") for voice in item["condition"].get("voices", [])
                if voice.startswith("cv17:")
            )
    fixture = json.loads(PROBE_FIXTURE.read_text(encoding="utf-8"))
    sentences.update(line["sentence"] for line in fixture["lines"])
    return {"sentences": sentences, "speakers": speakers, "clips": {line["clip"] for line in fixture["lines"]}}


def prepare_clips(work_dir: Path, ffmpeg: str) -> list[dict[str, Any]]:
    """Pick, decode and store the training clips; returns them with audio."""
    excluded = held_out()
    with _cv_download(CV_TRAIN_TSV).open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), key=lambda row: row["path"])
    low, high = CV_SENTENCE_CHARS
    eligible = [
        row for row in rows
        if int(row["up_votes"]) >= CV_MIN_UP_VOTES and int(row["down_votes"]) == 0
        and low <= len(row["sentence"]) <= high
    ]
    archive = CommonVoiceClips(_cv_download(CV_TRAIN_TAR), ffmpeg)
    out = work_dir / "data" / "cv-train5"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    clips: list[dict[str, Any]] = []
    speakers: set[str] = set()
    skipped = {"held_out_sentence": 0, "held_out_speaker": 0, "held_out_clip": 0, "speaker_taken": 0, "undecodable": 0}
    try:
        for row in eligible:
            speaker = row["client_id"][:12]
            reason = (
                "held_out_sentence" if row["sentence"] in excluded["sentences"]
                else "held_out_speaker" if speaker in excluded["speakers"]
                else "held_out_clip" if row["path"] in excluded["clips"]
                else "speaker_taken" if row["client_id"] in speakers
                else None
            )
            if reason is None:
                audio = archive.decode(row["path"])
                reason = "undecodable" if audio is None or audio.size == 0 else None
            if reason is not None:
                skipped[reason] += 1
                continue
            audio.astype("<f4").tofile(out / f"{Path(row['path']).stem}.f32")
            clips.append({
                "clip": row["path"], "sentence": row["sentence"], "speaker": speaker,
                "seconds": round(audio.size / SAMPLE_RATE_HZ, 3),
                "audio_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(), "audio": audio,
            })
            speakers.add(row["client_id"])
            if len(clips) == CLIP_COUNT:
                break
    finally:
        archive.close()
    if len(clips) < CLIP_COUNT:
        raise SystemExit(f"only {len(clips)} eligible train clips")
    manifest = {
        "source": f"https://huggingface.co/datasets/{CV_REPO}", "revision": CV_REVISION,
        "files": [CV_TRAIN_TSV, CV_TRAIN_TAR], "license": "CC0-1.0", "attribution": CV_ATTRIBUTION,
        "filter": f"up_votes >= {CV_MIN_UP_VOTES}, down_votes == 0, {low}-{high} sentence characters",
        "held_out": {"sets": list(HELD_OUT_SETS), "fixture": PROBE_FIXTURE.name,
                     **{key: len(value) for key, value in excluded.items()}},
        "eligible_rows": len(eligible), "skipped_before_fifth": skipped,
        "clips": [{key: value for key, value in clip.items() if key != "audio"} for clip in clips],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return clips


# Training step -----------------------------------------------------------------


def build_labels(tokenizer: WhisperTokenizerFast, sentences: list[str]) -> torch.Tensor:
    """Decoder targets ``<|ja|><|transcribe|><|notimestamps|> text <|endoftext|>``;
    the model shifts them right behind ``<|startoftranscript|>``."""
    prefix = tokenizer.convert_tokens_to_ids(["<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    end = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    rows = [prefix + tokenizer(sentence, add_special_tokens=False).input_ids + [end] for sentence in sentences]
    width = max(len(row) for row in rows)
    return torch.tensor([row + [-100] * (width - len(row)) for row in rows])


def gradient_summary(model: PeftModel) -> dict[str, Any]:
    by_group: dict[str, list[float]] = {}
    nonzero = 0
    max_a = 0.0
    layers = lora_layers(model)
    for name, layer in layers.items():
        grad_b = layer.lora_B["default"].weight.grad
        grad_a = layer.lora_A["default"].weight.grad
        norm = float(grad_b.norm()) if grad_b is not None else 0.0
        nonzero += norm > 0.0
        if grad_a is not None:
            max_a = max(max_a, float(grad_a.abs().max()))
        by_group.setdefault(lora_group(name), []).append(norm)
    return {
        "modules": len(layers),
        "lora_B_with_nonzero_grad": nonzero,
        "lora_A_max_abs_grad": max_a,
        "lora_B_grad_norm_min_by_group": {group: min(norms) for group, norms in sorted(by_group.items())},
    }


def eval_loss(model: PeftModel, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean token loss without dropout."""
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return float(model(input_features=features, labels=labels).loss)


def train_one_step(source: Path, clips: list[dict[str, Any]], lr: float, adapter_dir: Path) -> dict[str, Any]:
    torch.manual_seed(0)
    extractor = WhisperFeatureExtractor.from_pretrained(source)
    tokenizer = WhisperTokenizerFast.from_pretrained(source)
    features = extractor(
        [clip["audio"] for clip in clips], sampling_rate=SAMPLE_RATE_HZ, return_tensors="pt",
    ).input_features.to("cuda", torch.float16)
    labels = build_labels(tokenizer, [clip["sentence"] for clip in clips]).to("cuda")

    base = load_base(source, torch.float16)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, lora_config()).to("cuda")
    torch.cuda.reset_peak_memory_stats()

    loss_before = eval_loss(model, features, labels)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(MAX_ATTEMPTS):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=features, labels=labels).loss
        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = gradient_summary(model)
        scaler.step(optimizer)
        scaler.update()
        applied = scaler.get_scale() >= scale
        attempts.append({"scale": scale, "train_loss": loss.item(), "applied": applied})
        if applied:
            break
    else:
        raise SystemExit(f"the gradient scaler skipped all {MAX_ATTEMPTS} attempts")
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started
    loss_after = eval_loss(model, features, labels)
    adapter = describe_adapter(model)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir)
    record = {
        "lr": lr,
        "optimizer": "AdamW, weight decay 0",
        "precision": "float16 weights and autocast, float32 adapter, GradScaler",
        "gradient_checkpointing": True,
        "batch": {"clips": len(clips), "seconds": round(sum(clip["seconds"] for clip in clips), 3),
                  "label_tokens": int((labels != -100).sum())},
        "loss_before": loss_before,
        "loss_after": loss_after,
        "attempts": attempts,
        "gradients": gradients,
        "adapter_after_step": adapter,
        "step_seconds": step_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate of the one step (default 1e-4)")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"), help="FFmpeg executable (default: from PATH)")
    args = parser.parse_args(argv)
    sets = resolve_bench_args(parser, args)
    if not args.ffmpeg:
        parser.error("ffmpeg must be on PATH or given with --ffmpeg")
    work = args.work_dir
    source = require_source(work)
    repro = ct2_dir(work, REPRO_LABEL)
    repro_runs = existing_runs(args.runs_dir, sets, REPRO_LABEL)

    clips = prepare_clips(work, args.ffmpeg)
    adapter_dir = work / "adapters" / LABEL
    training = train_one_step(source, clips, args.lr, adapter_dir)
    torch.cuda.empty_cache()

    merged = hf_dir(work, LABEL)
    merge_adapter(source, adapter_dir, merged)
    weights = compare_safetensors(source / SOURCE_WEIGHTS, merged / SOURCE_WEIGHTS)
    out = ct2_dir(work, LABEL)
    convert_to_ct2(merged, out)
    files = compare_model_dirs(repro, out)
    targets = ct2_target_summary(files["model_bin"], repro / "model.bin")

    runs = bench_decode(
        label=LABEL, model=str(out), sets=sets, runs_dir=args.runs_dir,
        bench_python=args.bench_python, work_dir=work,
    )
    texts = compare_run_sets(repro_runs, runs)
    scores = {
        set_id: bench_score(
            [repro_runs[set_id], runs[set_id]], out=work / "reports" / f"lora_train_step-{set_id}",
            bench_python=args.bench_python,
        )
        for set_id in runs
    }

    record = {
        "clips": [{key: value for key, value in clip.items() if key != "audio"} for clip in clips],
        "training": training,
        "merged_vs_source_weights": weights,
        "conversion_vs_repro": files,
        "conversion_targets": targets,
        "ct2_seed": CT2_SEED,
        "texts_repro_vs_lora_step": texts,
        "scores": scores,
        "packages": package_versions(),
    }
    path = write_result(work, "lora_train_step", record)

    print(f"clips: {len(clips)}, {training['batch']['seconds']:.1f} s, {training['batch']['label_tokens']} label tokens")
    print(f"loss {training['loss_before']:.4f} -> {training['loss_after']:.4f} after one step "
          f"(lr {args.lr:g}, {len(training['attempts'])} attempt(s), {training['step_seconds']:.2f} s)")
    gradients = training["gradients"]
    print(f"adapter B gradients nonzero: {gradients['lora_B_with_nonzero_grad']}/{gradients['modules']}; "
          f"max |B| after the step {training['adapter_after_step']['max_abs_lora_B']:.3g}")
    print(f"merged vs source safetensors: {weight_summary(weights)}; "
          f"targets changed {weights['differing_targets']}/{weights['lora_target_tensors']}, "
          f"outside targets {len(weights['differing_outside_targets'])}")
    print(f"model.bin vs repro: {weight_summary(files['model_bin'])}; "
          f"targets changed {targets['differing_targets']}/{targets['ct2_target_variables']}, "
          f"outside targets {len(targets['differing_outside_targets'])}")
    print(f"window texts, repro vs {LABEL}: {text_summary(texts)}")
    for set_id, figures in scores.items():
        for label, values in figures.items():
            print(f"{set_id} {label}: " + ", ".join(
                f"{key} {value:.4f}" if isinstance(value, float) else f"{key} {value}" for key, value in values.items()
            ))
    print(f"record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
