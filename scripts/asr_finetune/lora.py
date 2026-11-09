"""LoRA layout of the fine-tuning pilot, the merge back into float16
transformers weights, and the weight checks that follow a merge.

The adapter: rank 16, alpha 32, on the attention (q, k, v, out) and MLP
(fc1, fc2) projections of encoder layers 24 to 31 and of every decoder
layer, self and cross attention, under the standard transformers Whisper
module names. PEFT initializes B to zero, so a fresh adapter is a no-op.

A merge reloads the source weights in float32 on the CPU, applies the saved
adapter, folds it in (``merge_and_unload``), rounds once to float16 and saves
the result beside copies of the source's config, generation config and
tokenizer files, so the converter sees the same metadata as the shipped
export and only the weights differ.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer
from safetensors import safe_open
from transformers import WhisperForConditionalGeneration

from scripts.asr_finetune.common import SOURCE_WEIGHTS, compare_values, read_ct2_model, value_identical

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
ENCODER_LAYERS = tuple(range(24, 32))
_ATTENTION = r"(q_proj|k_proj|v_proj|out_proj)"
LORA_TARGETS = (
    rf"model\.encoder\.layers\.({'|'.join(map(str, ENCODER_LAYERS))})\.(self_attn\.{_ATTENTION}|fc1|fc2)"
    rf"|model\.decoder\.layers\.\d+\.((self_attn|encoder_attn)\.{_ATTENTION}|fc1|fc2)"
)
# The CTranslate2 variables those projections land in (q, k and v of self
# attention fused in linear_0; cross attention keeps q in linear_0 and fuses
# k and v in linear_1).
CT2_TARGETS = re.compile(
    rf"encoder/layer_({'|'.join(map(str, ENCODER_LAYERS))})/(self_attention/linear_[01]|ffn/linear_[01])/weight"
    r"|decoder/layer_\d+/(self_attention/linear_[01]|attention/linear_[012]|ffn/linear_[01])/weight"
)
_HF_TARGETS = re.compile(rf"({LORA_TARGETS})\.weight")
_PEFT_PREFIX = "base_model.model."


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=LORA_TARGETS, init_lora_weights=True,
    )


def load_base(source: Path, dtype: torch.dtype) -> WhisperForConditionalGeneration:
    return WhisperForConditionalGeneration.from_pretrained(source, dtype=dtype, attn_implementation="sdpa")


def lora_layers(model: PeftModel) -> dict[str, LoraLayer]:
    return {
        name.removeprefix(_PEFT_PREFIX): module
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }


def lora_group(name: str) -> str:
    """``model.decoder.layers.3.encoder_attn.q_proj`` -> ``decoder encoder_attn.q_proj``."""
    stack, _, rest = name.removeprefix("model.").partition(".layers.")
    return f"{stack} {rest.partition('.')[2]}"


def describe_adapter(model: PeftModel) -> dict[str, Any]:
    """Wrapped modules by group, trainable parameters and the largest |B|."""
    layers = lora_layers(model)
    groups: dict[str, int] = {}
    for name in layers:
        groups[lora_group(name)] = groups.get(lora_group(name), 0) + 1
    max_b = max(float(layer.lora_B["default"].weight.detach().abs().max()) for layer in layers.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    encoder_layers = sorted({int(name.split(".")[3]) for name in layers if name.startswith("model.encoder.")})
    return {
        "modules": len(layers),
        "groups": dict(sorted(groups.items())),
        "encoder_layers": encoder_layers,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "max_abs_lora_B": max_b,
    }


def expected_lora_modules(model: WhisperForConditionalGeneration) -> int:
    """Six projections per adapted encoder layer, ten per decoder layer."""
    return len(ENCODER_LAYERS) * 6 + model.config.decoder_layers * 10


def merge_adapter(source: Path, adapter_dir: Path, out: Path) -> None:
    """Fold a saved adapter into float32 source weights, round once to
    float16 and save a transformers checkpoint the converter takes."""
    base = load_base(source, torch.float32)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged = merged.to(torch.float16)
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(out, safe_serialization=True)
    for path in source.iterdir():
        if path.is_file() and path.name != SOURCE_WEIGHTS and not path.name.startswith("."):
            shutil.copy2(path, out / path.name)


def compare_safetensors(reference: Path, other: Path) -> dict[str, Any]:
    """Tensor-by-tensor comparison of two safetensors files, with the
    differing tensors split into LoRA targets and everything else."""
    differing: dict[str, Any] = {}
    identical = 0
    with safe_open(reference, framework="np") as a, safe_open(other, framework="np") as b:
        names_a, names_b = set(a.keys()), set(b.keys())
        for name in sorted(names_a & names_b):
            ta, tb = a.get_tensor(name), b.get_tensor(name)
            if ta.dtype != tb.dtype or ta.shape != tb.shape:
                differing[name] = {"reference": [list(ta.shape), str(ta.dtype)], "other": [list(tb.shape), str(tb.dtype)]}
            elif np.array_equal(ta.view(f"u{ta.itemsize}"), tb.view(f"u{tb.itemsize}")):
                identical += 1
            else:
                differing[name] = compare_values(ta, tb)
    return {
        "tensors": len(names_a),
        "only_in_reference": sorted(names_a - names_b),
        "only_in_other": sorted(names_b - names_a),
        "identical_tensors": identical,
        "value_identical": value_identical(differing),
        "differing_tensors": differing,
        "lora_target_tensors": sum(1 for name in names_a if _HF_TARGETS.fullmatch(name)),
        "differing_targets": sum(1 for name in differing if _HF_TARGETS.fullmatch(name)),
        "differing_outside_targets": sorted(name for name in differing if not _HF_TARGETS.fullmatch(name)),
    }


def ct2_target_summary(model_bin_comparison: dict[str, Any], reference_bin: Path) -> dict[str, Any]:
    """Which ``model.bin`` variables a merge changed, against the variables
    the adapter's projections land in."""
    targets = sorted(name for name in read_ct2_model(reference_bin).variables if CT2_TARGETS.fullmatch(name))
    differing = model_bin_comparison["differing_variables"]
    return {
        "ct2_target_variables": len(targets),
        "differing_targets": sum(1 for name in differing if CT2_TARGETS.fullmatch(name)),
        "differing_outside_targets": sorted(name for name in differing if not CT2_TARGETS.fullmatch(name)),
    }
