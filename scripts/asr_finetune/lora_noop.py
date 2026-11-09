"""Step 3 of the fine-tuning pilot: a LoRA adapter that changes nothing goes
through merge, conversion and decode and must come out as the shipped model.

Usage::

    python -m scripts.asr_finetune.lora_noop --bench-python <release venv python>
        [--work-dir <dir>] [--set <set dir> ...] [--runs-dir <dir>]

Needs step 2 (``reproduce_export``): its source snapshot, ``ct2\\repro``
and its ``repro`` runs. Attaches the pilot adapter (``lora.py``) to the
float32 source weights on the CPU and checks that it wraps the expected
modules and that every B is zero, saves it to ``adapters\\lora-noop``,
merges it into ``hf\\lora-noop`` and converts that into ``ct2\\lora-noop``
with the shipped settings. Compared: the merged weights against the source
tensor by tensor, the conversion against ``ct2\\repro`` file by file and
variable by variable, and its bench decode (label ``lora-noop``) against
the ``repro`` runs window by window. The record goes to
``results\\lora_noop.json``.
"""

from __future__ import annotations

import argparse
import shutil

import torch
from peft import get_peft_model

from scripts.asr_finetune.common import (
    CT2_SEED,
    REPRO_LABEL,
    SOURCE_WEIGHTS,
    add_common_args,
    bench_decode,
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
    compare_safetensors,
    describe_adapter,
    expected_lora_modules,
    load_base,
    lora_config,
    merge_adapter,
)

LABEL = "lora-noop"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser)
    args = parser.parse_args(argv)
    sets = resolve_bench_args(parser, args)
    work = args.work_dir
    source = require_source(work)
    repro = ct2_dir(work, REPRO_LABEL)
    repro_runs = existing_runs(args.runs_dir, sets, REPRO_LABEL)

    torch.manual_seed(0)
    base = load_base(source, torch.float32)
    expected = expected_lora_modules(base)
    model = get_peft_model(base, lora_config())
    adapter = describe_adapter(model)
    if adapter["modules"] != expected or adapter["max_abs_lora_B"] != 0.0:
        raise SystemExit(f"adapter is not the zero-B pilot layout ({expected} modules expected): {adapter}")
    adapter_dir = work / "adapters" / LABEL
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir)
    del model, base

    merged = hf_dir(work, LABEL)
    merge_adapter(source, adapter_dir, merged)
    weights = compare_safetensors(source / SOURCE_WEIGHTS, merged / SOURCE_WEIGHTS)
    out = ct2_dir(work, LABEL)
    convert_to_ct2(merged, out)
    files = compare_model_dirs(repro, out)

    runs = bench_decode(
        label=LABEL, model=str(out), sets=sets, runs_dir=args.runs_dir,
        bench_python=args.bench_python, work_dir=work,
    )
    texts = compare_run_sets(repro_runs, runs)

    record = {
        "adapter": {**adapter, "expected_modules": expected},
        "merged_vs_source_weights": weights,
        "conversion_vs_repro": files,
        "ct2_seed": CT2_SEED,
        "texts_repro_vs_lora_noop": texts,
        "packages": package_versions(),
    }
    path = write_result(work, "lora_noop", record)

    print(f"adapter: {adapter['modules']} modules, {adapter['trainable_parameters']:,} trainable parameters, "
          f"max |B| {adapter['max_abs_lora_B']}")
    print(f"merged vs source safetensors: {weight_summary(weights)}")
    for name, entry in files["files"].items():
        print(f"{name}: {'identical to repro' if entry['identical'] else 'differs from repro'} ({entry['sha256']})")
    if "model_bin" in files:
        print(f"model.bin vs repro: {weight_summary(files['model_bin'])}")
    print(f"window texts, repro vs {LABEL}: {text_summary(texts)}")
    print(f"record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
