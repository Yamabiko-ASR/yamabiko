"""Step 2 of the fine-tuning pilot: rebuild the base model's CTranslate2
export (``BASE_EXPORT_REPO``) of the 760M from its transformers source and
prove the two are the same.

Usage::

    python -m scripts.asr_finetune.reproduce_export --bench-python <release venv python>
        [--work-dir <dir>] [--set <set dir> ...] [--runs-dir <dir>]

Downloads ``efwkjn/whisper-ja-760M`` at ``SOURCE_REVISION`` into the work
directory, converts it with ``ct2-transformers-converter`` at the base
export's settings (``CONVERTER_ARGS``) into ``ct2\\repro``, and compares the
result file by file (SHA-256) with the base export at
``BASE_EXPORT_REVISION`` in the app's model cache, ``model.bin`` also
variable by variable. Then the bench decodes every comparison set twice,
both as local directories with the shipped decode settings: the base export's
pinned snapshot (label ``base``) and the conversion (label ``repro``), and
the window texts are compared. Steps 3 and 4 compare their decodes against
``repro``. The record goes to ``results\\reproduce_export.json``.
"""

from __future__ import annotations

import argparse

from scripts.asr_finetune.common import (
    BASE_EXPORT_REPO,
    BASE_EXPORT_REVISION,
    BASE_LABEL,
    CT2_SEED,
    REPRO_LABEL,
    SOURCE_REPO,
    SOURCE_REVISION,
    SOURCE_WEIGHTS,
    add_common_args,
    base_export_dir,
    bench_decode,
    compare_model_dirs,
    compare_run_sets,
    convert_to_ct2,
    ct2_dir,
    download_source,
    package_versions,
    resolve_bench_args,
    sha256_file,
    text_summary,
    weight_summary,
    write_result,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser)
    args = parser.parse_args(argv)
    sets = resolve_bench_args(parser, args)
    work = args.work_dir

    base = base_export_dir()
    source = download_source(work)
    out = ct2_dir(work, REPRO_LABEL)
    command = convert_to_ct2(source, out)
    files = compare_model_dirs(base, out)

    common = {"sets": sets, "runs_dir": args.runs_dir, "bench_python": args.bench_python, "work_dir": work}
    base_runs = bench_decode(label=BASE_LABEL, model=str(base), **common)
    repro_runs = bench_decode(label=REPRO_LABEL, model=str(out), **common)
    texts = compare_run_sets(base_runs, repro_runs)

    record = {
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "weights_sha256": sha256_file(source / SOURCE_WEIGHTS),
        },
        "base_export": {"repo": BASE_EXPORT_REPO, "revision": BASE_EXPORT_REVISION, "dir": str(base)},
        "converter_command": command,
        "conversion": str(out),
        "files": files,
        "ct2_seed": CT2_SEED,
        "texts_base_vs_repro": texts,
        "packages": package_versions(),
    }
    path = write_result(work, "reproduce_export", record)

    for name, entry in files["files"].items():
        state = "identical" if entry["identical"] else "same JSON" if entry.get("json_equal") else "differs"
        print(f"{name}: {state} ({entry['sha256']})")
    if "model_bin" in files:
        print(f"model.bin variables: {weight_summary(files['model_bin'])}")
    print(f"window texts, base vs repro: {text_summary(texts)}")
    print(f"record: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
