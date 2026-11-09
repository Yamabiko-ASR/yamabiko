"""Run ``scripts.asr_bench.run`` with CTranslate2's random generators seeded.

Usage::

    python -m scripts.asr_finetune.seeded_bench_run --seed N <scripts.asr_bench.run arguments>

faster-whisper falls back to sampling (temperature above 0) on a window
whose beam result fails its compression-ratio or log-probability gate, and
CTranslate2 seeds its sampler from ``std::random_device`` unless
``ctranslate2.set_random_seed`` runs first, so two decodes of one model can
differ on those windows. The pilot compares models window by window, so it
fixes the seed before anything loads; the runner, the chunker and the
production ASR path are otherwise unchanged. Runs in the release venv.
"""

from __future__ import annotations

import argparse
import runpy
import sys

import ctranslate2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=False)
    parser.add_argument("--seed", type=int, required=True)
    args, rest = parser.parse_known_args()
    ctranslate2.set_random_seed(args.seed)
    sys.argv = ["scripts.asr_bench.run", *rest]
    runpy.run_module("scripts.asr_bench.run", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
