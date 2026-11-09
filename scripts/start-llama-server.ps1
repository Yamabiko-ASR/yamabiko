# llama.cpp router mode — single pinned model, OpenAI-compatible endpoint.
#
# `--models-max 1` + `--sleep-idle-seconds 0` is the Dec 2025 replacement for
# llama-swap's `swap: false` pin behavior. Single-model pipeline = no swap logic.
#
# Primary model: qwen3.5-9b (default tier since 2026-07-12; won the blind
# GOLD bench over shisa-v2.1 on both clean and ASR-path input).
# Prefer the dahara1 Japanese-imatrix GGUF over generic quants: imatrix
# calibration on JP data preserves more quality at low bit-widths.
#   https://huggingface.co/dahara1/Qwen3.5-9B-UD-japanese-imatrix
# Small tier runs the same model at UD-IQ3_XXS. For A/B against another
# GGUF, swap $ModelPath and $Alias.

param(
    [string]$ServerExe = "llama-server.exe",
    [string]$ModelPath = "M:\models\Qwen3.5-9B-UD-Q3_K_XL.gguf",
    [string]$Alias     = "qwen3.5-9b",
    [int]$Port         = 8090,
    [int]$CtxSize      = 2048
)

& $ServerExe `
    --model $ModelPath `
    --alias $Alias `
    --port $Port `
    --ctx-size $CtxSize `
    -ngl 99 `
    -fa on `
    --jinja `
    --models-max 1 `
    --sleep-idle-seconds -1 `
    --no-webui
