"""Tests for the nvidia-smi VRAM probe (yamabiko.gpu).

All external behaviour is stubbed out via monkeypatch -- these tests
never actually shell out. Coverage matrix:

* nvidia-smi absent from PATH                 -> detect_vram_mib() is None
* nvidia-smi returns a clean "8192" line      -> 8192
* nvidia-smi returns "16303" (RTX 5080)       -> 16303
* nvidia-smi returns garbage                  -> None (ValueError swallowed)
* subprocess raises TimeoutExpired            -> None
* subprocess raises OSError                   -> None
* empty stdout                                -> None
* suggest_quant_tier threshold behaviour at 10240 MiB boundary
* detect_gpu_name happy path + failure path
* detect_gpu_summary one-query row            -> GpuSummary; unparsable row -> None
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from yamabiko import gpu


# ---------------------------------------------------------------------------
# detect_vram_mib
# ---------------------------------------------------------------------------


def test_detect_vram_mib_none_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert gpu.detect_vram_mib() is None


def test_detect_vram_mib_parses_8gb(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *a, **kw: "8192\n",
    )
    assert gpu.detect_vram_mib() == 8192


def test_detect_vram_mib_parses_16gb_rtx5080(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *a, **kw: "16303\n",
    )
    assert gpu.detect_vram_mib() == 16303


def test_detect_vram_mib_none_on_garbage(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *a, **kw: "not-a-number\n",
    )
    assert gpu.detect_vram_mib() is None


def test_detect_vram_mib_none_on_timeout(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")

    def _boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert gpu.detect_vram_mib() is None


def test_detect_vram_mib_none_on_oserror(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")

    def _boom(*args, **kwargs):
        raise OSError("driver crashed")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert gpu.detect_vram_mib() is None


def test_detect_vram_mib_none_on_empty_stdout(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "")
    assert gpu.detect_vram_mib() is None


def test_query_nvidia_smi_suppresses_console_window(monkeypatch):
    """Both nvidia-smi shell-outs (VRAM + GPU name) must pass
    ``creationflags=subprocess.CREATE_NO_WINDOW``. The wizard's
    QuantTierPage constructs inside the frozen yamabiko.exe, which is a
    windowed bundle (console=False in installer/yamabiko.spec); without
    the flag Windows allocates a visible conhost for each
    console-subsystem child, flashing windows over / stealing focus
    from the first-run wizard. Same convention as watchdog.py and its
    guard test test_sweep_suppresses_console_allocation.

    Recording stub -- NOT a silent ``**kw`` swallower (the pre-existing
    lambda stubs in this file pass whether or not the flag is set;
    charter §3.9 Wired-not-exercised).

    Seeded defect: drop ``creationflags=subprocess.CREATE_NO_WINDOW``
    from _query_nvidia_smi's check_output call -> this fails.
    """
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    calls: list[dict] = []

    def _record(*args, **kwargs):
        calls.append(kwargs)
        return "8192\n"

    monkeypatch.setattr(subprocess, "check_output", _record)

    assert gpu.detect_vram_mib() == 8192
    assert gpu.detect_gpu_name() == "8192"  # value irrelevant; flags aren't
    assert len(calls) == 2, f"expected 2 nvidia-smi invocations, saw {len(calls)}"
    for kwargs in calls:
        assert kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW, (
            f"nvidia-smi check_output missing "
            f"creationflags=CREATE_NO_WINDOW; a conhost window would "
            f"flash over the frozen wizard on every QuantTierPage "
            f"construction. kwargs={kwargs!r}"
        )


# ---------------------------------------------------------------------------
# detect_gpu_name
# ---------------------------------------------------------------------------


def test_detect_gpu_name_happy_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *a, **kw: "NVIDIA GeForce RTX 5080\n",
    )
    assert gpu.detect_gpu_name() == "NVIDIA GeForce RTX 5080"


def test_detect_gpu_name_none_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert gpu.detect_gpu_name() is None


# ---------------------------------------------------------------------------
# suggest_quant_tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vram,expected",
    [
        (None, "default"),
        (4096, "small"),
        (8192, "small"),
        (10239, "small"),
        (10240, "default"),
        (12288, "default"),
        (16303, "default"),
        (24576, "default"),
    ],
)
def test_suggest_quant_tier_thresholds(vram, expected):
    assert gpu.suggest_quant_tier(vram) == expected


# ---------------------------------------------------------------------------
# tier_overrides
# ---------------------------------------------------------------------------


def test_tier_overrides_default_returns_float16():
    """Standard tier couples FP16 faster-whisper to the default GGUF quant."""
    patch = gpu.tier_overrides("default")
    assert patch == {
        "translation": {"quant_tier": "default"},
        "asr": {"faster_whisper": {"compute_type": "float16"}},
    }


def test_tier_overrides_small_returns_int8_float16():
    """Small tier couples INT8/FP16 faster-whisper to the Q3_K_M GGUF quant.

    int8_float16 (NOT pure int8) is safe on sm_70+ incl. Blackwell;
    the pure-int8 sm_120 CUBLAS crash does not apply because the
    INT8 GEMM kernel isn't used at all under int8_float16.
    """
    patch = gpu.tier_overrides("small")
    assert patch == {
        "translation": {"quant_tier": "small"},
        "asr": {"faster_whisper": {"compute_type": "int8_float16"}},
    }


def test_tier_overrides_unknown_falls_back_to_default(caplog):
    """Malformed user_overrides.yaml must not crash callers."""
    import logging

    with caplog.at_level(logging.WARNING, logger="yamabiko.gpu"):
        patch = gpu.tier_overrides("garbage")
    assert patch == gpu.tier_overrides("default")
    assert any("unknown quant_tier" in r.message for r in caplog.records)


def test_tier_overrides_returns_fresh_copies():
    """Caller mutation must not leak into the module-level table."""
    a = gpu.tier_overrides("small")
    a["translation"]["quant_tier"] = "mutated"
    b = gpu.tier_overrides("small")
    assert b["translation"]["quant_tier"] == "small"


# ---------------------------------------------------------------------------
# detect_gpu_summary (the startup error window's GPU field)
# ---------------------------------------------------------------------------


def test_detect_gpu_summary_reads_name_vram_and_driver_in_one_query(monkeypatch):
    """L6: ONE nvidia-smi call asks for name, memory.total and
    driver_version and the first row becomes a GpuSummary.

    Seeded defects: three separate queries, a missing driver_version field,
    or the columns in another order -> red."""
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    calls: list[list[str]] = []

    def _record(args, **kwargs):
        calls.append(list(args))
        return "NVIDIA GeForce RTX 5080, 16303, 617.42\nNVIDIA GeForce RTX 3060, 12288, 617.42\n"

    monkeypatch.setattr(subprocess, "check_output", _record)
    assert gpu.detect_gpu_summary() == gpu.GpuSummary("NVIDIA GeForce RTX 5080", 16303, "617.42")
    assert [call[1] for call in calls] == ["--query-gpu=name,memory.total,driver_version"]


def test_detect_gpu_summary_is_none_for_a_row_it_cannot_parse(monkeypatch):
    """A driver that reports "[N/A]" for the memory must leave the window's
    GPU field at "None reported by Windows", not crash the error window.

    Seeded defect: int() unguarded -> ValueError escapes -> red."""
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **kw: "NVIDIA GeForce RTX 5080, [N/A], 617.42\n")
    assert gpu.detect_gpu_summary() is None
