"""Real-path tests for the nvidia-smi subprocess seam (yamabiko.gpu).

The project's real-dependency test policy requires a ``@pytest.mark.real``
test for every subprocess call site. ``yamabiko/gpu.py`` shells out to
``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` and every
test in ``tests/test_gpu.py`` stubs that shell-out (its own docstring
says "these tests never actually shell out"). A wrong query field, a
flag removed in a driver update, or a parse drift would make
``detect_vram_mib()`` return None on every real machine — and the
wizard's QuantTierPage (wizard_pages/quant_tier.py) would silently
recommend the "default" tier even on 8 GB cards.

Skips only on the environment precondition (no nvidia-smi on PATH =
no NVIDIA driver), mirroring the narrow-skip idiom of
tests/test_real_pyaudio.py. The probe functions themselves run with
NO mocks: real shutil.which, real subprocess, real driver output.
"""
from __future__ import annotations

import re
import shutil

import pytest

pytestmark = pytest.mark.real


def _skip_if_no_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not on PATH — no NVIDIA driver on this box")


def test_real_nvidia_smi_vram_probe_returns_positive_mib() -> None:
    """``detect_vram_mib()`` against the REAL nvidia-smi returns int > 0.

    Goes red if: the ``--query-gpu=memory.total`` field name drifts or
    is typoed, the ``--format=csv,noheader,nounits`` flags change shape
    (units would break int()), or the first-row parse breaks — every
    one of which collapses to a silent ``None`` in production.

    Seeded-defect check: change ``memory.total`` to ``memory.totall``
    in yamabiko/gpu.py -> nvidia-smi errors, the helper returns None,
    this fails.
    """
    _skip_if_no_nvidia_smi()
    from yamabiko.gpu import detect_vram_mib

    vram = detect_vram_mib()
    assert isinstance(vram, int), (
        f"detect_vram_mib() returned {vram!r} with a real NVIDIA driver "
        f"present — the nvidia-smi query/parse seam is broken (the "
        f"wizard would silently suggest the wrong quant tier)"
    )
    assert vram > 0, f"detect_vram_mib() returned a non-positive VRAM: {vram}"


def test_real_nvidia_smi_gpu_name_probe_returns_nonempty_string() -> None:
    """``detect_gpu_name()`` against the REAL nvidia-smi returns a
    non-empty marketing name.

    Goes red if the ``name`` query field or the csv,noheader format
    drifts — production would show no GPU name on the wizard page.

    Seeded-defect check: change ``--format=csv,noheader,nounits`` to
    a bogus format string in yamabiko/gpu.py -> subprocess fails,
    helper returns None, this fails.
    """
    _skip_if_no_nvidia_smi()
    from yamabiko.gpu import detect_gpu_name

    name = detect_gpu_name()
    assert isinstance(name, str) and name.strip(), (
        f"detect_gpu_name() returned {name!r} with a real NVIDIA driver "
        f"present — the nvidia-smi name query seam is broken"
    )


def test_real_nvidia_smi_gpu_summary_reports_name_vram_and_driver() -> None:
    """``detect_gpu_summary()`` against the REAL nvidia-smi: one query for
    name, memory.total and driver_version returns all three (the startup
    error window's GPU field).

    Seeded-defect check: change ``driver_version`` to ``driver_versio`` in
    yamabiko/gpu.py -> nvidia-smi rejects the query, the helper returns
    None, this fails.
    """
    _skip_if_no_nvidia_smi()
    from yamabiko.gpu import detect_gpu_summary

    summary = detect_gpu_summary()
    assert summary is not None, (
        "detect_gpu_summary() returned None with a real NVIDIA driver present "
        "-- the combined name/memory/driver query is broken"
    )
    assert summary.name.strip()
    assert summary.vram_mib > 0
    assert re.fullmatch(r"\d+\.\d+", summary.driver), summary.driver
