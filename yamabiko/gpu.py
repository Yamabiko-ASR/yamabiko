"""NVIDIA GPU detection via nvidia-smi shell-out.

Tiny helper to avoid pulling in torch / pynvml just for a one-off VRAM probe
at wizard time. Returns None if nvidia-smi is missing or fails (e.g. no
NVIDIA driver, WSL without passthrough, or a transient timeout). Callers
must handle None.

Syntax reference: `nvidia-smi --help-query-gpu` lists `memory.total` and
`name` as stable query fields; `--format=csv,noheader,nounits` gives
plain integer MiB (and plain strings for `name`). Verified against
driver 566.x on 2026-04-15.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import NamedTuple, Optional


def _query_nvidia_smi(field: str) -> Optional[str]:
    """Run ``nvidia-smi --query-gpu=<field>`` and return the first row.

    Returns ``None`` if nvidia-smi isn't on PATH, the subprocess fails,
    or the output is empty. Timeout is 5 s; a VRAM probe that takes
    longer than that on a healthy driver is already a bad sign, and we
    don't want to block the wizard.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        # CREATE_NO_WINDOW: the wizard runs inside the frozen
        # yamabiko.exe, a windowed (console=False) bundle -- without the
        # flag Windows allocates a visible conhost for nvidia-smi that
        # flashes over / steals focus from the first-run wizard
        # (same convention as watchdog.py's sweep).
        out = subprocess.check_output(
            [exe, f"--query-gpu={field}",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5.0,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    stripped = out.strip()
    if not stripped:
        return None
    return stripped.splitlines()[0].strip()


def detect_vram_mib() -> Optional[int]:
    """Return the first detected NVIDIA GPU's total VRAM in MiB, or None."""
    raw = _query_nvidia_smi("memory.total")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def detect_gpu_name() -> Optional[str]:
    """Return the first detected NVIDIA GPU's marketing name, or None."""
    return _query_nvidia_smi("name") or None


class GpuSummary(NamedTuple):
    """The first NVIDIA GPU as nvidia-smi reports it."""

    name: str
    vram_mib: int
    driver: str


def detect_gpu_summary() -> Optional[GpuSummary]:
    """Return the first NVIDIA GPU's name, total VRAM (MiB) and driver
    version from ONE nvidia-smi call, or None when nvidia-smi is missing,
    fails, or prints a row this can't parse. The startup error window's GPU
    field ("NVIDIA GeForce RTX 5080, 16 GB, driver 617.42")."""
    raw = _query_nvidia_smi("name,memory.total,driver_version")
    if raw is None:
        return None
    # rsplit: a marketing name may itself hold a comma; the last two
    # columns never do.
    parts = [part.strip() for part in raw.rsplit(",", 2)]
    if len(parts) != 3 or not parts[0] or not parts[2]:
        return None
    try:
        vram = int(parts[1])
    except ValueError:
        return None
    return GpuSummary(parts[0], vram, parts[2])


"""Canonical per-tier override patch.

The wizard's ``QuantTierPage`` and the overlay Settings dialog both
persist a coupled pair of keys when the user picks a tier:

* ``translation.quant_tier`` — selects which translator GGUF the wizard
  pulls on next launch (see ``yamabiko.models.TRANSLATOR_PINS``:
  ~4 GB for ``small``; ~5.1 GB for ``default``).
* ``asr.faster_whisper.compute_type`` — flips faster-whisper's CT2
  backend. ``small`` switches to ``int8_float16`` (INT8 weights with
  FP16 compute; saves ~2 GB VRAM vs FP16 at near-FP16 WER on sm_70+
  including Blackwell), which brings the total VRAM budget from
  ~8-9 GB down to ~6-7 GB so an 8 GB card fits Whisper + the
  translator + desktop overhead without OOM. ``default`` keeps
  ``float16`` for 12 GB+ cards where the headroom doesn't matter.

Coupled-write design: both keys are written together by the same UI
action so the runtime just reads ``asr.faster_whisper.compute_type``
as a regular config key — no load-time magic. If a user edits
``compute_type`` by hand post-tier-selection to tune further, that
edit stays intact; the next tier-change UI action will overwrite it
(documented tradeoff — advanced users who want both hand-tuned
compute_type *and* the standard tier can re-edit after).

Unknown tier names fall back to the ``default`` patch and log a
warning, so a malformed ``user_overrides.yaml`` never crashes the
callers of this helper.
"""


_TIER_OVERRIDES: dict[str, dict] = {
    "default": {
        "translation": {"quant_tier": "default"},
        "asr": {"faster_whisper": {"compute_type": "float16"}},
    },
    "small": {
        "translation": {"quant_tier": "small"},
        "asr": {"faster_whisper": {"compute_type": "int8_float16"}},
    },
}


def tier_overrides(tier: str) -> dict:
    """Return the coupled override dict for ``tier``.

    Unknown tier names are logged and coerced to ``default``; the
    returned dict is a fresh copy, safe for the caller to mutate.
    """
    import copy
    import logging

    if tier not in _TIER_OVERRIDES:
        logging.getLogger(__name__).warning(
            "unknown quant_tier %r; falling back to 'default'", tier,
        )
        tier = "default"
    return copy.deepcopy(_TIER_OVERRIDES[tier])


# The Smaller tier's floor: its budget (~6-7 GB, the ``small`` patch above)
# fits an 8 GB card; under that no tier fits. In whole GB as nvidia-smi's
# total rounds (an 8 GB card may report 8188 MiB).
SMALL_TIER_MIN_GB = 8


def suggest_quant_tier(vram_mib: Optional[int]) -> str:
    """Map detected VRAM to a quant_tier recommendation.

    Thresholds:
      >= 10240 MiB (~10 GB) -> default (UD-Q4_K_XL ~5.5 GB + whisper float16 ~3.5 GB)
      <  10240 MiB          -> small   (Q3_K_M ~4 GB + whisper float16 ~3.5 GB)
      None                  -> default (conservative fallback: assume the user
                                        knows their hardware; the wizard page
                                        surfaces the detection failure so the
                                        user can override manually)
    """
    if vram_mib is None:
        return "default"
    return "small" if vram_mib < 10240 else "default"
