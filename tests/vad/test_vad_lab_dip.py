"""vad_lab replay reproduces the shipped dip-splitter.

The offline lab (scripts/vad_lab.py) is the tool used to measure the gate on
minutes of real audio. If its Variant/replay path can't turn the dip-splitter
ON, or resets Silero state on a dip cut, its numbers diverge from the shipped
runtime and the measurement lies. These tests drive `replay` with a scripted
VAD (no model file, no audio decode) and assert the two behaviours that keep
it faithful to SileroChunker.run:

  1. a dip-enabled Variant actually reaches ChunkerConfig -> a glued A/B run
     splits into two chunks (plumbing);
  2. the per-emit Silero reset is SKIPPED on a dip cut and fires only at the
     true close (state continuity), mirroring run()'s
     `if not last_emit_was_dip_cut: reset_states()`.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_vad_lab():
    """Import scripts/vad_lab.py as a module (it is a script, not a package).

    The module must be registered in sys.modules before exec: its Variant
    dataclass uses PEP 563 string annotations, and dataclasses resolves them
    via sys.modules[cls.__module__]."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "vad_lab.py"
    spec = importlib.util.spec_from_file_location("vad_lab", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _ScriptedVAD:
    """Stand-in for SileroVAD: returns a fixed prob per frame and counts the
    reset_states() calls replay makes. Ignores the model_path replay passes."""

    def __init__(self, probs):
        self._probs = list(probs)
        self._i = 0
        self.reset_calls = 0

    def __call__(self, _frame):
        p = self._probs[self._i]
        self._i += 1
        return p

    def reset_states(self):
        self.reset_calls += 1


# A glued A/B exchange: run A (40) | sub-0.15 valley (3, argmin frame 40) |
# run B (40) | silence close (20). Dip ON must cut at the valley.
_GLUED = [0.9] * 40 + [0.05] * 3 + [0.9] * 40 + [0.01] * 20


def test_replay_dip_variant_splits_glued_run_and_keeps_state_on_cut(monkeypatch):
    vad_lab = _load_vad_lab()
    audio = np.zeros(len(_GLUED) * vad_lab.FRAME, dtype=np.float32)

    scripted = _ScriptedVAD(_GLUED)
    monkeypatch.setattr(vad_lab, "SileroVAD", lambda *a, **k: scripted)
    variant = vad_lab.Variant("attach-0.3-dip", threshold=0.3,
                              dip_split_enabled=True)
    chunks = vad_lab.replay(audio, variant, Path("unused.onnx"))

    # Plumbing: the dip cfg reached ChunkerConfig, so the glued run split.
    assert len(chunks) == 2, (
        f"dip config did not reach ChunkerConfig: expected a cut + a close, "
        f"got {len(chunks)} chunk(s)"
    )
    # State continuity: the reset was SKIPPED on the dip cut and fired only
    # at the close. Unconditional reset (the pre-fix bug) -> 2.
    assert scripted.reset_calls == 1, (
        f"reset_states must be skipped on the dip cut and fire only at the "
        f"close; got {scripted.reset_calls}"
    )


def test_replay_dip_off_does_not_split_same_trace(monkeypatch):
    """Control: same glued trace with dip OFF stays a single chunk, proving
    the split above is caused by the dip config, not the input shape."""
    vad_lab = _load_vad_lab()
    audio = np.zeros(len(_GLUED) * vad_lab.FRAME, dtype=np.float32)

    scripted = _ScriptedVAD(_GLUED)
    monkeypatch.setattr(vad_lab, "SileroVAD", lambda *a, **k: scripted)
    variant = vad_lab.Variant("attach-0.3", threshold=0.3)  # dip default OFF
    chunks = vad_lab.replay(audio, variant, Path("unused.onnx"))

    assert len(chunks) == 1, "dip OFF must not split a glued run"
    assert scripted.reset_calls == 1, "the single close must reset once"
