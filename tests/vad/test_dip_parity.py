"""Parity: the streaming FrameGate dip-splitter reproduces the offline
spike's piece boundaries on a recorded real prob trace.

dip_parity_trace.json holds one real digimon chunk (open..close probs) plus
``spike_spans`` -- the piece boundaries the offline dip spike
(vad_dip_spike.dip_split_chunk) produced for it. The chunk was chosen so
the spike's both-sides rule and this PR's online left-only rule agree
(every piece >= min_piece), making it a faithful parity anchor for the core
valley-cut logic: argmin selection, the min_piece gate, and loss-less
repartition. Driving the REAL gate with the recorded probs must yield the
same cut frames -- because a dip cut never resets Silero state, the prob
stream the offline spike scored is exactly the stream the online gate sees.

Seeded defect: off-by-one the cut index in _emit_dip_left (or drop/duplicate
a piece) -> the reconstructed boundary drifts off spike_spans and the test
goes red.

Note: this fixture does NOT exercise the argmin first-minimum tie-break --
its valley minima are unique (the deciding cut at frame 88 is a single-frame
run, and the earlier sub-threshold run has a unique argmin), so LAST- vs
FIRST-minimum selection produces identical boundaries here. The tie-break is
covered by test_dip_splits_glued_runs_into_two_pieces, whose valley is three
equal frames and asserts the cut lands on the FIRST.
"""
import json
from pathlib import Path

import numpy as np

from yamabiko.vad.chunker import ChunkerConfig
from yamabiko.vad.silero_chunker import FrameGate

FIXTURE = Path(__file__).parent / "dip_parity_trace.json"


def test_online_gate_reproduces_spike_piece_boundaries():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    probs = data["probs"]
    frame_samples = data["frame_samples"]
    expected_boundaries = [span[1] for span in data["spike_spans"]]

    cfg = ChunkerConfig(
        dip_split_enabled=True,
        dip_split_threshold=data["dip_split_threshold"],
        dip_split_min_piece_ms=data["dip_split_min_piece_ms"],
    )
    gate = FrameGate(cfg, threshold=data["threshold"],
                     frame_samples=frame_samples)

    # No pending -> pad_len 0, so an emit's frame count is exactly its span
    # of prob frames; cumulative counts give the boundaries in prob space.
    emit_frames, dip_flags = [], []
    for i, p in enumerate(probs):
        out = gate.push(np.full(frame_samples, float(i), dtype=np.float32), p)
        if out is not None:
            emit_frames.append(len(out) // frame_samples)
            dip_flags.append(gate.last_emit_was_dip_cut)

    boundaries = list(np.cumsum(emit_frames))
    assert boundaries == expected_boundaries, (
        f"online piece boundaries {boundaries} != spike {expected_boundaries}"
    )
    # The internal boundaries are dip cuts; the last emit is the close.
    assert dip_flags[:-1] == [True] * (len(dip_flags) - 1)
    assert dip_flags[-1] is False, "the final emit must be a true close"
    # Loss-less: the recorded chunk's frames are fully partitioned.
    assert boundaries[-1] == len(probs), (
        "pieces do not span the whole recorded chunk"
    )
