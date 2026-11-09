"""Keep E0 unit tests independent of the approved source pins.

``scripts.e0_evidence.contracts`` pins the exact production sources that real
E0 runs may record. Ordinary development edits those sources, so every test
not marked ``real`` approves the sources as they are now. Real-tier tests keep
the committed pins and the release-lock check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.e0_evidence import contracts, runtime
from scripts.e0_evidence.replay import vad_assist_layout
from yamabiko import __version__ as yamabiko_version
from yamabiko.asr.asr_client import FasterWhisperASR
from yamabiko.vad import sound_tagger
from yamabiko.vad.silero_chunker import FrameGate, SileroChunker, SileroVAD


_CONSTRAINTS_RELEASE = Path(__file__).resolve().parents[2] / "constraints-release.txt"


@pytest.fixture(autouse=True)
def _approve_current_sources(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.get_closest_marker("real") is not None:
        return
    monkeypatch.setattr(contracts, "YAMABIKO_VERSION_PIN", yamabiko_version)
    monkeypatch.setattr(
        contracts,
        "SILERO_VAD_SOURCE_SHA256",
        contracts.source_sha256(SileroVAD),
    )
    monkeypatch.setattr(
        contracts,
        "FRAME_GATE_SOURCE_SHA256",
        contracts.source_sha256(FrameGate),
    )
    monkeypatch.setattr(
        contracts,
        "WRAPPER_SOURCE_SHA256",
        contracts.source_sha256(FasterWhisperASR.transcribe),
    )
    monkeypatch.setattr(
        contracts,
        "SOUND_TAGGER_SOURCE_SHA256",
        contracts.source_sha256(sound_tagger),
    )
    monkeypatch.setattr(
        contracts,
        "SILERO_CHUNKER_SOURCE_SHA256",
        contracts.source_sha256(SileroChunker),
    )
    monkeypatch.setattr(
        contracts,
        "SOUND_TAGGER_MODEL_SHA256",
        sound_tagger.MODEL_SHA256,
    )
    monkeypatch.setattr(contracts, "SOUND_TAG_ASSIST_LAYOUT", vad_assist_layout())

    def accept_release_lock(_repo: Path) -> str:
        return hashlib.sha256(_CONSTRAINTS_RELEASE.read_bytes()).hexdigest()

    monkeypatch.setattr(runtime, "_verify_release_lock", accept_release_lock)
