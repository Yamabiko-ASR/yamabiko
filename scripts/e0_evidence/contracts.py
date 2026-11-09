"""The source and dependency contracts an E0 run is recorded under.

The replay and decoder recorders refuse to run unless the live sources and
dependencies match these values, and ``run`` copies every value into
``run-start.json`` as ``approved_contracts``. ``verify-run`` checks each record
against the run's own copy, so a later change here never re-judges a run that
was recorded under an earlier contract.

Read the values as ``contracts.NAME`` at call time, not with
``from ... import NAME``, so a test can replace them in one place.
"""

from __future__ import annotations

import hashlib
import inspect

from scripts.e0_evidence.wire import canonical_sha256


YAMABIKO_VERSION_PIN = "0.2.0-beta"
# SHA-256 of the UTF-8 ``inspect.getsource`` text: ``SileroVAD``, ``FrameGate``
# (whose private state the replay probes), and ``FasterWhisperASR.transcribe``.
SILERO_VAD_SOURCE_SHA256 = (
    "31925fe2029d748a3483c20366b36b39a6c57bc7780ec4d690966933507a7f82"
)
FRAME_GATE_SOURCE_SHA256 = (
    "39395f7b7a30c30f55255acde6d33fcc7df5fb10c5401c69d3ac007b7243b640"
)
WRAPPER_SOURCE_SHA256 = (
    "13c24dd2a35d46ae293ce57bb3a31b70ef2d2842f29b736b894fa5c87cd24601"
)
# SHA-256 of ``str(signature)`` for faster-whisper's ``WhisperModel.transcribe``
# without ``self``.
DECODER_SIGNATURE_SHA256 = (
    "5b3936efd79cc4aa99edaf6d3465eb745dd16b10a85a9c2b9472d4a485cc24f6"
)
FASTER_WHISPER_VERSION = "1.2.1"
# The VAD sound-tag assist production fuses into Silero's probability. SHA-256
# of the UTF-8 ``inspect.getsource`` text of the whole ``yamabiko.vad.sound_tagger``
# module (``SoundTagger``, ``SoundTagAssist`` and the helpers and constants they
# use) and of ``SileroChunker``, whose frame path the replay mirrors: frames
# wait in the assist, Silero scores each one as it leaves, the gate receives
# ``max(silero, term)``, and Silero resets after every true close.
SOUND_TAGGER_SOURCE_SHA256 = (
    "5499fd7efd2b4d8a9b9ee8ca1c13cca15fe4ffead7f82246d0b30a1910f767e4"
)
SILERO_CHUNKER_SOURCE_SHA256 = (
    "4c462817e38698a31ac76998d3c6e8e1b43d7f51171e1d948d8bb9e99f2cee16"
)
# The bundled tagger graph, ``sound_tagger.MODEL_SHA256``.
SOUND_TAGGER_MODEL_SHA256 = (
    "b0fdb11318fb666da2f21e59835d5e281278476854cf80442edd75446c81c7f0"
)
# The assist's fusion and streaming layout, as ``sound_tagger``'s constants
# define it. ``approved_contracts`` records its canonical SHA-256.
SOUND_TAG_ASSIST_LAYOUT: dict[str, object] = {
    "assist_weight": 1.5,
    "frame_samples": 512,
    "hop_frames": 12,
    "lookahead_frames": 6,
    "shout_classes": (
        "Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Children shouting",
        "Screaming",
    ),
    "tag_frame_samples": 640,
    "voice_classes": (
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Babbling", "Speech synthesizer", "Whispering", "Human voice",
    ),
    "window_frames": 250,
}
# The replay contract record's shape, the FrameGate fields it probes, and the
# frame record's shape.
PROBE_CONTRACT_VERSION = 4
# The faster-whisper decoder-call record's contract. The NeMo TDT adapter
# versions its own decoder-call records separately.
DECODER_CONTRACT_VERSION = 2
# Passed to ctranslate2.set_random_seed and recorded with every call. It does not
# make temperature-fallback sampling repeatable: CTranslate2 seeds its sampling
# generator once per thread, so reseeding before a later call has no effect.
DECODER_RANDOM_SEED = 105


def source_sha256(subject: object) -> str:
    """Hash a class or function's source the way the contracts above were."""

    return hashlib.sha256(inspect.getsource(subject).encode("utf-8")).hexdigest()


def approved_contracts() -> dict[str, object]:
    """Every contract value, read now, as ``run-start.json`` records it."""

    return {
        "decoder_contract_version": DECODER_CONTRACT_VERSION,
        "decoder_random_seed": DECODER_RANDOM_SEED,
        "decoder_signature_sha256": DECODER_SIGNATURE_SHA256,
        "faster_whisper_version": FASTER_WHISPER_VERSION,
        "frame_gate_source_sha256": FRAME_GATE_SOURCE_SHA256,
        "probe_contract_version": PROBE_CONTRACT_VERSION,
        "silero_chunker_source_sha256": SILERO_CHUNKER_SOURCE_SHA256,
        "silero_vad_source_sha256": SILERO_VAD_SOURCE_SHA256,
        "sound_tag_assist_layout_sha256": canonical_sha256(SOUND_TAG_ASSIST_LAYOUT),
        "sound_tagger_model_sha256": SOUND_TAGGER_MODEL_SHA256,
        "sound_tagger_source_sha256": SOUND_TAGGER_SOURCE_SHA256,
        "wrapper_source_sha256": WRAPPER_SOURCE_SHA256,
        "yamabiko_version_pin": YAMABIKO_VERSION_PIN,
    }
