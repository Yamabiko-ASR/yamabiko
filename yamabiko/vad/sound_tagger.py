"""Sound-event tags that raise Silero's speech probability (the VAD assist).

``SileroChunker`` hands every 512-sample frame to :class:`SoundTagAssist`. A
frame reaches Silero and ``FrameGate`` only once the bundled EfficientSED
tagger has tagged it, and the gate then receives

    max(silero, min(1, ASSIST_WEIGHT * (voice + shout)))

where ``voice`` and ``shout`` are the largest probabilities among the AudioSet
classes of each group in the frame's 40 ms tag frame. This is the fusion the
development ASR bench measured (``docs/ASR-BENCH.md``, "Sound tags" and
"Speech probability assists", on the ``feat/asr-improvements`` branch): on
real anime it caught more subtitle lines, short lines and interjections above
all, at the cost of the look-ahead below.

Streaming layout, the bench's: the graph hears one 10 s window (160,000
samples) and returns 250 tag frames of 640 samples. Block ``b`` since the last
reset keeps tag frames ``[12 b, 12 b + 12)`` from the window that ends 6 tag
frames (0.24 s) after the block, with zeros before the stream start. Chunker
frame ``k`` takes the tag frame holding its centre sample ``512 k + 256``. A
hop is 7,680 samples, exactly 15 chunker frames; the look-ahead is 3,840
samples, 7.5 chunker frames, so block ``b``'s window ends halfway through
chunker frame ``15 b + 22`` and runs once that frame arrives. A frame waits
0.26 to 0.70 s after it ends, plus one inference (about 35 ms on one CPU
thread).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from yamabiko.models import BundledModelInvalid
from yamabiko.paths import bundle_data_dir

_log = logging.getLogger(__name__)

# EfficientSED fmn10+TF:256 advanced-KD (theMoro/EfficientSED at
# e3449bfd1f4922b81d82410ed842653f462bfca8, MIT) exported to ONNX with its mel
# frontend inside the graph. Shipped in the bundle (installer/yamabiko.spec),
# license beside it in vad/data/.
MODEL_FILE = "efficientsed-fmn10-tf256-kd.onnx"
MODEL_SHA256 = "b0fdb11318fb666da2f21e59835d5e281278476854cf80442edd75446c81c7f0"

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # one chunker (Silero) frame
TAG_FRAME_SAMPLES = 640  # one tag frame, 40 ms
WINDOW_FRAMES = 250
WINDOW_SAMPLES = WINDOW_FRAMES * TAG_FRAME_SAMPLES  # 10 s
HOP_FRAMES = 12  # 0.48 s
LOOKAHEAD_FRAMES = 6  # 0.24 s
HOP_SAMPLES = HOP_FRAMES * TAG_FRAME_SAMPLES
LOOKAHEAD_SAMPLES = LOOKAHEAD_FRAMES * TAG_FRAME_SAMPLES
# First window row a block keeps: the rows after it are the hop, then the
# look-ahead.
_KEPT_ROW = WINDOW_FRAMES - LOOKAHEAD_FRAMES - HOP_FRAMES

# The bench's ``voice`` and ``shout`` groups (scripts/asr_bench/tag_sounds.py
# GROUPS on feat/asr-improvements), verbatim. The weight and the groups are
# the measured configuration; changing either needs a new bench run.
ASSIST_WEIGHT = 1.5
VOICE_CLASSES = (
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
    "Speech synthesizer", "Whispering", "Human voice",
)
SHOUT_CLASSES = (
    "Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Children shouting", "Screaming",
)

_METADATA_KEY = "yamabiko_tagger"
_METADATA_SCHEMA = "yamabiko-sound-tagger-onnx-1"


def default_model_path() -> Path:
    """The bundled graph, frozen or from a source checkout."""
    return bundle_data_dir() / "vad" / "data" / MODEL_FILE


def _verified_model(model_path: Path) -> Path:
    """The resolved file, unless it is missing or not the pinned graph."""
    real_file = Path(model_path).resolve()
    try:
        with real_file.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise BundledModelInvalid(
            f"sound tagger model cannot be read: {real_file} ({exc})",
            path=str(real_file),
        ) from exc
    if actual != MODEL_SHA256:
        raise BundledModelInvalid(
            f"sound tagger model fails its integrity check: expected SHA-256 "
            f"{MODEL_SHA256}, got {actual} for {real_file}",
            path=str(real_file),
        )
    return real_file


class SoundTagger:
    """The bundled tagger on ONNX Runtime's CPU provider.

    ``__call__`` takes one 10 s window (160,000 float32 samples) and returns
    the assist term of its 250 tag frames. One intra-op thread: about 35 ms
    a window, 7% of one core at one window per 0.48 s.
    """

    def __init__(self, model_path: Path) -> None:
        verified = _verified_model(model_path)

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # CPU only, named explicitly: a GPU-enabled onnxruntime must never
        # move the tagger onto the GPU beside CTranslate2.
        self._session = ort.InferenceSession(
            str(verified), sess_options=options, providers=["CPUExecutionProvider"],
        )
        raw = self._session.get_modelmeta().custom_metadata_map.get(_METADATA_KEY)
        metadata = json.loads(raw) if raw is not None else {}
        if metadata.get("schema") != _METADATA_SCHEMA:
            raise RuntimeError(f"{verified}: no {_METADATA_SCHEMA} metadata")
        classes = metadata["classes"]
        index = {name: column for column, name in enumerate(classes)}
        missing = [name for name in VOICE_CLASSES + SHOUT_CLASSES if name not in index]
        if missing:
            raise RuntimeError(f"{verified}: classes {missing} are not in the graph")
        self._voice = np.array([index[name] for name in VOICE_CLASSES])
        self._shout = np.array([index[name] for name in SHOUT_CLASSES])
        self._output_shape = (1, WINDOW_FRAMES, len(classes))
        _log.info(
            "vad sound-tag assist: loaded %s sha256=%s weight=%s hop_ms=%d "
            "lookahead_ms=%d provider=CPUExecutionProvider intra_op_threads=1",
            verified.name, MODEL_SHA256, ASSIST_WEIGHT,
            HOP_SAMPLES * 1000 // SAMPLE_RATE,
            LOOKAHEAD_SAMPLES * 1000 // SAMPLE_RATE,
        )

    def __call__(self, window: np.ndarray) -> np.ndarray:
        (probs,) = self._session.run(
            ["probs"], {"audio": window.reshape(1, WINDOW_SAMPLES)},
        )
        if probs.shape != self._output_shape:
            raise ValueError(f"sound tagger returned shape {probs.shape}")
        frames = probs[0]
        voice = frames[:, self._voice].max(axis=1)
        shout = frames[:, self._shout].max(axis=1)
        return np.minimum(np.float32(1.0), np.float32(ASSIST_WEIGHT) * (voice + shout))


# (frame, assist term, caller's per-frame value), in frame order.
Tagged = tuple[np.ndarray, float, object]


def _shift_in(window: np.ndarray, samples: np.ndarray) -> None:
    window[:-samples.size] = window[samples.size:]
    window[-samples.size:] = samples


def _block_window_end(block: int) -> int:
    """The stream sample, counted from the reset, at which the window that
    tags ``block`` ends: the block's end plus the look-ahead."""
    return (block + 1) * HOP_SAMPLES + LOOKAHEAD_SAMPLES


class SoundTagAssist:
    """Window buffer and delay line in front of Silero.

    ``push`` holds each 512-sample frame until the window that tags it has
    its 0.24 s of look-ahead, then returns the frames now tagged, in order.
    ``tagger`` maps a 10 s window to the assist term of its 250 tag frames
    (:class:`SoundTagger` in production). It runs in a worker thread through
    ``asyncio.to_thread`` so the event loop keeps going during inference; it
    is awaited before the next frame is admitted, so only one call is ever in
    flight and the window never changes under it.
    """

    def __init__(self, tagger: Callable[[np.ndarray], np.ndarray]) -> None:
        self._tagger = tagger
        self.reset()

    def reset(self) -> None:
        """Forget all audio and held frames: the next frame starts a new
        stream, with zeros before it."""
        self._window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        # Blocks tagged since the reset, and the stream sample the window ends at.
        self._blocks = 0
        self._window_end = 0
        # Frames received after the window's end, and their sample count.
        self._fill: list[np.ndarray] = []
        self._filled = 0
        # (frame ordinal since the reset, frame, caller's value), oldest first.
        self._held: deque[tuple[int, np.ndarray, object]] = deque()
        self._next_frame = 0

    @property
    def holding(self) -> bool:
        """True while frames wait for their tag."""
        return bool(self._held)

    async def push(self, frame: np.ndarray, meta: object) -> list[Tagged]:
        """Admit one frame; return every frame whose tag now exists."""
        self._held.append((self._next_frame, frame, meta))
        self._next_frame += 1
        self._fill.append(frame)
        self._filled += frame.size
        # The next block's window ends a hop and the look-ahead past its
        # start; it runs on the frame that completes it, and the samples of
        # that frame past its end wait for the window after it.
        end = _block_window_end(self._blocks)
        if self._window_end + self._filled < end:
            return []
        received = np.concatenate(self._fill)
        cut = end - self._window_end
        _shift_in(self._window, received[:cut])
        self._fill = [received[cut:]] if cut < received.size else []
        self._filled = received.size - cut
        self._window_end = end
        block = self._blocks
        self._blocks += 1
        return self._pop(block, await self._block_terms(self._window))

    async def drain(self) -> list[Tagged]:
        """Tag every held frame now, with zeros for the audio not received
        yet, as the bench pads an item's end.

        The stream state stays as it was: frames that arrive later continue
        the same timeline and are tagged from the real audio around them.
        """
        window = self._window.copy()
        window_end = self._window_end
        block = self._blocks
        tail = np.concatenate(self._fill) if self._fill else None
        tagged: list[Tagged] = []
        while self._held:
            end = _block_window_end(block)
            # The tail is shorter than this, or push would have run the window.
            incoming = np.zeros(end - window_end, dtype=np.float32)
            if tail is not None:
                incoming[:tail.size] = tail
                tail = None
            _shift_in(window, incoming)
            window_end = end
            tagged.extend(self._pop(block, await self._block_terms(window)))
            block += 1
        return tagged

    async def _block_terms(self, window: np.ndarray) -> np.ndarray:
        terms = await asyncio.to_thread(self._tagger, window)
        if terms.shape != (WINDOW_FRAMES,):
            raise ValueError(f"sound tagger returned {terms.shape} terms")
        return terms[_KEPT_ROW:_KEPT_ROW + HOP_FRAMES]

    def _pop(self, block: int, kept: np.ndarray) -> list[Tagged]:
        first = block * HOP_FRAMES
        tagged: list[Tagged] = []
        while self._held:
            ordinal, frame, meta = self._held[0]
            tag_frame = (ordinal * FRAME_SAMPLES + FRAME_SAMPLES // 2) // TAG_FRAME_SAMPLES
            if tag_frame >= first + HOP_FRAMES:
                break
            self._held.popleft()
            tagged.append((frame, float(kept[tag_frame - first]), meta))
        return tagged
