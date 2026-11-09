"""Standalone ASR smoke test.

Usage:
    python scripts/test_asr.py <path-to-wav>
    python scripts/test_asr.py            # uses silent 3s clip only

Loads the configured ASR backend, warms up, transcribes silence (to absorb
cold-start), then transcribes the provided WAV and reports wall-clock time.
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yamabiko.asr.asr_client import build_asr  # noqa: E402
from yamabiko.config import load_full_config  # noqa: E402


def load_wav_mono_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        idx = (np.arange(int(len(pcm) * ratio)) / ratio).astype(np.int64)
        pcm = pcm[idx]
    return pcm


def main() -> int:
    cfg = load_full_config()

    print("[test_asr] backend=faster_whisper")
    t0 = time.perf_counter()
    asr = build_asr(cfg["asr"])
    print(f"[test_asr] model load: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    asr.warmup()
    print(f"[test_asr] warmup (silent 1s): {time.perf_counter() - t0:.2f}s")

    if len(sys.argv) < 2:
        print("[test_asr] no WAV provided -- warmup only. Pass a WAV path to transcribe.")
        return 0

    wav_path = Path(sys.argv[1])
    pcm = load_wav_mono_16k(wav_path)
    print(f"[test_asr] loaded {wav_path.name}: {len(pcm)/16000:.2f}s, {len(pcm)} samples")

    for i in range(3):
        t0 = time.perf_counter()
        result = asr.transcribe(pcm, 16000)
        dt = time.perf_counter() - t0
        print(f"[test_asr] run {i+1}: {dt*1000:.0f} ms  text={result.text!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
