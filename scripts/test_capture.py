"""Standalone WASAPI loopback capture smoke test.

Usage:
    python scripts/test_capture.py          # 5 seconds -> capture_test.wav
    python scripts/test_capture.py 10       # 10 seconds

Runs LoopbackCapture for N seconds while whatever is playing through your
default output device is captured, then writes mono 16 kHz PCM to
capture_test.wav. Open the file and listen — it should sound like what
you played.
"""
from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yamabiko.capture.audio import AudioConfig, LoopbackCapture  # noqa: E402


async def amain(duration_s: float) -> int:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    capture = LoopbackCapture(AudioConfig(), loop, q)
    capture.start()
    info = capture._device_info
    assert info is not None
    print(f"[test_capture] device: {info['name']} @ {int(info['defaultSampleRate'])} Hz, {info['maxInputChannels']} ch")
    print(f"[test_capture] capturing {duration_s:.1f}s ...")

    buf: list[np.ndarray] = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        try:
            block = await asyncio.wait_for(q.get(), timeout=0.5)
            buf.append(block)
        except asyncio.TimeoutError:
            pass

    capture.stop()
    pcm = np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)
    print(f"[test_capture] captured {len(pcm)} samples = {len(pcm)/16000:.2f}s")

    if len(pcm) == 0:
        print("[test_capture] FAIL: no audio captured — is anything playing?")
        return 1

    rms = float(np.sqrt(np.mean(pcm**2)))
    peak = float(np.max(np.abs(pcm)))
    print(f"[test_capture] RMS={rms:.4f}  peak={peak:.4f}")

    out_path = ROOT / "capture_test.wav"
    pcm_i16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm_i16.tobytes())
    print(f"[test_capture] wrote {out_path}")
    return 0


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    sys.exit(asyncio.run(amain(dur)))
