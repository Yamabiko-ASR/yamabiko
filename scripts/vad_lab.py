"""Offline long-audio VAD lab: measure what the SileroChunker gate drops.

Born 2026-07-02: live anime audio produced 80s transcript gaps and
sub-second fragments while news audio was fine. The 3-second-snippet
test fixtures can never surface that, so this tool replays MINUTES of
audio through the production SileroVAD + FrameGate, offline, and
reports speech coverage against a reference timeline.

Usage (system Python 3.13, repo root):
  py -3.13 scripts/vad_lab.py --synth-minutes 4 --snr-db 8
  py -3.13 scripts/vad_lab.py --audio path\\to\\episode.mkv
  py -3.13 scripts/vad_lab.py --audio clip.wav --asr

Reference timeline:
  --synth-minutes : exact (the fixture generator knows every line).
  --audio         : full-file faster-whisper (default whisper pin) segments.

Variants replayed per run:
  legacy-full     pre-attach, pre-hysteresis, threshold 0.5 (the
                  original gate) -- attach_max_ms=200 shrinks the ring
                  back to the legacy speech_pad_ms window.
  pr71-gate       PR #71 shipped gate (hysteresis + 0.3) WITHOUT attach
                  (same shrunk ring as legacy-full).
  attach-0.3      attach-don't-extend, dip-splitter OFF (the
                  segmentation baseline).
  attach-0.3-dip  true shipped defaults: attach + dip-splitter ON
                  (matches default_config.yaml).

This is a DEV DIAGNOSTIC. It reuses production classes unmodified;
"legacy" behavior is emulated by setting attach_max_ms == the
speech_pad_ms default so production's pending-deque maxlen floor
(max(attach_frames, pad_frames, 1)) collapses to the legacy ring --
not by a separate code path. drop_floor_ms is inert in that config;
the module-level assert guards the attach_max_ms == speech_pad_ms
invariant the emulation depends on.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yamabiko.vad.chunker import ChunkerConfig  # noqa: E402
from yamabiko.vad.silero_chunker import (  # noqa: E402
    FrameGate,
    SileroVAD,
    _default_model_path,
)

SR = 16000
FRAME = 512

# The legacy-emulation variants (legacy-full, pr71-gate) set
# attach_max_ms=200 to shrink production's pending deque back to the
# pre-attach speech_pad_ms ring. That only holds while attach_max_ms
# <= the default speech_pad_ms (production's maxlen floor is
# max(attach_frames, pad_frames, 1)); if speech_pad_ms's default is
# ever lowered, these variants silently grow a wrong ring and the
# measurement "before" baseline breaks. Fail loud instead.
assert ChunkerConfig().speech_pad_ms >= 200, (
    "legacy-emulation variants assume speech_pad_ms default >= 200; "
    "update the attach_max_ms=200 literal in VARIANTS if this changed"
)

# ---------------------------------------------------------------- fixture

# Anime-flavored dialogue: interjections, mid-length lines, long lines.
# Mixed deliberately -- the live failure hit short reaction lines hardest.
_LINES = [
    "おはよう、今日もいい天気だね",
    "え?",
    "ちょっと待ってよ、そんなの聞いてないんだけど",
    "うん",
    "本当に大丈夫なの?怪我とかしてない?",
    "まさか!",
    "昨日の夜、変な音が聞こえた気がしたんだ",
    "はい",
    "それでね、先生が明日までに提出しろって言うんだよ",
    "嘘でしょ",
    "あの店のラーメン、めちゃくちゃおいしかったよ",
    "そうなんだ",
    "逃げろ!早くこっちに来い!",
    "いやだ",
    "君がいなかったら、僕はここまで来られなかったと思う",
    "ありがとう",
    "この先の森には何かが潜んでいる気がする",
    "行こう",
    "お前、いつからそこにいたんだ",
    "ずっとだよ",
    "みんなで力を合わせれば、きっと乗り越えられるはずだ",
    "そうだね",
    "静かに。誰かが近づいてくる",
    "分かった",
]

_VOICES = ["Microsoft Haruka", "Microsoft Ayumi", "Microsoft Ichiro", "Microsoft Sayaka"]


def _tts_lines(workdir: Path) -> list[Path]:
    """Render every dialogue line to a 16 kHz mono wav via SAPI."""
    ps_lines = [
        "Add-Type -AssemblyName System.Speech",
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)",
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
    ]
    outs = []
    for i, (line, voice) in enumerate(
        zip(_LINES, (_VOICES * len(_LINES))[: len(_LINES)])
    ):
        out = workdir / f"line{i:02d}.wav"
        outs.append(out)
        text = line.replace("'", "''")
        ps_lines += [
            f"$s.SelectVoice('{voice}')",
            f"$s.SetOutputToWaveFile('{out}', $fmt)",
            f"$s.Speak('{text}')",
        ]
    ps_lines.append("$s.Dispose()")
    script = workdir / "tts.ps1"
    # utf-8-sig: powershell.exe 5.1 parses BOM-less .ps1 as ANSI and
    # mangles the Japanese text into parse errors.
    script.write_text("\n".join(ps_lines), encoding="utf-8-sig")
    # pwsh 7, not powershell.exe 5.1: under Windows PowerShell the same
    # SelectVoice + SetOutputToWaveFile + Speak sequence writes 46-byte
    # header-only wavs (observed 2026-07-02); PowerShell 7 renders audio.
    import shutil
    shell = shutil.which("pwsh") or "powershell"
    subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        check=True, capture_output=True, timeout=300,
    )
    return outs


def _read_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, (
            f"{path}: expected 16 kHz mono, got {w.getframerate()} Hz "
            f"x{w.getnchannels()}"
        )
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _music_bed(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic BGM: detuned harmonic chords + noise hats + kick thumps.

    Not art -- a broadband, rhythmic masker resembling an anime music
    bed closely enough to stress the VAD the same way.
    """
    t = np.arange(n_samples, dtype=np.float32) / SR
    out = np.zeros(n_samples, dtype=np.float32)
    roots = [110.0, 87.31, 130.81, 98.0]  # A2 F2 C3 G2
    bar = 2.4
    for i in range(int(math.ceil(n_samples / SR / bar))):
        s = int(i * bar * SR)
        e = min(int((i + 1) * bar * SR), n_samples)
        if s >= e:
            break
        seg_t = t[s:e] - t[s]
        f0 = roots[i % len(roots)]
        seg = np.zeros(e - s, dtype=np.float32)
        for mult, amp in ((1, 0.5), (1.5, 0.35), (2, 0.3), (3, 0.15), (4, 0.08)):
            for detune in (0.998, 1.0, 1.003):
                seg += amp * np.sin(2 * np.pi * f0 * mult * detune * seg_t)
        seg *= (0.6 + 0.4 * np.exp(-seg_t / 1.8))  # bar envelope
        # kick thump at bar start
        kick_len = min(int(0.12 * SR), e - s)
        kt = seg_t[:kick_len]
        seg[:kick_len] += 1.2 * np.sin(2 * np.pi * (90 - 300 * kt) * kt) * np.exp(-kt / 0.04)
        out[s:e] += seg
    # noise hats every 0.3 s
    hat = 0.3
    for i in range(int(n_samples / SR / hat)):
        s = int(i * hat * SR)
        e = min(s + int(0.03 * SR), n_samples)
        out[s:e] += rng.normal(0, 0.25, e - s).astype(np.float32) * np.exp(
            -np.arange(e - s) / (0.008 * SR)
        )
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def make_fixture(minutes: float, snr_db: float, outdir: Path,
                 music: bool = True) -> tuple[Path, Path]:
    """Assemble dialogue-over-music fixture + exact reference JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outdir) as td:
        clips = [_read_wav_mono16k(p) for p in _tts_lines(Path(td))]
    rng = np.random.default_rng(20260702)
    total = int(minutes * 60 * SR)
    timeline = np.zeros(total, dtype=np.float32)
    ref = []
    cursor = int(1.0 * SR)
    i = 0
    while True:
        clip = clips[i % len(clips)].copy()
        clip *= 0.1 / max(_rms(clip), 1e-6)  # normalize speech to -20 dBFS RMS
        if cursor + len(clip) + SR >= total:
            break
        timeline[cursor:cursor + len(clip)] += clip
        ref.append({
            "start": cursor / SR,
            "end": (cursor + len(clip)) / SR,
            "text": _LINES[i % len(_LINES)],
        })
        # natural gaps; every 8th line a long music-only break
        gap = 8.0 if (i % 8 == 7) else float(rng.uniform(0.3, 2.2))
        cursor += len(clip) + int(gap * SR)
        i += 1
    if music:
        bed = _music_bed(total, rng)
        bed *= (0.1 * (10 ** (-snr_db / 20))) / max(_rms(bed), 1e-6)
        timeline += bed
    peak = np.max(np.abs(timeline))
    if peak > 0.99:
        timeline *= 0.99 / peak
    tag = f"{minutes:g}min_snr{snr_db:g}" if music else f"{minutes:g}min_clean"
    wav_path = outdir / f"fixture_{tag}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((timeline * 32767).astype(np.int16).tobytes())
    ref_path = outdir / f"fixture_{tag}.ref.json"
    ref_path.write_text(json.dumps(ref, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return wav_path, ref_path


# ---------------------------------------------------------------- decode

def decode_media(path: Path) -> np.ndarray:
    """Any media file -> 16 kHz mono float32 via PyAV (bundled dep)."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    pcm = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                pcm.append(np.frombuffer(rf.to_ndarray().tobytes(), dtype=np.int16))
    audio = np.concatenate(pcm).astype(np.float32) / 32768.0
    return audio


def whisper_reference(audio: np.ndarray) -> list[dict]:
    """Full-file default-whisper transcription as the speech-timeline reference."""
    from yamabiko import hf_env  # noqa: F401  (pin HF_HOME)
    from yamabiko.asr.faster_whisper_audio_compat import import_whisper_model
    from yamabiko.models import WHISPER_REPO, WHISPER_REVISION

    WhisperModel = import_whisper_model()
    model = WhisperModel(WHISPER_REPO, device="cuda", compute_type="float16",
                         revision=WHISPER_REVISION)
    segs, _info = model.transcribe(audio, language="ja", beam_size=5,
                                   vad_filter=False,
                                   condition_on_previous_text=False)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segs if s.text.strip()]


# ---------------------------------------------------------------- replay

@dataclass
class Variant:
    name: str
    threshold: float = 0.3
    # None = production gate as-is (built-in neg = threshold - 0.15).
    # A float overrides the derived sustain bar; setting it equal to
    # threshold emulates the pre-2026-07-02 single-threshold gate.
    neg_override: float | None = None
    reset_per_emit: bool = True
    # None = shipped attach behavior. attach_max_ms=200 shrinks
    # production's pending deque back to the legacy speech_pad_ms ring
    # (maxlen floor is max(attach_frames, pad_frames, 1), so this holds
    # only while attach_max_ms == the speech_pad_ms default). drop_floor_ms
    # is INERT in that config -- the trim's while-len>pad_frames can never
    # fire once maxlen already equals pad_frames -- but is set for clarity.
    attach_max_ms: int | None = None
    drop_floor_ms: int | None = None
    # Dip-splitter (cuts glued A/B exchanges at internal probability
    # valleys). OFF by default so legacy-emulation variants stay
    # byte-identical to the pre-feature gate; threshold/min_piece None =
    # the shipped ChunkerConfig tuning (0.15 / 1000 ms). Only plumbed into
    # ChunkerConfig when dip_split_enabled is True.
    dip_split_enabled: bool = False
    dip_split_threshold: float | None = None
    dip_split_min_piece_ms: int | None = None


@dataclass
class Chunk:
    start: float
    end: float
    audio: np.ndarray = field(repr=False)


def replay(audio: np.ndarray, variant: Variant, model_path: Path) -> list[Chunk]:
    """Mirror SileroChunker.run's synchronous core over a whole file."""
    vad = SileroVAD(model_path)
    cfg_kwargs = {}
    if variant.attach_max_ms is not None:
        cfg_kwargs["attach_max_ms"] = variant.attach_max_ms
    if variant.drop_floor_ms is not None:
        cfg_kwargs["drop_floor_ms"] = variant.drop_floor_ms
    if variant.dip_split_enabled:
        cfg_kwargs["dip_split_enabled"] = True
        if variant.dip_split_threshold is not None:
            cfg_kwargs["dip_split_threshold"] = variant.dip_split_threshold
        if variant.dip_split_min_piece_ms is not None:
            cfg_kwargs["dip_split_min_piece_ms"] = variant.dip_split_min_piece_ms
    gate = FrameGate(ChunkerConfig(**cfg_kwargs),
                     threshold=variant.threshold, frame_samples=FRAME)
    if variant.neg_override is not None:
        gate._neg_threshold = variant.neg_override
    pusher = gate
    chunks: list[Chunk] = []
    n_frames = len(audio) // FRAME
    open_fi = 0
    pad_s = ChunkerConfig().speech_pad_ms / 1000
    for fi in range(n_frames):
        frame = audio[fi * FRAME:(fi + 1) * FRAME]
        prob = vad(frame)
        was_open = gate._in_speech
        emit = pusher.push(frame, prob)
        if not was_open and gate._in_speech:
            open_fi = fi
        if emit is not None:
            # Span from the gate-open frame (minus pre-roll pad), not
            # end-minus-len: glue-cap emits trim trailing silence, which
            # would shift a len-derived start late and misclassify
            # captured short lines as missed.
            start = max(0.0, open_fi * FRAME / SR - pad_s)
            chunks.append(Chunk(start, (fi + 1) * FRAME / SR, emit))
            # Mirror SileroChunker.run: a dip cut leaves the gate open
            # mid-utterance, so the Silero hidden state must NOT reset or the
            # right piece is scored against a cold state and its own dip
            # boundaries drift from the offline spike. Reset only at true
            # closes (silence / max-chunk), same as production.
            if variant.reset_per_emit and not gate.last_emit_was_dip_cut:
                vad.reset_states()   # production behavior
    return chunks


# ---------------------------------------------------------------- metrics

def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _close_delay_percentile(ref, chunks, pct):
    """Per reference line covered by a chunk: chunk end minus line end.
    The subtitle-latency proxy (I2): must not grow vs legacy."""
    delays = []
    for r in ref:
        covering = [c for c in chunks
                    if min(c.end, r["end"]) - max(c.start, r["start"])
                    >= 0.5 * (r["end"] - r["start"])]
        if covering:
            delays.append(min(c.end for c in covering) - r["end"])
    return float(np.percentile(delays, pct)) if delays else float("nan")


def coverage(ref: list[dict], chunks: list[Chunk], audio_len_s: float) -> dict:
    # Union first: pre-roll padding makes adjacent spans overlap, and
    # double-counting intersections pushed coverage past 100%.
    spans = _union([(c.start, c.end) for c in chunks])

    def overlap(a0, a1):
        return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in spans)

    total = sum(r["end"] - r["start"] for r in ref)
    got = sum(overlap(r["start"], r["end"]) for r in ref)
    missed = [r for r in ref
              if overlap(r["start"], r["end"]) < 0.5 * (r["end"] - r["start"])]
    # Histogram over AUDIO length (what ASR receives), not span length:
    # glue-cap spans include held silence and would hide fragmentation.
    durs = [len(c.audio) / SR for c in chunks]
    return {
        "coverage_pct": 100.0 * got / total if total else 0.0,
        "ref_lines": len(ref),
        "missed_lines": missed,
        "chunks": len(chunks),
        "frag_lt_1p5s": sum(1 for d in durs if d < 1.5),
        "median_chunk_s": float(np.median(durs)) if durs else 0.0,
        "transcribed_ratio": sum(durs) / audio_len_s if audio_len_s else 0.0,
        "close_delay_p50": _close_delay_percentile(ref, chunks, 50),
        "close_delay_p95": _close_delay_percentile(ref, chunks, 95),
    }


def run_asr(chunks: list[Chunk]) -> list[str]:
    from yamabiko import hf_env  # noqa: F401
    from yamabiko.asr.faster_whisper_audio_compat import import_whisper_model
    from yamabiko.models import WHISPER_REPO, WHISPER_REVISION

    WhisperModel = import_whisper_model()
    model = WhisperModel(WHISPER_REPO, device="cuda", compute_type="float16",
                         revision=WHISPER_REVISION)
    out = []
    for c in chunks:
        segs, _ = model.transcribe(c.audio, language="ja", beam_size=5,
                                   condition_on_previous_text=False)
        out.append("".join(s.text for s in segs).strip())
    return out


# Per-emit reset STAYS in every candidate: on the 2026-07-02 real-episode
# run, removing it cratered coverage 81% -> 62% (92+ missed lines).
VARIANTS = [
    # Pre-attach, pre-hysteresis, threshold 0.5 (the original gate).
    Variant("legacy-full", threshold=0.5, neg_override=0.5,
            attach_max_ms=200, drop_floor_ms=32),
    # PR #71 shipped gate (hysteresis + 0.3) WITHOUT attach.
    Variant("pr71-gate", threshold=0.3,
            attach_max_ms=200, drop_floor_ms=32),
    # Attach-don't-extend, dip-splitter OFF: the segmentation baseline the
    # dip variant is compared against.
    Variant("attach-0.3", threshold=0.3),
    # Attach + dip-splitter ON, built like every variant from ChunkerConfig()
    # defaults (min_chunk 1000 ms, min_silence 500 ms), not from
    # default_config.yaml. Shows the dip-splitter's effect on fragmentation
    # next to attach-0.3.
    Variant("attach-0.3-dip", threshold=0.3, dip_split_enabled=True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-minutes", type=float)
    ap.add_argument("--snr-db", type=float, default=8.0)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--ref", type=Path, help="reference JSON (else whisper)")
    ap.add_argument("--asr", action="store_true",
                    help="also transcribe emitted chunks per variant")
    ap.add_argument("--asr-variants", default=VARIANTS[0].name,
                    help="comma-separated variant names for the --asr pass")
    ap.add_argument("--missed-for", default=None,
                    help="also print missed lines for this variant")
    ap.add_argument("--ignore-spans", default="",
                    help="comma list of a-b second ranges excluded from "
                         "the reference (OP/ED songs)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / ".vad-lab")
    args = ap.parse_args()

    if args.synth_minutes:
        wav, refp = make_fixture(args.synth_minutes, args.snr_db, args.out,
                                 music=not args.no_music)
        print(f"fixture: {wav}")
        audio = _read_wav_mono16k(wav)
        ref = json.loads(refp.read_text(encoding="utf-8"))
    elif args.audio:
        audio = decode_media(args.audio)
        print(f"decoded {args.audio}: {len(audio)/SR:.1f}s")
        if args.ref:
            ref = json.loads(args.ref.read_text(encoding="utf-8"))
        else:
            print("building whisper reference (full file) ...")
            ref = whisper_reference(audio)
            refp = args.out / (args.audio.stem + ".whisper-ref.json")
            refp.parent.mkdir(parents=True, exist_ok=True)
            refp.write_text(json.dumps(ref, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            print(f"reference saved: {refp} ({len(ref)} segments)")
    else:
        ap.error("need --synth-minutes or --audio")

    if args.ignore_spans:
        spans = [tuple(map(float, s.split("-")))
                 for s in args.ignore_spans.split(",")]
        before = len(ref)
        ref = [r for r in ref
               if not any(r["start"] < e and r["end"] > s for s, e in spans)]
        print(f"ignore-spans: {before - len(ref)} reference segments excluded")

    audio_len_s = len(audio) / SR
    model_path = _default_model_path()
    print(f"\n{'variant':14} {'coverage':>9} {'chunks':>7} {'<1.5s':>6} "
          f"{'median':>7} {'ratio':>6} {'p50':>6} {'p95':>6} "
          f"{'missed lines':>12}")
    results = {}
    for v in VARIANTS:
        chunks = replay(audio, v, model_path)
        m = coverage(ref, chunks, audio_len_s)
        results[v.name] = (m, chunks)
        print(f"{v.name:14} {m['coverage_pct']:8.1f}% {m['chunks']:7d} "
              f"{m['frag_lt_1p5s']:6d} {m['median_chunk_s']:6.2f}s "
              f"{m['transcribed_ratio']:6.2f} {m['close_delay_p50']:6.2f} "
              f"{m['close_delay_p95']:6.2f} "
              f"{len(m['missed_lines']):4d}/{m['ref_lines']}")

    show_missed = [VARIANTS[0].name]
    if args.missed_for and args.missed_for in results:
        show_missed.append(args.missed_for)
    for name in show_missed:
        missed = results[name][0]["missed_lines"]
        if missed:
            print(f"\n{name} missed lines (>=50% outside any chunk):")
            for r in missed[:40]:
                print(f"  [{r['start']:7.1f}s-{r['end']:7.1f}s] {r['text']}")
            if len(missed) > 40:
                print(f"  ... and {len(missed) - 40} more")

    if args.asr:
        for name in args.asr_variants.split(","):
            if name not in results:
                continue
            m, chunks = results[name]
            texts = run_asr(chunks)
            # False-positive readout: emitted chunks ASR can't hear
            # anything in are junk the low threshold let through.
            junk = sum(1 for t in texts if len(t) < 2)
            print(f"\n--- ASR through {name}: {len(chunks)} chunks, "
                  f"{junk} junk (<2 chars, {100 * junk / max(len(chunks), 1):.0f}%)")
            for c, text in zip(chunks, texts):
                print(f"  [{c.start:7.1f}s] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
