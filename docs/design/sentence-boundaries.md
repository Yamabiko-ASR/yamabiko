# Sentence Boundaries

How Yamabiko decides when a Japanese utterance has ended and is ready
to translate. Getting this wrong is the difference between smooth live
subtitles and a "wait several seconds, then a burst of lines all at
once" experience.

## The problem

Japanese sentence-final particles (です / ます / ね / よ / か / でしょう)
and trailing vowels carry voiced energy well past a naive silence
threshold. With a VAD-only cut policy you get two failure modes:

- **Under-cut**: the VAD waits for a long silence that never comes, and
  two or three sentences arrive fused together. The translator either
  compresses them into one garbled EN sentence, or the hard chunk-cap
  fires mid-sentence and truncates.
- **Over-cut**: the VAD cuts on a micro-pause inside a sentence, and
  the translator produces half-sentence EN output that has to be
  visually replaced when the rest arrives.

The fix is a two-stage pipeline: VAD for audio gating, and a Japanese
sentence splitter for the case where a single ASR chunk contained
multiple complete sentences. A stream-scoped buffer holds any trailing
partial sentence and prepends it to the next chunk.

## Silero VAD: audio gate

The pinned community Silero VAD v5-era ONNX graph runs on the 16 kHz
audio stream via `onnxruntime` (no PyTorch). It answers a simpler
question than "is the sentence over": *is there voice in this frame?*
We use it to decide when to hand a chunk of audio to ASR at all, not to
decide sentence boundaries.

The chunker hands the Yamabiko wrapper one 512-sample external frame at
a time. The wrapper prepends 64 samples of rolling context, so the raw
ONNX graph receives 576 samples per inference.

The chunker is a frame-by-frame state machine (`vad/silero_chunker.py`,
`FrameGate`). A chunk opens on the first frame whose speech probability
reaches `vad.threshold` (default `0.3`) and, once open, is sustained
down to a lower floor (`threshold - 0.15`, mirroring upstream Silero's
hysteresis) so a music bed dipping the probability mid-sentence does
not read as silence. It closes once `min_silence_ms` (default 400 ms)
of consecutive sub-floor frames have accumulated.

The speech probability the gate reads is Silero's raised by a sound
tagger (`vad/sound_tagger.py`): the bundled EfficientSED model scores
every 40 ms for voice and shouting classes, and each frame gets
`max(silero, min(1, 1.5 * (voice + shout)))`. The tagger looks 0.24 s
ahead, so every gate decision, and every chunk, comes 0.26 to 0.70 s
after the frame it is about. Silero still scores the frames in order
and is still reset at each true close. The threshold, the sustain floor,
the drop floor and the dip splitter all apply to this fused value.

The `0.3` default exists for produced media. Anime and drama mix
dialogue under music and effects that depress the speech probability;
at the conservative `0.5` gate, whole reaction lines on real episodes
were silently skipped. Raise it back toward `0.5` for clean-speech
sources such as news.

Several refinements keep boundaries clean on Japanese speech:

- **Pre-roll.** Up to `speech_pad_ms` (default 200 ms) of audio
  captured just before speech onset is prepended from a ring buffer, so
  the chunk does not clip the start of the first syllable.
- **Short-utterance glue.** If the speech in a closing chunk is shorter
  than `min_chunk_ms` (default 500 ms), the gate keeps the chunk open
  across the brief gap rather than emitting a fragment, which avoids
  over-segmenting Japanese filler. The glue is itself capped by
  `glue_max_silence_ms` (default 1500 ms): an isolated short utterance
  trailed by real silence is emitted instead of gluing all the way to
  the chunk cap.
- **Attach, don't extend.** Audio between chunks is buffered (up to
  `attach_max_ms`, default 6 s) and prepended to the next chunk, so the
  recognizer hears the borderline audio the gate was unsure about. The
  VAD segments; it does not filter. Only audio that stayed below
  `drop_floor_prob` (default 0.10) for `drop_floor_ms` (default 2 s)
  straight is discarded as silence. This roughly halves truncated
  one-word fragments on busy scenes without changing subtitle timing.

Hitting `max_chunk_ms` (default 8000 ms) force-emits the chunk
regardless of silence, so the pipeline can never stall waiting for a
pause that does not come.

The looser gate admits some non-speech, so a junk gate runs after
transcription and drops chunks using the recognizer's own confidence
signals (no-speech probability, average log-probability, compression
ratio). The gate is evaluated per transcription chunk, not per split
segment: faster-whisper computes those statistics per decode window
and copies them onto every segment it splits out, so a per-segment
rule would let a repetition loop escape through its own fragments.

### Dip splitting (optional, off by default)

A rapid back-and-forth exchange often has no real pause between turns, so
the gate never closes and both speakers land in one subtitle. Silero's
per-frame probability still dips at each turn boundary even when the
energy never falls to silence. The dip splitter watches for such a dip
inside an open chunk and, once the probability recovers, cuts the chunk at
the dip's lowest frame so the two turns become separate lines. It only
cuts once the left piece is at least `dip_split_min_piece_ms` (default
1000 ms) long, and only when the dip falls below `dip_split_threshold`
(default 0.15).

The feature ships disabled (`vad.dip_split_enabled: false`). At the
initial thresholds it regressed overall subtitle quality in a blind A/B:
it fired far more often than its burst-exchange target and cut inside
single sentences. The threshold keys stay in the config so burst-heavy
content can turn it on by hand, and it remains off by default until a
retune clears a full A/B without regressing. With it off the gate is
byte-identical to the pre-feature behavior.

## Incremental emission (Faster subtitles)

A window often holds several sentences, and without help each of them waits
for the window to close. With Faster subtitles on (`asr.incremental.enabled`,
the default), the ASR stage also decodes the open window as it grows: one
second after the gate opens it, then every second, or 250 ms after a decode
that leaves a complete sentence not yet committed
(`yamabiko/asr/incremental.py`). A sentence that ends in 。！？?! and reads
the same, with every sentence before it, in the last two decodes goes on at
once as an ordinary transcript item. A sentence that is only うん, はい or ええ
(or only punctuation) never goes on early: decodes of a window's first second
or two sometimes hear such a lone reply where the final decode writes none, so
it and every sentence after it wait for the close. When the window closes, its
final decode runs as before and only the text after the committed sentences
goes on, so the partial-sentence buffer below still applies to the window's
last sentence.

The decodes read only audio the gate has already admitted, so a committed
sentence never holds the next window's first words. Final decodes always run
before step decodes, and a step that falls more than a second behind is
dropped, so on a busy GPU the stage falls back toward waiting for the close.
Committed text is never withdrawn.

## fast-bunkai: sentence splitter

A single ASR chunk sometimes contains multiple complete sentences,
because the speaker ran through two before pausing. The cut is correct,
but the translator works best on one sentence at a time.

[fast-bunkai](https://github.com/hotchpotch/fast-bunkai) splits Japanese
text on `。！？` with JP-aware handling of the usual edge cases:
decimals (`3.14`), parenthetical annotations (`(笑)`), URLs, ellipses,
and trailing particles with no terminator. It also occasionally emits a
bare terminator as its own segment when one trails a parenthetical
(`すごい(笑)。`); the splitter merges any such orphan back onto the
previous sentence. fast-bunkai's ML linebreak model is available but off
by default, since it adds a download and per-call latency for marginal
gain on broadcast content.

### Partial-sentence buffer

When a chunk ends mid-sentence, the trailing partial is held in a
stream-scoped remainder (`pending_remainder`) and prepended to the next
chunk's transcript. The buffer only kicks in when the chunk *also*
carried at least one complete sentence, the genuine mid-sentence-cut
case. A window's final remainder after sentences of the same window went
out early (`follows_early`, see Incremental emission above) is held too,
even when it is a lone partial. A chunk that is a single partial with no
terminator is otherwise translated as-is: faster-whisper often drops the
terminal `。` even on complete utterances, so buffering those would stall
the pipeline.

If the remainder grows past 300 characters without ever seeing a
terminator, a long monologue read without natural pauses, the gate
force-flushes it with a `[...]` prefix so the user sees something rather
than silent accumulation.

### Pause/resume boundary

Pause stops new audio intake. Pause and resume are both hard continuity
boundaries: accumulated waveform/VAD state and any unfinished Japanese
sentence remainder are abandoned, so resumed audio and text cannot be
joined to partial pre-boundary input. Completed English lines in the
translation context are retained; the boundary does not reset that
established context.

Work already emitted downstream is not cancelled. An in-flight subtitle
may therefore finish and appear while paused; the overlay does not hide or
retract it.

## Observability

Each translated segment is logged to
`%LOCALAPPDATA%\Yamabiko\logs\latency.log` as one JSON line, used for
p50/p95 latency analysis:

```json
{"ts": 1718000000.0, "asr_ms": 115, "translate_ms": 130,
 "total_ms": 245, "jp_len": 24, "en_len": 31}
```

`latency_log_enabled: true` controls the file. With
`transcript_log_enabled: true` (off by default, because it writes
speech transcripts to disk) each record also carries the raw `jp` (ASR
text) and `en` (translation), which separates an ASR mishearing from a
translation error in the log alone.

Logs are split per process: the pipeline writes `main-pipeline.log`, the
watchdog `main-watchdog.log`, and the overlay `main-overlay.log`.
Boundary and buffering decisions (a dropped non-Japanese segment, a
partial force-flush) are logged inline in
`main-pipeline.log`, so you can trace why a given cut happened without
grepping the JSON.

## What this is not

- **Not speaker diarization.** A single active speaker at a time is
  assumed; multi-speaker audio currently merges into one subtitle
  stream.
