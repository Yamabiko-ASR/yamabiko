# Yamabiko Roadmap

Living document. Updated as plans shift. Sections are ordered by
dependency, not by promise; no dates.

## Current state

`v0.2.0-beta` focuses on making the pipeline hold up on real media and in the
field:

- ASR moved to Kodama-1, Yamabiko's own Japanese fine-tune of
  whisper-ja-760M.
- Voice detection was reworked for produced media (anime and drama mix
  dialogue under music), recovering whole lines that the conservative
  gate used to skip.
- Series-aware character names: pick a show and its character names
  keep their official English spellings.
- Field diagnosability: per-process logs, native-crash evidence, and a
  one-click diagnostics bundle from the tray.

The beta installer remains unsigned. Code signing is planned for a later
release.

ASR and translation both run locally. Model dependencies:

- [Kodama-1](https://huggingface.co/Yamabiko-ASR/kodama-1)
  (Yamabiko's fine-tune of [whisper-ja-760M](https://huggingface.co/efwkjn/whisper-ja-760M))
- [`Qwen3.5-9B`](https://huggingface.co/dahara1/Qwen3.5-9B-UD-japanese-imatrix)
  (Apache-2.0, both translation tiers)

---

## Near-term

### Series-aware names, next steps

The first version shipped with a manual series picker. Planned
follow-ups:

- **Auto-detect**: suggest the series from what you are watching
  instead of requiring a manual pick.
- **Consistency memory**: remember how a name was rendered earlier in a
  session and keep later lines consistent with it, beyond what the
  series list covers.

### Gaming and Learning genre presets

Currently ship four presets (variety, anime, news, drama). Two more
were originally scoped but deferred:

- **Gaming**: preserve game-specific terminology, shorter sentences,
  streamer-chat register.
- **Learning**: Japanese-only output mode (no English translation),
  intended for JP study. Furigana and romaji rendered, no translator
  call.

Slot these in when the base pipeline is stable and a gold eval set for
each register exists. No dependency on the work below.

---

## Translator LoRA on Qwen3.5

**Goal:** address recurring translation errors on anime, variety TV,
streaming, and gaming content without replacing the base model.

**Approach:** train LoRA adapters on top of `Qwen3.5-9B` (Apache-2.0,
the shipped translator since v0.2.0-beta), one per genre preset that
needs it (variety, anime, gaming, drama; news and learning use the base
model unchanged). The base model stays identical; adapters ship as GGUF
LoRAs that llama.cpp loads alongside it, so switching presets swaps the
adapter rather than reloading the base. Training data combines open
parallel corpora with synthetic in-domain pairs, gated by a
hand-corrected gold eval set per preset.

Note that we ship an imatrix-quantized GGUF, not the full-precision
base. LoRA training happens against the upstream base weights, and the
adapters are quantized to match the shipped quant afterwards, so this
work starts with confirming an adapter trained on the base still lands
correctly on the quant we actually serve.

---

## Smaller translator research

**Goal:** reduce the Smaller tier's VRAM use and generation time while keeping
the current 9B model available as Standard.

**Approach:** evaluate distilling the LoRA-specialized 9B translator into a
smaller Apache-2.0 base. Any replacement must remain compatible with the
existing Settings tier workflow.

**Bar:** a bounded, measured quality regression on the gold eval set at
roughly a third of the size. If the distilled model cannot hit that
bar, keep the 9B and revisit.

**Prerequisite:** the translator LoRA work above. Distillation needs the
high-quality teacher and eval sets it produces.

---

## Current ASR direction

The ASR backend is `faster-whisper` running Kodama-1, a decoder fine-tune
of whisper-ja-760M. Training an ASR model from scratch is out of scope:
Whisper-class models train on orders of magnitude more audio than is
legally available to us. Domain vocabulary gaps such as character names
and game terms are handled at the translation layer through the series
gazetteer and future adapter work.

**Revisit trigger:** a specific, measured JP ASR failure mode that
cannot be fixed elsewhere in the pipeline. Not before.

---

## Deferred

- **Speaker diarization and per-speaker subtitles**: real value for
  multi-person streams, but touches every pipeline stage.
- **Cloud sync of settings and glossaries**: conflicts with the current
  local-processing model; revisit only on repeated user demand.
- **Mobile companion app**: large scope; wait for demand.
- **Chunk-level semantic revision**: catch ASR hallucinations before
  translation amplifies them. Low priority.
- **Translation-quality self-eval**: have the translator rate its own
  confidence. Interesting, not obviously useful to users.
