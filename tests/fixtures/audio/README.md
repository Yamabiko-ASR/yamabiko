# Audio fixtures

## cv17_ja_8lines

Eight Japanese read-speech lines used by
`tests/test_real_pipeline_line_recall.py` to check that every spoken line
reaches the translator through the production VAD, ASR and translation
stages.

- `cv17_ja_8lines.s16le`: 16 kHz mono signed 16-bit little-endian PCM.
  0.5 s of silence, the eight clips separated by 1.0 s of silence, then
  1.5 s of silence.
- `cv17_ja_8lines.json`: source, pinned revision, license, attribution,
  sample rate, and each line's clip name, sentence and
  `[start_sample, end_sample)` span in the PCM file.

### Attribution and license

Mozilla Common Voice 17.0, Japanese test split (via
[fsicoli/common_voice_17_0](https://huggingface.co/datasets/fsicoli/common_voice_17_0)),
revision `8262c16bf297c87a9cd88c51997c4758ed7a8ba2`. Released under
CC0-1.0.

### Regenerating

From the repository root, with ffmpeg on `PATH`, a CUDA GPU, and the
default ASR model already in Yamabiko's model cache:

```
python tests/fixtures/audio/build_cv17_ja_lines.py
```

The dataset files (`transcript/ja/test.tsv` and
`audio/ja/test/ja_test_0.tar`, 199,857,664 bytes) are downloaded into
`--cache-dir`, which defaults to a directory under the system temp dir and
never points into the repository or Yamabiko's own model cache.

Selection: rows with `up_votes >= 2`, `down_votes == 0`, an 8 to 30
character sentence and no ASCII letters or digits, sorted by `path`, first
row per `client_id`. Each candidate is decoded by ffmpeg and transcribed
alone by the production ASR on the shipped default config; it is accepted
when its character error rate after minimal Japanese normalization is at
most 0.25. The first eight accepted clips are kept. The screen depends on
GPU ASR output, so a different GPU or model build can accept a different
set; the committed files are the reference.
