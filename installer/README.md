# Building the Yamabiko installer

## Prerequisites
- A Python 3.12 build venv holding exactly the release lock: `.venv-release`,
  created as `docs/RELEASE.md` ("Release venv") shows. `build.ps1` compares
  its `pip freeze` with `constraints-release.txt` and stops on any
  difference, so a dev venv with extra packages is refused.
- NSIS 3.x **Unicode** build on PATH. Verify via `makensis /HDRINFO | findstr NSIS_UNICODE`; output should include `NSIS_UNICODE`.
- `third_party/llama.cpp/llama-server.exe` + colocated CUDA DLLs (see `third_party/README.md`).
- An NVIDIA dev box (any sm_75+); CUDA 12 toolkit on PATH.

## Build
From the repo root:
```powershell
.\installer\build.ps1 -Venv .venv-release
```
`-Venv` takes a folder name under the repo root or an absolute path; without
it the build uses `.venv` (as CI does).
Produces `installer/yamabiko-<version>-setup.exe` (~1.5 GB; it bundles the
CUDA runtime + the vendored llama-server translation engine; the first-run
wizard then downloads ~6.6 GB of models, ~5.6 GB on the Smaller tier, into
the HF cache separately).

About 24 minutes of the build's 29 go to makensis compressing the bundle.

## The installer's look: art and license text

The installer's pages (design "Full landscape") come from committed files
next to `yamabiko.nsi`, so the build needs no renderer:

- `art/*.svg`: the landscape, seal and header art. `art/build_art.py`
  renders each one at 100, 150 and 200 % with headless Edge into the
  committed `art/<name>-<pct>.bmp` (24-bit). After editing an SVG run
  `.venv\Scripts\python.exe installer\art\build_art.py` (it needs Pillow
  from the dev venv and Edge) and commit the BMPs. The installer picks the
  scale that fits the system DPI when it starts.
- `LICENSE.rtf`: the license page's formatted copy of `LICENSE`, written
  by `installer\license_rtf.py` (standard library only). Run
  `.venv\Scripts\python.exe installer\license_rtf.py` after any change to
  `LICENSE`; `tests/test_installer_license_rtf.py` fails while the copy is
  stale.

Keep `yamabiko.nsi` UTF-8 with its BOM: makensis reads it as UTF-8 only
then, and the page text has non-ASCII characters.

## What gets bundled

PyInstaller onedir layout (6.x default):

- `yamabiko.exe` at the install root: the launcher.
- `_internal/`: everything else: Python 3.12 runtime, all pip deps,
  `ctranslate2` + cuDNN/cuBLAS DLLs (CUDA 12, sm_75+),
  `unidic-lite` + `sudachidict_core` data, the `yamabiko/` package (including
  the bundled BIZ UDPGothic, BIZ UDPMincho and Klee One fonts in
  `yamabiko/display/static/fonts/`, and the Settings preview photo in
  `yamabiko/overlay_app/assets/`, CC BY 2.0, see `NOTICE`),
  `default_config.yaml`, `assets/icon.ico`, `assets/icon.png`, `LICENSE`,
  `NOTICE`, `README.md`, `CHANGELOG.md`.
  (Torch is deliberately NOT bundled: the VAD runs on `onnxruntime`
  and faster-whisper uses ctranslate2, so neither needs torch.)
- `llama/` under `_internal/`: vendored `llama-server.exe` + CUDA
  DLLs, placed by `build.ps1` after PyInstaller runs.
- `Uninstall.exe` at the install root, written by NSIS.

## What does NOT get bundled
Model weights. The wizard downloads ~6.6 GB on first run:
- Qwen3.5-9B GGUF (~5.1 GB, Standard tier; see "8 GB VRAM tier" below)
- Kodama-1 speech model, CT2 (~1.5 GB)
- Silero VAD (~2 MB)

## Advanced: 8 GB VRAM tier

Yamabiko downloads the `Qwen3.5-9B-UD-Q3_K_XL.gguf` quant (~5.1 GB) by
default. The Standard tier prioritizes output quality. The Smaller tier uses
`Qwen3.5-9B-UD-IQ3_XXS.gguf` (~4 GB), reduces memory use, and usually responds
faster. Use Smaller on 8 GB cards and when Standard leaves too little VRAM for
the desktop and other applications.

The Smaller tier also flips `asr.faster_whisper.compute_type` from
`float16` to `int8_float16`, which runs Whisper with INT8 weights
and FP16 compute. Combined with the smaller Qwen3.5 quant, this is the
supported configuration for 8 GB cards.

First-run setup detects your graphics card's video memory with
`nvidia-smi` and selects the tier recommended for it on the **Quality**
step; a card with less than 8 GB gets no recommendation, and setup names
the 8 GB requirement. If the detection is wrong (or your rig is unusual),
pick the other tier there. Post-install, the same choice lives in Settings,
on the **Audio & model** tab under **Translation model**. Changing it there
downloads and validates the selected model first; the new tier takes
effect the next time Yamabiko starts.
Canceling or failing that download leaves the previously committed tier in
place.

For CLI-driven installs or when scripting a silent setup, you can
still write the choice directly to
`%LOCALAPPDATA%\Yamabiko\user_overrides.yaml` **before first run**:

```yaml
translation:
  quant_tier: small
asr:
  faster_whisper:
    compute_type: int8_float16
```

If you switch tiers after setup, Yamabiko downloads the selected quant into
the shared Hugging Face cache (`%LOCALAPPDATA%\Yamabiko\hf-cache\`), validates
it, and then persists the new tier. Already-downloaded files are reused.
