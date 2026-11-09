"""Real-tier probes: every model loader against a CJK (non-ASCII) path.

A JP-locale user's ``%LOCALAPPDATA%`` — and therefore the warmed HF
cache and every model file Yamabiko loads — sits under a CJK path
(``C:\\Users\\山彦\\AppData\\Local\\Yamabiko\\hf-cache\\...``).
This is the modal case for our target audience, yet not one loader was
ever exercised against a non-ASCII path before this test suite was added.

Each test below takes a REAL artifact from the warmed HF cache,
hardlinks it into a freshly-created CJK directory (``os.link`` — same
volume as the cache, so no multi-GB copy; ``shutil.copy2`` fallback for
the rare cross-volume case), and loads it through the REAL production
loader / library. No mocks on the subject: the subject IS
huggingface_hub / onnxruntime / tokenizers / faster-whisper /
llama-server resolving a CJK path.

- hub / onnxruntime / tokenizers are pure Python/Rust and proved
  Unicode-clean.
- llama-server (vendored C++ binary) was the known risk: upstream
  llama.cpp has an open non-ASCII-path bug (ggml-org/llama.cpp#18571,
  fix PR #21583 unmerged as of 2026-06) — its ``ggml_fopen`` mangles a
  non-ASCII ``--model`` path. The GGUF probe below caught it; the fix in
  ``watchdog._build_llama_cmd`` / ``_llama_cwd`` passes the ASCII
  basename as ``--model`` and sets the child's cwd to the model's parent
  dir (the broken char->wchar cast is harmless for ASCII argv bytes, and
  CreateProcessW sets the Unicode cwd). A GetShortPathNameW 8.3 fallback
  was evaluated but rejected — 8.3 returns a still-non-ASCII path for a
  CJK directory on this system. The llama probe is regression armor:
  revert the watchdog fix and it goes red again.

Skipped cleanly (never silently) when the relevant artifact isn't in
the warmed cache, or (for GPU/llama probes) when the GPU / vendored
binary is absent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

# Pin HF_HOME -> the warmed Yamabiko cache BEFORE any huggingface_hub
# import, exactly as production does. huggingface_hub captures
# HF_HUB_CACHE into a module constant at import time, so this import
# must come before any hub import for the cache lookups below to
# resolve the warmed artifacts (same idiom as test_real_onnxruntime.py
# / test_real_faster_whisper.py).
from yamabiko import hf_env  # noqa: F401

pytestmark = pytest.mark.real

# A CJK directory segment a real JP user's path would contain.
_CJK_SEGMENT = "日本語パス検査"

# Bind llama-server in the GGUF probe to a non-default port so it never
# collides with a dev llama-server on its default 8090.
_LLAMA_TEST_PORT = 18097


def _hardlink_or_copy(src: Path, dst: Path) -> Path:
    """Hardlink src -> dst on the same volume; fall back to a real copy.

    The warmed HF cache lives on the same volume as the OS temp dir on
    this box, so os.link is a metadata-only operation even for the
    4-5.5 GB GGUF and the ~1.6 GB CT2 snapshot — never copy those.

    HF cache files are SYMLINKS into ``blobs/``. ``os.link`` on a
    Windows symlink hardlinks the *reparse point*, not its target —
    the result is a dangling entry that ``listdir`` shows but
    ``exists()`` denies. So resolve the symlink to the real blob first
    (``os.path.realpath``) and hardlink that. Falls back to a
    symlink-following copy on a cross-volume / no-hardlink filesystem.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    real_src = os.path.realpath(src)  # follow the HF cache symlink to the blob
    try:
        os.link(real_src, dst)
    except OSError:
        # Cross-volume or filesystem without hardlink support.
        shutil.copy2(real_src, dst)
    return dst


def _cached_path(repo: str, filename: str, revision: str | None = None) -> Path | None:
    """Resolve a cached file to its on-disk path, or None if not cached.

    Pure cache lookup — never triggers a download.
    """
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    result = try_to_load_from_cache(repo, filename, revision=revision)
    if result is None or result is _CACHED_NO_EXIST:
        return None
    p = Path(str(result))
    return p if p.is_file() else None


def _cuda_available() -> bool:
    """Best-effort NVIDIA GPU detect without pulling torch in."""
    from yamabiko.gpu import detect_vram_mib

    return detect_vram_mib() is not None


# ---------------------------------------------------------------------------
# 1. huggingface_hub: cache hit under a CJK HF_HOME
# ---------------------------------------------------------------------------

def test_hf_cache_hit_under_cjk_hf_home(tmp_path, monkeypatch):
    """hf_hub_download writes to, and re-resolves offline from, a CJK cache.

    Repoints the HF hub cache at a CJK directory (the real-user
    condition — HF derives the hub cache as ``HF_HOME/hub`` and a JP
    user's HF_HOME is under ``C:\\Users\\<CJK>\\...``), downloads the
    tiny (~2 MB) silero ONNX into that fresh cache, then re-fetches with
    local_files_only=True and asserts the SAME on-disk path comes back,
    the file exists, and the CJK segment is in the resolved path.

    huggingface_hub reads ``constants.HF_HUB_CACHE`` at call time when
    ``cache_dir=None``; ``yamabiko.hf_env`` already pinned that constant
    to the warmed Yamabiko cache at import, so an env tweak alone is a
    no-op here — we monkeypatch the constant (and set HF_HOME for any
    code that re-derives it) to the CJK dir, which is precisely what a
    CJK HF_HOME does in production.

    Seeded defect: if huggingface_hub's cache layer mangled a non-ASCII
    cache dir (or our code did), the local_files_only=True re-fetch would
    raise LocalEntryNotFoundError (path written != path looked up), and
    this fails red. A second fetch returning a DIFFERENT path than the
    first also fails the equality assertion.
    """
    import huggingface_hub.constants as hf_constants
    from yamabiko import models

    cjk_hf_home = tmp_path / _CJK_SEGMENT / "hf-cache"
    cjk_hub_cache = cjk_hf_home / "hub"
    cjk_hub_cache.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(cjk_hf_home))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cjk_hub_cache))

    import httpx
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    # Online fetch into the CJK cache. Only network/Hub unavailability
    # may skip; a CJK-path break in the download itself fails red.
    try:
        first = hf_hub_download(
            models.SILERO_REPO, models.SILERO_FILE,
            revision=models.SILERO_REVISION,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError,
            OfflineModeIsEnabled, httpx.HTTPError) as exc:
        pytest.skip(f"HF Hub unreachable (env/network): {exc!r}")

    # Offline re-resolution from the SAME CJK cache — the subject.
    second = hf_hub_download(
        models.SILERO_REPO, models.SILERO_FILE,
        revision=models.SILERO_REVISION, local_files_only=True,
    )

    assert first == second, (
        f"offline re-resolution returned a different path than the "
        f"download:\n  download={first!r}\n  reresolve={second!r}\n"
        f"the CJK HF_HOME did not round-trip"
    )
    assert Path(second).is_file(), f"resolved CJK-cache path is not a file: {second}"
    assert _CJK_SEGMENT in str(second), (
        f"resolved path {second!r} does not contain the CJK segment "
        f"{_CJK_SEGMENT!r} — HF_HOME override was not honoured"
    )


# ---------------------------------------------------------------------------
# 2. onnxruntime: load silero from a CJK path
# ---------------------------------------------------------------------------

def test_onnxruntime_loads_silero_from_cjk_path(tmp_path):
    """ort.InferenceSession constructs from a silero ONNX at a CJK path.

    Hardlinks the cached silero model.onnx into a CJK directory and
    builds a real CPU InferenceSession from it. onnxruntime takes a
    str path; on Windows it must handle a wide (UTF-16) path.

    Seeded defect: were onnxruntime (or a future wrapper of ours) to
    pass the path through a cp1252/ascii byte encode before handing it
    to the C++ session, construction would raise on the CJK directory
    and this fails red. Verified positive by asserting the session
    actually exposes its input contract.
    """
    from yamabiko import models

    src = _cached_path(
        models.SILERO_REPO, models.SILERO_FILE, revision=models.SILERO_REVISION,
    )
    if src is None:
        pytest.skip("silero-vad ONNX not cached at the pinned revision")

    import onnxruntime as ort

    cjk_path = _hardlink_or_copy(src, tmp_path / _CJK_SEGMENT / "model.onnx")

    sess = ort.InferenceSession(
        str(cjk_path), providers=["CPUExecutionProvider"],
    )
    inputs = sess.get_inputs()
    assert inputs, "onnxruntime session from CJK path exposed no inputs"


# ---------------------------------------------------------------------------
# 3. tokenizers: load the whisper tokenizer from a CJK path
# ---------------------------------------------------------------------------

def test_tokenizers_loads_from_cjk_path(tmp_path):
    """Tokenizer.from_file loads the whisper tokenizer.json at a CJK path
    and encodes Japanese text.

    Hardlinks the cached whisper tokenizer.json into a CJK directory and
    loads it via the production API (Tokenizer.from_file, the same call
    shape faster-whisper uses). The Rust tokenizers lib takes the path
    as a str.

    Seeded defect: a non-ASCII path mishandled at the Rust FFI boundary
    would raise on from_file at the CJK directory; this fails red.
    Encoding 'こんにちは' and asserting non-empty ids confirms the
    loaded tokenizer is functional (positive evidence, not just
    no-throw).
    """
    from yamabiko import models

    src = _cached_path(
        models.WHISPER_REPO, "tokenizer.json",
        revision=models.WHISPER_REVISION,
    )
    if src is None:
        pytest.skip("whisper tokenizer.json not cached at the pinned revision")

    from tokenizers import Tokenizer

    cjk_path = _hardlink_or_copy(src, tmp_path / _CJK_SEGMENT / "tokenizer.json")

    tok = Tokenizer.from_file(str(cjk_path))
    ids = tok.encode("こんにちは").ids
    assert ids, "tokenizer loaded from CJK path returned no token ids for Japanese input"


# ---------------------------------------------------------------------------
# 4. faster-whisper: load the CT2 model from a CJK directory (GPU)
# ---------------------------------------------------------------------------

def test_faster_whisper_loads_from_cjk_path(tmp_path):
    """WhisperModel constructs from a CT2 snapshot at a CJK directory.

    Hardlinks the full faster-whisper CT2 snapshot (model.bin + the
    config/vocab/tokenizer JSONs) into a CJK directory and constructs a
    real CUDA WhisperModel pointed at that directory. CTranslate2 reads
    the model dir via a C++ path on Windows.

    Seeded defect: CTranslate2 mishandling a non-ASCII model directory
    would raise at construction on the CJK path; this fails red.
    Skipped without a GPU (CUDA WhisperModel needs one) or without the
    cached snapshot.
    """
    from yamabiko import models

    if not _cuda_available():
        pytest.skip("no NVIDIA GPU detected — CUDA WhisperModel needs one")

    # Resolve the cached snapshot dir offline (no download), at the pinned
    # revision the wizard warms (it writes no refs/main).
    from huggingface_hub import snapshot_download

    try:
        snap = Path(snapshot_download(
            models.WHISPER_REPO, revision=models.WHISPER_REVISION,
            local_files_only=True,
        ))
    except Exception as exc:
        pytest.skip(f"faster-whisper CT2 snapshot not cached: {exc!r}")

    if not (snap / "model.bin").is_file():
        pytest.skip(f"CT2 model.bin missing from cached snapshot {snap}")

    cjk_dir = tmp_path / _CJK_SEGMENT / "ct2-model"
    cjk_dir.mkdir(parents=True)
    # Hardlink every file faster-whisper needs (model.bin is the ~1.6 GB
    # blob — hardlink, never copy).
    for f in snap.iterdir():
        if f.is_file():
            _hardlink_or_copy(f, cjk_dir / f.name)

    from yamabiko.asr.faster_whisper_audio_compat import import_whisper_model

    WhisperModel = import_whisper_model()

    model = WhisperModel(
        str(cjk_dir), device="cuda", compute_type="float16",
    )
    # Positive evidence: the model exposes its feature extractor (CT2
    # actually parsed the model dir, not just stored the path string).
    assert model.feature_extractor is not None, (
        "WhisperModel from CJK dir has no feature_extractor — the model "
        "directory was not actually loaded"
    )


# ---------------------------------------------------------------------------
# 5. llama-server: load the shisa GGUF from a CJK path — THE risk probe
# ---------------------------------------------------------------------------

def test_llama_server_loads_gguf_from_cjk_path(tmp_path):
    """The vendored llama-server loads the shisa GGUF from a CJK path and
    serves /health, via the PRODUCTION spawn pair (#26 fix).

    The upstream llama.cpp ``ggml_fopen`` corrupts a non-ASCII ``--model``
    path (ggml-org/llama.cpp#18571, fix PR #21583 unmerged as of 2026-06),
    so a JP user whose model lives under a CJK ``%LOCALAPPDATA%`` couldn't
    translate. The fix in ``watchdog`` passes the ASCII basename as
    ``--model`` and spawns with cwd=<model parent>. This probe exercises
    exactly that production pair -- ``wd._build_llama_cmd(target, port)``
    + ``wd._llama_cwd(target)`` -- so it can't drift from what
    ``_make_real_spawn_llama._spawn`` does.

    Hardlinks the cached GGUF into a CJK directory, spawns llama-server
    on a non-default port (_LLAMA_TEST_PORT, to avoid colliding with a
    dev server on 8090), and polls /health until it returns 200 within
    ~120 s. /health 200 means the GGUF actually loaded into VRAM.

    Seeded defect: revert the watchdog fix to pass an absolute non-ASCII
    ``--model`` path -> the probe fails with "EXITED ... failed to open
    GGUF file".

    The process is always killed in finally.
    """
    import yamabiko.watchdog as wd
    from yamabiko import models

    # Resolve the real translator GGUF via the production resolver so the
    # quant-tier selection matches what the runtime would load. Falls
    # back to a direct cache lookup of the default quant if the resolver
    # returns None (e.g. config not warmed in this env).
    src = wd._resolve_translator_model_path()
    if src is None:
        pin = models.TRANSLATOR_PINS["default"]
        src = _cached_path(pin.repo, pin.filename)
    if src is None:
        pytest.skip(
            "translator GGUF not in the HF cache — run the wizard or "
            "preflight first"
        )

    port = _LLAMA_TEST_PORT
    cjk_gguf = _hardlink_or_copy(
        Path(src), tmp_path / _CJK_SEGMENT / Path(src).name,
    )

    # Spawn via the PRODUCTION pair: the same argv builder AND cwd helper
    # that _make_real_spawn_llama._spawn uses, so this probe verifies the
    # actual shipped behaviour (cmd carries the ASCII basename; cwd is the
    # CJK parent dir). Drift-proof by construction.
    cmd = wd._build_llama_cmd(cjk_gguf, port=port)
    cwd = wd._llama_cwd(cjk_gguf)
    if not Path(cmd[0]).is_file():
        pytest.skip(
            f"vendored llama-server.exe missing at {cmd[0]} — "
            f"build it (third_party/llama.cpp) first"
        )

    import httpx

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 120.0
    healthy = False
    try:
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    # Process exited before becoming healthy — a hard
                    # finding. The process has already exited, but read
                    # via communicate(timeout=...) rather than a bare
                    # stdout.read(): if a grandchild inherited the pipe
                    # write-end, a plain read() would block forever
                    # (same safe idiom as the "never healthy" branch).
                    try:
                        out = proc.communicate(timeout=10)[0] or ""
                    except subprocess.TimeoutExpired:
                        out = "<output read timed out>"
                    pytest.fail(
                        f"llama-server EXITED (code={proc.returncode}) before "
                        f"serving /health while loading a GGUF at a CJK path "
                        f"{cjk_gguf!r}.\n  argv: {cmd}\n"
                        f"  --- llama stdout/stderr ---\n{out}"
                    )
                try:
                    resp = client.get(health_url)
                    if resp.status_code == 200:
                        healthy = True
                        break
                except httpx.HTTPError:
                    pass  # server not up yet
                time.sleep(1.0)

        if not healthy:
            # Never got healthy within budget — also a finding. Drain
            # whatever output we can without blocking forever.
            proc.terminate()
            try:
                out = proc.communicate(timeout=10)[0] or ""
            except subprocess.TimeoutExpired:
                out = "<llama-server did not exit after terminate()>"
            pytest.fail(
                f"llama-server never served /health within 120 s while "
                f"loading a GGUF at a CJK path {cjk_gguf!r}.\n  argv: {cmd}\n"
                f"  --- llama stdout/stderr ---\n{out}"
            )

        # Reaching here means /health returned 200 within budget: the
        # vendored llama-server loaded the GGUF from the CJK path. Both
        # failure paths above call pytest.fail(), so no explicit success
        # assertion is needed here (an `assert healthy` would be
        # unreachable-on-failure dead code).
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
