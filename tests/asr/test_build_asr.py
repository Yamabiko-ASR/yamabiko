"""Unit tests for build_asr's offline-cache wiring.

The first-run wizard pre-downloads the whisper model into the pinned HF
cache. But the runtime constructed WhisperModel without local_files_only,
so faster-whisper called snapshot_download(local_files_only=False) on
EVERY startup -- an online revision re-check that fails on a flaky/absent
connection despite a fully-warmed cache. build_asr now loads offline when
the model is already cached.

These are unit tests: WhisperModel + try_to_load_from_cache are stubbed so
no real model loads and no network is touched. The real load path is covered
by the @pytest.mark.real tier (tests/asr/test_real_faster_whisper.py): its
module fixture builds via build_asr (so the offline branch executes against
the real library + cache when warmed), and
test_whisper_offline_load_constructs_and_warms_on_warmed_cache exercises
WhisperModel(..., local_files_only=True) explicitly.
"""
from __future__ import annotations

import huggingface_hub

from yamabiko.asr import asr_client
from yamabiko.asr.asr_client import build_asr

_CFG = {"faster_whisper": {"model": "repo/whisper-x", "compute_type": "float16", "device": "cuda"}}


def _stub_whisper(monkeypatch):
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None,
                     local_files_only=False, revision=None):
            captured["model"] = model
            captured["local_files_only"] = local_files_only
            captured["revision"] = revision

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _FakeWhisperModel,
    )
    return captured


def test_build_asr_loads_offline_when_model_cached(monkeypatch):
    """When the whisper weights are already in the HF cache, build_asr
    constructs WhisperModel with local_files_only=True so startup never
    contacts the Hub.

    Seeded defect: drop the local_files_only wiring ->
    captured value is False -> this fails.
    """
    captured = _stub_whisper(monkeypatch)
    # try_to_load_from_cache returns a real file path when cached.
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda **kw: __file__,
    )
    build_asr(_CFG)
    assert captured["local_files_only"] is True


def test_build_asr_allows_online_when_model_not_cached(monkeypatch):
    """When the weights are NOT cached (cold first run), build_asr must
    allow the online download (local_files_only=False) so the model can
    still be fetched.

    Seeded defect: hardcode local_files_only=True -> a cold run can never
    download -> this fails.
    """
    captured = _stub_whisper(monkeypatch)
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda **kw: None,
    )
    build_asr(_CFG)
    assert captured["local_files_only"] is False


def test_build_asr_online_when_cache_is_partial(monkeypatch):
    """A PARTIAL cache (model.bin present but a metadata file absent, e.g.
    a mid-download abort) must NOT trigger local_files_only=True --
    faster-whisper needs all 5 files, so an offline load would hard-crash
    at startup with no online fallback (audit verification: keying on
    model.bin alone made a partial cache fatal, worse than the old code).

    Seeded defect: probe only model.bin -> partial cache returns True ->
    this fails.
    """
    captured = _stub_whisper(monkeypatch)

    def _partial(repo_id=None, filename=None, **kw):
        # model.bin landed; the rest of the queue never finished.
        return __file__ if filename == "model.bin" else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _partial)
    build_asr(_CFG)
    assert captured["local_files_only"] is False


def test_build_asr_allows_online_when_cache_probe_errors(monkeypatch):
    """A probe failure must fall back to the safe online path, not crash
    startup."""
    captured = _stub_whisper(monkeypatch)

    def _boom(**kw):
        raise RuntimeError("hub probe blew up")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _boom)
    build_asr(_CFG)
    assert captured["local_files_only"] is False


def _stub_whisper_capturing_transcribe(monkeypatch):
    """Stub WhisperModel and capture the kwargs of the FIRST transcribe call,
    so we can assert decoder configuration actually reaches faster-whisper."""
    import numpy as np
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None,
                     local_files_only=False, revision=None):
            pass

        def transcribe(self, audio, **kw):
            captured.update(kw)
            return ([], None)  # no segments -> "" text; we only inspect kwargs

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _FakeWhisperModel,
    )
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)
    captured["_audio"] = np.zeros(16000, dtype=np.float32)
    return captured


def _cfg_for(model: str) -> dict:
    return {"faster_whisper": {
        "model": model, "compute_type": "float16", "device": "cuda",
    }}


def test_build_asr_loads_default_whisper_at_the_pinned_revision(monkeypatch):
    """build_asr must construct WhisperModel at the models.WHISPER_REVISION
    snapshot when the configured model is the DEFAULT repo, not float on
    upstream `main`. faster-whisper resolves the repo against the HF cache;
    pinning the revision is what makes the wizard-warmed snapshot the one the
    runtime loads offline (issue #59).

    Seeded defect: drop the revision wiring from the WhisperModel call
    -> captured["revision"] is None != the pinned SHA -> red.
    """
    from yamabiko.models import WHISPER_REPO, WHISPER_REVISION

    captured = _stub_whisper(monkeypatch)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)
    build_asr(_cfg_for(WHISPER_REPO))
    assert captured["revision"] == WHISPER_REVISION, (
        f"WhisperModel must load the default repo at the pinned revision; "
        f"got {captured.get('revision')!r}"
    )


def test_build_asr_omits_revision_pin_for_overridden_model(monkeypatch):
    """A user override of asr.faster_whisper.model to a DIFFERENT repo (e.g.
    anime-whisper) must NOT inherit the default commit pin:
    models.WHISPER_REVISION is a sha that exists only in the default repo
    (models.WHISPER_REPO), so passing it with any other repo is a guaranteed
    RevisionNotFoundError at startup.

    Seeded defect: pass `revision=WHISPER_REVISION` unconditionally in
    build_asr -> captured["revision"] is the pinned sha, not None -> red.
    """
    captured = _stub_whisper(monkeypatch)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)
    build_asr(_cfg_for("litagin/anime-whisper"))
    assert captured["revision"] is None, (
        f"an overridden model must load unpinned (revision=None); passing the "
        f"default-repo-only sha {captured.get('revision')!r} guarantees "
        f"RevisionNotFoundError on any other repo"
    )


def test_whisper_cache_probe_queries_default_repo_at_pinned_revision(monkeypatch):
    """The offline-cache probe (_whisper_cached) must ask the HF cache for the
    PINNED-revision snapshot when probing the DEFAULT repo. A bare
    (no-revision) probe defaults to `main`, so on a fresh install warmed by a
    revision-pinned wizard download (which writes only snapshots/<sha>/, no
    refs/main) the probe would miss the warm cache and force a surprise online
    HEAD-check / re-download (issue #59).

    Seeded defect: drop the revision from the try_to_load_from_cache call ->
    the captured revision is None -> red.
    """
    from yamabiko.asr.asr_client import _whisper_cached
    from yamabiko.models import WHISPER_REPO, WHISPER_REVISION

    seen_revisions = []

    def _capture(repo_id=None, filename=None, revision=None, **kw):
        seen_revisions.append(revision)
        return __file__  # pretend every file is cached

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _capture)
    assert _whisper_cached(WHISPER_REPO) is True
    assert seen_revisions, "probe never queried the cache"
    assert all(r == WHISPER_REVISION for r in seen_revisions), (
        f"every default-repo cache probe must use the pinned revision; "
        f"got {seen_revisions!r}"
    )


def test_whisper_cache_probe_omits_revision_for_overridden_model(monkeypatch):
    """Probing the cache for an OVERRIDDEN model must not use the default-repo
    revision pin: the sha does not exist in any other repo, so a pinned probe
    can never hit -- local_files_only stays False forever and a fully-warmed
    override cache still HEAD-checks the Hub on every startup.

    Seeded defect: pass revision=WHISPER_REVISION unconditionally in the
    probe -> captured revisions are the pinned sha, not None -> red.
    """
    from yamabiko.asr.asr_client import _whisper_cached

    seen_revisions = []

    def _capture(repo_id=None, filename=None, revision=None, **kw):
        seen_revisions.append(revision)
        return __file__  # pretend every file is cached

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _capture)
    assert _whisper_cached("litagin/anime-whisper") is True
    assert seen_revisions, "probe never queried the cache"
    assert all(r is None for r in seen_revisions), (
        f"an overridden model must be probed unpinned (revision=None); "
        f"got {seen_revisions!r}"
    )


def test_build_asr_passes_configured_beam_size_to_transcribe(monkeypatch):
    """asr.faster_whisper.beam_size flows through build_asr into the real
    WhisperModel.transcribe call, so it's a live A/B knob — not cosmetic.

    Seeded defect: hardcode beam_size=1 in FasterWhisperASR.transcribe ->
    the configured 8 never reaches transcribe ->
    captured["beam_size"] == 1 != 8 -> red.
    """
    captured = _stub_whisper_capturing_transcribe(monkeypatch)
    cfg = {"faster_whisper": {"model": "repo/whisper-x", "compute_type": "float16",
                              "device": "cuda", "beam_size": 8}}
    asr = build_asr(cfg)
    asr.transcribe(captured["_audio"], 16000)
    assert captured["beam_size"] == 8, (
        f"configured beam_size must reach transcribe; got {captured.get('beam_size')!r}"
    )


def test_build_asr_beam_size_defaults_to_5_when_unset(monkeypatch):
    """With no beam_size in config, build_asr uses faster-whisper's default of
    5 (not the old greedy 1), so a fresh install gets the accuracy default.

    Seeded defect: default to 1 in build_asr's w.get(...) -> captured is 1 -> red.
    """
    captured = _stub_whisper_capturing_transcribe(monkeypatch)
    asr = build_asr(_CFG)  # _CFG has no beam_size
    asr.transcribe(captured["_audio"], 16000)
    assert captured["beam_size"] == 5, (
        f"default beam_size must be 5; got {captured.get('beam_size')!r}"
    )


def test_faster_whisper_receives_wrapper_confidence_thresholds(monkeypatch):
    """The thresholds used by Yamabiko's wrapper gates must also be explicit
    decoder arguments, so both layers receive the same values instead of
    relying on faster-whisper's independently versioned defaults."""
    captured = _stub_whisper_capturing_transcribe(monkeypatch)
    asr = build_asr(_CFG)
    asr.transcribe(captured["_audio"], 16000)

    expected = {
        "no_speech_threshold": asr_client._NO_SPEECH_THRESHOLD,
        "log_prob_threshold": asr_client._LOGPROB_THRESHOLD,
        "compression_ratio_threshold": asr_client._COMPRESSION_RATIO_THRESHOLD,
    }
    assert {name: captured.get(name) for name in expected} == expected


def test_build_asr_empty_initial_prompt_disables_priming(monkeypatch):
    """asr.faster_whisper.initial_prompt = "" reaches transcribe as None (priming
    DISABLED), not as the literal empty string -- anime-whisper and other JA
    fine-tunes degrade WITH any initial_prompt, so the A/B config must be able to
    turn it fully off.

    Seeded defect: pass self._initial_prompt straight through instead of
    `self._initial_prompt or None` -> transcribe receives "" not None -> red.
    """
    captured = _stub_whisper_capturing_transcribe(monkeypatch)
    cfg = {"faster_whisper": {"model": "repo/whisper-x", "compute_type": "float16",
                              "device": "cuda", "initial_prompt": ""}}
    asr = build_asr(cfg)
    asr.transcribe(captured["_audio"], 16000)
    assert captured["initial_prompt"] is None, (
        f'empty initial_prompt must disable priming (None); got {captured.get("initial_prompt")!r}'
    )


def test_build_asr_passes_configured_initial_prompt(monkeypatch):
    """A non-empty configured initial_prompt reaches transcribe verbatim, so a
    model that wants a custom prime gets it.

    Seeded defect: hardcode the prompt in transcribe -> the custom value never
    arrives -> red.
    """
    captured = _stub_whisper_capturing_transcribe(monkeypatch)
    cfg = {"faster_whisper": {"model": "repo/whisper-x", "compute_type": "float16",
                              "device": "cuda", "initial_prompt": "カスタム。"}}
    asr = build_asr(cfg)
    asr.transcribe(captured["_audio"], 16000)
    assert captured["initial_prompt"] == "カスタム。", (
        f'configured initial_prompt must reach transcribe; got {captured.get("initial_prompt")!r}'
    )


def test_build_asr_wires_no_speech_only_threshold_to_the_segment_gate(monkeypatch):
    """asr.faster_whisper.no_speech_only_threshold flows through build_asr into
    the transcribe gate: with 0.95 a confident phantom (no_speech 0.97,
    avg_logprob -0.3, which the paired gate keeps) is dropped; with the key
    absent it is kept. End to end, config leaf -> kept text.

    Seeded defects: drop the kwarg from build_asr, or read it off ``cfg``
    instead of ``cfg["faster_whisper"]`` -> 0.95 never reaches the gate ->
    the phantom is kept -> red.
    """
    from types import SimpleNamespace

    import numpy as np

    phantom = SimpleNamespace(text="おはようございます。", no_speech_prob=0.97,
                              avg_logprob=-0.3, compression_ratio=1.0)
    clean = SimpleNamespace(text="本物のセリフ", no_speech_prob=0.02,
                            avg_logprob=-0.2, compression_ratio=1.0)

    class _FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None,
                     local_files_only=False, revision=None):
            pass

        def transcribe(self, audio, **kw):
            return iter([phantom, clean]), None

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _FakeWhisperModel,
    )
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)
    audio = np.zeros(16000, dtype=np.float32)

    gated = build_asr({"faster_whisper": {**_CFG["faster_whisper"],
                                          "no_speech_only_threshold": 0.95}})
    assert gated.transcribe(audio, 16000).text == "本物のセリフ", (
        "configured no_speech_only_threshold must drop the confident phantom"
    )
    ungated = build_asr(_CFG)  # _CFG has no no_speech_only_threshold
    assert ungated.transcribe(audio, 16000).text == "おはようございます。本物のセリフ", (
        "with the key absent the gate must stay off and keep the phantom"
    )


def test_user_override_of_1_keeps_segments_the_no_speech_only_gate_would_drop(
    tmp_path, monkeypatch,
):
    """A user turns the no-speech-only gate off with
    ``asr.faster_whisper.no_speech_only_threshold: 1.0`` in user_overrides.yaml.
    ``null`` cannot do it once the shipped default is a number: config.py
    restores the default over a null override. 1.0 must load, build, and keep
    a segment at no_speech_prob 1.0 (which 0.95 drops) whose avg_logprob the
    paired gate accepts. 1.0 sits on the boundary, so it also pins the gate's
    strict ``>``.

    Seeded defects: keep the config rule's maximum exclusive ->
    load_full_config raises ConfigValidationError -> red; keep the
    constructor's ``< 1.0`` -> ValueError -> red; compare with ``>=`` -> the
    segment is dropped -> red.
    """
    from types import SimpleNamespace

    import numpy as np

    from yamabiko import config as config_mod
    from yamabiko import paths

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    paths.ensure_dirs()
    paths.user_overrides_path().write_text(
        "asr:\n  faster_whisper:\n    no_speech_only_threshold: 1.0\n",
        encoding="utf-8",
    )
    cfg = config_mod.load_full_config()

    quiet = SimpleNamespace(text="はい", no_speech_prob=1.0,
                            avg_logprob=-0.3, compression_ratio=1.0)

    class _FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None,
                     local_files_only=False, revision=None):
            pass

        def transcribe(self, audio, **kw):
            return iter([quiet]), None

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _FakeWhisperModel,
    )
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)

    asr = build_asr(cfg["asr"])
    assert asr.transcribe(np.zeros(16000, dtype=np.float32), 16000).text == "はい", (
        "no_speech_only_threshold 1.0, like null, must apply only the paired "
        "gate and keep this segment"
    )


def _stub_whisper_capturing_audio(monkeypatch):
    """Stub WhisperModel and capture the LENGTH of the audio array handed to
    the first transcribe call, so we can assert the boundary pad reached the
    model through build_asr's wiring."""
    import numpy as np
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None,
                     local_files_only=False, revision=None):
            pass

        def transcribe(self, audio, **kw):
            captured["audio_len"] = len(audio)
            return ([], None)  # no segments -> "" text; we only inspect length

    monkeypatch.setattr(
        asr_client.faster_whisper_audio_compat,
        "import_whisper_model",
        lambda: _FakeWhisperModel,
    )
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)
    captured["_audio"] = np.zeros(16000, dtype=np.float32)  # 1.0 s clip
    return captured


def test_build_asr_wires_configured_boundary_pad_ms(monkeypatch):
    """asr.boundary_pad_ms flows through build_asr into FasterWhisperASR and
    onto the real transcribe call: a 1.0 s clip with boundary_pad_ms=500 (8000
    samples @ 16 kHz per side) reaches the model as 16000 + 2*8000 samples. End
    to end, config leaf -> model input.

    Seeded defect: drop the boundary_pad_ms wiring from build_asr -> the pad
    defaults to 0 (or the ctor ignores it) -> the model gets 16000 -> red.
    """
    captured = _stub_whisper_capturing_audio(monkeypatch)
    cfg = {"boundary_pad_ms": 500,
           "faster_whisper": {"model": "repo/whisper-x",
                              "compute_type": "float16", "device": "cuda"}}
    asr = build_asr(cfg)
    asr.transcribe(captured["_audio"], 16000)
    assert captured["audio_len"] == 16000 + 2 * 8000, (
        f"configured boundary_pad_ms must reach the model; got "
        f"{captured.get('audio_len')!r} samples"
    )


def test_build_asr_boundary_pad_defaults_to_500_when_unset(monkeypatch):
    """With no boundary_pad_ms in config, build_asr applies the shipped 500 ms
    default (the measured CER win is on by default), so a fresh install gets
    padding without needing the key.

    Seeded defect: default to 0 in build_asr's w-less get(...) -> the model
    gets an unpadded 16000 -> red.
    """
    captured = _stub_whisper_capturing_audio(monkeypatch)
    asr = build_asr(_CFG)  # _CFG has no boundary_pad_ms
    asr.transcribe(captured["_audio"], 16000)
    assert captured["audio_len"] == 16000 + 2 * 8000, (
        f"default boundary_pad_ms must be 500 ms; got "
        f"{captured.get('audio_len')!r} samples"
    )


def test_build_asr_boundary_pad_zero_disables_padding(monkeypatch):
    """asr.boundary_pad_ms=0 disables padding end to end: the model receives
    the clip at its original length (byte-identical to pre-padding behavior).

    Seeded defect: ignore the 0 and force a nonzero default -> the model gets a
    padded length -> red.
    """
    captured = _stub_whisper_capturing_audio(monkeypatch)
    cfg = {"boundary_pad_ms": 0,
           "faster_whisper": {"model": "repo/whisper-x",
                              "compute_type": "float16", "device": "cuda"}}
    asr = build_asr(cfg)
    asr.transcribe(captured["_audio"], 16000)
    assert captured["audio_len"] == 16000, (
        f"boundary_pad_ms=0 must pass the clip unpadded; got "
        f"{captured.get('audio_len')!r} samples"
    )
