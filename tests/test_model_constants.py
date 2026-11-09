"""Single-source-of-truth guard: the translator pins must have ONE source.

Repo/file/revision were once defined independently in the wizard
(yamabiko.wizard_pages.downloads, the downloader) and the runtime resolver
(yamabiko.watchdog) -- if a quant filename changed in one but not the
other, the wizard would fetch one file while the runtime looked for
another and the translator would never start. Both now import from
yamabiko.models; these tests fail if either re-inlines a value that
drifts. The same guard covers the served alias: watchdog spawns
llama-server with `--alias` and the Translator addresses requests to
`translation.model`; if those drift, every chat completion 404s.
"""
from __future__ import annotations


def test_translator_pins_have_a_single_source():
    """Seeded defect: re-inline a tier dict (or a stale pin copy) in
    watchdog or downloads instead of importing models.TRANSLATOR_PINS ->
    the identity/equality below breaks -> this fails.

    The revision rides in each pin because the wizard download and the
    runtime resolver both probe sha-pinned snapshots (a sha-pinned
    download writes only snapshots/<sha>/, so wizard and runtime must
    resolve the identical revision or the cache probe misses)."""
    from yamabiko import models
    from yamabiko.watchdog import _TRANSLATOR_PINS as wd_pins
    from yamabiko.wizard_pages.downloads import _TRANSLATOR_PINS as dl_pins

    assert wd_pins is models.TRANSLATOR_PINS
    assert dl_pins is models.TRANSLATOR_PINS
    assert set(models.TRANSLATOR_PINS) == {"default", "small"}


def test_translator_alias_matches_watchdog_and_shipped_config():
    """The three copies of the served model name must agree: the
    yamabiko.models.TRANSLATOR_ALIAS constant, the `--alias` watchdog
    hands llama-server, and the `translation.model` name the Translator
    addresses every request to (from default_config.yaml). A mismatch is
    a 404 on every translation -- the pipeline starts, then silently
    produces nothing.

    Seeded defect: edit default_config.yaml translation.model to the old
    `shisa-v2.1-qwen3-8b` (or re-inline a literal in watchdog's
    _LLAMA_ALIAS) -> this fails.
    """
    import yaml

    from yamabiko import models, paths
    from yamabiko.watchdog import _LLAMA_ALIAS

    assert _LLAMA_ALIAS == models.TRANSLATOR_ALIAS
    cfg = yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    assert cfg["translation"]["model"] == models.TRANSLATOR_ALIAS


def test_translator_revisions_are_pinned_commit_shas():
    """Every translator tier pin must carry a pinned commit revision like
    WHISPER/SILERO, not float on upstream `main` -- an upstream re-quant
    push could otherwise hand the wizard and the runtime different
    weights, or invalidate a warmed multi-GB cache and force a surprise
    re-download at pipeline startup.

    Seeded defect: set one pin's revision to None (or "") -> this fails.
    """
    import re

    from yamabiko import models

    for tier, pin in models.TRANSLATOR_PINS.items():
        assert isinstance(pin.revision, str), tier
        assert re.fullmatch(r"[0-9a-f]{40}", pin.revision), (
            f"{tier} pin revision must be a full 40-char commit SHA; "
            f"got {pin.revision!r}"
        )


def test_preflight_hf_checks_cover_every_translator_pin():
    """The release gate (preflight HF_CHECKS) must verify the SAME
    translator files the runtime resolves, one entry per tier pin at its
    pinned revision -- a hardcoded copy there would re-introduce
    producer/consumer drift in the verification path itself.

    Seeded defect: drop the small-tier entry from preflight's HF_CHECKS
    generation (or pin it at None revision) -> this fails.
    """
    from yamabiko import models
    from scripts.preflight import HF_CHECKS

    for tier, pin in models.TRANSLATOR_PINS.items():
        entry = next(
            (t for t in HF_CHECKS
             if t[0] == pin.repo and pin.filename in t[2]), None,
        )
        assert entry is not None, (
            f"preflight HF_CHECKS is missing the {tier} tier pin "
            f"({pin.repo}/{pin.filename})"
        )
        assert entry[1] == pin.revision, (
            f"preflight must verify the exact pinned snapshot for the "
            f"{tier} tier (an unpinned entry floats with upstream `main`)"
        )


def test_silero_constants_have_a_single_source():
    """Same producer/consumer drift class for the Silero VAD repo+file: the
    wizard (downloads) and the runtime (silero_chunker) must agree, via
    yamabiko.models -- including the pinned revision (an unpinned silero
    floats with upstream `main`; wizard and runtime could resolve different
    snapshots).

    Seeded defect: re-inline a different silero repo/file/revision in
    downloads -> this fails.
    """
    from yamabiko import models
    from yamabiko.wizard_pages.downloads import (
        _SILERO_REPO, _SILERO_FILE, _SILERO_REVISION,
    )
    assert _SILERO_REPO == models.SILERO_REPO
    assert _SILERO_FILE == models.SILERO_FILE
    assert _SILERO_REVISION == models.SILERO_REVISION


def test_whisper_constants_have_a_single_source():
    """The wizard's whisper repo/revision/file-list copies must come from
    yamabiko.models, not local re-inlines -- the same producer/consumer
    drift class the shisa/silero tests pin.

    Seeded defect: re-inline a different whisper file tuple or whisper
    revision in downloads.py -> this fails.
    """
    from yamabiko import models
    from yamabiko.wizard_pages.downloads import (
        _WHISPER_REPO, _WHISPER_FILES, _WHISPER_REVISION,
    )
    assert _WHISPER_REPO == models.WHISPER_REPO
    assert tuple(_WHISPER_FILES) == tuple(models.WHISPER_FILES)
    assert _WHISPER_REVISION == models.WHISPER_REVISION


def test_preflight_hf_checks_use_shared_whisper_constants():
    """Preflight's HF_CHECKS must not hardcode the whisper repo + file list
    independently -- a drift-prone copy in the release gate itself opens
    the same producer/consumer mismatch that the shisa guard was written
    to prevent. The entry must also carry the PINNED revision, so the gate
    verifies the exact snapshot the wizard downloads and the runtime
    resolves (an unpinned whisper floats with upstream `main`).

    Seeded defect: re-inline a whisper filename in preflight's HF_CHECKS
    that differs from models.WHISPER_FILES, or revert the revision slot to
    None -> this fails.
    """
    from yamabiko import models
    from scripts.preflight import HF_CHECKS

    whisper = next(t for t in HF_CHECKS if t[0] == models.WHISPER_REPO)
    assert whisper[1] is not None
    assert whisper[1] == models.WHISPER_REVISION
    assert set(whisper[2]) == set(models.WHISPER_FILES)


def test_preflight_hf_checks_use_shared_silero_constants():
    """The silero entry must carry the shared repo/file AND the pinned
    revision, so the gate verifies the exact snapshot the wizard downloads
    and the runtime resolves.

    Seeded defect: revert preflight's silero entry to
    ("onnx-community/silero-vad", None, ["onnx/model.onnx"]) -> the
    revision assertion fails.
    """
    from yamabiko import models
    from scripts.preflight import HF_CHECKS

    silero = next(t for t in HF_CHECKS if t[0] == models.SILERO_REPO)
    assert silero[1] == models.SILERO_REVISION
    assert models.SILERO_FILE in silero[2]


def test_asr_cache_probe_files_match_wizard_whisper_download_set(monkeypatch):
    """The runtime's offline-cache probe (asr_client._whisper_cached) must
    query EXACTLY the files the wizard downloads (models.WHISPER_FILES). The
    probe gates local_files_only=True; probing a stale/partial set opens a
    partial-cache crash: a file added to the wizard's download set but not
    probed lets an incomplete cache pass the probe, and faster-whisper then
    hard-crashes at startup under local_files_only=True with no online
    fallback.

    Behavioral (not structural): captures the filenames the probe actually
    sends to try_to_load_from_cache and compares them to the download set.

    Seeded defect: loop over a re-inlined stale 4-element tuple in the probe
    (drop "preprocessor_config.json") -> the probed set is missing a wizard
    file -> the set equality fails -> red.
    """
    import huggingface_hub

    from yamabiko import models
    from yamabiko.asr.asr_client import _whisper_cached

    probed: list[str] = []

    def _capture(repo_id=None, filename=None, revision=None, **kw):
        probed.append(filename)
        return __file__  # pretend every file is cached

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _capture)
    assert _whisper_cached(models.WHISPER_REPO) is True
    assert set(probed) == set(models.WHISPER_FILES), (
        f"probe queried {sorted(probed)!r} but the wizard downloads "
        f"{sorted(models.WHISPER_FILES)!r} -- drift re-opens the "
        f"partial-cache offline crash"
    )


def test_whisper_revision_is_a_pinned_commit_sha():
    """WHISPER_REPO must carry a pinned commit revision like SILERO,
    not float on upstream `main` -- an upstream push could otherwise hand the
    wizard and the runtime different weights, or invalidate a warmed cache and
    force a surprise re-download/HEAD-check at pipeline startup (issue #59).

    Seeded defect: set models.WHISPER_REVISION = None (or "") -> this fails.
    """
    import re

    from yamabiko import models

    assert isinstance(models.WHISPER_REVISION, str)
    assert re.fullmatch(r"[0-9a-f]{40}", models.WHISPER_REVISION), (
        f"WHISPER_REVISION must be a full 40-char commit SHA; "
        f"got {models.WHISPER_REVISION!r}"
    )


def test_default_config_asr_model_matches_whisper_repo():
    """The shipped default_config.yaml `asr.faster_whisper.model` literal is a
    hand-duplicated mirror of models.WHISPER_REPO. build_asr() reads the YAML
    literal for both the model load AND the offline cache probe, while the
    wizard, preflight, and the real ASR test all source models.WHISPER_REPO.
    If the two drift, the wizard warms repo A while the runtime loads repo B:
    _whisper_cached(B) is false -> an offline hard-load failure, or a surprise
    ~1.6 GB re-download on a warmed install (issue #58, rule #4 class).

    Seeded defect: change the YAML to `Yamabiko-ASR/kodama1` (drop the
    hyphen) while leaving models.WHISPER_REPO -> this fails.
    """
    import yaml

    from yamabiko import models, paths

    cfg = yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    assert cfg["asr"]["faster_whisper"]["model"] == models.WHISPER_REPO


def test_default_config_decodes_kodama_1_without_prompt_and_with_its_no_speech_gate():
    """Kodama-1 was accepted, and its model card says it must be decoded,
    with no initial prompt and the no-speech-only segment gate at 0.95. The
    shipped default_config.yaml carries both next to the model literal. With
    the old punctuation prime, or with the gate off, it runs in a mode it was
    never accepted in: the paired gate alone misses its confident text over
    non-speech. Tying the values to the pinned repo makes a future pin change
    revisit them.

    Seeded defects: restore the prime as initial_prompt -> red; leave
    no_speech_only_threshold at null (#130's default) or 0.5 (#140) -> red.
    """
    import yaml

    from yamabiko import models, paths

    cfg = yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    whisper = cfg["asr"]["faster_whisper"]
    assert models.WHISPER_REPO == "Yamabiko-ASR/kodama-1"
    assert whisper["model"] == models.WHISPER_REPO
    assert whisper["initial_prompt"] == ""
    assert whisper["no_speech_only_threshold"] == 0.95
