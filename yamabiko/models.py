"""Single source of truth for ALL HuggingFace repo / file / revision pins
the first-run wizard downloads, the runtime resolves, and the preflight
release gate verifies.

The translator pins are consumed by ``yamabiko/wizard_pages/downloads.py``
(the downloader), ``yamabiko/watchdog.py`` (the runtime resolver), and
``yamabiko/overlay_app/model_fetch.py`` (Settings tier switch); the
whisper pins are read by downloads.py and scripts/preflight.py.
Keeping every pin here, imported by all consumers, prevents a
producer/consumer filename mismatch where the wizard fetches one file
while the runtime looks for another and the translator never starts.
"""
from __future__ import annotations

from typing import NamedTuple


class TranslatorPin(NamedTuple):
    """One quant tier of the translator: HF repo + GGUF filename + commit
    + the GGUF's SHA-256.

    ``repo``/``revision`` ride along with the filename because the two
    tiers are allowed to live in DIFFERENT repos (during the 2026-07-12
    shisa -> Qwen3.5 swap the small tier stayed on shisa until its own
    quant passed a paired bench -- a tier only moves behind its own
    measurement, so a future partial swap needs per-tier pins again).
    ``sha256`` is the LFS object id at ``revision``; the downloader checks
    the landed file against it (``wizard_pages.downloads._HFDownloader``).
    """

    repo: str
    filename: str
    revision: str
    sha256: str


# The llama-server model alias. watchdog.py spawns the server with
# ``--alias TRANSLATOR_ALIAS`` and the Translator addresses requests to
# ``translation.model`` from default_config.yaml -- those two strings MUST
# be equal or every chat completion 404s (covered by a producer/consumer
# test in tests/test_model_constants.py).
TRANSLATOR_ALIAS = "qwen3.5-9b"

# Qwen3.5-9B, dahara1 Japanese-imatrix quants (Apache 2.0). Chosen over
# shisa-v2.1 on the 3-round GOLD bench 2026-07-12: clean input +7.5 RIGHT,
# ASR-path +3.1 RIGHT / -4.6 WRONG (paired sign test p=0.0034), and 39%
# fewer hallucination tags on garbled input. Bench + protocol:
# docs/_internal/BENCH-TRANSLATOR-SHORT-2026-07-12.md.
_QWEN_REPO = "dahara1/Qwen3.5-9B-UD-japanese-imatrix"
# Pinned commit (current `main` at pin time, verified via the HF API on
# 2026-07-12). Same rationale as WHISPER_REVISION / SILERO_REVISION: an
# upstream re-quant push to `main` could hand the wizard and the runtime
# different weights, or invalidate the warmed multi-GB cache and force a
# surprise re-download at pipeline startup. A sha-pinned download writes
# only ``snapshots/<sha>/`` (no ``refs/main``), so every cache probe MUST
# pass this revision or it defaults to `main` and misses a warm cache --
# see ``_repo_files_cached``.
_QWEN_REVISION = "db1f3ee98c2c1aa08ccad921871b82698bf4a73c"

# ``translation.quant_tier`` selects the pin: ``default`` for cards with
# 10 GB or more (~5.1 GB file), ``small`` for 8 GB cards (~4 GB file).
# The small tier passed its own paired blind bench vs the previous shisa
# Q3_K_M (2026-07-12: 80.3 vs 70.9 RIGHT, hallucination tags 11 vs 48),
# so BOTH tiers pin Qwen3.5 and shisa-v2.1 is fully retired.
TRANSLATOR_PINS: dict[str, TranslatorPin] = {
    "default": TranslatorPin(
        _QWEN_REPO, "Qwen3.5-9B-UD-Q3_K_XL.gguf", _QWEN_REVISION,
        "fb590f234bf6dd33e840269fc0701b2d73e17551ae7887a924a7241659793a5a",
    ),
    "small": TranslatorPin(
        _QWEN_REPO, "Qwen3.5-9B-UD-IQ3_XXS.gguf", _QWEN_REVISION,
        "b733348b46402e2f66b0048f2c9fc69d55758d20cc920fdc542f841a6d115906",
    ),
}


def translator_pin(tier: str) -> TranslatorPin:
    """The pin for ``tier``, falling back to the default tier.

    Unknown tiers fall back (never raise): a malformed
    ``user_overrides.yaml`` must degrade to the default download, not
    brick the wizard or the runtime resolver. Callers that want to warn
    on unknown values check membership themselves.
    """
    return TRANSLATOR_PINS.get(tier, TRANSLATOR_PINS["default"])

# faster-whisper CT2 weights: Kodama-1, Yamabiko's own Japanese speech model
# (CC BY-SA 4.0; licence and attributions in NOTICE). It is
# efwkjn/whisper-ja-760M (Apache-2.0; a Japanese fine-tune of
# openai/whisper-large-v3-turbo) with its decoder fine-tuned, converted to
# CTranslate2. It replaced efwkjn/faster-whisper-ja-760M, the base's own CT2
# export: on two development sets scored against subtitle references, it
# recognized more interjections (+9.3 / +8.6 points) and key words
# (+6.8 / +6.7), with fewer output windows matching no subtitle cue
# (-17 / -15 per hour), at a subtitle CER cost of +0.7 / +0.3 points. It
# is only run the way it was accepted: no initial prompt and the
# no-speech-only segment gate at 0.95 (default_config.yaml
# asr.faster_whisper.initial_prompt and no_speech_only_threshold, tied to
# this repo by tests/test_model_constants.py). Same five CT2 files and
# pruned vocabulary as the base (CTranslate2 >= 4.7.1; the shipped pin is
# >= 4.8.3), so the cache probe and offline load path are unchanged.
# Yamabiko only ever ingests Japanese (language pinned "ja"). The wizard
# downloads all five (wizard_pages/downloads.py); the runtime needs every
# one present before the offline local_files_only=True load
# (asr/asr_client.py::_whisper_cached).
WHISPER_REPO = "Yamabiko-ASR/kodama-1"
WHISPER_FILES = (
    "model.bin", "config.json", "tokenizer.json",
    "vocabulary.json", "preprocessor_config.json",
)
# Pinned commit of Yamabiko-ASR/kodama-1 (2026-11-02). It differs from the
# first upload (52b2866) only in the model card; model.bin is the same bytes.
# Without a revision the wizard and the runtime both float with `main`, so a
# push could hand them different weights, or invalidate the warmed cache and
# force a surprise re-download/HEAD-check at pipeline startup (see
# SILERO_REVISION for the full rationale). Threaded through the wizard
# download (downloads.py), the offline cache probe + WhisperModel load
# (asr/asr_client.py), and the preflight gate (scripts/preflight.py).
# faster-whisper's WhisperModel accepts `revision=`, so the pin reaches the
# actual load -- not just the download -- exactly like the silero pattern.
WHISPER_REVISION = "b077a2add507bf76e8f3b882d97769417ac547f5"
# SHA-256 of model.bin at WHISPER_REVISION (its LFS object id). The other
# WHISPER_FILES are small non-LFS text files that hub stores under their git
# blob ids, so only model.bin carries a pinned digest.
WHISPER_MODEL_BIN_SHA256 = (
    "c05297893d65f5cc661514a792d882f0e1b2170bcd47753c59b3f32b18890f11"
)

# Silero VAD ONNX -- same producer/consumer pair: the wizard downloads it
# (wizard_pages/downloads.py) and the runtime loads it
# (vad/silero_chunker.py). Kept here so they can't drift.
SILERO_REPO = "onnx-community/silero-vad"
SILERO_FILE = "onnx/model.onnx"
# Pinned community Silero VAD v5-era ONNX graph. Without a revision the
# wizard and runtime both float with upstream `main`; the explicit artifact
# digest also fails closed if the pinned cache entry is stale or corrupt.
SILERO_REVISION = "e71cae966052b992a7eca6b17738916ce0eca4ec"
SILERO_SHA256 = "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808"

# Exact byte sizes at the pinned revisions, beside the digest pins: the
# wizard shows the download as "3.5 of 6.6 GB" with a time left before any
# byte arrives, and one size per model row. Read from the HF API
# (HfApi.get_paths_info at each pinned revision, 2026-11-01; the whisper rows
# re-read at Kodama-1's revision on 2026-11-02, byte-identical to the
# previous pin's); a pin change without its size goes red in
# tests/test_real_hf_signature.py::
# test_all_wizard_hf_repos_exist_with_expected_files.
_PINNED_SIZES: dict[tuple[str, str, str], int] = {
    (TRANSLATOR_PINS["default"].repo, TRANSLATOR_PINS["default"].filename,
     TRANSLATOR_PINS["default"].revision): 5_088_437_312,
    (TRANSLATOR_PINS["small"].repo, TRANSLATOR_PINS["small"].filename,
     TRANSLATOR_PINS["small"].revision): 4_050_838_592,
    (WHISPER_REPO, "model.bin", WHISPER_REVISION): 1_511_322_369,
    (WHISPER_REPO, "config.json", WHISPER_REVISION): 1_939,
    (WHISPER_REPO, "tokenizer.json", WHISPER_REVISION): 614_662,
    (WHISPER_REPO, "vocabulary.json", WHISPER_REVISION): 176_885,
    (WHISPER_REPO, "preprocessor_config.json", WHISPER_REVISION): 340,
    (SILERO_REPO, SILERO_FILE, SILERO_REVISION): 2_243_022,
}


class ModelArtifactCorrupt(RuntimeError):
    """A cached model file failed its pinned SHA-256 check.

    Carries ``path``, the file holding the bad bytes, as a FIELD so the
    startup dialog can name the file to delete. Re-running the first-run
    wizard does not repair it: for a commit revision, hub returns the
    existing cached file without downloading it again. Lives in this
    dependency-free module, beside the digest pins, so ui/error_dialog.py
    can identify it without importing numpy or onnxruntime.
    """

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path


class BundledModelInvalid(RuntimeError):
    """A model file shipped inside the install is missing or fails its
    pinned SHA-256 check.

    Unlike :class:`ModelArtifactCorrupt`, nothing downloads this file: it
    comes with the installer, so only a reinstall restores it. Carries
    ``path`` as a FIELD for the startup dialog.
    """

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path


def whisper_revision_for(model: str) -> str | None:
    """Revision pin for a configured whisper model.

    WHISPER_REVISION is a commit sha that exists ONLY in the default
    WHISPER_REPO. ``asr.faster_whisper.model`` is user-overridable (e.g. to
    anime-whisper), and both a different HF repo and a local directory path
    must load UNPINNED -- passing the default-repo-only sha with any other
    repo is a guaranteed RevisionNotFoundError at startup. The pin's purpose
    (wizard and runtime resolve the exact same snapshot, offline) applies to
    the default repo only, so that is the only case that gets it.
    """
    return WHISPER_REVISION if model == WHISPER_REPO else None


def pinned_sha256(repo_id: str, filename: str, revision: str | None) -> str | None:
    """The pinned SHA-256 for ``repo_id``/``filename`` at ``revision``, or
    None if that file has no digest pin at that revision (configs,
    tokenizer, an unpinned repo, or another revision of a pinned repo)."""
    for pin in TRANSLATOR_PINS.values():
        if (repo_id, filename, revision) == (pin.repo, pin.filename, pin.revision):
            return pin.sha256
    if (repo_id, filename, revision) == (WHISPER_REPO, "model.bin", WHISPER_REVISION):
        return WHISPER_MODEL_BIN_SHA256
    if (repo_id, filename, revision) == (SILERO_REPO, SILERO_FILE, SILERO_REVISION):
        return SILERO_SHA256
    return None


def pinned_size(repo_id: str, filename: str, revision: str | None) -> int | None:
    """The exact byte size of ``repo_id``/``filename`` at ``revision``, or
    None for a file or revision without a size pin."""
    return _PINNED_SIZES.get((repo_id, filename, revision))


def _repo_files_cached(repo_id: str, filenames, revision: str | None) -> bool:
    """True iff EVERY ``filenames`` entry for ``repo_id`` is in the local HF
    cache at ``revision`` (None probes the ``main`` ref).

    Pure LOCAL probe: ``try_to_load_from_cache`` never touches the network.
    A revision-pinned query matters for sha-pinned downloads (whisper,
    silero): they write only ``snapshots/<sha>/`` (no ``refs/main``), so a
    bare probe would default to `main` and miss the warm cache.

    May raise (hub import failure, cache-layout probe error): callers pick
    their own policy -- asr_client fails CLOSED (online load path),
    ``required_models_cached`` fails OPEN (skip the wizard).
    """
    import os

    from yamabiko import hf_env  # noqa: F401 -- pin HF_HOME before any hub call
    from huggingface_hub import try_to_load_from_cache

    for fname in filenames:
        hit = try_to_load_from_cache(
            repo_id=repo_id, filename=fname, revision=revision,
        )
        if not (isinstance(hit, str) and os.path.isfile(hit)):
            return False
    return True


def whisper_files_cached(model: str = WHISPER_REPO) -> bool:
    """True iff EVERY WHISPER_FILES entry for ``model`` is in the local HF
    cache (pinned revision for the default repo, unpinned otherwise --
    see ``whisper_revision_for``).

    faster-whisper needs all five files to load offline, so a partial cache
    (mid-download abort) must report False -- an offline load against it
    would hard-crash with no online fallback. Local-only; may raise (see
    ``_repo_files_cached``).
    """
    return _repo_files_cached(model, WHISPER_FILES, whisper_revision_for(model))


def _translator_gguf_cached() -> bool:
    """True iff at least ONE translator GGUF quant tier is in the local HF
    cache.

    Either tier is enough: ``watchdog._resolve_translator_model_path``
    self-heals to the other cached tier when the configured one is missing,
    so llama-server can start whenever either file is present. Each tier is
    probed at ITS OWN pinned repo + revision (the tiers may live in
    different repos) -- a sha-pinned download writes only
    ``snapshots/<sha>/`` (no ``refs/main``), so a bare probe would default
    to `main` and miss the warm cache. May raise; see
    ``whisper_files_cached``.
    """
    return any(
        _repo_files_cached(pin.repo, (pin.filename,), pin.revision)
        for pin in TRANSLATOR_PINS.values()
    )


def _effective_whisper_model() -> str:
    """Resolve the EFFECTIVE ``asr.faster_whisper.model`` from merged config.

    The whisper model is user-overridable (a different HF repo such as
    anime-whisper, or a local CT2 directory -- see ``whisper_revision_for``
    and asr_client.build_asr), so the wizard gate must ask what the RUNTIME
    will actually load, not assume the default constant. Reads the same
    merged view (shipped defaults + user_overrides.yaml) the wizard's own
    downloader already reads (wizard_pages/downloads.py::
    _resolve_translator_pin), via ``yamabiko.config.load_full_config``.

    Function-local import on purpose: this module is the leaf
    constants module imported by config consumers up the stack, and a
    top-level ``yamabiko.config`` import would pull yaml/paths into every
    constants import and invite a cycle if config.py ever imports models
    (today it does not, but nothing enforces that).

    Never raises: any config-read failure -- or a non-string / empty
    value -- falls back to the default constant, so a broken
    user_overrides.yaml degrades to the default-probe behavior rather
    than silently skipping (or forcing) the wizard.
    """
    try:
        from yamabiko import config

        model = config.load_full_config()["asr"]["faster_whisper"]["model"]
        if isinstance(model, str) and model:
            return model
    except Exception:
        # Swallow KEPT (never raises -- see docstring), but the failure
        # must leave evidence: a broken user_overrides.yaml silently
        # flipping the wizard gate was undiagnosable from field logs.
        # Function-local logging import, same idiom as
        # required_models_cached below.
        import logging

        logging.getLogger(__name__).warning(
            "config read for asr.faster_whisper.model failed; falling "
            "back to the default %s", WHISPER_REPO, exc_info=True,
        )
    return WHISPER_REPO


def default_whisper_missing() -> bool:
    """True when the runtime will load the default WHISPER_REPO (the
    effective ``asr.faster_whisper.model``, see ``_effective_whisper_model``)
    and not every WHISPER_FILES entry is cached at WHISPER_REVISION. False
    for an overridden model: setup does not download those. Shared by
    ``required_models_cached`` (reopen setup) and
    ``wizard_state.speech_model_update_pending`` (Welcome's update line).
    Local-only; may raise (see ``_repo_files_cached``).
    """
    return _effective_whisper_model() == WHISPER_REPO and not (
        whisper_files_cached(WHISPER_REPO)
    )


def required_models_cached() -> bool:
    """LOCAL-ONLY probe: are the models the wizard is responsible for
    downloading already in the HF cache?

    Consumed by ``wizard_state.needs_wizard()`` to close the upgrade
    migration gap: the NSIS installer deliberately preserves state.json
    across upgrades, so a release that changes a model pin (e.g. the
    WHISPER_REPO deepdml -> kotoba -> efwkjn -> Kodama-1 swaps) would
    otherwise boot the pipeline with the new model uncached -- a surprise
    ~1.5 GB synchronous download inside pipeline startup, or an offline hard
    failure. Probes the DEFAULT constants, mirroring the wizard's download
    queue (downloads.py::download_queue): the default whisper repo at its
    pinned revision, either translator GGUF tier, and the silero VAD file
    at its pinned revision.

    The whisper arm is CONDITIONAL on the effective config: it first
    resolves ``asr.faster_whisper.model`` from merged config (shipped
    defaults + user overrides -- the same read the wizard's own downloader
    performs; see ``_effective_whisper_model``) and probes only when the
    effective model IS the default WHISPER_REPO. For an override (a
    different HF repo, or a local CT2 directory) the whisper arm is
    SKIPPED entirely: the wizard's queue hard-codes the default set, so
    re-running the wizard cannot satisfy an override -- it would download
    ~1.5 GB the runtime never loads, and OFFLINE the queue fails, leaving
    cancel -> exit 42 as the wizard's only exit on EVERY launch: a
    bricked install. Fetching (or erroring on) an override belongs to the
    runtime's own load path (asr_client.build_asr). A config-read failure
    falls back to the default constant, degrading to the default-probe
    behavior.

    Never raises. Any probe exception is logged and reported as True
    (fail-open): a probe bug must degrade to today's behavior, never
    force the wizard on every launch of a healthy install.
    """
    import logging

    try:
        if default_whisper_missing():
            return False
        if not _translator_gguf_cached():
            return False
        if not _repo_files_cached(SILERO_REPO, (SILERO_FILE,), SILERO_REVISION):
            return False
        return True
    except Exception:
        logging.getLogger(__name__).exception(
            "required_models_cached probe failed; assuming models are "
            "cached (fail-open) so a probe bug cannot force the wizard"
        )
        return True
