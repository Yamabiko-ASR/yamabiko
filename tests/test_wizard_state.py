"""wizard_state.py persists first-run wizard results to state.json.

Design contract:
  - Missing file ⇒ first-run not complete; `needs_wizard()` returns True.
  - Round-trip: save then load returns the same fields.
  - Corrupt JSON ⇒ treat as missing; `needs_wizard()` True.
  - Partial state with first_run_complete=False ⇒ wizard needs to re-run.
  - Complete state but the wizard-downloaded models missing from the HF
    cache (`models.required_models_cached()` False) ⇒ wizard re-runs.
    This is the UPGRADE path: the NSIS installer preserves state.json,
    so a release that changes a model pin (whisper deepdml → kotoba)
    lands with first_run_complete=True and the new model uncached.
  - User override of `asr.faster_whisper.model` (HF repo or local dir)
    ⇒ the probe SKIPS the whisper arm: the wizard can only download the
    default set, so it cannot satisfy an override -- forcing it would be
    a pointless (offline: bricking) download.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import huggingface_hub
import huggingface_hub.constants as hf_constants
import pytest

from yamabiko import models, paths, wizard_state


def _isolate_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tmp_path.mkdir(exist_ok=True)


def _complete_state_payload() -> dict[str, object]:
    return {
        "first_run_complete": True,
        "wizard_version": "0.1.0-beta",
        "wizard_completed_at": "2026-07-15T00:00:00+00:00",
        "audio_device_name": None,
        "genre": "variety",
        "llama_smoke_ok": True,
    }


def test_needs_wizard_when_state_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    assert wizard_state.needs_wizard() is True


def test_incomplete_state_loads_supported_defaults_and_requires_wizard(
    tmp_path, monkeypatch,
):
    """A partial non-complete state is accepted with typed defaults.

    Seeded defect: replace the completed-state condition with ``if True:`` ->
    the partial payload is rejected and ``load()`` returns None -> red.
    """
    _isolate_state(tmp_path, monkeypatch)
    paths.state_path().write_text(
        json.dumps({"first_run_complete": False}), encoding="utf-8"
    )
    loaded = wizard_state.load()
    assert loaded is not None
    assert isinstance(loaded, wizard_state.WizardResult)
    assert loaded.first_run_complete is False
    assert type(loaded.wizard_version) is str
    assert loaded.wizard_completed_at is None
    assert loaded.audio_device_name is None
    assert loaded.genre == "variety"
    assert loaded.llama_smoke_ok is False
    assert wizard_state.needs_wizard() is True


def test_needs_wizard_false_when_state_complete_and_model_probe_reports_cached(
    tmp_path, monkeypatch,
):
    """Completed state + models present (probe stubbed True) ⇒ no wizard.

    The probe is a collaborator here, stubbed so this test isolates the
    state-file trigger; the real-probe integration lives in
    test_needs_wizard_false_when_state_complete_and_models_warm_in_cache.

    Seeded defect: invert the state gate (`result.first_run_complete` →
    `not result.first_run_complete`) -> True here -> red.
    """
    monkeypatch.setattr(
        "yamabiko.models.required_models_cached", lambda: True,
    )
    _isolate_state(tmp_path, monkeypatch)
    paths.state_path().write_text(
        json.dumps(_complete_state_payload()), encoding="utf-8"
    )
    assert wizard_state.needs_wizard() is False


def test_corrupt_json_treated_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tmp_path.mkdir(exist_ok=True)
    paths.state_path().write_text("{{{not valid", encoding="utf-8")
    assert wizard_state.needs_wizard() is True


@pytest.mark.parametrize("body", ["null", "[]", '"text"', "1"])
def test_load_non_object_json_requires_wizard_without_raising(
    body, tmp_path, monkeypatch,
):
    """A non-object JSON root is invalid state, not an escaped exception.

    Seeded defect: delete the decoder's root ``isinstance(data, dict)``
    check -> TypeError/AttributeError escapes for null, list, and int -> red.
    """
    _isolate_state(tmp_path, monkeypatch)
    paths.state_path().write_text(body, encoding="utf-8")
    assert wizard_state.load() is None
    assert wizard_state.needs_wizard() is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_run_complete", "false"),
        ("wizard_version", 2),
        ("wizard_completed_at", 123),
        ("audio_device_name", []),
        ("genre", False),
        ("llama_smoke_ok", 1),
    ],
)
def test_wrong_field_type_cannot_mark_setup_complete(
    field, value, tmp_path, monkeypatch,
):
    """Persisted annotations are exact types, never coercion requests.

    Seeded defect: replace exact type membership with a one-line
    ``bool(value)`` gate limited to the two boolean keys -> all six malformed
    values produce completed WizardResult objects -> red.
    """
    _isolate_state(tmp_path, monkeypatch)
    payload = _complete_state_payload()
    payload[field] = value
    paths.state_path().write_text(json.dumps(payload), encoding="utf-8")
    assert wizard_state.load() is None
    assert wizard_state.needs_wizard() is True


@pytest.mark.parametrize(
    "missing_field",
    [
        "wizard_version",
        "wizard_completed_at",
        "audio_device_name",
        "genre",
        "llama_smoke_ok",
    ],
)
def test_completed_state_missing_required_field_is_rejected_and_requires_wizard(
    missing_field, tmp_path, monkeypatch,
):
    """A completion marker is valid only with every other current field.

    Seeded defect: replace the missing-field calculation with
    ``missing = set()`` -> the incomplete completed payload loads -> red.
    """
    _isolate_state(tmp_path, monkeypatch)
    payload = _complete_state_payload()
    del payload[missing_field]
    paths.state_path().write_text(json.dumps(payload), encoding="utf-8")
    assert wizard_state.load() is None
    assert wizard_state.needs_wizard() is True


def test_future_unknown_field_cannot_mark_setup_complete(tmp_path, monkeypatch):
    """Only the named legacy field bypasses the supported-field gate.

    Seeded defect: replace the unknown-field calculation with
    ``unknown = set()`` -> ``future_completion_gate`` is discarded -> red.
    """
    _isolate_state(tmp_path, monkeypatch)
    payload = _complete_state_payload()
    payload["future_completion_gate"] = True
    paths.state_path().write_text(json.dumps(payload), encoding="utf-8")
    assert wizard_state.load() is None
    assert wizard_state.needs_wizard() is True


def test_schema_failure_logs_warning_with_rejection_reason(
    tmp_path, monkeypatch, caplog,
):
    """A rejected schema leaves WARNING evidence naming the bad field.

    Seeded defect: downgrade ``_log.warning`` to ``_log.info`` -> no matching
    WARNING record is captured -> red.
    """
    _isolate_state(tmp_path, monkeypatch)
    payload = _complete_state_payload()
    payload["wizard_version"] = 2
    paths.state_path().write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="yamabiko.wizard_state"):
        assert wizard_state.load() is None

    records = [
        record for record in caplog.records
        if record.name == "yamabiko.wizard_state"
        and record.levelno == logging.WARNING
    ]
    assert records, "schema rejection emitted no wizard-state WARNING"
    assert "wizard_version" in records[0].getMessage()
    assert "unsupported type int" in records[0].getMessage()


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    result = wizard_state.WizardResult(
        first_run_complete=True,
        wizard_version="0.1.0-beta",
        audio_device_name="Speakers (Realtek) [Loopback]",
        genre="variety",
        llama_smoke_ok=True,
    )
    saved = wizard_state.save(result)
    # save() does NOT mutate the input
    assert result.wizard_completed_at is None
    # but the returned object carries the stamped timestamp
    assert saved.wizard_completed_at is not None
    loaded = wizard_state.load()
    assert loaded == saved


def test_load_tolerates_legacy_livekit_license_accepted_key(tmp_path, monkeypatch):
    """An install upgraded from a build that shipped the turn-detector wrote
    a ``livekit_license_accepted`` field into state.json. That field no
    longer exists on WizardResult. load() must FILTER the now-unknown key
    and still return a usable result (first_run_complete preserved), NOT
    treat the whole file as corrupt -- otherwise every upgraded install
    is force-re-run through the first-run wizard.

    Seeded defect: remove ``livekit_license_accepted`` from the decoder's
    legacy allowlist -> the key is rejected -> load() returns None -> the
    first_run_complete assertion below goes red.
    """
    _isolate_state(tmp_path, monkeypatch)
    payload = _complete_state_payload()
    payload["livekit_license_accepted"] = True
    paths.state_path().write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    loaded = wizard_state.load()
    assert loaded is not None, (
        "state.json from a turn-detector-era install failed to load; the "
        "legacy livekit_license_accepted key was treated as corruption"
    )
    assert loaded.first_run_complete is True
    assert not hasattr(loaded, "livekit_license_accepted")


def test_save_creates_app_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "nested" / "deep"))
    result = wizard_state.WizardResult(first_run_complete=True)
    wizard_state.save(result)
    assert paths.state_path().is_file()


def test_reset_removes_state(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tmp_path.mkdir(exist_ok=True)
    paths.state_path().write_text('{"first_run_complete": true}', encoding="utf-8")
    wizard_state.reset()
    assert not paths.state_path().exists()


def test_save_leaves_no_tmp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    wizard_state.save(wizard_state.WizardResult(first_run_complete=True))
    tmp_leftover = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tmp_leftover == [], f"save left temp file(s): {tmp_leftover}"


def test_save_removes_tmp_file_when_write_raises(tmp_path, monkeypatch):
    """A mid-write failure must not litter the dir with a `.tmp`.

    save() writes to ``state.json.tmp`` then ``os.replace``s it into
    place. If the write/flush/fsync inside the ``with open(tmp)`` block
    raises, the partial ``.tmp`` must be cleaned up (mirroring
    config_writer._atomic_write) AND the original exception must
    propagate -- otherwise repeated failed saves accumulate orphan
    ``.tmp`` litter in %LOCALAPPDATA%\\Yamabiko alongside the live
    state.json.

    Seeded defect: delete the ``except Exception: <unlink tmp>; raise``
    cleanup block from save() -> the OSError still propagates (so the
    pytest.raises clause stays green) but the ``state.json.tmp`` is left
    behind -> the no-leftover assertion goes red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    real_fsync = os.fsync

    def exploding_fsync(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", exploding_fsync)

    with pytest.raises(OSError):
        wizard_state.save(wizard_state.WizardResult(first_run_complete=True))

    monkeypatch.setattr(os, "fsync", real_fsync)
    leftover = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover == [], (
        f"failed save() left orphan temp file(s): {leftover} -- a "
        f"mid-write OSError must unlink the partial .tmp (mirrors "
        f"config_writer._atomic_write)"
    )


def test_save_fsyncs_flushed_temp_file_before_replace(tmp_path, monkeypatch):
    """os.replace journals the NTFS RENAME atomically but does NOT flush the
    temp file's data blocks. Without flush+fsync before the replace, power
    loss shortly after a save can publish a zero-length state.json -- which
    needs_wizard() reads as "first run not complete", so the first-run wizard
    re-runs on next launch and the user's completed setup silently vanishes.

    Asserts the established fsync-before-replace idiom (mirrors
    tests\test_glossary.py::test_save_fsyncs_flushed_temp_file_before_replace):
    at least one os.fsync runs BEFORE os.replace, and at fsync time the fd
    already holds data (size > 0 -- a single buffered fh.write stays in the
    Python userspace buffer until flushed, so the size clause makes the flush
    load-bearing for the json path).

    Seeded defect: revert wizard_state.save's temp write to
    ``tmp.write_text(...)`` (no flush/fsync) -> no fsync event before the
    replace -> red.
    """
    import os

    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    events: list[tuple[str, int]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        events.append(("fsync", os.fstat(fd).st_size))
        return real_fsync(fd)

    def recording_replace(src, dst):
        events.append(("replace", -1))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    wizard_state.save(wizard_state.WizardResult(first_run_complete=True))

    replace_indices = [
        i for i, (kind, _) in enumerate(events) if kind == "replace"
    ]
    assert replace_indices, "save() never reached os.replace"
    first_replace = replace_indices[0]
    flushed_fsyncs_before_replace = [
        i for i, (kind, size) in enumerate(events)
        if kind == "fsync" and size > 0 and i < first_replace
    ]
    assert flushed_fsyncs_before_replace, (
        f"no os.fsync of flushed data before os.replace -- state.json "
        f"is not durable across power loss; a zero-length state.json "
        f"makes needs_wizard() return True and the wizard re-runs on "
        f"next launch. Events: {events!r}"
    )


# ---------------------------------------------------------------------------
# Upgrade-migration gate: needs_wizard() must also fire when the models the
# wizard is responsible for downloading are missing from the HF cache
# (models.required_models_cached()). The NSIS installer preserves state.json
# across upgrades and the wizard is the ONLY downloader, so a release that
# changes a model pin (whisper deepdml -> kotoba) would otherwise boot the
# pipeline with the new model uncached: a surprise ~1.5 GB synchronous
# download inside startup, or an offline hard failure.
#
# These tests run the REAL probe (huggingface_hub.try_to_load_from_cache)
# against a temp HF cache tree laid out exactly like the real hub cache
# (models--<org>--<name>/snapshots/<sha>/<file>). try_to_load_from_cache
# reads constants.HF_HUB_CACHE at call time when cache_dir=None, so the
# redirect monkeypatches that constant (plus HF_HOME for anything that
# re-derives it) -- the same idiom as
# tests/test_real_cjk_model_paths.py::test_hf_cache_hit_under_cjk_hf_home.
# ---------------------------------------------------------------------------

_FAKE_SHA = "0" * 40  # a plausible non-pinned commit sha


def _fake_hub_cache(tmp_path, monkeypatch) -> Path:
    """Redirect the HF hub cache to an empty temp dir; return the hub dir.

    Also points YAMABIKO_APPDATA at tmp_path: required_models_cached()
    resolves the effective ``asr.faster_whisper.model`` from merged
    config (user_overrides.yaml under the app-data dir), so every probe
    test must be hermetic against the developer's REAL app data -- a
    real override there would silently skip the whisper arm and flip
    these tests. tmp_path has no overrides file, so the shipped-default
    (default-constant probe) path these tests pin still applies.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    hf_home = tmp_path / "hf-cache"
    hub = hf_home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub))
    return hub


def _warm_repo(hub: Path, repo: str, files, snapshot_sha: str,
               ref_name: str | None = None) -> None:
    """Lay out a fake warmed cache entry in the real hub-cache shape.

    ``ref_name`` writes ``refs/<ref_name>`` pointing at ``snapshot_sha``:
    hf_hub_download writes a ref for any non-sha revision ("main", or a
    tag like "v1.2") but NOT for a pinned commit sha.
    """
    repo_dir = hub / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / snapshot_sha
    snap.mkdir(parents=True, exist_ok=True)
    for fname in files:
        p = snap / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-model-bytes")
    if ref_name is not None:
        refs = repo_dir / "refs"
        refs.mkdir(exist_ok=True)
        (refs / ref_name).write_text(snapshot_sha, encoding="utf-8")


def _warm_default_whisper(hub: Path, files=None) -> None:
    # A wizard-warmed whisper cache: the revision-pinned download writes
    # ONLY snapshots/<sha>/ -- no refs/main. Reproduce that exactly.
    _warm_repo(
        hub, models.WHISPER_REPO,
        models.WHISPER_FILES if files is None else files,
        models.WHISPER_REVISION,
    )


def _warm_translator(hub: Path, tier: str = "default") -> None:
    # The translator GGUF downloads PINNED at its tier pin's revision
    # (wizard + watchdog resolve it at that sha), so its cache entry is
    # snapshots/<pinned sha>/ ONLY -- no refs/main, exactly like whisper.
    pin = models.TRANSLATOR_PINS[tier]
    _warm_repo(hub, pin.repo, [pin.filename], pin.revision)


def _warm_silero(hub: Path) -> None:
    # Sha-pinned download like whisper: snapshots/<pinned sha>/ only.
    _warm_repo(
        hub, models.SILERO_REPO, [models.SILERO_FILE],
        models.SILERO_REVISION,
    )


def _warm_all(hub: Path) -> None:
    """The full wizard-warmed cache: all three download sets
    (downloads.py::download_queue)."""
    _warm_default_whisper(hub)
    _warm_translator(hub)
    _warm_silero(hub)


def _write_complete_state(tmp_path, monkeypatch) -> None:
    _isolate_state(tmp_path, monkeypatch)
    paths.state_path().write_text(
        json.dumps(_complete_state_payload()), encoding="utf-8"
    )


def test_needs_wizard_true_when_state_complete_but_models_missing(
    tmp_path, monkeypatch,
):
    """THE upgrade-gap test: first_run_complete=True (preserved state.json)
    but an EMPTY HF cache ⇒ needs_wizard() must return True so the wizard
    (the only downloader) runs before the pipeline ever needs the models.

    Real probe against an empty temp cache -- no mocks on the subject.

    Seeded defect: revert needs_wizard() to the pre-migration body
    (`return result is None or not result.first_run_complete`) -> returns
    False here -> red.
    """
    _write_complete_state(tmp_path, monkeypatch)
    _fake_hub_cache(tmp_path, monkeypatch)  # empty: nothing cached
    assert wizard_state.needs_wizard() is True, (
        "state.json says first_run_complete but NO model is in the HF "
        "cache (the upgrade / pin-swap case); needs_wizard() must route "
        "to the wizard, the only downloader"
    )


def test_needs_wizard_false_when_state_complete_and_models_warm_in_cache(
    tmp_path, monkeypatch,
):
    """A healthy warmed install (complete state + all three wizard download
    sets in the cache: whisper, shisa GGUF, silero) must NOT be
    routed through the wizard -- a fail-closed probe bug here would force
    first-run setup on every launch of every healthy install.

    Real probe against a fully warmed fake cache tree in the exact
    wizard-warmed shape (whisper/silero/shisa: snapshots/<pinned sha>/
    only, no refs -- all three downloads are now revision-pinned).

    Seeded defect: probe any sha-pinned repo at revision="main" instead of
    its pin -> the refs-less wizard-warmed layout misses ->
    needs_wizard() True -> red.
    """
    _write_complete_state(tmp_path, monkeypatch)
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_all(hub)
    assert wizard_state.needs_wizard() is False, (
        "fully warmed cache + completed state, yet needs_wizard() still "
        "True -- the model probe is failing closed and would re-run "
        "first-run setup on every launch"
    )


def test_required_models_cached_false_when_whisper_snapshot_absent(
    tmp_path, monkeypatch,
):
    """Everything cached EXCEPT the whisper snapshot ⇒ False. Kills a probe
    that forgets the whisper arm (seeded defect: drop the whisper check
    from required_models_cached) -- exactly the deepdml -> kotoba upgrade
    shape, where every other model survives the upgrade but the whisper
    repo moved."""
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_translator(hub)
    _warm_silero(hub)
    assert models.required_models_cached() is False


def test_required_models_cached_false_when_no_gguf_cached(
    tmp_path, monkeypatch,
):
    """Everything cached but NO translator quant tier ⇒ False. Kills a
    probe that forgets the LLM arm (seeded defect: drop the
    _translator_gguf_cached check from required_models_cached) -- without
    the GGUF, llama-server polls the cache forever and translation is dead."""
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_default_whisper(hub)
    _warm_silero(hub)
    assert models.required_models_cached() is False


def test_required_models_cached_true_when_only_small_tier_gguf_cached(
    tmp_path, monkeypatch,
):
    """A small-tier install (8 GB VRAM: only the small-tier GGUF cached) is
    complete -- watchdog._resolve_translator_model_path self-heals to
    whichever tier is cached, so llama-server can start.

    Seeded defect: probe only the default pin in _translator_gguf_cached ->
    every small-tier user is forced back through the wizard on every
    launch -> False here -> red.
    """
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_default_whisper(hub)
    _warm_translator(hub, "small")
    _warm_silero(hub)
    assert models.required_models_cached() is True


def test_required_models_cached_false_on_partial_whisper_cache(
    tmp_path, monkeypatch,
):
    """A partial whisper cache (model.bin landed, a metadata file did not --
    mid-download abort) ⇒ False: faster-whisper needs ALL five files, so
    reporting cached would skip the wizard and startup would still miss.

    Seeded defect: key the whisper probe on model.bin alone -> partial
    cache reports True -> red.
    """
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_default_whisper(hub, files=models.WHISPER_FILES[:-1])
    _warm_translator(hub)
    _warm_silero(hub)
    assert models.required_models_cached() is False


def test_required_models_cached_false_when_whisper_cached_at_unpinned_revision(
    tmp_path, monkeypatch,
):
    """Whisper cached only at a DIFFERENT sha ⇒ False, mirroring
    asr_client._whisper_cached: the runtime loads offline at the pinned
    revision, so a wrong-revision cache still means an online HEAD-check /
    re-download at pipeline startup -- the thing this gate exists to route
    through the wizard instead.

    Seeded defect: drop the revision from the whisper probe
    (try_to_load_from_cache without revision resolves refs/main) -> the
    stale refs/main entry hits -> True -> red.
    """
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_repo(hub, models.WHISPER_REPO, models.WHISPER_FILES,
               _FAKE_SHA, ref_name="main")
    _warm_translator(hub)
    _warm_silero(hub)
    assert models.required_models_cached() is False


def test_required_models_cached_false_when_translator_cached_at_unpinned_revision(
    tmp_path, monkeypatch,
):
    """The translator cached only at a DIFFERENT sha (an old unpinned
    download whose refs/main points at a pre-pin commit) ⇒ False. The
    runtime resolver (watchdog._resolve_translator_model_path) probes at
    the tier pin's revision, and a sha-pinned download writes only
    snapshots/<sha>/; a wrong-revision cache means llama-server would poll
    forever for a snapshot that never lands, so route through the wizard
    instead. Mirrors the whisper unpinned-revision gate. This is ALSO the
    exact shape of the 2026-07-12 shisa -> Qwen3.5 pin swap on an upgraded
    install: the old model is warm, the new pins are not.

    Seeded defect: revert _translator_gguf_cached to revision=None
    (resolves refs/main) -> the stale refs/main entry hits -> True -> red.
    """
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_default_whisper(hub)
    pin = models.TRANSLATOR_PINS["default"]
    _warm_repo(hub, pin.repo, [pin.filename], _FAKE_SHA, ref_name="main")
    _warm_silero(hub)
    assert models.required_models_cached() is False


def test_required_models_cached_false_when_silero_absent(
    tmp_path, monkeypatch,
):
    """Everything cached EXCEPT the silero VAD ONNX ⇒ False. Silero is
    unconditional in the wizard queue and the runtime loads it offline
    (vad/silero_chunker.py), so a future SILERO_REVISION bump on an
    upgraded install is the same gap as the whisper repo swap.

    Seeded defect: drop the silero arm from required_models_cached ->
    True here -> red.
    """
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_default_whisper(hub)
    _warm_translator(hub)
    assert models.required_models_cached() is False


# ---------------------------------------------------------------------------
# Whisper-override skip: ``asr.faster_whisper.model`` is user-overridable
# (first-class: whisper_revision_for / asr_client.build_asr both handle it),
# but the wizard's download queue hard-codes the DEFAULT whisper repo
# (downloads.py::download_queue). If required_models_cached() probed
# the default constant regardless, an override install missing the default
# whisper snapshot would be forced through the wizard on EVERY launch --
# downloading ~1.5 GB the runtime never loads, or (offline) failing with
# cancel -> exit 42 as the only way out: a bricked install. So the whisper
# arm resolves the EFFECTIVE model from merged config and probes only when
# it IS the default; override fetching/erroring belongs to the runtime's
# own load path (asr_client.build_asr).
# ---------------------------------------------------------------------------


def _write_whisper_model_override(tmp_path, monkeypatch, model: str) -> None:
    """Write a REAL user_overrides.yaml overriding asr.faster_whisper.model.

    Points YAMABIKO_APPDATA at tmp_path so config.load_full_config() --
    the exact merged read required_models_cached() performs -- resolves
    this file. Subject is the real config-resolution + skip logic, so no
    stubbing of the loader. (JSON is valid YAML.)
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    tmp_path.mkdir(exist_ok=True)
    paths.user_overrides_path().write_text(
        json.dumps({"asr": {"faster_whisper": {"model": model}}}),
        encoding="utf-8",
    )


def test_required_models_cached_true_for_hf_repo_override_without_default_whisper(
    tmp_path, monkeypatch,
):
    """user_overrides.yaml points asr.faster_whisper.model at a DIFFERENT
    HF repo; the default whisper snapshot is NOT cached (everything else
    warm) ⇒ True: the wizard cannot download an override, so routing this
    install through it would fetch ~1.5 GB the runtime never loads -- or,
    offline, brick every launch (wizard download fails; only exit is
    cancel -> exit 42).

    Seeded defect: probe the default WHISPER_REPO regardless of the
    configured override -> default snapshot absent -> False -> red.
    """
    _write_whisper_model_override(
        tmp_path, monkeypatch, "litagin/anime-whisper",
    )
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_translator(hub)
    _warm_silero(hub)  # default whisper deliberately absent
    assert models.required_models_cached() is True, (
        "asr.faster_whisper.model is overridden to another HF repo; the "
        "default whisper snapshot's absence must NOT force the wizard -- "
        "its queue can only download the default set"
    )


def test_required_models_cached_true_for_local_dir_override_without_default_whisper(
    tmp_path, monkeypatch,
):
    """Same skip for a LOCAL-DIRECTORY override (faster-whisper accepts a
    CT2 model dir path): the HF cache probe is meaningless for a
    filesystem path and the wizard cannot download one.

    Seeded defect: recognize only HF-repo-shaped overrides (e.g. key the
    skip on ``"/" in model``) -- a Windows ``C:\\...`` path then falls
    through to the default-repo probe -> default absent -> False -> red.
    """
    local_dir = tmp_path / "ct2-models" / "anime-whisper-ct2"
    _write_whisper_model_override(tmp_path, monkeypatch, str(local_dir))
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_translator(hub)
    _warm_silero(hub)  # default whisper deliberately absent
    assert models.required_models_cached() is True


def test_required_models_cached_probes_default_whisper_when_config_read_raises(
    tmp_path, monkeypatch,
):
    """Config unreadable ⇒ the effective-model resolution falls back to
    the DEFAULT constant and the whisper arm still probes it (today's
    fail-open-to-default semantics). Asserted via default whisper absent
    + everything else warm -> False: the genuinely-missing default is
    still caught.

    Seeded defect: let the config-read exception escape to the outer
    fail-open handler (returns True), or make a read failure SKIP the
    whisper arm (also True) -> the is-False assertion goes red either
    way.
    """
    def _boom():
        raise RuntimeError("user_overrides.yaml unreadable")

    monkeypatch.setattr("yamabiko.config.load_full_config", _boom)
    hub = _fake_hub_cache(tmp_path, monkeypatch)
    _warm_translator(hub)
    _warm_silero(hub)  # default whisper deliberately absent
    assert models.required_models_cached() is False, (
        "config read failed, so the probe must fall back to the DEFAULT "
        "whisper repo (absent here) -- not skip the whisper arm, not "
        "fail open"
    )


def test_required_models_cached_fails_open_when_probe_raises(
    tmp_path, monkeypatch, caplog,
):
    """A probe bug must NEVER brick startup by forcing the wizard on every
    launch: any exception inside the probe is logged and reported as True
    (today's behavior). Contrast asr_client._whisper_cached, which fails
    CLOSED to the safe online-load path.

    Seeded defect: let the exception propagate (drop the try/except in
    required_models_cached) -> RuntimeError escapes -> red. Or fail closed
    (return False in the except) -> the is-True assertion goes red.
    """
    def _boom(**kw):
        raise RuntimeError("hub probe blew up")

    # Hermetic app data: the probe's effective-whisper-model resolution
    # reads user_overrides.yaml; the dev machine's real one must not leak.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _boom)
    with caplog.at_level(logging.ERROR, logger="yamabiko.models"):
        assert models.required_models_cached() is True
    assert "required_models_cached probe failed" in caplog.text, (
        "the fail-open path must log the swallowed probe exception for "
        "forensics"
    )
    assert "RuntimeError" in caplog.text, (
        "the traceback of the swallowed exception must reach the log"
    )


# ---------------------------------------------------------------------------
# Evidence-erasing swallows (2026-07-01 logging audit, Task 9): the two
# broad excepts below used to be bare `pass`. They must KEEP swallowing
# (fallback semantics unchanged) but now leave a WARNING with the
# traceback in the log.
# ---------------------------------------------------------------------------


def test_effective_whisper_model_config_read_failure_logs_warning_and_falls_back(
    tmp_path, monkeypatch, caplog,
):
    """models._effective_whisper_model's config-read swallow must log a
    WARNING (with traceback) while still falling back to the default
    constant -- a broken user_overrides.yaml previously degraded with
    ZERO evidence, making "why did the wizard re-run?" undiagnosable
    from the field logs.

    Seeded defect: re-silence the except (revert models.py's
    _effective_whisper_model to ``except Exception: pass``) -> no
    WARNING record -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    def _boom():
        raise RuntimeError("user_overrides.yaml unreadable")

    monkeypatch.setattr("yamabiko.config.load_full_config", _boom)

    with caplog.at_level(logging.WARNING, logger="yamabiko.models"):
        # Swallow semantics KEPT: no raise, default-constant fallback.
        assert models._effective_whisper_model() == models.WHISPER_REPO

    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.models" and r.levelno == logging.WARNING
    ]
    assert recs, "config-read failure left no WARNING evidence"
    assert recs[0].exc_info and recs[0].exc_info[0] is RuntimeError, (
        "the WARNING must carry the swallowed traceback (exc_info=True)"
    )


def test_save_cleanup_unlink_failure_logs_warning_and_reraises_original(
    tmp_path, monkeypatch, caplog,
):
    """When the write fails (fsync ENOSPC) AND the .tmp cleanup unlink also
    fails, save() must log a WARNING (with traceback) for the failed
    cleanup and still re-raise the ORIGINAL write error. The cleanup's
    ``except OSError: pass`` previously erased the unlink failure, so
    orphan-.tmp litter in %LOCALAPPDATA%\\Yamabiko was undiagnosable.

    Seeded defect: re-silence the cleanup except in wizard_state.save
    (revert to ``except OSError: pass``) -> no WARNING record -> red.
    """
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))

    def exploding_fsync(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", exploding_fsync)

    real_unlink = os.unlink

    def exploding_unlink(p, *args, **kwargs):
        if str(p).endswith(".tmp"):
            raise OSError(13, "Access is denied")
        return real_unlink(p, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", exploding_unlink)

    with caplog.at_level(logging.WARNING, logger="yamabiko.wizard_state"):
        with pytest.raises(OSError, match="No space left"):
            wizard_state.save(
                wizard_state.WizardResult(first_run_complete=True)
            )

    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.wizard_state" and r.levelno == logging.WARNING
    ]
    assert recs, "failed .tmp cleanup left no WARNING evidence"
    assert recs[0].exc_info and "Access is denied" in str(recs[0].exc_info[1]), (
        "the WARNING must carry the UNLINK failure's traceback, not the "
        "original write error's"
    )
