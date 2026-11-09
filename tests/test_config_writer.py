"""Sparse-write behaviour of yamabiko.config_writer.

Tests cover:

- A non-default value lands in user_overrides.yaml at the correct nested path.
- Setting a key back to its shipped default prunes the key, and prunes
  empty dict ancestors.
- Writing to a previously-empty overrides file creates the file.
- When every override has been pruned the file is deleted.
- Unrelated keys in a pre-existing overrides file are preserved across
  writes.
- Atomic write: an os.replace failure leaves the original file intact.

All tests isolate the app-data directory via YAMABIKO_APPDATA.
"""
from __future__ import annotations

import copy
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from yamabiko import config_writer, paths
from yamabiko.config import load_full_config
from yamabiko.config_writer import (
    write_override,
    write_overlay_style,
    write_overlay_position,
    write_audio_and_translation,
)


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path))
    return tmp_path


def _read_overrides(tmp_path: Path) -> dict:
    p = tmp_path / "user_overrides.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize(
    ("updates", "remove_paths"),
    [
        pytest.param({(): "anime"}, (), id="empty-update-path"),
        pytest.param({}, ((),), id="empty-removal-path"),
    ],
)
def test_write_overrides_rejects_empty_paths(appdata, updates, remove_paths):
    """Both public path inputs reject an empty path with ValueError.

    Seeded defect: delete the non-empty-path validation block -> tree
    navigation indexes the empty tuple and raises IndexError instead -> red.
    """
    with pytest.raises(ValueError, match="override paths must be non-empty"):
        config_writer.write_overrides(updates, remove_paths=remove_paths)
    assert not paths.user_overrides_path().exists()


def test_write_overrides_empty_batch_skips_reads_lock_and_flush(
    appdata, monkeypatch,
):
    """A truly empty batch performs no persistence-cycle work.

    Seeded defect: delete the empty-normalized-and-removals early return ->
    defaults/overrides reads, lock acquisition, and flush are recorded -> red.
    """
    from yamabiko import config as config_module

    events = []

    def record_defaults_read(_path):
        events.append("defaults-read")
        return {}

    def record_overrides_read(_path):
        events.append("overrides-read")
        return {}

    def record_lock(_path):
        events.append("lock")
        return nullcontext()

    def record_flush(_path, _overrides):
        events.append("flush")

    monkeypatch.setattr(config_module, "_read_yaml", record_defaults_read)
    monkeypatch.setattr(config_writer, "_read_yaml", record_overrides_read)
    monkeypatch.setattr(config_writer, "_locked_overrides", record_lock)
    monkeypatch.setattr(config_writer, "_flush_tree", record_flush)

    config_writer.write_overrides({})

    assert events == []


def test_write_overrides_removal_only_batch_deletes_last_override_file(
    appdata,
):
    """A batch containing only a removal persists that removal to disk.

    Seeded defect: change the early return to ``if not normalized: return`` ->
    the removal is skipped and the seeded overrides file survives -> red.
    """
    config_writer.write_override(["genre"], "anime")
    assert paths.user_overrides_path().exists(), (
        "precondition: removal-only batch needs a persisted override"
    )

    config_writer.write_overrides({}, remove_paths=(("genre",),))

    assert not paths.user_overrides_path().exists()


def test_write_overrides_persists_multiple_keys_in_one_atomic_replace(
    appdata, monkeypatch,
):
    """One batch publishes every requested value through one replace.

    Seeded defect: move ``_flush_tree`` inside the update loop -> both disk
    values still land, but two atomic writes are recorded -> red.
    """
    writes = []
    real = config_writer._atomic_write

    def recording_write(path, data):
        writes.append(copy.deepcopy(data))
        return real(path, data)

    monkeypatch.setattr(config_writer, "_atomic_write", recording_write)
    config_writer.write_overrides({
        ("audio", "device"): "CABLE Output",
        ("genre",): "anime",
    })

    merged = load_full_config()
    assert merged["audio"]["device"] == "CABLE Output"
    assert merged["genre"] == "anime"
    assert len(writes) == 1


def test_write_overrides_failure_leaves_original_file_byte_identical(
    appdata, monkeypatch,
):
    """A later-key failure cannot publish an earlier key from the batch.

    Seeded defect: move ``_flush_tree`` inside the update loop -> the genre
    update lands before the audio-shaped write fails -> bytes differ -> red.
    """
    config_writer.write_override(("genre",), "drama")
    before = paths.user_overrides_path().read_bytes()
    real = config_writer._atomic_write

    def fail_when_audio_update_is_present(path, data):
        if data.get("audio", {}).get("device") == "CABLE Output":
            raise OSError(28, "disk full")
        return real(path, data)

    monkeypatch.setattr(
        config_writer, "_atomic_write", fail_when_audio_update_is_present,
    )
    with pytest.raises(OSError, match="disk full"):
        config_writer.write_overrides({
            ("genre",): "anime",
            ("audio", "device"): "CABLE Output",
        })

    assert paths.user_overrides_path().read_bytes() == before


def test_write_audio_and_translation_never_persists_only_one_field(
    appdata, monkeypatch,
):
    """A failed Settings pair leaves neither field on disk.

    Seeded defect: restore two sequential ``write_override`` calls -> the
    audio-only first write lands before the endpoint-shaped second write
    fails -> the overrides file remains -> red.
    """
    endpoint = "http://127.0.0.1:9000/v1/chat/completions"
    real = config_writer._atomic_write

    def fail_when_endpoint_update_is_present(path, data):
        if data.get("translation", {}).get("endpoint") == endpoint:
            raise OSError("blocked")
        return real(path, data)

    monkeypatch.setattr(
        config_writer, "_atomic_write", fail_when_endpoint_update_is_present,
    )
    with pytest.raises(OSError, match="blocked"):
        config_writer.write_audio_and_translation(
            audio_device_name="CABLE Output",
            translation_endpoint=endpoint,
        )

    assert not paths.user_overrides_path().exists()


def test_write_override_creates_file_with_minimal_path(appdata):
    write_override(["overlay", "style", "jp_size"], 18)
    data = _read_overrides(appdata)
    assert data == {"overlay": {"style": {"jp_size": 18}}}


def test_write_override_recovers_from_corrupt_overrides_file(appdata):
    """A corrupt user_overrides.yaml must not turn every Settings/wizard
    save into a swallowed no-op. The writer must quarantine the bad file and
    persist the change fresh (the loader already tolerates corrupt overrides).

    Seeded-defect drill: drop the try/except in config_writer._read_yaml ->
    safe_load raises yaml.YAMLError on the corrupt file -> write_override
    raises -> this test errors.
    """
    p = paths.user_overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("overlay: 'unterminated string", encoding="utf-8")  # invalid YAML

    write_override(["overlay", "style", "jp_size"], 18)  # must NOT raise

    assert _read_overrides(appdata) == {"overlay": {"style": {"jp_size": 18}}}, (
        "write was dropped/garbled when the overrides file was corrupt"
    )
    backup = p.with_suffix(p.suffix + ".corrupt.bak")
    assert backup.exists(), "corrupt overrides file was not quarantined"


def test_non_utf8_overrides_file_boots_on_defaults_and_is_quarantined_on_save(
    appdata, caplog,
):
    """Windows PowerShell 5.1 ``Out-File`` writes UTF-16LE with a BOM, so a
    scripted setup can seed user_overrides.yaml in bytes the strict UTF-8
    read cannot decode. The decode error is a ValueError, which neither
    reader's parse-error guard caught: the pipeline crash-looped instead of
    booting on defaults, and every wizard/Settings save raised before the
    writer's quarantine could run. Both readers must treat the file as
    corrupt: the loader ignores it with a warning, and the writer moves it
    aside and persists the change fresh.

    Seeded defect: drop UnicodeDecodeError from either reader's guard ->
    the decode error escapes load_full_config / write_override -> red.
    """
    import codecs

    p = paths.user_overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = codecs.BOM_UTF16_LE + "overlay:\n  style:\n    jp_size: 31\n".encode(
        "utf-16-le",
    )
    p.write_bytes(raw)
    defaults = yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8"),
    )

    with caplog.at_level(logging.WARNING):
        merged = load_full_config()

    assert merged["overlay"]["style"]["jp_size"] == (
        defaults["overlay"]["style"]["jp_size"]
    )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ignoring overrides" in m for m in msgs), msgs

    write_override(["overlay", "style", "jp_size"], 18)

    assert _read_overrides(appdata) == {"overlay": {"style": {"jp_size": 18}}}
    backup = p.with_suffix(p.suffix + ".corrupt.bak")
    assert backup.is_file() and backup.read_bytes() == raw, (
        "the undecodable overrides file was not quarantined intact"
    )


def test_write_override_back_to_default_prunes_key(appdata):
    # Set non-default, verify it lands.
    write_override(["overlay", "style", "jp_size"], 18)
    assert _read_overrides(appdata) == {"overlay": {"style": {"jp_size": 18}}}
    # Set back to shipped default (16) — key should be pruned, and the
    # now-empty parent dicts pruned too, leaving the file deleted.
    write_override(["overlay", "style", "jp_size"], 16)
    assert not paths.user_overrides_path().exists()


def test_write_override_prunes_only_empty_ancestors(appdata):
    write_override(["overlay", "style", "jp_size"], 18)
    write_override(["overlay", "style", "en_size"], 30)
    # Reset just jp_size to default; en_size override should survive.
    write_override(["overlay", "style", "jp_size"], 16)
    data = _read_overrides(appdata)
    assert data == {"overlay": {"style": {"en_size": 30}}}


def test_write_override_on_empty_file_creates_file(appdata):
    assert not paths.user_overrides_path().exists()
    write_override(["genre"], "anime")
    assert paths.user_overrides_path().exists()
    assert _read_overrides(appdata) == {"genre": "anime"}


def test_write_override_delete_when_empty(appdata):
    # Write then unwrite: file should vanish.
    write_override(["genre"], "anime")
    assert paths.user_overrides_path().exists()
    write_override(["genre"], "variety")  # shipped default
    assert not paths.user_overrides_path().exists()


def test_write_overlay_style_skips_unknown_keys(appdata):
    """Unknown style keys (UI-only fields, typos) don't pollute overrides."""
    write_overlay_style({
        "jp_size": 20,
        "nonexistent_ui_field": "ignored",
    })
    data = _read_overrides(appdata)
    assert data == {"overlay": {"style": {"jp_size": 20}}}


def test_max_width_pct_flows_through_sparse_writer(tmp_path, monkeypatch):
    """Hard rule 4 consumer proof: the max_width_pct key survives
    write_overlay_style's shipped-defaults filter, which means it exists
    in default_config.yaml overlay.style.

    Seeded defect (one line): delete ``max_width_pct: 70`` from
    default_config.yaml's overlay.style block -> the writer's
    ``k in style_defaults`` filter drops it -> no items -> the file is
    never written -> read_text raises FileNotFoundError -> red.
    """
    from yamabiko import config_writer, paths
    monkeypatch.setattr(paths, "user_overrides_path", lambda: tmp_path / "user_overrides.yaml")
    config_writer.write_overlay_style({"max_width_pct": 55})
    data = yaml.safe_load((tmp_path / "user_overrides.yaml").read_text(encoding="utf-8"))
    assert data["overlay"]["style"]["max_width_pct"] == 55


def test_write_overlay_position_writes_keys_and_prunes_legacy_geometry(appdata):
    """write_overlay_position persists the non-default anchor keys AND, in
    the SAME locked write, prunes a pre-existing overlay.geometry override.

    Migration completion means the legacy rect must not linger: its shipped
    default is null, so a stale overlay.geometry override left behind would
    keep feeding the migration on every load. This test writes geometry
    first (the pre-migration state), then a position, and asserts geometry
    is gone while all position keys landed.

    Seeded defect (one line): delete the ``_remove_path_and_prune(overrides,
    ["overlay", "geometry"])`` call in write_overlay_position -> the legacy
    geometry override survives -> the ``"geometry" not in overlay``
    assertion goes red.

    Seeded defect: drop left_offset_pct from the key tuple -> the Left
    anchor is never saved -> red.
    """
    # Set up the pre-migration state directly via the low-level writer
    # (write_overlay_geometry was retired with the free-geometry model).
    write_override(["overlay", "geometry"], {"x": 100, "y": 200, "width": 800, "height": 300})
    assert "geometry" in _read_overrides(appdata)["overlay"], (
        "precondition: legacy geometry override must exist before migration"
    )

    write_overlay_position(
        {"screen": "\\\\.\\DISPLAY2", "v_offset_pct": 70.0, "h_offset_pct": 30.0,
         "left_offset_pct": 12.0}
    )
    overlay = _read_overrides(appdata)["overlay"]
    assert "geometry" not in overlay, (
        "legacy overlay.geometry override was not pruned by write_overlay_position"
    )
    assert overlay["position"] == {
        "screen": "\\\\.\\DISPLAY2", "v_offset_pct": 70.0, "h_offset_pct": 30.0,
        "left_offset_pct": 12.0,
    }


def test_write_overlay_position_prunes_default_valued_keys(appdata):
    """Sparse semantics per key: a position key equal to its shipped
    default is pruned, so the overrides file never accumulates redundant
    anchor values (and "reset to defaults" falls out for free).

    Here screen='primary' and h_offset_pct=50.0 ARE the shipped defaults,
    so only the non-default v_offset_pct=60.0 should persist.

    Seeded defect (one line): swap the ``_apply_to_tree`` call for
    ``_set_in_tree`` (unconditional set) -> screen and h_offset_pct persist
    as redundant defaults -> the exact-dict assertion goes red.
    """
    write_overlay_position(
        {"screen": "primary", "v_offset_pct": 60.0, "h_offset_pct": 50.0}
    )
    assert _read_overrides(appdata) == {"overlay": {"position": {"v_offset_pct": 60.0}}}


def test_write_audio_and_translation_skips_empty(appdata):
    write_audio_and_translation(audio_device_name=None, translation_endpoint=None)
    assert not paths.user_overrides_path().exists()
    write_audio_and_translation(audio_device_name="Speakers (WASAPI)", translation_endpoint="")
    data = _read_overrides(appdata)
    # Must write to `audio.device` (the key `default_config.yaml` defines
    # and `main.py` reads), NOT `audio.device_name`. The old test asserted
    # the wrong key and locked in the bug; users' Settings-dialog device
    # picks were silently stripped on the next config load with an
    # "unknown config key audio.device_name" warning.
    assert data == {"audio": {"device": "Speakers (WASAPI)"}}


def test_preserves_unrelated_keys(appdata):
    """A second write must not clobber existing overrides outside the touched path."""
    write_override(["genre"], "anime")
    write_override(["overlay", "style", "jp_size"], 18)
    data = _read_overrides(appdata)
    assert data == {
        "genre": "anime",
        "overlay": {"style": {"jp_size": 18}},
    }


def test_atomic_write_fsyncs_flushed_temp_file_before_replace(appdata, monkeypatch):
    """Durability: os.replace journals the RENAME atomically on NTFS
    but does not flush the file's DATA blocks. Without flush+fsync on the
    temp-file handle, power loss shortly after the replace publishes a
    zero-length/partial user_overrides.yaml, which _read_yaml then
    quarantines -- silently resetting every user setting on next launch.

    Asserts the established fsync-before-replace idiom: at least one
    os.fsync runs BEFORE os.replace, and at fsync time the fd already
    holds data (size > 0).

    Seeded defect: delete `f.flush(); os.fsync(f.fileno())` from
    _atomic_write -> no fsync event -> red.
    Note: dropping ONLY the f.flush() is an equivalent mutation in this
    context — PyYAML's emitter flushes its stream at document end — so
    the size>0 clause guards a future refactor to a non-flushing writer
    (e.g. json.dump), not today's yaml path.
    """
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

    write_override(["genre"], "anime")

    replace_indices = [i for i, (kind, _) in enumerate(events) if kind == "replace"]
    assert replace_indices, "write_override never reached os.replace"
    first_replace = replace_indices[0]
    flushed_fsyncs_before_replace = [
        i for i, (kind, size) in enumerate(events)
        if kind == "fsync" and size > 0 and i < first_replace
    ]
    assert flushed_fsyncs_before_replace, (
        f"no os.fsync of flushed data before os.replace — the write is "
        f"not durable across power loss. Events: {events!r}"
    )


def test_write_audio_and_translation_endpoint_only_writes_endpoint(appdata):
    """5G carry-forward (d): direct writer-level pin for the
    (None, endpoint) combination — previously only the settings-dialog
    test caught a writer-guard inversion. audio=None + endpoint set must
    persist EXACTLY the translation.endpoint override and leave the
    audio block untouched.

    Seeded defect: invert either guard in write_audio_and_translation
    (`if not audio_device_name:` writes audio.device=None -> extra key;
    `if not translation_endpoint:` drops the endpoint -> empty file) ->
    the exact-equality assert goes red either way.
    """
    write_audio_and_translation(
        audio_device_name=None,
        translation_endpoint="http://localhost:9999/v1/chat/completions",
    )
    data = _read_overrides(appdata)
    assert data == {
        "translation": {"endpoint": "http://localhost:9999/v1/chat/completions"}
    }


def test_write_audio_and_translation_default_endpoint_prunes_existing_override(
    appdata,
):
    """Passing the SHIPPED DEFAULT endpoint must reach the sparse
    writer and prune a pre-existing non-default override (this is how the
    Settings dialog clears a saved endpoint when the field is emptied).
    The prior `if translation_endpoint:` guard was fine for the default
    string (it's non-empty/truthy), but the INTENTIONAL-CLEAR signal had
    to be a real value, never ``""`` — `is not None` makes the contract
    explicit: None=untouched, any string (incl. the default)=write+prune.

    Seeded-defect check: change the guard back to
    `if translation_endpoint and translation_endpoint != _DEFAULT:` (i.e.
    skip the default) -> the stale override survives -> the
    `not in` assertion goes red.
    """
    import yaml as _yaml

    # The shipped default endpoint, read from the bundled defaults file.
    shipped = _yaml.safe_load(
        paths.shipped_defaults_path().read_text(encoding="utf-8")
    )
    default_endpoint = shipped["translation"]["endpoint"]

    # Seed a non-default override, then route the shipped default through
    # the writer (the dialog's clear-the-field path).
    write_override(["translation", "endpoint"], "http://stale:7/v1/chat/completions")
    assert _read_overrides(appdata)["translation"]["endpoint"] == (
        "http://stale:7/v1/chat/completions"
    )

    write_audio_and_translation(
        audio_device_name=None,
        translation_endpoint=default_endpoint,
    )
    data = _read_overrides(appdata)
    assert "endpoint" not in data.get("translation", {}), (
        f"writing the shipped default did not prune the stale endpoint "
        f"override (shipped-default write must prune): {data!r}"
    )


def test_write_audio_and_translation_none_endpoint_leaves_override_untouched(
    appdata,
):
    """Counterpart: translation_endpoint=None means 'field
    unchanged' and must NOT touch a saved endpoint override — only an
    explicit value (a new URL, or the shipped default to prune) writes.

    Seeded-defect check: change the guard to `if True:` (write
    unconditionally) -> `write_override(["translation","endpoint"], None)`
    fires, the loader later strips the None -> the saved override is
    clobbered -> the equality assertion goes red.
    """
    write_override(["translation", "endpoint"], "http://keep-me:8/v1/chat/completions")

    write_audio_and_translation(
        audio_device_name="Speakers (loopback)",
        translation_endpoint=None,
    )
    data = _read_overrides(appdata)
    assert data["translation"]["endpoint"] == (
        "http://keep-me:8/v1/chat/completions"
    ), f"None endpoint clobbered a saved override (None must mean 'untouched'): {data!r}"
    assert data["audio"]["device"] == "Speakers (loopback)"


def test_atomic_no_partial_file(appdata, monkeypatch):
    """os.replace failure during an update must leave the prior file intact."""
    # Seed with a successful write so a prior version of the file exists.
    write_override(["genre"], "anime")
    p = paths.user_overrides_path()
    original = p.read_text(encoding="utf-8")

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_override(["overlay", "style", "jp_size"], 18)
    assert p.read_text(encoding="utf-8") == original


def test_os_replace_sharing_violation_twice_then_success_saves(appdata, monkeypatch):
    """An antivirus or indexer briefly holding user_overrides.yaml makes the
    publishing os.replace raise PermissionError on Windows. Two transient
    failures must not drop the save.

    Seeded defect: remove the retry around os.replace in _atomic_write ->
    the first PermissionError propagates and nothing is saved.
    """
    real_replace = os.replace
    attempts: list = []

    def flaky_replace(src, dst):
        attempts.append(dst)
        if len(attempts) <= 2:
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    write_override(["genre"], "anime")

    assert len(attempts) == 3
    assert _read_overrides(appdata) == {"genre": "anime"}


# ---------------------------------------------------------------------------
# Cross-process lock on the user_overrides.yaml read-modify-write
# ---------------------------------------------------------------------------

# One click of Settings Apply/OK writes from TWO processes at once: the
# overlay process persists audio/endpoint/quant keys while the pipeline
# process (receiving the styleChanged WS set_style command) RMWs the
# style keys. An interleaving where one side reads before the other's
# write lands silently reverts a just-saved override (e.g.
# translation.quant_tier -> wrong GGUF quant on next launch).

_HOLD_LOCK_SRC = (
    "import sys, time\n"
    "from filelock import FileLock\n"
    "lock = FileLock(sys.argv[1])\n"
    "lock.acquire()\n"
    "print('held', flush=True)\n"
    "time.sleep(float(sys.argv[2]))\n"
    "lock.release()\n"
)


def _spawn_lock_holder(lock_path: str, hold_s: float) -> subprocess.Popen:
    """Start a REAL peer process holding the overrides lock; returns
    after the child confirms acquisition on stdout."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK_SRC, lock_path, str(hold_s)],
        stdout=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline().strip()
    if line != "held":
        proc.kill()
        raise RuntimeError(f"lock-holder child failed to start: {line!r}")
    return proc


@pytest.mark.real
def test_write_override_blocks_while_peer_process_holds_lock(appdata):
    """write_override must serialize behind the cross-process overrides
    lock: while a REAL peer process holds user_overrides.yaml.lock, the
    write must NOT land; it lands promptly after release.

    Fails red if: write_override takes no lock at all (the key lands
    during the hold window — the exact lost-update race shape), or
    acquires the lock only after writing.
    """
    lock_path = str(paths.user_overrides_path()) + ".lock"
    proc = _spawn_lock_holder(lock_path, hold_s=4.0)
    try:
        t = threading.Thread(
            target=write_override, args=(["translation", "quant_tier"], "small"),
        )
        t.start()
        time.sleep(1.5)
        during = _read_overrides(appdata)
        assert during.get("translation", {}).get("quant_tier") is None, (
            "write_override completed while a peer process held the "
            "overrides lock — the RMW is not serialized (race open)"
        )
        t.join(timeout=15)
        assert not t.is_alive(), (
            "write_override still blocked >15s after the peer released "
            "the lock"
        )
        after = _read_overrides(appdata)
        assert after.get("translation", {}).get("quant_tier") == "small", (
            "write never landed after the peer released the lock"
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.real
def test_write_override_falls_back_unlocked_after_lock_timeout(
    appdata, monkeypatch, caplog,
):
    """A wedged peer holding the lock past the timeout must NOT turn a
    Settings save into a silent no-op: after
    _LOCK_TIMEOUT_S, write_override logs a warning and writes anyway.

    Fails red if: the filelock Timeout propagates out of write_override
    (save raises -> swallowed by UI broad-excepts -> dropped), or no
    lock is attempted at all (no warning is logged while a peer holds
    the lock).
    """
    import yamabiko.config_writer as cw
    monkeypatch.setattr(cw, "_LOCK_TIMEOUT_S", 0.4)

    lock_path = str(paths.user_overrides_path()) + ".lock"
    proc = _spawn_lock_holder(lock_path, hold_s=8.0)
    try:
        with caplog.at_level(logging.WARNING, logger="yamabiko.config_writer"):
            write_override(["translation", "quant_tier"], "small")  # no raise
        data = _read_overrides(appdata)
        assert data.get("translation", {}).get("quant_tier") == "small", (
            "save was dropped when the lock timed out — a wedged peer "
            "must degrade to best-effort unlocked write, never a no-op"
        )
        msgs = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING]
        assert any("lock" in m.lower() for m in msgs), (
            f"no lock-timeout warning logged; either the lock isn't "
            f"being attempted or the fallback is silent. Warnings: {msgs!r}"
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_write_override_propagates_body_timeout_without_generator_error(
    appdata, monkeypatch,
):
    """A filelock.Timeout raised from INSIDE the locked body must
    propagate as-is. In the `try: with lock: yield / except Timeout:
    yield` shape, a body-raised Timeout reaches the acquisition
    fallback's except, which yields a SECOND time — and
    @contextmanager turns that into an opaque RuntimeError
    ("generator didn't stop after throw()"). Latent today (the body
    is _read_yaml + dict ops + _atomic_write, none of which raise
    Timeout) but the shape is one refactor away from real, so the
    except must cover acquisition only.

    Seeded-defect check: revert _locked_overrides to wrapping the
    yield in the try/except Timeout — this test fails with
    RuntimeError instead of the expected Timeout.
    """
    import yamabiko.config_writer as cw
    from filelock import Timeout

    def _boom(path):
        raise Timeout("synthetic-timeout-from-body")
    monkeypatch.setattr(cw, "_read_yaml", _boom)

    with pytest.raises(Timeout):
        write_override(["translation", "quant_tier"], "small")


def test_write_overlay_style_batches_into_one_atomic_write(
    appdata, monkeypatch,
):
    """Cross-process race surface reduction: one Settings Apply used to
    trigger 12 sequential read-modify-write cycles from the pipeline
    process — each a separate race window against the overlay process's
    own writes. The batch must be ONE read + ONE atomic write.

    Fails red if: write_overlay_style loops write_override per key
    (atomic-write count > 1).
    """
    import yamabiko.config_writer as cw
    calls: list = []
    real_write = cw._atomic_write

    def _counting_write(path, data):
        calls.append(path)
        return real_write(path, data)
    monkeypatch.setattr(cw, "_atomic_write", _counting_write)

    write_overlay_style({"jp_size": 18, "en_size": 33, "bg_opacity": 99})

    data = _read_overrides(appdata)
    style = data.get("overlay", {}).get("style", {})
    assert style.get("jp_size") == 18
    assert style.get("en_size") == 33
    assert style.get("bg_opacity") == 99
    assert len(calls) == 1, (
        f"expected ONE atomic write for the style batch; got "
        f"{len(calls)} separate RMW cycles"
    )


# ---------------------------------------------------------------------------
# Evidence-erasing swallows (2026-07-01 logging audit, Task 9): the two
# `except OSError: pass` blocks below must KEEP swallowing (never turn a
# best-effort cleanup/quarantine into a new failure) but now leave a
# WARNING with the traceback in the log.
# ---------------------------------------------------------------------------


def test_atomic_write_cleanup_unlink_failure_logs_warning_and_reraises_original(
    appdata, monkeypatch, caplog,
):
    """When the atomic write fails (os.replace ENOSPC) AND the .tmp cleanup
    unlink also fails, _atomic_write must log a WARNING (with traceback)
    for the failed cleanup and still re-raise the ORIGINAL write error.
    The cleanup's ``except OSError: pass`` previously erased the unlink
    failure, so accumulating .tmp litter next to user_overrides.yaml was
    undiagnosable from field logs.

    Seeded defect: re-silence the cleanup except in
    config_writer._atomic_write (revert to ``except OSError: pass``) ->
    no WARNING record -> red.
    """
    def exploding_replace(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", exploding_replace)

    real_unlink = os.unlink

    def exploding_unlink(p, *args, **kwargs):
        if str(p).endswith(".tmp"):
            raise OSError(13, "Access is denied")
        return real_unlink(p, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", exploding_unlink)

    with caplog.at_level(logging.WARNING, logger="yamabiko.config_writer"):
        with pytest.raises(OSError, match="No space left"):
            write_override(["overlay", "style", "jp_size"], 18)

    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.config_writer"
        and r.levelno == logging.WARNING
        and r.exc_info is not None
    ]
    assert recs, "failed .tmp cleanup left no WARNING evidence"
    assert "Access is denied" in str(recs[0].exc_info[1]), (
        "the WARNING must carry the UNLINK failure's traceback, not the "
        "original write error's"
    )


def test_read_yaml_quarantine_replace_failure_logs_warning_and_write_still_lands(
    appdata, monkeypatch, caplog,
):
    """When the corrupt-overrides quarantine rename fails (file locked by
    AV / permissions), _read_yaml must log a WARNING (with traceback) and
    proceed -- the subsequent atomic write overwrites the corrupt file in
    place, so the save still lands. The ``except OSError: pass`` on the
    quarantine previously erased the failure, hiding WHY no .corrupt.bak
    ever appeared next to a repeatedly-corrupted overrides file.

    Seeded defect: re-silence the quarantine except in
    config_writer._read_yaml (revert to ``except OSError: pass``) -> no
    quarantine WARNING record -> red.
    """
    p = paths.user_overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("overlay: 'unterminated string", encoding="utf-8")

    def exploding_path_replace(self, target):
        raise OSError(13, "Access is denied")

    # Only _read_yaml's quarantine uses Path.replace on this write path;
    # _atomic_write publishes via os.replace, which stays real.
    monkeypatch.setattr(Path, "replace", exploding_path_replace)

    with caplog.at_level(logging.WARNING, logger="yamabiko.config_writer"):
        write_override(["overlay", "style", "jp_size"], 18)  # must NOT raise

    # Swallow semantics KEPT: the save still lands over the corrupt file.
    assert _read_overrides(appdata) == {"overlay": {"style": {"jp_size": 18}}}

    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.config_writer"
        and r.levelno == logging.WARNING
        and r.exc_info is not None
        and "quarantine" in r.getMessage()
    ]
    assert recs, (
        "failed quarantine rename left no WARNING evidence (the "
        "pre-existing corrupt-file warning does not count -- the rename "
        "failure needs its own record with the traceback)"
    )


def test_quarantine_preserves_existing_corrupt_backup_with_numbered_variant(
    appdata,
):
    """Two corruptions in sequence must leave TWO pieces of evidence:
    the second quarantine must not path.replace over the first
    .corrupt.bak (destroying the earlier corrupt content) but pick a
    numbered sibling name instead.

    Seeded defect: revert the backup naming to the fixed .corrupt.bak ->
    path.replace clobbers the first backup -> the first-content assert
    goes red.
    """
    p = paths.user_overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    first_corrupt = "overlay: 'first unterminated string"
    p.write_text(first_corrupt, encoding="utf-8")
    write_override(["overlay", "style", "jp_size"], 18)
    backup = p.with_suffix(p.suffix + ".corrupt.bak")
    assert backup.read_text(encoding="utf-8") == first_corrupt, (
        "precondition: first quarantine must land at the plain .corrupt.bak"
    )

    second_corrupt = "overlay: 'second unterminated string"
    p.write_text(second_corrupt, encoding="utf-8")
    write_override(["overlay", "style", "en_size"], 33)

    assert backup.read_text(encoding="utf-8") == first_corrupt, (
        "second quarantine clobbered the first backup's evidence"
    )
    others = [
        f for f in p.parent.glob(p.name + ".corrupt*") if f != backup
    ]
    assert any(
        f.read_text(encoding="utf-8") == second_corrupt for f in others
    ), (
        f"second corrupt content was not quarantined under a "
        f"non-clobbering name; siblings: {[f.name for f in others]!r}"
    )
    # The save itself still landed over the corrupt file.
    assert _read_overrides(appdata) == {"overlay": {"style": {"en_size": 33}}}


def test_non_mapping_quarantine_replace_failure_logs_warning_and_write_still_lands(
    appdata, monkeypatch, caplog,
):
    """Non-mapping sibling of the corrupt-YAML fallback test above: when
    the quarantine rename of a valid-YAML-but-not-a-mapping overrides
    file fails (file locked by AV / permissions), _read_yaml must log a
    WARNING with the traceback and proceed -- the subsequent atomic
    write overwrites the file in place, so the save still lands.

    Seeded defect: re-silence the non-mapping branch's quarantine except
    in config_writer._read_yaml (revert to ``except OSError: pass``) ->
    no quarantine WARNING record -> red.
    """
    p = paths.user_overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("- oops\n- a-list-not-a-mapping\n", encoding="utf-8")

    def exploding_path_replace(self, target):
        raise OSError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", exploding_path_replace)

    with caplog.at_level(logging.WARNING, logger="yamabiko.config_writer"):
        write_override(["overlay", "style", "jp_size"], 18)  # must NOT raise

    assert _read_overrides(appdata) == {"overlay": {"style": {"jp_size": 18}}}
    recs = [
        r for r in caplog.records
        if r.name == "yamabiko.config_writer"
        and r.levelno == logging.WARNING
        and r.exc_info is not None
        and "quarantine" in r.getMessage()
    ]
    assert recs, (
        "failed non-mapping quarantine rename left no WARNING evidence "
        "with the traceback"
    )


# ---------------------------------------------------------------------------
# write_quant_tier: the coupled quant-tier / ASR-compute-type pair must land
# as ONE atomic read-modify-write. translation.quant_tier and
# asr.faster_whisper.compute_type are a unit (the small tier cuts VRAM and the
# ASR compute type is switched with it); persisting them as two independent
# write_override cycles lets a failure between the writes strand a mismatched
# pair on disk (tier=default paired with compute_type=int8_float16, or vice
# versa) with no retry.
# ---------------------------------------------------------------------------


def test_write_quant_tier_batches_small_pair_into_one_atomic_write(
    appdata, monkeypatch,
):
    """Selecting the 'small' tier persists BOTH coupled keys in exactly ONE
    atomic write. A per-key write_override loop would run two separate
    read-modify-write cycles (atomic-write count > 1), each a fresh
    cross-process race window and a partial-write seam.

    Seeded defect: route _persist through two write_override calls ->
    atomic-write count == 2 -> red.
    """
    import yamabiko.config_writer as cw
    from yamabiko.config_writer import write_quant_tier

    calls: list = []
    real_write = cw._atomic_write

    def _counting_write(path, data):
        calls.append(path)
        return real_write(path, data)

    monkeypatch.setattr(cw, "_atomic_write", _counting_write)

    write_quant_tier("small")

    overrides = _read_overrides(appdata)
    assert overrides == {
        "translation": {"quant_tier": "small"},
        "asr": {"faster_whisper": {"compute_type": "int8_float16"}},
    }, f"coupled pair did not both land: {overrides!r}"
    assert len(calls) == 1, (
        f"expected ONE atomic write for the coupled quant pair; got "
        f"{len(calls)} separate RMW cycles"
    )


def test_write_quant_tier_partial_failure_never_persists_mismatched_pair(
    appdata, monkeypatch,
):
    """Atomicity: if a write fails partway through, the on-disk pair must
    never be internally inconsistent (a small tier alongside the default
    compute_type, or the reverse).

    Move from the (empty) default state to 'small' -- both keys are
    WRITTEN, so the overrides tree stays non-empty through both -- while
    the SECOND atomic write raises. A two-call writer sets quant_tier on
    write #1, then fails on write #2, leaving quant_tier=small paired with
    compute_type=float16 (the shipped default) -- a mismatch. The batched
    writer performs a single atomic write, so the pair moves as a unit and
    the second-write failure never fires.

    Seeded defect: revert write_quant_tier to two write_override calls ->
    the mismatched pair persists -> red.
    """
    import yamabiko.config_writer as cw
    from yamabiko.config_writer import write_quant_tier
    from yamabiko.gpu import tier_overrides

    real_write = cw._atomic_write
    state = {"n": 0}

    def _fail_second_write(path, data):
        state["n"] += 1
        if state["n"] >= 2:
            raise OSError(28, "No space left on device")
        return real_write(path, data)

    monkeypatch.setattr(cw, "_atomic_write", _fail_second_write)

    try:
        write_quant_tier("small")
    except OSError:
        pass  # the propagated failure is fine; the pair consistency is the point

    overrides = _read_overrides(appdata)
    tier = overrides.get("translation", {}).get("quant_tier", "default")
    compute = (
        overrides.get("asr", {})
        .get("faster_whisper", {})
        .get("compute_type", "float16")
    )
    expected = tier_overrides(tier)["asr"]["faster_whisper"]["compute_type"]
    assert compute == expected, (
        f"mismatched pair persisted after a partial write: quant_tier={tier!r} "
        f"paired with compute_type={compute!r} (expected {expected!r})"
    )


def test_non_mapping_overrides_quarantined_not_silently_discarded(appdata, caplog):
    """A top-level list/scalar overrides file is VALID YAML, so it missed
    the parse-error quarantine branch: _read_yaml returned {} silently,
    and the next Settings save atomically rewrote the file with ONLY the
    new key -- every other user setting reset and the evidence destroyed.
    Must quarantine + warn exactly like the corrupt-YAML branch.

    Seeded defect: revert the non-mapping branch to a bare `return {}` ->
    no backup file, no warning -> red.
    """
    import logging

    p = paths.user_overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("- oops\n- a-list-not-a-mapping\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        write_override(["overlay", "style", "jp_size"], 18)

    assert _read_overrides(appdata) == {"overlay": {"style": {"jp_size": 18}}}
    backup = p.with_suffix(p.suffix + ".corrupt.bak")
    assert backup.exists(), "non-mapping overrides file was not quarantined"
    assert "not a mapping" in " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )


def test_atomic_write_emits_lf_only_line_endings(appdata):
    """mkstemp(text=True) + text-mode fdopen (newline=None) triple-
    translated line endings on Windows: user_overrides.yaml was written
    with \\r\\r\\n, parsing fine but rendering double-spaced in editors.
    fsutil.atomic_write_text already pins newline="\\n"; the writer must
    match. Load-bearing on the Windows CI runner (Linux never translates,
    so this can only go red where the bug lives).

    Seeded defect (on Windows): restore text=True / drop newline="\\n" ->
    the byte scan finds \\r -> red.
    """
    write_override(["overlay", "style", "jp_size"], 18)
    raw = paths.user_overrides_path().read_bytes()
    assert b"\r" not in raw, f"expected LF-only overrides file, got {raw!r}"
